from std.collections import InlineArray
from std.memory import Span, UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.helpers import Binding, RankView, BF16Ptr
from kernels.moe_router import SparseRoute
from kernels.profiling import Profiler
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
)

from butterquant.types import I8Ptr, F32Ptr
from butterquant.vnni import pack_and_colsum_vnni
from butterquant.weight import ButterquantWeight, ButterquantActivation
from butterquant_kernels.moe import dispatch_bq_phase1_gate_up
from quant.recipe import (
    QuantRecipe, PerRowQuant, NoGamma, SingleSided, PerRowCs, VnniPacked,
)

from prototypes.panel_i8 import dispatch_bq_phase1_gate_up_fused


comptime QUANT: QuantRecipe = PerRowQuant(
    128, NoGamma(), SingleSided(), PerRowCs(), VnniPacked())


def bind_weight[o: ImmutOrigin](
    data: Binding[Int8, o], scale: Binding[Float32, o],
    colsum: Binding[Float32, o],
) -> ButterquantWeight[QUANT, o]:
    return ButterquantWeight[QUANT, o](data, scale, colsum)


def bind_act[o: ImmutOrigin](
    data: Binding[Int8, o], scale: Binding[Float32, o],
) -> ButterquantActivation[o]:
    return ButterquantActivation[o](data, scale)

comptime HIDDEN = 6144
comptime INTERMEDIATE = 3072
comptime GATE_UP = 2 * INTERMEDIATE
comptime NUM_EXPERTS = 128
comptime TOP_K = 4

comptime ALIGNMENT = 64
comptime WARMUP = 10
comptime SAMPLES = 200
comptime NUM_SEQ_SIZES = 4
comptime MAX_SEQ = 1024
comptime PACK_SCRATCH = 64 * HIDDEN


def arena_alloc_all[T: AnyType](
    mut arenas: List[NumaArena[alignment=ALIGNMENT]], count: Int,
) -> UnsafePointer[T, MutAnyOrigin]:
    var first = UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
    for r in range(len(arenas)):
        var p = arenas[r].alloc[T](count)
        if not p:
            print("arena alloc failed for", count, "elements")
            return UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
        if r == 0:
            first = p.value()
    return first


def arena_bases(
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
) -> List[Int]:
    var bases = List[Int](capacity=len(arenas))
    for r in range(len(arenas)):
        bases.append(Int(arenas[r].base.value()))
    return bases^


def build_uniform_routing[o: ImmutOrigin](
    seq_len: Int,
    expert_offset: Binding[Int32, o],
    routes: Binding[SparseRoute, o],
    experts_per_rank: Int,
):
    var tp = expert_offset.degree()
    var w = Float32(1.0) / Float32(TOP_K)
    for r in range(tp):
        var first = r * experts_per_rank
        var last = first + experts_per_rank
        var ofs_r = expert_offset[r]
        var routes_r = routes[r]

        var counts = InlineArray[Int, NUM_EXPERTS](fill=0)
        for tok in range(seq_len):
            for k in range(TOP_K):
                var e = (tok * TOP_K + k) % NUM_EXPERTS
                if e >= first and e < last:
                    counts[e - first] += 1

        var running = Int32(0)
        var write_ofs = InlineArray[Int, NUM_EXPERTS](fill=0)
        for e in range(experts_per_rank):
            ofs_r[e] = running
            write_ofs[e] = Int(running)
            running += Int32(counts[e])
        ofs_r[experts_per_rank] = running

        for tok in range(seq_len):
            for k in range(TOP_K):
                var e = (tok * TOP_K + k) % NUM_EXPERTS
                if e >= first and e < last:
                    var local = e - first
                    var pos = write_ofs[local]
                    routes_r[pos] = SparseRoute(Int32(tok), w)
                    write_ofs[local] = pos + 1


def fill_raw_i8(ptr: I8Ptr, count: Int, salt: Int):
    for i in range(count):
        ptr[i] = Int8(((i + salt) % 17) - 8)


def fill_f32(ptr: F32Ptr, count: Int, base: Float32):
    for i in range(count):
        ptr[i] = base + Float32(i % 7) * 0.001


def parity_check[o: ImmutOrigin](
    out_ref: Binding[BFloat16, o],
    out_test: Binding[BFloat16, o],
    n: Int,
) -> Float32:
    var worst = Float32(0)
    for r in range(out_ref.degree()):
        var a = out_ref[r]
        var b = out_test[r]
        for i in range(n):
            var d = abs(a[i].cast[DType.float32]() - b[i].cast[DType.float32]())
            if d > worst:
                worst = d
    return worst


def time_orig[
    P: BurstThreadPool, //, o: ImmutOrigin,
](
    mut pools: List[P], mut samples: SampleBuffer, seq: Int,
    act: ButterquantActivation[o],
    ofs: Binding[Int32, o], routes: Binding[SparseRoute, o],
    weight: ButterquantWeight[QUANT, o],
    bucket: Binding[BFloat16, o], epr: Int,
) -> Int64:
    var prof = Profiler[False]()
    for _ in range(WARMUP):
        dispatch_bq_phase1_gate_up[hidden=HIDDEN, gate_up=GATE_UP, inter=INTERMEDIATE](
            act, ofs, routes, weight, bucket, epr, pools, prof)
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        dispatch_bq_phase1_gate_up[hidden=HIDDEN, gate_up=GATE_UP, inter=INTERMEDIATE](
            act, ofs, routes, weight, bucket, epr, pools, prof)
        var t1 = now_ns()
        samples.push(max_last_ts(pools) - t0, t1 - t0)
    keep(bucket[0][0])
    var ks = compute_stats(samples.kernel_ns, samples.n)
    var ws = compute_stats(samples.wall_ns, samples.n)
    print_row(String(t"seq={seq} bq      "), ks, ws, 0)
    return ws.p50


def time_fused[
    P: BurstThreadPool, //, o: ImmutOrigin,
](
    mut pools: List[P], mut samples: SampleBuffer, seq: Int,
    act: ButterquantActivation[o],
    ofs: Binding[Int32, o], routes: Binding[SparseRoute, o],
    weight: ButterquantWeight[QUANT, o],
    bucket: Binding[BFloat16, o], epr: Int,
) -> Int64:
    var prof = Profiler[False]()
    for _ in range(WARMUP):
        dispatch_bq_phase1_gate_up_fused[
            hidden=HIDDEN, gate_up=GATE_UP, inter=INTERMEDIATE,
        ](act, ofs, routes, weight, bucket, epr, pools, prof)
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        dispatch_bq_phase1_gate_up_fused[
            hidden=HIDDEN, gate_up=GATE_UP, inter=INTERMEDIATE,
        ](act, ofs, routes, weight, bucket, epr, pools, prof)
        var t1 = now_ns()
        samples.push(max_last_ts(pools) - t0, t1 - t0)
    keep(bucket[0][0])
    var ks = compute_stats(samples.kernel_ns, samples.n)
    var ws = compute_stats(samples.wall_ns, samples.n)
    print_row(String(t"seq={seq} bqfused"), ks, ws, 0)
    return ws.p50


def report_speedup(base: Int64, test: Int64):
    if test > 0 and base > 0:
        var pct = (base * 100) // test
        var frac = pct % 100
        var pad = "0" if frac < 10 else ""
        print(t"      fused: {pct // 100}.{pad}{frac}x vs bq")


def run_all[P: BurstThreadPool, //](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    if NUM_EXPERTS % tp != 0:
        print(t"degree={tp} does not divide num_experts={NUM_EXPERTS}; skip")
        return
    var epr = NUM_EXPERTS // tp

    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var packed_ptr = arena_alloc_all[Int8](arenas, epr * GATE_UP * HIDDEN)
    var scale_ptr = arena_alloc_all[Float32](arenas, epr * GATE_UP)
    var colsum_ptr = arena_alloc_all[Float32](arenas, epr * GATE_UP)
    var x_ptr = arena_alloc_all[Int8](arenas, MAX_SEQ * HIDDEN)
    var xsa_ptr = arena_alloc_all[Float32](arenas, MAX_SEQ)
    var ofs_ptr = arena_alloc_all[Int32](arenas, epr + 1)
    var routes_ptr = arena_alloc_all[SparseRoute](arenas, MAX_SEQ * TOP_K)
    var bucket_ref_ptr = arena_alloc_all[BFloat16](
        arenas, MAX_SEQ * TOP_K * INTERMEDIATE)
    var bucket_test_ptr = arena_alloc_all[BFloat16](
        arenas, MAX_SEQ * TOP_K * INTERMEDIATE)
    var raw_ptr = arena_alloc_all[Int8](arenas, GATE_UP * HIDDEN)
    var scratch_ptr = arena_alloc_all[Int8](arenas, PACK_SCRATCH)

    var packed = view.bind(packed_ptr)
    var scale = view.bind(scale_ptr)
    var colsum = view.bind(colsum_ptr)
    var x_i8 = view.bind(x_ptr)
    var x_sa = view.bind(xsa_ptr)
    var ofs = view.bind(ofs_ptr)
    var routes = view.bind(routes_ptr)
    var bucket_ref = view.bind(bucket_ref_ptr)
    var bucket_test = view.bind(bucket_test_ptr)

    var raw_b = view.bind(raw_ptr)
    var scratch_b = view.bind(scratch_ptr)
    for r in range(tp):
        fill_raw_i8(x_i8[r], MAX_SEQ * HIDDEN, 3)
        fill_f32(x_sa[r], MAX_SEQ, 0.02)
        fill_f32(scale[r], epr * GATE_UP, 0.01)
        for e in range(epr):
            fill_raw_i8(raw_b[r], GATE_UP * HIDDEN, e * 5 + 1)
            pack_and_colsum_vnni(
                raw_b[r].bitcast[UInt8](),
                (packed[r] + e * GATE_UP * HIDDEN).bitcast[UInt8](),
                scratch_b[r].bitcast[UInt8](),
                GATE_UP, HIDDEN, HIDDEN,
                colsum[r] + e * GATE_UP,
                True)
    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    var weight = bind_weight(packed, scale, colsum)
    var act = bind_act(x_i8, x_sa)

    var cap = pools[0].get_capacity()
    print(
        t"hidden={HIDDEN} inter={INTERMEDIATE} experts={NUM_EXPERTS} "
        t"top_k={TOP_K} experts/rank={epr} degree={tp} cap={cap}")

    var sizes = InlineArray[Int, NUM_SEQ_SIZES](uninitialized=True)
    sizes[0] = 1; sizes[1] = 64; sizes[2] = 256; sizes[3] = MAX_SEQ

    build_uniform_routing(MAX_SEQ, ofs, routes, epr)
    var pprof = Profiler[False]()
    dispatch_bq_phase1_gate_up[hidden=HIDDEN, gate_up=GATE_UP, inter=INTERMEDIATE](
        act, ofs, routes, weight, bucket_ref, epr, pools, pprof)
    dispatch_bq_phase1_gate_up_fused[
        hidden=HIDDEN, gate_up=GATE_UP, inter=INTERMEDIATE,
    ](act, ofs, routes, weight, bucket_test, epr, pools, pprof)
    var worst = parity_check(
        bucket_ref, bucket_test, MAX_SEQ * TOP_K * INTERMEDIATE)
    print(t"\nparity (bq vs bqfused) max abs diff: {worst}\n")

    print("=== int8 MoE phase1 gate/up: baseline vs fused (load-fusion) ===")
    var samples = SampleBuffer(SAMPLES)
    for s in range(NUM_SEQ_SIZES):
        var seq = sizes[s]
        build_uniform_routing(seq, ofs, routes, epr)
        var b = time_orig(
            pools, samples, seq, act, ofs, routes, weight, bucket_ref, epr)
        var f = time_fused(
            pools, samples, seq, act, ofs, routes, weight, bucket_test, epr)
        report_speedup(b, f)


def arena_bytes_for(tp: Int) -> Int:
    var epr = NUM_EXPERTS // tp
    return (epr * GATE_UP * HIDDEN + GATE_UP * HIDDEN
            + PACK_SCRATCH + 384 * 1024 * 1024)


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    print("int8 MoE phase1 fused gate/up benchmark (baseline vs fused)")
    var iso = len(topo.isolated_cpus)
    print(t"{tp} NUMA node(s), {iso} isolated cpus")

    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    var bytes = arena_bytes_for(tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], bytes))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_bench[P: BurstThreadPool, //](var selected_pools: List[P]):
        run_all(selected_pools, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_bench,
    ](topo, "mode: isolated", "mode: spin-backoff")
