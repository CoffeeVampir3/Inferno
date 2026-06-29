from std.memory import Span, UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.helpers import Binding, RankView
from kernels.moe_router import SparseRoute
from kernels.profiling import Profiler
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
)

from butterquant.weight import ButterquantWeight, ButterquantActivation
from butterquant.amx_tiles import prime_amx_environment
from quant.recipe import (
    QuantRecipe, PerRowQuant, NoGamma, SingleSided, PerRowCs, VnniPacked,
)

from prototypes.bq_moe_phase1 import (
    dispatch_bq_m3_phase1_gate_up, dispatch_bq_m3_phase1_gate_up_amx,
)


comptime ALIGNMENT = 64
comptime WARMUP = 5
comptime SAMPLES = 50

comptime HIDDEN = 6144
comptime INTER = 3072
comptime GATE_UP = 2 * INTER
comptime EXPERTS = 4
comptime TOP_K = 4
comptime MAX_SEQ = 1024
comptime MAX_RECORDS = MAX_SEQ * TOP_K
comptime X_SA = Float32(2.0)
comptime W_SCALE = Float32(0.05)

comptime Recipe: QuantRecipe = PerRowQuant(
    128, NoGamma(), SingleSided(), PerRowCs(), VnniPacked())

comptime I8Ptr = UnsafePointer[Int8, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]


def arena_bases(
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
) -> List[Int]:
    var bases = List[Int](capacity=len(arenas))
    for r in range(len(arenas)):
        bases.append(Int(arenas[r].base.value()))
    return bases^


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


def fill_i8(ptr: I8Ptr, count: Int):
    for i in range(count):
        ptr[i] = Int8((i % 7) - 3)


def fill_f32(ptr: F32Ptr, count: Int, val: Float32):
    for i in range(count):
        ptr[i] = val


def fill_routing(
    eoff: Binding[Int32, _], routes: Binding[SparseRoute, _],
    n_records: Int, seq_len: Int, tp: Int,
):
    var per = n_records // EXPERTS
    for r in range(tp):
        var eo = eoff[r]
        for e in range(EXPERTS):
            eo[e] = Int32(min(e * per, n_records))
        eo[EXPERTS] = Int32(n_records)
        var rb = routes[r]
        for rec in range(n_records):
            rb[rec] = SparseRoute(Int32(rec % seq_len), Float32(1.0))


def run_phase1[P: BurstThreadPool, o: ImmutOrigin, N: Int, //](
    mut pools: List[P],
    act: ButterquantActivation[o],
    eoff: Binding[Int32, o],
    routes: Binding[SparseRoute, o],
    weight: ButterquantWeight[Recipe, o],
    bucket: Binding[BFloat16, o],
    mut prof: Profiler[False, N],
):
    dispatch_bq_m3_phase1_gate_up[
        hidden=HIDDEN, gate_up=GATE_UP, inter=INTER,
    ](act, eoff, routes, weight, bucket, EXPERTS, pools, prof)


def run_phase1_amx[P: BurstThreadPool, o: ImmutOrigin, N: Int, //](
    mut pools: List[P],
    act: ButterquantActivation[o],
    eoff: Binding[Int32, o],
    routes: Binding[SparseRoute, o],
    weight: ButterquantWeight[Recipe, o],
    bucket: Binding[BFloat16, o],
    act_routed: Binding[Int8, o],
    sa_routed: Binding[Float32, o],
    mut prof: Profiler[False, N],
):
    dispatch_bq_m3_phase1_gate_up_amx[
        hidden=HIDDEN, gate_up=GATE_UP, inter=INTER,
    ](act, eoff, routes, weight, bucket, act_routed, sa_routed,
      EXPERTS, pools, prof)


def run_all[P: BurstThreadPool, //](
    var pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var x_i8_ptr = arena_alloc_all[Int8](arenas, MAX_SEQ * HIDDEN)
    var x_sa_ptr = arena_alloc_all[Float32](arenas, MAX_SEQ)
    var w_ptr = arena_alloc_all[Int8](arenas, EXPERTS * GATE_UP * HIDDEN)
    var wsc_ptr = arena_alloc_all[Float32](arenas, EXPERTS * GATE_UP)
    var cs_ptr = arena_alloc_all[Float32](arenas, EXPERTS * GATE_UP)
    var eoff_ptr = arena_alloc_all[Int32](arenas, EXPERTS + 1)
    var routes_ptr = arena_alloc_all[SparseRoute](arenas, MAX_RECORDS)
    var bucket_ptr = arena_alloc_all[BFloat16](arenas, MAX_RECORDS * INTER)
    var act_routed_ptr = arena_alloc_all[Int8](arenas, MAX_RECORDS * HIDDEN)
    var sa_routed_ptr = arena_alloc_all[Float32](arenas, MAX_RECORDS)

    for r in range(tp):
        fill_i8(view.bind(x_i8_ptr)[r], MAX_SEQ * HIDDEN)
        fill_f32(view.bind(x_sa_ptr)[r], MAX_SEQ, X_SA)
        fill_i8(view.bind(w_ptr)[r], EXPERTS * GATE_UP * HIDDEN)
        fill_f32(view.bind(wsc_ptr)[r], EXPERTS * GATE_UP, W_SCALE)
        fill_f32(view.bind(cs_ptr)[r], EXPERTS * GATE_UP, 0.0)
        _ = arenas[r].prefault(0, arenas[r].used())

    var act = ButterquantActivation(view.bind(x_i8_ptr), view.bind(x_sa_ptr))
    var weight = ButterquantWeight[Recipe](
        view.bind(w_ptr), view.bind(wsc_ptr), view.bind(cs_ptr))
    var eoff = view.bind(eoff_ptr)
    var routes = view.bind(routes_ptr)
    var bucket = view.bind(bucket_ptr)
    var act_routed = view.bind(act_routed_ptr)
    var sa_routed = view.bind(sa_routed_ptr)

    prime_amx_environment(pools)

    var cap = pools[0].get_capacity()
    print(
        t"bq moe phase1 (swiglu_oai): hidden={HIDDEN} inter={INTER} "
        t"experts={EXPERTS} top_k={TOP_K}")
    print(t"degree={tp} pool_capacity={cap} workers/node")
    print("\n=== Phase1 gate+up sweep (records = seq * top_k) ===")

    var seqs = [16, 64, 256, 1024]
    var s = SampleBuffer(SAMPLES)
    var prof = Profiler[False]()

    for si in range(len(seqs)):
        var seq = seqs[si]
        if seq > MAX_SEQ:
            continue
        var n_records = seq * TOP_K
        fill_routing(eoff, routes, n_records, seq, tp)

        var weight_bytes = EXPERTS * GATE_UP * HIDDEN
        print(t"seq={seq} records={n_records}")

        for _ in range(WARMUP):
            run_phase1(pools, act, eoff, routes, weight, bucket, prof)
            keep(bucket[0][0])
        s.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            run_phase1(pools, act, eoff, routes, weight, bucket, prof)
            var t1 = now_ns()
            s.push(max_last_ts(pools) - t0, t1 - t0)
        keep(bucket[0][0])
        print_row("  vnni  ",
            compute_stats(s.kernel_ns, s.n),
            compute_stats(s.wall_ns, s.n), weight_bytes)

        for _ in range(WARMUP):
            run_phase1_amx(pools, act, eoff, routes, weight, bucket,
                           act_routed, sa_routed, prof)
            keep(bucket[0][0])
        s.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            run_phase1_amx(pools, act, eoff, routes, weight, bucket,
                           act_routed, sa_routed, prof)
            var t1 = now_ns()
            s.push(max_last_ts(pools) - t0, t1 - t0)
        keep(bucket[0][0])
        print_row("  amx   ",
            compute_stats(s.kernel_ns, s.n),
            compute_stats(s.wall_ns, s.n), weight_bytes)


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    var iso = len(topo.isolated_cpus)
    print("MiniMax-M3 ButterQuant MoE phase1 (SwiGLU-OAI) benchmark")
    print(t"{tp} NUMA node(s), {iso} isolated cpus")

    comptime ARENA_BYTES = 1024 * 1024 * 1024
    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_bench[P: BurstThreadPool, //](var selected_pools: List[P]):
        run_all(selected_pools^, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_bench,
    ](topo, "mode: isolated", "mode: spin-backoff")
