from std.collections import InlineArray
from std.memory import Span, UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.helpers import Binding, RankView, BF16Ptr, F32Ptr, I32Ptr
from kernels.moe_router import SparseRoute
from kernels.profiling import Profiler
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
    DEFAULT_SAMPLES,
)

from prototypes.moe_experts_oai import (
    dispatch_minimax_m3_moe_experts,
    M3_MOE_HIDDEN, M3_MOE_INTERMEDIATE, M3_MOE_GATE_UP_FUSED,
)
from prototypes.moe_experts_oai_fused import dispatch_minimax_m3_moe_experts_fused


comptime HIDDEN = M3_MOE_HIDDEN
comptime INTERMEDIATE = M3_MOE_INTERMEDIATE
comptime GATE_UP_FUSED = M3_MOE_GATE_UP_FUSED
comptime NUM_EXPERTS = 128
comptime TOP_K = 4

comptime ALIGNMENT = 64
comptime WARMUP = 10
comptime SAMPLES = 200
comptime NUM_SEQ_SIZES = 5
comptime MAX_SEQ = 1024
comptime SCRATCH_PER_WORKER = 4 * 2 * 64
comptime MAX_WORKERS = 256
comptime WEIGHT_BYTES_PER_EXPERT = (
    GATE_UP_FUSED * HIDDEN + HIDDEN * INTERMEDIATE) * 2


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


def fill_bf16(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = BFloat16(Float32((i % 253) - 126) * 0.005)


def fill_bf16_all[o: ImmutOrigin](ptrs: Binding[BFloat16, o], count: Int):
    for r in range(ptrs.degree()):
        fill_bf16(ptrs[r], count)


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


def parity_check[o: ImmutOrigin](
    out_ref: Binding[BFloat16, o],
    out_test: Binding[BFloat16, o],
    seq_len: Int,
) -> Float32:
    var worst = Float32(0)
    for r in range(out_ref.degree()):
        var a = out_ref[r]
        var b = out_test[r]
        for i in range(seq_len * HIDDEN):
            var d = abs(a[i].cast[DType.float32]() - b[i].cast[DType.float32]())
            if d > worst:
                worst = d
    return worst


def time_orig[
    P: BurstThreadPool, //, o: ImmutOrigin,
](
    mut pools: List[P],
    mut samples: SampleBuffer,
    seq_len: Int,
    x_normed: Binding[BFloat16, o],
    expert_offset: Binding[Int32, o],
    routes: Binding[SparseRoute, o],
    experts_gate_up: Binding[BFloat16, o],
    experts_down: Binding[BFloat16, o],
    gate_scratch: Binding[Float32, o],
    hidden_bucket: Binding[BFloat16, o],
    moe_accum: Binding[Float32, o],
    moe_out: Binding[BFloat16, o],
    experts_per_rank: Int,
) -> Int64:
    var prof = Profiler[False]()
    for _ in range(WARMUP):
        dispatch_minimax_m3_moe_experts[hidden=HIDDEN, intermediate=INTERMEDIATE](
            x_normed, expert_offset, routes, experts_gate_up, experts_down,
            gate_scratch, hidden_bucket, moe_accum, moe_out,
            experts_per_rank, seq_len, pools, prof)
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        dispatch_minimax_m3_moe_experts[hidden=HIDDEN, intermediate=INTERMEDIATE](
            x_normed, expert_offset, routes, experts_gate_up, experts_down,
            gate_scratch, hidden_bucket, moe_accum, moe_out,
            experts_per_rank, seq_len, pools, prof)
        var t1 = now_ns()
        samples.push(max_last_ts(pools) - t0, t1 - t0)
    keep(moe_out[0][0])
    var ws = compute_stats(samples.wall_ns, samples.n)
    var ks = compute_stats(samples.kernel_ns, samples.n)
    print_row(String(t"seq={seq_len} orig    "), ks, ws, 0)
    return ws.p50


def time_fused[
    P: BurstThreadPool, //, o: ImmutOrigin, NR: Int,
](
    mut pools: List[P],
    mut samples: SampleBuffer,
    seq_len: Int,
    x_normed: Binding[BFloat16, o],
    expert_offset: Binding[Int32, o],
    routes: Binding[SparseRoute, o],
    experts_gate_up: Binding[BFloat16, o],
    experts_down: Binding[BFloat16, o],
    gate_scratch: Binding[Float32, o],
    hidden_bucket: Binding[BFloat16, o],
    moe_accum: Binding[Float32, o],
    moe_out: Binding[BFloat16, o],
    experts_per_rank: Int,
) -> Int64:
    var prof = Profiler[False]()
    for _ in range(WARMUP):
        dispatch_minimax_m3_moe_experts_fused[
            hidden=HIDDEN, intermediate=INTERMEDIATE, NR=NR,
        ](
            x_normed, expert_offset, routes, experts_gate_up, experts_down,
            gate_scratch, hidden_bucket, moe_accum, moe_out,
            experts_per_rank, seq_len, pools, prof)
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        dispatch_minimax_m3_moe_experts_fused[
            hidden=HIDDEN, intermediate=INTERMEDIATE, NR=NR,
        ](
            x_normed, expert_offset, routes, experts_gate_up, experts_down,
            gate_scratch, hidden_bucket, moe_accum, moe_out,
            experts_per_rank, seq_len, pools, prof)
        var t1 = now_ns()
        samples.push(max_last_ts(pools) - t0, t1 - t0)
    keep(moe_out[0][0])
    var ws = compute_stats(samples.wall_ns, samples.n)
    var ks = compute_stats(samples.kernel_ns, samples.n)
    print_row(String(t"seq={seq_len} fusedNR{NR}"), ks, ws, 0)
    return ws.p50


def report_speedup(label: String, base: Int64, test: Int64):
    if test > 0 and base > 0:
        var pct = (base * 100) // test
        var frac = pct % 100
        var pad = "0" if frac < 10 else ""
        print(t"      {label}: {pct // 100}.{pad}{frac}x vs orig")


def run_all[P: BurstThreadPool, //](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    if NUM_EXPERTS % tp != 0:
        print(t"degree={tp} does not divide num_experts={NUM_EXPERTS}; skip")
        return
    var epr = NUM_EXPERTS // tp

    var sizes = InlineArray[Int, NUM_SEQ_SIZES](fill=0)
    sizes[0] = 1
    sizes[1] = 16
    sizes[2] = 64
    sizes[3] = 256
    sizes[4] = MAX_SEQ

    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var x_ptr = arena_alloc_all[BFloat16](arenas, MAX_SEQ * HIDDEN)
    var ofs_ptr = arena_alloc_all[Int32](arenas, epr + 1)
    var routes_ptr = arena_alloc_all[SparseRoute](arenas, MAX_SEQ * TOP_K)
    var gate_up_ptr = arena_alloc_all[BFloat16](
        arenas, epr * GATE_UP_FUSED * HIDDEN)
    var down_ptr = arena_alloc_all[BFloat16](
        arenas, epr * HIDDEN * INTERMEDIATE)
    var scratch_ptr = arena_alloc_all[Float32](
        arenas, MAX_WORKERS * SCRATCH_PER_WORKER)
    var bucket_ptr = arena_alloc_all[BFloat16](
        arenas, MAX_SEQ * TOP_K * INTERMEDIATE)
    var accum_ptr = arena_alloc_all[Float32](arenas, MAX_SEQ * HIDDEN)
    var out_ptr = arena_alloc_all[BFloat16](arenas, MAX_SEQ * HIDDEN)
    var out2_ptr = arena_alloc_all[BFloat16](arenas, MAX_SEQ * HIDDEN)

    var x = view.bind(x_ptr)
    var ofs = view.bind(ofs_ptr)
    var routes = view.bind(routes_ptr)
    var gate_up = view.bind(gate_up_ptr)
    var down = view.bind(down_ptr)
    var scratch = view.bind(scratch_ptr)
    var bucket = view.bind(bucket_ptr)
    var accum = view.bind(accum_ptr)
    var out = view.bind(out_ptr)
    var out2 = view.bind(out2_ptr)

    fill_bf16_all(x, MAX_SEQ * HIDDEN)
    fill_bf16_all(gate_up, epr * GATE_UP_FUSED * HIDDEN)
    fill_bf16_all(down, epr * HIDDEN * INTERMEDIATE)
    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    var cap = pools[0].get_capacity()
    print(
        t"hidden={HIDDEN} intermediate={INTERMEDIATE} experts={NUM_EXPERTS} "
        t"top_k={TOP_K} experts/rank={epr}")
    print(t"degree={tp} pool_capacity={cap} workers/node")

    var samples = SampleBuffer(SAMPLES)

    # Parity once at the largest size: orig -> out, fused(NR=2) -> out2.
    build_uniform_routing(MAX_SEQ, ofs, routes, epr)
    var pprof = Profiler[False]()
    dispatch_minimax_m3_moe_experts[hidden=HIDDEN, intermediate=INTERMEDIATE](
        x, ofs, routes, gate_up, down, scratch, bucket, accum, out,
        epr, MAX_SEQ, pools, pprof)
    dispatch_minimax_m3_moe_experts_fused[
        hidden=HIDDEN, intermediate=INTERMEDIATE, NR=2,
    ](
        x, ofs, routes, gate_up, down, scratch, bucket, accum, out2,
        epr, MAX_SEQ, pools, pprof)
    var worst = parity_check(out, out2, MAX_SEQ)
    print(t"\nparity (orig vs fusedNR2) max abs diff over seq={MAX_SEQ}: {worst}")

    print(
        "\n=== M3 routed-expert path: baseline vs fused gate/up panel ===\n"
        "    fusedNR1 = gate/up load fusion only; fusedNR2/4 add column "
        "register-tiling")
    for s in range(NUM_SEQ_SIZES):
        var seq = sizes[s]
        build_uniform_routing(seq, ofs, routes, epr)
        var b = time_orig(
            pools, samples, seq, x, ofs, routes, gate_up, down,
            scratch, bucket, accum, out, epr)
        var f1 = time_fused[NR=1](
            pools, samples, seq, x, ofs, routes, gate_up, down,
            scratch, bucket, accum, out2, epr)
        var f2 = time_fused[NR=2](
            pools, samples, seq, x, ofs, routes, gate_up, down,
            scratch, bucket, accum, out2, epr)
        var f3 = time_fused[NR=3](
            pools, samples, seq, x, ofs, routes, gate_up, down,
            scratch, bucket, accum, out2, epr)
        report_speedup("fusedNR1", b, f1)
        report_speedup("fusedNR2", b, f2)
        report_speedup("fusedNR3", b, f3)


def arena_bytes_for(tp: Int) -> Int:
    var epr = NUM_EXPERTS // tp
    return epr * WEIGHT_BYTES_PER_EXPERT + 256 * 1024 * 1024


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    print("MiniMax-M3 MoE fused gate/up panel benchmark (baseline vs fused)")
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
