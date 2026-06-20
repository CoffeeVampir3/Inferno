from std.memory import Span, UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.helpers import Binding, RankView
from kernels.profiling import Profiler
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
)

from prototypes.sigmoid_router import (
    dispatch_minimax_m3_router, M3RouterCandidate,
    M3_HIDDEN, M3_NUM_EXPERTS, M3_TOP_K,
)


comptime ALIGNMENT = 64
comptime WARMUP = 10
comptime SAMPLES = 100
comptime HIDDEN = M3_HIDDEN
comptime NUM_EXPERTS = M3_NUM_EXPERTS
comptime TOP_K = M3_TOP_K
comptime MAX_SEQ = 4096
comptime MAX_WORKERS = 128
comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
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


def fill_bf16(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = BFloat16(Float32((i % 127) - 63) * 0.01)


def fill_f32(ptr: F32Ptr, count: Int):
    for i in range(count):
        ptr[i] = Float32((i % 23) - 11) * 0.02


def run_all[P: BurstThreadPool, //](
    var pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    if NUM_EXPERTS % tp != 0:
        print(t"degree={tp} does not divide num_experts={NUM_EXPERTS}; skip")
        return
    var epr = NUM_EXPERTS // tp
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var x_ptr = arena_alloc_all[BFloat16](arenas, MAX_SEQ * HIDDEN)
    var gate_ptr = arena_alloc_all[Float32](arenas, epr * HIDDEN)
    var bias_ptr = arena_alloc_all[Float32](arenas, NUM_EXPERTS)
    var cands_ptr = arena_alloc_all[M3RouterCandidate](
        arenas, MAX_WORKERS * MAX_SEQ * TOP_K)
    var route_idx_ptr = arena_alloc_all[Int32](arenas, MAX_SEQ * TOP_K)
    var route_w_ptr = arena_alloc_all[Float32](arenas, MAX_SEQ * TOP_K)

    for r in range(tp):
        fill_bf16(view.bind(x_ptr)[r], MAX_SEQ * HIDDEN)
        fill_f32(view.bind(gate_ptr)[r], epr * HIDDEN)
        var bb = view.bind(bias_ptr)[r]
        for e in range(NUM_EXPERTS):
            bb[e] = Float32((e % 9) - 4) * 0.05
        _ = arenas[r].prefault(0, arenas[r].used())

    var x = view.bind(x_ptr)
    var gate = view.bind(gate_ptr)
    var bias = view.bind(bias_ptr)
    var cands = view.bind(cands_ptr)
    var route_idx = view.bind(route_idx_ptr)
    var route_w = view.bind(route_w_ptr)

    var cap = pools[0].get_capacity()
    print(
        t"router: experts={NUM_EXPERTS} top_k={TOP_K} hidden={HIDDEN} "
        t"experts/rank={epr}")
    print(t"degree={tp} pool_capacity={cap} workers/node")

    var seqs = [128, 256, 512, 1024, 2048, 4096]
    var samples = SampleBuffer(SAMPLES)
    var prof = Profiler[False]()

    print(
        "\n=== Router sweep (compute-bound: the ~768 KB gate shard is "
        "cache-resident across tokens, so GMAC/s, not DRAM BW) ===")
    for si in range(len(seqs)):
        var n = seqs[si]
        if n > MAX_SEQ:
            continue

        for _ in range(WARMUP):
            dispatch_minimax_m3_router(
                x, gate, bias, cands, route_idx, route_w, epr, n, pools, prof)
            keep(route_w[0][0])

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            dispatch_minimax_m3_router(
                x, gate, bias, cands, route_idx, route_w, epr, n, pools, prof)
            var t1 = now_ns()
            samples.push(max_last_ts(pools) - t0, t1 - t0)
        keep(route_w[0][0])

        var ks = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        print_row(String(t"seq={n}"), ks, ws, 0)
        if ws.p50 > 0:
            var gmac_s = (Int64(n) * NUM_EXPERTS * HIDDEN) // ws.p50
            var ns_per_tok = ws.p50 // Int64(n)
            print(
                t"      router: {gmac_s} GMAC/s (all 128 experts) "
                t"| {ns_per_tok} ns/token")


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    var iso = len(topo.isolated_cpus)
    print("MiniMax-M3 sigmoid router benchmark")
    print(t"{tp} NUMA node(s), {iso} isolated cpus")

    comptime ARENA_BYTES = 256 * 1024 * 1024
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
