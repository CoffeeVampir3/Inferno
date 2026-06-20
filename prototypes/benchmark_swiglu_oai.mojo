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

from prototypes.swiglu_oai import dispatch_minimax_m3_swiglu_gate_up


comptime ALIGNMENT = 64
comptime WARMUP = 10
comptime SAMPLES = 100
comptime MAX_ELEMS = 8 * 1024 * 1024
comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]


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


def fill_pattern(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = BFloat16(Float32((i % 251) - 125) * 0.12)


def section_sweep[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    label: String,
    intermediate: Int,
    gate: Binding[BFloat16, o],
    up: Binding[BFloat16, o],
    dst: Binding[BFloat16, o],
):
    print(
        t"\n=== {label} (intermediate={intermediate}); "
        t"compute-bound on exp+divide, so Gelem/s not DRAM BW ===")
    var seqs = [128, 256, 512, 1024, 2048, 4096]
    var samples = SampleBuffer(SAMPLES)
    var prof = Profiler[False]()

    for si in range(len(seqs)):
        var n = seqs[si]
        if n * intermediate > MAX_ELEMS:
            continue

        for _ in range(WARMUP):
            dispatch_minimax_m3_swiglu_gate_up(
                gate, up, dst, intermediate, n, pools, prof)
            keep(dst[0][0])

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            dispatch_minimax_m3_swiglu_gate_up(
                gate, up, dst, intermediate, n, pools, prof)
            var t1 = now_ns()
            samples.push(max_last_ts(pools) - t0, t1 - t0)
        keep(dst[0][0])

        var ks = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        print_row(String(t"seq={n}"), ks, ws, 0)
        if ws.p50 > 0:
            var ge100 = (Int64(n) * Int64(intermediate) * 100) // ws.p50
            var ns_per_tok = ws.p50 // Int64(n)
            print(
                t"      activation: {ge100 // 100}.{ge100 % 100} Gelem/s "
                t"| {ns_per_tok} ns/token")


def run_all[P: BurstThreadPool, //](
    var pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var gate_ptr = arena_alloc_all[BFloat16](arenas, MAX_ELEMS)
    var up_ptr = arena_alloc_all[BFloat16](arenas, MAX_ELEMS)
    var dst_ptr = arena_alloc_all[BFloat16](arenas, MAX_ELEMS)

    for r in range(tp):
        fill_pattern(view.bind(gate_ptr)[r], MAX_ELEMS)
        fill_pattern(view.bind(up_ptr)[r], MAX_ELEMS)
        _ = arenas[r].prefault(0, arenas[r].used())

    var gate = view.bind(gate_ptr)
    var up = view.bind(up_ptr)
    var dst = view.bind(dst_ptr)

    var cap = pools[0].get_capacity()
    print(t"swiglu-oai: degree={tp} pool_capacity={cap} workers/node")

    section_sweep(pools, String("MoE / shared expert"), 3072, gate, up, dst)
    section_sweep(pools, String("Dense MLP"), 12288, gate, up, dst)


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    var iso = len(topo.isolated_cpus)
    print("MiniMax-M3 SwiGLU-OAI activation benchmark")
    print(t"{tp} NUMA node(s), {iso} isolated cpus")

    comptime ARENA_BYTES = 128 * 1024 * 1024
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
