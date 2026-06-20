from std.collections import InlineArray
from std.memory import Span, UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.helpers import Binding, RankView, BF16Ptr
from kernels.gemm import dispatch_gemm
from kernels.profiling import Profiler
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
    DEFAULT_SAMPLES,
)

from prototypes.gemm_panel import dispatch_gemm_panel


comptime ALIGNMENT = 64
comptime WARMUP = 10
comptime SAMPLES = 200

comptime HIDDEN = 6144
comptime ROWS = 4096
comptime MAX_M = 1024
comptime NUM_SIZES = 5


def arena_alloc_all[dtype: DType](
    mut arenas: List[NumaArena[alignment=ALIGNMENT]], count: Int,
) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    var first = UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()
    for r in range(len(arenas)):
        var ptr = arenas[r].alloc[Scalar[dtype]](count)
        if not ptr:
            print("arena alloc failed for", count, "elements")
            return UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()
        if r == 0:
            first = ptr.value()
    return first


def arena_bases(
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
) -> List[Int]:
    var bases = List[Int](capacity=len(arenas))
    for r in range(len(arenas)):
        bases.append(Int(arenas[r].base.value()))
    return bases^


def fill_pattern(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = BFloat16(Float32((i % 253) - 126) * 0.005)


def fill_pattern_all[o: ImmutOrigin](ptrs: Binding[BFloat16, o], count: Int):
    for r in range(ptrs.degree()):
        fill_pattern(ptrs[r], count)


def parity_check[o: ImmutOrigin](
    out_ref: Binding[BFloat16, o],
    out_test: Binding[BFloat16, o],
    m: Int,
) -> Float32:
    var worst = Float32(0)
    for r in range(out_ref.degree()):
        var a = out_ref[r]
        var b = out_test[r]
        for i in range(m * ROWS):
            var d = abs(a[i].cast[DType.float32]() - b[i].cast[DType.float32]())
            if d > worst:
                worst = d
    return worst


def time_baseline[
    P: BurstThreadPool, //, o: ImmutOrigin,
](
    mut pools: List[P], mut samples: SampleBuffer, m: Int,
    xs: Binding[BFloat16, o], ws: Binding[BFloat16, o],
    outs: Binding[BFloat16, o],
) -> Int64:
    var prof = Profiler[False]()
    for _ in range(WARMUP):
        dispatch_gemm[cols=HIDDEN, MR=4](xs, ws, outs, ROWS, m, pools, prof)
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        dispatch_gemm[cols=HIDDEN, MR=4](xs, ws, outs, ROWS, m, pools, prof)
        var t1 = now_ns()
        samples.push(max_last_ts(pools) - t0, t1 - t0)
    keep(outs[0][0])
    var ks = compute_stats(samples.kernel_ns, samples.n)
    var ws_ = compute_stats(samples.wall_ns, samples.n)
    print_row(String(t"M={m} gemm   "), ks, ws_, 0)
    return ws_.p50


def time_panel[
    P: BurstThreadPool, //, o: ImmutOrigin, NC: Int,
](
    mut pools: List[P], mut samples: SampleBuffer, m: Int,
    xs: Binding[BFloat16, o], ws: Binding[BFloat16, o],
    outs: Binding[BFloat16, o],
) -> Int64:
    var prof = Profiler[False]()
    for _ in range(WARMUP):
        dispatch_gemm_panel[cols=HIDDEN, MR=4, NC=NC](
            xs, ws, outs, ROWS, m, pools, prof)
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        dispatch_gemm_panel[cols=HIDDEN, MR=4, NC=NC](
            xs, ws, outs, ROWS, m, pools, prof)
        var t1 = now_ns()
        samples.push(max_last_ts(pools) - t0, t1 - t0)
    keep(outs[0][0])
    var ks = compute_stats(samples.kernel_ns, samples.n)
    var ws_ = compute_stats(samples.wall_ns, samples.n)
    print_row(String(t"M={m} panelNC{NC}"), ks, ws_, 0)
    return ws_.p50


def report_speedup(label: String, base: Int64, test: Int64):
    if test > 0 and base > 0:
        var pct = (base * 100) // test
        var frac = pct % 100
        var pad = "0" if frac < 10 else ""
        print(t"      {label}: {pct // 100}.{pad}{frac}x vs gemm")


def run_all[P: BurstThreadPool, //](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var x = arena_alloc_all[DType.bfloat16](arenas, MAX_M * HIDDEN)
    var w = arena_alloc_all[DType.bfloat16](arenas, ROWS * HIDDEN)
    var o_ref = arena_alloc_all[DType.bfloat16](arenas, MAX_M * ROWS)
    var o_test = arena_alloc_all[DType.bfloat16](arenas, MAX_M * ROWS)

    var xs = view.bind(x)
    var ws = view.bind(w)
    var outs_ref = view.bind(o_ref)
    var outs_test = view.bind(o_test)

    fill_pattern_all(xs, MAX_M * HIDDEN)
    fill_pattern_all(ws, ROWS * HIDDEN)
    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    var cap = pools[0].get_capacity()
    print(t"rows={ROWS} cols={HIDDEN} tp={tp} pool_capacity={cap}")

    var pprof = Profiler[False]()
    dispatch_gemm[cols=HIDDEN, MR=4](xs, ws, outs_ref, ROWS, MAX_M, pools, pprof)
    dispatch_gemm_panel[cols=HIDDEN, MR=4, NC=2](
        xs, ws, outs_test, ROWS, MAX_M, pools, pprof)
    var worst = parity_check(outs_ref, outs_test, MAX_M)
    print(t"\nparity (gemm vs panelNC2) max abs diff over M={MAX_M}: {worst}")

    print("\n=== GEMM baseline vs panel (GROUPS=1, NC tiling) ===")
    var sizes = InlineArray[Int, NUM_SIZES](uninitialized=True)
    sizes[0] = 1; sizes[1] = 16; sizes[2] = 64; sizes[3] = 256; sizes[4] = MAX_M
    var samples = SampleBuffer(SAMPLES)

    for i in range(NUM_SIZES):
        var m = sizes[i]
        var b = time_baseline(pools, samples, m, xs, ws, outs_ref)
        var p1 = time_panel[NC=1](pools, samples, m, xs, ws, outs_test)
        var p2 = time_panel[NC=2](pools, samples, m, xs, ws, outs_test)
        var p4 = time_panel[NC=4](pools, samples, m, xs, ws, outs_test)
        report_speedup("panelNC1", b, p1)
        report_speedup("panelNC2", b, p2)
        report_speedup("panelNC4", b, p4)


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    print("GEMM panel benchmark (baseline vs unified-panel GROUPS=1)")
    var iso = len(topo.isolated_cpus)
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
        run_all(selected_pools, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_bench,
    ](topo, "mode: isolated", "mode: spin-backoff")
