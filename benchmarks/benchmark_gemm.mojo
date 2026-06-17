from std.collections import InlineArray
from std.memory import Span, UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.helpers import Binding, RankView
from kernels.gemm import dispatch_gemm, dispatch_gemm_cols
from kernels.profiling import Profiler
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
    DEFAULT_SAMPLES,
)


comptime ALIGNMENT = 64
comptime WARMUP = 20
comptime SAMPLES = DEFAULT_SAMPLES

comptime HIDDEN = 2816
comptime INTERMEDIATE = 2112

comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime MAX_M = 4096


def arena_alloc[dtype: DType](
    mut arena: NumaArena[alignment=ALIGNMENT], count: Int,
) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    var ptr = arena.alloc[Scalar[dtype]](count)
    if not ptr:
        print("arena alloc failed for", count, "elements")
        return UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()
    return ptr.value()


def arena_bases(
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
) -> List[Int]:
    var bases = List[Int](capacity=len(arenas))
    for r in range(len(arenas)):
        bases.append(Int(arenas[r].base.value()))
    return bases^


def arena_alloc_all[dtype: DType](
    mut arenas: List[NumaArena[alignment=ALIGNMENT]], count: Int,
) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    var first = UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()
    for r in range(len(arenas)):
        var ptr = arena_alloc[dtype](arenas[r], count)
        if r == 0:
            first = ptr
    return first


def fill_pattern(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = BFloat16(Float32((i % 253) - 126) * 0.005)


def fill_pattern_all[o: ImmutOrigin](
    ptrs: Binding[BFloat16, o], count: Int,
):
    for r in range(ptrs.degree()):
        fill_pattern(ptrs[r], count)


def measure_gemm_m[
    P: BurstThreadPool, o: ImmutOrigin, //, cols: Int, MR: Int,
](
    mut pools: List[P],
    xs: Binding[BFloat16, o], ws: Binding[BFloat16, o],
    outs: Binding[BFloat16, o], output: BF16Ptr,
    rows: Int,
    m: Int, mut samples: SampleBuffer,
):
    var prof = Profiler[False]()
    for _ in range(WARMUP):
        dispatch_gemm[cols=cols, MR=MR](
            xs, ws, outs, rows, m, pools, prof)
    keep(output[0])
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        dispatch_gemm[cols=cols, MR=MR](
            xs, ws, outs, rows, m, pools, prof)
        var t1 = now_ns()
        var t_done = max_last_ts(pools)
        samples.push(t_done - t0, t1 - t0)
    keep(output[0])
    var ks = compute_stats(samples.kernel_ns, samples.n)
    var wsx = compute_stats(samples.wall_ns, samples.n)
    var bytes_payload = (m * cols + rows * cols) * 2
    print_row(String(t"M={m} MR={MR}"), ks, wsx, bytes_payload)


def measure_gemm_cols_m[
    P: BurstThreadPool, o: ImmutOrigin, //, rows: Int, MR: Int,
](
    mut pools: List[P],
    xs: Binding[BFloat16, o], ws: Binding[BFloat16, o],
    outs: Binding[BFloat16, o], output: BF16Ptr,
    cols: Int,
    m: Int, mut samples: SampleBuffer,
):
    var prof = Profiler[False]()
    for _ in range(WARMUP):
        dispatch_gemm_cols[rows=rows, MR=MR](
            xs, ws, outs, cols, m, pools, prof)
    keep(output[0])
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        dispatch_gemm_cols[rows=rows, MR=MR](
            xs, ws, outs, cols, m, pools, prof)
        var t1 = now_ns()
        var t_done = max_last_ts(pools)
        samples.push(t_done - t0, t1 - t0)
    keep(output[0])
    var ks = compute_stats(samples.kernel_ns, samples.n)
    var wsx = compute_stats(samples.wall_ns, samples.n)
    var bytes_payload = (m * cols + rows * cols) * 2
    print_row(String(t"M={m} MR={MR}"), ks, wsx, bytes_payload)


def section_m_sweep[
    P: BurstThreadPool, o: ImmutOrigin, //, cols: Int,
](
    mut pools: List[P],
    x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    view: RankView[o], rows: Int, label: String,
):
    var tp = len(pools)
    print(t"\n=== GEMM M-sweep, {label} (rows={rows} cols={cols} tp={tp}) ===")
    var xs = Binding[BFloat16, o](x, view)
    var ws = Binding[BFloat16, o](weight, view)
    var outs = Binding[BFloat16, o](output, view)
    var samples = SampleBuffer(SAMPLES)

    var ms = InlineArray[Int, 9](uninitialized=True)
    ms[0] = 1; ms[1] = 2; ms[2] = 4; ms[3] = 8
    ms[4] = 16; ms[5] = 64; ms[6] = 256; ms[7] = 1024; ms[8] = 4096

    for i in range(9):
        var m = ms[i]
        measure_gemm_m[cols=cols, MR=4](
            pools, xs, ws, outs, output, rows, m, samples)


def section_m_sweep_cols[
    P: BurstThreadPool, o: ImmutOrigin, //, rows: Int,
](
    mut pools: List[P],
    x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    view: RankView[o], cols: Int, label: String,
):
    var tp = len(pools)
    print(t"\n=== GEMM-cols M-sweep, {label} (rows={rows} cols={cols} tp={tp}) ===")
    var xs = Binding[BFloat16, o](x, view)
    var ws = Binding[BFloat16, o](weight, view)
    var outs = Binding[BFloat16, o](output, view)
    var samples = SampleBuffer(SAMPLES)

    var ms = InlineArray[Int, 9](uninitialized=True)
    ms[0] = 1; ms[1] = 2; ms[2] = 4; ms[3] = 8
    ms[4] = 16; ms[5] = 64; ms[6] = 256; ms[7] = 1024; ms[8] = 4096

    for i in range(9):
        var m = ms[i]
        measure_gemm_cols_m[rows=rows, MR=4](
            pools, xs, ws, outs, output, cols, m, samples)


def section_mr_sweep[
    P: BurstThreadPool, o: ImmutOrigin, //, cols: Int,
](
    mut pools: List[P],
    x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    view: RankView[o], rows: Int, label: String,
):
    var tp = len(pools)
    print(
        t"\n=== GEMM MR-sweep at M=64, {label} (rows={rows} cols={cols} tp={tp}) ==="
    )
    var xs = Binding[BFloat16, o](x, view)
    var ws = Binding[BFloat16, o](weight, view)
    var outs = Binding[BFloat16, o](output, view)
    var samples = SampleBuffer(SAMPLES)

    measure_gemm_m[cols=cols, MR=1](
        pools, xs, ws, outs, output, rows, 64, samples)
    measure_gemm_m[cols=cols, MR=2](
        pools, xs, ws, outs, output, rows, 64, samples)
    measure_gemm_m[cols=cols, MR=4](
        pools, xs, ws, outs, output, rows, 64, samples)
    measure_gemm_m[cols=cols, MR=8](
        pools, xs, ws, outs, output, rows, 64, samples)


def run_all[P: BurstThreadPool, //](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    var gate_up_rows = INTERMEDIATE // tp
    comptime gate_up_cols = HIDDEN
    comptime down_rows = HIDDEN
    var down_cols = INTERMEDIATE // tp

    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    comptime MAX_X_ELEMS = MAX_M * HIDDEN
    comptime MAX_W_ELEMS = HIDDEN * HIDDEN
    comptime MAX_O_ELEMS = MAX_M * HIDDEN

    var x = arena_alloc_all[DType.bfloat16](arenas, MAX_X_ELEMS)
    var w = arena_alloc_all[DType.bfloat16](arenas, MAX_W_ELEMS)
    var o = arena_alloc_all[DType.bfloat16](arenas, MAX_O_ELEMS)

    fill_pattern_all(view.bind(x), MAX_X_ELEMS)
    fill_pattern_all(view.bind(w), MAX_W_ELEMS)
    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    var cap = pools[0].get_capacity()
    print(t"pool capacity: {cap} workers")

    section_m_sweep[cols=gate_up_cols](
        pools, x, w, o, view, gate_up_rows, "FFN gate/up shape")
    section_m_sweep_cols[rows=down_rows](
        pools, x, w, o, view, down_cols, "FFN down shape")

    section_mr_sweep[cols=gate_up_cols](
        pools, x, w, o, view, gate_up_rows, "FFN gate/up shape")


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("GEMM kernel benchmark")
    var iso = len(topo.isolated_cpus)
    print(t"{tp} NUMA node(s), {iso} isolated cpus\n")

    comptime ARENA_BYTES = 1024 * 1024 * 1024
    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_gemm_tp[P: BurstThreadPool, //](var selected_pools: List[P]):
        run_all(selected_pools, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_gemm_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
