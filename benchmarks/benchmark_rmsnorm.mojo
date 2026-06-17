from std.collections import InlineArray
from std.memory import Span, UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.helpers import (
    DispatchBuffer, Binding, RankView,
)
from kernels.rmsnorm import (
    rms_reduce_row, rms_normalize_row, rms_norm_row,
    norm_residual_add_row,
    dispatch_rms_norm, fused_norm_residual_add,
    RmsNormTokenKernel,
)
from kernels.profiling import Profiler
from simd_math.ops import sqrt
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
    DEFAULT_SAMPLES,
)


comptime ALIGNMENT = 64
comptime WARMUP = 30
comptime SAMPLES = DEFAULT_SAMPLES

comptime HIDDEN = 2816
comptime SQRT_N = sqrt[DType.float32, 1](HIDDEN)
comptime N_EPS = HIDDEN * 1e-6

comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]


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


def fill_norm_input(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = BFloat16(Float32((i % 127) - 63) * 0.01)


def fill_ones(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = BFloat16(Float32(1.0) + Float32(i % 64) * 0.001)


def fill_norm_input_all[o: ImmutOrigin](
    ptrs: Binding[BFloat16, o], count: Int,
):
    for r in range(ptrs.degree()):
        fill_norm_input(ptrs[r], count)


def fill_ones_all[o: ImmutOrigin](
    ptrs: Binding[BFloat16, o], count: Int,
):
    for r in range(ptrs.degree()):
        fill_ones(ptrs[r], count)


def section_row_primitives(src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr):
    print(t"\n=== Row primitives (single token, HIDDEN={HIDDEN}) ===")

    var samples = SampleBuffer(SAMPLES)

    for _ in range(WARMUP):
        var s = rms_reduce_row[HIDDEN](src)
        keep(s)
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        var s = rms_reduce_row[HIDDEN](src)
        keep(s)
        var t1 = now_ns()
        samples.push(t1 - t0, t1 - t0)
    var ks = compute_stats(samples.kernel_ns, samples.n)
    var ws = compute_stats(samples.wall_ns, samples.n)
    print_row("reduce", ks, ws, HIDDEN * 2)

    for _ in range(WARMUP):
        rms_normalize_row[HIDDEN](src, dst, weight, Float32(0.5))
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        rms_normalize_row[HIDDEN](src, dst, weight, Float32(0.5))
        var t1 = now_ns()
        samples.push(t1 - t0, t1 - t0)
    keep(dst[0])
    var ks2 = compute_stats(samples.kernel_ns, samples.n)
    var ws2 = compute_stats(samples.wall_ns, samples.n)
    print_row("normalize", ks2, ws2, HIDDEN * 2 * 3)

    for _ in range(WARMUP):
        rms_norm_row[HIDDEN, SQRT_N, N_EPS](src, dst, weight)
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        rms_norm_row[HIDDEN, SQRT_N, N_EPS](src, dst, weight)
        var t1 = now_ns()
        samples.push(t1 - t0, t1 - t0)
    keep(dst[0])
    var ks3 = compute_stats(samples.kernel_ns, samples.n)
    var ws3 = compute_stats(samples.wall_ns, samples.n)
    print_row("full norm", ks3, ws3, HIDDEN * 2 * 3)

    for _ in range(WARMUP):
        rms_norm_row[HIDDEN, SQRT_N, N_EPS, scaled=False](src, dst, weight)
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        rms_norm_row[HIDDEN, SQRT_N, N_EPS, scaled=False](src, dst, weight)
        var t1 = now_ns()
        samples.push(t1 - t0, t1 - t0)
    keep(dst[0])
    var ks4 = compute_stats(samples.kernel_ns, samples.n)
    var ws4 = compute_stats(samples.wall_ns, samples.n)
    print_row("norm (no weight)", ks4, ws4, HIDDEN * 2 * 2)


def section_fused_primitives(
    src: BF16Ptr, residual: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
):
    print("\n=== Fused row primitives (single token) ===")

    var samples = SampleBuffer(SAMPLES)

    for _ in range(WARMUP):
        norm_residual_add_row[HIDDEN, SQRT_N, N_EPS](src, residual, dst, weight)
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        norm_residual_add_row[HIDDEN, SQRT_N, N_EPS](
            src, residual, dst, weight)
        var t1 = now_ns()
        samples.push(t1 - t0, t1 - t0)
    keep(dst[0])
    var ks = compute_stats(samples.kernel_ns, samples.n)
    var ws = compute_stats(samples.wall_ns, samples.n)
    print_row("norm+residual add", ks, ws, HIDDEN * 2 * 4)


def section_dispatch_overhead[P: BurstThreadPool](
    mut pool: P, src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
):
    print("\n=== Dispatch overhead isolation (seq_len=1) ===")

    var samples = SampleBuffer(SAMPLES)

    for _ in range(WARMUP):
        rms_norm_row[HIDDEN, SQRT_N, N_EPS](src, dst, weight)
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        rms_norm_row[HIDDEN, SQRT_N, N_EPS](src, dst, weight)
        var t1 = now_ns()
        samples.push(t1 - t0, t1 - t0)
    keep(dst[0])
    var ks_inline = compute_stats(samples.kernel_ns, samples.n)
    var ws_inline = compute_stats(samples.wall_ns, samples.n)
    print_row("inline", ks_inline, ws_inline, 0)

    var buf = DispatchBuffer[RmsNormTokenKernel[HIDDEN, SQRT_N, N_EPS]]()
    for _ in range(WARMUP):
        buf.slot()[] = RmsNormTokenKernel[HIDDEN, SQRT_N, N_EPS](src, dst, weight, 0, 1)
        buf.dispatch(pool)
        pool.join()
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        buf.slot()[] = RmsNormTokenKernel[HIDDEN, SQRT_N, N_EPS](src, dst, weight, 0, 1)
        buf.dispatch(pool)
        pool.join()
        var t1 = now_ns()
        var t_done = pool.last_worker_timestamp()
        samples.push(t_done - t0, t1 - t0)
    keep(dst[0])
    var ks_disp = compute_stats(samples.kernel_ns, samples.n)
    var ws_disp = compute_stats(samples.wall_ns, samples.n)
    print_row("1w dispatch", ks_disp, ws_disp, 0)


def section_seq_sweep[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    src: Binding[BFloat16, o],
    dst: Binding[BFloat16, o],
    weight: Binding[BFloat16, o],
):
    print("\n=== Standalone norm: seq_len sweep ===")
    var src0 = src[0]
    var dst0 = dst[0]
    var weight0 = weight[0]

    comptime NUM_SIZES = 9
    var sizes = InlineArray[Int, NUM_SIZES](fill=0)
    sizes[0] = 1
    sizes[1] = 2
    sizes[2] = 4
    sizes[3] = 8
    sizes[4] = 16
    sizes[5] = 32
    sizes[6] = 64
    sizes[7] = 128
    sizes[8] = 256

    var samples = SampleBuffer(SAMPLES)
    var prof = Profiler[False]()

    for s in range(NUM_SIZES):
        var seq = sizes[s]

        for _ in range(WARMUP):
            for tok in range(seq):
                rms_norm_row[HIDDEN, SQRT_N, N_EPS](
                    src0 + tok * HIDDEN, dst0 + tok * HIDDEN, weight0)
        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            for tok in range(seq):
                rms_norm_row[HIDDEN, SQRT_N, N_EPS](
                    src0 + tok * HIDDEN, dst0 + tok * HIDDEN, weight0)
            var t1 = now_ns()
            samples.push(t1 - t0, t1 - t0)
        keep(dst0[0])
        var ks_in = compute_stats(samples.kernel_ns, samples.n)
        var ws_in = compute_stats(samples.wall_ns, samples.n)
        print_row(String(t"seq={seq} inline"), ks_in, ws_in, seq * HIDDEN * 2)

        for _ in range(WARMUP):
            dispatch_rms_norm[hidden=HIDDEN, sqrt_n=SQRT_N, n_eps=N_EPS](
                src, dst, weight, seq, pools, prof)
        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            dispatch_rms_norm[hidden=HIDDEN, sqrt_n=SQRT_N, n_eps=N_EPS](
                src, dst, weight, seq, pools, prof)
            var t1 = now_ns()
            var t_done = max_last_ts(pools)
            samples.push(t_done - t0, t1 - t0)
        keep(dst0[0])
        var ks_d = compute_stats(samples.kernel_ns, samples.n)
        var ws_d = compute_stats(samples.wall_ns, samples.n)
        print_row(String(t"seq={seq} dispatch"), ks_d, ws_d, seq * HIDDEN * 2)


def section_fused_sweep[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    partial: Binding[BFloat16, o],
    residual: Binding[BFloat16, o],
    res_dst: Binding[BFloat16, o],
    weight: Binding[BFloat16, o],
):
    print("\n=== Norm+residual add: seq_len sweep ===")
    var partial0 = partial[0]
    var residual0 = residual[0]
    var res_dst0 = res_dst[0]
    var weight0 = weight[0]

    comptime NUM_SIZES = 9
    var sizes = InlineArray[Int, NUM_SIZES](fill=0)
    sizes[0] = 1
    sizes[1] = 2
    sizes[2] = 4
    sizes[3] = 8
    sizes[4] = 16
    sizes[5] = 32
    sizes[6] = 64
    sizes[7] = 128
    sizes[8] = 256

    var samples = SampleBuffer(SAMPLES)
    var prof = Profiler[False]()

    for s in range(NUM_SIZES):
        var seq = sizes[s]

        for _ in range(WARMUP):
            for tok in range(seq):
                norm_residual_add_row[HIDDEN, SQRT_N, N_EPS](
                    partial0 + tok * HIDDEN, residual0 + tok * HIDDEN,
                    res_dst0 + tok * HIDDEN, weight0)
        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            for tok in range(seq):
                norm_residual_add_row[HIDDEN, SQRT_N, N_EPS](
                    partial0 + tok * HIDDEN, residual0 + tok * HIDDEN,
                    res_dst0 + tok * HIDDEN, weight0)
            var t1 = now_ns()
            samples.push(t1 - t0, t1 - t0)
        keep(res_dst0[0])
        var ks_in = compute_stats(samples.kernel_ns, samples.n)
        var ws_in = compute_stats(samples.wall_ns, samples.n)
        print_row(String(t"seq={seq} inline"), ks_in, ws_in, seq * HIDDEN * 4)

        for _ in range(WARMUP):
            fused_norm_residual_add[
                hidden=HIDDEN, sqrt_n=SQRT_N, n_eps=N_EPS,
            ](partial, residual, res_dst, weight, seq, pools, prof)
        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            fused_norm_residual_add[
                hidden=HIDDEN, sqrt_n=SQRT_N, n_eps=N_EPS,
            ](partial, residual, res_dst, weight, seq, pools, prof)
            var t1 = now_ns()
            var t_done = max_last_ts(pools)
            samples.push(t_done - t0, t1 - t0)
        keep(res_dst0[0])
        var ks_d = compute_stats(samples.kernel_ns, samples.n)
        var ws_d = compute_stats(samples.wall_ns, samples.n)
        print_row(String(t"seq={seq} dispatch"), ks_d, ws_d, seq * HIDDEN * 4)


def run_all[P: BurstThreadPool, //](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    comptime MAX_TOKENS = 256
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))
    var src_ptr = arena_alloc_all[DType.bfloat16](arenas, MAX_TOKENS * HIDDEN)
    var dst_ptr = arena_alloc_all[DType.bfloat16](arenas, MAX_TOKENS * HIDDEN)
    var weight_ptr = arena_alloc_all[DType.bfloat16](arenas, HIDDEN)
    var partial_ptr = arena_alloc_all[DType.bfloat16](arenas, MAX_TOKENS * HIDDEN)
    var residual_ptr = arena_alloc_all[DType.bfloat16](arenas, MAX_TOKENS * HIDDEN)
    var res_dst_ptr = arena_alloc_all[DType.bfloat16](arenas, MAX_TOKENS * HIDDEN)

    var src = view.bind(src_ptr)
    var dst = view.bind(dst_ptr)
    var weight = view.bind(weight_ptr)
    var partial = view.bind(partial_ptr)
    var residual = view.bind(residual_ptr)
    var res_dst = view.bind(res_dst_ptr)

    fill_norm_input_all(src, MAX_TOKENS * HIDDEN)
    fill_norm_input_all(partial, MAX_TOKENS * HIDDEN)
    fill_norm_input_all(residual, MAX_TOKENS * HIDDEN)
    fill_ones_all(weight, HIDDEN)
    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    var cap = pools[0].get_capacity()
    print(t"pool capacity: {cap} workers")
    var hidden_bytes = HIDDEN * 2
    print(t"hidden: {HIDDEN} ({hidden_bytes} bytes bf16)")
    print(t"sqrt(N): {SQRT_N}, N*eps: {N_EPS}")

    section_row_primitives(src_ptr, dst_ptr, weight_ptr)
    section_fused_primitives(partial_ptr, residual_ptr, res_dst_ptr, weight_ptr)
    section_dispatch_overhead(pools[0], src_ptr, dst_ptr, weight_ptr)
    section_seq_sweep(pools, src, dst, weight)
    section_fused_sweep(pools, partial, residual, res_dst, weight)


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("RMSNorm kernel benchmark")
    var iso = len(topo.isolated_cpus)
    print(t"{tp} NUMA node(s), {iso} isolated cpus\n")

    comptime ARENA_BYTES = 256 * 1024 * 1024
    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_rmsnorm_tp[
        P: BurstThreadPool, //,
    ](var selected_pools: List[P]):
        run_all(selected_pools, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_rmsnorm_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
