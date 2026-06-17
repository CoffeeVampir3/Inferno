from std.collections import InlineArray
from std.memory import Span, UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.attention_ops import KVRunTable, flash_partial_stride
from kernels.helpers import Binding, RankView
from kernels.logsum_merge import MergeSegment
from kernels.profiling import Profiler
from butterquant_kernels import (
    dispatch_bq_sliding_attention, dispatch_bq_full_attention,
)
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
    DEFAULT_SAMPLES,
)

comptime ALIGNMENT = 64
comptime WARMUP = 30
comptime SAMPLES = DEFAULT_SAMPLES
comptime MAX_SEQ = 4096
comptime MAX_WORKERS = 128
comptime NUM_CTX_SIZES = 8

comptime SLIDING_HEAD_DIM = 256
comptime SLIDING_NUM_Q = 4
comptime SLIDING_NUM_KV = 2
comptime SLIDING_GQA_RATIO = 2
comptime SLIDING_KV_STRIDE = 512
comptime SLIDING_WINDOW = 4096

comptime FULL_HEAD_DIM = 512
comptime FULL_GLOBAL_NUM_Q = 16
comptime FULL_NUM_KV = 2
comptime FULL_GQA_RATIO = FULL_GLOBAL_NUM_Q // FULL_NUM_KV
comptime FULL_KV_STRIDE = 1024

comptime SLIDING_PSTRIDE = flash_partial_stride(SLIDING_NUM_Q, SLIDING_HEAD_DIM)
comptime FULL_PSTRIDE = flash_partial_stride(FULL_GLOBAL_NUM_Q, FULL_HEAD_DIM)

comptime I8Ptr = UnsafePointer[Int8, MutAnyOrigin]
comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]


def arena_bases(
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
) -> List[Int]:
    var bases = List[Int](capacity=len(arenas))
    for r in range(len(arenas)):
        bases.append(Int(arenas[r].base.value()))
    return bases^


def arena_alloc[T: AnyType](
    mut arena: NumaArena[alignment=ALIGNMENT], count: Int,
) -> UnsafePointer[T, MutAnyOrigin]:
    var ptr = arena.alloc[T](count)
    if not ptr:
        print("arena alloc failed for", count, "elements")
        return UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
    return ptr.value()


def arena_alloc_all[T: AnyType](
    mut arenas: List[NumaArena[alignment=ALIGNMENT]], count: Int,
) -> UnsafePointer[T, MutAnyOrigin]:
    var first = UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
    for r in range(len(arenas)):
        var ptr = arena_alloc[T](arenas[r], count)
        if r == 0:
            first = ptr
    return first


def run_bq_sliding_decode[P: BurstThreadPool, o: ImmutOrigin, //](
    q: Binding[Int8, o],
    qi_bias: Binding[Float32, o],
    f_q: Binding[Float32, o],
    k_cache: Binding[Int8, o],
    k_scale: Binding[Float32, o],
    v_cache: Binding[Int8, o],
    v_scale: Binding[Float32, o],
    output: Binding[BFloat16, o],
    partials: Binding[Float32, o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    base_pos: Int,
    mut pools: List[P],
):
    var prof = Profiler[False]()
    runs[].runs[0].base_pos = Int32(base_pos)
    dispatch_bq_sliding_attention[
        head_dim=SLIDING_HEAD_DIM, max_q=SLIDING_NUM_Q,
        gqa_ratio=SLIDING_GQA_RATIO,
        window=SLIDING_WINDOW, cache_size=SLIDING_WINDOW,
        page_len=SLIDING_WINDOW,
        max_worker_count=MAX_WORKERS,
    ](
        q, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
        output, partials, runs,
        SLIDING_NUM_Q, SLIDING_NUM_KV, SLIDING_PSTRIDE, SLIDING_KV_STRIDE,
        1, pools, prof)


def run_bq_full_decode[P: BurstThreadPool, o: ImmutOrigin, //](
    q: Binding[Int8, o],
    qi_bias: Binding[Float32, o],
    f_q: Binding[Float32, o],
    k_cache: Binding[Int8, o],
    k_scale: Binding[Float32, o],
    v_cache: Binding[Int8, o],
    v_scale: Binding[Float32, o],
    output: Binding[BFloat16, o],
    partials: Binding[Float32, o],
    segments: Binding[MergeSegment, o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    base_pos: Int,
    mut pools: List[P],
):
    var prof = Profiler[False]()
    var local_num_q = FULL_GLOBAL_NUM_Q // len(pools)
    runs[].runs[0].base_pos = Int32(base_pos)
    dispatch_bq_full_attention[
        head_dim=FULL_HEAD_DIM, num_q=FULL_GLOBAL_NUM_Q,
        num_kv=FULL_NUM_KV, gqa_ratio=FULL_GQA_RATIO,
        kv_stride=FULL_KV_STRIDE, partial_stride=FULL_PSTRIDE,
        page_len=MAX_SEQ,
        max_worker_count=MAX_WORKERS,
    ](
        q, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
        output, partials, segments, runs,
        local_num_q, 1, pools, prof)


def fill_i8(ptr: I8Ptr, count: Int, phase: Int):
    for i in range(count):
        var v = ((i * 17 + phase * 29) % 127) - 63
        ptr[i] = Int8(v)


def fill_i8_all[o: ImmutOrigin](
    ptrs: Binding[Int8, o], count: Int, phase: Int,
):
    for r in range(ptrs.degree()):
        fill_i8(ptrs[r], count, phase + r)


def fill_scales(ptr: F32Ptr, count: Int, base: Float32):
    for i in range(count):
        ptr[i] = base + Float32((i * 13) % 23) * Float32(0.003)


def fill_scales_all[o: ImmutOrigin](
    ptrs: Binding[Float32, o], count: Int, base: Float32,
):
    for r in range(ptrs.degree()):
        fill_scales(ptrs[r], count, base + Float32(r) * Float32(0.002))


def fill_q_aux[head_dim: Int, num_q: Int](
    q: I8Ptr, qi_bias: F32Ptr, f_q: F32Ptr,
):
    comptime inv_sqrt = (
        Float32(0.0625)
        if head_dim == 256 else Float32(0.04419417382415922)
    )
    for h in range(num_q):
        var qsum = Int(0)
        for j in range(head_dim):
            qsum += Int(q[h * head_dim + j])
        qi_bias[h] = Float32(qsum) * Float32(128.0)
        f_q[h] = (Float32(0.22) + Float32(h % 5) * Float32(0.015)) * inv_sqrt


def fill_q_aux_all[head_dim: Int, num_q: Int, o: ImmutOrigin](
    q: Binding[Int8, o], qi_bias: Binding[Float32, o],
    f_q: Binding[Float32, o],
):
    for r in range(q.degree()):
        fill_q_aux[head_dim, num_q](q[r], qi_bias[r], f_q[r])


@always_inline
def abs_f32(x: Float32) -> Float32:
    return -x if x < Float32(0) else x


def check_single_token(label: StringSlice, output: BF16Ptr, v: I8Ptr, vs: F32Ptr):
    var expected = BFloat16(Float32(Int(v[0])) * vs[0] / Float32(127.0)).cast[DType.float32]()
    var actual = output[0].cast[DType.float32]()
    var diff = abs_f32(actual - expected)
    var ok = diff < Float32(0.004)
    print(t"  {label}: actual={actual} expected={expected} diff={diff} ", "OK" if ok else "FAIL")


def has_nan(ptr: BF16Ptr, count: Int) -> Bool:
    for i in range(count):
        var v = ptr[i].cast[DType.float32]()
        if v != v:
            return True
    return False


def section_validation[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    sliding_runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    full_runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    sliding_q: Binding[Int8, o],
    sliding_qi_bias: Binding[Float32, o],
    sliding_f_q: Binding[Float32, o],
    sliding_k: Binding[Int8, o],
    sliding_ks: Binding[Float32, o],
    sliding_v: Binding[Int8, o],
    sliding_vs: Binding[Float32, o],
    sliding_output: Binding[BFloat16, o],
    sliding_partials: Binding[Float32, o],
    full_q: Binding[Int8, o],
    full_qi_bias: Binding[Float32, o],
    full_f_q: Binding[Float32, o],
    full_k: Binding[Int8, o],
    full_ks: Binding[Float32, o],
    full_v: Binding[Int8, o],
    full_vs: Binding[Float32, o],
    full_output: Binding[BFloat16, o],
    full_partials: Binding[Float32, o],
    full_segments: Binding[MergeSegment, o],
):
    print("\n=== Validation ===")
    var full_local_num_q = FULL_GLOBAL_NUM_Q // len(pools)

    run_bq_sliding_decode(
        sliding_q, sliding_qi_bias, sliding_f_q,
        sliding_k, sliding_ks, sliding_v, sliding_vs,
        sliding_output, sliding_partials, sliding_runs, 0, pools)
    check_single_token("sliding seq=1", sliding_output[0], sliding_v[0], sliding_vs[0])

    run_bq_full_decode(
        full_q, full_qi_bias, full_f_q,
        full_k, full_ks, full_v, full_vs,
        full_output, full_partials, full_segments, full_runs, 0, pools)
    check_single_token("full seq=1", full_output[0], full_v[0], full_vs[0])

    run_bq_sliding_decode(
        sliding_q, sliding_qi_bias, sliding_f_q,
        sliding_k, sliding_ks, sliding_v, sliding_vs,
        sliding_output, sliding_partials, sliding_runs, 63, pools)
    var sliding_bad = has_nan(
        sliding_output[0], SLIDING_NUM_Q * SLIDING_HEAD_DIM)

    run_bq_full_decode(
        full_q, full_qi_bias, full_f_q,
        full_k, full_ks, full_v, full_vs,
        full_output, full_partials, full_segments, full_runs, 63, pools)
    var full_bad = has_nan(
        full_output[0], full_local_num_q * FULL_HEAD_DIM)

    print("  sliding seq=64 ", "FAIL: NaN detected" if sliding_bad else "OK (no NaN)")
    print("  full seq=64 ", "FAIL: NaN detected" if full_bad else "OK (no NaN)")


def section_sliding_sweep[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    q: Binding[Int8, o],
    qi_bias: Binding[Float32, o],
    f_q: Binding[Float32, o],
    k: Binding[Int8, o],
    ks: Binding[Float32, o],
    v: Binding[Int8, o],
    vs: Binding[Float32, o],
    output: Binding[BFloat16, o],
    partials: Binding[Float32, o],
):
    print("\n=== Sliding decode sweep (BQ kernel + merge) ===")
    var sizes = InlineArray[Int, NUM_CTX_SIZES](fill=0)
    sizes[0] = 1; sizes[1] = 8; sizes[2] = 32; sizes[3] = 128
    sizes[4] = 256; sizes[5] = 512; sizes[6] = 1024; sizes[7] = 4096

    var samples = SampleBuffer(SAMPLES)
    for s in range(NUM_CTX_SIZES):
        var vl = sizes[s]
        if vl > SLIDING_WINDOW:
            continue
        var pos = vl - 1
        for _ in range(WARMUP):
            run_bq_sliding_decode(
                q, qi_bias, f_q, k, ks, v, vs, output, partials, runs, pos, pools)
            keep(output[0][0])

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            run_bq_sliding_decode(
                q, qi_bias, f_q, k, ks, v, vs, output, partials, runs, pos, pools)
            var t1 = now_ns()
            var t_done = max_last_ts(pools)
            samples.push(t_done - t0, t1 - t0)
        keep(output[0][0])

        var ks_stats = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        var kv_bytes = vl * SLIDING_KV_STRIDE * 2 * 2
        print_row(String(t"seq={vl}"), ks_stats, ws, kv_bytes)


def section_full_sweep[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    q: Binding[Int8, o],
    qi_bias: Binding[Float32, o],
    f_q: Binding[Float32, o],
    k: Binding[Int8, o],
    ks: Binding[Float32, o],
    v: Binding[Int8, o],
    vs: Binding[Float32, o],
    output: Binding[BFloat16, o],
    partials: Binding[Float32, o],
    segments: Binding[MergeSegment, o],
):
    print("\n=== Full decode sweep (BQ kernel + context merge) ===")
    var sizes = InlineArray[Int, NUM_CTX_SIZES](fill=0)
    sizes[0] = 1; sizes[1] = 8; sizes[2] = 32; sizes[3] = 128
    sizes[4] = 256; sizes[5] = 512; sizes[6] = 1024; sizes[7] = 4096

    var samples = SampleBuffer(SAMPLES)
    for s in range(NUM_CTX_SIZES):
        var vl = sizes[s]
        if vl > MAX_SEQ:
            continue
        var pos = vl - 1
        for _ in range(WARMUP):
            run_bq_full_decode(
                q, qi_bias, f_q, k, ks, v, vs,
                output, partials, segments, runs, pos, pools)
            keep(output[0][0])

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            run_bq_full_decode(
                q, qi_bias, f_q, k, ks, v, vs,
                output, partials, segments, runs, pos, pools)
            var t1 = now_ns()
            var t_done = max_last_ts(pools)
            samples.push(t_done - t0, t1 - t0)
        keep(output[0][0])

        var ks_stats = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        var kv_bytes = vl * FULL_KV_STRIDE * 2 * 2
        print_row(String(t"seq={vl}"), ks_stats, ws, kv_bytes)


def run_all[P: BurstThreadPool, //](
    var pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    if FULL_GLOBAL_NUM_Q % tp != 0:
        print("full Q heads must divide tensor-parallel degree")
        return
    if MAX_SEQ % tp != 0:
        print("MAX_SEQ must divide tensor-parallel degree")
        return
    var full_local_num_q = FULL_GLOBAL_NUM_Q // tp
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var sliding_q_ptr = arena_alloc_all[Int8](
        arenas, SLIDING_NUM_Q * SLIDING_HEAD_DIM)
    var sliding_qi_bias_ptr = arena_alloc_all[Float32](
        arenas, SLIDING_NUM_Q)
    var sliding_f_q_ptr = arena_alloc_all[Float32](
        arenas, SLIDING_NUM_Q)
    var sliding_k_ptr = arena_alloc_all[Int8](
        arenas, SLIDING_WINDOW * SLIDING_KV_STRIDE)
    var sliding_ks_ptr = arena_alloc_all[Float32](
        arenas, SLIDING_WINDOW * SLIDING_NUM_KV)
    var sliding_v_ptr = arena_alloc_all[Int8](
        arenas, SLIDING_WINDOW * SLIDING_KV_STRIDE)
    var sliding_vs_ptr = arena_alloc_all[Float32](
        arenas, SLIDING_WINDOW * SLIDING_NUM_KV)
    var sliding_output_ptr = arena_alloc_all[BFloat16](
        arenas, SLIDING_NUM_Q * SLIDING_HEAD_DIM)
    var sliding_partials_ptr = arena_alloc_all[Float32](
        arenas, MAX_WORKERS * SLIDING_PSTRIDE)

    var full_q_ptr = arena_alloc_all[Int8](
        arenas, FULL_GLOBAL_NUM_Q * FULL_HEAD_DIM)
    var full_qi_bias_ptr = arena_alloc_all[Float32](
        arenas, FULL_GLOBAL_NUM_Q)
    var full_f_q_ptr = arena_alloc_all[Float32](
        arenas, FULL_GLOBAL_NUM_Q)
    var full_k_ptr = arena_alloc_all[Int8](
        arenas, MAX_SEQ * FULL_KV_STRIDE)
    var full_ks_ptr = arena_alloc_all[Float32](
        arenas, MAX_SEQ * FULL_NUM_KV)
    var full_v_ptr = arena_alloc_all[Int8](
        arenas, MAX_SEQ * FULL_KV_STRIDE)
    var full_vs_ptr = arena_alloc_all[Float32](
        arenas, MAX_SEQ * FULL_NUM_KV)
    var full_output_ptr = arena_alloc_all[BFloat16](
        arenas, full_local_num_q * FULL_HEAD_DIM)
    var full_partials_ptr = arena_alloc_all[Float32](
        arenas, MAX_WORKERS * FULL_PSTRIDE)
    var full_segments_ptr = arena_alloc_all[MergeSegment](
        arenas, MAX_WORKERS * tp)

    var sliding_q = view.bind(sliding_q_ptr)
    var sliding_qi_bias = view.bind(sliding_qi_bias_ptr)
    var sliding_f_q = view.bind(sliding_f_q_ptr)
    var sliding_k = view.bind(sliding_k_ptr)
    var sliding_ks = view.bind(sliding_ks_ptr)
    var sliding_v = view.bind(sliding_v_ptr)
    var sliding_vs = view.bind(sliding_vs_ptr)
    var sliding_output = view.bind(sliding_output_ptr)
    var sliding_partials = view.bind(sliding_partials_ptr)

    var full_q = view.bind(full_q_ptr)
    var full_qi_bias = view.bind(full_qi_bias_ptr)
    var full_f_q = view.bind(full_f_q_ptr)
    var full_k = view.bind(full_k_ptr)
    var full_ks = view.bind(full_ks_ptr)
    var full_v = view.bind(full_v_ptr)
    var full_vs = view.bind(full_vs_ptr)
    var full_output = view.bind(full_output_ptr)
    var full_partials = view.bind(full_partials_ptr)
    var full_segments = view.bind(full_segments_ptr)

    fill_i8_all(sliding_q, SLIDING_NUM_Q * SLIDING_HEAD_DIM, 1)
    fill_i8_all(sliding_k, SLIDING_WINDOW * SLIDING_KV_STRIDE, 2)
    fill_i8_all(sliding_v, SLIDING_WINDOW * SLIDING_KV_STRIDE, 3)
    fill_scales_all(
        sliding_ks, SLIDING_WINDOW * SLIDING_NUM_KV, Float32(0.18))
    fill_scales_all(
        sliding_vs, SLIDING_WINDOW * SLIDING_NUM_KV, Float32(0.20))
    fill_q_aux_all[SLIDING_HEAD_DIM, SLIDING_NUM_Q](
        sliding_q, sliding_qi_bias, sliding_f_q)

    fill_i8_all(full_q, FULL_GLOBAL_NUM_Q * FULL_HEAD_DIM, 4)
    fill_i8_all(full_k, MAX_SEQ * FULL_KV_STRIDE, 5)
    fill_i8_all(full_v, MAX_SEQ * FULL_KV_STRIDE, 6)
    fill_scales_all(
        full_ks, MAX_SEQ * FULL_NUM_KV, Float32(0.16))
    fill_scales_all(
        full_vs, MAX_SEQ * FULL_NUM_KV, Float32(0.19))
    fill_q_aux_all[FULL_HEAD_DIM, FULL_GLOBAL_NUM_Q](
        full_q, full_qi_bias, full_f_q)

    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    var cap = pools[0].get_capacity()
    print(t"pool capacity: {cap} workers")
    print(
        t"sliding: head_dim={SLIDING_HEAD_DIM} num_q={SLIDING_NUM_Q} "
        t"num_kv={SLIDING_NUM_KV} gqa={SLIDING_GQA_RATIO} "
        t"window={SLIDING_WINDOW}"
    )
    print(
        t"full: head_dim={FULL_HEAD_DIM} num_q={full_local_num_q} "
        t"num_kv={FULL_NUM_KV} gqa={FULL_GQA_RATIO} max_seq={MAX_SEQ}"
    )

    var sliding_runs_table = KVRunTable()
    sliding_runs_table.begin_run(0, 0)
    sliding_runs_table.add_base_row(Int32(0))
    var sliding_runs = UnsafePointer(to=sliding_runs_table)

    var full_runs_table = KVRunTable()
    full_runs_table.begin_run(0, 0)
    full_runs_table.add_base_row(Int32(0))
    var full_runs = UnsafePointer(to=full_runs_table)

    section_validation(
        pools, sliding_runs, full_runs,
        sliding_q, sliding_qi_bias, sliding_f_q,
        sliding_k, sliding_ks, sliding_v, sliding_vs,
        sliding_output, sliding_partials,
        full_q, full_qi_bias, full_f_q,
        full_k, full_ks, full_v, full_vs,
        full_output, full_partials, full_segments,
    )
    section_sliding_sweep(
        pools, sliding_runs, sliding_q, sliding_qi_bias, sliding_f_q,
        sliding_k, sliding_ks, sliding_v, sliding_vs,
        sliding_output, sliding_partials,
    )
    section_full_sweep(
        pools, full_runs, full_q, full_qi_bias, full_f_q,
        full_k, full_ks, full_v, full_vs,
        full_output, full_partials, full_segments,
    )


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("ButterQuant attention benchmark (apples-apples decode kernel + merge)")
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
    def dispatch_bq_attention_tp[
        P: BurstThreadPool, //,
    ](var selected_pools: List[P]):
        run_all(selected_pools^, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_bq_attention_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
