from std.collections import InlineArray
from std.memory import Span, UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.attention_ops import KVRun, KVRunTable, pow2_shift
from kernels.helpers import Binding, RankView
from kernels.rope import (
    rope_head, dispatch_rope_cache_write,
    init_rope_table, init_rope_table_partial_strided,
)
from kernels.profiling import Profiler
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
    DEFAULT_SAMPLES,
)


comptime ALIGNMENT = 64
comptime WARMUP = 30
comptime SAMPLES = DEFAULT_SAMPLES

comptime HEAD_DIM_SLIDING = 256
comptime HEAD_DIM_FULL = 512
comptime HALF_SLIDING = 128
comptime HALF_FULL = 64
comptime Q_DIM_SLIDING = 4096
comptime KV_DIM_SLIDING = 2048
comptime Q_DIM_FULL = 8192
comptime KV_DIM_FULL = 1024
comptime MAX_POS = 4096
comptime SLIDING_WINDOW = 1024

comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]


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
        ptr[i] = BFloat16(Float32((i % 127) - 63) * 0.01)


def fill_pattern_all[o: ImmutOrigin](
    ptrs: Binding[BFloat16, o], count: Int,
):
    for r in range(ptrs.degree()):
        fill_pattern(ptrs[r], count)


def init_sliding_tables_all[o: ImmutOrigin](
    cos_sl: Binding[Float32, o],
    sin_sl: Binding[Float32, o],
):
    for r in range(cos_sl.degree()):
        init_rope_table[HALF_SLIDING, MAX_POS](
            cos_sl[r], sin_sl[r], 10000.0)


def init_full_tables_all[o: ImmutOrigin](
    cos_fl: Binding[Float32, o],
    sin_fl: Binding[Float32, o],
):
    for r in range(cos_fl.degree()):
        init_rope_table_partial_strided[HALF_FULL, MAX_POS](
            cos_fl[r], sin_fl[r], 1000000.0, HEAD_DIM_FULL, 0, 1)


def section_head_primitive(data: BF16Ptr, cos_sl: F32Ptr, sin_sl: F32Ptr,
                           cos_fl: F32Ptr, sin_fl: F32Ptr):
    print("\n=== rope_head primitive (single head) ===")

    var samples = SampleBuffer(SAMPLES)

    for _ in range(WARMUP):
        rope_head[HALF_SLIDING, HALF_SLIDING](data, cos_sl, sin_sl)
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        rope_head[HALF_SLIDING, HALF_SLIDING](data, cos_sl, sin_sl)
        var t1 = now_ns()
        samples.push(t1 - t0, t1 - t0)
    keep(data[0])
    var ks_sl = compute_stats(samples.kernel_ns, samples.n)
    var ws_sl = compute_stats(samples.wall_ns, samples.n)
    print_row("sliding (256 dim, full rot)", ks_sl, ws_sl,
        HEAD_DIM_SLIDING * 2 + HALF_SLIDING * 4 * 2)

    for _ in range(WARMUP):
        rope_head[HALF_FULL, HEAD_DIM_FULL // 2](data, cos_fl, sin_fl)
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        rope_head[HALF_FULL, HEAD_DIM_FULL // 2](data, cos_fl, sin_fl)
        var t1 = now_ns()
        samples.push(t1 - t0, t1 - t0)
    keep(data[0])
    var ks_fl = compute_stats(samples.kernel_ns, samples.n)
    var ws_fl = compute_stats(samples.wall_ns, samples.n)
    print_row("full (512 dim, 128 partial)", ks_fl, ws_fl,
        HALF_FULL * 2 * 2 + HALF_FULL * 4 * 2)


def section_token_scaling(data: BF16Ptr, cos_sl: F32Ptr, sin_sl: F32Ptr):
    print("\n=== rope_head loop: head count scaling (sliding, single pos) ===")

    comptime NUM_SIZES = 5
    var head_counts = InlineArray[Int, NUM_SIZES](fill=0)
    head_counts[0] = 1
    head_counts[1] = 2
    head_counts[2] = 4
    head_counts[3] = 8
    head_counts[4] = 16

    var samples = SampleBuffer(SAMPLES)

    for s in range(NUM_SIZES):
        var nh = head_counts[s]

        for _ in range(WARMUP):
            for h in range(nh):
                rope_head[HALF_SLIDING, HALF_SLIDING](
                    data + h * HEAD_DIM_SLIDING, cos_sl, sin_sl)

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            for h in range(nh):
                rope_head[HALF_SLIDING, HALF_SLIDING](
                    data + h * HEAD_DIM_SLIDING, cos_sl, sin_sl)
            var t1 = now_ns()
            samples.push(t1 - t0, t1 - t0)
        keep(data[0])

        var data_bytes = nh * HEAD_DIM_SLIDING * 2
        var ks = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        print_row(String(t"heads={nh}"), ks, ws, data_bytes * 2)


def section_sliding_cache_write[
    P: BurstThreadPool, o: ImmutOrigin, //,
](
    mut pools: List[P],
    qs: Binding[BFloat16, o],
    ks: Binding[BFloat16, o],
    vs: Binding[BFloat16, o],
    kc: Binding[BFloat16, o],
    vc: Binding[BFloat16, o],
    cos: Binding[Float32, o],
    sin: Binding[Float32, o],
):
    var tp = len(pools)
    var q_rows = Q_DIM_SLIDING // tp
    var kv_rows = KV_DIM_SLIDING // tp
    var num_q = q_rows // HEAD_DIM_SLIDING
    var num_kv = kv_rows // HEAD_DIM_SLIDING
    comptime POS = 513
    comptime page_shift = pow2_shift(SLIDING_WINDOW)
    comptime row_mask = SLIDING_WINDOW - 1
    var prof = Profiler[False]()

    var runs_table = KVRunTable()
    var run = KVRun(0, POS)
    run.base_rows.append(Int32(0))
    runs_table.runs.append(run^)
    var runs = UnsafePointer(to=runs_table)

    for _ in range(WARMUP):
        dispatch_rope_cache_write[
            half=HALF_SLIDING, pair_stride=HEAD_DIM_SLIDING // 2,
            head_dim=HEAD_DIM_SLIDING,
        ](qs, ks, vs, kc, vc, cos, sin, runs,
          num_q, num_kv, 1, page_shift, row_mask, 0, 1, pools, prof)

    var samples = SampleBuffer(SAMPLES)
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        dispatch_rope_cache_write[
            half=HALF_SLIDING, pair_stride=HEAD_DIM_SLIDING // 2,
            head_dim=HEAD_DIM_SLIDING,
        ](qs, ks, vs, kc, vc, cos, sin, runs,
          num_q, num_kv, 1, page_shift, row_mask, 0, 1, pools, prof)
        var t1 = now_ns()
        var t_done = max_last_ts(pools)
        samples.push(t_done - t0, t1 - t0)
    keep(qs[0][0])

    var ks_stats = compute_stats(samples.kernel_ns, samples.n)
    var ws_stats = compute_stats(samples.wall_ns, samples.n)
    var sl_bytes = (q_rows + 2 * kv_rows) * 2
    print_row("sliding", ks_stats, ws_stats, sl_bytes)


def section_full_cache_write[
    P: BurstThreadPool, o: ImmutOrigin, //,
](
    mut pools: List[P],
    qs: Binding[BFloat16, o],
    ks: Binding[BFloat16, o],
    vs: Binding[BFloat16, o],
    kc: Binding[BFloat16, o],
    vc: Binding[BFloat16, o],
    cos: Binding[Float32, o],
    sin: Binding[Float32, o],
):
    var tp = len(pools)
    comptime NUM_Q = Q_DIM_FULL // HEAD_DIM_FULL
    comptime NUM_KV = KV_DIM_FULL // HEAD_DIM_FULL
    comptime POS = 513
    comptime page_shift = pow2_shift(MAX_POS)
    comptime row_mask = MAX_POS - 1
    var prof = Profiler[False]()

    var runs_table = KVRunTable()
    var run = KVRun(0, POS)
    run.base_rows.append(Int32(0))
    runs_table.runs.append(run^)
    var runs = UnsafePointer(to=runs_table)

    for _ in range(WARMUP):
        dispatch_rope_cache_write[
            half=HALF_FULL, pair_stride=HEAD_DIM_FULL // 2,
            head_dim=HEAD_DIM_FULL,
        ](qs, ks, vs, kc, vc, cos, sin, runs,
          NUM_Q, NUM_KV, tp, page_shift, row_mask, 0, 1, pools, prof)

    var samples = SampleBuffer(SAMPLES)
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        dispatch_rope_cache_write[
            half=HALF_FULL, pair_stride=HEAD_DIM_FULL // 2,
            head_dim=HEAD_DIM_FULL,
        ](qs, ks, vs, kc, vc, cos, sin, runs,
          NUM_Q, NUM_KV, tp, page_shift, row_mask, 0, 1, pools, prof)
        var t1 = now_ns()
        var t_done = max_last_ts(pools)
        samples.push(t_done - t0, t1 - t0)
    keep(qs[0][0])

    var ks_stats = compute_stats(samples.kernel_ns, samples.n)
    var ws_stats = compute_stats(samples.wall_ns, samples.n)
    var fl_bytes = (Q_DIM_FULL + 2 * KV_DIM_FULL) * 2
    print_row("full", ks_stats, ws_stats, fl_bytes)


def section_model_cache_write[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    sliding_q: Binding[BFloat16, o],
    sliding_k: Binding[BFloat16, o],
    sliding_v: Binding[BFloat16, o],
    sliding_k_cache: Binding[BFloat16, o],
    sliding_v_cache: Binding[BFloat16, o],
    full_q: Binding[BFloat16, o],
    full_k: Binding[BFloat16, o],
    full_v: Binding[BFloat16, o],
    full_k_cache: Binding[BFloat16, o],
    full_v_cache: Binding[BFloat16, o],
    cos_sl: Binding[Float32, o],
    sin_sl: Binding[Float32, o],
    cos_fl: Binding[Float32, o],
    sin_fl: Binding[Float32, o],
):
    var tp = len(pools)
    print(t"\n=== dispatch_rope_cache_write model path (seq_len=1, TP={tp}) ===")

    section_sliding_cache_write(
        pools, sliding_q, sliding_k, sliding_v,
        sliding_k_cache, sliding_v_cache, cos_sl, sin_sl)
    section_full_cache_write(
        pools, full_q, full_k, full_v,
        full_k_cache, full_v_cache, cos_fl, sin_fl)


def run_all[P: BurstThreadPool, //](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    comptime MAX_SEQ = 128
    comptime MAX_HEADS = 16
    comptime MAX_DATA = MAX_SEQ * MAX_HEADS * HEAD_DIM_FULL
    var sl_q_rows = Q_DIM_SLIDING // tp
    var sl_kv_rows = KV_DIM_SLIDING // tp

    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))
    var data = arena_alloc_all[DType.bfloat16](arenas, MAX_DATA)
    var cos_sl = arena_alloc_all[DType.float32](
        arenas, MAX_POS * HALF_SLIDING)
    var sin_sl = arena_alloc_all[DType.float32](
        arenas, MAX_POS * HALF_SLIDING)
    var cos_fl = arena_alloc_all[DType.float32](arenas, MAX_POS * HALF_FULL)
    var sin_fl = arena_alloc_all[DType.float32](arenas, MAX_POS * HALF_FULL)

    var sliding_q = arena_alloc_all[DType.bfloat16](arenas, sl_q_rows)
    var sliding_k = arena_alloc_all[DType.bfloat16](arenas, sl_kv_rows)
    var sliding_v = arena_alloc_all[DType.bfloat16](arenas, sl_kv_rows)
    var sliding_k_cache = arena_alloc_all[DType.bfloat16](
        arenas, SLIDING_WINDOW * sl_kv_rows)
    var sliding_v_cache = arena_alloc_all[DType.bfloat16](
        arenas, SLIDING_WINDOW * sl_kv_rows)

    var full_q = arena_alloc_all[DType.bfloat16](arenas, Q_DIM_FULL)
    var full_k = arena_alloc_all[DType.bfloat16](arenas, KV_DIM_FULL)
    var full_v = arena_alloc_all[DType.bfloat16](arenas, KV_DIM_FULL)
    var full_k_cache = arena_alloc_all[DType.bfloat16](
        arenas, MAX_POS * KV_DIM_FULL)
    var full_v_cache = arena_alloc_all[DType.bfloat16](
        arenas, MAX_POS * KV_DIM_FULL)

    var data_b = view.bind(data)
    var cos_sl_b = view.bind(cos_sl)
    var sin_sl_b = view.bind(sin_sl)
    var cos_fl_b = view.bind(cos_fl)
    var sin_fl_b = view.bind(sin_fl)
    var sliding_q_b = view.bind(sliding_q)
    var sliding_k_b = view.bind(sliding_k)
    var sliding_v_b = view.bind(sliding_v)
    var sliding_k_cache_b = view.bind(sliding_k_cache)
    var sliding_v_cache_b = view.bind(sliding_v_cache)
    var full_q_b = view.bind(full_q)
    var full_k_b = view.bind(full_k)
    var full_v_b = view.bind(full_v)
    var full_k_cache_b = view.bind(full_k_cache)
    var full_v_cache_b = view.bind(full_v_cache)

    fill_pattern_all(data_b, MAX_DATA)
    fill_pattern_all(sliding_q_b, sl_q_rows)
    fill_pattern_all(sliding_k_b, sl_kv_rows)
    fill_pattern_all(sliding_v_b, sl_kv_rows)
    fill_pattern_all(full_q_b, Q_DIM_FULL)
    fill_pattern_all(full_k_b, KV_DIM_FULL)
    fill_pattern_all(full_v_b, KV_DIM_FULL)
    init_sliding_tables_all(cos_sl_b, sin_sl_b)
    init_full_tables_all(cos_fl_b, sin_fl_b)

    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    var cap = pools[0].get_capacity()
    print(t"pool capacity: {cap} workers")
    print("sliding: head_dim=256, half=128, theta=10000")
    print("full:    head_dim=512, rotary_half=64, theta=1000000")

    section_head_primitive(data, cos_sl, sin_sl, cos_fl, sin_fl)
    section_token_scaling(data, cos_sl, sin_sl)
    section_model_cache_write(
        pools,
        sliding_q_b, sliding_k_b, sliding_v_b,
        sliding_k_cache_b, sliding_v_cache_b,
        full_q_b, full_k_b, full_v_b, full_k_cache_b, full_v_cache_b,
        cos_sl_b, sin_sl_b, cos_fl_b, sin_fl_b)


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("RoPE kernel benchmark")
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
    def dispatch_rope_tp[
        P: BurstThreadPool, //,
    ](var selected_pools: List[P]):
        run_all(selected_pools, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_rope_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
