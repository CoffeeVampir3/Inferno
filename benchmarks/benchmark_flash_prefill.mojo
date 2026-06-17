from std.collections import InlineArray
from std.memory import Span, UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.attention_dispatch_kernels import (
    dispatch_sliding_attention, dispatch_full_attention,
)
from kernels.attention_ops import KVRun, KVRunTable, flash_partial_stride
from kernels.helpers import Binding, RankView
from kernels.logsum_merge import MergeSegment
from kernels.profiling import Profiler
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
    DEFAULT_SAMPLES,
)


comptime ALIGNMENT = 64
comptime WARMUP = 2
comptime SAMPLES = 3
comptime MAX_WORKERS = 128

comptime SLIDING_HEAD_DIM = 256
comptime SLIDING_NUM_Q = 4
comptime SLIDING_NUM_KV = 2
comptime SLIDING_GQA = SLIDING_NUM_Q // SLIDING_NUM_KV
comptime SLIDING_KV_STRIDE = SLIDING_NUM_KV * SLIDING_HEAD_DIM
comptime SLIDING_WINDOW = 4096
comptime SLIDING_CHUNK_SIZE = 4096
comptime SLIDING_MAX_SEQ = 16384

comptime FULL_HEAD_DIM = 512
comptime FULL_NUM_Q = 16
comptime FULL_NUM_KV = 2
comptime FULL_GQA = FULL_NUM_Q // FULL_NUM_KV
comptime FULL_KV_STRIDE = FULL_NUM_KV * FULL_HEAD_DIM
comptime FULL_CHUNK_SIZE = 512
comptime FULL_MAX_SEQ = 4096
comptime FULL_PAGE_LEN = 1024

comptime SLIDING_PSTRIDE = flash_partial_stride(SLIDING_NUM_Q, SLIDING_HEAD_DIM)
comptime FULL_PSTRIDE = flash_partial_stride(FULL_NUM_Q, FULL_HEAD_DIM)

comptime NUM_SLIDING_SIZES = 6
comptime NUM_FULL_SIZES = 5

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


def fill_pattern(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = BFloat16(Float32((i % 127) - 63) * 0.01)


def fill_pattern_all[o: ImmutOrigin](
    ptrs: Binding[BFloat16, o], count: Int,
):
    for r in range(ptrs.degree()):
        fill_pattern(ptrs[r], count)


def section_sliding_sweep[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    q: Binding[BFloat16, o],
    k_cache: Binding[BFloat16, o],
    v_cache: Binding[BFloat16, o],
    output: Binding[BFloat16, o],
    worker_scratch: Binding[Float32, o],
):
    print(t"\n=== Sliding-window prefill sweep (chunk_size={SLIDING_CHUNK_SIZE}) ===")

    var sizes = InlineArray[Int, NUM_SLIDING_SIZES](fill=0)
    sizes[0] = 64; sizes[1] = 256; sizes[2] = 1024
    sizes[3] = 4096; sizes[4] = 8192; sizes[5] = 16384

    var samples = SampleBuffer(SAMPLES)
    var prof = Profiler[False]()

    comptime q_stride = SLIDING_NUM_Q * SLIDING_HEAD_DIM

    for s in range(NUM_SLIDING_SIZES):
        var seq_len = sizes[s]
        if seq_len > SLIDING_MAX_SEQ:
            continue

        for _ in range(WARMUP):
            var done = 0
            while done < seq_len:
                var chunk_len = min(SLIDING_CHUNK_SIZE, seq_len - done)
                runs[].runs[0].base_pos = Int32(done)
                dispatch_sliding_attention[
                    head_dim=SLIDING_HEAD_DIM, max_q=SLIDING_NUM_Q,
                    gqa_ratio=SLIDING_GQA,
                    window=SLIDING_WINDOW, cache_size=SLIDING_WINDOW,
                    page_len=SLIDING_WINDOW,
                    max_worker_count=MAX_WORKERS,
                ](
                    q.shifted(done * q_stride), k_cache, v_cache,
                    output.shifted(done * q_stride), worker_scratch, runs,
                    SLIDING_NUM_Q, SLIDING_PSTRIDE, SLIDING_KV_STRIDE,
                    chunk_len, pools, prof)
                done += chunk_len
            keep(output[0][0])

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            var done = 0
            while done < seq_len:
                var chunk_len = min(SLIDING_CHUNK_SIZE, seq_len - done)
                runs[].runs[0].base_pos = Int32(done)
                dispatch_sliding_attention[
                    head_dim=SLIDING_HEAD_DIM, max_q=SLIDING_NUM_Q,
                    gqa_ratio=SLIDING_GQA,
                    window=SLIDING_WINDOW, cache_size=SLIDING_WINDOW,
                    page_len=SLIDING_WINDOW,
                    max_worker_count=MAX_WORKERS,
                ](
                    q.shifted(done * q_stride), k_cache, v_cache,
                    output.shifted(done * q_stride), worker_scratch, runs,
                    SLIDING_NUM_Q, SLIDING_PSTRIDE, SLIDING_KV_STRIDE,
                    chunk_len, pools, prof)
                done += chunk_len
            var t1 = now_ns()
            var t_done = max_last_ts(pools)
            samples.push(t_done - t0, t1 - t0)
        keep(output[0][0])

        var ks = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        var per_q_kv = seq_len if seq_len < SLIDING_WINDOW else SLIDING_WINDOW
        var kv_bytes = seq_len * per_q_kv * SLIDING_KV_STRIDE * 2
        print_row(String(t"seq={seq_len}"), ks, ws, kv_bytes)


def section_full_sweep[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    q: Binding[BFloat16, o],
    k_cache: Binding[BFloat16, o],
    v_cache: Binding[BFloat16, o],
    output: Binding[BFloat16, o],
    partials_scratch: Binding[Float32, o],
    merge_segments: Binding[MergeSegment, o],
):
    print(t"\n=== Full-attention prefill sweep (chunk_size={FULL_CHUNK_SIZE}) ===")
    var tp = len(pools)
    var local_num_q = FULL_NUM_Q // tp

    var sizes = InlineArray[Int, NUM_FULL_SIZES](fill=0)
    sizes[0] = 64; sizes[1] = 256; sizes[2] = 1024
    sizes[3] = 2048; sizes[4] = 4096

    var samples = SampleBuffer(SAMPLES)
    var prof = Profiler[False]()

    comptime q_stride = FULL_NUM_Q * FULL_HEAD_DIM
    var local_q_stride = local_num_q * FULL_HEAD_DIM

    for s in range(NUM_FULL_SIZES):
        var seq_len = sizes[s]
        if seq_len > FULL_MAX_SEQ:
            continue

        for _ in range(WARMUP):
            var done = 0
            while done < seq_len:
                var chunk_len = min(FULL_CHUNK_SIZE, seq_len - done)
                runs[].runs[0].base_pos = Int32(done)
                dispatch_full_attention[
                    head_dim=FULL_HEAD_DIM, num_q=FULL_NUM_Q,
                    gqa_ratio=FULL_GQA,
                    kv_stride=FULL_KV_STRIDE, partial_stride=FULL_PSTRIDE,
                    page_len=FULL_PAGE_LEN,
                    max_worker_count=MAX_WORKERS,
                ](
                    q.shifted(done * q_stride), k_cache, v_cache,
                    output.shifted(done * local_q_stride),
                    partials_scratch, merge_segments, runs, local_num_q,
                    chunk_len, pools, prof)
                done += chunk_len
            keep(output[0][0])

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            var done = 0
            while done < seq_len:
                var chunk_len = min(FULL_CHUNK_SIZE, seq_len - done)
                runs[].runs[0].base_pos = Int32(done)
                dispatch_full_attention[
                    head_dim=FULL_HEAD_DIM, num_q=FULL_NUM_Q,
                    gqa_ratio=FULL_GQA,
                    kv_stride=FULL_KV_STRIDE, partial_stride=FULL_PSTRIDE,
                    page_len=FULL_PAGE_LEN,
                    max_worker_count=MAX_WORKERS,
                ](
                    q.shifted(done * q_stride), k_cache, v_cache,
                    output.shifted(done * local_q_stride),
                    partials_scratch, merge_segments, runs, local_num_q,
                    chunk_len, pools, prof)
                done += chunk_len
            var t1 = now_ns()
            var t_done = max_last_ts(pools)
            samples.push(t_done - t0, t1 - t0)
        keep(output[0][0])

        var ks = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        # Per-rank KV bytes: causal triangle / degree.
        # Average local KV scan per Q ≈ seq_len / (2 * tp); total scans = seq_len.
        var per_rank_kv = (seq_len * (seq_len + 1) // 2) // tp
        var kv_bytes = per_rank_kv * FULL_KV_STRIDE * 2
        print_row(String(t"seq={seq_len}"), ks, ws, kv_bytes)


def section_validation_sliding[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    q: Binding[BFloat16, o],
    k_cache: Binding[BFloat16, o],
    v_cache: Binding[BFloat16, o],
    output: Binding[BFloat16, o],
    worker_scratch: Binding[Float32, o],
):
    print("\n=== Validation (sliding, seq_len=64) ===")
    comptime SL = 64
    var prof = Profiler[False]()
    runs[].runs[0].base_pos = Int32(0)

    dispatch_sliding_attention[
        head_dim=SLIDING_HEAD_DIM, max_q=SLIDING_NUM_Q,
        gqa_ratio=SLIDING_GQA,
        window=SLIDING_WINDOW, cache_size=SLIDING_WINDOW,
        page_len=SLIDING_WINDOW,
        max_worker_count=MAX_WORKERS,
    ](q, k_cache, v_cache, output, worker_scratch, runs,
      SLIDING_NUM_Q, SLIDING_PSTRIDE, SLIDING_KV_STRIDE,
      SL, pools, prof)

    var out0 = output[0]
    var o0 = out0[0].cast[DType.float32]()
    var o1 = out0[1].cast[DType.float32]()
    var o2 = out0[2].cast[DType.float32]()
    var o3 = out0[3].cast[DType.float32]()
    print(t"  output[0..3]: {o0} {o1} {o2} {o3}")
    var ok = True
    for i in range(SL * SLIDING_NUM_Q * SLIDING_HEAD_DIM):
        var v = out0[i].cast[DType.float32]()
        if v != v:
            ok = False
            break
    print("  ", "OK (no NaN)" if ok else "FAIL: NaN detected")


def section_validation_full[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    q: Binding[BFloat16, o],
    k_cache: Binding[BFloat16, o],
    v_cache: Binding[BFloat16, o],
    output: Binding[BFloat16, o],
    partials_scratch: Binding[Float32, o],
    merge_segments: Binding[MergeSegment, o],
):
    print("\n=== Validation (full, seq_len=64) ===")
    comptime SL = 64
    var tp = len(pools)
    var local_num_q = FULL_NUM_Q // tp
    var out_stride = local_num_q * FULL_HEAD_DIM
    comptime M_OFF = FULL_NUM_Q * FULL_HEAD_DIM
    comptime L_OFF = M_OFF + FULL_NUM_Q
    var prof = Profiler[False]()
    runs[].runs[0].base_pos = Int32(0)

    dispatch_full_attention[
        head_dim=FULL_HEAD_DIM, num_q=FULL_NUM_Q,
        gqa_ratio=FULL_GQA,
        kv_stride=FULL_KV_STRIDE, partial_stride=FULL_PSTRIDE,
        page_len=FULL_PAGE_LEN,
        max_worker_count=MAX_WORKERS,
    ](q, k_cache, v_cache, output, partials_scratch, merge_segments, runs,
      local_num_q, SL, pools, prof)

    var out0 = output[0]
    var o0 = out0[0].cast[DType.float32]()
    var o1 = out0[1].cast[DType.float32]()
    var o2 = out0[2].cast[DType.float32]()
    var o3 = out0[3].cast[DType.float32]()
    print(t"  output[0..3]: {o0} {o1} {o2} {o3}")

    var first_nan = -1
    for i in range(SL * out_stride):
        var v = out0[i].cast[DType.float32]()
        if v != v:
            first_nan = i
            break

    if first_nan < 0:
        print("  OK (no NaN)")
        return

    var nan_tok = first_nan // out_stride
    var nan_in_tok = first_nan % out_stride
    var nan_local_h = nan_in_tok // FULL_HEAD_DIM
    var nan_lane = nan_in_tok % FULL_HEAD_DIM
    print(
        t"  FAIL: first NaN at flat={first_nan} token={nan_tok} "
        t"local_h={nan_local_h} lane={nan_lane}"
    )

    var nan_global_h = nan_local_h
    print(
        t"  global_h = q_rank(0) * local({local_num_q}) + "
        t"{nan_local_h} = {nan_global_h}"
    )

    print(t"  per-rank partial m/l at (t={nan_tok}, global_h={nan_global_h}):")
    for r in range(tp):
        var pr = partials_scratch[r]
        var slot = pr + nan_tok * FULL_PSTRIDE
        var m_val = (slot + M_OFF + nan_global_h)[]
        var l_val = (slot + L_OFF + nan_global_h)[]
        var acc0 = (slot + nan_global_h * FULL_HEAD_DIM)[]
        print(t"    rank={r} m={m_val} l={l_val} acc[0]={acc0}")


def run_sliding[P: BurstThreadPool, //](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var q_ptr = arena_alloc_all[BFloat16](
        arenas, SLIDING_MAX_SEQ * SLIDING_NUM_Q * SLIDING_HEAD_DIM)
    var k_cache_ptr = arena_alloc_all[BFloat16](
        arenas, SLIDING_WINDOW * SLIDING_KV_STRIDE)
    var v_cache_ptr = arena_alloc_all[BFloat16](
        arenas, SLIDING_WINDOW * SLIDING_KV_STRIDE)
    var output_ptr = arena_alloc_all[BFloat16](
        arenas, SLIDING_MAX_SEQ * SLIDING_NUM_Q * SLIDING_HEAD_DIM)
    var worker_scratch_ptr = arena_alloc_all[Float32](
        arenas, MAX_WORKERS * SLIDING_PSTRIDE)

    var q = view.bind(q_ptr)
    var k_cache = view.bind(k_cache_ptr)
    var v_cache = view.bind(v_cache_ptr)
    var output = view.bind(output_ptr)
    var worker_scratch = view.bind(worker_scratch_ptr)

    fill_pattern_all(
        q, SLIDING_MAX_SEQ * SLIDING_NUM_Q * SLIDING_HEAD_DIM)
    fill_pattern_all(k_cache, SLIDING_WINDOW * SLIDING_KV_STRIDE)
    fill_pattern_all(v_cache, SLIDING_WINDOW * SLIDING_KV_STRIDE)

    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    print(
        t"sliding: head_dim={SLIDING_HEAD_DIM} num_q={SLIDING_NUM_Q} "
        t"num_kv={SLIDING_NUM_KV} gqa={SLIDING_GQA} window={SLIDING_WINDOW}"
    )

    var runs_table = KVRunTable()
    var run = KVRun(0, 0)
    run.base_rows.append(Int32(0))
    runs_table.runs.append(run^)
    var runs = UnsafePointer(to=runs_table)

    section_validation_sliding(
        pools, runs, q, k_cache, v_cache, output, worker_scratch)
    section_sliding_sweep(
        pools, runs, q, k_cache, v_cache, output, worker_scratch)


def run_full[P: BurstThreadPool, //](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    var local_num_q = FULL_NUM_Q // tp
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var q_ptr = arena_alloc_all[BFloat16](
        arenas, FULL_MAX_SEQ * FULL_NUM_Q * FULL_HEAD_DIM)
    var k_cache_ptr = arena_alloc_all[BFloat16](
        arenas, FULL_MAX_SEQ * FULL_KV_STRIDE)
    var v_cache_ptr = arena_alloc_all[BFloat16](
        arenas, FULL_MAX_SEQ * FULL_KV_STRIDE)
    var output_ptr = arena_alloc_all[BFloat16](
        arenas, FULL_MAX_SEQ * local_num_q * FULL_HEAD_DIM)
    var partials_ptr = arena_alloc_all[Float32](
        arenas, FULL_CHUNK_SIZE * FULL_PSTRIDE)
    var merge_ptr = arena_alloc_all[MergeSegment](arenas, MAX_WORKERS * tp)

    var q = view.bind(q_ptr)
    var k_cache = view.bind(k_cache_ptr)
    var v_cache = view.bind(v_cache_ptr)
    var output = view.bind(output_ptr)
    var partials_scratch = view.bind(partials_ptr)
    var merge_segments = view.bind(merge_ptr)

    fill_pattern_all(
        q, FULL_MAX_SEQ * FULL_NUM_Q * FULL_HEAD_DIM)
    fill_pattern_all(k_cache, FULL_MAX_SEQ * FULL_KV_STRIDE)
    fill_pattern_all(v_cache, FULL_MAX_SEQ * FULL_KV_STRIDE)

    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    print(
        t"full: head_dim={FULL_HEAD_DIM} num_q={FULL_NUM_Q} "
        t"local_num_q={local_num_q} num_kv={FULL_NUM_KV} "
        t"gqa={FULL_GQA} max_seq={FULL_MAX_SEQ}"
    )

    var runs_table = KVRunTable()
    var run = KVRun(0, 0)
    var rows_per_page = FULL_PAGE_LEN // tp
    for ordinal in range(FULL_MAX_SEQ // FULL_PAGE_LEN):
        run.base_rows.append(Int32(ordinal * rows_per_page))
    runs_table.runs.append(run^)
    var runs = UnsafePointer(to=runs_table)

    section_validation_full(
        pools, runs, q, k_cache, v_cache, output, partials_scratch,
        merge_segments)
    section_full_sweep(
        pools, runs, q, k_cache, v_cache, output, partials_scratch,
        merge_segments)


def run_all[P: BurstThreadPool, //](
    var pools: List[P],
    mut arenas_sliding: List[NumaArena[alignment=ALIGNMENT]],
    mut arenas_full: List[NumaArena[alignment=ALIGNMENT]],
):
    var cap = pools[0].get_capacity()
    print(t"pool capacity: {cap} workers")
    run_sliding(pools, arenas_sliding)
    run_full(pools, arenas_full)


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("Flash-attention prefill benchmark")
    var iso = len(topo.isolated_cpus)
    print(t"{tp} NUMA node(s), {iso} isolated cpus\n")

    comptime ARENA_BYTES = 512 * 1024 * 1024
    var arenas_sliding = List[
        NumaArena[alignment=ALIGNMENT]](capacity=tp)
    var arenas_full = List[
        NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas_sliding.append(
            NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas_sliding[i]:
            print("arena alloc failed on node", topo[i])
            return
        arenas_full.append(
            NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas_full[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_flash_prefill_tp[
        P: BurstThreadPool, //,
    ](var selected_pools: List[P]):
        run_all(selected_pools^, arenas_sliding, arenas_full)

    with_topological_rank_dispatch[
        dispatch=dispatch_flash_prefill_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
