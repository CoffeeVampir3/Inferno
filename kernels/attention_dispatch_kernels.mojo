from std.memory import Span, UnsafePointer

from threading.threading_traits import BurstThreadPool
from .helpers import (
    Binding, OutputPartitionedKernel, fanout_dispatch, fanout_dispatch_per_rank,
    DispatchBuffer, worker_range, join_all, min_pool_capacity,
)
from .attention_ops import (
    KVRunTable, PagedKV, RunSplitBand, full_local_kv_count, plan_run_splits,
    pow2_shift, TILE,
)
from .flash_attention import FlashAttentionKernel
from .flash_attention_prefill import (
    FlashPrefillSlidingKernel, FlashPrefillFullKernel,
    dispatch_merge_flash_prefill_partials,
)
from .logsum_merge import (
    MergeSegment, dispatch_merge_flash_partials,
    dispatch_merge_context_flash_partials,
    dispatch_merge_batched_flash_partials,
    dispatch_merge_batched_context_partials,
)
from .profiling import Profiler, DispatchSpan


comptime MIN_DECODE_SPLIT = TILE


@always_inline
def sliding_valid_len(pos: Int, window: Int) -> Int:
    if pos + 1 >= window:
        return window
    return pos + 1


def dispatch_flash_sliding[
    P: BurstThreadPool, DecodeK: OutputPartitionedKernel,
    PrefillK: OutputPartitionedKernel, Profile: Bool, N: Int, o: ImmutOrigin, //,
    head_dim: Int, window: Int, elt_bytes: Int,
    make_decode: def(Int, Int) capturing [_] -> DecodeK,
    make_prefill: def(Int) capturing [_] -> PrefillK,
    make_decode_run: def(Int, Int, Int) capturing [_] -> DecodeK,
    decode_label: StaticString, prefill_label: StaticString,
    batched_label: StaticString,
    max_worker_count: Int = 128,
](
    output: Binding[BFloat16, o],
    partials: Binding[Float32, o],
    runs: UnsafePointer[KVRunTable, MutUntrackedOrigin],
    num_q: Int,
    partial_stride: Int,
    kv_stride: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if seq_len <= 0:
        return
    if seq_len == 1:
        debug_assert(
            len(runs[].runs) == 1, "decode dispatch expects a single run")
        var base_pos = Int(runs[].runs[0].base_pos)
        var valid_len = sliding_valid_len(base_pos, window)
        if valid_len <= 0:
            return
        var start_pos = base_pos - valid_len + 1

        @parameter
        def make(r: Int) -> DecodeK:
            return make_decode(r, start_pos)

        @parameter
        def total_for(r: Int) -> Int:
            return valid_len

        @parameter
        def bytes_for(r: Int) -> Int:
            return valid_len * kv_stride * elt_bytes

        var nws = fanout_dispatch_per_rank[
            make, total_for, bytes_for,
            max_worker_count=max_worker_count,
            label=decode_label,
        ](pools, prof)

        dispatch_merge_flash_partials[
            head_dim,
            max_worker_count=max_worker_count,
        ](output, partials, nws, num_q, partial_stride, pools, prof)
        return

    var num_runs = len(runs[].runs)
    if num_runs == seq_len and num_runs > 1:
        # Pure-decode batch: every run is one query token. Sliding valid_len is
        # rank-independent, but bands are planned per rank to respect each pool's
        # worker budget. Splitting only pays off when every rank can give each
        # run its own split, so gate on the smallest pool's budget.
        var tp = len(pools)
        var split_budget = min_pool_capacity(pools, max_worker_count)
        if num_runs < split_budget:
            var kv_lens = List[Int](capacity=num_runs)
            var buf_starts = List[Int](capacity=num_runs)
            for i in range(num_runs):
                var p = Int(runs[].runs[i].base_pos)
                kv_lens.append(sliding_valid_len(p, window))
                buf_starts.append(Int(runs[].runs[i].buf_start))

            var flat_bands = List[RunSplitBand](capacity=tp * num_runs)
            for r in range(tp):
                var cap = min(max_worker_count, pools[r].get_capacity())
                var bands_r = plan_run_splits(
                    Span(kv_lens), Span(buf_starts), cap, MIN_DECODE_SPLIT)
                for i in range(num_runs):
                    flat_bands.append(bands_r[i])

            var span = DispatchSpan[Profile]()
            for r in range(tp):
                var buf = DispatchBuffer[DecodeK, max_worker_count]()
                for i in range(num_runs):
                    var band = flat_bands[r * num_runs + i]
                    if band.n_splits == 0:
                        continue
                    var start_pos = (
                        Int(runs[].runs[i].base_pos) - kv_lens[i] + 1)
                    var proto = make_decode_run(r, i, start_pos)
                    for s in range(band.n_splits):
                        var wr = worker_range(kv_lens[i], band.n_splits, s)
                        var item = proto
                        item.set_partition(band.split_base + s, wr[0], wr[1])
                        buf.slot()[] = item
                buf.dispatch(pools[r])
            span.issued()
            join_all(pools)
            span.finish(prof, pools, batched_label)

            dispatch_merge_batched_flash_partials[
                head_dim, max_worker_count=max_worker_count,
            ](output, partials, flat_bands.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
              num_runs, num_q, partial_stride, pools, prof)
            _ = flat_bands^
            return

        # Token count saturates the worker budget: the token-partitioned prefill
        # kernels already cover length-1 runs; only the byte estimate differs.
        var db = 0
        for i in range(num_runs):
            db += sliding_valid_len(Int(runs[].runs[i].base_pos), window)
        db *= kv_stride * elt_bytes
        fanout_dispatch[
            make_prefill, max_worker_count=max_worker_count,
            label=prefill_label,
        ](pools, prof, seq_len, db)
        return

    var per_q_kv = window if seq_len > window else seq_len
    var data_bytes = seq_len * per_q_kv * kv_stride * elt_bytes
    fanout_dispatch[
        make_prefill, max_worker_count=max_worker_count,
        label=prefill_label,
    ](pools, prof, seq_len, data_bytes)


def dispatch_flash_full[
    P: BurstThreadPool, DecodeK: OutputPartitionedKernel,
    PrefillK: OutputPartitionedKernel, Profile: Bool, N: Int, o: ImmutOrigin, //,
    head_dim: Int, kv_stride: Int, elt_bytes: Int,
    make_decode: def(Int) capturing [_] -> DecodeK,
    make_prefill: def(Int) capturing [_] -> PrefillK,
    make_decode_run: def(Int, Int) capturing [_] -> DecodeK,
    decode_label: StaticString, prefill_label: StaticString,
    batched_label: StaticString,
    max_worker_count: Int = 128,
](
    q_local_output: Binding[BFloat16, o],
    partials: Binding[Float32, o],
    segment_scratch: Binding[MergeSegment, o],
    runs: UnsafePointer[KVRunTable, MutUntrackedOrigin],
    num_q: Int,
    local_num_q: Int,
    partial_stride: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if seq_len <= 0:
        return
    var tp = len(pools)
    if seq_len == 1:
        debug_assert(
            len(runs[].runs) == 1, "decode dispatch expects a single run")
        var base_pos = Int(runs[].runs[0].base_pos)
        var valid_lens = List[Int]()
        for rank in range(tp):
            valid_lens.append(full_local_kv_count(rank, base_pos, tp))

        @parameter
        def total_for(r: Int) -> Int:
            return valid_lens[r]

        @parameter
        def bytes_for(r: Int) -> Int:
            return valid_lens[r] * kv_stride * elt_bytes

        var nws = fanout_dispatch_per_rank[
            make_decode, total_for, bytes_for,
            max_worker_count=max_worker_count,
            label=decode_label,
        ](pools, prof)

        dispatch_merge_context_flash_partials[
            head_dim,
            max_worker_count=max_worker_count,
        ](q_local_output, partials, segment_scratch, nws,
          num_q, local_num_q, partial_stride, pools, prof)
        return

    var num_runs = len(runs[].runs)
    if num_runs == seq_len and num_runs > 1:
        # Pure-decode batch. Context is round-robin sharded by position, so the
        # KV a rank owns for a run differs per rank: bands are built per rank and
        # stored rank-major for the cross-rank merge. Gate on the smallest pool's
        # budget so every rank can give each run at least one split.
        var split_budget = min_pool_capacity(pools, max_worker_count)
        if num_runs < split_budget:
            var buf_starts = List[Int](capacity=num_runs)
            for i in range(num_runs):
                buf_starts.append(Int(runs[].runs[i].buf_start))

            var flat_kv = List[Int](capacity=tp * num_runs)
            var flat_bands = List[RunSplitBand](capacity=tp * num_runs)
            for r in range(tp):
                var kv_lens = List[Int](capacity=num_runs)
                for i in range(num_runs):
                    kv_lens.append(full_local_kv_count(
                        r, Int(runs[].runs[i].base_pos), tp))
                var cap = min(max_worker_count, pools[r].get_capacity())
                var bands_r = plan_run_splits(
                    Span(kv_lens), Span(buf_starts), cap, MIN_DECODE_SPLIT)
                for i in range(num_runs):
                    flat_kv.append(kv_lens[i])
                    flat_bands.append(bands_r[i])

            var span = DispatchSpan[Profile]()
            for r in range(tp):
                var buf = DispatchBuffer[DecodeK, max_worker_count]()
                for i in range(num_runs):
                    var band = flat_bands[r * num_runs + i]
                    if band.n_splits == 0:
                        continue
                    var kv_len = flat_kv[r * num_runs + i]
                    var proto = make_decode_run(r, i)
                    for s in range(band.n_splits):
                        var wr = worker_range(kv_len, band.n_splits, s)
                        var item = proto
                        item.set_partition(band.split_base + s, wr[0], wr[1])
                        buf.slot()[] = item
                buf.dispatch(pools[r])
            span.issued()
            join_all(pools)
            span.finish(prof, pools, batched_label)

            dispatch_merge_batched_context_partials[
                head_dim, max_worker_count=max_worker_count,
            ](q_local_output, partials, segment_scratch,
              flat_bands.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), num_runs, num_q,
              local_num_q, partial_stride, pools, prof)
            _ = flat_bands^
            return

        # Token count saturates the worker budget: fall back to the
        # token-partitioned prefill kernels with a corrected byte estimate.
        var ctx_sum = 0
        for i in range(num_runs):
            ctx_sum += Int(runs[].runs[i].base_pos) + 1
        var db = (ctx_sum // tp + 1) * kv_stride * elt_bytes
        fanout_dispatch[
            make_prefill, max_worker_count=max_worker_count,
            label=prefill_label,
        ](pools, prof, seq_len, db)
        dispatch_merge_flash_prefill_partials[
            head_dim,
            max_worker_count=max_worker_count,
        ](q_local_output, partials, segment_scratch,
          num_q, local_num_q, partial_stride, seq_len, pools, prof)
        return

    var first_pos = Int(runs[].runs[0].base_pos)
    var avg_local_kv = (first_pos + seq_len // 2) // tp + 1
    var data_bytes = seq_len * avg_local_kv * kv_stride * elt_bytes
    fanout_dispatch[
        make_prefill, max_worker_count=max_worker_count,
        label=prefill_label,
    ](pools, prof, seq_len, data_bytes)

    dispatch_merge_flash_prefill_partials[
        head_dim,
        max_worker_count=max_worker_count,
    ](q_local_output, partials, segment_scratch,
      num_q, local_num_q, partial_stride, seq_len, pools, prof)


def dispatch_sliding_attention[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    head_dim: Int, max_q: Int, gqa_ratio: Int,
    window: Int, cache_size: Int, page_len: Int,
    max_worker_count: Int = 128,
](
    q: Binding[BFloat16, o],
    k_base: Binding[BFloat16, o],
    v_base: Binding[BFloat16, o],
    output: Binding[BFloat16, o],
    partials: Binding[Float32, o],
    runs: UnsafePointer[KVRunTable, MutUntrackedOrigin],
    num_q: Int,
    partial_stride: Int,
    kv_stride: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime DecodeK = FlashAttentionKernel[
        PagedKV, head_dim, max_q, gqa_ratio,
    ]
    comptime PrefillK = FlashPrefillSlidingKernel[
        head_dim, max_q, gqa_ratio, window, cache_size, page_len,
    ]
    comptime page_shift = pow2_shift(page_len)
    comptime row_mask = page_len - 1
    comptime page_mask = cache_size // page_len - 1
    var nq = num_q
    var ps = partial_stride
    var ks = kv_stride

    @parameter
    def make_decode(r: Int, start_pos: Int) -> DecodeK:
        var kv = PagedKV(
            runs[].row_ptr(0),
            page_shift, row_mask, page_mask)
        return DecodeK(kv, q[r], k_base[r], v_base[r], partials[r],
                       nq, ps, ks, 0, start_pos, 0, 0)

    @parameter
    def make_prefill(r: Int) -> PrefillK:
        return PrefillK(runs, q[r], k_base[r], v_base[r], output[r],
                        partials[r], nq, ps, ks, 0, 0, 0)

    @parameter
    def make_decode_run(r: Int, run_idx: Int, start_pos: Int) -> DecodeK:
        var kv = PagedKV(
            runs[].row_ptr(run_idx),
            page_shift, row_mask, page_mask)
        var q_off = Int(runs[].runs[run_idx].buf_start) * nq * head_dim
        return DecodeK(kv, q[r] + q_off, k_base[r], v_base[r], partials[r],
                       nq, ps, ks, 0, start_pos, 0, 0)

    dispatch_flash_sliding[
        head_dim, window, 2,
        make_decode, make_prefill, make_decode_run,
        "sliding_attn.flash", "sliding_attn.prefill",
        "sliding_attn.flash_batched",
        max_worker_count=max_worker_count,
    ](output, partials, runs, num_q, partial_stride, kv_stride, seq_len,
      pools, prof)


def dispatch_full_attention[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    head_dim: Int, num_q: Int, gqa_ratio: Int,
    kv_stride: Int, partial_stride: Int, page_len: Int,
    max_worker_count: Int = 128,
](
    q: Binding[BFloat16, o],
    k_base: Binding[BFloat16, o],
    v_base: Binding[BFloat16, o],
    q_local_output: Binding[BFloat16, o],
    partials: Binding[Float32, o],
    segment_scratch: Binding[MergeSegment, o],
    runs: UnsafePointer[KVRunTable, MutUntrackedOrigin],
    local_num_q: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    """`q_local_output` is the per-rank merged attention output, feeding the
    column-sharded o_proj. Q heads are replicated (num_q comptime); only the
    context shard `degree` is runtime."""
    comptime DecodeK = FlashAttentionKernel[
        PagedKV, head_dim, num_q, gqa_ratio,
    ]
    comptime PrefillK = FlashPrefillFullKernel[
        head_dim, num_q, gqa_ratio, partial_stride,
    ]
    var degree = len(pools)
    var rows_per_page = page_len // degree
    var page_shift = pow2_shift(rows_per_page)
    var row_mask = rows_per_page - 1

    @parameter
    def make_decode(r: Int) -> DecodeK:
        var kv = PagedKV(
            runs[].row_ptr(0), page_shift, row_mask, -1)
        return DecodeK(kv, q[r], k_base[r], v_base[r], partials[r],
                       num_q, partial_stride, kv_stride, 0, 0, 0, 0)

    @parameter
    def make_prefill(r: Int) -> PrefillK:
        return PrefillK(runs, q[r], k_base[r], v_base[r], partials[r],
                        kv_stride, degree, r, page_shift, row_mask, 0, 0)

    @parameter
    def make_decode_run(r: Int, run_idx: Int) -> DecodeK:
        var kv = PagedKV(
            runs[].row_ptr(run_idx),
            page_shift, row_mask, -1)
        var q_off = Int(runs[].runs[run_idx].buf_start) * num_q * head_dim
        return DecodeK(kv, q[r] + q_off, k_base[r], v_base[r], partials[r],
                       num_q, partial_stride, kv_stride, 0, 0, 0, 0)

    dispatch_flash_full[
        head_dim, kv_stride, 2,
        make_decode, make_prefill, make_decode_run,
        "full_attn.flash", "full_attn.prefill",
        "full_attn.flash_batched",
        max_worker_count=max_worker_count,
    ](q_local_output, partials, segment_scratch, runs,
      num_q, local_num_q, partial_stride, seq_len, pools, prof)
