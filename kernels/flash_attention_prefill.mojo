from std.collections import InlineArray
from std.memory import Span, UnsafePointer

from threading.threading_traits import BurstThreadPool
from .helpers import (
    BF16Ptr, F32Ptr, W,
    RangePartitionedKernel, WorkerRangePartitionedKernel,
    fanout_dispatch, Binding,
)
from .attention_ops import (
    KVRunTable, PagedKV, TILE, full_local_kv_count, pow2_shift,
    process_kv_tile, zero_accumulators,
)
from .logsum_merge import MergeSegment, write_finalized_head
from .profiling import Profiler


@fieldwise_init
struct FlashPrefillSlidingKernel[
    head_dim: Int, max_q: Int, gqa_ratio: Int,
    window: Int, cache_size: Int, page_len: Int,
](WorkerRangePartitionedKernel):
    """`page_len` and `cache_size / page_len` must be powers of two (slot
    resolution is shift/mask through the run's page ring). The per-rank head
    count `num_q` and `kv_stride` are runtime; per-head storage sizes to the
    comptime `max_q` cap (= model NUM_HEADS)."""

    var runs: UnsafePointer[KVRunTable, MutUntrackedOrigin]
    var q: BF16Ptr
    var k_base: BF16Ptr
    var v_base: BF16Ptr
    var output: BF16Ptr
    var partials: F32Ptr
    var num_q: Int
    var partial_stride: Int
    var kv_stride: Int
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var q_stride = self.num_q * Self.head_dim
        comptime page_shift = pow2_shift(Self.page_len)
        comptime row_mask = Self.page_len - 1
        comptime page_mask = Self.cache_size // Self.page_len - 1

        var scratch = self.partials + self.worker_id * self.partial_stride
        var acc_ptrs = InlineArray[F32Ptr, Self.max_q](uninitialized=True)
        var q_ptrs = InlineArray[BF16Ptr, Self.max_q](uninitialized=True)
        for h in range(self.num_q):
            acc_ptrs[h] = scratch + h * Self.head_dim

        ref run_list = self.runs[].runs
        var num_runs = len(run_list)
        var r = 0
        var kv = PagedKV(
            self.runs[].row_ptr(0), page_shift, row_mask, page_mask)
        var run_start = Int(run_list[0].buf_start)
        var run_pos = Int(run_list[0].base_pos)

        for t in range(self.start, self.end):
            while r + 1 < num_runs and t >= Int(run_list[r + 1].buf_start):
                r += 1
                kv = PagedKV(
                    self.runs[].row_ptr(r),
                    page_shift, row_mask, page_mask)
                run_start = Int(run_list[r].buf_start)
                run_pos = Int(run_list[r].base_pos)
            var abs_pos = run_pos + (t - run_start)
            var lo = max(0, abs_pos - Self.window + 1)
            var hi = abs_pos + 1

            var q_tok = self.q + t * q_stride
            var out_tok = self.output + t * q_stride

            var m = InlineArray[Float32, Self.max_q](fill=Float32(-1e30))
            var l = InlineArray[Float32, Self.max_q](fill=Float32(0))
            for h in range(self.num_q):
                q_ptrs[h] = q_tok + h * Self.head_dim

            zero_accumulators[Self.max_q, Self.head_dim](acc_ptrs, self.num_q)

            var pos = lo
            while pos < hi:
                var tile_len = min(TILE, hi - pos)
                process_kv_tile[
                    Self.head_dim, Self.gqa_ratio,
                ](kv, q_ptrs,
                  self.k_base,
                  self.v_base,
                  0, pos, tile_len, m, l, acc_ptrs,
                  self.num_q, self.kv_stride)
                pos += TILE

            for h in range(self.num_q):
                if l[h] > 0:
                    var inv_l = SIMD[DType.float32, W](
                        Float32(1.0) / l[h])
                    for j in range(0, Self.head_dim, W):
                        var v = (acc_ptrs[h] + j).load[width=W]() * inv_l
                        (out_tok + h * Self.head_dim + j).store(
                            v.cast[DType.bfloat16]())
                else:
                    for j in range(0, Self.head_dim, W):
                        (out_tok + h * Self.head_dim + j).store(
                            SIMD[DType.bfloat16, W](0))

    @always_inline
    def install_worker_range(
        mut self, worker_id: Int, start: Int, end: Int,
    ):
        self.worker_id = worker_id
        self.start = start
        self.end = end


@fieldwise_init
struct FlashPrefillFullKernel[
    head_dim: Int, num_q: Int, gqa_ratio: Int,
    partial_stride: Int,
](RangePartitionedKernel):
    """Full-attention prefill. Q heads are replicated (num_q comptime); the
    context-shard `degree`, `kv_stride`, and page geometry are runtime. Writes
    (acc, m, l) partials for the cross-rank logsum merge."""

    var runs: UnsafePointer[KVRunTable, MutUntrackedOrigin]
    var q: BF16Ptr
    var k_base: BF16Ptr
    var v_base: BF16Ptr
    var partials: F32Ptr
    var kv_stride: Int
    var degree: Int
    var rank: Int
    var page_shift: Int
    var row_mask: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime q_stride = Self.num_q * Self.head_dim
        comptime m_off = Self.num_q * Self.head_dim
        comptime l_off = m_off + Self.num_q

        var acc_ptrs = InlineArray[F32Ptr, Self.num_q](uninitialized=True)
        var q_ptrs = InlineArray[BF16Ptr, Self.num_q](uninitialized=True)

        ref run_list = self.runs[].runs
        var num_runs = len(run_list)
        var r = 0
        var kv = PagedKV(
            self.runs[].row_ptr(0),
            self.page_shift, self.row_mask, -1)
        var run_start = Int(run_list[0].buf_start)
        var run_pos = Int(run_list[0].base_pos)

        for t in range(self.start, self.end):
            while r + 1 < num_runs and t >= Int(run_list[r + 1].buf_start):
                r += 1
                kv = PagedKV(
                    self.runs[].row_ptr(r),
                    self.page_shift, self.row_mask, -1)
                run_start = Int(run_list[r].buf_start)
                run_pos = Int(run_list[r].base_pos)
            var abs_pos = run_pos + (t - run_start)
            var local_kv_count = full_local_kv_count(
                self.rank, abs_pos, self.degree)

            var partial_tok = self.partials + t * Self.partial_stride
            var q_tok = self.q + t * q_stride

            var m = InlineArray[Float32, Self.num_q](fill=Float32(-1e30))
            var l = InlineArray[Float32, Self.num_q](fill=Float32(0))

            comptime for h in range(Self.num_q):
                acc_ptrs[h] = partial_tok + h * Self.head_dim
                q_ptrs[h] = q_tok + h * Self.head_dim

            zero_accumulators[Self.num_q, Self.head_dim](acc_ptrs, Self.num_q)

            var pos = 0
            while pos < local_kv_count:
                var tile_len = min(TILE, local_kv_count - pos)
                process_kv_tile[
                    Self.head_dim, Self.gqa_ratio,
                ](kv, q_ptrs,
                  self.k_base,
                  self.v_base,
                  0, pos, tile_len, m, l, acc_ptrs, Self.num_q, self.kv_stride)
                pos += TILE

            comptime for h in range(Self.num_q):
                (partial_tok + m_off + h)[] = m[h]
                (partial_tok + l_off + h)[] = l[h]

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct PrefillMergeConfig[head_dim: Int, o: ImmutOrigin](TrivialRegisterPassable):
    var output: Binding[BFloat16, Self.o]
    var partials: Binding[Float32, Self.o]


@fieldwise_init
struct PrefillMergeKernel[
    head_dim: Int, o: ImmutOrigin,
](WorkerRangePartitionedKernel):
    var config: PrefillMergeConfig[Self.head_dim, Self.o]
    var q_rank: Int
    var segment_scratch: UnsafePointer[MergeSegment, MutUntrackedOrigin]
    var num_q: Int
    var local_num_q: Int
    var partial_stride: Int
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var out_stride = self.local_num_q * Self.head_dim
        var tp = self.config.partials.degree()
        var segs = self.segment_scratch + self.worker_id * tp
        var seg_span = Span(ptr=segs, length=tp)

        for flat in range(self.start, self.end):
            var t = flat // self.local_num_q
            var local_h = flat % self.local_num_q
            var global_h = self.q_rank * self.local_num_q + local_h

            for r in range(tp):
                segs[r] = MergeSegment(
                    self.config.partials[r] + t * self.partial_stride,
                    self.partial_stride, 1)

            var dst = self.config.output[self.q_rank] \
                      + t * out_stride + local_h * Self.head_dim
            write_finalized_head[Self.head_dim](
                dst, seg_span, self.num_q, global_h)

    @always_inline
    def install_worker_range(
        mut self, worker_id: Int, start: Int, end: Int,
    ):
        self.worker_id = worker_id
        self.start = start
        self.end = end


def dispatch_merge_flash_prefill_partials[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    head_dim: Int,
    max_worker_count: Int = 128,
](
    output: Binding[BFloat16, o],
    partials: Binding[Float32, o],
    segment_scratch: Binding[MergeSegment, o],
    num_q: Int, local_num_q: Int, partial_stride: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if seq_len <= 0:
        return

    var cfg = PrefillMergeConfig[head_dim, o](output, partials)
    comptime K = PrefillMergeKernel[head_dim, o]
    var nq = num_q
    var lnq = local_num_q
    var ps = partial_stride

    @parameter
    def make(q_rank: Int) -> K:
        return K(
            cfg, q_rank,
            segment_scratch[q_rank],
            nq, lnq, ps, 0, 0, 0)

    var total_units = seq_len * local_num_q
    var data_bytes = total_units * len(pools) * (head_dim + 2) * 4

    fanout_dispatch[
        make, max_worker_count=max_worker_count,
        label="merge_flash_prefill_partials",
    ](pools, prof, total_units, data_bytes)
