from std.collections import InlineArray
from std.memory import Span, UnsafePointer, memset_zero

from simd_math import pick_port_unroll, fast_exp_softmax_biased
from threading.threading_traits import BurstThreadPool
from .helpers import (
    BF16Ptr, F32Ptr, W,
    RangePartitionedKernel, WorkerRangePartitionedKernel,
    DispatchBuffer, tile_dispatch,
    fanout_dispatch, join_all, Binding,
)
from .dispatch_heuristics import (
    MERGE_INLINE_MAX_BYTES, MERGE_SATURATE_BYTES,
)
from .attention_ops import RunSplitBand
from .profiling import Profiler, DispatchSpan

comptime RunSplitBandPtr = UnsafePointer[RunSplitBand, MutUntrackedOrigin]


@fieldwise_init
struct MergeSegment(Copyable, ImplicitlyCopyable):
    var base: F32Ptr
    var stride: Int
    var n: Int


@always_inline
def merge_segments[
    head_dim: Int,
](
    segments: Span[MergeSegment, _],
    num_q: Int,
    h: Int,
    mut acc: InlineArray[SIMD[DType.float32, W], head_dim // W],
) -> Float32:
    var m_off = num_q * head_dim
    var l_off = m_off + num_q
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    var n_segments = len(segments)

    var global_m = Float32(-1e30)
    for seg_idx in range(n_segments):
        var seg = segments[seg_idx]
        for s in range(seg.n):
            var sm = (seg.base + s * seg.stride + m_off + h)[]
            if sm > global_m:
                global_m = sm

    var global_l = Float32(0)
    var first = True

    for seg_idx in range(n_segments):
        var seg = segments[seg_idx]
        var base = seg.base
        var stride = seg.stride
        var batch_start = 0
        while batch_start < seg.n:
            var batch_end = min(batch_start + W, seg.n)
            var batch_len = batch_end - batch_start
            var deltas = SIMD[DType.float32, W](-1e30)
            var batch_ls = SIMD[DType.float32, W](0)
            for b in range(batch_len):
                var sp = base + (batch_start + b) * stride
                deltas[b] = (sp + m_off + h)[] - global_m
                batch_ls[b] = (sp + l_off + h)[]
            var corrs = fast_exp_softmax_biased[W](
                max(SIMD[DType.float32, W](-87.0), deltas))
            corrs = batch_ls.gt(SIMD[DType.float32, W](0)).select(
                corrs, SIMD[DType.float32, W](0))
            global_l += (batch_ls * corrs).reduce_add()

            for b in range(batch_len):
                var c = corrs[b]
                if c <= 0:
                    continue
                var cv = SIMD[DType.float32, W](c)
                var src = base + (batch_start + b) * stride + h * head_dim
                if first:
                    for i in range(head_dim // STRIDE):
                        comptime for p in range(PU):
                            acc[i * PU + p] = (src + i * STRIDE + p * W).load[width=W]() * cv
                    first = False
                else:
                    for i in range(head_dim // STRIDE):
                        comptime for p in range(PU):
                            var v = (src + i * STRIDE + p * W).load[width=W]()
                            acc[i * PU + p] = v.fma(cv, acc[i * PU + p])
            batch_start += W

    return global_l


@always_inline
def write_finalized_head[
    head_dim: Int,
](
    dst: BF16Ptr,
    segments: Span[MergeSegment, _],
    num_q: Int,
    h: Int,
):
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    comptime LANES = head_dim // W

    var acc = InlineArray[SIMD[DType.float32, W], LANES](
        uninitialized=True)
    var global_l = merge_segments[head_dim](segments, num_q, h, acc)

    if global_l <= 0:
        for i in range(head_dim // STRIDE):
            comptime for p in range(PU):
                (dst + i * STRIDE + p * W).store(SIMD[DType.bfloat16, W](0))
        return

    var inv_l = SIMD[DType.float32, W](Float32(1.0) / global_l)
    for i in range(head_dim // STRIDE):
        comptime for p in range(PU):
            (dst + i * STRIDE + p * W).store(
                (acc[i * PU + p] * inv_l).cast[DType.bfloat16]())


@fieldwise_init
struct FinalizeKernel[head_dim: Int](
    RangePartitionedKernel
):
    var output: BF16Ptr
    var partials: F32Ptr
    var num_sources: Int
    var num_q: Int
    var partial_stride: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var segs = InlineArray[MergeSegment, 1](uninitialized=True)
        segs[0] = MergeSegment(
            self.partials, self.partial_stride, self.num_sources)
        var seg_span = Span(ptr=UnsafePointer(to=segs[0]), length=1)
        for h in range(self.start, self.end):
            write_finalized_head[Self.head_dim](
                self.output + h * Self.head_dim, seg_span, self.num_q, h)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct ContextFlashMergeConfig[
    head_dim: Int, o: ImmutOrigin, co: ImmutOrigin,
](TrivialRegisterPassable):
    var output: Binding[BFloat16, Self.o]
    var partials: Binding[Float32, Self.o]
    var num_sources: Span[Int, Self.co]


@fieldwise_init
struct ContextFinalizeKernel[
    head_dim: Int, o: ImmutOrigin, co: ImmutOrigin,
](WorkerRangePartitionedKernel):
    var config: ContextFlashMergeConfig[Self.head_dim, Self.o, Self.co]
    var q_rank: Int
    var segment_scratch: UnsafePointer[MergeSegment, MutUntrackedOrigin]
    var num_q: Int
    var local_num_q: Int
    var partial_stride: Int
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var tp = len(self.config.num_sources)
        var segs = self.segment_scratch + self.worker_id * tp
        for r in range(tp):
            segs[r] = MergeSegment(
                self.config.partials[r], self.partial_stride,
                self.config.num_sources[r])
        var seg_span = Span(ptr=segs, length=tp)
        for local_h in range(self.start, self.end):
            var global_h = self.q_rank * self.local_num_q + local_h
            var dst = self.config.output[self.q_rank] + local_h * Self.head_dim
            write_finalized_head[Self.head_dim](
                dst, seg_span, self.num_q, global_h)

    @always_inline
    def install_worker_range(
        mut self, worker_id: Int, start: Int, end: Int,
    ):
        self.worker_id = worker_id
        self.start = start
        self.end = end


@always_inline
def merge_workers(data_bytes: Int, capacity: Int) -> Int:
    if data_bytes >= MERGE_SATURATE_BYTES:
        return capacity
    return min(8, capacity)


def dispatch_merge_flash_partials[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    head_dim: Int,
    max_worker_count: Int = 128,
](
    output: Binding[BFloat16, o],
    partials_buf: Binding[Float32, o],
    num_sources: List[Int],
    num_q: Int,
    partial_stride: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
    inline_max_bytes: Int = MERGE_INLINE_MAX_BYTES,
):
    var tp = len(pools)
    comptime K = FinalizeKernel[head_dim]
    var buf = DispatchBuffer[K, max_worker_count]()
    var span = DispatchSpan[Profile]()
    var dispatched = False
    for r in range(tp):
        if num_sources[r] <= 0:
            memset_zero(output[r], num_q * head_dim)
            continue
        var data_bytes = num_sources[r] * (head_dim + 2) * 4 * num_q
        if data_bytes <= inline_max_bytes:
            var segs = InlineArray[MergeSegment, 1](uninitialized=True)
            segs[0] = MergeSegment(partials_buf[r], partial_stride, num_sources[r])
            var seg_span = Span(ptr=UnsafePointer(to=segs[0]), length=1)
            for h in range(num_q):
                write_finalized_head[head_dim](
                    output[r] + h * head_dim, seg_span, num_q, h)
            continue
        var nw = merge_workers(
            data_bytes, min(max_worker_count, pools[r].get_capacity()))
        _ = tile_dispatch(buf,
            K(output[r], partials_buf[r], num_sources[r], num_q, partial_stride, 0, 0),
            pools[r], num_q, num_workers=nw)
        dispatched = True
    if not dispatched:
        span.finish_inline(prof, "merge_flash_partials")
        return
    span.issued()
    join_all(pools)
    span.finish(prof, pools, "merge_flash_partials")


def dispatch_merge_context_flash_partials[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    head_dim: Int,
    max_worker_count: Int = 128,
](
    output: Binding[BFloat16, o],
    partials_buf: Binding[Float32, o],
    segment_scratch: Binding[MergeSegment, o],
    num_sources: List[Int],
    num_q: Int, local_num_q: Int, partial_stride: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var tp = len(pools)
    var total_sources = 0
    for r in range(tp):
        total_sources += num_sources[r]

    if total_sources <= 0:
        for r in range(tp):
            memset_zero(output[r], local_num_q * head_dim)
        return

    comptime co = origin_of(num_sources)
    var cfg = ContextFlashMergeConfig[head_dim, o, co](
        output, partials_buf, Span[Int, co](num_sources))
    comptime K = ContextFinalizeKernel[head_dim, o, co]
    var nq = num_q
    var lnq = local_num_q
    var ps = partial_stride

    @parameter
    def make(q_rank: Int) -> K:
        return K(cfg, q_rank, segment_scratch[q_rank], nq, lnq, ps, 0, 0, 0)

    fanout_dispatch[
        make,
        max_worker_count=max_worker_count,
        worker_policy=merge_workers,
        label="merge_context_flash_partials",
    ](pools, prof, local_num_q,
      total_sources * (head_dim + 2) * 4 * local_num_q)


@fieldwise_init
struct BatchedFinalizeKernel[head_dim: Int](RangePartitionedKernel):
    """Rank-local batched merge (sliding decode). Work units are (run, head)
    flattened as `run * num_q + head`; each run's flash partials occupy stripes
    `[band.split_base, band.split_base + band.n_splits)` and finalize into
    output token row `band.buf_start`. `bands` points at this rank's contiguous
    slice of the flat rank-major band table."""
    var output: BF16Ptr
    var partials: F32Ptr
    var bands: RunSplitBandPtr
    var num_runs: Int
    var num_q: Int
    var partial_stride: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var q_stride = self.num_q * Self.head_dim
        for flat in range(self.start, self.end):
            var i = flat // self.num_q
            var h = flat % self.num_q
            var band = self.bands[i]
            var segs = InlineArray[MergeSegment, 1](uninitialized=True)
            segs[0] = MergeSegment(
                self.partials + band.split_base * self.partial_stride,
                self.partial_stride, band.n_splits)
            var seg_span = Span(ptr=UnsafePointer(to=segs[0]), length=1)
            var dst = self.output + band.buf_start * q_stride + h * Self.head_dim
            write_finalized_head[Self.head_dim](dst, seg_span, self.num_q, h)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_merge_batched_flash_partials[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    head_dim: Int,
    max_worker_count: Int = 128,
](
    output: Binding[BFloat16, o],
    partials_buf: Binding[Float32, o],
    bands: RunSplitBandPtr,
    num_runs: Int,
    num_q: Int,
    partial_stride: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    """Sliding batched merge. `bands` is the flat rank-major table; rank r reads
    its slice `bands + r * num_runs`."""
    if num_runs <= 0:
        return
    comptime K = BatchedFinalizeKernel[head_dim]
    var bp = bands
    var nr = num_runs
    var nq = num_q
    var ps = partial_stride

    @parameter
    def make(r: Int) -> K:
        return K(output[r], partials_buf[r], bp + r * nr, nr, nq, ps, 0, 0)

    var total_units = num_runs * num_q
    fanout_dispatch[
        make,
        max_worker_count=max_worker_count,
        worker_policy=merge_workers,
        label="merge_batched_flash",
    ](pools, prof, total_units, total_units * (head_dim + 2) * 4)


@fieldwise_init
struct BatchedContextMergeConfig[
    head_dim: Int, o: ImmutOrigin,
](TrivialRegisterPassable):
    var output: Binding[BFloat16, Self.o]
    var partials: Binding[Float32, Self.o]
    var bands: RunSplitBandPtr
    var num_runs: Int


@fieldwise_init
struct BatchedContextFinalizeKernel[
    head_dim: Int, o: ImmutOrigin,
](WorkerRangePartitionedKernel):
    """Cross-rank batched merge (full decode). Work units are (run, local_head)
    flattened as `run * local_num_q + local_head`. For each run, every rank's
    partials are merged via the rank-major band table: rank r's stripes for run
    i start at `bands[r * num_runs + i].split_base`. The destination token row is
    `bands[q_rank * num_runs + i].buf_start` (rank-independent)."""
    var config: BatchedContextMergeConfig[Self.head_dim, Self.o]
    var q_rank: Int
    var segment_scratch: UnsafePointer[MergeSegment, MutUntrackedOrigin]
    var num_q: Int
    var local_num_q: Int
    var partial_stride: Int
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var tp = self.config.partials.degree()
        var out_stride = self.local_num_q * Self.head_dim
        var nr = self.config.num_runs
        var segs = self.segment_scratch + self.worker_id * tp
        var seg_span = Span(ptr=segs, length=tp)

        for flat in range(self.start, self.end):
            var i = flat // self.local_num_q
            var local_h = flat % self.local_num_q
            var global_h = self.q_rank * self.local_num_q + local_h

            for r in range(tp):
                var band = self.config.bands[r * nr + i]
                segs[r] = MergeSegment(
                    self.config.partials[r] + band.split_base * self.partial_stride,
                    self.partial_stride, band.n_splits)

            var buf_start = self.config.bands[self.q_rank * nr + i].buf_start
            var dst = self.config.output[self.q_rank] \
                      + buf_start * out_stride + local_h * Self.head_dim
            write_finalized_head[Self.head_dim](
                dst, seg_span, self.num_q, global_h)

    @always_inline
    def install_worker_range(
        mut self, worker_id: Int, start: Int, end: Int,
    ):
        self.worker_id = worker_id
        self.start = start
        self.end = end


def dispatch_merge_batched_context_partials[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    head_dim: Int,
    max_worker_count: Int = 128,
](
    output: Binding[BFloat16, o],
    partials_buf: Binding[Float32, o],
    segment_scratch: Binding[MergeSegment, o],
    bands: RunSplitBandPtr,
    num_runs: Int,
    num_q: Int, local_num_q: Int, partial_stride: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    """Full batched merge. `bands` is the flat rank-major table of length
    `tp * num_runs`; every merge worker reads all ranks' bands for its run."""
    if num_runs <= 0:
        return
    var cfg = BatchedContextMergeConfig[head_dim, o](
        output, partials_buf, bands, num_runs)
    comptime K = BatchedContextFinalizeKernel[head_dim, o]
    var nq = num_q
    var lnq = local_num_q
    var ps = partial_stride

    @parameter
    def make(q_rank: Int) -> K:
        return K(cfg, q_rank, segment_scratch[q_rank], nq, lnq, ps, 0, 0, 0)

    var total_units = num_runs * local_num_q
    fanout_dispatch[
        make,
        max_worker_count=max_worker_count,
        worker_policy=merge_workers,
        label="merge_batched_context",
    ](pools, prof, total_units,
      total_units * (head_dim + 2) * 4 * len(pools))
