from std.collections import InlineArray
from std.math import iota
from std.memory import Span, UnsafePointer

from simd_math import pick_port_unroll, reduce_top_k
from simd_math.ops import sqrt
from threading.threading_traits import BurstThreadPool

from kernels.helpers import (
    Binding, RangePartitionedKernel, BF16Ptr, F32Ptr, I32Ptr, BW, W,
    fanout_dispatch, tile_dispatch, DispatchBuffer, join_all,
)
from kernels.attention_ops import (
    KVRunTable, PagedKV, full_local_kv_count, pow2_shift,
)
from kernels.dot_products import bf16_panel_dot_to_scalars
from kernels.profiling import Profiler, DispatchSpan


def bank_cap(topk: Int, width: Int) -> Int:
    var cap = 1
    while cap < 2 * topk:
        cap <<= 1
    if cap < width:
        cap = width
    return cap


@always_inline
def bank_reduce[
    width: Int, regs: Int, cap: Int, topk: Int,
](
    read bank_v: InlineArray[SIMD[DType.float32, width], regs],
    read bank_b: InlineArray[Int32, cap],
    read tags: InlineArray[SIMD[DType.int32, width], regs],
    sentinel: Float32,
    mut keep_b: InlineArray[Int32, topk],
    mut keep_v: InlineArray[Float32, topk],
):
    var lanes = InlineArray[Int, topk](fill=0)
    reduce_top_k[DType.float32, width, regs, topk](
        bank_v, tags, sentinel, lanes, keep_v)
    comptime for s in range(topk):
        keep_b[s] = bank_b[lanes[s]]


@always_inline
def stream_select_blocks[
    block_size: Int, topk_blocks: Int, local_blocks: Int, init_blocks: Int,
    score_at: def(Int) capturing [_] -> Float32,
](
    num_blocks: Int,
    out_row: I32Ptr,
):
    comptime NEG = Float32(-1.0e30)
    comptime FORCE = Float32(1.0e30)
    comptime CAP = bank_cap(topk_blocks, W)
    comptime REGS = CAP // W

    var tags = InlineArray[SIMD[DType.int32, W], REGS](uninitialized=True)
    comptime for r in range(REGS):
        tags[r] = iota[DType.int32, W]() + SIMD[DType.int32, W](Int32(r * W))

    var local_start = num_blocks - local_blocks
    if local_start < 0:
        local_start = 0

    var bank_v = InlineArray[SIMD[DType.float32, W], REGS](
        fill=SIMD[DType.float32, W](NEG))
    var bank_b = InlineArray[Int32, CAP](fill=Int32(-1))
    var fill = 0

    for b in range(num_blocks):
        var score: Float32
        if b >= local_start or b < init_blocks:
            score = FORCE
        else:
            score = score_at(b)

        var r = fill // W
        var l = fill - r * W
        var v = bank_v[r]
        v[l] = score
        bank_v[r] = v
        bank_b[fill] = Int32(b)
        fill += 1

        if fill == CAP:
            var keep_b = InlineArray[Int32, topk_blocks](fill=Int32(-1))
            var keep_v = InlineArray[Float32, topk_blocks](fill=NEG)
            bank_reduce[W, REGS, CAP, topk_blocks](
                bank_v, bank_b, tags, NEG, keep_b, keep_v)
            comptime for rr in range(REGS):
                bank_v[rr] = SIMD[DType.float32, W](NEG)
            for i in range(CAP):
                bank_b[i] = Int32(-1)
            for s in range(topk_blocks):
                var sr = s // W
                var sl = s - sr * W
                var sv = bank_v[sr]
                sv[sl] = keep_v[s]
                bank_v[sr] = sv
                bank_b[s] = keep_b[s]
            fill = topk_blocks

    var keep_b = InlineArray[Int32, topk_blocks](fill=Int32(-1))
    var keep_v = InlineArray[Float32, topk_blocks](fill=NEG)
    bank_reduce[W, REGS, CAP, topk_blocks](
        bank_v, bank_b, tags, NEG, keep_b, keep_v)

    var ids = InlineArray[Int32, topk_blocks](fill=Int32(-1))
    var n = 0
    for s in range(topk_blocks):
        if keep_v[s] > NEG:
            ids[n] = keep_b[s]
            n += 1
    for i in range(1, n):
        var v = ids[i]
        var j = i - 1
        while j >= 0 and ids[j] > v:
            ids[j + 1] = ids[j]
            j -= 1
        ids[j + 1] = v
    for k in range(topk_blocks):
        out_row[k] = ids[k]


@fieldwise_init
struct IndexBlockScoreKernel[
    index_head_dim: Int, num_index_heads: Int, block_size: Int,
](RangePartitionedKernel):
    var runs: UnsafePointer[KVRunTable, MutAnyOrigin]
    var index_q: BF16Ptr
    var k_base: BF16Ptr
    var partial: F32Ptr
    var block_stride: Int
    var kv_stride: Int
    var scale: Float32
    var degree: Int
    var rank: Int
    var page_shift: Int
    var row_mask: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime q_row = Self.num_index_heads * Self.index_head_dim
        comptime PU = pick_port_unroll[BW, Self.index_head_dim]()
        comptime NEG = Float32(-1.0e30)

        ref run_list = self.runs[].runs
        var num_runs = len(run_list)
        var ri = 0
        var kv = PagedKV(
            self.runs[].row_ptr(0), self.page_shift, self.row_mask, -1)
        var run_start = Int(run_list[0].buf_start)
        var run_pos = Int(run_list[0].base_pos)
        var bstride = self.block_stride
        var tstride = Self.num_index_heads * bstride

        for t in range(self.start, self.end):
            while ri + 1 < num_runs and t >= Int(run_list[ri + 1].buf_start):
                ri += 1
                kv = PagedKV(
                    self.runs[].row_ptr(ri), self.page_shift, self.row_mask, -1)
                run_start = Int(run_list[ri].buf_start)
                run_pos = Int(run_list[ri].base_pos)
            var abs_pos = run_pos + (t - run_start)
            if abs_pos < 0:
                continue
            var num_blocks = abs_pos // Self.block_size + 1
            var local_kv = full_local_kv_count(self.rank, abs_pos, self.degree)

            var tok_base = self.partial + t * tstride
            comptime for h in range(Self.num_index_heads):
                var prow = tok_base + h * bstride
                var bz = 0
                while bz + W <= num_blocks:
                    (prow + bz).store(SIMD[DType.float32, W](NEG))
                    bz += W
                while bz < num_blocks:
                    prow[bz] = NEG
                    bz += 1

            var q_heads = InlineArray[BF16Ptr, Self.num_index_heads](
                uninitialized=True)
            var q_tok = self.index_q + t * q_row
            comptime for h in range(Self.num_index_heads):
                q_heads[h] = q_tok + h * Self.index_head_dim

            for pos in range(local_kv):
                var row = kv.slot(0, pos)
                var k_ptr = self.k_base + row * self.kv_stride
                var dots = bf16_panel_dot_to_scalars[
                    cols = Self.index_head_dim, port_unroll=PU,
                ](k_ptr, q_heads)
                var g = pos * self.degree + self.rank
                var b = g // Self.block_size
                comptime for h in range(Self.num_index_heads):
                    var sh = dots[h] * self.scale
                    var cell = tok_base + h * bstride + b
                    if sh > cell[]:
                        cell[] = sh

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_index_block_scores[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    index_head_dim: Int, num_index_heads: Int, block_size: Int,
    page_len: Int, max_worker_count: Int = 128,
](
    index_q: Binding[BFloat16, o],
    index_k: Binding[BFloat16, o],
    partial: Binding[Float32, o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    block_stride: Int,
    scale: Float32,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var degree = len(pools)
    comptime K = IndexBlockScoreKernel[
        index_head_dim, num_index_heads, block_size,
    ]
    comptime kv_stride = index_head_dim
    var rows_per_page = page_len // degree
    var page_shift = pow2_shift(rows_per_page)
    var row_mask = rows_per_page - 1
    var bstride = block_stride
    var sc = scale
    var deg = degree

    @parameter
    def make(r: Int) -> K:
        return K(runs, index_q[r], index_k[r], partial[r],
                 bstride, kv_stride, sc, deg, r, page_shift, row_mask, 0, 0)

    var base_pos = Int(runs[].runs[0].base_pos)
    var local_keys = (base_pos + seq_len) // max(1, degree) + 1
    var data_bytes = seq_len * local_keys * kv_stride * 2
    fanout_dispatch[
        make, max_worker_count=max_worker_count,
        label="lightning_indexer.score",
    ](pools, prof, seq_len, data_bytes)


@fieldwise_init
struct IndexTopkMergeKernel[
    block_size: Int, num_index_heads: Int, topk_blocks: Int,
    local_blocks: Int, init_blocks: Int,
    o: ImmutOrigin,
](RangePartitionedKernel):
    var partials: Binding[Float32, Self.o]
    var block_idx: Binding[Int32, Self.o]
    var out_rank: Int
    var block_stride: Int
    var degree: Int
    var base_pos: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var out = self.block_idx[self.out_rank]
        var tp = self.degree
        var bstride = self.block_stride
        var tstride = Self.num_index_heads * bstride
        comptime bi_tstride = Self.num_index_heads * Self.topk_blocks
        var bases = InlineArray[F32Ptr, 64](uninitialized=True)
        for r in range(tp):
            bases[r] = self.partials[r]

        for t in range(self.start, self.end):
            var abs_pos = self.base_pos + t
            if abs_pos < 0:
                continue
            var num_blocks = abs_pos // Self.block_size + 1

            for h in range(Self.num_index_heads):
                var row_base = t * tstride + h * bstride

                @parameter
                def merged(b: Int) -> Float32:
                    var best = Float32(-1.0e30)
                    for r in range(tp):
                        var v = (bases[r] + row_base + b)[]
                        if v > best:
                            best = v
                    return best

                stream_select_blocks[
                    Self.block_size, Self.topk_blocks,
                    Self.local_blocks, Self.init_blocks, merged,
                ](num_blocks, out + t * bi_tstride + h * Self.topk_blocks)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_merge_index_topk[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    block_size: Int, num_index_heads: Int, topk_blocks: Int,
    local_blocks: Int, init_blocks: Int,
    max_worker_count: Int = 128,
](
    partials: Binding[Float32, o],
    block_idx: Binding[Int32, o],
    block_stride: Int,
    base_pos: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if seq_len <= 0:
        return
    var tp = len(pools)
    var per_node = (seq_len + tp - 1) // tp
    comptime K = IndexTopkMergeKernel[
        block_size, num_index_heads, topk_blocks, local_blocks, init_blocks, o,
    ]
    var span = DispatchSpan[Profile]()
    var buf = DispatchBuffer[K, max_worker_count]()
    for r in range(tp):
        var lo = r * per_node
        var hi = min(lo + per_node, seq_len)
        var cnt = hi - lo
        if cnt <= 0:
            continue
        _ = tile_dispatch(
            buf,
            K(partials, block_idx, r, block_stride, tp, base_pos, 0, 0),
            pools[r], cnt, base=lo)
    span.issued()
    join_all(pools)
    span.finish(prof, pools, "lightning_indexer.merge_topk")


@fieldwise_init
struct BlockIdxBroadcastKernel[
    num_index_heads: Int, topk_blocks: Int, o: ImmutOrigin,
](RangePartitionedKernel):
    var block_idx: Binding[Int32, Self.o]
    var dest_rank: Int
    var per_node: Int
    var tp: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime stride = Self.num_index_heads * Self.topk_blocks
        var dst = self.block_idx[self.dest_rank]
        for tok in range(self.start, self.end):
            var owner = tok // self.per_node
            if owner >= self.tp:
                owner = self.tp - 1
            if owner == self.dest_rank:
                continue
            var src = self.block_idx[owner]
            var off = tok * stride
            for k in range(stride):
                dst[off + k] = src[off + k]

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_broadcast_block_idx[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    num_index_heads: Int, topk_blocks: Int, max_worker_count: Int = 128,
](
    block_idx: Binding[Int32, o],
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var tp = len(pools)
    if tp <= 1 or seq_len <= 0:
        return
    var per_node = (seq_len + tp - 1) // tp
    comptime K = BlockIdxBroadcastKernel[num_index_heads, topk_blocks, o]
    var span = DispatchSpan[Profile]()
    var buf = DispatchBuffer[K, max_worker_count]()
    for r in range(tp):
        _ = tile_dispatch(
            buf, K(block_idx, r, per_node, tp, 0, 0), pools[r], seq_len)
    span.issued()
    join_all(pools)
    span.finish(prof, pools, "lightning_indexer.broadcast")


def dispatch_lightning_indexer[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    index_head_dim: Int, num_index_heads: Int,
    block_size: Int, topk_blocks: Int,
    local_blocks: Int, init_blocks: Int,
    page_len: Int,
    max_worker_count: Int = 128,
](
    index_q: Binding[BFloat16, o],
    index_k: Binding[BFloat16, o],
    block_idx: Binding[Int32, o],
    partial: Binding[Float32, o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    scale: Float32,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if seq_len <= 0:
        return

    var base_pos = Int(runs[].runs[0].base_pos)
    var max_block = (base_pos + seq_len - 1) // block_size + 1
    var block_stride = (max_block + 15) // 16 * 16

    dispatch_index_block_scores[
        index_head_dim=index_head_dim, num_index_heads=num_index_heads,
        block_size=block_size, page_len=page_len,
        max_worker_count=max_worker_count,
    ](index_q, index_k, partial, runs, block_stride, scale, seq_len,
      pools, prof)

    dispatch_merge_index_topk[
        block_size=block_size, num_index_heads=num_index_heads,
        topk_blocks=topk_blocks,
        local_blocks=local_blocks, init_blocks=init_blocks,
        max_worker_count=max_worker_count,
    ](partial, block_idx, block_stride, base_pos, seq_len, pools, prof)

    dispatch_broadcast_block_idx[
        num_index_heads=num_index_heads, topk_blocks=topk_blocks,
        max_worker_count=max_worker_count,
    ](block_idx, seq_len, pools, prof)


comptime M3_INDEX_HEAD_DIM = 128
comptime M3_INDEX_NUM_HEADS = 4
comptime M3_INDEX_BLOCK = 128
comptime M3_INDEX_TOPK_BLOCKS = 16
comptime M3_INDEX_LOCAL_BLOCKS = 1
comptime M3_INDEX_INIT_BLOCKS = 0
comptime M3_INDEX_SCALE = Float32(1.0) / sqrt[DType.float32, 1](
    M3_INDEX_HEAD_DIM)


def dispatch_minimax_m3_indexer[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    page_len: Int, max_worker_count: Int = 128,
](
    index_q: Binding[BFloat16, o],
    index_k: Binding[BFloat16, o],
    block_idx: Binding[Int32, o],
    partial: Binding[Float32, o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    dispatch_lightning_indexer[
        index_head_dim=M3_INDEX_HEAD_DIM,
        num_index_heads=M3_INDEX_NUM_HEADS,
        block_size=M3_INDEX_BLOCK,
        topk_blocks=M3_INDEX_TOPK_BLOCKS,
        local_blocks=M3_INDEX_LOCAL_BLOCKS,
        init_blocks=M3_INDEX_INIT_BLOCKS,
        page_len=page_len,
        max_worker_count=max_worker_count,
    ](index_q, index_k, block_idx, partial, runs, M3_INDEX_SCALE, seq_len,
      pools, prof)
