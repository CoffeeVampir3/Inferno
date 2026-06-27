from std.collections import InlineArray
from std.memory import UnsafePointer

from threading.threading_traits import BurstThreadPool

from kernels.helpers import (
    BF16Ptr, F32Ptr, I32Ptr, W, Binding, RangePartitionedKernel,
    WorkerRangePartitionedKernel, fanout_dispatch, fanout_dispatch_per_rank,
    saturate_workers, scale_unrolled,
)
from kernels.attention_ops import (
    KVSlot, KVRunTable, PagedKV, TILE, online_softmax_tile, zero_accumulators,
    full_local_kv_count, pow2_shift,
)
from kernels.flash_attention_prefill import dispatch_merge_flash_prefill_partials
from kernels.logsum_merge import (
    MergeSegment, dispatch_merge_context_flash_partials,
)
from kernels.profiling import Profiler

from butterquant.dot_products import vnni_panel_score_dot, bq_score_unroll
from butterquant.types import I8Ptr

from prototypes.sparse_attention import block_row_lo, block_row_hi
from prototypes.lightning_indexer import M3_INDEX_BLOCK, M3_INDEX_TOPK_BLOCKS


@always_inline
def bq_process_kv_tile_head[
    gqa_ratio: Int, KV: KVSlot, //,
    head_dim: Int,
](
    kv: KV,
    read group_q: InlineArray[I8Ptr, gqa_ratio],
    read qi_bias: InlineArray[Float32, gqa_ratio],
    read f_q: InlineArray[Float32, gqa_ratio],
    k_base: I8Ptr, v_base: I8Ptr,
    k_scale: F32Ptr, v_scale: F32Ptr,
    kv_h: Int, head_off: Int,
    start_pos: Int, pos: Int, tile_len: Int,
    mut m: InlineArray[Float32, gqa_ratio],
    mut l: InlineArray[Float32, gqa_ratio],
    read acc_ptrs: InlineArray[F32Ptr, gqa_ratio],
    num_kv: Int, kv_stride: Int,
):
    comptime inv127 = Float32(1.0) / Float32(127.0)
    comptime inv127sq = inv127 * inv127
    comptime CU = bq_score_unroll[head_dim, gqa_ratio]()

    var slots = InlineArray[Int, TILE](uninitialized=True)
    for t in range(tile_len):
        slots[t] = kv.slot(start_pos, pos + t)

    var scores_mat = InlineArray[Float32, TILE * gqa_ratio](uninitialized=True)
    var weights_mat = InlineArray[Float32, TILE * gqa_ratio](uninitialized=True)

    for t in range(tile_len):
        var s_idx = slots[t]
        var k_head = k_base + s_idx * kv_stride + head_off
        var raw = vnni_panel_score_dot[head_dim, gqa_ratio, CU](k_head, group_q)
        var ks = k_scale[s_idx * num_kv + kv_h]
        comptime for r in range(gqa_ratio):
            scores_mat[t * gqa_ratio + r] = (
                (Float32(raw[r]) - qi_bias[r]) * f_q[r] * ks * inv127sq)

    comptime for r in range(gqa_ratio):
        var scores = SIMD[DType.float32, TILE](-1e30)
        for t in range(tile_len):
            scores[t] = scores_mat[t * gqa_ratio + r]
        var sm = online_softmax_tile[TILE](scores, m[r])
        scale_unrolled[cols=head_dim](acc_ptrs[r], sm[1])
        l[r] = l[r] * sm[1] + sm[2].reduce_add()
        m[r] = sm[0]
        for t in range(tile_len):
            weights_mat[t * gqa_ratio + r] = sm[2][t]

    for t in range(tile_len):
        var s_idx = slots[t]
        var v_head = v_base + s_idx * kv_stride + head_off
        var vs = v_scale[s_idx * num_kv + kv_h]
        var wts = InlineArray[SIMD[DType.float32, W], gqa_ratio](
            uninitialized=True)
        comptime for r in range(gqa_ratio):
            wts[r] = SIMD[DType.float32, W](
                weights_mat[t * gqa_ratio + r] * vs * inv127)
        for j in range(0, head_dim, W):
            var vv = (v_head + j).load[width=W]().cast[DType.float32]()
            comptime for r in range(gqa_ratio):
                var aptr = acc_ptrs[r] + j
                aptr.store(vv.fma(wts[r], aptr.load[width=W]()))


@fieldwise_init
struct BqBlockSparseFlashKernel[
    head_dim: Int, num_q: Int, num_kv: Int, gqa_ratio: Int,
    block_size: Int, topk_blocks: Int, partial_stride: Int,
](RangePartitionedKernel):
    var runs: UnsafePointer[KVRunTable, MutUntrackedOrigin]
    var q: I8Ptr
    var qi_bias: F32Ptr
    var f_q: F32Ptr
    var k_base: I8Ptr
    var k_scale: F32Ptr
    var v_base: I8Ptr
    var v_scale: F32Ptr
    var block_idx: I32Ptr
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
        comptime bi_tstride = Self.num_kv * Self.topk_blocks

        ref run_list = self.runs[].runs
        var num_runs = len(run_list)
        var ri = 0
        var kv = PagedKV(
            self.runs[].row_ptr(0), self.page_shift, self.row_mask, -1)
        var run_start = Int(run_list[0].buf_start)
        var run_pos = Int(run_list[0].base_pos)

        for t in range(self.start, self.end):
            while ri + 1 < num_runs and t >= Int(run_list[ri + 1].buf_start):
                ri += 1
                kv = PagedKV(
                    self.runs[].row_ptr(ri),
                    self.page_shift, self.row_mask, -1)
                run_start = Int(run_list[ri].buf_start)
                run_pos = Int(run_list[ri].base_pos)
            var abs_pos = run_pos + (t - run_start)
            if abs_pos < 0:
                continue
            var cap = full_local_kv_count(self.rank, abs_pos, self.degree)

            var partial_tok = self.partials + t * Self.partial_stride
            var q_tok = self.q + t * q_stride
            var bi_tok = self.block_idx + t * bi_tstride

            for h in range(Self.num_kv):
                var head_off = h * Self.head_dim
                var group_q = InlineArray[I8Ptr, Self.gqa_ratio](
                    uninitialized=True)
                var group_acc = InlineArray[F32Ptr, Self.gqa_ratio](
                    uninitialized=True)
                var qb = InlineArray[Float32, Self.gqa_ratio](
                    uninitialized=True)
                var fq = InlineArray[Float32, Self.gqa_ratio](
                    uninitialized=True)
                var m_grp = InlineArray[Float32, Self.gqa_ratio](
                    fill=Float32(-1e30))
                var l_grp = InlineArray[Float32, Self.gqa_ratio](
                    fill=Float32(0))
                for hh in range(Self.gqa_ratio):
                    var global_h = h * Self.gqa_ratio + hh
                    group_q[hh] = q_tok + global_h * Self.head_dim
                    group_acc[hh] = partial_tok + global_h * Self.head_dim
                    qb[hh] = self.qi_bias[t * Self.num_q + global_h]
                    fq[hh] = self.f_q[t * Self.num_q + global_h]
                zero_accumulators[Self.gqa_ratio, Self.head_dim](
                    group_acc, Self.gqa_ratio)

                var row = bi_tok + h * Self.topk_blocks
                for s in range(Self.topk_blocks):
                    var b = Int(row[s])
                    if b < 0:
                        break
                    var plo = block_row_lo(
                        b, Self.block_size, self.rank, self.degree)
                    var phi = min(
                        block_row_hi(
                            b, Self.block_size, self.rank, self.degree),
                        cap)
                    var pos = plo
                    while pos < phi:
                        var tile_len = min(TILE, phi - pos)
                        bq_process_kv_tile_head[
                            Self.head_dim,
                        ](kv, group_q, qb, fq, self.k_base, self.v_base,
                          self.k_scale, self.v_scale, h, head_off,
                          0, pos, tile_len, m_grp, l_grp, group_acc,
                          Self.num_kv, self.kv_stride)
                        pos += TILE

                for hh in range(Self.gqa_ratio):
                    var global_h = h * Self.gqa_ratio + hh
                    (partial_tok + m_off + global_h)[] = m_grp[hh]
                    (partial_tok + l_off + global_h)[] = l_grp[hh]

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_bq_block_sparse_flash[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    head_dim: Int, num_q: Int, num_kv: Int, gqa_ratio: Int,
    block_size: Int, topk_blocks: Int, partial_stride: Int,
    page_len: Int, max_worker_count: Int = 128,
](
    q: Binding[Int8, o],
    qi_bias: Binding[Float32, o],
    f_q: Binding[Float32, o],
    k_base: Binding[Int8, o],
    k_scale: Binding[Float32, o],
    v_base: Binding[Int8, o],
    v_scale: Binding[Float32, o],
    block_idx: Binding[Int32, o],
    partials: Binding[Float32, o],
    runs: UnsafePointer[KVRunTable, MutUntrackedOrigin],
    kv_stride: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var degree = len(pools)
    comptime K = BqBlockSparseFlashKernel[
        head_dim, num_q, num_kv, gqa_ratio,
        block_size, topk_blocks, partial_stride,
    ]
    var rows_per_page = page_len // degree
    var psh = pow2_shift(rows_per_page)
    var rmask = rows_per_page - 1
    var ks = kv_stride
    var deg = degree

    @parameter
    def make(r: Int) -> K:
        return K(runs, q[r], qi_bias[r], f_q[r], k_base[r], k_scale[r],
                 v_base[r], v_scale[r], block_idx[r], partials[r],
                 ks, deg, r, psh, rmask, 0, 0)

    var base_pos = Int(runs[].runs[0].base_pos)
    var causal_blocks = (base_pos + seq_len) // block_size + 1
    var per_q_blocks = min(topk_blocks, causal_blocks)
    var local_per_block = (block_size + degree - 1) // degree
    var data_bytes = (
        seq_len * num_kv * per_q_blocks * local_per_block * kv_stride)
    fanout_dispatch[
        make, max_worker_count=max_worker_count,
        label="bq_sparse_attention.flash",
    ](pools, prof, seq_len, data_bytes)


@fieldwise_init
struct BqBlockSparseFlashDecodeKernel[
    head_dim: Int, num_q: Int, num_kv: Int, gqa_ratio: Int,
    block_size: Int, topk_blocks: Int, partial_stride: Int,
](WorkerRangePartitionedKernel):
    """Decode-only worker-parallel sparse flash. The single query's work is the
    flat unit space `[0, num_kv * per_q_blocks)` (unit `u` -> kv-head
    `u // per_q_blocks`, block slot `u % per_q_blocks`); each worker owns a
    contiguous unit band and writes its own `(acc, m, l)` partial sub-row at
    `partials + worker_id * partial_stride`. A worker streams only the selected
    blocks it owns, so a single decode token spreads across the whole pool
    instead of one core. Heads the worker never touches keep the identity
    partial (`m=-1e30, l=0`); the cross-rank/cross-worker logsum merge combines
    every sub-row per global head, so block splits within a kv-head reconverge
    exactly like flash partials."""
    var runs: UnsafePointer[KVRunTable, MutUntrackedOrigin]
    var q: I8Ptr
    var qi_bias: F32Ptr
    var f_q: F32Ptr
    var k_base: I8Ptr
    var k_scale: F32Ptr
    var v_base: I8Ptr
    var v_scale: F32Ptr
    var block_idx: I32Ptr
    var partials: F32Ptr
    var kv_stride: Int
    var degree: Int
    var rank: Int
    var page_shift: Int
    var row_mask: Int
    var per_q_blocks: Int
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime m_off = Self.num_q * Self.head_dim
        comptime l_off = m_off + Self.num_q

        var partial_w = self.partials + self.worker_id * Self.partial_stride
        for gh in range(Self.num_q):
            (partial_w + m_off + gh)[] = Float32(-1e30)
            (partial_w + l_off + gh)[] = Float32(0)

        if self.start >= self.end:
            return

        ref run_list = self.runs[].runs
        var kv = PagedKV(
            self.runs[].row_ptr(0), self.page_shift, self.row_mask, -1)
        var abs_pos = Int(run_list[0].base_pos)
        if abs_pos < 0:
            return
        var cap = full_local_kv_count(self.rank, abs_pos, self.degree)
        var pqb = self.per_q_blocks

        var h0 = self.start // pqb
        var h1 = (self.end - 1) // pqb
        for h in range(h0, h1 + 1):
            var head_off = h * Self.head_dim
            var slot_lo = max(self.start, h * pqb) - h * pqb
            var slot_hi = min(self.end, (h + 1) * pqb) - h * pqb

            var group_q = InlineArray[I8Ptr, Self.gqa_ratio](uninitialized=True)
            var group_acc = InlineArray[F32Ptr, Self.gqa_ratio](
                uninitialized=True)
            var qb = InlineArray[Float32, Self.gqa_ratio](uninitialized=True)
            var fq = InlineArray[Float32, Self.gqa_ratio](uninitialized=True)
            var m_grp = InlineArray[Float32, Self.gqa_ratio](fill=Float32(-1e30))
            var l_grp = InlineArray[Float32, Self.gqa_ratio](fill=Float32(0))
            for hh in range(Self.gqa_ratio):
                var gh = h * Self.gqa_ratio + hh
                group_q[hh] = self.q + gh * Self.head_dim
                group_acc[hh] = partial_w + gh * Self.head_dim
                qb[hh] = self.qi_bias[gh]
                fq[hh] = self.f_q[gh]
            zero_accumulators[Self.gqa_ratio, Self.head_dim](
                group_acc, Self.gqa_ratio)

            var row = self.block_idx + h * Self.topk_blocks
            for s in range(slot_lo, slot_hi):
                var b = Int(row[s])
                if b < 0:
                    continue
                var plo = block_row_lo(
                    b, Self.block_size, self.rank, self.degree)
                var phi = min(
                    block_row_hi(
                        b, Self.block_size, self.rank, self.degree),
                    cap)
                var pos = plo
                while pos < phi:
                    var tile_len = min(TILE, phi - pos)
                    bq_process_kv_tile_head[
                        Self.head_dim,
                    ](kv, group_q, qb, fq, self.k_base, self.v_base,
                      self.k_scale, self.v_scale, h, head_off,
                      0, pos, tile_len, m_grp, l_grp, group_acc,
                      Self.num_kv, self.kv_stride)
                    pos += TILE

            for hh in range(Self.gqa_ratio):
                var gh = h * Self.gqa_ratio + hh
                (partial_w + m_off + gh)[] = m_grp[hh]
                (partial_w + l_off + gh)[] = l_grp[hh]

    @always_inline
    def install_worker_range(mut self, worker_id: Int, start: Int, end: Int):
        self.worker_id = worker_id
        self.start = start
        self.end = end


def dispatch_bq_block_sparse_attention[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    head_dim: Int, num_q: Int, num_kv: Int, gqa_ratio: Int,
    block_size: Int, topk_blocks: Int,
    page_len: Int, max_worker_count: Int = 128,
](
    q: Binding[Int8, o],
    qi_bias: Binding[Float32, o],
    f_q: Binding[Float32, o],
    k_base: Binding[Int8, o],
    k_scale: Binding[Float32, o],
    v_base: Binding[Int8, o],
    v_scale: Binding[Float32, o],
    block_idx: Binding[Int32, o],
    output: Binding[BFloat16, o],
    partials: Binding[Float32, o],
    segment_scratch: Binding[MergeSegment, o],
    runs: UnsafePointer[KVRunTable, MutUntrackedOrigin],
    kv_stride: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if seq_len <= 0:
        return
    var degree = len(pools)
    var local_num_q = num_q // degree
    comptime partial_stride = ((num_q * head_dim + 2 * num_q) * 4 + 63) // 64 * 16

    if seq_len == 1:
        var base_pos = Int(runs[].runs[0].base_pos)
        var causal_blocks = base_pos // block_size + 1
        var per_q_blocks = min(topk_blocks, causal_blocks)
        comptime DecodeK = BqBlockSparseFlashDecodeKernel[
            head_dim, num_q, num_kv, gqa_ratio,
            block_size, topk_blocks, partial_stride,
        ]
        var rows_per_page = page_len // degree
        var psh = pow2_shift(rows_per_page)
        var rmask = rows_per_page - 1
        var ks = kv_stride
        var deg = degree
        var pqb = per_q_blocks
        var local_per_block = (block_size + degree - 1) // degree

        @parameter
        def make(r: Int) -> DecodeK:
            return DecodeK(
                runs, q[r], qi_bias[r], f_q[r], k_base[r], k_scale[r],
                v_base[r], v_scale[r], block_idx[r], partials[r],
                ks, deg, r, psh, rmask, pqb, 0, 0, 0)

        @parameter
        def total_for(r: Int) -> Int:
            return num_kv * pqb

        @parameter
        def bytes_for(r: Int) -> Int:
            return num_kv * pqb * local_per_block * ks * 2

        var nws = fanout_dispatch_per_rank[
            make, total_for, bytes_for,
            max_worker_count=max_worker_count,
            worker_policy=saturate_workers,
            label="bq_sparse_attention.flash",
        ](pools, prof)

        dispatch_merge_context_flash_partials[
            head_dim, max_worker_count=max_worker_count,
        ](output, partials, segment_scratch, nws, num_q, local_num_q,
          partial_stride, pools, prof)
        return

    dispatch_bq_block_sparse_flash[
        head_dim=head_dim, num_q=num_q, num_kv=num_kv, gqa_ratio=gqa_ratio,
        block_size=block_size, topk_blocks=topk_blocks,
        partial_stride=partial_stride, page_len=page_len,
        max_worker_count=max_worker_count,
    ](q, qi_bias, f_q, k_base, k_scale, v_base, v_scale, block_idx, partials,
      runs, kv_stride, seq_len, pools, prof)

    dispatch_merge_flash_prefill_partials[
        head_dim, max_worker_count=max_worker_count,
    ](output, partials, segment_scratch,
      num_q, local_num_q, partial_stride, seq_len, pools, prof)


comptime M3_NUM_HEADS = 64
comptime M3_NUM_KV_HEADS = 4
comptime M3_HEAD_DIM = 128
comptime M3_GQA_RATIO = M3_NUM_HEADS // M3_NUM_KV_HEADS
comptime M3_KV_DIM = M3_NUM_KV_HEADS * M3_HEAD_DIM


def dispatch_bq_minimax_m3_sparse_attention[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    page_len: Int, max_worker_count: Int = 128,
](
    q: Binding[Int8, o],
    qi_bias: Binding[Float32, o],
    f_q: Binding[Float32, o],
    k_base: Binding[Int8, o],
    k_scale: Binding[Float32, o],
    v_base: Binding[Int8, o],
    v_scale: Binding[Float32, o],
    block_idx: Binding[Int32, o],
    output: Binding[BFloat16, o],
    partials: Binding[Float32, o],
    segment_scratch: Binding[MergeSegment, o],
    runs: UnsafePointer[KVRunTable, MutUntrackedOrigin],
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    dispatch_bq_block_sparse_attention[
        head_dim=M3_HEAD_DIM, num_q=M3_NUM_HEADS, num_kv=M3_NUM_KV_HEADS,
        gqa_ratio=M3_GQA_RATIO, block_size=M3_INDEX_BLOCK,
        topk_blocks=M3_INDEX_TOPK_BLOCKS, page_len=page_len,
        max_worker_count=max_worker_count,
    ](q, qi_bias, f_q, k_base, k_scale, v_base, v_scale, block_idx, output,
      partials, segment_scratch, runs, M3_KV_DIM, seq_len, pools, prof)
