from std.collections import InlineArray
from std.memory import Span, UnsafePointer

from threading.threading_traits import BurstThreadPool

from kernels.helpers import (
    Binding, RangePartitionedKernel, BF16Ptr, F32Ptr, I32Ptr,
    fanout_dispatch, W, scale_unrolled,
)
from kernels.attention_ops import (
    KVRunTable, PagedKV, KVSlot, full_local_kv_count, pow2_shift,
    online_softmax_tile, zero_accumulators, TILE,
)
from kernels.dot_products import bf16_panel_dot_to_scalars
from kernels.flash_attention_prefill import dispatch_merge_flash_prefill_partials
from kernels.logsum_merge import MergeSegment
from kernels.profiling import Profiler

from prototypes.lightning_indexer import (
    M3_INDEX_BLOCK, M3_INDEX_TOPK_BLOCKS,
)
from prototypes.sparse_attention import (
    block_row_lo, block_row_hi,
    M3_NUM_HEADS, M3_NUM_KV_HEADS, M3_HEAD_DIM, M3_GQA_RATIO, M3_KV_DIM,
)


@always_inline
def process_kv_tile_reuse[
    max_q: Int, KV: KVSlot, //,
    head_dim: Int, gqa_ratio: Int,
](
    kv: KV,
    read q_ptrs: InlineArray[BF16Ptr, max_q],
    k_base: BF16Ptr, v_base: BF16Ptr,
    start_pos: Int, pos: Int, tile_len: Int,
    mut m: InlineArray[Float32, max_q],
    mut l: InlineArray[Float32, max_q],
    read acc_ptrs: InlineArray[F32Ptr, max_q],
    num_q: Int, kv_stride: Int,
):
    """GQA K/V-stationary tile. Each K row is loaded once and dotted against the
    whole gqa group via the panel dot (weight=K, x_rows=queries); each V row is
    loaded once and fma'd into all `max_q` accumulators. The per-accumulator walk
    over t is preserved, so V accumulation stays bit-exact with the baseline; the
    score dot uses port_unroll=1 (max_q independent chains already saturate ILP),
    which reorders the K reduction vs the baseline's pick_port_unroll only at the
    ULP level."""
    debug_assert(num_q == max_q, "reuse tile expects the full gqa group")
    comptime PU = 1

    var slots = InlineArray[Int, TILE](uninitialized=True)
    for t in range(tile_len):
        slots[t] = kv.slot(start_pos, pos + t)

    var scores_mat = InlineArray[Float32, TILE * max_q](uninitialized=True)
    for t in range(tile_len):
        var k_head = k_base + slots[t] * kv_stride
        var sc = bf16_panel_dot_to_scalars[cols=head_dim, port_unroll=PU](
            k_head, q_ptrs)
        comptime for r in range(max_q):
            scores_mat[t * max_q + r] = sc[r]

    var weights_mat = InlineArray[Float32, TILE * max_q](uninitialized=True)
    comptime for r in range(max_q):
        var scores = SIMD[DType.float32, TILE](-1e30)
        for t in range(tile_len):
            scores[t] = scores_mat[t * max_q + r]
        var sm = online_softmax_tile[TILE](scores, m[r])
        var m_new = sm[0]
        var corr = sm[1]
        var weights = sm[2]
        scale_unrolled[cols=head_dim](acc_ptrs[r], corr)
        l[r] = l[r] * corr + weights.reduce_add()
        m[r] = m_new
        for t in range(tile_len):
            weights_mat[t * max_q + r] = weights[t]

    for t in range(tile_len):
        var v_head = v_base + slots[t] * kv_stride
        for j in range(0, head_dim, W):
            var vv = (v_head + j).load[width=W]().cast[DType.float32]()
            comptime for r in range(max_q):
                var aptr = acc_ptrs[r] + j
                var av = aptr.load[width=W]()
                aptr.store(
                    vv.fma(SIMD[DType.float32, W](weights_mat[t * max_q + r]), av))


@fieldwise_init
struct BlockSparseFlashReuseKernel[
    head_dim: Int, num_q: Int, gqa_ratio: Int, num_kv_heads: Int,
    block_size: Int, topk_blocks: Int, partial_stride: Int,
](RangePartitionedKernel):
    var runs: UnsafePointer[KVRunTable, MutAnyOrigin]
    var q: BF16Ptr
    var k_base: BF16Ptr
    var v_base: BF16Ptr
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
        comptime bi_tstride = Self.num_kv_heads * Self.topk_blocks

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
                    self.runs[].row_ptr(ri), self.page_shift, self.row_mask, -1)
                run_start = Int(run_list[ri].buf_start)
                run_pos = Int(run_list[ri].base_pos)
            var abs_pos = run_pos + (t - run_start)
            if abs_pos < 0:
                continue
            var cap = full_local_kv_count(self.rank, abs_pos, self.degree)

            var partial_tok = self.partials + t * Self.partial_stride
            var q_tok = self.q + t * q_stride
            var bi_tok = self.block_idx + t * bi_tstride

            for h in range(Self.num_kv_heads):
                var group_q = InlineArray[BF16Ptr, Self.gqa_ratio](
                    uninitialized=True)
                var group_acc = InlineArray[F32Ptr, Self.gqa_ratio](
                    uninitialized=True)
                var m_grp = InlineArray[Float32, Self.gqa_ratio](
                    fill=Float32(-1e30))
                var l_grp = InlineArray[Float32, Self.gqa_ratio](
                    fill=Float32(0))
                for hh in range(Self.gqa_ratio):
                    var global_h = h * Self.gqa_ratio + hh
                    group_q[hh] = q_tok + global_h * Self.head_dim
                    group_acc[hh] = partial_tok + global_h * Self.head_dim
                zero_accumulators[Self.gqa_ratio, Self.head_dim](
                    group_acc, Self.gqa_ratio)

                var kh_k = self.k_base + h * Self.head_dim
                var kh_v = self.v_base + h * Self.head_dim
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
                        process_kv_tile_reuse[
                            Self.head_dim, Self.gqa_ratio,
                        ](kv, group_q, kh_k, kh_v, 0, pos, tile_len,
                          m_grp, l_grp, group_acc, Self.gqa_ratio,
                          self.kv_stride)
                        pos += TILE

                for hh in range(Self.gqa_ratio):
                    var global_h = h * Self.gqa_ratio + hh
                    (partial_tok + m_off + global_h)[] = m_grp[hh]
                    (partial_tok + l_off + global_h)[] = l_grp[hh]

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_block_sparse_flash_reuse[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    head_dim: Int, num_q: Int, gqa_ratio: Int, num_kv_heads: Int,
    block_size: Int, topk_blocks: Int, partial_stride: Int,
    page_len: Int, max_worker_count: Int = 128,
](
    q: Binding[BFloat16, o],
    k_base: Binding[BFloat16, o],
    v_base: Binding[BFloat16, o],
    block_idx: Binding[Int32, o],
    partials: Binding[Float32, o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    kv_stride: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var degree = len(pools)
    comptime K = BlockSparseFlashReuseKernel[
        head_dim, num_q, gqa_ratio, num_kv_heads,
        block_size, topk_blocks, partial_stride,
    ]
    var rows_per_page = page_len // degree
    var psh = pow2_shift(rows_per_page)
    var rmask = rows_per_page - 1
    var ks = kv_stride
    var deg = degree

    @parameter
    def make(r: Int) -> K:
        return K(runs, q[r], k_base[r], v_base[r], block_idx[r], partials[r],
                 ks, deg, r, psh, rmask, 0, 0)

    var base_pos = Int(runs[].runs[0].base_pos)
    var causal_blocks = (base_pos + seq_len) // block_size + 1
    var per_q_blocks = min(topk_blocks, causal_blocks)
    var local_per_block = (block_size + degree - 1) // degree
    var data_bytes = (
        seq_len * num_kv_heads * per_q_blocks * local_per_block * kv_stride * 2)
    fanout_dispatch[
        make, max_worker_count=max_worker_count,
        label="sparse_attention.flash_reuse",
    ](pools, prof, seq_len, data_bytes)


def dispatch_block_sparse_attention_reuse[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    head_dim: Int, num_q: Int, gqa_ratio: Int, num_kv_heads: Int,
    block_size: Int, topk_blocks: Int,
    page_len: Int, max_worker_count: Int = 128,
](
    q: Binding[BFloat16, o],
    k_base: Binding[BFloat16, o],
    v_base: Binding[BFloat16, o],
    block_idx: Binding[Int32, o],
    output: Binding[BFloat16, o],
    partials: Binding[Float32, o],
    segment_scratch: Binding[MergeSegment, o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
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

    dispatch_block_sparse_flash_reuse[
        head_dim=head_dim, num_q=num_q, gqa_ratio=gqa_ratio,
        num_kv_heads=num_kv_heads, block_size=block_size,
        topk_blocks=topk_blocks, partial_stride=partial_stride,
        page_len=page_len, max_worker_count=max_worker_count,
    ](q, k_base, v_base, block_idx, partials, runs, kv_stride, seq_len,
      pools, prof)

    dispatch_merge_flash_prefill_partials[
        head_dim, max_worker_count=max_worker_count,
    ](output, partials, segment_scratch,
      num_q, local_num_q, partial_stride, seq_len, pools, prof)


def dispatch_minimax_m3_sparse_attention_reuse[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    page_len: Int, max_worker_count: Int = 128,
](
    q: Binding[BFloat16, o],
    k_base: Binding[BFloat16, o],
    v_base: Binding[BFloat16, o],
    block_idx: Binding[Int32, o],
    output: Binding[BFloat16, o],
    partials: Binding[Float32, o],
    segment_scratch: Binding[MergeSegment, o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    dispatch_block_sparse_attention_reuse[
        head_dim=M3_HEAD_DIM, num_q=M3_NUM_HEADS, gqa_ratio=M3_GQA_RATIO,
        num_kv_heads=M3_NUM_KV_HEADS, block_size=M3_INDEX_BLOCK,
        topk_blocks=M3_INDEX_TOPK_BLOCKS, page_len=page_len,
        max_worker_count=max_worker_count,
    ](q, k_base, v_base, block_idx, output, partials, segment_scratch,
      runs, M3_KV_DIM, seq_len, pools, prof)
