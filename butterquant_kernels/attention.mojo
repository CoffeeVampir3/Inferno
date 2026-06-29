from std.collections import InlineArray
from std.memory import UnsafePointer

from threading.threading_traits import BurstThreadPool
from kernels.helpers import (
    BF16Ptr, F32Ptr, W, Binding, RangePartitionedKernel,
    WorkerRangePartitionedKernel, fanout_dispatch,
    scale_unrolled,
)
from kernels.attention_ops import (
    KVSlot, KVRunTable, PagedKV, TILE, online_softmax_tile, zero_accumulators,
    full_local_kv_count, pow2_shift,
)
from kernels.attention_dispatch_kernels import (
    dispatch_flash_sliding, dispatch_flash_full,
)
from kernels.logsum_merge import MergeSegment
from kernels.dispatch_heuristics import ROPE_INLINE_TOKENS
from kernels.profiling import Profiler

from butterquant.convert import store_bf16
from butterquant.dot_products import vnni_panel_score_dot, bq_score_unroll
from butterquant.head_prep import prep_head_qk_i8, prep_head_v_i8
from butterquant.types import I8Ptr


@always_inline
def bq_process_kv_tile[
    max_q: Int, KV: KVSlot, //,
    head_dim: Int, gqa_ratio: Int,
](
    kv: KV,
    read q_ptrs: InlineArray[I8Ptr, max_q],
    read qi_bias: InlineArray[Float32, max_q],
    read f_q: InlineArray[Float32, max_q],
    k_base: I8Ptr, v_base: I8Ptr,
    k_scale: F32Ptr, v_scale: F32Ptr,
    start_pos: Int, pos: Int, tile_len: Int,
    mut m: InlineArray[Float32, max_q],
    mut l: InlineArray[Float32, max_q],
    read acc_ptrs: InlineArray[F32Ptr, max_q],
    num_q: Int, num_kv: Int, kv_stride: Int,
):
    comptime inv127 = Float32(1.0) / Float32(127.0)
    comptime inv127sq = inv127 * inv127
    comptime CU = bq_score_unroll[head_dim, gqa_ratio]()
    debug_assert(num_q % gqa_ratio == 0, "bq_process_kv_tile needs whole gqa groups")

    var slots = InlineArray[Int, TILE](uninitialized=True)
    for t in range(tile_len):
        slots[t] = kv.slot(start_pos, pos + t)

    var num_groups = num_q // gqa_ratio
    var scores_mat = InlineArray[Float32, TILE * gqa_ratio](uninitialized=True)
    var weights_mat = InlineArray[Float32, TILE * gqa_ratio](uninitialized=True)

    for g in range(num_groups):
        var base_q = g * gqa_ratio
        var head_off = g * head_dim
        var kv_h = g

        var group_q = InlineArray[I8Ptr, gqa_ratio](uninitialized=True)
        comptime for r in range(gqa_ratio):
            group_q[r] = q_ptrs[base_q + r]

        for t in range(tile_len):
            var s_idx = slots[t]
            var k_head = k_base + s_idx * kv_stride + head_off
            var raw = vnni_panel_score_dot[head_dim, gqa_ratio, CU](
                k_head, group_q)
            var ks = k_scale[s_idx * num_kv + kv_h]
            comptime for r in range(gqa_ratio):
                var qi = base_q + r
                scores_mat[t * gqa_ratio + r] = (
                    (Float32(raw[r]) - qi_bias[qi]) * f_q[qi] * ks * inv127sq)

        comptime for r in range(gqa_ratio):
            var qi = base_q + r
            var scores = SIMD[DType.float32, TILE](-1e30)
            for t in range(tile_len):
                scores[t] = scores_mat[t * gqa_ratio + r]
            var sm = online_softmax_tile[TILE](scores, m[qi])
            scale_unrolled[cols=head_dim](acc_ptrs[qi], sm[1])
            l[qi] = l[qi] * sm[1] + sm[2].reduce_add()
            m[qi] = sm[0]
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
                    var aptr = acc_ptrs[base_q + r] + j
                    aptr.store(vv.fma(wts[r], aptr.load[width=W]()))


@fieldwise_init
struct BqFlashAttentionKernel[
    KV: KVSlot,
    head_dim: Int, max_q: Int, gqa_ratio: Int,
](WorkerRangePartitionedKernel):
    var kv: Self.KV
    var q: I8Ptr
    var qi_bias: F32Ptr
    var f_q: F32Ptr
    var k_base: I8Ptr
    var k_scale: F32Ptr
    var v_base: I8Ptr
    var v_scale: F32Ptr
    var partials: F32Ptr
    var num_q: Int
    var num_kv: Int
    var kv_stride: Int
    var partial_stride: Int
    var worker_id: Int
    var start_pos: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var my_partial = self.partials + self.worker_id * self.partial_stride
        var m_off = self.num_q * Self.head_dim
        var l_off = m_off + self.num_q

        var acc_ptrs = InlineArray[F32Ptr, Self.max_q](uninitialized=True)
        var q_ptrs = InlineArray[I8Ptr, Self.max_q](uninitialized=True)
        var qb = InlineArray[Float32, Self.max_q](uninitialized=True)
        var fq = InlineArray[Float32, Self.max_q](uninitialized=True)
        var m = InlineArray[Float32, Self.max_q](fill=Float32(-1e30))
        var l = InlineArray[Float32, Self.max_q](fill=Float32(0))

        for h in range(self.num_q):
            acc_ptrs[h] = my_partial + h * Self.head_dim
            q_ptrs[h] = self.q + h * Self.head_dim
            qb[h] = self.qi_bias[h]
            fq[h] = self.f_q[h]

        zero_accumulators[Self.max_q, Self.head_dim](acc_ptrs, self.num_q)

        var pos = self.start
        while pos < self.end:
            var tile_len = min(TILE, self.end - pos)
            bq_process_kv_tile[
                Self.head_dim, Self.gqa_ratio,
            ](self.kv, q_ptrs, qb, fq, self.k_base, self.v_base,
              self.k_scale, self.v_scale,
              self.start_pos, pos, tile_len, m, l, acc_ptrs,
              self.num_q, self.num_kv, self.kv_stride)
            pos += TILE

        for h in range(self.num_q):
            (my_partial + m_off + h)[] = m[h]
            (my_partial + l_off + h)[] = l[h]

    @always_inline
    def install_worker_range(mut self, worker_id: Int, start: Int, end: Int):
        self.worker_id = worker_id
        self.start = start
        self.end = end


@fieldwise_init
struct BqFlashPrefillSlidingKernel[
    head_dim: Int, max_q: Int, gqa_ratio: Int,
    window: Int, cache_size: Int, page_len: Int,
](WorkerRangePartitionedKernel):
    var runs: UnsafePointer[KVRunTable, MutAnyOrigin]
    var q: I8Ptr
    var qi_bias: F32Ptr
    var f_q: F32Ptr
    var k_base: I8Ptr
    var k_scale: F32Ptr
    var v_base: I8Ptr
    var v_scale: F32Ptr
    var output: BF16Ptr
    var partials: F32Ptr
    var num_q: Int
    var num_kv: Int
    var kv_stride: Int
    var partial_stride: Int
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
        var q_ptrs = InlineArray[I8Ptr, Self.max_q](uninitialized=True)
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
            var qb = InlineArray[Float32, Self.max_q](uninitialized=True)
            var fq = InlineArray[Float32, Self.max_q](uninitialized=True)
            var m = InlineArray[Float32, Self.max_q](fill=Float32(-1e30))
            var l = InlineArray[Float32, Self.max_q](fill=Float32(0))
            for h in range(self.num_q):
                q_ptrs[h] = q_tok + h * Self.head_dim
                qb[h] = self.qi_bias[t * self.num_q + h]
                fq[h] = self.f_q[t * self.num_q + h]

            zero_accumulators[Self.max_q, Self.head_dim](acc_ptrs, self.num_q)

            var pos = lo
            while pos < hi:
                var tile_len = min(TILE, hi - pos)
                bq_process_kv_tile[
                    Self.head_dim, Self.gqa_ratio,
                ](kv, q_ptrs, qb, fq, self.k_base, self.v_base,
                  self.k_scale, self.v_scale,
                  0, pos, tile_len, m, l, acc_ptrs,
                  self.num_q, self.num_kv, self.kv_stride)
                pos += TILE

            for h in range(self.num_q):
                if l[h] > 0:
                    var inv_l = SIMD[DType.float32, W](Float32(1.0) / l[h])
                    for j in range(0, Self.head_dim, W):
                        var v = (acc_ptrs[h] + j).load[width=W]() * inv_l
                        store_bf16[W](v, out_tok + h * Self.head_dim + j)
                else:
                    for j in range(0, Self.head_dim, W):
                        (out_tok + h * Self.head_dim + j).store(
                            SIMD[DType.bfloat16, W](0))

    @always_inline
    def install_worker_range(mut self, worker_id: Int, start: Int, end: Int):
        self.worker_id = worker_id
        self.start = start
        self.end = end


@fieldwise_init
struct BqFlashPrefillFullKernel[
    head_dim: Int, num_q: Int, num_kv: Int, gqa_ratio: Int,
    partial_stride: Int,
](RangePartitionedKernel):
    var runs: UnsafePointer[KVRunTable, MutAnyOrigin]
    var q: I8Ptr
    var qi_bias: F32Ptr
    var f_q: F32Ptr
    var k_base: I8Ptr
    var k_scale: F32Ptr
    var v_base: I8Ptr
    var v_scale: F32Ptr
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
        var q_ptrs = InlineArray[I8Ptr, Self.num_q](uninitialized=True)

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
            var qb = InlineArray[Float32, Self.num_q](uninitialized=True)
            var fq = InlineArray[Float32, Self.num_q](uninitialized=True)
            var m = InlineArray[Float32, Self.num_q](fill=Float32(-1e30))
            var l = InlineArray[Float32, Self.num_q](fill=Float32(0))

            comptime for h in range(Self.num_q):
                acc_ptrs[h] = partial_tok + h * Self.head_dim
                q_ptrs[h] = q_tok + h * Self.head_dim
                qb[h] = self.qi_bias[t * Self.num_q + h]
                fq[h] = self.f_q[t * Self.num_q + h]

            zero_accumulators[Self.num_q, Self.head_dim](acc_ptrs, Self.num_q)

            var pos = 0
            while pos < local_kv_count:
                var tile_len = min(TILE, local_kv_count - pos)
                bq_process_kv_tile[
                    Self.head_dim, Self.gqa_ratio,
                ](kv, q_ptrs, qb, fq, self.k_base, self.v_base,
                  self.k_scale, self.v_scale,
                  0, pos, tile_len, m, l, acc_ptrs,
                  Self.num_q, Self.num_kv, self.kv_stride)
                pos += TILE

            comptime for h in range(Self.num_q):
                (partial_tok + m_off + h)[] = m[h]
                (partial_tok + l_off + h)[] = l[h]

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_bq_sliding_attention[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    head_dim: Int, max_q: Int, gqa_ratio: Int,
    window: Int, cache_size: Int, page_len: Int,
    max_worker_count: Int = 128,
](
    q: Binding[Int8, o],
    qi_bias: Binding[Float32, o],
    f_q: Binding[Float32, o],
    k_base: Binding[Int8, o],
    k_scale: Binding[Float32, o],
    v_base: Binding[Int8, o],
    v_scale: Binding[Float32, o],
    output: Binding[BFloat16, o],
    partials: Binding[Float32, o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    num_q: Int, num_kv: Int, partial_stride: Int, kv_stride: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime DecodeK = BqFlashAttentionKernel[
        PagedKV, head_dim, max_q, gqa_ratio,
    ]
    comptime PrefillK = BqFlashPrefillSlidingKernel[
        head_dim, max_q, gqa_ratio, window, cache_size, page_len,
    ]
    comptime page_shift = pow2_shift(page_len)
    comptime row_mask = page_len - 1
    comptime page_mask = cache_size // page_len - 1
    var nq = num_q
    var nkv = num_kv
    var ps = partial_stride
    var ks = kv_stride

    @parameter
    def make_decode(r: Int, start_pos: Int) -> DecodeK:
        var kv = PagedKV(
            runs[].row_ptr(0),
            page_shift, row_mask, page_mask)
        return DecodeK(kv, q[r], qi_bias[r], f_q[r], k_base[r], k_scale[r],
                       v_base[r], v_scale[r], partials[r],
                       nq, nkv, ks, ps, 0, start_pos, 0, 0)

    @parameter
    def make_prefill(r: Int) -> PrefillK:
        return PrefillK(runs, q[r], qi_bias[r], f_q[r], k_base[r], k_scale[r],
                        v_base[r], v_scale[r], output[r], partials[r],
                        nq, nkv, ks, ps, 0, 0, 0)

    @parameter
    def make_decode_run(r: Int, run_idx: Int, start_pos: Int) -> DecodeK:
        var kv = PagedKV(
            runs[].row_ptr(run_idx),
            page_shift, row_mask, page_mask)
        var q_off = Int(runs[].runs[run_idx].buf_start) * nq * head_dim
        var b_off = Int(runs[].runs[run_idx].buf_start) * nq
        return DecodeK(kv, q[r] + q_off, qi_bias[r] + b_off, f_q[r] + b_off,
                       k_base[r], k_scale[r], v_base[r], v_scale[r],
                       partials[r], nq, nkv, ks, ps, 0, start_pos, 0, 0)

    dispatch_flash_sliding[
        head_dim, window, 1,
        make_decode, make_prefill, make_decode_run,
        "bq_sliding_attn.flash", "bq_sliding_attn.prefill",
        "bq_sliding_attn.flash_batched",
        max_worker_count=max_worker_count,
    ](output, partials, runs, num_q, partial_stride, kv_stride, seq_len,
      pools, prof)


def dispatch_bq_full_attention[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    head_dim: Int, num_q: Int, num_kv: Int, gqa_ratio: Int,
    kv_stride: Int, partial_stride: Int, page_len: Int,
    max_worker_count: Int = 128,
](
    q: Binding[Int8, o],
    qi_bias: Binding[Float32, o],
    f_q: Binding[Float32, o],
    k_base: Binding[Int8, o],
    k_scale: Binding[Float32, o],
    v_base: Binding[Int8, o],
    v_scale: Binding[Float32, o],
    q_local_output: Binding[BFloat16, o],
    partials: Binding[Float32, o],
    segment_scratch: Binding[MergeSegment, o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    local_num_q: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime DecodeK = BqFlashAttentionKernel[
        PagedKV, head_dim, num_q, gqa_ratio,
    ]
    comptime PrefillK = BqFlashPrefillFullKernel[
        head_dim, num_q, num_kv, gqa_ratio, partial_stride,
    ]
    var degree = len(pools)
    var rows_per_page = page_len // degree
    var page_shift = pow2_shift(rows_per_page)
    var row_mask = rows_per_page - 1

    @parameter
    def make_decode(r: Int) -> DecodeK:
        var kv = PagedKV(
            runs[].row_ptr(0), page_shift, row_mask, -1)
        return DecodeK(kv, q[r], qi_bias[r], f_q[r], k_base[r], k_scale[r],
                       v_base[r], v_scale[r], partials[r],
                       num_q, num_kv, kv_stride, partial_stride, 0, 0, 0, 0)

    @parameter
    def make_prefill(r: Int) -> PrefillK:
        return PrefillK(runs, q[r], qi_bias[r], f_q[r], k_base[r], k_scale[r],
                        v_base[r], v_scale[r], partials[r],
                        kv_stride, degree, r, page_shift, row_mask, 0, 0)

    @parameter
    def make_decode_run(r: Int, run_idx: Int) -> DecodeK:
        var kv = PagedKV(
            runs[].row_ptr(run_idx),
            page_shift, row_mask, -1)
        var q_off = Int(runs[].runs[run_idx].buf_start) * num_q * head_dim
        var b_off = Int(runs[].runs[run_idx].buf_start) * num_q
        return DecodeK(kv, q[r] + q_off, qi_bias[r] + b_off, f_q[r] + b_off,
                       k_base[r], k_scale[r], v_base[r], v_scale[r],
                       partials[r], num_q, num_kv, kv_stride, partial_stride,
                       0, 0, 0, 0)

    dispatch_flash_full[
        head_dim, kv_stride, 1,
        make_decode, make_prefill, make_decode_run,
        "bq_full_attn.flash", "bq_full_attn.prefill",
        "bq_full_attn.flash_batched",
        max_worker_count=max_worker_count,
    ](q_local_output, partials, segment_scratch, runs,
      num_q, local_num_q, partial_stride, seq_len, pools, prof)


@fieldwise_init
struct BqAttnPrepKernel[
    head_dim: Int, rope_half: Int, pair_stride: Int,
    sqrt_n: Float32, n_eps: Float32, q_scale: Float32, norm_v: Bool,
](RangePartitionedKernel):
    var runs: UnsafePointer[KVRunTable, MutAnyOrigin]
    var q_src: BF16Ptr
    var k_src: BF16Ptr
    var v_src: BF16Ptr
    var q_norm: BF16Ptr
    var k_norm: BF16Ptr
    var q_i8: I8Ptr
    var qi_bias: F32Ptr
    var f_q: F32Ptr
    var k_cache: I8Ptr
    var k_scale: F32Ptr
    var v_cache: I8Ptr
    var v_scale: F32Ptr
    var cos_table: F32Ptr
    var sin_table: F32Ptr
    var num_q: Int
    var num_kv: Int
    var cache_degree: Int
    var rank: Int
    var page_shift: Int
    var row_mask: Int
    var page_mask: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var q_stride = self.num_q * Self.head_dim
        var kv_stride = self.num_kv * Self.head_dim
        ref run_list = self.runs[].runs
        var num_runs = len(run_list)
        var r = 0
        var kv = PagedKV(
            self.runs[].row_ptr(0),
            self.page_shift, self.row_mask, self.page_mask)
        var run_start = Int(run_list[0].buf_start)
        var run_pos = Int(run_list[0].base_pos)
        for tok in range(self.start, self.end):
            while r + 1 < num_runs and tok >= Int(run_list[r + 1].buf_start):
                r += 1
                kv = PagedKV(
                    self.runs[].row_ptr(r),
                    self.page_shift, self.row_mask, self.page_mask)
                run_start = Int(run_list[r].buf_start)
                run_pos = Int(run_list[r].base_pos)
            var pos = run_pos + (tok - run_start)
            var cos_row = self.cos_table + pos * Self.rope_half
            var sin_row = self.sin_table + pos * Self.rope_half

            var q_tok = self.q_src + tok * q_stride
            var qi_tok = self.q_i8 + tok * q_stride
            for h in range(self.num_q):
                var res = prep_head_qk_i8[
                    Self.head_dim, Self.rope_half, Self.pair_stride,
                    Self.sqrt_n, Self.n_eps,
                ](q_tok + h * Self.head_dim, self.q_norm, cos_row, sin_row,
                  qi_tok + h * Self.head_dim)
                (self.qi_bias + tok * self.num_q + h)[] = Float32(res[1]) * 128.0
                (self.f_q + tok * self.num_q + h)[] = res[0] * Self.q_scale

            if pos % self.cache_degree == self.rank:
                var slot = kv.slot(0, pos // self.cache_degree)
                var k_tok = self.k_src + tok * kv_stride
                var v_tok = self.v_src + tok * kv_stride
                var k_dst = self.k_cache + slot * kv_stride
                var v_dst = self.v_cache + slot * kv_stride
                var ks_dst = self.k_scale + slot * self.num_kv
                var vs_dst = self.v_scale + slot * self.num_kv
                for h in range(self.num_kv):
                    var sk = prep_head_qk_i8[
                        Self.head_dim, Self.rope_half, Self.pair_stride,
                        Self.sqrt_n, Self.n_eps,
                    ](k_tok + h * Self.head_dim, self.k_norm, cos_row, sin_row,
                      k_dst + h * Self.head_dim)
                    ks_dst[h] = sk[0]
                    var sv = prep_head_v_i8[
                        Self.head_dim, Self.norm_v, Self.sqrt_n, Self.n_eps,
                    ](v_tok + h * Self.head_dim, v_dst + h * Self.head_dim)
                    vs_dst[h] = sv

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_bq_attn_prep[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    head_dim: Int, rope_half: Int, pair_stride: Int,
    sqrt_n: Float32, n_eps: Float32, q_scale: Float32, norm_v: Bool,
    max_worker_count: Int = 128,
](
    q_src: Binding[BFloat16, o],
    k_src: Binding[BFloat16, o],
    v_src: Binding[BFloat16, o],
    q_norm: Binding[BFloat16, o],
    k_norm: Binding[BFloat16, o],
    q_i8: Binding[Int8, o],
    qi_bias: Binding[Float32, o],
    f_q: Binding[Float32, o],
    k_cache: Binding[Int8, o],
    k_scale: Binding[Float32, o],
    v_cache: Binding[Int8, o],
    v_scale: Binding[Float32, o],
    cos_table: Binding[Float32, o],
    sin_table: Binding[Float32, o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    num_q: Int, num_kv: Int, cache_degree: Int,
    page_shift: Int, row_mask: Int, page_mask: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime K = BqAttnPrepKernel[
        head_dim, rope_half, pair_stride, sqrt_n, n_eps, q_scale, norm_v]
    var row_bytes = (num_q + 2 * num_kv) * head_dim * 6
    var nq = num_q
    var nkv = num_kv
    var cd = cache_degree
    var ps = page_shift
    var rm = row_mask
    var pm = page_mask

    @parameter
    def make(r: Int) -> K:
        return K(runs, q_src[r], k_src[r], v_src[r], q_norm[r], k_norm[r],
                 q_i8[r], qi_bias[r], f_q[r],
                 k_cache[r], k_scale[r], v_cache[r], v_scale[r],
                 cos_table[r], sin_table[r],
                 nq, nkv, cd, r % cd, ps, rm, pm, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="bq_attn_prep"](
        pools, prof, seq_len, seq_len * row_bytes,
        inline_threshold_bytes=ROPE_INLINE_TOKENS * row_bytes)
