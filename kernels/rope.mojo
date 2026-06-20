from std.memory import UnsafePointer, memcpy
from std.sys.info import simd_width_of

from simd_math import sincos_simd
from threading.threading_traits import BurstThreadPool
from .helpers import (
    RangePartitionedKernel, Binding,
    fanout_dispatch,
    BF16Ptr, F32Ptr, W,
)
from .attention_ops import KVRunTable, PagedKV
from .dispatch_heuristics import ROPE_INLINE_TOKENS
from .profiling import Profiler


@always_inline
def rotate_pair_to[width: Int, pair_stride: Int](
    src: BF16Ptr, dst: BF16Ptr,
    cos_ptr: F32Ptr, sin_ptr: F32Ptr, j: Int,
):
    var x_lo = (src + j).load[width=width]().cast[DType.float32]()
    var x_hi = (src + pair_stride + j).load[width=width]().cast[DType.float32]()
    var cv = (cos_ptr + j).load[width=width]()
    var sv = (sin_ptr + j).load[width=width]()
    (dst + j).store((x_lo * cv - x_hi * sv).cast[DType.bfloat16]())
    (dst + pair_stride + j).store((x_hi * cv + x_lo * sv).cast[DType.bfloat16]())


@always_inline
def rope_head[half: Int, pair_stride: Int](
    head_ptr: BF16Ptr, cos_row: F32Ptr, sin_row: F32Ptr,
):
    for j in range(0, half, W):
        rotate_pair_to[W, pair_stride](
            head_ptr, head_ptr, cos_row, sin_row, j)


@always_inline
def rope_head_to[half: Int, pair_stride: Int, head_dim: Int](
    src: BF16Ptr, dst: BF16Ptr, cos_row: F32Ptr, sin_row: F32Ptr,
):
    for j in range(0, half, W):
        rotate_pair_to[W, pair_stride](src, dst, cos_row, sin_row, j)
    comptime
    if half < pair_stride:
        memcpy(dest=dst + half, src=src + half, count=pair_stride - half)
    comptime
    if pair_stride + half < head_dim:
        memcpy(dest=dst + pair_stride + half, src=src + pair_stride + half,
               count=head_dim - pair_stride - half)


@fieldwise_init
struct RopeCacheWriteKernel[
    half: Int,
    pair_stride: Int,
    head_dim: Int,
](RangePartitionedKernel):
    """The KV cache slot stride equals the per-token KV write size
    (`num_kv * head_dim`) in every Gemma4 cache layout, so it is derived rather
    than threaded separately."""
    var runs: UnsafePointer[KVRunTable, MutAnyOrigin]
    var q: BF16Ptr
    var k_src: BF16Ptr
    var v_src: BF16Ptr
    var k_cache: BF16Ptr
    var v_cache: BF16Ptr
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
            var cos_row = self.cos_table + pos * Self.half
            var sin_row = self.sin_table + pos * Self.half

            var q_tok = self.q + tok * q_stride
            for h in range(self.num_q):
                rope_head[Self.half, Self.pair_stride](
                    q_tok + h * Self.head_dim, cos_row, sin_row)

            if pos % self.cache_degree == self.rank:
                var slot = kv.slot(0, pos // self.cache_degree)
                var k_tok = self.k_src + tok * kv_stride
                var k_dst = self.k_cache + slot * kv_stride
                for h in range(self.num_kv):
                    rope_head_to[Self.half, Self.pair_stride, Self.head_dim](
                        k_tok + h * Self.head_dim,
                        k_dst + h * Self.head_dim,
                        cos_row, sin_row)

                var v_tok = self.v_src + tok * kv_stride
                var v_dst = self.v_cache + slot * kv_stride
                memcpy(dest=v_dst, src=v_tok, count=kv_stride)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_rope_cache_write[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    half: Int, pair_stride: Int, head_dim: Int,
    max_worker_count: Int = 128,
](
    q: Binding[BFloat16, o],
    k_src: Binding[BFloat16, o],
    v_src: Binding[BFloat16, o],
    k_cache: Binding[BFloat16, o],
    v_cache: Binding[BFloat16, o],
    cos_table: Binding[Float32, o],
    sin_table: Binding[Float32, o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    num_q: Int, num_kv: Int, cache_degree: Int,
    page_shift: Int, row_mask: Int, page_mask: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime K = RopeCacheWriteKernel[half, pair_stride, head_dim]
    var row_bytes = (num_q + 2 * num_kv) * head_dim * 2
    var nq = num_q
    var nkv = num_kv
    var cd = cache_degree
    var ps = page_shift
    var rm = row_mask
    var pm = page_mask

    @parameter
    def make(r: Int) -> K:
        return K(runs, q[r], k_src[r], v_src[r],
                 k_cache[r], v_cache[r],
                 cos_table[r], sin_table[r],
                 nq, nkv, cd, r % cd, ps, rm, pm, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="rope_cache_write"](
        pools, prof, seq_len, seq_len * row_bytes,
        inline_threshold_bytes=ROPE_INLINE_TOKENS * row_bytes)

@fieldwise_init
struct RopeKCacheWriteKernel[
    half: Int,
    pair_stride: Int,
    head_dim: Int,
](RangePartitionedKernel):
    """Key-only RoPE + paged cache write, for a selection branch with no value
    path (e.g. the MiniMax lightning indexer): ropes `num_q` query heads in
    place and ropes + writes `num_kv` key heads to the round-robin context cache.
    Identical to RopeCacheWriteKernel minus the V copy."""
    var runs: UnsafePointer[KVRunTable, MutAnyOrigin]
    var q: BF16Ptr
    var k_src: BF16Ptr
    var k_cache: BF16Ptr
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
            var cos_row = self.cos_table + pos * Self.half
            var sin_row = self.sin_table + pos * Self.half

            var q_tok = self.q + tok * q_stride
            for h in range(self.num_q):
                rope_head[Self.half, Self.pair_stride](
                    q_tok + h * Self.head_dim, cos_row, sin_row)

            if pos % self.cache_degree == self.rank:
                var slot = kv.slot(0, pos // self.cache_degree)
                var k_tok = self.k_src + tok * kv_stride
                var k_dst = self.k_cache + slot * kv_stride
                for h in range(self.num_kv):
                    rope_head_to[Self.half, Self.pair_stride, Self.head_dim](
                        k_tok + h * Self.head_dim,
                        k_dst + h * Self.head_dim,
                        cos_row, sin_row)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_rope_k_cache_write[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    half: Int, pair_stride: Int, head_dim: Int,
    max_worker_count: Int = 128,
](
    q: Binding[BFloat16, o],
    k_src: Binding[BFloat16, o],
    k_cache: Binding[BFloat16, o],
    cos_table: Binding[Float32, o],
    sin_table: Binding[Float32, o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    num_q: Int, num_kv: Int, cache_degree: Int,
    page_shift: Int, row_mask: Int, page_mask: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime K = RopeKCacheWriteKernel[half, pair_stride, head_dim]
    var row_bytes = (num_q + num_kv) * head_dim * 2
    var nq = num_q
    var nkv = num_kv
    var cd = cache_degree
    var ps = page_shift
    var rm = row_mask
    var pm = page_mask

    @parameter
    def make(r: Int) -> K:
        return K(runs, q[r], k_src[r], k_cache[r],
                 cos_table[r], sin_table[r],
                 nq, nkv, cd, r % cd, ps, rm, pm, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="rope_k_cache_write"](
        pools, prof, seq_len, seq_len * row_bytes,
        inline_threshold_bytes=ROPE_INLINE_TOKENS * row_bytes)


def init_rope_table_partial_strided[rotary_half: Int, rows: Int](
    cos_buf: F32Ptr, sin_buf: F32Ptr,
    theta: Float64, full_head_dim: Int,
    first_pos: Int, stride: Int,
):
    comptime f64w = simd_width_of[DType.float64]()

    for j in range(0, rotary_half, f64w):
        var inv = SIMD[DType.float64, f64w]()
        for k in range(f64w):
            var dim_idx = j + k
            inv[k] = 1.0 / (theta ** (Float64(2 * dim_idx) / Float64(full_head_dim)))

        for row in range(rows):
            var pos = first_pos + row * stride
            var sc = sincos_simd[polynomial_degree=8, width=f64w](
                SIMD[DType.float64, f64w](Float64(pos)) * inv)
            (cos_buf + row * rotary_half + j).store(sc.cos_val.cast[DType.float32]())
            (sin_buf + row * rotary_half + j).store(sc.sin_val.cast[DType.float32]())


def init_rope_table[half: Int, max_pos: Int](
    cos_buf: F32Ptr, sin_buf: F32Ptr,
    theta: Float64,
):
    init_rope_table_partial_strided[half, max_pos](
        cos_buf, sin_buf, theta, half * 2, 0, 1)
