from std.collections import InlineArray
from std.memory import Span, UnsafePointer

from simd_math import fast_exp_softmax_biased

from .dot_products import dot_to_scalar
from .helpers import BF16Ptr, F32Ptr, W, accumulate_scaled, scale_unrolled


comptime TILE = W


trait KVSlot(TrivialRegisterPassable):
    @always_inline
    def slot(self, start_pos: Int, pos_t: Int) -> Int: ...


@fieldwise_init
struct LinearKV(KVSlot):
    @always_inline
    def slot(self, start_pos: Int, pos_t: Int) -> Int:
        return pos_t


@fieldwise_init
struct RingKV[window: Int](KVSlot):
    @always_inline
    def slot(self, start_pos: Int, pos_t: Int) -> Int:
        return (start_pos + pos_t) & (Self.window - 1)


@always_inline
def pow2_shift(value: Int) -> Int:
    var shift = 0
    while (1 << shift) < value:
        shift += 1
    return shift


@fieldwise_init
struct PagedKV(KVSlot):
    var base_rows: UnsafePointer[Int32, MutAnyOrigin]
    var shift: Int
    var row_mask: Int
    var page_mask: Int

    @always_inline
    def slot(self, start_pos: Int, pos_t: Int) -> Int:
        var p = start_pos + pos_t
        return Int(self.base_rows[(p >> self.shift) & self.page_mask]) + (
            p & self.row_mask)


@fieldwise_init
struct KVRun(Copyable, Movable, ImplicitlyCopyable):
    var buf_start: Int32
    var base_pos: Int32
    var rows_off: Int32
    var page_count: Int32


struct KVRunTable(Movable):
    var runs: List[KVRun]
    var base_rows: List[Int32]

    def __init__(out self):
        self.runs = List[KVRun]()
        self.base_rows = List[Int32]()

    def clear(mut self):
        self.runs.clear()
        self.base_rows.clear()

    def begin_run(mut self, buf_start: Int, base_pos: Int):
        self.runs.append(KVRun(
            Int32(buf_start), Int32(base_pos),
            Int32(len(self.base_rows)), Int32(0)))

    def add_base_row(mut self, row: Int32):
        self.base_rows.append(row)
        self.runs[len(self.runs) - 1].page_count += 1

    @always_inline
    def row_ptr(self, run_idx: Int) -> UnsafePointer[Int32, MutAnyOrigin]:
        return UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=Int(self.base_rows.unsafe_ptr()),
        ) + Int(self.runs[run_idx].rows_off)


@fieldwise_init
struct RunSplitBand(Copyable, ImplicitlyCopyable):
    var buf_start: Int
    var split_base: Int
    var n_splits: Int


def plan_run_splits(
    kv_lens: Span[Int, _],
    buf_starts: Span[Int, _],
    cap: Int,
    min_split: Int,
) -> List[RunSplitBand]:
    var num_runs = len(kv_lens)
    var bands = List[RunSplitBand](capacity=num_runs)
    var max_splits = List[Int](capacity=num_runs)
    var remaining = cap
    for i in range(num_runs):
        var kv_len = kv_lens[i]
        var ceiling = (kv_len + min_split - 1) // min_split if kv_len > 0 else 0
        max_splits.append(ceiling)
        var seeded = 1 if (ceiling > 0 and remaining > 0) else 0
        remaining -= seeded
        bands.append(RunSplitBand(buf_starts[i], 0, seeded))

    while remaining > 0:
        var best = -1
        for i in range(num_runs):
            if bands[i].n_splits == 0 or bands[i].n_splits >= max_splits[i]:
                continue
            if best < 0:
                best = i
                continue
            # Prefer the run with the larger KV-per-split: kv[i]/n[i] > kv[best]/n[best]
            if kv_lens[i] * bands[best].n_splits > kv_lens[best] * bands[i].n_splits:
                best = i
        if best < 0:
            break
        bands[best].n_splits += 1
        remaining -= 1

    var running = 0
    for i in range(num_runs):
        bands[i].split_base = running
        running += bands[i].n_splits
    return bands^


@always_inline
def flash_partial_stride(num_q: Int, head_dim: Int) -> Int:
    return ((num_q * head_dim + 2 * num_q) * 4 + 63) // 64 * 16


@always_inline
def full_local_kv_count(rank: Int, abs_pos: Int, degree: Int) -> Int:
    if abs_pos < 0:
        return 0
    if rank <= abs_pos % degree:
        return abs_pos // degree + 1
    return abs_pos // degree


@always_inline
def zero_accumulators[max_q: Int, head_dim: Int](
    read acc_ptrs: InlineArray[F32Ptr, max_q], num_q: Int,
):
    comptime assert head_dim % W == 0, (
        "attention head_dim must be divisible by f32 SIMD width")
    for h in range(num_q):
        for j in range(0, head_dim, W):
            (acc_ptrs[h] + j).store(SIMD[DType.float32, W](0))


@always_inline
def online_softmax_tile[
    tile: Int,
](
    scores: SIMD[DType.float32, tile],
    old_m: Float32,
) -> Tuple[Float32, Float32, SIMD[DType.float32, tile]]:
    var tile_max = scores.reduce_max()
    var m_new = tile_max if tile_max > old_m else old_m
    var corr = fast_exp_softmax_biased[1](
        max(SIMD[DType.float32, 1](-87.0),
            SIMD[DType.float32, 1](old_m - m_new)))[0]
    var weights = fast_exp_softmax_biased[tile](
        max(SIMD[DType.float32, tile](-87.0),
            scores - SIMD[DType.float32, tile](m_new)))
    return (m_new, corr, weights)


@always_inline
def process_kv_tile[
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
    var slots = InlineArray[Int, TILE](uninitialized=True)
    for t in range(tile_len):
        slots[t] = kv.slot(start_pos, pos + t)

    for q_idx in range(num_q):
        var kv_h = q_idx // gqa_ratio

        var scores = SIMD[DType.float32, TILE](-1e30)
        for t in range(tile_len):
            var k_head = k_base + slots[t] * kv_stride + kv_h * head_dim
            scores[t] = dot_to_scalar[head_dim](q_ptrs[q_idx], k_head)

        var sm = online_softmax_tile[TILE](scores, m[q_idx])
        var m_new = sm[0]
        var corr = sm[1]
        var weights = sm[2]

        scale_unrolled[cols=head_dim](acc_ptrs[q_idx], corr)
        l[q_idx] = l[q_idx] * corr + weights.reduce_add()
        m[q_idx] = m_new

        for t in range(tile_len):
            var v_head = v_base + slots[t] * kv_stride + kv_h * head_dim
            accumulate_scaled[cols=head_dim](
                v_head, weights[t], acc_ptrs[q_idx])
