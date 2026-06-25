from std.collections import InlineArray
from std.memory import Span, UnsafePointer

from simd_math import fast_exp_softmax_biased, pick_port_unroll

from .dot_products import bf16_panel_dot_to_scalars
from .helpers import BF16Ptr, F32Ptr, W, BW, scale_unrolled


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
    var base_rows: UnsafePointer[Int32, MutUntrackedOrigin]
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
    def row_ptr(self, run_idx: Int) -> UnsafePointer[Int32, MutUntrackedOrigin]:
        return self.base_rows.unsafe_ptr().unsafe_mut_cast[
            True
        ]().unsafe_origin_cast[
            MutUntrackedOrigin
        ]() + Int(self.runs[run_idx].rows_off)


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
def reuse_panel_pu[head_dim: Int, gqa_ratio: Int]() -> Int:
    comptime cap = pick_port_unroll[BW, head_dim]()
    comptime budget_pu = 32 // gqa_ratio
    var pu = 1
    while pu * 2 <= cap and pu * 2 <= budget_pu and pu * gqa_ratio < 8:
        pu *= 2
    return pu


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
    debug_assert(num_q % gqa_ratio == 0, "process_kv_tile needs whole gqa groups")
    comptime PU = reuse_panel_pu[head_dim, gqa_ratio]()

    var slots = InlineArray[Int, TILE](uninitialized=True)
    for t in range(tile_len):
        slots[t] = kv.slot(start_pos, pos + t)

    var num_groups = num_q // gqa_ratio
    var scores_mat = InlineArray[Float32, TILE * gqa_ratio](uninitialized=True)
    var weights_mat = InlineArray[Float32, TILE * gqa_ratio](uninitialized=True)

    for g in range(num_groups):
        var base_q = g * gqa_ratio
        var head_off = g * head_dim

        var group_q = InlineArray[BF16Ptr, gqa_ratio](uninitialized=True)
        comptime for r in range(gqa_ratio):
            group_q[r] = q_ptrs[base_q + r]

        for t in range(tile_len):
            var k_head = k_base + slots[t] * kv_stride + head_off
            var sc = bf16_panel_dot_to_scalars[cols=head_dim, port_unroll=PU](
                k_head, group_q)
            comptime for r in range(gqa_ratio):
                scores_mat[t * gqa_ratio + r] = sc[r]

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
            var v_head = v_base + slots[t] * kv_stride + head_off
            for j in range(0, head_dim, W):
                var vv = (v_head + j).load[width=W]().cast[DType.float32]()
                comptime for r in range(gqa_ratio):
                    var aptr = acc_ptrs[base_q + r] + j
                    aptr.store(vv.fma(
                        SIMD[DType.float32, W](weights_mat[t * gqa_ratio + r]),
                        aptr.load[width=W]()))
