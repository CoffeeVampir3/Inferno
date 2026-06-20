from std.collections import InlineArray

from .helpers import BF16Ptr, W, BW
from .dot_products import bf16_pair_dot


comptime ZMM = 32


@always_inline
def panel_accs[MR: Int, NC: Int, GROUPS: Int]() -> Int:
    return GROUPS * MR * NC


@always_inline
def panel_reserve[MR: Int, GROUPS: Int]() -> Int:
    return MR + GROUPS + 1


@always_inline
def bf16_panel_fits[MR: Int, NC: Int, GROUPS: Int]() -> Bool:
    return panel_accs[MR, NC, GROUPS]() + panel_reserve[MR, GROUPS]() <= ZMM


@always_inline
def pick_nc[MR: Int, GROUPS: Int, want: Int]() -> Int:
    """Largest NC in [1, want] whose GROUPS*MR*NC accumulators (+ reserve) fit
    the zmm budget. bf16_panel_fits is monotonic in NC, so solve in closed form
    rather than loop (a runtime trip variable can't index a comptime param)."""
    var nc_max = (ZMM - panel_reserve[MR, GROUPS]()) // (GROUPS * MR)
    if nc_max < 1:
        return 1
    if nc_max < want:
        return nc_max
    return want


@always_inline
def bf16_microtile[
    MR: Int, NC: Int, GROUPS: Int, contraction: Int,
](
    read x_rows: InlineArray[BF16Ptr, MR],
    read group_base: InlineArray[BF16Ptr, GROUPS],
) -> InlineArray[Float32, GROUPS * MR * NC]:
    """Register tile: MR token rows x NC output columns x GROUPS weight regions,
    contracted over a comptime dim. Each x chunk is loaded once and fed into all
    GROUPS*NC weight streams, so the activation read is amortized across every
    output the tile produces. The GROUPS*MR*NC independent accumulators supply
    the ILP that port_unroll provides in the MR x 1 kernels.

    Weight regions are row-major [cols, contraction]; column c of group g is at
    group_base[g] + c*contraction. Returned scalars are laid out at index
    (g*NC + c)*MR + r."""
    comptime assert bf16_panel_fits[MR, NC, GROUPS](), (
        "bf16 panel exceeds the zmm register budget; lower NC")
    comptime assert contraction % BW == 0, (
        "bf16_microtile: contraction must be a multiple of the bf16 SIMD width")

    var accs = InlineArray[SIMD[DType.float32, W], GROUPS * MR * NC](
        fill=SIMD[DType.float32, W](0))

    for i in range(contraction // BW):
        var off = i * BW
        var xv = InlineArray[SIMD[DType.bfloat16, BW], MR](uninitialized=True)
        comptime for r in range(MR):
            xv[r] = (x_rows[r] + off).load[width=BW]()
        comptime for g in range(GROUPS):
            comptime for c in range(NC):
                var wv = (group_base[g] + c * contraction + off).load[width=BW]()
                comptime for r in range(MR):
                    var k = (g * NC + c) * MR + r
                    accs[k] = bf16_pair_dot(accs[k], xv[r], wv)

    var out = InlineArray[Float32, GROUPS * MR * NC](uninitialized=True)
    comptime for idx in range(GROUPS * MR * NC):
        out[idx] = accs[idx].reduce_add()
    return out


@always_inline
def bf16_microtile_runtime[
    MR: Int, NC: Int, GROUPS: Int,
](
    read x_rows: InlineArray[BF16Ptr, MR],
    read group_base: InlineArray[BF16Ptr, GROUPS],
    contraction: Int,
) -> InlineArray[Float32, GROUPS * MR * NC]:
    """Runtime-contraction sibling of bf16_microtile, for column-sharded matmuls
    where the contraction (= dim//degree) is runtime. Full BW chunks ride the
    vector path; a scalar tail folds the final `contraction % BW` lanes into the
    reduced scalar. Column stride equals the runtime contraction (row-major)."""
    comptime assert bf16_panel_fits[MR, NC, GROUPS](), (
        "bf16 panel exceeds the zmm register budget; lower NC")

    var accs = InlineArray[SIMD[DType.float32, W], GROUPS * MR * NC](
        fill=SIMD[DType.float32, W](0))

    var blocks = contraction // BW
    for i in range(blocks):
        var off = i * BW
        var xv = InlineArray[SIMD[DType.bfloat16, BW], MR](uninitialized=True)
        comptime for r in range(MR):
            xv[r] = (x_rows[r] + off).load[width=BW]()
        comptime for g in range(GROUPS):
            comptime for c in range(NC):
                var wv = (group_base[g] + c * contraction + off).load[width=BW]()
                comptime for r in range(MR):
                    var k = (g * NC + c) * MR + r
                    accs[k] = bf16_pair_dot(accs[k], xv[r], wv)

    var out = InlineArray[Float32, GROUPS * MR * NC](uninitialized=True)
    comptime for idx in range(GROUPS * MR * NC):
        out[idx] = accs[idx].reduce_add()

    var tail = blocks * BW
    if tail < contraction:
        comptime for g in range(GROUPS):
            comptime for c in range(NC):
                var w_col = group_base[g] + c * contraction
                comptime for r in range(MR):
                    var s = Float32(0)
                    for kk in range(tail, contraction):
                        s += (
                            x_rows[r][kk].cast[DType.float32]()
                            * w_col[kk].cast[DType.float32]())
                    out[(g * NC + c) * MR + r] += s
    return out
