from std.collections import InlineArray
from std.sys import llvm_intrinsic
from std.sys.info import size_of

from simd_math import pick_port_unroll, tree_reduce_accs, has_avx512_bf16
from .helpers import BF16Ptr, F32Ptr, W, BW


@always_inline
def bf16_pair_dot(
    var acc: SIMD[DType.float32, W],
    var a: SIMD[DType.bfloat16, BW],
    var b: SIMD[DType.bfloat16, BW],
) -> SIMD[DType.float32, W]:
    """BF16xBF16 dot-into-F32 accumulator over the target SIMD width.

    On AVX-512BF16 emits one VDPBF16PS for the native SIMD width.
    Otherwise falls back to deinterleave + cast + FMAs with the same
    per-lane semantics.
    """
    comptime vector_bits = size_of[SIMD[DType.float32, W]]() * 8
    comptime if has_avx512_bf16():
        return llvm_intrinsic[
            "llvm.x86.avx512bf16.dpbf16ps." + String(vector_bits),
            SIMD[DType.float32, W],
        ](acc, a, b)
    else:
        var ae_ao = a.deinterleave()
        var be_bo = b.deinterleave()
        var ae = rebind[SIMD[DType.bfloat16, W]](ae_ao[0]).cast[DType.float32]()
        var ao = rebind[SIMD[DType.bfloat16, W]](ae_ao[1]).cast[DType.float32]()
        var be = rebind[SIMD[DType.bfloat16, W]](be_bo[0]).cast[DType.float32]()
        var bo = rebind[SIMD[DType.bfloat16, W]](be_bo[1]).cast[DType.float32]()
        var inner = ae.fma(be, acc)
        return ao.fma(bo, inner)


@always_inline
def bf16_panel_dot[
    panel: Int, port_unroll: Int, //,
    cols: Int,
](
    weight_row: BF16Ptr,
    read x_rows: InlineArray[BF16Ptr, panel],
    mut accs: InlineArray[
        InlineArray[SIMD[DType.float32, W], port_unroll], panel,
    ],
):
    comptime STRIDE = port_unroll * BW
    for i in range(cols // STRIDE):
        comptime for p in range(port_unroll):
            var off = i * STRIDE + p * BW
            var w_v = (weight_row + off).load[width=BW]()
            comptime for r in range(panel):
                var x_v = (x_rows[r] + off).load[width=BW]()
                accs[r][p] = bf16_pair_dot(accs[r][p], x_v, w_v)


@always_inline
def bf16_panel_dot_runtime[
    panel: Int, port_unroll: Int, //,
](
    weight_row: BF16Ptr,
    read x_rows: InlineArray[BF16Ptr, panel],
    mut accs: InlineArray[
        InlineArray[SIMD[DType.float32, W], port_unroll], panel,
    ],
    cols: Int,
):
    """Strip-mined panel dot with a RUNTIME contraction `cols` and a comptime
    `port_unroll`. The SIMD width and unroll stay comptime; only the block-count
    trip is runtime. Used by the column-sharded matmuls where cols = dim//degree.
    A `cols` that is not a multiple of `port_unroll * BW` is finished by a BW-wide
    remainder and a scalar remainder, both folded into lane 0 of accs[r][0]."""
    comptime STRIDE = port_unroll * BW
    var blocks = cols // STRIDE
    for i in range(blocks):
        comptime for p in range(port_unroll):
            var off = i * STRIDE + p * BW
            var w_v = (weight_row + off).load[width=BW]()
            comptime for r in range(panel):
                var x_v = (x_rows[r] + off).load[width=BW]()
                accs[r][p] = bf16_pair_dot(accs[r][p], x_v, w_v)

    var tail = blocks * STRIDE
    while tail + BW <= cols:
        var w_v = (weight_row + tail).load[width=BW]()
        comptime for r in range(panel):
            var x_v = (x_rows[r] + tail).load[width=BW]()
            accs[r][0] = bf16_pair_dot(accs[r][0], x_v, w_v)
        tail += BW

    if tail < cols:
        comptime for r in range(panel):
            var s = Float32(0)
            for k in range(tail, cols):
                s += (
                    x_rows[r][k].cast[DType.float32]()
                    * weight_row[k].cast[DType.float32]()
                )
            var v = accs[r][0]
            v[0] = v[0] + s
            accs[r][0] = v


@always_inline
def bf16_panel_dot_to_scalars_runtime[
    panel: Int, //,
    port_unroll: Int,
](
    weight_row: BF16Ptr,
    read x_rows: InlineArray[BF16Ptr, panel],
    cols: Int,
) -> InlineArray[Float32, panel]:
    var accs = InlineArray[
        InlineArray[SIMD[DType.float32, W], port_unroll], panel,
    ](
        fill=InlineArray[SIMD[DType.float32, W], port_unroll](
            fill=SIMD[DType.float32, W](0),
        ),
    )
    bf16_panel_dot_runtime[port_unroll=port_unroll](weight_row, x_rows, accs, cols)
    return panel_accs_to_scalars(accs)


@always_inline
def panel_accs_to_scalars[
    panel: Int, port_unroll: Int, //,
](
    read accs: InlineArray[
        InlineArray[SIMD[DType.float32, W], port_unroll], panel,
    ],
) -> InlineArray[Float32, panel]:
    var out = InlineArray[Float32, panel](uninitialized=True)
    comptime for r in range(panel):
        var s = SIMD[DType.float32, W](0)
        comptime for p in range(port_unroll):
            s += accs[r][p]
        out[r] = s.reduce_add()
    return out


@always_inline
def bf16_panel_dot_to_scalars[
    panel: Int, //,
    cols: Int, port_unroll: Int,
](
    weight_row: BF16Ptr,
    read x_rows: InlineArray[BF16Ptr, panel],
) -> InlineArray[Float32, panel]:
    var accs = InlineArray[
        InlineArray[SIMD[DType.float32, W], port_unroll], panel,
    ](
        fill=InlineArray[SIMD[DType.float32, W], port_unroll](
            fill=SIMD[DType.float32, W](0),
        ),
    )
    bf16_panel_dot[cols=cols](weight_row, x_rows, accs)
    return panel_accs_to_scalars(accs)


@always_inline
def dot_f32_bf16_into_accs[
    port_unroll: Int, //,
    cols: Int,
](
    x: F32Ptr,
    w: BF16Ptr,
    mut accs: InlineArray[SIMD[DType.float32, W], port_unroll],
):
    comptime STRIDE = port_unroll * W
    for i in range(cols // STRIDE):
        comptime for p in range(port_unroll):
            var off = i * STRIDE + p * W
            var xv = (x + off).load[width=W]()
            var wv = (w + off).load[width=W]().cast[DType.float32]()
            accs[p] = xv.fma(wv, accs[p])


@always_inline
def dot_to_scalar[cols: Int](
    x: BF16Ptr,
    weight_row: BF16Ptr,
) -> Float32:
    comptime PU = pick_port_unroll[BW, cols]()
    var x_rows = InlineArray[BF16Ptr, 1](uninitialized=True)
    x_rows[0] = x
    var scalars = bf16_panel_dot_to_scalars[
        cols=cols, port_unroll=PU,
    ](weight_row, x_rows)
    return scalars[0]


@always_inline
def dot_to_scalar[cols: Int](
    x: F32Ptr,
    weight_row: BF16Ptr,
) -> Float32:
    comptime PU = pick_port_unroll[W, cols]()
    var accs = InlineArray[SIMD[DType.float32, W], PU](
        fill=SIMD[DType.float32, W](0))
    dot_f32_bf16_into_accs[cols=cols](x, weight_row, accs)
    return tree_reduce_accs(accs)
