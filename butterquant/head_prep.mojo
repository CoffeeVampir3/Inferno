from std.collections import InlineArray
from std.math import max

from simd_math.ops import sqrt, quantize_i8
from butterquant.fwht import fwht_apply, fwht_width
from butterquant.types import BF16Ptr, F32Ptr, I8Ptr


@always_inline
def head_inv_rms[head_dim: Int, width: Int, regs: Int, sqrt_n: Float32, n_eps: Float32](
    read r: InlineArray[SIMD[DType.float32, width], regs],
) -> Float32:
    var ssq = r[0] * r[0]
    comptime for ri in range(1, regs):
        ssq = r[ri].fma(r[ri], ssq)
    return sqrt_n / sqrt[DType.float32, 1](ssq.reduce_add() + n_eps)[0]


@always_inline
def absmax_quantize_head[width: Int, regs: Int](
    read r: InlineArray[SIMD[DType.float32, width], regs],
    qi_out: I8Ptr,
) -> Tuple[Float32, Int32]:
    var vmax = SIMD[DType.float32, width](0)
    comptime for ri in range(regs):
        vmax = max(vmax, abs(r[ri]))
    var amax = vmax.reduce_max()
    if amax < Float32(1e-10):
        amax = Float32(1e-10)
    var inv = SIMD[DType.float32, width](Float32(127.0) / amax)
    var qsum = SIMD[DType.int32, width](0)
    comptime for ri in range(regs):
        var qi = quantize_i8[width](r[ri], inv)
        (qi_out + ri * width).store(qi)
        qsum += qi.cast[DType.int32]()
    return (amax, qsum.reduce_add())


@always_inline
def prep_head_qk_i8[
    head_dim: Int, rope_half: Int, pair_stride: Int,
    sqrt_n: Float32, n_eps: Float32,
](
    src: BF16Ptr, gamma: BF16Ptr, cos: F32Ptr, sin: F32Ptr, qi_out: I8Ptr,
) -> Tuple[Float32, Int32]:
    comptime width = fwht_width[DType.float32, head_dim]()
    comptime regs = head_dim // width
    comptime pair_reg_stride = pair_stride // width
    comptime rope_regs = rope_half // width

    var r = InlineArray[SIMD[DType.float32, width], regs](uninitialized=True)
    comptime for ri in range(regs):
        r[ri] = (src + ri * width).load[width=width]().cast[DType.float32]()

    var inv_rms = head_inv_rms[head_dim, width, regs, sqrt_n, n_eps](r)
    var fr = SIMD[DType.float32, width](inv_rms)
    comptime for ri in range(regs):
        var g = (gamma + ri * width).load[width=width]().cast[DType.float32]()
        r[ri] = r[ri] * fr * g

    comptime for ri in range(rope_regs):
        var x_lo = r[ri]
        var x_hi = r[pair_reg_stride + ri]
        var cv = (cos + ri * width).load[width=width]()
        var sv = (sin + ri * width).load[width=width]()
        r[ri] = x_lo * cv - x_hi * sv
        r[pair_reg_stride + ri] = x_hi * cv + x_lo * sv

    fwht_apply[DType.float32, head_dim](r)
    return absmax_quantize_head[width, regs](r, qi_out)


@always_inline
def prep_head_v_i8[
    head_dim: Int, sqrt_n: Float32, n_eps: Float32,
](
    src: BF16Ptr, vi_out: I8Ptr,
) -> Float32:
    comptime width = fwht_width[DType.float32, head_dim]()
    comptime regs = head_dim // width

    var r = InlineArray[SIMD[DType.float32, width], regs](uninitialized=True)
    comptime for ri in range(regs):
        r[ri] = (src + ri * width).load[width=width]().cast[DType.float32]()

    var inv_rms = head_inv_rms[head_dim, width, regs, sqrt_n, n_eps](r)
    var fr = SIMD[DType.float32, width](inv_rms)
    comptime for ri in range(regs):
        r[ri] = r[ri] * fr

    fwht_apply[DType.float32, head_dim](r)
    return absmax_quantize_head[width, regs](r, vi_out)[0]
