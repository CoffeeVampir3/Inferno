from simd_math.ops import sqrt
from kernels.rmsnorm import rms_reduce_row

from butterquant.fwht import fwht_row
from butterquant.quantize import (
    quantize_activation_per_block, quantize_activation_per_row,
)
from butterquant.types import BF16Ptr, F32Ptr, I8Ptr, WF


@always_inline
def prepare_norm_activation[
    per_block: Bool, hidden: Int, block: Int, sqrt_n: Float32, n_eps: Float32,
](
    src: BF16Ptr, gamma: BF16Ptr, x_i8: I8Ptr, sa: F32Ptr,
    row_workspace: F32Ptr,
):
    comptime assert hidden % WF == 0, (
        "hidden must be a multiple of the f32 SIMD width")
    comptime assert hidden % block == 0, (
        "hidden must be block-aligned")

    var sum_sq = rms_reduce_row[hidden](src)
    var inv_rms = sqrt_n / sqrt[DType.float32, 1](sum_sq + n_eps)
    var fr = SIMD[DType.float32, WF](inv_rms)
    var k = 0
    while k + WF <= hidden:
        var x = (src + k).load[width=WF]().cast[DType.float32]()
        var g = (gamma + k).load[width=WF]().cast[DType.float32]()
        (row_workspace + k).store(x * fr * g)
        k += WF

    fwht_row[block](row_workspace, hidden)
    comptime if per_block:
        quantize_activation_per_block[block](row_workspace, x_i8, sa, hidden)
    else:
        quantize_activation_per_row(row_workspace, x_i8, sa, hidden)


@always_inline
def prepare_head_activation[
    hidden: Int, block: Int, sqrt_n: Float32, n_eps: Float32,
](
    src: BF16Ptr, gamma: BF16Ptr, x_i8: I8Ptr, sa: F32Ptr,
    row_workspace: F32Ptr,
):
    prepare_norm_activation[True, hidden, block, sqrt_n, n_eps](
        src, gamma, x_i8, sa, row_workspace)


@always_inline
def prepare_norm_activation_per_row[
    hidden: Int, block: Int, sqrt_n: Float32, n_eps: Float32,
](
    src: BF16Ptr, gamma: BF16Ptr, x_i8: I8Ptr, sa: F32Ptr,
    row_workspace: F32Ptr,
):
    prepare_norm_activation[False, hidden, block, sqrt_n, n_eps](
        src, gamma, x_i8, sa, row_workspace)


@always_inline
def prepare_block_activation[
    block: Int, apply_fwht: Bool,
](
    src: BF16Ptr, x_i8: I8Ptr, sa: F32Ptr, row_workspace: F32Ptr, cols: Int,
):
    debug_assert(cols % WF == 0,
        "cols must be a multiple of the f32 SIMD width")
    debug_assert(cols % block == 0, "cols must be block-aligned")

    var k = 0
    while k + WF <= cols:
        (row_workspace + k).store(
            (src + k).load[width=WF]().cast[DType.float32]())
        k += WF

    comptime if apply_fwht:
        fwht_row[block](row_workspace, cols)
    quantize_activation_per_block[block](row_workspace, x_i8, sa, cols)
