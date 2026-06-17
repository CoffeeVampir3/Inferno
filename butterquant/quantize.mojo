from butterquant.kernels import quant_segment
from butterquant.types import F32Ptr, I8Ptr, WF


@always_inline
def quantize_activation_per_row(
    work: F32Ptr, qi: I8Ptr, sa: F32Ptr, cols: Int,
):
    quant_segment[False](work, qi, sa, cols)


@always_inline
def quantize_activation_per_block[block: Int](
    work: F32Ptr, qi: I8Ptr, sa: F32Ptr, cols: Int,
):
    comptime assert block % WF == 0, (
        "FWHT block must be a multiple of the f32 SIMD width")
    debug_assert(cols % block == 0,
        "quantize_activation_per_block: cols must be block-aligned")
    var nb = cols // block
    for b in range(nb):
        var off = b * block
        quant_segment[False](work + off, qi + off, sa + b, block)
