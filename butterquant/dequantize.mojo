from butterquant.convert import store_bf16
from butterquant.types import BF16Ptr, F32Ptr, I8Ptr, WF


@always_inline
def dequant_weight_row_per_block[block: Int](
    qi: I8Ptr, scales: F32Ptr, dst: F32Ptr, cols: Int,
):
    comptime assert block % WF == 0, (
        "FWHT block must be a multiple of the f32 SIMD width")
    debug_assert(cols % block == 0,
        "dequant_weight_row_per_block: cols must be block-aligned")
    var nb = cols // block
    for b in range(nb):
        var off = b * block
        var sb = SIMD[DType.float32, WF](scales[b])
        var k = 0
        while k + WF <= block:
            var v = (qi + off + k).load[width=WF]().cast[DType.float32]()
            (dst + off + k).store(v * sb)
            k += WF


@always_inline
def scale_cast_row[hidden: Int, scale: Float64](work: F32Ptr, dst: BF16Ptr):
    comptime assert hidden % WF == 0, (
        "hidden must be a multiple of the f32 SIMD width")
    var f = SIMD[DType.float32, WF](Float32(scale))
    var k = 0
    while k + WF <= hidden:
        store_bf16[WF]((work + k).load[width=WF]() * f, dst + k)
        k += WF


@always_inline
def zero_row[hidden: Int](dst: BF16Ptr):
    comptime assert hidden % WF == 0, (
        "hidden must be a multiple of the f32 SIMD width")
    var k = 0
    while k + WF <= hidden:
        (dst + k).store(SIMD[DType.bfloat16, WF](0))
        k += WF
