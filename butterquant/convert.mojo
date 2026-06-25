from std.memory import UnsafePointer

from simd_math import has_avx512_bf16
from butterquant.types import BF16Ptr


@always_inline
def store_bf16[width: Int](
    v: SIMD[DType.float32, width], dst: BF16Ptr,
):
    comptime if has_avx512_bf16():
        dst.store(v.cast[DType.bfloat16]())
    else:
        var bits = v.to_bits().cast[DType.uint32]()
        var rne = (bits + ((bits >> 16) & SIMD[DType.uint32, width](1))
            + SIMD[DType.uint32, width](0x7FFF))
        dst.bitcast[UInt16]().store((rne >> 16).cast[DType.uint16]())


@always_inline
def store_out[Out: DType, width: Int](
    v: SIMD[DType.float32, width],
    dst: UnsafePointer[Scalar[Out], MutUntrackedOrigin],
):
    comptime if Out == DType.bfloat16:
        store_bf16[width](v, dst.bitcast[Scalar[DType.bfloat16]]())
    else:
        dst.store(v.cast[Out]())
