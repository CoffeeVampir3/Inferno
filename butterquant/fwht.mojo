from std.collections import InlineArray
from std.math import iota
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from simd_math.matrixops import butterfly_shuffle, log2
from simd_math.ops import sqrt


comptime PtrF32 = UnsafePointer[Float32, MutUntrackedOrigin]


def fwht_width[T: DType, block: Int]() -> Int:
    comptime hw = simd_width_of[T]()
    comptime if block <= hw:
        return block
    else:
        return hw


@always_inline
def fwht_apply[T: DType, block: Int](
    mut r: InlineArray[
        SIMD[T, fwht_width[T, block]()],
        block // fwht_width[T, block](),
    ],
    ):
    comptime width = fwht_width[T, block]()
    comptime regs = block // width
    comptime stages = log2[block]()

    comptime for stage in range(stages):
        comptime stride = 1 << stage
        comptime if stride < width:
            comptime mask = butterfly_shuffle[width, stride]()
            var lane = iota[DType.int32, width]()
            var bit = (lane >> SIMD[DType.int32, width](Int32(stage))) & SIMD[
                DType.int32, width](1)
            var sign = SIMD[T, width](1) - bit.cast[T]() * SIMD[T, width](2)
            comptime for i in range(regs):
                var partner = r[i].shuffle[mask=mask](r[i])
                r[i] = r[i].fma(sign, partner)
        else:
            comptime reg_stride = stride // width
            comptime num_groups = regs // (2 * reg_stride)
            comptime for g in range(num_groups):
                comptime for j in range(reg_stride):
                    comptime a_idx = g * 2 * reg_stride + j
                    comptime b_idx = a_idx + reg_stride
                    var a_val = r[a_idx]
                    var b_val = r[b_idx]
                    r[a_idx] = a_val + b_val
                    r[b_idx] = a_val - b_val

    var sc = Scalar[T](1.0 / Float64(sqrt[T, 1](Scalar[T](block))))
    comptime for i in range(regs):
        r[i] = r[i] * sc


@always_inline
def fwht_block[block: Int](buf: PtrF32):
    comptime width = fwht_width[DType.float32, block]()
    comptime regs = block // width
    var r = InlineArray[SIMD[DType.float32, width], regs](
        fill=SIMD[DType.float32, width](0))
    comptime for i in range(regs):
        r[i] = (buf + i * width).load[width=width]()
    fwht_apply[DType.float32, block](r)
    comptime for i in range(regs):
        (buf + i * width).store(r[i])


@always_inline
def fwht_row[block: Int](buf: PtrF32, cols: Int):
    for b in range(cols // block):
        fwht_block[block](buf + b * block)
