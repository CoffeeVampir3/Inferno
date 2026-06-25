from std.memory import UnsafePointer
from std.sys.info import simd_width_of


comptime F32Ptr = UnsafePointer[Float32, MutUntrackedOrigin]
comptime BF16Ptr = UnsafePointer[BFloat16, MutUntrackedOrigin]
comptime I8Ptr = UnsafePointer[Int8, MutUntrackedOrigin]
comptime WF = simd_width_of[DType.float32]()
comptime WI = simd_width_of[DType.int32]()
