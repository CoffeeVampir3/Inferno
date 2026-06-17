from std.memory import UnsafePointer, memcpy

struct CpuMask[size: Int = 128](Copyable, TrivialRegisterPassable):
    """Linux CPU affinity mask.

    `size` is the number of bytes in the mask, matching the cpusetsize
    argument expected by sched_setaffinity(2). The mask can represent
    `size * 8` CPU IDs.
    """
    var bytes: SIMD[DType.uint8, Self.size]

    def __init__(out self):
        comptime assert Self.size > 0, "CpuMask size must be positive"
        self.bytes = SIMD[DType.uint8, Self.size](UInt8(0))

    def set(mut self, cpu_id: Int):
        if not Self.contains_cpu(cpu_id):
            return
        var byte_idx = cpu_id >> 3
        var bit_idx = cpu_id & 7
        self.bytes[byte_idx] |= UInt8(1 << bit_idx)

    def clear(mut self, cpu_id: Int):
        if not Self.contains_cpu(cpu_id):
            return
        var byte_idx = cpu_id >> 3
        var bit_idx = cpu_id & 7
        self.bytes[byte_idx] &= ~UInt8(1 << bit_idx)

    def test(ref self, cpu_id: Int) -> Bool:
        if not Self.contains_cpu(cpu_id):
            return False
        var byte_idx = cpu_id >> 3
        var bit_idx = cpu_id & 7
        return (self.bytes[byte_idx] & UInt8(1 << bit_idx)) != 0

    def clear_all(mut self):
        self.bytes = SIMD[DType.uint8, Self.size](UInt8(0))

    def set_all(mut self):
        self.bytes = SIMD[DType.uint8, Self.size](UInt8(0xFF))

    def count(ref self) -> Int:
        return self.bytes.reduce_bit_count()

    @always_inline
    def unsafe_address(ref self) -> Int:
        """Address of the contiguous mask bytes for syscall read-only use."""
        return Int(UnsafePointer(to=self.bytes))

    def copy_to[origin: MutOrigin](ref self, dest: UnsafePointer[UInt8, origin]):
        memcpy(dest=dest, src=UnsafePointer(to=self.bytes).bitcast[UInt8](), count=Self.size)

    @staticmethod
    def from_cpu_list(cpu_ids: List[Int]) -> Self:
        var mask = Self()
        for cpu in cpu_ids:
            mask.set(cpu)
        return mask

    @staticmethod
    def byte_size() -> Int:
        return Self.size

    @staticmethod
    def bit_capacity() -> Int:
        return Self.size * 8

    @staticmethod
    def contains_cpu(cpu_id: Int) -> Bool:
        return cpu_id >= 0 and cpu_id < Self.bit_capacity()
