from std.memory import UnsafePointer
from std.sys.info import size_of


def mint(size: Int) -> Optional[UnsafePointer[UInt8, MutUntrackedOrigin]]:
    return alloc[UInt8](size)


struct ScratchArena(Movable):
    var base: Optional[UnsafePointer[UInt8, MutUntrackedOrigin]]
    var size: Int
    var offset: Int

    def __init__(out self, size: Int):
        self.size = size
        self.offset = 0
        self.base = mint(size)
        if not self.base:
            self.size = 0

    def __del__(deinit self):
        if self.base:
            self.base.value().free()

    def __bool__(self) -> Bool:
        return self.base != None

    def carve[T: AnyType](mut self, count: Int = 1) -> Optional[UnsafePointer[T, MutUntrackedOrigin]]:
        var need = size_of[T]() * count
        if not self.base or self.offset + need > self.size:
            return None
        var p = (self.base.value() + self.offset).bitcast[T]()
        self.offset += need
        return p


def axpy(dst: UnsafePointer[Int32, MutUntrackedOrigin], src: UnsafePointer[Int32, MutUntrackedOrigin], n: Int):
    comptime W = 4
    var i = 0
    while i + W <= n:
        var a = dst.load[width=W](i)
        var b = src.load[width=W](i)
        dst.store(i, a + b * 2)
        i += W


def main():
    var arena = ScratchArena(256)
    if not arena:
        print("alloc failed")
        return

    var a = arena.carve[Int32](8).value()
    var b = arena.carve[Int32](8).value()
    for i in range(8):
        a[i] = Int32(i)
        b[i] = 10

    axpy(a, b, 8)

    var got = a.load[width=8]()
    print("sum:", Int(got.reduce_add()))
    print("offset:", arena.offset)
