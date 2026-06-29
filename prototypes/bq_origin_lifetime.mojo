from std.collections import InlineArray
from std.memory import UnsafePointer


@always_inline
def fill_seq[origin: MutOrigin](p: UnsafePointer[Int32, origin], n: Int):
    for i in range(n):
        (p + i).store(Int32((i + 1) * (i + 1)))


@always_inline
def sum_seq[origin: MutOrigin](p: UnsafePointer[Int32, origin], n: Int) -> Int32:
    var s = Int32(0)
    for i in range(n):
        s += (p + i).load()
    return s


def clean_tracked() -> Int32:
    var c = InlineArray[Int32, 8](uninitialized=True)
    var cp = c.unsafe_ptr()
    fill_seq(cp, 8)
    return sum_seq(cp, 8)


def severed_untracked() -> Int32:
    var c = InlineArray[Int32, 8](uninitialized=True)
    var cp = UnsafePointer(to=c).bitcast[Int32]().unsafe_origin_cast[
        MutUntrackedOrigin]()
    fill_seq(cp, 8)
    return sum_seq(cp, 8)


def severed_with_keepalive() -> Int32:
    var c = InlineArray[Int32, 8](uninitialized=True)
    var cp = UnsafePointer(to=c).bitcast[Int32]().unsafe_origin_cast[
        MutUntrackedOrigin]()
    fill_seq(cp, 8)
    var r = sum_seq(cp, 8)
    _ = c
    return r


def main():
    var want = Int32(204)
    print("expected sum 1+4+9+16+25+36+49+64 =", want)
    print("clean_tracked          =", clean_tracked())
    print("severed_untracked      =", severed_untracked())
    print("severed_with_keepalive =", severed_with_keepalive())
