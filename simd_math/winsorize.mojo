from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of


@always_inline
def winsorize_row[
    T: DType, //, cols: Int, q: Float64,
](row: UnsafePointer[Scalar[T], MutAnyOrigin]):
    """Symmetric in-place winsorization of one length-`cols` row at quantile `q`.

    Clamps every component to +/- the q-quantile of the row's absolute values.
    Exact and cheap because `q` is high: pass one streams the row tracking the
    `cols - floor(q*(cols-1))` largest magnitudes in an ascending inline buffer
    (a small comptime-fixed size), which are exactly the order statistics that
    bracket the quantile; pass two clamps with linear interpolation between the
    bottom two, matching torch.quantile's default.
    """
    comptime g = q * Float64(cols - 1)
    comptime lo = Int(g)
    comptime top_n = cols - lo
    comptime assert top_n >= 2 and top_n <= cols, "winsorize: quantile out of range for cols"
    comptime frac = g - Float64(lo)
    comptime width = simd_width_of[T]()

    var buf = InlineArray[Scalar[T], top_n](uninitialized=True)
    comptime for i in range(top_n):
        buf[i] = Scalar[T](-1)

    var off = 0
    while off + width <= cols:
        var v = (row + off).load[width=width]()
        var av = max(v, -v)
        if av.gt(SIMD[T, width](buf[0])).reduce_or():
            comptime for lane in range(width):
                var a = av[lane]
                if a > buf[0]:
                    var p = 0
                    while p + 1 < top_n and a > buf[p + 1]:
                        buf[p] = buf[p + 1]
                        p += 1
                    buf[p] = a
        off += width
    while off < cols:
        var x = (row + off).load[width=1]()
        var a = max(x, -x)
        if a > buf[0]:
            var p = 0
            while p + 1 < top_n and a > buf[p + 1]:
                buf[p] = buf[p + 1]
                p += 1
            buf[p] = a
        off += 1

    var thr = buf[0] + Scalar[T](frac) * (buf[1] - buf[0])
    var pos = SIMD[T, width](thr)
    var neg = SIMD[T, width](-thr)
    var off2 = 0
    while off2 + width <= cols:
        var v = (row + off2).load[width=width]()
        (row + off2).store(min(max(v, neg), pos))
        off2 += width
    while off2 < cols:
        var x = (row + off2).load[width=1]()
        (row + off2).store(min(max(x, -thr), thr))
        off2 += 1


def winsorize_rows[
    T: DType, //, cols: Int, q: Float64,
](data: UnsafePointer[Scalar[T], MutAnyOrigin], rows: Int):
    """Apply `winsorize_row` independently to each of `rows` contiguous rows."""
    for r in range(rows):
        winsorize_row[cols=cols, q=q](data + r * cols)
