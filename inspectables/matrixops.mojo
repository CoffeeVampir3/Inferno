from std.collections import InlineArray
from std.memory import UnsafePointer
from std.utils import IndexList
from std.math import iota
from std.benchmark import keep


def is_power_of_two[N: Int]() -> Bool:
    return N > 0 and (N & (N - 1)) == 0


def log2[N: Int]() -> Int:
    comptime assert is_power_of_two[N](), "log2 requires a positive power-of-two input"
    comptime if N == 1:
        return 0
    else:
        return 1 + log2[N // 2]()


def bit_reverse[bits: Int, x: Int]() -> Int:
    comptime if bits == 0:
        return 0
    else:
        comptime lsb = x & 1
        comptime rest = x >> 1
        return (lsb << (bits - 1)) | bit_reverse[bits - 1, rest]()


def interleave_idx[N: Int, i: Int, stride: Int, high: Bool]() -> Int:
    comptime half = N // 2
    comptime src_offset = half if high else 0
    comptime pair = i // (2 * stride)
    comptime within = i % (2 * stride)
    comptime if within < stride:
        return src_offset + pair * stride + within
    else:
        return N + src_offset + pair * stride + (within - stride)


def interleave_mask[N: Int, stride: Int, high: Bool]() -> IndexList[N]:
    comptime assert is_power_of_two[N](), "interleave_mask requires power-of-two N"
    comptime assert stride > 0 and stride < N, "interleave_mask stride must be in [1, N)"
    var result = IndexList[N]()
    comptime for i in range(N):
        result[i] = interleave_idx[N, i, stride, high]()
    return result


@always_inline
def simd_interleave[T: DType, N: Int, stride: Int, high: Bool](
    a: SIMD[T, N], b: SIMD[T, N],
) -> SIMD[T, N]:
    comptime assert is_power_of_two[N](), "simd_interleave requires power-of-two N"
    comptime assert stride > 0 and stride < N, "simd_interleave stride must be in [1, N)"
    comptime idx = interleave_mask[N, stride, high]()
    return a.shuffle[mask=idx](b)


@always_inline
def transpose_rows[T: DType, N: Int, dst_origin: MutOrigin](
    mut rows: InlineArray[SIMD[T, N], N],
    dst: UnsafePointer[Scalar[T], dst_origin], dst_stride: Int,
):
    comptime assert is_power_of_two[N](), "transpose_rows requires power-of-two N"
    comptime num_stages = log2[N]()
    comptime for stage in range(num_stages):
        comptime stride = 1 << stage
        comptime groups = N // (2 * stride)
        comptime for g in range(groups):
            comptime for j in range(stride):
                comptime idx0 = g * 2 * stride + j
                comptime idx1 = idx0 + stride
                var lo = simd_interleave[T, N, stride, False](rows[idx0], rows[idx1])
                var hi = simd_interleave[T, N, stride, True](rows[idx0], rows[idx1])
                rows[idx0] = lo
                rows[idx1] = hi

    comptime for i in range(N):
        comptime j = bit_reverse[num_stages, i]()
        comptime if i < j:
            var tmp = rows[i]
            rows[i] = rows[j]
            rows[j] = tmp

    comptime for i in range(N):
        (dst + i * dst_stride).store(rows[i])


@always_inline
def transpose_generic[T: DType, N: Int, dst_origin: MutOrigin](
    src: UnsafePointer[Scalar[T], _], src_stride: Int,
    dst: UnsafePointer[Scalar[T], dst_origin], dst_stride: Int,
    mut scratch: InlineArray[SIMD[T, N], N],
):
    comptime assert is_power_of_two[N](), "transpose_generic requires power-of-two N"
    comptime for i in range(N):
        scratch[i] = (src + i * src_stride).load[width=N]()
    transpose_rows[T, N](scratch, dst, dst_stride)


def butterfly_partner[i: Int, stride: Int]() -> Int:
    return i ^ stride


def butterfly_shuffle[width: Int, stride: Int]() -> IndexList[width]:
    comptime assert is_power_of_two[width](), "butterfly_shuffle requires power-of-two width"
    comptime assert stride > 0 and stride < width, "butterfly_shuffle stride must be in [1, width)"
    var result = IndexList[width]()
    comptime for i in range(width):
        result[i] = butterfly_partner[i, stride]()
    return result


@always_inline
def reduce_argmax[T: DType, width: Int, regs: Int](
    mut values: InlineArray[SIMD[T, width], regs],
    mut indices: InlineArray[SIMD[DType.int32, width], regs],
) -> Tuple[Scalar[T], Int32]:
    comptime assert is_power_of_two[regs](), "reduce_argmax requires power-of-two regs"
    comptime assert is_power_of_two[width](), "reduce_argmax requires power-of-two width"
    comptime stages_across = log2[regs]()
    comptime stages_within = log2[width]()

    comptime for stage in range(stages_across):
        comptime stride = 1 << stage
        comptime groups = regs // (2 * stride)
        comptime for g in range(groups):
            comptime for j in range(stride):
                comptime a = g * 2 * stride + j
                comptime b = a + stride
                var mask = values[a].ge(values[b])
                values[a] = mask.select(values[a], values[b])
                indices[a] = mask.select(indices[a], indices[b])

    comptime for stage in range(stages_within):
        comptime stride = 1 << stage
        comptime shuf_mask = butterfly_shuffle[width, stride]()
        var partner_v = values[0].shuffle[mask=shuf_mask](values[0])
        var partner_i = indices[0].shuffle[mask=shuf_mask](indices[0])
        var cmp = values[0].ge(partner_v)
        values[0] = cmp.select(values[0], partner_v)
        indices[0] = cmp.select(indices[0], partner_i)

    return (values[0][0], indices[0][0])


@always_inline
def reduce_top_k[T: DType, width: Int, regs: Int, k: Int](
    source_values: InlineArray[SIMD[T, width], regs],
    source_indices: InlineArray[SIMD[DType.int32, width], regs],
    sentinel: Scalar[T],
    mut out_indices: InlineArray[Int, k],
    mut out_values: InlineArray[Scalar[T], k],
):
    comptime assert is_power_of_two[regs](), "reduce_top_k requires power-of-two regs"
    comptime assert is_power_of_two[width](), "reduce_top_k requires power-of-two width"
    comptime assert k >= 0 and k <= regs * width, "reduce_top_k requires 0 <= k <= regs * width"
    var work_v = InlineArray[SIMD[T, width], regs](uninitialized=True)
    var work_i = InlineArray[SIMD[DType.int32, width], regs](uninitialized=True)
    comptime for r in range(regs):
        work_v[r] = source_values[r]
        work_i[r] = source_indices[r]

    var lane_iota = iota[DType.int32, width]()
    var sentinel_vec = SIMD[T, width](sentinel)

    for sel in range(k):
        var tv = InlineArray[SIMD[T, width], regs](uninitialized=True)
        var ti = InlineArray[SIMD[DType.int32, width], regs](uninitialized=True)
        comptime for r in range(regs):
            tv[r] = work_v[r]
            ti[r] = work_i[r]

        var winner = reduce_argmax[T, width, regs](tv, ti)
        out_values[sel] = winner[0]
        var winner_idx = Int(winner[1])
        out_indices[sel] = winner_idx

        var wr = winner_idx // width
        var wl = winner_idx - wr * width
        var lane_eq = lane_iota.eq(SIMD[DType.int32, width](Int32(wl)))
        work_v[wr] = lane_eq.select(sentinel_vec, work_v[wr])


@always_inline
def port_unroll_for[count: Int]() -> Int:
    comptime assert count > 0, "port_unroll_for requires positive count"
    return 8 if count >= 8 else 4 if count >= 4 else 2 if count >= 2 else 1


@always_inline
def pick_port_unroll[width: Int, cols: Int]() -> Int:
    comptime assert cols >= width, "pick_port_unroll requires cols >= width"
    comptime if cols % (8 * width) == 0:
        return 8
    elif cols % (4 * width) == 0:
        return 4
    elif cols % (2 * width) == 0:
        return 2
    else:
        return 1


@always_inline
def runtime_pick_port_unroll(width: Int, cols: Int) -> Int:
    if cols % (8 * width) == 0:
        return 8
    elif cols % (4 * width) == 0:
        return 4
    elif cols % (2 * width) == 0:
        return 2
    return 1


@always_inline
def tree_merge_accs[T: DType, width: Int, port_unroll: Int, //](
    mut accs: InlineArray[SIMD[T, width], port_unroll],
) -> SIMD[T, width]:
    comptime assert is_power_of_two[port_unroll](), (
        "tree_merge_accs requires power-of-two port_unroll"
    )
    comptime for stride in range(1, port_unroll):
        comptime if (stride & (stride - 1)) == 0:
            comptime for i in range(0, port_unroll, 2 * stride):
                accs[i] += accs[i + stride]
    return accs[0]


@always_inline
def tree_reduce_accs[T: DType, width: Int, port_unroll: Int, //](
    mut accs: InlineArray[SIMD[T, width], port_unroll],
) -> Scalar[T]:
    return tree_merge_accs(accs).reduce_add()


@no_inline
def probe_interleave_lo(
    a: SIMD[DType.float32, 8], b: SIMD[DType.float32, 8]
) -> SIMD[DType.float32, 8]:
    return simd_interleave[DType.float32, 8, 1, False](a, b)


@no_inline
def probe_interleave_hi(
    a: SIMD[DType.float32, 8], b: SIMD[DType.float32, 8]
) -> SIMD[DType.float32, 8]:
    return simd_interleave[DType.float32, 8, 1, True](a, b)


@no_inline
def probe_transpose_i32_8[o: MutOrigin](
    src: UnsafePointer[Int32, _], dst: UnsafePointer[Int32, o]
):
    var scratch = InlineArray[SIMD[DType.int32, 8], 8](uninitialized=True)
    transpose_generic[DType.int32, 8](src, 8, dst, 8, scratch)


@no_inline
def probe_reduce_argmax(p: UnsafePointer[Float32, _]) -> Float32:
    comptime W = 8
    comptime R = 4
    var vals = InlineArray[SIMD[DType.float32, W], R](uninitialized=True)
    var idxs = InlineArray[SIMD[DType.int32, W], R](uninitialized=True)
    comptime for r in range(R):
        vals[r] = (p + r * W).load[width=W]()
        idxs[r] = iota[DType.int32, W]() + SIMD[DType.int32, W](r * W)
    var res = reduce_argmax[DType.float32, W, R](vals, idxs)
    keep(res[1])
    return res[0]


@no_inline
def probe_reduce_top_k(
    p: UnsafePointer[Float32, _],
    mut out_i: InlineArray[Int, 3],
    mut out_v: InlineArray[Float32, 3],
):
    comptime W = 8
    comptime R = 4
    comptime K = 3
    var sv = InlineArray[SIMD[DType.float32, W], R](uninitialized=True)
    var si = InlineArray[SIMD[DType.int32, W], R](uninitialized=True)
    comptime for r in range(R):
        sv[r] = (p + r * W).load[width=W]()
        si[r] = iota[DType.int32, W]() + SIMD[DType.int32, W](r * W)
    reduce_top_k[DType.float32, W, R, K](sv, si, -3.0e38, out_i, out_v)


@no_inline
def probe_tree_reduce(p: UnsafePointer[Float32, _]) -> Float32:
    comptime W = 8
    comptime PU = 4
    var accs = InlineArray[SIMD[DType.float32, W], PU](uninitialized=True)
    comptime for i in range(PU):
        accs[i] = (p + i * W).load[width=W]()
    return tree_reduce_accs(accs)


@no_inline
def probe_tree_merge(p: UnsafePointer[Float32, _]) -> SIMD[DType.float32, 8]:
    comptime W = 8
    comptime PU = 4
    var accs = InlineArray[SIMD[DType.float32, W], PU](uninitialized=True)
    comptime for i in range(PU):
        accs[i] = (p + i * W).load[width=W]()
    return tree_merge_accs(accs)


def main():
    var fbuf = InlineArray[Float32, 32](uninitialized=True)
    for i in range(32):
        fbuf[i] = Float32(i) * 1.5 - 7.0
    var fp = UnsafePointer(to=fbuf[0])

    var a = fp.load[width=8]()
    var b = (fp + 8).load[width=8]()
    keep(probe_interleave_lo(a, b)[0])
    keep(probe_interleave_hi(a, b)[0])

    keep(probe_reduce_argmax(fp))

    var out_i = InlineArray[Int, 3](fill=0)
    var out_v = InlineArray[Float32, 3](fill=0.0)
    probe_reduce_top_k(fp, out_i, out_v)
    keep(out_i[0])
    keep(out_v[0])

    keep(probe_tree_reduce(fp))
    keep(probe_tree_merge(fp)[0])

    var src = InlineArray[Int32, 64](uninitialized=True)
    for i in range(64):
        src[i] = Int32(i)
    var dst = InlineArray[Int32, 64](fill=0)
    probe_transpose_i32_8(UnsafePointer(to=src[0]), UnsafePointer(to=dst[0]))
    keep(dst[0])
