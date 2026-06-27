from std.memory import UnsafePointer
from std.math import max
from std.os import abort

from simd_math.ops import quantize_i8, sqrt
from butterquant.convert import store_bf16
from butterquant.fwht import fwht_block, fwht_row
from butterquant.types import WF


comptime PtrF32 = UnsafePointer[Float32, MutUntrackedOrigin]
comptime PtrBF16 = UnsafePointer[Scalar[DType.bfloat16], MutUntrackedOrigin]
comptime PtrI8 = UnsafePointer[Scalar[DType.int8], MutUntrackedOrigin]
comptime SrcPtr[dtype: DType] = UnsafePointer[Scalar[dtype], MutUntrackedOrigin]
comptime WIDTH = WF

@always_inline
def apply_gamma_in_place(work_row: PtrF32, gamma: PtrF32, cols: Int):
    var k = 0
    while k + WIDTH <= cols:
        (work_row + k).store(
            (work_row + k).load[width=WIDTH]() *
            (gamma + k).load[width=WIDTH]())
        k += WIDTH


@always_inline
def gamma_sqrt_abs_in_place(gamma: PtrF32, cols: Int):
    var k = 0
    while k + WIDTH <= cols:
        var v = abs((gamma + k).load[width=WIDTH]())
        (gamma + k).store(sqrt[DType.float32, WIDTH](v))
        k += WIDTH


@always_inline
def bake_split_gain_in_place(gamma: PtrBF16, cols: Int, eps: Float32 = 1e-12):
    var floor = SIMD[DType.float32, WIDTH](eps)
    var zero = SIMD[DType.float32, WIDTH](0)
    var k = 0
    while k + WIDTH <= cols:
        var g = (gamma + k).load[width=WIDTH]().cast[DType.float32]()
        var s = sqrt[DType.float32, WIDTH](max(abs(g), floor))
        store_bf16[WIDTH](g.lt(zero).select(-s, s), gamma + k)
        k += WIDTH


@always_inline
def add_offset_in_place[dt: DType](gain: SrcPtr[dt], count: Int, offset: Float32):
    var add = SIMD[DType.float32, WIDTH](offset)
    var k = 0
    while k + WIDTH <= count:
        var v = (gain + k).load[width=WIDTH]().cast[DType.float32]() + add
        comptime
        if dt == DType.bfloat16:
            store_bf16[WIDTH](v, (gain + k).bitcast[BFloat16]())
        else:
            (gain + k).store(v.cast[dt]())
        k += WIDTH


@always_inline
def row_absmax(work_row: PtrF32, cols: Int) -> Float32:
    var vmax = SIMD[DType.float32, WIDTH](0)
    var k = 0
    while k + WIDTH <= cols:
        vmax = max(vmax, abs((work_row + k).load[width=WIDTH]()))
        k += WIDTH
    return vmax.reduce_max()


@always_inline
def quantize_inv(work: PtrF32, qi: PtrI8, inv: Float32, n: Int):
    var vinv = SIMD[DType.float32, WIDTH](inv)
    var k = 0
    while k + WIDTH <= n:
        var v = (work + k).load[width=WIDTH]()
        (qi + k).store(quantize_i8[WIDTH](v, vinv))
        k += WIDTH


@always_inline
def quant_segment[store_divided: Bool](
    work: PtrF32, qi: PtrI8, scale_slot: PtrF32, n: Int,
):
    var amax = row_absmax(work, n)
    comptime if store_divided:
        scale_slot[0] = amax / Float32(127.0)
    else:
        scale_slot[0] = amax
    var inv = Float32(127.0) / amax if amax > Float32(0) else Float32(0)
    quantize_inv(work, qi, inv, n)


def fwht_rotate_rows[block: Int](work: PtrF32, rows: Int, cols: Int):
    if cols % block != 0:
        abort(t"butterquant: cols={cols} not divisible by K-axis FWHT block={block}")
    for r in range(rows):
        fwht_row[block](work + r * cols, cols)


def fwht_rotate_columns[head_dim: Int](work: PtrF32, rows: Int, cols: Int):
    if rows % head_dim != 0:
        abort(t"butterquant: rows={rows} not divisible by M-axis FWHT block={head_dim}")
    var scratch_buf = List[Float32](length=head_dim, fill=Float32(0))
    var scratch = scratch_buf.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var num_heads = rows // head_dim
    for h in range(num_heads):
        var base = h * head_dim
        for c in range(cols):
            for r in range(head_dim):
                (scratch + r).store((work + (base + r) * cols + c).load())
            fwht_block[head_dim](scratch)
            for r in range(head_dim):
                (work + (base + r) * cols + c).store((scratch + r).load())
    _ = scratch_buf^


def quant_rows_per_row(work: PtrF32, qi: PtrI8, scales: PtrF32, rows: Int, cols: Int):
    for r in range(rows):
        quant_segment[True](work + r * cols, qi + r * cols, scales + r, cols)


def quant_rows_per_block[block: Int](
    work: PtrF32, qi: PtrI8, scales: PtrF32, rows: Int, cols: Int,
):
    var num_blocks = cols // block
    for r in range(rows):
        var work_row = work + r * cols
        var qi_row = qi + r * cols
        var scale_row = scales + r * num_blocks
        for b in range(num_blocks):
            var off = b * block
            quant_segment[True](work_row + off, qi_row + off, scale_row + b, block)


def dispatch_fwht_block[
    *,
    dispatch: def[block: Int]() capturing [_] -> None,
](block: Int) -> Bool:
    if block == 16:
        dispatch[16]()
        return True
    if block == 32:
        dispatch[32]()
        return True
    if block == 64:
        dispatch[64]()
        return True
    if block == 128:
        dispatch[128]()
        return True
    if block == 256:
        dispatch[256]()
        return True
    if block == 512:
        dispatch[512]()
        return True
    return False


def rotate_and_quant[per_block: Bool](
    block: Int, work: PtrF32, qi: PtrI8, scales: PtrF32,
    rows: Int, cols: Int, two_sided_head_dim: Int = 0,
):
    @parameter
    def rotate_columns[m_block: Int]():
        fwht_rotate_columns[m_block](work, rows, cols)

    @parameter
    def rotate_rows_and_quant[k_block: Int]():
        fwht_rotate_rows[k_block](work, rows, cols)
        if two_sided_head_dim != 0:
            if not dispatch_fwht_block[dispatch=rotate_columns](two_sided_head_dim):
                abort(t"butterquant: unsupported M-axis FWHT block={two_sided_head_dim}")
        comptime if per_block:
            quant_rows_per_block[k_block](work, qi, scales, rows, cols)
        else:
            quant_rows_per_row(work, qi, scales, rows, cols)

    if not dispatch_fwht_block[dispatch=rotate_rows_and_quant](block):
        abort(t"butterquant: unsupported K-axis FWHT block={block}")


def router_center_impl[src_dtype: DType, emit_gauge: Bool](
    src: SrcPtr[src_dtype], gauge: PtrF32,
    centered_bf16: PtrBF16, gauge_bf16: PtrBF16,
    rows: Int, cols: Int,
):
    var k = 0
    while k + WIDTH <= cols:
        (gauge + k).store(SIMD[DType.float32, WIDTH](0))
        k += WIDTH

    for r in range(rows):
        var row = src + r * cols
        k = 0
        while k + WIDTH <= cols:
            var v = (row + k).load[width=WIDTH]().cast[DType.float32]()
            (gauge + k).store((gauge + k).load[width=WIDTH]() + v)
            k += WIDTH

    var inv_rows = SIMD[DType.float32, WIDTH](Float32(1.0) / Float32(rows))
    k = 0
    while k + WIDTH <= cols:
        var g = (gauge + k).load[width=WIDTH]() * inv_rows
        (gauge + k).store(g)
        comptime if emit_gauge:
            store_bf16[WIDTH](g, gauge_bf16 + k)
        k += WIDTH

    for r in range(rows):
        var row = src + r * cols
        var out = centered_bf16 + r * cols
        k = 0
        while k + WIDTH <= cols:
            var v = (row + k).load[width=WIDTH]().cast[DType.float32]()
            var c = v - (gauge + k).load[width=WIDTH]()
            store_bf16[WIDTH](c, out + k)
            k += WIDTH


def router_center[src_dtype: DType](
    src: SrcPtr[src_dtype], gauge: PtrF32,
    centered_bf16: PtrBF16, gauge_bf16: PtrBF16,
    rows: Int, cols: Int,
):
    router_center_impl[src_dtype, True](
        src, gauge, centered_bf16, gauge_bf16, rows, cols)


def router_center_softmax[src_dtype: DType](
    src: SrcPtr[src_dtype], gauge: PtrF32,
    centered_bf16: PtrBF16,
    rows: Int, cols: Int,
):
    router_center_impl[src_dtype, False](
        src, gauge, centered_bf16, centered_bf16, rows, cols)
