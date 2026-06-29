from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from butterquant.convert import store_out
from butterquant.vnni import VNNI_N_STEP
from butterquant.types import F32Ptr, I8Ptr
from butterquant.amx_tiles import (
    AMX_TILE_M, AMX_TILE_N, AMX_K_STEP,
    tilezero, tileload, tilestore, tdpbssd,
)


comptime I32Ptr = UnsafePointer[Int32, MutUntrackedOrigin]
comptime WIDTH = simd_width_of[DType.int32]()
comptime INV127 = Float32(1.0) / Float32(127.0)
comptime CTILE = AMX_TILE_M * AMX_TILE_N
comptime HALF_BYTES = AMX_TILE_M * AMX_K_STEP
comptime C_STRIDE = AMX_TILE_N * 4
comptime AMX_MIN_ROWS = 16


@always_inline
def amx_b_tile_base(t: Int, k_off: Int, k_dim: Int) -> Int:
    return (t * VNNI_N_STEP) * k_dim + k_off * VNNI_N_STEP


@always_inline
def dequant_fused[
    write: def(Int, Int, SIMD[DType.float32, WIDTH]) capturing [_] -> None,
    origin: MutOrigin,
](
    c: UnsafePointer[Int32, origin], row_base: Int, n_base: Int, m_rows: Int,
    act_scale: F32Ptr, wsc: F32Ptr,
):
    for r in range(m_rows):
        var ad = act_scale[row_base + r] * INV127
        var nj = 0
        while nj < AMX_TILE_N:
            var cv = (c + r * AMX_TILE_N + nj).load[width=WIDTH]().cast[
                DType.float32]()
            var ws = (wsc + n_base + nj).load[width=WIDTH]()
            write(row_base + r, n_base + nj, cv * ad * ws)
            nj += WIDTH


@always_inline
def block_accumulate[
    c_origin: MutOrigin, f_origin: MutOrigin,
](
    c: UnsafePointer[Int32, c_origin], facc: UnsafePointer[Float32, f_origin],
    row_base: Int, b: Int, nb: Int, m_rows: Int,
    act_scale: F32Ptr,
):
    for r in range(m_rows):
        var adv = act_scale[(row_base + r) * nb + b] * INV127
        var nj = 0
        while nj < AMX_TILE_N:
            var slot = facc + r * AMX_TILE_N + nj
            var cv = (c + r * AMX_TILE_N + nj).load[width=WIDTH]().cast[
                DType.float32]()
            slot.store(slot.load[width=WIDTH]() + cv * adv)
            nj += WIDTH


@always_inline
def block_finalize[
    write: def(Int, Int, SIMD[DType.float32, WIDTH]) capturing [_] -> None,
    origin: MutOrigin,
](
    facc: UnsafePointer[Float32, origin], row_base: Int, n_base: Int,
    m_rows: Int, wsc: F32Ptr,
):
    for r in range(m_rows):
        var nj = 0
        while nj < AMX_TILE_N:
            var fv = (facc + r * AMX_TILE_N + nj).load[width=WIDTH]()
            var ws = (wsc + n_base + nj).load[width=WIDTH]()
            write(row_base + r, n_base + nj, fv * ws)
            nj += WIDTH


@always_inline
def amx_panel_2x32[
    block: Int,
    write: def(Int, Int, SIMD[DType.float32, WIDTH]) capturing [_] -> None,
](
    act: I8Ptr, m_panel: Int, k_dim: Int, act_scale: F32Ptr, weight: I8Ptr,
    wsc: F32Ptr, t: Int,
):
    var nb = k_dim // block
    var c = InlineArray[Int32, 4 * CTILE](uninitialized=True)
    var cp = UnsafePointer(to=c).bitcast[Int32]()
    var c00 = cp
    var c01 = cp + CTILE
    var c10 = cp + 2 * CTILE
    var c11 = cp + 3 * CTILE
    var rb0 = m_panel
    var rb1 = m_panel + AMX_TILE_M
    var n0 = t * VNNI_N_STEP
    var n1 = n0 + AMX_TILE_N

    if nb == 1:
        tilezero[4]()
        tilezero[5]()
        tilezero[6]()
        tilezero[7]()
        for k_off in range(0, k_dim, AMX_K_STEP):
            tileload[0, DType.int8](act + rb0 * k_dim + k_off, k_dim)
            tileload[1, DType.int8](act + rb1 * k_dim + k_off, k_dim)
            var base = amx_b_tile_base(t, k_off, k_dim)
            tileload[2, DType.int8](weight + base, AMX_K_STEP)
            tileload[3, DType.int8](weight + base + HALF_BYTES, AMX_K_STEP)
            tdpbssd[4, 0, 2]()
            tdpbssd[5, 0, 3]()
            tdpbssd[6, 1, 2]()
            tdpbssd[7, 1, 3]()
        tilestore[4, DType.int32](c00, C_STRIDE)
        tilestore[5, DType.int32](c01, C_STRIDE)
        tilestore[6, DType.int32](c10, C_STRIDE)
        tilestore[7, DType.int32](c11, C_STRIDE)
        dequant_fused[write](c00, rb0, n0, AMX_TILE_M, act_scale, wsc)
        dequant_fused[write](c01, rb0, n1, AMX_TILE_M, act_scale, wsc)
        dequant_fused[write](c10, rb1, n0, AMX_TILE_M, act_scale, wsc)
        dequant_fused[write](c11, rb1, n1, AMX_TILE_M, act_scale, wsc)
    else:
        var facc = InlineArray[Float32, 4 * CTILE](fill=Float32(0))
        var fp = UnsafePointer(to=facc).bitcast[Float32]()
        var f00 = fp
        var f01 = fp + CTILE
        var f10 = fp + 2 * CTILE
        var f11 = fp + 3 * CTILE
        for b in range(nb):
            tilezero[4]()
            tilezero[5]()
            tilezero[6]()
            tilezero[7]()
            for k_off in range(b * block, (b + 1) * block, AMX_K_STEP):
                tileload[0, DType.int8](act + rb0 * k_dim + k_off, k_dim)
                tileload[1, DType.int8](act + rb1 * k_dim + k_off, k_dim)
                var base = amx_b_tile_base(t, k_off, k_dim)
                tileload[2, DType.int8](weight + base, AMX_K_STEP)
                tileload[3, DType.int8](weight + base + HALF_BYTES, AMX_K_STEP)
                tdpbssd[4, 0, 2]()
                tdpbssd[5, 0, 3]()
                tdpbssd[6, 1, 2]()
                tdpbssd[7, 1, 3]()
            tilestore[4, DType.int32](c00, C_STRIDE)
            tilestore[5, DType.int32](c01, C_STRIDE)
            tilestore[6, DType.int32](c10, C_STRIDE)
            tilestore[7, DType.int32](c11, C_STRIDE)
            block_accumulate(c00, f00, rb0, b, nb, AMX_TILE_M, act_scale)
            block_accumulate(c01, f01, rb0, b, nb, AMX_TILE_M, act_scale)
            block_accumulate(c10, f10, rb1, b, nb, AMX_TILE_M, act_scale)
            block_accumulate(c11, f11, rb1, b, nb, AMX_TILE_M, act_scale)
        block_finalize[write](f00, rb0, n0, AMX_TILE_M, wsc)
        block_finalize[write](f01, rb0, n1, AMX_TILE_M, wsc)
        block_finalize[write](f10, rb1, n0, AMX_TILE_M, wsc)
        block_finalize[write](f11, rb1, n1, AMX_TILE_M, wsc)
        _ = facc
    _ = c


@always_inline
def amx_panel_1x32[
    block: Int,
    write: def(Int, Int, SIMD[DType.float32, WIDTH]) capturing [_] -> None,
](
    act: I8Ptr, m_panel: Int, m_rows: Int, k_dim: Int, act_scale: F32Ptr,
    weight: I8Ptr, wsc: F32Ptr, t: Int,
):
    var nb = k_dim // block
    var c = InlineArray[Int32, 2 * CTILE](uninitialized=True)
    var cp = UnsafePointer(to=c).bitcast[Int32]()
    var c0 = cp
    var c1 = cp + CTILE
    var n0 = t * VNNI_N_STEP
    var n1 = n0 + AMX_TILE_N

    if nb == 1:
        tilezero[4]()
        tilezero[5]()
        for k_off in range(0, k_dim, AMX_K_STEP):
            tileload[0, DType.int8](act + m_panel * k_dim + k_off, k_dim)
            var base = amx_b_tile_base(t, k_off, k_dim)
            tileload[2, DType.int8](weight + base, AMX_K_STEP)
            tileload[3, DType.int8](weight + base + HALF_BYTES, AMX_K_STEP)
            tdpbssd[4, 0, 2]()
            tdpbssd[5, 0, 3]()
        tilestore[4, DType.int32](c0, C_STRIDE)
        tilestore[5, DType.int32](c1, C_STRIDE)
        dequant_fused[write](c0, m_panel, n0, m_rows, act_scale, wsc)
        dequant_fused[write](c1, m_panel, n1, m_rows, act_scale, wsc)
    else:
        var facc = InlineArray[Float32, 2 * CTILE](fill=Float32(0))
        var fp = UnsafePointer(to=facc).bitcast[Float32]()
        var f0 = fp
        var f1 = fp + CTILE
        for b in range(nb):
            tilezero[4]()
            tilezero[5]()
            for k_off in range(b * block, (b + 1) * block, AMX_K_STEP):
                tileload[0, DType.int8](act + m_panel * k_dim + k_off, k_dim)
                var base = amx_b_tile_base(t, k_off, k_dim)
                tileload[2, DType.int8](weight + base, AMX_K_STEP)
                tileload[3, DType.int8](weight + base + HALF_BYTES, AMX_K_STEP)
                tdpbssd[4, 0, 2]()
                tdpbssd[5, 0, 3]()
            tilestore[4, DType.int32](c0, C_STRIDE)
            tilestore[5, DType.int32](c1, C_STRIDE)
            block_accumulate(c0, f0, m_panel, b, nb, m_rows, act_scale)
            block_accumulate(c1, f1, m_panel, b, nb, m_rows, act_scale)
        block_finalize[write](f0, m_panel, n0, m_rows, wsc)
        block_finalize[write](f1, m_panel, n1, m_rows, wsc)
        _ = facc
    _ = c


def amx_gemm[
    block: Int,
    write: def(Int, Int, SIMD[DType.float32, WIDTH]) capturing [_] -> None,
](
    act: I8Ptr, m: Int, k_dim: Int, act_scale: F32Ptr, weight: I8Ptr,
    wsc: F32Ptr, start_tile: Int, end_tile: Int,
):
    comptime assert block % AMX_K_STEP == 0, "block must be AMX K-step aligned"
    debug_assert(k_dim % block == 0, "block must divide k_dim")
    debug_assert(k_dim % AMX_K_STEP == 0, "k_dim must be AMX K-step aligned")
    debug_assert(m >= AMX_TILE_M,
        "amx_gemm needs m >= AMX_TILE_M; smaller m must route to vpdpbusd")
    var m_panel = 0
    while m_panel + 2 * AMX_TILE_M <= m:
        for t in range(start_tile, end_tile):
            amx_panel_2x32[block, write](
                act, m_panel, k_dim, act_scale, weight, wsc, t)
        m_panel += 2 * AMX_TILE_M
    while m_panel + AMX_TILE_M <= m:
        for t in range(start_tile, end_tile):
            amx_panel_1x32[block, write](
                act, m_panel, AMX_TILE_M, k_dim, act_scale, weight, wsc, t)
        m_panel += AMX_TILE_M
    if m_panel < m:
        for t in range(start_tile, end_tile):
            amx_panel_1x32[block, write](
                act, m - AMX_TILE_M, AMX_TILE_M, k_dim, act_scale, weight, wsc, t)


def amx_gemm_linear_store[block: Int, Out: DType](
    act: I8Ptr, m: Int, n_rows: Int, k_dim: Int, act_scale: F32Ptr,
    weight: I8Ptr, wsc: F32Ptr, output: UnsafePointer[Scalar[Out], MutUntrackedOrigin],
    start_tile: Int, end_tile: Int,
):
    @parameter
    def write(row: Int, n_base: Int, res: SIMD[DType.float32, WIDTH]):
        store_out[Out, WIDTH](res, output + row * n_rows + n_base)

    amx_gemm[block, write](
        act, m, k_dim, act_scale, weight, wsc, start_tile, end_tile)
