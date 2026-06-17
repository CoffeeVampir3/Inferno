from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from butterquant.convert import store_out
from butterquant.dot_products import (
    act_broadcast_vnni, dot_loaded, vnni_colsum_correct,
)
from butterquant.types import F32Ptr, I8Ptr
from butterquant.vnni import VNNI_BLK, VNNI_K_STEP, VNNI_N_STEP, VNNI_TILE_N


# Largest M-panel PR (<= MR) whose live vector footprint stays in registers.
# Steady-state liveness, with acc_count = VNNI_N_STEP // width accumulators/row:
#   per-row:   PR*acc_count accumulators + PR broadcasts            = PR*(acc_count + 1)
#   per-block: + PR*acc_count f32 facc held across the block loop   = PR*(2*acc_count + 1)
# plus ~passes transient weight regs (w0/w1 double-buffer) and a scratch.
# The x86 register file is pinned to the width on both targets: width=8 -> 16 ymm,
# width=16 -> 32 zmm. Solve PR*per_pr + reserve <= regs for the largest PR.
@always_inline
def reg_capped_panel[MR: Int, per_block: Bool]() -> Int:
    comptime width = simd_width_of[DType.int32]()
    comptime acc_count = VNNI_N_STEP // width
    comptime regs = 2 * width
    comptime per_pr = (2 * acc_count if per_block else acc_count) + 1
    comptime reserve = VNNI_TILE_N // width + 1
    comptime fits = (regs - reserve) // per_pr
    comptime if fits < 1:
        return 1
    elif fits < MR:
        return fits
    else:
        return MR


@always_inline
def accumulate_tiles[
    width: Int, PR: Int, row_ptr: def(Int) capturing [_] -> I8Ptr,
](
    wpacked: I8Ptr,
    packed_base: Int,
    k_base: Int,
    k_len: Int,
    mut acc: InlineArray[SIMD[DType.int32, width], PR * (VNNI_N_STEP // width)],
):
    comptime passes = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width
    comptime dc_count = VNNI_K_STEP // VNNI_BLK
    comptime tile_dc_bytes = VNNI_TILE_N * VNNI_BLK
    comptime tile_ks_bytes = dc_count * tile_dc_bytes

    var packed_off = packed_base
    for ks in range(0, k_len, VNNI_K_STEP):
        for dc in range(dc_count):
            var k_pos = k_base + ks + dc * VNNI_BLK
            var ab = InlineArray[SIMD[DType.uint8, width * 4], PR](
                uninitialized=True)
            comptime for r in range(PR):
                ab[r] = act_broadcast_vnni[width](row_ptr(r), k_pos)
            var t0 = packed_off + dc * tile_dc_bytes
            var t1 = t0 + tile_ks_bytes
            comptime for p in range(passes):
                var w0 = (wpacked + t0 + p * bytes_per_pass).load[
                    width = width * 4, non_temporal=True]()
                comptime for r in range(PR):
                    acc[r * acc_count + p] = dot_loaded[width](
                        acc[r * acc_count + p], ab[r], w0)
            comptime for p in range(passes):
                var w1 = (wpacked + t1 + p * bytes_per_pass).load[
                    width = width * 4, non_temporal=True]()
                comptime for r in range(PR):
                    acc[r * acc_count + passes + p] = dot_loaded[width](
                        acc[r * acc_count + passes + p], ab[r], w1)
        packed_off += 2 * tile_ks_bytes


@always_inline
def accumulate_n_step[width: Int, PR: Int](
    act: I8Ptr,
    m_panel: Int,
    k_stride: Int,
    wpacked: I8Ptr,
    packed_base: Int,
    k_base: Int,
    k_len: Int,
    mut acc: InlineArray[SIMD[DType.int32, width], PR * (VNNI_N_STEP // width)],
):
    @parameter
    def row_ptr(r: Int) -> I8Ptr:
        return act + (m_panel + r) * k_stride

    accumulate_tiles[width, PR, row_ptr](
        wpacked, packed_base, k_base, k_len, acc)


@always_inline
def accumulate_n_step_gathered[width: Int, PR: Int](
    rows: InlineArray[I8Ptr, PR],
    wpacked: I8Ptr,
    packed_base: Int,
    k_base: Int,
    k_len: Int,
    mut acc: InlineArray[SIMD[DType.int32, width], PR * (VNNI_N_STEP // width)],
):
    @parameter
    def row_ptr(r: Int) -> I8Ptr:
        return rows[r]

    accumulate_tiles[width, PR, row_ptr](
        wpacked, packed_base, k_base, k_len, acc)


@always_inline
def gemm_i8_per_row_panel[K: Int, PR: Int, Out: DType](
    act: I8Ptr,
    m_panel: Int,
    n_rows: Int,
    act_scale: F32Ptr,
    wpacked: I8Ptr,
    wsc: F32Ptr,
    colsum: F32Ptr,
    dst: UnsafePointer[Scalar[Out], MutAnyOrigin],
    ns: Int,
):
    comptime width = simd_width_of[DType.int32]()
    comptime acc_count = VNNI_N_STEP // width
    var iacc = InlineArray[SIMD[DType.int32, width], PR * acc_count](
        fill=SIMD[DType.int32, width](0))
    accumulate_n_step[width, PR](
        act, m_panel, K, wpacked, ns * K, 0, K, iacc)

    var inv127 = Float32(1.0) / Float32(127.0)
    comptime for r in range(PR):
        var ad = act_scale[m_panel + r] * inv127
        comptime for a in range(acc_count):
            var n_base = ns + a * width
            var cs = (colsum + n_base).load[width=width]()
            var corrected = vnni_colsum_correct[width](
                iacc[r * acc_count + a], cs)
            var res = corrected * ad * (wsc + n_base).load[width=width]()
            store_out[Out, width](res, dst + (m_panel + r) * n_rows + n_base)


def gemm_i8_per_row[K: Int, MR: Int, Out: DType](
    act: I8Ptr,
    m: Int,
    n_rows: Int,
    act_scale: F32Ptr,
    wpacked: I8Ptr,
    wsc: F32Ptr,
    colsum: F32Ptr,
    dst: UnsafePointer[Scalar[Out], MutAnyOrigin],
    start_tile: Int,
    end_tile: Int,
):
    comptime PR = reg_capped_panel[MR, False]()
    var m_panel = 0
    while m_panel + PR <= m:
        for t in range(start_tile, end_tile):
            gemm_i8_per_row_panel[K, PR, Out](
                act, m_panel, n_rows, act_scale, wpacked, wsc, colsum, dst,
                t * VNNI_N_STEP)
        m_panel += PR
    while m_panel < m:
        for t in range(start_tile, end_tile):
            gemm_i8_per_row_panel[K, 1, Out](
                act, m_panel, n_rows, act_scale, wpacked, wsc, colsum, dst,
                t * VNNI_N_STEP)
        m_panel += 1


@always_inline
def gemm_i8_per_block_panel[N: Int, block: Int, PR: Int, Out: DType](
    act: I8Ptr,
    m_panel: Int,
    k_dim: Int,
    act_scale: F32Ptr,
    wpacked: I8Ptr,
    wsc: F32Ptr,
    colsum: F32Ptr,
    dst: UnsafePointer[Scalar[Out], MutAnyOrigin],
    ns: Int,
):
    comptime width = simd_width_of[DType.int32]()
    comptime acc_count = VNNI_N_STEP // width
    comptime inv127 = Float32(1.0) / Float32(127.0)
    comptime blk_bytes = block * VNNI_N_STEP
    var nb = k_dim // block

    var facc = InlineArray[SIMD[DType.float32, width], PR * acc_count](
        fill=SIMD[DType.float32, width](0))

    for b in range(nb):
        var iacc = InlineArray[SIMD[DType.int32, width], PR * acc_count](
            fill=SIMD[DType.int32, width](0))
        accumulate_n_step[width, PR](
            act, m_panel, k_dim, wpacked, ns * k_dim + b * blk_bytes, b * block, block,
            iacc)
        comptime for r in range(PR):
            var adv = SIMD[DType.float32, width](
                act_scale[(m_panel + r) * nb + b] * inv127)
            comptime for a in range(acc_count):
                var n_base = ns + a * width
                var cs = (colsum + b * N + n_base).load[width=width]()
                var corrected = vnni_colsum_correct[width](
                    iacc[r * acc_count + a], cs)
                facc[r * acc_count + a] = corrected.fma(
                    adv, facc[r * acc_count + a])

    comptime for r in range(PR):
        comptime for a in range(acc_count):
            var n_base = ns + a * width
            var res = facc[r * acc_count + a] * (wsc + n_base).load[width=width]()
            store_out[Out, width](res, dst + (m_panel + r) * N + n_base)


def gemm_i8_per_block[N: Int, block: Int, MR: Int, Out: DType](
    act: I8Ptr,
    m: Int,
    k_dim: Int,
    act_scale: F32Ptr,
    wpacked: I8Ptr,
    wsc: F32Ptr,
    colsum: F32Ptr,
    dst: UnsafePointer[Scalar[Out], MutAnyOrigin],
    start_tile: Int,
    end_tile: Int,
):
    comptime PR = reg_capped_panel[MR, True]()
    var m_panel = 0
    while m_panel + PR <= m:
        for t in range(start_tile, end_tile):
            gemm_i8_per_block_panel[N, block, PR, Out](
                act, m_panel, k_dim, act_scale, wpacked, wsc, colsum, dst,
                t * VNNI_N_STEP)
        m_panel += PR
    while m_panel < m:
        for t in range(start_tile, end_tile):
            gemm_i8_per_block_panel[N, block, 1, Out](
                act, m_panel, k_dim, act_scale, wpacked, wsc, colsum, dst,
                t * VNNI_N_STEP)
        m_panel += 1
