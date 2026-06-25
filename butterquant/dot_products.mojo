from std.collections import InlineArray
from std.sys import llvm_intrinsic
from std.sys.info import CompilationTarget

from butterquant.types import F32Ptr, I8Ptr, WI


@always_inline
def vpdpbusd[width: Int](
    acc: SIMD[DType.int32, width],
    a: SIMD[DType.uint8, width * 4],
    b: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd." + String(width * 32),
        SIMD[DType.int32, width],
    ](acc, a, b)


@always_inline
def act_broadcast_vnni[width: Int](act_row: I8Ptr, k_pos: Int) -> SIMD[
    DType.uint8, width * 4,
]:
    var b4 = (act_row + k_pos).bitcast[UInt8]().load[width=4]() ^ SIMD[
        DType.uint8, 4](0x80)
    var out = SIMD[DType.uint8, width * 4]()
    comptime for lane in range(width):
        out = out.insert[offset = lane * 4](b4)
    return out


@always_inline
def dot_loaded[width: Int](
    acc: SIMD[DType.int32, width],
    act_bytes: SIMD[DType.uint8, width * 4],
    weights: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    return vpdpbusd[width](acc, act_bytes, weights)


@always_inline
def bq_score_unroll[head_dim: Int, gqa_ratio: Int]() -> Int:
    comptime bytes = WI * 4
    comptime chunks = head_dim // bytes
    var cu = 1
    while (
        cu * 2 <= chunks
        and (cu * 2) * gqa_ratio <= 28
        and cu * gqa_ratio < 8
        and head_dim % ((cu * 2) * bytes) == 0
    ):
        cu *= 2
    return cu


@always_inline
def vnni_panel_score_dot[
    block: Int, panel: Int, unroll: Int,
](
    k: I8Ptr,
    read q_ptrs: InlineArray[I8Ptr, panel],
) -> InlineArray[Int32, panel]:
    comptime assert CompilationTarget.has_vnni(), (
        "butterquant VNNI dot requires VNNI")
    comptime bytes = WI * 4
    comptime assert block % (unroll * bytes) == 0, (
        "vnni_panel_score_dot: block must be a multiple of unroll*4*WI")
    var accs = InlineArray[SIMD[DType.int32, WI], panel * unroll](
        fill=SIMD[DType.int32, WI](0))
    comptime STRIDE = unroll * bytes
    for base in range(0, block, STRIDE):
        comptime for u in range(unroll):
            var off = base + u * bytes
            var av = (k + off).bitcast[UInt8]().load[width=bytes]() ^ SIMD[
                DType.uint8, bytes](0x80)
            comptime for r in range(panel):
                var bv = (q_ptrs[r] + off).load[width=bytes]()
                accs[r * unroll + u] = vpdpbusd[WI](accs[r * unroll + u], av, bv)
    var out = InlineArray[Int32, panel](uninitialized=True)
    comptime for r in range(panel):
        var s = SIMD[DType.int32, WI](0)
        comptime for u in range(unroll):
            s += accs[r * unroll + u]
        out[r] = s.reduce_add()
    return out


@always_inline
def vnni_shifted_dot[block: Int, emit_rhs_sum: Bool](
    a: I8Ptr, b: I8Ptr,
) -> Tuple[SIMD[DType.int32, WI], Int32]:
    comptime assert CompilationTarget.has_vnni(), (
        "butterquant VNNI dot requires VNNI")
    comptime bytes = WI * 4
    comptime assert block % bytes == 0, (
        "VNNI dot block must be a multiple of four i32 SIMD vectors")
    var acc = SIMD[DType.int32, WI](0)
    var sum_acc = SIMD[DType.int32, WI](0)
    var ones = SIMD[DType.uint8, bytes](1)
    for k in range(0, block, bytes):
        var av = (a + k).bitcast[UInt8]().load[width=bytes]() ^ SIMD[
            DType.uint8, bytes](0x80)
        var bv = (b + k).load[width=bytes]()
        acc = vpdpbusd[WI](acc, av, bv)
        comptime if emit_rhs_sum:
            sum_acc = vpdpbusd[WI](sum_acc, ones, bv)
    var rhs_sum = sum_acc.reduce_add() if emit_rhs_sum else Int32(0)
    return (acc, rhs_sum)


@always_inline
def i8_vnni_block_dot[block: Int](a: I8Ptr, b: I8Ptr) -> Int32:
    var r = vnni_shifted_dot[block, True](a, b)
    return r[0].reduce_add() - Int32(128) * r[1]


@always_inline
def vnni_colsum_correct[width: Int](
    iacc: SIMD[DType.int32, width], colsum: SIMD[DType.float32, width],
) -> SIMD[DType.float32, width]:
    return iacc.cast[DType.float32]() - Float32(128) * colsum


@always_inline
def head_logit_row[block: Int](
    x_i8: I8Ptr, sa: F32Ptr, weight: I8Ptr, scales: F32Ptr, cols: Int,
) -> Float32:
    debug_assert(cols % block == 0,
        "head_logit_row: cols must be block-aligned")
    var nb = cols // block
    var acc = Float32(0)
    for b in range(nb):
        var off = b * block
        var r = i8_vnni_block_dot[block](x_i8 + off, weight + off)
        acc += Float32(r) * (sa[b] / Float32(127.0)) * scales[b]
    return acc
