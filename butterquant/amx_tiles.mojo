from std.collections import InlineArray
from std.math import min
from std.memory import UnsafePointer
from std.os import abort
from std.sys import llvm_intrinsic

from threading.threading_traits import BurstKernel, BurstThreadPool
from kernels.helpers import DispatchBuffer, join_all
from simd_math import has_amx_int8


comptime AMX_TILE_M = 16
comptime AMX_TILE_N = 16
comptime AMX_K_STEP = 64


@fieldwise_init
struct TileConfig(Movable):
    var palette_id: UInt8
    var start_row: UInt8
    var reserved: InlineArray[UInt8, 14]
    var colsb: InlineArray[UInt16, 16]
    var rows: InlineArray[UInt8, 16]

    def __init__(out self):
        self.palette_id = 0
        self.start_row = 0
        self.reserved = InlineArray[UInt8, 14](fill=0)
        self.colsb = InlineArray[UInt16, 16](fill=0)
        self.rows = InlineArray[UInt8, 16](fill=0)


def make_224_i8_config() -> TileConfig:
    var cfg = TileConfig()
    cfg.palette_id = 1
    for i in range(8):
        cfg.rows[i] = 16
        cfg.colsb[i] = 64
    return cfg^


def request_amx_xstate() -> Bool:
    comptime SYS_arch_prctl = 158
    comptime ARCH_REQ_XCOMP_PERM = 0x1023
    comptime XFEATURE_XTILEDATA = 18
    var result = __mlir_op.`pop.external_call`[
        func = "syscall".value,
        _type = Int64,
    ](
        Int64(SYS_arch_prctl),
        Int64(ARCH_REQ_XCOMP_PERM),
        Int64(XFEATURE_XTILEDATA),
    )
    return result == 0


@always_inline
def ldtilecfg[origin: MutOrigin](cfg: UnsafePointer[TileConfig, origin]):
    llvm_intrinsic["llvm.x86.ldtilecfg", NoneType](cfg)


@always_inline
def tilerelease():
    llvm_intrinsic["llvm.x86.tilerelease", NoneType]()


@always_inline
def tilezero[tile: Int]():
    comptime assert tile >= 0 and tile < 8, "tile must be 0-7"
    llvm_intrinsic["llvm.x86.tilezero", NoneType](Int8(tile))


@always_inline
def tileload[tile: Int, dtype: DType, origin: MutOrigin](
    ptr: UnsafePointer[Scalar[dtype], origin],
    stride: Int,
):
    comptime assert tile >= 0 and tile < 8, "tile must be 0-7"
    llvm_intrinsic["llvm.x86.tileloadd64", NoneType](
        Int8(tile), ptr, Int64(stride))


@always_inline
def tilestore[tile: Int, dtype: DType, origin: MutOrigin](
    ptr: UnsafePointer[Scalar[dtype], origin],
    stride: Int,
):
    comptime assert tile >= 0 and tile < 8, "tile must be 0-7"
    llvm_intrinsic["llvm.x86.tilestored64", NoneType](
        Int8(tile), ptr, Int64(stride))


@always_inline
def tdpbssd[dst: Int, src_a: Int, src_b: Int]():
    comptime assert dst >= 0 and dst < 8
    comptime assert src_a >= 0 and src_a < 8
    comptime assert src_b >= 0 and src_b < 8
    llvm_intrinsic["llvm.x86.tdpbssd", NoneType](
        Int8(dst), Int8(src_a), Int8(src_b))


@always_inline
def prime_amx_worker():
    comptime if has_amx_int8():
        var cfg = make_224_i8_config()
        ldtilecfg(UnsafePointer(to=cfg))


@fieldwise_init
struct AmxPrimeKernel(BurstKernel):
    def execute(mut self):
        prime_amx_worker()


def prime_amx_environment[
    P: BurstThreadPool, //, max_worker_count: Int = 128,
](mut pools: List[P]):
    comptime if has_amx_int8():
        if not request_amx_xstate():
            abort("butterquant: AMX init failed; machine denied AMX tile xstate permission")
        prime_amx_worker()
        var buf = DispatchBuffer[AmxPrimeKernel, max_worker_count]()
        for r in range(len(pools)):
            var cap = min(max_worker_count, pools[r].get_capacity())
            for _ in range(cap):
                buf.slot()[] = AmxPrimeKernel()
            buf.dispatch(pools[r])
        join_all(pools)
