from std.algorithm import vectorize

from threading.threading_traits import BurstThreadPool
from simd_math.ops import gelu_tanh_f32
from .helpers import (
    RangePartitionedKernel, Binding,
    fanout_dispatch,
    BF16Ptr, W,
)
from .dispatch_heuristics import (
    GELU_GATE_UP_INLINE_TOKENS, SCALAR_MUL_INLINE_TOKENS,
)
from .profiling import Profiler


@always_inline
def gelu_gate_up_row(
    gate: BF16Ptr, up: BF16Ptr, dst: BF16Ptr, intermediate: Int,
):
    def step[width: Int](idx: Int) {read}:
        var g = (gate + idx).load[width=width]().cast[DType.float32]()
        var u = (up + idx).load[width=width]().cast[DType.float32]()
        var v = gelu_tanh_f32[width](g) * u
        (dst + idx).store(v.cast[DType.bfloat16]())

    vectorize[W](intermediate, step)


@fieldwise_init
struct GeluGateUpTokenKernel(RangePartitionedKernel):
    var gate: BF16Ptr
    var up: BF16Ptr
    var dst: BF16Ptr
    var intermediate: Int
    var start: Int
    var end: Int

    def execute(mut self):
        for tok in range(self.start, self.end):
            var off = tok * self.intermediate
            gelu_gate_up_row(
                self.gate + off, self.up + off, self.dst + off,
                self.intermediate)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_gelu_gate_up[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    gate: Binding[BFloat16, o],
    up: Binding[BFloat16, o],
    dst: Binding[BFloat16, o],
    intermediate: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var ip = intermediate

    @parameter
    def make(r: Int) -> GeluGateUpTokenKernel:
        return GeluGateUpTokenKernel(gate[r], up[r], dst[r], ip, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="gelu_gate_up"](
        pools, prof, seq_len, seq_len * intermediate * 6,
        inline_threshold_bytes=GELU_GATE_UP_INLINE_TOKENS * intermediate * 6)


@always_inline
def scalar_mul_row[hidden: Int](
    src: BF16Ptr, dst: BF16Ptr, scalar: Float32,
):
    def step[width: Int](idx: Int) {read}:
        var x = (src + idx).load[width=width]().cast[DType.float32]()
        var factor = SIMD[DType.float32, width](scalar)
        (dst + idx).store((x * factor).cast[DType.bfloat16]())

    vectorize[W](hidden, step)


@fieldwise_init
struct ScalarMulTokenKernel[hidden: Int](RangePartitionedKernel):
    var src: BF16Ptr
    var dst: BF16Ptr
    var scalar: Float32
    var start: Int
    var end: Int

    def execute(mut self):
        for tok in range(self.start, self.end):
            var off = tok * Self.hidden
            scalar_mul_row[Self.hidden](
                self.src + off, self.dst + off, self.scalar)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_scalar_mul[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    hidden: Int, max_worker_count: Int = 128,
](
    src: Binding[BFloat16, o],
    dst: Binding[BFloat16, o],
    scalar: Float32,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime K = ScalarMulTokenKernel[hidden]

    @parameter
    def make(r: Int) -> K:
        return K(src[r], dst[r], scalar, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="scalar_mul"](
        pools, prof, seq_len, seq_len * hidden * 4,
        inline_threshold_bytes=SCALAR_MUL_INLINE_TOKENS * hidden * 4)
