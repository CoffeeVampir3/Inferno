from std.algorithm import vectorize

from threading.threading_traits import BurstThreadPool
from simd_math.ops import gelu_tanh_f32, exp_f32
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
def swiglu_oai_activate[width: Int, alpha: Float32, limit: Float32](
    g: SIMD[DType.float32, width], u: SIMD[DType.float32, width],
) -> SIMD[DType.float32, width]:
    var gc = min(g, SIMD[DType.float32, width](limit))
    var uc = max(
        SIMD[DType.float32, width](-limit),
        min(u, SIMD[DType.float32, width](limit)))
    var z = SIMD[DType.float32, width](alpha) * gc
    var glu = gc / (SIMD[DType.float32, width](1.0) + exp_f32[width](-z))
    return (uc + SIMD[DType.float32, width](1.0)) * glu


@always_inline
def gate_up_activate[
    width: Int, activation: StaticString, alpha: Float32, limit: Float32,
](
    g: SIMD[DType.float32, width], u: SIMD[DType.float32, width],
) -> SIMD[DType.float32, width]:
    comptime if activation == "swiglu_oai":
        return swiglu_oai_activate[width, alpha, limit](g, u)
    elif activation == "gelu":
        return gelu_tanh_f32[width](g) * u
    else:
        comptime assert False, (
            "gate_up_activate: unknown activation (expected 'gelu' or"
            " 'swiglu_oai')")


@always_inline
def gate_up_row[activation: StaticString, alpha: Float32, limit: Float32](
    gate: BF16Ptr, up: BF16Ptr, dst: BF16Ptr, intermediate: Int,
):
    def step[width: Int](idx: Int) {read}:
        var g = (gate + idx).load[width=width]().cast[DType.float32]()
        var u = (up + idx).load[width=width]().cast[DType.float32]()
        var v = gate_up_activate[width, activation, alpha, limit](g, u)
        (dst + idx).store(v.cast[DType.bfloat16]())

    vectorize[W](intermediate, step)


@fieldwise_init
struct GateUpTokenKernel[
    activation: StaticString, alpha: Float32, limit: Float32,
](RangePartitionedKernel):
    var gate: BF16Ptr
    var up: BF16Ptr
    var dst: BF16Ptr
    var intermediate: Int
    var start: Int
    var end: Int

    def execute(mut self):
        for tok in range(self.start, self.end):
            var off = tok * self.intermediate
            gate_up_row[Self.activation, Self.alpha, Self.limit](
                self.gate + off, self.up + off, self.dst + off,
                self.intermediate)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_gate_up_act[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    activation: StaticString, alpha: Float32 = 0.0, limit: Float32 = 0.0,
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
    comptime K = GateUpTokenKernel[activation, alpha, limit]
    var ip = intermediate

    @parameter
    def make(r: Int) -> K:
        return K(gate[r], up[r], dst[r], ip, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="gate_up"](
        pools, prof, seq_len, seq_len * intermediate * 6,
        inline_threshold_bytes=GELU_GATE_UP_INLINE_TOKENS * intermediate * 6)


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
    dispatch_gate_up_act[
        activation="gelu", max_worker_count=max_worker_count,
    ](gate, up, dst, intermediate, seq_len, pools, prof)


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


@always_inline
def residual_add_row[hidden: Int](
    a: BF16Ptr, b: BF16Ptr, dst: BF16Ptr,
):
    def step[width: Int](idx: Int) {read}:
        var x = (a + idx).load[width=width]().cast[DType.float32]()
        var y = (b + idx).load[width=width]().cast[DType.float32]()
        (dst + idx).store((x + y).cast[DType.bfloat16]())

    vectorize[W](hidden, step)


@fieldwise_init
struct ResidualAddTokenKernel[hidden: Int](RangePartitionedKernel):
    var a: BF16Ptr
    var b: BF16Ptr
    var dst: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        for tok in range(self.start, self.end):
            var off = tok * Self.hidden
            residual_add_row[Self.hidden](
                self.a + off, self.b + off, self.dst + off)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_residual_add[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    hidden: Int, max_worker_count: Int = 128,
](
    a: Binding[BFloat16, o],
    b: Binding[BFloat16, o],
    dst: Binding[BFloat16, o],
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime K = ResidualAddTokenKernel[hidden]

    @parameter
    def make(r: Int) -> K:
        return K(a[r], b[r], dst[r], 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="residual_add"](
        pools, prof, seq_len, seq_len * hidden * 4,
        inline_threshold_bytes=SCALAR_MUL_INLINE_TOKENS * hidden * 4)
