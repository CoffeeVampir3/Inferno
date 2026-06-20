from std.algorithm import vectorize

from threading.threading_traits import BurstThreadPool
from simd_math.ops import exp_f32

from kernels.helpers import (
    RangePartitionedKernel, Binding, fanout_dispatch, BF16Ptr, W,
)
from kernels.dispatch_heuristics import GELU_GATE_UP_INLINE_TOKENS
from kernels.profiling import Profiler


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
def swiglu_oai_gate_up_row[alpha: Float32, limit: Float32](
    gate: BF16Ptr, up: BF16Ptr, dst: BF16Ptr, intermediate: Int,
):
    def step[width: Int](idx: Int) {read}:
        var g = (gate + idx).load[width=width]().cast[DType.float32]()
        var u = (up + idx).load[width=width]().cast[DType.float32]()
        var v = swiglu_oai_activate[width, alpha, limit](g, u)
        (dst + idx).store(v.cast[DType.bfloat16]())

    vectorize[W](intermediate, step)


@fieldwise_init
struct SwigluOaiGateUpKernel[alpha: Float32, limit: Float32](
    RangePartitionedKernel
):
    var gate: BF16Ptr
    var up: BF16Ptr
    var dst: BF16Ptr
    var intermediate: Int
    var start: Int
    var end: Int

    def execute(mut self):
        for tok in range(self.start, self.end):
            var off = tok * self.intermediate
            swiglu_oai_gate_up_row[Self.alpha, Self.limit](
                self.gate + off, self.up + off, self.dst + off,
                self.intermediate)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_swiglu_oai_gate_up[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    alpha: Float32, limit: Float32, max_worker_count: Int = 128,
](
    gate: Binding[BFloat16, o],
    up: Binding[BFloat16, o],
    dst: Binding[BFloat16, o],
    intermediate: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime K = SwigluOaiGateUpKernel[alpha, limit]
    var ip = intermediate

    @parameter
    def make(r: Int) -> K:
        return K(gate[r], up[r], dst[r], ip, 0, 0)

    fanout_dispatch[
        make, max_worker_count=max_worker_count, label="swiglu_oai_gate_up",
    ](pools, prof, seq_len, seq_len * intermediate * 6,
      inline_threshold_bytes=GELU_GATE_UP_INLINE_TOKENS * intermediate * 6)


comptime M3_SWIGLU_ALPHA = Float32(1.702)
comptime M3_SWIGLU_LIMIT = Float32(7.0)


def dispatch_minimax_m3_swiglu_gate_up[
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
    dispatch_swiglu_oai_gate_up[
        alpha=M3_SWIGLU_ALPHA, limit=M3_SWIGLU_LIMIT,
        max_worker_count=max_worker_count,
    ](gate, up, dst, intermediate, seq_len, pools, prof)
