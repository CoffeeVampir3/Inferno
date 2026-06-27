from std.collections import InlineArray

from threading.threading_traits import BurstThreadPool
from simd_math.ops import sqrt
from kernels.helpers import (
    WorkerRangePartitionedKernel, Binding,
    fanout_dispatch_per_rank,
    BF16Ptr, F32Ptr,
)
from kernels.moe_router import router_workers
from kernels.rmsnorm import rms_reduce_row
from kernels.profiling import Profiler

from prototypes.sigmoid_router import (
    M3RouterCandidate, M3RouterCandidatePtr, insert_m3_candidate, sigmoid_f32,
    dot_bf16_f32, dispatch_merge_m3_router,
    M3_HIDDEN, M3_NUM_EXPERTS, M3_TOP_K, M3_ROUTED_SCALING, M3_ROUTER_PU,
)


@fieldwise_init
struct M3RouterExpertKernelInvRms[
    hidden: Int, top_k: Int, port_unroll: Int,
    sqrt_n: Float32, n_eps: Float32,
](WorkerRangePartitionedKernel):
    var x: BF16Ptr
    var gate: F32Ptr
    var bias: F32Ptr
    var cands_out: M3RouterCandidatePtr
    var expert_base: Int
    var seq_len: Int
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime sentinel = Float32(-1.0e30)
        for tok in range(self.seq_len):
            var x_row = self.x + tok * Self.hidden
            var sum_sq = rms_reduce_row[Self.hidden](x_row)
            var inv_rms = Self.sqrt_n / sqrt[DType.float32, 1](
                sum_sq + Self.n_eps)

            var cands = InlineArray[M3RouterCandidate, Self.top_k](
                fill=M3RouterCandidate(Int32(0), sentinel, Float32(0)))
            for e in range(self.start, self.end):
                var gate_row = self.gate + e * Self.hidden
                var logit = inv_rms * dot_bf16_f32[
                    Self.hidden, Self.port_unroll](x_row, gate_row)
                var ge = self.expert_base + e
                var weight = sigmoid_f32(logit)
                var score = weight + self.bias[ge]
                insert_m3_candidate[Self.top_k](Int32(ge), score, weight, cands)

            var dst = self.cands_out + (
                self.worker_id * self.seq_len + tok) * Self.top_k
            comptime for k in range(Self.top_k):
                dst[k] = cands[k]

    @always_inline
    def install_worker_range(mut self, worker_id: Int, start: Int, end: Int):
        self.worker_id = worker_id
        self.start = start
        self.end = end


def dispatch_m3_router_expert_invrms[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    hidden: Int, top_k: Int, port_unroll: Int,
    sqrt_n: Float32, n_eps: Float32,
    max_worker_count: Int = 128,
](
    x: Binding[BFloat16, o],
    gate: Binding[Float32, o],
    bias: Binding[Float32, o],
    cands_out: Binding[M3RouterCandidate, o],
    experts_per_rank: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
) -> List[Int]:
    comptime K = M3RouterExpertKernelInvRms[
        hidden, top_k, port_unroll, sqrt_n, n_eps]
    var epr = experts_per_rank

    @parameter
    def make(r: Int) -> K:
        var cands_owned = cands_out[r]
        return K(x[r], gate[r], bias[r], cands_owned, r * epr,
                 seq_len, 0, 0, 0)

    @parameter
    def total_for(r: Int) -> Int:
        return epr

    @parameter
    def data_bytes_for(r: Int) -> Int:
        return epr * hidden * 4

    return fanout_dispatch_per_rank[
        make, total_for, data_bytes_for,
        max_worker_count=max_worker_count,
        worker_policy=router_workers,
        label="m3_router_expert_invrms",
    ](pools, prof)


comptime M3_SQRT_N = sqrt[DType.float32, 1](M3_HIDDEN)
comptime M3_RMS_EPS = Float32(M3_HIDDEN) * Float32(1e-6)


def dispatch_minimax_m3_router_invrms[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    sqrt_n: Float32 = M3_SQRT_N, n_eps: Float32 = M3_RMS_EPS,
    max_worker_count: Int = 128,
](
    x: Binding[BFloat16, o],
    gate: Binding[Float32, o],
    bias: Binding[Float32, o],
    cands: Binding[M3RouterCandidate, o],
    route_idx: Binding[Int32, o],
    route_w: Binding[Float32, o],
    experts_per_rank: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var nws = dispatch_m3_router_expert_invrms[
        hidden=M3_HIDDEN, top_k=M3_TOP_K, port_unroll=M3_ROUTER_PU,
        sqrt_n=sqrt_n, n_eps=n_eps, max_worker_count=max_worker_count,
    ](x, gate, bias, cands, experts_per_rank, seq_len, pools, prof)

    dispatch_merge_m3_router[
        top_k=M3_TOP_K, scaling=M3_ROUTED_SCALING,
        max_worker_count=max_worker_count,
    ](cands, nws, route_idx, route_w, seq_len, pools, prof)
