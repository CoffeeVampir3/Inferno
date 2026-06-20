from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import size_of

from threading.threading_traits import BurstThreadPool
from simd_math.ops import exp_f32
from kernels.helpers import (
    WorkerRangePartitionedKernel, RangePartitionedKernel, Binding,
    fanout_dispatch_per_rank,
    DispatchBuffer, tile_dispatch, join_all,
    BF16Ptr, F32Ptr, I32Ptr, W,
)
from kernels.moe_router import RouteGatherKernel, router_workers
from kernels.logsum_merge import merge_workers
from kernels.profiling import Profiler, DispatchSpan


comptime IntPtr = UnsafePointer[Int, MutAnyOrigin]
comptime MAX_MERGE_TP = 64


@fieldwise_init
struct M3RouterCandidate(Copyable, ImplicitlyCopyable):
    var expert: Int32
    var score: Float32
    var weight: Float32


comptime M3RouterCandidatePtr = UnsafePointer[M3RouterCandidate, MutAnyOrigin]


@always_inline
def insert_m3_candidate[top_k: Int](
    expert: Int32, score: Float32, weight: Float32,
    mut cands: InlineArray[M3RouterCandidate, top_k],
):
    for k in range(top_k):
        var c = cands[k]
        if score > c.score or (score == c.score and Int(expert) < Int(c.expert)):
            var j = top_k - 1
            while j > k:
                cands[j] = cands[j - 1]
                j -= 1
            cands[k] = M3RouterCandidate(expert, score, weight)
            return


@always_inline
def sigmoid_f32(x: Float32) -> Float32:
    var e = exp_f32[1](SIMD[DType.float32, 1](-x))[0]
    return Float32(1.0) / (Float32(1.0) + e)


@always_inline
def dot_bf16_f32[cols: Int, port_unroll: Int](
    x: BF16Ptr, gate: F32Ptr,
) -> Float32:
    var accs = InlineArray[SIMD[DType.float32, W], port_unroll](
        fill=SIMD[DType.float32, W](0))
    var j = 0
    while j + port_unroll * W <= cols:
        comptime for p in range(port_unroll):
            var off = j + p * W
            var xv = (x + off).load[width=W]().cast[DType.float32]()
            var gv = (gate + off).load[width=W]()
            accs[p] = xv * gv + accs[p]
        j += port_unroll * W
    var acc = accs[0]
    comptime for p in range(1, port_unroll):
        acc += accs[p]
    while j + W <= cols:
        var xv = (x + j).load[width=W]().cast[DType.float32]()
        var gv = (gate + j).load[width=W]()
        acc = xv * gv + acc
        j += W
    var total = acc.reduce_add()
    while j < cols:
        total += x[j].cast[DType.float32]() * gate[j]
        j += 1
    return total


@fieldwise_init
struct M3RouterExpertKernel[
    hidden: Int, top_k: Int, port_unroll: Int,
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

            var cands = InlineArray[M3RouterCandidate, Self.top_k](
                fill=M3RouterCandidate(Int32(0), sentinel, Float32(0)))
            for e in range(self.start, self.end):
                var gate_row = self.gate + e * Self.hidden
                var logit = dot_bf16_f32[Self.hidden, Self.port_unroll](
                    x_row, gate_row)
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


def dispatch_m3_router_expert[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    hidden: Int, top_k: Int, port_unroll: Int,
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
    comptime K = M3RouterExpertKernel[hidden, top_k, port_unroll]
    var epr = experts_per_rank

    @parameter
    def make(r: Int) -> K:
        return K(x[r], gate[r], bias[r], cands_out[r],
                 r * epr, seq_len, 0, 0, 0)

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
        label="m3_router_expert",
    ](pools, prof)


@fieldwise_init
struct M3RouterMergeKernel[
    top_k: Int, scaling: Float32, o: ImmutOrigin,
](RangePartitionedKernel):
    var cands: Binding[M3RouterCandidate, Self.o]
    var nws: IntPtr
    var tp: Int
    var route_idx: I32Ptr
    var route_w: F32Ptr
    var seq_len: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime sentinel = Float32(-1.0e30)
        var local_nws = InlineArray[Int, MAX_MERGE_TP](uninitialized=True)
        for r in range(self.tp):
            local_nws[r] = self.nws[r]

        for tok in range(self.start, self.end):
            var merged = InlineArray[M3RouterCandidate, Self.top_k](
                fill=M3RouterCandidate(Int32(0), sentinel, Float32(0)))
            for r in range(self.tp):
                var base = self.cands[r]
                for w in range(local_nws[r]):
                    var src = base + (w * self.seq_len + tok) * Self.top_k
                    for k in range(Self.top_k):
                        var c = src[k]
                        insert_m3_candidate[Self.top_k](
                            c.expert, c.score, c.weight, merged)

            var sum_w = Float32(0)
            comptime for k in range(Self.top_k):
                sum_w += merged[k].weight
            var inv_sum = Float32(1.0) / sum_w

            var idx_dst = self.route_idx + tok * Self.top_k
            var w_dst = self.route_w + tok * Self.top_k
            comptime for k in range(Self.top_k):
                idx_dst[k] = merged[k].expert
                w_dst[k] = merged[k].weight * inv_sum * Self.scaling

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_merge_m3_router[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    top_k: Int, scaling: Float32, max_worker_count: Int = 128,
](
    cands: Binding[M3RouterCandidate, o],
    nws: List[Int],
    route_idx: Binding[Int32, o],
    route_w: Binding[Float32, o],
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if seq_len <= 0:
        return
    var tp = len(pools)
    var total_sources = 0
    for r in range(tp):
        total_sources += nws[r]
    var data_bytes = total_sources * seq_len * top_k * size_of[
        M3RouterCandidate]()

    var nws_ptr = IntPtr(unsafe_from_address=Int(nws.unsafe_ptr()))
    var per_node = (seq_len + tp - 1) // tp
    comptime MK = M3RouterMergeKernel[top_k, scaling, o]
    comptime GK = RouteGatherKernel[top_k, o]

    var span_a = DispatchSpan[Profile]()
    var buf_a = DispatchBuffer[MK, max_worker_count]()
    for r in range(tp):
        var lo = r * per_node
        var hi = min(lo + per_node, seq_len)
        var cnt = hi - lo
        if cnt <= 0:
            continue
        var cap = min(max_worker_count, pools[r].get_capacity())
        _ = tile_dispatch(buf_a,
            MK(cands, nws_ptr, tp, route_idx[r], route_w[r], seq_len, 0, 0),
            pools[r], cnt, base=lo, num_workers=merge_workers(data_bytes, cap))
    span_a.issued()
    join_all(pools)
    span_a.finish(prof, pools, "m3_router_merge.reduce")

    if tp <= 1:
        return

    var route_bytes = seq_len * top_k * size_of[Int32]()
    var span_b = DispatchSpan[Profile]()
    var buf_b = DispatchBuffer[GK, max_worker_count]()
    for r in range(tp):
        var cap = min(max_worker_count, pools[r].get_capacity())
        _ = tile_dispatch(buf_b,
            GK(route_idx, route_w, r, per_node, tp, 0, 0),
            pools[r], seq_len, num_workers=merge_workers(route_bytes, cap))
    span_b.issued()
    join_all(pools)
    span_b.finish(prof, pools, "m3_router_merge.broadcast")


comptime M3_HIDDEN = 6144
comptime M3_NUM_EXPERTS = 128
comptime M3_TOP_K = 4
comptime M3_ROUTED_SCALING = Float32(2.0)
comptime M3_ROUTER_PU = 4


def dispatch_minimax_m3_router[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
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
    var nws = dispatch_m3_router_expert[
        hidden=M3_HIDDEN, top_k=M3_TOP_K, port_unroll=M3_ROUTER_PU,
        max_worker_count=max_worker_count,
    ](x, gate, bias, cands, experts_per_rank, seq_len, pools, prof)

    dispatch_merge_m3_router[
        top_k=M3_TOP_K, scaling=M3_ROUTED_SCALING,
        max_worker_count=max_worker_count,
    ](cands, nws, route_idx, route_w, seq_len, pools, prof)
