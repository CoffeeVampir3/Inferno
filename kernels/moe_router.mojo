from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import size_of

from threading.threading_traits import BurstThreadPool
from simd_math import fast_exp_softmax_biased
from simd_math.ops import sqrt
from .helpers import (
    WorkerRangePartitionedKernel, RangePartitionedKernel, Binding,
    fanout_dispatch_per_rank, recommended_workers,
    DispatchBuffer, tile_dispatch, join_all,
    BF16Ptr, F32Ptr, I32Ptr, W,
)
from .dispatch_heuristics import (
    ROUTER_DISPATCH_BW_PRODUCT, ROUTER_MAX_WORKERS, SCHEDULE_INLINE_TOKENS,
)
from .logsum_merge import merge_workers
from .rmsnorm import rms_reduce_row
from .dot_products import dot_to_scalar
from .profiling import Profiler, DispatchSpan


comptime IntPtr = UnsafePointer[Int, MutUntrackedOrigin]
comptime MAX_MERGE_TP = 64


@fieldwise_init
struct RouterCandidate(Copyable, ImplicitlyCopyable):
    var expert: Int32
    var logit: Float32


@fieldwise_init
struct SparseRoute(Copyable, ImplicitlyCopyable):
    var token: Int32
    var weight: Float32


comptime RouterCandidatePtr = UnsafePointer[RouterCandidate, MutUntrackedOrigin]
comptime SparseRoutePtr = UnsafePointer[SparseRoute, MutUntrackedOrigin]


@always_inline
def insert_candidate[top_k: Int](
    expert: Int32,
    logit: Float32,
    mut cands: InlineArray[RouterCandidate, top_k],
):
    for k in range(top_k):
        var c = cands[k]
        if logit > c.logit or (logit == c.logit and Int(expert) < Int(c.expert)):
            var j = top_k - 1
            while j > k:
                cands[j] = cands[j - 1]
                j -= 1
            cands[k] = RouterCandidate(expert, logit)
            return


@fieldwise_init
struct BuildScheduleKernel[max_experts: Int, top_k: Int](RangePartitionedKernel):
    var route_idx: I32Ptr
    var route_w: F32Ptr
    var expert_offset: I32Ptr
    var routes: SparseRoutePtr
    var expert_base: Int
    var experts_per_rank: Int
    var seq_len: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var counts = InlineArray[Int32, Self.max_experts](fill=Int32(0))
        var first = self.expert_base
        var last = first + self.experts_per_rank

        for tok in range(self.seq_len):
            for k in range(Self.top_k):
                var e = Int(self.route_idx[tok * Self.top_k + k])
                if e >= first and e < last:
                    counts[e - first] += Int32(1)

        var running = Int32(0)
        var write_offsets = InlineArray[Int32, Self.max_experts](
            uninitialized=True)
        for e in range(self.experts_per_rank):
            self.expert_offset[e] = running
            write_offsets[e] = running
            running += counts[e]
        self.expert_offset[self.experts_per_rank] = running

        for tok in range(self.seq_len):
            for k in range(Self.top_k):
                var slot = tok * Self.top_k + k
                var e = Int(self.route_idx[slot])
                if e >= first and e < last:
                    var local = e - first
                    var pos = Int(write_offsets[local])
                    self.routes[pos] = SparseRoute(
                        Int32(tok), self.route_w[slot])
                    write_offsets[local] = Int32(pos + 1)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_build_expert_schedules[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_experts: Int, top_k: Int, max_worker_count: Int = 128,
](
    route_idx: Binding[Int32, o],
    route_w: Binding[Float32, o],
    expert_offset: Binding[Int32, o],
    routes: Binding[SparseRoute, o],
    experts_per_rank: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if seq_len <= 0:
        return
    var tp = len(pools)
    comptime K = BuildScheduleKernel[max_experts, top_k]

    if seq_len <= SCHEDULE_INLINE_TOKENS:
        var span = DispatchSpan[Profile]()
        for r in range(tp):
            var item = K(route_idx[r], route_w[r], expert_offset[r], routes[r],
                         r * experts_per_rank, experts_per_rank, seq_len, 0, 0)
            item.execute()
        span.finish_inline(prof, "build_schedules")
        return

    var span = DispatchSpan[Profile]()
    var buf = DispatchBuffer[K, max_worker_count]()
    for r in range(tp):
        _ = tile_dispatch(buf,
            K(route_idx[r], route_w[r], expert_offset[r], routes[r],
              r * experts_per_rank, experts_per_rank, seq_len, 0, 0),
            pools[r], 1, num_workers=1)
    span.issued()
    join_all(pools)
    span.finish(prof, pools, "build_schedules")


@fieldwise_init
struct RouterExpertKernel[
    hidden: Int, sqrt_n: Float32, n_eps: Float32,
    top_k: Int,
](WorkerRangePartitionedKernel):
    var x: BF16Ptr
    var router_proj: BF16Ptr
    var router_scale: BF16Ptr
    var scaled_scratch: F32Ptr
    var cands_out: RouterCandidatePtr
    var expert_base: Int
    var seq_len: Int
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime sentinel = Float32(-1.0e30)
        var scratch = self.scaled_scratch + self.worker_id * Self.hidden
        for tok in range(self.seq_len):
            var x_row = self.x + tok * Self.hidden

            var sum_sq = rms_reduce_row[Self.hidden](x_row)
            var inv_rms = Self.sqrt_n / sqrt[DType.float32, 1](
                sum_sq + Self.n_eps)
            var inv_vec = SIMD[DType.float32, W](inv_rms)
            for j in range(0, Self.hidden, W):
                var xv = (x_row + j).load[width=W]().cast[DType.float32]()
                var sv = (self.router_scale + j).load[width=W]().cast[
                    DType.float32]()
                (scratch + j).store(xv * sv * inv_vec)

            var cands = InlineArray[RouterCandidate, Self.top_k](
                fill=RouterCandidate(Int32(0), sentinel))
            for e in range(self.start, self.end):
                var row = self.router_proj + e * Self.hidden
                var logit = dot_to_scalar[Self.hidden](scratch, row)
                insert_candidate[Self.top_k](
                    Int32(self.expert_base + e), logit, cands)

            var dst = self.cands_out + (
                self.worker_id * self.seq_len + tok) * Self.top_k
            comptime for k in range(Self.top_k):
                dst[k] = cands[k]

    @always_inline
    def install_worker_range(mut self, worker_id: Int, start: Int, end: Int):
        self.worker_id = worker_id
        self.start = start
        self.end = end


@always_inline
def router_workers(data_bytes: Int, capacity: Int) -> Int:
    return recommended_workers[ROUTER_DISPATCH_BW_PRODUCT](
        data_bytes, min(capacity, ROUTER_MAX_WORKERS))


def dispatch_router_expert[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    hidden: Int, sqrt_n: Float32, n_eps: Float32,
    top_k: Int,
    max_worker_count: Int = 128,
](
    x: Binding[BFloat16, o],
    router_proj: Binding[BFloat16, o],
    router_scale: Binding[BFloat16, o],
    scaled_scratch: Binding[Float32, o],
    cands_out: Binding[RouterCandidate, o],
    experts_per_rank: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
) -> List[Int]:
    comptime K = RouterExpertKernel[hidden, sqrt_n, n_eps, top_k]
    var epr = experts_per_rank

    @parameter
    def make(r: Int) -> K:
        return K(x[r], router_proj[r], router_scale[r],
                 scaled_scratch[r], cands_out[r],
                 r * epr, seq_len, 0, 0, 0)

    @parameter
    def total_for(r: Int) -> Int:
        return epr

    @parameter
    def data_bytes_for(r: Int) -> Int:
        return epr * hidden * 2

    return fanout_dispatch_per_rank[
        make, total_for, data_bytes_for,
        max_worker_count=max_worker_count,
        worker_policy=router_workers,
        label="router_expert",
    ](pools, prof)


@fieldwise_init
struct RouterMergeKernel[top_k: Int, o: ImmutOrigin](RangePartitionedKernel):
    var cands: Binding[RouterCandidate, Self.o]
    var nws: IntPtr
    var tp: Int
    var per_expert_scale: BF16Ptr
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
            var merged = InlineArray[RouterCandidate, Self.top_k](
                fill=RouterCandidate(Int32(0), sentinel))
            for r in range(self.tp):
                var base = self.cands[r]
                for w in range(local_nws[r]):
                    var src = base + (w * self.seq_len + tok) * Self.top_k
                    for k in range(Self.top_k):
                        var c = src[k]
                        insert_candidate[Self.top_k](c.expert, c.logit, merged)

            var max_logit = merged[0].logit
            var sum_v = Float32(0)
            var exp_values = InlineArray[Float32, Self.top_k](uninitialized=True)
            comptime for k in range(Self.top_k):
                var ev = fast_exp_softmax_biased[1](
                    SIMD[DType.float32, 1](merged[k].logit - max_logit))[0]
                exp_values[k] = ev
                sum_v += ev
            var inv_sum = Float32(1.0) / sum_v

            var idx_dst = self.route_idx + tok * Self.top_k
            var w_dst = self.route_w + tok * Self.top_k
            comptime for k in range(Self.top_k):
                var expert = merged[k].expert
                var scale = (self.per_expert_scale + Int(expert))[].cast[
                    DType.float32]()
                idx_dst[k] = expert
                w_dst[k] = exp_values[k] * inv_sum * scale

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct RouteGatherKernel[top_k: Int, o: ImmutOrigin](RangePartitionedKernel):
    var route_idx: Binding[Int32, Self.o]
    var route_w: Binding[Float32, Self.o]
    var dest_rank: Int
    var per_node: Int
    var tp: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var didx = self.route_idx[self.dest_rank]
        var dw = self.route_w[self.dest_rank]
        for tok in range(self.start, self.end):
            var owner = tok // self.per_node
            if owner >= self.tp:
                owner = self.tp - 1
            if owner == self.dest_rank:
                continue
            var sidx = self.route_idx[owner]
            var sw = self.route_w[owner]
            var off = tok * Self.top_k
            for k in range(Self.top_k):
                didx[off + k] = sidx[off + k]
                dw[off + k] = sw[off + k]

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_merge_router_candidates[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    top_k: Int, max_worker_count: Int = 128,
](
    cands: Binding[RouterCandidate, o],
    nws: List[Int],
    per_expert_scale: Binding[BFloat16, o],
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
    var data_bytes = total_sources * seq_len * top_k * size_of[RouterCandidate]()

    var nws_ptr = nws.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var per_node = (seq_len + tp - 1) // tp
    comptime MK = RouterMergeKernel[top_k, o]
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
            MK(cands, nws_ptr, tp, per_expert_scale[r],
               route_idx[r], route_w[r], seq_len, 0, 0),
            pools[r], cnt, base=lo, num_workers=merge_workers(data_bytes, cap))
    span_a.issued()
    join_all(pools)
    span_a.finish(prof, pools, "router_merge.reduce")

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
    span_b.finish(prof, pools, "router_merge.broadcast")
