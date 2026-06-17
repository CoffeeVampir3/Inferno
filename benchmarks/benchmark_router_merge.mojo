from std.collections import InlineArray
from std.memory import Span, UnsafePointer
from std.sys.info import size_of
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from kernels.helpers import (
    RangePartitionedKernel, DispatchBuffer, tile_dispatch, join_all,
    Binding, RankView, I32Ptr, F32Ptr,
)
from kernels.profiling import Profiler
from kernels.moe_router import RouterCandidate, dispatch_merge_router_candidates
from modeling.gemma4_common import Gemma4BaseConfig
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
)


comptime NUM_EXPERTS = Gemma4BaseConfig.NUM_EXPERTS
comptime TOP_K = Gemma4BaseConfig.TOP_K

comptime ALIGNMENT = 64
comptime WARMUP = 30
comptime SAMPLES = 400
comptime MAX_SEQ = 4096
comptime NWS = 8
comptime CAND_BYTES = size_of[RouterCandidate]()

comptime NUM_SEQ_SIZES = 5

comptime RouterCandidatePtr = UnsafePointer[RouterCandidate, MutAnyOrigin]


@fieldwise_init
struct FillCandsKernel[top_k: Int](RangePartitionedKernel):
    var base: RouterCandidatePtr
    var expert_base: Int
    var experts_per_rank: Int
    var seq_len: Int
    var num_sources: Int
    var start: Int
    var end: Int

    def execute(mut self):
        for tok in range(self.start, self.end):
            for w in range(self.num_sources):
                var slot = (w * self.seq_len + tok) * Self.top_k
                for k in range(Self.top_k):
                    var e = self.expert_base + (
                        (w * 13 + k * 7 + tok) % self.experts_per_rank)
                    var logit = Float32(
                        ((w * 31 + k * 17 + tok * 5) % 211) - 105) * 0.01
                    (self.base + slot + k)[] = RouterCandidate(Int32(e), logit)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_fill_cands[
    P: BurstThreadPool, o: ImmutOrigin, //,
](
    cands: Binding[RouterCandidate, o],
    experts_per_rank: Int,
    seq_len: Int,
    num_sources: Int,
    mut pools: List[P],
):
    comptime K = FillCandsKernel[TOP_K]
    var buf = DispatchBuffer[K]()
    for r in range(len(pools)):
        var cap = pools[r].get_capacity()
        _ = tile_dispatch(buf,
            K(cands[r], r * experts_per_rank, experts_per_rank,
              seq_len, num_sources, 0, 0),
            pools[r], seq_len, num_workers=cap)
    join_all(pools)


def arena_alloc[T: AnyType](
    mut arena: NumaArena[alignment=ALIGNMENT], count: Int,
) -> UnsafePointer[T, MutAnyOrigin]:
    var ptr = arena.alloc[T](count)
    if not ptr:
        print("arena alloc failed for", count, "elements")
        return UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
    return ptr.value()


def arena_alloc_all[T: AnyType](
    mut arenas: List[NumaArena[alignment=ALIGNMENT]], count: Int,
) -> UnsafePointer[T, MutAnyOrigin]:
    var first = UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
    for r in range(len(arenas)):
        var ptr = arena_alloc[T](arenas[r], count)
        if r == 0:
            first = ptr
    return first


def arena_bases(
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
) -> List[Int]:
    var bases = List[Int](capacity=len(arenas))
    for r in range(len(arenas)):
        bases.append(Int(arenas[r].base.value()))
    return bases^


def fill_scale[o: ImmutOrigin](scale: Binding[BFloat16, o]):
    for r in range(scale.degree()):
        for e in range(NUM_EXPERTS):
            scale[r][e] = BFloat16(1.0)


def verify_softmax(idx: I32Ptr, w: F32Ptr, seq_len: Int) -> Bool:
    for tok in range(seq_len):
        var sum_w = Float32(0)
        for k in range(TOP_K):
            var e = idx[tok * TOP_K + k]
            if e < 0 or Int(e) >= NUM_EXPERTS:
                print("    invalid expert at tok", tok, "slot", k, ":", e)
                return False
            sum_w += w[tok * TOP_K + k]
        var d = sum_w - Float32(1.0)
        if d < 0:
            d = -d
        if d > Float32(1e-3):
            print("    softmax sum off at tok", tok, ":", sum_w)
            return False
    return True


def run_all[P: BurstThreadPool, //](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    var experts_per_rank = NUM_EXPERTS // tp
    var total_sources = tp * NWS

    var sizes = InlineArray[Int, NUM_SEQ_SIZES](fill=0)
    sizes[0] = 16; sizes[1] = 64; sizes[2] = 256
    sizes[3] = 1024; sizes[4] = MAX_SEQ

    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var cands_ptr = arena_alloc_all[RouterCandidate](
        arenas, NWS * MAX_SEQ * TOP_K)
    var scale_ptr = arena_alloc_all[BFloat16](arenas, NUM_EXPERTS)
    var route_idx_ptr = arena_alloc_all[Int32](arenas, MAX_SEQ * TOP_K)
    var route_w_ptr = arena_alloc_all[Float32](arenas, MAX_SEQ * TOP_K)

    var cands = view.bind(cands_ptr)
    var scale = view.bind(scale_ptr)
    var route_idx = view.bind(route_idx_ptr)
    var route_w = view.bind(route_w_ptr)

    fill_scale(scale)

    var pool_cap = pools[0].get_capacity()
    print(t"experts/rank={experts_per_rank} top_k={TOP_K} sources={total_sources}")
    print(t"pool_capacity={pool_cap} across {tp} node(s)")

    var nws_list = List[Int]()
    for _ in range(tp):
        nws_list.append(NWS)

    var samples = SampleBuffer(SAMPLES)
    var prof = Profiler[False]()

    for s in range(NUM_SEQ_SIZES):
        var seq = sizes[s]
        var data_bytes = total_sources * seq * TOP_K * CAND_BYTES
        dispatch_fill_cands(cands, experts_per_rank, seq, NWS, pools)
        for r in range(tp):
            _ = arenas[r].prefault(0, arenas[r].used())

        print(t"\n=== seq_len={seq}  candidate_reads={data_bytes} bytes ===")

        for _ in range(WARMUP):
            dispatch_merge_router_candidates[TOP_K](
                cands, nws_list, scale, route_idx, route_w, seq, pools, prof)
            keep(route_idx[0])

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            dispatch_merge_router_candidates[TOP_K](
                cands, nws_list, scale, route_idx, route_w, seq, pools, prof)
            var t1 = now_ns()
            var t_done = max_last_ts(pools)
            samples.push(t_done - t0, t1 - t0)
        keep(route_idx[0])

        var ks = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        print_row(String("  reduce+broadcast"), ks, ws, data_bytes)

        if verify_softmax(route_idx_ptr, route_w_ptr, seq):
            print("  verify: PASS (valid top-k softmax routing)")
        else:
            print("  verify: FAIL")


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("router merge benchmark: reduce+broadcast across NUMA nodes")
    var iso = len(topo.isolated_cpus)
    print(t"{tp} NUMA node(s), {iso} isolated cpus\n")

    comptime ARENA_BYTES = 64 * 1024 * 1024
    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_router_merge_bench[P: BurstThreadPool, //](
        var selected_pools: List[P],
    ):
        run_all(selected_pools, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_router_merge_bench,
    ](topo, "mode: isolated", "mode: spin-backoff")
