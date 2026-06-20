from std.collections import InlineArray
from std.memory import Span, UnsafePointer

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.helpers import Binding, RankView
from kernels.profiling import Profiler

from prototypes.sigmoid_router import (
    dispatch_minimax_m3_router,
    M3RouterCandidate, insert_m3_candidate, dot_bf16_f32, sigmoid_f32,
    M3_HIDDEN, M3_NUM_EXPERTS, M3_TOP_K, M3_ROUTED_SCALING, M3_ROUTER_PU,
)


comptime ALIGNMENT = 64
comptime SEQ_LEN = 256
comptime HIDDEN = M3_HIDDEN
comptime NUM_EXPERTS = M3_NUM_EXPERTS
comptime TOP_K = M3_TOP_K
comptime MAX_WORKERS = 128


def arena_bases(
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
) -> List[Int]:
    var bases = List[Int](capacity=len(arenas))
    for r in range(len(arenas)):
        bases.append(Int(arenas[r].base.value()))
    return bases^


def arena_alloc_all[T: AnyType](
    mut arenas: List[NumaArena[alignment=ALIGNMENT]], count: Int,
) -> UnsafePointer[T, MutAnyOrigin]:
    var first = UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
    for r in range(len(arenas)):
        var p = arenas[r].alloc[T](count)
        if not p:
            print("arena alloc failed for", count, "elements")
            return UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
        if r == 0:
            first = p.value()
    return first


@always_inline
def x_val(t: Int, j: Int) -> BFloat16:
    return BFloat16(Float32((t * 3 + j) % 17 - 8) * 0.05)


@always_inline
def gate_val(e: Int, j: Int) -> Float32:
    return Float32(((e * 13 + j * 7) % 23) - 11) * 0.02


@always_inline
def bias_val(e: Int) -> Float32:
    return Float32((e % 9) - 4) * 0.05


def run_router[P: BurstThreadPool, //](
    var pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    if NUM_EXPERTS % tp != 0:
        print(t"  degree={tp} does not divide num_experts={NUM_EXPERTS}; skip")
        return
    var epr = NUM_EXPERTS // tp
    print(t"  distributing over degree={tp} (experts sharded: {epr}/rank)")
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var x_ptr = arena_alloc_all[BFloat16](arenas, SEQ_LEN * HIDDEN)
    var gate_ptr = arena_alloc_all[Float32](arenas, epr * HIDDEN)
    var bias_ptr = arena_alloc_all[Float32](arenas, NUM_EXPERTS)
    var cands_ptr = arena_alloc_all[M3RouterCandidate](
        arenas, MAX_WORKERS * SEQ_LEN * TOP_K)
    var route_idx_ptr = arena_alloc_all[Int32](arenas, SEQ_LEN * TOP_K)
    var route_w_ptr = arena_alloc_all[Float32](arenas, SEQ_LEN * TOP_K)

    for r in range(tp):
        var xb = view.bind(x_ptr)[r]
        for t in range(SEQ_LEN):
            for j in range(HIDDEN):
                xb[t * HIDDEN + j] = x_val(t, j)
        var gb = view.bind(gate_ptr)[r]
        for e in range(epr):
            var ge = r * epr + e
            for j in range(HIDDEN):
                gb[e * HIDDEN + j] = gate_val(ge, j)
        var bb = view.bind(bias_ptr)[r]
        for e in range(NUM_EXPERTS):
            bb[e] = bias_val(e)
        var rib = view.bind(route_idx_ptr)[r]
        var rwb = view.bind(route_w_ptr)[r]
        for i in range(SEQ_LEN * TOP_K):
            rib[i] = Int32(-2)
            rwb[i] = Float32(-1)
        _ = arenas[r].prefault(0, arenas[r].used())

    var prof = Profiler[False]()
    var x = view.bind(x_ptr)
    var gate = view.bind(gate_ptr)
    var bias = view.bind(bias_ptr)
    var cands = view.bind(cands_ptr)
    var route_idx = view.bind(route_idx_ptr)
    var route_w = view.bind(route_w_ptr)

    dispatch_minimax_m3_router(
        x, gate, bias, cands, route_idx, route_w, epr, SEQ_LEN, pools, prof)

    var x0 = x[0]
    var ridx = route_idx[0]
    var rw = route_w[0]
    var ok = True
    var worst = Float32(0)
    var bad = 0
    var bias_flips = 0
    for t in range(SEQ_LEN):
        var x_row = x0 + t * HIDDEN

        var sel = InlineArray[M3RouterCandidate, TOP_K](
            fill=M3RouterCandidate(Int32(0), Float32(-1.0e30), Float32(0)))
        var pure = InlineArray[M3RouterCandidate, TOP_K](
            fill=M3RouterCandidate(Int32(0), Float32(-1.0e30), Float32(0)))
        for ge in range(NUM_EXPERTS):
            var src_rank = ge // epr
            var local = ge % epr
            var gate_row = gate[src_rank] + local * HIDDEN
            var weight = sigmoid_f32(
                dot_bf16_f32[HIDDEN, M3_ROUTER_PU](x_row, gate_row))
            insert_m3_candidate[TOP_K](
                Int32(ge), weight + bias_val(ge), weight, sel)
            insert_m3_candidate[TOP_K](Int32(ge), weight, weight, pure)

        var sum_w = Float32(0)
        for k in range(TOP_K):
            sum_w += sel[k].weight
        var inv = Float32(1.0) / sum_w

        var flipped = False
        for k in range(TOP_K):
            if sel[k].expert != pure[k].expert:
                flipped = True
            var want_e = sel[k].expert
            var want_w = sel[k].weight * inv * M3_ROUTED_SCALING
            var got_e = ridx[t * TOP_K + k]
            var got_w = rw[t * TOP_K + k]
            if got_e != want_e:
                ok = False
                if bad < 8:
                    print(t"  IDX MISMATCH t={t} k={k} want={want_e} got={got_e}")
                bad += 1
            var diff = abs(want_w - got_w)
            if diff > worst:
                worst = diff
            if diff > Float32(1e-4):
                ok = False
                if bad < 8:
                    print(t"  W MISMATCH t={t} k={k} want={want_w} got={got_w}")
                bad += 1
        if flipped:
            bias_flips += 1

    print(t"  bias changed selection on {bias_flips}/{SEQ_LEN} tokens")
    if ok:
        print(t"smoke: PASS (degree={tp}, worst_w_diff={worst})")
    else:
        print(t"smoke: FAIL (degree={tp}, bad={bad}, worst_w_diff={worst})")


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    print(t"sigmoid router smoke: {tp} NUMA node(s), seq_len={SEQ_LEN}")

    comptime ARENA_BYTES = 64 * 1024 * 1024
    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_smoke[P: BurstThreadPool, //](var selected_pools: List[P]):
        run_router(selected_pools^, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_smoke,
    ](topo, "mode: isolated", "mode: spin-backoff")
