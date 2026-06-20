from std.collections import InlineArray
from std.math import exp
from std.memory import Span, UnsafePointer

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.helpers import Binding, RankView, BF16Ptr, F32Ptr, I32Ptr, W
from kernels.moe_router import SparseRoute, SparseRoutePtr
from kernels.profiling import Profiler

from prototypes.moe_experts_oai import (
    dispatch_minimax_m3_moe_experts,
    M3_MOE_HIDDEN, M3_MOE_INTERMEDIATE,
)
from prototypes.swiglu_oai import M3_SWIGLU_ALPHA, M3_SWIGLU_LIMIT


comptime ALIGNMENT = 64
comptime HIDDEN = M3_MOE_HIDDEN
comptime INTERMEDIATE = M3_MOE_INTERMEDIATE
comptime GATE_UP_FUSED = 2 * INTERMEDIATE
comptime NUM_EXPERTS = 8
comptime TOP_K = 4
comptime SEQ_LEN = 64
comptime MAX_WORKERS = 256
comptime SCRATCH_PER_WORKER = 4 * 2 * 64


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


def fill_bf16(ptr: BF16Ptr, count: Int, period: Int, center: Int, scale: Float32):
    for i in range(count):
        ptr[i] = BFloat16(Float32((i % period) - center) * scale)


@always_inline
def ref_dot(a: BF16Ptr, b: BF16Ptr, n: Int) -> Float32:
    var acc = SIMD[DType.float32, W](0)
    var j = 0
    while j + W <= n:
        var av = (a + j).load[width=W]().cast[DType.float32]()
        var bv = (b + j).load[width=W]().cast[DType.float32]()
        acc = av * bv + acc
        j += W
    var total = acc.reduce_add()
    while j < n:
        total += a[j].cast[DType.float32]() * b[j].cast[DType.float32]()
        j += 1
    return total


@always_inline
def swiglu_ref(g: Float32, u: Float32) -> Float32:
    var gc = min(g, M3_SWIGLU_LIMIT)
    var uc = max(-M3_SWIGLU_LIMIT, min(u, M3_SWIGLU_LIMIT))
    var glu = gc / (Float32(1.0) + exp(-M3_SWIGLU_ALPHA * gc))
    return (uc + Float32(1.0)) * glu


def build_routing(
    seq_len: Int, first: Int, last: Int,
    expert_offset: I32Ptr, routes: SparseRoutePtr, weight: Float32,
):
    var epr = last - first
    var counts = InlineArray[Int, NUM_EXPERTS](fill=0)
    for tok in range(seq_len):
        for k in range(TOP_K):
            var e = (tok * TOP_K + k) % NUM_EXPERTS
            if e >= first and e < last:
                counts[e - first] += 1

    var running = Int32(0)
    var write_ofs = InlineArray[Int, NUM_EXPERTS](fill=0)
    for e in range(epr):
        expert_offset[e] = running
        write_ofs[e] = Int(running)
        running += Int32(counts[e])
    expert_offset[epr] = running

    for tok in range(seq_len):
        for k in range(TOP_K):
            var e = (tok * TOP_K + k) % NUM_EXPERTS
            if e >= first and e < last:
                var local = e - first
                var pos = write_ofs[local]
                routes[pos] = SparseRoute(Int32(tok), weight)
                write_ofs[local] = pos + 1


def run_experts[P: BurstThreadPool, //](
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
    var ofs_ptr = arena_alloc_all[Int32](arenas, epr + 1)
    var routes_ptr = arena_alloc_all[SparseRoute](arenas, SEQ_LEN * TOP_K)
    var gate_up_ptr = arena_alloc_all[BFloat16](
        arenas, epr * GATE_UP_FUSED * HIDDEN)
    var down_ptr = arena_alloc_all[BFloat16](
        arenas, epr * HIDDEN * INTERMEDIATE)
    var scratch_ptr = arena_alloc_all[Float32](
        arenas, MAX_WORKERS * SCRATCH_PER_WORKER)
    var bucket_ptr = arena_alloc_all[BFloat16](
        arenas, SEQ_LEN * TOP_K * INTERMEDIATE)
    var accum_ptr = arena_alloc_all[Float32](arenas, SEQ_LEN * HIDDEN)
    var out_ptr = arena_alloc_all[BFloat16](arenas, SEQ_LEN * HIDDEN)

    var weight = Float32(1.0) / Float32(TOP_K)
    for r in range(tp):
        fill_bf16(view.bind(x_ptr)[r], SEQ_LEN * HIDDEN, 17, 8, 0.02)
        fill_bf16(view.bind(gate_up_ptr)[r],
            epr * GATE_UP_FUSED * HIDDEN, 23, 11, 0.01)
        fill_bf16(view.bind(down_ptr)[r],
            epr * HIDDEN * INTERMEDIATE, 19, 9, 0.01)
        var first = r * epr
        build_routing(SEQ_LEN, first, first + epr,
            view.bind(ofs_ptr)[r], view.bind(routes_ptr)[r], weight)
        _ = arenas[r].prefault(0, arenas[r].used())

    var prof = Profiler[False]()
    var x = view.bind(x_ptr)
    var ofs = view.bind(ofs_ptr)
    var routes = view.bind(routes_ptr)
    var gate_up = view.bind(gate_up_ptr)
    var down = view.bind(down_ptr)
    var scratch = view.bind(scratch_ptr)
    var bucket = view.bind(bucket_ptr)
    var accum = view.bind(accum_ptr)
    var out = view.bind(out_ptr)

    dispatch_minimax_m3_moe_experts[hidden=HIDDEN, intermediate=INTERMEDIATE](
        x, ofs, routes, gate_up, down, scratch, bucket, accum, out,
        epr, SEQ_LEN, pools, prof)

    var ok = True
    var worst = Float32(0)
    var bad = 0
    var h = InlineArray[BFloat16, INTERMEDIATE](uninitialized=True)
    for r in range(tp):
        var first = r * epr
        var last = first + epr
        var x_r = x[r]
        var gu_r = gate_up[r]
        var dn_r = down[r]
        var out_r = out[r]
        for t in range(SEQ_LEN):
            var x_row = x_r + t * HIDDEN
            var out_ref = InlineArray[Float32, HIDDEN](fill=Float32(0))
            for k in range(TOP_K):
                var e = (t * TOP_K + k) % NUM_EXPERTS
                if e < first or e >= last:
                    continue
                var le = e - first
                var gu_w = gu_r + le * GATE_UP_FUSED * HIDDEN
                var dn_w = dn_r + le * HIDDEN * INTERMEDIATE
                for m in range(INTERMEDIATE):
                    var g = ref_dot(x_row, gu_w + m * HIDDEN, HIDDEN)
                    var u = ref_dot(
                        x_row, gu_w + (INTERMEDIATE + m) * HIDDEN, HIDDEN)
                    h[m] = BFloat16(swiglu_ref(g, u))
                var h_ptr = h.unsafe_ptr().as_unsafe_any_origin()
                for hd in range(HIDDEN):
                    var acc = ref_dot(
                        h_ptr, dn_w + hd * INTERMEDIATE, INTERMEDIATE)
                    out_ref[hd] += weight * acc

            for hd in range(HIDDEN):
                var want = out_ref[hd]
                var got = out_r[t * HIDDEN + hd].cast[DType.float32]()
                var diff = abs(want - got)
                if diff > worst:
                    worst = diff
                if diff > Float32(5e-4) + Float32(0.003) * abs(want):
                    ok = False
                    if bad < 8:
                        print(t"  MISMATCH r={r} t={t} hd={hd} "
                              t"want={want} got={got}")
                    bad += 1

    if ok:
        print(t"smoke: PASS (degree={tp}, worst_abs_diff={worst})")
    else:
        print(t"smoke: FAIL (degree={tp}, bad={bad}, worst_abs_diff={worst})")


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    print(t"moe-experts-oai smoke: {tp} NUMA node(s), seq_len={SEQ_LEN}")

    var epr_max = NUM_EXPERTS
    if tp > 0:
        epr_max = NUM_EXPERTS // tp + 1
    var weight_bytes = epr_max * (GATE_UP_FUSED * HIDDEN + HIDDEN * INTERMEDIATE) * 2
    var arena_bytes = weight_bytes + 128 * 1024 * 1024

    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], arena_bytes))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_smoke[P: BurstThreadPool, //](var selected_pools: List[P]):
        run_experts(selected_pools^, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_smoke,
    ](topo, "mode: isolated", "mode: spin-backoff")
