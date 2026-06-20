from std.math import exp
from std.memory import Span, UnsafePointer

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.helpers import Binding, RankView
from kernels.profiling import Profiler

from prototypes.swiglu_oai import (
    dispatch_minimax_m3_swiglu_gate_up,
    M3_SWIGLU_ALPHA, M3_SWIGLU_LIMIT,
)


comptime ALIGNMENT = 64
comptime SEQ_LEN = 256
comptime ROW = 3072


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
def gate_val(t: Int, i: Int) -> Float32:
    return Float32(-12.0) + Float32(0.001) * Float32((t * 7 + i * 13) % 24000)


@always_inline
def up_val(t: Int, i: Int) -> Float32:
    return Float32(-12.0) + Float32(0.0017) * Float32((t * 11 + i * 5) % 14000)


@always_inline
def swiglu_ref(g: Float32, u: Float32) -> Float32:
    var gc = min(g, M3_SWIGLU_LIMIT)
    var uc = max(-M3_SWIGLU_LIMIT, min(u, M3_SWIGLU_LIMIT))
    var glu = gc / (Float32(1.0) + exp(-M3_SWIGLU_ALPHA * gc))
    return (uc + Float32(1.0)) * glu


def run_swiglu[P: BurstThreadPool, //](
    var pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    print(t"  distributing over degree={tp} (replicated activation)")
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var gate_ptr = arena_alloc_all[BFloat16](arenas, SEQ_LEN * ROW)
    var up_ptr = arena_alloc_all[BFloat16](arenas, SEQ_LEN * ROW)
    var dst_ptr = arena_alloc_all[BFloat16](arenas, SEQ_LEN * ROW)

    for r in range(tp):
        var gb = view.bind(gate_ptr)[r]
        var ub = view.bind(up_ptr)[r]
        var db = view.bind(dst_ptr)[r]
        for t in range(SEQ_LEN):
            for i in range(ROW):
                gb[t * ROW + i] = BFloat16(gate_val(t, i))
                ub[t * ROW + i] = BFloat16(up_val(t, i))
                db[t * ROW + i] = BFloat16(Float32(-99.0))
        _ = arenas[r].prefault(0, arenas[r].used())

    var prof = Profiler[False]()
    var gate = view.bind(gate_ptr)
    var up = view.bind(up_ptr)
    var dst = view.bind(dst_ptr)

    dispatch_minimax_m3_swiglu_gate_up(
        gate, up, dst, ROW, SEQ_LEN, pools, prof)

    var out = dst[0]
    var g_in = gate[0]
    var u_in = up[0]
    var worst = Float32(0)
    var mismatches = 0
    var gate_clamped = 0
    var up_clamped = 0
    for t in range(SEQ_LEN):
        for i in range(ROW):
            var g = g_in[t * ROW + i].cast[DType.float32]()
            var u = u_in[t * ROW + i].cast[DType.float32]()
            if g > M3_SWIGLU_LIMIT:
                gate_clamped += 1
            if u > M3_SWIGLU_LIMIT or u < -M3_SWIGLU_LIMIT:
                up_clamped += 1
            var want = swiglu_ref(g, u)
            var got = out[t * ROW + i].cast[DType.float32]()
            var diff = abs(want - got)
            if diff > worst:
                worst = diff
            if diff > Float32(1e-2) + Float32(0.02) * abs(want):
                if mismatches < 8:
                    print(t"  MISMATCH tok={t} i={i} want={want} got={got}")
                mismatches += 1

    print(t"  coverage: gate_clamped={gate_clamped} up_clamped={up_clamped}")
    if mismatches == 0:
        print(t"smoke: PASS (degree={tp}, worst_abs_diff={worst})")
    else:
        print(t"smoke: FAIL (degree={tp}, mismatches={mismatches}, worst={worst})")


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    print(t"swiglu-oai smoke: {tp} NUMA node(s), seq_len={SEQ_LEN}, row={ROW}")

    comptime ARENA_BYTES = 64 * 1024 * 1024
    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_smoke[P: BurstThreadPool, //](var selected_pools: List[P]):
        run_swiglu(selected_pools^, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_smoke,
    ](topo, "mode: isolated", "mode: spin-backoff")
