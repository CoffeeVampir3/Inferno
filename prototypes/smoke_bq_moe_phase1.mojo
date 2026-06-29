from std.memory import Span, UnsafePointer

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.helpers import Binding, RankView
from kernels.moe_router import SparseRoute
from kernels.profiling import Profiler
from kernels.elementwise import swiglu_oai_activate
from simd_math.ops import gelu_tanh_f32

from butterquant.weight import ButterquantWeight, ButterquantActivation
from quant.recipe import (
    QuantRecipe, PerRowQuant, NoGamma, SingleSided, PerRowCs, VnniPacked,
)

from prototypes.bq_moe_phase1 import (
    dispatch_bq_phase1_gate_up_act, M3_SWIGLU_ALPHA, M3_SWIGLU_LIMIT,
)


comptime ALIGNMENT = 64
comptime HIDDEN = 256
comptime INTER = 128
comptime GATE_UP = 2 * INTER
comptime EXPERTS = 1
comptime N_TOK = 20
comptime X_SA = Float32(2.0)
comptime W_SCALE = Float32(1.0)

comptime Recipe: QuantRecipe = PerRowQuant(
    128, NoGamma(), SingleSided(), PerRowCs(), VnniPacked())

comptime I8Ptr = UnsafePointer[Int8, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]


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
def golden_swiglu(g: Float32, u: Float32) -> Float32:
    return swiglu_oai_activate[1, M3_SWIGLU_ALPHA, M3_SWIGLU_LIMIT](
        SIMD[DType.float32, 1](g), SIMD[DType.float32, 1](u))[0]


@always_inline
def golden_gelu(g: Float32, u: Float32) -> Float32:
    return gelu_tanh_f32[1](SIMD[DType.float32, 1](g))[0] * u


def run_phase1[P: BurstThreadPool, //](
    var pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    print(t"  degree={tp}")
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var x_i8_ptr = arena_alloc_all[Int8](arenas, N_TOK * HIDDEN)
    var x_sa_ptr = arena_alloc_all[Float32](arenas, N_TOK)
    var w_ptr = arena_alloc_all[Int8](arenas, EXPERTS * GATE_UP * HIDDEN)
    var wsc_ptr = arena_alloc_all[Float32](arenas, EXPERTS * GATE_UP)
    var cs_ptr = arena_alloc_all[Float32](arenas, EXPERTS * GATE_UP)
    var eoff_ptr = arena_alloc_all[Int32](arenas, EXPERTS + 1)
    var routes_ptr = arena_alloc_all[SparseRoute](arenas, N_TOK)
    var gate_ptr = arena_alloc_all[BFloat16](arenas, N_TOK * INTER)
    var up_ptr = arena_alloc_all[BFloat16](arenas, N_TOK * INTER)
    var swi_ptr = arena_alloc_all[BFloat16](arenas, N_TOK * INTER)
    var gelu_ptr = arena_alloc_all[BFloat16](arenas, N_TOK * INTER)

    for r in range(tp):
        var xb = view.bind(x_i8_ptr)[r]
        for t in range(N_TOK):
            for c in range(HIDDEN):
                xb[t * HIDDEN + c] = Int8((t + c) % 7 - 3)
        var sab = view.bind(x_sa_ptr)[r]
        for t in range(N_TOK):
            sab[t] = X_SA

        var wb = view.bind(w_ptr)[r]
        for row in range(EXPERTS * GATE_UP):
            for i in range(HIDDEN // 2):
                var v = Int8((row + i) % 5 - 2)
                wb[row * HIDDEN + 2 * i] = v
                wb[row * HIDDEN + 2 * i + 1] = -v
        var wscb = view.bind(wsc_ptr)[r]
        var csb = view.bind(cs_ptr)[r]
        for j in range(EXPERTS * GATE_UP):
            wscb[j] = W_SCALE
            csb[j] = Float32(0)

        var eob = view.bind(eoff_ptr)[r]
        eob[0] = Int32(0)
        eob[1] = Int32(N_TOK)
        var rb = view.bind(routes_ptr)[r]
        for t in range(N_TOK):
            rb[t] = SparseRoute(Int32(t), Float32(1.0))
        _ = arenas[r].prefault(0, arenas[r].used())

    var prof = Profiler[False]()
    var act = ButterquantActivation(view.bind(x_i8_ptr), view.bind(x_sa_ptr))
    var weight = ButterquantWeight[Recipe](
        view.bind(w_ptr), view.bind(wsc_ptr), view.bind(cs_ptr))
    var eoff = view.bind(eoff_ptr)
    var routes = view.bind(routes_ptr)

    dispatch_bq_phase1_gate_up_act[
        hidden=HIDDEN, gate_up=GATE_UP, inter=INTER,
        activation="gate", alpha=Float32(0), limit=Float32(0),
    ](act, eoff, routes, weight, view.bind(gate_ptr), EXPERTS, pools, prof)
    dispatch_bq_phase1_gate_up_act[
        hidden=HIDDEN, gate_up=GATE_UP, inter=INTER,
        activation="up", alpha=Float32(0), limit=Float32(0),
    ](act, eoff, routes, weight, view.bind(up_ptr), EXPERTS, pools, prof)
    dispatch_bq_phase1_gate_up_act[
        hidden=HIDDEN, gate_up=GATE_UP, inter=INTER,
        activation="swiglu_oai", alpha=M3_SWIGLU_ALPHA, limit=M3_SWIGLU_LIMIT,
    ](act, eoff, routes, weight, view.bind(swi_ptr), EXPERTS, pools, prof)
    dispatch_bq_phase1_gate_up_act[
        hidden=HIDDEN, gate_up=GATE_UP, inter=INTER,
        activation="gelu", alpha=Float32(0), limit=Float32(0),
    ](act, eoff, routes, weight, view.bind(gelu_ptr), EXPERTS, pools, prof)

    var gate = view.bind(gate_ptr)[0]
    var up = view.bind(up_ptr)[0]
    var swi = view.bind(swi_ptr)[0]
    var gelu = view.bind(gelu_ptr)[0]

    var worst_swi = Float32(0)
    var worst_gelu = Float32(0)
    var span_lo = Float32(1e30)
    var span_hi = Float32(-1e30)
    var ok = True
    for i in range(N_TOK * INTER):
        var gv = gate[i].cast[DType.float32]()
        var uv = up[i].cast[DType.float32]()
        if gv < span_lo:
            span_lo = gv
        if gv > span_hi:
            span_hi = gv

        var swi_want = golden_swiglu(gv, uv)
        var swi_got = swi[i].cast[DType.float32]()
        var ds = abs(swi_want - swi_got)
        if ds > worst_swi:
            worst_swi = ds
        if ds > Float32(2e-2) + Float32(2e-2) * abs(swi_want):
            ok = False

        var gelu_want = golden_gelu(gv, uv)
        var gelu_got = gelu[i].cast[DType.float32]()
        var dg = abs(gelu_want - gelu_got)
        if dg > worst_gelu:
            worst_gelu = dg
        if dg > Float32(2e-2) + Float32(2e-2) * abs(gelu_want):
            ok = False

    print(t"  gate value span: [{span_lo}, {span_hi}]")
    print(t"  worst_swiglu_diff={worst_swi} worst_gelu_diff={worst_gelu}")
    if ok:
        print(t"smoke: PASS (degree={tp})")
    else:
        print(t"smoke: FAIL (degree={tp})")


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    print(t"bq moe phase1 swiglu-oai smoke: {tp} NUMA node(s)")

    comptime ARENA_BYTES = 64 * 1024 * 1024
    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_smoke[P: BurstThreadPool, //](var selected_pools: List[P]):
        run_phase1(selected_pools^, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_smoke,
    ](topo, "mode: isolated", "mode: spin-backoff")
