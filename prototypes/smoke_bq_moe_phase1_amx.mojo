from std.memory import Span, UnsafePointer

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.helpers import Binding, RankView
from kernels.moe_router import SparseRoute
from kernels.profiling import Profiler
from kernels.elementwise import swiglu_oai_activate

from butterquant.weight import ButterquantWeight, ButterquantActivation
from butterquant.amx_tiles import prime_amx_environment
from quant.recipe import (
    QuantRecipe, PerRowQuant, NoGamma, SingleSided, PerRowCs, VnniPacked,
)

from prototypes.bq_moe_phase1 import (
    dispatch_bq_phase1_gate_up_act, dispatch_bq_phase1_gate_up_act_amx,
    M3_SWIGLU_ALPHA, M3_SWIGLU_LIMIT,
)


comptime ALIGNMENT = 64
comptime HIDDEN = 256
comptime INTER = 128
comptime GATE_UP = 2 * INTER
comptime EXPERTS = 2
comptime TOKENS = 50
comptime RECORDS = 64
comptime EXPERT0_HI = 40
comptime X_SA = Float32(2.0)
comptime W_SCALE = Float32(0.05)
comptime W_CONST = 2

comptime Recipe: QuantRecipe = PerRowQuant(
    128, NoGamma(), SingleSided(), PerRowCs(), VnniPacked())

comptime I8Ptr = UnsafePointer[Int8, MutUntrackedOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutUntrackedOrigin]


def arena_bases(
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
) -> List[Int]:
    var bases = List[Int](capacity=len(arenas))
    for r in range(len(arenas)):
        bases.append(Int(arenas[r].base.value()))
    return bases^


def arena_alloc_all[T: AnyType](
    mut arenas: List[NumaArena[alignment=ALIGNMENT]], count: Int,
) -> UnsafePointer[T, MutUntrackedOrigin]:
    var first = UnsafePointer[T, MutUntrackedOrigin].unsafe_dangling()
    for r in range(len(arenas)):
        var p = arenas[r].alloc[T](count)
        if not p:
            print("arena alloc failed for", count, "elements")
            return UnsafePointer[T, MutUntrackedOrigin].unsafe_dangling()
        if r == 0:
            first = p.value()
    return first


@always_inline
def golden_swiglu(g: Float32, u: Float32) -> Float32:
    return swiglu_oai_activate[1, M3_SWIGLU_ALPHA, M3_SWIGLU_LIMIT](
        SIMD[DType.float32, 1](g), SIMD[DType.float32, 1](u))[0]


def token_of(rec: Int) -> Int:
    return (rec * 37) % TOKENS


def wsc_of(n: Int) -> Float32:
    return W_SCALE * (Float32(1.0) + Float32(0.013) * Float32(n % 13))


def x_of(t: Int, c: Int) -> Int:
    return (t * 3 + c) % 7


def run_phase1[P: BurstThreadPool, //](
    var pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    print(t"  degree={tp}")
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var x_i8_ptr = arena_alloc_all[Int8](arenas, TOKENS * HIDDEN)
    var x_sa_ptr = arena_alloc_all[Float32](arenas, TOKENS)
    var w_ptr = arena_alloc_all[Int8](arenas, EXPERTS * GATE_UP * HIDDEN)
    var wsc_ptr = arena_alloc_all[Float32](arenas, EXPERTS * GATE_UP)
    var cs_ptr = arena_alloc_all[Float32](arenas, EXPERTS * GATE_UP)
    var eoff_ptr = arena_alloc_all[Int32](arenas, EXPERTS + 1)
    var routes_ptr = arena_alloc_all[SparseRoute](arenas, RECORDS)
    var act_routed_ptr = arena_alloc_all[Int8](arenas, RECORDS * HIDDEN)
    var sa_routed_ptr = arena_alloc_all[Float32](arenas, RECORDS)
    var amx_gate_ptr = arena_alloc_all[BFloat16](arenas, RECORDS * INTER)
    var amx_up_ptr = arena_alloc_all[BFloat16](arenas, RECORDS * INTER)
    var amx_swi_ptr = arena_alloc_all[BFloat16](arenas, RECORDS * INTER)
    var vnni_gate_ptr = arena_alloc_all[BFloat16](arenas, RECORDS * INTER)
    var vnni_up_ptr = arena_alloc_all[BFloat16](arenas, RECORDS * INTER)

    for r in range(tp):
        var xb = view.bind(x_i8_ptr)[r]
        for t in range(TOKENS):
            for c in range(HIDDEN):
                xb[t * HIDDEN + c] = Int8(x_of(t, c))
        var sab = view.bind(x_sa_ptr)[r]
        for t in range(TOKENS):
            sab[t] = X_SA

        # Constant weight is layout-invariant: it reads as W_CONST under both the
        # VNNI4 pack walk and the AMX tile walk, so the dot is W_CONST*rowsum(act)
        # with a closed-form f32 golden -- no packer needed. Per-column wsc still
        # exercises column addressing and the gate/up tile pairing. colsum=0 with
        # non-negative activations keeps signed==unsigned exact.
        var wb = view.bind(w_ptr)[r]
        for i in range(EXPERTS * GATE_UP * HIDDEN):
            wb[i] = Int8(W_CONST)
        var wscb = view.bind(wsc_ptr)[r]
        var csb = view.bind(cs_ptr)[r]
        for e in range(EXPERTS):
            for n in range(GATE_UP):
                wscb[e * GATE_UP + n] = wsc_of(n)
                csb[e * GATE_UP + n] = Float32(HIDDEN * W_CONST)

        var eob = view.bind(eoff_ptr)[r]
        eob[0] = Int32(0)
        eob[1] = Int32(EXPERT0_HI)
        eob[2] = Int32(RECORDS)
        var rb = view.bind(routes_ptr)[r]
        for rec in range(RECORDS):
            rb[rec] = SparseRoute(Int32(token_of(rec)), Float32(1.0))
        _ = arenas[r].prefault(0, arenas[r].used())

    prime_amx_environment(pools)

    var prof = Profiler[False]()
    var act = ButterquantActivation(view.bind(x_i8_ptr), view.bind(x_sa_ptr))
    var weight = ButterquantWeight[Recipe](
        view.bind(w_ptr), view.bind(wsc_ptr), view.bind(cs_ptr))
    var eoff = view.bind(eoff_ptr)
    var routes = view.bind(routes_ptr)
    var act_routed = view.bind(act_routed_ptr)
    var sa_routed = view.bind(sa_routed_ptr)

    dispatch_bq_phase1_gate_up_act[
        hidden=HIDDEN, gate_up=GATE_UP, inter=INTER,
        activation="gate", alpha=Float32(0), limit=Float32(0),
    ](act, eoff, routes, weight, view.bind(vnni_gate_ptr), EXPERTS, pools, prof)
    dispatch_bq_phase1_gate_up_act[
        hidden=HIDDEN, gate_up=GATE_UP, inter=INTER,
        activation="up", alpha=Float32(0), limit=Float32(0),
    ](act, eoff, routes, weight, view.bind(vnni_up_ptr), EXPERTS, pools, prof)

    dispatch_bq_phase1_gate_up_act_amx[
        hidden=HIDDEN, gate_up=GATE_UP, inter=INTER,
        activation="gate", alpha=Float32(0), limit=Float32(0),
    ](act, eoff, routes, weight, view.bind(amx_gate_ptr),
      act_routed, sa_routed, EXPERTS, pools, prof)
    dispatch_bq_phase1_gate_up_act_amx[
        hidden=HIDDEN, gate_up=GATE_UP, inter=INTER,
        activation="up", alpha=Float32(0), limit=Float32(0),
    ](act, eoff, routes, weight, view.bind(amx_up_ptr),
      act_routed, sa_routed, EXPERTS, pools, prof)
    dispatch_bq_phase1_gate_up_act_amx[
        hidden=HIDDEN, gate_up=GATE_UP, inter=INTER,
        activation="swiglu_oai", alpha=M3_SWIGLU_ALPHA, limit=M3_SWIGLU_LIMIT,
    ](act, eoff, routes, weight, view.bind(amx_swi_ptr),
      act_routed, sa_routed, EXPERTS, pools, prof)

    var xb0 = view.bind(x_i8_ptr)[0]
    var rowsum = List[Int](capacity=TOKENS)
    for t in range(TOKENS):
        var s = 0
        for c in range(HIDDEN):
            s += Int(xb0[t * HIDDEN + c])
        rowsum.append(s)

    var ag = view.bind(amx_gate_ptr)[0]
    var au = view.bind(amx_up_ptr)[0]
    var asw = view.bind(amx_swi_ptr)[0]
    var vg = view.bind(vnni_gate_ptr)[0]

    var ad = X_SA / Float32(127)
    var worst_amx_g = Float32(0)
    var worst_amx_u = Float32(0)
    var worst_amx_s = Float32(0)
    var worst_vnni_g = Float32(0)
    var ok = True

    @parameter
    def check(got: Float32, want: Float32, mut worst: Float32):
        var d = abs(got - want)
        if d > worst:
            worst = d
        if d > Float32(2e-2) + Float32(2e-2) * abs(want):
            ok = False

    for rec in range(RECORDS):
        var raw = Float32(W_CONST) * Float32(rowsum[token_of(rec)])
        for i in range(INTER):
            var g = raw * ad * wsc_of(i)
            var u = raw * ad * wsc_of(INTER + i)
            var idx = rec * INTER + i
            check(ag[idx].cast[DType.float32](), g, worst_amx_g)
            check(au[idx].cast[DType.float32](), u, worst_amx_u)
            check(asw[idx].cast[DType.float32](), golden_swiglu(g, u),
                  worst_amx_s)
            check(vg[idx].cast[DType.float32](), g, worst_vnni_g)

    print(t"  worst amx vs golden: gate={worst_amx_g} up={worst_amx_u}"
          t" swiglu={worst_amx_s}")
    print(t"  worst vnni vs golden: gate={worst_vnni_g}")
    if ok:
        print(t"smoke: PASS (degree={tp})")
    else:
        print(t"smoke: FAIL (degree={tp})")


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    print(t"bq moe phase1 AMX smoke: {tp} NUMA node(s)")

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
