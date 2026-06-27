from std.collections import InlineArray
from std.memory import Span, UnsafePointer

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from simd_math.ops import sqrt
from kernels.helpers import Binding, RankView
from kernels.profiling import Profiler

from prototypes.sigmoid_router import (
    dispatch_minimax_m3_router,
    M3RouterCandidate, insert_m3_candidate, sigmoid_f32,
    M3_HIDDEN, M3_NUM_EXPERTS, M3_TOP_K, M3_ROUTED_SCALING,
)
from prototypes.sigmoid_router_rawnorm import (
    dispatch_minimax_m3_router_invrms, M3_SQRT_N, M3_RMS_EPS,
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
def x_val(t: Int, j: Int) -> BFloat16:
    return BFloat16(Float32((t * 3 + j) % 17 - 8) * 0.05)


@always_inline
def scale_val(j: Int) -> BFloat16:
    # post_attn_norm weight (j-pattern) plus the +1.0 NORM_GAIN_OFFSET.
    return BFloat16(Float32((j % 11) - 5) * 0.04 + 1.0)


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

    var x_raw_ptr = arena_alloc_all[BFloat16](arenas, SEQ_LEN * HIDDEN)
    var x_norm_ptr = arena_alloc_all[BFloat16](arenas, SEQ_LEN * HIDDEN)
    var scale_ptr = arena_alloc_all[BFloat16](arenas, HIDDEN)
    var gate_ptr = arena_alloc_all[Float32](arenas, epr * HIDDEN)
    var gate_folded_ptr = arena_alloc_all[Float32](arenas, epr * HIDDEN)
    var bias_ptr = arena_alloc_all[Float32](arenas, NUM_EXPERTS)
    var cands_ref_ptr = arena_alloc_all[M3RouterCandidate](
        arenas, MAX_WORKERS * SEQ_LEN * TOP_K)
    var cands_the_ptr = arena_alloc_all[M3RouterCandidate](
        arenas, MAX_WORKERS * SEQ_LEN * TOP_K)
    var ridx_ref_ptr = arena_alloc_all[Int32](arenas, SEQ_LEN * TOP_K)
    var rw_ref_ptr = arena_alloc_all[Float32](arenas, SEQ_LEN * TOP_K)
    var ridx_the_ptr = arena_alloc_all[Int32](arenas, SEQ_LEN * TOP_K)
    var rw_the_ptr = arena_alloc_all[Float32](arenas, SEQ_LEN * TOP_K)

    comptime sqrt_n = M3_SQRT_N
    comptime n_eps = M3_RMS_EPS

    for r in range(tp):
        var xrb = view.bind(x_raw_ptr)[r]
        var xnb = view.bind(x_norm_ptr)[r]
        var scb = view.bind(scale_ptr)[r]
        for j in range(HIDDEN):
            scb[j] = scale_val(j)
        for t in range(SEQ_LEN):
            for j in range(HIDDEN):
                xrb[t * HIDDEN + j] = x_val(t, j)
            # Reference pre-normed bf16 input: RMSNorm(x_raw, scale) in f32 -> bf16.
            var sum_sq = Float32(0)
            for j in range(HIDDEN):
                var v = Float32(xrb[t * HIDDEN + j])
                sum_sq += v * v
            var inv_rms = sqrt_n / sqrt[DType.float32, 1](sum_sq + n_eps)
            for j in range(HIDDEN):
                var v = Float32(xrb[t * HIDDEN + j])
                var s = Float32(scb[j])
                xnb[t * HIDDEN + j] = BFloat16(v * s * inv_rms)
        var gb = view.bind(gate_ptr)[r]
        var gfb = view.bind(gate_folded_ptr)[r]
        for e in range(epr):
            var ge = r * epr + e
            for j in range(HIDDEN):
                gb[e * HIDDEN + j] = gate_val(ge, j)
                # GainFold quantizer bake: fold full bf16 gamma (post_attn_norm)
                # into the router_gate column-wise, once. gfb = gate * gamma.
                gfb[e * HIDDEN + j] = gate_val(ge, j) * Float32(scb[j])
        var bb = view.bind(bias_ptr)[r]
        for e in range(NUM_EXPERTS):
            bb[e] = bias_val(e)
        var rib1 = view.bind(ridx_ref_ptr)[r]
        var rwb1 = view.bind(rw_ref_ptr)[r]
        var rib2 = view.bind(ridx_the_ptr)[r]
        var rwb2 = view.bind(rw_the_ptr)[r]
        for i in range(SEQ_LEN * TOP_K):
            rib1[i] = Int32(-2)
            rwb1[i] = Float32(-1)
            rib2[i] = Int32(-3)
            rwb2[i] = Float32(-1)
        _ = arenas[r].prefault(0, arenas[r].used())

    var prof = Profiler[False]()
    var x_raw = view.bind(x_raw_ptr)
    var x_norm = view.bind(x_norm_ptr)
    var scale = view.bind(scale_ptr)
    var gate = view.bind(gate_ptr)
    var gate_folded = view.bind(gate_folded_ptr)
    var bias = view.bind(bias_ptr)
    var cands_ref = view.bind(cands_ref_ptr)
    var cands_the = view.bind(cands_the_ptr)
    var ridx_ref = view.bind(ridx_ref_ptr)
    var rw_ref = view.bind(rw_ref_ptr)
    var ridx_the = view.bind(ridx_the_ptr)
    var rw_the = view.bind(rw_the_ptr)

    # Reference path: the existing kernel on a pre-normed bf16 input.
    dispatch_minimax_m3_router(
        x_norm, gate, bias, cands_ref, ridx_ref, rw_ref, epr, SEQ_LEN,
        pools, prof)

    # Thesis path: the buffer-free kernel on the raw residual. Gamma is folded
    # into the gate (gate_folded), inv_rms is a per-token scalar -- no scratch.
    dispatch_minimax_m3_router_invrms[sqrt_n=sqrt_n, n_eps=n_eps](
        x_raw, gate_folded, bias, cands_the, ridx_the, rw_the,
        epr, SEQ_LEN, pools, prof)

    var xr0 = x_raw[0]
    var sc0 = scale[0]
    var rir = ridx_ref[0]
    var rwr = rw_ref[0]
    var rit = ridx_the[0]
    var rwt = rw_the[0]

    var the_sel_ok = True
    var worst_w_the = Float32(0)
    var worst_w_ref = Float32(0)
    var ref_sel_misses = 0
    var the_sel_misses = 0
    var the_fixes_ref = 0

    for t in range(SEQ_LEN):
        # f64 golden: normalize + dot in f64, sigmoid via the shared f32 path.
        var sum_sq = Float64(0)
        for j in range(HIDDEN):
            var v = Float64(xr0[t * HIDDEN + j].cast[DType.float64]())
            sum_sq += v * v
        var inv_rms = Float64(sqrt_n.cast[DType.float64]()) / sqrt[
            DType.float64, 1](sum_sq + Float64(n_eps.cast[DType.float64]()))

        var gold = InlineArray[M3RouterCandidate, TOP_K](
            fill=M3RouterCandidate(Int32(0), Float32(-1.0e30), Float32(0)))
        for ge in range(NUM_EXPERTS):
            var src_rank = ge // epr
            var local = ge % epr
            var gate_row = gate[src_rank] + local * HIDDEN
            var dot = Float64(0)
            for j in range(HIDDEN):
                var v = Float64(xr0[t * HIDDEN + j].cast[DType.float64]())
                var s = Float64(sc0[j].cast[DType.float64]())
                var g = Float64(gate_row[j].cast[DType.float64]())
                dot += (v * s * inv_rms) * g
            var weight = sigmoid_f32(Float32(dot.cast[DType.float32]()))
            insert_m3_candidate[TOP_K](
                Int32(ge), weight + bias_val(ge), weight, gold)

        var sum_w = Float32(0)
        for k in range(TOP_K):
            sum_w += gold[k].weight
        var inv = Float32(1.0) / sum_w

        for k in range(TOP_K):
            var want_e = gold[k].expert
            var want_w = gold[k].weight * inv * M3_ROUTED_SCALING

            var got_e_the = rit[t * TOP_K + k]
            var got_w_the = rwt[t * TOP_K + k]
            var got_e_ref = rir[t * TOP_K + k]
            var got_w_ref = rwr[t * TOP_K + k]

            if got_e_the != want_e:
                the_sel_ok = False
                the_sel_misses += 1
            if got_e_ref != want_e:
                ref_sel_misses += 1
                # thesis matches golden where the bf16-prenorm path does not.
                if got_e_the == want_e:
                    the_fixes_ref += 1

            var dthe = abs(want_w - got_w_the)
            var dref = abs(want_w - got_w_ref)
            if dthe > worst_w_the:
                worst_w_the = dthe
            if dref > worst_w_ref:
                worst_w_ref = dref

    print(t"  thesis(folded-gate,inv_rms) vs golden: sel_misses={the_sel_misses}"
          t" worst_w={worst_w_the}")
    print(t"  reference(bf16 prenorm) vs golden: sel_misses={ref_sel_misses}"
          t" worst_w={worst_w_ref}")
    print(t"  thesis corrects {the_fixes_ref} routing decisions the bf16 path"
          t" gets wrong")
    # The thesis is sound iff it reproduces the f64 golden routing, and is at
    # least as accurate as the bf16-prenorm buffer approach it replaces.
    var the_at_least_as_good = worst_w_the <= worst_w_ref
    var ok = the_sel_ok and the_at_least_as_good
    if ok:
        print(t"smoke: PASS (degree={tp}, thesis==golden, thesis>=bf16 accuracy)")
    else:
        print(t"smoke: FAIL (degree={tp}, the_sel_ok={the_sel_ok}"
              t" the<=ref={the_at_least_as_good})")


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    print(t"m3 router internal-norm smoke: {tp} NUMA node(s), seq_len={SEQ_LEN}")

    comptime ARENA_BYTES = 128 * 1024 * 1024
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
