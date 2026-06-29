from std.collections import InlineArray
from std.memory import Span, UnsafePointer

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from simd_math.ops import sqrt
from kernels.attention_ops import KVRunTable
from kernels.helpers import Binding, RankView
from kernels.gemm import dispatch_gemm
from kernels.rmsnorm import dispatch_rms_norm
from kernels.profiling import Profiler

from prototypes.lightning_indexer import (
    dispatch_minimax_m3_indexer,
    M3_INDEX_HEAD_DIM, M3_INDEX_NUM_HEADS, M3_INDEX_BLOCK,
    M3_INDEX_TOPK_BLOCKS,
)


comptime ALIGNMENT = 64
comptime SEQ_LEN = 2560
comptime HIDDEN = 6144
comptime IHD = M3_INDEX_HEAD_DIM
comptime INH = M3_INDEX_NUM_HEADS
comptime Q_DIM = INH * IHD
comptime K_DIM = IHD
comptime PAGE_LEN = M3_INDEX_BLOCK
comptime NUM_KEY_ROWS = ((SEQ_LEN + PAGE_LEN - 1) // PAGE_LEN) * PAGE_LEN
comptime MAX_BLOCK = (SEQ_LEN - 1) // M3_INDEX_BLOCK + 1
comptime BLOCK_STRIDE = (MAX_BLOCK + 15) // 16 * 16
comptime RMS_EPS = Float32(1e-6)

# Three input scalings, all encoding the SAME f64-exact head-normed query/key
# set (per-head RMSNorm is scale-invariant, so any positive per-token factor
# cancels). They differ only in where bf16 rounds the pre-projection input:
#   A  : factor = inv_rms  (a true RMSNorm input)
#   B0 : factor = 1.0      (gain only -- the thesis: drop inv_rms)
#   B1 : factor = 1.40625  (gain only at a different non-pow2 scale)
# B0-vs-B1 is the inherent bf16 block-selection noise floor; if A-vs-B0 sits at
# that floor, applying vs dropping inv_rms is indistinguishable noise.
comptime B1_SCALE = Float32(1.40625)


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
    var amp = Float32(0.02) + Float32(0.0015) * Float32(t % 7)
    return BFloat16(Float32((t * 7 + j * 3) % 31 - 15) * amp)


@always_inline
def gain_val(j: Int) -> BFloat16:
    # input_layernorm weight (j-pattern) plus the +1.0 NORM_GAIN_OFFSET.
    return BFloat16(Float32((j % 13) - 6) * 0.03 + 1.0)


@always_inline
def wq_val(oc: Int, j: Int) -> BFloat16:
    return BFloat16(Float32(((oc * 5 + j * 11) % 19) - 9) * 0.02)


@always_inline
def wk_val(oc: Int, j: Int) -> BFloat16:
    return BFloat16(Float32(((oc * 17 + j * 3) % 23) - 11) * 0.02)


@always_inline
def fill_x_inp[o: ImmutOrigin](
    use_inv_rms: Bool,
    const_scale: Float32,
    x_raw: Binding[BFloat16, o],
    gain: Binding[BFloat16, o],
    x_inp: Binding[BFloat16, o],
    rank: Int,
):
    comptime sqrt_h = sqrt[DType.float32, 1](HIDDEN)
    comptime h_eps = Float32(HIDDEN) * RMS_EPS
    var xr = x_raw[rank]
    var gn = gain[rank]
    var xi = x_inp[rank]
    for t in range(SEQ_LEN):
        var factor = const_scale
        if use_inv_rms:
            var sum_sq = Float32(0)
            for j in range(HIDDEN):
                var v = Float32(xr[t * HIDDEN + j])
                sum_sq += v * v
            factor = sqrt_h / sqrt[DType.float32, 1](sum_sq + h_eps)
        for j in range(HIDDEN):
            var v = Float32(xr[t * HIDDEN + j]) * Float32(gn[j]) * factor
            xi[t * HIDDEN + j] = BFloat16(v)


@always_inline
def scatter_cache[o: ImmutOrigin](
    k_proj: Binding[BFloat16, o],
    cache: Binding[BFloat16, o],
    tp: Int,
    rank: Int,
):
    var kp = k_proj[rank]
    var ch = cache[rank]
    for lr in range(NUM_KEY_ROWS):
        var g = lr * tp + rank
        if g < SEQ_LEN:
            for d in range(K_DIM):
                ch[lr * K_DIM + d] = kp[g * K_DIM + d]
        else:
            for d in range(K_DIM):
                ch[lr * K_DIM + d] = BFloat16(0)


def build_path[P: BurstThreadPool, o: ImmutOrigin, //](
    use_inv_rms: Bool,
    const_scale: Float32,
    x_raw: Binding[BFloat16, o],
    gain: Binding[BFloat16, o],
    x_inp: Binding[BFloat16, o],
    wq: Binding[BFloat16, o],
    wk: Binding[BFloat16, o],
    qn_gain: Binding[BFloat16, o],
    kn_gain: Binding[BFloat16, o],
    index_q: Binding[BFloat16, o],
    k_proj: Binding[BFloat16, o],
    cache: Binding[BFloat16, o],
    tp: Int,
    mut pools: List[P],
    mut prof: Profiler[False],
):
    comptime sqrt_ihd = sqrt[DType.float32, 1](IHD)
    comptime ihd_eps = Float32(IHD) * RMS_EPS

    # Apply the input gain and the per-token factor in f32, store bf16 -- one
    # host pass per rank (replicated activation), then the real kernels once.
    for r in range(tp):
        fill_x_inp(use_inv_rms, const_scale, x_raw, gain, x_inp, r)

    dispatch_gemm[cols=HIDDEN](x_inp, wq, index_q, Q_DIM, SEQ_LEN, pools, prof)
    dispatch_gemm[cols=HIDDEN](x_inp, wk, k_proj, K_DIM, SEQ_LEN, pools, prof)

    # Per-head RMSNorm after projection (the scale-invariant step that, in exact
    # arithmetic, cancels the input factor).
    dispatch_rms_norm[hidden=IHD, sqrt_n=sqrt_ihd, n_eps=ihd_eps](
        index_q, index_q, qn_gain, SEQ_LEN * INH, pools, prof)
    dispatch_rms_norm[hidden=IHD, sqrt_n=sqrt_ihd, n_eps=ihd_eps](
        k_proj, k_proj, kn_gain, SEQ_LEN, pools, prof)

    for r in range(tp):
        scatter_cache(k_proj, cache, tp, r)


def build_path_wfold[P: BurstThreadPool, o: ImmutOrigin, //](
    x_raw: Binding[BFloat16, o],
    gain: Binding[BFloat16, o],
    x_inp: Binding[BFloat16, o],
    wq: Binding[BFloat16, o],
    wk: Binding[BFloat16, o],
    wqf: Binding[BFloat16, o],
    wkf: Binding[BFloat16, o],
    qn_gain: Binding[BFloat16, o],
    kn_gain: Binding[BFloat16, o],
    index_q: Binding[BFloat16, o],
    k_proj: Binding[BFloat16, o],
    cache: Binding[BFloat16, o],
    tp: Int,
    mut pools: List[P],
    mut prof: Profiler[False],
):
    comptime sqrt_ihd = sqrt[DType.float32, 1](IHD)
    comptime ihd_eps = Float32(IHD) * RMS_EPS

    # The GainFold quantizer bake: fold the input gain column-wise into the
    # projection weights (wqf = wq * gain), then project the RAW residual -- no
    # input gain, no inv_rms. Mirrors the GainFold recipe in the wired bq path.
    for r in range(tp):
        var gn = gain[r]
        var wqr = wq[r]
        var wqfr = wqf[r]
        for oc in range(Q_DIM):
            for j in range(HIDDEN):
                wqfr[oc * HIDDEN + j] = BFloat16(
                    Float32(wqr[oc * HIDDEN + j]) * Float32(gn[j]))
        var wkr = wk[r]
        var wkfr = wkf[r]
        for oc in range(K_DIM):
            for j in range(HIDDEN):
                wkfr[oc * HIDDEN + j] = BFloat16(
                    Float32(wkr[oc * HIDDEN + j]) * Float32(gn[j]))
        var xr = x_raw[r]
        var xi = x_inp[r]
        for t in range(SEQ_LEN):
            for j in range(HIDDEN):
                xi[t * HIDDEN + j] = xr[t * HIDDEN + j]

    dispatch_gemm[cols=HIDDEN](x_inp, wqf, index_q, Q_DIM, SEQ_LEN, pools, prof)
    dispatch_gemm[cols=HIDDEN](x_inp, wkf, k_proj, K_DIM, SEQ_LEN, pools, prof)

    dispatch_rms_norm[hidden=IHD, sqrt_n=sqrt_ihd, n_eps=ihd_eps](
        index_q, index_q, qn_gain, SEQ_LEN * INH, pools, prof)
    dispatch_rms_norm[hidden=IHD, sqrt_n=sqrt_ihd, n_eps=ihd_eps](
        k_proj, k_proj, kn_gain, SEQ_LEN, pools, prof)

    for r in range(tp):
        scatter_cache(k_proj, cache, tp, r)


@always_inline
def mismatch_count[o: ImmutOrigin](
    a: Binding[Int32, o], b: Binding[Int32, o], n: Int,
) -> Int:
    var pa = a[0]
    var pb = b[0]
    var c = 0
    for i in range(n):
        if pa[i] != pb[i]:
            c += 1
    return c


def run_indexer[P: BurstThreadPool, //](
    var pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    var rows_per_page = PAGE_LEN // tp
    if PAGE_LEN % tp != 0 or rows_per_page < 1 or (
        rows_per_page & (rows_per_page - 1)) != 0:
        print(t"  degree={tp} cannot pow2-shard PAGE_LEN={PAGE_LEN}; skipping")
        return
    print(t"  distributing over degree={tp}, seq_len={SEQ_LEN},"
          t" blocks={MAX_BLOCK}, topk={M3_INDEX_TOPK_BLOCKS}")
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var x_raw_p = arena_alloc_all[BFloat16](arenas, SEQ_LEN * HIDDEN)
    var gain_p = arena_alloc_all[BFloat16](arenas, HIDDEN)
    var wq_p = arena_alloc_all[BFloat16](arenas, Q_DIM * HIDDEN)
    var wk_p = arena_alloc_all[BFloat16](arenas, K_DIM * HIDDEN)
    var qn_p = arena_alloc_all[BFloat16](arenas, IHD)
    var kn_p = arena_alloc_all[BFloat16](arenas, IHD)
    var xinp_p = arena_alloc_all[BFloat16](arenas, SEQ_LEN * HIDDEN)
    var kproj_p = arena_alloc_all[BFloat16](arenas, SEQ_LEN * K_DIM)

    var iq_a_p = arena_alloc_all[BFloat16](arenas, SEQ_LEN * Q_DIM)
    var iq_b0_p = arena_alloc_all[BFloat16](arenas, SEQ_LEN * Q_DIM)
    var cache_a_p = arena_alloc_all[BFloat16](arenas, NUM_KEY_ROWS * K_DIM)
    var cache_b0_p = arena_alloc_all[BFloat16](arenas, NUM_KEY_ROWS * K_DIM)
    var cache_b1_p = arena_alloc_all[BFloat16](arenas, NUM_KEY_ROWS * K_DIM)
    var iq_b1_p = arena_alloc_all[BFloat16](arenas, SEQ_LEN * Q_DIM)
    var wqf_p = arena_alloc_all[BFloat16](arenas, Q_DIM * HIDDEN)
    var wkf_p = arena_alloc_all[BFloat16](arenas, K_DIM * HIDDEN)
    var iq_wf_p = arena_alloc_all[BFloat16](arenas, SEQ_LEN * Q_DIM)
    var cache_wf_p = arena_alloc_all[BFloat16](arenas, NUM_KEY_ROWS * K_DIM)
    var bi_a_p = arena_alloc_all[Int32](arenas, SEQ_LEN * INH * M3_INDEX_TOPK_BLOCKS)
    var bi_b0_p = arena_alloc_all[Int32](arenas, SEQ_LEN * INH * M3_INDEX_TOPK_BLOCKS)
    var bi_b1_p = arena_alloc_all[Int32](arenas, SEQ_LEN * INH * M3_INDEX_TOPK_BLOCKS)
    var bi_wf_p = arena_alloc_all[Int32](arenas, SEQ_LEN * INH * M3_INDEX_TOPK_BLOCKS)
    var part_p = arena_alloc_all[Float32](arenas, SEQ_LEN * INH * BLOCK_STRIDE)

    for r in range(tp):
        var xr = view.bind(x_raw_p)[r]
        for t in range(SEQ_LEN):
            for j in range(HIDDEN):
                xr[t * HIDDEN + j] = x_val(t, j)
        var gn = view.bind(gain_p)[r]
        for j in range(HIDDEN):
            gn[j] = gain_val(j)
        var wq = view.bind(wq_p)[r]
        for oc in range(Q_DIM):
            for j in range(HIDDEN):
                wq[oc * HIDDEN + j] = wq_val(oc, j)
        var wk = view.bind(wk_p)[r]
        for oc in range(K_DIM):
            for j in range(HIDDEN):
                wk[oc * HIDDEN + j] = wk_val(oc, j)
        var qn = view.bind(qn_p)[r]
        var kn = view.bind(kn_p)[r]
        for d in range(IHD):
            qn[d] = BFloat16(1.0)
            kn[d] = BFloat16(1.0)
        _ = arenas[r].prefault(0, arenas[r].used())

    var prof = Profiler[False]()
    var x_raw = view.bind(x_raw_p)
    var gain = view.bind(gain_p)
    var wq = view.bind(wq_p)
    var wk = view.bind(wk_p)
    var qn = view.bind(qn_p)
    var kn = view.bind(kn_p)

    # Build the three input encodings. xinp/kproj are scratch reused per path;
    # each path's head-normed index_q and sharded key cache are kept separate.
    build_path(
        True, Float32(0.0), x_raw, gain, view.bind(xinp_p), wq, wk, qn, kn,
        view.bind(iq_a_p), view.bind(kproj_p), view.bind(cache_a_p),
        tp, pools, prof)
    build_path(
        False, Float32(1.0), x_raw, gain, view.bind(xinp_p), wq, wk, qn, kn,
        view.bind(iq_b0_p), view.bind(kproj_p), view.bind(cache_b0_p),
        tp, pools, prof)
    build_path(
        False, B1_SCALE, x_raw, gain, view.bind(xinp_p), wq, wk, qn, kn,
        view.bind(iq_b1_p), view.bind(kproj_p), view.bind(cache_b1_p),
        tp, pools, prof)
    # Weight-fold path (the wired bq design): gain folded into the projection
    # weights, raw residual projected, inv_rms dropped.
    build_path_wfold(
        x_raw, gain, view.bind(xinp_p), wq, wk, view.bind(wqf_p),
        view.bind(wkf_p), qn, kn, view.bind(iq_wf_p), view.bind(kproj_p),
        view.bind(cache_wf_p), tp, pools, prof)

    # Run table for the index-K cache, built immediately before use and kept
    # alive past the dispatches (a raw MutUntracked pointer does not keep the
    # local KVRunTable alive under ASAP destruction).
    var runs_table = KVRunTable()
    runs_table.begin_run(0, 0)
    var num_local_rows = (SEQ_LEN + tp - 1) // tp
    var num_pages = (num_local_rows + rows_per_page - 1) // rows_per_page
    for g in range(num_pages):
        runs_table.add_base_row(Int32(g * rows_per_page))
    var runs = UnsafePointer(to=runs_table).as_unsafe_any_origin()

    dispatch_minimax_m3_indexer[page_len=PAGE_LEN](
        view.bind(iq_a_p), view.bind(cache_a_p), view.bind(bi_a_p),
        view.bind(part_p), runs, SEQ_LEN, pools, prof)
    dispatch_minimax_m3_indexer[page_len=PAGE_LEN](
        view.bind(iq_b0_p), view.bind(cache_b0_p), view.bind(bi_b0_p),
        view.bind(part_p), runs, SEQ_LEN, pools, prof)
    dispatch_minimax_m3_indexer[page_len=PAGE_LEN](
        view.bind(iq_b1_p), view.bind(cache_b1_p), view.bind(bi_b1_p),
        view.bind(part_p), runs, SEQ_LEN, pools, prof)
    dispatch_minimax_m3_indexer[page_len=PAGE_LEN](
        view.bind(iq_wf_p), view.bind(cache_wf_p), view.bind(bi_wf_p),
        view.bind(part_p), runs, SEQ_LEN, pools, prof)
    _ = runs_table^

    var total = SEQ_LEN * INH * M3_INDEX_TOPK_BLOCKS
    var floor = mismatch_count(view.bind(bi_b0_p), view.bind(bi_b1_p), total)
    var ab0 = mismatch_count(view.bind(bi_a_p), view.bind(bi_b0_p), total)
    var ab1 = mismatch_count(view.bind(bi_a_p), view.bind(bi_b1_p), total)
    var awf = mismatch_count(view.bind(bi_a_p), view.bind(bi_wf_p), total)
    var wfb0 = mismatch_count(view.bind(bi_wf_p), view.bind(bi_b0_p), total)

    var iqa = view.bind(iq_a_p)[0]
    var iqb = view.bind(iq_b0_p)[0]
    var worst_q = Float32(0)
    for i in range(SEQ_LEN * Q_DIM):
        var d = abs(Float32(iqa[i]) - Float32(iqb[i]))
        if d > worst_q:
            worst_q = d

    print(t"  bf16 noise floor  B0-vs-B1 (same input regime): {floor}/{total}")
    print(t"  thesis A(rmsnorm)-vs-B0(gain only):             {ab0}/{total}")
    print(t"  cross  A(rmsnorm)-vs-B1(gain only, alt scale):  {ab1}/{total}")
    print(t"  wfold  A(rmsnorm)-vs-WF(GainFold weights):      {awf}/{total}")
    print(t"  wfold  WF-vs-B0 (input-gain vs weight-gain):    {wfb0}/{total}")
    print(t"  worst head-normed |iq_A - iq_B0| = {worst_q}")
    # Sound iff dropping inv_rms does not perturb block selection beyond the
    # inherent bf16 rounding noise floor (A sits within ~2x of B0-vs-B1). The
    # GainFold weight-fold path (WF) must sit in the same floor as the input-side
    # gain (B0), confirming folding gamma into the weights is equivalent.
    var bound = 2 * floor + total // 1000
    var ok = ab0 <= bound and ab1 <= bound and awf <= bound and wfb0 <= bound
    if ok:
        print(t"smoke: PASS (degree={tp}, gain-only (input and weight-fold)"
              t" within bf16 noise floor of rmsnorm: inv_rms selection-irrelevant)")
    else:
        print(t"smoke: FAIL (degree={tp}, ab0={ab0} awf={awf} wfb0={wfb0}"
              t" bound={bound})")


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    print(t"m3 indexer inv_rms-cancellation smoke: {tp} NUMA node(s)")

    comptime ARENA_BYTES = 384 * 1024 * 1024
    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_smoke[P: BurstThreadPool, //](var selected_pools: List[P]):
        run_indexer(selected_pools^, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_smoke,
    ](topo, "mode: isolated", "mode: spin-backoff")
