from std.collections import InlineArray
from std.math import exp
from std.memory import Span, UnsafePointer

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.attention_ops import KVRunTable
from kernels.helpers import Binding, RankView
from kernels.logsum_merge import MergeSegment
from kernels.profiling import Profiler

from butterquant.dot_products import i8_vnni_block_dot
from butterquant.types import I8Ptr

from prototypes.lightning_indexer import M3_INDEX_BLOCK, M3_INDEX_TOPK_BLOCKS
from prototypes.bq_sparse_attention import (
    dispatch_bq_minimax_m3_sparse_attention,
    M3_NUM_HEADS, M3_NUM_KV_HEADS, M3_HEAD_DIM, M3_GQA_RATIO, M3_KV_DIM,
)


comptime ALIGNMENT = 64
comptime MAX_WORKERS = 128
comptime SEQ_LEN = 2600
comptime PAGE_LEN = M3_INDEX_BLOCK
comptime Q_ROW = M3_NUM_HEADS * M3_HEAD_DIM
comptime NUM_KEY_ROWS = ((SEQ_LEN + PAGE_LEN - 1) // PAGE_LEN) * PAGE_LEN
comptime PARTIAL_STRIDE = (
    (M3_NUM_HEADS * M3_HEAD_DIM + 2 * M3_NUM_HEADS) * 4 + 63) // 64 * 16
comptime F_Q = Float32(200.0)
comptime INV127 = Float32(1.0) / Float32(127.0)
comptime INV127SQ = INV127 * INV127


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
def qi8(t: Int, qh: Int, d: Int) -> Int8:
    return Int8((t + qh + d) % 7 - 3)


@always_inline
def ki8(g: Int, kh: Int, d: Int) -> Int8:
    return Int8((g + kh + 2 * d) % 5 - 2)


@always_inline
def vi8(g: Int, kh: Int, d: Int) -> Int8:
    return Int8((g + 3 * d + kh) % 11 - 5)


@always_inline
def ksc(g: Int) -> Float32:
    return Float32(1.0) + Float32(0.01) * Float32(g % 5)


@always_inline
def vsc(g: Int) -> Float32:
    return Float32(0.5) + Float32(0.01) * Float32(g % 7)


def run_attention[P: BurstThreadPool, //](
    var pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    var rows_per_page = PAGE_LEN // tp
    if PAGE_LEN % tp != 0 or rows_per_page < 1 or (
        rows_per_page & (rows_per_page - 1)) != 0:
        print(
            t"  degree={tp} does not shard PAGE_LEN={PAGE_LEN} into a power of "
            t"two; skipping (infra requires page_len/degree to be pow2)")
        return
    var local_num_q = M3_NUM_HEADS // tp
    if M3_NUM_HEADS % tp != 0:
        print(t"  degree={tp} does not divide num_heads={M3_NUM_HEADS}; skip")
        return
    print(t"  distributing over degree={tp} (K/V round-robin sharded)")
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var q_ptr = arena_alloc_all[Int8](arenas, SEQ_LEN * Q_ROW)
    var qi_bias_ptr = arena_alloc_all[Float32](arenas, SEQ_LEN * M3_NUM_HEADS)
    var f_q_ptr = arena_alloc_all[Float32](arenas, SEQ_LEN * M3_NUM_HEADS)
    var k_ptr = arena_alloc_all[Int8](arenas, NUM_KEY_ROWS * M3_KV_DIM)
    var k_scale_ptr = arena_alloc_all[Float32](
        arenas, NUM_KEY_ROWS * M3_NUM_KV_HEADS)
    var v_ptr = arena_alloc_all[Int8](arenas, NUM_KEY_ROWS * M3_KV_DIM)
    var v_scale_ptr = arena_alloc_all[Float32](
        arenas, NUM_KEY_ROWS * M3_NUM_KV_HEADS)
    var block_idx_ptr = arena_alloc_all[Int32](
        arenas, SEQ_LEN * M3_NUM_KV_HEADS * M3_INDEX_TOPK_BLOCKS)
    var output_ptr = arena_alloc_all[BFloat16](arenas, SEQ_LEN * Q_ROW)
    var attn_partial_ptr = arena_alloc_all[Float32](
        arenas, SEQ_LEN * PARTIAL_STRIDE)
    var segment_ptr = arena_alloc_all[MergeSegment](arenas, MAX_WORKERS * tp)

    comptime bi_tstride = M3_NUM_KV_HEADS * M3_INDEX_TOPK_BLOCKS

    for r in range(tp):
        var qb = view.bind(q_ptr)[r]
        var qbias = view.bind(qi_bias_ptr)[r]
        var fqb = view.bind(f_q_ptr)[r]
        for t in range(SEQ_LEN):
            for qh in range(M3_NUM_HEADS):
                var qsum = 0
                for d in range(M3_HEAD_DIM):
                    var qv = qi8(t, qh, d)
                    qb[t * Q_ROW + qh * M3_HEAD_DIM + d] = qv
                    qsum += Int(qv)
                qbias[t * M3_NUM_HEADS + qh] = Float32(128 * qsum)
                fqb[t * M3_NUM_HEADS + qh] = F_Q

        var kb = view.bind(k_ptr)[r]
        var vb = view.bind(v_ptr)[r]
        var ksb = view.bind(k_scale_ptr)[r]
        var vsb = view.bind(v_scale_ptr)[r]
        for lr in range(NUM_KEY_ROWS):
            var g = lr * tp + r
            for kh in range(M3_NUM_KV_HEADS):
                for d in range(M3_HEAD_DIM):
                    var off = lr * M3_KV_DIM + kh * M3_HEAD_DIM + d
                    if g < SEQ_LEN:
                        kb[off] = ki8(g, kh, d)
                        vb[off] = vi8(g, kh, d)
                    else:
                        kb[off] = Int8(0)
                        vb[off] = Int8(0)
                ksb[lr * M3_NUM_KV_HEADS + kh] = ksc(g)
                vsb[lr * M3_NUM_KV_HEADS + kh] = vsc(g)

        var bib = view.bind(block_idx_ptr)[r]
        for t in range(SEQ_LEN):
            var num_blocks = t // M3_INDEX_BLOCK + 1
            for kh in range(M3_NUM_KV_HEADS):
                var row = bib + t * bi_tstride + kh * M3_INDEX_TOPK_BLOCKS
                row[0] = Int32(0)
                var s = 1
                for b in range(1, num_blocks):
                    if s >= M3_INDEX_TOPK_BLOCKS:
                        break
                    if (b % 2) == (kh % 2):
                        row[s] = Int32(b)
                        s += 1
                while s < M3_INDEX_TOPK_BLOCKS:
                    row[s] = Int32(-1)
                    s += 1

        var ob = view.bind(output_ptr)[r]
        for i in range(SEQ_LEN * Q_ROW):
            ob[i] = BFloat16(Float32(-1))
        _ = arenas[r].prefault(0, arenas[r].used())

    var runs_table = KVRunTable()
    runs_table.begin_run(0, 0)
    var num_local_rows = (SEQ_LEN + tp - 1) // tp
    var num_pages = (num_local_rows + rows_per_page - 1) // rows_per_page
    for g in range(num_pages):
        runs_table.add_base_row(Int32(g * rows_per_page))
    var runs = UnsafePointer(to=runs_table).unsafe_origin_cast[
        MutUntrackedOrigin]()

    var prof = Profiler[False]()
    var q = view.bind(q_ptr)
    var qi_bias = view.bind(qi_bias_ptr)
    var f_q = view.bind(f_q_ptr)
    var k = view.bind(k_ptr)
    var k_scale = view.bind(k_scale_ptr)
    var v = view.bind(v_ptr)
    var v_scale = view.bind(v_scale_ptr)
    var block_idx = view.bind(block_idx_ptr)
    var output = view.bind(output_ptr)
    var attn_partial = view.bind(attn_partial_ptr)
    var segments = view.bind(segment_ptr)

    dispatch_bq_minimax_m3_sparse_attention[page_len=PAGE_LEN](
        q, qi_bias, f_q, k, k_scale, v, v_scale, block_idx, output,
        attn_partial, segments, runs, SEQ_LEN, pools, prof)
    _ = runs_table

    var probe_tok = [0, 1, 200, 2048, 2560, 2599]
    var probe_head = [0, 15, 16, 33, 63]
    var out_stride = local_num_q * M3_HEAD_DIM
    var worst = Float32(0)
    var sparse_seen = False
    var ok = True
    for pti in range(len(probe_tok)):
        var t = probe_tok[pti]
        if t >= SEQ_LEN:
            continue
        var num_blocks = t // M3_INDEX_BLOCK + 1
        for phi in range(len(probe_head)):
            var qh = probe_head[phi]
            var kh = qh // M3_GQA_RATIO
            var bi = view.bind(block_idx_ptr)[0] \
                + t * bi_tstride + kh * M3_INDEX_TOPK_BLOCKS
            var q_i8base = view.bind(q_ptr)[0] + t * Q_ROW + qh * M3_HEAD_DIM
            var fq = view.bind(f_q_ptr)[0][t * M3_NUM_HEADS + qh]

            var m = Float32(-1e30)
            var l = Float32(0)
            var acc = InlineArray[Float32, M3_HEAD_DIM](fill=Float32(0))
            var selected = 0
            for s in range(M3_INDEX_TOPK_BLOCKS):
                var b = Int(bi[s])
                if b < 0:
                    break
                selected += 1
                var g0 = b * M3_INDEX_BLOCK
                var g1 = min((b + 1) * M3_INDEX_BLOCK, t + 1)
                for g in range(g0, g1):
                    var src_rank = g % tp
                    var lr = g // tp
                    var k_i8g = view.bind(k_ptr)[src_rank] \
                        + lr * M3_KV_DIM + kh * M3_HEAD_DIM
                    var v_i8g = view.bind(v_ptr)[src_rank] \
                        + lr * M3_KV_DIM + kh * M3_HEAD_DIM
                    var ks = view.bind(k_scale_ptr)[src_rank][
                        lr * M3_NUM_KV_HEADS + kh]
                    var vs = view.bind(v_scale_ptr)[src_rank][
                        lr * M3_NUM_KV_HEADS + kh]
                    var raw = i8_vnni_block_dot[M3_HEAD_DIM](k_i8g, q_i8base)
                    var score = Float32(Int(raw)) * fq * ks * INV127SQ
                    var m_new = max(m, score)
                    var corr = exp(m - m_new)
                    var w = exp(score - m_new)
                    for d in range(M3_HEAD_DIM):
                        acc[d] = acc[d] * corr \
                            + w * (Float32(Int(v_i8g[d])) * vs * INV127)
                    l = l * corr + w
                    m = m_new
            if selected < num_blocks:
                sparse_seen = True

            var dst_rank = qh // local_num_q
            var local_h = qh % local_num_q
            var optr = view.bind(output_ptr)[dst_rank] \
                + t * out_stride + local_h * M3_HEAD_DIM
            var pair_worst = Float32(0)
            for d in range(M3_HEAD_DIM):
                var want = acc[d] / l if l > 0 else Float32(0)
                var got = optr[d].cast[DType.float32]()
                var diff = abs(want - got)
                if diff > pair_worst:
                    pair_worst = diff
                if diff > Float32(2e-3) + Float32(0.02) * abs(want):
                    ok = False
            if pair_worst > worst:
                worst = pair_worst
            print(
                t"  tok={t} qh={qh} kh={kh} blocks={num_blocks} "
                t"selected={selected} worst_abs_diff={pair_worst}")

    if ok:
        print(
            t"smoke: PASS (degree={tp}, worst_abs_diff={worst}, "
            t"sparse_exercised={sparse_seen})")
    else:
        print(t"smoke: FAIL (degree={tp}, worst_abs_diff={worst})")


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    print(t"bq sparse attention smoke: {tp} NUMA node(s), seq_len={SEQ_LEN}")

    comptime ARENA_BYTES = 256 * 1024 * 1024
    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_smoke[P: BurstThreadPool, //](var selected_pools: List[P]):
        run_attention(selected_pools^, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_smoke,
    ](topo, "mode: isolated", "mode: spin-backoff")
