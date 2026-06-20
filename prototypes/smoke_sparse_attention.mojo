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

from prototypes.lightning_indexer import (
    dispatch_minimax_m3_indexer,
    M3_INDEX_HEAD_DIM, M3_INDEX_NUM_HEADS, M3_INDEX_BLOCK,
    M3_INDEX_TOPK_BLOCKS,
)
from prototypes.sparse_attention import (
    dispatch_minimax_m3_sparse_attention,
    M3_NUM_HEADS, M3_NUM_KV_HEADS, M3_HEAD_DIM, M3_GQA_RATIO, M3_KV_DIM,
    M3_ATTN_SCALE,
)


comptime ALIGNMENT = 64
comptime MAX_WORKERS = 128
comptime SEQ_LEN = 2600
comptime PAGE_LEN = M3_INDEX_BLOCK
comptime Q_ROW = M3_NUM_HEADS * M3_HEAD_DIM
comptime INDEX_Q_ROW = M3_INDEX_NUM_HEADS * M3_INDEX_HEAD_DIM
comptime NUM_KEY_ROWS = ((SEQ_LEN + PAGE_LEN - 1) // PAGE_LEN) * PAGE_LEN
comptime MAX_BLOCK = (SEQ_LEN - 1) // M3_INDEX_BLOCK + 1
comptime BLOCK_STRIDE = (MAX_BLOCK + 15) // 16 * 16
comptime PARTIAL_STRIDE = (
    (M3_NUM_HEADS * M3_HEAD_DIM + 2 * M3_NUM_HEADS) * 4 + 63) // 64 * 16


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
def qval(t: Int, qh: Int, d: Int) -> BFloat16:
    var base = Float32(0.01) + Float32(0.0001) * Float32((t + qh + d) % 7)
    return BFloat16(base * M3_ATTN_SCALE)


@always_inline
def kval(g: Int, kh: Int, d: Int) -> BFloat16:
    return BFloat16(Float32(0.01) + Float32(0.0001) * Float32((g + kh + d) % 5))


@always_inline
def vval(g: Int, kh: Int, d: Int) -> BFloat16:
    return BFloat16(Float32(0.001) * Float32((g + 2 * d) % 11))


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
    print(t"  distributing over degree={tp} (K/V round-robin sharded)")
    var local_num_q = M3_NUM_HEADS // tp
    if M3_NUM_HEADS % tp != 0:
        print(t"  degree={tp} does not divide num_heads={M3_NUM_HEADS}; skip")
        return
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var index_q_ptr = arena_alloc_all[BFloat16](arenas, SEQ_LEN * INDEX_Q_ROW)
    var index_k_ptr = arena_alloc_all[BFloat16](
        arenas, NUM_KEY_ROWS * M3_INDEX_HEAD_DIM)
    var block_idx_ptr = arena_alloc_all[Int32](
        arenas, SEQ_LEN * M3_INDEX_NUM_HEADS * M3_INDEX_TOPK_BLOCKS)
    var index_partial_ptr = arena_alloc_all[Float32](
        arenas, SEQ_LEN * M3_INDEX_NUM_HEADS * BLOCK_STRIDE)

    var q_ptr = arena_alloc_all[BFloat16](arenas, SEQ_LEN * Q_ROW)
    var k_ptr = arena_alloc_all[BFloat16](arenas, NUM_KEY_ROWS * M3_KV_DIM)
    var v_ptr = arena_alloc_all[BFloat16](arenas, NUM_KEY_ROWS * M3_KV_DIM)
    var output_ptr = arena_alloc_all[BFloat16](arenas, SEQ_LEN * Q_ROW)
    var attn_partial_ptr = arena_alloc_all[Float32](
        arenas, SEQ_LEN * PARTIAL_STRIDE)
    var segment_ptr = arena_alloc_all[MergeSegment](arenas, MAX_WORKERS * tp)

    for r in range(tp):
        var iq = view.bind(index_q_ptr)[r]
        for i in range(SEQ_LEN * INDEX_Q_ROW):
            iq[i] = BFloat16(Float32(0.01))
        var ik = view.bind(index_k_ptr)[r]
        for lr in range(NUM_KEY_ROWS):
            var g = lr * tp + r
            var kv = (
                BFloat16(Float32(NUM_KEY_ROWS - g) * 0.001)
                if g < NUM_KEY_ROWS else BFloat16(Float32(0)))
            for d in range(M3_INDEX_HEAD_DIM):
                ik[lr * M3_INDEX_HEAD_DIM + d] = kv

        var qb = view.bind(q_ptr)[r]
        for t in range(SEQ_LEN):
            for qh in range(M3_NUM_HEADS):
                for d in range(M3_HEAD_DIM):
                    qb[t * Q_ROW + qh * M3_HEAD_DIM + d] = qval(t, qh, d)

        var kb = view.bind(k_ptr)[r]
        var vb = view.bind(v_ptr)[r]
        for lr in range(NUM_KEY_ROWS):
            var g = lr * tp + r
            for kh in range(M3_NUM_KV_HEADS):
                for d in range(M3_HEAD_DIM):
                    var off = lr * M3_KV_DIM + kh * M3_HEAD_DIM + d
                    if g < SEQ_LEN:
                        kb[off] = kval(g, kh, d)
                        vb[off] = vval(g, kh, d)
                    else:
                        kb[off] = BFloat16(Float32(0))
                        vb[off] = BFloat16(Float32(0))

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
    var runs = UnsafePointer(to=runs_table).as_unsafe_any_origin()

    var prof = Profiler[False]()
    var index_q = view.bind(index_q_ptr)
    var index_k = view.bind(index_k_ptr)
    var block_idx = view.bind(block_idx_ptr)
    var index_partial = view.bind(index_partial_ptr)

    dispatch_minimax_m3_indexer[page_len=PAGE_LEN](
        index_q, index_k, block_idx, index_partial, runs, SEQ_LEN, pools, prof)

    var q = view.bind(q_ptr)
    var k = view.bind(k_ptr)
    var v = view.bind(v_ptr)
    var output = view.bind(output_ptr)
    var attn_partial = view.bind(attn_partial_ptr)
    var segments = view.bind(segment_ptr)

    dispatch_minimax_m3_sparse_attention[page_len=PAGE_LEN](
        q, k, v, block_idx, output, attn_partial, segments, runs, SEQ_LEN,
        pools, prof)

    var probe_tok = [0, 1, 200, 2048, 2560, 2599]
    var probe_head = [0, 15, 16, 33, 63]
    comptime bi_tstride = M3_INDEX_NUM_HEADS * M3_INDEX_TOPK_BLOCKS
    var out_stride = local_num_q * M3_HEAD_DIM
    var worst = Float32(0)
    var sparse_seen = False
    var ok = True
    for pti in range(len(probe_tok)):
        var t = probe_tok[pti]
        if t >= SEQ_LEN:
            continue
        var abs_pos = t
        var num_blocks = abs_pos // M3_INDEX_BLOCK + 1
        for phi in range(len(probe_head)):
            var qh = probe_head[phi]
            var kh = qh // M3_GQA_RATIO
            var bi = view.bind(block_idx_ptr)[0] \
                + t * bi_tstride + kh * M3_INDEX_TOPK_BLOCKS

            var m = Float32(-1e30)
            var l = Float32(0)
            var acc = InlineArray[Float32, M3_HEAD_DIM](fill=Float32(0))
            var qbase = view.bind(q_ptr)[0] + t * Q_ROW + qh * M3_HEAD_DIM
            var selected = 0
            for s in range(M3_INDEX_TOPK_BLOCKS):
                var b = Int(bi[s])
                if b < 0:
                    break
                selected += 1
                var g0 = b * M3_INDEX_BLOCK
                var g1 = min((b + 1) * M3_INDEX_BLOCK, abs_pos + 1)
                for g in range(g0, g1):
                    var src_rank = g % tp
                    var lr = g // tp
                    var kbg = view.bind(k_ptr)[src_rank] \
                        + lr * M3_KV_DIM + kh * M3_HEAD_DIM
                    var vbg = view.bind(v_ptr)[src_rank] \
                        + lr * M3_KV_DIM + kh * M3_HEAD_DIM
                    var score = Float32(0)
                    for d in range(M3_HEAD_DIM):
                        score += qbase[d].cast[DType.float32]() \
                            * kbg[d].cast[DType.float32]()
                    var m_new = max(m, score)
                    var corr = exp(m - m_new)
                    var w = exp(score - m_new)
                    for d in range(M3_HEAD_DIM):
                        acc[d] = acc[d] * corr \
                            + w * vbg[d].cast[DType.float32]()
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
                var want = acc[d] / l
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
    print(t"sparse attention smoke: {tp} NUMA node(s), seq_len={SEQ_LEN}")

    comptime ARENA_BYTES = 384 * 1024 * 1024
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
