from std.collections import InlineArray
from std.memory import Span, UnsafePointer

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.attention_ops import KVRunTable
from kernels.helpers import Binding, RankView
from kernels.flash_attention_prefill import dispatch_merge_flash_prefill_partials
from kernels.logsum_merge import MergeSegment
from kernels.profiling import Profiler

from prototypes.lightning_indexer import M3_INDEX_BLOCK, M3_INDEX_TOPK_BLOCKS
from prototypes.bq_sparse_attention import (
    dispatch_bq_minimax_m3_sparse_attention,
    dispatch_bq_block_sparse_flash,
    M3_NUM_HEADS, M3_NUM_KV_HEADS, M3_HEAD_DIM, M3_GQA_RATIO, M3_KV_DIM,
)


comptime ALIGNMENT = 64
comptime MAX_WORKERS = 128
comptime PAGE_LEN = M3_INDEX_BLOCK
comptime Q_ROW = M3_NUM_HEADS * M3_HEAD_DIM
comptime MAX_CONTEXT = 32768
comptime PARTIAL_STRIDE = (
    (M3_NUM_HEADS * M3_HEAD_DIM + 2 * M3_NUM_HEADS) * 4 + 63) // 64 * 16
comptime F_Q = Float32(0.3)

comptime I8Ptr = UnsafePointer[Int8, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime I32Ptr = UnsafePointer[Int32, MutAnyOrigin]


def arena_bases(mut arenas: List[NumaArena[alignment=ALIGNMENT]]) -> List[Int]:
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


def fill_i8(ptr: I8Ptr, count: Int):
    for i in range(count):
        ptr[i] = Int8((i % 13) - 6)


def fill_f32(ptr: F32Ptr, count: Int, val: Float32):
    for i in range(count):
        ptr[i] = val


def fill_block_idx(block_idx: Binding[Int32, _], base_pos: Int, tp: Int):
    comptime bi_tstride = M3_NUM_KV_HEADS * M3_INDEX_TOPK_BLOCKS
    var nb = base_pos // M3_INDEX_BLOCK + 1
    var cnt = min(M3_INDEX_TOPK_BLOCKS, nb)
    var stride = max(1, nb // M3_INDEX_TOPK_BLOCKS)
    for r in range(tp):
        var b = block_idx[r]
        for kh in range(M3_NUM_KV_HEADS):
            var row = b + kh * M3_INDEX_TOPK_BLOCKS
            for s in range(M3_INDEX_TOPK_BLOCKS):
                if s < cnt:
                    row[s] = Int32(min(nb - 1, s * stride))
                else:
                    row[s] = Int32(-1)


def run_all[P: BurstThreadPool, //](
    var pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    var rows_per_page = PAGE_LEN // tp
    var local_num_q = M3_NUM_HEADS // tp
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var q_ptr = arena_alloc_all[Int8](arenas, Q_ROW)
    var qi_bias_ptr = arena_alloc_all[Float32](arenas, M3_NUM_HEADS)
    var f_q_ptr = arena_alloc_all[Float32](arenas, M3_NUM_HEADS)
    var k_ptr = arena_alloc_all[Int8](arenas, MAX_CONTEXT * M3_KV_DIM)
    var k_scale_ptr = arena_alloc_all[Float32](arenas, MAX_CONTEXT * M3_NUM_KV_HEADS)
    var v_ptr = arena_alloc_all[Int8](arenas, MAX_CONTEXT * M3_KV_DIM)
    var v_scale_ptr = arena_alloc_all[Float32](arenas, MAX_CONTEXT * M3_NUM_KV_HEADS)
    var block_idx_ptr = arena_alloc_all[Int32](
        arenas, M3_NUM_KV_HEADS * M3_INDEX_TOPK_BLOCKS)
    var out_new_ptr = arena_alloc_all[BFloat16](arenas, Q_ROW)
    var out_ref_ptr = arena_alloc_all[BFloat16](arenas, Q_ROW)
    var part_new_ptr = arena_alloc_all[Float32](arenas, MAX_WORKERS * PARTIAL_STRIDE)
    var part_ref_ptr = arena_alloc_all[Float32](arenas, MAX_WORKERS * PARTIAL_STRIDE)
    var seg_new_ptr = arena_alloc_all[MergeSegment](arenas, MAX_WORKERS * tp)
    var seg_ref_ptr = arena_alloc_all[MergeSegment](arenas, MAX_WORKERS * tp)

    for r in range(tp):
        fill_i8(view.bind(q_ptr)[r], Q_ROW)
        fill_f32(view.bind(qi_bias_ptr)[r], M3_NUM_HEADS, 0.0)
        fill_f32(view.bind(f_q_ptr)[r], M3_NUM_HEADS, F_Q)
        fill_i8(view.bind(k_ptr)[r], MAX_CONTEXT * M3_KV_DIM)
        fill_f32(view.bind(k_scale_ptr)[r], MAX_CONTEXT * M3_NUM_KV_HEADS, 1.0)
        fill_i8(view.bind(v_ptr)[r], MAX_CONTEXT * M3_KV_DIM)
        fill_f32(view.bind(v_scale_ptr)[r], MAX_CONTEXT * M3_NUM_KV_HEADS, 1.0)
        _ = arenas[r].prefault(0, arenas[r].used())

    var q = view.bind(q_ptr)
    var qi_bias = view.bind(qi_bias_ptr)
    var f_q = view.bind(f_q_ptr)
    var k = view.bind(k_ptr)
    var k_scale = view.bind(k_scale_ptr)
    var v = view.bind(v_ptr)
    var v_scale = view.bind(v_scale_ptr)
    var block_idx = view.bind(block_idx_ptr)
    var out_new = view.bind(out_new_ptr)
    var out_ref = view.bind(out_ref_ptr)
    var part_new = view.bind(part_new_ptr)
    var part_ref = view.bind(part_ref_ptr)
    var seg_new = view.bind(seg_new_ptr)
    var seg_ref = view.bind(seg_ref_ptr)

    var runs_table = KVRunTable()
    runs_table.begin_run(0, 0)
    var num_local_rows = (MAX_CONTEXT + tp - 1) // tp
    var num_pages = (num_local_rows + rows_per_page - 1) // rows_per_page
    for g in range(num_pages):
        runs_table.add_base_row(Int32(g * rows_per_page))
    var runs = UnsafePointer(to=runs_table).as_unsafe_any_origin()

    var prof = Profiler[False]()
    var contexts = [300, 1024, 4096, 16384, 32768]
    var worst_all = Float32(0.0)
    var failed = False

    for ci in range(len(contexts)):
        var ctx = contexts[ci]
        if ctx > MAX_CONTEXT:
            continue
        runs[].runs[0].base_pos = Int32(ctx - 1)
        fill_block_idx(block_idx, ctx - 1, tp)

        # candidate: worker-split decode (seq_len == 1 routes through new branch)
        dispatch_bq_minimax_m3_sparse_attention[page_len=PAGE_LEN](
            q, qi_bias, f_q, k, k_scale, v, v_scale, block_idx,
            out_new, part_new, seg_new, runs, 1, pools, prof)

        # reference: token-partitioned flash + prefill merge (old decode path)
        dispatch_bq_block_sparse_flash[
            head_dim=M3_HEAD_DIM, num_q=M3_NUM_HEADS, num_kv=M3_NUM_KV_HEADS,
            gqa_ratio=M3_GQA_RATIO, block_size=M3_INDEX_BLOCK,
            topk_blocks=M3_INDEX_TOPK_BLOCKS, partial_stride=PARTIAL_STRIDE,
            page_len=PAGE_LEN,
        ](q, qi_bias, f_q, k, k_scale, v, v_scale, block_idx, part_ref,
          runs, M3_KV_DIM, 1, pools, prof)
        dispatch_merge_flash_prefill_partials[M3_HEAD_DIM](
            out_ref, part_ref, seg_ref, M3_NUM_HEADS, local_num_q,
            PARTIAL_STRIDE, 1, pools, prof)

        var worst = Float32(0.0)
        for r in range(tp):
            var on = out_new[r]
            var orf = out_ref[r]
            for i in range(local_num_q * M3_HEAD_DIM):
                var d = abs(Float32(on[i]) - Float32(orf[i]))
                if d > worst:
                    worst = d
        if worst > worst_all:
            worst_all = worst
        if worst > 0.02:
            failed = True
        print(t"ctx={ctx} worst_abs_diff={worst}")

    if failed:
        print(t"decode smoke: FAIL (degree={tp}, worst_abs_diff={worst_all})")
    else:
        print(t"decode smoke: PASS (degree={tp}, worst_abs_diff={worst_all})")
    _ = runs_table


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    print(t"bq sparse decode equivalence smoke: {tp} NUMA node(s)")

    comptime ARENA_BYTES = 1024 * 1024 * 1024
    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_bench[P: BurstThreadPool, //](var selected_pools: List[P]):
        run_all(selected_pools^, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_bench,
    ](topo, "mode: isolated", "mode: spin-backoff")
