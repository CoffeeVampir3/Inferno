from std.collections import InlineArray
from std.memory import Span, UnsafePointer

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.attention_ops import KVRunTable
from kernels.helpers import Binding, RankView
from kernels.profiling import Profiler

from prototypes.lightning_indexer import (
    dispatch_minimax_m3_indexer,
    M3_INDEX_HEAD_DIM, M3_INDEX_NUM_HEADS, M3_INDEX_BLOCK,
    M3_INDEX_TOPK_BLOCKS, M3_INDEX_LOCAL_BLOCKS,
)


comptime ALIGNMENT = 64
comptime SEQ_LEN = 5000
comptime PAGE_LEN = M3_INDEX_BLOCK
comptime Q_ROW = M3_INDEX_NUM_HEADS * M3_INDEX_HEAD_DIM
comptime NUM_KEY_ROWS = ((SEQ_LEN + PAGE_LEN - 1) // PAGE_LEN) * PAGE_LEN
comptime MAX_BLOCK = (SEQ_LEN - 1) // M3_INDEX_BLOCK + 1
comptime BLOCK_STRIDE = (MAX_BLOCK + 15) // 16 * 16


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


def run_indexer[P: BurstThreadPool, //](
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
    print(t"  distributing over degree={tp} (index-K round-robin sharded)")
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var index_q_ptr = arena_alloc_all[BFloat16](arenas, SEQ_LEN * Q_ROW)
    var index_k_ptr = arena_alloc_all[BFloat16](
        arenas, NUM_KEY_ROWS * M3_INDEX_HEAD_DIM)
    var block_idx_ptr = arena_alloc_all[Int32](
        arenas, SEQ_LEN * M3_INDEX_NUM_HEADS * M3_INDEX_TOPK_BLOCKS)
    var partial_ptr = arena_alloc_all[Float32](
        arenas, SEQ_LEN * M3_INDEX_NUM_HEADS * BLOCK_STRIDE)

    for r in range(tp):
        var iq = view.bind(index_q_ptr)[r]
        for i in range(SEQ_LEN * Q_ROW):
            iq[i] = BFloat16(Float32(0.01))
        var ik = view.bind(index_k_ptr)[r]
        for lr in range(NUM_KEY_ROWS):
            var g = lr * tp + r
            var kval = (
                BFloat16(Float32(NUM_KEY_ROWS - g) * 0.001)
                if g < NUM_KEY_ROWS else BFloat16(Float32(0)))
            for d in range(M3_INDEX_HEAD_DIM):
                ik[lr * M3_INDEX_HEAD_DIM + d] = kval
        var bi = view.bind(block_idx_ptr)[r]
        for i in range(SEQ_LEN * M3_INDEX_NUM_HEADS * M3_INDEX_TOPK_BLOCKS):
            bi[i] = Int32(-2)
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
    var index_q = view.bind(index_q_ptr)
    var index_k = view.bind(index_k_ptr)
    var block_idx = view.bind(block_idx_ptr)
    var partial = view.bind(partial_ptr)

    dispatch_minimax_m3_indexer[page_len=PAGE_LEN](
        index_q, index_k, block_idx, partial, runs, SEQ_LEN, pools, prof)

    var out = block_idx[0]
    var probe = [0, 1, 299, 2560, 4999]
    var topk = M3_INDEX_TOPK_BLOCKS
    var heads = M3_INDEX_NUM_HEADS
    var ok = True
    for pi in range(len(probe)):
        var tok = probe[pi]
        var nb = tok // M3_INDEX_BLOCK + 1

        var expected = InlineArray[Int32, M3_INDEX_TOPK_BLOCKS](fill=Int32(-1))
        var ne = 0
        if nb <= topk:
            for b in range(nb):
                expected[ne] = Int32(b)
                ne += 1
        else:
            for b in range(topk - 1):
                expected[ne] = Int32(b)
                ne += 1
            expected[ne] = Int32(nb - 1)
            ne += 1

        var line = String(t"tok={tok} blocks={nb} head0=[")
        var tok_ok = True
        for h in range(heads):
            var row = out + (tok * heads + h) * topk
            for k in range(topk):
                var b = row[k]
                if h == 0:
                    if k > 0:
                        line += ", "
                    line += String(b)
                if b != expected[k]:
                    tok_ok = False
        line += "]  "
        line += "ok" if tok_ok else "MISMATCH"
        print(line)
        if not tok_ok:
            ok = False
    if ok:
        print(t"smoke: PASS (degree={tp}, all {heads} heads match)")
    else:
        print(t"smoke: FAIL (degree={tp})")


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    print(t"lightning indexer smoke: {tp} NUMA node(s), seq_len={SEQ_LEN}")

    comptime ARENA_BYTES = 64 * 1024 * 1024
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
