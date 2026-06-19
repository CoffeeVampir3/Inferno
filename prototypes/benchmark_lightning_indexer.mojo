from std.collections import InlineArray
from std.memory import Span, UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.attention_ops import KVRunTable
from kernels.helpers import Binding, RankView
from kernels.profiling import Profiler
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
)

from prototypes.lightning_indexer import (
    dispatch_minimax_m3_indexer,
    M3_INDEX_HEAD_DIM, M3_INDEX_NUM_HEADS, M3_INDEX_BLOCK,
    M3_INDEX_TOPK_BLOCKS,
)


comptime ALIGNMENT = 64
comptime WARMUP = 10
comptime SAMPLES = 100

comptime PAGE_LEN = M3_INDEX_BLOCK
comptime Q_ROW = M3_INDEX_NUM_HEADS * M3_INDEX_HEAD_DIM
comptime MAX_CONTEXT = 131072
comptime MAX_PREFILL = 4096

comptime MAX_BLOCK = (MAX_CONTEXT - 1) // M3_INDEX_BLOCK + 1
comptime BLOCK_STRIDE = (MAX_BLOCK + 15) // 16 * 16
comptime PREFILL_BLOCK = (MAX_PREFILL - 1) // M3_INDEX_BLOCK + 1
comptime PREFILL_STRIDE = (PREFILL_BLOCK + 15) // 16 * 16
comptime PARTIAL_ELEMS = MAX_PREFILL * PREFILL_STRIDE
comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]


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


def fill_pattern(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = BFloat16(Float32((i % 127) - 63) * 0.01)


def section_validation[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    index_q: Binding[BFloat16, o],
    index_k: Binding[BFloat16, o],
    block_idx: Binding[Int32, o],
    partial: Binding[Float32, o],
):
    print("\n=== Validation (decode, context=8192) ===")
    comptime CTX = 8192
    var prof = Profiler[False]()
    runs[].runs[0].base_pos = Int32(CTX - 1)

    dispatch_minimax_m3_indexer[page_len=PAGE_LEN](
        index_q, index_k, block_idx, partial, runs, 1, pools, prof)

    var out = block_idx[0]
    var nb = (CTX - 1) // M3_INDEX_BLOCK + 1
    var ok = True
    var prev = Int32(-1)
    var has_local = False
    var line = String("  selected=[")
    for k in range(M3_INDEX_TOPK_BLOCKS):
        var b = out[k]
        if k > 0:
            line += ", "
        line += String(b)
        if b >= 0:
            if Int(b) <= Int(prev) or Int(b) >= nb:
                ok = False
            prev = b
            if Int(b) == nb - 1:
                has_local = True
        elif b != Int32(-1):
            ok = False
    print(line + "]")
    print("  ", "OK" if ok and has_local else "FAIL")


def section_decode_sweep[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    index_q: Binding[BFloat16, o],
    index_k: Binding[BFloat16, o],
    block_idx: Binding[Int32, o],
    partial: Binding[Float32, o],
):
    print("\n=== Decode sweep (1 query, vary context; index-K scan bytes) ===")
    var contexts = [1024, 4096, 16384, 65536, 131072]
    var samples = SampleBuffer(SAMPLES)
    var prof = Profiler[False]()

    for ci in range(len(contexts)):
        var ctx = contexts[ci]
        if ctx > MAX_CONTEXT:
            continue
        runs[].runs[0].base_pos = Int32(ctx - 1)

        for _ in range(WARMUP):
            dispatch_minimax_m3_indexer[page_len=PAGE_LEN](
                index_q, index_k, block_idx, partial, runs, 1, pools, prof)
            keep(block_idx[0][0])

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            dispatch_minimax_m3_indexer[page_len=PAGE_LEN](
                index_q, index_k, block_idx, partial, runs, 1, pools, prof)
            var t1 = now_ns()
            samples.push(max_last_ts(pools) - t0, t1 - t0)
        keep(block_idx[0][0])

        var ks = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        var scan_bytes = ctx * M3_INDEX_HEAD_DIM * 2
        print_row(String(t"ctx={ctx}"), ks, ws, scan_bytes)


def section_prefill_sweep[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    index_q: Binding[BFloat16, o],
    index_k: Binding[BFloat16, o],
    block_idx: Binding[Int32, o],
    partial: Binding[Float32, o],
):
    print("\n=== Prefill sweep (N queries from pos 0; triangular scan bytes) ===")
    var seqs = [256, 512, 1024, 2048, 4096]
    var samples = SampleBuffer(SAMPLES)
    var prof = Profiler[False]()

    for si in range(len(seqs)):
        var n = seqs[si]
        if n > MAX_PREFILL:
            continue
        runs[].runs[0].base_pos = Int32(0)

        for _ in range(WARMUP):
            dispatch_minimax_m3_indexer[page_len=PAGE_LEN](
                index_q, index_k, block_idx, partial, runs, n, pools, prof)
            keep(block_idx[0][0])

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            dispatch_minimax_m3_indexer[page_len=PAGE_LEN](
                index_q, index_k, block_idx, partial, runs, n, pools, prof)
            var t1 = now_ns()
            samples.push(max_last_ts(pools) - t0, t1 - t0)
        keep(block_idx[0][0])

        var ks = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        var scan_bytes = (n * (n + 1) // 2) * M3_INDEX_HEAD_DIM * 2
        print_row(String(t"seq={n}"), ks, ws, scan_bytes)


def run_all[P: BurstThreadPool, //](
    var pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    var rows_per_page = PAGE_LEN // tp
    if PAGE_LEN % tp != 0 or rows_per_page < 1 or (
        rows_per_page & (rows_per_page - 1)) != 0:
        print(
            t"degree={tp} does not shard PAGE_LEN={PAGE_LEN} into a power of "
            t"two; skipping")
        return

    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var index_q_ptr = arena_alloc_all[BFloat16](arenas, MAX_PREFILL * Q_ROW)
    var index_k_ptr = arena_alloc_all[BFloat16](
        arenas, MAX_CONTEXT * M3_INDEX_HEAD_DIM)
    var block_idx_ptr = arena_alloc_all[Int32](
        arenas, MAX_PREFILL * M3_INDEX_TOPK_BLOCKS)
    var partial_ptr = arena_alloc_all[Float32](arenas, PARTIAL_ELEMS)

    for r in range(tp):
        fill_pattern(view.bind(index_q_ptr)[r], MAX_PREFILL * Q_ROW)
        fill_pattern(view.bind(index_k_ptr)[r], MAX_CONTEXT * M3_INDEX_HEAD_DIM)
        _ = arenas[r].prefault(0, arenas[r].used())

    var index_q = view.bind(index_q_ptr)
    var index_k = view.bind(index_k_ptr)
    var block_idx = view.bind(block_idx_ptr)
    var partial = view.bind(partial_ptr)

    var cap = pools[0].get_capacity()
    print(
        t"index: heads={M3_INDEX_NUM_HEADS} head_dim={M3_INDEX_HEAD_DIM} "
        t"block={M3_INDEX_BLOCK} topk={M3_INDEX_TOPK_BLOCKS}")
    print(t"degree={tp} pool_capacity={cap} workers/node")

    var rows_per_page_v = rows_per_page
    var runs_table = KVRunTable()
    runs_table.begin_run(0, 0)
    var num_local_rows = (MAX_CONTEXT + tp - 1) // tp
    var num_pages = (num_local_rows + rows_per_page_v - 1) // rows_per_page_v
    for g in range(num_pages):
        runs_table.add_base_row(Int32(g * rows_per_page_v))
    var runs = UnsafePointer(to=runs_table).as_unsafe_any_origin()

    section_validation(pools, runs, index_q, index_k, block_idx, partial)
    section_decode_sweep(pools, runs, index_q, index_k, block_idx, partial)
    section_prefill_sweep(pools, runs, index_q, index_k, block_idx, partial)


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    var iso = len(topo.isolated_cpus)
    print("MiniMax-M3 lightning indexer benchmark")
    print(t"{tp} NUMA node(s), {iso} isolated cpus")

    comptime ARENA_BYTES = 256 * 1024 * 1024
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
