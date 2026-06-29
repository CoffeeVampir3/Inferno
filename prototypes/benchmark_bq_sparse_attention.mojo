from std.collections import InlineArray
from std.memory import Span, UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.attention_ops import KVRunTable
from kernels.helpers import Binding, RankView
from kernels.logsum_merge import MergeSegment
from kernels.profiling import Profiler
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
)

from prototypes.lightning_indexer import M3_INDEX_BLOCK, M3_INDEX_TOPK_BLOCKS
from prototypes.bq_sparse_attention import (
    dispatch_bq_minimax_m3_sparse_attention,
    M3_NUM_HEADS, M3_NUM_KV_HEADS, M3_HEAD_DIM, M3_KV_DIM,
)


comptime ALIGNMENT = 64
comptime MAX_WORKERS = 128
comptime WARMUP = 10
comptime SAMPLES = 100

comptime PAGE_LEN = M3_INDEX_BLOCK
comptime ATTN_Q_ROW = M3_NUM_HEADS * M3_HEAD_DIM
comptime MAX_CONTEXT = 131072
comptime MAX_PREFILL = 4096
comptime ATTN_PARTIAL_STRIDE = (
    (M3_NUM_HEADS * M3_HEAD_DIM + 2 * M3_NUM_HEADS) * 4 + 63) // 64 * 16

comptime I8Ptr = UnsafePointer[Int8, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime I32Ptr = UnsafePointer[Int32, MutAnyOrigin]
comptime F_Q = Float32(0.3)


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


def fill_i8(ptr: I8Ptr, count: Int):
    for i in range(count):
        ptr[i] = Int8((i % 13) - 6)


def fill_f32(ptr: F32Ptr, count: Int, val: Float32):
    for i in range(count):
        ptr[i] = val


def fill_block_idx(
    block_idx: Binding[Int32, _], base_pos: Int, seq_len: Int, tp: Int,
):
    comptime bi_tstride = M3_NUM_KV_HEADS * M3_INDEX_TOPK_BLOCKS
    for r in range(tp):
        var b = block_idx[r]
        for t in range(seq_len):
            var nb = (base_pos + t) // M3_INDEX_BLOCK + 1
            var cnt = min(M3_INDEX_TOPK_BLOCKS, nb)
            var stride = max(1, nb // M3_INDEX_TOPK_BLOCKS)
            for kh in range(M3_NUM_KV_HEADS):
                var row = b + t * bi_tstride + kh * M3_INDEX_TOPK_BLOCKS
                for s in range(M3_INDEX_TOPK_BLOCKS):
                    if s < cnt:
                        row[s] = Int32(min(nb - 1, s * stride))
                    else:
                        row[s] = Int32(-1)


@always_inline
def attn_read_bytes(seq_len: Int, base_pos: Int) -> Int:
    var total = 0
    for i in range(seq_len):
        var nb = (base_pos + i) // M3_INDEX_BLOCK + 1
        var used = min(M3_INDEX_TOPK_BLOCKS, nb)
        total += M3_NUM_KV_HEADS * used * M3_INDEX_BLOCK * M3_HEAD_DIM * 2
    return total


@fieldwise_init
struct FusedBuffers[o: ImmutOrigin](Copyable, ImplicitlyCopyable):
    var q: Binding[Int8, Self.o]
    var qi_bias: Binding[Float32, Self.o]
    var f_q: Binding[Float32, Self.o]
    var k: Binding[Int8, Self.o]
    var k_scale: Binding[Float32, Self.o]
    var v: Binding[Int8, Self.o]
    var v_scale: Binding[Float32, Self.o]
    var block_idx: Binding[Int32, Self.o]
    var output: Binding[BFloat16, Self.o]
    var attn_partial: Binding[Float32, Self.o]
    var segments: Binding[MergeSegment, Self.o]


def run_attention[P: BurstThreadPool, o: ImmutOrigin, N: Int, //](
    mut pools: List[P],
    buf: FusedBuffers[o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    seq_len: Int,
    mut prof: Profiler[False, N],
):
    dispatch_bq_minimax_m3_sparse_attention[page_len=PAGE_LEN](
        buf.q, buf.qi_bias, buf.f_q, buf.k, buf.k_scale, buf.v, buf.v_scale,
        buf.block_idx, buf.output, buf.attn_partial, buf.segments,
        runs, seq_len, pools, prof)


def section_decode_sweep[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    buf: FusedBuffers[o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    tp: Int,
):
    print("\n=== Decode sweep (1 query; topk blocks scattered over context) ===")
    var contexts = [1024, 4096, 16384, 65536, 131072]
    var s_attn = SampleBuffer(SAMPLES)
    var prof = Profiler[False]()

    for ci in range(len(contexts)):
        var ctx = contexts[ci]
        if ctx > MAX_CONTEXT:
            continue
        runs[].runs[0].base_pos = Int32(ctx - 1)
        fill_block_idx(buf.block_idx, ctx - 1, 1, tp)

        for _ in range(WARMUP):
            run_attention(pools, buf, runs, 1, prof)
            keep(buf.output[0][0])

        s_attn.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            run_attention(pools, buf, runs, 1, prof)
            var t1 = now_ns()
            s_attn.push(max_last_ts(pools) - t0, t1 - t0)
        keep(buf.output[0][0])

        print(t"ctx={ctx}")
        print_row("  attention",
            compute_stats(s_attn.kernel_ns, s_attn.n),
            compute_stats(s_attn.wall_ns, s_attn.n),
            attn_read_bytes(1, ctx - 1))


def section_prefill_sweep[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    buf: FusedBuffers[o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    tp: Int,
):
    print("\n=== Prefill sweep (N queries from pos 0) ===")
    var seqs = [256, 512, 1024, 2048, 4096]
    var s_attn = SampleBuffer(SAMPLES)
    var prof = Profiler[False]()

    for si in range(len(seqs)):
        var n = seqs[si]
        if n > MAX_PREFILL:
            continue
        runs[].runs[0].base_pos = Int32(0)
        fill_block_idx(buf.block_idx, 0, n, tp)

        for _ in range(WARMUP):
            run_attention(pools, buf, runs, n, prof)
            keep(buf.output[0][0])

        s_attn.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            run_attention(pools, buf, runs, n, prof)
            var t1 = now_ns()
            s_attn.push(max_last_ts(pools) - t0, t1 - t0)
        keep(buf.output[0][0])

        print(t"seq={n}")
        print_row("  attention",
            compute_stats(s_attn.kernel_ns, s_attn.n),
            compute_stats(s_attn.wall_ns, s_attn.n),
            attn_read_bytes(n, 0))


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
    if M3_NUM_HEADS % tp != 0:
        print(t"degree={tp} does not divide num_heads={M3_NUM_HEADS}; skipping")
        return

    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))

    var q_ptr = arena_alloc_all[Int8](arenas, MAX_PREFILL * ATTN_Q_ROW)
    var qi_bias_ptr = arena_alloc_all[Float32](
        arenas, MAX_PREFILL * M3_NUM_HEADS)
    var f_q_ptr = arena_alloc_all[Float32](arenas, MAX_PREFILL * M3_NUM_HEADS)
    var k_ptr = arena_alloc_all[Int8](arenas, MAX_CONTEXT * M3_KV_DIM)
    var k_scale_ptr = arena_alloc_all[Float32](
        arenas, MAX_CONTEXT * M3_NUM_KV_HEADS)
    var v_ptr = arena_alloc_all[Int8](arenas, MAX_CONTEXT * M3_KV_DIM)
    var v_scale_ptr = arena_alloc_all[Float32](
        arenas, MAX_CONTEXT * M3_NUM_KV_HEADS)
    var block_idx_ptr = arena_alloc_all[Int32](
        arenas, MAX_PREFILL * M3_NUM_KV_HEADS * M3_INDEX_TOPK_BLOCKS)
    var output_ptr = arena_alloc_all[BFloat16](arenas, MAX_PREFILL * ATTN_Q_ROW)
    var attn_partial_ptr = arena_alloc_all[Float32](
        arenas, MAX_PREFILL * ATTN_PARTIAL_STRIDE)
    var segment_ptr = arena_alloc_all[MergeSegment](arenas, MAX_WORKERS * tp)

    for r in range(tp):
        fill_i8(view.bind(q_ptr)[r], MAX_PREFILL * ATTN_Q_ROW)
        fill_f32(view.bind(qi_bias_ptr)[r], MAX_PREFILL * M3_NUM_HEADS, 0.0)
        fill_f32(view.bind(f_q_ptr)[r], MAX_PREFILL * M3_NUM_HEADS, F_Q)
        fill_i8(view.bind(k_ptr)[r], MAX_CONTEXT * M3_KV_DIM)
        fill_f32(view.bind(k_scale_ptr)[r], MAX_CONTEXT * M3_NUM_KV_HEADS, 1.0)
        fill_i8(view.bind(v_ptr)[r], MAX_CONTEXT * M3_KV_DIM)
        fill_f32(view.bind(v_scale_ptr)[r], MAX_CONTEXT * M3_NUM_KV_HEADS, 1.0)
        _ = arenas[r].prefault(0, arenas[r].used())

    var buf = FusedBuffers(
        view.bind(q_ptr), view.bind(qi_bias_ptr), view.bind(f_q_ptr),
        view.bind(k_ptr), view.bind(k_scale_ptr),
        view.bind(v_ptr), view.bind(v_scale_ptr),
        view.bind(block_idx_ptr), view.bind(output_ptr),
        view.bind(attn_partial_ptr), view.bind(segment_ptr))

    var cap = pools[0].get_capacity()
    print(
        t"bq sparse attn: heads={M3_NUM_HEADS} kv_heads={M3_NUM_KV_HEADS} "
        t"head_dim={M3_HEAD_DIM} block={M3_INDEX_BLOCK} "
        t"topk={M3_INDEX_TOPK_BLOCKS}")
    print(t"degree={tp} pool_capacity={cap} workers/node")

    var runs_table = KVRunTable()
    runs_table.begin_run(0, 0)
    var num_local_rows = (MAX_CONTEXT + tp - 1) // tp
    var num_pages = (num_local_rows + rows_per_page - 1) // rows_per_page
    for g in range(num_pages):
        runs_table.add_base_row(Int32(g * rows_per_page))
    var runs = UnsafePointer(to=runs_table).as_unsafe_any_origin()

    section_decode_sweep(pools, buf, runs, tp)
    section_prefill_sweep(pools, buf, runs, tp)
    _ = runs_table


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    var iso = len(topo.isolated_cpus)
    print("MiniMax-M3 ButterQuant sparse attention benchmark")
    print(t"{tp} NUMA node(s), {iso} isolated cpus")

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
