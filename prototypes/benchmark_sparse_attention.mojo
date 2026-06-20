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

from prototypes.lightning_indexer import (
    dispatch_minimax_m3_indexer,
    M3_INDEX_HEAD_DIM, M3_INDEX_NUM_HEADS, M3_INDEX_BLOCK,
    M3_INDEX_TOPK_BLOCKS,
)
from prototypes.sparse_attention import (
    dispatch_minimax_m3_sparse_attention,
    M3_NUM_HEADS, M3_NUM_KV_HEADS, M3_HEAD_DIM, M3_KV_DIM,
)
from prototypes.sparse_attention_reuse import (
    dispatch_minimax_m3_sparse_attention_reuse,
)


comptime ALIGNMENT = 64
comptime MAX_WORKERS = 128
comptime WARMUP = 10
comptime SAMPLES = 100

comptime PAGE_LEN = M3_INDEX_BLOCK
comptime INDEX_Q_ROW = M3_INDEX_NUM_HEADS * M3_INDEX_HEAD_DIM
comptime ATTN_Q_ROW = M3_NUM_HEADS * M3_HEAD_DIM
comptime MAX_CONTEXT = 131072
comptime MAX_PREFILL = 4096

comptime PREFILL_BLOCK = (MAX_PREFILL - 1) // M3_INDEX_BLOCK + 1
comptime PREFILL_STRIDE = (PREFILL_BLOCK + 15) // 16 * 16
comptime INDEX_PARTIAL_ELEMS = MAX_PREFILL * M3_INDEX_NUM_HEADS * PREFILL_STRIDE
comptime ATTN_PARTIAL_STRIDE = (
    (M3_NUM_HEADS * M3_HEAD_DIM + 2 * M3_NUM_HEADS) * 4 + 63) // 64 * 16
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


@always_inline
def indexer_scan_bytes(seq_len: Int, base_pos: Int) -> Int:
    if seq_len == 1:
        return (base_pos + 1) * M3_INDEX_HEAD_DIM * 2
    return (seq_len * (seq_len + 1) // 2) * M3_INDEX_HEAD_DIM * 2


@always_inline
def attn_read_bytes(seq_len: Int, base_pos: Int) -> Int:
    var total = 0
    for i in range(seq_len):
        var abs_pos = base_pos + i
        var nb = abs_pos // M3_INDEX_BLOCK + 1
        var used = min(M3_INDEX_TOPK_BLOCKS, nb)
        total += M3_NUM_KV_HEADS * used * M3_INDEX_BLOCK * M3_HEAD_DIM * 2 * 2
    return total


def speedup_line(baseline_p50: Int64, reuse_p50: Int64) -> String:
    if reuse_p50 <= 0:
        return String("    speedup: n/a")
    var x100 = baseline_p50 * 100 // reuse_p50
    var whole = x100 // 100
    var frac = x100 % 100
    var pad = "0" if frac < 10 else ""
    return String(t"    speedup (kernel p50): {whole}.{pad}{frac}x")


@fieldwise_init
struct FusedBuffers[o: ImmutOrigin](Copyable, ImplicitlyCopyable):
    var index_q: Binding[BFloat16, Self.o]
    var index_k: Binding[BFloat16, Self.o]
    var block_idx: Binding[Int32, Self.o]
    var index_partial: Binding[Float32, Self.o]
    var q: Binding[BFloat16, Self.o]
    var k: Binding[BFloat16, Self.o]
    var v: Binding[BFloat16, Self.o]
    var output: Binding[BFloat16, Self.o]
    var attn_partial: Binding[Float32, Self.o]
    var segments: Binding[MergeSegment, Self.o]


def run_indexer[P: BurstThreadPool, o: ImmutOrigin, N: Int, //](
    mut pools: List[P],
    buf: FusedBuffers[o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    seq_len: Int,
    mut prof: Profiler[False, N],
):
    dispatch_minimax_m3_indexer[page_len=PAGE_LEN](
        buf.index_q, buf.index_k, buf.block_idx, buf.index_partial,
        runs, seq_len, pools, prof)


def run_attention[P: BurstThreadPool, o: ImmutOrigin, N: Int, //](
    mut pools: List[P],
    buf: FusedBuffers[o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    seq_len: Int,
    mut prof: Profiler[False, N],
):
    dispatch_minimax_m3_sparse_attention[page_len=PAGE_LEN](
        buf.q, buf.k, buf.v, buf.block_idx, buf.output, buf.attn_partial,
        buf.segments, runs, seq_len, pools, prof)


def run_attention_reuse[P: BurstThreadPool, o: ImmutOrigin, N: Int, //](
    mut pools: List[P],
    buf: FusedBuffers[o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    seq_len: Int,
    mut prof: Profiler[False, N],
):
    dispatch_minimax_m3_sparse_attention_reuse[page_len=PAGE_LEN](
        buf.q, buf.k, buf.v, buf.block_idx, buf.output, buf.attn_partial,
        buf.segments, runs, seq_len, pools, prof)


def section_validation[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    buf: FusedBuffers[o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
):
    print("\n=== Validation (fused decode, context=8192) ===")
    comptime CTX = 8192
    var prof = Profiler[False]()
    runs[].runs[0].base_pos = Int32(CTX - 1)

    var sentinel = BFloat16(Float32(-999))
    var ob = buf.output[0]
    for d in range(M3_HEAD_DIM):
        ob[d] = sentinel

    run_indexer(pools, buf, runs, 1, prof)
    run_attention(pools, buf, runs, 1, prof)

    var sel = buf.block_idx[0]
    var nb = (CTX - 1) // M3_INDEX_BLOCK + 1
    var ok = True
    var prev = Int32(-1)
    var has_local = False
    for k in range(M3_INDEX_TOPK_BLOCKS):
        var b = sel[k]
        if b >= 0:
            if Int(b) <= Int(prev) or Int(b) >= nb:
                ok = False
            prev = b
            if Int(b) == nb - 1:
                has_local = True
        elif b != Int32(-1):
            ok = False

    var finite = True
    var changed = False
    for d in range(M3_HEAD_DIM):
        var x = ob[d].cast[DType.float32]()
        if not (x == x) or abs(x) > Float32(1e30):
            finite = False
        if ob[d] != sentinel:
            changed = True
    var sel_ok = ok and has_local
    print(
        t"  block_select_ok={sel_ok} output_finite={finite} "
        t"output_written={changed}")
    print("  ", "OK" if (sel_ok and finite and changed) else "FAIL")


def parity_pass[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    buf: FusedBuffers[o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    label: StringSlice,
    base_pos: Int,
    seq_len: Int,
):
    var prof = Profiler[False]()
    runs[].runs[0].base_pos = Int32(base_pos)
    var total = seq_len * ATTN_Q_ROW

    run_indexer(pools, buf, runs, seq_len, prof)
    run_attention(pools, buf, runs, seq_len, prof)
    var ob = buf.output[0]
    var saved = List[Float32](capacity=total)
    for i in range(total):
        saved.append(ob[i].cast[DType.float32]())

    run_attention_reuse(pools, buf, runs, seq_len, prof)
    var max_abs = Float32(0)
    var max_rel = Float32(0)
    for i in range(total):
        var a = saved[i]
        var b = ob[i].cast[DType.float32]()
        var d = abs(a - b)
        if d > max_abs:
            max_abs = d
        var denom = abs(a) if abs(a) > Float32(1e-6) else Float32(1e-6)
        var rel = d / denom
        if rel > max_rel:
            max_rel = rel
    print(t"  {label}: n={total} max_abs={max_abs} max_rel={max_rel}")


def section_parity[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    buf: FusedBuffers[o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
):
    print("\n=== Parity: baseline vs K/V-reuse (final merged output) ===")
    parity_pass(pools, buf, runs, "decode  ctx=65536", 65535, 1)
    parity_pass(pools, buf, runs, "prefill seq=128  ", 0, 128)


def section_decode_sweep[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    buf: FusedBuffers[o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
):
    print("\n=== Decode sweep (1 query; indexer grows, attention ~flat) ===")
    var contexts = [1024, 4096, 16384, 65536, 131072]
    var s_idx = SampleBuffer(SAMPLES)
    var s_attn = SampleBuffer(SAMPLES)
    var s_reuse = SampleBuffer(SAMPLES)
    var s_fused = SampleBuffer(SAMPLES)
    var prof = Profiler[False]()

    for ci in range(len(contexts)):
        var ctx = contexts[ci]
        if ctx > MAX_CONTEXT:
            continue
        runs[].runs[0].base_pos = Int32(ctx - 1)

        for _ in range(WARMUP):
            run_indexer(pools, buf, runs, 1, prof)
            run_attention(pools, buf, runs, 1, prof)
            run_attention_reuse(pools, buf, runs, 1, prof)
            keep(buf.output[0][0])

        s_idx.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            run_indexer(pools, buf, runs, 1, prof)
            var t1 = now_ns()
            s_idx.push(max_last_ts(pools) - t0, t1 - t0)
        keep(buf.block_idx[0][0])

        s_attn.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            run_attention(pools, buf, runs, 1, prof)
            var t1 = now_ns()
            s_attn.push(max_last_ts(pools) - t0, t1 - t0)
        keep(buf.output[0][0])

        s_reuse.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            run_attention_reuse(pools, buf, runs, 1, prof)
            var t1 = now_ns()
            s_reuse.push(max_last_ts(pools) - t0, t1 - t0)
        keep(buf.output[0][0])

        s_fused.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            run_indexer(pools, buf, runs, 1, prof)
            run_attention(pools, buf, runs, 1, prof)
            var t1 = now_ns()
            s_fused.push(max_last_ts(pools) - t0, t1 - t0)
        keep(buf.output[0][0])

        var idx_b = indexer_scan_bytes(1, ctx - 1)
        var attn_b = attn_read_bytes(1, ctx - 1)
        var attn_st = compute_stats(s_attn.kernel_ns, s_attn.n)
        var reuse_st = compute_stats(s_reuse.kernel_ns, s_reuse.n)
        print(t"ctx={ctx}")
        print_row("  indexer  ",
            compute_stats(s_idx.kernel_ns, s_idx.n),
            compute_stats(s_idx.wall_ns, s_idx.n), idx_b)
        print_row("  attention",
            attn_st, compute_stats(s_attn.wall_ns, s_attn.n), attn_b)
        print_row("  attn-reuse",
            reuse_st, compute_stats(s_reuse.wall_ns, s_reuse.n), attn_b)
        print(speedup_line(attn_st.p50, reuse_st.p50))
        print_row("  fused    ",
            compute_stats(s_fused.kernel_ns, s_fused.n),
            compute_stats(s_fused.wall_ns, s_fused.n), idx_b + attn_b)


def section_prefill_sweep[P: BurstThreadPool, o: ImmutOrigin, //](
    mut pools: List[P],
    buf: FusedBuffers[o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
):
    print("\n=== Prefill sweep (N queries from pos 0) ===")
    var seqs = [256, 512, 1024, 2048, 4096]
    var s_idx = SampleBuffer(SAMPLES)
    var s_attn = SampleBuffer(SAMPLES)
    var s_reuse = SampleBuffer(SAMPLES)
    var s_fused = SampleBuffer(SAMPLES)
    var prof = Profiler[False]()

    for si in range(len(seqs)):
        var n = seqs[si]
        if n > MAX_PREFILL:
            continue
        runs[].runs[0].base_pos = Int32(0)

        for _ in range(WARMUP):
            run_indexer(pools, buf, runs, n, prof)
            run_attention(pools, buf, runs, n, prof)
            run_attention_reuse(pools, buf, runs, n, prof)
            keep(buf.output[0][0])

        s_idx.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            run_indexer(pools, buf, runs, n, prof)
            var t1 = now_ns()
            s_idx.push(max_last_ts(pools) - t0, t1 - t0)
        keep(buf.block_idx[0][0])

        s_attn.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            run_attention(pools, buf, runs, n, prof)
            var t1 = now_ns()
            s_attn.push(max_last_ts(pools) - t0, t1 - t0)
        keep(buf.output[0][0])

        s_reuse.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            run_attention_reuse(pools, buf, runs, n, prof)
            var t1 = now_ns()
            s_reuse.push(max_last_ts(pools) - t0, t1 - t0)
        keep(buf.output[0][0])

        s_fused.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            run_indexer(pools, buf, runs, n, prof)
            run_attention(pools, buf, runs, n, prof)
            var t1 = now_ns()
            s_fused.push(max_last_ts(pools) - t0, t1 - t0)
        keep(buf.output[0][0])

        var idx_b = indexer_scan_bytes(n, 0)
        var attn_b = attn_read_bytes(n, 0)
        var attn_st = compute_stats(s_attn.kernel_ns, s_attn.n)
        var reuse_st = compute_stats(s_reuse.kernel_ns, s_reuse.n)
        print(t"seq={n}")
        print_row("  indexer  ",
            compute_stats(s_idx.kernel_ns, s_idx.n),
            compute_stats(s_idx.wall_ns, s_idx.n), idx_b)
        print_row("  attention",
            attn_st, compute_stats(s_attn.wall_ns, s_attn.n), attn_b)
        print_row("  attn-reuse",
            reuse_st, compute_stats(s_reuse.wall_ns, s_reuse.n), attn_b)
        print(speedup_line(attn_st.p50, reuse_st.p50))
        print_row("  fused    ",
            compute_stats(s_fused.kernel_ns, s_fused.n),
            compute_stats(s_fused.wall_ns, s_fused.n), idx_b + attn_b)


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

    var index_q_ptr = arena_alloc_all[BFloat16](arenas, MAX_PREFILL * INDEX_Q_ROW)
    var index_k_ptr = arena_alloc_all[BFloat16](
        arenas, MAX_CONTEXT * M3_INDEX_HEAD_DIM)
    var block_idx_ptr = arena_alloc_all[Int32](
        arenas, MAX_PREFILL * M3_INDEX_NUM_HEADS * M3_INDEX_TOPK_BLOCKS)
    var index_partial_ptr = arena_alloc_all[Float32](arenas, INDEX_PARTIAL_ELEMS)

    var q_ptr = arena_alloc_all[BFloat16](arenas, MAX_PREFILL * ATTN_Q_ROW)
    var k_ptr = arena_alloc_all[BFloat16](arenas, MAX_CONTEXT * M3_KV_DIM)
    var v_ptr = arena_alloc_all[BFloat16](arenas, MAX_CONTEXT * M3_KV_DIM)
    var output_ptr = arena_alloc_all[BFloat16](arenas, MAX_PREFILL * ATTN_Q_ROW)
    var attn_partial_ptr = arena_alloc_all[Float32](
        arenas, MAX_PREFILL * ATTN_PARTIAL_STRIDE)
    var segment_ptr = arena_alloc_all[MergeSegment](arenas, MAX_WORKERS * tp)

    for r in range(tp):
        fill_pattern(view.bind(index_q_ptr)[r], MAX_PREFILL * INDEX_Q_ROW)
        fill_pattern(view.bind(index_k_ptr)[r], MAX_CONTEXT * M3_INDEX_HEAD_DIM)
        fill_pattern(view.bind(q_ptr)[r], MAX_PREFILL * ATTN_Q_ROW)
        fill_pattern(view.bind(k_ptr)[r], MAX_CONTEXT * M3_KV_DIM)
        fill_pattern(view.bind(v_ptr)[r], MAX_CONTEXT * M3_KV_DIM)
        _ = arenas[r].prefault(0, arenas[r].used())

    var buf = FusedBuffers(
        view.bind(index_q_ptr), view.bind(index_k_ptr),
        view.bind(block_idx_ptr), view.bind(index_partial_ptr),
        view.bind(q_ptr), view.bind(k_ptr), view.bind(v_ptr),
        view.bind(output_ptr), view.bind(attn_partial_ptr),
        view.bind(segment_ptr))

    var cap = pools[0].get_capacity()
    print(
        t"attn: heads={M3_NUM_HEADS} kv_heads={M3_NUM_KV_HEADS} "
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

    section_validation(pools, buf, runs)
    section_parity(pools, buf, runs)
    section_decode_sweep(pools, buf, runs)
    section_prefill_sweep(pools, buf, runs)


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    var iso = len(topo.isolated_cpus)
    print("MiniMax-M3 fused indexer + sparse attention benchmark")
    print(t"{tp} NUMA node(s), {iso} isolated cpus")

    comptime ARENA_BYTES = 1536 * 1024 * 1024
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
