from std.math import exp
from std.memory import Span, UnsafePointer, memset_zero

from numa import NumaArena
from threading import BurstPool
from threading.threading_traits import BurstThreadPool
from kernels.attention_ops import KVRunTable, flash_partial_stride, pow2_shift
from kernels.helpers import Binding, RankView
from kernels.profiling import Profiler
from butterquant_kernels import dispatch_bq_attn_prep, dispatch_bq_sliding_attention
from simd_math.ops import sqrt


comptime ALIGNMENT = 64
comptime HEAD_DIM = 256
comptime GLOBAL_NUM_Q = 16
comptime GLOBAL_NUM_KV = 8
comptime GQA_RATIO = GLOBAL_NUM_Q // GLOBAL_NUM_KV
comptime GLOBAL_Q_STRIDE = GLOBAL_NUM_Q * HEAD_DIM
comptime GLOBAL_KV_STRIDE = GLOBAL_NUM_KV * HEAD_DIM
comptime PAGE_LEN = 64
comptime WINDOW = 64
comptime RING_PAGES = 2
comptime CACHE_SIZE = RING_PAGES * PAGE_LEN
comptime ROPE_HALF = HEAD_DIM // 2
comptime PAIR_STRIDE = HEAD_DIM // 2
comptime SQRT_HD = sqrt[DType.float32, 1](HEAD_DIM)
comptime HD_EPS = Float32(HEAD_DIM) * Float32(1e-6)
comptime NUM_PAGES = 6
comptime MAX_TOKENS = 128
comptime ROPE_ROWS = 512
comptime POOL_WORKERS = 4
comptime ARENA_BYTES = 128 * 1024 * 1024


def sequence_lengths() -> List[Int]:
    var lens = List[Int]()
    lens.append(150)
    lens.append(120)
    lens.append(200)
    return lens^


def sequence_pages(seq: Int) -> List[Int]:
    var pages = List[Int]()
    if seq == 0:
        pages.append(3)
        pages.append(1)
    elif seq == 1:
        pages.append(0)
        pages.append(4)
    else:
        pages.append(2)
        pages.append(5)
    return pages^


def content_value(key: Int, column: Int, salt: Int) -> Float32:
    var h = (UInt64(key) * 2654435761
             + UInt64(column) * 2246822519
             + UInt64(salt) * 3266489917)
    h ^= h >> 13
    h *= 1099511628211
    h ^= h >> 29
    return Float32(Int(h % 17) - 8) * Float32(0.05)


def arena_alloc[T: AnyType](
    mut arena: NumaArena[alignment=ALIGNMENT], count: Int,
) -> UnsafePointer[T, MutAnyOrigin]:
    var ptr = arena.alloc[T](count)
    if not ptr:
        print("arena alloc failed for", count, "elements")
        return UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
    return ptr.value()


def arena_alloc_all[T: AnyType](
    mut arenas: List[NumaArena[alignment=ALIGNMENT]], count: Int,
) -> UnsafePointer[T, MutAnyOrigin]:
    var first = UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
    for r in range(len(arenas)):
        var ptr = arena_alloc[T](arenas[r], count)
        if r == 0:
            first = ptr
    return first


def ring_slot(pos: Int, read pages: List[Int]) -> Int:
    return pages[(pos // PAGE_LEN) % RING_PAGES] * PAGE_LEN + pos % PAGE_LEN


def reference_head(
    degree: Int,
    head: Int,
    pos: Int,
    read k_ptrs: List[UnsafePointer[Int8, MutAnyOrigin]],
    read ks_ptrs: List[UnsafePointer[Float32, MutAnyOrigin]],
    read v_ptrs: List[UnsafePointer[Int8, MutAnyOrigin]],
    read vs_ptrs: List[UnsafePointer[Float32, MutAnyOrigin]],
    read pages: List[Int],
    q_row: UnsafePointer[Int8, MutAnyOrigin],
    qf: Float32,
    mut acc: List[Float32],
):
    var local_num_q = GLOBAL_NUM_Q // degree
    var local_num_kv = GLOBAL_NUM_KV // degree
    var kv_stride = local_num_kv * HEAD_DIM
    var rank = head // local_num_q
    var local_kv = (head % local_num_q) // GQA_RATIO
    comptime inv127 = Float32(1.0) / Float32(127.0)
    comptime inv127sq = inv127 * inv127
    var m = Float32(-1e30)
    var l = Float32(0)
    for j in range(HEAD_DIM):
        acc[j] = Float32(0)
    var lo = pos - WINDOW + 1
    if lo < 0:
        lo = 0
    for p in range(lo, pos + 1):
        var slot = ring_slot(p, pages)
        var k_head = k_ptrs[rank] + slot * kv_stride + local_kv * HEAD_DIM
        var dot = 0
        for j in range(HEAD_DIM):
            dot += Int(k_head[j]) * Int(q_row[j])
        var ks = ks_ptrs[rank][slot * local_num_kv + local_kv]
        var s = Float32(dot) * qf * ks * inv127sq
        var m_new = s if s > m else m
        var corr = exp(m - m_new)
        var w = exp(s - m_new)
        l = l * corr + w
        var vs = vs_ptrs[rank][slot * local_num_kv + local_kv]
        var v_head = v_ptrs[rank] + slot * kv_stride + local_kv * HEAD_DIM
        var vw = w * vs * inv127
        for j in range(HEAD_DIM):
            acc[j] = acc[j] * corr + Float32(Int(v_head[j])) * vw
        m = m_new
    if l > 0:
        var inv_l = Float32(1.0) / l
        for j in range(HEAD_DIM):
            acc[j] *= inv_l


def run_step[P: BurstThreadPool, o: ImmutOrigin, //](
    degree: Int,
    read token_seq: List[Int],
    read token_pos: List[Int],
    read run_buf: List[Int],
    read run_pos: List[Int],
    read run_seq: List[Int],
    q_src: Binding[BFloat16, o],
    k_src: Binding[BFloat16, o],
    v_src: Binding[BFloat16, o],
    q_norm: Binding[BFloat16, o],
    k_norm: Binding[BFloat16, o],
    q_i8: Binding[Int8, o],
    qi_bias: Binding[Float32, o],
    f_q: Binding[Float32, o],
    k_cache: Binding[Int8, o],
    k_scale: Binding[Float32, o],
    v_cache: Binding[Int8, o],
    v_scale: Binding[Float32, o],
    cos_table: Binding[Float32, o],
    sin_table: Binding[Float32, o],
    attn_out: Binding[BFloat16, o],
    partials: Binding[Float32, o],
    read k_ptrs: List[UnsafePointer[Int8, MutAnyOrigin]],
    read ks_ptrs: List[UnsafePointer[Float32, MutAnyOrigin]],
    read v_ptrs: List[UnsafePointer[Int8, MutAnyOrigin]],
    read vs_ptrs: List[UnsafePointer[Float32, MutAnyOrigin]],
    mut pools: List[P],
    mut out: List[Float32],
):
    var n_tokens = len(token_seq)
    var local_num_q = GLOBAL_NUM_Q // degree
    var local_num_kv = GLOBAL_NUM_KV // degree
    var q_stride = local_num_q * HEAD_DIM
    var kv_stride = local_num_kv * HEAD_DIM
    var partial_stride = flash_partial_stride(local_num_q, HEAD_DIM)

    for r in range(degree):
        for t in range(n_tokens):
            var key = token_seq[t] * 512 + token_pos[t]
            for c in range(q_stride):
                q_src[r][t * q_stride + c] = content_value(
                    key, r * q_stride + c, 1).cast[DType.bfloat16]()
            for c in range(kv_stride):
                k_src[r][t * kv_stride + c] = content_value(
                    key, r * kv_stride + c, 2).cast[DType.bfloat16]()
                v_src[r][t * kv_stride + c] = content_value(
                    key, r * kv_stride + c, 3).cast[DType.bfloat16]()

    var table = KVRunTable()
    for i in range(len(run_buf)):
        table.begin_run(run_buf[i], run_pos[i])
        var pages = sequence_pages(run_seq[i])
        for k in range(RING_PAGES):
            table.add_base_row(Int32(pages[k] * PAGE_LEN))
    var runs_ptr = UnsafePointer(to=table)

    var prof = Profiler[False]()
    dispatch_bq_attn_prep[
        head_dim=HEAD_DIM, rope_half=ROPE_HALF, pair_stride=PAIR_STRIDE,
        sqrt_n=SQRT_HD, n_eps=HD_EPS,
        q_scale=Float32(1.0) / SQRT_HD, norm_v=False,
        max_worker_count=POOL_WORKERS,
    ](q_src, k_src, v_src, q_norm, k_norm, q_i8, qi_bias, f_q,
      k_cache, k_scale, v_cache, v_scale, cos_table, sin_table,
      runs_ptr, local_num_q, local_num_kv, 1,
      pow2_shift(PAGE_LEN), PAGE_LEN - 1, RING_PAGES - 1,
      n_tokens, pools, prof)

    dispatch_bq_sliding_attention[
        head_dim=HEAD_DIM, max_q=GLOBAL_NUM_Q, gqa_ratio=GQA_RATIO,
        window=WINDOW, cache_size=CACHE_SIZE, page_len=PAGE_LEN,
        max_worker_count=POOL_WORKERS,
    ](q_i8, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
      attn_out, partials, runs_ptr,
      local_num_q, local_num_kv, partial_stride, kv_stride,
      n_tokens, pools, prof)

    var step_out_base = len(out)
    for t in range(n_tokens):
        for gh in range(GLOBAL_NUM_Q):
            var rank = gh // local_num_q
            var lh = gh % local_num_q
            var base = attn_out[rank]
            for j in range(HEAD_DIM):
                out.append(base[
                    t * q_stride + lh * HEAD_DIM + j,
                ].cast[DType.float32]())

    var acc = List[Float32](length=HEAD_DIM, fill=Float32(0))
    var step_worst = Float32(0)
    var bad_tokens = 0
    for t in range(n_tokens):
        var run = 0
        for i in range(len(run_buf)):
            if run_buf[i] <= t:
                run = i
        var abs_pos = run_pos[run] + (t - run_buf[run])
        var pages = sequence_pages(run_seq[run])
        var token_worst = Float32(0)
        var token_head = -1
        for h in range(GLOBAL_NUM_Q):
            var rank = h // local_num_q
            var lh = h % local_num_q
            var q_row = q_i8[rank] + t * q_stride + lh * HEAD_DIM
            var qf = f_q[rank][t * local_num_q + lh]
            reference_head(
                degree, h, abs_pos,
                k_ptrs, ks_ptrs, v_ptrs, vs_ptrs, pages, q_row, qf, acc)
            for j in range(HEAD_DIM):
                var d = out[
                    step_out_base + (t * GLOBAL_NUM_Q + h) * HEAD_DIM + j,
                ] - acc[j]
                if d < 0:
                    d = -d
                if d > token_worst:
                    token_worst = d
                    token_head = h
        if token_worst > Float32(0.05):
            bad_tokens += 1
            if bad_tokens <= 8:
                print(
                    t"    ref mismatch token {t} (seq {run_seq[run]} "
                    t"pos {abs_pos}): diff={token_worst} head {token_head}")
        if token_worst > step_worst:
            step_worst = token_worst
    print(
        t"  reference check: max_diff={step_worst} "
        t"| {bad_tokens} bad tokens of {n_tokens}")


def append_run(
    mut token_seq: List[Int], mut token_pos: List[Int],
    mut run_buf: List[Int], mut run_pos: List[Int], mut run_seq: List[Int],
    seq: Int, start: Int, count: Int,
):
    run_buf.append(len(token_seq))
    run_pos.append(start)
    run_seq.append(seq)
    for p in range(start, start + count):
        token_seq.append(seq)
        token_pos.append(p)


def run_config(degree: Int) -> List[Float32]:
    var local_num_q = GLOBAL_NUM_Q // degree
    var local_num_kv = GLOBAL_NUM_KV // degree
    var q_stride = local_num_q * HEAD_DIM
    var kv_stride = local_num_kv * HEAD_DIM
    var cache_rows = NUM_PAGES * PAGE_LEN
    var partial_stride = flash_partial_stride(local_num_q, HEAD_DIM)

    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=degree)
    for _ in range(degree):
        arenas.append(NumaArena[alignment=ALIGNMENT](0, ARENA_BYTES))
        if not arenas[len(arenas) - 1]:
            print("arena allocation failed")
            return List[Float32]()
    var pools = List[BurstPool[]](capacity=degree)
    for _ in range(degree):
        pools.append(BurstPool[](POOL_WORKERS))

    var bases = List[Int](capacity=degree)
    for r in range(degree):
        bases.append(Int(arenas[r].base.value()))
    var view = RankView(Span(bases))

    var q_src = view.bind(arena_alloc_all[BFloat16](
        arenas, MAX_TOKENS * GLOBAL_Q_STRIDE))
    var k_src = view.bind(arena_alloc_all[BFloat16](
        arenas, MAX_TOKENS * GLOBAL_KV_STRIDE))
    var v_src = view.bind(arena_alloc_all[BFloat16](
        arenas, MAX_TOKENS * GLOBAL_KV_STRIDE))
    var q_norm = view.bind(arena_alloc_all[BFloat16](arenas, HEAD_DIM))
    var k_norm = view.bind(arena_alloc_all[BFloat16](arenas, HEAD_DIM))
    var q_i8 = view.bind(arena_alloc_all[Int8](
        arenas, MAX_TOKENS * GLOBAL_Q_STRIDE))
    var qi_bias = view.bind(arena_alloc_all[Float32](
        arenas, MAX_TOKENS * GLOBAL_NUM_Q))
    var f_q = view.bind(arena_alloc_all[Float32](
        arenas, MAX_TOKENS * GLOBAL_NUM_Q))
    var k_cache = view.bind(arena_alloc_all[Int8](
        arenas, cache_rows * GLOBAL_KV_STRIDE))
    var k_scale = view.bind(arena_alloc_all[Float32](
        arenas, cache_rows * GLOBAL_NUM_KV))
    var v_cache = view.bind(arena_alloc_all[Int8](
        arenas, cache_rows * GLOBAL_KV_STRIDE))
    var v_scale = view.bind(arena_alloc_all[Float32](
        arenas, cache_rows * GLOBAL_NUM_KV))
    var cos_table = view.bind(arena_alloc_all[Float32](
        arenas, ROPE_ROWS * ROPE_HALF))
    var sin_table = view.bind(arena_alloc_all[Float32](
        arenas, ROPE_ROWS * ROPE_HALF))
    var attn_out = view.bind(arena_alloc_all[BFloat16](
        arenas, MAX_TOKENS * GLOBAL_Q_STRIDE))
    var partials = view.bind(arena_alloc_all[Float32](
        arenas, 64 * flash_partial_stride(GLOBAL_NUM_Q, HEAD_DIM)))

    for r in range(degree):
        for j in range(HEAD_DIM):
            q_norm[r][j] = Float32(1.0).cast[DType.bfloat16]()
            k_norm[r][j] = Float32(1.0).cast[DType.bfloat16]()
        for j in range(ROPE_ROWS * ROPE_HALF):
            cos_table[r][j] = Float32(1.0)
            sin_table[r][j] = Float32(0.0)
        memset_zero(k_cache[r], cache_rows * kv_stride)
        memset_zero(v_cache[r], cache_rows * kv_stride)
        memset_zero(k_scale[r], cache_rows * local_num_kv)
        memset_zero(v_scale[r], cache_rows * local_num_kv)
        _ = arenas[r].prefault(0, arenas[r].used())
    print(t"config degree={degree} ready")

    var k_ptrs = List[UnsafePointer[Int8, MutAnyOrigin]]()
    var ks_ptrs = List[UnsafePointer[Float32, MutAnyOrigin]]()
    var v_ptrs = List[UnsafePointer[Int8, MutAnyOrigin]]()
    var vs_ptrs = List[UnsafePointer[Float32, MutAnyOrigin]]()
    for r in range(degree):
        k_ptrs.append(k_cache[r])
        ks_ptrs.append(k_scale[r])
        v_ptrs.append(v_cache[r])
        vs_ptrs.append(v_scale[r])

    var out = List[Float32]()
    for step in range(10):
        var token_seq = List[Int]()
        var token_pos = List[Int]()
        var run_buf = List[Int]()
        var run_pos = List[Int]()
        var run_seq = List[Int]()
        if step == 0:
            append_run(token_seq, token_pos, run_buf, run_pos, run_seq,
                       0, 0, 64)
        elif step == 1:
            append_run(token_seq, token_pos, run_buf, run_pos, run_seq,
                       0, 64, 64)
        elif step == 2:
            append_run(token_seq, token_pos, run_buf, run_pos, run_seq,
                       0, 128, 22)
            append_run(token_seq, token_pos, run_buf, run_pos, run_seq,
                       1, 0, 42)
        elif step == 3:
            append_run(token_seq, token_pos, run_buf, run_pos, run_seq,
                       1, 42, 64)
        elif step == 4:
            append_run(token_seq, token_pos, run_buf, run_pos, run_seq,
                       1, 106, 14)
            append_run(token_seq, token_pos, run_buf, run_pos, run_seq,
                       2, 0, 50)
        elif step == 5:
            append_run(token_seq, token_pos, run_buf, run_pos, run_seq,
                       2, 50, 64)
        elif step == 6:
            append_run(token_seq, token_pos, run_buf, run_pos, run_seq,
                       2, 114, 64)
        elif step == 7:
            append_run(token_seq, token_pos, run_buf, run_pos, run_seq,
                       0, 150, 1)
            append_run(token_seq, token_pos, run_buf, run_pos, run_seq,
                       1, 120, 1)
            append_run(token_seq, token_pos, run_buf, run_pos, run_seq,
                       2, 178, 22)
        elif step == 8:
            append_run(token_seq, token_pos, run_buf, run_pos, run_seq,
                       0, 151, 1)
            append_run(token_seq, token_pos, run_buf, run_pos, run_seq,
                       1, 121, 1)
            append_run(token_seq, token_pos, run_buf, run_pos, run_seq,
                       2, 200, 1)
        else:
            append_run(token_seq, token_pos, run_buf, run_pos, run_seq,
                       0, 152, 1)
        var n = len(token_seq)
        print(t"  step {step}: {n} tokens")
        run_step(
            degree, token_seq, token_pos, run_buf, run_pos, run_seq,
            q_src, k_src, v_src, q_norm, k_norm, q_i8, qi_bias, f_q,
            k_cache, k_scale, v_cache, v_scale, cos_table, sin_table,
            attn_out, partials,
            k_ptrs, ks_ptrs, v_ptrs, vs_ptrs, pools, out)

    var arena_keepalive = 0
    for r in range(degree):
        arena_keepalive += arenas[r].used()
    if arena_keepalive < 0:
        print("unreachable", arena_keepalive)

    return out^


def main():
    print("sliding attention degree parity")
    var baseline = run_config(1)
    for d in range(2, 5, 2):
        print(t"=== degree {d} vs degree 1 ===")
        var test = run_config(d)
        if len(baseline) == 0 or len(baseline) != len(test):
            var n_ref = len(baseline)
            var n_test = len(test)
            print(t"output size mismatch: {n_ref} vs {n_test}")
            continue
        var worst = Float32(0)
        var bad = 0
        for i in range(len(baseline)):
            var d_abs = baseline[i] - test[i]
            if d_abs < 0:
                d_abs = -d_abs
            if d_abs > worst:
                worst = d_abs
            if d_abs > Float32(0.05):
                bad += 1
        var verdict = String("OK") if bad == 0 else String("FAIL")
        print(t"parity: max_diff={worst} | {bad} bad elements | {verdict}")
