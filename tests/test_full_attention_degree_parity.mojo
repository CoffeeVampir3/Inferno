from std.math import exp
from std.memory import Span, UnsafePointer, memset_zero

from numa import NumaArena
from threading import BurstPool
from threading.threading_traits import BurstThreadPool
from kernels.attention_ops import KVRunTable, flash_partial_stride, pow2_shift
from kernels.helpers import Binding, RankView
from kernels.logsum_merge import MergeSegment, write_finalized_head
from kernels.profiling import Profiler
from butterquant_kernels import dispatch_bq_attn_prep, dispatch_bq_full_attention
from simd_math.ops import sqrt


comptime ALIGNMENT = 64
comptime HEAD_DIM = 512
comptime NUM_Q = 16
comptime NUM_KV = 2
comptime GQA_RATIO = NUM_Q // NUM_KV
comptime KV_STRIDE = NUM_KV * HEAD_DIM
comptime Q_STRIDE = NUM_Q * HEAD_DIM
comptime PAGE_LEN = 64
comptime ROPE_HALF = 64
comptime PAIR_STRIDE = HEAD_DIM // 2
comptime SQRT_HD = sqrt[DType.float32, 1](HEAD_DIM)
comptime HD_EPS = Float32(HEAD_DIM) * Float32(1e-6)
comptime PSTRIDE = flash_partial_stride(NUM_Q, HEAD_DIM)
comptime NUM_PAGES = 12
comptime MAX_TOKENS = 512
comptime ROPE_ROWS = 512
comptime POOL_WORKERS = 4
comptime ARENA_BYTES = 256 * 1024 * 1024


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
        pages.append(7)
        pages.append(11)
    elif seq == 1:
        pages.append(0)
        pages.append(5)
        pages.append(9)
    else:
        pages.append(2)
        pages.append(4)
        pages.append(6)
        pages.append(8)
        pages.append(10)
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


def run_step[P: BurstThreadPool, o: ImmutOrigin, //](
    degree: Int,
    rows_per_page: Int,
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
    q_local: Binding[BFloat16, o],
    partials: Binding[Float32, o],
    segments: Binding[MergeSegment, o],
    read k_ptrs: List[UnsafePointer[Int8, MutAnyOrigin]],
    read ks_ptrs: List[UnsafePointer[Float32, MutAnyOrigin]],
    read v_ptrs: List[UnsafePointer[Int8, MutAnyOrigin]],
    read vs_ptrs: List[UnsafePointer[Float32, MutAnyOrigin]],
    mut pools: List[P],
    mut out: List[Float32],
):
    var n_tokens = len(token_seq)
    var local_num_q = NUM_Q // degree
    var page_shift = pow2_shift(rows_per_page)
    var row_mask = rows_per_page - 1
    print(t"  step n_tokens={n_tokens} filling sources")

    for r in range(degree):
        for t in range(n_tokens):
            var key = token_seq[t] * 512 + token_pos[t]
            for c in range(Q_STRIDE):
                q_src[r][t * Q_STRIDE + c] = content_value(
                    key, c, 1).cast[DType.bfloat16]()
            for c in range(KV_STRIDE):
                k_src[r][t * KV_STRIDE + c] = content_value(
                    key, c, 2).cast[DType.bfloat16]()
                v_src[r][t * KV_STRIDE + c] = content_value(
                    key, c, 3).cast[DType.bfloat16]()

    var table = KVRunTable()
    for i in range(len(run_buf)):
        table.begin_run(run_buf[i], run_pos[i])
        var pages = sequence_pages(run_seq[i])
        for k in range(len(pages)):
            table.add_base_row(Int32(pages[k] * rows_per_page))
    var runs_ptr = UnsafePointer(to=table)
    print("  sources filled, dispatching prep")

    var prof = Profiler[False]()
    dispatch_bq_attn_prep[
        head_dim=HEAD_DIM, rope_half=ROPE_HALF, pair_stride=PAIR_STRIDE,
        sqrt_n=SQRT_HD, n_eps=HD_EPS,
        max_worker_count=POOL_WORKERS,
    ](q_src, k_src, v_src, q_norm, k_norm, q_i8, qi_bias, f_q,
      k_cache, k_scale, v_cache, v_scale, cos_table, sin_table,
      runs_ptr, NUM_Q, NUM_KV, degree,
      page_shift, row_mask, -1, n_tokens, pools, prof)

    print("  prep done, dispatching attention")
    dispatch_bq_full_attention[
        head_dim=HEAD_DIM, num_q=NUM_Q, num_kv=NUM_KV, gqa_ratio=GQA_RATIO,
        kv_stride=KV_STRIDE, partial_stride=PSTRIDE, page_len=PAGE_LEN,
        max_worker_count=POOL_WORKERS,
    ](q_i8, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
      q_local, partials, segments, runs_ptr,
      local_num_q, n_tokens, pools, prof)

    print("  attention done, harvesting")
    var step_out_base = len(out)
    for t in range(n_tokens):
        for gh in range(NUM_Q):
            var rank = gh // local_num_q
            var lh = gh % local_num_q
            var base = q_local[rank]
            for j in range(HEAD_DIM):
                out.append(base[
                    t * local_num_q * HEAD_DIM + lh * HEAD_DIM + j,
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
        for h in range(NUM_Q):
            var q_row = q_i8[0] + t * Q_STRIDE + h * HEAD_DIM
            var qf = f_q[0][t * NUM_Q + h]
            _ = reference_head(
                degree, rows_per_page, h, abs_pos, -1, False,
                q_row, qf, k_ptrs, ks_ptrs, v_ptrs, vs_ptrs, pages, acc)
            for j in range(HEAD_DIM):
                var d = out[
                    step_out_base + (t * NUM_Q + h) * HEAD_DIM + j] - acc[j]
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
                var probe_q = q_i8[0] + t * Q_STRIDE + token_head * HEAD_DIM
                var probe_f = f_q[0][t * NUM_Q + token_head]
                var variant_line = String("      variants:")
                for variant in range(3):
                    var vpos = abs_pos
                    var drop = False
                    if variant == 1:
                        drop = True
                    elif variant == 2:
                        vpos = abs_pos + 1
                    _ = reference_head(
                        degree, rows_per_page, token_head, vpos, -1, drop,
                        probe_q, probe_f, k_ptrs, ks_ptrs, v_ptrs, vs_ptrs,
                        pages, acc)
                    var vw = Float32(0)
                    for j in range(HEAD_DIM):
                        var vd = out[
                            step_out_base
                            + (t * NUM_Q + token_head) * HEAD_DIM + j] - acc[j]
                        if vd < 0:
                            vd = -vd
                        if vd > vw:
                            vw = vd
                    if variant == 0:
                        variant_line += String(t" full={vw}")
                    elif variant == 1:
                        variant_line += String(t" drop_own={vw}")
                    else:
                        variant_line += String(t" extra_next={vw}")
                print(variant_line)
                var kv_h = token_head // GQA_RATIO
                comptime inv127sq = Float32(1.0) / Float32(127.0 * 127.0)
                for r in range(degree):
                    var sp = partials[r] + t * PSTRIDE
                    var mv = sp[NUM_Q * HEAD_DIM + token_head]
                    var lv = sp[NUM_Q * HEAD_DIM + NUM_Q + token_head]
                    var score_line = String(
                        t"      rank {r} kernel m={mv} l={lv} scores:")
                    for p in range(abs_pos + 3):
                        if p % degree != r:
                            continue
                        var local_row = p // degree
                        var slot = pages[local_row // rows_per_page] \
                            * rows_per_page + (local_row % rows_per_page)
                        var k_head = k_ptrs[r] + slot * KV_STRIDE \
                            + kv_h * HEAD_DIM
                        var idot = 0
                        for j in range(HEAD_DIM):
                            idot += Int(k_head[j]) * Int(probe_q[j])
                        var sc = (Float32(idot) * probe_f
                                  * ks_ptrs[r][slot * NUM_KV + kv_h]
                                  * inv127sq)
                        score_line += String(t" p{p}={sc}")
                    print(score_line)
        if token_worst > step_worst:
            step_worst = token_worst
    print(
        t"  reference check: max_diff={step_worst} "
        t"| {bad_tokens} bad tokens of {n_tokens}")


def reference_head(
    degree: Int,
    rows_per_page: Int,
    head: Int,
    pos: Int,
    keep_parity: Int,
    drop_last: Bool,
    q_row: UnsafePointer[Int8, MutAnyOrigin],
    qf: Float32,
    read k_ptrs: List[UnsafePointer[Int8, MutAnyOrigin]],
    read ks_ptrs: List[UnsafePointer[Float32, MutAnyOrigin]],
    read v_ptrs: List[UnsafePointer[Int8, MutAnyOrigin]],
    read vs_ptrs: List[UnsafePointer[Float32, MutAnyOrigin]],
    read pages: List[Int],
    mut acc: List[Float32],
) -> Float32:
    var kv_h = head // GQA_RATIO
    comptime inv127 = Float32(1.0) / Float32(127.0)
    comptime inv127sq = inv127 * inv127
    var m = Float32(-1e30)
    var l = Float32(0)
    for j in range(HEAD_DIM):
        acc[j] = Float32(0)
    var last = pos - 1 if drop_last else pos
    for p in range(last + 1):
        if keep_parity >= 0 and p % degree != keep_parity:
            continue
        var rank = p % degree
        var local_row = p // degree
        var slot = pages[local_row // rows_per_page] * rows_per_page + (
            local_row % rows_per_page)
        var k_head = k_ptrs[rank] + slot * KV_STRIDE + kv_h * HEAD_DIM
        var dot = 0
        for j in range(HEAD_DIM):
            dot += Int(k_head[j]) * Int(q_row[j])
        var ks = ks_ptrs[rank][slot * NUM_KV + kv_h]
        var s = Float32(dot) * qf * ks * inv127sq
        var m_new = s if s > m else m
        var corr = exp(m - m_new)
        var w = exp(s - m_new)
        l = l * corr + w
        var vs = vs_ptrs[rank][slot * NUM_KV + kv_h]
        var v_head = v_ptrs[rank] + slot * KV_STRIDE + kv_h * HEAD_DIM
        var vw = w * vs * inv127
        for j in range(HEAD_DIM):
            acc[j] = acc[j] * corr + Float32(Int(v_head[j])) * vw
        m = m_new
    if l > 0:
        var inv_l = Float32(1.0) / l
        for j in range(HEAD_DIM):
            acc[j] *= inv_l
    return l



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
    var rows_per_page = PAGE_LEN // degree
    var cache_rows = NUM_PAGES * rows_per_page
    var local_num_q = NUM_Q // degree

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
        arenas, MAX_TOKENS * Q_STRIDE))
    var k_src = view.bind(arena_alloc_all[BFloat16](
        arenas, MAX_TOKENS * KV_STRIDE))
    var v_src = view.bind(arena_alloc_all[BFloat16](
        arenas, MAX_TOKENS * KV_STRIDE))
    var q_norm = view.bind(arena_alloc_all[BFloat16](arenas, HEAD_DIM))
    var k_norm = view.bind(arena_alloc_all[BFloat16](arenas, HEAD_DIM))
    var q_i8 = view.bind(arena_alloc_all[Int8](
        arenas, MAX_TOKENS * Q_STRIDE))
    var qi_bias = view.bind(arena_alloc_all[Float32](
        arenas, MAX_TOKENS * NUM_Q))
    var f_q = view.bind(arena_alloc_all[Float32](
        arenas, MAX_TOKENS * NUM_Q))
    var k_cache = view.bind(arena_alloc_all[Int8](
        arenas, cache_rows * KV_STRIDE))
    var k_scale = view.bind(arena_alloc_all[Float32](
        arenas, cache_rows * NUM_KV))
    var v_cache = view.bind(arena_alloc_all[Int8](
        arenas, cache_rows * KV_STRIDE))
    var v_scale = view.bind(arena_alloc_all[Float32](
        arenas, cache_rows * NUM_KV))
    var cos_table = view.bind(arena_alloc_all[Float32](
        arenas, ROPE_ROWS * ROPE_HALF))
    var sin_table = view.bind(arena_alloc_all[Float32](
        arenas, ROPE_ROWS * ROPE_HALF))
    var q_local = view.bind(arena_alloc_all[BFloat16](
        arenas, MAX_TOKENS * local_num_q * HEAD_DIM))
    var partials = view.bind(arena_alloc_all[Float32](
        arenas, (MAX_TOKENS + 8) * PSTRIDE))
    var segments = view.bind(arena_alloc_all[MergeSegment](arenas, 256))

    for r in range(degree):
        for j in range(HEAD_DIM):
            q_norm[r][j] = Float32(1.0).cast[DType.bfloat16]()
            k_norm[r][j] = Float32(1.0).cast[DType.bfloat16]()
        for j in range(ROPE_ROWS * ROPE_HALF):
            cos_table[r][j] = Float32(1.0)
            sin_table[r][j] = Float32(0.0)
        memset_zero(k_cache[r], cache_rows * KV_STRIDE)
        memset_zero(v_cache[r], cache_rows * KV_STRIDE)
        memset_zero(k_scale[r], cache_rows * NUM_KV)
        memset_zero(v_scale[r], cache_rows * NUM_KV)
        _ = arenas[r].prefault(0, arenas[r].used())
    print(t"config degree={degree} buffers ready")

    var lens = sequence_lengths()
    var out = List[Float32]()

    var k_ptrs = List[UnsafePointer[Int8, MutAnyOrigin]]()
    var ks_ptrs = List[UnsafePointer[Float32, MutAnyOrigin]]()
    var v_ptrs = List[UnsafePointer[Int8, MutAnyOrigin]]()
    var vs_ptrs = List[UnsafePointer[Float32, MutAnyOrigin]]()
    for r in range(degree):
        k_ptrs.append(k_cache[r])
        ks_ptrs.append(k_scale[r])
        v_ptrs.append(v_cache[r])
        vs_ptrs.append(v_scale[r])

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
            degree, rows_per_page, token_seq, token_pos,
            run_buf, run_pos, run_seq,
            q_src, k_src, v_src, q_norm, k_norm, q_i8, qi_bias, f_q,
            k_cache, k_scale, v_cache, v_scale, cos_table, sin_table,
            q_local, partials, segments,
            k_ptrs, ks_ptrs, v_ptrs, vs_ptrs, pools, out)

    var arena_keepalive = 0
    for r in range(degree):
        arena_keepalive += arenas[r].used()
    if arena_keepalive < 0:
        print("unreachable", arena_keepalive)

    return out^


def main():
    print("full attention degree parity")
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
