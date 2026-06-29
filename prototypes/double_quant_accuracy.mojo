from std.math import max, sqrt, log, cos

from simd_math.ops import quantize_i8, roundeven
from butterquant.fwht import fwht_row


comptime PtrF32 = UnsafePointer[Float32, MutAnyOrigin]
comptime FWHT_BLOCK = 128
comptime TWO_PI = Float32(6.283185307179586)


struct Rng(Copyable, Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next_u64(mut self) -> UInt64:
        self.state = self.state + 0x9E3779B97F4A7C15
        var z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    def uniform(mut self) -> Float32:
        var u = (self.next_u64() >> 11).cast[DType.float64]() * (
            1.0 / 9007199254740992.0
        )
        return Float32(u)

    def gauss(mut self) -> Float32:
        var u1 = max(self.uniform(), Float32(1e-7))
        var u2 = self.uniform()
        return sqrt(Float32(-2.0) * log(u1)) * cos(TWO_PI * u2)

    def pareto(mut self, alpha: Float32) -> Float32:
        var u = max(Float32(1.0) - self.uniform(), Float32(1e-7))
        return u ** (Float32(-1.0) / alpha)


def absmax_block(p: PtrF32, off: Int, n: Int) -> Float32:
    var m = Float32(0)
    for k in range(n):
        m = max(m, abs((p + off + k).load()))
    return m


def quant_q8_into(src: PtrF32, dst: PtrF32, cols: Int, block: Int):
    var nb = cols // block
    for b in range(nb):
        var off = b * block
        var amax = absmax_block(src, off, block)
        var scale = amax / Float32(127.0)
        var inv = (Float32(127.0) / amax) if amax > 0 else Float32(0)
        for k in range(block):
            var q = quantize_i8[1]((src + off + k).load(), inv)
            (dst + off + k).store(q.cast[DType.float32]() * scale)


def quant_q4_into(src: PtrF32, dst: PtrF32, cols: Int, block: Int):
    var nb = cols // block
    for b in range(nb):
        var off = b * block
        var amax = absmax_block(src, off, block)
        var scale = amax / Float32(7.0)
        var inv = (Float32(7.0) / amax) if amax > 0 else Float32(0)
        for k in range(block):
            var qf = roundeven[DType.float32, 1]((src + off + k).load() * inv)
            var qc = max(min(qf, Float32(7.0)), Float32(-8.0))
            (dst + off + k).store(qc * scale)


def copy_into(src: PtrF32, dst: PtrF32, n: Int):
    for i in range(n):
        (dst + i).store((src + i).load())


def matmul(x: PtrF32, w: PtrF32, y: PtrF32, M: Int, N: Int, K: Int):
    for m in range(M):
        for n in range(N):
            var acc = Float32(0)
            for k in range(K):
                acc += (x + m * K + k).load() * (w + n * K + k).load()
            (y + m * N + n).store(acc)


struct ErrStats(Copyable, Movable):
    var fro: Float32
    var col_mean: Float32
    var col_p99: Float32
    var col_max: Float32

    def __init__(
        out self,
        fro: Float32,
        col_mean: Float32,
        col_p99: Float32,
        col_max: Float32,
    ):
        self.fro = fro
        self.col_mean = col_mean
        self.col_p99 = col_p99
        self.col_max = col_max


def eval_error(
    y: PtrF32, gold: PtrF32, M: Int, N: Int
) -> ErrStats:
    var num = Float32(0)
    var den = Float32(0)
    for i in range(M * N):
        var d = (y + i).load() - (gold + i).load()
        num += d * d
        den += (gold + i).load() * (gold + i).load()
    var fro = sqrt(num / max(den, Float32(1e-30)))

    var cols = List[Float32](length=N, fill=Float32(0))
    for n in range(N):
        var cn = Float32(0)
        var cd = Float32(0)
        for m in range(M):
            var d = (y + m * N + n).load() - (gold + m * N + n).load()
            cn += d * d
            cd += (gold + m * N + n).load() * (gold + m * N + n).load()
        cols[n] = sqrt(cn / max(cd, Float32(1e-30)))

    for i in range(1, N):
        var key = cols[i]
        var j = i - 1
        while j >= 0 and cols[j] > key:
            cols[j + 1] = cols[j]
            j -= 1
        cols[j + 1] = key

    var mean = Float32(0)
    for n in range(N):
        mean += cols[n]
    mean /= Float32(N)
    var p99_idx = Int(Float32(0.99) * Float32(N - 1))
    return ErrStats(fro, mean, cols[p99_idx], cols[N - 1])


def print_row(name: StaticString, e: ErrStats):
    print(
        t"  {name}  fro={e.fro}  col_mean={e.col_mean}  col_p99={e.col_p99}  col_max={e.col_max}"
    )


def run_case(
    do_fwht: Bool,
    q8_block_arg: Int,
    q4_block: Int,
    M: Int,
    N: Int,
    K: Int,
    x0: PtrF32,
    w0: PtrF32,
    yref: PtrF32,
):
    var q8_block = q8_block_arg if q8_block_arg > 0 else K

    var xrot = List[Float32](length=M * K, fill=Float32(0))
    var wrot = List[Float32](length=N * K, fill=Float32(0))
    var xp = xrot.unsafe_ptr().as_unsafe_any_origin()
    var wp = wrot.unsafe_ptr().as_unsafe_any_origin()
    copy_into(x0, xp, M * K)
    copy_into(w0, wp, N * K)

    if do_fwht:
        for m in range(M):
            fwht_row[FWHT_BLOCK](xp + m * K, K)
        for n in range(N):
            fwht_row[FWHT_BLOCK](wp + n * K, K)

    var wq8 = List[Float32](length=N * K, fill=Float32(0))
    var wq4 = List[Float32](length=N * K, fill=Float32(0))
    var wdq = List[Float32](length=N * K, fill=Float32(0))
    var q8p = wq8.unsafe_ptr().as_unsafe_any_origin()
    var q4p = wq4.unsafe_ptr().as_unsafe_any_origin()
    var dqp = wdq.unsafe_ptr().as_unsafe_any_origin()

    for n in range(N):
        quant_q8_into(wp + n * K, q8p + n * K, K, q8_block)
        quant_q4_into(wp + n * K, q4p + n * K, K, q4_block)
        quant_q8_into(q4p + n * K, dqp + n * K, K, q8_block)

    var y = List[Float32](length=M * N, fill=Float32(0))
    var yp = y.unsafe_ptr().as_unsafe_any_origin()

    var tag = "FWHT" if do_fwht else "raw "
    print(
        t"--- {tag}  q8_block={q8_block}  q4_block={q4_block} ---"
    )
    matmul(xp, q8p, yp, M, N, K)
    print_row("q8-direct (oracle-q8)", eval_error(yp, yref, M, N))
    matmul(xp, q4p, yp, M, N, K)
    print_row("q4-direct            ", eval_error(yp, yref, M, N))
    matmul(xp, dqp, yp, M, N, K)
    print_row("q4->q8 (double-quant)", eval_error(yp, yref, M, N))


def main():
    var M = 32
    var N = 512
    var K = 2048
    var alpha = Float32(1.2)
    var outlier_frac = Float32(0.03)

    var rng = Rng(0xC0FFEE123)

    var chan = List[Float32](length=K, fill=Float32(1.0))
    for k in range(K):
        if rng.uniform() < outlier_frac:
            chan[k] = rng.pareto(alpha)

    var X = List[Float32](length=M * K, fill=Float32(0))
    var W = List[Float32](length=N * K, fill=Float32(0))
    for m in range(M):
        for k in range(K):
            X[m * K + k] = rng.gauss() * chan[k]
    for n in range(N):
        for k in range(K):
            W[n * K + k] = rng.gauss() * chan[k] * Float32(0.02)

    var xp = X.unsafe_ptr().as_unsafe_any_origin()
    var wp = W.unsafe_ptr().as_unsafe_any_origin()

    var yref = List[Float32](length=M * N, fill=Float32(0))
    var yrefp = yref.unsafe_ptr().as_unsafe_any_origin()
    matmul(xp, wp, yrefp, M, N, K)

    print(t"double-quant accuracy probe  M={M} N={N} K={K} alpha={alpha} outlier_frac={outlier_frac}")
    print("metric: relative error of X@W^T vs full-precision oracle (lower=better)")
    print("")

    run_case(False, 0, 64, M, N, K, xp, wp, yrefp)
    run_case(False, 128, 64, M, N, K, xp, wp, yrefp)
    run_case(True, 0, 64, M, N, K, xp, wp, yrefp)
    run_case(True, 128, 64, M, N, K, xp, wp, yrefp)
    run_case(True, 128, 32, M, N, K, xp, wp, yrefp)
