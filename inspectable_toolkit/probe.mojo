from std.algorithm import vectorize

from simd_math.ops import sqrt
from kernels.helpers import (
    BF16Ptr, F32Ptr, W, accumulate_scaled, scale_unrolled, copy_row,
)
from kernels.dot_products import dot_to_scalar


struct ContrastSet[cols: Int](Movable):
    var high: List[BFloat16]
    var low: List[BFloat16]
    var n_high: Int
    var n_low: Int
    var capacity: Int

    def __init__(out self, capacity: Int):
        self.capacity = capacity
        self.high = List[BFloat16](
            length=capacity * Self.cols, fill=BFloat16(0))
        self.low = List[BFloat16](
            length=capacity * Self.cols, fill=BFloat16(0))
        self.n_high = 0
        self.n_low = 0

    @always_inline
    def high_ptr(mut self, i: Int) -> BF16Ptr:
        return self.high.unsafe_ptr() + i * Self.cols

    @always_inline
    def low_ptr(mut self, i: Int) -> BF16Ptr:
        return self.low.unsafe_ptr() + i * Self.cols

    def add_row(mut self, is_high: Bool, src: BF16Ptr):
        if is_high:
            debug_assert(self.n_high < self.capacity, "high class overflow")
            copy_row[Self.cols](src, self.high_ptr(self.n_high))
            self.n_high += 1
        else:
            debug_assert(self.n_low < self.capacity, "low class overflow")
            copy_row[Self.cols](src, self.low_ptr(self.n_low))
            self.n_low += 1


@always_inline
def zero_f32[cols: Int](p: F32Ptr):
    def step[width: Int](idx: Int) {read}:
        (p + idx).store(SIMD[DType.float32, width](0))

    vectorize[W](cols, step)


def class_mean[cols: Int](rows: BF16Ptr, n: Int, mean: F32Ptr):
    zero_f32[cols](mean)
    if n <= 0:
        return
    for i in range(n):
        accumulate_scaled[cols](rows + i * cols, Float32(1), mean)
    scale_unrolled[cols](mean, Float32(1) / Float32(n))


def build_direction[cols: Int](
    mean_high: F32Ptr, mean_low: F32Ptr, direction: BF16Ptr,
) -> Float32:
    comptime assert cols % W == 0, "build_direction requires cols % W == 0"
    var acc_hh = SIMD[DType.float32, W](0)
    var acc_ll = SIMD[DType.float32, W](0)
    var acc_hl = SIMD[DType.float32, W](0)
    for off in range(0, cols, W):
        var h = (mean_high + off).load[width=W]()
        var l = (mean_low + off).load[width=W]()
        acc_hh = h.fma(h, acc_hh)
        acc_ll = l.fma(l, acc_ll)
        acc_hl = h.fma(l, acc_hl)
    var shh = acc_hh.reduce_add()
    var sll = acc_ll.reduce_add()
    var shl = acc_hl.reduce_add()
    var d_dot_mu = Float32(0.5) * (shh - sll)
    var mu_sq = Float32(0.25) * (shh + sll + Float32(2) * shl)
    var d_sq = shh + sll - Float32(2) * shl
    var beta = Float32(0)
    if mu_sq > Float32(0):
        beta = d_dot_mu / mu_sq
    var norm = sqrt[DType.float32, 1](d_sq - beta * d_dot_mu)
    if norm[0] <= Float32(0):
        return Float32(0)
    var inv = SIMD[DType.float32, W](Float32(1) / norm[0])
    var hi_scale = SIMD[DType.float32, W](Float32(1) - Float32(0.5) * beta)
    var lo_scale = SIMD[DType.float32, W](Float32(1) + Float32(0.5) * beta)
    for off in range(0, cols, W):
        var h = (mean_high + off).load[width=W]()
        var l = (mean_low + off).load[width=W]()
        var dp = hi_scale * h - lo_scale * l
        (direction + off).store((dp * inv).cast[DType.bfloat16]())
    return norm[0]


def projection_stats[cols: Int](
    rows: BF16Ptr, n: Int, direction: BF16Ptr,
) -> Tuple[Float64, Float64]:
    var s = Float64(0)
    var s2 = Float64(0)
    for i in range(n):
        var p = Float64(dot_to_scalar[cols](rows + i * cols, direction))
        s += p
        s2 += p * p
    if n <= 0:
        return (Float64(0), Float64(0))
    var m = s / Float64(n)
    return (m, s2 / Float64(n) - m * m)


def fisher_ratio(
    mean_high: Float64, var_high: Float64,
    mean_low: Float64, var_low: Float64,
) -> Float64:
    var denom = var_high + var_low
    if denom <= Float64(0):
        return Float64(0)
    var sep = mean_high - mean_low
    return sep * sep / denom


@fieldwise_init
struct ProbeResult(Copyable, Movable, ImplicitlyCopyable):
    var layer: Int
    var fr: Float64
    var separation: Float64
    var mean_high: Float64
    var mean_low: Float64
    var var_high: Float64
    var var_low: Float64

    @always_inline
    def pooled_std(self) -> Float64:
        var pooled = (self.var_high + self.var_low) / Float64(2)
        if pooled <= Float64(0):
            return Float64(0)
        return Float64(sqrt[DType.float64, 1](pooled)[0])


def build_probe[cols: Int](
    mut samples: ContrastSet[cols],
    layer: Int,
    mean_high: F32Ptr,
    mean_low: F32Ptr,
    direction: BF16Ptr,
) -> ProbeResult:
    class_mean[cols](samples.high_ptr(0), samples.n_high, mean_high)
    class_mean[cols](samples.low_ptr(0), samples.n_low, mean_low)
    var sep = build_direction[cols](mean_high, mean_low, direction)
    var hi = projection_stats[cols](
        samples.high_ptr(0), samples.n_high, direction)
    var lo = projection_stats[cols](
        samples.low_ptr(0), samples.n_low, direction)
    var fr = fisher_ratio(hi[0], hi[1], lo[0], lo[1])
    return ProbeResult(layer, fr, Float64(sep), hi[0], lo[0], hi[1], lo[1])


def mean_row_norm[cols: Int](mut samples: ContrastSet[cols]) -> Float64:
    var total = Float64(0)
    var n = samples.n_high + samples.n_low
    if n <= 0:
        return Float64(0)
    for i in range(samples.n_high):
        var p = samples.high_ptr(i)
        total += Float64(sqrt[DType.float32, 1](
            dot_to_scalar[cols](p, p))[0])
    for i in range(samples.n_low):
        var p = samples.low_ptr(i)
        total += Float64(sqrt[DType.float32, 1](
            dot_to_scalar[cols](p, p))[0])
    return total / Float64(n)
