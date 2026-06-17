from std.algorithm import vectorize
from simd_math.ops import sqrt
from simd_math.winsorize import winsorize_row

from kernels.helpers import Binding, BF16Ptr, F32Ptr, W
from modeling.gemma4_common import Gemma4BaseConfig


comptime C = Gemma4BaseConfig

comptime MEASURE_OFF = 0
comptime MEASURE_RESIDUAL = 1
comptime MEASURE_BASELINE = 2
comptime MEASURE_MODIFIED = 3

comptime CAPTURE_POINTS = C.NUM_LAYERS + 1


@always_inline
def copy_cast_row[hidden: Int](src: BF16Ptr, dst: F32Ptr):
    def step[width: Int](idx: Int) {read}:
        (dst + idx).store(
            (src + idx).load[width=width]().cast[DType.float32]())

    vectorize[W](hidden, step)


@always_inline
def add_row[hidden: Int](src: F32Ptr, dst: F32Ptr):
    def step[width: Int](idx: Int) {read}:
        (dst + idx).store(
            (dst + idx).load[width=width]() + (src + idx).load[width=width]())

    vectorize[W](hidden, step)


def accumulate_residual_mean[
    o: ImmutOrigin, //, hidden: Int, q: Float64 = 0.995,
](
    x_main: Binding[BFloat16, o],
    read rows: List[Int],
    num_slots: Int,
    capture_idx: Int,
    acc: F32Ptr,
    scratch: F32Ptr,
):
    var src0 = x_main[0]
    var dst = acc + capture_idx * hidden
    for s in range(num_slots):
        var off = rows[s] * hidden
        copy_cast_row[hidden](src0 + off, scratch)
        winsorize_row[cols=hidden, q=q](scratch)
        add_row[hidden](scratch, dst)


struct MeasureState(Movable):
    var mode: Int
    var current_is_bad: Bool
    var good_acc: List[Float32]
    var bad_acc: List[Float32]
    var good_count: Int
    var bad_count: Int
    var base_row_offset: Int
    var kl_sum: Float64
    var kl_rows: Int
    var scratch: List[Float32]

    def __init__(out self):
        comptime size = CAPTURE_POINTS * C.HIDDEN
        self.mode = MEASURE_OFF
        self.current_is_bad = False
        self.good_acc = List[Float32](length=size, fill=Float32(0))
        self.bad_acc = List[Float32](length=size, fill=Float32(0))
        self.good_count = 0
        self.bad_count = 0
        self.base_row_offset = 0
        self.kl_sum = 0.0
        self.kl_rows = 0
        self.scratch = List[Float32](length=C.HIDDEN, fill=Float32(0))

    @always_inline
    def armed(self) -> Bool:
        return self.mode != MEASURE_OFF

    @always_inline
    def acc_ptr(mut self) -> F32Ptr:
        if self.current_is_bad:
            return self.bad_acc.unsafe_ptr().as_unsafe_any_origin()
        return self.good_acc.unsafe_ptr().as_unsafe_any_origin()

    @always_inline
    def scratch_ptr(mut self) -> F32Ptr:
        return self.scratch.unsafe_ptr().as_unsafe_any_origin()

    def arm_residual(mut self, is_bad: Bool):
        self.mode = MEASURE_RESIDUAL
        self.current_is_bad = is_bad

    def arm_baseline(mut self):
        self.mode = MEASURE_BASELINE
        self.base_row_offset = 0

    def arm_modified(mut self):
        self.mode = MEASURE_MODIFIED
        self.base_row_offset = 0

    def disarm(mut self):
        self.mode = MEASURE_OFF

    def reset_directions(mut self):
        for k in range(len(self.good_acc)):
            self.good_acc[k] = Float32(0)
            self.bad_acc[k] = Float32(0)
        self.good_count = 0
        self.bad_count = 0

    def reset_kl(mut self):
        self.kl_sum = 0.0
        self.kl_rows = 0

    def kl_value(self) -> Float64:
        if self.kl_rows == 0:
            return 0.0
        return self.kl_sum / Float64(self.kl_rows)

    def finalize_directions(self) -> List[BFloat16]:
        var out = List[BFloat16](
            length=CAPTURE_POINTS * C.HIDDEN, fill=BFloat16(0))
        var gc = Float32(self.good_count) if self.good_count > 0 else Float32(1)
        var bc = Float32(self.bad_count) if self.bad_count > 0 else Float32(1)
        for cap_idx in range(CAPTURE_POINTS):
            var base = cap_idx * C.HIDDEN
            var sumsq = Float64(0)
            for j in range(C.HIDDEN):
                var d = self.bad_acc[base + j] / bc - self.good_acc[base + j] / gc
                sumsq += Float64(d) * Float64(d)
            var inv = Float32(0)
            if sumsq > 0:
                inv = Float32(1.0) / sqrt[DType.float64, 1](sumsq).cast[
                    DType.float32]()
            for j in range(C.HIDDEN):
                var d = self.bad_acc[base + j] / bc - self.good_acc[base + j] / gc
                out[base + j] = (d * inv).cast[DType.bfloat16]()
        return out^
