from std.sys.info import simd_width_of, size_of

from simd_math.ops import sqrt
from threading.threading_traits import BurstThreadPool
from modeling.model_spec import F32
from modeling.slot import SlotLike, BindContext
from .helpers import RangePartitionedKernel, Binding, fanout_dispatch, F32Ptr
from .reductions import dispatch_allreduce_inplace
from .profiling import Profiler


@always_inline
def rsqrt32(x: Float32) -> Float32:
    return Float32(1) / sqrt[DType.float32, 1](x)[0]


@fieldwise_init
struct RowSumSqKernel[WT: DType](RangePartitionedKernel):
    var weight: UnsafePointer[Scalar[Self.WT], MutAnyOrigin]
    var out_m: F32Ptr
    var cols: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime width = simd_width_of[DType.float32]()
        for i in range(self.start, self.end):
            var row = self.weight + i * self.cols
            var acc = SIMD[DType.float32, width](0)
            var j = 0
            while j + width <= self.cols:
                var w = (row + j).load[width=width]().cast[DType.float32]()
                acc = w.fma(w, acc)
                j += width
            var s = acc.reduce_add()
            while j < self.cols:
                var x = (row + j).load().cast[DType.float32]()
                s += x * x
                j += 1
            self.out_m[i] = s

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct ColProjectKernel[WT: DType](RangePartitionedKernel):
    var weight: UnsafePointer[Scalar[Self.WT], MutAnyOrigin]
    var v: F32Ptr
    var m: F32Ptr
    var out_p: F32Ptr
    var rows: Int
    var cols: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime width = simd_width_of[DType.float32]()
        var c = self.start
        while c < self.end:
            self.out_p[c] = Float32(0)
            c += 1
        for i in range(self.rows):
            var mi = self.m[i]
            if mi <= Float32(0):
                continue
            var u = self.v[i] * rsqrt32(mi)
            var uvec = SIMD[DType.float32, width](u)
            var row = self.weight + i * self.cols
            var col = self.start
            while col + width <= self.end:
                var w = (row + col).load[width=width]().cast[DType.float32]()
                var pv = (self.out_p + col).load[width=width]()
                (self.out_p + col).store(w.fma(uvec, pv))
                col += width
            while col < self.end:
                self.out_p[col] = (
                    self.out_p[col] + u * (row + col).load().cast[DType.float32]())
                col += 1

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct RowDotKernel[WT: DType](RangePartitionedKernel):
    var weight: UnsafePointer[Scalar[Self.WT], MutAnyOrigin]
    var p: F32Ptr
    var out_a: F32Ptr
    var cols: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime width = simd_width_of[DType.float32]()
        for i in range(self.start, self.end):
            var row = self.weight + i * self.cols
            var acc = SIMD[DType.float32, width](0)
            var j = 0
            while j + width <= self.cols:
                var w = (row + j).load[width=width]().cast[DType.float32]()
                var pv = (self.p + j).load[width=width]()
                acc = w.fma(pv, acc)
                j += width
            var s = acc.reduce_add()
            while j < self.cols:
                s += (row + j).load().cast[DType.float32]() * self.p[j]
                j += 1
            self.out_a[i] = s

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct ApplyEditKernel[WT: DType](RangePartitionedKernel):
    var pris: UnsafePointer[Scalar[Self.WT], MutAnyOrigin]
    var live: UnsafePointer[Scalar[Self.WT], MutAnyOrigin]
    var v: F32Ptr
    var m: F32Ptr
    var a: F32Ptr
    var p: F32Ptr
    var cols: Int
    var alpha: Float32
    var pnorm: Float32
    var start: Int
    var end: Int

    def execute(mut self):
        comptime width = simd_width_of[DType.float32]()
        for i in range(self.start, self.end):
            var prow = self.pris + i * self.cols
            var lrow = self.live + i * self.cols
            var ss = self.m[i]
            if ss <= Float32(0):
                var j0 = 0
                while j0 + width <= self.cols:
                    (lrow + j0).store((prow + j0).load[width=width]())
                    j0 += width
                while j0 < self.cols:
                    (lrow + j0).store((prow + j0).load())
                    j0 += 1
                continue
            var inv_norm = rsqrt32(ss)
            var norm = ss * inv_norm
            var ai = self.a[i] * inv_norm
            var av = self.alpha * self.v[i]
            var denom = Float32(1) - Float32(2) * av * ai + av * av * self.pnorm
            if denom < Float32(1e-12):
                denom = Float32(1e-12)
            var s = rsqrt32(denom)
            var c = s * av * norm
            var svec = SIMD[DType.float32, width](s)
            var cvec = SIMD[DType.float32, width](c)
            var j = 0
            while j + width <= self.cols:
                var w = (prow + j).load[width=width]().cast[DType.float32]()
                var pv = (self.p + j).load[width=width]()
                (lrow + j).store((w * svec - pv * cvec).cast[Self.WT]())
                j += width
            while j < self.cols:
                var w = (prow + j).load().cast[DType.float32]()
                (lrow + j).store((w * s - self.p[j] * c).cast[Self.WT]())
                j += 1

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_row_sumsq[
    WT: DType, P: BurstThreadPool, Profile: Bool, N: Int,
    ow: ImmutOrigin, os: ImmutOrigin, //, max_worker_count: Int = 128,
](
    weight: Binding[Scalar[WT], ow], out_m: Binding[Float32, os],
    rows: Int, cols: Int, mut pools: List[P], mut prof: Profiler[Profile, N],
):
    comptime K = RowSumSqKernel[WT]
    var ncols = cols

    @parameter
    def make(r: Int) -> K:
        return K(weight[r], out_m[r], ncols, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="ablit.m"](
        pools, prof, rows, rows * cols * size_of[Scalar[WT]]())


def dispatch_col_project[
    WT: DType, P: BurstThreadPool, Profile: Bool, N: Int,
    ow: ImmutOrigin, os: ImmutOrigin, //, max_worker_count: Int = 128,
](
    weight: Binding[Scalar[WT], ow], v: Binding[Float32, os],
    m: Binding[Float32, os], out_p: Binding[Float32, os],
    rows: Int, cols: Int, mut pools: List[P], mut prof: Profiler[Profile, N],
):
    comptime K = ColProjectKernel[WT]
    var nrows = rows
    var ncols = cols

    @parameter
    def make(r: Int) -> K:
        return K(weight[r], v[r], m[r], out_p[r], nrows, ncols, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="ablit.p"](
        pools, prof, cols, rows * cols * size_of[Scalar[WT]]())


def dispatch_row_dot[
    WT: DType, P: BurstThreadPool, Profile: Bool, N: Int,
    ow: ImmutOrigin, os: ImmutOrigin, //, max_worker_count: Int = 128,
](
    weight: Binding[Scalar[WT], ow], p: Binding[Float32, os],
    out_a: Binding[Float32, os],
    rows: Int, cols: Int, mut pools: List[P], mut prof: Profiler[Profile, N],
):
    comptime K = RowDotKernel[WT]
    var ncols = cols

    @parameter
    def make(r: Int) -> K:
        return K(weight[r], p[r], out_a[r], ncols, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="ablit.a"](
        pools, prof, rows, rows * cols * size_of[Scalar[WT]]())


def dispatch_apply[
    WT: DType, P: BurstThreadPool, Profile: Bool, N: Int,
    ow: ImmutOrigin, os: ImmutOrigin, //, max_worker_count: Int = 128,
](
    pris: Binding[Scalar[WT], ow], live: Binding[Scalar[WT], ow],
    v: Binding[Float32, os], m: Binding[Float32, os],
    a: Binding[Float32, os], p: Binding[Float32, os],
    read pnorms: List[Float32],
    rows: Int, cols: Int, alpha: Float32,
    mut pools: List[P], mut prof: Profiler[Profile, N],
):
    comptime K = ApplyEditKernel[WT]
    var ncols = cols
    var al = alpha

    @parameter
    def make(r: Int) -> K:
        return K(pris[r], live[r], v[r], m[r], a[r], p[r], ncols, al,
                 pnorms[r], 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="ablit.apply"](
        pools, prof, rows, rows * cols * size_of[Scalar[WT]]())


def abliterate_matrix[
    WT: DType, P: BurstThreadPool, Profile: Bool, N: Int,
    ow: ImmutOrigin, os: ImmutOrigin, //, reduce: Bool,
](
    pris: Binding[Scalar[WT], ow], live: Binding[Scalar[WT], ow],
    v: Binding[Float32, os], m: Binding[Float32, os],
    a: Binding[Float32, os], p: Binding[Float32, os],
    rows: Int, cols: Int, alpha: Float32,
    mut pools: List[P], mut prof: Profiler[Profile, N],
):
    dispatch_row_sumsq(pris, m, rows, cols, pools, prof)
    comptime if reduce:
        dispatch_allreduce_inplace[F32](m, rows, pools, prof)
    dispatch_col_project(pris, v, m, p, rows, cols, pools, prof)
    dispatch_row_dot(pris, p, a, rows, cols, pools, prof)
    comptime if reduce:
        dispatch_allreduce_inplace[F32](a, rows, pools, prof)

    var degree = len(pools)
    var pnorms = List[Float32](capacity=degree)
    for r in range(degree):
        var mr = m[r]
        var ar = a[r]
        var vr = v[r]
        var acc = Float32(0)
        for i in range(rows):
            var mi = mr[i]
            if mi > Float32(0):
                acc += vr[i] * (ar[i] * rsqrt32(mi))
        pnorms.append(acc)

    dispatch_apply(pris, live, v, m, a, p, pnorms, rows, cols, alpha, pools, prof)


def dispatch_abliterate_dense[
    S: SlotLike, P: BurstThreadPool, Profile: Bool, N: Int,
    ow: ImmutOrigin, os: ImmutOrigin, //, reduce: Bool,
](
    read pris: S, read live: S,
    pctx: BindContext[ow], lctx: BindContext[ow],
    v: Binding[Float32, os], m: Binding[Float32, os],
    a: Binding[Float32, os], p: Binding[Float32, os],
    alpha: Float32, mut pools: List[P], mut prof: Profiler[Profile, N],
):
    comptime WT = S.ENCODING.DTYPE
    var degree = lctx.degree()
    var rows = S.SHAPE.data_n(degree)
    var cols = S.SHAPE.data_m(degree)
    var pris_ptr = UnsafePointer[Scalar[WT], MutAnyOrigin](
        unsafe_from_address=pctx.layer_address() + pris.get_offset())
    var live_ptr = UnsafePointer[Scalar[WT], MutAnyOrigin](
        unsafe_from_address=lctx.layer_address() + live.get_offset())
    abliterate_matrix[reduce=reduce](
        pctx.bind(pris_ptr), lctx.bind(live_ptr), v, m, a, p,
        rows, cols, alpha, pools, prof)


def dispatch_abliterate_experts[
    S: SlotLike, P: BurstThreadPool, Profile: Bool, N: Int,
    ow: ImmutOrigin, os: ImmutOrigin, //,
](
    read pris: S, read live: S,
    pctx: BindContext[ow], lctx: BindContext[ow],
    v: Binding[Float32, os], m: Binding[Float32, os],
    a: Binding[Float32, os], p: Binding[Float32, os],
    rows_per_expert: Int, alpha: Float32,
    mut pools: List[P], mut prof: Profiler[Profile, N],
):
    comptime WT = S.ENCODING.DTYPE
    var degree = lctx.degree()
    var cols = S.SHAPE.data_m(degree)
    var total = S.SHAPE.data_n(degree)
    var pris_ptr = UnsafePointer[Scalar[WT], MutAnyOrigin](
        unsafe_from_address=pctx.layer_address() + pris.get_offset())
    var live_ptr = UnsafePointer[Scalar[WT], MutAnyOrigin](
        unsafe_from_address=lctx.layer_address() + live.get_offset())
    var pris0 = pctx.bind(pris_ptr)
    var live0 = lctx.bind(live_ptr)
    var n = total // rows_per_expert
    for k in range(n):
        var off = k * rows_per_expert * cols
        abliterate_matrix[reduce=False](
            pris0.shifted(off), live0.shifted(off), v, m, a, p,
            rows_per_expert, cols, alpha, pools, prof)
