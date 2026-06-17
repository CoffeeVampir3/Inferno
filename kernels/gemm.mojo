from std.collections import InlineArray

from simd_math import pick_port_unroll, runtime_pick_port_unroll
from threading.threading_traits import BurstThreadPool
from .helpers import (
    Chain, RangePartitionedKernel,
    fanout_dispatch, saturate_workers,
    Binding, BF16Ptr, BW,
)
from .dispatch_heuristics import GEMV_INLINE_ROWS
from .dot_products import (
    bf16_panel_dot_to_scalars, bf16_panel_dot_to_scalars_runtime,
)
from .profiling import Profiler


@always_inline
def gemm_row_panel[
    panel: Int, //,
    cols: Int, port_unroll: Int,
](
    x_base: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    rows: Int, m_panel: Int, start: Int, end: Int,
):
    var x_rows = InlineArray[BF16Ptr, panel](uninitialized=True)
    comptime for r in range(panel):
        x_rows[r] = x_base + (m_panel + r) * cols

    for n in range(start, end):
        var w_row = weight + n * cols
        var scalars = bf16_panel_dot_to_scalars[
            cols=cols, port_unroll=port_unroll,
        ](w_row, x_rows)
        comptime for r in range(panel):
            (output + (m_panel + r) * rows + n)[] = (
                scalars[r].cast[DType.bfloat16]())


@always_inline
def gemm_range[
    cols: Int, MR: Int,
](
    x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    rows: Int, m: Int, start: Int, end: Int,
):
    comptime PU = pick_port_unroll[BW, cols]()
    var m_panel = 0
    while m_panel + MR <= m:
        gemm_row_panel[
            panel=MR, cols=cols, port_unroll=PU,
        ](x, weight, output, rows, m_panel, start, end)
        m_panel += MR
    while m_panel < m:
        gemm_row_panel[
            panel=1, cols=cols, port_unroll=PU,
        ](x, weight, output, rows, m_panel, start, end)
        m_panel += 1


@fieldwise_init
struct GemmKernel[cols: Int, MR: Int = 4](
    RangePartitionedKernel
):
    """x: [m, cols] bf16 row-major, weight: [rows, cols] bf16 row-major,
    output: [m, rows] bf16 row-major. Partition over the rows axis. The
    contraction `cols` is comptime (HIDDEN); the output `rows` is runtime."""
    var x: BF16Ptr
    var weight: BF16Ptr
    var output: BF16Ptr
    var rows: Int
    var m: Int
    var start: Int
    var end: Int

    def execute(mut self):
        gemm_range[Self.cols, Self.MR](
            self.x, self.weight, self.output, self.rows, self.m,
            self.start, self.end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_gemm[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    cols: Int, MR: Int = 4,
    max_worker_count: Int = 128,
](
    x: Binding[BFloat16, o],
    weight: Binding[BFloat16, o],
    output: Binding[BFloat16, o],
    rows: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if seq_len <= 0:
        return
    comptime K = GemmKernel[cols, MR]
    var nrows = rows

    @parameter
    def make(r: Int) -> K:
        return K(x[r], weight[r], output[r], nrows, seq_len, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="gemm"](
        pools, prof, rows,
        seq_len * cols * 2 + rows * cols * 2,
        inline_threshold_bytes=GEMV_INLINE_ROWS * cols * 2)


@always_inline
def gemm_col_panel[
    panel: Int, //,
    rows: Int, port_unroll: Int,
](
    x_base: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    cols: Int, m_panel: Int, start: Int, end: Int,
):
    var x_rows = InlineArray[BF16Ptr, panel](uninitialized=True)
    comptime for r in range(panel):
        x_rows[r] = x_base + (m_panel + r) * cols

    for n in range(start, end):
        var w_row = weight + n * cols
        var scalars = bf16_panel_dot_to_scalars_runtime[
            port_unroll=port_unroll,
        ](w_row, x_rows, cols)
        comptime for r in range(panel):
            (output + (m_panel + r) * rows + n)[] = (
                scalars[r].cast[DType.bfloat16]())


@always_inline
def gemm_col_range[
    rows: Int, MR: Int, port_unroll: Int,
](
    x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    cols: Int, m: Int, start: Int, end: Int,
):
    var m_panel = 0
    while m_panel + MR <= m:
        gemm_col_panel[
            panel=MR, rows=rows, port_unroll=port_unroll,
        ](x, weight, output, cols, m_panel, start, end)
        m_panel += MR
    while m_panel < m:
        gemm_col_panel[
            panel=1, rows=rows, port_unroll=port_unroll,
        ](x, weight, output, cols, m_panel, start, end)
        m_panel += 1


@fieldwise_init
struct GemmColKernel[rows: Int, MR: Int = 4, port_unroll: Int = 4](
    RangePartitionedKernel
):
    """Column-sharded matmul: the contraction `cols` (= dim//degree) is runtime,
    strip-mined over a comptime `port_unroll`; the output `rows` (= HIDDEN) is
    comptime. Partition over the rows axis."""
    var x: BF16Ptr
    var weight: BF16Ptr
    var output: BF16Ptr
    var cols: Int
    var m: Int
    var start: Int
    var end: Int

    def execute(mut self):
        gemm_col_range[Self.rows, Self.MR, Self.port_unroll](
            self.x, self.weight, self.output, self.cols, self.m,
            self.start, self.end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def _dispatch_gemm_cols_fixed[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    rows: Int, port_unroll: Int, MR: Int = 4,
    max_worker_count: Int = 128,
](
    x: Binding[BFloat16, o],
    weight: Binding[BFloat16, o],
    output: Binding[BFloat16, o],
    cols: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if seq_len <= 0:
        return
    comptime K = GemmColKernel[rows, MR, port_unroll]
    var ncols = cols

    @parameter
    def make(r: Int) -> K:
        return K(x[r], weight[r], output[r], ncols, seq_len, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="gemm_cols"](
        pools, prof, rows,
        seq_len * cols * 2 + rows * cols * 2,
        inline_threshold_bytes=GEMV_INLINE_ROWS * cols * 2)


def dispatch_gemm_cols[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    rows: Int, MR: Int = 4, port_unroll: Int = 0,
    max_worker_count: Int = 128,
](
    x: Binding[BFloat16, o],
    weight: Binding[BFloat16, o],
    output: Binding[BFloat16, o],
    cols: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime if port_unroll != 0:
        _dispatch_gemm_cols_fixed[
            rows, port_unroll=port_unroll, MR=MR,
            max_worker_count=max_worker_count,
        ](x, weight, output, cols, seq_len, pools, prof)
        return

    var pu = runtime_pick_port_unroll(BW, cols)
    if pu == 8:
        _dispatch_gemm_cols_fixed[
            rows, port_unroll=8, MR=MR,
            max_worker_count=max_worker_count,
        ](x, weight, output, cols, seq_len, pools, prof)
    elif pu == 4:
        _dispatch_gemm_cols_fixed[
            rows, port_unroll=4, MR=MR,
            max_worker_count=max_worker_count,
        ](x, weight, output, cols, seq_len, pools, prof)
    elif pu == 2:
        _dispatch_gemm_cols_fixed[
            rows, port_unroll=2, MR=MR,
            max_worker_count=max_worker_count,
        ](x, weight, output, cols, seq_len, pools, prof)
    else:
        _dispatch_gemm_cols_fixed[
            rows, port_unroll=1, MR=MR,
            max_worker_count=max_worker_count,
        ](x, weight, output, cols, seq_len, pools, prof)


@fieldwise_init
struct ScaledGemmKernel[
    cols: Int, MR: Int,
](RangePartitionedKernel):
    var x: BF16Ptr
    var weight: BF16Ptr
    var output: BF16Ptr
    var rows: Int
    var m: Int
    var numer: Int
    var denom: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var my_start = self.start * self.numer // self.denom
        var my_end = self.end * self.numer // self.denom
        gemm_range[Self.cols, Self.MR](
            self.x, self.weight, self.output, self.rows, self.m,
            my_start, my_end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_gemm_chained_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    cols: Int, MR: Int = 4,
    max_worker_count: Int = 128,
](
    x: Binding[BFloat16, o],
    q_weight: Binding[BFloat16, o],
    k_weight: Binding[BFloat16, o],
    v_weight: Binding[BFloat16, o],
    q_out: Binding[BFloat16, o],
    k_out: Binding[BFloat16, o],
    v_out: Binding[BFloat16, o],
    q_rows: Int, kv_rows: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if seq_len <= 0:
        return
    var total_rows = q_rows + kv_rows + kv_rows
    comptime Kern = ScaledGemmKernel[cols, MR]
    comptime QK = Chain[Kern, Kern]
    comptime QKV = Chain[QK, Kern]
    var qr = q_rows
    var kr = kv_rows
    var tr = total_rows

    @parameter
    def make(r: Int) -> QKV:
        return QKV(
            QK(
                Kern(x[r], q_weight[r], q_out[r], qr, seq_len, qr, tr, 0, 0),
                Kern(x[r], k_weight[r], k_out[r], kr, seq_len, kr, tr, 0, 0),
            ),
            Kern(x[r], v_weight[r], v_out[r], kr, seq_len, kr, tr, 0, 0),
        )

    fanout_dispatch[
        make,
        max_worker_count=max_worker_count,
        worker_policy=saturate_workers,
        label="gemm_chained_qkv",
    ](pools, prof, total_rows, seq_len * cols * 2 + total_rows * cols * 2)
