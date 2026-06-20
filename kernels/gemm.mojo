from std.collections import InlineArray

from threading.threading_traits import BurstThreadPool
from .helpers import (
    Chain, RangePartitionedKernel,
    fanout_dispatch, saturate_workers,
    Binding, BF16Ptr,
)
from .dispatch_heuristics import GEMV_INLINE_ROWS
from .panel import bf16_microtile, bf16_microtile_runtime, pick_nc
from .profiling import Profiler


@always_inline
def emit_gemm_panel[
    panel: Int, NC: Int, cols: Int,
](
    x_base: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    rows: Int, m_panel: Int, start: Int, end: Int,
):
    var x_rows = InlineArray[BF16Ptr, panel](uninitialized=True)
    comptime for r in range(panel):
        x_rows[r] = x_base + (m_panel + r) * cols

    var gb = InlineArray[BF16Ptr, 1](uninitialized=True)
    var n = start
    while n + NC <= end:
        gb[0] = weight + n * cols
        var vals = bf16_microtile[
            panel, NC, GROUPS=1, contraction=cols,
        ](x_rows, gb)
        comptime for c in range(NC):
            comptime for r in range(panel):
                (output + (m_panel + r) * rows + n + c)[] = (
                    vals[c * panel + r].cast[DType.bfloat16]())
        n += NC

    while n < end:
        gb[0] = weight + n * cols
        var vals = bf16_microtile[
            panel, 1, GROUPS=1, contraction=cols,
        ](x_rows, gb)
        comptime for r in range(panel):
            (output + (m_panel + r) * rows + n)[] = (
                vals[r].cast[DType.bfloat16]())
        n += 1


@always_inline
def gemm_range[cols: Int, MR: Int, NC: Int](
    x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    rows: Int, m: Int, start: Int, end: Int,
):
    var m_panel = 0
    while m_panel + MR <= m:
        emit_gemm_panel[MR, NC, cols](
            x, weight, output, rows, m_panel, start, end)
        m_panel += MR
    while m_panel < m:
        emit_gemm_panel[1, NC, cols](
            x, weight, output, rows, m_panel, start, end)
        m_panel += 1


@fieldwise_init
struct GemmKernel[cols: Int, MR: Int = 4, NC: Int = 4](
    RangePartitionedKernel
):
    """x: [m, cols] bf16 row-major, weight: [rows, cols] bf16 row-major,
    output: [m, rows] bf16 row-major. Partition over the rows axis. The
    contraction `cols` is comptime (HIDDEN); the output `rows` is runtime.
    GROUPS=1 instance of the unified bf16 panel: NC output columns share each
    x-chunk load, the MR*NC accumulators carry the ILP (no port_unroll)."""
    var x: BF16Ptr
    var weight: BF16Ptr
    var output: BF16Ptr
    var rows: Int
    var m: Int
    var start: Int
    var end: Int

    def execute(mut self):
        gemm_range[Self.cols, Self.MR, pick_nc[Self.MR, 1, Self.NC]()](
            self.x, self.weight, self.output, self.rows, self.m,
            self.start, self.end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_gemm[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    cols: Int, MR: Int = 4, NC: Int = 4,
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
    comptime K = GemmKernel[cols, MR, NC]
    var nrows = rows

    @parameter
    def make(r: Int) -> K:
        return K(x[r], weight[r], output[r], nrows, seq_len, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="gemm"](
        pools, prof, rows,
        seq_len * cols * 2 + rows * cols * 2,
        inline_threshold_bytes=GEMV_INLINE_ROWS * cols * 2)


@always_inline
def emit_gemm_col_panel[
    rows: Int, panel: Int, NC: Int,
](
    x_base: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    cols: Int, m_panel: Int, start: Int, end: Int,
):
    var x_rows = InlineArray[BF16Ptr, panel](uninitialized=True)
    comptime for r in range(panel):
        x_rows[r] = x_base + (m_panel + r) * cols

    var gb = InlineArray[BF16Ptr, 1](uninitialized=True)
    var n = start
    while n + NC <= end:
        gb[0] = weight + n * cols
        var vals = bf16_microtile_runtime[
            panel, NC, GROUPS=1,
        ](x_rows, gb, cols)
        comptime for c in range(NC):
            comptime for r in range(panel):
                (output + (m_panel + r) * rows + n + c)[] = (
                    vals[c * panel + r].cast[DType.bfloat16]())
        n += NC

    while n < end:
        gb[0] = weight + n * cols
        var vals = bf16_microtile_runtime[
            panel, 1, GROUPS=1,
        ](x_rows, gb, cols)
        comptime for r in range(panel):
            (output + (m_panel + r) * rows + n)[] = (
                vals[r].cast[DType.bfloat16]())
        n += 1


@always_inline
def gemm_col_range[rows: Int, MR: Int, NC: Int](
    x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    cols: Int, m: Int, start: Int, end: Int,
):
    var m_panel = 0
    while m_panel + MR <= m:
        emit_gemm_col_panel[rows, MR, NC](
            x, weight, output, cols, m_panel, start, end)
        m_panel += MR
    while m_panel < m:
        emit_gemm_col_panel[rows, 1, NC](
            x, weight, output, cols, m_panel, start, end)
        m_panel += 1


@fieldwise_init
struct GemmColKernel[rows: Int, MR: Int = 4, NC: Int = 4](
    RangePartitionedKernel
):
    """Column-sharded matmul: the contraction `cols` (= dim//degree) is runtime;
    the output `rows` (= HIDDEN) is comptime. Partition over the rows axis.
    GROUPS=1 instance of the runtime-contraction panel."""
    var x: BF16Ptr
    var weight: BF16Ptr
    var output: BF16Ptr
    var cols: Int
    var m: Int
    var start: Int
    var end: Int

    def execute(mut self):
        gemm_col_range[Self.rows, Self.MR, pick_nc[Self.MR, 1, Self.NC]()](
            self.x, self.weight, self.output, self.cols, self.m,
            self.start, self.end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_gemm_cols[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    rows: Int, MR: Int = 4, NC: Int = 4,
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
    comptime K = GemmColKernel[rows, MR, NC]
    var ncols = cols

    @parameter
    def make(r: Int) -> K:
        return K(x[r], weight[r], output[r], ncols, seq_len, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="gemm_cols"](
        pools, prof, rows,
        seq_len * cols * 2 + rows * cols * 2,
        inline_threshold_bytes=GEMV_INLINE_ROWS * cols * 2)


@fieldwise_init
struct ScaledGemmKernel[
    cols: Int, MR: Int, NC: Int = 4,
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
        gemm_range[Self.cols, Self.MR, pick_nc[Self.MR, 1, Self.NC]()](
            self.x, self.weight, self.output, self.rows, self.m,
            my_start, my_end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_gemm_chained_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    cols: Int, MR: Int = 4, NC: Int = 4,
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
    comptime Kern = ScaledGemmKernel[cols, MR, NC]
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
