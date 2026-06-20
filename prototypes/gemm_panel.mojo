from std.collections import InlineArray

from threading.threading_traits import BurstThreadPool
from kernels.helpers import (
    RangePartitionedKernel, Binding, BF16Ptr,
    fanout_dispatch,
)
from kernels.dispatch_heuristics import GEMV_INLINE_ROWS
from kernels.profiling import Profiler

from prototypes.panel_bf16 import bf16_microtile


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
def gemm_panel_range[cols: Int, MR: Int, NC: Int](
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
struct GemmPanelKernel[cols: Int, MR: Int = 4, NC: Int = 4](
    RangePartitionedKernel
):
    """Panel gemm: x [m, cols] bf16, weight [rows, cols] bf16, output [m, rows]
    bf16. Partition over rows. GROUPS=1 instance of the unified bf16 panel: NC
    output columns share each x-chunk load, the MR*NC accumulators carry the
    ILP (no port_unroll)."""
    var x: BF16Ptr
    var weight: BF16Ptr
    var output: BF16Ptr
    var rows: Int
    var m: Int
    var start: Int
    var end: Int

    def execute(mut self):
        gemm_panel_range[Self.cols, Self.MR, Self.NC](
            self.x, self.weight, self.output, self.rows, self.m,
            self.start, self.end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_gemm_panel[
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
    comptime K = GemmPanelKernel[cols, MR, NC]
    var nrows = rows

    @parameter
    def make(r: Int) -> K:
        return K(x[r], weight[r], output[r], nrows, seq_len, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="gemm_panel"](
        pools, prof, rows,
        seq_len * cols * 2 + rows * cols * 2,
        inline_threshold_bytes=GEMV_INLINE_ROWS * cols * 2)
