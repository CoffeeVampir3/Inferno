from std.collections import InlineArray

from threading.threading_traits import BurstThreadPool
from kernels.helpers import (
    WorkerRangePartitionedKernel, Binding,
    BF16Ptr, F32Ptr, I32Ptr, W, BW,
    fanout_dispatch, saturate_workers,
)
from kernels.moe_router import SparseRoute, SparseRoutePtr
from kernels.moe_experts import dispatch_phase2_down
from kernels.profiling import Profiler

from prototypes.swiglu_oai import (
    swiglu_oai_activate, M3_SWIGLU_ALPHA, M3_SWIGLU_LIMIT,
)
from prototypes.panel_bf16 import bf16_microtile


@always_inline
def emit_gate_up_microtile[
    panel: Int, ncol: Int, hidden: Int, tile_j: Int,
](
    read x_rows: InlineArray[BF16Ptr, panel],
    gate_col0: BF16Ptr,
    up_col0: BF16Ptr,
    gate_part: F32Ptr,
    up_part: F32Ptr,
    jc: Int,
):
    """One register tile: `panel` token rows x `ncol` intermediate columns,
    gate and up fused. Each contraction chunk of every x row is loaded once
    and fed into ncol*2 dpbf16ps chains (gate column + up column), so the x
    stream is amortized across both projections and across ncol outputs. The
    panel*ncol*2 independent accumulators supply the instruction-level
    parallelism that port_unroll used to provide in the MR x 1 kernel."""
    var gb = InlineArray[BF16Ptr, 2](uninitialized=True)
    gb[0] = gate_col0
    gb[1] = up_col0
    var vals = bf16_microtile[
        panel, ncol, GROUPS=2, contraction=hidden,
    ](x_rows, gb)

    comptime for r in range(panel):
        comptime for c in range(ncol):
            gate_part[r * tile_j + jc + c] = vals[c * panel + r]
            up_part[r * tile_j + jc + c] = vals[(ncol + c) * panel + r]


@always_inline
def emit_gate_up_panel_fused[
    panel: Int, NR: Int, hidden: Int, intermediate: Int, tile_j: Int,
    alpha: Float32, limit: Float32,
](
    routes: SparseRoutePtr,
    x_normed: BF16Ptr,
    rec_start: Int,
    gate_w_base: BF16Ptr,
    up_w_base: BF16Ptr,
    gate_part: F32Ptr,
    up_part: F32Ptr,
    bucket_base: BF16Ptr,
    n_cols: Int,
):
    var x_rows = InlineArray[BF16Ptr, panel](uninitialized=True)
    comptime for r in range(panel):
        x_rows[r] = x_normed + Int(routes[rec_start + r].token) * hidden

    var jc = 0
    while jc + NR <= n_cols:
        emit_gate_up_microtile[
            panel=panel, ncol=NR, hidden=hidden, tile_j=tile_j,
        ](
            x_rows, gate_w_base + jc * hidden, up_w_base + jc * hidden,
            gate_part, up_part, jc,
        )
        jc += NR

    while jc < n_cols:
        emit_gate_up_microtile[
            panel=panel, ncol=1, hidden=hidden, tile_j=tile_j,
        ](
            x_rows, gate_w_base + jc * hidden, up_w_base + jc * hidden,
            gate_part, up_part, jc,
        )
        jc += 1

    comptime for r in range(panel):
        var bucket_row = bucket_base + r * intermediate
        var src_g = gate_part + r * tile_j
        var src_u = up_part + r * tile_j
        for j_off in range(0, n_cols, W):
            var g = (src_g + j_off).load[width=W]()
            var u = (src_u + j_off).load[width=W]()
            var v = swiglu_oai_activate[W, alpha, limit](g, u)
            (bucket_row + j_off).store(v.cast[DType.bfloat16]())


@fieldwise_init
struct M3Phase1GateUpFusedKernel[
    hidden: Int, gate_up_fused: Int, intermediate: Int,
    alpha: Float32, limit: Float32,
    tile_j: Int = 64, MR: Int = 4, NR: Int = 2,
](WorkerRangePartitionedKernel):
    """Combined gate/up phase-1 projection. Identical partition scheme to
    M3Phase1GateUpKernel (column slice of intermediate per worker, every
    routed expert streamed), but the inner microkernel is the fused panel:
    x loaded once and shared across the gate column, the up column, and NR
    adjacent columns. MR*NR*2 accumulators carry the ILP, so no port_unroll."""

    var x_normed: BF16Ptr
    var expert_offset: I32Ptr
    var routes: SparseRoutePtr
    var experts_gate_up: BF16Ptr
    var gate_scratch: F32Ptr
    var hidden_bucket: BF16Ptr
    var experts_per_rank: Int
    var worker_id: Int
    var col_start: Int
    var col_end: Int

    def execute(mut self):
        comptime assert Self.intermediate % W == 0, (
            "Phase1Fused: intermediate must be a multiple of f32 SIMD width")
        comptime assert Self.tile_j % W == 0, (
            "Phase1Fused: tile_j must be a multiple of f32 SIMD width")
        comptime assert Self.hidden % BW == 0, (
            "Phase1Fused: hidden must be divisible by the bf16 SIMD width")
        comptime assert Self.gate_up_fused == 2 * Self.intermediate, (
            "Phase1Fused: gate_up_fused must be 2 * intermediate")
        comptime worker_part = Self.MR * 2 * Self.tile_j

        debug_assert(
            self.col_start >= 0 and self.col_start <= self.col_end
            and self.col_end <= Self.intermediate,
            "Phase1Fused: column range out of bounds",
        )

        var worker_base = self.gate_scratch + self.worker_id * worker_part
        var gate_part = worker_base
        var up_part = worker_base + Self.MR * Self.tile_j

        for expert in range(self.experts_per_rank):
            var rec_lo = Int(self.expert_offset[expert])
            var rec_hi = Int(self.expert_offset[expert + 1])
            var n_tok = rec_hi - rec_lo
            if n_tok <= 0:
                continue

            var gu_w = self.experts_gate_up + expert * Self.gate_up_fused * Self.hidden

            var j = self.col_start
            while j < self.col_end:
                var n_cols = min(Self.tile_j, self.col_end - j)
                var gate_w_base = gu_w + j * Self.hidden
                var up_w_base = gu_w + (Self.intermediate + j) * Self.hidden

                var rec_block = 0
                while rec_block + Self.MR <= n_tok:
                    var bucket_base = (
                        self.hidden_bucket
                        + (rec_lo + rec_block) * Self.intermediate
                        + j)
                    emit_gate_up_panel_fused[
                        panel=Self.MR, NR=Self.NR, hidden=Self.hidden,
                        intermediate=Self.intermediate, tile_j=Self.tile_j,
                        alpha=Self.alpha, limit=Self.limit,
                    ](
                        self.routes, self.x_normed, rec_lo + rec_block,
                        gate_w_base, up_w_base,
                        gate_part, up_part, bucket_base, n_cols,
                    )
                    rec_block += Self.MR

                while rec_block < n_tok:
                    var bucket_base = (
                        self.hidden_bucket
                        + (rec_lo + rec_block) * Self.intermediate
                        + j)
                    emit_gate_up_panel_fused[
                        panel=1, NR=Self.NR, hidden=Self.hidden,
                        intermediate=Self.intermediate, tile_j=Self.tile_j,
                        alpha=Self.alpha, limit=Self.limit,
                    ](
                        self.routes, self.x_normed, rec_lo + rec_block,
                        gate_w_base, up_w_base,
                        gate_part, up_part, bucket_base, n_cols,
                    )
                    rec_block += 1

                j += Self.tile_j

    @always_inline
    def install_worker_range(
        mut self, worker_id: Int, start: Int, end: Int,
    ):
        self.worker_id = worker_id
        self.col_start = start * W
        self.col_end = end * W


def dispatch_m3_phase1_gate_up_fused[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    hidden: Int, gate_up_fused: Int, intermediate: Int,
    alpha: Float32, limit: Float32,
    tile_j: Int = 64, MR: Int = 4, NR: Int = 2, max_worker_count: Int = 128,
](
    x_normed: Binding[BFloat16, o],
    expert_offset: Binding[Int32, o],
    routes: Binding[SparseRoute, o],
    experts_gate_up: Binding[BFloat16, o],
    gate_scratch: Binding[Float32, o],
    hidden_bucket: Binding[BFloat16, o],
    experts_per_rank: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime K = M3Phase1GateUpFusedKernel[
        hidden, gate_up_fused, intermediate, alpha, limit, tile_j, MR, NR,
    ]
    comptime n_strides = intermediate // W
    var epr = experts_per_rank

    @parameter
    def make(r: Int) -> K:
        return K(x_normed[r], expert_offset[r], routes[r],
                 experts_gate_up[r], gate_scratch[r], hidden_bucket[r],
                 epr, 0, 0, 0)

    fanout_dispatch[
        make,
        max_worker_count=max_worker_count,
        worker_policy=saturate_workers,
        label="m3_phase1_gate_up_fused",
    ](pools, prof, n_strides, experts_per_rank * gate_up_fused * hidden * 2)


def dispatch_minimax_m3_moe_experts_fused[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    hidden: Int, intermediate: Int, NR: Int = 2, max_worker_count: Int = 128,
](
    x_normed: Binding[BFloat16, o],
    expert_offset: Binding[Int32, o],
    routes: Binding[SparseRoute, o],
    experts_gate_up: Binding[BFloat16, o],
    experts_down: Binding[BFloat16, o],
    gate_scratch: Binding[Float32, o],
    hidden_bucket: Binding[BFloat16, o],
    moe_accum: Binding[Float32, o],
    moe_out: Binding[BFloat16, o],
    experts_per_rank: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    dispatch_m3_phase1_gate_up_fused[
        hidden=hidden, gate_up_fused=2 * intermediate, intermediate=intermediate,
        alpha=M3_SWIGLU_ALPHA, limit=M3_SWIGLU_LIMIT, NR=NR,
        max_worker_count=max_worker_count,
    ](x_normed, expert_offset, routes, experts_gate_up,
      gate_scratch, hidden_bucket, experts_per_rank, pools, prof)

    dispatch_phase2_down[
        hidden=hidden, intermediate=intermediate,
        max_worker_count=max_worker_count,
    ](expert_offset, routes, hidden_bucket, experts_down,
      moe_accum, moe_out, experts_per_rank, seq_len, pools, prof)
