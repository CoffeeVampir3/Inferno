from std.collections import InlineArray

from threading.threading_traits import BurstThreadPool
from kernels.helpers import (
    WorkerRangePartitionedKernel, Binding,
    BF16Ptr, F32Ptr, I32Ptr, W, BW,
    fanout_dispatch, saturate_workers,
)
from kernels.moe_router import SparseRoute, SparseRoutePtr
from kernels.moe_experts import dispatch_phase2_down
from kernels.panel import bf16_microtile, pick_nc
from kernels.profiling import Profiler

from prototypes.swiglu_oai import (
    swiglu_oai_activate, M3_SWIGLU_ALPHA, M3_SWIGLU_LIMIT,
)


@always_inline
def emit_gate_up_panel_oai[
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
        var gb = InlineArray[BF16Ptr, 2](uninitialized=True)
        gb[0] = gate_w_base + jc * hidden
        gb[1] = up_w_base + jc * hidden
        var vals = bf16_microtile[
            panel, NR, GROUPS=2, contraction=hidden,
        ](x_rows, gb)
        comptime for r in range(panel):
            comptime for c in range(NR):
                gate_part[r * tile_j + jc + c] = vals[c * panel + r]
                up_part[r * tile_j + jc + c] = vals[(NR + c) * panel + r]
        jc += NR

    while jc < n_cols:
        var gb = InlineArray[BF16Ptr, 2](uninitialized=True)
        gb[0] = gate_w_base + jc * hidden
        gb[1] = up_w_base + jc * hidden
        var vals = bf16_microtile[
            panel, 1, GROUPS=2, contraction=hidden,
        ](x_rows, gb)
        comptime for r in range(panel):
            gate_part[r * tile_j + jc] = vals[r]
            up_part[r * tile_j + jc] = vals[panel + r]
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
struct M3Phase1GateUpKernel[
    hidden: Int, gate_up_fused: Int, intermediate: Int,
    alpha: Float32, limit: Float32,
    tile_j: Int = 64, MR: Int = 4, NR: Int = 3,
](WorkerRangePartitionedKernel):
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
            "Phase1: intermediate must be a multiple of f32 SIMD width")
        comptime assert Self.tile_j % W == 0, (
            "Phase1: tile_j must be a multiple of f32 SIMD width")
        comptime assert Self.hidden % BW == 0, (
            "Phase1: hidden must be a multiple of the bf16 SIMD width")
        comptime assert Self.gate_up_fused == 2 * Self.intermediate, (
            "Phase1: gate_up_fused must be 2 * intermediate")
        comptime worker_part = Self.MR * 2 * Self.tile_j
        comptime ENR = pick_nc[Self.MR, 2, Self.NR]()

        debug_assert(
            self.col_start >= 0 and self.col_start <= self.col_end
            and self.col_end <= Self.intermediate,
            "Phase1: column range out of bounds",
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
                    emit_gate_up_panel_oai[
                        panel=Self.MR, NR=ENR, hidden=Self.hidden,
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
                    emit_gate_up_panel_oai[
                        panel=1, NR=ENR, hidden=Self.hidden,
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


def dispatch_m3_phase1_gate_up[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    hidden: Int, gate_up_fused: Int, intermediate: Int,
    alpha: Float32, limit: Float32,
    tile_j: Int = 64, MR: Int = 4, NR: Int = 3, max_worker_count: Int = 128,
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
    comptime K = M3Phase1GateUpKernel[
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
        label="m3_phase1_gate_up",
    ](pools, prof, n_strides, experts_per_rank * gate_up_fused * hidden * 2)


comptime M3_MOE_HIDDEN = 6144
comptime M3_MOE_INTERMEDIATE = 3072
comptime M3_MOE_GATE_UP_FUSED = 2 * M3_MOE_INTERMEDIATE
comptime M3_DENSE_INTERMEDIATE = 12288
comptime M3_DENSE_GATE_UP_FUSED = 2 * M3_DENSE_INTERMEDIATE


def dispatch_minimax_m3_moe_experts[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    hidden: Int, intermediate: Int, NR: Int = 3, max_worker_count: Int = 128,
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
    dispatch_m3_phase1_gate_up[
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
