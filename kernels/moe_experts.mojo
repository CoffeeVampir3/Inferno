from std.collections import InlineArray

from threading.threading_traits import BurstThreadPool
from simd_math.ops import gelu_tanh_f32
from .helpers import (
    RangePartitionedKernel, WorkerRangePartitionedKernel, Binding,
    BF16Ptr, F32Ptr, I32Ptr, W, BW,
    fanout_dispatch, saturate_workers,
)
from .dot_products import bf16_panel_dot_to_scalars
from .moe_router import SparseRoute, SparseRoutePtr
from .profiling import Profiler


@always_inline
def emit_gate_up_panel[
    panel: Int, hidden: Int, intermediate: Int, tile_j: Int, port_unroll: Int,
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

    for j_off in range(n_cols):
        var w_row_g = gate_w_base + j_off * hidden
        var w_row_u = up_w_base + j_off * hidden
        var g_vals = bf16_panel_dot_to_scalars[
            cols=hidden, port_unroll=port_unroll,
        ](w_row_g, x_rows)
        var u_vals = bf16_panel_dot_to_scalars[
            cols=hidden, port_unroll=port_unroll,
        ](w_row_u, x_rows)
        comptime for r in range(panel):
            gate_part[r * tile_j + j_off] = g_vals[r]
            up_part[r * tile_j + j_off] = u_vals[r]

    comptime for r in range(panel):
        var bucket_row = bucket_base + r * intermediate
        var src_g = gate_part + r * tile_j
        var src_u = up_part + r * tile_j
        for j_off in range(0, n_cols, W):
            var g = (src_g + j_off).load[width=W]()
            var u = (src_u + j_off).load[width=W]()
            var v = gelu_tanh_f32[W](g) * u
            (bucket_row + j_off).store(v.cast[DType.bfloat16]())


@fieldwise_init
struct Phase1GateUpKernel[
    hidden: Int, gate_up_fused: Int, intermediate: Int,
    tile_j: Int = 64, MR: Int = 4,
](WorkerRangePartitionedKernel):
    """Column-partitioned gate/up projection. Each worker owns an
    intermediate-column slice [col_start, col_end) and streams that slice
    of every routed expert's gate_up weight. In decode a token routes to
    only top_k experts, so partitioning over experts would leave each
    expert's weight to a single worker; partitioning over columns spreads
    every expert's weight stream across all workers. Bucket writes are
    disjoint across workers by column."""
    comptime PU_GU = 4

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
        comptime PU_GU = Self.PU_GU
        comptime STRIDE_GU = PU_GU * BW

        comptime assert Self.intermediate % W == 0, (
            "Phase1: intermediate must be a multiple of f32 SIMD width")
        comptime assert Self.tile_j % W == 0, (
            "Phase1: tile_j must be a multiple of f32 SIMD width")
        comptime assert Self.hidden % STRIDE_GU == 0, (
            "Phase1: hidden must be divisible by STRIDE_GU")
        comptime assert Self.gate_up_fused == 2 * Self.intermediate, (
            "Phase1: gate_up_fused must be 2 * intermediate")
        comptime worker_part = Self.MR * 2 * Self.tile_j

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
                    emit_gate_up_panel[
                        panel=Self.MR, hidden=Self.hidden,
                        intermediate=Self.intermediate, tile_j=Self.tile_j,
                        port_unroll=PU_GU,
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
                    emit_gate_up_panel[
                        panel=1, hidden=Self.hidden,
                        intermediate=Self.intermediate, tile_j=Self.tile_j,
                        port_unroll=PU_GU,
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


def dispatch_phase1_gate_up[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    hidden: Int, gate_up_fused: Int, intermediate: Int,
    tile_j: Int = 64, MR: Int = 4, max_worker_count: Int = 128,
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
    comptime K = Phase1GateUpKernel[
        hidden, gate_up_fused, intermediate, tile_j, MR,
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
        label="phase1_gate_up",
    ](pools, prof, n_strides, experts_per_rank * gate_up_fused * hidden * 2)


@always_inline
def emit_down_panel[
    panel: Int, hidden: Int, intermediate: Int, port_unroll: Int,
](
    routes: SparseRoutePtr,
    moe_accum: F32Ptr,
    rec_start: Int,
    hm_base: BF16Ptr,
    down_w: BF16Ptr,
    start: Int, end: Int,
):
    var hm_rows = InlineArray[BF16Ptr, panel](uninitialized=True)
    var dst_rows = InlineArray[F32Ptr, panel](uninitialized=True)
    var weights = InlineArray[Float32, panel](uninitialized=True)
    comptime for r in range(panel):
        hm_rows[r] = hm_base + r * intermediate
        var rec = routes[rec_start + r]
        dst_rows[r] = moe_accum + Int(rec.token) * hidden
        weights[r] = rec.weight

    for m in range(start, end):
        var w_row = down_w + m * intermediate
        var vals = bf16_panel_dot_to_scalars[
            cols=intermediate, port_unroll=port_unroll,
        ](w_row, hm_rows)
        comptime for r in range(panel):
            (dst_rows[r] + m)[] = (dst_rows[r] + m)[] + vals[r] * weights[r]


@fieldwise_init
struct Phase2DownKernel[
    hidden: Int, intermediate: Int,
](RangePartitionedKernel):
    comptime TOK_TILE = 64
    comptime MR = 4
    comptime PU_DN = 2

    var expert_offset: I32Ptr
    var routes: SparseRoutePtr
    var hidden_bucket: BF16Ptr
    var experts_down: BF16Ptr
    var moe_accum: F32Ptr
    var moe_partial: BF16Ptr
    var experts_per_rank: Int
    var seq_len: Int
    var col_start: Int
    var col_end: Int

    def execute(mut self):
        comptime PU_DN = Self.PU_DN
        comptime STRIDE_DN = PU_DN * BW
        comptime MR = Self.MR
        comptime assert Self.intermediate % STRIDE_DN == 0, (
            "Phase2: intermediate must divide STRIDE_DN")

        for tok in range(self.seq_len):
            var acc_row = self.moe_accum + tok * Self.hidden
            var m = self.col_start
            while m + W <= self.col_end:
                (acc_row + m).store(SIMD[DType.float32, W](0))
                m += W
            while m < self.col_end:
                (acc_row + m)[] = Float32(0)
                m += 1

        for e in range(self.experts_per_rank):
            var rec_lo = Int(self.expert_offset[e])
            var rec_hi = Int(self.expert_offset[e + 1])
            var n_tok_total = rec_hi - rec_lo
            if n_tok_total <= 0:
                continue
            var down_w = self.experts_down + e * Self.hidden * Self.intermediate

            var tok_base = 0
            while tok_base < n_tok_total:
                var n_tok = min(Self.TOK_TILE, n_tok_total - tok_base)

                var rec_block = 0
                while rec_block + MR <= n_tok:
                    var rec_start = rec_lo + tok_base + rec_block
                    var hm_base = self.hidden_bucket + rec_start * Self.intermediate
                    emit_down_panel[
                        panel=MR, hidden=Self.hidden,
                        intermediate=Self.intermediate, port_unroll=PU_DN,
                    ](
                        self.routes, self.moe_accum, rec_start,
                        hm_base, down_w, self.col_start, self.col_end,
                    )
                    rec_block += MR

                while rec_block < n_tok:
                    var rec_start = rec_lo + tok_base + rec_block
                    var hm_base = self.hidden_bucket + rec_start * Self.intermediate
                    emit_down_panel[
                        panel=1, hidden=Self.hidden,
                        intermediate=Self.intermediate, port_unroll=PU_DN,
                    ](
                        self.routes, self.moe_accum, rec_start,
                        hm_base, down_w, self.col_start, self.col_end,
                    )
                    rec_block += 1

                tok_base += n_tok

        for tok in range(self.seq_len):
            var acc_row = self.moe_accum + tok * Self.hidden
            var dst_row = self.moe_partial + tok * Self.hidden
            var m = self.col_start
            while m + W <= self.col_end:
                var v = (acc_row + m).load[width=W]()
                (dst_row + m).store(v.cast[DType.bfloat16]())
                m += W
            while m < self.col_end:
                (dst_row + m)[] = (acc_row + m)[].cast[DType.bfloat16]()
                m += 1

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.col_start = start * W
        self.col_end = end * W


def dispatch_phase2_down[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    hidden: Int, intermediate: Int,
    max_worker_count: Int = 128,
](
    expert_offset: Binding[Int32, o],
    routes: Binding[SparseRoute, o],
    hidden_bucket: Binding[BFloat16, o],
    experts_down: Binding[BFloat16, o],
    moe_accum: Binding[Float32, o],
    moe_partial: Binding[BFloat16, o],
    experts_per_rank: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime K = Phase2DownKernel[hidden, intermediate]
    comptime hidden_strides = hidden // W
    var epr = experts_per_rank

    @parameter
    def make(r: Int) -> K:
        return K(expert_offset[r], routes[r], hidden_bucket[r],
                 experts_down[r], moe_accum[r], moe_partial[r],
                 epr, seq_len, 0, 0)

    fanout_dispatch[
        make,
        max_worker_count=max_worker_count,
        worker_policy=saturate_workers,
        label="phase2_down",
    ](pools, prof, hidden_strides, seq_len * hidden * 2)
