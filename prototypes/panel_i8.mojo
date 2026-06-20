from std.collections import InlineArray
from std.sys.info import simd_width_of

from threading.threading_traits import BurstThreadPool
from simd_math.ops import gelu_tanh_f32
from kernels.helpers import (
    RangePartitionedKernel, Binding, fanout_dispatch, saturate_workers,
    BF16Ptr, I32Ptr,
)
from kernels.moe_router import SparseRoute, SparseRoutePtr
from kernels.profiling import Profiler

from butterquant.convert import store_bf16
from butterquant.dot_products import act_broadcast_vnni, dot_loaded, vnni_colsum_correct
from butterquant.vnni import VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK
from butterquant.types import F32Ptr, I8Ptr
from butterquant.weight import (
    ButterquantWeight, ButterquantActivation,
    quant_vnni_packed, quant_has_colsum,
)
from quant.recipe import QuantRecipe


@always_inline
def vnni_panel_fits[PR: Int, GROUPS: Int, per_block: Bool]() -> Bool:
    """i32 register budget for the grouped VNNI panel. Each tile costs acc_count
    i32 accumulators per row per group (acc_count = VNNI_N_STEP/width = 2). The
    activation broadcast is shared across groups, so it is counted once per row,
    not per group. Mirrors butterquant.gemm.reg_capped_panel, generalized to
    GROUPS so gate/up fusion (GROUPS=2) can be budget-checked."""
    comptime width = simd_width_of[DType.int32]()
    comptime acc_count = VNNI_N_STEP // width
    comptime regs = 2 * width
    comptime per_pr = GROUPS * (2 * acc_count if per_block else acc_count) + 1
    comptime reserve = VNNI_TILE_N // width + 1
    return PR * per_pr + reserve <= regs


@always_inline
def accumulate_tiles_grouped[
    width: Int, PR: Int, GROUPS: Int, row_ptr: def(Int) capturing [_] -> I8Ptr,
](
    wpacked: I8Ptr,
    read packed_base: InlineArray[Int, GROUPS],
    k_base: Int,
    k_len: Int,
    mut acc: InlineArray[
        SIMD[DType.int32, width], GROUPS * PR * (VNNI_N_STEP // width),
    ],
):
    """One VNNI K-walk feeding GROUPS weight regions from a single activation
    broadcast. For each (ks, dc) chunk the activation bytes are loaded and
    bias-folded once into `ab[r]`, then vpdpbusd'd against every group's tile.
    This is the int8 analog of "load each x chunk once, feed both projections":
    the gate and up streams share the activation read instead of re-walking it.
    Per-group accumulators are contiguous (group g at offset g*PR*acc_count),
    so each group finalizes exactly like the single-group kernel."""
    comptime passes = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width
    comptime dc_count = VNNI_K_STEP // VNNI_BLK
    comptime tile_dc_bytes = VNNI_TILE_N * VNNI_BLK
    comptime tile_ks_bytes = dc_count * tile_dc_bytes
    comptime group_stride = PR * acc_count

    var packed_off = InlineArray[Int, GROUPS](uninitialized=True)
    comptime for g in range(GROUPS):
        packed_off[g] = packed_base[g]

    for ks in range(0, k_len, VNNI_K_STEP):
        for dc in range(dc_count):
            var k_pos = k_base + ks + dc * VNNI_BLK
            var ab = InlineArray[SIMD[DType.uint8, width * 4], PR](
                uninitialized=True)
            comptime for r in range(PR):
                ab[r] = act_broadcast_vnni[width](row_ptr(r), k_pos)
            comptime for g in range(GROUPS):
                var t0 = packed_off[g] + dc * tile_dc_bytes
                var t1 = t0 + tile_ks_bytes
                comptime for p in range(passes):
                    var w0 = (wpacked + t0 + p * bytes_per_pass).load[
                        width = width * 4, non_temporal=True]()
                    comptime for r in range(PR):
                        var k = g * group_stride + r * acc_count + p
                        acc[k] = dot_loaded[width](acc[k], ab[r], w0)
                comptime for p in range(passes):
                    var w1 = (wpacked + t1 + p * bytes_per_pass).load[
                        width = width * 4, non_temporal=True]()
                    comptime for r in range(PR):
                        var k = g * group_stride + r * acc_count + passes + p
                        acc[k] = dot_loaded[width](acc[k], ab[r], w1)
        comptime for g in range(GROUPS):
            packed_off[g] += 2 * tile_ks_bytes


@always_inline
def emit_bq_gate_up_fused[
    hidden: Int, gate_up: Int, inter: Int, n_inter_tiles: Int, PR: Int,
](
    it: Int,
    rec_start: Int,
    x_i8: I8Ptr,
    x_sa: F32Ptr,
    routes: SparseRoutePtr,
    w: I8Ptr,
    wsc: F32Ptr,
    cs: F32Ptr,
    bucket: BF16Ptr,
):
    comptime assert vnni_panel_fits[PR, 2, False](), (
        "bq gate/up fused panel exceeds the i32 register budget; lower PR")
    comptime width = simd_width_of[DType.int32]()
    comptime acc_count = VNNI_N_STEP // width
    comptime group_stride = PR * acc_count
    comptime inv127 = Float32(1.0) / Float32(127.0)
    var gate_tile = it
    var up_tile = n_inter_tiles + it

    var rows = InlineArray[I8Ptr, PR](uninitialized=True)
    var scales = InlineArray[Float32, PR](uninitialized=True)
    comptime for r in range(PR):
        var tok = Int(routes[rec_start + r].token)
        rows[r] = x_i8 + tok * hidden
        scales[r] = x_sa[tok]

    @parameter
    def row_ptr(r: Int) -> I8Ptr:
        return rows[r]

    var pbase = InlineArray[Int, 2](uninitialized=True)
    pbase[0] = gate_tile * VNNI_N_STEP * hidden
    pbase[1] = up_tile * VNNI_N_STEP * hidden

    var acc = InlineArray[SIMD[DType.int32, width], 2 * PR * acc_count](
        fill=SIMD[DType.int32, width](0))
    accumulate_tiles_grouped[width, PR, 2, row_ptr](w, pbase, 0, hidden, acc)

    comptime for r in range(PR):
        var ad = scales[r] * inv127
        var bucket_row = bucket + (rec_start + r) * inter + it * VNNI_N_STEP
        comptime for a in range(acc_count):
            var ng = gate_tile * VNNI_N_STEP + a * width
            var nu = up_tile * VNNI_N_STEP + a * width
            var gcs = (cs + ng).load[width=width]()
            var ucs = (cs + nu).load[width=width]()
            var gv = vnni_colsum_correct[width](
                acc[r * acc_count + a], gcs) * ad * (wsc + ng).load[width=width]()
            var uv = vnni_colsum_correct[width](
                acc[group_stride + r * acc_count + a], ucs) * ad * (
                wsc + nu).load[width=width]()
            var res = gelu_tanh_f32[width](gv) * uv
            store_bf16[width](res, bucket_row + a * width)


@fieldwise_init
struct BqPhase1GateUpFusedKernel[
    hidden: Int, gate_up: Int, inter: Int, MR: Int = 4,
](RangePartitionedKernel):
    var x_i8: I8Ptr
    var x_sa: F32Ptr
    var expert_offset: I32Ptr
    var routes: SparseRoutePtr
    var experts_gate_up: I8Ptr
    var gate_up_scale: F32Ptr
    var gate_up_colsum: F32Ptr
    var hidden_bucket: BF16Ptr
    var experts_per_rank: Int
    var tile_start: Int
    var tile_end: Int

    def execute(mut self):
        comptime n_inter_tiles = Self.inter // VNNI_N_STEP
        for expert in range(self.experts_per_rank):
            var rec_lo = Int(self.expert_offset[expert])
            var rec_hi = Int(self.expert_offset[expert + 1])
            var n_tok = rec_hi - rec_lo
            if n_tok <= 0:
                continue
            var w = self.experts_gate_up + expert * Self.gate_up * Self.hidden
            var wsc = self.gate_up_scale + expert * Self.gate_up
            var cs = self.gate_up_colsum + expert * Self.gate_up
            for it in range(self.tile_start, self.tile_end):
                var rb = 0
                while rb + Self.MR <= n_tok:
                    emit_bq_gate_up_fused[
                        Self.hidden, Self.gate_up, Self.inter, n_inter_tiles, Self.MR,
                    ](it, rec_lo + rb, self.x_i8, self.x_sa, self.routes,
                      w, wsc, cs, self.hidden_bucket)
                    rb += Self.MR
                while rb < n_tok:
                    emit_bq_gate_up_fused[
                        Self.hidden, Self.gate_up, Self.inter, n_inter_tiles, 1,
                    ](it, rec_lo + rb, self.x_i8, self.x_sa, self.routes,
                      w, wsc, cs, self.hidden_bucket)
                    rb += 1

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.tile_start = start
        self.tile_end = end


def dispatch_bq_phase1_gate_up_fused[
    P: BurstThreadPool, quant: QuantRecipe, o: ImmutOrigin,
    Profile: Bool, N: Int, //,
    hidden: Int, gate_up: Int, inter: Int,
    MR: Int = 4, max_worker_count: Int = 128,
](
    act: ButterquantActivation[o],
    expert_offset: Binding[Int32, o],
    routes: Binding[SparseRoute, o],
    experts_gate_up: ButterquantWeight[quant, o],
    hidden_bucket: Binding[BFloat16, o],
    experts_per_rank: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime assert quant_vnni_packed[quant](), "bq phase1 consumes VNNI-packed experts"
    comptime assert quant_has_colsum[quant](), "bq phase1 requires a colsum sidecar"
    comptime n_inter_tiles = inter // VNNI_N_STEP
    comptime Kern = BqPhase1GateUpFusedKernel[hidden, gate_up, inter, MR]
    var cs = experts_gate_up.colsum_checked()

    @parameter
    def make(r: Int) -> Kern:
        return Kern(act.data[r], act.scale[r], expert_offset[r], routes[r],
                    experts_gate_up.data[r], experts_gate_up.scale[r], cs[r],
                    hidden_bucket[r], experts_per_rank, 0, 0)

    fanout_dispatch[
        make,
        max_worker_count=max_worker_count,
        worker_policy=saturate_workers,
        label="bq_phase1_gate_up_fused",
    ](pools, prof, n_inter_tiles, experts_per_rank * gate_up * hidden)
