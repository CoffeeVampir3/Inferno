from std.collections import InlineArray
from std.sys.info import simd_width_of

from threading.threading_traits import BurstThreadPool
from simd_math.ops import gelu_tanh_f32
from kernels.helpers import (
    RangePartitionedKernel, Binding, fanout_dispatch, saturate_workers,
    BF16Ptr, I32Ptr,
)
from kernels.elementwise import swiglu_oai_activate
from kernels.moe_router import SparseRoute, SparseRoutePtr
from kernels.profiling import Profiler

from butterquant.convert import store_bf16
from butterquant.gemm import accumulate_tiles_grouped, vnni_grouped_fits
from butterquant.dot_products import vnni_colsum_correct
from butterquant.vnni import VNNI_N_STEP
from butterquant.types import F32Ptr, I8Ptr
from butterquant.weight import (
    ButterquantWeight, ButterquantActivation,
    quant_vnni_packed, quant_has_colsum,
)
from quant.recipe import QuantRecipe


comptime M3_SWIGLU_ALPHA = Float32(1.702)
comptime M3_SWIGLU_LIMIT = Float32(7.0)


@always_inline
def bq_phase1_activate[
    width: Int, activation: StaticString, alpha: Float32, limit: Float32,
](
    g: SIMD[DType.float32, width], u: SIMD[DType.float32, width],
) -> SIMD[DType.float32, width]:
    comptime if activation == "swiglu_oai":
        return swiglu_oai_activate[width, alpha, limit](g, u)
    elif activation == "gelu":
        return gelu_tanh_f32[width](g) * u
    elif activation == "gate":
        return g
    elif activation == "up":
        return u
    else:
        comptime assert False, (
            "bq_phase1_activate: activation must be 'swiglu_oai', 'gelu',"
            " 'gate', or 'up'")
        return g


@always_inline
def emit_bq_gate_up_panel[
    hidden: Int, gate_up: Int, inter: Int, n_inter_tiles: Int, PR: Int,
    activation: StaticString, alpha: Float32, limit: Float32,
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
    comptime assert vnni_grouped_fits[PR, 2, False](), (
        "bq gate/up fused panel exceeds the i32 register budget; lower MR")
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
            var res = bq_phase1_activate[width, activation, alpha, limit](gv, uv)
            store_bf16[width](res, bucket_row + a * width)


@fieldwise_init
struct BqPhase1GateUpActKernel[
    hidden: Int, gate_up: Int, inter: Int, MR: Int,
    activation: StaticString, alpha: Float32, limit: Float32,
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
                    emit_bq_gate_up_panel[
                        Self.hidden, Self.gate_up, Self.inter, n_inter_tiles,
                        Self.MR, Self.activation, Self.alpha, Self.limit,
                    ](it, rec_lo + rb, self.x_i8, self.x_sa, self.routes,
                      w, wsc, cs, self.hidden_bucket)
                    rb += Self.MR
                while rb < n_tok:
                    emit_bq_gate_up_panel[
                        Self.hidden, Self.gate_up, Self.inter, n_inter_tiles, 1,
                        Self.activation, Self.alpha, Self.limit,
                    ](it, rec_lo + rb, self.x_i8, self.x_sa, self.routes,
                      w, wsc, cs, self.hidden_bucket)
                    rb += 1

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.tile_start = start
        self.tile_end = end


def dispatch_bq_phase1_gate_up_act[
    P: BurstThreadPool, quant: QuantRecipe, o: ImmutOrigin,
    Profile: Bool, N: Int, //,
    hidden: Int, gate_up: Int, inter: Int,
    activation: StaticString, alpha: Float32, limit: Float32,
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
    comptime Kern = BqPhase1GateUpActKernel[
        hidden, gate_up, inter, MR, activation, alpha, limit]
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
        label="bq_phase1_gate_up_act",
    ](pools, prof, n_inter_tiles, experts_per_rank * gate_up * hidden)


def dispatch_bq_m3_phase1_gate_up[
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
    dispatch_bq_phase1_gate_up_act[
        hidden=hidden, gate_up=gate_up, inter=inter,
        activation="swiglu_oai", alpha=M3_SWIGLU_ALPHA, limit=M3_SWIGLU_LIMIT,
        MR=MR, max_worker_count=max_worker_count,
    ](act, expert_offset, routes, experts_gate_up, hidden_bucket,
      experts_per_rank, pools, prof)
