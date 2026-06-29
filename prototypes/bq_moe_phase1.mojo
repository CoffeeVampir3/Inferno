from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from threading.threading_traits import BurstThreadPool
from simd_math import has_amx_int8
from simd_math.ops import gelu_tanh_f32
from kernels.helpers import (
    RangePartitionedKernel, Binding, fanout_dispatch, fanout_dispatch_per_rank,
    saturate_workers, BF16Ptr, I32Ptr,
)
from kernels.elementwise import swiglu_oai_activate
from kernels.moe_router import SparseRoute, SparseRoutePtr
from kernels.profiling import Profiler

from butterquant.convert import store_bf16
from butterquant.gemm import accumulate_tiles_grouped, vnni_grouped_fits
from butterquant.dot_products import vnni_colsum_correct
from butterquant.vnni import VNNI_N_STEP
from butterquant.types import F32Ptr, I8Ptr
from butterquant.amx_tiles import AMX_TILE_M, AMX_TILE_N, AMX_K_STEP
from butterquant.amx_gemm import amx_panel_1x32, amx_panel_2x32, AMX_MIN_ROWS
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


@fieldwise_init
struct Phase1RoutePermuteKernel[hidden: Int](RangePartitionedKernel):
    """Materialize a route-ordered, contiguous activation slab so each expert's
    records form an AMX-loadable M-tile. record `rec` (route order) copies
    token `routes[rec].token`'s row out of the per-token activation. Local read,
    local write; the gather cost is O(records * hidden) bytes versus the
    O(records * hidden * inter) MAC contraction, i.e. ~1/inter of the GEMM."""
    var x_i8: I8Ptr
    var x_sa: F32Ptr
    var routes: SparseRoutePtr
    var act_routed: I8Ptr
    var sa_routed: F32Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        comptime width = simd_width_of[DType.int8]()
        for rec in range(self.start, self.end):
            var tok = Int(self.routes[rec].token)
            var src = self.x_i8 + tok * Self.hidden
            var dst = self.act_routed + rec * Self.hidden
            var j = 0
            while j + width <= Self.hidden:
                (dst + j).store((src + j).load[width=width]())
                j += width
            while j < Self.hidden:
                dst[j] = src[j]
                j += 1
            self.sa_routed[rec] = self.x_sa[tok]

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_phase1_route_permute[
    P: BurstThreadPool, o: ImmutOrigin, Profile: Bool, N: Int, //,
    hidden: Int, max_worker_count: Int = 128,
](
    act: ButterquantActivation[o],
    expert_offset: Binding[Int32, o],
    routes: Binding[SparseRoute, o],
    act_routed: Binding[Int8, o],
    sa_routed: Binding[Float32, o],
    experts_per_rank: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime Kern = Phase1RoutePermuteKernel[hidden]
    var epr = experts_per_rank

    @parameter
    def make(r: Int) -> Kern:
        return Kern(act.data[r], act.scale[r], routes[r],
                    act_routed[r], sa_routed[r], 0, 0)

    @parameter
    def total_for(r: Int) -> Int:
        return Int(expert_offset[r][epr])

    @parameter
    def bytes_for(r: Int) -> Int:
        return Int(expert_offset[r][epr]) * hidden

    _ = fanout_dispatch_per_rank[
        make, total_for, bytes_for,
        max_worker_count=max_worker_count,
        worker_policy=saturate_workers,
        label="phase1_route_permute",
    ](pools, prof)


def amx_phase1_gate_up_tile[
    hidden: Int, inter: Int, n_inter_tiles: Int,
    activation: StaticString, alpha: Float32, limit: Float32,
](
    it: Int,
    act: I8Ptr,
    act_scale: F32Ptr,
    w: I8Ptr,
    wsc: F32Ptr,
    n_tok: Int,
    bucket: BF16Ptr,
):
    comptime width = simd_width_of[DType.int32]()
    comptime acc_count = VNNI_N_STEP // width
    var gate_base_n = it * VNNI_N_STEP
    var up_base_n = (n_inter_tiles + it) * VNNI_N_STEP

    var gate_buf = InlineArray[Float32, 2 * AMX_TILE_M * VNNI_N_STEP](
        uninitialized=True)
    var up_buf = InlineArray[Float32, 2 * AMX_TILE_M * VNNI_N_STEP](
        uninitialized=True)
    var gp = UnsafePointer(to=gate_buf).bitcast[Float32]()
    var up = UnsafePointer(to=up_buf).bitcast[Float32]()
    var panel_base = 0
    var pbp = UnsafePointer(to=panel_base)

    @parameter
    def gate_write(row: Int, n_base: Int, res: SIMD[DType.float32, width]):
        (gp + (row - pbp[]) * VNNI_N_STEP + (n_base - gate_base_n)).store(res)

    @parameter
    def up_write(row: Int, n_base: Int, res: SIMD[DType.float32, width]):
        (up + (row - pbp[]) * VNNI_N_STEP + (n_base - up_base_n)).store(res)

    @parameter
    def emit(rows: Int):
        for rl in range(rows):
            var brow = bucket + (pbp[] + rl) * inter + it * VNNI_N_STEP
            comptime for a in range(acc_count):
                var g = (gp + rl * VNNI_N_STEP + a * width).load[width=width]()
                var u = (up + rl * VNNI_N_STEP + a * width).load[width=width]()
                var rv = bq_phase1_activate[width, activation, alpha, limit](g, u)
                store_bf16[width](rv, brow + a * width)

    var m_panel = 0
    while m_panel + 2 * AMX_TILE_M <= n_tok:
        panel_base = m_panel
        amx_panel_2x32[hidden, gate_write](
            act, m_panel, hidden, act_scale, w, wsc, it)
        amx_panel_2x32[hidden, up_write](
            act, m_panel, hidden, act_scale, w, wsc, n_inter_tiles + it)
        emit(2 * AMX_TILE_M)
        m_panel += 2 * AMX_TILE_M
    while m_panel + AMX_TILE_M <= n_tok:
        panel_base = m_panel
        amx_panel_1x32[hidden, gate_write](
            act, m_panel, AMX_TILE_M, hidden, act_scale, w, wsc, it)
        amx_panel_1x32[hidden, up_write](
            act, m_panel, AMX_TILE_M, hidden, act_scale, w, wsc,
            n_inter_tiles + it)
        emit(AMX_TILE_M)
        m_panel += AMX_TILE_M
    if m_panel < n_tok:
        panel_base = n_tok - AMX_TILE_M
        amx_panel_1x32[hidden, gate_write](
            act, panel_base, AMX_TILE_M, hidden, act_scale, w, wsc, it)
        amx_panel_1x32[hidden, up_write](
            act, panel_base, AMX_TILE_M, hidden, act_scale, w, wsc,
            n_inter_tiles + it)
        emit(AMX_TILE_M)


@fieldwise_init
struct BqPhase1GateUpActAmxKernel[
    hidden: Int, gate_up: Int, inter: Int, MR: Int,
    activation: StaticString, alpha: Float32, limit: Float32,
](RangePartitionedKernel):
    var x_i8: I8Ptr
    var x_sa: F32Ptr
    var act_routed: I8Ptr
    var sa_routed: F32Ptr
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

            comptime if has_amx_int8():
                if n_tok >= AMX_MIN_ROWS:
                    var act = self.act_routed + rec_lo * Self.hidden
                    var sa = self.sa_routed + rec_lo
                    var bkt = self.hidden_bucket + rec_lo * Self.inter
                    for it in range(self.tile_start, self.tile_end):
                        amx_phase1_gate_up_tile[
                            Self.hidden, Self.inter, n_inter_tiles,
                            Self.activation, Self.alpha, Self.limit,
                        ](it, act, sa, w, wsc, n_tok, bkt)
                    continue

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


def dispatch_bq_phase1_gate_up_act_amx[
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
    act_routed: Binding[Int8, o],
    sa_routed: Binding[Float32, o],
    experts_per_rank: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime assert quant_vnni_packed[quant](), "bq phase1 consumes VNNI-packed experts"
    comptime assert quant_has_colsum[quant](), "bq phase1 requires a colsum sidecar"
    comptime n_inter_tiles = inter // VNNI_N_STEP

    dispatch_phase1_route_permute[
        hidden=hidden, max_worker_count=max_worker_count,
    ](act, expert_offset, routes, act_routed, sa_routed,
      experts_per_rank, pools, prof)

    comptime Kern = BqPhase1GateUpActAmxKernel[
        hidden, gate_up, inter, MR, activation, alpha, limit]
    var cs = experts_gate_up.colsum_checked()

    @parameter
    def make(r: Int) -> Kern:
        return Kern(act.data[r], act.scale[r], act_routed[r], sa_routed[r],
                    expert_offset[r], routes[r],
                    experts_gate_up.data[r], experts_gate_up.scale[r], cs[r],
                    hidden_bucket[r], experts_per_rank, 0, 0)

    fanout_dispatch[
        make,
        max_worker_count=max_worker_count,
        worker_policy=saturate_workers,
        label="bq_phase1_gate_up_act_amx",
    ](pools, prof, n_inter_tiles, experts_per_rank * gate_up * hidden)


def dispatch_bq_m3_phase1_gate_up_amx[
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
    act_routed: Binding[Int8, o],
    sa_routed: Binding[Float32, o],
    experts_per_rank: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    dispatch_bq_phase1_gate_up_act_amx[
        hidden=hidden, gate_up=gate_up, inter=inter,
        activation="swiglu_oai", alpha=M3_SWIGLU_ALPHA, limit=M3_SWIGLU_LIMIT,
        MR=MR, max_worker_count=max_worker_count,
    ](act, expert_offset, routes, experts_gate_up, hidden_bucket,
      act_routed, sa_routed, experts_per_rank, pools, prof)


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
