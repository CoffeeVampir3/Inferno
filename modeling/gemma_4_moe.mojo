from std.pathlib import Path
from std.memory import Span, UnsafePointer
from std.time import perf_counter_ns
from simd_math.ops import sqrt

from numa import NumaArena, NumaTopology
from threading import BurstPool
from threading.threading_traits import BurstThreadPool
from kernels.helpers import RankView, Binding, prime_fp_environment, copy_row
from kernels.attention_ops import KVRunTable, pow2_shift
from kernels.reductions import dispatch_allreduce_inplace
from kernels.embedding import dispatch_embed_lookup
from kernels.rmsnorm import dispatch_rms_norm, dispatch_rms_norm_qkv_heads
from kernels.rmsnorm import fused_norm_residual_add
from kernels.flash_sample import (
    SamplingParams, SampleAccum, SampleOutcome, dispatch_flash_sample,
)
from kernels.gemm import dispatch_gemm, dispatch_gemm_cols, dispatch_gemm_chained_qkv
from kernels.rope import dispatch_rope_cache_write
from kernels.attention_ops import flash_partial_stride
from kernels.attention_dispatch_kernels import (
    dispatch_sliding_attention, dispatch_full_attention,
)
from kernels.logsum_merge import MergeSegment
from kernels.moe_router import (
    RouterCandidate, SparseRoute,
    dispatch_router_expert, dispatch_merge_router_candidates,
    dispatch_build_expert_schedules,
)
from kernels.moe_experts import (
    dispatch_phase1_gate_up, dispatch_phase2_down,
)
from kernels.elementwise import dispatch_gelu_gate_up, dispatch_scalar_mul
from kernels.profiling import Profiler
from modeling.temporal_scratch import (
    ScratchBuffer, ScratchIsland, ScratchPhase, ScratchPhaseOrder, ScaleClass,
    TemporalScratchPool, ScratchPlan,
    derive_checked_plan, aggregate_scratch_peak,
)

from modeling.model_spec import (
    BF16,
    TensorColumnSharded, ContextRowSharded,
    DEFAULT_ALIGNMENT,
)
from modeling.gemma4_common import (
    Gemma4BaseConfig, LAYER_SCHEDULE, LayerKind,
)
from modeling.modeling_common import (
    pack_slot_starts, collect_emit_plan, stage_sampling_inputs,
)
from inspectable_toolkit.steer import (
    SteerState, Steerable, InjectOp, apply_steer_ops,
)
from inspectable_toolkit.measure import (
    MeasureState, accumulate_residual_mean,
    MEASURE_RESIDUAL, MEASURE_BASELINE, MEASURE_MODIFIED,
)
from inspectable_toolkit.flash_kl import dispatch_flash_kl
from modeling.slot import (
    Slot, BindContext,
)
from modeling.kv_policy import (
    KVPoolMirror, pool_specs, dispatch_prefix_copies, bind_pool_run_table,
)
from modeling.gemma4_topology import (
    MAX_WORKERS, PAGE_LEN, CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM,
    SLIDING_POOL, FULL_POOL, SLIDING_RING_PAGES,
    Gemma4Recipes, KVSlotGroup,
    Gemma4Shapes, Gemma4TailShapes,
    BodyRefs,
    Gemma4Layout,
    gemma4_kv_mirrors, gemma4_bake_router_scales, gemma4_init_rope_tables,
    gemma4_load_arenas,
)
from quant.recipe import QuantRecipe, Passthrough
from continuous_batching.schedule import (
    Schedule, ScheduledModel, MAXIMUM_SAMPLING_LOGITS,
)
from continuous_batching.paging import (
    KVPageAccountant, BatchGeometry,
)


comptime C = Gemma4BaseConfig


struct PassthroughRecipes(Gemma4Recipes):
    comptime FFN_BLOCK = 1
    comptime SlidingQkv: QuantRecipe = Passthrough()
    comptime SlidingOut: QuantRecipe = Passthrough()
    comptime FullQkv: QuantRecipe = Passthrough()
    comptime FullOut: QuantRecipe = Passthrough()
    comptime DenseGateUp: QuantRecipe = Passthrough()
    comptime DenseDown: QuantRecipe = Passthrough()
    comptime Router: QuantRecipe = Passthrough()
    comptime MoeGateUp: QuantRecipe = Passthrough()
    comptime MoeDown: QuantRecipe = Passthrough()
    comptime Embed: QuantRecipe = Passthrough()


struct SlidingKVSlots[max_resident_seqs: Int](
    Copyable, ImplicitlyCopyable, KVSlotGroup,
):
    comptime CacheShape = TensorColumnSharded[
        Self.max_resident_seqs * SLIDING_RING_PAGES * PAGE_LEN, C.KV_DIM_SLIDING,
    ]
    var k: Slot[BF16, Self.CacheShape]
    var v: Slot[BF16, Self.CacheShape]


struct FullKVSlots[batching_seq_len: Int](
    Copyable, ImplicitlyCopyable, KVSlotGroup,
):
    comptime CacheShape = ContextRowSharded[Self.batching_seq_len, C.KV_DIM_FULL]
    var k: Slot[BF16, Self.CacheShape]
    var v: Slot[BF16, Self.CacheShape]


comptime R = PassthroughRecipes
comptime SH = Gemma4Shapes[R.FFN_BLOCK]
comptime Layout[
    max_seq_len: Int, batching_seq_len: Int, max_resident_seqs: Int,
    steer_vectors: Int, measure_rows: Int,
] = Gemma4Layout[
    PassthroughRecipes,
    SlidingKVSlots[max_resident_seqs],
    FullKVSlots[batching_seq_len],
    max_seq_len, steer_vectors, measure_rows,
]


comptime SLIDING_NUM_Q_MAX = C.Q_DIM_SLIDING // C.HEAD_DIM_SLIDING
comptime SLIDING_PARTIAL_STRIDE_MAX = flash_partial_stride(
    SLIDING_NUM_Q_MAX, C.HEAD_DIM_SLIDING)
comptime FULL_NUM_Q = C.Q_DIM_FULL // C.HEAD_DIM_FULL
comptime FULL_PARTIAL_STRIDE = flash_partial_stride(FULL_NUM_Q, C.HEAD_DIM_FULL)


@fieldwise_init
struct Gemma4SlidingScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder["qkv", "attention", "o_proj"]

    var q_band: ScratchPhase["qkv", "o_proj"]
    var q: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.Q_DIM_SLIDING, ScaleClass.PER_DEGREE,
    ]

    var kv_band: ScratchPhase["qkv", "attention"]
    var kv: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.KV_DIM_SLIDING * 2, ScaleClass.PER_DEGREE,
    ]

    var partials_band: ScratchPhase["attention", "attention"]
    var partials: ScratchBuffer[
        Float32, SLIDING_PARTIAL_STRIDE_MAX, ScaleClass.PER_WORKER,
    ]


@fieldwise_init
struct Gemma4FullScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder["prep", "flash", "merge"]

    var q_band: ScratchPhase["prep", "flash"]
    var q: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.Q_DIM_FULL, ScaleClass.FIXED,
    ]

    var kv_band: ScratchPhase["prep", "prep"]
    var kv: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.KV_DIM_FULL * 2, ScaleClass.FIXED,
    ]

    var partials_band: ScratchPhase["flash", "merge"]
    var partials: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * FULL_PARTIAL_STRIDE, ScaleClass.FIXED,
    ]

    var q_local_band: ScratchPhase["merge", "merge"]
    var q_local: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.Q_DIM_FULL, ScaleClass.PER_DEGREE,
    ]

    var merge_segments_band: ScratchPhase["merge", "merge"]
    var merge_segments: ScratchBuffer[
        MergeSegment, 1, ScaleClass.PER_WORKER_PER_DEGREE,
    ]


@fieldwise_init
struct Gemma4FfnMoeScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASE1_TILE_J = 64
    comptime PHASE1_MR = 4

    comptime PHASES = ScratchPhaseOrder[
        "dense_gate_up", "router_select", "moe_setup",
        "phase1", "phase2", "dense_down_post",
    ]

    var ffn_gate_band: ScratchPhase["dense_gate_up", "dense_down_post"]
    var ffn_gate: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.INTERMEDIATE, ScaleClass.PER_DEGREE,
    ]

    var ffn_up_band: ScratchPhase["dense_gate_up", "dense_gate_up"]
    var ffn_up: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.INTERMEDIATE, ScaleClass.PER_DEGREE,
    ]

    var router_workspace: ScratchPhase["router_select", "router_select"]
    var moe_router_scaled: ScratchBuffer[Float32, C.HIDDEN, ScaleClass.PER_WORKER]

    var router_cands: ScratchPhase["router_select", "router_select"]
    var moe_cands: ScratchBuffer[
        RouterCandidate, C.SLIDING_WINDOW * C.TOP_K, ScaleClass.PER_WORKER,
    ]

    var router_products: ScratchPhase["router_select", "moe_setup"]
    var moe_route_idx: ScratchBuffer[Int32, C.SLIDING_WINDOW * C.TOP_K, ScaleClass.FIXED]
    var moe_route_w: ScratchBuffer[Float32, C.SLIDING_WINDOW * C.TOP_K, ScaleClass.FIXED]

    var expert_input: ScratchPhase["moe_setup", "phase1"]
    var moe_x_normed: ScratchBuffer[BFloat16, C.SLIDING_WINDOW * C.HIDDEN, ScaleClass.FIXED]

    var schedule_products: ScratchPhase["moe_setup", "phase2"]
    var moe_expert_offset: ScratchBuffer[Int32, C.NUM_EXPERTS + 1, ScaleClass.FIXED]
    var moe_routes: ScratchBuffer[SparseRoute, C.SLIDING_WINDOW * C.TOP_K, ScaleClass.FIXED]

    var hidden_bucket: ScratchPhase["phase1", "phase2"]
    var moe_hidden_bucket: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.TOP_K * C.MOE_INTERMEDIATE, ScaleClass.FIXED,
    ]

    var phase1_workspace: ScratchPhase["phase1", "phase1"]
    var moe_gate_scratch: ScratchBuffer[
        Float32, Self.PHASE1_MR * 2 * Self.PHASE1_TILE_J, ScaleClass.PER_WORKER,
    ]

    var phase2_accum: ScratchPhase["phase2", "phase2"]
    var moe_accum: ScratchBuffer[Float32, C.SLIDING_WINDOW * C.HIDDEN, ScaleClass.FIXED]

    var dense_band: ScratchPhase["dense_down_post", "dense_down_post"]
    var ffn_dense_out: ScratchBuffer[BFloat16, C.SLIDING_WINDOW * C.HIDDEN, ScaleClass.FIXED]


@fieldwise_init
struct Gemma4HeadScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder["sample"]

    var sample_band: ScratchPhase["sample", "sample"]
    var accums: ScratchBuffer[
        SampleAccum[MAXIMUM_SAMPLING_LOGITS],
        CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM, ScaleClass.PER_WORKER,
    ]
    var head_x: ScratchBuffer[
        BFloat16,
        CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM * C.HIDDEN, ScaleClass.FIXED,
    ]
    var emit_rows: ScratchBuffer[
        Int32,
        CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM, ScaleClass.FIXED,
    ]
    var sample_params: ScratchBuffer[
        SamplingParams,
        CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM, ScaleClass.FIXED,
    ]
    var outcome: ScratchBuffer[
        SampleOutcome[MAXIMUM_SAMPLING_LOGITS],
        CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM, ScaleClass.FIXED,
    ]
    var kl_accums: ScratchBuffer[
        Float32,
        CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM, ScaleClass.PER_WORKER,
    ]
    var kl_partials: ScratchBuffer[
        Float32,
        CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM, ScaleClass.FIXED,
    ]
    var mod_logz: ScratchBuffer[
        Float32,
        CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM, ScaleClass.FIXED,
    ]


@fieldwise_init
struct Gemma4ForwardScratch(Copyable, ImplicitlyCopyable):
    var sliding: Gemma4SlidingScratch
    var full: Gemma4FullScratch
    var ffn: Gemma4FfnMoeScratch
    var head: Gemma4HeadScratch


def calculate_peak_scratch(degree: Int, max_workers: Int) -> Int:
    return aggregate_scratch_peak[Gemma4ForwardScratch](degree, max_workers)


def dispatch_sliding_attention_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin,
    steer_vectors: Int, measure_rows: Int, //,
    max_seq_len: Int, batching_seq_len: Int, max_resident_seqs: Int,
    max_worker_count: Int = 128,
](
    layout: Layout[
        max_seq_len, batching_seq_len, max_resident_seqs, steer_vectors,
        measure_rows,
    ],
    ctx: BindContext[o],
    runs: UnsafePointer[KVRunTable, MutUntrackedOrigin],
    seq_len: Int,
    layer_idx: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var degree = ctx.degree()
    comptime head_dim = C.HEAD_DIM_SLIDING
    comptime gqa_ratio = C.NUM_HEADS // C.NUM_KV_HEADS_SLIDING
    comptime sqrt_hd = sqrt[DType.float32, 1](head_dim)
    comptime hd_eps = Float32(head_dim) * C.RMS_NORM_EPS
    comptime rope_half = C.ROPE_HALF_SLIDING
    comptime cache_size = SLIDING_RING_PAGES * PAGE_LEN
    comptime max_q = SLIDING_NUM_Q_MAX
    var q_rows = SH.SlidingQ.data_n(degree)
    var kv_rows = SH.SlidingKV.data_n(degree)
    var num_q_heads = q_rows // head_dim
    var num_kv_heads = kv_rows // head_dim
    var partial_stride = flash_partial_stride(num_q_heads, head_dim)

    debug_assert(
        seq_len <= C.SLIDING_WINDOW,
        "sliding attention chunk exceeds SLIDING_WINDOW",
    )

    var attn_ctx = ctx.with_layer(layout.sliding.base(layer_idx))
    var attn = layout.sliding.proto.attn

    var q_outs = scratch.binding[Gemma4SlidingScratch, "q"](ctx, plan)
    var k_outs = scratch.binding[Gemma4SlidingScratch, "kv"](ctx, plan)
    var v_outs = k_outs.shifted(seq_len * kv_rows)
    var xs = layout.activations.x_residual.state_binding(ctx)

    dispatch_gemm_chained_qkv[
        cols=C.HIDDEN, max_worker_count=max_worker_count,
    ](xs,
      attn.q_proj.binding(attn_ctx),
      attn.k_proj.binding(attn_ctx),
      attn.v_proj.binding(attn_ctx),
      q_outs, k_outs, v_outs, q_rows, kv_rows, seq_len, pools, prof)

    dispatch_rms_norm_qkv_heads[
        head_dim=head_dim, sqrt_n=sqrt_hd, n_eps=hd_eps,
        max_worker_count=max_worker_count,
    ](q_outs, q_outs, k_outs, k_outs, v_outs, v_outs,
      attn.q_norm.binding(attn_ctx),
      attn.k_norm.binding(attn_ctx),
      num_q_heads, num_kv_heads, seq_len, pools, prof)

    var kv_ctx = ctx.with_layer(layout.sliding_kv.base(layer_idx))
    var k_kv = layout.sliding_kv.proto.k.binding(kv_ctx)
    var v_kv = layout.sliding_kv.proto.v.binding(kv_ctx)

    comptime page_shift = pow2_shift(PAGE_LEN)
    comptime row_mask = PAGE_LEN - 1
    comptime page_mask = cache_size // PAGE_LEN - 1

    dispatch_rope_cache_write[
        half=rope_half, pair_stride=head_dim // 2,
        head_dim=head_dim,
        max_worker_count=max_worker_count,
    ](q_outs, k_outs, v_outs,
      k_kv, v_kv,
      layout.sliding_rope.cos.state_binding(ctx),
      layout.sliding_rope.sin.state_binding(ctx),
      runs, num_q_heads, num_kv_heads, 1,
      page_shift, row_mask, page_mask, seq_len, pools, prof)

    var partials = scratch.binding[Gemma4SlidingScratch, "partials"](ctx, plan)

    dispatch_sliding_attention[
        head_dim=head_dim, max_q=max_q, gqa_ratio=gqa_ratio,
        window=C.SLIDING_WINDOW, cache_size=cache_size, page_len=PAGE_LEN,
        max_worker_count=max_worker_count,
    ](q_outs, k_kv, v_kv, q_outs, partials, runs,
      num_q_heads, partial_stride, kv_rows, seq_len, pools, prof)

    dispatch_gemm_cols[
        rows=C.HIDDEN, max_worker_count=max_worker_count,
    ](
        q_outs,
        attn.o_proj.binding(attn_ctx),
        xs, q_rows, seq_len, pools, prof)


def dispatch_full_attention_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin,
    steer_vectors: Int, measure_rows: Int, //,
    max_seq_len: Int, batching_seq_len: Int, max_resident_seqs: Int,
    max_worker_count: Int = 128,
](
    layout: Layout[
        max_seq_len, batching_seq_len, max_resident_seqs, steer_vectors,
        measure_rows,
    ],
    ctx: BindContext[o],
    runs: UnsafePointer[KVRunTable, MutUntrackedOrigin],
    seq_len: Int,
    layer_idx: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var degree = ctx.degree()
    comptime head_dim = C.HEAD_DIM_FULL
    comptime q_rows = C.Q_DIM_FULL
    comptime k_rows = C.KV_DIM_FULL
    comptime num_q_heads = q_rows // head_dim
    comptime num_kv_heads = k_rows // head_dim
    comptime gqa_ratio = C.NUM_HEADS // C.NUM_KV_HEADS_FULL
    comptime sqrt_hd = sqrt[DType.float32, 1](head_dim)
    comptime hd_eps = Float32(head_dim) * C.RMS_NORM_EPS
    comptime rope_half = C.ROPE_HALF_FULL
    comptime pair_stride = head_dim // 2
    comptime partial_stride = FULL_PARTIAL_STRIDE
    var local_q_rows = SH.FullO.data_m(degree)
    var local_num_q_heads = local_q_rows // head_dim

    var attn_ctx = ctx.with_layer(layout.full.base(layer_idx))
    var attn = layout.full.proto.attn

    var q_outs = scratch.binding[Gemma4FullScratch, "q"](ctx, plan)
    var k_outs = scratch.binding[Gemma4FullScratch, "kv"](ctx, plan)
    var v_outs = k_outs.shifted(seq_len * k_rows)
    var xs = layout.activations.x_residual.state_binding(ctx)

    dispatch_gemm[
        cols=C.HIDDEN, max_worker_count=max_worker_count,
    ](xs, attn.q_proj.binding(attn_ctx), q_outs, q_rows, seq_len, pools, prof)
    dispatch_gemm[
        cols=C.HIDDEN, max_worker_count=max_worker_count,
    ](xs, attn.k_proj.binding(attn_ctx), k_outs, k_rows, seq_len, pools, prof)

    # Full attention reuses the raw K projection as V before K normalization.
    dispatch_rms_norm_qkv_heads[
        head_dim=head_dim, sqrt_n=sqrt_hd, n_eps=hd_eps,
        max_worker_count=max_worker_count,
    ](q_outs, q_outs, k_outs, k_outs, k_outs, v_outs,
      attn.q_norm.binding(attn_ctx),
      attn.k_norm.binding(attn_ctx),
      num_q_heads, num_kv_heads, seq_len, pools, prof)

    var kv_ctx = ctx.with_layer(layout.full_kv.base(layer_idx))
    var k_kv = layout.full_kv.proto.k.binding(kv_ctx)
    var v_kv = layout.full_kv.proto.v.binding(kv_ctx)

    var rows_per_page = PAGE_LEN // degree
    var page_shift = pow2_shift(rows_per_page)
    var row_mask = rows_per_page - 1

    dispatch_rope_cache_write[
        half=rope_half, pair_stride=pair_stride,
        head_dim=head_dim,
        max_worker_count=max_worker_count,
    ](q_outs, k_outs, v_outs,
      k_kv, v_kv,
      layout.full_rope.cos.state_binding(ctx),
      layout.full_rope.sin.state_binding(ctx),
      runs, num_q_heads, num_kv_heads, degree,
      page_shift, row_mask, -1, seq_len, pools, prof)

    var q_local_outs = scratch.binding[Gemma4FullScratch, "q_local"](ctx, plan)
    var partials = scratch.binding[Gemma4FullScratch, "partials"](ctx, plan)
    var merge_segments = scratch.binding[
        Gemma4FullScratch, "merge_segments",
    ](ctx, plan)

    dispatch_full_attention[
        head_dim=head_dim, num_q=num_q_heads, gqa_ratio=gqa_ratio,
        kv_stride=k_rows, partial_stride=partial_stride, page_len=PAGE_LEN,
        max_worker_count=max_worker_count,
    ](q_outs, k_kv, v_kv, q_local_outs, partials,
      merge_segments, runs, local_num_q_heads, seq_len, pools, prof)

    dispatch_gemm_cols[
        rows=C.HIDDEN, max_worker_count=max_worker_count,
    ](
        q_local_outs,
        attn.o_proj.binding(attn_ctx),
        xs, local_q_rows, seq_len, pools, prof)


def dispatch_moe[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    body: BodyRefs[R],
    ctx: BindContext[o],
    x_input: Binding[BFloat16, o],
    moe_out: Binding[BFloat16, o],
    seq_len: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var degree = ctx.degree()
    var experts_per_rank = C.NUM_EXPERTS // degree
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = Float32(C.HIDDEN) * C.RMS_NORM_EPS

    var x_normed = scratch.binding[Gemma4FfnMoeScratch, "moe_x_normed"](ctx, plan)
    var cands = scratch.binding[Gemma4FfnMoeScratch, "moe_cands"](ctx, plan)
    var router_scaled = scratch.binding[Gemma4FfnMoeScratch, "moe_router_scaled"](ctx, plan)
    var route_idx = scratch.binding[Gemma4FfnMoeScratch, "moe_route_idx"](ctx, plan)
    var route_w = scratch.binding[Gemma4FfnMoeScratch, "moe_route_w"](ctx, plan)
    var expert_offset = scratch.binding[Gemma4FfnMoeScratch, "moe_expert_offset"](ctx, plan)
    var routes = scratch.binding[Gemma4FfnMoeScratch, "moe_routes"](ctx, plan)
    var hidden_bucket = scratch.binding[Gemma4FfnMoeScratch, "moe_hidden_bucket"](ctx, plan)
    var moe_accum = scratch.binding[Gemma4FfnMoeScratch, "moe_accum"](ctx, plan)
    var gate_scratch = scratch.binding[Gemma4FfnMoeScratch, "moe_gate_scratch"](ctx, plan)

    var nws = dispatch_router_expert[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        top_k=C.TOP_K,
        max_worker_count=max_worker_count,
    ](x_input,
      body.router_proj.binding(ctx),
      body.router_scale.binding(ctx),
      router_scaled, cands, experts_per_rank, seq_len, pools, prof)

    dispatch_merge_router_candidates[
        C.TOP_K, max_worker_count=max_worker_count,
    ](cands, nws, body.router_pes.binding(ctx), route_idx, route_w,
      seq_len, pools, prof)

    dispatch_rms_norm[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        max_worker_count=max_worker_count,
    ](x_input, x_normed,
      body.pre_ffn_norm_2.binding(ctx), seq_len, pools, prof)

    dispatch_build_expert_schedules[
        C.NUM_EXPERTS, C.TOP_K, max_worker_count=max_worker_count,
    ](route_idx, route_w, expert_offset, routes,
      experts_per_rank, seq_len, pools, prof)

    dispatch_phase1_gate_up[
        hidden=C.HIDDEN, gate_up_fused=C.MOE_GATE_UP_FUSED,
        intermediate=C.MOE_INTERMEDIATE,
        tile_j=Gemma4FfnMoeScratch.PHASE1_TILE_J, MR=Gemma4FfnMoeScratch.PHASE1_MR,
        max_worker_count=max_worker_count,
    ](x_normed, expert_offset, routes,
      body.experts_gate_up.binding(ctx),
      gate_scratch, hidden_bucket, experts_per_rank, pools, prof)

    dispatch_phase2_down[
        hidden=C.HIDDEN, intermediate=C.MOE_INTERMEDIATE,
        max_worker_count=max_worker_count,
    ](expert_offset, routes, hidden_bucket,
      body.experts_down.binding(ctx),
      moe_accum, moe_out, experts_per_rank, seq_len, pools, prof)

    dispatch_allreduce_inplace[
        BF16, max_worker_count=max_worker_count,
    ](moe_out, seq_len * C.HIDDEN, pools, prof)


def dispatch_ffn[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    body: BodyRefs[R],
    ctx: BindContext[o],
    x_main: Binding[BFloat16, o],
    x_residual: Binding[BFloat16, o],
    seq_len: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var degree = ctx.degree()
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = Float32(C.HIDDEN) * C.RMS_NORM_EPS
    var intermediate_per_rank = SH.GateUp.data_n(degree)

    var layer_scalar_ptr = body.layer_scalar.at(ctx.layer_address())

    var gate = scratch.binding[Gemma4FfnMoeScratch, "ffn_gate"](ctx, plan)
    var up = scratch.binding[Gemma4FfnMoeScratch, "ffn_up"](ctx, plan)
    var dense_out = scratch.binding[Gemma4FfnMoeScratch, "ffn_dense_out"](ctx, plan)

    dispatch_rms_norm[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        max_worker_count=max_worker_count,
    ](x_main, x_residual,
      body.pre_ffn_norm.binding(ctx), seq_len, pools, prof)

    dispatch_gemm[
        cols=C.HIDDEN, max_worker_count=max_worker_count,
    ](x_residual, body.gate_proj.binding(ctx), gate, intermediate_per_rank, seq_len, pools, prof)

    dispatch_gemm[
        cols=C.HIDDEN, max_worker_count=max_worker_count,
    ](x_residual, body.up_proj.binding(ctx), up, intermediate_per_rank, seq_len, pools, prof)

    dispatch_gelu_gate_up[
        max_worker_count=max_worker_count,
    ](gate, up, gate, intermediate_per_rank, seq_len, pools, prof)

    dispatch_moe[max_worker_count=max_worker_count](
        body, ctx, x_main, x_residual, seq_len, scratch, plan, pools, prof)

    dispatch_gemm_cols[
        rows=C.HIDDEN, max_worker_count=max_worker_count,
    ](gate, body.down_proj.binding(ctx), dense_out, intermediate_per_rank, seq_len, pools, prof)

    dispatch_allreduce_inplace[
        BF16, max_worker_count=max_worker_count,
    ](dense_out, seq_len * C.HIDDEN, pools, prof)

    dispatch_rms_norm[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        max_worker_count=max_worker_count,
    ](dense_out, dense_out,
      body.post_ffn_norm_1.binding(ctx), seq_len, pools, prof)

    fused_norm_residual_add[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        max_worker_count=max_worker_count,
    ](x_residual, dense_out, dense_out,
      body.post_ffn_norm_2.binding(ctx), seq_len, pools, prof)

    fused_norm_residual_add[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        max_worker_count=max_worker_count,
    ](dense_out, x_main, x_main,
      body.post_ffn_norm.binding(ctx), seq_len, pools, prof)

    var ls_value = layer_scalar_ptr[0].cast[DType.float32]()
    dispatch_scalar_mul[
        hidden=C.HIDDEN, max_worker_count=max_worker_count,
    ](x_main, x_main, ls_value, seq_len, pools, prof)


struct Gemma4[
    max_seq_len: Int = 8192,
    batching_seq_len: Int = 8192,
    max_resident_seqs: Int = 4,
    Pool: BurstThreadPool = BurstPool[],
    steer_vectors: Int = 0,
    measure_rows: Int = 0,
    profile: Bool = False, profile_slots: Int = 64,
](Movable, ScheduledModel, Steerable):
    comptime POSITIONS_PER_PAGE = PAGE_LEN
    comptime STEER_VECTORS = Self.steer_vectors
    comptime Recipes = R

    var arenas: List[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var pools: List[Self.Pool]
    var layout: Layout[
        Self.max_seq_len, Self.batching_seq_len, Self.max_resident_seqs,
        Self.steer_vectors, Self.measure_rows,
    ]
    var scratch: TemporalScratchPool
    var arena_bases: List[Int]
    var degree: Int
    var kv_mirrors: List[KVPoolMirror]
    var sliding_plan: ScratchPlan
    var full_plan: ScratchPlan
    var ffn_plan: ScratchPlan
    var head_plan: ScratchPlan
    var sliding_runs: KVRunTable
    var full_runs: KVRunTable
    var profiler: Profiler[Self.profile, Self.profile_slots]
    var steer: SteerState
    var measure: MeasureState
    var tokens_processed: Int

    def __init__(out self,
        var arenas: List[NumaArena[alignment=DEFAULT_ALIGNMENT]],
        var pools: List[Self.Pool],
        layout: Layout[
            Self.max_seq_len, Self.batching_seq_len, Self.max_resident_seqs,
            Self.steer_vectors, Self.measure_rows,
        ],
        degree: Int,
        max_workers: Int,
    ):
        self.degree = degree
        self.arena_bases = List[Int]()
        for r in range(degree):
            self.arena_bases.append(Int(arenas[r].base.value()))
        self.layout = layout
        self.arenas = arenas^
        self.pools = pools^
        self.scratch = TemporalScratchPool(self.layout.arena.scratch_off)
        self.kv_mirrors = gemma4_kv_mirrors[
            batching_seq_len=Self.batching_seq_len,
            max_resident_seqs=Self.max_resident_seqs,
        ](self.layout, degree)
        self.sliding_plan = derive_checked_plan[Gemma4SlidingScratch](degree, max_workers)
        self.full_plan = derive_checked_plan[Gemma4FullScratch](degree, max_workers)
        self.ffn_plan = derive_checked_plan[Gemma4FfnMoeScratch](degree, max_workers)
        self.head_plan = derive_checked_plan[Gemma4HeadScratch](degree, max_workers)
        self.sliding_runs = KVRunTable()
        self.full_runs = KVRunTable()
        self.profiler = Profiler[Self.profile, Self.profile_slots]()
        self.steer = SteerState(CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM)
        self.measure = MeasureState()
        self.tokens_processed = 0

    def model_init(mut self):
        prime_fp_environment(self.pools)
        gemma4_bake_router_scales(self.layout, self.arena_bases)
        gemma4_init_rope_tables(self.layout, self.arena_bases)

    def set_steer_vector(mut self, idx: Int, read vec: List[BFloat16]):
        comptime if Self.steer_vectors > 0:
            for r in range(self.degree):
                var base = self.arena_bases[r]
                var p = self.layout.steer.vectors.at(base) + idx * C.HIDDEN
                for j in range(C.HIDDEN):
                    p[j] = vec[j]

    def set_inject_ops(mut self, var ops: List[InjectOp]):
        self.steer.set_inject(ops^)

    def disarm_steer(mut self):
        self.steer.disarm()

    def measure_residual(mut self, is_bad: Bool):
        self.measure.arm_residual(is_bad)

    def measure_baseline(mut self):
        self.measure.arm_baseline()

    def measure_modified(mut self):
        self.measure.arm_modified()

    def disarm_measure(mut self):
        self.measure.disarm()

    def reset_measure_directions(mut self):
        self.measure.reset_directions()

    def reset_measure_kl(mut self):
        self.measure.reset_kl()

    def refusal_directions(self) -> List[BFloat16]:
        return self.measure.finalize_directions()

    def measured_kl(self) -> Float64:
        return self.measure.kl_value()

    def batch_geometry(self) -> BatchGeometry:
        return BatchGeometry(
            max_seqs=Self.max_resident_seqs,
            max_slots=CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM,
            max_step_tokens=C.SLIDING_WINDOW,
            pools=pool_specs(self.kv_mirrors))

    def run_prefix_copies(mut self, read schedule: Schedule):
        dispatch_prefix_copies(
            self.kv_mirrors, schedule, self.arena_bases,
            self.pools, self.profiler)

    def bind_step_runs(
        mut self, read schedule: Schedule, read pages: KVPageAccountant,
    ):
        bind_pool_run_table(
            self.sliding_runs, schedule, pages,
            SLIDING_POOL, self.kv_mirrors[SLIDING_POOL])
        bind_pool_run_table(
            self.full_runs, schedule, pages,
            FULL_POOL, self.kv_mirrors[FULL_POOL])

    def execute(
        mut self,
        read schedule: Schedule,
        read pages: KVPageAccountant,
    ) -> List[SampleOutcome[MAXIMUM_SAMPLING_LOGITS]]:
        ref layout = self.layout
        var degree = self.degree
        comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
        comptime n_eps = Float32(C.HIDDEN) * C.RMS_NORM_EPS
        comptime embed_scale = Float64(sqrt[DType.float32, 1](C.HIDDEN)
            .cast[DType.bfloat16]().cast[DType.float32]())
        var vocab_per_rank = C.VOCAB_SIZE // degree
        var shard_rows = Gemma4TailShapes.Embed.data_n(degree)

        var wall_t0 = perf_counter_ns()
        var num_slots = len(schedule.slots)
        var total = len(schedule.tokens)
        debug_assert(num_slots > 0, "execute called with no slots")
        debug_assert(
            num_slots <= CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM,
            "execute slot count exceeds parallelism cap",
        )
        debug_assert(
            total <= C.SLIDING_WINDOW,
            "execute packed tokens exceed SLIDING_WINDOW",
        )
        self.tokens_processed += total

        var ctx = BindContext(RankView(Span(self.arena_bases)), 0)
        var tail_ctx = ctx.with_layer(layout.tail.base(0))

        var x_main_ranks = layout.activations.x_main.state_binding(ctx)
        var x_res_ranks = layout.activations.x_residual.state_binding(ctx)
        var accums = self.scratch.binding[
            Gemma4HeadScratch, "accums",
        ](ctx, self.head_plan)
        var sample_params = self.scratch.binding[
            Gemma4HeadScratch, "sample_params",
        ](ctx, self.head_plan)
        var head_x = self.scratch.binding[
            Gemma4HeadScratch, "head_x",
        ](ctx, self.head_plan)
        var emit_rows = self.scratch.binding[
            Gemma4HeadScratch, "emit_rows",
        ](ctx, self.head_plan)
        var outcome = self.scratch.binding[
            Gemma4HeadScratch, "outcome",
        ](ctx, self.head_plan)

        var buf_starts = pack_slot_starts(schedule)
        if self.steer.armed or self.measure.armed():
            self.steer.record_step(schedule, buf_starts, num_slots)
        var emit_plan = collect_emit_plan(schedule, buf_starts)
        var num_emit = emit_plan.count()
        self.run_prefix_copies(schedule)
        self.bind_step_runs(schedule, pages)
        var full_runs = UnsafePointer(to=self.full_runs).unsafe_origin_cast[
            MutUntrackedOrigin]()
        var sliding_runs = UnsafePointer(to=self.sliding_runs).unsafe_origin_cast[
            MutUntrackedOrigin]()

        dispatch_embed_lookup[
            hidden=C.HIDDEN, scale=embed_scale,
        ](Span(schedule.tokens),
          layout.tail.proto.embed.binding(tail_ctx),
          x_main_ranks, shard_rows, total, self.pools, self.profiler)
        dispatch_allreduce_inplace[BF16](
            x_main_ranks, total * C.HIDDEN, self.pools, self.profiler)

        if self.measure.mode == MEASURE_RESIDUAL:
            accumulate_residual_mean[hidden=C.HIDDEN](
                x_main_ranks, emit_plan.rows, num_emit,
                0, self.measure.acc_ptr(), self.measure.scratch_ptr())
            if self.measure.current_is_bad:
                self.measure.bad_count += num_emit
            else:
                self.measure.good_count += num_emit

        for i in range(C.NUM_LAYERS):
            if schedule.fully_cancelled():
                return List[SampleOutcome[MAXIMUM_SAMPLING_LOGITS]]()
            var entry = LAYER_SCHEDULE[i]
            var lb: Int
            var body: BodyRefs[R]
            if entry.kind == LayerKind.FULL:
                lb = layout.full.base(entry.local_idx)
                body = layout.full.proto.body
            else:
                lb = layout.sliding.base(entry.local_idx)
                body = layout.sliding.proto.body
            var layer_ctx = ctx.with_layer(lb)

            dispatch_rms_norm[
                hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
            ](x_main_ranks, x_res_ranks,
              body.input_norm.binding(layer_ctx),
              total, self.pools, self.profiler)

            if entry.kind == LayerKind.FULL:
                dispatch_full_attention_qkv[
                    max_seq_len=Self.max_seq_len,
                    batching_seq_len=Self.batching_seq_len,
                    max_resident_seqs=Self.max_resident_seqs,
                ](
                    layout, ctx, full_runs, total, entry.local_idx,
                    self.scratch, self.full_plan, self.pools, self.profiler)
            else:
                dispatch_sliding_attention_qkv[
                    max_seq_len=Self.max_seq_len,
                    batching_seq_len=Self.batching_seq_len,
                    max_resident_seqs=Self.max_resident_seqs,
                ](
                    layout, ctx, sliding_runs, total, entry.local_idx,
                    self.scratch, self.sliding_plan, self.pools, self.profiler)

            dispatch_allreduce_inplace[BF16](
                x_res_ranks, total * C.HIDDEN, self.pools, self.profiler)

            fused_norm_residual_add[
                hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
            ](x_res_ranks, x_main_ranks, x_main_ranks,
              body.post_attn_norm.binding(layer_ctx),
              total, self.pools, self.profiler)

            dispatch_ffn(
                body, layer_ctx, x_main_ranks, x_res_ranks, total,
                self.scratch, self.ffn_plan, self.pools, self.profiler)

            if self.steer.armed:
                apply_steer_ops[hidden=C.HIDDEN](
                    self.steer,
                    layout.steer.vectors.state_binding(ctx),
                    schedule, buf_starts, x_main_ranks,
                    num_slots, total, i, self.pools, self.profiler)

            if self.measure.mode == MEASURE_RESIDUAL:
                accumulate_residual_mean[hidden=C.HIDDEN](
                    x_main_ranks, emit_plan.rows,
                    num_emit, i + 1, self.measure.acc_ptr(),
                    self.measure.scratch_ptr())

        var outcomes = List[SampleOutcome[MAXIMUM_SAMPLING_LOGITS]]()

        if num_emit > 0:
            debug_assert(
                num_emit <= CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM,
                "execute emit count exceeds parallelism cap",
            )
            var x_head = stage_sampling_inputs[hidden=C.HIDDEN](
                emit_plan, schedule, x_main_ranks, head_x,
                emit_rows, sample_params, self.pools, self.profiler)

            dispatch_rms_norm[
                hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
            ](x_head, x_head,
              layout.tail.proto.final_norm.binding(tail_ctx),
              num_emit, self.pools, self.profiler)

            var out_ptr = outcome[0]
            dispatch_flash_sample[
                cols=C.HIDDEN, cap=C.LOGIT_SOFTCAP,
                n_max=MAXIMUM_SAMPLING_LOGITS,
            ](x_head, layout.tail.proto.embed.binding(tail_ctx),
              accums, sample_params, out_ptr, num_emit, vocab_per_rank,
              self.pools, self.profiler)

            if self.measure.mode == MEASURE_BASELINE:
                var bh = layout.measure.base_head.state_binding(ctx)
                var bl = layout.measure.base_logz.state_binding(ctx)
                var ofs = self.measure.base_row_offset
                for r in range(degree):
                    for j in range(num_emit):
                        copy_row[C.HIDDEN](
                            x_head[r] + j * C.HIDDEN,
                            bh[r] + (ofs + j) * C.HIDDEN)
                    var dl = bl[r] + ofs
                    for j in range(num_emit):
                        (dl + j)[] = (out_ptr + j)[].logz
                self.measure.base_row_offset += num_emit
            elif self.measure.mode == MEASURE_MODIFIED:
                var bh = layout.measure.base_head.state_binding(ctx)
                var bl = layout.measure.base_logz.state_binding(ctx)
                var ofs = self.measure.base_row_offset
                var kl_accums = self.scratch.binding[
                    Gemma4HeadScratch, "kl_accums",
                ](ctx, self.head_plan)
                var kl_partials = self.scratch.binding[
                    Gemma4HeadScratch, "kl_partials",
                ](ctx, self.head_plan)
                var mod_logz = self.scratch.binding[
                    Gemma4HeadScratch, "mod_logz",
                ](ctx, self.head_plan)
                for r in range(degree):
                    var ml = mod_logz[r]
                    for j in range(num_emit):
                        (ml + j)[] = (out_ptr + j)[].logz
                dispatch_flash_kl[
                    cols=C.HIDDEN, cap=C.LOGIT_SOFTCAP,
                ](x_head, bh.shifted(ofs * C.HIDDEN),
                  layout.tail.proto.embed.binding(tail_ctx),
                  bl.shifted(ofs), mod_logz, kl_accums, kl_partials,
                  num_emit, vocab_per_rank, self.pools, self.profiler)
                var p0 = kl_partials[0]
                var ksum = Float64(0)
                for j in range(num_emit):
                    ksum += Float64((p0 + j)[])
                self.measure.kl_sum += ksum
                self.measure.kl_rows += num_emit
                self.measure.base_row_offset += num_emit

            for j in range(num_emit):
                outcomes.append((out_ptr + j)[])

        self.profiler.add_wall(Int(perf_counter_ns() - wall_t0))
        return outcomes^

    @staticmethod
    def load(
        dir_path: Path,
        topo: NumaTopology,
        var pools: List[Self.Pool],
    ) -> Optional[Self]:
        var degree = len(pools)
        var max_workers = 0
        for r in range(degree):
            var cap = min(MAX_WORKERS, pools[r].get_capacity())
            if cap > max_workers:
                max_workers = cap

        var arenas = List[NumaArena[alignment=DEFAULT_ALIGNMENT]](capacity=degree)
        var layout_opt = gemma4_load_arenas[
            PassthroughRecipes,
            SlidingKVSlots[Self.max_resident_seqs],
            FullKVSlots[Self.batching_seq_len],
            Self.max_seq_len, Self.batching_seq_len, Self.max_resident_seqs,
            Self.steer_vectors, Self.measure_rows,
        ](dir_path, topo, degree, max_workers,
          calculate_peak_scratch(degree, max_workers), arenas)
        if not layout_opt:
            return None

        var model = Self(
            arenas^, pools^, layout_opt.take(), degree, max_workers)
        model.model_init()
        return model^
