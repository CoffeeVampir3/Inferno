from std.pathlib import Path
from std.memory import Span, UnsafePointer
from std.time import perf_counter_ns
from simd_math.ops import sqrt

from numa import NumaArena, NumaTopology
from threading import BurstPool
from threading.threading_traits import BurstThreadPool
from kernels.helpers import RankView, Binding, prime_fp_environment
from kernels.attention_ops import KVRunTable, pow2_shift, flash_partial_stride
from kernels.profiling import Profiler
from kernels.reductions import dispatch_allreduce_inplace
from kernels.rmsnorm import dispatch_rms_norm, fused_norm_residual_add
from kernels.elementwise import dispatch_gelu_gate_up, dispatch_scalar_mul
from kernels.flash_sample import (
    SamplingParams, SampleAccum, SampleOutcome,
)
from kernels.logsum_merge import MergeSegment
from kernels.moe_router import (
    RouterCandidate, SparseRoute,
    dispatch_router_expert, dispatch_merge_router_candidates,
    dispatch_build_expert_schedules,
)
from butterquant_kernels import (
    dispatch_bq_embed_lookup, dispatch_bq_norm_quant, dispatch_bq_qkv,
    dispatch_bq_linear, dispatch_bq_attn_prep,
    dispatch_bq_sliding_attention, dispatch_bq_full_attention,
    dispatch_bq_block_quant, dispatch_bq_block_linear,
    dispatch_bq_phase1_gate_up, dispatch_bq_phase2_down,
    dispatch_bq_head_prep, dispatch_bq_flash_sample,
)
from butterquant import (
    PackColsumTask, dispatch_pack_colsum,
    bake_split_gain_in_place, ButterquantActivation, ButterquantBlockActivation,
)
from butterquant.amx_tiles import prime_amx_environment
from modeling.temporal_scratch import (
    ScratchBuffer, ShardedScratchBuffer, ScratchIsland, ScratchPhase,
    ScratchPhaseOrder, ScaleClass,
    TemporalScratchPool, ScratchPlan,
    derive_checked_plan, aggregate_scratch_peak,
)

from modeling.model_spec import (
    BF16, F32, I8,
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
from modeling.slot import (
    Slot, BindContext, emit_pack_tasks,
)
from modeling.kv_policy import (
    KVPoolMirror, pool_specs, dispatch_prefix_copies, bind_pool_run_table,
)
from modeling.gemma4_topology import (
    MAX_WORKERS, PAGE_LEN, CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM,
    SLIDING_POOL, FULL_POOL, SLIDING_RING_PAGES,
    Gemma4Recipes, KVSlotGroup,
    Gemma4Shapes, Gemma4TailShapes,
    SlidingLayerRefs, FullLayerRefs, BodyRefs, TailRefs,
    Gemma4Layout,
    gemma4_kv_mirrors, gemma4_bake_router_scales, gemma4_init_rope_tables,
    gemma4_load_arenas,
)
from quant.recipe import (
    QuantRecipe, PerRowQuant, PerBlockQuant, SoftmaxRouterCenter,
    SplitGamma, NoGamma, SingleSided, PerRowCs, PerBlockCs, NoColsum,
    VnniPacked, RowMajor,
)
from quant.quantizer import Quantizer
from continuous_batching.schedule import (
    Schedule, ScheduledModel, MAXIMUM_SAMPLING_LOGITS,
)
from continuous_batching.paging import (
    KVPageAccountant, BatchGeometry,
)


comptime C = Gemma4BaseConfig


comptime SplitGainPerRowCs[fwht: Int, gamma: StaticString]: QuantRecipe = PerRowQuant(
    fwht, SplitGamma(gamma), SingleSided(), PerRowCs(), VnniPacked(),
)


comptime PlainPerRowBlockCs[fwht: Int]: QuantRecipe = PerRowQuant(
    fwht, NoGamma(), SingleSided(), PerBlockCs(), VnniPacked(),
)


comptime TiedHeadEmbed[fwht: Int]: QuantRecipe = PerBlockQuant(
    fwht, NoGamma(), SingleSided(), NoColsum(), RowMajor(),
)


struct ButterquantRecipes(Gemma4Recipes):
    comptime FFN_BLOCK = C.DOWN_FWHT_BLOCK
    comptime SlidingQkv: QuantRecipe = SplitGainPerRowCs[
        128, "input_layernorm.weight"]
    comptime SlidingOut: QuantRecipe = PlainPerRowBlockCs[C.HEAD_DIM_SLIDING]
    comptime FullQkv: QuantRecipe = SplitGainPerRowCs[
        128, "input_layernorm.weight"]
    comptime FullOut: QuantRecipe = PlainPerRowBlockCs[C.HEAD_DIM_FULL]
    comptime DenseGateUp: QuantRecipe = SplitGainPerRowCs[
        128, "pre_feedforward_layernorm.weight"]
    comptime DenseDown: QuantRecipe = PlainPerRowBlockCs[C.DOWN_FWHT_BLOCK]
    comptime Router: QuantRecipe = SoftmaxRouterCenter()
    comptime MoeGateUp: QuantRecipe = SplitGainPerRowCs[
        128, "pre_feedforward_layernorm_2.weight"]
    comptime MoeDown: QuantRecipe = PlainPerRowBlockCs[C.DOWN_FWHT_BLOCK]
    comptime Embed: QuantRecipe = TiedHeadEmbed[128]


struct SlidingKVSlots[max_resident_seqs: Int](
    Copyable, ImplicitlyCopyable, KVSlotGroup,
):
    comptime CacheShape = TensorColumnSharded[
        Self.max_resident_seqs * SLIDING_RING_PAGES * PAGE_LEN, C.KV_DIM_SLIDING,
    ]
    comptime ScaleShape = TensorColumnSharded[
        Self.max_resident_seqs * SLIDING_RING_PAGES * PAGE_LEN,
        C.NUM_KV_HEADS_SLIDING,
    ]
    var k:       Slot[I8,  Self.CacheShape]
    var k_scale: Slot[F32, Self.ScaleShape]
    var v:       Slot[I8,  Self.CacheShape]
    var v_scale: Slot[F32, Self.ScaleShape]


struct FullKVSlots[batching_seq_len: Int](
    Copyable, ImplicitlyCopyable, KVSlotGroup,
):
    comptime CacheShape = ContextRowSharded[Self.batching_seq_len, C.KV_DIM_FULL]
    comptime ScaleShape = ContextRowSharded[
        Self.batching_seq_len, C.NUM_KV_HEADS_FULL,
    ]
    var k:       Slot[I8,  Self.CacheShape]
    var k_scale: Slot[F32, Self.ScaleShape]
    var v:       Slot[I8,  Self.CacheShape]
    var v_scale: Slot[F32, Self.ScaleShape]


comptime R = ButterquantRecipes
comptime SH = Gemma4Shapes[R.FFN_BLOCK]
comptime Layout[
    max_seq_len: Int, batching_seq_len: Int, max_resident_seqs: Int,
    steer_vectors: Int, measure_rows: Int,
] = Gemma4Layout[
    ButterquantRecipes,
    SlidingKVSlots[max_resident_seqs],
    FullKVSlots[batching_seq_len],
    max_seq_len, steer_vectors, measure_rows,
]


comptime SLIDING_NUM_Q_MAX = C.Q_DIM_SLIDING // C.HEAD_DIM_SLIDING
comptime SLIDING_PARTIAL_STRIDE_MAX = flash_partial_stride(
    SLIDING_NUM_Q_MAX, C.HEAD_DIM_SLIDING)
comptime FULL_NUM_Q = C.Q_DIM_FULL // C.HEAD_DIM_FULL
comptime FULL_PARTIAL_STRIDE = flash_partial_stride(FULL_NUM_Q, C.HEAD_DIM_FULL)
comptime MOE_NB_DOWN = C.MOE_INTERMEDIATE // C.DOWN_FWHT_BLOCK
comptime HEAD_NB = C.HIDDEN // 128


@fieldwise_init
struct Gemma4SlidingScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder[
        "norm_quant", "qkv", "attn_prep", "attention", "o_prep",
    ]

    var x_i8_band: ScratchPhase["norm_quant", "qkv"]
    var x_i8: ScratchBuffer[Int8, C.SLIDING_WINDOW * C.HIDDEN]
    var x_sa: ScratchBuffer[Float32, C.SLIDING_WINDOW]
    var x_row_workspace_band: ScratchPhase["norm_quant", "norm_quant"]
    var x_row_workspace: ScratchBuffer[Float32, C.HIDDEN, ScaleClass.PER_WORKER]

    var q_band: ScratchPhase["qkv", "o_prep"]
    var q: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.Q_DIM_SLIDING, ScaleClass.PER_DEGREE,
    ]

    var kv_band: ScratchPhase["qkv", "attn_prep"]
    var kv: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.KV_DIM_SLIDING * 2, ScaleClass.PER_DEGREE,
    ]

    var qprep_band: ScratchPhase["attn_prep", "attention"]
    var q_i8: ScratchBuffer[
        Int8, C.SLIDING_WINDOW * C.Q_DIM_SLIDING, ScaleClass.PER_DEGREE,
    ]
    var qi_bias: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * SLIDING_NUM_Q_MAX, ScaleClass.PER_DEGREE,
    ]
    var f_q: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * SLIDING_NUM_Q_MAX, ScaleClass.PER_DEGREE,
    ]

    var partials_band: ScratchPhase["attention", "attention"]
    var partials: ScratchBuffer[
        Float32, SLIDING_PARTIAL_STRIDE_MAX, ScaleClass.PER_WORKER,
    ]

    var o_band: ScratchPhase["o_prep", "o_prep"]
    var o_i8: ScratchBuffer[
        Int8, C.SLIDING_WINDOW * C.Q_DIM_SLIDING, ScaleClass.PER_DEGREE,
    ]
    var o_sa: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * SLIDING_NUM_Q_MAX, ScaleClass.PER_DEGREE,
    ]
    var o_row_workspace: ScratchBuffer[
        Float32, C.Q_DIM_SLIDING, ScaleClass.PER_WORKER,
    ]


@fieldwise_init
struct Gemma4FullScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder[
        "norm_quant", "qkv", "attn_prep", "flash", "merge", "o_prep",
    ]

    var x_i8_band: ScratchPhase["norm_quant", "qkv"]
    var x_i8: ScratchBuffer[Int8, C.SLIDING_WINDOW * C.HIDDEN]
    var x_sa: ScratchBuffer[Float32, C.SLIDING_WINDOW]
    var x_row_workspace_band: ScratchPhase["norm_quant", "norm_quant"]
    var x_row_workspace: ScratchBuffer[Float32, C.HIDDEN, ScaleClass.PER_WORKER]

    var q_band: ScratchPhase["qkv", "attn_prep"]
    var q: ScratchBuffer[BFloat16, C.SLIDING_WINDOW * C.Q_DIM_FULL]

    var kv_band: ScratchPhase["qkv", "attn_prep"]
    var kv: ScratchBuffer[BFloat16, C.SLIDING_WINDOW * C.KV_DIM_FULL * 2]

    var qprep_band: ScratchPhase["attn_prep", "flash"]
    var q_i8: ScratchBuffer[Int8, C.SLIDING_WINDOW * C.Q_DIM_FULL]
    var qi_bias: ScratchBuffer[Float32, C.SLIDING_WINDOW * FULL_NUM_Q]
    var f_q: ScratchBuffer[Float32, C.SLIDING_WINDOW * FULL_NUM_Q]

    var partials_band: ScratchPhase["flash", "merge"]
    var partials: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * FULL_PARTIAL_STRIDE,
    ]

    var merge_segments_band: ScratchPhase["merge", "merge"]
    var merge_segments: ScratchBuffer[
        MergeSegment, 1, ScaleClass.PER_WORKER_PER_DEGREE,
    ]

    var q_local_band: ScratchPhase["merge", "o_prep"]
    var q_local: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.Q_DIM_FULL, ScaleClass.PER_DEGREE,
    ]

    var o_band: ScratchPhase["o_prep", "o_prep"]
    var o_i8: ScratchBuffer[
        Int8, C.SLIDING_WINDOW * C.Q_DIM_FULL, ScaleClass.PER_DEGREE,
    ]
    var o_sa: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * FULL_NUM_Q, ScaleClass.PER_DEGREE,
    ]
    var o_row_workspace: ScratchBuffer[
        Float32, C.Q_DIM_FULL, ScaleClass.PER_WORKER,
    ]


@fieldwise_init
struct Gemma4FfnMoeScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder[
        "dense_norm", "dense_gate_up", "dense_down_quant",
        "router_select", "moe_norm", "build_schedules", "phase1",
        "bucket_quant", "phase2", "dense_down_post",
    ]

    var dense_x_band: ScratchPhase["dense_norm", "dense_gate_up"]
    var dense_x_i8: ScratchBuffer[Int8, C.SLIDING_WINDOW * C.HIDDEN]
    var dense_x_sa: ScratchBuffer[Float32, C.SLIDING_WINDOW]
    var dense_x_row_workspace_band: ScratchPhase["dense_norm", "dense_norm"]
    var dense_x_row_workspace: ScratchBuffer[
        Float32, C.HIDDEN, ScaleClass.PER_WORKER,
    ]

    var ffn_gate_band: ScratchPhase["dense_gate_up", "dense_down_quant"]
    var ffn_gate: ShardedScratchBuffer[
        BFloat16, C.SLIDING_WINDOW, SH.GateUp,
    ]

    var ffn_up_band: ScratchPhase["dense_gate_up", "dense_gate_up"]
    var ffn_up: ShardedScratchBuffer[
        BFloat16, C.SLIDING_WINDOW, SH.GateUp,
    ]

    var dense_gate_band: ScratchPhase["dense_down_quant", "dense_down_post"]
    var dense_gate_i8: ShardedScratchBuffer[
        Int8, C.SLIDING_WINDOW, SH.GateUp,
    ]
    var dense_gate_sa: ShardedScratchBuffer[
        Float32, C.SLIDING_WINDOW, SH.GateUp, C.DOWN_FWHT_BLOCK,
    ]
    var dense_gate_row_workspace_band: ScratchPhase[
        "dense_down_quant", "dense_down_quant",
    ]
    var dense_gate_row_workspace: ScratchBuffer[
        Float32, C.INTERMEDIATE, ScaleClass.PER_WORKER,
    ]

    var router_workspace: ScratchPhase["router_select", "router_select"]
    var moe_router_scaled: ScratchBuffer[Float32, C.HIDDEN, ScaleClass.PER_WORKER]

    var router_cands: ScratchPhase["router_select", "router_select"]
    var moe_cands: ScratchBuffer[
        RouterCandidate, C.SLIDING_WINDOW * C.TOP_K, ScaleClass.PER_WORKER,
    ]

    var router_products: ScratchPhase["router_select", "build_schedules"]
    var moe_route_idx: ScratchBuffer[Int32, C.SLIDING_WINDOW * C.TOP_K]
    var moe_route_w: ScratchBuffer[Float32, C.SLIDING_WINDOW * C.TOP_K]

    var expert_input: ScratchPhase["moe_norm", "phase1"]
    var moe_x_i8: ScratchBuffer[Int8, C.SLIDING_WINDOW * C.HIDDEN]
    var moe_x_sa: ScratchBuffer[Float32, C.SLIDING_WINDOW]
    var moe_x_row_workspace_band: ScratchPhase["moe_norm", "moe_norm"]
    var moe_x_row_workspace: ScratchBuffer[
        Float32, C.HIDDEN, ScaleClass.PER_WORKER,
    ]

    var schedule_products: ScratchPhase["build_schedules", "phase2"]
    var moe_expert_offset: ScratchBuffer[Int32, C.NUM_EXPERTS + 1]
    var moe_routes: ScratchBuffer[SparseRoute, C.SLIDING_WINDOW * C.TOP_K]

    var hidden_bucket: ScratchPhase["phase1", "bucket_quant"]
    var moe_hidden_bucket: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.TOP_K * C.MOE_INTERMEDIATE,
    ]

    var bucket_i8_band: ScratchPhase["bucket_quant", "phase2"]
    var moe_bucket_i8: ScratchBuffer[
        Int8, C.SLIDING_WINDOW * C.TOP_K * C.MOE_INTERMEDIATE,
    ]
    var moe_bucket_sa: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * C.TOP_K * MOE_NB_DOWN,
    ]
    var bucket_row_workspace_band: ScratchPhase["bucket_quant", "bucket_quant"]
    var bucket_row_workspace: ScratchBuffer[
        Float32, C.MOE_INTERMEDIATE, ScaleClass.PER_WORKER,
    ]

    var phase2_accum: ScratchPhase["phase2", "phase2"]
    var moe_accum: ScratchBuffer[Float32, C.SLIDING_WINDOW * C.HIDDEN]

    var dense_band: ScratchPhase["dense_down_post", "dense_down_post"]
    var ffn_dense_out: ScratchBuffer[BFloat16, C.SLIDING_WINDOW * C.HIDDEN]


@fieldwise_init
struct Gemma4HeadScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder["embed", "sample"]

    var embed_row_workspace_band: ScratchPhase["embed", "embed"]
    var embed_row_workspace: ScratchBuffer[
        Float32, C.HIDDEN, ScaleClass.PER_WORKER,
    ]

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
    var head_x_i8: ScratchBuffer[
        Int8,
        CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM * C.HIDDEN, ScaleClass.FIXED,
    ]
    var head_x_sa: ScratchBuffer[
        Float32,
        CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM * HEAD_NB, ScaleClass.FIXED,
    ]
    var head_row_workspace: ScratchBuffer[
        Float32, C.HIDDEN, ScaleClass.PER_WORKER,
    ]


@fieldwise_init
struct Gemma4ForwardScratch(Copyable, ImplicitlyCopyable):
    var sliding: Gemma4SlidingScratch
    var full: Gemma4FullScratch
    var ffn: Gemma4FfnMoeScratch
    var head: Gemma4HeadScratch


def calculate_peak_scratch(degree: Int, max_workers: Int) -> Int:
    return aggregate_scratch_peak[Gemma4ForwardScratch](degree, max_workers)


def dispatch_bq_sliding_attention_qkv[
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
    act: ButterquantActivation[o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
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

    dispatch_bq_qkv[
        hidden=C.HIDDEN, qn_full=C.Q_DIM_SLIDING, kvn_full=C.KV_DIM_SLIDING,
        max_worker_count=max_worker_count,
    ](
        act,
        attn.q_proj.bq_weight(attn_ctx),
        attn.k_proj.bq_weight(attn_ctx),
        attn.v_proj.bq_weight(attn_ctx),
        q_outs, k_outs, v_outs, q_rows, kv_rows, seq_len, pools, prof)

    var q_i8 = scratch.binding[Gemma4SlidingScratch, "q_i8"](ctx, plan)
    var qi_bias = scratch.binding[Gemma4SlidingScratch, "qi_bias"](ctx, plan)
    var f_q = scratch.binding[Gemma4SlidingScratch, "f_q"](ctx, plan)

    var kv_ctx = ctx.with_layer(layout.sliding_kv.base(layer_idx))
    var k_cache = layout.sliding_kv.proto.k.binding(kv_ctx)
    var k_scale = layout.sliding_kv.proto.k_scale.binding(kv_ctx)
    var v_cache = layout.sliding_kv.proto.v.binding(kv_ctx)
    var v_scale = layout.sliding_kv.proto.v_scale.binding(kv_ctx)

    comptime page_shift = pow2_shift(PAGE_LEN)
    comptime row_mask = PAGE_LEN - 1
    comptime page_mask = cache_size // PAGE_LEN - 1

    dispatch_bq_attn_prep[
        head_dim=head_dim, rope_half=rope_half, pair_stride=head_dim // 2,
        sqrt_n=sqrt_hd, n_eps=hd_eps,
        max_worker_count=max_worker_count,
    ](q_outs, k_outs, v_outs,
      attn.q_norm.binding(attn_ctx), attn.k_norm.binding(attn_ctx),
      q_i8, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
      layout.sliding_rope.cos.state_binding(ctx),
      layout.sliding_rope.sin.state_binding(ctx),
      runs, num_q_heads, num_kv_heads, 1,
      page_shift, row_mask, page_mask, seq_len, pools, prof)

    var partials = scratch.binding[Gemma4SlidingScratch, "partials"](ctx, plan)

    dispatch_bq_sliding_attention[
        head_dim=head_dim, max_q=max_q, gqa_ratio=gqa_ratio,
        window=C.SLIDING_WINDOW, cache_size=cache_size, page_len=PAGE_LEN,
        max_worker_count=max_worker_count,
    ](q_i8, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
      q_outs, partials, runs, num_q_heads, num_kv_heads, partial_stride,
      kv_rows, seq_len, pools, prof)

    var o_i8 = scratch.binding[Gemma4SlidingScratch, "o_i8"](ctx, plan)
    var o_sa = scratch.binding[Gemma4SlidingScratch, "o_sa"](ctx, plan)
    var o_row_workspace = scratch.binding[
        Gemma4SlidingScratch, "o_row_workspace"](ctx, plan)

    dispatch_bq_block_quant[
        block=head_dim, apply_fwht=False, max_worker_count=max_worker_count,
    ](q_outs, o_i8, o_sa, o_row_workspace, q_rows, seq_len, pools, prof)

    var o_act = ButterquantBlockActivation(o_i8, o_sa)
    var xs = layout.activations.x_residual.state_binding(ctx)

    dispatch_bq_block_linear[
        n_rows=C.HIDDEN, max_worker_count=max_worker_count,
    ](o_act, attn.o_proj.bq_weight(attn_ctx), xs, q_rows, seq_len, pools, prof)


def dispatch_bq_full_attention_qkv[
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
    act: ButterquantActivation[o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
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

    dispatch_bq_linear[hidden=C.HIDDEN, max_worker_count=max_worker_count](
        act, attn.q_proj.bq_weight(attn_ctx), q_outs, q_rows, seq_len, pools, prof)
    dispatch_bq_linear[hidden=C.HIDDEN, max_worker_count=max_worker_count](
        act, attn.k_proj.bq_weight(attn_ctx), k_outs, k_rows, seq_len, pools, prof)

    var q_i8 = scratch.binding[Gemma4FullScratch, "q_i8"](ctx, plan)
    var qi_bias = scratch.binding[Gemma4FullScratch, "qi_bias"](ctx, plan)
    var f_q = scratch.binding[Gemma4FullScratch, "f_q"](ctx, plan)

    var kv_ctx = ctx.with_layer(layout.full_kv.base(layer_idx))
    var k_cache = layout.full_kv.proto.k.binding(kv_ctx)
    var k_scale = layout.full_kv.proto.k_scale.binding(kv_ctx)
    var v_cache = layout.full_kv.proto.v.binding(kv_ctx)
    var v_scale = layout.full_kv.proto.v_scale.binding(kv_ctx)

    var rows_per_page = PAGE_LEN // degree
    var page_shift = pow2_shift(rows_per_page)
    var row_mask = rows_per_page - 1

    dispatch_bq_attn_prep[
        head_dim=head_dim, rope_half=rope_half, pair_stride=pair_stride,
        sqrt_n=sqrt_hd, n_eps=hd_eps,
        max_worker_count=max_worker_count,
    ](q_outs, k_outs, k_outs,
      attn.q_norm.binding(attn_ctx), attn.k_norm.binding(attn_ctx),
      q_i8, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
      layout.full_rope.cos.state_binding(ctx),
      layout.full_rope.sin.state_binding(ctx),
      runs, num_q_heads, num_kv_heads, degree,
      page_shift, row_mask, -1, seq_len, pools, prof)

    var q_local = scratch.binding[Gemma4FullScratch, "q_local"](ctx, plan)
    var partials = scratch.binding[Gemma4FullScratch, "partials"](ctx, plan)
    var merge_segments = scratch.binding[
        Gemma4FullScratch, "merge_segments"](ctx, plan)

    dispatch_bq_full_attention[
        head_dim=head_dim, num_q=num_q_heads, num_kv=num_kv_heads,
        gqa_ratio=gqa_ratio, kv_stride=k_rows, partial_stride=partial_stride,
        page_len=PAGE_LEN,
        max_worker_count=max_worker_count,
    ](q_i8, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
      q_local, partials, merge_segments, runs, local_num_q_heads,
      seq_len, pools, prof)

    var o_i8 = scratch.binding[Gemma4FullScratch, "o_i8"](ctx, plan)
    var o_sa = scratch.binding[Gemma4FullScratch, "o_sa"](ctx, plan)
    var o_row_workspace = scratch.binding[
        Gemma4FullScratch, "o_row_workspace"](ctx, plan)

    dispatch_bq_block_quant[
        block=head_dim, apply_fwht=False, max_worker_count=max_worker_count,
    ](q_local, o_i8, o_sa, o_row_workspace, local_q_rows, seq_len, pools, prof)

    var o_act = ButterquantBlockActivation(o_i8, o_sa)
    var xs = layout.activations.x_residual.state_binding(ctx)

    dispatch_bq_block_linear[
        n_rows=C.HIDDEN, max_worker_count=max_worker_count,
    ](o_act, attn.o_proj.bq_weight(attn_ctx), xs, local_q_rows, seq_len, pools, prof)


def dispatch_bq_moe[
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

    var router_scaled = scratch.binding[Gemma4FfnMoeScratch, "moe_router_scaled"](ctx, plan)
    var cands = scratch.binding[Gemma4FfnMoeScratch, "moe_cands"](ctx, plan)
    var route_idx = scratch.binding[Gemma4FfnMoeScratch, "moe_route_idx"](ctx, plan)
    var route_w = scratch.binding[Gemma4FfnMoeScratch, "moe_route_w"](ctx, plan)
    var expert_offset = scratch.binding[Gemma4FfnMoeScratch, "moe_expert_offset"](ctx, plan)
    var routes = scratch.binding[Gemma4FfnMoeScratch, "moe_routes"](ctx, plan)
    var moe_x_i8 = scratch.binding[Gemma4FfnMoeScratch, "moe_x_i8"](ctx, plan)
    var moe_x_sa = scratch.binding[Gemma4FfnMoeScratch, "moe_x_sa"](ctx, plan)
    var moe_x_row_workspace = scratch.binding[
        Gemma4FfnMoeScratch, "moe_x_row_workspace"](ctx, plan)
    var bucket = scratch.binding[Gemma4FfnMoeScratch, "moe_hidden_bucket"](ctx, plan)
    var bucket_i8 = scratch.binding[Gemma4FfnMoeScratch, "moe_bucket_i8"](ctx, plan)
    var bucket_sa = scratch.binding[Gemma4FfnMoeScratch, "moe_bucket_sa"](ctx, plan)
    var bucket_row_workspace = scratch.binding[
        Gemma4FfnMoeScratch, "bucket_row_workspace"](ctx, plan)
    var moe_accum = scratch.binding[Gemma4FfnMoeScratch, "moe_accum"](ctx, plan)

    var nws = dispatch_router_expert[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        top_k=C.TOP_K, max_worker_count=max_worker_count,
    ](x_input, body.router_proj.bq_router(ctx).centered,
      body.router_scale.binding(ctx), router_scaled, cands,
      experts_per_rank, seq_len, pools, prof)

    dispatch_merge_router_candidates[
        C.TOP_K, max_worker_count=max_worker_count,
    ](cands, nws, body.router_pes.binding(ctx), route_idx, route_w,
      seq_len, pools, prof)

    dispatch_bq_norm_quant[
        hidden=C.HIDDEN, block=128, sqrt_n=sqrt_n, n_eps=n_eps,
        max_worker_count=max_worker_count,
    ](x_input, body.pre_ffn_norm_2.binding(ctx),
      moe_x_i8, moe_x_sa, moe_x_row_workspace, seq_len, pools, prof)

    dispatch_build_expert_schedules[
        C.NUM_EXPERTS, C.TOP_K, max_worker_count=max_worker_count,
    ](route_idx, route_w, expert_offset, routes,
      experts_per_rank, seq_len, pools, prof)

    var moe_act = ButterquantActivation(moe_x_i8, moe_x_sa)
    dispatch_bq_phase1_gate_up[
        hidden=C.HIDDEN, gate_up=C.MOE_GATE_UP_FUSED,
        inter=C.MOE_INTERMEDIATE, max_worker_count=max_worker_count,
    ](moe_act, expert_offset, routes,
      body.experts_gate_up.bq_weight(ctx), bucket, experts_per_rank, pools, prof)

    var num_routes = seq_len * C.TOP_K
    dispatch_bq_block_quant[
        block=C.DOWN_FWHT_BLOCK, apply_fwht=True,
        max_worker_count=max_worker_count,
    ](bucket, bucket_i8, bucket_sa, bucket_row_workspace,
      C.MOE_INTERMEDIATE, num_routes, pools, prof)

    var bucket_act = ButterquantBlockActivation(bucket_i8, bucket_sa)
    dispatch_bq_phase2_down[
        hidden=C.HIDDEN, inter=C.MOE_INTERMEDIATE,
        max_worker_count=max_worker_count,
    ](bucket_act, expert_offset, routes,
      body.experts_down.bq_weight(ctx), moe_accum, moe_out,
      experts_per_rank, seq_len, pools, prof)

    dispatch_allreduce_inplace[
        BF16, max_worker_count=max_worker_count,
    ](moe_out, seq_len * C.HIDDEN, pools, prof)


def dispatch_bq_ffn[
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

    var dense_x_i8 = scratch.binding[Gemma4FfnMoeScratch, "dense_x_i8"](ctx, plan)
    var dense_x_sa = scratch.binding[Gemma4FfnMoeScratch, "dense_x_sa"](ctx, plan)
    var dense_x_row_workspace = scratch.binding[
        Gemma4FfnMoeScratch, "dense_x_row_workspace"](ctx, plan)
    var gate = scratch.binding[Gemma4FfnMoeScratch, "ffn_gate"](ctx, plan)
    var up = scratch.binding[Gemma4FfnMoeScratch, "ffn_up"](ctx, plan)
    var dense_gate_i8 = scratch.binding[Gemma4FfnMoeScratch, "dense_gate_i8"](ctx, plan)
    var dense_gate_sa = scratch.binding[Gemma4FfnMoeScratch, "dense_gate_sa"](ctx, plan)
    var dense_gate_row_workspace = scratch.binding[
        Gemma4FfnMoeScratch, "dense_gate_row_workspace"](ctx, plan)
    var dense_out = scratch.binding[Gemma4FfnMoeScratch, "ffn_dense_out"](ctx, plan)

    dispatch_bq_norm_quant[
        hidden=C.HIDDEN, block=128, sqrt_n=sqrt_n, n_eps=n_eps,
        max_worker_count=max_worker_count,
    ](x_main, body.pre_ffn_norm.binding(ctx),
      dense_x_i8, dense_x_sa, dense_x_row_workspace, seq_len, pools, prof)

    var dense_act = ButterquantActivation(dense_x_i8, dense_x_sa)
    dispatch_bq_linear[hidden=C.HIDDEN, max_worker_count=max_worker_count](
        dense_act, body.gate_proj.bq_weight(ctx), gate,
        intermediate_per_rank, seq_len, pools, prof)
    dispatch_bq_linear[hidden=C.HIDDEN, max_worker_count=max_worker_count](
        dense_act, body.up_proj.bq_weight(ctx), up,
        intermediate_per_rank, seq_len, pools, prof)

    dispatch_gelu_gate_up[max_worker_count=max_worker_count](
        gate, up, gate, intermediate_per_rank, seq_len, pools, prof)

    dispatch_bq_block_quant[
        block=C.DOWN_FWHT_BLOCK, apply_fwht=True,
        max_worker_count=max_worker_count,
    ](gate, dense_gate_i8, dense_gate_sa, dense_gate_row_workspace,
      intermediate_per_rank, seq_len, pools, prof)

    dispatch_bq_moe[max_worker_count=max_worker_count](
        body, ctx, x_main, x_residual, seq_len, scratch, plan, pools, prof)

    var dense_gate_act = ButterquantBlockActivation(dense_gate_i8, dense_gate_sa)
    dispatch_bq_block_linear[
        n_rows=C.HIDDEN, max_worker_count=max_worker_count,
    ](dense_gate_act, body.down_proj.bq_weight(ctx), dense_out,
      intermediate_per_rank, seq_len, pools, prof)

    dispatch_allreduce_inplace[
        BF16, max_worker_count=max_worker_count,
    ](dense_out, seq_len * C.HIDDEN, pools, prof)

    dispatch_rms_norm[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        max_worker_count=max_worker_count,
    ](dense_out, dense_out, body.post_ffn_norm_1.binding(ctx), seq_len, pools, prof)

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
        self.tokens_processed = 0

    def init_state(mut self):
        var tasks = List[PackColsumTask]()
        for i in range(C.NUM_LAYERS):
            var entry = LAYER_SCHEDULE[i]
            if entry.kind == LayerKind.FULL:
                _ = emit_pack_tasks[FullLayerRefs[R]](
                    self.layout.full.base(entry.local_idx),
                    self.degree, tasks)
            else:
                _ = emit_pack_tasks[SlidingLayerRefs[R]](
                    self.layout.sliding.base(entry.local_idx),
                    self.degree, tasks)
        _ = emit_pack_tasks[TailRefs[R]](
            self.layout.tail.base(0), self.degree, tasks)

        var nodes = List[Int]()
        for r in range(self.degree):
            nodes.append(self.arenas[r].node)
        var noprof = Profiler[False]()
        dispatch_pack_colsum[max_worker_count=MAX_WORKERS](
            self.pools, noprof, self.arena_bases, nodes, tasks)

    def model_init(mut self):
        ref layout = self.layout

        prime_fp_environment(self.pools)
        prime_amx_environment(self.pools)

        for rank in range(self.degree):
            var base = self.arena_bases[rank]
            for i in range(C.NUM_LAYERS):
                var entry = LAYER_SCHEDULE[i]
                if entry.kind == LayerKind.FULL:
                    var lb = base + layout.full.base(entry.local_idx)
                    ref fbody = layout.full.proto.body
                    bake_split_gain_in_place(fbody.input_norm.at(lb), C.HIDDEN)
                    bake_split_gain_in_place(fbody.pre_ffn_norm.at(lb), C.HIDDEN)
                    bake_split_gain_in_place(fbody.pre_ffn_norm_2.at(lb), C.HIDDEN)
                else:
                    var lb = base + layout.sliding.base(entry.local_idx)
                    ref sbody = layout.sliding.proto.body
                    bake_split_gain_in_place(sbody.input_norm.at(lb), C.HIDDEN)
                    bake_split_gain_in_place(sbody.pre_ffn_norm.at(lb), C.HIDDEN)
                    bake_split_gain_in_place(sbody.pre_ffn_norm_2.at(lb), C.HIDDEN)

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
        var embed_row_workspace = self.scratch.binding[
            Gemma4HeadScratch, "embed_row_workspace",
        ](ctx, self.head_plan)
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
        var head_x_i8 = self.scratch.binding[
            Gemma4HeadScratch, "head_x_i8",
        ](ctx, self.head_plan)
        var head_x_sa = self.scratch.binding[
            Gemma4HeadScratch, "head_x_sa",
        ](ctx, self.head_plan)
        var head_row_workspace = self.scratch.binding[
            Gemma4HeadScratch, "head_row_workspace",
        ](ctx, self.head_plan)

        var buf_starts = pack_slot_starts(schedule)
        if self.steer.armed:
            self.steer.record_step(schedule, buf_starts, num_slots)
        self.run_prefix_copies(schedule)
        self.bind_step_runs(schedule, pages)
        var full_runs = UnsafePointer(to=self.full_runs).as_unsafe_any_origin()
        var sliding_runs = UnsafePointer(to=self.sliding_runs).as_unsafe_any_origin()

        dispatch_bq_embed_lookup[
            hidden=C.HIDDEN, scale=embed_scale,
        ](Span(schedule.tokens),
          layout.tail.proto.embed.bq_weight(tail_ctx),
          x_main_ranks, embed_row_workspace, shard_rows, total,
          self.pools, self.profiler)
        dispatch_allreduce_inplace[BF16](
            x_main_ranks, total * C.HIDDEN, self.pools, self.profiler)

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

            if entry.kind == LayerKind.FULL:
                var fx_i8 = self.scratch.binding[
                    Gemma4FullScratch, "x_i8",
                ](ctx, self.full_plan)
                var fx_sa = self.scratch.binding[
                    Gemma4FullScratch, "x_sa",
                ](ctx, self.full_plan)
                var fx_row_workspace = self.scratch.binding[
                    Gemma4FullScratch, "x_row_workspace",
                ](ctx, self.full_plan)
                dispatch_bq_norm_quant[
                    hidden=C.HIDDEN, block=128, sqrt_n=sqrt_n, n_eps=n_eps,
                ](x_main_ranks, body.input_norm.binding(layer_ctx),
                  fx_i8, fx_sa, fx_row_workspace,
                  total, self.pools, self.profiler)
                var full_act = ButterquantActivation(fx_i8, fx_sa)
                dispatch_bq_full_attention_qkv[
                    max_seq_len=Self.max_seq_len,
                    batching_seq_len=Self.batching_seq_len,
                    max_resident_seqs=Self.max_resident_seqs,
                ](
                    layout, ctx, full_act, full_runs, total, entry.local_idx,
                    self.scratch, self.full_plan, self.pools, self.profiler)
            else:
                var sx_i8 = self.scratch.binding[
                    Gemma4SlidingScratch, "x_i8",
                ](ctx, self.sliding_plan)
                var sx_sa = self.scratch.binding[
                    Gemma4SlidingScratch, "x_sa",
                ](ctx, self.sliding_plan)
                var sx_row_workspace = self.scratch.binding[
                    Gemma4SlidingScratch, "x_row_workspace",
                ](ctx, self.sliding_plan)
                dispatch_bq_norm_quant[
                    hidden=C.HIDDEN, block=128, sqrt_n=sqrt_n, n_eps=n_eps,
                ](x_main_ranks, body.input_norm.binding(layer_ctx),
                  sx_i8, sx_sa, sx_row_workspace,
                  total, self.pools, self.profiler)
                var sl_act = ButterquantActivation(sx_i8, sx_sa)
                dispatch_bq_sliding_attention_qkv[
                    max_seq_len=Self.max_seq_len,
                    batching_seq_len=Self.batching_seq_len,
                    max_resident_seqs=Self.max_resident_seqs,
                ](
                    layout, ctx, sl_act, sliding_runs, total, entry.local_idx,
                    self.scratch, self.sliding_plan, self.pools, self.profiler)

            dispatch_allreduce_inplace[BF16](
                x_res_ranks, total * C.HIDDEN, self.pools, self.profiler)

            fused_norm_residual_add[
                hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
            ](x_res_ranks, x_main_ranks, x_main_ranks,
              body.post_attn_norm.binding(layer_ctx),
              total, self.pools, self.profiler)

            dispatch_bq_ffn(
                body, layer_ctx, x_main_ranks, x_res_ranks, total,
                self.scratch, self.ffn_plan, self.pools, self.profiler)

            if self.steer.armed:
                apply_steer_ops[hidden=C.HIDDEN](
                    self.steer,
                    layout.steer.vectors.state_binding(ctx),
                    schedule, buf_starts, x_main_ranks,
                    num_slots, total, i, self.pools, self.profiler)

        var outcomes = List[SampleOutcome[MAXIMUM_SAMPLING_LOGITS]]()
        var emit_plan = collect_emit_plan(schedule, buf_starts)
        var num_emit = emit_plan.count()

        if num_emit > 0:
            debug_assert(
                num_emit <= CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM,
                "execute emit count exceeds parallelism cap",
            )
            var x_head = stage_sampling_inputs[hidden=C.HIDDEN](
                emit_plan, schedule, x_main_ranks, head_x,
                emit_rows, sample_params, self.pools, self.profiler)

            var head_act = ButterquantActivation(head_x_i8, head_x_sa)
            dispatch_bq_head_prep[
                hidden=C.HIDDEN, block=128, sqrt_n=sqrt_n, n_eps=n_eps,
            ](x_head, layout.tail.proto.final_norm.binding(tail_ctx),
              head_act, head_row_workspace, num_emit,
              self.pools, self.profiler)

            var out_ptr = outcome[0]
            dispatch_bq_flash_sample[
                cols=C.HIDDEN, cap=C.LOGIT_SOFTCAP,
                n_max=MAXIMUM_SAMPLING_LOGITS,
            ](head_act, layout.tail.proto.embed.bq_weight(tail_ctx),
              accums, sample_params, out_ptr, num_emit, vocab_per_rank,
              self.pools, self.profiler)

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
            ButterquantRecipes,
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
        model.init_state()
        model.model_init()
        return model^

    @staticmethod
    def quantize(
        source_dir: Path, output_path: Path,
        topo: NumaTopology, var pools: List[Self.Pool],
    ) -> Bool:
        var q = Quantizer(source_dir, output_path)
        if not q:
            return False
        for i in range(C.NUM_LAYERS):
            var entry = LAYER_SCHEDULE[i]
            var prefix = String(t"model.language_model.layers.{entry.idx}.")
            if entry.kind == LayerKind.FULL:
                if not q.plan_walk[FullLayerRefs[R]](prefix, entry.idx):
                    return False
            else:
                if not q.plan_walk[SlidingLayerRefs[R]](prefix, entry.idx):
                    return False
        if not q.plan_walk[TailRefs[R]](String(""), -1):
            return False
        if not q.write_header():
            return False
        return q.execute(topo, pools^)
