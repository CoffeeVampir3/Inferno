from std.os import abort
from std.pathlib import Path
from std.memory import Span, UnsafePointer
from std.time import perf_counter_ns
from simd_math.ops import sqrt

from numa import NumaArena, NumaTopology
from threading import BurstPool
from threading.threading_traits import BurstThreadPool
from kernels.helpers import RankView, Binding, prime_fp_environment
from kernels.attention_ops import KVRunTable, pow2_shift
from kernels.flash_sample import SampleOutcome, SampleAccum, SamplingParams
from kernels.logsum_merge import MergeSegment
from kernels.moe_router import SparseRoute, dispatch_build_expert_schedules
from kernels.profiling import Profiler
from kernels.reductions import dispatch_allreduce_inplace
from kernels.embedding import dispatch_embed_lookup
from kernels.rmsnorm import dispatch_rms_norm
from kernels.gemm import dispatch_gemm
from kernels.rope import dispatch_rope_k_cache_write
from kernels.elementwise import dispatch_residual_add, dispatch_gate_up_act

from butterquant import (
    PackColsumTask, dispatch_pack_colsum, bake_split_gain_in_place,
    ButterquantActivation, ButterquantBlockActivation,
)
from butterquant.amx_tiles import prime_amx_environment
from butterquant_kernels.linear import (
    dispatch_bq_norm_quant, dispatch_bq_qkv, dispatch_bq_linear,
    dispatch_bq_block_quant, dispatch_bq_block_linear,
)
from butterquant_kernels.attention import (
    dispatch_bq_attn_prep, dispatch_bq_full_attention,
)
from butterquant_kernels.moe import dispatch_bq_phase2_down
from butterquant_kernels.head import (
    dispatch_bq_head_prep, dispatch_bq_flash_sample,
)
from prototypes.sigmoid_router import (
    M3RouterCandidate, dispatch_minimax_m3_router,
)
from prototypes.lightning_indexer import dispatch_minimax_m3_indexer
from prototypes.bq_sparse_attention import (
    dispatch_bq_minimax_m3_sparse_attention,
)
from prototypes.bq_moe_phase1 import dispatch_bq_m3_phase1_gate_up

from modeling.temporal_scratch import (
    ScratchBuffer, ScratchIsland, ScratchPhase, ScratchPhaseOrder, ScaleClass,
    TemporalScratchPool, ScratchPlan,
    derive_checked_plan, aggregate_scratch_peak,
)
from modeling.model_spec import (
    BF16, F32, I8, ContextRowSharded, DEFAULT_ALIGNMENT,
)
from modeling.modeling_common import (
    pack_slot_starts, collect_emit_plan, stage_sampling_inputs,
)
from modeling.slot import Slot, BindContext, emit_pack_tasks
from modeling.gemma4_topology import KVSlotGroup
from modeling.kv_policy import (
    KVPoolMirror, pool_specs, dispatch_prefix_copies, bind_pool_run_table,
)
from quant.recipe import (
    QuantRecipe, Passthrough, PerRowQuant, PerBlockQuant, RouterCenter,
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

from modeling.minimax_common import (
    MinimaxM3Config, LAYER_SCHEDULE, LayerKind,
)
from modeling.minimax_topology import (
    MAX_WORKERS, PAGE_LEN, CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM,
    FULL_POOL, INDEX_POOL, FULL_PARTIAL_STRIDE, NORM_GAIN_OFFSET,
    MinimaxM3Recipes, MinimaxM3Shapes, MinimaxM3TailShapes,
    AttnRefs, IndexerRefs, DenseMlpRefs, MoeRefs,
    DenseLayerRefs, SparseLayerRefs, TailRefs,
    MinimaxM3Layout,
    minimax_m3_load_arenas, minimax_m3_kv_mirrors, minimax_m3_init_rope_tables,
)


comptime C = MinimaxM3Config


comptime SplitGainPerRowCs[fwht: Int, gamma: StaticString]: QuantRecipe = PerRowQuant(
    fwht, SplitGamma(gamma, NORM_GAIN_OFFSET), SingleSided(), PerRowCs(), VnniPacked(),
)


comptime PlainPerBlockCs[fwht: Int]: QuantRecipe = PerRowQuant(
    fwht, NoGamma(), SingleSided(), PerBlockCs(), VnniPacked(),
)


comptime HeadEmbed[fwht: Int]: QuantRecipe = PerBlockQuant(
    fwht, NoGamma(), SingleSided(), NoColsum(), RowMajor(),
)


struct ButterquantRecipes(MinimaxM3Recipes):
    comptime Qkv: QuantRecipe = SplitGainPerRowCs[128, "input_layernorm.weight"]
    comptime Out: QuantRecipe = PlainPerBlockCs[C.HEAD_DIM]
    comptime IndexProj: QuantRecipe = Passthrough()
    comptime DenseGateUp: QuantRecipe = SplitGainPerRowCs[
        128, "post_attention_layernorm.weight"]
    comptime DenseDown: QuantRecipe = PlainPerBlockCs[128]
    comptime Router: QuantRecipe = RouterCenter("")
    comptime MoeGateUp: QuantRecipe = SplitGainPerRowCs[
        128, "post_attention_layernorm.weight"]
    comptime MoeDown: QuantRecipe = PlainPerBlockCs[128]
    comptime SharedGateUp: QuantRecipe = SplitGainPerRowCs[
        128, "post_attention_layernorm.weight"]
    comptime SharedDown: QuantRecipe = PlainPerBlockCs[128]
    comptime Embed: QuantRecipe = Passthrough()
    comptime LmHead: QuantRecipe = HeadEmbed[128]


comptime R = ButterquantRecipes


comptime BQ_BLOCK = 128
comptime HEAD_NB = C.HIDDEN // BQ_BLOCK
comptime DENSE_NB_DOWN = C.DENSE_INTERMEDIATE // BQ_BLOCK
comptime MOE_NB_DOWN = C.MOE_INTERMEDIATE // BQ_BLOCK
comptime SHARED_NB_DOWN = C.SHARED_INTERMEDIATE // BQ_BLOCK


struct FullKVSlots[batching_seq_len: Int](Copyable, ImplicitlyCopyable, KVSlotGroup):
    comptime CacheShape = ContextRowSharded[Self.batching_seq_len, C.KV_DIM]
    comptime ScaleShape = ContextRowSharded[Self.batching_seq_len, C.NUM_KV_HEADS]
    var k:       Slot[I8,  Self.CacheShape]
    var k_scale: Slot[F32, Self.ScaleShape]
    var v:       Slot[I8,  Self.CacheShape]
    var v_scale: Slot[F32, Self.ScaleShape]


struct IndexKSlots[batching_seq_len: Int](Copyable, ImplicitlyCopyable, KVSlotGroup):
    comptime CacheShape = ContextRowSharded[Self.batching_seq_len, C.INDEX_K_DIM]
    var k: Slot[BF16, Self.CacheShape]


@fieldwise_init
struct MinimaxM3FullAttnScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder[
        "norm_quant", "qkv", "attn_prep", "flash", "merge", "o_prep",
    ]

    var x_i8_band: ScratchPhase["norm_quant", "qkv"]
    var x_i8: ScratchBuffer[Int8, PAGE_LEN * C.HIDDEN]
    var x_sa: ScratchBuffer[Float32, PAGE_LEN]
    var x_row_workspace_band: ScratchPhase["norm_quant", "norm_quant"]
    var x_row_workspace: ScratchBuffer[Float32, C.HIDDEN, ScaleClass.PER_WORKER]

    var q_band: ScratchPhase["qkv", "attn_prep"]
    var q: ScratchBuffer[BFloat16, PAGE_LEN * C.Q_DIM]

    var kv_band: ScratchPhase["qkv", "attn_prep"]
    var kv: ScratchBuffer[BFloat16, PAGE_LEN * C.KV_DIM * 2]

    var qprep_band: ScratchPhase["attn_prep", "flash"]
    var q_i8: ScratchBuffer[Int8, PAGE_LEN * C.Q_DIM]
    var qi_bias: ScratchBuffer[Float32, PAGE_LEN * C.NUM_HEADS]
    var f_q: ScratchBuffer[Float32, PAGE_LEN * C.NUM_HEADS]

    var partials_band: ScratchPhase["flash", "merge"]
    var partials: ScratchBuffer[Float32, PAGE_LEN * FULL_PARTIAL_STRIDE]

    var q_local_band: ScratchPhase["merge", "o_prep"]
    var q_local: ScratchBuffer[BFloat16, PAGE_LEN * C.Q_DIM, ScaleClass.PER_DEGREE]

    var merge_band: ScratchPhase["merge", "merge"]
    var merge_segments: ScratchBuffer[MergeSegment, 1, ScaleClass.PER_WORKER_PER_DEGREE]

    var o_band: ScratchPhase["o_prep", "o_prep"]
    var o_i8: ScratchBuffer[Int8, PAGE_LEN * C.Q_DIM, ScaleClass.PER_DEGREE]
    var o_sa: ScratchBuffer[Float32, PAGE_LEN * C.NUM_HEADS, ScaleClass.PER_DEGREE]
    var o_row_workspace: ScratchBuffer[Float32, C.Q_DIM, ScaleClass.PER_WORKER]


@fieldwise_init
struct MinimaxM3MsaScratch[batching_seq_len: Int](
    ScratchIsland, Copyable, ImplicitlyCopyable
):
    comptime MAX_INDEX_BLOCKS = (Self.batching_seq_len - 1) // C.INDEX_BLOCK + 1
    comptime MAX_BLOCK_STRIDE = (Self.MAX_INDEX_BLOCKS + 15) // 16 * 16

    comptime PHASES = ScratchPhaseOrder[
        "norm_quant", "qkv", "attn_prep", "index_score", "block_select",
        "sparse_flash", "o_prep",
    ]

    var x_i8_band: ScratchPhase["norm_quant", "qkv"]
    var x_i8: ScratchBuffer[Int8, PAGE_LEN * C.HIDDEN]
    var x_sa: ScratchBuffer[Float32, PAGE_LEN]
    var x_row_workspace_band: ScratchPhase["norm_quant", "norm_quant"]
    var x_row_workspace: ScratchBuffer[Float32, C.HIDDEN, ScaleClass.PER_WORKER]

    var q_band: ScratchPhase["qkv", "attn_prep"]
    var q: ScratchBuffer[BFloat16, PAGE_LEN * C.Q_DIM]

    var kv_band: ScratchPhase["qkv", "attn_prep"]
    var kv: ScratchBuffer[BFloat16, PAGE_LEN * C.KV_DIM * 2]

    var index_q_band: ScratchPhase["qkv", "index_score"]
    var index_q: ScratchBuffer[BFloat16, PAGE_LEN * C.INDEX_Q_DIM]
    var index_k: ScratchBuffer[BFloat16, PAGE_LEN * C.INDEX_K_DIM]

    var qprep_band: ScratchPhase["attn_prep", "sparse_flash"]
    var q_i8: ScratchBuffer[Int8, PAGE_LEN * C.Q_DIM]
    var qi_bias: ScratchBuffer[Float32, PAGE_LEN * C.NUM_HEADS]
    var f_q: ScratchBuffer[Float32, PAGE_LEN * C.NUM_HEADS]

    var score_band: ScratchPhase["index_score", "block_select"]
    var index_scores: ScratchBuffer[
        Float32, PAGE_LEN * C.INDEX_NUM_HEADS * Self.MAX_BLOCK_STRIDE,
    ]

    var block_band: ScratchPhase["block_select", "sparse_flash"]
    var block_idx: ScratchBuffer[
        Int32, PAGE_LEN * C.INDEX_NUM_HEADS * C.INDEX_TOPK_BLOCKS,
    ]

    var partials_band: ScratchPhase["sparse_flash", "sparse_flash"]
    var partials: ScratchBuffer[Float32, PAGE_LEN * FULL_PARTIAL_STRIDE]

    var q_local_band: ScratchPhase["sparse_flash", "o_prep"]
    var q_local: ScratchBuffer[BFloat16, PAGE_LEN * C.Q_DIM, ScaleClass.PER_DEGREE]

    var merge_band: ScratchPhase["sparse_flash", "sparse_flash"]
    var merge_segments: ScratchBuffer[MergeSegment, 1, ScaleClass.PER_WORKER_PER_DEGREE]

    var o_band: ScratchPhase["o_prep", "o_prep"]
    var o_i8: ScratchBuffer[Int8, PAGE_LEN * C.Q_DIM, ScaleClass.PER_DEGREE]
    var o_sa: ScratchBuffer[Float32, PAGE_LEN * C.NUM_HEADS, ScaleClass.PER_DEGREE]
    var o_row_workspace: ScratchBuffer[Float32, C.Q_DIM, ScaleClass.PER_WORKER]


@fieldwise_init
struct MinimaxM3DenseMlpScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder[
        "norm_quant", "gate_up", "down_quant", "down",
    ]

    var x_i8_band: ScratchPhase["norm_quant", "gate_up"]
    var x_i8: ScratchBuffer[Int8, PAGE_LEN * C.HIDDEN]
    var x_sa: ScratchBuffer[Float32, PAGE_LEN]
    var x_row_workspace_band: ScratchPhase["norm_quant", "norm_quant"]
    var x_row_workspace: ScratchBuffer[Float32, C.HIDDEN, ScaleClass.PER_WORKER]

    var gate_band: ScratchPhase["gate_up", "down_quant"]
    var gate: ScratchBuffer[
        BFloat16, PAGE_LEN * C.DENSE_INTERMEDIATE, ScaleClass.PER_DEGREE,
    ]

    var up_band: ScratchPhase["gate_up", "gate_up"]
    var up: ScratchBuffer[
        BFloat16, PAGE_LEN * C.DENSE_INTERMEDIATE, ScaleClass.PER_DEGREE,
    ]

    var gate_i8_band: ScratchPhase["down_quant", "down"]
    var gate_i8: ScratchBuffer[
        Int8, PAGE_LEN * C.DENSE_INTERMEDIATE, ScaleClass.PER_DEGREE,
    ]
    var gate_sa: ScratchBuffer[
        Float32, PAGE_LEN * DENSE_NB_DOWN, ScaleClass.PER_DEGREE,
    ]
    var gate_row_workspace_band: ScratchPhase["down_quant", "down_quant"]
    var gate_row_workspace: ScratchBuffer[
        Float32, C.DENSE_INTERMEDIATE, ScaleClass.PER_WORKER,
    ]

    var out_band: ScratchPhase["down", "down"]
    var dense_out: ScratchBuffer[BFloat16, PAGE_LEN * C.HIDDEN]


@fieldwise_init
struct MinimaxM3MoeScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder[
        "router", "moe_norm", "setup", "phase1", "bucket_quant", "phase2",
        "shared_gate_up", "shared_quant", "shared",
    ]

    var router_band: ScratchPhase["router", "router"]
    var cands: ScratchBuffer[M3RouterCandidate, PAGE_LEN * C.TOP_K, ScaleClass.PER_WORKER]

    var setup_band: ScratchPhase["router", "phase2"]
    var route_idx: ScratchBuffer[Int32, PAGE_LEN * C.TOP_K]
    var route_w: ScratchBuffer[Float32, PAGE_LEN * C.TOP_K]
    var expert_offset: ScratchBuffer[Int32, C.NUM_EXPERTS + 1]
    var routes: ScratchBuffer[SparseRoute, PAGE_LEN * C.TOP_K]

    var moe_x_band: ScratchPhase["moe_norm", "shared_gate_up"]
    var moe_x_i8: ScratchBuffer[Int8, PAGE_LEN * C.HIDDEN]
    var moe_x_sa: ScratchBuffer[Float32, PAGE_LEN]
    var moe_x_row_workspace_band: ScratchPhase["moe_norm", "moe_norm"]
    var moe_x_row_workspace: ScratchBuffer[Float32, C.HIDDEN, ScaleClass.PER_WORKER]

    var hidden_band: ScratchPhase["phase1", "bucket_quant"]
    var hidden_bucket: ScratchBuffer[
        BFloat16, PAGE_LEN * C.TOP_K * C.MOE_INTERMEDIATE,
    ]

    var bucket_i8_band: ScratchPhase["bucket_quant", "phase2"]
    var bucket_i8: ScratchBuffer[Int8, PAGE_LEN * C.TOP_K * C.MOE_INTERMEDIATE]
    var bucket_sa: ScratchBuffer[Float32, PAGE_LEN * C.TOP_K * MOE_NB_DOWN]
    var bucket_row_workspace_band: ScratchPhase["bucket_quant", "bucket_quant"]
    var bucket_row_workspace: ScratchBuffer[
        Float32, C.MOE_INTERMEDIATE, ScaleClass.PER_WORKER,
    ]

    var accum_band: ScratchPhase["phase2", "phase2"]
    var moe_accum: ScratchBuffer[Float32, PAGE_LEN * C.HIDDEN]

    var out_band: ScratchPhase["phase2", "shared"]
    var moe_out: ScratchBuffer[BFloat16, PAGE_LEN * C.HIDDEN]

    var shared_gate_band: ScratchPhase["shared_gate_up", "shared_quant"]
    var shared_gate: ScratchBuffer[
        BFloat16, PAGE_LEN * C.SHARED_INTERMEDIATE, ScaleClass.PER_DEGREE,
    ]
    var shared_up_band: ScratchPhase["shared_gate_up", "shared_gate_up"]
    var shared_up: ScratchBuffer[
        BFloat16, PAGE_LEN * C.SHARED_INTERMEDIATE, ScaleClass.PER_DEGREE,
    ]

    var shared_i8_band: ScratchPhase["shared_quant", "shared"]
    var shared_gate_i8: ScratchBuffer[
        Int8, PAGE_LEN * C.SHARED_INTERMEDIATE, ScaleClass.PER_DEGREE,
    ]
    var shared_gate_sa: ScratchBuffer[
        Float32, PAGE_LEN * SHARED_NB_DOWN, ScaleClass.PER_DEGREE,
    ]
    var shared_row_workspace_band: ScratchPhase["shared_quant", "shared_quant"]
    var shared_row_workspace: ScratchBuffer[
        Float32, C.SHARED_INTERMEDIATE, ScaleClass.PER_WORKER,
    ]

    var shared_out_band: ScratchPhase["shared", "shared"]
    var shared_out: ScratchBuffer[BFloat16, PAGE_LEN * C.HIDDEN]


@fieldwise_init
struct MinimaxM3HeadScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
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
        Int32, CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM, ScaleClass.FIXED,
    ]
    var sample_params: ScratchBuffer[
        SamplingParams, CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM, ScaleClass.FIXED,
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
struct MinimaxM3ForwardScratch[batching_seq_len: Int](
    Copyable, ImplicitlyCopyable
):
    var full: MinimaxM3FullAttnScratch
    var msa: MinimaxM3MsaScratch[Self.batching_seq_len]
    var dense_mlp: MinimaxM3DenseMlpScratch
    var moe: MinimaxM3MoeScratch
    var head: MinimaxM3HeadScratch


def calculate_peak_scratch[batching_seq_len: Int](
    degree: Int, max_workers: Int,
) -> Int:
    return aggregate_scratch_peak[MinimaxM3ForwardScratch[batching_seq_len]](
        degree, max_workers)


comptime M3_SWIGLU_ALPHA = Float32(1.702)
comptime M3_SWIGLU_LIMIT = Float32(7.0)


def bq_full_attention_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_seq_len: Int, batching_seq_len: Int,
    max_worker_count: Int = 128,
](
    layout: MinimaxM3Layout[
        ButterquantRecipes,
        FullKVSlots[batching_seq_len],
        IndexKSlots[batching_seq_len],
        max_seq_len,
    ],
    ctx: BindContext[o],
    act: ButterquantActivation[o],
    runs: UnsafePointer[KVRunTable, MutUntrackedOrigin],
    seq_len: Int,
    layer_idx: Int,
    local_idx: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var degree = ctx.degree()
    comptime head_dim = C.HEAD_DIM
    comptime sqrt_hd = sqrt[DType.float32, 1](head_dim)
    comptime hd_eps = Float32(head_dim) * C.RMS_NORM_EPS
    comptime num_q_heads = C.NUM_HEADS
    comptime num_kv_heads = C.NUM_KV_HEADS
    var local_q_rows = MinimaxM3Shapes.O.data_m(degree)
    var local_num_q_heads = local_q_rows // head_dim

    var attn_ctx = ctx.with_layer(layout.dense.base(local_idx))
    var attn = layout.dense.proto.attn
    var kv_ctx = ctx.with_layer(layout.full_kv.base(layer_idx))
    var k_cache = layout.full_kv.proto.k.binding(kv_ctx)
    var k_scale = layout.full_kv.proto.k_scale.binding(kv_ctx)
    var v_cache = layout.full_kv.proto.v.binding(kv_ctx)
    var v_scale = layout.full_kv.proto.v_scale.binding(kv_ctx)

    var q_outs = scratch.binding[MinimaxM3FullAttnScratch, "q"](ctx, plan)
    var k_outs = scratch.binding[MinimaxM3FullAttnScratch, "kv"](ctx, plan)
    var v_outs = k_outs.shifted(seq_len * C.KV_DIM)

    dispatch_bq_qkv[
        hidden=C.HIDDEN, qn_full=C.Q_DIM, kvn_full=C.KV_DIM,
        max_worker_count=max_worker_count,
    ](act,
      attn.q_proj.bq_weight(attn_ctx),
      attn.k_proj.bq_weight(attn_ctx),
      attn.v_proj.bq_weight(attn_ctx),
      q_outs, k_outs, v_outs, C.Q_DIM, C.KV_DIM, seq_len, pools, prof)

    var q_i8 = scratch.binding[MinimaxM3FullAttnScratch, "q_i8"](ctx, plan)
    var qi_bias = scratch.binding[MinimaxM3FullAttnScratch, "qi_bias"](ctx, plan)
    var f_q = scratch.binding[MinimaxM3FullAttnScratch, "f_q"](ctx, plan)

    var rows_per_page = PAGE_LEN // degree
    var page_shift = pow2_shift(rows_per_page)
    var row_mask = rows_per_page - 1

    dispatch_bq_attn_prep[
        head_dim=head_dim, rope_half=C.ROPE_HALF, pair_stride=C.ROPE_HALF,
        sqrt_n=sqrt_hd, n_eps=hd_eps, max_worker_count=max_worker_count,
    ](q_outs, k_outs, v_outs,
      attn.q_norm.binding(attn_ctx), attn.k_norm.binding(attn_ctx),
      q_i8, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
      layout.main_rope.cos.state_binding(ctx),
      layout.main_rope.sin.state_binding(ctx),
      runs, num_q_heads, num_kv_heads, degree,
      page_shift, row_mask, -1, seq_len, pools, prof)

    var q_local = scratch.binding[MinimaxM3FullAttnScratch, "q_local"](ctx, plan)
    var partials = scratch.binding[MinimaxM3FullAttnScratch, "partials"](ctx, plan)
    var merge_segments = scratch.binding[
        MinimaxM3FullAttnScratch, "merge_segments"](ctx, plan)

    dispatch_bq_full_attention[
        head_dim=head_dim, num_q=num_q_heads, num_kv=num_kv_heads,
        gqa_ratio=C.GQA_RATIO, kv_stride=C.KV_DIM,
        partial_stride=FULL_PARTIAL_STRIDE, page_len=PAGE_LEN,
        max_worker_count=max_worker_count,
    ](q_i8, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
      q_local, partials, merge_segments, runs, local_num_q_heads,
      seq_len, pools, prof)

    var o_i8 = scratch.binding[MinimaxM3FullAttnScratch, "o_i8"](ctx, plan)
    var o_sa = scratch.binding[MinimaxM3FullAttnScratch, "o_sa"](ctx, plan)
    var o_row_workspace = scratch.binding[
        MinimaxM3FullAttnScratch, "o_row_workspace"](ctx, plan)

    dispatch_bq_block_quant[
        block=head_dim, apply_fwht=False, max_worker_count=max_worker_count,
    ](q_local, o_i8, o_sa, o_row_workspace, local_q_rows, seq_len, pools, prof)

    var o_act = ButterquantBlockActivation(o_i8, o_sa)
    var xs = layout.activations.x_residual.state_binding(ctx)
    dispatch_bq_block_linear[
        n_rows=C.HIDDEN, max_worker_count=max_worker_count,
    ](o_act, attn.o_proj.bq_weight(attn_ctx), xs, local_q_rows, seq_len,
      pools, prof)


def bq_lightning_indexer[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_seq_len: Int, batching_seq_len: Int,
    max_worker_count: Int = 128,
](
    layout: MinimaxM3Layout[
        ButterquantRecipes,
        FullKVSlots[batching_seq_len],
        IndexKSlots[batching_seq_len],
        max_seq_len,
    ],
    ctx: BindContext[o],
    index_runs: UnsafePointer[KVRunTable, MutUntrackedOrigin],
    seq_len: Int,
    local_idx: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime SC = MinimaxM3MsaScratch[batching_seq_len]
    var degree = ctx.degree()
    comptime ihd = C.INDEX_HEAD_DIM
    comptime sqrt_ihd = sqrt[DType.float32, 1](ihd)
    comptime ihd_eps = Float32(ihd) * C.RMS_NORM_EPS
    comptime num_index_heads = C.INDEX_NUM_HEADS

    var idx_ctx = ctx.with_layer(layout.sparse.base(local_idx))
    var indexer = layout.sparse.proto.indexer
    var ikv_ctx = ctx.with_layer(layout.index_kv.base(local_idx))
    var index_k_cache = layout.index_kv.proto.k.binding(ikv_ctx)

    # TODO(bq): the lightning indexer + its Passthrough index_q/k_proj gemms require the
    # full-gamma bf16 input-norm output (x_hat * gamma), exactly as the reference reads
    # layout.activations.x_residual. In the bq path input_norm is replaced by
    # dispatch_bq_norm_quant (emits int8 only) and model_init bakes input_norm to split-gain
    # sqrt(gamma), so x_residual here does NOT hold the gamma-normed hidden the indexer needs.
    # Options: (a) keep an unbaked-gamma norm weight + an extra dispatch_rms_norm into a new
    # bf16 scratch buffer, (b) extend dispatch_bq_norm_quant to also emit the bf16 normed row,
    # (c) fold sqrt(gamma) into index_q_proj/index_k_proj weights (a recipe change). Wired
    # against x_residual as a placeholder so block_idx still flows to sparse attention.
    var x_in = layout.activations.x_residual.state_binding(ctx)

    var index_q = scratch.binding[SC, "index_q"](ctx, plan)
    var index_k = scratch.binding[SC, "index_k"](ctx, plan)
    var index_scores = scratch.binding[SC, "index_scores"](ctx, plan)
    var block_idx = scratch.binding[SC, "block_idx"](ctx, plan)

    dispatch_gemm[cols=C.HIDDEN, max_worker_count=max_worker_count](
        x_in, indexer.index_q_proj.binding(idx_ctx), index_q,
        C.INDEX_Q_DIM, seq_len, pools, prof)
    dispatch_gemm[cols=C.HIDDEN, max_worker_count=max_worker_count](
        x_in, indexer.index_k_proj.binding(idx_ctx), index_k,
        C.INDEX_K_DIM, seq_len, pools, prof)

    dispatch_rms_norm[
        hidden=ihd, sqrt_n=sqrt_ihd, n_eps=ihd_eps,
        max_worker_count=max_worker_count,
    ](index_q, index_q, indexer.index_q_norm.binding(idx_ctx),
      seq_len * num_index_heads, pools, prof)
    dispatch_rms_norm[
        hidden=ihd, sqrt_n=sqrt_ihd, n_eps=ihd_eps,
        max_worker_count=max_worker_count,
    ](index_k, index_k, indexer.index_k_norm.binding(idx_ctx),
      seq_len, pools, prof)

    var rows_per_page = PAGE_LEN // degree
    var page_shift = pow2_shift(rows_per_page)
    var row_mask = rows_per_page - 1
    dispatch_rope_k_cache_write[
        half=C.ROPE_HALF, pair_stride=C.ROPE_HALF, head_dim=ihd,
        max_worker_count=max_worker_count,
    ](index_q, index_k, index_k_cache,
      layout.main_rope.cos.state_binding(ctx),
      layout.main_rope.sin.state_binding(ctx),
      index_runs, num_index_heads, 1, degree,
      page_shift, row_mask, -1, seq_len, pools, prof)

    dispatch_minimax_m3_indexer[
        page_len=PAGE_LEN, max_worker_count=max_worker_count,
    ](index_q, index_k_cache, block_idx, index_scores, index_runs, seq_len,
      pools, prof)


def bq_msa_attention_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_seq_len: Int, batching_seq_len: Int,
    max_worker_count: Int = 128,
](
    layout: MinimaxM3Layout[
        ButterquantRecipes,
        FullKVSlots[batching_seq_len],
        IndexKSlots[batching_seq_len],
        max_seq_len,
    ],
    ctx: BindContext[o],
    act: ButterquantActivation[o],
    runs: UnsafePointer[KVRunTable, MutUntrackedOrigin],
    index_runs: UnsafePointer[KVRunTable, MutUntrackedOrigin],
    seq_len: Int,
    layer_idx: Int,
    local_idx: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime SC = MinimaxM3MsaScratch[batching_seq_len]
    var degree = ctx.degree()
    comptime head_dim = C.HEAD_DIM
    comptime sqrt_hd = sqrt[DType.float32, 1](head_dim)
    comptime hd_eps = Float32(head_dim) * C.RMS_NORM_EPS
    comptime num_q_heads = C.NUM_HEADS
    comptime num_kv_heads = C.NUM_KV_HEADS
    var local_q_rows = MinimaxM3Shapes.O.data_m(degree)

    var attn_ctx = ctx.with_layer(layout.sparse.base(local_idx))
    var attn = layout.sparse.proto.attn
    var kv_ctx = ctx.with_layer(layout.full_kv.base(layer_idx))
    var k_cache = layout.full_kv.proto.k.binding(kv_ctx)
    var k_scale = layout.full_kv.proto.k_scale.binding(kv_ctx)
    var v_cache = layout.full_kv.proto.v.binding(kv_ctx)
    var v_scale = layout.full_kv.proto.v_scale.binding(kv_ctx)

    var q_outs = scratch.binding[SC, "q"](ctx, plan)
    var k_outs = scratch.binding[SC, "kv"](ctx, plan)
    var v_outs = k_outs.shifted(seq_len * C.KV_DIM)

    dispatch_bq_qkv[
        hidden=C.HIDDEN, qn_full=C.Q_DIM, kvn_full=C.KV_DIM,
        max_worker_count=max_worker_count,
    ](act,
      attn.q_proj.bq_weight(attn_ctx),
      attn.k_proj.bq_weight(attn_ctx),
      attn.v_proj.bq_weight(attn_ctx),
      q_outs, k_outs, v_outs, C.Q_DIM, C.KV_DIM, seq_len, pools, prof)

    var q_i8 = scratch.binding[SC, "q_i8"](ctx, plan)
    var qi_bias = scratch.binding[SC, "qi_bias"](ctx, plan)
    var f_q = scratch.binding[SC, "f_q"](ctx, plan)

    var rows_per_page = PAGE_LEN // degree
    var page_shift = pow2_shift(rows_per_page)
    var row_mask = rows_per_page - 1

    dispatch_bq_attn_prep[
        head_dim=head_dim, rope_half=C.ROPE_HALF, pair_stride=C.ROPE_HALF,
        sqrt_n=sqrt_hd, n_eps=hd_eps, max_worker_count=max_worker_count,
    ](q_outs, k_outs, v_outs,
      attn.q_norm.binding(attn_ctx), attn.k_norm.binding(attn_ctx),
      q_i8, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
      layout.main_rope.cos.state_binding(ctx),
      layout.main_rope.sin.state_binding(ctx),
      runs, num_q_heads, num_kv_heads, degree,
      page_shift, row_mask, -1, seq_len, pools, prof)

    bq_lightning_indexer[
        max_seq_len=max_seq_len, batching_seq_len=batching_seq_len,
        max_worker_count=max_worker_count,
    ](layout, ctx, index_runs, seq_len, local_idx, scratch, plan, pools, prof)

    var block_idx = scratch.binding[SC, "block_idx"](ctx, plan)
    var q_local = scratch.binding[SC, "q_local"](ctx, plan)
    var partials = scratch.binding[SC, "partials"](ctx, plan)
    var merge_segments = scratch.binding[SC, "merge_segments"](ctx, plan)

    dispatch_bq_minimax_m3_sparse_attention[
        page_len=PAGE_LEN, max_worker_count=max_worker_count,
    ](q_i8, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale, block_idx,
      q_local, partials, merge_segments, runs, seq_len, pools, prof)

    var o_i8 = scratch.binding[SC, "o_i8"](ctx, plan)
    var o_sa = scratch.binding[SC, "o_sa"](ctx, plan)
    var o_row_workspace = scratch.binding[SC, "o_row_workspace"](ctx, plan)

    dispatch_bq_block_quant[
        block=head_dim, apply_fwht=False, max_worker_count=max_worker_count,
    ](q_local, o_i8, o_sa, o_row_workspace, local_q_rows, seq_len, pools, prof)

    var o_act = ButterquantBlockActivation(o_i8, o_sa)
    var xs = layout.activations.x_residual.state_binding(ctx)
    dispatch_bq_block_linear[
        n_rows=C.HIDDEN, max_worker_count=max_worker_count,
    ](o_act, attn.o_proj.bq_weight(attn_ctx), xs, local_q_rows, seq_len,
      pools, prof)


def bq_dense_mlp[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    ctx: BindContext[o],
    layer_ctx: BindContext[o],
    dl: DenseLayerRefs[ButterquantRecipes],
    x_main: Binding[BFloat16, o],
    seq_len: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var degree = ctx.degree()
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = Float32(C.HIDDEN) * C.RMS_NORM_EPS
    var intermediate_per_rank = MinimaxM3Shapes.DenseGateUp.data_n(degree)

    var x_i8 = scratch.binding[MinimaxM3DenseMlpScratch, "x_i8"](ctx, plan)
    var x_sa = scratch.binding[MinimaxM3DenseMlpScratch, "x_sa"](ctx, plan)
    var x_row_workspace = scratch.binding[
        MinimaxM3DenseMlpScratch, "x_row_workspace"](ctx, plan)
    var gate = scratch.binding[MinimaxM3DenseMlpScratch, "gate"](ctx, plan)
    var up = scratch.binding[MinimaxM3DenseMlpScratch, "up"](ctx, plan)
    var gate_i8 = scratch.binding[MinimaxM3DenseMlpScratch, "gate_i8"](ctx, plan)
    var gate_sa = scratch.binding[MinimaxM3DenseMlpScratch, "gate_sa"](ctx, plan)
    var gate_row_workspace = scratch.binding[
        MinimaxM3DenseMlpScratch, "gate_row_workspace"](ctx, plan)
    var dense_out = scratch.binding[
        MinimaxM3DenseMlpScratch, "dense_out"](ctx, plan)

    dispatch_bq_norm_quant[
        hidden=C.HIDDEN, block=BQ_BLOCK, sqrt_n=sqrt_n, n_eps=n_eps,
        max_worker_count=max_worker_count,
    ](x_main, dl.post_attn_norm.binding(layer_ctx),
      x_i8, x_sa, x_row_workspace, seq_len, pools, prof)

    var act = ButterquantActivation(x_i8, x_sa)
    dispatch_bq_linear[hidden=C.HIDDEN, max_worker_count=max_worker_count](
        act, dl.mlp.gate_proj.bq_weight(layer_ctx), gate,
        intermediate_per_rank, seq_len, pools, prof)
    dispatch_bq_linear[hidden=C.HIDDEN, max_worker_count=max_worker_count](
        act, dl.mlp.up_proj.bq_weight(layer_ctx), up,
        intermediate_per_rank, seq_len, pools, prof)

    dispatch_gate_up_act[
        activation="swiglu_oai", alpha=M3_SWIGLU_ALPHA, limit=M3_SWIGLU_LIMIT,
        max_worker_count=max_worker_count,
    ](gate, up, gate, intermediate_per_rank, seq_len, pools, prof)

    dispatch_bq_block_quant[
        block=BQ_BLOCK, apply_fwht=True, max_worker_count=max_worker_count,
    ](gate, gate_i8, gate_sa, gate_row_workspace,
      intermediate_per_rank, seq_len, pools, prof)

    var gate_act = ButterquantBlockActivation(gate_i8, gate_sa)
    dispatch_bq_block_linear[
        n_rows=C.HIDDEN, max_worker_count=max_worker_count,
    ](gate_act, dl.mlp.down_proj.bq_weight(layer_ctx), dense_out,
      intermediate_per_rank, seq_len, pools, prof)

    dispatch_allreduce_inplace[BF16, max_worker_count=max_worker_count](
        dense_out, seq_len * C.HIDDEN, pools, prof)


def bq_m3_router[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    ctx: BindContext[o],
    layer_ctx: BindContext[o],
    sl: SparseLayerRefs[ButterquantRecipes],
    x_main: Binding[BFloat16, o],
    seq_len: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var degree = ctx.degree()
    var experts_per_rank = C.NUM_EXPERTS // degree
    var cands = scratch.binding[MinimaxM3MoeScratch, "cands"](ctx, plan)
    var route_idx = scratch.binding[MinimaxM3MoeScratch, "route_idx"](ctx, plan)
    var route_w = scratch.binding[MinimaxM3MoeScratch, "route_w"](ctx, plan)

    # TODO(bq): dispatch_minimax_m3_router (prototypes/sigmoid_router.mojo) takes the
    # full-gamma bf16 post_attn_norm output as x plus a raw F32 router_gate + F32 router_bias.
    # Two gaps here: (a) like the indexer, no gamma-normed bf16 input exists in the bq path
    # (post_attn_norm is baked to sqrt(gamma) and norm_quant emits int8 only) -- x_main passed
    # below is the UN-normed residual, a placeholder. (b) the Router=RouterCenter("") recipe
    # expects router_gate centered, but model_init does not bake router centering (cf.
    # gemma4_bake_router_scales). Resolve by either adding a center-bake in model_init so the
    # F32 router_gate is drop-in, or light-altering dispatch_minimax_m3_router to consume a
    # centered/bq_router binding + gauge. Also resolve the bf16 normed-input source.
    dispatch_minimax_m3_router[max_worker_count=max_worker_count](
        x_main,
        sl.moe.router_gate.binding(layer_ctx),
        sl.moe.router_bias.binding(layer_ctx),
        cands, route_idx, route_w,
        experts_per_rank, seq_len, pools, prof)


def bq_moe[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    ctx: BindContext[o],
    layer_ctx: BindContext[o],
    sl: SparseLayerRefs[ButterquantRecipes],
    x_main: Binding[BFloat16, o],
    moe_out: Binding[BFloat16, o],
    seq_len: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var degree = ctx.degree()
    var experts_per_rank = C.NUM_EXPERTS // degree
    var shared_inter_per_rank = MinimaxM3Shapes.SharedGateUp.data_n(degree)
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = Float32(C.HIDDEN) * C.RMS_NORM_EPS

    var route_idx = scratch.binding[MinimaxM3MoeScratch, "route_idx"](ctx, plan)
    var route_w = scratch.binding[MinimaxM3MoeScratch, "route_w"](ctx, plan)
    var expert_offset = scratch.binding[
        MinimaxM3MoeScratch, "expert_offset"](ctx, plan)
    var routes = scratch.binding[MinimaxM3MoeScratch, "routes"](ctx, plan)
    var moe_x_i8 = scratch.binding[MinimaxM3MoeScratch, "moe_x_i8"](ctx, plan)
    var moe_x_sa = scratch.binding[MinimaxM3MoeScratch, "moe_x_sa"](ctx, plan)
    var moe_x_row_workspace = scratch.binding[
        MinimaxM3MoeScratch, "moe_x_row_workspace"](ctx, plan)
    var hidden_bucket = scratch.binding[
        MinimaxM3MoeScratch, "hidden_bucket"](ctx, plan)
    var bucket_i8 = scratch.binding[MinimaxM3MoeScratch, "bucket_i8"](ctx, plan)
    var bucket_sa = scratch.binding[MinimaxM3MoeScratch, "bucket_sa"](ctx, plan)
    var bucket_row_workspace = scratch.binding[
        MinimaxM3MoeScratch, "bucket_row_workspace"](ctx, plan)
    var moe_accum = scratch.binding[MinimaxM3MoeScratch, "moe_accum"](ctx, plan)
    var shared_gate = scratch.binding[
        MinimaxM3MoeScratch, "shared_gate"](ctx, plan)
    var shared_up = scratch.binding[MinimaxM3MoeScratch, "shared_up"](ctx, plan)
    var shared_gate_i8 = scratch.binding[
        MinimaxM3MoeScratch, "shared_gate_i8"](ctx, plan)
    var shared_gate_sa = scratch.binding[
        MinimaxM3MoeScratch, "shared_gate_sa"](ctx, plan)
    var shared_row_workspace = scratch.binding[
        MinimaxM3MoeScratch, "shared_row_workspace"](ctx, plan)
    var shared_out = scratch.binding[
        MinimaxM3MoeScratch, "shared_out"](ctx, plan)

    dispatch_bq_norm_quant[
        hidden=C.HIDDEN, block=BQ_BLOCK, sqrt_n=sqrt_n, n_eps=n_eps,
        max_worker_count=max_worker_count,
    ](x_main, sl.post_attn_norm.binding(layer_ctx),
      moe_x_i8, moe_x_sa, moe_x_row_workspace, seq_len, pools, prof)

    var moe_act = ButterquantActivation(moe_x_i8, moe_x_sa)

    dispatch_build_expert_schedules[
        C.NUM_EXPERTS, C.TOP_K, max_worker_count=max_worker_count,
    ](route_idx, route_w, expert_offset, routes,
      experts_per_rank, seq_len, pools, prof)

    dispatch_bq_m3_phase1_gate_up[
        hidden=C.HIDDEN, gate_up=C.MOE_GATE_UP_FUSED, inter=C.MOE_INTERMEDIATE,
        max_worker_count=max_worker_count,
    ](moe_act, expert_offset, routes,
      sl.moe.experts_gate_up.bq_weight(layer_ctx), hidden_bucket,
      experts_per_rank, pools, prof)

    var num_routes = seq_len * C.TOP_K
    dispatch_bq_block_quant[
        block=BQ_BLOCK, apply_fwht=True, max_worker_count=max_worker_count,
    ](hidden_bucket, bucket_i8, bucket_sa, bucket_row_workspace,
      C.MOE_INTERMEDIATE, num_routes, pools, prof)

    var bucket_act = ButterquantBlockActivation(bucket_i8, bucket_sa)
    dispatch_bq_phase2_down[
        hidden=C.HIDDEN, inter=C.MOE_INTERMEDIATE,
        max_worker_count=max_worker_count,
    ](bucket_act, expert_offset, routes,
      sl.moe.experts_down.bq_weight(layer_ctx), moe_accum, moe_out,
      experts_per_rank, seq_len, pools, prof)

    dispatch_allreduce_inplace[BF16, max_worker_count=max_worker_count](
        moe_out, seq_len * C.HIDDEN, pools, prof)

    dispatch_bq_linear[hidden=C.HIDDEN, max_worker_count=max_worker_count](
        moe_act, sl.moe.shared_gate.bq_weight(layer_ctx), shared_gate,
        shared_inter_per_rank, seq_len, pools, prof)
    dispatch_bq_linear[hidden=C.HIDDEN, max_worker_count=max_worker_count](
        moe_act, sl.moe.shared_up.bq_weight(layer_ctx), shared_up,
        shared_inter_per_rank, seq_len, pools, prof)

    dispatch_gate_up_act[
        activation="swiglu_oai", alpha=M3_SWIGLU_ALPHA, limit=M3_SWIGLU_LIMIT,
        max_worker_count=max_worker_count,
    ](shared_gate, shared_up, shared_gate, shared_inter_per_rank, seq_len,
      pools, prof)

    dispatch_bq_block_quant[
        block=BQ_BLOCK, apply_fwht=True, max_worker_count=max_worker_count,
    ](shared_gate, shared_gate_i8, shared_gate_sa, shared_row_workspace,
      shared_inter_per_rank, seq_len, pools, prof)

    var shared_act = ButterquantBlockActivation(shared_gate_i8, shared_gate_sa)
    dispatch_bq_block_linear[
        n_rows=C.HIDDEN, max_worker_count=max_worker_count,
    ](shared_act, sl.moe.shared_down.bq_weight(layer_ctx), shared_out,
      shared_inter_per_rank, seq_len, pools, prof)

    dispatch_allreduce_inplace[BF16, max_worker_count=max_worker_count](
        shared_out, seq_len * C.HIDDEN, pools, prof)

    dispatch_residual_add[hidden=C.HIDDEN, max_worker_count=max_worker_count](
        moe_out, shared_out, moe_out, seq_len, pools, prof)


struct MinimaxM3[
    max_seq_len: Int = 8192,
    batching_seq_len: Int = 8192,
    Pool: BurstThreadPool = BurstPool[],
    profile: Bool = False, profile_slots: Int = 64,
](Movable, ScheduledModel):
    comptime POSITIONS_PER_PAGE = PAGE_LEN
    comptime Recipes = R

    var arenas: List[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var pools: List[Self.Pool]
    var layout: MinimaxM3Layout[
        ButterquantRecipes,
        FullKVSlots[Self.batching_seq_len],
        IndexKSlots[Self.batching_seq_len],
        Self.max_seq_len,
    ]
    var scratch: TemporalScratchPool
    var arena_bases: List[Int]
    var degree: Int
    var kv_mirrors: List[KVPoolMirror]
    var full_runs: KVRunTable
    var index_runs: KVRunTable
    var full_plan: ScratchPlan
    var msa_plan: ScratchPlan
    var dense_mlp_plan: ScratchPlan
    var moe_plan: ScratchPlan
    var head_plan: ScratchPlan
    var profiler: Profiler[Self.profile, Self.profile_slots]
    var tokens_processed: Int

    def __init__(out self,
        var arenas: List[NumaArena[alignment=DEFAULT_ALIGNMENT]],
        var pools: List[Self.Pool],
        layout: MinimaxM3Layout[
            ButterquantRecipes,
            FullKVSlots[Self.batching_seq_len],
            IndexKSlots[Self.batching_seq_len],
            Self.max_seq_len,
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
        self.kv_mirrors = minimax_m3_kv_mirrors[
            batching_seq_len=Self.batching_seq_len,
        ](self.layout, degree)
        self.full_runs = KVRunTable()
        self.index_runs = KVRunTable()
        self.full_plan = derive_checked_plan[MinimaxM3FullAttnScratch](degree, max_workers)
        self.msa_plan = derive_checked_plan[
            MinimaxM3MsaScratch[Self.batching_seq_len]](degree, max_workers)
        self.dense_mlp_plan = derive_checked_plan[MinimaxM3DenseMlpScratch](degree, max_workers)
        self.moe_plan = derive_checked_plan[MinimaxM3MoeScratch](degree, max_workers)
        self.head_plan = derive_checked_plan[MinimaxM3HeadScratch](degree, max_workers)
        self.profiler = Profiler[Self.profile, Self.profile_slots]()
        self.tokens_processed = 0

    def init_state(mut self):
        var tasks = List[PackColsumTask]()
        for i in range(C.NUM_LAYERS):
            var entry = LAYER_SCHEDULE[i]
            if entry.kind == LayerKind.DENSE:
                _ = emit_pack_tasks[DenseLayerRefs[R]](
                    self.layout.dense.base(entry.local_idx),
                    self.degree, tasks)
            else:
                _ = emit_pack_tasks[SparseLayerRefs[R]](
                    self.layout.sparse.base(entry.local_idx),
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
                if entry.kind == LayerKind.DENSE:
                    var lb = base + layout.dense.base(entry.local_idx)
                    ref dbody = layout.dense.proto
                    bake_split_gain_in_place(dbody.input_norm.at(lb), C.HIDDEN)
                    bake_split_gain_in_place(dbody.post_attn_norm.at(lb), C.HIDDEN)
                else:
                    var lb = base + layout.sparse.base(entry.local_idx)
                    ref sbody = layout.sparse.proto
                    bake_split_gain_in_place(sbody.input_norm.at(lb), C.HIDDEN)
                    bake_split_gain_in_place(sbody.post_attn_norm.at(lb), C.HIDDEN)

        minimax_m3_init_rope_tables(self.layout, self.arena_bases)

    def batch_geometry(self) -> BatchGeometry:
        return BatchGeometry(
            max_seqs=CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM,
            max_slots=CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM,
            max_step_tokens=PAGE_LEN,
            pools=pool_specs(self.kv_mirrors))

    def run_prefix_copies(mut self, read schedule: Schedule):
        dispatch_prefix_copies(
            self.kv_mirrors, schedule, self.arena_bases,
            self.pools, self.profiler)

    def bind_step_runs(
        mut self, read schedule: Schedule, read pages: KVPageAccountant,
    ):
        bind_pool_run_table(
            self.full_runs, schedule, pages,
            FULL_POOL, self.kv_mirrors[FULL_POOL])
        bind_pool_run_table(
            self.index_runs, schedule, pages,
            INDEX_POOL, self.kv_mirrors[INDEX_POOL])

    def execute(
        mut self,
        read schedule: Schedule,
        read pages: KVPageAccountant,
    ) -> List[SampleOutcome[MAXIMUM_SAMPLING_LOGITS]]:
        ref layout = self.layout
        var degree = self.degree
        comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
        comptime n_eps = Float32(C.HIDDEN) * C.RMS_NORM_EPS
        comptime embed_scale = Float64(1.0)
        var vocab_per_rank = C.VOCAB_SIZE // degree
        var shard_rows = MinimaxM3TailShapes.Embed.data_n(degree)

        var wall_t0 = perf_counter_ns()
        var num_slots = len(schedule.slots)
        var total = len(schedule.tokens)
        debug_assert(num_slots > 0, "execute called with no slots")
        debug_assert(
            num_slots <= CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM,
            "execute slot count exceeds parallelism cap",
        )
        debug_assert(
            total <= PAGE_LEN, "execute packed tokens exceed PAGE_LEN")
        self.tokens_processed += total

        var ctx = BindContext(RankView(Span(self.arena_bases)), 0)
        var tail_ctx = ctx.with_layer(layout.tail.base(0))

        var x_main = layout.activations.x_main.state_binding(ctx)
        var x_res = layout.activations.x_residual.state_binding(ctx)
        var accums = self.scratch.binding[
            MinimaxM3HeadScratch, "accums"](ctx, self.head_plan)
        var sample_params = self.scratch.binding[
            MinimaxM3HeadScratch, "sample_params"](ctx, self.head_plan)
        var head_x = self.scratch.binding[
            MinimaxM3HeadScratch, "head_x"](ctx, self.head_plan)
        var emit_rows = self.scratch.binding[
            MinimaxM3HeadScratch, "emit_rows"](ctx, self.head_plan)
        var outcome = self.scratch.binding[
            MinimaxM3HeadScratch, "outcome"](ctx, self.head_plan)
        var head_x_i8 = self.scratch.binding[
            MinimaxM3HeadScratch, "head_x_i8"](ctx, self.head_plan)
        var head_x_sa = self.scratch.binding[
            MinimaxM3HeadScratch, "head_x_sa"](ctx, self.head_plan)
        var head_row_workspace = self.scratch.binding[
            MinimaxM3HeadScratch, "head_row_workspace"](ctx, self.head_plan)

        var buf_starts = pack_slot_starts(schedule)
        var emit_plan = collect_emit_plan(schedule, buf_starts)
        var num_emit = emit_plan.count()
        self.run_prefix_copies(schedule)
        self.bind_step_runs(schedule, pages)
        var full_runs = UnsafePointer(to=self.full_runs).unsafe_origin_cast[
            MutUntrackedOrigin]()
        var index_runs = UnsafePointer(to=self.index_runs).unsafe_origin_cast[
            MutUntrackedOrigin]()

        dispatch_embed_lookup[hidden=C.HIDDEN, scale=embed_scale](
            Span(schedule.tokens),
            layout.tail.proto.embed.binding(tail_ctx),
            x_main, shard_rows, total, self.pools, self.profiler)
        dispatch_allreduce_inplace[BF16](
            x_main, total * C.HIDDEN, self.pools, self.profiler)

        for i in range(C.NUM_LAYERS):
            if schedule.fully_cancelled():
                return List[SampleOutcome[MAXIMUM_SAMPLING_LOGITS]]()
            var entry = LAYER_SCHEDULE[i]
            if entry.kind == LayerKind.DENSE:
                var layer_ctx = ctx.with_layer(layout.dense.base(entry.local_idx))
                var dl = layout.dense.proto

                var x_i8 = self.scratch.binding[
                    MinimaxM3FullAttnScratch, "x_i8"](ctx, self.full_plan)
                var x_sa = self.scratch.binding[
                    MinimaxM3FullAttnScratch, "x_sa"](ctx, self.full_plan)
                var x_row_workspace = self.scratch.binding[
                    MinimaxM3FullAttnScratch, "x_row_workspace"](
                    ctx, self.full_plan)
                dispatch_bq_norm_quant[
                    hidden=C.HIDDEN, block=BQ_BLOCK, sqrt_n=sqrt_n, n_eps=n_eps,
                ](x_main, dl.input_norm.binding(layer_ctx),
                  x_i8, x_sa, x_row_workspace, total, self.pools, self.profiler)
                var act = ButterquantActivation(x_i8, x_sa)

                bq_full_attention_qkv[
                    max_seq_len=Self.max_seq_len,
                    batching_seq_len=Self.batching_seq_len,
                ](layout, ctx, act, full_runs, total, entry.idx, entry.local_idx,
                  self.scratch, self.full_plan, self.pools, self.profiler)

                dispatch_allreduce_inplace[BF16](
                    x_res, total * C.HIDDEN, self.pools, self.profiler)
                dispatch_residual_add[hidden=C.HIDDEN](
                    x_main, x_res, x_main, total, self.pools, self.profiler)

                bq_dense_mlp(
                    ctx, layer_ctx, dl, x_main, total,
                    self.scratch, self.dense_mlp_plan, self.pools, self.profiler)
                var dense_out = self.scratch.binding[
                    MinimaxM3DenseMlpScratch, "dense_out"](
                    ctx, self.dense_mlp_plan)
                dispatch_residual_add[hidden=C.HIDDEN](
                    x_main, dense_out, x_main, total, self.pools, self.profiler)
            else:
                var layer_ctx = ctx.with_layer(
                    layout.sparse.base(entry.local_idx))
                var sl = layout.sparse.proto

                var x_i8 = self.scratch.binding[
                    MinimaxM3MsaScratch[Self.batching_seq_len], "x_i8"](
                    ctx, self.msa_plan)
                var x_sa = self.scratch.binding[
                    MinimaxM3MsaScratch[Self.batching_seq_len], "x_sa"](
                    ctx, self.msa_plan)
                var x_row_workspace = self.scratch.binding[
                    MinimaxM3MsaScratch[Self.batching_seq_len],
                    "x_row_workspace"](ctx, self.msa_plan)
                dispatch_bq_norm_quant[
                    hidden=C.HIDDEN, block=BQ_BLOCK, sqrt_n=sqrt_n, n_eps=n_eps,
                ](x_main, sl.input_norm.binding(layer_ctx),
                  x_i8, x_sa, x_row_workspace, total, self.pools, self.profiler)
                var act = ButterquantActivation(x_i8, x_sa)

                bq_msa_attention_qkv[
                    max_seq_len=Self.max_seq_len,
                    batching_seq_len=Self.batching_seq_len,
                ](layout, ctx, act, full_runs, index_runs, total,
                  entry.idx, entry.local_idx,
                  self.scratch, self.msa_plan, self.pools, self.profiler)

                dispatch_allreduce_inplace[BF16](
                    x_res, total * C.HIDDEN, self.pools, self.profiler)
                dispatch_residual_add[hidden=C.HIDDEN](
                    x_main, x_res, x_main, total, self.pools, self.profiler)

                bq_m3_router(
                    ctx, layer_ctx, sl, x_main, total,
                    self.scratch, self.moe_plan, self.pools, self.profiler)
                var moe_out = self.scratch.binding[
                    MinimaxM3MoeScratch, "moe_out"](ctx, self.moe_plan)
                bq_moe(
                    ctx, layer_ctx, sl, x_main, moe_out, total,
                    self.scratch, self.moe_plan, self.pools, self.profiler)
                dispatch_residual_add[hidden=C.HIDDEN](
                    x_main, moe_out, x_main, total, self.pools, self.profiler)

        var outcomes = List[SampleOutcome[MAXIMUM_SAMPLING_LOGITS]]()

        if num_emit > 0:
            debug_assert(
                num_emit <= CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM,
                "execute emit count exceeds parallelism cap",
            )
            var x_head = stage_sampling_inputs[hidden=C.HIDDEN](
                emit_plan, schedule, x_main, head_x,
                emit_rows, sample_params, self.pools, self.profiler)

            var head_act = ButterquantActivation(head_x_i8, head_x_sa)
            dispatch_bq_head_prep[
                hidden=C.HIDDEN, block=BQ_BLOCK, sqrt_n=sqrt_n, n_eps=n_eps,
            ](x_head, layout.tail.proto.final_norm.binding(tail_ctx),
              head_act, head_row_workspace, num_emit,
              self.pools, self.profiler)

            var out_ptr = outcome[0]
            dispatch_bq_flash_sample[
                cols=C.HIDDEN, cap=Float64(0.0),
                n_max=MAXIMUM_SAMPLING_LOGITS,
            ](head_act, layout.tail.proto.lm_head.bq_weight(tail_ctx),
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

        for r in range(degree):
            pools[r].sleep()

        var arenas = List[NumaArena[alignment=DEFAULT_ALIGNMENT]](capacity=degree)
        var layout_opt = minimax_m3_load_arenas[
            ButterquantRecipes,
            FullKVSlots[Self.batching_seq_len],
            IndexKSlots[Self.batching_seq_len],
            Self.max_seq_len, Self.batching_seq_len,
        ](dir_path, topo, degree, max_workers,
          calculate_peak_scratch[Self.batching_seq_len](degree, max_workers),
          arenas)
        if not layout_opt:
            return None

        for r in range(degree):
            pools[r].wake()

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
            var prefix = String(t"language_model.model.layers.{entry.idx}.")
            if entry.kind == LayerKind.DENSE:
                if not q.plan_walk[DenseLayerRefs[R]](prefix, entry.idx):
                    return False
            else:
                if not q.plan_walk[SparseLayerRefs[R]](prefix, entry.idx):
                    return False
        if not q.plan_walk[TailRefs[R]](String(""), -1):
            return False
        if not q.write_header():
            return False
        return q.execute(topo, pools^)
