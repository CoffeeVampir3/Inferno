from std.pathlib import Path
from std.memory import Span, UnsafePointer
from std.time import perf_counter_ns
from simd_math.ops import sqrt

from numa import NumaArena, NumaTopology
from threading import BurstPool
from threading.threading_traits import BurstThreadPool
from kernels.helpers import RankView, Binding, prime_fp_environment
from kernels.attention_ops import KVRunTable, pow2_shift
from kernels.flash_sample import (
    SamplingParams, SampleAccum, SampleOutcome, dispatch_flash_sample,
)
from kernels.logsum_merge import MergeSegment
from kernels.moe_router import SparseRoute, dispatch_build_expert_schedules
from kernels.profiling import Profiler
from kernels.rope import dispatch_rope_cache_write, dispatch_rope_k_cache_write
from kernels.reductions import dispatch_allreduce_inplace
from kernels.embedding import dispatch_embed_lookup
from kernels.rmsnorm import dispatch_rms_norm
from kernels.gemm import dispatch_gemm, dispatch_gemm_cols
from kernels.attention_dispatch_kernels import dispatch_full_attention
from kernels.elementwise import dispatch_residual_add, dispatch_gate_up_act
from kernels.moe_experts import dispatch_phase1_gate_up, dispatch_phase2_down

from modeling.temporal_scratch import (
    ScratchBuffer, ScratchIsland, ScratchPhaseOrder, ScratchPhase, ScaleClass,
    TemporalScratchPool, ScratchPlan,
    derive_checked_plan, aggregate_scratch_peak,
)
from modeling.model_spec import (
    BF16, ContextRowSharded, DEFAULT_ALIGNMENT,
)
from modeling.modeling_common import (
    pack_slot_starts, collect_emit_plan, stage_sampling_inputs,
)
from modeling.slot import (
    Slot, BindContext,
)
from modeling.gemma4_topology import KVSlotGroup
from modeling.kv_policy import (
    KVPoolMirror, pool_specs, dispatch_prefix_copies, bind_pool_run_table,
)
from prototypes.sigmoid_router import (
    M3RouterCandidate, dispatch_minimax_m3_router,
)
from prototypes.lightning_indexer import dispatch_minimax_m3_indexer
from prototypes.sparse_attention import dispatch_minimax_m3_sparse_attention
from quant.recipe import QuantRecipe, Passthrough
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
    bake_minimax_gain_inplace,
    minimax_m3_load_arenas, minimax_m3_kv_mirrors, minimax_m3_init_rope_tables,
)


comptime C = MinimaxM3Config


struct PassthroughRecipes(MinimaxM3Recipes):
    comptime Qkv: QuantRecipe = Passthrough()
    comptime Out: QuantRecipe = Passthrough()
    comptime IndexProj: QuantRecipe = Passthrough()
    comptime DenseGateUp: QuantRecipe = Passthrough()
    comptime DenseDown: QuantRecipe = Passthrough()
    comptime Router: QuantRecipe = Passthrough()
    comptime MoeGateUp: QuantRecipe = Passthrough()
    comptime MoeDown: QuantRecipe = Passthrough()
    comptime SharedGateUp: QuantRecipe = Passthrough()
    comptime SharedDown: QuantRecipe = Passthrough()
    comptime Embed: QuantRecipe = Passthrough()
    comptime LmHead: QuantRecipe = Passthrough()


comptime R = PassthroughRecipes


struct FullKVSlots[batching_seq_len: Int](Copyable, ImplicitlyCopyable, KVSlotGroup):
    comptime CacheShape = ContextRowSharded[Self.batching_seq_len, C.KV_DIM]
    var k: Slot[BF16, Self.CacheShape]
    var v: Slot[BF16, Self.CacheShape]


struct IndexKSlots[batching_seq_len: Int](Copyable, ImplicitlyCopyable, KVSlotGroup):
    comptime CacheShape = ContextRowSharded[Self.batching_seq_len, C.INDEX_K_DIM]
    var k: Slot[BF16, Self.CacheShape]


@fieldwise_init
struct MinimaxM3FullAttnScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder["prep", "flash", "merge"]

    var q_band: ScratchPhase["prep", "flash"]
    var q: ScratchBuffer[BFloat16, PAGE_LEN * C.Q_DIM, ScaleClass.FIXED]

    var kv_band: ScratchPhase["prep", "prep"]
    var kv: ScratchBuffer[BFloat16, PAGE_LEN * C.KV_DIM * 2, ScaleClass.FIXED]

    var partials_band: ScratchPhase["flash", "merge"]
    var partials: ScratchBuffer[Float32, PAGE_LEN * FULL_PARTIAL_STRIDE, ScaleClass.FIXED]

    var q_local_band: ScratchPhase["merge", "merge"]
    var q_local: ScratchBuffer[BFloat16, PAGE_LEN * C.Q_DIM, ScaleClass.PER_DEGREE]

    var merge_band: ScratchPhase["merge", "merge"]
    var merge_segments: ScratchBuffer[MergeSegment, 1, ScaleClass.PER_WORKER_PER_DEGREE]


@fieldwise_init
struct MinimaxM3MsaScratch[batching_seq_len: Int](
    ScratchIsland, Copyable, ImplicitlyCopyable
):
    comptime MAX_INDEX_BLOCKS = (Self.batching_seq_len - 1) // C.INDEX_BLOCK + 1
    comptime MAX_BLOCK_STRIDE = (Self.MAX_INDEX_BLOCKS + 15) // 16 * 16

    comptime PHASES = ScratchPhaseOrder[
        "qkv", "index_score", "block_select", "sparse_flash",
    ]

    var q_band: ScratchPhase["qkv", "sparse_flash"]
    var q: ScratchBuffer[BFloat16, PAGE_LEN * C.Q_DIM, ScaleClass.FIXED]

    var kv_band: ScratchPhase["qkv", "qkv"]
    var kv: ScratchBuffer[BFloat16, PAGE_LEN * C.KV_DIM * 2, ScaleClass.FIXED]

    var index_q_band: ScratchPhase["qkv", "index_score"]
    var index_q: ScratchBuffer[BFloat16, PAGE_LEN * C.INDEX_Q_DIM, ScaleClass.FIXED]
    var index_k: ScratchBuffer[BFloat16, PAGE_LEN * C.INDEX_K_DIM, ScaleClass.FIXED]

    var score_band: ScratchPhase["index_score", "block_select"]
    var index_scores: ScratchBuffer[
        Float32, PAGE_LEN * C.INDEX_NUM_HEADS * Self.MAX_BLOCK_STRIDE,
        ScaleClass.FIXED,
    ]

    var block_band: ScratchPhase["block_select", "sparse_flash"]
    var block_idx: ScratchBuffer[
        Int32, PAGE_LEN * C.INDEX_NUM_HEADS * C.INDEX_TOPK_BLOCKS, ScaleClass.FIXED,
    ]

    var partials_band: ScratchPhase["sparse_flash", "sparse_flash"]
    var partials: ScratchBuffer[Float32, PAGE_LEN * FULL_PARTIAL_STRIDE, ScaleClass.FIXED]

    var q_local_band: ScratchPhase["sparse_flash", "sparse_flash"]
    var q_local: ScratchBuffer[BFloat16, PAGE_LEN * C.Q_DIM, ScaleClass.PER_DEGREE]

    var merge_band: ScratchPhase["sparse_flash", "sparse_flash"]
    var merge_segments: ScratchBuffer[MergeSegment, 1, ScaleClass.PER_WORKER_PER_DEGREE]


@fieldwise_init
struct MinimaxM3DenseMlpScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder["gate_up", "down"]

    var gate_band: ScratchPhase["gate_up", "down"]
    var gate: ScratchBuffer[BFloat16, PAGE_LEN * C.DENSE_INTERMEDIATE, ScaleClass.PER_DEGREE]

    var up_band: ScratchPhase["gate_up", "gate_up"]
    var up: ScratchBuffer[BFloat16, PAGE_LEN * C.DENSE_INTERMEDIATE, ScaleClass.PER_DEGREE]

    var out_band: ScratchPhase["down", "down"]
    var dense_out: ScratchBuffer[BFloat16, PAGE_LEN * C.HIDDEN, ScaleClass.FIXED]


@fieldwise_init
struct MinimaxM3MoeScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASE1_TILE_J = 64
    comptime PHASE1_MR = 4

    comptime PHASES = ScratchPhaseOrder[
        "router", "setup", "phase1", "phase2", "shared",
    ]

    var router_band: ScratchPhase["router", "router"]
    var cands: ScratchBuffer[M3RouterCandidate, PAGE_LEN * C.TOP_K, ScaleClass.PER_WORKER]

    var setup_band: ScratchPhase["router", "phase2"]
    var route_idx: ScratchBuffer[Int32, PAGE_LEN * C.TOP_K, ScaleClass.FIXED]
    var route_w: ScratchBuffer[Float32, PAGE_LEN * C.TOP_K, ScaleClass.FIXED]
    var expert_offset: ScratchBuffer[Int32, C.NUM_EXPERTS + 1, ScaleClass.FIXED]
    var routes: ScratchBuffer[SparseRoute, PAGE_LEN * C.TOP_K, ScaleClass.FIXED]

    var hidden_band: ScratchPhase["phase1", "phase2"]
    var hidden_bucket: ScratchBuffer[
        BFloat16, PAGE_LEN * C.TOP_K * C.MOE_INTERMEDIATE, ScaleClass.FIXED,
    ]

    var gate_band: ScratchPhase["phase1", "phase1"]
    var gate_scratch: ScratchBuffer[
        Float32, Self.PHASE1_MR * 2 * Self.PHASE1_TILE_J, ScaleClass.PER_WORKER,
    ]

    var accum_band: ScratchPhase["phase2", "phase2"]
    var moe_accum: ScratchBuffer[Float32, PAGE_LEN * C.HIDDEN, ScaleClass.FIXED]

    var out_band: ScratchPhase["phase2", "shared"]
    var moe_out: ScratchBuffer[BFloat16, PAGE_LEN * C.HIDDEN, ScaleClass.FIXED]

    var shared_band: ScratchPhase["shared", "shared"]
    var shared_gate: ScratchBuffer[BFloat16, PAGE_LEN * C.SHARED_INTERMEDIATE, ScaleClass.PER_DEGREE]
    var shared_up: ScratchBuffer[BFloat16, PAGE_LEN * C.SHARED_INTERMEDIATE, ScaleClass.PER_DEGREE]
    var shared_out: ScratchBuffer[BFloat16, PAGE_LEN * C.HIDDEN, ScaleClass.FIXED]


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


def attn_qkv_project_norm_rope[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    ctx: BindContext[o],
    attn_ctx: BindContext[o],
    attn: AttnRefs[R],
    xs: Binding[BFloat16, o],
    q_outs: Binding[BFloat16, o],
    k_outs: Binding[BFloat16, o],
    v_outs: Binding[BFloat16, o],
    k_kv: Binding[BFloat16, o],
    v_kv: Binding[BFloat16, o],
    cos: Binding[Float32, o],
    sin: Binding[Float32, o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var degree = ctx.degree()
    comptime head_dim = C.HEAD_DIM
    comptime sqrt_hd = sqrt[DType.float32, 1](head_dim)
    comptime hd_eps = Float32(head_dim) * C.RMS_NORM_EPS
    comptime attn_scale = Float32(1.0) / sqrt_hd
    comptime q_norm_sqrt_n = sqrt_hd * attn_scale
    comptime num_q_heads = C.NUM_HEADS
    comptime num_kv_heads = C.NUM_KV_HEADS

    dispatch_gemm[cols=C.HIDDEN, max_worker_count=max_worker_count](
        xs, attn.q_proj.binding(attn_ctx), q_outs, C.Q_DIM, seq_len, pools, prof)
    dispatch_gemm[cols=C.HIDDEN, max_worker_count=max_worker_count](
        xs, attn.k_proj.binding(attn_ctx), k_outs, C.KV_DIM, seq_len, pools, prof)
    dispatch_gemm[cols=C.HIDDEN, max_worker_count=max_worker_count](
        xs, attn.v_proj.binding(attn_ctx), v_outs, C.KV_DIM, seq_len, pools, prof)

    dispatch_rms_norm[
        hidden=head_dim, sqrt_n=q_norm_sqrt_n, n_eps=hd_eps,
        max_worker_count=max_worker_count,
    ](q_outs, q_outs, attn.q_norm.binding(attn_ctx),
      seq_len * num_q_heads, pools, prof)
    dispatch_rms_norm[
        hidden=head_dim, sqrt_n=sqrt_hd, n_eps=hd_eps,
        max_worker_count=max_worker_count,
    ](k_outs, k_outs, attn.k_norm.binding(attn_ctx),
      seq_len * num_kv_heads, pools, prof)

    var rows_per_page = PAGE_LEN // degree
    var page_shift = pow2_shift(rows_per_page)
    var row_mask = rows_per_page - 1

    dispatch_rope_cache_write[
        half=C.ROPE_HALF, pair_stride=C.ROPE_HALF, head_dim=head_dim,
        max_worker_count=max_worker_count,
    ](q_outs, k_outs, v_outs, k_kv, v_kv, cos, sin,
      runs, num_q_heads, num_kv_heads, degree,
      page_shift, row_mask, -1, seq_len, pools, prof)


def dispatch_full_attention_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_seq_len: Int, batching_seq_len: Int,
    max_worker_count: Int = 128,
](
    layout: MinimaxM3Layout[
        PassthroughRecipes,
        FullKVSlots[batching_seq_len],
        IndexKSlots[batching_seq_len],
        max_seq_len,
    ],
    ctx: BindContext[o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
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
    comptime num_q_heads = C.NUM_HEADS
    var local_q_rows = MinimaxM3Shapes.O.data_m(degree)
    var local_num_q_heads = local_q_rows // head_dim

    var attn_ctx = ctx.with_layer(layout.dense.base(local_idx))
    var attn = layout.dense.proto.attn
    var kv_ctx = ctx.with_layer(layout.full_kv.base(layer_idx))
    var k_kv = layout.full_kv.proto.k.binding(kv_ctx)
    var v_kv = layout.full_kv.proto.v.binding(kv_ctx)

    var q_outs = scratch.binding[MinimaxM3FullAttnScratch, "q"](ctx, plan)
    var k_outs = scratch.binding[MinimaxM3FullAttnScratch, "kv"](ctx, plan)
    var v_outs = k_outs.shifted(seq_len * C.KV_DIM)
    var xs = layout.activations.x_residual.state_binding(ctx)

    attn_qkv_project_norm_rope[max_worker_count=max_worker_count](
        ctx, attn_ctx, attn, xs, q_outs, k_outs, v_outs, k_kv, v_kv,
        layout.main_rope.cos.state_binding(ctx),
        layout.main_rope.sin.state_binding(ctx),
        runs, seq_len, pools, prof)

    var q_local = scratch.binding[MinimaxM3FullAttnScratch, "q_local"](ctx, plan)
    var partials = scratch.binding[MinimaxM3FullAttnScratch, "partials"](ctx, plan)
    var merge_segments = scratch.binding[
        MinimaxM3FullAttnScratch, "merge_segments"](ctx, plan)

    dispatch_full_attention[
        head_dim=head_dim, num_q=num_q_heads, gqa_ratio=C.GQA_RATIO,
        kv_stride=C.KV_DIM, partial_stride=FULL_PARTIAL_STRIDE, page_len=PAGE_LEN,
        max_worker_count=max_worker_count,
    ](q_outs, k_kv, v_kv, q_local, partials, merge_segments,
      runs, local_num_q_heads, seq_len, pools, prof)

    dispatch_gemm_cols[rows=C.HIDDEN, max_worker_count=max_worker_count](
        q_local, attn.o_proj.binding(attn_ctx), xs, local_q_rows, seq_len,
        pools, prof)


def dispatch_lightning_indexer[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_seq_len: Int, batching_seq_len: Int,
    max_worker_count: Int = 128,
](
    layout: MinimaxM3Layout[
        PassthroughRecipes,
        FullKVSlots[batching_seq_len],
        IndexKSlots[batching_seq_len],
        max_seq_len,
    ],
    ctx: BindContext[o],
    index_runs: UnsafePointer[KVRunTable, MutAnyOrigin],
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


def dispatch_msa_attention_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_seq_len: Int, batching_seq_len: Int,
    max_worker_count: Int = 128,
](
    layout: MinimaxM3Layout[
        PassthroughRecipes,
        FullKVSlots[batching_seq_len],
        IndexKSlots[batching_seq_len],
        max_seq_len,
    ],
    ctx: BindContext[o],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    index_runs: UnsafePointer[KVRunTable, MutAnyOrigin],
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
    var local_q_rows = MinimaxM3Shapes.O.data_m(degree)

    var attn_ctx = ctx.with_layer(layout.sparse.base(local_idx))
    var attn = layout.sparse.proto.attn
    var kv_ctx = ctx.with_layer(layout.full_kv.base(layer_idx))
    var k_kv = layout.full_kv.proto.k.binding(kv_ctx)
    var v_kv = layout.full_kv.proto.v.binding(kv_ctx)

    var q_outs = scratch.binding[SC, "q"](ctx, plan)
    var k_outs = scratch.binding[SC, "kv"](ctx, plan)
    var v_outs = k_outs.shifted(seq_len * C.KV_DIM)
    var xs = layout.activations.x_residual.state_binding(ctx)

    attn_qkv_project_norm_rope[max_worker_count=max_worker_count](
        ctx, attn_ctx, attn, xs, q_outs, k_outs, v_outs, k_kv, v_kv,
        layout.main_rope.cos.state_binding(ctx),
        layout.main_rope.sin.state_binding(ctx),
        runs, seq_len, pools, prof)

    dispatch_lightning_indexer[
        max_seq_len=max_seq_len, batching_seq_len=batching_seq_len,
        max_worker_count=max_worker_count,
    ](layout, ctx, index_runs, seq_len, local_idx, scratch, plan, pools, prof)

    var block_idx = scratch.binding[SC, "block_idx"](ctx, plan)
    var q_local = scratch.binding[SC, "q_local"](ctx, plan)
    var partials = scratch.binding[SC, "partials"](ctx, plan)
    var merge_segments = scratch.binding[SC, "merge_segments"](ctx, plan)

    dispatch_minimax_m3_sparse_attention[
        page_len=PAGE_LEN, max_worker_count=max_worker_count,
    ](q_outs, k_kv, v_kv, block_idx, q_local, partials, merge_segments,
      runs, seq_len, pools, prof)

    dispatch_gemm_cols[rows=C.HIDDEN, max_worker_count=max_worker_count](
        q_local, attn.o_proj.binding(attn_ctx), xs, local_q_rows, seq_len,
        pools, prof)


comptime M3_SWIGLU_ALPHA = Float32(1.702)
comptime M3_SWIGLU_LIMIT = Float32(7.0)


def dispatch_swiglu_oai_gate_up[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    gate: Binding[BFloat16, o],
    up: Binding[BFloat16, o],
    dst: Binding[BFloat16, o],
    intermediate: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    dispatch_gate_up_act[
        activation="swiglu_oai", alpha=M3_SWIGLU_ALPHA, limit=M3_SWIGLU_LIMIT,
        max_worker_count=max_worker_count,
    ](gate, up, dst, intermediate, seq_len, pools, prof)


def dispatch_dense_mlp[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    ctx: BindContext[o],
    mlp: DenseMlpRefs[R],
    x_in: Binding[BFloat16, o],
    seq_len: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var degree = ctx.degree()
    var intermediate_per_rank = MinimaxM3Shapes.DenseGateUp.data_n(degree)

    var gate = scratch.binding[MinimaxM3DenseMlpScratch, "gate"](ctx, plan)
    var up = scratch.binding[MinimaxM3DenseMlpScratch, "up"](ctx, plan)
    var dense_out = scratch.binding[MinimaxM3DenseMlpScratch, "dense_out"](ctx, plan)

    dispatch_gemm[cols=C.HIDDEN, max_worker_count=max_worker_count](
        x_in, mlp.gate_proj.binding(ctx), gate, intermediate_per_rank, seq_len,
        pools, prof)
    dispatch_gemm[cols=C.HIDDEN, max_worker_count=max_worker_count](
        x_in, mlp.up_proj.binding(ctx), up, intermediate_per_rank, seq_len,
        pools, prof)
    dispatch_swiglu_oai_gate_up[max_worker_count=max_worker_count](
        gate, up, gate, intermediate_per_rank, seq_len, pools, prof)
    dispatch_gemm_cols[rows=C.HIDDEN, max_worker_count=max_worker_count](
        gate, mlp.down_proj.binding(ctx), dense_out, intermediate_per_rank,
        seq_len, pools, prof)
    dispatch_allreduce_inplace[BF16, max_worker_count=max_worker_count](
        dense_out, seq_len * C.HIDDEN, pools, prof)


def dispatch_m3_router[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    ctx: BindContext[o],
    moe: MoeRefs[R],
    x_input: Binding[BFloat16, o],
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

    dispatch_minimax_m3_router[max_worker_count=max_worker_count](
        x_input,
        moe.router_gate.binding(ctx),
        moe.router_bias.binding(ctx),
        cands, route_idx, route_w,
        experts_per_rank, seq_len, pools, prof)


def dispatch_moe[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    ctx: BindContext[o],
    moe: MoeRefs[R],
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
    var shared_inter_per_rank = MinimaxM3Shapes.SharedGateUp.data_n(degree)

    var route_idx = scratch.binding[MinimaxM3MoeScratch, "route_idx"](ctx, plan)
    var route_w = scratch.binding[MinimaxM3MoeScratch, "route_w"](ctx, plan)
    var expert_offset = scratch.binding[
        MinimaxM3MoeScratch, "expert_offset"](ctx, plan)
    var routes = scratch.binding[MinimaxM3MoeScratch, "routes"](ctx, plan)
    var hidden_bucket = scratch.binding[
        MinimaxM3MoeScratch, "hidden_bucket"](ctx, plan)
    var gate_scratch = scratch.binding[
        MinimaxM3MoeScratch, "gate_scratch"](ctx, plan)
    var moe_accum = scratch.binding[MinimaxM3MoeScratch, "moe_accum"](ctx, plan)
    var shared_gate = scratch.binding[
        MinimaxM3MoeScratch, "shared_gate"](ctx, plan)
    var shared_up = scratch.binding[MinimaxM3MoeScratch, "shared_up"](ctx, plan)
    var shared_out = scratch.binding[MinimaxM3MoeScratch, "shared_out"](ctx, plan)

    dispatch_build_expert_schedules[
        C.NUM_EXPERTS, C.TOP_K, max_worker_count=max_worker_count,
    ](route_idx, route_w, expert_offset, routes,
      experts_per_rank, seq_len, pools, prof)

    dispatch_phase1_gate_up[
        hidden=C.HIDDEN, gate_up_fused=2 * C.MOE_INTERMEDIATE,
        intermediate=C.MOE_INTERMEDIATE,
        activation="swiglu_oai", alpha=M3_SWIGLU_ALPHA, limit=M3_SWIGLU_LIMIT,
        max_worker_count=max_worker_count,
    ](x_input, expert_offset, routes, moe.experts_gate_up.binding(ctx),
      gate_scratch, hidden_bucket, experts_per_rank, pools, prof)

    dispatch_phase2_down[
        hidden=C.HIDDEN, intermediate=C.MOE_INTERMEDIATE,
        max_worker_count=max_worker_count,
    ](expert_offset, routes, hidden_bucket, moe.experts_down.binding(ctx),
      moe_accum, moe_out, experts_per_rank, seq_len, pools, prof)

    dispatch_allreduce_inplace[BF16, max_worker_count=max_worker_count](
        moe_out, seq_len * C.HIDDEN, pools, prof)

    dispatch_gemm[cols=C.HIDDEN, max_worker_count=max_worker_count](
        x_input, moe.shared_gate.binding(ctx), shared_gate,
        shared_inter_per_rank, seq_len, pools, prof)
    dispatch_gemm[cols=C.HIDDEN, max_worker_count=max_worker_count](
        x_input, moe.shared_up.binding(ctx), shared_up,
        shared_inter_per_rank, seq_len, pools, prof)
    dispatch_swiglu_oai_gate_up[max_worker_count=max_worker_count](
        shared_gate, shared_up, shared_gate, shared_inter_per_rank, seq_len,
        pools, prof)
    dispatch_gemm_cols[rows=C.HIDDEN, max_worker_count=max_worker_count](
        shared_gate, moe.shared_down.binding(ctx), shared_out,
        shared_inter_per_rank, seq_len, pools, prof)
    dispatch_allreduce_inplace[BF16, max_worker_count=max_worker_count](
        shared_out, seq_len * C.HIDDEN, pools, prof)

    dispatch_residual_add[hidden=C.HIDDEN, max_worker_count=max_worker_count](
        moe_out, shared_out, moe_out, seq_len, pools, prof)


def show_weight[
    dt: DType, //,
](label: StaticString, p: UnsafePointer[Scalar[dt], MutAnyOrigin]):
    print(label, p[0])


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
        PassthroughRecipes,
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
            PassthroughRecipes,
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

    def model_init(mut self):
        prime_fp_environment(self.pools)
        minimax_m3_init_rope_tables(self.layout, self.arena_bases)
        for rank in range(self.degree):
            var base = self.arena_bases[rank]
            for i in range(C.NUM_LAYERS):
                var entry = LAYER_SCHEDULE[i]
                if entry.kind == LayerKind.DENSE:
                    var lb = base + self.layout.dense.base(entry.local_idx)
                    bake_minimax_gain_inplace(
                        self.layout.dense.proto.input_norm.at(lb), C.HIDDEN)
                    bake_minimax_gain_inplace(
                        self.layout.dense.proto.post_attn_norm.at(lb), C.HIDDEN)
                    bake_minimax_gain_inplace(
                        self.layout.dense.proto.attn.q_norm.at(lb), C.HEAD_DIM)
                    bake_minimax_gain_inplace(
                        self.layout.dense.proto.attn.k_norm.at(lb), C.HEAD_DIM)
                else:
                    var lb = base + self.layout.sparse.base(entry.local_idx)
                    bake_minimax_gain_inplace(
                        self.layout.sparse.proto.input_norm.at(lb), C.HIDDEN)
                    bake_minimax_gain_inplace(
                        self.layout.sparse.proto.post_attn_norm.at(lb), C.HIDDEN)
                    bake_minimax_gain_inplace(
                        self.layout.sparse.proto.attn.q_norm.at(lb), C.HEAD_DIM)
                    bake_minimax_gain_inplace(
                        self.layout.sparse.proto.attn.k_norm.at(lb), C.HEAD_DIM)
                    bake_minimax_gain_inplace(
                        self.layout.sparse.proto.indexer.index_q_norm.at(lb),
                        C.INDEX_HEAD_DIM)
                    bake_minimax_gain_inplace(
                        self.layout.sparse.proto.indexer.index_k_norm.at(lb),
                        C.INDEX_HEAD_DIM)
            var tail_base = base + self.layout.tail.base(0)
            bake_minimax_gain_inplace(
                self.layout.tail.proto.final_norm.at(tail_base), C.HIDDEN)

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

        var buf_starts = pack_slot_starts(schedule)
        var emit_plan = collect_emit_plan(schedule, buf_starts)
        var num_emit = emit_plan.count()
        self.run_prefix_copies(schedule)
        self.bind_step_runs(schedule, pages)
        var full_runs = UnsafePointer(to=self.full_runs).as_unsafe_any_origin()
        var index_runs = UnsafePointer(to=self.index_runs).as_unsafe_any_origin()

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

                dispatch_rms_norm[hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps](
                    x_main, x_res, dl.input_norm.binding(layer_ctx),
                    total, self.pools, self.profiler)

                dispatch_full_attention_qkv[
                    max_seq_len=Self.max_seq_len,
                    batching_seq_len=Self.batching_seq_len,
                ](layout, ctx, full_runs, total, entry.idx, entry.local_idx,
                  self.scratch, self.full_plan, self.pools, self.profiler)

                dispatch_allreduce_inplace[BF16](
                    x_res, total * C.HIDDEN, self.pools, self.profiler)
                dispatch_residual_add[hidden=C.HIDDEN](
                    x_main, x_res, x_main, total, self.pools, self.profiler)

                dispatch_rms_norm[hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps](
                    x_main, x_res, dl.post_attn_norm.binding(layer_ctx),
                    total, self.pools, self.profiler)

                dispatch_dense_mlp(
                    layer_ctx, dl.mlp, x_res, total,
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

                dispatch_rms_norm[hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps](
                    x_main, x_res, sl.input_norm.binding(layer_ctx),
                    total, self.pools, self.profiler)

                dispatch_msa_attention_qkv[
                    max_seq_len=Self.max_seq_len,
                    batching_seq_len=Self.batching_seq_len,
                ](layout, ctx, full_runs, index_runs, total,
                  entry.idx, entry.local_idx,
                  self.scratch, self.msa_plan, self.pools, self.profiler)

                dispatch_allreduce_inplace[BF16](
                    x_res, total * C.HIDDEN, self.pools, self.profiler)
                dispatch_residual_add[hidden=C.HIDDEN](
                    x_main, x_res, x_main, total, self.pools, self.profiler)

                dispatch_rms_norm[hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps](
                    x_main, x_res, sl.post_attn_norm.binding(layer_ctx),
                    total, self.pools, self.profiler)

                dispatch_m3_router(
                    layer_ctx, sl.moe, x_res, total,
                    self.scratch, self.moe_plan, self.pools, self.profiler)
                var moe_out = self.scratch.binding[
                    MinimaxM3MoeScratch, "moe_out"](ctx, self.moe_plan)
                dispatch_moe(
                    layer_ctx, sl.moe, x_res, moe_out, total,
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

            dispatch_rms_norm[hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps](
                x_head, x_head,
                layout.tail.proto.final_norm.binding(tail_ctx),
                num_emit, self.pools, self.profiler)

            var out_ptr = outcome[0]
            dispatch_flash_sample[
                cols=C.HIDDEN, cap=Float64(0.0),
                n_max=MAXIMUM_SAMPLING_LOGITS,
            ](x_head, layout.tail.proto.lm_head.binding(tail_ctx),
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
            PassthroughRecipes,
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
        model.model_init()
        return model^

    def check_weights(self):
        var base = self.arena_bases[0]

        var tail_off = self.layout.tail.base(0)
        var tail = self.layout.tail.proto
        show_weight("tail.embed[0]      ", tail.embed.at(base + tail_off))
        show_weight("tail.lm_head[0]    ", tail.lm_head.at(base + tail_off))
        show_weight("tail.final_norm[0] ", tail.final_norm.at(base + tail_off))

        var d0 = self.layout.dense.base(0)
        var dense = self.layout.dense.proto
        show_weight("L0.input_norm[0]   ", dense.input_norm.at(base + d0))
        show_weight("L0.attn.q_proj[0]  ", dense.attn.q_proj.at(base + d0))
        show_weight("L0.attn.q_norm[0]  ", dense.attn.q_norm.at(base + d0))
        show_weight("L0.mlp.gate[0]     ", dense.mlp.gate_proj.at(base + d0))

        var s0 = self.layout.sparse.base(0)
        var sp = self.layout.sparse.proto
        show_weight("L3.attn.q_proj[0]  ", sp.attn.q_proj.at(base + s0))
        show_weight("L3.index_q_proj[0] ", sp.indexer.index_q_proj.at(base + s0))
        show_weight("L3.index_k_proj[0] ", sp.indexer.index_k_proj.at(base + s0))
        show_weight("L3.router_gate[0]  ", sp.moe.router_gate.at(base + s0))
        show_weight("L3.router_bias[0]  ", sp.moe.router_bias.at(base + s0))
        show_weight("L3.shared_gate[0]  ", sp.moe.shared_gate.at(base + s0))

        var experts = sp.moe.experts_gate_up.at(base + s0)
        show_weight("L3.expert0.w1[0]   ", experts)
        show_weight("L3.expert0.w3[0]   ", experts + C.MOE_INTERMEDIATE * C.HIDDEN)
