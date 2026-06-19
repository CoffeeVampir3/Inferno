from std.os import abort
from std.pathlib import Path
from std.memory import Span, UnsafePointer
from std.sys.info import simd_width_of
from simd_math.ops import sqrt

from numa import NumaArena, NumaTopology
from threading import BurstPool
from threading.threading_traits import BurstThreadPool
from kernels.helpers import RankView, Binding
from kernels.attention_ops import KVRunTable
from kernels.flash_sample import (
    SamplingParams, SampleAccum, SampleOutcome,
)
from kernels.logsum_merge import MergeSegment
from kernels.moe_router import RouterCandidate, SparseRoute
from kernels.profiling import Profiler
from kernels.rope import init_rope_table, init_rope_table_partial_strided

from modeling.temporal_scratch import (
    ScratchBuffer, ScratchIsland, ScratchPhaseOrder, ScratchPhase, ScaleClass,
    TemporalScratchPool, ScratchPlan,
    derive_checked_plan, aggregate_scratch_peak,
)
from modeling.model_spec import (
    BF16, F32,
    Shape, WeightDesc,
    Replicated, TensorRowSharded, TensorColumnSharded,
    ContextRowSharded, ExpertRowBlockSharded, VocabularyRowSharded,
    DEFAULT_ALIGNMENT, align_up,
)
from modeling.modeling_common import (
    Repeated, ArenaLayout,
)
from modeling.slot import (
    Slot, SlotGroup, SourceSpec, BindContext, stamp_offsets, emit_descs,
)
from modeling.gemma4_topology import KVSlotGroup
from modeling.loader import discover_shards, load_weights_from_descs
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


struct MinimaxM3Config:
    comptime HIDDEN = 6144
    comptime NUM_LAYERS = 60
    comptime NUM_DENSE_LAYERS = 3
    comptime NUM_SPARSE_LAYERS = 57

    comptime NUM_HEADS = 64
    comptime NUM_KV_HEADS = 4
    comptime HEAD_DIM = 128
    comptime Q_DIM = 8192
    comptime KV_DIM = 512
    comptime GQA_RATIO = Self.NUM_HEADS // Self.NUM_KV_HEADS

    comptime ROPE_HALF = 32
    comptime ROPE_THETA = 5000000.0

    comptime DENSE_INTERMEDIATE = 12288
    comptime SHARED_INTERMEDIATE = 3072
    comptime MOE_INTERMEDIATE = 3072
    comptime MOE_GATE_UP_FUSED = 2 * Self.MOE_INTERMEDIATE
    comptime NUM_EXPERTS = 128
    comptime TOP_K = 4
    comptime ROUTED_SCALING = 2.0

    comptime INDEX_NUM_HEADS = 4
    comptime INDEX_HEAD_DIM = 128
    comptime INDEX_Q_DIM = 512
    comptime INDEX_K_DIM = 128
    comptime INDEX_BLOCK = 128
    comptime INDEX_TOPK_BLOCKS = 16
    comptime INDEX_LOCAL_BLOCKS = 1
    comptime INDEX_ROPE_HALF = 64

    comptime SWIGLU_ALPHA = 1.702
    comptime SWIGLU_LIMIT = 7.0

    comptime VOCAB_SIZE = 200064
    comptime RMS_NORM_EPS = 1e-6


comptime C = MinimaxM3Config


struct LayerKind:
    comptime DENSE = 0
    comptime SPARSE = 1


@fieldwise_init
struct LayerEntry(Copyable, ImplicitlyCopyable):
    var idx: Int
    var kind: Int
    var local_idx: Int


@always_inline
def build_layer_schedule() -> InlineArray[LayerEntry, C.NUM_LAYERS]:
    var out = InlineArray[LayerEntry, C.NUM_LAYERS](uninitialized=True)
    var di = 0
    var si = 0
    for i in range(C.NUM_LAYERS):
        if i < C.NUM_DENSE_LAYERS:
            out[i] = LayerEntry(idx=i, kind=LayerKind.DENSE, local_idx=di)
            di += 1
        else:
            out[i] = LayerEntry(idx=i, kind=LayerKind.SPARSE, local_idx=si)
            si += 1
    return out


comptime LAYER_SCHEDULE = build_layer_schedule()


comptime MAX_WORKERS = 128
comptime PAGE_LEN = 256
comptime CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM = 32

trait MinimaxM3Recipes:
    comptime Qkv: QuantRecipe
    comptime Out: QuantRecipe
    comptime IndexProj: QuantRecipe
    comptime DenseGateUp: QuantRecipe
    comptime DenseDown: QuantRecipe
    comptime Router: QuantRecipe
    comptime MoeGateUp: QuantRecipe
    comptime MoeDown: QuantRecipe
    comptime SharedGateUp: QuantRecipe
    comptime SharedDown: QuantRecipe
    comptime Embed: QuantRecipe
    comptime LmHead: QuantRecipe


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


comptime SplitGainPerRowCs[fwht: Int, gamma: StaticString]: QuantRecipe = PerRowQuant(
    fwht, SplitGamma(gamma), SingleSided(), PerRowCs(), VnniPacked(),
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
    comptime Router: QuantRecipe = RouterCenter(
        "block_sparse_moe.e_score_correction_bias")
    comptime MoeGateUp: QuantRecipe = SplitGainPerRowCs[
        128, "post_attention_layernorm.weight"]
    comptime MoeDown: QuantRecipe = PlainPerBlockCs[128]
    comptime SharedGateUp: QuantRecipe = SplitGainPerRowCs[
        128, "post_attention_layernorm.weight"]
    comptime SharedDown: QuantRecipe = PlainPerBlockCs[128]
    comptime Embed: QuantRecipe = Passthrough()
    comptime LmHead: QuantRecipe = HeadEmbed[128]


comptime R = PassthroughRecipes


def gen_gate_up[num_experts: Int](prefix: String, mut names: List[String]):
    for e in range(num_experts):
        var ep = prefix + String(t"block_sparse_moe.experts.{e}.")
        names.append(ep + "w1.weight")
        names.append(ep + "w3.weight")


def gen_down[num_experts: Int](prefix: String, mut names: List[String]):
    for e in range(num_experts):
        var ep = prefix + String(t"block_sparse_moe.experts.{e}.")
        names.append(ep + "w2.weight")


struct MinimaxM3Shapes:
    comptime Q       = TensorRowSharded[C.Q_DIM, C.HIDDEN]
    comptime K       = Replicated[C.KV_DIM, C.HIDDEN]
    comptime V       = Replicated[C.KV_DIM, C.HIDDEN]
    comptime O       = TensorColumnSharded[C.HIDDEN, C.Q_DIM]

    comptime IndexQ  = Replicated[C.INDEX_Q_DIM, C.HIDDEN]
    comptime IndexK  = Replicated[C.INDEX_K_DIM, C.HIDDEN]

    comptime DenseGateUp = TensorRowSharded[C.DENSE_INTERMEDIATE, C.HIDDEN]
    comptime DenseDown   = TensorColumnSharded[C.HIDDEN, C.DENSE_INTERMEDIATE]

    comptime RouterProj = ExpertRowBlockSharded[C.NUM_EXPERTS, 1, C.HIDDEN]
    comptime ExpertsGateUp = ExpertRowBlockSharded[
        C.NUM_EXPERTS, C.MOE_GATE_UP_FUSED, C.HIDDEN,
    ]
    comptime ExpertsDown = ExpertRowBlockSharded[
        C.NUM_EXPERTS, C.HIDDEN, C.MOE_INTERMEDIATE,
    ]
    comptime SharedGateUp = TensorRowSharded[C.SHARED_INTERMEDIATE, C.HIDDEN]
    comptime SharedDown   = TensorColumnSharded[C.HIDDEN, C.SHARED_INTERMEDIATE]


struct MinimaxM3TailShapes:
    comptime FinalNorm = Replicated[C.HIDDEN, 1]
    comptime Embed = VocabularyRowSharded[C.VOCAB_SIZE, C.HIDDEN]
    comptime LmHead = VocabularyRowSharded[C.VOCAB_SIZE, C.HIDDEN]


struct AttnRefs[R: MinimaxM3Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = MinimaxM3Shapes
    var q_proj: Slot[BF16, Self.S.Q, "self_attn.q_proj.weight", Self.R.Qkv]
    var k_proj: Slot[BF16, Self.S.K, "self_attn.k_proj.weight", Self.R.Qkv]
    var v_proj: Slot[BF16, Self.S.V, "self_attn.v_proj.weight", Self.R.Qkv]
    var o_proj: Slot[BF16, Self.S.O, "self_attn.o_proj.weight", Self.R.Out]
    var q_norm: Slot[BF16, Shape[C.HEAD_DIM, 1], "self_attn.q_norm.weight"]
    var k_norm: Slot[BF16, Shape[C.HEAD_DIM, 1], "self_attn.k_norm.weight"]


struct IndexerRefs[R: MinimaxM3Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = MinimaxM3Shapes
    var index_q_proj: Slot[BF16, Self.S.IndexQ, "self_attn.index_q_proj.weight", Self.R.IndexProj]
    var index_k_proj: Slot[BF16, Self.S.IndexK, "self_attn.index_k_proj.weight", Self.R.IndexProj]
    var index_q_norm: Slot[BF16, Shape[C.INDEX_HEAD_DIM, 1], "self_attn.index_q_norm.weight"]
    var index_k_norm: Slot[BF16, Shape[C.INDEX_HEAD_DIM, 1], "self_attn.index_k_norm.weight"]


struct DenseMlpRefs[R: MinimaxM3Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = MinimaxM3Shapes
    var gate_proj: Slot[BF16, Self.S.DenseGateUp, "mlp.gate_proj.weight", Self.R.DenseGateUp]
    var up_proj:   Slot[BF16, Self.S.DenseGateUp, "mlp.up_proj.weight", Self.R.DenseGateUp]
    var down_proj: Slot[BF16, Self.S.DenseDown,   "mlp.down_proj.weight", Self.R.DenseDown]


struct MoeRefs[R: MinimaxM3Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = MinimaxM3Shapes
    var router_gate: Slot[F32, Self.S.RouterProj, "block_sparse_moe.gate.weight", Self.R.Router]
    var router_bias: Slot[F32, Shape[C.NUM_EXPERTS, 1], "block_sparse_moe.e_score_correction_bias"]
    var experts_gate_up: Slot[
        BF16, Self.S.ExpertsGateUp,
        SourceSpec.grouped(2 * C.NUM_EXPERTS, gen_gate_up[C.NUM_EXPERTS]),
        quant=Self.R.MoeGateUp,
    ]
    var experts_down: Slot[
        BF16, Self.S.ExpertsDown,
        SourceSpec.grouped(C.NUM_EXPERTS, gen_down[C.NUM_EXPERTS]),
        quant=Self.R.MoeDown,
    ]
    var shared_gate: Slot[BF16, Self.S.SharedGateUp, "block_sparse_moe.shared_experts.gate_proj.weight", Self.R.SharedGateUp]
    var shared_up:   Slot[BF16, Self.S.SharedGateUp, "block_sparse_moe.shared_experts.up_proj.weight", Self.R.SharedGateUp]
    var shared_down: Slot[BF16, Self.S.SharedDown,   "block_sparse_moe.shared_experts.down_proj.weight", Self.R.SharedDown]


struct DenseLayerRefs[R: MinimaxM3Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    var input_norm:     Slot[BF16, Shape[C.HIDDEN, 1], "input_layernorm.weight"]
    var post_attn_norm: Slot[BF16, Shape[C.HIDDEN, 1], "post_attention_layernorm.weight"]
    var attn: AttnRefs[Self.R]
    var mlp: DenseMlpRefs[Self.R]


struct SparseLayerRefs[R: MinimaxM3Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    var input_norm:     Slot[BF16, Shape[C.HIDDEN, 1], "input_layernorm.weight"]
    var post_attn_norm: Slot[BF16, Shape[C.HIDDEN, 1], "post_attention_layernorm.weight"]
    var attn: AttnRefs[Self.R]
    var indexer: IndexerRefs[Self.R]
    var moe: MoeRefs[Self.R]


struct TailRefs[R: MinimaxM3Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = MinimaxM3TailShapes
    var final_norm: Slot[BF16, Self.S.FinalNorm, "language_model.model.norm.weight"]
    var embed:      Slot[BF16, Self.S.Embed, "language_model.model.embed_tokens.weight", Self.R.Embed]
    var lm_head:    Slot[BF16, Self.S.LmHead, "language_model.lm_head.weight", Self.R.LmHead]


struct FullKVSlots[batching_seq_len: Int](Copyable, ImplicitlyCopyable, KVSlotGroup):
    comptime CacheShape = ContextRowSharded[Self.batching_seq_len, C.KV_DIM]
    var k: Slot[BF16, Self.CacheShape]
    var v: Slot[BF16, Self.CacheShape]


struct IndexKSlots[batching_seq_len: Int](Copyable, ImplicitlyCopyable, KVSlotGroup):
    comptime CacheShape = ContextRowSharded[Self.batching_seq_len, C.INDEX_K_DIM]
    var k: Slot[BF16, Self.CacheShape]


struct RopeSlots[half: Int, max_seq_len: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    var cos: Slot[F32, Replicated[Self.max_seq_len, Self.half]]
    var sin: Slot[F32, Replicated[Self.max_seq_len, Self.half]]


struct ActivationSlots(Copyable, ImplicitlyCopyable, SlotGroup):
    var x_main:     Slot[BF16, Shape[PAGE_LEN, C.HIDDEN]]
    var x_residual: Slot[BF16, Shape[PAGE_LEN, C.HIDDEN]]


@fieldwise_init
struct MinimaxM3Layout[
    R: MinimaxM3Recipes, FKV: KVSlotGroup, IKV: KVSlotGroup, max_seq_len: Int,
](Copyable, ImplicitlyCopyable):
    var arena: ArenaLayout
    var dense: Repeated[DenseLayerRefs[Self.R]]
    var sparse: Repeated[SparseLayerRefs[Self.R]]
    var full_kv: Repeated[Self.FKV]
    var index_kv: Repeated[Self.IKV]
    var activations: ActivationSlots
    var main_rope: RopeSlots[C.ROPE_HALF, Self.max_seq_len]
    var index_rope: RopeSlots[C.INDEX_ROPE_HALF, Self.max_seq_len]
    var tail: Repeated[TailRefs[Self.R]]


@always_inline
def degree_contracts_ok(degree: Int) -> Bool:
    return (
        degree > 0
        and C.NUM_HEADS % degree == 0
        and C.Q_DIM % degree == 0
        and C.DENSE_INTERMEDIATE % degree == 0
        and C.SHARED_INTERMEDIATE % degree == 0
        and C.MOE_INTERMEDIATE % degree == 0
        and C.NUM_EXPERTS % degree == 0
        and C.VOCAB_SIZE % degree == 0
    )


def build_minimax_m3_plan[
    R: MinimaxM3Recipes, FKV: KVSlotGroup, IKV: KVSlotGroup,
    max_seq_len: Int, batching_seq_len: Int,
](
    degree: Int, max_workers: Int, scratch_cap: Int,
    mut descs: List[WeightDesc],
) -> MinimaxM3Layout[R, FKV, IKV, max_seq_len]:
    if not degree_contracts_ok(degree):
        abort(t"minimax_m3: degree {degree} does not divide the model dimensions")

    var dl_proto = DenseLayerRefs[R]()
    var dl_stride = stamp_offsets(dl_proto, degree)
    var sl_proto = SparseLayerRefs[R]()
    var sl_stride = stamp_offsets(sl_proto, degree)

    var dl_off = 0
    var sl_off = dl_off + C.NUM_DENSE_LAYERS * dl_stride
    var distributed = sl_off + C.NUM_SPARSE_LAYERS * sl_stride

    for i in range(C.NUM_LAYERS):
        var entry = LAYER_SCHEDULE[i]
        var prefix = String(t"language_model.model.layers.{entry.idx}.")
        if entry.kind == LayerKind.DENSE:
            _ = emit_descs[DenseLayerRefs[R]](
                prefix, dl_off + entry.local_idx * dl_stride, degree, descs)
        else:
            var region_base = sl_off + entry.local_idx * sl_stride
            _ = emit_descs[SparseLayerRefs[R]](prefix, region_base, degree, descs)

    var tail_proto = TailRefs[R]()
    var tail_bytes = stamp_offsets(tail_proto, degree)
    _ = emit_descs[TailRefs[R]]("", distributed, degree, descs)
    var tail = Repeated[TailRefs[R]](tail_proto, distributed, tail_bytes, 1)
    distributed += tail_bytes

    var state_cursor = distributed

    var fkv_proto = FKV()
    var fkv_stride = stamp_offsets(fkv_proto, degree)
    var full_kv = Repeated[FKV](
        fkv_proto, state_cursor, fkv_stride, C.NUM_DENSE_LAYERS + C.NUM_SPARSE_LAYERS)
    state_cursor = align_up(state_cursor + C.NUM_LAYERS * fkv_stride)

    var ikv_proto = IKV()
    var ikv_stride = stamp_offsets(ikv_proto, degree)
    var index_kv = Repeated[IKV](
        ikv_proto, state_cursor, ikv_stride, C.NUM_SPARSE_LAYERS)
    state_cursor = align_up(state_cursor + C.NUM_SPARSE_LAYERS * ikv_stride)

    var activations = ActivationSlots()
    state_cursor = stamp_offsets(activations, degree, state_cursor)

    state_cursor = align_up(state_cursor)
    var scratch_off = state_cursor
    state_cursor = align_up(state_cursor + scratch_cap)

    var main_rope = RopeSlots[C.ROPE_HALF, max_seq_len]()
    state_cursor = stamp_offsets(main_rope, degree, state_cursor)
    var index_rope = RopeSlots[C.INDEX_ROPE_HALF, max_seq_len]()
    state_cursor = stamp_offsets(index_rope, degree, state_cursor)

    var arena = ArenaLayout(
        distributed_bytes=distributed,
        state_bytes=state_cursor - distributed,
        host_bytes=align_up(state_cursor),
        scratch_off=scratch_off,
    )
    return MinimaxM3Layout[R, FKV, IKV, max_seq_len](
        arena=arena,
        dense=Repeated[DenseLayerRefs[R]](
            dl_proto, dl_off, dl_stride, C.NUM_DENSE_LAYERS),
        sparse=Repeated[SparseLayerRefs[R]](
            sl_proto, sl_off, sl_stride, C.NUM_SPARSE_LAYERS),
        full_kv=full_kv,
        index_kv=index_kv,
        activations=activations,
        main_rope=main_rope,
        index_rope=index_rope,
        tail=tail)


@fieldwise_init
struct MinimaxM3FullAttnScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder["prep", "flash", "merge"]

    var q_band: ScratchPhase["prep", "flash"]
    var q: ScratchBuffer[BFloat16, PAGE_LEN * C.Q_DIM, ScaleClass.FIXED]

    var kv_band: ScratchPhase["prep", "prep"]
    var kv: ScratchBuffer[BFloat16, PAGE_LEN * C.KV_DIM * 2, ScaleClass.FIXED]

    var partials_band: ScratchPhase["flash", "merge"]
    var partials: ScratchBuffer[Float32, PAGE_LEN * C.HEAD_DIM, ScaleClass.FIXED]

    var merge_band: ScratchPhase["merge", "merge"]
    var merge_segments: ScratchBuffer[MergeSegment, 1, ScaleClass.PER_WORKER_PER_DEGREE]


@fieldwise_init
struct MinimaxM3MsaScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
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
    var index_scores: ScratchBuffer[Float32, PAGE_LEN * C.INDEX_NUM_HEADS, ScaleClass.PER_WORKER]

    var block_band: ScratchPhase["block_select", "sparse_flash"]
    var block_idx: ScratchBuffer[Int32, PAGE_LEN * C.INDEX_TOPK_BLOCKS, ScaleClass.FIXED]

    var partials_band: ScratchPhase["sparse_flash", "sparse_flash"]
    var partials: ScratchBuffer[Float32, PAGE_LEN * C.HEAD_DIM, ScaleClass.FIXED]


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
    var router_scaled: ScratchBuffer[Float32, C.HIDDEN, ScaleClass.PER_WORKER]
    var cands: ScratchBuffer[RouterCandidate, PAGE_LEN * C.TOP_K, ScaleClass.PER_WORKER]

    var setup_band: ScratchPhase["router", "phase2"]
    var route_idx: ScratchBuffer[Int32, PAGE_LEN * C.TOP_K, ScaleClass.FIXED]
    var route_w: ScratchBuffer[Float32, PAGE_LEN * C.TOP_K, ScaleClass.FIXED]
    var x_normed: ScratchBuffer[BFloat16, PAGE_LEN * C.HIDDEN, ScaleClass.FIXED]
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

    var shared_band: ScratchPhase["shared", "shared"]
    var shared_gate: ScratchBuffer[BFloat16, PAGE_LEN * C.SHARED_INTERMEDIATE, ScaleClass.PER_DEGREE]
    var shared_up: ScratchBuffer[BFloat16, PAGE_LEN * C.SHARED_INTERMEDIATE, ScaleClass.PER_DEGREE]


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
struct MinimaxM3ForwardScratch(Copyable, ImplicitlyCopyable):
    var full: MinimaxM3FullAttnScratch
    var msa: MinimaxM3MsaScratch
    var dense_mlp: MinimaxM3DenseMlpScratch
    var moe: MinimaxM3MoeScratch
    var head: MinimaxM3HeadScratch


def calculate_peak_scratch(degree: Int, max_workers: Int) -> Int:
    return aggregate_scratch_peak[MinimaxM3ForwardScratch](degree, max_workers)


def dispatch_full_attention_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    ctx: BindContext[o],
    attn: AttnRefs[R],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    seq_len: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    abort("minimax_m3: dispatch_full_attention_qkv forward not implemented")


def dispatch_lightning_indexer[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    ctx: BindContext[o],
    indexer: IndexerRefs[R],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    seq_len: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    abort("minimax_m3: lightning indexer (scoring + block top-k) not implemented")


def dispatch_block_sparse_attention[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    ctx: BindContext[o],
    attn: AttnRefs[R],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    seq_len: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    abort("minimax_m3: block-sparse attention not implemented")


def dispatch_msa_attention_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    ctx: BindContext[o],
    layer: SparseLayerRefs[R],
    runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    index_runs: UnsafePointer[KVRunTable, MutAnyOrigin],
    seq_len: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    abort("minimax_m3: MSA attention forward not implemented")


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
    abort("minimax_m3: swiglu-oai activation not implemented")


def dispatch_dense_mlp[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    ctx: BindContext[o],
    mlp: DenseMlpRefs[R],
    x_main: Binding[BFloat16, o],
    x_residual: Binding[BFloat16, o],
    seq_len: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    abort("minimax_m3: dense MLP forward not implemented")


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
    abort("minimax_m3: sigmoid+bias router not implemented")


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
    abort("minimax_m3: MoE forward not implemented")


def show_weight[
    dt: DType, //,
](label: StaticString, p: UnsafePointer[Scalar[dt], MutAnyOrigin]):
    print(label, p[0])


def minimax_m3_load_arenas[
    R: MinimaxM3Recipes, FKV: KVSlotGroup, IKV: KVSlotGroup,
    max_seq_len: Int, batching_seq_len: Int,
](
    dir_path: Path,
    topo: NumaTopology,
    degree: Int,
    max_workers: Int,
    scratch_cap: Int,
    mut arenas: List[NumaArena[alignment=DEFAULT_ALIGNMENT]],
) -> Optional[MinimaxM3Layout[R, FKV, IKV, max_seq_len]]:
    var shards = discover_shards(dir_path)
    if len(shards) == 0:
        print(t"no safetensors shards found in {dir_path}")
        return None
    var n_shards = len(shards)
    print(t"found {n_shards} shard(s)")

    var descs = List[WeightDesc]()
    var layout = build_minimax_m3_plan[
        R, FKV, IKV, max_seq_len, batching_seq_len,
    ](degree, max_workers, scratch_cap, descs)

    var size = layout.arena.host_arena_bytes()
    var size_mb = size // (1024 * 1024)
    var weights_mb = layout.arena.distributed_bytes // (1024 * 1024)
    var state_mb = layout.arena.state_bytes // (1024 * 1024)
    print(
        t"allocating {size_mb} MB x {degree} rank(s) "
        t"({weights_mb} MB weights + {state_mb} MB state each)"
    )

    var arena_bases = List[Int]()
    for rank in range(degree):
        arenas.append(NumaArena[alignment=DEFAULT_ALIGNMENT](topo.node(rank), size))
        if not arenas[rank]:
            var node = topo.node(rank)
            print(t"arena allocation failed on node {node}")
            return None
        arena_bases.append(Int(arenas[rank].base.value()))

    var load_result = load_weights_from_descs(descs, shards, arena_bases, topo)
    if not load_result:
        print("weight loading failed")
        return None
    var loaded = load_result.take()
    var loaded_mb = loaded.bytes_loaded // (1024 * 1024)
    print(t"loaded {loaded_mb} MB in {loaded.num_ops} ops")

    for rank in range(degree):
        _ = arenas[rank].prefault(
            layout.arena.distributed_bytes, layout.arena.state_bytes)

    return layout


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
        self.full_runs = KVRunTable()
        self.index_runs = KVRunTable()
        self.full_plan = derive_checked_plan[MinimaxM3FullAttnScratch](degree, max_workers)
        self.msa_plan = derive_checked_plan[MinimaxM3MsaScratch](degree, max_workers)
        self.dense_mlp_plan = derive_checked_plan[MinimaxM3DenseMlpScratch](degree, max_workers)
        self.moe_plan = derive_checked_plan[MinimaxM3MoeScratch](degree, max_workers)
        self.head_plan = derive_checked_plan[MinimaxM3HeadScratch](degree, max_workers)
        self.profiler = Profiler[Self.profile, Self.profile_slots]()
        self.tokens_processed = 0

    def model_init(mut self):
        for rank in range(self.degree):
            var base = self.arena_bases[rank]
            init_rope_table_partial_strided[C.ROPE_HALF, Self.max_seq_len](
                self.layout.main_rope.cos.at(base),
                self.layout.main_rope.sin.at(base),
                C.ROPE_THETA, C.HEAD_DIM, 0, 1)
            init_rope_table[C.INDEX_ROPE_HALF, Self.max_seq_len](
                self.layout.index_rope.cos.at(base),
                self.layout.index_rope.sin.at(base),
                C.ROPE_THETA)

    def quant_model_init(mut self):
        """Post-load init for a ButterquantRecipes checkpoint. The bf16 path
        needs none of this (model_init covers RoPE); these are the steps a
        quantized load must run after weights land in the arena and before the
        first forward.

        1. Split-gain (butterquant §4.2/§4.4). Each SplitGamma weight pairs an
           offline weight-side factor sqrt(|gamma|) (baked into W by the
           quantizer) with an activation-side factor sigma = sign(gamma) *
           sqrt(|gamma|), epsilon-floored on the activation side only. gamma is
           the gemma (1 + w) gain, so sigma is formed from (1 + w), NOT the
           stored w; the offline get_gamma must apply the same (1 + w) or the two
           halves do not multiply back to gamma. This (1 + w) offset is the one
           correctness item flagged when wiring the recipes.

           MiniMax cannot reuse gemma's in-place norm bake. Both split norms also
           feed a non-split consumer: input_layernorm feeds the Passthrough
           indexer in addition to Qkv, and post_attention_layernorm feeds the
           gauged router in addition to the experts. Those consumers need the
           full gain (1 + w) * x / rms, so sigma must be applied at the split
           weight's activation prep, leaving the shared norm weight at (1 + w)
           for the full-gain consumers -- it is not folded into the norm weight.

        2. Colsum (butterquant §2.5). PerRowCs / PerBlockCs reserve a colsum
           member in the arena that the loader never fills. Compute it from the
           loaded int8 weights here: cs[n] = sum_k W_i8[n, k] (per-row) or
           cs[n, b] (per-block), per-expert for the grouped MoE / shared weights.
           NoColsum slots (LmHead) skip this.

        3. Router (butterquant §13.2/§13.4). RouterCenter weights are centered
           bf16 plus a bf16 gauge plus an f32 bias. The gauge pivot
           p = sum_k x_k g[k] is per-token (runtime), so init only confirms the
           gauge and bias sidecars resolved at their slot offsets.

        4. Shared with model_init: RoPE tables (main + index) and the floating
           point environment priming the quant kernels assume.
        """
        abort("minimax_m3: quant_model_init not implemented")

    def batch_geometry(self) -> BatchGeometry:
        abort("minimax_m3: batch_geometry not implemented (index KV pool spec pending)")

    def run_prefix_copies(mut self, read schedule: Schedule):
        abort("minimax_m3: run_prefix_copies not implemented")

    def bind_step_runs(
        mut self, read schedule: Schedule, read pages: KVPageAccountant,
    ):
        abort("minimax_m3: bind_step_runs not implemented")

    def execute(
        mut self,
        read schedule: Schedule,
        read pages: KVPageAccountant,
    ) -> List[SampleOutcome[MAXIMUM_SAMPLING_LOGITS]]:
        abort("minimax_m3: forward not implemented")

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
          calculate_peak_scratch(degree, max_workers), arenas)
        if not layout_opt:
            return None

        for r in range(degree):
            pools[r].wake()

        var model = Self(
            arenas^, pools^, layout_opt.take(), degree, max_workers)
        model.model_init()
        return model^

    @staticmethod
    def quantize[Rec: MinimaxM3Recipes = R](
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
                if not q.plan_walk[DenseLayerRefs[Rec]](prefix, entry.idx):
                    return False
            else:
                if not q.plan_walk[SparseLayerRefs[Rec]](prefix, entry.idx):
                    return False
        if not q.plan_walk[TailRefs[Rec]](String(""), -1):
            return False
        if not q.write_header():
            return False
        return q.execute(topo, pools^)

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
