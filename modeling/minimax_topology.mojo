from std.os import abort
from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from numa import NumaArena, NumaTopology
from kernels.attention_ops import flash_partial_stride
from kernels.rope import init_rope_table_partial_strided
from quant.recipe import QuantRecipe, NormGain
from continuous_batching.paging import PagePoolSpec
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
    Slot, SlotGroup, SourceSpec, stamp_offsets, emit_descs,
)
from modeling.gemma4_topology import KVSlotGroup
from modeling.loader import discover_shards, load_weights_from_descs
from modeling.kv_policy import KVPoolMirror, kv_components
from modeling.minimax_common import (
    MinimaxM3Config, LAYER_SCHEDULE, LayerKind,
)


comptime C = MinimaxM3Config
comptime MAX_WORKERS = 128
comptime PAGE_LEN = 256
comptime CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM = 32

comptime FULL_POOL = 0
comptime INDEX_POOL = 1

comptime FULL_PARTIAL_STRIDE = flash_partial_stride(C.NUM_HEADS, C.HEAD_DIM)

comptime NORM_GAIN_OFFSET = Float32(1.0)


def bake_minimax_gain_inplace(
    p: UnsafePointer[BFloat16, MutAnyOrigin], count: Int,
):
    comptime width = simd_width_of[DType.bfloat16]()
    var one = SIMD[DType.bfloat16, width](1.0)
    for j in range(0, count, width):
        var lane = p + j
        lane.store(lane.load[width=width]() + one)


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
    comptime Q       = Replicated[C.Q_DIM, C.HIDDEN]
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
    var q_norm: Slot[BF16, Shape[C.HEAD_DIM, 1], "self_attn.q_norm.weight", quant=NormGain(NORM_GAIN_OFFSET)]
    var k_norm: Slot[BF16, Shape[C.HEAD_DIM, 1], "self_attn.k_norm.weight", quant=NormGain(NORM_GAIN_OFFSET)]


struct IndexerRefs[R: MinimaxM3Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = MinimaxM3Shapes
    var index_q_proj: Slot[BF16, Self.S.IndexQ, "self_attn.index_q_proj.weight", Self.R.IndexProj]
    var index_k_proj: Slot[BF16, Self.S.IndexK, "self_attn.index_k_proj.weight", Self.R.IndexProj]
    var index_q_norm: Slot[BF16, Shape[C.INDEX_HEAD_DIM, 1], "self_attn.index_q_norm.weight", quant=NormGain(NORM_GAIN_OFFSET)]
    var index_k_norm: Slot[BF16, Shape[C.INDEX_HEAD_DIM, 1], "self_attn.index_k_norm.weight", quant=NormGain(NORM_GAIN_OFFSET)]


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
    var input_norm:     Slot[BF16, Shape[C.HIDDEN, 1], "input_layernorm.weight", quant=NormGain(NORM_GAIN_OFFSET)]
    var post_attn_norm: Slot[BF16, Shape[C.HIDDEN, 1], "post_attention_layernorm.weight", quant=NormGain(NORM_GAIN_OFFSET)]
    var attn: AttnRefs[Self.R]
    var mlp: DenseMlpRefs[Self.R]


struct SparseLayerRefs[R: MinimaxM3Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    var input_norm:     Slot[BF16, Shape[C.HIDDEN, 1], "input_layernorm.weight", quant=NormGain(NORM_GAIN_OFFSET)]
    var post_attn_norm: Slot[BF16, Shape[C.HIDDEN, 1], "post_attention_layernorm.weight", quant=NormGain(NORM_GAIN_OFFSET)]
    var attn: AttnRefs[Self.R]
    var indexer: IndexerRefs[Self.R]
    var moe: MoeRefs[Self.R]


struct TailRefs[R: MinimaxM3Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = MinimaxM3TailShapes
    var final_norm: Slot[BF16, Self.S.FinalNorm, "language_model.model.norm.weight", quant=NormGain(NORM_GAIN_OFFSET)]
    var embed:      Slot[BF16, Self.S.Embed, "language_model.model.embed_tokens.weight", Self.R.Embed]
    var lm_head:    Slot[BF16, Self.S.LmHead, "language_model.lm_head.weight", Self.R.LmHead]


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
        tail=tail)


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


def minimax_m3_kv_mirrors[
    R: MinimaxM3Recipes, FKV: KVSlotGroup, IKV: KVSlotGroup, max_seq_len: Int, //,
    batching_seq_len: Int,
](
    read layout: MinimaxM3Layout[R, FKV, IKV, max_seq_len], degree: Int,
) -> List[KVPoolMirror]:
    var mirrors = List[KVPoolMirror]()
    mirrors.append(KVPoolMirror(
        page_len=PAGE_LEN,
        pos_shard=degree,
        region_off=layout.full_kv.off,
        stride=layout.full_kv.stride,
        layers=layout.full_kv.count,
        components=kv_components(layout.full_kv.proto, degree),
        spec=PagePoolSpec(
            num_pages=batching_seq_len // PAGE_LEN,
            fixed_pages_per_seq=0,
            max_pages_per_seq=max_seq_len // PAGE_LEN)))
    mirrors.append(KVPoolMirror(
        page_len=PAGE_LEN,
        pos_shard=degree,
        region_off=layout.index_kv.off,
        stride=layout.index_kv.stride,
        layers=layout.index_kv.count,
        components=kv_components(layout.index_kv.proto, degree),
        spec=PagePoolSpec(
            num_pages=batching_seq_len // PAGE_LEN,
            fixed_pages_per_seq=0,
            max_pages_per_seq=max_seq_len // PAGE_LEN)))
    return mirrors^


def minimax_m3_init_rope_tables[
    R: MinimaxM3Recipes, FKV: KVSlotGroup, IKV: KVSlotGroup, max_seq_len: Int, //,
](
    read layout: MinimaxM3Layout[R, FKV, IKV, max_seq_len],
    read arena_bases: List[Int],
):
    for rank in range(len(arena_bases)):
        var base = arena_bases[rank]
        init_rope_table_partial_strided[C.ROPE_HALF, max_seq_len](
            layout.main_rope.cos.at(base),
            layout.main_rope.sin.at(base),
            C.ROPE_THETA, C.ROPE_ROTARY_DIM, 0, 1)
