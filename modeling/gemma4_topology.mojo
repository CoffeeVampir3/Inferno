from std.os import abort
from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from simd_math.ops import sqrt

from numa import NumaArena, NumaTopology
from butterquant.vnni import VNNI_N_STEP, VNNI_K_STEP
from kernels.rope import init_rope_table, init_rope_table_partial_strided
from quant.recipe import QuantRecipe
from continuous_batching.paging import PagePoolSpec
from modeling.model_spec import (
    BF16, F32,
    Shape, WeightDesc,
    Replicated,
    TensorRowSharded, TensorColumnSharded,
    ExpertRowBlockSharded, VocabularyRowSharded,
    DEFAULT_ALIGNMENT,
    align_up,
)
from modeling.gemma4_common import (
    Gemma4BaseConfig, LAYER_SCHEDULE, LayerKind,
)
from modeling.modeling_common import (
    Repeated, ArenaLayout,
)
from modeling.slot import (
    Slot, SlotGroup, stamp_offsets, emit_descs, vnni_pack_contract_ok,
)
from modeling.loader import discover_shards, load_weights_from_descs
from modeling.kv_policy import KVPoolMirror, kv_components


comptime C = Gemma4BaseConfig
comptime MAX_WORKERS = 128
comptime PAGE_LEN = C.SLIDING_WINDOW
comptime CONTINUOUS_BATCHING_MAX_SEQ_PARALLELISM = 32
comptime SLIDING_POOL = 0
comptime FULL_POOL = 1
comptime SLIDING_RING_PAGES = 2
comptime MAX_STEER_VECTORS = 16


trait Gemma4Recipes:
    comptime FFN_BLOCK: Int
    comptime SlidingQkv: QuantRecipe
    comptime SlidingOut: QuantRecipe
    comptime FullQkv: QuantRecipe
    comptime FullOut: QuantRecipe
    comptime DenseGateUp: QuantRecipe
    comptime DenseDown: QuantRecipe
    comptime Router: QuantRecipe
    comptime MoeGateUp: QuantRecipe
    comptime MoeDown: QuantRecipe
    comptime Embed: QuantRecipe


trait KVSlotGroup(
    SlotGroup, Copyable, ImplicitlyCopyable, ImplicitlyDeletable,
):
    pass


@always_inline
def degree_contracts_ok(degree: Int) -> Bool:
    return (
        degree > 0
        and C.NUM_HEADS % degree == 0
        and C.NUM_KV_HEADS_SLIDING % degree == 0
        and C.Q_DIM_SLIDING % degree == 0
        and C.KV_DIM_SLIDING % degree == 0
        and C.Q_DIM_FULL % degree == 0
        and C.INTERMEDIATE % degree == 0
        and C.NUM_EXPERTS % degree == 0
        and C.VOCAB_SIZE % degree == 0
    )


@always_inline
def paging_contracts_ok[
    max_seq_len: Int, batching_seq_len: Int, max_resident_seqs: Int,
](degree: Int) -> Bool:
    var rows_per_page = PAGE_LEN // degree
    return (
        PAGE_LEN % degree == 0
        and rows_per_page > 0
        and (rows_per_page & (rows_per_page - 1)) == 0
        and max_resident_seqs > 0
        and batching_seq_len % PAGE_LEN == 0
        and max_seq_len % PAGE_LEN == 0
        and batching_seq_len >= max_seq_len
    )


struct Gemma4Shapes[ffn_block: Int = 1]:
    comptime GateUp      = TensorRowSharded[
        C.INTERMEDIATE, C.HIDDEN, block=Self.ffn_block,
    ]
    comptime Down        = TensorColumnSharded[
        C.HIDDEN, C.INTERMEDIATE, block=Self.ffn_block,
    ]
    comptime SlidingQ    = TensorRowSharded[C.Q_DIM_SLIDING, C.HIDDEN]
    comptime SlidingKV   = TensorRowSharded[C.KV_DIM_SLIDING, C.HIDDEN]
    comptime SlidingO    = TensorColumnSharded[C.HIDDEN, C.Q_DIM_SLIDING]
    comptime FullQ       = Replicated[C.Q_DIM_FULL, C.HIDDEN]
    comptime FullK       = Replicated[C.KV_DIM_FULL, C.HIDDEN]
    comptime FullO       = TensorColumnSharded[C.HIDDEN, C.Q_DIM_FULL]
    comptime RouterProj  = ExpertRowBlockSharded[C.NUM_EXPERTS, 1, C.HIDDEN]
    comptime ExpertsGateUp = ExpertRowBlockSharded[
        C.NUM_EXPERTS, C.MOE_GATE_UP_FUSED, C.HIDDEN,
    ]
    comptime ExpertsDown = ExpertRowBlockSharded[
        C.NUM_EXPERTS, C.HIDDEN, C.MOE_INTERMEDIATE,
    ]


struct Gemma4TailShapes:
    comptime FinalNorm = Replicated[C.HIDDEN, 1]
    comptime Embed = VocabularyRowSharded[C.VOCAB_SIZE, C.HIDDEN]


struct SlidingAttnRefs[R: Gemma4Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4Shapes[Self.R.FFN_BLOCK]
    var q_proj: Slot[BF16, Self.S.SlidingQ,  "self_attn.q_proj.weight", Self.R.SlidingQkv]
    var k_proj: Slot[BF16, Self.S.SlidingKV, "self_attn.k_proj.weight", Self.R.SlidingQkv]
    var v_proj: Slot[BF16, Self.S.SlidingKV, "self_attn.v_proj.weight", Self.R.SlidingQkv]
    var o_proj: Slot[BF16, Self.S.SlidingO,  "self_attn.o_proj.weight", Self.R.SlidingOut]
    var q_norm: Slot[BF16, Shape[C.HEAD_DIM_SLIDING, 1], "self_attn.q_norm.weight"]
    var k_norm: Slot[BF16, Shape[C.HEAD_DIM_SLIDING, 1], "self_attn.k_norm.weight"]


struct FullAttnRefs[R: Gemma4Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4Shapes[Self.R.FFN_BLOCK]
    var q_proj: Slot[BF16, Self.S.FullQ, "self_attn.q_proj.weight", Self.R.FullQkv]
    var k_proj: Slot[BF16, Self.S.FullK, "self_attn.k_proj.weight", Self.R.FullQkv]
    var o_proj: Slot[BF16, Self.S.FullO, "self_attn.o_proj.weight", Self.R.FullOut]
    var q_norm: Slot[BF16, Shape[C.HEAD_DIM_FULL, 1], "self_attn.q_norm.weight"]
    var k_norm: Slot[BF16, Shape[C.HEAD_DIM_FULL, 1], "self_attn.k_norm.weight"]


struct BodyRefs[R: Gemma4Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4Shapes[Self.R.FFN_BLOCK]
    var input_norm:      Slot[BF16, Shape[C.HIDDEN, 1],         "input_layernorm.weight"]
    var post_attn_norm:  Slot[BF16, Shape[C.HIDDEN, 1],         "post_attention_layernorm.weight"]
    var pre_ffn_norm:    Slot[BF16, Shape[C.HIDDEN, 1],         "pre_feedforward_layernorm.weight"]
    var pre_ffn_norm_2:  Slot[BF16, Shape[C.HIDDEN, 1],         "pre_feedforward_layernorm_2.weight"]
    var post_ffn_norm_1: Slot[BF16, Shape[C.HIDDEN, 1],         "post_feedforward_layernorm_1.weight"]
    var post_ffn_norm_2: Slot[BF16, Shape[C.HIDDEN, 1],         "post_feedforward_layernorm_2.weight"]
    var post_ffn_norm:   Slot[BF16, Shape[C.HIDDEN, 1],         "post_feedforward_layernorm.weight"]
    var gate_proj:       Slot[BF16, Self.S.GateUp,              "mlp.gate_proj.weight", Self.R.DenseGateUp]
    var up_proj:         Slot[BF16, Self.S.GateUp,              "mlp.up_proj.weight", Self.R.DenseGateUp]
    var down_proj:       Slot[BF16, Self.S.Down,                "mlp.down_proj.weight", Self.R.DenseDown]
    var router_proj:     Slot[BF16, Self.S.RouterProj,          "router.proj.weight", Self.R.Router]
    var router_scale:    Slot[BF16, Shape[C.HIDDEN, 1],         "router.scale"]
    var router_pes:      Slot[BF16, Shape[C.NUM_EXPERTS, 1],    "router.per_expert_scale"]
    var experts_gate_up: Slot[BF16, Self.S.ExpertsGateUp,       "experts.gate_up_proj", Self.R.MoeGateUp]
    var experts_down:    Slot[BF16, Self.S.ExpertsDown,         "experts.down_proj", Self.R.MoeDown]
    var layer_scalar:    Slot[BF16, Shape[1, 1],                "layer_scalar"]


struct SlidingLayerRefs[R: Gemma4Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    var attn: SlidingAttnRefs[Self.R]
    var body: BodyRefs[Self.R]


struct FullLayerRefs[R: Gemma4Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    var attn: FullAttnRefs[Self.R]
    var body: BodyRefs[Self.R]


struct RopeSlots[half: Int, max_seq_len: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    var cos: Slot[F32, Replicated[Self.max_seq_len, Self.half]]
    var sin: Slot[F32, Replicated[Self.max_seq_len, Self.half]]


struct SteerVectorSlots[capacity: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    var vectors: Slot[BF16, Replicated[Self.capacity, C.HIDDEN]]


struct MeasureSlots[rows_cap: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    var base_head: Slot[BF16, Replicated[Self.rows_cap, C.HIDDEN]]
    var base_logz: Slot[F32, Replicated[Self.rows_cap, 1]]


struct ActivationSlots(Copyable, ImplicitlyCopyable, SlotGroup):
    var x_main:     Slot[BF16, Shape[C.SLIDING_WINDOW, C.HIDDEN]]
    var x_residual: Slot[BF16, Shape[C.SLIDING_WINDOW, C.HIDDEN]]


struct TailRefs[R: Gemma4Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4TailShapes
    var final_norm: Slot[BF16, Self.S.FinalNorm, "model.language_model.norm.weight"]
    var embed:      Slot[BF16, Self.S.Embed, "model.language_model.embed_tokens.weight", Self.R.Embed]


@fieldwise_init
struct Gemma4Layout[
    R: Gemma4Recipes, SKV: KVSlotGroup, FKV: KVSlotGroup, max_seq_len: Int,
    steer_vectors: Int, measure_rows: Int,
](Copyable, ImplicitlyCopyable):
    var arena: ArenaLayout
    var sliding: Repeated[SlidingLayerRefs[Self.R]]
    var full: Repeated[FullLayerRefs[Self.R]]

    var sliding_kv: Repeated[Self.SKV]
    var full_kv: Repeated[Self.FKV]
    var activations: ActivationSlots
    var sliding_rope: RopeSlots[C.ROPE_HALF_SLIDING, Self.max_seq_len]
    var full_rope: RopeSlots[C.ROPE_HALF_FULL, Self.max_seq_len]
    var steer: SteerVectorSlots[Self.steer_vectors]
    var measure: MeasureSlots[Self.measure_rows]

    var tail: Repeated[TailRefs[Self.R]]


def build_gemma4_plan[
    R: Gemma4Recipes, SKV: KVSlotGroup, FKV: KVSlotGroup,
    max_seq_len: Int, batching_seq_len: Int, max_resident_seqs: Int,
    steer_vectors: Int, measure_rows: Int,
](
    degree: Int, max_workers: Int, scratch_cap: Int,
    mut descs: List[WeightDesc],
) -> Gemma4Layout[
    R, SKV, FKV, max_seq_len, steer_vectors, measure_rows,
]:
    if not degree_contracts_ok(degree):
        abort(t"gemma4: degree {degree} does not divide the model dimensions")
    if not paging_contracts_ok[
        max_seq_len, batching_seq_len, max_resident_seqs,
    ](degree):
        abort(t"gemma4: page geometry contracts violated at degree {degree}")
    if not (
        vnni_pack_contract_ok[SlidingLayerRefs[R]](degree)
        and vnni_pack_contract_ok[FullLayerRefs[R]](degree)
        and vnni_pack_contract_ok[TailRefs[R]](degree)
    ):
        abort(
            t"gemma4: degree {degree} breaks VNNI pack alignment; every packed "
            t"weight's per-rank rows must be a multiple of {VNNI_N_STEP} and "
            t"cols a multiple of {VNNI_K_STEP}")
    if max_workers <= 0:
        abort(t"gemma4: max_workers must be positive, got {max_workers}")
    if max_workers > C.SLIDING_WINDOW:
        abort(
            t"gemma4: full-attention partials require max_workers <= "
            t"SLIDING_WINDOW ({C.SLIDING_WINDOW}), got {max_workers}")

    var sl_proto = SlidingLayerRefs[R]()
    var sl_stride = stamp_offsets(sl_proto, degree)
    var fl_proto = FullLayerRefs[R]()
    var fl_stride = stamp_offsets(fl_proto, degree)

    var sl_off = 0
    var fl_off = sl_off + C.NUM_SLIDING_LAYERS * sl_stride
    var distributed = fl_off + C.NUM_FULL_LAYERS * fl_stride

    for i in range(C.NUM_LAYERS):
        var entry = LAYER_SCHEDULE[i]
        var prefix = String(t"model.language_model.layers.{entry.idx}.")
        if entry.kind == LayerKind.FULL:
            _ = emit_descs[FullLayerRefs[R]](
                prefix, fl_off + entry.local_idx * fl_stride, degree, descs)
        else:
            _ = emit_descs[SlidingLayerRefs[R]](
                prefix, sl_off + entry.local_idx * sl_stride, degree, descs)

    var tail_proto = TailRefs[R]()
    var tail_bytes = stamp_offsets(tail_proto, degree)
    _ = emit_descs[TailRefs[R]]("", distributed, degree, descs)
    var tail = Repeated[TailRefs[R]](tail_proto, distributed, tail_bytes, 1)
    distributed += tail_bytes

    var state_cursor = distributed

    var skv_proto = SKV()
    var skv_stride = stamp_offsets(skv_proto, degree)
    var sliding_kv = Repeated[SKV](
        skv_proto, state_cursor, skv_stride, C.NUM_SLIDING_LAYERS)
    state_cursor = align_up(state_cursor + C.NUM_SLIDING_LAYERS * skv_stride)

    var fkv_proto = FKV()
    var fkv_stride = stamp_offsets(fkv_proto, degree)
    var full_kv = Repeated[FKV](
        fkv_proto, state_cursor, fkv_stride, C.NUM_FULL_LAYERS)
    state_cursor = align_up(state_cursor + C.NUM_FULL_LAYERS * fkv_stride)

    var activations = ActivationSlots()
    state_cursor = stamp_offsets(activations, degree, state_cursor)

    state_cursor = align_up(state_cursor)
    var scratch_off = state_cursor
    state_cursor = align_up(state_cursor + scratch_cap)

    var sliding_rope = RopeSlots[C.ROPE_HALF_SLIDING, max_seq_len]()
    state_cursor = stamp_offsets(sliding_rope, degree, state_cursor)
    var full_rope = RopeSlots[C.ROPE_HALF_FULL, max_seq_len]()
    state_cursor = stamp_offsets(full_rope, degree, state_cursor)

    var steer = SteerVectorSlots[steer_vectors]()
    state_cursor = stamp_offsets(steer, degree, state_cursor)

    var measure = MeasureSlots[measure_rows]()
    state_cursor = stamp_offsets(measure, degree, state_cursor)

    var arena = ArenaLayout(
        distributed_bytes=distributed,
        state_bytes=state_cursor - distributed,
        host_bytes=align_up(state_cursor),
        scratch_off=scratch_off,
    )
    return Gemma4Layout[
        R, SKV, FKV, max_seq_len, steer_vectors, measure_rows,
    ](
        arena=arena,
        sliding=Repeated[SlidingLayerRefs[R]](
            sl_proto, sl_off, sl_stride, C.NUM_SLIDING_LAYERS),
        full=Repeated[FullLayerRefs[R]](
            fl_proto, fl_off, fl_stride, C.NUM_FULL_LAYERS),
        sliding_kv=sliding_kv, full_kv=full_kv,
        activations=activations,
        sliding_rope=sliding_rope, full_rope=full_rope,
        steer=steer,
        measure=measure,
        tail=tail)


def gemma4_kv_mirrors[
    R: Gemma4Recipes, SKV: KVSlotGroup, FKV: KVSlotGroup, max_seq_len: Int,
    steer_vectors: Int, measure_rows: Int, //,
    batching_seq_len: Int, max_resident_seqs: Int,
](
    read layout: Gemma4Layout[
        R, SKV, FKV, max_seq_len, steer_vectors, measure_rows,
    ], degree: Int,
) -> List[KVPoolMirror]:
    var mirrors = List[KVPoolMirror]()
    mirrors.append(KVPoolMirror(
        page_len=PAGE_LEN,
        pos_shard=1,
        region_off=layout.sliding_kv.off,
        stride=layout.sliding_kv.stride,
        layers=layout.sliding_kv.count,
        components=kv_components(layout.sliding_kv.proto, degree),
        spec=PagePoolSpec(
            num_pages=max_resident_seqs * SLIDING_RING_PAGES,
            fixed_pages_per_seq=SLIDING_RING_PAGES,
            max_pages_per_seq=SLIDING_RING_PAGES)))
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
    return mirrors^


def gemma4_bake_router_scales[
    R: Gemma4Recipes, SKV: KVSlotGroup, FKV: KVSlotGroup, max_seq_len: Int,
    steer_vectors: Int, measure_rows: Int, //,
](
    read layout: Gemma4Layout[
        R, SKV, FKV, max_seq_len, steer_vectors, measure_rows,
    ],
    read arena_bases: List[Int],
):
    comptime width = simd_width_of[DType.float32]()
    comptime inv_sqrt_hidden = 1.0 / sqrt[DType.float32, 1](C.HIDDEN)
    for rank in range(len(arena_bases)):
        var arena_base = arena_bases[rank]
        for i in range(C.NUM_LAYERS):
            var entry = LAYER_SCHEDULE[i]
            var p: UnsafePointer[BFloat16, MutUntrackedOrigin]
            if entry.kind == LayerKind.FULL:
                var lb = arena_base + layout.full.base(entry.local_idx)
                p = layout.full.proto.body.router_scale.at(lb)
            else:
                var lb = arena_base + layout.sliding.base(entry.local_idx)
                p = layout.sliding.proto.body.router_scale.at(lb)
            for j in range(0, C.HIDDEN, width):
                var lane = p + j
                var v = lane.load[width=width]().cast[DType.float32]()
                lane.store((v * inv_sqrt_hidden).cast[DType.bfloat16]())
    print("  router constants baked")


def gemma4_init_rope_tables[
    R: Gemma4Recipes, SKV: KVSlotGroup, FKV: KVSlotGroup, max_seq_len: Int,
    steer_vectors: Int, measure_rows: Int, //,
](
    read layout: Gemma4Layout[
        R, SKV, FKV, max_seq_len, steer_vectors, measure_rows,
    ],
    read arena_bases: List[Int],
):
    for rank in range(len(arena_bases)):
        var base = arena_bases[rank]
        var sl_cos = layout.sliding_rope.cos.at(base)
        var sl_sin = layout.sliding_rope.sin.at(base)
        init_rope_table[C.ROPE_HALF_SLIDING, max_seq_len](
            sl_cos, sl_sin, 10000.0)
        var fl_cos = layout.full_rope.cos.at(base)
        var fl_sin = layout.full_rope.sin.at(base)
        init_rope_table_partial_strided[
            C.ROPE_HALF_FULL, max_seq_len,
        ](fl_cos, fl_sin, 1000000.0, C.HEAD_DIM_FULL, 0, 1)
    print("  rope tables initialized")


def gemma4_load_arenas[
    R: Gemma4Recipes, SKV: KVSlotGroup, FKV: KVSlotGroup,
    max_seq_len: Int, batching_seq_len: Int, max_resident_seqs: Int,
    steer_vectors: Int, measure_rows: Int,
](
    dir_path: Path,
    topo: NumaTopology,
    degree: Int,
    max_workers: Int,
    scratch_cap: Int,
    mut arenas: List[NumaArena[alignment=DEFAULT_ALIGNMENT]],
) -> Optional[Gemma4Layout[
    R, SKV, FKV, max_seq_len, steer_vectors, measure_rows,
]]:
    var shards = discover_shards(dir_path)
    if len(shards) == 0:
        print(t"no safetensors shards found in {dir_path}")
        return None
    var n_shards = len(shards)
    print(t"found {n_shards} shard(s)")

    var descs = List[WeightDesc]()
    var layout = build_gemma4_plan[
        R, SKV, FKV,
        max_seq_len, batching_seq_len, max_resident_seqs, steer_vectors,
        measure_rows,
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
