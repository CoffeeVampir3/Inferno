from std.memory import UnsafePointer, Span
from std.pathlib import Path

from numa import NumaArena, NumaTopology
from kernels.helpers import RankView
from kernels.profiling import Profiler
from kernels.copy_kernels import dispatch_copy_slot
from kernels.abliterate_kernels import (
    dispatch_abliterate_dense, dispatch_abliterate_experts,
)
from threading.threading_traits import BurstThreadPool
from modeling.model_spec import DEFAULT_ALIGNMENT, BF16
from modeling.slot import Slot, SlotLike, SlotGroup, BindContext, stamp_offsets
from modeling.modeling_common import Repeated
from modeling.gemma4_common import Gemma4BaseConfig, LAYER_SCHEDULE, LayerKind
from modeling.gemma4_topology import (
    Gemma4Recipes, KVSlotGroup, Gemma4Shapes, Gemma4Layout,
)
from modeling.loader import discover_shards
from inspectable_toolkit.checkpoint_writer import copy_checkpoint, patch_slot
from safetensors.parser import SafetensorsHeader, parse_safetensors_header


comptime C = Gemma4BaseConfig


struct AbliterateWorkspace(Movable):
    """Per-rank f32 scratch for the norm-preserving edit, NUMA-bound to mirror
    the model's per-rank arenas. Each rank's block holds `v`, `m`, `a`
    (each `rows`) followed by `p` (`pmax`) at identical intra-block offsets, so
    a `RankView` over `bases` resolves rank r by constant per-rank delta and the
    existing allreduce reduces `m`/`a` across ranks. The block is independent of
    the model arena: directions are an external artifact and this scratch is
    transient edit workspace, reused across optimizer trials."""
    var arenas: List[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var bases: List[Int]
    var rows: Int
    var pmax: Int

    def __init__(out self, topo: NumaTopology, degree: Int, rows: Int, pmax: Int):
        self.rows = rows
        self.pmax = pmax
        self.arenas = List[NumaArena[alignment=DEFAULT_ALIGNMENT]]()
        self.bases = List[Int]()
        var count = 3 * rows + pmax
        var size = count * 4
        for r in range(degree):
            var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](topo.node(r), size)
            var blk = arena.alloc[Float32](count)
            if not blk:
                self.bases.append(0)
            else:
                self.bases.append(Int(blk.value()))
            _ = arena.prefault()
            self.arenas.append(arena^)

    @always_inline
    def ok(self) -> Bool:
        for r in range(len(self.bases)):
            if self.bases[r] == 0:
                return False
        return len(self.bases) > 0

    @always_inline
    def v_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.bases[0])

    @always_inline
    def m_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.bases[0] + self.rows * 4)

    @always_inline
    def a_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.bases[0] + 2 * self.rows * 4)

    @always_inline
    def p_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.bases[0] + 3 * self.rows * 4)


struct ShadowSlidingRefs[R: Gemma4Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4Shapes[Self.R.FFN_BLOCK]
    var o_proj:       Slot[BF16, Self.S.SlidingO,    "self_attn.o_proj.weight", Self.R.SlidingOut]
    var down_proj:    Slot[BF16, Self.S.Down,        "mlp.down_proj.weight", Self.R.DenseDown]
    var experts_down: Slot[BF16, Self.S.ExpertsDown, "experts.down_proj", Self.R.MoeDown]


struct ShadowFullRefs[R: Gemma4Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4Shapes[Self.R.FFN_BLOCK]
    var o_proj:       Slot[BF16, Self.S.FullO,       "self_attn.o_proj.weight", Self.R.FullOut]
    var down_proj:    Slot[BF16, Self.S.Down,        "mlp.down_proj.weight", Self.R.DenseDown]
    var experts_down: Slot[BF16, Self.S.ExpertsDown, "experts.down_proj", Self.R.MoeDown]


struct ShadowWeights[R: Gemma4Recipes](Movable):
    """Caller-owned, NUMA-local copy of the editable weights (`o_proj`,
    `down_proj`, `experts_down` per layer) used by the norm-preserving edit as
    its unmodified read-source and rollback source. Mirrors the live per-rank
    sharding so rank r's shadow shard sits on the same node as its live shard,
    keeping the frequent edit-time read local. Independent of the model arena:
    populated once after load by copying the live weights, then reused across
    optimizer trials."""
    var arenas: List[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var bases: List[Int]
    var sliding: Repeated[ShadowSlidingRefs[Self.R]]
    var full: Repeated[ShadowFullRefs[Self.R]]
    var degree: Int

    def __init__(out self, topo: NumaTopology, degree: Int):
        self.degree = degree
        var ps_proto = ShadowSlidingRefs[Self.R]()
        var ps_stride = stamp_offsets(ps_proto, degree)
        var pf_proto = ShadowFullRefs[Self.R]()
        var pf_stride = stamp_offsets(pf_proto, degree)
        var ps_off = 0
        var pf_off = ps_off + C.NUM_SLIDING_LAYERS * ps_stride
        var total = pf_off + C.NUM_FULL_LAYERS * pf_stride
        self.sliding = Repeated[ShadowSlidingRefs[Self.R]](
            ps_proto, ps_off, ps_stride, C.NUM_SLIDING_LAYERS)
        self.full = Repeated[ShadowFullRefs[Self.R]](
            pf_proto, pf_off, pf_stride, C.NUM_FULL_LAYERS)
        self.arenas = List[NumaArena[alignment=DEFAULT_ALIGNMENT]]()
        self.bases = List[Int]()
        for r in range(degree):
            var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](topo.node(r), total)
            var blk = arena.alloc[UInt8](total)
            if not blk:
                self.bases.append(0)
            else:
                self.bases.append(Int(blk.value()))
            _ = arena.prefault()
            self.arenas.append(arena^)

    @always_inline
    def ok(self) -> Bool:
        for r in range(len(self.bases)):
            if self.bases[r] == 0:
                return False
        return len(self.bases) > 0


@always_inline
def rank_view(read bases: List[Int]) -> RankView[ImmutAnyOrigin]:
    return RankView[ImmutAnyOrigin](
        Span[Int, ImmutAnyOrigin](
            ptr=UnsafePointer[Int, ImmutAnyOrigin](
                unsafe_from_address=Int(bases.unsafe_ptr())),
            length=len(bases)))


def build_shadow[
    R: Gemma4Recipes, SKV: KVSlotGroup, FKV: KVSlotGroup,
    msl: Int, sv: Int, mr: Int, //,
](
    live_layout: Gemma4Layout[R, SKV, FKV, msl, sv, mr],
    topo: NumaTopology,
    degree: Int,
) -> ShadowWeights[R]:
    return ShadowWeights[R](topo, degree)


@always_inline
def copy_pair[
    S: SlotLike, P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    to_shadow: Bool,
](
    read live_slot: S, read shadow_slot: S,
    lctx: BindContext[o], sctx: BindContext[o],
    mut pools: List[P], mut prof: Profiler[Profile, N],
):
    comptime if to_shadow:
        dispatch_copy_slot(live_slot, shadow_slot, lctx, sctx, pools, prof)
    else:
        dispatch_copy_slot(shadow_slot, live_slot, sctx, lctx, pools, prof)


def sync_shadow_layers[
    R: Gemma4Recipes, SKV: KVSlotGroup, FKV: KVSlotGroup,
    msl: Int, sv: Int, mr: Int,
    P: BurstThreadPool, Profile: Bool, N: Int, //,
    to_shadow: Bool,
](
    read shadow: ShadowWeights[R],
    live_layout: Gemma4Layout[R, SKV, FKV, msl, sv, mr],
    read arena_bases: List[Int],
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var lctx0 = BindContext(rank_view(arena_bases), 0)
    var sctx0 = BindContext(rank_view(shadow.bases), 0)
    for i in range(C.NUM_LAYERS):
        var entry = LAYER_SCHEDULE[i]
        if entry.kind == LayerKind.FULL:
            var lctx = lctx0.with_layer(live_layout.full.base(entry.local_idx))
            var sctx = sctx0.with_layer(shadow.full.base(entry.local_idx))
            var attn = live_layout.full.proto.attn
            var body = live_layout.full.proto.body
            var sh = shadow.full.proto
            copy_pair[to_shadow=to_shadow](
                attn.o_proj, sh.o_proj, lctx, sctx, pools, prof)
            copy_pair[to_shadow=to_shadow](
                body.down_proj, sh.down_proj, lctx, sctx, pools, prof)
            copy_pair[to_shadow=to_shadow](
                body.experts_down, sh.experts_down, lctx, sctx, pools, prof)
        else:
            var lctx = lctx0.with_layer(
                live_layout.sliding.base(entry.local_idx))
            var sctx = sctx0.with_layer(shadow.sliding.base(entry.local_idx))
            var attn = live_layout.sliding.proto.attn
            var body = live_layout.sliding.proto.body
            var sh = shadow.sliding.proto
            copy_pair[to_shadow=to_shadow](
                attn.o_proj, sh.o_proj, lctx, sctx, pools, prof)
            copy_pair[to_shadow=to_shadow](
                body.down_proj, sh.down_proj, lctx, sctx, pools, prof)
            copy_pair[to_shadow=to_shadow](
                body.experts_down, sh.experts_down, lctx, sctx, pools, prof)


def populate_shadow[
    R: Gemma4Recipes, SKV: KVSlotGroup, FKV: KVSlotGroup,
    msl: Int, sv: Int, mr: Int,
    P: BurstThreadPool, Profile: Bool, N: Int, //,
](
    read shadow: ShadowWeights[R],
    live_layout: Gemma4Layout[R, SKV, FKV, msl, sv, mr],
    read arena_bases: List[Int],
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    sync_shadow_layers[to_shadow=True](
        shadow, live_layout, arena_bases, pools, prof)


def restore_from_shadow[
    R: Gemma4Recipes, SKV: KVSlotGroup, FKV: KVSlotGroup,
    msl: Int, sv: Int, mr: Int,
    P: BurstThreadPool, Profile: Bool, N: Int, //,
](
    read shadow: ShadowWeights[R],
    live_layout: Gemma4Layout[R, SKV, FKV, msl, sv, mr],
    read arena_bases: List[Int],
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    sync_shadow_layers[to_shadow=False](
        shadow, live_layout, arena_bases, pools, prof)


def abliterate_schedule[
    R: Gemma4Recipes, SKV: KVSlotGroup, FKV: KVSlotGroup,
    msl: Int, sv: Int, mr: Int,
    P: BurstThreadPool, Profile: Bool, N: Int, //,
](
    read shadow: ShadowWeights[R],
    live_layout: Gemma4Layout[R, SKV, FKV, msl, sv, mr],
    read arena_bases: List[Int],
    degree: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
    read directions: List[BFloat16],
    read attn_alpha: List[Float32],
    read down_alpha: List[Float32],
    mut ws: AbliterateWorkspace,
):
    var lctx0 = BindContext(rank_view(arena_bases), 0)
    var sctx0 = BindContext(rank_view(shadow.bases), 0)
    var ws_view = rank_view(ws.bases)
    var v = ws_view.bind(ws.v_ptr())
    var m = ws_view.bind(ws.m_ptr())
    var a = ws_view.bind(ws.a_ptr())
    var p = ws_view.bind(ws.p_ptr())
    for i in range(C.NUM_LAYERS):
        var a_alpha = attn_alpha[i]
        var d_alpha = down_alpha[i]
        if a_alpha == Float32(0) and d_alpha == Float32(0):
            continue
        var db = (i + 1) * C.HIDDEN
        for r in range(degree):
            var vp = v[r]
            for j in range(C.HIDDEN):
                vp[j] = directions[db + j].cast[DType.float32]()
        var entry = LAYER_SCHEDULE[i]
        if entry.kind == LayerKind.FULL:
            var lctx = lctx0.with_layer(live_layout.full.base(entry.local_idx))
            var sctx = sctx0.with_layer(shadow.full.base(entry.local_idx))
            var attn = live_layout.full.proto.attn
            var body = live_layout.full.proto.body
            var sh = shadow.full.proto
            if a_alpha != Float32(0):
                dispatch_abliterate_dense[reduce=True](
                    sh.o_proj, attn.o_proj, sctx, lctx, v, m, a, p,
                    a_alpha, pools, prof)
            if d_alpha != Float32(0):
                dispatch_abliterate_dense[reduce=True](
                    sh.down_proj, body.down_proj, sctx, lctx, v, m, a, p,
                    d_alpha, pools, prof)
                dispatch_abliterate_experts(
                    sh.experts_down, body.experts_down, sctx, lctx,
                    v, m, a, p, C.HIDDEN, d_alpha, pools, prof)
        else:
            var lctx = lctx0.with_layer(
                live_layout.sliding.base(entry.local_idx))
            var sctx = sctx0.with_layer(shadow.sliding.base(entry.local_idx))
            var attn = live_layout.sliding.proto.attn
            var body = live_layout.sliding.proto.body
            var sh = shadow.sliding.proto
            if a_alpha != Float32(0):
                dispatch_abliterate_dense[reduce=True](
                    sh.o_proj, attn.o_proj, sctx, lctx, v, m, a, p,
                    a_alpha, pools, prof)
            if d_alpha != Float32(0):
                dispatch_abliterate_dense[reduce=True](
                    sh.down_proj, body.down_proj, sctx, lctx, v, m, a, p,
                    d_alpha, pools, prof)
                dispatch_abliterate_experts(
                    sh.experts_down, body.experts_down, sctx, lctx,
                    v, m, a, p, C.HIDDEN, d_alpha, pools, prof)


def save_abliterated[
    R: Gemma4Recipes, SKV: KVSlotGroup, FKV: KVSlotGroup,
    msl: Int, sv: Int, mr: Int, //,
](
    live_layout: Gemma4Layout[R, SKV, FKV, msl, sv, mr],
    degree: Int,
    read arena_bases: List[Int],
    source_dir: Path,
    dest_dir: Path,
) -> Bool:
    """Write the current (edited) weights to a new checkpoint: copy the
    source checkpoint verbatim, then overwrite only the abliteration-
    touched tensors (o_proj, down_proj, experts_down per layer) in place
    with their global tensors gathered from the live arenas. Shapes and
    dtypes are preserved, so the source header stays valid."""
    if String(source_dir) == String(dest_dir):
        print("save_abliterated: dest_dir must differ from source_dir")
        return False
    if not copy_checkpoint(source_dir, dest_dir):
        return False
    var shard_paths = discover_shards(dest_dir)
    if len(shard_paths) == 0:
        print(t"save_abliterated: no shards in {dest_dir}")
        return False
    var headers = List[SafetensorsHeader]()
    for i in range(len(shard_paths)):
        var h = parse_safetensors_header(shard_paths[i])
        if not h:
            var p = shard_paths[i]
            print(t"save_abliterated: failed to parse {p}")
            return False
        headers.append(h.take())

    for i in range(C.NUM_LAYERS):
        var entry = LAYER_SCHEDULE[i]
        var prefix = String(t"model.language_model.layers.{entry.idx}.")
        if entry.kind == LayerKind.FULL:
            var lb = live_layout.full.base(entry.local_idx)
            var attn = live_layout.full.proto.attn
            var body = live_layout.full.proto.body
            if not patch_slot(attn.o_proj, lb, prefix, degree,
                arena_bases, headers, shard_paths):
                return False
            if not patch_slot(body.down_proj, lb, prefix, degree,
                arena_bases, headers, shard_paths):
                return False
            if not patch_slot(body.experts_down, lb, prefix, degree,
                arena_bases, headers, shard_paths):
                return False
        else:
            var lb = live_layout.sliding.base(entry.local_idx)
            var attn = live_layout.sliding.proto.attn
            var body = live_layout.sliding.proto.body
            if not patch_slot(attn.o_proj, lb, prefix, degree,
                arena_bases, headers, shard_paths):
                return False
            if not patch_slot(body.down_proj, lb, prefix, degree,
                arena_bases, headers, shard_paths):
                return False
            if not patch_slot(body.experts_down, lb, prefix, degree,
                arena_bases, headers, shard_paths):
                return False
    print(t"save_abliterated: wrote {dest_dir}")
    return True
