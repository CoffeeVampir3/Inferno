from std.memory import UnsafePointer
from std.reflection import reflect

from kernels.helpers import RankView, Binding
from modeling.model_spec import (
    Encoding, ShapeLike, WeightDesc, Replicated,
    DISTRIBUTED, align_up,
)
from modeling.utilities import FieldwiseDefault
from quant.recipe import (
    QuantRecipe, Passthrough, RouterCenter, SoftmaxRouterCenter,
)
from quant.manifest import (
    quant_manifest, manifest_arena_bytes, member_rel_off, has_role, QuantRole,
)
from butterquant.weight import (
    ButterquantWeight, ButterquantRouter,
    quant_vnni_packed, quant_colsum_per_block, quant_k_block,
    quant_has_colsum,
)
from butterquant.pack import PackColsumTask
from butterquant.vnni import VNNI_N_STEP, VNNI_K_STEP, COLSUM_NARROW_WIDTH


def noop_name_gen(prefix: String, mut names: List[String]):
    pass


@fieldwise_init
struct SourceSpec(Copyable, Movable, ImplicitlyCopyable):
    var name: StaticString
    var group_count: Int
    var gen: def(String, mut List[String]) thin -> None

    @implicit
    def __init__(out self, name: StaticString):
        self = Self(name, 1, noop_name_gen)

    @implicit
    def __init__(out self, lit: StringLiteral):
        self = Self(StaticString(lit), 1, noop_name_gen)

    @staticmethod
    def grouped(
        count: Int, gen: def(String, mut List[String]) thin -> None,
    ) -> Self:
        return Self("", count, gen)


trait SlotLike:
    comptime ENCODING: Encoding
    comptime SHAPE: ShapeLike
    comptime SOURCE: SourceSpec
    comptime NAME: Optional[StaticString]
    comptime TARGET_RANK: Int
    comptime QUANT: QuantRecipe

    @always_inline
    def set_offset(mut self, off: Int): ...

    @always_inline
    def get_offset(self) -> Int: ...


trait SlotGroup(FieldwiseDefault):
    pass


@fieldwise_init
struct BindContext[o: ImmutOrigin](Copyable, ImplicitlyCopyable):
    """Per-call binding context. `layer_off` is the current layer's
    arena-relative offset for weight-slot resolution; promotion to an
    absolute address happens only here, anchored at the rank-0 arena base.
    The `RankView` carries the runtime tensor-parallel degree (`len` of the
    borrowed bases)."""
    var view: RankView[Self.o]
    var layer_off: Int

    @always_inline
    def degree(self) -> Int:
        return self.view.degree()

    @always_inline
    def with_layer(self, layer_off: Int) -> Self:
        var c = self
        c.layer_off = layer_off
        return c

    @always_inline
    def layer_address(self) -> Int:
        return self.view.bases[0] + self.layer_off

    @always_inline
    def bind[T: AnyType](
        self, ptr: UnsafePointer[T, MutUntrackedOrigin],
    ) -> Binding[T, Self.o]:
        return self.view.bind(ptr)


struct Slot[
    encoding: Encoding, shape: ShapeLike, source: SourceSpec = SourceSpec(""),
    quant: QuantRecipe = Passthrough(),
    target_rank: Int = DISTRIBUTED,
](SlotLike, Defaultable, Copyable, ImplicitlyCopyable):
    comptime ENCODING = Self.encoding
    comptime SHAPE = Self.shape
    comptime SOURCE = Self.source
    comptime NAME = Optional[StaticString](None) if Self.source.name == StaticString(
        "") else Optional[StaticString](Self.source.name)
    comptime TARGET_RANK = Self.target_rank
    comptime QUANT = Self.quant

    var offset: Int

    def __init__(out self):
        self.offset = -1

    @implicit
    def __init__(out self, offset: Int):
        self.offset = offset

    @always_inline
    def set_offset(mut self, off: Int):
        self.offset = off

    @always_inline
    def get_offset(self) -> Int:
        return self.offset

    @always_inline
    def at(self, base: Int) -> UnsafePointer[Scalar[Self.ENCODING.DTYPE], MutUntrackedOrigin]:
        return UnsafePointer[Scalar[Self.ENCODING.DTYPE], MutUntrackedOrigin](
            unsafe_from_address=base + self.offset)

    @always_inline
    def binding[o: ImmutOrigin](
        self, base: Int, view: RankView[o],
    ) -> Binding[Scalar[Self.ENCODING.DTYPE], o]:
        return view.bind(self.at(base))

    @always_inline
    def binding[o: ImmutOrigin](
        self, ctx: BindContext[o],
    ) -> Binding[Scalar[Self.ENCODING.DTYPE], o]:
        return self.binding(ctx.layer_address(), ctx.view)

    @always_inline
    def state_binding[o: ImmutOrigin](
        self, ctx: BindContext[o],
    ) -> Binding[Scalar[Self.ENCODING.DTYPE], o]:
        return self.binding(ctx.view.bases[0], ctx.view)

    @always_inline
    def bq_weight[o: ImmutOrigin](
        self, ctx: BindContext[o],
    ) -> ButterquantWeight[Self.QUANT, o]:
        """Bind the int8 weight + scale + colsum sidecars of a quantized slot
        from the same per-slot offsets `emit_quant_descs` wrote them to, resolved
        at the runtime `degree`. The colsum binding points at the weight base when
        the recipe declares none; `colsum_checked` gates access at comptime."""
        var degree = ctx.degree()
        var scale_off = member_rel_off[
            Self.ENCODING, Self.SHAPE, Self.QUANT, QuantRole.SCALE](degree)
        var cs_off = member_rel_off[
            Self.ENCODING, Self.SHAPE, Self.QUANT, QuantRole.COLSUM](degree)
        var base = ctx.layer_address() + self.offset
        var data = UnsafePointer[Int8, MutUntrackedOrigin](unsafe_from_address=base)
        var scale = UnsafePointer[Float32, MutUntrackedOrigin](
            unsafe_from_address=base + scale_off)
        var colsum = UnsafePointer[Float32, MutUntrackedOrigin](
            unsafe_from_address=base + cs_off)
        return ButterquantWeight[Self.QUANT, o](
            ctx.bind(data), ctx.bind(scale), ctx.bind(colsum))

    @always_inline
    def bq_router[o: ImmutOrigin](
        self, ctx: BindContext[o],
    ) -> ButterquantRouter[Self.QUANT, o]:
        """Bind a router-centered slot. The centered bf16 weight is always
        present; the gauge and bias are bound only when the recipe stores them,
        so a SoftmaxRouterCenter slot binds neither."""
        comptime assert (
            Self.QUANT.isa[RouterCenter]() or Self.QUANT.isa[SoftmaxRouterCenter]()
        ), "Slot.bq_router requires a router-centered slot."
        var degree = ctx.degree()
        var base = ctx.layer_address() + self.offset
        var centered = ctx.bind(UnsafePointer[BFloat16, MutUntrackedOrigin](
            unsafe_from_address=base))
        var gauge = Optional[Binding[BFloat16, o]](None)
        var bias = Optional[Binding[Float32, o]](None)
        if has_role[Self.ENCODING, Self.SHAPE, Self.QUANT, QuantRole.GAUGE](degree):
            var gauge_off = member_rel_off[
                Self.ENCODING, Self.SHAPE, Self.QUANT, QuantRole.GAUGE](degree)
            gauge = Optional[Binding[BFloat16, o]](
                ctx.bind(UnsafePointer[BFloat16, MutUntrackedOrigin](
                    unsafe_from_address=base + gauge_off)))
        if has_role[Self.ENCODING, Self.SHAPE, Self.QUANT, QuantRole.BIAS](degree):
            var bias_off = member_rel_off[
                Self.ENCODING, Self.SHAPE, Self.QUANT, QuantRole.BIAS](degree)
            bias = Optional[Binding[Float32, o]](
                ctx.bind(UnsafePointer[Float32, MutUntrackedOrigin](
                    unsafe_from_address=base + bias_off)))
        return ButterquantRouter[Self.QUANT, o](centered, gauge, bias)


@always_inline
def slot_arena_bytes[
    encoding: Encoding, shape: ShapeLike, quant: QuantRecipe,
](degree: Int) -> Int:
    """Per-rank arena bytes for a slot at runtime `degree`: weight + every
    sidecar implied by `quant`, summed from the shared manifest."""
    return manifest_arena_bytes[encoding, shape, quant](degree)


@always_inline
def emit_member_descs[
    encoding: Encoding, shape: ShapeLike, quant: QuantRecipe,
](
    full: String, slot_arena_off: Int, target_rank: Int, degree: Int,
    mut ops: List[WeightDesc],
):
    var manifest = quant_manifest[encoding, shape, quant](degree)
    for i in range(manifest.count):
        var member = manifest.members[i]
        if member.role != QuantRole.COLSUM:
            ops.append(WeightDesc(
                name=full + String(member.suffix),
                arena_offset=slot_arena_off + member.rel_off,
                dtype=member.dtype, element_bytes=member.element_bytes,
                global_rows=member.global_rows, global_cols=member.global_cols,
                local_cols=member.local_cols,
                data_rows=member.data_rows, data_cols=member.data_cols,
                target_rank=target_rank,
            ))


@always_inline
def emit_quant_descs[
    encoding: Encoding, shape: ShapeLike, quant: QuantRecipe,
    name: StaticString, target_rank: Int,
](
    prefix: String, slot_arena_off: Int, degree: Int, mut ops: List[WeightDesc],
):
    emit_member_descs[encoding, shape, quant](
        prefix + String(name), slot_arena_off, target_rank, degree, ops)


@always_inline
def emit_group[
    encoding: Encoding, shape: ShapeLike, quant: QuantRecipe, count: Int,
](
    names: List[String], region_off: Int, degree: Int, mut ops: List[WeightDesc],
):
    comptime per_elem_rows = shape.SIZE_ON_DISK_N // count
    comptime ElementShape = Replicated[per_elem_rows, shape.SIZE_ON_DISK_M]
    var whole = quant_manifest[encoding, shape, quant](degree)
    var elem = quant_manifest[encoding, ElementShape, quant](1)
    var per_rank = count // degree
    for mi in range(elem.count):
        var em = elem.members[mi]
        if em.role == QuantRole.COLSUM:
            continue
        var member_base = whole.members[mi].rel_off
        var stride = em.data_rows * em.data_cols * em.element_bytes
        for e in range(len(names)):
            var rank = e // per_rank
            var local = e % per_rank
            ops.append(WeightDesc(
                name=names[e] + String(em.suffix),
                arena_offset=region_off + member_base + local * stride,
                dtype=em.dtype, element_bytes=em.element_bytes,
                global_rows=em.global_rows, global_cols=em.global_cols,
                local_cols=em.local_cols,
                data_rows=em.data_rows, data_cols=em.data_cols,
                target_rank=rank,
            ))


def stamp_offsets[T: AnyType](mut t: T, degree: Int, off_in: Int = 0) -> Int:
    """Walk T (recursing into SlotGroup fields), stamping each Slot's
    within-region byte offset for the runtime `degree`. Returns total bytes
    consumed. Each slot's byte footprint comes from `slot_arena_bytes`, which is
    recipe- and degree-aware. Does NOT emit any loader records."""
    var off = off_in
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            ref slot = reflect[T].field_ref[i](t)
            slot.set_offset(off)
            off = align_up(off + slot_arena_bytes[
                FT.ENCODING, FT.SHAPE, FT.QUANT,
            ](degree))
        comptime if conforms_to(FT, SlotGroup):
            ref nested = reflect[T].field_ref[i](t)
            off = stamp_offsets(nested, degree, off)
    return off


def emit_descs[T: AnyType](
    prefix: String,
    region_base: Int,
    degree: Int,
    mut ops: List[WeightDesc],
    off_in: Int = 0,
) -> Int:
    """Walk T comptime, emitting one or more WeightDescs per named Slot at
    region_base + within-region offset for the runtime `degree`. Recurses into
    SlotGroup fields. Returns total bytes (must match stamp_offsets for the same
    T and degree)."""
    var off = off_in
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            comptime if FT.SOURCE.group_count == 1:
                comptime if FT.NAME:
                    emit_quant_descs[
                        FT.ENCODING, FT.SHAPE, FT.QUANT,
                        FT.NAME.value(), FT.TARGET_RANK,
                    ](prefix, region_base + off, degree, ops)
            comptime if FT.SOURCE.group_count != 1:
                var names = List[String]()
                FT.SOURCE.gen(prefix, names)
                emit_group[
                    FT.ENCODING, FT.SHAPE, FT.QUANT, FT.SOURCE.group_count,
                ](names, region_base + off, degree, ops)
            off = align_up(off + slot_arena_bytes[
                FT.ENCODING, FT.SHAPE, FT.QUANT,
            ](degree))
        comptime if conforms_to(FT, SlotGroup):
            off = emit_descs[FT](prefix, region_base, degree, ops, off)
    return off


def emit_pack_tasks[T: AnyType](
    region_base: Int,
    degree: Int,
    mut tasks: List[PackColsumTask],
    off_in: Int = 0,
) -> Int:
    """Walk T comptime, emitting one VNNI pack task per VnniPacked weight slot
    (row-major / passthrough / router slots declare no pack and are skipped) at
    region_base + within-region offset for the runtime `degree`. Offset
    accumulation matches emit_descs so weight/colsum offsets land on the same
    arena bytes the loader wrote. Returns total bytes consumed."""
    var off = off_in
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            comptime if quant_vnni_packed[FT.QUANT]():
                comptime assert quant_has_colsum[FT.QUANT](), (
                    "VNNI packed slots require a colsum member for in-place "
                    "pack/colsum initialization")
                comptime per_block = quant_colsum_per_block[FT.QUANT]()
                comptime block_pb = quant_k_block[FT.QUANT]()
                var rows = FT.SHAPE.data_n(degree)
                var cols = FT.SHAPE.data_m(degree)
                var block_cols = block_pb if per_block else cols
                var cs_off = member_rel_off[
                    FT.ENCODING, FT.SHAPE, FT.QUANT, QuantRole.COLSUM](degree)
                tasks.append(PackColsumTask(
                    weight_off=region_base + off,
                    colsum_off=region_base + off + cs_off,
                    rows=rows,
                    cols=cols,
                    block_cols=block_cols,
                    colsum_row_major=not per_block))
            off = align_up(off + slot_arena_bytes[
                FT.ENCODING, FT.SHAPE, FT.QUANT,
            ](degree))
        comptime if conforms_to(FT, SlotGroup):
            off = emit_pack_tasks[FT](region_base, degree, tasks, off)
    return off


@always_inline
def vnni_pack_slot_contract_ok[
    shape: ShapeLike, quant: QuantRecipe,
](degree: Int) -> Bool:
    var rows = shape.data_n(degree)
    var cols = shape.data_m(degree)
    comptime per_block = quant_colsum_per_block[quant]()
    comptime block_pb = quant_k_block[quant]()
    var block_cols = block_pb if per_block else cols
    return (
        rows % VNNI_N_STEP == 0
        and cols % VNNI_K_STEP == 0
        and block_cols > 0
        and cols % block_cols == 0
        and block_cols >= COLSUM_NARROW_WIDTH
        and block_cols % COLSUM_NARROW_WIDTH == 0
    )


def vnni_pack_contract_ok[T: AnyType](degree: Int) -> Bool:
    """Reflectively validate every VNNI-packed slot in T, including nested
    SlotGroups. This keeps model-specific plan builders from maintaining a
    parallel hand-written list of packed weights."""
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            comptime if quant_vnni_packed[FT.QUANT]():
                if not vnni_pack_slot_contract_ok[
                    FT.SHAPE, FT.QUANT,
                ](degree):
                    return False
        comptime if conforms_to(FT, SlotGroup):
            if not vnni_pack_contract_ok[FT](degree):
                return False
    return True
