from std.collections import InlineArray

from modeling.model_spec import Encoding, ShapeLike, align_up
from quant.recipe import (
    QuantRecipe, Passthrough, PerRowQuant, PerBlockQuant, RouterCenter,
    SoftmaxRouterCenter, NoColsum, PerRowCs, PerBlockCs, RowMajor,
)


comptime SCALE_SUFFIX: StaticString = ".scale"
comptime COLSUM_SUFFIX: StaticString = ".colsum"
comptime GAUGE_SUFFIX: StaticString = ".gauge"
comptime BIAS_SUFFIX: StaticString = ".bias"


struct QuantRole:
    comptime WEIGHT = 0
    comptime SCALE = 1
    comptime COLSUM = 2
    comptime GAUGE = 3
    comptime BIAS = 4


@fieldwise_init
struct QuantMember(Copyable, Movable, ImplicitlyCopyable):
    """One physical tensor in a slot's encoding. `rel_off` is the within-slot
    byte offset (tight packing, no inter-member alignment). `reserved_bytes` is
    the arena footprint for this member; `out_ndim` selects the safetensors
    header rank (1 emits a single shape dim, 2 emits rows+cols). The loader desc
    shape fields (`global_*`/`data_*`/`local_cols`) drive the row-shard /
    col-shard / replicated read path identically to a hand-written WeightDesc."""
    var role: Int
    var suffix: StaticString
    var dtype: DType
    var element_bytes: Int
    var global_rows: Int
    var global_cols: Int
    var local_cols: Int
    var data_rows: Int
    var data_cols: Int
    var out_ndim: Int
    var reserved_bytes: Int
    var rel_off: Int


comptime MAX_QUANT_MEMBERS = 4


struct QuantManifest(Copyable, Movable, ImplicitlyCopyable):
    var members: InlineArray[QuantMember, MAX_QUANT_MEMBERS]
    var count: Int

    @always_inline
    def __init__(out self):
        self.members = InlineArray[QuantMember, MAX_QUANT_MEMBERS](
            fill=QuantMember(
                role=-1, suffix=StaticString(""), dtype=DType.int8,
                element_bytes=0, global_rows=0, global_cols=0, local_cols=0,
                data_rows=0, data_cols=0, out_ndim=0, reserved_bytes=0,
                rel_off=0))
        self.count = 0

    @always_inline
    def push(mut self, var m: QuantMember):
        self.members[self.count] = m^
        self.count += 1


@always_inline
def quant_manifest[
    encoding: Encoding, shape: ShapeLike, quant: QuantRecipe,
](degree: Int) -> QuantManifest:
    """The single source of truth for a slot's physical tensors. Both the loader
    descriptors (`emit_quant_descs`), the arena byte footprint
    (`manifest_arena_bytes`), the quantizer output entries, and the runtime
    binding bridge derive from this. The recipe dispatch is comptime; the
    per-rank extents are runtime functions of the runtime tensor-parallel
    `degree`."""
    var data_n = shape.data_n(degree)
    var data_m = shape.data_m(degree)
    var out = QuantManifest()

    comptime if quant.isa[Passthrough]():
        out.push(QuantMember(
            role=QuantRole.WEIGHT, suffix=StaticString(""),
            dtype=encoding.DTYPE, element_bytes=encoding.ELEMENT_BYTES,
            global_rows=shape.SIZE_ON_DISK_N, global_cols=shape.SIZE_ON_DISK_M,
            local_cols=data_m, data_rows=data_n, data_cols=data_m,
            out_ndim=2, reserved_bytes=shape.bytes(degree, encoding.ELEMENT_BYTES),
            rel_off=0))
        return out^

    comptime if quant.isa[PerRowQuant]():
        comptime QT = quant[PerRowQuant]
        comptime assert (not QT.pack.isa[RowMajor]()) or QT.colsum.isa[NoColsum](), (
            "RowMajor weight with a colsum is not supported")
        var off = 0
        out.push(QuantMember(
            role=QuantRole.WEIGHT, suffix=StaticString(""),
            dtype=DType.int8, element_bytes=1,
            global_rows=shape.SIZE_ON_DISK_N, global_cols=shape.SIZE_ON_DISK_M,
            local_cols=data_m, data_rows=data_n, data_cols=data_m,
            out_ndim=2, reserved_bytes=data_n * data_m, rel_off=off))
        off += data_n * data_m
        out.push(QuantMember(
            role=QuantRole.SCALE, suffix=SCALE_SUFFIX,
            dtype=DType.float32, element_bytes=4,
            global_rows=shape.SIZE_ON_DISK_N, global_cols=1, local_cols=1,
            data_rows=data_n, data_cols=1,
            out_ndim=1, reserved_bytes=data_n * 4, rel_off=off))
        off += data_n * 4
        comptime if QT.colsum.isa[PerRowCs]():
            out.push(QuantMember(
                role=QuantRole.COLSUM, suffix=COLSUM_SUFFIX,
                dtype=DType.float32, element_bytes=4,
                global_rows=shape.SIZE_ON_DISK_N, global_cols=1, local_cols=1,
                data_rows=data_n, data_cols=1,
                out_ndim=1, reserved_bytes=data_n * 4, rel_off=off))
        comptime if QT.colsum.isa[PerBlockCs]():
            comptime nb_global = shape.SIZE_ON_DISK_M // QT.fwht_block
            var nb_local = data_m // QT.fwht_block
            out.push(QuantMember(
                role=QuantRole.COLSUM, suffix=COLSUM_SUFFIX,
                dtype=DType.float32, element_bytes=4,
                global_rows=shape.SIZE_ON_DISK_N, global_cols=nb_global,
                local_cols=nb_local, data_rows=data_n, data_cols=nb_local,
                out_ndim=2, reserved_bytes=data_n * nb_local * 4,
                rel_off=off))
        return out^

    comptime if quant.isa[PerBlockQuant]():
        comptime QT = quant[PerBlockQuant]
        comptime assert (not QT.pack.isa[RowMajor]()) or QT.colsum.isa[NoColsum](), (
            "RowMajor weight with a colsum is not supported")
        comptime nb_global = shape.SIZE_ON_DISK_M // QT.fwht_block
        var nb_local = data_m // QT.fwht_block
        var off = 0
        out.push(QuantMember(
            role=QuantRole.WEIGHT, suffix=StaticString(""),
            dtype=DType.int8, element_bytes=1,
            global_rows=shape.SIZE_ON_DISK_N, global_cols=shape.SIZE_ON_DISK_M,
            local_cols=data_m, data_rows=data_n, data_cols=data_m,
            out_ndim=2, reserved_bytes=data_n * data_m, rel_off=off))
        off += data_n * data_m
        out.push(QuantMember(
            role=QuantRole.SCALE, suffix=SCALE_SUFFIX,
            dtype=DType.float32, element_bytes=4,
            global_rows=shape.SIZE_ON_DISK_N, global_cols=nb_global,
            local_cols=nb_local, data_rows=data_n, data_cols=nb_local,
            out_ndim=2, reserved_bytes=data_n * nb_local * 4, rel_off=off))
        off += data_n * nb_local * 4
        comptime if QT.colsum.isa[PerBlockCs]():
            out.push(QuantMember(
                role=QuantRole.COLSUM, suffix=COLSUM_SUFFIX,
                dtype=DType.float32, element_bytes=4,
                global_rows=shape.SIZE_ON_DISK_N, global_cols=nb_global,
                local_cols=nb_local, data_rows=data_n, data_cols=nb_local,
                out_ndim=2, reserved_bytes=data_n * nb_local * 4,
                rel_off=off))
        return out^

    comptime if quant.isa[RouterCenter]():
        comptime QT = quant[RouterCenter]
        var off = 0
        out.push(QuantMember(
            role=QuantRole.WEIGHT, suffix=StaticString(""),
            dtype=DType.bfloat16, element_bytes=2,
            global_rows=shape.SIZE_ON_DISK_N, global_cols=shape.SIZE_ON_DISK_M,
            local_cols=data_m, data_rows=data_n, data_cols=data_m,
            out_ndim=2, reserved_bytes=data_n * data_m * 2,
            rel_off=off))
        off += data_n * data_m * 2
        out.push(QuantMember(
            role=QuantRole.GAUGE, suffix=GAUGE_SUFFIX,
            dtype=DType.bfloat16, element_bytes=2,
            global_rows=shape.SIZE_ON_DISK_M, global_cols=1, local_cols=1,
            data_rows=data_m, data_cols=1,
            out_ndim=1, reserved_bytes=data_m * 2, rel_off=off))
        off += data_m * 2
        comptime if QT.bias_name != StaticString(""):
            out.push(QuantMember(
                role=QuantRole.BIAS, suffix=BIAS_SUFFIX,
                dtype=DType.float32, element_bytes=4,
                global_rows=shape.SIZE_ON_DISK_N, global_cols=1, local_cols=1,
                data_rows=data_n, data_cols=1,
                out_ndim=1, reserved_bytes=data_n * 4, rel_off=off))
        return out^

    comptime if quant.isa[SoftmaxRouterCenter]():
        out.push(QuantMember(
            role=QuantRole.WEIGHT, suffix=StaticString(""),
            dtype=DType.bfloat16, element_bytes=2,
            global_rows=shape.SIZE_ON_DISK_N, global_cols=shape.SIZE_ON_DISK_M,
            local_cols=data_m, data_rows=data_n, data_cols=data_m,
            out_ndim=2, reserved_bytes=data_n * data_m * 2,
            rel_off=0))
        return out^

    return out^


@always_inline
def manifest_arena_bytes[
    encoding: Encoding, shape: ShapeLike, quant: QuantRecipe,
](degree: Int) -> Int:
    var m = quant_manifest[encoding, shape, quant](degree)
    var total = 0
    for i in range(m.count):
        total += m.members[i].reserved_bytes
    return total


@always_inline
def member_rel_off[
    encoding: Encoding, shape: ShapeLike, quant: QuantRecipe, role: Int,
](degree: Int) -> Int:
    var m = quant_manifest[encoding, shape, quant](degree)
    for i in range(m.count):
        if m.members[i].role == role:
            return m.members[i].rel_off
    return 0


@always_inline
def has_role[
    encoding: Encoding, shape: ShapeLike, quant: QuantRecipe, role: Int,
](degree: Int) -> Bool:
    var m = quant_manifest[encoding, shape, quant](degree)
    for i in range(m.count):
        if m.members[i].role == role:
            return True
    return False
