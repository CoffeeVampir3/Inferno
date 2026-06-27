from std.collections import InlineArray
from std.math import min
from std.memory import Span, UnsafePointer
from std.os import makedirs
from std.os.path import dirname
from std.pathlib import Path
from std.reflection import reflect
from std.sys.info import simd_width_of, size_of

from linux.io_uring import IoRing, ReadOp, WriteOp, ReadMode, WriteMode
import linux.sys as linux
from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstKernel, BurstThreadPool

from safetensors.parser import (
    SafetensorsHeader, parse_safetensors_header,
    dtype_tag, dtype_byte_size,
)
from safetensors.writer import OutputEntry, build_header
from modeling.loader import discover_shards
from modeling.slot import SlotLike, SlotGroup
from modeling.model_spec import Replicated, Encoding, ShapeLike

from butterquant.kernels import (
    apply_gamma_in_place, gamma_sqrt_abs_in_place, add_offset_in_place,
    rotate_and_quant, router_center, router_center_softmax,
)

from quant.recipe import (
    QuantRecipe, Passthrough, NormGain, PerRowQuant, PerBlockQuant, RouterCenter,
    SoftmaxRouterCenter, SplitGamma, AbsorbedGamma, TwoSided,
    GammaMode, RotationMode,
)
from quant.plan import (
    SlotIdentity, GammaRef, PassthroughPlan, QuantPlan, RouterPlan, SlotPlan,
    ScratchCapacity,
)
from quant.manifest import quant_manifest, QuantRole


comptime PtrU8 = UnsafePointer[UInt8, MutUntrackedOrigin]
comptime PtrF32 = UnsafePointer[Float32, MutUntrackedOrigin]
comptime PtrBF16 = UnsafePointer[Scalar[DType.bfloat16], MutUntrackedOrigin]
comptime PtrI8 = UnsafePointer[Scalar[DType.int8], MutUntrackedOrigin]
comptime W = simd_width_of[DType.float32]()

comptime PANEL_ROWS = 2048
comptime COPY_CHUNK = 16 * 1024 * 1024
comptime QD = 64
comptime QUANT_SCRATCH_ALIGNMENT = 64


@fieldwise_init
struct LocatedTensor(Copyable):
    var shard: Int
    var data_start: Int
    var byte_size: Int
    var dtype: DType
    var rows: Int
    var cols: Int


def fold_shape(ref shape: List[Int]) -> Tuple[Int, Int]:
    if len(shape) == 0:
        return (1, 1)
    if len(shape) == 1:
        return (shape[0], 1)
    var rows = 1
    for i in range(len(shape) - 1):
        rows *= shape[i]
    return (rows, shape[len(shape) - 1])


def find_tensor(
    name: String, headers: Span[SafetensorsHeader, MutUntrackedOrigin],
) -> Optional[LocatedTensor]:
    for i in range(len(headers)):
        ref h = headers[i]
        var meta_opt = h.tensors.get(name)
        if meta_opt:
            ref m = meta_opt.value()
            var rc = fold_shape(m.shape)
            return LocatedTensor(
                shard=i,
                data_start=h.data_offset + m.start,
                byte_size=m.byte_size(),
                dtype=m.dtype,
                rows=rc[0],
                cols=rc[1],
            )
    return None


@always_inline
def supports_decode_to_f32(dt: DType) -> Bool:
    return (
        dt == DType.bfloat16
        or dt == DType.float32
        or dt == DType.float16
    )


@always_inline
def supports_router_source(dt: DType) -> Bool:
    return dt == DType.bfloat16 or dt == DType.float32


@always_inline
def gamma_ref_from[gam: GammaMode](prefix: String) -> GammaRef:
    comptime if gam.isa[SplitGamma]():
        return GammaRef.named(
            prefix + String(gam[SplitGamma].name), False, gam[SplitGamma].offset)
    comptime if gam.isa[AbsorbedGamma]():
        return GammaRef.named(
            prefix + String(gam[AbsorbedGamma].name), True, gam[AbsorbedGamma].offset)
    return GammaRef.none()


@always_inline
def two_sided_m_of[rot: RotationMode]() -> Int:
    comptime if rot.isa[TwoSided]():
        return rot[TwoSided].m_block
    return 0


@always_inline
def as_mut_untracked_span[T: Movable](
    ref items: List[T],
) -> Span[T, MutUntrackedOrigin]:
    """Strip origin tracking from `items` for storage in worker structs.
    Caller guarantees `items` outlives the returned span."""
    return Span[T, MutUntrackedOrigin](
        ptr=items.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        length=len(items),
    )


@always_inline
def decode_from[dt: DType](src: PtrU8, dst: PtrF32, count: Int):
    var p = src.bitcast[Scalar[dt]]()
    var k = 0
    while k + W <= count:
        (dst + k).store((p + k).load[width=W]().cast[DType.float32]())
        k += W
    while k < count:
        dst[k] = p[k].cast[DType.float32]()
        k += 1


def decode_to_f32(dt: DType, src: PtrU8, dst: PtrF32, count: Int) -> Bool:
    if dt == DType.bfloat16:
        decode_from[DType.bfloat16](src, dst, count)
        return True
    if dt == DType.float32:
        decode_from[DType.float32](src, dst, count)
        return True
    if dt == DType.float16:
        decode_from[DType.float16](src, dst, count)
        return True
    return False


def read_sync(
    mut ring: IoRing[QD], fd_idx: Int, offset: Int, length: Int, dest: PtrU8,
) -> Bool:
    try:
        _ = ring.submit_one(ReadOp(
            file_idx=fd_idx, offset=offset, length=length, dest=dest, id=0))
        var c = ring.drain_one()
        if Int(c.result) != length:
            var got = Int(c.result)
            print(t"quant: short read at {offset}: got {got}/{length}")
            return False
        return True
    except err:
        print(t"quant: read failed: {err.error_message()}")
        return False


def write_sync(
    mut ring: IoRing[QD], fd_idx: Int, offset: Int, length: Int, src: PtrU8,
) -> Bool:
    try:
        _ = ring.submit_one(WriteOp(
            file_idx=fd_idx, offset=offset, length=length, src=src, id=0))
        var c = ring.drain_one()
        if Int(c.result) != length:
            var got = Int(c.result)
            print(t"quant: short write at {offset}: got {got}/{length}")
            return False
        return True
    except err:
        print(t"quant: write failed: {err.error_message()}")
        return False


def write_sync_many(mut ring: IoRing[QD], ops: Span[WriteOp[], MutUntrackedOrigin]) -> Bool:
    """Submit `ops` in one or more batches and drain all completions.
    Assigns each op's `id` to its index in the span so completion validation
    is O(1). Caller guarantees the span lives until this returns."""
    var n = len(ops)
    if n == 0:
        return True
    for i in range(n):
        ops[i].id = i
    try:
        var submitted = 0
        while submitted < n:
            var got = ring.submit_many[WriteOp[]](ops, submitted)
            if got == 0:
                var c = ring.drain_one()
                if Int(c.result) != ops[c.id].length:
                    var got_b = Int(c.result)
                    var expected = ops[c.id].length
                    print(t"quant: short write op {c.id}: {got_b}/{expected}")
                    return False
            submitted += got
        var to_drain = n
        while to_drain > 0:
            var c = ring.drain_one()
            if Int(c.result) != ops[c.id].length:
                var got_b = Int(c.result)
                var expected = ops[c.id].length
                print(t"quant: short write op {c.id}: {got_b}/{expected}")
                return False
            to_drain -= 1
    except err:
        print(t"quant: batched write failed: {err.error_message()}")
        return False
    return True


@always_inline
def align_quant(value: Int) -> Int:
    return ((value + QUANT_SCRATCH_ALIGNMENT - 1)
        // QUANT_SCRATCH_ALIGNMENT) * QUANT_SCRATCH_ALIGNMENT


@always_inline
def quant_byte_count[T: AnyType](count: Int) -> Int:
    return count * size_of[T]()


@always_inline
def place_quant_region(mut cursor: Int, bytes: Int) -> Int:
    cursor = align_quant(cursor)
    var off = cursor
    cursor += bytes
    return off


@fieldwise_init
struct QuantScratchLayout(TrivialRegisterPassable):
    """Runtime scratch schema. Offsets are byte offsets into one arena-owned
    block, matching the model's layout-first scratch style while allowing the
    quantizer's sizes to come from the parsed plan."""
    var raw_off: Int
    var f32_work_off: Int
    var i8_quant_off: Int
    var f32_scales_off: Int
    var f32_gamma_off: Int
    var bf16_centered_off: Int
    var bf16_gauge_off: Int
    var total_bytes: Int
    var cap: ScratchCapacity

    @staticmethod
    def build(cap: ScratchCapacity) -> Self:
        var cursor = 0
        var raw_off = place_quant_region(cursor, cap.raw_bytes)
        var f32_work_off = place_quant_region(
            cursor, quant_byte_count[Float32](cap.f32_work))
        var i8_quant_off = place_quant_region(
            cursor, quant_byte_count[Int8](cap.i8_quant))
        var f32_scales_off = place_quant_region(
            cursor, quant_byte_count[Float32](cap.f32_scales))
        var f32_gamma_off = place_quant_region(
            cursor, quant_byte_count[Float32](cap.f32_gamma))
        var bf16_centered_off = place_quant_region(
            cursor, quant_byte_count[BFloat16](cap.bf16_centered))
        var bf16_gauge_off = place_quant_region(
            cursor, quant_byte_count[BFloat16](cap.bf16_gauge))
        return Self(
            raw_off=raw_off,
            f32_work_off=f32_work_off,
            i8_quant_off=i8_quant_off,
            f32_scales_off=f32_scales_off,
            f32_gamma_off=f32_gamma_off,
            bf16_centered_off=bf16_centered_off,
            bf16_gauge_off=bf16_gauge_off,
            total_bytes=align_quant(cursor),
            cap=cap,
        )


@fieldwise_init
struct QuantScratch(TrivialRegisterPassable):
    """Non-owning typed view over a rank-local scratch arena.

    The surrounding `NumaArena` owns the bytes. This value is intentionally a
    small descriptor that can ride in the `BurstKernel` mailbox.
    """
    var base: PtrU8
    var layout: QuantScratchLayout

    @always_inline
    def raw(self) -> PtrU8:
        return self.base + self.layout.raw_off

    @always_inline
    def work(self) -> PtrF32:
        return (self.base + self.layout.f32_work_off).bitcast[Float32]()

    @always_inline
    def quant_i8(self) -> PtrI8:
        return (self.base + self.layout.i8_quant_off).bitcast[
            Scalar[DType.int8]]()

    @always_inline
    def scales(self) -> PtrF32:
        return (self.base + self.layout.f32_scales_off).bitcast[Float32]()

    @always_inline
    def gamma(self) -> PtrF32:
        return (self.base + self.layout.f32_gamma_off).bitcast[Float32]()

    @always_inline
    def router_centered(self) -> PtrBF16:
        return (self.base + self.layout.bf16_centered_off).bitcast[
            Scalar[DType.bfloat16]]()

    @always_inline
    def router_gauge(self) -> PtrBF16:
        return (self.base + self.layout.bf16_gauge_off).bitcast[
            Scalar[DType.bfloat16]]()

    @always_inline
    def raw_bytes(self) -> Int:
        return self.layout.cap.raw_bytes

    @always_inline
    def work_count(self) -> Int:
        return self.layout.cap.f32_work

    @always_inline
    def quant_i8_count(self) -> Int:
        return self.layout.cap.i8_quant

    @always_inline
    def scale_count(self) -> Int:
        return self.layout.cap.f32_scales

    @always_inline
    def gamma_count(self) -> Int:
        return self.layout.cap.f32_gamma

    @always_inline
    def router_centered_count(self) -> Int:
        return self.layout.cap.bf16_centered

    @always_inline
    def router_gauge_count(self) -> Int:
        return self.layout.cap.bf16_gauge


struct Quantizer(Movable):
    """Coordinator: discovers shards, parses headers, opens fds, walks slot
    plans, writes the safetensors header, then dispatches the per-slot work
    to one or more QuantWorker instances. The coordinator's ring is used only
    for the header write; workers each set up their own ring."""

    var ring: IoRing[QD]
    var headers: List[SafetensorsHeader]
    var shard_paths: List[Path]
    var output_path: Path
    var fds: List[Int32]
    var output_fd_idx: Int
    var data_start: Int
    var data_cursor: Int
    var entries: List[OutputEntry]
    var slots: List[SlotPlan]
    var scratch_cap: ScratchCapacity
    var ready: Bool

    def __init__(out self, source_dir: Path, output_path: Path):
        self.ring = IoRing[QD]()
        self.headers = List[SafetensorsHeader]()
        self.shard_paths = List[Path]()
        self.output_path = output_path
        self.fds = List[Int32]()
        self.output_fd_idx = -1
        self.data_start = -1
        self.data_cursor = 0
        self.entries = List[OutputEntry]()
        self.slots = List[SlotPlan]()
        self.scratch_cap = ScratchCapacity.zero(PANEL_ROWS)
        self.ready = False

        if not self.ring:
            print("quant: io_uring setup failed")
            return

        self.shard_paths = discover_shards(source_dir)
        if len(self.shard_paths) == 0:
            print(t"quant: no shards in {source_dir}")
            return
        for i in range(len(self.shard_paths)):
            var h_opt = parse_safetensors_header(self.shard_paths[i])
            if not h_opt:
                ref p = self.shard_paths[i]
                print(t"quant: failed to parse header {p}")
                return
            self.headers.append(h_opt.take())

        var sys = linux.linux_sys()
        for i in range(len(self.shard_paths)):
            var path_str = String(self.shard_paths[i])
            var fd = sys.sys_openat(linux.AT_FDCWD, path_str,
                ReadMode.OPEN_FLAGS, ReadMode.CREATE_MODE)
            if fd < 0:
                print(t"quant: open failed for {path_str}: errno {fd}")
                return
            self.fds.append(Int32(fd))

        var out_path_str = String(output_path)
        var out_dir = dirname(out_path_str)
        if out_dir.byte_length() > 0:
            try:
                makedirs(out_dir, exist_ok=True)
            except err:
                print(t"quant: mkdir failed for {out_dir}: {err}")
                return
        var ofd = sys.sys_openat(linux.AT_FDCWD, out_path_str,
            WriteMode.OPEN_FLAGS, WriteMode.CREATE_MODE)
        if ofd < 0:
            print(t"quant: open failed for {out_path_str}: errno {ofd}")
            return
        self.fds.append(Int32(ofd))
        self.output_fd_idx = len(self.fds) - 1

        try:
            _ = self.ring.register_fds(Span(self.fds))
        except err:
            print(t"quant: register_fds failed: {err.error_message()}")
            return

        self.ready = True

    def __del__(deinit self):
        var sys = linux.linux_sys()
        for fd in self.fds:
            if fd >= 0:
                _ = sys.sys_close(Int(fd))

    def __bool__(self) -> Bool:
        return self.ready

    def plan_walk[T: AnyType](
        mut self, prefix: String, layer_idx: Int = -1,
    ) -> Bool:
        """Walk `T` reflectively, emitting plans for every named SlotLike
        field and recursing into SlotGroup fields. `layer_idx` is stamped
        onto each emitted plan and propagates through recursion — set it to
        the layer index for per-layer plan walks and leave it -1 for
        layer-agnostic walks (tail tensors)."""
        comptime for i in range(reflect[T].field_count()):
            comptime FT = reflect[T].field_types()[i]
            comptime if conforms_to(FT, SlotLike):
                comptime if FT.SOURCE.group_count == 1:
                    comptime if FT.NAME:
                        if not self.plan_slot[FT](prefix, layer_idx):
                            return False
                comptime if FT.SOURCE.group_count != 1:
                    if not self.plan_group[FT](prefix, layer_idx):
                        return False
            comptime if conforms_to(FT, SlotGroup):
                if not self.plan_walk[FT](prefix, layer_idx):
                    return False
        return True

    def plan_slot[FT: SlotLike](mut self, prefix: String, layer_idx: Int) -> Bool:
        return self.plan_one[FT.ENCODING, FT.SHAPE, FT.QUANT](
            prefix + String(FT.NAME.value()), String(FT.NAME.value()),
            prefix, layer_idx)

    def plan_group[FT: SlotLike](mut self, prefix: String, layer_idx: Int) -> Bool:
        comptime per_elem_rows = FT.SHAPE.SIZE_ON_DISK_N // FT.SOURCE.group_count
        comptime ElementShape = Replicated[per_elem_rows, FT.SHAPE.SIZE_ON_DISK_M]
        var names = List[String]()
        FT.SOURCE.gen(prefix, names)
        for e in range(len(names)):
            if not self.plan_one[FT.ENCODING, ElementShape, FT.QUANT](
                names[e], names[e], prefix, layer_idx):
                return False
        return True

    def plan_one[
        encoding: Encoding, shape: ShapeLike, quant: QuantRecipe,
    ](
        mut self, full: String, local: String, prefix: String, layer_idx: Int,
    ) -> Bool:
        comptime ROWS = shape.SIZE_ON_DISK_N
        comptime COLS = shape.SIZE_ON_DISK_M
        comptime SRC = encoding.DTYPE
        comptime QV = quant

        var loc_opt = find_tensor(full, as_mut_untracked_span(self.headers))
        if not loc_opt:
            print(t"quant plan: missing {full}")
            return False
        var loc = loc_opt.take()

        if loc.dtype != SRC:
            print(t"quant plan: dtype mismatch for {full}: got {loc.dtype} expected {SRC}")
            return False
        if loc.rows != ROWS or loc.cols != COLS:
            var lr = loc.rows
            var lc = loc.cols
            print(t"quant plan: shape mismatch for {full}: got {lr}x{lc} expected {ROWS}x{COLS}")
            return False

        var offs = InlineArray[Int, 5](fill=-1)
        comptime MANIFEST = quant_manifest[encoding, shape, quant](1)
        comptime for i in range(MANIFEST.count):
            comptime MEMBER = MANIFEST.members[i]
            comptime if MEMBER.role != QuantRole.COLSUM:
                var s1 = MEMBER.global_cols if MEMBER.out_ndim == 2 else 0
                offs[MEMBER.role] = self.add_entry(
                    full + String(MEMBER.suffix), MEMBER.dtype,
                    MEMBER.global_rows, s1,
                    MEMBER.data_rows * MEMBER.data_cols * MEMBER.element_bytes)

        comptime if QV.isa[Passthrough]():
            self.plan_passthrough(full, local, layer_idx, loc,
                offs[QuantRole.WEIGHT])

        comptime if QV.isa[NormGain]():
            self.plan_passthrough(full, local, layer_idx, loc,
                offs[QuantRole.WEIGHT], offset=QV[NormGain].offset)

        comptime if QV.isa[PerRowQuant]():
            comptime QT = QV[PerRowQuant]
            if not self.plan_quant(full, local, layer_idx, loc,
                per_block=False, fwht=QT.fwht_block,
                two_sided_m=two_sided_m_of[QT.rotation](),
                gamma=gamma_ref_from[QT.gamma](prefix),
                weight_off=offs[QuantRole.WEIGHT], scale_off=offs[QuantRole.SCALE]):
                return False

        comptime if QV.isa[PerBlockQuant]():
            comptime QT = QV[PerBlockQuant]
            if not self.plan_quant(full, local, layer_idx, loc,
                per_block=True, fwht=QT.fwht_block,
                two_sided_m=two_sided_m_of[QT.rotation](),
                gamma=gamma_ref_from[QT.gamma](prefix),
                weight_off=offs[QuantRole.WEIGHT], scale_off=offs[QuantRole.SCALE]):
                return False

        comptime if QV.isa[RouterCenter]():
            comptime QT = QV[RouterCenter]
            var bias_name = String("")
            comptime if QT.bias_name != StaticString(""):
                bias_name = prefix + String(QT.bias_name)
            if not self.plan_router(full, local, layer_idx, loc, bias_name,
                weight_off=offs[QuantRole.WEIGHT], gauge_off=offs[QuantRole.GAUGE],
                emit_gauge=True, bias_off=offs[QuantRole.BIAS]):
                return False

        comptime if QV.isa[SoftmaxRouterCenter]():
            if not self.plan_router(full, local, layer_idx, loc, String(""),
                weight_off=offs[QuantRole.WEIGHT], gauge_off=-1,
                emit_gauge=False, bias_off=-1):
                return False

        return True

    def plan_passthrough(
        mut self, name: String, local: String, layer_idx: Int,
        loc: LocatedTensor, weight_off: Int, offset: Float32 = 0.0,
    ):
        var bytes = loc.rows * loc.cols * size_of_dtype(loc.dtype)
        if offset != 0.0:
            self.scratch_cap.absorb_raw(bytes)
        else:
            self.scratch_cap.absorb_raw(min(COPY_CHUNK, bytes))
        var id = SlotIdentity(
            name=name, local_name=local, layer_idx=layer_idx,
            shard=loc.shard, src_offset=loc.data_start,
            src_dtype=loc.dtype, rows=loc.rows, cols=loc.cols,
            weight_off=weight_off,
        )
        self.slots.append(PassthroughPlan(id^, bytes, offset))

    def locate_gamma(mut self, mut g: GammaRef, expected_cols: Int) -> Bool:
        var loc_opt = find_tensor(g.name, as_mut_untracked_span(self.headers))
        if not loc_opt:
            print(t"quant plan: missing gamma {g.name}")
            return False
        var loc = loc_opt.take()
        var cols = loc.rows * loc.cols
        if cols != expected_cols:
            print(t"quant plan: gamma {g.name} cols {cols} != expected {expected_cols}")
            return False
        if not supports_decode_to_f32(loc.dtype):
            print(t"quant plan: unsupported gamma dtype for {g.name}: {loc.dtype}")
            return False
        g.locate(loc.shard, loc.data_start, loc.byte_size, loc.dtype, cols)
        return True

    def plan_quant(
        mut self, name: String, local: String, layer_idx: Int,
        loc: LocatedTensor, per_block: Bool,
        fwht: Int, two_sided_m: Int, var gamma: GammaRef,
        weight_off: Int, scale_off: Int,
    ) -> Bool:
        if loc.rows <= 0 or loc.cols <= 0:
            print(t"quant plan: invalid quant shape for {name}: {loc.rows}x{loc.cols}")
            return False
        if not supports_decode_to_f32(loc.dtype):
            print(t"quant plan: unsupported source dtype for quant {name}: {loc.dtype}")
            return False
        if loc.cols % fwht != 0:
            print(t"quant plan: cols {loc.cols} not divisible by K-axis FWHT block {fwht} for {name}")
            return False
        if two_sided_m != 0:
            if loc.rows % two_sided_m != 0:
                print(t"quant plan: rows {loc.rows} not divisible by M-axis FWHT block {two_sided_m} for {name}")
                return False
        if gamma.is_present():
            if not self.locate_gamma(gamma, loc.cols):
                return False

        var id = SlotIdentity(
            name=name, local_name=local, layer_idx=layer_idx,
            shard=loc.shard, src_offset=loc.data_start,
            src_dtype=loc.dtype, rows=loc.rows, cols=loc.cols,
            weight_off=weight_off,
        )
        var plan = QuantPlan(
            id^, per_block, fwht, two_sided_m, gamma^, scale_off,
        )
        self.scratch_cap.absorb_quant(plan, size_of_dtype(loc.dtype))
        self.slots.append(plan^)
        return True

    def plan_router(
        mut self, name: String, local: String, layer_idx: Int,
        loc: LocatedTensor, bias_name: String,
        weight_off: Int, gauge_off: Int, emit_gauge: Bool, bias_off: Int,
    ) -> Bool:
        if loc.rows <= 0 or loc.cols <= 0:
            print(t"quant plan: invalid router shape for {name}: {loc.rows}x{loc.cols}")
            return False
        if not supports_router_source(loc.dtype):
            print(t"quant plan: router_center only supports bf16/f32 source for {name}: {loc.dtype}")
            return False
        if loc.cols % W != 0:
            print(t"quant plan: router cols {loc.cols} not divisible by SIMD width {W} for {name}")
            return False

        var bias_shard = 0
        var bias_src_offset = 0
        var bias_byte_size = 0
        var bias_src_dtype = DType.float32
        if bias_name.byte_length() > 0:
            var bias_loc_opt = find_tensor(bias_name, as_mut_untracked_span(self.headers))
            if not bias_loc_opt:
                print(t"quant plan: missing router bias {bias_name}")
                return False
            var bias_loc = bias_loc_opt.take()
            var bias_rows = bias_loc.rows * bias_loc.cols
            if bias_rows != loc.rows:
                print(t"quant plan: router bias {bias_name} rows {bias_rows} != expected {loc.rows}")
                return False
            if not supports_decode_to_f32(bias_loc.dtype):
                print(t"quant plan: unsupported router bias dtype for {bias_name}: {bias_loc.dtype}")
                return False
            bias_shard = bias_loc.shard
            bias_src_offset = bias_loc.data_start
            bias_byte_size = bias_loc.byte_size
            bias_src_dtype = bias_loc.dtype

        var src_bytes_per = size_of_dtype(loc.dtype)
        self.scratch_cap.absorb_raw(loc.rows * loc.cols * src_bytes_per)
        self.scratch_cap.absorb_f32_work(loc.cols)
        self.scratch_cap.absorb_bf16_centered(loc.rows * loc.cols)
        if emit_gauge:
            self.scratch_cap.absorb_bf16_gauge(loc.cols)
        if bias_name.byte_length() > 0:
            self.scratch_cap.absorb_raw(bias_byte_size)
            self.scratch_cap.absorb_f32_work(loc.rows)

        var id = SlotIdentity(
            name=name, local_name=local, layer_idx=layer_idx,
            shard=loc.shard, src_offset=loc.data_start,
            src_dtype=loc.dtype, rows=loc.rows, cols=loc.cols,
            weight_off=weight_off,
        )
        self.slots.append(RouterPlan(
            id^, gauge_off, emit_gauge, bias_name, bias_off,
            bias_shard, bias_src_offset, bias_byte_size, bias_src_dtype))
        return True

    def add_entry(
        mut self, name: String, dt: DType, s0: Int, s1: Int, size: Int,
    ) -> Int:
        var off = self.data_cursor
        self.entries.append(OutputEntry(
            name=name, dtype=dt, shape0=s0, shape1=s1,
            data_start=off, data_end=off + size,
        ))
        self.data_cursor += size
        return off

    def write_header(mut self) -> Bool:
        var header = build_header(self.entries)
        var header_size = len(header)
        var hp = header.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        if not write_sync(self.ring, self.output_fd_idx, 0, header_size, hp):
            return False
        self.data_start = header_size
        _ = header^
        return True

    def execute[P: BurstThreadPool, //](
        mut self, topo: NumaTopology, var pools: List[P],
    ) -> Bool:
        """Run the plan across the caller's `pools`, one job per pool worker. The
        caller is responsible for constructing pools — typically via
        `with_topological_rank_dispatch` so isolation mode and pinning match
        the rest of the system. Slots are bin-packed across workers; each
        worker opens its own io_uring and writes to preassigned, disjoint
        regions of the output file."""
        var num_ranks = len(pools)
        if num_ranks <= 0:
            print("quant: empty pool list")
            return False
        if num_ranks > len(topo):
            var rl = len(topo)
            print(t"quant: pool count {num_ranks} exceeds topology rank length {rl}")
            return False

        var workers_per_rank = List[Int](capacity=num_ranks)
        var worker_rank = List[Int]()
        var worker_idx = List[Int]()
        for r in range(num_ranks):
            var cap = pools[r].get_capacity()
            if cap <= 0:
                print(t"quant: pool {r} has no workers")
                return False
            workers_per_rank.append(cap)
            for w in range(cap):
                worker_rank.append(r)
                worker_idx.append(w)

        var num_workers = len(worker_rank)
        if num_workers <= 0:
            print("quant: empty worker list")
            return False

        var buckets = partition_slots(self.slots, num_workers)
        var scratch_layout = QuantScratchLayout.build(self.scratch_cap)
        var scratch_bytes = scratch_layout.total_bytes
        if scratch_bytes <= 0:
            print("quant: invalid scratch arena size")
            return False

        var scratch_arenas = List[
            NumaArena[alignment=QUANT_SCRATCH_ALIGNMENT]
        ](capacity=num_ranks)
        var scratches = List[QuantScratch](capacity=num_workers)
        for r in range(num_ranks):
            var rank_scratch_bytes = scratch_bytes * workers_per_rank[r]
            scratch_arenas.append(
                NumaArena[alignment=QUANT_SCRATCH_ALIGNMENT](
                    topo.node(r), rank_scratch_bytes))
            if not scratch_arenas[r]:
                var node = topo.node(r)
                print(t"quant: scratch arena allocation failed on node {node}")
                return False
            for _ in range(workers_per_rank[r]):
                var base = scratch_arenas[r].alloc[UInt8](scratch_bytes)
                if not base:
                    print(t"quant: scratch arena exhausted for rank {r}")
                    return False
                scratches.append(QuantScratch(base.value(), scratch_layout))
            _ = scratch_arenas[r].prefault(0, scratch_arenas[r].used())

        var kernels = List[QuantShardKernel](capacity=num_workers)
        for w in range(num_workers):
            kernels.append(QuantShardKernel(
                fds=as_mut_untracked_span(self.fds),
                output_fd_idx=self.output_fd_idx,
                headers=as_mut_untracked_span(self.headers),
                slots=as_mut_untracked_span(buckets[w]),
                data_start=self.data_start,
                scratch=scratches[w],
                rank=worker_rank[w],
                worker_idx=worker_idx[w],
            ))

        var pool_base = pools.unsafe_ptr()
        var worker_start = 0
        for r in range(num_ranks):
            var jobs = workers_per_rank[r]
            var kernel_span = Span[QuantShardKernel, MutUntrackedOrigin](
                ptr=UnsafePointer(to=kernels[worker_start]).unsafe_origin_cast[MutUntrackedOrigin](),
                length=jobs)
            (pool_base + r)[].dispatch(kernel_span, jobs)
            worker_start += jobs
        for r in range(num_ranks):
            (pool_base + r)[].join()

        _ = kernels^
        _ = scratches^
        _ = scratch_arenas^
        _ = buckets
        _ = worker_rank^
        _ = worker_idx^
        _ = workers_per_rank^
        _ = pools^
        return True


@always_inline
def size_of_dtype(dt: DType) -> Int:
    return dtype_byte_size(dt)


@always_inline
def gamma_label(ref g: GammaRef) -> StaticString:
    if not g.is_present(): return "none"
    if g.absorbed: return "absorbed"
    return "split"


def slot_name(read plan: SlotPlan) -> String:
    if plan.isa[PassthroughPlan]():
        return plan[PassthroughPlan].id.name
    if plan.isa[QuantPlan]():
        return plan[QuantPlan].id.name
    if plan.isa[RouterPlan]():
        return plan[RouterPlan].id.name
    return String("<unknown>")


struct QuantWorker(Movable):
    """One thread's worth of quantization state: its own io_uring and
    arena-backed staging scratch. Fields holding spans into shared, read-only
    storage (fds / headers) carry origins stripped to MutUntrackedOrigin so that
    QuantWorker is safe to construct on a worker thread; the caller is
    responsible for keeping the backing storage alive."""

    var ring: IoRing[QD]
    var scratch: QuantScratch
    var fds: Span[Int32, MutUntrackedOrigin]
    var output_fd_idx: Int
    var headers: Span[SafetensorsHeader, MutUntrackedOrigin]
    var data_start: Int
    var rank: Int
    var worker_idx: Int
    var ready: Bool

    def __init__(
        out self,
        fds: Span[Int32, MutUntrackedOrigin],
        output_fd_idx: Int,
        headers: Span[SafetensorsHeader, MutUntrackedOrigin],
        data_start: Int,
        scratch: QuantScratch,
        rank: Int = 0,
        worker_idx: Int = 0,
    ):
        self.fds = fds
        self.output_fd_idx = output_fd_idx
        self.headers = headers
        self.data_start = data_start
        self.rank = rank
        self.worker_idx = worker_idx
        self.ring = IoRing[QD]()
        self.scratch = scratch
        self.ready = False
        if not self.ring:
            print("quant worker: io_uring setup failed")
            return
        try:
            _ = self.ring.register_fds(self.fds)
        except err:
            print(t"quant worker: register_fds failed: {err.error_message()}")
            return
        self.ready = True

    def __bool__(self) -> Bool:
        return self.ready

    def run(mut self, slots: Span[SlotPlan, MutUntrackedOrigin]) -> Bool:
        for i in range(len(slots)):
            var plan = slots[i].copy()
            if not self.execute_slot(plan):
                self.report_slot_failure(plan)
                return False
        return True

    def execute_slot(mut self, plan: SlotPlan) -> Bool:
        self.log_slot(plan)
        if plan.isa[PassthroughPlan]():
            return self.do_passthrough(plan[PassthroughPlan])
        if plan.isa[QuantPlan]():
            return self.do_quant(plan[QuantPlan])
        if plan.isa[RouterPlan]():
            return self.do_router(plan[RouterPlan].copy())
        return False

    def log_slot(self, plan: SlotPlan):
        """One line per slot: rank/worker tag, layer marker, slot's local
        name, then a variant-specific description. Streamed via t-string
        so each line is one print call with no String allocation."""
        if plan.isa[PassthroughPlan]():
            ref pp = plan[PassthroughPlan]
            self.log_line(pp.id,
                t"passthrough {dtype_tag(pp.id.src_dtype)} {pp.byte_count}B")
        elif plan.isa[QuantPlan]():
            var qp = plan[QuantPlan].copy()
            var shape = StaticString("per-block") if qp.per_block else StaticString("per-row")
            var g = gamma_label(qp.gamma)
            if qp.two_sided_m != 0:
                self.log_line(qp.id,
                    t"{shape} fwht={qp.fwht_block} 2x={qp.two_sided_m} γ={g}")
            else:
                self.log_line(qp.id,
                    t"{shape} fwht={qp.fwht_block} γ={g}")
        elif plan.isa[RouterPlan]():
            var rp = plan[RouterPlan].copy()
            if rp.bias_name.byte_length() > 0:
                self.log_line(rp.id,
                    t"router-center +bias({rp.bias_name})")
            elif not rp.emit_gauge:
                self.log_line(rp.id, t"softmax-router-center")
            else:
                self.log_line(rp.id, t"router-center")

    def log_line[W: Writable](self, id: SlotIdentity, desc: W):
        if id.layer_idx >= 0:
            print(t"[r{self.rank}/w{self.worker_idx} L{id.layer_idx}] {id.local_name} :: {desc}")
        else:
            print(t"[r{self.rank}/w{self.worker_idx} tail ] {id.local_name} :: {desc}")

    def report_slot_failure(self, plan: SlotPlan):
        var name = slot_name(plan)
        print(t"quant: failed at {name}")

    def get_gamma(mut self, ref g: GammaRef) -> Optional[PtrF32]:
        var cols = g.cols
        if g.byte_size > self.scratch.raw_bytes():
            print(t"quant: gamma {g.name} raw bytes exceed scratch")
            return None
        if cols > self.scratch.gamma_count():
            print(t"quant: gamma {g.name} cols exceed scratch")
            return None

        if not read_sync(self.ring, g.shard, g.src_offset, g.byte_size,
                self.scratch.raw()):
            return None

        if not decode_to_f32(g.src_dtype, self.scratch.raw(),
                self.scratch.gamma(), cols):
            print(t"quant: unsupported gamma dtype for {g.name}: {g.src_dtype}")
            return None

        if g.offset != 0.0:
            add_offset_in_place[DType.float32](self.scratch.gamma(), cols, g.offset)
        if not g.absorbed:
            gamma_sqrt_abs_in_place(self.scratch.gamma(), cols)
        return self.scratch.gamma()

    def do_gain(mut self, p: PassthroughPlan) -> Bool:
        if p.id.src_dtype != DType.bfloat16:
            print(t"quant: gain offset requires bf16 source for {p.id.name}")
            return False
        if p.byte_count > self.scratch.raw_bytes():
            print("quant: gain tensor exceeds scratch")
            return False
        if not read_sync(self.ring, p.id.shard, p.id.src_offset,
                p.byte_count, self.scratch.raw()):
            return False
        add_offset_in_place[DType.bfloat16](
            self.scratch.raw().bitcast[Scalar[DType.bfloat16]](),
            p.byte_count // 2, p.offset)
        if not write_sync(self.ring, self.output_fd_idx,
                self.data_start + p.id.weight_off, p.byte_count,
                self.scratch.raw()):
            return False
        return True

    def do_passthrough(mut self, p: PassthroughPlan) -> Bool:
        if p.offset != 0.0:
            return self.do_gain(p)
        var chunk_cap = min(COPY_CHUNK, self.scratch.raw_bytes())
        if chunk_cap <= 0:
            print("quant: passthrough scratch unavailable")
            return False
        var copied = 0
        while copied < p.byte_count:
            var n = min(chunk_cap, p.byte_count - copied)
            if not read_sync(self.ring, p.id.shard,
                    p.id.src_offset + copied, n, self.scratch.raw()):
                return False
            if not write_sync(self.ring, self.output_fd_idx,
                    self.data_start + p.id.weight_off + copied, n,
                    self.scratch.raw()):
                return False
            copied += n
        return True

    def do_quant(mut self, p: QuantPlan) -> Bool:
        var src_bytes_per = size_of_dtype(p.id.src_dtype)
        if src_bytes_per <= 0:
            print(t"quant: bad src dtype {p.id.src_dtype}")
            return False
        if (p.id.src_dtype != DType.bfloat16
                and p.id.src_dtype != DType.float32
                and p.id.src_dtype != DType.float16):
            print(t"quant: unsupported source dtype for quant: {p.id.src_dtype}")
            return False

        var gamma_ptr = Optional[PtrF32](None)
        if p.gamma.is_present():
            var g = self.get_gamma(p.gamma)
            if not g:
                return False
            gamma_ptr = g.value()

        var nb = p.id.cols // p.fwht_block if p.per_block else 1
        var scale_per_row = nb if p.per_block else 1

        var pr = min(PANEL_ROWS, p.id.rows)
        var row_off = 0
        while row_off < p.id.rows:
            var panel_rows = min(pr, p.id.rows - row_off)
            var panel_bytes = panel_rows * p.id.cols * src_bytes_per
            if panel_bytes > self.scratch.raw_bytes():
                print("quant: source panel exceeds scratch")
                return False
            if panel_rows * p.id.cols > self.scratch.work_count():
                print("quant: work panel exceeds scratch")
                return False
            if panel_rows * p.id.cols > self.scratch.quant_i8_count():
                print("quant: quant panel exceeds scratch")
                return False
            if panel_rows * scale_per_row > self.scratch.scale_count():
                print("quant: scale panel exceeds scratch")
                return False
            var src_off = p.id.src_offset + row_off * p.id.cols * src_bytes_per
            if not read_sync(self.ring, p.id.shard, src_off, panel_bytes,
                    self.scratch.raw()):
                return False

            var work = self.scratch.work()
            var qi = self.scratch.quant_i8()
            var scales = self.scratch.scales()
            if not decode_to_f32(p.id.src_dtype, self.scratch.raw(),
                    work, panel_rows * p.id.cols):
                return False

            if gamma_ptr:
                var gp = gamma_ptr.value()
                for r in range(panel_rows):
                    apply_gamma_in_place(
                        work + r * p.id.cols, gp, p.id.cols)

            if p.per_block:
                rotate_and_quant[True](
                    p.fwht_block, work, qi, scales,
                    panel_rows, p.id.cols, p.two_sided_m)
            else:
                rotate_and_quant[False](
                    p.fwht_block, work, qi, scales,
                    panel_rows, p.id.cols, p.two_sided_m)

            var w_off = self.data_start + p.id.weight_off + row_off * p.id.cols
            var s_off = (self.data_start + p.scale_off
                + row_off * scale_per_row * 4)

            var ops = InlineArray[WriteOp[], 2](uninitialized=True)
            ops[0] = WriteOp(
                file_idx=self.output_fd_idx, offset=w_off,
                length=panel_rows * p.id.cols,
                src=qi.bitcast[UInt8](), id=0)
            ops[1] = WriteOp(
                file_idx=self.output_fd_idx, offset=s_off,
                length=panel_rows * scale_per_row * 4,
                src=scales.bitcast[UInt8](), id=1)
            var n_ops = 2

            var ops_span = Span[WriteOp[], MutUntrackedOrigin](
                ptr=UnsafePointer(to=ops[0]).unsafe_origin_cast[MutUntrackedOrigin](),
                length=n_ops)
            var write_ok = write_sync_many(self.ring, ops_span)
            _ = ops^
            if not write_ok:
                return False

            row_off += panel_rows
        return True

    def do_router(mut self, var p: RouterPlan) -> Bool:
        var rows = p.id.rows
        var cols = p.id.cols
        if p.id.src_dtype != DType.bfloat16 and p.id.src_dtype != DType.float32:
            print(t"quant: router_center only supports bf16/f32 source; got {p.id.src_dtype}")
            return False
        var src_bytes_per = size_of_dtype(p.id.src_dtype)
        var src_bytes = rows * cols * src_bytes_per
        if src_bytes > self.scratch.raw_bytes():
            print("quant: router source exceeds scratch")
            return False
        if cols > self.scratch.work_count():
            print("quant: router gauge exceeds scratch")
            return False
        if rows * cols > self.scratch.router_centered_count():
            print("quant: router centered output exceeds scratch")
            return False
        if p.emit_gauge and cols > self.scratch.router_gauge_count():
            print("quant: router gauge bf16 exceeds scratch")
            return False

        var src_buf = self.scratch.raw()
        var gauge_f32 = self.scratch.work()
        var centered = self.scratch.router_centered()
        var gauge_bf16 = self.scratch.router_gauge()

        var ok = read_sync(self.ring, p.id.shard, p.id.src_offset,
            rows * cols * src_bytes_per, src_buf)

        if ok:
            if p.id.src_dtype == DType.bfloat16:
                if p.emit_gauge:
                    router_center[DType.bfloat16](
                        src_buf.bitcast[Scalar[DType.bfloat16]](),
                        gauge_f32, centered, gauge_bf16, rows, cols)
                else:
                    router_center_softmax[DType.bfloat16](
                        src_buf.bitcast[Scalar[DType.bfloat16]](),
                        gauge_f32, centered, rows, cols)
            else:
                if p.emit_gauge:
                    router_center[DType.float32](
                        src_buf.bitcast[Float32](),
                        gauge_f32, centered, gauge_bf16, rows, cols)
                else:
                    router_center_softmax[DType.float32](
                        src_buf.bitcast[Float32](),
                        gauge_f32, centered, rows, cols)

            ok = write_sync(self.ring, self.output_fd_idx,
                self.data_start + p.id.weight_off, rows * cols * 2,
                centered.bitcast[UInt8]())

        if ok and p.emit_gauge:
            ok = write_sync(self.ring, self.output_fd_idx,
                self.data_start + p.gauge_off, cols * 2,
                gauge_bf16.bitcast[UInt8]())

        if ok and p.bias_name.byte_length() > 0:
            ok = self.write_router_bias(p)

        return ok

    def write_router_bias(mut self, ref p: RouterPlan) -> Bool:
        var rows = p.id.rows
        if p.bias_byte_size > self.scratch.raw_bytes():
            print(t"quant: router bias {p.bias_name} raw bytes exceed scratch")
            return False
        if rows > self.scratch.work_count():
            print(t"quant: router bias {p.bias_name} rows exceed scratch")
            return False
        if not read_sync(self.ring, p.bias_shard, p.bias_src_offset,
                p.bias_byte_size, self.scratch.raw()):
            return False
        if not decode_to_f32(p.bias_src_dtype, self.scratch.raw(),
                self.scratch.work(), rows):
            print(t"quant: unsupported router bias dtype for {p.bias_name}: {p.bias_src_dtype}")
            return False
        var ok = write_sync(self.ring, self.output_fd_idx,
            self.data_start + p.bias_off, rows * 4,
            self.scratch.work().bitcast[UInt8]())
        return ok


def estimate_slot_bytes(plan: SlotPlan) -> Int:
    """Rough I/O+compute weight for bin-packing slots across workers.
    Counts source-read bytes plus output bytes; gamma reads are amortized
    across the slots that share a gamma and are dropped here."""
    if plan.isa[PassthroughPlan]():
        return plan[PassthroughPlan].byte_count
    if plan.isa[QuantPlan]():
        var q = plan[QuantPlan].copy()
        var rows = q.id.rows
        var cols = q.id.cols
        var src = rows * cols * dtype_byte_size(q.id.src_dtype)
        var weight = rows * cols
        var nb = (cols // q.fwht_block) if q.per_block else 1
        var scale = rows * nb * 4
        return src + weight + scale
    if plan.isa[RouterPlan]():
        var r = plan[RouterPlan].copy()
        return r.id.rows * r.id.cols * 3
    return 0


def partition_slots(
    ref slots: List[SlotPlan], num_workers: Int,
) -> List[List[SlotPlan]]:
    """Greedy descending bin-pack: sort slots by estimated bytes desc and
    assign each to the worker with least current load. Selection sort over
    indices is O(n²) but n~1500 and this runs once per quantize."""
    var n = len(slots)
    var sizes = List[Int](capacity=n)
    var order = List[Int](capacity=n)
    for i in range(n):
        sizes.append(estimate_slot_bytes(slots[i]))
        order.append(i)
    for i in range(n - 1):
        var max_idx = i
        for j in range(i + 1, n):
            if sizes[order[j]] > sizes[order[max_idx]]:
                max_idx = j
        if max_idx != i:
            var tmp = order[i]
            order[i] = order[max_idx]
            order[max_idx] = tmp

    var buckets = List[List[SlotPlan]](capacity=num_workers)
    var loads = List[Int](capacity=num_workers)
    for _ in range(num_workers):
        buckets.append(List[SlotPlan]())
        loads.append(0)

    for k in range(n):
        var idx = order[k]
        var sz = sizes[idx]
        var min_w = 0
        for w in range(1, num_workers):
            if loads[w] < loads[min_w]:
                min_w = w
        buckets[min_w].append(slots[idx].copy())
        loads[min_w] += sz
    return buckets^


@fieldwise_init
struct QuantShardKernel(BurstKernel):
    """Per-worker payload dispatched through a BurstPool mailbox. Owns no
    resources — every field is POD with caller-managed backing storage.
    Mirrors `LoadShardKernel` in `linux/io_uring.mojo:601`."""
    var fds: Span[Int32, MutUntrackedOrigin]
    var output_fd_idx: Int
    var headers: Span[SafetensorsHeader, MutUntrackedOrigin]
    var slots: Span[SlotPlan, MutUntrackedOrigin]
    var data_start: Int
    var scratch: QuantScratch
    var rank: Int
    var worker_idx: Int

    def execute(mut self):
        var sys = linux.linux_sys()
        var worker = QuantWorker(
            fds=self.fds, output_fd_idx=self.output_fd_idx,
            headers=self.headers, data_start=self.data_start,
            scratch=self.scratch, rank=self.rank, worker_idx=self.worker_idx,
        )
        if not worker:
            print("quant worker: setup failed")
            sys.sys_exit_group(1)
            return
        if not worker.run(self.slots):
            print("quant worker: run failed")
            sys.sys_exit_group(1)
            return
