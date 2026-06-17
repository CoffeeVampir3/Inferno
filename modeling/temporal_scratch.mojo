from std.builtin.rebind import downcast
from std.collections import InlineArray
from std.memory import UnsafePointer
from std.reflection import reflect
from std.sys.info import size_of

from kernels.helpers import Binding
from modeling.model_spec import ShapeLike
from modeling.slot import BindContext


comptime SCRATCH_ALIGNMENT = 64
comptime MAX_SCRATCH_SLOTS = 64


@always_inline
def aligned_scratch_bytes(nbytes: Int) -> Int:
    return ((nbytes + SCRATCH_ALIGNMENT - 1) // SCRATCH_ALIGNMENT) * SCRATCH_ALIGNMENT


struct ScaleClass:
    """How a scratch buffer's element count scales with the runtime layout.
    `base_elems` is the degree-1, single-worker unit. PER_DEGREE divides by the
    runtime tensor-parallel degree (sharded buffers); PER_WORKER multiplies by
    the runtime worker count (per-worker partials); PER_WORKER_PER_DEGREE
    multiplies by both for per-worker cross-rank scratch; FIXED is constant."""
    comptime FIXED = 0
    comptime PER_DEGREE = 1
    comptime PER_WORKER = 2
    comptime PER_WORKER_PER_DEGREE = 3


@always_inline
def resolve_scratch_elems(
    base_elems: Int, scale: Int, degree: Int, workers: Int,
) -> Int:
    var n = base_elems
    if scale == ScaleClass.PER_DEGREE:
        n = base_elems // degree
    elif scale == ScaleClass.PER_WORKER:
        n = base_elems * workers
    elif scale == ScaleClass.PER_WORKER_PER_DEGREE:
        n = base_elems * workers * degree
    return n


trait ScratchBufferLike:
    comptime Element: AnyType
    comptime ELEMENT_SIZE: Int

    @staticmethod
    def scratch_elems(degree: Int, workers: Int) -> Int: ...


@fieldwise_init
struct ScratchBuffer[T: AnyType, base_elems: Int, scale: Int = ScaleClass.FIXED](
    ScratchBufferLike, Copyable, ImplicitlyCopyable
):
    comptime Element = Self.T
    comptime ELEMENT_SIZE = size_of[Self.T]()

    @staticmethod
    @always_inline
    def scratch_elems(degree: Int, workers: Int) -> Int:
        return resolve_scratch_elems(
            Self.base_elems, Self.scale, degree, workers)


@fieldwise_init
struct ShardedScratchBuffer[
    T: AnyType, rows: Int, S: ShapeLike, col_block: Int = 1,
](ScratchBufferLike, Copyable, ImplicitlyCopyable):
    """Per-rank scratch for row-sharded tensor products: `rows` step rows by
    the shape's per-rank extent. Sizing flows through `S.data_n(degree)` so the
    shard padding has a single source of truth shared with the weight pack and
    the dispatch row counts. `col_block` shrinks the per-rank extent for
    sidecars that carry one element per quantization block."""
    comptime Element = Self.T
    comptime ELEMENT_SIZE = size_of[Self.T]()

    @staticmethod
    @always_inline
    def scratch_elems(degree: Int, workers: Int) -> Int:
        return Self.rows * (Self.S.data_n(degree) // Self.col_block)


trait ScratchPhaseOrderLike:
    @staticmethod
    def index[name: StaticString]() -> Int: ...


struct ScratchPhaseOrder[*names: StaticString](ScratchPhaseOrderLike):
    @staticmethod
    def index[name: StaticString]() -> Int:
        comptime for i in range(len(Self.names)):
            comptime if Self.names[i] == name:
                return i
        return -1


trait ScratchPhaseRange:
    comptime FIRST_NAME: StaticString
    comptime LAST_NAME: StaticString


@fieldwise_init
struct ScratchPhase[first: StaticString, last: StaticString](
    ScratchPhaseRange, Copyable, ImplicitlyCopyable
):
    comptime FIRST_NAME = Self.first
    comptime LAST_NAME = Self.last


trait ScratchPhaseSchema:
    comptime PHASES: ScratchPhaseOrderLike

    @staticmethod
    def phase_index[name: StaticString]() -> Int:
        return Self.PHASES.index[name]()


trait ScratchIsland(ScratchPhaseSchema):
    pass


@fieldwise_init
struct ScratchPlan(Copyable, ImplicitlyCopyable):
    var offsets: InlineArray[Int, MAX_SCRATCH_SLOTS]
    var peak: Int


def derive_scratch_plan[T: ScratchPhaseSchema](
    degree: Int, workers: Int,
) -> ScratchPlan:
    """Greedy interval bin-packing for one island at the runtime (degree,
    workers). The phase/lifetime structure is comptime; only the per-buffer
    sizes are runtime. Buffers with disjoint phase intervals may share bytes."""
    comptime assert reflect[T].field_count() <= MAX_SCRATCH_SLOTS, (
        "scratch island field count exceeds scratch offset capacity")
    var sizes = InlineArray[Int, MAX_SCRATCH_SLOTS](fill=0)
    var firsts = InlineArray[Int, MAX_SCRATCH_SLOTS](fill=0)
    var lasts = InlineArray[Int, MAX_SCRATCH_SLOTS](fill=0)
    var fields = InlineArray[Int, MAX_SCRATCH_SLOTS](fill=0)
    var field_offsets = InlineArray[Int, MAX_SCRATCH_SLOTS](fill=0)
    var n = 0
    var cur_first = -1
    var cur_last = -1

    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, ScratchPhaseRange):
            comptime first = T.phase_index[FT.FIRST_NAME]()
            comptime last = T.phase_index[FT.LAST_NAME]()
            comptime assert first >= 0, "scratch phase start is not declared in PHASES"
            comptime assert last >= 0, "scratch phase end is not declared in PHASES"
            comptime assert last >= first, "scratch phase end precedes start"
            cur_first = first
            cur_last = last
        comptime if conforms_to(FT, ScratchBufferLike):
            debug_assert(
                cur_first >= 0 and cur_last >= cur_first,
                "scratch buffer declared without a valid phase",
            )
            sizes[n] = aligned_scratch_bytes(
                FT.scratch_elems(degree, workers) * FT.ELEMENT_SIZE)
            firsts[n] = cur_first
            lasts[n] = cur_last
            fields[n] = i
            n += 1

    var order = InlineArray[Int, MAX_SCRATCH_SLOTS](fill=0)
    for k in range(n):
        order[k] = k
    for i in range(n):
        var best = i
        for j in range(i + 1, n):
            if sizes[order[j]] > sizes[order[best]]:
                best = j
        var tmp = order[i]
        order[i] = order[best]
        order[best] = tmp

    var placed_offsets = InlineArray[Int, MAX_SCRATCH_SLOTS](fill=0)
    var placed = InlineArray[Bool, MAX_SCRATCH_SLOTS](fill=False)
    var peak = 0
    for k in range(n):
        var idx = order[k]
        var x = 0
        var stable = False
        while not stable:
            stable = True
            for j in range(n):
                if not placed[j]:
                    continue
                if firsts[idx] > lasts[j] or lasts[idx] < firsts[j]:
                    continue
                var jl = placed_offsets[j]
                var jh = placed_offsets[j] + sizes[j]
                if x < jh and jl < x + sizes[idx]:
                    x = jh
                    stable = False
                    break
        placed_offsets[idx] = x
        placed[idx] = True
        field_offsets[fields[idx]] = x
        if x + sizes[idx] > peak:
            peak = x + sizes[idx]

    return ScratchPlan(offsets=field_offsets, peak=peak)


def co_live_buffers_overlap[T: ScratchPhaseSchema](
    plan: ScratchPlan, degree: Int, workers: Int,
) -> Bool:
    """Soundness check: any two buffers whose phase intervals overlap must hold
    disjoint byte ranges. Run as a load-time debug_assert."""
    var sizes = InlineArray[Int, MAX_SCRATCH_SLOTS](fill=0)
    var firsts = InlineArray[Int, MAX_SCRATCH_SLOTS](fill=0)
    var lasts = InlineArray[Int, MAX_SCRATCH_SLOTS](fill=0)
    var offs = InlineArray[Int, MAX_SCRATCH_SLOTS](fill=0)
    var n = 0
    var cur_first = -1
    var cur_last = -1
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, ScratchPhaseRange):
            cur_first = T.phase_index[FT.FIRST_NAME]()
            cur_last = T.phase_index[FT.LAST_NAME]()
        comptime if conforms_to(FT, ScratchBufferLike):
            sizes[n] = aligned_scratch_bytes(
                FT.scratch_elems(degree, workers) * FT.ELEMENT_SIZE)
            firsts[n] = cur_first
            lasts[n] = cur_last
            offs[n] = plan.offsets[i]
            n += 1
    for a in range(n):
        for b in range(a + 1, n):
            if firsts[a] > lasts[b] or lasts[a] < firsts[b]:
                continue
            var la = offs[a]
            var ha = offs[a] + sizes[a]
            var lb = offs[b]
            var hb = offs[b] + sizes[b]
            if la < hb and lb < ha:
                return True
    return False


def derive_checked_plan[T: ScratchPhaseSchema](
    degree: Int, workers: Int,
) -> ScratchPlan:
    var plan = derive_scratch_plan[T](degree, workers)
    debug_assert(
        not co_live_buffers_overlap[T](plan, degree, workers),
        "scratch plan overlaps co-live buffers",
    )
    return plan


def aggregate_scratch_peak[T: AnyType](degree: Int, workers: Int) -> Int:
    var m = 0
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, ScratchIsland):
            var peak = derive_scratch_plan[FT](degree, workers).peak
            if peak > m:
                m = peak
    return m


struct TemporalScratchPool(Movable, Copyable, ImplicitlyCopyable):
    var scratch_off: Int

    def __init__(out self, scratch_off: Int):
        self.scratch_off = scratch_off

    @always_inline
    def slot[
        I: ScratchIsland, name: StringLiteral,
    ](self, arena_base: Int, plan: ScratchPlan) -> UnsafePointer[
        downcast[reflect[I].field_type[name].T, ScratchBufferLike].Element,
        MutAnyOrigin,
    ]:
        comptime idx = reflect[I].field_index[name]()
        var off = plan.offsets[idx]
        return UnsafePointer[
            downcast[reflect[I].field_type[name].T, ScratchBufferLike].Element,
            MutAnyOrigin,
        ](unsafe_from_address=arena_base + self.scratch_off + off)

    @always_inline
    def binding[
        I: ScratchIsland, name: StringLiteral, o: ImmutOrigin,
    ](self, ctx: BindContext[o], plan: ScratchPlan) -> Binding[
        downcast[reflect[I].field_type[name].T, ScratchBufferLike].Element,
        o,
    ]:
        return ctx.view.bind(self.slot[I, name](ctx.view.bases[0], plan))
