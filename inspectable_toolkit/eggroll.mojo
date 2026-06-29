from std.memory import UnsafePointer, Span

from numa import NumaArena, NumaTopology
from kernels.helpers import RankView, Binding
from simd_math.sampling_rng import splitmix64
from modeling.model_spec import BF16, DEFAULT_ALIGNMENT
from modeling.slot import Slot, SlotGroup, BindContext, stamp_offsets
from modeling.modeling_common import Repeated
from modeling.gemma4_common import Gemma4BaseConfig, LAYER_SCHEDULE, LayerKind
from modeling.gemma4_topology import (
    Gemma4Recipes, KVSlotGroup, Gemma4Shapes, Gemma4TailShapes, Gemma4Layout,
)
from continuous_batching.schedule import Schedule


comptime C = Gemma4BaseConfig

comptime FACTOR_A = 0
comptime FACTOR_B = 1


struct DiffSlidingRefs[R: Gemma4Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4Shapes[Self.R.FFN_BLOCK]
    var q_proj:          Slot[BF16, Self.S.SlidingQ]
    var k_proj:          Slot[BF16, Self.S.SlidingKV]
    var v_proj:          Slot[BF16, Self.S.SlidingKV]
    var o_proj:          Slot[BF16, Self.S.SlidingO]
    var gate_proj:       Slot[BF16, Self.S.GateUp]
    var up_proj:         Slot[BF16, Self.S.GateUp]
    var down_proj:       Slot[BF16, Self.S.Down]
    var router_proj:     Slot[BF16, Self.S.RouterProj]
    var experts_gate_up: Slot[BF16, Self.S.ExpertsGateUp]
    var experts_down:    Slot[BF16, Self.S.ExpertsDown]


struct DiffFullRefs[R: Gemma4Recipes](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4Shapes[Self.R.FFN_BLOCK]
    var q_proj:          Slot[BF16, Self.S.FullQ]
    var k_proj:          Slot[BF16, Self.S.FullK]
    var o_proj:          Slot[BF16, Self.S.FullO]
    var gate_proj:       Slot[BF16, Self.S.GateUp]
    var up_proj:         Slot[BF16, Self.S.GateUp]
    var down_proj:       Slot[BF16, Self.S.Down]
    var router_proj:     Slot[BF16, Self.S.RouterProj]
    var experts_gate_up: Slot[BF16, Self.S.ExpertsGateUp]
    var experts_down:    Slot[BF16, Self.S.ExpertsDown]


struct DiffTailRefs(Copyable, ImplicitlyCopyable, SlotGroup):
    var embed: Slot[BF16, Gemma4TailShapes.Embed]


@always_inline
def rank_view(read bases: List[Int]) -> RankView[ImmutAnyOrigin]:
    return RankView[ImmutAnyOrigin](
        Span[Int, ImmutAnyOrigin](
            ptr=UnsafePointer[Int, ImmutAnyOrigin](
                unsafe_from_address=Int(bases.unsafe_ptr())),
            length=len(bases)))


struct EggrollDiff[R: Gemma4Recipes](Movable):
    var arenas: List[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var bases: List[Int]
    var sliding: Repeated[DiffSlidingRefs[Self.R]]
    var full: Repeated[DiffFullRefs[Self.R]]
    var tail_proto: DiffTailRefs
    var tail_off: Int
    var degree: Int

    def __init__(out self, topo: NumaTopology, degree: Int):
        self.degree = degree
        var ps_proto = DiffSlidingRefs[Self.R]()
        var ps_stride = stamp_offsets(ps_proto, degree)
        var pf_proto = DiffFullRefs[Self.R]()
        var pf_stride = stamp_offsets(pf_proto, degree)
        var pt_proto = DiffTailRefs()
        var pt_stride = stamp_offsets(pt_proto, degree)
        var ps_off = 0
        var pf_off = ps_off + C.NUM_SLIDING_LAYERS * ps_stride
        var pt_off = pf_off + C.NUM_FULL_LAYERS * pf_stride
        var total = pt_off + pt_stride
        self.sliding = Repeated[DiffSlidingRefs[Self.R]](
            ps_proto, ps_off, ps_stride, C.NUM_SLIDING_LAYERS)
        self.full = Repeated[DiffFullRefs[Self.R]](
            pf_proto, pf_off, pf_stride, C.NUM_FULL_LAYERS)
        self.tail_proto = pt_proto
        self.tail_off = pt_off
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
    def context(self) -> BindContext[ImmutAnyOrigin]:
        return BindContext(rank_view(self.bases), 0)

    @always_inline
    def sliding_layer(self, local_idx: Int) -> BindContext[ImmutAnyOrigin]:
        return self.context().with_layer(self.sliding.base(local_idx))

    @always_inline
    def full_layer(self, local_idx: Int) -> BindContext[ImmutAnyOrigin]:
        return self.context().with_layer(self.full.base(local_idx))

    @always_inline
    def tail(self) -> BindContext[ImmutAnyOrigin]:
        return self.context().with_layer(self.tail_off)


def build_diff[
    R: Gemma4Recipes, SKV: KVSlotGroup, FKV: KVSlotGroup,
    msl: Int, sv: Int, mr: Int, //,
](
    live_layout: Gemma4Layout[R, SKV, FKV, msl, sv, mr],
    topo: NumaTopology,
    degree: Int,
) -> EggrollDiff[R]:
    return EggrollDiff[R](topo, degree)


struct EggrollWorkspace(Movable):
    var arenas: List[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var bases: List[Int]
    var rows_cap: Int
    var cols_cap: Int
    var rank: Int
    var worker_tile: Int

    def __init__(
        out self, topo: NumaTopology, degree: Int,
        rows_cap: Int, cols_cap: Int, rank: Int, worker_tile: Int = 1,
    ):
        self.rows_cap = rows_cap
        self.cols_cap = cols_cap
        self.rank = rank
        self.worker_tile = worker_tile
        var count = (rows_cap + cols_cap) * rank * worker_tile
        var size = count * 4
        self.arenas = List[NumaArena[alignment=DEFAULT_ALIGNMENT]]()
        self.bases = List[Int]()
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
    def a_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.bases[0])

    @always_inline
    def b_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.bases[0]
            + self.rows_cap * self.rank * self.worker_tile * 4)


@always_inline
def eggroll_worker_seed(base_seed: UInt64, step: Int, worker: Int) -> UInt64:
    return splitmix64(
        base_seed ^ (UInt64(step) << 40) ^ (UInt64(worker) << 8))


@always_inline
def eggroll_factor_counter(
    worker_seed: UInt64, matrix_id: Int, side: Int, index: Int,
) -> UInt64:
    return splitmix64(
        worker_seed
        ^ (UInt64(matrix_id) << 33)
        ^ (UInt64(side) << 32)
        ^ UInt64(index))


struct EggrollState(Movable):
    var armed: Bool
    var pop_size: Int
    var rank: Int
    var sigma: Float32
    var alpha: Float32
    var decay: Float32
    var base_seed: UInt64
    var step: Int
    var max_slots: Int
    var slot_worker: List[Int]
    var last_num_slots: Int

    def __init__(out self, max_slots: Int):
        self.armed = False
        self.pop_size = 0
        self.rank = 1
        self.sigma = Float32(0)
        self.alpha = Float32(0)
        self.decay = Float32(0)
        self.base_seed = UInt64(0)
        self.step = 0
        self.max_slots = max_slots
        self.slot_worker = List[Int](length=max_slots, fill=0)
        self.last_num_slots = 0

    def arm(
        mut self, pop_size: Int, rank: Int, sigma: Float32, alpha: Float32,
        decay: Float32, base_seed: UInt64,
    ):
        self.pop_size = pop_size
        self.rank = rank
        self.sigma = sigma
        self.alpha = alpha
        self.decay = decay
        self.base_seed = base_seed
        self.armed = True

    def disarm(mut self):
        self.armed = False

    def advance(mut self):
        self.step += 1

    def assign_workers(mut self, read workers: List[Int], num_slots: Int):
        for s in range(num_slots):
            self.slot_worker[s] = workers[s]
        self.last_num_slots = num_slots

    @always_inline
    def worker_for(self, slot: Int) -> Int:
        return self.slot_worker[slot]

    @always_inline
    def worker_seed(self, worker: Int) -> UInt64:
        return eggroll_worker_seed(self.base_seed, self.step, worker)


trait Evolvable:
    comptime EGGROLL_WORKERS: Int
    def arm_eggroll(
        mut self, pop_size: Int, rank: Int, sigma: Float32, alpha: Float32,
        decay: Float32, base_seed: UInt64,
    ): ...
    def disarm_eggroll(mut self): ...
