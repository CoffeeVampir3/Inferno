from kernels.helpers import Binding
from kernels.gather import dispatch_gather_rows
from kernels.flash_sample import SamplingParams
from kernels.profiling import Profiler
from threading.threading_traits import BurstThreadPool
from continuous_batching.schedule import Schedule


@fieldwise_init
struct ArenaLayout(Copyable, ImplicitlyCopyable):
    """Common arena metadata shared by every model topology.

    All fields are arena-relative; the layout holds no absolute addresses.
    The sizing fields describe the layout the loader/runtime expects:
    distributed (weights) + state (activations, KV cache, rope, scratch)
    form the main arena. `host_bytes` is the allocation ceiling for that
    arena plus any optional rank-targeted tensors a model topology chooses
    to append.
    """
    var distributed_bytes: Int
    var state_bytes: Int
    var host_bytes: Int
    var scratch_off: Int

    def host_arena_bytes(self) -> Int:
        return self.host_bytes


@fieldwise_init
struct Repeated[T: ImplicitlyCopyable & ImplicitlyDeletable](Copyable, ImplicitlyCopyable):
    var proto: Self.T
    var off: Int
    var stride: Int
    var count: Int

    @always_inline
    def base(self, idx: Int) -> Int:
        return self.off + idx * self.stride


def pack_slot_starts(read schedule: Schedule) -> List[Int]:
    var starts = List[Int](capacity=len(schedule.slots))
    var cursor = 0
    for i in range(len(schedule.slots)):
        starts.append(cursor)
        cursor += schedule.slots[i].n_tokens
    debug_assert(
        cursor == len(schedule.tokens),
        "slot token counts must sum to len(tokens)",
    )
    return starts^


@fieldwise_init
struct EmitPlan(Movable):
    var slots: List[Int]
    var rows: List[Int]
    var contiguous: Bool

    @always_inline
    def count(self) -> Int:
        return len(self.slots)


def collect_emit_plan(
    read schedule: Schedule, read buf_starts: List[Int],
) -> EmitPlan:
    var slots = List[Int]()
    var rows = List[Int]()
    for i in range(len(schedule.slots)):
        if schedule.slots[i].emit:
            slots.append(i)
            rows.append(buf_starts[i] + schedule.slots[i].n_tokens - 1)
    var contiguous = True
    for j in range(len(rows)):
        if rows[j] != rows[0] + j:
            contiguous = False
    return EmitPlan(slots^, rows^, contiguous)


def stage_sampling_inputs[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    hidden: Int,
](
    read plan: EmitPlan,
    read schedule: Schedule,
    x_main: Binding[BFloat16, o],
    head_x: Binding[BFloat16, o],
    emit_rows: Binding[Int32, o],
    sample_params: Binding[SamplingParams, o],
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
) -> Binding[BFloat16, o]:
    var degree = x_main.degree()
    var num_emit = plan.count()
    for r in range(degree):
        for j in range(num_emit):
            (sample_params[r] + j)[] = schedule.slots[plan.slots[j]].sampling
    if plan.contiguous:
        return x_main.shifted(plan.rows[0] * hidden)
    for r in range(degree):
        for j in range(num_emit):
            (emit_rows[r] + j)[] = Int32(plan.rows[j])
    dispatch_gather_rows[cols=hidden](
        x_main, head_x, emit_rows, num_emit, pools, prof)
    return head_x
