from std.math import min
from std.memory import Span

from numa import NumaArena
from threading.threading_traits import BurstThreadPool
from kernels.helpers import (
    WorkerRangePartitionedKernel, fanout_dispatch, saturate_workers,
)
from kernels.profiling import Profiler

from butterquant.vnni import (
    L2_TARGET, PtrU8, PtrF32, pack_and_colsum_vnni,
)


comptime PACK_SCRATCH_ALIGN = 64


@fieldwise_init
struct PackColsumTask(Copyable, ImplicitlyCopyable):
    var weight_off: Int
    var colsum_off: Int
    var rows: Int
    var cols: Int
    var block_cols: Int
    var colsum_row_major: Bool


@fieldwise_init
struct PackColsumKernel[tasks_origin: ImmutOrigin](WorkerRangePartitionedKernel):
    var tasks: Span[PackColsumTask, Self.tasks_origin]
    var arena_base: Int
    var scratch_base: Int
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var scratch = PtrU8(
            unsafe_from_address=self.scratch_base + self.worker_id * L2_TARGET)
        for i in range(self.start, self.end):
            var t = self.tasks[i]
            var weight = PtrU8(
                unsafe_from_address=self.arena_base + t.weight_off)
            var colsum = PtrF32(
                unsafe_from_address=self.arena_base + t.colsum_off)
            pack_and_colsum_vnni(
                weight, weight, scratch,
                t.rows, t.cols, t.block_cols,
                colsum, t.colsum_row_major)

    @always_inline
    def install_worker_range(mut self, worker_id: Int, start: Int, end: Int):
        self.worker_id = worker_id
        self.start = start
        self.end = end


def dispatch_pack_colsum[
    P: BurstThreadPool, Profile: Bool, N: Int, //,
    max_worker_count: Int = 128,
](
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
    arena_bases: List[Int],
    nodes: List[Int],
    tasks: List[PackColsumTask],
):
    var num_tasks = len(tasks)
    if num_tasks == 0:
        return
    var degree = len(pools)
    var view = Span(tasks)
    comptime K = PackColsumKernel[origin_of(tasks)]

    var scratch = List[NumaArena[alignment=PACK_SCRATCH_ALIGN]](capacity=degree)
    for r in range(degree):
        var cap = min(max_worker_count, pools[r].get_capacity())
        scratch.append(
            NumaArena[alignment=PACK_SCRATCH_ALIGN](nodes[r], cap * L2_TARGET))
        debug_assert(scratch[r].__bool__(),
            "dispatch_pack_colsum: scratch allocation failed")

    @parameter
    def proto_for(r: Int) -> K:
        return K(
            tasks=view,
            arena_base=arena_bases[r],
            scratch_base=Int(scratch[r].base.value()),
            worker_id=0, start=0, end=0)

    fanout_dispatch[
        proto_for,
        max_worker_count=max_worker_count,
        worker_policy=saturate_workers,
        label="pack_colsum",
    ](pools, prof, num_tasks, num_tasks * L2_TARGET)

    _ = scratch^
