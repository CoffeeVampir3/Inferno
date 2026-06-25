from std.memory import UnsafePointer

from threading.threading_traits import BurstThreadPool
from .helpers import BF16Ptr, BW, RangePartitionedKernel, fanout_dispatch, Binding
from .profiling import Profiler


@fieldwise_init
struct GatherRowsKernel[cols: Int](RangePartitionedKernel):
    var src: BF16Ptr
    var dst: BF16Ptr
    var rows: UnsafePointer[Int32, MutUntrackedOrigin]
    var start: Int
    var end: Int

    def execute(mut self):
        comptime assert Self.cols % BW == 0, "gather cols must be a multiple of bf16 width"
        for j in range(self.start, self.end):
            var s = self.src + Int(self.rows[j]) * Self.cols
            var d = self.dst + j * Self.cols
            for c in range(0, Self.cols, BW):
                (d + c).store((s + c).load[width=BW]())

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_gather_rows[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    cols: Int,
    max_worker_count: Int = 128,
](
    src: Binding[BFloat16, o],
    dst: Binding[BFloat16, o],
    rows: Binding[Int32, o],
    num_rows: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if num_rows <= 0:
        return
    comptime K = GatherRowsKernel[cols]
    var sr = src
    var ds = dst
    var rw = rows

    @parameter
    def make(r: Int) -> K:
        return K(sr[r], ds[r], rw[r], 0, 0)

    fanout_dispatch[
        make, max_worker_count=max_worker_count, label="gather_emit_rows",
    ](pools, prof, num_rows, num_rows * cols * 2)
