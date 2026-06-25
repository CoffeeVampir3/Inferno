from std.memory import UnsafePointer, memcpy

from threading.threading_traits import BurstThreadPool
from .helpers import RangePartitionedKernel, fanout_dispatch
from .profiling import Profiler


comptime COPY_SPLIT_ALIGNMENT = 64


@fieldwise_init
struct CopyJob(Copyable, Movable, ImplicitlyCopyable):
    var src_off: Int
    var dst_off: Int
    var byte_count: Int


@always_inline
def align_split(offset: Int, total: Int) -> Int:
    var aligned = ((offset + COPY_SPLIT_ALIGNMENT - 1)
                   // COPY_SPLIT_ALIGNMENT * COPY_SPLIT_ALIGNMENT)
    return min(aligned, total)


@fieldwise_init
struct PageCopyKernel(RangePartitionedKernel):
    var jobs: UnsafePointer[CopyJob, MutUntrackedOrigin]
    var bounds: UnsafePointer[Int, MutUntrackedOrigin]
    var num_jobs: Int
    var arena_base: Int
    var byte_lo: Int
    var byte_hi: Int

    def execute(mut self):
        var total = self.bounds[self.num_jobs]
        var lo = align_split(self.byte_lo, total)
        var hi = align_split(self.byte_hi, total)
        if lo >= hi:
            return
        var j = 0
        while self.bounds[j + 1] <= lo:
            j += 1
        var cursor = lo
        while cursor < hi:
            var job = self.jobs[j]
            var seg_end = min(self.bounds[j + 1], hi)
            var off = cursor - self.bounds[j]
            memcpy(
                dest=UnsafePointer[UInt8, MutUntrackedOrigin](
                    unsafe_from_address=self.arena_base + job.dst_off + off),
                src=UnsafePointer[UInt8, MutUntrackedOrigin](
                    unsafe_from_address=self.arena_base + job.src_off + off),
                count=seg_end - cursor)
            cursor = seg_end
            j += 1

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.byte_lo = start
        self.byte_hi = end


def dispatch_copy_jobs[
    P: BurstThreadPool, Profile: Bool, N: Int, //,
    max_worker_count: Int = 128,
](
    read jobs: List[CopyJob],
    read arena_bases: List[Int],
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var total = len(jobs)
    if total == 0:
        return
    var bounds = List[Int](capacity=total + 1)
    var total_bytes = 0
    bounds.append(0)
    for i in range(total):
        total_bytes += jobs[i].byte_count
        bounds.append(total_bytes)
    if total_bytes == 0:
        return
    var job_ptr = jobs.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var bounds_ptr = bounds.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var base_ptr = arena_bases.unsafe_ptr()

    @parameter
    def make(r: Int) -> PageCopyKernel:
        return PageCopyKernel(job_ptr, bounds_ptr, total, base_ptr[r], 0, 0)

    fanout_dispatch[
        make, max_worker_count=max_worker_count, label="page_copy",
    ](pools, prof, total_bytes, total_bytes)
