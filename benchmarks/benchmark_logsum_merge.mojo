from std.collections import InlineArray
from std.memory import Span, UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.logsum_merge import dispatch_merge_flash_partials
from kernels.attention_ops import flash_partial_stride
from kernels.helpers import (
    OutputPartitionedKernel, DispatchBuffer, tile_dispatch,
    Binding, RankView,
)
from kernels.profiling import Profiler
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
    DEFAULT_SAMPLES,
)


comptime ALIGNMENT = 64
comptime WARMUP = 30
comptime SAMPLES = DEFAULT_SAMPLES
comptime MAX_SOURCES = 128
comptime FORCE_INLINE = 1 << 30

comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]


@fieldwise_init
struct NoopKernel(OutputPartitionedKernel):
    var dst: F32Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        self.dst[self.start] = Float32(0)

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.start = start
        self.end = end


def warm_pool[P: BurstThreadPool](scratch: F32Ptr, mut pool: P):
    var buf = DispatchBuffer[NoopKernel]()
    _ = tile_dispatch(buf, NoopKernel(scratch, 0, 0), pool, pool.get_capacity())
    pool.join()


def arena_alloc[dtype: DType](
    mut arena: NumaArena[alignment=ALIGNMENT], count: Int,
) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    var ptr = arena.alloc[Scalar[dtype]](count)
    if not ptr:
        print("arena alloc failed")
        return UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()
    return ptr.value()


def arena_bases(
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
) -> List[Int]:
    var bases = List[Int](capacity=len(arenas))
    for r in range(len(arenas)):
        bases.append(Int(arenas[r].base.value()))
    return bases^


def arena_alloc_all[dtype: DType](
    mut arenas: List[NumaArena[alignment=ALIGNMENT]], count: Int,
) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    var first = UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()
    for r in range(len(arenas)):
        var ptr = arena_alloc[dtype](arenas[r], count)
        if r == 0:
            first = ptr
    return first


def fill_partials[head_dim: Int, num_q: Int](
    buf: F32Ptr, stride: Int, num_sources: Int,
):
    comptime m_off = num_q * head_dim
    comptime l_off = m_off + num_q
    for s in range(num_sources):
        var sp = buf + s * stride
        for i in range(num_q * head_dim):
            sp[i] = Float32((i % 127) - 63) * 0.001
        for h in range(num_q):
            (sp + m_off + h)[] = Float32(s % 5) * 0.5 - 1.0
            (sp + l_off + h)[] = Float32(1.0)


def fill_partials_all[head_dim: Int, num_q: Int, o: ImmutOrigin](
    ptrs: Binding[Float32, o],
    stride: Int, num_sources: Int,
):
    for r in range(ptrs.degree()):
        fill_partials[head_dim, num_q](ptrs[r], stride, num_sources)


def source_counts(tp: Int, num_sources: Int) -> List[Int]:
    var out = List[Int](capacity=tp)
    for _ in range(tp):
        out.append(num_sources)
    return out^


def run_config[
    P: BurstThreadPool, //, head_dim: Int, num_q: Int,
](
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
    mut pools: List[P],
):
    var tp = len(pools)
    comptime stride = flash_partial_stride(num_q, head_dim)
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))
    var partials = arena_alloc_all[DType.float32](arenas, MAX_SOURCES * stride)
    var output = arena_alloc_all[DType.bfloat16](arenas, num_q * head_dim)
    var scratch = arena_alloc[DType.float32](arenas[0], pools[0].get_capacity())
    var partial_bind = view.bind(partials)
    var output_bind = view.bind(output)

    fill_partials_all[head_dim, num_q](partial_bind, stride, MAX_SOURCES)
    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    var pool_cap = pools[0].get_capacity()
    print(t"\n=== head_dim={head_dim} num_q={num_q} pool_capacity={pool_cap} ===")

    var counts = InlineArray[Int, 7](fill=0)
    counts[0] = 2; counts[1] = 4; counts[2] = 8
    counts[3] = 16; counts[4] = 32; counts[5] = 64
    counts[6] = 128

    var samples = SampleBuffer(SAMPLES)
    var prof = Profiler[False]()

    for s in range(7):
        var ns = counts[s]
        var data_bytes = ns * (head_dim + 2) * 4 * num_q

        warm_pool(scratch, pools[0])
        for _ in range(WARMUP):
            dispatch_merge_flash_partials[head_dim](
                output_bind, partial_bind,
                source_counts(tp, ns), num_q, stride, pools, prof,
                inline_max_bytes=FORCE_INLINE)
            keep(output[0])

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            dispatch_merge_flash_partials[head_dim](
                output_bind, partial_bind,
                source_counts(tp, ns), num_q, stride, pools, prof,
                inline_max_bytes=FORCE_INLINE)
            var t1 = now_ns()
            samples.push(t1 - t0, t1 - t0)
        keep(output[0])

        var iks = compute_stats(samples.kernel_ns, samples.n)
        var iws = compute_stats(samples.wall_ns, samples.n)
        print_row(String(t"sources={ns} inline"), iks, iws, data_bytes)

        warm_pool(scratch, pools[0])
        for _ in range(WARMUP):
            dispatch_merge_flash_partials[head_dim](
                output_bind, partial_bind,
                source_counts(tp, ns), num_q, stride, pools, prof,
                inline_max_bytes=0)
            keep(output[0])

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            dispatch_merge_flash_partials[head_dim](
                output_bind, partial_bind,
                source_counts(tp, ns), num_q, stride, pools, prof,
                inline_max_bytes=0)
            var t1 = now_ns()
            var t_done = max_last_ts(pools)
            samples.push(t_done - t0, t1 - t0)
        keep(output[0])

        var dks = compute_stats(samples.kernel_ns, samples.n)
        var dws = compute_stats(samples.wall_ns, samples.n)
        print_row(String(t"sources={ns} dispatched"), dks, dws, data_bytes)


def run_all[P: BurstThreadPool, //](
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
    mut pools: List[P],
):
    run_config[head_dim=256, num_q=8](arenas, pools)
    run_config[head_dim=512, num_q=16](arenas, pools)


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("logsum_merge worker count sweep")
    var iso = len(topo.isolated_cpus)
    print(t"{tp} NUMA node(s), {iso} isolated cpus\n")

    comptime ARENA_BYTES = 128 * 1024 * 1024
    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_logsum_merge_tp[
        P: BurstThreadPool, //,
    ](var selected_pools: List[P]):
        run_all(arenas, selected_pools)

    with_topological_rank_dispatch[
        dispatch=dispatch_logsum_merge_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
