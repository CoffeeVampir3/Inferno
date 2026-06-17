from std.memory import UnsafePointer
from std.benchmark import keep
from std.sys.info import simd_width_of

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstKernel, BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.helpers import (
    OutputPartitionedKernel, DispatchBuffer, RankBuffers,
    tile_dispatch, join_all, worker_range,
)
from kernels.reductions import dispatch_allreduce
from kernels.profiling import Profiler
from modeling.model_spec import BF16
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
    DEFAULT_SAMPLES,
)


comptime ALIGNMENT = 64
comptime WARMUP = 30
comptime SAMPLES = DEFAULT_SAMPLES
comptime MAX_ELEMS = 32 * 1024 * 1024
comptime BUF_ELEMS = 2816 * 4096

comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]


def arena_alloc[dtype: DType](
    mut arena: NumaArena[alignment=ALIGNMENT], count: Int,
) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    var ptr = arena.alloc[Scalar[dtype]](count)
    if not ptr:
        print("arena alloc failed for", count, "elements")
        return UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()
    return ptr.value()


def fill_pattern(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = BFloat16(Float32(i % 256))


@fieldwise_init
struct ReadSweepKernel(OutputPartitionedKernel):
    var src: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        comptime W = simd_width_of[DType.bfloat16]()
        var acc = SIMD[DType.float32, W](0)
        var pos = self.start
        while pos + W <= self.end:
            acc += (self.src + pos).load[width=W]().cast[DType.float32]()
            pos += W
        keep(acc.reduce_add())

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct WriteSweepKernel(OutputPartitionedKernel):
    var dst: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        comptime W = simd_width_of[DType.bfloat16]()
        var val = SIMD[DType.bfloat16, W](1)
        var pos = self.start
        while pos + W <= self.end:
            (self.dst + pos).store(val)
            pos += W

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct CopySweepKernel(OutputPartitionedKernel):
    var dst: BF16Ptr
    var src: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        comptime W = simd_width_of[DType.bfloat16]()
        var pos = self.start
        while pos + W <= self.end:
            (self.dst + pos).store((self.src + pos).load[width=W]())
            pos += W

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct NoopKernel(BurstKernel):
    var pad: Int
    def execute(mut self):
        pass


def timed_read[P: BurstThreadPool](
    mut pool: P, src: BF16Ptr, count: Int, mut samples: SampleBuffer,
):
    var buf = DispatchBuffer[ReadSweepKernel]()
    for _ in range(WARMUP):
        _ = tile_dispatch(buf, ReadSweepKernel(src, 0, 0), pool, count)
        pool.join()
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        _ = tile_dispatch(buf, ReadSweepKernel(src, 0, 0), pool, count)
        pool.join()
        var t1 = now_ns()
        var t_done = pool.last_worker_timestamp()
        samples.push(t_done - t0, t1 - t0)


def timed_write[P: BurstThreadPool](
    mut pool: P, dst: BF16Ptr, count: Int, mut samples: SampleBuffer,
):
    var buf = DispatchBuffer[WriteSweepKernel]()
    for _ in range(WARMUP):
        _ = tile_dispatch(buf, WriteSweepKernel(dst, 0, 0), pool, count)
        pool.join()
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        _ = tile_dispatch(buf, WriteSweepKernel(dst, 0, 0), pool, count)
        pool.join()
        var t1 = now_ns()
        var t_done = pool.last_worker_timestamp()
        samples.push(t_done - t0, t1 - t0)


def timed_copy[P: BurstThreadPool](
    mut pool: P, dst: BF16Ptr, src: BF16Ptr, count: Int,
    mut samples: SampleBuffer,
):
    var buf = DispatchBuffer[CopySweepKernel]()
    for _ in range(WARMUP):
        _ = tile_dispatch(buf, CopySweepKernel(dst, src, 0, 0), pool, count)
        pool.join()
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        _ = tile_dispatch(buf, CopySweepKernel(dst, src, 0, 0), pool, count)
        pool.join()
        var t1 = now_ns()
        var t_done = pool.last_worker_timestamp()
        samples.push(t_done - t0, t1 - t0)


def section_local_bw[P: BurstThreadPool, //](
    mut pools: List[P],
    src: List[BF16Ptr],
    dst: List[BF16Ptr],
    count: Int,
):
    var tp = len(pools)
    var mb = count * 2 // 1024 // 1024
    print(t"\n=== Local BW per node ({mb} MB bf16) ===")
    var samples = SampleBuffer(SAMPLES)
    for n in range(tp):
        timed_read(pools[n], src[n], count, samples)
        var rk = compute_stats(samples.kernel_ns, samples.n)
        var rw = compute_stats(samples.wall_ns, samples.n)
        print_row(String(t"n{n} read"), rk, rw, count * 2)

        timed_write(pools[n], dst[n], count, samples)
        var wk = compute_stats(samples.kernel_ns, samples.n)
        var ww = compute_stats(samples.wall_ns, samples.n)
        print_row(String(t"n{n} write"), wk, ww, count * 2)

        timed_copy(pools[n], dst[n], src[n], count, samples)
        var ck = compute_stats(samples.kernel_ns, samples.n)
        var cw = compute_stats(samples.wall_ns, samples.n)
        print_row(String(t"n{n} copy"), ck, cw, count * 4)


def section_remote_cached[P: BurstThreadPool, //](
    mut pools: List[P],
    src: List[BF16Ptr],
    count: Int,
):
    var tp = len(pools)
    if tp <= 1:
        return
    print("\n=== Cached remote read BW (data in reader L3 after warmup) ===")
    var samples = SampleBuffer(SAMPLES)
    for reader in range(tp):
        for owner in range(tp):
            if reader == owner:
                continue
            timed_read(pools[reader], src[owner], count, samples)
            var ks = compute_stats(samples.kernel_ns, samples.n)
            var ws = compute_stats(samples.wall_ns, samples.n)
            print_row(
                String(t"reader=n{reader} owner=n{owner}"),
                ks, ws, count * 2)


def section_remote_fresh[P: BurstThreadPool, //](
    mut pools: List[P],
    bufs: List[BF16Ptr],
    count: Int,
):
    var tp = len(pools)
    if tp <= 1:
        return
    print("\n=== Fresh remote read BW (owner writes, reader reads) ===")
    var samples = SampleBuffer(SAMPLES)
    for reader in range(tp):
        for owner in range(tp):
            if reader == owner:
                continue
            var rbuf = DispatchBuffer[ReadSweepKernel]()
            var wbuf = DispatchBuffer[WriteSweepKernel]()
            for _ in range(WARMUP):
                _ = tile_dispatch(wbuf, WriteSweepKernel(bufs[owner], 0, 0),
                    pools[owner], count)
                pools[owner].join()
                _ = tile_dispatch(rbuf, ReadSweepKernel(bufs[owner], 0, 0),
                    pools[reader], count)
                pools[reader].join()
            samples.clear()
            for _ in range(SAMPLES):
                _ = tile_dispatch(wbuf, WriteSweepKernel(bufs[owner], 0, 0),
                    pools[owner], count)
                pools[owner].join()
                var t0 = now_ns()
                _ = tile_dispatch(rbuf, ReadSweepKernel(bufs[owner], 0, 0),
                    pools[reader], count)
                pools[reader].join()
                var t1 = now_ns()
                var t_done = pools[reader].last_worker_timestamp()
                samples.push(t_done - t0, t1 - t0)
            var ks = compute_stats(samples.kernel_ns, samples.n)
            var ws = compute_stats(samples.wall_ns, samples.n)
            print_row(
                String(t"reader=n{reader} owner=n{owner}"),
                ks, ws, count * 2)


def section_contended[P: BurstThreadPool, //](
    mut pools: List[P],
    bufs: List[BF16Ptr],
    count: Int,
):
    var tp = len(pools)
    if tp <= 1:
        return
    print(t"\n=== Contended read BW ({tp} pools read one node, chunked) ===")
    var chunk = count // tp
    var samples = SampleBuffer(SAMPLES)
    for src_node in range(tp):
        for _ in range(WARMUP):
            var buf = DispatchBuffer[ReadSweepKernel]()
            for r in range(tp):
                _ = tile_dispatch(buf, ReadSweepKernel(bufs[src_node], 0, 0),
                    pools[r], chunk, chunk * r)
            join_all(pools)
        samples.clear()
        for _ in range(SAMPLES):
            var buf = DispatchBuffer[ReadSweepKernel]()
            var t0 = now_ns()
            for r in range(tp):
                _ = tile_dispatch(buf, ReadSweepKernel(bufs[src_node], 0, 0),
                    pools[r], chunk, chunk * r)
            join_all(pools)
            var t1 = now_ns()
            var t_done = max_last_ts(pools)
            samples.push(t_done - t0, t1 - t0)
        var ks = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        print_row(
            String(t"src=n{src_node} ({tp} readers)"),
            ks, ws, count * 2)


def section_dispatch[P: BurstThreadPool, //](
    mut pools: List[P],
):
    var tp = len(pools)
    print("\n=== Dispatch + join overhead ===")
    var samples = SampleBuffer(SAMPLES)
    for _ in range(WARMUP):
        var buf = DispatchBuffer[NoopKernel]()
        for r in range(tp):
            buf.slot()[] = NoopKernel(0)
            buf.dispatch(pools[r])
        join_all(pools)
    samples.clear()
    for _ in range(SAMPLES):
        var buf = DispatchBuffer[NoopKernel]()
        var t0 = now_ns()
        for r in range(tp):
            buf.slot()[] = NoopKernel(0)
            buf.dispatch(pools[r])
        join_all(pools)
        var t1 = now_ns()
        var t_done = max_last_ts(pools)
        samples.push(t_done - t0, t1 - t0)
    var ks = compute_stats(samples.kernel_ns, samples.n)
    var ws = compute_stats(samples.wall_ns, samples.n)
    print_row(String(t"noop dispatch+join ({tp} pools x 1 job)"), ks, ws, 0)

    for _ in range(WARMUP):
        var buf = DispatchBuffer[NoopKernel]()
        for r in range(tp):
            var cap = pools[r].get_capacity()
            for _ in range(cap):
                buf.slot()[] = NoopKernel(0)
            buf.dispatch(pools[r])
        join_all(pools)
    samples.clear()
    for _ in range(SAMPLES):
        var buf = DispatchBuffer[NoopKernel]()
        var t0 = now_ns()
        for r in range(tp):
            var cap = pools[r].get_capacity()
            for _ in range(cap):
                buf.slot()[] = NoopKernel(0)
            buf.dispatch(pools[r])
        join_all(pools)
        var t1 = now_ns()
        var t_done = max_last_ts(pools)
        samples.push(t_done - t0, t1 - t0)
    var ks2 = compute_stats(samples.kernel_ns, samples.n)
    var ws2 = compute_stats(samples.wall_ns, samples.n)
    print_row(
        String(t"noop dispatch+join ({tp} pools x all workers)"), ks2, ws2, 0
    )


def section_worker_scaling[P: BurstThreadPool, //](
    mut pools: List[P],
    src: List[BF16Ptr],
    count: Int,
):
    var cap = pools[0].get_capacity()
    var mb = count * 2 // 1024 // 1024
    print(t"\n=== Worker scaling on node 0 (capacity={cap}, {mb} MB) ===")

    var samples = SampleBuffer(SAMPLES)
    var n = 1
    while n <= cap:
        var noop_buf = DispatchBuffer[NoopKernel]()
        for _ in range(WARMUP):
            for _ in range(n):
                noop_buf.slot()[] = NoopKernel(0)
            noop_buf.dispatch(pools[0])
            pools[0].join()
        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            for _ in range(n):
                noop_buf.slot()[] = NoopKernel(0)
            noop_buf.dispatch(pools[0])
            pools[0].join()
            var t1 = now_ns()
            var t_done = pools[0].last_worker_timestamp()
            samples.push(t_done - t0, t1 - t0)
        var dk = compute_stats(samples.kernel_ns, samples.n)
        var dw = compute_stats(samples.wall_ns, samples.n)
        print_row(String(t"workers={n} noop dispatch"), dk, dw, 0)

        var read_buf = DispatchBuffer[ReadSweepKernel]()
        for _ in range(WARMUP):
            for w in range(n):
                var wr = worker_range(count, n, w)
                read_buf.slot()[] = ReadSweepKernel(src[0], wr[0], wr[1])
            read_buf.dispatch(pools[0])
            pools[0].join()
        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            for w in range(n):
                var wr = worker_range(count, n, w)
                read_buf.slot()[] = ReadSweepKernel(src[0], wr[0], wr[1])
            read_buf.dispatch(pools[0])
            pools[0].join()
            var t1 = now_ns()
            var t_done = pools[0].last_worker_timestamp()
            samples.push(t_done - t0, t1 - t0)
        var rk = compute_stats(samples.kernel_ns, samples.n)
        var rw = compute_stats(samples.wall_ns, samples.n)
        print_row(String(t"workers={n} read"), rk, rw, count * 2)

        if n < 4:
            n *= 2
        elif n < cap:
            n = min(n * 2, cap)
        else:
            break


def section_sweep[P: BurstThreadPool, //](
    mut pools: List[P],
    src: List[BF16Ptr],
    dst: List[BF16Ptr],
):
    var tp = len(pools)
    print(t"\n=== Allreduce bf16 sweep (tp={tp}) ===")

    comptime NUM_SIZES = 18
    var sizes = InlineArray[Int, NUM_SIZES](fill=0)
    sizes[0] = 512
    sizes[1] = 1024
    sizes[2] = 2048
    sizes[3] = 4096
    sizes[4] = 8192
    sizes[5] = 16384
    sizes[6] = 32768
    sizes[7] = 65536
    sizes[8] = 131072
    sizes[9] = 262144
    sizes[10] = 524288
    sizes[11] = 1048576
    sizes[12] = 2097152
    sizes[13] = 4194304
    sizes[14] = 8388608
    sizes[15] = BUF_ELEMS
    sizes[16] = 16777216
    sizes[17] = MAX_ELEMS

    var samples = SampleBuffer(SAMPLES)
    var prof = Profiler[False]()

    for s in range(NUM_SIZES):
        var count = sizes[s]
        var rb = RankBuffers[DType.bfloat16, ImmutAnyOrigin](count=count)
        var db = RankBuffers[DType.bfloat16, MutAnyOrigin](count=count)
        for r in range(tp):
            rb.add(src[r].as_immutable())
            db.add(dst[r])

        for _ in range(WARMUP):
            dispatch_allreduce[BF16](rb, db, pools, prof)

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            dispatch_allreduce[BF16](rb, db, pools, prof)
            var t1 = now_ns()
            var t_done = max_last_ts(pools)
            samples.push(t_done - t0, t1 - t0)
        keep(db.ptrs[0][0])

        var total_bytes = count * 2 * tp * 2
        var sz_kb = count * 2 // 1024
        var label: String
        if sz_kb < 1024:
            label = String(t"allreduce {sz_kb}KB")
        else:
            var sz_mb = sz_kb // 1024
            label = String(t"allreduce {sz_mb}MB")
        var ks = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        print_row(label, ks, ws, total_bytes)


def run_all[P: BurstThreadPool, //](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    var src = List[BF16Ptr](capacity=tp)
    var dst = List[BF16Ptr](capacity=tp)
    for r in range(tp):
        src.append(arena_alloc[DType.bfloat16](arenas[r], MAX_ELEMS))
        dst.append(arena_alloc[DType.bfloat16](arenas[r], MAX_ELEMS))
        fill_pattern(src[r], MAX_ELEMS)

    section_local_bw(pools, src, dst, BUF_ELEMS)
    section_remote_cached(pools, src, BUF_ELEMS)
    section_remote_fresh(pools, dst, BUF_ELEMS)
    section_contended(pools, src, BUF_ELEMS)
    section_dispatch(pools)
    section_worker_scaling(pools, src, BUF_ELEMS)
    section_sweep(pools, src, dst)


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("Primitives benchmark")
    var iso = len(topo.isolated_cpus)
    print(t"{tp} NUMA node(s), {iso} isolated cpus\n")

    comptime ARENA_BYTES = 512 * 1024 * 1024
    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_primitives_tp[
        P: BurstThreadPool, //,
    ](var selected_pools: List[P]):
        run_all(selected_pools, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_primitives_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
