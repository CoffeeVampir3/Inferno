from std.algorithm import vectorize
from std.memory import UnsafePointer, memcpy
from std.sys.info import simd_width_of

from modeling.model_spec import Encoding
from threading.threading_traits import BurstThreadPool
from .helpers import (
    RangePartitionedKernel, RankBuffers, DispatchBuffer, Binding,
    join_all, tile_dispatch, recommended_workers,
)
from .profiling import Profiler, DispatchSpan


comptime DEFAULT_INLINE_BYTES = 16384
comptime SrcPtr[dtype: DType] = UnsafePointer[Scalar[dtype], ImmutAnyOrigin]
comptime DstPtr[dtype: DType] = UnsafePointer[Scalar[dtype], MutAnyOrigin]


@fieldwise_init
struct ReduceConfig[
    E: Encoding, buffers_origin: ImmutOrigin,
](TrivialRegisterPassable):
    var src: Span[SrcPtr[Self.E.DTYPE], Self.buffers_origin]
    var dst: Span[DstPtr[Self.E.DTYPE], Self.buffers_origin]
    var chunk: Int
    var rem: Int


@fieldwise_init
struct ReduceStoreKernel[
    E: Encoding, buffers_origin: ImmutOrigin,
    Accum: DType = DType.float32,
](RangePartitionedKernel):
    var config: ReduceConfig[Self.E, Self.buffers_origin]
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        reduce_store_range[
            Self.E, Self.buffers_origin, Self.Accum,
        ](self.config, self.rank, self.start, self.end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct GatherKernel[
    E: Encoding, buffers_origin: ImmutOrigin,
](RangePartitionedKernel):
    var config: ReduceConfig[Self.E, Self.buffers_origin]
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        gather_chunks[Self.E, Self.buffers_origin](
            self.config, self.rank, self.start, self.end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


@always_inline
def rank_chunk_count(chunk: Int, rem: Int, rank: Int, tp: Int) -> Int:
    if rank == tp - 1:
        return chunk + rem
    return chunk


@always_inline
def reduce_store_range[
    E: Encoding, buffers_origin: ImmutOrigin,
    Accum: DType = DType.float32,
](
    config: ReduceConfig[E, buffers_origin],
    out_rank: Int, start: Int, end: Int,
):
    var tp = len(config.src)
    var srcs = config.src.unsafe_ptr()
    var dst = config.dst[out_rank]

    def step[width: Int](idx: Int) {read}:
        var pos = start + idx
        var acc = (srcs[0] + pos).load[width=width]().cast[Accum]()
        for r in range(1, tp):
            acc += (srcs[r] + pos).load[width=width]().cast[Accum]()
        (dst + pos).store(acc.cast[E.DTYPE]())

    vectorize[simd_width_of[Accum]()](end - start, step)


def gather_chunks[
    E: Encoding, buffers_origin: ImmutOrigin,
](
    config: ReduceConfig[E, buffers_origin],
    dst_rank: Int, start: Int, end: Int,
):
    var tp = len(config.src)
    for src_rank in range(tp):
        if src_rank == dst_rank:
            continue
        var src_start = config.chunk * src_rank
        var src_count = rank_chunk_count(config.chunk, config.rem, src_rank, tp)
        var lo = max(start, src_start)
        var hi = min(end, src_start + src_count)
        if lo < hi:
            memcpy(dest=config.dst[dst_rank] + lo,
                   src=config.dst[src_rank] + lo, count=hi - lo)


def dispatch_allreduce[
    P: BurstThreadPool, Profile: Bool, N: Int, //,
    E: Encoding, Accum: DType = DType.float32,
    max_worker_count: Int = 128,
](
    src: RankBuffers[E.DTYPE, ImmutAnyOrigin],
    output: RankBuffers[E.DTYPE, MutAnyOrigin],
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
    inline_max_bytes: Int = DEFAULT_INLINE_BYTES,
):
    if src.count <= 0:
        return

    var tp = len(pools)
    var chunk = src.count // tp
    comptime buffers_origin = origin_of(src.ptrs, output.ptrs)
    var cfg = ReduceConfig[E, buffers_origin](
        Span[SrcPtr[E.DTYPE], buffers_origin](src.ptrs),
        Span[DstPtr[E.DTYPE], buffers_origin](output.ptrs),
        chunk, src.count - chunk * tp,
    )

    if src.count * E.ELEMENT_BYTES <= inline_max_bytes or tp <= 1:
        var inline_span = DispatchSpan[Profile]()
        reduce_store_range[E, buffers_origin, Accum](
            cfg, 0, 0, src.count)
        for r in range(1, tp):
            if cfg.dst[r] != cfg.dst[0]:
                memcpy(dest=cfg.dst[r], src=cfg.dst[0], count=src.count)
        inline_span.finish_inline(prof, "allreduce")
        return

    var data_bytes = src.count * E.ELEMENT_BYTES
    var reduce_span = DispatchSpan[Profile]()
    var reduce_buf = DispatchBuffer[
        ReduceStoreKernel[E, buffers_origin, Accum],
        max_worker_count,
    ]()
    for r in range(tp):
        var rank_start = cfg.chunk * r
        var rank_count = rank_chunk_count(cfg.chunk, cfg.rem, r, tp)
        var nw = recommended_workers(
            rank_count * E.ELEMENT_BYTES,
            min(max_worker_count, pools[r].get_capacity()),
        )
        _ = tile_dispatch(reduce_buf,
            ReduceStoreKernel[E, buffers_origin, Accum](
                cfg, r, 0, 0),
            pools[r], rank_count, rank_start, nw)
    reduce_span.issued()
    join_all(pools)
    reduce_span.finish(prof, pools, "allreduce.reduce")

    var gather_span = DispatchSpan[Profile]()
    var gather_buf = DispatchBuffer[
        GatherKernel[E, buffers_origin], max_worker_count,
    ]()
    for r in range(tp):
        var nw = recommended_workers(
            data_bytes, min(max_worker_count, pools[r].get_capacity()))
        _ = tile_dispatch(gather_buf,
            GatherKernel[E, buffers_origin](cfg, r, 0, 0),
            pools[r], src.count, num_workers=nw)
    gather_span.issued()
    join_all(pools)
    gather_span.finish(prof, pools, "allreduce.gather")


@always_inline
def dispatch_allreduce_inplace[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    E: Encoding, Accum: DType = DType.float32,
    max_worker_count: Int = 128,
](
    buf: Binding[Scalar[E.DTYPE], o], count: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
    inline_max_bytes: Int = DEFAULT_INLINE_BYTES,
):
    var tp = len(pools)
    var src = RankBuffers[E.DTYPE, ImmutAnyOrigin](count=count)
    var dst = RankBuffers[E.DTYPE, MutAnyOrigin](count=count)
    for r in range(tp):
        src.add(buf[r].as_immutable())
        dst.add(buf[r])
    dispatch_allreduce[E, Accum, max_worker_count=max_worker_count](
        src, dst, pools, prof, inline_max_bytes=inline_max_bytes)
