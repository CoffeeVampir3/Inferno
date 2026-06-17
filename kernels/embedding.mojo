from std.algorithm import vectorize
from std.memory import Span

from threading.threading_traits import BurstThreadPool
from .helpers import (
    RangePartitionedKernel, Binding,
    fanout_dispatch,
    BF16Ptr, W,
)
from .dispatch_heuristics import EMBED_INLINE_TOKENS
from .profiling import Profiler


@always_inline
def embed_scaled_copy_row[hidden: Int, scale: Float64](
    src: BF16Ptr, dst: BF16Ptr,
):
    def step[width: Int](idx: Int) {read}:
        var x = (src + idx).load[width=width]().cast[DType.float32]()
        var f = SIMD[DType.float32, width](Float32(scale))
        (dst + idx).store((x * f).cast[DType.bfloat16]())

    vectorize[W](hidden, step)


@always_inline
def embed_zero_row[hidden: Int](dst: BF16Ptr):
    def step[width: Int](idx: Int) {read}:
        (dst + idx).store(SIMD[DType.bfloat16, width](0))

    vectorize[W](hidden, step)


@fieldwise_init
struct EmbedLookupKernel[
    tok_origin: ImmutOrigin,
    hidden: Int, scale: Float64,
](RangePartitionedKernel):
    var token_ids: Span[Int32, Self.tok_origin]
    var embed: BF16Ptr
    var dst: BF16Ptr
    var rank: Int
    var shard_rows: Int
    var start: Int
    var end: Int

    def execute(mut self):
        for tok in range(self.start, self.end):
            var tid = Int(self.token_ids[tok])
            var owner = tid // self.shard_rows
            var dst_row = self.dst + tok * Self.hidden
            if owner == self.rank:
                var local_row = tid - owner * self.shard_rows
                var src_row = self.embed + local_row * Self.hidden
                embed_scaled_copy_row[Self.hidden, Self.scale](
                    src_row, dst_row)
            else:
                embed_zero_row[Self.hidden](dst_row)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_embed_lookup[
    P: BurstThreadPool, tok_origin: ImmutOrigin, Profile: Bool, N: Int,
    o: ImmutOrigin, //,
    hidden: Int, scale: Float64,
    max_worker_count: Int = 128,
](
    token_ids: Span[Int32, tok_origin],
    embed: Binding[BFloat16, o],
    dst: Binding[BFloat16, o],
    shard_rows: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    """Unowned tokens write zero; caller follows with allreduce to replicate."""
    comptime K = EmbedLookupKernel[tok_origin, hidden, scale]
    var sr = shard_rows

    @parameter
    def make(r: Int) -> K:
        return K(token_ids, embed[r], dst[r], r, sr, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="embed_lookup"](
        pools, prof, seq_len, seq_len * hidden * 6,
        inline_threshold_bytes=EMBED_INLINE_TOKENS * hidden * 6)
