from std.memory import Span

from threading.threading_traits import BurstThreadPool
from kernels.helpers import (
    WorkerRangePartitionedKernel, Binding, fanout_dispatch, BF16Ptr,
)
from kernels.dispatch_heuristics import EMBED_INLINE_TOKENS
from kernels.profiling import Profiler

from butterquant.fwht import fwht_row
from butterquant.dequantize import (
    dequant_weight_row_per_block, scale_cast_row, zero_row,
)
from butterquant.types import F32Ptr, I8Ptr
from butterquant.weight import (
    ButterquantWeight, quant_k_block, quant_per_block,
)
from quant.recipe import QuantRecipe


@fieldwise_init
struct BqEmbedLookupKernel[
    tok_origin: ImmutOrigin,
    hidden: Int, block: Int, scale: Float64,
](WorkerRangePartitionedKernel):
    var token_ids: Span[Int32, Self.tok_origin]
    var weight: I8Ptr
    var scales: F32Ptr
    var dst: BF16Ptr
    var row_workspace: F32Ptr
    var shard_rows: Int
    var rank: Int
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime nb = Self.hidden // Self.block
        var row_workspace = self.row_workspace + self.worker_id * Self.hidden
        for tok in range(self.start, self.end):
            var tid = Int(self.token_ids[tok])
            var owner = tid // self.shard_rows
            var dst_row = self.dst + tok * Self.hidden
            if owner == self.rank:
                var local_row = tid - owner * self.shard_rows
                dequant_weight_row_per_block[Self.block](
                    self.weight + local_row * Self.hidden,
                    self.scales + local_row * nb,
                    row_workspace, Self.hidden)
                fwht_row[Self.block](row_workspace, Self.hidden)
                scale_cast_row[Self.hidden, Self.scale](
                    row_workspace, dst_row)
            else:
                zero_row[Self.hidden](dst_row)

    @always_inline
    def install_worker_range(
        mut self, worker_id: Int, start: Int, end: Int,
    ):
        self.worker_id = worker_id
        self.start = start
        self.end = end


def dispatch_bq_embed_lookup[
    P: BurstThreadPool, tok_origin: ImmutOrigin,
    quant: QuantRecipe, o: ImmutOrigin,
    Profile: Bool, N: Int, //,
    hidden: Int, scale: Float64,
    max_worker_count: Int = 128,
](
    token_ids: Span[Int32, tok_origin],
    weight: ButterquantWeight[quant, o],
    dst: Binding[BFloat16, o],
    row_workspace: Binding[Float32, o],
    shard_rows: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime assert quant_per_block[quant](), "embed lookup expects a per-block weight scale"
    comptime K = BqEmbedLookupKernel[
        tok_origin, hidden, quant_k_block[quant](), scale,
    ]

    @parameter
    def make(r: Int) -> K:
        return K(
            token_ids, weight.data[r], weight.scale[r], dst[r],
            row_workspace[r], shard_rows,
            r, 0, 0, 0,
        )

    fanout_dispatch[make, max_worker_count=max_worker_count, label="bq_embed_lookup"](
        pools, prof, seq_len, seq_len * hidden * 6,
        inline_threshold_bytes=EMBED_INLINE_TOKENS * hidden * 6)
