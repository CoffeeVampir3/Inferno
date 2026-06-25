from std.collections import InlineArray

from .helpers import (
    BF16Ptr, F32Ptr,
    WorkerRangePartitionedKernel,
)
from .attention_ops import KVSlot, TILE, process_kv_tile, zero_accumulators


@fieldwise_init
struct FlashAttentionKernel[
    KV: KVSlot,
    head_dim: Int, max_q: Int, gqa_ratio: Int,
](WorkerRangePartitionedKernel):
    var kv: Self.KV
    var q: BF16Ptr
    var k_base: BF16Ptr
    var v_base: BF16Ptr
    var partials: F32Ptr
    var num_q: Int
    var partial_stride: Int
    var kv_stride: Int
    var worker_id: Int
    var start_pos: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var my_partial = self.partials + self.worker_id * self.partial_stride
        var m_off = self.num_q * Self.head_dim
        var l_off = m_off + self.num_q

        var acc_ptrs = InlineArray[F32Ptr, Self.max_q](uninitialized=True)
        var q_ptrs = InlineArray[BF16Ptr, Self.max_q](uninitialized=True)
        var m = InlineArray[Float32, Self.max_q](fill=Float32(-1e30))
        var l = InlineArray[Float32, Self.max_q](fill=Float32(0))

        for h in range(self.num_q):
            acc_ptrs[h] = my_partial + h * Self.head_dim
            q_ptrs[h] = self.q + h * Self.head_dim

        zero_accumulators[Self.max_q, Self.head_dim](acc_ptrs, self.num_q)

        var pos = self.start
        while pos < self.end:
            var tile_len = min(TILE, self.end - pos)
            process_kv_tile[
                Self.head_dim, Self.gqa_ratio,
            ](self.kv, q_ptrs,
              self.k_base,
              self.v_base,
              self.start_pos, pos, tile_len, m, l, acc_ptrs,
              self.num_q, self.kv_stride)
            pos += TILE

        for h in range(self.num_q):
            (my_partial + m_off + h)[] = m[h]
            (my_partial + l_off + h)[] = l[h]

    @always_inline
    def install_worker_range(
        mut self, worker_id: Int, start: Int, end: Int,
    ):
        self.worker_id = worker_id
        self.start = start
        self.end = end
