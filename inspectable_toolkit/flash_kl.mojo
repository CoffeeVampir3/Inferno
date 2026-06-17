from std.collections import InlineArray
from std.memory import UnsafePointer

from threading.threading_traits import BurstThreadPool
from simd_math import pick_port_unroll
from simd_math.ops import exp_f32, softcap_value

from kernels.dot_products import bf16_panel_dot_to_scalars
from kernels.helpers import (
    BF16Ptr, F32Ptr, BW, WorkerRangePartitionedKernel, Binding,
    fanout_dispatch_per_rank, saturate_workers,
)
from kernels.reductions import dispatch_allreduce_inplace
from kernels.profiling import Profiler
from modeling.model_spec import F32


@always_inline
def kl_absorb_panel[
    panel: Int, //,
    cols: Int, cap: Float64, port_unroll: Int,
](
    mod_x: BF16Ptr,
    base_x: BF16Ptr,
    wrow: BF16Ptr,
    accums: F32Ptr,
    base_logz: F32Ptr,
    mod_logz: F32Ptr,
    row_start: Int,
):
    var mod_rows = InlineArray[BF16Ptr, panel](uninitialized=True)
    var base_rows = InlineArray[BF16Ptr, panel](uninitialized=True)
    comptime for r in range(panel):
        mod_rows[r] = mod_x + (row_start + r) * cols
        base_rows[r] = base_x + (row_start + r) * cols
    var mdots = bf16_panel_dot_to_scalars[
        cols=cols, port_unroll=port_unroll,
    ](wrow, mod_rows)
    var bdots = bf16_panel_dot_to_scalars[
        cols=cols, port_unroll=port_unroll,
    ](wrow, base_rows)
    comptime for r in range(panel):
        var row = row_start + r
        var lb = softcap_value[cap](
            SIMD[DType.float32, 1](bdots[r])) - (base_logz + row)[]
        var lm = softcap_value[cap](
            SIMD[DType.float32, 1](mdots[r])) - (mod_logz + row)[]
        (accums + row)[] = (accums + row)[] + exp_f32[1](lb) * (lb - lm)


@fieldwise_init
struct FlashKLKernel[cols: Int, cap: Float64, MR: Int](
    WorkerRangePartitionedKernel
):
    var mod_x: BF16Ptr
    var base_x: BF16Ptr
    var weight: BF16Ptr
    var base_logz: F32Ptr
    var mod_logz: F32Ptr
    var accums: F32Ptr
    var num_rows: Int
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime PU = pick_port_unroll[BW, Self.cols]()
        var base = self.accums + self.worker_id * self.num_rows
        for j in range(self.num_rows):
            (base + j)[] = Float32(0)
        for vrow in range(self.start, self.end):
            var wrow = self.weight + vrow * Self.cols
            var row = 0
            while row + Self.MR <= self.num_rows:
                kl_absorb_panel[
                    panel=Self.MR,
                    cols=Self.cols, cap=Self.cap, port_unroll=PU,
                ](self.mod_x, self.base_x, wrow, base,
                  self.base_logz, self.mod_logz, row)
                row += Self.MR
            comptime if Self.MR >= 4:
                while row + 4 <= self.num_rows:
                    kl_absorb_panel[
                        panel=4,
                        cols=Self.cols, cap=Self.cap, port_unroll=PU,
                    ](self.mod_x, self.base_x, wrow, base,
                      self.base_logz, self.mod_logz, row)
                    row += 4
            comptime if Self.MR >= 2:
                while row + 2 <= self.num_rows:
                    kl_absorb_panel[
                        panel=2,
                        cols=Self.cols, cap=Self.cap, port_unroll=PU,
                    ](self.mod_x, self.base_x, wrow, base,
                      self.base_logz, self.mod_logz, row)
                    row += 2
            while row < self.num_rows:
                kl_absorb_panel[
                    panel=1,
                    cols=Self.cols, cap=Self.cap, port_unroll=PU,
                ](self.mod_x, self.base_x, wrow, base,
                  self.base_logz, self.mod_logz, row)
                row += 1

    @always_inline
    def install_worker_range(mut self, worker_id: Int, start: Int, end: Int):
        self.worker_id = worker_id
        self.start = start
        self.end = end


def dispatch_flash_kl_fixed[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    cols: Int, cap: Float64, MR: Int,
    max_worker_count: Int = 128,
](
    mod_x: Binding[BFloat16, o],
    base_x: Binding[BFloat16, o],
    weight: Binding[BFloat16, o],
    base_logz: Binding[Float32, o],
    mod_logz: Binding[Float32, o],
    accums: Binding[Float32, o],
    partials: Binding[Float32, o],
    num_rows: Int,
    vocab_per_rank: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime K = FlashKLKernel[cols, cap, MR]
    var nr = num_rows
    var vpr = vocab_per_rank

    @parameter
    def make(r: Int) -> K:
        return K(mod_x[r], base_x[r], weight[r], base_logz[r], mod_logz[r],
                 accums[r], nr, 0, 0, 0)

    @parameter
    def total_for(r: Int) -> Int:
        return vpr

    @parameter
    def data_bytes_for(r: Int) -> Int:
        return vpr * cols * 2 + 2 * nr * cols * 2

    var worker_counts = fanout_dispatch_per_rank[
        make, total_for, data_bytes_for,
        max_worker_count=max_worker_count,
        worker_policy=saturate_workers,
        label="flash_kl",
    ](pools, prof)

    var tp = len(pools)
    for r in range(tp):
        var pr = partials[r]
        var ar = accums[r]
        for j in range(num_rows):
            (pr + j)[] = Float32(0)
        for w in range(worker_counts[r]):
            for j in range(num_rows):
                (pr + j)[] = (pr + j)[] + (ar + w * num_rows + j)[]

    dispatch_allreduce_inplace[F32, Accum = DType.float32](
        partials, num_rows, pools, prof)


def dispatch_flash_kl[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    cols: Int, cap: Float64,
    max_worker_count: Int = 128,
](
    mod_x: Binding[BFloat16, o],
    base_x: Binding[BFloat16, o],
    weight: Binding[BFloat16, o],
    base_logz: Binding[Float32, o],
    mod_logz: Binding[Float32, o],
    accums: Binding[Float32, o],
    partials: Binding[Float32, o],
    num_rows: Int,
    vocab_per_rank: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if num_rows >= 4:
        dispatch_flash_kl_fixed[
            cols, cap, 4, max_worker_count=max_worker_count,
        ](mod_x, base_x, weight, base_logz, mod_logz, accums, partials,
          num_rows, vocab_per_rank, pools, prof)
    elif num_rows >= 2:
        dispatch_flash_kl_fixed[
            cols, cap, 2, max_worker_count=max_worker_count,
        ](mod_x, base_x, weight, base_logz, mod_logz, accums, partials,
          num_rows, vocab_per_rank, pools, prof)
    else:
        dispatch_flash_kl_fixed[
            cols, cap, 1, max_worker_count=max_worker_count,
        ](mod_x, base_x, weight, base_logz, mod_logz, accums, partials,
          num_rows, vocab_per_rank, pools, prof)
