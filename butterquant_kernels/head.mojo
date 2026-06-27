from std.memory import UnsafePointer

from threading.threading_traits import BurstThreadPool
from kernels.helpers import (
    WorkerRangePartitionedKernel, Binding, fanout_dispatch,
    fanout_dispatch_per_rank, saturate_workers, BF16Ptr,
)
from kernels.dispatch_heuristics import NORM_INLINE_TOKENS
from kernels.flash_sample import (
    SamplingParams, SampleAccum, SampleOutcome, finalize_outcomes,
)
from kernels.profiling import Profiler
from simd_math.ops import softcap_value
from simd_math.sampling_rng import gumbel_noise, rng_counter

from butterquant.dot_products import head_logit_row
from butterquant.runtime import prepare_head_activation
from butterquant.types import F32Ptr, I8Ptr
from butterquant.weight import (
    ButterquantWeight, ButterquantActivation, quant_k_block, quant_per_block,
)
from quant.recipe import QuantRecipe


@fieldwise_init
struct BqHeadPrepKernel[
    hidden: Int, block: Int, sqrt_n: Float32, n_eps: Float32,
](WorkerRangePartitionedKernel):
    var src: BF16Ptr
    var gamma: BF16Ptr
    var x_i8: I8Ptr
    var sa: F32Ptr
    var row_workspace: F32Ptr
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime nb = Self.hidden // Self.block
        var row_workspace = self.row_workspace + self.worker_id * Self.hidden
        for row in range(self.start, self.end):
            prepare_head_activation[
                Self.hidden, Self.block, Self.sqrt_n, Self.n_eps,
            ](self.src + row * Self.hidden, self.gamma,
              self.x_i8 + row * Self.hidden, self.sa + row * nb,
              row_workspace)

    @always_inline
    def install_worker_range(mut self, worker_id: Int, start: Int, end: Int):
        self.worker_id = worker_id
        self.start = start
        self.end = end


def dispatch_bq_head_prep[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    hidden: Int, block: Int, sqrt_n: Float32, n_eps: Float32,
    max_worker_count: Int = 128,
](
    src: Binding[BFloat16, o],
    gamma: Binding[BFloat16, o],
    dst: ButterquantActivation[o],
    row_workspace: Binding[Float32, o],
    num_rows: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime K = BqHeadPrepKernel[hidden, block, sqrt_n, n_eps]

    @parameter
    def make(r: Int) -> K:
        return K(src[r], gamma[r], dst.data[r], dst.scale[r],
                 row_workspace[r], 0, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="bq_head_prep"](
        pools, prof, num_rows, num_rows * hidden * 6,
        inline_threshold_bytes=NORM_INLINE_TOKENS * hidden * 6)


@fieldwise_init
struct BqFlashSampleKernel[
    cols: Int, block: Int, cap: Float64, n_max: Int,
](WorkerRangePartitionedKernel):
    var x_i8: I8Ptr
    var x_sa: F32Ptr
    var weight: I8Ptr
    var wscales: F32Ptr
    var accums: UnsafePointer[SampleAccum[Self.n_max], MutUntrackedOrigin]
    var params: UnsafePointer[SamplingParams, MutUntrackedOrigin]
    var num_rows: Int
    var rank_base: Int
    var row_base: Int
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime nb = Self.cols // Self.block
        var base = self.accums + self.worker_id * self.num_rows
        for j in range(self.num_rows):
            (base + j)[] = SampleAccum[Self.n_max]()
        for vrow in range(self.start, self.end):
            var gidx = self.rank_base + vrow
            var wrow = self.weight + vrow * Self.cols
            var wsc = self.wscales + vrow * nb
            for row in range(self.num_rows):
                var raw = head_logit_row[Self.block](
                    self.x_i8 + row * Self.cols,
                    self.x_sa + row * nb,
                    wrow, wsc, Self.cols)
                var p = (self.params + row)[]
                var inv_t = Float32(1.0) / p.temperature
                var capped = SIMD[DType.float32, 1](raw)
                comptime
                if Self.cap > 0.0:
                    capped = softcap_value[Self.cap](capped)
                var y = capped.cast[DType.bfloat16]().cast[DType.float32]()
                var g = Float32(0.0) if p.greedy else gumbel_noise(
                    p.seed, rng_counter(self.row_base + row, gidx))
                (base + row)[].absorb(y, gidx, inv_t, g, Self.n_max)

    @always_inline
    def install_worker_range(mut self, worker_id: Int, start: Int, end: Int):
        self.worker_id = worker_id
        self.start = start
        self.end = end


def dispatch_bq_flash_sample[
    P: BurstThreadPool, quant: QuantRecipe, o: ImmutOrigin,
    Profile: Bool, N: Int, //,
    cols: Int, cap: Float64, n_max: Int,
    max_worker_count: Int = 128,
](
    act: ButterquantActivation[o],
    weight: ButterquantWeight[quant, o],
    accums: Binding[SampleAccum[n_max], o],
    params: Binding[SamplingParams, o],
    outcome: UnsafePointer[SampleOutcome[n_max], MutUntrackedOrigin],
    num_rows: Int,
    vocab_per_rank: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime assert quant_per_block[quant](), "flash sample consumes a per-block weight scale"
    comptime K = BqFlashSampleKernel[cols, quant_k_block[quant](), cap, n_max]
    var nr = num_rows
    var vpr = vocab_per_rank

    @parameter
    def make(r: Int) -> K:
        return K(act.data[r], act.scale[r], weight.data[r], weight.scale[r],
                 accums[r], params[r], nr, r * vpr, 0, 0, 0, 0)

    @parameter
    def total_for(r: Int) -> Int:
        return vpr

    @parameter
    def data_bytes_for(r: Int) -> Int:
        return vpr * cols + nr * cols

    var worker_counts = fanout_dispatch_per_rank[
        make, total_for, data_bytes_for,
        max_worker_count=max_worker_count,
        worker_policy=saturate_workers,
        label="bq_flash_sample",
    ](pools, prof)
    finalize_outcomes(accums, params, outcome, num_rows, worker_counts)
