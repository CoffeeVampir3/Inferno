from std.collections import InlineArray
from std.memory import UnsafePointer

from threading.threading_traits import BurstThreadPool
from simd_math import pick_port_unroll
from simd_math.ops import exp_f32, log_f32, softcap_value
from simd_math.sampling_rng import gumbel_noise, rng_counter

from .dot_products import bf16_panel_dot_to_scalars
from .helpers import (
    BF16Ptr, BW, WorkerRangePartitionedKernel, Binding,
    fanout_dispatch_per_rank, saturate_workers,
)
from .profiling import Profiler


comptime NEG_INF = Float32(-1.0e30)


@fieldwise_init
struct SamplingParams(Copyable, ImplicitlyCopyable):
    var temperature: Float32
    var min_p: Float32
    var top_k: Int
    var seed: UInt64
    var n_keep: Int
    var greedy: Bool


struct SampleAccum[n_max: Int](Copyable, Movable):
    var n: Int
    var topn_val: InlineArray[Float32, Self.n_max]
    var topn_g: InlineArray[Float32, Self.n_max]
    var topn_idx: InlineArray[Int32, Self.n_max]
    var lse_max: Float32
    var lse_sum: Float32
    var samp_score: Float32
    var samp_idx: Int32

    def __init__(out self):
        self.n = 0
        self.topn_val = InlineArray[Float32, Self.n_max](fill=NEG_INF)
        self.topn_g = InlineArray[Float32, Self.n_max](fill=NEG_INF)
        self.topn_idx = InlineArray[Int32, Self.n_max](fill=Int32(-1))
        self.lse_max = NEG_INF
        self.lse_sum = 0.0
        self.samp_score = NEG_INF
        self.samp_idx = Int32(-1)

    @always_inline
    def topn_insert(mut self, y: Float32, g: Float32, idx: Int, cap: Int):
        if self.n == cap and y <= self.topn_val[cap - 1]:
            return
        var p: Int
        if self.n < cap:
            p = self.n
            self.n += 1
        else:
            p = cap - 1
        while p > 0 and self.topn_val[p - 1] < y:
            self.topn_val[p] = self.topn_val[p - 1]
            self.topn_g[p] = self.topn_g[p - 1]
            self.topn_idx[p] = self.topn_idx[p - 1]
            p -= 1
        self.topn_val[p] = y
        self.topn_g[p] = g
        self.topn_idx[p] = Int32(idx)

    @always_inline
    def lse_push(mut self, yt: Float32):
        if self.lse_sum <= 0.0:
            self.lse_max = yt
            self.lse_sum = 1.0
        elif yt > self.lse_max:
            self.lse_sum = self.lse_sum * exp_f32[1](self.lse_max - yt) + 1.0
            self.lse_max = yt
        else:
            self.lse_sum = self.lse_sum + exp_f32[1](yt - self.lse_max)

    @always_inline
    def absorb(mut self, y: Float32, idx: Int, inv_t: Float32, g: Float32, cap: Int):
        var yt = y * inv_t
        self.lse_push(yt)
        var score = yt + g
        if score > self.samp_score:
            self.samp_score = score
            self.samp_idx = Int32(idx)
        self.topn_insert(y, score, idx, cap)

    def merge(mut self, read other: Self, cap: Int):
        if other.lse_sum > 0.0:
            if self.lse_sum <= 0.0:
                self.lse_max = other.lse_max
                self.lse_sum = other.lse_sum
            elif other.lse_max > self.lse_max:
                self.lse_sum = self.lse_sum * exp_f32[1](
                    self.lse_max - other.lse_max) + other.lse_sum
                self.lse_max = other.lse_max
            else:
                self.lse_sum = self.lse_sum + other.lse_sum * exp_f32[1](
                    other.lse_max - self.lse_max)
        if other.samp_score > self.samp_score:
            self.samp_score = other.samp_score
            self.samp_idx = other.samp_idx
        for k in range(other.n):
            self.topn_insert(
                other.topn_val[k], other.topn_g[k], Int(other.topn_idx[k]), cap)

    @always_inline
    def logz(self) -> Float32:
        return self.lse_max + log_f32[1](self.lse_sum)


struct SampleOutcome[n_max: Int](Copyable, Movable, ImplicitlyCopyable):
    var token_id: Int32
    var logz: Float32
    var n: Int
    var topn_val: InlineArray[Float32, Self.n_max]
    var topn_idx: InlineArray[Int32, Self.n_max]

    def __init__(out self):
        self.token_id = Int32(-1)
        self.logz = NEG_INF
        self.n = 0
        self.topn_val = InlineArray[Float32, Self.n_max](fill=NEG_INF)
        self.topn_idx = InlineArray[Int32, Self.n_max](fill=Int32(-1))


@always_inline
def absorb_sample_panel[
    panel: Int, //,
    cols: Int, cap: Float64, n_max: Int, port_unroll: Int,
](
    x: BF16Ptr,
    wrow: BF16Ptr,
    accums: UnsafePointer[SampleAccum[n_max], MutAnyOrigin],
    params: UnsafePointer[SamplingParams, MutAnyOrigin],
    row_start: Int,
    gidx: Int,
    row_base: Int,
):
    var x_rows = InlineArray[BF16Ptr, panel](uninitialized=True)
    comptime for r in range(panel):
        x_rows[r] = x + (row_start + r) * cols
    var dots = bf16_panel_dot_to_scalars[
        cols=cols, port_unroll=port_unroll,
    ](wrow, x_rows)
    comptime for r in range(panel):
        var row = row_start + r
        var p = (params + row)[]
        var inv_t = Float32(1.0) / p.temperature
        var y = softcap_value[cap](
            SIMD[DType.float32, 1](dots[r])).cast[DType.bfloat16]().cast[
            DType.float32]()
        var g = Float32(0.0) if p.greedy else gumbel_noise(
            p.seed, rng_counter(row_base + row, gidx))
        (accums + row)[].absorb(y, gidx, inv_t, g, n_max)


@fieldwise_init
struct FlashSampleKernel[cols: Int, cap: Float64, n_max: Int, MR: Int](
    WorkerRangePartitionedKernel
):
    var x: BF16Ptr
    var weight: BF16Ptr
    var accums: UnsafePointer[SampleAccum[Self.n_max], MutAnyOrigin]
    var params: UnsafePointer[SamplingParams, MutAnyOrigin]
    var num_rows: Int
    var rank_base: Int
    var row_base: Int
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime PU = pick_port_unroll[BW, Self.cols]()
        var base = self.accums + self.worker_id * self.num_rows
        for j in range(self.num_rows):
            (base + j)[] = SampleAccum[Self.n_max]()
        for vrow in range(self.start, self.end):
            var gidx = self.rank_base + vrow
            var wrow = self.weight + vrow * Self.cols
            var row = 0
            while row + Self.MR <= self.num_rows:
                absorb_sample_panel[
                    panel=Self.MR,
                    cols=Self.cols, cap=Self.cap, n_max=Self.n_max,
                    port_unroll=PU,
                ](self.x, wrow, base, self.params, row, gidx, self.row_base)
                row += Self.MR
            comptime if Self.MR >= 4:
                while row + 4 <= self.num_rows:
                    absorb_sample_panel[
                        panel=4,
                        cols=Self.cols, cap=Self.cap, n_max=Self.n_max,
                        port_unroll=PU,
                    ](self.x, wrow, base, self.params, row, gidx, self.row_base)
                    row += 4
            comptime if Self.MR >= 2:
                while row + 2 <= self.num_rows:
                    absorb_sample_panel[
                        panel=2,
                        cols=Self.cols, cap=Self.cap, n_max=Self.n_max,
                        port_unroll=PU,
                    ](self.x, wrow, base, self.params, row, gidx, self.row_base)
                    row += 2
            while row < self.num_rows:
                absorb_sample_panel[
                    panel=1,
                    cols=Self.cols, cap=Self.cap, n_max=Self.n_max,
                    port_unroll=PU,
                ](self.x, wrow, base, self.params, row, gidx, self.row_base)
                row += 1

    @always_inline
    def install_worker_range(mut self, worker_id: Int, start: Int, end: Int):
        self.worker_id = worker_id
        self.start = start
        self.end = end


def finalize_outcomes[
    n_max: Int, o: ImmutOrigin, //,
](
    accums: Binding[SampleAccum[n_max], o],
    params: Binding[SamplingParams, o],
    outcome: UnsafePointer[SampleOutcome[n_max], MutAnyOrigin],
    num_rows: Int,
    read worker_counts: List[Int],
):
    var prow = params[0]
    for j in range(num_rows):
        var final = SampleAccum[n_max]()
        for r in range(len(worker_counts)):
            var ar = accums[r]
            for w in range(worker_counts[r]):
                final.merge((ar + w * num_rows + j)[], n_max)
        var p = (prow + j)[]
        var n_out = final.n if final.n < p.n_keep else p.n_keep
        var token = final.samp_idx
        var use_min_p = p.min_p > 0.0
        var use_top_k = p.top_k > 0
        if (use_min_p or use_top_k) and final.n > 0:
            var limit = final.n
            if use_top_k and p.top_k < limit:
                limit = p.top_k
            var thr = NEG_INF
            if use_min_p:
                thr = final.topn_val[0] + p.temperature * log_f32[1](p.min_p)
            var best_idx = final.topn_idx[0]
            var best_g = final.topn_g[0]
            for k in range(1, limit):
                if final.topn_val[k] >= thr and final.topn_g[k] > best_g:
                    best_g = final.topn_g[k]
                    best_idx = final.topn_idx[k]
            token = best_idx
        var op = outcome + j
        op[].token_id = token
        op[].logz = final.logz()
        op[].n = n_out
        for k in range(n_out):
            op[].topn_val[k] = final.topn_val[k]
            op[].topn_idx[k] = final.topn_idx[k]


def dispatch_flash_sample_fixed[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    cols: Int, cap: Float64, n_max: Int, MR: Int,
    max_worker_count: Int = 128,
](
    x: Binding[BFloat16, o],
    weight: Binding[BFloat16, o],
    accums: Binding[SampleAccum[n_max], o],
    params: Binding[SamplingParams, o],
    outcome: UnsafePointer[SampleOutcome[n_max], MutAnyOrigin],
    num_rows: Int,
    vocab_per_rank: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime K = FlashSampleKernel[cols, cap, n_max, MR]
    var nr = num_rows
    var vpr = vocab_per_rank

    @parameter
    def make(r: Int) -> K:
        return K(x[r], weight[r], accums[r], params[r], nr, r * vpr, 0,
                 0, 0, 0)

    @parameter
    def total_for(r: Int) -> Int:
        return vpr

    @parameter
    def data_bytes_for(r: Int) -> Int:
        return vpr * cols * 2 + nr * cols * 2

    var worker_counts = fanout_dispatch_per_rank[
        make, total_for, data_bytes_for,
        max_worker_count=max_worker_count,
        worker_policy=saturate_workers,
        label="flash_sample",
    ](pools, prof)
    finalize_outcomes(accums, params, outcome, num_rows, worker_counts)


def dispatch_flash_sample[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    cols: Int, cap: Float64, n_max: Int, MR: Int = 0,
    max_worker_count: Int = 128,
](
    x: Binding[BFloat16, o],
    weight: Binding[BFloat16, o],
    accums: Binding[SampleAccum[n_max], o],
    params: Binding[SamplingParams, o],
    outcome: UnsafePointer[SampleOutcome[n_max], MutAnyOrigin],
    num_rows: Int,
    vocab_per_rank: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime if MR != 0:
        dispatch_flash_sample_fixed[
            cols, cap, n_max, MR,
            max_worker_count=max_worker_count,
        ](x, weight, accums, params, outcome, num_rows, vocab_per_rank,
          pools, prof)
        return

    if num_rows >= 4:
        dispatch_flash_sample_fixed[
            cols, cap, n_max, 4,
            max_worker_count=max_worker_count,
        ](x, weight, accums, params, outcome, num_rows, vocab_per_rank,
          pools, prof)
    elif num_rows >= 2:
        dispatch_flash_sample_fixed[
            cols, cap, n_max, 2,
            max_worker_count=max_worker_count,
        ](x, weight, accums, params, outcome, num_rows, vocab_per_rank,
          pools, prof)
    else:
        dispatch_flash_sample_fixed[
            cols, cap, n_max, 1,
            max_worker_count=max_worker_count,
        ](x, weight, accums, params, outcome, num_rows, vocab_per_rank,
          pools, prof)
