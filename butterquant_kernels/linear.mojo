from threading.threading_traits import BurstThreadPool
from kernels.helpers import (
    Chain, RangePartitionedKernel, WorkerRangePartitionedKernel,
    Binding, fanout_dispatch,
    matmul_workers, BF16Ptr,
)
from kernels.dispatch_heuristics import NORM_INLINE_TOKENS, GEMV_INLINE_ROWS
from kernels.profiling import Profiler

from butterquant.runtime import (
    prepare_norm_activation_per_row, prepare_block_activation,
)
from butterquant.gemm import gemm_i8_per_row, gemm_i8_per_block
from butterquant.amx_gemm import amx_gemm_linear_store, AMX_MIN_ROWS
from butterquant.vnni import VNNI_N_STEP
from simd_math import has_amx_int8
from butterquant.types import F32Ptr, I8Ptr
from butterquant.weight import (
    ButterquantWeight, ButterquantActivation, ButterquantBlockActivation,
    quant_vnni_packed, quant_has_colsum, quant_colsum_per_block, quant_k_block,
)
from quant.recipe import QuantRecipe


@fieldwise_init
struct BqNormQuantKernel[
    hidden: Int, block: Int, sqrt_n: Float32, n_eps: Float32,
](WorkerRangePartitionedKernel):
    var src: BF16Ptr
    var gamma: BF16Ptr
    var x_i8: I8Ptr
    var x_sa: F32Ptr
    var row_workspace: F32Ptr
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var row_workspace = (
            self.row_workspace + self.worker_id * Self.hidden)
        for tok in range(self.start, self.end):
            prepare_norm_activation_per_row[
                Self.hidden, Self.block, Self.sqrt_n, Self.n_eps,
            ](
                self.src + tok * Self.hidden, self.gamma,
                self.x_i8 + tok * Self.hidden, self.x_sa + tok,
                row_workspace)

    @always_inline
    def install_worker_range(
        mut self, worker_id: Int, start: Int, end: Int,
    ):
        self.worker_id = worker_id
        self.start = start
        self.end = end


def dispatch_bq_norm_quant[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    hidden: Int, block: Int, sqrt_n: Float32, n_eps: Float32,
    max_worker_count: Int = 128,
](
    src: Binding[BFloat16, o],
    gamma: Binding[BFloat16, o],
    x_i8: Binding[Int8, o],
    x_sa: Binding[Float32, o],
    row_workspace: Binding[Float32, o],
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime Kern = BqNormQuantKernel[hidden, block, sqrt_n, n_eps]

    @parameter
    def make(r: Int) -> Kern:
        return Kern(
            src[r], gamma[r], x_i8[r], x_sa[r], row_workspace[r],
            0, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="bq_norm_quant"](
        pools, prof, seq_len, seq_len * hidden * 6,
        inline_threshold_bytes=NORM_INLINE_TOKENS * hidden * 6)


@fieldwise_init
struct BqLinearKernel[K: Int, MR: Int, numer: Int = 1, denom: Int = 1](
    RangePartitionedKernel
):
    var act: I8Ptr
    var act_scale: F32Ptr
    var weight: I8Ptr
    var wsc: F32Ptr
    var colsum: F32Ptr
    var output: BF16Ptr
    var m: Int
    var n_rows: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var my_start = self.start * Self.numer // Self.denom
        var my_end = self.end * Self.numer // Self.denom
        comptime if has_amx_int8():
            if self.m >= AMX_MIN_ROWS:
                amx_gemm_linear_store[Self.K, DType.bfloat16](
                    self.act, self.m, self.n_rows, Self.K, self.act_scale,
                    self.weight, self.wsc, self.output, my_start, my_end)
                return
        gemm_i8_per_row[Self.K, Self.MR, DType.bfloat16](
            self.act, self.m, self.n_rows, self.act_scale, self.weight, self.wsc,
            self.colsum, self.output, my_start, my_end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_bq_linear[
    P: BurstThreadPool, quant: QuantRecipe, o: ImmutOrigin,
    Profile: Bool, N: Int, //,
    hidden: Int, MR: Int = 4, max_worker_count: Int = 128,
](
    act: ButterquantActivation[o],
    weight: ButterquantWeight[quant, o],
    output: Binding[BFloat16, o],
    n_rows: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime assert quant_vnni_packed[quant](), "bq linear consumes a VNNI-packed weight"
    comptime assert quant_has_colsum[quant](), "bq linear requires a colsum sidecar"
    if seq_len <= 0:
        return
    var num_tiles = n_rows // VNNI_N_STEP
    comptime Kern = BqLinearKernel[hidden, MR]
    var cs = weight.colsum_checked()

    @parameter
    def make(r: Int) -> Kern:
        return Kern(act.data[r], act.scale[r], weight.data[r], weight.scale[r],
                    cs[r], output[r], seq_len, n_rows, 0, 0)

    fanout_dispatch[
        make, max_worker_count=max_worker_count,
        worker_policy=matmul_workers, label="bq_linear",
    ](pools, prof, num_tiles, n_rows * hidden * seq_len,
      inline_threshold_bytes=GEMV_INLINE_ROWS * hidden)


@fieldwise_init
struct BqBlockQuantKernel[block: Int, apply_fwht: Bool](
    WorkerRangePartitionedKernel
):
    var src: BF16Ptr
    var x_i8: I8Ptr
    var x_sa: F32Ptr
    var row_workspace: F32Ptr
    var cols: Int
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var nb = self.cols // Self.block
        var ws = self.row_workspace + self.worker_id * self.cols
        for tok in range(self.start, self.end):
            prepare_block_activation[Self.block, Self.apply_fwht](
                self.src + tok * self.cols, self.x_i8 + tok * self.cols,
                self.x_sa + tok * nb, ws, self.cols)

    @always_inline
    def install_worker_range(
        mut self, worker_id: Int, start: Int, end: Int,
    ):
        self.worker_id = worker_id
        self.start = start
        self.end = end


def dispatch_bq_block_quant[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    block: Int, apply_fwht: Bool,
    max_worker_count: Int = 128,
](
    src: Binding[BFloat16, o],
    x_i8: Binding[Int8, o],
    x_sa: Binding[Float32, o],
    row_workspace: Binding[Float32, o],
    cols: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime Kern = BqBlockQuantKernel[block, apply_fwht]

    @parameter
    def make(r: Int) -> Kern:
        return Kern(
            src[r], x_i8[r], x_sa[r], row_workspace[r], cols, 0, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="bq_block_quant"](
        pools, prof, seq_len, seq_len * cols * 6,
        inline_threshold_bytes=NORM_INLINE_TOKENS * cols * 6)


@fieldwise_init
struct BqBlockLinearKernel[N: Int, block: Int, MR: Int](
    RangePartitionedKernel
):
    var act: I8Ptr
    var act_scale: F32Ptr
    var weight: I8Ptr
    var wsc: F32Ptr
    var colsum: F32Ptr
    var output: BF16Ptr
    var m: Int
    var k_dim: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime if has_amx_int8():
            if self.m >= AMX_MIN_ROWS:
                amx_gemm_linear_store[Self.block, DType.bfloat16](
                    self.act, self.m, Self.N, self.k_dim, self.act_scale,
                    self.weight, self.wsc, self.output, self.start, self.end)
                return
        gemm_i8_per_block[Self.N, Self.block, Self.MR, DType.bfloat16](
            self.act, self.m, self.k_dim, self.act_scale, self.weight, self.wsc,
            self.colsum, self.output, self.start, self.end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_bq_block_linear[
    P: BurstThreadPool, quant: QuantRecipe, o: ImmutOrigin,
    Profile: Bool, N: Int, //,
    n_rows: Int, MR: Int = 4, max_worker_count: Int = 128,
](
    act: ButterquantBlockActivation[o],
    weight: ButterquantWeight[quant, o],
    output: Binding[BFloat16, o],
    k_dim: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime assert quant_vnni_packed[quant](), "bq block linear consumes a VNNI-packed weight"
    comptime assert quant_colsum_per_block[quant](), "bq block linear requires a per-block colsum sidecar"
    if seq_len <= 0:
        return
    comptime num_tiles = n_rows // VNNI_N_STEP
    comptime Kern = BqBlockLinearKernel[n_rows, quant_k_block[quant](), MR]
    var cs = weight.colsum_checked()

    @parameter
    def make(r: Int) -> Kern:
        return Kern(act.data[r], act.scale[r], weight.data[r], weight.scale[r],
                    cs[r], output[r], seq_len, k_dim, 0, 0)

    fanout_dispatch[
        make, max_worker_count=max_worker_count,
        worker_policy=matmul_workers, label="bq_block_linear",
    ](pools, prof, num_tiles, n_rows * k_dim * seq_len,
      inline_threshold_bytes=GEMV_INLINE_ROWS * k_dim)


def dispatch_bq_qkv[
    P: BurstThreadPool, quant: QuantRecipe, o: ImmutOrigin,
    Profile: Bool, N: Int, //,
    hidden: Int, qn_full: Int, kvn_full: Int,
    MR: Int = 4, max_worker_count: Int = 128,
](
    act: ButterquantActivation[o],
    q_weight: ButterquantWeight[quant, o],
    k_weight: ButterquantWeight[quant, o],
    v_weight: ButterquantWeight[quant, o],
    q_out: Binding[BFloat16, o],
    k_out: Binding[BFloat16, o],
    v_out: Binding[BFloat16, o],
    qn: Int, kvn: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime assert quant_vnni_packed[quant](), "bq qkv consumes VNNI-packed weights"
    comptime assert quant_has_colsum[quant](), "bq qkv requires colsum sidecars"
    if seq_len <= 0:
        return
    comptime q_split = qn_full // VNNI_N_STEP
    comptime kv_split = kvn_full // VNNI_N_STEP
    comptime total_split = q_split + kv_split + kv_split
    var num_tiles = qn // VNNI_N_STEP + (kvn // VNNI_N_STEP) * 2
    comptime QKern = BqLinearKernel[hidden, MR, q_split, total_split]
    comptime KKern = BqLinearKernel[hidden, MR, kv_split, total_split]
    comptime QK = Chain[QKern, KKern]
    comptime QKV = Chain[QK, KKern]

    var qcs = q_weight.colsum_checked()
    var kcs = k_weight.colsum_checked()
    var vcs = v_weight.colsum_checked()

    @parameter
    def make(r: Int) -> QKV:
        return QKV(
            QK(
                QKern(act.data[r], act.scale[r], q_weight.data[r],
                      q_weight.scale[r], qcs[r], q_out[r], seq_len, qn, 0, 0),
                KKern(act.data[r], act.scale[r], k_weight.data[r],
                      k_weight.scale[r], kcs[r], k_out[r], seq_len, kvn, 0, 0),
            ),
            KKern(act.data[r], act.scale[r], v_weight.data[r],
                  v_weight.scale[r], vcs[r], v_out[r], seq_len, kvn, 0, 0),
        )

    fanout_dispatch[
        make,
        max_worker_count=max_worker_count,
        worker_policy=matmul_workers,
        label="bq_qkv",
    ](pools, prof, num_tiles, (qn + kvn + kvn) * hidden * seq_len)
