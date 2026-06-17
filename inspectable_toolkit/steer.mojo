from std.algorithm import vectorize

from threading.threading_traits import BurstThreadPool
from kernels.helpers import (
    Binding, BF16Ptr, BW, W, RangePartitionedKernel, fanout_dispatch, copy_row,
)
from kernels.dispatch_heuristics import SCALAR_MUL_INLINE_TOKENS
from kernels.profiling import Profiler
from continuous_batching.schedule import Schedule
from modeling.gemma4_common import Gemma4BaseConfig


comptime C = Gemma4BaseConfig


@fieldwise_init
struct InjectOp(Copyable, Movable, ImplicitlyCopyable):
    var layer: Int
    var vec_idx: Int
    var alpha: Float32


struct SteerState(Movable):
    var armed: Bool
    var tap_layers: List[Int]
    var tap_pos: List[Int]
    var max_slots: Int
    var sink: List[BFloat16]
    var last_rows: List[Int]
    var last_step_requests: List[Int]
    var last_num_slots: Int
    var verify_rank: Int
    var mismatch_count: Int
    var inject_ops: List[InjectOp]
    var per_request: Bool
    var req_ops: List[List[InjectOp]]

    def __init__(out self, max_slots: Int):
        self.armed = False
        self.tap_layers = List[Int]()
        self.tap_pos = List[Int](length=C.NUM_LAYERS, fill=-1)
        self.max_slots = max_slots
        self.sink = List[BFloat16]()
        self.last_rows = List[Int]()
        self.last_step_requests = List[Int]()
        self.last_num_slots = 0
        self.verify_rank = -1
        self.mismatch_count = 0
        self.inject_ops = List[InjectOp]()
        self.per_request = False
        self.req_ops = List[List[InjectOp]]()

    def arm(mut self, var layers: List[Int], verify_rank: Int = -1):
        self.tap_layers = layers^
        self.tap_pos = List[Int](length=C.NUM_LAYERS, fill=-1)
        for k in range(len(self.tap_layers)):
            self.tap_pos[self.tap_layers[k]] = k
        var size = len(self.tap_layers) * self.max_slots * C.HIDDEN
        self.sink = List[BFloat16](length=size, fill=BFloat16(0))
        self.verify_rank = verify_rank
        self.mismatch_count = 0
        self.armed = True

    def set_inject(mut self, var ops: List[InjectOp]):
        self.inject_ops = ops^
        self.per_request = False
        self.armed = True

    def set_request_inject(
        mut self, request_id: Int, var ops: List[InjectOp]
    ):
        while len(self.req_ops) <= request_id:
            self.req_ops.append(List[InjectOp]())
        self.req_ops[request_id] = ops^
        self.per_request = True
        self.armed = True

    def clear_inject(mut self):
        self.inject_ops = List[InjectOp]()
        self.req_ops = List[List[InjectOp]]()
        self.per_request = False

    def disarm(mut self):
        self.armed = False

    def record_step(
        mut self, read schedule: Schedule, read buf_starts: List[Int],
        num_slots: Int,
    ):
        self.last_rows = List[Int](capacity=num_slots)
        self.last_step_requests = List[Int](capacity=num_slots)
        for s in range(num_slots):
            self.last_rows.append(
                buf_starts[s] + schedule.slots[s].n_tokens - 1)
            self.last_step_requests.append(schedule.slots[s].request_id)
        self.last_num_slots = num_slots

    @always_inline
    def tap_index(self, layer_idx: Int) -> Int:
        return self.tap_pos[layer_idx]

    @always_inline
    def sink_ptr(mut self) -> BF16Ptr:
        return self.sink.unsafe_ptr().as_unsafe_any_origin()

    @always_inline
    def captured_ptr(mut self, tap_pos: Int, slot: Int) -> BF16Ptr:
        return self.sink_ptr() + (tap_pos * self.max_slots + slot) * C.HIDDEN


trait Steerable:
    comptime STEER_VECTORS: Int
    def set_steer_vector(mut self, idx: Int, read vec: List[BFloat16]): ...
    def set_inject_ops(mut self, var ops: List[InjectOp]): ...
    def disarm_steer(mut self): ...


@always_inline
def row_mismatch[hidden: Int](a: BF16Ptr, b: BF16Ptr) -> Int:
    var count = 0
    for i in range(hidden):
        if a[i] != b[i]:
            count += 1
    return count


def dispatch_steer_point[
    o: ImmutOrigin, //, hidden: Int,
](
    x_main: Binding[BFloat16, o],
    read last_rows: List[Int],
    num_slots: Int,
    tap_pos: Int,
    sink: BF16Ptr,
    max_slots: Int,
    verify_rank: Int = -1,
) -> Int:
    var src0 = x_main[0]
    var do_verify = (
        verify_rank > 0 and verify_rank < x_main.degree()
    )
    var alt = x_main[verify_rank] if do_verify else src0
    var mismatches = 0
    for s in range(num_slots):
        var off = last_rows[s] * hidden
        var dst = sink + (tap_pos * max_slots + s) * hidden
        copy_row[hidden](src0 + off, dst)
        if do_verify:
            mismatches += row_mismatch[hidden](src0 + off, alt + off)
    return mismatches


@always_inline
def steer_add_row[hidden: Int](x: BF16Ptr, v: BF16Ptr, alpha: Float32):
    def step[width: Int](idx: Int) {read}:
        var xv = (x + idx).load[width=width]().cast[DType.float32]()
        var vv = (v + idx).load[width=width]().cast[DType.float32]()
        var a = SIMD[DType.float32, width](alpha)
        (x + idx).store(vv.fma(a, xv).cast[DType.bfloat16]())

    vectorize[W](hidden, step)


@fieldwise_init
struct SteerAddKernel[hidden: Int](RangePartitionedKernel):
    var x: BF16Ptr
    var v: BF16Ptr
    var alpha: Float32
    var start: Int
    var end: Int

    def execute(mut self):
        for tok in range(self.start, self.end):
            steer_add_row[Self.hidden](
                self.x + tok * Self.hidden, self.v, self.alpha)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_steer_add[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    hidden: Int, max_worker_count: Int = 128,
](
    x: Binding[BFloat16, o],
    v: Binding[BFloat16, o],
    alpha: Float32,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime K = SteerAddKernel[hidden]

    @parameter
    def make(r: Int) -> K:
        return K(x[r], v[r], alpha, 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="steer_add"](
        pools, prof, seq_len, seq_len * hidden * 4,
        inline_threshold_bytes=SCALAR_MUL_INLINE_TOKENS * hidden * 4)


def apply_steer_ops[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    hidden: Int,
](
    mut steer: SteerState,
    steer_vectors: Binding[BFloat16, o],
    read schedule: Schedule,
    read buf_starts: List[Int],
    x_main: Binding[BFloat16, o],
    num_slots: Int,
    total: Int,
    layer: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if steer.per_request:
        for s in range(num_slots):
            var rid = schedule.slots[s].request_id
            if rid >= len(steer.req_ops):
                continue
            var sstart = buf_starts[s]
            var sn = schedule.slots[s].n_tokens
            for k in range(len(steer.req_ops[rid])):
                var op = steer.req_ops[rid][k]
                if op.layer == layer:
                    var vec = steer_vectors.shifted(op.vec_idx * hidden)
                    dispatch_steer_add[hidden=hidden](
                        x_main.shifted(sstart * hidden),
                        vec, op.alpha, sn, pools, prof)
    else:
        for k in range(len(steer.inject_ops)):
            var op = steer.inject_ops[k]
            if op.layer == layer:
                var vec = steer_vectors.shifted(op.vec_idx * hidden)
                dispatch_steer_add[hidden=hidden](
                    x_main, vec, op.alpha, total, pools, prof)
    var tp = steer.tap_index(layer)
    if tp >= 0:
        var sink = steer.sink_ptr()
        var mism = dispatch_steer_point[hidden=hidden](
            x_main, steer.last_rows, steer.last_num_slots, tp,
            sink, steer.max_slots, steer.verify_rank)
        steer.mismatch_count += mism
