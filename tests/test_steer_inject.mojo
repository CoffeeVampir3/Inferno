from std.pathlib import Path

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from tokenizer import load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform
from modeling_config import (
    Model, TOKENIZER_PATH, MODEL_DIR, stop_tokens, BOS_TOKEN_ID,
)
from modeling.gemma4_common import Gemma4BaseConfig
from inspectable_toolkit.steer import InjectOp
from kernels.flash_sample import SamplingParams
from continuous_batching.schedule import MAXIMUM_SAMPLING_LOGITS
from continuous_batching.scheduler import ContinuousBatchScheduler


comptime C = Gemma4BaseConfig
comptime STEP_BUDGET = Gemma4BaseConfig.SLIDING_WINDOW
comptime TAP_LAYER = 15
comptime VEC_FILL = 0.5
comptime ALPHA = 4.0


def encode(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read text: String,
) -> List[Int32]:
    var ids = List[Int32]()
    ids.append(Int32(BOS_TOKEN_ID))
    var enc = tok.encode(text)
    for i in range(len(enc)):
        ids.append(Int32(enc[i]))
    return ids^


def slot_for_rid(read requests: List[Int], n: Int, rid: Int) -> Int:
    for s in range(n):
        if requests[s] == rid:
            return s
    return -1


def run[
    P: BurstThreadPool, //,
](
    topo: NumaTopology,
    var pools: List[P],
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read prompt: String,
):
    var model_opt = Model[steer_vectors=16, Pool=P].load(Path(MODEL_DIR), topo, pools^)
    if not model_opt:
        print("model load failed")
        return
    var model = model_opt.take()
    print(t"loaded (degree {model.degree})")

    var greedy = SamplingParams(
        Float32(1.0), Float32(0.0), 0, 0, MAXIMUM_SAMPLING_LOGITS, True)
    var sched = ContinuousBatchScheduler[
        Model[steer_vectors=16, Pool=P].POSITIONS_PER_PAGE,
    ](model.batch_geometry(), STEP_BUDGET, stop_tokens())

    var tap = List[Int]()
    tap.append(TAP_LAYER)
    model.steer.arm(tap^, verify_rank=1)

    # baseline pass: tap at TAP_LAYER, no inject.
    var rid0 = sched.submit(encode(tok, prompt), greedy, 1).value()
    while len(sched.requests[rid0].generated) == 0:
        if sched.step(model) == 0:
            print("stalled on baseline prefill")
            return
    var s0 = slot_for_rid(
        model.steer.last_step_requests, model.steer.last_num_slots, rid0)
    var x0 = List[Float64](length=C.HIDDEN, fill=Float64(0))
    var p0 = model.steer.captured_ptr(0, s0)
    for i in range(C.HIDDEN):
        x0[i] = Float64(p0[i].cast[DType.float32]())

    # set steering vector 0 to a constant, then inject at TAP_LAYER.
    var vec = List[BFloat16](length=C.HIDDEN, fill=BFloat16(VEC_FILL))
    model.set_steer_vector(0, vec)
    var ops = List[InjectOp]()
    ops.append(InjectOp(TAP_LAYER, 0, Float32(ALPHA)))
    model.steer.set_inject(ops^)

    var rid1 = sched.submit(encode(tok, prompt), greedy, 1).value()
    while len(sched.requests[rid1].generated) == 0:
        if sched.step(model) == 0:
            print("stalled on injected prefill")
            return
    var s1 = slot_for_rid(
        model.steer.last_step_requests, model.steer.last_num_slots, rid1)

    var expected = Float64(ALPHA) * Float64(VEC_FILL)
    var p1 = model.steer.captured_ptr(0, s1)
    var sum_diff = Float64(0)
    var max_dev = Float64(0)
    for i in range(C.HIDDEN):
        var x1i = Float64(p1[i].cast[DType.float32]())
        var d = x1i - x0[i]
        sum_diff += d
        var dev = abs(d - expected)
        if dev > max_dev:
            max_dev = dev
    var mean_diff = sum_diff / Float64(C.HIDDEN)

    print()
    print(t"inject layer      : {TAP_LAYER}")
    print(t"alpha * v_fill     : {expected}")
    print(t"mean(x1 - x0)      : {mean_diff}")
    print(t"max |diff - exp|   : {max_dev}")
    if model.degree > 1:
        print(
            t"replication ok    : {model.steer.mismatch_count == 0}  "
            t"({model.steer.mismatch_count} mismatched elems)")
    else:
        print("replication ok    : n/a (single rank)")

    var shift_ok = abs(mean_diff - expected) < 0.05
    var repl_ok = model.steer.mismatch_count == 0
    if shift_ok and repl_ok:
        print("INJECT CHECK: PASS")
    else:
        print("INJECT CHECK: FAIL")


def main():
    print("steer inject test")
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print(t"failed to load tokenizer from {TOKENIZER_PATH}")
        return
    var tok = tok_opt.take()

    var prompt = String("The weather today is quite pleasant and mild.")

    var topo = NumaTopology()
    var nodes = topo.num_nodes()
    print(t"{nodes} NUMA nodes")

    @parameter
    def dispatch_tp[
        P: BurstThreadPool, //,
    ](var selected_pools: List[P]):
        run(topo, selected_pools^, tok, prompt)

    with_topological_rank_dispatch[
        dispatch=dispatch_tp,
    ](
        topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
