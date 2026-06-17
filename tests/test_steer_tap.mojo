from std.pathlib import Path

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from tokenizer import load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform
from kernels.helpers import BF16Ptr
from modeling_config import (
    Model, TOKENIZER_PATH, MODEL_DIR, stop_tokens, BOS_TOKEN_ID,
)
from modeling.gemma4_common import Gemma4BaseConfig
from kernels.flash_sample import SamplingParams
from continuous_batching.schedule import MAXIMUM_SAMPLING_LOGITS
from continuous_batching.scheduler import ContinuousBatchScheduler


comptime C = Gemma4BaseConfig
comptime STEP_BUDGET = Gemma4BaseConfig.SLIDING_WINDOW


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


def sumsq(p: BF16Ptr, n: Int) -> Float64:
    var acc = Float64(0)
    for i in range(n):
        var v = Float64(p[i].cast[DType.float32]())
        acc += v * v
    return acc


def approx_eq(a: Float64, b: Float64) -> Bool:
    var scale = max(abs(a), Float64(1))
    return abs(a - b) <= 1e-6 * scale


def relative_gap(a: Float64, b: Float64) -> Float64:
    var scale = max(abs(a), Float64(1))
    return abs(a - b) / scale


def run[
    P: BurstThreadPool, //,
](
    topo: NumaTopology,
    var pools: List[P],
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read prompts: List[String],
):
    var model_opt = Model[Pool=P].load(Path(MODEL_DIR), topo, pools^)
    if not model_opt:
        print("model load failed")
        return
    var model = model_opt.take()

    var layers = List[Int]()
    layers.append(10)
    layers.append(15)
    layers.append(20)
    model.steer.arm(layers^, verify_rank=1)
    print(t"armed taps at layers 10, 15, 20 (degree {model.degree})")

    var greedy = SamplingParams(
        Float32(1.0), Float32(0.0), 0, 0, MAXIMUM_SAMPLING_LOGITS, True)
    var sched = ContinuousBatchScheduler[
        Model[Pool=P].POSITIONS_PER_PAGE,
    ](model.batch_geometry(), STEP_BUDGET, stop_tokens())

    var rids = List[Int]()
    for i in range(len(prompts)):
        var ids = encode(tok, prompts[i])
        var rid = sched.submit(ids^, greedy, 1).value()
        rids.append(rid)
    print(t"submitted {len(prompts)} prompts")

    var n = sched.step(model)
    if n == 0:
        print("scheduler stalled on prefill")
        return

    var ntap = len(model.steer.tap_layers)
    var nslots = model.steer.last_num_slots
    print(t"captured {nslots} slots across {ntap} tap layers")
    print()

    for k in range(ntap):
        var layer = model.steer.tap_layers[k]
        for slot in range(nslots):
            var rid = model.steer.last_step_requests[slot]
            var energy = sumsq(model.steer.captured_ptr(k, slot), C.HIDDEN)
            print(t"layer {layer} | slot {slot} | rid {rid} | energy {energy}")
    print()

    var energy0 = List[Float64](length=len(prompts), fill=Float64(0))
    var seen = List[Bool](length=len(prompts), fill=False)
    for slot in range(nslots):
        var rid = model.steer.last_step_requests[slot]
        energy0[rid] = sumsq(model.steer.captured_ptr(0, slot), C.HIDDEN)
        seen[rid] = True

    var nonzero = True
    for slot in range(nslots):
        var rid = model.steer.last_step_requests[slot]
        if energy0[rid] <= Float64(0):
            nonzero = False

    var dup_match = (
        seen[rids[0]] and seen[rids[2]]
        and approx_eq(energy0[rids[0]], energy0[rids[2]])
    )
    var distinct = (
        seen[rids[0]] and seen[rids[1]]
        and relative_gap(energy0[rids[0]], energy0[rids[1]]) > 1e-3
    )

    var repl_ok = model.steer.mismatch_count == 0
    print(t"non-zero captures : {nonzero}")
    print(t"duplicate match   : {dup_match}  (rid {rids[0]} vs rid {rids[2]})")
    print(t"distinct prompts  : {distinct}  (rid {rids[0]} vs rid {rids[1]})")
    if model.degree > 1:
        print(
            t"replication ok    : {repl_ok}  (rank 0 vs rank 1, "
            t"{model.steer.mismatch_count} mismatched elems)")
    else:
        print("replication ok    : n/a (single rank, not exercised)")
    if nonzero and dup_match and distinct and repl_ok:
        print("TAP CHECK: PASS")
    else:
        print("TAP CHECK: FAIL")


def main():
    print("steer tap test")
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print(t"failed to load tokenizer from {TOKENIZER_PATH}")
        return
    var tok = tok_opt.take()

    var prompts = List[String]()
    prompts.append("I love loud parties and meeting new strangers.")
    prompts.append("I prefer a quiet evening alone with a good book.")
    prompts.append("I love loud parties and meeting new strangers.")

    var topo = NumaTopology()
    var nodes = topo.num_nodes()
    print(t"{nodes} NUMA nodes")

    @parameter
    def dispatch_tp[
        P: BurstThreadPool, //,
    ](var selected_pools: List[P]):
        run(topo, selected_pools^, tok, prompts)

    with_topological_rank_dispatch[
        dispatch=dispatch_tp,
    ](
        topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
