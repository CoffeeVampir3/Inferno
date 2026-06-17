from std.pathlib import Path

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from tokenizer import load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform
from modeling_config import (
    Model, TOKENIZER_PATH, MODEL_DIR, stop_tokens, format_prompt,
)
from modeling.gemma4_common import Gemma4BaseConfig
from inspectable_toolkit.slider_pack import load_pack
from kernels.flash_sample import SamplingParams
from continuous_batching.schedule import MAXIMUM_SAMPLING_LOGITS
from continuous_batching.scheduler import ContinuousBatchScheduler


comptime PACK_PATH = "sliders/ocean.json"
comptime STEP_BUDGET = Gemma4BaseConfig.SLIDING_WINDOW
comptime DEMO_NEW_TOKENS = 48
comptime POSITION = 0.8


def decode_int32(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read ids: List[Int32],
) -> String:
    var as_int = List[Int](capacity=len(ids))
    for i in range(len(ids)):
        as_int.append(Int(ids[i]))
    return tok.decode(as_int)


def run[
    P: BurstThreadPool, //,
](
    topo: NumaTopology,
    var pools: List[P],
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
):
    var model_opt = Model[steer_vectors=16, Pool=P].load(Path(MODEL_DIR), topo, pools^)
    if not model_opt:
        print("model load failed")
        return
    var model = model_opt.take()
    print(t"loaded (degree {model.degree})")

    var bank_opt = load_pack(model, PACK_PATH)
    if not bank_opt:
        print("pack load failed")
        return
    var bank = bank_opt.take()
    print(t"loaded pack with {bank.count()} slider(s)")
    var idx = bank.index_of("extraversion")
    if idx < 0:
        print("DRIVE CHECK: FAIL (extraversion slider absent)")
        return
    var cfg = bank.configs[idx]
    print(t"extraversion -> layer {cfg.layer} corridor "
          t"[{cfg.alpha_min}, {cfg.alpha_max}]")
    print()

    var greedy = SamplingParams(
        Float32(1.0), Float32(0.0), 0, 0, MAXIMUM_SAMPLING_LOGITS, True)
    var sched = ContinuousBatchScheduler[
        Model[steer_vectors=16, Pool=P].POSITIONS_PER_PAGE,
    ](model.batch_geometry(), STEP_BUDGET, stop_tokens())

    var demo = String("Tell me about your ideal weekend.")
    for variant in range(3):
        if variant == 0:
            bank.neutral()
            print("--- baseline (neutral) ---")
        elif variant == 1:
            _ = bank.set_position_by_name("extraversion", Float32(POSITION))
            print("--- +position (toward high) ---")
        else:
            _ = bank.set_position_by_name("extraversion", Float32(-POSITION))
            print("--- -position (toward low) ---")
        bank.apply(model)

        var rid = sched.submit(
            format_prompt(tok, demo), greedy, DEMO_NEW_TOKENS).value()
        var guard = 0
        while not sched.requests[rid].done:
            guard += 1
            if guard > 4 * DEMO_NEW_TOKENS:
                print("generation stalled")
                return
            if sched.step(model) == 0:
                print("scheduler stalled during generation")
                return
        print(decode_int32(tok, sched.requests[rid].generated))
        print()
        _ = sched.retire(rid)

    model.steer.disarm()
    print("DRIVE CHECK: PASS")


def main():
    print("steer drive test")
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print(t"failed to load tokenizer from {TOKENIZER_PATH}")
        return
    var tok = tok_opt.take()

    var topo = NumaTopology()
    var nodes = topo.num_nodes()
    print(t"{nodes} NUMA nodes")

    @parameter
    def dispatch_tp[
        P: BurstThreadPool, //,
    ](var selected_pools: List[P]):
        run(topo, selected_pools^, tok)

    with_topological_rank_dispatch[
        dispatch=dispatch_tp,
    ](
        topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
