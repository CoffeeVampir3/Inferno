from std.pathlib import Path

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from tokenizer import load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform
from modeling_config import (
    Model, TOKENIZER_PATH, MODEL_DIR, stop_tokens,
    BOS_TOKEN_ID, TURN_START_TOKEN_ID, TURN_END_TOKEN_ID, format_prompt,
)
from modeling.gemma4_common import Gemma4BaseConfig
from inspectable_toolkit.steer import InjectOp
from inspectable_toolkit.probe import (
    ContrastSet, ProbeResult, build_probe, mean_row_norm,
)
from kernels.helpers import BF16Ptr, F32Ptr
from kernels.flash_sample import SamplingParams
from continuous_batching.schedule import MAXIMUM_SAMPLING_LOGITS
from continuous_batching.scheduler import ContinuousBatchScheduler


comptime C = Gemma4BaseConfig
comptime STEP_BUDGET = Gemma4BaseConfig.SLIDING_WINDOW
comptime FIRST_TAP_LAYER = 5
comptime NUM_TAP_LAYERS = 20
comptime SAMPLES_PER_CLASS = 12
comptime WAVE_SIZE = 4
comptime ALPHA_NORM_FRACTION = 0.12
comptime DEMO_NEW_TOKENS = 32


def high_prompts() -> List[String]:
    var p = List[String]()
    p.append("I am the life of the party and I love it.")
    p.append("Meeting new people fills me with energy and excitement.")
    p.append("I love being the center of attention at big gatherings.")
    p.append("Talking to strangers is one of my favorite things.")
    p.append("I feel happiest surrounded by a loud crowd of friends.")
    p.append("I always start conversations with people I just met.")
    p.append("Big social events make me feel alive and energized.")
    p.append("I love telling stories to a room full of people.")
    p.append("Hosting huge parties for all my friends is my favorite hobby.")
    p.append("I get excited when my calendar is packed with social plans.")
    p.append("I thrive on noise, laughter, and busy social scenes.")
    p.append("Chatting with everyone in the room comes naturally to me.")
    return p^


def low_prompts() -> List[String]:
    var p = List[String]()
    p.append("I prefer to spend my evenings alone with a quiet book.")
    p.append("Large crowds drain my energy very quickly.")
    p.append("I avoid parties whenever I possibly can.")
    p.append("Speaking to strangers makes me uncomfortable and tired.")
    p.append("I feel most at peace in silence and solitude.")
    p.append("I rarely start conversations with people I do not know.")
    p.append("Big social events exhaust me for days afterward.")
    p.append("I would rather listen quietly than speak in a group.")
    p.append("Staying home alone on weekends is my favorite choice.")
    p.append("I keep my social calendar as empty as possible.")
    p.append("I find noise and busy gatherings overwhelming.")
    p.append("I prefer writing my thoughts down to saying them aloud.")
    return p^


comptime STATEMENT_USER = "Tell me about yourself and your ideal social life."


def encode_statement(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read statement: String,
) -> List[Int32]:
    var ids = List[Int32]()
    ids.append(Int32(BOS_TOKEN_ID))
    ids.append(Int32(TURN_START_TOKEN_ID))
    var user = tok.encode("user\n" + STATEMENT_USER)
    for i in range(len(user)):
        ids.append(Int32(user[i]))
    ids.append(Int32(TURN_END_TOKEN_ID))
    var sep = tok.encode("\n")
    for i in range(len(sep)):
        ids.append(Int32(sep[i]))
    ids.append(Int32(TURN_START_TOKEN_ID))
    var model_turn = tok.encode("model\n" + statement)
    for i in range(len(model_turn)):
        ids.append(Int32(model_turn[i]))
    return ids^


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

    var tap = List[Int](capacity=NUM_TAP_LAYERS)
    for k in range(NUM_TAP_LAYERS):
        tap.append(FIRST_TAP_LAYER + k)
    model.steer.arm(tap^)

    var greedy = SamplingParams(
        Float32(1.0), Float32(0.0), 0, 0, MAXIMUM_SAMPLING_LOGITS, True)
    var sched = ContinuousBatchScheduler[
        Model[steer_vectors=16, Pool=P].POSITIONS_PER_PAGE,
    ](model.batch_geometry(), STEP_BUDGET, stop_tokens())

    var dataset = List[ContrastSet[C.HIDDEN]](capacity=NUM_TAP_LAYERS)
    for _ in range(NUM_TAP_LAYERS):
        dataset.append(ContrastSet[C.HIDDEN](SAMPLES_PER_CLASS))

    var prompts = List[String]()
    var labels = List[Bool]()
    var highs = high_prompts()
    var lows = low_prompts()
    for i in range(len(highs)):
        prompts.append(highs[i])
        labels.append(True)
    for i in range(len(lows)):
        prompts.append(lows[i])
        labels.append(False)

    var cursor = 0
    while cursor < len(prompts):
        var wave_end = min(cursor + WAVE_SIZE, len(prompts))
        var wave_rids = List[Int]()
        var wave_labels = List[Bool]()
        var harvested = List[Bool]()
        for j in range(cursor, wave_end):
            var rid = sched.submit(
                encode_statement(tok, prompts[j]), greedy, 1).value()
            wave_rids.append(rid)
            wave_labels.append(labels[j])
            harvested.append(False)

        var guard = 0
        while True:
            var all_done = True
            for w in range(len(wave_rids)):
                if not sched.requests[wave_rids[w]].done:
                    all_done = False
            if all_done:
                break
            guard += 1
            if guard > 16:
                print("extraction stalled")
                return
            if sched.step(model) == 0:
                print("scheduler stalled during extraction")
                return
            for s in range(model.steer.last_num_slots):
                var rid = model.steer.last_step_requests[s]
                for w in range(len(wave_rids)):
                    if (
                        wave_rids[w] == rid
                        and not harvested[w]
                        and len(sched.requests[rid].generated) > 0
                    ):
                        for k in range(NUM_TAP_LAYERS):
                            dataset[k].add_row(
                                wave_labels[w],
                                model.steer.captured_ptr(k, s))
                        harvested[w] = True

        for w in range(len(wave_rids)):
            if not harvested[w]:
                print(t"sample not harvested in wave at {cursor}")
                return
            _ = sched.retire(wave_rids[w])
        cursor = wave_end

    print(
        t"extracted {dataset[0].n_high} high / {dataset[0].n_low} low "
        t"samples across {NUM_TAP_LAYERS} layers")
    print()

    var mean_high = List[Float32](length=C.HIDDEN, fill=Float32(0))
    var mean_low = List[Float32](length=C.HIDDEN, fill=Float32(0))
    var direction = List[BFloat16](length=C.HIDDEN, fill=BFloat16(0))
    var best_direction = List[BFloat16](length=C.HIDDEN, fill=BFloat16(0))
    var mh_ptr: F32Ptr = mean_high.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var ml_ptr: F32Ptr = mean_low.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var dir_ptr: BF16Ptr = direction.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()

    var best = ProbeResult(
        -1, Float64(0), Float64(0), Float64(0), Float64(0),
        Float64(0), Float64(0))
    var best_k = -1
    var all_valid = True
    print("layer | fisher ratio | separation | mu_high | mu_low")
    for k in range(NUM_TAP_LAYERS):
        var r = build_probe[C.HIDDEN](
            dataset[k], model.steer.tap_layers[k], mh_ptr, ml_ptr, dir_ptr)
        print(
            t"{r.layer} | {r.fr} | {r.separation} "
            t"| {r.mean_high} | {r.mean_low}")
        if not (r.fr > Float64(0) and r.separation > Float64(0)):
            all_valid = False
        if r.mean_high - r.mean_low <= Float64(0):
            all_valid = False
        if r.fr > best.fr:
            best = r
            best_k = k
            for i in range(C.HIDDEN):
                best_direction[i] = direction[i]
    print()
    print(t"selected layer {best.layer} (fisher ratio {best.fr})")

    if not all_valid or best_k < 0:
        print("PROBE CHECK: FAIL")
        return
    print("PROBE CHECK: PASS")
    print()

    var residual_norm = mean_row_norm[C.HIDDEN](dataset[best_k])
    var alpha = Float32(ALPHA_NORM_FRACTION * residual_norm)
    print(t"mean residual norm {residual_norm} -> alpha {alpha}")
    model.set_steer_vector(0, best_direction)

    var demo = String("Tell me about your ideal weekend.")
    for variant in range(3):
        if variant == 0:
            model.steer.disarm()
            print("--- baseline ---")
        elif variant == 1:
            var ops = List[InjectOp]()
            ops.append(InjectOp(best.layer, 0, alpha))
            model.steer.set_inject(ops^)
            print("--- steered (+alpha, toward high) ---")
        else:
            var ops = List[InjectOp]()
            ops.append(InjectOp(best.layer, 0, -alpha))
            model.steer.set_inject(ops^)
            print("--- steered (-alpha, toward low) ---")

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


def main():
    print("steer probe test")
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
