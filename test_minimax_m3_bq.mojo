from std.pathlib import Path
from std.time import perf_counter_ns

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from tokenizer import load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform
from kernels.flash_sample import SamplingParams
from continuous_batching.schedule import MAXIMUM_SAMPLING_LOGITS
from continuous_batching.scheduler import ContinuousBatchScheduler

from modeling.minimax_m3_bq import MinimaxM3, PAGE_LEN


comptime MODEL_DIR = "checkpoints/minimax-m3-bq"
comptime TOKENIZER_PATH = "checkpoints/minimax-m3-ablit/tokenizer.json"
comptime MAX_NEW_TOKENS = 128
comptime STEP_BUDGET = PAGE_LEN

comptime BOD_TOKEN_ID = 200034
comptime BOS_TOKEN_ID = 200019
comptime EOS_TOKEN_ID = 200020


def elapsed_ms_since(start_ns: UInt) -> Int:
    return Int((perf_counter_ns() - start_ns) / 1_000_000)


def tokens_per_second(token_count: Int, elapsed_ms: Int) -> Int:
    if elapsed_ms == 0:
        return 0
    return token_count * 1000 // elapsed_ms


def append_encoded(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    mut token_ids: List[Int32],
    text: String,
):
    var encoded = tok.encode(text)
    for i in range(len(encoded)):
        token_ids.append(Int32(encoded[i]))


def format_prompt(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    prompt: String,
) -> List[Int32]:
    var token_ids = List[Int32]()
    token_ids.append(Int32(BOD_TOKEN_ID))
    token_ids.append(Int32(BOS_TOKEN_ID))
    append_encoded(tok, token_ids, "user\n" + prompt)
    token_ids.append(Int32(EOS_TOKEN_ID))
    append_encoded(tok, token_ids, "\n")
    token_ids.append(Int32(BOS_TOKEN_ID))
    append_encoded(tok, token_ids, "ai\n")
    return token_ids^


def stop_tokens() -> List[Int32]:
    var ids = List[Int32]()
    ids.append(Int32(EOS_TOKEN_ID))
    return ids^


def decode_int32(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read ids: List[Int32],
) -> String:
    var as_int = List[Int](capacity=len(ids))
    for i in range(len(ids)):
        as_int.append(Int(ids[i]))
    return tok.decode(as_int)


def print_prompt(prompt: String, read token_ids: List[Int32]):
    var prompt_repr = repr(prompt)
    print(t"prompt: {prompt_repr}")
    var n_tokens = len(token_ids)
    print(t"tokens: {n_tokens} ids:", end="")
    for i in range(n_tokens):
        print("", token_ids[i], end="")
    print()


def load_and_run[
    P: BurstThreadPool, //,
](
    topo: NumaTopology,
    var pools: List[P],
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read token_ids: List[Int32],
):
    var t0 = perf_counter_ns()
    var model_opt = MinimaxM3[profile=True, Pool=P].load(
        Path(MODEL_DIR), topo, pools^)
    if not model_opt:
        return
    var model = model_opt.take()
    var load_ms = elapsed_ms_since(t0)
    print(t"model loaded in {load_ms} ms")
    print()

    var greedy = SamplingParams(
        Float32(1.0), Float32(0.0), 0, 0, MAXIMUM_SAMPLING_LOGITS, True)
    var sched = ContinuousBatchScheduler[
        MinimaxM3[profile=True, Pool=P].POSITIONS_PER_PAGE,
    ](model.batch_geometry(), STEP_BUDGET, stop_tokens())

    var prompt_tokens = List[Int32](capacity=len(token_ids))
    for i in range(len(token_ids)):
        prompt_tokens.append(token_ids[i])
    var request_id = sched.submit(prompt_tokens^, greedy, MAX_NEW_TOKENS).value()

    var prompt_len = len(token_ids)
    var t1 = perf_counter_ns()
    var prefill_ms = 0
    while len(sched.requests[request_id].generated) == 0:
        if sched.step(model) == 0:
            print("scheduler stalled during prefill")
            return
        prefill_ms = elapsed_ms_since(t1)
    model.profiler.report("prefill")
    model.profiler.reset()

    var decode_start = perf_counter_ns()
    while sched.pending_work():
        if sched.step(model) == 0:
            print("scheduler stalled during decode")
            return
    model.profiler.report("decode")

    var prefill_tps = tokens_per_second(prompt_len, prefill_ms)
    print(t"prompt  | {prompt_len} tokens | {prefill_ms} ms | {prefill_tps} t/s")

    var decode_elapsed_ms = elapsed_ms_since(decode_start)
    var decode_tokens = len(sched.requests[request_id].generated) - 1
    var decode_tps = tokens_per_second(decode_tokens, decode_elapsed_ms)
    print(t"decode  | {decode_tokens} tokens | {decode_elapsed_ms} ms | {decode_tps} t/s")

    var n_generated = len(sched.requests[request_id].generated)
    var full_text = decode_int32(tok, sched.requests[request_id].tokens)
    print()
    print(t"=== generated {n_generated} tokens ===")
    print(full_text)


def main():
    print("Launching.")
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print(t"failed to load tokenizer from {TOKENIZER_PATH}")
        return
    var tok = tok_opt.take()

    var prompt = """The Roman aqueducts were a system of engineering structures built by the ancient Romans to transport water from distant sources into cities and towns. Constructed from a combination of stone, brick, and a special volcanic cement known as pozzolana, these channels supplied public baths, latrines, fountains, and private households across the empire. The water flowed largely by gravity alone, descending along a very gentle gradient maintained over distances that sometimes exceeded a hundred kilometres.

The earliest aqueduct in Rome, the Aqua Appia, was commissioned in 312 BC and ran almost entirely underground to protect it from contamination and enemy sabotage. As the city's population grew, later aqueducts such as the Aqua Marcia and the Aqua Claudia carried far greater volumes and rose onto towering arched bridges where the terrain dipped. At the height of the empire, eleven major aqueducts served the capital, together delivering an estimated one million cubic metres of water each day.

Beyond Rome itself, provincial cities throughout Gaul, Hispania, and North Africa built their own aqueducts, many of which still stand today. The Pont du Gard in southern France and the aqueduct of Segovia in Spain remain among the best preserved, their multi-tiered arches a testament to the durability of Roman construction. Maintenance was the responsibility of a dedicated office, and a permanent staff of workers inspected the channels, cleared sediment, and repaired leaks.

The decline of the aqueduct network paralleled the broader collapse of Roman administrative power in the West. As central authority weakened, the resources and expertise needed to maintain the channels disappeared, and many fell into disrepair or were deliberately cut during sieges. Nevertheless, the underlying principles of gradient flow and durable masonry influenced water engineering for centuries, and several aqueducts were restored and returned to service during the Renaissance.

The Roman aqueducts were a system of engineering structures built by the ancient Romans to transport water from distant sources into cities and towns. Constructed from a combination of stone, brick, and a special volcanic cement known as pozzolana, these channels supplied public baths, latrines, fountains, and private households across the empire. The water flowed largely by gravity alone, descending along a very gentle gradient maintained over distances that sometimes exceeded a hundred kilometres.

The earliest aqueduct in Rome, the Aqua Appia, was commissioned in 312 BC and ran almost entirely underground to protect it from contamination and enemy sabotage. As the city's population grew, later aqueducts such as the Aqua Marcia and the Aqua Claudia carried far greater volumes and rose onto towering arched bridges where the terrain dipped. At the height of the empire, eleven major aqueducts served the capital, together delivering an estimated one million cubic metres of water each day.

Beyond Rome itself, provincial cities throughout Gaul, Hispania, and North Africa built their own aqueducts, many of which still stand today. The Pont du Gard in southern France and the aqueduct of Segovia in Spain remain among the best preserved, their multi-tiered arches a testament to the durability of Roman construction. Maintenance was the responsibility of a dedicated office, and a permanent staff of workers inspected the channels, cleared sediment, and repaired leaks.

The decline of the aqueduct network paralleled the broader collapse of Roman administrative power in the West. As central authority weakened, the resources and expertise needed to maintain the channels disappeared, and many fell into disrepair or were deliberately cut during sieges. Nevertheless, the underlying principles of gradient flow and durable masonry influenced water engineering for centuries, and several aqueducts were restored and returned to service during the Renaissance.

The Roman aqueducts were a system of engineering structures built by the ancient Romans to transport water from distant sources into cities and towns. Constructed from a combination of stone, brick, and a special volcanic cement known as pozzolana, these channels supplied public baths, latrines, fountains, and private households across the empire. The water flowed largely by gravity alone, descending along a very gentle gradient maintained over distances that sometimes exceeded a hundred kilometres.

The earliest aqueduct in Rome, the Aqua Appia, was commissioned in 312 BC and ran almost entirely underground to protect it from contamination and enemy sabotage. As the city's population grew, later aqueducts such as the Aqua Marcia and the Aqua Claudia carried far greater volumes and rose onto towering arched bridges where the terrain dipped. At the height of the empire, eleven major aqueducts served the capital, together delivering an estimated one million cubic metres of water each day.

Beyond Rome itself, provincial cities throughout Gaul, Hispania, and North Africa built their own aqueducts, many of which still stand today. The Pont du Gard in southern France and the aqueduct of Segovia in Spain remain among the best preserved, their multi-tiered arches a testament to the durability of Roman construction. Maintenance was the responsibility of a dedicated office, and a permanent staff of workers inspected the channels, cleared sediment, and repaired leaks.

The decline of the aqueduct network paralleled the broader collapse of Roman administrative power in the West. As central authority weakened, the resources and expertise needed to maintain the channels disappeared, and many fell into disrepair or were deliberately cut during sieges. Nevertheless, the underlying principles of gradient flow and durable masonry influenced water engineering for centuries, and several aqueducts were restored and returned to service during the Renaissance.

The Roman aqueducts were a system of engineering structures built by the ancient Romans to transport water from distant sources into cities and towns. Constructed from a combination of stone, brick, and a special volcanic cement known as pozzolana, these channels supplied public baths, latrines, fountains, and private households across the empire. The water flowed largely by gravity alone, descending along a very gentle gradient maintained over distances that sometimes exceeded a hundred kilometres.

The earliest aqueduct in Rome, the Aqua Appia, was commissioned in 312 BC and ran almost entirely underground to protect it from contamination and enemy sabotage. As the city's population grew, later aqueducts such as the Aqua Marcia and the Aqua Claudia carried far greater volumes and rose onto towering arched bridges where the terrain dipped. At the height of the empire, eleven major aqueducts served the capital, together delivering an estimated one million cubic metres of water each day.

Beyond Rome itself, provincial cities throughout Gaul, Hispania, and North Africa built their own aqueducts, many of which still stand today. The Pont du Gard in southern France and the aqueduct of Segovia in Spain remain among the best preserved, their multi-tiered arches a testament to the durability of Roman construction. Maintenance was the responsibility of a dedicated office, and a permanent staff of workers inspected the channels, cleared sediment, and repaired leaks.

The decline of the aqueduct network paralleled the broader collapse of Roman administrative power in the West. As central authority weakened, the resources and expertise needed to maintain the channels disappeared, and many fell into disrepair or were deliberately cut during sieges. Nevertheless, the underlying principles of gradient flow and durable masonry influenced water engineering for centuries, and several aqueducts were restored and returned to service during the Renaissance."""
    var token_ids = format_prompt(tok, prompt)
    print_prompt(prompt, token_ids)

    var topo = NumaTopology()
    var nodes = topo.num_nodes()
    var iso = len(topo.isolated_cpus)
    print(t"{nodes} NUMA nodes, {iso} isolated cpus")

    @parameter
    def dispatch_minimax_m3_tp[
        P: BurstThreadPool, //,
    ](var selected_pools: List[P]):
        load_and_run(topo, selected_pools^, tok, token_ids)

    with_topological_rank_dispatch[
        dispatch=dispatch_minimax_m3_tp,
    ](
        topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
