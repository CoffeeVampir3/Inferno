from std.pathlib import Path
from std.time import perf_counter_ns

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from tokenizer import load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform
from modeling_config import (
    Model, TOKENIZER_PATH, MODEL_DIR, stop_tokens, format_prompt,
)
from modeling.gemma4_common import Gemma4BaseConfig
from kernels.flash_sample import SamplingParams
from continuous_batching.schedule import MAXIMUM_SAMPLING_LOGITS
from continuous_batching.scheduler import ContinuousBatchScheduler
from test_sequences import sample_prompts


comptime MAX_NEW_TOKENS = 128
comptime STEP_BUDGET = Gemma4BaseConfig.SLIDING_WINDOW


def ns_to_ms(elapsed_ns: Int) -> Int:
    return elapsed_ns // 1_000_000


def tokens_per_second(token_count: Int, elapsed_ns: Int) -> Int:
    if elapsed_ns == 0:
        return 0
    return token_count * 1_000_000_000 // elapsed_ns


def decode_int32(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read ids: List[Int32],
) -> String:
    var as_int = List[Int](capacity=len(ids))
    for i in range(len(ids)):
        as_int.append(Int(ids[i]))
    return tok.decode(as_int)


def load_and_run[
    P: BurstThreadPool, //,
](
    topo: NumaTopology,
    var pools: List[P],
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read prompts: List[List[Int32]],
):
    var t0 = perf_counter_ns()
    var model_opt = Model[profile=True, Pool=P].load(
        Path(MODEL_DIR), topo, pools^)
    if not model_opt:
        return
    var model = model_opt.take()
    var load_ms = ns_to_ms(Int(perf_counter_ns() - t0))
    print(t"model loaded in {load_ms} ms")

    var geometry = model.batch_geometry()
    var max_seqs = geometry.max_seqs
    var num_requests = len(prompts)
    print(t"{num_requests} requests over {max_seqs} resident sequences")
    print()

    var greedy = SamplingParams(
        Float32(1.0), Float32(0.0), 0, 0, MAXIMUM_SAMPLING_LOGITS, True)
    var sched = ContinuousBatchScheduler[
        Model[profile=True, Pool=P].POSITIONS_PER_PAGE,
    ](geometry, STEP_BUDGET, stop_tokens())

    var request_ids = List[Int](capacity=num_requests)
    var total_prompt_tokens = 0
    for k in range(num_requests):
        var toks = List[Int32](capacity=len(prompts[k]))
        for t in range(len(prompts[k])):
            toks.append(prompts[k][t])
        total_prompt_tokens += len(prompts[k])
        request_ids.append(sched.submit(toks^, greedy, MAX_NEW_TOKENS).value())

    var steps = 0
    var prefill_steps = 0
    var decode_steps = 0
    var mixed_steps = 0
    var prefill_ns = 0
    var decode_ns = 0
    var mixed_ns = 0
    var prefill_bucket_tokens = 0
    var decode_bucket_tokens = 0
    var mixed_prefill_tokens = 0
    var mixed_decode_tokens = 0
    var total_prefill_tokens = 0
    var total_decode_tokens = 0
    var run_t0 = perf_counter_ns()
    while sched.pending_work():
        var step_t0 = perf_counter_ns()
        if sched.step(model) == 0:
            print("scheduler stalled with work pending")
            return
        var step_ns = Int(perf_counter_ns() - step_t0)
        steps += 1
        var pf = sched.schedule.prefill_tokens
        var dc = sched.schedule.decode_tokens
        total_prefill_tokens += pf
        total_decode_tokens += dc
        if dc == 0:
            prefill_steps += 1
            prefill_ns += step_ns
            prefill_bucket_tokens += pf
        elif pf == 0:
            decode_steps += 1
            decode_ns += step_ns
            decode_bucket_tokens += dc
        else:
            mixed_steps += 1
            mixed_ns += step_ns
            mixed_prefill_tokens += pf
            mixed_decode_tokens += dc
    var run_ns = Int(perf_counter_ns() - run_t0)

    for k in range(num_requests):
        var rid = request_ids[k]
        var n_generated = len(sched.requests[rid].generated)
        print(t"=== request {k}: {n_generated} predicted tokens ===")
        print(decode_int32(tok, sched.requests[rid].generated))
        print()

    var run_ms = ns_to_ms(run_ns)
    print(t"steps   | {steps} total | {prefill_steps} prefill | {mixed_steps} mixed | {decode_steps} decode | {run_ms} ms")
    var prefill_ms = ns_to_ms(prefill_ns)
    var prefill_tps = tokens_per_second(prefill_bucket_tokens, prefill_ns)
    print(t"prefill | {prefill_bucket_tokens} tokens | {prefill_ms} ms | {prefill_tps} t/s")
    if mixed_steps > 0:
        var mixed_ms = ns_to_ms(mixed_ns)
        var mixed_total = mixed_prefill_tokens + mixed_decode_tokens
        var mixed_tps = tokens_per_second(mixed_total, mixed_ns)
        print(t"mixed   | {mixed_prefill_tokens} prefill + {mixed_decode_tokens} decode tokens | {mixed_ms} ms | {mixed_tps} t/s")
    var decode_ms = ns_to_ms(decode_ns)
    var decode_tps = tokens_per_second(decode_bucket_tokens, decode_ns)
    print(t"decode  | {decode_bucket_tokens} tokens | {decode_ms} ms | {decode_tps} t/s")
    var total_tps = tokens_per_second(
        total_prefill_tokens + total_decode_tokens, run_ns)
    print(t"total   | {total_prefill_tokens} prefill + {total_decode_tokens} decode tokens | {total_tps} t/s")
    print()
    model.profiler.report("continuous batching")


def main():
    print("Continuous-batching hard test")
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print(t"failed to load tokenizer from {TOKENIZER_PATH}")
        return
    var tok = tok_opt.take()

    var raw = sample_prompts()
    var prompts = List[List[Int32]]()
    for k in range(len(raw)):
        prompts.append(format_prompt(tok, raw[k]))
        var n_tokens = len(prompts[k])
        print(t"sequence {k}: {n_tokens} prompt tokens")

    var topo = NumaTopology()
    var nodes = topo.num_nodes()
    var iso = len(topo.isolated_cpus)
    print(t"{nodes} NUMA nodes, {iso} isolated cpus")

    @parameter
    def dispatch_gemma4_cb[
        P: BurstThreadPool, //,
    ](var selected_pools: List[P]):
        load_and_run(topo, selected_pools^, tok, prompts)

    with_topological_rank_dispatch[
        dispatch=dispatch_gemma4_cb,
    ](
        topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
