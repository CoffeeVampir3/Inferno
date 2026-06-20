from std.pathlib import Path
from std.time import perf_counter_ns

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from kernels.flash_sample import SamplingParams
from continuous_batching.schedule import MAXIMUM_SAMPLING_LOGITS
from continuous_batching.scheduler import ContinuousBatchScheduler

from modeling.minimax_m3 import MinimaxM3, PAGE_LEN


comptime MODEL_DIR = "checkpoints/minimax-m3"
comptime MAX_NEW_TOKENS = 8
comptime STEP_BUDGET = PAGE_LEN


def load_and_run[
    P: BurstThreadPool, //,
](
    topo: NumaTopology,
    var pools: List[P],
):
    var t0 = perf_counter_ns()
    var model_opt = MinimaxM3[profile=True, Pool=P].load(
        Path(MODEL_DIR), topo, pools^)
    if not model_opt:
        print("load returned None")
        return
    var model = model_opt.take()
    var load_ms = Int((perf_counter_ns() - t0) // 1_000_000)
    print(t"model loaded in {load_ms} ms")

    var greedy = SamplingParams(
        Float32(1.0), Float32(0.0), 0, 0, MAXIMUM_SAMPLING_LOGITS, True)
    var sched = ContinuousBatchScheduler[
        MinimaxM3[profile=True, Pool=P].POSITIONS_PER_PAGE,
    ](model.batch_geometry(), STEP_BUDGET, Int32(-1))

    # Raw token ids (no tokenizer dependency) -- this is a structural smoke
    # test of the forward, not a quality check.
    var prompt = List[Int32]()
    for t in range(16):
        prompt.append(Int32(100 + t))
    var prompt_len = len(prompt)
    var request_id = sched.submit(prompt^, greedy, MAX_NEW_TOKENS).value()

    var t1 = perf_counter_ns()
    while len(sched.requests[request_id].generated) == 0:
        if sched.step(model) == 0:
            print("scheduler stalled during prefill")
            return
    var prefill_ms = Int((perf_counter_ns() - t1) // 1_000_000)
    model.profiler.report("prefill")

    while sched.pending_work():
        if sched.step(model) == 0:
            print("scheduler stalled during decode")
            return

    print(t"prompt {prompt_len} tokens | prefill {prefill_ms} ms")
    var n = len(sched.requests[request_id].generated)
    print(t"=== generated {n} tokens ===")
    for i in range(n):
        print("", sched.requests[request_id].generated[i], end="")
    print()


def main():
    print("minimax-m3 forward smoke")
    var topo = NumaTopology()
    var nodes = topo.num_nodes()
    print(t"{nodes} NUMA nodes")

    @parameter
    def dispatch_m3_tp[
        P: BurstThreadPool, //,
    ](var selected_pools: List[P]):
        load_and_run(topo, selected_pools^)

    with_topological_rank_dispatch[
        dispatch=dispatch_m3_tp,
    ](
        topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
