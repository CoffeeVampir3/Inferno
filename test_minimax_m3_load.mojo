from std.pathlib import Path
from std.time import perf_counter_ns

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from modeling.minimax_m3 import MinimaxM3


comptime MODEL_DIR = "checkpoints/minimax-m3"


def load_and_check[
    P: BurstThreadPool, //,
](
    topo: NumaTopology,
    var pools: List[P],
):
    var t0 = perf_counter_ns()
    var model_opt = MinimaxM3[Pool=P].load(Path(MODEL_DIR), topo, pools^)
    if not model_opt:
        print("load returned None")
        return
    var model = model_opt.take()
    var load_ns = perf_counter_ns() - t0
    var load_ms = Int(load_ns // 1_000_000)
    print(t"model loaded in {load_ms} ms")
    print()
    print("=== loaded weight spot checks (rank 0) ===")
    model.check_weights()


def main():
    print("minimax-m3 load check")
    var topo = NumaTopology()
    var nodes = topo.num_nodes()
    print(t"{nodes} NUMA nodes")

    @parameter
    def dispatch_m3_tp[
        P: BurstThreadPool, //,
    ](var selected_pools: List[P]):
        load_and_check(topo, selected_pools^)

    with_topological_rank_dispatch[
        dispatch=dispatch_m3_tp,
    ](
        topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
