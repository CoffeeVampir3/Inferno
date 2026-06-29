from std.pathlib import Path
from std.time import perf_counter_ns

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from modeling.minimax_m3_bq import MinimaxM3


comptime MODEL_DIR = "checkpoints/minimax-m3-ablit"


def main():
    var source = String(MODEL_DIR)
    var output = source + "-bq/model.safetensors"
    print(t"source: {source}")
    print(t"output: {output}")

    var topo = NumaTopology()
    var nodes = topo.num_nodes()
    print(t"{nodes} NUMA nodes")

    @parameter
    def dispatch_quantize[
        P: BurstThreadPool, //,
    ](var pools: List[P]):
        var t0 = perf_counter_ns()
        var ok = MinimaxM3[Pool=P].quantize(
            Path(source), Path(output), topo, pools^)
        var elapsed_s = (perf_counter_ns() - t0) / 1_000_000_000
        if ok:
            print(t"quantize ok in {elapsed_s} s")
        else:
            print(t"quantize failed after {elapsed_s} s")

    with_topological_rank_dispatch[
        dispatch=dispatch_quantize,
    ](topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
