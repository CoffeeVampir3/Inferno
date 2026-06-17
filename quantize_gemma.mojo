from std.pathlib import Path
from std.time import perf_counter_ns

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from modeling.gemma_4_moe_bq import Gemma4
from modeling_config import MODEL_DIR


def main():
    var source = String(MODEL_DIR)
    var output = source + "-bq/model.safetensors"
    print(t"source: {source}")
    print(t"output: {output}")

    var topo = NumaTopology()
    var nodes = topo.num_nodes()
    var iso = len(topo.isolated_cpus)
    print(t"{nodes} NUMA nodes, {iso} isolated cpus")

    @parameter
    def dispatch_quantize[
        P: BurstThreadPool, //,
    ](var pools: List[P]):
        var t0 = perf_counter_ns()
        var ok = Gemma4[Pool=P].quantize(
            Path(source), Path(output), topo, pools^)
        var elapsed_s = (perf_counter_ns() - t0) / 1_000_000_000
        if ok:
            print(t"quantize ok in {elapsed_s} s")
        else:
            print(t"quantize failed after {elapsed_s} s")

    with_topological_rank_dispatch[
        dispatch=dispatch_quantize,
    ](topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
