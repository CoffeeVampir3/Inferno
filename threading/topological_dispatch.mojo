from numa import NumaTopology
from .burst_threading import BurstPool
from .isolated_burst_pool import IsolatedBurstPool
from .threading_traits import BurstThreadPool


def dispatch_topological_rank_pools[
    P: BurstThreadPool,
    //,
    *,
    dispatch: def[Q: BurstThreadPool, //](
        var List[Q]
    ) capturing [_] -> None,
](
    read topo: NumaTopology,
    read mode: String,
    var pools: List[P],
):
    var tp = len(topo)
    print(mode)
    for i in range(tp):
        print(t"  node {topo.node(i)}: {pools[i].get_capacity()} workers")
    print("")
    dispatch(pools^)


def with_topological_rank_dispatch[
    *,
    dispatch: def[P: BurstThreadPool, //](
        var List[P]
    ) capturing [_] -> None,
](
    read topo: NumaTopology,
    read isolated_mode: String,
    read cold_mode: String,
):
    if topo.has_isolation():
        var pools = List[IsolatedBurstPool[]](capacity=len(topo))
        for i in range(len(topo)):
            pools.append(IsolatedBurstPool[].for_rank(topo, i))
        dispatch_topological_rank_pools[dispatch=dispatch](
            topo, isolated_mode, pools^)
    else:
        var pools = List[BurstPool[]](capacity=len(topo))
        for i in range(len(topo)):
            pools.append(BurstPool[].for_rank(topo, i))
        dispatch_topological_rank_pools[dispatch=dispatch](
            topo, cold_mode, pools^)
