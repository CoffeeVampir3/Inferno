from std.pathlib import Path
from std.memory import Span, UnsafePointer

from modeling.model_spec import WeightDesc
from safetensors.parser import parse_safetensors_header, SafetensorsHeader, TensorMeta
from linux.io_uring import ReadOp, run_reads_multi
from threading.burst_threading import BurstPool
from numa import NumaTopology


comptime DEFAULT_IO_DEPTH = 2048
comptime DEFAULT_MASK_SIZE = 128


def discover_shards(path: Path) -> List[Path]:
    """Enumerate safetensors shard paths, sorted by name.

    Accepts either:
      - a directory containing one or more *.safetensors files
      - a direct *.safetensors file path

    This covers multi-shard HF checkpoints (model-00001-of-000NN.safetensors),
    single-file HF checkpoints (model.safetensors), and single-file quantizer
    outputs.
    """
    var shards = List[Path]()
    if String(path).endswith(".safetensors"):
        shards.append(path)
        return shards^

    var names = List[String]()
    try:
        for entry in path.listdir():
            var name = String(entry)
            if name.endswith(".safetensors"):
                shards.append(path / name)
                names.append(name^)
    except:
        pass
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if names[j] < names[i]:
                var tmp_name = names[i]
                names[i] = names[j]
                names[j] = tmp_name
                var tmp_shard = shards[i]
                shards[i] = shards[j]
                shards[j] = tmp_shard
    return shards^


def validate_weight(
    desc: WeightDesc, found_dtype: DType, found_shape: List[Int],
) -> Bool:
    if desc.dtype != found_dtype:
        print(
            t"dtype mismatch for {desc.name}: expected {desc.dtype} "
            t"got {found_dtype}"
        )
        return False

    if len(found_shape) == 1:
        var expected = desc.global_rows * desc.global_cols
        var got = found_shape[0]
        if expected != got:
            print(t"shape mismatch for {desc.name}: expected [{expected}] got [{got}]")
            return False
    elif len(found_shape) == 2:
        if desc.global_rows != found_shape[0] or desc.global_cols != found_shape[1]:
            var s0 = found_shape[0]
            var s1 = found_shape[1]
            print(
                t"shape mismatch for {desc.name}: "
                t"expected [{desc.global_rows}, {desc.global_cols}] "
                t"got [{s0}, {s1}]"
            )
            return False
    elif len(found_shape) == 3:
        var folded_rows = found_shape[0] * found_shape[1]
        if desc.global_rows != folded_rows or desc.global_cols != found_shape[2]:
            var s0 = found_shape[0]
            var s1 = found_shape[1]
            var s2 = found_shape[2]
            print(
                t"shape mismatch for {desc.name}: "
                t"expected [{desc.global_rows}, {desc.global_cols}] "
                t"got [{s0}, {s1}, {s2}]"
            )
            return False
    else:
        var rank = len(found_shape)
        print(t"unexpected rank for {desc.name}: {rank}")
        return False

    return True


def emit_reads(
    desc: WeightDesc,
    file_idx: Int,
    file_data_start: Int,
    arena_base: Int,
    rank: Int,
    mut ops: List[ReadOp[]],
):
    """Append ReadOps for the given weight into `ops`. Each op's id is its
    local index within `ops` — the ring uses id as a direct lookup index."""
    var dest = arena_base + desc.arena_offset

    if desc.data_rows == desc.global_rows and desc.data_cols == desc.global_cols:
        var data_bytes = desc.data_rows * desc.data_cols * desc.element_bytes
        ops.append(ReadOp(
            file_idx=file_idx, offset=file_data_start, length=data_bytes,
            dest=UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dest),
            id=len(ops),
        ))
    elif desc.data_rows != desc.global_rows:
        var row_start = rank * desc.data_rows
        var real_rows = desc.global_rows - row_start
        if real_rows > desc.data_rows:
            real_rows = desc.data_rows
        if real_rows > 0:
            var data_bytes = real_rows * desc.global_cols * desc.element_bytes
            var file_off = file_data_start + row_start * desc.global_cols * desc.element_bytes
            ops.append(ReadOp(
                file_idx=file_idx, offset=file_off, length=data_bytes,
                dest=UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dest),
                id=len(ops),
            ))
    else:
        var file_cols = desc.data_cols
        var stride_cols = desc.local_cols
        var col_start = rank * file_cols
        var real_cols = desc.global_cols - col_start
        if real_cols > file_cols:
            real_cols = file_cols
        if real_cols > 0:
            var real_row_bytes = real_cols * desc.element_bytes
            var stride_bytes = stride_cols * desc.element_bytes
            for r in range(desc.data_rows):
                var src = file_data_start + (r * desc.global_cols + col_start) * desc.element_bytes
                var dst = dest + r * stride_bytes
                ops.append(ReadOp(
                    file_idx=file_idx, offset=src, length=real_row_bytes,
                    dest=UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst),
                    id=len(ops),
                ))


@fieldwise_init
struct LocatedTensor(Copyable):
    var shard: Int
    var meta: TensorMeta


def find_tensor(
    name: String,
    ref headers: List[SafetensorsHeader],
) -> Optional[LocatedTensor]:
    for i in range(len(headers)):
        var meta_opt = headers[i].tensors.get(name)
        if meta_opt:
            return LocatedTensor(i, meta_opt.value().copy())
    return None


def resolve_and_emit(
    w: WeightDesc,
    ref headers: List[SafetensorsHeader],
    arena_bases: List[Int],
    ranks: List[Int],
    mut ops_per_rank: List[List[ReadOp[]]],
) -> Bool:
    var found = find_tensor(w.name, headers)
    if not found:
        print(t"missing tensor: {w.name}")
        return False
    var loc = found.value().copy()
    if not validate_weight(w, loc.meta.dtype, loc.meta.shape):
        return False
    var data_start = headers[loc.shard].data_offset + loc.meta.start
    for i in range(len(ranks)):
        var r = ranks[i]
        emit_reads(w, loc.shard, data_start, arena_bases[r], r, ops_per_rank[r])
    return True


@fieldwise_init
struct LoadResult(Movable):
    var bytes_loaded: Int
    var num_ops: Int


def load_weights_from_descs[
    io_depth: Int = DEFAULT_IO_DEPTH,
    mask_size: Int = DEFAULT_MASK_SIZE,
](
    descs: List[WeightDesc],
    paths: List[Path],
    arena_bases: List[Int],
    topo: NumaTopology,
) -> Optional[LoadResult]:
    """Runtime variant — takes a prebuilt List[WeightDesc]."""
    var headers = List[SafetensorsHeader](capacity=len(paths))
    for i in range(len(paths)):
        var header_opt = parse_safetensors_header(paths[i])
        if not header_opt:
            var p = paths[i]
            print(t"failed to parse: {p}")
            return None
        headers.append(header_opt.take())

    var targeted_weights = List[WeightDesc]()
    var distributed_weights = List[WeightDesc]()
    for i in range(len(descs)):
        var d = descs[i].copy()
        if d.target_rank >= 0:
            targeted_weights.append(d^)
        else:
            distributed_weights.append(d^)

    var tp = len(arena_bases)
    var ops_per_rank = List[List[ReadOp[]]]()
    for _ in range(tp):
        ops_per_rank.append(List[ReadOp[]]())

    var all_ranks = List[Int]()
    for r in range(tp):
        all_ranks.append(r)

    for w in targeted_weights:
        var ranks = List[Int]()
        ranks.append(w.target_rank % tp)
        if not resolve_and_emit(w, headers, arena_bases, ranks, ops_per_rank):
            return None
    for w in distributed_weights:
        if not resolve_and_emit(w, headers, arena_bases, all_ranks, ops_per_rank):
            return None

    return run_load[io_depth, mask_size](paths, topo, ops_per_rank^)


def run_load[
    io_depth: Int,
    mask_size: Int,
](
    paths: List[Path],
    topo: NumaTopology,
    var ops_per_rank: List[List[ReadOp[]]],
) -> Optional[LoadResult]:
    """Build transient load pools (one 1-capacity pool per rank's NUMA
    node), run the multi-pool read dispatch, tally bytes/ops. The load
    pools are destroyed on return; the inference-time pools are a
    separate concern of the caller."""
    var total_bytes = 0
    var total_ops = 0
    for r in range(len(ops_per_rank)):
        for i in range(len(ops_per_rank[r])):
            total_bytes += ops_per_rank[r][i].length
            total_ops += 1

    var tp = len(ops_per_rank)
    var load_pools = List[BurstPool[mask_size]](capacity=tp)
    for r in range(tp):
        var mask = topo.mask[mask_size](r)
        load_pools.append(BurstPool[mask_size](
            capacity=1, cpu_mask=mask, numa_node=topo.node(r)))
        if not load_pools[r]:
            print(t"load pool setup failed for rank {r}")
            return None

    var pools_span = Span[BurstPool[mask_size], MutAnyOrigin](
        ptr=load_pools.unsafe_ptr().as_unsafe_any_origin(), length=len(load_pools))
    run_reads_multi[io_depth, mask_size](pools_span, paths, ops_per_rank)

    return LoadResult(total_bytes, total_ops)
