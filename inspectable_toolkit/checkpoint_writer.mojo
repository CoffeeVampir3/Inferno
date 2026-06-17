from std.pathlib import Path
from std.memory import Span, UnsafePointer, memcpy
from std.os import makedirs
from std.os.path import isdir

from modeling.model_spec import WeightDesc
from modeling.slot import SlotLike, emit_quant_descs
from modeling.loader import find_tensor
from safetensors.parser import SafetensorsHeader


comptime COPY_CHUNK = 64 * 1024 * 1024


def copy_file(read src: Path, read dst: Path) -> Bool:
    try:
        with open(src, "r") as fin:
            with open(dst, "w") as fout:
                while True:
                    var part = fin.read_bytes(COPY_CHUNK)
                    if len(part) == 0:
                        break
                    fout.write_all(Span(part))
        return True
    except e:
        print(t"copy_file: {src} -> {dst} failed: {e}")
        return False


def copy_checkpoint(read source_dir: Path, read dest_dir: Path) -> Bool:
    """Byte-copy every top-level regular file from source_dir to dest_dir
    (skipping subdirectories), producing a standalone checkpoint whose
    weight bytes will then be patched in place."""
    try:
        makedirs(String(dest_dir), exist_ok=True)
    except e:
        print(t"copy_checkpoint: mkdir {dest_dir} failed: {e}")
        return False
    var names = List[String]()
    try:
        for entry in source_dir.listdir():
            names.append(String(entry))
    except e:
        print(t"copy_checkpoint: listdir {source_dir} failed: {e}")
        return False
    for i in range(len(names)):
        var src = source_dir / names[i]
        if isdir(src):
            continue
        if not copy_file(src, dest_dir / names[i]):
            return False
        print(t"  copied {names[i]}")
    return True


def gather_weight(
    read desc: WeightDesc, read arena_bases: List[Int],
) -> List[UInt8]:
    """Reconstruct the global row-major tensor for `desc` from the per-rank
    arenas, inverting `loader.emit_reads`. Returns exactly
    global_rows * global_cols * element_bytes bytes."""
    var eb = desc.element_bytes
    var total = desc.global_rows * desc.global_cols * eb
    var out = List[UInt8](length=total, fill=0)
    var op = out.unsafe_ptr()
    var degree = len(arena_bases)

    if desc.target_rank >= 0:
        var r = desc.target_rank % degree
        var base = arena_bases[r] + desc.arena_offset
        memcpy(
            dest=op,
            src=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=base),
            count=total)
        return out^

    if desc.data_rows == desc.global_rows and desc.data_cols == desc.global_cols:
        var base = arena_bases[0] + desc.arena_offset
        memcpy(
            dest=op,
            src=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=base),
            count=total)
        return out^

    if desc.data_rows != desc.global_rows:
        for r in range(degree):
            var row_start = r * desc.data_rows
            var real_rows = desc.global_rows - row_start
            if real_rows > desc.data_rows:
                real_rows = desc.data_rows
            if real_rows <= 0:
                continue
            var base = arena_bases[r] + desc.arena_offset
            memcpy(
                dest=op + row_start * desc.global_cols * eb,
                src=UnsafePointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=base),
                count=real_rows * desc.global_cols * eb)
        return out^

    var stride_bytes = desc.local_cols * eb
    for r in range(degree):
        var col_start = r * desc.data_cols
        var real_cols = desc.global_cols - col_start
        if real_cols > desc.data_cols:
            real_cols = desc.data_cols
        if real_cols <= 0:
            continue
        var row_bytes = real_cols * eb
        var base = arena_bases[r] + desc.arena_offset
        for row in range(desc.data_rows):
            memcpy(
                dest=op + (row * desc.global_cols + col_start) * eb,
                src=UnsafePointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=base + row * stride_bytes),
                count=row_bytes)
    return out^


def patch_tensor(
    read desc: WeightDesc,
    read arena_bases: List[Int],
    mut headers: List[SafetensorsHeader],
    read shard_paths: List[Path],
) -> Bool:
    """Locate `desc` in the copied shards by name and overwrite its data
    region in place with the bytes gathered from the arenas."""
    var found = find_tensor(desc.name, headers)
    if not found:
        print(t"checkpoint_writer: missing tensor {desc.name}")
        return False
    var loc = found.value().copy()
    var expect = desc.global_rows * desc.global_cols * desc.element_bytes
    if loc.meta.byte_size() != expect:
        var got = loc.meta.byte_size()
        print(t"checkpoint_writer: size mismatch {desc.name}: {got} != {expect}")
        return False
    var abs_off = headers[loc.shard].data_offset + loc.meta.start
    var buf = gather_weight(desc, arena_bases)
    try:
        with open(shard_paths[loc.shard], "rw") as f:
            _ = f.seek(UInt64(abs_off))
            f.write_all(Span(buf))
    except e:
        print(t"checkpoint_writer: write {desc.name} failed: {e}")
        return False
    return True


def patch_slot[
    S: SlotLike, //,
](
    read slot: S,
    layer_base: Int,
    read prefix: String,
    degree: Int,
    read arena_bases: List[Int],
    mut headers: List[SafetensorsHeader],
    read shard_paths: List[Path],
) -> Bool:
    """Emit the canonical WeightDesc(s) for a single live slot at its arena
    offset, then patch each into the copied checkpoint."""
    var descs = List[WeightDesc]()
    emit_quant_descs[
        S.ENCODING, S.SHAPE, S.QUANT, S.NAME.value(), S.TARGET_RANK,
    ](prefix, layer_base + slot.get_offset(), degree, descs)
    for i in range(len(descs)):
        if not patch_tensor(descs[i], arena_bases, headers, shard_paths):
            return False
    return True
