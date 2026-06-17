from std.reflection import reflect

from kernels.attention_ops import KVRunTable
from kernels.page_copy import CopyJob, dispatch_copy_jobs
from kernels.profiling import Profiler
from threading.threading_traits import BurstThreadPool
from continuous_batching.schedule import Schedule, PageCopy
from continuous_batching.paging import KVPageAccountant, PagePoolSpec
from modeling.slot import SlotLike


@fieldwise_init
struct KVComponent(Copyable, Movable, ImplicitlyCopyable):
    var offset: Int
    var row_bytes: Int


@fieldwise_init
struct KVPoolMirror(Copyable, Movable):
    """Runtime mirror of one KV page pool's physical layout. `pos_shard` is the
    number of token positions folded into one cache row per rank: 1 for
    position-replicated caches (every rank holds every position, feature
    dimension sharded), `degree` for position-sharded caches (positions
    round-robin across ranks)."""
    var page_len: Int
    var pos_shard: Int
    var region_off: Int
    var stride: Int
    var layers: Int
    var components: List[KVComponent]
    var spec: PagePoolSpec

    @always_inline
    def rows_per_page(self) -> Int:
        return self.page_len // self.pos_shard


def kv_components[T: AnyType](read group: T, degree: Int) -> List[KVComponent]:
    var comps = List[KVComponent]()
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            ref slot = reflect[T].field_ref[i](group)
            if FT.SHAPE.data_n(degree) > 0:
                comps.append(KVComponent(
                    offset=slot.get_offset(),
                    row_bytes=FT.SHAPE.data_m(degree) * FT.ENCODING.ELEMENT_BYTES,
                ))
    return comps^


def pool_specs(read mirrors: List[KVPoolMirror]) -> List[PagePoolSpec]:
    var specs = List[PagePoolSpec](capacity=len(mirrors))
    for p in range(len(mirrors)):
        specs.append(mirrors[p].spec)
    return specs^


def append_prefix_copy_jobs(
    read mirror: KVPoolMirror, read copy: PageCopy, mut jobs: List[CopyJob],
):
    var rows_per_page = mirror.rows_per_page()
    var row_start = copy.pos_start // mirror.pos_shard
    var row_end = (
        (copy.pos_start + copy.pos_count + mirror.pos_shard - 1)
        // mirror.pos_shard
    )
    var rows = row_end - row_start
    for l in range(mirror.layers):
        var layer_base = mirror.region_off + l * mirror.stride
        for c in range(len(mirror.components)):
            ref comp = mirror.components[c]
            var src = (copy.src_page * rows_per_page + row_start) * comp.row_bytes
            var dst = (copy.dst_page * rows_per_page + row_start) * comp.row_bytes
            jobs.append(CopyJob(
                layer_base + comp.offset + src,
                layer_base + comp.offset + dst,
                rows * comp.row_bytes))


def dispatch_prefix_copies[
    P: BurstThreadPool, Profile: Bool, N: Int, //,
](
    read mirrors: List[KVPoolMirror],
    read schedule: Schedule,
    read arena_bases: List[Int],
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if len(schedule.copies) == 0:
        return
    var jobs = List[CopyJob]()
    for c in range(len(schedule.copies)):
        append_prefix_copy_jobs(
            mirrors[schedule.copies[c].pool], schedule.copies[c], jobs)
    dispatch_copy_jobs(jobs, arena_bases, pools, prof)


def bind_pool_run_table(
    mut runs: KVRunTable,
    read schedule: Schedule,
    read pages: KVPageAccountant,
    pool_id: Int,
    read mirror: KVPoolMirror,
):
    var rows_per_page = mirror.rows_per_page()
    runs.clear()
    var buf_start = 0
    for i in range(len(schedule.slots)):
        var seq_id = schedule.slots[i].seq_id
        var base_pos = schedule.slots[i].base_pos
        runs.begin_run(buf_start, base_pos)
        var ordinals: Int
        if mirror.spec.fixed_pages_per_seq > 0:
            ordinals = mirror.spec.fixed_pages_per_seq
        else:
            var last_pos = base_pos + schedule.slots[i].n_tokens - 1
            ordinals = last_pos // mirror.page_len + 1
        for ordinal in range(ordinals):
            var page = pages.page_index(pool_id, seq_id, ordinal)
            debug_assert(page >= 0, "run references unmapped page")
            runs.add_base_row(Int32(page * rows_per_page))
        buf_start += schedule.slots[i].n_tokens
