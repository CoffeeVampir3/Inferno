@fieldwise_init
struct PagePoolSpec(Copyable, Movable, ImplicitlyCopyable):
    var num_pages: Int
    var fixed_pages_per_seq: Int
    var max_pages_per_seq: Int


@fieldwise_init
struct BatchGeometry(Copyable, Movable):
    var max_seqs: Int
    var max_slots: Int
    var max_step_tokens: Int
    var pools: List[PagePoolSpec]


struct KVPageAllocator(Copyable, Movable):
    var free_stack: List[Int32]
    var holds: List[Int32]

    def __init__(out self, num_pages: Int):
        self.free_stack = List[Int32](capacity=num_pages)
        for i in range(num_pages):
            self.free_stack.append(Int32(num_pages - 1 - i))
        self.holds = List[Int32](length=num_pages, fill=Int32(0))

    @always_inline
    def acquire(mut self) -> Int:
        if len(self.free_stack) == 0:
            return -1
        var page = Int(self.free_stack.pop())
        self.holds[page] = 1
        return page

    @always_inline
    def retain(mut self, page: Int):
        debug_assert(
            self.holds[page] > 0,
            "page allocator: retain of an unheld page",
        )
        self.holds[page] += 1

    @always_inline
    def release(mut self, page: Int):
        debug_assert(
            self.holds[page] > 0,
            "page allocator: release of an unheld page",
        )
        self.holds[page] -= 1
        if self.holds[page] == 0:
            self.free_stack.append(Int32(page))

    @always_inline
    def hold_count(self, page: Int) -> Int:
        return Int(self.holds[page])

    @always_inline
    def available(self) -> Int:
        return len(self.free_stack)


struct KVPageTable(Copyable, Movable):
    var entries: List[Int32]
    var max_seqs: Int
    var max_pages_per_seq: Int

    def __init__(out self, max_seqs: Int, max_pages_per_seq: Int):
        self.max_seqs = max_seqs
        self.max_pages_per_seq = max_pages_per_seq
        self.entries = List[Int32](
            length=max_seqs * max_pages_per_seq, fill=Int32(-1))

    @always_inline
    def entry_index(self, seq_id: Int, page_ordinal: Int) -> Int:
        debug_assert(
            page_ordinal < self.max_pages_per_seq,
            "page table: ordinal beyond per-sequence bound",
        )
        return seq_id * self.max_pages_per_seq + page_ordinal

    @always_inline
    def map_page(mut self, seq_id: Int, page_ordinal: Int, page_index: Int):
        self.entries[self.entry_index(seq_id, page_ordinal)] = Int32(page_index)

    @always_inline
    def page_index(self, seq_id: Int, page_ordinal: Int) -> Int:
        return Int(self.entries[self.entry_index(seq_id, page_ordinal)])


@fieldwise_init
struct PagePool(Copyable, Movable):
    var spec: PagePoolSpec
    var pages: KVPageAllocator
    var table: KVPageTable


struct KVPageAccountant(Movable):
    var max_seqs: Int
    var pools: List[PagePool]
    var active: List[Bool]

    def __init__(out self, read geometry: BatchGeometry):
        self.max_seqs = geometry.max_seqs
        self.pools = List[PagePool]()
        for i in range(len(geometry.pools)):
            var spec = geometry.pools[i]
            self.pools.append(PagePool(
                spec,
                KVPageAllocator(spec.num_pages),
                KVPageTable(geometry.max_seqs, spec.max_pages_per_seq)))
        self.active = List[Bool](length=geometry.max_seqs, fill=False)

    @always_inline
    def pool_count(self) -> Int:
        return len(self.pools)

    @always_inline
    def pool_available(self, pool_idx: Int) -> Int:
        return self.pools[pool_idx].pages.available()

    @always_inline
    def pool_fixed_pages(self, pool_idx: Int) -> Int:
        return self.pools[pool_idx].spec.fixed_pages_per_seq

    @always_inline
    def page_index(self, pool_idx: Int, seq_id: Int, ordinal: Int) -> Int:
        return self.pools[pool_idx].table.page_index(seq_id, ordinal)

    @always_inline
    def page_holds(self, pool_idx: Int, page: Int) -> Int:
        return self.pools[pool_idx].pages.hold_count(page)

    @always_inline
    def pool_pages(self, pool_idx: Int) -> Int:
        return self.pools[pool_idx].spec.num_pages

    @always_inline
    def pool_seq_page_limit(self, pool_idx: Int) -> Int:
        return self.pools[pool_idx].spec.max_pages_per_seq

    def freeable_pages(self, pool_idx: Int, seq_id: Int) -> Int:
        var count = 0
        for ordinal in range(self.pools[pool_idx].spec.max_pages_per_seq):
            var page = self.pools[pool_idx].table.page_index(seq_id, ordinal)
            if page >= 0 and self.pools[pool_idx].pages.hold_count(page) == 1:
                count += 1
        return count

    @always_inline
    def retain_page(mut self, pool_idx: Int, page: Int):
        self.pools[pool_idx].pages.retain(page)

    @always_inline
    def release_page(mut self, pool_idx: Int, page: Int):
        self.pools[pool_idx].pages.release(page)

    def admit(mut self) -> Int:
        for seq_id in range(self.max_seqs):
            if self.active[seq_id]:
                continue
            for p in range(len(self.pools)):
                if (self.pools[p].spec.fixed_pages_per_seq
                        > self.pools[p].pages.available()):
                    return -1
            for p in range(len(self.pools)):
                for ordinal in range(self.pools[p].spec.fixed_pages_per_seq):
                    var page = self.pools[p].pages.acquire()
                    self.pools[p].table.map_page(seq_id, ordinal, page)
            self.active[seq_id] = True
            return seq_id
        return -1

    def release(mut self, seq_id: Int):
        if not self.active[seq_id]:
            return
        for p in range(len(self.pools)):
            for ordinal in range(self.pools[p].spec.max_pages_per_seq):
                var page = self.pools[p].table.page_index(seq_id, ordinal)
                if page >= 0:
                    self.pools[p].pages.release(page)
                    self.pools[p].table.map_page(seq_id, ordinal, -1)
        self.active[seq_id] = False

    def share(mut self, pool_idx: Int, src_seq: Int, dst_seq: Int, ordinal: Int):
        var page = self.pools[pool_idx].table.page_index(src_seq, ordinal)
        debug_assert(page >= 0, "share: source ordinal is unmapped")
        debug_assert(
            self.pools[pool_idx].table.page_index(dst_seq, ordinal) < 0,
            "share: destination ordinal already mapped",
        )
        self.pools[pool_idx].pages.retain(page)
        self.pools[pool_idx].table.map_page(dst_seq, ordinal, page)

    def truncate(mut self, seq_id: Int, last_keep_ordinal: Int):
        for p in range(len(self.pools)):
            if self.pools[p].spec.fixed_pages_per_seq > 0:
                continue
            for ordinal in range(
                    last_keep_ordinal + 1,
                    self.pools[p].spec.max_pages_per_seq):
                var page = self.pools[p].table.page_index(seq_id, ordinal)
                if page >= 0:
                    self.pools[p].pages.release(page)
                    self.pools[p].table.map_page(seq_id, ordinal, -1)

    def replace_with_private(
        mut self, pool_idx: Int, seq_id: Int, ordinal: Int,
    ) -> Int:
        var fresh = self.pools[pool_idx].pages.acquire()
        if fresh < 0:
            return -1
        var old = self.pools[pool_idx].table.page_index(seq_id, ordinal)
        debug_assert(old >= 0, "replace_with_private: ordinal is unmapped")
        self.pools[pool_idx].pages.release(old)
        self.pools[pool_idx].table.map_page(seq_id, ordinal, fresh)
        return fresh

    def pages_needed(self, seq_id: Int, last_ordinal: Int, mut need: List[Int]):
        for p in range(len(self.pools)):
            if self.pools[p].spec.fixed_pages_per_seq > 0:
                continue
            for ordinal in range(last_ordinal + 1):
                if self.pools[p].table.page_index(seq_id, ordinal) < 0:
                    need[p] += 1

    def fits(self, read need: List[Int]) -> Bool:
        for p in range(len(self.pools)):
            if need[p] > self.pools[p].pages.available():
                return False
        return True

    def reserve(mut self, seq_id: Int, last_ordinal: Int) -> Bool:
        for p in range(len(self.pools)):
            if self.pools[p].spec.fixed_pages_per_seq > 0:
                continue
            for ordinal in range(last_ordinal + 1):
                if self.pools[p].table.page_index(seq_id, ordinal) >= 0:
                    continue
                var page = self.pools[p].pages.acquire()
                if page < 0:
                    return False
                self.pools[p].table.map_page(seq_id, ordinal, page)
        return True
