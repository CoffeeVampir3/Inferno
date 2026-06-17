from std.memory import Span
from std.time import perf_counter_ns

from kernels.flash_sample import SamplingParams, SampleOutcome

from .schedule import (
    Schedule, BatchSlot, PageCopy, CancelToken, ScheduledModel,
    MAXIMUM_SAMPLING_LOGITS,
)
from .paging import KVPageAccountant, BatchGeometry
from .slot_registry import SlotRegistry
from .prefix_hash import refresh_chain, hashed_prefix_len


@fieldwise_init
struct PrefixReuse(Copyable, Movable, ImplicitlyCopyable):
    var seq_id: Int
    var adopted: Bool
    var trim_len: Int
    var copy_start: Int
    var copy_count: Int


@fieldwise_init
struct PagePin(Copyable, Movable, ImplicitlyCopyable):
    var pool: Int
    var page: Int


struct Request(Movable):
    var tokens: List[Int32]
    var chain: List[UInt64]
    var generated: List[Int32]
    var sampling: SamplingParams
    var max_new_tokens: Int
    var seq_id: Int
    var done: Bool
    var live: Bool
    var no_share: Bool
    var cancel: CancelToken

    def __init__(
        out self, var tokens: List[Int32], sampling: SamplingParams,
        max_new_tokens: Int, no_share: Bool = False,
    ):
        self.tokens = tokens^
        self.chain = List[UInt64]()
        self.generated = List[Int32]()
        self.sampling = sampling
        self.max_new_tokens = max_new_tokens
        self.seq_id = -1
        self.done = False
        self.live = True
        self.no_share = no_share
        self.cancel = CancelToken()

    def rehash[positions_per_page: Int](mut self):
        refresh_chain[positions_per_page](self.chain, Span(self.tokens))


struct ContinuousBatchScheduler[positions_per_page: Int](Movable):
    var pages: KVPageAccountant
    var registry: SlotRegistry[Self.positions_per_page]
    var requests: List[Request]
    var schedule: Schedule
    var need: List[Int]
    var max_slots: Int
    var step_budget: Int
    var decode_band: Int
    var stop_tokens: List[Int32]
    var pending: List[Int]
    var free_slots: List[Int]
    var ring_capacity: Int
    var seq_ceiling: Int
    var ring_floor: List[Int]
    var reuse: List[PrefixReuse]
    var reuse_copies: List[PageCopy]
    var pins: List[PagePin]
    var reuse_dirty: List[Bool]
    var unsharable: List[Bool]

    def __init__(
        out self, read geometry: BatchGeometry, step_budget: Int,
        var stop_tokens: List[Int32], decode_band: Int = 4,
    ):
        self.pages = KVPageAccountant(geometry)
        self.registry = SlotRegistry[Self.positions_per_page](
            geometry.max_seqs)
        self.requests = List[Request]()
        self.schedule = Schedule()
        self.need = List[Int](length=len(geometry.pools), fill=0)
        self.max_slots = geometry.max_slots
        self.step_budget = min(step_budget, geometry.max_step_tokens)
        self.decode_band = decode_band
        self.stop_tokens = stop_tokens^
        self.pending = List[Int]()
        self.free_slots = List[Int]()
        var ring_capacity = 1 << 62
        var seq_ceiling = 1 << 62
        for p in range(len(geometry.pools)):
            if geometry.pools[p].fixed_pages_per_seq > 0:
                var cap = (geometry.pools[p].fixed_pages_per_seq
                           * Self.positions_per_page)
                if cap < ring_capacity:
                    ring_capacity = cap
            else:
                var page_cap = min(
                    geometry.pools[p].max_pages_per_seq,
                    geometry.pools[p].num_pages)
                var cap = page_cap * Self.positions_per_page
                if cap < seq_ceiling:
                    seq_ceiling = cap
        self.ring_capacity = ring_capacity
        self.seq_ceiling = seq_ceiling
        self.ring_floor = List[Int](length=geometry.max_seqs, fill=0)
        self.reuse = List[PrefixReuse]()
        self.reuse_copies = List[PageCopy]()
        self.pins = List[PagePin]()
        self.reuse_dirty = List[Bool](length=geometry.max_seqs, fill=False)
        self.unsharable = List[Bool](length=geometry.max_seqs, fill=False)

    def __init__(
        out self, read geometry: BatchGeometry, step_budget: Int,
        eos_token: Int32, decode_band: Int = 4,
    ):
        var stops = List[Int32]()
        stops.append(eos_token)
        self = Self(geometry, step_budget, stops^, decode_band)

    @always_inline
    def is_stop_token(self, token: Int32) -> Bool:
        for k in range(len(self.stop_tokens)):
            if token == self.stop_tokens[k]:
                return True
        return False

    def submit(
        mut self, var tokens: List[Int32], sampling: SamplingParams,
        max_new_tokens: Int, no_share: Bool = False,
    ) -> Optional[Int]:
        if len(tokens) == 0 or len(tokens) > self.seq_ceiling:
            return None
        var request_id: Int
        if len(self.free_slots) > 0:
            request_id = self.free_slots.pop()
            self.requests[request_id] = Request(
                tokens^, sampling, max_new_tokens, no_share)
        else:
            request_id = len(self.requests)
            self.requests.append(
                Request(tokens^, sampling, max_new_tokens, no_share))
        self.requests[request_id].rehash[Self.positions_per_page]()
        self.pending.append(request_id)
        return request_id

    def cancel_token(self, request_id: Int) -> CancelToken:
        return self.requests[request_id].cancel

    def retire(mut self, request_id: Int) -> Bool:
        if not self.requests[request_id].live:
            return False
        if not self.requests[request_id].done:
            return False
        self.requests[request_id].live = False
        self.requests[request_id].tokens.clear()
        self.requests[request_id].chain.clear()
        self.requests[request_id].generated.clear()
        var kept = List[Int](capacity=len(self.pending))
        for k in range(len(self.pending)):
            if self.pending[k] != request_id:
                kept.append(self.pending[k])
        self.pending = kept^
        self.free_slots.append(request_id)
        return True

    def pending_work(self) -> Bool:
        for k in range(len(self.pending)):
            if not self.requests[self.pending[k]].done:
                return True
        return False

    def evict_warm(mut self) -> Bool:
        var victim = self.registry.lru_victim()
        if victim < 0:
            return False
        self.pages.release(victim)
        self.registry.close(victim)
        self.unsharable[victim] = False
        return True

    @always_inline
    def is_warm(self, seq_id: Int) -> Bool:
        return (self.registry.is_resident(seq_id)
                and self.registry.owner_of(seq_id) < 0)

    def evict_warm_for(mut self) -> Bool:
        var victim = -1
        var oldest = UInt(0)
        for sid in range(self.registry.max_seqs):
            if not self.is_warm(sid):
                continue
            var frees = False
            for p in range(self.pages.pool_count()):
                if self.need[p] <= self.pages.pool_available(p):
                    continue
                if self.pages.freeable_pages(p, sid) > 0:
                    frees = True
            if not frees:
                continue
            if victim < 0 or self.registry.last_used[sid] < oldest:
                victim = sid
                oldest = self.registry.last_used[sid]
        if victim < 0:
            victim = self.chained_warm_victim()
        if victim < 0:
            return False
        self.pages.release(victim)
        self.registry.close(victim)
        self.unsharable[victim] = False
        return True

    def chained_warm_victim(self) -> Int:
        var victim = -1
        var oldest = UInt(0)
        for p in range(self.pages.pool_count()):
            if self.need[p] <= self.pages.pool_available(p):
                continue
            var warm_holds = List[Int](
                length=self.pages.pool_pages(p), fill=0)
            for sid in range(self.registry.max_seqs):
                if not self.is_warm(sid):
                    continue
                for ordinal in range(self.pages.pool_seq_page_limit(p)):
                    var page = self.pages.page_index(p, sid, ordinal)
                    if page >= 0:
                        warm_holds[page] += 1
            for sid in range(self.registry.max_seqs):
                if not self.is_warm(sid):
                    continue
                var chained = False
                for ordinal in range(self.pages.pool_seq_page_limit(p)):
                    var page = self.pages.page_index(p, sid, ordinal)
                    if page < 0:
                        continue
                    if (self.pages.page_holds(p, page) > 1
                            and warm_holds[page]
                            == self.pages.page_holds(p, page)):
                        chained = True
                if not chained:
                    continue
                if victim < 0 or self.registry.last_used[sid] < oldest:
                    victim = sid
                    oldest = self.registry.last_used[sid]
        return victim

    @always_inline
    def window_intact(self, seq_id: Int, lcp: Int) -> Bool:
        var seq_len = self.registry.length(seq_id)
        if lcp >= seq_len:
            return True
        var live_start = lcp - Self.positions_per_page
        if live_start < 0:
            live_start = 0
        var floor = self.ring_floor[seq_id]
        if seq_len - self.ring_capacity > floor:
            floor = seq_len - self.ring_capacity
        return live_start >= floor

    def release_copy_pins(mut self):
        for k in range(len(self.pins)):
            self.pages.release_page(self.pins[k].pool, self.pins[k].page)
        self.pins.clear()

    def best_reusable(
        self, request_id: Int, want_owned: Bool,
    ) -> Tuple[Int, Int]:
        var best = -1
        var best_lcp = 0
        for sid in range(self.registry.max_seqs):
            if not self.registry.is_resident(sid):
                continue
            if self.unsharable[sid]:
                continue
            if self.reuse_dirty[sid]:
                continue
            if (self.registry.owner_of(sid) >= 0) != want_owned:
                continue
            var lcp = self.registry.prefix_len(
                sid, Span(self.requests[request_id].tokens),
                self.requests[request_id].chain)
            if lcp <= best_lcp:
                continue
            if not self.window_intact(sid, lcp):
                continue
            best = sid
            best_lcp = lcp
        if best >= 0:
            var exact = self.registry.exact_prefix_len(
                best, Span(self.requests[request_id].tokens))
            if exact < best_lcp:
                best_lcp = exact
                if best_lcp == 0 or not self.window_intact(best, best_lcp):
                    return (-1, 0)
        return (best, best_lcp)

    def dirty_donor_exists(self, request_id: Int) -> Bool:
        for sid in range(self.registry.max_seqs):
            if not self.reuse_dirty[sid]:
                continue
            if not self.registry.is_resident(sid):
                continue
            var lcp = self.registry.prefix_len(
                sid, Span(self.requests[request_id].tokens),
                self.requests[request_id].chain)
            if lcp < Self.positions_per_page:
                continue
            if self.window_intact(sid, lcp):
                return True
        return False

    def adopt_prefix(
        mut self, sid: Int, request_id: Int, lcp: Int, now: UInt,
    ) -> Bool:
        var req_len = len(self.requests[request_id].tokens)
        var seq_len = self.registry.length(sid)
        var take = lcp
        var resume = take - 1 if take == req_len else take
        var keep_last = (take - 1) // Self.positions_per_page
        var private_ordinal = resume // Self.positions_per_page
        var do_private = False
        if private_ordinal <= keep_last:
            for p in range(self.pages.pool_count()):
                if self.pages.pool_fixed_pages(p) > 0:
                    continue
                var page = self.pages.page_index(p, sid, private_ordinal)
                if self.pages.page_holds(p, page) > 1:
                    do_private = True
        if do_private:
            for p in range(len(self.need)):
                self.need[p] = 0
            for p in range(self.pages.pool_count()):
                if self.pages.pool_fixed_pages(p) > 0:
                    continue
                var page = self.pages.page_index(p, sid, private_ordinal)
                if self.pages.page_holds(p, page) > 1:
                    self.need[p] += 1
            while not self.pages.fits(self.need):
                if self.evict_warm_for():
                    continue
                take = private_ordinal * Self.positions_per_page
                do_private = False
                break
            if not do_private:
                if take < 1:
                    return False
                keep_last = (take - 1) // Self.positions_per_page
        self.registry.adopt(sid, request_id, now)
        var copy_start = len(self.reuse_copies)
        var trim = take
        if do_private:
            var valid_rows = take - private_ordinal * Self.positions_per_page
            for p in range(self.pages.pool_count()):
                if self.pages.pool_fixed_pages(p) > 0:
                    continue
                var old_page = self.pages.page_index(p, sid, private_ordinal)
                if self.pages.page_holds(p, old_page) <= 1:
                    continue
                var fresh = self.pages.replace_with_private(
                    p, sid, private_ordinal)
                debug_assert(
                    fresh >= 0, "adopt: privatization page must fit")
                self.pages.retain_page(p, old_page)
                self.pins.append(PagePin(p, old_page))
                self.reuse_copies.append(PageCopy(
                    p, old_page, fresh, 0, valid_rows))
            trim = private_ordinal * Self.positions_per_page
        if take < seq_len:
            var stale_floor = seq_len - self.ring_capacity
            if stale_floor > self.ring_floor[sid]:
                self.ring_floor[sid] = stale_floor
            self.pages.truncate(sid, keep_last)
            self.registry.seed(
                sid, Span(self.requests[request_id].tokens), take, now)
        self.reuse.append(PrefixReuse(
            sid, True, trim, copy_start,
            len(self.reuse_copies) - copy_start))
        if (len(self.reuse_copies) > copy_start
                or (take == req_len
                    and take % Self.positions_per_page == 0)):
            self.reuse_dirty[sid] = True
        self.unsharable[sid] = self.requests[request_id].no_share
        self.requests[request_id].seq_id = sid
        return True

    def fork_prefix(
        mut self, donor: Int, request_id: Int, lcp: Int, now: UInt,
    ) -> Bool:
        var req_len = len(self.requests[request_id].tokens)
        var resume = lcp - 1 if lcp == req_len else lcp
        var sid = self.pages.admit()
        if sid < 0:
            if not self.evict_warm():
                return False
            sid = self.pages.admit()
            if sid < 0:
                return False
        var sealed = resume // Self.positions_per_page
        var last_ordinal = (lcp - 1) // Self.positions_per_page
        for p in range(self.pages.pool_count()):
            if self.pages.pool_fixed_pages(p) > 0:
                continue
            for ordinal in range(sealed):
                self.pages.share(p, donor, sid, ordinal)
        while True:
            for p in range(len(self.need)):
                self.need[p] = 0
            self.pages.pages_needed(sid, last_ordinal, self.need)
            if self.pages.fits(self.need):
                break
            if not self.evict_warm_for():
                self.pages.release(sid)
                return False
        if not self.pages.reserve(sid, last_ordinal):
            self.pages.release(sid)
            return False
        var copy_start = len(self.reuse_copies)
        var partial_rows = lcp - sealed * Self.positions_per_page
        if partial_rows > 0:
            for p in range(self.pages.pool_count()):
                if self.pages.pool_fixed_pages(p) > 0:
                    continue
                self.reuse_copies.append(PageCopy(
                    p,
                    self.pages.page_index(p, donor, sealed),
                    self.pages.page_index(p, sid, sealed),
                    0, partial_rows))
        var live_start = lcp - Self.positions_per_page
        if live_start < 0:
            live_start = 0
        for p in range(self.pages.pool_count()):
            var ring_pages = self.pages.pool_fixed_pages(p)
            if ring_pages == 0:
                continue
            var a = live_start
            while a < lcp:
                var page_end = (((a // Self.positions_per_page) + 1)
                                * Self.positions_per_page)
                var seg_end = min(lcp, page_end)
                var ring_slot = (a // Self.positions_per_page) % ring_pages
                self.reuse_copies.append(PageCopy(
                    p,
                    self.pages.page_index(p, donor, ring_slot),
                    self.pages.page_index(p, sid, ring_slot),
                    a % Self.positions_per_page, seg_end - a))
                a = seg_end
        self.registry.open(sid, request_id, now)
        self.registry.seed(
            sid, Span(self.requests[request_id].tokens), lcp, now)
        self.ring_floor[sid] = live_start
        self.unsharable[sid] = self.requests[request_id].no_share
        self.requests[request_id].seq_id = sid
        self.reuse.append(PrefixReuse(
            sid, False, 0, copy_start,
            len(self.reuse_copies) - copy_start))
        self.reuse_dirty[sid] = True
        return True

    def request_common_prefix(self, a: Int, b: Int) -> Int:
        return hashed_prefix_len[Self.positions_per_page](
            Span(self.requests[a].tokens), self.requests[a].chain,
            Span(self.requests[b].tokens), self.requests[b].chain)

    @always_inline
    def slot_want(self, request_id: Int) -> Int:
        var sid = self.requests[request_id].seq_id
        var base_pos = self.registry.length(sid)
        if base_pos >= len(self.requests[request_id].tokens):
            base_pos = len(self.requests[request_id].tokens) - 1
        return len(self.requests[request_id].tokens) - base_pos

    def admit_slot(mut self, request_id: Int, mut budget: Int):
        var sid = self.requests[request_id].seq_id
        var base_pos = self.registry.length(sid)
        if base_pos >= len(self.requests[request_id].tokens):
            base_pos = len(self.requests[request_id].tokens) - 1
        var want = len(self.requests[request_id].tokens) - base_pos
        var feed = min(want, budget)
        for kj in range(len(self.pending)):
            var j = self.pending[kj]
            if j == request_id or self.requests[j].seq_id >= 0:
                continue
            var common = self.request_common_prefix(request_id, j)
            if common < Self.positions_per_page:
                continue
            var limit = common + Self.positions_per_page - base_pos
            if limit >= 1 and limit < feed:
                feed = limit
        var emit = (base_pos + feed == len(self.requests[request_id].tokens))
        for t in range(feed):
            self.schedule.tokens.append(
                self.requests[request_id].tokens[base_pos + t])
        var prompt_len = (len(self.requests[request_id].tokens)
                          - len(self.requests[request_id].generated))
        var slot = BatchSlot(
            sid, request_id, base_pos, feed, prompt_len, emit,
            self.requests[request_id].sampling,
            self.requests[request_id].cancel)
        self.schedule.prefill_tokens += slot.prefill_count()
        self.schedule.decode_tokens += slot.decode_count()
        self.schedule.slots.append(slot)
        budget -= feed

    def preempt_last_slot(mut self, now: UInt):
        var dropped = self.schedule.slots.pop()
        for _ in range(dropped.n_tokens):
            _ = self.schedule.tokens.pop()
        self.schedule.prefill_tokens -= dropped.prefill_count()
        self.schedule.decode_tokens -= dropped.decode_count()
        var sid = dropped.seq_id
        var owner = self.registry.owner_of(sid)
        var adopted_record = -1
        var fork_record = -1
        for k in range(len(self.reuse)):
            if self.reuse[k].seq_id != sid:
                continue
            if self.reuse[k].adopted:
                adopted_record = k
            else:
                fork_record = k
            self.reuse[k].seq_id = -1
        if adopted_record >= 0:
            self.registry.seed(
                sid, Span(self.requests[owner].tokens),
                self.reuse[adopted_record].trim_len, now)
            self.registry.set_warm(sid)
        elif fork_record >= 0:
            self.pages.release(sid)
            self.registry.close(sid)
            self.unsharable[sid] = False
        else:
            self.registry.set_warm(sid)
        self.requests[owner].seq_id = -1

    def unwind_prefix_reuse(mut self, now: UInt):
        for k in range(len(self.reuse)):
            var sid = self.reuse[k].seq_id
            if sid < 0:
                continue
            var owner = self.registry.owner_of(sid)
            if self.reuse[k].adopted:
                if owner >= 0:
                    self.requests[owner].seq_id = -1
                    self.registry.seed(
                        sid, Span(self.requests[owner].tokens),
                        self.reuse[k].trim_len, now)
                self.registry.set_warm(sid)
            else:
                if owner >= 0:
                    self.requests[owner].seq_id = -1
                self.pages.release(sid)
                self.registry.close(sid)
                self.unsharable[sid] = False
        self.reuse.clear()
        self.reuse_copies.clear()
        self.release_copy_pins()

    def build_schedule(mut self, now: UInt) -> Bool:
        var kept = List[Int](capacity=len(self.pending))
        for k in range(len(self.pending)):
            var i = self.pending[k]
            if self.requests[i].done:
                continue
            if self.requests[i].cancel.cancelled():
                self.requests[i].done = True
                var sid = self.requests[i].seq_id
                if sid >= 0:
                    self.registry.set_warm(sid)
                    self.requests[i].seq_id = -1
                continue
            kept.append(i)
        self.pending = kept^
        self.reuse.clear()
        self.reuse_copies.clear()
        for sid in range(len(self.reuse_dirty)):
            self.reuse_dirty[sid] = False

        for k in range(len(self.pending)):
            var i = self.pending[k]
            if self.requests[i].seq_id >= 0:
                continue
            if not self.requests[i].no_share:
                var warm_match = self.best_reusable(i, False)
                var active_match = self.best_reusable(i, True)
                var warm_sid = warm_match[0]
                var warm_lcp = warm_match[1]
                var active_sid = active_match[0]
                var active_lcp = active_match[1]
                var can_adopt = warm_sid >= 0 and warm_lcp >= 1
                var can_fork = (active_sid >= 0
                                and active_lcp >= Self.positions_per_page)
                var matched = False
                if can_adopt and (not can_fork
                                  or warm_lcp + Self.positions_per_page
                                  > active_lcp):
                    matched = self.adopt_prefix(warm_sid, i, warm_lcp, now)
                    if not matched and can_fork:
                        matched = self.fork_prefix(
                            active_sid, i, active_lcp, now)
                elif can_fork:
                    matched = self.fork_prefix(active_sid, i, active_lcp, now)
                    if not matched and can_adopt:
                        matched = self.adopt_prefix(warm_sid, i, warm_lcp, now)
                if matched:
                    continue
                if self.dirty_donor_exists(i):
                    continue
            var sid = self.pages.admit()
            if sid < 0:
                if not self.evict_warm():
                    continue
                sid = self.pages.admit()
                if sid < 0:
                    continue
            self.registry.open(sid, i, now)
            self.unsharable[sid] = self.requests[i].no_share
            self.ring_floor[sid] = 0
            self.requests[i].seq_id = sid

        self.schedule.clear()
        var budget = self.step_budget
        var prefill_waiting = False
        for k in range(len(self.pending)):
            var i = self.pending[k]
            if (self.requests[i].seq_id >= 0
                    and self.slot_want(i) > self.decode_band):
                prefill_waiting = True
        var decode_cap = self.max_slots - 1 if prefill_waiting else self.max_slots
        var slotted = List[Bool](length=len(self.pending), fill=False)
        for k in range(len(self.pending)):
            var i = self.pending[k]
            if self.requests[i].seq_id < 0:
                continue
            if len(self.schedule.slots) >= decode_cap or budget < 1:
                break
            if self.slot_want(i) > self.decode_band:
                continue
            self.admit_slot(i, budget)
            slotted[k] = True
        for k in range(len(self.pending)):
            var i = self.pending[k]
            if slotted[k] or self.requests[i].seq_id < 0:
                continue
            if len(self.schedule.slots) >= self.max_slots or budget < 1:
                break
            self.admit_slot(i, budget)

        while len(self.schedule.slots) > 0:
            for p in range(len(self.need)):
                self.need[p] = 0
            for s in range(len(self.schedule.slots)):
                var last_pos = (self.schedule.slots[s].base_pos
                                + self.schedule.slots[s].n_tokens - 1)
                self.pages.pages_needed(
                    self.schedule.slots[s].seq_id,
                    last_pos // Self.positions_per_page, self.need)
            if self.pages.fits(self.need):
                break
            if self.evict_warm_for():
                continue
            if len(self.schedule.slots) == 1:
                self.unwind_prefix_reuse(now)
                return False
            self.preempt_last_slot(now)

        if len(self.schedule.slots) == 0:
            self.unwind_prefix_reuse(now)
            return False

        for s in range(len(self.schedule.slots)):
            var last_pos = (self.schedule.slots[s].base_pos
                            + self.schedule.slots[s].n_tokens - 1)
            if not self.pages.reserve(
                    self.schedule.slots[s].seq_id,
                    last_pos // Self.positions_per_page):
                self.unwind_prefix_reuse(now)
                return False

        for s in range(len(self.schedule.slots)):
            var sid = self.schedule.slots[s].seq_id
            var first_ordinal = (self.schedule.slots[s].base_pos
                                 // Self.positions_per_page)
            var last_ordinal = ((self.schedule.slots[s].base_pos
                                 + self.schedule.slots[s].n_tokens - 1)
                                // Self.positions_per_page)
            for p in range(self.pages.pool_count()):
                if self.pages.pool_fixed_pages(p) > 0:
                    continue
                for ordinal in range(first_ordinal, last_ordinal + 1):
                    debug_assert(
                        self.pages.page_holds(
                            p, self.pages.page_index(p, sid, ordinal)) == 1,
                        "schedule: write range crosses a held page",
                    )

        for s in range(len(self.schedule.slots)):
            var sid = self.schedule.slots[s].seq_id
            if not self.unsharable[sid]:
                continue
            var last_ordinal = ((self.schedule.slots[s].base_pos
                                 + self.schedule.slots[s].n_tokens - 1)
                                // Self.positions_per_page)
            for p in range(self.pages.pool_count()):
                if self.pages.pool_fixed_pages(p) > 0:
                    continue
                for ordinal in range(last_ordinal + 1):
                    debug_assert(
                        self.pages.page_holds(
                            p, self.pages.page_index(p, sid, ordinal)) == 1,
                        "schedule: unsharable seq holds a shared page",
                    )

        for k in range(len(self.reuse)):
            if self.reuse[k].seq_id < 0:
                continue
            var copy_end = self.reuse[k].copy_start + self.reuse[k].copy_count
            for c in range(self.reuse[k].copy_start, copy_end):
                self.schedule.copies.append(self.reuse_copies[c])
        self.release_copy_pins()
        return True

    def absorb(
        mut self,
        read outs: List[SampleOutcome[MAXIMUM_SAMPLING_LOGITS]],
        now: UInt,
    ):
        var buf_start = 0
        var emit_idx = 0
        for s in range(len(self.schedule.slots)):
            var seq_id = self.schedule.slots[s].seq_id
            var base_pos = self.schedule.slots[s].base_pos
            var fed = self.schedule.slots[s].n_tokens
            var request_id = self.registry.owner_of(seq_id)
            self.registry.extend(
                seq_id, Span(self.schedule.tokens),
                buf_start, base_pos, fed, now)
            buf_start += fed
            if self.schedule.slots[s].emit:
                var token = outs[emit_idx].token_id
                emit_idx += 1
                self.requests[request_id].generated.append(token)
                self.requests[request_id].tokens.append(token)
                self.requests[request_id].rehash[Self.positions_per_page]()
                if (self.is_stop_token(token)
                    or len(self.requests[request_id].generated)
                        >= self.requests[request_id].max_new_tokens
                    or len(self.requests[request_id].tokens)
                        > self.seq_ceiling):
                    self.requests[request_id].done = True
                    self.registry.set_warm(seq_id)

    def discard_step(mut self):
        for s in range(len(self.schedule.slots)):
            var sid = self.schedule.slots[s].seq_id
            var written_end = (self.schedule.slots[s].base_pos
                               + self.schedule.slots[s].n_tokens)
            var floor = written_end - self.ring_capacity
            if floor > self.ring_floor[sid]:
                self.ring_floor[sid] = floor

    def step[M: ScheduledModel, //](mut self, mut model: M) -> Int:
        comptime assert M.POSITIONS_PER_PAGE == Self.positions_per_page, (
            "scheduler page granularity must match the model's")
        if not self.build_schedule(perf_counter_ns()):
            return 0
        var outs = model.execute(self.schedule, self.pages)
        if self.schedule.fully_cancelled():
            self.discard_step()
        else:
            self.absorb(outs, perf_counter_ns())
        return len(self.schedule.slots)
