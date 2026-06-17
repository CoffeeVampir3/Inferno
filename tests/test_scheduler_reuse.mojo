from kernels.flash_sample import SamplingParams, SampleOutcome
from continuous_batching.schedule import (
    Schedule, CancelToken, ScheduledModel, MAXIMUM_SAMPLING_LOGITS,
)
from continuous_batching.paging import (
    KVPageAccountant, BatchGeometry, PagePoolSpec,
)
from continuous_batching.scheduler import ContinuousBatchScheduler


comptime PPP = 4
comptime RING_POOL = 0
comptime GROW_POOL = 1
comptime RING_PAGES = 8
comptime GROW_PAGES = 16


struct StubModel(Movable, ScheduledModel):
    comptime POSITIONS_PER_PAGE = PPP
    var steps: Int
    var fed_tokens: Int
    var ring_copy_positions: Int
    var grow_copy_positions: Int
    var next_token: Int32
    var cancel_at_step: Int
    var victim: CancelToken
    var aborted_steps: Int

    def __init__(out self):
        self.steps = 0
        self.fed_tokens = 0
        self.ring_copy_positions = 0
        self.grow_copy_positions = 0
        self.next_token = Int32(9000)
        self.cancel_at_step = -1
        self.victim = CancelToken()
        self.aborted_steps = 0

    def batch_geometry(self) -> BatchGeometry:
        var pools = List[PagePoolSpec]()
        pools.append(PagePoolSpec(
            num_pages=RING_PAGES, fixed_pages_per_seq=2, max_pages_per_seq=2))
        pools.append(PagePoolSpec(
            num_pages=GROW_PAGES, fixed_pages_per_seq=0,
            max_pages_per_seq=GROW_PAGES))
        return BatchGeometry(
            max_seqs=4, max_slots=4, max_step_tokens=8, pools=pools^)

    def execute(
        mut self,
        read schedule: Schedule,
        read pages: KVPageAccountant,
    ) -> List[SampleOutcome[MAXIMUM_SAMPLING_LOGITS]]:
        self.steps += 1
        if self.steps == self.cancel_at_step:
            self.victim.cancel()
        if schedule.fully_cancelled():
            self.aborted_steps += 1
            return List[SampleOutcome[MAXIMUM_SAMPLING_LOGITS]]()
        self.fed_tokens += len(schedule.tokens)
        for c in range(len(schedule.copies)):
            if schedule.copies[c].pool == RING_POOL:
                self.ring_copy_positions += schedule.copies[c].pos_count
            else:
                self.grow_copy_positions += schedule.copies[c].pos_count
        var outs = List[SampleOutcome[MAXIMUM_SAMPLING_LOGITS]]()
        for s in range(len(schedule.slots)):
            if schedule.slots[s].emit:
                var out = SampleOutcome[MAXIMUM_SAMPLING_LOGITS]()
                out.token_id = self.next_token
                self.next_token += 1
                outs.append(out)
        return outs^


def check(cond: Bool, msg: String) -> Int:
    if cond:
        print("  ok  -", msg)
        return 0
    print("  FAIL-", msg)
    return 1


def run_until_idle(
    mut sched: ContinuousBatchScheduler[PPP], mut model: StubModel,
) -> Int:
    var steps = 0
    while sched.pending_work():
        if sched.step(model) == 0:
            break
        steps += 1
    return steps


def main():
    var failures = 0
    var model = StubModel()
    var greedy = SamplingParams(
        Float32(1.0), Float32(0.0), 0, 0, MAXIMUM_SAMPLING_LOGITS, True)
    var sched = ContinuousBatchScheduler[PPP](
        model.batch_geometry(), 8, Int32(-1))

    print("--- fresh prefill ---")
    var prompt_f = List[Int32]()
    for t in range(9):
        prompt_f.append(Int32(100 + t))
    var rf = sched.submit(prompt_f^, greedy, 1).value()
    var steps = run_until_idle(sched, model)
    failures += check(steps == 2, "9-token prompt prefills in two steps")
    failures += check(model.fed_tokens == 9, "prefill feeds exactly the prompt")
    failures += check(sched.requests[rf].done, "request completes")
    var sid_f = sched.requests[rf].seq_id

    print("--- adoption (take) and fork (point + copy) ---")
    var prompt_k = List[Int32]()
    for t in range(len(sched.requests[rf].tokens)):
        prompt_k.append(sched.requests[rf].tokens[t])
    prompt_k.append(Int32(111))
    var rk = sched.submit(prompt_k^, greedy, 1).value()
    var prompt_l = List[Int32]()
    for t in range(9):
        prompt_l.append(Int32(100 + t))
    prompt_l.append(Int32(222))
    var rl = sched.submit(prompt_l^, greedy, 1).value()

    var fed_before = model.fed_tokens
    steps = run_until_idle(sched, model)
    failures += check(steps == 1, "adoption and fork settle in one step")
    failures += check(
        model.fed_tokens - fed_before == 3,
        "21 requested tokens cost 3 fed tokens")
    failures += check(
        sched.requests[rk].seq_id == sid_f,
        "warm sequence is donated to the continuation")
    var sid_l = sched.requests[rl].seq_id
    failures += check(sid_l != sid_f, "fork gets its own sequence")
    failures += check(
        sched.pages.page_index(GROW_POOL, sid_f, 0)
        == sched.pages.page_index(GROW_POOL, sid_l, 0),
        "fork points at the donor's first sealed page")
    failures += check(
        sched.pages.page_index(GROW_POOL, sid_f, 1)
        == sched.pages.page_index(GROW_POOL, sid_l, 1),
        "fork points at the donor's second sealed page")
    failures += check(
        sched.pages.page_holds(
            GROW_POOL, sched.pages.page_index(GROW_POOL, sid_f, 0)) == 2,
        "shared sealed page carries two holds")
    failures += check(
        sched.pages.page_index(GROW_POOL, sid_f, 2)
        != sched.pages.page_index(GROW_POOL, sid_l, 2),
        "divergence pages stay private")
    failures += check(
        model.grow_copy_positions == 1,
        "divergence copy moves one position")
    failures += check(
        model.ring_copy_positions == 4,
        "ring copy moves exactly the live window")

    print("--- adoption with truncation privatizes a shared page ---")
    var prompt_m = List[Int32]()
    for t in range(7):
        prompt_m.append(Int32(100 + t))
    prompt_m.append(Int32(333))
    var rm = sched.submit(prompt_m^, greedy, 1).value()
    fed_before = model.fed_tokens
    var grow_before = model.grow_copy_positions
    var ring_before = model.ring_copy_positions
    steps = run_until_idle(sched, model)
    failures += check(steps == 1, "edited retry settles in one step")
    failures += check(
        sched.requests[rm].seq_id == sid_f,
        "intact warm sequence wins over the stale fork")
    failures += check(
        model.fed_tokens - fed_before == 1,
        "8 requested tokens cost 1 fed token")
    failures += check(
        model.grow_copy_positions - grow_before == 3,
        "privatization copies the kept rows of the shared page")
    failures += check(
        model.ring_copy_positions == ring_before,
        "adoption never copies the ring")
    failures += check(
        sched.pages.page_index(GROW_POOL, sid_f, 1)
        != sched.pages.page_index(GROW_POOL, sid_l, 1),
        "divergence page was replaced with a private copy")
    failures += check(
        sched.pages.page_holds(
            GROW_POOL, sched.pages.page_index(GROW_POOL, sid_f, 1)) == 1,
        "private replacement has a single hold")
    failures += check(
        sched.pages.page_holds(
            GROW_POOL, sched.pages.page_index(GROW_POOL, sid_l, 1)) == 1,
        "copy source returns to a single hold once unpinned")

    print("--- cancellation demotes the sequence to a warm prefix ---")
    var prompt_x = List[Int32]()
    for t in range(9):
        prompt_x.append(Int32(600 + t))
    var prompt_y = List[Int32]()
    for t in range(9):
        prompt_y.append(Int32(600 + t))
    prompt_y.append(Int32(777))
    var rx = sched.submit(prompt_x^, greedy, 1).value()
    _ = sched.step(model)
    failures += check(
        not sched.requests[rx].done, "long prompt is still mid-prefill")
    var sid_x = sched.requests[rx].seq_id
    var token = sched.cancel_token(rx)
    token.cancel()
    _ = run_until_idle(sched, model)
    failures += check(sched.requests[rx].done, "cancelled request resolves as done")
    failures += check(
        sched.registry.is_resident(sid_x)
        and sched.registry.owner_of(sid_x) == -1,
        "cancelled sequence is demoted to a warm prefix")
    failures += check(
        sched.registry.length(sid_x) == 8,
        "partial prefill survives cancellation")

    var fed_before_retry = model.fed_tokens
    var ry = sched.submit(prompt_y^, greedy, 1).value()
    _ = run_until_idle(sched, model)
    failures += check(
        sched.requests[ry].seq_id == sid_x,
        "retry re-adopts the cancelled prefix")
    failures += check(
        model.fed_tokens - fed_before_retry == 2,
        "retry pays only the unprefilled tail")

    print("--- retirement recycles request slots ---")
    var table_len = len(sched.requests)
    failures += check(sched.retire(rx), "done request retires")
    failures += check(sched.retire(ry), "second request retires")
    failures += check(not sched.retire(rx), "double retirement is refused")
    var prompt_z = List[Int32]()
    for t in range(4):
        prompt_z.append(Int32(800 + t))
    var rz = sched.submit(prompt_z^, greedy, 1).value()
    failures += check(rz < table_len, "new request reuses a retired slot")
    failures += check(
        len(sched.requests) == table_len, "request table stops growing")
    _ = run_until_idle(sched, model)
    failures += check(sched.requests[rz].done, "recycled slot completes")

    print("--- mid-step cancellation aborts and discards the step ---")
    var prompt_w = List[Int32]()
    for t in range(9):
        prompt_w.append(Int32(900 + t))
    var rw = sched.submit(prompt_w^, greedy, 1).value()
    _ = sched.step(model)
    var sid_w = sched.requests[rw].seq_id
    failures += check(
        sched.registry.length(sid_w) == 8, "first step prefills the budget")
    model.victim = sched.cancel_token(rw)
    model.cancel_at_step = model.steps + 1
    var fed_before_abort = model.fed_tokens
    var aborted_slots = sched.step(model)
    failures += check(
        aborted_slots == 1, "aborted step still reports its slot")
    failures += check(
        model.aborted_steps == 1, "model aborts the fully-cancelled step")
    failures += check(
        model.fed_tokens == fed_before_abort, "aborted step feeds nothing")
    failures += check(
        sched.registry.length(sid_w) == 8,
        "discarded step leaves the history unextended")
    failures += check(
        len(sched.requests[rw].generated) == 0,
        "no token is emitted from a discarded step")
    failures += check(
        sched.ring_floor[sid_w] == 1,
        "discard raises the ring floor over the partial writes")
    _ = run_until_idle(sched, model)
    failures += check(
        sched.requests[rw].done, "cancelled request resolves after discard")
    failures += check(
        sched.registry.is_resident(sid_w)
        and sched.registry.owner_of(sid_w) == -1,
        "discarded sequence still demotes to a warm prefix")

    print("--- hold balance after eviction ---")
    while sched.evict_warm():
        pass
    failures += check(
        sched.pages.pool_available(RING_POOL) == RING_PAGES,
        "ring pool drains to empty")
    failures += check(
        sched.pages.pool_available(GROW_POOL) == GROW_PAGES,
        "growing pool drains to empty")

    print("--- decode-first scheduling under a long prefill ---")
    var pools2 = List[PagePoolSpec]()
    pools2.append(PagePoolSpec(
        num_pages=8, fixed_pages_per_seq=2, max_pages_per_seq=2))
    pools2.append(PagePoolSpec(
        num_pages=6, fixed_pages_per_seq=0, max_pages_per_seq=8))
    var m2 = StubModel()
    var sched2 = ContinuousBatchScheduler[PPP](
        BatchGeometry(
            max_seqs=4, max_slots=2, max_step_tokens=8, pools=pools2^),
        8, Int32(-1))
    var prompt_pa = List[Int32]()
    for t in range(20):
        prompt_pa.append(Int32(100 + t))
    var rpa = sched2.submit(prompt_pa^, greedy, 1).value()
    _ = sched2.step(m2)
    failures += check(
        not sched2.requests[rpa].done, "long prefill is mid-flight")
    var prompt_pd = List[Int32]()
    prompt_pd.append(Int32(200))
    prompt_pd.append(Int32(201))
    var rpd = sched2.submit(prompt_pd^, greedy, 2).value()
    _ = sched2.step(m2)
    failures += check(
        len(sched2.requests[rpd].generated) >= 1,
        "late short request emits during the long prefill")
    failures += check(
        not sched2.requests[rpa].done,
        "prefill is still in flight while the decode emits")
    _ = run_until_idle(sched2, m2)
    failures += check(
        sched2.requests[rpa].done and sched2.requests[rpd].done,
        "both requests complete")
    var sid_pa = sched2.requests[rpa].seq_id
    var sid_pd = sched2.requests[rpd].seq_id

    print("--- productive eviction and demote-on-preempt ---")
    var prompt_ad = List[Int32]()
    for t in range(12):
        prompt_ad.append(Int32(300 + t))
    var rad = sched2.submit(prompt_ad^, greedy, 5).value()
    _ = sched2.step(m2)
    var sid_ad = sched2.requests[rad].seq_id
    failures += check(
        not sched2.registry.is_resident(sid_pa),
        "eviction reclaims the oldest cache that frees pages")
    failures += check(
        sched2.registry.is_resident(sid_pd),
        "eviction stops once the shortage is covered")
    _ = sched2.step(m2)
    var prompt_fb = List[Int32]()
    for t in range(8):
        prompt_fb.append(Int32(300 + t))
    prompt_fb.append(Int32(400))
    prompt_fb.append(Int32(401))
    var rfb = sched2.submit(prompt_fb^, greedy, 1).value()
    _ = sched2.step(m2)
    var sid_fb = sched2.requests[rfb].seq_id
    failures += check(
        sched2.requests[rfb].done and sid_fb != sid_ad,
        "fork against the active donor completes")

    var prompt_dd = List[Int32]()
    for t in range(16):
        prompt_dd.append(Int32(500 + t))
    var rdd = sched2.submit(prompt_dd^, greedy, 1).value()
    var prompt_cc = List[Int32]()
    for t in range(8):
        prompt_cc.append(Int32(300 + t))
    for t in range(6):
        prompt_cc.append(Int32(600 + t))
    var rcc = sched2.submit(prompt_cc^, greedy, 1).value()
    _ = sched2.step(m2)
    var sid_dd = sched2.requests[rdd].seq_id
    failures += check(
        sched2.requests[rcc].seq_id == sid_fb,
        "unslotted edit still adopts and truncates the warm fork")
    failures += check(
        sched2.pages.page_index(GROW_POOL, sid_fb, 2) < 0,
        "truncation released the fork's private tail page")
    var cc_token = sched2.cancel_token(rcc)
    cc_token.cancel()
    _ = sched2.step(m2)
    failures += check(
        sched2.registry.is_resident(sid_fb),
        "shared-only warm cache survives the page shortage")
    failures += check(
        sched2.requests[rdd].seq_id == -1
        and sched2.registry.is_resident(sid_dd)
        and sched2.registry.owner_of(sid_dd) == -1,
        "preempted prefill demotes to warm instead of releasing")
    failures += check(
        sched2.registry.length(sid_dd) == 7,
        "preempted prefill keeps its absorbed history")
    _ = run_until_idle(sched2, m2)
    failures += check(
        sched2.requests[rdd].done and sched2.requests[rad].done,
        "demoted prefill re-attaches and completes")
    failures += check(
        sched2.requests[rdd].seq_id == sid_dd,
        "re-attachment reuses the preserved sequence")
    while sched2.evict_warm():
        pass
    failures += check(
        sched2.pages.pool_available(RING_POOL) == 8
        and sched2.pages.pool_available(GROW_POOL) == 6,
        "pressure rig drains clean")

    print("--- submit validation and the sequence ceiling ---")
    var m3 = StubModel()
    var sched3 = ContinuousBatchScheduler[PPP](
        m3.batch_geometry(), 8, Int32(-1))
    var empty_prompt = List[Int32]()
    failures += check(
        not sched3.submit(empty_prompt^, greedy, 1),
        "empty prompt is rejected at submit")
    var oversize = List[Int32]()
    for t in range(65):
        oversize.append(Int32(t))
    failures += check(
        not sched3.submit(oversize^, greedy, 1),
        "prompt beyond the page ceiling is rejected at submit")
    var ceiling_prompt = List[Int32]()
    for t in range(64):
        ceiling_prompt.append(Int32(1000 + t))
    var rcl_opt = sched3.submit(ceiling_prompt^, greedy, 8)
    failures += check(
        rcl_opt is not None, "prompt at the ceiling is accepted")
    var rcl = rcl_opt.value()
    _ = run_until_idle(sched3, m3)
    failures += check(
        sched3.requests[rcl].done,
        "ceiling-length request completes instead of overflowing")
    failures += check(
        len(sched3.requests[rcl].generated) == 1,
        "generation stops at the sequence ceiling")

    print("--- stop-token list ---")
    var m4 = StubModel()
    var stops = List[Int32]()
    stops.append(Int32(42))
    stops.append(Int32(9000))
    var sched4 = ContinuousBatchScheduler[PPP](
        m4.batch_geometry(), 8, stops^)
    var prompt_s = List[Int32]()
    for t in range(4):
        prompt_s.append(Int32(50 + t))
    var rs = sched4.submit(prompt_s^, greedy, 5).value()
    _ = run_until_idle(sched4, m4)
    failures += check(
        sched4.requests[rs].done
        and len(sched4.requests[rs].generated) == 1,
        "stop-token list ends generation early")

    print("--- same-step reuse chaining defers one step ---")
    var m5 = StubModel()
    var sched5 = ContinuousBatchScheduler[PPP](
        m5.batch_geometry(), 8, Int32(-1))
    var prompt_w0 = List[Int32]()
    for t in range(8):
        prompt_w0.append(Int32(700 + t))
    var rw0 = sched5.submit(prompt_w0^, greedy, 1).value()
    _ = run_until_idle(sched5, m5)
    var sid_w0 = sched5.requests[rw0].seq_id
    var prompt_ca = List[Int32]()
    for t in range(8):
        prompt_ca.append(Int32(700 + t))
    var prompt_cb = List[Int32]()
    for t in range(8):
        prompt_cb.append(Int32(700 + t))
    prompt_cb.append(Int32(666))
    prompt_cb.append(Int32(667))
    var rca = sched5.submit(prompt_ca^, greedy, 3).value()
    var rcb = sched5.submit(prompt_cb^, greedy, 1).value()
    var fed_chain = m5.fed_tokens
    _ = sched5.step(m5)
    failures += check(
        sched5.requests[rca].seq_id == sid_w0,
        "page-aligned replay adopts the warm donor")
    failures += check(
        sched5.requests[rcb].seq_id == -1,
        "chained match defers while the donor write range is unsettled")
    _ = run_until_idle(sched5, m5)
    var sid_cb = sched5.requests[rcb].seq_id
    failures += check(
        sched5.requests[rcb].done and sid_cb != sid_w0,
        "deferred request forks on the next step")
    failures += check(
        sched5.pages.page_index(GROW_POOL, sid_w0, 0)
        == sched5.pages.page_index(GROW_POOL, sid_cb, 0),
        "deferred fork still shares the donor's sealed pages")
    failures += check(
        m5.fed_tokens - fed_chain == 5,
        "deferral pays only the divergent tails")

    print()
    if failures == 0:
        print("RESULT: PASS -- prefix holds, donation, and copies hold")
    else:
        print(t"RESULT: FAIL -- {failures} check(s)")
