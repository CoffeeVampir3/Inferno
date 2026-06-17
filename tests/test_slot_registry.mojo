from std.memory import Span

from continuous_batching.slot_registry import SlotRegistry
from continuous_batching.prefix_hash import refresh_chain

comptime PPP = 2


def check(cond: Bool, msg: String) -> Int:
    if cond:
        print("  ok  -", msg)
        return 0
    print("  FAIL-", msg)
    return 1


def seq(read xs: List[Int32]) -> Span[Int32, origin_of(xs)]:
    return Span(xs)


def plen[ppp: Int](
    read reg: SlotRegistry[ppp], sid: Int, read q: List[Int32],
) -> Int:
    var chain = List[UInt64]()
    refresh_chain[ppp](chain, Span(q))
    return reg.prefix_len(sid, Span(q), chain)


def best_prefix[ppp: Int](
    read reg: SlotRegistry[ppp], read incoming: List[Int32], want_owned: Bool,
) -> Tuple[Int, Int]:
    var chain = List[UInt64]()
    refresh_chain[ppp](chain, Span(incoming))
    var best = -1
    var best_lcp = 0
    for sid in range(reg.max_seqs):
        if not reg.is_resident(sid):
            continue
        if (reg.owner_of(sid) >= 0) != want_owned:
            continue
        var lcp = reg.prefix_len(sid, Span(incoming), chain)
        if lcp > best_lcp:
            best = sid
            best_lcp = lcp
    return (best, best_lcp)


def test_match_on_empty() -> Int:
    print("match_on_empty")
    var f = 0
    var reg = SlotRegistry[PPP](4)
    var q: List[Int32] = [1, 2, 3]
    var warm = best_prefix(reg, q, False)
    var active = best_prefix(reg, q, True)
    f += check(warm[0] == -1, "no resident slots -> no warm match")
    f += check(active[0] == -1, "no resident slots -> no active match")
    f += check(reg.lru_victim() == -1, "no resident slots -> no victim")
    return f


def test_prefix_len() -> Int:
    print("prefix_len")
    var f = 0
    var reg = SlotRegistry[PPP](4)
    var prefix: List[Int32] = [1, 2, 3]
    reg.open(0, owner_id=10, now=UInt(100))
    reg.extend(0, seq(prefix), 0, 0, 3, UInt(100))
    reg.set_warm(0)

    var cont: List[Int32] = [1, 2, 3, 4, 5]
    f += check(plen(reg, 0, cont) == 3,
               "continuation matches whole history")
    f += check(reg.length(0) == 3, "history length tracks fed tokens")

    var other: List[Int32] = [9, 9, 9, 9]
    f += check(plen(reg, 0, other) == 0,
               "fully diverged request shares nothing")

    var edited: List[Int32] = [1, 2, 9, 9]
    f += check(plen(reg, 0, edited) == 2,
               "edited request matches up to the divergence")

    var exact: List[Int32] = [1, 2, 3]
    f += check(plen(reg, 0, exact) == 3,
               "exact-length request matches its full length")

    var shorter: List[Int32] = [1, 2]
    f += check(plen(reg, 0, shorter) == 2,
               "shorter request is capped at its own length")
    return f


def test_owner_partition() -> Int:
    print("owner_partition")
    var f = 0
    var reg = SlotRegistry[PPP](4)
    var prefix: List[Int32] = [7, 8]
    reg.open(2, owner_id=5, now=UInt(50))
    reg.extend(2, seq(prefix), 0, 0, 2, UInt(50))

    var cont: List[Int32] = [7, 8, 9]
    var warm = best_prefix(reg, cont, False)
    var active = best_prefix(reg, cont, True)
    f += check(warm[0] == -1, "a live slot never matches as warm")
    f += check(active[0] == 2 and active[1] == 2,
               "a live slot matches as active")
    f += check(reg.lru_victim() == -1, "a live slot is never an LRU victim")

    reg.set_warm(2)
    warm = best_prefix(reg, cont, False)
    active = best_prefix(reg, cont, True)
    f += check(warm[0] == 2, "set_warm moves the slot to the warm side")
    f += check(active[0] == -1, "set_warm removes it from the active side")
    f += check(reg.lru_victim() == 2, "set_warm makes it evictable")
    return f


def test_longest_prefix_wins() -> Int:
    print("longest_prefix_wins")
    var f = 0
    var reg = SlotRegistry[PPP](4)
    var short: List[Int32] = [1, 2]
    var long: List[Int32] = [1, 2, 3, 4]
    reg.open(0, owner_id=1, now=UInt(10))
    reg.extend(0, seq(short), 0, 0, 2, UInt(10))
    reg.set_warm(0)
    reg.open(1, owner_id=2, now=UInt(20))
    reg.extend(1, seq(long), 0, 0, 4, UInt(20))
    reg.set_warm(1)

    var cont: List[Int32] = [1, 2, 3, 4, 5, 6]
    var m = best_prefix(reg, cont, False)
    f += check(m[0] == 1 and m[1] == 4, "longest matching prefix wins")

    var diverging: List[Int32] = [1, 2, 9, 9]
    m = best_prefix(reg, diverging, False)
    f += check(m[1] == 2, "divergence caps the reported prefix")
    return f


def test_lru_victim() -> Int:
    print("lru_victim")
    var f = 0
    var reg = SlotRegistry[PPP](4)
    var a: List[Int32] = [1]
    var b: List[Int32] = [2]
    var c: List[Int32] = [3]
    reg.open(0, owner_id=1, now=UInt(300))
    reg.extend(0, seq(a), 0, 0, 1, UInt(300))
    reg.set_warm(0)
    reg.open(1, owner_id=2, now=UInt(100))
    reg.extend(1, seq(b), 0, 0, 1, UInt(100))
    reg.set_warm(1)
    reg.open(2, owner_id=3, now=UInt(200))
    reg.extend(2, seq(c), 0, 0, 1, UInt(200))
    reg.set_warm(2)

    f += check(reg.lru_victim() == 1, "oldest last_used is the victim")

    reg.close(1)
    f += check(reg.lru_victim() == 2, "after eviction next-oldest is the victim")
    f += check(reg.length(1) == 0, "close drops history")
    f += check(not reg.is_resident(1), "closed slot is no longer resident")
    var cont: List[Int32] = [2, 2]
    var m = best_prefix(reg, cont, False)
    f += check(m[0] == -1, "closed slot no longer matches")
    return f


def test_seed_and_truncate() -> Int:
    print("seed_and_truncate")
    var f = 0
    var reg = SlotRegistry[PPP](4)
    var donor_toks: List[Int32] = [5, 6, 7, 8]
    reg.open(0, owner_id=1, now=UInt(100))
    reg.seed(0, seq(donor_toks), 3, UInt(100))
    f += check(reg.length(0) == 3, "seed preloads the prefix length")

    var cont: List[Int32] = [5, 6, 7, 9, 9]
    reg.set_warm(0)
    var m = best_prefix(reg, cont, False)
    f += check(m[0] == 0 and m[1] == 3, "seeded tokens match as a prefix")

    reg.adopt(0, 7, UInt(200))
    reg.seed(0, seq(cont), 2, UInt(200))
    f += check(reg.length(0) == 2, "re-seed truncates the history")
    f += check(reg.owner_of(0) == 7, "adopt installs the new owner")
    return f


def test_divergent_reseed() -> Int:
    print("divergent_reseed")
    var f = 0
    var reg = SlotRegistry[PPP](4)
    var original: List[Int32] = [1, 2, 3, 4]
    reg.open(0, owner_id=1, now=UInt(10))
    reg.extend(0, seq(original), 0, 0, 4, UInt(10))
    reg.set_warm(0)
    f += check(plen(reg, 0, original) == 4, "original history fully matches")

    var divergent: List[Int32] = [1, 9, 3, 4]
    reg.adopt(0, 2, UInt(20))
    reg.seed(0, seq(divergent), 4, UInt(20))
    f += check(plen(reg, 0, divergent) == 4,
               "re-seeded tokens fully match the new history")
    f += check(plen(reg, 0, original) == 1,
               "stale chain entries do not survive a divergent re-seed")
    return f


def test_hashed_matches_exact() -> Int:
    print("hashed_matches_exact")
    var f = 0
    comptime WIDE = 16
    var reg = SlotRegistry[WIDE](2)
    var history = List[Int32]()
    for i in range(64):
        history.append(Int32(1000 + i * 7))
    reg.open(0, owner_id=1, now=UInt(10))
    reg.extend(0, seq(history), 0, 0, 64, UInt(10))
    reg.set_warm(0)

    var mismatched = 0
    for divergence in range(65):
        var query = List[Int32]()
        for i in range(64):
            query.append(history[i])
        if divergence < 64:
            query[divergence] = Int32(-1)
        var hashed = plen(reg, 0, query)
        var exact = reg.exact_prefix_len(0, Span(query))
        if hashed != exact or exact != min(divergence, 64):
            mismatched += 1
    f += check(mismatched == 0,
               "hashed prefix equals exact prefix at every divergence point")

    var longer = List[Int32]()
    for i in range(80):
        longer.append(Int32(1000 + i * 7))
    f += check(plen(reg, 0, longer) == 64,
               "longer continuation is capped at the history length")
    return f


def main():
    var failures = 0
    failures += test_match_on_empty()
    failures += test_prefix_len()
    failures += test_owner_partition()
    failures += test_longest_prefix_wins()
    failures += test_lru_victim()
    failures += test_seed_and_truncate()
    failures += test_divergent_reseed()
    failures += test_hashed_matches_exact()
    print()
    if failures == 0:
        print("all slot-registry checks passed")
    else:
        print(failures, "check(s) FAILED")
