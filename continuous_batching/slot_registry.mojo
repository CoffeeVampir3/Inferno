from std.memory import Span

from .prefix_hash import refresh_chain, token_prefix_len, hashed_prefix_len


struct SlotRegistry[positions_per_page: Int](Movable):
    var max_seqs: Int
    var resident: List[Bool]
    var owner: List[Int]
    var last_used: List[UInt]
    var tokens: List[List[Int32]]
    var chains: List[List[UInt64]]

    def __init__(out self, max_seqs: Int):
        self.max_seqs = max_seqs
        self.resident = List[Bool](length=max_seqs, fill=False)
        self.owner = List[Int](length=max_seqs, fill=-1)
        self.last_used = List[UInt](length=max_seqs, fill=UInt(0))
        self.tokens = List[List[Int32]]()
        self.chains = List[List[UInt64]]()
        for _ in range(max_seqs):
            self.tokens.append(List[Int32]())
            self.chains.append(List[UInt64]())

    @always_inline
    def length(self, seq_id: Int) -> Int:
        return len(self.tokens[seq_id])

    @always_inline
    def owner_of(self, seq_id: Int) -> Int:
        return self.owner[seq_id]

    def open(mut self, seq_id: Int, owner_id: Int, now: UInt):
        self.resident[seq_id] = True
        self.owner[seq_id] = owner_id
        self.last_used[seq_id] = now
        self.tokens[seq_id].clear()
        self.chains[seq_id].clear()

    def adopt(mut self, seq_id: Int, owner_id: Int, now: UInt):
        self.owner[seq_id] = owner_id
        self.last_used[seq_id] = now

    def extend(
        mut self, seq_id: Int, source: Span[Int32, _],
        start: Int, base_pos: Int, count: Int, now: UInt,
    ):
        debug_assert(
            base_pos <= len(self.tokens[seq_id]),
            "slot registry: extend would leave a position gap",
        )
        for t in range(count):
            if base_pos + t < len(self.tokens[seq_id]):
                continue
            self.tokens[seq_id].append(source[start + t])
        refresh_chain[Self.positions_per_page](
            self.chains[seq_id], Span(self.tokens[seq_id]))
        self.last_used[seq_id] = now

    def seed(
        mut self, seq_id: Int, source: Span[Int32, _], count: Int, now: UInt,
    ):
        var same = token_prefix_len(Span(self.tokens[seq_id]), source)
        var keep = min(same, count) // Self.positions_per_page
        while len(self.chains[seq_id]) > keep:
            _ = self.chains[seq_id].pop()
        self.tokens[seq_id].clear()
        for i in range(count):
            self.tokens[seq_id].append(source[i])
        refresh_chain[Self.positions_per_page](
            self.chains[seq_id], Span(self.tokens[seq_id]))
        self.last_used[seq_id] = now

    def set_warm(mut self, seq_id: Int):
        self.owner[seq_id] = -1

    def close(mut self, seq_id: Int):
        self.resident[seq_id] = False
        self.owner[seq_id] = -1
        self.tokens[seq_id].clear()
        self.chains[seq_id].clear()

    @always_inline
    def is_resident(self, seq_id: Int) -> Bool:
        return self.resident[seq_id]

    def prefix_len(
        self, seq_id: Int, incoming: Span[Int32, _],
        read incoming_chain: List[UInt64],
    ) -> Int:
        return hashed_prefix_len[Self.positions_per_page](
            Span(self.tokens[seq_id]), self.chains[seq_id],
            incoming, incoming_chain)

    def exact_prefix_len(self, seq_id: Int, incoming: Span[Int32, _]) -> Int:
        return token_prefix_len(Span(self.tokens[seq_id]), incoming)

    def lru_victim(self) -> Int:
        var victim = -1
        var oldest = UInt(0)
        for sid in range(self.max_seqs):
            if not self.resident[sid] or self.owner[sid] >= 0:
                continue
            if victim < 0 or self.last_used[sid] < oldest:
                victim = sid
                oldest = self.last_used[sid]
        return victim
