from std.os import getenv
from std.time import perf_counter_ns
from threading.threading_traits import BurstThreadPool


comptime PROFILE_RESERVOIR = 1024  # fixed samples kept per metric for p50/p99


def two_dp(ns: Int, unit: Int) -> String:
    var scaled = ns * 100 // unit
    var frac = scaled % 100
    var fs = String(frac)
    if frac < 10:
        fs = String("0") + fs
    return String(scaled // 100) + "." + fs


def human_ns(ns: Int) -> String:
    """Render a nanosecond duration with an adaptive unit."""
    if ns < 1000:
        return String(ns) + "ns"
    if ns < 1_000_000:
        return two_dp(ns, 1000) + "us"
    if ns < 1_000_000_000:
        return two_dp(ns, 1_000_000) + "ms"
    return two_dp(ns, 1_000_000_000) + "s"


def pct_str(part: Int, whole: Int) -> String:
    if whole <= 0:
        return "0.0%"
    var tenths = part * 1000 // whole
    return String(tenths // 10) + "." + String(tenths % 10) + "%"


def pad_left(s: String, w: Int) -> String:
    var pad = String("")
    for _ in range(w - s.byte_length()):
        pad += " "
    return pad + s


def pad_right(s: String, w: Int) -> String:
    var pad = String("")
    for _ in range(w - s.byte_length()):
        pad += " "
    return s + pad


def rule(n: Int) -> String:
    var s = String("")
    for _ in range(n):
        s += "-"
    return s


def pad_center(s: String, w: Int) -> String:
    var n = s.byte_length()
    if n >= w:
        return s
    var left = (w - n) // 2
    var right = w - n - left
    var out = String("")
    for _ in range(left):
        out += " "
    out += s
    for _ in range(right):
        out += " "
    return out


def color_enabled() -> Bool:
    return getenv("NO_COLOR", default="") == ""


def colorize(s: String, code: String, enabled: Bool) -> String:
    if not enabled:
        return s
    return "\U0000001B[" + code + "m" + s + "\U0000001B[0m"


def pct_tenths(part: Int, whole: Int) -> Int:
    if whole <= 0:
        return 0
    return part * 1000 // whole


def heat_pct(part: Int, whole: Int, s: String, enabled: Bool) -> String:
    var tenths = pct_tenths(part, whole)
    if tenths >= 100:
        return colorize(s, "1;31", enabled)
    if tenths >= 30:
        return colorize(s, "33", enabled)
    if tenths >= 10:
        return colorize(s, "36", enabled)
    if part == 0:
        return colorize(s, "2", enabled)
    return s


def metric_header(cw: Int, pct_w: Int) -> String:
    return (pad_left("mean", cw)
        + " "
        + pad_left("p99", cw)
        + " "
        + pad_left("%wall", pct_w))


struct ReservoirMetric[N: Int = PROFILE_RESERVOIR](Copyable, Movable):
    """Bounded per-metric stats. count/sum/min/max are exact running
    values; a fixed-N uniform reservoir (Vitter algorithm R) on the heap
    backs the quantiles, so memory is capped at N regardless of sample
    count and the enclosing value type stays small."""
    var samples: List[Int]
    var count: Int
    var total: Int
    var minv: Int
    var maxv: Int
    var rng: UInt64

    @always_inline
    def __init__(out self):
        self.samples = List[Int]()
        self.count = 0
        self.total = 0
        self.minv = 0
        self.maxv = 0
        self.rng = 0x9E3779B97F4A7C15

    @always_inline
    def next_rand(mut self) -> UInt64:
        var x = self.rng
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        self.rng = x
        return x

    @always_inline
    def add(mut self, raw: Int):
        var v = raw if raw > 0 else 0
        var seen = self.count
        if seen == 0 or v < self.minv:
            self.minv = v
        if seen == 0 or v > self.maxv:
            self.maxv = v
        self.total += v
        if seen < Self.N:
            self.samples.append(v)
        else:
            var j = Int(self.next_rand() % UInt64(seen + 1))
            if j < Self.N:
                self.samples[j] = v
        self.count = seen + 1

    @always_inline
    def mean(self) -> Int:
        if self.count == 0:
            return 0
        return self.total // self.count

    def quantiles(self, q_lo: Float64, q_hi: Float64) -> Tuple[Int, Int]:
        """(p_lo, p_hi) from the reservoir. Returns exact min/max bounds
        when the reservoir is empty/degenerate."""
        var filled = len(self.samples)
        if filled == 0:
            return (0, 0)
        var ordered = self.samples.copy()
        sort(ordered)
        var lo_idx = Int(q_lo * Float64(filled))
        var hi_idx = Int(q_hi * Float64(filled))
        if lo_idx >= filled:
            lo_idx = filled - 1
        if hi_idx >= filled:
            hi_idx = filled - 1
        return (ordered[lo_idx], ordered[hi_idx])

    def fmt_cells(self, cw: Int, pct_w: Int, grand: Int, colors: Bool) -> String:
        var q = self.quantiles(0.5, 0.99)
        var p50 = q[0]
        var p99 = q[1]
        var mean_s = pad_left(human_ns(self.mean()), cw)
        var p99_s = pad_left(human_ns(p99), cw)
        if (p50 > 0 and p99 > p50 * 4 and p99 > 10_000) or (
            p50 == 0 and p99 > 10_000
        ):
            p99_s = colorize(p99_s, "35", colors)
        var wall_s = heat_pct(
            self.total, grand, pad_left(pct_str(self.total, grand), pct_w),
            colors,
        )
        return mean_s + " " + p99_s + " " + wall_s


struct ProfileRecord(Copyable, Movable):
    var label: StaticString
    var dispatch: ReservoirMetric[]
    var compute: ReservoirMetric[]
    var join: ReservoirMetric[]

    @always_inline
    def __init__(out self, label: StaticString = ""):
        self.label = label
        self.dispatch = ReservoirMetric[]()
        self.compute = ReservoirMetric[]()
        self.join = ReservoirMetric[]()

    @always_inline
    def add(mut self, dispatch_ns: Int, compute_ns: Int, join_ns: Int):
        self.dispatch.add(dispatch_ns)
        self.compute.add(compute_ns)
        self.join.add(join_ns)

struct Profiler[Profile: Bool, N: Int = 64](Copyable, Movable):
    """Per-dispatch timing sink with bounded per-label statistics.

    `Profile=False` => zero-sized, every method a no-op. When on, holds N
    label slots; each tracks exact count/sum/min/max and a fixed-size
    reservoir for p50/p99 on dispatch / compute / join. Touched only on
    the dispatch thread, so no atomics and no cross-NUMA traffic."""
    comptime CAP = Self.N if Self.Profile else 0
    var records: InlineArray[ProfileRecord, Self.CAP]
    var count: Int
    var wall_ns: Int  # true elapsed forward time over the current window

    @always_inline
    def __init__(out self):
        self.records = InlineArray[ProfileRecord, Self.CAP](fill=ProfileRecord())
        self.count = 0
        self.wall_ns = 0

    @always_inline
    def add_wall(mut self, ns: Int):
        comptime if Self.Profile:
            self.wall_ns += ns

    @always_inline
    def record(mut self, label: StaticString,
               dispatch_ns: Int, compute_ns: Int, join_ns: Int):
        comptime if Self.Profile:
            for i in range(self.count):
                if self.records[i].label == label:
                    self.records[i].add(dispatch_ns, compute_ns, join_ns)
                    return
            if self.count < Self.N:
                self.records[self.count] = ProfileRecord(label)
                self.records[self.count].add(dispatch_ns, compute_ns, join_ns)
                self.count += 1

    @always_inline
    def reset(mut self):
        comptime if Self.Profile:
            self.count = 0
            self.wall_ns = 0

    def report(self, title: StaticString = "dispatch profile"):
        comptime if Self.Profile:
            if self.count == 0:
                return
            var name_w = 7  # byte_length("section")
            for i in range(self.count):
                var w = self.records[i].label.byte_length()
                if w > name_w:
                    name_w = w
            var calls_w = 6
            var cw = 8
            var pct_w = 6
            var group_w = cw * 2 + pct_w + 2
            var colors = color_enabled()

            # per-label totals, then order labels by cost desc
            var totals = List[Int](capacity=self.count)
            var order = List[Int](capacity=self.count)
            for i in range(self.count):
                ref r = self.records[i]
                var lt = r.dispatch.total + r.compute.total + r.join.total
                totals.append(lt)
                order.append(i)
            for a in range(1, self.count):
                var key = order[a]
                var kt = totals[key]
                var b = a - 1
                while b >= 0 and totals[order[b]] < kt:
                    order[b + 1] = order[b]
                    b -= 1
                order[b + 1] = key

            var section_h = pad_right("section", name_w)
            var calls_h = pad_left("calls", calls_w)
            var header_top = (
                section_h + calls_h
                + " | " + pad_center("dispatch", group_w)
                + " | " + pad_center("compute", group_w)
                + " | " + pad_center("join", group_w)
            )
            var blanks = pad_right("", name_w + calls_w)
            var metrics = metric_header(cw, pct_w)
            var header_bottom = (
                blanks
                + " | " + metrics
                + " | " + metrics
                + " | " + metrics
            )
            print()
            print(t"=== {title} : per-dispatch latency ===")
            print(header_top)
            print(header_bottom)
            print(rule(header_top.byte_length()))
            var td = 0
            var tc = 0
            var tj = 0
            for oi in range(self.count):
                ref r = self.records[order[oi]]
                var label = pad_right(String(r.label), name_w)
                var calls = pad_left(String(r.dispatch.count), calls_w)
                var dispatch = r.dispatch.fmt_cells(
                    cw, pct_w, self.wall_ns, colors)
                var compute = r.compute.fmt_cells(
                    cw, pct_w, self.wall_ns, colors)
                var join = r.join.fmt_cells(
                    cw, pct_w, self.wall_ns, colors)
                print(t"{label}{calls} | {dispatch} | {compute} | {join}")
                td += r.dispatch.total
                tc += r.compute.total
                tj += r.join.total
            print(rule(header_top.byte_length()))
            var accounted = td + tc + tj
            var wall = self.wall_ns
            var dark = wall - accounted
            if dark < 0:
                dark = 0
            var td_h = human_ns(td)
            var td_pct = pct_str(td, wall)
            var tc_h = human_ns(tc)
            var tc_pct = pct_str(tc, wall)
            var tj_h = human_ns(tj)
            var tj_pct = pct_str(tj, wall)
            var td_pct_hot = heat_pct(td, wall, td_pct, colors)
            var tc_pct_hot = heat_pct(tc, wall, tc_pct, colors)
            var tj_pct_hot = heat_pct(tj, wall, tj_pct, colors)
            print(t"by phase   dispatch {td_h} ({td_pct_hot})   compute {tc_h} ({tc_pct_hot})   join {tj_h} ({tj_pct_hot})")
            var wall_h = human_ns(wall)
            var accounted_h = human_ns(accounted)
            var accounted_pct = pct_str(accounted, wall)
            var dark_h = human_ns(dark)
            var dark_pct = pct_str(dark, wall)
            var dark_pct_hot = heat_pct(dark, wall, dark_pct, colors)
            print(t"wall {wall_h}   accounted {accounted_h} ({accounted_pct})   dark {dark_h} ({dark_pct_hot})")


@always_inline
def compute_end_across[
    P: BurstThreadPool, //,
](mut pools: List[P]) -> Int:
    """Max worker completion timestamp across all ranks from the last
    dispatch. Call after join. perf_counter_ns is monotonic and shared
    across cores, so this is comparable to the dispatcher's own clock."""
    var m = 0
    for r in range(len(pools)):
        var ts = pools[r].last_worker_timestamp()
        if ts > m:
            m = ts
    return m


struct DispatchSpan[Profile: Bool](Copyable, ImplicitlyCopyable):
    """Comptime-gated timing scope for one dispatch/compute/join cycle.

    Construct at issue start (captures t0), call `issued()` after the
    dispatch loop (t1), then `finish(...)` after join (t2 + cross-pool
    compute end, clamped to the measured span). `Profile=False` =>
    zero-sized, every method no-op."""
    comptime M = 2 if Self.Profile else 0
    var ts: InlineArray[Int, Self.M]  # [0]=t0 (issue start), [1]=t1 (issue end)

    @always_inline
    def __init__(out self):
        comptime if Self.Profile:
            var now = Int(perf_counter_ns())
            self.ts = InlineArray[Int, Self.M](fill=now)
        else:
            self.ts = InlineArray[Int, Self.M](uninitialized=True)

    @always_inline
    def issued(mut self):
        comptime if Self.Profile:
            self.ts[1] = Int(perf_counter_ns())

    @always_inline
    def finish[
        P: BurstThreadPool, N: Int, //,
    ](self, mut prof: Profiler[Self.Profile, N], mut pools: List[P],
      label: StaticString):
        comptime if Self.Profile:
            var t2 = Int(perf_counter_ns())
            var compute_end = compute_end_across(pools)
            if compute_end < self.ts[1]:
                compute_end = self.ts[1]
            if compute_end > t2:
                compute_end = t2
            var dispatch_ns = self.ts[1] - self.ts[0]
            var compute_ns = compute_end - self.ts[1]
            var join_ns = t2 - compute_end
            prof.record(label, dispatch_ns, compute_ns, join_ns)

    @always_inline
    def finish_inline[
        N: Int, //,
    ](self, mut prof: Profiler[Self.Profile, N], label: StaticString):
        """Record a main-thread inline execution (no dispatch/join)."""
        comptime if Self.Profile:
            var t_end = Int(perf_counter_ns())
            prof.record(label, 0, t_end - self.ts[0], 0)
