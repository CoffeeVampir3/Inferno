from .ops import log_f32


comptime SM_A = UInt64(0x9E3779B97F4A7C15)
comptime SM_B = UInt64(0xBF58476D1CE4E5B9)
comptime SM_C = UInt64(0x94D049BB133111EB)
comptime RNG_TOP24 = Float32(16777216.0)


@always_inline
def splitmix64(x: UInt64) -> UInt64:
    """Stateless counter-based bijection (the SplitMix64 finalizer). Keying it
    on a logical position turns a random draw into a pure function of
    (seed, position), so the same value is produced whether a reduction runs
    over the whole vocabulary, in tiles, or sharded across ranks."""
    var z = x + SM_A
    z = (z ^ (z >> 30)) * SM_B
    z = (z ^ (z >> 27)) * SM_C
    return z ^ (z >> 31)


@always_inline
def rng_counter(row: Int, idx: Int) -> UInt64:
    """Pack a logical output position (row, vocabulary index) into one 64-bit
    counter. Partition-independent: the same (row, idx) yields the same counter
    on every rank."""
    return (UInt64(row) << 32) ^ UInt64(idx)


@always_inline
def rng_uniform(seed: UInt64, counter: UInt64) -> Float32:
    """Reproducible uniform in the open interval (0, 1). The open interval
    avoids the infinities that 0 or 1 would feed the Gumbel transform."""
    var h = splitmix64(seed ^ splitmix64(counter))
    return (Float32(Int(h >> 40)) + 1.0) / (RNG_TOP24 + 1.0)


@always_inline
def gumbel_noise(seed: UInt64, counter: UInt64) -> Float32:
    """One i.i.d. Gumbel(0, 1) draw at the logical position `counter`. Exact
    categorical sampling is `argmax_i(logit_i + gumbel_noise(seed, pos_i))`."""
    var u = rng_uniform(seed, counter)
    return -log_f32[1](-log_f32[1](u))
