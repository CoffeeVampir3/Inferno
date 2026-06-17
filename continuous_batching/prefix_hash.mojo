from std.memory import Span
from std.sys.info import simd_width_of

from simd_math import splitmix64

comptime TOKEN_LANES = simd_width_of[DType.int32]()
comptime BLOCK_C1 = UInt32(0xCC9E2D51)
comptime BLOCK_C2 = UInt32(0x1B873593)
comptime BLOCK_C3 = UInt32(0xE6546B64)
comptime CHAIN_SEED = UInt64(0x9AE16A3B2F90404F)


@always_inline
def rotl32[width: Int, bits: Int](
    x: SIMD[DType.uint32, width],
) -> SIMD[DType.uint32, width]:
    comptime left = SIMD[DType.uint32, width](bits)
    comptime right = SIMD[DType.uint32, width](32 - bits)
    return (x << left) | (x >> right)


def hash_token_block(tokens: Span[Int32, _], start: Int, count: Int) -> UInt64:
    var src = tokens.unsafe_ptr() + start
    var lanes = SIMD[DType.uint32, TOKEN_LANES](0)
    var i = 0
    while i + TOKEN_LANES <= count:
        var mixed = (src + i).load[width=TOKEN_LANES]().cast[
            DType.uint32]() * BLOCK_C1
        mixed = rotl32[TOKEN_LANES, 15](mixed) * BLOCK_C2
        lanes = lanes ^ mixed
        lanes = rotl32[TOKEN_LANES, 13](lanes) * 5 + BLOCK_C3
        i += TOKEN_LANES
    var acc = splitmix64(UInt64(count))
    for lane in range(TOKEN_LANES):
        acc = splitmix64(acc ^ ((UInt64(lane) << 32) | UInt64(lanes[lane])))
    while i < count:
        acc = splitmix64(acc ^ UInt64(tokens[start + i].cast[DType.uint32]()))
        i += 1
    return acc


@always_inline
def chain_link(prev: UInt64, block_hash: UInt64) -> UInt64:
    return splitmix64(prev ^ splitmix64(block_hash))


def refresh_chain[positions_per_page: Int](
    mut chain: List[UInt64], tokens: Span[Int32, _],
):
    var sealed = len(tokens) // positions_per_page
    while len(chain) > sealed:
        _ = chain.pop()
    while len(chain) < sealed:
        var ordinal = len(chain)
        var prev = chain[ordinal - 1] if ordinal > 0 else CHAIN_SEED
        chain.append(chain_link(prev, hash_token_block(
            tokens, ordinal * positions_per_page, positions_per_page)))


def first_mismatch(
    a: Span[Int32, _], b: Span[Int32, _], start: Int, limit: Int,
) -> Int:
    var pa = a.unsafe_ptr()
    var pb = b.unsafe_ptr()
    var i = start
    while i + TOKEN_LANES <= limit:
        var same = (pa + i).load[width=TOKEN_LANES]().eq(
            (pb + i).load[width=TOKEN_LANES]())
        if not same.reduce_and():
            for j in range(TOKEN_LANES):
                if not same[j]:
                    return i + j
        i += TOKEN_LANES
    while i < limit:
        if a[i] != b[i]:
            return i
        i += 1
    return limit


@always_inline
def token_prefix_len(a: Span[Int32, _], b: Span[Int32, _]) -> Int:
    return first_mismatch(a, b, 0, min(len(a), len(b)))


def hashed_prefix_len[positions_per_page: Int](
    a_tokens: Span[Int32, _], read a_chain: List[UInt64],
    b_tokens: Span[Int32, _], read b_chain: List[UInt64],
) -> Int:
    var pages = min(len(a_chain), len(b_chain))
    var matched = 0
    while matched < pages and a_chain[matched] == b_chain[matched]:
        matched += 1
    var n = min(len(a_tokens), len(b_tokens))
    var base = matched * positions_per_page
    return first_mismatch(
        a_tokens, b_tokens, base, min(n, base + positions_per_page))
