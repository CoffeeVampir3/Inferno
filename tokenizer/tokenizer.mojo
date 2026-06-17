from std.collections import Dict, Set
from std.memory import Span
from std.math import iota

from .unicode_props import (
    LETTER_RANGES,
    LETTER_PAIR_COUNT,
    LETTER_MIN,
    LETTER_MAX,
    NUMBER_RANGES,
    NUMBER_PAIR_COUNT,
    NUMBER_MIN,
    NUMBER_MAX,
    WHITESPACE_RANGES,
    WHITESPACE_PAIR_COUNT,
    WHITESPACE_MIN,
    WHITESPACE_MAX,
)
from .unicode_psm_props import (
    MARK_RANGES,
    MARK_PAIR_COUNT,
    MARK_MIN,
    MARK_MAX,
    PUNCT_SYMBOL_RANGES,
    PUNCT_SYMBOL_PAIR_COUNT,
    PUNCT_SYMBOL_MIN,
    PUNCT_SYMBOL_MAX,
)
from .bpe import (
    bpe_merge_ids,
    PieceCache,
)


def bytes_to_gpt2(data: Span[Byte, _]) -> String:
    var cp_table = materialize[BYTE_TO_CODEPOINT]()
    var out = List[Byte]()
    out.reserve(len(data) * 2)
    for b in data:
        var cp = Codepoint(unsafe_unchecked_codepoint=UInt32(cp_table[Int(b)]))
        var base = len(out)
        out.resize(unsafe_uninit_length=base + cp.utf8_byte_length())
        _ = cp.unsafe_write_utf8[True](out.unsafe_ptr() + base)
    return String(unsafe_from_utf8=Span(out))


def gpt2_to_bytes(text: String) -> List[Byte]:
    var cp_table = materialize[CODEPOINT_TO_BYTE]()
    var out = List[Byte]()
    out.reserve(text.byte_length())
    for cp in text.codepoints():
        var val = Int(cp.to_u32())
        if val < 324:
            var byte_val = cp_table[val]
            if byte_val >= 0:
                out.append(Byte(byte_val))
    return out^


trait ByteTransformCapability(Movable, ImplicitlyDeletable):
    def encode_bytes(self, data: Span[Byte, _]) -> String:
        return bytes_to_gpt2(data)

    def decode_bytes(self, text: String) -> List[Byte]:
        return gpt2_to_bytes(text)


trait PreTokenizerCapability(Movable, ImplicitlyDeletable):
    def pre_tokenize(self, text: String) -> List[String]:
        ...


trait Tokenizer(Movable):
    def encode(mut self, text: String) -> List[Int]: ...
    def decode(self, ids: List[Int]) -> String: ...
    def vocab_size(self) -> Int: ...
    def token_to_id(self, token: String) -> Optional[Int]: ...
    def id_to_token(self, id: Int) -> Optional[String]: ...


def make_byte_to_codepoint() -> InlineArray[Int, 256]:
    var table = InlineArray[Int, 256](fill=0)
    var n = 256
    for b in range(256):
        if (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255):
            table[b] = b
        else:
            table[b] = n
            n += 1
    return table^


def make_codepoint_to_byte() -> InlineArray[Int, 324]:
    var table = InlineArray[Int, 324](fill=-1)
    var fwd = make_byte_to_codepoint()
    for b in range(256):
        table[fwd[b]] = b
    return table^


comptime BYTE_TO_CODEPOINT = make_byte_to_codepoint()
comptime CODEPOINT_TO_BYTE = make_codepoint_to_byte()


struct UnicodeContext(TrivialRegisterPassable):
    """Zero-size marker passed to classification functions. All table
    access goes through comptime indexing — no pointers, no lifetime issues."""
    def __init__(out self):
        pass


comptime ASCII_TAB: Byte = 0x09
comptime ASCII_LF: Byte = 0x0A
comptime ASCII_CR: Byte = 0x0D
comptime ASCII_SPACE: Byte = 0x20
comptime ASCII_APOSTROPHE: Byte = 0x27
comptime ASCII_SLASH: Byte = 0x2F


@always_inline
def is_ascii_letter(b: Byte) -> Bool:
    return ((b | Byte(0x20)) - Byte(97)) < Byte(26)


@always_inline
def is_ascii_digit(b: Byte) -> Bool:
    return (b - Byte(48)) < Byte(10)


@always_inline
def is_ascii_regex_space(b: Byte) -> Bool:
    return (b >= Byte(9) and b <= Byte(13)) or b == Byte(32)


@always_inline
def decode_utf8_codepoint(data: Span[Byte, _], pos: Int, n: Int) -> Tuple[UInt32, Int]:
    var b0 = data[pos]
    if b0 < Byte(0x80):
        return (UInt32(b0), 1)
    if b0 < Byte(0xE0):
        if pos + 1 < n:
            var cp = (UInt32(b0 & Byte(0x1F)) << UInt32(6)) | UInt32(data[pos + 1] & Byte(0x3F))
            return (cp, 2)
        return (UInt32(b0), 1)
    if b0 < Byte(0xF0):
        if pos + 2 < n:
            var cp = (
                (UInt32(b0 & Byte(0x0F)) << UInt32(12))
                | (UInt32(data[pos + 1] & Byte(0x3F)) << UInt32(6))
                | UInt32(data[pos + 2] & Byte(0x3F))
            )
            return (cp, 3)
        return (UInt32(b0), 1)
    if pos + 3 < n:
        var cp = (
            (UInt32(b0 & Byte(0x07)) << UInt32(18))
            | (UInt32(data[pos + 1] & Byte(0x3F)) << UInt32(12))
            | (UInt32(data[pos + 2] & Byte(0x3F)) << UInt32(6))
            | UInt32(data[pos + 3] & Byte(0x3F))
        )
        return (cp, 4)
    return (UInt32(b0), 1)


@always_inline
def in_comptime_ranges[
    ranges: InlineArray[UInt32, _],
    pair_count: Int,
](cp: UInt32) -> Bool:
    var lo = 0
    var hi = pair_count
    while lo < hi:
        var mid = (lo + hi) // 2
        var start = ranges[mid * 2]
        var end = ranges[mid * 2 + 1]
        if cp < start:
            hi = mid
        elif cp > end:
            lo = mid + 1
        else:
            return True
    return False


@always_inline
def is_unicode_letter_cp(cp: UInt32, ctx: UnicodeContext) -> Bool:
    if cp < UInt32(0x80):
        return is_ascii_letter(Byte(cp))
    if cp < LETTER_MIN or cp > LETTER_MAX:
        return False
    return in_comptime_ranges[LETTER_RANGES, LETTER_PAIR_COUNT](cp)


@always_inline
def is_unicode_number_cp(cp: UInt32, ctx: UnicodeContext) -> Bool:
    if cp < UInt32(0x80):
        return is_ascii_digit(Byte(cp))
    if cp < NUMBER_MIN or cp > NUMBER_MAX:
        return False
    return in_comptime_ranges[NUMBER_RANGES, NUMBER_PAIR_COUNT](cp)


@always_inline
def is_unicode_whitespace_cp(cp: UInt32, ctx: UnicodeContext) -> Bool:
    if cp < UInt32(0x80):
        return is_ascii_regex_space(Byte(cp))
    if cp < WHITESPACE_MIN or cp > WHITESPACE_MAX:
        return False
    return in_comptime_ranges[WHITESPACE_RANGES, WHITESPACE_PAIR_COUNT](cp)


@always_inline
def is_ascii_punct_symbol(b: Byte) -> Bool:
    return (
        (b >= Byte(33) and b <= Byte(47))
        or (b >= Byte(58) and b <= Byte(64))
        or (b >= Byte(91) and b <= Byte(96))
        or (b >= Byte(123) and b <= Byte(126))
    )


@always_inline
def is_unicode_punct_symbol_cp(cp: UInt32, ctx: UnicodeContext) -> Bool:
    if cp < UInt32(0x80):
        return is_ascii_punct_symbol(Byte(cp))
    if cp < PUNCT_SYMBOL_MIN or cp > PUNCT_SYMBOL_MAX:
        return False
    return in_comptime_ranges[PUNCT_SYMBOL_RANGES, PUNCT_SYMBOL_PAIR_COUNT](cp)


@always_inline
def is_unicode_mark_cp(cp: UInt32, ctx: UnicodeContext) -> Bool:
    if cp < MARK_MIN or cp > MARK_MAX:
        return False
    return in_comptime_ranges[MARK_RANGES, MARK_PAIR_COUNT](cp)


@always_inline
def is_number_start_at(data: Span[Byte, _], pos: Int, n: Int, ctx: UnicodeContext) -> Bool:
    var parsed = decode_utf8_codepoint(data, pos, n)
    return is_unicode_number_cp(parsed[0], ctx)


@always_inline
def is_whitespace_start_at(data: Span[Byte, _], pos: Int, n: Int, ctx: UnicodeContext) -> Bool:
    var parsed = decode_utf8_codepoint(data, pos, n)
    return is_unicode_whitespace_cp(parsed[0], ctx)


@always_inline
def span_to_string(data: Span[Byte, _], start: Int, end: Int) -> String:
    return String(unsafe_from_utf8=data.unsafe_subspan(offset=start, length=end - start))


comptime PRETOKENIZE_SIMD_WIDTH = 16


@always_inline
def simd_ascii_letters[w: Int](block: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    return ((block | Byte(0x20)) - Byte(97)).le(Byte(25))


@always_inline
def simd_ascii_digits[w: Int](block: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    return (block - Byte(48)).le(Byte(9))


@always_inline
def simd_spaces[w: Int](block: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    return (block - Byte(9)).le(Byte(4)) | block.eq(Byte(32))


@always_inline
def skip_while_matching[
    scalar_pred: def(Byte) thin -> Bool,
    simd_pred: def[w: Int](SIMD[DType.uint8, w]) thin -> SIMD[DType.bool, w],
    width: Int = PRETOKENIZE_SIMD_WIDTH,
](data: Span[Byte, _], pos: Int, n: Int) -> Int:
    comptime assert width > 0, "skip_while_matching requires positive width"
    var i = pos
    var data_ptr = data.unsafe_ptr()
    while i + width <= n:
        var block = (data_ptr + i).load[width=width]()
        var mask = simd_pred[width](block)
        if all(mask):
            i += width
            continue
        var positions = (~mask).select(
            iota[DType.int32, width](),
            SIMD[DType.int32, width](width),
        )
        return i + Int(positions.reduce_min())
    while i < n and scalar_pred(data[i]):
        i += 1
    return i


@always_inline
def consume_codepoint_run[
    pred: def(UInt32, UnicodeContext) thin -> Bool,
](data: Span[Byte, _], start: Int, n: Int, ctx: UnicodeContext) -> Int:
    var i = start
    while i < n:
        var parsed = decode_utf8_codepoint(data, i, n)
        if not pred(parsed[0], ctx):
            break
        i += parsed[1]
    return i


@always_inline
def consume_ascii_then_codepoint_run[
    ascii_pred: def(Byte) thin -> Bool,
    simd_pred: def[w: Int](SIMD[DType.uint8, w]) thin -> SIMD[DType.bool, w],
    cp_pred: def(UInt32, UnicodeContext) thin -> Bool,
](data: Span[Byte, _], start: Int, n: Int, ctx: UnicodeContext) -> Int:
    var i = start
    while i < n:
        var b = data[i]
        if ascii_pred(b):
            i = skip_while_matching[ascii_pred, simd_pred](data, i, n)
            continue
        var parsed = decode_utf8_codepoint(data, i, n)
        if cp_pred(parsed[0], ctx):
            i += parsed[1]
            continue
        break
    return i


@always_inline
def try_contraction_at[
    case_insensitive: Bool = False,
    accept_unicode_quote: Bool = False,
](data: Span[Byte, _], pos: Int, n: Int) -> Int:
    """Match an English-style contraction suffix `'s|'t|'d|'m|'ll|'re|'ve` at
    `pos`. Returns total bytes matched (apostrophe + suffix), or 0.

    `case_insensitive` ASCII-folds the suffix letters via `| 0x20`.
    `accept_unicode_quote` also matches U+2019 (UTF-8: E2 80 99) as the apostrophe."""
    if pos >= n:
        return 0
    var q = data[pos]
    var apostrophe_len = 1
    if q != Byte(39):
        comptime if accept_unicode_quote:
            if q != Byte(0xE2) or pos + 2 >= n:
                return 0
            if data[pos + 1] != Byte(0x80) or data[pos + 2] != Byte(0x99):
                return 0
            apostrophe_len = 3
        else:
            return 0

    if pos + apostrophe_len >= n:
        return 0

    comptime fold_mask = Byte(0x20) if case_insensitive else Byte(0)
    var c = data[pos + apostrophe_len] | fold_mask
    if c == Byte(115) or c == Byte(116) or c == Byte(100) or c == Byte(109):
        return apostrophe_len + 1
    if pos + apostrophe_len + 1 < n:
        var c2 = data[pos + apostrophe_len + 1] | fold_mask
        if c == Byte(108) and c2 == Byte(108): return apostrophe_len + 2
        if c == Byte(114) and c2 == Byte(101): return apostrophe_len + 2
        if c == Byte(118) and c2 == Byte(101): return apostrophe_len + 2
    return 0


def sort_strings_by_byte_length_desc(mut values: List[String]):
    for i in range(1, len(values)):
        var cur = values[i]
        var cur_len = cur.byte_length()
        var j = i
        while j > 0 and values[j - 1].byte_length() < cur_len:
            values[j] = values[j - 1]
            j -= 1
        values[j] = cur


@always_inline
def span_matches_at(data: Span[Byte, _], pos: Int, pattern: Span[Byte, _]) -> Bool:
    if pos + len(pattern) > len(data):
        return False
    for i in range(len(pattern)):
        if data[pos + i] != pattern[i]:
            return False
    return True


def find_added_token_match(
    text: Span[Byte, _],
    pos: Int,
    added_token_order: List[String],
    added_tokens: Dict[String, Int],
) -> Tuple[Int, Int]:
    for tok in added_token_order:
        var tok_bytes = tok.as_bytes()
        if span_matches_at(text, pos, tok_bytes):
            var found = added_tokens.get(tok)
            if found:
                return (found.value(), len(tok_bytes))
    return (-1, 0)


def split_numbers(
    piece: String, max_group: Int, ctx: UnicodeContext, mut out: List[String],
):
    """Split digit/number runs. max_group=1 for individual digits (GPT2),
    max_group=3 for DeepSeek-style 1-3 grouping."""
    var data = piece.as_bytes()
    var n = len(data)
    if n == 0:
        return

    var i = 0
    var chunk_start = 0
    while i < n:
        if is_number_start_at(data, i, n, ctx):
            if chunk_start < i:
                out.append(span_to_string(data, chunk_start, i))

            var j = i
            var count = 0
            while j < n and count < max_group and is_number_start_at(data, j, n, ctx):
                var parsed = decode_utf8_codepoint(data, j, n)
                j += parsed[1]
                count += 1

            out.append(span_to_string(data, i, j))
            i = j
            chunk_start = i
            continue

        var parsed = decode_utf8_codepoint(data, i, n)
        i += parsed[1]

    if chunk_start < n:
        out.append(span_to_string(data, chunk_start, n))


comptime HEX_DIGITS = "0123456789ABCDEF"


def byte_to_hex_token(b: Byte) -> String:
    var hex = HEX_DIGITS.as_bytes()
    var buf = List[Byte](capacity=6)
    buf.append(Byte(0x3C))
    buf.append(Byte(0x30))
    buf.append(Byte(0x78))
    buf.append(hex[Int(b) >> 4])
    buf.append(hex[Int(b) & 0xF])
    buf.append(Byte(0x3E))
    return String(unsafe_from_utf8=buf^)


struct BPETokenizer[
    pretokenizer_type: PreTokenizerCapability,
    byte_transform_type: ByteTransformCapability,
](Tokenizer):
    var vocab: Dict[String, Int]
    var vocab_rev: List[String]
    var merge_count: Int
    var merge_pair_ranks: Dict[UInt64, Int]
    var merge_pair_out: Dict[UInt64, Int]
    var added_tokens: Dict[String, Int]
    var added_token_order: List[String]
    var special_tokens: Dict[String, Int]
    var special_ids: Set[Int]
    var fuse_unk: Bool
    var byte_fallback: Bool
    var unk_token: String
    var unk_token_id: Int
    var add_bos_token: Bool
    var add_eos_token: Bool
    var bos_token_id: Int
    var eos_token_id: Int
    var _vocab_size: Int
    var use_piece_cache: Bool
    var piece_cache: PieceCache
    var hex_token_cache: List[String]
    var pretokenizer: Self.pretokenizer_type
    var byte_transform: Self.byte_transform_type

    def __init__(
        out self,
        var vocab: Dict[String, Int],
        merge_count: Int,
        var merge_pair_ranks: Dict[UInt64, Int],
        var merge_pair_out: Dict[UInt64, Int],
        var added_tokens: Dict[String, Int],
        var added_token_order: List[String],
        var special_tokens: Dict[String, Int],
        var special_ids: Set[Int],
        fuse_unk: Bool,
        byte_fallback: Bool,
        unk_token: String,
        add_bos_token: Bool,
        add_eos_token: Bool,
        bos_token_id: Int,
        eos_token_id: Int,
        vocab_size: Int,
        use_piece_cache: Bool,
        var pretokenizer: Self.pretokenizer_type,
        var byte_transform: Self.byte_transform_type,
    ):
        var max_id = vocab_size - 1
        for item in vocab.items():
            if item.value > max_id:
                max_id = item.value
        for item in added_tokens.items():
            if item.value > max_id:
                max_id = item.value
        var rev_size = max_id + 1

        var vocab_rev = List[String](length=rev_size, fill=String(""))
        for item in vocab.items():
            var id = item.value
            if id >= 0 and id < rev_size:
                vocab_rev[id] = item.key.copy()
        for item in added_tokens.items():
            var id = item.value
            if id >= 0 and id < rev_size:
                vocab_rev[id] = item.key.copy()

        sort_strings_by_byte_length_desc(added_token_order)

        var unk_id = -1
        if unk_token.byte_length() > 0:
            var found = vocab.get(unk_token)
            if found:
                unk_id = found.value()

        self.vocab = vocab^
        self.vocab_rev = vocab_rev^
        self.merge_count = merge_count
        self.merge_pair_ranks = merge_pair_ranks^
        self.merge_pair_out = merge_pair_out^
        self.added_tokens = added_tokens^
        self.added_token_order = added_token_order^
        self.special_tokens = special_tokens^
        self.special_ids = special_ids^
        self.fuse_unk = fuse_unk
        self.byte_fallback = byte_fallback
        self.unk_token = unk_token
        self.unk_token_id = unk_id
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self._vocab_size = vocab_size
        self.use_piece_cache = use_piece_cache
        self.piece_cache = PieceCache()
        var hex_cache = List[String](capacity=256)
        for b in range(256):
            hex_cache.append(byte_to_hex_token(Byte(b)))
        self.hex_token_cache = hex_cache^
        self.pretokenizer = pretokenizer^
        self.byte_transform = byte_transform^

    def vocab_size(self) -> Int:
        return self._vocab_size

    def token_to_id(self, token: String) -> Optional[Int]:
        var found = self.vocab.get(token)
        if found:
            return found.value()
        var added = self.added_tokens.get(token)
        if added:
            return added.value()
        return None

    def id_to_token(self, id: Int) -> Optional[String]:
        if id >= 0 and id < len(self.vocab_rev):
            var tok = self.vocab_rev[id]
            if tok.byte_length() > 0:
                return tok
        return None

    def is_special_token(self, token: String) -> Bool:
        return self.special_tokens.__contains__(token)

    def is_special_id(self, id: Int) -> Bool:
        return id in self.special_ids

    def num_merges(self) -> Int:
        return self.merge_count

    def num_special_tokens(self) -> Int:
        return len(self.special_tokens)

    def encode_piece(mut self, piece: String, mut ids: List[Int]):
        if piece.byte_length() == 0:
            return

        if self.use_piece_cache and self.piece_cache.get(piece, ids):
            return

        var transformed = self.byte_transform.encode_bytes(piece.as_bytes())
        var symbol_ids = List[Int]()
        for slice in transformed.codepoint_slices():
            var found = self.vocab.get(String(slice))
            if found:
                symbol_ids.append(found.value())
            elif self.byte_fallback:
                var ch_bytes = slice.as_bytes()
                for j in range(slice.byte_length()):
                    var fb = self.vocab.get(self.hex_token_cache[Int(ch_bytes[j])])
                    if fb:
                        symbol_ids.append(fb.value())
                    elif self.unk_token_id >= 0:
                        symbol_ids.append(self.unk_token_id)
            elif self.unk_token_id >= 0:
                symbol_ids.append(self.unk_token_id)
        symbol_ids = bpe_merge_ids(symbol_ids, self.merge_pair_ranks, self.merge_pair_out)

        if self.use_piece_cache:
            self.piece_cache.put(piece.copy(), symbol_ids)

        for id in symbol_ids:
            ids.append(id)

    def encode_span(mut self, data: Span[Byte, _], start: Int, end: Int, mut ids: List[Int]):
        if end <= start:
            return
        var chunk = span_to_string(data, start, end)
        var pieces = self.pretokenizer.pre_tokenize(chunk)
        for piece in pieces:
            self.encode_piece(piece, ids)

    def encode(mut self, text: String) -> List[Int]:
        var ids = List[Int]()

        if self.add_bos_token and self.bos_token_id >= 0:
            ids.append(self.bos_token_id)

        if text.byte_length() > 0:
            var data = text.as_bytes()
            var n = len(data)
            var i = 0
            var chunk_start = 0
            while i < n:
                var matched = find_added_token_match(data, i, self.added_token_order, self.added_tokens)
                var tok_id = matched[0]
                var tok_len = matched[1]
                if tok_id >= 0 and tok_len > 0:
                    if chunk_start < i:
                        self.encode_span(data, chunk_start, i, ids)
                    ids.append(tok_id)
                    i += tok_len
                    chunk_start = i
                    continue
                i += 1
            if chunk_start < n:
                self.encode_span(data, chunk_start, n, ids)

        if self.add_eos_token and self.eos_token_id >= 0:
            ids.append(self.eos_token_id)

        return ids^

    def decode(self, ids: List[Int]) -> String:
        var encoded_parts = List[Byte]()
        var decoded = String()
        for id in ids:
            if id < 0 or id >= len(self.vocab_rev):
                continue

            var tok = self.vocab_rev[id]
            if self.is_special_id(id):
                if len(encoded_parts) > 0:
                    var raw_bytes = self.byte_transform.decode_bytes(
                        String(unsafe_from_utf8=Span(encoded_parts))
                    )
                    decoded += String(from_utf8_lossy=Span(raw_bytes))
                    encoded_parts.resize(unsafe_uninit_length=0)
                decoded += tok
                continue

            var tok_data = tok.as_bytes()
            for j in range(len(tok_data)):
                encoded_parts.append(tok_data[j])

        if len(encoded_parts) > 0:
            var raw_bytes = self.byte_transform.decode_bytes(
                String(unsafe_from_utf8=Span(encoded_parts))
            )
            decoded += String(from_utf8_lossy=Span(raw_bytes))
        return decoded^
