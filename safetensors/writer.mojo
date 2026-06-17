from std.memory import Span
from std.pathlib import Path

from safetensors.parser import dtype_tag


@fieldwise_init
struct OutputEntry(Copyable, Movable):
    var name: String
    var dtype: DType
    var shape0: Int
    var shape1: Int
    var data_start: Int
    var data_end: Int


@always_inline
def append_static(mut buf: List[UInt8], s: StaticString):
    var bytes = s.as_bytes()
    for i in range(len(bytes)):
        buf.append(bytes[i])


@always_inline
def append_string(mut buf: List[UInt8], s: String):
    var bytes = s.as_bytes()
    for i in range(len(bytes)):
        buf.append(bytes[i])


@always_inline
def append_int(mut buf: List[UInt8], v: Int):
    var s = String(v)
    var bytes = s.as_bytes()
    for i in range(len(bytes)):
        buf.append(bytes[i])


def build_header(ref entries: List[OutputEntry]) -> List[UInt8]:
    """Emit the safetensors header preceded by its 8-byte little-endian length.
    The header bytes start at offset 8 of the returned buffer; the leading 8
    bytes are patched in after the JSON length is known."""
    var buf = List[UInt8](capacity=128 * 1024)
    for _ in range(8):
        buf.append(0)
    comptime JSON_START = 8

    buf.append(0x7B)  # '{'
    for i in range(len(entries)):
        if i > 0:
            buf.append(0x2C)  # ','
        ref e = entries[i]
        buf.append(0x22)  # '"'
        append_string(buf, e.name)
        append_static(buf, '":{"dtype":"')
        append_static(buf, dtype_tag(e.dtype))
        append_static(buf, '","shape":[')
        append_int(buf, e.shape0)
        if e.shape1 > 0:
            buf.append(0x2C)
            append_int(buf, e.shape1)
        append_static(buf, '],"data_offsets":[')
        append_int(buf, e.data_start)
        buf.append(0x2C)
        append_int(buf, e.data_end)
        append_static(buf, "]}")
    buf.append(0x7D)  # '}'

    while (len(buf) - JSON_START) % 8 != 0:
        buf.append(0x20)  # ' '

    var json_len = UInt64(len(buf) - JSON_START)
    for i in range(8):
        buf[i] = UInt8((json_len >> UInt64(i * 8)) & 0xFF)
    return buf^


def write_safetensors(
    path: Path, ref entries: List[OutputEntry], read payload: List[UInt8],
) -> Bool:
    var header = build_header(entries)
    try:
        with open(path, "w") as f:
            f.write_bytes(Span(header))
            f.write_bytes(Span(payload))
        return True
    except e:
        print(t"write_safetensors: failed to write {path}: {e}")
        return False
