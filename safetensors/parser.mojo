from std.bit import byte_swap
from std.collections import Dict
from std.memory import Span, UnsafePointer
from std.pathlib import Path
from std.sys.info import size_of, is_big_endian

from jsontools.parser import (
    Parser,
    ParseError,
    LBRACE,
    RBRACE,
    LBRACKET,
    RBRACKET,
)

comptime HEADER_LEN_BYTES = 8
comptime MAX_HEADER_SIZE = 100 * 1024 * 1024

def parse_dtype(s: String) -> DType:
    if s == "BOOL":
        return DType.bool
    if s == "U8":
        return DType.uint8
    if s == "I8":
        return DType.int8
    if s == "I16":
        return DType.int16
    if s == "U16":
        return DType.uint16
    if s == "F16":
        return DType.float16
    if s == "BF16":
        return DType.bfloat16
    if s == "I32":
        return DType.int32
    if s == "U32":
        return DType.uint32
    if s == "F32":
        return DType.float32
    if s == "F64":
        return DType.float64
    if s == "F8_E4M3" or s == "F8_E4M3FN":
        return DType.float8_e4m3fn
    if s == "F8_E5M2":
        return DType.float8_e5m2
    if s == "I64":
        return DType.int64
    if s == "U64":
        return DType.uint64
    return DType.invalid


def dtype_tag(dt: DType) -> StaticString:
    if dt == DType.bool: return "BOOL"
    if dt == DType.uint8: return "U8"
    if dt == DType.int8: return "I8"
    if dt == DType.int16: return "I16"
    if dt == DType.uint16: return "U16"
    if dt == DType.float16: return "F16"
    if dt == DType.bfloat16: return "BF16"
    if dt == DType.int32: return "I32"
    if dt == DType.uint32: return "U32"
    if dt == DType.float32: return "F32"
    if dt == DType.float64: return "F64"
    if dt == DType.float8_e4m3fn: return "F8_E4M3"
    if dt == DType.float8_e5m2: return "F8_E5M2"
    if dt == DType.int64: return "I64"
    if dt == DType.uint64: return "U64"
    return "UNKNOWN"

def dtype_byte_size(dtype: DType) -> Int:
    if dtype == DType.bool:
        return size_of[Scalar[DType.bool]]()
    if dtype == DType.uint8:
        return size_of[UInt8]()
    if dtype == DType.int8:
        return size_of[Int8]()
    if dtype == DType.int16:
        return size_of[Int16]()
    if dtype == DType.uint16:
        return size_of[UInt16]()
    if dtype == DType.float16:
        return size_of[Float16]()
    if dtype == DType.bfloat16:
        return size_of[BFloat16]()
    if dtype == DType.int32:
        return size_of[Int32]()
    if dtype == DType.uint32:
        return size_of[UInt32]()
    if dtype == DType.float32:
        return size_of[Float32]()
    if dtype == DType.float64:
        return size_of[Float64]()
    if dtype == DType.float8_e4m3fn:
        return size_of[Float8_e4m3fn]()
    if dtype == DType.float8_e5m2:
        return size_of[Float8_e5m2]()
    if dtype == DType.int64:
        return size_of[Int64]()
    if dtype == DType.uint64:
        return size_of[UInt64]()
    return -1

def checked_numel(shape: List[Int]) -> Optional[Int]:
    var n = 1
    for dim in shape:
        if dim < 0:
            return None
        if dim != 0 and n > Int.MAX // dim:
            return None
        n *= dim
    return n

def checked_tensor_byte_size(dtype: DType, shape: List[Int]) -> Optional[Int]:
    var elem_size = dtype_byte_size(dtype)
    if elem_size <= 0:
        return None
    var numel = checked_numel(shape)
    if not numel:
        return None
    if numel.value() != 0 and numel.value() > Int.MAX // elem_size:
        return None
    return numel.value() * elem_size

struct TensorMeta(Copyable, Writable):
    var dtype: DType
    var shape: List[Int]
    var start: Int
    var end: Int

    def __init__(out self, dtype: DType, var shape: List[Int], start: Int, end: Int):
        self.dtype = dtype
        self.shape = shape^
        self.start = start
        self.end = end

    def byte_size(self) -> Int:
        return self.end - self.start

    def numel(self) -> Int:
        var n = 1
        for dim in self.shape:
            n *= dim
        return n

@fieldwise_init
struct SafetensorsHeader(Copyable):
    var path: Path
    var tensors: Dict[String, TensorMeta]
    var data_offset: Int
    var file_len: Int

def parse_offsets(mut parser: Parser) raises ParseError -> Tuple[Int, Int]:
    if not parser.consume(LBRACKET):
        raise ParseError("expected '[' for offsets", parser.pos)
    parser.skip_whitespace()
    var start_val = parser.parse_uint()
    if not parser.delimited_next(RBRACKET):
        raise ParseError("expected two offsets", parser.pos)
    var end_val = parser.parse_uint()
    parser.skip_whitespace()
    if not parser.consume(RBRACKET):
        raise ParseError("expected ']' after offsets", parser.pos)
    return (start_val, end_val)

def parse_shape(mut parser: Parser) raises ParseError -> List[Int]:
    if not parser.consume(LBRACKET):
        raise ParseError("expected '[' for shape", parser.pos)
    parser.skip_whitespace()
    var shape = List[Int]()
    if parser.consume(RBRACKET):
        return shape^
    while True:
        shape.append(parser.parse_uint())
        if not parser.delimited_next(RBRACKET):
            break
    return shape^

def parse_tensor(mut parser: Parser) raises ParseError -> TensorMeta:
    if not parser.consume(LBRACE):
        raise ParseError("expected '{' for tensor", parser.pos)
    parser.skip_whitespace()
    if parser.consume(RBRACE):
        raise ParseError("empty tensor object", parser.pos)
    var has_offsets = False
    var has_dtype = False
    var has_shape = False
    var start = 0
    var end = 0
    var dtype = DType.invalid
    var shape = List[Int]()
    while True:
        var key_val = parser.object_key()
        if key_val == "data_offsets":
            var offs = parse_offsets(parser)
            start = offs[0]
            end = offs[1]
            has_offsets = True
        elif key_val == "dtype":
            var dtype_str = parser.parse_string()
            dtype = parse_dtype(dtype_str)
            has_dtype = True
        elif key_val == "shape":
            shape = parse_shape(parser)
            has_shape = True
        else:
            parser.skip_value()
        if not parser.delimited_next(RBRACE):
            break
    if not has_offsets or not has_dtype or not has_shape:
        raise ParseError("tensor missing required fields (dtype, shape, data_offsets)", parser.pos)
    if dtype == DType.invalid:
        raise ParseError("unsupported tensor dtype", parser.pos)
    return TensorMeta(dtype, shape^, start, end)

def parse_safetensors_dict(mut parser: Parser, payload_size: Int = -1) raises ParseError -> Dict[String, TensorMeta]:
    var tensors = Dict[String, TensorMeta]()
    var range_starts = List[Int]()
    var range_ends = List[Int]()
    var indexed_bytes = 0
    parser.skip_whitespace()
    if not parser.consume(LBRACE):
        raise ParseError("expected '{' at start", parser.pos)
    parser.skip_whitespace()
    if parser.consume(RBRACE):
        if payload_size > 0:
            raise ParseError("payload bytes are not indexed by tensor metadata", parser.pos)
        return tensors^
    while True:
        var key_value = parser.object_key()
        if key_value == "__metadata__":
            parser.skip_value()
        else:
            if key_value in tensors:
                raise ParseError("duplicate tensor name", parser.pos)
            var meta = parse_tensor(parser)
            if meta.end < meta.start:
                raise ParseError("invalid tensor offsets", parser.pos)
            if payload_size >= 0 and meta.end > payload_size:
                raise ParseError("tensor offsets exceed file payload", parser.pos)
            var expected_size = checked_tensor_byte_size(meta.dtype, meta.shape)
            if not expected_size:
                raise ParseError("invalid tensor shape byte size", parser.pos)
            if meta.byte_size() != expected_size.value():
                raise ParseError("tensor byte size does not match dtype and shape", parser.pos)
            for i in range(len(range_starts)):
                if meta.start < range_ends[i] and meta.end > range_starts[i]:
                    raise ParseError("overlapping tensor data offsets", parser.pos)
            if expected_size.value() != 0 and indexed_bytes > Int.MAX - expected_size.value():
                raise ParseError("indexed tensor byte count overflow", parser.pos)
            indexed_bytes += expected_size.value()
            range_starts.append(meta.start)
            range_ends.append(meta.end)
            tensors[key_value^] = meta^
        if not parser.delimited_next(RBRACE):
            break
    parser.skip_whitespace()
    if parser.has_more():
        raise ParseError("trailing content after root object", parser.pos)
    if payload_size >= 0 and indexed_bytes != payload_size:
        raise ParseError("payload bytes are not fully indexed by tensor metadata", parser.pos)
    return tensors^

def read_u64_le(ptr: UnsafePointer[Byte, _]) -> UInt64:
    var v = ptr.bitcast[UInt64]()[]
    comptime if is_big_endian():
        return byte_swap(v)
    return v

def parse_safetensors_header[simd_width: Int = 16](path: Path) -> Optional[SafetensorsHeader]:
    var header_bytes: List[Byte]
    var header_size: Int
    var file_len: UInt64
    try:
        with open(path, "r") as f:
            file_len = f.seek(0, 2)
            _ = f.seek(0, 0)
            if file_len < UInt64(HEADER_LEN_BYTES):
                print("load: file too small")
                return None
            var header_len_bytes = f.read_bytes(size=HEADER_LEN_BYTES)
            if len(header_len_bytes) != HEADER_LEN_BYTES:
                print("load: file too small")
                return None
            var header_len = read_u64_le(header_len_bytes.unsafe_ptr())
            if header_len > UInt64(MAX_HEADER_SIZE):
                print("load: header too large")
                return None
            if header_len > file_len - UInt64(HEADER_LEN_BYTES):
                print("load: header length exceeds file")
                return None
            header_size = Int(header_len)
            header_bytes = List[Byte](unsafe_uninit_length=header_size)
            var bytes_read = f.read(Span(header_bytes))
            if bytes_read != header_size:
                print("load: header length exceeds file")
                return None
    except e:
        print(t"load: failed to read file: {e}")
        return None
    var parser = Parser[simd_width=simd_width](Span(header_bytes))
    try:
        var tensors = parse_safetensors_dict(parser, Int(file_len) - HEADER_LEN_BYTES - header_size)
        return SafetensorsHeader(path, tensors^, HEADER_LEN_BYTES + header_size, Int(file_len))
    except e:
        print(t"load: parse error at pos {e.pos}: {e.message}")
        return None
