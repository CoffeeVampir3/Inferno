from std.memory import Span

from .tokenizer import (
    BPETokenizer, PreTokenizerCapability, ByteTransformCapability,
)


def utf8_stable_stop(read buf: List[Byte]) -> Int:
    var n = len(buf)
    if n == 0:
        return 0
    var i = n - 1
    while i > 0 and (buf[i] & Byte(0xC0)) == Byte(0x80):
        i -= 1
    var lead = buf[i]
    var need = 1
    if lead >= Byte(0xF0):
        need = 4
    elif lead >= Byte(0xE0):
        need = 3
    elif lead >= Byte(0xC0):
        need = 2
    if i + need <= n:
        return n
    return i


struct StreamDetokenizer(Movable):
    var tail: List[Byte]

    def __init__(out self):
        self.tail = List[Byte]()

    def reset(mut self):
        self.tail.clear()

    def drain_tail(mut self) -> String:
        if len(self.tail) == 0:
            return String("")
        var out = String(from_utf8_lossy=Span(self.tail))
        self.tail.clear()
        return out^

    def flush(mut self) -> String:
        return self.drain_tail()

    def push[
        Pre: PreTokenizerCapability, Bt: ByteTransformCapability, //,
    ](mut self, read tok: BPETokenizer[Pre, Bt], id: Int32) -> String:
        var i = Int(id)
        if i < 0 or i >= len(tok.vocab_rev):
            return String("")
        if tok.is_special_id(i):
            var out = self.drain_tail()
            out += tok.vocab_rev[i]
            return out^
        var raw = tok.byte_transform.decode_bytes(tok.vocab_rev[i])
        var buf = List[Byte](capacity=len(self.tail) + len(raw))
        for k in range(len(self.tail)):
            buf.append(self.tail[k])
        for k in range(len(raw)):
            buf.append(raw[k])
        var stop = utf8_stable_stop(buf)
        var out = String(from_utf8_lossy=Span(buf).unsafe_subspan(
            offset=0, length=stop))
        self.tail.clear()
        for k in range(stop, len(buf)):
            self.tail.append(buf[k])
        return out^
