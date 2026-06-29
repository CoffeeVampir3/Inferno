from std.memory import Span, UnsafePointer

import linux.sys as linux


comptime READ_CHUNK = 65536
comptime POLL_CHUNK = 256

comptime PASTE_ON = "\x1b[?2004h"
comptime PASTE_OFF = "\x1b[?2004l"
comptime PLACEHOLDER_OPEN = "\x1b[90m"
comptime PLACEHOLDER_CLOSE = "\x1b[0m"

comptime CTRL_C = Byte(3)
comptime CTRL_D = Byte(4)
comptime CTRL_W = Byte(23)
comptime BACKSPACE = Byte(8)
comptime LF = Byte(10)
comptime CR = Byte(13)
comptime ESC = Byte(27)
comptime SPACE = Byte(32)
comptime DIGIT0 = Byte(48)
comptime DIGIT1 = Byte(49)
comptime DIGIT2 = Byte(50)
comptime SS3_OPEN = Byte(79)     # 'O'
comptime CSI_OPEN = Byte(91)     # '['
comptime TILDE = Byte(126)
comptime DEL = Byte(127)
comptime CSI_FINAL_LO = Byte(0x40)
comptime CSI_FINAL_HI = Byte(0x7e)
comptime UTF8_CONT = Byte(0x80)
comptime UTF8_LEAD2 = Byte(0xC0)
comptime UTF8_LEAD3 = Byte(0xE0)
comptime UTF8_LEAD4 = Byte(0xF0)

comptime END_MARKER_LEN = 6

comptime PASTE_INLINE_LIMIT = 1024


@fieldwise_init
struct EditUnit(Copyable, Movable):
    var byte_count: Int
    var view_count: Int
    var cell_count: Int


struct RawGuard(Movable):
    """Linear scope guard that puts the tty into cbreak input mode.

    Construction switches off canonical mode, echo, and signal generation
    and turns on bracketed paste. The destructor restores the saved
    settings, so the terminal can never be left raw by a missed reset.
    """

    var fd: Int
    var saved: linux.Termios
    var active: Bool

    def __init__(out self, fd: Int):
        self.fd = fd
        self.saved = linux.Termios()
        self.active = False
        var sys = linux.linux_sys()
        if sys.sys_tcgetattr(self.fd, UnsafePointer(to=self.saved)) != 0:
            return
        var work = self.saved.copy()
        work.c_lflag &= ~UInt32(
            linux.TermLocalFlag.ICANON | linux.TermLocalFlag.ECHO
            | linux.TermLocalFlag.ISIG | linux.TermLocalFlag.IEXTEN)
        work.c_iflag &= ~UInt32(
            linux.TermInputFlag.ICRNL | linux.TermInputFlag.IXON)
        work.c_cc[linux.TermControlChar.VMIN] = UInt8(1)
        work.c_cc[linux.TermControlChar.VTIME] = UInt8(0)
        _ = sys.sys_tcsetattr(self.fd, UnsafePointer(to=work))
        _ = work.c_lflag
        print(PASTE_ON, end="", flush=True)
        self.active = True

    def __del__(deinit self):
        if self.active:
            print(PASTE_OFF, end="", flush=True)
            _ = linux.linux_sys().sys_tcsetattr(
                self.fd, UnsafePointer(to=self.saved))


struct CancelWatch(Movable):
    """Non-blocking watch for an interrupt during model generation.

    Construction puts the tty into cbreak mode with signal generation off
    and reads that return immediately, so `triggered` can poll for Esc or
    Ctrl+C between decode steps without ever blocking the generation loop.
    The destructor restores the saved settings, so the terminal is left
    exactly as it was found.
    """

    var fd: Int
    var saved: linux.Termios
    var active: Bool

    def __init__(out self, fd: Int = 0):
        self.fd = fd
        self.saved = linux.Termios()
        self.active = False
        var sys = linux.linux_sys()
        if sys.sys_tcgetattr(self.fd, UnsafePointer(to=self.saved)) != 0:
            return
        var work = self.saved.copy()
        work.c_lflag &= ~UInt32(
            linux.TermLocalFlag.ICANON | linux.TermLocalFlag.ECHO
            | linux.TermLocalFlag.ISIG | linux.TermLocalFlag.IEXTEN)
        work.c_iflag &= ~UInt32(
            linux.TermInputFlag.ICRNL | linux.TermInputFlag.IXON)
        work.c_cc[linux.TermControlChar.VMIN] = UInt8(0)
        work.c_cc[linux.TermControlChar.VTIME] = UInt8(0)
        _ = sys.sys_tcsetattr(self.fd, UnsafePointer(to=work))
        self.active = True

    def triggered(mut self) -> Bool:
        # Drain whatever is queued without blocking (VMIN/VTIME are 0, so a
        # read returns at once) and report whether Esc or Ctrl+C is among it.
        if not self.active:
            return False
        var sys = linux.linux_sys()
        var chunk = InlineArray[Byte, POLL_CHUNK](fill=Byte(0))
        var hit = False
        while True:
            var rc = sys.sys_read(self.fd, Int(UnsafePointer(to=chunk)), POLL_CHUNK)
            if rc == linux.EINTR:
                continue
            if rc <= 0:
                break
            for i in range(rc):
                if chunk[i] == CTRL_C or chunk[i] == ESC:
                    hit = True
            if rc < POLL_CHUNK:
                break
        return hit

    def __del__(deinit self):
        if self.active:
            _ = linux.linux_sys().sys_tcsetattr(
                self.fd, UnsafePointer(to=self.saved))


struct TerminalReader(Movable):
    var fd: Int
    var buf: List[Byte]
    var pos: Int
    var eof: Bool

    def __init__(out self, fd: Int = 0):
        self.fd = fd
        self.buf = List[Byte]()
        self.pos = 0
        self.eof = False

    def bytes_to_string(self, read b: List[Byte]) -> String:
        return String(unsafe_from_utf8=Span(b))

    def drain_chunk(mut self) -> Int:
        # Read one chunk straight into the tail of `buf` (no intermediate copy),
        # so reads stay back-to-back and the tty's input buffer is not left to
        # overflow while we parse. EINTR/EAGAIN are transient, not end of input.
        var old = len(self.buf)
        self.buf.resize(unsafe_uninit_length=old + READ_CHUNK)
        var sys = linux.linux_sys()
        var got: Int
        while True:
            var rc = sys.sys_read(
                self.fd, Int(self.buf.unsafe_ptr() + old), READ_CHUNK)
            if rc == linux.EINTR or rc == linux.EAGAIN:
                continue
            got = rc
            break
        if got > 0:
            self.buf.resize(unsafe_uninit_length=old + got)
        else:
            self.buf.resize(unsafe_uninit_length=old)
            self.eof = True
        return got

    def refill(mut self):
        if self.pos > 0:
            var rem = List[Byte]()
            for i in range(self.pos, len(self.buf)):
                rem.append(self.buf[i])
            self.buf = rem^
            self.pos = 0
        _ = self.drain_chunk()

    def next_byte(mut self) -> Optional[Byte]:
        while self.pos >= len(self.buf):
            if self.eof:
                return None
            self.refill()
        var b = self.buf[self.pos]
        self.pos += 1
        return b

    def read_message(mut self, read prompt: String) -> Optional[String]:
        var guard = RawGuard(self.fd)
        var raw = guard.active
        var result: Optional[String]
        if raw:
            var cols = self.query_cols()
            var split = self.split_prompt(prompt)
            print(split[0], end="", flush=True)
            result = self.input_loop(split[1], self.visible_cells(split[1]), cols)
        else:
            print(prompt, end="", flush=True)
            result = self.read_line_plain()
        _ = guard^
        return result^

    def query_cols(self) -> Int:
        # Width of the terminal in cells; the editor wraps and re-anchors against
        # it. Falls back to 80 when stdin is not a tty or the ioctl is refused.
        var ws = linux.Winsize()
        var sys = linux.linux_sys()
        if sys.sys_get_winsize(self.fd, UnsafePointer(to=ws)) != 0:
            return 80
        if ws.ws_col == 0:
            return 80
        return Int(ws.ws_col)

    def split_prompt(self, read prompt: String) -> Tuple[String, String]:
        # Split into a lead (everything through the last newline, printed once to
        # advance onto a fresh row) and a label (the visible prompt that shares
        # the input row and is redrawn on every edit).
        var data = prompt.as_bytes()
        var n = len(data)
        var cut = 0
        for i in range(n):
            if data[i] == LF:
                cut = i + 1
        var lead = List[Byte]()
        for i in range(cut):
            lead.append(data[i])
        var label = List[Byte]()
        for i in range(cut, n):
            label.append(data[i])
        return (self.bytes_to_string(lead), self.bytes_to_string(label))

    def visible_cells(self, read s: String) -> Int:
        # Display width of a string, skipping CSI/SS3 escapes and UTF-8
        # continuation bytes so colored prompts measure their printed glyphs.
        var data = s.as_bytes()
        var n = len(data)
        var i = 0
        var cells = 0
        while i < n:
            var c = data[i]
            if c == ESC:
                i += 1
                if i < n and data[i] == CSI_OPEN:
                    i += 1
                    while i < n and not (data[i] >= CSI_FINAL_LO
                                         and data[i] <= CSI_FINAL_HI):
                        i += 1
                    if i < n:
                        i += 1
                elif i < n:
                    i += 1
                continue
            if c >= UTF8_CONT and c < UTF8_LEAD2:
                i += 1
                continue
            cells += 1
            i += 1
        return cells

    def render(
        mut self,
        read label: String,
        prompt_cells: Int,
        read view: List[Byte],
        read units: List[EditUnit],
        cols: Int,
        mut maxrows: Int,
    ):
        # Redraw the whole input region in place. The cursor always sits at the
        # end, so we step up to the first input row, clear to the end of the
        # screen, and reprint the label and view. This is what lets edits wrap
        # back across row boundaries instead of trapping at column zero.
        var content = 0
        for i in range(len(units)):
            content += units[i].cell_count
        var cells = prompt_cells + content
        var rows = cells // cols
        if cells % cols != 0:
            rows += 1
        if rows < 1:
            rows = 1
        var out = String("")
        if maxrows > 1:
            out += "\x1b[" + String(maxrows - 1) + "A"
        out += "\r\x1b[0J"
        out += label
        out += self.bytes_to_string(view)
        maxrows = rows
        print(out, end="", flush=True)

    def read_line_plain(mut self) -> Optional[String]:
        var line = List[Byte]()
        while True:
            var b_opt = self.next_byte()
            if not b_opt:
                if len(line) == 0:
                    return None
                return self.bytes_to_string(line)
            var b = b_opt.value()
            if b == LF:
                return self.bytes_to_string(line)
            if b == CR:
                continue
            line.append(b)

    def input_loop(mut self, read label: String, prompt_cells: Int, cols: Int) -> Optional[String]:
        var line = List[Byte]()
        var view = List[Byte]()
        var units = List[EditUnit]()
        var maxrows = 1
        self.render(label, prompt_cells, view, units, cols, maxrows)
        while True:
            var b_opt = self.next_byte()
            if not b_opt:
                if len(line) == 0:
                    return None
                print("\r\n", end="", flush=True)
                return self.bytes_to_string(line)
            var b = b_opt.value()

            if b == CR or b == LF:
                print("\r\n", end="", flush=True)
                return self.bytes_to_string(line)
            elif b == CTRL_C:
                print("^C\r\n", end="", flush=True)
                return None
            elif b == CTRL_D:
                if len(line) == 0:
                    return None
                continue
            elif b == CTRL_W:
                self.do_word_delete(line, view, units)
            elif b == DEL or b == BACKSPACE:
                self.do_backspace(line, view, units)
            elif b == ESC:
                self.handle_escape(line, view, units)
            elif b < SPACE:
                continue
            else:
                self.handle_text(b, line, view, units)
            self.render(label, prompt_cells, view, units, cols, maxrows)

    def do_backspace(mut self, mut line: List[Byte], mut view: List[Byte], mut units: List[EditUnit]):
        if len(units) == 0:
            return
        var unit = units.pop()
        for _ in range(unit.byte_count):
            _ = line.pop()
        for _ in range(unit.view_count):
            _ = view.pop()

    def do_word_delete(mut self, mut line: List[Byte], mut view: List[Byte], mut units: List[EditUnit]):
        while len(units) > 0:
            if not (units[len(units) - 1].byte_count == 1
                    and line[len(line) - 1] == SPACE):
                break
            self.do_backspace(line, view, units)
        while len(units) > 0:
            if (units[len(units) - 1].byte_count == 1
                    and line[len(line) - 1] == SPACE):
                break
            self.do_backspace(line, view, units)

    def handle_text(mut self, lead: Byte, mut line: List[Byte], mut view: List[Byte], mut units: List[EditUnit]):
        var width = 1
        if lead >= UTF8_LEAD4:
            width = 4
        elif lead >= UTF8_LEAD3:
            width = 3
        elif lead >= UTF8_LEAD2:
            width = 2
        var count = 1
        line.append(lead)
        view.append(lead)
        for _ in range(width - 1):
            var nb_opt = self.next_byte()
            if not nb_opt:
                break
            var nb = nb_opt.value()
            if nb < UTF8_CONT or nb >= UTF8_LEAD2:
                self.pos -= 1
                break
            line.append(nb)
            view.append(nb)
            count += 1
        units.append(EditUnit(count, count, 1))

    def handle_escape(mut self, mut line: List[Byte], mut view: List[Byte], mut units: List[EditUnit]):
        var b_opt = self.next_byte()
        if not b_opt:
            return
        var b = b_opt.value()
        if b == CSI_OPEN:
            self.handle_csi(line, view, units)
        elif b == SS3_OPEN:
            _ = self.next_byte()
        elif b == DEL or b == BACKSPACE:
            self.do_word_delete(line, view, units)

    def handle_csi(mut self, mut line: List[Byte], mut view: List[Byte], mut units: List[EditUnit]):
        var params = List[Byte]()
        while True:
            var b_opt = self.next_byte()
            if not b_opt:
                return
            var b = b_opt.value()
            if b >= CSI_FINAL_LO and b <= CSI_FINAL_HI:
                if b == TILDE and Self.is_paste_start(params):
                    self.read_paste(line, view, units)
                return
            params.append(b)

    @staticmethod
    def is_paste_start(read params: List[Byte]) -> Bool:
        return (len(params) == 3 and params[0] == DIGIT2
                and params[1] == DIGIT0 and params[2] == DIGIT0)

    def find_end_marker(self, from_idx: Int) -> Int:
        # Locate ESC [ 2 0 1 ~ in buf at or after `from_idx`, else -1.
        var n = len(self.buf)
        if n < END_MARKER_LEN:
            return -1
        var i = from_idx
        if i < 0:
            i = 0
        var limit = n - END_MARKER_LEN
        while i <= limit:
            if (self.buf[i] == ESC and self.buf[i + 1] == CSI_OPEN
                    and self.buf[i + 2] == DIGIT2 and self.buf[i + 3] == DIGIT0
                    and self.buf[i + 4] == DIGIT1 and self.buf[i + 5] == TILDE):
                return i
            i += 1
        return -1

    def read_paste(mut self, mut line: List[Byte], mut view: List[Byte], mut units: List[EditUnit]):
        # buf[self.pos] is the first pasted byte. Drain raw chunks back-to-back
        # until the end marker is in the buffer, scanning only the fresh region
        # (with overlap so a split marker is still caught), then parse it.
        var start = self.pos
        var scan = start
        while True:
            var e = self.find_end_marker(scan)
            if e >= 0:
                self.commit_paste_range(start, e, line, view, units)
                self.pos = e + END_MARKER_LEN
                return
            if self.eof:
                self.commit_paste_range(start, len(self.buf), line, view, units)
                self.pos = len(self.buf)
                return
            scan = len(self.buf) - (END_MARKER_LEN - 1)
            if scan < start:
                scan = start
            _ = self.drain_chunk()

    def commit_paste_range(mut self, start: Int, end: Int, mut line: List[Byte], mut view: List[Byte], mut units: List[EditUnit]):
        # Strip CRs and gather the pasted bytes, counting lines and glyphs so we
        # can choose how to surface them.
        var clean = List[Byte]()
        var newlines = 0
        var glyphs = 0
        for i in range(start, end):
            var b = self.buf[i]
            if b == CR:
                continue
            clean.append(b)
            if b == LF:
                newlines += 1
            elif b < UTF8_CONT or b >= UTF8_LEAD2:
                glyphs += 1
        if len(clean) == 0:
            return
        # A small single-line paste reads best inserted literally, as if typed.
        # Larger or multi-line pastes collapse to a compact placeholder so they
        # neither flood the input row nor wrap raw newlines through the editor.
        if newlines == 0 and glyphs <= PASTE_INLINE_LIMIT:
            self.commit_literal(clean, line, view, units)
            return
        var nlines = newlines
        if clean[len(clean) - 1] != LF:
            nlines += 1
        for i in range(len(clean)):
            line.append(clean[i])
        var label: String
        if nlines == 1:
            label = String("[Paste: 1 line]")
        else:
            label = "[Paste: " + String(nlines) + " lines]"
        var marked = PLACEHOLDER_OPEN + label + PLACEHOLDER_CLOSE
        var mb = marked.as_bytes()
        for i in range(len(mb)):
            view.append(mb[i])
        units.append(EditUnit(len(clean), len(mb), label.byte_length()))

    def commit_literal(self, read clean: List[Byte], mut line: List[Byte], mut view: List[Byte], mut units: List[EditUnit]):
        # Re-emit pasted bytes as ordinary glyph units so each character edits
        # and erases exactly like typed input.
        var n = len(clean)
        var i = 0
        while i < n:
            var lead = clean[i]
            var width = 1
            if lead >= UTF8_LEAD4:
                width = 4
            elif lead >= UTF8_LEAD3:
                width = 3
            elif lead >= UTF8_LEAD2:
                width = 2
            line.append(lead)
            view.append(lead)
            i += 1
            var count = 1
            while count < width and i < n:
                var nb = clean[i]
                if nb < UTF8_CONT or nb >= UTF8_LEAD2:
                    break
                line.append(nb)
                view.append(nb)
                i += 1
                count += 1
            units.append(EditUnit(count, count, 1))
