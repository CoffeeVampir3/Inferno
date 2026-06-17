from std.reflection import reflect


@fieldwise_init
struct CmdResult(Equatable, ImplicitlyCopyable, Copyable, Movable):
    var code: Int

    comptime HANDLED = CmdResult(0)
    comptime QUIT = CmdResult(1)
    comptime PASS = CmdResult(2)

    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        return self.code != other.code


trait ArgLike(Copyable, Movable, Defaultable, ImplicitlyDeletable):
    @staticmethod
    def parse(token: String) -> Optional[Self]: ...

    @staticmethod
    def on_absent() -> Optional[Self]: ...

    @staticmethod
    def usage() -> String: ...


trait Parsable(Copyable, Movable, Defaultable, ImplicitlyDeletable):
    pass


def fill[C: Parsable](read parts: List[String], start: Int = 1) -> Optional[C]:
    var cmd = C()
    comptime r = reflect[C]
    comptime types = r.field_types()
    var pos = start
    var ok = True
    comptime for i in range(r.field_count()):
        comptime AT = types[i]
        comptime if conforms_to(AT, ArgLike):
            if ok and pos < len(parts):
                var got = AT.parse(String(parts[pos]))
                pos += 1
                if got:
                    r.field_ref[i](cmd) = got.value().copy()
                else:
                    ok = False
            elif ok:
                var blank = AT.on_absent()
                if blank:
                    r.field_ref[i](cmd) = blank.value().copy()
                else:
                    print(t"  missing argument {AT.usage()}")
                    ok = False
    if not ok:
        return None
    return cmd^


struct PositiveFloat(ArgLike, Copyable, Movable):
    var value: Float32

    def __init__(out self):
        self.value = 0.0

    def __init__(out self, value: Float32):
        self.value = value

    @staticmethod
    def parse(token: String) -> Optional[Self]:
        try:
            var v = Float32(atof(token))
            if v > 0.0:
                return Self(v)
        except:
            pass
        print(t"  '{token}' must be a number > 0")
        return None

    @staticmethod
    def on_absent() -> Optional[Self]:
        return None

    @staticmethod
    def usage() -> String:
        return String("<float > 0>")


struct UnitFloat(ArgLike, Copyable, Movable):
    var value: Float32

    def __init__(out self):
        self.value = 0.0

    def __init__(out self, value: Float32):
        self.value = value

    @staticmethod
    def parse(token: String) -> Optional[Self]:
        try:
            var v = Float32(atof(token))
            if v >= 0.0 and v < 1.0:
                return Self(v)
        except:
            pass
        print(t"  '{token}' must be in [0, 1)")
        return None

    @staticmethod
    def on_absent() -> Optional[Self]:
        return None

    @staticmethod
    def usage() -> String:
        return String("<float in [0, 1)>")


struct BoundedInt[hi: Int](ArgLike, Copyable, Movable):
    var value: Int

    def __init__(out self):
        self.value = 0

    def __init__(out self, value: Int):
        self.value = value

    @staticmethod
    def parse(token: String) -> Optional[Self]:
        try:
            var v = atol(token)
            if v >= 0 and v <= Self.hi:
                return Self(v)
        except:
            pass
        print(t"  '{token}' must be an int in [0, {Self.hi}]")
        return None

    @staticmethod
    def on_absent() -> Optional[Self]:
        return None

    @staticmethod
    def usage() -> String:
        return String(t"<int 0..{Self.hi}>")


struct NonNegInt(ArgLike, Copyable, Movable):
    var value: Int

    def __init__(out self):
        self.value = 0

    def __init__(out self, value: Int):
        self.value = value

    @staticmethod
    def parse(token: String) -> Optional[Self]:
        try:
            var v = atol(token)
            if v >= 0:
                return Self(v)
        except:
            pass
        print(t"  '{token}' must be a non-negative integer")
        return None

    @staticmethod
    def on_absent() -> Optional[Self]:
        return None

    @staticmethod
    def usage() -> String:
        return String("<non-negative int>")


struct Toggle(ArgLike, Copyable, Movable):
    var mode: Int

    def __init__(out self):
        self.mode = 2

    def __init__(out self, mode: Int):
        self.mode = mode

    @staticmethod
    def parse(token: String) -> Optional[Self]:
        if token == "on":
            return Self(1)
        if token == "off":
            return Self(0)
        print(t"  '{token}' is not on or off")
        return None

    @staticmethod
    def on_absent() -> Optional[Self]:
        return Self(2)

    @staticmethod
    def usage() -> String:
        return String("[on|off]")

    def resolve(self, current: Bool) -> Bool:
        if self.mode == 1:
            return True
        if self.mode == 0:
            return False
        return not current
