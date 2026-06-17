from std.memory import Span
from std.os import makedirs
from std.os.path import dirname
from std.pathlib import Path

from threading.threading_traits import BurstThreadPool
from jsontools.parser import Parser, ParseError, LBRACE, RBRACE, LBRACKET, RBRACKET
from safetensors.parser import parse_safetensors_header, TensorMeta
from safetensors.writer import OutputEntry, write_safetensors
from modeling.gemma4_common import Gemma4BaseConfig
from inspectable_toolkit.steer import InjectOp, Steerable


comptime C = Gemma4BaseConfig


@fieldwise_init
struct SliderConfig(Copyable, Movable, ImplicitlyCopyable):
    var vec_idx: Int
    var layer: Int
    var alpha_min: Float32
    var alpha_max: Float32
    var bidirectional: Bool


@fieldwise_init
struct SliderCalibration(Copyable, Movable, ImplicitlyCopyable):
    var fisher_ratio: Float64
    var holdout_sigma: Float64
    var steer_norm: Float64
    var measure_layer: Int
    var measure_std: Float64


struct SliderBank(Movable):
    var configs: List[SliderConfig]
    var names: List[String]
    var alphas: List[Float32]

    def __init__(out self):
        self.configs = List[SliderConfig]()
        self.names = List[String]()
        self.alphas = List[Float32]()

    def register(mut self, read name: String, cfg: SliderConfig):
        self.configs.append(cfg)
        self.names.append(name)
        self.alphas.append(Float32(0))

    def count(self) -> Int:
        return len(self.configs)

    def index_of(self, read name: String) -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1

    def position_to_alpha(self, idx: Int, pos: Float32) -> Float32:
        var cfg = self.configs[idx]
        if pos == Float32(0):
            return Float32(0)
        if pos < Float32(0) and not cfg.bidirectional:
            return Float32(0)
        var mag = abs(pos)
        var a = cfg.alpha_min + mag * (cfg.alpha_max - cfg.alpha_min)
        return a if pos > Float32(0) else -a

    def set_position(mut self, idx: Int, pos: Float32):
        self.alphas[idx] = self.position_to_alpha(idx, pos)

    def set_position_by_name(mut self, read name: String, pos: Float32) -> Bool:
        var i = self.index_of(name)
        if i < 0:
            return False
        self.set_position(i, pos)
        return True

    def set_alpha(mut self, idx: Int, alpha: Float32):
        var cfg = self.configs[idx]
        var a = alpha
        if a > cfg.alpha_max:
            a = cfg.alpha_max
        var floor = -cfg.alpha_max if cfg.bidirectional else Float32(0)
        if a < floor:
            a = floor
        self.alphas[idx] = a

    def neutral(mut self):
        for i in range(len(self.alphas)):
            self.alphas[i] = Float32(0)

    def build_ops(self) -> List[InjectOp]:
        var ops = List[InjectOp]()
        for i in range(len(self.configs)):
            if self.alphas[i] != Float32(0):
                ops.append(InjectOp(
                    self.configs[i].layer, self.configs[i].vec_idx,
                    self.alphas[i]))
        return ops^

    def apply[M: Steerable, //](self, mut model: M):
        var ops = self.build_ops()
        if len(ops) == 0:
            model.disarm_steer()
        else:
            model.set_inject_ops(ops^)


def basename(read p: String) -> String:
    var bytes = p.as_bytes()
    var last = -1
    for i in range(len(bytes)):
        if bytes[i] == Byte(47):
            last = i
    if last < 0:
        return p
    return String(unsafe_from_utf8=Span(bytes).unsafe_subspan(
        offset=last + 1, length=len(bytes) - last - 1))


def append_bool(mut js: String, v: Bool):
    if v:
        js += "true"
    else:
        js += "false"


def build_manifest_json(
    read names: List[String], read configs: List[SliderConfig],
    read cals: List[SliderCalibration], hidden: Int, num_layers: Int,
    read arch: String, read weights: String,
) -> String:
    var n = len(configs)
    var js = String('{\n')
    js += '  "format": "gemma4-slider-pack",\n'
    js += '  "version": 1,\n'
    js += '  "model": { "arch": "'
    js += arch
    js += '", "hidden": '
    js += String(hidden)
    js += ', "num_layers": '
    js += String(num_layers)
    js += ' },\n'
    js += '  "weights": "'
    js += weights
    js += '",\n'
    js += '  "sliders": [\n'
    for i in range(n):
        js += '    { "trait": "'
        js += names[i]
        js += '", "tensor": "'
        js += names[i]
        js += '.vector", "steer_layer": '
        js += String(configs[i].layer)
        js += ', "alpha_min": '
        js += String(Float64(configs[i].alpha_min))
        js += ', "alpha_max": '
        js += String(Float64(configs[i].alpha_max))
        js += ', "bidirectional": '
        append_bool(js, configs[i].bidirectional)
        js += ', "calibration": { "fisher_ratio": '
        js += String(cals[i].fisher_ratio)
        js += ', "holdout_sigma": '
        js += String(cals[i].holdout_sigma)
        js += ', "steer_norm": '
        js += String(cals[i].steer_norm)
        js += ', "measure_layer": '
        js += String(cals[i].measure_layer)
        js += ', "measure_std": '
        js += String(cals[i].measure_std)
        js += ' } }'
        if i + 1 < n:
            js += ','
        js += '\n'
    js += '  ]\n}\n'
    return js^


def write_pack(
    read base_path: String, read names: List[String],
    read directions: List[BFloat16], read configs: List[SliderConfig],
    read cals: List[SliderCalibration], hidden: Int, num_layers: Int,
    read arch: String,
) -> Bool:
    var n = len(configs)
    if n == 0 or len(names) != n or len(cals) != n:
        print("write_pack: mismatched slider inputs")
        return False
    if len(directions) < n * hidden:
        print("write_pack: directions buffer too small")
        return False

    var dir_dir = dirname(base_path)
    if dir_dir.byte_length() > 0:
        try:
            makedirs(dir_dir, exist_ok=True)
        except e:
            print(t"write_pack: cannot create {dir_dir}: {e}")
            return False

    var entries = List[OutputEntry]()
    for i in range(n):
        var start = i * hidden * 2
        var end = (i + 1) * hidden * 2
        entries.append(OutputEntry(
            names[i] + ".vector", DType.bfloat16, hidden, 0, start, end))

    var total_bytes = n * hidden * 2
    var payload = List[UInt8](capacity=total_bytes)
    var byte_ptr = directions.unsafe_ptr().bitcast[UInt8]()
    for b in range(total_bytes):
        payload.append(byte_ptr[b])

    var weights = basename(base_path) + ".safetensors"
    if not write_safetensors(
            Path(base_path + ".safetensors"), entries, payload):
        return False

    var js = build_manifest_json(
        names, configs, cals, hidden, num_layers, arch, weights)
    try:
        with open(Path(base_path + ".json"), "w") as f:
            f.write_bytes(js.as_bytes())
        return True
    except e:
        print(t"write_pack: failed to write manifest: {e}")
        return False


@fieldwise_init
struct SliderEntry(Copyable, Movable):
    var trait_name: String
    var tensor: String
    var layer: Int
    var alpha_min: Float32
    var alpha_max: Float32
    var bidirectional: Bool


struct ManifestData(Movable):
    var hidden: Int
    var weights: String
    var sliders: List[SliderEntry]

    def __init__(out self):
        self.hidden = 0
        self.weights = String("")
        self.sliders = List[SliderEntry]()


def parse_model_block(mut p: Parser) raises ParseError -> Int:
    var hidden = 0
    if not p.consume(LBRACE):
        raise ParseError("expected '{' for model", p.pos)
    p.skip_whitespace()
    if p.consume(RBRACE):
        return hidden
    while True:
        var key = p.object_key()
        if key == "hidden":
            hidden = p.parse_uint()
        else:
            p.skip_value()
        if not p.delimited_next(RBRACE):
            break
    return hidden


def parse_one_slider(mut p: Parser) raises ParseError -> SliderEntry:
    var trait_name = String("")
    var tensor = String("")
    var layer = -1
    var amin = Float32(0)
    var amax = Float32(0)
    var bidir = False
    if not p.consume(LBRACE):
        raise ParseError("expected '{' for slider", p.pos)
    p.skip_whitespace()
    if p.consume(RBRACE):
        return SliderEntry(trait_name^, tensor^, layer, amin, amax, bidir)
    while True:
        var key = p.object_key()
        if key == "trait":
            trait_name = p.parse_string()
        elif key == "tensor":
            tensor = p.parse_string()
        elif key == "steer_layer":
            layer = p.parse_uint()
        elif key == "alpha_min":
            amin = Float32(p.parse_number())
        elif key == "alpha_max":
            amax = Float32(p.parse_number())
        elif key == "bidirectional":
            bidir = p.parse_bool()
        else:
            p.skip_value()
        if not p.delimited_next(RBRACE):
            break
    return SliderEntry(trait_name^, tensor^, layer, amin, amax, bidir)


def parse_sliders(mut p: Parser) raises ParseError -> List[SliderEntry]:
    var out = List[SliderEntry]()
    if not p.consume(LBRACKET):
        raise ParseError("expected '[' for sliders", p.pos)
    p.skip_whitespace()
    if p.consume(RBRACKET):
        return out^
    while True:
        out.append(parse_one_slider(p))
        if not p.delimited_next(RBRACKET):
            break
    return out^


def parse_manifest(mut p: Parser) raises ParseError -> ManifestData:
    var data = ManifestData()
    p.skip_whitespace()
    if not p.consume(LBRACE):
        raise ParseError("expected root object", p.pos)
    p.skip_whitespace()
    if p.consume(RBRACE):
        return data^
    while True:
        var key = p.object_key()
        if key == "model":
            data.hidden = parse_model_block(p)
        elif key == "weights":
            data.weights = p.parse_string()
        elif key == "sliders":
            data.sliders = parse_sliders(p)
        else:
            p.skip_value()
        if not p.delimited_next(RBRACE):
            break
    return data^


def read_vector(
    read path: Path, data_offset: Int, read meta: TensorMeta,
) -> Optional[List[BFloat16]]:
    var nbytes = meta.end - meta.start
    var raw: List[Byte]
    try:
        with open(path, "r") as f:
            _ = f.seek(UInt64(data_offset + meta.start), 0)
            raw = f.read_bytes(size=nbytes)
    except e:
        print(t"load_pack: failed reading vector: {e}")
        return None
    if len(raw) != nbytes:
        print("load_pack: short vector read")
        return None
    var n = nbytes // 2
    var vec = List[BFloat16](capacity=n)
    var p = raw.unsafe_ptr().bitcast[Scalar[DType.bfloat16]]()
    for i in range(n):
        vec.append(p[i])
    return vec^


def load_pack[
    M: Steerable, //,
](
    mut model: M, read json_path: String,
) -> Optional[SliderBank]:
    var data: List[Byte]
    try:
        data = Path(json_path).read_bytes()
    except e:
        print(t"load_pack: cannot read manifest {json_path}: {e}")
        return None

    var parser = Parser(Span(data))
    var manifest: ManifestData
    try:
        manifest = parse_manifest(parser)
    except e:
        print(t"load_pack: manifest parse error at {e.pos}: {e.message}")
        return None

    if manifest.hidden != C.HIDDEN:
        print(t"load_pack: hidden mismatch (pack {manifest.hidden} "
              t"!= model {C.HIDDEN})")
        return None
    if len(manifest.sliders) == 0:
        print("load_pack: manifest has no sliders")
        return None
    if len(manifest.sliders) > M.STEER_VECTORS:
        print(t"load_pack: too many sliders ({len(manifest.sliders)} > "
              t"{M.STEER_VECTORS})")
        return None

    var weights_path = Path(dirname(json_path) + "/" + manifest.weights)
    var hdr_opt = parse_safetensors_header(weights_path)
    if not hdr_opt:
        print("load_pack: failed to read weights header")
        return None
    var hdr = hdr_opt.take()

    var bank = SliderBank()
    for i in range(len(manifest.sliders)):
        ref s = manifest.sliders[i]
        if s.layer < 0 or s.layer >= C.NUM_LAYERS:
            print(t"load_pack: slider {s.trait_name} layer {s.layer} out of range")
            return None
        if s.alpha_min > s.alpha_max:
            print(t"load_pack: slider {s.trait_name} alpha_min > alpha_max")
            return None
        var meta_opt = hdr.tensors.get(s.tensor)
        if not meta_opt:
            print(t"load_pack: tensor {s.tensor} missing from weights")
            return None
        ref meta = meta_opt.value()
        if meta.dtype != DType.bfloat16:
            print(t"load_pack: tensor {s.tensor} not BF16")
            return None
        if len(meta.shape) != 1 or meta.shape[0] != C.HIDDEN:
            print(t"load_pack: tensor {s.tensor} shape mismatch")
            return None
        var vec_opt = read_vector(weights_path, hdr.data_offset, meta)
        if not vec_opt:
            return None
        model.set_steer_vector(i, vec_opt.value())
        bank.register(
            s.trait_name,
            SliderConfig(i, s.layer, s.alpha_min, s.alpha_max,
                         s.bidirectional))

    return bank^
