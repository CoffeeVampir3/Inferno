from std.utils.variant import Variant


@fieldwise_init
struct SlotIdentity(Copyable, Movable):
    """`name` is the full safetensors key (e.g.
    `model.language_model.layers.5.self_attn.q_proj.weight`). `local_name`
    is the slot's own annotation (e.g. `self_attn.q_proj.weight`), useful
    when logging without the layer prefix noise. `layer_idx` is -1 for
    layer-agnostic slots (e.g. `model.language_model.norm.weight`)."""
    var name: String
    var local_name: String
    var layer_idx: Int
    var shard: Int
    var src_offset: Int
    var src_dtype: DType
    var rows: Int
    var cols: Int
    var weight_off: Int


@fieldwise_init
struct GammaRef(Copyable, Movable):
    """Empty `name` signals no gamma; `absorbed` selects between the
    sqrt-abs split factor and the raw absorbed gain."""
    var name: String
    var absorbed: Bool
    var shard: Int
    var src_offset: Int
    var byte_size: Int
    var src_dtype: DType
    var cols: Int

    @staticmethod
    def none() -> Self:
        return Self(String(""), False, 0, 0, 0, DType.float32, 0)

    @staticmethod
    def named(name: String, absorbed: Bool) -> Self:
        return Self(name, absorbed, 0, 0, 0, DType.float32, 0)

    def is_present(self) -> Bool:
        return self.name.byte_length() > 0

    @always_inline
    def locate(
        mut self, shard: Int, src_offset: Int, byte_size: Int,
        src_dtype: DType, cols: Int,
    ):
        self.shard = shard
        self.src_offset = src_offset
        self.byte_size = byte_size
        self.src_dtype = src_dtype
        self.cols = cols


@fieldwise_init
struct PassthroughPlan(Copyable, Movable):
    var id: SlotIdentity
    var byte_count: Int


@fieldwise_init
struct QuantPlan(Copyable, Movable):
    var id: SlotIdentity
    var per_block: Bool
    var fwht_block: Int
    var two_sided_m: Int
    var gamma: GammaRef
    var scale_off: Int


@fieldwise_init
struct RouterPlan(Copyable, Movable):
    """Empty `bias_name` skips bias emission (§13.2 minus the bias term);
    a non-empty `bias_name` reads the source tensor and writes it as f32
    per §13.4."""
    var id: SlotIdentity
    var gauge_off: Int
    var emit_gauge: Bool
    var bias_name: String
    var bias_off: Int
    var bias_shard: Int
    var bias_src_offset: Int
    var bias_byte_size: Int
    var bias_src_dtype: DType


comptime SlotPlan = Variant[PassthroughPlan, QuantPlan, RouterPlan]


@fieldwise_init
struct ScratchCapacity(TrivialRegisterPassable):
    """Worst-case staging sizes computed during plan walk.

    The quantizer turns this into a rank-local arena layout. Counts are typed
    element counts except `raw_bytes`, which is the largest byte-oriented read
    staging region needed by any planned operation.
    """
    var max_panel_rows: Int
    var raw_bytes: Int
    var f32_work: Int
    var i8_quant: Int
    var f32_scales: Int
    var f32_gamma: Int
    var bf16_centered: Int
    var bf16_gauge: Int

    @staticmethod
    def zero(max_panel_rows: Int) -> Self:
        return Self(max_panel_rows, 0, 0, 0, 0, 0, 0, 0)

    @always_inline
    def absorb_raw(mut self, bytes: Int):
        if bytes > self.raw_bytes:
            self.raw_bytes = bytes

    @always_inline
    def absorb_f32_work(mut self, count: Int):
        if count > self.f32_work:
            self.f32_work = count

    @always_inline
    def absorb_i8_quant(mut self, count: Int):
        if count > self.i8_quant:
            self.i8_quant = count

    @always_inline
    def absorb_f32_scales(mut self, count: Int):
        if count > self.f32_scales:
            self.f32_scales = count

    @always_inline
    def absorb_f32_gamma(mut self, count: Int):
        if count > self.f32_gamma:
            self.f32_gamma = count

    @always_inline
    def absorb_bf16_centered(mut self, count: Int):
        if count > self.bf16_centered:
            self.bf16_centered = count

    @always_inline
    def absorb_bf16_gauge(mut self, count: Int):
        if count > self.bf16_gauge:
            self.bf16_gauge = count

    @always_inline
    def absorb_quant(mut self, p: QuantPlan, src_bytes_per: Int):
        var panel_rows = self.max_panel_rows
        if p.id.rows < panel_rows:
            panel_rows = p.id.rows
        var panel_elems = panel_rows * p.id.cols
        self.absorb_raw(panel_elems * src_bytes_per)
        self.absorb_f32_work(panel_elems)
        self.absorb_i8_quant(panel_elems)
        var spr = (p.id.cols // p.fwht_block) if p.per_block else 1
        self.absorb_f32_scales(panel_rows * spr)
        if p.gamma.is_present():
            self.absorb_raw(p.id.cols * 4)
            self.absorb_f32_gamma(p.id.cols)
