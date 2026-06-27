from std.utils.variant import Variant


@fieldwise_init
struct NoGamma(Copyable, Movable, ImplicitlyCopyable):
    pass


@fieldwise_init
struct SplitGamma(Copyable, Movable, ImplicitlyCopyable):
    var name: StaticString
    var offset: Float32

    @implicit
    def __init__(out self, name: StaticString):
        self = Self(name, 0.0)


@fieldwise_init
struct AbsorbedGamma(Copyable, Movable, ImplicitlyCopyable):
    var name: StaticString
    var offset: Float32

    @implicit
    def __init__(out self, name: StaticString):
        self = Self(name, 0.0)


comptime GammaMode = Variant[NoGamma, SplitGamma, AbsorbedGamma]


@fieldwise_init
struct SingleSided(Copyable, Movable, ImplicitlyCopyable):
    pass


@fieldwise_init
struct TwoSided(Copyable, Movable, ImplicitlyCopyable):
    var m_block: Int


comptime RotationMode = Variant[SingleSided, TwoSided]


@fieldwise_init
struct NoColsum(Copyable, Movable, ImplicitlyCopyable):
    pass


@fieldwise_init
struct PerRowCs(Copyable, Movable, ImplicitlyCopyable):
    pass


@fieldwise_init
struct PerBlockCs(Copyable, Movable, ImplicitlyCopyable):
    pass


comptime ColsumMode = Variant[NoColsum, PerRowCs, PerBlockCs]
comptime PerBlockColsumMode = Variant[NoColsum, PerBlockCs]


@fieldwise_init
struct VnniPacked(Copyable, Movable, ImplicitlyCopyable):
    pass


@fieldwise_init
struct RowMajor(Copyable, Movable, ImplicitlyCopyable):
    pass


comptime PackMode = Variant[VnniPacked, RowMajor]


@fieldwise_init
struct PerRowQuant(Copyable, Movable, ImplicitlyCopyable):
    var fwht_block: Int
    var gamma: GammaMode
    var rotation: RotationMode
    var colsum: ColsumMode
    var pack: PackMode


@fieldwise_init
struct PerBlockQuant(Copyable, Movable, ImplicitlyCopyable):
    var fwht_block: Int
    var gamma: GammaMode
    var rotation: RotationMode
    var colsum: PerBlockColsumMode
    var pack: PackMode


@fieldwise_init
struct RouterCenter(Copyable, Movable, ImplicitlyCopyable):
    """§13.2 centered-bf16 router. `bias_name=""` skips the bias output
    (algebraically valid; §13.2 minus the additive bias term). Non-empty
    `bias_name` reads the named source tensor and writes it as f32 per §13.4."""
    var bias_name: StaticString


@fieldwise_init
struct SoftmaxRouterCenter(Copyable, Movable, ImplicitlyCopyable):
    """Centered-bf16 router for shift-invariant softmax/top-k routers. The
    gauge is intentionally not stored because subtracting one per-token scalar
    from every expert logit leaves softmax probabilities and top-k unchanged."""
    pass


@fieldwise_init
struct Passthrough(Copyable, Movable, ImplicitlyCopyable):
    pass


@fieldwise_init
struct NormGain(Copyable, Movable, ImplicitlyCopyable):
    var offset: Float32


comptime QuantRecipe = Variant[
    Passthrough, NormGain, PerRowQuant, PerBlockQuant, RouterCenter,
    SoftmaxRouterCenter,
]
