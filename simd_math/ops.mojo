from std.sys import llvm_intrinsic


@always_inline
def sqrt[dtype: DType, width: Int](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """SIMD sqrt — lowers to vsqrtps (f32) or vsqrtpd (f64)."""
    return llvm_intrinsic[
        "llvm.sqrt",
        SIMD[dtype, width],
        SIMD[dtype, width],
    ](x)


@always_inline
def roundeven[dtype: DType, width: Int](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Round to nearest even — lowers to vroundps/vrndscaleps (f32) or
    vroundpd/vrndscalepd (f64)."""
    return llvm_intrinsic[
        "llvm.nearbyint",
        SIMD[dtype, width],
        SIMD[dtype, width],
    ](x)


@always_inline
def quantize_i8[width: Int](
    v: SIMD[DType.float32, width], inv_scale: SIMD[DType.float32, width],
) -> SIMD[DType.int8, width]:
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    return min(max(roundeven(v * inv_scale), lo), hi).cast[DType.int8]()


@always_inline
def exp_f32[width: Int](
    x: SIMD[DType.float32, width],
) -> SIMD[DType.float32, width]:
    comptime LN2_HI = Float32(0.693145751953125)
    comptime LN2_LO = Float32(1.4286068203094172e-06)
    comptime INV_LN2 = Float32(1.4426950408889634)
    comptime EXP_LO = Float32(-87.0)
    comptime EXP_HI = Float32(88.0)

    var lo_mask = ((x - EXP_LO).to_bits() >> 31) & 1
    var xc = x * (1 - lo_mask.cast[DType.float32]()) + (
        SIMD[DType.float32, width](EXP_LO) * lo_mask.cast[DType.float32]())
    var hi_mask = ((EXP_HI - xc).to_bits() >> 31) & 1
    xc = xc * (1 - hi_mask.cast[DType.float32]()) + (
        SIMD[DType.float32, width](EXP_HI) * hi_mask.cast[DType.float32]())

    var xn = xc * INV_LN2
    var sign = (xn.to_bits() >> 31).cast[DType.float32]()
    var n = (xn + 0.5 - sign).cast[DType.int32]()

    var nf = n.cast[DType.float32]()
    var r = (xc - nf * LN2_HI) - nf * LN2_LO

    var p = SIMD[DType.float32, width](1.0) + r * (
        Float32(0.9999999995) + r * (
        Float32(0.5000000004) + r * (
        Float32(0.1666666456) + r * (
        Float32(0.04166685110) + r * (
        Float32(0.008333621758) + r * (
        Float32(0.001389404636)))))))

    var pow2n = SIMD[DType.float32, width](
        from_bits=(n + 127).cast[DType.uint32]() << 23
    )
    return p * pow2n


@always_inline
def exp_f32_fast[width: Int](
    x: SIMD[DType.float32, width],
) -> SIMD[DType.float32, width]:
    # Fast exp(x) for x <= 0. ~0.3% relative error, ~8 SIMD ops vs ~23 for
    # exp_f32. Critically, clamps the input lower bound at -87 so the i32
    # range reduction can't overflow.
    comptime LN2 = Float32(0.6931471805599453)
    comptime INV_LN2 = Float32(1.4426950408889634)

    var xc = max(x, SIMD[DType.float32, width](-87.0))
    var xn = xc * INV_LN2
    var n = roundeven(xn).cast[DType.int32]()
    var r = xc - n.cast[DType.float32]() * LN2

    var p = SIMD[DType.float32, width](1.0) + r * (
        Float32(0.9999) + r * (
        Float32(0.4985) + r * (
        Float32(0.1681))))

    var n_clamped = max(n, SIMD[DType.int32, width](-126))
    var pow2n = SIMD[DType.float32, width](
        from_bits=(n_clamped + 127).cast[DType.uint32]() << 23
    )
    return p * pow2n


@always_inline
def log_f32[width: Int](
    x: SIMD[DType.float32, width],
) -> SIMD[DType.float32, width]:
    comptime LN2 = Float32(0.6931471805599453)
    comptime SQRT2 = Float32(1.4142135623730951)

    var bits = x.to_bits()
    var e = (bits >> 23).cast[DType.int32]() - 127
    var m_bits = (bits & 0x007FFFFF) | 0x3F800000
    var m = SIMD[DType.float32, width](from_bits=m_bits)

    var big = (SIMD[DType.float32, width](SQRT2) - m).to_bits() >> 31
    var big_f = big.cast[DType.float32]()
    m = m * (1.0 - 0.5 * big_f)
    e = e + big.cast[DType.int32]()

    var z = (m - 1.0) / (m + 1.0)
    var z2 = z * z
    var t = SIMD[DType.float32, width](Float32(1.0 / 11.0))
    t = Float32(1.0 / 9.0) + z2 * t
    t = Float32(1.0 / 7.0) + z2 * t
    t = Float32(1.0 / 5.0) + z2 * t
    t = Float32(1.0 / 3.0) + z2 * t
    t = Float32(1.0) + z2 * t
    return 2.0 * z * t + e.cast[DType.float32]() * LN2


@always_inline
def tanh_f32[width: Int](
    x: SIMD[DType.float32, width],
) -> SIMD[DType.float32, width]:
    # tanh(x) = (1 - exp(-2|x|)) / (1 + exp(-2|x|)) * sign(x)
    # Computed on |x| keeps the exp argument <= 0 so exp_f32_fast's clamp
    # never engages with relevant precision loss; sign restored via xor.
    var sign_bits = x.to_bits() & 0x80000000
    var ax = SIMD[DType.float32, width](
        from_bits=x.to_bits() & 0x7FFFFFFF)
    var e = exp_f32_fast(SIMD[DType.float32, width](-2.0) * ax)
    var t = (SIMD[DType.float32, width](1.0) - e) / (
        SIMD[DType.float32, width](1.0) + e)
    return SIMD[DType.float32, width](
        from_bits=t.to_bits() ^ sign_bits)


@always_inline
def gelu_tanh_f32[width: Int](
    x: SIMD[DType.float32, width],
) -> SIMD[DType.float32, width]:
    # gelu_pytorch_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Uses tanh_f32 so large |x| no longer overflows the exp range.
    comptime sqrt_2_over_pi = SIMD[DType.float32, width](0.7978845608028654)
    comptime gelu_a = SIMD[DType.float32, width](0.044715)
    var x3 = x * x * x
    var inner = sqrt_2_over_pi * (x + gelu_a * x3)
    var t = tanh_f32(inner)
    return SIMD[DType.float32, width](0.5) * x * (
        SIMD[DType.float32, width](1.0) + t)


@always_inline
def softcap_value[
    cap: Float64,
](
    x: SIMD[DType.float32, 1],
) -> SIMD[DType.float32, 1]:
    comptime assert cap > 0.0, "softcap cap must be positive"
    comptime c = SIMD[DType.float32, 1](cap)
    return tanh_f32[1](x / c) * c
