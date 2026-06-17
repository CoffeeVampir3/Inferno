def sollya_remez_log_coeffs_unbiased() -> InlineArray[InlineArray[Float32, 7], 8]:
    var t = InlineArray[InlineArray[Float32, 7], 8](
        fill=InlineArray[Float32, 7](fill=Float32(0.0))
    )

    t[2][0] = Float32(0.33331731188)
    t[2][1] = Float32(0.20430398344)

    t[3][0] = Float32(0.33333342633)
    t[3][1] = Float32(0.19994359436)
    t[3][2] = Float32(0.14791023431)

    t[4][0] = Float32(0.33333333277)
    t[4][1] = Float32(0.20000061358)
    t[4][2] = Float32(0.14275370677)
    t[4][3] = Float32(0.11666133894)

    t[5][0] = Float32(0.33333333334)
    t[5][1] = Float32(0.19999999394)
    t[5][2] = Float32(0.14285878125)
    t[5][3] = Float32(0.11095653825)
    t[5][4] = Float32(0.096821185285)

    t[6][0] = Float32(0.33333333333)
    t[6][1] = Float32(0.20000000006)
    t[6][2] = Float32(0.14285712055)
    t[6][3] = Float32(0.11111432483)
    t[6][4] = Float32(0.090700447554)
    t[6][5] = Float32(0.083116173892)

    t[7][0] = Float32(0.33333333333)
    t[7][1] = Float32(0.2)
    t[7][2] = Float32(0.14285714313)
    t[7][3] = Float32(0.11111105539)
    t[7][4] = Float32(0.090914464436)
    t[7][5] = Float32(0.076658039819)
    t[7][6] = Float32(0.073088684097)

    return t^


comptime LOG_MINIMAX_UNBIASED = sollya_remez_log_coeffs_unbiased()


# Error profile — max relative error, sweep x ∈ [0.1, 10] log-uniform, default flags.
# Reproduce / regression-asserted in tests/test_log.mojo.
#
# log_simd's polynomial converges very fast on the centered domain
# z² ∈ [0, 0.0294], so the error floor is set by f32 evaluation noise,
# NOT by polynomial truncation. Past N=3 there is no measurable gain.
#
#   polynomial_degree=2: rel ≈ 6.1e-7   (~5.1 f32 ULPs / ~6e-4 f16 ULPs)
#                        loss-free for f16/bf16; small polynomial residual at f32.
#   polynomial_degree=3: rel ≈ 2.1e-7   (~1.8 f32 ULPs)
#                        f32-evaluation-noise saturated — recommended default.
#   polynomial_degree≥4: rel ≈ 2.1e-7   (no improvement; same as N=3).
#                        Higher degrees are computationally wasted.
@always_inline
def log_simd[
    polynomial_degree: Int,
    width: Int,
    *,
    ieee_corrections: Bool = False,
](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    comptime assert polynomial_degree >= 2 and polynomial_degree <= 7, (
        "log_simd polynomial_degree must be in [2, 7]"
    )

    comptime LN2 = Float32(0.6931471805599453)
    comptime SQRT2 = Float32(1.4142135623730951)
    comptime FLT_MIN_NORMAL = Float32(1.1754944e-38)
    comptime FLT_MAX = Float32(3.4028235e38)

    var xc = x
    comptime if ieee_corrections:
        xc = min(
            max(x, SIMD[DType.float32, width](FLT_MIN_NORMAL)),
            SIMD[DType.float32, width](FLT_MAX),
        )

    var bits = xc.to_bits()
    var e = (bits >> 23).cast[DType.int32]() - 127
    var m_bits = (bits & 0x007FFFFF) | 0x3F800000
    var m = SIMD[DType.float32, width](from_bits=m_bits)

    var big = (SIMD[DType.float32, width](SQRT2) - m).to_bits() >> 31
    var big_f = big.cast[DType.float32]()
    m = m * (1.0 - 0.5 * big_f)
    e = e + big.cast[DType.int32]()

    var z = (m - 1.0) / (m + 1.0)
    var z2 = z * z

    ref coeffs = LOG_MINIMAX_UNBIASED[polynomial_degree]

    var t = SIMD[DType.float32, width](coeffs[polynomial_degree - 1])
    comptime for k_off in range(polynomial_degree - 1):
        comptime k = polynomial_degree - 2 - k_off
        t = SIMD[DType.float32, width](coeffs[k]) + z2 * t
    t = SIMD[DType.float32, width](1.0) + z2 * t

    return 2.0 * z * t + e.cast[DType.float32]() * LN2
