def sollya_remez_exp_coeffs_unbiased() -> InlineArray[InlineArray[Float32, 7], 8]:
    var t = InlineArray[InlineArray[Float32, 7], 8](
        fill=InlineArray[Float32, 7](fill=Float32(0.0))
    )

    t[2][0] = Float32(1.0184489029)
    t[2][1] = Float32(0.5138389751)

    t[3][0] = Float32(1.0037793568)
    t[3][1] = Float32(0.50375479995)
    t[3][2] = Float32(0.12521220002)

    t[4][0] = Float32(0.99995155985)
    t[4][1] = Float32(0.49995962941)
    t[4][2] = Float32(0.16808630622)
    t[4][3] = Float32(0.042380524549)

    t[5][0] = Float32(0.99999372435)
    t[5][1] = Float32(0.49999375495)
    t[5][2] = Float32(0.16687654923)
    t[5][3] = Float32(0.041875271933)
    t[5][4] = Float32(0.0069533250692)

    t[6][0] = Float32(1.0000000496)
    t[6][1] = Float32(0.50000004337)
    t[6][2] = Float32(0.16666365277)
    t[6][3] = Float32(0.041664775388)
    t[6][4] = Float32(0.0083790397226)
    t[6][5] = Float32(0.001406179907)

    t[7][0] = Float32(1.0000000047)
    t[7][1] = Float32(0.50000000468)
    t[7][2] = Float32(0.16666635164)
    t[7][3] = Float32(0.04166635394)
    t[7][4] = Float32(0.0083385787937)
    t[7][5] = Float32(0.0013941038937)
    t[7][6] = Float32(0.00017378912648)

    return t^


comptime EXP_MINIMAX_UNBIASED = sollya_remez_exp_coeffs_unbiased()


def sollya_never_overshoot_exp_coeffs_biased() -> InlineArray[InlineArray[Float32, 7], 8]:
    var t = InlineArray[InlineArray[Float32, 7], 8](
        fill=InlineArray[Float32, 7](fill=Float32(0.0))
    )

    t[2][0] = Float32(1.0000023001)
    t[2][1] = Float32(0.4469212768)

    t[3][0] = Float32(1.0000000956)
    t[3][1] = Float32(0.49992886129)
    t[3][2] = Float32(0.16276842496)

    t[4][0] = Float32(0.99999986547)
    t[4][1] = Float32(0.49989399303)
    t[4][2] = Float32(0.16745989708)
    t[4][3] = Float32(0.042105568259)

    t[5][0] = Float32(0.99999972076)
    t[5][1] = Float32(0.49998828852)
    t[5][2] = Float32(0.16667474464)
    t[5][3] = Float32(0.041923133233)
    t[5][4] = Float32(0.0083014020581)

    t[6][0] = Float32(0.99999995293)
    t[6][1] = Float32(0.50000068244)
    t[6][2] = Float32(0.16666523167)
    t[6][3] = Float32(0.041653788694)
    t[6][4] = Float32(0.0083724147225)
    t[6][5] = Float32(0.0014517845378)

    t[7][0] = Float32(1.0000000642)
    t[7][1] = Float32(0.50000011611)
    t[7][2] = Float32(0.1666634715)
    t[7][3] = Float32(0.041664080495)
    t[7][4] = Float32(0.0083792499598)
    t[7][5] = Float32(0.0014050422425)
    t[7][6] = Float32(0.0)

    return t^


comptime EXP_MINIMAX_BIASED = sollya_never_overshoot_exp_coeffs_biased()


# Error profile — max relative error, sweep x ∈ [-10, 10], default flags.
# Expect similar relative ULP profile with undershoot but with bias.
# Reproduce / regression-asserted in tests/test_exp.mojo.
#   polynomial_degree=2: rel ≈ 2.3e-3   (~0.30 bf16 ULP / ~2.4 f16 ULPs)
#                        loss-free for bf16; visible at f16.
#   polynomial_degree=3: rel ≈ 4.3e-4   (~0.44 f16 ULP)
#                        loss-free for f16.
#   polynomial_degree=4: rel ≈ 3.6e-6   (~30 f32 ULPs)
#                        sub-f16; not yet at f32 ULP.
#   polynomial_degree=5: rel ≈ 5.1e-7   (~4.3 f32 ULPs)
#                        approaching f32 ULP.
#   polynomial_degree=6: rel ≈ 1.0e-7   (~0.9 f32 ULP)
#                        loss-free for f32 — recommended for f32 output.
#   polynomial_degree=7: rel ≈ 8.6e-8   (~0.7 f32 ULP)
#                        f32 ULP-saturated; no benefit over N=6.
@always_inline
def exp_simd[
    polynomial_degree: Int,
    width: Int,
    *,
    ieee_corrections: Bool = False,
    never_overshoot: Bool = False,
](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    comptime assert polynomial_degree >= 2 and polynomial_degree <= 7, (
        "exp_simd polynomial_degree must be in [2, 7]"
    )

    comptime LN2 = Float32(0.6931471805599453)
    comptime LN2_HI = Float32(0.693145751953125)
    comptime LN2_LO = Float32(1.4286068203094172e-06)
    comptime INV_LN2 = Float32(1.4426950408889634)
    comptime EXP_LO = Float32(-103.972) if ieee_corrections else Float32(-87.0)
    comptime EXP_HI = Float32(88.7228)

    var xc = min(
        SIMD[DType.float32, width](EXP_HI),
        max(SIMD[DType.float32, width](EXP_LO), x),
    )

    var xn = xc * INV_LN2
    var sign = (xn.to_bits() >> 31).cast[DType.float32]()
    var n = (xn + 0.5 - sign).cast[DType.int32]()
    var nf = n.cast[DType.float32]()

    var r = xc - nf * LN2
    comptime if ieee_corrections:
        r = (xc - nf * LN2_HI) - nf * LN2_LO

    ref coeffs = (
        EXP_MINIMAX_BIASED[polynomial_degree] if never_overshoot
        else EXP_MINIMAX_UNBIASED[polynomial_degree]
    )

    var p = SIMD[DType.float32, width](coeffs[polynomial_degree - 1])
    comptime for k_off in range(polynomial_degree - 1):
        comptime k = polynomial_degree - 2 - k_off
        p = SIMD[DType.float32, width](coeffs[k]) + r * p
    p = SIMD[DType.float32, width](1.0) + r * p

    var n_pow = n
    comptime if ieee_corrections:
        n_pow = max(n_pow, SIMD[DType.int32, width](-127))

    var pow2n = SIMD[DType.float32, width](
        from_bits=(n_pow + 127).cast[DType.uint32]() << 23
    )

    return p * pow2n


# Fast biased Schraudolph-style exp for approximate bf16 softmax/attention.
#
# This is intentionally not a general exp replacement. It is a raw no-clamp path
# for post-shift softmax inputs where x <= 0 and the caller wants a biased
# undershooting exp. The coefficients were fitted in study/fit_schraudolph_one_sided.py
# by a constrained linear program over dense f32 grid samples in [-30, 0] with
# the constraint approx(x) <= exp(x).
#
# Error profile on that fitted range:
#   max relative error ≈ -6.88e-3
#   max rounded-bf16 distance = 2 ULPs
#   ≈95.9% within 1 rounded-bf16 ULP
#
# Raw finite bit-path range for these constants is approximately [-87.5219, 0].
# Inputs below that can produce NaN/negative nonsense from the bitcast. In
# softmax those lanes are far below f32 relevance after max subtraction, so the
# caller should mask/drop/zero them before using this function rather than paying
# a per-lane clamp inside the hot exp path.
@always_inline
def fast_exp_softmax_biased[width: Int](
    x: SIMD[DType.float32, width],
) -> SIMD[DType.float32, width]:
    comptime A_MAGIC = Float32(12102203.16156148)  # 2^23 / ln 2
    comptime BIAS_F = Float32(1059208216.0)        # 127*2^23 - 6_145_000
    comptime INV_TWO23 = Float32(1.0) / Float32(1 << 23)

    comptime QC_A = Float32(1.6501418352127075)
    comptime QC_B = Float32(-0.37554836273193359)
    comptime QC_C = Float32(0.38696467876434326)

    var i = (A_MAGIC * x + BIAS_F).cast[DType.int32]()
    var u = i.cast[DType.uint32]()
    var k = SIMD[DType.float32, width](from_bits=u)
    var fbits = u & SIMD[DType.uint32, width](0x7FFFFF)
    var f = fbits.cast[DType.float32]() * INV_TWO23
    return k * (QC_A + f * (QC_B + f * QC_C))
