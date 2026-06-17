def sin_coeffs_unbiased() -> InlineArray[InlineArray[Float64, 11], 11]:
    var t = InlineArray[InlineArray[Float64, 11], 11](
        fill=InlineArray[Float64, 11](fill=Float64(0.0))
    )

    t[4][0] = Float64(0.99999924560571451515)
    t[4][1] = Float64(-0.16665682729492128922)
    t[4][2] = Float64(0.0083132587141652863894)
    t[4][3] = Float64(-0.00018524357347636210267)

    t[5][0] = Float64(0.99999999573280273779)
    t[5][1] = Float64(-0.16666657991914243797)
    t[5][2] = Float64(0.008333051063454987098)
    t[5][3] = Float64(-0.00019809075291951581932)
    t[5][4] = Float64(2.605224917267714213e-06)

    t[6][0] = Float64(0.99999999998301947191)
    t[6][1] = Float64(-0.16666666617018802143)
    t[6][2] = Float64(0.0083333309807957830356)
    t[6][3] = Float64(-0.00019840861926843925069)
    t[6][4] = Float64(2.7525304432758222077e-06)
    t[6][5] = Float64(-2.388977897426537208e-08)

    t[7][0] = Float64(0.99999999999994992894)
    t[7][1] = Float64(-0.16666666666467339075)
    t[7][2] = Float64(0.0083333333203898449226)
    t[7][3] = Float64(-0.00019841266688576451697)
    t[7][4] = Float64(2.7556953338474975951e-06)
    t[7][5] = Float64(-2.5030283520756438983e-08)
    t[7][6] = Float64(1.5411426861689931305e-10)

    t[8][0] = Float64(0.99999999999999988898)
    t[8][1] = Float64(-0.16666666666666074548)
    t[8][2] = Float64(0.0083333333332828614382)
    t[8][3] = Float64(-0.00019841269824887760556)
    t[8][4] = Float64(2.7557316612068370916e-06)
    t[8][5] = Float64(-2.505188212471233718e-08)
    t[8][6] = Float64(1.6048173564501866327e-10)
    t[8][7] = Float64(-7.374448251684149567e-13)

    t[9][0] = Float64(1.0)
    t[9][1] = Float64(-0.16666666666666665741)
    t[9][2] = Float64(0.0083333333333331875009)
    t[9][3] = Float64(-0.00019841269841208762723)
    t[9][4] = Float64(2.755731921124346343e-06)
    t[9][5] = Float64(-2.5052106891767694322e-08)
    t[9][6] = Float64(1.6058940935873569415e-10)
    t[9][7] = Float64(-7.6430279891879434731e-13)
    t[9][8] = Float64(2.7215894431114360458e-15)

    t[10][0] = Float64(1.0)
    t[10][1] = Float64(-0.16666666666666665741)
    t[10][2] = Float64(0.0083333333333333332177)
    t[10][3] = Float64(-0.00019841269841269670491)
    t[10][4] = Float64(2.755731922394075836e-06)
    t[10][5] = Float64(-2.5052108378607547386e-08)
    t[10][6] = Float64(1.6059043206669027558e-10)
    t[10][7] = Float64(-7.6471277547197355244e-13)
    t[10][8] = Float64(2.8102148538895010548e-15)
    t[10][9] = Float64(-7.982547681148690007e-18)

    return t^


def cos_coeffs_unbiased() -> InlineArray[InlineArray[Float64, 11], 11]:
    var t = InlineArray[InlineArray[Float64, 11], 11](
        fill=InlineArray[Float64, 11](fill=Float64(0.0))
    )

    t[4][0] = Float64(0.99999995346667014395)
    t[4][1] = Float64(-0.49999905347076728512)
    t[4][2] = Float64(0.041663584693107838519)
    t[4][3] = Float64(-0.0013853704308231897576)
    t[4][4] = Float64(2.3153931659053876487e-05)

    t[5][0] = Float64(0.99999999978065168271)
    t[5][1] = Float64(-0.49999999358471769462)
    t[5][2] = Float64(0.041666636258070294252)
    t[5][3] = Float64(-0.001388836140027525012)
    t[5][4] = Float64(2.4760161352583123003e-05)
    t[5][5] = Float64(-2.6051495215482709941e-07)

    t[6][0] = Float64(0.9999999999992518207)
    t[6][1] = Float64(-0.49999999997024030529)
    t[6][2] = Float64(0.041666666473385197134)
    t[6][3] = Float64(-0.0013888884180011647294)
    t[6][4] = Float64(2.4801040648797846881e-05)
    t[6][5] = Float64(-2.7524696389812372577e-07)
    t[6][6] = Float64(1.9907856852657760445e-09)

    t[7][0] = Float64(0.99999999999999811262)
    t[7][1] = Float64(-0.49999999999989963584)
    t[7][2] = Float64(0.041666666665811744052)
    t[7][3] = Float64(-0.0013888888861136173929)
    t[7][4] = Float64(2.4801582876045364528e-05)
    t[7][5] = Float64(-2.7556935768737297612e-07)
    t[7][6] = Float64(2.0858327960118538175e-09)
    t[7][7] = Float64(-1.1008071636607462125e-11)

    t[8][0] = Float64(1.0)
    t[8][1] = Float64(-0.49999999999999972244)
    t[8][2] = Float64(0.041666666666663888796)
    t[8][3] = Float64(-0.0013888888888773172521)
    t[8][4] = Float64(2.4801587277443952164e-05)
    t[8][5] = Float64(-2.7557316393535509533e-07)
    t[8][6] = Float64(2.0876561960138396177e-09)
    t[8][7] = Float64(-1.146290489963444752e-11)
    t[8][8] = Float64(4.6090073768525871442e-14)

    t[9][0] = Float64(1.0)
    t[9][1] = Float64(-0.5)
    t[9][2] = Float64(0.041666666666666657415)
    t[9][3] = Float64(-0.0013888888888888529464)
    t[9][4] = Float64(2.4801587301492745584e-05)
    t[9][5] = Float64(-2.7557319209666753951e-07)
    t[9][6] = Float64(2.0876755667423674151e-09)
    t[9][7] = Float64(-1.1470670199183163334e-11)
    t[9][8] = Float64(4.7768729810118967628e-14)
    t[9][9] = Float64(-1.5119893746914475645e-16)

    t[10][0] = Float64(1.0)
    t[10][1] = Float64(-0.5)
    t[10][2] = Float64(0.041666666666666664354)
    t[10][3] = Float64(-0.0013888888888888887251)
    t[10][4] = Float64(2.4801587301587023739e-05)
    t[10][5] = Float64(-2.7557319223933224002e-07)
    t[10][6] = Float64(2.0876756981654129174e-09)
    t[10][7] = Float64(-1.1470745126775754607e-11)
    t[10][8] = Float64(4.7794543940744220719e-14)
    t[10][9] = Float64(-1.5612263430418807802e-16)
    t[10][10] = Float64(3.991265464089678298e-19)

    return t^


comptime SIN_MINIMAX = sin_coeffs_unbiased()
comptime COS_MINIMAX = cos_coeffs_unbiased()


@fieldwise_init
struct SinCosResult[width: Int = 1]:
    var sin_val: SIMD[DType.float64, Self.width]
    var cos_val: SIMD[DType.float64, Self.width]


# Error profile — max absolute error (sin/cos bounded by 1, abs ≈ rel near peak).
# Sweep θ ∈ [-π, π], 2^18 samples, default flags. Reproduce / asserted in
# tests/test_sincos.mojo.
#
# Sincos is f64-native; the parametric range buys you precision tiers from
# the existing N=4 (~1e-6) all the way to f64 ULP at N=8.
#
#   polynomial_degree=4 : abs ≈ 1.18e-6   (~9.9 f32 ULPs)
#                         existing-equivalent; ample for f32 output.
#   polynomial_degree=5 : abs ≈ 6.7e-9    (sub-f32 ULP)
#                         f32-loss-free with margin.
#   polynomial_degree=6 : abs ≈ 2.7e-11   (~1.2e5 f64 ULPs)
#   polynomial_degree=7 : abs ≈ 7.9e-14   (~350 f64 ULPs)
#   polynomial_degree=8 : abs ≈ 7.7e-16   (~3.4 f64 ULPs)
#                         loss-free for f64 — recommended for precision-critical
#                         RoPE / attention rotations.
#   polynomial_degree=9 : abs ≈ 6.9e-16   (~3.1 f64 ULPs)  saturated
#   polynomial_degree=10: abs ≈ 6.9e-16   (~3.1 f64 ULPs)  saturated
#
# cody_waite=True: 2-piece 2π split for f64-precise reduction at large angles.
# Without it, naive `n * TWO_PI` reduction loses precision at ~1e-9 per ±10⁵ angle.
# Required for RoPE at large positions (>10⁵). Cost: 1 extra FMA.
#   |θ| ≤ 10³ : naive ≈ 4e-14 (cody_waite no-op)
#   |θ| ≤ 10⁵ : naive ≈ 4e-12, cody_waite ≈ 7e-16
#   |θ| ≤ 10⁶ : naive ≈ 4e-11, cody_waite ≈ 9e-16
#   |θ| ≤ 10⁷ : naive ≈ 4e-10, cody_waite ≈ 9e-16
# 2-piece exact for n up to 2²⁸ (≈10⁹ angles); past that, 3-piece would be needed.
@always_inline
def sincos_simd[
    polynomial_degree: Int,
    width: Int,
    *,
    cody_waite: Bool = False,
](angles: SIMD[DType.float64, width]) -> SinCosResult[width]:
    comptime assert polynomial_degree >= 4 and polynomial_degree <= 10, (
        "sincos_simd polynomial_degree must be in [4, 10]"
    )

    comptime HALF_PI = Float64(1.57079632679489661923)
    comptime TWO_PI = Float64(6.28318530717958647692)
    comptime INV_TWO_PI = Float64(0.15915494309189533577)
    comptime TWO_PI_HI = Float64(6.2831853032112121582)
    comptime TWO_PI_LO = Float64(3.9683743187221617008e-09)

    var n = (angles * INV_TWO_PI).cast[DType.int64]().cast[DType.float64]()
    var x: SIMD[DType.float64, width]
    comptime if cody_waite:
        x = (angles - n * TWO_PI_HI) - n * TWO_PI_LO
    else:
        x = angles - n * TWO_PI

    var neg = x.to_bits() >> 63
    x = x + TWO_PI * neg.cast[DType.float64]()

    var quad = (x / HALF_PI).cast[DType.int64]()
    var under4 = ((quad - 4) >> 63) & 1
    quad = quad * under4 + SIMD[DType.int64, width](3) * (1 - under4)
    var r = x - quad.cast[DType.float64]() * HALF_PI

    var r2 = r * r

    ref sin_coeffs = SIN_MINIMAX[polynomial_degree]
    var sin_t = SIMD[DType.float64, width](sin_coeffs[polynomial_degree - 1])
    comptime for k_off in range(polynomial_degree - 1):
        comptime k = polynomial_degree - 2 - k_off
        sin_t = SIMD[DType.float64, width](sin_coeffs[k]) + r2 * sin_t
    var sin_r = r * sin_t

    ref cos_coeffs = COS_MINIMAX[polynomial_degree]
    var cos_r = SIMD[DType.float64, width](cos_coeffs[polynomial_degree])
    comptime for k_off in range(polynomial_degree):
        comptime k = polynomial_degree - 1 - k_off
        cos_r = SIMD[DType.float64, width](cos_coeffs[k]) + r2 * cos_r

    var swap = (quad & 1).cast[DType.float64]()
    var s_base = sin_r + swap * (cos_r - sin_r)
    var c_base = cos_r + swap * (sin_r - cos_r)
    var sin_sign = 1.0 - 2.0 * (quad >> 1).cast[DType.float64]()
    var cos_sign = 1.0 - 2.0 * ((quad & 1) ^ (quad >> 1)).cast[DType.float64]()

    return SinCosResult[width](s_base * sin_sign, c_base * cos_sign)
