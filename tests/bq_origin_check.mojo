from std.collections import List
from std.math import sqrt

from butterquant.fwht import fwht_block
from butterquant.convert import store_bf16
from butterquant.types import F32Ptr, BF16Ptr


def ilog2(n: Int) -> Int:
    var s = 0
    var v = n
    while v > 1:
        v >>= 1
        s += 1
    return s


def scalar_fwht(read x: List[Float32], block: Int) -> List[Float32]:
    var cur = List[Float32](capacity=block)
    for i in range(block):
        cur.append(x[i])
    var stages = ilog2(block)
    for stage in range(stages):
        var stride = 1 << stage
        var nxt = List[Float32](capacity=block)
        for k in range(block):
            var partner = k ^ stride
            if ((k >> stage) & 1) == 0:
                nxt.append(cur[k] + cur[partner])
            else:
                nxt.append(cur[k] - cur[partner])
        cur = nxt^
    var norm = Float32(1.0) / Float32(sqrt(Float64(block)))
    for i in range(block):
        cur[i] = cur[i] * norm
    return cur^


def fwht_energy_case[block: Int]() -> Bool:
    var buf = List[Float32](capacity=block)
    var energy_in = Float32(0)
    for i in range(block):
        var x = Float32(i + 1) - Float32(block) * Float32(0.5)
        buf.append(x)
        energy_in += x * x

    var ptr: F32Ptr = buf.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    fwht_block[block](ptr)

    var saw_nan = False
    var energy_out = Float32(0)
    for i in range(block):
        var got = (ptr + i).load()
        if got != got:
            saw_nan = True
        energy_out += got * got

    var rel = abs(energy_out - energy_in) / (energy_in + Float32(1e-6))
    var ok = (not saw_nan) and rel < Float32(1e-3)
    print("  fwht energy block=", block, " nan=", saw_nan,
        " E_in=", energy_in, " E_out=", energy_out, " rel=", rel,
        " ->", "PASS" if ok else "FAIL")
    _ = buf
    return ok


def fwht_dot_case[block: Int]() -> Bool:
    var xb = List[Float32](capacity=block)
    var wb = List[Float32](capacity=block)
    var dot_before = Float32(0)
    for i in range(block):
        var xi = Float32(i + 1) - Float32(block) * Float32(0.5)
        var wi = Float32((i * 7 + 3) % 11) - Float32(5.0)
        xb.append(xi)
        wb.append(wi)
        dot_before += xi * wi

    var xp: F32Ptr = xb.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var wp: F32Ptr = wb.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    fwht_block[block](xp)
    fwht_block[block](wp)

    var saw_nan = False
    var dot_after = Float32(0)
    for i in range(block):
        var a = (xp + i).load()
        var b = (wp + i).load()
        if a != a or b != b:
            saw_nan = True
        dot_after += a * b

    var rel = abs(dot_after - dot_before) / (abs(dot_before) + Float32(1e-3))
    var ok = (not saw_nan) and rel < Float32(1e-3)
    print("  fwht dot    block=", block, " nan=", saw_nan,
        " dot_before=", dot_before, " dot_after=", dot_after, " rel=", rel,
        " ->", "PASS" if ok else "FAIL")
    _ = xb
    _ = wb
    return ok


def run_store_bf16_case() -> Bool:
    var vals = SIMD[DType.float32, 4](1.0, -2.5, 0.333333, 12345.0)
    var dst = List[BFloat16](capacity=4)
    for _ in range(4):
        dst.append(BFloat16(0))
    var dptr: BF16Ptr = dst.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    store_bf16[4](vals, dptr)

    var ok = True
    for i in range(4):
        var got = Float32((dptr + i).load())
        var want = Float32(vals[i].cast[DType.bfloat16]())
        var rel = abs(got - want) / (abs(want) + Float32(1e-6))
        if rel > Float32(0.02):
            ok = False
        print("  store_bf16 lane", i, " got=", got, " ref=", want,
            "->", "PASS" if rel <= Float32(0.02) else "FAIL")
    _ = dst
    return ok


def main():
    print("== butterquant origin / primitive check ==")
    var all_ok = True

    print("[FWHT dot-product invariance]")
    all_ok = fwht_dot_case[4]() and all_ok
    all_ok = fwht_dot_case[8]() and all_ok
    all_ok = fwht_dot_case[16]() and all_ok
    all_ok = fwht_dot_case[32]() and all_ok
    all_ok = fwht_dot_case[64]() and all_ok
    all_ok = fwht_dot_case[128]() and all_ok
    all_ok = fwht_dot_case[256]() and all_ok

    print("[FWHT energy preservation]")
    all_ok = fwht_energy_case[8]() and all_ok
    all_ok = fwht_energy_case[16]() and all_ok
    all_ok = fwht_energy_case[64]() and all_ok

    print("[store_bf16]")
    all_ok = run_store_bf16_case() and all_ok

    print()
    if all_ok:
        print("RESULT: ALL PASS")
    else:
        print("RESULT: FAIL (butterquant primitive miscompiled)")
