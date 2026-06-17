from std.collections import InlineArray


# Stand-in for modeling.model_spec.WeightDesc -- only what placement needs.
@fieldwise_init
struct DescLite(Copyable, Movable):
    var name: String
    var off: Int
    var rows: Int
    var cols: Int
    var rank: Int


# ----------------------------------------------------------------------------
# GENERATORS: pure name enumeration. The ONLY model-specific knowledge is the
# string convention. No offsets, no shapes, no ranks -- just the ordered group
# of K source names. K is the comptime loop bound, so len(names) is known at
# plan-build time.
# ----------------------------------------------------------------------------
def gen_gate_up[num_experts: Int](prefix: String, mut names: List[String]):
    for e in range(num_experts):
        var ep = prefix + String(t"block_sparse_moe.experts.{e}.")
        names.append(ep + "w1.weight")  # gate
        names.append(ep + "w3.weight")  # up   (order encodes the fusion)


def gen_down[num_experts: Int](prefix: String, mut names: List[String]):
    for e in range(num_experts):
        var ep = prefix + String(t"block_sparse_moe.experts.{e}.")
        names.append(ep + "w2.weight")


# ----------------------------------------------------------------------------
# CONSUMER: uniform derivation. Everything comes from the slot's declared
# region geometry + K + degree. No per-model knowledge, no header reads.
#   element_bytes = region_bytes / K        (equal-tiling, entailed by shape)
#   rank(i)       = i // (K / degree)        (block sharding, from the shape)
#   off(i)        = (i % per_rank) * chunk   (contiguous, emission order)
# This is what emit_descs would do per emitted name.
# ----------------------------------------------------------------------------
def derive_descs(
    names: List[String], region_rows: Int, region_cols: Int, elt: Int,
    region_base: Int, degree: Int, mut out: List[DescLite],
):
    var k = len(names)
    var per_rank = k // degree
    var chunk_rows = region_rows // k
    var chunk_bytes = chunk_rows * region_cols * elt
    for i in range(k):
        var rank = i // per_rank
        var local = i % per_rank
        out.append(DescLite(
            names[i], region_base + local * chunk_bytes,
            chunk_rows, region_cols, rank))


def main():
    comptime NUM_EXPERTS = 128
    comptime HIDDEN = 6144
    comptime MOE_INTERMEDIATE = 3072
    comptime GATE_UP_FUSED = 2 * MOE_INTERMEDIATE
    comptime ELT = 2

    var prefix = String("language_model.model.layers.3.")

    # Region geometry as the slot SHAPE already declares it:
    #   ExpertRowBlockSharded[NUM_EXPERTS, GATE_UP_FUSED, HIDDEN]
    var gu_rows = NUM_EXPERTS * GATE_UP_FUSED
    var gu_cols = HIDDEN

    var up_half = MOE_INTERMEDIATE * HIDDEN * ELT
    var gate_up_bytes = 2 * up_half

    print("=== gate_up, degree=1 ===")
    var names = List[String]()
    gen_gate_up[NUM_EXPERTS](prefix, names)
    var d = List[DescLite]()
    derive_descs(names, gu_rows, gu_cols, ELT, 0, 1, d)
    print("K =", len(names), "(expect 256)")
    for i in range(4):
        print("  ", d[i].name, "off=", d[i].off, "rows=", d[i].rows,
              "cols=", d[i].cols, "rank=", d[i].rank)
    print("  EXPECT e0.w3 off=", up_half, " e1.w1 off=", gate_up_bytes,
          " rows=", MOE_INTERMEDIATE)

    print("=== gate_up, degree=2 (experts block-shard, offset resets) ===")
    var d2 = List[DescLite]()
    derive_descs(names, gu_rows, gu_cols, ELT, 0, 2, d2)
    print("  i=127:", d2[127].name, "rank=", d2[127].rank, "off=", d2[127].off)
    print("  i=128:", d2[128].name, "rank=", d2[128].rank, "off=", d2[128].off,
          "(expect rank=1 off=0)")

    print("=== down, degree=1 ===")
    var dn_rows = NUM_EXPERTS * HIDDEN
    var dn_cols = MOE_INTERMEDIATE
    var names_d = List[String]()
    gen_down[NUM_EXPERTS](prefix, names_d)
    var dd = List[DescLite]()
    derive_descs(names_d, dn_rows, dn_cols, ELT, 0, 1, dd)
    print("  K =", len(names_d), "(expect 128)")
    print("  ", dd[0].name, "rows=", dd[0].rows, "cols=", dd[0].cols, "off=", dd[0].off)
    print("  ", dd[1].name, "off=", dd[1].off, " EXPECT", HIDDEN * MOE_INTERMEDIATE * ELT)
