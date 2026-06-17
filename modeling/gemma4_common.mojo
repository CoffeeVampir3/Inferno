from std.collections import InlineArray


struct Gemma4BaseConfig:
    comptime HIDDEN = 2816
    comptime NUM_LAYERS = 30
    comptime NUM_HEADS = 16

    comptime HEAD_DIM_SLIDING = 256
    comptime NUM_KV_HEADS_SLIDING = 8
    comptime Q_DIM_SLIDING = 4096
    comptime KV_DIM_SLIDING = 2048
    comptime ROPE_HALF_SLIDING = Self.HEAD_DIM_SLIDING // 2

    comptime HEAD_DIM_FULL = 512
    comptime NUM_KV_HEADS_FULL = 2
    comptime Q_DIM_FULL = 8192
    comptime KV_DIM_FULL = 1024
    # Partial rotary: 0.25 * HEAD_DIM_FULL / 2 = 64 dims rotate, rest pass through.
    comptime ROPE_HALF_FULL = 64

    comptime INTERMEDIATE = 2112
    comptime MOE_GATE_UP_FUSED = 1408
    comptime MOE_INTERMEDIATE = 704
    comptime DOWN_FWHT_BLOCK = 64
    comptime NUM_EXPERTS = 128
    comptime TOP_K = 8

    comptime VOCAB_SIZE = 262144
    comptime NUM_SLIDING_LAYERS = 25
    comptime NUM_FULL_LAYERS = 5
    comptime SLIDING_WINDOW = 1024
    comptime RMS_NORM_EPS = 1e-6
    comptime LOGIT_SOFTCAP = 30.0


struct LayerKind:
    comptime FULL = 0
    comptime SLIDING = 1


@fieldwise_init
struct LayerEntry(Copyable, ImplicitlyCopyable):
    var idx: Int
    var kind: Int
    var local_idx: Int


@always_inline
def build_layer_schedule() -> InlineArray[
    LayerEntry, Gemma4BaseConfig.NUM_LAYERS,
]:
    var out = InlineArray[LayerEntry, Gemma4BaseConfig.NUM_LAYERS](
        uninitialized=True,
    )
    var si = 0
    var fi = 0
    for i in range(Gemma4BaseConfig.NUM_LAYERS):
        if (i + 1) % 6 == 0:
            out[i] = LayerEntry(idx=i, kind=LayerKind.FULL, local_idx=fi)
            fi += 1
        else:
            out[i] = LayerEntry(
                idx=i, kind=LayerKind.SLIDING, local_idx=si,
            )
            si += 1
    return out


comptime LAYER_SCHEDULE = build_layer_schedule()
