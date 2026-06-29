from std.collections import InlineArray


struct MinimaxM3Config:
    comptime HIDDEN = 6144
    comptime NUM_LAYERS = 60
    comptime NUM_DENSE_LAYERS = 3
    comptime NUM_SPARSE_LAYERS = 57

    comptime NUM_HEADS = 64
    comptime NUM_KV_HEADS = 4
    comptime HEAD_DIM = 128
    comptime Q_DIM = 8192
    comptime KV_DIM = 512
    comptime GQA_RATIO = Self.NUM_HEADS // Self.NUM_KV_HEADS

    comptime ROPE_HALF = 32
    comptime ROPE_ROTARY_DIM = 2 * Self.ROPE_HALF
    comptime ROPE_THETA = 5000000.0

    comptime DENSE_INTERMEDIATE = 12288
    comptime SHARED_INTERMEDIATE = 3072
    comptime MOE_INTERMEDIATE = 3072
    comptime MOE_GATE_UP_FUSED = 2 * Self.MOE_INTERMEDIATE
    comptime NUM_EXPERTS = 128
    comptime TOP_K = 4
    comptime ROUTED_SCALING = 2.0

    comptime INDEX_NUM_HEADS = 4
    comptime INDEX_HEAD_DIM = 128
    comptime INDEX_Q_DIM = 512
    comptime INDEX_K_DIM = 128
    comptime INDEX_BLOCK = 128
    comptime INDEX_TOPK_BLOCKS = 16
    comptime INDEX_LOCAL_BLOCKS = 1

    comptime SWIGLU_ALPHA = 1.702
    comptime SWIGLU_LIMIT = 7.0

    comptime VOCAB_SIZE = 200064
    comptime RMS_NORM_EPS = 1e-6


struct LayerKind:
    comptime DENSE = 0
    comptime SPARSE = 1


@fieldwise_init
struct LayerEntry(Copyable, ImplicitlyCopyable):
    var idx: Int
    var kind: Int
    var local_idx: Int


@always_inline
def build_layer_schedule() -> InlineArray[
    LayerEntry, MinimaxM3Config.NUM_LAYERS,
]:
    var out = InlineArray[LayerEntry, MinimaxM3Config.NUM_LAYERS](
        uninitialized=True,
    )
    var di = 0
    var si = 0
    for i in range(MinimaxM3Config.NUM_LAYERS):
        if i < MinimaxM3Config.NUM_DENSE_LAYERS:
            out[i] = LayerEntry(idx=i, kind=LayerKind.DENSE, local_idx=di)
            di += 1
        else:
            out[i] = LayerEntry(idx=i, kind=LayerKind.SPARSE, local_idx=si)
            si += 1
    return out


comptime LAYER_SCHEDULE = build_layer_schedule()
