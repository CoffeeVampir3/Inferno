trait Encoding:
    comptime DTYPE: DType
    comptime ELEMENT_BYTES: Int


struct BF16(Encoding):
    comptime DTYPE = DType.bfloat16
    comptime ELEMENT_BYTES = 2

struct F32(Encoding):
    comptime DTYPE = DType.float32
    comptime ELEMENT_BYTES = 4

struct I8(Encoding):
    comptime DTYPE = DType.int8
    comptime ELEMENT_BYTES = 1


comptime DEFAULT_ALIGNMENT = 64


@always_inline
def align_up(value: Int, alignment: Int = DEFAULT_ALIGNMENT) -> Int:
    return ((value + alignment - 1) // alignment) * alignment


trait ShapeLike:
    comptime SIZE_ON_DISK_N: Int
    comptime SIZE_ON_DISK_M: Int
    comptime SHARD_N: Bool
    comptime SHARD_M: Bool

    @staticmethod
    def logical_n(degree: Int) -> Int: ...

    @staticmethod
    def logical_m(degree: Int) -> Int: ...

    @staticmethod
    def data_n(degree: Int) -> Int: ...

    @staticmethod
    def data_m(degree: Int) -> Int: ...

    @staticmethod
    def bytes(degree: Int, elt_bytes: Int) -> Int: ...


struct Shape[
    rows_on_disk: Int, cols_on_disk: Int,
    shard_n: Bool = False, shard_m: Bool = False,
    block_n: Int = 1, block_m: Int = 1,
](ShapeLike):
    """Degree-free shape: the logical extent + shard axis are comptime
    (intrinsic to the model); the physical per-rank extent is a runtime
    function of the runtime tensor-parallel `degree`."""
    comptime SIZE_ON_DISK_N = Self.rows_on_disk
    comptime SIZE_ON_DISK_M = Self.cols_on_disk
    comptime SHARD_N = Self.shard_n
    comptime SHARD_M = Self.shard_m

    @always_inline
    @staticmethod
    def logical_n(degree: Int) -> Int:
        return (
            align_up(Self.rows_on_disk, degree * Self.block_n)
            if Self.shard_n else Self.rows_on_disk
        )

    @always_inline
    @staticmethod
    def logical_m(degree: Int) -> Int:
        return (
            align_up(Self.cols_on_disk, degree * Self.block_m)
            if Self.shard_m else Self.cols_on_disk
        )

    @always_inline
    @staticmethod
    def data_n(degree: Int) -> Int:
        return Self.logical_n(degree) // degree if Self.shard_n else Self.rows_on_disk

    @always_inline
    @staticmethod
    def data_m(degree: Int) -> Int:
        return Self.logical_m(degree) // degree if Self.shard_m else Self.cols_on_disk

    @always_inline
    @staticmethod
    def bytes(degree: Int, elt_bytes: Int) -> Int:
        return Self.data_n(degree) * Self.data_m(degree) * elt_bytes


comptime Replicated[n: Int, m: Int] = Shape[
    n, m, shard_n=False, shard_m=False,
]
comptime TensorRowSharded[n: Int, m: Int, block: Int = 1] = Shape[
    n, m, shard_n=True, block_n=block,
]
comptime TensorColumnSharded[n: Int, m: Int, block: Int = 1] = Shape[
    n, m, shard_m=True, block_m=block,
]
comptime ContextRowSharded[n: Int, m: Int] = Shape[
    n, m, shard_n=True,
]
comptime ExpertRowBlockSharded[
    experts: Int, rows_per_expert: Int, cols: Int,
] = Shape[
    experts * rows_per_expert, cols, shard_n=True,
]
comptime VocabularyRowSharded[n: Int, m: Int] = Shape[
    n, m, shard_n=True,
]


comptime DISTRIBUTED = -1


@fieldwise_init
struct WeightDesc(Copyable):
    var name: String
    var arena_offset: Int
    var dtype: DType
    var element_bytes: Int
    var global_rows: Int
    var global_cols: Int
    var local_cols: Int
    var data_rows: Int
    var data_cols: Int
    var target_rank: Int
