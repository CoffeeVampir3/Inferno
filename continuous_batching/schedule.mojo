from std.atomic import Atomic, Ordering
from std.memory import ArcPointer

from kernels.flash_sample import SamplingParams, SampleOutcome

from .paging import KVPageAccountant, BatchGeometry


comptime MAXIMUM_SAMPLING_LOGITS = 256


struct CancelToken(Copyable, Movable, ImplicitlyCopyable):
    var flag: ArcPointer[Scalar[DType.int]]

    def __init__(out self):
        self.flag = ArcPointer(Scalar[DType.int](0))

    @always_inline
    def cancel(mut self):
        Atomic[DType.int].store[ordering=Ordering.RELEASE](
            self.flag.unsafe_ptr(), 1)

    @always_inline
    def cancelled(self) -> Bool:
        return Atomic[DType.int].load[ordering=Ordering.ACQUIRE](
            self.flag.unsafe_ptr()) != 0


@fieldwise_init
struct BatchSlot(Copyable, Movable, ImplicitlyCopyable):
    var seq_id: Int
    var request_id: Int
    var base_pos: Int
    var n_tokens: Int
    var prompt_len: Int
    var emit: Bool
    var sampling: SamplingParams
    var cancel: CancelToken

    @always_inline
    def prefill_count(self) -> Int:
        var prefill = self.prompt_len - self.base_pos
        if prefill < 0:
            return 0
        if prefill > self.n_tokens:
            return self.n_tokens
        return prefill

    @always_inline
    def decode_count(self) -> Int:
        return self.n_tokens - self.prefill_count()


@fieldwise_init
struct PageCopy(Copyable, Movable, ImplicitlyCopyable):
    var pool: Int
    var src_page: Int
    var dst_page: Int
    var pos_start: Int
    var pos_count: Int


struct Schedule(Movable):
    var slots: List[BatchSlot]
    var tokens: List[Int32]
    var copies: List[PageCopy]
    var prefill_tokens: Int
    var decode_tokens: Int

    def __init__(out self):
        self.slots = List[BatchSlot]()
        self.tokens = List[Int32]()
        self.copies = List[PageCopy]()
        self.prefill_tokens = 0
        self.decode_tokens = 0

    def clear(mut self):
        self.slots.clear()
        self.tokens.clear()
        self.copies.clear()
        self.prefill_tokens = 0
        self.decode_tokens = 0

    def fully_cancelled(self) -> Bool:
        if len(self.slots) == 0:
            return False
        for s in range(len(self.slots)):
            if not self.slots[s].cancel.cancelled():
                return False
        return True


trait ScheduledModel:
    comptime POSITIONS_PER_PAGE: Int

    def batch_geometry(self) -> BatchGeometry: ...

    def execute(
        mut self,
        read schedule: Schedule,
        read pages: KVPageAccountant,
    ) -> List[SampleOutcome[MAXIMUM_SAMPLING_LOGITS]]: ...
