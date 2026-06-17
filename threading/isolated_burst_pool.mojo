from std.sys.info import size_of
from std.memory import Span, UnsafePointer, memcpy
from std.os import abort
from std.time import perf_counter_ns
import linux.sys as linux
from std.atomic import Ordering
from numa import NumaTopology, CpuMask
from .threading_traits import BurstKernel, BurstThreadPool
from .threading_shared import (
    AtomicInt32, KernelFn, JoinFlag, SlotLayout,
    MAILBOX_DATA_SLOTS, MAILBOX_DATA_BYTES,
    kernel_trampoline,
    compute_slot_size, ptr,
)

comptime SPIN_PAUSES_PER_CYCLE = 33


@align(64)
struct WorkerMailbox:
    """Dispatch slot on the worker's NUMA node. Worker reads locally."""
    var job_ready: AtomicInt32  # 0=idle, 1=work available
    var func_ptr: Int
    var data: InlineArray[Int, MAILBOX_DATA_SLOTS]

    def __init__(out self):
        self.job_ready = AtomicInt32(0)
        self.func_ptr = 0
        self.data = InlineArray[Int, MAILBOX_DATA_SLOTS](fill=0)


@align(64)
struct SharedState:
    var shutdown: AtomicInt32
    var parked: AtomicInt32

    def __init__(out self):
        self.shutdown = AtomicInt32(0)
        self.parked = AtomicInt32(0)


@fieldwise_init
struct WorkerStackHead[mask_size: Int]:
    var entry: Int
    var slot_base: Int
    var parent_fs: Int
    var worker_id: Int
    var mailbox: UnsafePointer[WorkerMailbox, MutAnyOrigin]
    var join_flag: UnsafePointer[JoinFlag, MutAnyOrigin]
    var shared: UnsafePointer[SharedState, MutAnyOrigin]
    var altstack_base: Int
    var altstack_size: Int
    var cpu_mask: CpuMask[Self.mask_size]


struct WorkerSlot(Movable):
    var base: UnsafePointer[UInt8, MutAnyOrigin]
    var child_tid: UnsafePointer[Int32, MutAnyOrigin]
    var stack_top: UnsafePointer[UInt8, MutAnyOrigin]

    def __init__(out self, slot_base: Int):
        self.base = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=slot_base)
        self.child_tid = UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=slot_base + SlotLayout.CHILD_TID)
        self.stack_top = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=slot_base + SlotLayout.HEADER + SlotLayout.GUARD)


struct IsolatedBurstPool[mask_size: Int = 128](BurstThreadPool):
    """Dual-mailbox burst pool for isolated cores.

    Workers spin on local mailboxes. Join polls local flags.
    Zero cross-NUMA reads on the hot path.
    """
    # Worker-side (on worker's NUMA node)
    var slots: List[WorkerSlot]
    var mailboxes: UnsafePointer[WorkerMailbox, MutAnyOrigin]
    var shared: UnsafePointer[SharedState, MutAnyOrigin]
    var worker_arena: Int
    var worker_arena_size: Int

    # Main-thread-side (on main's NUMA node)
    var join_flags: UnsafePointer[JoinFlag, MutAnyOrigin]
    var join_arena: Int
    var join_arena_size: Int

    # State
    var capacity: Int
    var active_jobs: Int
    var stack_size: Int
    var slot_size: Int
    var numa_node: Optional[Int]
    var workers_alive: Bool
    var futex_flags: Int

    def __init__(out self, capacity: Int,
                 var cpu_mask: CpuMask[Self.mask_size],
                 numa_node: Optional[Int] = None,
                 stack_size: Int = SlotLayout.DEFAULT_STACK):
        self.capacity = capacity
        self.active_jobs = 0
        self.stack_size = stack_size
        self.slot_size = compute_slot_size(stack_size)
        self.numa_node = numa_node
        self.workers_alive = False
        self.futex_flags = linux.Futex2.SIZE_U32 | linux.Futex2.PRIVATE
        self.slots = List[WorkerSlot](capacity=capacity)
        self.worker_arena = 0
        self.worker_arena_size = 0
        self.join_arena = 0
        self.join_arena_size = 0
        self.mailboxes = UnsafePointer[WorkerMailbox, MutAnyOrigin].unsafe_dangling()
        self.shared = UnsafePointer[SharedState, MutAnyOrigin].unsafe_dangling()
        self.join_flags = UnsafePointer[JoinFlag, MutAnyOrigin].unsafe_dangling()

        var sys = linux.linux_sys()

        var mailbox_bytes = capacity * size_of[WorkerMailbox]()
        self.worker_arena_size = (self.slot_size * capacity
            + size_of[SharedState]() + mailbox_bytes)
        self.worker_arena = sys.sys_mmap[
            prot=linux.Prot.RW,
            flags=linux.MapFlag.PRIVATE | linux.MapFlag.ANONYMOUS
                | linux.MapFlag.NORESERVE | linux.MapFlag.POPULATE
        ](0, self.worker_arena_size)
        if self.worker_arena < 0:
            self.worker_arena = 0
            return

        if numa_node is not None:
            var nodemask = UInt64(1) << UInt64(numa_node.value())
            if sys.sys_mbind[policy=linux.Mempolicy.BIND](
                self.worker_arena, self.worker_arena_size, nodemask
            ) < 0:
                _ = sys.sys_munmap(self.worker_arena, self.worker_arena_size)
                self.worker_arena = 0
                return

        var shared_addr = self.worker_arena + self.slot_size * capacity
        self.shared = ptr[SharedState](shared_addr)
        self.shared[] = SharedState()

        self.mailboxes = ptr[WorkerMailbox](shared_addr + size_of[SharedState]())
        for i in range(capacity):
            (self.mailboxes + i)[] = WorkerMailbox()

        for i in range(capacity):
            var slot_base = self.worker_arena + i * self.slot_size
            if sys.sys_mprotect(slot_base + SlotLayout.HEADER,
                                SlotLayout.GUARD, linux.Prot.NONE) != 0:
                _ = sys.sys_munmap(self.worker_arena, self.worker_arena_size)
                self.worker_arena = 0
                return
            if sys.sys_mprotect(
                slot_base + SlotLayout.HEADER + SlotLayout.GUARD + self.stack_size,
                SlotLayout.ALT_GUARD, linux.Prot.NONE,
            ) != 0:
                _ = sys.sys_munmap(self.worker_arena, self.worker_arena_size)
                self.worker_arena = 0
                return
            self.slots.append(WorkerSlot(slot_base))

        self.join_arena_size = capacity * size_of[JoinFlag]()
        self.join_arena = sys.sys_mmap[
            prot=linux.Prot.RW,
            flags=linux.MapFlag.PRIVATE | linux.MapFlag.ANONYMOUS | linux.MapFlag.POPULATE
        ](0, self.join_arena_size)
        if self.join_arena < 0:
            _ = sys.sys_munmap(self.worker_arena, self.worker_arena_size)
            self.worker_arena = 0
            self.join_arena = 0
            return

        # No mbind — stays on the allocating (main) thread's node by first-touch
        self.join_flags = ptr[JoinFlag](self.join_arena)
        for i in range(capacity):
            (self.join_flags + i)[] = JoinFlag()

        self.spawn_workers(cpu_mask)

    def __bool__(self) -> Bool:
        return self.worker_arena != 0 and self.join_arena != 0

    def spawn_workers(mut self, cpu_mask: CpuMask[Self.mask_size]):
        var sys = linux.linux_sys()
        var parent_fs = sys.arch_thread_pointer()

        for i in range(self.capacity):
            # Per-core pinning: worker i → i-th set bit in mask
            var worker_mask = CpuMask[Self.mask_size]()
            var bit_count = 0
            for bit in range(CpuMask[Self.mask_size].bit_capacity()):
                if cpu_mask.test(bit):
                    if bit_count == i:
                        worker_mask.set(bit)
                        break
                    bit_count += 1

            var stack_top_addr = Int(self.slots[i].stack_top) + self.stack_size
            var head_addr = (stack_top_addr - size_of[WorkerStackHead[Self.mask_size]]()) & ~15
            var head = ptr[WorkerStackHead[Self.mask_size]](head_addr)
            var entry_fn = isolated_worker_main[Self.mask_size]
            var slot_base = Int(self.slots[i].base)
            var altstack_base = (
                slot_base + SlotLayout.HEADER + SlotLayout.GUARD
                + self.stack_size + SlotLayout.ALT_GUARD
            )
            head[] = WorkerStackHead[Self.mask_size](
                UnsafePointer(to=entry_fn).bitcast[Int]()[],
                slot_base,
                parent_fs,
                i,
                self.mailboxes + i,
                self.join_flags + i,
                self.shared,
                altstack_base,
                SlotLayout.ALTSTACK_SIZE,
                worker_mask,
            )
            var tcb_addr = slot_base + SlotLayout.TCB
            var clone_args = linux.Clone3Args.thread(
                Int(self.slots[i].stack_top),
                head_addr - Int(self.slots[i].stack_top),
                tcb_addr,
                Int(self.slots[i].child_tid),
            )
            var tid = sys.sys_clone3_with_entry(UnsafePointer(to=clone_args), size_of[linux.Clone3Args]())
            if tid < 0:
                print(t"clone3 failed for worker {i}")
                abort("IsolatedBurstPool.spawn_workers: clone3 failed")
        self.workers_alive = True

    def dispatch[K: BurstKernel, origin: MutOrigin](
        mut self, kernels: Span[K, origin], num_jobs: Int = -1):
        """Typed dispatch: copy kernels[i] into mailbox[i], invoke via trampoline."""
        comptime assert size_of[K]() <= MAILBOX_DATA_BYTES, "kernel exceeds mailbox capacity"

        var jobs = num_jobs if num_jobs >= 0 else len(kernels)
        if jobs <= 0:
            return
        if jobs > self.capacity:
            print(t"IsolatedBurstPool.dispatch invalid job count jobs {jobs} capacity {self.capacity}")
            abort("IsolatedBurstPool.dispatch: num_jobs exceeds pool capacity")

        debug_assert(jobs <= self.capacity, "num_jobs must be <= pool capacity")
        debug_assert(jobs <= len(kernels), "num_jobs must be <= len(kernels)")
        if self.active_jobs != 0:
            print(t"IsolatedBurstPool.dispatch invalid while jobs active active_jobs {self.active_jobs} capacity {self.capacity}")
            abort("IsolatedBurstPool.dispatch: previous dispatch still in flight")

        debug_assert(self.active_jobs == 0,
            "previous dispatch still in flight; call join() first")

        var tramp = kernel_trampoline[K]
        var tramp_ptr = UnsafePointer(to=tramp).bitcast[Int]()[]

        self.active_jobs = jobs

        # Pass 1: write dispatch data (no ordering between workers)
        for i in range(jobs):
            var mb = self.mailboxes + i
            mb[].func_ptr = tramp_ptr
            UnsafePointer(to=mb[].data[0]).bitcast[K]()[] = kernels[i]

        # Pass 2: set all job_ready flags
        for i in range(jobs):
            AtomicInt32.store[ordering=Ordering.RELEASE](
                UnsafePointer(to=(self.mailboxes + i)[].job_ready.value), 1)

    def join(mut self):
        """Wait for all dispatched jobs. Polls join flags on main's NUMA node."""
        var sys = linux.linux_sys()
        for i in range(self.active_jobs):
            var done_ptr = UnsafePointer(to=(self.join_flags + i)[].done.value)
            while AtomicInt32.load[ordering=Ordering.ACQUIRE](done_ptr) == 0:
                sys.arch_cpu_relax()
            AtomicInt32.store[ordering=Ordering.RELAXED](done_ptr, 0)
        self.active_jobs = 0

    def get_capacity(self) -> Int:
        return self.capacity

    def last_worker_timestamp(self) -> Int:
        """Max completion timestamp across all workers from the last dispatch.
        Call after join(). Workers write perf_counter_ns() before setting done."""
        var max_ts = 0
        for i in range(self.capacity):
            var ts = (self.join_flags + i)[].timestamp
            if ts > max_ts:
                max_ts = ts
        return max_ts

    def wake(mut self):
        """Unpark workers from sleep."""
        var parked_ptr = UnsafePointer(to=self.shared[].parked.value)
        AtomicInt32.store[ordering=Ordering.RELEASE](parked_ptr, 0)
        var sys = linux.linux_sys()
        _ = sys.sys_futex_wake(Int(parked_ptr), self.capacity, self.futex_flags)

    def sleep(mut self):
        """Park workers. They futex_wait until wake() is called."""
        AtomicInt32.store[ordering=Ordering.RELEASE](
            UnsafePointer(to=self.shared[].parked.value), 1)

    def __del__(deinit self):
        if not self.workers_alive:
            return
        AtomicInt32.store[ordering=Ordering.RELEASE](
            UnsafePointer(to=self.shared[].shutdown.value), 1)
        # Unpark in case workers are sleeping
        self.wake()
        var sys = linux.linux_sys()
        for i in range(self.capacity):
            while self.slots[i].child_tid[] != 0:
                sys.arch_cpu_relax()
        if self.worker_arena != 0:
            _ = sys.sys_munmap(self.worker_arena, self.worker_arena_size)
        if self.join_arena != 0:
            _ = sys.sys_munmap(self.join_arena, self.join_arena_size)

    @staticmethod
    def for_rank(topo: NumaTopology, rank: Int,
                 stack_size: Int = SlotLayout.DEFAULT_STACK) -> Self:
        """Create pool pinned to the given rank's isolated worker cores."""
        var node = topo.node(rank)
        var mask = topo.worker_mask[Self.mask_size](rank)
        var cap = mask.count()
        if cap == 0:
            cap = 1
            mask = topo.mask[Self.mask_size](rank)
        return Self(cap, mask, node, stack_size)


def isolated_worker_main[mask_size: Int](stack_head_ptr: Int):
    var head = ptr[WorkerStackHead[mask_size]](stack_head_ptr)
    var sys = linux.linux_sys()

    var ss = linux.StackT()
    ss.ss_sp = head[].altstack_base
    ss.ss_size = UInt64(head[].altstack_size)
    ss.ss_flags = 0
    _ = sys.sys_sigaltstack(UnsafePointer(to=ss))

    var slot_base = head[].slot_base
    var tcb_addr = slot_base + SlotLayout.TCB
    comptime TLS_TCB_SIZE = SlotLayout.TLS_SIZE + SlotLayout.TCB_SIZE
    memcpy(
        dest=ptr[Int8](slot_base),
        src=ptr[Int8](head[].parent_fs - SlotLayout.TLS_SIZE),
        count=TLS_TCB_SIZE,
    )
    ptr[Int](tcb_addr + SlotLayout.TCB_SELF_OFFSET)[] = tcb_addr
    ptr[Int](slot_base + SlotLayout.WORKER_ID)[] = head[].worker_id
    ptr[Int](slot_base + SlotLayout.WORKER_MAGIC)[] = SlotLayout.WORKER_MAGIC_VALUE

    _ = sys.sys_sched_setaffinity(
        0, CpuMask[mask_size].byte_size(), head[].cpu_mask.unsafe_address())

    var mailbox = head[].mailbox
    var join_flag = head[].join_flag
    var shared = head[].shared
    var ready_ptr = UnsafePointer(to=mailbox[].job_ready.value)
    var done_ptr = UnsafePointer(to=join_flag[].done.value)
    var futex_flags = linux.Futex2.SIZE_U32 | linux.Futex2.PRIVATE

    # Main loop: spin on local mailbox, write done to remote join flag.
    while True:
        if AtomicInt32.load[ordering=Ordering.ACQUIRE](ready_ptr) != 0:
            # Read dispatch data (local)
            var func_addr = mailbox[].func_ptr
            var data_ptr = Int(UnsafePointer(to=mailbox[].data[0]))
            # Clear job_ready (local write)
            AtomicInt32.store[ordering=Ordering.RELEASE](ready_ptr, 0)
            # Execute kernel — single pointer to NUMA-local data area
            UnsafePointer(to=func_addr).bitcast[KernelFn]()[](data_ptr)
            # Signal completion (remote writes to main's NUMA node)
            # Timestamp first, then done — TSO orders these stores
            join_flag[].timestamp = Int(perf_counter_ns())
            AtomicInt32.store[ordering=Ordering.RELEASE](done_ptr, 1)
        else:
            if shared[].shutdown.load[ordering=Ordering.RELAXED]() != 0:
                break
            if shared[].parked.load[ordering=Ordering.ACQUIRE]() != 0:
                var parked_ptr = UnsafePointer(to=shared[].parked.value)
                _ = sys.sys_futex_wait(Int(parked_ptr), 1, futex_flags)
            else:
                comptime for _ in range(0, SPIN_PAUSES_PER_CYCLE):
                    sys.arch_cpu_relax()

    sys.sys_exit()
