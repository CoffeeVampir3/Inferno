from std.sys.info import size_of
from std.memory import Span, UnsafePointer, memcpy
from std.os import abort
from std.time import perf_counter_ns
import linux.sys as linux
from std.atomic import Ordering, fence
from numa import NumaTopology, CpuMask
from .threading_traits import BurstKernel, BurstThreadPool
from .threading_shared import (
    AtomicInt32, KernelFn, JoinFlag, SlotLayout,
    MAILBOX_DATA_SLOTS, MAILBOX_DATA_BYTES,
    kernel_trampoline,
    compute_slot_size, ptr,
)


@align(64)
struct WorkerMailbox:
    """Dispatch slot on the worker's NUMA node.

    Worker reads locally. Dispatcher writes remotely.
    sleeping flag is for Dekker backoff protocol.
    data holds the kernel's args struct (up to MAILBOX_DATA_BYTES).
    """
    var job_ready: AtomicInt32  # 0=idle, 1=work available
    var sleeping: AtomicInt32   # Dekker: worker sets before futex_wait
    var func_ptr: Int
    var data: InlineArray[Int, MAILBOX_DATA_SLOTS]

    def __init__(out self):
        self.job_ready = AtomicInt32(0)
        self.sleeping = AtomicInt32(0)
        self.func_ptr = 0
        self.data = InlineArray[Int, MAILBOX_DATA_SLOTS](fill=0)


@align(64)
struct SharedState:
    var shutdown: AtomicInt32

    def __init__(out self):
        self.shutdown = AtomicInt32(0)


@always_inline
def current_worker_id() -> Int:
    """Return worker id when running in a BurstPool worker, else -1."""
    var sys = linux.linux_sys()
    var magic = sys.arch_tls_load_i64[offset=SlotLayout.WORKER_MAGIC_FROM_FS]()
    if magic != SlotLayout.WORKER_MAGIC_VALUE:
        return -1
    return sys.arch_tls_load_i64[offset=SlotLayout.WORKER_ID_FROM_FS]()


def burst_sigsegv_handler(signo: Int32, info: Int, ucontext: Int):
    var sys = linux.linux_sys()
    var ctx = sys.arch_decode_sigsegv(info, ucontext)
    var worker = current_worker_id()
    var pid = sys.sys_getpid()
    var tid = sys.sys_gettid()

    print(
        "burst: SIGSEGV worker=", worker,
        "pid=", pid,
        "tid=", tid,
        "rip=", hex(ctx.ip),
        "rsp=", hex(ctx.sp),
        "addr=", hex(ctx.fault_addr),
    )

    _ = sys.sys_tgkill(pid, tid, linux.Signal.SEGV)
    sys.sys_exit_group(128 + Int(signo))

def install_burst_sigsegv_handler():
    var sys = linux.linux_sys()
    var handler_copy = burst_sigsegv_handler
    var handler_addr = UnsafePointer(to=handler_copy).bitcast[Int]()[]

    var act = linux.RtSigAction(
        handler=handler_addr,
        flags=UInt64(linux.SigActionFlag.SIGINFO | linux.SigActionFlag.ONSTACK),
        mask=linux.SigSet64(0),
    )

    _ = sys.sys_rt_sigaction(linux.Signal.SEGV, UnsafePointer(to=act))


struct WorkerSlot(Movable, ImplicitlyDeletable):
    var base: UnsafePointer[UInt8, MutUntrackedOrigin]
    var child_tid: UnsafePointer[Int32, MutUntrackedOrigin]
    var stack_top: UnsafePointer[UInt8, MutUntrackedOrigin]

    def __init__(out self, slot_base: Int):
        self.base = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=slot_base)
        self.child_tid = UnsafePointer[Int32, MutUntrackedOrigin](
            unsafe_from_address=slot_base + SlotLayout.CHILD_TID)
        self.stack_top = UnsafePointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=slot_base + SlotLayout.HEADER + SlotLayout.GUARD)

    @always_inline
    def is_alive(self) -> Bool:
        return self.child_tid[] != 0


@fieldwise_init
struct WorkerStackHead[mask_size: Int]:
    var entry: Int
    var slot_base: Int
    var parent_fs: Int
    var worker_id: Int
    var mailbox: UnsafePointer[WorkerMailbox, MutUntrackedOrigin]
    var join_flag: UnsafePointer[JoinFlag, MutUntrackedOrigin]
    var shared: UnsafePointer[SharedState, MutUntrackedOrigin]
    var futex_flags: Int
    var altstack_base: Int
    var altstack_size: Int
    var pinned: Int
    var cpu_mask: CpuMask[Self.mask_size]


comptime SPIN_LIMIT = 100_000

struct BurstPool[mask_size: Int = 128](BurstThreadPool):
    """Spin-backoff pool with dual-mailbox NUMA-aware dispatch.

    Workers spin on local mailboxes, then Dekker-sleep via futex.
    Join polls local JoinFlags. Zero cross-NUMA reads on the join path.
    """
    # Worker-side (on worker's NUMA node)
    var slots: List[WorkerSlot]
    var mailboxes: UnsafePointer[WorkerMailbox, MutUntrackedOrigin]
    var shared: UnsafePointer[SharedState, MutUntrackedOrigin]
    var worker_arena: Int
    var worker_arena_size: Int

    # Main-thread-side (on main's NUMA node)
    var join_flags: UnsafePointer[JoinFlag, MutUntrackedOrigin]
    var join_arena: Int
    var join_arena_size: Int

    # State
    var capacity: Int
    var active_jobs: Int
    var stack_size: Int
    var slot_size: Int
    var cpu_mask: CpuMask[Self.mask_size]
    var numa_node: Optional[Int]
    var workers_alive: Bool
    var futex_flags: Int
    var pinned: Bool

    def __init__(out self, capacity: Int,
                 var cpu_mask: CpuMask[Self.mask_size] = CpuMask[Self.mask_size](),
                 numa_node: Optional[Int] = None,
                 stack_size: Int = SlotLayout.DEFAULT_STACK):
        self.capacity = capacity
        self.active_jobs = 0
        self.stack_size = stack_size
        self.slot_size = compute_slot_size(stack_size)
        self.pinned = cpu_mask.count() > 0
        self.cpu_mask = cpu_mask
        self.numa_node = numa_node
        self.workers_alive = False
        self.futex_flags = linux.Futex2.SIZE_U32 | linux.Futex2.PRIVATE
        self.slots = List[WorkerSlot](capacity=capacity)
        self.worker_arena = 0
        self.worker_arena_size = 0
        self.join_arena = 0
        self.join_arena_size = 0
        self.mailboxes = UnsafePointer[WorkerMailbox, MutUntrackedOrigin].unsafe_dangling()
        self.shared = UnsafePointer[SharedState, MutUntrackedOrigin].unsafe_dangling()
        self.join_flags = UnsafePointer[JoinFlag, MutUntrackedOrigin].unsafe_dangling()

        install_burst_sigsegv_handler()

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
        self.shared = UnsafePointer[SharedState, MutUntrackedOrigin](unsafe_from_address=shared_addr)
        self.shared[] = SharedState()

        self.mailboxes = UnsafePointer[WorkerMailbox, MutUntrackedOrigin](
            unsafe_from_address=shared_addr + size_of[SharedState]())
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
            var slot = WorkerSlot(slot_base)
            slot.child_tid[] = 0
            self.slots.append(slot^)

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

        # No mbind — stays on main thread's node by first-touch
        self.join_flags = UnsafePointer[JoinFlag, MutUntrackedOrigin](unsafe_from_address=self.join_arena)
        for i in range(capacity):
            (self.join_flags + i)[] = JoinFlag()

        self.spawn_workers()

    def __bool__(self) -> Bool:
        return self.worker_arena != 0 and self.join_arena != 0

    def dispatch[K: BurstKernel, origin: MutOrigin](
        mut self, kernels: Span[K, origin], num_jobs: Int = -1):
        """Typed dispatch: copy kernels[i] into mailbox[i], invoke via trampoline."""
        comptime assert size_of[K]() <= MAILBOX_DATA_BYTES, "kernel exceeds mailbox capacity"

        var jobs = num_jobs if num_jobs >= 0 else len(kernels)
        if jobs <= 0:
            return
        if jobs > self.capacity:
            print(t"BurstPool.dispatch invalid job count jobs {jobs} capacity {self.capacity}")
            abort("BurstPool.dispatch: num_jobs exceeds pool capacity")

        debug_assert(jobs <= self.capacity, "num_jobs must be <= pool capacity")
        debug_assert(jobs <= len(kernels), "num_jobs must be <= len(kernels)")
        if self.active_jobs != 0:
            print(t"BurstPool.dispatch invalid while jobs active active_jobs {self.active_jobs} capacity {self.capacity}")
            abort("BurstPool.dispatch: previous dispatch still in flight")

        debug_assert(self.active_jobs == 0,
            "previous dispatch still in flight; call join() first")

        var tramp = kernel_trampoline[K]
        var tramp_ptr = UnsafePointer(to=tramp).bitcast[Int]()[]

        self.active_jobs = jobs

        # Pass 1: write dispatch data to worker mailboxes (remote to worker's node)
        for i in range(jobs):
            var mb = self.mailboxes + i
            mb[].func_ptr = tramp_ptr
            UnsafePointer(to=mb[].data[0]).bitcast[K]()[] = kernels[i]

        # Pass 2: set job_ready flags (remote writes, RELEASE ordering)
        for i in range(jobs):
            AtomicInt32.store[ordering=Ordering.RELEASE](
                UnsafePointer(to=(self.mailboxes + i)[].job_ready.value), 1)

        # Sequential fence closes the Dekker store-load race: job_ready must be
        # published before the dispatcher observes sleeping, or the worker can
        # miss the job and park without a wake.
        fence[ordering=Ordering.SEQUENTIAL]()

        # Pass 3: Dekker wake — check sleeping, futex_wake if needed
        # If worker stored sleeping=1 but missed our job_ready store (x86 store-load
        # reordering), the futex_wait's atomic check catches it (sees job_ready=1,
        # returns EAGAIN). So this is an optimization, not a correctness requirement.
        var sys = linux.linux_sys()
        for i in range(jobs):
            if AtomicInt32.load[ordering=Ordering.ACQUIRE](
                UnsafePointer(to=(self.mailboxes + i)[].sleeping.value)
            ) != 0:
                _ = sys.sys_futex_wake(
                    Int(UnsafePointer(to=(self.mailboxes + i)[].job_ready.value)),
                    1, self.futex_flags)

    def join(mut self):
        """Wait for all dispatched jobs. Polls JoinFlags on main's NUMA node."""
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
        """Max completion timestamp across workers from the last dispatch.
        Call after join(). Workers write perf_counter_ns() before setting done."""
        var max_ts = 0
        for i in range(self.capacity):
            var ts = (self.join_flags + i)[].timestamp
            if ts > max_ts:
                max_ts = ts
        return max_ts

    def wake(mut self):
        pass

    def sleep(mut self):
        pass

    def __del__(deinit self):
        if self.worker_arena == 0:
            return

        var sys = linux.linux_sys()
        if self.workers_alive:
            AtomicInt32.store[ordering=Ordering.RELEASE](
                UnsafePointer(to=self.shared[].shutdown.value), 1)
            # Wake all sleeping workers so they see the shutdown flag
            for i in range(self.capacity):
                var ready_ptr = UnsafePointer(to=(self.mailboxes + i)[].job_ready.value)
                AtomicInt32.store[ordering=Ordering.RELEASE](ready_ptr, 1)
                _ = sys.sys_futex_wake(Int(ready_ptr), 1, self.futex_flags)
            for i in range(self.capacity):
                while self.slots[i].is_alive():
                    sys.arch_cpu_relax()

        if self.worker_arena != 0:
            _ = sys.sys_munmap(self.worker_arena, self.worker_arena_size)
        if self.join_arena != 0:
            _ = sys.sys_munmap(self.join_arena, self.join_arena_size)

    def spawn_workers(mut self):
        var sys = linux.linux_sys()
        var parent_fs = sys.arch_thread_pointer()

        for i in range(self.capacity):
            var worker_mask = CpuMask[Self.mask_size]()
            if self.pinned:
                worker_mask = self.cpu_mask.copy()

            var stack_top_addr = Int(self.slots[i].stack_top) + self.stack_size
            var head_addr = (stack_top_addr - size_of[WorkerStackHead[Self.mask_size]]()) & ~15
            var head = ptr[WorkerStackHead[Self.mask_size]](head_addr)
            var entry_fn = worker_main[Self.mask_size]
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
                self.futex_flags,
                altstack_base,
                SlotLayout.ALTSTACK_SIZE,
                Int(self.pinned),
                worker_mask,
            )
            var tcb_addr = slot_base + SlotLayout.TCB
            var clone_args = linux.Clone3Args.thread(
                Int(self.slots[i].stack_top),
                head_addr - Int(self.slots[i].stack_top),
                tcb_addr,
                Int(self.slots[i].child_tid),
            )
            var result = sys.sys_clone3_with_entry(
                UnsafePointer(to=clone_args), size_of[linux.Clone3Args]())
            if result < 0:
                print(t"clone3 failed for worker {i}")
                abort("BurstPool.spawn_workers: clone3 failed")
        self.workers_alive = True

    @staticmethod
    def for_rank(topo: NumaTopology, rank: Int,
                 stack_size: Int = SlotLayout.DEFAULT_STACK) -> Self:
        """Create pool for the given rank's NUMA node.

        With isolation: pin workers to isolated physical cores on the node.
        Without isolation: capacity = physical core count, affinity = whole
        NUMA node (all logical cores including HT siblings).
        """
        var node = topo.node(rank)
        if topo.has_isolation():
            var mask = topo.worker_mask[Self.mask_size](rank)
            var cap = mask.count()
            if cap == 0:
                cap = 1
                mask = topo.mask[Self.mask_size](rank)
            return Self(cap, mask, node, stack_size)
        var cap = topo.cpus_on(rank)
        if cap == 0:
            cap = 1
        var mask = topo.mask[Self.mask_size](rank)
        return Self(cap, mask, node, stack_size)


def worker_main[mask_size: Int](stack_head_ptr: Int):
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

    if head[].pinned != 0:
        _ = sys.sys_sched_setaffinity(
            0, CpuMask[mask_size].byte_size(), head[].cpu_mask.unsafe_address())

    var mailbox = head[].mailbox
    var join_flag = head[].join_flag
    var shared = head[].shared
    var futex_flags = head[].futex_flags
    var ready_ptr = UnsafePointer(to=mailbox[].job_ready.value)
    var sleeping_ptr = UnsafePointer(to=mailbox[].sleeping.value)
    var done_ptr = UnsafePointer(to=join_flag[].done.value)

    # Main loop: spin on local mailbox, back off to futex_wait.
    # Shutdown checked before job_ready: destructor sets both flags to wake
    # sleeping workers, and stale job_ready must not trigger execution of
    # a freed dispatch.
    while True:
        if shared[].shutdown.load[ordering=Ordering.ACQUIRE]() != 0:
            break

        if AtomicInt32.load[ordering=Ordering.ACQUIRE](ready_ptr) != 0:
            # Read dispatch data (local reads from worker's NUMA node)
            var func_addr = mailbox[].func_ptr
            var data_ptr = Int(UnsafePointer(to=mailbox[].data[0]))
            # Clear job_ready (local write)
            AtomicInt32.store[ordering=Ordering.RELEASE](ready_ptr, 0)
            # Execute kernel — single pointer to NUMA-local data area
            UnsafePointer(to=func_addr).bitcast[KernelFn]()[](data_ptr)
            # Signal completion (remote writes to main's NUMA node)
            join_flag[].timestamp = Int(perf_counter_ns())
            AtomicInt32.store[ordering=Ordering.RELEASE](done_ptr, 1)
            continue

        # Spin phase — brief spin on local job_ready
        var spins = 0
        while AtomicInt32.load[ordering=Ordering.RELAXED](ready_ptr) == 0:
            if shared[].shutdown.load[ordering=Ordering.RELAXED]() != 0:
                break
            if spins < SPIN_LIMIT:
                sys.arch_cpu_relax()
                spins += 1
            else:
                # Dekker sleep: publish sleeping=1, recheck job_ready.
                # Dispatcher publishes job_ready=1 then checks sleeping.
                # If both miss (x86 store-load reordering), futex_wait's
                # atomic check sees job_ready=1 and returns EAGAIN.
                AtomicInt32.store[ordering=Ordering.RELEASE](sleeping_ptr, 1)
                if AtomicInt32.load[ordering=Ordering.ACQUIRE](ready_ptr) != 0:
                    AtomicInt32.store[ordering=Ordering.RELEASE](sleeping_ptr, 0)
                    break
                if shared[].shutdown.load[ordering=Ordering.ACQUIRE]() != 0:
                    AtomicInt32.store[ordering=Ordering.RELEASE](sleeping_ptr, 0)
                    break
                _ = sys.sys_futex_wait(Int(ready_ptr), 0, futex_flags)
                AtomicInt32.store[ordering=Ordering.RELEASE](sleeping_ptr, 0)
                spins = 0

    sys.sys_exit()
