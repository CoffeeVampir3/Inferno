from std.memory import Span


trait BurstKernel(TrivialRegisterPassable):
    """Trivial mailbox payload that knows how to execute itself.

    The dispatcher transports kernels through worker-local raw mailbox storage.
    Kernel fields must therefore be trivial descriptor data, such as pointers,
    spans, scalars, and SIMD values, with no ownership or destructor semantics.
    """
    def execute(mut self): ...


trait SleepableThreadPool:
    def wake(mut self): ...
    def sleep(mut self): ...


trait BurstThreadPool(Movable, ImplicitlyDeletable, SleepableThreadPool):
    def get_capacity(self) -> Int: ...

    def dispatch[K: BurstKernel, origin: MutOrigin](
        mut self, kernels: Span[K, origin], num_jobs: Int): ...

    def join(mut self): ...

    def last_worker_timestamp(self) -> Int: ...
