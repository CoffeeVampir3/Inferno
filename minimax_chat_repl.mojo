from std.pathlib import Path
from std.reflection import reflect

from numa import NumaTopology
from terminal import (
    TerminalReader, CancelWatch, CmdResult, Parsable, fill,
    PositiveFloat, UnitFloat, BoundedInt, NonNegInt, Toggle,
)
from threading.threading_traits import BurstThreadPool, SleepableThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from tokenizer import (
    load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform,
    StreamDetokenizer,
)
from kernels.flash_sample import SamplingParams
from continuous_batching.schedule import MAXIMUM_SAMPLING_LOGITS
from continuous_batching.scheduler import ContinuousBatchScheduler

from modeling.minimax_m3_bq import MinimaxM3, PAGE_LEN


comptime MODEL_DIR = "checkpoints/minimax-m3-bq"
comptime TOKENIZER_PATH = "checkpoints/minimax-m3-ablit/tokenizer.json"

comptime BOD_TOKEN_ID = 200034
comptime BOS_TOKEN_ID = 200019
comptime EOS_TOKEN_ID = 200020
comptime THINK_OPEN_TOKEN_ID = 200050
comptime THINK_CLOSE_TOKEN_ID = 200051
comptime MM_THINK_OPEN_TOKEN_ID = 200059
comptime MM_THINK_CLOSE_TOKEN_ID = 200060

comptime DEFAULT_SYSTEM = "You are MiniMax, a helpful assistant."
comptime STEP_BUDGET = PAGE_LEN
comptime MAX_CONTEXT = 32768
comptime CLEAR_SCREEN = "\x1b[2J\x1b[3J\x1b[H"
comptime YOU_PROMPT = "\n\n\x1b[32myou> \x1b[0m"
comptime MODEL_PROMPT = "\x1b[38;5;211mmodel> \x1b[0m"
comptime DIM = "\x1b[90m"
comptime RESET = "\x1b[0m"


def stop_tokens() -> List[Int32]:
    var ids = List[Int32]()
    ids.append(Int32(EOS_TOKEN_ID))
    return ids^


def is_think_open(tid: Int32) -> Bool:
    return tid == THINK_OPEN_TOKEN_ID or tid == MM_THINK_OPEN_TOKEN_ID


def is_think_close(tid: Int32) -> Bool:
    return tid == THINK_CLOSE_TOKEN_ID or tid == MM_THINK_CLOSE_TOKEN_ID


def on_off(flag: Bool) -> String:
    return String("on") if flag else String("off")


@fieldwise_init
struct ChatTurn(Copyable, Movable):
    var user: String
    var reply: String


def append_encoded(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    mut token_ids: List[Int32],
    text: String,
):
    var encoded = tok.encode(text)
    for i in range(len(encoded)):
        token_ids.append(Int32(encoded[i]))


def emit_role_turn(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    mut token_ids: List[Int32],
    role: String,
    body: String,
):
    token_ids.append(Int32(BOS_TOKEN_ID))
    append_encoded(tok, token_ids, role + "\n" + body)
    token_ids.append(Int32(EOS_TOKEN_ID))
    append_encoded(tok, token_ids, "\n")


def thinking_instructions(mode: String) -> String:
    var base = String(
        "\n\n<thinking_instructions>\nYou have a thinking capability that "
        "allows you to reason step by step before responding. When thinking "
        "is enabled, wrap your reasoning in <mm:think></mm:think> tags before "
        "your response. When thinking is disabled, begin your response "
        "directly after the </mm:think> prefix. When thinking is adaptive, "
        "decide on your own whether to think for the current turn.\n")
    var line: String
    if mode == "enabled":
        line = String(
            "Current thinking mode: enabled. You MUST think step by step "
            "before every response.\n")
    elif mode == "disabled":
        line = String(
            "Current thinking mode: disabled. Do not output any thinking "
            "process.\n")
    else:
        line = String(
            "Current thinking mode: adaptive. You are encouraged to think for "
            "complex decision-making and multi-step reasoning.\n")
    return base + line + "</thinking_instructions>"


def build_input(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    system: String,
    mode: String,
    read history: List[ChatTurn],
    user_text: String,
) -> List[Int32]:
    var token_ids = List[Int32]()
    token_ids.append(Int32(BOD_TOKEN_ID))
    emit_role_turn(tok, token_ids, "system", system + thinking_instructions(mode))
    emit_role_turn(tok, token_ids, "developer", "You are a helpful assistant.")
    for i in range(len(history)):
        emit_role_turn(tok, token_ids, "user", history[i].user)
        token_ids.append(Int32(BOS_TOKEN_ID))
        append_encoded(tok, token_ids, "ai\n")
        token_ids.append(Int32(MM_THINK_CLOSE_TOKEN_ID))
        append_encoded(tok, token_ids, history[i].reply)
        token_ids.append(Int32(EOS_TOKEN_ID))
        append_encoded(tok, token_ids, "\n")
    emit_role_turn(tok, token_ids, "user", user_text)
    token_ids.append(Int32(BOS_TOKEN_ID))
    append_encoded(tok, token_ids, "ai\n")
    if mode == "enabled":
        token_ids.append(Int32(MM_THINK_OPEN_TOKEN_ID))
    elif mode == "disabled":
        token_ids.append(Int32(MM_THINK_CLOSE_TOKEN_ID))
    return token_ids^


def print_help():
    print("commands:")
    print("  /system [text]     show the system prompt, or set it (keeps the chat)")
    print("  /thinking [mode]   thinking mode: on|off|adaptive (default adaptive)")
    print("  /greedy            reset the sampler to greedy (argmax)")
    print("  /temp <value>      set sampling temperature (>0, enables sampling)")
    print("  /min-p <value>     set min_p cutoff ([0, 1), enables sampling)")
    print("  /top-k <value>     set top_k window (0 disables, enables sampling)")
    print("  /seed <value>      set the sampling seed")
    print("  /sampler           show the active sampler settings")
    print("  /reset             clear the conversation context")
    print("  /rewind            undo the last turn (you + model)")
    print("  /retry             undo the last turn without clearing the screen")
    print("  /help              show this help")
    print("  /quit              exit")


def top_k_label(top_k: Int) -> String:
    return String("off") if top_k <= 0 else String(top_k)


def print_sampler(read p: SamplingParams):
    if p.greedy:
        print(t"  sampler: greedy (temp {p.temperature}, min_p {p.min_p}, "
              t"top_k {top_k_label(p.top_k)}, seed {p.seed}; "
              t"sampling knobs inactive while greedy)")
    else:
        print(t"  sampler: temp {p.temperature}, min_p {p.min_p}, "
              t"top_k {top_k_label(p.top_k)}, seed {p.seed}")


def render_chat(read history: List[ChatTurn]):
    print(CLEAR_SCREEN, end="", flush=True)
    print("minimax-m3 chat")
    for i in range(len(history)):
        print(YOU_PROMPT + history[i].user)
        print(MODEL_PROMPT + history[i].reply)


struct ChatState(Movable):
    var sampling: SamplingParams
    var system: String
    var thinking: String
    var history: List[ChatTurn]

    def __init__(out self):
        self.sampling = SamplingParams(
            Float32(1.0), Float32(0.0), 0, 0, MAXIMUM_SAMPLING_LOGITS, True)
        self.system = String(DEFAULT_SYSTEM)
        self.thinking = String("adaptive")
        self.history = List[ChatTurn]()


trait Command(Parsable):
    @staticmethod
    def keys() -> List[String]: ...

    def apply(self, mut st: ChatState) -> CmdResult: ...


struct QuitCmd(Command, Copyable, Movable):
    def __init__(out self):
        pass

    @staticmethod
    def keys() -> List[String]:
        return [String("/quit"), String("/exit"), String("/q")]

    def apply(self, mut st: ChatState) -> CmdResult:
        return CmdResult.QUIT


struct HelpCmd(Command, Copyable, Movable):
    def __init__(out self):
        pass

    @staticmethod
    def keys() -> List[String]:
        return [String("/help"), String("/h")]

    def apply(self, mut st: ChatState) -> CmdResult:
        print_help()
        return CmdResult.HANDLED


struct ResetCmd(Command, Copyable, Movable):
    def __init__(out self):
        pass

    @staticmethod
    def keys() -> List[String]:
        return [String("/reset")]

    def apply(self, mut st: ChatState) -> CmdResult:
        st.history = List[ChatTurn]()
        render_chat(st.history)
        print("  context cleared")
        return CmdResult.HANDLED


struct GreedyCmd(Command, Copyable, Movable):
    def __init__(out self):
        pass

    @staticmethod
    def keys() -> List[String]:
        return [String("/greedy")]

    def apply(self, mut st: ChatState) -> CmdResult:
        st.sampling = SamplingParams(
            Float32(1.0), Float32(0.0), 0, 0, MAXIMUM_SAMPLING_LOGITS, True)
        print_sampler(st.sampling)
        return CmdResult.HANDLED


struct SamplerCmd(Command, Copyable, Movable):
    def __init__(out self):
        pass

    @staticmethod
    def keys() -> List[String]:
        return [String("/sampler")]

    def apply(self, mut st: ChatState) -> CmdResult:
        print_sampler(st.sampling)
        return CmdResult.HANDLED


struct TempCmd(Command, Copyable, Movable):
    var temp: PositiveFloat

    def __init__(out self):
        self.temp = PositiveFloat()

    @staticmethod
    def keys() -> List[String]:
        return [String("/temp")]

    def apply(self, mut st: ChatState) -> CmdResult:
        st.sampling.temperature = self.temp.value
        st.sampling.greedy = False
        print_sampler(st.sampling)
        return CmdResult.HANDLED


struct MinPCmd(Command, Copyable, Movable):
    var min_p: UnitFloat

    def __init__(out self):
        self.min_p = UnitFloat()

    @staticmethod
    def keys() -> List[String]:
        return [String("/min-p"), String("/min_p")]

    def apply(self, mut st: ChatState) -> CmdResult:
        st.sampling.min_p = self.min_p.value
        st.sampling.greedy = False
        print_sampler(st.sampling)
        return CmdResult.HANDLED


struct TopKCmd(Command, Copyable, Movable):
    var k: BoundedInt[MAXIMUM_SAMPLING_LOGITS]

    def __init__(out self):
        self.k = BoundedInt[MAXIMUM_SAMPLING_LOGITS]()

    @staticmethod
    def keys() -> List[String]:
        return [String("/top-k"), String("/top_k")]

    def apply(self, mut st: ChatState) -> CmdResult:
        st.sampling.top_k = self.k.value
        if self.k.value > 0:
            st.sampling.greedy = False
        print_sampler(st.sampling)
        return CmdResult.HANDLED


struct SeedCmd(Command, Copyable, Movable):
    var seed: NonNegInt

    def __init__(out self):
        self.seed = NonNegInt()

    @staticmethod
    def keys() -> List[String]:
        return [String("/seed")]

    def apply(self, mut st: ChatState) -> CmdResult:
        st.sampling.seed = UInt64(self.seed.value)
        print_sampler(st.sampling)
        return CmdResult.HANDLED


struct RewindCmd(Command, Copyable, Movable):
    def __init__(out self):
        pass

    @staticmethod
    def keys() -> List[String]:
        return [String("/rewind"), String("/undo")]

    def apply(self, mut st: ChatState) -> CmdResult:
        if len(st.history) == 0:
            print("  nothing to rewind")
        else:
            _ = st.history.pop()
            render_chat(st.history)
            print(t"  rewound last turn ({len(st.history)} left)")
        return CmdResult.HANDLED


struct RetryCmd(Command, Copyable, Movable):
    def __init__(out self):
        pass

    @staticmethod
    def keys() -> List[String]:
        return [String("/retry"), String("/regen")]

    def apply(self, mut st: ChatState) -> CmdResult:
        if len(st.history) == 0:
            print("  nothing to retry")
        else:
            _ = st.history.pop()
            print(t"  rewound last turn ({len(st.history)} left)")
        return CmdResult.HANDLED


struct Registry:
    var quit: QuitCmd
    var help: HelpCmd
    var reset: ResetCmd
    var greedy: GreedyCmd
    var sampler: SamplerCmd
    var temp: TempCmd
    var min_p: MinPCmd
    var top_k: TopKCmd
    var seed: SeedCmd
    var rewind: RewindCmd
    var retry: RetryCmd


def has_key[C: Command](cmd: String) -> Bool:
    for k in C.keys():
        if k == cmd:
            return True
    return False


def dispatch[Reg: AnyType](read parts: List[String], mut st: ChatState) -> CmdResult:
    comptime r = reflect[Reg]
    comptime types = r.field_types()
    var result = CmdResult.PASS
    comptime for i in range(r.field_count()):
        comptime CT = types[i]
        comptime if conforms_to(CT, Command):
            if result == CmdResult.PASS and has_key[CT](parts[0]):
                var c = fill[CT](parts)
                if c:
                    result = c.value().apply(st)
                else:
                    result = CmdResult.HANDLED
    return result


def park[T: SleepableThreadPool, //](mut pool: T):
    pool.sleep()


def unpark[T: SleepableThreadPool, //](mut pool: T):
    pool.wake()


def run[
    P: BurstThreadPool, //,
](
    topo: NumaTopology,
    var pools: List[P],
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
):
    var model_opt = MinimaxM3[
        max_seq_len=MAX_CONTEXT, batching_seq_len=MAX_CONTEXT, Pool=P,
    ].load(Path(MODEL_DIR), topo, pools^)
    if not model_opt:
        print("model load failed")
        return
    var model = model_opt.take()
    print(t"loaded (degree {model.degree})")

    var state = ChatState()
    var sched = ContinuousBatchScheduler[PAGE_LEN](
        model.batch_geometry(), STEP_BUDGET, stop_tokens())

    print()
    print_help()
    print_sampler(state.sampling)

    var console = TerminalReader()

    while True:
        for i in range(len(model.pools)):
            park(model.pools[i])
        var line_opt = console.read_message(YOU_PROMPT)
        if not line_opt:
            print()
            break
        for i in range(len(model.pools)):
            unpark(model.pools[i])
        var s = String(line_opt.value().strip())
        if s.byte_length() == 0:
            continue

        if s.startswith("/"):
            var parts = s.split()
            var args = List[String]()
            for p in parts:
                args.append(String(p))
            var cmd = args[0]
            var verdict = dispatch[Registry](args, state)
            if verdict == CmdResult.QUIT:
                break
            elif verdict == CmdResult.HANDLED:
                continue
            if cmd == "/system":
                if len(args) < 2:
                    if state.system.byte_length() == 0:
                        print("  (no system prompt)")
                    else:
                        print(t"  system: {state.system}")
                else:
                    state.system = String(s[byte=cmd.byte_length():].strip())
                    print(t"  system prompt set ({state.system.byte_length()} "
                          t"chars); conversation kept")
            elif cmd == "/thinking" or cmd == "/think":
                if len(args) < 2:
                    print(t"  thinking mode: {state.thinking}")
                else:
                    var m = args[1]
                    if m == "on" or m == "enabled":
                        state.thinking = String("enabled")
                    elif m == "off" or m == "disabled":
                        state.thinking = String("disabled")
                    elif m == "adaptive" or m == "auto":
                        state.thinking = String("adaptive")
                    else:
                        print(t"  unknown mode '{m}' (use on|off|adaptive)")
                    print(t"  thinking mode: {state.thinking}")
            else:
                print(t"  unknown command {cmd} (try /help)")
            continue

        var input = build_input(tok, state.system, state.thinking, state.history, s)
        var reply_budget = MAX_CONTEXT - len(input)
        if reply_budget < 1:
            print("  (context full — use /reset to clear)")
            continue
        var rid_opt = sched.submit(input^, state.sampling, reply_budget)
        if not rid_opt:
            print("  (context full — use /reset to clear)")
            continue
        var rid = rid_opt.value()

        var detok = StreamDetokenizer()
        var consumed = 0
        var guard = 0
        var stalled = False
        var canceled = False
        var in_thought = state.thinking == "enabled"
        var reply = String("")
        print(MODEL_PROMPT, end="", flush=True)
        if in_thought:
            print(DIM, end="", flush=True)
        var watch = CancelWatch()
        while not sched.requests[rid].done:
            if watch.triggered():
                var token = sched.cancel_token(rid)
                token.cancel()
                _ = sched.step(model)
                canceled = True
                break
            guard += 1
            if guard > 8 * reply_budget:
                print("\n  (generation stalled)")
                stalled = True
                break
            if sched.step(model) == 0:
                print("\n  (scheduler stalled)")
                stalled = True
                break
            ref cur_gen = sched.requests[rid].generated
            while consumed < len(cur_gen):
                var tid = cur_gen[consumed]
                consumed += 1
                if sched.is_stop_token(tid):
                    continue
                if is_think_open(tid):
                    in_thought = True
                    print(DIM, end="", flush=True)
                    continue
                if is_think_close(tid):
                    in_thought = False
                    print(RESET, end="", flush=True)
                    continue
                var piece = detok.push(tok, tid)
                if piece.byte_length() > 0:
                    print(piece, end="", flush=True)
                    if not in_thought:
                        reply += piece

        if canceled:
            print(RESET + "\n  (stopped)")
            _ = sched.retire(rid)
            _ = watch^
            continue

        if stalled:
            _ = sched.retire(rid)
            _ = watch^
            continue

        var tail = detok.flush()
        if tail.byte_length() > 0:
            print(tail, end="", flush=True)
            if not in_thought:
                reply += tail
        if in_thought:
            print(RESET, end="", flush=True)
        print()

        _ = watch^
        _ = sched.retire(rid)
        state.history.append(ChatTurn(s.copy(), reply^))

    print("bye")


def main():
    print("minimax-m3 chat")
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print(t"failed to load tokenizer from {TOKENIZER_PATH}")
        return
    var tok = tok_opt.take()

    var topo = NumaTopology()
    var nodes = topo.num_nodes()
    print(t"{nodes} NUMA nodes")

    @parameter
    def dispatch_tp[
        P: BurstThreadPool, //,
    ](var selected_pools: List[P]):
        run(topo, selected_pools^, tok)

    with_topological_rank_dispatch[
        dispatch=dispatch_tp,
    ](
        topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
