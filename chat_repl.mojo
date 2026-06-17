from std.pathlib import Path
from std.reflection import reflect

from numa import NumaTopology
from terminal import (
    TerminalReader, CmdResult, Parsable, fill,
    PositiveFloat, UnitFloat, BoundedInt, NonNegInt, Toggle,
)
from threading.threading_traits import BurstThreadPool, SleepableThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from tokenizer import (
    load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform,
    StreamDetokenizer,
)
from modeling_config import (
    Model, ChatFormat, TOKENIZER_PATH, MODEL_DIR, stop_tokens,
)
from prompt_formatting import (
    TurnRecord, CHANNEL_OPEN_TOKEN_ID, CHANNEL_CLOSE_TOKEN_ID,
)
from modeling.gemma4_common import Gemma4BaseConfig
from inspectable_toolkit.slider_pack import load_pack, SliderBank
from kernels.flash_sample import SamplingParams
from continuous_batching.schedule import MAXIMUM_SAMPLING_LOGITS
from continuous_batching.scheduler import ContinuousBatchScheduler


comptime PACK_PATH = "sliders/ocean.json"
comptime DEFAULT_SYSTEM = "You are Gemma, a helpful assistant."
comptime STEP_BUDGET = Gemma4BaseConfig.SLIDING_WINDOW
comptime MAX_CONTEXT = 65536
comptime CLEAR_SCREEN = "\x1b[2J\x1b[3J\x1b[H"
comptime YOU_PROMPT = "\n\n\x1b[32myou> \x1b[0m"
comptime MODEL_PROMPT = "\x1b[38;5;211mmodel> \x1b[0m"
comptime CYAN = "\x1b[36m"
comptime DIM = "\x1b[90m"
comptime RESET = "\x1b[0m"


def on_off(flag: Bool) -> String:
    return String("on") if flag else String("off")


@fieldwise_init
struct ChatTurn(Copyable, Movable):
    var user: String
    var reply: String


def print_help():
    print("commands:")
    print("  /<trait> <value>   set a trait; [-1, 1] is the safe range, "
          "|value| > 1 overdrives past the corridor (0 = off)")
    print("  /sliders           list traits and current doses")
    print("  /system [text]     show the system prompt, or set it (keeps the chat)")
    print("  /greedy            reset the sampler to greedy (argmax)")
    print("  /temp <value>      set sampling temperature (>0, enables sampling)")
    print("  /min-p <value>     set min_p cutoff ([0, 1), enables sampling)")
    print("  /top-k <value>     set top_k window (0 disables, enables sampling)")
    print("  /seed <value>      set the sampling seed")
    print("  /sampler           show the active sampler settings")
    print("  /global-thinking [on|off]  thinking ability for the conversation")
    print("  /thinking [on|off] whether turns actually reason")
    print("  /reset             clear the conversation context")
    print("  /rewind            undo the last turn (you + model)")
    print("  /retry             undo the last turn without clearing the screen")
    print("  /help              show this help")
    print("  /quit              exit")


def print_sliders(read bank: SliderBank):
    if bank.count() == 0:
        print("  (no sliders loaded)")
        return
    for i in range(bank.count()):
        var cfg = bank.configs[i]
        print(t"  {bank.names[i]}: alpha {bank.alphas[i]} "
              t"(layer {cfg.layer}, corridor [{cfg.alpha_min}, "
              t"{cfg.alpha_max}])")


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


def render_chat(read transcript: List[ChatTurn]):
    print(CLEAR_SCREEN, end="", flush=True)
    print("gemma personality chat")
    for i in range(len(transcript)):
        print(YOU_PROMPT + transcript[i].user)
        print(MODEL_PROMPT + transcript[i].reply)


struct ChatState(Movable):
    var sampling: SamplingParams
    var system: String
    var global_thinking: Bool
    var turn_thinking: Bool
    var turns: List[TurnRecord]
    var transcript: List[ChatTurn]

    def __init__(out self):
        self.sampling = SamplingParams(
            Float32(1.0), Float32(0.0), 0, 0, MAXIMUM_SAMPLING_LOGITS, True)
        self.system = String(DEFAULT_SYSTEM)
        self.global_thinking = True
        self.turn_thinking = True
        self.turns = List[TurnRecord]()
        self.transcript = List[ChatTurn]()


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
        st.turns = List[TurnRecord]()
        st.transcript = List[ChatTurn]()
        render_chat(st.transcript)
        print("  context cleared")
        return CmdResult.HANDLED


struct GlobalThinkingCmd(Command, Copyable, Movable):
    var flag: Toggle

    def __init__(out self):
        self.flag = Toggle()

    @staticmethod
    def keys() -> List[String]:
        return [String("/global-thinking"), String("/global")]

    def apply(self, mut st: ChatState) -> CmdResult:
        st.global_thinking = self.flag.resolve(st.global_thinking)
        print(t"  thinking ability: {on_off(st.global_thinking)}; "
              t"conversation kept")
        return CmdResult.HANDLED


struct ThinkingCmd(Command, Copyable, Movable):
    var flag: Toggle

    def __init__(out self):
        self.flag = Toggle()

    @staticmethod
    def keys() -> List[String]:
        return [String("/thinking"), String("/think")]

    def apply(self, mut st: ChatState) -> CmdResult:
        st.turn_thinking = self.flag.resolve(st.turn_thinking)
        print(t"  thinking mode: {on_off(st.turn_thinking)}")
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
        if len(st.turns) == 0:
            print("  nothing to rewind")
        else:
            _ = st.turns.pop()
            _ = st.transcript.pop()
            render_chat(st.transcript)
            print(t"  rewound last turn ({len(st.turns)} left)")
        return CmdResult.HANDLED


struct RetryCmd(Command, Copyable, Movable):
    def __init__(out self):
        pass

    @staticmethod
    def keys() -> List[String]:
        return [String("/retry"), String("/regen")]

    def apply(self, mut st: ChatState) -> CmdResult:
        if len(st.turns) == 0:
            print("  nothing to retry")
        else:
            _ = st.turns.pop()
            _ = st.transcript.pop()
            print(t"  rewound last turn ({len(st.turns)} left)")
        return CmdResult.HANDLED


struct Registry:
    var quit: QuitCmd
    var help: HelpCmd
    var reset: ResetCmd
    var global_thinking: GlobalThinkingCmd
    var thinking: ThinkingCmd
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
    var model_opt = Model[steer_vectors=16, max_seq_len=MAX_CONTEXT, batching_seq_len=MAX_CONTEXT, Pool=P].load(Path(MODEL_DIR), topo, pools^)
    if not model_opt:
        print("model load failed")
        return
    var model = model_opt.take()
    print(t"loaded (degree {model.degree})")

    var bank_opt = load_pack(model, PACK_PATH)
    var bank: SliderBank
    if bank_opt:
        bank = bank_opt.take()
        print(t"loaded {bank.count()} slider(s) from {PACK_PATH}")
    else:
        bank = SliderBank()
        print(t"no pack at {PACK_PATH} (chat unsteered)")
    bank.apply(model)

    var state = ChatState()
    var sched = ContinuousBatchScheduler[
        Model[steer_vectors=16, max_seq_len=MAX_CONTEXT, batching_seq_len=MAX_CONTEXT, Pool=P].POSITIONS_PER_PAGE,
    ](model.batch_geometry(), STEP_BUDGET, stop_tokens())

    print()
    print_help()
    print_sliders(bank)
    print_sampler(state.sampling)

    var fmt = ChatFormat()
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
            elif cmd == "/sliders" or cmd == "/list":
                print_sliders(bank)
            else:
                var matched = -1
                for i in range(bank.count()):
                    if cmd == String("/") + bank.names[i]:
                        matched = i
                if matched < 0:
                    print(t"  unknown command {cmd} (try /help)")
                elif len(args) < 2:
                    print("  usage: /<trait> <value>  ([-1, 1] safe, "
                          "beyond overdrives)")
                else:
                    var val = Float32(0)
                    var bad = False
                    try:
                        val = Float32(atof(args[1]))
                    except:
                        bad = True
                    if bad:
                        print(t"  '{args[1]}' is not a number")
                    else:
                        bank.set_position(matched, val)
                        bank.apply(model)
                        print(t"{CYAN}  {bank.names[matched]} -> position {val}, "
                              t"alpha {bank.alphas[matched]} at layer "
                              t"{bank.configs[matched].layer}{RESET}")
            continue

        var input = fmt.build_input(
            tok, state.system, state.global_thinking, state.turns, s,
            state.turn_thinking)
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
        var in_thought = False
        var reply = String("")
        print(MODEL_PROMPT, end="", flush=True)
        while not sched.requests[rid].done:
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
                if tid == CHANNEL_OPEN_TOKEN_ID:
                    in_thought = True
                    print(DIM, end="", flush=True)
                    continue
                if tid == CHANNEL_CLOSE_TOKEN_ID:
                    in_thought = False
                    print(RESET, end="", flush=True)
                    continue
                var piece = detok.push(tok, tid)
                if piece.byte_length() > 0:
                    print(piece, end="", flush=True)
                    if not in_thought:
                        reply += piece

        if stalled:
            _ = sched.retire(rid)
            continue

        var tail = detok.flush()
        if tail.byte_length() > 0:
            print(tail, end="", flush=True)
            if not in_thought:
                reply += tail
        if in_thought:
            print(RESET, end="", flush=True)
        print()

        var record = fmt.record_turn(tok, s, sched.requests[rid].generated)
        state.turns.append(record^)
        _ = sched.retire(rid)
        state.transcript.append(ChatTurn(s.copy(), reply^))

    model.steer.disarm()
    print("bye")


def main():
    print("gemma personality chat")
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
