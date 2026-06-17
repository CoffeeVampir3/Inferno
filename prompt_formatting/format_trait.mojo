from tokenizer import BPETokenizer, AutoPreTokenizer, AutoByteTransform


comptime ChatTokenizer = BPETokenizer[AutoPreTokenizer, AutoByteTransform]


def append_encoded[W: Writable, //](
    mut tok: ChatTokenizer,
    mut history: List[Int32],
    read piece: W,
):
    var enc = tok.encode(String(piece))
    for i in range(len(enc)):
        history.append(Int32(enc[i]))


struct TurnRecord(Copyable, Movable):
    var user_tokens: List[Int32]
    var thought_tokens: List[Int32]
    var response_tokens: List[Int32]

    def __init__(
        out self,
        var user_tokens: List[Int32],
        var thought_tokens: List[Int32],
        var response_tokens: List[Int32],
    ):
        self.user_tokens = user_tokens^
        self.thought_tokens = thought_tokens^
        self.response_tokens = response_tokens^


trait PromptFormat(Copyable, Movable):
    comptime WEAVES: Bool
    comptime CHANNEL_OPEN: Int32
    comptime CHANNEL_CLOSE: Int32

    def push_prefix(
        self,
        mut tok: ChatTokenizer,
        mut history: List[Int32],
        read system: String,
        global_thinking: Bool,
    ):
        ...

    def push_user(
        self,
        mut tok: ChatTokenizer,
        mut history: List[Int32],
        read message: String,
    ):
        ...

    def push_model_open(
        self, mut tok: ChatTokenizer, mut history: List[Int32]
    ):
        ...

    def push_thought_stub(
        self, mut tok: ChatTokenizer, mut history: List[Int32]
    ):
        ...

    def push_separator(
        self, mut tok: ChatTokenizer, mut history: List[Int32]
    ):
        append_encoded(tok, history, String("\n"))

    def push_past_turn(
        self,
        mut tok: ChatTokenizer,
        mut history: List[Int32],
        read turn: TurnRecord,
    ):
        for i in range(len(turn.user_tokens)):
            history.append(turn.user_tokens[i])
        self.push_model_open(tok, history)
        if Self.WEAVES:
            for i in range(len(turn.thought_tokens)):
                history.append(turn.thought_tokens[i])
        for i in range(len(turn.response_tokens)):
            history.append(turn.response_tokens[i])
        self.push_separator(tok, history)

    def build_input(
        self,
        mut tok: ChatTokenizer,
        read system: String,
        global_thinking: Bool,
        read turns: List[TurnRecord],
        read message: String,
        turn_thinking: Bool,
    ) -> List[Int32]:
        var history = List[Int32]()
        self.push_prefix(tok, history, system, global_thinking)
        for i in range(len(turns)):
            self.push_past_turn(tok, history, turns[i])
        self.push_user(tok, history, message)
        self.push_model_open(tok, history)
        if not (global_thinking and turn_thinking):
            self.push_thought_stub(tok, history)
        return history^

    def record_turn(
        self,
        mut tok: ChatTokenizer,
        read message: String,
        read gen: List[Int32],
    ) -> TurnRecord:
        var user_tokens = List[Int32]()
        self.push_user(tok, user_tokens, message)
        var thought_tokens = List[Int32]()
        var response_tokens = List[Int32]()
        var in_thought = False
        var done_thought = False
        for i in range(len(gen)):
            var tid = gen[i]
            if not done_thought and not in_thought and tid == Self.CHANNEL_OPEN:
                in_thought = True
                thought_tokens.append(tid)
            elif in_thought:
                thought_tokens.append(tid)
                if tid == Self.CHANNEL_CLOSE:
                    in_thought = False
                    done_thought = True
            else:
                response_tokens.append(tid)
        return TurnRecord(
            user_tokens^, thought_tokens^, response_tokens^
        )
