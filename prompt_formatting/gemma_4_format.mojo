from .format_trait import (
    PromptFormat, ChatTokenizer, TurnRecord, append_encoded,
)


comptime THINK_TOKEN_ID: Int32 = 98
comptime CHANNEL_OPEN_TOKEN_ID: Int32 = 100
comptime CHANNEL_CLOSE_TOKEN_ID: Int32 = 101


struct UserTurn[origin: ImmutOrigin](Writable):
    var content: Pointer[String, Self.origin]

    def __init__(out self: UserTurn[origin_of(content)], ref content: String):
        self.content = Pointer(to=content)

    def write_to(self, mut w: Some[Writer]):
        t"<|turn>user\n{self.content[]}<turn|>\n".write_to(w)


struct SystemTurn[origin: ImmutOrigin](Writable):
    var content: Pointer[String, Self.origin]
    var thinking: Bool

    def __init__(
        out self: SystemTurn[origin_of(content)],
        ref content: String,
        thinking: Bool,
    ):
        self.content = Pointer(to=content)
        self.thinking = thinking

    def write_to(self, mut w: Some[Writer]):
        if self.thinking:
            t"<|turn>system\n<|think|>\n{self.content[]}<turn|>\n".write_to(w)
        else:
            t"<|turn>system\n{self.content[]}<turn|>\n".write_to(w)


struct Gemma4Format(PromptFormat):
    comptime WEAVES = False
    comptime CHANNEL_OPEN = CHANNEL_OPEN_TOKEN_ID
    comptime CHANNEL_CLOSE = CHANNEL_CLOSE_TOKEN_ID

    def __init__(out self):
        pass

    def push_prefix(
        self,
        mut tok: ChatTokenizer,
        mut history: List[Int32],
        read system: String,
        global_thinking: Bool,
    ):
        append_encoded(tok, history, String("<bos>"))
        if global_thinking or system.byte_length() > 0:
            append_encoded(tok, history, SystemTurn(system, global_thinking))

    def push_user(
        self,
        mut tok: ChatTokenizer,
        mut history: List[Int32],
        read message: String,
    ):
        append_encoded(tok, history, UserTurn(message))

    def push_model_open(
        self, mut tok: ChatTokenizer, mut history: List[Int32]
    ):
        append_encoded(tok, history, String("<|turn>model\n"))

    def push_thought_stub(
        self, mut tok: ChatTokenizer, mut history: List[Int32]
    ):
        append_encoded(tok, history, String("<|channel>thought\n<channel|>"))
