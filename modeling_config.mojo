from tokenizer import BPETokenizer, AutoPreTokenizer, AutoByteTransform
from modeling.gemma_4_moe_bq import Gemma4
from prompt_formatting import Gemma4Format


comptime Model = Gemma4
comptime ChatFormat = Gemma4Format
comptime TOKENIZER_PATH = "checkpoints/gemma-4-26B-A4B-it-abliterated/tokenizer.json"
comptime MODEL_DIR = "checkpoints/gemma-4-26B-A4B-it-abliterated-bq"

comptime BOS_TOKEN_ID = 2
comptime EOS_TOKEN_ID = 1
comptime TURN_START_TOKEN_ID = 105
comptime TURN_END_TOKEN_ID = 106
comptime TOOL_RESPONSE_TOKEN_ID = 50


def append_encoded(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    mut token_ids: List[Int32],
    text: String,
):
    var encoded = tok.encode(text)
    for i in range(len(encoded)):
        token_ids.append(Int32(encoded[i]))


def format_prompt(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    prompt: String,
) -> List[Int32]:
    var token_ids = List[Int32]()
    token_ids.append(Int32(BOS_TOKEN_ID))
    token_ids.append(Int32(TURN_START_TOKEN_ID))
    append_encoded(tok, token_ids, "user\n" + prompt)
    token_ids.append(Int32(TURN_END_TOKEN_ID))
    append_encoded(tok, token_ids, "\n")
    token_ids.append(Int32(TURN_START_TOKEN_ID))
    append_encoded(tok, token_ids, "model\n")
    return token_ids^


def stop_tokens() -> List[Int32]:
    var ids = List[Int32]()
    ids.append(Int32(EOS_TOKEN_ID))
    ids.append(Int32(TURN_END_TOKEN_ID))
    ids.append(Int32(TOOL_RESPONSE_TOKEN_ID))
    return ids^
