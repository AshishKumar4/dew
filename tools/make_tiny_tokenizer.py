#!/usr/bin/env python3
"""Write the tiny chat tokenizers under tests/fixtures/tokenizers/.

Each fixture is a byte-level BPE tokenizer with a real chat template, so the
SFT tests render the same ids a Hugging Face tokenizer renders.

`tiny-chat` holds the BPE model with the Qwen special tokens and the chat
template TRL trains Qwen3 with (`chat_template.jinja`, copied from TRL 1.12
`trl/chat_templates/qwen3_training.jinja`, Apache-2.0) and the eos token.
Its BPE trainer sees the template's literal words and a few sentences of
ordinary text, so the fixture conversation tokenizes into word pieces
rather than bytes; characters outside that corpus read as `<unk>`, which
the TRL parity fixture (tests/fixtures/rl/chat.npz) was written against.

`tiny-tools` holds the ChatML template with tool-call ids
(`tests/fixtures/tokenizers/tiny-tools/chat_template.jinja`) over a BPE
model trained with the whole byte alphabet, so JSON arguments, ids and
names tokenize to distinct ids and the structured-chat tests can tell one
argument from another.

Rerunning this script must reproduce the committed files byte for byte; it
trains on a fixed corpus in a fixed order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOKENIZERS = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tokenizers"

CORPUS = [
     # The template's literal words, so headers and markers merge as pieces.
    "<|im_start|>system <|im_end|> user assistant think tool_response tool_call",
    "You are a helpful assistant that answers questions.",
    "What is two plus two? Two plus two is four.",
    "And three plus three? Three plus three is six.",
    "The quick brown fox jumps over 123 lazy dogs. Punctuation: commas, periods!",
    "Numbers count: 0 1 2 3 4 5 6 7 8 9, andauf mixed CASE Words.",
    "A second sentence exercises merges across word boundaries and newlines.\n"
    "Newlines start new pre-tokens, as every assistant header does.",
    # The template's generated spans, so they merge into clean pieces.
    "<think>\n\n</think>\n\n",
    "<tool_response>\n4\n</tool_response>",
]

TOOLS_CORPUS = CORPUS + [
    # JSON punctuation and the tool template's markers, so calls tokenize
    # into pieces rather than single bytes.
    '{"name": "get_weather", "arguments": {"city": "Paris"}}',
    '<tool_call id="call_1">get_weather {"city": "Athens"}</tool_call>',
    '<tool_response id="call_2" name="get_weather">22</tool_response>',
    "Tools: Weather in Paris and Athens? Thanks, which is warmer?",
]

VOCAB_SIZE = 384
"""256 bytes plus the special tokens plus room for merges. A smaller table
cannot hold every byte, and bytes without an entry decode as unknowns."""
EOS = "<|endoftext|>"
SPECIAL = [EOS, "<|im_start|>", "<|im_end|>", "<unk>"]


def build(corpus: list[str], every_byte: bool) -> str:
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE, min_frequency=1, special_tokens=SPECIAL,
        initial_alphabet=ByteLevel.alphabet() if every_byte else [],
        show_progress=False,
    )
    tokenizer.train_from_iterator(iter(corpus), trainer=trainer)
    for token in SPECIAL:
        token_id = tokenizer.token_to_id(token)
        assert token_id is not None, f"special token {token} missing after training"
    return tokenizer.to_str()


def write(out: Path, corpus: list[str], every_byte: bool, embed_template: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokenizer.json").write_text(build(corpus, every_byte) + "\n")
    config = {
        "eos_token": EOS,
        "unk_token": "<unk>",
        "clean_up_tokenization_spaces": False,
        "add_bos_token": False,
        "add_eos_token": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    if embed_template:
        # The TRL parity fixture was written against the template embedded
        # here; chat_template.jinja beside it is the same text.
        config = {"chat_template": (out / "chat_template.jinja").read_text(), **config}
    (out / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {out / 'tokenizer.json'} and {out / 'tokenizer_config.json'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=TOKENIZERS)
    out = parser.parse_args(argv).out
    write(out / "tiny-chat", CORPUS, every_byte=False, embed_template=True)
    write(out / "tiny-tools", TOOLS_CORPUS, every_byte=True, embed_template=False)


if __name__ == "__main__":
    main()
