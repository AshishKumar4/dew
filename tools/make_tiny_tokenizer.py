#!/usr/bin/env python3
"""Write the tiny chat tokenizer under tests/fixtures/tokenizers/tiny-chat/.

The fixture is a byte-level BPE tokenizer with a real chat template, so the
SFT parity test renders the same ids TRL renders: `tokenizer.json` holds the
BPE model with the Qwen special tokens, `tokenizer_config.json` holds the
chat template TRL trains Qwen3 with (`chat_template.jinja`, copied from TRL
1.12 `trl/chat_templates/qwen3_training.jinja`, Apache-2.0) and the eos token.

The BPE trainer sees the template's literal words and a few sentences of
ordinary text, so the fixture conversation tokenizes into word pieces rather
than bytes. Byte fallback covers the rest, so any text tokenizes. Rerunning
this script must reproduce the committed files byte for byte; it trains on a
fixed corpus in a fixed order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tokenizers" / "tiny-chat"

TEMPLATE = (FIXTURES / "chat_template.jinja").read_text()

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

VOCAB_SIZE = 384
"""256 bytes plus the special tokens plus room for merges. A smaller table
cannot hold every byte, and bytes without an entry decode as unknowns."""
EOS = "<|endoftext|>"
SPECIAL = [EOS, "<|im_start|>", "<|im_end|>", "<unk>"]


def build() -> str:
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
        show_progress=False,
    )
    tokenizer.train_from_iterator(iter(CORPUS), trainer=trainer)
    for token in SPECIAL:
        token_id = tokenizer.token_to_id(token)
        assert token_id is not None, f"special token {token} missing after training"
    return tokenizer.to_str()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURES)
    out = parser.parse_args(argv).out
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokenizer.json").write_text(build() + "\n")
    config = {
        "chat_template": TEMPLATE,
        "eos_token": EOS,
        "unk_token": "<unk>",
        "clean_up_tokenization_spaces": False,
        "add_bos_token": False,
        "add_eos_token": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    (out / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {out / 'tokenizer.json'} and {out / 'tokenizer_config.json'}")


if __name__ == "__main__":
    main()
