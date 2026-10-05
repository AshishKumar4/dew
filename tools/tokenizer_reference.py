#!/usr/bin/env python3
"""Write tests/fixtures/tokenizers/reference.json: what transformers'
`AutoTokenizer` makes of fixed texts, for the tokenizers `HFTokenizer`
wraps.

Four tokenizers: GPT-2's published byte-level BPE (tokenizers/gpt2, the
Hub's files at openai-community/gpt2@607a30d7), T5's published Unigram,
which closes a sequence with `</s>` (tokenizers/t5-small, the Hub's files
at google-t5/t5-small@df1b051c), the tiny Qwen 3.8 one
(hf/qwen38-dense-tiny: byte-level BPE with NFC normalization and 33 added
chat tokens) and the llama.cpp fixtures' tiny SentencePiece-style one
(llama_cpp/tokenizer: byte fallback, `<s>` and `</s>`). Each
text is encoded with and without special tokens and its ids decoded by
`decode`, transformers 5.16.1's defaults throughout. The texts cover
whitespace runs, accented, CJK, Korean, Arabic and combining text, an
emoji ZWJ sequence, control characters, and each tokenizer's own special
and added tokens written out as text; the record also keeps the
vocabulary length and the eos and bos ids.

    PYTHONPATH=src python tools/tokenizer_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "tokenizers" / "reference.json"
TOKENIZERS = {"gpt2": "tests/fixtures/tokenizers/gpt2", "t5": "tests/fixtures/tokenizers/t5-small",
              "qwen38": "tests/fixtures/hf/qwen38-dense-tiny",
              "byte_fallback": "tests/fixtures/llama_cpp/tokenizer"}
TEXTS = ["hello world", "  spaced   out\n\tand tabbed  ", "ünïcödé — π≈3.14159", "日本語のテキスト",
         "한국어", "مرحبا بالعالم", "e\u0301 and \u00e9", "emoji 🚀🔥 and 👩🏽\u200d💻",
         "\x00 control \x7f bytes", ""]


def main() -> None:
    import transformers
    from transformers import AutoTokenizer

    if transformers.__version__ != "5.16.1":
        raise SystemExit(f"the fixture pins transformers 5.16.1, got {transformers.__version__}")
    record = {"transformers": transformers.__version__, "tokenizers": {}}
    for name, directory in TOKENIZERS.items():
        tokenizer = AutoTokenizer.from_pretrained(ROOT / directory)
        specials = list(dict.fromkeys([*tokenizer.all_special_tokens, *tokenizer.get_added_vocab()]))
        texts = [*TEXTS, "".join(specials[:8]), f"say {specials[0]} then {specials[-1]} done"]
        cases = []
        for text in texts:
            ids = tokenizer.encode(text)
            plain = tokenizer.encode(text, add_special_tokens=False)
            cases.append({"text": text, "ids": ids, "plain": plain, "decoded": tokenizer.decode(ids),
                          "plain_decoded": tokenizer.decode(plain)})
        record["tokenizers"][name] = {"directory": directory, "length": len(tokenizer),
                                      "eos": tokenizer.eos_token_id, "bos": tokenizer.bos_token_id,
                                      "cases": cases}
    FIXTURE.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    print(f"{FIXTURE}: " + ", ".join(f"{name} {len(entry['cases'])}" for name, entry in
                                     record["tokenizers"].items()))


if __name__ == "__main__":
    main()
