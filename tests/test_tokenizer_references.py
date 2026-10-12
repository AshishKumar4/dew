"""The byte tokenizer against UTF-8's own definitions, and `HFTokenizer`
against transformers' `AutoTokenizer` on committed tokenizers
(`tools/tokenizer_reference.py`)."""

import json
from pathlib import Path

import pytest

from dew.data import ByteTokenizer, HFTokenizer

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "tests" / "fixtures" / "tokenizers" / "reference.json").read_text())


@pytest.mark.parametrize("text, encoded", [
    ("A\u2262\u0391.", "41 E2 89 A2 CE 91 2E"),
    ("\uD55C\uAD6D\uC5B4", "ED 95 9C EA B5 AD EC 96 B4"),
    ("\u65E5\u672C\u8A9E", "E6 97 A5 E6 9C AC E8 AA 9E"),
    ("\uFEFF\U000233B4", "EF BB BF F0 A3 8E B4"),
])
def test_the_byte_tokenizer_encodes_rfc_3629s_examples(text, encoded):
    """RFC 3629 section 7's four examples, ids being the UTF-8 bytes, and
    each decodes back to its text (the BOM included)."""
    tokenizer = ByteTokenizer()
    assert tokenizer.encode(text) == list(bytes.fromhex(encoded))
    assert tokenizer.decode(tokenizer.encode(text)) == text
    for sample in ("hello, world", "\u00fcn\u00efc\u00f6d\u00e9 \u2014 \u03c0\u22483.14159",
                   "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8",
                   "emoji \U0001f680\U0001f525 and \x00 control bytes\n\t"):
        assert tokenizer.decode(tokenizer.encode(sample)) == sample


@pytest.mark.parametrize("name", sorted(REFERENCE["tokenizers"]))
def test_hf_tokenizer_is_autotokenizer(name):
    """transformers 5.16.1's `AutoTokenizer` on GPT-2's published BPE, T5's
    published Unigram (which appends `</s>`), a byte-level BPE with NFC
    normalization and added chat tokens, and a byte fallback one: the ids
    with and without special tokens, their decodes, the vocabulary length
    and the eos and bos ids are its, for text in several scripts, combining
    marks, a ZWJ emoji, control characters and each tokenizer's own special
    and added tokens written out."""
    entry = REFERENCE["tokenizers"][name]
    tokenizer = HFTokenizer(str(ROOT / entry["directory"]), local_files_only=True)
    expected = (entry["length"], entry["eos"], entry["bos"])
    assert (tokenizer.vocab_size, tokenizer.eos_id, tokenizer.bos_id) == expected
    for case in entry["cases"]:
        assert tokenizer.encode(case["text"]) == case["ids"], case["text"]
        assert tokenizer.encode(case["text"], add_special_tokens=False) == case["plain"], case["text"]
        assert tokenizer.decode(case["ids"]) == case["decoded"], case["text"]
        assert tokenizer.decode(case["plain"]) == case["plain_decoded"], case["text"]
    if name == "gpt2":
        assert tokenizer.decode(tokenizer.encode("hello world")) == "hello world"
