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


@pytest.mark.parametrize("encoded, decoded", [
    ("C0 AF E0 80 BF F0 81 82 41", "FFFD " * 8 + "0041"),
    ("ED A0 80 ED BF BF ED AF 41", "FFFD " * 8 + "0041"),
    ("F4 91 92 93 FF 41 80 BF 42", "FFFD FFFD FFFD FFFD FFFD 0041 FFFD FFFD 0042"),
    ("E1 80 E2 F0 91 92 F1 BF 41", "FFFD FFFD FFFD FFFD 0041"),
], ids=["table 3-8", "table 3-9", "table 3-10", "table 3-11"])
def test_the_byte_tokenizer_replaces_maximal_subparts(encoded, decoded):
    """Generated ids are any bytes, so decoding replaces the ill-formed
    ones: Unicode 15.0's Tables 3-8 to 3-11 (section 3.9, U+FFFD
    substitution of maximal subparts), the practice the W3C encoding
    standard follows. Non-shortest forms, surrogates and out-of-range
    sequences take one U+FFFD per byte; a truncated sequence that was
    well-formed so far takes one in all."""
    assert ByteTokenizer().decode(list(bytes.fromhex(encoded))) == "".join(
        chr(int(code, 16)) for code in decoded.split())


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
