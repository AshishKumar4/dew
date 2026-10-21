"""The binary schema shared by distinct token-corpus generators."""

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def write_token_meta(directory: Path, train_tokens: int, val_tokens: int = 0, *, dtype: str = "uint16",
                     tokenizer: str = "byte", vocab_size: int = 256, eos_id: int | None = None) -> None:
    """A token store's `meta.json` in `directory`, which every token store, binary or records,
    carries; no `eos_id` where its documents state no end."""
    meta: dict[str, object] = {"tokenizer": tokenizer, "vocab_size": vocab_size, "dtype": dtype,
                               "train_tokens": train_tokens, "val_tokens": val_tokens}
    if eos_id is not None:
        meta["eos_id"] = eos_id
    (directory / "meta.json").write_text(json.dumps(meta))


def write_token_corpus(directory: Path, train: np.ndarray, val: np.ndarray | None = None,
                       *, tokenizer: str = "byte", vocab_size: int = 256,
                       eos_id: int | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train.bin").write_bytes(train.tobytes())
    if val is not None:
        assert train.dtype == val.dtype
        (directory / "val.bin").write_bytes(val.tobytes())
    write_token_meta(directory, len(train), 0 if val is None else len(val), dtype=train.dtype.name,
                     tokenizer=tokenizer, vocab_size=vocab_size, eos_id=eos_id)
    return directory


def encoded_token_fixture(directory: Path, documents: Iterable[str], *, tokenizer: str = "byte",
                          val_fraction: float = 0.1, eos: bool = False) -> Path:
    """The legacy token-store fixture format, each document encoded by the library's tokenize step
    (`Tokenize`), empty ones left out."""
    from dew.data.text import Tokenize

    step = Tokenize(tokenizer, eos=eos)
    vocabulary = step.vocabulary
    dtype = (np.uint8 if vocabulary.vocab_size <= 256 else
             np.uint16 if vocabulary.vocab_size <= 65536 else np.uint32)
    rows = [step.map({"text": document})["text"] for document in documents if document]
    tokens = np.concatenate([np.zeros(0, np.int32), *rows]).astype(dtype)
    held = int(len(tokens) * val_fraction)
    return write_token_corpus(directory, tokens[held:], tokens[:held], tokenizer=tokenizer,
                              vocab_size=vocabulary.vocab_size, eos_id=vocabulary.eos_id if eos else None)


def export_byte_tokens(root: Path, documents: Iterable[str], *flags: str) -> Path:
    """`documents` written as records under `root` and exported by `dew data export records` as byte
    tokens, each document ended by its EOS, into `root/tokens`, with the export's further `flags`."""
    from dew.cli.main import main

    records = root / "documents.jsonl"
    records.write_text("".join(json.dumps({"text": text}) + "\n" for text in documents))
    exported = root / "tokens"
    fields = json.dumps({"source": str(records)})
    assert main(["data", "export", "records", str(exported), "--fields", fields, "--tokenizer", "byte",
                 "--eos", *flags]) == 0
    return exported
