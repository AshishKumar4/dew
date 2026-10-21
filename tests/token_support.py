"""The binary schema shared by distinct token-corpus generators."""

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def write_token_corpus(directory: Path, train: np.ndarray, val: np.ndarray | None = None,
                       *, tokenizer: str = "byte", vocab_size: int = 256,
                       eos_id: int | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train.bin").write_bytes(train.tobytes())
    if val is not None:
        assert train.dtype == val.dtype
        (directory / "val.bin").write_bytes(val.tobytes())
    meta = {"tokenizer": tokenizer, "vocab_size": vocab_size, "dtype": train.dtype.name,
            "train_tokens": len(train), "val_tokens": 0 if val is None else len(val)}
    if eos_id is not None:
        meta["eos_id"] = eos_id
    (directory / "meta.json").write_text(json.dumps(meta))
    return directory


def encoded_token_fixture(directory: Path, documents: Iterable[str], *, tokenizer: str = "byte",
                          val_fraction: float = 0.1, eos: bool = False) -> Path:
    """The legacy token-store fixture format, encoded once per document."""
    from dew.data.text import tokenizer_for

    vocabulary = tokenizer_for(tokenizer)
    pieces = []
    for document in documents:
        if not document:
            continue
        ids = vocabulary.encode(document)
        if eos:
            if vocabulary.eos_id is None:
                raise ValueError("an EOS fixture needs a tokenizer with an EOS id")
            ids.append(vocabulary.eos_id)
        pieces.extend(ids)
    dtype = (np.uint8 if vocabulary.vocab_size <= 256 else
             np.uint16 if vocabulary.vocab_size <= 65536 else np.uint32)
    tokens = np.asarray(pieces, dtype=dtype)
    held = int(len(tokens) * val_fraction)
    return write_token_corpus(directory, tokens[held:], tokens[:held], tokenizer=tokenizer,
                              vocab_size=vocabulary.vocab_size, eos_id=vocabulary.eos_id if eos else None)


def repeated_token_batch(vocabulary: int, rows: int, length: int, seed: int = 0):
    """One batch of `rows` token rows `length` long under the key "text", drawn at `seed` from
    `vocabulary`, yielded forever: a train loader whose every step reads the same batch."""
    tokens = np.random.default_rng(seed).integers(0, vocabulary, size=(rows, length))
    batch = {"text": tokens.astype(np.int32)}
    while True:
        yield batch


def byte_level_tokenizer(directory: Path):
    """A byte-level BPE over all 256 bytes, saved in `directory` and read back through
    transformers: any text has a spelling, and a code point may split across tokens."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    alphabet = bytes_to_unicode()
    backend = Tokenizer(models.BPE({alphabet[byte]: byte for byte in range(256)}, [], unk_token=None))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    PreTrainedTokenizerFast(tokenizer_object=backend).save_pretrained(directory)
    return AutoTokenizer.from_pretrained(directory, local_files_only=True)
