"""The binary schema shared by distinct token-corpus generators."""

import json
from collections.abc import Iterable
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from dew.objectives.lm import TEXT_KEY
from tools.token_corpus import write_token_meta


def token_batch_of(vocab: int, seq: int, batch: int = 4, seed: int = 0) -> dict:
    """A `[batch, seq + 1]` batch of ids under `vocab`, drawn from `seed`, as the token pipeline packs
    them."""
    ids = np.random.RandomState(seed).randint(0, vocab, (batch, seq + 1))
    return {TEXT_KEY: jnp.asarray(ids, jnp.int32)}


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


def token_rows(vocabulary: int, rows: int, length: int, *, seed: int = 0, low: int = 0) -> dict:
    """A batch of `rows` token rows `length` long under the key "text", drawn at `seed` from
    `low` up to `vocabulary`."""
    tokens = np.random.default_rng(seed).integers(low, vocabulary, size=(rows, length))
    return {"text": tokens.astype(np.int32)}


def repeated_token_batch(vocabulary: int, rows: int, length: int, seed: int = 0):
    """One batch of `token_rows` yielded forever: a train loader whose every step reads it."""
    batch = token_rows(vocabulary, rows, length, seed=seed)
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
