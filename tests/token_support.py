"""Token corpora the tests read, written as `dew data export` writes a token store."""

import dataclasses
import itertools
import json
from collections.abc import Iterable
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from dew.batch import TEXT_KEY


@dataclasses.dataclass(frozen=True)
class Corpus:
    """A token corpus's committed stores, as `dew data export` writes them: its training tokens, and
    its validation tokens where it has them, each read as `TokenFiles(url)`."""

    train: str
    val: str | None = None


def token_batch_of(vocab: int, seq: int, batch: int = 4, seed: int = 0) -> dict:
    """A `[batch, seq + 1]` batch of ids under `vocab`, drawn from `seed`, as the token pipeline packs
    them."""
    ids = np.random.RandomState(seed).randint(0, vocab, (batch, seq + 1))
    return {TEXT_KEY: jnp.asarray(ids, jnp.int32)}


def write_token_corpus(directory: Path, train: np.ndarray, val: np.ndarray | None = None,
                       *, tokenizer: str = "byte", vocab_size: int = 256,
                       eos_id: int | None = None) -> Corpus:
    """`train`, and `val` where given, each a stream of token ids, written as committed token stores
    at `directory/train` and `directory/val`: each document a record, the tokens through each
    `eos_id` and any tail past the last; without an `eos_id`, the whole stream one record."""
    directory.mkdir(parents=True, exist_ok=True)
    train_url = _token_store(directory / "train", train, tokenizer=tokenizer, vocab_size=vocab_size,
                             eos_id=eos_id)
    val_url = None if val is None else _token_store(directory / "val", val, tokenizer=tokenizer,
                                                    vocab_size=vocab_size, eos_id=eos_id)
    return Corpus(train_url, val_url)


def _token_store(url: Path, tokens: np.ndarray, *, tokenizer: str, vocab_size: int,
                 eos_id: int | None) -> str:
    from dew.data import Memory
    from dew.data.sinks import export

    ids = np.asarray(tokens).astype(np.int32)
    ends = [] if eos_id is None else (np.flatnonzero(ids == eos_id) + 1).tolist()
    bounds = [0, *ends, *([] if ends and ends[-1] == len(ids) else [len(ids)])]
    documents = [ids[start:stop] for start, stop in itertools.pairwise(bounds) if stop > start]
    stated = {"token_store": True, "tokenizer": tokenizer, "vocab_size": vocab_size, "dtype": "int32",
              "eos_id": eos_id}
    export(Memory([{"text": document} for document in documents]), str(url), format="array_record",
           length_field="text", metadata=stated)
    return str(url)


def encoded_token_fixture(directory: Path, documents: Iterable[str], *, tokenizer: str = "byte",
                          val_fraction: float = 0.1, eos: bool = False) -> Corpus:
    """`documents` encoded by the library's tokenize step (`Tokenize`), empty ones left out, as a
    token corpus whose first `val_fraction` of the tokens is its validation store."""
    from dew.data.text import Tokenize

    step = Tokenize(tokenizer, eos=eos)
    vocabulary = step.vocabulary
    rows = [step.map({"text": document})["text"] for document in documents if document]
    tokens = np.concatenate([np.zeros(0, np.int32), *rows])
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


def wordpiece_tokenizer(directory: Path):
    """A BERT-style WordPiece tokenizer over the lowercase letters and their continuations, saved
    in `directory` and read back through transformers: [CLS] and [SEP] around each encoding, and a
    decoder that joins a continuation to the word before it and spaces the words."""
    from string import ascii_lowercase

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    pieces = [*specials, *ascii_lowercase, *(f"##{letter}" for letter in ascii_lowercase)]
    vocabulary = {piece: index for index, piece in enumerate(pieces)}
    backend = Tokenizer(models.WordPiece(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.decoder = decoders.WordPiece()
    backend.post_processor = processors.BertProcessing(("[SEP]", 3), ("[CLS]", 2))
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", cls_token="[CLS]",
                            sep_token="[SEP]", pad_token="[PAD]").save_pretrained(directory)
    return AutoTokenizer.from_pretrained(directory, local_files_only=True)


def export_byte_tokens(root: Path, documents: Iterable[str], *flags: str) -> Path:
    """`documents` written as records under `root` and exported by `dew data export` as byte tokens,
    each document ended by its EOS, into `root/tokens`, with the export's further `flags`."""
    from dew.cli.main import main

    records = root / "documents.jsonl"
    records.write_text("".join(json.dumps({"text": text}) + "\n" for text in documents))
    exported = root / "tokens"
    assert main(["data", "export", str(records), str(exported), "--tokenizer", "byte", "--eos", *flags]) == 0
    return exported
