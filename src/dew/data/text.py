"""Tokenizers for language-model data, utf-8 bytes or any HF tokenizer, and
the text record steps: tokenizing, chunking, dropping, normalizing and
filtering text.

`import dew.data` stays cheap. Nothing here imports `transformers` at module
scope, and ByteTokenizer needs nothing but numpy. HFTokenizer loads its
tokenizer on first use, so a host without the hub cache still imports
`dew.data.text` (and everything that re-exports it) fine.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import threading
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import grain.python as grain
import numpy as np
from filelock import FileLock

from dew.data.tokens import chunk_table

if TYPE_CHECKING:
    from jax.typing import ArrayLike
    from transformers import PreTrainedConfig, PreTrainedTokenizerBase

    from dew.data.dataset import Batch
    from dew.data.pipeline import Dataset
    from dew.data.sources.base import Read


_load_lock = threading.Lock()
_loaded: dict[tuple[str, str | None, bool, str | None], PreTrainedTokenizerBase] = {}

RENAMED_TOKENIZERS: Mapping[str, str] = {
    # Dream-org/Dream-v0-Base-7B's tokenization_dream.py at 6572adb5 is transformers 4.46's
    # tokenization_qwen2.py with the class renamed, line for line; under 5.16.1 the two encode the
    # same ids, special tokens and the mask included.
    "DreamTokenizer": "Qwen2Tokenizer",
}
"""Tokenizer classes a release ships as remote code that are a transformers
class under another name: `load_tokenizer` loads such a release with that
class, and runs none of the release's code."""


_TOKENIZER_FILES = ["*.json", "*.jinja", "*.model", "*.txt", "*.tiktoken"]
"""The files a tokenizer's loader can read: its vocabulary, merges and templates."""


@dataclasses.dataclass(frozen=True)
class PinnedTokenizer:
    """A Hugging Face tokenizer resolved once to immutable files
    (`pin_tokenizer`): the snapshot its identity hashes and the directory it
    reads its vocabulary and templates from, which are one revision by
    construction."""

    snapshot: Mapping[str, object]
    directory: str

    def transformers(self) -> PreTrainedTokenizerBase:
        """The transformers tokenizer of these files, whose chat template
        renders conversations. A directory is read where it is, so the load
        reaches no hub, and it is the one load every reader of these files
        shares (`load_tokenizer`): keyed by their hashes, so a directory whose
        files changed since a load loads again."""
        files = json.dumps(self.snapshot["files"], sort_keys=True).encode()
        return load_tokenizer(self.directory, contents=hashlib.sha256(files).hexdigest())


def pin_tokenizer(name: str, *, local_files_only: bool = False) -> PinnedTokenizer:
    """Resolve `name` once: a local directory, or a hub repo (optionally
    `repo@revision`) downloaded at the commit its revision names, with every
    file its loader can read hashed. The Hub cache keeps a commit's files in
    a directory named for the commit, which is the revision recorded; offline
    (`HF_HUB_OFFLINE`), or with `local_files_only`, the name resolves to the
    commit the local cache holds."""
    if name == "byte":
        raise ValueError("'byte' names Dew's byte vocabulary (ByteTokenizer), which has no tokenizer files")
    directory = Path(name)
    repo = resolved = None
    if not directory.is_dir():
        from huggingface_hub import snapshot_download

        repo, separator, revision = name.rpartition("@")
        repo, revision = (repo, revision) if separator else (name, None)
        directory = Path(snapshot_download(repo, revision=revision, allow_patterns=_TOKENIZER_FILES,
                                           local_files_only=local_files_only))
        resolved = directory.name
    files = {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
             for pattern in _TOKENIZER_FILES
             for path in directory.rglob(pattern) if path.is_file()}
    snapshot = {"name": repo, "revision": resolved, "files": files}
    return PinnedTokenizer(snapshot, str(directory))


def load_tokenizer(name: str, *, revision: str | None = None, local_files_only: bool = False,
                   contents: str | None = None) -> PreTrainedTokenizerBase:
    """The HF tokenizer of a hub repo or local directory, loaded once per process, or once per
    `contents`, the digest of a directory's files where a caller pinned them (`PinnedTokenizer`).

    Every Dew path that needs an HF tokenizer loads it here, so one name at
    one revision is one object however many readers share it. A release whose
    tokenizer is a transformers class renamed in its own remote code loads as
    that class (`RENAMED_TOKENIZERS`).
    """
    repo, separator, pinned = name.rpartition("@")
    if separator and not Path(name).exists():
        if revision is not None and revision != pinned:
            raise ValueError(f"tokenizer {name} and revision={revision!r} disagree")
        name, revision = repo, pinned
    key = (name, revision, local_files_only, contents)
    held = _loaded.get(key)
    if held is not None:
        return held
    # Readers that miss together wait here and find the first one's load.
    # The lock also serializes the first transformers import, whose lazy
    # module two threads can otherwise catch half built.
    with _load_lock:
        held = _loaded.get(key)
        if held is None:
            import transformers
            from transformers import AutoTokenizer, PreTrainedTokenizerBase
            from transformers.models.auto.tokenization_auto import get_tokenizer_config

            stated = get_tokenizer_config(name, revision=revision, local_files_only=local_files_only)
            renamed = RENAMED_TOKENIZERS.get(str(stated.get("tokenizer_class")))
            loaded = (AutoTokenizer.from_pretrained(
                name, revision=revision, local_files_only=local_files_only,
                config=_untyped_config(name, revision, local_files_only)) if renamed is None
                else getattr(transformers, renamed).from_pretrained(name, revision=revision,
                                                                    local_files_only=local_files_only))
            # from_pretrained's annotation names classes transformers binds
            # to None without their backend, so its result is untyped until here.
            if not isinstance(loaded, PreTrainedTokenizerBase):
                raise TypeError(f"{name!r} loads as {type(loaded).__name__}, not a transformers tokenizer")
            held = _loaded[key] = loaded
    return held


_DISPATCH_FIELDS = ("model_type", "model_name", "tokenizer_class")
"""The model config's fields AutoTokenizer dispatches on, beside its `_name_or_path`
(tokenization_auto.py:779-809 and 931, transformers 5.16.1)."""


def _untyped_config(name: str, revision: str | None, local_files_only: bool) -> PreTrainedConfig | None:
    """AutoTokenizer's model config where transformers registers no class for the model's type;
    None where it does or the repo has no config, for AutoTokenizer to build its own.

    An unregistered type reaches AutoTokenizer as an untyped PreTrainedConfig of every field the
    file holds, which transformers validates against its own vocabulary: Dory's 'mlp' layer types
    fail `validate_layer_type`, though no tokenizer reads them. This config holds the fields
    AutoTokenizer dispatches on, read as `PreTrainedConfig.get_config_dict` resolves the file, so
    an unregistered type transformers still dispatches (InternLM2's) dispatches as before."""
    from transformers import PreTrainedConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    stated, _ = PreTrainedConfig.get_config_dict(name, revision=revision, local_files_only=local_files_only)
    if not stated or stated.get("model_type") in CONFIG_MAPPING:
        return None
    return PreTrainedConfig.from_dict(
        {"_name_or_path": name, **{field: stated[field] for field in _DISPATCH_FIELDS if field in stated}})


@runtime_checkable
class TokenArray(Protocol):
    """Holds ids as an array rather than as a list.

    numpy's and jax's arrays both hand their ids over through `tolist`, and a
    decode takes either beside a plain sequence, because a sampler returns a
    row of a device array while a caller writes a list. A row has a length;
    the scalars `ArrayLike` also covers are one id and have none.
    """

    def __len__(self) -> int: ...

    def tolist(self) -> list[int]: ...


class Tokenizer(Protocol):
    """What a text vocabulary offers, as `ByteTokenizer` and `HFTokenizer` do:
    text to ids, with or without the vocabulary's special tokens, ids back
    to text, and the id a sequence starts with, or None."""

    @property
    def bos_id(self) -> int | None: ...

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]: ...

    def decode(self, ids: ArrayLike | Sequence[int]) -> str: ...


def _token_row(ids: ArrayLike | Sequence[int]) -> list[int]:
    values = ids.tolist() if isinstance(ids, TokenArray) else ids
    if not isinstance(values, Sequence):
        raise TypeError("decode takes a row of token ids, not one id")
    return [int(token) for token in values]


class ByteTokenizer:
    """Encodes text as one id per utf-8 byte, over a vocabulary of 256.

    It needs no training and no download, so it is the default for small
    corpora and for tests. Its eos id is 255, a byte that never occurs in
    utf-8 text, and it has no bos id. Decoding inverts encoding for any
    unicode text. Ids that are not valid utf-8, which a model can generate,
    decode with replacement characters.
    """

    name = "byte"
    """The name a run records for this vocabulary, which `tokenizer_for` resolves back to it."""

    def __init__(self):
        self.vocab_size = 256
        self.eos_id = 255
        self.bos_id = None

    def dependencies(self) -> Mapping[str, object]:
        return {"name": "byte", "encoding": "utf-8", "eos": 255}

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Return the text's utf-8 bytes as ids.

        `add_special_tokens` has no effect, because this vocabulary has no
        special tokens to insert.
        """
        return list(text.encode("utf-8"))

    def decode(self, ids: ArrayLike | Sequence[int]) -> str:
        return bytes(_token_row(ids)).decode("utf-8", errors="replace")

    def __repr__(self):
        return self.__class__.__name__ + "()"


class HFTokenizer:
    """A Hugging Face tokenizer, resolved from its hub name or local directory
    on first use.

    It resolves its name once (`pinned`), and its identity (`dependencies`)
    and every encoding read that one resolution, here and in any process it
    is pickled to, so a hub name that moves, or another revision of it a
    process already loaded, never encodes rows under the identity of this
    one. Because it resolves lazily, `import dew.data.text` (and
    `import dew.data`) does not import `transformers` or look anything up on
    the hub until a caller uses the tokenizer.
    """

    def __init__(self, name: str, *, local_files_only: bool = False):
        self.name = name
        self.local_files_only = local_files_only

    @functools.cached_property
    def pinned(self) -> PinnedTokenizer:
        """The name resolved once to immutable files (`pin_tokenizer`)."""
        return pin_tokenizer(self.name, local_files_only=self.local_files_only)

    def dependencies(self) -> Mapping[str, object]:
        return self.pinned.snapshot

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.pinned.transformers()

    @property
    def vocab_size(self) -> int:
        # len() counts every vocabulary entry, including added tokens;
        # some fast tokenizers report a smaller .vocab_size than they emit.
        return len(self.tokenizer)

    @property
    def eos_id(self) -> int | None:
        """The eos token's id, or None for a tokenizer that declares none.

        Raises `TypeError` when the tokenizer reports something other than
        one id.
        """
        eos = self.tokenizer.eos_token_id
        if eos is None or isinstance(eos, int):
            return eos
        raise TypeError(f"tokenizer {self.name!r} reports eos_token_id {eos!r}, not one id")

    @property
    def bos_id(self) -> int | None:
        """The id the tokenizer starts a sequence with, or None where it has none.

        Raises `TypeError` when the tokenizer reports something other than
        one id.
        """
        bos = self.tokenizer.bos_token_id
        if bos is None or isinstance(bos, int):
            return bos
        raise TypeError(f"{self.name} names a bos_token_id that is not one id: {bos!r}")

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: ArrayLike | Sequence[int]) -> str:
        # batch_decode of one row is decode's text, typed as one str.
        return self.tokenizer.batch_decode([_token_row(ids)])[0]

    def save_pretrained(self, directory) -> None:
        """Write this tokenizer's own files into an export directory.

        The tokenizer writes tokenizer.json, tokenizer_config.json and its
        vocabulary itself, so the export copies no files by hand and the
        result loads in anything that reads the HF layout.
        """
        self.tokenizer.save_pretrained(str(directory))

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"


def tokenizer_name(tokenizer: str | PreTrainedTokenizerBase) -> str:
    """A tokenizer as the name Dew's steps carry: a name as it is, and a
    transformers tokenizer object saved once to a directory named for its
    files' hash, which every worker reads and every run hashes alike."""
    if isinstance(tokenizer, str):
        return tokenizer
    import shutil
    import tempfile

    from dew.cache import dew_cache_dir

    root = Path(dew_cache_dir(), "tokenizers")
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=root))
    try:
        tokenizer.save_pretrained(str(staging))
        digest = hashlib.sha256()
        for path in sorted(path for path in staging.rglob("*") if path.is_file()):
            digest.update(str(path.relative_to(staging)).encode() + b"\0" + path.read_bytes())
        directory = root / digest.hexdigest()
        # One publisher at a time: another process saving the same tokenizer writes the same
        # files, and whichever comes second finds the directory there.
        with FileLock(str(directory) + ".lock"):
            if not directory.is_dir():
                staging.rename(directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return str(directory)


def tokenize(tokenizer: str | PreTrainedTokenizerBase = "byte", *, field: str = "text",
             eos: bool | None = None) -> Tokenize:
    """A record step that encodes the text in `field` to token ids under
    "text", each document ended by the tokenizer's EOS with `eos`, or where
    the tokenizer has one with None.
    `tokenizer` is "byte" (Dew's utf-8 vocabulary), a Hub repo or directory,
    or a transformers tokenizer object."""
    return Tokenize(tokenizer_name(tokenizer), field, eos)


@dataclasses.dataclass(frozen=True)
class Tokenize(grain.MapTransform):
    """The text in `field` as token ids under "text", each document ended by
    the tokenizer's EOS with `eos`, or where the tokenizer has one with None."""

    tokenizer: str = "byte"
    field: str = "text"
    eos: bool | None = False

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """Token ids of each document's length under "text", by `tokenizer`, `vocab_size` of them."""
        from dew.data.facts import unshaped

        return {**unshaped(facts, self.field, "text"), "tokenizer": self.tokenizer,
                "vocab_size": self.vocabulary.vocab_size}

    @functools.cached_property
    def vocabulary(self) -> ByteTokenizer | HFTokenizer:
        """The tokenizer `tokenizer` names, which resolves its name once
        (`HFTokenizer.pinned`): the step's identity and its encoding both read
        that one resolution, in this process and in any worker it is sent to."""
        return tokenizer_for(self.tokenizer)

    def dependencies(self) -> Mapping[str, object]:
        return {"tokenizer": self.vocabulary.dependencies()}

    def map(self, element: Batch) -> Batch:
        value = element[self.field]
        if not isinstance(value, str):
            raise ValueError(f"Tokenize field {self.field!r} must contain text")
        tokenizer = self.vocabulary
        ids = tokenizer.encode(value)
        if (self.eos or (self.eos is None and tokenizer.eos_id is not None)) and ids:
            if tokenizer.eos_id is None:
                raise ValueError(f"tokenizer {self.tokenizer} has no EOS id")
            ids.append(tokenizer.eos_id)
        kept = {key: value for key, value in element.items() if key != self.field}
        return {**kept, "text": np.asarray(ids, np.int32), "_dew_length": len(ids)}


@dataclasses.dataclass(frozen=True)
class Chunks:
    """A document's token `fields` cut into pieces of at most `length`, each
    starting `length - overlap` after the last (`flat_map`), before `pack`,
    which refuses a document longer than its window. With `overlap=1` every
    token's next-token target stays in one piece: pack to `length` with
    `seq_len=length`."""

    length: int
    _: dataclasses.KW_ONLY
    fields: tuple[str, ...] = ("text",)
    overlap: int = 0

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """Pieces of the documents, each field of no fixed length still."""
        return facts

    def __post_init__(self) -> None:
        if not 0 <= self.overlap < self.length:
            raise ValueError(f"chunks of {self.length} overlap by 0 to {self.length - 1}, not {self.overlap}")

    def flat_map(self, element: Batch) -> Sequence[Batch]:
        count = len(element[self.fields[0]])
        _, starts = chunk_table([count], self.length, self.overlap)
        return tuple(
            {
                key: value[start : start + self.length] if key in self.fields else value
                for key, value in element.items()
            }
            | {"_dew_length": min(self.length, count - start)}
            for start in starts.tolist()
        )


@dataclasses.dataclass(frozen=True)
class Dropped(grain.RandomMapTransform):
    """The text in `field` emptied with probability `p`, as classifier-free
    guidance trains its unconditioned branch."""

    field: str = "caption"
    p: float = 0.1

    def __post_init__(self) -> None:
        if not 0 <= self.p <= 1:
            raise ValueError(f"a drop probability is in [0, 1], not {self.p}")

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        return {**element, self.field: ""} if rng.random() < self.p else element


@dataclasses.dataclass(frozen=True)
class Normalized(grain.MapTransform):
    """The text in `field` in Unicode normal form `form` (`unicodedata.normalize`), so text written
    with composed, decomposed or compatibility characters (a ligature, a full-width digit) reads and
    tokenizes alike; other fields pass through."""

    form: Literal["NFC", "NFKC", "NFD", "NFKD"] = "NFKC"
    _: dataclasses.KW_ONLY
    field: str = "text"

    def map(self, element: Batch) -> Batch:
        text = element[self.field]
        text = text.decode("utf-8") if isinstance(text, bytes) else str(text)
        return {**element, self.field: unicodedata.normalize(self.form, text)}


@dataclasses.dataclass(frozen=True)
class MinChars:
    """Whether the text in `field` holds at least `count` characters."""

    count: int
    field: str = "text"
    reason = "too_short"

    def __call__(self, element: Batch) -> bool:
        return len(str(element[self.field])) >= self.count


def export_tokens(dataset: Dataset, url: str, tokenizer: str, *, field: str = "text", eos: bool = True,
                  rows_per_shard: int = 100_000, read: Read = "streamed", provision: bool = False
                  ) -> Mapping[str, object]:
    """`dataset`'s documents in `field` tokenized by `tokenizer` and written to `url` as a committed
    token store (ArrayRecord shards, each record a document's ids under "text", its manifest naming
    the tokenizer, vocabulary and EOS), which `dew.data.sources.TokenFiles(url)` reads back and
    `presets.lm(ds, None, ...)` packs. Each document is ended by the tokenizer's EOS with `eos`.
    Returns the store's manifest."""
    from dew.data.compile import iterate
    from dew.data.sinks import export

    vocabulary = tokenizer_for(tokenizer)
    documents = dataset.map(Tokenize(tokenizer, field, eos))
    metadata = {"token_store": True, "tokenizer": tokenizer, "vocab_size": vocabulary.vocab_size,
                "dtype": "int32", "eos_id": vocabulary.eos_id if eos else None}
    return export(iterate(documents, read=read, provision=provision), url, format="array_record",
                  rows_per_shard=rows_per_shard, metadata=metadata, length_field="text")


def train_tokenizer(dataset: Dataset, base: str, path: str, *, vocab_size: int, field: str = "text",
                    records: int | None = None) -> str:
    """A tokenizer of `vocab_size` tokens trained on the text in `field` of `dataset`'s records, the
    first `records` where given, by the algorithm, normalization and special tokens of `base` (a Hub
    repo or directory: a BPE tokenizer's learns merges, a unigram's pieces), saved at `path`, which it
    returns for `tokenize(path)` and the `lm` and `sft` presets to read. A `vocab_size` smaller than
    the base's alphabet and special tokens, which its algorithm always keeps, is refused, and so is a
    base without a fast tokenizer, which has no algorithm to train."""
    if vocab_size < 1:
        raise ValueError("a trained tokenizer needs a positive vocab_size")
    rows = iter(dataset if records is None else dataset.take(records))

    def texts() -> Iterator[str]:
        for row in rows:
            text = row[field]
            if not isinstance(text, str):
                raise ValueError(f"train_tokenizer reads text from {field!r}, which holds "
                                 f"{type(text).__name__}")
            yield text

    from transformers import PreTrainedTokenizerFast

    learning = load_tokenizer(base)
    if not isinstance(learning, PreTrainedTokenizerFast):
        raise ValueError(f"{base} loads as {type(learning).__name__}; training learns a fast (tokenizers) "
                         "tokenizer's algorithm, so give a base that has one")
    trained = learning.train_new_from_iterator(texts(), vocab_size)
    if len(trained) > vocab_size:
        raise ValueError(f"{base}'s algorithm keeps {len(trained)} tokens of its alphabet and special tokens "
                         f"and those it learned, more than vocab_size={vocab_size}: give at least that many")
    trained.save_pretrained(path)
    return path


def tokenizer_for(name: str, *, local_files_only: bool = False) -> ByteTokenizer | HFTokenizer:
    """Builds the tokenizer `name` asks for.

    `byte` is dew's own utf-8 vocabulary; any other name is the HF tokenizer
    of that repo or local directory. Resolving names here alone keeps a
    training run and an export of what it trained reading one name one way.
    An export passes `local_files_only`, since writing a checkpoint out is no
    reason to reach the hub.
    """
    if name == "byte":
        return ByteTokenizer()
    return HFTokenizer(name, local_files_only=local_files_only)
