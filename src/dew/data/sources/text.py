"""Tokenized corpora as one stream of ids, and the two records cut out of it.

A corpus is a stream of token ids, whatever holds it: the `.bin` files
`TokenCorpus.write` (and `dew tokenize`) writes, ArrayRecord shards of token
arrays, or a parquet column of them. `TokenSource` is that stream, read by slice, and the
three readers below are the three stores it can live in. `token_corpus`
resolves a directory to the train and validation pair a run needs, by what
the files in it are.

`TokenWindowSource` reads a record as a contiguous window of `seq_len + 1`
ids starting at `i * stride`. The default stride is `seq_len`, so record
i's last token is record i+1's first and every transition appears once.
Smaller strides overlap windows. There is no decoding or randomness here;
the shuffle lives in the sampler.

`TokenDocumentSource` reads a record as one document: the span from after the
previous eos id through its own. It exists for the packed pipeline, which
cares where documents end and lets grain pack several of them into one
window. Both read through `TokenSource` and nothing else, so the same corpus
in any of the three stores gives the same windows and the same packing plan.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from dew.data.text import ByteTokenizer, HFTokenizer

# meta.json's "dtype" names numpy dtypes; uint16 covers byte tokenizers and
# most HF ones, uint32 the rest.
_DEFAULT_DTYPE = np.dtype("<u2")


@runtime_checkable
class TokenSource(Protocol):
    """Reads a tokenized corpus as one stream of ids.

    `len` is the tokens it holds and `source[start:stop]` is that span of
    them. Both record readers below cut their records out of this and nothing
    else: a fixed window is a strided span, and a document is the span
    between two eos ids.

    `eos_id` is the id that closes a document, which only the packed reader
    needs and a corpus written without boundaries does not have. A source
    also describes itself by what it reads rather than by its address, which
    is what a saved position compares against (`describe`).
    """

    @property
    def eos_id(self) -> int | None: ...

    def __len__(self) -> int: ...

    def __getitem__(self, span: slice) -> np.ndarray: ...


def _meta(root: Path) -> Mapping[str, object]:
    """The `meta.json` a tokenize run wrote in `root`, or nothing."""
    beside = root / "meta.json"
    if not beside.is_file():
        return {}
    with open(beside) as record:
        held = json.load(record)
    if not isinstance(held, dict):
        raise ValueError(f"{beside} holds {type(held).__name__}, not a JSON object")
    return held


def _recorded(meta: Mapping[str, object], key: str) -> int | None:
    """One integer meta.json records, or None when it records none."""
    held = meta.get(key)
    return None if held is None else int(str(held))


def _dtype(meta: Mapping[str, object]) -> np.dtype:
    """The width the ids were written at, uint16 being nanoGPT's default."""
    return np.dtype(str(meta.get("dtype", _DEFAULT_DTYPE)))


class _Reopened:
    """A source whose open handle is dropped from its pickle and reopened.

    Neither a memmap nor an arrayrecord reader survives grain's round trip
    to a worker, and the rest of the state says how to open one. `handle`
    names the attribute holding it and `open_handle` opens it again.
    """

    handle: str

    def open_handle(self):
        """The handle this source reads through, opened afresh."""
        raise NotImplementedError

    def __getstate__(self):
        return {**self.__dict__, self.handle: None}

    def __setstate__(self, state):
        self.__dict__.update(state)
        setattr(self, self.handle, self.open_handle())


# Characters read per chunk of a text file; small enough that the encoded ids
# of one chunk are a rounding error against memory, large enough to amortize
# reads.
CHUNK_CHARS = 1 << 20


@dataclasses.dataclass(frozen=True)
class TokenCorpus:
    """A token directory: `train.bin`, `val.bin` and the `meta.json` that
    records these fields."""

    tokenizer: str
    vocab_size: int
    dtype: str
    train_tokens: int
    val_tokens: int
    eos_id: int | None

    @classmethod
    def read(cls, directory: str | os.PathLike[str]) -> TokenCorpus:
        """What `meta.json` in `directory` records."""
        return cls(**json.loads((Path(directory) / "meta.json").read_text()))

    @classmethod
    def write(cls, documents: str | os.PathLike[str] | Iterable[str], out: str | os.PathLike[str], *,
              tokenizer: str = "byte", val_fraction: float = 0.01, pack: bool = False) -> TokenCorpus:
        """Tokenize a corpus into the directory `TokenWindows` and `PackedTokens` read.

        `documents` is a text file, a directory read as every `*.txt` under it
        in path order (each file one document), or any iterable of strings, one
        document each, such as `(row["text"] for row in hf_split)`. A str is
        always a path; pass text itself in a list. `tokenizer` is `"byte"` or a
        Hugging Face tokenizer name, recorded in `meta.json` so a run can check
        the ids against its model.

        The ids go to `train.bin` and `val.bin` at the smallest unsigned width
        the vocabulary fits, with `val_fraction` of the stream, from its head,
        held out. The ids the tokenizer adds to one encode, a bos id for many,
        are written once per document. `pack` ends every document with the
        tokenizer's eos id, which `PackedTokens` cuts documents at. A file is
        read in line-bounded chunks so a corpus larger than memory costs disk,
        and each chunk ends at a newline, so a tokenizer that merges across its
        input sees whole lines.
        """
        from dew.data.text import tokenizer_for

        if not 0.0 <= val_fraction < 1.0:
            raise ValueError(f"val_fraction is a fraction of the stream in [0, 1), got {val_fraction}")
        encoder = tokenizer_for(tokenizer)
        eos = encoder.eos_id if pack else None
        if pack and eos is None:
            raise ValueError(f"pack ends every document with an eos id, and tokenizer {tokenizer!r} has none")
        root = Path(out)
        root.mkdir(parents=True, exist_ok=True)
        dtype = dtype_for(encoder.vocab_size)

        # One encode pass writes the whole stream to a scratch file; the split
        # point needs the total count, and slicing a memmap of it costs a linear
        # copy, not a second tokenization.
        scratch = root / "all.bin"
        opening, closing = _added(encoder)
        # Every document is terminated, the last included, because the packing
        # source reads a record as the span up to an eos.
        closing = closing if eos is None else [*closing, eos]
        total = 0
        try:
            with open(scratch, "wb") as handle:
                for document in _documents(documents):
                    written = 0
                    for chunk in document:
                        ids = encoder.encode(chunk, add_special_tokens=False)
                        if ids and not written:
                            ids = [*opening, *ids]
                        handle.write(np.asarray(ids, dtype=dtype).tobytes())
                        written += len(ids)
                    if written:
                        handle.write(np.asarray(closing, dtype=dtype).tobytes())
                        written += len(closing)
                    total += written
            if total < 2:
                raise ValueError(f"the corpus tokenized to {total} tokens; a window needs at least 2")
            held_out = min(round(total * val_fraction), total - 1)
            stream = np.memmap(scratch, dtype=dtype, mode="r")
            stream[:held_out].tofile(root / "val.bin")
            stream[held_out:].tofile(root / "train.bin")
        finally:
            scratch.unlink(missing_ok=True)

        corpus = cls(tokenizer=tokenizer, vocab_size=encoder.vocab_size, dtype=dtype.name,
                     train_tokens=total - held_out, val_tokens=held_out, eos_id=eos)
        (root / "meta.json").write_text(json.dumps(dataclasses.asdict(corpus), indent=2) + "\n")
        return corpus


def _added(encoder: ByteTokenizer | HFTokenizer) -> tuple[list[int], list[int]]:
    """The ids the tokenizer adds before and after the text of one encode.

    A document is encoded in chunks, each without them, so a tokenizer that
    starts a sequence with its bos id puts it once at the document's start
    rather than once per chunk. Read off one probe encode.
    """
    plain = encoder.encode("a", add_special_tokens=False)
    whole = encoder.encode("a")
    for start in range(len(whole) - len(plain) + 1):
        if whole[start:start + len(plain)] == plain:
            return whole[:start], whole[start + len(plain):]
    raise ValueError(
        f"{encoder!r} encodes a probe with special tokens to {whole}, which does not "
        f"contain its plain encoding {plain}; the ids it adds cannot be placed per document")


def dtype_for(vocab_size: int) -> np.dtype:
    """The smallest unsigned dtype that holds every id a vocabulary emits."""
    for dtype in (np.dtype("uint8"), np.dtype("uint16")):
        if vocab_size <= np.iinfo(dtype).max + 1:
            return dtype
    return np.dtype("uint32")


def _documents(documents: str | os.PathLike[str] | Iterable[str]) -> Iterator[Iterable[str]]:
    """Each document of `documents` as the chunks of text it is encoded in."""
    if not isinstance(documents, (str, os.PathLike)):
        for document in documents:
            yield (document,)
        return
    root = Path(documents)
    if root.is_file():
        paths = [root]
    elif root.is_dir():
        paths = sorted(path for path in root.rglob("*.txt") if path.is_file())
        if not paths:
            raise ValueError(f"{root} holds no *.txt file to tokenize")
    else:
        raise FileNotFoundError(f"{root} is neither a text file nor a directory of them")
    for path in paths:
        yield _chunks(path)


def _chunks(path: Path) -> Iterator[str]:
    """`path`'s text in chunks of about `CHUNK_CHARS`, each ending at a newline
    except the file's last; a line longer than a chunk grows until it ends."""
    with open(path, encoding="utf-8") as handle:
        carry = ""
        while chunk := handle.read(CHUNK_CHARS):
            chunk = carry + chunk
            split = chunk.rfind("\n")
            if split < 0:
                carry = chunk
                continue
            yield chunk[:split + 1]
            carry = chunk[split + 1:]
        if carry:
            yield carry


class TokenBytes(_Reopened):
    """Reads a flat `.bin` of token ids through a memmap.

    The dtype comes from the sibling `meta.json` when present, where the
    tokenize tool records it, and is uint16 otherwise. The file is never
    loaded into memory: a worker reads only the span it is asked for.
    """

    handle = "_tokens"

    def __init__(self, path: str, eos_id: int | None = None):
        self.path = str(path)
        meta = _meta(Path(self.path).parent)
        self.dtype = _dtype(meta)
        self.eos_id = eos_id if eos_id is not None else _recorded(meta, "eos_id")
        self._tokens = self.open_handle()

    def open_handle(self) -> np.memmap:
        return np.memmap(self.path, dtype=self.dtype, mode="r")

    def __repr__(self) -> str:
        return f"TokenBytes(path={self.path!r})"

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, span: slice) -> np.ndarray:
        return np.asarray(self._tokens[span])


class _Sharded:
    """Reads token arrays as one stream, in the order they are stored.

    A corpus of records or rows is the concatenation of them, so where a span
    falls is a binary search over their lengths, read once at construction. A
    span that crosses a boundary reads both pieces and joins them; one that
    does not reads one. Only the pieces a record covers are read, so a corpus
    is no more in memory than the memmap is.
    """

    def __init__(self, lengths: Sequence[int], dtype: np.dtype):
        self.dtype = dtype
        self._ends = np.cumsum(np.asarray(lengths, np.int64)) if lengths else np.zeros(0, np.int64)

    def piece(self, index: int) -> np.ndarray:
        """Record or row `index`, as its token ids."""
        raise NotImplementedError

    def __len__(self) -> int:
        return int(self._ends[-1]) if len(self._ends) else 0

    def __getitem__(self, span: slice) -> np.ndarray:
        start, stop, step = span.indices(len(self))
        if step != 1:
            raise ValueError(
                f"a token corpus is read in contiguous spans, and this one asks "
                f"for every {step}th token")
        pieces = []
        for index in range(int(np.searchsorted(self._ends, start, side="right")),
                           len(self._ends)):
            begin = 0 if index == 0 else int(self._ends[index - 1])
            if begin >= stop:
                break
            pieces.append(self.piece(index)[max(start - begin, 0):stop - begin])
        if not pieces:
            return np.empty(0, self.dtype)
        return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]


class TokenRecords(_Reopened, _Sharded):
    """Reads token arrays in ArrayRecord shards as one stream.

    Each record holds one array of ids, as its raw bytes or under `field` of
    the packed dict `dew.data.images.pack_dict_of_byte_arrays` writes. The
    records are the corpus in file order, so a tokenizer that wrote one
    document per record and one that wrote fixed blocks read back the same.
    """

    handle = "_records"

    def __init__(self, paths: Sequence[str], *, field: str | None = None,
                 dtype: np.dtype = _DEFAULT_DTYPE, eos_id: int | None = None):
        if not paths:
            raise ValueError("TokenRecords needs at least one arrayrecord file")
        self.paths = [str(path) for path in paths]
        self.field = field
        self.eos_id = None if eos_id is None else int(eos_id)
        self._records = self.open_handle()
        super().__init__([len(self._ids(index, np.dtype(dtype)))
                          for index in range(len(self._records))], np.dtype(dtype))

    def _ids(self, index: int, dtype: np.dtype) -> np.ndarray:
        """Record `index`'s token ids, out of its bytes."""
        from ..images import unpack_dict_of_byte_arrays

        raw = self._records[index]
        if self.field is not None:
            raw = unpack_dict_of_byte_arrays(raw)[self.field]
        return np.frombuffer(raw, dtype=dtype)

    def piece(self, index: int) -> np.ndarray:
        return self._ids(index, self.dtype)

    def open_handle(self):
        from array_record.python.array_record_data_source import ArrayRecordDataSource

        return ArrayRecordDataSource(list(self.paths))

    def __repr__(self) -> str:
        return f"TokenRecords(paths={self.paths!r}, field={self.field!r})"


class TokenColumn(_Sharded):
    """Reads one column of token ids in parquet files as one stream.

    The column holds a list of integers per row, which is what a tokenized
    dataset written with `to_parquet` holds. The rows are the corpus in file
    order. `column` names which column to read, and a file that holds one
    column needs no name.
    """

    def __init__(self, paths: Sequence[str], *, column: str | None = None,
                 dtype: np.dtype = _DEFAULT_DTYPE, eos_id: int | None = None):
        import pyarrow.parquet as pq

        if not paths:
            raise ValueError("TokenColumn needs at least one parquet file")
        self.paths = [str(path) for path in paths]
        self.eos_id = None if eos_id is None else int(eos_id)
        table = pq.read_table(self.paths)
        self.column = _named_column(table.column_names, column, self.paths)
        self._rows = [np.asarray(row, dtype) for row in
                      table.column(self.column).to_pylist()]
        super().__init__([len(row) for row in self._rows], np.dtype(dtype))

    def piece(self, index: int) -> np.ndarray:
        return self._rows[index]

    def __repr__(self) -> str:
        return f"TokenColumn(paths={self.paths!r}, column={self.column!r})"


def _named_column(held: Sequence[str], named: str | None,
                  paths: Sequence[str]) -> str:
    """The column the ids are in: the one named, or the only one there is."""
    if named is not None:
        if named not in held:
            raise ValueError(
                f"{list(paths)} holds no column {named!r}; it holds {sorted(held)}")
        return named
    if len(held) != 1:
        raise ValueError(
            f"{list(paths)} holds {sorted(held)}, so which column holds the token "
            f"ids has to be named")
    return held[0]


_STORES = (".bin", ".array_record", ".parquet")
"""What a tokenized corpus is held in, named by the suffix of its files."""


def same_tokenizer(paths: list[str]) -> None:
    """Refuse tokenized directories whose ids come from different tokenizers.

    A mixture reads one vocabulary, so every corpus's `meta.json` has to
    record the same tokenizer and eos id; a directory without the record
    says nothing and is not refused.
    """
    recorded = {path: (meta.get("tokenizer"), meta.get("eos_id"))
                for path in paths for meta in (_meta(Path(path)),) if meta}
    if len(set(recorded.values())) > 1:
        raise ValueError(
            f"a mixture reads one vocabulary, and these corpora record different "
            f"(tokenizer, eos_id): {recorded}")


def token_corpus(path: str | None, name: str, *, field: str | None = None
                 ) -> tuple[TokenSource, TokenSource]:
    """The `(train, val)` corpora of a tokenized directory, both required.

    A split's files are the ones named for it, and their suffix says which
    store they are in: `train.bin`, `train*.array_record*` shards, or
    `train*.parquet`. Reading train in val's place would score the validation
    pass on the windows the model trains on, so a directory holding one
    without the other is refused rather than halved.
    """
    if not path:
        raise ValueError(
            f"{name} needs path= set to the directory `dew tokenize` or TokenCorpus.write wrote")
    root = Path(path)
    return _split(root, "train", name, field), _split(root, "val", name, field)


def _split(root: Path, split: str, name: str, field: str | None) -> TokenSource:
    """One split of a tokenized directory, out of the store its files are in."""
    meta = _meta(root)
    dtype, eos_id = _dtype(meta), _recorded(meta, "eos_id")
    found = {store: sorted(str(held) for held in root.glob(f"{split}*{store}*"))
             for store in _STORES}
    stores = [store for store, files in found.items() if files]
    if len(stores) > 1:
        raise ValueError(
            f"{root} holds the {split} corpus as {stores}; one corpus is in one "
            f"store, so keep the one the run reads and move the rest")
    if not stores:
        raise ValueError(
            f"{root} holds no {split} corpus: {name} reads {split}.bin, "
            f"{split}*.array_record* shards or {split}*.parquet, and "
            f"`dew tokenize --val-fraction` writes the held-out split")
    files = found[stores[0]]
    if stores[0] == ".bin":
        return TokenBytes(files[0])
    if stores[0] == ".array_record":
        return TokenRecords(files, field=field, dtype=dtype, eos_id=eos_id)
    return TokenColumn(files, column=field, dtype=dtype, eos_id=eos_id)


class TokenWindowSource:
    """Reads fixed `seq_len + 1` windows over a token corpus, by index.

    Record i starts at `i * stride`. With the default stride of `seq_len`,
    the last token of one window is the first of the next. A stride of one
    reads every complete contiguous window; incomplete tails are excluded.
    """

    def __init__(self, tokens: TokenSource, seq_len: int, *, stride: int | None = None):
        if seq_len < 1:
            raise ValueError(f"seq_len must be at least 1, got {seq_len}")
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = seq_len if stride is None else stride
        if self.stride < 1:
            raise ValueError(f"stride must be at least 1, got {self.stride}")
        if len(tokens) < seq_len + 1:
            raise ValueError(
                f"{tokens!r} holds {len(tokens)} tokens, too few for even "
                f"one window of seq_len {seq_len}"
            )

    def __repr__(self) -> str:
        # The description a saved position compares against (`describe`).
        stride = "" if self.stride == self.seq_len else f", stride={self.stride}"
        return f"TokenWindowSource(tokens={self.tokens!r}, seq_len={self.seq_len}{stride})"

    def __len__(self) -> int:
        return (len(self.tokens) - self.seq_len - 1) // self.stride + 1

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        # A memmap slice past its end yields an empty array instead of an error,
        # so the bounds are checked here.
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.stride
        return {"text": self.tokens[start:start + self.seq_len + 1].astype(np.int32)}


class TokenDocumentSource:
    """Reads one document per record over a token corpus, by index.

    A document is the span from after the previous `eos_id` through its own,
    so the eos tokens are the record separators. The tail after the last eos
    is a document too, and a split with no eos at all is one document. A
    train/val split cuts the stream wherever the token fraction falls, and
    --pack closes input files instead of that cut, so the head of the stream
    can carry no boundary while the tokens are still a document.

    `eos_id` is the corpus's own unless one is given; without it the stream
    has no boundaries to find. Finding them reads the corpus once at
    construction, after which a worker touches only the span it is asked for.

    `chunk_len` cuts every document into consecutive records of at most that
    many tokens, as the packer would (`dew.data.tokens.DocumentChunks`), so a
    record of a document longer than a window reads its own span rather than
    the whole document.
    """

    def __init__(self, tokens: TokenSource, eos_id: int | None = None, *,
                 chunk_len: int | None = None):
        self.tokens = tokens
        found = tokens.eos_id if eos_id is None else eos_id
        if found is None:
            raise ValueError(
                f"{tokens!r} records no eos_id: document boundaries are the "
                "eos tokens `dew tokenize` writes with --pack")
        self.eos_id = int(found)

        held = tokens[0:len(tokens)]
        ends = (np.flatnonzero(held == self.eos_id) + 1).astype(np.int64)
        if len(ends) == 0 or ends[-1] < len(held):
            ends = np.append(ends, len(held))
        starts = np.concatenate([[0], ends[:-1]])
        self.chunk_len = chunk_len
        if chunk_len is not None:
            counts = -(-(ends - starts) // chunk_len)
            first = np.repeat(starts, counts)
            within = np.arange(len(first), dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
            starts = first + within * chunk_len
            ends = np.minimum(starts + chunk_len, np.repeat(ends, counts))
        # Exclusive span ends; record i is tokens[starts[i] : ends[i]].
        self._ends = ends
        self._starts = starts
        self.lengths = self._ends - self._starts

    def __repr__(self) -> str:
        # The description a saved position compares against (`describe`); the
        # packed loader plans its order over these documents.
        return (f"TokenDocumentSource(tokens={self.tokens!r}, eos_id={self.eos_id}, "
                f"chunk_len={self.chunk_len})")

    def __len__(self) -> int:
        return len(self._ends)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        window = self.tokens[int(self._starts[index]):int(self._ends[index])]
        return {"text": window.astype(np.int32)}
