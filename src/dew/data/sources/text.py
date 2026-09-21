"""Tokenized corpora as one stream of ids, and the two records cut out of it.

A corpus is a stream of token ids, whatever holds it: the `.bin` files
`tools/tokenize_text.py` writes, ArrayRecord shards of token arrays, or a
parquet column of them. `TokenSource` is that stream, read by slice, and the
three readers below are the three stores it can live in. `token_corpus`
resolves a directory to the train and validation pair a run needs, by what
the files in it are.

`TokenWindowSource` reads a record as a contiguous window of `seq_len + 1`
ids starting at `i * seq_len`, so record i's last token is record i+1's first
and the model sees every transition exactly once. There is no decoding and
no randomness here; the shuffle lives in the sampler.

`TokenDocumentSource` reads a record as one document: the span from after the
previous eos id through its own. It exists for the packed pipeline, which
cares where documents end and lets grain pack several of them into one
window. Both read through `TokenSource` and nothing else, so the same corpus
in any of the three stores gives the same windows and the same packing plan.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

# meta.json's "dtype" names numpy dtypes; uint16 covers byte tokenizers and
# most HF ones, uint32 the rest.
_DEFAULT_DTYPE = np.dtype("<u2")


@runtime_checkable
class TokenSource(Protocol):
    """A tokenized corpus as one stream of ids.

    `len` is the tokens it holds and `source[start:stop]` is that span of
    them. Both record readers below cut their records out of this and
    nothing else: a fixed window is a strided span, and a document is the
    span between two eos ids.

    `eos_id` is the id that closes a document, which only the packed reader
    needs and a corpus written without boundaries does not have. A saved
    position names the order it counts into, so a source also describes
    itself by what it reads rather than by its address in this process.
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


class TokenBytes:
    """A flat `.bin` of token ids, memmapped.

    The dtype comes from the sibling `meta.json` when present (the tokenize
    tool records it there), else uint16. The file is never loaded into
    memory: a worker reads only the span it is asked for.
    """

    def __init__(self, path: str, eos_id: int | None = None):
        self.path = str(path)
        meta = _meta(Path(self.path).parent)
        self.dtype = _dtype(meta)
        self.vocab_size = _recorded(meta, "vocab_size")
        self.eos_id = eos_id if eos_id is not None else _recorded(meta, "eos_id")
        self._tokens = np.memmap(self.path, dtype=self.dtype, mode="r")

    def __repr__(self) -> str:
        return f"TokenBytes(path={self.path!r})"

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, span: slice) -> np.ndarray:
        return np.asarray(self._tokens[span])

    def __getstate__(self):
        # The memmap does not survive grain's pickle round trip to workers;
        # the path and dtype are enough to reopen it there.
        state = dict(self.__dict__)
        state["_tokens"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokens = np.memmap(self.path, dtype=self.dtype, mode="r")


class _Sharded:
    """Token arrays read as one stream, in the order they are stored.

    A corpus of records or rows is the concatenation of them, so where a
    span falls is a binary search over their lengths, read once at
    construction. A span that crosses a boundary reads both pieces and joins
    them; one that does not reads one. Only the pieces a record covers are
    read, so a corpus is no more in memory than the memmap is.
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


class TokenRecords(_Sharded):
    """Token arrays in ArrayRecord shards, read as one stream.

    Each record holds one array of ids, as its raw bytes or under `field` of
    the packed dict `dew.data.images.pack_dict_of_byte_arrays` writes. The
    records are the corpus in file order, so a tokenizer that wrote one
    document per record and one that wrote fixed blocks read back the same.
    """

    def __init__(self, paths: Sequence[str], *, field: str | None = None,
                 dtype: np.dtype = _DEFAULT_DTYPE, eos_id: int | None = None):
        if not paths:
            raise ValueError("TokenRecords needs at least one arrayrecord file")
        self.paths = [str(path) for path in paths]
        self.field = field
        self.eos_id = None if eos_id is None else int(eos_id)
        self._records = _array_records(self.paths)
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

    def __repr__(self) -> str:
        return f"TokenRecords(paths={self.paths!r}, field={self.field!r})"

    def __getstate__(self):
        # The arrayrecord reader holds open files, which do not survive
        # grain's pickle; the paths reopen it in the worker.
        state = dict(self.__dict__)
        state["_records"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._records = _array_records(self.paths)


def _array_records(paths: Sequence[str]):
    """The arrayrecord reader over `paths`, imported on use."""
    from array_record.python.array_record_data_source import ArrayRecordDataSource

    return ArrayRecordDataSource(list(paths))


class TokenColumn(_Sharded):
    """One column of token ids in parquet files, read as one stream.

    The column holds a list of integers per row, which is what a tokenized
    dataset written with `to_parquet` holds. The rows are the corpus in file
    order; `column` names which column to read, and a file that holds one
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


def token_corpus(path: str | None, name: str, *, field: str | None = None
                 ) -> tuple[TokenSource, TokenSource]:
    """The `(train, val)` corpora of a tokenized directory, both required.

    A split's files are the ones named for it, and their suffix says which
    store they are in: `train.bin`, `train*.array_record*` shards, or
    `train*.parquet`. Reading train in val's place would score the
    validation pass on the windows the model trains on, so a directory
    holding one without the other is refused rather than halved.
    """
    if not path:
        raise ValueError(
            f"{name} needs path= set to the directory tools/tokenize_text.py wrote")
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
            f"tools/tokenize_text.py --val-fraction writes the held-out split")
    files = found[stores[0]]
    if stores[0] == ".bin":
        return TokenBytes(files[0])
    if stores[0] == ".array_record":
        return TokenRecords(files, field=field, dtype=dtype, eos_id=eos_id)
    return TokenColumn(files, column=field, dtype=dtype, eos_id=eos_id)


class TokenWindowSource:
    """Fixed `seq_len + 1` windows over a token corpus, by index.

    Record i is `tokens[i * seq_len : i * seq_len + seq_len + 1]`, so the
    last token of one window is the first of the next and the model sees
    every transition exactly once.
    """

    def __init__(self, tokens: TokenSource, seq_len: int):
        if seq_len < 1:
            raise ValueError(f"seq_len must be at least 1, got {seq_len}")
        self.tokens = tokens
        self.seq_len = seq_len
        if len(tokens) < seq_len + 1:
            raise ValueError(
                f"{tokens!r} holds {len(tokens)} tokens, too few for even "
                f"one window of seq_len {seq_len}"
            )

    def __repr__(self) -> str:
        # A saved data position names the order it counts into by naming its
        # source, and a resume refuses a description it cannot match, so this
        # describes the corpus, not an address in this process.
        return f"TokenWindowSource(tokens={self.tokens!r}, seq_len={self.seq_len})"

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        # A memmap slice past its end yields an empty array instead of an error,
        # so the bounds are checked here.
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.seq_len
        return {"text": self.tokens[start:start + self.seq_len + 1].astype(np.int32)}


class TokenDocumentSource:
    """One document per record over a token corpus, by index.

    A document is the span from after the previous `eos_id` through its own,
    so the eos tokens are the record separators. The tail after the last eos
    is a document too, and a split with no eos at all is one document. A
    train/val split cuts the stream wherever the token fraction falls, and
    --pack closes input files instead of that cut, so the head of the stream
    can carry no boundary while the tokens are still a document.

    `eos_id` is the corpus's own unless one is given; without it the stream
    has no boundaries to find. Finding them reads the corpus once at
    construction; after that a worker touches only the span it is asked for.
    """

    def __init__(self, tokens: TokenSource, eos_id: int | None = None):
        self.tokens = tokens
        found = tokens.eos_id if eos_id is None else eos_id
        if found is None:
            raise ValueError(
                f"{tokens!r} records no eos_id: document boundaries are the "
                "eos tokens tools/tokenize_text.py writes with --pack")
        self.eos_id = int(found)

        held = tokens[0:len(tokens)]
        ends = (np.flatnonzero(held == self.eos_id) + 1).astype(np.int64)
        if len(ends) == 0 or ends[-1] < len(held):
            ends = np.append(ends, len(held))
        # Exclusive span ends; record i is tokens[starts[i] : ends[i]].
        self._ends = ends
        self._starts = np.concatenate([[0], ends[:-1]])
        self.lengths = self._ends - self._starts

    def __repr__(self) -> str:
        # A saved data position names the order it counts into, and the packed
        # loader's order is planned over these documents, so this describes the
        # corpus rather than an address in the process that wrote the position.
        return f"TokenDocumentSource(tokens={self.tokens!r}, eos_id={self.eos_id})"

    def __len__(self) -> int:
        return len(self._ends)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        window = self.tokens[int(self._starts[index]):int(self._ends[index])]
        return {"text": window.astype(np.int32)}
