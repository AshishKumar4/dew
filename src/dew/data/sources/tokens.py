"""Produced token stores, their document indexes and contiguous spans."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property

import grain.python as grain
import numpy as np
from fsspec.spec import AbstractFileSystem

from dew.data.dataset import Batch, Records
from dew.data.pipeline import Dataset
from dew.data.storage import Store, content_version, local

from .base import Read, Shard, ShardManifest
from .files import Files
from .index import ArrayIndex


def _version(fs: AbstractFileSystem, path: str) -> str:
    """A file's version as every host holding a copy names it (`content_version`)."""
    return content_version(fs, path, int(fs.info(path)["size"])) if local(fs) else str(fs.ukey(path))


@dataclass(frozen=True)
class TokenBytes:
    """A contiguous token store. Local spans use mmap; URL spans use range reads."""

    url: str
    dtype: str = "uint16"
    eos_id: int | None = None
    storage_options: Mapping[str, object] | None = None

    @cached_property
    def count(self) -> int:
        size = Store(self.url, self.storage_options).size()
        width = np.dtype(self.dtype).itemsize
        if size % width:
            raise ValueError(f"token file {self.url} size {size} is not divisible by dtype {self.dtype}")
        return size // width

    def __len__(self) -> int:
        return self.count

    @cached_property
    def local(self) -> np.memmap | None:
        fs, path = Store(self.url, self.storage_options).resolve()
        protocols = (fs.protocol,) if isinstance(fs.protocol, str) else fs.protocol
        return np.memmap(path, dtype=self.dtype, mode="r") if "file" in protocols and self.count else None

    def __getitem__(self, span: slice) -> np.ndarray:
        start, stop, step = span.indices(self.count)
        if step != 1:
            raise ValueError("token stores are read in contiguous spans")
        if self.local is not None:
            return np.asarray(self.local[start:stop])
        if stop <= start:
            return np.empty(0, dtype=self.dtype)
        fs, path = Store(self.url, self.storage_options).resolve()
        width = np.dtype(self.dtype).itemsize
        return np.frombuffer(fs.cat_file(path, start=start * width, end=stop * width), dtype=self.dtype)

    def __getstate__(self) -> Mapping[str, object]:
        return {key: value for key, value in self.__dict__.items() if key != "local"}


@dataclass(frozen=True)
class TokenRecords:
    """ArrayRecord token documents with a producer's authoritative length index."""

    paths: tuple[str, ...]
    lengths: np.ndarray | None
    dtype: str = "int32"
    field: str | None = None
    eos_id: int | None = None
    storage_options: Mapping[str, object] | None = None
    record_codec: bool = False
    cumulative: np.ndarray | None = None
    shard_rows: tuple[int, ...] | None = None

    @cached_property
    def records(self):
        from .records import ArrayRecords

        return ArrayRecords(self.paths, self.storage_options, counts=self.shard_rows)

    @cached_property
    def ends(self) -> np.ndarray:
        if self.cumulative is not None:
            return self.cumulative
        if self.lengths is None:
            raise FileNotFoundError(
                "ArrayRecord token spans need the producer's length index; "
                "set provision=True or choose read='memory'"
            )
        return np.cumsum(self.lengths, dtype=np.int64)

    def materialize_lengths(self) -> np.ndarray:
        if self.lengths is None:
            self.__dict__["lengths"] = np.asarray(
                [len(self.piece(index)) for index in range(len(self.records))], dtype=np.int64
            )
        return self.__dict__["lengths"]

    def piece(self, index: int) -> np.ndarray:
        from .codec import decode, unpack_dict_of_byte_arrays

        payload = self.records[index]
        if not isinstance(payload, bytes):
            raise TypeError("ArrayRecord token document is not bytes")
        if self.record_codec:
            value = decode(payload)
            if self.field is None or not isinstance(value, Mapping) or self.field not in value:
                raise ValueError(f"token document lacks field {self.field!r}")
            values = np.asarray(value[self.field], dtype=self.dtype)
        elif self.field is not None:
            payload = unpack_dict_of_byte_arrays(payload)[self.field]
            values = np.frombuffer(payload, dtype=self.dtype)
        else:
            values = np.frombuffer(payload, dtype=self.dtype)
        if self.lengths is not None and len(values) != int(self.lengths[index]):
            raise ValueError(f"token document {index} differs from the producer's length index")
        return values

    def __len__(self) -> int:
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, span: slice) -> np.ndarray:
        start, stop, step = span.indices(len(self))
        if step != 1:
            raise ValueError("token stores are read in contiguous spans")
        pieces = []
        for index in range(int(np.searchsorted(self.ends, start, side="right")), len(self.ends)):
            begin = 0 if index == 0 else int(self.ends[index - 1])
            if begin >= stop:
                break
            pieces.append(self.piece(index)[max(start - begin, 0) : stop - begin])
        return np.concatenate(pieces) if pieces else np.empty(0, self.dtype)


@dataclass(frozen=True)
class TokenShards:
    """Produced token spans over per-shard memory-mapped length and end columns."""

    paths: tuple[str, ...]
    indexes: tuple[ArrayIndex, ...]
    dtype: str = "int32"
    field: str = "text"
    eos_id: int | None = None
    storage_options: Mapping[str, object] | None = None

    @cached_property
    def shard_ends(self) -> np.ndarray:
        totals = [int(index.values[-1, 1]) if len(index.values) else 0 for index in self.indexes]
        return np.cumsum(totals, dtype=np.int64)

    @cached_property
    def row_ends(self) -> np.ndarray:
        return np.cumsum([len(index.values) for index in self.indexes], dtype=np.int64)

    @cached_property
    def records(self):
        from .records import ArrayRecords

        return ArrayRecords(
            self.paths,
            self.storage_options,
            record_codec=True,
            counts=[len(index.values) for index in self.indexes],
        )

    def __len__(self) -> int:
        return int(self.shard_ends[-1]) if len(self.shard_ends) else 0

    def __getitem__(self, span: slice) -> np.ndarray:
        start, stop, step = span.indices(len(self))
        if step != 1:
            raise ValueError("token stores are read in contiguous spans")
        pieces = []
        for shard in range(int(np.searchsorted(self.shard_ends, start, side="right")), len(self.paths)):
            beginning = int(self.shard_ends[shard - 1]) if shard else 0
            if beginning >= stop:
                break
            row_base = int(self.row_ends[shard - 1]) if shard else 0
            columns = self.indexes[shard].values
            ends = columns[:, 1]
            for row in range(int(np.searchsorted(ends, max(start - beginning, 0), side="right")), len(ends)):
                row_start = beginning + (int(ends[row - 1]) if row else 0)
                if row_start >= stop:
                    break
                record = self.records[row_base + row]
                if not isinstance(record, Mapping):
                    raise ValueError("exported token document is not a named record")
                tokens = np.asarray(record[self.field], dtype=self.dtype)
                if len(tokens) != int(columns[row, 0]):
                    raise ValueError("exported token document differs from its producer length")
                pieces.append(tokens[max(start - row_start, 0) : stop - row_start])
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=self.dtype)


def _document_ends(tokens: TokenBytes | TokenRecords | TokenShards) -> np.ndarray:
    """Each document ends just past its EOS, or at the stream's end for an unterminated tail."""
    block = 1 << 24
    found = [
        np.flatnonzero(tokens[start : start + block] == tokens.eos_id) + start + 1
        for start in range(0, len(tokens), block)
    ]
    ends = np.concatenate([np.zeros(0, np.int64), *found]).astype(np.int64)
    return ends if len(ends) and ends[-1] == len(tokens) else np.append(ends, len(tokens))


@dataclass(frozen=True)
class TokenFiles(Dataset):
    """A produced train/val token directory, with metadata and document boundaries."""

    url: str
    split: str = "train"
    storage_options: Mapping[str, object] | None = None
    field: str | None = None

    def with_split(self, name: str) -> TokenFiles:
        if name != self.split and Files(self.url, storage_options=self.storage_options).committed is not None:
            raise ValueError(
                f"token export {self.url!r} has one split; give an explicit TokenFiles URL for {name!r}"
            )
        return replace(self, split=name)

    def describe(self) -> str:
        return f"tokens:{self.url}:{self.split}:field={self.field}"

    @cached_property
    def array_paths(self) -> tuple[str, ...]:
        fs, path = Store(self.url, self.storage_options).resolve()
        return tuple(
            fs.unstrip_protocol(name)
            for name in sorted(fs.glob(path.rstrip("/") + f"/{self.split}*.array_record*"))
            if fs.isfile(name)
        )

    @property
    def length_index(self) -> Store:
        return Store(self.url, self.storage_options).child(f"{self.split}.lengths.npy")

    @cached_property
    def mapped_index(self) -> ArrayIndex:
        return ArrayIndex(self.length_index.url, self.storage_options)

    @property
    def document_index(self) -> Store:
        return Store(self.url, self.storage_options).child(f"{self.split}.documents.npy")

    @cached_property
    def mapped_documents(self) -> ArrayIndex:
        return ArrayIndex(self.document_index.url, self.storage_options)

    @property
    def row_manifest(self) -> Store:
        return Store(self.url, self.storage_options).child(f"{self.split}.shards.json")

    @cached_property
    def meta(self) -> Mapping[str, object]:
        committed = Files(self.url).committed
        if committed is not None:
            metadata = committed.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise ValueError(f"token export metadata is not a mapping at {self.url}")
            return metadata
        fs, path = Store(self.url, self.storage_options).child("meta.json").resolve()
        return json.loads(fs.cat_file(path)) if fs.exists(path) else {}

    @cached_property
    def tokens(self) -> TokenBytes | TokenRecords | TokenShards:
        eos = self.meta.get("eos_id")
        if eos is not None and not isinstance(eos, int):
            raise ValueError("token metadata eos_id must be an integer or null")
        committed = Files(self.url).committed
        if committed is not None:
            if committed["format"] != "array_record":
                raise ValueError("token exports must use ArrayRecord")
            root = Store(self.url, self.storage_options)
            indexes = []
            paths = []
            entries = committed["files"]
            if not isinstance(entries, list):
                raise ValueError("token export files must be a list")
            for entry in entries:
                if entry["lengths"] is None:
                    raise FileNotFoundError(
                        "token export needs the producer's lengths; export with length_field='text'"
                    )
                index = ArrayIndex(root.child(entry["lengths"]["path"]).url, self.storage_options)
                index.verify(entry["lengths"])
                if index.values.ndim != 2 or index.values.shape[1] != 2:
                    raise ValueError("token export index needs length and cumulative end columns")
                indexes.append(index)
                paths.append(root.child(entry["path"]).url)
            return TokenShards(
                tuple(paths),
                tuple(indexes),
                str(self.meta.get("dtype", "int32")),
                "text",
                eos,
                self.storage_options,
            )
        if self.array_paths:
            binary = Store(self.url, self.storage_options).child(f"{self.split}.bin")
            fs, path = binary.resolve()
            if fs.exists(path):
                raise ValueError(f"{self.url} holds {self.split} in both binary and ArrayRecord stores")
            fs, path = self.length_index.resolve()
            lengths = cumulative = None
            rows = None
            if fs.exists(path):
                index = self.mapped_index.values
                if index.ndim != 2 or index.shape[1] != 2:
                    raise ValueError(
                        "token length index needs length and cumulative end columns; provision it again"
                    )
                lengths, cumulative = index[:, 0], index[:, 1]
                rows = tuple(json.loads(self.row_manifest.read())["rows"])
            return TokenRecords(
                self.array_paths,
                lengths,
                str(self.meta.get("dtype", "uint16")),
                self.field,
                eos,
                self.storage_options,
                cumulative=cumulative,
                shard_rows=rows,
            )
        return TokenBytes(
            self.url.rstrip("/") + f"/{self.split}.bin",
            str(self.meta.get("dtype", "uint16")),
            eos,
            self.storage_options,
        )

    def metadata(self) -> Mapping[str, object]:
        facts = dict(self.meta)
        tokens = self.tokens
        if not isinstance(tokens, TokenRecords) or tokens.lengths is not None:
            facts.update(tokens=len(tokens), **{f"{self.split}_tokens": len(tokens)})
        elif f"{self.split}_tokens" in facts:
            facts["tokens"] = facts[f"{self.split}_tokens"]
        return {**facts, "token_store": True}

    def fingerprint(self) -> Mapping[str, object]:
        return self._fingerprint

    @cached_property
    def _fingerprint(self) -> Mapping[str, object]:
        """Tokens and metadata: the document index is derived from the tokens' EOS positions."""
        store, _ = Store(self.url, self.storage_options).resolve()
        # A local store is named by its content, so hosts that mounted it at different paths agree.
        place = {} if local(store) else {"url": self.url}
        if isinstance(self.tokens, (TokenRecords, TokenShards)):
            fs, path = self.length_index.resolve()
            return {
                "store": "tokens",
                **place,
                "split": self.split,
                "metadata": self.meta,
                "field": self.field,
                "shards": Files(
                    self.tokens.paths, format="array_record", storage_options=self.storage_options
                ).fingerprint(),
                "index": _version(fs, path) if fs.exists(path) else None,
                "shard_rows": self.tokens.shard_rows if isinstance(self.tokens, TokenRecords) else None,
                "producer": Files(self.url, storage_options=self.storage_options).committed,
            }
        fs, path = Store(self.tokens.url, self.storage_options).resolve()
        return {
            "store": "tokens",
            **place,
            "split": self.split,
            "metadata": self.meta,
            "field": self.field,
            "bytes": fs.size(path),
            "version": _version(fs, path),
        }

    def supports(self, read: Read) -> bool:
        if read == "memory":
            return True
        tokens = self.tokens
        return not isinstance(tokens, TokenRecords) or tokens.lengths is not None

    def can_provision(self, read: Read) -> bool:
        return True

    def provision(self) -> TokenFiles:
        """Build the indexes the store's reads need: a record store's record
        lengths, and, for a store with an EOS id, where each document ends
        (`document_ends`)."""
        tokens = self.tokens
        if isinstance(tokens, TokenRecords):
            from .records import ArrayRecords

            lengths = tokens.materialize_lengths()
            encoded = io.BytesIO()
            np.save(encoded, np.column_stack((lengths, np.cumsum(lengths))), allow_pickle=False)
            self.length_index.write(encoded.getvalue())
            counts = []
            for path in tokens.paths:
                reader = ArrayRecords((path,), self.storage_options)
                try:
                    counts.append(len(reader))
                finally:
                    reader.close()
            self.row_manifest.write(json.dumps({"rows": counts}).encode())
            self.__dict__.pop("tokens", None)
        if tokens.eos_id is not None:
            encoded = io.BytesIO()
            np.save(encoded, _document_ends(tokens), allow_pickle=False)
            self.document_index.write(encoded.getvalue())
            self.__dict__.pop("mapped_documents", None)
            self.__dict__.pop("_ends", None)
        return self

    def records(self, read: Read) -> Records:
        """Each document by index (`document_ends`), in every read; without EOS, one document."""
        if not self.supports(read):
            raise FileNotFoundError(
                f"{self.describe()} needs its producer length index; "
                "run dew data provision or choose read='memory'"
            )
        if read == "memory" and isinstance(self.tokens, TokenRecords):
            self.tokens.materialize_lengths()
        ends = self.document_ends()
        return _TokenDocuments(self.tokens, np.asarray([len(self.tokens)]) if ends is None else ends)

    def document_ends(self) -> np.ndarray | None:
        """Document ends from the persisted index or one EOS scan; None without an EOS id."""
        return self._ends

    @property
    def documented(self) -> bool:
        """Whether an index or an EOS id states the documents, without scanning their tokens."""
        fs, path = self.document_index.resolve()
        return fs.exists(path) or self.tokens.eos_id is not None

    @cached_property
    def _ends(self) -> np.ndarray | None:
        fs, path = self.document_index.resolve()
        if fs.exists(path):
            return np.asarray(self.mapped_documents.values, np.int64)
        return None if self.tokens.eos_id is None else _document_ends(self.tokens)

    def manifest(self) -> ShardManifest:
        count = len(self.records("indexed"))
        size = 1024
        return ShardManifest(
            tuple(
                Shard(f"{self.url}:{self.split}:{start}", self.url, start, min(start + size, count),
                      min(size, count - start))
                for start in range(0, count, size)
            ),
            self._read_shard,
        )

    def _read_shard(self, shard: Shard) -> grain.IterDataset:
        return grain.MapDataset.source(self.records("indexed"))[shard.start : shard.stop].to_iter_dataset()

    def lengths(self) -> np.ndarray | None:
        ends = self.document_ends()
        return None if ends is None else np.diff(ends, prepend=0)


@dataclass(frozen=True)
class _TokenDocuments:
    """A token store's documents by index, each the span its index names."""

    tokens: TokenBytes | TokenRecords | TokenShards
    ends: np.ndarray

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, index: int) -> Batch:
        start = int(self.ends[index - 1]) if index else 0
        return {"text": self.tokens[start : int(self.ends[index])].astype(np.int32)}
