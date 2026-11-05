"""Committed token exports, their document indexes and contiguous spans."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from dew._records import boolean, json_value, record
from dew.batch import Batch
from dew._cache import dew_cache_dir
from dew.data.dataset import Ranges, Records
from dew.data.declared import Declared
from dew.data.pipeline import Dataset
from dew.data.storage import Store, local
from dew._files import write_atomically

from .base import Read, ShardManifest, _indexed_manifest, url_label
from .files import Files, export_content
from .index import ArrayIndex

_SPLIT = "train"
"""The split a token export's description, fingerprint, shards and document index have always named
at the default: an export holds one, so a run saved before reads the same plan and a store provisioned
before keeps its index. A store once read under another split's name is named "train" now."""


@dataclass(frozen=True)
class _TokenShard:
    """One shard's producer length index, its length and cumulative end columns as mmap views, and
    the place of its first document in the export."""

    columns: np.ndarray
    row_start: int

    @property
    def ends(self) -> np.ndarray:
        return self.columns[:, 1]


@dataclass(frozen=True)
class TokenRecords:
    """A token export's documents as one stream of contiguous spans, each read through the length
    index its producer wrote beside its shard.

    Index columns remain mmap views. Only the per-shard totals are combined;
    ArrayRecords owns the bounded set of open physical shards.
    """

    paths: tuple[str, ...]
    indexes: tuple[ArrayIndex, ...]
    dtype: str
    field: str
    eos_id: int | None = None
    storage_options: Mapping[str, object] | None = None
    declared: tuple[Declared, ...] = ()
    """Each shard as the export's manifest declares it (`Declared`), read as declared or refused."""

    @cached_property
    def records(self):
        from .records import ArrayRecords

        return ArrayRecords(self.paths, self.storage_options, counts=self.documents.counts,
                            declared=self.declared or None)

    @cached_property
    def documents(self) -> Ranges:
        """The documents each shard holds, one range after another."""
        return Ranges(tuple(len(index.values) for index in self.indexes))

    @cached_property
    def shards(self) -> tuple[_TokenShard, ...]:
        # Each shard's first document's place among the export's documents.
        return tuple(_TokenShard(index.values, self.documents.start(shard))
                     for shard, index in enumerate(self.indexes))

    def _document(self, index: int) -> np.ndarray:
        from .codec import decode

        payload = self.records[index]
        if not isinstance(payload, bytes):
            raise TypeError("ArrayRecord token document is not bytes")
        value = decode(payload)
        if not isinstance(value, Mapping) or self.field not in value:
            raise ValueError(f"token document lacks field {self.field!r}")
        return np.asarray(value[self.field], dtype=self.dtype)

    @cached_property
    def spans(self) -> Ranges:
        """The tokens each shard holds, one range after another."""
        return Ranges(tuple(int(shard.ends[-1]) if len(shard.ends) else 0 for shard in self.shards))

    def __len__(self) -> int:
        return len(self.spans)

    def __getitem__(self, span: slice) -> np.ndarray:
        start, stop, step = span.indices(len(self))
        if step != 1:
            raise ValueError("token stores are read in contiguous spans")
        pieces = []
        first = self.spans.locate(start)[0] if start < len(self) else len(self.shards)
        for index in range(first, len(self.shards)):
            beginning = self.spans.start(index)
            if beginning >= stop:
                break
            shard = self.shards[index]
            ends = shard.ends
            for row in range(int(np.searchsorted(ends, max(start - beginning, 0), side="right")), len(ends)):
                row_start = beginning + (int(ends[row - 1]) if row else 0)
                if row_start >= stop:
                    break
                tokens = self._document(shard.row_start + row)
                if len(tokens) != int(shard.columns[row, 0]):
                    raise ValueError("token document differs from the producer's length index")
                pieces.append(tokens[max(start - row_start, 0) : stop - row_start])
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=self.dtype)

    def __getstate__(self) -> Mapping[str, object]:
        """Workers reopen producer indexes instead of pickling their mapped columns."""
        return {name: value for name, value in self.__dict__.items() if name not in ("shards", "spans")}


def _earlier_ends(tokens: TokenRecords) -> np.ndarray:
    """Where a Dew before row 3028 ended a token store's documents, which it read by their EOS: just past
    each, or at the stream's end for an unterminated tail; an empty stream holds none. Read only to resume a
    plan it saved (`TokenFiles.earlier`). Carried to 1.0 (ledger 1552)."""
    block = 1 << 24
    found = [np.flatnonzero(tokens[start:start + block] == tokens.eos_id) + start + 1
             for start in range(0, len(tokens), block)]
    ends = np.concatenate([np.zeros(0, np.int64), *found]).astype(np.int64)
    return ends if (ends[-1] if len(ends) else 0) == len(tokens) else np.append(ends, len(tokens))


@dataclass(frozen=True)
class TokenFiles(Dataset):
    """A committed token export (`dew.data.text.export_tokens`, `dew data export --tokenizer`): its
    documents of token ids, the metadata its manifest states and where its documents end, each a record
    its producer wrote, as its length index states it (`document_ends`). An export holds one split;
    validation tokens are an export of their own."""

    url: str
    storage_options: Mapping[str, object] | None = None
    field: str | None = None

    @cached_property
    def committed(self) -> Mapping[str, object]:
        """The export's committed manifest, read through this store's own storage options."""
        committed = Files(self.url, storage_options=self.storage_options).committed
        if committed is None:
            raise FileNotFoundError(f"{self.url} holds no committed token export; write one with "
                                    "dew.data.text.export_tokens or `dew data export --tokenizer`")
        return committed

    def label(self) -> str:
        return url_label(self.url)

    def describe(self) -> str:
        return f"tokens:{self.url}:{_SPLIT}:field={self.field}"

    @cached_property
    def meta(self) -> Mapping[str, object]:
        metadata = self.committed.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"token export metadata is not a mapping at {self.url}")
        return metadata

    @cached_property
    def tokens(self) -> TokenRecords:
        eos = self.meta.get("eos_id")
        if eos is not None and not isinstance(eos, int):
            raise ValueError("token metadata eos_id must be an integer or null")
        committed = self.committed
        if committed["format"] != "array_record":
            raise ValueError("token exports must use ArrayRecord")
        root = Store(self.url, self.storage_options)
        indexes = []
        paths = []
        declared = []
        entries = committed["files"]
        if not isinstance(entries, list):
            raise ValueError("token export files must be a list")
        for entry in entries:
            if entry["lengths"] is None:
                raise FileNotFoundError(
                    f"token export needs the producer's lengths; export with length_field="
                    f"{self.field or 'text'!r}"
                )
            lengths = root.child(entry["lengths"]["path"])
            index = ArrayIndex(lengths.url, self.storage_options, Declared(lengths, entry["lengths"]))
            if index.values.ndim != 2 or index.values.shape[1] != 2:
                raise ValueError("token export index needs length and cumulative end columns")
            indexes.append(index)
            paths.append(root.child(entry["path"]).url)
            declared.append(Declared(root.child(entry["path"]), entry))
        # The field of each record that holds its tokens, "text" where the export names none.
        return TokenRecords(tuple(paths), tuple(indexes), str(self.meta.get("dtype", "int32")),
                            self.field or "text", eos, self.storage_options, tuple(declared))

    def stated(self) -> Mapping[str, object]:
        """What the store's manifest says of its tokens, none of them read (`Dataset.facts`): their
        tokenizer, vocabulary and EOS, as token ids under "text"; an export not here states nothing."""
        if Files(self.url, storage_options=self.storage_options).committed is None:
            return {}
        return {**self.meta, "token_store": True}

    def metadata(self) -> Mapping[str, object]:
        # Its records are token spans, which carry no count of Dew's.
        return {**self.meta, "tokens": len(self.tokens), "token_store": True, "count_rules": []}

    def fingerprint(self) -> Mapping[str, object]:
        return self._fingerprint

    @cached_property
    def _fingerprint(self) -> Mapping[str, object]:
        """Tokens and metadata, by the export's committed manifest and the files it names, each by its
        content and place in the manifest, not the name its writer gave it (`export_content`), so two
        exports of the same documents agree: the documents are the producer's records, which its length
        index states, as `documents` says, which no Dew before row 3028 did: a plan it saved names the
        store otherwise, and resumes where it read the same documents (`read_alike`,
        `dew.data.plan.earlier_plan`). The keys an export's fingerprint has always had (`_SPLIT`)."""
        store, _ = Store(self.url, self.storage_options).resolve()
        # A local store is named by its content, so hosts that mounted it at different paths agree.
        place = {} if local(store) else {"url": self.url}
        return {
            "store": "tokens",
            **place,
            "split": _SPLIT,
            "metadata": self.meta,
            "field": self.field,
            # An export of no rows has no shards to name.
            "shards": Files(self.url, storage_options=self.storage_options).fingerprint() if self.tokens.paths
            else [],
            "index": None,
            "shard_rows": None,
            "producer": export_content(self.committed),
            "documents": "records",
        }

    def read_alike(self) -> bool:
        """Whether a Dew before row 3028, which read a token store's documents by its EOS, read this store's
        as its producer's records: stating no EOS, one record holding every token; stating one, each record
        ended by that EOS and holding no other (`_earlier_ends`), a scan of every token of the field read,
        kept in Dew's cache under the export's content (`export_content`), which no write changes, and that
        field. Read only to resume a plan that Dew saved (`dew.data.plan.earlier_plan`). Carried to 1.0
        (ledger 1552)."""
        return self._read_alike

    @cached_property
    def _read_alike(self) -> bool:
        tokens, ends = self.tokens, self.document_ends()
        if tokens.eos_id is None:
            return np.array_equal(ends, [len(tokens)] if len(tokens) else [])
        read = {"export": export_content(self.committed), "field": tokens.field}
        content = json.dumps(json_value(read, "token export's field"), sort_keys=True)
        named = hashlib.sha256(content.encode()).hexdigest()
        kept = Path(dew_cache_dir()) / "token-documents" / f"{named}.json"
        if kept.exists():
            held = record(json.loads(kept.read_text()), "token documents")
            return boolean(held["read_alike"], "token documents read alike")
        alike = bool(np.array_equal(_earlier_ends(tokens), ends))
        kept.parent.mkdir(parents=True, exist_ok=True)
        write_atomically(kept, json.dumps({"read_alike": alike}))
        return alike

    def supports(self, read: Read) -> bool:
        return True

    def can_provision(self, read: Read) -> bool:
        return True

    def provision(self) -> TokenFiles:
        """Nothing to write: the producer's length index, which every export carries, states where each
        document ends (`document_ends`)."""
        return self

    def records(self, read: Read) -> Records:
        """Each document by index (`document_ends`), in every read."""
        return _TokenDocuments(self.tokens, self.document_ends())

    def document_ends(self) -> np.ndarray:
        """Where each document ends among the export's tokens: each record its producer wrote, by its
        length index (`TokenRecords.shards`), a shard's after the shard before it, an EOS its tokens spell
        within it or not, as the documents it exported were (`dew.data.text.export_tokens`). An empty
        record holds no document, so none is read of it."""
        return self._ends

    @cached_property
    def _ends(self) -> np.ndarray:
        tokens = self.tokens
        each = (shard.ends.astype(np.int64) + tokens.spans.start(index)
                for index, shard in enumerate(tokens.shards))
        ends = np.concatenate([np.zeros(0, np.int64), *each])
        return ends[np.diff(ends, prepend=0) > 0]

    def manifest(self) -> ShardManifest:
        return _indexed_manifest(self, self.url, 1024)

    def lengths(self) -> np.ndarray:
        return np.diff(self.document_ends(), prepend=0)


@dataclass(frozen=True)
class _TokenDocuments:
    """A token store's documents by index, each the span its index names."""

    tokens: TokenRecords
    ends: np.ndarray

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, index: int) -> Batch:
        start = int(self.ends[index - 1]) if index else 0
        return {"text": self.tokens[start : int(self.ends[index])].astype(np.int32)}
