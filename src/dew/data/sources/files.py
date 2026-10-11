"""Indexed files and checkpointable physical-shard readers over filesystem URLs."""

from __future__ import annotations

import bisect
import csv
import io
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import PurePosixPath
from typing import BinaryIO, Literal

import grain.python as grain
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs

from dew.data.dataset import Batch, Records
from dew.data.storage import Store

from .base import Read, Shard, ShardManifest

type Format = Literal["jsonl", "text", "csv", "parquet", "arrow", "array_record", "files", "webdataset"]


def _format(url: str) -> Format:
    suffix = PurePosixPath(url).suffix.lower()
    formats: dict[str, Format] = {
        ".jsonl": "jsonl",
        ".txt": "text",
        ".csv": "csv",
        ".parquet": "parquet",
        ".arrow": "arrow",
        ".feather": "arrow",
        ".array_record": "array_record",
        ".tar": "webdataset",
    }
    if suffix not in formats:
        raise ValueError(f"source {url!r} needs format=; a URL does not declare its record format")
    return formats[suffix]


def _arrow_filesystem(fs) -> pafs.FileSystem:
    protocols = (fs.protocol,) if isinstance(fs.protocol, str) else fs.protocol
    if "file" in protocols:
        return pafs.LocalFileSystem()
    if "s3" in protocols:
        from urllib.parse import urlsplit

        endpoint = fs.client_kwargs.get("endpoint_url") or fs.endpoint_url
        parsed = urlsplit(endpoint) if endpoint else None
        return pafs.S3FileSystem(
            access_key=fs.key,
            secret_key=fs.secret,
            session_token=fs.token,
            anonymous=fs.anon,
            region=fs.client_kwargs.get("region_name"),
            endpoint_override=parsed.netloc if parsed else None,
            scheme=parsed.scheme if parsed else "https",
        )
    if "gs" in protocols and fs.token is None:
        return pafs.GcsFileSystem()
    return pafs.PyFileSystem(pafs.FSSpecHandler(fs))


def _fragment_rows(fragment) -> int:
    if isinstance(fragment, ds.ParquetFileFragment):
        return sum(group.num_rows for group in fragment.row_groups)
    return fragment.count_rows()


@dataclass(frozen=True)
class Files:
    """Files at URLs, with producer indexes or explicit provisioning for text.

    Parquet partitions are discovered with Arrow's Hive partitioning. A
    predicate runs within each streamed fragment; indexed predicates need
    a provisioned accepted-row index and are refused here.
    """

    url: str | tuple[str, ...]
    format: Format | None = None
    index: str | None = None
    columns: tuple[str, ...] | None = None
    filter: ds.Expression | None = None
    partitioning: str | None = "hive"
    shard_bytes: int = 64 * 2**20
    storage_options: Mapping[str, object] | None = field(default=None, repr=False)
    spark: bool = False

    @cached_property
    def kind(self) -> Format:
        if self.format is None and self.committed is not None:
            return _format("data." + str(self.committed["format"]))
        return self.format or _format(self.url if isinstance(self.url, str) else self.url[0])

    @cached_property
    def committed(self) -> Mapping[str, object] | None:
        if not isinstance(self.url, str):
            return None
        store = Store(self.url, self.storage_options).child("_dew_manifest.json")
        fs, path = store.resolve()
        return json.loads(fs.cat_file(path)) if fs.exists(path) else None

    @cached_property
    def paths(self) -> tuple[str, ...]:
        if self.committed is not None:
            root = Store(str(self.url), self.storage_options)
            entries = self.committed["files"]
            if not isinstance(entries, list):
                raise ValueError(f"export manifest has invalid files at {self.url}")
            return tuple(root.child(str(entry["path"])).url for entry in entries)
        urls = (self.url,) if isinstance(self.url, str) else self.url
        found = []
        for url in urls:
            fs, path = Store(url, self.storage_options).resolve()
            if self.spark and not fs.exists(path.rstrip("/") + "/_SUCCESS"):
                raise ValueError(f"Spark source {url!r} is uncommitted: _SUCCESS is missing")
            if any(character in path for character in "*?["):
                names = sorted(fs.glob(path))
            elif fs.isdir(path):
                names = sorted(
                    name
                    for name in fs.find(path)
                    if not any(part.startswith(("_", ".")) for part in PurePosixPath(name[len(path) :]).parts)
                )
                extensions = {
                    "parquet": ".parquet",
                    "arrow": ".arrow",
                    "jsonl": ".jsonl",
                    "text": ".txt",
                    "csv": ".csv",
                    "array_record": ".array_record",
                    "webdataset": ".tar",
                }
                if self.kind != "files":
                    names = [name for name in names if name.endswith(extensions[self.kind])]
            else:
                names = [path]
            found.extend(fs.unstrip_protocol(name) for name in names if fs.isfile(name))
        if not found:
            raise FileNotFoundError(f"source {self.url!r} matched no {self.kind} files")
        return tuple(found)

    def describe(self) -> str:
        return f"files:{self.kind}:{self.url}"

    def snapshot(self) -> Mapping[str, object]:
        files = []
        indexes = (
            ()
            if self.kind not in ("jsonl", "text", "csv")
            else (self.index,)
            if self.index
            else tuple(self._index_url(url) for url in self.paths)
        )
        for url in (*self.paths, *indexes):
            fs, path = Store(url, self.storage_options).resolve()
            if url in indexes and not fs.exists(path):
                continue
            metadata = fs.info(path)
            files.append(
                {
                    "uri": url,
                    "bytes": metadata["size"],
                    "version": metadata.get(
                        "ETag", metadata.get("etag", metadata.get("generation", fs.ukey(path)))
                    ),
                }
            )
        return {
            "store": self.kind,
            "files": files,
            "columns": self.columns,
            "filter": str(self.filter),
            "partitioning": self.partitioning,
            "transport": Store(self.paths[0], self.storage_options).describe() if self.paths else {},
        }

    def supports(self, read: Read) -> bool:
        if read != "indexed" or self.kind in ("files", "array_record"):
            return True
        if self.filter is not None:
            return False
        if self.kind in ("arrow", "parquet"):
            return True
        if self.kind == "webdataset":
            return False
        indexes = (self.index,) if self.index else tuple(self._index_url(path) for path in self.paths)
        return all(
            Store(url, self.storage_options)
            .resolve()[0]
            .exists(Store(url, self.storage_options).resolve()[1])
            for url in indexes
        )

    @staticmethod
    def _index_url(url: str) -> str:
        return url.rsplit(".", 1)[0] + ".index.npy"

    def provision(self) -> Files:
        if self.kind not in ("jsonl", "text", "csv"):
            raise ValueError(f"{self.describe()} has no provisionable offset index")
        for url in self.paths:
            fs, path = Store(url, self.storage_options).resolve()
            offsets = []
            with fs.open(path, "rb") as file:
                if self.kind == "csv":
                    reader = csv.DictReader(_CsvLines(file))
                    _ = reader.fieldnames
                    while True:
                        start = file.tell()
                        try:
                            next(reader)
                        except StopIteration:
                            break
                        offsets.append((0, start, file.tell() - start))
                else:
                    while True:
                        start = file.tell()
                        line = file.readline()
                        if not line:
                            break
                        if self.kind == "jsonl":
                            if not line.strip():
                                continue
                            json.loads(line)
                        offsets.append((0, start, len(line)))
            encoded = io.BytesIO()
            np.save(
                encoded,
                np.array(offsets, dtype=[("shard", "<u4"), ("offset", "<u8"), ("bytes", "<u8")]),
                allow_pickle=False,
            )
            Store(self._index_url(url), self.storage_options).write(encoded.getvalue())
        self.__dict__.pop("paths", None)
        return self

    def records(self, read: Read) -> Records:
        if not self.supports(read):
            raise FileNotFoundError(
                f"{self.describe()} cannot read {read}; provision its offset or accepted "
                "index with dew data provision, or choose read='streamed'"
            )
        if self.kind == "files":
            return _Paths(self.paths)
        if self.kind in ("parquet", "arrow"):
            rows = _ArrowRecords(self)
        elif self.kind == "array_record":
            from .records import ArrayRecords

            rows = ArrayRecords(self.paths, self.storage_options, record_codec=self.committed is not None)
        elif read == "memory":
            return tuple(row for shard in self.manifest().shards for row in self._read_shard(shard))
        else:
            rows = _IndexedLines(self)
        return tuple(rows[index] for index in range(len(rows))) if read == "memory" else rows

    @cached_property
    def arrow(self) -> ds.Dataset:
        fs, _ = Store(self.paths[0], self.storage_options).resolve()
        paths = [Store(url, self.storage_options).resolve()[1] for url in self.paths]
        return ds.dataset(
            paths,
            filesystem=_arrow_filesystem(fs),
            format="ipc" if self.kind == "arrow" else "parquet",
            partitioning=self.partitioning,
        )

    @cached_property
    def fragments(self):
        if not self.paths:
            return ()
        return tuple(
            piece
            for fragment in self.arrow.get_fragments()
            for piece in (fragment.split_by_row_group() if self.kind == "parquet" else (fragment,))
        )

    def manifest(self) -> ShardManifest:
        shards = []
        if self.kind in ("parquet", "arrow"):
            for number, fragment in enumerate(self.fragments):
                rows = None if self.filter is not None else _fragment_rows(fragment)
                shards.append(Shard(f"{fragment.path}:{number}", fragment.path, number, number + 1, rows))
        elif self.kind == "array_record":
            shards.extend(Shard(url, url) for url in self.paths)
        else:
            for url in self.paths:
                size = Store(url, self.storage_options).size()
                step = self.shard_bytes if self.kind in ("jsonl", "text") else max(size, 1)
                if step < 1:
                    raise ValueError("file shard_bytes must be positive")
                shards.extend(
                    Shard(f"{url}:{start}", url, start, min(start + step, size))
                    for start in range(0, size, step)
                )
        return ShardManifest(tuple(shards), self._read_shard)

    def _read_shard(self, shard: Shard) -> grain.IterDataset:
        if self.kind in ("parquet", "arrow"):
            return grain.MapDataset.source(_ArrowFragment(self, shard.start)).to_iter_dataset()
        if self.kind == "array_record":
            from .records import ArrayRecords

            return grain.MapDataset.source(
                ArrayRecords((shard.uri,), self.storage_options, record_codec=self.committed is not None)
            ).to_iter_dataset()
        if self.kind == "webdataset":
            from .webdataset import TarRows

            return TarRows(shard.uri, self.storage_options)
        if self.kind == "files":
            return grain.MapDataset.source(_Paths((shard.uri,))).to_iter_dataset()
        return _Lines(self, shard)

    def metadata(self) -> Mapping[str, object]:
        if self.committed is not None:
            metadata = self.committed.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise ValueError("export metadata must be a mapping")
            return {**metadata, "rows": self.committed["rows"], "format": self.committed["format"]}
        if self.kind in ("parquet", "arrow"):
            schema = self.arrow.schema.metadata or {}
            details = json.loads(schema[b"huggingface"]) if b"huggingface" in schema else {}
            features = details.get("info", {}).get("features", {})
            names = {
                name: tuple(feature["names"])
                for name, feature in features.items()
                if isinstance(feature, Mapping) and feature.get("_type") == "ClassLabel"
            }
            counts = (
                {"rows": sum(_fragment_rows(fragment) for fragment in self.fragments)}
                if self.filter is None
                else {}
            )
            return {"class_names": names, **counts}
        return {}

    def lengths(self) -> np.ndarray | None:
        if self.kind in ("parquet", "arrow") and "length" in self.arrow.schema.names:
            return self.arrow.to_table(columns=["length"])["length"].to_numpy()
        if self.index:
            with io.BytesIO(Store(self.index, self.storage_options).read()) as file:
                index = np.load(file, allow_pickle=False)
                if index.dtype.names and "length" in index.dtype.names:
                    return index["length"]
        return None


@dataclass(frozen=True)
class _Paths:
    paths: Sequence[str]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Batch:
        return {"path": self.paths[index]}


@dataclass(frozen=True)
class _ArrowFragment:
    source: Files
    index: int

    @cached_property
    def table(self) -> pa.Table:
        return self.source.fragments[self.index].to_table(
            schema=self.source.arrow.schema,
            columns=list(self.source.columns) if self.source.columns is not None else None,
            filter=self.source.filter,
        )

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, index: int) -> Batch:
        return self.table.slice(index, 1).to_pylist()[0]


class _ArrowRecords:
    def __init__(self, source: Files):
        self.source = source
        self.offsets = np.cumsum([0, *(_fragment_rows(fragment) for fragment in source.fragments)])

    @cached_property
    def _group(self):
        @lru_cache(maxsize=1)
        def opened(index: int) -> _ArrowFragment:
            return _ArrowFragment(self.source, index)

        return opened

    def __getstate__(self) -> Mapping[str, object]:
        return {key: value for key, value in self.__dict__.items() if key != "_group"}

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int) -> Batch:
        if not 0 <= index < len(self):
            raise IndexError(index)
        group = bisect.bisect_right(self.offsets, index) - 1
        return self._group(group)[index - int(self.offsets[group])]


class _IndexedLines:
    def __init__(self, source: Files):
        self.source = source
        self.indexes = (
            (source.index,) if source.index else tuple(source._index_url(url) for url in source.paths)
        )
        self.tables = tuple(
            np.load(io.BytesIO(Store(url, source.storage_options).read()), allow_pickle=False)
            for url in self.indexes
        )
        self.offsets = np.cumsum([0, *(len(index) for index in self.tables)])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int) -> Batch:
        if not 0 <= index < len(self):
            raise IndexError(index)
        group = bisect.bisect_right(self.offsets, index) - 1
        entry = self.tables[group][index - int(self.offsets[group])]
        shard = int(entry["shard"]) if self.source.index else group
        fs, path = Store(self.source.paths[shard], self.source.storage_options).resolve()
        with fs.open(path, "rb") as file:
            names = next(csv.reader(_CsvLines(file))) if self.source.kind == "csv" else None
            file.seek(int(entry["offset"]))
            payload = file.read(int(entry["bytes"]))
        if names is not None:
            return next(csv.DictReader(io.StringIO(payload.decode()), fieldnames=names))
        return json.loads(payload) if self.source.kind == "jsonl" else {"text": payload.decode()}


class _CsvLines(Iterator[str]):
    def __init__(self, file: BinaryIO):
        self.file = file

    def __next__(self) -> str:
        line = self.file.readline()
        if not line:
            raise StopIteration
        return line.decode()


class _Lines(grain.IterDataset):
    def __init__(self, source: Files, shard: Shard):
        super().__init__()
        self.source = source
        self.shard = shard

    def __iter__(self) -> grain.DatasetIterator:
        return _LineIterator(self.source, self.shard)


class _LineIterator(grain.DatasetIterator):
    def __init__(self, source: Files, shard: Shard):
        super().__init__()
        self.source, self.shard = source, shard
        fs, path = Store(shard.uri, source.storage_options).resolve()
        self.file = fs.open(path, "rb")
        self.csv = csv.DictReader(_CsvLines(self.file)) if source.kind == "csv" else None
        if self.csv is not None:
            _ = self.csv.fieldnames
        elif shard.start:
            self.file.seek(shard.start - 1)
            if self.file.read(1) != b"\n":
                self.file.readline()

    def __next__(self) -> Batch:
        if self.shard.stop is not None and self.file.tell() >= self.shard.stop:
            raise StopIteration
        if self.csv is not None:
            return next(self.csv)
        while True:
            line = self.file.readline()
            if not line:
                raise StopIteration
            if self.source.kind == "text":
                return {"text": line.decode()}
            if line.strip():
                return json.loads(line)
            if self.shard.stop is not None and self.file.tell() >= self.shard.stop:
                raise StopIteration

    def get_state(self) -> dict[str, int]:
        return {"offset": self.file.tell()}

    def set_state(self, state: Mapping[str, object]) -> None:
        offset = state["offset"]
        if not isinstance(offset, int) or offset < self.shard.start:
            raise ValueError(f"invalid file reader position for {self.shard.id}")
        self.file.seek(offset)

    def close(self) -> None:
        self.file.close()


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
    lengths: np.ndarray
    dtype: str = "int32"
    field: str | None = None
    eos_id: int | None = None
    storage_options: Mapping[str, object] | None = None
    record_codec: bool = False

    @cached_property
    def records(self):
        from .records import ArrayRecords

        return ArrayRecords(self.paths, self.storage_options)

    @cached_property
    def ends(self) -> np.ndarray:
        return np.cumsum(self.lengths, dtype=np.int64)

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
        if len(values) != int(self.lengths[index]):
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
class TokenFiles:
    """A produced train/val token directory, with metadata and document boundaries."""

    url: str
    split: str = "train"
    storage_options: Mapping[str, object] | None = None
    chunk_tokens: int = 65536

    def describe(self) -> str:
        return f"tokens:{self.url}:{self.split}"

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
    def tokens(self) -> TokenBytes | TokenRecords:
        eos = self.meta.get("eos_id")
        if eos is not None and not isinstance(eos, int):
            raise ValueError("token metadata eos_id must be an integer or null")
        committed = Files(self.url).committed
        if committed is not None:
            if committed["format"] != "array_record":
                raise ValueError("token exports must use ArrayRecord")
            root = Store(self.url, self.storage_options)
            lengths = []
            paths = []
            entries = committed["files"]
            if not isinstance(entries, list):
                raise ValueError("token export files must be a list")
            for entry in entries:
                if entry["lengths"] is None:
                    raise FileNotFoundError(
                        "token export needs the producer's lengths; export with length_field='text'"
                    )
                payload = root.child(entry["lengths"]["path"]).read()
                if len(payload) != entry["lengths"]["bytes"]:
                    raise ValueError("token export length index size differs from manifest")
                import hashlib

                if hashlib.sha256(payload).hexdigest() != entry["lengths"]["sha256"]:
                    raise ValueError("token export length index checksum differs from manifest")
                lengths.append(np.load(io.BytesIO(payload), allow_pickle=False))
                paths.append(root.child(entry["path"]).url)
            return TokenRecords(
                tuple(paths),
                np.concatenate(lengths) if lengths else np.zeros(0, np.int64),
                str(self.meta.get("dtype", "int32")),
                "text",
                eos,
                self.storage_options,
                record_codec=True,
            )
        return TokenBytes(
            self.url.rstrip("/") + f"/{self.split}.bin",
            str(self.meta.get("dtype", "uint16")),
            eos,
            self.storage_options,
        )

    def metadata(self) -> Mapping[str, object]:
        return {
            **self.meta,
            "tokens": len(self.tokens),
            "token_store": True,
            f"{self.split}_tokens": len(self.tokens),
        }

    def snapshot(self) -> Mapping[str, object]:
        if isinstance(self.tokens, TokenRecords):
            return {
                "store": "tokens",
                "url": self.url,
                "split": self.split,
                "metadata": self.meta,
                "shards": Files(self.url).snapshot(),
            }
        fs, path = Store(self.tokens.url, self.storage_options).resolve()
        return {
            "store": "tokens",
            "url": self.url,
            "split": self.split,
            "metadata": self.meta,
            "bytes": fs.size(path),
            "version": fs.ukey(path),
        }

    def supports(self, read: Read) -> bool:
        return True

    def records(self, read: Read) -> Records:
        indexed = _TokenChunks(self.tokens, self.chunk_tokens)
        return tuple(indexed[index] for index in range(len(indexed))) if read == "memory" else indexed

    def manifest(self) -> ShardManifest:
        count = len(self.records("indexed"))
        return ShardManifest(
            tuple(
                Shard(f"{self.url}:{self.split}:{index}", self.url, index, index + 1, 1)
                for index in range(count)
            ),
            self._read_shard,
        )

    def _read_shard(self, shard: Shard) -> grain.IterDataset:
        return grain.MapDataset.source(self.records("indexed"))[shard.start : shard.stop].to_iter_dataset()

    def lengths(self) -> np.ndarray | None:
        return None


@dataclass(frozen=True)
class _TokenChunks:
    tokens: TokenBytes | TokenRecords
    size: int

    def __len__(self) -> int:
        return (len(self.tokens) + self.size - 1) // self.size

    def __getitem__(self, index: int) -> Batch:
        return {"text": self.tokens[index * self.size : (index + 1) * self.size].astype(np.int32)}
