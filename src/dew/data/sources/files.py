"""Indexed files and checkpointable physical-shard readers over filesystem URLs."""

from __future__ import annotations

import bisect
import csv
import hashlib
import io
import itertools
import json
import posixpath
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache
from pathlib import PurePosixPath
from typing import BinaryIO, Literal

import grain.python as grain
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from fsspec.spec import AbstractFileSystem

from dew.data.dataset import Batch, Records
from dew.data.pipeline import Dataset
from dew.data.storage import Store, content_version, local
from dew.records import integer, record

from .base import Read, Shard, ShardManifest, preparing
from .index import ArrayIndex

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
    if (
        "gs" in protocols
        and fs.credentials.method in ("google_default", "cloud", "anon")
        and not fs.requester_pays
    ):
        from urllib.parse import urlsplit

        endpoint = urlsplit(fs._location)
        return pafs.GcsFileSystem(
            anonymous=fs.credentials.method == "anon",
            scheme=endpoint.scheme,
            endpoint_override=endpoint.netloc,
        )
    return pafs.PyFileSystem(pafs.FSSpecHandler(fs))


def _mount_root(paths: Sequence[str]) -> str:
    """The directory local `paths` share, a single file's own directory."""
    if not paths:
        return ""
    return posixpath.dirname(paths[0]) if len(set(paths)) == 1 else posixpath.commonpath(paths)


def _version(fs: AbstractFileSystem, path: str) -> str:
    """A file's version as every host holding a copy names it (`content_version`)."""
    return content_version(fs, path, int(fs.info(path)["size"])) if local(fs) else str(fs.ukey(path))


def _fragment_rows(fragment) -> int:
    if isinstance(fragment, ds.ParquetFileFragment):
        return sum(group.num_rows for group in fragment.row_groups)
    return fragment.count_rows()


@dataclass(frozen=True)
class Files(Dataset):
    """Files at URLs, with producer indexes or explicit provisioning for text.

    Parquet partitions are discovered with Arrow's Hive partitioning. A
    predicate runs within each streamed fragment; indexed predicates need
    a provisioned accepted-row index and are refused here.
    """

    url: str | tuple[str, ...]
    format: Format | None = None
    index: str | None = None
    columns: tuple[str, ...] | None = None
    where: ds.Expression | None = None
    partitioning: str | None = "hive"
    shard_bytes: int = 64 * 2**20
    storage_options: Mapping[str, object] | None = field(default=None, repr=False)
    spark: bool = False
    file_info: Mapping[str, Mapping[str, object]] | None = None

    @cached_property
    def declared(self) -> Mapping[str, Mapping[str, object]]:
        if self.file_info is not None:
            return self.file_info
        committed = self.committed
        if committed is None:
            return {}
        entries = committed["files"]
        if not isinstance(entries, list):
            raise ValueError("export manifest files must be a list")
        return {
            Store(str(self.url), self.storage_options).child(entry["path"]).url: entry for entry in entries
        }

    def _header(self, url: str) -> Mapping[str, object]:
        if url in self.declared:
            stated = self.declared[url]
            return {"name": url, "size": stated["bytes"], **stated}
        fs, path = Store(url, self.storage_options).resolve()
        return fs.info(path)

    @cached_property
    def kind(self) -> Format:
        if self.committed is not None:
            declared = _format("data." + str(self.committed["format"]))
            if self.format is not None and self.format != declared:
                raise ValueError(f"source {self.url!r} declares format={declared}, not {self.format}")
            return declared
        return self.format or _format(self.url if isinstance(self.url, str) else self.url[0])

    @cached_property
    def committed(self) -> Mapping[str, object] | None:
        if self.file_info is not None or not isinstance(self.url, str):
            return None
        store = Store(self.url, self.storage_options).child("_dew_manifest.json")
        fs, path = store.resolve()
        if not fs.exists(path):
            return None
        try:
            manifest = json.loads(fs.cat_file(path))
        except ValueError as error:
            raise ValueError(f"committed export manifest is corrupt: {store.url}") from error
        if not isinstance(manifest, dict) or manifest.get("schema") != 1:
            raise ValueError(f"committed export manifest has an unknown schema: {store.url}")
        return manifest

    @cached_property
    def paths(self) -> tuple[str, ...]:
        if self.file_info is not None:
            urls = (self.url,) if isinstance(self.url, str) else self.url
            missing = [url for url in urls if url not in self.file_info]
            if missing:
                raise ValueError(f"published source facts do not name {missing[0]}")
            return urls
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
                    if self.kind not in ("parquet", "arrow")
                    or not any(part.startswith(("_", ".")) for part in PurePosixPath(name[len(path) :]).parts)
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

    def fingerprint(self) -> Mapping[str, object]:
        return self._snapshot

    @cached_property
    def _snapshot(self) -> Mapping[str, object]:
        files = []
        indexes = (
            ()
            if self.kind not in ("jsonl", "text", "csv")
            else (self.index,)
            if self.index
            else tuple(self._index_url(url) for url in self.paths)
        )
        resolved = [(url, *Store(url, self.storage_options).resolve()) for url in (*self.paths, *indexes)]
        # Local copies are named by their place under the files' common directory and their
        # content, so hosts that mounted one dataset at different paths name it alike.
        root = _mount_root([path for _, fs, path in resolved if local(fs)])
        for url, fs, path in resolved:
            if url in indexes and url not in self.declared and not fs.exists(path):
                continue
            metadata = self._header(url)
            version = (
                metadata.get("sha256")
                or metadata.get("ETag")
                or metadata.get("etag")
                or metadata.get("generation")
            )
            if version is None and local(fs):
                # About 4 MiB read over the whole store, each file's first and last blocks at least.
                version = content_version(fs, path, integer(metadata["size"], "size"),
                                          blocks=max(2, 64 // len(resolved)))
            elif version is None:
                version = fs.ukey(path)
            files.append(
                {
                    "uri": posixpath.relpath(path, root) if local(fs) else url,
                    "bytes": metadata["size"],
                    "version": version,
                }
            )
        return {
            "store": self.kind,
            "files": files,
            "columns": self.columns,
            "where": str(self.where),
            "partitioning": self.partitioning,
            "shard_bytes": self.shard_bytes,
            "manifest": None
            if self.committed is None
            else hashlib.sha256(
                json.dumps(self.committed, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "transport": Store(self.paths[0], self.storage_options).describe() if self.paths else {},
        }

    def supports(self, read: Read) -> bool:
        if read != "indexed" or self.kind == "files":
            return True
        if self.kind == "array_record":
            if self.committed is not None:
                return True
            for url in self.paths:
                fs, _ = Store(url, self.storage_options).resolve()
                protocols = (fs.protocol,) if isinstance(fs.protocol, str) else fs.protocol
                if not {"file", "gs", "gcs"}.intersection(protocols):
                    return False
            return True
        if self.where is not None:
            return False
        if self.kind in ("arrow", "parquet"):
            return True
        if self.kind == "webdataset":
            return False
        indexes = (self.index,) if self.index else tuple(self._index_url(path) for path in self.paths)
        if "index_headers" in self.__dict__:
            return True
        headers = {}
        for url in indexes:
            try:
                headers[url] = self._header(url)
            except FileNotFoundError:
                return False
        self.__dict__["index_headers"] = headers
        return True

    @staticmethod
    def _index_url(url: str) -> str:
        return url.rsplit(".", 1)[0] + ".index.npy"

    def provision(self) -> Files:
        if self.kind == "array_record":
            return self._provision_arrayrecord()
        if self.kind not in ("jsonl", "text", "csv"):
            raise ValueError(f"{self.describe()} has no provisionable offset index")
        stores = [Store(url, self.storage_options).resolve() for url in self.paths]
        preparing(self, sum(fs.size(path) or 0 for fs, path in stores), "an offset index beside each file")
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
        self.__dict__.pop("_snapshot", None)
        self.__dict__.pop("_metadata", None)
        self.__dict__.pop("index_headers", None)
        return self

    def _provision_arrayrecord(self) -> Files:
        from .records import ArrayRecords

        if not isinstance(self.url, str) or any(mark in self.url for mark in "*?["):
            raise ValueError("ArrayRecord provisioning needs one directory URL")
        root = Store(self.url, self.storage_options)
        fs, directory = root.resolve()
        if not fs.isdir(directory):
            raise ValueError("ArrayRecord provisioning needs its containing directory URL")
        files = []
        for url in self.paths:
            filesystem, path = Store(url, self.storage_options).resolve()
            opened = ArrayRecords((url,), self.storage_options)
            try:
                count = len(opened)
                mirror, mirrored_path = Store(url, self.storage_options).cached().resolve()
                with mirror.open(mirrored_path, "rb") as reader:
                    if not isinstance(reader.name, str):
                        raise ValueError(f"ArrayRecord provisioning needs a mirror that can hold {url}")
                    with open(reader.name, "rb") as local:
                        digest = hashlib.file_digest(local, "sha256").hexdigest()
            finally:
                opened.close()
            files.append(
                {
                    "path": path[len(directory) :].lstrip("/"),
                    "rows": count,
                    "bytes": filesystem.size(path),
                    "sha256": digest,
                }
            )
        manifest = {
            "schema": 1,
            "format": "array_record",
            "codec": None,
            "rows": sum(entry["rows"] for entry in files),
            "files": files,
            "metadata": {},
        }
        root.child("_dew_manifest.json").put_if_absent(json.dumps(manifest, sort_keys=True).encode())
        self.__dict__.pop("committed", None)
        self.__dict__.pop("declared", None)
        self.__dict__.pop("_snapshot", None)
        self.__dict__.pop("_metadata", None)
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

            committed = self.committed
            counts = self._declared_rows()
            rows = ArrayRecords(
                self.paths,
                self.storage_options,
                record_codec=committed is not None and committed.get("codec") == 1,
                counts=counts,
            )
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
                rows = None if self.where is not None else _fragment_rows(fragment)
                shards.append(Shard(f"{fragment.path}:{number}", fragment.path, number, number + 1, rows))
        elif self.kind == "array_record":
            shards.extend(Shard(url, url) for url in self.paths)
        else:
            for url in self.paths:
                size = integer(self._header(url)["size"], "file bytes")
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
                ArrayRecords(
                    (shard.uri,),
                    self.storage_options,
                    record_codec=self.committed is not None and self.committed.get("codec") == 1,
                )
            ).to_iter_dataset()
        if self.kind == "webdataset":
            from .webdataset import TarRows

            return TarRows(shard.uri, self.storage_options)
        if self.kind == "files":
            return grain.MapDataset.source(_Paths((shard.uri,))).to_iter_dataset()
        return _Lines(self, shard)

    def locality(self) -> tuple[tuple[int, ...], bool]:
        remote = False
        for url in self.paths:
            fs, _ = Store(url, self.storage_options).resolve()
            protocols = (fs.protocol,) if isinstance(fs.protocol, str) else fs.protocol
            remote |= "file" not in protocols and "memory" not in protocols
        if self.kind in ("parquet", "arrow") and self.where is None:
            return tuple(_fragment_rows(fragment) for fragment in self.fragments), remote
        if self.kind in ("jsonl", "text", "csv") and self.supports("indexed"):
            indexed = _IndexedLines(self)
            if self.index is None:
                return tuple(len(table) for table in indexed.tables), remote
            shards = indexed.tables[0]["shard"]
            boundaries = [bisect.bisect_left(shards, shard) for shard in range(len(self.paths) + 1)]
            return tuple(right - left for left, right in itertools.pairwise(boundaries)), remote
        if self.kind == "array_record" and self.committed is not None:
            return self._declared_rows() or (), remote
        if self.kind == "files":
            return tuple(1 for _ in self.paths), remote
        return (), remote

    def _declared_rows(self) -> tuple[int, ...] | None:
        if self.committed is None:
            return None
        entries = self.committed["files"]
        if not isinstance(entries, list):
            raise ValueError("export manifest files must be a list")
        return tuple(integer(record(entry, "export file")["rows"], "export shard rows") for entry in entries)

    def order_fingerprints(self) -> Mapping[str, str]:
        if self.kind not in ("jsonl", "text", "csv"):
            return {}
        if "index_headers" not in self.__dict__ and not self.supports("indexed"):
            return {}
        indexes = (self.index,) if self.index else tuple(self._index_url(path) for path in self.paths)
        known = self.__dict__["index_headers"]
        headers = []
        for url in indexes:
            fs, path = Store(url, self.storage_options).resolve()
            metadata = known[url]
            version = metadata.get("sha256") or metadata.get("etag") or metadata.get("ETag")
            headers.append({"url": url, "version": fs.ukey(path) if version is None else version})
        return {"row_index": hashlib.sha256(json.dumps(headers, sort_keys=True).encode()).hexdigest()}

    def metadata(self) -> Mapping[str, object]:
        return self._metadata

    @cached_property
    def _metadata(self) -> Mapping[str, object]:
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
                if self.where is None
                else {}
            )
            return {"class_names": names, **counts}
        if self.kind in ("jsonl", "text", "csv") and self.supports("indexed"):
            return {"rows": len(_IndexedLines(self))}
        if self.kind == "files":
            return {"rows": len(self.paths)}
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
    lock: threading.RLock = field(default_factory=threading.RLock, compare=False, repr=False)

    @property
    def table(self) -> pa.Table:
        with self.lock:
            if "_table" not in self.__dict__:
                self.__dict__["_table"] = self.source.fragments[self.index].to_table(
                    schema=self.source.arrow.schema,
                    columns=list(self.source.columns) if self.source.columns is not None else None,
                    filter=self.source.where,
                )
            return self.__dict__["_table"]

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, index: int) -> Batch:
        return self.table.slice(index, 1).to_pylist()[0]

    def __getstate__(self) -> Mapping[str, object]:
        return {"source": self.source, "index": self.index}

    def __setstate__(self, state: Mapping[str, object]) -> None:
        self.__dict__.update(state)
        self.__dict__["lock"] = threading.RLock()


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
        headers = source.__dict__.get("index_headers", {})
        self.index_files = tuple(
            ArrayIndex(url, source.storage_options, headers.get(url)) for url in self.indexes
        )
        self.tables = tuple(index.values for index in self.index_files)
        self.offsets = np.cumsum([0, *(len(index) for index in self.tables)])

    def __getstate__(self) -> Mapping[str, object]:
        return {"source": self.source}

    def __setstate__(self, state: Mapping[str, object]) -> None:
        source = state["source"]
        if not isinstance(source, Files):
            raise ValueError("invalid indexed file source state")
        self.__init__(source)

    @cached_property
    def fieldnames(self) -> tuple[tuple[str, ...], ...]:
        headers = []
        for url in self.source.paths:
            fs, path = Store(url, self.source.storage_options).resolve()
            with fs.open(path, "rb") as file:
                headers.append(tuple(next(csv.reader(_CsvLines(file)))))
        return tuple(headers)

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int) -> Batch:
        if not 0 <= index < len(self):
            raise IndexError(index)
        group = bisect.bisect_right(self.offsets, index) - 1
        entry = self.tables[group][index - int(self.offsets[group])]
        shard = int(entry["shard"]) if self.source.index else group
        fs, path = Store(self.source.paths[shard], self.source.storage_options).resolve()
        names = self.fieldnames[shard] if self.source.kind == "csv" else None
        offset = int(entry["offset"])
        payload = fs.cat_file(path, start=offset, end=offset + int(entry["bytes"]))
        if names is not None:
            return next(csv.DictReader(io.StringIO(payload.decode()), fieldnames=names))
        return json.loads(payload) if self.source.kind == "jsonl" else {"text": payload.decode()}


class _CsvLines(Iterator[str]):
    def __init__(self, file: BinaryIO):
        csv.field_size_limit(2**31 - 1)
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


@dataclass(frozen=True)
class TokenFiles(Dataset):
    """A produced train/val token directory, with metadata and document boundaries."""

    url: str
    split: str = "train"
    storage_options: Mapping[str, object] | None = None
    chunk_tokens: int = 65536
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
        documents, indexed = self.document_index.resolve()
        boundaries = {"documents": _version(documents, indexed)} if documents.exists(indexed) else {}
        # A local store is named by its content, so hosts that mounted it at different paths agree.
        place = {} if local(documents) else {"url": self.url}
        if isinstance(self.tokens, (TokenRecords, TokenShards)):
            fs, path = self.length_index.resolve()
            return {
                "store": "tokens",
                **place,
                "split": self.split,
                "metadata": self.meta,
                "field": self.field,
                "chunk_tokens": self.chunk_tokens,
                "shards": Files(
                    self.tokens.paths, format="array_record", storage_options=self.storage_options
                ).fingerprint(),
                "index": _version(fs, path) if fs.exists(path) else None,
                "shard_rows": self.tokens.shard_rows if isinstance(self.tokens, TokenRecords) else None,
                "producer": Files(self.url, storage_options=self.storage_options).committed,
                **boundaries,
            }
        fs, path = Store(self.tokens.url, self.storage_options).resolve()
        return {
            "store": "tokens",
            **place,
            "split": self.split,
            "metadata": self.meta,
            "field": self.field,
            "chunk_tokens": self.chunk_tokens,
            "bytes": fs.size(path),
            "version": _version(fs, path),
            **boundaries,
        }

    def supports(self, read: Read) -> bool:
        if read == "memory":
            return True
        tokens = self.tokens
        return not isinstance(tokens, TokenRecords) or tokens.lengths is not None

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
            block = 1 << 24
            found = [
                np.flatnonzero(tokens[start : start + block] == tokens.eos_id) + start + 1
                for start in range(0, len(tokens), block)
            ]
            ends = np.concatenate([np.zeros(0, np.int64), *found]).astype(np.int64)
            if not len(ends) or ends[-1] != len(tokens):
                ends = np.append(ends, len(tokens))
            encoded = io.BytesIO()
            np.save(encoded, ends, allow_pickle=False)
            self.document_index.write(encoded.getvalue())
            self.__dict__.pop("mapped_documents", None)
        return self

    def records(self, read: Read) -> Records:
        """Each document by index where the store has a document index
        (`document_ends`), and otherwise uniform chunks of `chunk_tokens`,
        or for a memory read the whole stream as one record."""
        if not self.supports(read):
            raise FileNotFoundError(
                f"{self.describe()} needs its producer length index; "
                "run dew data provision or choose read='memory'"
            )
        if read == "memory" and isinstance(self.tokens, TokenRecords):
            self.tokens.materialize_lengths()
        ends = self.document_ends()
        if ends is not None:
            return _TokenDocuments(self.tokens, ends)
        if read == "memory":
            return ({"text": self.tokens[:].astype(np.int32)},)
        return _TokenChunks(self.tokens, self.chunk_tokens)

    def document_ends(self) -> np.ndarray | None:
        """Where each document ends in the token stream, each running through
        its EOS, from the index provisioning builds; None without one."""
        fs, path = self.document_index.resolve()
        return np.asarray(self.mapped_documents.values, np.int64) if fs.exists(path) else None

    def manifest(self) -> ShardManifest:
        count = len(self.records("indexed"))
        size = 1 if self.document_ends() is None else 1024
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


@dataclass(frozen=True)
class _TokenChunks:
    tokens: TokenBytes | TokenRecords | TokenShards
    size: int

    def __len__(self) -> int:
        return (len(self.tokens) + self.size - 1) // self.size

    def __getitem__(self, index: int) -> Batch:
        return {"text": self.tokens[index * self.size : (index + 1) * self.size].astype(np.int32)}
