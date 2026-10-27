"""Indexed files and checkpointable physical-shard readers over filesystem URLs."""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
import posixpath
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import PurePosixPath
from typing import BinaryIO, Literal, get_args
from urllib.parse import urlsplit

import grain.python as grain
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from fsspec.spec import AbstractFileSystem
from fsspec.utils import infer_compression
from gcsfs import GCSFileSystem
from s3fs import S3FileSystem

from dew._records import integer, record, text
from dew.batch import Batch
from dew.data.dataset import Ranges, Records
from dew.data.pipeline import Dataset
from dew.data.steps.generic import Fields
from dew.data.storage import Readable, Store, commit_export, content_version, in_memory, local, on_gcs

from .base import Read, Shard, ShardManifest, preparing, url_label
from .index import ArrayIndex

type Format = Literal["jsonl", "text", "csv", "parquet", "arrow", "array_record", "files", "webdataset"]


def _format(url: str) -> Format:
    inferred = _inferred(url)
    if inferred is None:
        raise ValueError(f"source {url!r} needs format=; a URL does not declare its record format")
    return inferred


_SUFFIXES: dict[Format, tuple[str, ...]] = {
    "jsonl": (".jsonl", ".jsonl.gz", ".jsonl.zst"),
    "text": (".txt",),
    "csv": (".csv",),
    "parquet": (".parquet",),
    "arrow": (".arrow", ".feather"),
    "array_record": (".array_record",),
    "webdataset": (".tar",),
}
"""The suffixes each record format's files carry, which a URL's format is inferred from and a
directory's files of a format are found by."""


FORMATS: tuple[Format, ...] = get_args(Format.__value__)


def declared_format(value: str, store: str) -> Format:
    """The record format `value` names, as a committed manifest and a native writer state it
    (`dew.data.sinks.Writer`): one of `FORMATS`, never read by a suffix, which "text",
    "webdataset" and "files" do not spell."""
    for kind in FORMATS:
        if value == kind:
            return kind
    raise ValueError(f"{store} declares record format {value!r}, none of {', '.join(FORMATS)}")


def _inferred(url: str) -> Format | None:
    """The record format a URL's suffix declares, or None where it declares none: a directory's too,
    spelled with a trailing separator or without."""
    name = PurePosixPath(url).name.lower()
    return next((kind for kind, suffixes in _SUFFIXES.items() if name.endswith(suffixes)), None)


def _arrow_filesystem(fs: AbstractFileSystem) -> pafs.FileSystem:
    """Arrow's filesystem over `fs`: Arrow's own S3 or GCS client where it
    carries every setting `fs` reads with (`_native_s3`, `_native_gcs`), as
    it reads Parquet row groups 1.4x (S3) to 2.1x (GCS) faster than Arrow
    over fsspec does; Arrow over `fs` itself otherwise, which keeps them all."""
    if local(fs):
        return pafs.LocalFileSystem()
    if isinstance(fs, S3FileSystem) and _native_s3(fs):
        endpoint = urlsplit(fs.endpoint_url or fs.client_kwargs.get("endpoint_url") or "")
        # An anonymous s3fs keeps keys it does not sign with, which Arrow's client refuses beside anonymity.
        return pafs.S3FileSystem(
            access_key=None if fs.anon else fs.key,
            secret_key=None if fs.anon else fs.secret,
            session_token=None if fs.anon else fs.token,
            anonymous=fs.anon,
            region=fs.client_kwargs.get("region_name"),
            endpoint_override=endpoint.netloc or None,
            scheme=endpoint.scheme or ("https" if fs.use_ssl else "http"),
        )
    if isinstance(fs, GCSFileSystem) and _native_gcs(fs):
        endpoint = urlsplit(fs.base)
        return pafs.GcsFileSystem(
            anonymous=fs.credentials.method == "anon",
            scheme=endpoint.scheme,
            endpoint_override=endpoint.netloc,
            project_id=fs.project,
            default_bucket_location=fs.default_location or "US",
        )
    return pafs.PyFileSystem(pafs.FSSpecHandler(fs))


def _native_s3(fs: S3FileSystem) -> bool:
    """Whether Arrow's S3 client carries every setting `fs` reads with: keys,
    anonymity, region, endpoint and TLS, and nothing it cannot express, such
    as a profile, a session (an injected one shows in the arguments fsspec
    records, as s3fs's own does not), botocore configuration, requester pays,
    extra request arguments or object versions."""
    return (not (fs.kwargs or fs.config_kwargs or fs.req_kw or fs.s3_additional_kwargs or fs.version_aware)
            and "session" not in fs.storage_options
            and set(fs.client_kwargs) <= {"region_name", "endpoint_url"})


def _native_gcs(fs: GCSFileSystem) -> bool:
    """Whether Arrow's GCS client carries every setting `fs` reads with:
    anonymity, or the default credentials at full access, the project and
    the endpoint, and nothing it cannot express, such as other credentials
    (the instance's own, forced), narrower scopes, requester pays,
    checksums, timeouts, session arguments or object versions."""
    return ((fs.credentials.method == "anon"
             or (fs.credentials.method == "google_default" and fs.credentials.access == "full_control"))
            and not (fs.requester_pays or fs.session_kwargs or fs.version_aware)
            and fs.consistency in (None, "none")
            and fs.requests_timeout is None and fs.timeout is None)


def export_place(path: str) -> str:
    """Where a committed export holds a file, its path without the directory its writer named by a
    nonce of its own (`dew.data.sinks.export`: `shards/<writer>/...`, `_dew_indexes/<writer>/...`):
    its partitions and its name stay, so two exports of the same rows place their files alike."""
    head, _, rest = path.partition("/")
    if head in ("shards", "_dew_indexes") and "/" in rest:
        return f"{head}/{rest.partition('/')[2]}"
    return path


def export_content(committed: Mapping[str, object]) -> Mapping[str, object]:
    """A committed export's manifest with each file where it is placed (`export_place`), not where
    its writer put it, so two exports of the same rows hold the same content, which the manifest's
    order, places, sizes and hashes say alike."""
    files = committed.get("files")
    if not isinstance(files, list) or not all(isinstance(entry, Mapping) for entry in files):
        raise ValueError("export manifest files must be a list of file entries")

    def placed(entry: Mapping[str, object]) -> Mapping[str, object]:
        """`entry` with its path, and its length index's, where the export places them."""
        return {key: (export_place(fact) if key == "path" and isinstance(fact, str)
                      else placed(fact) if key == "lengths" and isinstance(fact, Mapping) else fact)
                for key, fact in entry.items()}

    return {**committed, "files": [placed(entry) for entry in files]}


def _mount_root(paths: Sequence[str]) -> str:
    """The directory local `paths` share, a single file's own directory."""
    if not paths:
        return ""
    return posixpath.dirname(paths[0]) if len(set(paths)) == 1 else posixpath.commonpath(paths)


def _fragment_rows(fragment) -> int:
    if isinstance(fragment, ds.ParquetFileFragment):
        return sum(group.num_rows for group in fragment.row_groups)
    return fragment.count_rows()


def _open_lines(fs: AbstractFileSystem, path: str):
    compression = infer_compression(path.lower())
    if compression == "zstd":
        from zstandard import ZstdDecompressor

        return io.BufferedReader(ZstdDecompressor().stream_reader(fs.open(path, "rb")))
    return fs.open(path, "rb", compression=compression)


def _record_starts(file: BinaryIO, size: int, parts: int, kind: Format) -> list[int]:
    """Where at most `parts` parts of a `kind` line file of `size` bytes start, each at a record
    (a line; a JSON line that is not blank, as `_lines` reads them), in order and distinct: part
    `k` at the first record at or after the even cut `size * k / parts`, or after the record the
    part before it starts at where that is further on. A record longer than the cuts apart (a
    10,000-character first line, say) moves the cuts it holds on to the records after it rather
    than onto one another; a file of fewer records than parts has as many parts as records."""
    starts: list[int] = []
    for part in range(parts):
        cut = size * part // parts
        at = _record_start(file, cut if not starts else max(cut, starts[-1] + 1), kind)
        if at >= size:
            break
        starts.append(at)
    return starts


def _record_start(file: BinaryIO, offset: int, kind: Format) -> int:
    """Where the first record of a `kind` line file that starts at `offset` or after it starts: the
    next line start, past blank lines in JSON lines, which hold no record."""
    at = _line_start(file, offset)
    file.seek(at)
    while kind == "jsonl" and (line := file.readline()) and not line.strip():
        at = file.tell()
    return at


_SCAN_BYTES = 1 << 20
"""The bytes a search for a line start reads at a time (`_line_start`), so a record of any length is
passed over in bounded memory."""


def _line_start(file: BinaryIO, offset: int, stop: int | None = None) -> int:
    """Where the first line that starts at `offset` or after it starts in `file`, its end where none
    does: a range of a file's bytes reads the lines that start in it. With `stop`, the search reads
    no byte at or past it and gives `stop` where no line starts before it, so a range's reader
    passes over a long record's bytes within its own range alone, a chunk at a time."""
    if not offset:
        return 0
    file.seek(offset - 1)
    at = offset - 1
    while stop is None or at < stop - 1:
        chunk = file.read(_SCAN_BYTES if stop is None else min(_SCAN_BYTES, stop - 1 - at))
        if not chunk:
            return at
        found = chunk.find(b"\n")
        if found >= 0:
            return at + found + 1
        at += len(chunk)
    return stop


def _seek_lines(file: BinaryIO, offset: int) -> None:
    if file.seekable():
        file.seek(offset)
        return
    while file.tell() < offset:
        if not file.read(min(offset - file.tell(), 2**20)):
            raise ValueError("file reader position exceeds its decompressed stream")


@dataclass(frozen=True)
class Files(Dataset):
    """Files at URLs, with producer indexes or explicit provisioning for text.

    Parquet partitions are discovered with Arrow's Hive partitioning. A
    predicate runs within each streamed fragment; indexed predicates need
    a provisioned accepted-row index and are refused here. `columns`
    selects the fields of every format's rows.

    CSV is parsed by Python's `csv` module under the process's field size
    limit, 128 KiB a field by default; a source with longer fields reads
    once its caller raises `csv.field_size_limit`.

    `suffixes` keeps the files whose names end in one of them, any case, with
    or without the dot, as the store lists them when it is read: a directory's
    media, not the lists and notes beside it (`dew.data.video.media`).
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
    suffixes: tuple[str, ...] | None = None

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
            declared = declared_format(text(self.committed["format"], "export format"),
                                       f"committed export {self.url!r}")
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
        kept = self._kept(self._listed)
        if not kept:
            raise FileNotFoundError(f"source {self.url!r} matched no {self.kind} files")
        return kept

    def _kept(self, paths: Sequence[str]) -> tuple[str, ...]:
        """`paths` whose names end in one of `suffixes`, all of them where it names none."""
        if self.suffixes is None:
            return tuple(paths)
        wanted = {"." + suffix.lower().removeprefix(".") for suffix in self.suffixes}
        return tuple(path for path in paths if PurePosixPath(path).suffix.lower() in wanted)

    @cached_property
    def _listed(self) -> tuple[str, ...]:
        """The files the store names, the published facts' or a committed manifest's, else those its
        URL lists, none where it lists none."""
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
                if self.kind != "files":
                    names = [name for name in names if name.lower().endswith(_SUFFIXES[self.kind])]
            else:
                names = [path]
            found.extend(fs.unstrip_protocol(name) for name in names if fs.isfile(name))
        return tuple(found)

    def label(self) -> str:
        return url_label(self.url if isinstance(self.url, str) else self.url[0])

    def describe(self) -> str:
        return f"files:{self.kind}:{self.url}"

    def fingerprint(self) -> Mapping[str, object]:
        return self._snapshot

    @property
    def _indexes(self) -> tuple[str, ...]:
        if self.kind not in ("jsonl", "text", "csv"):
            return ()
        return (self.index,) if self.index else tuple(self._index_url(url) for url in self.paths)

    @cached_property
    def _resolved(self) -> tuple[tuple[str, AbstractFileSystem, str], ...]:
        """Each file and length index, by URL, with the filesystem and path it resolves to."""
        return tuple((url, *Store(url, self.storage_options).resolve())
                     for url in (*self.paths, *self._indexes))

    @cached_property
    def _places(self) -> Mapping[str, str]:
        """Each file's place, by URL, which the files' fingerprint and their shards' ids name it by: where
        a committed export places it (`export_place`), a local copy's path under the files' common
        directory, so hosts that mounted one dataset at different paths name it alike, else its URL. A
        catalog of files (`format="files"`), whose records are their URLs, is named by its URLs."""
        root = _mount_root([path for _, fs, path in self._resolved if local(fs)])
        places = self._export_places
        return {url: places.get(url) or (posixpath.relpath(path, root) if local(fs) and self.kind != "files"
                                         else url)
                for url, fs, path in self._resolved}

    def _shard_id(self, number: int, url: str, start: int) -> str:
        """The id of the shard of the file at `url`, number `number` among the files, that starts at
        `start`: the file's place (`_places`), which two files of an export may share once their
        writers' directories are left out, and its number, which tells them apart."""
        return f"{self._places[url]}#{number}:{start}"

    @cached_property
    def _file_facts(self) -> tuple[tuple[str, Mapping[str, object]], ...]:
        files = []
        indexed = set(self._indexes)
        # Local copies are named by their place (`_places`) and their content.
        for url, fs, path in self._resolved:
            if url in indexed and not self._present(url):
                continue
            metadata = self._header(url)
            version = (
                metadata.get("sha256")
                or metadata.get("ETag")
                or metadata.get("etag")
                or metadata.get("generation")
            )
            if local(fs):
                # A copy on this host's disk is named by what it holds, which its producer's
                # declaration, where it states one, must describe.
                size = integer(fs.info(path)["size"], "size")
                version = content_version(fs, path, size)
                stated = self.declared.get(url, {})
                if stated.get("bytes", size) != size or stated.get("sha256", version) != version:
                    raise ValueError(
                        f"{url} declares {stated.get('bytes')} bytes (sha256 {stated.get('sha256')}), but "
                        f"the file holds {size} (sha256 {version}): the store's metadata is stale")
                metadata = {**metadata, "size": size}
            elif version is None:
                version = fs.ukey(path)
            files.append(
                (url, {
                    "uri": self._places[url],
                    "bytes": metadata["size"],
                    "version": version,
                })
            )
        return tuple(files)

    @property
    def _read_options(self) -> Mapping[str, object]:
        """The options that decide which rows and shards a read of the files makes."""
        return {"columns": self.columns, "where": str(self.where), "shard_bytes": self.shard_bytes}

    @cached_property
    def _snapshot(self) -> Mapping[str, object]:
        return {
            "store": self.kind,
            "files": [metadata for _, metadata in self._file_facts],
            **self._read_options,
            "partitioning": self.partitioning,
            "manifest": None
            if self.committed is None
            else hashlib.sha256(
                json.dumps(export_content(self.committed), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            # How an export is reached is its root's; its files are under its writer's own names.
            "transport": Store(str(self.url) if self.committed is not None else self.paths[0],
                               self.storage_options).describe() if self.paths else {},
        }

    @cached_property
    def _export_places(self) -> Mapping[str, str]:
        """Each file of a committed export, and each length index beside one, by URL: where the export
        places it (`export_place`); none for other files."""
        if self.committed is None:
            return {}
        root = Store(str(self.url), self.storage_options)
        entries = self.committed["files"]
        if not isinstance(entries, list):
            raise ValueError(f"export manifest has invalid files at {self.url}")
        named = [entry for entry in entries if isinstance(entry, Mapping)]
        lengths = [entry["lengths"] for entry in named if isinstance(entry.get("lengths"), Mapping)]
        return {root.child(str(entry["path"])).url: export_place(str(entry["path"]))
                for entry in (*named, *lengths)}

    def supports(self, read: Read) -> bool:
        if read != "indexed" or self.kind == "files":
            return True
        if self.kind == "array_record":
            if self.committed is not None:
                return True
            for url in self.paths:
                fs, _ = Store(url, self.storage_options).resolve()
                # ArrayRecord's own reader opens a local path or a GCS URL (`ArrayRecords.reader`).
                if not (local(fs) or on_gcs(fs)):
                    return False
            return True
        if self.where is not None:
            return False
        if self.kind in ("arrow", "parquet"):
            return True
        if self.kind == "webdataset":
            return False
        if "index_headers" in self.__dict__:
            return True
        if not all(self._present(url) for url in self._indexes):
            return False
        self.__dict__["index_headers"] = {url: self._header(url) for url in self._indexes}
        return True

    def _present(self, url: str) -> bool:
        """Whether the index at `url` exists: stated by the files' producer,
        or found in its store, which reads an unreachable store as holding
        none (`exists`), as the Hub offline is."""
        if url in self.declared:
            return True
        fs, path = Store(url, self.storage_options).resolve()
        return fs.exists(path)

    @staticmethod
    def _index_url(url: str) -> str:
        if infer_compression(url.lower()) is not None:
            return url + ".index.npy"
        return url.rsplit(".", 1)[0] + ".index.npy"

    def can_provision(self, read: Read) -> bool:
        return (read == "indexed" and self.where is None
                and self.kind in ("array_record", "jsonl", "text", "csv"))

    def provision(self) -> Files:
        if self.kind == "array_record":
            return self._provision_arrayrecord()
        if self.kind not in ("jsonl", "text", "csv"):
            raise ValueError(f"{self.describe()} has no provisionable offset index")
        stores = [Store(url, self.storage_options).resolve() for url in self.paths]
        preparing(self, sum(fs.size(path) or 0 for fs, path in stores), "an offset index beside each file")
        for url in self.paths:
            fs, path = Store(url, self.storage_options).resolve()
            with _open_lines(fs, path) as file:
                names = _header(file) if self.kind == "csv" else None
                offsets = [(0, start, size) for _, start, size in _lines(self.kind, file, names)]
            encoded = io.BytesIO()
            np.save(
                encoded,
                np.array(offsets, dtype=[("shard", "<u4"), ("offset", "<u8"), ("bytes", "<u8")]),
                allow_pickle=False,
            )
            Store(self._index_url(url), self.storage_options).write(encoded.getvalue())
        self.__dict__.pop("paths", None)
        self.__dict__.pop("_file_facts", None)
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
        commit_export(root, format="array_record", codec=None, rows=sum(entry["rows"] for entry in files),
                      files=files)
        self.__dict__.pop("committed", None)
        self.__dict__.pop("declared", None)
        self.__dict__.pop("_file_facts", None)
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
            return self._projected(_Paths(self.paths))
        if read == "memory" and self.kind != "array_record":
            # Each shard as its reader reads it: an Arrow fragment as `where` filters it, a file's lines.
            return tuple(row for shard in self.manifest().shards for row in self._read_shard(shard))
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
        else:
            rows = _IndexedLines(self)
        rows = self._projected(rows)
        return tuple(rows[index] for index in range(len(rows))) if read == "memory" else rows

    def _projected(self, rows: Records) -> Records:
        """`rows` cut to `columns`, which an Arrow source's own reader selects: each row checked to
        hold them (`_Columns`), then projected to exactly those fields (`Fields`)."""
        if self.columns is None or self.kind in ("parquet", "arrow"):
            return rows
        return (grain.MapDataset.source(rows).map(_Columns(self.columns, self.describe()))
                .map(Fields(self.columns)))

    @cached_property
    def _arrow_files(self) -> tuple[pafs.FileSystem, tuple[str, ...], Mapping[str, str], str | None]:
        """Where Arrow reads the files (`Store.readable`), one filesystem for all of them; each
        path Arrow reads's name where the file is; and, where Arrow reads copies, the directory
        partitions are discovered under, so a copy's location adds none: a Hub dataset's files
        read from huggingface_hub's cache where it holds every one, and from the Hub otherwise."""
        stores = [Store(url, self.storage_options) for url in self.paths]
        located = [store.readable() for store in stores]
        # Some of a Hub dataset's files in huggingface_hub's cache, on this host's disk, and some not.
        if len({local(place.fs) for place in located}) > 1:
            located = [Readable(*store.resolve(), None) for store in stores]
        filesystem = _arrow_filesystem(located[0].fs)
        paths = tuple(filesystem.normalize_path(place.path) for place in located)
        names = dict(zip(paths, self.paths, strict=True))
        roots = [place.root for place in located if place.root is not None]
        base = posixpath.commonpath(roots) if len(roots) == len(located) else None
        return filesystem, paths, names, base

    def _shard_name(self, path: str) -> str:
        """The place of the file Arrow reads at `path` (`_places`), whichever copy Arrow reads it from
        (`_arrow_files`), which its shards' ids are made of."""
        return self._places[self._arrow_files[2][path]]

    @cached_property
    def arrow(self) -> ds.Dataset:
        filesystem, paths, _, base = self._arrow_files
        return ds.dataset(
            list(paths),
            filesystem=filesystem,
            format="ipc" if self.kind == "arrow" else "parquet",
            partitioning=self.partitioning,
            partition_base_dir=base,
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
        """The files' shards, each named by its file's place (`_places`) and where in it it starts, so a
        fill keyed by a shard's rows finds them wherever the files are mounted."""
        shards = []
        if self.kind in ("parquet", "arrow"):
            for number, fragment in enumerate(self.fragments):
                rows = None if self.where is not None else _fragment_rows(fragment)
                name = self._shard_name(fragment.path)
                shards.append(Shard(f"{name}:{number}", name, number, number + 1, rows))
        elif self.kind == "array_record":
            shards.extend(Shard(self._shard_id(number, url, 0), url) for number, url in enumerate(self.paths))
        else:
            for number, url in enumerate(self.paths):
                size = integer(self._header(url)["size"], "file bytes")
                if infer_compression(url.lower()) is not None:
                    # Compressed bytes cannot be cut at logical record offsets.
                    shards.append(Shard(self._shard_id(number, url, 0), url))
                    continue
                step = self.shard_bytes if self.kind in ("jsonl", "text") else max(size, 1)
                if step < 1:
                    raise ValueError("file shard_bytes must be positive")
                # Equal ranges of at least `step` bytes, so no range is a short tail a record ending
                # in it covers whole: a range holds a record's start unless a record is longer than
                # it, which its reader refuses (`_LineIterator`).
                parts = max(1, size // step) if size else 0
                cuts = [size * part // parts for part in range(parts + 1)]
                shards.extend(Shard(self._shard_id(number, url, start), url, start, stop)
                              for start, stop in itertools.pairwise(cuts))
        divisible = self.kind in ("jsonl", "text") and not any(
            infer_compression(url.lower()) for url in self.paths)
        return ShardManifest(tuple(shards), self._read_shard, self._subdivided if divisible else None)

    def _subdivided(self, count: int) -> tuple[Shard, ...]:
        """The JSON-lines or text files cut into `count` parts, for a streamed read with more
        readers and workers than its shards (`ShardManifest.subdivide`), as Hadoop splits a text
        file: each file a share of the parts by its bytes, each part the records that start from
        its cut on (`_record_starts`), so each part holds a record and a reader or worker given
        one reads. A file of fewer records than its share keeps a part for each and hands the rest
        to the files that hold more, by their bytes; a store of fewer records than parts is
        refused."""
        sizes = [integer(self._header(url)["size"], "file bytes") for url in self.paths]
        parts = {number: 1 for number, size in enumerate(sizes) if size}
        starts: dict[int, list[int]] = {}
        out: set[int] = set()
        while True:
            short, open_files = count - sum(parts.values()), [number for number in parts if number not in out]
            if short > 0 and open_files:
                total = sum(sizes[number] for number in open_files)
                shares = {number: short * sizes[number] / total for number in open_files}
                for number, share in shares.items():
                    parts[number] += math.floor(share)
                for number in sorted(open_files, key=lambda held: math.floor(shares[held]) - shares[held]):
                    if sum(parts.values()) >= count:
                        break
                    parts[number] += 1
            planned = [number for number in parts if len(starts.get(number, ())) != parts[number]]
            if not planned:
                break
            for number in planned:
                fs, path, _ = Store(self.paths[number], self.storage_options).readable()
                with _open_lines(fs, path) as file:
                    starts[number] = _record_starts(file, sizes[number], parts[number], self.kind)
                if len(starts[number]) < parts[number]:
                    out.add(number)
                    parts[number] = len(starts[number])
        shards = []
        for number, held in sorted(starts.items()):
            url = self.paths[number]
            shards.extend(Shard(self._shard_id(number, url, start), url, start, stop)
                          for start, stop in zip(held, [*held[1:], sizes[number]], strict=True))
        if len(shards) < count:
            raise ValueError(f"{self.describe()} holds {len(shards)} records for {count} readers × workers; "
                             "read it with fewer")
        return tuple(shards)

    def _read_shard(self, shard: Shard) -> grain.IterDataset:
        if self.kind in ("parquet", "arrow"):
            return grain.MapDataset.source(_ArrowFragment(self, shard.start)).to_iter_dataset()
        rows = self._shard_rows(shard)
        return rows if self.columns is None else rows.map(_Columns(self.columns, self.describe())).map(
            Fields(self.columns))

    def _shard_rows(self, shard: Shard) -> grain.IterDataset:
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
            remote |= not (local(fs) or in_memory(fs))
        if self.kind in ("parquet", "arrow") and self.where is None:
            return tuple(_fragment_rows(fragment) for fragment in self.fragments), remote
        if self.kind in ("jsonl", "text", "csv") and self.supports("indexed"):
            indexed = _IndexedLines(self)
            if self.index is None:
                return tuple(len(table) for table in indexed.tables), remote
            # The index's records each name their file's shard, in whichever unsigned type it wrote.
            counts = np.bincount(indexed.tables[0]["shard"], minlength=len(self.paths))[: len(self.paths)]
            return tuple(int(count) for count in counts), remote
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
        indexes = set(self._indexes)
        headers = [metadata for url, metadata in self._file_facts if url in indexes]
        return {"row_index": hashlib.sha256(json.dumps(headers, sort_keys=True).encode()).hexdigest()}

    def metadata(self) -> Mapping[str, object]:
        return self._metadata

    def stated(self) -> Mapping[str, object]:
        """What its committed manifest says, or a parquet or Arrow store's schema names (class
        names), no record read; files not here state nothing."""
        if self.committed is not None:
            return self._metadata
        if self.kind in ("parquet", "arrow") and self._kept(self._listed):
            return {"class_names": self._metadata.get("class_names", {})}
        return {}

    @cached_property
    def _metadata(self) -> Mapping[str, object]:
        if self.committed is not None:
            metadata = self.committed.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise ValueError("export metadata must be a mapping")
            rules = {"count_rules": self.committed["count_rules"]} if "count_rules" in self.committed else {}
            return {**metadata, **rules, "rows": self.committed["rows"], "format": self.committed["format"]}
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
            # Rows with no counted targets column hold no count of Dew's (`CountRule`).
            rules = {} if "_dew_valid_targets" in self.arrow.schema.names else {"count_rules": []}
            return {"class_names": names, **counts, **rules}
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
class _Columns:
    """A row that holds `columns`, as an Arrow reader's projection reads them: a row that is not a
    mapping, or lacks a column, is refused, naming the store. A missing row (`None`, which a store's
    records may hold) never reaches it: Grain's maps pass one on as it is."""

    columns: tuple[str, ...]
    described: str

    def __call__(self, row: Batch | bytes) -> Batch:
        if not isinstance(row, Mapping):
            raise ValueError(
                f"{self.described} reads {type(row).__name__} rows, which have no columns to select")
        missing = [name for name in self.columns if name not in row]
        if missing:
            raise ValueError(f"{self.described} has a row without the columns {missing}")
        return row


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
        self.ranges = Ranges(tuple(_fragment_rows(fragment) for fragment in source.fragments))

    @cached_property
    def _group(self):
        @lru_cache(maxsize=1)
        def opened(index: int) -> _ArrowFragment:
            return _ArrowFragment(self.source, index)

        return opened

    def __getstate__(self) -> Mapping[str, object]:
        return {key: value for key, value in self.__dict__.items() if key != "_group"}

    def __len__(self) -> int:
        return len(self.ranges)

    def __getitem__(self, index: int) -> Batch:
        group, row = self.ranges.locate(index)
        return self._group(group)[row]


class _IndexedLines:
    def __init__(self, source: Files):
        self.source = source
        self.indexes = source._indexes
        headers = source.__dict__.get("index_headers", {})
        self.index_files = tuple(
            ArrayIndex(url, source.storage_options, headers.get(url)) for url in self.indexes
        )
        self.tables = tuple(index.values for index in self.index_files)
        self.ranges = Ranges(tuple(len(index) for index in self.tables))

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
            fs, path, _ = Store(url, self.source.storage_options).readable()
            with _open_lines(fs, path) as file:
                headers.append(_header(file))
        return tuple(headers)

    def __len__(self) -> int:
        return len(self.ranges)

    def __getitem__(self, index: int) -> Batch:
        group, row = self.ranges.locate(index)
        entry = self.tables[group][row]
        shard = int(entry["shard"]) if self.source.index else group
        fs, path, _ = Store(self.source.paths[shard], self.source.storage_options).readable()
        names = self.fieldnames[shard] if self.source.kind == "csv" else None
        offset = int(entry["offset"])
        compression = infer_compression(path.lower())
        if compression is not None:
            with _open_lines(fs, path) as file:
                _seek_lines(file, offset)
                payload = file.read(int(entry["bytes"]))
        else:
            payload = fs.cat_file(path, start=offset, end=offset + int(entry["bytes"]))
        held = [line for line, _, _ in _lines(self.source.kind, io.BytesIO(payload), names)]
        if len(held) != 1:
            raise ValueError(f"{self.source.describe()}'s index names {len(held)} records, not one, at byte "
                             f"{offset} of {path}: provision it again")
        return _decoded(self.source.kind, held[0])


def _header(file: BinaryIO) -> tuple[str, ...]:
    """A CSV file's field names, its first row, read from `file`, which is left past it."""
    return tuple(next(csv.reader(_CsvLines(file)), ()))


def _lines(kind: str, file: BinaryIO, names: Sequence[str] | None = None,
           stop: int | None = None) -> Iterator[tuple[bytes | dict[str, str], int, int]]:
    """Each record of a `kind` line file from `file`'s position, undecoded (`_decoded`), with where
    its bytes start and how many they are, until a record would start at `stop`: a CSV row of the
    fields `names` (its quoted newlines its own, so it is parsed to be found), a JSON line's bytes
    (blank lines skipped) or a text line's. The offsets an index provisions, an indexed read of one
    record's bytes and a stream all find records by it; only a read decodes them."""
    if kind == "csv":
        reader = csv.DictReader(_CsvLines(file), fieldnames=names)
        while stop is None or file.tell() < stop:
            start = file.tell()
            row = next(reader, None)
            if row is None:
                return
            yield row, start, file.tell() - start
        return
    while stop is None or file.tell() < stop:
        start = file.tell()
        line = file.readline()
        if not line:
            return
        if kind == "text" or line.strip():
            yield line, start, len(line)


def _decoded(kind: str, line: bytes | dict[str, str]) -> Batch:
    """The record a line `_lines` found holds: a CSV row as parsed, a JSON line's object or a text
    line's text."""
    if isinstance(line, dict):
        return line
    return {"text": line.decode()} if kind == "text" else json.loads(line)


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
    # Whether `file` is open: an iterator whose opening failed has nothing to close.
    _opened = False

    def __init__(self, source: Files, shard: Shard):
        super().__init__()
        self.source, self.shard = source, shard
        self.fs, self.path, _ = Store(shard.uri, source.storage_options).readable()
        self.compression = infer_compression(self.path.lower())
        self.file = _open_lines(self.fs, self.path)
        self._opened = True
        self.names = _header(self.file) if source.kind == "csv" else None
        if self.names is None and shard.start:
            begin = _line_start(self.file, shard.start, shard.stop)
            if shard.stop is not None and begin >= shard.stop:
                raise ValueError(
                    f"no line of {shard.uri} starts in its bytes {shard.start} to {shard.stop}: one "
                    f"record runs through all {shard.stop - shard.start} of them, longer than "
                    f"shard_bytes={source.shard_bytes}, and the range would read nothing. Read it with "
                    "shard_bytes of at least its longest record, or provision it (`dew data "
                    "provision`) to read it record by record")
            self.file.seek(begin)
        self.records = _lines(source.kind, self.file, self.names, shard.stop)

    def __next__(self) -> Batch:
        line, _, _ = next(self.records)
        return _decoded(self.source.kind, line)

    def get_state(self) -> dict[str, int]:
        return {"offset": self.file.tell()}

    def set_state(self, state: Mapping[str, object]) -> None:
        offset = state["offset"]
        if not isinstance(offset, int) or offset < self.shard.start:
            raise ValueError(f"invalid file reader position for {self.shard.id}")
        if self.compression is not None:
            # Zstandard streams seek forward only; restoration starts from a fresh decoder.
            self.file.close()
            self.file = _open_lines(self.fs, self.path)
        _seek_lines(self.file, offset)
        # The records read on from the restored offset, after a pass that read them all too.
        self.records = _lines(self.source.kind, self.file, self.names, self.shard.stop)

    def close(self) -> None:
        if self._opened:
            self.file.close()
