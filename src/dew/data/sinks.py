"""Immutable exports to filesystem URLs: ArrayRecord shards Dew writes, and Parquet or any other
format a native writer writes from one Arrow stream (`Writer`), all committed by one publisher."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import itertools
import tempfile
import uuid
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, runtime_checkable

import grain.python as grain
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from array_record.python.array_record_module import ArrayRecordWriter

from dew.batch import Batch
from dew.data.dataset import Closeable
from dew.data.sources.base import Source
from dew.data.sources.codec import encode
from dew.data.sources.files import Format, _arrow_filesystem, declared_format
from dew.data.steps.text import count_rules, stated_rules
from dew.data.storage import ObjectRef, Store, commit_export, store_version

if TYPE_CHECKING:
    from dew.data.pipeline import Dataset, Pipeline


def _chunks(values: Iterable[Batch | bytes | None], rows: int) -> Iterator[list[Batch | bytes | None]]:
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) == rows:
            yield batch
            batch = []
    if batch:
        yield batch


def _hub_export(
    values, url: str, *, format, rows_per_shard, partitioning, storage_options, metadata, length_field
) -> Mapping[str, object]:
    from huggingface_hub import CommitOperationAdd, HfApi, HfFileSystem
    from huggingface_hub.hf_file_system import HfFileSystemResolvedRepositoryPath

    fs, path = Store(url, storage_options).resolve()
    if not isinstance(fs, HfFileSystem):
        raise ValueError(f"HF export URL did not resolve to HfFileSystem: {url}")
    resolved = fs.resolve_path(path)
    if not isinstance(resolved, HfFileSystemResolvedRepositoryPath):
        raise ValueError(f"HF export needs a repository URL, not a bucket URL: {url}")
    api = HfApi(token=fs.token)
    parent = api.repo_info(resolved.repo_id, repo_type=resolved.repo_type, revision=resolved.revision).sha
    if parent is None:
        raise ValueError(f"HF export has no parent commit at {url}")
    if fs.exists(path.rstrip("/") + "/_dew_manifest.json"):
        raise FileExistsError(f"HF export is already committed at {url}")
    with tempfile.TemporaryDirectory(prefix="dew-hf-export-") as temporary:
        root = Path(temporary) / "export"
        manifest = export(
            values,
            str(root),
            format=format,
            rows_per_shard=rows_per_shard,
            partitioning=partitioning,
            metadata=metadata,
            length_field=length_field,
        )
        prefix = resolved.path_in_repo.rstrip("/")
        operations = [
            CommitOperationAdd(f"{prefix}/{file.relative_to(root).as_posix()}".lstrip("/"), file)
            for file in sorted(root.rglob("*"))
            if file.is_file() and not file.name.endswith(".cas.lock")
        ]
        api.create_commit(
            repo_id=resolved.repo_id,
            repo_type=resolved.repo_type,
            revision=resolved.revision,
            parent_commit=parent,
            operations=operations,
            commit_message="Publish data export",
        )
        return manifest


def _length_index(root: Store, writer: str, number: int, batch, field: str | None) -> ObjectRef | None:
    if field is None:
        return None
    lengths = []
    for row in batch:
        if not isinstance(row, Mapping) or field not in row:
            raise ValueError(f"export row is missing length_field={field!r}")
        lengths.append(len(row[field]))
    encoded = io.BytesIO()
    sizes = np.asarray(lengths, np.int64)
    np.save(encoded, np.column_stack((sizes, np.cumsum(sizes))), allow_pickle=False)
    path = f"_dew_indexes/{writer}/part-{number:06d}.lengths.npy"
    return {"path": path, **root.child(path).verified(encoded.getvalue())}


def write_array_record(path: Path, payloads: Iterable[bytes]) -> None:
    """`payloads` written to a local ArrayRecord file at `path`, one record each in their order, the
    file closed whether or not they all were: the shard an export and a cache fill each upload."""
    writer = ArrayRecordWriter(str(path), "group_size:1")
    try:
        for payload in payloads:
            writer.write(payload)
    finally:
        writer.close()


def _array_shard(root: Store, writer: str, number: int, batch, index: ObjectRef | None):
    with tempfile.TemporaryDirectory(prefix="dew-export-") as temporary:
        path = Path(temporary) / "part.array_record"
        write_array_record(path, (encode(row) for row in batch))
        relative = f"shards/{writer}/part-{number:06d}.array_record"
        return {
            "path": relative,
            "rows": len(batch),
            "lengths": index,
            **root.child(relative).verified(path.read_bytes()),
        }


@runtime_checkable
class Writer(Protocol):
    """A native writer an export hands its rows to as one Arrow stream (`pyarrow.RecordBatchReader`,
    the stream Arrow's dataset writer, Delta's, Lance's and DuckDB's take, and a Spark session reads
    as a table): it writes them as files of `format`, one `Files` reads, under `directory` of
    `filesystem`, Arrow's over the export's storage. The export then lists what it wrote and commits
    its manifest and `_SUCCESS` after them (`commit_export`), the one publisher every export shares,
    so a reader finds the whole export or, where the writer stopped, none. `Files` reads the files
    in the order of their paths."""

    format: Format

    def write(self, rows: pa.RecordBatchReader, directory: str, filesystem: pafs.FileSystem) -> None: ...


@dataclasses.dataclass(frozen=True)
class _Parquet:
    """Arrow's Parquet dataset writer (`pyarrow.dataset.write_dataset`), the writer of
    `export(format="parquet")`: a file of each batch of the stream, numbered in its order, so a
    read of the export's files by path reads the rows in it, each partition column a Hive
    `name=value` directory, as Spark lays out its own Parquet output."""

    partitioning: tuple[str, ...] | None = None
    format: ClassVar[Format] = "parquet"

    def write(self, rows: pa.RecordBatchReader, directory: str, filesystem: pafs.FileSystem) -> None:
        for number, batch in enumerate(rows):
            ds.write_dataset(
                pa.Table.from_batches([batch]),
                directory,
                format="parquet",
                filesystem=filesystem,
                partitioning=None if self.partitioning is None else list(self.partitioning),
                partitioning_flavor=None if self.partitioning is None else "hive",
                basename_template=f"part-{number:06d}-{{i}}.parquet",
                existing_data_behavior="overwrite_or_ignore",
            )


def _streamed(chunks: Iterator[list[Batch | bytes | None]]) -> pa.RecordBatchReader | None:
    """`chunks` of records with named fields as one Arrow stream, typed by the first; None where
    there is none, which no writer is handed."""

    def named() -> Iterator[list[Batch | bytes | None]]:
        for chunk in chunks:
            if any(not isinstance(row, Mapping) for row in chunk):
                raise TypeError("an export through an Arrow writer takes records with named fields")
            yield chunk

    tables = named()
    first = next(tables, None)
    if first is None:
        return None
    schema = pa.Table.from_pylist(first).schema
    return pa.RecordBatchReader.from_batches(schema, (pa.RecordBatch.from_pylist(chunk, schema=schema)
                                                      for chunk in itertools.chain([first], tables)))


def _written(target: Store, prefix: str) -> list[Mapping[str, object]]:
    """The files a writer left under `target`, each by its path under the export, `prefix` and its
    path under `target`, its bytes and digest, and the store's version of it where the store states one
    (`store_version`), which a reader checks it is still at; the names Spark, Hadoop and Arrow pass over
    as hidden
    (a part starting with `_` or `.`, as a `_SUCCESS` or a checksum's) left out, as their readers
    leave them out."""
    fs, directory = target.resolve()
    files = []
    for path in sorted(fs.find(directory)) if fs.exists(directory) else []:
        relative = path[len(directory):].lstrip("/")
        if any(part.startswith(("_", ".")) for part in PurePosixPath(relative).parts):
            continue
        payload = fs.cat_file(path)
        version = store_version(fs.info(path))
        files.append({"path": f"{prefix}/{relative}", "bytes": len(payload),
                      "sha256": hashlib.sha256(payload).hexdigest(),
                      **({"version": version} if version else {})})
    return files


def passed(values: Source | Dataset) -> Iterator[Batch]:
    """One pass over a store's or a chain's records, in their one order, as `export` writes them: a
    store as a chain of it, by a read it supports, physical shards or none. Close it when done."""
    from dew.data.analysis import _preferred_read
    from dew.data.compile import iterate
    from dew.data.dataset import Loading
    from dew.data.pipeline import Dataset, Pipeline

    chain = values._chain() if isinstance(values, Dataset) else Pipeline(values)
    read = chain.mode or _preferred_read((chain,), provision=True)
    return iter(iterate(chain, read=read, provision=True, loading=chain._loading or Loading()))


def export(
    values: Source | grain.IterDataset | Iterable[Batch | bytes | None] | Pipeline,
    url: str,
    *,
    format: Literal["parquet", "array_record"] | Writer = "parquet",
    rows_per_shard: int = 100_000,
    partitioning: tuple[str, ...] | None = None,
    storage_options: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    length_field: str | None = None,
) -> Mapping[str, object]:
    """Write immutable shards, then conditionally commit their manifest and _SUCCESS.

    ArrayRecord shards are Dew's own, one of `rows_per_shard` records each. Parquet is Arrow's
    writer (`_Parquet`), each partition column a Hive directory; any other `Writer`, a native one
    of the caller's, writes the rows as one Arrow stream. Each export owns a UUID directory, so
    racing exporters cannot share shards, and commits only once its writer has written them all.
    """
    if rows_per_shard < 1:
        raise ValueError("export rows_per_shard must be positive")
    if not isinstance(format, Writer) and format not in ("parquet", "array_record"):
        raise ValueError(f"unknown export format {format!r}")
    if isinstance(format, Writer):
        # A record format `Files` reads: the manifest commits it once (`put_if_absent`), and an
        # export committed under a format no reader opens could never be read or written again there.
        declared_format(format.format, f"export writer {format!r}")
    if format != "parquet" and partitioning is not None:
        raise ValueError("only a Parquet export has partition columns; a writer of its own lays out its own")
    if url.startswith("hf://"):
        return _hub_export(
            values,
            url,
            format=format,
            rows_per_shard=rows_per_shard,
            partitioning=partitioning,
            storage_options=storage_options,
            metadata=metadata,
            length_field=length_field,
        )
    from dew.data.pipeline import Dataset

    rows = passed(values) if isinstance(values, (Source, Dataset)) else iter(values)
    root = Store(url, storage_options)
    writer = uuid.uuid4().hex
    target = root.child(f"shards/{writer}")
    files = []
    count = 0
    rules: set[tuple[int | None, int | None]] = set()

    def counted() -> Iterator[list[Batch | bytes | None]]:
        nonlocal count
        for chunk in _chunks(rows, rows_per_shard):
            count += len(chunk)
            rules.update(stated_rules(count_rules(chunk)))
            yield chunk

    try:
        if format == "array_record":
            for number, batch in enumerate(counted()):
                index = _length_index(root, writer, number, batch, length_field)
                files.append(_array_shard(root, writer, number, batch, index))
            written: Format = "array_record"
        else:
            native = _Parquet(partitioning) if format == "parquet" else format
            stream = _streamed(counted())
            if stream is not None:
                fs, directory = target.resolve()
                fs.makedirs(directory, exist_ok=True)
                native.write(stream, directory, _arrow_filesystem(fs))
            files, written = _written(target, f"shards/{writer}"), native.format
        # The rules other than every target's its rows were counted by, which a pack of them by a
        # plan of their lengths could not count (`dew.data.analysis._ruled`).
        return commit_export(root, format=written, codec=1 if format == "array_record" else None, rows=count,
                             files=files, metadata=metadata,
                             certified={"count_rules": [list(rule) for rule in sorted(rules, key=repr)],
                                        "partitioning": None if partitioning is None else list(partitioning)})
    finally:
        if isinstance(rows, Closeable):
            rows.close()
