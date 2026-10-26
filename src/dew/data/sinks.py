"""Immutable Parquet and ArrayRecord exports to filesystem URLs."""

from __future__ import annotations

import hashlib
import io
import tempfile
import uuid
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import grain.python as grain
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from array_record.python.array_record_module import ArrayRecordWriter

from dew.data.dataset import Batch, Closeable
from dew.data.sources.base import Source
from dew.data.sources.codec import encode
from dew.data.sources.files import _arrow_filesystem
from dew.data.steps.text import count_rules, stated_rules
from dew.data.storage import ObjectRef, Store, commit_export

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


def _parquet_shard(target: Store, number: int, batch, partitioning: tuple[str, ...] | None) -> None:
    if any(not isinstance(row, Mapping) for row in batch):
        raise TypeError("Parquet export takes records with named fields")
    table = pa.Table.from_pylist(batch)
    fs, directory = target.resolve()
    fs.makedirs(directory, exist_ok=True)
    ds.write_dataset(
        table,
        directory,
        format="parquet",
        filesystem=_arrow_filesystem(fs),
        partitioning=list(partitioning) if partitioning is not None else None,
        partitioning_flavor="hive" if partitioning is not None else None,
        basename_template=f"part-{number:06d}-{{i}}.parquet",
        existing_data_behavior="overwrite_or_ignore",
    )


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
    format: Literal["parquet", "array_record"] = "parquet",
    rows_per_shard: int = 100_000,
    partitioning: tuple[str, ...] | None = None,
    storage_options: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    length_field: str | None = None,
) -> Mapping[str, object]:
    """Write immutable shards, then conditionally commit their manifest and _SUCCESS.

    Arrow discovers and writes partition columns with Hive directory names.
    Each export owns a UUID directory; racing exporters cannot share shards.
    """
    if rows_per_shard < 1:
        raise ValueError("export rows_per_shard must be positive")
    if format not in ("parquet", "array_record"):
        raise ValueError(f"unknown export format {format!r}")
    if format == "array_record" and partitioning is not None:
        raise ValueError("ArrayRecord export has no partition columns")
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
    try:
        for number, batch in enumerate(_chunks(rows, rows_per_shard)):
            count += len(batch)
            rules.update(stated_rules(count_rules(batch)))
            index = _length_index(root, writer, number, batch, length_field)
            if format == "array_record":
                files.append(_array_shard(root, writer, number, batch, index))
            else:
                _parquet_shard(target, number, batch, partitioning)
        if format == "parquet":
            fs, directory = target.resolve()
            for path in sorted(fs.find(directory)) if fs.exists(directory) else []:
                if not path.endswith(".parquet"):
                    continue
                payload = fs.cat_file(path)
                relative = f"shards/{writer}/" + path[len(directory) :].lstrip("/")
                files.append(
                    {"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                )
        # The rules other than every target's its rows were counted by, which a pack of them by a
        # plan of their lengths could not count (`dew.data.analysis._ruled`).
        return commit_export(root, format=format, codec=1 if format == "array_record" else None, rows=count,
                             files=files, metadata=metadata,
                             certified={"count_rules": [list(rule) for rule in sorted(rules, key=repr)],
                                        "partitioning": None if partitioning is None else list(partitioning)})
    finally:
        if isinstance(rows, Closeable):
            rows.close()
