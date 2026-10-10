"""Read-mode capabilities, range splits, partition discovery and shard resume."""

import io
import json
import tarfile
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pytest

from dew.data.dataset import DataPartition, Loading
from dew.data.shards import streamed
from dew.data.sinks import export
from dew.data.sources import Files, Memory
from dew.data.storage import Store


def test_memory_columns_snapshot_changes_with_content():
    first = Memory({"x": np.arange(5), "y": np.arange(5) * 3}, shard_rows=2)
    changed = Memory({"x": np.arange(5) + 1, "y": np.arange(5) * 3}, shard_rows=2)
    assert first.snapshot() != changed.snapshot()
    assert first.supports("streamed")
    rows = [row for shard in first.manifest().shards for row in first.manifest().reader(shard)]
    assert [row["x"] for row in rows] == list(range(5))
    with pytest.raises(ValueError, match="different row counts"):
        len(Memory({"x": [1], "y": [1, 2]}).records("indexed"))


def test_jsonl_ranges_cover_rows_once_even_at_line_boundaries():
    rows = [{"id": index, "text": "escaped\nline"} for index in range(30)]
    root = Store(f"memory://jsonl-{uuid4().hex}/rows.jsonl")
    root.write(b"\n".join(json.dumps(row).encode() for row in rows))
    files = Files(root.url, shard_bytes=67)
    assert not files.supports("indexed")
    assert files.supports("streamed")
    actual = [row for shard in files.manifest().shards for row in files.manifest().reader(shard)]
    assert actual == rows
    files.provision()
    assert files.supports("indexed")
    indexed = files.records("indexed")
    assert [indexed[index] for index in range(len(indexed))] == rows
    reader = iter(files.manifest().reader(files.manifest().shards[0]))
    next(reader)
    state = reader.get_state()
    expected = list(reader)
    resumed = iter(files.manifest().reader(files.manifest().shards[0]))
    resumed.set_state(state)
    assert list(resumed) == expected


def test_csv_quoted_newlines_are_never_byte_split():
    root = Store(f"memory://csv-{uuid4().hex}/rows.csv")
    root.write(b'id,text\n0,"two\nlines"\n1,"last"\n')
    files = Files(root.url, shard_bytes=2)
    assert len(files.manifest().shards) == 1
    assert list(files.manifest().reader(files.manifest().shards[0])) == [
        {"id": "0", "text": "two\nlines"}, {"id": "1", "text": "last"}]
    files.provision()
    indexed = files.records("indexed")
    assert indexed[0] == {"id": "0", "text": "two\nlines"}
    assert indexed[1] == {"id": "1", "text": "last"}


def test_arrow_discovers_hive_partitions_and_pushes_projection_and_predicate():
    root = Store(f"memory://arrow-{uuid4().hex}/table")
    fs, path = root.resolve()
    table = pa.table({"value": list(range(20)), "split": ["train"] * 10 + ["held"] * 10,
                      "unused": ["large"] * 20})
    ds.write_dataset(table, path, format="parquet", filesystem=pafs.PyFileSystem(pafs.FSSpecHandler(fs)),
                      partitioning=["split"], partitioning_flavor="hive", max_rows_per_group=3)
    files = Files(root.url, format="parquet", columns=("value", "split"), filter=ds.field("value") >= 15)
    assert not files.supports("indexed")
    actual = [row for shard in files.manifest().shards for row in files.manifest().reader(shard)]
    assert actual == [{"value": index, "split": "held"} for index in range(15, 20)]
    indexed = Files(root.url, format="parquet").records("indexed")
    assert len(indexed) == 20
    assert sorted(indexed[index]["value"] for index in range(20)) == list(range(20))


def test_shard_root_is_disjoint_across_real_workers_and_shares():
    values = Memory([{ "id": index} for index in range(80)], shard_rows=5)
    all_rows = []
    for share in range(2):
        rows = streamed(values.manifest(), partition=DataPartition(share, 2), seed=4,
                        cycle_length=2, loading=Loading(workers=2, threads=2, read_buffer=2))
        import grain.python as grain

        rows = rows.batch(2).mp_prefetch(grain.MultiprocessingOptions(num_workers=2))
        reader = iter(rows)
        batch = next(reader)
        state = reader.get_state()
        rest = list(reader)
        resumed = iter(rows)
        resumed.set_state(state)
        actual = list(resumed)
        assert all(np.array_equal(left["id"], right["id"]) for left, right in zip(rest, actual, strict=True))
        all_rows.extend(int(index) for held in [batch, *rest] for index in held["id"])
        reader.close()
        resumed.close()
    assert sorted(all_rows) == list(range(80))
    assert len(set(all_rows)) == 80


@pytest.mark.parametrize("format", ["parquet", "array_record"])
def test_exports_are_committed_and_readable_at_memory_urls(format):
    root = f"memory://export-{uuid4().hex}"
    rows = [{"id": index, "text": str(index)} for index in range(13)]
    manifest = export(Memory(rows), root, format=format, rows_per_shard=4)
    assert manifest["rows"] == 13
    assert Store(root + "/_SUCCESS").read()
    files = Files(root, format=format)
    if format == "array_record":
        from dew.data.sources.codec import decode

        actual = [decode(row) for shard in files.manifest().shards for row in files.manifest().reader(shard)]
    else:
        actual = [row for shard in files.manifest().shards for row in files.manifest().reader(shard)]
    assert actual == rows


def test_spark_directory_requires_success_marker(tmp_path):
    table = pa.table({"value": [1, 2]})
    ds.write_dataset(table, str(tmp_path), format="parquet")
    with pytest.raises(ValueError, match="_SUCCESS"):
        Files(str(tmp_path), format="parquet", spark=True).records("indexed")
    (tmp_path / "_SUCCESS").write_bytes(b"done")
    assert len(Files(str(tmp_path), format="parquet", spark=True).records("indexed")) == 2


def test_webdataset_groups_members_and_restores_inside_shard():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for key in range(4):
            for extension in ("txt", "jpg"):
                data = f"{key}-{extension}".encode()
                entry = tarfile.TarInfo(f"{key}.{extension}")
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
    root = Store(f"memory://tar-{uuid4().hex}/data.tar")
    root.write(buffer.getvalue())
    files = Files(root.url)
    iterator = iter(files.manifest().reader(files.manifest().shards[0]))
    assert next(iterator) == {"__key__": "0", "txt": b"0-txt", "jpg": b"0-jpg"}
    state = iterator.get_state()
    expected = list(iterator)
    iterator.set_state(state)
    assert list(iterator) == expected
