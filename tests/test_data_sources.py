"""Read-mode capabilities, range splits, partition discovery and shard resume."""

import dataclasses
import hashlib
import io
import itertools
import json
import os
import pickle
import shutil
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import grain.python as grain
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pytest
from reader_support import assert_a_rewrite_moves_the_identity, assert_resumes_where_it_stopped
from storage_support import moto_bucket
from token_support import export_byte_tokens

from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loading
from dew.data.shards import streamed
from dew.data.sinks import export
from dew.data.sources import Files, Memory
from dew.data.storage import Store


def test_memory_columns_snapshot_changes_with_content():
    first = Memory({"x": np.arange(5), "y": np.arange(5) * 3}, shard_rows=2)
    changed = Memory({"x": np.arange(5) + 1, "y": np.arange(5) * 3}, shard_rows=2)
    assert first.fingerprint() != changed.fingerprint()
    assert first.supports("streamed")
    rows = [row for shard in first.manifest().shards for row in first.manifest().reader(shard)]
    assert [row["x"] for row in rows] == list(range(5))
    assert Memory([{"x": np.int16(4)}]).fingerprint() != Memory([{"x": np.int32(4)}]).fingerprint()
    assert Memory([{"x": 1}], shard_rows=1).fingerprint() != Memory([{"x": 1}], shard_rows=2).fingerprint()


@dataclasses.dataclass(frozen=True)
class Numbered:
    """Adds one to each record's id."""

    def __call__(self, row):
        return {**row, "id": int(row["id"]) + 1}


@pytest.mark.parametrize("table", ["pandas", "arrow"])
@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_memory_reads_a_dataframe_or_arrow_table_as_its_rows(read, table):
    """A DataFrame with an index of its own, or an Arrow table, read by `Memory`: each row a record
    of its columns and nothing of the index, as the same rows given as records read; a changed
    cell changes the store."""
    import pandas as pd
    import pyarrow as pa

    from dew.data.compile import compile
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Pipeline

    columns = {"id": list(range(6)), "text": [f"row {index}" for index in range(6)]}
    frame = pd.DataFrame(columns, index=[10 * index for index in range(6)])
    given = frame if table == "pandas" else pa.table(columns)
    records = [{"id": index, "text": f"row {index}"} for index in range(6)]

    def read_back(memory):
        chain = Pipeline(memory).map(Numbered(), identity="numbered@1").batch(2)
        reader = compile(chain, read=read, loading=Loading(workers=0, threads=1, read_buffer=1)).train(
            DataPartition())
        try:
            return [(np.asarray(batch["id"]).tolist(), list(batch["text"]))
                    for batch in (next(reader) for _ in range(3))]
        finally:
            reader.close()

    assert read_back(Memory(given)) == read_back(Memory(records))
    assert Memory(given).fingerprint()["sha256"] == Memory(records).fingerprint()["sha256"]
    changed = frame.assign(text=[*columns["text"][:5], "changed"])
    assert Memory(changed).fingerprint()["sha256"] != Memory(frame).fingerprint()["sha256"]


def test_memory_columns_of_different_row_counts_are_refused():
    assert len(Memory({"x": [1, 2], "y": [1, 2]}).records("indexed")) == 2
    with pytest.raises(ValueError):
        len(Memory({"x": [1], "y": [1, 2]}).records("indexed"))


def test_tfds_source_metadata_and_decoder_identity_need_no_data_scan():
    import tensorflow_datasets as tfds

    from dew.data.sources import TFDS

    root = Path(__file__).parent / "fixtures" / "tfds" / "dew_images" / "1.0.0"
    decoded = TFDS("dew_images", path=str(root))
    raw = TFDS("dew_images", path=str(root), decoders={"image": tfds.decode.SkipDecoding()})
    assert decoded.metadata()["rows"] == 16
    assert decoded.metadata()["class_names"]["label"] == tuple(
        (root / "label.labels.txt").read_text().splitlines()
    )
    assert decoded.fingerprint() != raw.fingerprint()
    assert isinstance(raw.records("indexed")[0]["image"], bytes)
    assert raw.with_split("test").metadata()["rows"] == 4


def test_remote_tfds_requires_explicit_provisioning_before_native_reads():
    from dew.data.sources import TFDS

    fixture = Path(__file__).parent / "fixtures" / "tfds" / "dew_images" / "1.0.0"
    root = Store(f"memory://tfds-{uuid4().hex}/1.0.0")
    for path in fixture.iterdir():
        root.child(path.name).write(path.read_bytes())
    source = TFDS("dew_images", path=root.url)
    assert not source.supports("indexed")
    with pytest.raises(FileNotFoundError):
        source.records("indexed")
    source.provision()
    assert source.supports("indexed")
    assert len(source.records("indexed")) == 16
    assert not Path(source.directory, "dew_images-test.array_record-00000-of-00001").exists()


def test_tfds_label_definitions_participate_in_source_identity(tmp_path):
    import shutil

    from dew.data.sources import TFDS

    fixture = Path(__file__).parent / "fixtures" / "tfds" / "dew_images" / "1.0.0"
    shutil.copytree(fixture, tmp_path, dirs_exist_ok=True)
    before = TFDS("dew_images", path=str(tmp_path)).fingerprint()
    (tmp_path / "label.labels.txt").write_text("rouge\nbleu\n")
    after = TFDS("dew_images", path=str(tmp_path)).fingerprint()
    assert before != after


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


@pytest.mark.parametrize("extension", ["gz", "zst"])
@pytest.mark.parametrize("transport", ["local", "memory"])
@pytest.mark.parametrize("discovery", ["inferred", "explicit", "directory"])
def test_compressed_jsonl_reads_and_resumes_in_every_read_mode(tmp_path, extension, transport, discovery):
    import gzip

    import zstandard

    rows = [{"id": index, "text": "long escaped\nline " * 30} for index in range(30)]
    payload = b"\n" + b"\n\n".join(json.dumps(row).encode() for row in rows)
    compressed = gzip.compress(payload) if extension == "gz" else zstandard.ZstdCompressor().compress(payload)
    root = Store(str(tmp_path) if transport == "local" else f"memory://compressed-{uuid4().hex}")
    file = root.child(f"rows.jsonl.{extension}")
    file.write(compressed)
    url = root.url if discovery == "directory" else file.url
    files = Files(url, format=None if discovery == "inferred" else "jsonl", shard_bytes=7)
    manifest = files.manifest()
    assert len(manifest.shards) == 1
    reader = iter(manifest.reader(manifest.shards[0]))
    try:
        assert next(reader) == rows[0]
        saved = reader.get_state()
        assert list(reader) == rows[1:]
        reader.set_state(saved)
        assert list(reader) == rows[1:]
    finally:
        reader.close()
    resumed = iter(manifest.reader(manifest.shards[0]))
    try:
        resumed.set_state(saved)
        assert list(resumed) == rows[1:]
    finally:
        resumed.close()
    assert list(files.records("memory")) == rows
    files.provision()
    assert files.supports("indexed")
    indexed = files.records("indexed")
    assert len(indexed) == len(rows)
    assert [indexed[index] for index in (29, 0, 15, 29)] == [rows[index] for index in (29, 0, 15, 29)]


def test_csv_quoted_newlines_are_never_byte_split():
    root = Store(f"memory://csv-{uuid4().hex}/rows.csv")
    root.write(b'id,text\n0,"two\nlines"\n1,"last"\n')
    files = Files(root.url, shard_bytes=2)
    assert len(files.manifest().shards) == 1
    assert list(files.manifest().reader(files.manifest().shards[0])) == [
        {"id": "0", "text": "two\nlines"},
        {"id": "1", "text": "last"},
    ]
    files.provision()
    indexed = files.records("indexed")
    assert indexed[0] == {"id": "0", "text": "two\nlines"}
    assert indexed[1] == {"id": "1", "text": "last"}


def test_file_catalog_keeps_distinct_nested_names_including_underscore_directories(tmp_path):
    names = [f"speaker-{speaker}/{clip}/00001.mp4" for speaker in (1, 2) for clip in ("_clip", "other")]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
    rows = Files(str(tmp_path), format="files").records("indexed")
    assert [row["path"] for row in rows] == [(tmp_path / name).as_uri() for name in sorted(names)]


def test_csv_large_quoted_fields_survive_streaming_and_offset_provisioning():
    import csv

    text = "a" * 140_000 + "\n" + "b" * 140_000
    root = Store(f"memory://csv-large-{uuid4().hex}/rows.csv")
    root.write(('id,text\n0,"' + text + '"\n').encode())
    files = Files(root.url)
    # A field past Python's default limit reads under the limit its caller raises (`Files`).
    previous = csv.field_size_limit(len(text) + 1)
    try:
        assert list(files.manifest().reader(files.manifest().shards[0])) == [{"id": "0", "text": text}]
        files.provision()
        assert files.records("indexed")[0] == {"id": "0", "text": text}
    finally:
        csv.field_size_limit(previous)


def test_arrow_discovers_hive_partitions_and_pushes_projection_and_predicate():
    root = Store(f"memory://arrow-{uuid4().hex}/table")
    fs, path = root.resolve()
    table = pa.table(
        {"value": list(range(20)), "split": ["train"] * 10 + ["held"] * 10, "unused": ["large"] * 20}
    )
    ds.write_dataset(
        table,
        path,
        format="parquet",
        filesystem=pafs.PyFileSystem(pafs.FSSpecHandler(fs)),
        partitioning=["split"],
        partitioning_flavor="hive",
        max_rows_per_group=3,
    )
    files = Files(root.url, format="parquet", columns=("value", "split"), where=ds.field("value") >= 15)
    assert not files.supports("indexed")
    actual = [row for shard in files.manifest().shards for row in files.manifest().reader(shard)]
    assert actual == [{"value": index, "split": "held"} for index in range(15, 20)]
    indexed = Files(root.url, format="parquet").records("indexed")
    assert len(indexed) == 20
    assert sorted(indexed[index]["value"] for index in range(20)) == list(range(20))


def test_a_predicate_held_in_memory_keeps_the_rows_it_selects(tmp_path):
    """A memory read of a filtered parquet source holds the rows the predicate keeps, as a
    streamed read does, from a row group it filters in part and one it filters whole."""
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"x": [0, 1, 2, 3]}), tmp_path / "rows.parquet", row_group_size=2)
    for where, kept in ((ds.field("x") > 0, [1, 2, 3]), (ds.field("x") > 1, [2, 3])):
        files = Files(str(tmp_path / "rows.parquet"), where=where)
        streamed_rows = [row for shard in files.manifest().shards for row in files.manifest().reader(shard)]
        assert [row["x"] for row in files.records("memory")] == [row["x"] for row in streamed_rows] == kept


@pytest.mark.parametrize("format", ["jsonl", "csv"])
def test_a_projected_line_source_reads_only_its_columns_in_every_read_mode(tmp_path, format):
    """`columns=` selects a mapping row's fields however a line source is read, as an Arrow
    source's projection does, and a field the rows lack is refused by name."""
    rows = [{"id": str(index), "unused": "x" * index} for index in range(4)]
    path = tmp_path / f"rows.{format}"
    path.write_text("id,unused\n" + "".join(f"{row['id']},{row['unused']}\n" for row in rows)
                    if format == "csv" else "".join(json.dumps(row) + "\n" for row in rows))
    Files(str(path)).provision()
    projected = Files(str(path), columns=("id",))
    expected = [{"id": row["id"]} for row in rows]
    manifest = projected.manifest()
    assert [row for shard in manifest.shards for row in manifest.reader(shard)] == expected
    assert list(projected.records("memory")) == expected
    indexed = projected.records("indexed")
    assert [indexed[index] for index in range(len(indexed))] == expected
    with pytest.raises(ValueError):
        list(Files(str(path), columns=("missing",)).records("memory"))


@pytest.mark.parametrize("read", ["indexed", "streamed"])
def test_a_csv_field_past_pythons_limit_reads_by_the_limit_its_caller_sets(tmp_path, read):
    """CSV is parsed by Python's csv module under the process's own field size limit, which a
    source never changes: a 200 KiB quoted field is refused under the default limit, and under a
    limit its caller raised it reads through a loader of four reader threads at its provisioned
    offsets and resumes where it stopped. A source once raised the limit for every parser in the
    process each time it opened a file."""
    import csv

    from dew.data.compile import compile

    texts = [f"row {index}" if index != 2 else "x" * (200 * 1024) + "\nand more" for index in range(6)]
    expected = [(str(index), text) for index, text in enumerate(texts)]
    path = tmp_path / "rows.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "text"])
        writer.writerows([str(index), text] for index, text in enumerate(texts))

    def rows(reader) -> list[tuple[str, str]]:
        return [(str(batch["id"][0]), str(batch["text"][0])) for batch in reader]

    previous = csv.field_size_limit(128 * 1024)
    try:
        refused = Files(str(path))
        with pytest.raises(csv.Error):
            list(refused.records("memory"))
        csv.field_size_limit(1024 * 1024)
        Files(str(path)).provision()
        data = compile(Files(str(path)).repeat(1).batch(1), read=read,
                       loading=Loading(workers=0, threads=4, read_buffer=4))
        whole = data.train(DataPartition())
        try:
            assert rows(whole) == expected
        finally:
            whole.close()
        stopped = data.train(DataPartition())
        try:
            head = rows(itertools.islice(stopped, 3))
            state = stopped.get_state()
        finally:
            stopped.close()
        resumed = data.train(DataPartition())
        try:
            resumed.set_state(state)
            assert head + rows(resumed) == expected
        finally:
            resumed.close()
        assert csv.field_size_limit() == 1024 * 1024
    finally:
        csv.field_size_limit(previous)


@pytest.mark.parametrize("trickled", ["headers", "body"])
def test_a_fetch_from_a_server_that_trickles_its_response_ends_at_its_timeout(trickled):
    """A server that sends its headers or its body a byte every tenth of a second holds a fetch
    no longer than its timeout: the attempt is dropped as transient, where it once read on byte
    by byte, each byte inside the socket's timeout."""
    import http.server
    import time

    from dew.data.pipeline import Rejected
    from dew.data.sources.urls import Fetch

    class Trickle(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if trickled == "headers":
                self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Slow: ")
            else:
                self.send_response(200)
                self.send_header("Content-Length", "60")
                self.end_headers()
            for _ in range(60):
                self.wfile.write(b"x")
                self.wfile.flush()
                time.sleep(0.1)

        def log_message(self, *arguments):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Trickle)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        began = time.monotonic()
        fetched = Fetch(timeout=1.0)({"url": f"http://127.0.0.1:{server.server_port}/"})
        waited = time.monotonic() - began
    finally:
        server.shutdown()
        server.server_close()
    assert isinstance(fetched, Rejected) and not fetched.permanent
    assert waited < 1.5


def test_shard_root_is_disjoint_across_real_workers_and_shares():
    values = Memory([{"id": index} for index in range(80)], shard_rows=5)
    all_rows = []
    for share in range(2):
        rows = streamed(
            values.manifest(),
            partition=DataPartition(share, 2),
            seed=4,
            cycle_length=2,
            loading=Loading(workers=2, threads=2, read_buffer=2),
        )
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
    actual = [row for shard in files.manifest().shards for row in files.manifest().reader(shard)]
    assert actual == rows
    with pytest.raises(ValueError):
        Files(root, format="jsonl").records("memory")


def test_a_source_exports_the_records_its_pipeline_exports():
    """A map-style Torch source has records but no physical shards; exported directly, it writes
    the records its pipeline does, where a direct export once needed a shard manifest."""
    import torch
    from torch.utils.data import TensorDataset

    from dew.data import from_torch

    stored = from_torch(TensorDataset(torch.arange(10).reshape(5, 2)), fields=("id",), identity="five@1")

    def exported(values) -> list:
        root = f"memory://export-{uuid4().hex}"
        export(values, root, rows_per_shard=2)
        files = Files(root, format="parquet")
        return [np.asarray(row["id"]).tolist() for shard in files.manifest().shards
                for row in files.manifest().reader(shard)]

    assert exported(stored) == exported(stored._chain()) == [[2 * index, 2 * index + 1] for index in range(5)]


def test_spark_directory_requires_success_marker(tmp_path):
    table = pa.table({"value": [1, 2]})
    ds.write_dataset(table, str(tmp_path), format="parquet")
    with pytest.raises(ValueError):
        Files(str(tmp_path), format="parquet", spark=True).records("indexed")
    (tmp_path / "_SUCCESS").write_bytes(b"done")
    assert len(Files(str(tmp_path), format="parquet", spark=True).records("indexed")) == 2


def test_a_corrupt_export_manifest_is_refused_and_never_falls_back_to_raw_shards():
    root = Store(f"memory://corrupt-export-{uuid4().hex}")
    export(Memory([{"id": 1}]), root.url)
    assert len(Files(root.url, format="parquet").records("indexed")) == 1
    root.child("_dew_manifest.json").write(b"not-json")
    with pytest.raises(ValueError):
        Files(root.url, format="parquet").records("indexed")


def test_export_manifest_cannot_reference_an_object_outside_its_root():
    """An export's manifest reads; the same manifest naming its shard by a path
    that leaves the root is refused."""
    root = Store(f"memory://escaping-export-{uuid4().hex}/dataset")
    export(Memory([{"id": 1}]), root.url)
    assert len(Files(root.url).manifest().shards) == 1
    manifest = json.loads(root.child("_dew_manifest.json").read())
    manifest["files"][0]["path"] = "../" + manifest["files"][0]["path"]
    root.child("_dew_manifest.json").write(json.dumps(manifest).encode())
    with pytest.raises(ValueError):
        Files(root.url).manifest()


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


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_text_preserves_exact_line_terminators_through_byte_tokenization(read):
    from dew.data.text import Tokenize

    root = Store(f"memory://text-{uuid4().hex}/source.txt")
    original = b"first\r\nsecond\n\nlast without newline"
    root.write(original)
    files = Files(root.url, shard_bytes=9)
    if read == "indexed":
        files.provision()
    if read == "streamed":
        records = [row for shard in files.manifest().shards for row in files.manifest().reader(shard)]
    else:
        indexed = files.records(read)
        records = [indexed[index] for index in range(len(indexed))]
    tokens = np.concatenate([Tokenize().map(row)["text"] for row in records])
    assert tokens.tolist() == list(original)


def test_a_listing_keeps_the_files_whose_names_end_in_its_suffixes_in_any_case(tmp_path):
    """A directory's media beside its notes: suffixes keep the media, whatever case either is
    written in, as the store is read, here and under a glob."""
    for name in ("a.MP4", "b.webm", "notes.txt", "list.csv"):
        (tmp_path / name).write_bytes(b"")
    for url in (str(tmp_path), str(tmp_path / "*")):
        listing = Files(url, format="files", suffixes=("mp4", ".WEBM"))
        names = sorted(row["path"].rsplit("/", 1)[-1] for row in listing.records("memory"))
        assert names == ["a.MP4", "b.webm"]


def test_a_token_export_reads_as_its_one_split_and_refuses_another(tmp_path):
    """An export holds one split: named "train", the default, it reads as itself; named another,
    it is refused rather than read as that split."""
    from dew.data.sources import TokenFiles, source

    export([{"text": np.arange(3, dtype=np.int32)}], str(tmp_path), format="array_record",
           length_field="text", metadata={"token_store": True, "dtype": "int32", "tokenizer": "byte",
                                          "vocab_size": 256, "eos_id": None})
    assert isinstance(source(str(tmp_path), split="train"), TokenFiles)
    with pytest.raises(ValueError):
        source(str(tmp_path), split="validation")


@pytest.mark.parametrize("transport", ["local", "memory"])
def test_two_exports_of_the_same_rows_are_the_same_store_and_of_other_rows_another(tmp_path, transport):
    """Each host of a run exports its corpus itself, and the run checks that every host reads the same
    data: two exports of the same rows, each writer naming its shards its own way, are one store,
    as tokens and as files. On a host's disk, wherever each mounted it; remote, where a store is
    named by its URL, the same rows written again in its place. An export of other rows is another."""
    import fsspec

    from dew.data.sources import Files, TokenFiles

    metadata = {"token_store": True, "dtype": "int32", "tokenizer": "byte", "vocab_size": 256, "eos_id": 0}
    base = str(tmp_path) if transport == "local" else f"memory://exports-{uuid4().hex}"

    def exported(name, last):
        url = f"{base}/{name}"
        rows = [{"text": np.asarray([1, 2, 0], np.int32)}, {"text": np.asarray([3, last, 0], np.int32)}]
        export(rows, url, format="array_record", length_field="text", metadata=metadata)
        return url, TokenFiles(url).fingerprint(), Files(url).fingerprint()

    url, *one = exported("one", 4)
    if transport == "memory":
        fsspec.filesystem("memory").rm(url, recursive=True)
    _, *again = exported("one" if transport == "memory" else "again", 4)
    _, *other = exported("other", 5)
    assert one == again
    assert one[0] != other[0] and one[1] != other[1]


def test_partitioned_exports_of_the_same_values_under_other_partitions_are_other_stores(tmp_path):
    """Rows exported partitioned by a field keep that field in the partition's directory alone, so
    two exports whose files hold the same bytes under other partitions read other rows: other
    stores; the same rows under the same partitions are one."""
    from dew.data.sources import Files

    def exported(name, language):
        export([{"text": "hello", "language": language}], str(tmp_path / name), partitioning=("language",))
        return Files(str(tmp_path / name)).fingerprint()

    english, again, french = exported("en", "en"), exported("en-again", "en"), exported("fr", "fr")
    assert english == again != french


def test_a_token_export_of_no_documents_reads_as_an_empty_corpus(tmp_path):
    """An export whose input yields no rows commits no shards, and reads back as no tokens and no
    documents rather than failing to open."""
    from dew.data.compile import iterate
    from dew.data.pipeline import Pipeline
    from dew.data.sources.tokens import TokenFiles

    manifest = export([], str(tmp_path), format="array_record", length_field="text",
                      metadata={"token_store": True, "dtype": "int32", "tokenizer": "byte",
                                "vocab_size": 256, "eos_id": None})
    assert manifest["rows"] == 0 and manifest["files"] == []
    stored = TokenFiles(str(tmp_path))
    assert len(stored.tokens) == 0 and stored.metadata()["tokens"] == 0
    assert stored.fingerprint()["shards"] == []
    assert list(iterate(Pipeline(stored), read="memory")) == []


def test_exported_token_lengths_support_indexed_spans_without_record_scan():
    from dew.data.sources.tokens import TokenFiles

    root = f"memory://token-export-{uuid4().hex}"
    export(
        [{"text": np.arange(start, start + 3, dtype=np.int32)} for start in (0, 3, 6)],
        root,
        format="array_record",
        rows_per_shard=2,
        length_field="text",
        metadata={
            "token_store": True,
            "dtype": "int32",
            "tokenizer": "byte",
            "vocab_size": 256,
            "eos_id": None,
        },
    )
    source = TokenFiles(root)
    assert len(source.tokens) == 9
    assert source.tokens[2:8].tolist() == list(range(2, 8))
    assert source.metadata()["tokenizer"] == "byte"
    from dew.data.sources import source as resolve

    assert isinstance(resolve(root), TokenFiles)
    # An export holds one split: validation tokens are an export of their own.
    with pytest.raises(ValueError):
        source.holdout(split="val")


@pytest.fixture
def token_span_store(tmp_path):
    """A token export of six documents, three of them empty, two to a shard, each ended by EOS (0)."""
    documents = [[], [], [1, 2, 0], [], [3, 0], [4, 5, 6, 0]]
    metadata = {"token_store": True, "dtype": "int32", "tokenizer": "byte",
                "vocab_size": 256, "eos_id": 0}
    export([{"text": np.asarray(document, np.int32)} for document in documents], str(tmp_path),
           format="array_record", rows_per_shard=2, length_field="text", metadata=metadata)
    return tmp_path


def test_flat_and_sharded_token_indexes_read_the_same_spans_and_documents(token_span_store):
    from dew.data.compile import iterate
    from dew.data.pipeline import Pipeline
    from dew.data.sources.tokens import TokenFiles

    stored = TokenFiles(str(token_span_store)).provision()
    tokens = stored.tokens
    expected = np.asarray([1, 2, 0, 3, 0, 4, 5, 6, 0], np.int32)
    assert len(tokens) == len(expected)
    for span in (slice(None), slice(1, 8), slice(-5, -1), slice(3, 5), slice(5, 5), slice(8, 2)):
        np.testing.assert_array_equal(tokens[span], expected[span])
    with pytest.raises(ValueError):
        tokens[::2]
    assert stored.documented
    assert stored.document_ends().tolist() == [3, 5, 9]
    for read in ("memory", "indexed", "streamed"):
        rows = iterate(Pipeline(stored), read=read, loading=Loading(workers=0, threads=1, read_buffer=0))
        assert [row["text"].tolist() for row in rows] == [[1, 2, 0], [3, 0], [4, 5, 6, 0]]
    for shard, index in zip(tokens.shards, tokens.indexes, strict=True):
        assert shard.columns is index.values
        assert isinstance(shard.columns, np.memmap)
    reopened = pickle.loads(pickle.dumps(tokens))
    np.testing.assert_array_equal(reopened[1:8], expected[1:8])
    assert all(isinstance(shard.columns, np.memmap) for shard in reopened.shards)


def test_each_token_document_is_checked_against_its_producer_length(token_span_store):
    from dew.data.sources.tokens import TokenFiles

    root = token_span_store
    TokenFiles(str(root)).provision()
    manifest_path = root / "_dew_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    path = root / manifest["files"][1]["lengths"]["path"]
    columns = np.load(path, allow_pickle=False)
    columns[0, 0] += 1
    columns[:, 1] += 1
    encoded = io.BytesIO()
    np.save(encoded, columns, allow_pickle=False)
    path.write_bytes(encoded.getvalue())
    manifest["files"][1]["lengths"].update(
        bytes=len(encoded.getvalue()), sha256=hashlib.sha256(encoded.getvalue()).hexdigest())
    manifest_path.write_text(json.dumps(manifest))
    stored = TokenFiles(str(root))
    with pytest.raises(ValueError):
        stored.tokens[:3]


@pytest.mark.parametrize("format_args", [[], ["--format", "array_record"]])
def test_cli_token_exports_are_readable_by_the_dispatched_token_source(tmp_path, format_args):
    from dew.data.sources import TokenFiles, source
    from dew.data.text import ByteTokenizer

    documents = ["first café", "next document", "last"]
    destination = export_byte_tokens(tmp_path, documents, "--rows-per-shard", "2", *format_args)
    stored = source(str(destination))
    assert isinstance(stored, TokenFiles)
    vocabulary = ByteTokenizer()
    expected = np.concatenate([np.append(vocabulary.encode(text), vocabulary.eos_id) for text in documents])
    assert stored.tokens[:].tolist() == expected.tolist()
    assert stored.supports("indexed")
    assert stored.metadata()["eos_id"] == vocabulary.eos_id
    assert stored.metadata()["tokens"] == len(expected)
    records = stored.records("indexed")
    actual = np.concatenate([records[index]["text"] for index in range(len(records))])
    assert actual.tolist() == expected.tolist()


@pytest.mark.parametrize("read", ["streamed", "indexed"])
def test_a_parquet_store_reads_with_the_credentials_its_profile_names(tmp_path, monkeypatch, read):
    """An S3 Parquet store whose storage options name an AWS profile reads its rows with that
    profile's credentials, on an endpoint that refuses every other request: every setting the
    store was given reaches the reads, not only those Arrow's own S3 client can take."""
    import boto3
    import fsspec
    import pyarrow.parquet as pq
    from moto.core import set_initial_no_auth_action_count
    from moto.server import ThreadedMotoServer

    from dew.data.compile import iterate
    from dew.data.pipeline import Pipeline

    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
                 "AWS_DEFAULT_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    _, port = server.get_host_and_port()
    endpoint = f"http://127.0.0.1:{port}"

    # The endpoint authenticates every request after the three that make the profile's user.
    @set_initial_no_auth_action_count(3)
    def read_through_the_profile():
        iam = boto3.Session().client("iam", endpoint_url=endpoint, region_name="us-east-1",
                                     aws_access_key_id="setup", aws_secret_access_key="setup")
        # The endpoint's users outlive it in this process, so each test names its own.
        user = "reader-" + uuid4().hex
        iam.create_user(UserName=user)
        iam.put_user_policy(UserName=user, PolicyName="s3", PolicyDocument=json.dumps({
            "Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": "s3:*", "Resource": "*"}]}))
        key = iam.create_access_key(UserName=user)["AccessKey"]
        (tmp_path / "credentials").write_text(
            f"[reader]\naws_access_key_id = {key['AccessKeyId']}\n"
            f"aws_secret_access_key = {key['SecretAccessKey']}\n")
        options = {"profile": "reader", "endpoint_url": endpoint}
        bucket = "profiles-" + uuid4().hex
        remote = fsspec.filesystem("s3", **options)
        remote.mkdir(bucket)
        written = io.BytesIO()
        pq.write_table(pa.Table.from_pylist([{"id": index} for index in range(5)]), written)
        remote.pipe(f"{bucket}/rows.parquet", written.getvalue())
        source = Files(f"s3://{bucket}/rows.parquet", storage_options=options)
        return sorted(int(row["id"]) for row in iterate(Pipeline(source), read=read))

    try:
        assert read_through_the_profile() == list(range(5))
    finally:
        server.stop()


def test_an_anonymous_storage_profile_reads_parquet_though_it_names_keys():
    """A storage profile over S3 that reads anonymously reads a Parquet store's rows, though it
    names keys too, which an anonymous client does not sign with."""
    import fsspec
    import pyarrow.parquet as pq
    from moto.server import ThreadedMotoServer

    from dew.data.compile import iterate
    from dew.data.pipeline import Pipeline
    from dew.data.storage import _profile

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    try:
        _, port = server.get_host_and_port()
        endpoint = f"http://127.0.0.1:{port}"
        bucket = "public-" + uuid4().hex
        remote = fsspec.filesystem("s3", key="testing", secret="testing", endpoint_url=endpoint)
        remote.mkdir(bucket, acl="public-read")
        written = io.BytesIO()
        pq.write_table(pa.Table.from_pylist([{"id": index} for index in range(5)]), written)
        remote.pipe(f"{bucket}/rows.parquet", written.getvalue(), ACL="public-read")
        _profile("publics3", "s3a", {"key": "unused", "secret": "unused", "anon": True,
                                     "endpoint_url": endpoint})
        source = Files(f"publics3://{bucket}/rows.parquet")
        assert sorted(int(row["id"]) for row in iterate(Pipeline(source), read="streamed")) == list(range(5))
    finally:
        server.stop()


def test_a_token_export_reads_through_the_storage_options_it_is_given(tmp_path):
    """A committed token export that only its storage options reach, an S3 endpoint of its own,
    reads its manifest, metadata and tokens through them, as its shards and index do."""
    import fsspec

    from dew.data.sources import TokenFiles
    from dew.data.text import ByteTokenizer

    destination = export_byte_tokens(tmp_path, ("first", "next"), "--format", "array_record")
    with moto_bucket("tokens") as (bucket, options):
        remote = fsspec.filesystem("s3", **options)
        remote.put(str(destination), f"{bucket}/export", recursive=True)
        stored = TokenFiles(f"s3://{bucket}/export", storage_options=options)
        vocabulary = ByteTokenizer()
        assert stored.meta["eos_id"] == vocabulary.eos_id
        assert stored.tokens[:].tolist() == [*vocabulary.encode("first"), vocabulary.eos_id,
                                             *vocabulary.encode("next"), vocabulary.eos_id]


def test_cli_refuses_an_unsupported_token_export_format_before_writing(tmp_path):
    from dew.cli.main import main

    input = tmp_path / "documents.jsonl"
    input.write_text('{"text":"a document"}\n')
    destination = tmp_path / "tokens"
    with pytest.raises(ValueError):
        main(["data", "export", str(input), str(destination), "--tokenizer", "byte", "--format", "parquet"])
    assert not destination.exists()


def test_cli_record_exports_still_default_to_parquet(tmp_path):
    from dew.cli.main import main
    from dew.data.sources import source

    input = tmp_path / "documents.jsonl"
    input.write_text('{"text":"a document"}\n')
    destination = tmp_path / "records"
    assert main(["data", "export", str(input), str(destination)]) == 0
    stored = source(str(destination))
    assert isinstance(stored, Files)
    assert stored.kind == "parquet"
    assert stored.records("indexed")[0] == {"text": "a document"}


def test_cli_exports_the_chain_of_the_file_it_names_and_no_other(tmp_path):
    """Two files of one name in two directories each export their own chain, and a file named as a
    module the process holds elsewhere (the standard library's `json`), or as a package's module
    (`email.utils`), is refused, not read as it."""
    from dew.cli.main import main
    from dew.data.sources import source

    for corpus in ("first", "second"):
        (tmp_path / corpus).mkdir()
        (tmp_path / corpus / "corpus.py").write_text(
            f"from dew.data import Memory\ndata = Memory([{{'text': '{corpus}'}}])\n")
        exported = tmp_path / f"{corpus}-out"
        assert main(["data", "export", str(tmp_path / corpus / "corpus.py"), str(exported)]) == 0
        assert source(str(exported)).records("indexed")[0] == {"text": corpus}
    for named in ("json.py", "email.utils.py"):
        (tmp_path / named).write_text("from dew.data import Memory\ndata = Memory([{'text': 'mine'}])\n")
        with pytest.raises(ValueError):
            main(["data", "export", str(tmp_path / named), str(tmp_path / f"{named}-out")])


@pytest.mark.parametrize("read", [None, "memory", "indexed"])
def test_snapshot_uses_requested_or_preferred_read_and_loading(tmp_path, read):
    import time

    from torch.utils.data import Dataset

    from dew.data import from_torch

    lock = threading.Lock()
    active = maximum = 0

    class Counted(Dataset):
        def __len__(self):
            return 32

        def __getitem__(self, index):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            try:
                time.sleep(0.01)
                return {"id": index}
            finally:
                with lock:
                    active -= 1

    chain = from_torch(Counted(), identity="snapshot@1").map(lambda row: {"id": row["id"] * 2}).threads(2)
    if read is not None:
        chain = chain.read(read)
    destination = tmp_path / "snapshot"
    manifest = chain.snapshot(str(destination), rows_per_shard=7)
    stored = Files(str(destination)).records("indexed")
    assert [int(stored[index]["id"]) for index in range(len(stored))] == list(range(0, 64, 2))
    assert manifest["rows"] == 32
    assert maximum <= 2


def test_snapshot_provisions_the_requested_indexed_read(tmp_path):
    input = tmp_path / "rows.jsonl"
    input.write_text('{"id":0}\n{"id":1}\n')
    files = Files(str(input))
    assert not files.supports("indexed")
    destination = tmp_path / "snapshot"
    files.read("indexed").threads(1).snapshot(str(destination))
    assert Files(str(input)).supports("indexed")
    stored = Files(str(destination)).records("indexed")
    assert [stored[index] for index in range(len(stored))] == [{"id": 0}, {"id": 1}]


def test_remote_indexed_arrayrecords_read_only_the_selected_physical_shard(monkeypatch):
    root = Store(f"memory://lazy-records-{uuid4().hex}")
    manifest = export(
        [{"id": index} for index in range(12)], root.url, format="array_record", rows_per_shard=4
    )
    filesystem, _ = root.resolve()
    downloaded = []
    get_file = filesystem.get_file

    def download(remote, local, **kwargs):
        downloaded.append(remote)
        return get_file(remote, local, **kwargs)

    monkeypatch.setattr(filesystem, "get_file", download)
    rows = Files(root.url, format="array_record").records("indexed")
    assert len(rows) == 12
    assert downloaded == []
    assert rows[9] == {"id": 9}
    assert len(downloaded) == 1
    assert downloaded[0].endswith(manifest["files"][2]["path"])


@pytest.fixture
def arrayrecord_paths(tmp_path):
    from array_record.python.array_record_module import ArrayRecordWriter

    paths = tuple(str(tmp_path / f"{shard}.array_record") for shard in range(Loading().threads + 1))
    for shard, path in enumerate(paths):
        writer = ArrayRecordWriter(path, "group_size:1")
        for index in range(8):
            writer.write(f"{shard}:{index}".encode())
        writer.close()
    return paths


def test_arrayrecords_threaded_reads_overlap(arrayrecord_paths, monkeypatch):
    import grain.python as grain

    from dew.data.sources import records

    source = records.ArrayRecords(arrayrecord_paths[:1], counts=(8,))
    active = maximum = 0
    lock = threading.Lock()
    entered = threading.Barrier(3)
    released = threading.Event()
    underlying_read = grain.ArrayRecordDataSource.__getitem__

    def blocked_read(reader, index):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            if index < 2:
                entered.wait(timeout=5)
                assert released.wait(timeout=5)
            return underlying_read(reader, index)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(grain.ArrayRecordDataSource, "__getitem__", blocked_read)
    try:
        with ThreadPoolExecutor(2) as pool:
            pending = [pool.submit(source.__getitem__, index) for index in range(8)]
            try:
                entered.wait(timeout=5)
                assert maximum == 2
            finally:
                released.set()
            assert [future.result(timeout=5) for future in pending] == [
                f"0:{index}".encode() for index in range(8)
            ]
    finally:
        source.close()


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("retire", ["eviction", "close"])
def test_arrayrecords_keep_busy_readers_alive_during_eviction_or_close(
    arrayrecord_paths, monkeypatch, fail, retire
):
    import grain.python as grain

    from dew.data.sources import records

    source = records.ArrayRecords(arrayrecord_paths, counts=(8,) * len(arrayrecord_paths))
    entered = threading.Barrier(3)
    released = [threading.Event(), threading.Event()]
    first_reader = []
    underlying_read = grain.ArrayRecordDataSource.__getitem__
    failed_read = ValueError("failed child read")

    def descriptors(path):
        real = os.path.realpath(path)
        return {opened.fd for opened in psutil.Process().open_files() if opened.path == real}

    def blocked_read(reader, index):
        if not first_reader:
            first_reader.append(id(reader))
        held = id(reader) == first_reader[0]
        payload = underlying_read(reader, index)
        # A failed read's traceback must not keep the external reader alive after Dew closes it.
        del reader
        if held and index < 2:
            entered.wait(timeout=5)
            assert released[index].wait(timeout=5)
            if fail:
                raise failed_read
        return payload

    monkeypatch.setattr(grain.ArrayRecordDataSource, "__getitem__", blocked_read)
    try:
        assert source[2] == b"0:2"
        held = descriptors(arrayrecord_paths[0])
        assert held
        with ThreadPoolExecutor(2) as pool:
            pending = [pool.submit(source.__getitem__, index) for index in range(2)]
            try:
                entered.wait(timeout=5)
                if retire == "eviction":
                    for shard in range(1, len(arrayrecord_paths)):
                        assert source[8 * shard] == f"{shard}:0".encode()
                else:
                    source.close()
                assert descriptors(arrayrecord_paths[0]) == held
                # Close retires the busy reader; eviction keeps it and retires an idle one.
                assert source[2] == b"0:2"
                path = arrayrecord_paths[-1] if retire == "eviction" else arrayrecord_paths[0]
                replacement = descriptors(path) - held
                assert replacement
                for index, future in enumerate(pending):
                    released[index].set()
                    if fail:
                        with pytest.raises(ValueError) as raised:
                            future.result(timeout=5)
                        assert raised.value is failed_read
                    else:
                        assert future.result(timeout=5) == f"0:{index}".encode()
                    still_open = descriptors(arrayrecord_paths[0])
                    assert held & still_open == (set() if retire == "close" and index == 1 else held)
                    assert replacement <= descriptors(path)
            finally:
                for release in released:
                    release.set()
    finally:
        source.close()
    assert all(not descriptors(path) for path in arrayrecord_paths)


def test_pickled_arrayrecords_open_independent_readers(arrayrecord_paths):
    from dew.data.sources.records import ArrayRecords

    source = ArrayRecords(arrayrecord_paths, counts=(8,) * len(arrayrecord_paths))
    assert source[0] == b"0:0"
    restored = pickle.loads(pickle.dumps(source))
    try:
        source.close()
        assert restored[1] == b"0:1"
        restored.close()
        assert source[2] == b"0:2"
    finally:
        source.close()
        restored.close()


@pytest.mark.parametrize("max_bytes", [1, 1 << 30])
def test_arrayrecords_refuse_a_mirror_without_room_for_the_read_width(arrayrecord_paths, tmp_path, max_bytes):
    """A local mirror with room for the shards a read holds open serves them;
    one of a byte is refused."""
    from dew.data.sources.records import ArrayRecords

    shard = Store(f"memory://small-record-mirror-{uuid4().hex}/0.array_record")
    shard.write(Path(arrayrecord_paths[0]).read_bytes())
    source = ArrayRecords(
        (shard.url,),
        {"dewcache": {"cache_storage": str(tmp_path / "mirror"), "max_bytes": max_bytes}},
        counts=(8,),
    )
    try:
        if max_bytes > 1:
            assert source[0] == b"0:0"
            return
        with pytest.raises(ValueError):
            source[0]
    finally:
        source.close()


@pytest.mark.parametrize("busy_oldest,retire", [(False, "eviction"), (True, "eviction"), (True, "close")])
def test_arrayrecords_release_an_idle_mirror_before_opening_its_replacement(
    arrayrecord_paths, tmp_path, monkeypatch, busy_oldest, retire
):
    import grain.python as grain

    from dew.data.sources.records import ArrayRecords

    root = Store(f"memory://bounded-record-mirror-{uuid4().hex}")
    shards = [root.child(f"{index}.array_record") for index in range(len(arrayrecord_paths))]
    for shard, path in zip(shards, arrayrecord_paths, strict=True):
        shard.write(Path(path).read_bytes())
    capacity = sum(Path(path).stat().st_size for path in arrayrecord_paths[:-1])
    source = ArrayRecords(
        [shard.url for shard in shards],
        {"dewcache": {"cache_storage": str(tmp_path / "mirror"), "max_bytes": capacity}},
        counts=(8,) * len(shards),
    )
    entered, released = threading.Event(), threading.Event()
    underlying_read = grain.ArrayRecordDataSource.__getitem__

    def held(reader, index):
        if busy_oldest and not entered.is_set():
            entered.set()
            assert released.wait(timeout=5)
        return underlying_read(reader, index)

    monkeypatch.setattr(grain.ArrayRecordDataSource, "__getitem__", held)
    try:
        if busy_oldest:
            with ThreadPoolExecutor(1) as pool:
                pending = pool.submit(source.__getitem__, 0)
                try:
                    assert entered.wait(timeout=5)
                    if retire == "close":
                        source.close()
                    for shard in range(1, len(shards)):
                        assert source[8 * shard] == f"{shard}:0".encode()
                finally:
                    released.set()
                assert pending.result(timeout=5) == b"0:0"
        for _ in range(2):
            for shard in range(len(shards)):
                assert source[8 * shard] == f"{shard}:0".encode()
    finally:
        source.close()


def test_arrayrecords_wait_for_a_busy_reader_before_opening_a_missing_shard(arrayrecord_paths, monkeypatch):
    import grain.python as grain

    from dew.data.sources.records import ArrayRecords

    source = ArrayRecords(arrayrecord_paths, counts=(8,) * len(arrayrecord_paths))
    width = Loading().threads
    entered = threading.Barrier(width + 1)
    released, queued = threading.Event(), threading.Event()
    lock = threading.Lock()
    admitted = 0
    underlying_read = grain.ArrayRecordDataSource.__getitem__
    condition_wait = threading.Condition.wait

    def held(reader, index):
        nonlocal admitted
        with lock:
            ordinal = admitted
            admitted += 1
        if ordinal < width:
            entered.wait(timeout=5)
            assert released.wait(timeout=5)
        return underlying_read(reader, index)

    def waiting(condition, timeout=None):
        if condition is source._lock:
            queued.set()
        return condition_wait(condition, timeout)

    monkeypatch.setattr(grain.ArrayRecordDataSource, "__getitem__", held)
    monkeypatch.setattr(threading.Condition, "wait", waiting)
    try:
        with ThreadPoolExecutor(width + 1) as pool:
            pending = [pool.submit(source.__getitem__, 8 * shard) for shard in range(width)]
            try:
                entered.wait(timeout=5)
                overflow = pool.submit(source.__getitem__, 8 * width)
                assert queued.wait(timeout=5)
                assert not overflow.done()
            finally:
                released.set()
            assert [future.result(timeout=5) for future in pending] == [
                f"{shard}:0".encode() for shard in range(width)
            ]
            assert overflow.result(timeout=5) == f"{width}:0".encode()
    finally:
        source.close()


def test_jsonl_offset_index_is_mapped_and_reopened_instead_of_pickled_as_rows():
    import pickle

    root = Store(f"memory://mapped-index-{uuid4().hex}/rows.jsonl")
    root.write(b'{"id":0}\n{"id":1}\n')
    source = Files(root.url).provision()
    indexed = source.records("indexed")
    assert isinstance(indexed.tables[0], np.memmap)
    resumed = pickle.loads(pickle.dumps(indexed))
    assert resumed[1] == {"id": 1}
    assert isinstance(resumed.tables[0], np.memmap)


@pytest.mark.parametrize("format", ["jsonl", "text", "csv"])
def test_copied_offset_indexes_define_one_order_at_different_mounts(tmp_path, format):
    """Indexed copies agree on their compiled order and read the same records."""
    from dew.data.compile import compile

    first = tmp_path / "host0" / "corpus"
    first.mkdir(parents=True)
    suffix = "txt" if format == "text" else format
    for shard in range(2):
        rows = [f"row {index}" for index in range(2 * shard, 2 * shard + 2)]
        payload = ("text\n" if format == "csv" else "") + "\n".join(
            json.dumps({"text": row}) if format == "jsonl" else row for row in rows
        ) + "\n"
        (first / f"part-{shard}.{suffix}").write_text(payload)
    Files(str(first), format=format).provision()
    second = tmp_path / "elsewhere" / "mounted"
    shutil.copytree(first, second)

    def read(directory):
        reader = compile(Files(str(directory), format=format).repeat().batch(2),
                         read="indexed", loading=Loading(workers=0)).train(DataPartition())
        try:
            return reader.identity(), [text for batch in itertools.islice(reader, 2)
                                       for text in batch["text"].tolist()]
        finally:
            reader.close()

    first_identity, first_rows = read(first)
    second_identity, second_rows = read(second)
    assert first_rows == second_rows == [f"row {index}" + ("\n" if format == "text" else "")
                                        for index in range(4)]
    assert first_identity == second_identity


@pytest.mark.parametrize("format", ["jsonl", "text", "csv"])
def test_a_copied_corpus_resumes_a_position_saved_at_another_mount(tmp_path, format):
    """A position saved reading one copy of a corpus resumes on a copy mounted elsewhere: an
    order is the same wherever its sources are mounted, as the pool's identity already
    holds, and the copy reads on from the saved record."""
    from dew.data.compile import compile

    first = tmp_path / "host0" / "corpus"
    first.mkdir(parents=True)
    suffix = "txt" if format == "text" else format
    rows = [f"row {index}" for index in range(4)]
    (first / f"part-0.{suffix}").write_text(("text\n" if format == "csv" else "") + "\n".join(
        json.dumps({"text": row}) if format == "jsonl" else row for row in rows) + "\n")
    Files(str(first), format=format).provision()
    second = tmp_path / "elsewhere" / "mounted"
    shutil.copytree(first, second)

    def reader(directory):
        return compile(Files(str(directory), format=format).repeat().batch(1),
                       read="indexed", loading=Loading(workers=0)).train(DataPartition())

    saving, resumed = reader(first), reader(second)
    try:
        next(saving)
        saved = saving.get_state()
        expected = next(saving)["text"].tolist()
        resumed.set_state(saved)
        assert next(resumed)["text"].tolist() == expected
    finally:
        saving.close()
        resumed.close()


def test_a_pool_is_shown_which_part_of_its_plan_differs():
    """Processes whose plans differ are refused with each one's plan, part by
    part, a digest each: a step's identity or a source's content shows as the
    one entry that differs, not as one digest of everything."""
    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline

    def parts(rows, version):
        doubled = Pipeline(Memory(rows)).map(lambda row: {"x": row["x"] * 2}, identity=f"doubled@{version}")
        return compile(doubled.repeat().batch(2), read="memory", loading=Loading(workers=0)).plan.parts()

    rows = tuple({"x": np.int32(index)} for index in range(4))
    base, stepped = parts(rows, 1), parts(rows, 2)
    changed = parts((*rows[:3], {"x": np.int32(9)}), 1)

    def differing(other) -> set[str]:
        return {name for name in base if base[name] != other[name]}

    assert differing(stepped) == {"manifest steps"}
    assert [old == new for old, new in zip(base["manifest steps"], stepped["manifest steps"], strict=True)
            ].count(False) == 1
    assert "manifest source" in differing(changed) and "manifest steps" not in differing(changed)


class _Drawn(grain.RandomMapTransform):
    """A record as it is, drawn by the record generator, so a streamed repeat after it stays where it
    is written, restoring its reader at each pass's start."""

    def random_map(self, element, rng):
        return element


@pytest.mark.parametrize(("kind", "name"), [("jsonl", "rows.jsonl"), ("csv", "rows.csv"),
                                            ("text", "rows.txt"), ("jsonl", "rows.jsonl.zst")])
def test_a_streamed_line_file_repeated_reads_every_record_each_pass(tmp_path, kind, name):
    """A line file streamed and repeated reads its records again in each pass, as the first: Grain
    restores the reader at its start after a pass, which starts its records over."""
    import zstandard

    from dew.data.compile import compile

    lines = {"jsonl": [json.dumps({"id": index}) for index in range(3)],
             "csv": ["id", *(str(index) for index in range(3))],
             "text": [str(index) for index in range(3)]}[kind]
    payload = ("\n".join(lines) + "\n").encode()
    (tmp_path / name).write_bytes(zstandard.ZstdCompressor().compress(payload) if name.endswith(".zst")
                                  else payload)
    field = "text" if kind == "text" else "id"
    reader = compile(Files(str(tmp_path / name), format=kind).random_map(_Drawn()).repeat().batch(1),
                     read="streamed", loading=Loading(workers=0, threads=1, read_buffer=1)
                     ).train(DataPartition())
    try:
        read = [str(batch[field][0]).strip() for batch in itertools.islice(reader, 9)]
    finally:
        reader.close()
    assert read == ["0", "1", "2"] * 3


def test_a_text_file_is_provisioned_by_its_lines_bytes_and_a_line_is_decoded_when_read(tmp_path):
    """Provisioning a text file finds its lines without decoding them, so a line that is not UTF-8
    is refused when it is read, not when the file is indexed, and the lines before it read."""
    (tmp_path / "rows.txt").write_bytes(b"ok\n\xff\nlast\n")
    files = Files(str(tmp_path / "rows.txt"))
    files.provision()
    records = files.records("indexed")
    assert len(records) == 3
    assert (records[0], records[2]) == ({"text": "ok\n"}, {"text": "last\n"})
    with pytest.raises(UnicodeDecodeError):
        records[1]


def test_a_hugging_face_stream_saved_again_before_it_reads_resumes_where_it_was_restored():
    """A Hugging Face stream restored to a position and saved again before it reads saves that
    position, not its dataset's start: a resume from it reads on where the first read stopped."""
    import datasets

    from dew.data.compile import compile
    from dew.data.sources.hf import from_hf

    native = datasets.Dataset.from_dict({"id": list(range(8))}).to_iterable_dataset()
    data = compile(from_hf(native, "rows@1").batch(1), read="streamed",
                   loading=Loading(workers=0, threads=1, read_buffer=0))

    def ids(reader, count):
        return [int(batch["id"][0]) for batch in itertools.islice(reader, count)]

    first = data.train(DataPartition())
    try:
        assert ids(first, 3) == [0, 1, 2]
        saved = first.get_state()
        rest = ids(first, 3)
    finally:
        first.close()
    restored = data.train(DataPartition())
    try:
        restored.set_state(saved)
        again = restored.get_state()
    finally:
        restored.close()
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(again)
        assert ids(resumed, 3) == rest == [3, 4, 5]
    finally:
        resumed.close()


def test_a_same_size_middle_write_changes_a_multifile_readers_identity(tmp_path):
    """An exact multifile reader identifies bytes between every shard's endpoints."""
    from dew.data.compile import compile

    for shard in range(33):
        (tmp_path / f"part-{shard:02d}.jsonl").write_text(
            json.dumps({"id": shard, "text": "a" * (192 * 1024)}) + "\n")

    def reader():
        return compile(Files(str(tmp_path), format="jsonl").repeat().batch(1),
                       read="streamed", loading=Loading(workers=0, threads=1, read_buffer=1)
                       ).train(DataPartition())

    def read(batch):
        assert 0 <= int(batch["id"][0]) < 33
        assert batch["text"].tolist() == ["a" * (192 * 1024)]

    assert_a_rewrite_moves_the_identity(reader, tmp_path / "part-16.jsonl", b"b", read)


@pytest.mark.parametrize("dtype", ["<u4", "<u8"])
def test_a_producer_index_in_either_unsigned_type_places_its_records_in_their_files(tmp_path, dtype):
    """A producer's offset index over several JSON lines files, its shard column in whichever unsigned
    integer type it wrote, reads each record by index and block-shuffles them by file: an empty file
    and files of uneven sizes among them, every record read once a pass."""
    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline

    files = [[0, 1, 2], [], [3, 4, 5, 6, 7], [8]]
    entries = []
    for shard, ids in enumerate(files):
        payload = b""
        for value in ids:
            line = (json.dumps({"id": value}) + "\n").encode()
            entries.append((shard, len(payload), len(line)))
            payload += line
        (tmp_path / f"part-{shard}.jsonl").write_bytes(payload)
    encoded = io.BytesIO()
    np.save(encoded, np.array(entries, dtype=[("shard", dtype), ("offset", "<u8"), ("bytes", "<u8")]),
            allow_pickle=False)
    (tmp_path / "rows.index.npy").write_bytes(encoded.getvalue())
    paths = tuple(str(tmp_path / f"part-{shard}.jsonl") for shard in range(len(files)))
    stored = Files(paths, format="jsonl", index=str(tmp_path / "rows.index.npy"))
    records = stored.records("indexed")
    assert [records[index]["id"] for index in range(len(records))] == list(range(9))
    reader = compile(Pipeline(stored).shuffle(3, block_shards=2).repeat().batch(1), read="indexed",
                     loading=Loading(workers=0, threads=1, read_buffer=0)).train(DataPartition())
    try:
        read = [int(batch["id"][0]) for batch in itertools.islice(reader, 9)]
    finally:
        reader.close()
    assert sorted(read) == list(range(9))


def test_published_file_facts_avoid_metadata_probes_and_verify_the_index(monkeypatch):
    root = Store(f"memory://published-facts-{uuid4().hex}/rows.jsonl")
    root.write(b'{"id":0}\n{"id":1}\n')
    Files(root.url).provision()
    index = root.url.removesuffix(".jsonl") + ".index.npy"
    facts = {
        url: {"bytes": len(Store(url).read()), "sha256": hashlib.sha256(Store(url).read()).hexdigest()}
        for url in (root.url, index)
    }
    source = Files((root.url,), format="jsonl", index=index, file_info=facts)
    fs, _ = root.resolve()

    def unexpected(*args, **kwargs):
        raise RuntimeError("published source performed an unexpected info/HEAD request")

    monkeypatch.setattr(fs, "info", unexpected)
    assert source.supports("indexed")
    assert source.metadata()["rows"] == 2
    assert source.fingerprint()["files"][0]["version"] == facts[root.url]["sha256"]
    assert source.records("indexed")[1] == {"id": 1}


def test_torch_source_reads_every_way_and_refuses_a_cache_without_content_identity():
    import torch
    from torch.utils.data import TensorDataset

    from dew.data.cache import Cache
    from dew.data.plan import Manifest
    from dew.data.sources import from_torch

    rows = from_torch(TensorDataset(torch.arange(4)), ("id",))
    assert rows.supports("streamed")
    assert int(rows.records("indexed")[2]["id"]) == 2
    import grain.python as grain

    with pytest.raises(ValueError):
        Cache(f"memory://torch-cache-{uuid4().hex}").mapped(
            grain.MapDataset.source(rows.records("indexed")), Manifest(rows.fingerprint())
        )


@pytest.mark.parametrize("kind,read", [("torch", "memory"), ("torch", "indexed"),
                                       ("grain", "indexed"), ("grain", "streamed"), ("iterator", "streamed")])
def test_an_unnamed_foreign_source_trains_and_resumes_only_where_told_its_content_is_unknown(
        tmp_path, kind, read):
    """A torch, Grain or iterator source given no name reads and trains; nothing says a later run
    reads the same records, so an exact resume of its position is refused, one with
    `resume="non-exact"` reads on from it, and no cache keys it."""
    import grain.python as grain
    import torch
    from torch.utils.data import TensorDataset

    from dew.data import from_grain, from_iter, from_torch
    from dew.data.compile import compile

    rows = [{"id": index} for index in range(32)]

    def source():
        if kind == "torch":
            return from_torch(TensorDataset(torch.arange(32)), fields=("id",))
        if kind == "grain":
            return from_grain(grain.MapDataset.source(rows))
        return from_iter(lambda: iter(rows), replay=True)

    def reader(resume):
        return compile(source().repeat().batch(4), read=read, resume=resume,
                       loading=Loading(workers=0)).train(DataPartition())

    original = reader("exact")
    try:
        assert [next(original)["id"].tolist() for _ in range(2)] == [[0, 1, 2, 3], [4, 5, 6, 7]]
        saved = original.get_state()
        remaining = [next(original)["id"].tolist() for _ in range(2)]
    finally:
        original.close()
    exact, loose = reader("exact"), reader("non-exact")
    try:
        with pytest.raises(ValueError):
            exact.set_state(saved)
        loose.set_state(saved)
        assert [next(loose)["id"].tolist() for _ in range(2)] == remaining
    finally:
        exact.close()
        loose.close()
    with pytest.raises(ValueError):
        compile(source().cache(str(tmp_path / "cache")).batch(4), read=read, loading=Loading(workers=0))


@pytest.mark.parametrize("kind,read", [("torch", "memory"), ("torch", "indexed"),
                                       ("grain", "indexed"), ("iterator", "streamed")])
@pytest.mark.parametrize("resume", ["exact", "non-exact"])
def test_foreign_sources_resume_only_their_declared_content_identity(kind, read, resume):
    """A source version resumes its remaining records; another version cannot take its position."""
    import grain.python as grain

    from dew.data import from_grain, from_iter, from_torch
    from dew.data.compile import compile

    def dataset(identity, offset=0):
        rows = [{"id": index + offset} for index in range(32)]
        if kind == "torch":
            import torch
            from torch.utils.data import TensorDataset

            source = from_torch(TensorDataset(torch.arange(32) + offset), fields=("id",), identity=identity)
        elif kind == "grain":
            source = from_grain(grain.MapDataset.source(rows), identity)
        else:
            source = from_iter(lambda: iter(rows), name=identity, replay=True)
        return compile(source.repeat().batch(4), read=read, resume=resume, loading=Loading(workers=0))

    original = dataset("tickets@1").train(DataPartition())
    try:
        next(original)
        next(original)
        saved = original.get_state()
        remaining = [next(original)["id"].tolist() for _ in range(3)]
        identity = original.identity()
    finally:
        original.close()
    resumed = dataset("tickets@1").train(DataPartition())
    changed = dataset("tickets@2", 100).train(DataPartition())
    try:
        resumed.set_state(saved)
        assert [next(resumed)["id"].tolist() for _ in range(3)] == remaining
        assert resumed.identity() == identity != changed.identity()
        with pytest.raises(ValueError):
            changed.set_state(saved)
    finally:
        resumed.close()
        changed.close()


@pytest.mark.parametrize("native_state", [False, True])
def test_from_iter_closes_the_native_file_before_restoring_and_on_close(tmp_path, native_state):
    from dew.data import from_iter

    path = tmp_path / "native.jsonl"
    path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(8)))
    events = []
    opened = []

    class Native:
        def __init__(self):
            self.file = path.open("rb")
            self.index = len(opened)
            opened.append(self)
            events.append(("open", self.index))

        def __iter__(self):
            return self

        def __next__(self):
            line = self.file.readline()
            if not line:
                raise StopIteration
            return json.loads(line)

        def close(self):
            assert not self.file.closed
            self.file.close()
            events.append(("close", self.index))

    # Without a native position, the stream resumes by replay, which it declares.
    options = ({"state": lambda it: it.file.tell(), "load_state": lambda it, saved: it.file.seek(saved)}
               if native_state else {"replay": True})
    source = from_iter(Native, name="native-file@1", **options)
    manifest = source.manifest()
    reader = iter(manifest.reader(manifest.shards[0]))
    try:
        assert next(reader) == {"id": 0}
        saved = reader.get_state()
        assert next(reader) == {"id": 1}
        reader.set_state(saved)
        assert events == [("open", 0), ("close", 0), ("open", 1)]
        assert next(reader) == {"id": 1}
    finally:
        reader.close()
        reader.close()
    assert events == [("open", 0), ("close", 0), ("open", 1), ("close", 1)]
    assert all(iterator.file.closed for iterator in opened)


def test_closing_a_compiled_from_iter_reader_finalizes_its_native_generators():
    from dew.data import from_iter
    from dew.data.compile import compile

    opened = []
    closed = []
    owned = []
    finalizers = []
    both_started = threading.Event()

    def native(identity, finalized):
        opened.append(identity)
        if len(opened) == 2:
            both_started.set()
        try:
            yield from ({"id": index} for index in range(64))
        finally:
            closed.append(identity)
            finalized.set()

    def factory():
        finalized = threading.Event()
        iterator = native(len(owned), finalized)
        owned.append(iterator)
        finalizers.append(finalized)
        return iterator

    source = from_iter(factory, name="closing-generator@1").repeat().batch(2)
    reader = compile(source, read="streamed", loading=Loading(workers=0, threads=1, read_buffer=1)
                     ).train(DataPartition())
    try:
        assert next(reader)["id"].tolist() == [0, 1]
        assert both_started.wait(timeout=5)
        assert sorted(opened) == [0, 1] and not closed
    finally:
        reader.close()
        reader.close()
    # Grain finalizes the unconsumed lookahead on its iterator-making executor.
    assert all(finalized.wait(timeout=5) for finalized in finalizers)
    assert sorted(closed) == sorted(opened) == [0, 1]


def test_from_iter_forwards_a_native_stop_request():
    from dew.data import from_iter
    from dew.data.dataset import Stoppable

    stopped = []

    class Native:
        def __iter__(self):
            return self

        def __next__(self):
            return {"id": 0}

        def request_stop(self):
            stopped.append(True)

    source = from_iter(Native, name="stoppable@1")
    manifest = source.manifest()
    reader = iter(manifest.reader(manifest.shards[0]))
    try:
        assert next(reader) == {"id": 0}
        assert isinstance(reader, Stoppable)
        reader.request_stop()
        assert stopped == [True]
    finally:
        reader.close()


def test_two_unnamed_grain_stores_keep_their_own_rows_and_counts_in_phases_and_a_mix():
    """Two Grain stores given no name, read one in each phase, each from its own first row; mixed,
    each counted apart. A store read again in a later phase goes on where it stopped."""
    import grain.python as grain

    from dew.data import from_grain
    from dew.data.compile import compile
    from dew.data.pipeline import Phases, Pipeline, mix

    loading = Loading(workers=0, threads=1, read_buffer=1)
    first = from_grain(grain.MapDataset.source([{"id": index} for index in range(100)]))
    second = from_grain(grain.MapDataset.source([{"id": 1000 + index} for index in range(100)]))
    phased = Pipeline(Phases((Pipeline(first).repeat(), Pipeline(second).repeat()), (4, None))).batch(1)
    stream = compile(phased, read="indexed", loading=loading).train(DataPartition())
    try:
        assert [int(next(stream)["id"][0]) for _ in range(6)] == [0, 1, 2, 3, 1000, 1001]
    finally:
        stream.close()
    again = Pipeline(Phases((Pipeline(first).repeat(), Pipeline(first).repeat()), (4, None))).batch(1)
    stream = compile(again, read="indexed", loading=loading).train(DataPartition())
    try:
        assert [int(next(stream)["id"][0]) for _ in range(6)] == [0, 1, 2, 3, 4, 5]
    finally:
        stream.close()
    counts = mix([Pipeline(first), Pipeline(second)]).stats(records=10)
    assert sorted(value for name, value in counts.items() if name.startswith("data/rows/")) == [5, 5]


@pytest.mark.parametrize("table", ["pandas", "arrow"])
def test_memory_reads_a_table_of_lists_bytes_and_nulls_as_records_hold_them(table):
    """A table whose columns hold token lists, encoded bytes and a missing value reads as the same
    records given as records do."""
    import pandas as pd
    import pyarrow as pa

    columns = {"tokens": [[1, 2], [3, 4, 5]], "image": [b"ab", b"cde"], "caption": ["a cat", None]}
    given = pd.DataFrame(columns) if table == "pandas" else pa.table(columns)
    records = [{"tokens": [1, 2], "image": b"ab", "caption": "a cat"},
               {"tokens": [3, 4, 5], "image": b"cde", "caption": None}]
    held = Memory(given).records("indexed")
    assert [held[index] for index in range(len(held))] == records
    assert Memory(given).fingerprint()["sha256"] == Memory(records).fingerprint()["sha256"]


@pytest.mark.parametrize("kind", ["grain", "iterator"])
def test_foreign_sources_cannot_use_an_empty_content_identity(kind):
    """A Grain or iterator source given an empty name declares no content version, which is no
    name at all said otherwise: it is refused as it is made."""
    import grain.python as grain

    from dew.data import from_grain, from_iter

    rows = [{"id": index} for index in range(8)]
    with pytest.raises(ValueError):
        if kind == "grain":
            from_grain(grain.MapDataset.source(rows), "")
        else:
            from_iter(lambda: iter(rows), name="")


def test_delta_snapshot_pins_version_and_discovers_partition_columns(tmp_path):
    from deltalake import write_deltalake

    from dew.data.sources.delta import Delta

    root = str(tmp_path / "delta")
    write_deltalake(root, pa.table({"value": [1, 2], "split": ["train", "held"]}), partition_by=["split"])
    first = Delta(root)
    snapshot = first.fingerprint()
    write_deltalake(root, pa.table({"value": [3], "split": ["train"]}), mode="append")
    rows = Delta(root, version=snapshot["version"]).records("indexed")
    assert sorted(rows[index]["value"] for index in range(len(rows))) == [1, 2]
    assert {rows[index]["split"] for index in range(len(rows))} == {"train", "held"}


def test_a_delta_run_resumes_with_its_version_pinned_and_refuses_another_predicate(tmp_path):
    """A table's identity is its resolved version, files and read options: a run started on the
    table's head resumes exactly with that version pinned, and a run reading it through another
    predicate is refused."""
    import pyarrow.dataset as ds
    from deltalake import write_deltalake

    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline
    from dew.data.sources.delta import Delta

    root = str(tmp_path / "delta")
    write_deltalake(root, pa.table({"value": list(range(8))}))
    loading = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)

    def opened(table):
        data = compile(Pipeline(table).repeat().batch(2), read="streamed", loading=loading)
        return data.train(DataPartition())

    head = opened(Delta(root))
    try:
        first = [np.asarray(next(head)["value"]).tolist()]
        saved = head.get_state()
        expected = [np.asarray(next(head)["value"]).tolist() for _ in range(2)]
    finally:
        head.close()
    pinned = opened(Delta(root, version=Delta(root).table.version()))
    try:
        pinned.set_state(saved)
        assert first + [np.asarray(next(pinned)["value"]).tolist() for _ in range(2)] == first + expected
    finally:
        pinned.close()
    filtered = opened(Delta(root, where=ds.field("value") >= 0))
    try:
        with pytest.raises(ValueError):
            filtered.set_state(saved)
    finally:
        filtered.close()


def test_artifact_codec_preserves_arrow_scalars_and_nested_arrays():
    import datetime
    from decimal import Decimal

    from dew.data.sources.codec import decode, encode

    row = {
        "when": datetime.datetime(2026, 1, 2, 3, 4, tzinfo=datetime.UTC),
        "price": Decimal("12.340"),
        "interval": datetime.timedelta(days=2, seconds=3),
        "features": [np.arange(4, dtype=np.uint16), {"nan": float("nan")}],
        "binary": b"\0\xff",
        "messages": np.asarray([[{"role": "user", "content": "hello"}]], dtype=object),
    }
    actual = decode(encode(row))
    assert actual["when"] == row["when"]
    assert actual["price"].as_tuple() == row["price"].as_tuple()
    assert actual["interval"] == row["interval"]
    assert np.array_equal(actual["features"][0], row["features"][0])
    assert np.isnan(actual["features"][1]["nan"])
    assert actual["binary"] == row["binary"]
    assert actual["messages"].dtype == row["messages"].dtype
    assert actual["messages"].tolist() == row["messages"].tolist()


def test_artifact_codec_preserves_extended_numpy_dtypes():
    import ml_dtypes

    from dew.data.sources.codec import decode, encode

    values = {"array": np.asarray([1.0, 2.5], dtype=ml_dtypes.bfloat16), "scalar": ml_dtypes.bfloat16(3.0)}
    restored = decode(encode(values))
    assert restored["array"].dtype == values["array"].dtype
    assert restored["scalar"].dtype == values["scalar"].dtype
    assert np.array_equal(restored["array"], values["array"])
    assert restored["scalar"] == values["scalar"]


def test_content_fingerprint_preserves_array_identity_and_sequence_order():
    from dew.data.sources.codec import content_fingerprint, decode, encode

    ordered = {"a": [b"first", b"last"], "z": np.arange(6, dtype=np.int32).reshape(2, 3)}
    reordered = {"z": ordered["z"], "a": ordered["a"]}
    digest = content_fingerprint(ordered)
    assert content_fingerprint(reordered) == digest == hashlib.sha256(encode(ordered)).digest()
    assert encode(reordered) != encode(ordered)
    assert list(decode(encode(reordered))) == ["z", "a"]
    variants = [
        {**ordered, "z": ordered["z"].astype(np.int64)},
        {**ordered, "z": ordered["z"].reshape(3, 2)},
        {**ordered, "z": ordered["z"] + 1},
        {**ordered, "a": list(reversed(ordered["a"]))},
        {**ordered, "a": tuple(ordered["a"])},
    ]
    assert all(content_fingerprint(row) != digest for row in variants)
    assert content_fingerprint(np.int16(4)) != content_fingerprint(np.int32(4))


def test_a_pass_reads_its_rows_after_any_run_of_empty_shards():
    """A pass over a stream reads the rows of its last shard however many empty shards come before
    it: four thousand and more in a row are no proof the stream holds none."""
    root = Store(f"memory://empty-run-{uuid4().hex}")
    for number in range(4097):
        # A blank line is a shard that holds no row.
        root.child(f"{number:05d}.jsonl").write(b'{"id": 7}\n' if number == 4096 else b"\n")
    dataset = streamed(Files(root.url, format="jsonl").manifest(), partition=DataPartition(), epochs=1,
                       cycle_length=1, loading=Loading(workers=0, threads=1, read_buffer=1))
    assert list(dataset) == [{"id": 7}]


def test_an_endless_read_of_a_store_that_states_it_holds_no_row_is_refused():
    """A store whose every shard states it holds no row would read nothing forever endlessly, which
    is refused; read once, it is a pass that yields nothing."""
    manifest = Memory([], shard_rows=4).manifest()
    loading = Loading(workers=0, threads=1, read_buffer=1)
    assert list(streamed(manifest, partition=DataPartition(), epochs=1, loading=loading)) == []
    with pytest.raises(ValueError):
        streamed(manifest, partition=DataPartition(), epochs=None, loading=loading)


@dataclasses.dataclass(frozen=True)
class _StreamOnly:
    """Memory documents that state their lengths but are read only as a
    stream, as a store filtered by a predicate is."""

    memory: Memory

    def describe(self):
        return f"stream-only {self.memory.describe()}"

    def label(self):
        return self.memory.label()

    def fingerprint(self):
        return self.memory.fingerprint()

    def supports(self, read):
        return read == "streamed"

    def records(self, read):
        return self.memory.records(read)

    def manifest(self):
        return self.memory.manifest()

    def metadata(self):
        return self.memory.metadata()

    def lengths(self):
        return np.full(len(self.memory.records("streamed")), 3)


def test_a_source_read_only_as_a_stream_packs_its_rows_as_they_arrive():
    """A source that states its documents' lengths but cannot be read by
    index plans no packing: its streamed rows pack as they arrive."""
    from dew.data.compile import iterate
    from dew.data.pipeline import Pipeline

    documents = Memory(tuple({"text": np.full(3, index + 1, np.int32)} for index in range(4)))
    packed = Pipeline(_StreamOnly(documents)).pack(6)
    windows = list(iterate(packed, read="streamed", loading=Loading(workers=0)))
    assert sorted({int(token) for window in windows for token in window["text"] if token}) == [1, 2, 3, 4]


def test_a_chain_that_states_no_read_reads_as_its_sources_choose(tmp_path):
    """Records held in memory read from memory, a store that reads by index
    reads by index and a stream streams, when the chain states no read; a read
    it states wins over each."""
    import pyarrow.parquet as pq

    rows = [{"value": index} for index in range(8)]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "rows.parquet")
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def read(source, stated=None):
        loaded = compile(source.repeat().batch(2), read=stated, loading=Loading(workers=0))
        assert loaded.plan is not None
        return loaded.plan.order["read"]

    listed, indexed = Memory(rows), Files(str(tmp_path / "rows.parquet"))
    streamed = Files(str(tmp_path / "rows.jsonl"))
    assert [read(listed), read(indexed), read(streamed)] == ["memory", "indexed", "streamed"]
    assert [read(listed, "streamed"), read(indexed, "memory"), read(streamed, "memory")] == [
        "streamed", "memory", "memory"]


def test_a_token_holdout_with_no_stated_read_reads_unindexed_text_into_memory(tmp_path):
    """A token split needs a finite indexed order, which JSON lines without an
    index can neither stream nor read by index, so the chain reads them into
    memory and holds out the first quarter of their tokens."""
    from dew.data import presets, text
    from dew.data.compile import held

    documents = [{"text": f"document {index} " * 6} for index in range(8)]
    (tmp_path / "text.jsonl").write_text("".join(json.dumps(document) + "\n" for document in documents))
    train, val = Files(str(tmp_path / "text.jsonl")).map(text.tokenize("byte")).holdout(tokens=0.25)
    windows = train.pipe(presets.lm, None, seq_len=8, batch=None, pack=None)
    loaded = compile(windows.shuffle(0).repeat().batch(2), loading=Loading(workers=0))
    assert loaded.plan is not None and loaded.plan.order["read"] == "memory"
    assert loaded.records
    assert next(iter(held(val.pipe(presets.lm, None, seq_len=8, batch=2, pack=None))(DataPartition())))


def test_a_local_store_whose_producer_declared_other_bytes_is_refused(tmp_path):
    """A producer's manifest states each file's bytes and checksum; a copy on this host's disk
    that holds other bytes (a cut of the files under the whole set's manifest) is refused, naming
    both, and one that holds what it declares is read."""
    import hashlib

    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(8)))
    url = str(path)
    held = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert [row["id"] for row in Files(url, file_info={url: held})] == list(range(8))
    for stale in ({**held, "bytes": 2 * held["bytes"]}, {**held, "sha256": "0" * 64}):
        with pytest.raises(ValueError):
            Files(url, file_info={url: stale}).fingerprint()


def test_an_exports_files_of_one_place_shard_apart(tmp_path):
    """Two files of a provisioned (committed) ArrayRecord directory whose writers' directories are left
    out of their places, so they share one, are still two shards: a read of the files, and of a
    concatenation of them, holds every record of both."""
    import array_record.python.array_record_module as array_record

    from dew.data.sources import Concatenate

    root = tmp_path / "export"
    for writer, values in (("a", (0, 1)), ("b", (2, 3))):
        (root / "shards" / writer).mkdir(parents=True)
        writing = array_record.ArrayRecordWriter(str(root / "shards" / writer / "part-000000.array_record"),
                                                 "group_size:1")
        for value in values:
            writing.write(str(value).encode())
        writing.close()
    files = Files(str(root), format="array_record").provision()
    assert len({shard.id for shard in files.manifest().shards}) == 2
    for source in (files, Concatenate((files,))):
        manifest = source.manifest()
        read = sorted(bytes(row) for shard in manifest.shards for row in manifest.reader(shard))
        assert read == [b"0", b"1", b"2", b"3"]


def _indexed_source(kind: str, root: Path):
    """A source of `kind` that streams its indexed records, and the shards (id, location, start,
    stop, rows) its manifest names: the ids a cache keys rows by and a saved position counts into,
    which name no directory a local store's fingerprint leaves out."""
    from token_support import write_token_corpus

    from dew.data.pipeline import Pack, Pipeline
    from dew.data.sources import TFDS, TokenFiles

    if kind == "memory":
        return Memory([{"id": index} for index in range(7)], shard_rows=3), [
            ("records:0", "memory", 0, 3, 3), ("records:3", "memory", 3, 6, 3),
            ("records:6", "memory", 6, 7, 1)]
    if kind == "tfds":
        import tensorflow_datasets as tfds

        path = str(Path(__file__).parent / "fixtures" / "tfds" / "dew_images" / "1.0.0")
        return TFDS("dew_images", path=path, split="all", shard_rows=8,
                    decoders={"image": tfds.decode.SkipDecoding()}), [
            ("records:0", path, 0, 8, 8), ("records:8", path, 8, 16, 8), ("records:16", path, 16, 20, 4)]
    if kind == "generation":
        from dew.data.cache import Cache
        from dew.data.compile import publish

        generation = publish(Pipeline(Memory([{"id": index} for index in range(4100)])),
                             Cache(str(root / "cache")), read="indexed",
                             loading=Loading(workers=0, threads=1, read_buffer=0))
        url = generation.url
        return generation.source(), [("records:0", url, 0, 4096, 4096), ("records:4096", url, 4096, 4100, 4)]
    if kind == "planned":
        from dew.data.packing import PackedSource

        documents = [np.arange(3, dtype=np.int32) + index for index in range(1030)]
        held = Memory({"text": documents, "length": [3] * 1030}, identity="documents")
        return PackedSource.planned(held, Pack(4, bins=1)), [
            ("records:0", "memory:documents", 0, 1024, 1024),
            ("records:1024", "memory:documents", 1024, 1030, 6)]
    corpus = write_token_corpus(root, np.tile(np.array([5, 255], np.uint16), 1030), eos_id=255)
    tokens = TokenFiles(corpus.train).provision()
    if kind == "tokens":
        return tokens, [("records:0", corpus.train, 0, 1024, 1024),
                        ("records:1024", corpus.train, 1024, 1030, 6)]
    from dew.data.steps.windows import WindowSource

    windows = WindowSource(tokens, tokens.tokens, 4, 1)
    described = f"{tokens.describe()} causal windows=4,stride=1"
    return windows, [("records:0", described, 0, 1024, 1024), ("records:1024", described, 1024, 2048, 1024),
                     ("records:2048", described, 2048, 2056, 8)]


@pytest.mark.parametrize("kind", ["memory", "tfds", "tokens", "windows", "generation", "planned"])
def test_an_indexed_source_streams_as_bounded_shards_of_its_records(tmp_path, kind):
    """A source read by index streams as consecutive shards of its records, each named by its first
    record, wherever the source is; a shard reads that slice of the indexed records, and the
    manifest pickles, reader and all, for Grain's worker processes."""
    source, expected = _indexed_source(kind, tmp_path)
    manifest = source.manifest()
    named = [(shard.id, shard.uri, shard.start, shard.stop, shard.rows) for shard in manifest.shards]
    assert named == expected
    reopened = pickle.loads(pickle.dumps(manifest))
    records = source.records("indexed")
    streamed_rows = [row for shard in reopened.shards for row in reopened.reader(shard)]
    assert len(streamed_rows) == len(records)
    for index, row in enumerate(streamed_rows):
        assert row.keys() == records[index].keys()
        for name, value in row.items():
            np.testing.assert_array_equal(value, records[index][name])


def test_native_map_datasets_read_as_streams_and_mix_with_a_stream_only_member():
    """A Grain `MapDataset` and a Hugging Face `Dataset` read as one stream each, as they read by
    index, so either mixes with a member that only streams (a Grain `IterDataset`), and the
    mixture resumes where it stopped."""
    import grain.python as grain
    from datasets import Dataset as Arrow

    from dew.data.compile import compile
    from dew.data.pipeline import mix
    from dew.data.sources.grain import from_grain
    from dew.data.sources.hf import from_hf

    indexed = from_grain(grain.MapDataset.source([{"id": index} for index in range(6)]), "indexed")
    table = from_hf(Arrow.from_dict({"id": list(range(10, 16))}), name="table")
    stream = grain.MapDataset.source([{"id": 20 + index} for index in range(6)]).to_iter_dataset()
    streamed = from_grain(stream, "streamed")
    loading = Loading(workers=0, threads=1, read_buffer=1)

    def ids(reader, count: int) -> list[int]:
        return [int(value) for batch in itertools.islice(reader, count) for value in batch["id"]]

    for native in (indexed, table):
        alone = compile(native.repeat(1).batch(2), read="streamed", loading=loading).train(DataPartition())
        try:
            assert sorted(int(value) for batch in alone for value in batch["id"]) == sorted(
                int(native.records("indexed")[index]["id"]) for index in range(6))
        finally:
            alone.close()
        data = compile(mix([native.repeat(), streamed.repeat()]).batch(2), read="streamed", loading=loading)
        assert_resumes_where_it_stopped(lambda data=data: data.train(DataPartition()), ids, 3, 3)


@dataclasses.dataclass(frozen=True)
class Stated(Memory):
    """Records with the facts their store states, as a producer's metadata states them."""

    facts: tuple[tuple[str, object], ...] = ()

    def metadata(self):
        return {**super().metadata(), **dict(self.facts)}


def test_a_concatenation_states_what_its_stores_state_of_their_records():
    """Two splits of one token store joined (`dew.data.Concatenate`) state its tokenizer contract and
    class names, the targets of both and the rows of both, so a token mixture weighs the joined
    store and a run checks its contract as of either split; a fact one split alone states or the
    two state apart is none of the join's, and two contracts make no one store."""
    from dew.data import Concatenate

    contract = (("tokenizer", "gpt2"), ("vocab_size", 50257), ("class_names", {"label": ("a", "b")}))
    first = Stated([{"id": np.int32(index)} for index in range(3)],
                   facts=(*contract, ("valid_targets", 30), ("split", "train")))
    second = Stated([{"id": np.int32(index)} for index in range(2)],
                    facts=(*contract, ("valid_targets", 20), ("split", "validation"), ("shuffled", True)))
    # Both are held in memory, whose rows carry no count of Dew's (`count_rules`).
    assert Concatenate((first, second)).metadata() == {
        **dict(contract), "rows": 5, "valid_targets": 50, "count_rules": []}
    other = Stated([{"id": np.int32(0)}], facts=(("tokenizer", "llama"),))
    with pytest.raises(ValueError):
        Concatenate((first, other)).metadata()


@pytest.mark.parametrize("read", ["indexed", "streamed", "memory"])
def test_concatenated_stores_train_as_one_store_and_resume_where_they_stopped(read):
    """Two stores read in turn (`dew.data.Concatenate`) are one store to the verbs: a pass over it
    holds every record of both once, in their order, steps chain on it as on any store, and a
    training run resumed after two batches reads the batches the uninterrupted run reads."""
    from dew.data import Concatenate
    from dew.data.compile import validation
    from dew.data.pipeline import Pipeline

    train = Memory([{"id": np.int32(index)} for index in range(6)])
    held = Memory([{"id": np.int32(index)} for index in range(100, 104)])
    joined = Concatenate((train, held))
    assert joined.map(lambda record: {"id": record["id"] + 1})._chain().source is joined
    loading = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)
    passed = validation(Pipeline(joined).batch(2), batch=2, read=read, loading=loading)(DataPartition())
    try:
        assert [int(value) for batch in passed for value in batch["id"]] == [*range(6), *range(100, 104)]
    finally:
        passed.close()

    trained = compile(Pipeline(joined).shuffle(3, window=4).repeat().batch(2), read=read, loading=loading,
                      seed=3)

    def opened():
        return trained.train(DataPartition())

    def ids(reader, count):
        return [np.asarray(next(reader)["id"]).tolist() for _ in range(count)]

    assert_resumes_where_it_stopped(opened, ids, 2, 6)


def test_a_directory_of_arrow_files_reads_its_feather_files_too(tmp_path):
    """Arrow IPC files carry `.arrow` or `.feather`, both of which a URL's format is inferred from:
    a directory read as Arrow reads both kinds, every row of each."""
    import pyarrow.feather as feather

    feather.write_feather(pa.table({"id": [0, 1]}), tmp_path / "first.feather")
    with pa.ipc.new_file(tmp_path / "second.arrow", pa.schema([("id", pa.int64())])) as writer:
        writer.write_table(pa.table({"id": [2, 3]}))
    source = Files(str(tmp_path), format="arrow")
    rows = source.records("memory")
    assert sorted(int(rows[index]["id"]) for index in range(len(rows))) == [0, 1, 2, 3]
    assert Files(str(tmp_path / "first.feather")).kind == "arrow"


def test_a_directory_named_for_its_format_declares_it_with_a_trailing_separator_too(tmp_path):
    """A directory whose name carries a format's suffix declares that format, spelled with a
    trailing separator or without: either spelling reads every file it holds."""
    import pyarrow.parquet as pq

    directory = tmp_path / "train.parquet"
    directory.mkdir()
    pq.write_table(pa.table({"id": [0, 1]}), directory / "part-0.parquet")
    for url in (str(directory), f"{directory}/"):
        rows = Files(url).records("memory")
        assert [int(rows[index]["id"]) for index in range(len(rows))] == [0, 1]


def test_a_concatenation_of_cold_stores_is_prepared_through_its_own_stores(tmp_path):
    """A concatenation of two JSONL stores that hold no offset index yet takes an indexed read once
    each of its stores prepares one: a pass asked to provision prepares both and reads every
    record, the first store's then the second's, where it once refused the read."""
    from dew.data.compile import validation
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Pipeline
    from dew.data.sources import Concatenate

    for name, ids in (("first", range(5)), ("second", range(5, 9))):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps({"id": i}) + "\n" for i in ids))
    joined = Concatenate((Files(str(tmp_path / "first.jsonl")), Files(str(tmp_path / "second.jsonl"))))
    assert not joined.supports("indexed")
    reader = validation(Pipeline(joined).batch(3), batch=3, read="indexed", provision=True,
                        loading=Loading(workers=0))(DataPartition())
    try:
        assert [int(value) for batch in reader for value in batch["id"]] == list(range(9))
    finally:
        reader.close()
    assert joined.supports("indexed")
