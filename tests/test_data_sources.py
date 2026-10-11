"""Read-mode capabilities, range splits, partition discovery and shard resume."""

import io
import json
import tarfile
from pathlib import Path
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
    assert Memory([{"x": np.int16(4)}]).snapshot() != Memory([{"x": np.int32(4)}]).snapshot()


def test_pinned_hub_revision_never_falls_through_to_a_different_conversion(monkeypatch):
    from types import SimpleNamespace

    from huggingface_hub import HfApi

    from dew.data.sources import Hub

    def metadata(self, repo, revision=None):
        return SimpleNamespace(sha="converted" if revision == "refs/convert/parquet" else "original")

    def listing(self, repo, *, repo_type, revision):
        return ["data/train-00000.parquet"] if revision == "converted" else ["train.tar"]

    monkeypatch.setattr(HfApi, "dataset_info", metadata)
    monkeypatch.setattr(HfApi, "list_repo_files", listing)
    with pytest.raises(FileNotFoundError, match=r"original.*converted branch.*provision=True"):
        Hub("owner/data", revision="original").files
    latest = Hub("owner/data").files
    assert latest.url == ("hf://datasets/owner/data@converted/data/train-00000.parquet",)


@pytest.mark.parametrize("format", ["jsonl", "arrow", "parquet"])
def test_hub_native_shards_keep_the_requested_revision(monkeypatch, format):
    from types import SimpleNamespace

    from huggingface_hub import HfApi

    from dew.data.sources import Hub

    monkeypatch.setattr(
        HfApi, "dataset_info", lambda self, repo, revision=None: SimpleNamespace(sha=revision)
    )
    monkeypatch.setattr(HfApi, "list_repo_files", lambda self, repo, **kwargs: [f"train.{format}"])
    native = Hub("owner/data", revision="pinned").files
    assert native.url == (f"hf://datasets/owner/data@pinned/train.{format}",)
    assert native.format == format


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
    assert decoded.snapshot() != raw.snapshot()
    assert isinstance(raw.records("indexed")[0]["image"], bytes)


def test_remote_tfds_requires_explicit_provisioning_before_native_reads():
    from dew.data.sources import TFDS

    fixture = Path(__file__).parent / "fixtures" / "tfds" / "dew_images" / "1.0.0"
    root = Store(f"memory://tfds-{uuid4().hex}/1.0.0")
    for path in fixture.iterdir():
        root.child(path.name).write(path.read_bytes())
    source = TFDS("dew_images", path=root.url)
    assert not source.supports("indexed")
    with pytest.raises(FileNotFoundError, match=r"provision=True.*gs://"):
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
    before = TFDS("dew_images", path=str(tmp_path)).snapshot()
    (tmp_path / "label.labels.txt").write_text("rouge\nbleu\n")
    after = TFDS("dew_images", path=str(tmp_path)).snapshot()
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


def test_csv_large_quoted_fields_survive_streaming_and_offset_provisioning():
    text = "a" * 140_000 + "\n" + "b" * 140_000
    root = Store(f"memory://csv-large-{uuid4().hex}/rows.csv")
    root.write(('id,text\n0,"' + text + '"\n').encode())
    files = Files(root.url)
    assert list(files.manifest().reader(files.manifest().shards[0])) == [{"id": "0", "text": text}]
    files.provision()
    assert files.records("indexed")[0] == {"id": "0", "text": text}


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
    files = Files(root.url, format="parquet", columns=("value", "split"), filter=ds.field("value") >= 15)
    assert not files.supports("indexed")
    actual = [row for shard in files.manifest().shards for row in files.manifest().reader(shard)]
    assert actual == [{"value": index, "split": "held"} for index in range(15, 20)]
    indexed = Files(root.url, format="parquet").records("indexed")
    assert len(indexed) == 20
    assert sorted(indexed[index]["value"] for index in range(20)) == list(range(20))


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


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_text_preserves_exact_line_terminators_through_byte_tokenization(read):
    from dew.data.steps.text import Tokenize

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


def test_exported_token_lengths_support_indexed_spans_without_record_scan():
    from dew.data.sources.files import TokenFiles

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


def test_torch_source_refuses_streaming_and_cache_without_content_identity():
    import torch
    from torch.utils.data import TensorDataset

    from dew.data.cache import Cache
    from dew.data.plan import Manifest
    from dew.data.sources import Torch

    rows = Torch(TensorDataset(torch.arange(4)), fields=("id",))
    assert not rows.supports("streamed")
    assert int(rows.records("indexed")[2]["id"]) == 2
    with pytest.raises(ValueError, match="cannot read streamed"):
        rows.records("streamed")
    import grain.python as grain

    with pytest.raises(ValueError, match="explicit content identity"):
        Cache(f"memory://torch-cache-{uuid4().hex}").mapped(
            grain.MapDataset.source(rows.records("indexed")), Manifest(rows.snapshot()), rows
        )


def test_delta_snapshot_pins_version_and_discovers_partition_columns(tmp_path):
    from deltalake import write_deltalake

    from dew.data.sources.delta import Delta

    root = str(tmp_path / "delta")
    write_deltalake(root, pa.table({"value": [1, 2], "split": ["train", "held"]}), partition_by=["split"])
    first = Delta(root)
    snapshot = first.snapshot()
    write_deltalake(root, pa.table({"value": [3], "split": ["train"]}), mode="append")
    rows = Delta(root, version=snapshot["version"]).records("indexed")
    assert sorted(rows[index]["value"] for index in range(len(rows))) == [1, 2]
    assert {rows[index]["split"] for index in range(len(rows))} == {"train", "held"}


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


def test_empty_shard_cycles_exhaust_a_named_search_budget():
    root = Store(f"memory://empty-shards-{uuid4().hex}/empty.jsonl")
    root.write(b" \n\n")
    files = Files(root.url)
    dataset = streamed(
        files.manifest(),
        partition=DataPartition(),
        epochs=None,
        empty_shard_budget=3,
        name="empty-corpus",
        cycle_length=1,
        loading=Loading(workers=0, threads=1, read_buffer=1),
    )
    reader = iter(dataset)
    with pytest.raises(ValueError, match=r"empty-corpus.*search budget"):
        next(reader)
    reader.close()


@pytest.mark.parametrize("field", [None, "ids"])
def test_raw_token_arrayrecords_require_explicit_lengths_for_indexed_reads(tmp_path, field):
    from array_record.python.array_record_module import ArrayRecordWriter

    from dew.data.sources import source as resolve
    from dew.data.sources.codec import pack_dict_of_byte_arrays
    from dew.data.sources.files import TokenFiles

    root = Store(str(tmp_path))
    root.child("meta.json").write(
        json.dumps(
            {
                "tokenizer": "byte",
                "dtype": "uint16",
                "vocab_size": 256,
                "train_tokens": 15,
                "val_tokens": 4,
                "eos_id": None,
            }
        ).encode()
    )
    for split, count in (("train", 15), ("val", 4)):
        writer = ArrayRecordWriter(str(tmp_path / f"{split}-0.array_record"), "group_size:1")
        for start, stop in zip((0, 2, 4), (2, 4, count), strict=True):
            raw = np.arange(start, stop, dtype=np.uint16).tobytes()
            writer.write(raw if field is None else pack_dict_of_byte_arrays({field: raw}))
        writer.close()
    source = TokenFiles(str(tmp_path), field=field)
    assert isinstance(resolve(str(tmp_path)), TokenFiles)
    assert source.metadata()["train_tokens"] == 15
    assert not source.supports("indexed")
    with pytest.raises(FileNotFoundError, match="length index"):
        source.records("indexed")
    source.records("memory")
    assert source.tokens[1:14].tolist() == list(range(1, 14))
    source.provision()
    reopened = TokenFiles(str(tmp_path), field=field)
    assert reopened.supports("indexed")
    assert reopened.tokens[2:15].tolist() == list(range(2, 15))
