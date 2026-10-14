"""Read-mode capabilities, range splits, partition discovery and shard resume."""

import dataclasses
import hashlib
import io
import itertools
import json
import pickle
import shutil
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
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
    assert first.fingerprint() != changed.fingerprint()
    assert first.supports("streamed")
    rows = [row for shard in first.manifest().shards for row in first.manifest().reader(shard)]
    assert [row["x"] for row in rows] == list(range(5))
    with pytest.raises(ValueError, match="different row counts"):
        len(Memory({"x": [1], "y": [1, 2]}).records("indexed"))
    assert Memory([{"x": np.int16(4)}]).fingerprint() != Memory([{"x": np.int32(4)}]).fingerprint()
    assert Memory([{"x": 1}], shard_rows=1).fingerprint() != Memory([{"x": 1}], shard_rows=2).fingerprint()


def test_pinned_hub_revision_never_falls_through_to_a_different_conversion(monkeypatch):
    from types import SimpleNamespace

    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import GitRefInfo, GitRefs

    from dew.data.sources import Hub

    def listing(self, repo, *, repo_type, revision):
        return ["data/train-00000.parquet"] if revision == "converted" else ["train.tar"]

    conversion = GitRefInfo(name="parquet", ref="refs/convert/parquet", target_commit="converted")
    monkeypatch.setattr(HfApi, "dataset_info", lambda self, repo, **kwargs: SimpleNamespace(sha="original"))
    monkeypatch.setattr(HfApi, "list_repo_files", listing)
    monkeypatch.setattr(HfApi, "list_repo_refs",
                        lambda self, repo, **kwargs: GitRefs(branches=[], converts=[conversion], tags=[]))
    with pytest.raises(FileNotFoundError, match=r"original.*converted branch.*provision=True"):
        _ = Hub("owner/data", revision="original").files
    latest = Hub("owner/data").files
    assert latest.url == ("hf://datasets/owner/data@converted/data/train-00000.parquet",)


@pytest.fixture
def moving_hub_conversion(monkeypatch, tmp_path, request):
    from types import SimpleNamespace

    import datasets
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem
    from huggingface_hub.hf_api import GitRefInfo, GitRefs

    converted = ["first"]
    kind = getattr(request, "param", "records")

    def columns(start):
        indices = range(start, start + 32)
        if kind == "prompts":
            return {"prompt": [[index + 1, index + 2] for index in indices],
                    "ground_truth": [str(index) for index in indices]}
        if kind == "decisions":
            return {"text": [f"row {index}" for index in indices],
                    "label": ["a" if index % 2 else "b" for index in indices]}
        return {"id": list(indices)}

    payloads = {}
    for commit, start in (("first", 0), ("second", 10)):
        sink = pa.BufferOutputStream()
        pq.write_table(pa.table(columns(start)), sink)
        payloads[f"datasets/owner/data@{commit}/default/train/0000.parquet"] = sink.getvalue().to_pybytes()

    def header(self, path, **kwargs):
        payload = payloads[path]
        return {"name": path, "type": "file", "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest()}

    def refs(self, repo, *, repo_type):
        conversion = GitRefInfo(name="parquet", ref="refs/convert/parquet", target_commit=converted[-1])
        return GitRefs(branches=[], converts=[conversion], tags=[])

    def listing(self, repo, *, repo_type, revision):
        return ["train.tar"] if revision == "original" else ["default/train/0000.parquet"]

    monkeypatch.setattr(HfApi, "dataset_info", lambda self, repo, **kwargs: SimpleNamespace(sha="original"))
    monkeypatch.setattr(HfApi, "list_repo_files", listing)
    monkeypatch.setattr(HfApi, "list_repo_refs", refs)
    monkeypatch.setattr(HfFileSystem, "info", header)
    monkeypatch.setattr(HfFileSystem, "_open", lambda self, path, **kwargs: io.BytesIO(payloads[path]))
    (tmp_path / "dataset_info.json").write_text("{}")
    prepared = datasets.Dataset.from_dict(columns(0))
    builder = SimpleNamespace(
        cache_dir=str(tmp_path),
        config=SimpleNamespace(data_files=None),
        info=SimpleNamespace(features=None, splits={"train": SimpleNamespace(num_examples=32)}),
        as_dataset=lambda **kwargs: prepared,
    )
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda *args, **kwargs: builder)
    return converted


def test_a_split_saved_as_arrow_streams_its_parquet_conversion(monkeypatch):
    """A dataset repo's `.arrow` files are what `save_to_disk` and `push_to_hub` write, Arrow's
    stream format, which no file reader opens; a split that holds only those streams its Parquet
    conversion, and a pinned revision without one is prepared rather than streamed."""
    from types import SimpleNamespace

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem
    from huggingface_hub.hf_api import GitRefInfo, GitRefs

    from dew.data.compile import compile
    from dew.data.dataset import DataPartition
    from dew.data.pipeline import Pipeline
    from dew.data.sources import Hub

    rows = pa.table({"id": list(range(12))})
    stream = pa.BufferOutputStream()
    with pa.ipc.new_stream(stream, rows.schema) as writer:
        writer.write_table(rows)
    converted = pa.BufferOutputStream()
    pq.write_table(rows, converted)
    payloads = {
        "datasets/owner/data@original/train/data-00000-of-00001.arrow": stream.getvalue().to_pybytes(),
        "datasets/owner/data@converted/default/train/0000.parquet": converted.getvalue().to_pybytes(),
    }

    def header(self, path, **kwargs):
        return {"name": path, "type": "file", "size": len(payloads[path]),
                "sha256": hashlib.sha256(payloads[path]).hexdigest()}

    def listing(self, repo, *, repo_type, revision):
        return (["default/train/0000.parquet"] if revision == "converted"
                else ["train/data-00000-of-00001.arrow", "train/state.json"])

    conversion = GitRefInfo(name="parquet", ref="refs/convert/parquet", target_commit="converted")
    monkeypatch.setattr(HfApi, "dataset_info", lambda self, repo, **kwargs: SimpleNamespace(sha="original"))
    monkeypatch.setattr(HfApi, "list_repo_files", listing)
    monkeypatch.setattr(HfApi, "list_repo_refs",
                        lambda self, repo, **kwargs: GitRefs(branches=[], converts=[conversion], tags=[]))
    monkeypatch.setattr(HfFileSystem, "info", header)
    monkeypatch.setattr(HfFileSystem, "_open", lambda self, path, **kwargs: io.BytesIO(payloads[path]))
    reader = compile(Pipeline(Hub("owner/data")).batch(4), read="streamed",
                     loading=Loading(workers=0)).train(DataPartition())
    try:
        assert [int(value) for _ in range(3) for value in next(reader)["id"]] == list(range(12))
    finally:
        reader.close()
    assert Hub("owner/data", revision="original").streamed is None


def test_a_conversion_that_moves_under_one_default_commit_is_another_snapshot(moving_hub_conversion):
    """One source pins its conversion once; a new source refuses the old plan
    after that conversion moves under the same default commit."""
    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline
    from dew.data.sources import Hub

    def plan():
        streamed = Pipeline(Hub("owner/data")).batch(2)
        return compile(streamed, read="streamed", loading=Loading(workers=0)).plan

    source = Hub("owner/data")
    snapshot = source.files.fingerprint()
    saved = plan()
    assert saved.manifest.source["sources"] == [snapshot]
    moving_hub_conversion.append("second")
    assert source.files.url == ("hf://datasets/owner/data@first/default/train/0000.parquet",)
    assert source.files.fingerprint() == snapshot != Hub("owner/data").files.fingerprint()
    with pytest.raises(ValueError, match=r"saved data plan (source|order) changed"):
        plan().check(saved.record())


@pytest.mark.parametrize("wrapper,read", [
    ("hash", "streamed"), ("concatenate", "streamed"), ("concatenate_hash", "streamed"),
    ("concatenate", "indexed"), ("slice", "indexed"), ("random", "indexed"), ("hash", "memory"),
])
def test_wrapped_hub_resume_checks_the_conversion_only_when_consumed(moving_hub_conversion, wrapper, read):
    from dew.config.data import HoldOut, RecordsData, loader
    from dew.data.sources import Hub
    from dew.data.sources.concatenate import Concatenate

    def loaded():
        source = Hub("owner/data")
        if wrapper.startswith("concatenate"):
            source = Concatenate((source, Memory([{"id": index} for index in range(40, 48)])))
        holdout = (
            HoldOut(fraction=.25, by="hash") if wrapper.endswith("hash") else
            HoldOut(records=8, by="index" if wrapper == "slice" else "random")
            if wrapper in ("slice", "random") else None
        )
        return loader(RecordsData(source, validation=holdout), batch=2, read=read, loading=Loading(workers=0))

    data = loaded()
    reader = data.train(DataPartition())
    try:
        next(reader)
        saved = reader.get_state()
        expected = next(reader)["id"]
    finally:
        reader.close()
    unchanged = loaded().train(DataPartition())
    try:
        unchanged.set_state(saved)
        np.testing.assert_array_equal(next(unchanged)["id"], expected)
    finally:
        unchanged.close()
    moving_hub_conversion.append("second")
    moved = loaded().train(DataPartition())
    try:
        if read == "streamed":
            with pytest.raises(ValueError, match=r"saved data plan (source|order) changed"):
                moved.set_state(saved)
        else:
            moved.set_state(saved)
            np.testing.assert_array_equal(next(moved)["id"], expected)
    finally:
        moved.close()


@pytest.mark.parametrize("recipe,moving_hub_conversion,read", [
    ("prompts", "prompts", "streamed"), ("prompts", "prompts", "indexed"), ("prompts", "prompts", "memory"),
    ("decisions", "decisions", "streamed"), ("decisions", "decisions", "indexed"),
    ("decisions", "decisions", "memory"),
], indirect=["moving_hub_conversion"])
def test_metadata_wrapped_hub_resume_keeps_facts_and_checks_consumed_files(
    moving_hub_conversion, recipe, read
):
    from dew.config.data import PromptData, loader
    from dew.data.sources import Hub
    from dew.decision.data import Decisions

    def loaded():
        source = Hub("owner/data")
        configured = (
            PromptData(source, "byte", max_prompt_len=4, reward_bytes=None if read == "memory" else 8)
            if recipe == "prompts" else Decisions(source, options=("a", "b"))
        )
        return loader(configured, batch=2, read=read, loading=Loading(workers=0))

    reader = loaded().train(DataPartition())
    try:
        next(reader)
        saved = reader.get_state()
        expected = next(reader)
    finally:
        reader.close()
    moving_hub_conversion.append("second")
    moved = loaded().train(DataPartition())
    try:
        if read == "streamed":
            with pytest.raises(ValueError, match=r"saved data plan (source|order) changed"):
                moved.set_state(saved)
        else:
            moved.set_state(saved)
            actual = next(moved)
            assert actual.keys() == expected.keys()
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
    finally:
        moved.close()


@pytest.mark.parametrize("format", ["jsonl", "parquet"])
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


@pytest.mark.parametrize("revision,native", [(None, True), ("original", True), (None, False)])
def test_hub_shard_capability_and_urls_list_each_revision_once(monkeypatch, revision, native):
    from types import SimpleNamespace

    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import GitRefInfo, GitRefs

    from dew.data.sources import Hub

    listed = []

    def listing(self, repo, *, repo_type, revision):
        listed.append(revision)
        if not native and revision == "original":
            return ["train.tar"]
        return ["train-1.parquet", "train-0.parquet"]

    conversion = GitRefInfo(name="parquet", ref="refs/convert/parquet", target_commit="converted")
    monkeypatch.setattr(HfApi, "dataset_info", lambda self, repo, **kwargs: SimpleNamespace(sha="original"))
    monkeypatch.setattr(HfApi, "list_repo_files", listing)
    monkeypatch.setattr(HfApi, "list_repo_refs",
                        lambda self, repo, **kwargs: GitRefs(branches=[], converts=[conversion], tags=[]))
    source = Hub("owner/data", revision=revision)
    commit = "original" if native else "converted"
    assert source.supports("streamed")
    assert source.streamed == commit
    assert source.files.url == tuple(f"hf://datasets/owner/data@{commit}/train-{index}.parquet"
                                     for index in range(2))
    assert source.files.format == "parquet"
    assert source.supports("streamed")
    assert listed == (["original"] if native else ["original", "converted"])


@pytest.mark.parametrize("revision", [None, "original"])
def test_hub_refuses_streaming_when_the_chosen_revision_has_no_shards_of_the_split(monkeypatch, revision):
    from types import SimpleNamespace

    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import GitRefInfo, GitRefs

    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline
    from dew.data.sources import Hub

    conversion = GitRefInfo(name="parquet", ref="refs/convert/parquet", target_commit="converted")
    monkeypatch.setattr(HfApi, "dataset_info", lambda self, repo, **kwargs: SimpleNamespace(sha="original"))
    monkeypatch.setattr(HfApi, "list_repo_files", lambda self, repo, **kwargs: ["default/test/0000.parquet"])
    monkeypatch.setattr(HfApi, "list_repo_refs",
                        lambda self, repo, **kwargs: GitRefs(branches=[], converts=[conversion], tags=[]))
    source = Hub("owner/data", revision=revision)
    assert not source.supports("streamed")
    assert source.streamed is None
    with pytest.raises(FileNotFoundError, match="split 'train'"):
        _ = source.files
    with pytest.raises(ValueError, match=r"read='streamed' is unavailable"):
        compile(Pipeline(source).batch(2), read="streamed", loading=Loading(workers=0))


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
    files = Files(root.url, format="parquet", columns=("value", "split"), where=ds.field("value") >= 15)
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
    with pytest.raises(ValueError, match="declares format"):
        Files(root, format="jsonl").records("memory")


def test_spark_directory_requires_success_marker(tmp_path):
    table = pa.table({"value": [1, 2]})
    ds.write_dataset(table, str(tmp_path), format="parquet")
    with pytest.raises(ValueError, match="_SUCCESS"):
        Files(str(tmp_path), format="parquet", spark=True).records("indexed")
    (tmp_path / "_SUCCESS").write_bytes(b"done")
    assert len(Files(str(tmp_path), format="parquet", spark=True).records("indexed")) == 2


def test_corrupt_export_manifest_names_the_object_and_never_falls_back_to_raw_shards():
    root = Store(f"memory://corrupt-export-{uuid4().hex}")
    export(Memory([{"id": 1}]), root.url)
    root.child("_dew_manifest.json").write(b"not-json")
    with pytest.raises(ValueError, match=r"corrupt-export.*_dew_manifest.json"):
        Files(root.url, format="parquet").records("indexed")


def test_export_manifest_cannot_reference_an_object_outside_its_root():
    root = Store(f"memory://escaping-export-{uuid4().hex}/dataset")
    root.child("_dew_manifest.json").write(
        json.dumps({"schema": 1, "format": "jsonl", "files": [{"path": "../outside.jsonl"}]}).encode()
    )
    with pytest.raises(ValueError, match="stay inside its root"):
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
    with pytest.raises(ValueError, match="one split"):
        source.with_split("val")


@pytest.mark.parametrize("format_args", [[], ["--format", "array_record"]])
def test_cli_token_exports_are_readable_by_the_dispatched_token_source(tmp_path, format_args):
    from dew.cli.main import main
    from dew.data.sources import TokenFiles, source
    from dew.data.text import ByteTokenizer

    documents = ["first café", "next document", "last"]
    input = tmp_path / "documents.jsonl"
    input.write_text("".join(json.dumps({"text": text}) + "\n" for text in documents))
    destination = tmp_path / "tokens"
    assert main(["data", "export", "records", str(destination),
                 "--fields", json.dumps({"source": str(input)}), "--tokenizer", "byte", "--eos",
                 "--rows-per-shard", "2", *format_args]) == 0
    stored = source(str(destination))
    assert isinstance(stored, TokenFiles)
    vocabulary = ByteTokenizer()
    expected = np.concatenate([np.append(vocabulary.encode(text), vocabulary.eos_id) for text in documents])
    assert stored.tokens[:].tolist() == expected.tolist()
    assert stored.supports("indexed")
    assert stored.metadata()["eos_id"] == vocabulary.eos_id
    assert stored.metadata()["train_tokens"] == len(expected)
    records = stored.records("indexed")
    actual = np.concatenate([records[index]["text"] for index in range(len(records))])
    assert actual.tolist() == expected.tolist()


def test_cli_refuses_an_unsupported_token_export_format_before_writing(tmp_path):
    from dew.cli.main import main

    input = tmp_path / "documents.jsonl"
    input.write_text('{"text":"a document"}\n')
    destination = tmp_path / "tokens"
    with pytest.raises(ValueError):
        main(["data", "export", "records", str(destination),
              "--fields", json.dumps({"source": str(input)}), "--tokenizer", "byte", "--format", "parquet"])
    assert not destination.exists()


def test_cli_record_exports_still_default_to_parquet(tmp_path):
    from dew.cli.main import main
    from dew.data.sources import source

    input = tmp_path / "documents.jsonl"
    input.write_text('{"text":"a document"}\n')
    destination = tmp_path / "records"
    assert main(["data", "export", "records", str(destination),
                 "--fields", json.dumps({"source": str(input)})]) == 0
    stored = source(str(destination))
    assert isinstance(stored, Files)
    assert stored.kind == "parquet"
    assert stored.records("indexed")[0] == {"text": "a document"}


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


@pytest.mark.parametrize("read", [None, "indexed"])
def test_snapshot_reads_a_prepared_hub_without_physical_shards(tmp_path, monkeypatch, read):
    from types import SimpleNamespace

    import datasets
    from huggingface_hub import HfApi

    from dew.data import Hub

    (tmp_path / "dataset_info.json").write_text("{}")
    prepared = datasets.Dataset.from_dict({"id": [4, 5, 6]})
    builder = SimpleNamespace(cache_dir=str(tmp_path), config=SimpleNamespace(data_files=None),
                              info=SimpleNamespace(features=None, splits=None),
                              as_dataset=lambda **kwargs: prepared)
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda *args, **kwargs: builder)
    monkeypatch.setattr(HfApi, "dataset_info", lambda *args, **kwargs: SimpleNamespace(sha="pinned"))
    monkeypatch.setattr(HfApi, "list_repo_files", lambda *args, **kwargs: ["train.tar"])
    chain = Hub("owner/data", revision="pinned").threads(1)
    if read is not None:
        chain = chain.read(read)
    destination = tmp_path / "snapshot"
    chain.snapshot(str(destination))
    stored = Files(str(destination)).records("indexed")
    assert [stored[index]["id"] for index in range(len(stored))] == [4, 5, 6]


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
    from dew.data.sources import records

    source = records.ArrayRecords(arrayrecord_paths, counts=(8,) * len(arrayrecord_paths))
    entered = threading.Barrier(3)
    released = [threading.Event(), threading.Event()]
    readers = []

    class HeldReader(records.ArrayRecords):
        def __init__(self, urls, *args, **kwargs):
            super().__init__(urls, *args, **kwargs)
            self.active = self.closes = 0
            self.closed_while_reading = False
            self.held = not readers
            self.read_lock = threading.Lock()
            readers.append(self)

        def __getitem__(self, index):
            with self.read_lock:
                self.active += 1
            try:
                if self.held and index < 2:
                    entered.wait(timeout=5)
                    assert released[index].wait(timeout=5)
                    if fail:
                        raise ValueError("failed child read")
                return super().__getitem__(index)
            finally:
                with self.read_lock:
                    self.active -= 1

        def close(self):
            self.closed_while_reading |= self.active > 0
            self.closes += 1
            super().close()

    monkeypatch.setattr(records, "ArrayRecords", HeldReader)
    try:
        with ThreadPoolExecutor(2) as pool:
            pending = [pool.submit(source.__getitem__, index) for index in range(2)]
            try:
                entered.wait(timeout=5)
                if retire == "eviction":
                    for shard in range(1, len(arrayrecord_paths)):
                        assert source[8 * shard] == f"{shard}:0".encode()
                else:
                    source.close()
                held = readers[0]
                assert held.closes == 0
                # Close retires the busy reader; eviction keeps it and retires an idle one.
                assert source[2] == b"0:2"
                replacement = readers[-1]
                for index, future in enumerate(pending):
                    released[index].set()
                    if fail:
                        with pytest.raises(ValueError, match="failed child read"):
                            future.result(timeout=5)
                    else:
                        assert future.result(timeout=5) == f"0:{index}".encode()
                    assert held.closes == (1 if retire == "close" and index == 1 else 0)
                    assert replacement.closes == 0
                assert not held.closed_while_reading
            finally:
                for release in released:
                    release.set()
    finally:
        source.close()


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


def test_arrayrecords_refuse_a_mirror_without_room_for_the_read_width(arrayrecord_paths, tmp_path):
    from dew.data.sources.records import ArrayRecords

    shard = Store(f"memory://small-record-mirror-{uuid4().hex}/0.array_record")
    shard.write(Path(arrayrecord_paths[0]).read_bytes())
    source = ArrayRecords(
        (shard.url,),
        {"dewcache": {"cache_storage": str(tmp_path / "mirror"), "max_bytes": 1}},
        counts=(8,),
    )
    try:
        with pytest.raises(ValueError, match=rf"room for {Loading().threads} physical shards"):
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

    before = reader()
    try:
        identity = before.identity()
        batch = next(before)
        assert 0 <= int(batch["id"][0]) < 33
        assert batch["text"].tolist() == ["a" * (192 * 1024)]
        position = before.get_state()
    finally:
        before.close()
    path = tmp_path / "part-16.jsonl"
    size = path.stat().st_size
    with path.open("r+b") as file:
        file.seek(size // 2)
        file.write(b"b")
    assert path.stat().st_size == size
    after = reader()
    try:
        assert after.identity() != identity
        with pytest.raises(ValueError):
            after.set_state(position)
    finally:
        after.close()


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


def test_torch_source_refuses_streaming_and_cache_without_content_identity():
    import torch
    from torch.utils.data import TensorDataset

    from dew.data.cache import Cache
    from dew.data.plan import Manifest
    from dew.data.sources import Torch

    rows = Torch(TensorDataset(torch.arange(4)), fields=("id",))
    assert not rows.supports("streamed")
    assert int(rows.records("indexed")[2]["id"]) == 2
    with pytest.raises(ValueError):
        rows.records("streamed")
    import grain.python as grain

    with pytest.raises(ValueError):
        Cache(f"memory://torch-cache-{uuid4().hex}").mapped(
            grain.MapDataset.source(rows.records("indexed")), Manifest(rows.fingerprint()), rows
        )


@pytest.mark.parametrize("read", ["memory", "indexed"])
@pytest.mark.parametrize("resume", ["exact", "non-exact"])
def test_torch_compilation_requires_identity_even_for_equal_shapes(read, resume):
    """Same-class, same-length datasets cannot name an exact position by their shape."""
    import torch
    from torch.utils.data import TensorDataset

    from dew.data import from_torch
    from dew.data.compile import compile

    for offset in (0, 100):
        source = from_torch(TensorDataset(torch.arange(32) + offset), fields=("id",))
        with pytest.raises(ValueError):
            compile(source.shuffle(4).repeat().batch(4), read=read, resume=resume,
                    loading=Loading(workers=0))


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


@pytest.mark.parametrize("kind", ["grain", "iterator"])
@pytest.mark.parametrize("resume", ["exact", "non-exact"])
def test_foreign_sources_cannot_use_an_empty_content_identity(kind, resume):
    """Grain and iterator adapters require a real source name, just as Torch requires identity."""
    import grain.python as grain

    from dew.data import from_grain, from_iter
    from dew.data.compile import compile

    rows = [{"id": index} for index in range(8)]
    source = (from_grain(grain.MapDataset.source(rows), "") if kind == "grain"
              else from_iter(lambda: iter(rows), name=""))
    with pytest.raises(ValueError):
        compile(source.repeat().batch(2), read="indexed" if kind == "grain" else "streamed",
                resume=resume, loading=Loading(workers=0))


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
    from dew.data.sources.tokens import TokenFiles

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


@dataclasses.dataclass(frozen=True)
class _StreamOnly:
    """Memory documents that state their lengths but are read only as a
    stream, as a store filtered by a predicate is."""

    memory: Memory

    def describe(self):
        return f"stream-only {self.memory.describe()}"

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


def test_a_run_that_states_no_read_reads_as_its_sources_choose(tmp_path):
    """Records held in memory read from memory, a store that reads by index
    reads by index and a stream streams, when the run states no read; a read
    it states wins over each."""
    import pyarrow.parquet as pq

    from dew.config import RunConfig
    from dew.config.data import RecordsData, loader

    rows = [{"value": index} for index in range(8)]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "rows.parquet")
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def read(source, stated=RunConfig.data_read):
        loaded = loader(RecordsData(source=source), batch=2, read=stated, loading=Loading(workers=0))
        assert loaded.plan is not None
        return loaded.plan.order["read"]

    listed, indexed, streamed = rows, str(tmp_path / "rows.parquet"), str(tmp_path / "rows.jsonl")
    assert [read(listed), read(indexed), read(streamed)] == ["memory", "indexed", "streamed"]
    assert [read(listed, "streamed"), read(indexed, "memory"), read(streamed, "memory")] == [
        "streamed", "memory", "memory"]


def test_a_run_that_states_no_read_takes_the_first_read_its_sources_can_all_take(tmp_path):
    """A CSV with no index prefers a stream, but its random holdout cannot be
    streamed and the CSV cannot be read by index, so the run reads it into
    memory and trains on the half the holdout leaves."""
    from dew.config.data import HoldOut, RecordsData, loader

    (tmp_path / "rows.csv").write_text("text\n" + "".join(f"row {index}\n" for index in range(8)))
    split = HoldOut(fraction=0.5, by="random")
    loaded = loader(RecordsData(source=str(tmp_path / "rows.csv"), validation=split), batch=2, read=None,
                  loading=Loading(workers=0))
    assert loaded.plan is not None and loaded.plan.order["read"] == "memory"
    assert loaded.records == 4


def test_a_token_holdout_with_no_stated_read_reads_unindexed_text_into_memory(tmp_path):
    """A token split needs a finite indexed order, which JSON lines without an
    index can neither stream nor read by index, so the run reads them into
    memory and holds out the first quarter of their tokens."""
    from dew.config.data import HoldOut, TextData, loader

    documents = [{"text": f"document {index} " * 6} for index in range(8)]
    (tmp_path / "text.jsonl").write_text("".join(json.dumps(document) + "\n" for document in documents))
    recipe = TextData(source=str(tmp_path / "text.jsonl"), seq_len=8, token_holdout=HoldOut(fraction=0.25))
    loaded = loader(recipe, batch=2, read=None, loading=Loading(workers=0))
    assert loaded.plan is not None and loaded.plan.order["read"] == "memory"
    assert loaded.val is not None and loaded.records


def test_a_token_holdout_over_an_indexed_store_reads_into_memory_when_no_read_is_stated(tmp_path):
    """Parquet reads by index, but a recipe's default order repeats an indexed
    read before its token split, which needs a finite order. Judged on the
    pipelines each read would run, the run reads into memory, where the order
    follows the split."""
    import pyarrow.parquet as pq

    from dew.config.data import HoldOut, TextData, loader

    rows = [{"text": f"document {index} " * 6} for index in range(8)]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "rows.parquet")
    recipe = TextData(source=str(tmp_path / "rows.parquet"), seq_len=8, token_holdout=HoldOut(fraction=0.25))
    loaded = loader(recipe, batch=2, read=None, loading=Loading(workers=0))
    assert loaded.plan is not None and loaded.plan.order["read"] == "memory"


def test_a_default_read_is_judged_on_the_order_the_recipe_adds_at_that_read(tmp_path):
    """A token split straight after an indexed source is a global, finite
    order by itself, but the default order a recipe adds at an indexed read
    repeats before it. The run reads into memory, where the order follows
    the split, rather than choosing indexed for the recipe as written."""
    import pyarrow.parquet as pq

    from dew.config.data import loader
    from dew.data.pipeline import Pipeline

    class Split:
        def pipelines(self):
            return Pipeline(Files(str(tmp_path / "rows.parquet"))).token_split(0.5, held=False), None, ()

    rows = [{"text": [index] * 4} for index in range(8)]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "rows.parquet")
    loaded = loader(Split(), batch=1, read=None, loading=Loading(workers=0))
    assert loaded.plan is not None and loaded.plan.order["read"] == "memory"


def test_a_token_store_without_its_length_index_is_refused_naming_provisioning(tmp_path):
    """A token store that cannot be read by span until its length index is
    built is not read whole into memory when the run states no read: the
    refusal names provisioning, and provision=True builds the index."""
    from array_record.python.array_record_module import ArrayRecordWriter

    from dew.config.data import TextData, loader

    Store(str(tmp_path)).child("meta.json").write(json.dumps({
        "tokenizer": "byte", "dtype": "uint16", "vocab_size": 256, "train_tokens": 40, "eos_id": None,
    }).encode())
    writer = ArrayRecordWriter(str(tmp_path / "train-0.array_record"), "group_size:1")
    for start in range(0, 40, 10):
        writer.write(np.arange(start, start + 10, dtype=np.uint16).tobytes())
    writer.close()
    with pytest.raises(ValueError, match="provision"):
        loader(TextData(source=str(tmp_path), seq_len=4), batch=1, read=None, loading=Loading(workers=0))
    loaded = loader(TextData(source=str(tmp_path), seq_len=4), batch=1, read=None, provision=True,
                  loading=Loading(workers=0))
    assert loaded.plan is not None and loaded.plan.order["read"] == "streamed"


def test_a_local_store_whose_producer_declared_other_bytes_is_refused(tmp_path):
    """A producer's manifest states each file's bytes and checksum; a copy on this host's disk
    that holds other bytes (a cut of the files under the whole set's manifest) is refused, naming
    both, and one that holds what it declares is read."""
    import hashlib

    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(8)))
    url = str(path)
    held = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert [row["id"] for row in Files(url, file_info={url: held}).iterate()] == list(range(8))
    for stale in ({**held, "bytes": 2 * held["bytes"]}, {**held, "sha256": "0" * 64}):
        with pytest.raises(ValueError):
            Files(url, file_info={url: stale}).fingerprint()
