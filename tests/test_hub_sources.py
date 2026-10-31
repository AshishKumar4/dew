"""Hub sources: the commit a split pins, its native shards or Parquet conversion, and its reads
with the Hub online and offline."""

import hashlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from hub_support import native_builder, stated_shards

from dew.data.dataset import DataPartition, Loading
from dew.data.sources import Files, Memory


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
    stated_shards(monkeypatch)
    with pytest.raises(FileNotFoundError):
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
        return {"name": path, "type": "file", "size": len(payload), "blob_id": "blob",
                "lfs": {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}}

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
    # The real builder of the split, prepared: the Arrow a pinned revision without a conversion reads.
    builder = native_builder({"train": datasets.Dataset.from_dict(columns(0))}, tmp_path / "files",
                             tmp_path / "arrow")
    builder.download_and_prepare()
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda *args, **kwargs: builder)
    return converted


def test_a_split_saved_as_arrow_streams_its_parquet_conversion(monkeypatch):
    """A dataset repo's `.arrow` files are what `save_to_disk` and `push_to_hub` write, Arrow's
    stream format, which no file reader opens; a split that holds only those streams its Parquet
    conversion, and a pinned revision without one is prepared rather than streamed. The read's
    metrics go by the repository's name, not the conversion's files it reads."""
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
        return {"name": path, "type": "file", "size": len(payloads[path]), "blob_id": "blob",
                "lfs": {"sha256": hashlib.sha256(payloads[path]).hexdigest(), "size": len(payloads[path])}}

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
    data = compile(Pipeline(Hub("owner/data")).batch(4), read="streamed", loading=Loading(workers=0))
    reader = data.train(DataPartition())
    try:
        assert [int(value) for _ in range(3) for value in next(reader)["id"]] == list(range(12))
        assert {name.split("/")[2] for name in data.statistics()} == {"data"}
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
    plan().check(saved.record())
    moving_hub_conversion.append("second")
    assert source.files.url == ("hf://datasets/owner/data@first/default/train/0000.parquet",)
    assert source.files.fingerprint() == snapshot != Hub("owner/data").files.fingerprint()
    with pytest.raises(ValueError):
        plan().check(saved.record())


@pytest.mark.parametrize("wrapper,read", [
    ("hash", "streamed"), ("concatenate", "streamed"), ("concatenate_hash", "streamed"),
    ("concatenate", "indexed"), ("slice", "indexed"), ("hash", "memory"),
])
def test_wrapped_hub_resume_checks_the_conversion_only_when_consumed(moving_hub_conversion, wrapper, read):
    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline
    from dew.data.sources import Hub
    from dew.data.sources.concatenate import Concatenate

    def loaded():
        source = Hub("owner/data")
        if wrapper.startswith("concatenate"):
            source = Concatenate((source, Memory([{"id": index} for index in range(40, 48)])))
        train = Pipeline(source)
        if wrapper.endswith("hash"):
            train, _ = train.holdout(.25)
        elif wrapper == "slice":
            train, _ = train.holdout(records=8)
        order = train.shuffle(0, window=64 if read == "streamed" else None).repeat()
        return compile(order.batch(2), read=read, loading=Loading(workers=0))

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
            with pytest.raises(ValueError):
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
    from dew.data import presets
    from dew.data.compile import compile
    from dew.data.sources import Hub
    from dew.decision import decisions

    def loaded():
        source = Hub("owner/data")
        configured = (
            source.pipe(presets.prompts, "byte", max_prompt_len=4, batch=2,
                        reward_bytes=None if read == "memory" else 8)
            if recipe == "prompts" else source.pipe(decisions, options=("a", "b")).batch(2)
        )
        return compile(configured, read=read, loading=Loading(workers=0))

    reader = loaded().train(DataPartition())
    try:
        next(reader)
        saved = reader.get_state()
        expected = next(reader)
    finally:
        reader.close()
    unchanged = loaded().train(DataPartition())
    try:
        unchanged.set_state(saved)
    finally:
        unchanged.close()
    moving_hub_conversion.append("second")
    moved = loaded().train(DataPartition())
    try:
        if read == "streamed":
            with pytest.raises(ValueError):
                moved.set_state(saved)
        else:
            moved.set_state(saved)
            actual = next(moved)
            assert actual.keys() == expected.keys()
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
    finally:
        moved.close()


@pytest.mark.parametrize(("url", "format", "read_as"), [
    ("hf://datasets/owner/data", None, "Hub"),
    ("hf://datasets/owner/data@pinned", None, "Hub"),
    ("hf://datasets/owner/data/train.jsonl", None, "Files"),
    ("hf://datasets/owner/data@pinned/train.csv", None, "Files"),
    ("hf://datasets/owner/data/train.txt", None, "Files"),
    ("hf://datasets/owner/data/train.array_record", None, "Files"),
    ("hf://datasets/owner/data/train.parquet", None, "Files"),
    ("hf://datasets/owner/data/records", "jsonl", "Files"),
])
def test_a_hub_url_reads_a_repository_or_the_files_it_names(monkeypatch, url, format, read_as):
    """An hf:// URL of a repository reads the repository; one naming a file Files reads by its
    suffix, or any URL given a format, reads that file, whatever its record format."""
    from huggingface_hub import HfFileSystem

    from dew.data.sources import Files, Hub, source

    monkeypatch.setattr(HfFileSystem, "exists", lambda self, path, **kwargs: False)
    resolved = source(url, format=format)
    assert type(resolved) is {"Hub": Hub, "Files": Files}[read_as]
    if read_as == "Files":
        assert resolved.url == url and resolved.format == format


@pytest.mark.parametrize(("url", "revision"), [
    pytest.param("hf://datasets/owner/data@refs/convert/parquet", "refs/convert/parquet",
                 id="parquet-conversion"),
    pytest.param("hf://datasets/owner/data@refs/pr/3/", "refs/pr/3", id="pull-request"),
    pytest.param("hf://datasets/owner/data@feature%2Fx", "feature/x", id="branch-with-a-slash"),
    pytest.param("hf://owner/data@v1", "v1", id="tag-without-type"),
])
def test_a_hub_url_reads_its_repository_at_a_revision_holding_a_slash(monkeypatch, url, revision):
    """An hf:// URL's revision runs from its `@` to the path after it, as the Hub's own URLs spell
    one: a reference of the Hub's (its Parquet conversion, a pull request) whole, and a branch whose
    name holds a `/` with it spelled `%2F`. Such a URL names the repository at that revision, where
    the slashes it holds read as a path below the repository and its files were read instead."""
    from huggingface_hub import HfFileSystem

    from dew.data.sources import Hub, source

    monkeypatch.setattr(HfFileSystem, "exists", lambda self, path, **kwargs: False)
    resolved = source(url)
    assert type(resolved) is Hub
    assert (resolved.repo, resolved.revision) == ("owner/data", revision)


@pytest.mark.parametrize("url", [
    pytest.param("hf://datasets/owner/data/export", id="directory"),
    pytest.param("hf://datasets/owner/data@pinned/exports/run/", id="directory-at-a-revision"),
    pytest.param("hf://datasets/owner/data@refs/convert/parquet/default/train",
                 id="directory-of-the-conversion"),
    pytest.param("hf://datasets/owner/data@feature/x", id="directory-at-a-branch"),
])
def test_a_hub_path_below_its_repository_reads_as_the_files_there(monkeypatch, url):
    """An hf:// URL of a directory in a repository, such as an export's (`export` commits one there),
    is read as its files, whose manifest names their record format, never as a repository of that
    name: no suffix names its format, as none names a repository's. A revision ends at the first `/`
    past its `@` unless it is a reference of the Hub's, so `@feature/x` is the directory `x` at the
    branch `feature`, as `HfFileSystem` reads it."""
    from huggingface_hub import HfFileSystem

    from dew.data.sources import Files, source

    monkeypatch.setattr(HfFileSystem, "exists", lambda self, path, **kwargs: False)
    resolved = source(url)
    assert type(resolved) is Files
    assert resolved.url == url


@pytest.mark.parametrize("format", ["jsonl", "parquet"])
def test_hub_native_shards_keep_the_requested_revision(monkeypatch, format):
    from types import SimpleNamespace

    from huggingface_hub import HfApi

    from dew.data.sources import Hub

    monkeypatch.setattr(
        HfApi, "dataset_info", lambda self, repo, revision=None: SimpleNamespace(sha=revision)
    )
    monkeypatch.setattr(HfApi, "list_repo_files", lambda self, repo, **kwargs: [f"train.{format}"])
    stated_shards(monkeypatch)
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
        if revision == "converted":
            # The conversion holds each configuration's splits in a directory of its name.
            return ["default/train/0001.parquet", "default/train/0000.parquet"]
        if not native:
            return ["train.tar"]
        return ["train-1.parquet", "train-0.parquet"]

    conversion = GitRefInfo(name="parquet", ref="refs/convert/parquet", target_commit="converted")
    monkeypatch.setattr(HfApi, "dataset_info", lambda self, repo, **kwargs: SimpleNamespace(sha="original"))
    monkeypatch.setattr(HfApi, "list_repo_files", listing)
    monkeypatch.setattr(HfApi, "list_repo_refs",
                        lambda self, repo, **kwargs: GitRefs(branches=[], converts=[conversion], tags=[]))
    stated_shards(monkeypatch)
    source = Hub("owner/data", revision=revision)
    commit = "original" if native else "converted"
    assert source.supports("streamed")
    assert source.streamed == commit
    shards = [f"train-{index}.parquet" if native else f"default/train/000{index}.parquet"
              for index in range(2)]
    assert source.files.url == tuple(f"hf://datasets/owner/data@{commit}/{shard}" for shard in shards)
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
    with pytest.raises(FileNotFoundError):
        _ = source.files
    with pytest.raises(ValueError):
        compile(Pipeline(source).batch(2), read="streamed", loading=Loading(workers=0))


@pytest.mark.parametrize("read", [None, "indexed"])
def test_snapshot_reads_a_prepared_hub_without_physical_shards(tmp_path, monkeypatch, read):
    from types import SimpleNamespace

    import datasets
    from huggingface_hub import HfApi

    from dew.data import Hub

    builder = native_builder({"train": datasets.Dataset.from_dict({"id": [4, 5, 6]})}, tmp_path / "files",
                             tmp_path / "arrow")
    builder.download_and_prepare()
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


def test_a_hub_split_prepared_online_reads_its_own_commits_rows_with_the_hub_offline(tmp_path, monkeypatch):
    """A Hub split prepared online at two commits reads, with the Hub offline (HF_HUB_OFFLINE),
    the rows of the commit each source pins: the commit is the answer the Hub gave online
    (`dew.cache.hub_answer`), the default branch's the last given, even where another tool has since
    moved the Hub cache's ref of the branch, and the Arrow is the cache `datasets` prepared at that
    commit, which `datasets` itself does not find offline."""
    import datasets
    from hub_support import native_hub, offline

    from dew.cache import hub_ref
    from dew.data import Hub
    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline

    first, second = "0123456789abcdef0123456789abcdef01234567", "89abcdef0123456789abcdef0123456789abcdef"
    cache = str(tmp_path / "cache")

    def read(revision=None):
        pipeline = Pipeline(Hub("acme/records", revision=revision, cache_dir=cache)).batch(4)
        reader = compile(pipeline, read="indexed", loading=Loading(workers=0), provision=True).train(
            DataPartition())
        try:
            return sorted(int(value) for _ in range(4) for value in next(reader)["id"])
        finally:
            reader.close()

    for commit, start in ((first, 0), (second, 100)):
        rows = datasets.Dataset.from_dict({"id": list(range(start, start + 16))})
        with pytest.MonkeyPatch.context() as online:
            native_hub(online, tmp_path / commit, "acme/records", {"train": rows}, commit)
            assert read(commit) == list(range(start, start + 16))
            assert read() == list(range(start, start + 16))
    # Another tool sharing the Hub cache moves the branch's ref back to the first commit.
    hub_ref("acme/records", None, repo_type="dataset").write_text(first)
    offline(monkeypatch)
    assert read(first) == list(range(16))
    assert read() == list(range(100, 116))
    assert Hub("acme/records", cache_dir=cache).commit == second


def test_a_hub_splits_shards_keep_their_identity_with_the_hub_offline(moving_hub_conversion, monkeypatch):
    """A Hub split's streamed shards keep the identity a compiled cache is keyed by with the Hub
    offline: the commit, the commit's files, its conversion and each shard's facts are the Hub's
    answers, recorded online. A repo never answered online raises the offline error."""
    from hub_support import offline
    from huggingface_hub.errors import OfflineModeIsEnabled

    from dew.data.sources import Hub

    answered = Hub("owner/data").files
    identity = answered.fingerprint()
    offline(monkeypatch)
    recorded = Hub("owner/data").files
    assert recorded.url == answered.url
    assert recorded.fingerprint() == identity
    with pytest.raises(OfflineModeIsEnabled):
        assert Hub("owner/elsewhere").commit


@pytest.mark.parametrize("named", ["hub", "profile"])
@pytest.mark.parametrize("format", ["parquet", "jsonl"])
def test_a_downloaded_hub_split_streams_offline_as_it_streamed_online(tmp_path, monkeypatch, format, named):
    """A Hub split whose commit was downloaded (huggingface_hub's cache holds its files, as
    `hf_hub_download`, `snapshot_download` or a prepare leave them) streams from that cache with
    the Hub offline (HF_HUB_OFFLINE), as `huggingface_hub` reads its own files: the identity,
    shards and rows it streamed from the Hub online, fields and all, wherever the cache is,
    through the Hub source or a storage profile over the Hub."""
    from types import SimpleNamespace

    import pyarrow.parquet as pq
    from hub_support import offline
    from huggingface_hub import HfApi, HfFileSystem, constants

    from dew.data.compile import iterate
    from dew.data.pipeline import Pipeline
    from dew.data.sources import Hub
    from dew.data.storage import _profile

    commit = "0123456789abcdef0123456789abcdef01234567"
    repository = tmp_path / "repository" / "data"
    repository.mkdir(parents=True)
    rows = [{"id": index, "text": f"row {index}"} for index in range(10)]
    for split, part in (("train", rows[:6]), ("test", rows[6:])):
        path = repository / f"{split}.{format}"
        if format == "parquet":
            pq.write_table(pa.Table.from_pylist(part), path)
        else:
            path.write_text("".join(json.dumps(row) + "\n" for row in part))
    names = [f"data/{split}.{format}" for split in ("train", "test")]
    # The Hub, online, serving the repository's files at the commit.
    monkeypatch.setattr(HfApi, "dataset_info", lambda self, repo, revision=None, **options: SimpleNamespace(
        sha=commit))
    monkeypatch.setattr(HfApi, "list_repo_files", lambda self, repo, **options: names)
    monkeypatch.setattr(HfFileSystem, "info", lambda self, path, **options: {
        "name": path, "type": "file", "size": (repository / Path(path).name).stat().st_size,
        "blob_id": f"blob:{Path(path).name}"})
    monkeypatch.setattr(HfFileSystem, "_open", lambda self, path, mode="rb", **options: io.BytesIO(
        (repository / Path(path).name).read_bytes()))
    # A cache under a directory named as a Hive partition, which the rows do not hold.
    cache = tmp_path / "experiment=17" / "hub"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))

    def split():
        files = Hub("owner/rows").files
        if named == "hub":
            return files
        _profile("hubmirror", "hf", {})
        url = files.paths[0].replace("hf://", "hubmirror://")
        return Files(url, format=format, file_info={url: files.declared[files.paths[0]]})

    def streamed(source):
        read = sorted((int(row["id"]), str(row["text"]), tuple(sorted(row)))
                      for row in iterate(Pipeline(source), read="streamed"))
        return source.fingerprint(), [shard.id for shard in source.manifest().shards], read

    online = streamed(split())
    assert online[2] == [(row["id"], row["text"], ("id", "text")) for row in rows[:6]]
    shutil.copytree(repository.parent, cache / "datasets--owner--rows" / "snapshots" / commit)
    offline(monkeypatch)
    assert streamed(split()) == online


def test_a_hub_jsonl_shard_keeps_its_offset_index_in_its_identity_with_the_hub_offline(monkeypatch):
    """A Hub split's JSON lines shard and the offset index its commit holds beside it keep, with the
    Hub offline, the identity they had online, the index's facts among the shard's: an index the
    offline Hub cannot confirm is not dropped from it."""
    from types import SimpleNamespace

    from hub_support import offline
    from huggingface_hub import HfApi, HfFileSystem

    from dew.data.sources import Hub

    names = {"train.jsonl": 400, "train.index.npy": 96}
    stated = {}
    for index, (name, size) in enumerate(names.items()):
        path = f"datasets/owner/lines@pinned/{name}"
        stated[path] = {"name": path, "type": "file", "size": size, "blob_id": f"blob:{name}",
                        "lfs": {"sha256": f"{index:064x}", "size": size}}
    monkeypatch.setattr(HfApi, "dataset_info",
                        lambda self, repo, revision=None: SimpleNamespace(sha=revision))
    monkeypatch.setattr(HfApi, "list_repo_files", lambda self, repo, **kwargs: list(names))
    monkeypatch.setattr(HfFileSystem, "info", lambda self, path, **kwargs: stated[path])
    identity = Hub("owner/lines", revision="pinned").files.fingerprint()
    assert [facts["bytes"] for facts in identity["files"]] == [400, 96]
    offline(monkeypatch)
    # The offline Hub confirms no file: `HfFileSystem.exists` reads an unreachable Hub as absent.
    monkeypatch.setattr(HfFileSystem, "exists", lambda self, path, **kwargs: False)
    assert Hub("owner/lines", revision="pinned").files.fingerprint() == identity


@pytest.mark.parametrize("listing, split, expected", [
    (["data/dev-00000.parquet", "data/train-00000.parquet"], "validation", ["data/dev-00000.parquet"]),
    (["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet",
      "data/test-00000-of-00001.parquet"], "train",
     ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]),
    (["train/a.jsonl", "valid/b.jsonl", "README.md"], "validation", ["valid/b.jsonl"]),
    (["rows.parquet", "README.md"], "train", ["rows.parquet"]),
    (["pretraining/x.parquet", "train.parquet"], "train", ["train.parquet"]),
    ([], "train", []),
], ids=["dev", "sharded", "directories", "lone", "keyword-inside-a-word", "empty"])
def test_a_hub_split_holds_the_files_datasets_resolves_to_it(listing, split, expected):
    """A split's shards are the files `datasets` gives it from the repository's listing, by its own
    default patterns (`split_files`): `dev` and `valid` are `validation`'s, a sharded name's split is
    its own, a lone file is `train`'s, a keyword inside another word names no split, and a listing of
    no file (a configuration's directory holding none) gives none."""
    from dew.data.sources.hub import split_files

    assert split_files(listing, split) == expected


CARD = """---
configs:
- config_name: cola
  data_files:
  - split: train
    path: cola/train-*
  - split: validation
    path: cola/validation-*
- config_name: wiki
  data_dir: text
  data_files:
  - split: train
    path: en-*.parquet
---
"""
"""A card stating two configurations: one by its files' patterns, one under a directory that names
another (`data_dir`), its files of one language among two."""


def carded_hub(monkeypatch, directory: Path, card: str, listings: dict[str, list[str]]):
    """A repository at commit `original` whose card is `card`, each revision listing its own files
    (`listings`), its Parquet conversion at `converted`, which `refs/convert/parquet` names."""
    from types import SimpleNamespace

    from huggingface_hub import HfApi, HfFileSystem
    from huggingface_hub.hf_api import GitRefInfo, GitRefs

    (directory / "README.md").write_text(card)
    conversion = GitRefInfo(name="parquet", ref="refs/convert/parquet", target_commit="converted")
    commits = {None: "original", "refs/convert/parquet": "converted"}
    monkeypatch.setattr(HfApi, "dataset_info", lambda self, repo, revision=None, **kwargs: SimpleNamespace(
        sha=commits.get(revision, revision)))
    monkeypatch.setattr(HfApi, "list_repo_files",
                        lambda self, repo, *, repo_type, revision: listings[revision])
    monkeypatch.setattr(HfApi, "list_repo_refs",
                        lambda self, repo, **kwargs: GitRefs(branches=[], converts=[conversion], tags=[]))
    monkeypatch.setattr(HfFileSystem, "_open", lambda self, path, mode="rb", **options: io.BytesIO(
        (directory / Path(path).name).read_bytes()))
    stated_shards(monkeypatch)


@pytest.mark.parametrize("name, default, expected", [
    ("cola", None, ["cola/train-00000-of-00001.parquet"]),
    ("wiki", None, ["text/en-00000.parquet"]),
    (None, "wiki", ["text/en-00000.parquet"]),
    (None, None, None),
    ("mnli", None, None),
], ids=["named", "under-its-directory", "default", "several-unnamed", "unstated"])
def test_a_hub_split_reads_the_files_of_the_configuration_its_card_states(tmp_path, monkeypatch, name,
                                                                          default, expected):
    """A repository whose card states its configurations (`configs`) streams the split of the one named,
    or of the card's default where none is (one marked so, as `datasets` chooses it), as its patterns
    resolve under its directory: its own files, never another configuration's because a directory names
    it. Naming none where the card states several and marks no default, or naming one it does not state,
    is refused, as `datasets` refuses both. Offline, the split reads the files it read online."""
    from hub_support import offline

    from dew.data.sources import Hub

    card = CARD if default is None else CARD.replace(f"- config_name: {default}\n",
                                                     f"- config_name: {default}\n  default: true\n")
    listing = ["README.md", "cola/train-00000-of-00001.parquet", "cola/validation-00000-of-00001.parquet",
               "mnli/train-00000-of-00001.parquet", "text/en-00000.parquet", "text/fr-00000.parquet"]
    carded_hub(monkeypatch, tmp_path, card, {"original": listing})
    source = Hub("owner/carded", name=name)
    if expected is None:
        with pytest.raises(ValueError):
            _ = source.files
        return
    assert source.files.url == tuple(f"hf://datasets/owner/carded@original/{path}" for path in expected)
    offline(monkeypatch)
    assert Hub("owner/carded", name=name).files.url == source.files.url


@pytest.mark.parametrize("name, default, expected", [
    ("cola", None, "cola/train/0000.parquet"),
    (None, "wiki", "wiki/train/0000.parquet"),
    (None, None, None),
], ids=["named", "default", "several-unnamed"])
def test_a_hub_splits_parquet_conversion_reads_its_configurations_directory(tmp_path, monkeypatch, name,
                                                                           default, expected):
    """A repository with no shards a stream reads (Arrow) streams its Parquet conversion, which holds each
    configuration's splits in a directory of its name (`<config>/<split>/`): the named configuration's,
    or the default its card states, never every configuration's split of that name; naming none where the
    card states several and marks no default is refused."""
    from dew.data.sources import Hub

    card = CARD if default is None else CARD.replace(f"- config_name: {default}\n",
                                                     f"- config_name: {default}\n  default: true\n")
    carded_hub(monkeypatch, tmp_path, card, {
        "original": ["README.md", "cola/train.arrow", "text/en.arrow"],
        "converted": ["cola/train/0000.parquet", "cola/validation/0000.parquet", "wiki/train/0000.parquet"]})
    source = Hub("owner/converted", name=name)
    if expected is None:
        with pytest.raises(ValueError):
            _ = source.files
        return
    assert source.files.url == (f"hf://datasets/owner/converted@converted/{expected}",)


@pytest.mark.parametrize("revision", ["refs/convert/parquet", "converted"])
@pytest.mark.parametrize("name, expected", [("cola", "cola/train/0000.parquet"), (None, None)],
                         ids=["named", "several-unnamed"])
def test_a_split_pinned_to_its_parquet_conversion_reads_its_configurations_directory(tmp_path, monkeypatch,
                                                                                    revision, name, expected):
    """A split pinned to the Hub's Parquet conversion, by its ref or its commit, reads the named
    configuration's directory, never every configuration's split of that name; naming none where it
    holds several is refused, as its listing, which holds no card, marks none the default."""
    from dew.data.sources import Hub

    carded_hub(monkeypatch, tmp_path, CARD, {
        "converted": ["cola/train/0000.parquet", "cola/validation/0000.parquet", "wiki/train/0000.parquet"]})
    source = Hub("owner/converted", name=name, revision=revision)
    if expected is None:
        with pytest.raises(ValueError):
            _ = source.files
        return
    assert source.files.url == (f"hf://datasets/owner/converted@converted/{expected}",)


@pytest.mark.parametrize("pattern, expected", [
    ("hf://datasets/owner/carded/cola/train-*", "cola/train-00000-of-00001.parquet"),
    ("hf://datasets/owner/carded@original/cola/train-*", "cola/train-00000-of-00001.parquet"),
    ("hf://datasets/someone/else/train-*", None),
    ("{local}/train.parquet", None),
], ids=["its-repository", "its-commit", "another-repository", "a-local-path"])
def test_a_cards_pattern_naming_its_repository_by_url_reads_the_commits_listing(tmp_path, monkeypatch,
                                                                                pattern, expected):
    """A configuration whose card names its files by a URL of the repository, at its commit or none,
    reads them from the commit's listing, with no file system the listing is not; one naming files of
    another repository, or an absolute path, holds none of this commit's, so the split streams none of
    its files, a local file of that path among them."""
    from fsspec.implementations.local import LocalFileSystem
    from huggingface_hub import HfFileSystem

    from dew.data.sources import Hub

    def walked(*args, **options):
        raise AssertionError("a file system beside the listing's walked")

    (tmp_path / "local").mkdir()
    (tmp_path / "local" / "train.parquet").write_bytes(b"")
    pattern = pattern.format(local=tmp_path / "local")
    card = f"---\nconfigs:\n- config_name: cola\n  data_files:\n  - split: train\n    path: {pattern}\n---\n"
    carded_hub(monkeypatch, tmp_path, card, {
        "original": ["README.md", "cola/train-00000-of-00001.parquet", "mnli/train-00000-of-00001.parquet"],
        "converted": []})
    monkeypatch.setattr(HfFileSystem, "glob", walked)
    monkeypatch.setattr(LocalFileSystem, "glob", walked)
    source = Hub("owner/carded", name="cola")
    if expected is None:
        with pytest.raises(FileNotFoundError):
            _ = source.files
        return
    assert source.files.url == (f"hf://datasets/owner/carded@original/{expected}",)


@pytest.mark.parametrize("listing", [
    ["default/train/0000.parquet", "extra/train/0000.parquet"],
    ["part1/train/0000.parquet", "part2/train/0000.parquet"],
], ids=["default-and-extra", "parts"])
def test_a_card_less_repository_laid_out_as_a_conversion_reads_its_one_configuration(tmp_path, monkeypatch,
                                                                                     listing):
    """A repository whose own files are laid out as the Parquet conversion lays its out, with no card,
    holds the one configuration `datasets` gives it, its split every file its patterns resolve to it: the
    conversion's ref names another commit, so its directories are no configurations of its."""
    from dew.data.sources import Hub

    carded_hub(monkeypatch, tmp_path, "", {"original": listing})
    assert Hub("owner/laid-out").files.url == tuple(
        f"hf://datasets/owner/laid-out@original/{path}" for path in listing)

