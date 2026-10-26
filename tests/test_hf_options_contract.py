"""Hub sources pin revisions and open prepared Arrow only when requested."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hub_support import native_builder

from dew.data.sources.hub import Hub


def test_hub_builder_uses_the_resolved_commit_and_named_configuration(tmp_path, monkeypatch):
    import datasets
    from huggingface_hub import HfApi

    requested = []
    # The real builder `datasets` makes of one split, which Hub reads as the Hub's would be read.
    expected = native_builder({"validation": datasets.Dataset.from_dict({"id": [0, 1]})},
                              tmp_path / "files", tmp_path / "arrow")
    monkeypatch.setattr(HfApi, "dataset_info", lambda self, repo_id, **kwargs: SimpleNamespace(sha="frozen"))
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda repo, **kwargs:
                        requested.append((repo, kwargs)) or expected)
    source = Hub("acme/records", split="validation", name="named", revision="release", cache_dir="arrow")
    assert source.builder is expected
    assert requested == [("acme/records", {"name": "named", "revision": "frozen", "cache_dir": "arrow"})]
    assert source.fingerprint() == {"store": "hf", "repo": "acme/records", "revision": "frozen",
                                 "config": "named", "split": "validation"}


def test_a_split_prepared_in_two_caches_is_one_snapshot_and_one_plan(tmp_path, monkeypatch):
    """Two hosts prepare one pinned split, each in a cache directory of its
    own, and a third has not prepared it. Where a host keeps its Arrow files,
    or whether it has any, is no part of the data, so all three name one
    snapshot and the two that read it save one plan."""
    import datasets
    from huggingface_hub import HfApi

    from dew.data import Loading
    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline

    rows = {"train": datasets.Dataset.from_dict({"id": list(range(8))})}
    builders = {str(cache): native_builder(rows, tmp_path, cache)
                for cache in (tmp_path / "first", tmp_path / "second")}
    for builder in builders.values():
        builder.download_and_prepare()
    monkeypatch.setattr(HfApi, "dataset_info", lambda self, repo_id, **kwargs: SimpleNamespace(sha="frozen"))
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda repo, cache_dir, **_: builders[cache_dir])
    sources = [Hub("acme/records", revision="frozen", cache_dir=cache) for cache in builders]
    plans = [compile(Pipeline(source).batch(2), loading=Loading(workers=0)).plan for source in sources]
    unprepared = Hub("acme/records", revision="frozen", cache_dir=str(tmp_path / "third"))
    assert sources[0].fingerprint() == sources[1].fingerprint()
    assert plans[0] is not None and plans[1] is not None
    assert plans[0].record() == plans[1].record()
    assert unprepared.fingerprint() == sources[0].fingerprint()


def test_a_split_with_no_shards_and_no_conversion_reads_indexed_but_not_streamed(tmp_path, monkeypatch):
    """A default revision whose split has no native shards, in a repo with no
    Parquet conversion, has nothing to stream. That is a capability it lacks:
    its prepared Arrow compiles and reads by index, and a streamed read is
    refused as unavailable."""
    import datasets
    import httpx
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RevisionNotFoundError
    from huggingface_hub.hf_api import GitRefs

    from dew.data import Loading
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition
    from dew.data.pipeline import Pipeline

    def metadata(self, repo_id, revision=None, **kwargs):
        if revision == "refs/convert/parquet":
            url = f"https://huggingface.co/api/datasets/{repo_id}/revision/refs%2Fconvert%2Fparquet"
            response = httpx.Response(404, request=httpx.Request("GET", url))
            raise RevisionNotFoundError("404 Client Error: Revision Not Found", response=response)
        return SimpleNamespace(sha="frozen")

    builder = native_builder({"train": datasets.Dataset.from_dict({"id": list(range(8))})}, tmp_path,
                             tmp_path / "arrow")
    builder.download_and_prepare()
    monkeypatch.setattr(HfApi, "dataset_info", metadata)
    monkeypatch.setattr(HfApi, "list_repo_files", lambda self, repo_id, **kwargs: ["train.tar"])
    monkeypatch.setattr(HfApi, "list_repo_refs",
                        lambda self, repo_id, **kwargs: GitRefs(branches=[], converts=[], tags=[]))
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda *args, **kwargs: builder)
    assert Hub("acme/records").streamed is None
    pipeline = Pipeline(Hub("acme/records")).batch(2)
    reader = compile(pipeline, read="indexed", loading=Loading(workers=0)).train(DataPartition())
    try:
        assert sorted(int(value) for _ in range(4) for value in next(reader)["id"]) == list(range(8))
    finally:
        reader.close()
    with pytest.raises(ValueError):
        compile(Pipeline(Hub("acme/records")).batch(2), read="streamed", loading=Loading(workers=0))


@pytest.mark.parametrize("read", ["indexed", "memory"])
def test_a_prepared_arrow_read_resumes_when_only_the_parquet_conversion_moves(tmp_path, monkeypatch, read):
    import datasets
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import GitRefInfo, GitRefs

    from dew.data import Loading
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition
    from dew.data.pipeline import Pipeline

    builder = native_builder({"train": datasets.Dataset.from_dict({"id": list(range(8))})}, tmp_path,
                             tmp_path / "arrow")
    builder.download_and_prepare()
    converted = ["first"]

    def refs(self, repo_id, **kwargs):
        ref = GitRefInfo(name="parquet", ref="refs/convert/parquet", target_commit=converted[-1])
        return GitRefs(branches=[], converts=[ref], tags=[])

    def listing(self, repo_id, *, revision, repo_type):
        return ["train.tar"] if revision == "original" else ["default/train/0000.parquet"]

    monkeypatch.setattr(HfApi, "dataset_info",
                        lambda self, repo_id, **kwargs: SimpleNamespace(sha="original"))
    monkeypatch.setattr(HfApi, "list_repo_files", listing)
    monkeypatch.setattr(HfApi, "list_repo_refs", refs)
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda *args, **kwargs: builder)

    def reader():
        pipeline = Pipeline(Hub("acme/records")).shuffle(0).repeat().batch(2)
        return compile(pipeline, read=read, loading=Loading(workers=0)).train(DataPartition())

    first = reader()
    try:
        next(first)
        state = first.get_state()
        expected = next(first)["id"].tolist()
    finally:
        first.close()
    converted.append("second")
    resumed = reader()
    try:
        resumed.set_state(state)
        assert resumed.manifest_changes == ()
        assert next(resumed)["id"].tolist() == expected
    finally:
        resumed.close()


def test_an_indexed_hub_read_does_not_list_refs_or_files(tmp_path, monkeypatch):
    import datasets
    from huggingface_hub import HfApi

    from dew.data import Loading
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition
    from dew.data.pipeline import Pipeline

    builder = native_builder({"train": datasets.Dataset.from_dict({"id": list(range(8))})}, tmp_path,
                             tmp_path / "arrow")
    builder.download_and_prepare()

    def unused(*args, **kwargs):
        raise AssertionError("an indexed read does not use the repo's physical shards or conversion")

    monkeypatch.setattr(HfApi, "dataset_info",
                        lambda self, repo_id, **kwargs: SimpleNamespace(sha="original"))
    monkeypatch.setattr(HfApi, "list_repo_files", unused)
    monkeypatch.setattr(HfApi, "list_repo_refs", unused)
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda *args, **kwargs: builder)
    pipeline = Pipeline(Hub("acme/records")).batch(2)
    reader = compile(pipeline, read="indexed", loading=Loading(workers=0)).train(DataPartition())
    try:
        assert sorted(int(value) for _ in range(4) for value in next(reader)["id"]) == list(range(8))
    finally:
        reader.close()


def test_an_unprepared_arrow_store_is_refused_without_preparation(tmp_path, monkeypatch):
    import datasets
    from huggingface_hub import HfApi

    builder = native_builder({"train": datasets.Dataset.from_dict({"id": [0, 1, 2]})},
                             tmp_path / "files", tmp_path / "arrow")
    monkeypatch.setattr(HfApi, "dataset_info", lambda self, repo_id, **kwargs: SimpleNamespace(sha="frozen"))
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda *args, **kwargs: builder)
    source = Hub("acme/records")
    with pytest.raises(FileNotFoundError):
        source.records("indexed")
    assert not Path(builder.cache_dir, "dataset_info.json").exists()
    source.provision()
    rows = source.records("indexed")
    assert [int(rows[index]["id"]) for index in range(len(rows))] == [0, 1, 2]


def test_an_unprepared_pinned_csv_split_selects_the_read_arrow_provisions(tmp_path, monkeypatch):
    """A pinned CSV split prepares real Arrow and reads it by index, never as native shards."""
    import datasets
    from huggingface_hub import HfApi

    from dew.data import Loading
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition

    rows = tmp_path / "train.csv"
    rows.write_text("id,value\n" + "".join(f"{index},{index * 3}\n" for index in range(8)))
    builder = datasets.load_dataset_builder("csv", data_files={"train": str(rows)},
                                            cache_dir=str(tmp_path / "arrow"))
    monkeypatch.setattr(HfApi, "dataset_info", lambda self, repo_id, **kwargs:
                        SimpleNamespace(sha="frozen"))
    monkeypatch.setattr(HfApi, "list_repo_files", lambda self, repo_id, **kwargs: ["train.csv"])
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda *args, **kwargs: builder)
    source = Hub("acme/records", revision="frozen")
    loading = Loading(workers=0, threads=1, read_buffer=2)
    with pytest.raises(ValueError):
        compile(source.batch(2), loading=loading)
    assert not source.supports("indexed") and not source.supports("streamed")
    data = compile(source.batch(2), provision=True, loading=loading)
    assert data.plan.order["read"] == "indexed"
    stream = data.train(DataPartition())
    try:
        batches = [next(stream) for _ in range(4)]
        assert [int(value) for batch in batches for value in batch["id"]] == list(range(8))
        for batch in batches:
            assert (batch["value"] == batch["id"] * 3).all()
    finally:
        stream.close()


def test_importing_hub_without_datasets_does_not_load_the_optional_package():
    # An environment without the optional package: Python refuses an import of a module whose
    # entry in sys.modules is None, as it refuses one that is not installed.
    code = """
import sys
sys.modules['datasets'] = None
from dew.data.sources.hub import Hub
source = Hub('acme/records', name='named', revision='frozen')
try:
    source.builder
except ImportError as error:
    assert "pip install 'dewml[streaming]'" in str(error)
    assert isinstance(error.__cause__, ModuleNotFoundError)
else:
    raise AssertionError('opening Arrow without datasets succeeded')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": "src"}, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


def test_a_memory_read_of_a_hub_image_split_holds_its_encoded_images(tmp_path, monkeypatch):
    """A Hub image split's records are the encoded images it stores, which
    the image steps decode: a memory read of the flowers set held every
    decoded image (9.3 GB for 6,552 of them, tutorial 01), where this holds
    the split. The images still come out decoded, resized and exact."""
    import io

    import datasets
    import numpy as np
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import GitRefs
    from PIL import Image as PILImage

    from dew.data import Loading, presets
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition

    pixels = [np.full((8, 8, 3), 30 * index, np.uint8) for index in range(4)]
    encoded = []
    for value in pixels:
        buffer = io.BytesIO()
        PILImage.fromarray(value).save(buffer, format="PNG")
        encoded.append({"bytes": buffer.getvalue(), "path": None})
    images = datasets.Dataset.from_dict({"image": encoded}).cast_column("image", datasets.Image())
    builder = native_builder({"train": images}, tmp_path, tmp_path / "arrow")
    builder.download_and_prepare()
    monkeypatch.setattr(HfApi, "dataset_info", lambda self, repo_id, **kwargs: SimpleNamespace(sha="frozen"))
    monkeypatch.setattr(HfApi, "list_repo_files", lambda self, repo_id, **kwargs: ["train.tar"])
    monkeypatch.setattr(HfApi, "list_repo_refs",
                        lambda self, repo_id, **kwargs: GitRefs(branches=[], converts=[], tags=[]))
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda *args, **kwargs: builder)
    held = Hub("acme/images").records("memory")
    assert [held[index]["image"]["bytes"] for index in range(4)] == [value["bytes"] for value in encoded]
    chain = Hub("acme/images").shuffle(0).repeat().pipe(presets.images, 8, batch=4, rejects=False)
    data = compile(chain, read="memory", loading=Loading(workers=0, threads=1, read_buffer=1))
    reader = data.train(DataPartition())
    try:
        batch = next(reader)
    finally:
        reader.close()
    assert sorted(int(image[0, 0, 0]) for image in batch["image"]) == [0, 30, 60, 90]

