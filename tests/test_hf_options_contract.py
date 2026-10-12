"""Hub sources pin revisions and open prepared Arrow only when requested."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from dew.data.sources.hub import Hub


def test_hub_builder_uses_the_resolved_commit_and_named_configuration(tmp_path, monkeypatch):
    import datasets
    from huggingface_hub import HfApi

    requested = []
    marker = tmp_path / "dataset_info.json"
    marker.write_text("{}")
    expected = SimpleNamespace(cache_dir=str(tmp_path),
                               as_dataset=lambda **kwargs: SimpleNamespace(cache_files=[]))
    monkeypatch.setattr(HfApi, "dataset_info", lambda self, repo_id, **kwargs: SimpleNamespace(sha="frozen"))
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda repo, **kwargs:
                        requested.append((repo, kwargs)) or expected)
    source = Hub("acme/records", split="validation", config="named", revision="release", cache_dir="arrow")
    assert source.builder is expected
    assert requested == [("acme/records", {"name": "named", "revision": "frozen", "cache_dir": "arrow"})]
    assert source.snapshot() == {"store": "hf", "repo": "acme/records", "revision": "frozen",
                                 "streamed": "frozen", "config": "named", "split": "validation"}


def test_a_split_prepared_in_two_caches_is_one_snapshot_and_one_plan(tmp_path, monkeypatch):
    """Two hosts prepare one pinned split, each in a cache directory of its
    own, and a third has not prepared it. Where a host keeps its Arrow files,
    or whether it has any, is no part of the data, so all three name one
    snapshot and the two that read it save one plan."""
    import datasets
    from huggingface_hub import HfApi

    from dew.data import Loading, Pipeline, compile

    rows = tmp_path / "rows.parquet"
    datasets.Dataset.from_dict({"id": list(range(8))}).to_parquet(rows)
    builders = {
        cache: datasets.load_dataset_builder("parquet", data_files={"train": str(rows)}, cache_dir=cache)
        for cache in (str(tmp_path / "first"), str(tmp_path / "second"))
    }
    for builder in builders.values():
        builder.download_and_prepare()
    monkeypatch.setattr(HfApi, "dataset_info", lambda self, repo_id, **kwargs: SimpleNamespace(sha="frozen"))
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda repo, cache_dir, **_: builders[cache_dir])
    sources = [Hub("acme/records", revision="frozen", cache_dir=cache) for cache in builders]
    plans = [compile(Pipeline(source).batch(2), loading=Loading(workers=0)).plan for source in sources]
    unprepared = Hub("acme/records", revision="frozen", cache_dir=str(tmp_path / "third"))
    assert sources[0].snapshot() == sources[1].snapshot()
    assert plans[0] is not None and plans[1] is not None
    assert plans[0].record() == plans[1].record()
    assert unprepared.snapshot() == sources[0].snapshot()


def test_a_split_with_no_shards_and_no_conversion_reads_indexed_but_not_streamed(tmp_path, monkeypatch):
    """A default revision whose split has no native shards, in a repo with no
    Parquet conversion, has nothing to stream. That is a capability it lacks:
    its snapshot names no streamed commit, its prepared Arrow compiles and
    reads by index, and a streamed read is refused as unavailable."""
    import datasets
    import httpx
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RevisionNotFoundError
    from huggingface_hub.hf_api import GitRefs

    from dew.data import DataPartition, Loading, Pipeline, compile

    def metadata(self, repo_id, revision=None, **kwargs):
        if revision == "refs/convert/parquet":
            url = f"https://huggingface.co/api/datasets/{repo_id}/revision/refs%2Fconvert%2Fparquet"
            response = httpx.Response(404, request=httpx.Request("GET", url))
            raise RevisionNotFoundError("404 Client Error: Revision Not Found", response=response)
        return SimpleNamespace(sha="frozen")

    rows = tmp_path / "rows.parquet"
    datasets.Dataset.from_dict({"id": list(range(8))}).to_parquet(rows)
    builder = datasets.load_dataset_builder("parquet", data_files={"train": str(rows)},
                                            cache_dir=str(tmp_path / "arrow"))
    builder.download_and_prepare()
    monkeypatch.setattr(HfApi, "dataset_info", metadata)
    monkeypatch.setattr(HfApi, "list_repo_files", lambda self, repo_id, **kwargs: ["train.tar"])
    monkeypatch.setattr(HfApi, "list_repo_refs",
                        lambda self, repo_id, **kwargs: GitRefs(branches=[], converts=[], tags=[]))
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda *args, **kwargs: builder)
    assert Hub("acme/records").snapshot()["streamed"] is None
    pipeline = Pipeline(Hub("acme/records")).batch(2)
    reader = compile(pipeline, read="indexed", loading=Loading(workers=0)).train(DataPartition())
    try:
        assert sorted(int(value) for _ in range(4) for value in next(reader)["id"]) == list(range(8))
    finally:
        reader.close()
    with pytest.raises(ValueError, match=r"read='streamed' is unavailable for hf:acme/records@frozen"):
        compile(Pipeline(Hub("acme/records")).batch(2), read="streamed", loading=Loading(workers=0))


def test_hub_bindings_are_frozen():
    source = Hub("acme/records", config="named")
    with pytest.raises(FrozenInstanceError):
        source.config = "another"


def test_an_unprepared_arrow_store_is_refused_without_preparation(tmp_path, monkeypatch):
    import datasets
    from huggingface_hub import HfApi

    prepared = []
    builder = SimpleNamespace(cache_dir=str(tmp_path), download_and_prepare=lambda: prepared.append(True))
    monkeypatch.setattr(HfApi, "dataset_info", lambda self, repo_id, **kwargs: SimpleNamespace(sha="frozen"))
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda *args, **kwargs: builder)
    source = Hub("acme/records")
    with pytest.raises(FileNotFoundError, match="provision"):
        source.records("indexed")
    assert prepared == []
    source.provision()
    assert prepared == [True]


def test_importing_hub_without_datasets_does_not_load_the_optional_package():
    code = """
import builtins
original = builtins.__import__
def absent(name, *args, **kwargs):
    if name == 'datasets':
        raise ModuleNotFoundError('datasets is absent')
    return original(name, *args, **kwargs)
builtins.__import__ = absent
from dew.data.sources.hub import Hub
source = Hub('acme/records', config='named', revision='frozen')
source.__dict__['commit'] = 'frozen'
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
