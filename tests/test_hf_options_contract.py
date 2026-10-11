"""Hub sources pin revisions and open prepared Arrow only when requested."""

import builtins
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from dew.data.sources.hub import Hub


def test_hub_builder_uses_the_resolved_commit_and_named_configuration(monkeypatch):
    import datasets
    from huggingface_hub import HfApi

    requested = []
    expected = object()
    monkeypatch.setattr(HfApi, "dataset_info", lambda self, repo_id, **kwargs: SimpleNamespace(sha="frozen"))
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda repo, **kwargs:
                        requested.append((repo, kwargs)) or expected)
    source = Hub("acme/records", split="validation", config="named", revision="release", cache_dir="arrow")
    assert source.builder is expected
    assert requested == [("acme/records", {"name": "named", "revision": "frozen", "cache_dir": "arrow"})]
    assert source.snapshot() == {"store": "hf", "repo": "acme/records", "revision": "frozen",
                                 "config": "named", "split": "validation"}


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
    assert 'datasets is absent' in str(error)
else:
    raise AssertionError('opening Arrow without datasets succeeded')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": "src"}, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
