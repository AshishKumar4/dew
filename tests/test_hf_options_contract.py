"""HF option values and the optional dependency's real failure boundary."""

import builtins
from dataclasses import FrozenInstanceError

import pytest

from dew.data.sources.hf import HFOptions


@pytest.mark.parametrize("streaming", [False, True])
def test_default_hf_load_forwards_the_complete_library_contract(monkeypatch, streaming):
    import datasets

    calls = []
    table = object()

    def load_dataset(path, **kwargs):
        calls.append((path, kwargs))
        return table

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    options = HFOptions()
    assert options.load("acme/records", "validation", streaming=streaming) is table
    expected = {
        "name": None, "split": "validation", "streaming": streaming,
        "data_dir": None, "data_files": None, "cache_dir": None, "features": None,
        "download_config": None, "download_mode": None, "verification_mode": None,
        "keep_in_memory": None, "save_infos": False, "revision": None,
        "token": None, "num_proc": None, "storage_options": None,
    }
    assert calls == [("acme/records", expected)]
    assert expected["save_infos"] is False


def test_hf_option_bindings_are_frozen_without_freezing_the_users_opaque_objects():
    storage = {"anonymous": False}
    options = HFOptions(storage_options=storage)
    with pytest.raises(FrozenInstanceError):
        options.config = "another"
    storage["anonymous"] = True
    assert options.storage_options is storage
    assert options.storage_options["anonymous"] is True


def test_a_missing_datasets_package_names_the_extra_and_preserves_the_import_failure(monkeypatch):
    original = builtins.__import__
    fault = ImportError("datasets is absent")

    def absent(name, *args, **kwargs):
        if name == "datasets":
            raise fault
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", absent)
    with pytest.raises(ImportError, match=r"pip install 'dewml\[streaming\]'") as failure:
        HFOptions().load("acme/records", "train", streaming=False)
    assert failure.value.__cause__ is fault
