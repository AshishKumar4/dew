"""HF option values and the optional dependency's real failure boundary."""

import builtins
from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace

import pytest

from dew.data.sources import hf
from dew.data.sources.hf import HFOptions


@pytest.mark.parametrize("streaming", [False, True])
def test_default_hf_load_forwards_the_complete_library_contract(monkeypatch, streaming):
    calls = []
    table = object()

    def load_dataset(path, **kwargs):
        calls.append((path, kwargs))
        return table

    monkeypatch.setattr(hf, "_hf_datasets", lambda: SimpleNamespace(load_dataset=load_dataset))
    options = HFOptions()
    assert options.load("acme/records", "validation", streaming=streaming) is table
    expected = {field.name: getattr(options, field.name) for field in fields(options)}
    expected["name"] = expected.pop("config")
    expected.update(split="validation", streaming=streaming)
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
        hf._hf_datasets()
    assert failure.value.__cause__ is fault
