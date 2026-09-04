"""safetensors round-trips for Flax parameter trees, and the hub round trip.

The tree is nested and the file is flat, so what these tests hold onto is the
'/'-joined naming: nesting, values and dtypes have to survive a save and a
load, the names on disk have to be readable by a safetensors reader that knows
nothing about dew, and a missing optional dependency has to say so. The hub
pair is tested against a recording stand-in for the hub client: what matters
is the call it makes and the directory it hands over, and no test reaches the
network.
"""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.dit import TextContext
from dew.interop import (
    hub, load_params, pull_from_hub, push_to_hub, save_hf_layout, save_params,
)
from dew.nn.backbones.dit import SimpleDiT

safetensors_numpy = pytest.importorskip("safetensors.numpy")


@pytest.fixture
def params(rng):
    model = SimpleDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2, mlp_ratio=1)
    x = jax.random.normal(rng, (1, 8, 8, 3))
    return model.init(rng, x, jnp.ones((1,)),
                      TextContext(jnp.ones((1, 77, 768)), jnp.ones((1, 77), bool)))


def flat_names(tree):
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {"/".join(entry.key for entry in path) for path, _ in leaves}


def test_round_trip_keeps_the_tree_and_the_values(params, tmp_path):
    path = tmp_path / "model.safetensors"
    save_params(params, path)
    loaded = load_params(path)

    assert jax.tree_util.tree_structure(loaded) == jax.tree_util.tree_structure(params)
    for saved, restored in zip(jax.tree.leaves(params), jax.tree.leaves(loaded)):
        assert np.array_equal(np.asarray(saved), restored)


def test_round_trip_keeps_bfloat16(params, tmp_path):
    """The point of writing bf16 is the bytes, so a load must not widen them."""
    narrowed = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), params)
    path = tmp_path / "bf16.safetensors"
    save_params(narrowed, path)
    loaded = load_params(path)

    for saved, restored in zip(jax.tree.leaves(narrowed), jax.tree.leaves(loaded)):
        assert restored.dtype == jnp.bfloat16
        assert np.array_equal(np.asarray(saved), restored)


def test_names_on_disk_are_the_slash_joined_paths(params, tmp_path):
    """A reader outside dew sees flat names, and they are the module paths."""
    path = tmp_path / "model.safetensors"
    save_params(params, path)

    names = set(safetensors_numpy.load_file(str(path)))
    assert names == flat_names(params)
    assert all(name.startswith("params/") for name in names)


def test_hf_layout_writes_the_pair_a_loader_looks_for(params, tmp_path):
    config = {"architecture": "simple_dit", "patch_size": 4, "emb_features": 32}
    export = tmp_path / "export"
    save_hf_layout(params, config, export)

    assert json.loads((export / "config.json").read_text()) == config
    loaded = load_params(export / "model.safetensors")
    assert jax.tree_util.tree_structure(loaded) == jax.tree_util.tree_structure(params)


def test_a_key_holding_the_separator_is_refused(tmp_path):
    with pytest.raises(ValueError, match="'/'"):
        save_params({"params": {"a/b": jnp.ones((2,))}}, tmp_path / "model.safetensors")


def test_missing_safetensors_names_the_extra(params, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "safetensors", None)
    with pytest.raises(ImportError, match=r"dew-ml\[interop\]"):
        save_params(params, tmp_path / "model.safetensors")


# ---------------------------------------------------------------------------------
# Hub push and pull
# ---------------------------------------------------------------------------------

class _RecordingApi:
    """Stands in for HfApi and keeps every call push_to_hub makes."""

    def __init__(self):
        self.created = []
        self.uploaded = []

    def create_repo(self, repo_id, **kwargs):
        self.created.append((repo_id, kwargs))

    def upload_folder(self, **kwargs):
        self.uploaded.append(kwargs)


@pytest.fixture
def api(monkeypatch):
    recording = _RecordingApi()
    monkeypatch.setattr(hub, "HfApi", lambda: recording)
    return recording


def test_push_creates_the_repo_and_uploads_the_export_directory(params, tmp_path, api):
    export = tmp_path / "export"
    save_hf_layout(params, {"architecture": "simple_dit"}, export)

    push_to_hub(export, "acme/dew-export")

    assert api.created == [("acme/dew-export", {"private": False, "exist_ok": True})]
    assert api.uploaded == [{
        "repo_id": "acme/dew-export",
        "folder_path": str(export),
        "commit_message": "Upload dew export",
    }]
    uploaded = Path(api.uploaded[0]["folder_path"])
    assert {entry.name for entry in uploaded.iterdir()} == {
        "model.safetensors", "config.json"}


def test_push_passes_the_private_flag_and_the_commit_message_through(params, tmp_path, api):
    export = tmp_path / "export"
    save_hf_layout(params, {"architecture": "simple_dit"}, export)

    push_to_hub(export, "acme/held-back", private=True, commit_message="step 1000")

    assert api.created == [("acme/held-back", {"private": True, "exist_ok": True})]
    assert api.uploaded[0]["commit_message"] == "step 1000"


def test_pull_returns_the_snapshot_directory(tmp_path, monkeypatch):
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)

    assert pull_from_hub("acme/dew-export", revision="v2") == tmp_path / "snapshot"
    assert pull_from_hub("acme/dew-export") == tmp_path / "snapshot"
    assert calls == [
        {"repo_id": "acme/dew-export", "revision": "v2"},
        {"repo_id": "acme/dew-export", "revision": None},
    ]
