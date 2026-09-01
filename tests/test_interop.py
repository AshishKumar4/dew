"""safetensors round-trips for Flax parameter trees.

The tree is nested and the file is flat, so what these tests hold onto is the
'/'-joined naming: nesting, values and dtypes have to survive a save and a
load, the names on disk have to be readable by a safetensors reader that knows
nothing about dew, and a missing optional dependency has to say so.
"""

import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop import load_params, save_hf_layout, save_params
from dew.nn.backbones.dit import SimpleDiT

safetensors_numpy = pytest.importorskip("safetensors.numpy")


@pytest.fixture
def params(rng):
    model = SimpleDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2, mlp_ratio=1)
    x = jax.random.normal(rng, (1, 8, 8, 3))
    return model.init(rng, x, jnp.ones((1,)), jnp.ones((1, 77, 768)))


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
