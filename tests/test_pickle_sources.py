"""PyTorch pickle checkpoints through a one-time safetensors conversion.

A `pytorch_model.bin`, or the shards its index names, is unpickled once with
torch, written as safetensors under Dew's cache (`$XDG_CACHE_HOME/dew`), and mapped from there; a
later load of the same files maps the cache and never imports torch. The
offline cases use the committed mamba2-tiny fixture pickled into a temporary
directory, so the whole load is checked against its transformers logits.
"""

import json
import shutil
import sys
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
import pytest
from safetensors.numpy import load_file

from dew.interop import hf_decoders, load_pretrained

torch = pytest.importorskip("torch")

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hf" / "mamba2-tiny"


@pytest.fixture
def cache(monkeypatch, tmp_path):
    root = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(root))
    return root


def pickled_fixture(directory: Path) -> Path:
    """The mamba2-tiny fixture as transformers' pickle layout: config.json
    beside one pytorch_model.bin."""
    directory.mkdir()
    shutil.copyfile(FIXTURE / "config.json", directory / "config.json")
    tensors = load_file(FIXTURE / "model.safetensors")
    torch.save({name: torch.from_numpy(array) for name, array in tensors.items()},
               directory / "pytorch_model.bin")
    return directory


def logits(source) -> np.ndarray:
    return np.asarray(source.model.apply(source.variables, np.load(FIXTURE / "input_ids.npy")))


def test_a_pickle_checkpoint_converts_once_and_then_loads_without_torch(tmp_path, cache, monkeypatch):
    """The first load converts and reproduces the reference logits within
    the fixture's own fp32 bound (tests/test_mamba2_interop.py); the second
    maps the same conversion with torch unimportable."""
    source = pickled_fixture(tmp_path / "source")
    reference = np.load(FIXTURE / "logits.npy")

    first = load_pretrained(source, dtype="float32", attention_impl="reference")
    conversions = sorted((cache / "dew" / "converted").iterdir())
    monkeypatch.setitem(sys.modules, "torch", None)
    second = load_pretrained(source, dtype="float32", attention_impl="reference")

    assert float(np.max(np.abs(logits(first) - reference))) < 1e-5
    assert len(conversions) == 1 and sorted((cache / "dew" / "converted").iterdir()) == conversions
    assert jax.tree.all(jax.tree.map(np.array_equal, first.variables, second.variables))


def test_sharded_pickles_read_through_their_index_in_their_stored_dtype(tmp_path, cache):
    """transformers' sharded layout, a bfloat16 tensor and a transposed view
    (a pickle keeps a view's strides): every name the index maps, with its
    dtype and values, and nothing from a file the index does not name."""
    embedding = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    kernel = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.save({"model.embed_tokens.weight": embedding}, tmp_path / "pytorch_model-00001-of-00002.bin")
    torch.save({"model.norm.weight": kernel.T}, tmp_path / "pytorch_model-00002-of-00002.bin")
    torch.save({"stale.weight": kernel}, tmp_path / "pytorch_model.bin")
    (tmp_path / "pytorch_model.bin.index.json").write_text(json.dumps({"weight_map": {
        "model.embed_tokens.weight": "pytorch_model-00001-of-00002.bin",
        "model.norm.weight": "pytorch_model-00002-of-00002.bin"}}))

    tensors = hf_decoders._load_shards(tmp_path)

    assert sorted(tensors) == ["model.embed_tokens.weight", "model.norm.weight"]
    assert tensors["model.embed_tokens.weight"].dtype == ml_dtypes.bfloat16
    assert np.array_equal(tensors["model.embed_tokens.weight"].astype(np.float32),
                          np.arange(12, dtype=np.float32).reshape(3, 4))
    assert np.array_equal(tensors["model.norm.weight"], np.arange(6, dtype=np.float32).reshape(2, 3).T)


def test_without_torch_a_pickle_checkpoint_is_refused_naming_the_extra(tmp_path, cache, monkeypatch):
    source = pickled_fixture(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(ImportError, match=r"dew-ml\[torch\].*huggingface\.co/spaces/safetensors/convert"):
        load_pretrained(source)



@pytest.mark.network
def test_mamba2_130ms_pickle_converts_to_its_safetensors_conversions_weights(cache, monkeypatch):
    """Both routes on state-spaces/mamba2-130m's main commit: SFconvertbot's
    refs/pr/1, then the pytorch_model.bin itself, with the pull request
    lookup made to find nothing. The same fp32 weights, bit for bit."""
    commit = "3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49"
    pull = load_pretrained("state-spaces/mamba2-130m", revision=commit, dtype="float32")
    monkeypatch.setattr(hf_decoders, "_conversion_revision", lambda name, commit: None)
    pickled = load_pretrained("state-spaces/mamba2-130m", revision=commit, dtype="float32")

    assert (pull.revision, pickled.revision) == ("ea6060f68a4289e9c06f80effa896629ba519216", commit)
    assert jax.tree.all(jax.tree.map(np.array_equal, pull.variables, pickled.variables))
