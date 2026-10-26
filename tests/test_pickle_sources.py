"""PyTorch pickle checkpoints through a one-time safetensors conversion.

A `pytorch_model.bin`, or the shards its index names, is unpickled once with
torch, written as safetensors under Dew's cache (`$XDG_CACHE_HOME/dew`), and mapped from there; a
later load of the same files maps the cache and never imports torch. The
offline cases use the committed mamba2-tiny fixture pickled into a temporary
directory, so the whole load is checked against its transformers logits.
"""

import json
import multiprocessing
import os
import shutil
import sys
import tempfile
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
import pytest
import torch
from safetensors.numpy import load_file

from dew.interop import Pretrained, sources

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

    first = Pretrained.load(source, dtype="float32", attention_impl="reference")
    conversions = sorted((cache / "dew" / "converted").iterdir())
    monkeypatch.setitem(sys.modules, "torch", None)
    second = Pretrained.load(source, dtype="float32", attention_impl="reference")

    assert float(np.max(np.abs(logits(first) - reference))) < 1e-5
    assert len(conversions) == 1 and sorted((cache / "dew" / "converted").iterdir()) == conversions
    assert jax.tree.all(jax.tree.map(np.array_equal, first.variables, second.variables))


def _converting(root: str, rounds: int, barrier, failures) -> None:
    """One pool process: each round, once every process is at the barrier, convert and read that round's
    pickle (`sources.tensor_file`, as a release's readout loads), each failure to find its weights put on
    `failures`."""
    os.environ["XDG_CACHE_HOME"] = str(Path(root) / "cache")
    from dew.interop import sources

    for round_ in range(rounds):
        barrier.wait()
        try:
            sources.tensor_file(Path(root) / f"release-{round_}" / "readout_head.pt")
        except FileNotFoundError as failure:
            failures.put(f"{round_}: {failure}")


def test_processes_converting_one_pickle_at_once_each_read_its_weights():
    """A pool's processes loading one release's pickle at once each convert it and read the conversion,
    round after round, under a cache on the home directory's filesystem, where a pool's ~/.cache lies (an
    overlay in CI, whose listing of a folder another process was rewriting a file in found no weights).
    Each conversion is published whole, so every read finds them."""
    rounds, processes = 16, 8
    home = Path.home() / ".cache"
    home.mkdir(parents=True, exist_ok=True)
    root = tempfile.mkdtemp(dir=home, prefix="dew-pickle-race-")
    try:
        for round_ in range(rounds):
            release = Path(root) / f"release-{round_}"
            release.mkdir()
            torch.save({f"w{index}": torch.randn(64, 64) for index in range(8)}, release / "readout_head.pt")
        context = multiprocessing.get_context("spawn")
        barrier, failures = context.Barrier(processes), context.Queue()
        pool = [context.Process(target=_converting, args=(root, rounds, barrier, failures))
                for _ in range(processes)]
        for process in pool:
            process.start()
        for process in pool:
            process.join()
        assert [process.exitcode for process in pool] == [0] * processes
        found = []
        while not failures.empty():
            found.append(failures.get())
        assert not found, found[:3]
    finally:
        shutil.rmtree(root, ignore_errors=True)


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

    tensors = sources.load_shards(tmp_path)

    assert sorted(tensors) == ["model.embed_tokens.weight", "model.norm.weight"]
    assert tensors["model.embed_tokens.weight"].dtype == ml_dtypes.bfloat16
    assert np.array_equal(tensors["model.embed_tokens.weight"].astype(np.float32),
                          np.arange(12, dtype=np.float32).reshape(3, 4))
    assert np.array_equal(tensors["model.norm.weight"], np.arange(6, dtype=np.float32).reshape(2, 3).T)
