"""Weight sources beside or instead of safetensors, over a Hub on the loopback (tests/local_hub.py), which
keeps every request it answers. A commit of transformers' pickles alone loads SFconvertbot's open
safetensors conversion of that commit where one sits on it, fetching none of its pickles, and converts the
pickles where none does (`sources.snapshot`); the same load offline reads what the online one cached. A
directory holding GGUF files alone, other pickles alone or no weights is refused, naming what loads."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from flax.traverse_util import flatten_dict
from local_hub import LocalHub, Pull, hub_environment
from safetensors.numpy import load_file

from dew.interop import Pretrained

FIXTURE = Path(__file__).parent / "fixtures" / "hf" / "llama-tiny"
CONVERSION = "Adding `safetensors` variant of this model"
"""The title SFconvertbot's conversion pull requests carry (`sources._conversion_revision`)."""

LOAD = """
import sys
import numpy as np
from flax.traverse_util import flatten_dict
from dew.interop import Pretrained
loaded = Pretrained.load(sys.argv[1], dtype="float32", attention_impl="reference")
flat = flatten_dict(loaded.variables)
np.savez(sys.argv[2], **{"/".join(path): np.asarray(leaf) for path, leaf in flat.items()})
"""
"""A load in a process of its own, which reads `HF_ENDPOINT` as it imports the Hub client: its variables
written flat to the file named."""


def pickled(directory: Path) -> Path:
    """llama-tiny's tensors as transformers' pickle, `pytorch_model.bin`."""
    import torch

    directory.mkdir(parents=True)
    tensors = load_file(FIXTURE / "model.safetensors")
    torch.save({name: torch.from_numpy(array) for name, array in tensors.items()},
               directory / "pytorch_model.bin")
    return directory / "pytorch_model.bin"


def loaded(repo: str, environment: dict, out: Path) -> dict[str, np.ndarray]:
    """`repo` loaded in its own process under `environment`, its variables flat by path."""
    done = subprocess.run([sys.executable, "-c", LOAD, repo, str(out)], env=environment, capture_output=True,
                          text=True, check=False)
    assert done.returncode == 0, done.stderr[-3000:]
    with np.load(out) as stored:
        return {name: stored[name] for name in stored.files}


def assert_the_fixture(found: dict[str, np.ndarray]) -> None:
    """`found` holds llama-tiny's variables, leaf for leaf, as its safetensors load them."""
    held = {"/".join(path): np.asarray(leaf)
            for path, leaf in flatten_dict(Pretrained.load(FIXTURE, dtype="float32").variables).items()}
    assert found.keys() == held.keys()
    for name, leaf in held.items():
        np.testing.assert_array_equal(found[name], leaf, err_msg=name)


def test_a_pickle_commit_loads_sfconvertbots_conversion_of_it_and_offline_what_it_cached(tmp_path):
    """A commit holding transformers' pickles alone, with SFconvertbot's open conversion pull request on
    it, loads the conversion's safetensors: llama-tiny leaf for leaf, and not one byte of the pickle
    fetched. Offline, with the Hub gone, the same load reads the conversion its answers recorded."""
    files = {"config.json": FIXTURE / "config.json", "pytorch_model.bin": pickled(tmp_path / "pickles")}
    # The conversion adds the safetensors to the commit's tree, its pickle kept beside them.
    conversion = Pull(CONVERSION, "SFconvertbot",
                      {**files, "model.safetensors": FIXTURE / "model.safetensors"})
    with LocalHub({"local/pickled": files}, {"local/pickled": {1: conversion}}) as hub:
        assert_the_fixture(loaded("local/pickled", hub_environment(hub, tmp_path), tmp_path / "online.npz"))
        assert hub.sent("local/pickled", "pytorch_model.bin") == []
        assert hub.sent("local/pickled", "model.safetensors")
    offline = hub_environment(None, tmp_path, offline=True)
    assert_the_fixture(loaded("local/pickled", offline, tmp_path / "offline.npz"))


@pytest.mark.parametrize("pull", [None, Pull(CONVERSION, "someone", {}),
                                  Pull("Update README", "SFconvertbot", {})],
                         ids=["no pull request", "another author", "another title"])
def test_a_pickle_commit_without_a_conversion_converts_its_pickles(pull, tmp_path):
    """A commit of pickles alone with no SFconvertbot conversion on it (none open, one by another author,
    one of another title) fetches its pickle and loads its conversion: llama-tiny leaf for leaf."""
    files = {"config.json": FIXTURE / "config.json", "pytorch_model.bin": pickled(tmp_path / "pickles")}
    with LocalHub({"local/pickled": files}, None if pull is None else {"local/pickled": {1: pull}}) as hub:
        assert_the_fixture(loaded("local/pickled", hub_environment(hub, tmp_path), tmp_path / "loaded.npz"))
        assert hub.sent("local/pickled", "pytorch_model.bin")


@pytest.mark.parametrize("stored", [{"model.gguf": b"GGUF"}, {"weights.pt": b"\x80"}, {}],
                         ids=["GGUF alone", "other pickles", "no weights"])
def test_a_directory_without_weights_dew_reads_is_refused(stored, tmp_path):
    """A checkpoint directory holding GGUF files alone (which load through `gguf_file`), pickles under
    names transformers does not save, or no weights is refused before any model is built."""
    (tmp_path / "config.json").write_text((FIXTURE / "config.json").read_text())
    for name, data in stored.items():
        (tmp_path / name).write_bytes(data)
    with pytest.raises(FileNotFoundError):
        Pretrained.load(tmp_path, dtype="float32")


TRAIN_PRETRAINED = """
import sys
from dew.config.lm import LMRunConfig
from dew.data import Loading, presets
from dew.data.sources.tokens import TokenFiles

tokens, root = sys.argv[1:3]
chain = TokenFiles(tokens).pipe(presets.lm, None, seq_len=8, batch=8, pack=None).loading(Loading(workers=0))
LMRunConfig.cli(["--pretrained", "local/llama", "--trainer.name", "run", "--trainer.checkpoint-dir", root,
                 "--trainer.steps", "1", "--trainer.compilation-cache-dir", "None", "--trainer.multi-host",
                 "False", "--trainer.eval-every", "None", "--trainer.checkpoint-every", "1",
                 "--max-new-tokens", "0"], default=LMRunConfig(data=chain)).run()
"""
"""A recorded LM run from a Hub backbone, through the command line, in a process of its own, which reads
`HF_ENDPOINT` as it imports the Hub client: the token files and the runs' root."""

PINNED_DECISION = """
from dew.decision.config import DecisionRunConfig

print(DecisionRunConfig(pretrained="local/llama").pinned().revision)
"""
"""A decision run's pinned backbone revision, in a process of its own."""


def test_a_run_records_its_hub_backbone_at_its_commit_online_and_from_the_cache_offline(tmp_path):
    """An LM run whose `--pretrained` names a Hub repo alone, run from the command line, writes the
    commit the Hub serves into its run.json; with the Hub gone and none of Dew's recorded answers, the
    same command reads the Hub cache's ref the online run wrote and records the same commit. A decision
    run pins its backbone's revision to that commit alike."""
    import json

    from token_support import write_token_corpus

    ids = np.random.RandomState(0).randint(1, 60, 640).astype(np.uint8)
    # Written with the backbone's own tokenizer, by its name: the checkpoint carries no tokenizer files.
    tokens = write_token_corpus(tmp_path / "tokens", ids, tokenizer="local/llama")
    files = {name: FIXTURE / name for name in ("config.json", "model.safetensors")}

    def ran(name, environment):
        root = tmp_path / name
        done = subprocess.run([sys.executable, "-c", TRAIN_PRETRAINED, tokens.train, str(root)],
                              env=environment, capture_output=True, text=True, check=False)
        assert done.returncode == 0, done.stderr[-3000:]
        return json.loads((root / "run" / "run.json").read_text())["fields"]["pretrained"]

    with LocalHub({"local/llama": files}) as hub:
        commit = hub.commit("local/llama")
        assert ran("online", hub_environment(hub, tmp_path)) == f"local/llama@{commit}"
        decided = subprocess.run([sys.executable, "-c", PINNED_DECISION], env=hub_environment(hub, tmp_path),
                                 capture_output=True, text=True, check=False)
    assert decided.stdout.strip().splitlines()[-1] == commit
    gone = {**hub_environment(None, tmp_path, offline=True), "DEW_HUB_ANSWERS": str(tmp_path / "no-answers")}
    assert ran("offline", gone) == f"local/llama@{commit}"
