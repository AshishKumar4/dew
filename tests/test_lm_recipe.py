"""recipes/lm/train.py: what it refuses, and a run over real token files."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tyro

from dew.data import PackedTokens, TokenWindows
from dew.objectives.lm import Samples

pytestmark = pytest.mark.mesh

REPO_ROOT = Path(__file__).resolve().parents[1]
SEQ = 32


def load_recipe():
    path = REPO_ROOT / "recipes" / "lm" / "train.py"
    spec = importlib.util.spec_from_file_location("recipe_lm", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_token_files(root, train_tokens, val_tokens, tokenizer="byte", eos_id=None):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for name, count in (("train.bin", train_tokens), ("val.bin", val_tokens)):
        tokens = rng.randint(1, 250, count).astype(np.uint8)
        if eos_id is not None:
            tokens[::7] = eos_id
        (root / name).write_bytes(tokens.tobytes())
    (root / "meta.json").write_text(json.dumps(
        {"tokenizer": tokenizer, "vocab_size": 256, "dtype": "uint8", "eos_id": eos_id}))
    return root


def run_config(recipe, tokens, *args):
    # A dataset subcommand has to come before its flags, so `args` leads.
    return tyro.cli(tyro.conf.CascadeSubcommandArgs[recipe.LmRunConfig], args=[
        *args, "--data.path", str(tokens), "--data.seq-len", str(SEQ), "--data.loading.workers", "0",
        "--trainer.batch-size", "8", "--trainer.checkpoint-dir", str(tokens.parent / "runs"),
        "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
        "--trainer.log-every", "1", "--model.dtype", "float32",
        "--model.config", '{"emb_features": 16, "num_layers": 1, "num_heads": 2}'])


def test_the_sampling_budget_decides_the_context_the_model_is_built_for():
    recipe = load_recipe()
    config = recipe.LmRunConfig(data=TokenWindows(seq_len=64))
    assert recipe.context_length(config, None) == 64
    assert recipe.context_length(config, Samples([1, 2, 3], 8)) == 64
    assert recipe.context_length(config, Samples([1, 2, 3], 100)) == 103


def test_a_dataset_that_is_not_a_token_directory_says_so(tmp_path):
    recipe = load_recipe()
    with pytest.raises(FileNotFoundError, match="meta.json"):
        recipe.read_meta(str(tmp_path))
    with pytest.raises(ValueError, match="--data.path"):
        recipe.read_meta(None)


def test_a_tokenizer_that_does_not_match_the_token_files_is_rejected(tmp_path):
    recipe = load_recipe()
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ)
    with pytest.raises(ValueError, match="written with byte"):
        recipe.main(run_config(recipe, tokens, "--tokenizer", "gpt2", "--trainer.steps", "1"))


def test_a_corpus_too_small_for_one_batch_is_refused(tmp_path):
    recipe = load_recipe()
    tokens = write_token_files(tmp_path / "tokens", 2 * SEQ, 2 * SEQ)
    with pytest.raises(ValueError, match="do not fill one batch"):
        recipe.main(run_config(recipe, tokens, "--trainer.epochs", "1"))


@pytest.mark.parametrize("packed", [False, True])
def test_the_recipe_trains_on_tokenized_files(tmp_path, packed):
    """A run from the command line: the windows or packed documents through
    the trainer, perplexity scored on val.bin, the run spec and a checkpoint
    at the final step."""
    recipe = load_recipe()
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    args = ["--trainer.epochs", "1", "--sample-prompt", "the ", "--sample-tokens", "4",
            "--trainer.name", "run"]
    if packed:
        args = ["data:packed-tokens", "--data.packing-bins", "2", *args]
    config = run_config(recipe, tokens, *args)
    assert isinstance(config.data, PackedTokens if packed else TokenWindows)

    state = recipe.main(config)

    data = config.data.load(batch=8)
    assert int(state.step) == data.steps_per_epoch > 0
    assert recipe.LmRunConfig.load(str(tmp_path / "runs" / "run")) == config
    assert (tmp_path / "runs" / "run" / str(int(state.step))).is_dir()
