"""recipes/lm/train.py: what it refuses, and a run over real token files."""

import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
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
        recipe.token_directory(str(tmp_path))
    with pytest.raises(ValueError, match="--data.path"):
        recipe.token_directory(None)


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


def export_tiny_decoder(directory, *, tokenizer="byte", vocab_size=256):
    """A local HF-layout decoder, the way a --pretrained run is pointed at one.

    Exported through save_pretrained_decoder, so the directory has the shape
    a hub repo has: config.json, model.safetensors and the generation_config
    that records which tokenizer its ids come from.
    """
    from dew.interop.hf_decoders import save_pretrained_decoder
    from dew.registry import models

    model = models.build("causal_transformer", vocab_size=vocab_size, emb_features=16,
                         num_layers=1, num_heads=2, num_kv_heads=1, mlp_features=32,
                         max_seq_len=SEQ, tie_embeddings=False)
    variables = model.init(jax.random.key(0), jnp.ones((1, 4), jnp.int32))
    save_pretrained_decoder(model, variables, str(directory), tokenizer_name=tokenizer)
    return directory


def recipe_args(tokens, *args, model_config="{}"):
    return [*args, "--data.path", str(tokens), "--data.seq-len", str(SEQ),
            "--data.loading.workers", "0", "--trainer.batch-size", "8",
            "--trainer.checkpoint-dir", str(tokens.parent / "runs"),
            "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
            "--trainer.log-every", "1", "--model.dtype", "float32",
            "--model.config", model_config]


def pretrained_config(recipe, tokens, pretrained, *args, model_config="{}"):
    return tyro.cli(tyro.conf.CascadeSubcommandArgs[recipe.LmRunConfig], args=recipe_args(
        tokens, "--pretrained", str(pretrained), *args, model_config=model_config))


def test_the_recipe_continues_a_pretrained_decoder(tmp_path):
    """The path a --pretrained user runs, end to end: a local HF-layout
    checkpoint through load_pretrained_decoder, the tokenizer of the token
    files checked against the one the checkpoint records, a step taken on
    the loaded weights and the run spec written back.

    The checkpoint decides the architecture, so what the run builds is its
    one layer of width 16 rather than the recipe's defaults.
    """
    recipe = load_recipe()
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    checkpoint = export_tiny_decoder(tmp_path / "checkpoint")
    config = pretrained_config(recipe, tokens, checkpoint, "--trainer.steps", "1",
                               "--sample-tokens", "0", "--trainer.name", "continued")

    state = recipe.main(config)

    assert int(state.step) == 1
    assert recipe.LmRunConfig.load(str(tmp_path / "runs" / "continued")) == config
    kernel = state.params["params"]["layers_0"]["self_attn"]["q_proj"]["kernel"]
    assert kernel.shape == (16, 16)
    assert np.all(np.isfinite(np.asarray(kernel)))


def test_a_pretrained_run_starts_from_the_checkpoints_weights(tmp_path):
    """Zero steps hold exactly what the checkpoint carries, which is what
    makes the load a continuation rather than a fresh init of the same
    shape."""
    from dew.interop.hf_decoders import load_pretrained_decoder

    recipe = load_recipe()
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    checkpoint = export_tiny_decoder(tmp_path / "checkpoint")
    config = pretrained_config(recipe, tokens, checkpoint, "--trainer.steps", "0",
                               "--sample-tokens", "0", "--trainer.name", "zero")

    state = recipe.main(config)

    _, expected, _ = load_pretrained_decoder(str(checkpoint), dtype="float32",
                                             attention_impl="reference")
    for path, leaf in jax.tree_util.tree_flatten_with_path(expected["params"])[0]:
        held = state.params["params"]
        for entry in path:
            held = held[entry.key]
        np.testing.assert_array_equal(np.asarray(held), np.asarray(leaf))


def test_a_pretrained_run_refuses_overrides_and_a_foreign_tokenizer(tmp_path):
    """The two refusals on that path: the checkpoint owns every architecture
    field but max_seq_len, and ids from another vocabulary would train the
    embedding table against noise."""
    recipe = load_recipe()
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    checkpoint = export_tiny_decoder(tmp_path / "checkpoint")

    with pytest.raises(ValueError, match="which the checkpoint at"):
        recipe.main(pretrained_config(
            recipe, tokens, checkpoint, "--trainer.steps", "1",
            model_config='{"emb_features": 32}'))

    foreign = export_tiny_decoder(tmp_path / "foreign", tokenizer="gpt2")
    with pytest.raises(ValueError, match="expects gpt2"):
        recipe.main(pretrained_config(recipe, tokens, foreign, "--trainer.steps", "1"))


def test_a_pretrained_run_refuses_a_checkpoint_too_narrow_for_the_ids(tmp_path):
    recipe = load_recipe()
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    narrow = export_tiny_decoder(tmp_path / "narrow", vocab_size=128)

    with pytest.raises(ValueError, match="has room for 128 ids"):
        recipe.main(pretrained_config(recipe, tokens, narrow, "--trainer.steps", "1"))


def test_the_recipe_balances_a_sparse_run(tmp_path):
    """--balance-rate reaches the objective: a sparse run moves every
    router's bias by the rate each step, which the recipe could not ask for
    before, and an unbalanced run leaves it at zero."""
    recipe = load_recipe()
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    sparse = ('{"emb_features": 16, "num_layers": 2, "num_heads": 2, '
              '"mixture": {"experts": 8, "top_k": 2, "layers": [1], "bias": true}}')

    def run(name, *extra):
        config = tyro.cli(tyro.conf.CascadeSubcommandArgs[recipe.LmRunConfig],
                          args=recipe_args(tokens, "--trainer.steps", "2",
                                           "--sample-tokens", "0",
                                           "--trainer.name", name, *extra,
                                           model_config=sparse))
        state = recipe.main(config)
        return np.asarray(state.params["moe"]["layers_1"]["mlp"]["gate"]
                          ["e_score_correction_bias"])

    balanced = run("balanced", "--balance-rate", "0.01")
    assert np.any(balanced != 0), "the bias never moved"
    np.testing.assert_allclose(np.abs(balanced) / 0.01,
                               np.round(np.abs(balanced) / 0.01), atol=1e-4)
    assert np.all(run("unbalanced") == 0)
