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
# The committed byte-level BPE a run can train with offline.
TOKENIZER = REPO_ROOT / "tests" / "fixtures" / "tokenizers" / "tiny-tools"


def load_recipe():
    path = REPO_ROOT / "recipes" / "lm" / "train.py"
    spec = importlib.util.spec_from_file_location("recipe_lm", path)
    assert spec is not None and spec.loader is not None
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
    assert data.steps_per_epoch is not None and int(state.step) == data.steps_per_epoch > 0
    assert recipe.LmRunConfig.load(str(tmp_path / "runs" / "run")) == config
    assert (tmp_path / "runs" / "run" / str(int(state.step))).is_dir()


def test_the_recipe_trains_muonclip_with_the_clip_firing(tmp_path):
    """`--optim.optimizer muonclip` through `recipe.main`: the per-head maxima
    travel from the loss to the optimizer inside the compiled step, so the
    query kernel lands away from a Muon run at the same seed. Observed on
    CPU: 4 steps, kernels differ by 0.48."""
    recipe = load_recipe()
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)

    def run(name, *args):
        config = run_config(recipe, tokens, "--trainer.name", name,
                            "--trainer.epochs", "1", *args)
        return recipe.main(config)

    muon = run("muon", "--optim.optimizer", "muon")
    clipped = run("clip", "--optim.optimizer", "muonclip",
                  "--optim.optimizer-opts", '{"qk_clip_threshold": 1.0}')

    def q_kernel(state):
        return np.asarray(
            state.params["params"]["layers_0"]["self_attn"]["q_proj"]["kernel"])

    assert int(clipped.step) == int(muon.step) > 0
    assert bool(jnp.all(jnp.isfinite(q_kernel(clipped))))
    assert float(np.max(np.abs(q_kernel(clipped) - q_kernel(muon)))) > 1e-6


def test_the_recipe_trains_a_quantized_trunk(tmp_path):
    """`quantization:quantization --quantization.dtype int8` through
    `recipe.main`: the run completes to finite weights and the record
    round-trips the value. Observed on CPU: 4 steps, all leaves finite."""
    pytest.importorskip("qwix")
    recipe = load_recipe()
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    config = run_config(recipe, tokens, "--trainer.name", "quant",
                        "--trainer.epochs", "1", "quantization:quantization",
                        "--quantization.dtype", "int8")
    state = recipe.main(config)
    assert int(state.step) > 0
    assert all(bool(jnp.all(jnp.isfinite(leaf)))
               for leaf in jax.tree.leaves(state.params["params"]))
    assert recipe.LmRunConfig.load(str(tmp_path / "runs" / "quant")) == config


def export_tiny_decoder(directory, *, tokenizer="byte", vocab_size=256):
    """A local HF-layout decoder, the way a --pretrained run is pointed at one.

    Exported through save_pretrained_decoder, so the directory has the shape
    a hub repo has: config.json, model.safetensors, the generation_config
    that records which tokenizer its ids come from, and that tokenizer's own
    files whenever it is one that has any.
    """
    from dew.interop.hf_decoders import save_pretrained_decoder
    from dew.registry import models

    model = models.build("causal_transformer", vocab_size=vocab_size, emb_features=16,
                         num_layers=1, num_heads=2, num_kv_heads=1, mlp_features=32,
                         max_seq_len=SEQ, tie_embeddings=False)
    variables = model.init(jax.random.key(0), jnp.ones((1, 4), jnp.int32))
    save_pretrained_decoder(model, variables, str(directory), tokenizer=tokenizer)
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
    checkpoint through load_pretrained, the tokenizer of the token
    files checked against the one the checkpoint records, a step taken on
    the loaded weights and the run spec written back.

    The checkpoint decides the architecture, so what the run builds is its
    one layer of width 16, not the recipe's defaults.
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
    """Zero steps hold what the checkpoint carries, leaf for leaf: the load
    is a continuation, not a fresh init of the same shape."""
    from dew.interop import load_pretrained

    recipe = load_recipe()
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    checkpoint = export_tiny_decoder(tmp_path / "checkpoint")
    config = pretrained_config(recipe, tokens, checkpoint, "--trainer.steps", "0",
                               "--sample-tokens", "0", "--trainer.name", "zero")

    state = recipe.main(config)

    expected = load_pretrained(str(checkpoint), dtype="float32",
                               attention_impl="reference").variables
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

    # A name an export can resolve offline, since it now writes the assets of
    # the tokenizer it names rather than only recording the name.
    foreign = export_tiny_decoder(tmp_path / "foreign", tokenizer=str(TOKENIZER),
                                  vocab_size=384)
    with pytest.raises(ValueError, match="expects .*tiny-tools"):
        recipe.main(pretrained_config(recipe, tokens, foreign, "--trainer.steps", "1"))


def test_a_trained_export_round_trips_with_its_tokenizer(tmp_path):
    """The artifact a --pretrained run leaves behind, taken all the way back.

    The checkpoint is exported with the tokenizer its ids came from, the
    recipe continues it for a step, `Pretrained.save` writes the trained
    weights back out, and the second directory is loaded again: it carries
    the same tokenizer files, so the load hands back a processor and a text
    prompt generates text without the caller naming a tokenizer anywhere.
    An export that only recorded a `tokenizer_name` lands here as a
    `processor` of None, which is also what leaves a directory that
    `ollama create` and llama.cpp's converter refuse.
    """
    from dew.data import tokenizer_for
    from dew.interop import load_pretrained

    recipe = load_recipe()
    tokenizer = tokenizer_for(str(TOKENIZER), local_files_only=True)
    ids = np.asarray(tokenizer.encode((REPO_ROOT / "CONTRIBUTING.md").read_text()), np.uint16)
    tokens = tmp_path / "tokens"
    tokens.mkdir(parents=True)
    (tokens / "train.bin").write_bytes(ids[:40 * SEQ].tobytes())
    (tokens / "val.bin").write_bytes(ids[40 * SEQ:48 * SEQ].tobytes())
    (tokens / "meta.json").write_text(json.dumps(
        {"tokenizer": str(TOKENIZER), "vocab_size": tokenizer.vocab_size,
         "dtype": "uint16", "eos_id": None}))
    checkpoint = export_tiny_decoder(tmp_path / "checkpoint", tokenizer=str(TOKENIZER),
                                     vocab_size=tokenizer.vocab_size)

    state = recipe.main(pretrained_config(
        recipe, tokens, checkpoint, "--trainer.steps", "1", "--sample-tokens", "0",
        "--tokenizer", str(TOKENIZER), "--trainer.name", "exported"))

    trained = tmp_path / "trained"
    load_pretrained(str(checkpoint), dtype="float32", attention_impl="reference").save(
        trained, variables=state.params)
    again = load_pretrained(str(trained), dtype="float32", attention_impl="reference")

    assert again.processor is not None, "the saved export carries no tokenizer"
    generated = again.text_generation()("The trainer", 4, key=jax.random.key(0))
    assert int(generated.lengths[0]) == 4
    assert again.processor.decode(generated.tokens)[0].startswith("The trainer")


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


def test_the_recipe_trains_the_prediction_depths_on_request(tmp_path):
    """--mtp-weight reaches the objective: with the term on, a prediction
    depth's fused projection ends two steps somewhere else than the same run
    without it, whose depth sees no gradient; and the flag on a model
    without depths raises a ValueError naming num_nextn_predict_layers."""
    recipe = load_recipe()
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    deep = ('{"emb_features": 16, "num_layers": 1, "num_heads": 2, '
            '"num_nextn_predict_layers": 1}')

    def run(name, *extra, model_config=deep):
        config = tyro.cli(tyro.conf.CascadeSubcommandArgs[recipe.LmRunConfig],
                          args=recipe_args(tokens, "--trainer.steps", "2",
                                           "--sample-tokens", "0",
                                           "--trainer.name", name, *extra,
                                           model_config=model_config))
        state = recipe.main(config)
        return np.asarray(state.params["params"]["mtp_0"]["eh_proj"]["kernel"])

    assert np.any(run("mtp", "--mtp-weight", "0.3") != run("plain")), \
        "the term never reached the depth"
    with pytest.raises(ValueError, match="num_nextn_predict_layers"):
        run("dense", "--mtp-weight", "0.3",
            model_config='{"emb_features": 16, "num_layers": 1, "num_heads": 2}')


def test_an_unknown_objective_is_refused():
    recipe = load_recipe()
    with pytest.raises(ValueError, match="--objective"):
        recipe.LmRunConfig(data=TokenWindows(seq_len=64), objective="ctc")


def test_the_masked_objective_is_reachable_by_name(tmp_path):
    recipe = load_recipe()
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ)
    assert run_config(recipe, tokens, "--objective", "masked_diffusion").objective == \
        "masked_diffusion"


def test_masked_diffusion_without_a_mask_id_is_refused():
    """The builder reads the id from the run's fields. A run without one
    raises naming it; id zero would corrupt the wrong token."""
    from dew.registry import models

    recipe = load_recipe()
    model = models.build("causal_transformer", vocab_size=256, emb_features=16,
                         num_layers=1, num_heads=2, max_seq_len=SEQ, causal=False)
    with pytest.raises(ValueError, match="mask token id"):
        recipe.build_masked_objective(
            recipe.LmRunConfig(data=TokenWindows(seq_len=SEQ),
                               objective="masked_diffusion"),
            model, {})


def test_masked_diffusion_on_a_causal_model_is_refused():
    """The objective reads the whole corrupted row, so a causal model names
    the flag it needs."""
    from dew.registry import models

    recipe = load_recipe()
    model = models.build("causal_transformer", vocab_size=256, emb_features=16,
                         num_layers=1, num_heads=2, max_seq_len=SEQ)
    with pytest.raises(ValueError, match="causal=False"):
        recipe.build_masked_objective(
            recipe.LmRunConfig(data=TokenWindows(seq_len=SEQ),
                               objective="masked_diffusion"),
            model, {"mask_token_id": 5})


def test_official_block_diffusion_is_a_complete_pretrained_recipe(tmp_path):
    from dew.interop import load_pretrained

    recipe = load_recipe()
    checkpoint = REPO_ROOT / "tests/fixtures/hf/diffusion-gemma-sft"
    directory = tmp_path / "tokens"
    directory.mkdir()
    for split, count in (("train", 160), ("val", 32)):
        ids = np.random.RandomState(count).randint(4, 32, count).astype(np.uint8)
        (directory / f"{split}.bin").write_bytes(ids.tobytes())
    (directory / "meta.json").write_text(json.dumps(
        {"tokenizer": str(checkpoint), "vocab_size": 32, "dtype": "uint8", "eos_id": 1}))
    config = tyro.cli(tyro.conf.CascadeSubcommandArgs[recipe.LmRunConfig], args=[
        "--pretrained", str(checkpoint), "--objective", "block_diffusion",
        "--tokenizer", str(checkpoint), "--data.path", str(directory), "--data.seq-len", "11",
        "--block-prompt-tokens", "4", "--data.loading.workers", "0",
        "--model.dtype", "float32", "--model.attention-impl", "xla",
        "--trainer.batch-size", "8", "--trainer.steps", "1", "--trainer.log-every", "1",
        "--trainer.checkpoint-dir", str(tmp_path / "runs"), "--trainer.name", "block",
        "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
        "--ema-decay", "None", "--sample-tokens", "0", "--optim.learning-rate", "0.001"])
    state = recipe.main(config)
    assert int(state.updates) == 1
    original = load_pretrained(checkpoint, dtype="float32", attention_impl="xla")
    initial = recipe.build_block_objective(config, original.model, original.variables).init(jax.random.key(0))
    difference = max(float(jnp.max(jnp.abs(a - b)))
                     for a, b in zip(jax.tree.leaves(state.params), jax.tree.leaves(initial)))
    assert difference > 1e-5
    restored = recipe.main(config)
    for wanted, actual in zip(jax.tree.leaves(state.params), jax.tree.leaves(restored.params)):
        np.testing.assert_array_equal(actual, wanted)
