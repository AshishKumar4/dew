"""recipes/lm/train.py: what it refuses, and a run over real token files."""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from token_support import write_token_corpus
from tool_support import load

import dew
from dew.config.data import loader
from dew.data import Loading
from dew.data.dataset import DataPhase, Loader
from dew.inference import TextGeneration
from dew.objectives.lm import LMObjective, LMRunConfig
from dew.sampling import Sampling

pytestmark = pytest.mark.mesh

REPO_ROOT = Path(__file__).resolve().parents[1]
SEQ = 32


@pytest.mark.parametrize("steps", [7_250, 725_000])
def test_rigel_empty_phases_extend_the_preceding_mixture(steps):
    rigel = load("rigel", "recipes/lm")
    assert rigel.phases({"stem": "data/stem", "other": "data/other"}, steps) == (
        DataPhase("data/stem", round(515_000 * steps / rigel.STEPS)),
        DataPhase({"data/stem": 11.5, "data/other": 4.9}),
    )


def test_rigel_leading_empty_phases_take_the_first_available_mixture():
    rigel = load("rigel", "recipes/lm")
    assert rigel.phases({"other": "data/other"}, 7_250) == (DataPhase("data/other"),)
    assert rigel.phases({"stem": "data/stem"}, 7_250) == (DataPhase("data/stem"),)
    with pytest.raises(ValueError):
        rigel.phases({"unknown": "data/unknown"}, 7_250)


# The committed byte-level BPE a run can train with offline.
def write_token_files(root, train_tokens, val_tokens, tokenizer="byte", eos_id=None):
    rng = np.random.RandomState(0)
    splits = []
    for count in (train_tokens, val_tokens):
        tokens = rng.randint(1, 250, count).astype(np.uint8)
        if eos_id is not None:
            tokens[::7] = eos_id
        splits.append(tokens)
    return write_token_corpus(root, *splits, tokenizer=tokenizer, eos_id=eos_id)


def model_flags(record):
    """Give each constructor field its typed flag; nested records take one JSON value."""
    flags = []
    for key, value in json.loads(record).items():
        if isinstance(value, bool):
            flags.append(f"--model.{'' if value else 'no-'}{key}")
        else:
            flags.extend((f"--model.{key}", json.dumps(value)
                          if isinstance(value, (dict, list)) else str(value)))
    return flags


def run_config(tokens, *args, model='{"emb_features": 16, "num_layers": 1, "num_heads": 2}'):
    # A dataset subcommand has to come before its flags, so `args` leads.
    return LMRunConfig.cli([
        *args, "--data.source", str(tokens), "--data.seq-len", str(SEQ), "--loading.workers", "0",
        "--trainer.batch-size", "8", "--trainer.checkpoint-dir", str(tokens.parent / "runs"),
        "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
        "--trainer.log-every", "1", "--model.dtype", "float32", *model_flags(model)])


@pytest.mark.parametrize("packed", [False, True])
def test_the_recipe_trains_on_tokenized_files(tmp_path, packed):
    """A run from the command line: the windows or packed documents through
    the trainer, perplexity scored on val.bin, the run spec and a checkpoint
    at the final step."""
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    args = ["--trainer.epochs", "1", "--prompt", "the ", "--max-new-tokens", "4",
            "--trainer.name", "run"]
    if packed:
        args = ["--data.pack", "first-fit", "--data.packing-bins", "2", "--data-read", "memory", *args]
    config = run_config(tokens, *args)
    assert config.data.pack == ('first-fit' if packed else None)

    state = config.run()

    data = loader(config.data, batch=8, read="memory", loading=Loading(workers=0))
    assert data.steps_per_epoch is not None and int(state.step) == data.steps_per_epoch > 0
    run = tmp_path / "runs" / "run"
    assert (run / str(int(state.step))).is_dir()
    # run.json is the resolved spec: the model as built, vocabulary and
    # context included, so the front door rebuilds it without the recipe.
    recorded = LMRunConfig.load(str(run))
    assert recorded.model.fields["vocab_size"] == 256 and recorded.model.fields["max_seq_len"] == SEQ
    assert dataclasses.replace(recorded, model=config.model) == config
    task = dew.pipeline(str(run))
    assert isinstance(task, TextGeneration) and task.max_new_tokens == 4
    drawn = task("the ", key=1, sampling=Sampling(temperature=0))
    assert drawn.host().tokens.shape == (1, len("the ") + 4) and len(drawn.text[0]) > 0
    np.testing.assert_array_equal(
        drawn.host().tokens,
        TextGeneration(task.model, state.variables)([list(b"the ")], 4, key=1,
                                                 sampling=Sampling(temperature=0)).host().tokens)
    objective = LMObjective(task.model, SEQ, samples=config.samples())
    trained = objective.pipeline(state, processor=task.processor)
    actual = task("the ", key=11).host()
    expected = trained("the ", key=11).host()
    np.testing.assert_array_equal(actual.tokens, expected.tokens)
    np.testing.assert_allclose(actual.behavior_log_probs, expected.behavior_log_probs, atol=1e-7, rtol=1e-7)


def test_the_recipe_trains_on_weighted_corpora(tmp_path):
    """`--data.source a 0.7 b 0.3` parses to the weighted mapping and trains:
    the vocabulary comes from the corpora's shared meta.json, and run.json
    records the mapping so the run reloads as configured."""
    first = write_token_files(tmp_path / "first", 40 * SEQ, 8 * SEQ, eos_id=0)
    second = write_token_files(tmp_path / "second", 24 * SEQ, 8 * SEQ, eos_id=0)
    config = LMRunConfig.cli([
        "--data.pack", "first-fit", "--data.source", json.dumps({str(first): 0.7, str(second): 0.3}),
        "--data.seq-len", str(SEQ), "--data.packing-bins", "2", "--data-read", "memory",
        "--loading.workers", "0",
        "--trainer.batch-size", "8", "--trainer.checkpoint-dir", str(tmp_path / "runs"),
        "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
        "--trainer.steps", "2", "--trainer.name", "mixed", "--max-new-tokens", "0",
        "--model.dtype", "float32",
        "--model.emb_features", "16", "--model.num_layers", "1", "--model.num_heads", "2"])
    assert config.data.source == {str(first): 0.7, str(second): 0.3}

    state = config.run()

    assert int(state.step) == 2
    recorded = LMRunConfig.load(str(tmp_path / "runs" / "mixed"))
    assert recorded.data.source == config.data.source
    assert recorded.model.fields["vocab_size"] == 256


def test_the_recipe_trains_muonclip_with_the_clip_firing(tmp_path):
    """`--optim.optimizer muonclip` through `run`: the per-head maxima
    travel from the loss to the optimizer inside the compiled step, so the
    query kernel lands away from a Muon run at the same seed. The model has
    no QK-norm, which the clip refuses, as docs/recipes.md says. Observed on
    CPU: 4 steps, kernels differ by 0.32."""
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)

    def run(name, *args):
        config = run_config(tokens, "--trainer.name", name, "--trainer.epochs", "1", *args,
                            model='{"emb_features": 16, "num_layers": 1, "num_heads": 2, "qk_norm": false}')
        return config.run()

    muon = run("muon", "--optim.optimizer", "muon")
    clipped = run("clip", "--optim.optimizer", "muonclip",
                  "--optim.optimizer-opts", '{"qk_clip_threshold": 1.0}')

    def q_kernel(state):
        return np.asarray(
            state.variables["params"]["layers_0"]["self_attn"]["q_proj"]["kernel"])

    assert int(clipped.step) == int(muon.step) > 0
    assert bool(jnp.all(jnp.isfinite(q_kernel(clipped))))
    assert float(np.max(np.abs(q_kernel(clipped) - q_kernel(muon)))) > 1e-6


def export_tiny_decoder(directory, *, tokenizer="byte", vocab_size=256):
    """A local HF-layout decoder, the way a --pretrained run is pointed at one.

    Exported through `PretrainedDecoder.from_model(...).save`, so the directory has the shape
    a hub repo has: config.json, model.safetensors, the generation_config
    that records which tokenizer its ids come from, and that tokenizer's own
    files whenever it is one that has any.
    """
    from dew.interop import PretrainedDecoder
    from dew.registry import models

    model = models.build("causal_transformer", vocab_size=vocab_size, emb_features=16,
                         num_layers=1, num_heads=2, num_kv_heads=1, mlp_features=32,
                         max_seq_len=SEQ, tie_embeddings=False)
    variables = model.init(jax.random.key(0), jnp.ones((1, 4), jnp.int32))
    PretrainedDecoder.from_model(model, variables, tokenizer=tokenizer).save(str(directory))
    return directory


def recipe_args(tokens, *args, model_config="{}"):
    return [*args, "--data.source", str(tokens), "--data.seq-len", str(SEQ),
            "--loading.workers", "0", "--trainer.batch-size", "8",
            "--trainer.checkpoint-dir", str(tokens.parent / "runs"),
            "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
            "--trainer.log-every", "1", "--model.dtype", "float32",
            *model_flags(model_config)]


def pretrained_config(tokens, pretrained, *args, model_config="{}"):
    return LMRunConfig.cli(recipe_args(tokens, "--pretrained", str(pretrained), *args,
                                              model_config=model_config))


def test_a_pretrained_run_starts_from_the_checkpoints_weights(tmp_path):
    """Zero steps hold what the checkpoint carries, leaf for leaf: the load
    is a continuation, not a fresh init of the same shape."""
    from dew.interop import Pretrained

    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    checkpoint = export_tiny_decoder(tmp_path / "checkpoint")
    config = pretrained_config(tokens, checkpoint, "--trainer.steps", "0",
                               "--max-new-tokens", "0", "--trainer.name", "zero")

    state = config.run()

    expected = Pretrained.load(str(checkpoint), dtype="float32",
                               attention_impl="reference").variables
    for path, leaf in jax.tree_util.tree_flatten_with_path(expected["params"])[0]:
        held = state.variables["params"]
        for entry in path:
            held = held[entry.key]
        np.testing.assert_array_equal(np.asarray(held), np.asarray(leaf))


@pytest.mark.parametrize("lora", [False, True], ids=["full", "lora"])
def test_a_pretrained_runs_record_trains_again(tmp_path, lora):
    """A fine-tune's own run.json goes through `dew train` a second time.

    The local checkpoint owns the geometry on both loads. Full fine-tuning
    and LoRA each train a step, then the recorded run resumes to two steps
    through the command's `--set trainer.steps=2`. The restored parameters
    move again, so the replay reaches training rather than only parsing.
    """
    from dew.checkpoints import Checkpoints
    from dew.cli.main import main

    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    checkpoint = export_tiny_decoder(tmp_path / "checkpoint")
    adapter = ("--lora.rank", "4", "--lora.modules", "q_proj") if lora else ()
    config = pretrained_config(tokens, checkpoint, "--trainer.steps", "1", "--max-new-tokens", "0",
                               "--trainer.name", "replayed", *adapter)
    state = config.run()
    run = tmp_path / "runs" / "replayed"

    assert main(["train", str(run / "run.json"), "--set", "trainer.steps=2"]) == 0

    saved = Checkpoints(str(run))
    assert saved.latest == 2
    after = saved.variables(step=2)["params"]
    assert any(not np.array_equal(np.asarray(before), np.asarray(again))
               for before, again in zip(jax.tree.leaves(state.variables["params"]),
                                         jax.tree.leaves(after), strict=True))


def test_a_pretrained_run_stores_its_parameters_in_the_dtype_it_names(tmp_path):
    """`--pretrained-param-dtype bfloat16` loads the checkpoint's parameters
    as bf16 and trains them so, and the run records it; with no
    `--pretrained` the field is refused rather than ignored."""
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    checkpoint = export_tiny_decoder(tmp_path / "checkpoint")
    config = pretrained_config(tokens, checkpoint, "--trainer.steps", "1", "--max-new-tokens", "0",
                               "--trainer.name", "bf16", "--pretrained-param-dtype", "bfloat16")
    state = config.run()
    params = jax.tree.leaves(state.variables["params"])
    floating = [leaf for leaf in params if jnp.issubdtype(leaf.dtype, jnp.floating)]
    assert floating and all(leaf.dtype == jnp.bfloat16 for leaf in floating)
    record = json.loads((tmp_path / "runs" / "bf16" / "run.json").read_text())["fields"]
    assert record["pretrained_param_dtype"] == "bfloat16"
    with pytest.raises(ValueError):
        LMRunConfig.cli(recipe_args(tokens, "--pretrained-param-dtype", "bfloat16"))


def test_a_pretrained_run_trains_a_lora_that_saves_from_the_run_alone(tmp_path):
    """`--pretrained <decoder> --lora.rank 4 --lora.modules q_proj`: the recipe
    binds the adapter to the loaded weights, two steps move every factor and
    leave the base bitwise, and `Pretrained.from_run` rebuilds the adapter
    from the run's own record, so its `adapter.save` writes the files the
    in-process adapter writes, byte for byte."""
    from dew.interop import Pretrained
    from dew.lora import LoRA
    from dew.objectives.base import FROZEN

    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    checkpoint = export_tiny_decoder(tmp_path / "checkpoint")
    lora = ("--lora.rank", "4", "--lora.modules", "q_proj")
    config = pretrained_config(tokens, checkpoint, "--trainer.steps", "2", "--max-new-tokens", "0",
                               "--trainer.name", "lora", *lora)
    assert config.lora == LoRA(rank=4, modules=("q_proj",))

    state = config.run()

    source = Pretrained.load(str(checkpoint), dtype="float32", attention_impl="reference")
    moved = jax.tree_util.tree_leaves_with_path(state.variables["params"])
    assert moved and all(path[-1].key in ("lora_A", "lora_B") for path, _ in moved)
    assert all(bool(jnp.any(leaf)) for path, leaf in moved if path[-1].key == "lora_B")
    for before, after in zip(jax.tree.leaves(source.variables["params"]),
                             jax.tree.leaves(state.variables[FROZEN]), strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    tuned = source.adapt(config.lora, key=config.trainer.key)
    assert tuned.adapter is not None
    tuned.adapter.save(state.variables, tmp_path / "in-process")
    run = Pretrained.from_run(str(tmp_path / "runs" / "lora"))
    assert run.adapter is not None
    run.adapter.save(run.variables, tmp_path / "from-run")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        assert (tmp_path / "from-run" / name).read_bytes() == (tmp_path / "in-process" / name).read_bytes()


def test_a_scratch_run_trains_a_lora_that_saves_from_the_run_alone(tmp_path):
    """From scratch, `--lora.rank 2 --lora.modules q_proj v_proj`
    binds the adapter to a fresh draw of the model from the run's key and
    the factors from the key folded with 1: two steps move every B and
    leave that draw bitwise under `frozen`, and `Adapter.from_run` writes
    the files an adapter bound the same way in process writes, byte for
    byte. (A model with no exported family, as this one, has no
    `Pretrained` bundle; the adapter needs none.)"""
    from dew.lora import Adapter
    from dew.objectives.base import FROZEN

    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    config = run_config(tokens, "--trainer.steps", "2", "--max-new-tokens", "0", "--trainer.name",
                        "scratch", "--lora.rank", "2", "--lora.modules", "q_proj", "v_proj")

    state = config.run()

    recorded = LMRunConfig.load(str(tmp_path / "runs" / "scratch"))
    assert recorded.lora == config.lora
    model = recorded.model.build()
    key = jax.random.key(recorded.trainer.key)
    base = model.init(key, jnp.zeros((1, SEQ), jnp.int32))
    for before, after in zip(jax.tree.leaves(base["params"]), jax.tree.leaves(state.variables[FROZEN]),
                             strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    moved = jax.tree_util.tree_leaves_with_path(state.variables["params"])
    assert moved and all(path[-1].key in ("lora_A", "lora_B") for path, _ in moved)
    assert all(bool(jnp.any(leaf)) for path, leaf in moved if path[-1].key == "lora_B")
    adapter = recorded.lora.apply(model, base, key=jax.random.fold_in(key, 1))
    adapter.save(state.variables, tmp_path / "in-process")
    rebuilt = Adapter.from_run(tmp_path / "runs" / "scratch")
    rebuilt.save(rebuilt.variables, tmp_path / "from-run")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        assert (tmp_path / "from-run" / name).read_bytes() == (tmp_path / "in-process" / name).read_bytes()


def test_the_recipe_balances_a_sparse_run(tmp_path):
    """--objective.balance-rate reaches the objective: a sparse run moves every
    router's bias by the rate each step, which the recipe could not ask for
    before, and an unbalanced run leaves it at zero."""
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    sparse = ('{"emb_features": 16, "num_layers": 2, "num_heads": 2, '
              '"mixture": {"experts": 8, "top_k": 2, "layers": [1], "bias": true}}')

    def run(name, *extra):
        config = LMRunConfig.cli(
                          recipe_args(tokens, "--trainer.steps", "2",
                                           "--max-new-tokens", "0",
                                           "--trainer.name", name, *extra,
                                           model_config=sparse))
        state = config.run()
        return np.asarray(state.variables["moe"]["layers_1"]["mlp"]["gate"]
                          ["e_score_correction_bias"])

    balanced = run("balanced", "--objective.balance-rate", "0.01")
    assert np.any(balanced != 0), "the bias never moved"
    np.testing.assert_allclose(np.abs(balanced) / 0.01,
                               np.round(np.abs(balanced) / 0.01), atol=1e-4)
    assert np.all(run("unbalanced") == 0)


def test_the_recipe_trains_the_prediction_depths_on_request(tmp_path):
    """--objective.mtp-weight reaches the objective: with the term on, a prediction
    depth's fused projection ends two steps somewhere else than the same run
    without it, whose depth sees no gradient; and the flag on a model
    without depths raises a ValueError naming num_nextn_predict_layers."""
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    deep = ('{"emb_features": 16, "num_layers": 1, "num_heads": 2, '
            '"num_nextn_predict_layers": 1}')

    def run(name, *extra, model_config=deep):
        config = LMRunConfig.cli(
                          recipe_args(tokens, "--trainer.steps", "2",
                                           "--max-new-tokens", "0",
                                           "--trainer.name", name, *extra,
                                           model_config=model_config))
        state = config.run()
        return np.asarray(state.variables["params"]["mtp_0"]["eh_proj"]["kernel"])

    assert np.any(run("mtp", "--objective.mtp-weight", "0.3") != run("plain")), \
        "the term never reached the depth"
    with pytest.raises(ValueError):
        run("dense", "--objective.mtp-weight", "0.3",
            model_config='{"emb_features": 16, "num_layers": 1, "num_heads": 2}')


def test_masked_diffusion_trains_on_packed_documents(tmp_path):
    """`--objective masked_diffusion` over packed documents: the packed
    windows reach the objective with their packing, so a validation batch
    scores its documents and weighs its padded tail at nothing. A batch that
    lost its segment ids on the way would count the tail as text."""
    from dew.data.dataset import DataPartition
    from dew.interop import Pretrained
    from dew.objectives.base import Step

    checkpoint = REPO_ROOT / "tests/fixtures/hf/llada-tiny"
    directory = tmp_path / "tokens"
    directory.mkdir()
    for split, count in (("train", 400), ("val", 96)):
        ids = np.random.RandomState(count).randint(4, 100, count).astype(np.uint8)
        ids[6::7] = 1  # every seventh id ends a document
        (directory / f"{split}.bin").write_bytes(ids.tobytes())
    (directory / "meta.json").write_text(json.dumps(
        {"tokenizer": str(checkpoint), "vocab_size": 100, "dtype": "uint8", "eos_id": 1,
         "train_tokens": 400, "val_tokens": 96}))
    config = LMRunConfig.cli([
        "--data.pack", "first-fit", "--pretrained", str(checkpoint), "--objective", "masked_diffusion",
        "--tokenizer", str(checkpoint), "--data.source", str(directory),
        "--data.seq-len", "11", "--data-read", "memory", "--loading.workers", "0",
        "--model.dtype", "float32", "--model.attention-impl", "xla",
        "--trainer.batch-size", "8", "--trainer.steps", "2", "--trainer.log-every", "1",
        "--trainer.checkpoint-dir", str(tmp_path / "runs"), "--trainer.name", "packed",
        "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
        "--objective.ema-decay", "None", "--max-new-tokens", "0"])

    state = config.run()

    assert int(state.updates) == 2
    original = Pretrained.load(str(checkpoint), dtype="float32", attention_impl="xla")
    objective = config.masked_objective(original.model, original.model_config, None)
    batch = next(iter(loader(config.data, batch=8, read="memory",
        loading=Loading(workers=0)).val(DataPartition())))
    padding = np.asarray(batch["text_coordinates"].segments) == 0
    assert padding.any() and not padding.all()
    scored = objective.evaluate(state.variables, batch, Step(jnp.asarray(0), jax.random.key(0), None))
    np.testing.assert_array_equal(np.asarray(scored.weights), ~padding)


def test_masked_diffusion_continues_a_pretrained_checkpoint(tmp_path):
    """`--pretrained` over a masked-diffusion checkpoint continues from its
    weights. One step at 1e-3 leaves the trained state within 1.1e-3 of the
    LLaDA fixture, while the fresh init the same run would otherwise draw is
    1.45 away, so a run that dropped the weights could not pass.

    A window is `--data.seq-len + 1` ids wide and masked diffusion has no
    shift, so the objective denoises all twelve; the trainer's step would
    raise on an eleven-token objective."""
    from dew.interop import Pretrained

    checkpoint = REPO_ROOT / "tests/fixtures/hf/llada-tiny"
    directory = tmp_path / "tokens"
    directory.mkdir()
    for split, count in (("train", 400), ("val", 96)):
        ids = np.random.RandomState(count).randint(4, 100, count).astype(np.uint8)
        (directory / f"{split}.bin").write_bytes(ids.tobytes())
    (directory / "meta.json").write_text(json.dumps(
        {"tokenizer": str(checkpoint), "vocab_size": 100, "dtype": "uint8", "eos_id": 1,
         "train_tokens": 400, "val_tokens": 96}))
    config = LMRunConfig.cli([
        "--pretrained", str(checkpoint), "--objective", "masked_diffusion",
        "--tokenizer", str(checkpoint), "--data.source", str(directory),
        "--data.seq-len", "11", "--loading.workers", "0",
        "--model.dtype", "float32", "--model.attention-impl", "xla",
        "--trainer.batch-size", "8", "--trainer.steps", "1", "--trainer.log-every", "1",
        "--trainer.checkpoint-dir", str(tmp_path / "runs"), "--trainer.name", "masked",
        "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
        "--objective.ema-decay", "None", "--max-new-tokens", "0", "--optim.learning-rate", "0.001"])

    state = config.run()

    assert int(state.updates) == 1
    original = Pretrained.load(str(checkpoint), dtype="float32", attention_impl="xla")
    objective = config.masked_objective(original.model, original.model_config, original.variables)
    assert objective.seq_len == 12, "the objective has to take the window's whole width"
    held = jax.tree.leaves(original.variables)
    distance = max(float(jnp.max(jnp.abs(a - b)))
                   for a, b in zip(jax.tree.leaves(state.variables), held, strict=True))
    drawn = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(
        jax.tree.leaves(config.masked_objective(original.model, original.model_config, None).init(
            jax.random.key(0))), held, strict=True))
    assert 1e-4 < distance < 1e-2, f"the step moved the checkpoint {distance:.3e}"
    assert drawn > 1.0, f"a fresh init is only {drawn:.3e} from the checkpoint"


def test_official_block_diffusion_is_a_complete_pretrained_recipe(tmp_path):
    from dew.interop import Pretrained

    checkpoint = REPO_ROOT / "tests/fixtures/hf/diffusion-gemma-sft"
    directory = tmp_path / "tokens"
    directory.mkdir()
    for split, count in (("train", 160), ("val", 32)):
        ids = np.random.RandomState(count).randint(4, 32, count).astype(np.uint8)
        (directory / f"{split}.bin").write_bytes(ids.tobytes())
    (directory / "meta.json").write_text(json.dumps(
        {"tokenizer": str(checkpoint), "vocab_size": 32, "dtype": "uint8", "eos_id": 1}))
    config = LMRunConfig.cli([
        "--pretrained", str(checkpoint), "--objective", "block_diffusion",
        "--tokenizer", str(checkpoint), "--data.source", str(directory), "--data.seq-len", "11",
        "--objective.prompt-length", "4", "--loading.workers", "0",
        "--model.dtype", "float32", "--model.attention-impl", "xla",
        "--trainer.batch-size", "8", "--trainer.steps", "1", "--trainer.log-every", "1",
        "--trainer.checkpoint-dir", str(tmp_path / "runs"), "--trainer.name", "block",
        "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
        "--max-new-tokens", "0", "--optim.learning-rate", "0.001"])
    state = config.run()
    assert int(state.updates) == 1
    original = Pretrained.load(checkpoint, dtype="float32", attention_impl="xla", max_seq_len=12)
    initial = config.block_objective(original.model, original.variables).init(jax.random.key(0))
    difference = max(float(jnp.max(jnp.abs(a - b)))
                     for a, b in zip(jax.tree.leaves(state.variables), jax.tree.leaves(initial), strict=True))
    assert difference > 1e-5
    restored = config.run()
    for wanted, actual in zip(jax.tree.leaves(state.variables), jax.tree.leaves(restored.variables),
                              strict=True):
        np.testing.assert_array_equal(actual, wanted)
    task = dew.pipeline(str(tmp_path / "runs" / "block"), ema=False)
    trained = config.block_objective(original.model, original.variables).pipeline(
        state, ema=False
    )
    prompt = [[4, 5, 6, 7]]
    np.testing.assert_array_equal(task(prompt, 4, key=9).host().tokens,
                                  trained(prompt, 4, key=9).host().tokens)


def test_block_diffusion_trains_quantized_as_any_lm_recipe_does(tmp_path):
    """`--trainer.quantization` wraps the module block-diffusion SFT trains, as it does any
    objective's (`quantize_trunk`): the recipe takes it, its step runs on the quantized forward,
    and it moves the checkpoint other than the same step without it does."""
    from dew.quantization import Quantization

    checkpoint = REPO_ROOT / "tests/fixtures/hf/diffusion-gemma-sft"
    directory = tmp_path / "tokens"
    directory.mkdir()
    for split, count in (("train", 160), ("val", 32)):
        ids = np.random.RandomState(count).randint(4, 32, count).astype(np.uint8)
        (directory / f"{split}.bin").write_bytes(ids.tobytes())
    (directory / "meta.json").write_text(json.dumps(
        {"tokenizer": str(checkpoint), "vocab_size": 32, "dtype": "uint8", "eos_id": 1}))

    def run(name, *quantized):
        return LMRunConfig.cli([
            "--pretrained", str(checkpoint), "--objective", "block_diffusion",
            "--tokenizer", str(checkpoint), "--data.source", str(directory), "--data.seq-len", "11",
            "--objective.prompt-length", "4", "--loading.workers", "0",
            "--model.dtype", "float32", "--model.attention-impl", "xla",
            "--trainer.batch-size", "8", "--trainer.steps", "1", "--trainer.log-every", "1",
            "--trainer.checkpoint-dir", str(tmp_path / "runs"), "--trainer.name", name,
            "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
            "--max-new-tokens", "0", "--optim.learning-rate", "0.001", *quantized])

    quantized = run("quantized")
    quantized = dataclasses.replace(quantized, trainer=dataclasses.replace(
        quantized.trainer, quantization=Quantization()))
    state, plain = quantized.run(), run("plain").run()
    assert int(state.updates) == int(plain.updates) == 1
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(state.variables),
                                                      jax.tree.leaves(plain.variables), strict=True))


def test_a_trained_block_diffusion_tree_saves_back_over_its_source(tmp_path):
    """The SFT trains the per-layer skip scale, which the source declares a
    constant and the objective moves into `params`, so a trained tree and the
    source it came from disagree about where that tensor lives. The tree is
    what is being written, so `source.save(destination)` takes the trained one
    and the model read back out of that directory computes the trained loss.
    The checkpoint is the published layout's own (experts and a vision
    tower): Google's dense text-only model has no published reader, and its
    export is refused (tests/test_block_sft.py)."""
    import optax

    from dew import Trainer
    from dew.interop import Pretrained
    from dew.objectives.base import Step, thaw
    from dew.objectives.diffusion.block import BlockDiffusionObjective

    checkpoint = REPO_ROOT / "tests/fixtures/hf/diffusion-gemma-workflow"
    source = Pretrained.load(checkpoint, dtype="float32", attention_impl="xla", max_seq_len=32)
    objective = BlockDiffusionObjective(source.model, prompt_length=4, num_canvases=2,
                                        variables=source.variables)
    rows = 2 * jax.device_count()
    batch = {"text": np.random.default_rng(0).integers(3, 60, (rows, 12)).astype(np.int32)}
    data = Loader(train=lambda partition: iter([batch, batch]), val=None, records=rows, batch=rows)

    state = Trainer(objective, optax.sgd(0.05), key=jax.random.key(0)).fit(data, steps=1, log_every=1)
    source.save(tmp_path / "trained", variables=thaw(state.variables))

    read = Pretrained.load(tmp_path / "trained", dtype="float32", attention_impl="xla", max_seq_len=32)
    restored = BlockDiffusionObjective(read.model, prompt_length=4, num_canvases=2,
                                       variables=read.variables)
    rebuilt = restored.init(jax.random.key(0))
    for wanted, actual in zip(jax.tree.leaves(state.variables), jax.tree.leaves(rebuilt), strict=True):
        np.testing.assert_array_equal(actual, wanted)
    step = Step(step=jnp.asarray(0, jnp.int32), key=jax.random.key(1), ema=None)
    np.testing.assert_array_equal(restored.scalar_loss(rebuilt, batch, step)[0],
                                  objective.scalar_loss(state.variables, batch, step)[0])
