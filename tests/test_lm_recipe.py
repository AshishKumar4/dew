"""recipes/lm/train.py: what it refuses, and a run over real token files."""

import dataclasses
import importlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from token_support import write_token_corpus
from tool_support import load

import dew
from dew.config.lm import LMRunConfig
from dew.data import Loading, presets
from dew.data.compile import compile, held
from dew.data.dataset import Loader
from dew.data.pipeline import Dataset
from dew.data.sources.tokens import TokenFiles
from dew.inference import TextGeneration
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling

pytestmark = pytest.mark.mesh

REPO_ROOT = Path(__file__).resolve().parents[1]
SEQ = 32


@pytest.mark.parametrize("steps", [7_250, 725_000])
def test_rigel_empty_phases_extend_the_preceding_mixture(steps):
    rigel = load("rigel", "recipes/lm")
    assert rigel.schedule({"stem": "data/stem", "other": "data/other"}, steps) == (
        ({"data/stem": 1.0}, round(515_000 * steps / rigel.STEPS)),
        ({"data/stem": 11.5, "data/other": 4.9}, None),
    )


def test_rigel_leading_empty_phases_take_the_first_available_mixture():
    rigel = load("rigel", "recipes/lm")
    assert rigel.schedule({"other": "data/other"}, 7_250) == (({"data/other": 1.0}, None),)
    assert rigel.schedule({"stem": "data/stem"}, 7_250) == (({"data/stem": 1.0}, None),)
    with pytest.raises(ValueError):
        rigel.schedule({"unknown": "data/unknown"}, 7_250)


def test_the_recipes_default_chain_trains_an_epoch_of_another_store(tmp_path):
    """recipes/lm/train.py's default chain, pointed at another store with `--set`, trains by epochs:
    its windows are read by index and ordered after, so the run knows an epoch's length. 40 x SEQ
    tokens are 39 windows of SEQ inputs and their targets: five batches of eight, the fifth holding
    the last seven and the next epoch's first, as an order that repeats for ever reads on."""
    recipe = load("train", "recipes/lm")
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ)
    config = LMRunConfig.cli(recipe_args(
        tokens, "--set", f"data.source.url={tokens.train}", "--set", f"data.token_windows.seq_len={SEQ}",
        "--set", "data.token_windows.batch=8", "--trainer.epochs", "1", "--trainer.name", "default",
        "--max-new-tokens", "0", model_config='{"emb_features": 16, "num_layers": 1, "num_heads": 2}'),
        default=LMRunConfig(data=recipe.WINDOWS))
    assert int(config.run().step) == 5


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


def windows(corpus, held: bool = False, **settings):
    """The corpus's training store's windows of SEQ tokens in batches of 8 (`presets.lm`), or its
    validation store's where `held`."""
    store = TokenFiles(corpus.val if held else corpus.train)
    chain = store.pipe(presets.lm, None, **{"seq_len": SEQ, "batch": 8, "pack": None, **settings})
    return chain.loading(Loading(workers=0))


def run_config(tokens, *args, model='{"emb_features": 16, "num_layers": 1, "num_heads": 2}', **settings):
    """The recipe's run from its command line over the store's windows (`windows`), scored on its
    validation split."""
    default = LMRunConfig(data=windows(tokens, **settings), validation=windows(tokens, held=True, **settings))
    return LMRunConfig.cli([
        *args, "--trainer.checkpoint-dir", str(Path(tokens.train).parents[1] / "runs"),
        "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
        "--trainer.log-every", "1", "--model.dtype", "float32", *model_flags(model)], default=default)


@pytest.mark.parametrize("packed", [False, True])
def test_the_recipe_trains_on_tokenized_files(tmp_path, packed):
    """A run from the command line: the windows or packed documents through
    the trainer, perplexity scored on val.bin, the run spec and a checkpoint
    at the final step."""
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    args = ["--trainer.epochs", "1", "--prompt", "the ", "--max-new-tokens", "4",
            "--trainer.name", "run"]
    settings = {"pack": "first-fit", "bins": 2} if packed else {}
    config = run_config(tokens, *args, **settings)
    assert config.tokens().packed is packed

    state = config.run()

    data = compile(config.chain().read("memory"))
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


def test_the_recipe_trains_an_lm_objective_it_names_by_import_path(tmp_path):
    """An objective Dew does not ship, named by its import path, builds from the arguments the recipe
    gives an LM objective and trains, with no edit to the recipe."""
    # A run names a package outside Dew once it is imported.
    importlib.import_module("lm_objectives")

    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    config = run_config(tokens, "--trainer.steps", "1", "--trainer.name", "own", "--objective",
                        "lm_objectives:Anchored")
    assert int(config.run().step) == 1


def test_a_token_budget_alone_trains_over_data_with_no_epoch_length(tmp_path):
    """Packing a streamed read online leaves a run no epoch to count, and a run whose length is its
    tokens alone needs none: it trains until its targets reach the budget."""
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    config = run_config(tokens, "--trainer.tokens", str(2 * 8 * SEQ), "--trainer.name", "budget",
                        "--trainer.eval-every", "None", "--trainer.checkpoint-every", "None",
                        pack="first-fit", bins=2)
    config = dataclasses.replace(config, data=config.chain().read("streamed"))
    assert compile(config.chain()).steps_per_epoch is None
    assert int(config.run().step) >= 2


def test_the_recipe_trains_on_weighted_corpora(tmp_path):
    """Two token stores, each packed by itself and mixed by their tokens 0.7 to 0.3, train one run:
    the vocabulary comes from the stores' shared meta.json, and run.json records the mixture's chain
    so the run reloads as configured."""
    from dew.data import mix

    first = write_token_files(tmp_path / "first", 40 * SEQ, 8 * SEQ, eos_id=0)
    second = write_token_files(tmp_path / "second", 24 * SEQ, 8 * SEQ, eos_id=0)

    def packed(store):
        return TokenFiles(store.train).pipe(presets.lm, None, seq_len=SEQ, batch=None, pack="first-fit",
                                            bins=2)

    mixture = mix([packed(first), packed(second)], [0.7, 0.3], by="tokens").batch(8).read("memory")
    config = LMRunConfig.cli([
        "--trainer.checkpoint-dir", str(tmp_path / "runs"),
        "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
        "--trainer.steps", "2", "--trainer.name", "mixed", "--max-new-tokens", "0",
        "--model.dtype", "float32",
        "--model.emb_features", "16", "--model.num_layers", "1", "--model.num_heads", "2"],
        default=LMRunConfig(data=mixture.loading(Loading(workers=0))))
    mixed = config.data.source
    assert [child.source.url for child in mixed.pipelines] == [first.train, second.train]
    assert mixed.weights == (0.7, 0.3)

    state = config.run()

    assert int(state.step) == 2
    recorded = LMRunConfig.load(str(tmp_path / "runs" / "mixed"))
    assert recorded.to_dict()["data"] == config.to_dict()["data"]
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


def test_a_token_budget_trains_over_a_stream_with_no_epoch_length(tmp_path):
    """`--trainer.tokens` alone bounds a run over windows read as a stream, each share its own order,
    which has no epoch length to count: the run trains until its real tokens reach the budget, three
    batches of eight windows of SEQ targets."""
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ)
    streamed = windows(tokens).read("streamed")
    assert compile(streamed).steps_per_epoch is None
    model = '{"emb_features": 16, "num_layers": 1, "num_heads": 2}'
    config = LMRunConfig.cli(recipe_args(
        tokens, "--trainer.tokens", str(3 * 8 * SEQ), "--trainer.eval-every", "None",
        "--trainer.checkpoint-every", "None", "--trainer.name", "tokens", "--max-new-tokens", "0",
        model_config=model), default=LMRunConfig(data=streamed))
    assert int(config.run().step) == 3


def recipe_args(tokens, *args, model_config="{}"):
    return [*args, "--trainer.checkpoint-dir", str(Path(tokens.train).parents[1] / "runs"),
            "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
            "--trainer.log-every", "1", "--model.dtype", "float32",
            *model_flags(model_config)]


def pretrained_config(tokens, pretrained, *args, model_config="{}"):
    return LMRunConfig.cli(recipe_args(tokens, "--pretrained", str(pretrained), *args,
                                       model_config=model_config),
                           default=LMRunConfig(data=windows(tokens), validation=windows(tokens, held=True)))


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


def test_a_run_resumed_under_its_name_trains_the_model_it_recorded(tmp_path):
    """A run with checkpoints resumes under its name only as the model and objective it recorded: a
    rope base that keeps every shape, or a quantized forward pass, is refused before the record is
    replaced or a weight restored, and the run as recorded trains on."""
    from dew.checkpoints import Checkpoints

    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    model = {"emb_features": 16, "num_layers": 1, "num_heads": 2}
    run_config(tokens, "--trainer.steps", "1", "--trainer.name", "kept", model=json.dumps(model)).run()
    run = tmp_path / "runs" / "kept"
    recorded = (run / "run.json").read_bytes()

    from dew.quantization import Quantization

    rotated = run_config(tokens, "--trainer.steps", "2", "--trainer.name", "kept",
                         model=json.dumps({**model, "rope_theta": 500.0}))
    same = run_config(tokens, "--trainer.steps", "2", "--trainer.name", "kept", model=json.dumps(model))
    quantized = dataclasses.replace(same, trainer=dataclasses.replace(same.trainer,
                                                                      quantization=Quantization()))
    for other in (rotated, quantized):
        with pytest.raises(ValueError):
            other.run()
        assert (run / "run.json").read_bytes() == recorded
        assert Checkpoints(str(run)).latest == 1

    # The record may name its class by its import path, as the registry reads it too.
    record = json.loads((run / "run.json").read_text())
    (run / "run.json").write_text(json.dumps({**record, "class": "dew.config.lm:LMRunConfig"}))
    run_config(tokens, "--trainer.steps", "2", "--trainer.name", "kept", model=json.dumps(model)).run()
    assert Checkpoints(str(run)).latest == 2


def test_fresh_processes_train_a_runs_record_and_export_it(tmp_path):
    """`dew train run.json` and `dew export` each run in a process of their own, which imports what
    the record names and nothing a test happened to import before: the recorded run trains on to
    two steps, and the export reads back as that step's weights, leaf for leaf."""
    import os
    import subprocess
    import sys

    from dew.checkpoints import Checkpoints
    from dew.interop import Pretrained

    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    checkpoint = export_tiny_decoder(tmp_path / "checkpoint")
    pretrained_config(tokens, checkpoint, "--trainer.steps", "1", "--max-new-tokens", "0",
                      "--trainer.name", "fresh").run()
    run, exported = tmp_path / "runs" / "fresh", tmp_path / "exported"

    def dew(*arguments: str) -> None:
        """The command line on this checkout's source, as a fresh interpreter imports it."""
        environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
        done = subprocess.run([sys.executable, "-m", "dew.cli.main", *arguments], capture_output=True,
                              text=True, check=False, env=environment)
        assert done.returncode == 0, done.stdout + done.stderr

    dew("train", str(run / "run.json"), "--set", "trainer.steps=2")
    dew("export", str(run), str(exported))

    assert Checkpoints(str(run)).latest == 2
    trained = Checkpoints(str(run)).variables(step=2)["params"]
    loaded = Pretrained.load(str(exported), dtype="float32", attention_impl="reference").variables["params"]
    for path, leaf in jax.tree_util.tree_flatten_with_path(loaded)[0]:
        held = trained
        for entry in path:
            held = held[entry.key]
        np.testing.assert_array_equal(np.asarray(held), np.asarray(leaf), err_msg=jax.tree_util.keystr(path))


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
        LMRunConfig.cli(recipe_args(tokens, "--pretrained-param-dtype", "bfloat16"),
                        default=LMRunConfig(data=windows(tokens)))


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


class Started(LMRunConfig):
    """A run that hands back the objective it would train, rather than training it."""

    def train(self, objective, dataset, **settings):
        return objective


@pytest.mark.parametrize(("lora", "whole"), [(True, False), (True, True), (False, False)],
                         ids=["lora", "lora-frozen-whole", "full"])
def test_a_pretrained_runs_source_loads_where_its_start_keeps_it(tmp_path, lora, whole):
    """A LoRA run from a source reads each leaf of its frozen base onto its trainer's fsdp mesh,
    sharded as its layout shards the state (every leaf past `min_shard=1`, so some split), never into
    a host array (`TrainerConfig.source_placement`): its start places none of the base again, the
    state's frozen leaves the source's arrays themselves, each where the state keeps it. A layout
    that keeps the frozen tree whole, and a run that trains the source itself, read it on the host."""
    import optax

    from dew.objectives.base import FROZEN
    from dew.training import Layout, MeshSpec, Trainer

    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    checkpoint = export_tiny_decoder(tmp_path / "checkpoint")
    adapter = ("--lora.rank", "4", "--lora.modules", "q_proj") if lora else ()
    config = Started.cli(recipe_args(tokens, "--pretrained", str(checkpoint), "--trainer.steps", "1",
                                     "--max-new-tokens", "0", *adapter),
                         default=Started(data=windows(tokens), validation=windows(tokens, held=True)))
    sharded = Layout(min_shard=1, frozen_whole=whole)
    config = dataclasses.replace(config, trainer=dataclasses.replace(
        config.trainer, mesh=MeshSpec(fsdp=jax.device_count()), layout=sharded))
    objective = config.prepare().train("placed")
    held = objective.held_variables()
    assert held is not None
    base = held[FROZEN] if lora else held["params"]
    if whole or not lora:
        assert not any(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(base))
        return
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(base))
    assert any(leaf.addressable_shards[0].data.shape != leaf.shape for leaf in jax.tree.leaves(base))
    trainer = Trainer(objective, optax.sgd(0.1), key=jax.random.key(0), mesh=config.trainer.mesh,
                      layout=config.trainer.layout)
    state, shardings = trainer.place()[:2]
    assert ({id(leaf) for leaf in jax.tree.leaves(state.variables[FROZEN])}
            == {id(leaf) for leaf in jax.tree.leaves(base)})
    placed = jax.tree.leaves(shardings.variables[FROZEN])
    for leaf, sharding in zip(jax.tree.leaves(base), placed, strict=True):
        assert leaf.sharding == sharding


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
                                           model_config=sparse), default=LMRunConfig(data=windows(tokens)))
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
                                           model_config=model_config),
            default=LMRunConfig(data=windows(tokens)))
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
    splits = [np.random.RandomState(count).randint(4, 100, count) for count in (400, 96)]
    for ids in splits:
        ids[6::7] = 1  # every seventh id ends a document
    corpus = write_token_corpus(tmp_path / "tokens", *splits, tokenizer=str(checkpoint), vocab_size=100,
                                eos_id=1)

    def packed(url):
        windows = TokenFiles(url).pipe(presets.lm, None, seq_len=11, batch=8, pack="first-fit")
        return windows.read("memory").loading(Loading(workers=0))

    config = LMRunConfig.cli([
        "--pretrained", str(checkpoint), "--objective", "masked_diffusion", "--tokenizer", str(checkpoint),
        "--model.dtype", "float32", "--model.attention-impl", "xla",
        "--trainer.steps", "2", "--trainer.log-every", "1",
        "--trainer.checkpoint-dir", str(tmp_path / "runs"), "--trainer.name", "packed",
        "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
        "--objective.ema-decay", "None", "--max-new-tokens", "0"],
        default=LMRunConfig(data=packed(corpus.train), validation=packed(corpus.val)))

    state = config.run()

    assert int(state.updates) == 2
    original = Pretrained.load(str(checkpoint), dtype="float32", attention_impl="xla")
    objective = config.masked_objective(original.model, original.model_config, None)
    assert isinstance(config.validation, Dataset)
    batch = next(iter(held(config.validation, objective=objective.data())(DataPartition())))
    padding = np.asarray(batch["text_coordinates"].segments) == 0
    assert padding.any() and not padding.all()
    scored = objective.evaluate(state.variables, batch, Step(jnp.asarray(0), jax.random.key(0), None))
    np.testing.assert_array_equal(np.asarray(scored.weights), ~padding)


def test_masked_diffusion_continues_a_pretrained_checkpoint(tmp_path):
    """`--pretrained` over a masked-diffusion checkpoint continues from its
    weights. One step at 1e-3 leaves the trained state within 1.1e-3 of the
    LLaDA fixture, while the fresh init the same run would otherwise draw is
    1.45 away, so a run that dropped the weights could not pass.

    A window is `seq_len + 1` ids wide and masked diffusion has no
    shift, so the objective denoises all twelve; the trainer's step would
    raise on an eleven-token objective."""
    from dew.interop import Pretrained

    checkpoint = REPO_ROOT / "tests/fixtures/hf/llada-tiny"
    corpus = write_token_corpus(tmp_path / "tokens", np.random.RandomState(400).randint(4, 100, 400),
                                tokenizer=str(checkpoint), vocab_size=100, eos_id=1)
    data = TokenFiles(corpus.train).pipe(presets.lm, None, seq_len=11, batch=8, pack=None).loading(
        Loading(workers=0))
    config = LMRunConfig.cli([
        "--pretrained", str(checkpoint), "--objective", "masked_diffusion", "--tokenizer", str(checkpoint),
        "--model.dtype", "float32", "--model.attention-impl", "xla",
        "--trainer.steps", "1", "--trainer.log-every", "1",
        "--trainer.checkpoint-dir", str(tmp_path / "runs"), "--trainer.name", "masked",
        "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
        "--objective.ema-decay", "None", "--max-new-tokens", "0", "--optim.learning-rate", "0.001"],
        default=LMRunConfig(data=data))

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


DIFFUSION_GEMMA = REPO_ROOT / "tests/fixtures/hf/diffusion-gemma-sft"


def gemma_spans(directory):
    """Windows of 11 inputs, eight a batch, over a token store of random ids in DIFFUSION_GEMMA's
    vocabulary written to `directory`: what a block-diffusion recipe run trains on."""
    corpus = write_token_corpus(directory, np.random.RandomState(160).randint(4, 32, 160),
                                tokenizer=str(DIFFUSION_GEMMA), vocab_size=32, eos_id=1)
    return TokenFiles(corpus.train).pipe(presets.lm, None, seq_len=11, batch=8, pack=None).loading(
        Loading(workers=0))


def test_official_block_diffusion_is_a_complete_pretrained_recipe(tmp_path):
    from dew.interop import Pretrained

    checkpoint = DIFFUSION_GEMMA
    spans = gemma_spans(tmp_path / "tokens")
    config = LMRunConfig.cli([
        "--pretrained", str(checkpoint), "--objective", "block_diffusion", "--tokenizer", str(checkpoint),
        "--objective.prompt-length", "4", "--model.dtype", "float32", "--model.attention-impl", "xla",
        "--trainer.steps", "1", "--trainer.log-every", "1",
        "--trainer.checkpoint-dir", str(tmp_path / "runs"), "--trainer.name", "block",
        "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
        "--max-new-tokens", "0", "--optim.learning-rate", "0.001"], default=LMRunConfig(data=spans))
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

    checkpoint = DIFFUSION_GEMMA
    spans = gemma_spans(tmp_path / "tokens")

    def run(name, *quantized):
        return LMRunConfig.cli([
            "--pretrained", str(checkpoint), "--objective", "block_diffusion", "--tokenizer", str(checkpoint),
            "--objective.prompt-length", "4", "--model.dtype", "float32", "--model.attention-impl", "xla",
            "--trainer.steps", "1", "--trainer.log-every", "1",
            "--trainer.checkpoint-dir", str(tmp_path / "runs"), "--trainer.name", name,
            "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
            "--max-new-tokens", "0", "--optim.learning-rate", "0.001", *quantized],
            default=LMRunConfig(data=spans))

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


def test_a_pretrained_run_loops_its_decoder_as_a_decision_run_does(tmp_path):
    """`--loop-steps 2 --loop-layers 0 1` give a --pretrained decoder with no loop one over its own
    block, as they do a decision run's (`DecoderRunConfig`): the objective trains the looped decoder
    over the checkpoint's weights as they were, and the run's record holds the loop, which a resume
    takes again. A fresh decoder's run refuses them: its model states its own loop."""
    from dew.interop import Pretrained
    from dew.nn.backbones.decoder_stack import Loop

    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    checkpoint = export_tiny_decoder(tmp_path / "checkpoint")
    config = Started.cli(recipe_args(tokens, "--pretrained", str(checkpoint), "--trainer.steps", "1",
                                     "--max-new-tokens", "0", "--loop-steps", "2", "--loop-layers", "0", "1"),
                         default=Started(data=windows(tokens), validation=windows(tokens, held=True)))
    objective = config.prepare().train("looped")
    assert objective.model.loop == Loop(2, step_norm=False, layers=(0, 1))
    loaded = Pretrained.load(checkpoint, dtype="float32")
    for mine, theirs in zip(jax.tree.leaves(objective.held_variables()["params"]),
                            jax.tree.leaves(loaded.variables["params"]), strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))
    recorded = LMRunConfig.from_dict(config.to_dict())
    assert (recorded.loop_steps, recorded.loop_layers) == (2, (0, 1))
    with pytest.raises(ValueError):
        LMRunConfig(loop_steps=2)


def test_a_run_that_names_no_metrics_and_validates_nothing_trains(tmp_path):
    """`--val-metrics '[]'` scores no metric: a run whose data holds no validation split trains with no
    suite at all, as one that names none does, rather than an empty set of splits."""
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    config = dataclasses.replace(run_config(tokens, "--trainer.steps", "2", "--val-metrics", "[]"),
                                 validation=None)
    assert config.val_metrics == ()
    assert int(config.run().step) == 2
