"""The shipped recipes run at a tiny size on CPU: each its documented command's flags (the recipe's docstring)
over its own run (`default`, its `main`'s where the command sets nothing a run file states), with the
model's sizes, the run's length, the data and the pretrained source swapped for small local ones as a user
edits a recipe's run, and the run it saves checked. A recipe whose defaults
refuse its own command is red: the decision recipe's `--trainer.horizon.steps 2000` was refused at its default
cadence ("epoch"), which no test of the recipes ran."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import tyro
from objective_data import IMAGES
from run_kinds import QWEN_TINY, TINY_LAYA, pictures
from token_support import write_token_corpus
from tool_support import load

from dew.config.diffusion import DiffusionRunConfig
from dew.config.jepa import JepaRunConfig
from dew.config.lm import LMRunConfig
from dew.data import TFDS, Memory
from dew.data.text import export_tokens
from dew.registry import ModelConfig

TRAINED = ["--trainer.multi-host", "False", "--trainer.compilation-cache-dir", "None"]
"""What every run here adds: one process, no shared compilation cache."""

IMAGE_MODEL = {"patch_size": 2, "emb_features": 16, "num_layers": 1, "num_heads": 2}
"""The tiny image model every image recipe's run trains here."""


def with_model(run, name: str | None = None, **fields):
    """`run` with its model's `fields` over its own, of the class `name` names where given: what a user states
    by editing a recipe's run."""
    model = run.model if name is None else ModelConfig(name)
    return dataclasses.replace(run, model=dataclasses.replace(model, fields={**model.fields, **fields}))


def saved(root: Path, state) -> bool:
    """Whether the run a recipe saved under `root/runs` holds the state it returned: its configuration beside
    its checkpoints, its latest step the state's, its parameters read back the state's (row 11749: a run's
    configuration is written before it trains, so a run that saved no step passed)."""
    import jax

    from dew.checkpoints import Checkpoints

    [configuration] = sorted((root / "runs").glob("**/run.json"))
    checkpoints = Checkpoints(str(configuration.parent))
    assert checkpoints.latest == int(state.step), (checkpoints.latest, int(state.step))
    restored = checkpoints.variables(step=checkpoints.latest, ema=False, on_host=True)["params"]
    jax.tree.map(np.testing.assert_array_equal, restored, jax.device_get(state.variables["params"]))
    return True


def tickets():
    """A table of support tickets as BANKING77's rows are, each a `text` and its `label_text`, a tenth
    held out."""
    asked = [("charged twice for March", "billing"), ("the site is down", "technical"),
             ("how much is the pro plan", "sales"), ("refund my last invoice", "billing")]
    return Memory([{"text": f"{text}, ticket {index}", "label_text": label}
                   for index in range(40) for text, label in asked]).holdout(0.1)


def byte_store(root: Path, name: str) -> str:
    """A small store of byte tokens at `root/name`, as `dew data export` writes one."""
    documents = [{"text": f"document {index}: " + "to be or not to be " * (1 + index % 3)}
                 for index in range(64)]
    export_tokens(Memory(documents), str(root / name), "byte")
    return str(root / name)


@pytest.mark.mesh
@pytest.mark.parametrize("command", [
    ["--trainer.horizon.epochs", "1"],
    ["--pretrained", str(QWEN_TINY), "--lora.rank", "2", "--lora.modules", "q_proj", "v_proj",
     "--optim.learning-rate", "1e-4", "--trainer.horizon.steps", "2"],
], ids=["epochs", "lora"])
def test_the_decision_recipe_trains_and_saves_its_run(command, tmp_path):
    """recipes/decision/train.py's two documented commands over a small table: a run by epochs over the
    default model (a tiny LAYA's), and a LoRA fine-tune of a tiny Qwen by steps, each saving its run."""
    recipe = load("train", "recipes/decision")
    pretrained = [] if "--pretrained" in command else ["--pretrained", str(TINY_LAYA)]
    state = recipe.main([*pretrained, *command, *TRAINED, "--trainer.checkpoint-dir", str(tmp_path / "runs")],
                        tickets())
    assert int(state.step) == 2 if "--trainer.horizon.steps" in command else int(state.step) > 0
    assert saved(tmp_path, state)


@pytest.mark.mesh
def test_the_diffusion_recipe_trains_and_saves_its_run(tmp_path):
    """recipes/diffusion/train.py's documented command at 8 pixels, a batch of eight, one epoch of a small
    picture table, captioned by a character table in place of the Hub's encoder, and no CLIP score."""
    recipe = load("train", "recipes/diffusion")
    run = with_model(recipe.default(*pictures()), "simple_dit", **IMAGE_MODEL).override({
        "data.images.size": 8, "data.images.batch": 8, "validation.images.size": 8,
        "validation.images.batch": 8})
    state = DiffusionRunConfig.cli(["--trainer.horizon.epochs", "1", *TRAINED, "--trainer.checkpoint-dir",
                                    str(tmp_path / "runs"), "--val-metrics", "[]", "--text.encoder",
                                    "char_table", "--text.checkpoint", "char_table"], default=run).run()
    assert int(state.step) > 0 and saved(tmp_path, state)


@pytest.mark.mesh
def test_the_jepa_recipe_trains_and_saves_its_run(tmp_path):
    """recipes/jepa/train.py's documented command at 16 pixels, a batch of eight and two probe classes, one
    epoch of a small image table."""
    recipe = load("train", "recipes/jepa")
    images = TFDS("dew_images", path=str(IMAGES), split="train")
    run = with_model(recipe.default(*images.holdout(split="test")), **IMAGE_MODEL).override({
        "data.images.size": 16, "data.images.batch": 8, "validation.images.size": 16,
        "validation.images.batch": 8})
    state = JepaRunConfig.cli(["--trainer.horizon.epochs", "1", "--probe-classes", "2", *TRAINED,
                               "--trainer.checkpoint-dir", str(tmp_path / "runs")], default=run).run()
    assert int(state.step) > 0 and saved(tmp_path, state)


@pytest.mark.mesh
def test_the_lm_recipe_trains_an_epoch_of_its_store_and_saves_its_run(tmp_path):
    """recipes/lm/train.py's documented command over a byte store of 40 x 16 tokens, 39 windows of 16 inputs
    and their targets: five batches of eight, the fifth holding the last seven and the next epoch's first, as
    an order that repeats for ever reads on, its windows read by index and ordered after, so the run knows an
    epoch's length."""
    recipe = load("train", "recipes/lm")
    tokens = np.random.default_rng(0).integers(1, 250, 40 * 16).astype(np.uint8)
    store = write_token_corpus(tmp_path / "shakespeare-byte", tokens).train
    run = with_model(recipe.default(store), emb_features=16, num_layers=1, num_heads=2).override({
        "data.token_windows.seq_len": 16, "data.token_windows.batch": 8})
    state = LMRunConfig.cli(["--trainer.horizon.epochs", "1", *TRAINED, "--trainer.checkpoint-dir",
                             str(tmp_path / "runs"), "--max-new-tokens", "0"], default=run).run()
    assert int(state.step) == 5 and saved(tmp_path, state)


@pytest.mark.mesh
@pytest.mark.parametrize("corpora", [("web", "code"), ("web",)], ids=["two corpora", "one corpus proxy"])
def test_the_rigel_recipe_trains_its_phases_and_saves_its_run(corpora, tmp_path):
    """recipes/lm/rigel.py's documented commands over small byte stores, a width of 128 and eight steps, the
    fewest whose rescaled phases each end past the one before: the full mixture of two corpora, and the muP
    proxy on one."""
    recipe = load("rigel", "recipes/lm")
    named = [word for name in corpora for word in (name, byte_store(tmp_path, name))]
    args = tyro.cli(recipe.RigelArgs, args=["--corpora", *named, "--width", "128", "--steps", "8",
                                            "--batch-size", "8", "--seq-len", "16", "--tokenizer", "byte"])
    run = with_model(recipe.run_config(args), num_layers=4,
                     layer_types=["mamba", "mamba", "mamba", "attention"])
    state = LMRunConfig.cli(["--max-new-tokens", "0", *TRAINED, "--trainer.checkpoint-dir",
                             str(tmp_path / "runs")], default=run).run()
    assert int(state.step) == 8 and saved(tmp_path, state)


@pytest.mark.mesh
@pytest.mark.network
def test_the_mixture_recipe_counts_each_sources_targets():
    """recipes/lm/mixture.py's documented command over its first eight mixed documents, read from the Hub:
    it counts the valid targets of the sources the mix drew."""
    import tyro

    recipe = load("mixture", "recipes/lm")
    targets = recipe.main(tyro.cli(recipe.Args, args=["--show", "8"]))
    assert set(targets) == set(recipe.WEIGHTS) and sum(targets.values()) > 0


@pytest.mark.mesh
def test_the_mixture_recipe_counts_each_sources_targets_over_known_documents():
    """recipes/lm/mixture.py's count over three sources whose every document holds a known number of
    targets (3, 5 and 7), named otherwise than its keys, is each source's targets among the first 24
    documents its token mix draws: each draw the source furthest below its share of the targets drawn
    (`WEIGHTS`, 70/20/10), the first of equals, as the token mix states it draws
    (`dew.data.mixing.TokenDeficit`): 19 web documents, 3 code and 2 chat (row 11750: a source counted
    zero, or a mix of equal documents, passed; its review: a source named otherwise counted zero)."""
    recipe = load("mixture", "recipes/lm")
    per = {"web": 3, "code": 5, "chat": 7}
    parts = {name: Memory([{"text": np.arange(1, count + 2, dtype=np.int32)} for _ in range(64)])
             .read("memory").named(f"source {name}") for name, count in per.items()}
    assert recipe.main(recipe.Args(show=24), parts) == {"web": 57, "code": 15, "chat": 14}

