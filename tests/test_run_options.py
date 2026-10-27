"""Every option of the common run layer reaches what consumes it, in every kind of run (row 1901).

Seven of twenty defects a run's consumers found on GPUs were options a kind of run or the command line
did not hand on (rows 1610, 1772, 1786, 1884, 1889, 1894, 1895). Every kind builds its objective, its
data and its metrics from `RunConfig`'s fields (`prepare`), and trains through `RunConfig.train`, where
alone the trainer, `fit`'s arguments, the checkpoints and the tracker are built from its
`TrainerConfig`. So each kind here, a language model's, a decision's, a diffusion model's and a JEPA's,
is set its options and observed where each acts: the scalars and the checkpoints its run writes, the
state it ends on and the model it trains; and an option a kind builds nothing from is refused before
it trains, rather than dropped. Nothing here reads a config's fields back."""

import csv
import dataclasses
import json
from pathlib import Path

import jax
import numpy as np
import pytest
from objective_cases import IMAGES
from test_lm_recipe import SEQ, write_token_files

from dew.batch import FROZEN
from dew.cli.main import main
from dew.config import ObjectiveConfig, RunConfig, SourceOptions, TrainerConfig
from dew.config.diffusion import DiffusionRunConfig
from dew.config.jepa import JepaRunConfig
from dew.config.lm import LMRunConfig
from dew.data import TFDS, Files, Loading, images as image_steps, presets
from dew.data.sources.tokens import TokenFiles
from dew.decision import decisions
from dew.decision.config import DecisionRunConfig
from dew.diffusion.presets import Flow
from dew.inputs import TextCondition
from dew.lora import LoRA
from dew.registry import ModelConfig
from dew.sampling import Euler
from dew.training import Layout, MeshSpec
from dew.training.evaluation import Summary

pytestmark = pytest.mark.mesh

TINY_LAYA = Path(__file__).parent / "fixtures" / "laya" / "tiny"
RATE = 0.0123
"""A learning rate no kind takes by default, which the run's optimizer logs as it steps."""


def read(chain):
    return chain.read("memory").loading(Loading(workers=0))


def lm(root: Path) -> RunConfig:
    corpus = write_token_files(root / "tokens", 40 * SEQ, 8 * SEQ)

    def windows(path):
        return read(TokenFiles(path).shuffle().pipe(presets.lm, None, seq_len=SEQ, batch=8, pack=None))
    return LMRunConfig(model=ModelConfig("causal_transformer", {"emb_features": 16, "num_layers": 1,
                                                                "num_heads": 2, "dtype": "float32"}),
                       data=windows(corpus.train), validation=windows(corpus.val))


def decision(root: Path) -> RunConfig:
    table = root / "tickets.csv"
    tickets = [("charged twice for March", "billing"), ("the site is down", "technical"),
               ("how much is the pro plan", "sales"), ("refund my last invoice", "billing")] * 8
    with table.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])
        writer.writerows(tickets)
    train, held = Files(str(table), format="csv").holdout(0.25)

    def asked(rows):
        return read(rows.shuffle().pipe(decisions, question="team").batch(8))
    return DecisionRunConfig(pretrained=str(TINY_LAYA), model=ModelConfig("causal_transformer",
                                                                          {"dtype": "float32"}),
                             data=asked(train), validation=asked(held))


def pictures():
    return TFDS("dew_images", path=str(IMAGES), split="all").holdout(records=4)


def diffusion(root: Path) -> RunConfig:
    del root
    train, held = pictures()

    def captioned(rows):
        return read(rows.shuffle().pipe(presets.images, 4, batch=8, caption=image_steps.caption()))
    return DiffusionRunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 2, "emb_features": 16, "num_layers": 1,
                                         "num_heads": 2}),
        data=captioned(train), validation=captioned(held), preset=Flow(), val_metrics=(),
        text=TextCondition(encoder="char_table", checkpoint="char_table"),
        objective=ObjectiveConfig("diffusion", {"solver": Euler(), "steps": 2}))


def jepa(root: Path) -> RunConfig:
    del root
    train, held = pictures()

    def patched(rows):
        return read(rows.shuffle().pipe(presets.images, 16, batch=8))
    return JepaRunConfig(
        model=ModelConfig("jepa_encoder", {"patch_size": 2, "emb_features": 16, "num_layers": 1,
                                           "num_heads": 2, "mlp_ratio": 2, "dtype": "float32",
                                           "attention_impl": "xla"}),
        predictor={"num_layers": 1, "predictor_features": 8}, num_target_blocks=2,
        data=patched(train), validation=patched(held))


KINDS = {"lm": lm, "decision": decision, "diffusion": diffusion, "jepa": jepa}
DEFAULT_METRICS = {"lm": "val/perplexity", "decision": "val/accuracy"}
"""A metric each kind scores where `val_metrics` names none, for the kinds that score one."""


def scalars(run: Path) -> dict[int, dict[str, float]]:
    """Each step's scalars the run logged, joined."""
    logged = {}
    for line in (run / "tracking" / "scalars.jsonl").read_text().splitlines():
        row = json.loads(line)
        logged.setdefault(row["step"], {}).update(row["scalars"])
    return logged


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("kind", list(KINDS))
def test_every_trainer_option_set_on_the_command_line_reaches_the_kinds_run(kind, tmp_path):
    """`dew train` over a kind's run.json with its options set (`--set`): two steps pooled into one
    update commit once, at the second step's end (`train/grad_norm`), under a dynamic loss scale
    (`train/loss_scale`) and the learning rate set (`train/learning_rate`); every step is logged and
    scored on the held split, with no metric the kind would score by default and the summary named
    formed from the scores; and the run's one checkpoint kept is its last, in the directory named."""
    record = tmp_path / "record"
    KINDS[kind](tmp_path).save(str(record))
    runs = tmp_path / "runs"
    settings = {"trainer.name": "set", "trainer.checkpoint_dir": str(runs), "trainer.steps": "2",
                "trainer.accumulation": "2", "trainer.log_every": "1", "trainer.eval_every": "1",
                "trainer.checkpoint_every": "1", "trainer.keep": "1", "trainer.dynamic_scale": "true",
                "trainer.multi_host": "false", "trainer.compilation_cache_dir": "None",
                "optim.learning_rate": str(RATE), "val_metrics": "[]",
                "val_summaries": json.dumps([{"name": "board", "weights": {"val/loss": 1},
                                              "better": "lower"}])}
    assert main(["train", str(record / "run.json"),
                 *(flag for path, value in settings.items() for flag in ("--set", f"{path}={value}"))]) == 0
    run = runs / "set"
    logged = scalars(run)
    assert {step for step, row in logged.items() if "train/loss" in row} == {1, 2}
    assert [step for step, row in logged.items() if "train/grad_norm" in row] == [2]
    for step in (1, 2):
        row = logged[step]
        assert row["train/learning_rate"] == pytest.approx(RATE)
        assert "train/loss_scale" in row
        assert row[f"{Summary.SPLIT}/board"] == row["val/loss"]
        assert DEFAULT_METRICS.get(kind) not in row
    assert sorted(path.name for path in run.iterdir() if path.name.isdigit()) == ["2"]


def trained(kind: str, root: Path, name: str, **fields):
    """A kind's run trained one step from Python (`prepare().train`) over a mesh of four data shards
    by two fsdp shards, its trainer the fields given: its state, and its step's scalars."""
    run = KINDS[kind](root)
    trainer = TrainerConfig(checkpoint_dir=str(root / "runs"), steps=1, log_every=1, eval_every=None,
                            checkpoint_every=None, checkpoint_final=False, multi_host=False,
                            mesh=MeshSpec(fsdp=2), layout=Layout(min_shard=64), **fields.pop("trainer", {}))
    state = dataclasses.replace(run, trainer=trainer, **fields).prepare().train(name)
    return state, scalars(root / "runs" / name)[1]


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("kind", list(KINDS))
def test_a_kinds_key_data_seed_mesh_and_adapter_reach_what_it_trains(kind, tmp_path):
    """From Python, a kind's run trains under each option it states: another `trainer.key` draws
    other parameters, another `data_seed` reads another first batch from parameters drawn alike, its
    mesh and layout split a parameter over its fsdp shards, and a `lora` trains its factors over a
    frozen base."""
    base, first = trained(kind, tmp_path, "base", trainer={"key": 1})
    keyed, _ = trained(kind, tmp_path, "keyed", trainer={"key": 2})
    _, reordered = trained(kind, tmp_path, "seeded", trainer={"key": 1}, data_seed=5)
    held = [np.asarray(leaf) for leaf in jax.tree.leaves(base.variables["params"])]
    assert any(not np.array_equal(mine, np.asarray(theirs))
               for mine, theirs in zip(held, jax.tree.leaves(keyed.variables["params"]), strict=True))
    assert first["train/loss"] != reordered["train/loss"]
    assert any(len(leaf.sharding.device_set) > 1 and not leaf.sharding.is_fully_replicated
               for leaf in jax.tree.leaves(base.variables["params"]))
    adapted, _ = trained(kind, tmp_path, "adapted", trainer={"key": 1}, lora=LoRA(rank=2))
    assert FROZEN in adapted.variables
    factors = sum(leaf.size for leaf in jax.tree.leaves(adapted.variables["params"]))
    assert 0 < factors < sum(leaf.size for leaf in held)


@pytest.mark.parametrize("kind", ["lm", "diffusion", "jepa"])
def test_a_kind_that_builds_its_model_fresh_refuses_source_options(kind, tmp_path):
    """`source` says what a run builds of the source it loads, and a run whose model is built from
    `model` loads none: its source options are refused as it is prepared, not dropped."""
    run = KINDS[kind](tmp_path)
    with pytest.raises(ValueError):
        dataclasses.replace(run, source=SourceOptions(text_only=True)).prepare()


@pytest.mark.parametrize("kind", ["lm", "decision"])
def test_a_kind_whose_source_decides_its_model_refuses_another_model_name(kind, tmp_path):
    """A run that starts from a source takes its architecture from it, and its model's fields'
    few choices alone: a model it names is refused as it is prepared, not dropped."""
    run = KINDS[kind](tmp_path)
    model = ModelConfig("simple_dit", {})
    pretrained = {"pretrained": str(TINY_LAYA)} if kind == "lm" else {}
    with pytest.raises(ValueError):
        dataclasses.replace(run, model=model, **pretrained).prepare()


@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("option", ["xla_flags", "compilation_cache_dir", "multi_host"])
def test_a_process_option_the_process_does_not_carry_is_refused_before_training(kind, option, tmp_path):
    """XLA's flags, the compilation cache and the process pool are set before JAX opens a backend,
    as `run()` and `dew train` do: a run prepared and trained in a process that carries none of the
    one it states is refused before it trains, not trained without it."""
    stated = {"xla_flags": "--xla_cpu_enable_fast_math=false",
              "compilation_cache_dir": str(tmp_path / "cache"), "multi_host": True}[option]
    run = KINDS[kind](tmp_path)
    trainer = TrainerConfig(checkpoint_dir=str(tmp_path / "runs"), steps=1, multi_host=False,
                            **{option: stated})
    prepared = dataclasses.replace(run, trainer=trainer).prepare()
    with pytest.raises(ValueError):
        prepared.train("refused")
    assert not (tmp_path / "runs" / "refused").exists()
