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
from dew.data.pipeline import Rejected
from dew.data.sources.tokens import TokenFiles
from dew.decision import decisions
from dew.decision.config import DecisionRunConfig
from dew.devices import flags_hold
from dew.diffusion.presets import Flow
from dew.inputs import TextCondition
from dew.lora import LoRA
from dew.registry import ModelConfig
from dew.sampling import Euler
from dew.training import Layout, MeshSpec, prepare_process
from dew.training.evaluation import Summary

pytestmark = pytest.mark.mesh

TINY_LAYA = Path(__file__).parent / "fixtures" / "laya" / "tiny"
RATE = 0.0123
"""A learning rate no kind takes by default, which the run's optimizer logs as it steps."""


@dataclasses.dataclass(frozen=True)
class MostlyKept:
    """A record kept by its draw three times in four (`Pipeline.random_filter_map`): which records a
    batch holds is then the run's data seed's."""

    identity = "mostly-kept@1"

    def __call__(self, row, rng: np.random.Generator):
        return row if rng.random() < 0.75 else Rejected("drawn out")


@dataclasses.dataclass(frozen=True)
class Dimmed:
    """An image dimmed by a factor its record draws (`presets.images`' `augment`): what a batch's
    pixels hold is then the run's data seed's."""

    identity = "dimmed@1"

    def __call__(self, row, rng: np.random.Generator):
        image = np.asarray(row["image"])
        return {**row, "image": (image * rng.uniform(0.5, 1.0)).astype(image.dtype)}

    def declared(self, facts):
        # The image keeps its shape and dtype.
        return facts


def read(chain):
    return chain.read("memory").loading(Loading(workers=0))


def lm(root: Path, drawn: bool) -> RunConfig:
    corpus = write_token_files(root / "tokens", 40 * SEQ, 8 * SEQ)

    def windows(path, *, drawn: bool):
        chain = TokenFiles(path).pipe(presets.lm, None, seq_len=SEQ, batch=None, pack=None)
        return read((chain.random_filter_map(MostlyKept()) if drawn else chain).batch(8))
    return LMRunConfig(model=ModelConfig("causal_transformer", {"emb_features": 16, "num_layers": 1,
                                                                "num_heads": 2, "dtype": "float32"}),
                       data=windows(corpus.train, drawn=drawn), validation=windows(corpus.val, drawn=False))


def decision(root: Path, drawn: bool) -> RunConfig:
    table = root / "tickets.csv"
    asked = [("charged twice for March", "billing"), ("the site is down", "technical"),
             ("how much is the pro plan", "sales"), ("refund my last invoice", "billing")]
    # Each ticket its own text, so the held-out split, cut by a row's content, holds some of each.
    tickets = [(f"{text}, ticket {index}", label) for index in range(8) for text, label in asked]
    with table.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])
        writer.writerows(tickets)
    train, held = Files(str(table), format="csv").holdout(0.25)

    def asked(rows, *, drawn: bool):
        chain = rows.pipe(decisions, question="team").read("memory")
        kept = chain.random_filter_map(MostlyKept()) if drawn else chain
        return kept.batch(8).loading(Loading(workers=0))
    return DecisionRunConfig(pretrained=str(TINY_LAYA), model=ModelConfig("causal_transformer",
                                                                          {"dtype": "float32"}),
                             data=asked(train, drawn=drawn), validation=asked(held, drawn=False))


def pictures():
    return TFDS("dew_images", path=str(IMAGES), split="all").holdout(records=4)


def diffusion(root: Path, drawn: bool) -> RunConfig:
    del root
    train, held = pictures()

    def captioned(rows, *, drawn: bool):
        return read(rows.pipe(presets.images, 4, batch=8, caption=image_steps.caption(),
                              augment=Dimmed() if drawn else False))
    return DiffusionRunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 2, "emb_features": 16, "num_layers": 1,
                                         "num_heads": 2}),
        data=captioned(train, drawn=drawn), validation=captioned(held, drawn=False), preset=Flow(),
        val_metrics=(),
        text=TextCondition(encoder="char_table", checkpoint="char_table"),
        objective=ObjectiveConfig("diffusion", {"solver": Euler(), "steps": 2}))


def jepa(root: Path, drawn: bool) -> RunConfig:
    del root
    train, held = pictures()

    def patched(rows, *, drawn: bool):
        return read(rows.pipe(presets.images, 16, batch=8, augment=Dimmed() if drawn else False))
    return JepaRunConfig(
        model=ModelConfig("jepa_encoder", {"patch_size": 2, "emb_features": 16, "num_layers": 1,
                                           "num_heads": 2, "mlp_ratio": 2, "dtype": "float32",
                                           "attention_impl": "xla"}),
        predictor={"num_layers": 1, "predictor_features": 8}, num_target_blocks=2,
        data=patched(train, drawn=drawn), validation=patched(held, drawn=False))


KINDS = {"lm": lm, "decision": decision, "diffusion": diffusion, "jepa": jepa}
ADAPTED = {"lm": ("q_proj",), "decision": ("q_proj",), "diffusion": ("ada_proj",), "jepa": ("up_proj",)}
"""A module of each kind's model a `lora` adapts."""
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
    """`dew train` over a kind's run.json with its options set (`--set`): four steps pooled two to an
    update commit at the second's and the fourth's end (`train/grad_norm`), under a dynamic loss
    scale (`train/loss_scale`) and the learning rate set (`train/learning_rate`); every step is
    logged and scored on the held split, with no metric the kind would score by default and the
    summary named formed from the scores; and every step's checkpoint is kept, four, where the
    default keeps the latest two and the best, in the directory named."""
    record = tmp_path / "record"
    KINDS[kind](tmp_path, drawn=False).save(str(record))
    runs = tmp_path / "runs"
    settings = {"trainer.name": "set", "trainer.checkpoint_dir": str(runs), "trainer.steps": "4",
                "trainer.accumulation": "2", "trainer.log_every": "1", "trainer.eval_every": "1",
                "trainer.checkpoint_every": "1", "trainer.keep": "4", "trainer.dynamic_scale": "True",
                "trainer.multi_host": "False", "trainer.compilation_cache_dir": "None",
                "optim.learning_rate": str(RATE), "val_metrics": "[]",
                "val_summaries": json.dumps([{"name": "board", "weights": {"val/loss": 1},
                                              "better": "lower"}])}
    assert main(["train", str(record / "run.json"),
                 *(flag for path, value in settings.items() for flag in ("--set", f"{path}={value}"))]) == 0
    run = runs / "set"
    logged = scalars(run)
    assert {step for step, row in logged.items() if "train/loss" in row} == {1, 2, 3, 4}
    assert sorted(step for step, row in logged.items() if "train/grad_norm" in row) == [2, 4]
    for step in (1, 2, 3, 4):
        row = logged[step]
        # The optimizer holds its rate in float32, which the log reads back.
        assert row["train/learning_rate"] == float(np.float32(RATE))
        assert "train/loss_scale" in row
        assert row[f"{Summary.SPLIT}/board"] == row["val/loss"]
        if kind in DEFAULT_METRICS:
            assert DEFAULT_METRICS[kind] not in row
    assert sorted(path.name for path in run.iterdir() if path.name.isdigit()) == ["1", "2", "3", "4"]


def trained(run: RunConfig, root: Path, name: str, **fields):
    """`run` trained one step from Python (`prepare().train`) over a mesh of four data shards by two
    fsdp shards, its trainer the fields given: its state, and its step's scalars."""
    trainer = TrainerConfig(checkpoint_dir=str(root / "runs"), steps=1, log_every=1, eval_every=None,
                            checkpoint_every=None, checkpoint_final=False, multi_host=False,
                            mesh=MeshSpec(fsdp=2), layout=Layout(min_shard=64), **fields.pop("trainer", {}))
    state = dataclasses.replace(run, trainer=trainer, **fields).prepare().train(name)
    return state, scalars(root / "runs" / name)[1]


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("kind", list(KINDS))
def test_a_kinds_key_data_seed_mesh_and_adapter_reach_what_it_trains(kind, tmp_path):
    """From Python, a kind's run trains under each option it states: another `trainer.key` is another
    run key in its state, and draws other parameters where the kind draws them fresh (a decision run
    reads its release's), another `data_seed` reads another first batch from parameters drawn alike,
    its mesh and layout split a parameter over its fsdp shards, and a `lora` trains its factors over a
    frozen base."""
    run = KINDS[kind](tmp_path, drawn=True)
    base, first = trained(run, tmp_path, "base", trainer={"key": 1})
    keyed, _ = trained(run, tmp_path, "keyed", trainer={"key": 2})
    _, reordered = trained(run, tmp_path, "seeded", trainer={"key": 1}, data_seed=5)
    held = [np.asarray(leaf) for leaf in jax.tree.leaves(base.variables["params"])]
    assert not np.array_equal(jax.random.key_data(base.key), jax.random.key_data(keyed.key))
    drawn = any(not np.array_equal(mine, np.asarray(theirs))
                for mine, theirs in zip(held, jax.tree.leaves(keyed.variables["params"]), strict=True))
    assert drawn is (kind != "decision")
    assert first["train/loss"] != reordered["train/loss"]
    assert any(len(leaf.sharding.device_set) > 1 and not leaf.sharding.is_fully_replicated
               for leaf in jax.tree.leaves(base.variables["params"]))
    adapted, _ = trained(run, tmp_path, "adapted", trainer={"key": 1},
                         lora=LoRA(rank=2, modules=ADAPTED[kind]))
    assert FROZEN in adapted.variables
    factors = sum(leaf.size for leaf in jax.tree.leaves(adapted.variables["params"]))
    assert 0 < factors < sum(leaf.size for leaf in held)


@pytest.mark.parametrize("kind", ["lm", "diffusion", "jepa"])
def test_a_kind_that_builds_its_model_fresh_refuses_source_options(kind, tmp_path):
    """`source` says what a run builds of the source it loads, and a run whose model is built from
    `model` loads none: its source options are refused as it is prepared, not dropped."""
    run = KINDS[kind](tmp_path, drawn=False)
    with pytest.raises(ValueError):
        dataclasses.replace(run, source=SourceOptions(text_only=True)).prepare()


@pytest.mark.parametrize("kind", ["lm", "decision"])
def test_a_kind_whose_source_decides_its_model_refuses_another_model_name(kind, tmp_path):
    """A run that starts from a source takes its architecture from it, and its model's fields'
    few choices alone: a model it names is refused as it is prepared, not dropped."""
    run = KINDS[kind](tmp_path, drawn=False)
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
    stated = {"xla_flags": "--xla_gpu_deterministic_ops=true",
              "compilation_cache_dir": str(tmp_path / "cache"), "multi_host": True}[option]
    run = KINDS[kind](tmp_path, drawn=False)
    trainer = TrainerConfig(**{"checkpoint_dir": str(tmp_path / "runs"), "steps": 1, "eval_every": None,
                               "multi_host": False, option: stated})
    prepared = dataclasses.replace(run, trainer=trainer).prepare()
    with pytest.raises(ValueError):
        prepared.train("refused")
    assert not (tmp_path / "runs" / "refused").exists()


def test_a_flag_holds_as_xla_reads_it_its_last_occurrence_winning(monkeypatch):
    """A run's XLA flag holds where XLA would read its value: a later occurrence of the flag overrides
    an earlier one, and a bare flag reads as true."""
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_deterministic_ops=true --xla_dump_hlo_as_text "
                                    "--xla_gpu_deterministic_ops=false")
    assert not flags_hold("--xla_gpu_deterministic_ops=true")
    assert flags_hold("--xla_gpu_deterministic_ops=false --xla_dump_hlo_as_text")


def test_flags_set_after_the_backend_opened_are_refused_where_the_process_is_prepared(monkeypatch):
    """XLA reads its flags as JAX opens its backend: once it has, preparing the process with flags
    that do not already hold is refused, not appended to no effect; flags that hold are accepted."""
    jax.devices()
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_hlo_as_text")
    with pytest.raises(ValueError):
        prepare_process(xla_flags="--xla_gpu_deterministic_ops=true")
    assert flags_hold("--xla_dump_hlo_as_text") and not flags_hold("--xla_gpu_deterministic_ops=true")
