"""Options of the common run layer reach what consumes them in every kind of run, part of row 1901.

Seven of twenty defects a run's consumers found on GPUs were options a kind of run or the command line
did not hand on (rows 1610, 1772, 1786, 1884, 1889, 1894, 1895). Every kind builds its objective, its
data and its metrics from `RunConfig`'s fields (`prepare`), and trains through `RunConfig.train`, where
alone the trainer, `fit`'s arguments, the checkpoints and the tracker are built from its
`TrainerConfig`. Each kind here, a language model's, a decision's, a diffusion model's and a JEPA's, is
set these and observed where each acts, in the scalars and checkpoints its run writes and the state it
ends on: on the command line, its name and checkpoint directory, steps, accumulation, cadences of logs,
evaluations and checkpoints, checkpoints kept, dynamic loss scale, learning rate, validation metrics and
summaries; from Python, its key, data seed, mesh, layout and adapter. An option a kind builds nothing
from (source options of a fresh model, a model name its source decides, process options its process
does not carry) is refused before it trains, rather than dropped. The trainer's other options, each kind
over, are row 1901's still. Nothing here reads a config's fields back.

Each option a saved task is built with (its objective's `settings`) is derived from its type and passed
through `dew train` to the task the run saves (row 2285, which would have caught row 1061's blockwise
walk, an option with no public path)."""

import dataclasses
import json

import jax
import numpy as np
import pytest
from option_witness import SETTINGS, TASK_RUNS, same, saved_with, witnessed_on
from run_kinds import ADAPTED, DEFAULT_METRICS, KINDS, RATE, TINY_LAYA, decision, dreamed, scalars, trained

from dew.batch import FROZEN
from dew.cli.main import main
from dew.config import SourceOptions, TrainerConfig
from dew.lora import LoRA
from dew.registry import ModelConfig, objectives
from dew.training.evaluation import Summary

pytestmark = pytest.mark.mesh



@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("kind", list(KINDS))
def test_a_kinds_run_set_on_the_command_line_pools_logs_scores_and_keeps_as_set(kind, tmp_path):
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
                "trainer.checkpoint_every": "1", "trainer.keep": "4", "model.dtype": "float16",
                "trainer.multi_host": "False", "trainer.compilation_cache_dir": "None",
                "optim.learning_rate": str(RATE), "val_metrics": "[]",
                "val_summaries": json.dumps([{"name": "board", "weights": {"val/loss": 1},
                                              "better": "lower"}])}
    assert main(["train", str(record / "run.json"),
                 *(flag for path, value in settings.items() for flag in ("--set", f"{path}={value}"))]) == 0
    run = runs / "set"
    logged = scalars(run)
    assert {step for step, row in logged.items() if "train/loss" in row} == {1, 2, 3, 4}
    committed = sorted(step for step, row in logged.items() if "train/grad_norm" in row)
    # Its model computes in float16, which scales the loss (`DynamicScale`): a commit whose gradients
    # overflow skips its update and lowers the scale, as a loaded backbone's first may.
    assert set(committed) <= {2, 4} and 4 in committed
    if 2 not in committed:
        assert logged[2]["train/loss_scale"] < logged[1]["train/loss_scale"]
    for step in (1, 2, 3, 4):
        row = logged[step]
        # The optimizer holds its rate in float32, which the log reads back.
        assert row["train/learning_rate"] == float(np.float32(RATE))
        assert "train/loss_scale" in row
        assert row[f"{Summary.SPLIT}/board"] == row["val/loss"]
        if kind in DEFAULT_METRICS:
            assert DEFAULT_METRICS[kind] not in row
    assert sorted(path.name for path in run.iterdir() if path.name.isdigit()) == ["1", "2", "3", "4"]


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
    # A flag value no lane sets (tests/lane_environment.py), so this process carries none of it.
    stated = {"xla_flags": "--xla_dump_hlo_pass_re=run-options-1901",
              "compilation_cache_dir": str(tmp_path / "cache"), "multi_host": True}[option]
    run = KINDS[kind](tmp_path, drawn=False)
    trainer = TrainerConfig(**{"checkpoint_dir": str(tmp_path / "runs"), "steps": 1, "eval_every": None,
                               "multi_host": False, option: stated})
    prepared = dataclasses.replace(run, trainer=trainer).prepare()
    with pytest.raises(ValueError):
        prepared.train("refused")
    assert not (tmp_path / "runs" / "refused").exists()



# ----------------------------------------------------------- options a saved task is built with (row 2285)

@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("settings", SETTINGS, ids=lambda settings: settings.__name__)
def test_each_option_a_saved_task_is_built_with_reaches_it_from_the_command_line(settings, tmp_path):
    """Every option a saved task is built with that its settings' types declare alternatives for
    (`alternatives`: a literal's values, an optional's, a flag's, Dew's classes that are one, a saved
    dataclass with one field changed) reaches the task `dew train` saves from its path (`reach`): beside
    a run that sets none, a run sets each alone to an alternative its settings accept other than the one
    saved (the next where one does not witness it, `witnessed_by`), and the task saves it and, loaded,
    holds it where the first held another, and a request of it answers apart from the same task holding
    there what it held before (`unread`), the consumer's work, not its record. An option with no path is row
    1061's blockwise walk,
    unless its settings declare it derived from the run (`metadata={"derived": ...}`): from its data,
    which the run fills, or a value the run builds of fields of its own (a decision run's layout), whose
    fields are its options (`options_of`). A field of Dew's types that declares fewer than two
    alternatives fails, unless derived or a record (`unwitnessed`). Every settings type an objective
    saves (`SETTINGS`) has a run here (`TASK_RUNS`), and each option is witnessed on one of them that
    holds it (`witnessed_on`).

    A request of a loaded task answers alike twice (`answered_alike`), so an answer apart is the
    option's alone."""
    assert settings.__name__ in TASK_RUNS, f"no run here trains an objective whose task reads {settings}"
    from dew.objectives.base import ReleasedTask

    # Each run witnesses the options no run before it did, each option set once on the runs that hold it.
    witnessed, done = [], frozenset()
    for builder, objective in TASK_RUNS[settings.__name__]:
        witnessed.append(witnessed_on(builder, objective, settings, tmp_path / builder.__name__, done))
        options, failures, _ = witnessed[-1]
        done |= {name for name in options if name not in failures}
    found = [(options, failures) for options, failures, _ in witnessed]
    # An option is witnessed on a run that holds it, which a release's settings may not (a split's).
    missed = {name: [failures[name] for options, failures in found if name in options]
              for name in {name for options, _ in found for name in options}
              if all(name not in options or name in failures for options, failures in found)}
    assert not missed, f"{settings.__name__}'s options no run here witnesses: {missed}"
    # A released task's options reach an export on one of its runs at least.
    saved_task = objectives[TASK_RUNS[settings.__name__][0][1]].saved_task
    assert not (saved_task is not None and issubclass(saved_task, ReleasedTask)) or any(
        exports for _, _, exports in witnessed), f"no run of {settings.__name__}'s exports"


@pytest.mark.parametrize("release", ["dream", "diffucoder"])
def test_a_masked_run_over_a_release_saves_and_reads_the_walk_the_release_declares(release, tmp_path):
    """A masked-diffusion run over a Dream release (dream-tiny as the release ships it, its generation
    config and its own sampling code, `test_dream_walk.snapshot`), setting nothing of its task, saves the
    walk the release's own task declares (`Pretrained.text_generation`): its process, the grid's endpoint
    among it (ledger 2563), its solver, its steps and its blocks; and the grid its loaded task walks runs
    to that endpoint, a loaded source's default reaching its consumer. A walk's tokens show the endpoint
    only where a last step leaves a place masked, which no request here reaches."""
    from dew.inference.tasks import MaskedSettings
    from dew.interop import Pretrained

    run = dreamed(tmp_path, drawn=False, release=release)
    source = Pretrained.load(tmp_path / release, dtype="float32", attention_impl="reference")
    declared = source.text_generation()
    _, task = saved_with(run, tmp_path, "over", MaskedSettings, {})
    options = {name: getattr(declared, name) for name in ("process", "solver", "steps", "block")}
    assert {name: same(getattr(task, name), option) for name, option in options.items()} == dict.fromkeys(
        options, True)
    # The grid its walk takes runs to that endpoint (`DiscreteProcess.times`), not MDLM's.
    unended = dataclasses.replace(task.process, endpoint=None)
    assert not np.array_equal(task.process.times(task.steps), unended.times(task.steps))


def test_a_decision_run_over_a_release_saves_and_reads_the_task_the_release_declares(tmp_path):
    """A decision run over a release (tiny Laya), setting nothing of its task, saves the task the
    release's own declares (`Decide.from_pretrained`): its layout, its special tokens and its confidence,
    its budget the run's data's (`DecideSettings.questions`). What reads each, the gate above witnesses
    (`witnessed_on`)."""
    from dew.decision.task import Decide, DecideSettings

    declared = Decide.from_pretrained(TINY_LAYA)
    _, task = saved_with(decision(tmp_path, drawn=False), tmp_path, "over", DecideSettings, {})
    options = {name: getattr(declared, name) for name in ("layout", "specials", "confidence")}
    assert {name: same(getattr(task, name), option) for name, option in options.items()} == dict.fromkeys(
        options, True)
