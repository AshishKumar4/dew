"""A recorded run reaches the typed options the library has (theme T57, rows 2239-2245).

Each option `Trainer.fit`, `Checkpoints`, `EvalSuite`, Optax, `TextGeneration.from_run` or the attention
template takes is set where a run is set, on its command line, in its `run.json` or with `dew train
--set`, and observed where it acts: the step a plateau stops at and the model a run ends on, the steps a
split is scored at, the local checkpoints written, the step an export reads, the updates an optimizer
makes, the placement a harness loads into, and the form a run's attention trains in."""

import dataclasses
import json

import numpy as np
import optax
import pytest
from test_block_sft import assert_tree_equal
from test_lm_recipe import SEQ, export_tiny_decoder, recipe_args, write_token_files
from test_run_option_matrix import recorded
from test_run_options import decision, lm, scalars, trained

from dew.checkpoints import Checkpoints
from dew.cli.main import main
from dew.config.lm import LMRunConfig
from dew.data import Loading, presets
from dew.data.sources.tokens import TokenFiles
from dew.optim import OneCycle, OptimConfig, ParamGroup, optimizers
from dew.registry import ModelConfig
from dew.training._lowering import Lowered

pytestmark = pytest.mark.mesh


TINY = '{"emb_features": 16, "num_layers": 1, "num_heads": 2}'


def descending(tokens, *flags: str, pretrained=None) -> LMRunConfig:
    """A language model's run from its command line whose every batch and whole validation split are
    the training store's first eight windows: each step descends the loss its next evaluation scores,
    at a rate small enough that the step lowers it, so the first evaluation scores the highest
    perplexity and a ranking of the highest keeps the first step, whatever the draw."""
    windows = TokenFiles(tokens.train).pipe(presets.lm, None, seq_len=SEQ, batch=None, pack=None).take(8)
    default = LMRunConfig(data=windows.repeat().batch(8).loading(Loading(workers=0)),
                          validation=windows.batch(8).loading(Loading(workers=0)))
    source = () if pretrained is None else ("--pretrained", str(pretrained))
    return LMRunConfig.cli(recipe_args(tokens, *source, *flags, "--trainer.best",
                                       json.dumps({"metric": "val/perplexity", "mode": "max"}),
                                       model_config="{}" if pretrained else TINY), default=default)


def test_a_run_stops_on_its_plateau_and_ends_on_its_best_model(tmp_path):
    """`trainer.stop:plateau` stops a run once the metric it names, as `--trainer.best` names one, has
    not improved past `min_delta` for `evals` evaluations, and `--trainer.restore-best` ends the run on
    the model its ranking chose (row 2239): a plateau of one evaluation that nothing improves past
    stops a run of eight steps at its second evaluation, and the run ends on the first step, whose
    perplexity a ranking of the highest keeps (`descending`), with that checkpoint's weights."""
    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ)
    state = descending(
        tokens, "--trainer.name", "plateau", "--trainer.steps", "8", "--trainer.eval-every", "1",
        "--trainer.checkpoint-every", "1", "--trainer.keep", "8", "--trainer.restore-best",
        "trainer.stop:plateau", "--trainer.stop.metric", "val/perplexity", "--trainer.stop.evals", "1",
        "--trainer.stop.min-delta", "1e9").run()
    checkpoints = Checkpoints(str(tmp_path / "runs" / "plateau"))
    assert checkpoints.latest == 2
    assert int(state.step) == 1
    assert_tree_equal(state.variables["params"], checkpoints.variables(step=1)["params"])


def test_a_saved_run_scores_a_split_at_its_own_cadence_and_writes_its_local_tier(tmp_path):
    """`dew train --set` over a run.json scores each split at its own cadence, `val_every` naming one's
    and the others at `trainer.eval_every` (row 2240), stops on a plateau of a split's metric named by
    its record (row 2239), and writes the local checkpoint tier its `trainer.local_directory` and
    `trainer.local_every` name (row 2241): of eight steps scored every step, the test split is scored
    every second, a plateau of one test evaluation that nothing improves past stops the run at the
    fourth, and the host's own directory holds that step under the run's name, which another run
    sharing the directory does not resume."""
    record, runs, local = tmp_path / "record", tmp_path / "runs", tmp_path / "local"
    run = lm(tmp_path, drawn=False)
    dataclasses.replace(run, validation={"val": run.validation, "test": run.validation}).save(str(record))
    plateau = {"class": "dew.training.budget:Plateau",
               "fields": {"metric": "test/perplexity", "evals": 1, "min_delta": 1e9}}
    settings = {"trainer.checkpoint_dir": str(runs), "trainer.steps": "8", "trainer.log_every": "1",
                "trainer.eval_every": "1", "val_every": json.dumps({"test": 2}),
                "trainer.stop": json.dumps(plateau), "trainer.checkpoint_every": "None",
                "trainer.local_directory": str(local), "trainer.local_every": "1",
                "trainer.multi_host": "False", "trainer.compilation_cache_dir": "None"}
    flags = [flag for path, value in settings.items() for flag in ("--set", f"{path}={value}")]
    for name in ("cadenced", "other"):
        assert main(["train", str(record / "run.json"), "--set", f"trainer.name={name}", *flags]) == 0
    logged = scalars(runs / "cadenced")
    assert sorted(step for step, row in logged.items() if "val/loss" in row) == [1, 2, 3, 4]
    assert sorted(step for step, row in logged.items() if "test/loss" in row) == [2, 4]
    assert max(int(path.name) for path in (local / "cadenced" / "process0").iterdir()
               if path.name.isdigit()) == 4
    assert min(scalars(runs / "other")) == 1


def test_dew_export_reads_the_step_a_ranking_chose(tmp_path):
    """`dew export --step best` exports the step the run's ranking kept best, not its latest (row
    2242): ranked by the highest validation perplexity, which the first of three steps scores
    (`descending`), the export reads back as that step's weights, leaf for leaf."""
    from dew.interop import Pretrained

    tokens = write_token_files(tmp_path / "tokens", 40 * SEQ, 8 * SEQ, eos_id=0)
    checkpoint = export_tiny_decoder(tmp_path / "checkpoint")
    descending(tokens, "--trainer.name", "ranked", "--trainer.steps", "3", "--trainer.eval-every", "1",
               "--trainer.checkpoint-every", "1", "--trainer.keep", "3", "--max-new-tokens", "0",
               pretrained=checkpoint).run()
    run, exported = tmp_path / "runs" / "ranked", tmp_path / "exported"
    assert main(["export", str(run), str(exported), "--step", "best"]) == 0
    checkpoints, best = Checkpoints(str(run)), 1
    assert checkpoints.latest == 3
    loaded = Pretrained.load(str(exported), dtype="float32", attention_impl="reference").variables["params"]
    assert_tree_equal(loaded, checkpoints.variables(step=best)["params"])


def test_an_optax_factory_named_by_its_path_makes_its_own_updates_in_a_saved_run(tmp_path):
    """`OptimConfig(optimizer="optax:sgd")` builds Optax's SGD with the options the config gives, its
    updates the factory's own over three steps, and a saved run trains on it through `dew train`, where
    the optimizer was one of five names (row 2243)."""
    config = OptimConfig(optimizer="optax:sgd", optimizer_opts={"momentum": 0.9}, learning_rate=0.05)
    params = {"w": np.linspace(-1, 1, 6, dtype=np.float32).reshape(2, 3)}
    gradients = [{"w": np.full((2, 3), scale, np.float32)} for scale in (0.5, -1.0, 2.0)]

    def updates(solver):
        state, found = solver.init(params), []
        for gradient in gradients:
            update, state = solver.update(gradient, state, params)
            found.append(update)
        return found
    assert_tree_equal(updates(config.build(10)), updates(optax.sgd(0.05, momentum=0.9)))

    # A group's momentum schedule needs a factory that takes b1, which Optax's adamw names.
    grouped = OptimConfig(optimizer="optax:adamw", param_groups=(ParamGroup("all", ("*",), b1=OneCycle(
        peak_value=0.85, init_value=0.95, end_value=0.95)),)).build(10)
    grouped.update(gradients[0], grouped.init(params), params)

    record, runs = tmp_path / "record", tmp_path / "runs"
    dataclasses.replace(lm(tmp_path, drawn=False), optim=config).save(str(record))
    settings = {"trainer.name": "sgd", "trainer.checkpoint_dir": str(runs), "trainer.steps": "2",
                "trainer.log_every": "1", "trainer.eval_every": "None", "trainer.checkpoint_every": "None",
                "trainer.multi_host": "False", "trainer.compilation_cache_dir": "None"}
    assert main(["train", str(record / "run.json"),
                 *(flag for path, value in settings.items() for flag in ("--set", f"{path}={value}"))]) == 0
    assert all(row["train/learning_rate"] == float(np.float32(0.05)) for row in scalars(runs / "sgd").values()
               if "train/learning_rate" in row)


@pytest.mark.mesh(devices=8)
def test_a_runs_recorded_attention_form_is_the_form_it_trains_in(tmp_path):
    """`trainer.attention_form` is the form every attention of the run trains in (row 2245), on the
    default route ('auto'): lowered for an H100, a 16-bit run recorded "accurate" routes each attention
    to the template or the reference path, which compute the form, where the same run recorded with
    none routes it to an old route (row 2350), and `dew train` of an accurate run's own record, under a
    new name, runs its attention on the kernels it ran."""
    run = lm(tmp_path, drawn=False)

    def kernels(name: str) -> set[str]:
        (compiled, *_) = recorded(tmp_path / "runs" / name, "StepCompiled")
        return {route["kernel"] for route in compiled["plan"]["kernels"]["routes"]
                if route["choice"] == "attention"}

    # On an H100 target, which its step is lowered for (`--trainer.lower-for`), a 16-bit run takes an old
    # route in the fast form, so the routes it lowers say which form the run took.
    sixteen = dataclasses.replace(run, model=ModelConfig(run.model.name, {**run.model.fields,
                                                                          "dtype": "bfloat16"}))

    def lowered(name: str, fields) -> set[str]:
        with pytest.raises(Lowered) as raised:
            trained(sixteen, tmp_path, name, trainer={**fields, "lower_for": "sm90"})
        return {route["kernel"] for route in raised.value.kernels.record()["routes"]
                if route["choice"] == "attention"}

    accurate = lowered("accurate", {"attention_form": "accurate"})
    assert accurate and accurate <= {"template", "reference"}
    assert not lowered("fast", {}) & {"template", "reference"}
    trained(run, tmp_path, "cpu", trainer={"attention_form": "accurate"})
    recorded_run = tmp_path / "runs" / "cpu" / "run.json"
    assert main(["train", str(recorded_run), "--set", "trainer.name=again"]) == 0
    assert kernels("again") == kernels("cpu")


def test_a_record_naming_a_function_for_its_optimizer_is_refused_before_it_is_called(tmp_path):
    """A run record's optimizer names an import path, which must be declared to return an Optax gradient
    transformation before it is called (row 2349: `os:spawnv` ran a command before its result was
    checked): `os:mkdir` is refused, its directory never made, and Optax's LARS, which takes a decay
    and no decay mask, builds with one (row 2351)."""
    made = tmp_path / "made"
    with pytest.raises(ValueError):
        OptimConfig(optimizer="os:mkdir", optimizer_opts={"path": str(made)}).build(4)
    assert not made.exists()
    params = {"w": np.ones((2, 3), np.float32)}
    # Dew's own optimizers name the same factory by their alias and by their import path.
    assert optimizers["adamw"] is optimizers[optimizers.paths["adamw"]]
    solver = OptimConfig(optimizer="optax:lars", learning_rate=0.1, weight_decay=1e-4).build(4)
    update, _ = solver.update({"w": np.full((2, 3), 0.5, np.float32)}, solver.init(params), params)
    assert np.all(np.asarray(update["w"]) < 0)


def test_a_decision_run_that_ends_on_its_best_loads_at_its_calibrated_step(tmp_path):
    """A decision run that ends on its best step (`--trainer.restore-best`) calibrates that step, and its
    saved task loads there by default, through `Decide.from_run` and `dew.pipeline` alike, where the
    run's latest checkpoint is its last step (row 2352: the load refused the calibration's step)."""
    import dew
    from dew.decision.task import Decide
    from dew.training.selection import Best

    runs = tmp_path / "runs"
    run = decision(tmp_path, drawn=False)
    trainer = dataclasses.replace(run.trainer, name="ended", checkpoint_dir=str(runs), steps=3, eval_every=1,
                                  checkpoint_every=1, keep=3, restore_best=True, multi_host=False,
                                  compilation_cache_dir=None, best=Best("val/accuracy", mode="min"))
    dataclasses.replace(run, trainer=trainer).prepare().train("ended")
    checkpoints = Checkpoints(str(runs / "ended"))
    best = checkpoints.resolve("best")
    assert best != checkpoints.latest, "the last step ranked best: the default step is the latest"
    for loaded in (Decide.from_run(str(runs / "ended")), dew.pipeline(str(runs / "ended"))):
        assert loaded.weights.step == best


def test_a_run_recorded_before_its_stop_took_a_plateau_still_pauses(tmp_path):
    """A run.json written while `trainer.stop` held a Pause alone records the pause by its fields, with no
    class; `dew train` of it pauses where it says (row 2239's review): half of four steps."""
    record, runs = tmp_path / "record", tmp_path / "runs"
    lm(tmp_path, drawn=False).save(str(record))
    written = json.loads((record / "run.json").read_text())
    written["fields"]["trainer"]["stop"] = {"fraction": 0.5}
    (record / "run.json").write_text(json.dumps(written))
    settings = {"trainer.name": "paused", "trainer.checkpoint_dir": str(runs), "trainer.steps": "4",
                "trainer.log_every": "1", "trainer.eval_every": "None", "trainer.checkpoint_every": "None",
                "trainer.multi_host": "False", "trainer.compilation_cache_dir": "None"}
    assert main(["train", str(record / "run.json"),
                 *(flag for path, value in settings.items() for flag in ("--set", f"{path}={value}"))]) == 0
    assert max(scalars(runs / "paused")) == 2
