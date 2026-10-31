"""Live inspection (row 2317): what a run watches is summarized inside its step, from the arrays its commit
holds, and logged one log late; a run that watches nothing compiles its step unchanged."""

import contextlib
import json

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import test_run_budgets
from run_kinds import KINDS, lm, scalars

from dew.cli.main import main
from dew.data.dataset import Loader
from dew.nn.inspection import statistics
from dew.telemetry.instrumentation import Compiles
from dew.training import Inspection, Trainer, inspected

BATCH, RATE = 8, 0.25
FEATURES = test_run_budgets.FEATURES
ROWS = np.random.default_rng(0).normal(size=(BATCH, FEATURES)).astype(np.float32)
STATISTICS = ("mean", "rms", "abs_max")
QUARTERS = (np.random.default_rng(0).integers(-8, 9, size=(BATCH, FEATURES)) / 4).astype(np.float32)
"""Rows of quarters, over which every sum a summary of the step takes is exact in float32: no order of
summation moves it, so the step's summaries are NumPy's, bit for bit."""


class Pulled(test_run_budgets.Pulled):
    """`test_run_budgets`' weight pulled toward the batch's rows, whose gradient is 2 (w - the rows' mean)
    / FEATURES, starting at 1, 2, ... FEATURES, where no entry is zero."""

    def fresh_variables(self, key, held):
        return {"params": {"w": jnp.arange(1, FEATURES + 1, dtype=jnp.float32)}}


class Kept:
    def __init__(self):
        self.rows: dict[int, dict[str, float]] = {}

    def log(self, scalars, step):
        self.rows.setdefault(step, {}).update(scalars)

    def artifact(self, value, step):
        pass

    def close(self):
        pass


def stream(rows=ROWS):
    while True:
        yield {"x": rows}


def trainer(inspection, tracker=None):
    return Trainer(Pulled(), optax.sgd(RATE), key=jax.random.key(0), tracker=tracker, inspection=inspection)


def trained(inspection, steps, rows=ROWS) -> dict[int, dict[str, float]]:
    tracker = Kept()
    trainer(inspection, tracker).fit(Loader(lambda partition: stream(rows), None, batch=BATCH, records=None),
                                     steps=steps, log_every=1, eval_every=None, checkpoint_every=None)
    return tracker.rows


def summary(values):
    """`dew.training.inspection`'s statistics of `values`, in their own dtype."""
    return {"mean": values.mean(), "rms": np.sqrt(np.square(values).mean()), "abs_max": np.abs(values).max()}


def test_a_sampled_steps_summaries_are_its_commits_arrays():
    """A step sampled for `w` reports, for the parameters its update applies to, the gradient the
    optimizer is handed, the update and the parameters' change, each statistic NumPy's in float32 of
    the same arrays, bit for bit over rows whose every sum is exact (`QUARTERS`), its L2 norms and its
    update-to-weight ratio too, and no non-finite or zero entries."""
    logged = trained(Inspection(paths=("w",), every=1), 1, rows=QUARTERS)[1]
    weight = np.arange(1, FEATURES + 1, dtype=np.float32)
    gradient = 2 * (weight - QUARTERS.mean(0)) / FEATURES
    update = -np.float32(RATE) * gradient
    arrays = {"params": weight, "grads": gradient, "updates": update, "deltas": (weight + update) - weight}
    expected = {"inspect/update_to_weight": np.linalg.norm(update) / np.linalg.norm(weight)}
    for target, values in arrays.items():
        expected.update({f"inspect/{target}/w/{name}": summary(values)[name] for name in STATISTICS})
        expected.update({f"inspect/{target}/l2": np.linalg.norm(values),
                         f"inspect/{target}/w/non_finite": 0, f"inspect/{target}/w/zero_fraction": 0})
    assert {name: logged[name] for name in expected} == {
        name: float(value) for name, value in expected.items()}


def test_a_run_logs_each_sample_once_at_its_cadence():
    """Sampled every second step and logged every step, a run's logs carry a sample at the second and
    fourth steps alone."""
    rows = trained(Inspection(every=2), 4)
    assert sorted(step for step, row in rows.items() if "inspect/params/l2" in row) == [2, 4]


@pytest.mark.parametrize("inspection", [None, Inspection(paths=("w",), every=1)], ids=["off", "on"])
def test_a_run_that_inspects_nothing_compiles_its_step_alone(inspection):
    """A run that inspects nothing compiles one training step and logs no summary; one that inspects
    compiles its sampled step beside it, once."""
    with Compiles("step") as compiles:
        rows = trained(inspection, 2)
    assert compiles.count == (1 if inspection is None else 2)
    summarized = any(name.startswith("inspect/") for row in rows.values() for name in row)
    assert summarized == (inspection is not None)


def test_a_loaded_runs_view_is_the_live_runs():
    """The view of one sampled step offline (`inspected`), from the trainer's start, is what a live run
    logs at its first step, bit for bit."""
    inspection = Inspection(paths=("w",), every=1)
    live = trained(inspection, 1)[1]
    offline = inspected(trainer(inspection), {"x": ROWS})
    assert offline == {name: value for name, value in live.items() if name.startswith("inspect/")}


def test_a_custom_step_refuses_an_inspection():
    """A custom step (`Trainer(step=...)`) commits no update an inspection reads, so asking for one is
    refused as the trainer is built, not dropped."""
    with pytest.raises(ValueError):
        Trainer(Pulled(), optax.sgd(RATE), key=jax.random.key(0), inspection=Inspection(),
                step=lambda objective, optimizer: None)


def test_a_non_finite_parameter_shows_in_its_summary():
    """A parameter holding an infinity reports it in its non-finite share, one entry in four, and its
    largest magnitude, not a finite stand-in."""
    class Infinite(Pulled):
        def fresh_variables(self, key, held):
            return {"params": {"w": jnp.asarray([jnp.inf, 1.0, 2.0, 3.0])}}

    logged = inspected(Trainer(Infinite(), optax.sgd(RATE), key=jax.random.key(0),
                               inspection=Inspection(paths=("w",), targets=("params",))), {"x": ROWS})
    assert logged["inspect/params/w/non_finite"] == 0.25 and logged["inspect/params/w/abs_max"] == np.inf


def test_an_offline_view_builds_a_configured_optimizer():
    """`inspected` builds a trainer's `OptimConfig` over the run's `steps`, as `fit` does, and reports
    the sampled step's update: Adam's first as that config builds it over those steps (`OptimConfig.build`)
    of the same gradient, its largest magnitude bit for bit, an elementwise result no order of summation
    moves, where SGD's would be the gradient's quarter."""
    from dew.optim import OptimConfig

    config = OptimConfig(optimizer="adam", learning_rate=RATE)
    configured = Trainer(Pulled(), config, key=jax.random.key(0),
                         inspection=Inspection(paths=("w",), targets=("updates",)))
    logged = inspected(configured, {"x": QUARTERS}, steps=4)
    weight = jnp.arange(1, FEATURES + 1, dtype=jnp.float32)
    gradient = 2 * (weight - jnp.asarray(QUARTERS).mean(0)) / FEATURES
    solver = config.build(4)
    update = solver.update({"w": gradient}, solver.init({"w": weight}), {"w": weight})[0]["w"]
    assert logged["inspect/updates/w/abs_max"] == float(jnp.abs(update).max())


def test_a_recorded_run_inspects_what_its_command_line_names(tmp_path):
    """`dew train --set trainer.inspection=...` over a run.json samples what it names at its cadence."""
    record, runs = tmp_path / "record", tmp_path / "runs"
    lm(tmp_path, drawn=False).save(str(record))
    inspection = {"class": "dew.training.inspection:Inspection",
                  "fields": {"paths": ["*q_proj*"], "every": 2}}
    settings = {"trainer.name": "inspected", "trainer.checkpoint_dir": str(runs), "trainer.steps": "2",
                "trainer.log_every": "1", "trainer.eval_every": "None", "trainer.checkpoint_every": "None",
                "trainer.inspection": json.dumps(inspection), "trainer.multi_host": "False",
                "trainer.compilation_cache_dir": "None"}
    assert main(["train", str(record / "run.json"),
                 *(flag for path, value in settings.items() for flag in ("--set", f"{path}={value}"))]) == 0
    logged = scalars(runs / "inspected")
    assert "inspect/grads/l2" in logged[2] and "inspect/grads/l2" not in logged[1]
    assert any("q_proj" in name and name.endswith("/rms") for name in logged[2])


def test_a_saved_runs_view_from_the_command_line_is_what_it_logs(tmp_path, capsys):
    """`dew inspect` over a run.json views one sampled update over the run's first batch, from its
    checkpoints' latest where it has them, else its start (`RunConfig.inspected`): before the run trains, the
    view its first step logs, bit for bit; it trains nothing and writes nothing (2317's offline view from
    the command line, train 332's review)."""
    record, runs = tmp_path / "record", tmp_path / "runs"
    lm(tmp_path, drawn=False).save(str(record))
    inspection = {"class": "dew.training.inspection:Inspection",
                  "fields": {"paths": ["*q_proj*"], "every": 1}}
    settings = {"trainer.name": "viewed", "trainer.checkpoint_dir": str(runs), "trainer.steps": "1",
                "trainer.log_every": "1", "trainer.eval_every": "None", "trainer.checkpoint_every": "None",
                "trainer.inspection": json.dumps(inspection), "trainer.multi_host": "False",
                "trainer.compilation_cache_dir": "None"}
    flags = [flag for path, value in settings.items() for flag in ("--set", f"{path}={value}")]
    capsys.readouterr()
    assert main(["inspect", str(record / "run.json"), *flags]) == 0
    offline = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert not runs.exists()
    assert main(["train", str(record / "run.json"), *flags]) == 0
    live = {name: value for name, value in scalars(runs / "viewed")[1].items() if name.startswith("inspect/")}
    assert live and offline == live


class Block(nn.Module):
    """A residual tanh layer, whose output a run watches by its kind."""

    @nn.compact
    def __call__(self, x, _=None):
        return x + jnp.tanh(nn.Dense(x.shape[-1], name="dense")(x)), None


class Stacked(nn.Module):
    """Three blocks, run as a remat-scanned stack (`nn.scan` over `nn.remat`) or one after another."""

    scanned: bool

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        if self.scanned:
            scan = nn.scan(nn.remat(Block), variable_axes={True: 0}, split_rngs={True: True}, length=3)
            x, _ = scan(name="blocks")(x, None)
        else:
            for _ in range(3):
                x, _ = Block()(x)
        return nn.Dense(1, name="head")(x)


def watched_run(model, inspection):
    from dew.inputs import Field, InputSpec
    from dew.objectives.supervised import Supervised

    objective = Supervised(model, lambda outputs, batch: jnp.square(outputs[:, 0]),
                           inputs=InputSpec(Field("x", (FEATURES,))))
    return Trainer(objective, optax.sgd(RATE), key=jax.random.key(0), inspection=inspection)


def outputs_of(run, layers):
    """Each block's output on the rows, from the run's own start, as the step's forward computes it."""
    state, _, _ = run.place()
    def blocks(module, name):
        return isinstance(module, Block)
    _, captured = run.objective.model.apply(jax.device_get(state.variables), jnp.asarray(ROWS),
                                            capture_intermediates=blocks, mutable=["intermediates"])
    leaves = jax.tree_util.tree_flatten_with_path(captured["intermediates"])[0]
    found = [np.asarray(leaf) for path, leaf in leaves if "__call__" in jax.tree_util.keystr(path)]
    # A scanned stack captures its blocks' outputs stacked on a first axis, in one leaf.
    return list(found[0]) if len(found) == 1 else found[:layers]


@pytest.mark.parametrize("scanned", [False, True], ids=["unrolled", "remat-scanned"])
def test_a_sampled_step_summarizes_each_watched_modules_output(scanned):
    """A sampled step summarizes the output of each module of a watched kind as its forward computed it:
    each block's, one a layer under a remat-scanned stack, its largest magnitude, non-finite share and
    zero fraction those of that very output to the bit, its mean and deviation beside them, and nothing
    of a kind it does not watch (the head)."""
    run = watched_run(Stacked(scanned=scanned), Inspection(modules=("Block",), targets=("params",)))
    logged = inspected(run, {"x": ROWS})
    blocks = sorted({name.rsplit("/", 1)[0] for name in logged if name.startswith("inspect/activations/")})
    assert len(blocks) == 3 and not any("head" in name for name in logged if "activations" in name)
    for prefix, values in zip(blocks, outputs_of(run, 3), strict=True):
        # The statistics no reduction order changes, over the captured output, identify that very tensor;
        # its mean and deviation come of the one `statistics` a parameter's summary is gated through.
        expected = {name: float(value) for name, value in statistics(jnp.asarray(values)).items()}
        for statistic in ("abs_max", "non_finite", "zero_fraction"):
            assert logged[f"{prefix}/{statistic}"] == expected[statistic], statistic
        summarized = {name.rsplit("/", 1)[1] for name in logged if name.startswith(prefix)}
        assert {"mean", "std", "rms"} <= summarized


def test_a_watched_histogram_sums_to_its_whole_output():
    """An opt-in histogram reports the share of a watched output's magnitudes in each decade, one scalar
    a bin, which together are the whole output."""
    watching = Inspection(modules=("Block",), targets=("params",), histogram=True)
    run = watched_run(Stacked(scanned=False), watching)
    logged = inspected(run, {"x": ROWS})
    bins = np.array([value for name, value in logged.items() if "/Block_0/histogram_" in name])
    # Each bin a count over the output's entries: whole counts that sum to every entry.
    counts = np.rint(bins * ROWS.size).astype(int)
    assert len(bins) > 1 and counts.sum() == ROWS.size


def test_watching_a_kind_no_module_is_refused():
    """A run watching a kind no module of its model is, which would log nothing it asked for, is refused
    at its sampled step's trace, naming the kinds."""
    with pytest.raises(ValueError):
        unmatched = Inspection(modules=("*Attention",), targets=("params",))
        inspected(watched_run(Stacked(scanned=False), unmatched), {"x": ROWS})


@pytest.mark.parametrize("kind", list(KINDS))
def test_every_kinds_run_logs_the_activations_its_command_line_watches(kind, tmp_path):
    """A run of every kind watching module kinds on its command line (`trainer.inspection.modules`)
    logs a summary of each watched module's output at its sampled step, its mean, standard deviation,
    largest magnitude and non-finite share, and none at a step it does not sample."""
    record, runs = tmp_path / "record", tmp_path / "runs"
    (tmp_path / "data").mkdir()
    KINDS[kind](tmp_path / "data", drawn=False).save(str(record))
    inspection = {"class": "dew.training.inspection:Inspection",
                  "fields": {"modules": ["*Attention*", "*Block*", "*Mlp*", "*MLP*"], "every": 2,
                             "targets": ["params"]}}
    settings = {"trainer.name": "watched", "trainer.checkpoint_dir": str(runs), "trainer.steps": "2",
                "trainer.log_every": "1", "trainer.eval_every": "None", "trainer.checkpoint_every": "None",
                "trainer.inspection": json.dumps(inspection), "trainer.multi_host": "False",
                "trainer.compilation_cache_dir": "None"}
    assert main(["train", str(record / "run.json"),
                 *(flag for path, value in settings.items() for flag in ("--set", f"{path}={value}"))]) == 0
    logged = scalars(runs / "watched")
    watched = {name.rsplit("/", 1)[1] for name in logged[2] if name.startswith("inspect/activations/")}
    assert {"mean", "std", "abs_max", "non_finite"} <= watched
    assert not any(name.startswith("inspect/activations/") for name in logged[1])


class Headed(nn.Module):
    """A module with a child named `output`, as a DiT's head is."""

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        return nn.Dense(1, name="output")(jnp.tanh(nn.Dense(x.shape[-1], name="hidden")(x)))


def test_a_watched_module_with_a_child_named_output_is_summarized():
    """A watched module whose child is named `output`, as a DiT's head is, is summarized beside it, its
    summary's name no child's (`dew.nn.inspection.SOWN`)."""
    logged = inspected(watched_run(Headed(), Inspection(modules=("Headed",), targets=("params",))),
                       {"x": ROWS})
    assert "inspect/activations/abs_max" in logged


def accumulating(inspection, tracker=None, checkpoints=None, split=1):
    """A trainer of two `split`-microbatch steps an update, over four data shards, whose rows split in two."""
    from dew.training import MeshSpec

    return Trainer(Pulled(), optax.sgd(RATE), key=jax.random.key(0), tracker=tracker, inspection=inspection,
                   accumulation=2, checkpoints=checkpoints, split=split,
                   mesh=MeshSpec().build(jax.devices()[:4]))


def test_a_log_at_a_step_that_commits_nothing_keeps_the_sample_before():
    """With two microbatches an update, a log at a step that commits nothing carries the latest sample
    that committed, not the uncommitted step's zeros."""
    tracker = Kept()
    accumulating(Inspection(paths=("w",), targets=("params",), every=1), tracker).fit(
        Loader(lambda partition: stream(), None, batch=BATCH, records=None), steps=3, log_every=3,
        eval_every=None, checkpoint_every=None)
    assert tracker.rows[3]["inspect/params/l2"] > 0


def test_an_accumulating_trainers_offline_view_steps_its_whole_window():
    """The offline view of a trainer that accumulates steps its whole window and reports the update
    that commits, its parameters' norm nonzero, not an uncommitted step's zeros."""
    logged = inspected(accumulating(Inspection(paths=("w",), targets=("params",))), {"x": ROWS})
    assert logged["inspect/params/l2"] > 0


@pytest.mark.parametrize("split", [1, 2])
def test_an_offline_view_restored_mid_window_steps_to_its_commit(tmp_path, split):
    """The offline view of a run restored with part of its window stepped, each step `split` microbatches
    of it, steps the rest and reports the update that
    commits, not the next window's first, uncommitted step."""
    from dew.checkpoints import Checkpoints

    watched = Inspection(paths=("w",), targets=("params",))
    with Checkpoints(str(tmp_path / "runs")) as checkpoints:
        accumulating(watched, checkpoints=checkpoints, split=split).fit(
            Loader(lambda partition: stream(), None, batch=BATCH, records=None), steps=1, log_every=1,
            eval_every=None, checkpoint_every=1)
    with Checkpoints(str(tmp_path / "runs")) as checkpoints:
        restored = accumulating(watched, checkpoints=checkpoints, split=split)
        assert int(restored.place()[0].microstep) % (2 * split) == split
        assert inspected(restored, {"x": ROWS})["inspect/params/l2"] > 0


TRANSFORMED = ("mean_flow", "ladd", "rcm")
"""Objectives whose trained forward runs inside a transform (a `jax.jvp`, a `lax.cond`), whose watched
forward at its inputs XLA computes once with it; Flow-GRPO's, inside its scan, returns its own
(`test_flow_grpo`)."""


@pytest.mark.parametrize("name", TRANSFORMED)
def test_a_sampled_step_summarizes_its_trained_forward_at_no_forward_more(name):
    """A sampled step of an objective whose trained forward runs inside a transform summarizes that very
    forward, returned through the transform, and computes the model no more than an unsampled step does:
    the sampled loss and gradient run the unsampled's matmuls, their work the same (`step_flops`), the
    summaries elementwise alone (row 2512: one stop-gradient forward more, at its inputs)."""
    from objective_cases import cases

    from dew.nn.inspection import Watched, watching
    from dew.objectives.base import Step
    from dew.telemetry.instrumentation import step_flops

    objective, batch = cases(None)[name]()
    variables = objective.init(jax.random.key(0))
    numeric = {key: value for key, value in batch.items()
               if getattr(value, "dtype", None) is not None and value.dtype.kind in "biuf"}
    held = {key: value for key, value in batch.items() if key not in numeric}
    kinds = ("*Block*", "*Attention*", "*Mlp*", "*MLP*")

    def work(*, watched: bool) -> tuple[float, object]:
        # The step's count is an argument, so a branch on it (rCM's phases) stays a branch.
        def loss(params, numeric, count):
            step = Step(step=count, key=jax.random.key(1), ema=None,
                        watched=Watched(kinds) if watched else None)
            with watching(kinds) if watched else contextlib.nullcontext():
                stats, aux = objective.loss({**variables, "params": params}, {**numeric, **held}, step)
            return objective.reduce_loss(stats)[0], aux.inspected
        stepped = jax.value_and_grad(loss, has_aux=True)
        count = jnp.zeros((), jnp.int32)
        flops = step_flops(jax.jit(stepped).lower(variables["params"], numeric, count).compile())
        assert flops is not None
        return flops.hardware, jax.eval_shape(stepped, variables["params"], numeric, count)[0][1]

    plain, _ = work(watched=False)
    sampled, inspected = work(watched=True)
    assert inspected and sampled == plain
