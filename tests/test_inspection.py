"""Live inspection (row 2317): what a run watches is summarized inside its step, from the arrays its commit
holds, and logged one log late; a run that watches nothing compiles its step unchanged."""

import collections
import contextlib
import itertools
import json

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import test_run_budgets
from objective_cases import cases
from reference_error import assert_computes_the_oracle, assert_fp32_reduction_bound
from run_kinds import KINDS, lm, scalars

from dew.cli.main import main
from dew.data.dataset import Loader
from dew.nn.inspection import HISTOGRAM_EDGES, statistics
from dew.telemetry.compiles import Compiles
from dew.training import Inspection, MeshSpec, Trainer, inspected

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


def test_an_offline_view_of_an_objective_naming_its_input_alone_declares_it_from_its_batch():
    """The offline view of a trainer whose objective names its input field alone (`Supervised(inputs="x")`,
    the documented form) declares that input at the shape its batch gives, as `fit` does from its loader's
    first, and reports the sampled update (row 2719)."""
    from dew.objectives.supervised import Supervised

    objective = Supervised(nn.Dense(2), lambda outputs, batch: jnp.sum(outputs ** 2, axis=-1), inputs="x")
    trainer = Trainer(objective, optax.sgd(RATE), key=jax.random.key(0),
                      inspection=Inspection(paths=("*",), targets=("params",)))
    assert inspected(trainer, {"x": ROWS})["inspect/params/l2"] > 0


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


def test_a_watched_module_path_selects_its_layer_alone():
    """A glob over a module's path (`Block_1`) watches that call alone (row 2324): a sampled step
    summarizes the second block's output as its forward computed it, and no other block's."""
    run = watched_run(Stacked(scanned=False), Inspection(modules=("Block_1",), targets=("params",)))
    logged = inspected(run, {"x": ROWS})
    blocks = sorted({name.rsplit("/", 1)[0] for name in logged if name.startswith("inspect/activations/")})
    assert blocks == ["inspect/activations/Block_1"]
    expected = statistics(jnp.asarray(outputs_of(run, 3)[1]))
    for statistic in ("abs_max", "non_finite", "zero_fraction"):
        assert logged[f"{blocks[0]}/{statistic}"] == float(expected[statistic]), statistic


def test_a_summary_over_no_entries_is_a_zero_one():
    """A summary of no entries (a row set all padding) is a zero tensor's, as a cotangent probe no
    cotangent reached reads (row 3143): mean, deviation, root mean square and non-finite share 0, its zero
    share 1, its histogram all in the lowest decade, none of it NaN."""
    summary = statistics(jnp.ones((3, 4, 5)), histogram=True, where=jnp.zeros((3, 4), bool))
    assert {name: float(value) for name, value in summary.items()} == {
        "mean": 0.0, "std": 0.0, "rms": 0.0, "abs_max": 0.0, "non_finite": 0.0, "zero_fraction": 1.0,
        **{f"histogram_{index}": float(index == 0) for index in range(len(HISTOGRAM_EDGES) + 1)}}


def test_a_summarys_deviation_holds_far_from_zero():
    """A summary's standard deviation of values far from zero, 1e4 apart by 1e-1, is float64's within
    float32's rounding of their sum (row 3143): taken about the values' own mean, not as the gap between
    two squares a float32 cannot tell apart."""
    values = (1e4 + 0.1 * np.random.default_rng(3).normal(size=(64, 32))).astype(np.float32)
    variance = np.var(values.astype(np.float64))
    deviation = float(statistics(jnp.asarray(values))["std"])
    # The variance sums its 2048 entries' squares about their mean, which itself sums them.
    assert_fp32_reduction_bound(deviation ** 2, variance, variance, terms=2 * values.size)


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
    logs a summary of each watched module's output and of the loss's cotangent there at its sampled step,
    its mean, standard deviation and non-finite share, the output's largest magnitude, and none at a step
    it does not sample: every objective's model calls carry the step's cotangent probes (row 2324)."""
    record, runs = tmp_path / "record", tmp_path / "runs"
    (tmp_path / "data").mkdir()
    KINDS[kind](tmp_path / "data", drawn=False).save(str(record))
    inspection = {"class": "dew.training.inspection:Inspection",
                  "fields": {"modules": ["*Attention*", "*Block*", "*Mlp*", "*MLP*"], "every": 2,
                             "targets": ["params"], "cotangents": True}}
    settings = {"trainer.name": "watched", "trainer.checkpoint_dir": str(runs), "trainer.steps": "2",
                "trainer.log_every": "1", "trainer.eval_every": "None", "trainer.checkpoint_every": "None",
                "trainer.inspection": json.dumps(inspection), "trainer.multi_host": "False",
                "trainer.compilation_cache_dir": "None"}
    assert main(["train", str(record / "run.json"),
                 *(flag for path, value in settings.items() for flag in ("--set", f"{path}={value}"))]) == 0
    logged = scalars(runs / "watched")
    watched = {name.rsplit("/", 1)[1] for name in logged[2] if name.startswith("inspect/activations/")}
    assert {"mean", "std", "abs_max", "non_finite"} <= watched
    cotangents = {name.rsplit("/", 1)[1] for name in logged[2] if name.startswith("inspect/cotangents/")}
    assert {"mean", "std", "rms", "non_finite"} <= cotangents
    assert not any(name.startswith("inspect/") for name in logged[1])


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


def accumulating(inspection, tracker=None, checkpoints=None, split=1, accumulation=2):
    """A trainer of `accumulation` `split`-microbatch steps an update, over four data shards, whose rows
    split in two."""
    from dew.training import MeshSpec

    return Trainer(Pulled(), optax.sgd(RATE), key=jax.random.key(0), tracker=tracker, inspection=inspection,
                   accumulation=accumulation, checkpoints=checkpoints, split=split,
                   mesh=MeshSpec().build(jax.devices()[:4]))


def test_a_log_at_a_step_that_commits_nothing_keeps_the_sample_before():
    """With two microbatches an update, a log at a step that commits nothing carries the latest sample
    that committed, not the uncommitted step's zeros."""
    tracker = Kept()
    accumulating(Inspection(paths=("w",), targets=("params",), every=1), tracker).fit(
        Loader(lambda partition: stream(), None, batch=BATCH, records=None), steps=3, log_every=3,
        eval_every=None, checkpoint_every=None)
    assert tracker.rows[3]["inspect/params/l2"] > 0


@pytest.mark.parametrize("every", [2, 3])
def test_an_inspection_samples_every_nth_update_of_an_accumulating_run(every):
    """`every` counts optimizer updates (row 2720): with three steps an update, logged at each update,
    no log carries a sample before the `every`-th update's commit, which carries one, whether or not
    `every` divides the steps an update takes."""
    tracker = Kept()
    accumulating(Inspection(paths=("w",), targets=("params",), every=every), tracker, accumulation=3).fit(
        Loader(lambda partition: stream(), None, batch=BATCH, records=None), steps=3 * every, log_every=3,
        eval_every=None, checkpoint_every=None)
    sampled = [step for step, row in sorted(tracker.rows.items()) if "inspect/params/l2" in row]
    assert sampled == [3 * every]


def test_an_examples_runs_short_last_window_samples_its_commit():
    """A run on the examples clock whose last window closes short samples the update that commits there
    (row 2720): with three steps an update, every second update sampled and five batches, the second
    update commits early, at the fifth step, and its log carries the sample."""
    from dew.data import Memory
    from dew.data.pipeline import Pipeline

    tracker = Kept()
    chain = Pipeline(Memory({"x": np.concatenate([ROWS] * 5)})).read("streamed").batch(BATCH)
    Trainer(Pulled(), optax.sgd(RATE), key=jax.random.key(0), tracker=tracker, accumulation=3,
            inspection=Inspection(paths=("w",), targets=("params",), every=2)).fit(
        chain, examples=5 * BATCH, log_every=1, eval_every=None, checkpoint_every=None)
    sampled = [step for step, row in sorted(tracker.rows.items()) if "inspect/params/l2" in row]
    assert sampled == [5]


def test_a_rejected_step_skips_one_sample_and_the_run_samples_on():
    """A step whose loss is not finite under a dynamic loss scale is rejected and drops its window,
    shifting the windows after it;
    the sampled window it shifted holds no commit, so that sample is skipped, and the next sampled window's
    commit is sampled: with three steps an update and every second window sampled, a rejection at the
    fourth step moves the updates to the seventh and tenth steps, and the tenth's log carries the sample
    (row 2720)."""
    poisoned = ROWS.copy()
    poisoned[0, 0] = np.inf

    def rows(partition):
        for step in itertools.count(1):
            yield {"x": poisoned if step == 4 else ROWS}

    from flax.training.dynamic_scale import DynamicScale

    tracker = Kept()
    Trainer(Pulled(), optax.sgd(RATE), key=jax.random.key(0), tracker=tracker, accumulation=3,
            dynamic_scale=DynamicScale(), inspection=Inspection(paths=("w",), targets=("params",), every=2)
            ).fit(Loader(rows, None, batch=BATCH, records=None), steps=12, log_every=1, eval_every=None,
                  checkpoint_every=None)
    sampled = [step for step, row in sorted(tracker.rows.items()) if "inspect/params/l2" in row]
    assert sampled[0] == 10


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
"""Objectives whose trained forward runs inside a transform (a `jax.jvp`, a `lax.cond`) or beside one, which
returns what the watched modules sow through it; Flow-GRPO's, inside its scan, returns its own
(`test_flow_grpo`)."""


@pytest.mark.parametrize("name", TRANSFORMED)
def test_a_sampled_step_summarizes_its_trained_forward_at_no_forward_more(name):
    """A sampled step of an objective whose trained forward runs inside a transform summarizes that very
    forward, returned through the transform, and traces the model no more than an unsampled step does:
    as an H100 would run it (`generation.assumed`), the sampled loss and gradient hold the unsampled's
    products and kernel launches, so no second forward rides on a compiler merging it, which two
    attention kernels of different forms never are (rows 2512 and 2718)."""
    from reference_error import equations

    from dew.nn.inspection import Watched, watching
    from dew.nn.kernels import generation
    from dew.objectives.base import Step

    objective, batch = cases(None)[name]()
    variables = objective.init(jax.random.key(0))
    numeric = {key: value for key, value in batch.items()
               if getattr(value, "dtype", None) is not None and value.dtype.kind in "biuf"}
    held = {key: value for key, value in batch.items() if key not in numeric}
    kinds = ("*Block*", "*Attention*", "*Mlp*", "*MLP*")

    def work(*, watched: bool) -> tuple[collections.Counter, object]:
        # The step's count is an argument, so a branch on it (rCM's phases) stays a branch.
        def loss(params, numeric, count):
            step = Step(step=count, key=jax.random.key(1), ema=None,
                        watched=Watched(kinds) if watched else None)
            with watching(kinds) if watched else contextlib.nullcontext():
                stats, aux = objective.loss({**variables, "params": params}, {**numeric, **held}, step)
            return objective.reduce_loss(stats)[0], aux.inspected
        stepped = jax.value_and_grad(loss, has_aux=True)
        count = jnp.zeros((), jnp.int32)
        with generation.assumed("sm90"):
            graph, shapes = jax.make_jaxpr(stepped, return_shape=True)(variables["params"], numeric, count)
        names = (equation.primitive.name for equation in equations(graph, platform="cuda"))
        launched = collections.Counter(name for name in names if name in ("dot_general", "pallas_call"))
        return launched, shapes[0][1]

    plain, _ = work(watched=False)
    sampled, inspected = work(watched=True)
    assert inspected and plain["dot_general"] and sampled == plain


def test_a_watched_summary_over_given_positions_is_their_statistics():
    """Watched with the positions a summary reads (`watched_positions`, `[rows, tokens]`), a module's
    output is summarized over those positions alone, every feature of each, as its statistics over them
    are; an output whose leading axes are no `[rows, tokens]` is summarized whole."""
    from dew.nn.inspection import INSPECTED, SOWN, watched_positions, watching

    class Mapped(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(3, name="inner")(x).sum(axis=1)

    values = jnp.asarray(np.random.default_rng(2).normal(size=(4, 5, 2)), jnp.float32)
    positions = jnp.asarray(np.arange(20).reshape(4, 5) % 3 != 0)
    variables = Mapped().init(jax.random.key(0), values)

    def summarized(variables, values, positions):
        with watching(("Dense", "Mapped")), watched_positions(positions):
            output, sown = Mapped().apply(variables, values, mutable=[INSPECTED])
        inner = nn.Dense(3).apply({"params": variables["params"]["inner"]}, values)
        return sown[INSPECTED], inner, statistics(output)

    sown, inner, whole = jax.jit(summarized)(variables, values, positions)
    # The statistics no reduction order changes identify the very entries summarized.
    expected = statistics(jnp.asarray(np.asarray(inner)[np.asarray(positions)]))
    for statistic in ("abs_max", "non_finite", "zero_fraction"):
        assert float(sown["inner"][SOWN][statistic]) == float(expected[statistic]), statistic
    assert jax.tree.map(float, sown[SOWN]) == jax.tree.map(float, whole)


def padding_invariant_summaries(tail, *, packed: bool = False, pipelined: bool = False,
                                warmup: bool = False) -> dict[str, float]:
    """A language model's watched MLP summaries over eight rows, two padded past their text: the pad id
    `tail` fills them, or, `packed`, its coordinates mark them unreal (no pad id) and `tail` is what
    they hold; `pipelined`, over two stages of two microbatches; `warmup`, an MLA decoder's indexer
    warm-up, whose loss is the indexer's alone."""
    from model_support import TINY_DECODER

    from dew.batch import coordinates
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import IndexerTraining, LMObjective
    from dew.registry import models
    from dew.training import MeshSpec

    model = (CausalTransformer(vocab_size=64, emb_features=32, num_layers=2, num_heads=2, head_dim=16,
                               mlp_features=64, max_seq_len=16, mixer={"class": "mla", "fields": {
                                   "q_lora_rank": 8, "kv_lora_rank": 8, "qk_nope_head_dim": 8,
                                   "qk_rope_head_dim": 8, "v_head_dim": 8, "index_topk": None,
                                   "index_n_heads": 2, "index_head_dim": 16}}) if warmup
             else models.build("causal_transformer", vocab_size=64, **{**TINY_DECODER, "num_layers": 4},
                               num_kv_heads=2))
    indexer = {"indexer": IndexerTraining("warmup")} if warmup else {}
    staged = ({"microbatches": 2, "mesh": MeshSpec(fsdp=2, stage=2).build(jax.devices()[:4])} if pipelined
              else {})
    run = Trainer(LMObjective(model, 16, pad_id=None if packed else tail, **indexer), optax.sgd(RATE),
                  key=jax.random.key(0), inspection=Inspection(modules=("*MLP",), targets=("params",)),
                  **staged)
    tokens = np.random.default_rng(1).integers(1, 63, size=(8, 17)).astype(np.int32)
    real = np.ones(tokens.shape, bool)
    real[1, 10:], real[6, 3:] = False, False
    tokens[~real] = tail
    batch = {"text": tokens, **({"text_coordinates": coordinates(valid=real, xp=np)} if packed else {})}
    return {name: value for name, value in inspected(run, batch).items() if "activations" in name}


@pytest.mark.parametrize("layout", ["padded", "packed", "indexer warm-up"])
def test_a_language_models_watched_layers_read_none_of_its_padding(layout):
    """A language model's watched layers summarize their outputs at its real tokens alone (row 2324): two
    rows padded past their text, once with one id and once with another, whose causal forward is the
    same at every real token and differs at the padding, give the same summary of each MLP to the bit,
    the padding marked by the pad id or by the batch's coordinates."""
    first, second = (padding_invariant_summaries(pad, packed=layout == "packed",
                                                 warmup=layout == "indexer warm-up") for pad in (0, 63))
    assert len(first) >= 12 and first == second


@pytest.mark.mesh(devices=4)
def test_a_pipelined_language_models_watched_layers_read_none_of_its_padding():
    """Pipelined over two stages of two microbatches, each microbatch's summary of a watched layer reads
    that microbatch's real tokens alone: the two pad ids give the same summaries to the bit."""
    first, second = (padding_invariant_summaries(pad, pipelined=True) for pad in (0, 63))
    assert len(first) >= 48 and first == second


def test_a_preference_runs_watched_layers_read_each_halfs_real_tokens_alone():
    """A preference pair objective's watched layers are summarized over each half's real tokens alone
    (row 2324), as its pairs' layout marks them, whatever their completion marks: a pair whose chosen half
    ends in an unmarked token, its padding filled with another id, gives the same summaries to the bit,
    and that last real token changed gives others."""
    from objective_cases import decoder

    from dew.data import Memory, presets
    from dew.objectives.rl import DPOObjective
    from dew.training import MeshSpec

    pair = {"chosen": [1, 2, 3, 4], "rejected": [1, 2, 5], "chosen_mask": [0, 0, 1, 0],
            "rejected_mask": [0, 0, 1]}
    (batch,) = list(Memory([pair] * 2).pipe(presets.preferences, seq_len=6, batch=2))
    run = Trainer(DPOObjective(decoder(), 5), optax.sgd(RATE), key=jax.random.key(0),
                  mesh=MeshSpec().build(jax.devices()[:1]),
                  inspection=Inspection(modules=("*MLP",), targets=("params",)))
    ids = np.asarray(batch["input_ids"])
    padding = np.arange(ids.shape[-1]) >= np.array([4, 3])[None, :, None]
    last = np.zeros(ids.shape, bool)
    last[:, 0, 3] = True

    def summaries(rows):
        return {name: value for name, value in inspected(run, {**batch, "input_ids": rows}).items()
                if "activations" in name}

    first = summaries(ids)
    assert first and first == summaries(np.where(padding, 7, ids).astype(ids.dtype))
    assert first != summaries(np.where(last, 9, ids).astype(ids.dtype))


@pytest.mark.mesh(devices=4)
def test_a_pipelined_stacks_watched_layers_each_report_every_microbatch():
    """A decoder pipelined over two stages of two microbatches each reports every layer it watches (row
    2324): each of its four MLPs' summary, one a microbatch the pipeline runs, finite, from the sampled
    step's own forward, each named by its layer's place in the decoder, not in its stage."""
    from model_support import TINY_DECODER

    from dew.objectives.lm import LMObjective
    from dew.registry import models
    from dew.training import MeshSpec

    model = models.build("causal_transformer", vocab_size=64, **{**TINY_DECODER, "num_layers": 4},
                         num_kv_heads=2)
    run = Trainer(LMObjective(model, 16), optax.sgd(RATE), key=jax.random.key(0), microbatches=2,
                  mesh=MeshSpec(fsdp=2, stage=2).build(jax.devices()[:4]),
                  inspection=Inspection(modules=("*MLP",), targets=("params",)))
    tokens = np.random.default_rng(1).integers(1, 64, size=(8, 17)).astype(np.int32)
    logged = inspected(run, {"text": tokens})
    summaries = {name.rsplit("/", 1)[0]: value for name, value in logged.items()
                 if name.endswith("/rms") and "activations" in name}
    assert set(summaries) == {f"inspect/activations/layers_{layer}/mlp/{microbatch}"
                              for layer in range(4) for microbatch in range(2)}
    assert all(np.isfinite(value) and value > 0 for value in summaries.values())


def test_a_watched_cotangent_is_the_updates_loss_whatever_its_split():
    """A window split into microbatches summarizes the cotangent of the update's pooled loss at its
    committing microbatch, not the microbatch's own loss (row 3144): over a batch of one row repeated, the
    split of two logs the cotangent the whole batch's step logs, and a loss stated as a scalar is
    summarized too."""
    def logged(split: int, *, scalar: bool = False) -> dict[str, float]:
        objective = watched_run(Stacked(scanned=False), None).objective
        if scalar:
            # A loss its objective states as one scalar, pooled by the microbatches' mean.
            objective.loss = lambda variables, batch, step, loss=objective.loss: (
                lambda stats, aux: (stats.mean()[0], aux))(*loss(variables, batch, step))
        run = Trainer(objective, optax.sgd(RATE), key=jax.random.key(0), split=split,
                      mesh=MeshSpec().build(jax.devices()[:1]),
                      inspection=Inspection(modules=("Block_1",), targets=("params",), cotangents=True))
        # One row throughout, so any microbatch's rows are the batch's own, however the split deals them.
        rows = np.repeat(ROWS[:1], BATCH, axis=0)
        return {name: value for name, value in inspected(run, {"x": rows}).items() if "/cotangents/" in name}

    whole, split = logged(1), logged(2)
    assert set(whole) == set(split) and logged(2, scalar=True)
    # The mean square sums the committing microbatch's entries, half the batch's, each the same row's.
    squares = np.square(whole["inspect/cotangents/Block_1/rms"])
    assert_fp32_reduction_bound(np.square(split["inspect/cotangents/Block_1/rms"]), squares, squares,
                                terms=ROWS.size)


def test_a_watched_cotangent_is_summarized_unscaled_under_a_dynamic_loss_scale():
    """Under a dynamic loss scale, whose backward is seeded with the scale, a watched block's cotangent is
    summarized as the unscaled loss's (row 2324 review): its every statistic, the histogram's decades
    included, the unscaled run's to the bit, a power of two scaling and unscaling each entry exactly."""
    def logged(**scaled) -> dict[str, float]:
        objective = watched_run(Stacked(scanned=False), None).objective
        run = Trainer(objective, optax.sgd(RATE), key=jax.random.key(0), **scaled,
                      inspection=Inspection(modules=("Block_1",), targets=("params",), cotangents=True,
                                            histogram=True))
        return {name: value for name, value in inspected(run, {"x": ROWS}).items() if "/cotangents/" in name}

    plain, scaled = logged(), logged(dynamic_scale=True)
    assert any("histogram" in name for name in plain) and scaled == plain


def test_a_watched_layers_cotangent_is_summarized_as_the_losses_cotangent_there():
    """With cotangents watched, a sampled step summarizes the loss's cotangent at each watched block's
    output (row 2324): the root mean square, mean and shares it logs are those of the cotangent Linen's
    own perturbation of the block's output receives under the same loss, from a probe's few numbers
    rather than a tensor of the output's size; its gradients and activations are the unwatched step's to
    the bit."""

    class Perturbed(nn.Module):
        """`Stacked(scanned=False)`'s layers under its names, the second block's output perturbed."""

        @nn.compact
        def __call__(self, x):
            x = x.astype(jnp.float32)
            for index in range(3):
                x, _ = Block()(x)
                if index == 1:
                    x = self.perturb("block_1", x)
            return nn.Dense(1, name="head")(x)

    def logged(*, cotangents: bool) -> dict[str, float]:
        run = watched_run(Stacked(scanned=False), Inspection(modules=("Block_1",), targets=("grads",),
                                                             cotangents=cotangents))
        return inspected(run, {"x": ROWS})

    watched, unwatched = logged(cotangents=True), logged(cotangents=False)
    state, _, _ = watched_run(Stacked(scanned=False), None).place()
    params = jax.device_get(state.variables)["params"]
    perturbations = Perturbed().init(jax.random.key(0), jnp.asarray(ROWS))["perturbations"]

    def loss(perturbations):
        outputs = Perturbed().apply({"params": params, "perturbations": perturbations}, jnp.asarray(ROWS))
        return jnp.mean(jnp.square(outputs[:, 0]))

    expected = statistics(jax.grad(loss)(perturbations)["block_1"])
    for statistic in ("rms", "mean", "non_finite", "zero_fraction"):
        # Each statistic sums the cotangent's entries, each entry its rows' squared outputs' derivatives.
        assert_computes_the_oracle(watched[f"inspect/cotangents/Block_1/{statistic}"], expected[statistic],
                                   f"the watched block's cotangent {statistic}", roundings=ROWS.size)
    assert {name: value for name, value in watched.items() if "/cotangents/" not in name} == unwatched


@pytest.fixture(scope="module")
def windows(tmp_path_factory):
    from objective_data import corpus_windows

    return corpus_windows(tmp_path_factory.mktemp("corpus"))


@pytest.mark.parametrize("name", sorted(set(cases(None)) - {"flow_grpo"}))
def test_every_objectives_watched_cotangents_reach_its_model_calls(name, windows, tmp_path):
    """Every objective's step that summarizes its watched outputs' cotangents (row 2324) carries its
    cotangent probes to each model call it watches, which no step refuses, and the loss's cotangent
    reaches each probe: a count of entries and a finite sum of their squares, the parameters' gradient
    the unprobed step's to the bit. Flow-GRPO's, trained on rollouts, is `test_flow_grpo`'s."""
    from test_whole_validation import runnable

    from dew.nn.inspection import Watched
    from dew.objectives.base import Step
    from dew.training.transaction import probed_loss

    objective, batch = runnable(name, windows, tmp_path)
    if name == "ppo":
        mask = np.asarray(batch["response_mask"], np.float32)
        batch = {**batch, "old_values": np.zeros_like(mask), "returns": mask}
    variables = objective.init(jax.random.key(0))
    numeric = {key: value for key, value in batch.items()
               if getattr(value, "dtype", None) is not None and value.dtype.kind in "biuf"}
    held = {key: value for key, value in batch.items() if key not in numeric}
    watched = Watched(("*Block*", "*Attention*", "*Mlp*", "*MLP*"), cotangents=True)

    def step(watched):
        return Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(1), ema=variables, watched=watched)

    def reduced(stats):
        return objective.reduce_loss(stats)[0]

    def gradients(numeric):
        stats, back, _ = probed_loss(objective, variables, {**numeric, **held}, step(watched))
        _, seed = jax.vjp(reduced, stats)
        return back(seed(jnp.ones(()))[0])

    def plain(numeric):
        def loss(params):
            return reduced(objective._loss({**variables, "params": params}, {**numeric, **held},
                                           step(Watched(watched.modules)))[0])
        return jax.grad(loss)(variables["params"])

    (params, probes), expected = jax.jit(gradients)(numeric), jax.jit(plain)(numeric)
    # rCM's student phase runs its fake score under no gradient (`test_an_rcm_critic_updates_...`).
    trained = {key: tree for key, tree in probes.items() if not (name == "rcm" and key == "fake_score")}
    assert trained and all(leaf[..., 0].min() > 0 and np.isfinite(leaf[..., 2]).all()
                           for leaf in jax.tree.leaves(trained))
    assert all((left == right).all()
               for left, right in zip(jax.tree.leaves(params), jax.tree.leaves(expected), strict=True))


@pytest.mark.mesh(devices=4)
def test_a_pipelined_stacks_watched_cotangents_are_each_layers_over_its_microbatches():
    """A decoder pipelined over two stages of two microbatches summarizes the loss's cotangent at each
    watched layer over every microbatch it ran (row 2324), one summary a layer: its entries those of the
    unpipelined step's, none of a bubble iteration's discarded zeros, so its zero and non-finite shares
    are the unpipelined step's to the bit."""
    from model_support import TINY_DECODER

    from dew.objectives.lm import LMObjective
    from dew.registry import models
    from dew.training import MeshSpec

    def cotangents(**placed) -> dict[str, float]:
        model = models.build("causal_transformer", vocab_size=64, **{**TINY_DECODER, "num_layers": 4},
                             num_kv_heads=2)
        run = Trainer(LMObjective(model, 16), optax.sgd(RATE), key=jax.random.key(0), **placed,
                      inspection=Inspection(modules=("*MLP",), targets=("params",), cotangents=True))
        tokens = np.random.default_rng(1).integers(1, 64, size=(8, 17)).astype(np.int32)
        return {name: value for name, value in inspected(run, {"text": tokens}).items()
                if name.startswith("inspect/cotangents/")}

    piped = cotangents(microbatches=2, mesh=MeshSpec(fsdp=2, stage=2).build(jax.devices()[:4]))
    whole = cotangents(mesh=MeshSpec(fsdp=2).build(jax.devices()[:2]))
    assert set(piped) == set(whole)
    assert {f"inspect/cotangents/layers_{layer}/mlp/rms" for layer in range(4)} <= set(piped)
    for name, value in whole.items():
        if name.endswith(("/zero_fraction", "/non_finite")):
            assert piped[name] == value, name
        assert np.isfinite(piped[name])


def rcm_probes(windows, tmp_path, *, student: bool, consistency_weight: float | None = None) -> dict:
    """The gradient of an rCM step's watched blocks' cotangent probes (`probed_loss`), at the first update
    past its warm-up of the phase `student` names, its consistency weight `consistency_weight` if given."""
    from test_whole_validation import runnable

    from dew.nn.inspection import Watched
    from dew.objectives.base import Step
    from dew.training.transaction import probed_loss

    objective, batch = runnable("rcm", windows, tmp_path)
    if consistency_weight is not None:
        objective.consistency_weight = consistency_weight
    variables = objective.init(jax.random.key(0))
    numeric = {key: value for key, value in batch.items()
               if getattr(value, "dtype", None) is not None and value.dtype.kind in "biuf"}
    held = {key: value for key, value in batch.items() if key not in numeric}
    update = next(update for update in range(objective.tangent_warmup, objective.tangent_warmup + 64)
                  if bool(objective._student(update)) == student)
    step = Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(1), ema=variables,
                committed=jnp.asarray(update), watched=Watched(("*Block*",), cotangents=True))

    def probed(numeric):
        stats, back, _ = probed_loss(objective, variables, {**numeric, **held}, step)
        _, seed = jax.vjp(lambda stats: objective.reduce_loss(stats)[0], stats)
        return back(seed(jnp.ones(()))[0])[1]

    return jax.jit(probed)(numeric)


def test_an_rcm_critic_updates_watched_cotangents_are_its_fake_scores(windows, tmp_path):
    """An rCM critic update past its warm-up, which trains the fake score and runs the student under no
    gradient, summarizes the loss's cotangent at the fake score's watched modules, each reached by it, and
    the student's as the zero it is (row 2324 review): no statistic NaN, a probe no cotangent reached
    counting none."""
    from dew.nn.inspection import summarized
    from dew.objectives.diffusion.consistency import FAKE_SCORE

    probes = rcm_probes(windows, tmp_path, student=False)
    fake = jax.tree.leaves(probes[FAKE_SCORE])
    student = jax.tree.leaves({name: tree for name, tree in probes.items() if name != FAKE_SCORE})
    assert fake and student
    assert all(leaf[..., 0].min() > 0 for leaf in fake) and all((leaf[..., 0] == 0).all() for leaf in student)
    for leaf in student:
        summary = summarized(leaf.reshape(-1, leaf.shape[-1])[0], histogram=False)
        assert all(np.isfinite(float(value)) for value in summary.values())
        assert float(summary["rms"]) == 0 and float(summary["zero_fraction"]) == 1


def test_a_distribution_matching_student_updates_cotangent_reaches_its_trained_call(windows, tmp_path):
    """An rCM student update by distribution matching alone, whose simulation's every call but the last
    runs under no gradient, summarizes the loss's cotangent at the student's watched modules from the call
    that trains it, the simulation's last (row 2324 review): each student probe counts that call's
    entries."""
    from dew.objectives.diffusion.consistency import FAKE_SCORE

    probes = rcm_probes(windows, tmp_path, student=True, consistency_weight=0.0)
    trained = jax.tree.leaves({name: tree for name, tree in probes.items() if name != FAKE_SCORE})
    assert trained and all(leaf[..., 0].min() > 0 and np.isfinite(leaf[..., 2]).all() for leaf in trained)


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("stacked", ["scanned", "pipelined"])
def test_a_path_glob_selects_the_layer_its_log_names(stacked):
    """A glob over a module's path names it as the log does, by its layer's place in the decoder (row
    3126): `layers_2/*` logs layer 2's modules alone, every entry and value a run watching every module
    logs for them, their outputs and their cotangents, whether the decoder scans its layers as one run
    (`layers_0_3`) or pipelines them over two stages (`stages/layers_j`); a glob naming no layer of the
    decoder is refused."""
    from model_support import TINY_DECODER

    from dew.objectives.lm import LMObjective
    from dew.registry import models
    from dew.training import MeshSpec

    def logged(*modules: str) -> dict[str, float]:
        fields = {**TINY_DECODER, "num_layers": 4, **({"scan_layers": True} if stacked == "scanned" else {})}
        model = models.build("causal_transformer", vocab_size=64, **fields, num_kv_heads=2)
        placed = ({"microbatches": 2, "mesh": MeshSpec(fsdp=2, stage=2).build(jax.devices()[:4])}
                  if stacked == "pipelined" else {})
        run = Trainer(LMObjective(model, 16), optax.sgd(RATE), key=jax.random.key(0), **placed,
                      inspection=Inspection(modules=modules, targets=("params",), cotangents=True))
        tokens = np.random.default_rng(1).integers(1, 64, size=(8, 17)).astype(np.int32)
        return {name: value for name, value in inspected(run, {"text": tokens}).items()
                if name.startswith(("inspect/activations/", "inspect/cotangents/"))}

    selected, every = logged("layers_2/*"), logged("layers_*/*")
    reference = {name: value for name, value in every.items() if name.split("/")[2] == "layers_2"}
    assert set(selected) == set(reference) and any("/cotangents/" in name for name in selected)
    # The statistics no reduction order changes are the very entries' to the bit; a cotangent's root mean
    # square, which sums every entry of the layer's output (8 rows of 16 tokens by its width), the reference's
    # as one order of that sum is another's.
    for name in selected:
        if name.endswith(("/abs_max", "/non_finite", "/zero_fraction")):
            assert selected[name] == reference[name], name
        elif "/cotangents/" in name and name.endswith("/rms"):
            assert_computes_the_oracle(selected[name], reference[name], name, roundings=8 * 16 * 64)
    with pytest.raises(ValueError):
        logged("layers_9/*")


@pytest.mark.mesh(devices=4)
def test_a_pipeline_stage_modules_cotangent_is_refused():
    """A pipeline's stage module (`stages`), which runs a stage's layers and which the log names by no
    layer, has its watched cotangent refused as the step traces, naming the layers to watch instead
    (row 3126 review)."""
    from model_support import TINY_DECODER

    from dew.objectives.lm import LMObjective
    from dew.registry import models
    from dew.training import MeshSpec

    model = models.build("causal_transformer", vocab_size=64, **{**TINY_DECODER, "num_layers": 4},
                         num_kv_heads=2)
    run = Trainer(LMObjective(model, 16), optax.sgd(RATE), key=jax.random.key(0), microbatches=2,
                  mesh=MeshSpec(fsdp=2, stage=2).build(jax.devices()[:4]),
                  inspection=Inspection(modules=("stages",), targets=("params",), cotangents=True))
    with pytest.raises(ValueError):
        inspected(run, {"text": np.random.default_rng(1).integers(1, 64, size=(8, 17)).astype(np.int32)})
