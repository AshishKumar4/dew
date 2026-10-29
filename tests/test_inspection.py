"""Live inspection (row 2317): what a run watches is summarized inside its step, from the arrays its commit
holds, and logged one log late; a run that watches nothing compiles its step unchanged."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from reference_error import assert_as_exact_as_the_reference
from test_run_options import lm, scalars

from dew.cli.main import main
from dew.data.dataset import Loader
from dew.objectives.base import Aux, Objective
from dew.telemetry.instrumentation import Compiles
from dew.training import Inspection, Trainer, inspected

FEATURES, BATCH, RATE = 4, 8, 0.25
ROWS = np.random.default_rng(0).normal(size=(BATCH, FEATURES)).astype(np.float32)
STATISTICS = ("mean", "rms", "abs_max")


class Pulled(Objective):
    """A weight pulled toward the batch's rows: the mean over them and their coordinates of (w - x)^2,
    whose gradient is 2 (w - the rows' mean) / FEATURES."""

    def fresh_variables(self, key, held):
        return {"params": {"w": jnp.arange(1, FEATURES + 1, dtype=jnp.float32)}}

    def loss(self, variables, batch, step):
        return self.row_mean(jnp.square(variables["params"]["w"] - batch["x"]), batch), Aux({})


class Kept:
    def __init__(self):
        self.rows: dict[int, dict[str, float]] = {}

    def log(self, scalars, step):
        self.rows.setdefault(step, {}).update(scalars)

    def artifact(self, value, step):
        pass

    def close(self):
        pass


def stream():
    while True:
        yield {"x": ROWS}


def trainer(inspection, tracker=None):
    return Trainer(Pulled(), optax.sgd(RATE), key=jax.random.key(0), tracker=tracker, inspection=inspection)


def trained(inspection, steps) -> dict[int, dict[str, float]]:
    tracker = Kept()
    trainer(inspection, tracker).fit(Loader(lambda partition: stream(), None, batch=BATCH, records=None),
                                     steps=steps, log_every=1, eval_every=None, checkpoint_every=None)
    return tracker.rows


def summary(values):
    """`dew.training.inspection`'s statistics of `values`, in their own dtype."""
    return {"mean": values.mean(), "rms": np.sqrt(np.square(values).mean()), "abs_max": np.abs(values).max()}


def test_a_sampled_steps_summaries_are_its_commits_arrays():
    """A step sampled for `w` reports, for the parameters its update applies to, the gradient the
    optimizer is handed, the update and the parameters' change, each statistic as exact as NumPy's in
    float32 against float64's of the same arrays, its L2 norms and its update-to-weight ratio too, and
    no non-finite or zero entries."""
    logged = trained(Inspection(paths=("w",), every=1), 1)[1]

    def arrays(dtype):
        weight, rows = np.arange(1, FEATURES + 1).astype(dtype), ROWS.astype(dtype)
        gradient = 2 * (weight - rows.mean(0)) / FEATURES
        update = -dtype(RATE) * gradient
        return {"params": weight, "grads": gradient, "updates": update, "deltas": (weight + update) - weight}

    single, double = arrays(np.float32), arrays(np.float64)
    names, dew, reference, truth = [], [], [], []
    for target in single:
        for statistic in STATISTICS:
            names.append(f"inspect/{target}/w/{statistic}")
            reference.append(summary(single[target])[statistic])
            truth.append(summary(double[target])[statistic])
        names.append(f"inspect/{target}/l2")
        reference.append(np.linalg.norm(single[target]))
        truth.append(np.linalg.norm(double[target]))
        assert logged[f"inspect/{target}/w/non_finite"] == logged[f"inspect/{target}/w/zero_fraction"] == 0
    names.append("inspect/update_to_weight")
    reference.append(np.linalg.norm(single["updates"]) / np.linalg.norm(single["params"]))
    truth.append(np.linalg.norm(double["updates"]) / np.linalg.norm(double["params"]))
    dew = [logged[name] for name in names]
    assert_as_exact_as_the_reference(np.asarray(dew, np.float32), np.asarray(reference, np.float32),
                                     np.asarray(truth), "inspection summaries")


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
