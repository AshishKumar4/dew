"""A run's surface (API03): its everyday settings named plainly and once, in Python and on its command
line, each choice a subcommand of plain flags."""

import dataclasses
import itertools

import jax
import numpy as np
import optax
import pytest
from affine_run import Regression, dataset
from recording import RecordingTracker

from dew.config import RunConfig
from dew.optim import AdamW, Cosine, OptimConfig
from dew.training import Trainer


@pytest.mark.parametrize(("flags", "peak"), [
    (["--optim.lr", "3e-4"], None),
    (["--optim.lr", "1e-2", "optim.schedule:cosine", "--optim.schedule.warmup-steps", "2"], 1e-2)],
    ids=["constant", "cosine"])
def test_a_command_lines_rate_is_the_rate_each_update_takes(flags, peak):
    """A run's rate on the command line is what its updates step at: a constant (`--optim.lr`) each one's,
    and a schedule (`optim.schedule:cosine`), its peak the rate the line names (`OptimConfig.lr`), rising
    from zero over its warmup to that peak and falling after it, as the run's optimizer reports each
    update's rate."""
    tracker = RecordingTracker()
    Trainer(Regression(), RunConfig.cli(flags).optim, key=0, tracker=tracker).fit(dataset(), steps=4,
                                                                                 log_every=1)
    taken = [np.float32(scalars["train/learning_rate"]) for _, scalars in tracker.scalars
             if "train/learning_rate" in scalars]
    assert len(taken) == 4, taken
    if peak is None:
        assert set(taken) == {np.float32(3e-4)}, taken
        return
    top = taken.index(max(taken))
    assert max(taken) == np.float32(peak) and 0 < top < len(taken) - 1, taken
    assert all(earlier < later for earlier, later in itertools.pairwise(taken[:top + 1])), taken
    assert all(later < earlier for earlier, later in itertools.pairwise(taken[top:])), taken


def test_each_choice_of_a_run_is_a_subcommand_of_plain_flags():
    """Every choice a run makes on its command line is a subcommand whose settings are plain flags, no
    record written as JSON (API03): its optimizer's (`optim.optimizer:muon`), a flag of the default's own
    settings choosing the default (`--optim.optimizer.b2` an AdamW's), its schedule's, chosen by a flag
    of its settings alone, its numerics' preset (`trainer.precision:bf16-true`), its checkpoints kept and
    ranked, and each op's kernel; and `--help` renders every one."""
    from dew.config.lm import LMRunConfig
    from dew.kernels import Kernels
    from dew.nn.precision import numerics_of
    from dew.optim import AdamW, Cosine, Muon
    from dew.training import Best, Keep

    chosen = LMRunConfig.cli(["optim.optimizer:muon", "--optim.optimizer.consistent-rms", "0.2",
                              "--optim.schedule.warmup-steps", "10", "trainer.precision:bf16-true",
                              "--trainer.keep.latest", "3", "--trainer.best.metric", "val/accuracy",
                              "--trainer.best.mode", "max", "--diagnostics.kernels.attention", "xla",
                              "--trainer.steps", "100", "--trainer.accumulation", "2"])
    assert chosen.optim.optimizer == Muon(consistent_rms=0.2), chosen.optim
    assert isinstance(chosen.optim.schedule, Cosine) and chosen.optim.schedule.warmup_steps == 10
    assert chosen.optim.schedule.peak_value == chosen.optim.lr, chosen.optim
    assert chosen.trainer.precision == numerics_of("bf16-true"), chosen.trainer.precision
    assert (chosen.trainer.keep, chosen.trainer.best) == (Keep(latest=3), Best("val/accuracy", mode="max"))
    assert chosen.diagnostics.kernels == Kernels(attention="xla")
    assert (chosen.trainer.steps, chosen.trainer.accumulation) == (100, 2)
    default = LMRunConfig.cli([]).optim.optimizer
    assert isinstance(default, AdamW), default
    assert LMRunConfig.cli(["--optim.optimizer.b2", "0.95"]).optim.optimizer == dataclasses.replace(default,
                                                                                                    b2=0.95)
    with pytest.raises(SystemExit) as helped:
        LMRunConfig.cli(["--help"])
    assert helped.value.code == 0


@pytest.mark.parametrize("optimizer", [optax.sgd(0.1), OptimConfig(lr=0.1)], ids=["optax", "config"])
@pytest.mark.parametrize("stated", [{"lr": 3e-4}, {"schedule": Cosine(warmup_steps=1)}, {"clip": 1.0}],
                         ids=["lr", "schedule", "clip"])
def test_a_rate_beside_an_optimizer_that_states_its_own_is_refused(optimizer, stated):
    """A trainer's `lr`, `schedule` and `clip` set the run of an optimizer's spec (`Trainer(objective,
    Muon(), lr=3e-4)`); beside an `OptimConfig` or an Optax transformation, each of which states its own, a
    rate is named twice and the trainer is refused as it is made, never one silently written over the
    other."""
    with pytest.raises(ValueError):
        Trainer(Regression(), optimizer, **stated)


def test_a_spec_and_its_rate_train_as_the_config_holding_them():
    """An optimizer's spec given its rate, schedule and clip beside it (`Trainer(objective, AdamW(),
    lr=...)`) trains as the `OptimConfig` that holds them, to the bit."""
    stated = {"lr": 1e-2, "schedule": Cosine(warmup_steps=1), "clip": 0.5}
    beside = Trainer(Regression(), AdamW(), key=0, **stated).fit(dataset(), steps=3, log_every=100)
    held = Trainer(Regression(), OptimConfig(optimizer=AdamW(), **stated), key=0).fit(dataset(), steps=3,
                                                                                      log_every=100)
    jax.tree.map(np.testing.assert_array_equal, *jax.device_get((beside.variables, held.variables)))


def test_a_flag_of_a_recipes_optimizer_moves_that_optimizer():
    """A flag of an optimizer's settings given alone moves the optimizer of the run the command line parses
    over (`cli`'s `default`, a recipe's), never the class's default one: a recipe's Adam takes
    `--optim.optimizer.b2 0.95` as its own b2 and stays Adam (review of API03: it became an AdamW)."""
    from dew.optim import Adam

    recipe = RunConfig(optim=OptimConfig(optimizer=Adam()))
    parsed = RunConfig.cli(["--optim.optimizer.b2", "0.95"], default=recipe).optim.optimizer
    assert type(parsed) is Adam and parsed == Adam(b2=0.95), parsed
