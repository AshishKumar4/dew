"""Optimizers that hold an update back on some calls (`dew.held_updates`): the run's clocks count the
updates they apply, a resume inside a held window lands where the whole run does, and one whose state
does not say which call applied an update is refused (row 1870)."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import FEATURES, Affine, Counting, dataset
from recording import RecordingTracker
from test_trainer import TokenRegression, make_trainer

from dew.objectives.base import Aux, Objective


def held_optimizer(held: str, rate: float = 0.1):
    """An optimizer that applies an update every second call, holding the first back: optax's
    `MultiSteps` (alone, or after a stateless transform in a chain) or `apply_every`, which sums the two
    calls' updates where `MultiSteps` averages them; or `apply_if_finite`, which holds a non-finite
    update back and applies every finite one."""
    return {"multi_steps": optax.MultiSteps(optax.sgd(rate), every_k_schedule=2).gradient_transformation(),
            "chained": optax.chain(optax.clip_by_global_norm(1e6),
                                   optax.MultiSteps(optax.sgd(rate), 2).gradient_transformation()),
            "apply_every": optax.chain(optax.sgd(rate / 2), optax.apply_every(2)),
            "finite": optax.apply_if_finite(optax.sgd(rate), max_consecutive_errors=4)}[held]


class Effected(Objective[jax.Array, jax.Array]):
    """`Regression`'s squared error, counting in its own collection each update it commits, as a
    post-step effect (`Aux.effects`)."""

    def __init__(self):
        self.model = Affine()

    def fresh_variables(self, key, held):
        return {**self.model.init(key, jnp.zeros((1, FEATURES))), "stats": {"seen": jnp.zeros(())}}

    def loss(self, variables, batch, step):
        prediction = self.model.apply({"params": variables["params"]}, batch["x"])
        return jnp.mean((prediction - batch["y"]) ** 2), Aux({}, effects=jnp.asarray(1.0))

    def apply_effects(self, variables, effects):
        return {"stats": {"seen": variables["stats"]["seen"] + effects}}


def every_other(step: jax.typing.ArrayLike, **_) -> jax.Array:
    """Whether `optax.conditionally_mask` applies the update of call `step`: every second one."""
    return jnp.asarray(step) % 2 == 1


class Drawing(TokenRegression):
    """`TokenRegression`, reporting a uniform draw of the step's examples' key (`Step.example_key`),
    which a trace of its shapes alone holds none of."""

    def loss(self, variables, batch, step):
        loss, aux = super().loss(variables, batch, step)
        drawn = jnp.zeros(()) if step.example_key is None else jax.random.uniform(step.example_key)
        return loss, Aux({**aux.metrics, "draw": drawn})


@pytest.mark.parametrize("held", ["multi_steps", "chained", "apply_every"])
def test_an_optimizer_that_holds_updates_back_counts_its_clocks_by_the_updates_it_applies(held):
    """An optimizer that holds every other update back (row 1870) moves the run's clocks as the trainer's
    own window of two does: a held call moves no weight, no average and no update count, and the tokens
    it trains on wait for the update that applies them, so a budget of 30 tokens ends at the fourth step,
    the second update, which commits 24 and then 56; and each call draws its examples afresh."""
    tracker, window = RecordingTracker(), RecordingTracker()
    state = make_trainer(objective=Drawing(), optimizer=held_optimizer(held), tracker=tracker).fit(
        dataset(), tokens=30, log_every=1)
    pooled = make_trainer(objective=Drawing(), tracker=window, accumulation=2).fit(
        dataset(), tokens=30, log_every=1)
    assert int(state.step) == int(pooled.step) == 4 and int(state.updates) == int(pooled.updates) == 2

    def logged(recorded, name):
        return [scalars[name] for _, scalars in recorded.scalars if name in scalars]

    assert logged(tracker, "train/tokens") == logged(window, "train/tokens") == [0, 24, 0, 56]
    assert len(set(logged(tracker, "train/draw"))) == 4

    trainer = make_trainer(objective=Drawing(), optimizer=held_optimizer(held))
    batch = next(iter(Counting()))
    stepped = trainer.initial_state()
    step = trainer.compile(stepped, batch)
    for call in range(4):
        before = jax.tree.map(np.asarray, (stepped.variables, stepped.ema))
        stepped, *_ = step(stepped, batch)
        after = jax.tree.map(np.asarray, (stepped.variables, stepped.ema))
        moved = [not np.array_equal(old, new) for old, new in zip(jax.tree.leaves(before),
                                                                     jax.tree.leaves(after), strict=True)]
        applied = call % 2 == 1
        assert (any(moved), int(stepped.updates)) == (applied, (call + 1) // 2)


@pytest.mark.parametrize("held", ["multi_steps", "finite"])
def test_a_run_resumed_inside_a_held_window_lands_where_the_whole_run_does(tmp_path, held):
    """A run whose optimizer holds updates back, stopped after a held call and resumed from its
    checkpoint, keeps the tokens that call trained on waiting (`TrainState.deferred`) and lands where the
    whole run does: its weights, average, update count and committed tokens."""
    def run(directory, **bound):
        tracker = RecordingTracker()
        state = make_trainer(directory, objective=TokenRegression(), optimizer=held_optimizer(held),
                             tracker=tracker).fit(dataset(), checkpoint_every=1, log_every=1, **bound)
        return state, [scalars["train/tokens"] for _, scalars in tracker.scalars if "train/tokens" in scalars]

    whole, committed = run(tmp_path / "whole", steps=4)
    run(tmp_path / "split", steps=3)
    resumed, rest = run(tmp_path / "split", steps=4)
    assert int(resumed.updates) == int(whole.updates)
    assert rest == committed[3:]
    for mine, theirs in zip(jax.tree.leaves((resumed.variables, resumed.ema)),
                            jax.tree.leaves((whole.variables, whole.ema)), strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


@pytest.mark.parametrize("held", ["conditional", "masked", "partitioned", "followed", "effects"])
def test_an_optimizer_whose_held_updates_no_state_tells_is_refused_before_any_step(held):
    """An optimizer that holds updates back where its state does not say which call applied one is
    refused before any step: `optax.conditionally_mask`, whose condition its state does not hold; a held
    update under a mask or a partition, which moves some parameters every call; a held update followed in
    a chain by a transform that keeps state, which moves the weights on a held call. So is an objective
    with post-step effects under a held optimizer, whose held calls' effects no update applies. An
    optimizer that applies every update trains."""
    multi = optax.MultiSteps(optax.sgd(0.1), 2).gradient_transformation()
    optimizer = {"conditional": optax.conditionally_mask(optax.sgd(0.1), every_other),
                 "masked": optax.masked(multi, lambda params: jax.tree.map(lambda _: True, params)),
                 "partitioned": optax.partition({"held": multi, "every": optax.sgd(0.1)},
                                                lambda params: jax.tree.map(lambda _: "held", params)),
                 "followed": optax.chain(multi, optax.trace(0.9)),
                 "effects": multi}[held]
    with pytest.raises(ValueError):
        make_trainer(objective=Effected() if held == "effects" else None, optimizer=optimizer).fit(
            dataset(), steps=2, log_every=1)
    assert int(make_trainer(objective=Effected()).fit(dataset(), steps=2, log_every=1).updates) == 2
