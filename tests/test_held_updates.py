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

from dew.held_updates import phased
from dew.objectives.base import Aux, Objective
from dew.optim import power_profiles


def held_optimizer(held: str, rate: float = 0.1):
    """An optimizer that applies an update every second call, holding the first back: optax's
    `MultiSteps`, alone, after a stateless transform in a chain or a named chain, or around an optimizer
    whose hyperparameters it injects; or `apply_every` last in a chain, which sums the two calls'
    updates where `MultiSteps` averages them."""
    multi = optax.MultiSteps(optax.sgd(rate), every_k_schedule=2).gradient_transformation()
    return {"multi_steps": multi,
            "chained": optax.chain(optax.clip_by_global_norm(1e6), multi),
            "named": optax.named_chain(("clip", optax.clip_by_global_norm(1e6)), ("held", multi)),
            "injected": optax.MultiSteps(optax.inject_hyperparams(optax.sgd)(learning_rate=rate),
                                         2).gradient_transformation(),
            "apply_every": optax.chain(optax.sgd(rate / 2), optax.apply_every(2))}[held]


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


@pytest.mark.parametrize("held", ["multi_steps", "chained", "named", "injected", "apply_every"])
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


@pytest.mark.parametrize("held", ["multi_steps", "apply_every"])
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


def test_an_input_the_optimizer_skips_trains_nothing_and_commits_no_tokens():
    """`optax.MultiSteps` with a skip predicate discards the inputs it skips: they move no weight, no
    average and no update count, and their tokens wait for no update, as its state says by moving
    neither its mini-step nor its gradient step; each skipped call draws afresh, so a predicate that
    rejects a draw meets another; one that skips none trains as `MultiSteps` alone does."""
    def skipping(*, skip: bool):
        def skipped(updates, gradient_step, params) -> tuple[jax.Array, tuple[()]]:
            # Its diagnostics empty, as a predicate may return them.
            return jnp.asarray(skip), ()

        return optax.MultiSteps(optax.sgd(0.1), 2, should_skip_update_fn=skipped).gradient_transformation()

    tracker = RecordingTracker()
    trainer = make_trainer(objective=Drawing(), optimizer=skipping(skip=True), tracker=tracker)
    first = trainer.initial_state()
    start = [np.asarray(leaf) for leaf in jax.tree.leaves((first.variables, first.ema))]
    state = trainer.fit(dataset(), steps=4, log_every=1)
    assert int(state.updates) == 0 and np.asarray(state.deferred).tolist() == [0, 0]
    assert all(np.array_equal(old, np.asarray(new)) for old, new in zip(
        start, jax.tree.leaves((state.variables, state.ema)), strict=True))
    assert [scalars["train/tokens"] for _, scalars in tracker.scalars if "train/tokens" in scalars] == [0] * 4
    assert len({scalars["train/draw"] for _, scalars in tracker.scalars if "train/draw" in scalars}) == 4
    kept = make_trainer(objective=TokenRegression(), optimizer=skipping(skip=False)).fit(dataset(), steps=4)
    plain = make_trainer(objective=TokenRegression(), optimizer=held_optimizer("multi_steps")).fit(
        dataset(), steps=4)
    for mine, theirs in zip(jax.tree.leaves(kept.variables), jax.tree.leaves(plain.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


REFUSED = ["finite", "conditional", "chained_mask", "same_phases", "nested", "masked", "partitioned",
           "conditioned", "followed", "decayed", "named_followed", "effects"]


@pytest.mark.parametrize("held", REFUSED)
def test_an_optimizer_whose_held_updates_no_state_tells_is_refused_before_any_step(held):
    """An optimizer that holds updates back where its state does not say which call applied one is
    refused before any step, as is a power-function average over it: `optax.apply_if_finite`, which
    applies a non-finite update past its limit unrecorded; `optax.conditionally_mask` anywhere but under
    `phased`, which says each update trains one of the networks it alternates, as its state does not hold
    its condition (alone, in a chain, or in a partition whose groups may all hold the same call); and a
    held update anywhere but over the whole update: within another,
    under a mask, a partition or a condition, or before a later transform in a chain, which moves the
    weights on a held call whether it keeps state or not. So is an objective with post-step effects
    under a held optimizer, whose held calls' effects no update applies. An optimizer that applies every
    update trains, the complementary phases of a partition's conditional masks under `phased` among
    them."""
    sgd = optax.sgd(0.1)
    multi = optax.MultiSteps(sgd, 2).gradient_transformation()

    def labelled(label: str):
        return lambda params: jax.tree.map(lambda _: label, params)

    def every(params):
        return jax.tree.map(lambda _: True, params)

    optimizer = {
        "finite": optax.apply_if_finite(sgd, max_consecutive_errors=2),
        "conditional": optax.conditionally_mask(sgd, every_other),
        "chained_mask": optax.chain(optax.conditionally_mask(sgd, every_other), optax.identity()),
        "same_phases": optax.partition({"a": optax.conditionally_mask(sgd, every_other),
                                        "b": optax.conditionally_mask(sgd, every_other)}, labelled("a")),
        "nested": optax.MultiSteps(multi, 2).gradient_transformation(),
        "masked": optax.masked(multi, every),
        "partitioned": optax.partition({"held": multi, "every": sgd}, labelled("held")),
        "conditioned": optax.conditionally_transform(multi, every_other),
        "followed": optax.chain(multi, optax.trace(0.9)),
        "decayed": optax.chain(multi, optax.add_decayed_weights(0.1)),
        "named_followed": optax.named_chain(("held", multi), ("momentum", optax.trace(0.9))),
        "effects": multi}[held]
    with pytest.raises(ValueError):
        make_trainer(objective=Effected() if held == "effects" else None, optimizer=optimizer).fit(
            dataset(), steps=2, log_every=1)
    if held != "effects":
        with pytest.raises(ValueError):
            power_profiles(optimizer, (0.05,)).init({"w": jnp.zeros(2)})
    def kernel_apart(params):
        return jax.tree_util.tree_map_with_path(
            lambda path, _: "student" if "kernel" in jax.tree_util.keystr(path) else "critic", params)

    # Two networks trained in turn, as a consistency objective's phases are: each call applies one's update.
    def critic(step, **_):
        return ~every_other(step)

    phases = phased(optax.partition({"student": optax.conditionally_mask(sgd, every_other),
                                     "critic": optax.conditionally_mask(sgd, critic)}, kernel_apart))
    assert int(make_trainer(objective=Effected(), optimizer=phases).fit(dataset(), steps=2,
                                                                        log_every=1).updates) == 2
