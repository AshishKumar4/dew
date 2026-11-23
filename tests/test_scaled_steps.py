"""A dynamically scaled step's acceptance: an accepted step is the unscaled run's, bit for bit, and a
rejected one leaves no trace, its choice read on the loss and gradient alone (row 6748)."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.data import Memory
from dew.data.pipeline import Pipeline
from dew.objectives.base import Aux, EMASpec, Objective
from dew.training import FitPolicy, Trainer, UpdateBatch


class ScaledObjective(Objective):
    """loss = scale * sum(w^2), with the scale carried by the batch so that
    one batch overflows the scaled float32 loss while the params stay sane."""

    def __init__(self):
        self.ema = EMASpec(decay=lambda step: 0.5)

    def fresh_variables(self, key, held):
        return {"params": {"w": jnp.ones((2,))}}

    def loss(self, variables, batch, step):
        return jnp.sum(variables["params"]["w"] ** 2) * batch["scale"][0], Aux({})


def held_weights(state):
    return np.array(state.variables["params"]["w"]), np.array(state.ema["params"]["w"])


def unscaled_weights(accum: int, microsteps: int):
    """The weights and their average after `microsteps` good microbatches of the same trainer with no
    dynamic scale: what a scaled run's accepted steps must equal bit for bit, its scale a power of two
    that multiplies the loss and divides the gradients exactly."""
    trainer = Trainer(ScaledObjective(), optax.sgd(.1), key=jax.random.key(0),
                      update=UpdateBatch(batches=accum))
    state = trainer.initial_state()
    good = {"scale": jnp.ones((jax.device_count(),), jnp.float32)}
    step = trainer.compile(state, good)
    for _ in range(microsteps):
        state, *_ = step(state, good)
    return held_weights(state)


def test_a_rejected_dynamic_scale_step_leaves_adams_state_as_it_was():
    """A scaled step whose loss is not finite, under Adam, leaves the parameters, their average and the
    optimizer's count and moments bit for bit as they were: the optimizer takes no step for it, not even one
    of zero gradient (row 6748, the choice read on the loss and gradient alone)."""
    trainer = Trainer(ScaledObjective(), optax.adam(.1), key=jax.random.key(0), dynamic_scale=True)
    state = trainer.initial_state()
    good = {"scale": jnp.ones((jax.device_count(),), jnp.float32)}
    bad = {"scale": jnp.full((jax.device_count(),), 1e35, jnp.float32)}
    step = trainer.compile(state, good)
    state, *_ = step(state, good)
    before = jax.device_get((state.variables, state.ema, state.opt_state))
    state, _, _, _, accepted, _ = step(state, bad)
    assert not bool(accepted)
    for got, kept in zip(jax.tree.leaves(jax.device_get((state.variables, state.ema, state.opt_state))),
                         jax.tree.leaves(before), strict=True):
        np.testing.assert_array_equal(got, kept)


def test_a_dynamic_scale_step_is_accepted_on_its_gradient_finite():
    """A scaled step whose loss and gradient are finite is accepted though its update overflows the
    parameters, as GradScaler's `found_inf` reads the gradients alone: the choice is made before the update
    reads them, so it runs in place, holding no second state (row 6748); the run stops on the non-finite
    state at its next log. Reading the new state's finiteness rejected it."""
    trainer = Trainer(ScaledObjective(), optax.sgd(1e30), key=jax.random.key(0), dynamic_scale=True)
    state = trainer.initial_state()
    # A gradient of 1e10, its norm finite, which a step of 1e30 carries past float32.
    huge = {"scale": jnp.full((jax.device_count(),), 5e9, jnp.float32)}
    state, _, _, finite, accepted, _ = trainer.compile(state, huge)(state, huge)
    assert bool(finite) and bool(accepted)
    assert not np.all(np.isfinite(np.asarray(state.variables["params"]["w"])))


@pytest.mark.parametrize("split", [1, 2])
def test_a_scaled_steps_state_report_names_the_leaf_its_committed_state_holds(split):
    """A scaled step whose update overflows its state reports its committed state's first leaf not finite
    and how many are not, as the state it leaves holds them, whatever microbatches it steps in: a step of
    two parts reports its last part's, where the two parts' reports were summed and named another leaf
    (row 8269)."""
    from dew.training.transaction import checked

    trainer = Trainer(ScaledObjective(), optax.sgd(1e30), key=jax.random.key(0), dynamic_scale=True,
                      fit=FitPolicy(split=split))
    state = trainer.initial_state()
    huge = {"scale": jnp.full((split * jax.device_count(),), 5e9, jnp.float32)}
    state, _, _, _, accepted, committed = trainer.compile(state, huge)(state, huge)
    assert bool(accepted)
    finite = [bool(np.all(np.isfinite(np.asarray(leaf)))) for leaf in jax.tree.leaves(checked(state))]
    assert np.asarray(committed.unfinite_state).tolist() == [finite.index(False), finite.count(False)]


@pytest.mark.parametrize("accum", [1, 2])
def test_a_rejected_dynamic_scale_step_leaves_no_trace(accum):
    """A dynamic scale's accepted steps are the unscaled run's, bit for bit, and a rejected one leaves
    no trace: after it, the run steps on as the unscaled run of its accepted microbatches alone."""
    trainer = Trainer(ScaledObjective(), optax.sgd(.1), key=jax.random.key(0),
                      update=UpdateBatch(batches=accum), dynamic_scale=True)
    state = trainer.initial_state()
    good = {"scale": jnp.ones((jax.device_count(),), jnp.float32)}
    bad = {"scale": jnp.full((jax.device_count(),), 1e35, jnp.float32)}
    step = trainer.compile(state, good)
    for _ in range(2 * accum - 1):
        state, *_ = step(state, good)
    w, ema = held_weights(state)
    assert np.all(w != 1.0)
    for got, want in zip((w, ema), unscaled_weights(accum, 2 * accum - 1), strict=True):
        np.testing.assert_array_equal(got, want)
    # the step consumes the state
    counted, advanced, scaled = int(state.step), int(state.microstep), float(state.scale.scale)
    state, _, _, finite, accepted, _ = step(state, bad)
    assert bool(finite) and not bool(accepted)
    assert int(state.step) == counted + 1
    # The rejection drops its window, the accepted microbatch before it with it (`dropped_window`).
    assert int(state.microstep) == advanced - advanced % accum
    assert float(state.scale.scale) == scaled / 2
    np.testing.assert_array_equal(state.variables["params"]["w"], w)
    np.testing.assert_array_equal(state.ema["params"]["w"], ema)
    for _ in range(accum):
        state, *_ = step(state, good)
    w, ema = held_weights(state)
    # Two accepted updates: the rejection dropped its window, the pending microbatch with it.
    for got, want in zip((w, ema), unscaled_weights(accum, 2 * accum), strict=True):
        np.testing.assert_array_equal(got, want)
    assert int(state.microstep) == 2 * accum


def test_a_gradient_that_would_overflow_adams_second_moment_is_rejected_by_its_norm():
    """A scaled step whose gradient's square passes float32 (1e20 here) is rejected: its global norm, a sum of
    the squares, is not finite, so no accepted step's gradient can carry Adam's second moment, a running
    average of those squares, past float32, and the moments stay finite; the step's state is kept bit for
    bit. One just under (1e19) is accepted."""
    for scale, taken in ((5e18, True), (5e19, False)):
        trainer = Trainer(ScaledObjective(), optax.adam(1e-3), key=jax.random.key(0), dynamic_scale=True)
        state = trainer.initial_state()
        batch = {"scale": jnp.full((jax.device_count(),), scale, jnp.float32)}
        before = jax.device_get(state.opt_state)
        state, _, _, _, accepted, _ = trainer.compile(state, batch)(state, batch)
        assert bool(accepted) == taken
        assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(state.opt_state))
        if not taken:
            for got, kept in zip(jax.tree.leaves(jax.device_get(state.opt_state)), jax.tree.leaves(before),
                                 strict=True):
                np.testing.assert_array_equal(got, kept)


def test_a_scaled_run_whose_update_overflows_its_state_stops_at_the_next_log():
    """A scaled run whose first update, from a finite gradient, overflows its parameters stops at the log
    after it, naming the leaf, rather than reading on as its scale backs off (row 6748): its committed
    state's finiteness is read after each commit."""
    rows = Pipeline(Memory({"scale": np.full(4 * jax.device_count(), 5e9, np.float32)})).repeat()
    trainer = Trainer(ScaledObjective(), optax.sgd(1e30), key=jax.random.key(0), dynamic_scale=True)
    with pytest.raises(RuntimeError):
        trainer.fit(rows.batch(jax.device_count()), steps=4, log_every=1)
