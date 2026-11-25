"""A split step whose loss a window replays rather than pools (a `Weighted` loss, `Transaction.shared`
off): its window's commit replays each microbatch under the pooled loss's cotangent, the current one
realized afresh, so no microbatch's forward residuals live through a backward of its own."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from dew.data import Memory
from dew.data.pipeline import Pipeline
from dew.objectives.base import Aux, Objective, Ratio, Weighted
from dew.state import TrainState
from dew.training import Trainer

WIDTH, ROWS = 256, 64


class Deep(nn.Module):
    """Four tanh layers of WIDTH, whose activations a backward keeps."""

    @nn.compact
    def __call__(self, x):
        for _ in range(4):
            x = jnp.tanh(nn.Dense(WIDTH)(x))
        return nn.Dense(WIDTH)(x)


class Squared(Objective):
    """Squared error against the inputs reversed, a `Ratio` over its entries, or where `weighted` the same
    `Ratio` as a `Weighted` loss of one term, which a window replays; where `terms` is two, a `Weighted`
    loss of two terms each over its own rows (those whose first, or second, input is positive), whose
    masses differ from microbatch to microbatch."""

    def __init__(self, weighted: bool, terms: int = 1):
        self.model, self.weighted, self.terms = Deep(), weighted, terms

    def fresh_variables(self, key, held):
        return self.model.init(key, jnp.zeros((1, WIDTH)))

    def loss(self, variables, batch, step):
        x = batch["x"]
        errors = self.model.apply(variables, x) - x[:, ::-1]
        squares = errors ** 2
        if self.terms == 2:
            first, second = (x[:, 0] > 0).astype(x.dtype), (x[:, 1] > 0.5).astype(x.dtype)
            terms = (Ratio(jnp.sum(squares * first[:, None]), jnp.sum(first) * WIDTH),
                     Ratio(jnp.sum(jnp.abs(errors) * second[:, None]), jnp.sum(second) * WIDTH))
            return Weighted(terms, (1.0, 0.25)), Aux({})
        term = Ratio(jnp.sum(squares), jnp.asarray(squares.size, squares.dtype))
        return (Weighted((term,), (1.0,)) if self.weighted else term), Aux({})


def fitted(weighted: bool, terms: int = 1, split: int = 2) -> tuple[Trainer, TrainState]:
    """A trainer of `Squared` fit two steps of ROWS rows, each read in `split` microbatches, and its
    state."""
    x = np.random.default_rng(0).standard_normal((4 * ROWS, WIDTH)).astype(np.float32)
    trainer = Trainer(Squared(weighted, terms), optax.sgd(0.1), key=jax.random.key(0), split=split)
    return trainer, trainer.fit(Pipeline(Memory({"x": x})).repeat().batch(ROWS), steps=2)


def test_a_replayed_window_trains_as_the_pooled_one_and_holds_no_more():
    """A step whose `Weighted` loss of one term its window replays trains as the same loss written as a
    `Ratio`, which the window pools, to float rounding, and its compiled step holds no more temporaries
    than a sixth over the pooled one's: the replay realizes the current microbatch again where its
    gradient is read, where it held this microbatch's forward residuals through a backward of their own
    and the replay of the others (a 4B decision step whose loss weighs a consistency term: 28.4 GiB a step
    against 18.5 GiB, at 8 layers)."""
    (pooled, pooled_state), (replayed, replayed_state) = fitted(weighted=False), fitted(weighted=True)
    for one, other in zip(jax.tree.leaves(pooled_state.variables["params"]),
                          jax.tree.leaves(replayed_state.variables["params"]), strict=True):
        np.testing.assert_allclose(np.asarray(other), np.asarray(one), rtol=1e-5, atol=1e-6)
    held = [trainer.plan["probes"][-1]["temp"] for trainer in (pooled, replayed)]
    assert all(isinstance(temp, int) for temp in held), held
    assert held[1] <= held[0] * 7 // 6, held


def test_a_replayed_window_of_two_terms_trains_as_its_whole_batch():
    """A `Weighted` loss of two terms, each over its own rows, so each microbatch's masses differ from the
    other's: its step read in two microbatches, which the window replays under the pooled loss's
    cotangent, trains as the step read whole, to float rounding, each term divided by its own whole
    batch's mass."""
    _, whole = fitted(weighted=True, terms=2, split=1)
    _, split = fitted(weighted=True, terms=2, split=2)
    for one, other in zip(jax.tree.leaves(whole.variables["params"]),
                          jax.tree.leaves(split.variables["params"]), strict=True):
        np.testing.assert_allclose(np.asarray(other), np.asarray(one), rtol=1e-5, atol=1e-6)
