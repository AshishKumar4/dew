"""A split step whose loss a window replays rather than pools (a `Weighted` loss, `Transaction.shared`
off): its window's commit replays each microbatch under the pooled loss's cotangent, the current one
realized afresh, so no microbatch's forward residuals live through a backward of its own."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from reference_error import assert_computes_the_oracle

from dew.data import Memory
from dew.data.pipeline import Pipeline
from dew.nn.kernels.generation import compile_only_gpus
from dew.objectives.base import Aux, Objective, Ratio, Weighted
from dew.state import TrainState
from dew.training import MeshSpec, Trainer
from dew.training.ahead import compile_ahead

WIDTH, ROWS = 64, 256
"""The width and rows of a float64 step on a CPU, whose arithmetic the replay witnesses compare."""


class Deep(nn.Module):
    """Four tanh layers of the input's width in `dtype`, whose activations a backward keeps."""

    dtype: jnp.dtype = jnp.float64

    @nn.compact
    def __call__(self, x):
        width, x = x.shape[-1], x.astype(self.dtype)
        for _ in range(4):
            x = jnp.tanh(nn.Dense(width, dtype=self.dtype, param_dtype=self.dtype)(x))
        return nn.Dense(width, dtype=self.dtype, param_dtype=self.dtype)(x)


class Squared(Objective):
    """Squared error against the inputs reversed, a `Ratio` over its entries, or where `weighted` the same
    `Ratio` as a `Weighted` loss of one term, which a window replays; where `terms` is two, a `Weighted`
    loss of two terms each over its own rows (those whose first, or second, input is positive), whose
    masses differ from microbatch to microbatch."""

    def __init__(self, weighted: bool, terms: int = 1, dtype: jnp.dtype = jnp.float64, width: int = WIDTH):
        self.model, self.weighted, self.terms, self.width = Deep(dtype), weighted, terms, width

    def fresh_variables(self, key, held):
        return self.model.init(key, jnp.zeros((1, self.width)))

    def loss(self, variables, batch, step):
        x = batch["x"]
        errors = self.model.apply(variables, x) - x[:, ::-1].astype(self.model.dtype)
        squares = errors ** 2
        if self.terms == 2:
            first, second = (x[:, 0] > 0).astype(errors.dtype), (x[:, 1] > 0.5).astype(errors.dtype)
            terms = (Ratio(jnp.sum(squares * first[:, None]), jnp.sum(first) * self.width),
                     Ratio(jnp.sum(jnp.abs(errors) * second[:, None]), jnp.sum(second) * self.width))
            return Weighted(terms, (1.0, 0.25)), Aux({})
        term = Ratio(jnp.sum(squares), jnp.asarray(squares.size, errors.dtype))
        return (Weighted((term,), (1.0,)) if self.weighted else term), Aux({})


def fitted(weighted: bool, terms: int = 1, split: int = 2) -> tuple[Trainer, TrainState]:
    """A trainer of `Squared` fit two steps of ROWS rows in float64, each read in `split` microbatches, and
    its state."""
    x = np.random.default_rng(0).standard_normal((4 * ROWS, WIDTH))
    trainer = Trainer(Squared(weighted, terms), optax.sgd(0.1), key=jax.random.key(0), split=split)
    with jax.enable_x64(new_val=True):
        return trainer, trainer.fit(Pipeline(Memory({"x": x})).repeat().batch(ROWS), steps=2)


def parameters(state: TrainState) -> np.ndarray:
    """Every parameter `state` holds, flat."""
    return np.concatenate([np.ravel(leaf) for leaf in jax.tree.leaves(state.variables["params"])])


def need_ahead(weighted: bool) -> int:
    """The bytes a float32 step of `Squared` over 4096 rows of 256, read in two microbatches, needs
    compiled ahead for an H100 (`compile_ahead`): its activations outweigh its gradients, as a model's do."""
    trainer = Trainer(Squared(weighted, dtype=jnp.float32, width=256), optax.sgd(0.1),
                      key=jax.random.key(0), mesh=MeshSpec().build(compile_only_gpus("sm90", 1, 80 * 2**30)),
                      split=2)
    ahead = compile_ahead(trainer, {"x": np.zeros((4096, 256), np.float32)})
    return ahead.plan["pool"]["limit"] - ahead.plan["probes"][-1]["headroom"]


def test_a_replayed_window_trains_as_the_pooled_one_and_needs_no_more():
    """A step whose `Weighted` loss of one term its window replays trains as the same loss written as a
    `Ratio`, which the window pools, to the rounding of its sums, and compiled for an H100 it needs no
    more than a sixth over the pooled one: the replay realizes the current microbatch again where its
    gradient is read, where it held this microbatch's forward residuals through a backward of their own
    and the replay of the others (a 4B decision step whose loss weighs a consistency term: 28.4 GiB a step
    against 18.5 GiB, at 8 layers)."""
    (_, pooled), (_, replayed) = fitted(weighted=False), fitted(weighted=True)
    assert_computes_the_oracle(parameters(replayed), parameters(pooled), "replayed against pooled",
                               roundings=ROWS * WIDTH)
    needs = [need_ahead(weighted=False), need_ahead(weighted=True)]
    assert needs[1] <= needs[0] * 7 // 6, needs


def test_a_replayed_window_of_two_terms_trains_as_its_whole_batch():
    """A `Weighted` loss of two terms, each over its own rows, so each microbatch's masses differ from the
    other's: its step read in two microbatches, which the window replays under the pooled loss's
    cotangent, trains as the step read whole, to the rounding of its sums, each term divided by its own whole
    batch's mass."""
    _, whole = fitted(weighted=True, terms=2, split=1)
    _, split = fitted(weighted=True, terms=2, split=2)
    assert_computes_the_oracle(parameters(split), parameters(whole), "split against whole",
                               roundings=ROWS * WIDTH)
