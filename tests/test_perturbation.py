"""Training without backpropagation by node perturbation (`dew.objectives.perturbation`).

The estimate is the loss's gradient in expectation where the loss is quadratic in each layer's output
and every unit's loss reads its own output alone: an antithetic pair's reward is then exactly the
derivative along its jitter. Over many independent steps' keys, the estimates are the gradient JAX
computes in expectation: a t-test rejects in no coordinate. A run trains with no
derivative of its model taken, which a layer that refuses its backward proves, and a model whose
layers are mapped hides their calls from the capture and is refused.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from scipy import stats

from dew.data import Memory
from dew.inputs import Field, InputSpec
from dew.objectives.base import Step
from dew.objectives.perturbation import NodePerturbation
from dew.objectives.supervised import Supervised
from dew.training import Layout, Trainer

LAYOUT = Layout(min_shard=1, tolerance=1.0)
INPUTS = InputSpec(Field("x", (3,)))


def squared(outputs, batch):
    """Each example's squared error against its target."""
    return jnp.sum(jnp.square(outputs - batch["y"]), -1)


class Forward(NodePerturbation, Supervised):
    """`Supervised` trained by node perturbation, 64 pairs a step."""

    draws = 64


@jax.custom_vjp
def forward_only(x):
    return x


def _refused(x):
    return x, None


def _backward(_, cotangent):
    raise RuntimeError("the model was differentiated")


forward_only.defvjp(_refused, _backward)


class Linear(nn.Module):
    """Two Dense layers with no nonlinearity between them, and a scale no Dense holds; with
    `refuse_backward`, a model whose derivative raises."""

    refuse_backward: bool = False

    @nn.compact
    def __call__(self, x):
        hidden = nn.Dense(4)(x)
        out = nn.Dense(2)(hidden) * self.param("scale", nn.initializers.ones, ())
        return forward_only(out) if self.refuse_backward else out


def rows(seed: int = 0) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    x = generator.normal(size=(16, 3)).astype(np.float32)
    return {"x": x, "y": (x @ generator.normal(size=(3, 2))).astype(np.float32)}


def test_the_estimate_is_the_gradient_in_expectation():
    """Over 256 steps' keys, each estimate of 64 antithetic pairs, the estimates of the two Dense layers'
    kernels and biases, and of the scale weight-space jitters estimate, are `jax.grad`'s in expectation:
    Student's t-test does not reject at one in a billion in any coordinate."""
    objective = Forward(Linear(), squared, inputs=INPUTS)
    variables = objective.init(jax.random.key(0))
    batch = {name: jnp.asarray(value) for name, value in rows().items()}

    def estimate(key):
        return objective.estimate(variables, batch, Step(jnp.asarray(0), key, None))

    estimates = jax.vmap(jax.jit(estimate))(jax.random.split(jax.random.key(1), 256))

    def summed(params):
        return jnp.sum(squared(objective.model.apply({"params": params}, batch["x"]), batch))

    truth = jax.grad(summed)(variables["params"])
    for path in (("Dense_0", "kernel"), ("Dense_0", "bias"), ("Dense_1", "kernel"), ("Dense_1", "bias"),
                 ("scale",)):
        drawn, exact = estimates, truth
        for key in path:
            drawn, exact = drawn[key], exact[key]
        # A one-sample t-test of the estimates against the gradient, coordinate by coordinate.
        tested = stats.ttest_1samp(np.asarray(drawn, np.float64), np.asarray(exact, np.float64))
        assert tested.pvalue.min() > 1e-9, path


def test_a_run_trains_without_differentiating_its_model():
    """A run of a model whose backward raises trains by node perturbation for 40 steps, moving every
    parameter, the scale no Dense holds among them, where the same objective by backpropagation fails at
    its first step: no derivative of the model is taken. (That the steps descend is the estimate's
    expectation, held above.)"""
    data = rows()
    batches = Memory([{"x": x, "y": y} for x, y in zip(data["x"], data["y"], strict=True)]).repeat().batch(16)
    refusing = Linear(refuse_backward=True)
    with pytest.raises(RuntimeError):
        Trainer(Supervised(refusing, squared, inputs=INPUTS), optax.sgd(0.01), key=0, layout=LAYOUT).fit(
            batches, steps=1)
    objective = Forward(refusing, squared, inputs=INPUTS)
    start = objective.init(jax.random.key(0))
    state = Trainer(objective, optax.sgd(0.01), key=0, layout=LAYOUT).fit(batches, steps=40)
    for moved, held in zip(jax.tree.leaves(state.variables), jax.tree.leaves(start), strict=True):
        assert not np.array_equal(np.asarray(moved), np.asarray(held))


class Mapped(nn.Module):
    """A Dense mapped over its rows, its calls inside the map."""

    @nn.compact
    def __call__(self, x):
        return nn.vmap(nn.Dense, variable_axes={"params": None}, split_rngs={"params": False})(2)(x)


def test_a_model_that_maps_its_layers_is_refused():
    """A Dense under `nn.vmap` is called inside the map, where a capture outside cannot read it: the
    estimate is refused, naming the unrolled model it needs."""
    objective = Forward(Mapped(), squared, inputs=INPUTS)
    variables = objective.init(jax.random.key(0))
    batch = {name: jnp.asarray(value) for name, value in rows().items()}
    with pytest.raises(ValueError):
        objective.estimate(variables, batch, Step(jnp.asarray(0), jax.random.key(0), None))
