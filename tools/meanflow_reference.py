"""MeanFlow's training loss and its gradient by the official code, for
tests/fixtures/meanflow.

Gsunshine/meanflow's `MeanFlow` (meanflow.py, read at a pinned commit, the
class extracted and run as published, in JAX like Dew) trains a closed-form
average-velocity network with a weight per term and pixel through its own
`forward`: the interval draw, the guided velocity with omega and kappa, the
condition dropout, the JVP target and the adaptive weighting, and the
gradient of that loss in the network's weights. The class reads
`jax.random` for every draw; here that name records each draw in the
float32 run and hands the same draws, widened, to the float64 one, so both
runs see one set of inputs. A subclass records the two time draws the
interval is made of. What lands: the pixels and their float32 image, the
classes, the draws, the weights, and the loss and gradient in float32 and in
float64, at norm_p 0 and 1.

    PYTHONPATH=src python tools/meanflow_reference.py
"""

from __future__ import annotations

import ast
import json
import types
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

jax.config.update("jax_enable_x64", val=True)

MEANFLOW = "https://raw.githubusercontent.com/Gsunshine/meanflow/d70cb55d298ee03c53bf6da67bec281082e4e2d9/meanflow.py"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "meanflow"
SETTINGS = {"omega": 1.5, "kappa": 0.3, "t_start": 0.3, "t_end": 0.7, "class_dropout_prob": 0.25,
            "data_proportion": 0.5, "norm_eps": 0.01, "num_classes": 10}
SHAPE = (16, 4, 4, 3)


class Tiny(nn.Module):
    """The closed-form average velocity u(x, t, h, y) the test runs too: four
    terms, each with a weight per pixel and channel."""

    @nn.compact
    def __call__(self, x, t, h, y, train=False, key=None):
        def column(value):
            return value.reshape(-1, 1, 1, 1)

        weights = self.param("weights", nn.initializers.zeros, (4, *x.shape[1:]))
        return (jnp.tanh(x) * weights[0] + jnp.sin(3 * column(t)) * x * weights[1]
                + column(h) * jnp.cos(x) * weights[2] + column(y.astype(x.dtype)) * weights[3])


class Numpy:
    """jax.numpy as the reference was written against: its `clip` still
    took `a_min` and `a_max`, which current JAX names `min` and `max`."""

    def __getattr__(self, name):
        return getattr(jnp, name)

    @staticmethod
    def clip(x, a_min=None, a_max=None):
        return jnp.clip(x, a_min, a_max)


class Draws:
    """`jax.random` as the reference calls it: each draw recorded, or, given
    recorded draws, the next one widened to the dtype asked for."""

    def __init__(self, replayed: list[np.ndarray] | None = None):
        self.drawn: list[np.ndarray] = []
        self.replayed = None if replayed is None else list(replayed)

    def __getattr__(self, name):
        return getattr(jax.random, name)

    def _next(self, value, dtype):
        if self.replayed is not None:
            return jnp.asarray(self.replayed.pop(0), dtype)
        self.drawn.append(np.asarray(value))
        return value

    def normal(self, key, shape=(), dtype=jnp.float32):
        return self._next(jax.random.normal(key, shape, dtype=jnp.float32), dtype)

    def uniform(self, key, shape=(), dtype=jnp.float32, minval=0.0, maxval=1.0):
        return self._next(jax.random.uniform(key, shape, jnp.float32, minval, maxval), dtype)


def published(draws: Draws, dtype) -> type:
    text = urllib.request.urlopen(MEANFLOW).read().decode()
    scope = {"nn": nn, "jax": types.SimpleNamespace(**{**vars(jax), "random": draws}), "jnp": Numpy(),
             "models_dit": types.SimpleNamespace(Tiny=lambda **kwargs: Tiny(name=kwargs["name"]))}
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == "MeanFlow":
            exec(ast.get_source_segment(text, node), scope)

    class Recorded(scope["MeanFlow"]):
        def _logit_normal_dist(self, bz):
            drawn = super()._logit_normal_dist(bz)
            self.sow("intermediates", "times", drawn)
            return drawn

    Recorded.dtype = dtype
    return Recorded


def run(settings: dict, dtype, x, labels, weights, replayed=None):
    draws = Draws(replayed)
    model = published(draws, dtype)(model_str="Tiny", model_config={}, **settings)

    def loss(weights):
        (value, _), recorded = model.apply(
            {"params": {"net": {"weights": weights}}}, jnp.asarray(x, dtype), labels,
            rngs={"gen": jax.random.PRNGKey(3)}, method=model.forward, mutable=["intermediates"])
        return value, recorded["intermediates"]["times"]

    (value, times), gradient = jax.value_and_grad(loss, has_aux=True)(jnp.asarray(weights, dtype))
    assert value.dtype == dtype and gradient.dtype == dtype, (value.dtype, dtype)
    return value, gradient, times, draws.drawn


def main() -> None:
    generator = np.random.default_rng(0)
    pixels = generator.integers(0, 256, SHAPE, dtype=np.uint8)
    x = (pixels.astype(np.float32) - np.float32(127.5)) / np.float32(127.5)
    labels = jnp.asarray(np.arange(SHAPE[0]) % SETTINGS["num_classes"])
    weights = (generator.standard_normal((4, *SHAPE[1:])) * 0.3).astype(np.float32)
    FIXTURE.mkdir(parents=True, exist_ok=True)
    # norm_p 0 leaves the plain squared error, which the adaptive weight at
    # norm_p 1 flattens toward one per row; both are held.
    for norm_p in (0.0, 1.0):
        settings = {**SETTINGS, "norm_p": norm_p}
        loss, gradient, times, drawn = run(settings, jnp.float32, x, labels, weights)
        loss64, gradient64, _, _ = run(settings, jnp.float64, x, labels, weights, replayed=drawn)
        later, earlier, noise, uniform = drawn
        np.savez(FIXTURE / f"loss_p{norm_p:g}.npz", pixels=pixels, x=x, classes=np.asarray(labels),
                 weights=weights, settings=np.asarray(json.dumps(settings)),
                 later=np.asarray(times[0]).ravel(), earlier=np.asarray(times[1]).ravel(),
                 normals=np.stack([later.ravel(), earlier.ravel()]), noise=noise, uniform=uniform,
                 loss=np.asarray(loss), loss_f64=np.asarray(loss64),
                 grad=np.asarray(gradient), grad_f64=np.asarray(gradient64))
    print(f"{FIXTURE}: MeanFlow loss and gradient over {SHAPE[0]} rows at norm_p 0 and 1")


if __name__ == "__main__":
    main()
