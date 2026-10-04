"""Shortcut models' training loss and its gradient by their official code, for
tests/fixtures/shortcut.

kvfrans/shortcut-models at a pinned commit, run as published, in JAX like
Dew: `get_targets` (targets_shortcut.py) builds one batch of
self-consistency (bootstrap) targets, two half steps of the EMA network,
and flow-matching targets with the condition dropped, and `loss_fn`, read
out of train.py's `update`, scores the network against them. A closed-form
velocity network with a weight per term and pixel stands in for the model,
its EMA copy holding other weights; the loss is differentiated in the
network's weights as `update` differentiates it. `update`'s shuffle of the
batch is the data's order and is left out.

`get_targets` reads `jax.random` for every draw; here that name records each
draw in the float32 run and hands the same draws, widened, to the float64
one, so both runs see one set of inputs. Its time runs from noise at 0 to
data at 1 and its network reads the step as a level, log2 of the steps it
takes; the test maps both. What lands: the pixels and their float32 image,
the classes, the draws, both weights, every target, and the loss and
gradient in float32 and in float64.

    PYTHONPATH=src python tools/shortcut_reference.py
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

COMMIT = "601004348667094e1b71f30942199759412d4432"
SOURCE = f"https://raw.githubusercontent.com/kvfrans/shortcut-models/{COMMIT}/"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "shortcut"
MODEL = {"denoise_timesteps": 8, "bootstrap_every": 2, "bootstrap_dt_bias": 0, "bootstrap_cfg": 0,
         "bootstrap_ema": 1, "class_dropout_prob": 0.25, "num_classes": 10, "cfg_scale": 4.0,
         "train_type": "shortcut"}
SHAPE = (12, 4, 4, 3)


def source(path: str) -> str:
    return urllib.request.urlopen(SOURCE + path).read().decode()


class Draws:
    """`jax.random` as `get_targets` calls it: each draw recorded, or, given
    recorded draws, the next one, a float one widened to `dtype`."""

    def __init__(self, dtype, replayed: list[np.ndarray] | None = None):
        self.dtype = dtype
        self.drawn: list[np.ndarray] = []
        self.replayed = None if replayed is None else list(replayed)

    def __getattr__(self, name):
        return getattr(jax.random, name)

    def _next(self, value):
        if self.replayed is not None:
            value = jnp.asarray(self.replayed.pop(0))
        else:
            self.drawn.append(np.asarray(value))
        return value.astype(self.dtype) if jnp.issubdtype(value.dtype, jnp.floating) else value

    def normal(self, key, shape=()):
        return self._next(jax.random.normal(key, shape, jnp.float32))

    def randint(self, key, shape, minval, maxval):
        return self._next(jax.random.randint(key, shape, minval, maxval))

    def bernoulli(self, key, p, shape):
        return self._next(jax.random.bernoulli(key, p, shape))


def velocity(weights, x, t, level, labels):
    """The closed-form velocity (data minus noise) the test runs too: four
    terms, each with a weight per pixel and channel."""
    def column(value):
        return jnp.asarray(value, x.dtype).reshape(-1, 1, 1, 1)

    return (jnp.tanh(x) * weights[0] + jnp.sin(2 * column(t)) * x * weights[1]
            + column(level) * jnp.cos(x) * weights[2] + column(labels) * weights[3])


def run(dtype, images, labels, weights, teacher, replayed=None):
    # The reference is written for JAX's default float32: its untyped arrays
    # follow the run's precision.
    jax.config.update("jax_enable_x64", val=dtype == jnp.float64)
    images, weights, teacher = (jnp.asarray(value, dtype) for value in (images, weights, teacher))
    draws = Draws(dtype, replayed)
    targets: dict = {}
    exec(source("targets_shortcut.py"), targets)
    targets["jax"] = types.SimpleNamespace(**{**vars(jax), "random": draws})
    text = source("train.py")
    update = next(node for node in ast.walk(ast.parse(text))
                  if isinstance(node, ast.FunctionDef) and node.name == "loss_fn")
    flags = types.SimpleNamespace(batch_size=images.shape[0], model=MODEL)

    def call_model(x, t, level, labels, train=False, rngs=None, params=None, return_activations=False):
        output = velocity(params, x, t, level, labels)
        return (output, None, {}) if return_activations else output

    def call_model_ema(*args, **kwargs):
        return call_model(*args, params=teacher, **kwargs)

    state = types.SimpleNamespace(call_model=call_model, call_model_ema=call_model_ema)
    x_t, v_t, t, level, dropped, _ = targets["get_targets"](flags, jax.random.PRNGKey(3), state, images,
                                                             labels)
    scope = {"jnp": jnp, "jax": jax, "FLAGS": flags, "train_state": state, "x_t": x_t, "t": t,
             "dt_base": level, "labels": dropped, "v_t": v_t, "dropout_key": jax.random.PRNGKey(4)}
    exec(ast.get_source_segment(text, update), scope)
    (loss, _), gradient = jax.value_and_grad(scope["loss_fn"], has_aux=True)(weights)
    assert loss.dtype == dtype and gradient.dtype == dtype, (loss.dtype, dtype)
    return {"x_t": x_t, "v_t": v_t, "t": t, "level": level, "labels": dropped, "loss": loss,
            "grad": gradient}, draws.drawn


def main() -> None:
    generator = np.random.default_rng(1)
    pixels = generator.integers(0, 256, SHAPE, dtype=np.uint8)
    x = (pixels.astype(np.float32) - np.float32(127.5)) / np.float32(127.5)
    labels = np.arange(SHAPE[0]) % MODEL["num_classes"]
    weights, teacher = ((generator.standard_normal((4, *SHAPE[1:])) * 0.3).astype(np.float32)
                        for _ in range(2))
    out, drawn = run(jnp.float32, x, labels, weights, teacher)
    wide, _ = run(jnp.float64, x, labels, weights, teacher, replayed=drawn)
    bootstrap_times, bootstrap_noise, dropout, flow_times, flow_noise = drawn
    FIXTURE.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE / "targets.npz", pixels=pixels, x=x, classes=np.asarray(labels), weights=weights,
             teacher=teacher, settings=np.asarray(json.dumps({**MODEL, "batch_size": SHAPE[0]})),
             bootstrap_times=bootstrap_times, bootstrap_noise=bootstrap_noise, dropout=dropout,
             flow_times=flow_times, flow_noise=flow_noise,
             **{name: np.asarray(value) for name, value in out.items()},
             loss_f64=np.asarray(wide["loss"]), grad_f64=np.asarray(wide["grad"]),
             v_t_f64=np.asarray(wide["v_t"]))
    print(f"{FIXTURE}: one batch of {SHAPE[0]} shortcut rows, {SHAPE[0] // MODEL['bootstrap_every']} "
          "bootstrapped, with its loss and gradient")


if __name__ == "__main__":
    main()
