"""Shortcut models' training targets by their official code, for
tests/fixtures/shortcut.

kvfrans/shortcut-models' `get_targets` (targets_shortcut.py, read at a
pinned commit and run as published, in JAX like Dew) builds one batch of
self-consistency (bootstrap) and flow-matching targets from a closed-form
velocity network, the EMA one included. Its time runs from noise at 0 to
data at 1 and its network reads the step as a level, log2 of the steps it
takes; the test maps both. What lands is the batch and every target.

    PYTHONPATH=src python tools/shortcut_reference.py
"""

from __future__ import annotations

import json
import types
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

TARGETS = ("https://raw.githubusercontent.com/kvfrans/shortcut-models/"
           "601004348667094e1b71f30942199759412d4432/targets_shortcut.py")
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "shortcut"
MODEL = {"denoise_timesteps": 8, "bootstrap_every": 2, "bootstrap_dt_bias": 0, "bootstrap_cfg": 0,
         "bootstrap_ema": 1, "class_dropout_prob": 0.25, "num_classes": 10, "cfg_scale": 4.0}


def velocity(x, t, level, labels, train=False):
    """The closed-form velocity (data minus noise) the test runs too."""
    def column(value):
        return jnp.asarray(value, jnp.float32).reshape(-1, 1, 1, 1)

    return (jnp.tanh(x) * 0.5 + jnp.sin(2 * column(t)) * x * 0.3
            + 0.1 * column(level) * jnp.cos(x) + 0.05 * column(labels))


def main() -> None:
    scope: dict = {}
    exec(urllib.request.urlopen(TARGETS).read().decode(), scope)
    flags = types.SimpleNamespace(batch_size=12, model=MODEL)
    state = types.SimpleNamespace(call_model=velocity, call_model_ema=velocity)
    images = jax.random.normal(jax.random.PRNGKey(0), (12, 4, 4, 3))
    labels = jnp.arange(12) % MODEL["num_classes"]
    x_t, v_t, t, level, dropped, _ = scope["get_targets"](flags, jax.random.PRNGKey(3), state, images, labels)
    FIXTURE.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE / "targets.npz", x_t=np.asarray(x_t), v_t=np.asarray(v_t), t=np.asarray(t),
             level=np.asarray(level), labels=np.asarray(dropped), classes=np.asarray(labels),
             settings=np.asarray(json.dumps({**MODEL, "batch_size": 12})))
    print(f"{FIXTURE}: one batch of 12 shortcut targets, 6 bootstrapped")


if __name__ == "__main__":
    main()
