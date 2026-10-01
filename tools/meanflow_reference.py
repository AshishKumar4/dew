"""MeanFlow's training loss by its official code, for tests/fixtures/meanflow.

Gsunshine/meanflow's `MeanFlow` (meanflow.py, read at a pinned commit, the
class extracted and run as published, in JAX like Dew) trains a closed-form
average-velocity network through its own `forward`: the interval draw, the
guided velocity with omega and kappa, the condition dropout, the JVP target
and the adaptive weighting. A subclass records every draw the loss reads, so
the test replays them. What lands is the batch, the draws, both guided
velocities and the loss.

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

MEANFLOW = "https://raw.githubusercontent.com/Gsunshine/meanflow/d70cb55d298ee03c53bf6da67bec281082e4e2d9/meanflow.py"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "meanflow"
SETTINGS = {"omega": 1.5, "kappa": 0.3, "t_start": 0.1, "t_end": 0.9, "class_dropout_prob": 0.25,
            "data_proportion": 0.5, "norm_eps": 0.01, "num_classes": 10}


class Tiny(nn.Module):
    """The closed-form average velocity u(x, t, h, y) the test runs too."""

    @nn.compact
    def __call__(self, x, t, h, y, train=False, key=None):
        def column(value):
            return value.reshape(-1, 1, 1, 1)

        return (jnp.tanh(x) * 0.6 + jnp.sin(3 * column(t)) * x * 0.3
                + column(h) * jnp.cos(x) * 0.2 + column(y.astype(jnp.float32)) * 0.05)


class Numpy:
    """jax.numpy as the reference was written against: its `clip` still
    took `a_min` and `a_max`, which current JAX names `min` and `max`."""

    def __getattr__(self, name):
        return getattr(jnp, name)

    @staticmethod
    def clip(x, a_min=None, a_max=None):
        return jnp.clip(x, a_min, a_max)


def main() -> None:
    text = urllib.request.urlopen(MEANFLOW).read().decode()
    scope = {"nn": nn, "jax": jax, "jnp": Numpy(),
             "models_dit": types.SimpleNamespace(Tiny=lambda **kwargs: Tiny(name=kwargs["name"]))}
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == "MeanFlow":
            exec(ast.get_source_segment(text, node), scope)
    official = scope["MeanFlow"]

    class Recorded(official):
        def sample_tr(self, bz):
            t, r = super().sample_tr(bz)
            self.sow("intermediates", "t", t)
            self.sow("intermediates", "r", r)
            return t, r

        def guidance_fn(self, v_t, z_t, t, y, train=False):
            guided = super().guidance_fn(v_t, z_t, t, y, train=train)
            self.sow("intermediates", "z", z_t)
            self.sow("intermediates", "v", v_t)
            self.sow("intermediates", "guided", guided)
            return guided

        def cond_drop(self, v_t, v_g, labels):
            y_inp, dropped = super().cond_drop(v_t, v_g, labels)
            self.sow("intermediates", "labels", y_inp)
            self.sow("intermediates", "dropped", dropped)
            return y_inp, dropped

    x = jax.random.normal(jax.random.PRNGKey(0), (8, 4, 4, 3))
    labels = jnp.arange(8) % SETTINGS["num_classes"]
    FIXTURE.mkdir(parents=True, exist_ok=True)
    # norm_p 0 leaves the plain squared error, which the adaptive weight at
    # norm_p 1 flattens toward one per row; both are held.
    for norm_p in (0.0, 1.0):
        settings = {**SETTINGS, "norm_p": norm_p}
        model = Recorded(model_str="Tiny", model_config={}, **settings)
        variables = model.init({"params": jax.random.PRNGKey(1), "gen": jax.random.PRNGKey(2)},
                               x, jnp.ones((8,)), labels)
        (loss, _), recorded = model.apply(variables, x, labels, rngs={"gen": jax.random.PRNGKey(3)},
                                          method=model.forward, mutable=["intermediates"])
        arrays = {name: np.asarray(value[0]) for name, value in recorded["intermediates"].items()}
        np.savez(FIXTURE / f"loss_p{norm_p:g}.npz", x=np.asarray(x), classes=np.asarray(labels),
                 loss=np.asarray(loss), settings=np.asarray(json.dumps(settings)), **arrays)
    print(f"{FIXTURE}: MeanFlow losses over 8 rows at norm_p 0 and 1")


if __name__ == "__main__":
    main()
