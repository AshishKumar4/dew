"""Probe (not for landing): the spread of the KDA gate gradients' reference errors."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import widened
from test_kda import GROUPS, S, E, layer_and_params, oracle_layer, oracle_params
from tools.kda_reference import oracle_chunk


def test_probe():
    def loss_oracle(x, p, cotangent):
        return jnp.sum(oracle_layer(x, p, -5.0, 1e-5, partial(oracle_chunk, chunk_size=4)) * cotangent)

    theirs = jax.jit(jax.vmap(lambda x, p, c: jax.grad(loss_oracle, argnums=(0, 1))(x, p, c)))
    params = oracle_params(layer_and_params(-5.0)[2])
    stacked = jax.tree.map(lambda leaf: np.broadcast_to(leaf, (GROUPS, *leaf.shape)), params)
    errors = {name: [] for name in ("A_log", "dt_bias", "x")}
    for seed in range(64):
        x, cotangent = np.random.RandomState(seed).randn(2, GROUPS, 16, S, E).astype(np.float32)
        gx, found = theirs(x, stacked, cotangent)
        found = {"x": gx, **found}
        with jax.enable_x64(True):
            tx, truth = theirs(*widened((x, stacked, cotangent)))
            truth = {"x": tx, **truth}
            for name in errors:
                err = np.asarray(found[name], np.float64) - np.asarray(truth[name])
                errors[name].append(np.sqrt(np.mean(err.reshape(GROUPS, -1) ** 2, axis=1)))
    for name, values in errors.items():
        values = np.concatenate(values)
        q = np.quantile(values, [0.5, 0.9, 0.99, 1.0])
        print("PROBE", name, " ".join(f"{v:.2e}" for v in q), f"mean2/median2 {np.mean(values**2)/np.median(values)**2:.1f}",
              flush=True)
        worst = np.argsort(values)[-5:]
        print("PROBE worst", name, worst // GROUPS, worst % GROUPS, flush=True)
