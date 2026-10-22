"""Probe (not for landing): per-seed errors of the KDA gradient reference for the gate leaves."""
import jax
import jax.numpy as jnp
import numpy as np
from reference_error import widened
from test_kda import ROWS, S, E, layer_and_params, oracle_layer, oracle_params
from functools import partial
from tools.kda_reference import oracle_chunk


def test_probe():
    def loss_oracle(x, p, cotangent):
        return jnp.sum(oracle_layer(x, p, -5.0, 1e-5, partial(oracle_chunk, chunk_size=4)) * cotangent)

    theirs = jax.jit(lambda x, p, c: jax.grad(loss_oracle, argnums=(0, 1))(x, p, c)[1])
    for seed in range(24):
        rng = np.random.RandomState(seed)
        params = oracle_params(layer_and_params(-5.0, seed=1000 + seed)[2])
        x, cotangent = rng.randn(2, 8 * ROWS, S, E).astype(np.float32)
        found = theirs(x, params, cotangent)
        with jax.enable_x64(True):
            truth = theirs(*widened((x, params, cotangent)))
            for name in ("A_log", "dt_bias", "b_proj", "f_a_proj", "q_proj"):
                err = np.asarray(found[name], np.float64) - np.asarray(truth[name])
                print("PROBE", seed, name, f"{np.sqrt(np.mean(err**2)):.3e}", f"{np.sqrt(np.mean(np.asarray(truth[name])**2)):.3e}",
                      bool(np.isfinite(np.asarray(found[name])).all()), flush=True)
