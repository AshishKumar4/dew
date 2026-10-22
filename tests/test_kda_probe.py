"""Probe (not for landing): the KDA gated norm's row RMS over its cohort, at both head widths."""
import jax.numpy as jnp
import numpy as np
from functools import partial
from test_kda import S, layer_and_params, oracle_params
from tools.kda_reference import oracle_chunk


def core_rms(x, p, K=4):
    import test_kda as t
    batch, length, _ = x.shape
    H, D = p["A_log"].shape[-1], p["o_norm"].shape[-1]
    mixed = jnp.concatenate([x @ p["q_proj"], x @ p["k_proj"], x @ p["v_proj"]], axis=-1)
    taps = jnp.concatenate([p[name][:, 0, :] for name in ("q_conv1d", "k_conv1d", "v_conv1d")])
    padded = jnp.pad(jnp.swapaxes(mixed, 1, 2), ((0, 0), (0, 0), (K - 1, 0)))
    conv = sum(padded[:, :, k:k + length] * taps[None, :, k, None] for k in range(K))
    conv = conv / (1 + jnp.exp(-conv))
    q, k, v = (part.reshape(batch, length, H, D) for part in jnp.split(jnp.swapaxes(conv, 1, 2), 3, axis=-1))
    gate = ((x @ p["f_a_proj"]) @ p["f_b_proj"] + p["dt_bias"]).reshape(batch, length, H, D)
    rate = jnp.exp(p["A_log"])[None, None, :, None]
    g = -5.0 / (1 + jnp.exp(-(rate * gate)))
    beta = 1 / (1 + jnp.exp(-(x @ p["b_proj"])))
    core, _ = oracle_chunk(q, k, v, g, beta, jnp.zeros((batch, H, D, D), x.dtype), chunk_size=4)
    return np.sqrt(np.mean(np.asarray(core) ** 2, axis=-1))


def test_probe():
    for wide in ({}, {"E": 16, "H": 4, "D": 8}):
        _, _, variables = layer_and_params(-5.0, **wide)
        p = oracle_params(variables)
        E = wide.get("E", 8)
        rms = np.concatenate([core_rms(np.random.RandomState(seed).randn(64, S, E).astype(np.float32), p).ravel()
                              for seed in range(16)])
        median = np.median(rms)
        q = np.quantile(rms / median, [1e-4, 1e-3, 1e-2, 0.1, 0.5])
        print("PROBE", wide or "toy", f"median {median:.3e}", " ".join(f"{v:.3e}" for v in q), f"min {rms.min()/median:.3e}",
              flush=True)
        # by position
        per = np.stack([core_rms(np.random.RandomState(s).randn(64, S, E).astype(np.float32), p) for s in range(4)])
        print("PROBE by token", wide or "toy", " ".join(f"{np.median(per[:, :, t]) / median:.2f}" for t in range(S)), flush=True)
