import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_recurrence_stability as st


def variants(q, g, do, dfinal, chunk):
    """dS0 for the write-gated scalar decay (no delta rule) four ways; q [T, K], g [T], do [T, V]."""
    T = q.shape[0]
    G = jnp.cumsum(g)
    out = {}
    # direct: one sum over every token
    out["direct"] = (q * jnp.exp(G)[:, None]).T @ do + jnp.exp(G[-1]) * dfinal
    # chunked: per chunk sum, carried back
    D = dfinal
    for start in reversed(range(0, T, chunk)):
        gc = jnp.cumsum(g[start:start + chunk])
        A = (q[start:start + chunk] * jnp.exp(gc)[:, None]).T @ do[start:start + chunk]
        D = A + jnp.exp(gc[-1]) * D
    out["chunked"] = D
    # horner: token by token
    D = dfinal
    for t in reversed(range(T)):
        D = jnp.exp(g[t]) * (D + jnp.outer(q[t], do[t]))
    out["horner"] = D
    # chunked, horner within the chunk
    D = dfinal
    for start in reversed(range(0, T, chunk)):
        for t in reversed(range(start, min(start + chunk, T))):
            D = jnp.exp(g[t]) * (D + jnp.outer(q[t], do[t]))
    out["chunk-horner"] = D
    # chunked, outer products summed in a pairwise order (a matmul of fp32 by rows, one rounding per term)
    D = dfinal
    for start in reversed(range(0, T, chunk)):
        gc = jnp.cumsum(g[start:start + chunk])
        terms = (q[start:start + chunk] * jnp.exp(gc)[:, None])[:, :, None] * do[start:start + chunk][:, None, :]
        D = jnp.sum(terms, axis=0) + jnp.exp(gc[-1]) * D
    out["chunked-sum"] = D
    return out


@pytest.mark.parametrize("regime", ["saturated gates", "decays near one", "a long sequence"])
def test_state(regime):
    hooks = st.SITES["scalar write"]
    errs = {}
    for seed in range(64):
        operands, (dout, dfinal) = st.cotangents(hooks, regime, seed)
        for b in range(operands.value.shape[0]):
            for h in range(operands.value.shape[2]):
                q = operands.query[b, :, h // (operands.value.shape[2] // operands.key.shape[2])]
                g = operands.log_decay[b, :, h]
                do, df = dout[b, :, h], dfinal[b, h]
                with jax.enable_x64(True):
                    truth = {k: np.asarray(v) for k, v in variants(*(jnp.asarray(np.asarray(x), jnp.float64) for x in (q, g, do, df)), 16).items()}
                mine = variants(q, g, do, df, 16)
                t = truth["horner"]
                for k, v in mine.items():
                    errs.setdefault(k, []).append(float(np.sum((np.asarray(v, np.float64) - t) ** 2)))
    rms = {k: float(np.sqrt(np.mean(v))) for k, v in errs.items()}
    raise AssertionError(f"STATE|{regime}|" + " ".join(f"{k}={v:.3e} ({v / rms['horner']:.2f})" for k, v in rms.items()))
