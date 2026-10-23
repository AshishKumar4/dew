import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_recurrence_stability as st
from reference_error import widened, with_gradients
from test_recurrence_chunk import token_by_token

from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.recurrence_chunk import chunk

ORIGINAL = {name: getattr(algebra, name) for name in ("readout", "carry", "decays", "portable")}


def wide_readout(query, entering, entered, key, pairs, corrected, issue=algebra.portable):
    memory = ((query * entering).astype(jnp.float64) @ entered.astype(jnp.float64)).astype(query.dtype)
    return memory + issue("local", algebra.paired("scores", query, key, pairs, issue), corrected)


def wide_carry(state, step, issue=algebra.portable):
    k_i, w_i, u_i, grown_i, last_i, kept_i = step
    v_corrected = u_i - issue("corrected", w_i, state)
    leaving = (state.astype(jnp.float64) * jnp.swapaxes(grown_i[..., -1:, :], -1, -2).astype(jnp.float64) * kept_i
               + issue("write", jnp.swapaxes(k_i * last_i, -1, -2), v_corrected).astype(jnp.float64))
    return leaving.astype(state.dtype), (state, v_corrected)


def wide_issue(name, left, right):
    return (left.astype(jnp.float64) @ right.astype(jnp.float64)).astype(left.dtype)


@jax.custom_vjp
def memory_product(decayed, entered):
    return decayed @ entered


def _memory_fwd(decayed, entered):
    return decayed @ entered, (decayed, entered)


def _memory_bwd(residual, out_grad):
    decayed, entered = residual
    # d entered: sum over the chunk's tokens of decayed[t]^T out_grad[t], smallest (last) first
    def step(total, t):
        row, grad = t
        return total + row[..., :, None] * grad[..., None, :], None
    rows = jnp.moveaxis(decayed, -2, 0)[::-1]
    grads = jnp.moveaxis(out_grad, -2, 0)[::-1]
    total, _ = jax.lax.scan(step, jnp.zeros(entered.shape, entered.dtype), (rows, grads))
    return out_grad @ jnp.swapaxes(entered, -1, -2), total


memory_product.defvjp(_memory_fwd, _memory_bwd)


def reversed_readout(query, entering, entered, key, pairs, corrected, issue=algebra.portable):
    return (memory_product(query * entering, entered)
            + issue("local", algebra.paired("scores", query, key, pairs, issue), corrected))


VARIANTS = {"reversed readout": {"readout": reversed_readout},"portable": {}, "wide readout": {"readout": wide_readout}, "wide carry": {"carry": wide_carry},
            "wide readout+carry": {"readout": wide_readout, "carry": wide_carry},
            "wide products": {"portable": wide_issue}}


@pytest.mark.parametrize("regime", ["saturated gates", "decays near one", "aligned keys"])
@pytest.mark.parametrize("name", ["scalar write", "scalar delta", "vector delta", "gated"])
def test_decomp(name, regime, monkeypatch):
    hooks = st.SITES[name]
    reference = jax.jit(with_gradients(functools.partial(token_by_token, hooks)))
    errors = {"reference": []}
    draws = [st.cotangents(hooks, regime, seed) for seed in range(64)]
    truths = []
    for drawn in draws:
        with jax.enable_x64(True):
            truths.append(np.asarray(reference(*widened(drawn))[1].state))
        errors["reference"].append(float(np.sum((np.asarray(reference(*drawn)[1].state, np.float64) - truths[-1]) ** 2)))
    for label, patches in VARIANTS.items():
        for attr, fn in ORIGINAL.items():
            monkeypatch.setattr(algebra, attr, fn)
        for attr, fn in patches.items():
            monkeypatch.setattr(algebra, attr, fn)
        with jax.enable_x64(True):
            arm = jax.jit(with_gradients(functools.partial(chunk, hooks, chunk_size=16, ported=False)))
            errors[label] = [float(np.sum((np.asarray(arm(*drawn)[1].state, np.float64) - truth) ** 2))
                             for drawn, truth in zip(draws, truths)]
    rms = {k: float(np.sqrt(np.mean(v))) for k, v in errors.items()}
    raise AssertionError(f"DECOMP|{name}|{regime}|" + " ".join(f"{k}={v / rms['reference']:.2f}" for k, v in rms.items()))
