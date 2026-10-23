import functools

import jax
import jax.numpy as jnp
import pytest
import test_recurrence_stability as st
from reference_error import assert_as_exact, with_gradients
from test_recurrence_chunk import token_by_token

from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.recurrence_chunk import chunk


def residual_apply(inverse, keyed, valued, entering, issue=algebra.portable):
    return jnp.concatenate([inverse, keyed * entering], -1), valued


def residual_carry(state, step, issue=algebra.portable):
    k_i, w_i, u_i, grown_i, last_i, kept_i = step
    size = u_i.shape[-2]
    inverse, keyed = w_i[..., :size], w_i[..., size:]
    v_corrected = issue("apply_u", inverse, u_i - issue("corrected", keyed, state))
    return (state * jnp.swapaxes(grown_i[..., -1:, :], -1, -2) * kept_i
            + issue("write", jnp.swapaxes(k_i * last_i, -1, -2), v_corrected), (state, v_corrected))


@pytest.mark.parametrize("variant", ["portable", "residual"])
@pytest.mark.parametrize("name", ["scalar delta", "vector delta", "gated"])
def test_aligned(name, variant, monkeypatch):
    hooks = st.SITES[name]
    regime = "aligned keys"
    if variant == "residual":
        monkeypatch.setattr(algebra, "apply", residual_apply)
        monkeypatch.setattr(algebra, "carry", residual_carry)
    dew = jax.jit(with_gradients(functools.partial(chunk, hooks, chunk_size=st.CHUNK, ported=False)))
    reference = jax.jit(with_gradients(functools.partial(token_by_token, hooks)))
    try:
        assert_as_exact(dew, reference, functools.partial(st.cotangents, hooks, regime), name,
                        seeds=st.GRADIENT_SEEDS[name, regime])
        verdict = "passes"
    except AssertionError as error:
        verdict = " / ".join(l.split(":")[0][-30:] + l[l.find(" is "):l.find(" times")]
                             for l in str(error).split("\n") if "times" in l)[:400]
    raise AssertionError(f"ALIGNED|{name}|{variant}|{verdict}")
