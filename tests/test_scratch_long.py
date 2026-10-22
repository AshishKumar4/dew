import functools

import jax
import jax.numpy as jnp
import pytest
import test_recurrence_stability as st
from reference_error import assert_as_exact
from test_recurrence_chunk import token_by_token

from dew.nn.kernels import fla_gdn
from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.recurrence_chunk import chunk


def residual_apply(inverse, keyed, valued, entering, issue=algebra.portable):
    return jnp.concatenate([inverse, keyed * entering], -1), valued


def residual_carry(state, step, issue=algebra.portable):
    k_i, w_i, u_i, gc_i, last_i, kept_i = step
    size = u_i.shape[-2]
    inverse, keyed = w_i[..., :size], w_i[..., size:]
    v_corrected = issue("apply_u", inverse, u_i - issue("corrected", keyed, state))
    return (state * jnp.swapaxes(jnp.exp(gc_i[..., -1:, :]), -1, -2) * kept_i
            + issue("write", jnp.swapaxes(k_i * last_i, -1, -2), v_corrected), (state, v_corrected))


VARIANTS = ["portable-16", "portable-32", "portable-64", "residual-64", "residual-32", "port"]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", ["scalar delta", "vector delta", "gated"])
def test_long(name, variant, monkeypatch):
    hooks = st.SITES[name]
    if variant.startswith("residual"):
        monkeypatch.setattr(algebra, "apply", residual_apply)
        monkeypatch.setattr(algebra, "carry", residual_carry)
    size = int(variant.split("-")[1]) if "-" in variant else 64
    dew = jax.jit(functools.partial(chunk, hooks, chunk_size=size, ported=variant == "port"))
    reference = jax.jit(functools.partial(token_by_token, hooks))
    try:
        assert_as_exact(dew, reference, functools.partial(st.stressed, hooks, "a long sequence"), name,
                        seeds=st.SEEDS[name, "a long sequence"])
        verdict = "passes"
    except AssertionError as error:
        verdict = " / ".join(l for l in str(error).split("\n") if "times" in l)[:400]
    raise AssertionError(f"LONG|{name}|{variant}|{verdict}")
