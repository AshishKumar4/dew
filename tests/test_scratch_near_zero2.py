import functools

import jax
import jax.numpy as jnp
import pytest
import test_recurrence_stability as st
from reference_error import assert_as_exact, with_gradients, without_ynn
from test_recurrence_chunk import token_by_token

from dew.nn.kernels import fla_gdn
from dew.nn.kernels.recurrence_chunk import chunk

ORIGINAL = fla_gdn._summed_back


def wide_summed_back(index, pairs, after):
    with jax.enable_x64(True):
        p, a = pairs.astype(jnp.float64), after.astype(jnp.float64)
        through = jnp.cumsum(p, axis=1) - p
        return jnp.sum(jnp.where(index[:, None] >= index[None, :], through + a[:, None], 0.0), axis=0).astype(jnp.float32)


def exclusive_summed_back(index, pairs, after):
    strict = index[:, None] < index[None, :]
    through = jnp.sum(jnp.where(strict[None, :, :], pairs[:, :, None], 0.0), axis=1)
    return jnp.sum(jnp.where(index[:, None] >= index[None, :], through + after[:, None], 0.0), axis=0)


@pytest.mark.parametrize("variant", ["port", "wide", "exclusive"])
@pytest.mark.parametrize("regime", ["decays near zero", "the decay bound"])
def test_variant(regime, variant, monkeypatch):
    hooks = st.SITES["scalar delta"]
    if variant == "wide":
        monkeypatch.setattr(fla_gdn, "_summed_back", wide_summed_back)
    elif variant == "exclusive":
        monkeypatch.setattr(fla_gdn, "_summed_back", exclusive_summed_back)
    dew = without_ynn(with_gradients(functools.partial(chunk, hooks, chunk_size=st.CHUNK, ported=True)))
    reference = without_ynn(with_gradients(functools.partial(token_by_token, hooks)))
    try:
        assert_as_exact(dew, reference, functools.partial(st.cotangents, hooks, regime), "x",
                        seeds=st.GRADIENT_SEEDS["scalar delta", regime])
        verdict = "passes"
    except AssertionError as error:
        verdict = " / ".join(l.split(":")[0][-20:] + l[l.find(" is "):l.find(" times")]
                             for l in str(error).split("\n") if "times" in l)[:400]
    raise AssertionError(f"NZ|{regime}|{variant}|{verdict}")
