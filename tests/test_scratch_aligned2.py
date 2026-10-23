import functools

import jax
import jax.numpy as jnp
import pytest
import test_recurrence_stability as st
from reference_error import assert_as_exact, with_gradients
from test_recurrence_chunk import token_by_token

from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.recurrence_chunk import chunk

ORIGINAL = {name: getattr(algebra, name) for name in ("factor", "apply", "intra", "carry", "readout", "decays")}


def widened(fn):
    """`fn` computed in float64, its outputs back in their dtypes."""
    def run(*args, **kwargs):
        dtypes = jax.tree.map(lambda x: x.dtype if hasattr(x, 'dtype') else None, args)
        with jax.enable_x64(True):
            wide = jax.tree.map(lambda x: x.astype(jnp.float64) if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating) else x, args)
            out = fn(*wide, **{k: v for k, v in kwargs.items() if k != 'issue'})
        dtype = jnp.float32
        return jax.tree.map(lambda x: x.astype(dtype) if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating) else x, out)
    return run


@pytest.mark.parametrize("piece", ["none", "factor", "apply", "intra", "carry", "readout"])
@pytest.mark.parametrize("name", ["scalar delta", "vector delta"])
def test_piece(name, piece, monkeypatch):
    hooks = st.SITES[name]
    regime = "aligned keys"
    if piece != "none":
        monkeypatch.setattr(algebra, piece, widened(ORIGINAL[piece]))
    with jax.enable_x64(True):
        dew = jax.jit(with_gradients(functools.partial(chunk, hooks, chunk_size=st.CHUNK, ported=False)))
    reference = jax.jit(with_gradients(functools.partial(token_by_token, hooks)))
    try:
        assert_as_exact(dew, reference, functools.partial(st.cotangents, hooks, regime), name,
                        seeds=st.GRADIENT_SEEDS[name, regime])
        verdict = "passes"
    except AssertionError as error:
        verdict = " / ".join(l.split(":")[0][-28:] + l[l.find(" is "):l.find(" times")]
                             for l in str(error).split("\n") if "times" in l)[:400]
    raise AssertionError(f"PIECE|{name}|{piece}|{verdict}")
