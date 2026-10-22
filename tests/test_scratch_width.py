import functools

import jax
import pytest
import test_recurrence_chunk as rc
import test_recurrence_stability as st
from reference_error import assert_as_exact
from test_recurrence_chunk import token_by_token

from dew.nn.kernels.recurrence_chunk import chunk


@pytest.mark.parametrize("size", [32, 64])
@pytest.mark.parametrize("width", [16, 32, 64, 128])
@pytest.mark.parametrize("name", ["scalar delta", "vector delta", "gated"])
def test_width(name, width, size, monkeypatch):
    monkeypatch.setattr(rc, "KEY_WIDTH", width)
    hooks = st.SITES[name]
    dew = jax.jit(functools.partial(chunk, hooks, chunk_size=size, ported=False))
    reference = jax.jit(functools.partial(token_by_token, hooks))
    try:
        assert_as_exact(dew, reference, functools.partial(st.stressed, hooks, "a long sequence"), name,
                        seeds=st.SEEDS[name, "a long sequence"])
        verdict = "passes"
    except AssertionError as error:
        verdict = " / ".join(l for l in str(error).split("\n") if "times" in l)[:400]
    raise AssertionError(f"WIDTH|{name}|{width}|{size}|{verdict}")
