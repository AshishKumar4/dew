import functools

import jax
import pytest
import test_recurrence_stability as st
from reference_error import seeds_for
from test_recurrence_chunk import token_by_token

SITES = sorted({(name, regime) for name in ('gated', 'rated') for regime in st.REGIMES})


@pytest.mark.parametrize("name, regime", SITES, ids=[f"{n}-{r}" for n, r in SITES])
def test_size(name, regime):
    hooks = st.SITES[name]
    reference = jax.jit(functools.partial(token_by_token, hooks))
    try:
        count = seeds_for(reference, functools.partial(st.stressed, hooks, regime), f"{name} {regime}")
    except AssertionError as error:
        count = str(error)[-200:]
    raise AssertionError(f"SIZE|{name}|{regime}|{count}")


PORTED = sorted({(name, regime) for name in ('scalar delta', 'vector delta', 'gated') for regime in st.REGIMES
                 if st.REGIMES[regime][1] is None})


@pytest.mark.parametrize("name, regime", PORTED, ids=[f"{n}-{r}" for n, r in PORTED])
def test_port(name, regime):
    """The interpreted port arm in every regime, at the site's declared seeds."""
    from dew.nn.kernels.recurrence_chunk import chunk
    from reference_error import assert_as_exact
    from test_recurrence_chunk import CHUNK
    hooks = st.SITES[name]
    dew = jax.jit(functools.partial(chunk, hooks, chunk_size=CHUNK, ported=True))
    reference = jax.jit(functools.partial(token_by_token, hooks))
    try:
        assert_as_exact(dew, reference, functools.partial(st.stressed, hooks, regime), f"{name} {regime}",
                        seeds=st.SEEDS[name, regime])
        verdict = "passes"
    except AssertionError as error:
        verdict = str(error).split("\n")[0][:200]
    raise AssertionError(f"PORT|{name}|{regime}|{verdict}")
