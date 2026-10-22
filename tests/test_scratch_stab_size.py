import functools

import jax
import pytest
import test_recurrence_stability as st
from reference_error import seeds_for
from test_recurrence_chunk import token_by_token

SITES = sorted({(name, regime) for name in st.SITES for regime in st.REGIMES})


@pytest.mark.parametrize("name, regime", SITES, ids=[f"{n}-{r}" for n, r in SITES])
def test_size(name, regime):
    hooks = st.SITES[name]
    reference = jax.jit(functools.partial(token_by_token, hooks))
    try:
        count = seeds_for(reference, functools.partial(st.stressed, hooks, regime), f"{name} {regime}")
    except AssertionError as error:
        count = str(error)[-200:]
    raise AssertionError(f"SIZE|{name}|{regime}|{count}")
