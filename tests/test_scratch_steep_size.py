import functools

import jax
import pytest
import test_recurrence_stability as st
from reference_error import seeds_for
from test_recurrence_chunk import token_by_token

SITES = sorted({(name, 'steep then mild') for name in st.SITES})


@pytest.mark.parametrize("kind", ["forward", "gradient"])
@pytest.mark.parametrize("name, regime", SITES, ids=[f"{n}-{r}" for n, r in SITES])
def test_size(name, regime, kind):
    hooks = st.SITES[name]
    draws = functools.partial(st.stressed, hooks, regime)
    fn = functools.partial(token_by_token, hooks)
    fn = st.pulled(fn) if kind == "gradient" else fn
    try:
        found = seeds_for(jax.jit(fn), draws, kind)
    except AssertionError as error:
        found = str(error)[-150:]
    raise AssertionError(f"SIZE|{name}|{regime}|{kind}|{found}")
