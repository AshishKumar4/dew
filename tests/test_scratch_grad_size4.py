import functools

import jax
import pytest
import test_recurrence_stability as st
from reference_error import seeds_for, with_gradients
from test_recurrence_chunk import token_by_token

SITES = [(name, regime) for name in st.SITES for regime in st.REGIMES]


@pytest.mark.parametrize("name, regime", SITES, ids=[f"{n}-{r}" for n, r in SITES])
def test_size(name, regime):
    hooks = st.SITES[name]
    draws = functools.partial(st.cotangents, hooks, regime)
    fn = with_gradients(st.restarting(functools.partial(token_by_token, hooks), regime))
    try:
        found = seeds_for(jax.jit(fn), draws, "gradient")
    except AssertionError as error:
        found = str(error)[-200:].replace("\n", " ")
    raise AssertionError(f"SIZE|{name}|{regime}|{found}|")
