import functools

import jax
import pytest
import test_recurrence_stability as st
from reference_error import seeds_for
from test_recurrence_chunk import token_by_token

SITES = sorted({(name, 'decays near zero') for name in st.SITES})


@pytest.mark.parametrize("name, regime", SITES, ids=[f"{n}-{r}" for n, r in SITES])
def test_size(name, regime):
    hooks = st.SITES[name]
    draws = functools.partial(st.stressed, hooks, regime)
    found = []
    try:
        found.append(seeds_for(jax.jit(st.pulled(functools.partial(token_by_token, hooks))), draws, "gradient"))
    except AssertionError as error:
        found.append(str(error)[-150:])
    if True:
        try:
            found.append(seeds_for(jax.jit(functools.partial(token_by_token, hooks)), draws, "forward"))
        except AssertionError as error:
            found.append(str(error)[-150:])
    raise AssertionError(f"SIZE|{name}|{regime}|" + "|".join(str(f) for f in found))
