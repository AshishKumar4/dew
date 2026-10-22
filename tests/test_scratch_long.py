import functools

import jax
import pytest
import test_recurrence_stability as st
from reference_error import assert_as_exact
from test_recurrence_chunk import token_by_token

from dew.nn.kernels import fla_gdn
from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.recurrence_chunk import chunk


@pytest.mark.parametrize("variant", ["portable-16", "portable-64", "port", "port-doubling"])
@pytest.mark.parametrize("name", ["scalar delta", "gated"])
def test_long(name, variant, monkeypatch):
    hooks = st.SITES[name]
    if variant == "port-doubling":
        import test_scratch_port_factor_lib as lib
        monkeypatch.setattr(fla_gdn, "_kkt_solve_chunk", lib.doubling_solve_chunk)
    size = 64 if variant == "portable-64" else 16
    dew = jax.jit(functools.partial(chunk, hooks, chunk_size=size, ported=variant.startswith("port-") or variant == "port"))
    reference = jax.jit(functools.partial(token_by_token, hooks))
    try:
        assert_as_exact(dew, reference, functools.partial(st.stressed, hooks, "a long sequence"), name,
                        seeds=st.SEEDS[name, "a long sequence"])
        verdict = "passes"
    except AssertionError as error:
        verdict = " / ".join(l for l in str(error).split("\n") if "times" in l)[:300]
    raise AssertionError(f"LONG|{name}|{variant}|{verdict}")
