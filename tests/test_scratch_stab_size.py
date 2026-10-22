import functools

import jax
import pytest
import test_recurrence_stability as st
from reference_error import seeds_for
from test_recurrence_chunk import token_by_token

SITES = sorted({(name, regime) for name in st.SITES for regime in ('spread gates', 'a fading memory')})


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


@pytest.mark.parametrize("name, regime", PORTED, ids=[f"{n}-{r}" for n, r in PORTED])
@pytest.mark.parametrize("arm", ["portable", "port"])
def test_grad(name, regime, arm):
    """The gradient of each arm, through a fixed cotangent on the output and the final memory, against the
    token-by-token recurrence's, check 1."""
    from dew.nn.kernels.recurrence_chunk import chunk
    from reference_error import assert_as_exact
    from test_recurrence_chunk import CHUNK
    hooks = st.SITES[name]

    def pulled(fn):
        def run(operands):
            outputs, pull = jax.vjp(fn, operands)
            keys = jax.random.split(jax.random.key(1 << 20), 2)
            cotangents = tuple(jax.random.normal(key, out.shape, out.dtype) for key, out in zip(keys, outputs))
            (grads,) = pull(cotangents)
            return {field: getattr(grads, field) for field in ('query', 'key', 'value', 'beta', 'log_decay', 'gate')
                    if getattr(grads, field) is not None and getattr(operands, field).dtype.kind == 'f'}
        return run

    dew = jax.jit(pulled(functools.partial(chunk, hooks, chunk_size=CHUNK, ported=arm == "port")))
    reference = jax.jit(pulled(functools.partial(token_by_token, hooks)))
    try:
        assert_as_exact(dew, reference, functools.partial(st.stressed, hooks, regime), f"{name} {regime} grad",
                        seeds=st.SEEDS[name, regime])
        verdict = "passes"
    except AssertionError as error:
        verdict = str(error).split("\n")[0][:200]
    raise AssertionError(f"GRAD|{arm}|{name}|{regime}|{verdict}")
