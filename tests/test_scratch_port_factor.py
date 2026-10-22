import functools

import jax
import jax.numpy as jnp
import pytest
import test_recurrence_stability as st
from jax.experimental import pallas as pl
from reference_error import assert_as_exact
from test_recurrence_chunk import CHUNK, token_by_token

from dew.nn.kernels import fla_gdn
from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.chunked_launch import _gates, _rows, read_rows, write_rows
from dew.nn.kernels.recurrence_chunk import chunk


def doubling_solve_chunk(k_ref, g_ref, beta_ref, a_ref, start, taken, *, chunk: int, sub: int, bk: int,
                         exact: bool):
    """The port's kkt-solve with its inverse taken by the algebra's doubling (`recurrence_algebra.factor`)
    in place of fla's forward substitution and merges; interpreted only."""
    width = k_ref.shape[-1]
    count = chunk // sub

    def part(ref, i, *rest):
        return read_rows(_rows(ref, start + i * sub, sub, *rest), taken - i * sub)

    gates = [_gates(g_ref, start, taken, sub, i * sub) for i in range(count)]
    betas = [part(beta_ref, i).astype(jnp.float32) for i in range(count)]
    index = jnp.arange(sub)
    strict = index[:, None] > index[None, :]

    def keys(i, c):
        return part(k_ref, i, pl.ds(c * bk, bk))

    def block(i, j):
        product = sum(fla_gdn._dot(keys(i, c), keys(j, c).T, exact=exact) for c in range(width // bk))
        decay = jnp.exp2(fla_gdn._pairs(gates[i], gates[j]))
        product = product * (jnp.where(strict, decay, 0.0) if i == j else decay)
        return product * betas[i][:, None]

    rows = [jnp.concatenate([block(i, j) if j <= i else jnp.zeros((sub, sub), jnp.float32)
                             for j in range(count)], axis=1) for i in range(count)]
    inverse = algebra.factor(-jnp.concatenate(rows, axis=0))
    for i in range(count):
        for j in range(count):
            write_rows(_rows(a_ref, start + i * sub, sub, pl.ds(j * sub, sub)), taken - i * sub,
                       inverse[i * sub:(i + 1) * sub, j * sub:(j + 1) * sub])


@pytest.mark.parametrize("variant", ["fla", "doubling"])
@pytest.mark.parametrize("name, regime", [("vector delta", "aligned keys"), ("gated", "aligned keys"),
                                          ("scalar delta", "saturated gates"), ("gated", "saturated gates"),
                                          ("scalar delta", "resets on chunk edges"),
                                          ("scalar delta", "the decay bound")])
def test_port_variant(name, regime, variant, monkeypatch):
    if variant == "doubling":
        monkeypatch.setattr(fla_gdn, "_kkt_solve_chunk", doubling_solve_chunk)
    hooks = st.SITES[name]
    dew = jax.jit(functools.partial(chunk, hooks, chunk_size=CHUNK, ported=True))
    reference = jax.jit(functools.partial(token_by_token, hooks))
    try:
        assert_as_exact(dew, reference, functools.partial(st.stressed, hooks, regime), f"{name} {regime}",
                        seeds=8)
        verdict = "passes"
    except AssertionError as error:
        verdict = str(error).split("\n")[0][:220]
    raise AssertionError(f"VARIANT|{name}|{regime}|{variant}|{verdict}")


@pytest.mark.parametrize("size", [16, 64])
@pytest.mark.parametrize("name, regime", [("vector delta", "aligned keys"), ("gated", "aligned keys"),
                                          ("scalar delta", "saturated gates"), ("gated", "saturated gates"),
                                          ("scalar delta", "resets on chunk edges"),
                                          ("scalar delta", "the decay bound")])
def test_portable_chunk(name, regime, size):
    hooks = st.SITES[name]
    dew = jax.jit(functools.partial(chunk, hooks, chunk_size=size, ported=False))
    reference = jax.jit(functools.partial(token_by_token, hooks))
    try:
        assert_as_exact(dew, reference, functools.partial(st.stressed, hooks, regime), f"{name} {regime}",
                        seeds=8)
        verdict = "passes"
    except AssertionError as error:
        verdict = str(error).split("\n")[0][:220]
    raise AssertionError(f"PORTABLE|{name}|{regime}|{size}|{verdict}")
