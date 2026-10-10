"""Pallas in-chunk correction, its inverse and VJP against XLA and float64.

Host tests interpret the kernels. Relative errors stay below 1e-5, and
every output and operand gradient's RMS error below twice XLA's error
against float64 (tests/reference_error.py). Aligned keys exercise the
case where powers of the strictly lower matrix catastrophically cancel.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference
from test_delta_chunks import BOUND, INTERPRETED, cotangents, relative, rule_operands

from dew.nn import kernels
from dew.nn.kernels import delta_prep
from dew.nn.linear import (
    chunk_decay,
    chunk_gated_delta_rule,
    chunk_states_kernel,
    l2norm,
    strictly_lower_inverse,
)


def prep_operands(shape, *, aligned=False):
    batch, heads, chunks, size, width, columns = shape
    keys = jax.random.split(jax.random.key(11), 5)
    lead = (batch, heads, chunks, size)
    k = l2norm(jax.random.normal(keys[1], (*lead, width)))
    hi, lo = chunk_decay(-jnp.exp(jax.random.normal(keys[2], (*lead, 1)) - 3.0), halves=True)
    hi, lo = hi[..., 0], lo[..., 0]
    v = jax.random.normal(keys[4], (*lead, columns))
    beta = jax.nn.sigmoid(jax.random.normal(jax.random.key(19), hi.shape))
    if aligned:
        k = jnp.broadcast_to(k[..., :1, :], k.shape)
        beta = jnp.full_like(beta, 0.95)
        hi = jnp.broadcast_to(-0.0001 * jnp.arange(1, hi.shape[-1] + 1), hi.shape)
        lo = jnp.zeros_like(lo)
    return k, v, beta, hi, lo


def xla_prep(k, v, beta, hi, lo):
    index = jnp.arange(hi.shape[-1])
    lower = index[:, None] >= index[None, :]
    diff = (hi[..., :, None] - hi[..., None, :]) + (lo[..., :, None] - lo[..., None, :])
    decay = jnp.where(lower, jnp.exp(jnp.where(lower, diff, 0.0)), 0.0)
    kb = k * beta[..., None]
    a = jnp.where(index[:, None] > index[None, :], -(kb @ jnp.swapaxes(k, -1, -2)) * decay, 0.0)
    inverse = strictly_lower_inverse(a)
    return inverse @ (kb * jnp.exp(hi)[..., None]), inverse @ (v * beta[..., None]), inverse


def kernel_prep(*operands):
    return delta_prep.chunk_prep(*operands, INTERPRETED)


def xla_products(*operands):
    w, u, _ = xla_prep(*operands)
    return w, u


@jax.default_matmul_precision("highest")
@pytest.mark.parametrize("shape", [(2, 2, 3, 16, 16, 32), (1, 2, 2, 32, 32, 64),
                                    (1, 1, 2, 64, 128, 128)])
def test_prep_and_all_gradients_compute_xla_and_are_as_exact(shape):
    operands = prep_operands(shape)
    reference = xla_products(*operands)
    seed = cotangents(reference)
    expected = jax.vjp(xla_products, *operands)[1](seed)
    actual = kernel_prep(*operands)
    gradients = jax.vjp(kernel_prep, *operands)[1](seed)
    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(new_val=True), jax.default_device(cpu):
        exact = [jax.device_put(np.asarray(t, np.float64), cpu) for t in operands]
        truth = xla_products(*exact)
        truth_gradients = jax.vjp(xla_products, *exact)[1](
            tuple(jnp.asarray(np.asarray(t), jnp.float64) for t in seed))
    for name, mine, theirs, want in zip(("w", "u", "k", "v", "beta", "hi", "lo"),
                                        (*actual, *gradients), (*reference, *expected),
                                        (*truth, *truth_gradients), strict=True):
        assert relative(mine, theirs) < BOUND, name
        assert_as_exact_as_the_reference(mine, theirs, want, name)


@jax.default_matmul_precision("highest")
@pytest.mark.parametrize("aligned", [False, True])
def test_saved_inverse_is_bounded_even_when_the_keys_align(aligned):
    operands = prep_operands((1, 1, 1, 64, 128, 128), aligned=aligned)
    expected = xla_prep(*operands)
    actual = delta_prep._prep_with_inverse(*operands, INTERPRETED)
    for name, mine, theirs in zip(("w", "u", "inverse"), actual, expected, strict=True):
        assert np.all(np.isfinite(np.asarray(mine))), name
        assert relative(mine, theirs) < BOUND, name
    if aligned:
        assert np.max(np.abs(np.asarray(actual[2]))) <= 1.0


@pytest.mark.parametrize("shape,reason", [
    ((128, 128, 128), "prep chunks of 128"),
    ((64, 256, 128), "prep keys 256"),
    ((64, 128, 512), "prep values 512"),
])
def test_prep_refusals_name_the_reason_once(monkeypatch, caplog, shape, reason):
    from dew.nn import linear

    monkeypatch.setitem(kernels.KERNELS, "gated_delta_rule", {"sm80": "pallas"})
    monkeypatch.setattr(kernels.generation, "device_generation", lambda: "sm80")
    monkeypatch.setattr(kernels.generation, "_logged", set())
    monkeypatch.setattr(linear, "triton_runs", lambda: True)
    size, width, columns = shape
    k = jax.ShapeDtypeStruct((1, 1, 1, size, width), jnp.float32)
    v = jax.ShapeDtypeStruct((1, 1, 1, size, columns), jnp.float32)
    gc = jax.ShapeDtypeStruct((1, 1, 1, size, 1), jnp.float32)
    assert [chunk_states_kernel("auto", k, v, gc) for _ in range(2)] == ["xla"] * 2
    logged = [r.getMessage() for r in caplog.records if r.name == kernels.generation.__name__]
    assert len(logged) == 1 and reason in logged[0]


@jax.default_matmul_precision("highest")
def test_the_rule_with_fused_prep_is_as_exact_as_xla():
    """The public rule, including a ragged last chunk, the initial state and
    autodiff of the compensated cumsum, at the same precision as XLA."""
    operands = rule_operands()

    def rule(implementation):
        return lambda *args: chunk_gated_delta_rule(*args, implementation=implementation)

    reference = rule("xla")(*operands)
    seed = cotangents(reference)
    expected = jax.vjp(rule("xla"), *operands)[1](seed)
    actual = rule("pallas")(*operands)
    gradients = jax.vjp(rule("pallas"), *operands)[1](seed)
    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(new_val=True), jax.default_device(cpu):
        exact = [jax.device_put(np.asarray(t, np.float64), cpu) for t in operands]
        truth = rule("xla")(*exact)
        truth_gradients = jax.vjp(rule("xla"), *exact)[1](
            tuple(jnp.asarray(np.asarray(t), jnp.float64) for t in seed))
    for name, mine, theirs, want in zip(("output", "state", "q", "k", "v", "g", "beta", "initial"),
                                        (*actual, *gradients), (*reference, *expected),
                                        (*truth, *truth_gradients), strict=True):
        assert relative(mine, theirs) < BOUND, name
        assert_as_exact_as_the_reference(mine, theirs, want, name)
