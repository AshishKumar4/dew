"""Chunk-local Pallas output and its six cotangents against XLA and float64.

The host uses the Pallas interpreter; a GPU runs the same cases through
Triton. Relative error is bounded by 1e-5 and every output and gradient's
RMS distance from float64 by twice XLA's (tests/reference_error.py).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference
from test_delta_chunks import BOUND, INTERPRETED, cotangents, relative, rule_operands

from dew.nn import kernels
from dew.nn.kernels import delta_output
from dew.nn.linear import chunk_decay, chunk_gated_delta_rule, chunk_output_kernel, l2norm


def output_operands(shape):
    batch, heads, chunks, size, width, columns = shape
    keys = jax.random.split(jax.random.key(11), 5)
    lead = (batch, heads, chunks, size)
    q, k = (l2norm(jax.random.normal(drawn, (*lead, width))) for drawn in keys[:2])
    g = -jnp.exp(jax.random.normal(keys[2], (*lead, 1)) - 3.0)
    hi, lo = chunk_decay(g, halves=True)
    h = 0.1 * jax.random.normal(keys[3], (batch, heads, chunks, width, columns))
    v = jax.random.normal(keys[4], (*lead, columns))
    return q * width ** -0.5, k, hi[..., 0], lo[..., 0], h, v


def xla_output(q, k, hi, lo, h, v):
    index = jnp.arange(hi.shape[-1])
    lower = index[:, None] >= index[None, :]
    diff = (hi[..., :, None] - hi[..., None, :]) + (lo[..., :, None] - lo[..., None, :])
    decay = jnp.where(lower, jnp.exp(jnp.where(lower, diff, 0.0)), 0.0)
    return (q * jnp.exp(hi)[..., None]) @ h + ((q @ jnp.swapaxes(k, -1, -2)) * decay) @ v


def kernel_output(*operands):
    return delta_output.chunk_output(*operands, INTERPRETED)


@jax.default_matmul_precision("highest")
@pytest.mark.parametrize("shape", [(2, 2, 3, 16, 16, 32), (1, 2, 2, 32, 32, 64),
                                    (1, 1, 2, 64, 128, 128)])
def test_output_and_all_gradients_compute_xla_and_are_as_exact(shape):
    operands = output_operands(shape)
    reference = xla_output(*operands)
    seed, = cotangents((reference,))
    expected = jax.vjp(xla_output, *operands)[1](seed)
    output = kernel_output(*operands)
    gradients = jax.vjp(kernel_output, *operands)[1](seed)
    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(True), jax.default_device(cpu):
        exact = [jax.device_put(np.asarray(t, np.float64), cpu) for t in operands]
        truth = xla_output(*exact)
        truth_gradients = jax.vjp(xla_output, *exact)[1](jnp.asarray(np.asarray(seed), jnp.float64))
    for name, mine, theirs, want in zip(("output", "q", "k", "hi", "lo", "h", "v"),
                                        (output, *gradients), (reference, *expected),
                                        (truth, *truth_gradients), strict=True):
        assert relative(mine, theirs) < BOUND, name
        assert_as_exact_as_the_reference(mine, theirs, want, name)


def test_unused_positive_differences_do_not_overflow_the_backward():
    q, k, hi, lo, h, v = output_operands((1, 1, 1, 64, 32, 64))
    hi = jnp.broadcast_to(-32.0 * jnp.arange(64), hi.shape)
    operands = q, k, hi, lo, h, v
    reference = xla_output(*operands)
    seed, = cotangents((reference,))
    actual = (kernel_output(*operands), *jax.vjp(kernel_output, *operands)[1](seed))
    expected = (reference, *jax.vjp(xla_output, *operands)[1](seed))
    for mine, theirs in zip(actual, expected, strict=True):
        assert np.all(np.isfinite(np.asarray(mine)))
        assert relative(mine, theirs) < BOUND


@jax.default_matmul_precision("highest")
def test_the_rule_with_fused_output_is_as_exact_as_xla():
    operands = rule_operands()

    def rule(implementation):
        return lambda *args: chunk_gated_delta_rule(*args, implementation=implementation)

    reference = rule("xla")(*operands)
    seed = cotangents(reference)
    expected = jax.vjp(rule("xla"), *operands)[1](seed)
    actual = rule("pallas")(*operands)
    gradients = jax.vjp(rule("pallas"), *operands)[1](seed)
    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(True), jax.default_device(cpu):
        exact = [jax.device_put(np.asarray(t, np.float64), cpu) for t in operands]
        truth = rule("xla")(*exact)
        truth_gradients = jax.vjp(rule("xla"), *exact)[1](
            tuple(jnp.asarray(np.asarray(t), jnp.float64) for t in seed))
    for name, mine, theirs, want in zip(("output", "state", "q", "k", "v", "g", "beta", "initial"),
                                        (*actual, *gradients), (*reference, *expected),
                                        (*truth, *truth_gradients), strict=True):
        assert relative(mine, theirs) < BOUND, name
        assert_as_exact_as_the_reference(mine, theirs, want, name)


@pytest.mark.parametrize("generation,chosen", [("sm80", "pallas"), ("sm89", "xla"), ("cpu", "xla")])
def test_output_auto_uses_its_own_measurement(monkeypatch, generation, chosen):
    monkeypatch.setitem(kernels.KERNELS, "gated_delta_output", {"sm80": "pallas"})
    monkeypatch.setattr(kernels.generation, "device_generation", lambda: generation)
    _, k, _, _, _, v = output_operands((1, 1, 1, 16, 32, 64))
    assert chunk_output_kernel("auto", "pallas", k, v) == chosen


@pytest.mark.parametrize("recurrence,shape,reason", [
    ("xla", (64, 128, 128), "recurrence does not run on Pallas"),
    ("pallas", (128, 128, 128), "output chunks of 128"),
    ("pallas", (64, 256, 128), "output keys 256"),
    ("pallas", (64, 128, 96), "output values 96"),
])
def test_output_refusals_name_the_reason_once(monkeypatch, caplog, recurrence, shape, reason):
    monkeypatch.setitem(kernels.KERNELS, "gated_delta_output", {"sm80": "pallas"})
    monkeypatch.setattr(kernels.generation, "device_generation", lambda: "sm80")
    monkeypatch.setattr(kernels.generation, "_logged", set())
    size, width, columns = shape
    k = jax.ShapeDtypeStruct((1, 1, 1, size, width), jnp.float32)
    v = jax.ShapeDtypeStruct((1, 1, 1, size, columns), jnp.float32)
    assert [chunk_output_kernel("auto", recurrence, k, v) for _ in range(2)] == ["xla"] * 2
    logged = [r.getMessage() for r in caplog.records if r.name == kernels.generation.__name__]
    assert len(logged) == 1 and reason in logged[0]
