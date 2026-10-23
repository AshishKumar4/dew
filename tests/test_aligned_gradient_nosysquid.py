"""Minimal, deterministic proof of the WY gradient's aligned-key cancellation."""
import sys

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.kernels import recurrence_chunk as c
from dew.nn.attention import forward_mode_attention


def inputs():
    unit = jnp.ones((1, 4, 1, 1), jnp.float32)
    value = jnp.array([0.314159, -0.271828, 0.577216, 1.414214], jnp.float32).reshape(unit.shape)
    log_decay = jnp.array([-.17, -.23, -.31, -.41], jnp.float32).reshape(unit.shape)
    beta = jnp.ones((1, 4, 1), jnp.float32)
    return unit, value, log_decay, beta


def gradient(fn):
    unit, value, log_decay, beta = inputs()
    return jax.jit(jax.grad(lambda g: jnp.sum(fn(
        unit, unit, value, g, beta, chunk_size=4, scale=1.0)[0])))(log_decay)


def unregrouped(*args, **kwargs):
    with forward_mode_attention():
        return c.chunk_delta_rule(*args, **kwargs)


def test_the_aligned_write_erases_decay_before_its_gradient_is_rounded():
    # In one dimension k=beta=1, each update is S'=v, exactly independent of decay.
    # Summing the WY inverse/readout paths instead leaves roundoff in that zero.
    found = np.asarray(gradient(c.chunk_delta_rule))
    np.testing.assert_array_equal(found, np.zeros_like(found))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        implementation = {"current": unregrouped, "chunk-adjoint": c.chunk_delta_rule}[sys.argv[1]]
        found = np.asarray(gradient(implementation))
        print(sys.argv[1], found.ravel())
        np.testing.assert_array_equal(found, np.zeros_like(found))
        raise SystemExit(0)
    print("WY fp32", np.asarray(gradient(unregrouped)).ravel())
    print("chunk-parallel adjoint fp32", np.asarray(gradient(c.chunk_delta_rule)).ravel())
    with jax.enable_x64(True):
        unit, value, g, beta = (x.astype(jnp.float64) for x in inputs())
        grad64 = jax.jit(jax.grad(lambda g: jnp.sum(unregrouped(
            unit, unit, value, g, beta, chunk_size=4, scale=1.0)[0])))(g)
        print("WY fp64", np.asarray(grad64).ravel())
