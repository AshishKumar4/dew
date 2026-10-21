"""A hot path's steady state (tests/steady_state.py): the programs a block compiles, counted through
jax.monitoring's compile events, and JAX's own transfer guard refusing a block's implicit
device-to-host read while it takes an explicit one, on the CPU as on a GPU."""

import jax
import jax.numpy as jnp
import pytest
from steady_state import guarded, steady_state


def test_a_steady_state_counts_the_programs_it_compiles():
    """A block that brings the one new program it declares passes; one that compiles a program the
    steady state does not expect fails."""
    x = jnp.arange(4.0)
    with steady_state(compiles=1):
        jax.jit(lambda v: v * 3.0)(x).block_until_ready()
    with pytest.raises(AssertionError), steady_state():
        jax.jit(lambda v: v + 2.0)(x).block_until_ready()


def test_a_guarded_block_refuses_an_implicit_read_and_takes_an_explicit_one():
    """`float` of an array is an implicit read, which the guard refuses; `jax.device_get` is the read
    a caller asks for by name. Each reads an array of its own: a read that copies (a GPU's, or a CPU
    array assembled from several devices) leaves its host copy cached, and a later read of that copy
    moves nothing for the guard to see."""
    total, other = (jax.jit(jnp.sum)(jnp.arange(4.0) * scale) for scale in (1.0, 2.0))
    with guarded():
        with pytest.raises(RuntimeError):
            float(total)
        assert float(jax.device_get(other)) == 12.0


def test_a_guarded_block_refuses_every_implicit_read_of_a_cpu_array():
    """On the CPU, where an array reaches numpy as a view: a scalar's `float`, `int` and `item`, and
    `np.asarray` of a vector, each refused under the guard; `jax.device_get` taken."""
    import numpy as np

    scalars = [jax.jit(jnp.sum)(jnp.arange(4.0) * scale) for scale in (1.0, 2.0, 3.0)]
    vector = jax.jit(lambda v: v * 2.0)(jnp.arange(4.0))
    with guarded():
        for read in (lambda: float(scalars[0]), lambda: int(scalars[1]), lambda: scalars[2].item(),
                     lambda: np.asarray(vector)):
            with pytest.raises(RuntimeError):
                read()
        assert jax.device_get(vector).tolist() == [0.0, 2.0, 4.0, 6.0]
