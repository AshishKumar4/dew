"""A hot path's steady state (tests/steady_state.py): the programs a block compiles, counted through
jax.monitoring's compile events, and on a GPU JAX's own transfer guard refusing a block's implicit
device-to-host read while it takes an explicit one."""

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


@pytest.mark.gpu(reason="JAX's guard refuses a device-to-host read only where it copies, on a GPU")
def test_a_guarded_block_refuses_an_implicit_read_and_takes_an_explicit_one():
    """`float` of an array is an implicit read, which the guard refuses; `jax.device_get` is the read
    a caller asks for by name. Each reads an array of its own: a read leaves the array's host copy
    cached, and a later read of that copy moves nothing for the guard to see."""
    total, other = (jax.jit(jnp.sum)(jnp.arange(4.0) * scale) for scale in (1.0, 2.0))
    with guarded():
        with pytest.raises(RuntimeError):
            float(total)
        assert float(jax.device_get(other)) == 12.0
