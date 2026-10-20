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
    """`jax.device_get` is the read a caller asks for by name; `float` of an array is an implicit one,
    which the guard refuses."""
    total = jax.jit(jnp.sum)(jnp.arange(4.0))
    with guarded():
        assert float(jax.device_get(total)) == 6.0
        with pytest.raises(RuntimeError):
            float(total)
