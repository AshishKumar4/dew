"""Dew's dropout (`dew.nn.dropout`): a float32 run drops flax's own units, bit for bit, so no recorded
run moves, and a float64 run drops exactly the float32 run's, where flax's own draws other units."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.dropout import Dropout, kept

RNGS = {"dropout": jax.random.key(3)}


@pytest.mark.parametrize("broadcast_dims", [(), (1,)])
def test_a_float32_run_drops_flaxs_own_units(broadcast_dims):
    inputs = jax.random.normal(jax.random.key(0), (4, 16, 8))
    ours = Dropout(0.3, broadcast_dims=broadcast_dims).apply({}, inputs, deterministic=False, rngs=RNGS)
    theirs = nn.Dropout(0.3, broadcast_dims=broadcast_dims).apply({}, inputs, deterministic=False, rngs=RNGS)
    np.testing.assert_array_equal(ours, theirs)
    np.testing.assert_array_equal(kept(RNGS["dropout"], 0.7, (4, 16, 8)),
                                  jax.random.bernoulli(RNGS["dropout"], 0.7, (4, 16, 8)))


def test_a_float64_run_drops_the_float32_runs_units():
    inputs = jax.random.normal(jax.random.key(0), (4, 16, 8))
    narrow = np.asarray(Dropout(0.3).apply({}, inputs, deterministic=False, rngs=RNGS)) != 0
    with jax.enable_x64(new_val=True):
        wide = inputs.astype(jnp.float64)
        ours = np.asarray(Dropout(0.3).apply({}, wide, deterministic=False, rngs=RNGS)) != 0
        theirs = np.asarray(nn.Dropout(0.3).apply({}, wide, deterministic=False, rngs=RNGS)) != 0
        drawn = np.asarray(kept(RNGS["dropout"], 0.7, (4, 16, 8)))
    np.testing.assert_array_equal(ours, narrow)
    np.testing.assert_array_equal(drawn, narrow)
    # Flax's own draws a float64 uniform from other bits there: the units this module keeps the same.
    assert not np.array_equal(theirs, narrow)


def test_no_rate_and_the_whole_rate_need_no_draw():
    inputs = jnp.ones((2, 3))
    np.testing.assert_array_equal(Dropout(0.0).apply({}, inputs, deterministic=False), inputs)
    np.testing.assert_array_equal(Dropout(1.0).apply({}, inputs, deterministic=False), jnp.zeros((2, 3)))
    np.testing.assert_array_equal(Dropout(0.5).apply({}, inputs, deterministic=True), inputs)
