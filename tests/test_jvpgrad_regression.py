"""The decay adjoint regression against the replay before its shared decay read."""

import jax.numpy as jnp
import numpy as np
from test_algebra_scan import GATED, KEY, VALUE, gated_delta, gated_readout

from dew.algebra import SATURATION, scan


def test_a_chunks_adjoint_meets_its_decay_once():
    chunk = {"q": jnp.zeros((1, KEY)), "k": jnp.zeros((1, KEY)).at[0, 0].set(1.0),
             "v": jnp.zeros((1, VALUE)), "g": jnp.full((1,), -1.0), "beta": jnp.ones((1,))}
    forms = scan.derive_scan(scan.Scan(gated_delta, gated_readout, GATED, walked="time",
                                       state=((KEY, VALUE), jnp.float32)), saturation=SATURATION)
    entering = jnp.zeros((KEY, VALUE)).at[0].set(2.0 ** 120)
    d_state, d_operands = forms.backward(chunk, (None, entering), jnp.full((KEY, VALUE), 2.0 ** 20),
                                         jnp.zeros((1, VALUE)))
    assert np.all(np.isfinite(np.asarray(d_state)))
    assert np.all(np.isfinite(np.asarray(d_operands["g"])))
    np.testing.assert_array_equal(d_operands["g"], jnp.zeros_like(chunk["g"]))
