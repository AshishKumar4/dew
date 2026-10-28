"""The frozen reference: the EMA tree at unit decay.

A preference or RL objective reads its reference from `step.ema` with the
decay schedule fixed at 1.0. The average must then pass through untouched:
the arithmetic form multiplies the live tree by zero, and zero times a
non-finite parameter is NaN, so one bad parameter would poison the frozen
tree on the step it appears. These tests force non-finite parameters through
the update itself, through whole compiled steps with and without
`dynamic_scale`, and through a checkpoint round trip. The general restore
path is covered in test_trainer.py; what is asserted here is the frozen
value: bit-identical before and after.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from affine_run import Counting
from affine_training import frozen_trainer

from dew.data.dataset import Loader


def frozen_bytes(state):
    return [np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(state.ema)]


def poison(state):
    """Replace a live kernel without changing the aliased frozen reference."""
    params = dict(state.variables)
    collection = dict(params["params"])
    layer = dict(collection["Dense_0"])
    layer["kernel"] = jnp.full_like(layer["kernel"], jnp.inf)
    collection["Dense_0"] = layer
    params["params"] = collection
    return dataclasses.replace(state, variables=params)


@pytest.mark.parametrize("dynamic_scale", [False, True])
def test_a_non_finite_live_parameter_leaves_the_reference_bit_identical(dynamic_scale):
    """One compiled step over infinite parameters, with and without the
    mixed-precision gate. The parameters may go wherever the update takes
    them; the frozen tree stays bitwise where it was."""
    trainer = frozen_trainer(dynamic_scale=dynamic_scale)
    state = poison(trainer.initial_state())
    before = frozen_bytes(state)
    batch = next(Counting())
    step = trainer.compile(state, batch)
    new_state, loss, _, finite, _, _ = step(state, batch)

    assert not bool(finite), "the step should have seen the poison"
    assert not bool(jnp.isfinite(loss)), "the loss should have seen the poison"
    assert frozen_bytes(new_state) == before


def test_a_resumed_run_restores_the_frozen_reference(tmp_path):
    """Two steps freeze the reference at init; the checkpoint carries those
    bytes and the resumed run trains on."""
    trainer = frozen_trainer(tmp_path)
    # The placed init is what fit starts from; the eager init rounds the
    # random draws differently on a GPU.
    initial = frozen_bytes(trainer.place()[0])
    data = Loader(lambda partition: Counting(), None, batch=8, records=None)
    trainer.fit(data, steps=2, log_every=1)

    resumed = frozen_trainer(tmp_path)
    state, _, _ = resumed.place()

    assert int(state.step) == 2
    assert frozen_bytes(state) == initial
    resumed.fit(data, steps=4, log_every=1)
    assert frozen_bytes(resumed.place()[0]) == initial
