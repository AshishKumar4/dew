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
import json

from flax import linen as nn
from flax.training import dynamic_scale as dynamic_scale_lib
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.objectives.base import Aux, EMASpec, Objective
from dew.training import Checkpoints, Layout, Trainer
from dew.training.trainer import ema_update

FEATURES = 3


class Affine(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(2)(x)


class Regression(Objective):
    """Squared error of an affine map, with the reference frozen at init."""

    def __init__(self):
        self.model = Affine()
        self.ema = EMASpec(decay=optax.constant_schedule(1.0))

    def init(self, key):
        return self.model.init(key, jnp.zeros((1, FEATURES)))

    def loss(self, params, batch, step):
        prediction = self.model.apply(params, batch["x"])
        return jnp.mean((prediction - batch["y"]) ** 2), Aux({"probe": jnp.asarray(1.0)})


class Counting:
    """A deterministic, checkpointable stream of regression batches."""

    def __init__(self, batch=8):
        self.index = 0
        self.batch = batch

    def __iter__(self):
        return self

    def __next__(self):
        rng = np.random.default_rng(self.index)
        self.index += 1
        x = rng.normal(size=(self.batch, FEATURES)).astype(np.float32)
        return {"x": x, "y": 2 * x[:, :2]}

    def get_state(self):
        return json.dumps({"index": self.index}).encode()

    def set_state(self, state):
        self.index = json.loads(state)["index"]


class Data:
    """The `Dataset` contract the trainer reads: train, val, batch, records."""

    def __init__(self, train=Counting, val=None, batch=8, records=None):
        self._train, self._val = train, val
        self.batch, self.records = batch, records

    def train(self):
        return self._train()

    @property
    def val(self):
        return self._val

    @property
    def steps_per_epoch(self):
        return None if self.records is None else self.records // self.batch


def make_trainer(tmp_path=None, **kwargs):
    checkpoints = None if tmp_path is None else Checkpoints(str(tmp_path / "run"))
    return Trainer(
        Regression(),
        optax.sgd(0.1),
        key=jax.random.key(0),
        layout=Layout(min_shard=1, tolerance=1.0),
        checkpoints=checkpoints,
        **kwargs,
    )


def frozen_bytes(state):
    return [np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(state.ema)]


def test_unit_decay_returns_the_average_untouched():
    """`0.0 * NaN` is NaN, so the old form `1.0 * average + 0.0 * live`
    poisons the tree wherever the live tree is non-finite. The select keeps
    every leaf bit-identical, including a NaN the average itself holds."""
    ema = {"params": {"kernel": jnp.array([1.0, jnp.nan]), "bias": jnp.array(0.5)}}
    live = {"params": {"kernel": jnp.array([jnp.inf, 3.0]), "bias": jnp.array(jnp.nan)}}

    updated = ema_update(ema, live, jnp.asarray(1.0))

    for before, after in zip(jax.tree.leaves(ema), jax.tree.leaves(updated), strict=True):
        assert np.asarray(before).tobytes() == np.asarray(after).tobytes()
    moved = ema_update(ema, live, 0.5)
    assert any(np.asarray(before).tobytes() != np.asarray(after).tobytes()
               for before, after in zip(jax.tree.leaves(ema), jax.tree.leaves(moved),
                                        strict=True)), "a lower decay must still average"


def poison(state):
    """The state's parameters with an infinite kernel, ema untouched.

    Every leaf is a fresh buffer: `select` aliases the ema leaves with the
    params leaves, and the compiled step donates the state, so shared
    buffers would be donated twice."""
    params = jax.tree.map(lambda leaf: jnp.asarray(np.asarray(leaf)), state.params)
    collection = dict(params["params"])
    layer = dict(collection["Dense_0"])
    layer["kernel"] = jnp.full_like(layer["kernel"], jnp.inf)
    collection["Dense_0"] = layer
    params["params"] = collection
    return dataclasses.replace(state, params=params)


@pytest.mark.parametrize("dynamic_scale", [False, True])
def test_a_non_finite_live_parameter_leaves_the_reference_bit_identical(dynamic_scale):
    """One compiled step over infinite parameters, with and without the
    mixed-precision gate. The parameters may go wherever the update takes
    them; the frozen tree stays bitwise where it was."""
    trainer = make_trainer(dynamic_scale=dynamic_scale)
    state = poison(trainer.initial_state())
    before = frozen_bytes(state)
    batch = next(Counting())
    
    step = trainer.compile(state, batch)

    new_state, loss, _, finite, _ = step(state, batch)

    assert not bool(finite), "the step should have seen the poison"
    assert not bool(jnp.isfinite(loss)), "the loss should have seen the poison"
    assert frozen_bytes(new_state) == before


def test_a_resumed_run_restores_the_frozen_reference(tmp_path):
    """Two steps freeze the reference at init; the checkpoint carries those
    bytes and the resumed run trains on."""
    trainer = make_trainer(tmp_path)
    # The placed init is what fit starts from; the eager init rounds the
    # random draws differently on a GPU.
    initial = frozen_bytes(trainer.place()[0])
    trainer.fit(Data(), steps=2, log_every=1)

    resumed = make_trainer(tmp_path)
    state, _, _ = resumed.place()

    assert int(state.step) == 2
    assert frozen_bytes(state) == initial
    resumed.fit(Data(), steps=4, log_every=1)
    assert frozen_bytes(resumed.place()[0]) == initial
