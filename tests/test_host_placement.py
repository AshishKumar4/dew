"""Optimizer state and EMA kept in pinned host memory between steps.

`Layout.host` changes where those fields live, not what they hold: the step
fetches them to the device, runs the update it always ran, and writes them
back. JAX 0.11.1 types memory spaces, so a field left on the host would fail
the first device op that touched it, and a field fetched but not written back
would land on the device; the memory kinds on the state leaves are asserted
for that reason. The CPU backend exposes `pinned_host` beside `device`, so the
placement, the checkpoint round trip and the resume run here as written.
"""

import jax
import numpy as np
import optax
import pytest

from dew.training import Checkpoints, Layout, Trainer
from test_trainer import Data, RecordingTracker, Regression, val_batches

DEVICE = Layout(min_shard=1, tolerance=1.0)
HOST = Layout(min_shard=1, tolerance=1.0, host=("opt_state", "ema"))


def fit(layout, directory, steps):
    trainer = Trainer(Regression(), optax.adam(0.1), key=jax.random.key(0), layout=layout,
                      checkpoints=Checkpoints(str(directory), keep=3), tracker=RecordingTracker())
    state = trainer.fit(Data(val=val_batches()), steps=steps, log_every=1, eval_every=2,
                        checkpoint_every=2)
    trainer.checkpoints.wait()
    return state


def identical(left, right):
    return all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True))


def memory_kinds(tree):
    return {leaf.sharding.memory_kind for leaf in jax.tree.leaves(tree)}


def test_host_resident_state_trains_the_same_model(tmp_path):
    on_device = fit(DEVICE, tmp_path / "device", 4)
    on_host = fit(HOST, tmp_path / "host", 4)
    assert memory_kinds(on_host.opt_state) == memory_kinds(on_host.ema) == {"pinned_host"}
    assert memory_kinds(on_host.params) == memory_kinds(on_device.opt_state) == {"device"}
    assert identical(on_host.params, on_device.params)
    assert identical(on_host.opt_state, on_device.opt_state)
    assert identical(on_host.ema, on_device.ema)


def test_host_resident_state_resumes_from_its_checkpoint(tmp_path):
    saved = fit(HOST, tmp_path / "host", 4)
    fit(DEVICE, tmp_path / "device", 4)
    resumed = fit(HOST, tmp_path / "host", 6)
    reference = fit(DEVICE, tmp_path / "device", 6)
    assert int(resumed.step) == 6
    assert memory_kinds(resumed.opt_state) == memory_kinds(resumed.ema) == {"pinned_host"}
    assert identical(resumed.params, reference.params)
    assert identical(resumed.opt_state, reference.opt_state)
    assert identical(resumed.ema, reference.ema)
    assert not identical(resumed.params, saved.params)


def test_a_layout_places_only_the_state_it_can_fetch():
    with pytest.raises(ValueError, match="opt_state.*ema.*params"):
        Layout(host=("params",))
