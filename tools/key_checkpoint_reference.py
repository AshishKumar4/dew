"""Save an actual old-code legacy-key checkpoint, then verify its continuation."""
import argparse
import json
import subprocess
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from dew import Trainer
from dew.data import Dataset
from dew.objectives.base import Aux, EMASpec, Objective
from dew.training import Checkpoints, Layout, MeshSpec

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("old", "new"))
parser.add_argument("--out", type=Path, required=True)
options = parser.parse_args()
directory = options.out.resolve()
directory.mkdir(parents=True, exist_ok=True)


class Linear(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(2)(x)


class Regression(Objective):
    model = Linear()
    ema = EMASpec(optax.constant_schedule(.5))

    def init(self, key, variables=None):
        return self.model.init(key, jnp.zeros((1, 3)))

    def loss(self, variables, batch, step):
        prediction = self.model.apply(variables, batch["x"])
        factor = .5 + jax.random.uniform(step.key)
        return factor * jnp.mean((prediction - batch["y"]) ** 2), Aux({})


class Rows:
    def __init__(self):
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        x = np.random.default_rng(self.index).normal(size=(8, 3)).astype(np.float32)
        self.index += 1
        return {"x": x, "y": 2 * x[:, :2]}

    def get_state(self):
        return json.dumps({"index": self.index}).encode()

    def set_state(self, saved):
        self.index = json.loads(saved)["index"]


data = Dataset(train=lambda partition: Rows(), val=None, records=None, batch=8)


def build(checkpoints=None):
    return Trainer(Regression(), optax.adam(1e-3), key=jax.random.PRNGKey(0),
                   mesh=MeshSpec(fsdp=2), layout=Layout(min_shard=1, tolerance=1.0), checkpoints=checkpoints)


def raw(leaf):
    return jax.random.key_data(leaf) if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key) else leaf


if options.mode == "old":
    old = build(Checkpoints(str(directory / "old-checkpoint"), keep=2))
    prefix = old.fit(data, steps=2, log_every=1, checkpoint_every=2)
    assert prefix.key.dtype == jnp.uint32, "old mode must use a pre-cutover Dew checkout"
    old.checkpoints.wait()
    shutil.copytree(directory / "old-checkpoint", directory / "old-continuation-checkpoints")
    old_continued = build(Checkpoints(str(directory / "old-continuation-checkpoints"), keep=2)).fit(data, steps=4, log_every=1)
    uninterrupted = build().fit(data, steps=4, log_every=1)
    for actual, expected in zip(jax.tree.leaves(old_continued), jax.tree.leaves(uninterrupted), strict=True):
        np.testing.assert_array_equal(np.asarray(raw(actual)), np.asarray(raw(expected)))
    np.savez(directory / "old-continuation.npz", **{str(index): np.asarray(raw(leaf))
             for index, leaf in enumerate(jax.tree.leaves(old_continued))})
    import dew
    checkout = Path(dew.__file__).parents[2]
    (directory / "old-metadata.json").write_text(json.dumps({
        "commit": subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip(),
        "jax": jax.__version__, "key_dtype": str(prefix.key.dtype), "key_sharding": str(prefix.key.sharding.spec),
        "fsdp": 2, "devices": len(jax.devices()), "steps": 2, "continuation_steps": 4}, indent=2) + "\n")
    print("old checkpoint and old continuation saved", flush=True)
else:
    new = build(Checkpoints(str(directory / "old-checkpoint"), keep=2))
    restored, shardings, _ = new.place()
    assert int(restored.step) == 2, "the old checkpoint must be the step-2 prefix, not a completed run"
    assert jnp.issubdtype(restored.key.dtype, jax.dtypes.prng_key)
    assert shardings.key.spec == jax.sharding.PartitionSpec()
    continued = new.fit(data, steps=4, log_every=1)
    expected = np.load(directory / "old-continuation.npz")
    for index, leaf in enumerate(jax.tree.leaves(continued)):
        actual = np.asarray(raw(leaf))
        assert actual.dtype == expected[str(index)].dtype and actual.shape == expected[str(index)].shape
        assert actual.tobytes() == expected[str(index)].tobytes(), index
    print("legacy checkpoint from actual old e868862f resumes with every state leaf BITWISE identical on fsdp=2", flush=True)
