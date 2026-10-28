"""The affine regression the loop-level suites train, with the streams and
the metric they read: small enough to step in a test, with an EMA, a metric
of its own and a checkpointable stream."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from dew.artifacts import Representations
from dew.data.dataset import Loader
from dew.objectives.base import Aux, EMASpec, Objective, mean_of_totals, merge_totals

BATCH = 8
FEATURES = 3


class Affine(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(2)(x)


class Regression(Objective):
    """Squared error of an affine map against `2 * x[:, :2]`."""

    def __init__(self, ema_decay=0.5, *, probe=True):
        self.model = Affine()
        self.ema = EMASpec(decay=optax.constant_schedule(ema_decay))
        self.probe = probe

    def fresh_variables(self, key, held):
        return self.model.init(key, jnp.zeros((1, FEATURES)))

    def loss(self, variables, batch, step):
        prediction = self.model.apply(variables, batch["x"])
        return jnp.mean((prediction - batch["y"]) ** 2), Aux(
            {"probe": jnp.asarray(1.0)} if self.probe else {})


class TimingRegression(Regression):
    """Instrumentation's affine step: a slower EMA and no probe metric."""

    def __init__(self):
        super().__init__(ema_decay=0.9, probe=False)


def batches():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(BATCH, FEATURES)).astype(np.float32)
    batch = {"x": x, "y": 2 * x[:, :2]}
    while True:
        yield batch


class Counting:
    """An endless, checkpointable stream whose batches say which they are."""

    def __init__(self, batch=BATCH):
        self.index = 0
        self.batch = batch

    def __iter__(self):
        return self

    def __next__(self):
        rng = np.random.default_rng(self.index)
        self.index += 1
        x = rng.normal(size=(self.batch, FEATURES)).astype(np.float32)
        return {"x": x, "y": 2 * x[:, :2], "index": np.full((self.batch,), self.index - 1)}

    def get_state(self):
        return json.dumps({"index": self.index}).encode()

    def set_state(self, state):
        self.index = json.loads(state)["index"]


class Replaying:
    """An endless, checkpointable stream of `batch_at(position)`, the batch for each step it has
    handed out, its state that count."""

    def __init__(self, batch_at):
        self.batch_at, self.position = batch_at, 0

    def __iter__(self):
        return self

    def __next__(self):
        self.position += 1
        return self.batch_at(self.position - 1)

    def get_state(self):
        return str(self.position).encode()

    def set_state(self, state):
        self.position = int(state)


def dataset(train=Counting, val=None, batch=BATCH) -> Loader:
    """A dataset over fresh train and validation iterators, without an epoch count."""
    return Loader(train=lambda partition: train(),
                   val=None if val is None else lambda partition: val(),
                   batch=batch, records=None)


def val_batches(count=3):
    def stream():
        source = Counting()
        for _ in range(count):
            yield next(source)
    return stream


def raw_leaf(leaf):
    return jax.random.key_data(leaf) if jnp.issubdtype(
        leaf.dtype, jax.dtypes.prng_key) else leaf


class Features(Regression):
    """An objective whose evaluation returns its predictions as representations."""

    def evaluate(self, params, batch, step):
        params = params if step.ema is None else step.ema
        return Representations(features=self.model.apply(params, batch["x"]),
                               labels=batch["index"])


class Spread:
    name = "spread"
    reads = Representations

    def __init__(self, seen):
        self.seen = seen

    def __call__(self, artifact, batch):
        self.seen.append((np.asarray(artifact.features).shape, np.asarray(batch["x"]).shape))
        return float(jnp.std(artifact.features)), 1

    merge = staticmethod(merge_totals)
    finalize = staticmethod(mean_of_totals)
