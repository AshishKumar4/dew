# Resuming training

A training checkpoint holds variables, optimizer state, attempted/accepted/update clocks, the root key, EMA when configured, scaler history, and the actual partial accumulation window. Exact data continuation also needs the iterator position. Resume with the same model, optimizer, accumulation length, and scaler configuration.

## Save and restore a small run

The complete example below trains for five steps, restores the state, and continues to step ten. Its custom iterator records a batch index as bytes. It uses a temporary directory so repeated runs do not overwrite another experiment.

```python
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from dew import Checkpoints, Trainer
from dew.data import Dataset
from dew.objectives.base import Aux, Mean, Objective


class Regression(Objective):
    def __init__(self):
        self.model = nn.Dense(features=1)

    def init(self, key, variables=None):
        return self.model.init(key, jnp.zeros((1, 1), jnp.float32))

    def loss(self, variables, batch, step):
        prediction = self.model.apply(variables, batch["x"])
        errors = (prediction - batch["y"]) ** 2
        return Mean(jnp.sum(errors), jnp.asarray(errors.size)), Aux(metrics={})


class Batches:
    def __init__(self):
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        rng = np.random.default_rng(self.index)
        x = rng.normal(size=(8, 1)).astype(np.float32)
        self.index += 1
        return {"x": x, "y": 2 * x + 1}

    def get_state(self):
        return str(self.index).encode()

    def set_state(self, state):
        self.index = int(state.decode())


data = Dataset(train=Batches, val=None, records=None, batch=8)
with tempfile.TemporaryDirectory(prefix="dew-checkpoint-") as directory:
    checkpoints = Checkpoints(directory)
    trainer = Trainer(Regression(), optax.sgd(0.1), key=jax.random.key(0),
                      checkpoints=checkpoints)
    trainer.fit(data, steps=5, checkpoint_every=5)
    checkpoints.wait()
    assert checkpoints.latest == 5
    assert Path(checkpoints.path(5)).is_dir()
    assert not (Path(directory) / "run.json").exists()

    resumed = Trainer(Regression(), optax.sgd(0.1), key=jax.random.key(0),
                      checkpoints=checkpoints)
    restored, _, position = resumed.place()
    assert int(restored.step) == 5
    assert position == b"5"
    final = resumed.fit(data, steps=10, checkpoint_every=5)
    assert int(final.step) == 10
    assert checkpoints.latest == 10
    print("Restored step 5 and data position 5; continued to step 10.")
```

`steps` is a final target, not an additional step count. The second `fit(..., steps=10)` starts at step five and performs five more steps. `place()` returns the restored `TrainState`, its placement description, and the recorded iterator position. Prefetch may read ahead, but the position saved for this ordinary run corresponds to the last batch the loop consumed.

The script verifies checkpoint directories and continuation before the temporary directory is removed. To retain a run, replace the temporary-directory context with a dedicated persistent path and keep that path for later calls. Never reuse an unrelated experiment directory.

## What a checkpoint does not save

`Checkpoints` does not create `run.json`. Recipes using `RunConfig.save` or `RunConfig.train` write the run configuration separately. Saving model state alone does not preserve the source code, package versions, tokenizer files, dataset revision, or all external service state. Record those in your experiment metadata.

A plain Python generator generally has no restorable position. To continue the data sequence, use a checkpointable built-in source or implement both iterator-state methods. Reconstructing an iterator from the beginning may replay records even when model weights restore correctly.

## Change placement deliberately

Persistent checkpoints can restore into a compatible placement through the trainer's restore template. A saved iterator position can change the process count when it is a global record count, which every record dataset built on `train_stream` reports; a position that is one process's own shard offset, as the packed sources (`PackedTokens`, `ChatMessages`) and custom iterators report, requires the count that wrote it, and restore rejects a different count naming both. Position-free checkpoints can change process count when the tensor layout is compatible. Local emergency checkpoints contain each process's available shards and impose additional placement restrictions. See [resume from a data position](../concepts/data.md#resume-from-a-data-position).

Use [distributed training](../concepts/distributed.md) for topology requirements and [the TPU guide](../tpu.md) for remote setup. A local save/restore test does not prove cross-host recovery or remote storage behavior.

## Current limits

Scaled-gradient rejection consumes an attempt but preserves previously accepted accumulation records, optimizer state, EMA and mutable contributions. Restore keeps the scaler's finite streak and scale, including a partially filled window. The trainer rejects a changed accumulation length. Resuming at the saved attempted-work target performs no new data, evaluation, compilation, or save work.

Deterministic CPU regressions cover partial and rejected-attempt checkpoints, including composite replay. They do not establish GPU/TPU cross-host recovery or replay of external rollout side effects. Record software and data versions with the run; the per-fit nonfinite-loss abort counter is not checkpointed. Older training checkpoints lack the required clocks and accumulation fields and cannot resume through this interface. Parameter-only loading remains available.
