# Resuming training

This guide assumes you understand the [training example](../getting-started.md) and Flax variables. A checkpoint can hold model variables, optimizer state, the training step, the run key, EMA variables when configured, and the training iterator's position. Continuing a run requires the same model and compatible optimizer and data configuration.

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
from dew.objectives.base import Aux, Objective


class Regression(Objective):
    def __init__(self):
        self.model = nn.Dense(features=1)

    def init(self, key):
        return self.model.init(key, jnp.zeros((1, 1), jnp.float32))

    def loss(self, variables, batch, step):
        prediction = self.model.apply(variables, batch["x"])
        return jnp.mean((prediction - batch["y"]) ** 2), Aux(metrics={})


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

Persistent checkpoints can restore into a compatible placement through the trainer's restore template. A checkpoint containing saved iterator position requires the same JAX process count; restore rejects a different count even for persistent storage. Position-free checkpoints can change process count when the tensor layout is compatible. Local emergency checkpoints contain each process's available shards and impose additional placement restrictions.

Use [distributed training](../concepts/distributed.md) for topology requirements and [the TPU guide](../tpu.md) for remote setup. A local save/restore test does not prove cross-host recovery or remote storage behavior.

## Current limits

This example uses ordinary float32 training with no accumulation or loss scaling. The current overflow path has a confirmed discrepancy between attempted loop steps and saved `state.step`; dynamic loss-scale state also resets on restore. Exact continuation after rejected scaled-gradient updates is not established. Unequal-mask gradient accumulation is a separate normalization issue. Track these before relying on such runs for reproducibility.

Prefetch iterator cancellation and cleanup after exceptions also remain under review. The successful small example verifies the normal checkpoint path only.
