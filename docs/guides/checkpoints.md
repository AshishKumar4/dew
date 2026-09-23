# Resuming training

A training checkpoint holds the model variables, the optimizer state, three step counters (attempted batches, accepted microbatches and optimizer updates), the root key, the EMA copy when you configure one, the loss scaler's history, and any half-filled gradient accumulation window. To continue the data sequence exactly, the checkpoint also needs the data iterator's position. When you resume, use the same model, optimizer, accumulation length and scaler configuration you trained with.

## Save and restore a small run

The example below trains for five steps, restores the state, and continues to step ten. Its iterator records its batch index as bytes, so the checkpoint can store the data position. It writes into a temporary directory so that repeated runs do not overwrite another experiment.

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


data = Dataset(train=lambda partition: Batches(), val=None, records=None, batch=8)
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

`steps` is the step count to finish at, not a number of steps to add. The second `fit(..., steps=10)` starts at step five and runs five more steps. `place()` returns three things: the restored `TrainState`, its placement, and the saved iterator position. Prefetching may read batches ahead of the loop, but in this run the saved position is the last batch the loop consumed.

The script checks the checkpoint directory and the continued run before the temporary directory is deleted. To keep a run, replace the temporary directory with a persistent path of its own and pass that path on later calls. Do not reuse the directory of an unrelated experiment.

## What a checkpoint does not save

`Checkpoints` does not write `run.json`. Recipes that use `RunConfig.save` or `RunConfig.train` write the run configuration separately. A checkpoint also does not save your source code, package versions, tokenizer files, dataset revision, or the state of any external service. Record those in your experiment metadata.

A plain Python generator usually cannot report its position. To continue the data sequence, use a built-in source that supports checkpointing, or give your iterator both `get_state` and `set_state` as the example does. If you rebuild an iterator from the start instead, it may replay records even though the model weights restore correctly.

## Change placement deliberately

A persistent checkpoint can restore into a different, compatible placement through the trainer's restore template. Whether the process count can change depends on the saved iterator position:

- A global record count can be read at any process count. Every record dataset built on `train_stream` saves this kind of position.
- One share's own offset, which custom iterators like the one above report, can only be read by a reader of the same share (`DataPartition`), whichever processes those are. Restore refuses any other share and names the shares the checkpoint holds.
- A checkpoint without a position can change the process count when the tensor layout is compatible.

Local emergency checkpoints hold only the shards each process had, so they restrict placement further. See [resume from a data position](../concepts/data.md#resume-from-a-data-position).

## Several hosts need one checkpoint directory

Every process of a pool writes its own shards of a persistent checkpoint, process 0 writes the metadata and commits the step, and a restore reads shards other processes wrote. So the directory must be one that every process reads and writes: a shared filesystem or a bucket. On disks of their own, process 0 would commit steps without the other processes' shards, and those processes would see no step at all. The first time a pool uses `Checkpoints`, process 0 writes a file into the directory and every process checks that it can see it. If any process can't, every process raises an error that names the processes that can't see it. Buckets (`gs://...`) skip this check. Each host's own disk can still hold the local checkpoints (`local_directory`) next to the shared directory.

See [distributed training](../concepts/distributed.md) for topology requirements and [the TPU guide](../tpu.md) for remote setup. A local save and restore test does not show that cross-host recovery or remote storage work.

## Current limits

When the loss scaler rejects a step's gradients, the attempt still counts. The accumulation records, optimizer state, EMA and mutable contributions accepted before it stay as they were. Restoring keeps the scaler's streak of finite steps and its scale, including a partially filled window. The trainer refuses to resume with a different accumulation length. If the checkpoint already reached the target step, resuming reads no data and does no evaluation, compilation or saving.

Deterministic CPU tests cover checkpoints taken in the middle of a window and after rejected attempts, including composite replay. They do not cover cross-host recovery on GPU or TPU, or replaying the side effects of external rollouts. The per-fit counter that stops a run after repeated non-finite losses is not checkpointed. A training checkpoint without the step counters and accumulation fields cannot resume through this interface, but you can still load its parameters alone.
