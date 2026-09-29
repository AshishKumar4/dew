# Checkpoints

`dew.Checkpoints(directory)` holds the checkpoints of one run. Given to a `Trainer`, it saves the training state and the data iterator's position every `checkpoint_every` steps and at the end of `fit`, and a later `fit` with the same directory resumes from the newest step. A training checkpoint holds the model variables, the optimizer state, three step counters (attempted batches, accepted microbatches and optimizer updates), the root key, the EMA copy when the objective keeps one, the loss scaler's history, and any half-filled gradient accumulation window.

## Example

The example trains for five steps, restores the state in a new `Trainer`, and continues to step ten. Its iterator reports its batch index as bytes through `get_state` and `set_state`, so the checkpoint can store the data position.

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
    print("latest:", checkpoints.latest)
    print(sorted(path.name for path in Path(directory).iterdir()))

    resumed = Trainer(Regression(), optax.sgd(0.1), key=jax.random.key(0),
                      checkpoints=checkpoints)
    restored, _, position = resumed.place()
    print("restored step:", int(restored.step), "data position:", position)
    final = resumed.fit(data, steps=10, checkpoint_every=5)
    print("final step:", int(final.step), "latest:", checkpoints.latest)
```

```text
Training from step 0 to 5 on 1 × cpu, 2 parameters
Trained 5 steps in 0:00:00: first step after 0.23 s, then 284.7 step/s
3.7% of the wall time in steps, final loss 0.2252
latest: 5
['5']
Resumed from step 5 in /mnt/scratch/dew/tmp/dew-checkpoint-ey8r19jk
restored step: 5 data position: b'5'
Resumed from step 5 in /mnt/scratch/dew/tmp/dew-checkpoint-ey8r19jk
Training from step 5 to 10 on 1 × cpu, 2 parameters
Trained 5 steps in 0:00:00: first step after 0.12 s, then 1890.3 step/s
0.9% of the wall time in steps, final loss 0.0271
final step: 10 latest: 10
```

`steps` is the step count to finish at, not a number of steps to add: the second `fit(..., steps=10)` starts at step five and runs five more. `place()` returns the restored `TrainState`, its placement and the saved iterator position. Prefetching may read batches ahead of the loop, but the saved position is the last batch the loop consumed. The example writes into a temporary directory; a run that should persist needs a directory of its own, passed again on later calls.

## Checkpoints

`Checkpoints(directory, *, keep=2, local_directory=None, local_every=None)`:

| Argument | Meaning |
|---|---|
| `directory` | Where the persistent checkpoints go: a path on a filesystem every process shares, or a bucket (`gs://...`). One numbered subdirectory per step. |
| `keep` | How many of the latest steps to keep. The step with the lowest `loss` metric a save reported is kept as well. |
| `local_directory`, `local_every` | A path on every host's own disk where `fit` writes one more checkpoint, the latest, every `local_every` steps. Both or neither. |

| Member | Meaning |
|---|---|
| `latest` | The newest step every process can read, local or persistent. |
| `best` | The kept step with the lowest reported loss. |
| `path(step)` | The directory of one step. |
| `restore(template, step=None)` | Read a step into the structure of `template`. |
| `stored(step=None)` | The shapes and dtypes each saved state field holds, without reading values. |
| `wait()` | Block until asynchronous saves have finished. |

Saves are asynchronous; `wait()` returns once they are durable. Constructing `Checkpoints` opens nothing; the Orbax managers are created on first use.

## The EMA copy on disk

Orbax compresses every array of a checkpoint with zstd, which in the two runs below saves 6 to 9% on weights and Adam moments: past the sign and exponent, their bits look random. An EMA copy is different, because it agrees with the weights it follows in its sign, exponent and leading mantissa bits. `Checkpoints` stores each EMA leaf that has its weight's floating dtype and shape as the XOR of the two, split into byte planes: a leading `uint8` axis with one plane per byte, least significant first. The XOR is zero wherever the two agree, and the planes gather those zeros into long runs, which zstd compresses well. The step's metadata lists the leaves stored this way. `restore` and `stored` return them bit for bit, as the run trained them, in the dtype and placement the template asks for and on any mesh. Checkpoints written before this change list no such leaves and restore as they were written.

| Run | EMA, zstd alone | XOR, then zstd | XOR and byte planes, then zstd |
|---|---|---|---|
| 176M-parameter DiT, fp32, step 1.35M | 652.7 MB | 560.3 MB | 488.6 MB (−25%) |
| 8.7M-parameter DiT, fp32, step 750, decay 0.999 | 32.2 MB | 30.8 MB | 28.2 MB (−12%) |

How much the XOR saves depends on how close a run's EMA sits to its weights, which follows its updates and its decay; the table measures two runs. Written through `Checkpoints` with JAX's CPU backend, the 176M model's weights and EMA take 1139.4 MB instead of 1306.0 MB. In three processes of three saves and restores each, alternating with processes running the code before this change, a compiled save blocked the step for 0.19 to 0.46 s instead of 0.04 to 0.14 s, the time to compute the planes, and landed on disk in 1.9 to 3.7 s instead of 1.8 to 2.3 s, apart from saves that took 17 to 74 s either way while the shared disk was busy. Restores took 1.45 to 1.59 s instead of 1.31 to 1.71 s. The first save and the first restore of a process each compile one small program per distinct EMA leaf shape (19 for this model): there, a save blocked for 1.0 to 2.7 s and a restore took 2.0 to 3.0 s.

Planes are computed on the devices that hold the EMA. So a save holds one more EMA-sized buffer there until Orbax has copied it to the host, which fits in memory that the step's gradient frees between steps. EMA leaves in pinned host memory, as a host layout keeps them, are written as themselves: Orbax writes them from their own buffers, and differencing them would keep a second copy in the memory the layout exists to spare. EMA leaves whose dtype differs from their weight's are also stored as themselves.

## Preemption

A scheduler stops a job with SIGTERM and kills it a grace period later: Slurm's `KillWait`, Kubernetes' termination grace period, a spot VM's notice. `Trainer.fit` stops at the next step every process agrees on (JAX's `reached_preemption_sync_point` in a pool, the signal itself in a lone process), writes that step's state and data position, skips the final validation, and raises `dew.training.Preempted`. Uncaught, it ends the program with exit status 143, SIGTERM's, so the scheduler sees a stopped job rather than a finished one; a Kubernetes pod failure policy can ignore that code. The same command run again resumes from the checkpoint, and with deterministic ops its losses are the uninterrupted run's to the bit.

The save has to fit in the scheduler's grace: a step and one checkpoint write. `dew launch` gives a pool it was told to stop 300 seconds before it kills the ranks, and a second signal kills them at once ([Training on several nodes](multi-node.md)).

## Data position

A plain Python generator usually cannot report its position. To continue the data sequence, use a built-in source that supports checkpointing, or give the iterator both `get_state` and `set_state` as the example does. An iterator rebuilt from the start may replay records even though the model weights restore correctly. A stream that cannot save its position cannot produce a resumable checkpoint, so its `checkpoint_every` must be `None`; the final state is still written.

A persistent checkpoint can restore into a different, compatible placement through the trainer's restore template. Whether the process count can change depends on the saved position:

- A global record count can be read at any process count. Every record dataset built on `train_stream` saves this kind of position.
- One share's own offset, which custom iterators like the one above report, can only be read by a reader of the same share (`DataPartition`), whichever processes those are. Restore refuses any other share and names the shares the checkpoint holds.
- A checkpoint without a position can change the process count when the tensor layout is compatible.

Local checkpoints hold only the shards each process had, so they restore onto the placement they were written with. See [Training data](../concepts/data.md) for how datasets save their position.

## Several hosts

Every process of a pool writes its own shards of a persistent checkpoint, process 0 writes the metadata and commits the step, and a restore reads shards other processes wrote. So every process must read and write the same directory, on a shared filesystem or in a bucket. On disks of their own, process 0 would commit steps without the other processes' shards, and those processes would see no step at all. The first time a pool uses `Checkpoints`, process 0 writes a file into the directory and every process checks that it can see it; if any cannot, every process raises an error that names the processes that cannot. Buckets skip this check. Each host's own disk can still hold the local checkpoints next to the shared directory.

[Distributed training](../concepts/distributed.md) covers topology requirements and [Cloud TPUs](../tpu.md) remote setup. A local save and restore does not show that cross-host recovery or remote storage work.

## What a checkpoint does not save

`Checkpoints` does not write `run.json`; recipes that use `RunConfig.save` or `RunConfig.train` write the run configuration separately. `RunConfig.load` reads a record an older Dew wrote: a field the record lacks takes its default, which is what runs recorded before the field existed did. It refuses a field it does not know, such as one a newer Dew wrote. A checkpoint also does not save source code, package versions, tokenizer files, the dataset revision, or the state of any external service. Record those in the experiment metadata.

## Limits

Resume with the same model, optimizer, accumulation length and scaler configuration the run trained with; the trainer refuses a different accumulation length. When the loss scaler rejects a step's gradients, the attempt still counts, and the accumulation records, optimizer state, EMA and mutable contributions accepted before it stay as they were. Restoring keeps the scaler's streak of finite steps and its scale, including a partially filled window. If the checkpoint already reached the target step, resuming reads no data and does no evaluation, compilation or saving.

Deterministic CPU tests cover checkpoints taken in the middle of a window and after rejected attempts, including composite replay. They do not cover cross-host recovery on GPU or TPU, or replaying the side effects of external rollouts. The per-fit counter that stops a run after repeated non-finite losses is not checkpointed. A training checkpoint without the step counters and accumulation fields cannot resume through this interface, but its parameters can still be loaded alone.
