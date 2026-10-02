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
from dew.objectives.base import Aux, Ratio, Objective


class Regression(Objective):
    def __init__(self):
        self.model = nn.Dense(features=1)

    def init(self, key, variables=None):
        return self.model.init(key, jnp.zeros((1, 1), jnp.float32))

    def loss(self, variables, batch, step):
        prediction = self.model.apply(variables, batch["x"])
        errors = (prediction - batch["y"]) ** 2
        return Ratio(jnp.sum(errors), jnp.asarray(errors.size)), Aux(metrics={})


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
Training Dense from step 0 to 5: 2 parameters, on 1 × cpu, batch 8, float32
Trained 5 steps in 0:00:00: first step after 0.20 s, then 506.0 step/s
3.3% of the wall time in steps, final loss 0.2252
latest: 5
['5']
Resumed from step 5 in /tmp/dew-checkpoint-ryuexi2c
restored step: 5 data position: b'5'
Resumed from step 5 in /tmp/dew-checkpoint-ryuexi2c
Training Dense from step 5 to 10: 2 parameters, on 1 × cpu, batch 8, float32
Trained 5 steps in 0:00:00: first step after 0.10 s, then 1501.2 step/s
2.1% of the wall time in steps, final loss 0.02705
final step: 10 latest: 10
```

`steps` is the step count to finish at, not a number of steps to add: the second `fit(..., steps=10)` starts at step five and runs five more. `place()` returns the restored `TrainState`, its placement and the saved iterator position. Prefetching may read batches ahead of the loop, but the saved position is the last batch the loop consumed. The example writes into a temporary directory; a run that should persist needs a directory of its own, passed again on later calls.

## Checkpoints

`Checkpoints(directory, *, keep=2, local_directory=None, local_every=None)`:

| Argument | Meaning |
|---|---|
| `directory` | Where the persistent checkpoints go: a path on a filesystem every process shares, or a bucket (`gs://...`). One numbered subdirectory per step. |
| `keep` | Latest full states to retain, unioned with best-K and post-hoc snapshots. An int is latest-N; `Keep` also adds periodic, wall-time-spaced and predicate-selected checkpoints. |
| `local_directory`, `local_every` | A path on every host's own disk where `fit` writes one more checkpoint, the latest, every `local_every` steps. Both or neither. |

| Member | Meaning |
|---|---|
| `latest` | The newest step every process can read, local or persistent. |
| `best` | The best committed step for the first evaluation tracker, or None if none is ranked. |
| `path(step)` | The directory of one step. |
| `restore(template, step=None)` | Read the latest full state, a numbered step, `"best"`, or `"best:<metric key>"` into the structure of `template`. |
| `stored(step=None)` | The shapes and dtypes each saved state field holds, without reading values. |
| `wait()` | Block until asynchronous saves have finished. |
| `kept()` | Committed steps with their metrics, ranking rules and state/weights kind. |
| `profile_steps()`, `profile_metadata(step)`, `restore_profiles(step)` | The post-hoc EMA snapshots (below). |

Saves are asynchronous; `wait()` returns once they are durable. Constructing `Checkpoints` opens nothing; the Orbax managers are created on first use.

## Best weights and early stopping

Evaluation decides what “best” means; the checkpointer decides where those weights stay. `fit` takes the metric objects it evaluates, so a selector names a producer, not an unrelated log string. The metric's `Shown(better="lower" | "higher")` supplies its direction. A missing direction is refused unless `Best(..., mode="min" | "max")` supplies one.

```python
from dew import Best, Checkpoints, Plateau, Trainer
from dew.eval import FID, CLIPScore

fid = FID()
clip = CLIPScore()
trainer = Trainer(objective, optimizer, key=key,
                  checkpoints=Checkpoints(run_directory, keep=2))
state = trainer.fit(data, steps=100_000, metrics=[fid, clip],
                    eval_every=1_000, checkpoint_every=5_000,
                    best=fid, stop=Plateau(fid, evals=5, min_delta=0.01))
```

With no selector, a run with validation ranks by the objective's validation loss; a run without validation ranks by its training loss. A validation pass always refers to the checkpoint's own state, including its EMA overlay when evaluation uses averaged weights. Evaluation runs at the checkpoint cadence too, so a save never inherits a score from earlier weights. The default objective loss adds a forward pass when validation metrics do not already report loss. Its statistics-only program is compiled once per objective and reused by mesh and batch shape; it does not compile or run an optimizer gradient. An explicit metric selector uses the existing metric pass and avoids that additional forward pass. An evaluation between checkpoint steps writes a state only if its score enters a best-K set. With `checkpoint_every=None` and no explicit `best`, only the final state is saved, as before; an explicit best policy can instead save only its winners. A regular checkpoint that also enters best is written once.

Each save records all metrics from that evaluation and the training loss under their names, such as `val/fid` and `train/loss`. Ranking values, directions and top-K limits have separate metadata. A save without a selected score is unranked, not ranked by a different loss. In particular, an emergency preemption save does not invent a validation score. Old checkpoints' `loss` key keeps its historical training-loss meaning.

Several trackers retain the union of their winners. A callable reads metrics by object, handles unhashable metric objects, and minimizes its result unless `mode="max"` is supplied. `threshold` admits only scores strictly better than that value in the selected direction.

```python
# Three best FID steps and two best CLIP-score steps, plus the latest two states.
trainer.fit(data, steps=100_000, metrics=[fid, clip], eval_every=1_000,
            checkpoint_every=5_000,
            best=[Best(fid, top=3), Best(clip, top=2)])

# An aggregate of metrics evaluated on the same weights.
trainer.fit(data, steps=100_000, metrics=[fid, clip], eval_every=1_000,
            best=Best(lambda m: m[fid] - 0.1 * m[clip], threshold=20.0))
```

The supplied metrics run together. If a callable needs a score absent from that evaluation, its tracker leaves the step unranked: it never fills a missing value from a previous evaluation. For training reports, `best=objective.loss` selects the objective's training loss; `objective.scalars.ce` selects a value the objective declares in `shown`. Aggregates can index those objects too. Strings are for recorded configurations, reader selectors and log keys, not for selecting a declared evaluation metric in code.

`Plateau` counts eligible evaluations, not training steps. An improvement must exceed `min_delta`, an absolute amount in the selected direction. Missing scores do not advance patience; nonfinite objective validation losses are errors. Patience and its policy are checkpoint metadata, so resuming with the same policy continues the same count. A changed policy is refused. Stopping writes a final full checkpoint and reports “Stopped: validation plateau”; its control metadata includes the reason. Nothing is added to `TrainState`.

The existing readers accept best selectors; `"best"` chooses the first tracker and `"best:val/clip_score"` chooses the named tracker. An aggregate is named by its position, such as `"best:aggregate:0"`.

```python
restored, position = trainer.checkpoints.restore(template, step="best")
image_model = TextToImage.from_run(run_directory, step="best:val/clip_score")
for checkpoint in trainer.checkpoints.kept():
    print(checkpoint.step, checkpoint.kind, checkpoint.metrics, checkpoint.rankings)
```

`restore_best=True` on `fit` returns the best *whole* training state, including its earlier optimizer, counters and RNG, not earlier weights paired with the final optimizer. It does not change the default return value, which is the final state. `Best(fid, weights_only=True)` writes only parameters and EMA for an off-cadence winner. Such a snapshot is inference-only: full-state restore and `restore_best=True` refuse it. When a winner falls on a full-checkpoint step, the one full save also serves the best tracker; there is no duplicate write. Latest-N retention counts full states so a small inference snapshot cannot replace the state a run resumes from.

Several validation splits use explicit reader names; a metric selector needs its split when more than one is present. A callable uses `(split, metric)` keys.

```python
trainer.fit(data, steps=100_000, metrics=[fid, clip], eval_every=1_000,
            validation={"flowers": flowers.val, "faces": faces.val},
            best=Best(fid, split="flowers"))
# A cross-split aggregate: Best(lambda m: m["flowers", fid] + m["faces", fid])
```

Recorded runs use `TrainerConfig.best`, a metric name or `Best("fid", top=3, mode="min")`; several selectors can be a tuple of named `Best` policies. A callable cannot be a recorded selector and raises a clear error.

## Time cadence and retention

A checkpoint cadence is steps or a `timedelta`, through the same parameter. A duration saves at the next safe training step after it elapses; processes agree on that step. Recorded configurations use a duration such as `checkpoint_every="30m"` (`s`, `m` and `h` are supported).

```python
from datetime import timedelta
from dew import Keep

checkpoints = Checkpoints(run_directory, keep=Keep(
    latest=2, every=10_000, interval=timedelta(hours=1),
    where=lambda c: c.metrics[fid] < 20.0))
trainer = Trainer(objective, optimizer, key=key, checkpoints=checkpoints)
trainer.fit(data, steps=100_000, checkpoint_every=timedelta(minutes=15))
```

`Keep.every` retains steps divisible by that period. `Keep.interval` retains checkpoints at least that much wall time apart, starting with the oldest saved step; it does **not** mean “keep every checkpoint older than this age.” `Keep.where` is a code-only predicate over a checkpoint's step and metrics. Records accept metric objects as keys as well as stored names; with several splits, use `c.metrics["flowers", fid]`. A missing score leaves that predicate unselected. These policies, latest full states, all best-K sets and post-hoc snapshots are unioned. Filenames remain step numbers; scores live in metadata and `kept()`, not in names that readers must parse.

Ranking reads completed evaluation scalars on the host and adds no parameter transfer. The same evaluation is shared by all trackers, and entering several best-K sets still requests one save. Weights-only winners omit the optimizer, clocks and data position rather than compressing a full training checkpoint and calling it small.

## The EMA copy on disk

Orbax compresses every array of a checkpoint with zstd, which in the two runs below saves 6 to 9% on weights and Adam moments: past the sign and exponent, their bits look random. An EMA copy is different, because it agrees with the weights it follows in its sign, exponent and leading mantissa bits. `Checkpoints` stores each EMA leaf that has its weight's floating dtype and shape as the XOR of the two, split into byte planes: a leading `uint8` axis with one plane per byte, least significant first. The XOR is zero wherever the two agree, and the planes gather those zeros into long runs, which zstd compresses well. The step's metadata lists the leaves stored this way. `restore` and `stored` return them bit for bit, as the run trained them, in the dtype and placement the template asks for and on any mesh. Checkpoints written before this change list no such leaves and restore as they were written.

| Run | EMA, zstd alone | XOR, then zstd | XOR and byte planes, then zstd |
|---|---|---|---|
| 176M-parameter DiT, fp32, step 1.35M | 652.7 MB | 560.3 MB | 488.6 MB (−25%) |
| 8.7M-parameter DiT, fp32, step 750, decay 0.999 | 32.2 MB | 30.8 MB | 28.2 MB (−12%) |

How much the XOR saves depends on how close a run's EMA sits to its weights, which follows its updates and its decay; the table measures two runs. Written through `Checkpoints` with JAX's CPU backend, the 176M model's weights and EMA take 1139.4 MB instead of 1306.0 MB. In three processes of three saves and restores each, alternating with processes running the code before this change, a compiled save blocked the step for 0.19 to 0.46 s instead of 0.04 to 0.14 s, the time to compute the planes, and landed on disk in 1.9 to 3.7 s instead of 1.8 to 2.3 s, apart from saves that took 17 to 74 s either way while the shared disk was busy. Restores took 1.45 to 1.59 s instead of 1.31 to 1.71 s. The first save and the first restore of a process each compile one small program per distinct EMA leaf shape (19 for this model): there, a save blocked for 1.0 to 2.7 s and a restore took 2.0 to 3.0 s.

On an NVIDIA A100 at integration commit `845b75bc`, with JAX 0.11.2, the same checkpoint's weights and EMA (175.6M fp32 parameters, step 1.35M) gave the following timings. The variants were interleaved in one process for five saves and restores each; the table gives medians of attempts 1–4, excluding the first calls.

| EMA storage | Weights and EMA | Save blocks the step | Save becomes durable | Restore |
|---|---|---|---|---|
| EMA as itself, zstd | 1306.0 MB | 0.49 s | 5.78 s | 2.35 s |
| XOR byte planes, zstd | 1139.4 MB | 0.59 s | 5.28 s | 2.53 s |

Both variants restored the EMA bit for bit. These are checkpoint timings, not training-step timings.

Planes are computed on the devices that hold the EMA. So a save holds one more EMA-sized buffer there until Orbax has copied it to the host, which fits in memory that the step's gradient frees between steps. EMA leaves in pinned host memory, as a host layout keeps them, are written as themselves: Orbax writes them from their own buffers, and differencing them would keep a second copy in the memory the layout exists to spare. EMA leaves whose dtype differs from their weight's are also stored as themselves.

## Post-hoc EMA

An EMA's length is usually picked before training and judged after it. Post-hoc EMA (Karras et al. 2024, [arXiv:2312.02696](https://arxiv.org/abs/2312.02696)) picks it afterwards. `OptimConfig.ema_profiles`, such as `(0.05, 0.10)`, wraps the optimizer in `dew.training.optim.power_profiles`, which keeps one power-function EMA of the weights per relative standard deviation (the paper's σ_rel) in the optimizer state. They are sharded, placed and saved with that state. A snapshot is the ordinary checkpoint itself, retained by Orbax's public preservation policy even after `keep` would prune it. There is one atomic save, not a separate write of the averages: an interrupted save cannot publish a state without its profiles.

```python
from dew.training.posthoc import reconstruct

averaged = reconstruct(run_directory, 0.07)             # at the latest snapshot
averaged = reconstruct(run_directory, 0.07, step=40000)
```

`reconstruct` returns the `params` collection as host arrays, in the weights' structure and dtypes. It weights every snapshot up to the step by the least-squares solve of the paper's Algorithm 3 (`dew.training.posthoc.coefficients`), which gives the same weights as NVlabs' `phema.py` to the last bit on the paper's setting, and reads one snapshot at a time. Its accuracy depends on how many snapshots there are: in a 96-step CPU run tracking 0.05 and 0.10, an average of 0.07 rebuilt from snapshots every 4 steps differed from one tracked directly by at most 1.2e-7 (weights of scale 2), every 8 steps by 4.8e-7, every 16 steps by 5.7e-6; the tracked 0.05 average was 1.3e-4 from it.

Each profile holds one more copy of the weights in memory. Retaining the whole checkpoint costs disk. With fp32 weights, an ordinary EMA, fp32 Adam moments, two power profiles and no accumulation buffers, a 176M model holds six weight-sized trees per snapshot: about 4.2 GB before compression. A 1.35M-update run saving every 10k updates keeps 135 snapshots, about 570 GB raw. Keeping only the two averages would take about 190 GB, but the separate-manager design added blocking setup and a gap between commits; the one-checkpoint design keeps the state and profiles atomic. Empty `ema_profiles` retains the ordinary `keep` policy.

In paired, interleaved CPU saves of an 8.4M-parameter fp32 tree, the same state saved with and without snapshot retention blocked for median 14.9 and 15.2 ms respectively (six warmed pairs). No extra parameter bytes are transferred or serialized for a snapshot. On an NVIDIA A100 at integration commit `845b75bc`, with JAX 0.11.2 and 175.6M fp32 parameters, snapshot retention likewise added no measured blocking cost. The full state included weights, EMA, Adam moments and two power profiles. The variants were interleaved for five saves each; these are medians of attempts 1–4. Restore was not timed for this comparison.

| Retention | Full checkpoint | Save blocks the step | Save becomes durable |
|---|---|---|---|
| Inline profiles, no snapshot retention | 3762.1 MB | 1.41 s | 14.80 s |
| Checkpoint retained as a snapshot | 3762.1 MB | 1.38 s | 14.71 s |

Both variants wrote the same bytes. Snapshot retention changes which steps stay on disk, not what each save transfers.

## Preemption

A scheduler stops a job with SIGTERM and kills it a grace period later: Slurm's `KillWait`, Kubernetes' termination grace period, a spot VM's notice. `Trainer.fit` stops at the next step every process agrees on (JAX's `reached_preemption_sync_point` in a pool, the signal itself in a lone process), writes that step's state and data position, skips the final validation, and raises `dew.training.Preempted`. Uncaught, it ends the program with exit status 143, SIGTERM's, so the scheduler sees a stopped job rather than a finished one; a Kubernetes pod failure policy can ignore that code. The same command run again resumes from the checkpoint. On a GPU, `--xla_gpu_deterministic_ops=true` orders the reductions, so a resumed run can match the uninterrupted one to the bit. XLA still picks GEMM and convolution kernels at compile time by live timing, and a resumed process compiles again, so it can pick kernels that round differently ([XLA's determinism notes](https://openxla.org/xla/determinism)). `--xla_gpu_autotune_level=0` removes that choice, at a cost. On an RTX 4080 the 176M hybrid DiT's step went from 139 to 151 ms (8%), and a 67M decoder's from 79.8 to 81.5 ms (2%). Whether the choice actually differs in practice is unconfirmed. Two runs of a CIFAR-10 LADD distillation under deterministic ops alone ended a step apart from an uninterrupted run, by 3.5e-14 in the patch convolution's gradient, which Adam carried to 8e-6. But 20 fresh processes training a DiT under deterministic ops alone all agreed to the bit, as did 20 with autotuning off. Dew's CUDA test lane sets both flags as a precaution.

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
