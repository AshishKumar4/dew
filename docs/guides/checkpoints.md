# Checkpoints

`dew.Checkpoints(directory)` manages the checkpoints of one run. When you pass it to a `Trainer`, `fit` saves the training state and the data iterator's position every `checkpoint_every` steps and at the end of the run. A later `fit` with the same directory resumes from the newest step.

A training checkpoint holds the model variables, the optimizer state and three step counters: attempted batches, accepted microbatches and optimizer updates. It also holds the root key, the EMA copy when the objective keeps one, the loss scaler's history, and any half-filled gradient accumulation window.

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
from dew.objectives.base import Aux, Objective


class Regression(Objective):
    def __init__(self):
        self.model = nn.Dense(features=1)

    def init(self, key, variables=None):
        return self.model.init(key, jnp.zeros((1, 1), jnp.float32))

    def loss(self, variables, batch, step):
        prediction = self.model.apply(variables, batch["x"])
        errors = (prediction - batch["y"]) ** 2
        return self.row_mean(errors, batch), Aux(metrics={})


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

`steps` is the step to finish at, not a number of steps to add, so the second `fit(..., steps=10)` starts at step five and runs five more. `place()` returns the restored `TrainState`, its placement and the saved iterator position. Prefetching may read batches ahead of the loop, but the saved position is the last batch the loop consumed. The example writes into a temporary directory. For a run you want to keep, give it a directory of its own and pass that directory again on later calls.

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
| `kept()` | Committed steps with their metrics, ranking rules and kind: state, weights or tree. |
| `profile_steps()`, `profile_metadata(step)`, `restore_profiles(step)` | The post-hoc EMA snapshots (below). |

Saves are asynchronous, and `wait()` returns once they are durable. Constructing `Checkpoints` opens nothing, because the Orbax managers are created on first use.

A state that is not a training run's, such as a simulation's, has no optimizer, average or loss scale. `save(step, state)` takes it as a mapping of arrays in place of a `TrainState` and writes it as it is, with `metrics`, `ranking` and `control` as a train state's are. `restore(template)` reads it back through a mapping template of `jax.ShapeDtypeStruct` leaves, placed on their shardings, or as host arrays with no template. `kept()` reports its kind as `tree`, and a train-state template is refused for it.

```python
checkpoints.save(step, {"state": state, "key": key}, metrics={"rate": rate})
state_tree, _ = checkpoints.restore({"state": state_shapes, "key": key_shape})
```

## Weights no step moves

Every variables collection except `params` goes to `frozen/<digest>` beside the step directories, once per distinct content. That includes a LoRA run's `frozen` base, a pipeline's text towers and autoencoder, `constants` and batch statistics, and the same collections of the EMA copy. Each step records the digest of each collection. A LoRA fine-tune of Stable Diffusion 1.5 writes its 2.1 GB base once instead of in every checkpoint, and each step holds the 13 MB of factors plus their optimizer state (sizes from a LoRA rank-16 run). A collection that changes, such as batch statistics, gets a new entry each time it changes. The next save deletes any entry that no kept step records, so pruning a step never removes weights a kept step still needs.

The digest covers the bytes as stored. A restore checks each entry against the step and refuses one whose content changed, whatever dtypes JAX allows at the time. Each step names its entries relative to the run directory, so a run directory copied elsewhere restores as it is. `dew.io.publish` uploads or references a step together with its `run.json` and the entries the step records, in the same layout.

## Best weights and early stopping

Evaluation produces the scores that decide which step is best, and the checkpointer keeps those steps on disk. To choose the score, pass `best` the object that produces it, usually one of the metric objects `fit` evaluates. The metric's `Shown(better="lower" | "higher")` gives the direction. A metric without a direction is refused unless `Best(..., mode="min" | "max")` gives one.

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

With no selector, a run with validation ranks checkpoints by the objective's validation loss, and a run without validation ranks them by its training loss. A validation pass always scores the checkpoint's own state, including its EMA overlay when evaluation uses averaged weights. Evaluation also runs at the checkpoint cadence, so a save never inherits a score from earlier weights.

When the validation metrics do not already report a loss, ranking by the default objective loss adds a forward pass. That pass computes only loss statistics. Its program is compiled once per objective for each mesh and batch shape and then reused, and it never compiles or runs an optimizer gradient. An explicit metric selector uses the existing metric pass and avoids the extra forward pass.

An evaluation between checkpoint steps writes a state only if its score enters a best-K set. With `checkpoint_every=None` and no explicit `best`, only the final state is saved, while an explicit best policy can save only its winners. A regular checkpoint that also enters a best-K set is written once.

Each save records every metric from that evaluation and the training loss under their names, such as `val/fid` and `train/loss`. Ranking values, directions and top-K limits are kept in separate metadata. A save without a selected score stays unranked, and no other loss stands in for the missing score. For example, an emergency save on preemption gets no invented validation score. In old checkpoints, the `loss` key still means the training loss.

With several selectors, each has its own tracker, and the run keeps the union of their winners. A callable selector looks up metrics by object, works with metric objects that are not hashable, and is minimized unless you pass `mode="max"`. `threshold` admits only scores strictly better than that value in the selected direction.

```python
# Three best FID steps and two best CLIP-score steps, plus the latest two states.
trainer.fit(data, steps=100_000, metrics=[fid, clip], eval_every=1_000,
            checkpoint_every=5_000,
            best=[Best(fid, top=3), Best(clip, top=2)])

# An aggregate of metrics evaluated on the same weights.
trainer.fit(data, steps=100_000, metrics=[fid, clip], eval_every=1_000,
            best=Best(lambda m: m[fid] - 0.1 * m[clip], threshold=20.0))
```

The metrics you pass are evaluated together, on the same weights. If a callable needs a score that evaluation did not produce, its tracker leaves the step unranked and never fills the gap from a previous evaluation. To rank by training reports, `best=objective.loss` selects the objective's training loss, and `objective.scalars.ce` selects a value the objective declares in `shown`. Aggregates can index those objects too. In code, select a declared evaluation metric by its object. Strings are for recorded configurations, reader selectors and log keys.

`Plateau` stops the run after `evals` eligible evaluations without an improvement, and it counts evaluations, not training steps. An improvement must exceed `min_delta`, an absolute amount in the selected direction. Missing scores do not advance the patience count, and a non-finite objective validation loss is an error. The patience count and its policy are stored in the checkpoint metadata, so resuming with the same policy continues the same count. Resuming with a changed policy is refused. Stopping writes a final full checkpoint and reports "Stopped: validation plateau", and the checkpoint's control metadata records the reason. Nothing is added to `TrainState`.

The readers that take a step, such as `restore` and `from_run`, also accept best selectors. `"best"` chooses the first tracker's best step, and `"best:val/clip_score"` chooses the best step of the tracker for that metric. An aggregate is named by its position, such as `"best:aggregate:0"`.

```python
restored, position = trainer.checkpoints.restore(template, step="best")
image_model = TextToImage.from_run(run_directory, step="best:val/clip_score")
for checkpoint in trainer.checkpoints.kept():
    print(checkpoint.step, checkpoint.kind, checkpoint.metrics, checkpoint.rankings)
```

`restore_best=True` on `fit` returns the whole training state of the best step, with that step's optimizer state, counters and RNG, so you do not get earlier weights paired with the final optimizer. Without it, `fit` returns the final state as usual.

`Best(fid, weights_only=True)` writes only the parameters and EMA for a winner that falls between checkpoint steps. Such a snapshot is for inference only, and full-state restore and `restore_best=True` refuse it. When a winner falls on a full-checkpoint step, the one full save also serves the best tracker, so nothing is written twice. Latest-N retention counts only full states, so a small inference snapshot cannot replace the state a run resumes from.

With several validation splits, each split has an explicit reader name, given as its key in `validation`, and a metric selector has to name its split. A callable uses `(split, metric)` keys.

```python
trainer.fit(data, steps=100_000, metrics=[fid, clip], eval_every=1_000,
            validation={"flowers": flowers.val, "faces": faces.val},
            best=Best(fid, split="flowers"))
# A cross-split aggregate: Best(lambda m: m["flowers", fid] + m["faces", fid])
```

A recorded run sets `TrainerConfig.best` to a metric name or to a policy such as `Best("fid", top=3, mode="min")`. Several selectors can be a tuple of named `Best` policies. A callable cannot be recorded, and trying raises an error that says so.

## Time cadence and retention

`checkpoint_every` takes either a step count or a `timedelta`. With a duration, the save happens at the next safe training step after the duration elapses, and all processes agree on that step. Recorded configurations write the duration as a string such as `checkpoint_every="30m"`, with `s`, `m` or `h` as the unit.

```python
from datetime import timedelta
from dew import Keep

checkpoints = Checkpoints(run_directory, keep=Keep(
    latest=2, every=10_000, interval=timedelta(hours=1),
    where=lambda c: c.metrics[fid] < 20.0))
trainer = Trainer(objective, optimizer, key=key, checkpoints=checkpoints)
trainer.fit(data, steps=100_000, checkpoint_every=timedelta(minutes=15))
```

`Keep.every` retains steps divisible by that period. `Keep.interval` retains checkpoints at least that much wall time apart, starting with the oldest saved step. It sets the spacing between kept checkpoints and does not mean "keep every checkpoint older than this age". `Keep.where` is a predicate over a checkpoint's step and metrics, and it can only be set in code. Its records accept metric objects as keys as well as stored names, and with several splits you write `c.metrics["flowers", fid]`. A checkpoint missing the score is not selected by the predicate.

The checkpointer keeps the union of these policies, the latest full states, all best-K sets and the post-hoc snapshots. Directory names stay step numbers. Scores are in the metadata and in `kept()`, so a reader never has to parse them out of a name.

Ranking reads the finished evaluation scalars on the host and transfers no parameters. All trackers share the same evaluation, and a step that enters several best-K sets still requests one save. A weights-only winner is small because it leaves out the optimizer state, the clocks and the data position.

## The EMA copy on disk

Orbax compresses every array of a checkpoint with zstd. In the two runs below, that saves 6 to 9% on the weights and Adam moments, because past the sign and exponent their bits look random. An EMA copy is different. It agrees with the weights it follows in its sign, its exponent and its leading mantissa bits.

So for each EMA leaf with the same floating dtype and shape as its weight, `Checkpoints` stores the XOR of the two, split into byte planes. The planes sit on a leading `uint8` axis, one plane per byte, least significant first. The XOR is zero wherever the two values agree, and the planes gather those zeros into long runs, which zstd compresses well. The step's metadata lists the leaves stored this way. `restore` and `stored` return them bit for bit as the run trained them, in the dtype and placement the template asks for and on any mesh. Checkpoints written before Dew stored the EMA this way list no such leaves and restore as they were written.

| Run | EMA, zstd alone | XOR, then zstd | XOR and byte planes, then zstd |
|---|---|---|---|
| 176M-parameter DiT, fp32, step 1.35M | 652.7 MB | 560.3 MB | 488.6 MB (−25%) |
| 8.7M-parameter DiT, fp32, step 750, decay 0.999 | 32.2 MB | 30.8 MB | 28.2 MB (−12%) |

How much the XOR saves depends on how close a run's EMA is to its weights, and that depends on the run's updates and its decay. The table measures two runs. Written through `Checkpoints` with JAX's CPU backend, the 176M model's weights and EMA take 1139.4 MB, against 1306.0 MB with the EMA stored as itself.

I ran three processes of three saves and restores each, alternating with processes that stored the EMA as itself. After compilation, a save blocked the step for 0.19 to 0.46 s, against 0.04 to 0.14 s, and the difference is the time to compute the planes. Saves landed on disk in 1.9 to 3.7 s, against 1.8 to 2.3 s, apart from saves that took 17 to 74 s in both cases while the shared disk was busy. Restores took 1.45 to 1.59 s, against 1.31 to 1.71 s. The first save and the first restore in a process each compile one small program per distinct EMA leaf shape, 19 for this model. Those first saves blocked for 1.0 to 2.7 s, and the first restores took 2.0 to 3.0 s.

On an NVIDIA A100 at integration commit `845b75bc`, with JAX 0.11.2, I timed the same checkpoint's weights and EMA (175.6M fp32 parameters, step 1.35M). The two variants were interleaved in one process for five saves and restores each. The table gives medians of attempts 1 to 4, excluding the first calls.

| EMA storage | Weights and EMA | Save blocks the step | Save becomes durable | Restore |
|---|---|---|---|---|
| EMA as itself, zstd | 1306.0 MB | 0.49 s | 5.78 s | 2.35 s |
| XOR byte planes, zstd | 1139.4 MB | 0.59 s | 5.28 s | 2.53 s |

Both variants restored the EMA bit for bit. These are checkpoint timings, not training-step timings.

The planes are computed on the devices that hold the EMA. So a save holds one more EMA-sized buffer there until Orbax has copied it to the host, and that buffer fits in the memory the step's gradient frees between steps. A host layout keeps EMA leaves in pinned host memory, and those leaves are written as themselves. Orbax writes them from their own buffers, and computing the XOR would keep a second copy in the host memory the layout is there to save. EMA leaves whose dtype differs from their weight's are also stored as themselves.

## Post-hoc EMA

You usually pick an EMA's length before training and judge it afterwards. Post-hoc EMA (Karras et al. 2024, [arXiv:2312.02696](https://arxiv.org/abs/2312.02696)) lets you pick it after training. Setting `OptimConfig.ema_profiles`, for example to `(0.05, 0.10)`, wraps the optimizer in `dew.training.optim.power_profiles`. That keeps one power-function EMA of the weights per relative standard deviation (the paper's σ_rel) in the optimizer state, so the profiles are sharded, placed and saved with that state.

A snapshot is the ordinary checkpoint itself, which Orbax's public preservation policy keeps even after `keep` would prune it. The state and the averages go into one atomic save, so an interrupted save cannot publish a state without its profiles.

```python
from dew import Checkpoints

checkpoints = Checkpoints(run_directory)
averaged = checkpoints.posthoc_ema(0.07)               # at the latest snapshot
averaged = checkpoints.posthoc_ema(0.07, step=40000)
```

`posthoc_ema` returns the `params` collection as host arrays, in the weights' structure and dtypes. It reads one snapshot at a time and weights every snapshot up to the step by the least-squares solve of the paper's Algorithm 3 (`dew.training.posthoc.coefficients`). On the paper's setting, those weights match NVlabs' `phema.py` to the last bit.

The accuracy depends on how many snapshots there are. In a 96-step CPU run tracking 0.05 and 0.10, I rebuilt an average of 0.07 and compared it with one tracked directly, on weights of scale 2. From snapshots every 4 steps the two differed by at most 1.2e-7, from every 8 steps by 4.8e-7, and from every 16 steps by 5.7e-6. For comparison, the tracked 0.05 average was 1.3e-4 from the directly tracked 0.07.

Each profile holds one more copy of the weights in memory, and retaining whole checkpoints costs disk. With fp32 weights, an ordinary EMA, fp32 Adam moments, two power profiles and no accumulation buffers, a 176M model holds six weight-sized trees per snapshot, about 4.2 GB before compression. A 1.35M-update run saving every 10k updates keeps 135 snapshots, about 570 GB raw. Keeping only the two averages would take about 190 GB. A separate checkpoint manager for the averages added blocking setup and a gap between commits, though, so Dew keeps the state and the profiles in one atomic checkpoint. With empty `ema_profiles`, the ordinary `keep` policy applies.

In paired, interleaved CPU saves of an 8.4M-parameter fp32 tree, the same state blocked for a median of 14.9 ms with snapshot retention and 15.2 ms without it, over six warmed pairs. A snapshot transfers and serializes no extra parameter bytes. On an NVIDIA A100 at integration commit `845b75bc`, with JAX 0.11.2 and 175.6M fp32 parameters, snapshot retention also added no measured blocking cost. The full state held weights, EMA, Adam moments and two power profiles. The variants were interleaved for five saves each, and the table gives medians of attempts 1 to 4. I did not time restores for this comparison.

| Retention | Full checkpoint | Save blocks the step | Save becomes durable |
|---|---|---|---|
| Inline profiles, no snapshot retention | 3762.1 MB | 1.41 s | 14.80 s |
| Checkpoint retained as a snapshot | 3762.1 MB | 1.38 s | 14.71 s |

Both variants wrote the same bytes. Snapshot retention only changes which steps stay on disk.

## Preemption

A scheduler stops a job with SIGTERM and kills it after a grace period, such as Slurm's `KillWait`, Kubernetes' termination grace period or a spot VM's notice. `Trainer.fit` then stops at the next step every process agrees on. In a pool it finds that step with JAX's `reached_preemption_sync_point`, and in a lone process it uses the signal itself. It writes that step's state and data position, skips the final validation, and raises `dew.training.Preempted`.

If nothing catches `Preempted`, the program exits with status 143, the status for SIGTERM, so the scheduler sees a stopped job and not a finished one. A Kubernetes pod failure policy can ignore that code. Run the same command again to resume from the checkpoint.

On a GPU, `--xla_gpu_deterministic_ops=true` orders the reductions, so a resumed run can match the uninterrupted one to the bit. XLA still picks GEMM and convolution kernels at compile time by timing them, and a resumed process compiles again, so it can pick kernels that round differently ([XLA's determinism notes](https://openxla.org/xla/determinism)). `--xla_gpu_autotune_level=0` removes that choice, but steps get slower. On an RTX 4080 the 176M hybrid DiT's step went from 139 to 151 ms (8%), and a 67M decoder's from 79.8 to 81.5 ms (2%).

I have not confirmed that the autotuner's choice actually differs between processes in practice. Two runs of a CIFAR-10 LADD distillation with deterministic ops alone differed from an uninterrupted run at one step, by 3.5e-14 in the patch convolution's gradient, and Adam grew that difference to 8e-6. But 20 fresh processes training a DiT with deterministic ops alone all agreed to the bit, as did 20 with autotuning off. Dew's CUDA test lane sets both flags as a precaution.

On a CPU with performance and efficiency cores, such as Intel's hybrid parts, XLA's float32 convolution rounds one of two ways for the life of a process. The core type that ran the process's first convolution decides which ([openxla/xla#50022](https://github.com/openxla/xla/issues/50022)), so a resumed process can get the other one. Pinning the run to one core type with `taskset` removes the choice.

The grace period has to cover one step and one checkpoint write. When `dew launch` is told to stop a pool, it waits 300 seconds before it kills the ranks, and a second signal kills them at once ([Training on several nodes](multi-node.md)).

## Data position

A plain Python generator usually cannot report its position. To continue the data sequence after a resume, use a built-in source that supports checkpointing, or give the iterator both `get_state` and `set_state` as the example does. An iterator rebuilt from the start may replay records even though the model weights restore correctly. A stream that cannot save its position cannot produce a resumable checkpoint, so `fit` refuses a `checkpoint_every` for it and you have to pass `checkpoint_every=None`. The final state is still written.

A persistent checkpoint can restore into a different, compatible placement through the trainer's restore template. Whether the process count can change depends on the saved position:

- A global record count can be read at any process count. Every record dataset built on `train_stream` saves this kind of position.
- One share's own offset, which custom iterators like the one above report, can only be read by a reader of the same share (`DataPartition`), whichever processes those are. Restore refuses any other share and names the shares the checkpoint holds.
- A checkpoint without a position can change the process count when the tensor layout is compatible.

Local checkpoints hold only the shards each process had, so they restore onto the placement they were written with. See [Training data](../concepts/data.md) for how datasets save their position.

## Several hosts

Every process of a pool writes its own shards of a persistent checkpoint. Process 0 writes the metadata and commits the step, and a restore reads shards that other processes wrote. So every process must read and write the same directory, on a shared filesystem or in a bucket. If each process wrote to a disk of its own, process 0 would commit steps without the other processes' shards, and those processes would see no step at all.

The first time a pool uses `Checkpoints`, process 0 writes a file into the directory and every process checks that it can see it. If any process cannot, every process raises an error that names the ones that cannot. Buckets skip this check. Each host's own disk can still hold the local checkpoints, next to the shared directory.

[Distributed training](../concepts/distributed.md) covers topology requirements and [Cloud TPUs](../tpu.md) remote setup. A local save and restore does not show that cross-host recovery or remote storage work.

## What a checkpoint does not save

`Checkpoints` does not write `run.json`. Recipes that use `RunConfig.save` or `RunConfig.train` write the run configuration separately. `RunConfig.load` can read a record an older Dew wrote. A field the record lacks takes its default, which matches how runs behaved before the field existed. `load` refuses a field it does not know, such as one a newer Dew wrote. A checkpoint also does not save source code, package versions, tokenizer files, the dataset revision, or the state of any external service, so record those in the experiment metadata.

## Limits

Resume with the same model, optimizer, accumulation length and scaler configuration the run trained with. The trainer refuses a different accumulation length. When the loss scaler rejects a step's gradients, the attempt still counts, and the accumulation records, optimizer state, EMA and mutable contributions accepted before it stay as they were. Restoring keeps the scaler's streak of finite steps and its scale, including a partially filled window. If the checkpoint has already reached the target step, resuming reads no data and does no evaluation, compilation or saving.

Deterministic CPU tests cover checkpoints taken in the middle of a window and after rejected attempts, including composite replay. They do not cover cross-host recovery on GPU or TPU, or replaying the side effects of external rollouts. The per-fit counter that stops a run after repeated non-finite losses is not checkpointed. A training checkpoint without the step counters and accumulation fields cannot resume through this interface, but its parameters can still be loaded alone.
