# Checkpoints

Use `dew.Checkpoints(directory)` to manage one run's checkpoints. Pass it to `Trainer` to save training state and the data iterator's position every `checkpoint_every` steps and at the end of `fit`. A later `fit` with the same directory resumes from the newest step.

A training checkpoint stores model variables, optimizer state and three counters: attempted batches, accepted microbatches and optimizer updates. It also stores the root key, the EMA copy when the objective keeps one, loss-scaler history and any partly filled gradient-accumulation window.

## Example

Train for five steps, restore into a new `Trainer`, then continue to step ten. The iterator exposes its batch index as bytes through `get_state` and `set_state`, so the checkpoint can save its position.

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

`steps` is the target step count. The second `fit(..., steps=10)` starts at step five and runs five more steps. `place()` returns the restored `TrainState`, its placement and the saved iterator position. Prefetching may read ahead, but checkpoints record the last batch consumed by the loop. This example uses a temporary directory. For a persistent run, choose a directory and pass it again on later calls.

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

Saves run asynchronously. Call `wait()` to wait until they are durable. Constructing `Checkpoints` opens nothing; it creates Orbax managers on first use.

## Best weights and early stopping

Select the best checkpoint using a metric object passed to `fit`. The selector must refer to a metric that evaluation produces. Its `Shown(better="lower" | "higher")` specifies which direction is better. If that direction is missing, supply `Best(..., mode="min" | "max")`; otherwise, the selector is rejected. The checkpointer retains the selected weights.

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

Without a selector, a run with validation ranks checkpoints by the objective's validation loss. Without validation, it uses training loss. Validation always scores the checkpoint's state, including its EMA overlay when evaluation uses averaged weights. Evaluation also runs at checkpoint intervals, so a save cannot inherit a score from earlier weights.

If validation metrics do not already report loss, default ranking adds an objective forward pass. This statistics-only program compiles once per objective and is reused by mesh and batch shape. It does not compile or run an optimizer gradient. An explicit metric selector uses the existing metric pass and avoids the extra forward pass.

Between checkpoint steps, evaluation saves state only when its score enters a best-K set. With `checkpoint_every=None` and no explicit `best`, the run saves only its final state. An explicit best policy can save only its winners. A regular checkpoint that also enters a best set is written once.

Each save records all metrics from that evaluation and the training loss under names such as `val/fid` and `train/loss`. Ranking values, directions and top-K limits have separate metadata. A save without a selected score stays unranked. It does not substitute another loss. Emergency preemption saves, for example, have no invented validation score. In old checkpoints, `loss` still means training loss.

With several trackers, the run retains all their winners. A callable reads metrics by object and handles unhashable metric objects. Its result is minimized unless you set `mode="max"`. With `threshold`, only scores strictly better than that value in the selected direction qualify.

```python
# Three best FID steps and two best CLIP-score steps, plus the latest two states.
trainer.fit(data, steps=100_000, metrics=[fid, clip], eval_every=1_000,
            checkpoint_every=5_000,
            best=[Best(fid, top=3), Best(clip, top=2)])

# An aggregate of metrics evaluated on the same weights.
trainer.fit(data, steps=100_000, metrics=[fid, clip], eval_every=1_000,
            best=Best(lambda m: m[fid] - 0.1 * m[clip], threshold=20.0))
```

The supplied metrics run together. If a callable needs a score missing from that evaluation, its tracker leaves the step unranked. It never substitutes a value from an earlier evaluation. For training reports, `best=objective.loss` selects the objective's training loss. `objective.scalars.ce` selects a value declared in `shown`. Aggregates can index those objects too. Use metric objects to select declared evaluation metrics in code. Strings are for recorded configurations, reader selectors and log keys.

`Plateau` counts eligible evaluations. An improvement must exceed `min_delta`, an absolute amount in the selected direction. Missing scores do not advance patience. Non-finite objective validation losses are errors.

Checkpoints store patience and its policy as metadata, so resuming with the same policy continues the count. A changed policy is rejected. Stopping writes a final full checkpoint and reports "Stopped: validation plateau". The control metadata records the reason, with nothing added to `TrainState`.

Readers accept best selectors. `"best"` chooses the first tracker; `"best:val/clip_score"` chooses the named tracker. An aggregate uses its position as its name, such as `"best:aggregate:0"`.

```python
restored, position = trainer.checkpoints.restore(template, step="best")
image_model = TextToImage.from_run(run_directory, step="best:val/clip_score")
for checkpoint in trainer.checkpoints.kept():
    print(checkpoint.step, checkpoint.kind, checkpoint.metrics, checkpoint.rankings)
```

With `restore_best=True`, `fit` returns the complete training state from the best step, including that step's optimizer, counters and RNG. By default, it returns the final state.

`Best(fid, weights_only=True)` saves only parameters and EMA when a winner falls between regular checkpoint steps. These snapshots are inference-only. Full-state restore and `restore_best=True` reject them. On a regular checkpoint step, a winning full save also serves the best tracker without a second write. Latest-N retention counts full states, so an inference snapshot cannot replace the state needed to resume.

Give each validation split an explicit reader name. With several splits, a metric selector must name its split. A callable uses `(split, metric)` keys.

```python
trainer.fit(data, steps=100_000, metrics=[fid, clip], eval_every=1_000,
            validation={"flowers": flowers.val, "faces": faces.val},
            best=Best(fid, split="flowers"))
# A cross-split aggregate: Best(lambda m: m["flowers", fid] + m["faces", fid])
```

For recorded runs, set `TrainerConfig.best` to a metric name or a policy such as `Best("fid", top=3, mode="min")`. For several selectors, use a tuple of named `Best` policies. Recording a callable selector raises an error.

## Time cadence and retention

Set the checkpoint interval in steps or as a `timedelta`, using the same parameter. With a duration, saving occurs at the next safe training step after the interval elapses. Processes agree on that step. In recorded configurations, use a duration such as `checkpoint_every="30m"`; `s`, `m` and `h` are supported.

```python
from datetime import timedelta
from dew import Keep

checkpoints = Checkpoints(run_directory, keep=Keep(
    latest=2, every=10_000, interval=timedelta(hours=1),
    where=lambda c: c.metrics[fid] < 20.0))
trainer = Trainer(objective, optimizer, key=key, checkpoints=checkpoints)
trainer.fit(data, steps=100_000, checkpoint_every=timedelta(minutes=15))
```

`Keep.every` retains steps divisible by its period. `Keep.interval` retains checkpoints spaced by at least that much wall time, starting with the oldest saved step. It specifies spacing, not a cutoff age. `Keep.where` is a code-only predicate over a checkpoint's step and metrics. Metric records accept object keys or stored names. With several splits, use `c.metrics["flowers", fid]`. A missing score leaves the predicate unselected.

Retention combines these policies, latest full states, all best-K sets and post-hoc snapshots. Filenames stay as step numbers. Scores are in metadata and `kept()`, so readers do not need to parse them from filenames.

Ranking reads completed evaluation scalars on the host without transferring parameters. All trackers use the same evaluation. A step that enters several best-K sets still requests one save. Weights-only winners save parameters and EMA without the optimizer, clocks or data position; they do not compress a full training checkpoint.

## The EMA copy on disk

Orbax compresses every checkpoint array with zstd. In the two runs below, that saves 6 to 9% on weights and Adam moments. Their bits beyond the sign and exponent look random and compress poorly.

An EMA often shares its weights' sign, exponent and leading mantissa bits. For an EMA leaf with the same floating dtype and shape as its weight, `Checkpoints` stores their XOR. It splits the XOR into one plane per byte on a leading `uint8` axis, least significant byte first. Matching bits become zeros. Byte planes group those zeros into long runs that zstd compresses well.

Step metadata lists leaves stored this way. `restore` and `stored` return them bit for bit in the requested template dtype and placement, on any mesh. Older checkpoints list no such leaves and restore as originally written.

| Run | EMA, zstd alone | XOR, then zstd | XOR and byte planes, then zstd |
|---|---|---|---|
| 176M-parameter DiT, fp32, step 1.35M | 652.7 MB | 560.3 MB | 488.6 MB (−25%) |
| 8.7M-parameter DiT, fp32, step 750, decay 0.999 | 32.2 MB | 30.8 MB | 28.2 MB (−12%) |

XOR savings depend on how closely EMA values follow the weights, which depends on the updates and decay. The table measures two runs. Through `Checkpoints` on JAX's CPU backend, the 176M model's weights and EMA occupy 1139.4 MB, down from 1306.0 MB.

I measured three processes with three saves and restores each, alternating with processes using the earlier zstd-only storage. Once compiled, a save blocked the step for 0.19 to 0.46 s with XOR byte planes, compared with 0.04 to 0.14 s with zstd alone. That extra time computes the planes. Saves reached disk in 1.9 to 3.7 s, compared with 1.8 to 2.3 s. When the shared disk was busy, either variant took 17 to 74 s. Restores took 1.45 to 1.59 s with planes and 1.31 to 1.71 s with zstd alone.

The first save and restore each compile a small program for every distinct EMA leaf shape, 19 for this model. Those first saves blocked for 1.0 to 2.7 s; first restores took 2.0 to 3.0 s.

On an NVIDIA A100 at integration commit `845b75bc` with JAX 0.11.2, I timed the same checkpoint's weights and EMA: 175.6M fp32 parameters at step 1.35M. The variants were interleaved in one process for five saves and restores each. The table gives medians of attempts 1 to 4, excluding the first calls.

| EMA storage | Weights and EMA | Save blocks the step | Save becomes durable | Restore |
|---|---|---|---|---|
| EMA as itself, zstd | 1306.0 MB | 0.49 s | 5.78 s | 2.35 s |
| XOR byte planes, zstd | 1139.4 MB | 0.59 s | 5.28 s | 2.53 s |

Both variants restored the EMA bit for bit. The timings measure checkpoint operations only; they do not measure training steps.

The devices holding the EMA compute its planes. Saving needs one extra EMA-sized buffer there until Orbax copies it to the host. That buffer fits in the memory freed by the step's gradients between steps.

With a host layout, EMA leaves in pinned host memory are stored directly. Orbax writes from their buffers, avoiding a second copy in the host memory the layout is meant to save. EMA leaves with a different dtype from their weights are also stored directly.

## Post-hoc EMA

You usually choose an EMA length before training and assess it afterwards. Post-hoc EMA (Karras et al. 2024, [arXiv:2312.02696](https://arxiv.org/abs/2312.02696)) lets you choose it after training.

Set `OptimConfig.ema_profiles`, for example to `(0.05, 0.10)`, to wrap the optimizer in `dew.training.optim.power_profiles`. It keeps one power-function EMA per relative standard deviation, the paper's σ_rel, in optimizer state. These profiles are sharded, placed and saved with that state.

A snapshot is an ordinary checkpoint retained through Orbax's public preservation policy, even when `keep` would prune it. State and profiles share one atomic save. An interrupted save cannot publish state without its profiles.

```python
from dew import Checkpoints

checkpoints = Checkpoints(run_directory)
averaged = checkpoints.posthoc_ema(0.07)               # at the latest snapshot
averaged = checkpoints.posthoc_ema(0.07, step=40000)
```

`posthoc_ema` returns the `params` collection as host arrays with the weights' structure and dtypes. It reads one snapshot at a time and weights every snapshot up to the selected step. The weights come from the least-squares solve in the paper's Algorithm 3, implemented by `dew.training.posthoc.coefficients`. On the paper's setting, they match NVlabs' `phema.py` bit for bit.

Accuracy depends on snapshot count. In a 96-step CPU run tracking 0.05 and 0.10, I rebuilt an average of 0.07 and compared it with one tracked directly. With snapshots every 4 steps, the largest difference was 1.2e-7 for weights of scale 2. Every 8 steps gave 4.8e-7; every 16 gave 5.7e-6. The tracked 0.05 average differed from the directly tracked 0.07 by 1.3e-4.

Each profile adds a copy of the weights in memory. Retaining full checkpoints also costs disk. A 176M model with fp32 weights, an ordinary EMA, fp32 Adam moments and two power profiles uses six weight-sized trees per snapshot, excluding accumulation buffers. That is about 4.2 GB before compression. A 1.35M-update run saving every 10k updates keeps 135 snapshots, about 570 GB raw.

Keeping only the two averages would use about 190 GB. However, a separate manager added blocking setup and a gap between commits. Saving one checkpoint keeps state and profiles atomic. With empty `ema_profiles`, the ordinary `keep` policy applies.

For an 8.4M-parameter fp32 tree on CPU, I interleaved six warmed pairs of saves of the same state. Median blocking time was 14.9 ms with snapshot retention and 15.2 ms without it. A snapshot transfers and serializes no extra parameter bytes.

On an NVIDIA A100 at integration commit `845b75bc`, with JAX 0.11.2 and 175.6M fp32 parameters, snapshot retention also added no measured blocking cost. Full state included weights, EMA, Adam moments and two power profiles. The variants were interleaved for five saves each. These are medians of attempts 1 to 4. I did not time restore for this comparison.

| Retention | Full checkpoint | Save blocks the step | Save becomes durable |
|---|---|---|---|
| Inline profiles, no snapshot retention | 3762.1 MB | 1.41 s | 14.80 s |
| Checkpoint retained as a snapshot | 3762.1 MB | 1.38 s | 14.71 s |

Both variants wrote the same bytes. Snapshot retention determines which steps stay on disk. Each save transfers the same data.

## Preemption

Schedulers send SIGTERM, then kill the job after a grace period. Examples include Slurm's `KillWait`, Kubernetes' termination grace period and a spot VM's notice. `Trainer.fit` stops at the next step every process agrees on. In a process pool, it uses JAX's `reached_preemption_sync_point`; in a single process, it uses the signal itself. It saves that step's state and data position, skips final validation and raises `dew.training.Preempted`.

Uncaught, `Preempted` exits with status 143, the code for SIGTERM. The scheduler then records a stopped job instead of a completed one. A Kubernetes pod failure policy can ignore that code. Run the same command again to resume from the checkpoint.

On GPU, `--xla_gpu_deterministic_ops=true` orders reductions so a resumed run can match an uninterrupted run bit for bit. XLA still chooses GEMM and convolution kernels by timing them during compilation. A resumed process compiles again and may choose kernels that round differently; see [XLA's determinism notes](https://openxla.org/xla/determinism). Setting `--xla_gpu_autotune_level=0` removes that choice but slows steps. On an RTX 4080, the 176M hybrid DiT's step rose from 139 to 151 ms (8%). A 67M decoder's step rose from 79.8 to 81.5 ms (2%).

I have not confirmed whether different autotuner choices cause different rounding in practice. Two CIFAR-10 LADD distillation runs with deterministic ops alone differed at a step from an uninterrupted run. The patch convolution's gradient differed by 3.5e-14, which Adam amplified to 8e-6. However, 20 fresh processes training a DiT with deterministic ops alone matched bit for bit, as did 20 with autotuning off. Dew's CUDA test lane sets both flags as a precaution.

On CPUs with performance and efficiency cores, such as Intel's hybrid parts, XLA's float32 convolution has two rounding behaviors. The core type that runs the first convolution chooses the behavior for that process; see [openxla/xla#50022](https://github.com/openxla/xla/issues/50022). A resumed process may use the other behavior. Pin the run to one core type with `taskset` to remove that choice.

The scheduler's grace period must cover one step and one checkpoint write. After a stop request, `dew launch` gives the pool 300 seconds before killing its ranks. A second signal kills them immediately; see [Training on several nodes](multi-node.md).

## Data position

A plain Python generator usually cannot report its position. To resume the data sequence, use a built-in source with checkpoint support or implement `get_state` and `set_state`, as above. Rebuilding an iterator from the start may replay records even when weights restore correctly. For a stream that cannot save its position, set `checkpoint_every=None`. Such a stream cannot produce a resumable checkpoint, but the final state is still written.

A persistent checkpoint can restore into a different, compatible placement through the trainer's restore template. Whether the process count can change depends on the saved position:

- A global record count can be read at any process count. Every record dataset built on `train_stream` saves this kind of position.
- One share's own offset, which custom iterators like the one above report, can only be read by a reader of the same share (`DataPartition`), whichever processes those are. Restore refuses any other share and names the shares the checkpoint holds.
- A checkpoint without a position can change the process count when the tensor layout is compatible.

Local checkpoints hold only the shards each process had, so they restore onto the placement they were written with. See [Training data](../concepts/data.md) for how datasets save their position.

## Several hosts

Each process writes its shards of a persistent checkpoint. Process 0 writes metadata and commits the step. Restoring reads shards written by other processes, so every process needs read and write access to the same directory. Use a shared filesystem or bucket. With separate disks, process 0 would commit without the other shards, and other processes would see no step.

On a pool's first use of `Checkpoints`, process 0 writes a file and every process checks that it can see it. If any process cannot, all processes raise an error listing those that failed. Buckets skip this check. You can also keep local checkpoints on each host's disk alongside the shared checkpoints.

[Distributed training](../concepts/distributed.md) covers topology requirements and [Cloud TPUs](../tpu.md) remote setup. A local save and restore does not show that cross-host recovery or remote storage work.

## What a checkpoint does not save

`Checkpoints` does not write `run.json`. Recipes save configuration separately through `RunConfig.save` or `RunConfig.train`. `RunConfig.load` can read records from older Dew versions. A missing field takes its default, matching runs recorded before that field existed. Unknown fields, such as those written by a newer version, are rejected.

Checkpoints do not save source code, package versions, tokenizer files, dataset revisions or external-service state. Record these in the experiment metadata.

## Limits

Resume with the same model, optimizer, accumulation length and scaler configuration. The trainer rejects a different accumulation length. If the loss scaler rejects a step's gradients, the attempt still counts. Previously accepted accumulation records, optimizer state, EMA and mutable contributions remain unchanged. Restoring preserves the scaler's finite-step streak and scale, including a partly filled window. If the checkpoint has already reached the target step, resuming reads no data and runs no evaluation, compilation or save.

Deterministic CPU tests cover checkpoints taken in the middle of a window and after rejected attempts, including composite replay. They do not cover cross-host recovery on GPU or TPU, or replaying the side effects of external rollouts. The per-fit counter that stops a run after repeated non-finite losses is not checkpointed. A training checkpoint without the step counters and accumulation fields cannot resume through this interface, but its parameters can still be loaded alone.
