# Evaluation and tracking

During `Trainer.fit`, evaluation scores a separate validation split. The objective's `evaluate` method returns artifacts that metrics read to calculate scores. A tracker receives training scalars, validation metrics, previews and run records.

To run evaluation, the dataset needs a validation split and `fit` needs `eval_every`. You must also supply `metrics` that read the objective's artifact, or `preview=True` with a tracker. Otherwise, `fit` rejects the unused evaluation pass. Passing `metrics` alone does not enable evaluation.

## Example

Train a tiny next-token decoder on synthetic token rows and measure perplexity on a different row. This example needs no tokenizer, downloaded weights or accelerator.

```python
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew import Trainer
from dew.data import Dataset
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective, Perplexity

train_tokens = np.tile(np.array([0, 1, 2, 3, 0, 1, 2, 3, 0], np.int32), (8, 1))
val_tokens = np.tile(np.array([1, 2, 3, 0, 1, 2, 3, 0, 1], np.int32), (8, 1))
data = Dataset(train=lambda partition: itertools.repeat({"text": train_tokens}),
               val=lambda partition: iter([{"text": val_tokens}]), records=8, batch=8)
model = CausalTransformer(vocab_size=4, emb_features=16, num_layers=1,
                          num_heads=2, mlp_features=32, max_seq_len=16,
                          dtype=jnp.float32, attention_impl="xla")
objective = LMObjective(model, seq_len=8)
trainer = Trainer(objective, optax.adam(0.01), key=0)
state = trainer.fit(data, steps=10, log_every=5, eval_every=5,
                    metrics=(Perplexity(),))
assert int(state.step) == 10
```

```text
Training CausalTransformer from step 0 to 10: 2,688 parameters, on 1 × cpu, batch 8, float32
step  5/10  loss 0.6666  ce 0.6666  perplexity 1.948  token_accuracy 75.0%  step_time_ms 6.443  samples_per_sec 1,242  accepted 100.0%
eval val at step 5: perplexity 2.416 (8 records in 0.34 s)
step 10/10  loss 0.2373  ce 0.2373  perplexity 1.268  token_accuracy 87.5%  step_time_ms 69.95  samples_per_sec 114.4  accepted 100.0%
eval val at step 10: perplexity 2.500 ↑ 3.5% (8 records in 0.00 s)
Trained 10 steps in 0:00:03: first step after 2.75 s, then 483.1 step/s
0.6% of the wall time in steps, final loss 0.2373
```

Each row has nine IDs. The model reads the first eight and predicts the last eight, so `seq_len=8`. With `vocab_size=4`, valid IDs are zero to three. The run logs training loss and validation perplexity at steps five and ten.

Perplexity exponentiates cross entropy weighted by the number of valid targets. Lower is better when using the same validation data and tokenizer. Validation scores the trained weights. If the objective keeps an EMA, as in `LMObjective(..., ema_decay=0.999)`, it scores the average and prints `eval val (ema)`. This cyclic task does not measure general language ability.

## Artifacts and metrics

`Objective.evaluate` returns scoring artifacts for the whole coordinated batch:

| Objective | Artifact |
|---|---|
| `LMObjective` | Teacher-forced `TokenScores`. |
| Diffusion | One generated image or video per real row. |
| JEPA | Representations. |
| Masked diffusion | Generated token rows, one batch's worth, undecoded, for custom text metrics. These are not teacher-forced perplexity and not held-out NELBO. |

A `Metric[S]` declares the artifact type it `reads`. Its `__call__` computes sufficient statistics for one batch. `merge(accumulated, contribution)` combines those statistics across batches, and `finalize(accumulated)` returns the scalar score. The first contribution starts the pass.

The trainer stores only the accumulator and current contribution. Metric instances keep no state between calls. Each accumulator belongs to one pass, and a metric can merge its own NumPy buffers in place.

The built-in metrics reduce their batches as follows:

- Perplexity sums weighted losses and target weights, then exponentiates.
- Image means weight each image once. Video PSNR and SSIM weight each frame once.
- Paired image metrics and CLIP need as many generated rows as reference or prompt rows.
- FID pools float64 counts, means and centered second moments for the generated and real populations. It then computes one distance using unbiased covariances. Each population needs at least two rows. State uses O(D²) memory regardless of batch count, plus bounded workspace for batch features and the matrix square root. A small-population FID result is not FID-50k.

  With default weights, features and distance match pytorch-fid 0.3.0 using its published weights. Resizing is bilinear to 299x299 without antialiasing, matching `F.interpolate(align_corners=False)`. `tests/test_metrics.py` checks 1e-4 absolute error on features and 1e-5 relative error on distance. Compare these values with pytorch-fid. They are not comparable with clean-fid's antialiased bicubic resize or the TF1 TTUR code.

  On TPU, the feature extractor inserts an optimization barrier before each strided 2D convolution. This prevents XLA's space-to-batch rewrite from miscompiling small per-device batches. It adds no images to the batch. CPU and GPU compilation removes the barrier.

- JEPA's `linear_probe` and `knn_probe` fit on the first half of each batch and test on the second half. They log the mean of the batch accuracies as `val/batch_linear_probe_accuracy` and `val/batch_knn_probe_accuracy`. These numbers depend on how the batch is split and are not a probe over the full dataset.

Training metrics use the `train/` prefix. Reduced validation metrics use `val/`. Metric names must be unique within a pass.

`key=0` is the same root key as `key=jax.random.key(0)`. The fit record
also keeps the supplied integer seed.

`Mean` turns per-example values or a `(total, count)` pair into a metric.
It sums counts across uneven batches, giving a small last batch its own
weight. Set `better` and `reads` explicitly, because evaluation selects
the exact artifact type. This LM metric ranks top-1 accuracy from the logits:

```python
from dew import Checkpoints, Mean
from dew.artifacts import TokenScores

accuracy = Mean(
    lambda scores, batch: (np.sum(scores.correct * scores.weights), np.sum(scores.weights)),
    reads=TokenScores, name="accuracy", better="higher",
)
language_model = LMObjective(model, seq_len=8, ema_decay=None)
accuracy_data = Dataset.from_records({"text": train_tokens}, batch=8,
                                    validation={"text": val_tokens})
run = Trainer(language_model, optax.adam(0.01), key=jax.random.key(0),
              checkpoints=Checkpoints("runs/lm-accuracy"))
state = run.fit(
    accuracy_data, steps=10, eval_every=5, metrics=[accuracy], best=accuracy,
)
run.checkpoints.wait()
assert run.checkpoints.best is not None
```

`Mean` starts a pass with its first contribution and finalizes on the
host without collectives. A vector counts each example once. A pair can
specify token counts or fractional weights. A scalar batch mean is rejected.
`TokenScores.correct` comes from the same chunked head as its losses,
without retaining a full logits tensor. The name `accuracy` becomes
`val/accuracy`. Passing `name="val/accuracy"` raises an error because
the prefix is already present.

## Previews

To run `Objective.preview`, pass `fit(preview=True)`. Process zero must have a tracker, and evaluation must have a coordinated batch. The trainer calls it once per evaluation event. A tracker that accepts only scalars does not enable previews.

LM previews use the objective's `Samples` configuration. Diffusion draws at most four display samples. Masked diffusion uses its configured preview count. DPO and GRPO preview the live policy because their EMA stores a frozen reference. The base `preview` reuses the first scoring artifacts, so JEPA's representation histogram needs no second encoder pass. Preview samples never enter scoring metrics. Without metrics, evaluation skips scoring. Without a tracker, it skips previews.

## Trackers

Diagnostic messages are separate from scalar and record trackers. Dew uses Python's
`"dew"` logger, at `WARNING` level by default. Rich formats logs on stderr;
redirected stderr is plain. The live training panel shares its console with the
log handler. Change verbosity with
`logging.getLogger("dew").setLevel(logging.INFO)`. A handler configured on `"dew"` before importing
Dew is left untouched. To use your application's root handlers instead, remove Dew's handlers
and set `logging.getLogger("dew").propagate = True`.

`Tracker` has three methods: `log(scalars, step)`, `artifact(value, step)` and `close()`. `Trainer.fit` uses the tracker without closing it. Whoever constructs it must close it. If closing a tracker context manager fails during another exception, the original exception still propagates. Import trackers from `dew.training`:

| Tracker | Writes | Install |
|---|---|---|
| `LocalTracker(path)` | JSONL journals of scalars and typed records, written synchronously, plus preview files. | |
| `WandbTracker` | Weights & Biases; `offline=True` opens no online session. | `dewml[wandb]` |
| `MLflowTracker(experiment, name, uri=store)` | One MLflow run through `MlflowClient`. | `dewml[mlflow]` |
| `TensorBoardTracker(path)` | One event file through TensorBoard's own `EventFileWriter`, without TensorFlow. | `dewml[tensorboard]` |
| `Trackers(*trackers)` | Every tracker receives each report even if another fails; the first failure is raised. | |

`LocalTracker` never overwrites a recipe's `run.json`. Recipes create it under the checkpoint directory. If checkpoints use a URI, the run prints its local tracking path. Recipes preserve a configured W&B preview request. Local-only reporting does not enable previews.

Local JSON stores non-finite metric values as `"NaN"`, `"+Inf"` and `"-Inf"`. A perfect PSNR remains positive infinity. Read these fields with `float(value)`.

For plots, install `dewml[plots]` and call `tracker.plot()`. To render them when the tracker closes, use `LocalTracker(path, plots=True)`. Matplotlib uses the Agg backend without opening a display. Nothing is plotted during training. Plots mark non-finite points and omit them from curve segments, while the journal keeps their exact values.

`MLflowTracker` leaves MLflow's global active run untouched. Scalars become metrics. Records become JSON artifacts at `records/<type>-<step>.json`. MLflow rejects a second value for a param, so artifacts allow a run to report several records of the same type. Previews upload the files produced by the local renderers. If a `FitEnded` record has a status other than "completed", the run ends as `FAILED`. Set `uri` to any store MLflow can read. From MLflow 3.16 on, a local file store also needs `MLFLOW_ALLOW_FILE_STORE=true`.

`TensorBoardTracker` writes scalars as scalar summaries and records as text summaries under `reporting/<type>`. Images use image summaries under `val/samples/<index>`. The image plugin shows clips as animated GIFs. Representations use a histogram of their per-dimension spread. `TokenScores` has no summary form and raises an error, as with W&B.

Reporting writes every report synchronously, with no background queue or dropped reports. This I/O takes time at each log step. On an i9-12900K, a report of six scalars took 0.005 ms for a local journal, 0.033 ms for an event file and 0.26 ms for an MLflow file store. A thousand reports therefore cost 5 ms, 33 ms and 0.26 s respectively. Across 10,000 steps of a small regression with `log_every=10`, local and TensorBoard tracking stayed within the run-to-run variation of an untracked run. MLflow run creation, batch logging and termination added 0.5 s to 6.2 s.

## Run records

`dew.telemetry.records` defines the typed records a tracker receives:

| Record | Meaning |
|---|---|
| `RunRecord` | The resolved model, data and optimizer configuration, and package versions. |
| `FitStarted` | The fit began. |
| `StepCompiled` | A training step compiled for a new batch shape, with the seconds it took, the remat it compiled under, and the per-axis link bandwidths and projection-spreading choices. |
| `CheckpointRequested` | A checkpoint save was submitted asynchronously; it does not mean the checkpoint is durable. |
| `ProfileWindow` | The directory a `Trainer` profile window wrote and the number of steps it traced, reported once the trace stopped. It copies no per-step layer tensors. A standalone `dew.Profiler` capture reports no record. |
| `FitEnded` | The fit ended, with its status. |
| `TrialFinished` | One sweep trial. |

Dew does not hash data contents or source revisions. Put their identities in the run summary when you need them, and save the optimizer, data source and evaluation settings separately from the checkpoints.

## Hyperparameter sweeps

Use `config.sweep(space, *, train, trials, ledger, tracker, search=random_search, seed=0)` on a `RunConfig` to train one trial per point in a search space. The `train` entry point trains a configuration and returns its score. Each trial uses the normal training loop, with no scheduler, and gets its own record, checkpoints and tracking directory under `<trainer.name>/trial-<index>`.

The space maps dotted paths in the run record to candidate values. `to_dict` and `from_dict` apply each point to the configuration. If the class does not declare a path, the call raises an error rather than training with the unchanged configuration.

| Search | Behavior |
|---|---|
| `random_search` | Draws each field independently, reproducibly from the trial number. |
| `grid_search` | Walks the cartesian product in order. |
| `optuna_search` | Asks Optuna's sampler for the next point and tells it the trials in the ledger. Install `dewml[hpo]`. |

Each finished trial goes into the ledger before the tracker receives it. Rerunning an interrupted sweep resumes the interrupted trial and skips finished trials. A ledger for a different space is rejected. The tracker receives `sweep/value` at the trial's number and a `TrialFinished` record. Set `trainer.name` so that trials get separate names and cannot resume from each other's checkpoints.

The example continues the previous one and reuses `objective`, `data`, `Perplexity` and `jax`:

```python
from dew import Evaluation, LocalTracker
from dew.config import ModelConfig, OptimConfig, RunConfig, TrainerConfig
from dew.config.sweep import grid_search
from dew.data import TokenWindows

config = RunConfig(
    # The run records the model it trains, and the batches above stand in
    # for the dataset this names.
    model=ModelConfig.from_model(objective.model),
    data=TokenWindows(seq_len=8),
    optim=OptimConfig(optimizer="adam"),
    trainer=TrainerConfig(name="lm-rate", checkpoint_dir="runs/sweep", steps=10, batch_size=8,
                          eval_every=None, checkpoint_every=None),
)


def trial(run: RunConfig) -> float:
    """Train one point and score it: the perplexity its own run ends on."""
    state = run.train(objective, data, name=run.trainer.name or "lm-rate")
    return float(Evaluation.run(objective, state.variables, data.val, metrics=(Perplexity(),),
                                key=jax.random.key(1), step=int(state.step)).scores["val/perplexity"])


with LocalTracker("runs/sweep/tracking") as tracker:
    trials = config.sweep({"optim.learning_rate": [0.01, 0.003]}, train=trial, trials=2,
                          ledger="runs/sweep/ledger.json", tracker=tracker, search=grid_search)
print(min(trials, key=lambda trial: trial.value).overrides)
```

```text
Experiment_Name: lm-rate/trial-0
Local tracking: /tmp/dew-docs/docs_guides_evaluation.md/runs/sweep/lm-rate/trial-0/tracking
Training CausalTransformer from step 0 to 10: 2,688 parameters, on 1 × cpu, batch 8, float32
Trained 10 steps in 0:00:01: first step after 0.66 s, then 2072.9 step/s
0.6% of the wall time in steps, final loss 0.2373
Experiment_Name: lm-rate/trial-1
Local tracking: /tmp/dew-docs/docs_guides_evaluation.md/runs/sweep/lm-rate/trial-1/tracking
Training CausalTransformer from step 0 to 10: 2,688 parameters, on 1 × cpu, batch 8, float32
Trained 10 steps in 0:00:01: first step after 0.68 s, then 2683.2 step/s
0.4% of the wall time in steps, final loss 0.6440
{'optim.learning_rate': 0.01}
```

The trials train under `runs/sweep/lm-rate/trial-0` and `runs/sweep/lm-rate/trial-1`. Each saves its learning rate in its own `run.json`. `sweep` returns the trials from the ledger. `trial.value` is the score returned by the entry point; `trial.overrides` is the point used for training. A second run of the script reads the ledger without training again.

## Reproducibility and limits

Each evaluation event derives its random key from the run key and training step. Scoring batch indices and preview sampling use separate fixed RNG domains. Repeating an event with the same state and batches therefore reproduces its samples. Adding a tracker leaves scoring draws unchanged.

All processes run the objective's numerical work and gather global arrays. Only process zero computes host metrics, decodes previews and writes tracking output. Host metric kernels use one process-local device.

Evaluation uses `dew.artifacts.collective_host` to check local conversions and addressable shards before transfer. Processes agree on the global gather plan and share each transfer's outcome before starting the next gather. Built-in previews agree on local setup and generation outcomes before transfer. If a custom hook has its own collectives, call `agree_process_phase` at those same points. The trainer's final agreement after the hook cannot replace these calls.

Finish all gathers before decoding on process zero alone. For local arrays, plain `host` still works on process zero alone. A dead process, blocked loader or failure inside a running device collective still requires the distributed runtime's timeout and launcher failure handling.

Validation stops when the shortest shard ends. Remaining rows in longer shards do not enter the metrics. `evaluation/coordinated_batches`, `evaluation/records` and `evaluation/uneven_shards` describe the consumed prefix, which may be smaller than the full split.

The record count comes from the placed global batch. Sequence and pipeline-stage replicas count once; scalar metadata adds no rows. The printed evaluation line reports scores, record count, elapsed time and whether shards were uneven. An empty coordinated pass returns no metric values or preview. A non-empty pass with no counted language target raises an error.

To compare generative scores, record the seed, generated and real sample counts, conditioning data, solver, sampling steps and guidance. Include the feature and preprocessing definition, and keep the run configuration with the results.
