# Evaluation and tracking

Evaluation scores a separate validation split while `Trainer.fit` runs, through metrics that read the artifacts the objective's `evaluate` returns. A tracker receives the training scalars, the validation metrics, previews and run records. Evaluation runs only when the dataset has a validation split and `fit` has `eval_every` set, and `fit` refuses an `eval_every` whose pass nothing reads: it needs `metrics` that read the objective's artifact, or `preview=True` with a tracker. Passing `metrics` alone does not turn evaluation on.

## Example

The example trains a tiny next-token decoder on synthetic token rows and measures perplexity on a different row. It needs no tokenizer, downloaded weights or accelerator.

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

Each row has nine IDs: the model reads the first eight and predicts the last eight, so `seq_len=8`, and `vocab_size=4` makes the valid IDs zero to three. The run logs the training loss at steps five and ten and the validation perplexity at the same steps. Perplexity is the exponential of the cross entropy, weighted by the number of valid targets; lower is better when compared on the same validation data and tokenizer. Validation scores the weights the run trained; an objective that keeps an EMA (`LMObjective(..., ema_decay=0.999)`) is scored on the average instead, and the line then reads `eval val (ema)`. This cyclic task does not measure general language ability.

## Artifacts and metrics

`Objective.evaluate` returns scoring artifacts for the whole coordinated batch:

| Objective | Artifact |
|---|---|
| `LMObjective` | Teacher-forced `TokenScores`. |
| Diffusion | One generated image or video per real row. |
| JEPA | Representations. |
| Masked diffusion | Generated token rows, one batch's worth, undecoded, for custom text metrics. These are not teacher-forced perplexity and not held-out NELBO. |

A `Metric[S]` declares which artifact type it `reads`. It computes one batch's sufficient statistics with `__call__`, combines them with `merge(accumulated, contribution)`, and reports its scalar with `finalize(accumulated)`. The first contribution starts the pass. The trainer keeps only the accumulator and the current contribution; metric instances keep no state between calls. An accumulator belongs to one pass, and a metric may merge its own NumPy buffers in place.

The built-in metrics reduce their batches as follows:

- Perplexity sums weighted losses and target weights, then exponentiates.
- Image means weight each image once. Video PSNR and SSIM weight each frame once.
- Paired image metrics and CLIP need as many generated rows as reference or prompt rows.
- FID pools float64 counts, means and centered second moments for the generated and the real population, then computes one distance with unbiased covariances. It needs at least two rows in each population. Its state takes O(D²) memory however many batches it is fed, plus bounded workspace for batch features and the matrix square root. FID over a small population is not FID-50k. With the default weights, the features and the distance reproduce pytorch-fid 0.3.0 on the weights it publishes: bilinear resize to 299x299 without antialiasing, as its `F.interpolate(align_corners=False)` does. `tests/test_metrics.py` holds them to 1e-4 absolute on features and 1e-5 relative on the distance. The values compare with pytorch-fid's, not with clean-fid's (antialiased bicubic resize) or the TF1 TTUR code's. On TPU the feature extractor puts an optimization barrier before each strided 2D convolution, which keeps XLA's space-to-batch rewrite from miscompiling it at small per-device batches; no images are added to the batch, and CPU and GPU compile the barrier away.
- JEPA's `linear_probe` and `knn_probe` fit on the first half of each batch and test on the second half. They log the mean of the batch accuracies as `val/batch_linear_probe_accuracy` and `val/batch_knn_probe_accuracy`. These numbers depend on how the batch is split and are not a probe over the full dataset.

Training metrics are named under `train/`, and reduced validation metrics under `val/`. Metric names must be unique within a pass.

`key=0` is the same root key as `key=jax.random.key(0)`. The fit record
also keeps the supplied integer seed.

`Mean` turns per-example values or a `(total, count)` pair into a metric.
It sums counts across uneven batches, so a small last batch has its own
weight. Choose `better` and `reads` explicitly: evaluation selects the exact
artifact type. This LM metric ranks the logits' top-1 accuracy:

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

The helper starts a pass from its first contribution and finalizes on the
host without collectives. A vector counts each example once; a pair can
carry token counts or fractional weights. A scalar batch mean is refused.
`TokenScores.correct` comes from the same chunked head as its losses; it
does not retain a full logits tensor. The unprefixed name `accuracy` is
reported as `val/accuracy`. Passing `name="val/accuracy"` gives a clear error
instead of adding the prefix twice.

## Previews

The trainer calls `Objective.preview` once per evaluation event, only when `fit(preview=True)` is passed, process zero has a tracker, and there is a coordinated batch. A tracker that only takes scalars does not turn previews on. LM previews use the objective's `Samples` configuration. Diffusion draws at most four display samples. Masked diffusion uses its configured preview count. DPO and GRPO preview the live policy, because their EMA holds a frozen reference. The base `preview` reuses the first scoring artifacts, so JEPA's representation histogram needs no second encoder pass. Preview samples never enter the scoring metrics. Without metrics, evaluation skips the scoring work; without a tracker, it skips the preview work.

## Trackers

Diagnostic messages are separate from scalar and record trackers. Dew logs through Python's
`"dew"` logger, at `WARNING` level by default, using Rich on stderr; redirected stderr is plain,
and a live training panel shares its console with the log handler. Change verbosity with
`logging.getLogger("dew").setLevel(logging.INFO)`. A handler configured on `"dew"` before importing
Dew is left untouched. To use your application's root handlers instead, remove Dew's handlers
and set `logging.getLogger("dew").propagate = True`.

`Tracker` has three methods: `log(scalars, step)`, `artifact(value, step)` and `close()`. `Trainer.fit` borrows the tracker; whoever constructed it closes it. When a tracker used as a context manager fails to close while another exception is active, the original exception still surfaces. The trackers are importable from `dew.training`:

| Tracker | Writes | Install |
|---|---|---|
| `LocalTracker(path)` | JSONL journals of scalars and typed records, written synchronously, plus preview files. | |
| `WandbTracker` | Weights & Biases; `offline=True` opens no online session. | `dewml[wandb]` |
| `MLflowTracker(experiment, name, uri=store)` | One MLflow run through `MlflowClient`. | `dewml[mlflow]` |
| `TensorBoardTracker(path)` | One event file through TensorBoard's own `EventFileWriter`, without TensorFlow. | `dewml[tensorboard]` |
| `Trackers(*trackers)` | Every tracker receives each report even if another fails; the first failure is raised. | |

`LocalTracker` never overwrites a recipe's `run.json`. Recipes create a local tracker under their checkpoint directory; a run whose checkpoints live at a URI prints its local tracking path. Recipes keep a configured W&B preview request, but do not turn previews on when the only reporting is local. The local JSON writes non-finite metric values as the strings `"NaN"`, `"+Inf"` and `"-Inf"`; a perfect PSNR stays positive infinity. Read these fields with `float(value)`. Plots are opt-in: install `dewml[plots]`, then call `tracker.plot()`, or construct `LocalTracker(path, plots=True)` to render the plots when the tracker closes. Matplotlib uses the Agg backend and never opens a display, and nothing is plotted during training. Non-finite points are marked and left out of the curve segments; their exact values stay in the journal.

`MLflowTracker` never touches MLflow's global active run. Scalars become the run's metrics. Each record becomes a JSON artifact at `records/<type>-<step>.json`, rather than a param, because MLflow refuses a second value for a param and a run reports several records of the same type. A preview is uploaded as the files the local renderers write. A `FitEnded` record whose status is not "completed" ends the run as `FAILED`. `uri` can be any store MLflow reads; from MLflow 3.16 on, a local file store also needs `MLFLOW_ALLOW_FILE_STORE=true`.

`TensorBoardTracker` writes scalars as scalar summaries and records as text summaries under `reporting/<type>`. Images become image summaries under `val/samples/<index>`, a clip becomes an animated GIF that the image plugin shows, and representations become a histogram of their per-dimension spread. `TokenScores` has no summary form, so it raises, as it does for W&B.

Reporting has no background queue and never drops a report, so the I/O costs time at every log step. On an i9-12900K, one report of six scalars took 0.005 ms into a local journal, 0.033 ms into an event file and 0.26 ms into an MLflow file store, so a thousand reports cost 5 ms, 33 ms and 0.26 s. Over 10,000 steps of a small regression at `log_every=10`, the local and TensorBoard trackers stayed within the run-to-run spread of an untracked run. MLflow's run creation, batch logging and termination added 0.5 s to 6.2 s.

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

`config.sweep(space, *, train, trials, ledger, tracker, search=random_search, seed=0)` on a `RunConfig` trains one trial per point of a search space through the entry point `train`, which trains a config and returns its score. Each trial is an ordinary run with its own record, checkpoints and tracking directory under `<trainer.name>/trial-<index>`. It uses the normal training loop and has no scheduler.

A space maps dotted paths into the run record to the values a trial can take. A point is applied through `to_dict` and `from_dict`, so a path the config class does not declare raises instead of silently training the unchanged config.

| Search | Behavior |
|---|---|
| `random_search` | Draws each field independently, reproducibly from the trial number. |
| `grid_search` | Walks the cartesian product in order. |
| `optuna_search` | Asks Optuna's sampler for the next point and tells it the trials in the ledger. Install `dewml[hpo]`. |

A finished trial is written to the ledger before it is reported, so rerunning an interrupted sweep continues at the trial it stopped on and does not retrain finished trials. A ledger written for a different space is refused. The tracker receives each trial's score as `sweep/value` at the trial's number, plus its `TrialFinished` record. A sweep needs `trainer.name`, because trials sharing one name would resume from each other's checkpoints.

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

The two trials train under `runs/sweep/lm-rate/trial-0` and `runs/sweep/lm-rate/trial-1`, each with its own `run.json` recording the learning rate it used. `sweep` returns the ledger's trials: `trial.value` is the score the entry point returned and `trial.overrides` the point it trained. A second run of the script reads the ledger and trains nothing.

## Reproducibility and limits

Each evaluation event derives its random key from the run key and the training step. Scoring batch indices and preview sampling use separate fixed RNG domains, so repeating an event with the same state and batches reproduces its samples, and adding a tracker does not change the scoring draws.

All processes take part in the objective's numerical work and in global-array gathers. Only process zero computes host metrics, decodes previews and writes tracking output. Host metric kernels use one process-local device. Evaluation uses `dew.artifacts.collective_host` to check local conversions and addressable shards up front, agree on the global gather plan, and share each transfer's outcome before the next gather. Built-in previews agree on local setup and generation outcomes before they start that transfer. A custom hook with its own collectives must also call `agree_process_phase` at those points; the trainer's final agreement on the hook cannot do it. Finish all gathers before decoding on process zero alone. Plain `host` still works on process zero alone for local arrays. A dead process, a blocked loader, or a failure inside a device collective that is already running still needs the distributed runtime's timeout and the launcher's failure handling.

Validation reads the shortest coordinated prefix of the shards. If one shard ends before another, the remaining local rows do not enter the metrics. The `evaluation/coordinated_batches`, `evaluation/records` and `evaluation/uneven_shards` fields describe that consumed prefix, which may not be the whole split. The record count comes from the placed global batch, so sequence and pipeline-stage replicas count once and scalar metadata adds no rows. The trainer's printed evaluation line shows the scores, the record count, the elapsed time and, when shards were uneven, that fact. An empty coordinated validation pass generates no preview and returns no metric values. A non-empty pass with no counted language target raises.

When comparing generative scores, record the seed, the numbers of generated and real samples consumed, the conditioning data, the solver, the number of sampling steps, the guidance, and the feature and preprocessing definition, and keep the run configuration with the results.
