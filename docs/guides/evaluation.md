# Evaluation and tracking

This guide assumes you have done the [first training run](../getting-started.md). The training loss measures the batches the optimizer trains on. Validation uses a separate dataset, and it runs only when you set how often it should run.

## Enable evaluation explicitly

To evaluate, provide `Dataset.val`, set `eval_every` in `Trainer.fit`, and pick metrics that can read the artifact your objective returns. Passing `metrics` on its own does not turn evaluation on.

The example below trains a tiny next-token decoder on synthetic token sequences and measures perplexity on a different sequence. It needs no tokenizer, downloaded weights or accelerator.

```python
import itertools

import jax
import numpy as np
import optax

from dew import Trainer, metrics
from dew.data import Dataset
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.lm import LMObjective

train_tokens = np.tile(np.array([0, 1, 2, 3, 0, 1, 2, 3, 0], np.int32), (8, 1))
val_tokens = np.tile(np.array([1, 2, 3, 0, 1, 2, 3, 0, 1], np.int32), (8, 1))
data = Dataset(train=lambda partition: itertools.repeat({"text": train_tokens}),
               val=lambda partition: iter([{"text": val_tokens}]), records=8, batch=8)
model = CausalTransformer(vocab_size=4, emb_features=16, num_layers=1,
                          num_heads=2, mlp_features=32, max_seq_len=16,
                          dtype="float32", attention_impl="xla")
objective = LMObjective(model, seq_len=8)
trainer = Trainer(objective, optax.adam(0.01), key=jax.random.key(0))
state = trainer.fit(data, steps=10, log_every=5, eval_every=5,
                    metrics=(metrics.perplexity(),))
assert int(state.step) == 10
```

Each token row has nine IDs: the model reads the first eight and predicts the last eight. `seq_len=8` counts prediction positions, and `vocab_size=4` means the valid IDs are zero to three. The arrays are int32. The model computes in float32 with the XLA attention implementation, which suits a small example.

The run prints the training loss at steps five and ten, and the validation perplexity at the same steps. Perplexity is the exponential of the cross entropy, weighted by the number of valid targets. Lower is better, as long as you compare on the same validation data and tokenizer. This synthetic cyclic task does not measure general language ability.

## Artifacts and metrics

`Objective.evaluate` returns scoring artifacts for the whole coordinated batch. `LMObjective` produces teacher-forced `TokenScores`. Diffusion produces one generated image or video per real row. JEPA produces representations. Masked diffusion produces generated token rows, one batch's worth, for custom text metrics; it does not decode them. Those samples are not teacher-forced perplexity and not held-out NELBO.

A `Metric[S]` declares which artifact type it `reads`. It computes one batch's sufficient statistics with `__call__`, combines them with `merge(accumulated, contribution)`, and reports its scalar with `finalize(accumulated)`. The first contribution starts the pass. The trainer keeps only the accumulator and the current contribution; metric instances keep no state between calls. An accumulator belongs to one pass, and a metric may merge its own NumPy buffers in place.

Each built-in metric reduces its batches as follows:

- Perplexity sums weighted losses and target weights, then exponentiates.
- Image means weight each image once. Video PSNR and SSIM weight each frame once.
- Paired image metrics and CLIP need as many generated rows as reference or prompt rows.
- FID pools float64 counts, means and centered second moments for the generated and the real population, then computes one distance with unbiased covariances. It needs at least two rows in each population. Its state takes O(D²) memory however many batches you feed it, plus bounded workspace for batch features and the matrix square root. FID over a small population is not FID-50k. With the default weights, the features and the distance reproduce pytorch-fid 0.3.0 on the weights it publishes: bilinear resize to 299x299 without antialiasing, as its `F.interpolate(align_corners=False)` does. `tests/test_metrics.py` holds them to 1e-4 absolute on features and 1e-5 relative on the distance. The values compare with pytorch-fid's, not with clean-fid's (antialiased bicubic resize) or the TF1 TTUR code's.

JEPA's `linear_probe` and `knn_probe` fit on the first half of each batch and test on the second half. They log the mean of the batch accuracies as `val/batch_linear_probe_accuracy` and `val/batch_knn_probe_accuracy`. These numbers depend on how the batch is split, and they are not a probe over the full dataset.

The trainer calls `Objective.preview` once per evaluation event, and only when you pass `fit(preview=True)`, process zero has a tracker, and there is a coordinated batch. A tracker that only takes scalars does not turn previews on. LM previews use the objective's `Samples` configuration. Diffusion draws at most four display samples. Masked diffusion uses its configured preview count. DPO and GRPO preview the live policy, because their EMA holds a frozen reference and not averaged policy weights. The base `preview` reuses the first scoring artifacts, so JEPA's representation histogram needs no second encoder pass. Preview samples never enter the scoring metrics. Without metrics, evaluation skips the scoring work; without a tracker, it skips the preview work.

## Use a tracker

Import `LocalTracker`, `WandbTracker`, `MLflowTracker`, `TensorBoardTracker` and `Trackers` from `dew.training`.

`Tracker` has three methods: `log(scalars, step)`, `artifact(value, step)` and `close()`. `Trainer.fit` borrows the tracker; whoever constructed it closes it. When a tracker is used as a context manager and closing fails while another exception is active, you still see the original exception.

`LocalTracker("runs/example/tracking")` writes JSONL journals of scalars and typed records, synchronously, plus preview files. It never overwrites a recipe's `run.json`. Recipes create this local tracker under their checkpoint directory; a run whose checkpoints live at a URI prints its local tracking path. Recipes keep the W&B preview request you configured, but they do not turn previews on when the only reporting is local. W&B reporting is optional and uses the same interface. Pass `offline=True` to keep W&B from opening an online session.

`MLflowTracker("experiment", name, uri=store)` opens one MLflow run through `MlflowClient`, so it never touches MLflow's global active run. Scalars become the run's metrics. Each record becomes a JSON artifact at `records/<type>-<step>.json`. A preview is uploaded as the same files the local renderers write. A `FitEnded` record whose status is not "completed" ends the run as `FAILED`. Records are artifacts and not params because MLflow refuses a second value for a param, and a run reports several records of the same type. `uri` can be any store MLflow reads; from MLflow 3.16 on, a local file store also needs MLflow's own `MLFLOW_ALLOW_FILE_STORE=true`. Install `dew-ml[mlflow]`.

`TensorBoardTracker("runs/example/events")` writes one event file with TensorBoard's own `EventFileWriter` and does not need TensorFlow. Scalars become scalar summaries and records become text summaries under `reporting/<type>`. Images become image summaries under `val/samples/<index>`, a clip becomes an animated GIF that the image plugin shows, and representations become a histogram of their per-dimension spread. `TokenScores` has no summary form, so it raises, as it does for W&B. Install `dew-ml[tensorboard]`.

To report to several trackers, use `with Trackers(LocalTracker(path), TensorBoardTracker(events)) as tracker:`. Every tracker receives each report even if another one fails, and the first failure is raised. Reporting has no background queue and never drops a report, so the I/O costs time at every log step. On an i9-12900K, one report of six scalars took 0.005 ms into a local journal, 0.033 ms into an event file and 0.26 ms into an MLflow file store, so a thousand reports cost 5 ms, 33 ms and 0.26 s. Over 10,000 steps of the small regression at `log_every=10`, the local and TensorBoard trackers stayed within the run-to-run spread of an untracked run. MLflow's run creation, batch logging and termination added 0.5 s to 6.2 s.

The local JSON writes non-finite metric values as the strings `"NaN"`, `"+Inf"` and `"-Inf"`. A perfect PSNR stays positive infinity; it is not written as null or as a made-up finite score. Use `float(value)` when you read these fields.

Plots are opt-in. Install `dew-ml[plots]`, then call `tracker.plot()`, or construct `LocalTracker(path, plots=True)` to render the plots when the tracker closes. Matplotlib uses the Agg backend and never opens a display. Nothing is plotted during training log steps. Non-finite points are marked and left out of the curve segments; their exact values stay in the journal.

The run records in `dew.telemetry.records` are:

- `RunRecord`: the resolved model, data and optimizer configuration, and package versions.
- `FitStarted`.
- `CheckpointRequested`: records that a checkpoint save was submitted asynchronously, not that it is durable. Checkpoint waits work as before this record was added.
- `ProfileWindow`: the directory a `Trainer` profile window wrote and the number of steps it traced. It is reported once the trace has stopped, and it copies no per-step layer tensors. A standalone `dew.profile` capture writes its own directory and reports no record.
- `FitEnded`.
- `TrialFinished`: one sweep trial.

Dew does not hash data contents or source revisions for you. Put their identities in the run summary when you need them.

Training metrics are named under `train/`, and reduced validation metrics under `val/`. The logging cadence decides when the tracker receives values. If you need a record of the optimizer, data source and evaluation settings, save the configuration separately from the checkpoints.

## Search a hyperparameter space

`dew.config.sweep.sweep` trains one trial per point of a search space, through `RunConfig.train`. Each trial is an ordinary run with its own record, checkpoints and tracking directory under `<trainer.name>/trial-<index>`. `sweep` takes the config, the space, the entry point that trains a config and returns the trial's score, a trial budget, a ledger path and a tracker. It uses the normal training loop and has no scheduler.

A space maps dotted paths into the run record to the values a trial can take. `override` applies a point through `to_dict` and `from_dict`, so a path the config class does not declare raises an error instead of silently training the unchanged config. There are three search backends:

- `random_search` draws each field independently, reproducibly from the trial number.
- `grid_search` walks the cartesian product in order.
- `optuna_search` asks Optuna's sampler for the next point and tells it the trials in the ledger. Install `dew-ml[hpo]` for it.

A finished trial is written to the ledger before it is reported. If a sweep is interrupted, rerunning the same call continues at the trial it stopped on and does not retrain finished trials. A ledger written for a different space is refused. The tracker you pass receives each trial's score as `sweep/value` at the trial's number, plus its `TrialFinished` record. A sweep needs `trainer.name`, because trials sharing one name would resume from each other's checkpoints.

The example below continues the previous one and reuses `objective`, `data`, `metrics` and `jax` from it.

```python
from dew import LocalTracker, evaluate
from dew.config import ModelConfig, OptimConfig, RunConfig, TrainerConfig
from dew.config.sweep import grid_search, sweep
from dew.registry import datasets

config = RunConfig(
    model=ModelConfig("causal_transformer", {"vocab_size": 4, "emb_features": 16,
                                             "num_layers": 1, "num_heads": 2,
                                             "mlp_features": 32, "max_seq_len": 16}),
    # The synthetic batches above stand in for the dataset this names.
    data=datasets["token_windows"](seq_len=8),
    optim=OptimConfig(optimizer="adam"),
    trainer=TrainerConfig(name="lm-rate", checkpoint_dir="runs/sweep", steps=10, batch_size=8,
                          eval_every=None, checkpoint_every=None),
)


def trial(run: RunConfig) -> float:
    """Train one point and score it: the perplexity its own run ends on."""
    state = run.train(objective, data, name=run.trainer.name or "lm-rate")
    return float(evaluate(objective, state.params, data.val, metrics=(metrics.perplexity(),),
                          key=jax.random.key(1), step=int(state.step)).scores["val/perplexity"])


with LocalTracker("runs/sweep/tracking") as tracker:
    trials = sweep(config, {"optim.learning_rate": [0.01, 0.003]}, train=trial, trials=2,
                   ledger="runs/sweep/ledger.json", tracker=tracker, search=grid_search)
print(min(trials, key=lambda trial: trial.value).overrides)
```

The two trials train under `runs/sweep/lm-rate/trial-0` and `runs/sweep/lm-rate/trial-1`, each with its own `run.json` recording the learning rate it used, and the script prints the better rate. `sweep` returns the ledger's trials, so `trial.value` is the score the entry point returned and `trial.overrides` is the point it trained.

## Reproducibility and limits

Each evaluation event derives its random key from the run key and the training step. Scoring batch indices and preview sampling use separate fixed RNG domains. Repeating an event with the same state and batches reproduces its samples, and adding a tracker does not change the scoring draws.

All processes take part in the objective's numerical work and in global-array gathers. Only process zero computes host metrics, decodes previews and writes tracking output. Host metric kernels use one process-local device. Evaluation uses `dew.artifacts.collective_host` to check local conversions and addressable shards up front, agree on the global gather plan, and share each transfer's outcome before the next gather. Built-in previews agree on local setup and generation outcomes before they start that transfer. A custom hook with its own collectives must also call `agree_process_phase` at those points; the trainer's final agreement on the hook cannot do it for you. Finish all gathers before decoding on process zero alone. Plain `host` still works on process zero alone for local arrays. A dead process, a blocked loader, or a failure inside a device collective that is already running still needs the distributed runtime's timeout and the launcher's failure handling.

Validation reads the shortest coordinated prefix of the shards. If one shard ends before another, the remaining local rows do not enter the metrics. The `evaluation/coordinated_batches`, `evaluation/records` and `evaluation/uneven_shards` fields describe that consumed prefix, which may not be the whole split. The record count comes from the placed global batch, so sequence and pipeline-stage replicas count once and scalar metadata adds no rows. The trainer prints the event key next to those counts. Metric names stay under `val/` and must be unique within a pass. An empty coordinated validation pass generates no preview and returns no metric values. A non-empty pass with no counted language target raises an error.

When you compare generative scores, record the seed, the numbers of generated and real samples consumed, the conditioning data, the solver, the number of sampling steps, the guidance, and the feature and preprocessing definition. Keep the run configuration with the results.
