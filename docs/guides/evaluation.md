# Evaluation and tracking

This guide assumes the [first training run](../getting-started.md). Training loss measures the batches used for optimization. Validation uses a separate dataset and runs only when you configure its cadence.

## Enable evaluation explicitly

Provide `Dataset.val`, set `eval_every` in `Trainer.fit`, and choose metrics compatible with the objective's returned artifact. Passing `metrics` alone does not enable evaluation.

The complete example below trains a tiny next-token decoder over synthetic token sequences and evaluates perplexity on a different sequence. It needs no tokenizer, downloaded weights, or accelerator.

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
data = Dataset(train=lambda: itertools.repeat({"text": train_tokens}),
               val=lambda: iter([{"text": val_tokens}]), records=8, batch=8)
model = CausalTransformer(vocab_size=4, emb_features=16, num_layers=1,
                          num_heads=2, mlp_features=32, max_seq_len=16,
                          dtype="float32", attention_impl="xla")
objective = LMObjective(model, seq_len=8)
trainer = Trainer(objective, optax.adam(0.01), key=jax.random.key(0))
state = trainer.fit(data, steps=10, log_every=5, eval_every=5,
                    metrics=(metrics.perplexity(),))
assert int(state.step) == 10
```

Each token row has nine IDs: the model reads the first eight and predicts the last eight. `seq_len=8` counts prediction positions, and `vocab_size=4` means valid IDs are zero through three. The arrays use int32. The model computes in float32 with the XLA attention implementation for this small CPU example.

The run prints training loss at steps five and ten and validation perplexity at the same step targets. Perplexity is the exponential of the valid-target-weighted cross entropy; smaller is better on the same validation data and tokenizer. This synthetic cyclic task does not measure general language capability.

## Artifacts and metrics

`Objective.evaluate` returns scoring artifacts for the complete coordinated batch. `LMObjective` produces teacher-forced `TokenScores`; diffusion produces one generated image or video per real row; JEPA produces representations. Masked diffusion produces batch-sized generated token rows for custom text metrics, without decoding. Those samples are neither teacher-forced perplexity nor held-out NELBO.

A `Metric[S]` declares `reads`, computes one batch's sufficient statistics with `__call__`, combines them with `merge(accumulated, contribution)`, and reports its scalar with `finalize(accumulated)`. The first contribution initializes the pass. The trainer retains only the accumulator and current contribution; metric instances retain no pass state. An accumulator belongs to one pass, and a metric may merge its owned NumPy buffers in place.

Perplexity sums weighted losses and target weights before exponentiating. Image means weight each image once; video PSNR and SSIM weight each frame once. Paired image metrics and CLIP require equal generated and reference/prompt row counts. FID pools float64 counts, means and centered second moments for the generated and real populations, then computes one distance with unbiased covariances. It requires at least two rows in each population. Its persistent state uses O(D²) memory, independent of the number of batches, plus bounded batch feature and matrix-square-root workspace. A small consumed population does not constitute FID-50k.

JEPA's `linear_probe` and `knn_probe` fit on the first half of each batch and test on its second half. They log the arithmetic mean of batch accuracies as `val/batch_linear_probe_accuracy` and `val/batch_knn_probe_accuracy`. These diagnostics depend on the batch partition and do not measure a full-dataset probe.

The trainer calls `Objective.preview` once per evaluation event only when `fit(preview=True)` is explicitly requested, process zero has a tracker, and a coordinated batch exists. A scalar-reporting sink alone does not enable preview computation. LM uses its `Samples` configuration; diffusion draws at most four separate display samples; masked diffusion uses its configured preview count. DPO and GRPO preview the live policy, because their EMA holds a frozen reference rather than averaged policy weights. The base preview reuses the first scoring artifacts, retaining JEPA's representation histogram without another encoder pass. Preview draws do not enter scoring metrics. Without metrics, evaluation skips scoring work; without a tracker, it skips preview work.

## Use a tracker

Import `LocalTracker`, `WandbTracker`, `MLflowTracker`, `TensorBoardTracker`, and `Trackers` from `dew.training`.

`Tracker` has three methods: `log(scalars, step)`, `artifact(value, step)`, and `close()`. A tracker is borrowed by `Trainer.fit`; its constructing owner closes it. Context managers preserve the active exception if closing also fails.

`LocalTracker("runs/example/tracking")` writes synchronous scalar and typed-record JSONL journals plus preview files. It never overwrites the recipe's `run.json`. Recipes preserve the existing W&B preview request but do not enable previews for local-only reporting. Recipes create this local sink under their checkpoint directory; URI-backed checkpoint runs print their local tracking path. Optional W&B reporting uses the same interface. Explicitly use `offline=True` to prevent a W&B online session.

`MLflowTracker("experiment", name, uri=store)` opens one MLflow run through `MlflowClient`, so it never uses MLflow's global active run: scalars are the run's metrics, each record is a JSON artifact under `records/<type>-<step>.json`, a preview is uploaded as the files the local renderers write, and a `FitEnded` that did not complete terminates the run `FAILED`. A record is an artifact rather than a param because MLflow refuses a second value for a param and a run reports several records of one type. `uri` is any store MLflow reads; a local file store needs MLflow's own `MLFLOW_ALLOW_FILE_STORE=true` from 3.16 on. Install `dew-ml[mlflow]`.

`TensorBoardTracker("runs/example/events")` writes one event file with tensorboard's own `EventFileWriter` and needs no TensorFlow: scalars are scalar summaries, records are text summaries under `reporting/<type>`, images are image summaries under `val/samples/<index>`, a clip is the animated GIF the image plugin renders, and representations are a histogram of their per-dimension spread. `TokenScores` has no summary and raises, as it does for W&B. Install `dew-ml[tensorboard]`.

Use `with Trackers(LocalTracker(path), TensorBoardTracker(events)) as tracker:` to own several sinks. Every sink receives each report even if another fails; the first failure propagates. There is no asynchronous reporting queue or silent drop policy: synchronous I/O has a cost at the log cadence. One report of six scalars measured 0.005 ms into a local journal, 0.033 ms into an event file and 0.26 ms into an MLflow file store on an i9-12900K, so a thousand reports cost 5 ms, 33 ms and 0.26 s. Over 10,000 steps of the small regression at `log_every=10` the local and TensorBoard sinks stayed inside the run-to-run spread of the untracked run; MLflow's run creation, batch logging and termination added 0.5 s to 6.2 s.

Local JSON encodes nonfinite metric values as strings `"NaN"`, `"+Inf"`, and `"-Inf"`. Perfect PSNR remains positive infinity, not null or a fabricated finite score. Use `float(value)` when reading these fields.

Plotting is explicit: install `dew-ml[plots]`, then call `tracker.plot()` or construct `LocalTracker(path, plots=True)` to render at close. Matplotlib uses Agg, never a display backend. No plots render on training log ticks. Nonfinite points are annotated and omitted from curve segments; their exact values remain in the journal.

Run records in `dew.telemetry.records` are `RunRecord` (resolved model/data/optimizer configuration and package versions), `FitStarted`, `CheckpointRequested`, `ProfileWindow`, `FitEnded`, and `TrialFinished` (one sweep trial). Checkpoint requests record asynchronous submission, not durability; existing checkpoint waits are unchanged. Profile records link explicit JAX trace windows, without per-step layer tensor copies. Data contents and source revisions are not automatically hashed: include their identities in the run summary when needed.

Training metrics use names under `train/`; reduced validation metrics use `val/`. The logging cadence controls when the tracker receives values. Save the configuration separately from checkpoints when you need a record of the optimizer, data source, and evaluation settings.

## Search a hyperparameter space

`dew.config.sweep.sweep` trains one trial per point of a search space through `RunConfig.train`, so a trial is an ordinary run with its own record, checkpoints and tracking directory under `<trainer.name>/trial-<index>`. It takes the config, the space, the entry point that trains a config and returns that trial's score, a trial budget, a ledger path and a tracker. There is no second training loop and no scheduler.

A space maps dotted paths into the run record to the values a trial draws from. `override` applies a point through `to_dict`/`from_dict`, so a path the config class does not declare raises rather than training the unchanged config. The backends are `random_search` (one independent draw per field, reproducible from the trial number), `grid_search` (the cartesian product in order) and `optuna_search` (Optuna's sampler, asked for the point and told the ledger's trials; install `dew-ml[hpo]`).

A finished trial reaches the ledger before it is reported, so rerunning the same call continues an interrupted sweep at the trial it stopped on and retrains no finished one. A ledger written over another space is refused. The caller's tracker receives each trial's score as `sweep/value` at the trial's number and its `TrialFinished` record. A sweep needs `trainer.name`, because trials sharing one name would resume from each other's checkpoints.

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

The two trials train under `runs/sweep/lm-rate/trial-0` and `-1`, each with its own `run.json` recording the rate it trained with, and the better rate is printed. `sweep` returns the ledger, so `trial.value` is the score the entry point returned and `trial.overrides` the point it trained.

## Reproducibility and limits

Each event derives its random key from the run key and training step. Scoring batch indices and preview sampling use separate fixed RNG domains. Repeating an event with the same state and batches reproduces its samples, while adding a tracker leaves scoring draws unchanged.

All ranks participate in objective numerical work and global-array gathers. Only process zero computes host metrics, decodes previews and writes tracking output. Host metric kernels use one process-local device. Evaluation uses `dew.artifacts.collective_host` to preflight local conversions and addressable shards, agree the global gather plan, and propagate each transfer outcome before the next gather. Built-in previews agree local setup and generation outcomes before entering that transfer operation. A custom hook with internal collectives must likewise use `agree_process_phase` at those boundaries; the trainer's final hook agreement cannot replace them. Complete all gathers before root-only decoding. Plain `host` remains usable on root alone for local arrays. A dead process, blocked loader or failure inside an in-flight device collective still requires the distributed runtime's timeout and launcher failure handling.

Validation consumes the shortest coordinated shard prefix. If one shard ends before another, remaining local rows do not enter the metrics. The `evaluation/coordinated_batches`, `evaluation/records` and `evaluation/uneven_shards` fields describe the consumed prefix, not necessarily the whole split. The record count comes from the placed global batch, so sequence and pipeline-stage replicas count once and scalar metadata contributes no rows. The trainer prints the event key alongside those counts. Metric names remain under `val/` and must be unique within a pass. Empty coordinated validation generates no preview and returns no metric values. A nonempty pass with no counted language target raises an error.

Record the seed, consumed generated and real counts, conditioning data, solver, sampling steps, guidance, and feature/preprocessing definition when comparing generative scores. Keep the run configuration with the results.
