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

`Objective.evaluate` returns structured artifacts. `LMObjective` produces `TokenScores`; diffusion evaluation produces image or video grids; JEPA produces representations. A metric's `reads` property declares the artifact it consumes. The trainer passes that artifact and the corresponding batch to the metric, then calls the metric's reducer over the validation pass.

Perplexity aggregates weighted loss and target counts. Other metrics can use different reductions. A mean of per-batch diagnostic values is not automatically a dataset-level statistic. In particular, current image previews and batch FID diagnostics do not establish a dataset FID-50k measurement.

The trainer sends artifacts from the first validation batch to the tracker. This is a display policy, not a guarantee that preview computation only happens once: the current objective evaluation path can regenerate previews for later batches.

## Use a tracker

Without a tracker, the trainer prints losses, validation results, and timing summaries to the terminal. Recipes can configure WandB tracking through their run configuration. Set the project and account information explicitly before using a remote service; the offline quickstart does not contact one.

Training metrics use names under `train/`; reduced validation metrics use `val/`. The logging cadence controls when the tracker receives values. Save the configuration separately from checkpoints when you need a record of the optimizer, data source, and evaluation settings.

## Reproducibility and limits

The current evaluation loop reuses one `Step.key` across validation batches. Repeated unconditional diffusion previews can therefore contain the same samples. Preview generation, scoring cadence, and pass-level generative metrics need separate contracts; these issues are tracked for repair.

This guide's deterministic token-scoring example does not depend on evaluation sampling. For stochastic evaluation, document the seed, number of distinct generated samples, conditioning data, and reduction method. Do not treat a repeated four-image preview as a larger evaluation set.
