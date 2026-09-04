# The objectives seam

The trainer owns the mechanics: sharding, gradients, EMA bookkeeping, checkpoints, logging, the loops. An `Objective` owns what is being learned: the parameters it holds, the loss it computes from a batch, and what evaluation produces. Swapping the objective swaps the research question without touching any of the mechanics.

## The interface

`dew.objectives.Objective` is three methods and three attributes.

```python
import jax
from dew import Aux, EMASpec, InputSpec
from dew.artifacts import Artifacts
from dew.objectives.base import Variables

class Objective:
    inputs: InputSpec              # per-example shapes the tree is initialised from
    ema: EMASpec | None = None     # which leaves get an exponential moving average, and how fast
    artifact: type | None = None   # what evaluate returns

    def init(self, key) -> Variables: ...
    def loss(self, params, batch, step) -> tuple[jax.Array, Aux]: ...
    def evaluate(self, params, batch, step) -> Artifacts | None: ...
```

`init` builds the whole variables tree from one key, every collection, and the tree can hold several modules: JEPA's holds a context encoder and a predictor, a diffusion objective's holds the model and its frozen condition encoders. It is pure, and the trainer traces it once for shapes and once for values.

`loss` returns the scalar and an `Aux`: a dict of metrics the trainer logs under `train/`, and optionally `variables`, non-parameter collections to write back into the tree. The trainer differentiates with respect to `params["params"]` only; every other collection is read as state. `Aux.variables` is the one channel for anything a step updates without a gradient, which is how a mixture-of-experts objective moves its routing bias and how batch statistics or sown values would travel.

`step` is a `Step`: the step number, the key for this step (`jax.random.fold_in(run_key, step)`, so every draw in a step is reproducible and no random state is carried or checkpointed), and `ema`, the averaged weights.

`evaluate` returns a typed artifact or a tuple of them: an `ImageGrid`, `VideoGrid`, `TextSamples`, `TokenScores` or `Representations` from `dew.artifacts`. It is called once per validation batch with the arrays already on the mesh and outside any jit, so an objective jits its own device work inside it and decodes to host strings after. Nothing in an objective logs, opens a file or knows about a tracker; the trainer hands the artifacts to the metrics and the tracker.

`EMASpec(decay, select)` says which leaves the EMA copy tracks and how fast. `decay` is an optax schedule read at the count of completed optimizer updates, so a momentum ramp works. `select` is a `PathFilter`, a predicate over a leaf's path; the default `everything` averages the whole tree, and `under("params", "context_encoder")` is what JEPA uses, because its target encoder is the EMA of that subtree and the predictor must stay out of the average. The same filter type labels optimizer groups and frozen subtrees, so there is one way to name a part of the tree.

## What the trainer does with it

`dew.training.Trainer` builds one compiled step around the objective's loss:

- `jax.value_and_grad(loss, has_aux=True)` on the global batch. The loss is a mean over the batch-sharded axis, so its gradient carries the cross-device all-reduce by itself; there is no hand-written `pmean`.
- With `dynamic_scale=True`, the mixed-precision branch runs the same loss through `DynamicScale.value_and_grad`; a step whose gradients came back non-finite keeps the old params and optimizer state, does not advance the step and does not move the EMA.
- The EMA runs on the update clock, not the micro-batch clock. Under `accumulation=k` the params only move on every k-th micro-step, so the average happens on that step and the decay schedule is indexed by completed updates.
- The collections in `Aux.variables` are written back into the tree after the update.
- The step is compiled once with explicit in and out shardings and donates the train state. `Trainer.compile(state, batch)` is that executable, which is what the benchmarks time.

Evaluation is the objective's too. Every `eval_every` steps the trainer runs the validation pass, calls `evaluate` on each batch, hands each artifact to the metrics that read its type, and to the tracker, which renders it. A `Metric` names the artifact type it `reads`, measures one batch, and `reduce`s a pass; `metrics.perplexity()` reads `TokenScores` and reduces to exp of the target-weighted mean over the whole pass, so a batch of padding weighs nothing. Every process agrees how many validation batches it holds before the pass starts, so a pod cannot wedge on an uneven split, and an exception in evaluation fails the run rather than printing.

A `Trainer` opens nothing when constructed: no tracker, no checkpoint directory, no mesh. `fit` does, and only through the `Checkpoints` and `Tracker` it was given; a run with neither trains and validates locally and logs to the terminal.

## The three objectives

`dew.objectives.diffusion.DiffusionObjective(model, process, inputs, ...)`: sample a noise level from the process's schedule, corrupt the sample, predict, transform the prediction, weight the per-sample losses by the process's weighting. The conditions are dropped on `unconditional_prob` of each batch so classifier-free guidance has an unconditional model to steer against. Its `evaluate` samples `VALIDATION_SAMPLES` images through `dew.sampling.sample` with the solver, guidance and step count it was constructed with, and returns an `ImageGrid` or a `VideoGrid`. If an autoencoder is configured it encodes the batch first, which is how latent diffusion runs on the same path; the model's channel fields are then the autoencoder's `latent_channels`, 4 for the Stable Diffusion VAE, while `InputSpec` still takes the pixel shape. The frozen condition encoders' parameters live under `params["encoders"]`, placed by the layout like any other leaf, so the compiled step carries no encoder constants.

`dew.objectives.jepa.JepaObjective(encoder, predictor, mask, sample, ...)` predicts the representation of masked target blocks from the representation of the visible context, in latent space. The context encoder is trained, the target encoder is the EMA copy and is stop-gradiented, and the predictor maps context embeddings plus target positions to the target representations. Targets are layer-normalised without a learned affine before the L2 loss, which fixes the scale of the prediction problem. Every step reports `repr_std` and `repr_cov_offdiag`, because the characteristic failure of this objective is silent collapse: both encoders agree on a constant and the loss goes to zero. Its `evaluate` returns `Representations`, which `metrics.linear_probe` and `metrics.knn_probe` read.

`dew.objectives.lm.LMObjective(model, seq_len, ...)` is next-token cross entropy through the chunked fp32 head, with padding and packed-document boundaries weighted out. Its `evaluate` returns `TokenScores` for every pass and `TextSamples` when `samples` are configured. `dew.diffusion.discrete.MaskedDiffusionObjective` is the same model with full attention under a masking process, on the same data path; nothing in the trainer knows the difference.

## Writing one

Implement `init`, `loss` and `evaluate`, set `inputs` and, if wanted, `ema`, register the class with `@objectives(name)`, and hand an instance to the `Trainer`. `tests/test_objectives.py` drives a two-parameter `ConstantObjective` through the same loop, which is the shortest example of what the seam requires.
