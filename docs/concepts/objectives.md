# The objectives seam

The trainer owns the mechanics: sharding, gradients, EMA bookkeeping, checkpoints, logging, the loops. An `Objective` owns what is being learned: the parameters it holds, the loss it computes from a batch, and what validation produces. Swapping the objective swaps the research question without touching any of the mechanics.

## The interface

`dew.objectives.Objective` is four methods and one field.

```python
class Objective(ABC):
    tag: str = "objective"   # names the checkpoint artifact this run publishes
    ema: EMASpec

    def init_params(self, rng) -> Any: ...
    def loss(self, params, ema_params, batch, rng, step) -> Tuple[jax.Array, Dict[str, jax.Array]]: ...
    def make_validation_step(self, **kwargs) -> Callable[[Any, Any], Any]: ...
    def log_validation_artifacts(self, wandb, artifacts, step: int): ...
```

`loss` returns the scalar and a dict of auxiliary metrics alongside it, so an objective with several loss terms or with per-step diagnostics can report them without the trainer knowing what they mean. The trainer differentiates with `has_aux=True` and logs whatever is in the dict under `train/`.

`EMASpec(decay, path)` says which slice of the parameter tree the EMA copy tracks and how fast. `decay` is an optax schedule read at the current update index, so a momentum ramp works. `path` is a tuple of keys into the tree; the default `()` averages everything. JEPA sets it to the context encoder alone, because its target encoder is the EMA of that subtree and the predictor must stay out of the average.

## What the trainer does with it

`dew.training.ObjectiveTrainer` builds one jitted step around the objective's loss:

- `jax.value_and_grad(objective_loss, has_aux=True)` on the global batch. The loss is a mean over the batch-sharded axis, so its gradient carries the cross-device all-reduce by itself; there is no hand-written `pmean`.
- With `use_dynamic_scale`, the mixed-precision branch runs the same loss through `DynamicScale.value_and_grad` and keeps the old params and optimizer state wherever the gradients came back non-finite.
- The EMA runs on the update clock, not the micro-batch clock. Under gradient accumulation the params only move on every `grad_accum_steps`-th micro-step, so the average happens on that step and the decay schedule is indexed by completed updates.
- The step is `jax.jit`ed with explicit `in_shardings` and `out_shardings` and donates the train state.

Validation is the objective's too: `make_validation_step(**kwargs)` returns a `(val_state, batch) -> artifacts` function, which the trainer runs and hands to whatever `EvaluationMetric` objects the run was given. `log_validation_artifacts` draws them if there is anything to draw.

## The two objectives

`dew.objectives.diffusion.DiffusionObjective` is the default: sample a noise level from the schedule, corrupt the sample, predict, transform the prediction, weight the per-sample losses by the schedule's weights. Its validation step is a sampler run, and its artifacts are generated images or videos. If an autoencoder is configured it encodes the batch first, which is how latent diffusion runs on the same path.

`dew.objectives.jepa.JepaObjective` predicts the representation of masked target blocks from the representation of the visible context, in latent space. The context encoder is trained, the target encoder is the EMA copy and is stop-gradiented, and the predictor maps context embeddings plus target positions to the target representations. Targets are layer-normalized without a learned affine before the L2 loss, which fixes the scale of the prediction problem. Every step reports `repr_std` and `repr_cov_offdiag`, because the characteristic failure of this objective is silent collapse: both encoders agree on a constant and the loss goes to zero.

## Writing one

Implement the four methods, give the class a `tag` and an `ema`, and hand an instance to the trainer as `objective=`. `tests/test_objectives.py` drives a two-parameter `ConstantObjective` through the same loop, which is the shortest example of what the seam actually requires.
