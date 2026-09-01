"""The optimizer a recipe builds from an OptimConfig.

Every recipe wires the same solver: a warmup-cosine schedule when one is
asked for, weight decay folded into the optimizer's own kwargs, global-norm
clipping, and gradient accumulation over several micro-batches. That wiring
is library behavior, so it lives here and the recipes call it.
"""

import optax

from dew.config import OptimConfig

OPTIMIZER_MAP = {
    'adam': optax.adam,
    'adamw': optax.adamw,
    'lamb': optax.lamb,
}


def build_optimizer(config: OptimConfig, steps_per_epoch: int) -> optax.GradientTransformation:
    """The solver, with its schedule, clipping and gradient accumulation."""
    learning_rate = config.learning_rate
    if config.learning_rate_schedule == 'cosine':
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate, peak_value=config.learning_rate_peak,
            warmup_steps=config.learning_rate_warmup_steps,
            decay_steps=steps_per_epoch * config.learning_rate_decay_epochs,
            end_value=config.learning_rate_end,
        )
    opts = dict(config.optimizer_opts)
    if config.weight_decay is not None:
        opts['weight_decay'] = config.weight_decay
    solver = OPTIMIZER_MAP[config.optimizer](learning_rate, **opts)

    if config.clip_grads > 0:
        solver = optax.chain(optax.clip_by_global_norm(config.clip_grads), solver)
    if config.grad_accum_steps > 1:
        # Accumulate over several micro-batches so the effective batch can
        # exceed what fits in device memory at once.
        solver = optax.MultiSteps(solver, every_k_schedule=config.grad_accum_steps)
    return solver
