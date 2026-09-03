"""The optimizer a recipe builds from an OptimConfig.

Every recipe wires the same solver: a warmup-cosine schedule when one is
asked for, weight decay folded into the optimizer's own kwargs, global-norm
clipping, and gradient accumulation over several micro-batches. That wiring
is library behavior, so it lives here and the recipes call it.

The 'muon' entry is the production parameter-group split the labs converged
on (docs/research/frontier-training.md:183): AdamW on the embeddings, the
head, the router and the norms, Muon on the matrices. `optax.contrib.muon`
owns the masked composition itself (it partitions with `optax.masked` per
group, optax/contrib/_muon.py:694), so what Dew supplies is the parameter
spec that says which group a parameter belongs to and which of its axes are
the matrix.
"""

import jax
import optax

from dew.config import OptimConfig
from dew.training import distributed

# Attention stores its projections either as one matrix over the flattened
# head space or as a dimension per head, so the head dimensions count as one
# side of the matrix and both layouts get the same orthogonalized update.
HEAD_AXES = frozenset({'heads', 'head_dim', 'kv'})

# An expert dimension stacks whole matrices, one per expert, so it is a batch
# axis: it names neither side, and optax orthogonalizes each expert on its
# own (optax/contrib/_muon.py:56-74).
BATCH_AXES = frozenset({'exp'})

# A parameter that maps into or out of a discrete index is a lookup rather
# than a matrix, so AdamW keeps the embeddings, the head and the router
# (docs/research/frontier-training.md:183). An expert dimension is one of
# these when it is the output, where it counts the experts a router scores,
# and a batch axis when it leads, where it stacks one matrix per expert.
SELECTION_AXES = frozenset({'vocab', 'output'})


def _matrix_sides(
    path, axes: distributed.LogicalAxes,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The contracted axes and the output axes of a declared parameter.

    A dimension continues the side before it when the table leaves it unnamed,
    as the spatial dimensions of a patch embedding are, or when it and its
    predecessor are both head dimensions. What is left has to be two sides,
    one contracted and one output, which is what MaxText's per-name table
    produces for its own trees (maxtext utils/muon_utils.py:100-175).
    """
    sides: list[list[int]] = []
    for dimension, name in enumerate(axes):
        if name in BATCH_AXES:
            continue
        previous = axes[dimension - 1] if dimension else None
        continues = bool(sides) and (
            name is None or (name in HEAD_AXES and previous in HEAD_AXES))
        if continues:
            sides[-1].append(dimension)
        else:
            sides.append([dimension])
    if len(sides) != 2:
        raise ValueError(
            f"{jax.tree_util.keystr(path)} is declared {axes}, which is not one "
            f"contracted side and one output side, so Muon cannot tell what its "
            f"matrix is. An axis that stacks matrices rather than being part of "
            f"one belongs in BATCH_AXES.")
    return tuple(sides[0]), tuple(sides[1])


def muon_weight_dimension_numbers(params):
    """A `MuonDimensionNumbers` per parameter, None where AdamW steps in.

    Which group a parameter lands in is read off the logical axes declared in
    `distributed.DEFAULT_LOGICAL_PARAM_AXES`, the table the sharding
    derivation already reads, so one declaration answers both questions. A
    parameter of rank below two, a bias, and a parameter that maps into or out
    of a discrete index, the vocabulary, the model's output space or the
    expert a router picks, go to AdamW, which is the split four labs
    cross-confirmed. Everything else is a matrix and goes to Muon.

    An undeclared matrix of rank two takes Linen's kernel convention,
    contracting axis 0 into axis 1. An undeclared parameter of higher rank
    raises: its matrix axes are exactly what this spec cannot guess, and
    orthogonalizing the wrong pair would show up as a worse loss curve and
    nothing else.

    Optax reads one spec tree shaped like the parameters and treats a None
    leaf as an AdamW parameter (optax/contrib/_muon.py:660-675).
    """
    def leaf(path, param):
        if param.ndim < 2 or getattr(path[-1], 'key', None) == 'bias':
            return None
        axes = distributed.logical_axes(path, param.ndim)
        if axes is None:
            if param.ndim > 2:
                raise ValueError(
                    f"{jax.tree_util.keystr(path)} has rank {param.ndim} and no "
                    "declared logical axes, so Muon cannot tell which of its axes "
                    "form the matrix. Declare it in DEFAULT_LOGICAL_PARAM_AXES.")
            return optax.contrib.MuonDimensionNumbers()
        if SELECTION_AXES & set(axes) or axes[-1] in BATCH_AXES:
            return None
        return optax.contrib.MuonDimensionNumbers(*_matrix_sides(path, axes))
    return jax.tree_util.tree_map_with_path(leaf, params)


def _muon_groups(learning_rate, **opts):
    """Muon over the matrices and AdamW over everything else, one schedule."""
    return optax.contrib.muon(
        learning_rate,
        muon_weight_dimension_numbers=muon_weight_dimension_numbers,
        **opts)


OPTIMIZER_MAP = {
    'adam': optax.adam,
    'adamw': optax.adamw,
    'lamb': optax.lamb,
    'muon': _muon_groups,
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
        if config.optimizer == 'muon':
            # Weight decay reaches the AdamW group too, which is where the
            # norm scales live, the one place Moonlight calls it crucial
            # for stability (docs/research/frontier-training.md:184).
            opts.setdefault('adam_weight_decay', config.weight_decay)
    solver = OPTIMIZER_MAP[config.optimizer](learning_rate, **opts)

    if config.clip_grads > 0:
        solver = optax.chain(optax.clip_by_global_norm(config.clip_grads), solver)
    if config.grad_accum_steps > 1:
        # Accumulate over several micro-batches so the effective batch can
        # exceed what fits in device memory at once.
        solver = optax.MultiSteps(solver, every_k_schedule=config.grad_accum_steps)
    return solver


