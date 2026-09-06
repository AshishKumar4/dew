"""The optimizer a recipe builds from an OptimConfig.

Every recipe wires the same solver: a warmup-cosine schedule when one is
asked for, weight decay folded into the optimizer's own kwargs, and
global-norm clipping. That wiring is library behavior, so it lives here and
the recipes call it. Gradient accumulation is the Trainer's, which wraps the
solver in `optax.MultiSteps`.

The 'muon' entry is the production parameter-group split the labs converged
on (docs/research/frontier-training.md:183): AdamW on the embeddings, the
head, the router and the norms, Muon on the matrices. `optax.contrib.muon`
owns the masked composition (it partitions with `optax.masked` per
group, optax/contrib/_muon.py:694), so what Dew supplies is the parameter
spec that says which group a parameter belongs to and which of its axes are
the matrix.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax

from dew.nn.sharding import LogicalAxes, declared_axes

if TYPE_CHECKING:
    from dew.config import OptimConfig

# Attention stores its projections either as one matrix over the flattened
# head space or as a dimension per head, so the head dimensions count as one
# side of the matrix and both layouts get the same orthogonalized update.
HEAD_AXES = frozenset({'heads', 'head_dim', 'kv'})

# An expert dimension stacks whole matrices, one per expert, so it is a batch
# axis: it names neither side, and optax orthogonalizes each expert on its
# own (optax/contrib/_muon.py:56-74).
BATCH_AXES = frozenset({'exp'})

# A parameter that maps into or out of a discrete index is a lookup, so AdamW
# keeps the embeddings, the head and the router
# (docs/research/frontier-training.md:183). An expert dimension is one of
# these when it is the output, where it counts the experts a router scores,
# and a batch axis when it leads, where it stacks one matrix per expert.
SELECTION_AXES = frozenset({'vocab', 'output'})


def _matrix_sides(path: jax.tree_util.KeyPath, axes: LogicalAxes) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The contracted axes and the output axes of a declared parameter.

    A dimension continues the side before it when the declaration leaves it
    unnamed, as the spatial dimensions of a patch embedding are, or when it
    and its predecessor are both head dimensions. What is left has to be two
    sides, one contracted and one output, as MaxText's per-name
    table produces for its own trees (maxtext utils/muon_utils.py:100-175).
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

    Which group a parameter lands in is read off the logical axes its module
    declares (`dew.nn.sharding`), the table the sharding derivation reads,
    so one declaration answers both questions. A parameter of rank
    below two, a bias, and a parameter that maps into or out of a discrete
    index, the vocabulary, the model's output space or the expert a router
    picks, go to AdamW, which is the split four labs cross-confirmed.
    Everything else is a matrix and goes to Muon.

    An undeclared matrix of rank two takes Linen's kernel convention,
    contracting axis 0 into axis 1. An undeclared parameter of higher rank
    raises. Its matrix axes are what this spec cannot guess, and
    orthogonalizing the wrong pair would show up as a worse loss curve.

    Optax reads one spec tree shaped like the parameters and treats a None
    leaf as an AdamW parameter (optax/contrib/_muon.py:660-675).
    """
    def leaf(path: jax.tree_util.KeyPath, param: jax.Array) -> optax.contrib.MuonDimensionNumbers | None:
        last = path[-1]
        if param.ndim < 2 or (isinstance(last, jax.tree_util.DictKey) and last.key == "bias"):
            return None
        axes = declared_axes(path, param.ndim)
        if axes is None:
            if param.ndim > 2:
                raise ValueError(
                    f"{jax.tree_util.keystr(path)} has rank {param.ndim} and no "
                    "declared logical axes, so Muon cannot tell which of its axes "
                    "form the matrix. Declare it with @logical_axes on its module.")
            return optax.contrib.MuonDimensionNumbers()
        named = {axis for axis in axes if axis is not None}
        if SELECTION_AXES & named or axes[-1] in BATCH_AXES:
            return None
        return optax.contrib.MuonDimensionNumbers(*_matrix_sides(path, axes))
    return jax.tree_util.tree_map_with_path(leaf, params)


def _muon_groups(learning_rate, **opts):
    """Muon over the matrices and AdamW over everything else, one schedule."""
    return optax.contrib.muon(
        learning_rate,
        muon_weight_dimension_numbers=muon_weight_dimension_numbers,
        **opts)


QK_PROJECTIONS = frozenset({'q_proj', 'q_b_proj', 'k_proj', 'kv_b_proj'})
"""The projection names the clip rescales: the query and key projections of
grouped-query attention and, for latent attention, the per-head output
projections. Anything else keeps its update untouched."""


def _dict_names(path: jax.tree_util.KeyPath) -> tuple[str, ...]:
    """The dict keys along `path`, dropping the sequence indices a stacked
    view never produces on a parameter tree."""
    return tuple(entry.key for entry in path
                 if isinstance(entry, jax.tree_util.DictKey))


def _sown(node, name: str):
    """The array a sowed collection holds under `name`, past its one-tuple."""
    value = node.get(name) if isinstance(node, Mapping) else None
    if value is None:
        return None
    return value[0] if isinstance(value, (tuple, list)) else value


def _clip_scale(s_max: jax.Array, tau: float) -> jax.Array:
    """Per-head rescale: `min(1, tau / s)` past the threshold, 1.0 elsewhere.

    A head whose every logit is non-positive clips nothing: MaxText's
    formula reads `minimum(1, tau / (s + 1e-6))`, which goes negative there
    and would flip the weights' sign, so Dew holds those heads at 1.0."""
    return jnp.where(s_max > 0, jnp.minimum(1.0, tau / (s_max + 1e-6)), 1.0)


def _query_widths(params) -> dict[tuple[str, ...], int]:
    """Each module's query projection width, read off static shapes: what a
    key projection's head count is measured against."""
    widths = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        names = _dict_names(path)
        if (len(names) >= 2 and names[-1] == 'kernel'
                and names[-2] in ('q_proj', 'q_b_proj')):
            widths[tuple(names[:-2])] = leaf.shape[-1]
    return widths


def _rescaled_update(update: jax.Array, param: jax.Array, gamma: jax.Array,
                     split: int) -> jax.Array:
    """The update whose application rescales the stepped weights by `gamma`:
    `(gamma - 1) * param + gamma * update`, over `split` heads."""
    width = param.shape[-1] // split
    shape = param.shape[:-1] + (split, width)
    gamma = jnp.asarray(gamma, update.dtype)
    out = (gamma - 1) * param.reshape(shape) + gamma * update.reshape(shape)
    return out.reshape(param.shape)


def _mla_query_gamma(scale: jax.Array, nope: jax.Array, width: int) -> jax.Array:
    """`[heads, width]`: `sqrt` on the nope slice, full on the rope slice.
    The comparison keeps the sowed width dynamic; only the shape is static."""
    is_nope = jnp.arange(width)[None, :] < nope
    return jnp.where(is_nope, jnp.sqrt(scale)[:, None], scale[:, None])


def _mla_key_gamma(scale: jax.Array, nope: jax.Array, width: int) -> jax.Array:
    """`[heads, width]`: `sqrt` on the nope slice, 1.0 on the values."""
    is_nope = jnp.arange(width)[None, :] < nope
    return jnp.where(is_nope, jnp.sqrt(scale)[:, None],
                     jnp.ones((), scale.dtype))


def _clip_leaf(qk_stats, tau: float, qdims: dict[tuple[str, ...], int],
               path: jax.tree_util.KeyPath, update: jax.Array,
               param: jax.Array) -> jax.Array:
    """`update` rescaled the way the post-update weights rescale.

    The clip fires on the weights after the step: `gamma * (W + update)`.
    Written as an update, `(gamma - 1) * W + gamma * update`, so the chain
    runs it after Muon and the applied tree lands on the same weights
    MaxText's post-step rescale writes. Leaves outside `QK_PROJECTIONS`
    keep their update; a named leaf whose layer sowed no maxima raises
    naming the layer, since stepping it unclipped would train a different
    model than the maxima describe.

    Queries rescale per head over the whole projection, the gate half of an
    output-gated projection with its query head, and the latent query's rope
    slice by the full gamma. Keys rescale per head where the head count is
    known from static shapes: equal query and key widths are multi-head
    attention. Anything else is grouped-query attention, whose per-group
    minima need the group count where only dynamic values cross. There the
    whole projection takes the layer's strongest head, the conservative
    side: it never under-clips."""
    names = _dict_names(path)
    if len(names) < 2 or names[-1] != 'kernel' or names[-2] not in QK_PROJECTIONS:
        return update
    node = qk_stats
    for name in names[:-2]:
        node = node.get(name) if isinstance(node, Mapping) else None
        if node is None:
            break
    logged = _sown(node, 'max_logits')
    if logged is None:
        raise ValueError(
            f"QK-Clip reaches {'.'.join(names)} but its layer sowed no max "
            "logits; the model has to sow them under the 'qk' collection "
            "for the clip to know what fires")
    s_max = jnp.max(jnp.asarray(logged, jnp.float32), axis=0)
    heads = s_max.shape[0]
    scale = _clip_scale(s_max, tau)
    last = param.shape[-1]
    if last % heads:
        raise ValueError(
            f"QK-Clip cannot split {'.'.join(names)} of width {last} into "
            f"{heads} heads")
    proj = names[-2]
    nope = _sown(node, 'qk_nope')
    if proj in ('q_proj', 'q_b_proj'):
        if nope is None:
            return _rescaled_update(update, param, jnp.sqrt(scale)[:, None],
                                    heads)
        return _rescaled_update(
            update, param, _mla_query_gamma(scale, nope, last // heads), heads)
    if proj == 'kv_b_proj':
        if nope is None:
            raise ValueError(
                f"QK-Clip reaches {'.'.join(names)} with no nope width sowed; "
                "a latent key projection needs its layer's 'qk_nope'")
        return _rescaled_update(
            update, param, _mla_key_gamma(scale, nope, last // heads), heads)
    width = qdims.get(tuple(names[:-2]))
    if width is not None and width == last:
        return _rescaled_update(update, param, jnp.sqrt(scale)[:, None], heads)
    scoped = jnp.min(scale).astype(update.dtype)
    return (scoped - 1) * param + scoped * update

def scale_by_qk_clip(tau: float = 100.0) -> optax.GradientTransformationExtraArgs:
    """QK-Clip after the update: heads past `tau` rescale their query and
    key projections, Kimi K2's MuonClip (arXiv 2507.20534).

    The per-head maxima arrive as `qk_stats`, the `qk` collection the model
    sowed, which the trainer forwards from the loss's `Aux`. Without them
    the transform steps aside, leaving every other optimizer and every run
    whose loss never opened the collection on its old update."""
    if tau <= 0:
        raise ValueError(f"the clip threshold bounds positive logits, got {tau}")

    def init_fn(params) -> optax.EmptyState:
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None, qk_stats=None):
        if qk_stats is None or params is None:
            return updates, state
        clipped = jax.tree_util.tree_map_with_path(
            functools.partial(_clip_leaf, qk_stats, tau, _query_widths(params)),
            updates, params)
        return clipped, state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _muonclip_groups(learning_rate, qk_clip_threshold: float = 100.0, **opts):
    """Muon over the matrices, AdamW over the rest, and the QK-Clip after."""
    return optax.chain(
        _muon_groups(learning_rate, **opts),
        scale_by_qk_clip(qk_clip_threshold))


OPTIMIZER_MAP = {
    'adam': optax.adam,
    'adamw': optax.adamw,
    'lamb': optax.lamb,
    'muon': _muon_groups,
    'muonclip': _muonclip_groups,
}


def build_optimizer(config: "OptimConfig", steps: int) -> optax.GradientTransformation:
    """The solver, with its schedule and clipping; `steps` is the run's length,
    which a cosine schedule decays over unless the config names its own."""
    learning_rate = config.learning_rate
    if config.learning_rate_schedule == 'cosine':
        decay_steps = (steps if config.learning_rate_decay_steps is None
                       else config.learning_rate_decay_steps)
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate, peak_value=config.learning_rate_peak,
            warmup_steps=config.learning_rate_warmup_steps,
            decay_steps=decay_steps,
            end_value=config.learning_rate_end,
        )
    opts = dict(config.optimizer_opts)
    if config.weight_decay is not None:
        opts['weight_decay'] = config.weight_decay
        if config.optimizer in ('muon', 'muonclip'):
            # Weight decay reaches the AdamW group as well. That is where the
            # norm scales live, and Moonlight calls it crucial for stability
            # there (docs/research/frontier-training.md:184).
            opts.setdefault('adam_weight_decay', config.weight_decay)
    solver = OPTIMIZER_MAP[config.optimizer](learning_rate, **opts)

    if config.clip_grads > 0:
        solver = optax.chain(optax.clip_by_global_norm(config.clip_grads), solver)
    return solver
