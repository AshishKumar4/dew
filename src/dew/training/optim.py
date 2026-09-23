"""The optimizer a recipe builds from an OptimConfig.

Every recipe wires the same solver: a warmup-cosine schedule when one is
asked for, weight decay folded into the optimizer's own kwargs, and
global-norm clipping. That wiring is library behavior, so it lives here and
the recipes call it. The Trainer forms the normalized effective-window
gradient before calling this solver.

The 'muon' entry is the production parameter-group split the labs converged
on (docs/research/frontier-training.md:183). AdamW takes the embeddings, the
head, the router and the norms; Muon takes the matrices.

`optax.contrib.muon` owns the masked composition, partitioning with
`optax.masked` per group (optax/contrib/_muon.py:694). What Dew supplies is
the parameter spec: which group a parameter belongs to, and which of its
axes are the matrix.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import functools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax

from dew.nn.sharding import LogicalAxes, declared_axes
from dew.registry import schedules

if TYPE_CHECKING:
    from dew.config import OptimConfig

# Attention stores its projections either as one matrix over the flattened
# head space or as a dimension per head. The head dimensions count as one
# side of the matrix, so both layouts get the same orthogonalized update.
HEAD_AXES = frozenset({'heads', 'head_dim', 'kv'})

# An expert dimension stacks whole matrices, one per expert, so it is a batch
# axis. It names neither side, and optax orthogonalizes each expert on its
# own (optax/contrib/_muon.py:56-74).
BATCH_AXES = frozenset({'exp'})

# A parameter that maps into or out of a discrete index is a lookup, so AdamW
# keeps the embeddings, the head and the router
# (docs/research/frontier-training.md:183). An expert dimension is one of
# these when it is the output, counting the experts a router scores, and a
# batch axis when it leads, stacking one matrix per expert.
SELECTION_AXES = frozenset({'vocab', 'output'})


def _matrix_sides(path: jax.tree_util.KeyPath, axes: LogicalAxes) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split a declared parameter's axes into a contracted and an output side.

    A dimension continues the side before it when the declaration leaves it
    unnamed, as the spatial dimensions of a patch embedding are. It also
    continues that side when it and its predecessor are both head
    dimensions. What is left has to be two sides, one contracted and one
    output, as MaxText's per-name table produces for its own trees
    (maxtext utils/muon_utils.py:100-175).
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
    """Build a `MuonDimensionNumbers` per parameter, None where AdamW steps in.

    Which group a parameter lands in is read off the logical axes its module
    declares (`dew.nn.sharding`), the same table the sharding derivation
    reads, so one declaration answers both questions.

    AdamW takes a parameter of rank below two, a bias, and a parameter that
    maps into or out of a discrete index: the vocabulary, the model's output
    space, or the expert a router picks. That is the split four labs
    cross-confirmed. Everything else is a matrix and goes to Muon.

    An undeclared matrix of rank two takes Linen's kernel convention,
    contracting axis 0 into axis 1. An undeclared parameter of higher rank
    raises, because its matrix axes are what this spec cannot guess, and
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
    """Run Muon over the matrices and AdamW over the rest, on one schedule."""
    return optax.contrib.muon(
        learning_rate,
        muon_weight_dimension_numbers=muon_weight_dimension_numbers,
        **opts)


QK_PROJECTIONS = frozenset({'q_proj', 'q_b_proj', 'k_proj', 'kv_b_proj'})
"""The projection names the clip rescales: the query and key projections of
grouped-query attention and, for latent attention, the per-head output
projections. Anything else keeps its update untouched."""


def _dict_names(path: jax.tree_util.KeyPath) -> tuple[str, ...]:
    """List the dict keys along `path`, dropping the sequence indices a stacked
    view never produces on a parameter tree."""
    return tuple(entry.key for entry in path
                 if isinstance(entry, jax.tree_util.DictKey))


def _sown(node, name: str):
    """Read the array a sowed collection holds under `name`, past its one-tuple."""
    value = node.get(name) if isinstance(node, Mapping) else None
    if value is None:
        return None
    return value[0] if isinstance(value, (tuple, list)) else value


def _clip_scale(s_max: jax.Array, tau: float) -> jax.Array:
    """Compute each head's rescale: `min(1, tau / s)` past `tau`, else 1.0.

    A head whose every logit is non-positive clips nothing: MaxText's
    formula reads `minimum(1, tau / (s + 1e-6))`, which goes negative there
    and would flip the weights' sign, so Dew holds those heads at 1.0."""
    return jnp.where(s_max > 0, jnp.minimum(1.0, tau / (s_max + 1e-6)), 1.0)


def _query_widths(params) -> dict[tuple[str, ...], int]:
    """Read each module's query-projection width off the static shapes.

    A key projection's head count is measured against it."""
    widths = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        names = _dict_names(path)
        if (len(names) >= 2 and names[-1] == 'kernel'
                and names[-2] in ('q_proj', 'q_b_proj')):
            widths[tuple(names[:-2])] = leaf.shape[-1]
    return widths


def _rescaled_update(update: jax.Array, param: jax.Array, gamma: jax.Array,
                     split: int) -> jax.Array:
    """Rescale one update so that applying it rescales the weights by `gamma`.

    That update is `(gamma - 1) * param + gamma * update`, taken over
    `split` heads."""
    width = param.shape[-1] // split
    shape = (*param.shape[:-1], split, width)
    gamma = jnp.asarray(gamma, update.dtype)
    out = (gamma - 1) * param.reshape(shape) + gamma * update.reshape(shape)
    return out.reshape(param.shape)


def _mla_query_gamma(scale: jax.Array, nope: jax.Array, width: int) -> jax.Array:
    """Build the per-head rescale for a latent query projection.

    Returns `[heads, width]`. The no-positional-embedding slice takes
    `sqrt(scale)`, because query and key each carry half the clip; the rotary
    slice takes the full `scale`, since no key rescales with it. `nope` stays
    a traced value, so only the shape is static."""
    is_nope = jnp.arange(width)[None, :] < nope
    return jnp.where(is_nope, jnp.sqrt(scale)[:, None], scale[:, None])


def _mla_key_gamma(scale: jax.Array, nope: jax.Array, width: int) -> jax.Array:
    """Build the per-head rescale for a latent key and value projection.

    Returns `[heads, width]`. The no-positional-embedding slice takes
    `sqrt(scale)`, half the clip; the value slice keeps 1.0, since no logit
    passes through it."""
    is_nope = jnp.arange(width)[None, :] < nope
    return jnp.where(is_nope, jnp.sqrt(scale)[:, None],
                     jnp.ones((), scale.dtype))


def _clip_leaf(qk_stats, tau: float, qdims: dict[tuple[str, ...], int],
               path: jax.tree_util.KeyPath, update: jax.Array,
               param: jax.Array) -> jax.Array:
    """Fold QK-Clip's post-step weight rescale into one leaf's update.

    The clip acts on the weights after the step, `gamma * (W + update)`.
    Written as an update that is `(gamma - 1) * W + gamma * update`, so the
    optimizer chain can apply it after Muon and land on the same weights
    MaxText's post-step rescale writes.

    Leaves outside `QK_PROJECTIONS` keep their update. A named leaf whose
    layer sowed no maxima raises naming the layer, since stepping it
    unclipped would train a different model than the maxima describe.

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
    """Rescale the query and key projections of every head past `tau`.

    This is Kimi K2's MuonClip (arXiv 2507.20534), applied after the update.

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
    """Chain `_muon_groups` with the QK-Clip that follows the update."""
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


@dataclasses.dataclass(frozen=True)
class ParamGroup:
    """Parameters the optimizer moves at their own learning rate and decay.

    `patterns` are `fnmatch` patterns over a parameter's path, its dict keys
    joined by '/' (`layers_3/self_attn/q_proj/kernel`); `*` crosses '/'. A
    parameter joins the first group of `OptimConfig.param_groups` a pattern
    of which it matches, lm-engine's rule (optimization/params_group.py at
    45b6b57b), and one that matches none raises. The group's learning rate
    is the schedule's times `learning_rate_multiplier`; `weight_decay`
    replaces the config's, None keeping it.
    """

    name: str
    patterns: tuple[str, ...]
    learning_rate_multiplier: float = 1.0
    weight_decay: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "patterns", tuple(self.patterns))
        if not self.patterns:
            raise ValueError(f"param group {self.name!r} matches no pattern")
        if self.learning_rate_multiplier <= 0:
            raise ValueError(
                f"param group {self.name!r} scales the learning rate by a positive "
                f"number, got {self.learning_rate_multiplier}")


NO_DECAY_PATTERNS = ("*/bias", "*/scale", "*norm/weight", "*/dt_bias")
"""What lm-engine's `no_weight_decay` group holds (configs/param-groups/
mup.yml at 45b6b57b): biases, every norm's weight (an RMSNorm keeps `scale`
here, Mamba-2's gated norm `weight`) and Mamba-2's `dt_bias`."""


def mup_param_groups(width_multiplier: float) -> tuple[ParamGroup, ...]:
    """lm-engine's muP parameter groups (configs/param-groups/mup.yml at
    45b6b57b), in its order: norms, biases and `dt_bias` without weight
    decay at the base rate; the token embeddings at the base rate with decay;
    everything else, the router, `A_log`, `D` and the conv taps included, at
    the base rate divided by `width_multiplier` (lm-engine's m_width, the
    model's `logits_scaling`)."""
    return (
        ParamGroup("no_weight_decay", NO_DECAY_PATTERNS, weight_decay=0.0),
        ParamGroup("normal", ("*embed_tokens/*",)),
        ParamGroup("mup", ("*",), learning_rate_multiplier=1 / width_multiplier),
    )


def param_labels(groups: Sequence[ParamGroup]):
    """The `optax.multi_transform` labeller: each leaf's first matching group."""
    def labels(params):
        def label(path: jax.tree_util.KeyPath, _) -> str:
            name = "/".join(_dict_names(path))
            for group in groups:
                if any(fnmatch.fnmatchcase(name, pattern) for pattern in group.patterns):
                    return group.name
            raise ValueError(
                f"parameter {name} matches no param group's patterns; add a "
                f"catch-all group, patterns=('*',), if that is intended")
        return jax.tree_util.tree_map_with_path(label, params)
    return labels


def power_schedule(peak: float, warmup_steps: int, a: float, b: float, c: float = 1.0,
                   decay_start: int | None = None, decay_end: int | None = None,
                   end_value: float = 0.0) -> optax.Schedule:
    """lm-engine's power scheduler (optimization/lr_scheduler/power.py at
    45b6b57b) with an optional linear tail.

    Past the warmup the rate is `min(peak, a * (step * c) ** b)`: the power
    law of the batch size and step, arXiv 2408.13359, capped at `peak` (the
    optimizer's own rate there). The warmup rises linearly from zero to that
    value at `warmup_steps`. `decay_start` set follows the law to that step
    and then decays linearly to `end_value` at `decay_end`, which is how
    Rigel ends its run; lm-engine's scheduler has no tail.
    """
    def law(step):
        return jnp.minimum(peak, a * (jnp.asarray(step, jnp.float32) * c) ** b)

    warm = min(peak, a * (warmup_steps * c) ** b) if warmup_steps else peak
    pieces = [optax.linear_schedule(0.0, warm, warmup_steps)] if warmup_steps else []
    pieces.append(lambda count: law(count + warmup_steps))
    boundaries = [warmup_steps] if warmup_steps else []
    if decay_start is not None:
        if decay_end is None or not warmup_steps <= decay_start < decay_end:
            raise ValueError(
                f"the linear tail runs from decay_start past the warmup to a later "
                f"decay_end, got {decay_start} to {decay_end} after {warmup_steps} warmup steps")
        pieces.append(optax.linear_schedule(float(law(decay_start)), end_value,
                                            decay_end - decay_start))
        boundaries.append(decay_start)
    return optax.join_schedules(pieces, boundaries) if boundaries else pieces[0]


def linear_schedule(peak: float, warmup_steps: int, decay_start: int | None,
                    decay_end: int, end_value: float = 0.0) -> optax.Schedule:
    """lm-engine's linear scheduler (lr_scheduler/linear.py at 45b6b57b): from
    zero to `peak` over the warmup, constant to `decay_start` (None: the
    warmup's end), then linear to `end_value` at `decay_end`."""
    start = warmup_steps if decay_start is None else decay_start
    if not warmup_steps <= start < decay_end:
        raise ValueError(
            f"the linear decay runs from past the warmup to a later step, got "
            f"{start} to {decay_end} after {warmup_steps} warmup steps")
    return optax.join_schedules([
        optax.linear_schedule(0.0, peak, warmup_steps),
        optax.constant_schedule(peak),
        optax.linear_schedule(peak, end_value, decay_end - start),
    ], [warmup_steps, start])


def _scaled(learning_rate, multiplier: float):
    if multiplier == 1.0:
        return learning_rate
    if callable(learning_rate):
        return lambda count: multiplier * learning_rate(count)
    return multiplier * learning_rate


class ScheduleBase:
    """One learning-rate schedule's record: its own fields and its optax
    schedule. Registered under `dew.registry.schedules`, so a run's record
    names its kind and holds no field another schedule reads."""

    def schedule(self, steps: int) -> optax.Schedule:
        """The rate at each update of a `steps`-update run."""
        raise NotImplementedError


@schedules("cosine")
@dataclasses.dataclass(frozen=True)
class Cosine(ScheduleBase):
    """Linear warmup from `init` to `peak`, cosine to `end` at `decay_steps`
    (None: the run's end); `optax.warmup_cosine_decay_schedule`."""

    peak: float
    warmup_steps: int = 10000
    end: float = 0.0
    init: float = 0.0
    decay_steps: int | None = None

    def schedule(self, steps: int) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.init, peak_value=self.peak, warmup_steps=self.warmup_steps,
            decay_steps=steps if self.decay_steps is None else self.decay_steps,
            end_value=self.end)


@schedules("power")
@dataclasses.dataclass(frozen=True)
class Power(ScheduleBase):
    """lm-engine's power scheduler, `power_schedule`: warmup, then
    min(peak, a * (step * c) ** b), and from `decay_start` a linear tail to
    `end` at `decay_steps` (None: the run's end). lm-engine's examples take
    `a` = 4 * batch size and `c` = tokens per step."""

    peak: float
    warmup_steps: int
    a: float
    b: float = -0.51
    c: float = 1.0
    decay_start: int | None = None
    decay_steps: int | None = None
    end: float = 0.0

    def schedule(self, steps: int) -> optax.Schedule:
        return power_schedule(
            self.peak, self.warmup_steps, self.a, self.b, self.c, decay_start=self.decay_start,
            decay_end=(None if self.decay_start is None
                       else steps if self.decay_steps is None else self.decay_steps),
            end_value=self.end)


@schedules("linear")
@dataclasses.dataclass(frozen=True)
class Linear(ScheduleBase):
    """lm-engine's linear scheduler, `linear_schedule`: warmup to `peak`,
    constant to `decay_start` (None: the warmup's end), linear to `end` at
    `decay_steps` (None: the run's end)."""

    peak: float
    warmup_steps: int = 0
    decay_start: int | None = None
    decay_steps: int | None = None
    end: float = 0.0

    def schedule(self, steps: int) -> optax.Schedule:
        return linear_schedule(self.peak, self.warmup_steps, self.decay_start,
                               steps if self.decay_steps is None else self.decay_steps,
                               end_value=self.end)


def learning_rate_schedule(config: OptimConfig, steps: int):
    """The rate `config` names: its schedule over a `steps`-update run, or
    the constant `learning_rate` when it names none."""
    return config.learning_rate if config.schedule is None else config.schedule.schedule(steps)


def build_optimizer(config: OptimConfig, steps: int) -> optax.GradientTransformation:
    """Build the solver a config describes, with its schedule, parameter
    groups and clipping.

    `steps` is the run's length, which a schedule decays over unless the
    config names its own end. `param_groups` runs one solver per group under
    `optax.multi_transform`, each on the schedule times its multiplier and
    with its own weight decay; the global-norm clip still reads every
    gradient together, before the groups split them."""
    learning_rate = learning_rate_schedule(config, steps)
    opts = dict(config.optimizer_opts)
    if config.weight_decay is not None:
        opts['weight_decay'] = config.weight_decay
        if config.optimizer in ('muon', 'muonclip'):
            # Muon's weight_decay does not cover the AdamW group's norm scales.
            opts.setdefault('adam_weight_decay', config.weight_decay)
    if config.param_groups:
        names = [group.name for group in config.param_groups]
        if len(set(names)) != len(names):
            raise ValueError(f"param group names repeat: {names}")
        solvers = {}
        for group in config.param_groups:
            group_opts = dict(opts)
            if group.weight_decay is not None:
                group_opts['weight_decay'] = group.weight_decay
                if config.optimizer in ('muon', 'muonclip'):
                    group_opts['adam_weight_decay'] = group.weight_decay
            solvers[group.name] = OPTIMIZER_MAP[config.optimizer](
                _scaled(learning_rate, group.learning_rate_multiplier), **group_opts)
        solver = optax.multi_transform(solvers, param_labels(config.param_groups))
    else:
        solver = OPTIMIZER_MAP[config.optimizer](learning_rate, **opts)

    if config.clip_grads > 0:
        solver = optax.chain(optax.clip_by_global_norm(config.clip_grads), solver)
    return solver
