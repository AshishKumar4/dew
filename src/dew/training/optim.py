"""The optimizer a recipe builds from an OptimConfig.

Every recipe builds the same solver: the config's learning-rate schedule
when it names one, weight decay passed in the optimizer's own keyword
arguments, and global-norm clipping. That is library behavior, so the
pieces are here and the recipes call them. The Trainer pools and normalizes
the gradient over the whole accumulation window before it calls this solver.

The 'muon' entry uses the production parameter-group split that the labs
converged on (docs/research/frontier-training.md). AdamW updates the
embeddings, the head, the router and the norms, and Muon updates the
matrices.

`optax.contrib.muon` combines the two optimizers, applying each to its own
group of parameters. Dew supplies the parameter spec, which says which group
a parameter belongs to and which of its axes form the matrix.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
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
# (docs/research/frontier-training.md:257). An expert dimension is one of
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
    # optax partitions the parameters into the two groups and masks each
    # optimizer to its own (optax/contrib/_muon.py:694).
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


def _qk_normed(params) -> set[tuple[str, ...]]:
    """The layers whose parameters hold a query or key norm (`q_norm`,
    `k_norm`), read off the tree's paths."""
    layers = set()
    for path, _ in jax.tree_util.tree_leaves_with_path(params):
        names = _dict_names(path)
        if len(names) >= 2 and names[-2] in ('q_norm', 'k_norm'):
            layers.add(names[:-2])
    return layers


def _clip_leaf(qk_stats, tau: float, normed: set[tuple[str, ...]], path: jax.tree_util.KeyPath,
               update: jax.Array, param: jax.Array) -> jax.Array:
    """Fold QK-Clip's post-step weight rescale into one leaf's update.

    The clip acts on the weights after the step, `gamma * (W + update)`.
    Written as an update that is `(gamma - 1) * W + gamma * update`, so the
    optimizer chain can apply it after Muon and land on the same weights
    MaxText's post-step rescale writes.

    Leaves outside `QK_PROJECTIONS` keep their update. A named leaf whose
    layer sowed no maxima raises naming the layer, since stepping it
    unclipped would train a different model than the maxima describe. So
    does one whose layer norms its queries or keys (`normed`): the norm
    divides any scale of the kernels back out, so no rescale of them moves
    the logits, which the norm's own scales bound instead.

    Queries and keys rescale per query group, as Megatron Core's
    `SelfAttention.clip_qk` does (attention.py at core_v0.19.2), multi-head
    attention being the group of one: the group's rescale is the smallest
    of its heads', tau over its largest logit, and its query heads and its
    key head each take the square root, so every logit of the group scales
    by it. The layer sows its key-head count (`kv_heads`) and head width
    (`head_dim`) beside the maxima. An output-gated projection's head is its
    query and then its gate, and the gate keeps its weights: it is no
    logit, and the paper rescales the query and key weights alone. Latent
    attention rescales per head instead, the rope slice of its query by the
    full gamma."""
    names = _dict_names(path)
    if len(names) < 2 or names[-1] != 'kernel' or names[-2] not in QK_PROJECTIONS:
        return update
    if names[:-2] in normed:
        raise ValueError(
            f"QK-Clip reaches {'.'.join(names)}, whose layer norms its queries and keys: the "
            "norm divides the kernels' scale back out of the logits, so the clip cannot bound "
            "them. Train this model with 'muon', or build it with qk_norm=False")
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
    if proj in ('q_proj', 'q_b_proj') and nope is not None:
        return _rescaled_update(
            update, param, _mla_query_gamma(scale, nope, last // heads), heads)
    if proj == 'kv_b_proj':
        if nope is None:
            raise ValueError(
                f"QK-Clip reaches {'.'.join(names)} with no nope width sowed; "
                "a latent key projection needs its layer's 'qk_nope'")
        return _rescaled_update(
            update, param, _mla_key_gamma(scale, nope, last // heads), heads)
    kv_heads, head_dim = _sown(node, 'kv_heads'), _sown(node, 'head_dim')
    if kv_heads is None or head_dim is None:
        raise ValueError(
            f"QK-Clip reaches {'.'.join(names)} with no key-head count or head width "
            "sowed; a query or key projection needs its layer's 'kv_heads' and 'head_dim'")
    # Query head h is in group h // (heads / kv_heads), the grouping the
    # kernels repeat the keys over, and group g's rescale lands at index g.
    # `kv_heads` stays a traced value, so only the shapes are static.
    groups = jnp.arange(heads) // (heads // kv_heads)
    grouped = jax.ops.segment_min(scale, groups, num_segments=heads)
    if proj in ('q_proj', 'q_b_proj'):
        query = jnp.arange(last // heads)[None, :] < head_dim
        gamma = jnp.where(query, jnp.sqrt(grouped[groups])[:, None], jnp.ones((), scale.dtype))
        return _rescaled_update(update, param, gamma, heads)
    columns = jnp.arange(last) // (last // kv_heads)
    return _rescaled_update(update, param, jnp.sqrt(grouped[columns])[None], 1)


def scale_by_qk_clip(tau: float = 100.0) -> optax.GradientTransformationExtraArgs:
    """Rescale the query and key projections of every head past `tau`.

    This is Kimi K2's MuonClip (arXiv 2507.20534), applied after the update,
    per query group under grouped-query attention as Megatron Core clips it.

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
            functools.partial(_clip_leaf, qk_stats, tau, _qk_normed(params)), updates, params)
        return clipped, state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _muonclip_groups(learning_rate, qk_clip_threshold: float = 100.0, **opts):
    """Chain `_muon_groups` with the QK-Clip that follows the update."""
    return optax.chain(
        _muon_groups(learning_rate, **opts),
        scale_by_qk_clip(qk_clip_threshold))


def _mix(h: jax.Array) -> jax.Array:
    """murmur3's 32-bit finaliser: every input bit reaches every output bit."""
    h = h ^ (h >> 16)
    h = h * jnp.uint32(0x85EBCA6B)
    h = h ^ (h >> 13)
    h = h * jnp.uint32(0xC2B2AE35)
    return h ^ (h >> 16)


def stochastic_round_bf16(x: jax.Array, seed: jax.Array) -> jax.Array:
    """`x` rounded to bf16 up or down with the probability of its distance to
    each, from a hash of `seed` and each element's flat index.

    Adding 16 uniform bits below the kept mantissa and truncating rounds up
    exactly when the discarded bits plus the noise carry, which is the
    discarded fraction's probability. The noise is a counter-based hash, so a
    step's rounding is a pure function of the seed and the position: no key
    is split or carried, and nothing is read from memory. jax.random.bits
    (threefry) made the whole update 1.75x slower than fp32 state on a TPU
    v6e and 1.05x on an L4; this costs a few integer ops per element. NaN
    stays NaN.
    """
    x = x.astype(jnp.float32)
    index = jax.lax.iota(jnp.uint32, x.size).reshape(x.shape)
    noise = _mix(index * jnp.uint32(0x9E3779B1) + seed.astype(jnp.uint32)) & jnp.uint32(0xFFFF)
    bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
    rounded = jax.lax.bitcast_convert_type((bits + noise) & jnp.uint32(0xFFFF0000), jnp.float32)
    return jnp.where(jnp.isnan(x), x, rounded).astype(jnp.bfloat16)


def bf16_moments(b1: float, b2: float, eps: float, eps_root: float,
                 nesterov: bool) -> optax.GradientTransformation:
    """`optax.scale_by_adam` with both moments stored in bf16.

    Each update runs optax's own update one leaf at a time, on that leaf's
    moments widened to fp32, and writes the new moments back stochastically
    rounded (`stochastic_round_bf16`), seeded by the step count, the leaf and
    the moment: the step is optax's, only the storage is Dew's. Leaf by leaf
    keeps one leaf's fp32 moments live at a time; widening the whole state
    first held all of them and raised the update's peak from 4.49 to 7.37 GB
    on the lm-dense tree (RTX 4080). Round to nearest would lose every
    increment of the second moment smaller than half its bf16 spacing, which
    at b2 = 0.999 is most of them; a stochastic rounding keeps each one in
    expectation. The state keeps optax's `ScaleByAdamState` layout, so
    sharding and checkpoints read it as they read fp32 state.
    """
    inner = optax.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root, nesterov=nesterov)

    def init_fn(params):
        def zeros(leaf):
            return jnp.zeros(leaf.shape, jnp.bfloat16)
        return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32),
                                      mu=jax.tree.map(zeros, params),
                                      nu=jax.tree.map(zeros, params))

    def update_fn(updates, state, params=None):
        del params  # scale_by_adam reads none
        step = _mix(jnp.asarray(state.count).astype(jnp.uint32) * jnp.uint32(0x27D4EB2F))
        leaves, structure = jax.tree.flatten(updates)
        mus, nus = structure.flatten_up_to(state.mu), structure.flatten_up_to(state.nu)
        scaled, new_mu, new_nu, count = [], [], [], state.count
        for index, (gradient, mu, nu) in enumerate(zip(leaves, mus, nus, strict=True)):
            # One array is a pytree of one leaf, so optax updates the leaf alone.
            update, one = inner.update(gradient, optax.ScaleByAdamState(
                count=state.count, mu=mu.astype(jnp.float32), nu=nu.astype(jnp.float32)))
            if not isinstance(one, optax.ScaleByAdamState):
                raise TypeError(f"optax.scale_by_adam returned {type(one).__name__}")
            count = one.count
            salt = _mix(step + jnp.uint32(2 * index + 1) * jnp.uint32(0x165667B1))
            scaled.append(update)
            new_mu.append(stochastic_round_bf16(jnp.asarray(one.mu), salt))
            new_nu.append(stochastic_round_bf16(jnp.asarray(one.nu),
                                                _mix(salt + jnp.uint32(0x9E3779B9))))
        return structure.unflatten(scaled), optax.ScaleByAdamState(
            count=count, mu=structure.unflatten(new_mu), nu=structure.unflatten(new_nu))

    return optax.GradientTransformation(init_fn, update_fn)


def _coupled_decay(solver: optax.GradientTransformation, weight_decay: float) -> optax.GradientTransformation:
    """`solver` after torch's `Adam(weight_decay=...)`, which adds
    `weight_decay * param` to the gradient before the moments read it.

    That is an L2 penalty on the loss, so the moments rescale the decay with
    the rest of the gradient, where adamw decays the weights after
    (torch/optim/adam.py, `_single_tensor_adam`). A zero decay returns
    `solver` itself, so its state keeps optax's layout.
    """
    if not weight_decay:
        return solver
    return optax.chain(optax.add_decayed_weights(weight_decay), solver)


def _adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0, mu_dtype=None, weight_decay=0.0,
          *, nesterov: bool = False):
    """optax.adam, with torch's coupled weight decay (`_coupled_decay`)."""
    return _coupled_decay(optax.adam(learning_rate, b1, b2, eps, eps_root, mu_dtype, nesterov=nesterov),
                         weight_decay)


def _bf16_adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0, weight_decay=0.0, *,
               nesterov: bool = False):
    """optax.adam's chain over bf16 moments, with torch's coupled weight decay (`_coupled_decay`)."""
    return _coupled_decay(optax.chain(bf16_moments(b1, b2, eps, eps_root, nesterov),
                                     optax.scale_by_learning_rate(learning_rate)), weight_decay)


def _bf16_adamw(learning_rate, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0,
                weight_decay=1e-4, mask=None, *, nesterov: bool = False):
    """optax.adamw's chain over bf16 moments."""
    return optax.chain(bf16_moments(b1, b2, eps, eps_root, nesterov),
                       optax.add_decayed_weights(weight_decay, mask),
                       optax.scale_by_learning_rate(learning_rate))


class PowerProfilesState(NamedTuple):
    """The optimizer state of `power_profiles`, kept beside the wrapped solver's state."""
    updates: jax.Array
    """The number of updates made so far, the t in the averages' power-function profiles."""
    stds: jax.Array
    """The relative standard deviation of each average."""
    averages: tuple[optax.Params, ...]
    inner: optax.OptState


def power_profiles(solver: optax.GradientTransformation,
                   stds: Sequence[float]) -> optax.GradientTransformationExtraArgs:
    """`solver`, keeping one power-function average of the parameters it
    produces per relative standard deviation in `stds`, for post-hoc EMA
    (`dew.training.posthoc`).

    Update t keeps (1 - 1/t)^(γ + 1) of each average and blends in the rest
    of the parameters it just made (Karras et al. 2024, Eq. 127). The
    averages ride in the optimizer state, so they are sharded, placed and
    skipped on a rejected step exactly as its moments are. Each checkpoint
    is also the snapshot of its step (`Checkpoints.profile_steps`), retained
    whole so the state and averages share one atomic save. Each std is rounded to fp32 first, the
    precision the state records it in.
    """
    from dew.training.posthoc import power_decay
    from dew.training.transaction import ema_update

    stds = tuple(float(np.float32(std)) for std in stds)
    if not stds:
        raise ValueError("power_profiles tracks at least one average; name its relative std")
    decays = tuple(power_decay(std) for std in stds)
    solver = optax.with_extra_args_support(solver)

    def init_fn(params):
        return PowerProfilesState(updates=jnp.zeros([], jnp.int32), stds=jnp.asarray(stds, jnp.float32),
                                  averages=tuple(jax.tree.map(jnp.copy, params) for _ in stds),
                                  inner=solver.init(params))

    def update_fn(updates, state, params=None, **extra_args):
        if params is None:
            raise ValueError("power_profiles averages the parameters, so its update needs them")
        updates, inner = solver.update(updates, state.inner, params, **extra_args)
        produced = optax.apply_updates(params, updates)
        averages = tuple(ema_update(average, produced, decay(state.updates))
                         for average, decay in zip(state.averages, decays, strict=True))
        return updates, PowerProfilesState(state.updates + 1, state.stds, averages, inner)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


# The optimizers `OptimConfig.state_dtype='bfloat16'` builds: optax.adam and
# optax.adamw with the moments stored in bf16, their other options kept.
BF16_STATE_OPTIMIZERS = {'adam': _bf16_adam, 'adamw': _bf16_adamw}


OPTIMIZER_MAP = {
    'adam': _adam,
    'adamw': optax.adamw,
    'lamb': optax.lamb,
    'muon': _muon_groups,
    'muonclip': _muonclip_groups,
}


def _keep_within(lower: float, upper: float) -> optax.GradientTransformation:
    """Shorten each update so that the parameter it moves lands within `[lower, upper]`.

    torch recipes clamp parameters in place after `optimizer.step()`, as
    DCLS's `clamp_parameters` does its kernel positions. Written as an
    update, `clip(param + update) - param`, the clamp runs inside the
    optimizer chain, so a group's projection applies to its parameters alone
    and lands on the same values to within one rounding of the subtraction.
    optax's `projections.projection_box` clamps a parameter tree outside the
    chain, where it would reach every group at once.
    """
    if not lower < upper:
        raise ValueError(f"the bounds run from a lower to a higher value, got [{lower}, {upper}]")

    def init_fn(params) -> optax.EmptyState:
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("a bounded group reads the parameters, so its update needs them")
        kept = jax.tree.map(lambda update, param: jnp.clip(param + update, lower, upper) - param,
                            updates, params)
        return kept, state

    return optax.GradientTransformation(init_fn, update_fn)


@dataclasses.dataclass(frozen=True)
class ParamGroup:
    """A group of parameters the optimizer updates with their own learning
    rate, momentum, weight decay and bounds.

    `patterns` are `fnmatch` patterns matched against a parameter's path,
    which is its dict keys joined by '/' (`layers_3/self_attn/q_proj/kernel`);
    `*` also matches '/'. A parameter belongs to the first group in
    `OptimConfig.param_groups` that has a pattern it matches, as in
    lm-engine, and a parameter that matches no group raises `ValueError`.

    The group's learning rate is its own `schedule` (the config's rate when
    None) times `learning_rate_multiplier`. `b1` schedules the first-moment
    decay of adam, adamw and lamb, as torch's `OneCycleLR(cycle_momentum=True)`
    cycles Adam's beta1; None keeps the optimizer's own. `weight_decay`
    replaces the config's weight decay; None keeps it. For 'adam' that decay
    is torch's coupled L2 (`_coupled_decay`), and for 'adamw' the decoupled
    one. `bounds` clamps each parameter into `[lower, upper]` after every
    update (`_keep_within`).
    """

    name: str
    patterns: tuple[str, ...]
    learning_rate_multiplier: float = 1.0
    weight_decay: float | None = None
    schedule: ScheduleBase | None = None
    b1: ScheduleBase | None = None
    bounds: tuple[float, float] | None = None

    def __post_init__(self):
        object.__setattr__(self, "patterns", tuple(self.patterns))
        if not self.patterns:
            raise ValueError(f"param group {self.name!r} matches no pattern")
        if self.learning_rate_multiplier <= 0:
            raise ValueError(
                f"param group {self.name!r} scales the learning rate by a positive "
                f"number, got {self.learning_rate_multiplier}")
        if self.bounds is not None:
            lower, upper = self.bounds
            object.__setattr__(self, "bounds", (lower, upper))
            if not lower < upper:
                raise ValueError(f"param group {self.name!r} bounds its parameters from a lower to a "
                                 f"higher value, got {self.bounds}")

    def solver(self, make: Callable[..., optax.GradientTransformation],
               learning_rate: float | optax.Schedule, steps: int,
               opts: Mapping[str, object]) -> optax.GradientTransformation:
        """Build this group's optimizer: `make(learning_rate, **opts)` with the
        group's `b1` schedule and `bounds`, over a run of `steps` updates.

        A `b1` schedule replaces any `b1` in `opts`, and needs a `make` that
        names a `b1` argument (adam, adamw, lamb; not muon).
        """
        if self.b1 is None:
            solver = make(learning_rate, **opts)
        else:
            parameters = inspect.signature(make).parameters
            if 'b1' not in parameters:
                raise ValueError(f"param group {self.name!r} schedules b1, and its optimizer "
                                 f"takes no b1 (adam, adamw and lamb do)")
            fixed = {name: value for name, value in opts.items() if name != 'b1'}
            # optax rebuilds the optimizer each update from the values it
            # injects. Only the rate and b1 vary; every other argument, the
            # defaults included, stays the Python value the factory branches on.
            static = tuple(name for name in parameters if name not in ('learning_rate', 'b1'))
            solver = optax.inject_hyperparams(make, static_args=static)(
                learning_rate, b1=self.b1.schedule(steps), **fixed)
        if self.bounds is not None:
            solver = optax.chain(solver, _keep_within(*self.bounds))
        return solver

    @classmethod
    def mup(cls, width_multiplier: float) -> tuple[ParamGroup, ...]:
        """Return lm-engine's muP parameter groups, in lm-engine's order.

        The groups are: norms, biases and `dt_bias` at the base rate without
        weight decay; the token embeddings at the base rate with weight decay;
        and everything else, including the router, `A_log`, `D` and the conv
        taps, at the base rate divided by `width_multiplier`.
        `width_multiplier` is lm-engine's m_width, which is the model's
        `logits_scaling`.
        """
        # lm-engine's configs/param-groups/mup.yml at 45b6b57b.
        return (
            cls("no_weight_decay", NO_DECAY_PATTERNS, weight_decay=0.0),
            cls("normal", ("*embed_tokens/*",)),
            cls("mup", ("*",), learning_rate_multiplier=1 / width_multiplier),
        )


NO_DECAY_PATTERNS = ("*/bias", "*/scale", "*norm/weight", "*/dt_bias")
"""What lm-engine's `no_weight_decay` group holds (configs/param-groups/
mup.yml at 45b6b57b): biases, every norm's weight (an RMSNorm keeps `scale`
here, Mamba-2's gated norm `weight`) and Mamba-2's `dt_bias`."""


def param_labels(groups: Sequence[ParamGroup]):
    """The `optax.multi_transform` labeller: each leaf's first matching group."""
    # The first match wins, as in lm-engine's optimization/params_group.py at
    # 45b6b57b.
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


@dataclasses.dataclass(frozen=True)
class ScheduleBase:
    """The base of the schedule records, each of which holds its own fields
    and builds an optax schedule.

    Subclasses are registered under `dew.registry.schedules`, so a run's
    record names the schedule's kind and holds only the fields that schedule
    reads. A subclass builds its values in `values`; `schedule` reads them.
    """

    every: int = dataclasses.field(default=1, kw_only=True)
    """The updates each of the schedule's own steps lasts.

    torch recipes step a scheduler once an epoch, and with `every` set to the
    updates of an epoch this one holds each value through an epoch as torch's
    does. A run of `steps` updates is then `steps // every` of the
    schedule's own steps, and every field that counts steps (`warmup_steps`,
    `decay_steps`) counts those.
    """

    def __post_init__(self):
        if self.every < 1:
            raise ValueError(f"a schedule advances once every 1 or more updates, not {self.every}")

    def schedule(self, steps: int) -> optax.Schedule:
        """Return the value at each update of a run of `steps` updates, as an optax schedule."""
        if self.every == 1:
            return self.values(steps)
        own = steps // self.every
        if own < 1:
            raise ValueError(f"a run of {steps} updates never advances a schedule that steps once "
                             f"every {self.every}")
        values, every = self.values(own), self.every
        return lambda count: values(jnp.asarray(count) // every)

    def values(self, steps: int) -> optax.Schedule:
        """Return the value at each of the schedule's own steps, over `steps` of them."""
        raise NotImplementedError


@schedules("cosine")
@dataclasses.dataclass(frozen=True)
class Cosine(ScheduleBase):
    """A linear warmup from `init` to `peak`, then a cosine decay to `end` at
    step `decay_steps`, or at the run's end when that is None.

    It is `optax.warmup_cosine_decay_schedule`.
    """

    peak: float
    warmup_steps: int = 10000
    end: float = 0.0
    init: float = 0.0
    decay_steps: int | None = None

    def values(self, steps: int) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.init, peak_value=self.peak, warmup_steps=self.warmup_steps,
            decay_steps=steps if self.decay_steps is None else self.decay_steps,
            end_value=self.end)


@dataclasses.dataclass(frozen=True)
class PowerTail:
    """The linear tail of a power schedule, from the power law's rate at step
    `start` to `end` at step `steps`, or at the run's end when that is None."""

    start: int
    steps: int | None = None
    end: float = 0.0


@schedules("power")
@dataclasses.dataclass(frozen=True)
class Power(ScheduleBase):
    """lm-engine's power schedule (`power_schedule`), a power law of the step after a linear warmup.

    After the warmup, the rate is min(peak, a * (step * c) ** b). With a
    `tail`, a linear decay follows the power law, as in the last 29% of
    Rigel's run. lm-engine's examples set `a` to 4 * batch size and `c` to
    the tokens per step.
    """

    peak: float
    warmup_steps: int
    a: float
    b: float = -0.51
    c: float = 1.0
    tail: PowerTail | None = None

    def values(self, steps: int) -> optax.Schedule:
        tail = self.tail
        if tail is None:
            return power_schedule(self.peak, self.warmup_steps, self.a, self.b, self.c)
        return power_schedule(
            self.peak, self.warmup_steps, self.a, self.b, self.c, decay_start=tail.start,
            decay_end=steps if tail.steps is None else tail.steps, end_value=tail.end)


@schedules("linear")
@dataclasses.dataclass(frozen=True)
class Linear(ScheduleBase):
    """lm-engine's linear schedule (`linear_schedule`), with a warmup, a constant rate and a linear decay.

    The rate warms up from zero to `peak`, stays constant until
    `decay_start` (the end of the warmup when None), then falls linearly to
    `end` at `decay_steps` (the run's end when None).
    """

    peak: float
    warmup_steps: int = 0
    decay_start: int | None = None
    decay_steps: int | None = None
    end: float = 0.0

    def values(self, steps: int) -> optax.Schedule:
        return linear_schedule(self.peak, self.warmup_steps, self.decay_start,
                               steps if self.decay_steps is None else self.decay_steps,
                               end_value=self.end)


@schedules("one_cycle")
@dataclasses.dataclass(frozen=True)
class OneCycle(ScheduleBase):
    """torch's `OneCycleLR` with its default two phases and cosine annealing.

    The value follows a half cosine from `init` to `peak` at step
    `warmup_fraction * steps - 1`, then another to `end` at the run's last
    step, `steps - 1`, and holds `end` after it, where torch raises. `init`
    is torch's `max_lr / div_factor` (`peak / 25` when None) and `end` its
    `init / final_div_factor` (`init / 1e4` when None). Adam's momentum cycle
    under `cycle_momentum=True` is `OneCycle(peak=0.85, init=0.95, end=0.95)`
    as a group's `b1`.

    optax's `cosine_onecycle_schedule` places its phase boundaries at
    `pct_start * steps` and `steps`, one step later than torch's
    (torch/optim/lr_scheduler.py, `OneCycleLR.__init__`), so its rate peaks
    one step late and it cannot reproduce a torch run.
    """

    peak: float
    init: float | None = None
    end: float | None = None
    warmup_fraction: float = 0.3

    def values(self, steps: int) -> optax.Schedule:
        init = self.peak / 25 if self.init is None else self.init
        end = init / 1e4 if self.end is None else self.end
        # torch's phase ends, kept as floats as torch keeps them.
        rise, last = self.warmup_fraction * steps - 1, steps - 1
        if not 0 < rise < last:
            raise ValueError(f"a one-cycle schedule over {steps} steps with warmup fraction "
                             f"{self.warmup_fraction} has no rise or no fall; give it more steps")

        def anneal(start: float, stop: float, done: jax.Array) -> jax.Array:
            return stop + (start - stop) / 2 * (jnp.cos(jnp.pi * done) + 1)

        def value(count) -> jax.Array:
            step = jnp.minimum(jnp.asarray(count, jnp.float32), last)
            return jnp.where(step <= rise, anneal(init, self.peak, step / rise),
                             anneal(self.peak, end, (step - rise) / (last - rise)))

        return value


@schedules("exponential")
@dataclasses.dataclass(frozen=True)
class Exponential(ScheduleBase):
    """A geometric decay from `offset + init` to `offset + end` over
    `decay_steps` steps (the run's when None), then constant.

    Up to `decay_steps`, the value at step t is
    `offset + init * (end / init) ** (t / decay_steps)`. That is torch's
    `ExponentialLR` with `gamma = (end / init) ** (1 / decay_steps)`, stepped
    `decay_steps` times and then no more, plus `offset`. It is
    `optax.exponential_decay` with `end` as its bound. SNN-delays anneals
    DCLS's kernel width this way, the raw width decaying and a constant
    added to it.
    """

    init: float
    end: float
    decay_steps: int | None = None
    offset: float = 0.0

    def values(self, steps: int) -> optax.Schedule:
        span = steps if self.decay_steps is None else self.decay_steps
        if span <= 0 or self.init <= 0 or self.end <= 0:
            raise ValueError(f"an exponential decay needs positive steps, init and end, got "
                             f"{span}, {self.init} and {self.end}")
        decay = optax.exponential_decay(self.init, span, self.end / self.init, end_value=self.end)
        if not self.offset:
            return decay
        return lambda count: self.offset + decay(count)


def learning_rate_schedule(config: OptimConfig, steps: int):
    """The rate `config` names: its schedule over a `steps`-update run, or
    the constant `learning_rate` when it names none."""
    return config.learning_rate if config.schedule is None else config.schedule.schedule(steps)


__all__ = ["Cosine", "Exponential", "Linear", "OneCycle", "ParamGroup", "Power", "PowerProfilesState",
           "PowerTail", "ScheduleBase"]
