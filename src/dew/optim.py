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
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.hashing import fmix32
from dew.nn.precision import at_least_fp32
from dew.nn.sharding import LogicalAxes, declarations, declared_axes

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


def weight_decay_mask(params):
    """Mark every parameter but the biases and the vectors, which weight decay spares by default:
    nanoGPT's split (model.py, `GPT.configure_optimizers`), with Megatron Core's and timm's `bias`
    of any rank (core/optimizer/__init__.py, optim/_param_groups.py), a per-head one included."""
    return jax.tree_util.tree_map_with_path(lambda path, param: param.ndim >= 2 and not (
        isinstance(path[-1], jax.tree_util.DictKey) and path[-1].key == "bias"), params)


def muon_weight_dimension_numbers(params):
    """Build a `MuonDimensionNumbers` per parameter, None where AdamW steps in.

    Which group a parameter lands in is read off the logical axes its module
    declares (`dew.nn.sharding`), the same table the sharding derivation
    reads, so one declaration answers both questions.

    AdamW takes a parameter of rank below two, a bias, and a parameter that
    maps into or out of a discrete index: the vocabulary, the model's output
    space, or the expert a router picks. That is the split four labs
    cross-confirmed. Everything else is a matrix and goes to Muon.

    An undeclared matrix of rank two, or one declared with no axis named
    (replicated, as Gemma 3n's AltUp matrices are), takes Linen's kernel
    convention, contracting axis 0 into axis 1. An undeclared parameter of higher rank
    raises, because its matrix axes are what this spec cannot guess, and
    orthogonalizing the wrong pair would show up as a worse loss curve.

    Optax reads one spec tree shaped like the parameters and treats a None
    leaf as an AdamW parameter (optax/contrib/_muon.py:660-675).
    """
    declared = declarations(params)

    def leaf(path: jax.tree_util.KeyPath, param: jax.Array, decays: bool):
        if not decays:
            return None
        axes = declared_axes(path, param.ndim, declared)
        # A declaration that names no axis says how the parameter shards (whole, on every device),
        # not which of its axes form the matrix, so it reads as an undeclared one (AltUp's).
        if axes is None or all(axis is None for axis in axes):
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
    return jax.tree_util.tree_map_with_path(leaf, params, weight_decay_mask(params))


def _muon_groups(learning_rate, weight_decay=0.0, adam_weight_decay=None, **opts):
    """Run Muon over the matrices and AdamW over the rest, on one schedule. The AdamW group decays
    its norms and biases too, at `adam_weight_decay` or else `weight_decay`: Moonlight shares the
    decay among matrix and non-matrix parameters, RMSNorm's included (arXiv 2502.16982, 2.2)."""
    # optax partitions the parameters into the two groups and masks each
    # optimizer to its own (optax/contrib/_muon.py:694).
    return optax.contrib.muon(
        learning_rate, weight_decay=weight_decay,
        adam_weight_decay=weight_decay if adam_weight_decay is None else adam_weight_decay,
        muon_weight_dimension_numbers=muon_weight_dimension_numbers, **opts)


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


def _columns(gamma: jax.Array, split: int, last: int) -> jax.Array:
    """`gamma` over `split` heads of a leaf's last axis, `last` wide, as one factor a column: the
    `[last]` vector a step's rescale holds, the same shape however the clip groups the heads."""
    return jnp.broadcast_to(jnp.asarray(gamma, jnp.float32).reshape(split, -1),
                            (split, last // split)).reshape(last)


def _rescaled(stepped: jax.Array, columns: jax.Array) -> jax.Array:
    """`stepped`, a leaf after its step, rescaled by its float32 `columns` factors along its last axis:
    the product formed at float32 or wider and rounded to the leaf's dtype once, the paper's
    `gamma * (W + update)` (Megatron Core's `clip_qk`, MaxText's `_clip_mla_weight`)."""
    return (stepped.astype(at_least_fp32(stepped.dtype)) * columns).astype(stepped.dtype)


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
               param: jax.Array) -> jax.Array:
    """QK-Clip's post-step rescale of one query or key leaf (`_rescales` picks them), a factor a
    column of its last axis (`_columns`).

    The clip acts on the weights after the step, `gamma * (W + update)`, and the step applies it
    there (`stepped`): an update that would land on it, `(gamma - 1) * W + gamma * update`, rounds at
    `W`'s scale twice where the product rounds once (ledger 1659).

    A projection's kernel and its bias rescale together, column for column, so
    a biased projection's queries and keys scale as an unbiased one's do.
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
        return _columns(_mla_query_gamma(scale, nope, last // heads), heads, last)
    if proj == 'kv_b_proj':
        if nope is None:
            raise ValueError(
                f"QK-Clip reaches {'.'.join(names)} with no nope width sowed; "
                "a latent key projection needs its layer's 'qk_nope'")
        return _columns(_mla_key_gamma(scale, nope, last // heads), heads, last)
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
        return _columns(gamma, heads, last)
    columns = jnp.arange(last) // (last // kv_heads)
    return jnp.sqrt(grouped[columns]).astype(jnp.float32)


class QKClipState(NamedTuple):
    """The rescale QK-Clip leaves for the stepped weights (`stepped`): each query and key leaf's factor
    a column of its last axis by its path, every factor one before the first update or where no
    maxima were sown, so the state keeps one structure."""

    rescales: Mapping[str, jax.Array]


def scale_by_qk_clip(tau: float = 100.0) -> optax.GradientTransformationExtraArgs:
    """Rescale the query and key projections of every head past `tau`.

    This is Kimi K2's MuonClip (arXiv 2507.20534), applied after the update,
    per query group under grouped-query attention as Megatron Core clips it.
    The update passes through unchanged and the rescale rides the state
    (`QKClipState`), which `stepped` applies to the weights the update moved:
    one rounding of `gamma * (W + update)`, the paper's.

    The per-head maxima arrive as `qk_stats`, the `qk` collection the model
    sowed, which the trainer forwards from the loss's `Aux`. Without them
    the transform steps aside, leaving every other optimizer and every run
    whose loss never opened the collection on its old update."""
    if tau <= 0:
        raise ValueError(f"the clip threshold bounds positive logits, got {tau}")

    def init_fn(params) -> QKClipState:
        return QKClipState(_rescales(params, None, tau))

    def update_fn(updates, state, params=None, qk_stats=None):
        if params is None:
            return updates, state
        return updates, QKClipState(_rescales(params, qk_stats, tau))

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _rescales(params, qk_stats, tau: float) -> dict[str, jax.Array]:
    """Each query and key leaf's factor a column by its path (`_clip_leaf`), every factor one where no
    maxima were sown, so the state has one structure from its first step to its last."""
    normed = _qk_normed(params) if qk_stats is not None else set()
    held = {}
    for path, param in jax.tree_util.tree_leaves_with_path(params):
        names = _dict_names(path)
        if len(names) < 2 or names[-1] not in ('kernel', 'bias') or names[-2] not in QK_PROJECTIONS:
            continue
        held[jax.tree_util.keystr(path)] = (jnp.ones(param.shape[-1], jnp.float32) if qk_stats is None
                                             else _clip_leaf(qk_stats, tau, normed, path, param))
    return held


def stepped(params, updates, opt_state):
    """`params` moved by `updates`, then each leaf the optimizer's state holds a post-step rescale for
    (`QKClipState`, anywhere in a chain's state) rescaled once: the weights a step stores, and the
    only way to form them (the trainer's transaction, `power_profiles`' averages)."""
    moved = optax.apply_updates(params, updates)
    rescales: dict[str, jax.Array] = {}
    for held in jax.tree_util.tree_leaves(opt_state, is_leaf=lambda node: isinstance(node, QKClipState)):
        if isinstance(held, QKClipState):
            rescales.update(held.rescales)
    if not rescales:
        return moved
    lacking = rescales.keys() - {jax.tree_util.keystr(path)
                                 for path, _ in jax.tree_util.tree_leaves_with_path(moved)}
    if lacking:
        raise ValueError(f"the optimizer's state rescales {sorted(lacking)}, which the stepped "
                         "parameters lack: step the tree the optimizer was updated with")

    def rescaled(path, leaf):
        key = jax.tree_util.keystr(path)
        return _rescaled(leaf, rescales[key]) if key in rescales else leaf

    return jax.tree_util.tree_map_with_path(rescaled, moved)


def _muonclip_groups(learning_rate, qk_clip_threshold: float = 100.0, **opts):
    """Chain `_muon_groups` with the QK-Clip that follows the update."""
    return optax.chain(
        _muon_groups(learning_rate, **opts),
        scale_by_qk_clip(qk_clip_threshold))


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
    noise = fmix32(index * jnp.uint32(0x9E3779B1) + seed.astype(jnp.uint32)) & jnp.uint32(0xFFFF)
    bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
    rounded = jax.lax.bitcast_convert_type((bits + noise) & jnp.uint32(0xFFFF0000), jnp.float32)
    return jnp.where(jnp.isnan(x), x, rounded).astype(jnp.bfloat16)


def _bias_corrected(moment, decay, count):
    """`optax.tree.bias_correction`, with the correction of a Python `decay` in (0, 1) formed as
    -expm1(count log decay), its log taken in double before the trace.

    Optax forms 1 - decay**count inside a jit, where a Python decay is already rounded to float32:
    0.999 becomes 0.99900001, so the correction at count 1 is 1.3e-5 off (2.2e-4 at 0.9999), where
    torch forms it in double. A decay given as an array, a schedule's (`scaled_solver`), takes
    optax's own. google-deepmind/optax#1827 makes this optax's; delete this, and
    `scale_by_adam` with it, once a release carries it.
    """
    if isinstance(decay, (int, float)) and 0 < decay < 1:
        correction = -jnp.expm1(count * math.log(decay))
        return jax.tree.map(lambda leaf: leaf / correction.astype(leaf.dtype), moment)
    return optax.tree.bias_correction(moment, decay, count)


def scale_by_adam(b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0, mu_dtype=None, *,
                  nesterov: bool = False, nu_dtype=None) -> optax.GradientTransformation:
    """`optax.scale_by_adam` (optax 0.2.8) with its bias corrections formed in double
    (`_bias_corrected`): optax's moments, state layout (`optax.ScaleByAdamState`) and update,
    composed from optax's public tree math. `nu_dtype` stores the second moment as `mu_dtype` does
    the first, each the parameter's dtype where None; a gradient narrower than its moment is
    widened before either reads it, so an fp16 gradient's square does not underflow first."""
    mu_dtype = None if mu_dtype is None else jax.dtypes.canonicalize_dtype(mu_dtype)
    nu_dtype = None if nu_dtype is None else jax.dtypes.canonicalize_dtype(nu_dtype)

    def init_fn(params):
        return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32),
                                      mu=optax.tree.zeros_like(params, dtype=mu_dtype),
                                      nu=optax.tree.zeros_like(params, dtype=nu_dtype))

    def update_fn(updates, state, params=None):
        del params  # Adam reads only the gradients
        updates = jax.tree.map(lambda update, moment: None if update is None else update.astype(
            jnp.promote_types(update.dtype, moment.dtype)), updates, state.nu,
            is_leaf=lambda leaf: leaf is None)
        mu = optax.tree.update_moment(updates, state.mu, b1, 1)
        nu = optax.tree.update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count = optax.safe_increment(state.count)
        if nesterov:
            mu_hat = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g,
                                  _bias_corrected(mu, b1, optax.safe_increment(count)),
                                  _bias_corrected(updates, b1, count))
        else:
            mu_hat = _bias_corrected(mu, b1, count)
        nu_hat = _bias_corrected(nu, b2, count)
        scaled = jax.tree.map(lambda m, v: None if m is None else m / (jnp.sqrt(v + eps_root) + eps),
                              mu_hat, nu_hat, is_leaf=lambda leaf: leaf is None)
        return scaled, optax.ScaleByAdamState(count=count, mu=optax.tree.cast(mu, mu_dtype),
                                              nu=optax.tree.cast_like(nu, state.nu))

    return optax.GradientTransformation(init_fn, update_fn)


def bf16_moments(b1: float, b2: float, eps: float, eps_root: float,
                 nesterov: bool) -> optax.GradientTransformation:
    """`scale_by_adam` with both moments stored in bf16.

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
    inner = scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root, nesterov=nesterov)

    def init_fn(params):
        def zeros(leaf):
            return jnp.zeros(leaf.shape, jnp.bfloat16)
        return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32),
                                      mu=jax.tree.map(zeros, params),
                                      nu=jax.tree.map(zeros, params))

    def update_fn(updates, state, params=None):
        del params  # scale_by_adam reads none
        step = fmix32(jnp.asarray(state.count).astype(jnp.uint32) * jnp.uint32(0x27D4EB2F))
        leaves, structure = jax.tree.flatten(updates)
        mus, nus = structure.flatten_up_to(state.mu), structure.flatten_up_to(state.nu)
        scaled, new_mu, new_nu, count = [], [], [], state.count
        for index, (gradient, mu, nu) in enumerate(zip(leaves, mus, nus, strict=True)):
            # One array is a pytree of one leaf, so optax updates the leaf alone.
            update, one = inner.update(gradient, optax.ScaleByAdamState(
                count=state.count, mu=mu.astype(jnp.float32), nu=nu.astype(jnp.float32)))
            if not isinstance(one, optax.ScaleByAdamState):
                raise TypeError(f"scale_by_adam returned {type(one).__name__}")
            count = one.count
            salt = fmix32(step + jnp.uint32(2 * index + 1) * jnp.uint32(0x165667B1))
            scaled.append(update)
            new_mu.append(stochastic_round_bf16(jnp.asarray(one.mu), salt))
            new_nu.append(stochastic_round_bf16(jnp.asarray(one.nu),
                                                fmix32(salt + jnp.uint32(0x9E3779B9))))
        return structure.unflatten(scaled), optax.ScaleByAdamState(
            count=count, mu=structure.unflatten(new_mu), nu=structure.unflatten(new_nu))

    return optax.GradientTransformation(init_fn, update_fn)


def _coupled_decay(solver: optax.GradientTransformation, weight_decay: float,
                   mask=None) -> optax.GradientTransformation:
    """`solver` after torch's `Adam(weight_decay=...)`, which adds
    `weight_decay * param` to the gradient before the moments read it.

    That is an L2 penalty on the loss, so the moments rescale the decay with
    the rest of the gradient, where adamw decays the weights after
    (torch/optim/adam.py, `_single_tensor_adam`). A zero decay returns
    `solver` itself, so its state keeps optax's layout.
    """
    if not weight_decay:
        return solver
    return optax.chain(optax.add_decayed_weights(weight_decay, mask), solver)


def _adam_chain(moments: optax.GradientTransformation, learning_rate, weight_decay, mask, *,
                coupled: bool) -> optax.GradientTransformation:
    """An Adam variant over `moments` (`scale_by_adam`'s, or `bf16_moments`' stored in bf16): optax.adamw's
    chain, its decay decoupled, or optax.adam's with torch's coupled weight decay (`_coupled_decay`)."""
    if coupled:
        return _coupled_decay(optax.chain(moments, optax.scale_by_learning_rate(learning_rate)), weight_decay,
                              mask)
    return optax.chain(moments, optax.add_decayed_weights(weight_decay, mask),
                       optax.scale_by_learning_rate(learning_rate))


def _adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0, mu_dtype=None, weight_decay=0.0,
          mask=None, *, nesterov: bool = False, nu_dtype=None):
    """optax.adam over `scale_by_adam`, with torch's coupled weight decay."""
    return _adam_chain(scale_by_adam(b1, b2, eps, eps_root, mu_dtype, nesterov=nesterov, nu_dtype=nu_dtype),
                       learning_rate, weight_decay, mask, coupled=True)


def _adamw(learning_rate, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0, mu_dtype=None, weight_decay=1e-4,
           mask=None, *, nesterov: bool = False, nu_dtype=None):
    """optax.adamw over `scale_by_adam`."""
    return _adam_chain(scale_by_adam(b1, b2, eps, eps_root, mu_dtype, nesterov=nesterov, nu_dtype=nu_dtype),
                       learning_rate, weight_decay, mask, coupled=False)


def _bf16_adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0, weight_decay=0.0, mask=None, *,
               nesterov: bool = False):
    """optax.adam over bf16 moments, with torch's coupled weight decay."""
    return _adam_chain(bf16_moments(b1, b2, eps, eps_root, nesterov), learning_rate, weight_decay, mask,
                       coupled=True)


def _bf16_adamw(learning_rate, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0,
                weight_decay=1e-4, mask=None, *, nesterov: bool = False):
    """optax.adamw over bf16 moments."""
    return _adam_chain(bf16_moments(b1, b2, eps, eps_root, nesterov), learning_rate, weight_decay, mask,
                       coupled=False)


class PowerProfilesState(NamedTuple):
    """The optimizer state of `power_profiles`, kept beside the wrapped solver's state."""
    updates: jax.Array
    """The number of updates made so far, the t in the averages' power-function profiles."""
    stds: jax.Array
    """The relative standard deviation of each average."""
    powers: jax.Array
    """Each average's γ + 1, the power its decay raises (1 - 1/t) to, solved
    from its std when the state starts: a resumed average decays as it was
    tracked, whatever stds the resuming run configures."""
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
    from dew.training.posthoc import exponent, powered_decay
    from dew.training.transaction import ema_update

    stds = tuple(float(np.float32(std)) for std in stds)
    if not stds:
        raise ValueError("power_profiles tracks at least one average; name its relative std")
    solver = optax.with_extra_args_support(solver)

    def init_fn(params):
        return PowerProfilesState(updates=jnp.zeros([], jnp.int32), stds=jnp.asarray(stds, jnp.float32),
                                  powers=jnp.asarray([exponent(std) + 1 for std in stds], jnp.float32),
                                  averages=tuple(jax.tree.map(jnp.copy, params) for _ in stds),
                                  inner=solver.init(params))

    def update_fn(updates, state, params=None, **extra_args):
        if params is None:
            raise ValueError("power_profiles averages the parameters, so its update needs them")
        updates, inner = solver.update(updates, state.inner, params, **extra_args)
        produced = stepped(params, updates, inner)
        averages = tuple(ema_update(average, produced, powered_decay(state.powers[index])(state.updates))
                         for index, average in enumerate(state.averages))
        return updates, PowerProfilesState(state.updates + 1, state.stds, state.powers, averages, inner)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


Optimizer = Literal['adam', 'adamw', 'lamb', 'muon', 'muonclip']
"""The optimizers `OptimConfig.optimizer` names, each built by its `OPTIMIZER_MAP` entry."""

# The optimizers `OptimConfig.state_dtype='bfloat16'` builds: optax.adam and
# optax.adamw with the moments stored in bf16, their other options kept.
BF16_STATE_OPTIMIZERS: dict[Optimizer, Callable[..., optax.GradientTransformation]] = {
    'adam': _bf16_adam, 'adamw': _bf16_adamw}


OPTIMIZER_MAP: dict[Optimizer, Callable[..., optax.GradientTransformation]] = {
    'adam': _adam,
    'adamw': _adamw,
    'lamb': optax.lamb,
    'muon': _muon_groups,
    'muonclip': _muonclip_groups,
}


RATE_SCALE, RATE_SHAPE = 'learning_rate_scale', 'learning_rate_shape'
"""The hyperparameters a `scaled_solver` injects: its rate's scale, and its
schedule's shape at the last update, whose product is the rate."""


def scaled_solver(make: Callable[..., optax.GradientTransformation], scale: float,
                  shape: ScheduleBase | None, steps: int | None, opts: Mapping[str, object],
                  b1: ScheduleBase | None = None) -> optax.GradientTransformation:
    """`make(learning_rate, **opts)` at the rate `scale` times `shape`'s
    schedule (1 when None), with the scale an operand in the optimizer's
    state: runs at different rates of one shape lower to one executable.

    Optax injects the scale, the shape's value at each update and a `b1`
    schedule (`optax.inject_hyperparams`), and rebuilds the factory from
    their traced values; every other option keeps the Python value the
    factory branches on. The rate reaches the factory as a schedule, so
    Optax casts it to each update leaf's dtype, and the injected values are
    at least float32 on a bf16 tree. The factory is built once here too, so
    a bad option raises now rather than at the state's init.
    """
    rate = 1.0 if shape is None else shape.schedule(steps)
    make(rate, **opts)

    def factory(learning_rate_scale, learning_rate_shape, b1):
        given = opts if b1 is None else {**opts, 'b1': b1}
        return make(lambda count: learning_rate_scale * learning_rate_shape, **given)

    def injected(dtype):
        return optax.inject_hyperparams(factory, hyperparam_dtype=dtype)(
            **{RATE_SCALE: scale, RATE_SHAPE: rate}, b1=None if b1 is None else b1.schedule(steps))

    def init_fn(params):
        return injected(at_least_fp32(optax.tree.dtype(params, 'highest'))).init(params)

    def update_fn(updates, state, params=None, **extra_args):
        return injected(state.hyperparams[RATE_SCALE].dtype).update(updates, state, params, **extra_args)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def scaled(rate: float | ScheduleBase) -> tuple[float, ScheduleBase | None]:
    """A rate as a scale and the schedule of its ratios to it (None, a
    constant rate): a schedule's first nonzero rate-valued field, and the
    schedule with every rate-valued field (`metadata={'rate': True}`, nested
    records such as PowerTail too) divided by it. Timings, powers and the
    rates' ratios stay in the shape, so runs at different scales share it.
    A schedule of zeros has the scale 0 and a shape whose first rate field
    is 1, so it applies exactly zero. A None field keeps its derived
    default (OneCycle's init and end). A schedule that marks no field a
    rate, a subclass implementing `values` alone, is its own shape at the
    scale 1: its values are the rates."""
    if not isinstance(rate, ScheduleBase):
        return rate, None

    def rates(record):
        for field in dataclasses.fields(record):
            value = getattr(record, field.name)
            if field.metadata.get('rate') and value is not None:
                yield value
            elif dataclasses.is_dataclass(value):
                yield from rates(value)

    found = list(rates(rate))
    if not found:
        return 1.0, rate
    scale = next((value for value in found if value != 0), 0.0)

    def normalized(record):
        fields = {}
        for field in dataclasses.fields(record):
            value = getattr(record, field.name)
            if field.metadata.get('rate') and value is not None:
                fields[field.name] = value / (scale or 1.0)
            elif dataclasses.is_dataclass(value):
                fields[field.name] = normalized(value)
        return dataclasses.replace(record, **fields)

    shape = normalized(rate)
    first = next((field.name for field in dataclasses.fields(rate) if field.metadata.get('rate')), None)
    if scale == 0 and first is not None:
        shape = dataclasses.replace(shape, **{first: 1.0})
    return scale, shape


def learning_rate(opt_state: optax.OptState) -> jax.typing.ArrayLike | None:
    """Return the learning rate stored in `opt_state`, if the optimizer stores one.

    An OptimConfig's solver holds a rate scale and the schedule's value at
    the last applied update (`dew.optim.scaled_solver`), whose
    product is the rate; an Optax optimizer can inject `learning_rate`
    itself (`optax.inject_hyperparams`). The result is None when the
    optimizer closes over the rate, or when parameter groups hold several
    rates.
    """
    injected = (optax.InjectHyperparamsState, optax.InjectStatefulHyperparamsState)
    rates = []
    for node in jax.tree.leaves(opt_state, is_leaf=lambda node: isinstance(node, injected)):
        if not isinstance(node, injected):
            continue
        if RATE_SCALE in node.hyperparams:
            rates.append(node.hyperparams[RATE_SCALE] * node.hyperparams[RATE_SHAPE])
        elif "learning_rate" in node.hyperparams:
            rates.append(node.hyperparams["learning_rate"])
    return rates[0] if len(rates) == 1 else None


def with_rate_scales(opt_state: optax.OptState, scales: Mapping[str | None, float]) -> optax.OptState:
    """`opt_state` with each `scaled_solver`'s rate scale set to `scales`'
    value for its param group, by name, or under None for a solver of no
    group; each new scale keeps the old one's dtype and placement."""
    def replaced(path: jax.tree_util.KeyPath, leaf):
        if not path or path[-1] != jax.tree_util.DictKey(RATE_SCALE):
            return leaf
        group = next((entry.key for entry in path
                      if isinstance(entry, jax.tree_util.DictKey) and entry.key in scales), None)
        return jax.device_put(np.asarray(scales[group], leaf.dtype), leaf.sharding)
    return jax.tree_util.tree_map_with_path(replaced, opt_state)


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

    The group's learning rate is its own `schedule` (the config's `learning_rate` when
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

    def learning_rate(self, learning_rate: float | ScheduleBase) -> tuple[float, ScheduleBase | None]:
        """This group's rate scale and shape (`scaled`): its own `schedule`,
        else the config's `learning_rate`, times `learning_rate_multiplier`."""
        scale, shape = scaled(learning_rate if self.schedule is None else self.schedule)
        return scale * self.learning_rate_multiplier, shape

    def solver(self, make: Callable[..., optax.GradientTransformation],
               learning_rate: float | ScheduleBase, steps: int | None,
               opts: Mapping[str, object]) -> optax.GradientTransformation:
        """Build this group's optimizer: `make` at the group's `learning_rate`, its
        scale held in state (`scaled_solver`), with the group's `b1`
        schedule and `bounds`, over a run of `steps` updates.

        A `b1` schedule replaces any `b1` in `opts`, and needs a `make` that
        names a `b1` argument (adam, adamw, lamb; not muon).
        """
        if self.b1 is not None and 'b1' not in inspect.signature(make).parameters:
            raise ValueError(f"param group {self.name!r} schedules b1, and its optimizer "
                             f"takes no b1 (adam, adamw and lamb do)")
        solver = scaled_solver(make, *self.learning_rate(learning_rate), steps, opts, self.b1)
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

    A run's record names the schedule's class and holds only the fields that
    schedule reads; `dew.registry.schedules` holds the short aliases. A
    subclass builds its values in `values`; `schedule` reads them.
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

    def schedule(self, steps: int | None) -> optax.Schedule:
        """Return the value at each update of a run of `steps` updates, as an optax schedule."""
        if self.every == 1:
            return self.values(steps)
        own = None if steps is None else steps // self.every
        if own is not None and own < 1:
            raise ValueError(f"a run of {steps} updates never advances a schedule that steps once "
                             f"every {self.every}")
        values, every = self.values(own), self.every
        return lambda count: values(jnp.asarray(count) // every)

    def _horizon(self, steps: int | None, field: str) -> int:
        """Resolve a decay's step horizon, which a token budget cannot supply."""
        if steps is None:
            raise ValueError(f"{type(self).__name__} needs {field} or trainer.steps as its step horizon")
        return steps

    def values(self, steps: int | None) -> optax.Schedule:
        """Return the value at each of the schedule's own steps, over `steps` of them."""
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class Cosine(ScheduleBase):
    """A linear warmup from `init_value` to `peak_value`, then a cosine decay to `end_value` at
    step `decay_steps`, or at the run's end when that is None.

    It is `optax.warmup_cosine_decay_schedule`.
    `warmup_steps` must be smaller than the decay horizon in schedule steps.
    """

    peak_value: float = dataclasses.field(metadata={'rate': True})
    warmup_steps: int = 10000
    end_value: float = dataclasses.field(default=0.0, metadata={'rate': True})
    init_value: float = dataclasses.field(default=0.0, metadata={'rate': True})
    decay_steps: int | None = None

    def values(self, steps: int | None) -> optax.Schedule:
        decay_steps = self._horizon(steps if self.decay_steps is None else self.decay_steps, "decay_steps")
        if self.warmup_steps >= decay_steps:
            raise ValueError(f"Cosine needs warmup_steps={self.warmup_steps} smaller than "
                             f"decay_steps={decay_steps}")
        return optax.warmup_cosine_decay_schedule(
            init_value=self.init_value, peak_value=self.peak_value,
            warmup_steps=self.warmup_steps,
            decay_steps=decay_steps,
            end_value=self.end_value)


@dataclasses.dataclass(frozen=True)
class PowerTail:
    """The linear tail of a power schedule, from the power law's rate at step
    `start` to `end_value` at step `steps`, or at the run's end when that is None."""

    start: int
    steps: int | None = None
    end_value: float = dataclasses.field(default=0.0, metadata={'rate': True})


@dataclasses.dataclass(frozen=True)
class Power(ScheduleBase):
    """lm-engine's power schedule (`power_schedule`), a power law of the step after a linear warmup.

    After the warmup, the rate is min(peak_value, a * (step * c) ** b). With a
    `tail`, a linear decay follows the power law, as in the last 29% of
    Rigel's run. lm-engine's examples set `a` to 4 * batch size and `c` to
    the tokens per step.
    """

    peak_value: float = dataclasses.field(metadata={'rate': True})
    warmup_steps: int
    a: float = dataclasses.field(metadata={'rate': True})
    b: float = -0.51
    c: float = 1.0
    tail: PowerTail | None = None

    def values(self, steps: int | None) -> optax.Schedule:
        tail = self.tail
        if tail is None:
            return power_schedule(self.peak_value, self.warmup_steps, self.a, self.b, self.c)
        return power_schedule(
            self.peak_value, self.warmup_steps, self.a, self.b, self.c, decay_start=tail.start,
            decay_end=self._horizon(steps if tail.steps is None else tail.steps, "tail.steps"),
            end_value=tail.end_value)


@dataclasses.dataclass(frozen=True)
class Linear(ScheduleBase):
    """lm-engine's linear schedule (`linear_schedule`), with a warmup, a constant rate and a linear decay.

    The rate warms up from zero to `peak_value`, stays constant until
    `decay_start` (the end of the warmup when None), then falls linearly to
    `end_value` at `decay_steps` (the run's end when None).
    """

    peak_value: float = dataclasses.field(metadata={'rate': True})
    warmup_steps: int = 0
    decay_start: int | None = None
    decay_steps: int | None = None
    end_value: float = dataclasses.field(default=0.0, metadata={'rate': True})

    def values(self, steps: int | None) -> optax.Schedule:
        return linear_schedule(self.peak_value, self.warmup_steps, self.decay_start,
                               self._horizon(steps if self.decay_steps is None else self.decay_steps,
                                             "decay_steps"),
                               end_value=self.end_value)


@dataclasses.dataclass(frozen=True)
class OneCycle(ScheduleBase):
    """torch's `OneCycleLR` with its default two phases and cosine annealing.

    The value follows a half cosine from `init_value` to `peak_value` at step
    `warmup_fraction * steps - 1`, then another to `end_value` at the run's last
    step, `steps - 1`, and holds `end_value` after it, where torch raises. `init_value`
    is torch's `max_lr / div_factor` (`peak_value / 25` when None) and `end_value` its
    `init_value / final_div_factor` (`init_value / 1e4` when None). Adam's momentum cycle
    under `cycle_momentum=True` is `OneCycle(peak_value=0.85, init_value=0.95,
    end_value=0.95)` as a group's `b1`.

    optax's `cosine_onecycle_schedule` places its phase boundaries at
    `pct_start * steps` and `steps`, one step later than torch's
    (torch/optim/lr_scheduler.py, `OneCycleLR.__init__`), so its rate peaks
    one step late and it cannot reproduce a torch run.
    """

    peak_value: float = dataclasses.field(metadata={'rate': True})
    init_value: float | None = dataclasses.field(default=None, metadata={'rate': True})
    end_value: float | None = dataclasses.field(default=None, metadata={'rate': True})
    warmup_fraction: float = 0.3

    def values(self, steps: int | None) -> optax.Schedule:
        steps = self._horizon(steps, "trainer.steps")
        init = self.peak_value / 25 if self.init_value is None else self.init_value
        end = init / 1e4 if self.end_value is None else self.end_value
        # torch's phase ends, kept as floats as torch keeps them.
        rise, last = self.warmup_fraction * steps - 1, steps - 1
        if not 0 < rise < last:
            raise ValueError(f"a one-cycle schedule over {steps} steps with warmup fraction "
                             f"{self.warmup_fraction} has no rise or no fall; give it more steps")

        def anneal(start: float, stop: float, done: jax.Array) -> jax.Array:
            return stop + (start - stop) / 2 * (jnp.cos(jnp.pi * done) + 1)

        def value(count) -> jax.Array:
            step = jnp.minimum(jnp.asarray(count, jnp.float32), last)
            return jnp.where(step <= rise, anneal(init, self.peak_value, step / rise),
                             anneal(self.peak_value, end, (step - rise) / (last - rise)))

        return value


@dataclasses.dataclass(frozen=True)
class Exponential(ScheduleBase):
    """A geometric decay from `offset + init_value` to `offset + end_value` over
    `decay_steps` steps (the run's when None), then constant.

    Up to `decay_steps`, the value at step t is
    `offset + init_value * (end_value / init_value) ** (t / decay_steps)`. That is torch's
    `ExponentialLR` with `gamma = (end_value / init_value) ** (1 / decay_steps)`, stepped
    `decay_steps` times and then no more, plus `offset`. It is
    `optax.exponential_decay` with `end_value` as its bound. SNN-delays anneals
    DCLS's kernel width this way, the raw width decaying and a constant
    added to it.
    """

    init_value: float = dataclasses.field(metadata={'rate': True})
    end_value: float = dataclasses.field(metadata={'rate': True})
    decay_steps: int | None = None
    offset: float = dataclasses.field(default=0.0, metadata={'rate': True})

    def values(self, steps: int | None) -> optax.Schedule:
        span = self._horizon(steps if self.decay_steps is None else self.decay_steps, "decay_steps")
        if span <= 0 or self.init_value <= 0 or self.end_value <= 0:
            raise ValueError(f"an exponential decay needs positive steps, init_value and end_value, got "
                             f"{span}, {self.init_value} and {self.end_value}")
        decay = optax.exponential_decay(self.init_value, span, self.end_value / self.init_value,
                                        end_value=self.end_value)
        if not self.offset:
            return decay
        return lambda count: self.offset + decay(count)


__all__ = ["Cosine", "Exponential", "Linear", "OneCycle", "ParamGroup", "Power", "PowerProfilesState",
           "PowerTail", "QKClipState", "ScheduleBase", "stepped"]
