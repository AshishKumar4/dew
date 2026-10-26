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
from dew.nn.kernels.generation import backend
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
    """The rescale QK-Clip leaves for the stepped weights (`stepped`): the updates the clip has taken
    part in, `steps`, and each query and key leaf's factor a column of its last axis by its path,
    every factor one before the first update or where no maxima were sown, so the state keeps one
    structure."""

    steps: jax.Array
    rescales: Mapping[str, jax.Array]


class BoundsState(NamedTuple):
    """The clamp a bounded param group leaves for the stepped weights (`stepped`, `_keep_within`): the
    updates the group has taken part in, `steps`, and each of its leaves' `[lower, upper]` in the
    leaf's dtype, by its path."""

    steps: jax.Array
    bounds: Mapping[str, jax.Array]


def clips_qk(opt_state) -> bool:
    """Whether an optimizer state holds a QK-Clip's (`QKClipState`), however the optimizer composes it:
    a run then asks its objective for the step's attention logit maxima (`dew.objectives.Step.qk_stats`)
    and hands them to the update."""
    return any(isinstance(node, QKClipState) for node in jax.tree.leaves(
        opt_state, is_leaf=lambda node: isinstance(node, QKClipState)))


def scale_by_qk_clip(tau: float = 100.0) -> optax.GradientTransformationExtraArgs:
    """Rescale the query and key projections of every head past `tau`.

    This is Kimi K2's MuonClip (arXiv 2507.20534), applied after the update,
    per query group under grouped-query attention as Megatron Core clips it.
    The update passes through unchanged and the rescale rides the state
    (`QKClipState`), which `stepped` applies to the weights the update moved:
    one rounding of `gamma * (W + update)`, the paper's.

    The per-head maxima arrive as `qk_stats`, the `qk` collection the model
    sowed, which a run asks its objective for when its optimizer's state
    holds this transform's (`clips_qk`) and forwards from the loss's `Aux`,
    refusing a step whose objective returns none. Called without them, as a
    loop of plain optax calls may, the transform steps aside."""
    if tau <= 0:
        raise ValueError(f"the clip threshold bounds positive logits, got {tau}")

    def init_fn(params) -> QKClipState:
        return QKClipState(jnp.zeros([], jnp.int32), _rescales(params, None, tau))

    def update_fn(updates, state, params=None, qk_stats=None):
        if params is None:
            return updates, state
        return updates, state._replace(steps=optax.safe_increment(state.steps),
                                       rescales=_rescales(params, qk_stats, tau))

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


POST_STEP = (QKClipState, BoundsState)
"""The optimizer states whose effect acts on the weights an update moved (`stepped`)."""


def stepped(params, updates, before, after):
    """`params` moved by `updates`, then each post-step effect of the update that took the optimizer's
    state from `before` to `after`, in the chain's order: QK-Clip's rescale (`QKClipState`), one
    rounding of `gamma * (W + update)`, and a bounded group's clamp (`BoundsState`), exact. The
    weights a step stores, and the only way to form them (the trainer's transaction,
    `power_profiles`' averages). An effect whose transform the update did not run, a group a mask
    left out or an update a wrapper deferred or rejected (`optax.MultiSteps`, `optax.apply_if_finite`),
    whose state comes back as it was, leaves the weights as the update moved them."""
    moved = optax.apply_updates(params, updates)

    def effects(state):
        held = jax.tree_util.tree_leaves(state, is_leaf=lambda node: isinstance(node, POST_STEP))
        return [node for node in held if isinstance(node, POST_STEP)]

    pairs = list(zip(effects(before), effects(after), strict=True))
    if not pairs:
        return moved
    paths, leaves = zip(*jax.tree_util.tree_leaves_with_path(moved), strict=True)
    held = dict(zip(map(jax.tree_util.keystr, paths), leaves, strict=True))
    for previous, current in pairs:
        taken = current.steps != previous.steps
        acted = current.rescales if isinstance(current, QKClipState) else current.bounds
        lacking = acted.keys() - held.keys()
        if lacking:
            raise ValueError(f"the optimizer's state acts on {sorted(lacking)} after a step, which the "
                             "stepped parameters lack: step the tree the optimizer was updated with")
        for path, operand in acted.items():
            leaf = held[path]
            done = (_rescaled(leaf, operand) if isinstance(current, QKClipState)
                    else jnp.clip(leaf, operand[0], operand[1]))
            held[path] = jnp.where(taken, done, leaf)
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(moved), list(held.values()))


def _muonclip_groups(learning_rate, qk_clip_threshold: float = 100.0, **opts):
    """Chain `_muon_groups` with the QK-Clip that follows the update."""
    return optax.chain(
        _muon_groups(learning_rate, **opts),
        scale_by_qk_clip(qk_clip_threshold))


NEWTON_SCHULZ = (3.4445, -4.7750, 2.0315)
"""Muon's quintic Newton-Schulz coefficients, Jordan's (modded-nanogpt), optax's default."""


def newton_schulz(update: jax.Array, steps: int = 5,
                  dimension_numbers: optax.contrib.MuonDimensionNumbers | None = None) -> jax.Array:
    """`update` orthogonalized by Muon's quintic Newton-Schulz: each matrix over its Frobenius norm (plus
    1e-8), the wide way round, then `a X + (b A + c A A) X` with `A = X X^T`, `steps` times (`NEWTON_SCHULZ`).

    A rank-2 update is one matrix, its rows the reduction axis (a Linen kernel's); `dimension_numbers` name
    another update's reduction and output axes as Muon's do, its other axes a stack of matrices. On the
    CPU each matrix iterates alone (`jax.lax.map`): batched, XLA:CPU's products leave the iteration, which
    grows small singular values about 3.44 times a step, ten times further from float64; a GPU's batched
    ones round as one matrix's (an A100: 5.2e-6 either way), and keep the stack in one program (ledger 1650).
    """
    numbers = optax.contrib.MuonDimensionNumbers() if dimension_numbers is None else dimension_numbers
    reduction, output = (tuple(int(axis) % update.ndim for axis in np.atleast_1d(axes))
                         for axes in (numbers.reduction_axis, numbers.output_axis))
    stacked = tuple(axis for axis in range(update.ndim) if axis not in reduction + output)
    order = stacked + reduction + output
    sizes = tuple(math.prod(update.shape[axis] for axis in axes) for axes in (stacked, reduction, output))
    def orthogonalized(matrix):
        return _newton_schulz_matrix(matrix, steps)

    stack = update.transpose(order).reshape(sizes)
    matrices = jax.lax.map(orthogonalized, stack) if backend() == "cpu" else jax.vmap(orthogonalized)(stack)
    return matrices.reshape([update.shape[axis] for axis in order]).transpose(np.argsort(order))


def _newton_schulz_matrix(matrix: jax.Array, steps: int) -> jax.Array:
    a, b, c = (jnp.asarray(value, matrix.dtype) for value in NEWTON_SCHULZ)
    wide = matrix.T if matrix.shape[0] > matrix.shape[1] else matrix
    wide = wide / (jnp.linalg.norm(wide) + jnp.asarray(1e-8, matrix.dtype))
    for _ in range(steps):
        gram = wide @ wide.T
        wide = a * wide + (b * gram + c * gram @ gram) @ wide
    return wide.T if matrix.shape[0] > matrix.shape[1] else wide


class M3State(NamedTuple):
    """`scale_by_m3`'s state: the updates made so far, the fast momentum `M1`, the second moment `V`, the slow
    momentum `M2`, the gradients of the chunk in progress summed, and `M2` orthogonalized as the chunk
    began, `O2`."""
    updates: jax.Array
    fast: optax.Updates
    second: optax.Updates
    slow: optax.Updates
    chunk: optax.Updates
    slow_direction: optax.Updates


def scale_by_m3(*, beta1: float, beta2: float, beta3: float, alpha: float, eps: float, frequency: int,
                ns_steps: int = 5, weight_dimension_numbers=None) -> optax.GradientTransformation:
    """Multi-scale Momentum Muon, M3 (Behrouz et al., Nested Learning, arXiv 2512.24695, Algorithm 1):
    the direction `(O1 + alpha O2) / sqrt(V + eps)`, which `optax.scale_by_learning_rate` steps by.

    Each update adds `beta1 g` to the fast momentum `M1` and `beta2 g**2` to the second moment `V`, the
    algorithm's sums, neither decaying, and orthogonalizes `M1` by `ns_steps` Newton-Schulz iterations,
    Muon's (`newton_schulz`, each leaf shaped by its `weight_dimension_numbers`, a tree or a function of
    the updates as Muon's are; a leaf with none is one matrix, its rows the reduction). Its state is
    kept in float32 at least (`at_least_fp32`), so a bfloat16 matrix's momenta and gradients sum without
    rounding to its width, and the directions come back in that dtype. Once every
    `frequency` updates the slow momentum `M2` adds `beta3` times the chunk's gradients and is
    orthogonalized, `O2`, which the next chunk's updates read: the first chunk's is zero. The algorithm's
    sum, over `i = (k-1) f .. k f`, and eq. 75's, over `i = t - C .. t`, each hold f + 1 gradients, the one
    at a chunk's boundary in two chunks; here each enters `M2` once, with its own chunk's. The paper states
    no values for its factors, so each is the caller's.
    """
    if frequency < 1:
        raise ValueError(f"M3 updates its slow momentum once every `frequency` updates, got {frequency}")

    def orthogonalized(moment):
        if weight_dimension_numbers is None:
            return jax.tree.map(lambda leaf: newton_schulz(leaf, ns_steps), moment)
        numbers = (weight_dimension_numbers(moment) if callable(weight_dimension_numbers)
                   else weight_dimension_numbers)
        return jax.tree.map(lambda leaf, shape: newton_schulz(leaf, ns_steps, shape), moment, numbers,
                            is_leaf=lambda node: node is None)

    def init_fn(params) -> M3State:
        zeros = jax.tree.map(lambda param: jnp.zeros(param.shape, at_least_fp32(param.dtype)), params)
        return M3State(jnp.zeros([], jnp.int32), zeros, zeros, zeros, zeros, zeros)

    def update_fn(updates, state, params=None):
        del params
        updates = jax.tree.map(lambda gradient, moment: gradient.astype(moment.dtype), updates, state.fast)
        count = jnp.asarray(optax.safe_increment(state.updates))
        fast = jax.tree.map(lambda moment, gradient: moment + beta1 * gradient, state.fast, updates)
        second = jax.tree.map(lambda moment, gradient: moment + beta2 * jnp.square(gradient), state.second,
                              updates)
        direction = jax.tree.map(lambda fast, slow, second: (fast + alpha * slow) / jnp.sqrt(second + eps),
                                 orthogonalized(fast), state.slow_direction, second)
        chunk = jax.tree.map(jnp.add, state.chunk, updates)

        def closed(chunk):
            slow = jax.tree.map(lambda moment, summed: moment + beta3 * summed, state.slow, chunk)
            return slow, optax.tree.zeros_like(chunk), orthogonalized(slow)

        # The slow momentum and its orthogonalization run once a chunk, not every update.
        slow, chunk, slow_direction = jax.lax.cond(
            count % frequency == 0, closed, lambda chunk: (state.slow, chunk, state.slow_direction), chunk)
        return direction, M3State(count, fast, second, slow, chunk, slow_direction)

    return optax.GradientTransformation(init_fn, update_fn)


class DeltaState(NamedTuple):
    """`delta_gradient_descent`'s state: the updates made so far, which a scheduled rate reads."""
    updates: jax.Array


def delta_gradient_descent(learning_rate: float | optax.Schedule) -> optax.GradientTransformationExtraArgs:
    """Delta gradient descent (Behrouz et al., Nested Learning, arXiv 2512.24695, eqs. 56-57): each
    matrix steps to the minimizer of `|W x - u|**2 / 2 + |W - W_t|**2 / (2 eta)`, where `x` is the step's
    input and `u = -dL/dy` its output's negative gradient, so the step forgets what `W_t` held along `x`.

    By Sherman-Morrison the minimizer is `W_t (I - eta' x x^T) - eta' dL/dW` with
    `eta' = eta / (1 + eta |x|**2)`, the paper's `eta / (1 + eta)` at its normalized `|x| = 1`. A kernel
    holds its input on axis 0, as a Linen kernel does (`y = x K`), so its step is
    `-eta' (x (x^T K) + dL/dK)` over its other axes. The update takes `inputs`, a tree of `x` per leaf,
    the parameters' structure with None where a leaf has none: that leaf steps by gradient descent, the
    rule at `x = 0`. An update given no inputs, or no parameters, is refused rather than run as plain
    gradient descent.
    """

    def init_fn(params) -> DeltaState:
        del params
        return DeltaState(jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params=None, *, inputs=None, **extra_args):
        # A chain hands every transform each extra argument; the others' are not this one's.
        del extra_args
        if params is None or inputs is None:
            raise ValueError("delta gradient descent steps by each matrix's input and its weights: an update "
                             "takes `params` and `inputs` (None for a leaf with no input)")
        rate = jnp.asarray(learning_rate(state.updates) if callable(learning_rate) else learning_rate)

        def stepped(x, gradient, kernel):
            # A leaf another transform of an `optax.partition` steps reaches this one masked.
            if isinstance(gradient, optax.MaskedNode):
                return gradient
            if x is None:
                return -rate * gradient
            if x.shape != kernel.shape[:1]:
                raise ValueError(f"an input of shape {x.shape} is not the input axis of a kernel shaped "
                                 f"{kernel.shape}")
            shrunk = rate / (1 + rate * jnp.sum(jnp.square(x)))
            along = jnp.tensordot(x, jnp.tensordot(x, kernel, axes=(0, 0)), axes=0)
            return -shrunk * (along + gradient)

        stepped_updates = jax.tree.map(stepped, inputs, updates, params, is_leaf=lambda leaf: leaf is None)
        return stepped_updates, DeltaState(jnp.asarray(optax.safe_increment(state.updates)))

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


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
    torch forms it in double. A decay given as an array, a schedule's (`scheduled_solver`), takes
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


@dataclasses.dataclass(frozen=True)
class _Store:
    """How `_stored_moments` keeps Adam's moments: `zeros(params, signed)` a moment's stored zeros (the signed
    one Adam's first); `parts(structure, moment)` its stored leaves in the gradient's order; `decoded(part,
    signed)` one as float32; `encoded(moment, part, signed, salt)` a float32 one stored as `part` is, `salt` a
    uint32 of the step, the leaf and the moment; `whole(structure, parts)` the moment from its leaves."""
    zeros: Callable
    parts: Callable
    decoded: Callable
    encoded: Callable
    whole: Callable


def _stored_moments(b1: float, b2: float, eps: float, eps_root: float, nesterov: bool,
                    store: _Store) -> optax.GradientTransformation:
    """`scale_by_adam` with both moments kept as `store` keeps them. Each update runs optax's own update one
    leaf at a time, on that leaf's moments decoded to fp32, and stores the new ones: the step is optax's,
    only the storage is Dew's. Leaf by leaf keeps one leaf's fp32 moments live at a time; widening a bf16
    state whole first held all of them and raised the update's peak from 4.49 to 7.37 GB on the lm-dense
    tree (RTX 4080). The state keeps optax's `ScaleByAdamState`, its moments in `store`'s form."""
    inner = scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root, nesterov=nesterov)

    def init_fn(params):
        return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=store.zeros(params, signed=True),
                                      nu=store.zeros(params, signed=False))

    def update_fn(updates, state, params=None):
        del params  # scale_by_adam reads none
        step = fmix32(jnp.asarray(state.count).astype(jnp.uint32) * jnp.uint32(0x27D4EB2F))
        leaves, structure = jax.tree.flatten(updates)
        mus, nus = store.parts(structure, state.mu), store.parts(structure, state.nu)
        scaled, new_mu, new_nu, count = [], [], [], state.count
        for index, (gradient, mu, nu) in enumerate(zip(leaves, mus, nus, strict=True)):
            # One array is a pytree of one leaf, so optax updates the leaf alone.
            update, one = inner.update(gradient, optax.ScaleByAdamState(
                count=state.count, mu=store.decoded(mu, signed=True), nu=store.decoded(nu, signed=False)))
            if not isinstance(one, optax.ScaleByAdamState):
                raise TypeError(f"scale_by_adam returned {type(one).__name__}")
            count = one.count
            salt = fmix32(step + jnp.uint32(2 * index + 1) * jnp.uint32(0x165667B1))
            scaled.append(update)
            new_mu.append(store.encoded(jnp.asarray(one.mu), mu, signed=True, salt=salt))
            new_nu.append(store.encoded(jnp.asarray(one.nu), nu, signed=False,
                                        salt=fmix32(salt + jnp.uint32(0x9E3779B9))))
        return structure.unflatten(scaled), optax.ScaleByAdamState(
            count=count, mu=store.whole(structure, new_mu), nu=store.whole(structure, new_nu))

    return optax.GradientTransformation(init_fn, update_fn)


def bf16_moments(b1: float, b2: float, eps: float, eps_root: float,
                 nesterov: bool) -> optax.GradientTransformation:
    """`scale_by_adam` with both moments stored in bf16 (`_stored_moments`), written back stochastically
    rounded (`stochastic_round_bf16`), seeded by the step count, the leaf and the moment. Round to nearest
    would lose every increment of the second moment smaller than half its bf16 spacing, which at b2 =
    0.999 is most of them; a stochastic rounding keeps each one in expectation. The moments are trees of
    the parameters' shapes, so sharding and checkpoints read them as they read fp32 state."""
    return _stored_moments(b1, b2, eps, eps_root, nesterov, _Store(
        zeros=lambda params, signed: jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, jnp.bfloat16), params),
        parts=lambda structure, moment: structure.flatten_up_to(moment),
        decoded=lambda part, signed: part.astype(jnp.float32),
        encoded=lambda moment, part, signed, salt: stochastic_round_bf16(moment, salt),
        whole=lambda structure, parts: structure.unflatten(parts)))


def dynamic_map(*, signed: bool) -> np.ndarray:
    """The 256 values an 8-bit moment's codes stand for, ascending: Dettmers' dynamic tree map
    (arXiv 1511.04561), bitsandbytes' and torchao's `create_dynamic_map(signed)` at 7 exponent bits, the
    signed one for Adam's first moment and the unsigned one for its second. Each decade from 1e-6 to 1
    holds a fraction's midpoints, more the larger the decade, beside 0 and 1."""
    values, exponent_bits, fraction_bits = [], 7, 7
    for exponent in range(exponent_bits):
        count = 2 ** (exponent + fraction_bits - exponent_bits + (0 if signed else 1)) + 1
        bounds = np.linspace(0.1, 1, count, dtype=np.float32)
        means = (bounds[:-1] + bounds[1:]) / np.float32(2)
        decade = np.float32(10.0 ** (exponent - exponent_bits + 1))
        values += list(decade * means) + (list(-decade * means) if signed else [])
    values += [0.0, 1.0]
    assert len(values) == 256
    return np.sort(np.asarray(values, np.float32))


class QuantizedMoments(NamedTuple):
    """A moment of an optimizer's state in 8 bits (`int8_moments`), as two trees of the parameters' own
    paths, so each leaf is placed as its parameter is: `codes`, each value's code, the index of its nearest
    `dynamic_map` value, in the parameter's shape (a parameter `int8_moments` keeps whole, its float32
    moment), and `scales`, each block's largest magnitude along the last axis, which the codes' values are
    fractions of (None where the moment is kept whole)."""
    codes: object
    scales: object


QUANTIZED_SIZE = 4096
"""The fewest values a moment holds in 8 bits (`int8_moments`), bitsandbytes' and torchao's: a smaller
one's scales cost more than its codes save, and it stays float32."""


def int8_moments(b1: float, b2: float, eps: float, eps_root: float, nesterov: bool,
                 block: int = 256) -> optax.GradientTransformation:
    """`scale_by_adam` with both moments stored in 8 bits, blockwise (Dettmers et al., arXiv 2110.02861, as
    torchao's `AdamW8bit` stores them, through `_stored_moments`): each block of `block` values along a
    leaf's last axis keeps its largest magnitude in float32 and each value the code of the nearest
    `dynamic_map` value to its fraction of it (`quantized`), the first moment's map signed and the
    second's unsigned. A leaf of fewer than `QUANTIZED_SIZE` values, or whose last axis `block` does not
    divide, keeps float32 moments. A moment's bytes fall from 4 a value to 1 and its block's share of a
    float32 scale, 1/64 a value at the default block."""
    maps = {signed: jnp.asarray(dynamic_map(signed=signed)) for signed in (True, False)}

    def held(leaf) -> bool:
        return leaf.size >= QUANTIZED_SIZE and leaf.ndim > 0 and leaf.shape[-1] % block == 0

    def zeros(params, signed: bool) -> QuantizedMoments:
        zero = _zero_code(dynamic_map(signed=signed))
        return QuantizedMoments(
            jax.tree.map(lambda leaf: jnp.full(leaf.shape, zero, jnp.uint8) if held(leaf)
                         else jnp.zeros(leaf.shape, jnp.float32), params),
            jax.tree.map(lambda leaf: jnp.zeros((*leaf.shape[:-1], leaf.shape[-1] // block), jnp.float32)
                         if held(leaf) else None, params))

    def encoded(moment, part, signed, salt):
        return (moment, None) if part[1] is None else quantized(moment, maps[signed], block)

    return _stored_moments(b1, b2, eps, eps_root, nesterov, _Store(
        zeros=zeros,
        parts=lambda structure, moment: list(zip(structure.flatten_up_to(moment.codes),
                                                 structure.flatten_up_to(moment.scales), strict=True)),
        decoded=lambda part, signed: (part[0] if part[1] is None
                                      else dequantized(part[0], part[1], maps[signed], block)),
        encoded=encoded,
        whole=lambda structure, parts: QuantizedMoments(structure.unflatten([part[0] for part in parts]),
                                                        structure.unflatten([part[1] for part in parts]))))


def _zero_code(values: np.ndarray) -> int:
    """The code of 0 in a `dynamic_map`."""
    return int(np.flatnonzero(values == 0)[0])


def dequantized(codes: jax.Array, scales: jax.Array, values: jax.Array, block: int) -> jax.Array:
    """A moment stored in 8 bits as float32: its `codes`' `values` times their blocks' `scales`."""
    fractions = values[codes.astype(jnp.int32)]
    blocked = fractions.reshape(*fractions.shape[:-1], -1, block)
    return (blocked * scales[..., None]).reshape(fractions.shape)


def quantized(moment: jax.Array, values: jax.Array, block: int) -> tuple[jax.Array, jax.Array]:
    """`moment` in 8 bits, its codes and its blocks' scales: each block of `block` values along its last
    axis scaled by its largest magnitude (at least 1e-12), and each value coded by the nearest of `values`,
    ascending, a value halfway between two taking the larger, as torchao's `quantize_8bit_with_qmap` does."""
    blocked = moment.reshape(*moment.shape[:-1], -1, block)
    scales = jnp.maximum(jnp.max(jnp.abs(blocked), axis=-1), 1e-12)
    fractions = (blocked / scales[..., None]).reshape(moment.shape)
    # Unrolled, the search fuses with the rest of the quantization: its default loop ran each of its eight
    # halvings over the whole moment, and took a step's peak above fp32 state's (docs/performance.md).
    below = jnp.clip(jnp.searchsorted(values, fractions, side="right", method="scan_unrolled") - 1, 0,
                     values.shape[0] - 1)
    above = jnp.minimum(below + 1, values.shape[0] - 1)
    nearest = jnp.where(fractions - values[below] >= (values[above] - values[below]) * 0.5, above, below)
    return nearest.astype(jnp.uint8), scales


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


def _stored_adam(moments: Callable[..., optax.GradientTransformation], *, coupled: bool,
                 default_decay: float) -> Callable[..., optax.GradientTransformation]:
    """optax.adam (torch's coupled weight decay) or optax.adamw over `moments`, `scale_by_adam` with its
    moments stored another way, their other options kept."""
    def make(learning_rate, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0, weight_decay=default_decay, mask=None,
             *, nesterov: bool = False):
        return _adam_chain(moments(b1, b2, eps, eps_root, nesterov), learning_rate, weight_decay, mask,
                           coupled=coupled)
    return make


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
        produced = stepped(params, updates, state.inner, inner)
        averages = tuple(ema_update(average, produced, powered_decay(state.powers[index])(state.updates))
                         for index, average in enumerate(state.averages))
        return updates, PowerProfilesState(state.updates + 1, state.stds, state.powers, averages, inner)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


Optimizer = Literal['adam', 'adamw', 'lamb', 'muon', 'muonclip']
"""The optimizers `OptimConfig.optimizer` names, each built by its `OPTIMIZER_MAP` entry."""

STATE_OPTIMIZERS: dict[str, dict[Optimizer, Callable[..., optax.GradientTransformation]]] = {
    stored: {'adam': _stored_adam(moments, coupled=True, default_decay=0.0),
             'adamw': _stored_adam(moments, coupled=False, default_decay=1e-4)}
    for stored, moments in (('bfloat16', bf16_moments), ('int8', int8_moments))}
"""The optimizers each `OptimConfig.state_dtype` but float32 builds: optax.adam and optax.adamw with
their moments stored in bf16 (`bf16_moments`) or in 8 bits (`int8_moments`), their other options kept."""


OPTIMIZER_MAP: dict[Optimizer, Callable[..., optax.GradientTransformation]] = {
    'adam': _adam,
    'adamw': _adamw,
    'lamb': optax.lamb,
    'muon': _muon_groups,
    'muonclip': _muonclip_groups,
}


class RateState(NamedTuple):
    """A scheduled solver's schedule (`scheduled_solver`): the `updates` it has applied, and its
    schedule's rate-valued fields by path (`ScheduleBase.rates`), or, for a schedule that marks none,
    the factor its values take under `CONSTANT`."""

    updates: jax.Array
    rates: Mapping[str, jax.Array]


CONSTANT = 'value'
"""The name a constant rate is held under (`rates_of`): a `RateState`'s for a schedule that marks no
rate, and Optax's own `learning_rate` hyperparameter for a constant."""


def rates_of(rate: float | ScheduleBase, multiplier: float = 1.0) -> dict[str, float]:
    """`rate`'s rate-valued fields (`ScheduleBase.rates`), or the constant under `CONSTANT`, each times
    `multiplier` on the host: the values a solver's `RateState` holds, each rounded once into it. A
    schedule that marks no rate holds the multiplier alone, which scales its values."""
    held = (rate.rates() or {CONSTANT: 1.0}) if isinstance(rate, ScheduleBase) else {CONSTANT: rate}
    return {path: value * multiplier for path, value in held.items()}


class _Rate(optax.schedules.StatefulSchedule):
    """Optax's stateful hyperparameter of a scheduled solver: the schedule's value at each update, with
    its rate-valued fields operands in the state rather than constants in the program."""

    def __init__(self, rate: ScheduleBase, steps: int | None, dtype, multiplier: float = 1.0):
        self.rate, self.steps, self.dtype, self.rates = rate, steps, dtype, rates_of(rate, multiplier)

    def init(self) -> RateState:
        return RateState(jnp.zeros([], jnp.int32),
                         {path: jnp.asarray(value, self.dtype) for path, value in self.rates.items()})

    def update(self, state: RateState, **extra_args) -> RateState:
        del extra_args
        return state._replace(updates=optax.safe_increment(state.updates))

    def __call__(self, state: RateState, **extra_args) -> jax.typing.ArrayLike:
        del extra_args
        if not self.rate.rates():
            return state.rates[CONSTANT] * self.rate.schedule(self.steps)(state.updates)
        return self.rate.schedule(self.steps, state.rates)(state.updates)


def scheduled_solver(make: Callable[..., optax.GradientTransformation], rate: float | ScheduleBase,
                     steps: int | None, opts: Mapping[str, object], b1: ScheduleBase | None = None,
                     multiplier: float = 1.0) -> optax.GradientTransformation:
    """`make(learning_rate, **opts)` at `rate`, a constant or a schedule over a run of `steps` updates,
    times `multiplier`, its rate-valued fields operands in the optimizer's state (`RateState`): runs
    at different rates of one schedule's timings lower to one executable, and each update's rate is
    the schedule's own value from those fields, Optax's arithmetic where it is Optax's schedule.

    Optax injects the rate and a `b1` schedule (`optax.inject_hyperparams`), and rebuilds the
    factory from their traced values; every other option keeps the Python value the factory branches
    on. The rate reaches the factory as a schedule, so Optax casts it to each update leaf's dtype, and
    the injected values are at least float32 on a bf16 tree. The factory is built once here too, so a
    bad option raises now rather than at the state's init.
    """
    make(1.0 if isinstance(rate, ScheduleBase) else rate, **opts)

    def factory(learning_rate, b1):
        given = opts if b1 is None else {**opts, 'b1': b1}
        return make(lambda count: learning_rate, **given)

    def injected(dtype):
        return optax.inject_hyperparams(factory, hyperparam_dtype=dtype)(
            # A constant is Optax's own numeric hyperparameter, rounded once from its product.
            learning_rate=(_Rate(rate, steps, dtype, multiplier) if isinstance(rate, ScheduleBase)
                           else rates_of(rate, multiplier)[CONSTANT]),
            b1=None if b1 is None else b1.schedule(steps))

    def init_fn(params):
        return injected(at_least_fp32(optax.tree.dtype(params, 'highest'))).init(params)

    def update_fn(updates, state, params=None, **extra_args):
        return injected(state.hyperparams['learning_rate'].dtype).update(updates, state, params, **extra_args)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def learning_rate(opt_state: optax.OptState) -> jax.typing.ArrayLike | None:
    """Return the learning rate stored in `opt_state`, if the optimizer stores one.

    An OptimConfig's solver holds the rate it stepped at last (`dew.optim.scheduled_solver`), and an
    Optax optimizer can inject `learning_rate` itself (`optax.inject_hyperparams`); an injected
    wrapper of other hyperparameters is read through to the optimizer it holds. The result is None
    when the optimizer closes over the rate, or when parameter groups hold several rates.
    """
    injected = (optax.InjectHyperparamsState, optax.InjectStatefulHyperparamsState)
    rates = []

    def read(tree):
        for node in jax.tree.leaves(tree, is_leaf=lambda node: isinstance(node, injected)):
            if not isinstance(node, injected):
                continue
            if "learning_rate" in node.hyperparams:
                rates.append(node.hyperparams["learning_rate"])
            else:
                read(node.inner_state)

    read(opt_state)
    return rates[0] if len(rates) == 1 else None


def with_rates(opt_state: optax.OptState, rates: Mapping[str | None, Mapping[str, float]]) -> optax.OptState:
    """`opt_state` with each `scheduled_solver`'s rate set to `rates`' for its param group, by name, or
    under None for a solver of no group (`rates_of`): a schedule's rate-valued fields in its
    `RateState`, or a constant in Optax's `learning_rate` hyperparameter. A solver an injected wrapper
    of the run's own holds is reached through the wrapper's inner state. Each keeps the old value's
    dtype and placement."""
    injected = optax.InjectStatefulHyperparamsState

    def put(value, like):
        return jax.device_put(np.asarray(value, like.dtype), like.sharding)

    def walked(tree, prefix: jax.tree_util.KeyPath):
        def replaced(path: jax.tree_util.KeyPath, node):
            if not isinstance(node, injected):
                return node
            full = (*prefix, *path)
            node = node._replace(inner_state=walked(node.inner_state,
                                                    (*full, jax.tree_util.GetAttrKey('inner_state'))))
            if 'learning_rate' not in node.hyperparams:
                return node
            group = next((entry.key for entry in full
                          if isinstance(entry, jax.tree_util.DictKey) and entry.key in rates), None)
            wanted, schedule = rates[group], node.hyperparams_states.get('learning_rate')
            if isinstance(schedule, RateState):
                fields = {name: put(wanted[name], held) for name, held in schedule.rates.items()}
                states = {**node.hyperparams_states, 'learning_rate': schedule._replace(rates=fields)}
                return node._replace(hyperparams_states=states)
            rate = put(wanted[CONSTANT], node.hyperparams['learning_rate'])
            return node._replace(hyperparams={**node.hyperparams, 'learning_rate': rate})

        return jax.tree_util.tree_map_with_path(replaced, tree,
                                                is_leaf=lambda node: isinstance(node, injected))

    return walked(opt_state, ())


def _keep_within(lower: float, upper: float) -> optax.GradientTransformation:
    """Clamp each parameter of a group into `[lower, upper]` once its update has moved it.

    torch recipes clamp parameters in place after `optimizer.step()`, as
    DCLS's `clamp_parameters` does its kernel positions. The update passes
    through and the clamp rides the state (`BoundsState`), which `stepped`
    applies to the stepped weights exactly, after the group's other post-step
    effects (QK-Clip's rescale), so a group's projection reaches its
    parameters alone. optax's `projections.projection_box` clamps a parameter
    tree outside the chain, where it would reach every group at once.
    """
    if not lower < upper:
        raise ValueError(f"the bounds run from a lower to a higher value, got [{lower}, {upper}]")

    def init_fn(params) -> BoundsState:
        return BoundsState(jnp.zeros([], jnp.int32), {
            jax.tree_util.keystr(path): jnp.asarray([lower, upper], leaf.dtype)
            for path, leaf in jax.tree_util.tree_leaves_with_path(params)})

    def update_fn(updates, state, params=None):
        del params
        return updates, state._replace(steps=optax.safe_increment(state.steps))

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

    def rates(self, learning_rate: float | ScheduleBase) -> dict[str, float]:
        """This group's rate-valued fields (`RateState`): its own `schedule`'s, else the config's
        `learning_rate`'s, times `learning_rate_multiplier`."""
        return rates_of(learning_rate if self.schedule is None else self.schedule,
                      self.learning_rate_multiplier)

    def solver(self, make: Callable[..., optax.GradientTransformation],
               learning_rate: float | ScheduleBase, steps: int | None,
               opts: Mapping[str, object]) -> optax.GradientTransformation:
        """Build this group's optimizer: `make` at the group's rate, its fields held in state
        (`scheduled_solver`), with the group's `b1` schedule and `bounds`, over a run of `steps`
        updates.

        A `b1` schedule replaces any `b1` in `opts`, and needs a `make` that
        names a `b1` argument (adam, adamw, lamb; not muon).
        """
        if self.b1 is not None and 'b1' not in inspect.signature(make).parameters:
            raise ValueError(f"param group {self.name!r} schedules b1, and its optimizer "
                             f"takes no b1 (adam, adamw and lamb do)")
        solver = scheduled_solver(make, learning_rate if self.schedule is None else self.schedule, steps,
                                  opts, self.b1, self.learning_rate_multiplier)
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


def power_schedule(peak: jax.typing.ArrayLike, warmup_steps: int, a: jax.typing.ArrayLike, b: float,
                   c: float = 1.0,
                   decay_start: int | None = None, decay_end: int | None = None,
                   end_value: jax.typing.ArrayLike = 0.0) -> optax.Schedule:
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

    warm = jnp.minimum(peak, a * (warmup_steps * c) ** b) if warmup_steps else peak
    pieces = [optax.linear_schedule(0.0, warm, warmup_steps)] if warmup_steps else []
    pieces.append(lambda count: law(count + warmup_steps))
    boundaries = [warmup_steps] if warmup_steps else []
    if decay_start is not None:
        if decay_end is None or not warmup_steps <= decay_start < decay_end:
            raise ValueError(
                f"the linear tail runs from decay_start past the warmup to a later "
                f"decay_end, got {decay_start} to {decay_end} after {warmup_steps} warmup steps")
        pieces.append(optax.linear_schedule(law(decay_start), end_value,
                                            decay_end - decay_start))
        boundaries.append(decay_start)
    return optax.join_schedules(pieces, boundaries) if boundaries else pieces[0]


def linear_schedule(peak: jax.typing.ArrayLike, warmup_steps: int, decay_start: int | None,
                    decay_end: int, end_value: jax.typing.ArrayLike = 0.0) -> optax.Schedule:
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


type Rates = Mapping[str, jax.Array | float]
"""A schedule's rate-valued fields by path (`ScheduleBase.rates`): the configured values, or a solver's
operands for them (`RateState`)."""


@dataclasses.dataclass(frozen=True)
class ScheduleBase:
    """The base of the schedule records, each of which holds its own fields
    and builds an optax schedule.

    A run's record names the schedule's class and holds only the fields that
    schedule reads; `dew.registry.schedules` holds the short aliases. A
    subclass builds its values in `values`; `schedule` reads them. The fields
    it marks `metadata={'rate': True}` are its rates (`rates`), which a solver
    holds as operands (`scheduled_solver`) and `values` reads from `rates`, so
    a run's program does not depend on them; a subclass that marks none
    implements `values(steps)` alone.
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

    def rates(self) -> dict[str, float]:
        """Each rate-valued field's value (`metadata={'rate': True}`) by its dotted path, nested
        records such as PowerTail's too, a None field's derived default resolved where the schedule
        derives one (OneCycle's) and left out otherwise."""
        def walked(record, prefix):
            for field in dataclasses.fields(record):
                value = getattr(record, field.name)
                if field.metadata.get('rate') and value is not None:
                    yield prefix + field.name, value
                elif dataclasses.is_dataclass(value):
                    yield from walked(value, f"{prefix}{field.name}.")
        return dict(walked(self, ""))

    def schedule(self, steps: int | None, rates: Rates | None = None
                 ) -> optax.Schedule:
        """Return the value at each update of a run of `steps` updates, as an optax schedule, its
        rate-valued fields `rates`' (`rates()` by default): operands in a solver's state, or the
        configured values."""
        held = self.rates() if rates is None else rates

        def built(own):
            return self.values(own, held) if held else self.values(own)

        if self.every == 1:
            return built(steps)
        own = None if steps is None else steps // self.every
        if own is not None and own < 1:
            raise ValueError(f"a run of {steps} updates never advances a schedule that steps once "
                             f"every {self.every}")
        values, every = built(own), self.every
        return lambda count: values(jnp.asarray(count) // every)

    def _horizon(self, steps: int | None, field: str) -> int:
        """Resolve a decay's step horizon, which a token budget cannot supply."""
        if steps is None:
            raise ValueError(f"{type(self).__name__} needs {field} or trainer.steps as its step horizon")
        return steps

    def values(self, steps: int | None, rates: Rates | None = None
               ) -> optax.Schedule:
        """Return the value at each of the schedule's own steps, over `steps` of them, from its
        rate-valued fields `rates` (`rates()`). A value derived from the rates alone, as a ratio of
        two, is the configured fields', which a solver's operands keep: they differ from them by
        one factor, a param group's multiplier."""
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

    def values(self, steps: int | None, rates: Rates | None = None
               ) -> optax.Schedule:
        rates = self.rates() if rates is None else rates
        decay_steps = self._horizon(steps if self.decay_steps is None else self.decay_steps, "decay_steps")
        if self.warmup_steps >= decay_steps:
            raise ValueError(f"Cosine needs warmup_steps={self.warmup_steps} smaller than "
                             f"decay_steps={decay_steps}")
        # `optax.warmup_cosine_decay_schedule`'s own pieces, joined as it joins them; it forms the
        # floor's ratio by branching on the peak, so the ratio is the configured fields' (`values`).
        alpha = 0.0 if self.peak_value == 0.0 else self.end_value / self.peak_value
        return optax.join_schedules([
            optax.linear_schedule(rates['init_value'], rates['peak_value'], self.warmup_steps),
            optax.cosine_decay_schedule(rates['peak_value'], decay_steps - self.warmup_steps, alpha),
        ], [self.warmup_steps])


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

    def values(self, steps: int | None, rates: Rates | None = None
               ) -> optax.Schedule:
        rates = self.rates() if rates is None else rates
        tail = self.tail
        if tail is None:
            return power_schedule(rates['peak_value'], self.warmup_steps, rates['a'], self.b, self.c)
        return power_schedule(
            rates['peak_value'], self.warmup_steps, rates['a'], self.b, self.c, decay_start=tail.start,
            decay_end=self._horizon(steps if tail.steps is None else tail.steps, "tail.steps"),
            end_value=rates['tail.end_value'])


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

    def values(self, steps: int | None, rates: Rates | None = None
               ) -> optax.Schedule:
        rates = self.rates() if rates is None else rates
        return linear_schedule(rates['peak_value'], self.warmup_steps, self.decay_start,
                               self._horizon(steps if self.decay_steps is None else self.decay_steps,
                                             "decay_steps"),
                               end_value=rates['end_value'])


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

    def rates(self) -> dict[str, float]:
        """The peak, and the init and end values with torch's derived defaults resolved."""
        init = self.peak_value / 25 if self.init_value is None else self.init_value
        return {'peak_value': self.peak_value, 'init_value': init,
                'end_value': init / 1e4 if self.end_value is None else self.end_value}

    def values(self, steps: int | None, rates: Rates | None = None
               ) -> optax.Schedule:
        rates = self.rates() if rates is None else rates
        steps = self._horizon(steps, "trainer.steps")
        # torch's phase ends, kept as floats as torch keeps them.
        rise, last = self.warmup_fraction * steps - 1, steps - 1
        if not 0 < rise < last:
            raise ValueError(f"a one-cycle schedule over {steps} steps with warmup fraction "
                             f"{self.warmup_fraction} has no rise or no fall; give it more steps")
        # Each half cosine weighs its two ends by the cosine's halves, `start * c + stop * (1 - c)`, so it
        # lands on each end exactly and takes any two rates, a zero start or ends far apart; torch's
        # form, `stop + (start - stop) / 2 * (cos + 1)`, rounds the difference of the ends in float32,
        # which put the run's first rate 25 times further from float64 than optax's cosine decay.
        # The phase's angle is optax's cosine decay's, `pi * elapsed / span`, in that order: near a
        # phase's end the cosine's halves cancel, so the angle's rounding is most of the rate's.
        def anneal(start: jax.Array | float, stop: jax.Array | float, elapsed: jax.Array,
                   span: float) -> jax.Array:
            turned = jnp.cos(jnp.pi * elapsed / span)
            return start * (0.5 * (1 + turned)) + stop * (0.5 * (1 - turned))

        def value(count) -> jax.Array:
            step = jnp.minimum(jnp.asarray(count, jnp.float32), last)
            return jnp.where(step <= rise, anneal(rates['init_value'], rates['peak_value'], step, rise),
                             anneal(rates['peak_value'], rates['end_value'], step - rise, last - rise))

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

    def values(self, steps: int | None, rates: Rates | None = None
               ) -> optax.Schedule:
        rates = self.rates() if rates is None else rates
        span = self._horizon(steps if self.decay_steps is None else self.decay_steps, "decay_steps")
        if span <= 0 or self.init_value <= 0 or self.end_value <= 0:
            raise ValueError(f"an exponential decay needs positive steps, init_value and end_value, got "
                             f"{span}, {self.init_value} and {self.end_value}")
        # optax branches on the decay rate, so it is the configured fields' ratio (`values`).
        decay = optax.exponential_decay(rates['init_value'], span,
                                        self.end_value / self.init_value,
                                        end_value=rates['end_value'])
        if not self.offset:
            return decay
        return lambda count: rates['offset'] + decay(count)


__all__ = ["NEWTON_SCHULZ", "BoundsState", "Cosine", "DeltaState", "Exponential", "Linear", "M3State",
           "OneCycle", "ParamGroup", "Power", "PowerProfilesState", "PowerTail", "QKClipState",
           "ScheduleBase", "delta_gradient_descent", "newton_schulz", "scale_by_m3", "stepped"]
