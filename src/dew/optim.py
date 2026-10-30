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
import glob
import inspect
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from dew._hashing import fmix32
from dew._records import share
from dew.mesh import LogicalAxes
from dew.newton_schulz import (
    NEWTON_SCHULZ,
    Preconditioning,
    as_matrices,
    dot_algorithm,
    headed,
    matrix_axes,
    muon_leaf,
    orthogonalized,
    polar_express,
)
from dew.nn.precision import at_least_fp32, stochastic_bf16
from dew.nn.sharding import declarations, declared_axes
from dew.registry import Aliases, JsonDict, from_record, record_argument, schedules, to_record

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


def _matrix_sides(axes: LogicalAxes) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Split a declared parameter's axes into a contracted and an output side, None where they are
    not two.

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
        return None
    return tuple(sides[0]), tuple(sides[1])


FRESH_LEARNING_RATE = 2e-4
"""The peak rate a module drawn fresh beside a loaded base trains at where a run names no groups
(`OptimConfig.fresh_groups`), on the run's own schedule's shape: a base fine-tunes at a rate a fresh
module barely moves at. On a context-only task over a loaded Qwen3.5 slice held frozen, an IDEFICS block
at a base rate of 5e-6 learned on every seed from 5e-5 to 5e-4 under AdamW with decay and from 5e-5 to
5e-3 under the Muon hybrid; at 5e-6 the Muon hybrid learned nothing, and at 1e-3 one AdamW seed in three
did not learn (ledger 2419). This is the middle of the rates both held on every seed."""


def weight_decay_mask(params):
    """Mark every parameter but the biases and the vectors, which weight decay spares by default:
    nanoGPT's split (model.py, `GPT.configure_optimizers`), with Megatron Core's and timm's `bias`
    of any rank (core/optimizer/__init__.py, optim/_param_groups.py), a per-head one included."""
    return jax.tree_util.tree_map_with_path(lambda path, param: param.ndim >= 2 and not (
        isinstance(path[-1], jax.tree_util.DictKey) and path[-1].key == "bias"), params)


def muon_partition(params) -> dict[str, str]:
    """Where Muon's parameter split sends each parameter (`muon_weight_dimension_numbers`), by its path
    (`jax.tree_util.keystr`): 'muon', or 'adamw' and why, the record a run keeps of the split.

    AdamW takes a parameter of rank below two, a bias, and a parameter that
    maps into or out of a discrete index: the vocabulary, the model's output
    space, or the expert a router picks. That is the split four labs
    cross-confirmed. It takes too a per-channel filter, rank 3 with an input
    dimension of 1 (a depthwise convolution's `[channels, 1, taps]`, Gated
    DeltaNet's conv1d), a vector a channel. Everything else is a matrix, its
    stacked axes (an expert's) batched and its head axes one side, and goes to
    Muon through optax's dimension numbers; a parameter that is none of these,
    declared axes that are not one contracted and one output side or an
    undeclared tensor of rank above two, is refused, as which of its axes form
    the matrix is what the split cannot guess.

    Which group a parameter lands in is read off the logical axes its module
    declares (`dew.nn.sharding`), the same table the sharding derivation
    reads, so one declaration answers both questions. An undeclared matrix of
    rank two, or one declared with no axis named (replicated, as Gemma 3n's
    AltUp matrices are), takes Linen's kernel convention, contracting axis 0
    into axis 1.
    """
    return {jax.tree_util.keystr(path): reason
            for path, (_, reason) in _placements(params).items()}


def muon_weight_dimension_numbers(params):
    """Build a `MuonDimensionNumbers` per parameter, None where AdamW steps in (`muon_partition`).

    Optax reads one spec tree shaped like the parameters and treats a None
    leaf as an AdamW parameter (optax/contrib/_muon.py:660-675).
    """
    placed = _placements(params)
    return jax.tree_util.tree_map_with_path(lambda path, _: placed[path][0], params)


def _placements(params) -> dict[jax.tree_util.KeyPath, tuple[optax.contrib.MuonDimensionNumbers | None, str]]:
    """Each parameter's Muon dimension numbers, None for AdamW, and the reason (`muon_partition`), by its
    path. A matrix's declared sides are read first, so a projection of one head, `[embed, 1, head_dim]`,
    is the matrix its declaration names; only a parameter no declaration makes a matrix is read as a
    per-channel filter by its shape."""
    declared = declarations(params)
    leaves = jax.tree_util.tree_leaves_with_path(params)
    decaying = jax.tree_util.tree_leaves(weight_decay_mask(params))
    placed = {}
    for (path, param), decays in zip(leaves, decaying, strict=True):
        placed[path] = _placement(path, param, decays, declared)
    return placed


def _placement(path: jax.tree_util.KeyPath, param, decays: bool,
               declared) -> tuple[optax.contrib.MuonDimensionNumbers | None, str]:
    if not decays:
        return None, "adamw: a bias or of rank below two"
    axes = declared_axes(path, param.ndim, declared)
    # A declaration that names no axis says how the parameter shards (whole, on every device),
    # not which of its axes form the matrix, so it reads as an undeclared one (AltUp's).
    named = None if axes is None or all(axis is None for axis in axes) else axes
    if named is not None:
        if SELECTION_AXES & {axis for axis in named if axis is not None} or named[-1] in BATCH_AXES:
            return None, "adamw: a lookup into or out of a discrete index"
        sides = _matrix_sides(named)
        if sides is not None:
            return optax.contrib.MuonDimensionNumbers(*sides), "muon"
    if param.ndim == 3 and param.shape[1] == 1:
        return None, "adamw: a per-channel filter, [channels, 1, taps], a vector a channel"
    if named is not None:
        raise ValueError(
            f"{jax.tree_util.keystr(path)} is declared {named}, which is not one contracted side and one "
            "output side, so Muon cannot tell what its matrix is. An axis that stacks matrices rather "
            "than being part of one belongs in BATCH_AXES.")
    if param.ndim > 2:
        raise ValueError(
            f"{jax.tree_util.keystr(path)} has rank {param.ndim} and no declared logical axes, so Muon "
            "cannot tell which of its axes form the matrix. Declare it with @logical_axes on its module.")
    return optax.contrib.MuonDimensionNumbers(), "muon"


def _muon_groups(learning_rate, weight_decay=0.0, adam_weight_decay=None, *, ns_coeffs=NEWTON_SCHULZ,
                 ns_steps: int = 5, beta: float = 0.95, eps: float = 1e-8, weight_decay_mask=None,
                 mu_dtype=None, nesterov: bool = True, adaptive: bool = False,
                 preconditioning: Preconditioning = "frobenius", consistent_rms: float | None = None,
                 ns_precision: jax.lax.DotAlgorithmPreset | str | None = None,
                 per_head: Mapping[str, int] | None = None, neuron_beta2: float | None = None,
                 adam_b1: float = 0.9, adam_b2: float = 0.999, adam_eps_root: float = 0.0,
                 adam_learning_rate=None) -> optax.GradientTransformation:
    """Run Muon over the matrices and AdamW over the rest, on one schedule. The AdamW group decays
    its norms and biases too, at `adam_weight_decay` or else `weight_decay`: Moonlight shares the
    decay among matrix and non-matrix parameters, RMSNorm's included (arXiv 2502.16982, 2.2).

    optax.contrib.muon's composition, its arguments and state alike: the parameters partitioned into
    the two groups (`muon_weight_dimension_numbers`), Muon's momentum, optax's shape scaling, which
    optax 0.2.8 does not export (`_scale_by_shape`), the decay and the rate, its Newton-Schulz optax's
    own in the run's numerics, or at `ns_precision`, a dot algorithm a record holds by its name, over whole
    matrices spread across the devices (`scale_by_muon`). Opt-in beside it: `per_head`, the parameters
    whose output side is a stack of heads, by pattern (`parameter_name`), and each head's width, each
    head's block orthogonalized and scaled alone (Kimi K3's per-head Muon, arXiv 2607.24653 2.5);
    `ns_coeffs="polar_express"`, Polar Express's coefficients (`polar_express`); and `neuron_beta2`,
    NorMuon's neuron-wise normalization of the orthogonal update at that second-moment decay
    (`scale_by_neuron_norm`)."""
    def labels(params):
        return jax.tree.map(lambda numbers: "adam" if numbers is None else "muon",
                            muon_weight_dimension_numbers(params), is_leaf=muon_leaf)
    muon = optax.chain(
        scale_by_muon(ns_coeffs, ns_steps, beta, eps, mu_dtype, nesterov=nesterov, adaptive=adaptive,
                      preconditioning=preconditioning, ns_precision=ns_precision, per_head=per_head),
        *(() if neuron_beta2 is None else (scale_by_neuron_norm(neuron_beta2, per_head=per_head),)),
        _scale_by_shape(consistent_rms, per_head),
        optax.add_decayed_weights(weight_decay, weight_decay_mask),
        optax.scale_by_learning_rate(learning_rate))
    adam = optax.adamw(learning_rate if adam_learning_rate is None else adam_learning_rate, b1=adam_b1,
                       b2=adam_b2, eps=eps, eps_root=adam_eps_root, mu_dtype=mu_dtype, nesterov=nesterov,
                       weight_decay=weight_decay if adam_weight_decay is None else adam_weight_decay)
    return optax.partition({"muon": muon, "adam": adam}, labels)


def scale_by_muon(ns_coeffs=NEWTON_SCHULZ, ns_steps: int = 5, beta: float = 0.95, eps: float = 1e-8,
                  mu_dtype=None, *, nesterov: bool = True, adaptive: bool = False,
                  preconditioning: Preconditioning = "frobenius",
                  ns_precision: jax.lax.DotAlgorithmPreset | str | None = None,
                  per_head: Mapping[str, int] | None = None) -> optax.GradientTransformation:
    """optax.contrib.scale_by_muon, its momentum, Nesterov step, bias correction and state (`MuonState`) as
    optax's, its coefficients `ns_coeffs` as optax's Muon reads them (a preset's name too,
    `_coefficients`), and each matrix's Newton-Schulz optax's own in the run's numerics, or at
    `ns_precision`'s dot algorithm (`ns_operands`), run whole on one device, the matrices spread across the
    devices (`orthogonalized`). Each parameter's matrix is its dimension numbers'
    (`muon_weight_dimension_numbers`), or each of its heads' where `per_head` names it (`head_widths`)."""
    coefficients = _coefficients(ns_coeffs, ns_steps)
    algorithm = dot_algorithm(ns_precision)

    def init_fn(params):
        return optax.contrib.MuonState(count=jnp.zeros([], jnp.int32), ns_coeffs=coefficients,
                                       mu=optax.tree.zeros_like(params, dtype=mu_dtype))

    def update_fn(updates, state, params=None):
        del params
        mu = optax.tree.update_moment(updates, state.mu, beta, 1)
        count = optax.safe_increment(state.count)
        if nesterov:
            mu_hat = jax.tree.map(lambda moment, update: beta * moment + (1 - beta) * update,
                                  optax.tree.bias_correction(mu, beta, optax.safe_increment(count)),
                                  optax.tree.bias_correction(updates, beta, count))
        else:
            mu_hat = optax.tree.bias_correction(mu, beta, count)
        orthogonal = orthogonalized(mu_hat, muon_weight_dimension_numbers(mu_hat), state.ns_coeffs, ns_steps,
                                    eps, preconditioning, algorithm, head_widths(mu_hat, per_head))
        if adaptive:
            # The orthogonal update scaled by the dual norm of the momentum's (arXiv 2409.20325).
            orthogonal = jax.tree.map(lambda moment, update: jnp.sum(moment.conj() * update) * update,
                                      mu_hat, orthogonal)
        return orthogonal, optax.contrib.MuonState(count, optax.tree.cast(mu, mu_dtype), state.ns_coeffs)
    return optax.GradientTransformation(init_fn, update_fn)


def _coefficients(ns_coeffs, ns_steps: int) -> jax.Array:
    """The Newton-Schulz coefficients optax's Muon iterates by for `ns_coeffs`, a preset's name (`"dion"`,
    or Dew's `"polar_express"`, `polar_express`) or one triple or a schedule of them, as the state it
    builds holds them: its last `ns_steps` triples of a schedule."""
    if ns_coeffs == "polar_express":
        ns_coeffs = polar_express(ns_steps)
    template = {"matrix": jnp.zeros((1, 1))}
    state = optax.contrib.muon(0.0, ns_coeffs=ns_coeffs, ns_steps=ns_steps).init(template)
    found = [leaf for leaf in jax.tree.leaves(state, is_leaf=lambda node: isinstance(
        node, optax.contrib.MuonState)) if isinstance(leaf, optax.contrib.MuonState)]
    return jnp.asarray(found[0].ns_coeffs)


def _scale_by_shape(consistent_rms: float | None, per_head: Mapping[str, int] | None = None
                    ) -> optax.GradientTransformation:
    """optax.contrib's shape scaling of Muon's update, each matrix's fans its dimension numbers', a head's
    output fan its width where `per_head` names it (`head_widths`): Jordan's width transfer,
    `sqrt(max(1, fan_out / fan_in))`, or Moonlight's consistent RMS,
    `sqrt(max(fan_in, fan_out)) * consistent_rms`, where it is given."""
    def scaled(update, numbers, width):
        reduction, output = matrix_axes(update, numbers)
        fan_in = math.prod(update.shape[axis] for axis in reduction)
        fan_out = width or math.prod(update.shape[axis] for axis in output)
        if consistent_rms is None:
            return jnp.sqrt(jnp.maximum(1, fan_out / fan_in)) * update
        return jnp.sqrt(jnp.maximum(fan_in, fan_out)) * consistent_rms * update

    def update_fn(updates, state, params=None):
        del params
        numbers = muon_weight_dimension_numbers(updates)
        widths = jax.tree.leaves(head_widths(updates, per_head), is_leaf=lambda node: node is None)
        leaves, structure = jax.tree.flatten(updates)
        scaled_leaves = [scaled(leaf, shape, width) for leaf, shape, width in zip(
            leaves, jax.tree.leaves(numbers, is_leaf=muon_leaf), widths, strict=True)]
        return jax.tree.unflatten(structure, scaled_leaves), state
    return optax.GradientTransformation(lambda params: optax.EmptyState(), update_fn)


def head_widths(updates, per_head: Mapping[str, int] | None):
    """Each update's head width where `per_head` names its parameter by a pattern (`parameter_name`), the
    first that matches, else None: a tree of the updates' shape."""
    def width(path: jax.tree_util.KeyPath, _) -> int | None:
        name = parameter_name(path)
        return next((given for pattern, given in (per_head or {}).items()
                     if fnmatch.fnmatchcase(name, pattern)), None)
    return jax.tree_util.tree_map_with_path(width, updates)


def scale_by_neuron_norm(beta2: float = 0.95, eps: float = 1e-10, per_head: Mapping[str, int] | None = None
                         ) -> optax.GradientTransformation:
    """NorMuon's neuron-wise normalization of an orthogonal update (arXiv 2510.05491, Algorithm 1 lines
    7-9, its code's `normuon_update`): each output neuron's second moment, the mean square of its row of
    the update over the reduction side, decayed at `beta2`; the update divided by its root plus `eps`; and
    each matrix (a head's, where `per_head` names it) scaled back to the orthogonal update's own Frobenius
    norm, as its code keeps it, so the shape scaling after it (`_scale_by_shape`) sets its size: Jordan's
    as its code, or Moonlight's consistent RMS of 0.2 as its paper's `0.2 sqrt(mn) / ||O||_F`."""
    def stacks(tree) -> list[tuple[jax.Array, Callable[[jax.Array], jax.Array]]]:
        # Each leaf's matrices by its own declared dimension numbers, read over the whole tree.
        return [as_matrices(leaf, numbers) for leaf, numbers in zip(
            jax.tree.leaves(tree), jax.tree.leaves(muon_weight_dimension_numbers(tree), is_leaf=muon_leaf),
            strict=True)]

    def init_fn(params):
        return jax.tree.unflatten(jax.tree.structure(params), [
            jnp.zeros((matrices.shape[0], matrices.shape[2]), jnp.float32) for matrices, _ in stacks(params)])

    def update_fn(updates, state, params=None):
        del params
        widths = jax.tree.leaves(head_widths(updates, per_head), is_leaf=lambda node: node is None)
        structure = jax.tree.structure(updates)
        moments = jax.tree.leaves(state)
        normed, kept = [], []
        for (matrices, back), moment, width in zip(stacks(updates), moments, widths, strict=True):
            neuron = beta2 * moment + (1 - beta2) * jnp.mean(jnp.square(matrices.astype(jnp.float32)), axis=1)
            divided = matrices / (jnp.sqrt(neuron)[:, None, :] + eps).astype(matrices.dtype)
            blocks, blocked = headed(matrices, width), headed(divided, width)
            norm = (jnp.linalg.norm(blocks[0], axis=(1, 2), keepdims=True)
                    / (jnp.linalg.norm(blocked[0], axis=(1, 2), keepdims=True) + eps))
            normed.append(back(blocked[1](blocked[0] * norm.astype(matrices.dtype))))
            kept.append(neuron)
        return jax.tree.unflatten(structure, normed), jax.tree.unflatten(jax.tree.structure(state), kept)
    return optax.GradientTransformation(init_fn, update_fn)


QK_PROJECTIONS = frozenset({'q_proj', 'q_b_proj', 'k_proj', 'kv_b_proj'})
"""The projection names the clip rescales: the query and key projections of
grouped-query attention and, for latent attention, the per-head output
projections. Anything else keeps its update untouched."""


def _dict_names(path: jax.tree_util.KeyPath) -> tuple[str, ...]:
    """List the dict keys along `path`, dropping the sequence indices a stacked
    view never produces on a parameter tree."""
    return tuple(entry.key for entry in path
                 if isinstance(entry, jax.tree_util.DictKey))


def parameter_name(path: jax.tree_util.KeyPath) -> str:
    """A parameter's path under `params` as a pattern matches it: its dict keys joined by '/'
    (`layers_0/attn/q_proj/kernel`), as a `ParamGroup`'s and an inspection's patterns read it."""
    return "/".join(_dict_names(path))


def _sown(node, name: str):
    """Read the array a sowed collection holds under `name`, past its one-tuple."""
    value = node.get(name) if isinstance(node, Mapping) else None
    if value is None:
        return None
    return value[0] if isinstance(value, (tuple, list)) else value


def _clip_scale(s_max: jax.Array, tau: float, guard: float) -> jax.Array:
    """Compute each head's rescale: `min(1, tau / (s + guard))` past `tau`, else 1.0.

    The `guard` is each implementation's own: MaxText's latent clip reads
    `minimum(1, tau / (s + 1e-6))`, and Megatron Core's grouped `clip_qk`
    reads `min(1, tau / s)` (guard 0). A head whose every logit is
    non-positive clips nothing on either path: there MaxText's formula goes
    negative and would flip the weights' sign, and Megatron's square root of
    it is NaN, so Dew holds those heads at 1.0."""
    return jnp.where(s_max > 0, jnp.minimum(1.0, tau / (s_max + guard)), 1.0)


def _columns(gamma: jax.Array, split: int, last: int, dtype) -> jax.Array:
    """`gamma` over `split` heads of a leaf's last axis, `last` wide, as one factor a column: the
    `[last]` vector a step's rescale holds, the same shape however the clip groups the heads, at
    `dtype` (float32 or wider, the leaf's: `at_least_fp32`)."""
    return jnp.broadcast_to(jnp.asarray(gamma, dtype).reshape(split, -1),
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
    # Float32 or wider, the leaf's: a float32 or bf16 run's clip is float32 as ever, and a float64
    # read of it computes it in float64.
    wide = at_least_fp32(param.dtype)
    logged = jnp.asarray(logged)
    s_max = jnp.max(logged.astype(jnp.promote_types(wide, at_least_fp32(logged.dtype))), axis=0)
    heads = s_max.shape[0]
    last = param.shape[-1]
    if last % heads:
        raise ValueError(
            f"QK-Clip cannot split {'.'.join(names)} of width {last} into "
            f"{heads} heads")
    proj = names[-2]
    nope = _sown(node, 'qk_nope')
    # Latent attention is MaxText's clip, with its guard; grouped attention is Megatron Core's, without.
    scale = _clip_scale(s_max, tau, 1e-6 if nope is not None else 0.0)
    if proj in ('q_proj', 'q_b_proj') and nope is not None:
        return _columns(_mla_query_gamma(scale, nope, last // heads), heads, last, wide)
    if proj == 'kv_b_proj':
        if nope is None:
            raise ValueError(
                f"QK-Clip reaches {'.'.join(names)} with no nope width sowed; "
                "a latent key projection needs its layer's 'qk_nope'")
        return _columns(_mla_key_gamma(scale, nope, last // heads), heads, last, wide)
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
        return _columns(gamma, heads, last, wide)
    columns = jnp.arange(last) // (last // kv_heads)
    return jnp.sqrt(grouped[columns]).astype(wide)


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
        held[jax.tree_util.keystr(path)] = (
            jnp.ones(param.shape[-1], at_least_fp32(param.dtype)) if qk_stats is None
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


def _muonclip_groups(learning_rate, qk_clip_threshold: float = 100.0, **opts) -> optax.GradientTransformation:
    """Chain `_muon_groups` with the QK-Clip that follows the update."""
    return optax.chain(
        _muon_groups(learning_rate, **opts),
        scale_by_qk_clip(qk_clip_threshold))


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
                ns_steps: int = 5, weight_dimension_numbers=None,
                ns_precision: jax.lax.DotAlgorithmPreset | str | None = None) -> optax.GradientTransformation:
    """Multi-scale Momentum Muon, M3 (Behrouz et al., Nested Learning, arXiv 2512.24695, Algorithm 1):
    the direction `(O1 + alpha O2) / sqrt(V + eps)`, which `optax.scale_by_learning_rate` steps by.

    Each update adds `beta1 g` to the fast momentum `M1` and `beta2 g**2` to the second moment `V`, the
    algorithm's sums, neither decaying, and orthogonalizes `M1` by `ns_steps` Newton-Schulz iterations,
    Muon's in the run's numerics or at `ns_precision`'s dot algorithm, each matrix whole on one device and
    the matrices spread over the mesh
    (`orthogonalized`), each leaf shaped by its `weight_dimension_numbers`, a tree or a function of the
    updates as Muon's are; a leaf with none is one matrix, its rows the reduction. Its state is
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

    def orthogonal(moment):
        numbers = (jax.tree.map(lambda _: None, moment) if weight_dimension_numbers is None
                   else weight_dimension_numbers(moment) if callable(weight_dimension_numbers)
                   else weight_dimension_numbers)
        # A leaf with none is one matrix, its rows the reduction: MuonDimensionNumbers' own default.
        numbers = jax.tree.map(lambda shape: optax.contrib.MuonDimensionNumbers() if shape is None else shape,
                               numbers, is_leaf=muon_leaf)
        return orthogonalized(moment, numbers, jnp.asarray(NEWTON_SCHULZ), ns_steps, 1e-8,
                              algorithm=dot_algorithm(ns_precision))

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
                                 orthogonal(fast), state.slow_direction, second)
        chunk = jax.tree.map(jnp.add, state.chunk, updates)

        def closed(chunk):
            slow = jax.tree.map(lambda moment, summed: moment + beta3 * summed, state.slow, chunk)
            return slow, optax.tree.zeros_like(chunk), orthogonal(slow)

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
    stays NaN (`precision.stochastic_bf16`, its rounding).
    """
    x = x.astype(jnp.float32)
    index = jax.lax.iota(jnp.uint32, x.size).reshape(x.shape)
    return stochastic_bf16(x, fmix32(index * jnp.uint32(0x9E3779B1) + seed.astype(jnp.uint32)))


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


def moment_dtype(param, dtype):
    """The dtype a first moment of `param` is stored in under a requested `dtype`: the parameter's own
    where None, and for a complex parameter the complex type that holds `dtype`'s values in each
    component (complex64 for float32 or narrower), so its imaginary part is never dropped."""
    if dtype is None:
        return None
    if jnp.issubdtype(jnp.asarray(param).dtype, jnp.complexfloating):
        return jnp.promote_types(dtype, jnp.complex64)
    return dtype


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
        # A complex parameter's first moment keeps both components (`moment_dtype`); its second, of the
        # gradient's squared magnitude, is real.
        return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32),
                                      mu=jax.tree.map(lambda p: jnp.zeros_like(p, moment_dtype(p, mu_dtype)),
                                                      params),
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
        # A requested dtype is each leaf's own (`moment_dtype`): a complex leaf's first moment stays
        # complex. None keeps the promoted moment, as Optax's does.
        mu = mu if mu_dtype is None else optax.tree.cast_like(mu, state.mu)
        return scaled, optax.ScaleByAdamState(count=count, mu=mu, nu=optax.tree.cast_like(nu, state.nu))

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
    def zeros(params, signed: bool):
        # A complex parameter's first moment is kept whole (`_complex_moment`).
        return jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, _complex_moment(leaf, signed) or jnp.bfloat16),
                            params)

    def encoded(moment, part, signed, salt):
        return moment.astype(part.dtype) if _is_complex(part) else stochastic_round_bf16(moment, salt)

    return _stored_moments(b1, b2, eps, eps_root, nesterov, _Store(
        zeros=zeros,
        parts=lambda structure, moment: structure.flatten_up_to(moment),
        decoded=lambda part, signed: part if _is_complex(part) else part.astype(jnp.float32),
        encoded=encoded,
        whole=lambda structure, parts: structure.unflatten(parts)))


def _is_complex(leaf) -> bool:
    return jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.complexfloating)


def _complex_moment(leaf, signed: bool):
    """complex64 for a complex parameter's first moment (`signed`), whose imaginary part a real store would
    drop, kept whole by every reduced store; None otherwise. Its second moment, of the gradient's squared
    magnitude, is real and stored as any."""
    return jnp.complex64 if signed and _is_complex(leaf) else None


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
        # A complex parameter's moments are kept whole (`_complex_moment`).
        return (leaf.size >= QUANTIZED_SIZE and leaf.ndim > 0 and leaf.shape[-1] % block == 0
                and not _is_complex(leaf))

    def zeros(params, signed: bool) -> QuantizedMoments:
        zero = _zero_code(dynamic_map(signed=signed))
        return QuantizedMoments(
            jax.tree.map(lambda leaf: jnp.full(leaf.shape, zero, jnp.uint8) if held(leaf)
                         else jnp.zeros(leaf.shape, _complex_moment(leaf, signed) or jnp.float32), params),
            jax.tree.map(lambda leaf: jnp.zeros((*leaf.shape[:-1], leaf.shape[-1] // block), jnp.float32)
                         if held(leaf) else None, params))

    def encoded(moment, part, signed, salt):
        return (moment, None) if part[1] is None else quantized(moment, block, signed=signed)

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


CODE_WINDOW = 6
"""The codes around a value's estimated one (`_estimated_codes`) that `quantized` compares it with: the
estimate is within two of the code below the value, whose search they settle."""


def _estimated_codes(fractions: jax.Array, *, signed: bool) -> jax.Array:
    """Where each of `fractions` falls in `dynamic_map(signed=signed)`, to within two codes: its decade
    by its base-10 logarithm, and its place among the decade's evenly spaced values, the map's own
    layout (a decade `e` of 0 to 6 holds `2 ** e` values each side of 0, `2 ** (e + 1)` unsigned, the
    means of as many even steps from a tenth of the decade's top to its top)."""
    magnitude = jnp.abs(fractions)
    decade = jnp.clip(jnp.floor(jnp.log10(magnitude)) + 7, 0, 6).astype(jnp.int32)
    count = jnp.left_shift(1, decade + (0 if signed else 1))
    top = jnp.power(jnp.float32(10), (decade - 6).astype(jnp.float32))
    step = jnp.float32(0.9) / count.astype(jnp.float32)
    place = jnp.clip(jnp.floor((magnitude / top - jnp.float32(0.1)) / step), 0, count - 1).astype(jnp.int32)
    if not signed:
        return count - 1 + place  # past 0 and the 2 ** (e + 1) - 2 values of the decades below
    # 0 is code 127, each decade's positive values follow the 2 ** e - 1 of the decades below it, and
    # its negative values mirror them below 0.
    return jnp.where(fractions < 0, 127 - count - place, 127 + count + place)


def quantized(moment: jax.Array, block: int, *, signed: bool) -> tuple[jax.Array, jax.Array]:
    """`moment` in 8 bits, its codes and its blocks' scales: each block of `block` values along its last
    axis scaled by its largest magnitude (at least 1e-12), and each value coded by the nearest value of
    `dynamic_map(signed=signed)`, a value halfway between two taking the larger, as torchao's
    `quantize_8bit_with_qmap` does. The code below each value is found among the `CODE_WINDOW` around
    its estimate (`_estimated_codes`), each compared apart from the others: a search of the whole map
    ran its eight halvings one after another (docs/performance.md)."""
    values = jnp.asarray(dynamic_map(signed=signed))
    last = values.shape[0] - 1
    blocked = moment.reshape(*moment.shape[:-1], -1, block)
    scales = jnp.maximum(jnp.max(jnp.abs(blocked), axis=-1), 1e-12)
    fractions = (blocked / scales[..., None]).reshape(moment.shape)
    # The codes at most each value: those of the window's that are, past the codes below it. One code at
    # a time, the comparisons fuse with the coding; a window axis held six of each value's at once.
    first = _estimated_codes(fractions, signed=signed) - CODE_WINDOW // 2 - 1
    below = first
    for code in (first + offset for offset in range(1, CODE_WINDOW + 1)):
        below = below + ((code < 0) | ((code <= last) & (values[jnp.clip(code, 0, last)] <= fractions)))
    below = jnp.clip(below, 0, last)
    above = jnp.minimum(below + 1, last)
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
          mask=None, *, nesterov: bool = False, nu_dtype=None) -> optax.GradientTransformation:
    """optax.adam over `scale_by_adam`, with torch's coupled weight decay."""
    return _adam_chain(scale_by_adam(b1, b2, eps, eps_root, mu_dtype, nesterov=nesterov, nu_dtype=nu_dtype),
                       learning_rate, weight_decay, mask, coupled=True)


def _adamw(learning_rate, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0, mu_dtype=None, weight_decay=1e-4,
           mask=None, *, nesterov: bool = False, nu_dtype=None) -> optax.GradientTransformation:
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
    (`dew.averaging`).

    Update t keeps (1 - 1/t)^(γ + 1) of each average and blends in the rest
    of the parameters it just made (Karras et al. 2024, Eq. 127). The
    averages ride in the optimizer state, so they are sharded, placed and
    skipped on a rejected step exactly as its moments are. Each checkpoint
    is also the snapshot of its step (`Checkpoints.profile_steps`), retained
    whole so the state and averages share one atomic save. Each std is rounded to fp32 first, the
    precision the state records it in.
    """
    from dew.averaging import ema_update, exponent, powered_decay

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

optimizers: Aliases[Callable[..., optax.GradientTransformation], optax.GradientTransformation] = Aliases(
    "optimizer", {name: f"{make.__module__}:{make.__qualname__}" for name, make in OPTIMIZER_MAP.items()},
    base="optax:GradientTransformation")
"""What `OptimConfig.optimizer` names: Dew's optimizers by name (`OPTIMIZER_MAP`), or an Optax factory
by its import path, which must be declared to return a gradient transformation before it is called,
as every import path a record names is (`dew.registry.Aliases`)."""


class RateState(NamedTuple):
    """A scheduled solver's schedule (`scheduled_solver`): the `updates` it has applied, and its
    schedule's rate-valued fields by path (`ScheduleBase.rates`), or, for a schedule that marks none,
    the factor its values take under `CONSTANT`."""

    updates: jax.Array
    rates: Mapping[str, jax.Array]


class ProgressState(NamedTuple):
    """A scheduled solver's schedule over a run's progress (`ScheduleBase.progressed`): its `RateState`'s
    `updates` and `rates`, the progress the run stood at as its updates reached each count its law
    marks (`ScheduleBase.marks`), by name, 0 until then, and the progress it stood at as its updates last
    reached a multiple of the schedule's `every` (`held`), which its value holds to the next one, as a
    schedule's own steps hold it over updates."""

    updates: jax.Array
    rates: Mapping[str, jax.Array]
    marks: Mapping[str, jax.Array]
    held: jax.Array


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
    its rate-valued fields operands in the state rather than constants in the program. Where the run is
    `clocked` (bounded in examples, so it counts no updates ahead) and the schedule's horizon is the
    run's (`ScheduleBase.run_relative`), it reads the run's progress, which each update hands it
    (`progress=`), and keeps the progress its law marks (`ProgressState`)."""

    def __init__(self, rate: ScheduleBase, steps: int | None, dtype, multiplier: float = 1.0,
                 clocked: bool = False):
        self.rate, self.steps, self.dtype, self.rates = rate, steps, dtype, rates_of(rate, multiplier)
        self.progressed = clocked and rate.run_relative()

    def init(self) -> RateState | ProgressState:
        updates = jnp.zeros([], jnp.int32)
        rates = {path: jnp.asarray(value, self.dtype) for path, value in self.rates.items()}
        if not self.progressed:
            return RateState(updates, rates)
        return ProgressState(updates, rates, {name: jnp.zeros([], jnp.float32) for name in self.rate.marks()},
                             jnp.zeros([], jnp.float32))

    def update(self, state: RateState | ProgressState, *, progress: jax.Array | None = None,
               **extra_args) -> RateState | ProgressState:
        del extra_args
        updates = optax.safe_increment(state.updates)
        if not isinstance(state, ProgressState):
            return state._replace(updates=updates)
        assert progress is not None, "a schedule on a run's progress is handed it at each update"
        return state._replace(updates=updates, marks=self._marked(state, progress),
                              held=self._held(state, progress))

    def _held(self, state: ProgressState, progress: jax.typing.ArrayLike) -> jax.Array:
        """The progress the schedule's value reads at `state`'s update: the run's own where the update
        starts one of the schedule's steps (a multiple of `every`), else the one that step started at."""
        at = jnp.asarray(progress)
        return jnp.where(state.updates % self.rate.every == 0, at, state.held.astype(at.dtype))

    def _marked(self, state: ProgressState, progress: jax.typing.ArrayLike) -> dict[str, jax.Array]:
        """`state`'s marks with the progress of the update that stands at a marked count recorded: where
        the run stood as the piece the count starts began."""
        at = jnp.asarray(progress)
        return {name: jnp.where(state.updates == count, at, state.marks[name].astype(at.dtype))
                for name, count in self.rate.marks().items()}

    def __call__(self, state: RateState | ProgressState, *, progress: jax.Array | None = None,
                 **extra_args) -> jax.typing.ArrayLike:
        del extra_args
        if isinstance(state, ProgressState):
            # Optax reads the rate once as it builds its state, where the run has consumed nothing.
            progress = jnp.zeros([], jnp.float32) if progress is None else progress
            at, marks = self._held(state, progress), self._marked(state, progress)
            if not self.rate.rates():
                return state.rates[CONSTANT] * self.rate.progressed({}, at, state.updates, marks)
            return self.rate.progressed(state.rates, at, state.updates, marks)
        if not self.rate.rates():
            return state.rates[CONSTANT] * self.rate.schedule(self.steps)(state.updates)
        return self.rate.schedule(self.steps, state.rates)(state.updates)


def scheduled_solver(make: Callable[..., optax.GradientTransformation], rate: float | ScheduleBase,
                     steps: int | None, opts: Mapping[str, object], b1: ScheduleBase | None = None,
                     multiplier: float = 1.0, clocked: bool = False) -> optax.GradientTransformation:
    """`make(learning_rate, **opts)` at `rate`, a constant or a schedule over a run of `steps` updates,
    times `multiplier`, its rate-valued fields operands in the optimizer's state (`RateState`): runs
    at different rates of one schedule's timings lower to one executable, and each update's rate is
    the schedule's own value from those fields, Optax's arithmetic where it is Optax's schedule.

    Optax injects the rate and a `b1` schedule (`optax.inject_hyperparams`), and rebuilds the
    factory from their traced values; every other option keeps the Python value the factory branches
    on. The rate reaches the factory as a schedule, so Optax casts it to each update leaf's dtype, and
    the injected values are at least float32 on a bf16 tree. The factory is built once here too, so a
    bad option raises now rather than at the state's init. A `clocked` run counts no updates ahead (it
    is bounded in examples), so a schedule whose horizon is the run's reads the run's progress, which
    each update is handed (`_Rate`).
    """
    make(1.0 if isinstance(rate, ScheduleBase) else rate, **opts)

    def factory(learning_rate, b1):
        given = opts if b1 is None else {**opts, 'b1': b1}
        return make(lambda count: learning_rate, **given)

    def injected(dtype):
        return optax.inject_hyperparams(factory, hyperparam_dtype=dtype)(
            # A constant is Optax's own numeric hyperparameter, rounded once from its product.
            learning_rate=(_Rate(rate, steps, dtype, multiplier, clocked) if isinstance(rate, ScheduleBase)
                           else rates_of(rate, multiplier)[CONSTANT]),
            b1=(None if b1 is None else _Rate(b1, steps, dtype, clocked=True)
                if clocked and b1.run_relative() else b1.schedule(steps)))

    def init_fn(params):
        # A rate is real: a complex tree's is its components' dtype, at least float32.
        highest = optax.tree.dtype(params, 'highest')
        real = jnp.finfo(highest).dtype if jnp.issubdtype(highest, jnp.complexfloating) else highest
        return injected(at_least_fp32(real)).init(params)

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


def reclocked(opt_state: optax.OptState, made: optax.OptState) -> optax.OptState:
    """`opt_state`, a state supplied to a run, with each schedule's state in the form the run's own
    optimizer makes (`made`, its init's shapes): a schedule over a run's progress (`ProgressState`) that
    the run counts in updates keeps its updates and rates alone (`RateState`), and one the run reads
    over its progress starts its marks at none, as the run's clock starts afresh (`Trainer.fit`'s
    `state`). A schedule in its own form is kept as it is."""
    def converted(own, held):
        if not isinstance(own, (RateState, ProgressState)) or type(own) is type(held):
            return held
        if isinstance(own, RateState):
            return RateState(held.updates, held.rates)
        marks = {name: jnp.zeros([], jnp.float32) for name in own.marks}
        return ProgressState(held.updates, held.rates, marks, jnp.zeros([], jnp.float32))

    def scheduled(node) -> bool:
        return isinstance(node, (RateState, ProgressState))

    return jax.tree.map(converted, made, opt_state, is_leaf=scheduled)


def with_rates(opt_state: optax.OptState, rates: Mapping[str | None, Mapping[str, float]]) -> optax.OptState:
    """`opt_state` with each `scheduled_solver`'s rate set to `rates`' for its param group, by name, or
    under None for a solver of no group (`rates_of`): a schedule's rate-valued fields in its
    `RateState`, or a constant in Optax's `learning_rate` hyperparameter. A solver an injected wrapper
    of the run's own holds is reached through the wrapper's inner state (`_rated`). Each keeps the old
    value's dtype and placement."""
    def wanted(path: jax.tree_util.KeyPath, held: Mapping[str, jax.typing.ArrayLike],
               kind: str) -> Mapping[str, float]:
        return rates[next((entry.key for entry in path
                           if isinstance(entry, jax.tree_util.DictKey) and entry.key in rates), None)]

    return _rated(opt_state, wanted)


def held_rates(opt_state: optax.OptState) -> dict[jax.tree_util.KeyPath, dict[str, float]]:
    """Each injected solver's learning rate as `opt_state` holds it, on the host, by the solver's path
    (`_rated`): a schedule's rate-valued fields, or a constant under `CONSTANT`."""
    held: dict[jax.tree_util.KeyPath, dict[str, float]] = {}

    def read(path: jax.tree_util.KeyPath, rates: Mapping[str, jax.typing.ArrayLike], kind: str) -> None:
        held[path] = {name: float(np.asarray(value)) for name, value in rates.items()}

    _rated(opt_state, read)
    return held


def scaled_rates(opt_state: optax.OptState, held: Mapping[jax.tree_util.KeyPath, Mapping[str, float]],
                 factor: float) -> optax.OptState:
    """`opt_state` training at `factor` times the learning rate each injected solver held where `held`
    was read (`held_rates`), of a state of the same optimizer: a schedule's values are proportional to
    its rate-valued fields (`ScheduleBase.values`), so its every value is the held one's times
    `factor`."""
    return _rated(opt_state, lambda path, *_: {name: value * factor for name, value in held[path].items()})


def settable_rates(opt_state: optax.OptState) -> bool:
    """Whether `opt_state` holds a learning rate and every one it holds is one `with_rates` sets: a
    schedule's fields in its `RateState` or a constant, not a rate Optax computes from a step count
    alone at each update (`optax.inject_hyperparams` of a schedule function), read through `_rated`."""
    kinds: list[str] = []

    def read(path: jax.tree_util.KeyPath, held: Mapping[str, jax.typing.ArrayLike], kind: str) -> None:
        kinds.append(kind)

    _rated(opt_state, read)
    return bool(kinds) and "computed" not in kinds


type Wanted = Callable[[jax.tree_util.KeyPath, Mapping[str, jax.typing.ArrayLike], str],
                       Mapping[str, float] | None]
"""What a solver at a path should hold of the learning rate it holds, of its kind (`_rated`), None for
what it holds."""


def _rated(opt_state: optax.OptState, wanted: Wanted) -> optax.OptState:
    """`opt_state` with each injected solver's learning rate set to what `wanted` gives for the solver's
    path, the rate it holds and its kind: `scheduled`, a schedule's rate-valued fields in its
    `RateState` or `ProgressState`; `constant`, a constant's own value under `CONSTANT`; or `computed`,
    the value Optax computed at the last update from a schedule function, which it computes again at
    the next. None leaves it as it is. A solver an injected wrapper holds is reached through the
    wrapper's inner state. Each keeps the old value's dtype and placement."""
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
            schedule = node.hyperparams_states.get('learning_rate')
            scheduled = isinstance(schedule, RateState | ProgressState)
            held = schedule.rates if scheduled else {CONSTANT: node.hyperparams['learning_rate']}
            given = wanted(full, held, "scheduled" if scheduled else "constant" if schedule is None
                           else "computed")
            if given is None:
                return node
            if scheduled:
                fields = {name: put(given[name], held) for name, held in schedule.rates.items()}
                states = {**node.hyperparams_states, 'learning_rate': schedule._replace(rates=fields)}
                return node._replace(hyperparams_states=states)
            rate = put(given[CONSTANT], node.hyperparams['learning_rate'])
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
    patterns: tuple[str, ...] = ()
    learning_rate_multiplier: float = 1.0
    weight_decay: float | None = None
    spared: bool = False
    """Whether the group's decay spares its biases and vectors (`weight_decay_mask`). A group decays
    every parameter it claims unless it does, as a torch optimizer's parameter group does, so a run's
    groups mirror the reference's own groups (lm_engine's, a torch recipe's); the split an ungrouped
    optimizer takes (`OptimConfig.spared`) is the groups' only where they say so."""
    schedule: ScheduleBase | None = None
    b1: ScheduleBase | None = None
    bounds: tuple[float, float] | None = None
    fresh: bool = False
    """Whether the group claims the modules its run draws fresh beside a loaded base, ahead of its own
    `patterns` (`Objective.fresh_modules`): what `OptimConfig.with_fresh` names before a run records
    the group."""

    def __post_init__(self):
        object.__setattr__(self, "patterns", tuple(self.patterns))
        if not self.patterns and not self.fresh:
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

    def rate[R](self, learning_rate: R) -> R | ScheduleBase:
        """The rate its solver runs at: its own schedule, or the config's `learning_rate` where it
        names none."""
        return learning_rate if self.schedule is None else self.schedule

    def solver(self, make: Callable[..., optax.GradientTransformation],
               learning_rate: float | ScheduleBase, steps: int | None,
               opts: Mapping[str, object], *, clocked: bool = False) -> optax.GradientTransformation:
        """Build this group's optimizer: `make` at the group's rate, its fields held in state
        (`scheduled_solver`), with the group's `b1` schedule and `bounds`, over a run of `steps`
        updates, or of the progress of a `clocked` one.

        A `b1` schedule replaces any `b1` in `opts`, and needs a `make` that
        names a `b1` argument (adam, adamw, lamb; not muon).
        """
        if self.b1 is not None and 'b1' not in inspect.signature(make).parameters:
            raise ValueError(f"param group {self.name!r} schedules b1, and its optimizer "
                             f"takes no b1 (adam, adamw and lamb do)")
        solver = scheduled_solver(make, self.rate(learning_rate), steps, opts, self.b1,
                                  self.learning_rate_multiplier, clocked=clocked)
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
            name = parameter_name(path)
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
    law = power_law(peak, a, b, c)
    warm = law(warmup_steps) if warmup_steps else peak
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


def power_law(peak: jax.typing.ArrayLike, a: jax.typing.ArrayLike, b: float,
              c: float = 1.0) -> Callable[[jax.typing.ArrayLike], jax.Array]:
    """The power scheduler's law at a step, `min(peak, a * (step * c) ** b)` with its step in float32
    (`power_schedule`), as a schedule of steps and one over a run's progress read it (`Power`)."""
    def law(step: jax.typing.ArrayLike) -> jax.Array:
        return jnp.minimum(peak, a * (jnp.asarray(step, jnp.float32) * c) ** b)

    return law


def anneal(start: jax.Array | float, stop: jax.Array | float, elapsed: jax.Array,
           span: jax.Array | float) -> jax.Array:
    """A half cosine from `start` to `stop`, `elapsed` of `span` through it (`OneCycle`): each end weighed
    by the cosine's halves, `start * c + stop * (1 - c)`, so it lands on each end exactly and takes any two
    rates, a zero start or ends far apart; torch's form, `stop + (start - stop) / 2 * (cos + 1)`, rounds the
    difference of the ends in float32. The angle is optax's cosine decay's, `pi * elapsed / span`, in that
    order: near a phase's end the cosine's halves cancel, so the angle's rounding is most of the rate's."""
    turned = jnp.cos(jnp.pi * elapsed / span)
    return start * (0.5 * (1 + turned)) + stop * (0.5 * (1 - turned))


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


def _after(progress: jax.Array, start: jax.Array) -> jax.Array:
    """How far a run at `progress` stands through the piece from `start` to its end, 0 to 1."""
    return jnp.clip((progress - start) / jnp.maximum(1.0 - start, jnp.finfo(progress.dtype).tiny), 0.0, 1.0)


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

    def warmup(self, steps: int | None) -> int | None:
        """Its warmup in its own steps over a run of `steps` updates, its share of the run resolved
        (`_warmup`), as a run's plan records it; None for a schedule with no warmup of steps."""
        return None

    def _warmup(self, written: int | str, steps: int | None, horizon: int | None = None) -> int:
        """A warmup of `written` steps in its own steps over a run of `steps` of them: as written, or a
        share of the run such as `5%` (`dew._records.share`), the whole steps nearest it, of the run's
        steps or, where the run has none, of the schedule's own `horizon`."""
        if not isinstance(written, str):
            return written
        return round(share(written) * self._horizon(steps if steps is not None else horizon, "warmup_steps"))

    def values(self, steps: int | None, rates: Rates | None = None
               ) -> optax.Schedule:
        """Return the value at each of the schedule's own steps, over `steps` of them, from its
        rate-valued fields `rates` (`rates()`). A value derived from the rates alone, as a ratio of
        two, is the configured fields', which a solver's operands keep: they differ from them by
        one factor, a param group's multiplier."""
        raise NotImplementedError

    def peak(self) -> float:
        """The highest value the schedule takes: its largest rate-valued field, a warmup's or decay's
        ends, where its formula reaches no higher."""
        return max(self.rates().values(), default=1.0)

    def run_relative(self) -> bool:
        """Whether its horizon is the run's, so a run that counts no updates ahead (one bounded in
        examples) reads it over the run's progress (`progressed`); a schedule whose every horizon is its
        own reads its updates there as anywhere."""
        return False

    def written_warmup(self) -> int | str | float | None:
        """Its warmup as written, in its own steps or as a share of the run, which a run that counts no
        updates ahead records as it is (`OptimConfig.warmups`); None where it has none."""
        return None

    def marks(self) -> dict[str, int]:
        """The update counts its law on a run's progress starts a piece at (`progressed`), by name: the
        end of a warmup counted in updates, say. The run records where its progress stood as its updates
        reached each (`ProgressState.marks`)."""
        return {}

    def progressed(self, rates: Rates, progress: jax.Array, updates: jax.Array,
                   marks: Mapping[str, jax.Array]) -> jax.Array:
        """Its value where a run that counts no updates ahead stands: at `progress`, the share of the
        run's bound it has consumed (`dew.state.TrainState.consumed`), after `updates` updates, the
        progress it stood at as it reached each count it `marks`. Each piece written in updates (a
        warmup of 1000) reads the updates, and each that lasts to the run's end reads the progress from
        where it starts, its law kept: a share of the run (`5%`) is that share of the progress, and the
        run's end its whole. A schedule's own steps (`every`) step its updates alone."""
        raise NotImplementedError(f"{type(self).__name__} reads its run's horizon and states no law on the "
                                  "run's progress")

    def _counted(self, updates: jax.Array) -> jax.Array:
        """`updates` in its own steps (`every`)."""
        return jnp.asarray(updates) // self.every

    def _warm(self, written: int | str, progress: jax.Array, updates: jax.Array,
              marks: Mapping[str, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
        """A warmup of `written` on a run's progress: how far through it the run stands, 0 to 1 (`in`),
        whether it is over, and the progress it ended at, its law's next piece's start: written in
        updates, of its own steps and its end marked (`marks['warmup']`); a share of the run, of the
        progress."""
        if isinstance(written, str):
            span = share(written)
            through = jnp.clip(progress / span, 0.0, 1.0) if span else jnp.ones_like(progress)
            return through, progress >= span, jnp.asarray(span, progress.dtype)
        count = self._counted(updates)
        through = (jnp.clip(count.astype(progress.dtype) / written, 0.0, 1.0) if written
                   else jnp.ones_like(progress))
        return through, count >= written, marks.get("warmup", jnp.zeros([], progress.dtype))


@dataclasses.dataclass(frozen=True)
class Cosine(ScheduleBase):
    """A linear warmup from `init_value` to `peak_value`, then a cosine decay to `end_value` at
    step `decay_steps`, or at the run's end when that is None.

    It is `optax.warmup_cosine_decay_schedule`.
    `warmup_steps` must be smaller than the decay horizon in schedule steps.
    """

    peak_value: float = dataclasses.field(metadata={'rate': True})
    warmup_steps: int | str
    """Its warmup in its own steps, or a share of the run such as `5%` (`ScheduleBase._warmup`): stated,
    as `optax.warmup_cosine_decay_schedule` takes it, which has no default either."""
    end_value: float = dataclasses.field(default=0.0, metadata={'rate': True})
    init_value: float = dataclasses.field(default=0.0, metadata={'rate': True})
    decay_steps: int | None = None

    def warmup(self, steps: int | None) -> int:
        return self._warmup(self.warmup_steps, steps, self.decay_steps)

    def run_relative(self) -> bool:
        return self.decay_steps is None

    def written_warmup(self) -> int | str | float | None:
        return self.warmup_steps

    def marks(self) -> dict[str, int]:
        warmup = self.warmup_steps
        return {"warmup": warmup * self.every} if isinstance(warmup, int) and warmup else {}

    def progressed(self, rates: Rates, progress: jax.Array, updates: jax.Array,
                   marks: Mapping[str, jax.Array]) -> jax.Array:
        """Its warmup, then its cosine decay from where the warmup ended to the run's end."""
        if isinstance(self.warmup_steps, str) and share(self.warmup_steps) >= 1:
            raise ValueError(f"Cosine's warmup of {self.warmup_steps} leaves its run no decay")
        through, over, start = self._warm(self.warmup_steps, progress, updates, marks)
        alpha = 0.0 if self.peak_value == 0.0 else self.end_value / self.peak_value
        # `values`' optax pieces, each read at the fraction of its own span the run stands through.
        warm = optax.linear_schedule(rates['init_value'], rates['peak_value'], 1)(through)
        decayed = optax.cosine_decay_schedule(rates['peak_value'], 1, alpha)(_after(progress, start))
        return jnp.where(over, decayed, warm)

    def values(self, steps: int | None, rates: Rates | None = None
               ) -> optax.Schedule:
        rates = self.rates() if rates is None else rates
        decay_steps = self._horizon(steps if self.decay_steps is None else self.decay_steps, "decay_steps")
        warmup = self.warmup(steps)
        if warmup >= decay_steps:
            raise ValueError(f"Cosine needs warmup_steps={self.warmup_steps} smaller than "
                             f"decay_steps={decay_steps}")
        # `optax.warmup_cosine_decay_schedule`'s own pieces, joined as it joins them; it forms the
        # floor's ratio by branching on the peak, so the ratio is the configured fields' (`values`).
        alpha = 0.0 if self.peak_value == 0.0 else self.end_value / self.peak_value
        return optax.join_schedules([
            optax.linear_schedule(rates['init_value'], rates['peak_value'], warmup),
            optax.cosine_decay_schedule(rates['peak_value'], decay_steps - warmup, alpha),
        ], [warmup])


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
    warmup_steps: int | str
    """Its warmup in its own steps, or a share of the run such as `5%` (`ScheduleBase._warmup`)."""
    a: float = dataclasses.field(metadata={'rate': True})
    b: float = -0.51
    c: float = 1.0
    tail: PowerTail | None = None

    def warmup(self, steps: int | None) -> int:
        return self._warmup(self.warmup_steps, steps, None if self.tail is None else self.tail.steps)

    def peak(self) -> float:
        """`peak_value`, the cap the power law climbs to."""
        return self.peak_value

    def run_relative(self) -> bool:
        return isinstance(self.warmup_steps, str) or (self.tail is not None and self.tail.steps is None)

    def written_warmup(self) -> int | str | float | None:
        return self.warmup_steps

    def marks(self) -> dict[str, int]:
        tail = self.tail
        return {"tail": tail.start * self.every} if tail is not None and tail.steps is None else {}

    def progressed(self, rates: Rates, progress: jax.Array, updates: jax.Array,
                   marks: Mapping[str, jax.Array]) -> jax.Array:
        """Its power law of its updates after its warmup, and its tail from `tail.start` to the run's end.
        A warmup written as a share of the run rises over that share of the progress to the law's value
        at the update it reaches, the law scaled by how far through it the run stands."""
        count = self._counted(updates)
        law = power_law(rates['peak_value'], rates['a'], self.b, self.c)
        written = self.warmup_steps
        if isinstance(written, str):
            through, _, _ = self._warm(written, progress, updates, marks)
            value = law(jnp.maximum(count, 1)) * through
        elif written:
            value = jnp.where(count < written, optax.linear_schedule(0.0, law(written), written)(count),
                              law(count))
        else:
            value = law(count)
        tail = self.tail
        if tail is None:
            return value
        start = law(tail.start)
        if tail.steps is None:
            through_tail = _after(progress, marks["tail"])
            decayed = optax.linear_schedule(start, rates['tail.end_value'], 1)(through_tail)
        else:
            decayed = optax.linear_schedule(start, rates['tail.end_value'], tail.steps - tail.start)(
                count - tail.start)
        return jnp.where(count >= tail.start, decayed, value)

    def values(self, steps: int | None, rates: Rates | None = None
               ) -> optax.Schedule:
        rates = self.rates() if rates is None else rates
        tail = self.tail
        warmup = self.warmup(steps)
        if tail is None:
            return power_schedule(rates['peak_value'], warmup, rates['a'], self.b, self.c)
        return power_schedule(
            rates['peak_value'], warmup, rates['a'], self.b, self.c, decay_start=tail.start,
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
    warmup_steps: int | str = 0
    """Its warmup in its own steps, or a share of the run such as `5%` (`ScheduleBase._warmup`)."""
    decay_start: int | None = None
    decay_steps: int | None = None
    end_value: float = dataclasses.field(default=0.0, metadata={'rate': True})

    def warmup(self, steps: int | None) -> int:
        return self._warmup(self.warmup_steps, steps, self.decay_steps)

    def run_relative(self) -> bool:
        return self.decay_steps is None

    def written_warmup(self) -> int | str | float | None:
        return self.warmup_steps

    def marks(self) -> dict[str, int]:
        warmup, start = self.warmup_steps, self.decay_start
        return {**({"warmup": warmup * self.every} if isinstance(warmup, int) and warmup else {}),
                **({"decay_start": start * self.every} if start is not None else {})}

    def progressed(self, rates: Rates, progress: jax.Array, updates: jax.Array,
                   marks: Mapping[str, jax.Array]) -> jax.Array:
        """Its warmup, its constant rate to `decay_start` (the warmup's end where None), and its linear
        decay from there to the run's end."""
        through, over, start = self._warm(self.warmup_steps, progress, updates, marks)
        decaying = over
        if self.decay_start is not None:
            decaying, start = self._counted(updates) >= self.decay_start, marks["decay_start"]
        peak = rates['peak_value']
        # `linear_schedule`'s optax pieces, each read at the fraction of its own span the run stands through.
        decayed = optax.linear_schedule(peak, rates['end_value'], 1)(_after(progress, start))
        warm = optax.linear_schedule(0.0, peak, 1)(through)
        return jnp.where(decaying, decayed, jnp.where(over, peak, warm))

    def values(self, steps: int | None, rates: Rates | None = None
               ) -> optax.Schedule:
        rates = self.rates() if rates is None else rates
        return linear_schedule(rates['peak_value'], self.warmup(steps), self.decay_start,
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
    warmup_fraction: float | str = 0.3
    """The share of the run its rise lasts, torch's `pct_start`: a fraction, or a share such as `30%`."""

    def rates(self) -> dict[str, float]:
        """The peak, and the init and end values with torch's derived defaults resolved."""
        init = self.peak_value / 25 if self.init_value is None else self.init_value
        return {'peak_value': self.peak_value, 'init_value': init,
                'end_value': init / 1e4 if self.end_value is None else self.end_value}

    def _rise(self, steps: int) -> float:
        """The step its rate peaks at over a run of `steps`, torch's phase end, kept a float as torch
        keeps it."""
        fraction = self.warmup_fraction
        return (share(fraction) if isinstance(fraction, str) else fraction) * steps - 1

    def warmup(self, steps: int | None) -> int:
        """The steps of its rise, to the whole step its rate peaks at (`_rise`)."""
        return round(self._rise(self._horizon(steps, "trainer.steps")))

    def run_relative(self) -> bool:
        return True

    def written_warmup(self) -> int | str | float | None:
        return self.warmup_fraction

    def progressed(self, rates: Rates, progress: jax.Array, updates: jax.Array,
                   marks: Mapping[str, jax.Array]) -> jax.Array:
        """Its rise over `warmup_fraction` of the run's progress, then its fall to the run's end, each a
        half cosine as over a run of steps; torch's peak a step before the share's end and its last
        value a step before the run's are the share's and the run's ends here."""
        fraction = self.warmup_fraction
        rise = share(fraction) if isinstance(fraction, str) else fraction
        if not 0 < rise < 1:
            raise ValueError(f"a one-cycle schedule with warmup fraction {fraction} has no rise or no fall")

        at = jnp.clip(progress, 0.0, 1.0)
        return jnp.where(at <= rise, anneal(rates['init_value'], rates['peak_value'], at, rise),
                         anneal(rates['peak_value'], rates['end_value'], at - rise, 1 - rise))

    def values(self, steps: int | None, rates: Rates | None = None
               ) -> optax.Schedule:
        rates = self.rates() if rates is None else rates
        steps = self._horizon(steps, "trainer.steps")
        rise, last = self._rise(steps), steps - 1
        if not 0 < rise < last:
            raise ValueError(f"a one-cycle schedule over {steps} steps with warmup fraction "
                             f"{self.warmup_fraction} has no rise or no fall; give it more steps")
        # Each phase a half cosine (`anneal`); torch's own form put the run's first rate 25 times further
        # from float64 than optax's cosine decay.
        def value(count) -> jax.Array:
            # The default float: float32 in a run, float64 where a test reads the definition under x64.
            step = jnp.minimum(jnp.asarray(count, float), last)
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

    def peak(self) -> float:
        """Its higher end, `offset` added."""
        return self.offset + max(self.init_value, self.end_value)

    def run_relative(self) -> bool:
        return self.decay_steps is None

    def progressed(self, rates: Rates, progress: jax.Array, updates: jax.Array,
                   marks: Mapping[str, jax.Array]) -> jax.Array:
        """Its geometric decay over the run's progress: `init_value * (end_value / init_value) ** progress`,
        plus `offset`."""
        if self.init_value <= 0 or self.end_value <= 0:
            raise ValueError(f"an exponential decay needs positive init_value and end_value, got "
                             f"{self.init_value} and {self.end_value}")
        ratio = self.end_value / self.init_value
        return rates['offset'] + rates['init_value'] * ratio ** jnp.clip(progress, 0.0, 1.0)

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


if TYPE_CHECKING:
    # tyro reads the runtime annotation, a Union of the aliased schedules, and a type checker cannot
    # read a variable in a type expression. Both get what they need: the base class statically, the
    # union at runtime, once every schedule above is defined (`dew.registry.schedules`).
    type ScheduleSpec = ScheduleBase
else:
    ScheduleSpec = schedules.union

def _rate_argument() -> tyro.constructors.PrimitiveConstructorSpec:
    """The learning rate as one command-line value, the record a run holds: a number, the constant
    rate, or a schedule's record as JSON (`dew.registry.from_record`)."""
    annotation = float | ScheduleSpec

    def number(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def read(given: list[str]) -> float | ScheduleBase:
        value = json.loads(given[0])
        if number(value):
            return float(value)
        built = from_record(annotation, value, dtypes=False)
        if not isinstance(built, ScheduleBase):
            raise TypeError(f"--optim.learning-rate is a number or a schedule's record, not {given[0]}")
        return built

    return tyro.constructors.PrimitiveConstructorSpec(
        nargs=1, metavar="RATE|JSON", instance_from_str=read,
        is_instance=lambda given: number(given) or isinstance(given, ScheduleBase),
        str_from_instance=lambda given: [json.dumps(to_record(given, annotation))])


@dataclasses.dataclass(frozen=True)
class OptimConfig:
    """Holds the optimizer, learning-rate schedule and gradient clipping."""

    optimizer: Optimizer | str = "adamw"
    """One of Dew's optimizers (`OPTIMIZER_MAP`), or any Optax optimizer factory by its import path,
    `optax:sgd` or `optax:adafactor` (`optimizers`), built as `factory(learning_rate,
    **optimizer_opts)` with `b1`, `b2` and `weight_decay` where the config sets them, so an experiment
    on an Optax transformation records the same optimizer it ran. The decay spares biases and norm
    scales for a factory that takes a `mask`."""
    optimizer_opts: JsonDict = dataclasses.field(default_factory=dict)
    learning_rate: Annotated[float | ScheduleSpec, _rate_argument()] = 2.7e-4
    """The learning rate: a constant, or a schedule as one typed record per kind (`dew.optim`).

    The kinds are cosine, power (lm-engine's power law with an optional linear
    tail), linear, one_cycle (torch's `OneCycleLR`) and exponential; each record
    holds only its own fields, and `every` steps any of them once per that many
    updates. On the command line it is one value: `--optim.learning-rate 3e-4`, or a
    schedule's record, `--optim.learning-rate '{"class": "cosine", "fields": {...}}'`.
    """
    weight_decay: float | None = None
    """The weight decay passed to the optimizer, None keeping its own (adamw's 1e-4, the others'
    0); for 'adam' it is torch's coupled L2 penalty, added to the gradient before the moments
    read it, for 'adamw' the decoupled decay. Either decay, the default included, spares biases
    and vectors such as a norm's scale, timm's split for every optimizer and for an Optax factory
    given one (`dew.optim.weight_decay_mask`, its `mask`), unless `spared` is off; Muon decays them as
    Moonlight does, and a param group decays all it claims unless it says (`ParamGroup.spared`)."""
    spared: bool = True
    """Whether the decay spares biases and vectors such as a norm's scale (`weight_decay_mask`), where the
    optimizer takes a mask: nanoGPT's, Megatron Core's and timm's split, since decaying a norm's scale or
    a bias toward zero regularizes nothing a matrix's decay does and shifts what the layer computes.
    Optax decays every parameter (`mask=None`), which False restores. A param group decays all it claims
    unless its own `spared` says otherwise (`ParamGroup.spared`); a run's default groups follow this."""
    b1: float | None = None
    """Adam's first-moment decay, for adam, adamw and lamb; None keeps the optimizer's."""
    b2: float | None = None
    """Adam's second-moment decay, for adam, adamw and lamb; None keeps the optimizer's."""
    param_groups: Annotated[tuple[ParamGroup, ...], record_argument(tuple[ParamGroup, ...])] = ()
    """Per-group learning rates, momentum schedules, weight decay and bounds; the first
    matching group wins.

    Empty runs one optimizer over every parameter. `ParamGroup.mup` is lm-engine's muP split.
    """
    clip_grads: float = 0.0
    state_dtype: Literal["float32", "bfloat16", "int8"] = "float32"
    """The dtype of Adam's moments in memory.

    float32 stores adam's and adamw's two moments, and Muon's momentum, in
    float32 whatever the parameters' dtype, so bfloat16 or float16 weights
    train on full-precision moments. bfloat16 stores both with stochastic
    rounding (`dew.optim.bf16_moments`), for adam and adamw only. That halves
    the optimizer state and reduces the update's memory traffic. int8 stores
    both blockwise in 8 bits (`dew.optim.int8_moments`, bitsandbytes' and
    torchao's 8-bit Adam), for adam and adamw only: a quarter of float32's,
    about 2 bytes a parameter for the two.
    """
    forced_weight_normalization: bool = False
    """Whether to renormalize every magnitude-preserving weight (`dew.nn.mp.MPConv`) after each update.

    This is EDM2's forced weight normalization, which its `edm2_unet` trains with
    (`dew.nn.mp.forced_weight_normalization`).
    """
    ema_profiles: tuple[float, ...] = ()
    """Relative standard deviations of the power-function EMAs a run keeps for post-hoc EMA.

    See `dew.optim.power_profiles`; Karras et al. use (0.05, 0.10). Every
    checkpoint save snapshots these averages, and `Checkpoints.posthoc_ema` builds
    an average with any other relative standard deviation from the snapshots.
    Empty keeps none.
    """

    def warmups(self, steps: int | None, clocked: bool = False) -> dict[str, int | str | float]:
        """Each of its schedules' warmup in that schedule's own steps over a run of `steps` updates,
        its share of the run resolved (`ScheduleBase.warmup`), by where the config holds it, as a
        run's plan records it; a schedule with no warmup of steps is not among them. A `clocked` run
        counts no updates ahead, so a warmup over the run's progress is recorded as written, its share
        of the examples the run consumes (`ScheduleBase.progressed`)."""
        held: list[tuple[str, object]] = [] if self.param_groups else [("learning_rate", self.learning_rate)]
        for group in self.param_groups:
            # The schedules the group's solver runs (`ParamGroup.rate`): the config's own where the
            # group names none.
            rate = "learning_rate" if group.schedule is None else f"param_groups.{group.name}.schedule"
            held += [(rate, group.rate(self.learning_rate)), (f"param_groups.{group.name}.b1", group.b1)]
        warmups: dict[str, int | str | float] = {}
        for name, schedule in held:
            if not isinstance(schedule, ScheduleBase):
                continue
            if clocked and schedule.run_relative():
                written = schedule.written_warmup()
                if written:
                    warmups[name] = written
                continue
            warmup = schedule.warmup(None if steps is None else steps // schedule.every)
            if warmup is not None:
                warmups[name] = warmup
        return warmups

    def groups(self, params) -> dict[str, str] | None:
        """Where this optimizer's split sends each of `params` and why (`muon_partition`), for the
        optimizers that split by Muon's rule ('muon', 'muonclip'); None for one that splits none. Read
        under the axes the modules declare, as its state is initialized."""
        return muon_partition(params) if self.optimizer in ("muon", "muonclip") else None

    def build(self, steps: int | None, clocked: bool = False) -> optax.GradientTransformation:
        """Build the optimizer this config describes, with its schedule, parameter groups and clipping.

        `steps` is the run's length in updates, which the schedule decays over unless the
        config names its own end; a `clocked` run counts none ahead (it is bounded in examples), and a
        schedule whose horizon is the run's reads its progress, which each update is handed
        (`scheduled_solver`). With `param_groups`, one optimizer runs per group under
        `optax.multi_transform`, each on its own schedule (the config's when it names
        none) times its multiplier, with its own weight decay, `b1` schedule and bounds
        (`ParamGroup.solver`); the global-norm clip still reads every gradient
        together, before the groups split them. Each solver holds its rate's
        fields in its state (`dew.optim.scheduled_solver`), so runs at
        rates of one schedule's timings compile one step.
        """
        opts = dict(self.optimizer_opts)
        if {'b1', 'b2'} & opts.keys():
            raise ValueError("b1 and b2 are fields of OptimConfig, not optimizer_opts")
        given = {'b1': self.b1, 'b2': self.b2, 'weight_decay': self.weight_decay}
        opts.update({name: value for name, value in given.items() if value is not None})
        make = optimizers[self.optimizer]
        if self.state_dtype != 'float32':
            if self.optimizer not in STATE_OPTIMIZERS[self.state_dtype]:
                raise ValueError(
                    f"state_dtype={self.state_dtype!r} stores Adam's moments, which "
                    f"{sorted(STATE_OPTIMIZERS[self.state_dtype])} have; {self.optimizer!r} does not")
            make = STATE_OPTIMIZERS[self.state_dtype][self.optimizer]
        elif self.optimizer in ('adam', 'adamw'):
            opts = {'mu_dtype': 'float32', 'nu_dtype': 'float32', **opts}
        elif self.optimizer in ('muon', 'muonclip'):
            opts = {'mu_dtype': 'float32', **opts}
        if self.param_groups:
            names = [group.name for group in self.param_groups]
            if len(set(names)) != len(names):
                raise ValueError(f"param group names repeat: {names}")
            claiming = [group.name for group in self.param_groups if group.fresh]
            if claiming:
                raise ValueError(f"param groups {claiming} claim the modules a run draws fresh, which the "
                                 f"run names (`OptimConfig.with_fresh`) before it builds their optimizer")
            solvers = {}
            for group in self.param_groups:
                decay = {} if group.weight_decay is None else {'weight_decay': group.weight_decay}
                grouped = {**opts, **decay}
                spared = {'mask': weight_decay_mask} if group.spared and self._spares(make) else {}
                solvers[group.name] = group.solver(make, self.learning_rate, steps, {**grouped, **spared},
                                                   clocked=clocked)
            solver = optax.multi_transform(solvers, param_labels(self.param_groups))
        else:
            if self.spared and self._spares(make):
                opts['mask'] = weight_decay_mask
            solver = scheduled_solver(make, self.learning_rate, steps, opts, clocked=clocked)

        if self.clip_grads > 0:
            solver = optax.chain(optax.clip_by_global_norm(self.clip_grads), solver)
        if self.forced_weight_normalization:
            from dew.nn.mp import forced_weight_normalization

            solver = optax.chain(solver, forced_weight_normalization())
        if self.ema_profiles:
            solver = power_profiles(solver, self.ema_profiles)
        return solver

    def _spares(self, make: Callable[..., optax.GradientTransformation]) -> bool:
        """Whether the optimizer's decay takes timm's split (`weight_decay_mask`): a factory that takes a
        decay mask, and never Muon, which decays every parameter it updates as Moonlight does."""
        return self.optimizer not in ('muon', 'muonclip') and 'mask' in inspect.signature(make).parameters

    def fresh_groups(self) -> tuple[ParamGroup, ...]:
        """The groups a run takes where it names none and draws modules fresh beside a loaded base: those
        modules at `FRESH_LEARNING_RATE` at the peak of this config's schedule, on its shape (its rate
        times their quotient), or at its own where that is higher, and every other parameter at its own,
        each sparing what the ungrouped decay spares (`ParamGroup.spared`)."""
        rate = self.learning_rate
        peak = rate.peak() if isinstance(rate, ScheduleBase) else rate
        multiplier = max(1.0, FRESH_LEARNING_RATE / peak) if peak > 0 else 1.0
        # The groups a run takes in place of none decay as its ungrouped optimizer would.
        return (ParamGroup("fresh", fresh=True, learning_rate_multiplier=multiplier, spared=self.spared),
                ParamGroup("base", ("*",), spared=self.spared))

    def with_fresh(self, modules: Sequence[str]) -> OptimConfig:
        """This config with each group that claims the fresh modules (`ParamGroup.fresh`) matching the
        parameters of `modules`, each a module's path under `params` as a parameter's name begins
        (`parameter_name`), ahead of its own patterns: the groups a run records and trains in, its
        modules drawn fresh beside a loaded base (`Objective.fresh_modules`). Refused where such a group
        then matches nothing, in a run that draws nothing fresh."""
        claimed = tuple(f"{glob.escape(module)}/*" for module in modules)
        groups = []
        for group in self.param_groups:
            if group.fresh:
                if not claimed and not group.patterns:
                    raise ValueError(f"param group {group.name!r} claims the modules its run draws fresh "
                                     f"beside a loaded base, and it draws none")
                group = dataclasses.replace(group, patterns=(*claimed, *group.patterns), fresh=False)
            groups.append(group)
        return dataclasses.replace(self, param_groups=tuple(groups))

    def rates(self) -> dict[str | None, dict[str, float]]:
        """The rate fields each of `build`'s solvers holds in its state
        (`dew.optim.scheduled_solver`), by param group, or under None
        for the one solver of a config without groups. A resumed run sets
        them over its checkpoint's (`with_rates`), so the configured
        rate is the one it steps at."""
        if not self.param_groups:
            return {None: rates_of(self.learning_rate)}
        return {group.name: group.rates(self.learning_rate) for group in self.param_groups}


__all__ = ["NEWTON_SCHULZ", "BoundsState", "Cosine", "DeltaState", "Exponential", "Linear", "M3State",
           "OneCycle", "OptimConfig", "ParamGroup", "Power", "PowerProfilesState", "PowerTail",
           "QKClipState", "ScheduleBase", "ScheduleSpec", "delta_gradient_descent", "scale_by_m3", "stepped"]
