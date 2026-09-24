"""Mixture of experts: the router, and the experts as one grouped matmul.

Patterned on MaxText's `RoutedMoE` (`maxtext src/maxtext/layers/moe.py:419`,
Apache 2.0). The math ported is top-k selection (`:751` `get_topk`), DeepSeek's
group-limited routing, which selects on the biased scores and gates on the
unbiased ones (`:881-908` `deepseek_routing`), its weight scaling
(`:835-841`), the aux-loss-free bias update (`:238-261`, lifted below), and the
grouped matmul over tokens sorted by expert (`:1500` `sparse_matmul`).

The two routers this reproduces are `MixtralSparseMoeBlock`, which softmaxes
over the experts, takes the top k and renormalises, and `DeepseekV3MoE`, which
scores with a sigmoid, selects with a per-expert bias added, limits the choice
to the best expert groups, renormalises and scales, then adds one shared
expert every token takes. DeepSeek V4's `DeepseekV4TopKRouter` and
`DeepseekV4Experts` contribute the sqrt(softplus) score and the swiglu limit.
All are transformers 5.16.1; `tests/test_moe.py` holds the fp32 parity
numbers and `tools/moe_reference.py` writes the fixtures they run against.

Parameters follow the Hugging Face layout of a sparse decoder layer, with the
experts stacked on an expert dimension: `mlp/gate/kernel` is `[embed, exp]`,
`mlp/experts/gate_proj/kernel` and `mlp/experts/up_proj/kernel` are `[exp,
embed, mlp]`, and `mlp/experts/down_proj/kernel` is `[exp, mlp, embed]`. A
checkpoint's per-expert `mlp.experts.N.gate_proj.weight` tensors stack into one
leaf, which is the layout `jax.lax.ragged_dot` takes and the `expert` mesh axis
shards.
"""

import dataclasses
import functools
import importlib
import math
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn, struct
from flax.linen.dtypes import canonicalize_dtype, promote_dtype
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name
from jax.custom_derivatives import SymbolicZero
from jax.sharding import PartitionSpec as P

from .blocks import normal_kernel
from .kernels.generation import device_generation, triton_runs
from .kernels.grouped_matmul import grouped_projection, ragged_dot_runs
from .precision import rounded_operand, rounded_to
from .sharding import EXPERT_AXIS, STAGE_AXIS, LogicalAxes, batch_axes, logical_axes, logical_spec, mesh_axes

# 'softmax' normalizes a token's affinities over the experts (Mixtral,
# Qwen3.5); 'sigmoid' scores each expert on its own (DeepSeek V3, GLM, Kimi,
# LLaDA2); 'sqrtsoftplus' is DeepSeek V4's sqrt(softplus(logit))
# (transformers activations.py, SqrtSoftplusActivation), unbounded above
# where a sigmoid saturates.
SCORE_FUNCTIONS = ('softmax', 'sigmoid', 'sqrtsoftplus')

# 'auto' resolves per hardware generation through GROUPED_MATMUL_BY_GENERATION.
GROUPED_MATMULS = ('auto', 'xla', 'pallas', 'tokamax')
EXPERT_DISPATCHES = ('global', 'exchange')

# DeepSeek divides the selected weights by their sum plus this, so a token
# whose sigmoid scores are all zero stays finite
# (modeling_deepseek_v3.py:166, maxtext layers/moe.py:839). A softmax over the
# experts sums to one, which cannot reach it, and in fp32 the term changes no
# bit of a denominator above 1e-12.
WEIGHT_SUM_EPSILON = 1e-20

type Routes = tuple[jax.Array, jax.Array | None]
"""A routing replay for one layer: `[..., top_k]` expert ids the rollout
engine used, and `[...]` booleans marking the tokens its record covers (None
for all of them).

Routing replay (R3, arXiv 2510.11370; verl 12ebe0c `router_replay_patch.py`
over Megatron's `topk_routing_with_score_function`) trains a mixture on the
experts the engine used, so a top-k flip between the engine's numerics and
the trainer's cannot move a token to experts that never produced its sample.
Only the selection is replayed: the gate weights are gathered from this
forward's scores, so the router keeps its gradient. A token the record does
not cover (the last sampled id, padding) keeps the router's own choice, as
verl's replay mask does. The stack hands each layer its slice of a
`[B, S, layers, top_k]` record (`CausalTransformer.hidden_states`). The
sown `indices` are the replayed ones, so a balancing bias counts the experts
the tokens went to, as Megatron's R3 path in verl does."""


def chosen_experts(selected: Callable[[], jax.Array], shape: tuple[int, ...],
                   routes: Routes | None) -> jax.Array:
    """The experts a router uses: `selected()`, or the replayed record.

    `shape` is `[..., top_k]`, what the router's own choice has. `selected`
    is not traced when a record covers every token.
    """
    if routes is None:
        return selected()
    replayed, covered = routes
    if replayed.shape != shape:
        raise ValueError(
            f"replayed routing is {tuple(replayed.shape)} for {shape[:-1]} "
            f"tokens choosing {shape[-1]} experts each")
    if not jnp.issubdtype(replayed.dtype, jnp.integer):
        raise ValueError(f"replayed routing holds expert ids, got {replayed.dtype}")
    # Engines ship the ids in the narrowest unsigned type that holds them.
    replayed = replayed.astype(jnp.int32)
    if covered is None:
        return replayed
    if covered.shape != shape[:-1]:
        raise ValueError(f"replay coverage is {tuple(covered.shape)} for {shape[:-1]} tokens")
    return jnp.where(covered[..., None], replayed, selected())


@struct.dataclass
class RouterMoments:
    """Additive routed-position statistics for global load-times-score loss."""
    scores: jax.Array
    counts: jax.Array
    positions: jax.Array
    top_k: int = struct.field(pytree_node=False)


def router_moments(scores: jax.Array, indices: jax.Array) -> RouterMoments:
    experts = scores.shape[-1]
    dtype = jnp.promote_types(scores.dtype, jnp.float32)
    # These counts enter a floating loss, unlike the exact integer bias effects.
    return RouterMoments(
        jnp.sum(scores.astype(dtype), axis=(0, 1)),
        jnp.bincount(indices.ravel(), length=experts).astype(dtype),
        jnp.asarray(scores.shape[0] * scores.shape[1], dtype), indices.shape[-1])


def global_router_loss(stats: RouterMoments, alpha: float) -> jax.Array:
    """DeepSeek V2 auxiliary loss after all routed positions have pooled."""
    scores = stats.scores.astype(jnp.promote_types(stats.scores.dtype, jnp.float32))
    positions = jax.lax.stop_gradient(stats.positions.astype(scores.dtype))
    counts = jax.lax.stop_gradient(stats.counts.astype(scores.dtype))
    denominator = jnp.where(positions > 0, positions, 1) ** 2 * stats.top_k
    return alpha * scores.size * jnp.vdot(counts, scores) / denominator


def sequence_router_losses(scores: jax.Array, indices: jax.Array,
                           alpha: float) -> jax.Array:
    """Compute one DeepSeek V2 load-times-score loss per intact sequence."""
    _, length, experts = scores.shape
    scores = scores.astype(jnp.promote_types(scores.dtype, jnp.float32))
    chosen = jax.nn.one_hot(indices, experts, dtype=scores.dtype)
    load = jnp.sum(chosen, axis=(1, 2)) / (length * indices.shape[-1] / experts)
    return jnp.sum(load * jnp.mean(scores, axis=1), axis=1) * alpha


def deepseek_v2_aux_loss(scores, indices, alpha: float, seq_aux: bool = True):
    """Score DeepSeek V2's expert balance (arXiv 2405.04434, section 2.1.4).

    Scores are [batch, sequence, experts], choices [batch, sequence, top_k].
    `seq_aux` forms the product within each sequence before averaging rows.
    The global variant pools all routed positions before forming the
    product.
    """
    if seq_aux:
        return jnp.mean(sequence_router_losses(scores, indices, alpha))
    return global_router_loss(router_moments(scores, indices), alpha)


def load_balance_update(counts: jax.Array, rate: jax.typing.ArrayLike) -> jax.Array:
    """Bias displacement from nonnegative per-expert counts.

    Integer counts must individually fit their dtype; their total need not.
    """
    if jnp.issubdtype(counts.dtype, jnp.integer):
        # Narrow count elements do not imply a narrow expert-count divisor.
        minimum = jnp.uint32 if jnp.issubdtype(counts.dtype, jnp.unsignedinteger) else jnp.int32
        counts = counts.astype(jnp.promote_types(counts.dtype, minimum))
        # Partial sums are represented divided by the expert count. Their
        # quotients never exceed the largest count, even when the total does.
        divisor = counts.size
        quotients, remainders = jnp.divmod(counts, divisor)

        def combine(left, right):
            quotient, remainder = left
            other_quotient, other_remainder = right
            gap = divisor - other_remainder
            carry = remainder >= gap
            remainder = jnp.where(carry, remainder - gap, remainder + other_remainder)
            return quotient + other_quotient + carry.astype(counts.dtype), remainder

        zero = jnp.zeros((), counts.dtype)
        average, remainder = jax.lax.reduce(
            (quotients, remainders), (zero, zero), combine, dimensions=(0,))
        direction = jnp.where(counts > average, -1,
                              jnp.where((counts < average) | (remainder > 0), 1, 0))
    else:
        direction = jnp.sign(jnp.sum(counts) / counts.size - counts)
    return jnp.asarray(rate) * direction


class Router(nn.Module):
    """Choose which experts a token goes to, and with what weight: `[..., k]`.

    The gate projection runs in fp32 whatever dtype the activations carry,
    as DeepSeek's router does (`modeling_deepseek_v3.py:146`).

    `expert_bias` is DeepSeek's aux-loss-free balancing bias
    (`e_score_correction_bias`, arXiv 2408.15664), kept in fp32 in the `moe`
    collection. It enters the selection only: a token's weights are gathered
    from the unbiased scores, so moving the bias changes which experts a token
    uses without changing what they contribute. Nothing here writes it;
    transformers holds it in an `nn.Buffer` and MaxText hands the update back
    to its caller (`layers/moe.py:965-972`). `load_balance_update` is that
    update, and the step that applies it owns the write. Gradients
    cannot reach the bias either, since it feeds only `jax.lax.top_k`'s
    integer indices.

    `expert_groups` above one is DeepSeek's node limit: experts are cut into
    that many groups, each group is scored by its two best experts, and a token
    may only choose inside the best `groups_per_token` of them. `group_score`
    names the group's score: 'top2' is V3's sum of its two best experts
    (`noaux_tc`), 'max' is V2's best expert alone (`group_limited_greedy`,
    modeling_deepseek_v2.py `DeepseekV2TopkRouter`).

    `normalize_weights` divides a token's selected weights by their sum,
    the reference's `norm_topk_prob`; V2's released configs set it false.

    `hash_vocab` set makes this DeepSeek V4's hash router
    (`DeepseekV4HashRouter`, modeling_deepseek_v4.py:1045-1073): a token's
    experts are the `top_k` entries of a fixed `tid2eid` table at its
    vocabulary id, `[hash_vocab, top_k]` in the `moe` collection beside the
    bias (the checkpoint's persistent buffer), and the learned gate still
    weights them, gathered from the scores the same way. The caller passes
    the token ids; the table is never written by training.

    `media_bias` is DeepSeek-V4.1's second balancing bias, which selects for
    the tokens of an image span in place of `expert_bias` (the release's
    `bias_vl`, model.py:807-820); the caller passes the `media` mask. It sits
    in the `moe` collection beside the other and nothing here writes it.
    """
    num_experts: int
    in_features: int
    top_k: int
    score_function: str = 'softmax'
    normalize_weights: bool = True
    routed_scaling_factor: float = 1.0
    expert_groups: int = 1
    groups_per_token: int = 1
    group_score: str = 'top2'
    expert_bias: bool = False
    media_bias: bool = False
    hash_vocab: int | None = None
    init_std: float | None = None  # normal std of the kernel; None: lecun normal
    precision: PrecisionLike = None

    def setup(self):
        if self.group_score not in ('top2', 'max'):
            raise ValueError(
                f"group_score must be 'top2' or 'max', got {self.group_score!r}")
        if self.score_function not in SCORE_FUNCTIONS:
            raise ValueError(
                f"score_function must be one of {list(SCORE_FUNCTIONS)}, got "
                f"{self.score_function!r}")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError(
                f"top_k must be between 1 and num_experts ({self.num_experts}), "
                f"got {self.top_k}")
        if self.expert_groups < 1 or self.num_experts % self.expert_groups:
            raise ValueError(
                f"expert_groups ({self.expert_groups}) must divide num_experts "
                f"({self.num_experts})")
        if not 1 <= self.groups_per_token <= self.expert_groups:
            raise ValueError(
                f"groups_per_token must be between 1 and expert_groups "
                f"({self.expert_groups}), got {self.groups_per_token}")
        if self.expert_groups > 1 and self.group_score == 'top2':
            per_group = self.num_experts // self.expert_groups
            if per_group < 2:
                raise ValueError(
                    "group scores are the sum of a group's two best experts, so "
                    f"a group needs at least two: {self.num_experts} experts in "
                    f"{self.expert_groups} groups leaves {per_group}")
        if self.expert_groups > 1:
            per_group = self.num_experts // self.expert_groups
            if self.groups_per_token * per_group < self.top_k:
                raise ValueError(
                    f"{self.groups_per_token} of {self.expert_groups} groups hold "
                    f"{self.groups_per_token * per_group} experts, fewer than the "
                    f"top_k of {self.top_k}")
        # The kernel is the router's own parameter, not a nested nn.Dense, so
        # the leaf is `gate/kernel` where a Hugging Face sparse layer keeps
        # `gate.weight`.
        self.kernel = self.param(
            'kernel', normal_kernel(self.init_std, nn.initializers.lecun_normal())['kernel_init'],
            (self.in_features, self.num_experts), jnp.float32)
        if self.expert_bias:
            self.bias = self.variable(
                'moe', 'e_score_correction_bias', jnp.zeros,
                (self.num_experts,), jnp.float32)
        if self.media_bias:
            self.bias_vl = self.variable('moe', 'media_bias', jnp.zeros, (self.num_experts,), jnp.float32)
        if self.hash_vocab is not None:
            if self.expert_bias or self.expert_groups > 1:
                raise ValueError(
                    "hash routing selects by the token table alone, so it has no "
                    "balancing bias and no expert groups")
            self.tid2eid = self.variable(
                'moe', 'tid2eid', jnp.zeros, (self.hash_vocab, self.top_k), jnp.int32)

    def __call__(self, x, tokens=None, media=None, routes: Routes | None = None):
        logits = self.logits(x)
        scores = self._activated(logits)
        indices = chosen_experts(lambda: self._selected(scores, tokens, media),
                                 (*scores.shape[:-1], self.top_k), routes)
        # The load each expert took, for the step that balances the bias:
        # written only when a caller opens the 'router' collection, and never
        # into the tree init returns, where it is not a variable.
        if not self.is_initializing():
            self.sow('router', 'indices', indices)
            # The V2 balance loss reads the scores beside the choices.
            self.sow('router', 'scores', scores)
            # The router z-loss squares each position's log partition.
            self.sow('router', 'log_z', jax.nn.logsumexp(logits, axis=-1))
        weights = jnp.take_along_axis(scores, indices, axis=-1)
        if self.normalize_weights:
            weights = weights / (jnp.sum(weights, axis=-1, keepdims=True)
                                 + WEIGHT_SUM_EPSILON)
        return weights * self.routed_scaling_factor, indices

    def _selected(self, scores, tokens, media):
        """The experts this router chooses on its own: `[..., top_k]`."""
        if self.hash_vocab is not None:
            if tokens is None:
                raise ValueError("hash routing selects by the token ids, which the caller passes")
            return self.tid2eid.value[jnp.asarray(tokens)]
        if tokens is not None:
            raise ValueError("only a hash router reads the token ids")
        selection = scores if not self.expert_bias else scores + self.bias.value
        if media is not None:
            selection = jnp.where(media[..., None], scores + self.bias_vl.value, selection)
        if self.expert_groups > 1:
            selection = jnp.where(self.group_mask(selection), selection, -jnp.inf)
        _, indices = jax.lax.top_k(selection, self.top_k)
        return indices

    def logits(self, x):
        """Each token's fp32 gate logit for every expert: `[..., num_experts]`."""
        return jnp.einsum('...d,de->...e', x.astype(jnp.float32), self.kernel,
                          precision=self.precision)

    def scores(self, x):
        """Each token's fp32 affinity for every expert: `[..., num_experts]`."""
        return self._activated(self.logits(x))

    def _activated(self, logits):
        if self.score_function == 'softmax':
            return jax.nn.softmax(logits, axis=-1)
        if self.score_function == 'sqrtsoftplus':
            return jnp.sqrt(jax.nn.softplus(logits))
        return jax.nn.sigmoid(logits)

    def group_mask(self, selection):
        """True where an expert sits in one of a token's best groups.

        Patterned on `maxtext layers/moe.py:843` `expert_group_mask`, which
        scores a group by its two best experts. The groups are the expert index
        cut into `expert_groups` contiguous blocks, the layout DeepSeek's
        `n_group` means.
        """
        per_group = self.num_experts // self.expert_groups
        grouped = selection.reshape(*selection.shape[:-1], self.expert_groups, per_group)
        if self.group_score == 'max':
            group_scores = jnp.max(grouped, axis=-1)
        else:
            best_two, _ = jax.lax.top_k(grouped, 2)
            group_scores = jnp.sum(best_two, axis=-1)
        _, groups = jax.lax.top_k(group_scores, self.groups_per_token)
        kept = jnp.sum(
            jax.nn.one_hot(groups, self.expert_groups, dtype=jnp.float32), axis=-2)
        return jnp.repeat(kept > 0, per_group, axis=-1)


# The grouped matmul 'auto' runs, per hardware generation (`device_generation`):
# the measured winner, forward plus backward, at lm-moe's shape (8192 rows,
# 768 -> 2048, 8 experts, bf16) and at 128 experts; numbers in
# docs/performance.md. On sm80 (A100), sm86 (RTX 3090) and sm89 (L4, RTX 4080) that is JAX's
# own Pallas kernels (`dew.nn.kernels.grouped_matmul`), 5x to 61x faster than
# XLA, which runs ragged_dot there as a product over every expert. On a TPU
# v5e and v6e it is XLA's ragged_dot; the one kernel that beats it at 8
# experts (tokamax's mosaic_tpu_v2, 1.11x-1.38x) cannot be a dependency:
# tokamax 0.0.14 pins typeguard==2.13.3 where tyro needs >=4. Every
# generation not listed runs 'xla': sm75 cannot compile the kernels, and
# sm90 and sm120 are unmeasured.
GROUPED_MATMUL_BY_GENERATION = {'sm80': 'pallas', 'sm86': 'pallas', 'sm89': 'pallas',
                                'v5e': 'xla', 'v6e': 'xla'}

# The kernel 'tokamax' names, per generation. tokamax's own dispatch tries its
# Mosaic kernel first: on a TPU that is the v1 kernel, 4x to 13x slower than
# XLA, and on sm89 a Mosaic GPU config that exceeds shared memory and raises.
# Only the forward runs on tokamax (`expert_projection` differentiates on
# XLA): its Triton backward faults (CUDA_ERROR_ILLEGAL_ADDRESS) on sm80 and
# sm89. Unmeasured generations run tokamax's 'xla'.
TOKAMAX_KERNEL_BY_GENERATION = {'sm80': 'triton', 'sm89': 'triton',
                                'v5e': 'mosaic_tpu_v2', 'v6e': 'mosaic_tpu_v2'}


def grouped_matmul_kernel(implementation: str, compute: Dtype, operands: tuple[Dtype, ...],
                          precision: PrecisionLike) -> str:
    """The one choice of grouped matmul: 'xla', 'pallas' or 'tokamax'.

    'auto' takes the hardware generation's measured one
    (`GROUPED_MATMUL_BY_GENERATION`) and 'xla' on an unmeasured generation.
    'pallas', named or chosen, needs a GPU the kernels compile for and a
    product they compute exactly (`ragged_dot_runs`); elsewhere it is 'xla'.
    `operands` are the dtypes of the input and the kernel as stored.
    """
    if implementation not in GROUPED_MATMULS:
        raise ValueError(
            f"implementation must be one of {list(GROUPED_MATMULS)}, got "
            f"{implementation!r}")
    chosen = (GROUPED_MATMUL_BY_GENERATION.get(device_generation(), 'xla')
              if implementation == 'auto' else implementation)
    if chosen != 'pallas':
        return chosen
    runs = ragged_dot_runs(compute, operands, precision)
    placed = triton_runs() or (implementation == 'pallas' and jax.default_backend() == 'cpu')
    return 'pallas' if runs and placed else 'xla'


def grouped_matmul(tokens: jax.Array, kernel: jax.Array, group_sizes: jax.Array, *,
                   implementation: str, precision: PrecisionLike = None,
                   preferred_element_type: Dtype | None = None) -> jax.Array:
    """Each row of `tokens`, `[rows, in]`, through the matrix of its expert in
    `kernel`, `[exp, in, out]`, for rows already sorted by expert.

    `group_sizes` is how many leading rows belong to expert 0, then to expert
    1, and so on. This is the raw call with JAX's own differentiation rules:
    'tokamax' is `tokamax.ragged_dot` with the kernel named per generation
    (`TOKAMAX_KERNEL_BY_GENERATION`, XLA elsewhere; `maxtext
    layers/moe.py:1633`), and every other implementation is
    `jax.lax.ragged_dot`. The Pallas kernels have no JAX differentiation
    rule, so they run behind `expert_projection`, which adds the precision
    contract routed experts train under and the gradients of every
    implementation.
    """
    if implementation not in GROUPED_MATMULS:
        raise ValueError(
            f"implementation must be one of {list(GROUPED_MATMULS)}, got "
            f"{implementation!r}")
    if implementation == 'tokamax':
        # tokamax is not a dependency (docs/concepts/moe.md), so it is
        # imported at the call.
        tokamax = importlib.import_module('tokamax')
        return tokamax.ragged_dot(
            tokens, kernel, group_sizes, precision=precision,
            preferred_element_type=preferred_element_type,
            implementation=TOKAMAX_KERNEL_BY_GENERATION.get(device_generation(), 'xla'))
    # A 16-bit product is exact at any precision, and XLA's TPU ragged-dot
    # kernel refuses 16-bit operands at HIGHEST ("Bad lhs type"), so they
    # multiply at DEFAULT: the same products, summed in the preferred type.
    if all(jnp.finfo(operand.dtype).bits == 16 for operand in (tokens, kernel)):
        precision = jax.lax.Precision.DEFAULT
    # XLA's TPU kernel writes values into the rows past the groups, where
    # other backends write zeros, and its lhs cotangent is the same kernel.
    # Those rows are zeroed going in and coming out, so neither the output
    # nor a dropped row's gradient carries them.
    grouped = jnp.arange(tokens.shape[0])[:, None] < jnp.sum(group_sizes)
    out = jax.lax.ragged_dot(
        jnp.where(grouped, tokens, 0), kernel, group_sizes, precision=precision,
        preferred_element_type=preferred_element_type)
    return jnp.where(grouped, out, 0)


def _local(value: jax.Array) -> bool:
    """Whether the kernels, which see local arrays and carry no manual-axis
    type, can take `value` where it is traced: no mesh axis outside a
    `shard_map` splits it, and no `shard_map` that checks varying axes
    holds it."""
    mesh = jax.sharding.get_abstract_mesh()
    split = any(mesh.shape[name] > 1 for name in mesh.axis_names
                if name not in mesh.manual_axes)
    return not split and not jax.typeof(value).mat.varying


def gather_expert_bias(bias: jax.Array, expert_ids: jax.Array, dtype: Dtype) -> jax.Array:
    """Expert-major biases in compute dtype, with master-precision cotangent sums.

    Out-of-range IDs are transport padding, contributing neither a value
    nor a gradient. Promotion before gathering prevents a bf16 scatter-add
    or scan carry from rounding partial bias gradients.
    """
    work = jnp.result_type(bias.dtype, dtype, jnp.float32)
    values = jnp.asarray(rounded_operand(bias, dtype)).astype(work)
    return values.at[expert_ids].get(mode='fill', fill_value=0).astype(dtype)


def expert_projection(x: jax.Array, kernel: jax.Array, group_sizes: jax.Array,
                      dtype: Dtype | None, implementation: str,
                      precision: PrecisionLike) -> jax.Array:
    """`grouped_matmul` under one precision contract for both dispatches.

    The operands are cast to `dtype` (flax's promotion when None), every
    contraction accumulates in at least fp32 and rounds once to the compute
    dtype, so a width split over a mesh axis rounds no partial sum. The
    tangent is `dx @ Q(kernel) + Q(x) @ dkernel` with `Q` the rounded operand
    values and the tangents in their own dtypes: a kernel gradient keeps the
    master dtype, an input gradient its input's, and both accumulate over
    the widest of the compute, input and kernel dtypes. The contract does
    not depend on placement.

    'xla' and 'tokamax' hold it in every differentiation mode: the forward
    runs on the chosen kernel and the tangent contractions on
    `jax.lax.ragged_dot`, whose transposes JAX defines. 'pallas' holds it in
    first-order reverse mode (`dew.nn.kernels.grouped_matmul`); a trace the
    kernels cannot take (`_local`) runs 'xla'. Measured against NumPy
    float64 sums of the rounded operands and three Adam steps of the global
    path on every expert/fsdp layout in tests/test_moe_precision.py.
    """
    compute = canonicalize_dtype(x, kernel, dtype=dtype)
    chosen = grouped_matmul_kernel(implementation, compute, (x.dtype, kernel.dtype), precision)
    if chosen == 'pallas' and _local(x) and _local(kernel):
        return grouped_projection(x, kernel, group_sizes, compute, implementation == 'pallas')
    return _projection(x, kernel, group_sizes, dtype,
                       'xla' if chosen == 'pallas' else chosen, precision)


@functools.partial(jax.custom_jvp, nondiff_argnums=(3, 4, 5))
def _projection(x: jax.Array, kernel: jax.Array, group_sizes: jax.Array,
                dtype: Dtype | None, implementation: str,
                precision: PrecisionLike) -> jax.Array:
    """`expert_projection` in every differentiation mode."""
    x, kernel = promote_dtype(x, kernel, dtype=dtype)
    accumulated = grouped_matmul(
        x, kernel, group_sizes, implementation=implementation, precision=precision,
        preferred_element_type=jnp.promote_types(x.dtype, jnp.float32))
    return accumulated.astype(x.dtype)


def _projection_jvp(dtype: Dtype | None, implementation: str, precision: PrecisionLike,
                    primals: tuple[jax.Array, jax.Array, jax.Array],
                    tangents: tuple[jax.Array, jax.Array, jax.Array]
                    ) -> tuple[jax.Array, jax.Array | SymbolicZero]:
    x, kernel, group_sizes = primals
    dx, dkernel, _ = tangents
    output = jnp.asarray(_projection(x, kernel, group_sizes, dtype, implementation, precision))
    work = jnp.result_type(output.dtype, x.dtype, kernel.dtype, jnp.float32)
    # A symbolic zero tangent contributes no term. Only the others pass the
    # barrier: jax 0.11.2 cannot transpose one that holds a constant zero.
    live = [(name, tangent) for name, tangent in (('x', dx), ('kernel', dkernel))
            if not isinstance(tangent, SymbolicZero)]
    if not live:
        return output, SymbolicZero(jax.typeof(output))
    # The tangents keep their dtypes to this point whatever produced them, so
    # an activation's bf16 cotangent is not fused into a wider expression.
    held = dict(zip((name for name, _ in live),
                    jax.lax.optimization_barrier(tuple(tangent for _, tangent in live)),
                    strict=True))
    terms = []
    if 'x' in held:
        matrix = jnp.asarray(rounded_operand(kernel, output.dtype))
        terms.append(jax.lax.ragged_dot(
            held['x'].astype(work), matrix.astype(work), group_sizes, precision=precision,
            preferred_element_type=work))
    if 'kernel' in held:
        inputs = jnp.asarray(rounded_operand(x, output.dtype))
        terms.append(jax.lax.ragged_dot(
            inputs.astype(work), held['kernel'].astype(work), group_sizes, precision=precision,
            preferred_element_type=work))
    total = terms[0] if len(terms) == 1 else terms[0] + terms[1]
    tangent = jax.lax.optimization_barrier(total.astype(output.dtype))
    return output, tangent


_projection.defjvp(_projection_jvp, symbolic_zeros=True)


@jax.custom_jvp
def exact_gelu(x: jax.Array) -> jax.Array:
    """The erf gelu in at least fp32, returned in `x`'s dtype, in both
    differentiation directions: a bf16 gate's activation rounds once, wherever
    the compiler places it."""
    x = jax.lax.optimization_barrier(x)
    work = x.astype(jnp.promote_types(x.dtype, jnp.float32))
    return nn.gelu(work, approximate=False).astype(x.dtype)


@exact_gelu.defjvp
def _exact_gelu_jvp(primals: tuple[jax.Array], tangents: tuple[jax.Array]
                    ) -> tuple[jax.Array, jax.Array]:
    x, = primals
    dx, = tangents
    output = exact_gelu(x)
    work = x.astype(jnp.promote_types(x.dtype, jnp.float32))
    _, derivative = jax.jvp(lambda value: nn.gelu(value, approximate=False),
                            (work,), (jnp.ones_like(work),))
    tangent = derivative * jax.lax.optimization_barrier(dx).astype(work.dtype)
    return output, jax.lax.optimization_barrier(tangent.astype(x.dtype))


@dataclasses.dataclass(frozen=True)
class Situ:
    """Moonshot's SiTU gated product, `SituAndMul` (modeling_kimi_linear.py:64-82
    of moonshotai/Kimi-K3 at f831ab6): `beta * tanh(gate / beta) * sigmoid(gate)`
    times the up projection, which `linear_beta` soft-caps as `linear_beta *
    tanh(up / linear_beta)` when set. Both halves run in fp32 and the product
    returns in the gate's dtype, as the reference computes it. The released
    text config names the two `activation_situ_beta` and
    `activation_situ_linear_beta`. A gated MLP takes one of these as its
    activation in place of a name, since it transforms the up projection too.
    """
    beta: float = 1.0
    linear_beta: float | None = None

    def __call__(self, gate: jax.Array, up: jax.Array) -> jax.Array:
        work_gate, work_up = gate.astype(jnp.float32), up.astype(jnp.float32)
        activated = self.beta * jnp.tanh(work_gate / self.beta) * jax.nn.sigmoid(work_gate)
        if self.linear_beta is not None:
            work_up = self.linear_beta * jnp.tanh(work_up / self.linear_beta)
        return (activated * work_up).astype(gate.dtype)


GatedActivation = str | Situ
"""A gated MLP's activation: 'swiglu', 'geglu' or 'geglu_exact' by name, or a `Situ`."""


def gated_product(activation: GatedActivation) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """The product a gated MLP takes of its gate and up projections, rounded
    where torch rounds `act_fn(gate) * up`: the activation, silu ('swiglu'),
    the tanh gelu ('geglu') or the erf gelu ('geglu_exact', `exact_gelu`),
    runs in fp32 and rounds to the gate's dtype, and its product with up
    rounds again. Every rounding is made in place (`rounded_to`), the
    inputs' included, since a projection's fp32 sum can otherwise reach the
    activation past its own cast: written as `activate(gate) * up` on bf16
    values, jit left the roundings to XLA, and a third of Qwen3-0.6B's
    products differed from transformers' by up to 2 ulp. fp32 and wider
    compute round nowhere, so their product is `activate(gate) * up` as it
    was. A `Situ` computes its own product."""
    if isinstance(activation, Situ):
        return activation
    gates = {'swiglu': nn.silu, 'geglu': functools.partial(nn.gelu, approximate=True), 'geglu_exact': exact_gelu}
    if activation not in gates:
        raise ValueError(f"mlp must be 'swiglu', 'geglu', 'geglu_exact' or a Situ, got {activation!r}")
    activate = gates[activation]

    def product(gate: jax.Array, up: jax.Array) -> jax.Array:
        dtype = jnp.result_type(gate, up)
        if jnp.finfo(dtype).bits >= 32:
            return activate(gate) * up
        gate_fp32 = rounded_to(gate.astype(jnp.float32), gate.dtype)
        up_fp32 = rounded_to(up.astype(jnp.float32), up.dtype)
        activated = rounded_to(activate(gate_fp32), gate.dtype)
        return rounded_to(activated * up_fp32, dtype).astype(dtype)

    return product


class ExpertLinear(nn.Module):
    """One matrix per expert, `[exp, in_features, features]`, over tokens
    already sorted by expert, through `expert_projection` on `implementation`."""
    num_experts: int
    in_features: int
    features: int
    implementation: str = 'auto'
    init_std: float | None = None  # normal std of every expert; None: per-expert lecun normal
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        # With the expert dimension as a batch axis, fan_in is per expert and
        # every expert initialises like the matching nn.Dense of a dense MLP.
        self.kernel = self.param(
            'kernel',
            normal_kernel(self.init_std, nn.initializers.variance_scaling(
                1.0, 'fan_in', 'truncated_normal', in_axis=-2, out_axis=-1,
                batch_axis=(0,)))['kernel_init'],
            (self.num_experts, self.in_features, self.features), jnp.float32)

    def __call__(self, tokens, group_sizes):
        return jnp.asarray(expert_projection(
            tokens, self.kernel, group_sizes, self.dtype, self.implementation, self.precision))


def capacity_positions(indices: jax.Array, num_experts: int,
                       capacity_factor: float) -> tuple[jax.Array, int]:
    """Each routed slot's place in its expert's queue, and how many places a
    queue keeps.

    `indices` is `[..., length, top_k]`: every leading row is one sequence of
    `length` tokens (a rank-two routing is one sequence). A sequence queues
    its slots at each expert in token order; a token's choices are distinct
    experts, so they never compete for one queue. Each queue keeps
    `capacity` places, MaxText's `expert_capacity_per_batch`
    (`maxtext layers/moe.py` `generate_masks`): `max(ceil(length * top_k /
    num_experts) * capacity_factor, capacity_factor)`, truncated.

    Counting within a sequence is what makes the dropped slots a property of
    the batch rather than of its placement: data, expert, fsdp and tensor
    split whole rows, and the count runs over the whole row whatever splits
    its positions.
    """
    *_, length, top_k = indices.shape
    capacity = int(max(math.ceil(length * top_k / num_experts) * capacity_factor,
                       capacity_factor))
    flat = indices.reshape(-1, length * top_k)
    order = jnp.argsort(flat, axis=-1, stable=True)
    queued = jnp.take_along_axis(flat, order, axis=-1)
    counts = jax.vmap(functools.partial(jnp.bincount, length=num_experts))(flat)
    starts = jnp.cumsum(counts, axis=-1) - counts
    places = jnp.arange(length * top_k) - jnp.take_along_axis(starts, queued, axis=-1)
    rows = jnp.arange(flat.shape[0])[:, None]
    positions = jnp.zeros_like(flat).at[rows, order].set(places, unique_indices=True)
    return positions.reshape(indices.shape), capacity


def expert_dispatch[Parameters](
        project: Callable[[jax.Array, jax.Array, jax.Array, Parameters], jax.Array],
        x: jax.Array, indices: jax.Array, parameters: Parameters,
        parameter_axes: Sequence[LogicalAxes], *,
        num_experts: int, dispatch: str, output_dtype: Dtype,
        input_weights: jax.Array | None = None, initializing: bool = False,
        capacity_factor: float | None = None) -> jax.Array:
    """Run every routed token through its expert, and return the slots in
    token order.

    `project` takes rows sorted by expert, the group sizes, the sorted
    expert ids and the expert-major parameters, and returns rows of the same
    width. `parameter_axes` names each parameter's dimensions as its module
    declares them. The caller owns the activation, the biases and the output
    weights; `input_weights` scales each expert's input instead, as Llama 4
    does. `x` is `[batch, length, width]` or `[tokens, width]`.

    `dispatch='global'` sorts and gathers the tokens where they are: on a
    mesh, inside a `shard_map` that holds them where the residual stream
    does (`activation_batch`, `activation_length`), so each device routes
    its own rows through every expert and a layout computes each token once,
    and `project` sees device-local rows, as the Pallas kernels need
    (MaxText's `sparse_matmul_route_and_compute` has the same structure).

    `'exchange'` is expert parallelism, in the same `shard_map` with the
    expert axis manual: each device sorts its own slots by the expert shard
    that owns them and trades buckets of them with its peers in
    `all_to_all` rounds (`_exchange_shard`), keeping every selected slot. A
    device's experts enter it alone, split as the `exp` rule splits them.
    Rows the batch axes do not divide are padded until they do, so every
    shard sends tokens of its own.

    The parameters enter the map in the shards they are stored in and are
    gathered inside it, all but a device's own experts under the exchange,
    so their gradient leaves reduce-scattered onto those shards rather than
    summed whole on every device.

    `capacity_factor` drops instead, as GShard and MaxText do: each
    sequence keeps `capacity_positions`' count of slots per expert, in
    token order, and a dropped slot contributes nothing. Both dispatches
    drop the same slots whatever the placement, and the exchange then needs
    exactly one round, whose buckets hold what a device's sequences can
    keep.

    The parameters take part in at least fp32, so the gradient a device
    computes for them is summed with its peers' and over the rounds in fp32
    and meets the master dtype once, after the sum: the same arithmetic on
    one device and on any mesh.
    """
    if dispatch not in EXPERT_DISPATCHES:
        raise ValueError(f"dispatch must be one of {EXPERT_DISPATCHES}, got {dispatch!r}")
    if indices.shape[:-1] != x.shape[:-1] or (
            input_weights is not None and input_weights.shape != indices.shape):
        raise ValueError(f"routing {indices.shape} does not describe tokens {x.shape}")
    mesh = jax.sharding.get_abstract_mesh()
    shards = mesh.shape.get(EXPERT_AXIS, 1)
    if dispatch == 'exchange' and not initializing and (
            shards <= 1 or num_experts % shards):
        raise ValueError("exchange dispatch needs an expert mesh axis greater than one "
                         "that divides num_experts")
    capacity = None
    if capacity_factor is not None:
        positions, capacity = capacity_positions(indices, num_experts, capacity_factor)
        # A dropped slot takes the sentinel id past the last expert, which
        # no group counts and no bucket sends.
        indices = jnp.where(positions < capacity, indices, num_experts)
    parameters = _widened(parameters)
    exchanging = dispatch == 'exchange' and not initializing and x.size > 0
    if not exchanging and (initializing or mesh.empty):
        return _sorted_locally(project, x, indices, parameters, input_weights,
                               num_experts=num_experts)
    rows = x.shape[0]
    if exchanging:
        # Each expert shard sends tokens of its own: the rows are padded until
        # every axis `activation_batch` takes splits them (`batch_axes`), and
        # the padding routes to the sentinel.
        share = math.prod(mesh.shape[axis] for axis in batch_axes(mesh))
        padding = ((0, -rows % share),)
        x = jnp.pad(x, padding + ((0, 0),) * (x.ndim - 1))
        routed = padding + ((0, 0),) * (indices.ndim - 1)
        indices = jnp.pad(indices, routed, constant_values=num_experts)
        if input_weights is not None:
            input_weights = jnp.pad(input_weights, routed)
    positions_axes = ('activation_batch', 'activation_length')[:x.ndim - 1]
    tokens = logical_spec((*positions_axes, 'activation_embed'), x.shape)
    routing = logical_spec((*positions_axes, None), indices.shape)
    manual = {axis for entry in tokens for axis in mesh_axes(entry)}
    # A pipeline vmaps its stages with spmd_axis_name=stage, and a vmapped
    # shard_map can split the new dimension only over an axis it holds
    # manual; the operands, which name no stage, are replicated over it.
    # A stage axis of one runs no pipeline (`pipeline_stages`) and stays
    # automatic: held manual, it would have the map's transpose psum the
    # tokens' cotangents over it in their own dtype, and XLA's CPU compiler
    # aborts on such a bf16 all-reduce (its AllReducePromotion clones only a
    # binary reducer, and JAX wraps this one's add in a sharding constraint).
    if mesh.shape.get(STAGE_AXIS, 1) > 1 and STAGE_AXIS not in mesh.manual_axes:
        manual.add(STAGE_AXIS)
    stored = []
    for leaf, axes in zip(jax.tree.leaves(parameters), parameter_axes, strict=True):
        kept = [tuple(axis for axis in mesh_axes(entry) if axis in manual)
                for entry in logical_spec(axes, leaf.shape)]
        stored.append(P(*(entry if entry else None for entry in kept)))
    held = jax.tree.unflatten(jax.tree.structure(parameters), stored)
    if exchanging:
        if any(not spec or EXPERT_AXIS not in mesh_axes(spec[0]) for spec in stored):
            raise ValueError("exchange dispatch holds each device's own experts, so every "
                             "expert parameter splits its first dimension over the expert "
                             f"axis; the rule table places them {stored}")
        inner = functools.partial(_exchange_shard, project, num_experts=num_experts,
                                  shards=shards, output_dtype=output_dtype, capacity=capacity)
    else:
        inner = functools.partial(_sorted_locally, project, num_experts=num_experts)

    def body(x, indices, parameters, input_weights):
        def gathered(leaf: jax.Array, spec: P) -> jax.Array:
            for dimension, entry in enumerate(spec):
                axes = tuple(axis for axis in mesh_axes(entry)
                             if not (exchanging and dimension == 0 and axis == EXPERT_AXIS))
                if axes:
                    leaf = jax.lax.all_gather(leaf, axes, axis=dimension, tiled=True)
            return leaf
        return inner(x, indices, jax.tree.map(gathered, parameters, held), input_weights)

    # The kernels carry no manual-axis type, so this map does not check them;
    # MaxText hosts its gmm the same way.
    return jax.shard_map(
        body, mesh=mesh, axis_names=manual,
        in_specs=(tokens, routing, held, None if input_weights is None else routing),
        out_specs=P(*routing, None), check_vma=False)(
            x, indices, parameters, input_weights)[:rows]


def _sorted_locally[Parameters](
        project: Callable[[jax.Array, jax.Array, jax.Array, Parameters], jax.Array],
        x: jax.Array, indices: jax.Array, parameters: Parameters,
        input_weights: jax.Array | None, *, num_experts: int) -> jax.Array:
    """Project every slot of `x`'s tokens where they are: sort the slots by
    expert, gather their token rows and put the results back in slot order.
    A dropped slot's sentinel id sorts last, outside every group, and its
    row comes back zero."""
    top_k = indices.shape[-1]
    tokens = x.reshape(-1, x.shape[-1])
    experts = indices.ravel()
    order = jnp.argsort(experts)
    grouped = tokens[order // top_k]
    if input_weights is not None:
        grouped = grouped * input_weights.ravel()[order][:, None].astype(grouped.dtype)
    projected = project(grouped, jnp.bincount(experts, length=num_experts),
                        experts[order], parameters)
    projected = jnp.where((experts[order] < num_experts)[:, None], projected, 0)
    return projected[jnp.argsort(order)].reshape(*indices.shape, x.shape[-1])


def _exchange_shard[Parameters](
        project: Callable[[jax.Array, jax.Array, jax.Array, Parameters], jax.Array],
        x: jax.Array, indices: jax.Array, parameters: Parameters,
        input_weights: jax.Array | None, *, num_experts: int, shards: int,
        output_dtype: Dtype, capacity: int | None) -> jax.Array:
    """Run one device's share of `expert_dispatch`'s exchange, inside its
    shard_map.

    The device sorts its own slots by expert and sends each destination
    shard one bucket of them in a first round. Dropless, that bucket is the
    device's balanced share, its slots divided by the shards, and what a
    skewed routing leaves over follows in later rounds of the same size:
    the expert axis's maximum count, so every peer runs the same number, out
    of the most a routing of every slot to one shard can need. Under a
    capacity, a sequence keeps `capacity` slots per expert, so the first
    round's bucket holds what the device's sequences (or parts of one,
    where positions are split) can keep for one peer's experts, and no slot
    is left over.

    The first round's intermediates are kept for the backward pass like any
    layer's. The later rounds run in a scan, checkpointed so that it keeps
    none of theirs and recomputes each one it ran; a scan's cond stacks its
    residuals per iteration, so the later rounds are as few as the bucket
    size allows.
    """
    top_k = indices.shape[-1]
    tokens = x.reshape(-1, x.shape[-1])
    slots, per_shard = indices.size, num_experts // shards
    if capacity is None:
        first = -(-slots // shards)
    else:
        first = min(slots, math.prod(indices.shape[:-2]) * per_shard * capacity)
    order = jnp.argsort(indices.ravel())
    experts = indices.ravel()[order]
    sizes = jnp.bincount(experts // per_shard, length=shards + 1)[:shards]
    starts = jnp.cumsum(sizes) - sizes
    rows = order // top_k
    scales = None if input_weights is None else input_weights.ravel()[order]

    def exchange_round(offset: jax.Array | int) -> tuple[jax.Array, jax.Array]:
        """Send each shard its `first` slots from `offset` on, project them
        there, and bring them back.

        The rows are gathered from the tokens themselves, through the
        sort's index. Returns the returned rows and the slot each belongs
        at, in token order; padding's lies past the end, so the caller's
        scatter drops it.
        """
        offsets = offset + jnp.arange(first)
        valid = offsets[None, :] < sizes[:, None]
        addresses = jnp.minimum(starts[:, None] + offsets, slots - 1)
        send = jnp.where(valid[..., None], tokens[rows[addresses]], 0)
        if scales is not None:
            send = send * scales[addresses][..., None].astype(send.dtype)
        ids = jnp.where(valid, experts[addresses] % per_shard, per_shard)
        received = jax.lax.all_to_all(send, EXPERT_AXIS, 0, 0, tiled=True).reshape(
            -1, tokens.shape[-1])
        received_ids = jax.lax.all_to_all(ids, EXPERT_AXIS, 0, 0, tiled=True).ravel()
        permutation = jnp.argsort(received_ids)
        groups = jnp.bincount(received_ids, length=per_shard + 1)[:per_shard]
        computed = project(received[permutation], groups, received_ids[permutation], parameters)[
            jnp.argsort(permutation)]
        computed = jnp.where((received_ids < per_shard)[:, None], computed, 0)
        returned = jax.lax.all_to_all(computed.reshape(shards, first, -1),
                                     EXPERT_AXIS, 0, 0, tiled=True)
        # Every real slot has one writer across all rounds.
        return returned, jnp.where(valid, order[addresses], slots)

    returned, targets = exchange_round(0)
    combined = jnp.zeros((slots, tokens.shape[-1]), output_dtype).at[targets].set(
        returned, mode='drop')
    if capacity is None and slots > first:
        rounds = -(-(slots - first) // first)
        needed = jax.lax.pmax(jnp.max(-(-jnp.maximum(sizes - first, 0) // first)), EXPERT_AXIS)
        later = jax.checkpoint(exchange_round)

        def step(out: jax.Array, iteration: jax.Array) -> tuple[jax.Array, None]:
            """Scan one later round into the output rows; the carry is
            those rows. The scan's length is static, and the rounds past
            `needed` keep the carry unchanged."""
            def active(out: jax.Array) -> jax.Array:
                returned, targets = later((iteration + 1) * first)
                return out.at[targets].set(returned, mode='drop')
            return jax.lax.cond(iteration < needed, active, lambda out: out, out), None

        combined, _ = jax.lax.scan(step, combined, jnp.arange(rounds))
    return combined.reshape(*indices.shape, tokens.shape[-1])


def _widened[Tree](parameters: Tree) -> Tree:
    """Floating parameters at least fp32, so a cotangent summed across
    devices or over the exchange's rounds is summed before its master dtype
    rounds it."""
    def widen(leaf):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            return leaf.astype(jnp.promote_types(leaf.dtype, jnp.float32))
        return leaf
    return jax.tree.map(widen, parameters)


EXPERT_AXES: Mapping[str, LogicalAxes] = {
    'gate_proj': ('exp', 'embed', 'mlp'),
    'up_proj': ('exp', 'embed', 'mlp'),
    'down_proj': ('exp', 'mlp', 'embed'),
}
"""The stacked expert kernels' axes, in the order `ExpertMLP` projects through them."""


class ExpertMLP(nn.Module):
    """The routed experts of one layer: each token through the gated MLPs its
    router chose.

    Tokens are gathered into expert order, the three projections run as grouped
    matmuls over that order, and the results go back to token order and are
    summed with their router weights (`maxtext layers/moe.py:940` `permute` and
    `:1101` `unpermute`). The gather reads token rows directly, without a
    `top_k`-fold copy of them, which is MaxText's `moe_use_direct_token_gather`.

    `dispatch='global'` retains that path. `'exchange'` opts into all-to-all
    rounds on an expert mesh axis that divides the expert count. Each round
    uses a local-slot-sized message buffer and later rounds keep excess
    assignments, so no slot is dropped unless `capacity_factor` asks for it
    (`expert_dispatch`). Initialisation creates the same parameters without
    requiring a mesh. The projections run through `expert_projection`, whose
    precision contract is the same under both dispatches, so a routed layer
    computes the same forward, gradients and tangents whichever moves the
    tokens.

    The sum over a token's experts runs in fp32, because that is the dtype the
    router weights are computed in, and the result rejoins the residual stream
    in the compute dtype.

    `swiglu_limit` is DeepSeek V4's clamp before the activation
    (`modeling_deepseek_v4.py`, `DeepseekV4Experts._apply_gate`): the gate is
    capped at the limit from above and the up projection on both sides, which
    bounds what one expert can add. None is the plain gated MLP.

    `scale_inputs` is Llama 4's placement of the routing weight: each
    token's input to an expert is multiplied by its weight and the expert
    outputs are summed unweighted (`modeling_llama4.py`,
    `Llama4TextMoe.forward`, `routed_in * router_scores`), which is not the
    weighted sum of outputs because the gate is not linear.
    """
    num_experts: int
    hidden_features: int
    out_features: int
    activation: GatedActivation = 'swiglu'
    implementation: str = 'auto'
    dispatch: str = 'global'
    capacity_factor: float | None = None
    swiglu_limit: float | None = None
    scale_inputs: bool = False
    init_std: float | None = None  # gate/up normal std; None: per-expert lecun normal
    output_init_std: float | None = None  # down normal std; None follows init_std
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        if self.swiglu_limit is not None and self.swiglu_limit <= 0:
            raise ValueError(
                f"swiglu_limit caps the gate and up projections, so it is "
                f"positive, got {self.swiglu_limit}; None leaves them unclamped")
        expert = functools.partial(
            ExpertLinear, num_experts=self.num_experts,
            implementation=self.implementation, dtype=self.dtype,
            precision=self.precision)
        self.gate_proj = expert(in_features=self.out_features,
                                features=self.hidden_features, init_std=self.init_std,
                                name='gate_proj')
        self.up_proj = expert(in_features=self.out_features,
                              features=self.hidden_features, init_std=self.init_std,
                              name='up_proj')
        self.down_proj = expert(in_features=self.hidden_features,
                                features=self.out_features, name='down_proj',
                                init_std=(self.init_std if self.output_init_std is None
                                          else self.output_init_std))

    def _project(self, tokens: jax.Array, sizes: jax.Array, _expert_ids: jax.Array,
                 kernels: tuple[jax.Array, jax.Array, jax.Array], *, dtype: Dtype) -> jax.Array:
        def linear(x: jax.Array, kernel: jax.Array) -> jax.Array:
            return jnp.asarray(expert_projection(
                x, kernel, sizes, dtype, self.implementation, self.precision))

        # The same residual names as the dense MLP's, so one remat policy
        # covers both (causal_transformer.RESIDUALS).
        gate = checkpoint_name(linear(tokens, kernels[0]), 'gate_proj')
        up = checkpoint_name(linear(tokens, kernels[1]), 'up_proj')
        if self.swiglu_limit is not None:
            gate = jnp.minimum(gate, self.swiglu_limit)
            up = jnp.clip(up, -self.swiglu_limit, self.swiglu_limit)
        return checkpoint_name(linear(gated_product(self.activation)(gate, up), kernels[2]), 'down_proj')

    def _combine(self, slots: jax.Array, weights: jax.Array) -> jax.Array:
        if self.scale_inputs:
            return jnp.sum(slots.astype(jnp.float32), axis=-2).astype(slots.dtype)
        return jnp.einsum('...ke,...k->...e', slots.astype(jnp.float32),
                          weights.astype(jnp.float32), precision=self.precision).astype(slots.dtype)

    def __call__(self, x: jax.Array, weights: jax.Array, indices: jax.Array) -> jax.Array:
        if weights.shape != indices.shape:
            raise ValueError(f"routing {indices.shape} does not describe weights {weights.shape}")
        kernels = (self.gate_proj.kernel, self.up_proj.kernel, self.down_proj.kernel)
        # One compute dtype for all three projections, named rather than
        # inferred inside the dispatch, where the parameters may be widened.
        compute = canonicalize_dtype(x, *kernels, dtype=self.dtype)
        slots = expert_dispatch(
            functools.partial(self._project, dtype=compute), x, indices, kernels,
            tuple(EXPERT_AXES.values()), num_experts=self.num_experts, dispatch=self.dispatch,
            initializing=self.is_initializing(), output_dtype=compute,
            input_weights=weights if self.scale_inputs else None,
            capacity_factor=self.capacity_factor)
        return self._combine(slots, weights)


@logical_axes({
    ("gate",): ("embed", "exp"),
    # A sparse layer's experts are stacked on one leaf, so the expert
    # dimension is named here and the longer path wins over the dense
    # projection of the same name.
    **{("experts", name): axes for name, axes in EXPERT_AXES.items()},
    # The shared branch is one dense gated MLP beside the experts, sharded
    # like the dense layers' own.
    ("shared_experts", "gate_proj"): ("embed", "mlp"),
    ("shared_experts", "up_proj"): ("embed", "mlp"),
    ("shared_experts", "down_proj"): ("mlp", "embed"),
    ("shared_expert_gate",): ("embed", None),
    # Kimi K3's latent projections around the routed experts: the experts
    # themselves run at the latent width under the expert names above.
    ("routed_expert_down_proj",): ("embed", None),
    ("routed_expert_up_proj",): (None, "embed"),
})
class SparseMLP(nn.Module):
    """A router over `num_experts` gated MLPs, `top_k` of them per token.

    Goes where `GatedMLP` goes in a decoder block and holds the submodules a
    Hugging Face sparse layer names: `gate` for the router, `experts` for the
    stacked expert weights and, when `shared` is set, `shared_experts` for
    the dense branch every token takes. `shared` is a factory taking only a
    name, the shape of `DecoderBlock`'s slots, so the backbone hands in its
    own `GatedMLP` at the shared width and the sum is what `DeepseekV3MoE`
    computes: the routed output plus the shared branch of the same input.
    With shared_gate, a learned scalar sigmoid independently weights the
    shared branch, as Qwen3_5MoeSparseMoeBlock does.

    `latent_features` is Kimi K3's latent MoE (`KimiSparseMoeBlock`,
    modeling_kimi_linear.py:762-838 of moonshotai/Kimi-K3 at f831ab6): the
    router reads the full-width input, `routed_expert_down_proj` narrows it
    to the latent width the experts run at, and the weighted sum of their
    outputs goes through `latent_norm` (a factory taking a name, the
    backbone's RMSNorm; None for none) and `routed_expert_up_proj` back to
    `out_features`. The shared branch reads the full-width input. None is
    every other mixture, whose experts run at `out_features`.
    """
    num_experts: int
    top_k: int
    hidden_features: int
    out_features: int
    activation: GatedActivation = 'swiglu'
    implementation: str = 'auto'
    dispatch: str = 'global'
    capacity_factor: float | None = None
    score_function: str = 'softmax'
    normalize_weights: bool = True
    routed_scaling_factor: float = 1.0
    expert_groups: int = 1
    groups_per_token: int = 1
    group_score: str = 'top2'
    expert_bias: bool = False
    media_bias: bool = False
    hash_vocab: int | None = None
    swiglu_limit: float | None = None
    scale_inputs: bool = False
    shared: Callable[..., nn.Module] | None = None
    shared_gate: bool = False
    init_std: float | None = None
    """Normal std of the router and the experts' gate and up kernels, one std
    as lm-engine's MoE draws them (`up_std`); None keeps the modules' own."""
    output_init_std: float | None = None
    """Normal std of the experts' down kernels; None follows init_std."""
    latent_features: int | None = None
    latent_norm: Callable[..., nn.Module] | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        self.gate = Router(num_experts=self.num_experts,
                           in_features=self.out_features, top_k=self.top_k,
                           score_function=self.score_function,
                           normalize_weights=self.normalize_weights,
                           routed_scaling_factor=self.routed_scaling_factor,
                           expert_groups=self.expert_groups,
                           groups_per_token=self.groups_per_token,
                           group_score=self.group_score,
                           expert_bias=self.expert_bias,
                           media_bias=self.media_bias,
                           hash_vocab=self.hash_vocab, init_std=self.init_std,
                           precision=self.precision, name='gate')
        width = self.out_features if self.latent_features is None else self.latent_features
        self.experts = ExpertMLP(
            num_experts=self.num_experts, hidden_features=self.hidden_features,
            out_features=width, activation=self.activation,
            implementation=self.implementation, dispatch=self.dispatch,
            capacity_factor=self.capacity_factor,
            swiglu_limit=self.swiglu_limit,
            scale_inputs=self.scale_inputs,
            init_std=self.init_std, output_init_std=self.output_init_std,
            dtype=self.dtype, precision=self.precision, name='experts')
        if self.latent_features is not None:
            dense = functools.partial(nn.Dense, use_bias=False, dtype=self.dtype, precision=self.precision)
            self.routed_expert_down_proj = dense(self.latent_features, name='routed_expert_down_proj')
            self.routed_expert_up_proj = dense(self.out_features, name='routed_expert_up_proj')
            if self.latent_norm is not None:
                self.routed_expert_norm = self.latent_norm(name='routed_expert_norm')
        if self.shared_gate and self.shared is None:
            raise ValueError("shared_gate requires a shared expert")
        if self.shared is not None:
            self.shared_experts = self.shared(name='shared_experts')
            if self.shared_gate:
                self.shared_expert_gate = nn.Dense(
                    1, use_bias=False, dtype=self.dtype, precision=self.precision,
                    name='shared_expert_gate', **normal_kernel(self.init_std))

    def __call__(self, x, tokens=None, media=None, routes: Routes | None = None):
        weights, indices = self.gate(x, tokens, media, routes)
        if self.latent_features is None:
            routed = self.experts(x, weights, indices)
        else:
            routed = self.experts(self.routed_expert_down_proj(x), weights, indices)
            if self.latent_norm is not None:
                routed = self.routed_expert_norm(routed)
            routed = self.routed_expert_up_proj(routed)
        if self.shared is None:
            return routed
        shared = self.shared_experts(x)
        if self.shared_gate:
            shared = shared * jax.nn.sigmoid(self.shared_expert_gate(x))
        return routed + shared
