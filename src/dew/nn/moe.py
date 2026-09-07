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

import functools
import importlib
from collections.abc import Callable
from typing import Optional

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from flax import linen as nn, struct
from flax.linen.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike

from jax.sharding import PartitionSpec as P

from .sharding import EXPERT_AXIS, logical_axes

# 'softmax' normalizes a token's affinities over the experts (Mixtral,
# Qwen3.5); 'sigmoid' scores each expert on its own (DeepSeek V3, GLM, Kimi,
# LLaDA2); 'sqrtsoftplus' is DeepSeek V4's sqrt(softplus(logit))
# (transformers activations.py, SqrtSoftplusActivation), unbounded above
# where a sigmoid saturates.
SCORE_FUNCTIONS = ('softmax', 'sigmoid', 'sqrtsoftplus')

GROUPED_MATMULS = ('xla', 'tokamax')
EXPERT_DISPATCHES = ('global', 'exchange')

# DeepSeek divides the selected weights by their sum plus this, so a token
# whose sigmoid scores are all zero stays finite
# (modeling_deepseek_v3.py:166, maxtext layers/moe.py:839). A softmax over the
# experts sums to one, which cannot reach it, and in fp32 the term changes no
# bit of a denominator above 1e-12.
WEIGHT_SUM_EPSILON = 1e-20


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
    """One DeepSeek V2 load-times-score loss per intact sequence."""
    _, length, experts = scores.shape
    scores = scores.astype(jnp.promote_types(scores.dtype, jnp.float32))
    chosen = jax.nn.one_hot(indices, experts, dtype=scores.dtype)
    load = jnp.sum(chosen, axis=(1, 2)) / (length * indices.shape[-1] / experts)
    return jnp.sum(load * jnp.mean(scores, axis=1), axis=1) * alpha


def deepseek_v2_aux_loss(scores, indices, alpha: float, seq_aux: bool = True):
    """DeepSeek V2 expert balance (arXiv 2405.04434, section 2.1.4).

    Scores are [batch, sequence, experts], choices [batch, sequence, top_k].
    seq_aux forms the product within each sequence before averaging rows.
    The global variant pools all routed positions before forming the product.
    """
    if seq_aux:
        return jnp.mean(sequence_router_losses(scores, indices, alpha))
    return global_router_loss(router_moments(scores, indices), alpha)


def calculate_load_balance_updates(top_k_indices, num_experts, rate):
    """
    Computes a bias adjustment update based on expert load.
    Used in DeepSeek V3: https://arxiv.org/html/2412.19437v1.
    Implementation reference: https://arxiv.org/pdf/2408.15664.

    Args:
        top_k_indices: Shape (batch, sequence, top_k).
        num_experts: Total number of experts.
        rate: The update rate.

    Returns:
        update: The value to add to the expert bias. Shape (num_experts,).
    """
    flat_indices = top_k_indices.ravel()
    expert_counts = jnp.sum(jax.nn.one_hot(flat_indices, num_experts, dtype=jnp.int32), axis=0)
    return load_balance_update(expert_counts, rate)


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
    """Which experts a token goes to, and with what weight: `[..., k]` of each.

    The gate projection runs in fp32 whatever dtype the activations carry,
    as DeepSeek's router does (`modeling_deepseek_v3.py:146`).

    `expert_bias` is DeepSeek's aux-loss-free balancing bias
    (`e_score_correction_bias`, arXiv 2408.15664), kept in fp32 in the `moe`
    collection. It enters the selection only: a token's weights are gathered
    from the unbiased scores, so moving the bias changes which experts a token
    uses without changing what they contribute. Nothing here writes it;
    transformers holds it in an `nn.Buffer` and MaxText hands the update back
    to its caller (`layers/moe.py:965-972`). `calculate_load_balance_updates`
    is that update, and the step that applies it owns the write. Gradients
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
            'kernel', nn.initializers.lecun_normal(),
            (self.in_features, self.num_experts), jnp.float32)
        if self.expert_bias:
            self.bias = self.variable(
                'moe', 'e_score_correction_bias', jnp.zeros,
                (self.num_experts,), jnp.float32)

    def __call__(self, x):
        scores = self.scores(x)
        selection = scores if not self.expert_bias else scores + self.bias.value
        if self.expert_groups > 1:
            selection = jnp.where(self.group_mask(selection), selection, -jnp.inf)
        _, indices = jax.lax.top_k(selection, self.top_k)
        # The load each expert took, for the step that balances the bias:
        # written only when a caller opens the 'router' collection, and never
        # into the tree init returns, where it is not a variable.
        if not self.is_initializing():
            self.sow('router', 'indices', indices)
            # The V2 balance loss reads the scores beside the choices.
            self.sow('router', 'scores', scores)
        weights = jnp.take_along_axis(scores, indices, axis=-1)
        if self.normalize_weights:
            weights = weights / (jnp.sum(weights, axis=-1, keepdims=True)
                                 + WEIGHT_SUM_EPSILON)
        return weights * self.routed_scaling_factor, indices

    def scores(self, x):
        """Each token's fp32 affinity for every expert: `[..., num_experts]`."""
        logits = jnp.einsum('...d,de->...e', x.astype(jnp.float32), self.kernel,
                            precision=self.precision)
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


def grouped_matmul(tokens: jax.Array, kernel: jax.Array, group_sizes: jax.Array, *,
                   implementation: str, precision: PrecisionLike = None,
                   preferred_element_type: Optional[Dtype] = None) -> jax.Array:
    """Each row of `tokens`, `[rows, in]`, through the matrix of its expert in
    `kernel`, `[exp, in, out]`, for rows already sorted by expert.

    `group_sizes` is how many leading rows belong to expert 0, then to expert
    1, and so on, the form both grouped matmuls take. `implementation` picks
    between them, the seam `dew.nn.attention.scaled_dot_product_attention`
    has:

    - 'xla': `jax.lax.ragged_dot`, which lowers on every backend.
    - 'tokamax': `tokamax.ragged_dot`, the same call against tokamax's own
      kernels (`maxtext layers/moe.py:1633`); tokamax picks its Mosaic or
      Triton kernel where one exists and lowers to XLA elsewhere.

    This is the raw kernel call and JAX's own differentiation rules.
    `expert_projection` adds the precision contract routed experts train
    under.
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
            preferred_element_type=preferred_element_type)
    return jax.lax.ragged_dot(
        tokens, kernel, group_sizes, precision=precision,
        preferred_element_type=preferred_element_type)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def _rounded_operand(x: jax.Array, dtype: Dtype) -> jax.Array:
    # The value an operand takes in the compute dtype, held in its own dtype
    # so the straight-through tangent below never rounds a second time.
    return jax.lax.optimization_barrier(x.astype(dtype)).astype(x.dtype)


@_rounded_operand.defjvp
def _rounded_operand_jvp(dtype: Dtype, primals: tuple[jax.Array],
                         tangents: tuple[jax.Array]) -> tuple[jax.Array, jax.Array]:
    return jnp.asarray(_rounded_operand(primals[0], dtype)), tangents[0]


@functools.partial(jax.custom_jvp, nondiff_argnums=(3, 4, 5))
def expert_projection(x: jax.Array, kernel: jax.Array, group_sizes: jax.Array,
                      dtype: Optional[Dtype], implementation: str,
                      precision: PrecisionLike) -> jax.Array:
    """`grouped_matmul` under one precision contract for both dispatches.

    The operands are cast to `dtype` (flax's promotion when None), every
    contraction accumulates in at least fp32 and rounds once to the compute
    dtype, so a width split over a mesh axis rounds no partial sum. The
    tangent is `dx @ Q(kernel) + Q(x) @ dkernel` with `Q` the rounded operand
    values and the tangents in their own dtypes: a kernel gradient keeps the
    master dtype, an input gradient its input's, and both accumulate over
    the widest of the compute, input and kernel dtypes. The contract holds
    in both differentiation directions and does not depend on placement.

    The tangent contractions are `jax.lax.ragged_dot`, whose transposes JAX
    defines, so both directions differentiate under either kernel; tokamax's
    own rules stop at reverse mode. Measured against NumPy float64 sums of
    the rounded operands and three Adam steps of the global path on every
    expert/fsdp layout in tests/test_moe_precision.py; with fp32 operands
    the forward pass is the call it wraps.
    """
    x, kernel = promote_dtype(x, kernel, dtype=dtype)
    accumulated = grouped_matmul(
        x, kernel, group_sizes, implementation=implementation, precision=precision,
        preferred_element_type=jnp.promote_types(x.dtype, jnp.float32))
    return accumulated.astype(x.dtype)


@expert_projection.defjvp
def _expert_projection_jvp(dtype: Optional[Dtype], implementation: str,
                           precision: PrecisionLike,
                           primals: tuple[jax.Array, jax.Array, jax.Array],
                           tangents: tuple[jax.Array, jax.Array, jax.Array]
                           ) -> tuple[jax.Array, jax.Array]:
    x, kernel, group_sizes = primals
    dx, dkernel, _ = tangents
    # The tangents keep their dtypes to this point whatever produced them, so
    # an activation's bf16 cotangent is not fused into a wider expression.
    dx, dkernel = jax.lax.optimization_barrier((dx, dkernel))
    output = jnp.asarray(expert_projection(x, kernel, group_sizes, dtype, implementation, precision))
    work = jnp.result_type(output.dtype, x.dtype, kernel.dtype, jnp.float32)
    inputs = jnp.asarray(_rounded_operand(x, output.dtype))
    matrix = jnp.asarray(_rounded_operand(kernel, output.dtype))
    input_term = jax.lax.ragged_dot(
        dx.astype(work), matrix.astype(work), group_sizes, precision=precision,
        preferred_element_type=work)
    kernel_term = jax.lax.ragged_dot(
        inputs.astype(work), dkernel.astype(work), group_sizes, precision=precision,
        preferred_element_type=work)
    tangent = jax.lax.optimization_barrier((input_term + kernel_term).astype(output.dtype))
    return output, tangent


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


class ExpertLinear(nn.Module):
    """One matrix per expert, `[exp, in_features, features]`, over tokens
    already sorted by expert, through `expert_projection` on `implementation`."""
    num_experts: int
    in_features: int
    features: int
    implementation: str = 'xla'
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        # With the expert dimension as a batch axis, fan_in is per expert and
        # every expert initialises like the matching nn.Dense of a dense MLP.
        self.kernel = self.param(
            'kernel',
            nn.initializers.variance_scaling(
                1.0, 'fan_in', 'truncated_normal', in_axis=-2, out_axis=-1,
                batch_axis=(0,)),
            (self.num_experts, self.in_features, self.features), jnp.float32)

    def __call__(self, tokens, group_sizes):
        return jnp.asarray(expert_projection(
            tokens, self.kernel, group_sizes, self.dtype, self.implementation, self.precision))


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
    assignments. No expert capacity drops tokens. Initialisation creates the
    same parameters without requiring a mesh. The projections run through
    `expert_projection`, whose precision contract is the same under both
    dispatches, so a routed layer computes the same forward, gradients and
    tangents whichever moves the tokens.

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
    activation: str = 'swiglu'
    implementation: str = 'xla'
    dispatch: str = 'global'
    swiglu_limit: Optional[float] = None
    scale_inputs: bool = False
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        if self.dispatch not in EXPERT_DISPATCHES:
            raise ValueError(f"dispatch must be one of {EXPERT_DISPATCHES}, got {self.dispatch!r}")
        if self.activation not in ('swiglu', 'geglu', 'geglu_exact'):
            raise ValueError(
                f"mlp must be 'swiglu', 'geglu' or 'geglu_exact', got {self.activation!r}")
        if self.swiglu_limit is not None and self.swiglu_limit <= 0:
            raise ValueError(
                f"swiglu_limit caps the gate and up projections, so it is "
                f"positive, got {self.swiglu_limit}; None leaves them unclamped")
        expert = functools.partial(
            ExpertLinear, num_experts=self.num_experts,
            implementation=self.implementation, dtype=self.dtype,
            precision=self.precision)
        self.gate_proj = expert(in_features=self.out_features,
                                features=self.hidden_features, name='gate_proj')
        self.up_proj = expert(in_features=self.out_features,
                              features=self.hidden_features, name='up_proj')
        self.down_proj = expert(in_features=self.hidden_features,
                                features=self.out_features, name='down_proj')

    def _project(self, tokens: jax.Array, sizes: jax.Array,
                 kernels: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        def linear(x: jax.Array, kernel: jax.Array) -> jax.Array:
            return jnp.asarray(expert_projection(
                x, kernel, sizes, self.dtype, self.implementation, self.precision))

        # The same residual names as the dense MLP's, so one remat policy
        # covers both (causal_transformer.RESIDUALS).
        gate = checkpoint_name(linear(tokens, kernels[0]), 'gate_proj')
        up = checkpoint_name(linear(tokens, kernels[1]), 'up_proj')
        if self.swiglu_limit is not None:
            gate = jnp.minimum(gate, self.swiglu_limit)
            up = jnp.clip(up, -self.swiglu_limit, self.swiglu_limit)
        if self.activation == 'swiglu':
            gate = nn.silu(gate)
        elif self.activation == 'geglu':
            gate = nn.gelu(gate, approximate=True)
        else:
            gate = exact_gelu(gate)
        return checkpoint_name(linear(gate * up, kernels[2]), 'down_proj')

    def _combine(self, slots: jax.Array, weights: jax.Array) -> jax.Array:
        if self.scale_inputs:
            return jnp.sum(slots.astype(jnp.float32), axis=-2).astype(slots.dtype)
        return jnp.einsum('...ke,...k->...e', slots.astype(jnp.float32),
                          weights.astype(jnp.float32), precision=self.precision).astype(slots.dtype)

    def __call__(self, x: jax.Array, weights: jax.Array, indices: jax.Array) -> jax.Array:
        """`x` is `[..., embed]` and `weights` and `indices` are `[..., k]`."""
        if weights.shape != indices.shape or indices.shape[:-1] != x.shape[:-1]:
            raise ValueError(f"routing {indices.shape} does not describe tokens {x.shape}")
        kernels = (self.gate_proj.kernel, self.up_proj.kernel, self.down_proj.kernel)
        tokens = x.reshape(-1, x.shape[-1])
        mesh = jax.sharding.get_abstract_mesh()
        shards = mesh.shape.get(EXPERT_AXIS, 1)
        if self.dispatch == 'exchange' and not self.is_initializing() and (
                shards <= 1 or self.num_experts % shards):
            raise ValueError("exchange dispatch needs an expert mesh axis greater than one "
                             "that divides num_experts")
        if self.dispatch == 'exchange' and not self.is_initializing() and tokens.shape[0]:
            # Padding is only for a token axis the mesh cannot divide. The
            # sentinel expert is excluded from send counts, never dispatched.
            padding = -tokens.shape[0] % shards
            result = self._exchange(
                jnp.pad(tokens, ((0, padding), (0, 0))),
                jnp.pad(weights.reshape(-1, indices.shape[-1]), ((0, padding), (0, 0))),
                jnp.pad(indices.reshape(-1, indices.shape[-1]), ((0, padding), (0, 0)),
                        constant_values=self.num_experts), kernels)
            return result[:tokens.shape[0]].reshape(x.shape)

        experts = indices.ravel()
        order = jnp.argsort(experts)
        sizes = jnp.bincount(experts, length=self.num_experts)
        grouped = tokens[order // indices.shape[-1]]
        if self.scale_inputs:
            grouped = grouped * weights.ravel()[order][:, None].astype(grouped.dtype)
        projected = self._project(grouped, sizes, kernels)
        return self._combine(projected[jnp.argsort(order)].reshape(
            *indices.shape, self.out_features), weights)

    def _exchange(self, tokens: jax.Array, weights: jax.Array, indices: jax.Array,
                  kernels: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        """Stream destination buckets through fixed-size all-to-alls.

        With S expert shards and L local routing slots, a round carries
        S * ceil(L/S) rows. At most S rounds drain even a bucket holding all
        L slots. Bucket sizes determine the actual round count collectively;
        capacity bounds each message, never the number of accepted tokens.
        The return exchange restores slots before the original top-k sum.
        """
        mesh = jax.sharding.get_abstract_mesh()
        shards = mesh.shape[EXPERT_AXIS]

        @functools.partial(jax.shard_map, mesh=mesh, axis_names={EXPERT_AXIS},
                           in_specs=(P(EXPERT_AXIS), P(EXPERT_AXIS), P(EXPERT_AXIS),
                                     (P(EXPERT_AXIS),) * 3), out_specs=P(EXPERT_AXIS))
        def local(tokens: jax.Array, weights: jax.Array, indices: jax.Array,
                  kernels: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
            slots, per_shard = indices.size, kernels[0].shape[0]
            order = jnp.argsort(indices.ravel())
            experts = indices.ravel()[order]
            grouped = tokens[order // indices.shape[-1]]
            if self.scale_inputs:
                grouped = grouped * weights.ravel()[order][:, None].astype(grouped.dtype)
            sizes = jnp.bincount(experts // per_shard, length=shards + 1)[:shards]
            starts = jnp.cumsum(sizes) - sizes
            capacity = (slots + shards - 1) // shards
            rounds = jax.lax.pmax(jnp.max((sizes + capacity - 1) // capacity), EXPERT_AXIS)
            lanes = jnp.arange(capacity)
            dtype = jnp.dtype(self.dtype) if self.dtype is not None else jnp.result_type(tokens, *kernels)

            @jax.checkpoint
            def exchange_round(iteration: jax.Array) -> tuple[jax.Array, jax.Array]:
                offsets = iteration * capacity + lanes
                valid = offsets[None, :] < sizes[:, None]
                addresses = starts[:, None] + offsets
                send = grouped[jnp.minimum(addresses, slots - 1)]
                send = jnp.where(valid[..., None], send, 0)
                ids = jnp.where(valid, experts[jnp.minimum(addresses, slots - 1)] % per_shard,
                                per_shard)
                received = jax.lax.all_to_all(send, EXPERT_AXIS, 0, 0, tiled=True).reshape(
                    -1, tokens.shape[-1])
                received_ids = jax.lax.all_to_all(ids, EXPERT_AXIS, 0, 0, tiled=True).ravel()
                permutation = jnp.argsort(received_ids)
                groups = jnp.bincount(received_ids, length=per_shard + 1)[:per_shard]
                computed = self._project(received[permutation], groups, kernels)[
                    jnp.argsort(permutation)]
                computed = jnp.where((received_ids < per_shard)[:, None], computed, 0)
                returned = jax.lax.all_to_all(computed.reshape(shards, capacity, -1),
                                             EXPERT_AXIS, 0, 0, tiled=True)
                # Only padding gets an out-of-bounds scatter address. All
                # real slots have one unique writer across the rounds.
                return returned, jnp.where(valid, addresses, slots)

            def step(out: jax.Array, iteration: jax.Array) -> tuple[jax.Array, None]:
                def active(out: jax.Array) -> jax.Array:
                    returned, addresses = exchange_round(iteration)
                    return out.at[addresses].set(returned, mode='drop')

                return jax.lax.cond(iteration < rounds, active, lambda out: out, out), None

            initial = jax.lax.pcast(jnp.zeros((slots, self.out_features), dtype),
                                    EXPERT_AXIS, to='varying')
            result, _ = jax.lax.scan(step, initial, jnp.arange(shards))
            return self._combine(result[jnp.argsort(order)].reshape(
                *indices.shape, self.out_features), weights)

        return local(tokens, weights, indices, kernels)


@logical_axes({
    ("gate",): ("embed", "exp"),
    # A sparse layer's experts are stacked on one leaf, so the expert
    # dimension is named here and the longer path wins over the dense
    # projection of the same name.
    ("experts", "gate_proj"): ("exp", "embed", "mlp"),
    ("experts", "up_proj"): ("exp", "embed", "mlp"),
    ("experts", "down_proj"): ("exp", "mlp", "embed"),
    # The shared branch is one dense gated MLP beside the experts, sharded
    # like the dense layers' own.
    ("shared_experts", "gate_proj"): ("embed", "mlp"),
    ("shared_experts", "up_proj"): ("embed", "mlp"),
    ("shared_experts", "down_proj"): ("mlp", "embed"),
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
    """
    num_experts: int
    top_k: int
    hidden_features: int
    out_features: int
    activation: str = 'swiglu'
    implementation: str = 'xla'
    dispatch: str = 'global'
    score_function: str = 'softmax'
    normalize_weights: bool = True
    routed_scaling_factor: float = 1.0
    expert_groups: int = 1
    groups_per_token: int = 1
    group_score: str = 'top2'
    expert_bias: bool = False
    swiglu_limit: Optional[float] = None
    scale_inputs: bool = False
    shared: Optional[Callable[..., nn.Module]] = None
    dtype: Optional[Dtype] = None
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
                           precision=self.precision, name='gate')
        self.experts = ExpertMLP(
            num_experts=self.num_experts, hidden_features=self.hidden_features,
            out_features=self.out_features, activation=self.activation,
            implementation=self.implementation, dispatch=self.dispatch,
            swiglu_limit=self.swiglu_limit,
            scale_inputs=self.scale_inputs,
            dtype=self.dtype, precision=self.precision, name='experts')
        if self.shared is not None:
            self.shared_experts = self.shared(name='shared_experts')

    def __call__(self, x):
        weights, indices = self.gate(x)
        routed = self.experts(x, weights, indices)
        if self.shared is None:
            return routed
        return routed + self.shared_experts(x)
