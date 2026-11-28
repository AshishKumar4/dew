"""Mixture of experts: the router, and the experts as one grouped matmul.

Patterned on MaxText's `RoutedMoE` (`maxtext src/maxtext/layers/moe.py:419`,
Apache 2.0): top-k selection (`:751` `get_topk`), DeepSeek's group-limited
routing, which selects on the biased scores and gates on the unbiased ones
(`:881-908`), its weight scaling (`:835-841`), the aux-loss-free bias update
(`:238-261`) and the grouped matmul over tokens sorted by expert (`:1500`).

The routers reproduced are transformers 5.16.1's `MixtralSparseMoeBlock`
(softmax, top k, renormalise), `DeepseekV3MoE` (sigmoid, select with a bias
inside the best expert groups, renormalise, scale, add a shared expert) and
DeepSeek V4's sqrt(softplus) score and swiglu limit. `tests/test_moe.py`
holds the fp32 parity numbers against `tools/moe_reference.py`'s fixtures.

Parameters follow the Hugging Face sparse layer with the experts stacked:
`mlp/gate/kernel` `[embed, exp]`, `mlp/experts/{gate,up}_proj/kernel` `[exp,
embed, mlp]` and `mlp/experts/down_proj/kernel` `[exp, mlp, embed]`, the
layout `jax.lax.ragged_dot` takes and the `expert` mesh axis shards.
"""

import contextlib
import contextvars
import dataclasses
import functools
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn, struct
from flax.linen.dtypes import canonicalize_dtype
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name
from jax.custom_derivatives import SymbolicZero
from jax.sharding import PartitionSpec as P

from dew import quantization
from dew.batch import Variables, leaves_by_path
from dew.mesh import (
    EXPERT_AXIS,
    STAGE_AXIS,
    LayoutRefused,
    LogicalAxes,
    batch_axes,
    kept_axes,
    logical_spec,
    mesh_axes,
    spec_axes,
)
from dew.nn import categorical
from dew.nn.dense import Dense, amax_history, noted_amax, packed_matrix
from dew.nn.derivatives import KernelGlue, Protected, contracted, declared_tangent, with_tangents_formed_at
from dew.nn.kernels.templates.gemm import launched_whole, rows_gathered_wide
from dew.nn.kernels.templates.hooks import Hook, Side
from dew.nn.residuals import Residual

from .activations import (
    GatedActivation,
    Situ,
    gated_hook,
    gated_product,
    logistic,
    relu2,
    rounded_gated,
    swiglu_clamped,
)
from .blocks import normal_kernel
from .flops import MissingFlopsCount, counted_flops, projection_flops
from .gated_mlp import fuses_grouped
from .kernels import formats, gemm, generation, projections
from .precision import (
    at_least_fp32,
    capped_logits,
    compensated_sum,
    divided,
    exact_float32,
    float32_dot_general,
    float32_einsum,
    projection_role,
    recipe_role,
    rounded_operand,
    softplus,
)
from .protocols import ProjectionGroup, Routes, declared_groups
from .sharding import declared

# 'softmax' normalizes a token's affinities over the experts (Mixtral,
# Qwen3.5); 'sigmoid' scores each expert on its own (DeepSeek V3, GLM, Kimi,
# LLaDA2); 'sqrtsoftplus' is DeepSeek V4's sqrt(softplus(logit))
# (transformers activations.py, SqrtSoftplusActivation), unbounded above
# where a sigmoid saturates.
SCORE_FUNCTIONS = ('softmax', 'sigmoid', 'sqrtsoftplus')

# 'pallas' is the kernels, each launch running the first implementation that takes it
# (`grouped_matmul.GROUPED`) or the one `generation.chosen(grouped_matmul=...)` names ('triton',
# 'mosaic', 'cutlass'); 'auto' is the kernels where they run the call, else XLA
# (`expert_projection`, through the GEMM template).
GROUPED_MATMULS = ('auto', 'xla', 'pallas')
EXPERT_DISPATCHES = ('auto', 'global', 'exchange')


def exchanges(num_experts: int, shards: int) -> bool:
    """Whether `num_experts` experts over `shards` expert shards exchange their tokens: an expert axis
    greater than one that divides them, each shard holding whole experts."""
    return shards > 1 and not num_experts % shards


def resolved_dispatch(dispatch: str, num_experts: int, shards: int) -> str:
    """The dispatch `dispatch` names for `num_experts` experts over `shards` expert shards: 'auto' the
    exchange where they exchange (`exchanges`), the global dispatch anywhere else, either other as named."""
    if dispatch == 'auto':
        return 'exchange' if exchanges(num_experts, shards) else 'global'
    return dispatch

# What a router writes and reads, by name. Its choices, scores and log partitions are sown under
# `ROUTER_COLLECTION` where a caller opens it, and its balancing bias is a variable of
# `EXPERT_BIAS_COLLECTION`, which the step that balances the load rewrites (`dew.objectives.routing`).
ROUTER_COLLECTION = 'router'

@dataclasses.dataclass
class FixedRouters:
    """What a `fixed_routers` block finds of a trace: the module paths of the routers whose expert weight is
    fixed whatever they score (`weighs_one`) and whose scores' parameters the trace differentiates
    (`trained`), and the module paths of the routers whose statistics a term of its loss reads, a balance
    loss or a z-loss (`balanced`), each where its routing was sown."""

    trained: list[tuple[str, ...]] = dataclasses.field(default_factory=list)
    balanced: set[tuple[str, ...]] = dataclasses.field(default_factory=set)


_FIXED: contextvars.ContextVar[FixedRouters | None] = contextvars.ContextVar("dew_fixed_routers",
                                                                            default=None)
"""The open `fixed_routers` block's findings, None outside one."""


@contextlib.contextmanager
def fixed_routers() -> Iterator[FixedRouters]:
    """Within the block, find the fixed routers a trace trains and whether its loss balances them
    (`FixedRouters`), each as it reports itself (`fixed_router`, `balancing`): no term of a loss but a
    balance loss or a z-loss reaches a fixed router's scores (`dew.objectives.routing.refuse_fixed_routers`).
    A router is trained where the trace differentiates its parameters, whatever path a stack, a pipeline
    or a bank stores them under; a frozen one, a distillation's teacher's among them, is not."""
    found = FixedRouters()
    token = _FIXED.set(found)
    try:
        yield found
    finally:
        _FIXED.reset(token)


def weighs_one(top_k: int, normalized: bool, weight_sum_min: float | None = None, *,
               single_softmax: bool = False) -> bool:
    """Whether a router's weight is 1 whatever it scores: one expert's weight renormalized over itself
    (`top_k` 1, `normalized`), softmax's or a sigmoid's alike (a sigmoid's over itself plus its tiny epsilon),
    unless a floor on the weights' sum (`weight_sum_min`) keeps a small score small; or a softmax over a
    single expert (`single_softmax`), normalized or not."""
    return (top_k == 1 and normalized and not weight_sum_min) or single_softmax


def _unchanged[Tree](report: Callable[[], None], tree: Tree) -> Tree:
    """`tree` itself, whatever reads it."""
    return tree


def _reported(report, primals, tangents):
    report()
    return primals[0], tangents[0]


_differentiated = contracted("test_a_routers_differentiation_report_holds_every_order")(
    declared_tangent(Protected(), _unchanged, _reported, nondiff_argnums=(0,)))
"""`tree` itself, calling `report` where a trace differentiates it: JAX traces a custom tangent rule only
for a differentiated input. Its tangent is the input's, the span kept as it is (`Protected`)."""


def fixed_router(path: tuple[str, ...], params: jax.Array | Variables) -> None:
    """Report the router at module `path`, whose weight is fixed (`weighs_one`), to the open `fixed_routers`
    block, as trained where its trace differentiates `params`, the parameters its scores read
    (`_differentiated`)."""
    found = _FIXED.get()
    if found is not None:
        _differentiated(functools.partial(found.trained.append, tuple(path)), params)


def balancing(routers: Iterable[tuple[str, ...]]) -> None:
    """Report to the open `fixed_routers` block that a term of the loss reads the statistics of `routers`,
    each by the module path its routing was sown at."""
    found = _FIXED.get()
    if found is not None:
        found.balanced.update(tuple(router) for router in routers)
ROUTER_INDICES = 'indices'
ROUTER_SCORES = 'scores'
ROUTER_LOG_Z = 'log_z'
ROUTER_RAN = 'ran'
"""Whether a router's call ran, a bool a row sown beside its choice: True for every call made, False for
one a looped pass given a count per call skipped (`dew.nn.backbones.looping`), whose other sows are
zeros, so a reader of the routers' statistics leaves it out. A row's own, so a pipeline's microbatches
join it by their rows as they join the choices."""
EXPERT_BIAS_COLLECTION = 'moe'
EXPERT_BIAS = 'e_score_correction_bias'

# DeepSeek divides the selected weights by their sum plus this, so a token
# whose sigmoid scores are all zero stays finite
# (modeling_deepseek_v3.py:166, maxtext layers/moe.py:839). In fp32 the term
# changes no bit of a denominator above 1e-12. A softmax's choices renormalize
# as the softmax of their own logits instead (`renormalized_softmax`). A
# router states another as its `weight_sum_epsilon` (LFM2-MoE's 1e-6).
WEIGHT_SUM_EPSILON = 1e-20


def expert_softmax(logits: jax.Array) -> jax.Array:
    """The softmax over the experts, the last axis: each exponential over their sum, the sum rounded once
    (`compensated_sum`) and divided by IEEE division (`divided`). A float32 sum of the exponentials in one
    running order left a softmax router's weights 1.74 times as far from float64 as torch's softmax, whose
    vectorized sum rounds less, over vLLM's grouped_topk (tests/test_router_source.py's `softmax`); summed
    so, 0.66 times. Differentiated as a softmax is (`categorical.softmax_tangent`), as PyTorch's softmax
    backward takes it, never through its division, whose derivatives were most of a sparse decoder's
    programs (row 12805: 258k of the 376k operations its test's programs held)."""
    return _expert_softmax(logits)


def _expert_probabilities(logits: jax.Array) -> jax.Array:
    """`expert_softmax`'s value."""
    shifted = jnp.exp(logits - jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True)))
    return divided(shifted, compensated_sum(shifted, (shifted.ndim - 1,))[..., None])


def _expert_tangent(primals, tangents):
    (logits,), (d_logits,) = primals, tangents
    probability = _expert_softmax(logits)
    return probability, categorical.softmax_tangent(probability, d_logits)


_expert_softmax = contracted("test_an_expert_softmax_holds_every_order")(
    declared_tangent(Protected(), _expert_probabilities, _expert_tangent))
"""`expert_softmax`'s value with a softmax's tangent (`categorical.softmax_tangent`), its rule reading the
probabilities again, so a second order meets the rule."""


def renormalized_softmax(logits: jax.Array, indices: jax.Array) -> jax.Array:
    """The softmax's probabilities at `indices` renormalized to sum to one,
    `[..., k]`: the softmax of the chosen logits alone. Renormalizing the full
    softmax's probabilities instead underflows where a replayed choice
    (`chosen_experts`) sits far below the router's own, to 0 / 0."""
    return expert_softmax(jnp.take_along_axis(logits, indices, axis=-1))


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
    """Additive routed-position statistics for load-times-score loss."""
    scores: jax.Array
    counts: jax.Array
    positions: jax.Array
    top_k: int = struct.field(pytree_node=False)


def global_router_loss(stats: RouterMoments, alpha: float) -> jax.Array:
    """DeepSeek V2 auxiliary loss for each pool of routed positions."""
    scores = stats.scores.astype(at_least_fp32(stats.scores.dtype))
    positions = jax.lax.stop_gradient(stats.positions.astype(scores.dtype))
    counts = jax.lax.stop_gradient(stats.counts.astype(scores.dtype))
    denominator = jnp.where(positions > 0, positions, 1) ** 2 * stats.top_k
    return alpha * scores.shape[-1] * jnp.sum(counts * scores, axis=-1) / denominator


def sequence_router_moments(scores: jax.Array, indices: jax.Array, valid: jax.Array,
                            documents: jax.Array | None = None, *,
                            max_segments: int | None = None) -> RouterMoments:
    """Each document's router moments over its real queries."""
    rows, length, experts = scores.shape
    scores = scores.astype(at_least_fp32(scores.dtype))
    segments = (1 if documents is None else length) if max_segments is None else max_segments
    documents = jnp.ones((rows, length), jnp.int32) if documents is None else documents
    owners = jnp.where(valid & (documents != 0), documents - 1, segments)

    def moments(row_scores, row_indices, row_valid, row_owners):
        summed = jax.ops.segment_sum(jnp.where(row_valid[:, None], row_scores, 0), row_owners, segments)
        counts = jnp.zeros((segments, experts), scores.dtype).at[row_owners[:, None], row_indices].add(
            row_valid[:, None].astype(scores.dtype), mode="drop")
        positions = jax.ops.segment_sum(row_valid.astype(scores.dtype), row_owners, segments)
        return RouterMoments(summed, counts, positions, indices.shape[-1])

    return jax.vmap(moments)(scores, indices, valid, owners)


def sequence_router_losses(scores: jax.Array, indices: jax.Array, alpha: float) -> jax.Array:
    """One DeepSeek V2 balance loss per intact sequence (arXiv 2405.04434, section 2.1.4)."""
    stats = sequence_router_moments(scores, indices, jnp.ones(scores.shape[:2], bool))
    return global_router_loss(stats, alpha)[:, 0]


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


def sqrt_softplus(logits: jax.Array) -> jax.Array:
    """DeepSeek V4's score, `sqrt(softplus(logits))`, the plain formula wherever `exp(logits)` is above its
    dtype's epsilon, and below it its tail `exp(logits / 2)`, which the formula equals there to within
    the epsilon: where softplus underflows, the plain formula's derivative, `sqrt`'s at zero, is infinite
    and an unselected expert's zero cotangent made the router's gradient NaN; a selected expert far down
    the tail keeps its score where the plain formula rounded it to zero. Each branch reads only the
    logits its side takes, so neither's derivative is evaluated past its range."""
    tail = jnp.log(jnp.finfo(logits.dtype).eps)
    below = logits < tail
    plain = jnp.sqrt(softplus(jnp.where(below, tail, logits)))
    return jnp.where(below, jnp.exp(jnp.minimum(logits, tail) / 2), plain)


class Router(nn.Module):
    """Choose which experts a token goes to, and with what weight: `[..., k]`.

    The gate projection runs in fp32, as DeepSeek's does
    (`modeling_deepseek_v3.py:146`).

    `expert_bias` is DeepSeek's aux-loss-free balancing bias
    (`e_score_correction_bias`, arXiv 2408.15664), fp32 in the `moe`
    collection. It enters the selection only, so it changes which experts a
    token uses and not what they contribute, and no gradient reaches it.
    `load_balance_update` computes its update and the step that applies it owns
    the write (MaxText `layers/moe.py:965-972` hands it back the same way).
    `bias_in_weights` is LLaDA-MoE's reading of the same buffer: the biased
    scores are also the weights (modeling_lladamoe.py, LLaDAMoESparseMoeBlock).

    `expert_groups` above one is DeepSeek's node limit: a token chooses inside
    the best `groups_per_token` groups, scored by `group_score`, 'top2' (V3's
    `noaux_tc`) or 'max' (V2's `group_limited_greedy`). `normalize_weights` is
    the reference's `norm_topk_prob`; V2's released configs set it false.

    `hash_vocab` makes this DeepSeek V4's hash router (`DeepseekV4HashRouter`,
    modeling_deepseek_v4.py:1045-1073): the experts are the `top_k` entries of a
    fixed `tid2eid` table `[hash_vocab, top_k]` in the `moe` collection at the
    token id the caller passes, weighted by the learned gate. `media_bias` is
    DeepSeek-V4.1's second bias, selecting for image-span tokens in place of
    `expert_bias` (`bias_vl`, model.py:807-820), given the `media` mask.

    `routing_block` with `block_experts` is LLaDA2.2's block routing
    (modeling_llada2_moe.py, LLaDA2MoeGate.block_routing): a block's score for
    an expert is its best token's, and each token chooses among the block's
    `block_experts` best. The caller numbers the blocks (`blocks`,
    `dew.nn.inputs.numbered_blocks`): each row's and each packed document's
    own, of `routing_block` places counted over its real tokens, a partial
    last block routing by its own tokens and padding voting for none. The
    release flattens the batch's rows into one stream before cutting blocks
    and asserts whole ones, so a row's experts there depend on the rows beside
    it; Dew keeps each request's routing its own, which is the release's
    computation wherever a row is whole blocks of real tokens (one row, or
    rows a whole number of blocks long).
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
    bias_in_weights: bool = False
    media_bias: bool = False
    routing_block: int | None = None
    block_experts: int | None = None
    hash_vocab: int | None = None
    weight_sum_epsilon: float = WEIGHT_SUM_EPSILON
    weight_sum_min: float = 0.0
    """The least a renormalization divides the chosen weights by (ERNIE 4.5's `moe_norm_min`, which
    clamps their sum from below where a biased selection picks weights that sum to less): above 0, the
    chosen weights divide by `max(sum + weight_sum_epsilon, weight_sum_min)`, a softmax's too."""
    excluded_group_score: float | None = None
    """What an expert outside a token's best groups is chosen by (`expert_groups`): None for nothing, never
    chosen (DeepSeek V3's -inf), or the score its reference fills in, A.X-K2's 0.0
    (`masked_fill(..., 0.0)`, modeling_axk2.py:340), which a best group's experts biased below zero rank
    under."""
    logit_softcap: float | None = None
    """Where set, the cap `c` of each gate logit, `tanh(logit / c) * c` before its scores, in float32 as
    the logits are (Laguna's `moe_router_logit_softcapping`, modeling_laguna.py:174-175); every reader of
    the logits (the scores, the z-loss) reads them capped."""
    init_std: float | None = None  # normal std of the kernel; None: lecun normal
    precision: PrecisionLike = None

    kernel_axes: LogicalAxes | None = None
    """Its kernel's logical axes, as its owner declares them where it makes it."""
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
        if self.bias_in_weights and not self.expert_bias:
            raise ValueError("bias_in_weights gates on the balancing bias, which expert_bias holds")
        if self.logit_softcap is not None and not self.logit_softcap > 0:
            raise ValueError(f"logit_softcap is a positive cap on the gate logits, or None for none; got "
                             f"{self.logit_softcap}")
        if (self.routing_block is None) != (self.block_experts is None):
            raise ValueError("block routing names its block of tokens and the experts a block may use, "
                             f"got routing_block={self.routing_block} and block_experts={self.block_experts}")
        if self.routing_block is not None:
            if self.routing_block < 1:
                raise ValueError(f"routing_block counts a block's tokens, got {self.routing_block}")
            if self.block_experts is None or not self.top_k <= self.block_experts <= self.num_experts:
                raise ValueError(
                    f"block_experts must hold the top_k of {self.top_k} and at most the "
                    f"{self.num_experts} experts, got {self.block_experts}")
            if self.expert_groups > 1 or self.hash_vocab is not None:
                raise ValueError("block routing limits a token to its block's experts, so it has no "
                                 "expert groups and no hash table")
        # The kernel is the router's own parameter, not a nested Dense, so
        # the leaf is `gate/kernel` where a Hugging Face sparse layer keeps
        # `gate.weight`.
        self.kernel = self.param(
            'kernel', declared(normal_kernel(self.init_std, nn.initializers.lecun_normal())['kernel_init'],
                               self.kernel_axes), (self.in_features, self.num_experts), jnp.float32)
        if self.expert_bias:
            self.bias = self.variable(
                EXPERT_BIAS_COLLECTION, EXPERT_BIAS, declared(jnp.zeros, ('exp',)),
                (self.num_experts,), jnp.float32)
        if self.media_bias:
            self.bias_vl = self.variable(
                EXPERT_BIAS_COLLECTION, 'media_bias', declared(jnp.zeros, ('exp',)), (self.num_experts,),
                jnp.float32)
        if self.hash_vocab is not None:
            if self.expert_bias or self.expert_groups > 1 or self.routing_block is not None:
                raise ValueError(
                    "hash routing selects by the token table alone, so it has no "
                    "balancing bias, no expert groups and no routing blocks")
            # A token's experts, read by token: whole on every device.
            self.tid2eid = self.variable(
                EXPERT_BIAS_COLLECTION, 'tid2eid', declared(jnp.zeros, (None, None)),
                (self.hash_vocab, self.top_k), jnp.int32)

    def __call__(self, x, tokens=None, media=None, routes: Routes | None = None, blocks=None):
        logits = self.logits(x)
        scores = self._activated(logits)
        indices = chosen_experts(lambda: self._selected(scores, tokens, media, blocks),
                                 (*scores.shape[:-1], self.top_k), routes)
        # The load each expert took, for the step that balances the bias:
        # written only when a caller opens the 'router' collection, and never
        # into the tree init returns, where it is not a variable.
        if not self.is_initializing():
            self.sow(ROUTER_COLLECTION, ROUTER_INDICES, indices)
            self.sow(ROUTER_COLLECTION, ROUTER_RAN, jnp.ones(indices.shape[:1], bool))
            # The V2 balance loss reads the scores beside the choices.
            self.sow(ROUTER_COLLECTION, ROUTER_SCORES, scores)
            # The router z-loss squares each position's log partition.
            self.sow(ROUTER_COLLECTION, ROUTER_LOG_Z, jax.nn.logsumexp(logits, axis=-1))
        if weighs_one(self.top_k, self.normalize_weights, self.weight_sum_min,
                      single_softmax=self.num_experts == 1 and self.score_function == 'softmax'):
            fixed_router(self.path, self.kernel)
        if (self.normalize_weights and self.score_function == 'softmax' and not self.bias_in_weights
                and not self.weight_sum_min):
            weights = renormalized_softmax(logits, indices)
        else:
            gating = scores + self.bias.value if self.bias_in_weights else scores
            weights = jnp.take_along_axis(gating, indices, axis=-1)
            if self.normalize_weights:
                total = jnp.sum(weights, axis=-1, keepdims=True) + self.weight_sum_epsilon
                if self.weight_sum_min:
                    total = jnp.maximum(total, self.weight_sum_min)
                weights = weights / total
        return weights * self.routed_scaling_factor, indices

    def _selected(self, scores, tokens, media, blocks):
        """The experts this router chooses on its own: `[..., top_k]`."""
        if (blocks is None) != (self.routing_block is None):
            raise ValueError("a block router chooses each block's experts together, from the blocks its "
                             "caller numbers (dew.nn.inputs.numbered_blocks), and only a block router "
                             "reads them")
        if self.hash_vocab is not None:
            if tokens is None:
                raise ValueError("hash routing selects by the token ids, which the caller passes")
            # The table as it is held, a host array among them, indexed by traced ids.
            return jnp.asarray(self.tid2eid.value)[jnp.asarray(tokens)]
        if tokens is not None:
            raise ValueError("only a hash router reads the token ids")
        selection = self.selection(scores, media, blocks)
        if self.expert_groups > 1:
            fill = -jnp.inf if self.excluded_group_score is None else self.excluded_group_score
            selection = jnp.where(self.group_mask(selection), selection, fill)
        _, indices = jax.lax.top_k(selection, self.top_k)
        return indices

    def selection(self, scores, media=None, blocks=None):
        """What a token's experts are chosen by before its groups are: its scores with the balancing
        bias (a media token's own bias), and outside each block's shortlist -inf."""
        selection = scores if not self.expert_bias else scores + self.bias.value
        if media is not None:
            selection = jnp.where(media[..., None], scores + self.bias_vl.value, selection)
        if blocks is not None and self.block_experts is not None:
            selection = jnp.where(self.block_mask(selection, blocks, self.block_experts), selection, -jnp.inf)
        return selection

    def logits(self, x):
        """Each token's gate logit for every expert, in fp32 (`at_least_fp32`):
        `[..., num_experts]`. They multiply in IEEE float32 where nothing states a precision
        (`exact_float32`): a top-k choice flips between experts whose logits TF32's rounding moves
        past each other, where the references' float32 routers multiply in IEEE float32. A
        `logit_softcap` caps them (`capped_logits`)."""
        logits = float32_einsum('...d,de->...e', x.astype(at_least_fp32(x.dtype)), self.kernel,
                                precision=exact_float32(self.precision))
        return capped_logits(logits, self.logit_softcap)

    def _activated(self, logits):
        if self.score_function == 'softmax':
            return expert_softmax(logits)
        if self.score_function == 'sqrtsoftplus':
            return sqrt_softplus(logits)
        return logistic(logits)

    def group_mask(self, selection):
        """True where an expert sits in one of a token's best groups.

        Patterned on `maxtext layers/moe.py:843` `expert_group_mask`, which
        scores a group by its two best experts. The groups are the expert index
        cut into `expert_groups` contiguous blocks, the layout DeepSeek's
        `n_group` means.
        """
        per_group = self.num_experts // self.expert_groups
        _, groups = jax.lax.top_k(self.group_scores(selection), self.groups_per_token)
        kept = jnp.sum(
            jax.nn.one_hot(groups, self.expert_groups, dtype=jnp.int32), axis=-2)
        return jnp.repeat(kept > 0, per_group, axis=-1)

    def group_scores(self, selection):
        """Each expert group's score, by which a token's best groups are chosen (`group_mask`): its
        best expert's selection score, or its two best's sum. `[..., expert_groups]`."""
        grouped = selection.reshape(*selection.shape[:-1], self.expert_groups,
                                    self.num_experts // self.expert_groups)
        if self.group_score == 'max':
            return jnp.max(grouped, axis=-1)
        best_two, _ = jax.lax.top_k(grouped, 2)
        return jnp.sum(best_two, axis=-1)

    @staticmethod
    def block_mask(selection, blocks, experts: int):
        """True where an expert is one of the `experts` best of its token's block: each block's score
        for an expert is its best token's selection score (the release's `block_routing`). `blocks`
        `[..., S]` numbers each row's blocks from 0 (`numbered_blocks`); a padding token, numbered -1,
        is a block of its own beside them, so it moves no real block's score. A row holds at most as
        many blocks as tokens, so its blocks' scores take the selection's own size, and each block's
        shortlist is written by index, never as a one-hot of its experts."""
        *rows, length, count = selection.shape

        def row(scores, ids):
            ids = jnp.where(ids < 0, length, ids)
            best = jax.ops.segment_max(scores, ids, num_segments=length + 1)
            _, allowed = jax.lax.top_k(best, experts)
            kept = jnp.zeros((length + 1, count), bool).at[jnp.arange(length + 1)[:, None], allowed].set(True)
            return kept[ids]

        flat = jax.vmap(row)(selection.reshape(-1, length, count),
                             jnp.broadcast_to(blocks, (*rows, length)).reshape(-1, length))
        return flat.reshape(selection.shape)


BANKED_EXPERTS = 'banked_experts'
"""The collection a run over a bank in device memory hands its experts' kernels in: each `ExpertLinear`'s
kernel the bank's whole `[layers, experts, in, out]` and the layer read (`experts_apart`, `at_iteration`),
read where it lies (`Layered`)."""


class Layered(struct.PyTreeNode):
    """An expert kernel read in place from a bank: the run's stack `[layers, experts, in, out]` (or a
    packed load's stacked NVFP4 banks, `formats.NvFp4.layer`) and the layer read, which the GEMM template
    reads where it lies (`gemm.gemm`'s `layer`) rather than as a slice copied out a layer at a time (1.2 GB
    a layer at Qwen3-30B-A3B's experts, half a decode step on an H100)."""

    stack: "jax.Array | formats.NvFp4"
    layer: jax.Array

    @property
    def dtype(self) -> jnp.dtype:
        return self.stack.dtype


def _expert_kernel(path: tuple[str, ...]) -> bool:
    """Whether a run's parameter at `path` is an `ExpertMLP`'s expert kernel (`EXPERT_AXES`)."""
    return len(path) >= 3 and path[-1] == 'kernel' and path[-2] in EXPERT_AXES and path[-3] == 'experts'


def _layered(value) -> bool:
    return isinstance(value, Layered)


def holds_experts(params: Mapping) -> bool:
    """Whether a banked run's `params` hold an `ExpertMLP`'s expert kernels (`experts_apart`)."""
    return any(map(_expert_kernel, leaves_by_path(params)))


def experts_apart(variables: Mapping[str, Mapping]) -> dict:
    """A banked run's `params` and `BANKED_EXPERTS` collections with its experts' kernels moved from the
    first to the second, at their paths, so the run reads the rest a layer at a time and hands the kernels
    whole to every layer (`at_iteration`)."""
    params: dict = {}
    moved: dict = {}
    for keys, leaf in leaves_by_path(variables.get('params', {})).items():
        node = moved if _expert_kernel(keys) else params
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = leaf
    return {'params': params, BANKED_EXPERTS: moved}


def at_iteration(variables: Mapping[str, Mapping], layer: jax.Array) -> dict:
    """The `BANKED_EXPERTS` collection with `layer`, the layer of the run read, beside each kernel, as
    its `ExpertLinear` reads them (`Layered`)."""
    def indexed(tree: Mapping) -> dict:
        if 'kernel' in tree:
            return {**tree, 'layer': layer}
        return {key: indexed(value) for key, value in tree.items()}

    return {BANKED_EXPERTS: indexed(variables.get(BANKED_EXPERTS, {}))}


class Gathered(NamedTuple):
    """Expert-sorted rows read where they lie: row `i` is `tokens[rows[i]]`, slot `slots[i]`, times
    `scales[i]` where a dispatch scales its input (Llama 4's `scale_inputs`: each slot's routing weight)
    (`_sorted_locally`'s slots, for a project that takes them: `expert_dispatch`'s `gathered_project`). The
    expert projections multiply them through the template's gathered rows (`gemm.gemm`'s `gather`: Mosaic
    GPU's producer reads each slot's token where it lies, no sorted copy of the tokens made, row 1376), a
    scale in the product's prologue on the rows so read (row 12306), any other use reading them gathered
    (`gathered`); the last projection writes each row at its slot (`gemm.gemm`'s `scatter`), the slots in
    token order."""

    tokens: jax.Array
    rows: jax.Array
    slots: jax.Array
    scales: jax.Array | None = None


def gathered(x: jax.Array | Gathered) -> jax.Array:
    """`x`'s rows, gathered where they are `Gathered`, each times its scale in the tokens' dtype."""
    if not isinstance(x, Gathered):
        return x
    rows = x.tokens[x.rows]
    return rows if x.scales is None else rows * x.scales[:, None].astype(rows.dtype)


def _streamed(x: jax.Array | Gathered, compute, *, hooked: bool = True
              ) -> tuple[jax.Array, jax.Array | None, dict]:
    """What a projection of `x` hands the GEMM template: its tokens, the rows it gathers of them (None for
    rows taken whole), and the prologue that scales the rows so read (`Gathered.scales`), formed in the
    compute dtype as `gathered` forms them in the tokens', the same arithmetic where the two are one. Rows
    of tokens in another dtype, or for a product that takes no prologue (not `hooked`: a packed matrix's,
    `gemm._packed_gemm`), are gathered and scaled in theirs, then rounded once to the product's."""
    if not isinstance(x, Gathered):
        return x, None, {}
    if x.scales is None:
        return x.tokens, x.rows, {}
    if x.tokens.dtype != compute or not hooked:
        return gathered(x), None, {}
    return x.tokens, x.rows, {'prologue': Hook(_scaled, (Side(x.scales.astype(x.tokens.dtype), ('rows',)),))}


def _scaled(rows: jax.Array, scales: jax.Array) -> jax.Array:
    """A scaled stream's rows, each times its scale (`_streamed`'s prologue)."""
    return rows * scales[:, None]


def expert_projection(x: jax.Array | Gathered, kernel: jax.Array | formats.NvFp4 | Layered,
                      group_sizes: jax.Array, dtype: Dtype | None, implementation: str,
                      precision: PrecisionLike, bias: jax.Array | None = None,
                      scatter: jax.Array | None = None, tensor_scale: jax.Array | None = None) -> jax.Array:
    """The grouped matmul of `x`'s expert-sorted rows, `[rows, in]`, through
    each expert's matrix in `kernel`, `[experts, in, out]`, `group_sizes`
    counting each expert's leading rows, under one precision contract for both
    dispatches. A `Layered` kernel is read at its layer of its stack, in place.

    Operands cast to `dtype` (flax's promotion when None), every contraction
    accumulates in at least fp32 and rounds once to the compute dtype, so a
    split width rounds no partial sum. The tangent is `dx @ Q(kernel) + Q(x) @
    dkernel` with `Q` the rounded operands: a kernel gradient keeps the master
    dtype, an input gradient its input's, both accumulating over the widest of
    the compute, input and kernel dtypes, whatever the placement. `bias`, each
    expert's `[experts, out]` where given, is added to its rows' fp32 sums
    before their one rounding (the template's epilogue, which reads it at each
    row's expert: row 3963), rounded to the compute dtype as an operand is, its
    gradient summed over each expert's rows in at least fp32 before it meets
    the bias's own dtype. `Gathered` rows are read where they lie (`gemm.gemm`'s `gather`, at a stack's
    layer too), scaled in the product's prologue (`_streamed`), but by a run's recipe, which takes them
    gathered; with `scatter`, each output row lands at its entry's row (`gemm.gemm`'s `scatter`, a
    permutation). `tensor_scale`, the rows' NVFP4 tensor scale under a delayed-scaling recipe that takes the
    experts (`ExpertLinear.history`, `projections.recipe_projection`'s), which it requires (row 12721).

    Every implementation runs the GEMM template
    (`dew.nn.kernels.gemm.gemm`), which selects each of its
    launches where it runs (`grouped_matmul.GROUPED`: the template's kernels on
    a GPU, an old route on a declared cell, XLA's program off a GPU and on a
    call no launch takes, each route reported with its reason) and holds the
    law in every differentiation mode; 'xla' names XLA's ragged dot for every
    launch (`generation.chosen`), and 'auto' and 'pallas' alike leave the
    choice to the gemm. `generation.chosen(grouped_matmul=...)` names a
    kernel, a launch's name raising where it cannot run the call
    (`selection.KernelRefused`). A CPU process runs XLA's program: a test
    interprets the kernels by calling the template's interpreter itself. XLA's
    ragged dot runs a product over every expert, 70 to 200 times the kernels'
    time on an H100 at Qwen3-30B-A3B's and Kimi Linear's experts (GMM-h100-2),
    so where the kernels run they are the choice. tests/test_moe_precision.py
    measures it against float64 sums and three Adam steps on every expert/fsdp
    layout.
    """
    stack, layer = (kernel.stack, kernel.layer) if isinstance(kernel, Layered) else (kernel, None)
    packed = isinstance(stack, formats.NvFp4)
    compute = canonicalize_dtype(x.tokens if isinstance(x, Gathered) else x, _held(stack),
                                 *(() if bias is None else (bias,)), dtype=dtype)
    # The run's low-precision recipe takes the experts' product where their role takes it, the matrices'
    # values the compute dtype's held in the master's, so their gradient keeps the master dtype. A stack's
    # layer is cut only for it: elsewhere the grouped matmul reads the layer in place. A packed bank is its
    # checkpoint's own codes, frozen, its product the NVFP4 one (`gemm.nvfp4_product`) under any recipe.
    with projection_role("experts"):
        recipe = (None if recipe_role() is None or packed else projections.recipe_projection(
            jnp.asarray(gathered(x), compute),
            rounded_operand(stack if layer is None else stack[layer], compute), group_sizes,
            out_dtype=compute, bias=None if bias is None else rounded_operand(bias, compute),
            tensor_scale=tensor_scale))
    if recipe is not None:
        return recipe if scatter is None else jnp.zeros_like(recipe).at[scatter].set(recipe,
                                                                                    unique_indices=True)
    tokens, rows, prologue = _streamed(x, compute, hooked=not packed)
    with _named(implementation):
        return gemm.gemm(tokens, stack, group_sizes, compute=compute, out_dtype=compute, precision=precision,
                         layer=layer, gather=rows, scatter=scatter, **prologue, **_biased(bias, compute))


def _biased(bias: jax.Array | None, compute) -> dict:
    """The template's epilogue adding each expert's bias to its rows' sums (`expert_projection`): the bias
    rounded to `compute` as an operand is, held in at least fp32, so its gradient sums each expert's rows so
    before it meets the bias's dtype (`rounded_operand`)."""
    if bias is None:
        return {}
    work = jnp.result_type(bias.dtype, compute, jnp.float32)
    held = jnp.asarray(rounded_operand(bias, compute)).astype(work)
    return {'epilogue': Hook(_plus_bias, (Side(held, ('groups', 'columns')),))}


def _plus_bias(acc: jax.Array, bias: jax.Array) -> jax.Array:
    """An expert's rows' sums plus its bias, before their one rounding (`_biased`)."""
    return acc + bias


def paired_projections(x: jax.Array | Gathered, first: jax.Array | Layered, second: jax.Array | Layered,
                       group_sizes: jax.Array, dtype: Dtype | None, implementation: str,
                       precision: PrecisionLike) -> tuple[jax.Array, jax.Array]:
    """`expert_projection` of the same rows through two expert matrices of one shape (a gate's and an up
    projection's), as one paired product (`gemm.gemm`'s `paired`): a launch that takes the pair reads
    the rows once for both, and the call hands the two products on apart where its launch takes none
    (`grouped_matmul._mosaic_takes`), each under `expert_projection`'s law. On an H100 the pair took 0.79
    to 1.0 of the two products' time at the routings of DeepSeek-V3, Qwen3-30B-A3B and GPT-OSS-20B, its
    outputs theirs bit for bit (FS-3699-PAIRED2-h100, ledger 3699). Both products take one compute dtype,
    the rows' and both matrices' however a matrix is held; two stacks read at one layer index (`Layered`,
    the same index: `ExpertMLP`'s kernels, `_one_layer`) pair at it, read in place, and stacks read at two
    indices each at its own. `Gathered` rows are read and scaled where they lie, as `expert_projection`
    reads them."""
    compute, paired = _paired(x, first, second, dtype)
    first_rows, second_rows = (_apart(x, (first, second), group_sizes, compute, implementation, precision)
                               if paired is None else
                               _paired_call(paired, group_sizes, compute, implementation, precision))
    return first_rows, second_rows


def gated_experts(x: jax.Array | Gathered, gate: jax.Array | Layered, up: jax.Array | Layered,
                  down: jax.Array | Layered, group_sizes: jax.Array, dtype: Dtype | None, implementation: str,
                  precision: PrecisionLike, activation: GatedActivation, limit: float | None,
                  scatter: jax.Array | None = None, scales: tuple[jax.Array | None, ...] | None = None,
                  noted: Callable[[jax.Array], None] | None = None) -> jax.Array:
    """A gated expert MLP's output (gate, up and down projections of `x`'s expert-sorted rows, the gated
    product of `activation` between them, clamped by `limit`), as `dew.nn.gated_mlp.fused_products` runs a
    dense one (row 1376): the gate and up projections one paired product, their roundings and their gated
    product its epilogue (`activations.gated_hook`: the product's arithmetic on the tile, no launch of its
    own), the gate's and up's kept under the names a remat policy reads (`Residual.GATE_PROJ`,
    `Residual.UP_PROJ`); the down projection's prologue that gated product of the kept roundings, the
    epilogue's output its formed value (`gemm.gemm`'s `applied`). So neither backward recomputes the pair:
    the paired product's reads its kept roundings, the down projection's its prologue's operands, which a
    policy saving the two names keeps. The gated product's gradient is `gated_hook`'s portable one (its
    activation rounded to the compute dtype, as `GatedMLP`'s fused products take it), check 1 against
    float64 and bitwise the unfused hook's; a dense layer's standalone `norm_swiglu` launch multiplies by
    the unrounded fp32 activation, a bf16 ulp apart where they differ. It runs so where the gated MLP's
    selector chooses its fused kernel for the grouped call (`gated_mlp.fuses_grouped`, `GATED_MLP`'s
    `(dtype, 'grouped')` cell, row 12847): where a receipt declares it faster or the block names it
    (`generation.chosen(gated_mlp='fused')`) and its requirements hold. Where the selector takes it apart,
    or a requirement hands it off (the pair runs apart, `paired_projections`; the down projection takes no
    prologue, a packed bank's NVFP4 product or a run's recipe that takes the experts, `expert_projection`;
    the gated product's hook lowers on no launch's epilogue, a `Situ`, an exact GELU's erfc), each
    projection runs on its own and the gated product between them (`gated_product`). Every arm's output the
    compute dtype's. `scatter`, `expert_projection`'s. `scales`, the gate's, up's and down's
    delayed-scaling NVFP4 tensor scales (`ExpertLinear.history`, row 12721) where a run's delayed recipe
    takes the experts, which runs them apart, and `noted`, what takes the down projection's rows (their
    gated product) as its recipe reads them."""
    compute, paired = _paired(x, gate, up, dtype)
    down_stack, down_layer = (down.stack, down.layer) if isinstance(down, Layered) else (down, None)
    hook = None if isinstance(activation, Situ) else gated_hook(activation, limit)
    gate_stack = gate.stack if isinstance(gate, Layered) else gate
    # Fused where the gated MLP's selector chooses it (`gated_mlp.GATED_MLP`, its grouped cell, row 12847):
    # the pair one product of a layer the down projection's, which takes the gated product as its prologue
    # (no NVFP4 bank's, no recipe's that takes the experts), the hook lowering on the template's launches.
    packed = isinstance(gate_stack, formats.NvFp4) or isinstance(down_stack, formats.NvFp4)
    with _named(implementation):
        # Asked of every call, one whose pair runs apart among them, so the selector reports each grouped
        # MLP's route. A packed bank's matrices are its codes, of no product's shape a launch reads (`None`).
        # Asked of the calls the chain makes: its tokens and the gathered rows' indices, its group sizes, the
        # down projection's scatter and the layer of a stack of layers (`_paired_call`'s); whole where those
        # launches read every operand whole as placed (`launched_whole`), which their shapes cannot say: the
        # pair's streamed rows, scales and stacks (`_paired`), the down stack, the sizes and the scatter.
        tokens = x.tokens if isinstance(x, Gathered) else x
        fused = fuses_grouped(
            jax.ShapeDtypeStruct(tokens.shape, compute),
            None if packed else (jax.ShapeDtypeStruct(gate_stack.shape, compute),
                                 jax.ShapeDtypeStruct(down_stack.shape, compute)),
            precision, hook, paired=paired is not None and down_layer is paired[-1],
            down_prologue=not isinstance(down_stack, formats.NvFp4) and not _recipe_takes_experts(down_stack),
            whole=launched_whole(paired, down_stack, group_sizes, scatter), group_sizes=group_sizes,
            gather=x.rows if isinstance(x, Gathered) else None, scatter=scatter, layer=down_layer)
    # The selector fuses a pair alone, of a gated product with a hook (its `paired` and `hook` requirements).
    if paired is None or hook is None or not fused:
        gate_scale, up_scale, down_scale = scales or (None, None, None)
        gate_rows, up_rows = (_apart(x, (gate, up), group_sizes, compute, implementation, precision,
                                     (gate_scale, up_scale))
                              if paired is None else
                              _paired_call(paired, group_sizes, compute, implementation, precision))
        gate_rows, up_rows = (checkpoint_name(gate_rows, Residual.GATE_PROJ),
                              checkpoint_name(up_rows, Residual.UP_PROJ))
        hidden = gated_product(activation)(*swiglu_clamped(gate_rows, up_rows, limit))
        if noted is not None:
            noted(hidden)
        return _computed(expert_projection(hidden, down, group_sizes, compute, implementation, precision,
                                           scatter=scatter, tensor_scale=down_scale), compute)
    gate_rows, up_rows, hidden = _paired_call(
        paired, group_sizes, compute, implementation, precision, epilogue=rounded_gated(hook, compute),
        named=(Residual.GATE_PROJ, Residual.UP_PROJ, None))
    # Named where a remat policy reads them too, as the down projection's operands (`fused_products`).
    gate_rows, up_rows = (checkpoint_name(gate_rows, Residual.GATE_PROJ),
                          checkpoint_name(up_rows, Residual.UP_PROJ))
    with _named(implementation):
        out = gemm.gemm(gate_rows, down_stack, group_sizes, compute=compute, out_dtype=compute,
                        precision=precision, layer=down_layer, scatter=scatter,
                        prologue=Hook(hook, (Side(up_rows, ('rows', 'depth')),)), applied=hidden)
    return _computed(out, compute)


def _apart(x: jax.Array | Gathered, kernels: tuple, group_sizes: jax.Array, compute: Dtype,
           implementation: str, precision: PrecisionLike,
           scales: tuple[jax.Array | None, ...] | None = None) -> tuple[jax.Array, ...]:
    """`expert_projection` of `x` through each of `kernels` on its own, at its delayed-scaling NVFP4 tensor
    scale of `scales` where given, each output the compute dtype's."""
    scales = scales or (None,) * len(kernels)
    return tuple(_computed(expert_projection(x, kernel, group_sizes, compute, implementation, precision,
                                             tensor_scale=scale), compute)
                 for kernel, scale in zip(kernels, scales, strict=True))


def _paired_call(paired: tuple, group_sizes: jax.Array, compute: Dtype, implementation: str,
                 precision: PrecisionLike, *, epilogue=None, named: tuple = ()) -> tuple[jax.Array, ...]:
    """The paired product `_paired` describes (`paired`), its outputs the compute dtype's: the two products,
    or what `epilogue` makes of their accumulators, `named` naming them (`gemm.gemm`'s)."""
    tokens, rows, prologue, first_stack, second_stack, layer = paired
    count = len(named) if named else 2
    hooks = {} if epilogue is None else {"epilogue": Hook(epilogue), "named": named}
    with _named(implementation):
        outputs = gemm.gemm(tokens, first_stack, group_sizes, compute=compute, out_dtype=compute,
                            out_dtypes=(compute,) * count, precision=precision, paired=second_stack,
                            layer=layer, gather=rows, **prologue, **hooks)
    return tuple(_computed(output, compute) for output in outputs)


def _computed(output: jax.Array, compute: Dtype) -> jax.Array:
    """A projection's output in the compute dtype, whichever arm made it: paired or apart, its dtype the
    same (a no-op where it already is)."""
    return output.astype(compute)


def _paired(x: jax.Array | Gathered, first: jax.Array | Layered, second: jax.Array | Layered,
            dtype: Dtype | None) -> tuple[Dtype, tuple | None]:
    """The compute dtype of a paired product of `first` and `second` over `x` (`paired_projections`: the
    rows' and both matrices' however a matrix is held), and what the product takes: the template's tokens,
    the rows it gathers and its prologue (`_streamed`), both stacks and the layer read; None where the two
    run apart: matrices of two layer indices (or a stack's layer beside a matrix), a
    packed bank (its product the NVFP4 one, which pairs with none), or a run's low-precision recipe that
    takes the experts (`recipe_refusal`); experts the recipe leaves pair as without one."""
    (first_stack, layer), (second_stack, second_layer) = (
        (kernel.stack, kernel.layer) if isinstance(kernel, Layered) else (kernel, None)
        for kernel in (first, second))
    compute = canonicalize_dtype(x.tokens if isinstance(x, Gathered) else x, _held(first_stack),
                                 _held(second_stack), dtype=dtype)
    if (layer is not second_layer or isinstance(first_stack, formats.NvFp4)
            or isinstance(second_stack, formats.NvFp4) or _recipe_takes_experts(first_stack)):
        return compute, None
    tokens, rows, prologue = _streamed(x, compute)
    return compute, (tokens, rows, prologue, first_stack, second_stack, layer)


def _one_layer(kernels: tuple) -> tuple:
    """An `ExpertMLP`'s kernels as its projections read them: a bank's (`Layered`) every one at the first's
    layer index. They are one layer's by construction (`at_iteration` writes the iteration's index beside
    every kernel of the run), which the dispatch hands on as a tracer a kernel (`derivatives.span`), so the
    gate and up projections pair at that one index (`paired_projections`)."""
    if not all(isinstance(kernel, Layered) for kernel in kernels):
        return kernels
    layer = kernels[0].layer
    return tuple(Layered(kernel.stack, layer) for kernel in kernels)


def _recipe_takes_experts(matrix: jax.Array) -> bool:
    """Whether the run's low-precision recipe takes experts of `matrix`'s widths (`recipe_refusal`), in the
    experts' role `expert_projection` multiplies them in, whatever role its caller traces in."""
    with projection_role("experts"):
        return projections.recipe_takes(*matrix.shape[-2:])


def _held(kernel):
    """What an expert matrix's compute dtype reads of `kernel` (`canonicalize_dtype`, which takes arrays): a
    packed bank's dtype, the one its decode yields (`formats.NvFp4.dtype`), as a scalar of it, else the
    array."""
    return jnp.zeros((), kernel.dtype) if isinstance(kernel, formats.NvFp4) else kernel


def _named(implementation: str) -> contextlib.AbstractContextManager:
    """The launch choice an expert projection's `implementation` names: XLA's ragged dot for 'xla', the
    template's own for 'auto' and 'pallas' (`expert_projection`); any other name is refused."""
    if implementation not in GROUPED_MATMULS:
        raise ValueError(f"implementation must be one of {list(GROUPED_MATMULS)}, got {implementation!r}")
    if implementation != 'xla':
        return contextlib.nullcontext()
    return generation.chosen(**{generation.Choice.GROUPED_MATMUL: 'xla'})




class ExpertLinear(nn.Module):
    """One matrix per expert, `[exp, in_features, features]`; `ExpertMLP`
    reads the kernel and projects tokens sorted by expert through
    `expert_projection`. `kernel_axes` are its logical axes, as its owner
    declares them where it makes it."""
    num_experts: int
    in_features: int
    features: int
    init_std: float | None = None  # normal std of every expert; None: per-expert lecun normal
    kernel_axes: LogicalAxes | None = None

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """One matrix per dispatched token, not one for every stored expert."""
        return projection_flops(lengths, self.in_features, self.features)

    def setup(self):
        if self.has_variable(BANKED_EXPERTS, 'kernel'):
            # A scanned decode over a bank hands the run's stack and the iteration's layer.
            self.kernel = Layered(self.get_variable(BANKED_EXPERTS, 'kernel'),
                                  self.get_variable(BANKED_EXPERTS, 'layer'))
            self.history = None
            return
        # With the expert dimension as a batch axis, fan_in is per expert and
        # every expert initialises like the matching Dense of a dense MLP. A served checkpoint's packed
        # bank stands in for it where its collection holds one (`dew.nn.dense.packed_matrix`).
        self.kernel = packed_matrix(self, 'kernel', lambda: self.param(
            'kernel',
            declared(normal_kernel(self.init_std, nn.initializers.variance_scaling(
                1.0, 'fan_in', 'truncated_normal', in_axis=-2, out_axis=-1,
                batch_axis=(0,)))['kernel_init'], self.kernel_axes),
            (self.num_experts, self.in_features, self.features), jnp.float32))
        # Its amax history under a delayed-scaling NVFP4 recipe that takes the experts (`dense.amax_history`,
        # row 12721), as a Dense holds its own; a packed bank the recipe never takes holds none.
        with projection_role("experts"):
            self.history = (None if isinstance(self.kernel, formats.NvFp4)
                            else amax_history(self, 'amax_history', (self.in_features, self.features)))


def capacity_positions(indices: jax.Array, num_experts: int,
                       capacity_factor: float) -> tuple[jax.Array, int]:
    """Each routed slot's place in its expert's queue, and how many places a
    queue keeps.

    `indices` is `[..., length, top_k]`, each leading row one sequence. A
    sequence queues its slots at each expert in token order (a token's choices
    are distinct experts). A queue keeps MaxText's `expert_capacity_per_batch`
    (`layers/moe.py` `generate_masks`): `max(ceil(length * top_k / num_experts)
    * capacity_factor, capacity_factor)`, truncated. Counting within a sequence
    makes the dropped slots a property of the batch and not of its placement.
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


type Projected = tuple[jax.Array, tuple[jax.Array, ...]]
"""What an expert layer's projection hands its dispatch (`expert_dispatch`): a row per slot, and the
largest magnitude among the rows each of its projections holding a delayed-scaling NVFP4 history quantized
(`admitted_amax`), in its own order, () where none holds one: the dispatch takes the largest over its
calls, rounds and devices, and the layer records them (`dense.observed`, row 12721)."""


def admitted_amax(x: jax.Array | Gathered, group_sizes: jax.Array, compute: Dtype) -> jax.Array:
    """The largest magnitude among `x`'s rows a recipe's grouped product quantizes
    (`projections.recipe_projection`): those `group_sizes` admits, the first `sum(group_sizes)` as
    computed in `compute` (a slot capacity dropped, past them, it never quantizes), finite
    (`projections.nvfp4_amax`); its derivative none (row 12721)."""
    rows = jnp.asarray(gathered(x), compute)
    admitted = jnp.arange(rows.shape[0]) < jnp.sum(group_sizes)
    return jax.lax.stop_gradient(projections.nvfp4_amax(jnp.where(admitted[:, None], rows, 0)))


def _largest(first: tuple[jax.Array, ...], second: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    return tuple(jnp.maximum(one, other) for one, other in zip(first, second, strict=True))


def expert_dispatch[Parameters](
        project: Callable[[jax.Array, jax.Array, jax.Array, Parameters], Projected],
        x: jax.Array, indices: jax.Array, parameters: Parameters,
        parameter_axes: Sequence[LogicalAxes], *,
        num_experts: int, dispatch: str, output_dtype: Dtype,
        input_weights: jax.Array | None = None, initializing: bool = False,
        capacity_factor: float | None = None,
        gathered_project: Callable[[Gathered, jax.Array, jax.Array, Parameters], Projected] | None = None
        ) -> Projected:
    """Run every routed token through its expert, and return the slots in
    token order.

    `project` takes rows sorted by expert, the group sizes, the sorted expert
    ids and the expert-major parameters, and returns a row per slot, of its own
    width, which may differ from the tokens' (MoVA's values,
    `dew.nn.value_experts`), with its projections' largest magnitudes (`Projected`),
    which the dispatch returns beside the slots, each the largest over its calls,
    rounds and devices; `parameter_axes` names the parameters' dimensions.
    The caller owns the activation, biases and output weights; `input_weights`
    scales each expert's input instead, as Llama 4 does. `x` is
    `[batch, length, width]` or `[tokens, width]`.

    `dispatch='auto'`, the default, is `'exchange'` on an expert mesh axis greater
    than one that divides the experts, where the experts stay in place, and
    `'global'` on any other (ledger 1711: on a 4-way expert mesh, global
    gathered 200,704 bytes of eight experts' weights a step, exchange moved
    4,096). `dispatch='global'` routes the tokens where they are, in a `shard_map` that
    holds them as the residual stream does, so each device runs its own rows
    through every expert on device-local rows, as the Pallas kernels need
    (MaxText's `sparse_matmul_route_and_compute`). `'exchange'` is expert
    parallelism with the expert axis manual: each device sends each shard
    owning experts its slots' rows in one `ragged_all_to_all` and takes them
    back in another (`_exchange_shard`), keeping every slot; rows the batch axes
    do not divide are padded. Parameters enter in their stored shards and are
    gathered inside, so gradients leave reduce-scattered.

    `capacity_factor` drops as GShard and MaxText do (`capacity_positions`),
    the same slots under either dispatch and placement, and bounds what a shard
    receives. Parameters take part in at least fp32, so gradients sum across
    devices in fp32 and meet the master dtype once.

    `gathered_project`, `project` as it takes `Gathered` rows (`ExpertMLP`'s),
    is handed the slots of a dispatch that sorts them where they are and
    scales no input as the tokens, the rows each reads and the slots each
    lands at, no sorted copy of the tokens made, and returns its rows in slot
    order, no un-permute (row 1376).
    """
    if dispatch not in EXPERT_DISPATCHES:
        raise ValueError(f"dispatch must be one of {EXPERT_DISPATCHES}, got {dispatch!r}")
    if indices.shape[:-1] != x.shape[:-1] or (
            input_weights is not None and input_weights.shape != indices.shape):
        raise ValueError(f"routing {indices.shape} does not describe tokens {x.shape}")
    mesh = jax.sharding.get_abstract_mesh()
    shards = mesh.shape.get(EXPERT_AXIS, 1)
    dispatch = resolved_dispatch(dispatch, num_experts, shards)
    if dispatch == 'exchange' and not initializing and not exchanges(num_experts, shards):
        raise LayoutRefused("exchange dispatch needs an expert mesh axis greater than one "
                         "that divides num_experts")
    capacity = None
    if capacity_factor is not None:
        positions, capacity = capacity_positions(indices, num_experts, capacity_factor)
        # A dropped slot takes the sentinel id past the last expert, which
        # no group counts and no bucket sends.
        indices = jnp.where(positions < capacity, indices, num_experts)
    run = functools.partial(
        _dispatched, project, parameter_axes=parameter_axes, num_experts=num_experts,
        dispatch=dispatch, output_dtype=output_dtype, initializing=initializing,
        capacity=capacity, gathered_project=gathered_project)
    if mesh.shape.get(STAGE_AXIS, 1) > 1:
        # A pipeline's stages trace the dispatch inside their own manual
        # axes, which a jvp rule traced outside them cannot hold; and a
        # pipeline trains, so its forward has nothing to save by reading
        # the kernels as stored.
        return run(x, indices, _widened(parameters), input_weights)
    return _stored_primal(run, x, indices, parameters, input_weights)


def _stored(run, x, indices, parameters, input_weights):
    """`run` over the parameters as stored, and differentiated over them at
    least fp32 (`_widened`), so a gradient summed across devices meets the
    master dtype once, after the sum: a
    kernel's glue (`_stored_primal`, `dew.nn.derivatives.declared_tangent`),
    the whole expert layer, which the algebra pass rewrites as any other math
    around its launches (a protected span would hold every MoE layer out of
    the pass). A forward that nothing differentiates, serving above all, reads
    the stored kernels: widened, every layer's experts were copied to fp32 in
    the step, 14.35 GiB of live temporaries for gpt-oss-20b served over an
    expert axis of four RTX 3090s."""
    return run(x, indices, parameters, input_weights)


def expert_compute_dtype(x: jax.Array, *parameters: "jax.Array | formats.NvFp4",
                         dtype: Dtype | None) -> Dtype:
    """The dtype an expert layer computes in: `dtype` when the model names
    one, else flax's promotion of the stream and the parameters, except that
    16-bit parameters are never widened. A wider stream then rounds to them
    and every product accumulates in fp32, as a bf16 checkpoint's reference
    runs it; promoted, each layer's experts were copied to fp32, 14.35 GiB
    of live temporaries serving gpt-oss-20b over four RTX 3090s."""
    compute = canonicalize_dtype(x, *map(_held, parameters), dtype=dtype)
    stored = {parameter.dtype for parameter in parameters}
    if dtype is None and len(stored) == 1:
        (kind,) = stored
        # Narrower than the stream, not than the promotion: a bf16 stream over
        # fp16 experts promotes to fp32, and fp16 would overflow its values.
        if (jnp.issubdtype(kind, jnp.floating) and jnp.issubdtype(x.dtype, jnp.floating)
                and jnp.finfo(kind).bits < jnp.finfo(x.dtype).bits):
            return kind
    return compute


def _stored_primal_jvp(run, primals, tangents):
    x, indices, parameters, input_weights = primals
    dx, _, dparameters, dweights = tangents
    # Only the live tangents are differentiated; a symbolic zero stays a
    # closed-over primal, since jax 0.11.2 cannot transpose a zero tangent
    # through `portable_projection`'s barrier.
    leaves, tree = jax.tree.flatten((x, parameters, input_weights))
    directions = jax.tree.leaves((dx, dparameters, dweights),
                                 is_leaf=lambda leaf: isinstance(leaf, SymbolicZero))
    live = [index for index, direction in enumerate(directions)
            if not isinstance(direction, SymbolicZero)]

    def forward(*values):
        current = list(leaves)
        for index, value in zip(live, values, strict=True):
            current[index] = value
        x, parameters, input_weights = jax.tree.unflatten(tree, current)
        return run(x, indices, _widened(parameters), input_weights)

    if not live:
        output = forward()
        return output, jax.tree.map(lambda leaf: SymbolicZero(jax.typeof(leaf)), output)
    # Through the rule's dual (`with_tangents_formed_at`, `x` the first, the experts' parameters the rest):
    # reverse mode returns `x`'s cotangent only with the experts' gradients formed.
    directions = with_tangents_formed_at(tuple(directions), range(1, len(directions)))
    return jax.jvp(forward, tuple(leaves[index] for index in live),
                   tuple(directions[index] for index in live))


_stored_primal = contracted("test_a_stored_expert_primal_holds_every_order")(
    declared_tangent(KernelGlue(), _stored, _stored_primal_jvp, nondiff_argnums=(0,),
                     symbolic_zeros=True))


def _dispatched[Parameters](
        project: Callable[[jax.Array, jax.Array, jax.Array, Parameters], Projected],
        x: jax.Array, indices: jax.Array, parameters: Parameters,
        input_weights: jax.Array | None, *, parameter_axes: Sequence[LogicalAxes],
        num_experts: int, dispatch: str, output_dtype: Dtype, initializing: bool,
        capacity: int | None,
        gathered_project: Callable[[Gathered, jax.Array, jax.Array, Parameters], Projected] | None = None
        ) -> Projected:
    """`expert_dispatch` past its checks, over the parameters it is given."""
    mesh = jax.sharding.get_abstract_mesh()
    shards = mesh.shape.get(EXPERT_AXIS, 1)
    exchanging = dispatch == 'exchange' and not initializing and x.size > 0
    if not exchanging and (initializing or mesh.empty):
        return _sorted_locally(project, x, indices, parameters, input_weights,
                               num_experts=num_experts, gathered_project=gathered_project)
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
    manual = spec_axes(tokens)
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
    def spec(leaf: jax.Array, axes: LogicalAxes) -> P:
        return kept_axes(logical_spec(axes, leaf.shape), manual.__contains__)

    # A kernel read in place from a bank (`Layered`) holds its experts on its stack's second dimension.
    stored, experts = [], []
    for parameter, axes in zip(jax.tree.leaves(parameters, is_leaf=_layered), parameter_axes, strict=True):
        if _layered(parameter):
            stack = spec(parameter.stack, (None, *axes))
            stored += [stack, P()]
            experts.append(stack[1:])
        else:
            stored.append(spec(parameter, axes))
            experts.append(stored[-1])
    held = jax.tree.unflatten(jax.tree.structure(parameters), stored)
    if exchanging:
        if any(not entry or EXPERT_AXIS not in mesh_axes(entry[0]) for entry in experts):
            raise LayoutRefused("exchange dispatch holds each device's own experts, so every "
                             "expert parameter splits its expert dimension over the expert "
                             f"axis; the rule table places them {stored}")
        inner = functools.partial(_exchange_shard, project, num_experts=num_experts,
                                  shards=shards, output_dtype=output_dtype, capacity=capacity)
    else:
        inner = functools.partial(_sorted_locally, project, num_experts=num_experts,
                                  gathered_project=gathered_project)

    def body(x, indices, parameters, input_weights):
        def gathers(spec: P) -> list[tuple[int, tuple[str, ...]]]:
            """The dimensions a parameter of `spec` gathers over the manual axes, but its experts under
            the exchange."""
            named = [(dimension, tuple(axis for axis in mesh_axes(entry)
                                       if not (exchanging and dimension == 0 and axis == EXPERT_AXIS)))
                     for dimension, entry in enumerate(spec)]
            return [(dimension, axes) for dimension, axes in named if axes]

        def gathered(parameter, spec):
            if _layered(parameter):
                layer_spec = P(*spec.stack[1:])
                if not gathers(layer_spec):
                    return parameter
                # A gathered kernel is a new array whatever is read: its layer's alone is gathered.
                parameter = jax.lax.dynamic_index_in_dim(parameter.stack, parameter.layer, keepdims=False)
                spec = layer_spec
            for dimension, axes in gathers(spec):
                parameter = jax.lax.all_gather(parameter, axes, axis=dimension, tiled=True)
            return parameter

        projected, observed = inner(x, indices, jax.tree.map(gathered, parameters, held, is_leaf=_layered),
                                    input_weights)
        # The largest magnitudes over every device's rows, each stage's its own (a pipeline's stages run
        # other layers).
        return projected, jax.lax.pmax(observed, tuple(sorted(manual - {STAGE_AXIS})))

    # The kernels carry no manual-axis type, so this map does not check them;
    # MaxText hosts its gmm the same way.
    projected, observed = jax.shard_map(
        body, mesh=mesh, axis_names=manual,
        in_specs=(tokens, routing, held, None if input_weights is None else routing),
        out_specs=(P(*routing, None), P()), check_vma=False)(
            x, indices, parameters, input_weights)
    return projected[:rows], observed


def _sorted_locally[Parameters](
        project: Callable[[jax.Array, jax.Array, jax.Array, Parameters], Projected],
        x: jax.Array, indices: jax.Array, parameters: Parameters,
        input_weights: jax.Array | None, *, num_experts: int,
        gathered_project: Callable[[Gathered, jax.Array, jax.Array, Parameters], Projected] | None = None
        ) -> Projected:
    """Project every slot of `x`'s tokens where they are: sort the slots by
    expert, gather their token rows and put the results back in slot order
    (or hand `gathered_project` the tokens, the rows each slot reads, the
    slot each lands at and each row's input weight where the input is scaled,
    `Gathered`, and take its rows in slot order). A dropped slot's sentinel id
    sorts last, outside every group, and its row comes back zero."""
    top_k = indices.shape[-1]
    tokens = x.reshape(-1, x.shape[-1])
    experts = indices.ravel()
    order = jnp.argsort(experts).astype(jnp.int32)
    sizes = jnp.bincount(experts, length=num_experts)
    if gathered_project is not None:
        rows = Gathered(tokens, order // top_k, order,
                        None if input_weights is None else input_weights.ravel()[order])
        projected, observed = gathered_project(rows, sizes, experts[order], parameters)
        projected = jnp.where((experts < num_experts)[:, None], projected, 0)
        return projected.reshape(*indices.shape, projected.shape[-1]), observed
    grouped = tokens[order // top_k]
    if input_weights is not None:
        grouped = grouped * input_weights.ravel()[order][:, None].astype(grouped.dtype)
    projected, observed = project(grouped, sizes, experts[order], parameters)
    projected = jnp.where((experts[order] < num_experts)[:, None], projected, 0)
    return projected[jnp.argsort(order)].reshape(*indices.shape, projected.shape[-1]), observed


def _exchange_shard[Parameters](
        project: Callable[[jax.Array, jax.Array, jax.Array, Parameters], Projected],
        x: jax.Array, indices: jax.Array, parameters: Parameters,
        input_weights: jax.Array | None, *, num_experts: int, shards: int,
        output_dtype: Dtype, capacity: int | None) -> Projected:
    """Run one device's share of `expert_dispatch`'s exchange, inside its
    shard_map, its projections' largest magnitudes the largest over its rounds.

    The device sorts its slots by expert and sends each shard its run of them
    in one `ragged_all_to_all`, the shard projects what every device sent it,
    grouped by its experts, and a second one brings each row back where it was
    sent from: XLA's own collective, each row moved once, whatever the routing's
    skew. Every device learns the whole exchange's counts, each device's slots for
    each expert, from one gather of its own (`all_gather`), and so where its rows
    land on each shard, where theirs come back, and each received row's expert,
    as a device sends a shard its rows expert by expert. Each exchange is a round
    of at most a balanced share a pair of devices (`share`, a device's slots over the
    shards, or what a capacity lets it keep for one shard where that is fewer): a
    shard's receiving buffer holds every device's share once, the most a balanced
    routing sends it, and a skewed routing's overflow takes further rounds, as many
    as its largest run needs, each projected and sent back before the next.
    """
    top_k = indices.shape[-1]
    tokens = x.reshape(-1, x.shape[-1])
    slots, per_shard = indices.size, num_experts // shards
    # The most slots one device sends one shard.
    most = slots if capacity is None else min(slots, math.prod(indices.shape[:-2]) * per_shard * capacity)
    share = min(most, -(-slots // shards))
    order = jnp.argsort(indices.ravel())
    experts = indices.ravel()[order]
    rows = order // top_k
    # Each route's cotangent comes back over the exchange in the rows' own dtype, rounded there once, and a
    # token's routes sum wide (`rows_gathered_wide`, as a gathered product's do, row 12396).
    send = rows_gathered_wide(tokens, rows)
    if input_weights is not None:
        send = send * input_weights.ravel()[order][:, None].astype(send.dtype)
    # Every device's slots for every expert, `[sender, expert]` (a dropped slot's sentinel expert sorts last
    # and is sent to no shard), its slots for every shard, `[sender, receiver]`, and where each run starts on
    # its sender: after its runs for every earlier shard.
    counts = jax.lax.all_gather(jnp.bincount(experts, length=num_experts + 1)[:num_experts].astype(jnp.int32),
                                EXPERT_AXIS, tiled=False)
    table = counts.reshape(shards, shards, per_shard).sum(-1)
    me = jax.lax.axis_index(EXPERT_AXIS)
    sent_from = jnp.cumsum(table, axis=1) - table
    # This shard's experts' slots on each sender, whose run to it they make in order.
    ends = jnp.cumsum(jax.lax.dynamic_slice_in_dim(counts, me * per_shard, per_shard, axis=1), axis=1)
    rounds = -(-jnp.max(table) // share)
    regions = jnp.arange(shards, dtype=jnp.int32) * share

    def exchanged(start: int | jax.Array, returned: jax.Array | None) -> Projected:
        """One round: each sender's next `share` rows of its run to every shard, from `start` into it,
        projected there and sent back into `returned`, or, for the first round, into rows of the projection's
        own width (a value expert's need not be its input's)."""
        sending = jnp.clip(table[me] - start, 0, share)
        receiving = jnp.clip(table[:, me] - start, 0, share)
        # A sender's rows land in its own region of the buffer, `share` rows a sender.
        received = jax.lax.ragged_all_to_all(
            send, jnp.zeros((shards * share, tokens.shape[-1]), send.dtype), sent_from[me] + start, sending,
            jnp.full((shards,), me * share, jnp.int32), receiving, axis_name=EXPERT_AXIS)
        # A received row's expert on this shard, by where it sits in its sender's run; the region past what
        # arrived holds the sentinel.
        place = start + jnp.arange(share, dtype=jnp.int32)
        received_ids = jax.vmap(lambda bounds: jnp.searchsorted(bounds, place, side='right'))(ends)
        received_ids = jnp.where(jnp.arange(share) < receiving[:, None], received_ids, per_shard).ravel()
        permutation = jnp.argsort(received_ids)
        groups = jnp.bincount(received_ids, length=per_shard + 1)[:per_shard]
        computed, observed = project(received[permutation], groups,
                                     received_ids[permutation].astype(experts.dtype), parameters)
        computed = jnp.where((received_ids < per_shard)[:, None], computed[jnp.argsort(permutation)],
                             0).astype(output_dtype)
        if returned is None:
            returned = jnp.zeros((slots, computed.shape[-1]), output_dtype)
        # Each sender's rows back where it sent them from.
        return jax.lax.ragged_all_to_all(computed, returned, regions, receiving, sent_from[:, me] + start,
                                         sending, axis_name=EXPERT_AXIS), observed

    def round_of(index, held):
        # Every device takes the same rounds, from the gathered table; a round's largest magnitudes kept
        # beside the earlier rounds'.
        def taken(held):
            returned, observed = exchanged(index * share, held[0])
            return returned, _largest(held[1], observed)

        return jax.lax.cond(index < rounds, taken, lambda held: held, held)

    returned, observed = exchanged(0, None)
    # The most rounds any routing takes, a device's every slot sent to one shard; a capacity's bound past
    # the balanced share alone takes further ones.
    if most > share:
        returned, observed = jax.lax.fori_loop(1, -(-most // share), round_of, (returned, observed))
    # The rows in slot order; a dropped slot's, sent nowhere, stays zero.
    return returned[jnp.argsort(order)].reshape(*indices.shape, returned.shape[-1]), observed


def _widened[Tree](parameters: Tree) -> Tree:
    """Floating parameters at least fp32, so a cotangent summed across
    devices is summed before its master dtype rounds it."""
    def widen(leaf):
        if isinstance(leaf, formats.NvFp4):
            # A packed bank is frozen: its codes, block scales and tensor scales are read as they are.
            return leaf
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            return leaf.astype(at_least_fp32(leaf.dtype))
        return leaf
    return jax.tree.map(widen, parameters, is_leaf=lambda leaf: isinstance(leaf, formats.NvFp4))


EXPERT_AXES: Mapping[str, LogicalAxes] = {
    'gate_proj': ('exp', 'embed', 'mlp'),
    'up_proj': ('exp', 'embed', 'mlp'),
    'down_proj': ('exp', 'mlp', 'embed'),
}
"""The stacked expert kernels' axes, in the order `ExpertMLP` projects through them."""


class ExpertMLP(nn.Module):
    """The routed experts of one layer: each token through the MLPs its
    router chose.

    Tokens gather into expert order straight from their rows (MaxText's
    `moe_use_direct_token_gather`), the projections run as grouped
    matmuls, and the results return to token order summed with their router
    weights in fp32 (`maxtext layers/moe.py:940` `permute`, `:1101`
    `unpermute`). `dispatch` and `capacity_factor` are `expert_dispatch`'s;
    `expert_projection` holds one precision contract under both dispatches.

    `relu2` is Nemotron-H's ungated expert: up_proj, squared ReLU, down_proj.
    It allocates no gate and uses two grouped matmuls. Other activations
    keep the gated three-projection path.

    `swiglu_limit` is DeepSeek V4's clamp before the activation
    (`DeepseekV4Experts._apply_gate`): the gate is capped from above and the up
    projection on both sides. `scale_inputs` is Llama 4's placement of the
    routing weight on each expert's input with outputs summed unweighted
    (`Llama4TextMoe.forward`), not the same as weighting outputs, since the gate
    is not linear.
    """
    num_experts: int
    hidden_features: int
    out_features: int
    activation: GatedActivation = 'swiglu'
    implementation: str = 'auto'
    dispatch: str = 'auto'
    capacity_factor: float | None = None
    swiglu_limit: float | None = None
    scale_inputs: bool = False
    init_std: float | None = None  # gate/up normal std; None: per-expert lecun normal
    output_init_std: float | None = None  # down normal std; None follows init_std
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """Maps for dispatched rows, one row per active expert assignment."""
        if self.capacity_factor is not None:
            raise MissingFlopsCount("ExpertMLP needs the admitted routing counts when capacity drops tokens")
        maps = 2 if self.activation == 'relu2' else 3
        return maps * projection_flops(lengths, self.out_features, self.hidden_features)

    def setup(self):
        if self.swiglu_limit is not None and self.swiglu_limit <= 0:
            raise ValueError(
                f"swiglu_limit caps the gate and up projections, so it is "
                f"positive, got {self.swiglu_limit}; None leaves them unclamped")
        expert = functools.partial(ExpertLinear, num_experts=self.num_experts)
        if self.activation == 'relu2':
            if self.swiglu_limit is not None:
                raise ValueError('swiglu_limit requires gated experts')
        else:
            self.gate_proj = expert(in_features=self.out_features,
                                    features=self.hidden_features, init_std=self.init_std,
                                    name='gate_proj', kernel_axes=EXPERT_AXES['gate_proj'])
        self.up_proj = expert(in_features=self.out_features,
                              features=self.hidden_features, init_std=self.init_std,
                              name='up_proj', kernel_axes=EXPERT_AXES['up_proj'])
        self.down_proj = expert(in_features=self.hidden_features,
                                features=self.out_features, name='down_proj',
                                kernel_axes=EXPERT_AXES['down_proj'],
                                init_std=(self.init_std if self.output_init_std is None
                                          else self.output_init_std))

    @nn.nowrap
    def _projections(self) -> tuple[ExpertLinear, ...]:
        """The layer's projections in the order it runs them."""
        if self.activation == 'relu2':
            return self.up_proj, self.down_proj
        return self.gate_proj, self.up_proj, self.down_proj

    def _project(self, tokens: jax.Array | Gathered, sizes: jax.Array, _expert_ids: jax.Array,
                 kernels: tuple[jax.Array, ...], *, dtype: Dtype) -> Projected:
        kernels = _one_layer(kernels)
        layers = self._projections()
        # Each projection's delayed-scaling NVFP4 tensor scale, its history's (`ExpertLinear.history`, row
        # 12721), and the largest magnitude among the rows it quantizes for each holding one, the dispatch's
        # to return (`Projected`): the gate and up projections' its rows', the down projection's its hidden
        # rows'.
        scales = tuple(None if layer.history is None
                       else projections.delayed_tensor_scale(layer.history.value) for layer in layers)
        delayed = any(scale is not None for scale in scales)
        hidden_rows: list[jax.Array] = []

        def observed() -> tuple[jax.Array, ...]:
            held = [layer for layer in layers if layer.history is not None]
            if not held:
                return ()
            read = admitted_amax(tokens, sizes, dtype)
            return tuple(admitted_amax(hidden_rows[0], sizes, dtype) if layer is layers[-1] else read
                         for layer in held)

        def linear(x: jax.Array | Gathered, kernel: jax.Array, scatter: jax.Array | None = None,
                   tensor_scale: jax.Array | None = None) -> jax.Array:
            return jnp.asarray(expert_projection(
                x, kernel, sizes, dtype, self.implementation, self.precision, scatter=scatter,
                tensor_scale=tensor_scale))

        # Gathered rows' slots: the last projection writes each row at its slot (`Gathered`).
        slots = tokens.slots if isinstance(tokens, Gathered) else None

        # The same residual names as the dense MLP's, so one remat policy
        # covers both (residuals.RESIDUALS).
        if self.activation == 'relu2':
            up_scale, down_scale = scales
            up = checkpoint_name(linear(tokens, kernels[0], tensor_scale=up_scale), Residual.UP_PROJ)
            hidden_rows.append(relu2(up))
            out = checkpoint_name(linear(hidden_rows[0], kernels[1], slots, down_scale), Residual.DOWN_PROJ)
            return out, observed()
        # The gate and up projections read the same rows: one paired product, their gated product its
        # epilogue and the down projection's prologue (`gated_experts`).
        gate, up, down = kernels
        out = checkpoint_name(gated_experts(
            tokens, gate, up, down, sizes, dtype, self.implementation, self.precision, self.activation,
            self.swiglu_limit, slots, scales=scales if delayed else None,
            noted=hidden_rows.append if layers[-1].history is not None else None), Residual.DOWN_PROJ)
        return out, observed()

    def _combine(self, slots: jax.Array, weights: jax.Array) -> jax.Array:
        wide = at_least_fp32(slots.dtype)
        if self.scale_inputs:
            return jnp.sum(slots.astype(wide), axis=-2).astype(slots.dtype)
        return float32_einsum('...ke,...k->...e', slots.astype(wide),
                          weights.astype(wide), precision=self.precision).astype(slots.dtype)

    def __call__(self, x: jax.Array, weights: jax.Array, indices: jax.Array) -> jax.Array:
        if weights.shape != indices.shape:
            raise ValueError(f"routing {indices.shape} does not describe weights {weights.shape}")
        names = ('up_proj', 'down_proj') if self.activation == 'relu2' else tuple(EXPERT_AXES)
        kernels = tuple(getattr(self, name).kernel for name in names)
        # One compute dtype for every projection, named rather than
        # inferred inside the dispatch, where the parameters may be widened.
        compute = expert_compute_dtype(x, *(kernel.stack if isinstance(kernel, Layered) else kernel
                                            for kernel in kernels), dtype=self.dtype)
        project = functools.partial(self._project, dtype=compute)
        # Each leaf's axes: a packed bank's four its own (`quantization.expert_leaf_axes`).
        axes = tuple(leaf for kernel, name in zip(kernels, names, strict=True)
                     for leaf in quantization.expert_leaf_axes(kernel, EXPERT_AXES[name]))
        slots, observed = expert_dispatch(
            project, x, indices, kernels, axes, num_experts=self.num_experts, dispatch=self.dispatch,
            initializing=self.is_initializing(), output_dtype=compute,
            input_weights=weights if self.scale_inputs else None,
            capacity_factor=self.capacity_factor, gathered_project=project)
        # What each projection holding a history quantized, the largest over the dispatch's calls, rounds and
        # devices, recorded here, outside its transformations (row 12721).
        held = [layer for layer in self._projections() if layer.history is not None]
        for layer, amax in zip(held, observed, strict=True):
            noted_amax(layer, 'amax_history', amax)
        return self._combine(slots, weights)


class SparseMLP(nn.Module):
    """A router over `num_experts` MLPs, `top_k` of them per token.

    Goes where `GatedMLP` goes and holds the submodules a Hugging Face sparse
    layer names: `gate`, `experts`, and with `shared` (a factory taking a name)
    `shared_experts`, the dense branch every token takes, added to the routed
    output as `DeepseekV3MoE` does. `shared_gate` weights it by a learned
    scalar sigmoid, as Qwen3_5MoeSparseMoeBlock does.

    `latent_features` is Kimi K3's latent MoE (`KimiSparseMoeBlock`,
    modeling_kimi_linear.py:762-838 of moonshotai/Kimi-K3 at f831ab6): the
    router reads the full width, `routed_expert_down_proj` narrows to the
    latent width the experts run at, and their weighted sum goes through
    `latent_norm` and `routed_expert_up_proj` back to `out_features`. The shared
    branch reads the full width.
    """
    num_experts: int
    top_k: int
    hidden_features: int
    out_features: int
    activation: GatedActivation = 'swiglu'
    implementation: str = 'auto'
    dispatch: str = 'auto'
    capacity_factor: float | None = None
    score_function: str = 'softmax'
    normalize_weights: bool = True
    routed_scaling_factor: float = 1.0
    expert_groups: int = 1
    groups_per_token: int = 1
    group_score: str = 'top2'
    expert_bias: bool = False
    bias_in_weights: bool = False
    media_bias: bool = False
    routing_block: int | None = None
    block_experts: int | None = None
    hash_vocab: int | None = None
    weight_sum_epsilon: float = WEIGHT_SUM_EPSILON
    weight_sum_min: float = 0.0
    router_softcap: float | None = None
    excluded_group_score: float | None = None
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

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """Router, active expert assignments and the shared branch, never all experts."""
        width = self.out_features if self.latent_features is None else self.latent_features
        expert = ExpertMLP(num_experts=self.num_experts, hidden_features=self.hidden_features,
                           out_features=width, activation=self.activation,
                           capacity_factor=self.capacity_factor, parent=None)
        count = (projection_flops(lengths, self.out_features, self.num_experts)
                 + self.top_k * expert.model_flops(lengths))
        if not self.scale_inputs:
            # The experts' outputs combine at their own width, the latent one before its up projection.
            count += projection_flops(lengths, self.top_k, width)
        if self.latent_features is not None:
            count += 2 * projection_flops(lengths, self.out_features, self.latent_features)
        if self.shared is not None:
            count += counted_flops(self.shared(parent=None), lengths)
            if self.shared_gate:
                count += projection_flops(lengths, self.out_features, 1)
        return count

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
                           bias_in_weights=self.bias_in_weights,
                           media_bias=self.media_bias,
                           routing_block=self.routing_block,
                           block_experts=self.block_experts,
                           weight_sum_epsilon=self.weight_sum_epsilon, weight_sum_min=self.weight_sum_min,
                           logit_softcap=self.router_softcap, excluded_group_score=self.excluded_group_score,
                           hash_vocab=self.hash_vocab, init_std=self.init_std,
                           # Its experts split as the experts it chooses do.
                           kernel_axes=("embed", "exp"), precision=self.precision, name='gate')
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
            dense = functools.partial(Dense, use_bias=False, dtype=self.dtype, precision=self.precision,
                                      dot_general=float32_dot_general)
            self.routed_expert_down_proj = dense(self.latent_features, name='routed_expert_down_proj',
                                                 kernel_axes=("embed", None))
            self.routed_expert_up_proj = dense(self.out_features, name='routed_expert_up_proj',
                                               kernel_axes=(None, "embed"))
            if self.latent_norm is not None:
                self.routed_expert_norm = self.latent_norm(name='routed_expert_norm')
        if self.shared_gate and self.shared is None:
            raise ValueError("shared_gate requires a shared expert")
        if self.shared is not None:
            self.shared_experts = self.shared(name='shared_experts')
            if self.shared_gate:
                self.shared_expert_gate = Dense(
                    1, use_bias=False, dtype=self.dtype, precision=self.precision,
                    name='shared_expert_gate', **normal_kernel(self.init_std), kernel_axes=("embed", None),
                    dot_general=float32_dot_general)

    def projection_groups(self) -> tuple[ProjectionGroup, ...]:
        """The packed groups its shared branch declares (`ProjectionSites`);
        the routed experts keep their stacked kernels."""
        return () if self.shared is None else declared_groups(self.shared_experts)

    @property
    def routes_experts(self) -> bool:
        """It takes a layer's replayed experts (`RoutesExperts`)."""
        return True

    def __call__(self, x, tokens=None, media=None, routes: Routes | None = None, blocks=None):
        weights, indices = self.gate(x, tokens, media, routes, blocks)
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
            with projection_role("shared_gate"):  # The gate keeps the run's dtypes under a recipe.
                shared = shared * logistic(self.shared_expert_gate(x))
        return routed + shared
