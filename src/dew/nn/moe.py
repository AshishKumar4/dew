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

import functools
import math
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn, struct
from flax.linen.dtypes import canonicalize_dtype
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name
from jax.custom_derivatives import SymbolicZero
from jax.sharding import PartitionSpec as P

from dew.nn.dense import Dense

from .activations import GatedActivation, gated_product, relu2, swiglu_clamped
from .blocks import normal_kernel
from .flops import MissingFlopsCount, counted_flops, projection_flops
from .kernels import grouped_matmul
from .kernels.generation import backend, triton_runs
from .kernels.grouped_matmul import GROUPED, portable_projection
from .kernels.selection import select, selected
from .kernels.templates.gemm import template
from .kernels.templates.product import declared
from .precision import at_least_fp32, float32_dot_general, float32_einsum, float32_products, rounded_operand
from .protocols import ProjectionGroup, Routes, declared_groups
from .sharding import (
    EXPERT_AXIS,
    STAGE_AXIS,
    LayoutRefused,
    LogicalAxes,
    batch_axes,
    logical_axes,
    logical_spec,
    mesh_axes,
    seen_whole,
)

# 'softmax' normalizes a token's affinities over the experts (Mixtral,
# Qwen3.5); 'sigmoid' scores each expert on its own (DeepSeek V3, GLM, Kimi,
# LLaDA2); 'sqrtsoftplus' is DeepSeek V4's sqrt(softplus(logit))
# (transformers activations.py, SqrtSoftplusActivation), unbounded above
# where a sigmoid saturates.
SCORE_FUNCTIONS = ('softmax', 'sigmoid', 'sqrtsoftplus')

# 'pallas' is the kernels, each launch running the first implementation that takes it
# (`grouped_matmul.GROUPED`) or the one `generation.chosen(grouped_matmul=...)` names ('triton',
# 'mosaic_gpu', 'cutlass'); 'auto' is the kernels where they run the call, else XLA
# (`expert_projection`, through the GEMM template).
GROUPED_MATMULS = ('auto', 'xla', 'pallas')
EXPERT_DISPATCHES = ('global', 'exchange')

# DeepSeek divides the selected weights by their sum plus this, so a token
# whose sigmoid scores are all zero stays finite
# (modeling_deepseek_v3.py:166, maxtext layers/moe.py:839). In fp32 the term
# changes no bit of a denominator above 1e-12. A softmax's choices renormalize
# as the softmax of their own logits instead (`renormalized_softmax`).
WEIGHT_SUM_EPSILON = 1e-20


def renormalized_softmax(logits: jax.Array, indices: jax.Array) -> jax.Array:
    """The softmax's probabilities at `indices` renormalized to sum to one,
    `[..., k]`: the softmax of the chosen logits alone. Renormalizing the full
    softmax's probabilities instead underflows where a replayed choice
    (`chosen_experts`) sits far below the router's own, to 0 / 0."""
    return jax.nn.softmax(jnp.take_along_axis(logits, indices, axis=-1), axis=-1)


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
    scores = stats.scores.astype(jnp.promote_types(stats.scores.dtype, jnp.float32))
    positions = jax.lax.stop_gradient(stats.positions.astype(scores.dtype))
    counts = jax.lax.stop_gradient(stats.counts.astype(scores.dtype))
    denominator = jnp.where(positions > 0, positions, 1) ** 2 * stats.top_k
    return alpha * scores.shape[-1] * jnp.sum(counts * scores, axis=-1) / denominator


def sequence_router_moments(scores: jax.Array, indices: jax.Array, valid: jax.Array,
                            documents: jax.Array | None = None, *,
                            max_segments: int | None = None) -> RouterMoments:
    """Each document's router moments over its real queries."""
    rows, length, experts = scores.shape
    scores = scores.astype(jnp.promote_types(scores.dtype, jnp.float32))
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
        if self.bias_in_weights and not self.expert_bias:
            raise ValueError("bias_in_weights gates on the balancing bias, which expert_bias holds")
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
            'kernel', normal_kernel(self.init_std, nn.initializers.lecun_normal())['kernel_init'],
            (self.in_features, self.num_experts), jnp.float32)
        if self.expert_bias:
            self.bias = self.variable(
                'moe', 'e_score_correction_bias', jnp.zeros,
                (self.num_experts,), jnp.float32)
        if self.media_bias:
            self.bias_vl = self.variable('moe', 'media_bias', jnp.zeros, (self.num_experts,), jnp.float32)
        if self.hash_vocab is not None:
            if self.expert_bias or self.expert_groups > 1 or self.routing_block is not None:
                raise ValueError(
                    "hash routing selects by the token table alone, so it has no "
                    "balancing bias, no expert groups and no routing blocks")
            self.tid2eid = self.variable(
                'moe', 'tid2eid', jnp.zeros, (self.hash_vocab, self.top_k), jnp.int32)

    def __call__(self, x, tokens=None, media=None, routes: Routes | None = None, blocks=None):
        logits = self.logits(x)
        scores = self._activated(logits)
        indices = chosen_experts(lambda: self._selected(scores, tokens, media, blocks),
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
        if self.normalize_weights and self.score_function == 'softmax' and not self.bias_in_weights:
            weights = renormalized_softmax(logits, indices)
        else:
            gating = scores + self.bias.value if self.bias_in_weights else scores
            weights = jnp.take_along_axis(gating, indices, axis=-1)
            if self.normalize_weights:
                weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + WEIGHT_SUM_EPSILON)
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
            return self.tid2eid.value[jnp.asarray(tokens)]
        if tokens is not None:
            raise ValueError("only a hash router reads the token ids")
        selection = self.selection(scores, media, blocks)
        if self.expert_groups > 1:
            selection = jnp.where(self.group_mask(selection), selection, -jnp.inf)
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
        `[..., num_experts]`."""
        return float32_einsum('...d,de->...e', x.astype(at_least_fp32(x.dtype)), self.kernel,
                          precision=self.precision)

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
    """The grouped matmul of `x`'s expert-sorted rows, `[rows, in]`, through
    each expert's matrix in `kernel`, `[experts, in, out]`, `group_sizes`
    counting each expert's leading rows, under one precision contract for both
    dispatches.

    Operands cast to `dtype` (flax's promotion when None), every contraction
    accumulates in at least fp32 and rounds once to the compute dtype, so a
    split width rounds no partial sum. The tangent is `dx @ Q(kernel) + Q(x) @
    dkernel` with `Q` the rounded operands: a kernel gradient keeps the master
    dtype, an input gradient its input's, both accumulating over the widest of
    the compute, input and kernel dtypes, whatever the placement.

    'xla' runs `jax.lax.ragged_dot`; 'auto' and 'pallas' the GEMM template
    (`dew.nn.kernels.templates.gemm.template.gemm`), its launches where they run
    the call (a GPU of sm80 on that states the tensor rate their cost model
    reads, and for 'pallas' a CPU, which interprets them) and XLA's program
    elsewhere, a TPU's above all, which the route reports with the reason; both
    hold it in every differentiation mode. A call no launch takes here (the
    grouped selector's requirements, `grouped_matmul.GROUPED`: a product the
    kernels cannot compute exactly, a trace they cannot take, `seen_whole`)
    runs 'xla', reported; `generation.chosen(grouped_matmul=...)` names a kernel
    after the call's own 'xla', a launch's name raising where it cannot run the
    call (`selection.KernelRefused`). XLA's ragged dot runs a product over every
    expert, 70 to 200 times the kernels' time on an H100 at Qwen3-30B-A3B's and
    Kimi Linear's experts (GMM-h100-2), so where the kernels run they are the
    choice. tests/test_moe_precision.py measures it against float64 sums and
    three Adam steps on every expert/fsdp layout.
    """
    if implementation not in GROUPED_MATMULS:
        raise ValueError(f"implementation must be one of {list(GROUPED_MATMULS)}, got {implementation!r}")
    compute = canonicalize_dtype(x, kernel, dtype=dtype)
    resolved = float32_products(compute, precision)
    named = 'xla' if implementation == 'xla' else None
    # The template's forward launches as the call leaves them, which take it or leave it to XLA here,
    # before the template's dispatch selects each of its launches: the kernels run in a GPU's process,
    # or interpreted in a CPU's where 'pallas' asks.
    def spec(array):
        return jax.ShapeDtypeStruct(array.shape, compute)

    product = declared(compute, resolved, outputs=(compute,), operands=(compute, kernel.dtype))
    case = grouped_matmul.grouped_case(*grouped_matmul.gmm_call(
        spec(x), spec(kernel), jax.ShapeDtypeStruct(group_sizes.shape, jnp.int32), out_dtype=compute,
        trans_rhs=False, interpret=not triton_runs(), promoted=product.promoted), product=product,
        whole=seen_whole(x) and seen_whole(kernel),
        placed=triton_runs() or (implementation == 'pallas' and backend() == 'cpu'))
    if not selected(GROUPED, case, named).generic:
        # The GEMM template: its launches where they run the call (each reports its route), XLA's
        # program elsewhere, differentiable at every order.
        return template.gemm(x, kernel, group_sizes, compute=compute, out_dtype=compute, precision=resolved,
                             interpret_on_cpu=implementation == 'pallas')
    select(GROUPED, case, named)
    return portable_projection(x, kernel, group_sizes, dtype, resolved)




class ExpertLinear(nn.Module):
    """One matrix per expert, `[exp, in_features, features]`; `ExpertMLP`
    reads the kernel and projects tokens sorted by expert through
    `expert_projection`."""
    num_experts: int
    in_features: int
    features: int
    init_std: float | None = None  # normal std of every expert; None: per-expert lecun normal

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """One matrix per dispatched token, not one for every stored expert."""
        return projection_flops(lengths, self.in_features, self.features)

    def setup(self):
        # With the expert dimension as a batch axis, fan_in is per expert and
        # every expert initialises like the matching Dense of a dense MLP.
        self.kernel = self.param(
            'kernel',
            normal_kernel(self.init_std, nn.initializers.variance_scaling(
                1.0, 'fan_in', 'truncated_normal', in_axis=-2, out_axis=-1,
                batch_axis=(0,)))['kernel_init'],
            (self.num_experts, self.in_features, self.features), jnp.float32)


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


def expert_dispatch[Parameters](
        project: Callable[[jax.Array, jax.Array, jax.Array, Parameters], jax.Array],
        x: jax.Array, indices: jax.Array, parameters: Parameters,
        parameter_axes: Sequence[LogicalAxes], *,
        num_experts: int, dispatch: str, output_dtype: Dtype,
        input_weights: jax.Array | None = None, initializing: bool = False,
        capacity_factor: float | None = None) -> jax.Array:
    """Run every routed token through its expert, and return the slots in
    token order.

    `project` takes rows sorted by expert, the group sizes, the sorted expert
    ids and the expert-major parameters, and returns rows of the same width;
    `parameter_axes` names the parameters' dimensions. The caller owns the
    activation, biases and output weights; `input_weights` scales each expert's
    input instead, as Llama 4 does. `x` is `[batch, length, width]` or
    `[tokens, width]`.

    `dispatch='global'` routes the tokens where they are, in a `shard_map` that
    holds them as the residual stream does, so each device runs its own rows
    through every expert on device-local rows, as the Pallas kernels need
    (MaxText's `sparse_matmul_route_and_compute`). `'exchange'` is expert
    parallelism with the expert axis manual: each device trades buckets of its
    slots with the shards owning their experts in `all_to_all` rounds
    (`_exchange_shard`), keeping every slot; rows the batch axes do not divide
    are padded. Parameters enter in their stored shards and are gathered
    inside, so gradients leave reduce-scattered.

    `capacity_factor` drops as GShard and MaxText do (`capacity_positions`),
    the same slots under either dispatch and placement, and the exchange then
    needs one round. Parameters take part in at least fp32, so gradients sum
    across devices and rounds in fp32 and meet the master dtype once.
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
        capacity=capacity)
    if mesh.shape.get(STAGE_AXIS, 1) > 1:
        # A pipeline's stages trace the dispatch inside their own manual
        # axes, which a jvp rule traced outside them cannot hold; and a
        # pipeline trains, so its forward has nothing to save by reading
        # the kernels as stored.
        return run(x, indices, _widened(parameters), input_weights)
    return _stored_primal(run, x, indices, parameters, input_weights)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _stored_primal(run, x, indices, parameters, input_weights):
    """`run` over the parameters as stored, and differentiated over them at
    least fp32 (`_widened`), so a gradient summed across devices or over
    the exchange's rounds meets the master dtype once, after the sum. A
    forward that nothing differentiates, serving above all, reads the
    stored kernels: widened, every layer's experts were copied to fp32 in
    the step, 14.35 GiB of live temporaries for gpt-oss-20b served over an
    expert axis of four RTX 3090s."""
    return run(x, indices, parameters, input_weights)


def expert_compute_dtype(x: jax.Array, *parameters: jax.Array, dtype: Dtype | None) -> Dtype:
    """The dtype an expert layer computes in: `dtype` when the model names
    one, else flax's promotion of the stream and the parameters, except that
    16-bit parameters are never widened. A wider stream then rounds to them
    and every product accumulates in fp32, as a bf16 checkpoint's reference
    runs it; promoted, each layer's experts were copied to fp32, 14.35 GiB
    of live temporaries serving gpt-oss-20b over four RTX 3090s."""
    compute = canonicalize_dtype(x, *parameters, dtype=dtype)
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
        return output, SymbolicZero(jax.typeof(output))
    return jax.jvp(forward, tuple(leaves[index] for index in live),
                   tuple(directions[index] for index in live))


_stored_primal.defjvp(_stored_primal_jvp, symbolic_zeros=True)


def _dispatched[Parameters](
        project: Callable[[jax.Array, jax.Array, jax.Array, Parameters], jax.Array],
        x: jax.Array, indices: jax.Array, parameters: Parameters,
        input_weights: jax.Array | None, *, parameter_axes: Sequence[LogicalAxes],
        num_experts: int, dispatch: str, output_dtype: Dtype, initializing: bool,
        capacity: int | None) -> jax.Array:
    """`expert_dispatch` past its checks, over the parameters it is given."""
    mesh = jax.sharding.get_abstract_mesh()
    shards = mesh.shape.get(EXPERT_AXIS, 1)
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
            raise LayoutRefused("exchange dispatch holds each device's own experts, so every "
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

    The device sorts its slots by expert and sends each shard one bucket in a
    first round. Dropless, the bucket is its balanced share, and what a skewed
    routing leaves follows in later rounds of the same size, as many as the
    expert axis's maximum needs. Under a capacity the first bucket holds what
    the device's sequences can keep for one peer and nothing is left over. The
    later rounds run in a checkpointed scan, which recomputes rather than keeps
    their intermediates, so they are as few as the bucket size allows.
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
    dispatch: str = 'global'
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
                                    name='gate_proj')
        self.up_proj = expert(in_features=self.out_features,
                              features=self.hidden_features, init_std=self.init_std,
                              name='up_proj')
        self.down_proj = expert(in_features=self.hidden_features,
                                features=self.out_features, name='down_proj',
                                init_std=(self.init_std if self.output_init_std is None
                                          else self.output_init_std))

    def _project(self, tokens: jax.Array, sizes: jax.Array, _expert_ids: jax.Array,
                 kernels: tuple[jax.Array, ...], *, dtype: Dtype) -> jax.Array:
        def linear(x: jax.Array, kernel: jax.Array) -> jax.Array:
            return jnp.asarray(expert_projection(
                x, kernel, sizes, dtype, self.implementation, self.precision))

        # The same residual names as the dense MLP's, so one remat policy
        # covers both (mixer_base.RESIDUALS).
        if self.activation == 'relu2':
            up = checkpoint_name(linear(tokens, kernels[0]), 'up_proj')
            hidden = relu2(up)
            return checkpoint_name(linear(hidden, kernels[1]), 'down_proj')
        gate = checkpoint_name(linear(tokens, kernels[0]), 'gate_proj')
        up = checkpoint_name(linear(tokens, kernels[1]), 'up_proj')
        gate, up = swiglu_clamped(gate, up, self.swiglu_limit)
        return checkpoint_name(linear(gated_product(self.activation)(gate, up), kernels[2]), 'down_proj')

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
        compute = expert_compute_dtype(x, *kernels, dtype=self.dtype)
        slots = expert_dispatch(
            functools.partial(self._project, dtype=compute), x, indices, kernels,
            tuple(EXPERT_AXES[name] for name in names), num_experts=self.num_experts, dispatch=self.dispatch,
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
    dispatch: str = 'global'
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
            dense = functools.partial(Dense, use_bias=False, dtype=self.dtype, precision=self.precision,
                                      dot_general=float32_dot_general)
            self.routed_expert_down_proj = dense(self.latent_features, name='routed_expert_down_proj')
            self.routed_expert_up_proj = dense(self.out_features, name='routed_expert_up_proj')
            if self.latent_norm is not None:
                self.routed_expert_norm = self.latent_norm(name='routed_expert_norm')
        if self.shared_gate and self.shared is None:
            raise ValueError("shared_gate requires a shared expert")
        if self.shared is not None:
            self.shared_experts = self.shared(name='shared_experts')
            if self.shared_gate:
                self.shared_expert_gate = Dense(
                    1, use_bias=False, dtype=self.dtype, precision=self.precision,
                    name='shared_expert_gate', **normal_kernel(self.init_std),
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
            shared = shared * jax.nn.sigmoid(self.shared_expert_gate(x))
        return routed + shared
