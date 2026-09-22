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
import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from flax.linen.dtypes import canonicalize_dtype, promote_dtype
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name
from jax.custom_derivatives import SymbolicZero
from jax.sharding import PartitionSpec as P

from .blocks import normal_kernel
from .precision import rounded_operand
from .precision import precision_names, rounded_operand
from .sharding import EXPERT_AXIS, logical_axes

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
        if self.hash_vocab is not None:
            if self.expert_bias or self.expert_groups > 1:
                raise ValueError(
                    "hash routing selects by the token table alone, so it has no "
                    "balancing bias and no expert groups")
            self.tid2eid = self.variable(
                'moe', 'tid2eid', jnp.zeros, (self.hash_vocab, self.top_k), jnp.int32)

    def __call__(self, x, tokens=None):
        logits = self.logits(x)
        scores = self._activated(logits)
        if self.hash_vocab is not None:
            if tokens is None:
                raise ValueError("hash routing selects by the token ids, which the caller passes")
            indices = self.tid2eid.value[jnp.asarray(tokens)]
        else:
            if tokens is not None:
                raise ValueError("only a hash router reads the token ids")
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
            # The router z-loss squares each position's log partition.
            self.sow('router', 'log_z', jax.nn.logsumexp(logits, axis=-1))
        weights = jnp.take_along_axis(scores, indices, axis=-1)
        if self.normalize_weights:
            weights = weights / (jnp.sum(weights, axis=-1, keepdims=True)
                                 + WEIGHT_SUM_EPSILON)
        return weights * self.routed_scaling_factor, indices

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
# the measured winner at lm-moe's shape (8192 rows, 768 -> 2048, 8 experts,
# bf16), numbers in docs/performance.md. On sm89 (L4, RTX 4080) that is
# JAX's own Pallas/Triton kernels, where XLA runs ragged_dot as a product over
# every expert; on a TPU v6e it is XLA's ragged_dot, within 5% of the best
# kernel measured there. Every generation not listed is unmeasured and runs
# 'xla'.
GROUPED_MATMUL_BY_GENERATION = {'sm89': 'pallas', 'v6e': 'xla'}

# The kernel 'tokamax' names, per generation. tokamax's own dispatch tries its
# Mosaic kernel first: on a TPU that is the v1 kernel, 13x slower than XLA on
# a v6e, and on sm89 a Mosaic GPU config that exceeds shared memory and
# raises. Unmeasured generations run tokamax's 'xla'.
TOKAMAX_KERNEL_BY_GENERATION = {'sm89': 'triton', 'v6e': 'mosaic_tpu_v2'}

# `device_kind` of the TPU generations Dew names.
TPU_GENERATIONS = {'TPU v4': 'v4', 'TPU v5 lite': 'v5e', 'TPU v5': 'v5p', 'TPU v5p': 'v5p',
                   'TPU v6 lite': 'v6e'}


def device_generation() -> str:
    """The default device's hardware generation, as the kernel tables key it:
    'sm89' for a GPU of compute capability 8.9, 'v6e' for a TPU v6e, and the
    backend's name for anything else."""
    device = jax.devices()[0]
    if device.platform == 'gpu' and getattr(device, 'compute_capability', None):
        return 'sm' + device.compute_capability.replace('.', '')
    if device.platform == 'tpu':
        return TPU_GENERATIONS.get(device.device_kind, device.device_kind)
    return device.platform


def resolve_grouped_matmul(implementation: str) -> str:
    """The grouped matmul `implementation` names on the default device:
    itself, or its generation's measured one for 'auto'."""
    if implementation not in GROUPED_MATMULS:
        raise ValueError(
            f"implementation must be one of {list(GROUPED_MATMULS)}, got "
            f"{implementation!r}")
    if implementation != 'auto':
        return implementation
    return GROUPED_MATMUL_BY_GENERATION.get(device_generation(), 'xla')


def grouped_matmul(tokens: jax.Array, kernel: jax.Array, group_sizes: jax.Array, *,
                   implementation: str, precision: PrecisionLike = None,
                   preferred_element_type: Dtype | None = None) -> jax.Array:
    """Each row of `tokens`, `[rows, in]`, through the matrix of its expert in
    `kernel`, `[exp, in, out]`, for rows already sorted by expert.

    `group_sizes` is how many leading rows belong to expert 0, then to expert
    1, and so on, the form every grouped matmul takes. `implementation` picks
    between them, the seam `dew.nn.attention.scaled_dot_product_attention`
    has:

    - 'auto': the hardware generation's measured one,
      `GROUPED_MATMUL_BY_GENERATION`, and 'xla' on an unmeasured one.
    - 'xla': `jax.lax.ragged_dot`. On a GPU XLA lowers it to a product over
      every expert.
    - 'pallas': JAX's own Pallas/Triton `gmm` kernel, vendored in
      `dew.nn.kernels.ragged_dot`, where `pallas_runs` says it computes the
      product asked for and no mesh axis splits the call, and 'xla'
      elsewhere. On a CPU the kernel runs in
      the Pallas interpreter, which is how the CPU suite checks it; on any
      other backend 'pallas' is 'xla'.
    - 'tokamax': `tokamax.ragged_dot`, the same call against tokamax's own
      kernels (`maxtext layers/moe.py:1633`), the kernel named per
      generation by `TOKAMAX_KERNEL_BY_GENERATION` and XLA elsewhere.

    This is the raw kernel call. 'pallas' has no differentiation rule here;
    `expert_projection` adds the precision contract routed experts train
    under, and the gradients of every implementation.
    """
    implementation = resolve_grouped_matmul(implementation)
    if implementation == 'tokamax':
        # tokamax is not a dependency (docs/concepts/moe.md), so it is
        # imported at the call.
        tokamax = importlib.import_module('tokamax')
        return tokamax.ragged_dot(
            tokens, kernel, group_sizes, precision=precision,
            preferred_element_type=preferred_element_type,
            implementation=TOKAMAX_KERNEL_BY_GENERATION.get(device_generation(), 'xla'))
    if (implementation == 'pallas' and pallas_runs(tokens.dtype, kernel.dtype, precision)
            and _row_axes(tokens.shape[0]) == ()):
        return _gmm(tokens, kernel, group_sizes,
                    preferred_element_type or jnp.result_type(tokens, kernel))
    return jax.lax.ragged_dot(
        tokens, kernel, group_sizes, precision=precision,
        preferred_element_type=preferred_element_type)


def pallas_runs(lhs: Dtype, rhs: Dtype, precision: PrecisionLike) -> bool:
    """Whether the Pallas kernels compute the product asked for, at trace time.

    They multiply in the operands' promoted dtype, accumulate in fp32 and
    ignore `precision`. With 16-bit operands that is exact products summed in
    fp32, which is what any precision asks for. With fp32 operands it is
    TF32 on a GPU, which only the default precision asks for (explicitly or
    through `jax_default_matmul_precision`), and float64 they do not run.
    They are Triton kernels: a GPU of compute capability 8.0 or later runs
    them (JAX 0.11.2 deprecates the backend; see `TRITON_DEPRECATION`), and
    a CPU interprets them. Under
    `jax_enable_x64` their group offsets widen to int64 against int32 block
    indices (the vendored cumsum), so an x64 run is 'xla''s.
    """
    if jax.config.jax_enable_x64:
        return False
    backend = jax.default_backend()
    if backend == 'gpu':
        if tuple(int(part) for part in jax.devices()[0].compute_capability.split('.')) < (8, 0):
            return False
    elif backend != 'cpu':
        return False
    compute = jnp.promote_types(lhs, rhs)
    if compute not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
        if compute != jnp.dtype(jnp.float32):
            return False
        asked = precision_names(precision) or precision_names(
            jax.config.jax_default_matmul_precision)
        if asked - {'DEFAULT', 'BFLOAT16', 'FASTEST', 'TENSORFLOAT32'}:
            return False
    return True


def _row_axes(rows: int) -> tuple[str, ...] | None:
    """The mesh axes a `shard_map` splits the sorted rows over for the
    kernels, which see local arrays: every axis not already manual with more
    than one device. None when the rows do not divide among them."""
    mesh = jax.sharding.get_abstract_mesh()
    axes = tuple(name for name in mesh.axis_names
                 if name not in mesh.manual_axes and mesh.shape[name] > 1)
    count = math.prod(mesh.shape[name] for name in axes)
    return axes if rows % count == 0 else None


# jax 0.11.2 deprecates the Pallas Triton backend: every Triton pallas_call
# warns, at lowering, that it will be removed in favour of Mosaic GPU. The
# grouped matmul keeps these kernels on purpose. The Mosaic GPU grouped
# matmul JAX ships (`pallas/ops/gpu/ragged_dot_mgpu.py`) uses wgmma, which
# sm_80 and sm_89 do not have (it fails to compile on an RTX 4080), and
# tokamax's sm80 Mosaic config exceeds an Ada card's shared memory and has no
# backward. The warning is Dew's to carry until a Mosaic GPU grouped matmul
# replaces these kernels on sm_90 and later; it is filtered by its exact
# text once Dew first uses the kernels (`_filter_triton_deprecation`).
TRITON_DEPRECATION = (r"The Pallas Triton backend is deprecated and will be removed in"
                      r" a future JAX version\.")


@functools.cache
def _filter_triton_deprecation() -> None:
    """Ignore `TRITON_DEPRECATION`, once the grouped matmul first uses the
    kernels: only that message, only as a DeprecationWarning.

    JAX raises it when a pallas_call is lowered, which is when the jit
    around the whole step compiles, after Dew's call has returned and with no
    Dew frame on the stack. A filter scoped to the call site cannot see it,
    so this one lasts for the process, and a process that never runs the
    kernels never installs it."""
    warnings.filterwarnings('ignore', message=TRITON_DEPRECATION, category=DeprecationWarning)


def _gmm(tokens, kernel, group_sizes, out_dtype, *, trans_rhs: bool = False):
    from .kernels import ragged_dot
    compute = jnp.promote_types(tokens.dtype, kernel.dtype)
    _filter_triton_deprecation()
    return ragged_dot.gmm(
        tokens.astype(compute), kernel.astype(compute), group_sizes.astype(jnp.int32),
        **ragged_dot.block_sizes(compute), trans_rhs=trans_rhs,
        interpret=jax.default_backend() != 'gpu', compute_dtype=compute,
        out_dtype=jnp.dtype(out_dtype))


def _tgmm(tokens, cotangent, group_sizes, out_dtype):
    from .kernels import ragged_dot
    compute = jnp.promote_types(tokens.dtype, cotangent.dtype)
    _filter_triton_deprecation()
    return ragged_dot.tgmm(
        tokens.astype(compute), cotangent.astype(compute), group_sizes.astype(jnp.int32),
        **ragged_dot.block_sizes(compute), interpret=jax.default_backend() != 'gpu',
        compute_dtype=compute, out_dtype=jnp.dtype(out_dtype))


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
    first-order reverse mode, the way MaxText wires megablox: a custom VJP
    whose forward is `gmm`, whose input gradient is `gmm` against the
    transposed kernel and whose kernel gradient is `tgmm`. Its compute
    dtype's rounding makes the backward exact products summed in fp32 when
    the compute dtype is 16-bit. Where `pallas_runs` says the kernels would
    change the product, 'pallas' is 'xla'. Under a mesh, 'pallas' splits the
    sorted rows over every axis that is not already manual
    (`_sharded_pallas_projection`), so the kernels see local arrays. Measured against NumPy float64
    sums of the rounded operands and three Adam steps of the global path on
    every expert/fsdp layout in tests/test_moe_precision.py.
    """
    compute = canonicalize_dtype(x, kernel, dtype=dtype)
    axes = _row_axes(x.shape[0])
    # The kernels accumulate in fp32, so a float64 operand, whose gradient
    # the contract sums in float64, is 'xla''s.
    wide = jnp.dtype(jnp.float64) in (jnp.dtype(x.dtype), jnp.dtype(kernel.dtype))
    if (resolve_grouped_matmul(implementation) == 'pallas' and not wide
            and pallas_runs(compute, compute, precision) and axes is not None):
        return _sharded_pallas_projection(x, kernel, group_sizes, compute, axes)
    if implementation == 'pallas':
        implementation = 'xla'
    return _projection(x, kernel, group_sizes, dtype, implementation, precision)


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
                    ) -> tuple[jax.Array, jax.Array]:
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
    tangent = jax.lax.optimization_barrier(sum(terms[1:], terms[0]).astype(output.dtype))
    return output, tangent


_projection.defjvp(_projection_jvp, symbolic_zeros=True)


def _sharded_pallas_projection(x, kernel, group_sizes, compute: Dtype,
                               axes: tuple[str, ...]) -> jax.Array:
    """`_pallas_projection` with the sorted rows split over `axes`.

    Each shard multiplies its contiguous block of rows, with the group sizes
    cut to that block, against the whole kernel. The kernel enters at least
    fp32, replicated, so shard_map's transpose sums the per-shard kernel
    gradients in fp32 and the master dtype rounds once, after the sum.
    """
    if not axes:
        return _pallas_projection(x, kernel, group_sizes, compute,
                                  (_varying_axes(x), _varying_axes(kernel)))
    work = kernel.astype(jnp.promote_types(kernel.dtype, jnp.float32))
    rows = x.shape[0] // math.prod(jax.sharding.get_abstract_mesh().shape[name] for name in axes)

    def local(x, kernel, group_sizes):
        start = jax.lax.axis_index(axes).astype(jnp.int32) * rows
        ends = jnp.cumsum(group_sizes, dtype=jnp.int32)
        sizes = jnp.clip(jnp.minimum(ends, start + rows) - jnp.maximum(ends - group_sizes, start), 0)
        return _pallas_projection(x, kernel, sizes.astype(group_sizes.dtype), compute,
                                  (_varying_axes(x), _varying_axes(kernel)))

    spread, whole = P(axes), P()
    mesh = jax.sharding.get_abstract_mesh()
    explicit = {name for name, kind in zip(mesh.axis_names, mesh.axis_types, strict=True)
                if kind == jax.sharding.AxisType.Explicit}
    if explicit:
        # shard_map takes explicit-axis operands only as its specs place
        # them; automatic axes it places itself.
        x = jax.sharding.reshard(x, P(tuple(name for name in axes if name in explicit)))
        work, group_sizes = jax.sharding.reshard((work, group_sizes), whole)
    # Pallas outputs carry no varying-axes type, so the check is off. Its
    # transpose then sums the replicated kernel's per-shard cotangents.
    return jax.shard_map(local, in_specs=(spread, whole, whole), out_specs=spread,
                         axis_names=set(axes), check_vma=False)(x, work, group_sizes)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _pallas_projection(x: jax.Array, kernel: jax.Array, group_sizes: jax.Array,
                       dtype: Dtype | None,
                       varying: tuple[tuple[str, ...], tuple[str, ...]]) -> jax.Array:
    """`expert_projection` on the Pallas kernels, first-order reverse mode.
    `varying` is the manual mesh axes `x` and `kernel` vary over."""
    return _pallas_projection_fwd(x, kernel, group_sizes, dtype, varying)[0]


def _pallas_projection_fwd(x, kernel, group_sizes, dtype, varying):
    del varying
    inputs, matrix = promote_dtype(x, kernel, dtype=dtype)
    output = _gmm(inputs, matrix, group_sizes, jnp.promote_types(inputs.dtype, jnp.float32))
    # The residuals are the rounded operands, the values the forward
    # multiplied; the dtypes of the originals are what the gradients take.
    residuals = (inputs, matrix, group_sizes, jnp.zeros((0,), x.dtype),
                 jnp.zeros((0,), kernel.dtype))
    return output.astype(inputs.dtype), residuals


def _pallas_projection_bwd(dtype, varying, residuals, cotangent):
    del dtype
    inputs, matrix, group_sizes, x_like, kernel_like = residuals
    work = jnp.result_type(inputs.dtype, x_like.dtype, kernel_like.dtype, jnp.float32)
    # The cotangent is in the compute dtype, so with 16-bit compute both
    # products multiply values exact in it and sum in fp32: the work-dtype
    # contraction of the contract, without widening the operands.
    cotangent = cotangent.astype(inputs.dtype)
    d_inputs = _gmm(cotangent, matrix, group_sizes, work, trans_rhs=True)
    d_matrix = _tgmm(inputs, cotangent, group_sizes, work)
    # A Pallas output carries no manual-axis type, and inside a shard_map a
    # cotangent has to vary over the axes its primal varies over.
    x_axes, kernel_axes = varying
    return (_varying(d_inputs.astype(x_like.dtype), x_axes),
            _varying(d_matrix.astype(kernel_like.dtype), kernel_axes),
            np.zeros(group_sizes.shape, jax.dtypes.float0))


def _varying(value: jax.Array, axes: tuple[str, ...]) -> jax.Array:
    return jax.lax.pcast(value, axes, to='varying') if axes else value


def _varying_axes(value: jax.Array) -> tuple[str, ...]:
    return tuple(sorted(jax.typeof(value).mat.varying))



_pallas_projection.defvjp(_pallas_projection_fwd, _pallas_projection_bwd)


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
    """The product a gated MLP takes of its gate and up projections: silu
    ('swiglu'), the tanh gelu ('geglu') or the erf gelu rounded once from
    fp32 ('geglu_exact', `exact_gelu`) on the gate times up, or a `Situ`
    over both halves."""
    if isinstance(activation, Situ):
        return activation
    gates = {'swiglu': nn.silu, 'geglu': functools.partial(nn.gelu, approximate=True), 'geglu_exact': exact_gelu}
    if activation not in gates:
        raise ValueError(f"mlp must be 'swiglu', 'geglu', 'geglu_exact' or a Situ, got {activation!r}")
    activate = gates[activation]
    return lambda gate, up: activate(gate) * up


class ExpertLinear(nn.Module):
    """One matrix per expert, `[exp, in_features, features]`, over tokens
    already sorted by expert, through `expert_projection` on `implementation`."""
    num_experts: int
    in_features: int
    features: int
    init_std: float | None = None  # normal std of every expert; None: per-expert lecun normal
    implementation: str = 'auto'
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


def expert_dispatch[Parameters](
        project: Callable[[jax.Array, jax.Array, jax.Array, Parameters], jax.Array],
        x: jax.Array, indices: jax.Array, parameters: Parameters, *,
        num_experts: int, dispatch: str, output_dtype: Dtype,
        input_weights: jax.Array | None = None, initializing: bool = False) -> jax.Array:
    """Run every routed token through its expert, and return the slots in
    token order.

    `project` takes rows sorted by expert, the group sizes, the sorted
    expert ids and the expert-major parameters, and returns rows of the same
    width. The caller owns the activation, the biases and the output
    weights; `input_weights` scales each expert's input instead, as Llama 4
    does.

    `dispatch='global'` sorts and gathers locally. `'exchange'` runs `shards`
    rounds of all-to-all over the expert mesh axis, each carrying one
    capacity-sized buffer per shard, so no expert capacity drops tokens.
    Expert ids are local to their owner under exchange, and the padding that
    fills a round is discarded before the return exchange.
    """
    if dispatch not in EXPERT_DISPATCHES:
        raise ValueError(f"dispatch must be one of {EXPERT_DISPATCHES}, got {dispatch!r}")
    if indices.shape[:-1] != x.shape[:-1] or (
            input_weights is not None and input_weights.shape != indices.shape):
        raise ValueError(f"routing {indices.shape} does not describe tokens {x.shape}")
    tokens = x.reshape(-1, x.shape[-1])
    top_k = indices.shape[-1]
    mesh = jax.sharding.get_abstract_mesh()
    shards = mesh.shape.get(EXPERT_AXIS, 1)
    if dispatch == 'exchange' and not initializing and (
            shards <= 1 or num_experts % shards):
        raise ValueError("exchange dispatch needs an expert mesh axis greater than one "
                         "that divides num_experts")
    if dispatch == 'global' or initializing or not tokens.shape[0]:
        experts = indices.ravel()
        order = jnp.argsort(experts)
        grouped = tokens[order // top_k]
        if input_weights is not None:
            grouped = grouped * input_weights.ravel()[order][:, None].astype(grouped.dtype)
        projected = project(grouped, jnp.bincount(experts, length=num_experts),
                            experts[order], parameters)
        return projected[jnp.argsort(order)].reshape(*indices.shape, x.shape[-1])

    @functools.partial(jax.shard_map, mesh=mesh, axis_names={EXPERT_AXIS},
                       in_specs=(P(EXPERT_AXIS), P(EXPERT_AXIS), P(EXPERT_AXIS),
                                 None if input_weights is None else P(EXPERT_AXIS)),
                       out_specs=P(EXPERT_AXIS))
    def local(tokens: jax.Array, indices: jax.Array, parameters: Parameters,
              input_weights: jax.Array | None) -> jax.Array:
        """Run one expert shard's share of the exchange, inside shard_map.

        Every shard sorts its own rows by expert, cuts them into buckets of
        `capacity` per destination, and runs `rounds` exchanges. `rounds` is
        the mesh-wide maximum, so every shard runs the same number.
        """
        slots, per_shard = indices.size, num_experts // shards
        order = jnp.argsort(indices.ravel())
        experts = indices.ravel()[order]
        grouped = tokens[order // top_k]
        if input_weights is not None:
            grouped = grouped * input_weights.ravel()[order][:, None].astype(grouped.dtype)
        sizes = jnp.bincount(experts // per_shard, length=shards + 1)[:shards]
        starts = jnp.cumsum(sizes) - sizes
        capacity = (slots + shards - 1) // shards
        rounds = jax.lax.pmax(jnp.max((sizes + capacity - 1) // capacity), EXPERT_AXIS)
        lanes = jnp.arange(capacity)

        @jax.checkpoint
        def exchange_round(iteration: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Send one bucket to each shard, project it there, bring it back.

            Returns the returned rows and the local address each one belongs
            at. Padding addresses land past the end, so the caller's
            scatter drops them.
            """
            offsets = iteration * capacity + lanes
            valid = offsets[None, :] < sizes[:, None]
            addresses = starts[:, None] + offsets
            send = jnp.where(valid[..., None], grouped[jnp.minimum(addresses, slots - 1)], 0)
            ids = jnp.where(valid, experts[jnp.minimum(addresses, slots - 1)] % per_shard,
                            per_shard)
            received = jax.lax.all_to_all(send, EXPERT_AXIS, 0, 0, tiled=True).reshape(
                -1, tokens.shape[-1])
            received_ids = jax.lax.all_to_all(ids, EXPERT_AXIS, 0, 0, tiled=True).ravel()
            permutation = jnp.argsort(received_ids)
            groups = jnp.bincount(received_ids, length=per_shard + 1)[:per_shard]
            computed = project(received[permutation], groups, received_ids[permutation], parameters)[
                jnp.argsort(permutation)]
            computed = jnp.where((received_ids < per_shard)[:, None], computed, 0)
            returned = jax.lax.all_to_all(computed.reshape(shards, capacity, -1),
                                         EXPERT_AXIS, 0, 0, tiled=True)
            # Only padding has an out-of-bounds address; every real slot has
            # one unique writer across all rounds.
            return returned, jnp.where(valid, addresses, slots)

        def step(out: jax.Array, iteration: jax.Array) -> tuple[jax.Array, None]:
            """Scan one round into the output rows; the carry is those rows.

            The scan always runs `shards` iterations so its length is
            static, and the rounds past `rounds` keep the carry unchanged.
            """
            def active(out: jax.Array) -> jax.Array:
                returned, addresses = exchange_round(iteration)
                return out.at[addresses].set(returned, mode='drop')
            return jax.lax.cond(iteration < rounds, active, lambda out: out, out), None

        initial = jax.lax.pcast(jnp.zeros((slots, x.shape[-1]), output_dtype),
                                EXPERT_AXIS, to='varying')
        combined, _ = jax.lax.scan(step, initial, jnp.arange(shards))
        return combined[jnp.argsort(order)].reshape(*indices.shape, x.shape[-1])

    # Sentinel assignments from token-axis padding never enter send counts.
    padding = -tokens.shape[0] % shards
    token_indices = jnp.pad(indices.reshape(-1, top_k), ((0, padding), (0, 0)),
                            constant_values=num_experts)
    padded_weights = (None if input_weights is None else
                      jnp.pad(input_weights.reshape(-1, top_k), ((0, padding), (0, 0))))
    combined = local(jnp.pad(tokens, ((0, padding), (0, 0))), token_indices, parameters, padded_weights)
    return combined[:tokens.shape[0]].reshape(*indices.shape, x.shape[-1])



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
        slots = expert_dispatch(
            self._project, x, indices, kernels, num_experts=self.num_experts,
            dispatch=self.dispatch, initializing=self.is_initializing(),
            output_dtype=canonicalize_dtype(x, *kernels, dtype=self.dtype),
            input_weights=weights if self.scale_inputs else None)
        return self._combine(slots, weights)


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
                           hash_vocab=self.hash_vocab, init_std=self.init_std,
                           precision=self.precision, name='gate')
        width = self.out_features if self.latent_features is None else self.latent_features
        self.experts = ExpertMLP(
            num_experts=self.num_experts, hidden_features=self.hidden_features,
            out_features=width, activation=self.activation,
            implementation=self.implementation, dispatch=self.dispatch,
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

    def __call__(self, x, tokens=None):
        weights, indices = self.gate(x, tokens)
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
