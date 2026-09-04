"""Mixture of experts: the router, and the experts as one grouped matmul.

Patterned on MaxText's `RoutedMoE` (`maxtext src/maxtext/layers/moe.py:419`,
Apache 2.0), which is 3731 lines of NNX reading about forty fields off a config
object. What ports is the math: top-k selection (`:751` `get_topk`), DeepSeek's
group-limited routing, which selects on the biased scores and gates on the
unbiased ones (`:881-908` `deepseek_routing`), its weight scaling
(`:835-841`), the aux-loss-free bias update (`:238-261`, lifted below), and the
grouped matmul over tokens sorted by expert (`:1500` `sparse_matmul`).

The two routers this reproduces are `MixtralSparseMoeBlock`, which softmaxes
over the experts, takes the top k and renormalises, and `DeepseekV3MoE`, which
scores with a sigmoid, selects with a per-expert bias added, limits the choice
to the best expert groups, renormalises and scales. Both are transformers
5.16.1; `tests/test_moe.py` holds the fp32 parity numbers and
`tools/moe_reference.py` writes the fixtures they run against.

Parameters follow the Hugging Face layout of a sparse decoder layer, with the
experts stacked on an expert dimension: `mlp/gate/kernel` is `[embed, exp]`,
`mlp/experts/gate_proj/kernel` and `mlp/experts/up_proj/kernel` are `[exp,
embed, mlp]`, and `mlp/experts/down_proj/kernel` is `[exp, mlp, embed]`. A
checkpoint's per-expert `mlp.experts.N.gate_proj.weight` tensors stack into one
leaf, which is the layout `jax.lax.ragged_dot` takes and the `expert` mesh axis
shards.
"""

import functools
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike

from .sharding import logical_axes

# 'softmax' normalizes a token's affinities over the experts (Mixtral,
# Qwen3.5); 'sigmoid' scores each expert on its own (DeepSeek V3 and V4, GLM,
# Kimi, LLaDA2).
SCORE_FUNCTIONS = ('softmax', 'sigmoid')

GROUPED_MATMULS = ('xla', 'tokamax')

# DeepSeek divides the selected weights by their sum plus this, so a token
# whose sigmoid scores are all zero stays finite
# (modeling_deepseek_v3.py:166, maxtext layers/moe.py:839). A softmax over the
# experts sums to one, which cannot reach it, and in fp32 the term changes no
# bit of a denominator above 1e-12.
WEIGHT_SUM_EPSILON = 1e-20


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
    total_tokens = jnp.sum(expert_counts)
    average_load = total_tokens / num_experts
    direction = jnp.sign(average_load - expert_counts)
    output = direction * rate
    return output


class Router(nn.Module):
    """Which experts a token goes to, and with what weight: `[..., k]` of each.

    The gate projection runs in fp32 whatever dtype the activations carry,
    which is where DeepSeek's router runs
    (`modeling_deepseek_v3.py:146`) and what every frontier config asks for.

    `expert_bias` is DeepSeek's aux-loss-free balancing bias
    (`e_score_correction_bias`, arXiv 2408.15664), kept in fp32 in the `moe`
    collection. It enters the selection and nothing else: a token's weights are
    gathered from the unbiased scores, so moving the bias changes which experts
    a token uses without changing what they contribute. Nothing here writes it,
    which is also true of the reference: transformers holds it in an
    `nn.Buffer` and MaxText hands the update back to its caller
    (`layers/moe.py:965-972`). `calculate_load_balance_updates` is that update,
    and a step that applies it owns the write. Gradients cannot reach the bias
    either, since it feeds `jax.lax.top_k`'s integer indices and nothing else.

    `expert_groups` above one is DeepSeek's node limit: experts are cut into
    that many groups, each group is scored by its two best experts, and a token
    may only choose inside the best `groups_per_token` of them.
    """
    num_experts: int
    in_features: int
    top_k: int
    score_function: str = 'softmax'
    normalize_weights: bool = True
    routed_scaling_factor: float = 1.0
    expert_groups: int = 1
    groups_per_token: int = 1
    expert_bias: bool = False
    precision: PrecisionLike = None

    def setup(self):
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
        if self.expert_groups > 1:
            per_group = self.num_experts // self.expert_groups
            if per_group < 2:
                raise ValueError(
                    "group scores are the sum of a group's two best experts, so "
                    f"a group needs at least two: {self.num_experts} experts in "
                    f"{self.expert_groups} groups leaves {per_group}")
            if self.groups_per_token * per_group < self.top_k:
                raise ValueError(
                    f"{self.groups_per_token} of {self.expert_groups} groups hold "
                    f"{self.groups_per_token * per_group} experts, fewer than the "
                    f"top_k of {self.top_k}")
        # The kernel is the router's own parameter rather than a nested
        # nn.Dense, so the leaf is `gate/kernel` where a Hugging Face sparse
        # layer keeps `gate.weight`.
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
        best_two, _ = jax.lax.top_k(grouped, 2)
        _, groups = jax.lax.top_k(jnp.sum(best_two, axis=-1), self.groups_per_token)
        kept = jnp.sum(
            jax.nn.one_hot(groups, self.expert_groups, dtype=jnp.float32), axis=-2)
        return jnp.repeat(kept > 0, per_group, axis=-1)


class ExpertLinear(nn.Module):
    """One matrix per expert, `[exp, in_features, features]`, over tokens
    already sorted by expert.

    `group_sizes` is how many leading rows belong to expert 0, then to expert
    1, and so on, which is what both grouped matmuls take.
    `implementation` picks between them, the seam
    `dew.nn.attention.scaled_dot_product_attention` has:

    - 'xla': `jax.lax.ragged_dot`, which lowers on every backend.
    - 'tokamax': `tokamax.ragged_dot`, the same call against tokamax's own
      kernels (`maxtext layers/moe.py:1633`). tokamax is not a dependency of
      Dew, so its import is inside the branch that needs it.
    """
    num_experts: int
    in_features: int
    features: int
    implementation: str = 'xla'
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        if self.implementation not in GROUPED_MATMULS:
            raise ValueError(
                f"implementation must be one of {list(GROUPED_MATMULS)}, got "
                f"{self.implementation!r}")
        # fan_in per expert, not over the stack: with the expert dimension as a
        # batch axis every expert initialises exactly like the matching
        # nn.Dense of a dense MLP.
        self.kernel = self.param(
            'kernel',
            nn.initializers.variance_scaling(
                1.0, 'fan_in', 'truncated_normal', in_axis=-2, out_axis=-1,
                batch_axis=(0,)),
            (self.num_experts, self.in_features, self.features), jnp.float32)

    def __call__(self, tokens, group_sizes):
        tokens, kernel = promote_dtype(tokens, self.kernel, dtype=self.dtype)
        if self.implementation == 'tokamax':
            import tokamax
            return tokamax.ragged_dot(
                tokens, kernel, group_sizes, precision=self.precision,
                preferred_element_type=tokens.dtype)
        return jax.lax.ragged_dot(
            tokens, kernel, group_sizes, precision=self.precision,
            preferred_element_type=tokens.dtype)


class ExpertMLP(nn.Module):
    """The routed experts of one layer: each token through the gated MLPs its
    router chose.

    Tokens are gathered into expert order, the three projections run as grouped
    matmuls over that order, and the results go back to token order and are
    summed with their router weights (`maxtext layers/moe.py:940` `permute` and
    `:1101` `unpermute`). The gather reads token rows directly rather than a
    `top_k`-fold copy of them, which is MaxText's `moe_use_direct_token_gather`.

    The sum over a token's experts runs in fp32, because that is the dtype the
    router weights are computed in, and the result rejoins the residual stream
    in the compute dtype.
    """
    num_experts: int
    hidden_features: int
    out_features: int
    activation: str = 'swiglu'
    implementation: str = 'xla'
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        if self.activation not in ('swiglu', 'geglu'):
            raise ValueError(
                f"mlp must be 'swiglu' or 'geglu', got {self.activation!r}")
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

    def __call__(self, x, weights, indices):
        """`x` is `[..., embed]` and `weights` and `indices` are `[..., k]`."""
        if weights.shape != indices.shape or indices.shape[:-1] != x.shape[:-1]:
            raise ValueError(
                f"routing {indices.shape} does not describe tokens {x.shape}")
        top_k = indices.shape[-1]
        tokens = x.reshape(-1, x.shape[-1])
        experts = indices.reshape(-1)
        # Slot j of the flattened routing belongs to token j // top_k, so one
        # gather puts the token rows in expert order.
        order = jnp.argsort(experts)
        group_sizes = jnp.bincount(experts, length=self.num_experts)
        grouped = tokens[order // top_k]

        gate = self.gate_proj(grouped, group_sizes)
        gate = nn.silu(gate) if self.activation == 'swiglu' else nn.gelu(gate)
        expert_out = self.down_proj(gate * self.up_proj(grouped, group_sizes),
                                   group_sizes)

        per_slot = expert_out[jnp.argsort(order)].reshape(*indices.shape, -1)
        combined = jnp.einsum(
            '...ke,...k->...e', per_slot.astype(jnp.float32),
            weights.astype(jnp.float32), precision=self.precision)
        return combined.astype(expert_out.dtype)


@logical_axes({
    ("gate",): ("embed", "exp"),
    # A sparse layer's experts are stacked on one leaf, so the expert
    # dimension is named here and the longer path wins over the dense
    # projection of the same name.
    ("experts", "gate_proj"): ("exp", "embed", "mlp"),
    ("experts", "up_proj"): ("exp", "embed", "mlp"),
    ("experts", "down_proj"): ("exp", "mlp", "embed"),
})
class SparseMLP(nn.Module):
    """A router over `num_experts` gated MLPs, `top_k` of them per token.

    Goes where `GatedMLP` goes in a decoder block and holds the two submodules
    a Hugging Face sparse layer names, `gate` for the router and `experts` for
    the stacked expert weights.
    """
    num_experts: int
    top_k: int
    hidden_features: int
    out_features: int
    activation: str = 'swiglu'
    implementation: str = 'xla'
    score_function: str = 'softmax'
    routed_scaling_factor: float = 1.0
    expert_groups: int = 1
    groups_per_token: int = 1
    expert_bias: bool = False
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        self.gate = Router(num_experts=self.num_experts,
                           in_features=self.out_features, top_k=self.top_k,
                           score_function=self.score_function,
                           routed_scaling_factor=self.routed_scaling_factor,
                           expert_groups=self.expert_groups,
                           groups_per_token=self.groups_per_token,
                           expert_bias=self.expert_bias,
                           precision=self.precision, name='gate')
        self.experts = ExpertMLP(
            num_experts=self.num_experts, hidden_features=self.hidden_features,
            out_features=self.out_features, activation=self.activation,
            implementation=self.implementation, dtype=self.dtype,
            precision=self.precision, name='experts')

    def __call__(self, x):
        weights, indices = self.gate(x)
        return self.experts(x, weights, indices)
