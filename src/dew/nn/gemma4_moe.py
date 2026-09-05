"""Gemma 4's routed experts beside the dense MLP (the 26B-A4B size).

`Gemma4TextDecoderLayer` with `enable_moe_block` (modeling_gemma4.py) runs
the dense MLP and a routed branch on the same residual and sums them after a
norm each: `post_feedforward_layernorm_1(mlp(x))` plus
`post_feedforward_layernorm_2(experts(pre_feedforward_layernorm_2(x)))`,
where the router reads the raw residual. `Gemma4TextRouter` norms that
residual without a scale, multiplies by its own scale and `1 / sqrt(hidden)`,
projects, softmaxes in fp32, keeps the top k renormalised to one and scales
each choice by `per_expert_scale`. The experts are the plain gated MLPs of
`dew.nn.moe.ExpertMLP`, weighted on their outputs.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import RMSNorm
from dew.nn.moe import ExpertMLP
from dew.nn.sharding import logical_axes


@logical_axes({("proj",): ("embed", "exp")})
class Gemma4TextRouter(nn.Module):
    """Which experts a token takes and with what weight, `[..., k]` of each."""

    num_experts: int
    top_k: int
    norm_eps: float = 1e-6
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError(
                f"top_k selects among the experts, so it is between 1 and "
                f"{self.num_experts}, got {self.top_k}")
        hidden = x.shape[-1]
        normed = RMSNorm(epsilon=self.norm_eps, with_scale=False, dtype=self.dtype,
                         name='norm')(x)
        scale = self.param('scale', nn.initializers.ones, (hidden,), jnp.float32)
        scaled = normed * (scale * (hidden ** -0.5)).astype(normed.dtype)
        logits = nn.Dense(self.num_experts, use_bias=False, dtype=self.dtype,
                          precision=self.precision, name='proj')(scaled)
        probabilities = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        weights, indices = jax.lax.top_k(probabilities, self.top_k)
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
        per_expert = self.param('per_expert_scale', nn.initializers.ones,
                                (self.num_experts,), jnp.float32)
        return weights * per_expert[indices], indices


class Gemma4Experts(nn.Module):
    """The routed branch summed with the dense MLP's output.

    Takes the block's residual `x` and the dense MLP's output `mlp_out` and
    returns what `post_feedforward_layernorm` then norms; the three norms
    keep the names of what they normalise, as the block's own sandwich norms
    do, and the router and experts keep the reference's.
    """

    num_experts: int
    top_k: int
    hidden_features: int
    out_features: int
    activation: str = 'geglu'
    implementation: str = 'xla'
    norm_eps: float = 1e-6
    scale_offset: bool = False
    scale_after_cast: bool = False
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, mlp_out):
        def norm(name: str) -> RMSNorm:
            return RMSNorm(epsilon=self.norm_eps, scale_offset=self.scale_offset,
                           scale_after_cast=self.scale_after_cast, dtype=self.dtype, name=name)

        weights, indices = Gemma4TextRouter(
            num_experts=self.num_experts, top_k=self.top_k, norm_eps=self.norm_eps,
            dtype=self.dtype, precision=self.precision, name='router')(x)
        routed = ExpertMLP(
            num_experts=self.num_experts, hidden_features=self.hidden_features,
            out_features=self.out_features, activation=self.activation,
            implementation=self.implementation, dtype=self.dtype,
            precision=self.precision, name='experts')(norm('experts_input_norm')(x), weights, indices)
        return norm('mlp_branch_norm')(mlp_out) + norm('experts_output_norm')(routed)


