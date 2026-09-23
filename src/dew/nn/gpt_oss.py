"""GPT OSS's biased router and interleaved experts with the clamped SwiGLU.

`dew.interop.codecs` reads and writes the MXFP4 checkpoints these experts ship in.
"""

import functools
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.dtypes import canonicalize_dtype
from flax.typing import Dtype, PrecisionLike

from dew.nn.moe import Routes, chosen_experts, expert_dispatch, expert_projection, gather_expert_bias
from dew.nn.sharding import LogicalAxes, logical_axes

FUSED_EXPERT_AXES: Mapping[str, LogicalAxes] = {
    "gate_up_proj": ("exp", "embed", "mlp"),
    "gate_up_proj_bias": ("exp", "mlp"),
    "down_proj": ("exp", "mlp", "embed"),
    "down_proj_bias": ("exp", "embed"),
}
"""The fused expert leaves' axes, in the order `GptOssExperts` creates them."""


class GptOssExperts(nn.Module):
    """Interleaved gate/up matrices with the reference's clamped 1.702 SwiGLU.

    `implementation` names the grouped matmul, as `moe.expert_projection` takes it;
    `dispatch` and `capacity_factor` move the tokens, as `moe.expert_dispatch` takes them.
    """

    hidden_size: int
    intermediate_size: int
    num_local_experts: int
    implementation: str = 'auto'
    dispatch: str = 'global'
    capacity_factor: float | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x: jax.Array, weights: jax.Array, indices: jax.Array) -> jax.Array:
        initializer = nn.initializers.variance_scaling(
            1.0, "fan_in", "truncated_normal", in_axis=-2, out_axis=-1, batch_axis=(0,))
        gate_up = self.param("gate_up_proj", initializer,
                             (self.num_local_experts, self.hidden_size, 2 * self.intermediate_size))
        gate_bias = self.param("gate_up_proj_bias", nn.initializers.zeros,
                               (self.num_local_experts, 2 * self.intermediate_size))
        down = self.param("down_proj", initializer,
                          (self.num_local_experts, self.intermediate_size, self.hidden_size))
        down_bias = self.param("down_proj_bias", nn.initializers.zeros,
                               (self.num_local_experts, self.hidden_size))
        # Infer the shared compute dtype from every original operand, while
        # keeping master kernels uncast for the projection's derivative rule.
        compute_dtype = canonicalize_dtype(x, gate_up, gate_bias, down, down_bias, dtype=self.dtype)
        if weights.shape != indices.shape:
            raise ValueError(f"routing {indices.shape} does not describe weights {weights.shape}")
        slots = expert_dispatch(
            functools.partial(self._project, dtype=compute_dtype), x.astype(compute_dtype), indices,
            (gate_up, gate_bias, down, down_bias), tuple(FUSED_EXPERT_AXES.values()),
            num_experts=self.num_local_experts,
            dispatch=self.dispatch, output_dtype=compute_dtype, initializing=self.is_initializing(),
            capacity_factor=self.capacity_factor)
        return jnp.sum(slots * weights[..., None], axis=-2)

    def _project(self, tokens: jax.Array, sizes: jax.Array, expert_ids: jax.Array,
                 parameters: tuple[jax.Array, jax.Array, jax.Array, jax.Array], *,
                 dtype: Dtype) -> jax.Array:
        gate_up, gate_bias, down, down_bias = parameters
        projected = jnp.asarray(expert_projection(
            tokens, gate_up, sizes, dtype, self.implementation, self.precision))
        projected = projected + gather_expert_bias(gate_bias, expert_ids, dtype)
        gate = jnp.minimum(projected[..., ::2], 7.0)
        up = jnp.clip(projected[..., 1::2], -7.0, 7.0)
        activated = (up + 1) * (gate * jax.nn.sigmoid(gate * 1.702))
        output = jnp.asarray(expert_projection(
            activated, down, sizes, dtype, self.implementation, self.precision))
        return output + gather_expert_bias(down_bias, expert_ids, dtype)


@logical_axes({
    ("router",): ("embed", "exp"),
    **{("experts", name): axes for name, axes in FUSED_EXPERT_AXES.items()},
})
class GptOssMLP(nn.Module):
    """Softmax over the selected biased logits, then the selected expert sum.

    The experts keep the reference's fused leaves, stacked on the expert
    dimension the expert mesh axis splits, like `moe.SparseMLP`'s.
    """

    hidden_size: int
    intermediate_size: int
    num_local_experts: int
    num_experts_per_tok: int
    implementation: str = 'auto'
    dispatch: str = 'global'
    capacity_factor: float | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x: jax.Array, routes: Routes | None = None) -> jax.Array:
        logits = nn.Dense(self.num_local_experts, use_bias=True, dtype=self.dtype,
                           precision=self.precision, name="router")(x)
        k = self.num_experts_per_tok
        indices = chosen_experts(lambda: jax.lax.top_k(logits, k)[1], (*logits.shape[:-1], k), routes)
        top_logits = jnp.take_along_axis(logits, indices, axis=-1)
        weights = jax.nn.softmax(top_logits, axis=-1)
        return GptOssExperts(
            self.hidden_size, self.intermediate_size, self.num_local_experts,
            implementation=self.implementation, dispatch=self.dispatch,
            capacity_factor=self.capacity_factor, dtype=self.dtype,
            precision=self.precision, name="experts")(x, weights, indices)
