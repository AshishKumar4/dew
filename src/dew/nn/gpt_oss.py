"""GPT OSS's biased router, interleaved experts and MXFP4 checkpoint math."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike

from dew.nn.sharding import logical_axes


def dequantize_mxfp4(blocks: jax.Array, scales: jax.Array) -> jax.Array:
    """Packed [expert, output, group, 16] and E8M0 scales to bf16 [expert, input, output].

    The low nibble precedes the high nibble, and each scale covers 32 E2M1
    values. The final transpose is the released GPT OSS expert layout, as in
    transformers.integrations.mxfp4.convert_moe_packed_tensors.
    """
    if blocks.ndim != 4 or blocks.shape[-1] != 16 or blocks.shape[:-1] != scales.shape:
        raise ValueError("MXFP4 blocks must be [expert, output, group, 16] with one scale per group")
    if blocks.dtype != jnp.uint8 or scales.dtype != jnp.uint8:
        raise ValueError("MXFP4 blocks and scales must be uint8")
    lookup = jnp.asarray([0, 0.5, 1, 1.5, 2, 3, 4, 6,
                          -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], jnp.bfloat16)
    indices = jnp.stack((blocks & 15, blocks >> 4), axis=-1)
    unpacked = lookup[indices].reshape(*blocks.shape[:-1], 32)
    values = jnp.ldexp(unpacked, scales.astype(jnp.int32)[..., None] - 127)
    return values.reshape(*blocks.shape[:2], -1).swapaxes(1, 2).astype(jnp.bfloat16)


def unpack_mxfp4(tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Replace each `<name>_blocks` and `<name>_scales` pair with `<name>` in fp32.

    fp32 holds every bf16 value exactly, so the unpacked tensor is the one
    transformers materializes before its own forward pass.
    """
    unpacked = dict(tensors)
    for name in [name for name in tensors if name.endswith('_blocks')]:
        stem = name.removesuffix('_blocks')
        blocks, scales = unpacked.pop(name), unpacked.pop(stem + '_scales')
        unpacked[stem] = np.asarray(
            dequantize_mxfp4(jnp.asarray(blocks), jnp.asarray(scales)).astype(jnp.float32))
    return unpacked


class GptOssExperts(nn.Module):
    """Interleaved gate/up matrices with the reference's clamped 1.702 SwiGLU."""

    hidden_size: int
    intermediate_size: int
    num_local_experts: int
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
        x, gate_up, gate_bias, down, down_bias = promote_dtype(
            x, gate_up, gate_bias, down, down_bias, dtype=self.dtype)
        experts = indices.reshape(-1)
        order = jnp.argsort(experts)
        sorted_experts = experts[order]
        group_sizes = jnp.bincount(experts, length=self.num_local_experts)
        tokens = x.reshape(-1, self.hidden_size)[order // indices.shape[-1]]
        projected = jax.lax.ragged_dot(tokens, gate_up, group_sizes,
                                       precision=self.precision) + gate_bias[sorted_experts]
        gate = jnp.minimum(projected[..., ::2], 7.0)
        up = jnp.clip(projected[..., 1::2], -7.0, 7.0)
        activated = (up + 1) * (gate * jax.nn.sigmoid(gate * 1.702))
        output = jax.lax.ragged_dot(activated, down, group_sizes,
                                    precision=self.precision) + down_bias[sorted_experts]
        per_slot = output[jnp.argsort(order)].reshape(*indices.shape, self.hidden_size)
        return jnp.sum(per_slot * weights[..., None], axis=-2)


@logical_axes({("router",): ("embed", "exp")}, heuristic=(
    ("experts", "gate_up_proj"), ("experts", "gate_up_proj_bias"),
    ("experts", "down_proj"), ("experts", "down_proj_bias")))
class GptOssMLP(nn.Module):
    """Softmax over the selected biased logits, then the selected expert sum.

    The experts retain the reference's fused parameter leaves. Those leaves
    have different matrix axes, so their placement uses the shape heuristic.
    """

    hidden_size: int
    intermediate_size: int
    num_local_experts: int
    num_experts_per_tok: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        logits = nn.Dense(self.num_local_experts, use_bias=True, dtype=self.dtype,
                           precision=self.precision, name="router")(x)
        top_logits, indices = jax.lax.top_k(logits, self.num_experts_per_tok)
        weights = jax.nn.softmax(top_logits, axis=-1)
        return GptOssExperts(
            self.hidden_size, self.intermediate_size, self.num_local_experts,
            dtype=self.dtype, precision=self.precision, name="experts")(x, weights, indices)
