"""GPT OSS's biased router, interleaved experts and MXFP4 checkpoint math.

The MXFP4 arithmetic is NumPy's on the host. XLA on CPU reads and writes
float32 subnormals as zero, and the released encoder and reader keep them:
a group of 2 ** -127 weights encodes to the 0.5 code at the 2 ** -126 scale
and decodes back to 2 ** -127, which the same code under jax.numpy returns
as zeros (measured, jax 0.11 CPU, scale bytes 0 and 1).
"""

import functools
from collections.abc import Collection, Mapping

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from numpy.typing import ArrayLike
from flax import linen as nn
from flax.linen.dtypes import canonicalize_dtype
from flax.typing import Dtype, PrecisionLike

from dew.nn.moe import expert_dispatch, expert_projection, gather_expert_bias
from dew.nn.sharding import logical_axes

GROUP = 32
"""Values along the input axis that share one E8M0 scale; 16 packed bytes."""

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)
"""The value of each E2M1 code, in code order."""


def dequantize_mxfp4(blocks: ArrayLike, scales: ArrayLike) -> np.ndarray:
    """Packed [expert, output, group, 16] blocks and E8M0 scales to float32 [expert, input, output].

    transformers' `convert_moe_packed_tensors`: the low nibble precedes the
    high nibble, and every value is exact in the bf16 it decodes to.
    """
    blocks, scales = np.asarray(blocks), np.asarray(scales)
    if blocks.ndim != 4 or blocks.shape[-1] != GROUP // 2 or blocks.shape[:-1] != scales.shape:
        raise ValueError("MXFP4 blocks must be [expert, output, group, 16] with one scale per group")
    if blocks.dtype != np.uint8 or scales.dtype != np.uint8:
        raise ValueError("MXFP4 blocks and scales must be uint8")
    codes = np.stack((blocks & 15, blocks >> 4), axis=-1).reshape(*scales.shape, GROUP)
    values = np.ldexp(E2M1[codes], scales.astype(np.int32)[..., None] - 127)
    return values.reshape(*blocks.shape[:2], -1).swapaxes(1, 2)


def quantize_mxfp4(weight: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """[expert, input, output] weights to packed [expert, output, group, 16] blocks and E8M0 scales.

    The released encoder, transformers 5.16.1's `quantize_to_mxfp4` over
    triton_kernels' `downcast_to_mxfp(..., ROUND_UP)`: the weight rounds to
    bf16, a group's scale is its largest magnitude over 6 rounded up to a
    power of two on the float32 bits (an all-zero group takes the 0x00
    byte), and each value over that scale rounds to the nearest E2M1 value,
    ties to even. The round-up keeps every scaled value at or under 6, so
    the float4_e2m1fn cast never saturates. The scale rule is spelled out
    because no cast performs it: float8_e8m0fnu rounds to nearest and sends
    0 to the NaN byte 0xff. An infinite or NaN weight would take that byte
    too, and is refused instead.
    """
    values = np.asarray(weight)
    # jnp's dtype lattice counts ml_dtypes' bfloat16 as floating; NumPy's does not.
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise ValueError(f"MXFP4 encodes float weights, got {values.dtype}")
    if values.ndim != 3 or values.shape[1] % GROUP:
        raise ValueError(
            "MXFP4 takes an [expert, input, output] weight whose input axis is a "
            f"multiple of the {GROUP}-value group, got {values.shape}")
    rows = values.astype(ml_dtypes.bfloat16).astype(np.float32).swapaxes(1, 2)
    groups = np.ascontiguousarray(rows).reshape(*rows.shape[:2], -1, GROUP)
    largest = np.abs(groups).max(-1, keepdims=True)
    if not np.isfinite(largest).all():
        raise ValueError(
            "MXFP4 holds no infinite or NaN weight: E2M1 encodes neither, and this "
            "rule gives such a group the reserved E8M0 NaN scale 0xff, which reads "
            "back as 2 ** 128")
    rounded = ((largest / np.float32(6)).view(np.uint32) + 0x007fffff) & 0x7f800000
    scale = rounded.view(np.float32)
    with np.errstate(divide='ignore'):
        reciprocal = np.where(scale == 0, np.float32(0), np.float32(1) / scale)
    codes = (groups * reciprocal).astype(ml_dtypes.float4_e2m1fn).view(np.uint8)
    return codes[..., 0::2] | (codes[..., 1::2] << 4), (rounded >> 23).astype(np.uint8).squeeze(-1)


def mxfp4_stems(tensors: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """The names a checkpoint ships as an MXFP4 `<stem>_blocks`/`<stem>_scales` pair, sorted.

    Taken before `unpack_mxfp4`, after which nothing says which tensors
    arrived packed. Half a pair is refused: the checkpoint has lost a weight.
    """
    stems = sorted({name.removesuffix(suffix) for name in tensors
                    for suffix in ('_blocks', '_scales') if name.endswith(suffix)})
    for stem in stems:
        for suffix in ('_blocks', '_scales'):
            if stem + suffix not in tensors:
                raise ValueError(
                    f"{stem} arrives MXFP4 packed and the checkpoint holds no {stem}{suffix}")
    return tuple(stems)


def unpack_mxfp4(tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Replace each `<name>_blocks` and `<name>_scales` pair with `<name>` in fp32."""
    unpacked = dict(tensors)
    for stem in mxfp4_stems(tensors):
        unpacked[stem] = dequantize_mxfp4(unpacked.pop(stem + '_blocks'),
                                          unpacked.pop(stem + '_scales'))
    return unpacked


def pack_mxfp4(tensors: Mapping[str, np.ndarray],
               stems: Collection[str]) -> dict[str, np.ndarray]:
    """Replace each named `<stem>` with the `<stem>_blocks` and `<stem>_scales` it encodes to.

    Only the stems a source shipped packed (`mxfp4_stems`): every other
    tensor is written back as itself. A named stem the tensors no longer
    hold is refused, since the config would still promise its blocks.
    """
    packed = dict(tensors)
    for stem in stems:
        if stem not in packed:
            raise ValueError(
                f"{stem} arrived MXFP4 packed and is not among the tensors to write back")
        packed[f'{stem}_blocks'], packed[f'{stem}_scales'] = quantize_mxfp4(packed.pop(stem))
    return packed


class GptOssExperts(nn.Module):
    """Interleaved gate/up matrices with the reference's clamped 1.702 SwiGLU.

    `implementation` names the grouped matmul, as `moe.expert_projection` takes it.
    """

    hidden_size: int
    intermediate_size: int
    num_local_experts: int
    implementation: str = 'xla'
    dispatch: str = 'global'
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
            (gate_up, gate_bias, down, down_bias), num_experts=self.num_local_experts,
            dispatch=self.dispatch, output_dtype=compute_dtype, initializing=self.is_initializing())
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
    implementation: str = 'xla'
    dispatch: str = 'global'
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
            implementation=self.implementation, dispatch=self.dispatch, dtype=self.dtype,
            precision=self.precision, name="experts")(x, weights, indices)
