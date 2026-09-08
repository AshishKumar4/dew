"""GPT OSS's biased router, interleaved experts and MXFP4 checkpoint math."""

import functools
from collections.abc import Collection, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.dtypes import canonicalize_dtype
from flax.typing import Dtype, PrecisionLike

from dew.nn.moe import expert_dispatch, expert_projection, gather_expert_bias
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


def quantize_mxfp4(values: jax.Array) -> tuple[jax.Array, jax.Array]:
    """[expert, input, output] weights to packed [expert, output, group, 16] and E8M0 scales.

    The encoding `dequantize_mxfp4` reads back, lossy the way the format is:
    every 32 values along the input axis share one power-of-two scale and
    keep four bits each. The rule is the released encoder, transformers
    5.16.1's `quantize_to_mxfp4` over triton_kernels'
    `downcast_to_mxfp(..., ROUND_UP)`. It rounds the weight to bf16 first,
    takes a group's scale as its largest magnitude over 6 rounded up to a
    power of two -- `(bits + 0x007fffff) & 0x7f800000` on the fp32 quotient,
    which leaves an exact power of two where it is and sends an all-zero
    group to the 0x00 scale byte -- and writes each value over that scale as
    the nearest E2M1 value, ties to even, saturating at +-6. That element
    rounding is the float4_e2m1fn cast. The scale rule is spelled out
    because no cast performs it: float8_e8m0fnu rounds to nearest and takes
    0 to the reserved NaN byte 0xff, which a zero group is not.

    Three consequences worth naming. Rounding to bf16 is a second rounding:
    an fp32 weight just short of a tie in the E2M1 grid can be carried onto
    the tie and then rounded away from the value it started at. A group
    whose largest code is not +-6 or +-4 is rewritten with the scale this
    rule picks, so encoding a group that came from a checkpoint can move its
    bytes while every value it decodes to stays where it was. And the
    round-up stops at 2 ** -126, the reference's own fp32 floor, so a group
    whose largest magnitude is under 2 ** -128 has no scale left to reach it
    and rounds to zeros.
    """
    src = jnp.asarray(values)
    if not jnp.issubdtype(src.dtype, jnp.floating):
        raise ValueError(f"MXFP4 encodes float weights, got {src.dtype}")
    if src.ndim != 3 or src.shape[1] % 32:
        raise ValueError(
            "MXFP4 takes an [expert, input, output] weight whose input axis is a "
            f"multiple of the 32-value group, got {src.shape}")
    # bf16 first, as the reference does, and fp32 from there: a bf16 value
    # scaled by a power of two is exact in fp32, so nothing below rounds
    # except the one E2M1 conversion that is meant to.
    rows = src.astype(jnp.bfloat16).astype(jnp.float32).swapaxes(1, 2)
    groups = rows.reshape(*rows.shape[:2], -1, 32)
    # XLA reads a subnormal as zero and writes a subnormal result as zero,
    # where the reference's fp32 keeps both, so the group's largest is taken
    # from the bits: the pattern of |value| orders exactly as |value| does,
    # and an infinity or a NaN is every pattern at or above the exponent
    # field's last value.
    bits = jax.lax.bitcast_convert_type(groups, jnp.uint32)
    largest = jnp.max(bits & 0x7fffffff, axis=-1, keepdims=True)
    if not bool(jnp.all(largest < 0x7f800000)):
        raise ValueError(
            "MXFP4 holds no infinite or NaN weight: E2M1 encodes neither, and this "
            "rule gives such a group the reserved E8M0 NaN scale 0xff, which reads "
            "back as 2 ** 128")
    # The quotient of a group under 2 ** -64 (0x1f800000) is subnormal, and a
    # flushed one would say 2 ** 0 where the reference says 2 ** -126, so the
    # divide runs on a copy boosted into the normals and the boost comes off
    # the exponent, which then floors where the reference's subnormals do. A
    # group whose own largest is subnormal is boosted to zero here and floors
    # to the same 2 ** -126 the reference rounds it up to.
    tiny = largest < 0x1f800000
    largest_value = jax.lax.bitcast_convert_type(largest, jnp.float32)
    quotient = jax.lax.bitcast_convert_type(
        jnp.where(tiny, largest_value * 2.0 ** 96, largest_value) / 6, jnp.uint32)
    exponent = (((quotient + 0x007fffff) & 0x7f800000) >> 23).astype(jnp.int32)
    scales = jnp.where(largest == 0, 0,
                       jnp.maximum(exponent - jnp.where(tiny, 96, 0), 1)).astype(jnp.uint8)
    scale = jax.lax.bitcast_convert_type(scales.astype(jnp.uint32) << 23, jnp.float32)
    # The reciprocal is exact for every power of two the round-up can leave,
    # and a zero group multiplies by zero, which keeps a -0.0 weight's sign.
    # A subnormal weight is zero to that multiply and is not zero to the
    # format at the two smallest scales, so it goes through its significand
    # instead: read as an integer that is the weight times 2 ** 149, which at
    # 2 ** -23 is the weight times 2 ** 126 and normal, against a reciprocal
    # brought down by the same 2 ** 126.
    reciprocal = jnp.where(scale == 0, 0.0, 1 / scale)
    subnormal = (bits & 0x7f800000) == 0
    lifted = (bits & 0x007fffff).astype(jnp.float32) * 2.0 ** -23
    scaled = (jnp.where(subnormal, jnp.where(bits >> 31 == 1, -lifted, lifted), groups)
              * jnp.where(subnormal, reciprocal * 2.0 ** -126, reciprocal))
    codes = jax.lax.bitcast_convert_type(
        scaled.astype(jnp.float4_e2m1fn), jnp.uint4).astype(jnp.uint8)
    return codes[..., 0::2] | (codes[..., 1::2] << 4), scales.squeeze(-1)


def mxfp4_stems(tensors: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """The names a checkpoint ships as an MXFP4 `<stem>_blocks`/`<stem>_scales` pair.

    `unpack_mxfp4` spends this: the tensor it leaves behind no longer says it
    arrived quantized, so a loader that means to write the source's own
    format back reads the stems off the raw tensors first and carries them.
    Half a pair is refused either way round, since a checkpoint holding one
    of the two has lost the weight and not merely its format. Sorted, so the
    record does not depend on the order the shards were read in.
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
    """Replace each `<name>_blocks` and `<name>_scales` pair with `<name>` in fp32.

    fp32 holds every bf16 value exactly, so the unpacked tensor is the one
    transformers materializes before its own forward pass.
    """
    unpacked = dict(tensors)
    for stem in mxfp4_stems(tensors):
        blocks, scales = unpacked.pop(stem + '_blocks'), unpacked.pop(stem + '_scales')
        unpacked[stem] = np.asarray(
            dequantize_mxfp4(jnp.asarray(blocks), jnp.asarray(scales)).astype(jnp.float32))
    return unpacked


def pack_mxfp4(tensors: Mapping[str, np.ndarray],
               stems: Collection[str]) -> dict[str, np.ndarray]:
    """Replace each named `<stem>` with the `<stem>_blocks` and `<stem>_scales` it encodes to.

    What `unpack_mxfp4` undid, for the stems a source shipped packed
    (`mxfp4_stems`) and for nothing else: a tensor trained as a float weight
    -- the biases, the router, the attention sinks, the embeddings, a tied
    head -- is written back as itself rather than quantized for looking like
    a matrix. A named stem the tensors no longer hold is refused, since
    writing that checkpoint would leave its config still promising blocks.
    """
    packed = dict(tensors)
    for stem in stems:
        if stem not in packed:
            raise ValueError(
                f"{stem} arrived MXFP4 packed and is not among the tensors to write back")
        blocks, scales = quantize_mxfp4(jnp.asarray(packed.pop(stem)))
        packed[f'{stem}_blocks'] = np.asarray(blocks)
        packed[f'{stem}_scales'] = np.asarray(scales)
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
