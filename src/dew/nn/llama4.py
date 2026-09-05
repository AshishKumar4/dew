"""Llama 4's text attention: iRoPE, chunked local layers, temperature tuning.

`Llama4TextAttention` (modeling_llama4.py) differs from the standard mixer
in four places, each a field of the reference's config. Local layers rotate
interleaved pairs, L2-normalise queries and keys with no scale
(`use_qk_norm`) and attend inside chunks of `attention_chunk_size`; global
layers carry no positions at all (`no_rope_layers`) and instead scale each
query by a logarithm of its position (`attn_temperature_tuning`, arXiv
2501.19399). The routed experts scale each token's input by its routing
weight, which `dew.nn.moe.ExpertMLP` does under `scale_inputs`.
"""

import dataclasses
import functools
from collections.abc import Callable
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import (
    RopeScaling, causal_attention_mask, open_kv_cache, rotary_freqs,
    scaled_dot_product_attention,
)
from dew.nn.mixers import MixerBase, MixerContext, mixers
from dew.nn.mla import apply_rotary_interleave
from dew.nn.sharding import logical_axes


def l2_norm(x, eps: float):
    """`Llama4TextL2Norm`: RMS-normalised in fp32 with no scale, cast back."""
    fp32 = x.astype(jnp.float32)
    return (fp32 * jax.lax.rsqrt(jnp.mean(jnp.square(fp32), axis=-1, keepdims=True) + eps)
            ).astype(x.dtype)


def temperature_scale(positions, floor_scale: float, attn_scale: float):
    """The query multiplier of a global layer at each absolute position.

    `log1p(floor((p + 1) / floor_scale)) * attn_scale + 1`, computed in fp32
    as the reference does, so the first `floor_scale` positions scale by 1.
    """
    positions = jnp.asarray(positions, jnp.float32)
    return jnp.log1p(jnp.floor((positions + 1.0) / floor_scale)) * attn_scale + 1.0


def chunk_mask(query_positions, key_positions, chunk_size: int):
    """Boolean `[.., 1, T, S]` keeping keys in the query's chunk, both absolute.

    transformers' `chunked_overlay`: `kv // chunk == q // chunk`, positions
    counted from the sequence start, so a packed document's positions place
    its chunks from its own first token.
    """
    query_chunks = jnp.asarray(query_positions) // chunk_size
    key_chunks = jnp.asarray(key_positions) // chunk_size
    same = query_chunks[..., :, None] == key_chunks[..., None, :]
    return same[..., None, :, :] if same.ndim == 3 else same[None, None]


@logical_axes({
    ("q_proj",): ("embed", "heads"),
    ("k_proj",): ("embed", "kv"),
    ("v_proj",): ("embed", "kv"),
    ("o_proj",): ("attention", "embed"),
})
class Llama4Attention(nn.Module):
    """Grouped-query attention under Llama 4's local or global rule.

    `use_rope` is the layer's `no_rope_layers` entry: a local layer rotates
    interleaved pairs at `rope_theta` and attends inside its chunk, a global
    layer skips the rotation and scales its queries by their position.
    decode=True runs against the fixed-size KV cache the way
    `CausalSelfAttention` does; the chunk mask reads the cache slots, which
    are the absolute positions.
    """
    emb_features: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    causal: bool = True
    rope_theta: float = 500000.0
    rope_scaling: Optional[RopeScaling] = None
    use_rope: bool = True
    use_qk_norm: bool = True
    attention_chunk_size: Optional[int] = None
    attn_temperature_tuning: bool = True
    floor_scale: float = 8192.0
    attn_scale: float = 0.1
    norm_eps: float = 1e-5
    attention_bias: bool = False
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    attention_impl: Optional[str] = None
    force_fp32_for_softmax: bool = True

    def setup(self):
        if self.attention_chunk_size is not None and self.attention_chunk_size < 1:
            raise ValueError(
                f"attention_chunk_size is a positive chunk length, got "
                f"{self.attention_chunk_size}; None attends the whole sequence")
        if self.attention_chunk_size is not None and not self.use_rope:
            raise ValueError(
                "Llama 4 chunks its rotated local layers only; a global layer "
                "without rope attends the whole sequence")
        dense = functools.partial(
            nn.Dense, use_bias=self.attention_bias, dtype=self.dtype, precision=self.precision)
        self.q_proj = dense(self.num_heads * self.head_dim, name='q_proj')
        self.k_proj = dense(self.num_kv_heads * self.head_dim, name='k_proj')
        self.v_proj = dense(self.num_kv_heads * self.head_dim, name='v_proj')
        self.o_proj = dense(self.emb_features, name='o_proj')

    @nn.compact
    def __call__(self, x, decode: bool = False,
                 positions=None, segment_ids=None, kv_store=None):
        batch, length, _ = x.shape
        query = self.q_proj(x).reshape(batch, length, self.num_heads, self.head_dim)
        key = self.k_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        value = self.v_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)

        append = None
        if decode:
            if not self.causal:
                raise ValueError("full attention has no KV cache to decode against")
            positions, append = open_kv_cache(self, key, self.max_seq_len)
        elif positions is None:
            positions = jnp.arange(length)
        else:
            positions = jnp.asarray(positions)

        if self.use_rope:
            freqs_cos, freqs_sin = rotary_freqs(positions, self.head_dim, self.rope_theta,
                                                rope_scaling=self.rope_scaling)
            query = apply_rotary_interleave(query, freqs_cos, freqs_sin)
            key = apply_rotary_interleave(key, freqs_cos, freqs_sin)
            if self.use_qk_norm:
                # The reference norms after rotating; the norm has no scale
                # and a rotation keeps every pair's length, so the two
                # orders agree to rounding.
                query = l2_norm(query, self.norm_eps)
                key = l2_norm(key, self.norm_eps)
        elif self.attn_temperature_tuning:
            scale = temperature_scale(positions, self.floor_scale, self.attn_scale)
            query = (query * scale[..., :, None, None].astype(query.dtype)
                     if scale.ndim == 2 else query * scale[None, :, None, None].astype(query.dtype))

        causal, mask = self.causal, None
        key_positions = positions
        if append is not None:
            key, value = append(key, value)
            key_positions = jnp.arange(key.shape[-3])
            mask = causal_attention_mask(positions, key.shape[-3])
            causal = False
        elif segment_ids is not None:
            segment_ids = jnp.asarray(segment_ids)
            inside = ((segment_ids[:, :, None] == segment_ids[:, None, :])
                      & (segment_ids[:, :, None] != 0))[:, None]
            mask = inside
            if causal:
                mask = jnp.logical_and(inside, causal_attention_mask(jnp.arange(length), length))
            causal = False
        if self.attention_chunk_size is not None:
            chunks = chunk_mask(positions, key_positions, self.attention_chunk_size)
            mask = chunks if mask is None else jnp.logical_and(mask, chunks)
        implementation = self.attention_impl
        if mask is not None and implementation in ('auto', 'cudnn'):
            # cuDNN takes no mask, and the reference mixer's measurement of
            # the xla kernel under a mask stands here too.
            implementation = 'xla'
        attention = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            implementation=implementation, causal=causal, mask=mask)
        return self.o_proj(attention.reshape(batch, length, self.num_heads * self.head_dim))


@mixers("llama4")
@dataclasses.dataclass(frozen=True)
class Llama4Mixer(MixerBase):
    """The `llama4` kind: `Llama4TextAttention` under the reference's names.

    `use_rope` is the layer's `no_rope_layers` entry, so a config names the
    local kind with the chunk and the global kind without rope; the other
    fields are the config's. The head geometry, rope base and its llama3
    ramp, bias, norm epsilon and kernel choice come from the backbone
    through the context.
    """

    use_rope: bool = True
    use_qk_norm: bool = True
    attention_chunk_size: Optional[int] = None
    attn_temperature_tuning: bool = True
    floor_scale: float = 8192.0
    attn_scale: float = 0.1

    def build(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        unsupported = {
            "qk_norm": ctx.qk_norm,
            "v_norm": ctx.v_norm,
            "kv_shared": ctx.kv_shared,
            "sliding_window": ctx.sliding_window,
            "attention_scale": ctx.attention_scale,
            "attention_sinks": ctx.attention_sinks,
            "yarn": ctx.yarn,
            "partial_rotary_factor": ctx.partial_rotary_factor,
            "output_gate": ctx.output_gate,
            "scale_offset": ctx.scale_offset,
        }
        asked = sorted(name for name, value in unsupported.items() if value)
        if asked:
            raise ValueError(
                f"the llama4 mixer has no {', '.join(asked)}: it norms queries "
                "and keys with its own scale-free L2 norm under use_qk_norm, "
                "scales by 1/sqrt(head_dim), rotates whole interleaved pairs "
                "and attends by chunk rather than by window")
        return functools.partial(
            Llama4Attention,
            emb_features=ctx.emb_features,
            num_heads=ctx.num_heads,
            num_kv_heads=ctx.num_kv_heads,
            head_dim=ctx.head_dim,
            max_seq_len=ctx.max_seq_len,
            causal=ctx.causal,
            rope_theta=ctx.rope_theta,
            rope_scaling=ctx.rope_scaling,
            use_rope=self.use_rope,
            use_qk_norm=self.use_qk_norm,
            attention_chunk_size=self.attention_chunk_size,
            attn_temperature_tuning=self.attn_temperature_tuning,
            floor_scale=self.floor_scale,
            attn_scale=self.attn_scale,
            norm_eps=ctx.norm_eps,
            attention_bias=ctx.attention_bias,
            dtype=ctx.dtype,
            precision=ctx.precision,
            attention_impl=ctx.attention_impl,
            force_fp32_for_softmax=ctx.force_fp32_for_softmax)


def default_no_rope_layers(num_layers: int, interval: int) -> tuple[int, ...]:
    """`Llama4TextConfig`'s pattern: every `interval`th layer carries no rope."""
    return tuple(int((index + 1) % interval != 0) for index in range(num_layers))


def rope_layer_types(no_rope_layers) -> tuple[str, ...]:
    """The config's layer pattern: rotated layers chunk, the rest attend whole."""
    return tuple('chunked_attention' if rope else 'full_attention' for rope in no_rope_layers)


