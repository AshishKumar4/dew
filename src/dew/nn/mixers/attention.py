"""Grouped-query causal attention as a mixer kind.

The `attention` value builds this module: grouped-query projections, rotary
positions, q/k norms and a fixed-size KV cache.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import (
    RMSNorm, RopeScaling, apply_rotary, causal_attention_mask,
    max_attention_logits, open_kv_cache, rotary_freqs,
    scaled_dot_product_attention,
)
from dew.nn.inputs import AttentionMetadata
from dew.nn.mixers import MixerBase, MixerContext, mixers
from dew.nn.mla import YarnScaling, mla_rope_freqs
from dew.nn.sharding import logical_axes


@logical_axes({
    ("q_proj",): ("embed", "heads"),
    ("k_proj",): ("embed", "kv"),
    ("v_proj",): ("embed", "kv"),
    ("o_proj",): ("attention", "embed"),
})
class CausalSelfAttention(nn.Module):
    """Causal self-attention with grouped-query heads, rotary positions, qk
    RMSNorm and a fixed-size KV cache.

    decode=True runs the call against the cache: the first call writes the
    whole prompt and each later call appends one token, so prefill and decode
    are one code path. Keys are rotated before they enter the cache, so the
    rotary positions come from the cache index and not from the row index of
    the token.

    causal=False is full attention over the sequence, which a masked
    diffusion model reads the whole corrupted sequence with; there is no
    cache to decode against then, so decode=True raises.

    kv_shared marks a layer that owns no K/V projections (Gemma 3n/4 style
    cross-layer KV sharing): it reads the keys, values and their positions
    that the designated earlier layer of the same layer type stashed in
    `kv_store`, post rope and post norm, and keeps no cache of its own.
    """
    emb_features: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    causal: bool = True
    rope_theta: float = 10000.0
    rope_scaling: Optional[RopeScaling] = None  # Llama 3.1's ramp over the base frequencies
    qk_norm: bool = True
    qk_norm_scope: str = 'head'  # 'head': one RMSNorm per head; 'projection': over the whole q/k
    v_norm: bool = False
    norm_eps: float = 1e-5
    scale_offset: bool = False
    scale_after_cast: bool = False
    kv_shared: bool = False
    kv_store_key: Optional[str] = None
    sliding_window: Optional[int] = None
    attention_bias: bool = False  # q/k/v biases, as config.attention_bias in HF
    o_proj_bias: Optional[bool] = None  # None follows attention_bias; Qwen2 biases q/k/v only
    attention_scale: Optional[float] = None  # None: the kernel's own 1/sqrt(head_dim)
    attention_sinks: bool = False
    yarn: Optional[YarnScaling] = None
    attn_logit_softcap: Optional[float] = None  # Gemma 2's tanh on the logits, attn_logit_softcapping
    k_eq_v: bool = False  # Gemma 4's global layers project no values: the raw keys, values-normed
    output_gate: bool = False  # Qwen3.5 doubles q_proj and gates the branch with a sigmoid
    partial_rotary_factor: Optional[float] = None  # None: every head dim rotates
    partial_rotary_type: str = 'proportional'  # 'proportional' (Gemma 4) | 'default' (Qwen3.5)
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    attention_impl: Optional[str] = None
    force_fp32_for_softmax: bool = True
    bidirectional_images: bool = False
    mrope_section: tuple[int, int, int] | None = None

    def setup(self):
        dense = functools.partial(
            nn.Dense, use_bias=self.attention_bias, dtype=self.dtype, precision=self.precision)
        # The gate doubles the query projection: the reference chunks its
        # output in half, one half the query and the other the gate the
        # branch multiplies by (modeling_qwen3_5.py:670-673, 701).
        self.q_proj = dense(
            self.num_heads * self.head_dim * (2 if self.output_gate else 1), name='q_proj')
        # A sharing layer reads another layer's keys and values, so it owns
        # no projections or key norm of its own, as the reference skips them
        # (modeling_gemma4.py, Gemma4TextAttention.__init__).
        if not self.kv_shared:
            self.k_proj = dense(self.num_kv_heads * self.head_dim, name='k_proj')
            if not self.k_eq_v:
                self.v_proj = dense(self.num_kv_heads * self.head_dim, name='v_proj')
        self.o_proj = dense(self.emb_features, name='o_proj', use_bias=(
            self.attention_bias if self.o_proj_bias is None else self.o_proj_bias))
        if self.qk_norm:
            if self.qk_norm_scope not in ('head', 'projection'):
                raise ValueError(
                    f"qk_norm_scope is 'head' or 'projection', got {self.qk_norm_scope!r}")
            norm = functools.partial(
                RMSNorm, epsilon=self.norm_eps, scale_offset=self.scale_offset,
                scale_after_cast=self.scale_after_cast, dtype=self.dtype)
            self.q_norm = norm(name='q_norm')
            if not self.kv_shared:
                self.k_norm = norm(name='k_norm')
        if self.v_norm and not self.kv_shared:
            # Gemma 4 norms the values with a scale-free RMSNorm before they
            # are cached or shared (modeling_gemma4.py, Gemma4TextAttention).
            # The attribute cannot share the field's name, and it holds no
            # parameters either way.
            self.values_norm = RMSNorm(epsilon=self.norm_eps, with_scale=False,
                                       dtype=self.dtype, name='v_norm')

    def _rot_dim(self) -> int | None:
        """Head dims the rotary rotates, or None for all of them.

        `partial_rotary_type` says what the fraction means: 'proportional'
        rotates the first rot_dim dims of a head_dim-wide rope and passes the
        rest at frequency zero (Gemma 4), 'default' builds a rot_dim-wide rope
        and leaves the rest unrotated (Qwen3.5); `rotary_freqs` names the
        reference lines. Both rotate `int(head_dim * factor)` dims.
        """
        factor = self.partial_rotary_factor
        if factor is None:
            return None
        rot_dim = int(self.head_dim * factor)
        if not 0 < factor <= 1 or rot_dim % 2:
            raise ValueError(
                f"partial_rotary_factor must rotate an even positive number of "
                f"head dims, got {factor} of head_dim {self.head_dim}")
        return rot_dim

    def _multimodal_rotary(self, positions: jax.Array):
        """Qwen's interleaved temporal/height/width rotary frequency selection."""
        rotated = self._rot_dim() or self.head_dim
        if self.mrope_section is None or sum(self.mrope_section) != rotated // 2:
            raise ValueError("mrope_section must partition the rotated frequency pairs")
        if positions.shape[-1] != 3 or self.partial_rotary_type != "default":
            raise ValueError("M-RoPE requires three coordinates and default partial rotary")
        indices = jnp.arange(rotated // 2)
        axes = jnp.zeros((rotated // 2,), jnp.int32)
        for axis in (1, 2):
            axes = jnp.where((indices % 3 == axis) & (indices < self.mrope_section[axis] * 3), axis, axes)
        selected = jnp.take_along_axis(positions, axes[None, None, :], axis=-1)
        inv = self.rope_theta ** (-2 * indices.astype(jnp.float32) / rotated)
        angles = selected.astype(jnp.float32) * inv
        return jnp.cos(angles), jnp.sin(angles)

    def _metadata_mask(self, metadata: AttentionMetadata | None, slots,
                       batch: int, length: int, key_length: int, decode: bool):
        """Combine key validity, causality, image groups and the layer's window."""
        query_slots = (slots if decode else jnp.broadcast_to(jnp.arange(length), (batch, length)))
        groups = (None if metadata is None else metadata.image_groups)
        if groups is None:
            groups = jnp.full((batch, length), -1, jnp.int32)
        key_groups = groups
        if decode and self.bidirectional_images:
            from dew.nn.attention import _write_cache
            allocated = self.has_variable("cache", "cached_image_groups")
            stored = self.variable("cache", "cached_image_groups", jnp.full,
                                   (batch, self.max_seq_len), -1, jnp.int32)
            if allocated:
                stored.value = _write_cache(stored.value, groups, query_slots)
            key_groups = stored.value
        if decode:
            valid = self.get_variable("cache", "cache_valid")
        else:
            valid = None if metadata is None else metadata.valid
        keep = (causal_attention_mask(query_slots, key_length)
                if self.causal else jnp.ones((batch, 1, length, key_length), bool))
        if self.bidirectional_images:
            same_image = ((groups[:, :, None] == key_groups[:, None, :])
                          & (groups[:, :, None] >= 0))
            keep = keep | same_image[:, None]
        if self.sliding_window is not None:
            keep = keep & (jnp.arange(key_length)[None, None, None, :]
                           > query_slots[:, None, :, None] - self.sliding_window)
        if valid is not None:
            keep = keep & valid[:, None, None, :]
        if decode:
            keep = keep & (query_slots[:, None, :, None] >= 0)
        return keep


    @nn.compact
    def __call__(self, x, decode: bool = False,
                 positions=None, segment_ids=None, kv_store=None,
                 attention_metadata: AttentionMetadata | None = None):
        B, S, _ = x.shape
        logical_positions = positions
        projected = self.q_proj(x)
        # OLMo 3 norms the whole projection, one scale of heads * head_dim,
        # before the head split (modeling_olmo3.py:162-163, :178-179); Qwen3
        # and the Gemmas norm each head after it, which the reference marks
        # "unlike olmo, only on the head dim" (modeling_qwen3_moe.py:147).
        whole = self.qk_norm and self.qk_norm_scope == 'projection'
        if whole:
            projected = self.q_norm(projected)
        gate = None
        if self.output_gate:
            # The reference views the doubled output as [.., heads, 2*head_dim]
            # and chunks it: query, then gate (modeling_qwen3_5.py:670-673).
            query, gate = jnp.split(
                projected.reshape(B, S, self.num_heads, 2 * self.head_dim),
                2, axis=-1)
        else:
            query = projected.reshape(B, S, self.num_heads, self.head_dim)
        if self.kv_shared:
            # The provider ran earlier in the same forward pass and stashed
            # its post-norm, post-rope keys and values with their positions,
            # so there is nothing to project, norm, rotate or cache here.
            if kv_store is None or self.kv_store_key not in kv_store:
                raise ValueError(
                    f"layer shares K/V under {self.kv_store_key!r} but no provider "
                    "stashed them; the model has to pass one kv_store dict down "
                    "its layer stack")
            key, value, positions = kv_store[self.kv_store_key]
        else:
            key = self.k_proj(x)
            # attention_k_eq_v reads the values off the key projection before
            # its norm (modeling_gemma4.py, Gemma4TextAttention.forward).
            value = (key if self.k_eq_v else self.v_proj(x)).reshape(
                B, S, self.num_kv_heads, self.head_dim)
            if whole:
                key = self.k_norm(key)
            key = key.reshape(B, S, self.num_kv_heads, self.head_dim)
            if self.qk_norm and not whole:
                key = self.k_norm(key)
            if self.v_norm:
                value = self.values_norm(value)
        if self.qk_norm and not whole:
            query = self.q_norm(query)

        # The cache slot carries position while decoding, so the rotation and
        # the mask both read it and not the row index of the token. A packed
        # batch supplies the position inside its document in place of the
        # row index, and RoPE restarts at every boundary.
        append = None
        prefix = None
        kv_len = key.shape[-3]
        if decode:
            if self.causal:
                if not self.kv_shared:
                    positions, append = open_kv_cache(
                        self, key, self.max_seq_len,
                        valid=None if attention_metadata is None else attention_metadata.valid)
            elif self.kv_shared or segment_ids is not None:
                raise ValueError(
                    "a bidirectional canvas over a cache shares no keys across "
                    "layers and packs no segments: a sharing layer owns no "
                    "cache of its own, and packed positions do not continue "
                    "past a prefix")
            elif not self.has_variable("cache", "cached_key"):
                raise ValueError(
                    "a bidirectional canvas has no KV cache of its own: prefill "
                    "the prompt with the causal model first")
            else:
                # The encoder's frozen prefix: positions continue past it, and
                # the decoder never writes it back.
                prefix = self.get_variable("cache", "cache_index")
                positions = prefix[:, None] + jnp.arange(S)
        elif positions is None and not self.kv_shared:
            positions = jnp.arange(S)
        elif not self.kv_shared:
            positions = jnp.asarray(positions)
        rotary_positions = positions if logical_positions is None else logical_positions
        if attention_metadata is not None and attention_metadata.rotary_positions is not None:
            rotary_positions = attention_metadata.rotary_positions
        if self.mrope_section is not None and rotary_positions is not None and rotary_positions.ndim == 3:
            freqs_cos, freqs_sin = self._multimodal_rotary(rotary_positions)
        elif self.yarn is None:
            freqs_cos, freqs_sin = rotary_freqs(
                rotary_positions, self.head_dim, self.rope_theta, rot_dim=self._rot_dim(),
                partial_rotary_type=self.partial_rotary_type, rope_scaling=self.rope_scaling)
        else:
            if self.partial_rotary_factor is not None or self.rope_scaling is not None:
                raise ValueError(
                    "yarn rotates whole heads at its own frequencies, so it takes "
                    "neither partial_rotary_factor nor rope_scaling")
            freqs_cos, freqs_sin = mla_rope_freqs(
                rotary_positions, self.head_dim, self.rope_theta, self.yarn)
        # Every kernel path scales the logits by 1/sqrt(head_dim) itself, so the
        # query carries the ratio to the scale the checkpoint asks for.
        query = apply_rotary(
            query, freqs_cos, freqs_sin,
            scale=(None if self.attention_scale is None
                   else self.attention_scale * math.sqrt(self.head_dim)))
        if not self.kv_shared:
            key = apply_rotary(key, freqs_cos, freqs_sin)
            if kv_store is not None and self.kv_store_key is not None:
                # Post-norm, post-rope, the same tensors the reference hands
                # its sharing layers (modeling_gemma4.py, Gemma4TextAttention).
                kv_store[self.kv_store_key] = (key, value, positions)
        causal, mask = self.causal, None
        implementation = self.attention_impl
        window = None if decode else self.sliding_window
        if prefix is not None:
            # Every canvas query reads the same retained encoder keys and all
            # canvas keys (modeling_diffusion_gemma.py:1399-1401). A local
            # layer retains the last window-1 prefix keys.
            cached_key = self.get_variable("cache", "cached_key")
            cached_value = self.get_variable("cache", "cached_value")
            alloc = cached_key.shape[-3]
            prefix_slots = jnp.arange(alloc)[None, :]
            valid_prefix = self.get_variable("cache", "cache_valid")
            if self.sliding_window is not None:
                valid_prefix = valid_prefix & (prefix_slots > prefix[:, None] - self.sliding_window)
            canvas_valid = (jnp.ones((B, S), bool) if attention_metadata is None
                            or attention_metadata.valid is None else attention_metadata.valid)
            keep = jnp.concatenate([valid_prefix, canvas_valid], axis=-1)
            mask = jnp.broadcast_to(keep[:, None, None, :], (B, 1, S, alloc + S))
            key = jnp.concatenate([cached_key, key], axis=-3)
            value = jnp.concatenate([cached_value, value], axis=-3)
            causal, window = False, None
        elif self.kv_shared and decode:
            # No cache of its own: the provider's stashed keys carry the full
            # history, so the mask reads them the way the provider's own
            # decode mask does.
            mask = causal_attention_mask(positions, kv_len, self.sliding_window)
            causal = False
        elif append is not None:
            key, value = append(key, value)
            mask = causal_attention_mask(
                positions, key.shape[-3], self.sliding_window,
                key_valid=self.get_variable("cache", "cache_valid"))
            causal = False
            if kv_store is not None and self.kv_store_key is not None:
                kv_store[self.kv_store_key] = (key, value, positions)
        elif segment_ids is not None:
            # Attention stays inside each packed document: the segment ids
            # make the mask block-diagonal, padding (segment 0) sees nothing,
            # and causality (with the layer's window) travels in the same mask
            # and not as the kernels' flag.
            segment_ids = jnp.asarray(segment_ids)
            inside = ((segment_ids[:, :, None] == segment_ids[:, None, :])
                      & (segment_ids[:, :, None] != 0))[:, None]
            mask = inside
            if causal:
                mask = jnp.logical_and(
                    inside, causal_attention_mask(jnp.arange(S), S, self.sliding_window))
            causal, window = False, None
            if implementation in ('auto', 'cudnn'):
                # cuDNN has no mask argument: causality and the window are
                # flags, and jax hands the kernel a bool mask as an additive
                # bias of -2**41 in the compute dtype instead
                # (combine_bias_and_mask in
                # jax/_src/cudnn/fused_attention_stablehlo.py), which also
                # makes check_is_flash_attention refuse an odd length while
                # training. The xla kernel masks by exclusion, on every
                # backend and with the same fp32 softmax. It costs 83.6 ms
                # and 5.80 GiB a step where the fixed window on cuDNN costs
                # 75.8 ms and 4.99 GiB, measured in
                # docs/concepts/language_models.md.
                implementation = 'xla'
        if prefix is None and (attention_metadata is not None or self.bidirectional_images):
            mask = self._metadata_mask(attention_metadata, positions, B, S, key.shape[-3], decode)
            if segment_ids is not None and not decode:
                inside = ((segment_ids[:, :, None] == segment_ids[:, None, :])
                          & (segment_ids[:, :, None] != 0))
                mask = mask & inside[:, None]
            causal, window = False, None
            if implementation in ("auto", "cudnn"):
                implementation = "xla"
        if attention_metadata is not None and attention_metadata.pairwise_mask is not None:
            pairwise = jnp.asarray(attention_metadata.pairwise_mask)
            if pairwise.shape != (B, S, key.shape[-3]) or pairwise.dtype != jnp.bool_:
                raise ValueError("attention_pairwise_mask must be boolean [B, queries, keys]")
            mask = pairwise[:, None]
            key_positions = attention_metadata.key_positions
            if key_positions is not None:
                if key_positions.shape != (B, key.shape[-3]):
                    raise ValueError("attention_key_positions must be [B, keys]")
                if self.sliding_window is not None:
                    query_positions = jnp.broadcast_to(jnp.asarray(rotary_positions), (B, S))
                    distance = query_positions[:, :, None] - key_positions[:, None, :]
                    mask = mask & (jnp.abs(distance) < self.sliding_window)[:, None]
            causal, window = False, None
            if implementation in ("auto", "cudnn"):
                implementation = "xla"
        # The per-head maxima the QK-Clip reads. Computed only when a caller
        # opened the collection; the plain forward leaves it closed and its
        # leaves bitwise identical.
        if not self.is_initializing() and self.is_mutable_collection("qk"):
            self.sow("qk", "max_logits", max_attention_logits(
                query, key, causal=causal, sliding_window=window, mask=mask))
        attention = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            implementation=implementation, causal=causal,
            sliding_window=window, mask=mask,
            sinks=(self.param('sinks', nn.initializers.zeros, (self.num_heads,))
                   if self.attention_sinks else None),
            softcap=self.attn_logit_softcap)
        if gate is not None:
            # The branch multiplies by the sigmoid of its gate, then projects
            # (modeling_qwen3_5.py:701, and modeling_qwen4_exp.py:836 the same).
            attention = attention * jax.nn.sigmoid(gate).astype(attention.dtype)
        return self.o_proj(attention.reshape(B, S, self.num_heads * self.head_dim))


@mixers("attention")
@dataclasses.dataclass(frozen=True)
class AttentionMixer(MixerBase):
    """Grouped-query attention with optional image-block masking and M-RoPE.

    Geometry, norms and kernel policy come from the decoder context. Image
    bidirectionality and spatial rotary sections configure this mixer only.
    """

    bidirectional_images: bool = False
    mrope_section: tuple[int, int, int] | None = None

    def __post_init__(self):
        if self.mrope_section is not None:
            object.__setattr__(self, "mrope_section", tuple(self.mrope_section))
            if len(self.mrope_section) != 3 or any(value < 0 for value in self.mrope_section):
                raise ValueError("mrope_section must contain three nonnegative section widths")

    def build(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        return functools.partial(
            CausalSelfAttention,
            emb_features=ctx.emb_features,
            num_heads=ctx.num_heads,
            num_kv_heads=ctx.num_kv_heads,
            head_dim=ctx.head_dim,
            max_seq_len=ctx.max_seq_len,
            causal=ctx.causal,
            rope_theta=ctx.rope_theta,
            rope_scaling=ctx.rope_scaling,
            qk_norm=ctx.qk_norm,
            qk_norm_scope=ctx.qk_norm_scope,
            v_norm=ctx.v_norm,
            k_eq_v=ctx.k_eq_v,
            norm_eps=ctx.norm_eps,
            scale_offset=ctx.scale_offset,
            scale_after_cast=ctx.scale_after_cast,
            kv_shared=ctx.kv_shared,
            kv_store_key=ctx.kv_store_key,
            sliding_window=ctx.sliding_window,
            attention_bias=ctx.attention_bias,
            o_proj_bias=ctx.o_proj_bias,
            attention_scale=ctx.attention_scale,
            attention_sinks=ctx.attention_sinks,
            yarn=ctx.yarn,
            attn_logit_softcap=ctx.attn_logit_softcap,
            output_gate=ctx.output_gate,
            dtype=ctx.dtype,
            precision=ctx.precision,
            attention_impl=ctx.attention_impl,
            force_fp32_for_softmax=ctx.force_fp32_for_softmax,
            partial_rotary_factor=ctx.partial_rotary_factor,
            partial_rotary_type=ctx.partial_rotary_type,
            bidirectional_images=self.bidirectional_images, mrope_section=self.mrope_section)

