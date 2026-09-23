"""Grouped-query causal attention as a mixer kind.

The `attention` value builds this module: grouped-query projections, rotary
positions, q/k norms and a fixed-size KV cache.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name

from dew.nn.attention import (
    RMSNorm,
    causal_attention_mask,
    chunk_mask,
    combined_attention_mask,
    kernel_for_materialized_mask,
    local_attention,
    max_attention_logits,
    open_kv_cache,
    scaled_dot_product_attention,
    with_documents,
)
from dew.nn.blocks import normal_kernel
from dew.nn.inputs import AttentionMetadata
from dew.nn.kv_cache import Append, KVCache, rotated, write_cache
from dew.nn.mixers import MixerBase, MixerContext, mixers
from dew.nn.precision import scaled
from dew.nn.rope import RopeScaling, YarnScaling, apply_rotary, rotary_freqs, yarn_rope_freqs
from dew.nn.sharding import HEADS, KV_HEADS, constrain, logical_axes


def exclusive_self_attention(attention: jax.Array, value: jax.Array) -> jax.Array:
    """Remove each head's component along its token's own value vector.

    Exclusive self attention (arXiv 2603.09078) as lm-engine implements it
    (`SoftmaxAttention._compute_xsa_output`, softmax_attention/module.py at
    45b6b57b): `y - (<y, v> / <v, v>) v` per token and query head, in fp32
    at least, with the key/value head repeated over its query group. `attention` is
    `[B, S, heads, D]`, `value` `[B, S, kv_heads, D]`.

    lm-engine divides by `<v, v>` with no guard, so a zero value vector
    (padding under a values norm, or a zero-initialised projection) is NaN
    there; here it leaves the output unchanged, which is the limit of the
    projection onto a vanishing direction's span being empty.
    """
    heads, kv_heads = attention.shape[-2], value.shape[-2]
    if heads % kv_heads:
        raise ValueError(f"{heads} query heads do not group over {kv_heads} value heads")
    dtype = jnp.promote_types(jnp.promote_types(attention.dtype, value.dtype), jnp.float32)
    value = jnp.repeat(value.astype(dtype), heads // kv_heads, axis=-2)
    work = attention.astype(dtype)
    norm = jnp.sum(value * value, axis=-1, keepdims=True)
    along = jnp.sum(work * value, axis=-1, keepdims=True) / jnp.where(norm > 0, norm, 1)
    return (work - jnp.where(norm > 0, along, 0) * value).astype(attention.dtype)


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
    rope_scaling: RopeScaling | None = None  # Llama 3.1's ramp over the base frequencies
    qk_norm: bool = True
    qk_norm_scope: str = 'head'  # 'head': one RMSNorm per head; 'projection': over the whole q/k
    v_norm: bool = False
    norm_eps: float = 1e-5
    scale_offset: bool = False
    scale_after_cast: bool = False
    kv_shared: bool = False
    kv_store_key: str | None = None
    sliding_window: int | None = None
    attention_chunk: int | None = None  # chunked local attention: keys sharing the query's position // chunk
    attention_bias: bool = False  # q/k/v biases, as config.attention_bias in HF
    o_proj_bias: bool | None = None  # None follows attention_bias; Qwen2 biases q/k/v only
    attention_scale: float | None = None  # None: the kernel's own 1/sqrt(head_dim)
    attention_sinks: bool = False
    yarn: YarnScaling | None = None
    attn_logit_softcap: float | None = None  # Gemma 2's tanh on the logits, attn_logit_softcapping
    k_eq_v: bool = False  # Gemma 4's global layers project no values: the raw keys, values-normed
    output_gate: bool = False  # Qwen3.5 doubles q_proj and gates the branch with a sigmoid
    partial_rotary_factor: float | None = None  # None: every head dim rotates
    partial_rotary_type: str = 'proportional'  # 'proportional' (Gemma 4) | 'default' (Qwen3.5)
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl
    force_fp32_for_softmax: bool = True
    bidirectional_images: bool = False
    mrope_section: tuple[int, int, int] | None = None
    kv_cache: KVCache = KVCache()  # the decode cache's storage: dense or paged, full or quantized
    nope: bool = False
    """No positional encoding: q and k enter the kernel unrotated, and the
    logits keep their scale (lm-engine's `position_embedding_type="nope"`)."""
    exclusive_self_attention: bool = False
    """XSA (arXiv 2603.09078): each head's output loses its component along
    the token's own value vector before the output projection."""
    init_std: float | None = None
    """Normal std of the q/k/v kernels; None keeps flax's lecun normal."""
    output_init_std: float | None = None
    """Normal std of o_proj; None follows init_std."""

    def setup(self):
        if self.exclusive_self_attention and self.kv_shared:
            raise ValueError(
                "exclusive self attention subtracts the token's own value, and a "
                "KV-sharing layer projects none of its own")
        dense = functools.partial(
            nn.Dense, use_bias=self.attention_bias, dtype=self.dtype, precision=self.precision,
            **normal_kernel(self.init_std))
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
            self.attention_bias if self.o_proj_bias is None else self.o_proj_bias),
            **normal_kernel(self.init_std if self.output_init_std is None else self.output_init_std))
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
        if self.mrope_section is None:
            raise ValueError("multimodal rotary requires mrope_section")
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

    def _shared_kv(self, kv_store):
        """Read the keys, values and positions the provider layer stashed.

        The provider ran earlier in the same forward pass and left them
        post-norm and post-rope, so a sharing layer projects, norms,
        rotates and caches nothing of its own.
        """
        if kv_store is None or self.kv_store_key not in kv_store:
            raise ValueError(
                f"layer shares K/V under {self.kv_store_key!r} but no provider "
                "stashed them; the model has to pass one kv_store dict down "
                "its layer stack")
        return kv_store[self.kv_store_key]

    def _projected_kv(self, x, whole: bool):
        """Project the keys and values, norm them, and split them into heads.

        `whole` norms the key projection before the head split, which is
        OLMo 3's scope. Otherwise the key norm runs per head, after it.
        """
        batch, length, _ = x.shape
        key = checkpoint_name(self.k_proj(x), 'k_proj')
        # attention_k_eq_v reads the values off the key projection before
        # its norm (modeling_gemma4.py, Gemma4TextAttention.forward).
        value = (key if self.k_eq_v else checkpoint_name(self.v_proj(x), 'v_proj')).reshape(
            batch, length, self.num_kv_heads, self.head_dim)
        if whole:
            key = self.k_norm(key)
        key = key.reshape(batch, length, self.num_kv_heads, self.head_dim)
        if self.qk_norm and not whole:
            key = self.k_norm(key)
        if self.v_norm:
            value = self.values_norm(value)
        return constrain(key, KV_HEADS), constrain(value, KV_HEADS)

    def _rotary_angles(self, rotary_positions):
        """Build the rotary cos and sin this layer rotates its heads by.

        Interleaved mRoPE, YaRN and the plain rope each build their own
        angles; YaRN rotates whole heads at its own frequencies, so it
        takes neither a partial rotary nor a Llama 3.1 ramp.
        """
        if self.mrope_section is not None and rotary_positions is not None and rotary_positions.ndim == 3:
            return self._multimodal_rotary(rotary_positions)
        if self.yarn is None:
            return rotary_freqs(
                rotary_positions, self.head_dim, self.rope_theta, rot_dim=self._rot_dim(),
                partial_rotary_type=self.partial_rotary_type, rope_scaling=self.rope_scaling)
        if self.partial_rotary_factor is not None or self.rope_scaling is not None:
            raise ValueError(
                "yarn rotates whole heads at its own frequencies, so it takes "
                "neither partial_rotary_factor nor rope_scaling")
        return yarn_rope_freqs(rotary_positions, self.head_dim, self.rope_theta, self.yarn)

    def _metadata_mask(self, metadata: AttentionMetadata | None, slots,
                       batch: int, length: int, key_length: int, decode: bool):
        """Combine key validity, causality, image groups and the layer's window."""
        query_slots = (slots if decode else jnp.broadcast_to(jnp.arange(length), (batch, length)))
        groups = (None if metadata is None else metadata.image_groups)
        if groups is None:
            groups = jnp.full((batch, length), -1, jnp.int32)
        key_groups = groups
        if decode and self.bidirectional_images:
            allocated = self.has_variable("cache", "cached_image_groups")
            stored = self.variable("cache", "cached_image_groups", jnp.full,
                                   (batch, self.max_seq_len), -1, jnp.int32)
            if allocated:
                stored.value = write_cache(stored.value, groups, query_slots)
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

    def _restricts_visibility(self, metadata: AttentionMetadata | None, decode: bool) -> bool:
        """Whether metadata narrows who sees whom, so the mask has to be built.

        Key validity does, and image groups do on a layer that makes images
        bidirectional. Rotary positions rotate q and k and leave visibility
        alone, and an explicit pairwise mask builds its own below. Metadata
        that restricts nothing keeps causality and the window as the flags
        the fused kernels take, because a materialized [B, 1, S, S] mask
        sends the call to the xla kernel and costs the fused one's time and
        memory (the numbers are in docs/performance.md).

        A validity array is opaque at trace time, so its contents decide
        nothing here. An all-true one restricts as much as any other, and a
        host that knows a row is unpadded says so by passing none.

        Decoding always builds the mask, which carries the cache's own
        validity, and a bidirectional-image layer writes its cached groups
        while building it.
        """
        if decode:
            return metadata is not None or self.bidirectional_images
        return metadata is not None and (
            metadata.valid is not None
            or (self.bidirectional_images and metadata.image_groups is not None))

    @nn.compact
    def __call__(self, x, decode: bool = False,
                 positions=None, segment_ids=None, kv_store=None,
                 attention_metadata: AttentionMetadata | None = None):
        B, S, _ = x.shape
        logical_positions = positions
        # The projections and the kernel's output carry the names a remat
        # policy saves or offloads (causal_transformer.RESIDUALS).
        projected = checkpoint_name(self.q_proj(x), 'q_proj')
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
            key, value, positions = self._shared_kv(kv_store)
        else:
            key, value = self._projected_kv(x, whole)
        if self.qk_norm and not whole:
            query = self.q_norm(query)
        # Column-parallel under a tensor axis, each shard a run of whole
        # heads; o_proj's sum returns to the residual placement in the block.
        query = constrain(query, HEADS)

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
                        valid=None if attention_metadata is None else attention_metadata.valid,
                        layout=self.kv_cache)
            elif self.kv_cache != KVCache():
                raise ValueError(
                    "a bidirectional canvas reads its encoder prefix as dense, full-precision "
                    "keys; it takes the default kv_cache layout")
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
        freqs_cos = freqs_sin = None
        if self.nope:
            # NoPE rotates nothing; the query still carries the logit scale
            # the checkpoint asks for, which apply_rotary folds in otherwise.
            if self.attention_scale is not None:
                query = scaled(query, self.attention_scale * math.sqrt(self.head_dim))
        else:
            freqs_cos, freqs_sin = self._rotary_angles(rotary_positions)
            # Every kernel path scales the logits by 1/sqrt(head_dim) itself, so the
            # query carries the ratio to the scale the checkpoint asks for.
            query = apply_rotary(
                query, freqs_cos, freqs_sin,
                scale=(None if self.attention_scale is None
                       else self.attention_scale * math.sqrt(self.head_dim)))
        own_value = value
        if not self.kv_shared:
            if freqs_cos is not None and freqs_sin is not None:
                key = apply_rotary(key, freqs_cos, freqs_sin)
            if kv_store is not None and self.kv_store_key is not None:
                # Post-norm, post-rope, the same tensors the reference hands
                # its sharing layers (modeling_gemma4.py, Gemma4TextAttention).
                kv_store[self.kv_store_key] = (key, value, positions)
        sinks = (self.param('sinks', nn.initializers.zeros, (self.num_heads,))
                 if self.attention_sinks else None)
        if self._runs_local(attention_metadata, decode):
            attention = checkpoint_name(local_attention(
                query, key, value, window=self.sliding_window, chunk=self.attention_chunk,
                positions=None if logical_positions is None else positions,
                segment_ids=segment_ids,
                valid=None if attention_metadata is None else attention_metadata.valid,
                dtype=self.dtype, precision=self.precision,
                force_fp32_for_softmax=self.force_fp32_for_softmax,
                implementation=self.attention_impl, sinks=sinks,
                softcap=self.attn_logit_softcap), 'context')
            return self._output(attention, gate, B, S, own_value)
        causal, mask, documents = self.causal, None, None
        implementation = self.attention_impl
        masked = kernel_for_materialized_mask(
            implementation, query, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax)
        window = None if decode else self.sliding_window
        cursor = None  # the decode mask the paged kernel stands in for, when this call builds it
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
            # A quantized provider stashed its keys in the cache's rotation.
            rotation = self.kv_cache.key_rotation(self.head_dim)
            if rotation is not None:
                query = rotated(query, rotation)
        elif append is not None:
            key, value = append(key, value)
            query = append.query(query)
            cursor, mask = self._decode_masks(positions, key.shape[-3])
            causal = False
            if kv_store is not None and self.kv_store_key is not None:
                kv_store[self.kv_store_key] = (key, value, positions)
        elif segment_ids is not None:
            # Attention stays inside each packed document. The ids travel to
            # the kernel beside the causal flag and the window: splash compares
            # them per block, and every other kernel builds the document mask
            # from them (`attention_kernel`). A bidirectional layer reads its
            # whole document.
            documents = segment_ids
            if not causal:
                window = None
        if prefix is None and self._restricts_visibility(attention_metadata, decode):
            # A decode step's cache mask already holds the rows' validity; only
            # image groups add to it there.
            if not decode or self.bidirectional_images:
                mask = self._metadata_mask(attention_metadata, positions, B, S, key.shape[-3], decode)
                if segment_ids is not None and not decode:
                    mask = with_documents(mask, segment_ids)
            documents = None
            causal, window = False, None
            implementation = masked
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
            causal, window, documents = False, None, None
            implementation = masked
        if self.attention_chunk is not None:
            # The decode and diagnostic paths: the chunk joins the mask the
            # branch above built, keys placed at their cache slots while
            # decoding and at the positions the query side reads otherwise.
            key_places = (jnp.arange(key.shape[-3]) if decode else positions)
            if attention_metadata is not None and attention_metadata.key_positions is not None:
                key_places = attention_metadata.key_positions
            chunked = chunk_mask(positions, key_places, self.attention_chunk)
            if documents is not None:
                mask, documents = with_documents(mask, documents), None
            base = combined_attention_mask(S, key.shape[-3], causal, window, mask)
            mask = chunked if base is None else base & chunked
            causal, window = False, None
            implementation = masked
        # The per-head maxima the QK-Clip reads. Computed only when a caller
        # opened the collection; the plain forward leaves it closed and its
        # leaves bitwise identical.
        sowing = not self.is_initializing() and self.is_mutable_collection("qk")
        if sowing:
            self.sow("qk", "max_logits", max_attention_logits(
                query, key, causal=causal, sliding_window=window,
                mask=mask if documents is None else with_documents(mask, documents)))
        # A chunk, window or metadata mask replaces `cursor` and keeps the gather.
        if (append is not None and mask is cursor and S == 1 and sinks is None and not sowing
                and self.attention_impl in ('auto', 'tpu') and append.store.kernel()):
            attention = self._paged(append, query)
        else:
            attention = checkpoint_name(scaled_dot_product_attention(
                query, key, value, dtype=self.dtype, precision=self.precision,
                force_fp32_for_softmax=self.force_fp32_for_softmax,
                implementation=implementation, causal=causal,
                sliding_window=window, mask=mask, sinks=sinks,
                softcap=self.attn_logit_softcap, segment_ids=documents), 'context')
        return self._output(attention, gate, B, S, own_value)

    def _decode_masks(self, positions, key_length: int) -> tuple[jax.Array, jax.Array]:
        """The cursor mask over the cache's filled slots, and the layer's decode mask.

        They are the same array exactly when the layer narrows nothing, which
        is when the paged kernel, attending every filled slot, may stand in
        for the mask; a window builds a mask of its own.
        """
        valid = self.get_variable("cache", "cache_valid")
        cursor = causal_attention_mask(positions, key_length, key_valid=valid)
        if self.sliding_window is None:
            return cursor, cursor
        return cursor, causal_attention_mask(positions, key_length, self.sliding_window, key_valid=valid)

    def _paged(self, append: Append, query: jax.Array) -> jax.Array:
        """One decode step through the Pallas paged kernel, which reads the pool
        through the page table; the gathered keys go unread and XLA drops the gather."""
        return checkpoint_name(append.store.decode(
            query[:, 0], self.get_variable("cache", "cache_index"), self.attn_logit_softcap)[:, None],
            'context')

    def _runs_local(self, metadata: AttentionMetadata | None, decode: bool) -> bool:
        """Whether this call runs `local_attention`, which never builds the
        `[S, S]` mask a window or chunk otherwise costs.

        That is a causal local layer's whole-sequence pass, over row
        validity and packed documents. A cache, a pairwise mask or
        bidirectional image groups keep the mask path, and so does an open
        `qk` collection, whose maxima read the dense logits anyway.
        """
        if (self.sliding_window is None and self.attention_chunk is None) or not self.causal:
            return False
        if decode or (not self.is_initializing() and self.is_mutable_collection("qk")):
            return False
        return metadata is None or (
            metadata.pairwise_mask is None
            and not (self.bidirectional_images and metadata.image_groups is not None))

    def _output(self, attention, gate, batch: int, length: int, own_value):
        """Exclude the own value where the layer does, gate the attended values
        where it gates, then project them."""
        if self.exclusive_self_attention:
            attention = exclusive_self_attention(attention, own_value)
        if gate is not None:
            # The branch multiplies by the sigmoid of its gate, then projects
            # (modeling_qwen3_5.py:701, and modeling_qwen4_exp.py:836 the same).
            attention = attention * jax.nn.sigmoid(gate).astype(attention.dtype)
        attention = constrain(attention, HEADS)
        return checkpoint_name(
            self.o_proj(attention.reshape(batch, length, self.num_heads * self.head_dim)), 'o_proj')


@mixers("attention")
@dataclasses.dataclass(frozen=True)
class AttentionMixer(MixerBase):
    """Grouped-query attention with optional image-block masking and M-RoPE.

    Geometry, norms and kernel policy come from the decoder context. Image
    bidirectionality, spatial rotary sections, NoPE and exclusive self
    attention configure this mixer only: a hybrid names them on its
    attention kind, `{"kind": "attention", "nope": true,
    "exclusive_self_attention": true}`, and a mixer that does not implement
    them has no field to take them.
    """

    bidirectional_images: bool = False
    mrope_section: tuple[int, int, int] | None = None
    nope: bool = False
    """No positional encoding: q and k enter the kernel unrotated and the
    logits keep their scale (lm-engine's `position_embedding_type="nope"`,
    Granite 4.0-H)."""
    exclusive_self_attention: bool = False
    """XSA (arXiv 2603.09078, lm-engine's `exclusive_self_attention`): each
    head's output loses its component along the token's own value
    (`exclusive_self_attention`)."""

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
            attention_chunk=ctx.attention_chunk,
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
            kv_cache=ctx.kv_cache,
            nope=self.nope,
            exclusive_self_attention=self.exclusive_self_attention,
            init_std=ctx.init_std,
            output_init_std=ctx.output_init_std,
            bidirectional_images=self.bidirectional_images, mrope_section=self.mrope_section)

