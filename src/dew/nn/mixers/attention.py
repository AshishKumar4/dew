"""Grouped-query causal attention as a mixer kind.

The `attention` value builds this module: grouped-query projections, rotary
positions, q/k norms and a fixed-size KV cache.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name
from jax.typing import DTypeLike

from dew.nn.attention import (
    RMSNorm,
    alibi_bias,
    cached_validity,
    causal_attention_mask,
    chunk_mask,
    combined_attention_mask,
    cudnn_runs,
    kernel_for_materialized_mask,
    local_attention,
    max_attention_logits,
    open_kv_cache,
    reference_only,
    scaled_dot_product_attention,
    with_documents,
)
from dew.nn.blocks import normal_kernel
from dew.nn.inputs import Admitted, AttentionMetadata
from dew.nn.kv_cache import TABLE, Append, KVCache, KVStore, filled_slots, rotated, write_cache
from dew.nn.mixer_base import MixerBase, MixerContext
from dew.nn.precision import at_least_fp32, scaled
from dew.nn.protocols import ProjectionGroup
from dew.nn.rope import (
    LongRopeScaling,
    RopeScaling,
    YarnScaling,
    axis_halves,
    axis_tables,
    inverse_frequencies,
    rotary_freqs,
    rotate,
    yarn_rope_freqs,
)
from dew.nn.sharding import HEADS, KV_HEADS, constrain, logical_axes


def temperature_scale(positions, floor_scale: float, attn_scale: float, *, dtype: DTypeLike):
    """Llama 4's query multiplier at each absolute position (arXiv 2501.19399),
    `log1p(floor((p + 1) / floor_scale)) * attn_scale + 1` in `dtype`, so the
    first `floor_scale` positions scale by 1."""
    positions = jnp.asarray(positions, dtype)
    return jnp.log1p(jnp.floor((positions + 1.0) / floor_scale)) * attn_scale + 1.0


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
class _Masking(NamedTuple):
    """`CausalSelfAttention._masking`'s result: the kernel's operands and
    the visibility they are read under."""

    query: jax.Array
    key: jax.Array
    value: jax.Array
    causal: bool
    window: int | None
    mask: jax.Array | None
    documents: jax.Array | None
    implementation: str
    cursor: jax.Array | None


class CausalSelfAttention(nn.Module):
    """Causal self-attention with grouped-query heads, rotary positions, qk
    RMSNorm and a fixed-size KV cache.

    decode=True runs against the cache: the first call writes the prompt and
    each later call appends, so prefill and decode are one path. Keys are
    rotated before they are cached, so rotary positions come from the cache
    index. causal=False is full attention, which a masked diffusion model reads
    its corrupted sequence with and an encoder reads its input with; with
    `bidirectional_window` a window keeps the keys within window - 1 positions
    on either side (`dew.nn.attention.window_sides`). Decoding attends a bidirectional canvas
    over the frozen prefix a causal prefill cached, and writes nothing back.
    kv_shared marks a layer without K/V projections (Gemma 3n/4 cross-layer
    sharing): it reads the keys, values and positions its provider stashed in
    `kv_store`, post rope and norm, and keeps no cache.
    """
    emb_features: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    value_head_dim: int | None = None
    """The value heads' width; None is `head_dim`. A narrower value is cached
    at its own width and widened only inside the kernel."""
    causal: bool = True
    rope_theta: float = 10000.0
    rope_scaling: RopeScaling | LongRopeScaling | None = None
    qk_norm: bool = True
    qk_norm_scope: str = 'head'  # 'head': one RMSNorm per head; 'projection': over the whole q/k
    qk_norm_weight: bool = True  # False: Llama 4's weightless L2 norm (Llama4TextL2Norm)
    v_norm: bool = False
    norm_eps: float = 1e-5
    scale_offset: bool = False
    scale_after_cast: bool = False
    kv_shared: bool = False
    kv_store_key: str | None = None
    sliding_window: int | None = None
    bidirectional_window: bool = False  # a non-causal layer keeps its window on both sides
    attention_chunk: int | None = None  # chunked local attention: keys sharing the query's position // chunk
    attention_bias: bool = False  # q/k/v biases, as config.attention_bias in HF
    o_proj_bias: bool | None = None  # None follows attention_bias; Qwen2 biases q/k/v only
    attention_scale: float | None = None  # None: the kernel's own 1/sqrt(head_dim)
    value_scale: float | None = None
    """Multiplies the values as projected, before the cache and the kernel
    read them (MiMo-V2-Flash's `attention_value_scale`)."""
    temperature_tuning: tuple[float, float] | None = None
    """Llama 4's (floor_scale, attn_scale): an unrotated layer scales each
    query by its position (`temperature_scale`)."""
    attention_dropout_rate: float = 0.0
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
    rotary_axes: tuple[int, ...] | None = None
    """The channel widths of a rotary over several axes (`dew.nn.rope.axis_tables`),
    a vision tower's patch grid, whose ids `attention_metadata.rotary_positions`
    carries `[B, S, len(rotary_axes)]`; None rotates by the positions."""
    rotary_pairs: Literal['half', 'adjacent'] = 'half'
    """The channels each angle turns (`dew.nn.rope.rotate`): the HF decoders'
    halves, or the adjacent pairs of Llama 4's vision tower."""
    rotary_per_axis: bool = False
    """Whether each of the `rotary_axes` turns its own channels in halves,
    Gemma 4's vision rope (`apply_multidimensional_rope`), where Qwen 3.5's
    turns the axes' angles together over the whole head."""
    packed: bool = False
    """Whether the queries, keys and values project through one kernel,
    `qkv_proj`, at init as from a checkpoint that stores them so (the Qwen 3.5
    and DeepSeek-V4.1 vision towers)."""
    linear: Callable[..., nn.Dense] = nn.Dense
    """The projections' module: Gemma 4's vision tower clips their inputs and
    outputs (`dew.nn.vision.gemma4.Gemma4ClippableLinear`)."""
    kv_cache: KVCache = KVCache()  # the decode cache's storage: dense or paged, full or quantized
    nope: bool = False
    """No positional encoding: q and k enter the kernel unrotated, and the
    logits keep their scale (lm-engine's `position_embedding_type="nope"`)."""
    alibi: bool = False
    """Linear key-position bias on unrotated queries and keys, as in BLOOM."""
    exclusive_self_attention: bool = False
    """XSA (arXiv 2603.09078): each head's output loses its component along
    the token's own value vector before the output projection."""
    init_std: float | None = None
    """Normal std of the q/k/v kernels; None keeps flax's lecun normal."""
    output_init_std: float | None = None
    """Normal std of o_proj; None follows init_std."""

    def setup(self):
        if self.alibi and not self.nope:
            raise ValueError('ALiBi requires unrotated attention (nope=True)')
        if self.rotary_per_axis and self.rotary_axes is None:
            raise ValueError('rotary_per_axis turns each of the rotary_axes in its own halves, '
                             'so it needs rotary_axes')
        if self.exclusive_self_attention and self.kv_shared:
            raise ValueError(
                "exclusive self attention subtracts the token's own value, and a "
                "KV-sharing layer projects none of its own")
        if self.value_dim != self.head_dim and (self.output_gate or self.k_eq_v):
            raise ValueError(
                f"values {self.value_dim} wide under {self.head_dim}-wide heads take no output gate "
                "and are not the keys")
        dense = functools.partial(
            self.linear, use_bias=self.attention_bias, dtype=self.dtype, precision=self.precision,
            **normal_kernel(self.init_std))
        # The gate doubles the query projection: the reference chunks its
        # output in half, one half the query and the other the gate the
        # branch multiplies by (modeling_qwen3_5.py:670-673, 701).
        query_width = self.num_heads * self.head_dim * (2 if self.output_gate else 1)
        if self._packed():
            # A Server packs its constant weights once, not on every decode
            # step. The ordinary parameter tree remains the training layout.
            self.qkv_proj = dense(query_width + self.num_kv_heads * (self.head_dim + self.value_dim),
                                  name='qkv_proj')
        else:
            self.q_proj = dense(query_width, name='q_proj')
        # A sharing layer reads another layer's keys and values, so it owns
        # no projections or key norm of its own, as the reference skips them
        # (modeling_gemma4.py, Gemma4TextAttention.__init__).
        if not self.kv_shared and not self._packed():
            self.k_proj = dense(self.num_kv_heads * self.head_dim, name='k_proj')
            if not self.k_eq_v:
                self.v_proj = dense(self.num_kv_heads * self.value_dim, name='v_proj')
        self.o_proj = dense(self.emb_features, name='o_proj', use_bias=(
            self.attention_bias if self.o_proj_bias is None else self.o_proj_bias),
            **normal_kernel(self.init_std if self.output_init_std is None else self.output_init_std))
        if self.qk_norm:
            if self.qk_norm_scope not in ('head', 'projection'):
                raise ValueError(
                    f"qk_norm_scope is 'head' or 'projection', got {self.qk_norm_scope!r}")
            norm = functools.partial(
                RMSNorm, epsilon=self.norm_eps, with_scale=self.qk_norm_weight,
                scale_offset=self.scale_offset, scale_after_cast=self.scale_after_cast, dtype=self.dtype)
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

    @property
    def reads_train(self) -> bool:
        """Whether its call takes `train` (`ReadsTrain`): it drops attention
        probabilities while training."""
        return bool(self.attention_dropout_rate)

    def _packed(self) -> bool:
        """Whether one `qkv_proj` holds the projections: the layer's own
        layout, or a server's packing of the three (`projection_groups`)."""
        return self.packed or self.has_variable('params', 'qkv_proj')

    @property
    def value_dim(self) -> int:
        """The width of one value head."""
        return self.head_dim if self.value_head_dim is None else self.value_head_dim

    def projection_groups(self) -> tuple[ProjectionGroup, ...]:
        """Its query, key and value projections packed as `qkv_proj`
        (`ProjectionSites`), which `setup` reads in their place, where it
        projects keys and values of its own and its variables hold them."""
        if self.kv_shared or self.k_eq_v:
            return ()
        query = self.num_heads * self.head_dim * (2 if self.output_gate else 1)
        group = ProjectionGroup(tuple(self.path), 'qkv_proj', ('q_proj', 'k_proj', 'v_proj'),
                                (query, self.num_kv_heads * self.head_dim,
                                 self.num_kv_heads * self.value_dim))
        return (group,) if group.held(self.variables.get('params', {})) else ()

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

    def _multimodal_rotary(self, positions: jax.Array, dtype: jnp.dtype):
        """Qwen's interleaved temporal/height/width rotary frequency selection,
        its angles in `dtype`."""
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
        # transformers' 1 / theta ** (2i / dim) (Qwen2VLRotaryEmbedding), on the host.
        inv = inverse_frequencies(self.rope_theta, rotated, dtype=dtype)
        angles = selected.astype(dtype) * inv
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

    def _projected_kv(self, x, whole: bool, projected=None):
        """Project the keys and values, norm them, and split them into heads.

        `whole` norms the key projection before the head split, which is
        OLMo 3's scope. Otherwise the key norm runs per head, after it.
        """
        batch, length, _ = x.shape
        if projected is None:
            key = self.k_proj(x)
            # Gemma 4's global layers use the raw key projection as the
            # values before its norm (modeling_gemma4.py).
            value = key if self.k_eq_v else self.v_proj(x)
        else:
            key, value = projected
        key = checkpoint_name(key, 'k_proj')
        if not self.k_eq_v:
            value = checkpoint_name(value, 'v_proj')
        value = value.reshape(batch, length, self.num_kv_heads, self.value_dim)
        if self.value_scale is not None:
            value = scaled(value, self.value_scale)
        if whole:
            key = self.k_norm(key)
        key = key.reshape(batch, length, self.num_kv_heads, self.head_dim)
        if self.qk_norm and not whole:
            key = self.k_norm(key)
        if self.v_norm:
            value = self.values_norm(value)
        return constrain(key, KV_HEADS), constrain(value, KV_HEADS)

    def _rotary_angles(self, rotary_positions, heads: jax.Array):
        """Build the rotary cos and sin this layer rotates its heads by, in
        the arithmetic `rotate` rotates `heads` in.

        Interleaved mRoPE, YaRN and the plain rope each build their own
        angles; YaRN rotates whole heads at its own frequencies, so it
        takes neither a partial rotary nor a Llama 3.1 ramp.
        """
        dtype = at_least_fp32(heads.dtype)
        if self.rotary_axes is not None:
            if rotary_positions is None or rotary_positions.ndim != 3:
                raise ValueError(f"a rotary over {len(self.rotary_axes)} axes reads their ids "
                                 "[B, S, axes] from attention_metadata.rotary_positions")
            return axis_tables(rotary_positions, self.rotary_axes, self.rope_theta, dtype=dtype)
        if self.mrope_section is not None and rotary_positions is not None and rotary_positions.ndim == 3:
            return self._multimodal_rotary(rotary_positions, dtype)
        if self.yarn is None:
            return rotary_freqs(
                rotary_positions, self.head_dim, self.rope_theta, rot_dim=self._rot_dim(),
                partial_rotary_type=self.partial_rotary_type, rope_scaling=self.rope_scaling,
                dtype=dtype)
        if self.partial_rotary_factor is not None or self.rope_scaling is not None:
            raise ValueError(
                "yarn rotates whole heads at its own frequencies, so it takes "
                "neither partial_rotary_factor nor rope_scaling")
        return yarn_rope_freqs(rotary_positions, self.head_dim, self.rope_theta, self.yarn, dtype=dtype)

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
        valid = (cached_validity(self, key_length) if decode
                 else None if metadata is None else metadata.valid)
        keep = (causal_attention_mask(query_slots, key_length)
                if self.causal else jnp.ones((batch, 1, length, key_length), bool))
        if self.bidirectional_images:
            same_image = ((groups[:, :, None] == key_groups[:, None, :])
                          & (groups[:, :, None] >= 0))
            keep = keep | same_image[:, None]
        if self.sliding_window is not None and (self.causal or self.bidirectional_window):
            distance = query_slots[:, None, :, None] - jnp.arange(key_length)[None, None, None, :]
            # A causal window bounds only the keys behind the query, so an
            # image's later tokens stay visible to its earlier ones; a
            # two-sided one bounds both (`window_sides`).
            keep = keep & ((distance < self.sliding_window) if self.causal
                           else (jnp.abs(distance) < self.sliding_window))
        if valid is not None:
            keep = keep & valid[:, None, None, :]
        if decode:
            keep = keep & (query_slots[:, None, :, None] >= 0)
        return keep

    def _restricts_visibility(self, metadata: AttentionMetadata | None, decode: bool) -> bool:
        """Whether metadata narrows who sees whom, so the mask has to be built.

        Key validity does, and image groups on a bidirectional-image layer; rotary
        positions and an explicit pairwise mask do not. Otherwise causality and the
        window stay flags, since a materialized [B, 1, S, S] mask sends the call to
        the xla kernel (docs/performance.md). A validity array is opaque at trace
        time, so a host that knows a row is unpadded passes none. Decoding always
        builds the mask, which carries the cache's validity.
        """
        if decode:
            return metadata is not None or self.bidirectional_images
        return metadata is not None and (
            metadata.valid is not None
            or (self.bidirectional_images and metadata.image_groups is not None))

    @nn.compact
    def __call__(self, x, decode: bool = False,
                 positions=None, segment_ids=None, kv_store=None,
                 attention_metadata: AttentionMetadata | None = None, train: bool = False):
        B, S, _ = x.shape
        logical_positions = positions
        # The projections and the kernel's output carry the names a remat
        # policy saves or offloads (decoder_block.RESIDUALS).
        projected_kv = None
        if self._packed():
            width = self.num_heads * self.head_dim * (2 if self.output_gate else 1)
            projected, key, value = jnp.split(
                self.qkv_proj(x), (width, width + self.num_kv_heads * self.head_dim), axis=-1)
            projected_kv = (key, value)
        else:
            projected = self.q_proj(x)
        projected = checkpoint_name(projected, 'q_proj')
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
            key, value = self._projected_kv(x, whole, projected_kv)
        if self.qk_norm and not whole:
            query = self.q_norm(query)
        # Column-parallel under a tensor axis, each shard a run of whole
        # heads; o_proj's sum returns to the residual placement in the block.
        query = constrain(query, HEADS)

        kv_len = key.shape[-3]
        positions, append, prefix = self._step_positions(key, positions, segment_ids, attention_metadata,
                                                         decode, S)
        rotary_positions = positions if logical_positions is None else logical_positions
        if attention_metadata is not None and attention_metadata.rotary_positions is not None:
            rotary_positions = attention_metadata.rotary_positions
        freqs_cos = freqs_sin = None
        if self.nope:
            # NoPE rotates nothing; the query still carries the logit scale
            # the checkpoint asks for, which rotate folds in otherwise.
            if self.attention_scale is not None:
                query = scaled(query, self.attention_scale * math.sqrt(self.head_dim))
            if self.temperature_tuning is not None:
                scale = temperature_scale(positions, *self.temperature_tuning,
                                          dtype=at_least_fp32(query.dtype))
                query = query * (scale[..., None, None] if scale.ndim == 2
                                 else scale[None, :, None, None]).astype(query.dtype)
        else:
            freqs_cos, freqs_sin = self._rotary_angles(rotary_positions, query)
            # Every kernel path scales the logits by 1/sqrt(head_dim) itself, so the
            # query carries the ratio to the scale the checkpoint asks for.
            query = rotate(
                self._turned(query), freqs_cos, freqs_sin, pairs=self.rotary_pairs,
                scale=(None if self.attention_scale is None
                       else self.attention_scale * math.sqrt(self.head_dim)))
        own_value = value
        if not self.kv_shared:
            if freqs_cos is not None and freqs_sin is not None:
                key = rotate(self._turned(key), freqs_cos, freqs_sin, pairs=self.rotary_pairs)
            if kv_store is not None and self.kv_store_key is not None:
                # Post-norm, post-rope, the same tensors the reference hands
                # its sharing layers (modeling_gemma4.py, Gemma4TextAttention).
                kv_store[self.kv_store_key] = (key, value, positions)
        if decode and attention_metadata is not None and attention_metadata.admitted is not None:
            attention = self._mixed_attended(query, key, value, positions, attention_metadata.admitted)
            return self._output(attention, gate, B, S, own_value)
        sinks = (self.param('sinks', nn.initializers.zeros, (self.num_heads,))
                 if self.attention_sinks else None)
        if self._runs_local(attention_metadata, decode) and not (self.attention_dropout_rate and train):
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
        masking = self._masking(query, key, value, positions, rotary_positions, append, prefix, kv_len,
                                kv_store, segment_ids, attention_metadata, decode)
        bias = None
        if self.alibi:
            if decode:
                key_positions = jnp.arange(masking.key.shape[1])
            elif attention_metadata is not None and attention_metadata.key_positions is not None:
                key_positions = attention_metadata.key_positions
            elif attention_metadata is not None and attention_metadata.valid is not None:
                valid = attention_metadata.valid
                key_positions = (jnp.cumsum(valid, axis=-1) - 1) * valid
            else:
                key_positions = positions
            bias = alibi_bias(key_positions, self.num_heads, dtype=query.dtype)
        if attention_metadata is not None and attention_metadata.position_bias is not None:
            position_bias = attention_metadata.position_bias
            bias = position_bias if bias is None else bias + position_bias
        attention = self._attended(masking, positions, append, sinks, train, bias=bias)
        return self._output(attention, gate, B, S, own_value)

    def _turned(self, heads):
        """Query or key heads in the layout their rotation reads: with
        `rotary_per_axis`, each axis's halves gathered into the head's two
        (`dew.nn.rope.axis_halves`), which moves query and key alike."""
        if not self.rotary_per_axis:
            return heads
        assert self.rotary_axes is not None  # setup refuses per-axis halves without axes
        return axis_halves(heads, self.rotary_axes)

    def _step_positions(self, key, positions, segment_ids, attention_metadata: AttentionMetadata | None,
                        decode: bool, S: int):
        """The positions the rotation and the mask read, the cache append that
        writes a decode step's keys, and a bidirectional canvas's frozen encoder
        prefix. Decoding reads positions off the cache slot; a packed batch supplies
        each token's position in its document, so RoPE restarts at boundaries.
        """
        append = None
        prefix = None
        if decode and attention_metadata is not None and attention_metadata.admitted is not None:
            return self._mixed_positions(attention_metadata.admitted, attention_metadata.valid), None, None
        if decode:
            if self.causal:
                if not self.kv_shared:
                    positions, append = open_kv_cache(
                        self, key, self.max_seq_len, value_dim=self.value_dim,
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
        return positions, append, prefix

    def _mask_kernel(self, query, decode: bool) -> str:
        """The kernel a call that builds its mask runs: xla
        (`kernel_for_materialized_mask`), except a cache call over several
        queries (a prefill, a verified draft) under 'auto' or 'cudnn' where
        `cudnn_runs` (which also keeps the deterministic-ops lane off it); an
        explicit 'xla' stays xla. That call runs
        forward only, where cuDNN takes the mask as its additive bias with
        none of the backward's refusals: 8 prompts of 256 at Qwen3-0.6B's
        widths took 4.0 against xla's 6.2 ms over 28 layers on an RTX 4080.
        It rounds elsewhere, so the prefill is not bitwise xla's: its RMS
        distance from an fp32 forward is 1.04 times xla's on Qwen3-0.6B and
        1.00 on 1.7B (tests/reference_error.py allows 2), and greedy rows part
        only at bf16 near-ties (docs/performance.md)."""
        masked = kernel_for_materialized_mask(
            self.attention_impl, query, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax)
        if (decode and query.shape[1] > 1 and masked == 'xla' and self.attention_impl in ('auto', 'cudnn')
                and cudnn_runs(query, self.attn_logit_softcap)):
            return 'cudnn'
        return masked

    def _masking(
        self,
        query,
        key,
        value,
        positions,
        rotary_positions,
        append,
        prefix,
        kv_len: int,
        kv_store,
        segment_ids,
        attention_metadata: AttentionMetadata | None,
        decode: bool,
    ) -> _Masking:
        """What the kernel reads beside the rotated query and keys: the keys
        and values with the cache's or the prefix's joined, and the causal
        flag, window, mask, document ids and kernel the layer's visibility
        comes to (the canvas prefix, a shared or cached decode step, packed
        documents, metadata, a pairwise mask, a chunk)."""
        B, S = query.shape[:2]
        causal, mask, documents = self.causal, None, None
        implementation = self.attention_impl
        masked = self._mask_kernel(query, decode)
        # A bidirectional layer windows only where its kind keeps the window on
        # both sides; DiffusionGemma's decoder reads its whole canvas
        # (modeling_diffusion_gemma.py:1399-1401), and its prefix branch below
        # bounds the cached keys itself.
        window = (None if decode or not (causal or self.bidirectional_window)
                  else self.sliding_window)
        cursor = None  # the decode mask the paged kernel stands in for, when this call builds it
        if prefix is not None:
            # Every canvas query reads the same retained encoder keys and all
            # canvas keys (modeling_diffusion_gemma.py:1399-1401). A local
            # layer retains the last window-1 prefix keys.
            cached_key = self.get_variable("cache", "cached_key")
            cached_value = self.get_variable("cache", "cached_value")
            alloc = cached_key.shape[-3]
            prefix_slots = jnp.arange(alloc)[None, :]
            valid_prefix = cached_validity(self, alloc)
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
            # from them (`attention_kernel`).
            documents = segment_ids
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
        return _Masking(query, key, value, causal, window, mask, documents, implementation, cursor)

    def _attended(self, masking: _Masking, positions, append, sinks, train: bool, *, bias=None):
        """The attention output through the kernel the masking chose: the
        paged kernel or a per-row key count for a plain decode step, the
        general kernel otherwise, with the per-head logit maxima sown for
        QK-Clip when a caller opened that collection."""
        query, key, value, causal, window, mask, documents, implementation, cursor = masking
        B, S = query.shape[:2]
        # The per-head maxima the QK-Clip reads. Computed only when a caller
        # opened the collection; the plain forward leaves it closed and its
        # leaves bitwise identical.
        sowing = not self.is_initializing() and self.is_mutable_collection("qk")
        if sowing:
            self.sow("qk", "max_logits", max_attention_logits(
                query, key, causal=causal, sliding_window=window,
                mask=mask if documents is None else with_documents(mask, documents), bias=bias))
            self.sow("qk", "kv_heads", jnp.asarray(key.shape[-2]))
            self.sow("qk", "head_dim", jnp.asarray(query.shape[-1]))
        if self.attention_dropout_rate and train:
            return checkpoint_name(scaled_dot_product_attention(
                query, key, value, dtype=self.dtype, precision=self.precision,
                force_fp32_for_softmax=self.force_fp32_for_softmax,
                implementation=self.attention_impl, causal=causal, sliding_window=window,
                mask=mask, bias=bias, sinks=sinks, softcap=self.attn_logit_softcap, segment_ids=documents,
                dropout_rate=self.attention_dropout_rate, dropout_rng=self.make_rng("dropout"),
                deterministic=False), 'context')
        # A chunk, window or metadata mask replaces `cursor` and keeps the gather.
        plain_step = (append is not None and mask is cursor and S == 1 and sinks is None
                      and bias is None and not sowing)
        if append is not None and plain_step and self._page_kernel_runs(query) and append.store.kernel():
            attention = self._paged(append, query)
        elif plain_step:
            attention = self._decode_attention(query, key, value, jnp.asarray(positions).reshape(B, S)[:, 0])
        else:
            attention = checkpoint_name(scaled_dot_product_attention(
                query, key, value, dtype=self.dtype, precision=self.precision,
                force_fp32_for_softmax=self.force_fp32_for_softmax,
                implementation=implementation, causal=causal,
                sliding_window=window, mask=mask, bias=bias, sinks=sinks,
                softcap=self.attn_logit_softcap, segment_ids=documents), 'context')
        return attention

    def mixed_refusal(self) -> str | None:
        """Why this layer cannot run a serving step's mixed call
        (`dew.nn.inputs.Admitted`), or None: it reads a full-precision cache,
        dense or one page pool, whole and causal, and nothing a row carries
        besides."""
        layout = self.kv_cache
        refusals = {
            "a quantized or rotated cache": (layout.quantized is not None
                                              or layout.key_rotation(self.head_dim) is not None),
            "a page pool split into groups": layout.page_size is not None and layout.groups != 1,
            "a sliding window or chunk": self.sliding_window is not None or self.attention_chunk is not None,
            "attention sinks": self.attention_sinks,
            "ALiBi": self.alibi,
            "keys shared from another layer": self.kv_shared,
            "bidirectional attention or image groups": not self.causal or self.bidirectional_images,
        }
        named = [reason for reason, refused in refusals.items() if refused]
        return None if not named else f"{self.name}: " + ", ".join(named)

    def _mixed_positions(self, admitted: Admitted, valid) -> jax.Array:
        """Each token's slot in its row, `[1, tokens]`, -1 for padding: a
        decoding row's token at its cursor, a prompt's after its row's start."""
        refusal = self.mixed_refusal()
        if refusal is not None:
            raise ValueError(f"a mixed serving step needs plain attention; {refusal}")
        index = self.get_variable("cache", "cache_index")
        rows, pieces = index.shape[0], admitted.slots.shape[0]
        valid = jnp.asarray(valid, bool)[0]
        prompt = valid[rows:].reshape(pieces, -1)
        filled = admitted.cursors[:, None] + jnp.cumsum(prompt, axis=1, dtype=jnp.int32) - 1
        positions = jnp.concatenate([jnp.where(valid[:rows], index, -1),
                                     jnp.where(prompt, filled, -1).reshape(-1)])
        return positions[None]

    def _mixed_attended(self, query, key, value, positions, admitted: Admitted):
        """A mixed call's attention: every valid token's keys and values go
        into its row of the cache, then each decoding row's query reads its
        row as a decode step does, and each prompt's queries read their row
        up to their own slot."""
        index = self.variable("cache", "cache_index", jnp.zeros, (0,), jnp.int32)
        store = KVStore.open(self, self.kv_cache, index.value.shape[0], self.max_seq_len,
                             self.num_kv_heads, self.head_dim, key.dtype, value_dim=self.value_dim)
        rows, pieces = index.value.shape[0], admitted.slots.shape[0]
        width = (query.shape[1] - rows) // pieces
        positions = positions[0]
        if admitted.tables is not None:
            # A paged row reads and writes through the pages its admission assigned.
            self.put_variable("cache", TABLE, self.get_variable("cache", TABLE).at[admitted.slots].set(
                admitted.tables, mode="drop"))
        token_rows = jnp.concatenate([jnp.arange(rows), jnp.repeat(admitted.slots, width)])
        store.write_tokens(key[0], value[0], token_rows, positions)
        prompt = positions[rows:].reshape(pieces, width)
        index.value = (index.value + (positions[:rows] >= 0)).at[admitted.slots].set(
            admitted.cursors + jnp.sum(prompt >= 0, axis=1, dtype=jnp.int32), mode="drop")
        if self._page_kernel_runs(query) and store.kernel():
            decoded = store.decode(query[0, :rows], index.value, self.attn_logit_softcap)
            decoded = checkpoint_name(decoded[:, None], 'context')
        else:
            decoded = self._decode_attention(query[0, :rows, None], *store.read(), positions[:rows])
        queries = query[0, rows:].reshape(pieces, width, *query.shape[2:])
        if admitted.continuing:
            # A padding piece's row is past the cache's; it reads row 0 and is never drawn from.
            held = jnp.where(admitted.slots < rows, admitted.slots, 0)
            keys, values = store.read_rows(held)
            keep = causal_attention_mask(prompt, self.max_seq_len,
                                         key_valid=filled_slots(index.value[held], self.max_seq_len))
        else:
            # A piece that starts its row reads only its own keys, in its own order.
            keys, values = (x[0, rows:].reshape(pieces, width, *x.shape[2:]) for x in (key, value))
            keep = ((prompt[:, None, :, None] >= prompt[:, None, None, :]) & (prompt >= 0)[:, None, None, :])
        # A padding query reads one key, so its (unused) output stays finite.
        keep = keep | ((prompt < 0)[:, None, :, None] & (jnp.arange(keep.shape[-1]) == 0))
        prefilled = scaled_dot_product_attention(
            queries, keys, values, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            implementation=self._mask_kernel(queries, decode=True), causal=False, mask=keep,
            softcap=self.attn_logit_softcap)
        return jnp.concatenate([decoded.reshape(1, rows, *decoded.shape[2:]),
                                prefilled.reshape(1, -1, *prefilled.shape[2:])], axis=1)

    def _decode_attention(self, query, key, value, positions):
        """One query a row `[rows, 1, heads, head_dim]` over its row of the
        cache, whose token sits at `positions` `[rows]` (-1 for padding).

        The cache is compact, so a decode query reads the filled slots
        before its own: a key count per row, which cuDNN takes as its
        padding lengths (`scaled_dot_product_attention`) and every other
        kernel builds `cursor` from again. With the mask built here, 'auto'
        went to xla's two dense dots over the whole cache: 1.00 against
        cuDNN's 0.30 ms a layer at 128 rows on an RTX 4080. A row whose
        query is padding reads one key, so its (unused) output stays finite
        as the masked kernels leave it.
        """
        rows = query.shape[0]
        reads = jnp.maximum(positions + 1, 1)
        # Each group's query heads read the same keys, so they go in as
        # that key head's query positions: one pass over the group's keys
        # where cuDNN otherwise padded the lone query to two and ran each
        # query head on its own, 4.7% less a layer at 64 rows and the
        # same bits (docs/performance.md).
        heads, kv_heads = query.shape[-2], key.shape[-2]
        group = heads // kv_heads
        grouped = query.reshape(rows, kv_heads, group, query.shape[-1]).transpose(0, 2, 1, 3)
        attention = scaled_dot_product_attention(
            grouped, key, value, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            implementation=self.attention_impl, causal=False,
            softcap=self.attn_logit_softcap,
            key_value_seq_lengths=reads.astype(jnp.int32))
        return checkpoint_name(
            attention.transpose(0, 2, 1, 3).reshape(rows, 1, heads, attention.shape[-1]), 'context')

    def _page_kernel_runs(self, query) -> bool:
        """Whether a decode step may read a paged pool through its device's
        paged kernel (`KVStore.decode`) rather than the gathered rows. The
        paged kernels read keys and values of one width, so narrower values
        take the gathered rows, which the attention core widens."""
        if self.value_dim != self.head_dim:
            return False
        if jax.default_backend() != 'gpu':
            return self.attention_impl in ('auto', 'tpu')
        return (self.attention_impl in ('auto', 'cudnn')
                and cudnn_runs(query, self.attn_logit_softcap)
                and not reference_only(query, self.dtype, self.precision, self.force_fp32_for_softmax))

    def _decode_masks(self, positions, key_length: int) -> tuple[jax.Array, jax.Array]:
        """The cursor mask over the cache's filled slots, and the layer's decode mask.

        They are the same array exactly when the layer narrows nothing, which
        is when the paged kernel, attending every filled slot, may stand in
        for the mask; a window builds a mask of its own.
        """
        valid = cached_validity(self, key_length)
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
        if self.alibi or decode or (not self.is_initializing() and self.is_mutable_collection("qk")):
            return False
        return metadata is None or (
            metadata.pairwise_mask is None and metadata.position_bias is None
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
            self.o_proj(attention.reshape(batch, length, self.num_heads * self.value_dim)), 'o_proj')


@dataclasses.dataclass(frozen=True)
class AttentionMixer(MixerBase):
    """Grouped-query attention with optional image-block masking and M-RoPE.

    Geometry, norms and kernel policy come from the decoder context. Image
    bidirectionality, spatial rotary sections, NoPE and exclusive self attention
    are this mixer's own fields, which a hybrid names on its attention kind
    (`{"name": "attention", "fields": {"nope": true}}`).
    """

    bidirectional_images: bool = False
    mrope_section: tuple[int, int, int] | None = None
    nope: bool = False
    """No positional encoding: q and k enter the kernel unrotated and the
    logits keep their scale (lm-engine's `position_embedding_type="nope"`,
    Granite 4.0-H)."""
    alibi: bool = False
    """Per-head linear key-position bias; requires unrotated attention."""
    exclusive_self_attention: bool = False
    """XSA (arXiv 2603.09078, lm-engine's `exclusive_self_attention`): each
    head's output loses its component along the token's own value
    (`exclusive_self_attention`)."""

    # `CausalSelfAttention` reads the mixed call's `Admitted` layout.
    mixed_step = True

    def __post_init__(self):
        if self.alibi and not self.nope:
            raise ValueError("ALiBi requires unrotated attention (nope=True)")
        if self.mrope_section is not None:
            object.__setattr__(self, "mrope_section", tuple(self.mrope_section))
            if len(self.mrope_section) != 3 or any(value < 0 for value in self.mrope_section):
                raise ValueError("mrope_section must contain three nonnegative section widths")

    def build(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        # The context's fields and this kind's are CausalSelfAttention's, by name.
        context = {field.name: getattr(ctx, field.name) for field in dataclasses.fields(ctx)}
        kind = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        return functools.partial(CausalSelfAttention, **context, **kind)

