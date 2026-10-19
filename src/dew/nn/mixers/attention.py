"""Grouped-query causal attention as a mixer kind.

The `attention` value builds this module: grouped-query projections, rotary
positions, q/k norms and a fixed-size KV cache.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable
from typing import ClassVar, Literal, NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
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
    cudnn_takes_mask,
    kernel_for_materialized_mask,
    max_attention_logits,
    open_kv_cache,
    reference_only,
    scaled_dot_product_attention,
)
from dew.nn.blocks import normal_kernel
from dew.nn.documents import CrossDocuments, cross_document_mask, with_documents
from dew.nn.flops import attention_flops, projection_flops
from dew.nn.inputs import (
    PLAIN_CALL,
    Admitted,
    AttentionMetadata,
    Coordinates,
    MixerCall,
    coordinates_of,
    counted_places,
    document_reach,
)
from dew.nn.kernels.generation import backend
from dew.nn.kv_cache import TABLE, Append, KVCache, KVStore, filled_slots, rotated, write_cache
from dew.nn.local_attention import local_attention
from dew.nn.mixer_base import MixerBase, MixerContext
from dew.nn.ngpt import NGPT, gain, unit
from dew.nn.precision import at_least_fp32, rounded_to, scaled
from dew.nn.protocols import ProjectionGroup
from dew.nn.rope import (
    LongRopeScaling,
    axis_halves,
    axis_tables,
    mrope_tables,
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


def held_reads(index: jax.Array, positions: jax.Array) -> jax.Array:
    """The keys each row's decode query reads from a paged cache: the row's
    filled slots `index` `[rows]`, or one where its query is padding
    (`positions` -1), as the dense path's per-row key count reads. A freed
    row keeps its last request's count until the next one is admitted, and
    the paged kernel reads every key under it: on an A100 at 128 slots and
    32 requests a second, with 40 to 50 rows decoding, that was 5.1 ms of
    attention a step against cuDNN's 2.0 over the dense cache."""
    return jnp.where(jnp.asarray(positions) >= 0, index, 1)


def mixed_attention_refusal(ctx: MixerContext, *, alibi: bool, bidirectional_images: bool) -> str | None:
    """Why attention over `ctx` cannot run a serving step's mixed call
    (`dew.nn.inputs.Admitted`), or None: it reads a full-precision cache,
    dense or one page pool, whole and causal, and nothing a row carries
    besides."""
    layout = ctx.kv_cache
    refusals = {
        "a quantized or rotated cache": (layout.quantized is not None
                                          or layout.key_rotation(ctx.head_dim) is not None),
        "a page pool split into groups": layout.page_size is not None and layout.groups != 1,
        "a sliding window or chunk": ctx.sliding_window is not None or ctx.attention_chunk is not None,
        "attention sinks": ctx.attention_sinks,
        "ALiBi": alibi,
        "keys shared from another layer": ctx.kv_shared,
        "bidirectional attention or image groups": not ctx.causal or bidirectional_images,
    }
    named = [reason for reason, refused in refusals.items() if refused]
    return None if not named else ", ".join(named)


class _Masking(NamedTuple):
    """`CausalSelfAttention._masking`'s result: the kernel's operands and
    the visibility they are read under."""

    query: jax.Array
    key: jax.Array
    value: jax.Array
    causal: bool
    window: int | None
    mask: jax.Array | None
    documents: jax.Array | CrossDocuments | None
    """A call's packed documents, or a cache call's (`CausalSelfAttention._cached_documents`)."""
    max_segments: int | None
    """The most documents a row of `documents` holds (`Coordinates.max_segments`)."""
    implementation: str
    cursor: jax.Array | None
    key_places: jax.Array
    """Each key's place counted over its document's tokens (`counted_places`), which a chunk and ALiBi
    read: a cached key's where the layer keeps them (`CausalSelfAttention._cached_places`), else
    its slot."""


@logical_axes({
    ("q_proj",): ("embed", "heads"),
    ("k_proj",): ("embed", "kv"),
    ("v_proj",): ("embed", "kv"),
    # The output gate's own projection (`gate_projection='own'`) is per head
    # like the query's; the longer suffix outranks the MLP's `gate_proj`.
    ("self_attn", "gate_proj"): ("embed", "heads"),
    ("o_proj",): ("attention", "embed"),
})
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
    sharing): it reads the keys, values and places its provider stashed in
    `kv_store`, post rope and norm, and keeps no cache.
    """
    reads: ClassVar[frozenset[str]] = frozenset({"pairwise_mask", "key_positions", "position_bias",
                                                 "admitted"})
    """Every `MIXING` field of its call (`dew.nn.inputs.MixerCall`): it masks by a pairwise mask,
    places keys, adds a position bias and runs a server's mixed step."""

    ctx: MixerContext
    """The layer's geometry and attention policies, as the backbone resolves
    them for its kind (`MixerContext`)."""
    qk_norm_weight: bool = True  # False: Llama 4's weightless L2 norm (Llama4TextL2Norm)
    temperature_tuning: tuple[float, float] | None = None
    """Llama 4's (floor_scale, attn_scale): an unrotated layer scales each
    query by its position (`temperature_scale`)."""
    gate_projection: Literal['query', 'own'] = 'query'
    """Where the output gate comes from: the doubled query projection
    (Qwen3.5), or a `gate_proj` of its own over the layer's input (Muse
    Glimmer, modeling_muse_glimmer.py:305, 351). Either way the branch
    multiplies by its sigmoid."""
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
    nope: bool = False
    """No positional encoding: q and k enter the kernel unrotated, and the
    logits keep their scale (lm-engine's `position_embedding_type="nope"`)."""
    alibi: bool = False
    """Linear key-position bias added to the logits, after any rotation of the
    queries and keys; BLOOM's are unrotated (`nope`)."""
    exclusive_self_attention: bool = False
    """XSA (arXiv 2603.09078): each head's output loses its component along
    the token's own value vector before the output projection."""

    def setup(self):
        if self.rotary_per_axis and self.rotary_axes is None:
            raise ValueError('rotary_per_axis turns each of the rotary_axes in its own halves, '
                             'so it needs rotary_axes')
        if self.exclusive_self_attention and self.ctx.kv_shared:
            raise ValueError(
                "exclusive self attention subtracts the token's own value, and a "
                "KV-sharing layer projects none of its own")
        if self.value_dim != self.ctx.head_dim and (self.ctx.output_gate or self.ctx.k_eq_v):
            raise ValueError(
                f"values {self.value_dim} wide under {self.ctx.head_dim}-wide heads take no output gate "
                "and are not the keys")
        dense = functools.partial(
            self.linear, use_bias=self.ctx.attention_bias, dtype=self.ctx.dtype, precision=self.ctx.precision,
            **normal_kernel(self.ctx.init_std))
        projections = self.projections()
        if self._packed():
            # A Server packs its constant weights once, not on every decode
            # step. The ordinary parameter tree remains the training layout.
            group = self._group()
            if group is None:
                raise ValueError("a packed qkv_proj holds the layer's own keys and values, which a "
                                 "KV-sharing or keys-are-values layer does not project")
            self.qkv_proj = dense(sum(group.widths), name=group.packed)
        else:
            self.q_proj = dense(projections['q_proj'][1], name='q_proj')
        # A sharing layer reads another layer's keys and values, so it owns
        # no projections or key norm of its own, as the reference skips them
        # (modeling_gemma4.py, Gemma4TextAttention.__init__).
        if not self.ctx.kv_shared and not self._packed():
            self.k_proj = dense(projections['k_proj'][1], name='k_proj')
            if not self.ctx.k_eq_v:
                self.v_proj = dense(projections['v_proj'][1], name='v_proj')
        if 'gate_proj' in projections:
            self.gate_proj = dense(projections['gate_proj'][1], name='gate_proj', use_bias=False)
        self.o_proj = dense(projections['o_proj'][1], name='o_proj', use_bias=(
            self.ctx.attention_bias if self.ctx.o_proj_bias is None else self.ctx.o_proj_bias),
            **normal_kernel(self.ctx.init_std if self.ctx.output_init_std is None
                            else self.ctx.output_init_std))
        if self.ctx.qk_norm:
            if self.ctx.qk_norm_scope not in ('head', 'projection'):
                raise ValueError(
                    f"qk_norm_scope is 'head' or 'projection', got {self.ctx.qk_norm_scope!r}")
            norm = functools.partial(
                RMSNorm, epsilon=self.ctx.norm_eps, with_scale=self.qk_norm_weight,
                scale_offset=self.ctx.scale_offset, scale_after_cast=self.ctx.scale_after_cast,
                dtype=self.ctx.dtype)
            self.q_norm = norm(name='q_norm')
            if not self.ctx.kv_shared:
                self.k_norm = norm(name='k_norm')
        if self.ctx.ngpt is not None:
            self._ngpt_gains(self.ctx.ngpt)
        if self.ctx.v_norm and not self.ctx.kv_shared:
            # Gemma 4 norms the values with a scale-free RMSNorm before they
            # are cached or shared (modeling_gemma4.py, Gemma4TextAttention).
            # The attribute cannot share the field's name, and it holds no
            # parameters either way.
            self.values_norm = RMSNorm(epsilon=self.ctx.norm_eps, with_scale=False,
                                       dtype=self.ctx.dtype, name='v_norm')

    def _ngpt_gains(self, ngpt: NGPT):
        """nGPT's query gain and, under an output gate, the gate's logit gain, one a feature of the
        heads, each stored from its initializer's scale (`NGPT.start`)."""
        if self.ctx.qk_norm or self.ctx.kv_shared or self.ctx.k_eq_v or self.nope:
            raise ValueError("an nGPT attention unit-norms its own rotated queries and keys: it takes no "
                             "q/k norm, shared or value keys, or unrotated heads")
        width = self.ctx.num_heads * self.ctx.head_dim
        self.sqk = self.param('sqk', nn.initializers.constant(ngpt.start), (width,), jnp.float32)
        if self.ctx.output_gate:
            if self.gate_projection != 'own':
                raise ValueError("an nGPT attention gates its output by a projection of its own "
                                 "(gate_projection='own')")
            self.s_gate = self.param('s_gate', nn.initializers.constant(ngpt.start), (width,), jnp.float32)

    def _ngpt_rotated(self, ngpt: NGPT, query, key):
        """The rotated query and key heads on the unit sphere, the query times the square of its
        learned gain, in fp32 and back to its dtype (dory.py:139-149)."""
        heads = gain(self.sqk, ngpt.query(self.ctx.head_dim))
        heads = heads.reshape(self.ctx.num_heads, self.ctx.head_dim)
        query = (jnp.square(heads) * unit(query).astype(jnp.float32)).astype(query.dtype)
        return query, unit(key)

    @property
    def reads_train(self) -> bool:
        """Whether its call takes `train` (`ReadsTrain`): it drops attention
        probabilities while training."""
        return bool(self.ctx.attention_dropout_rate)

    def _packed(self) -> bool:
        """Whether one `qkv_proj` holds the projections: the layer's own
        layout, or a server's packing of the three (`projection_groups`)."""
        return self.packed or self.has_variable('params', 'qkv_proj')

    @property
    def value_dim(self) -> int:
        """The width of one value head."""
        return self.ctx.head_dim if self.ctx.value_head_dim is None else self.ctx.value_head_dim

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """Query/output maps and unshared K/V maps, plus QK and AV."""
        window = self.ctx.sliding_window if self.ctx.causal or self.ctx.bidirectional_window else None
        return (sum(projection_flops(lengths, *shape) for shape in self.projections().values())
                + attention_flops(lengths, self.ctx.num_heads, self.ctx.head_dim, self.value_dim,
                                  causal=self.ctx.causal, window=window, chunk=self.ctx.attention_chunk))

    @nn.nowrap
    def parameter_flops(self, lengths) -> float:
        """Learned QKV/output/gate maps, excluding data-dependent attention products."""
        window = self.ctx.sliding_window if self.ctx.causal or self.ctx.bidirectional_window else None
        products = attention_flops(lengths, self.ctx.num_heads, self.ctx.head_dim, self.value_dim,
                                   causal=self.ctx.causal, window=window, chunk=self.ctx.attention_chunk)
        return self.model_flops(lengths) - products

    @nn.nowrap
    def projections(self) -> dict[str, tuple[int, int]]:
        """The attention's maps as `{name: (inputs, outputs)}`, keys and
        values absent where the layer shares them or the keys are the values:
        what setup builds, the count reads and the packed group
        (`_group`) concatenates."""
        # The gate doubles the query projection: the reference chunks its
        # output in half, one half the query and the other the gate the
        # branch multiplies by (modeling_qwen3_5.py:670-673, 701).
        doubled = self.ctx.output_gate and self.gate_projection == 'query'
        query = self.ctx.num_heads * self.ctx.head_dim * (2 if doubled else 1)
        sites = {'q_proj': (self.ctx.emb_features, query),
                 'o_proj': (self.ctx.num_heads * self.value_dim, self.ctx.emb_features)}
        if self.ctx.output_gate and self.gate_projection == 'own':
            sites['gate_proj'] = (self.ctx.emb_features, self.ctx.num_heads * self.ctx.head_dim)
        if not self.ctx.kv_shared:
            sites['k_proj'] = (self.ctx.emb_features, self.ctx.num_kv_heads * self.ctx.head_dim)
            if not self.ctx.k_eq_v:
                sites['v_proj'] = (self.ctx.emb_features, self.ctx.num_kv_heads * self.value_dim)
        return sites

    def _group(self) -> ProjectionGroup | None:
        """Its query, key and value projections of the layer's input, packed
        as `qkv_proj`, where it projects keys and values of its own."""
        if self.ctx.kv_shared or self.ctx.k_eq_v:
            return None
        return ProjectionGroup.of(tuple(self.path), 'qkv_proj', ('q_proj', 'k_proj', 'v_proj'),
                                  self.projections())

    def projection_groups(self) -> tuple[ProjectionGroup, ...]:
        """Its group (`_group`) where its variables hold it (`ProjectionSites`),
        which `setup` reads packed in the members' place."""
        group = self._group()
        return (group,) if group is not None and group.held(self.variables.get('params', {})) else ()

    def _rot_dim(self) -> int | None:
        """Head dims the rotary rotates, or None for all of them.

        `partial_rotary_type` says what the fraction means: 'proportional'
        rotates the first rot_dim dims of a head_dim-wide rope and passes the
        rest at frequency zero (Gemma 4), 'default' builds a rot_dim-wide rope
        and leaves the rest unrotated (Qwen3.5); `rotary_freqs` names the
        reference lines. Both rotate `int(head_dim * factor)` dims.
        """
        factor = self.ctx.partial_rotary_factor
        if factor is None:
            return None
        rot_dim = int(self.ctx.head_dim * factor)
        if not 0 < factor <= 1 or rot_dim % 2:
            raise ValueError(
                f"partial_rotary_factor must rotate an even positive number of "
                f"head dims, got {factor} of head_dim {self.ctx.head_dim}")
        return rot_dim

    def _shared_kv(self, kv_store):
        """Read the keys, values, query positions and key places the provider layer stashed.

        The provider ran earlier in the same forward pass and left them
        post-norm and post-rope, so a sharing layer projects, norms,
        rotates and caches nothing of its own.
        """
        if kv_store is None or self.ctx.kv_store_key not in kv_store:
            raise ValueError(
                f"layer shares K/V under {self.ctx.kv_store_key!r} but no provider "
                "stashed them; the model has to pass one kv_store dict down "
                "its layer stack")
        return kv_store[self.ctx.kv_store_key]

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
            value = key if self.ctx.k_eq_v else self.v_proj(x)
        else:
            key, value = projected
        key = checkpoint_name(key, 'k_proj')
        if not self.ctx.k_eq_v:
            value = checkpoint_name(value, 'v_proj')
        value = value.reshape(batch, length, self.ctx.num_kv_heads, self.value_dim)
        if self.ctx.value_scale is not None:
            value = scaled(value, self.ctx.value_scale)
        if whole:
            key = self.k_norm(key)
        key = key.reshape(batch, length, self.ctx.num_kv_heads, self.ctx.head_dim)
        if self.ctx.qk_norm and not whole:
            key = self.k_norm(key)
        if self.ctx.v_norm:
            value = self.values_norm(value)
        return constrain(key, KV_HEADS), constrain(value, KV_HEADS)

    def _rotary_angles(self, rotary_positions, heads: jax.Array, tokens: Coordinates | None = None,
                       places=None):
        """Build the rotary cos and sin this layer rotates its heads by, in
        the arithmetic `rotate` rotates `heads` in, rounded to `heads`'
        dtype under `MixerContext.narrow_rotary_tables` (`rounded_to`, which
        compilation keeps). Under LongRoPE an uncached call's packed `tokens`
        give each document's furthest place (`document_reach` over `places`),
        which it picks its factors by, so a document rotates as it does alone."""
        longrope = isinstance(self.ctx.rope_scaling, LongRopeScaling)
        reach = (None if not longrope or tokens is None or tokens.segments is None
                 else document_reach(tokens.segments, tokens.valid, places))
        tables = self._rotary_tables(rotary_positions, at_least_fp32(heads.dtype), reach)
        if not self.ctx.narrow_rotary_tables:
            return tables
        return rounded_to(tables[0], heads.dtype), rounded_to(tables[1], heads.dtype)

    def _rotary_tables(self, rotary_positions, dtype, reach=None):
        """The rotary cos and sin at `dtype`. Interleaved mRoPE, YaRN and the
        plain rope each build their own angles; YaRN rotates whole heads at
        its own frequencies, so it takes neither a partial rotary nor a
        Llama 3.1 ramp."""
        if self.rotary_axes is not None:
            if rotary_positions is None or rotary_positions.ndim != 3:
                raise ValueError(f"a rotary over {len(self.rotary_axes)} axes reads their ids "
                                 "[B, S, axes] from attention_metadata.rotary_positions")
            return axis_tables(rotary_positions, self.rotary_axes, self.ctx.rope_theta, dtype=dtype)
        if self.mrope_section is not None and rotary_positions is not None and rotary_positions.ndim == 3:
            if self.ctx.partial_rotary_type != "default":
                raise ValueError("M-RoPE requires default partial rotary")
            return mrope_tables(rotary_positions, self.mrope_section, self._rot_dim() or self.ctx.head_dim,
                                self.ctx.rope_theta, dtype=dtype)
        if self.ctx.yarn is None:
            return rotary_freqs(
                rotary_positions, self.ctx.head_dim, self.ctx.rope_theta, rot_dim=self._rot_dim(),
                partial_rotary_type=self.ctx.partial_rotary_type, rope_scaling=self.ctx.rope_scaling,
                dtype=dtype, reach=reach)
        if self.ctx.partial_rotary_factor is not None or self.ctx.rope_scaling is not None:
            raise ValueError(
                "yarn rotates whole heads at its own frequencies, so it takes "
                "neither partial_rotary_factor nor rope_scaling")
        return yarn_rope_freqs(rotary_positions, self.ctx.head_dim, self.ctx.rope_theta, self.ctx.yarn,
                               dtype=dtype)

    def _metadata_mask(self, metadata: AttentionMetadata | None, slots, counted,
                       batch: int, length: int, key_length: int, decode: bool):
        """Combine key validity, causality, image groups and the layer's window, which counts
        tokens: a cache's slots are its tokens, and an uncached call's are `counted`
        (`counted_places`), padding among them skipped."""
        query_slots = (slots if decode else jnp.broadcast_to(jnp.arange(length), (batch, length)))
        groups = (None if metadata is None else metadata.image_groups)
        if groups is None:
            groups = jnp.full((batch, length), -1, jnp.int32)
        key_groups = groups
        if decode and self.bidirectional_images:
            allocated = self.has_variable("cache", "cached_image_groups")
            stored = self.variable("cache", "cached_image_groups", jnp.full,
                                   (batch, self.ctx.max_seq_len), -1, jnp.int32)
            if allocated:
                stored.value = write_cache(stored.value, groups, query_slots)
            key_groups = stored.value
        valid = (cached_validity(self, key_length) if decode
                 else None if metadata is None else metadata.valid)
        keep = (causal_attention_mask(query_slots, key_length)
                if self.ctx.causal else jnp.ones((batch, 1, length, key_length), bool))
        if self.bidirectional_images:
            same_image = ((groups[:, :, None] == key_groups[:, None, :])
                          & (groups[:, :, None] >= 0))
            keep = keep | same_image[:, None]
        if self.ctx.sliding_window is not None and (self.ctx.causal or self.ctx.bidirectional_window):
            places = jnp.broadcast_to(query_slots if decode else counted, (batch, length))
            key_places = jnp.arange(key_length)[None, :] if decode else places
            distance = places[:, None, :, None] - key_places[:, None, None, :]
            # A causal window bounds only the keys behind the query, so an
            # image's later tokens stay visible to its earlier ones; a
            # two-sided one bounds both (`window_sides`).
            keep = keep & ((distance < self.ctx.sliding_window) if self.ctx.causal
                           else (jnp.abs(distance) < self.ctx.sliding_window))
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
        time, so a host that knows a row is unpadded passes none, and one that
        right-pads its rows says so (`AttentionMetadata.right_padded`): causality
        alone keeps a real query off the padding after it. Decoding always
        builds the mask, which carries the cache's validity.
        """
        if decode:
            return metadata is not None or self.bidirectional_images
        # QK-Clip's logit maxima are taken under the mask (`_attended`), so a
        # layer that sows them keeps it.
        implied = (metadata is not None and metadata.right_padded and self.ctx.causal
                   and not (not self.is_initializing() and self.is_mutable_collection("qk")))
        return metadata is not None and (
            (metadata.valid is not None and not implied)
            or (self.bidirectional_images and metadata.image_groups is not None))

    @nn.compact
    def __call__(self, x, call: MixerCall = PLAIN_CALL, train: bool = False):
        decode, kv_store, attention_metadata = call.decode, call.kv_store, call.metadata
        B, S, _ = x.shape
        # The tokens' places and documents: the caller's or a packed batch's, else none given
        # (`Coordinates`).
        tokens = coordinates_of(attention_metadata, B, S)
        # Each token's logical place (`Coordinates.places`); a cached call's default is its slot.
        positions, segment_ids = tokens.positions if decode else tokens.places(), tokens.segments
        logical_positions = positions
        # The projections and the kernel's output carry the names a remat
        # policy saves or offloads (decoder_block.RESIDUALS).
        projected_kv = None
        group = self._group()
        if group is not None and self._packed():
            projected, key, value = group.split(self.qkv_proj(x))
            projected_kv = (key, value)
        else:
            projected = self.q_proj(x)
        projected = checkpoint_name(projected, 'q_proj')
        # OLMo 3 norms the whole projection, one scale of heads * head_dim,
        # before the head split (modeling_olmo3.py:162-163, :178-179); Qwen3
        # and the Gemmas norm each head after it, which the reference marks
        # "unlike olmo, only on the head dim" (modeling_qwen3_moe.py:147).
        whole = self.ctx.qk_norm and self.ctx.qk_norm_scope == 'projection'
        if whole:
            projected = self.q_norm(projected)
        gate = None
        if self.ctx.output_gate and self.gate_projection == 'query':
            # The reference views the doubled output as [.., heads, 2*head_dim]
            # and chunks it: query, then gate (modeling_qwen3_5.py:670-673).
            query, gate = jnp.split(
                projected.reshape(B, S, self.ctx.num_heads, 2 * self.ctx.head_dim),
                2, axis=-1)
        else:
            query = projected.reshape(B, S, self.ctx.num_heads, self.ctx.head_dim)
        if self.ctx.output_gate and self.gate_projection == 'own':
            gate = checkpoint_name(self.gate_proj(x), 'output_gate')
            gate = gate.reshape(B, S, self.ctx.num_heads, self.ctx.head_dim)
        held_places = None
        if self.ctx.kv_shared:
            key, value, positions, held_places = self._shared_kv(kv_store)
        else:
            key, value = self._projected_kv(x, whole, projected_kv)
        if self.ctx.qk_norm and not whole:
            query = self.q_norm(query)
        # Column-parallel under a tensor axis, each shard a run of whole
        # heads; o_proj's sum returns to the residual placement in the block.
        query = constrain(query, HEADS)

        kv_len = key.shape[-3]
        positions, append, prefix = self._step_positions(key, positions, segment_ids, attention_metadata,
                                                         decode, S)
        # The tokens' logical places (`Coordinates.positions`): the caller's, else the slots a cache
        # writes them at or their indices.
        places = positions if logical_positions is None else logical_positions
        # Windows, chunks and ALiBi count each document's tokens from its first place
        # (`counted_places`); rotation and NoPE's temperature read the places as given. A cached step
        # reads its own.
        counted = places if decode or logical_positions is None else counted_places(
            segment_ids, tokens.valid, logical_positions)
        rotary_positions = places
        if attention_metadata is not None and attention_metadata.rotary_positions is not None:
            rotary_positions = attention_metadata.rotary_positions
        freqs_cos = freqs_sin = None
        if self.nope:
            # NoPE rotates nothing; the query still carries the logit scale
            # the checkpoint asks for, which rotate folds in otherwise.
            if self.ctx.attention_scale is not None:
                query = scaled(query, self.ctx.attention_scale * math.sqrt(self.ctx.head_dim))
            if self.temperature_tuning is not None:
                scale = temperature_scale(places, *self.temperature_tuning, dtype=at_least_fp32(query.dtype))
                query = query * (scale[..., None, None] if scale.ndim == 2
                                 else scale[None, :, None, None]).astype(query.dtype)
        else:
            uncached = None if decode else tokens
            freqs_cos, freqs_sin = self._rotary_angles(rotary_positions, query, uncached, places)
            # Every kernel path scales the logits by 1/sqrt(head_dim) itself, so the
            # query carries the ratio to the scale the checkpoint asks for.
            query = rotate(
                self._turned(query), freqs_cos, freqs_sin, pairs=self.rotary_pairs,
                scale=(None if self.ctx.attention_scale is None
                       else self.ctx.attention_scale * math.sqrt(self.ctx.head_dim)))
        own_value = value
        if not self.ctx.kv_shared:
            if freqs_cos is not None and freqs_sin is not None:
                key = rotate(self._turned(key), freqs_cos, freqs_sin, pairs=self.rotary_pairs)
            if self.ctx.ngpt is not None:
                query, key = self._ngpt_rotated(self.ctx.ngpt, query, key)
            if kv_store is not None and self.ctx.kv_store_key is not None:
                # Post-norm, post-rope, the same tensors the reference hands
                # its sharing layers (modeling_gemma4.py, Gemma4TextAttention).
                kv_store[self.ctx.kv_store_key] = (key, value, positions, places)
        if decode and attention_metadata is not None and attention_metadata.admitted is not None:
            attention = self._mixed_attended(query, key, value, positions, attention_metadata)
            return self._output(attention, gate, B, S, own_value)
        sinks = (self.param('sinks', nn.initializers.zeros, (self.ctx.num_heads,))
                 if self.ctx.attention_sinks else None)
        if self._runs_local(attention_metadata, decode) and not (self.ctx.attention_dropout_rate and train):
            attention = checkpoint_name(local_attention(
                query, key, value, window=self.ctx.sliding_window, chunk=self.ctx.attention_chunk,
                places=None if logical_positions is None else counted,
                segment_ids=segment_ids,
                max_segments=None if segment_ids is None else tokens.max_segments,
                valid=tokens.valid, right_padded=tokens.right_padded,
                dtype=self.ctx.dtype, precision=self.ctx.precision,
                force_fp32_for_softmax=self.ctx.force_fp32_for_softmax,
                implementation=self.ctx.attention_impl, sinks=sinks,
                softcap=self.ctx.attn_logit_softcap), 'context')
            return self._output(attention, gate, B, S, own_value)
        masking = self._masking(query, key, value, positions, counted, held_places, rotary_positions, append,
                                prefix, kv_len, kv_store, segment_ids, attention_metadata, decode)
        bias = None
        if self.alibi:
            bias = alibi_bias(masking.key_places, self.ctx.num_heads, dtype=query.dtype)
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
            if self.ctx.causal:
                if not self.ctx.kv_shared:
                    positions, append = open_kv_cache(
                        self, key, self.ctx.max_seq_len, value_dim=self.value_dim,
                        valid=None if attention_metadata is None else attention_metadata.valid,
                        layout=self.ctx.kv_cache)
            elif self.ctx.kv_cache != KVCache():
                raise ValueError(
                    "a bidirectional canvas reads its encoder prefix as dense, full-precision "
                    "keys; it takes the default kv_cache layout")
            elif self.ctx.kv_shared:
                raise ValueError(
                    "a bidirectional canvas over a cache shares no keys across "
                    "layers: a sharing layer owns no cache of its own")
            elif not self.has_variable("cache", "cached_key"):
                raise ValueError(
                    "a bidirectional canvas has no KV cache of its own: prefill "
                    "the prompt with the causal model first")
            else:
                # The encoder's frozen prefix: positions continue past it, and
                # the decoder never writes it back.
                prefix = self.get_variable("cache", "cache_index")
                positions = prefix[:, None] + jnp.arange(S)
        elif positions is None and not self.ctx.kv_shared:
            positions = jnp.arange(S)
        elif not self.ctx.kv_shared:
            positions = jnp.asarray(positions)
        return positions, append, prefix

    def _mask_kernel(self, query, key, decode: bool) -> str:
        """The kernel a call that builds its mask runs: xla
        (`kernel_for_materialized_mask`), except a cache call over several
        queries (a prefill, a verified draft) where cuDNN takes the mask
        (`cudnn_takes_mask`); an explicit 'xla' stays xla. That call runs
        forward only, where cuDNN takes the mask as its additive bias with
        none of the backward's refusals: 8 prompts of 256 at Qwen3-0.6B's
        widths took 4.0 against xla's 6.2 ms over 28 layers on an RTX 4080.
        It rounds elsewhere, so the prefill is not bitwise xla's: its RMS
        distance from an fp32 forward is 1.04 times xla's on Qwen3-0.6B and
        1.00 on 1.7B (tests/reference_error.py allows 2), and greedy rows part
        only at bf16 near-ties (docs/performance.md)."""
        masked = kernel_for_materialized_mask(
            self.ctx.attention_impl, query, key, dtype=self.ctx.dtype, precision=self.ctx.precision,
            force_fp32_for_softmax=self.ctx.force_fp32_for_softmax)
        if decode and query.shape[1] > 1 and masked == 'xla' and cudnn_takes_mask(
                self.ctx.attention_impl, query, softcap=self.ctx.attn_logit_softcap,
                sinks=self.ctx.attention_sinks):
            return 'cudnn'
        return masked

    def _masking(
        self,
        query,
        key,
        value,
        positions,
        places,
        held_places,
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
        flag, window, mask, document ids, kernel and key places the layer's
        visibility comes to (the canvas prefix, a shared or cached decode step,
        packed documents, metadata, a pairwise mask, a chunk). `places` are the
        queries' places counted over their documents' tokens (`counted_places`),
        and `held_places` a sharing layer's keys'."""
        B, S = query.shape[:2]
        causal, mask, documents = self.ctx.causal, None, None
        implementation = self.ctx.attention_impl
        # An uncached call's keys are its own tokens; a cached call's keys are
        # placed by `_cached_places` where the layer reads their places. A
        # paged cache holds rows a server seats at canonical positions, where
        # a key's slot is its place, so it keeps none.
        key_places = places
        keeps_places = ((self.ctx.attention_chunk is not None or self.alibi)
                        and self.ctx.kv_cache.page_size is None)
        # A bidirectional layer windows only where its kind keeps the window on
        # both sides; DiffusionGemma's decoder reads its whole canvas
        # (modeling_diffusion_gemma.py:1399-1401), and its prefix branch below
        # bounds the cached keys itself.
        window = (None if decode or not (causal or self.ctx.bidirectional_window)
                  else self.ctx.sliding_window)
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
            if self.ctx.sliding_window is not None:
                valid_prefix = valid_prefix & (prefix_slots > prefix[:, None] - self.ctx.sliding_window)
            canvas_valid = (jnp.ones((B, S), bool) if attention_metadata is None
                            or attention_metadata.valid is None else attention_metadata.valid)
            keep = jnp.concatenate([valid_prefix, canvas_valid], axis=-1)
            mask = jnp.broadcast_to(keep[:, None, None, :], (B, 1, S, alloc + S))
            key = jnp.concatenate([cached_key, key], axis=-3)
            value = jnp.concatenate([cached_value, value], axis=-3)
            causal, window = False, None
            key_places = (jnp.concatenate([self.get_variable("cache", "cached_places"),
                                           jnp.broadcast_to(places, (B, S))], axis=-1)
                          if keeps_places else jnp.arange(alloc + S))
        elif self.ctx.kv_shared and decode:
            # No cache of its own: the provider's stashed keys carry the full history, read as
            # the provider's own cache is, a step's token by its mask and more by `_cached_call`.
            if S == 1:
                mask, causal = causal_attention_mask(positions, kv_len, self.ctx.sliding_window), False
            else:
                last = jnp.max(jnp.asarray(positions).reshape(B, S), axis=-1, keepdims=True)
                documents, causal, window = self._cached_call(positions, jnp.arange(kv_len)[None, :] <= last)
            key_places = held_places
            # A quantized provider stashed its keys in the cache's rotation.
            rotation = self.ctx.kv_cache.key_rotation(self.ctx.head_dim)
            if rotation is not None:
                query = rotated(query, rotation)
        elif append is not None:
            key, value = append(key, value)
            query = append.query(query)
            if S == 1:
                (cursor, mask), causal = self._decode_masks(positions, key.shape[-3]), False
            else:  # several tokens appended after the keys their rows held
                documents, causal, window = self._cached_call(positions, cached_validity(self, key.shape[-3]))
            key_places = self._cached_places(places, positions) if keeps_places else jnp.arange(key.shape[-3])
            if kv_store is not None and self.ctx.kv_store_key is not None:
                kv_store[self.ctx.kv_store_key] = (key, value, positions, key_places)
        elif segment_ids is not None:
            # Attention stays inside each packed document. The ids travel to
            # the kernel beside the causal flag and the window: splash compares
            # them per block, and every other kernel builds the document mask
            # from them (`attention_kernel`).
            documents = segment_ids
        # Over the keys the call reads, the cache's or the prefix's joined.
        masked = self._mask_kernel(query, key, decode)
        cached = isinstance(documents, CrossDocuments)
        if cached and self.ctx.attention_impl != 'auto':
            # A named kernel takes a cache call as it took the call's mask (`_mask_kernel`).
            implementation = masked
        if prefix is None and self._restricts_visibility(attention_metadata, decode) and not (
                cached and not self.bidirectional_images):
            # A decode step's cache mask already holds the rows' validity, and a
            # cache call's documents too; only image groups add to it there.
            if not decode or self.bidirectional_images:
                mask = self._metadata_mask(attention_metadata, positions, places, B, S, key.shape[-3], decode)
                if segment_ids is not None and not decode:
                    mask = with_documents(mask, segment_ids)
            documents = None
            causal, window = False, None
            implementation = masked
        if attention_metadata is not None and attention_metadata.pairwise_mask is not None:
            mask, key_positions = self._pairwise_mask(attention_metadata, rotary_positions, B, S,
                                                      key.shape[-3])
            key_places = key_places if key_positions is None else key_positions
            causal, window, documents = False, None, None
            implementation = masked
        if self.ctx.attention_chunk is not None:
            mask = self._chunked(mask, documents, causal, window, places, key_places, S, key.shape[-3],
                                 self.ctx.attention_chunk)
            causal, window, documents = False, None, None
            implementation = masked
        max_segments = (None if documents is None else 1 if isinstance(documents, CrossDocuments)
                        else coordinates_of(attention_metadata, B, S).max_segments)
        return _Masking(query, key, value, causal, window, mask, documents, max_segments, implementation,
                        cursor, key_places)

    @staticmethod
    def _chunked(mask, documents, causal: bool, window: int | None, places, key_places, length: int,
                 keys: int, chunk: int):
        """`mask` with a call's documents, causality and window folded in, joined with the layer's
        chunks of `chunk` places: the decode and diagnostic paths, each query and key in the chunk of
        its counted place, never its cache slot."""
        chunked = chunk_mask(places, key_places, chunk)
        if isinstance(documents, CrossDocuments):
            aligned = cross_document_mask(documents, causal=causal, sliding_window=window)
            mask = aligned if mask is None else aligned & mask
            documents, causal, window = None, False, None
        if documents is not None:
            mask = with_documents(mask, documents)
        base = combined_attention_mask(length, keys, causal, window, mask)
        return chunked if base is None else base & chunked

    def _pairwise_mask(self, metadata: AttentionMetadata, rotary_positions, B: int, S: int, keys: int):
        """A call's own mask `[B, 1, S, keys]` (`AttentionMetadata.pairwise_mask`) and the places its
        keys are read at, None where it names none; a window keeps each query to the keys within it
        of the query's rotary position."""
        pairwise = jnp.asarray(metadata.pairwise_mask)
        if pairwise.shape != (B, S, keys) or pairwise.dtype != jnp.bool_:
            raise ValueError("attention_pairwise_mask must be boolean [B, queries, keys]")
        mask = pairwise[:, None]
        key_positions = metadata.key_positions
        if key_positions is not None:
            if key_positions.shape != (B, keys):
                raise ValueError("attention_key_positions must be [B, keys]")
            if self.ctx.sliding_window is not None:
                query_positions = jnp.broadcast_to(jnp.asarray(rotary_positions), (B, S))
                distance = query_positions[:, :, None] - key_positions[:, None, :]
                mask = mask & (jnp.abs(distance) < self.ctx.sliding_window)[:, None]
        return mask, key_positions

    def _cached_places(self, places, slots) -> jax.Array:
        """Each cache slot's place `[rows, capacity]`, where a chunk and ALiBi
        read its key: the place the call that wrote the slot counted its token
        at (`counted_places`). A slot is that place only by default, so the
        cache keeps it beside the key, and this call's `places` go in at their
        `slots` as its keys do."""
        allocated = self.has_variable("cache", "cached_places")
        held = self.variable("cache", "cached_places", jnp.zeros, (slots.shape[0], self.ctx.max_seq_len),
                             jnp.int32)
        if allocated:
            held.value = write_cache(held.value, jnp.broadcast_to(places, slots.shape), slots)
        return held.value

    def _attended(self, masking: _Masking, positions, append, sinks, train: bool, *, bias=None):
        """The attention output through the kernel the masking chose: the
        paged kernel or a per-row key count for a plain decode step, the
        general kernel otherwise, with the per-head logit maxima sown for
        QK-Clip when a caller opened that collection."""
        query, key, value, causal, window, mask, documents, max_segments, implementation, cursor, _ = masking
        B, S = query.shape[:2]
        # The per-head maxima the QK-Clip reads. Computed only when a caller
        # opened the collection; the plain forward leaves it closed and its
        # leaves bitwise identical.
        sowing = not self.is_initializing() and self.is_mutable_collection("qk")
        if sowing:
            # A cache call's documents align to their last keys, which the mask carries in their place.
            aligned = isinstance(documents, CrossDocuments)
            seen = (cross_document_mask(documents, causal=causal, sliding_window=window) if aligned
                    else None if documents is None else with_documents(None, documents))
            self.sow("qk", "max_logits", max_attention_logits(
                query, key, causal=causal and not aligned, sliding_window=None if aligned else window,
                mask=seen if mask is None else mask if seen is None else mask & seen, bias=bias))
            self.sow("qk", "kv_heads", jnp.asarray(key.shape[-2]))
            self.sow("qk", "head_dim", jnp.asarray(query.shape[-1]))
        if self.ctx.attention_dropout_rate and train:
            return checkpoint_name(scaled_dot_product_attention(
                query, key, value, dtype=self.ctx.dtype, precision=self.ctx.precision,
                force_fp32_for_softmax=self.ctx.force_fp32_for_softmax,
                implementation=self.ctx.attention_impl, causal=causal, sliding_window=window,
                mask=mask, bias=bias, sinks=sinks, softcap=self.ctx.attn_logit_softcap, segment_ids=documents,
                dropout_rate=self.ctx.attention_dropout_rate, dropout_rng=self.make_rng("dropout"),
                deterministic=False), 'context')
        # A chunk, window or metadata mask replaces `cursor` and keeps the gather.
        plain_step = (append is not None and mask is cursor and S == 1 and sinks is None
                      and bias is None and not sowing)
        if (append is not None and plain_step and self._page_kernel_runs(query)
                and append.store.kernel(query[:, 0])):
            attention = self._paged(append, query, jnp.asarray(positions).reshape(B, S)[:, 0])
        elif plain_step:
            attention = self._decode_attention(query, key, value, jnp.asarray(positions).reshape(B, S)[:, 0])
        else:
            attention = checkpoint_name(scaled_dot_product_attention(
                query, key, value, dtype=self.ctx.dtype, precision=self.ctx.precision,
                force_fp32_for_softmax=self.ctx.force_fp32_for_softmax,
                implementation=implementation, causal=causal,
                sliding_window=window, mask=mask, bias=bias, sinks=sinks,
                softcap=self.ctx.attn_logit_softcap, segment_ids=documents, max_segments=max_segments),
                'context')
        return attention

    def _mixed_positions(self, admitted: Admitted, valid) -> jax.Array:
        """Each token's slot in its row, `[1, tokens]`, -1 for padding: a
        decoding row's token at its cursor, a prompt's after its row's start."""
        refusal = mixed_attention_refusal(self.ctx, alibi=self.alibi,
                                          bidirectional_images=self.bidirectional_images)
        if refusal is not None:
            raise ValueError(f"a mixed serving step needs plain attention; {self.name}: {refusal}")
        index = self.get_variable("cache", "cache_index")
        rows, pieces = index.shape[0], admitted.slots.shape[0]
        valid = jnp.asarray(valid, bool)[0]
        prompt = valid[rows:].reshape(pieces, -1)
        filled = admitted.cursors[:, None] + jnp.cumsum(prompt, axis=1, dtype=jnp.int32) - 1
        positions = jnp.concatenate([jnp.where(valid[:rows], index, -1),
                                     jnp.where(prompt, filled, -1).reshape(-1)])
        return positions[None]

    def _mixed_attended(self, query, key, value, positions, metadata: AttentionMetadata):
        """A mixed call's attention: every valid token's keys and values go
        into its row of the cache, then each decoding row's query reads its
        row as a decode step does, and each prompt's queries read their row
        up to their own slot. Its rows' caches are its whole visibility, so a
        call that also carries a pairwise mask, key positions or a position
        bias is refused by name rather than run without them."""
        carried = [name for name, given in (("a pairwise mask", metadata.pairwise_mask),
                                            ("key positions", metadata.key_positions),
                                            ("a position bias", metadata.position_bias)) if given is not None]
        if carried:
            raise ValueError(f"{self.name}: a serving step's mixed call reads each row's cache as a decode "
                             f"step does, and takes no {' or '.join(carried)}")
        admitted = metadata.admitted
        assert admitted is not None
        index = self.variable("cache", "cache_index", jnp.zeros, (0,), jnp.int32)
        store = KVStore.open(self, self.ctx.kv_cache, index.value.shape[0], self.ctx.max_seq_len,
                             self.ctx.num_kv_heads, self.ctx.head_dim, key.dtype, value_dim=self.value_dim)
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
        if self._page_kernel_runs(query) and store.kernel(query[0, :rows]):
            decoded = store.decode(query[0, :rows], held_reads(index.value, positions[:rows]),
                                   self.ctx.attn_logit_softcap)
            decoded = checkpoint_name(decoded[:, None], 'context')
        else:
            decoded = self._decode_attention(query[0, :rows, None], *store.read(), positions[:rows])
        queries = query[0, rows:].reshape(pieces, width, *query.shape[2:])
        if admitted.continuing:
            # A padding piece's row is past the cache's; it reads row 0 and is never drawn from.
            held = jnp.where(admitted.slots < rows, admitted.slots, 0)
            keys, values = store.read_rows(held)
            keep = causal_attention_mask(prompt, self.ctx.max_seq_len,
                                         key_valid=filled_slots(index.value[held], self.ctx.max_seq_len))
        else:
            # A piece that starts its row reads only its own keys, in its own order.
            keys, values = (x[0, rows:].reshape(pieces, width, *x.shape[2:]) for x in (key, value))
            keep = ((prompt[:, None, :, None] >= prompt[:, None, None, :]) & (prompt >= 0)[:, None, None, :])
        # A padding query reads one key, so its (unused) output stays finite.
        keep = keep | ((prompt < 0)[:, None, :, None] & (jnp.arange(keep.shape[-1]) == 0))
        prefilled = scaled_dot_product_attention(
            queries, keys, values, dtype=self.ctx.dtype, precision=self.ctx.precision,
            force_fp32_for_softmax=self.ctx.force_fp32_for_softmax,
            implementation=self._mask_kernel(queries, keys, decode=True), causal=False, mask=keep,
            softcap=self.ctx.attn_logit_softcap)
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
        reads = jnp.maximum(positions + 1, 1)
        # The core reads each key head's query heads as its query positions
        # (`decode_geometry.grouped_query`).
        attention = scaled_dot_product_attention(
            query, key, value, dtype=self.ctx.dtype, precision=self.ctx.precision,
            force_fp32_for_softmax=self.ctx.force_fp32_for_softmax,
            implementation=self.ctx.attention_impl, causal=False,
            softcap=self.ctx.attn_logit_softcap,
            key_value_seq_lengths=reads.astype(jnp.int32))
        return checkpoint_name(attention, 'context')

    def _page_kernel_runs(self, query) -> bool:
        """Whether a decode step may read a paged pool through its device's
        paged kernel (`KVStore.decode`) rather than the gathered rows. The
        paged kernels read keys and values of one width, so narrower values
        take the gathered rows, which the attention core widens. Neither
        device's paged kernel performs what only the reference path does
        (`reference_only`), so such a step takes the gathered rows, where an
        explicit kernel refuses it by name."""
        if self.value_dim != self.ctx.head_dim or reference_only(
                query, self.ctx.dtype, self.ctx.precision, self.ctx.force_fp32_for_softmax):
            return False
        if backend() != 'gpu':
            return self.ctx.attention_impl in ('auto', 'tpu')
        return self.ctx.attention_impl in ('auto', 'cudnn') and cudnn_runs(query, self.ctx.attn_logit_softcap)

    def _decode_masks(self, positions, key_length: int) -> tuple[jax.Array, jax.Array]:
        """The cursor mask over the cache's filled slots, and the layer's decode mask.

        They are the same array exactly when the layer narrows nothing, which
        is when the paged kernel, attending every filled slot, may stand in
        for the mask; a window builds a mask of its own.
        """
        valid = cached_validity(self, key_length)
        cursor = causal_attention_mask(positions, key_length, key_valid=valid)
        if self.ctx.sliding_window is None:
            return cursor, cursor
        return cursor, causal_attention_mask(positions, key_length, self.ctx.sliding_window, key_valid=valid)

    def _cached_call(self, positions, held) -> tuple[CrossDocuments, bool, int | None]:
        """A cache call of several tokens at `positions` `[B, S]` (-1 for padding) over the keys
        `held` `[B, K]` marks filled, as its documents, causality and window: one document a row, whose
        last keys are the call's tokens, appended in order after the keys the row held
        (`CrossDocuments`). Causality and the layer's window, where it narrows the K keys, align to
        those keys, which is the cache's own mask (`causal_attention_mask` over the slots); flash reads
        them as packed sequences and no kernel needs a mask for them."""
        window = self.ctx.sliding_window
        documents = CrossDocuments(jnp.asarray(jnp.asarray(positions) >= 0, jnp.int32),
                                   jnp.asarray(held, jnp.int32))
        return documents, True, None if window is None or window >= held.shape[-1] else window

    def _paged(self, append: Append, query: jax.Array, positions: jax.Array) -> jax.Array:
        """One decode step through the Pallas paged kernel, which reads the pool
        through the page table; the gathered keys go unread and XLA drops the gather.
        A row whose query is padding (`positions` -1) reads one key (`held_reads`)."""
        reads = held_reads(self.get_variable("cache", "cache_index"), positions)
        return checkpoint_name(append.store.decode(query[:, 0], reads, self.ctx.attn_logit_softcap)[:, None],
                               'context')

    def _runs_local(self, metadata: AttentionMetadata | None, decode: bool) -> bool:
        """Whether this call runs `local_attention`, which never builds the
        `[S, S]` mask a window or chunk otherwise costs.

        That is a causal local layer's whole-sequence pass, over row
        validity and packed documents. A cache, a pairwise mask or
        bidirectional image groups keep the mask path, and so does an open
        `qk` collection, whose maxima read the dense logits anyway.
        """
        if (self.ctx.sliding_window is None and self.ctx.attention_chunk is None) or not self.ctx.causal:
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
        if gate is not None and self.ctx.ngpt is not None:
            # nGPT scales the gate's logits by the square of their learned gain, gates in fp32 and
            # unit-norms the gated heads together (dory.py:154-157).
            scale = gain(self.s_gate, self.ctx.ngpt.gate(self.ctx.emb_features))
            logits = gate.astype(jnp.float32) * jnp.square(scale).reshape(self.ctx.num_heads,
                                                                           self.ctx.head_dim)
            attention = (attention.astype(jnp.float32) * jax.nn.sigmoid(logits)).astype(attention.dtype)
        elif gate is not None:
            # The branch multiplies by the sigmoid of its gate, then projects
            # (modeling_qwen3_5.py:701, and modeling_qwen4_exp.py:836 the same).
            attention = attention * jax.nn.sigmoid(gate).astype(attention.dtype)
        attention = constrain(attention, HEADS)
        joined = attention.reshape(batch, length, self.ctx.num_heads * self.value_dim)
        if self.ctx.ngpt is not None:
            joined = unit(joined)
        return checkpoint_name(self.o_proj(joined), 'o_proj')


@dataclasses.dataclass(frozen=True)
class AttentionMixer(MixerBase):
    """Grouped-query attention with optional image-block masking and M-RoPE.

    Geometry, norms and kernel policy come from the decoder context. Image
    bidirectionality, spatial rotary sections, NoPE and exclusive self attention
    are this mixer's own fields, which a hybrid names on its attention kind
    (`{"name": "attention", "fields": {"nope": true}}`).
    """

    residuals = ('q_proj', 'k_proj', 'v_proj', 'kv_proj', 'output_gate', 'o_proj', 'context')

    bidirectional_images: bool = False
    mrope_section: tuple[int, int, int] | None = None
    nope: bool = False
    """No positional encoding: q and k enter the kernel unrotated and the
    logits keep their scale (lm-engine's `position_embedding_type="nope"`,
    Granite 4.0-H)."""
    alibi: bool = False
    """Per-head linear key-position bias (`CausalSelfAttention.alibi`)."""
    exclusive_self_attention: bool = False
    """XSA (arXiv 2603.09078, lm-engine's `exclusive_self_attention`): each
    head's output loses its component along the token's own value
    (`exclusive_self_attention`)."""
    gate_projection: Literal['query', 'own'] = 'query'
    """Where the decoder's output gate comes from (`CausalSelfAttention.gate_projection`)."""
    qk_norm_weight: bool = True
    """Whether the decoder's qk norm has a weight (`CausalSelfAttention.qk_norm_weight`): Muse
    Glimmer's has none (modeling_muse_glimmer.py:307)."""

    # `CausalSelfAttention` reads the mixed call's `Admitted` layout.
    mixed_step = True

    def mixed_refusal(self, ctx: MixerContext) -> str | None:
        return mixed_attention_refusal(ctx, alibi=self.alibi, bidirectional_images=self.bidirectional_images)

    def __post_init__(self):
        if self.mrope_section is not None:
            object.__setattr__(self, "mrope_section", tuple(self.mrope_section))
            if len(self.mrope_section) != 3 or any(value < 0 for value in self.mrope_section):
                raise ValueError("mrope_section must contain three nonnegative section widths")

    def build(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        # This kind's fields are CausalSelfAttention's own, by name.
        kind = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        return functools.partial(CausalSelfAttention, ctx=ctx, **kind)

