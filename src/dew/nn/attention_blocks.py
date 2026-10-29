"""The attention modules and the transformer blocks built of them: `NormalAttention`, self- or
cross-attention over projected heads, `BasicTransformerBlock` and `TransformerBlock`, from diffusers'
attention_flax.py, and a UNet stage's attention (`Stage`, `stage_attention`).
"""

import dataclasses
import functools
import math
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import AttentionImpl
from dew.nn.dense import Dense, DenseGeneral
from dew.nn.dropout import Dropout

from .attention import scaled_dot_product_attention
from .flops import cross_attention_flops, projection_flops, self_attention_flops
from .norms import LayerNorm, RMSNorm
from .precision import float32_dot_general
from .rope import rotate
from .sharding import HEADS, KV_HEADS, constrain, logical_axes, split_positions


def projected_heads(x, name: str, heads: int, head_dim: int, *, flat: bool, bias: bool = True,
                    dtype: Dtype | None = None, precision: PrecisionLike = None,
                    norm: Callable[[jax.Array], jax.Array] | None = None, across_heads: bool = False
                    ) -> jax.Array:
    """`x` `[..., C]` projected by the matrix `name` to `[..., heads, head_dim]`
    in the calling module, then normalized by `norm`.

    `flat` stores the matrix as the published diffusion and vision attentions
    do, `[C, heads * head_dim]`, and splits its product into heads; otherwise
    it is `[C, heads, head_dim]` (`DenseGeneral`), Dew's own, whose heads a
    tensor axis splits. A checkpoint's reader maps one layout or the other, so
    each attention keeps its own. `norm` normalizes each head's channels, or
    with `across_heads` each token's whole projection before it splits, as
    Wan's `rms_norm_across_heads` does: another function, not another order.
    """
    width = heads * head_dim
    if flat:
        projected = Dense(width, use_bias=bias, dtype=dtype, precision=precision, name=name,
                          dot_general=float32_dot_general)(x)
    else:
        projected = DenseGeneral((heads, head_dim), use_bias=bias, dtype=dtype, precision=precision,
                                 name=name, dot_general=float32_dot_general)(x)
    if norm is not None and across_heads:
        projected = norm(projected.reshape(*x.shape[:-1], width))
    projected = projected.reshape(*x.shape[:-1], heads, head_dim)
    return norm(projected) if norm is not None and not across_heads else projected


def projected_output(attended, name: str, features: int, *, flat: bool, bias: bool = True,
                     dtype: Dtype | None = None, precision: PrecisionLike = None) -> jax.Array:
    """Heads `[..., heads, head_dim]` joined by the matrix `name` to `[..., features]`
    in the calling module, the matrix stored as `projected_heads` stores its
    own: `[heads * head_dim, features]` flat, else `[heads, head_dim, features]`."""
    if flat:
        # The width is named, not inferred: an empty stream (no context tokens) has nothing to infer it from.
        return Dense(features, use_bias=bias, dtype=dtype, precision=precision, name=name,
                     dot_general=float32_dot_general)(
            attended.reshape(*attended.shape[:-2], math.prod(attended.shape[-2:])))
    return DenseGeneral(features, axis=(-2, -1), use_bias=bias, dtype=dtype, precision=precision,
                        name=name, dot_general=float32_dot_general)(attended)


@dataclasses.dataclass(frozen=True)
class QKNorm:
    """How an attention RMS-normalizes its queries and keys: the norms' module `names`, their
    `epsilon`, whether over each head's channels or, `across_heads`, each token's whole projection
    before it splits (Wan's `rms_norm_across_heads`: another function, not another order), and
    whether the normalized row rounds to the weight's dtype before it scales (`scale_after_cast`,
    diffusers' `RMSNorm`)."""

    names: tuple[str, str] = ("q_norm", "k_norm")
    epsilon: float = 1e-6
    across_heads: bool = False
    scale_after_cast: bool = False


CROSS_KEY = "cross_key"
"""The cache variable of a cached call's context keys, `[rows, context_length, kv_heads, head_dim]`."""


CROSS_VALUE = "cross_value"
"""The cache variable of a cached call's context values, shaped as its keys."""


@logical_axes({
    ("to_q",): ("embed", "heads", "head_dim"),
    ("to_k",): ("embed", "heads", "head_dim"),
    ("to_v",): ("embed", "heads", "head_dim"),
    ("to_out_0",): ("heads", "head_dim", "embed"),
})
class NormalAttention(nn.Module):
    """Attend over a `[B, S, C]` or `[B, H, W, C]` input with multiple heads.

    `freqs_cis` rotates the queries and keys of a self-attention call;
    `rotary_freqs` gives the pair, and None leaves them unrotated.
    `use_bias` is the projections' bias; `qkv_bias`, when given, the
    query, key and value projections' alone, as timm's ViT attention
    (U-ViT's) has them without one and its output projection with one.
    `flat` stores each projection as the published attentions do,
    `[C, heads * head_dim]`, rather than Dew's headed `[C, heads, head_dim]`
    (`projected_heads`), under `names` (query, key, value, output): Wan's
    are the defaults, the AutoencoderKL mid-block's `query/key/value/
    proj_attn`. `qk_norm` normalizes the queries and keys (`QKNorm`), and
    `rotary_pairs` names the channels each angle of `freqs_cis` turns
    (`dew.nn.rope.rotate`).
    `mask`, boolean and broadcasting to `[B, H, S, S_context]`, keeps the keys
    each query may read. `key_value_seq_lengths` `[B]` ends each row's keys
    instead, for a right-padded context, which a fused kernel skips rather
    than reading as a dense bias per head. `segment_ids` `[B, S]` keeps each
    query to its packed document's keys, `key_segment_ids` `[B, S_context]`
    numbering the context's (`CrossDocuments`), at most `max_segments` a
    row, which a fused kernel runs as packed sequences rather than a mask.

    It is every cross-attention's attention, the diffusion blocks' and Wan's
    and a decoder's gated block's alike (`dew.nn.cross_attention`): `kv_heads`
    groups the queries over fewer key-value heads, and with `cache_length` a
    `cached` call keeps a context's keys and values (`CROSS_KEY`,
    `CROSS_VALUE`), so a decode step without the context reads them.
    """
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    use_bias: bool = True
    qkv_bias: bool | None = None
    force_fp32_for_softmax: bool = True
    qk_norm: QKNorm | None = None
    attention_impl: AttentionImpl = "auto"
    flat: bool = False
    names: tuple[str, str, str, str] = ("to_q", "to_k", "to_v", "to_out_0")
    rotary_pairs: Literal["half", "adjacent"] = "half"
    kv_heads: int | None = None
    """The keys' and values' heads, which the query heads share in equal groups; None gives each
    query head its own."""
    cache_length: int | None = None
    """The most context tokens a row's cached call keeps the keys and values of; None keeps none."""

    @property
    def key_heads(self) -> int:
        return self.heads if self.kv_heads is None else self.kv_heads

    @nn.nowrap
    def projections(self, context_features: int | None = None) -> dict[str, tuple[int, int]]:
        inner, keys = self.heads * self.dim_head, self.key_heads * self.dim_head
        context = self.query_dim if context_features is None else context_features
        query, key, value, output = self.names
        return {query: (self.query_dim, inner), key: (context, keys), value: (context, keys),
                output: (inner, self.query_dim)}

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """Self-attention maps and its full QK/AV products on real rows."""
        return self_attention_flops(lengths, self.projections(), self.heads, self.dim_head)

    @nn.nowrap
    def cross_flops(self, lengths, context_lengths, context_features: int) -> float:
        """Cross-attention with the context geometry supplied by its owner."""
        maps = self.projections(context_features)
        query, key, value, output = self.names
        return (sum(projection_flops(lengths, *maps[name]) for name in (query, output))
                + sum(projection_flops(context_lengths, *maps[name]) for name in (key, value))
                + cross_attention_flops(lengths, context_lengths, self.heads, self.dim_head, self.dim_head))

    def _heads(self, name: str, x, norm: str | None = None, *, keyed: bool = False):
        """`x` projected to `[B, S, heads, head_dim]`, the key-value heads where `keyed`."""
        normed = self.qk_norm if norm is not None else None
        rms = None if normed is None else RMSNorm(
            epsilon=normed.epsilon, scale_after_cast=normed.scale_after_cast, dtype=self.dtype, name=norm)
        projected = projected_heads(
            x, name, self.key_heads if keyed else self.heads, self.dim_head, flat=self.flat,
            bias=self.use_bias if self.qkv_bias is None else self.qkv_bias, dtype=self.dtype,
            precision=self.precision, across_heads=normed is not None and normed.across_heads, norm=rms)
        return constrain(projected, KV_HEADS if keyed and self.kv_heads is not None else HEADS)

    def _keys_values(self, context):
        """The context's keys and values, its positions split over a sequence axis where its link
        pays for it (`split_positions`)."""
        _, k_name, v_name, _ = self.names
        k_norm = self.qk_norm.names[1] if self.qk_norm is not None else None
        return split_positions(context, (self.key_heads, self.dim_head),
                               lambda tokens: (self._heads(k_name, tokens, k_norm, keyed=True),
                                               self._heads(v_name, tokens, keyed=True)))

    def _cached(self, rows: int, dtype, context):
        """The keys and values a cached call reads. A call that carries a context computes them
        from it and, once the cache is allocated, keeps them; a call without one reads what the
        cache keeps. As with a decoder's own cache, the first cached call allocates the cache and
        writes nothing. The call reads the keys and values as the cache keeps them, in `dtype`."""
        if self.cache_length is None:
            raise ValueError("a cached call keeps a context's keys and values, which this attention has "
                             "no cache_length for")
        allocated = self.has_variable("cache", CROSS_KEY)
        shape = (rows, self.cache_length, self.key_heads, self.dim_head)
        keys = self.variable("cache", CROSS_KEY, jnp.zeros, shape, dtype)
        values = self.variable("cache", CROSS_VALUE, jnp.zeros, shape, dtype)
        if context is None:
            return keys.value, values.value
        width = context.shape[1]
        if width > self.cache_length:
            raise ValueError(f"a cached call keeps at most {self.cache_length} context tokens a row "
                             f"(CrossAttention.context_length), got {width}")
        key, value = (projected.astype(dtype) for projected in self._keys_values(context))
        if allocated:
            padding = ((0, 0), (0, self.cache_length - width), (0, 0), (0, 0))
            keys.value = jnp.pad(key, padding)
            values.value = jnp.pad(value, padding)
        return key, value

    @nn.compact
    def __call__(self, x, context=None, freqs_cis=None, mask=None, key_value_seq_lengths=None,
                 segment_ids=None, key_segment_ids=None, max_segments=None, *, cached: bool = False,
                 cache_dtype=None):
        """`cached` reads and keeps the context's keys and values in the cache (`_cached`), in
        `cache_dtype` (the module's dtype by default); the context is then None on a step that reads
        the keys and values its prefill kept."""
        orig_x_shape = x.shape
        if len(x.shape) == 4:
            x = x.reshape((x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        if not cached:
            context = x if context is None else context
        if context is not None and len(context.shape) == 4:
            context = context.reshape(
                (context.shape[0], context.shape[1] * context.shape[2], context.shape[3]))
        # [B, S, heads, head_dim], column-parallel under a tensor axis.
        q_name, _, _, out_name = self.names
        q_norm = self.qk_norm.names[0] if self.qk_norm is not None else None
        query = self._heads(q_name, x, q_norm)
        if cached:
            dtype = cache_dtype or self.dtype or x.dtype
            key, value = self._cached(x.shape[0], dtype, context)
        else:
            key, value = self._keys_values(context)
        if freqs_cis is not None:
            query, key = (rotate(part, *freqs_cis, pairs=self.rotary_pairs) for part in (query, key))

        hidden_states = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            implementation=self.attention_impl, mask=mask, key_value_seq_lengths=key_value_seq_lengths,
            segment_ids=segment_ids, key_segment_ids=key_segment_ids, max_segments=max_segments,
        )
        proj = projected_output(constrain(hidden_states, HEADS), out_name, self.query_dim, flat=self.flat,
                                bias=self.use_bias, dtype=self.dtype, precision=self.precision)
        return proj.reshape(orig_x_shape)


class BasicTransformerBlock(nn.Module):
    """Run a first attention, cross-attention over `context`, then feed-forward.

    Each is pre-normed (`norm`, RMS or affine layer norm, at `norm_epsilon`)
    with a residual. `first_attention` names what the first attention reads:
    the tokens themselves (`self`), the `context` too (`context`, the SD
    checkpoints with no self-attention), or nothing, when the block has none
    (`none`). `only_pure_attention` runs the cross-attention alone, with no
    norm and no residual, which the UNets' stages do by default.
    `qkv_bias` is the attentions' (`NormalAttention`). `dropout_rate` drops
    each attention's output, inside the feed-forward and the block's output,
    as diffusers' Flax `FlaxBasicTransformerBlock` does in training (its
    PyTorch block drops no residual stream); `approximate_gelu` is the
    feed-forward's GELU (`geglu_feed_forward`).
    """
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    use_bias: bool = True
    qkv_bias: bool | None = None
    first_attention: Literal["self", "context", "none"] = "self"
    only_pure_attention:bool = False
    force_fp32_for_softmax: bool = True
    norm: Literal["rms", "layer"] = "rms"
    norm_epsilon: float = 1e-4
    dropout_rate: float = 0.0
    approximate_gelu: bool = True
    attention_impl: AttentionImpl = "auto"

    @nn.nowrap
    def cross_flops(self, lengths, context_lengths, context_features: int) -> float:
        """The configured attention branches and optional GEGLU maps."""
        from .mlp import geglu_feed_forward  # mlp's blocks import this module's norms

        attention = NormalAttention(self.query_dim, self.heads, self.dim_head, parent=None)
        crossed = attention.cross_flops(lengths, context_lengths, context_features)
        count = crossed
        if not self.only_pure_attention:
            first = {"self": attention.model_flops(lengths), "context": crossed, "none": 0}
            count += first[self.first_attention]
            count += geglu_feed_forward(self.query_dim, parent=None).model_flops(lengths)
        return count

    def setup(self):
        attention = functools.partial(
            NormalAttention,
            query_dim=self.query_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            precision=self.precision,
            use_bias=self.use_bias,
            qkv_bias=self.qkv_bias,
            dtype=self.dtype,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            attention_impl=self.attention_impl,
        )
        self.attention1 = attention(name='Attention1')
        self.attention2 = attention(name='Attention2')

        from .mlp import geglu_feed_forward  # mlp's blocks import this module's norms

        self.ff = geglu_feed_forward(self.query_dim, approximate=self.approximate_gelu, dtype=self.dtype,
                                     precision=self.precision, dropout_rate=self.dropout_rate)
        norm = RMSNorm if self.norm == "rms" else LayerNorm
        self.norm1, self.norm2, self.norm3 = (norm(epsilon=self.norm_epsilon, dtype=self.dtype)
                                              for _ in range(3))
        self.dropout = Dropout(self.dropout_rate)

    def _dropped(self, x, train: bool):
        return self.dropout(x, deterministic=not train)

    def __call__(self, x, context=None, *, train: bool = False):
        if self.only_pure_attention:
            return self.attention2(x, context)

        if self.first_attention != "none":
            first = context if self.first_attention == "context" else None
            x = x + self._dropped(self.attention1(self.norm1(x), first), train)
        x = x + self._dropped(self.attention2(self.norm2(x), context), train)
        x = x + self.ff(self.norm3(x), train=train)
        return self._dropped(x, train)


@dataclasses.dataclass(frozen=True)
class Stage:
    """One resolution stage's attention in a UNet; a stage without is `None`.

    Every field is a `TransformerBlock` setting. The head width is the stage's
    channel count over `heads`, which the unet knows and a config does not.
    `dew.registry.from_record` builds one from a record (`{"heads": 8}` from a
    command line), so a misspelled field raises at the build boundary.
    `dtype` and `precision` of None mean the model's. The softmax runs in
    fp32 by default, as every fused kernel computes it.
    """

    heads: int
    use_projection: bool = False
    use_self_and_cross: bool = True
    only_pure_attention: bool = True
    force_fp32_for_softmax: bool = True
    norm_inputs: bool = True
    explicitly_add_residual: bool = True
    norm_epsilon: float = 1e-4
    dtype: Dtype | None = None
    precision: PrecisionLike = None


def stage_attention(stage: Stage, channels: int, attention_impl: AttentionImpl, dtype: Dtype | None,
                    precision: PrecisionLike, name: str) -> "TransformerBlock":
    """Build the block a `Stage` describes, at the stage's channel count.

    A stage that names no dtype or precision takes the model's, and a model
    that names no dtype attends in fp32.
    """
    return TransformerBlock(
        heads=stage.heads, dim_head=channels // stage.heads,
        dtype=stage.dtype or dtype or jnp.float32,
        attention_impl=attention_impl, use_projection=stage.use_projection,
        use_self_and_cross=stage.use_self_and_cross,
        precision=stage.precision or precision,
        only_pure_attention=stage.only_pure_attention,
        force_fp32_for_softmax=stage.force_fp32_for_softmax,
        norm_inputs=stage.norm_inputs, explicitly_add_residual=stage.explicitly_add_residual,
        norm_epsilon=stage.norm_epsilon,
        name=name)


class TransformerBlock(nn.Module):
    """Run a `BasicTransformerBlock`, optionally at its own head width.

    `use_projection` puts a dense projection into and out of
    `heads * dim_head` around the block; without it the block runs at the
    input width.
    """
    heads: int = 4
    dim_head: int = 32
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    use_projection: bool = False
    use_self_and_cross:bool = True
    only_pure_attention:bool = False
    force_fp32_for_softmax: bool = True
    attention_impl: AttentionImpl = "auto"
    norm_inputs: bool = True
    explicitly_add_residual: bool = True
    norm_epsilon: float = 1e-4

    @nn.nowrap
    def cross_flops(self, lengths, input_features: int, context_lengths=None,
                    context_features: int | None = None) -> float:
        """Stage projections and its self/cross/FFN branches at actual widths."""
        inner = self.heads * self.dim_head if self.use_projection else input_features
        context_lengths = lengths if context_lengths is None else context_lengths
        context_features = inner if context_features is None else context_features
        block = BasicTransformerBlock(inner, self.heads, self.dim_head,
                                       first_attention="self" if self.use_self_and_cross else "none",
                                       only_pure_attention=self.only_pure_attention, parent=None)
        count = block.cross_flops(lengths, context_lengths, context_features)
        return count + (2 * projection_flops(lengths, input_features, inner) if self.use_projection else 0)

    @nn.compact
    def __call__(self, x, context=None):
        channels = x.shape[-1]
        if self.norm_inputs:
            x = RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)(x)

        def project(features: int, name: str):
            return Dense(features=features, use_bias=False, precision=self.precision,
                         dtype=self.dtype, name=name, dot_general=float32_dot_general)

        if self.use_projection:
            inner_dim = self.heads * self.dim_head
            hidden = project(inner_dim, 'project_in')(x)
        else:
            inner_dim, hidden = channels, x
        hidden = BasicTransformerBlock(
            query_dim=inner_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            name='Attention',
            precision=self.precision,
            use_bias=False,
            dtype=self.dtype,
            first_attention="self" if self.use_self_and_cross else "none",
            only_pure_attention=self.only_pure_attention,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            attention_impl=self.attention_impl,
            norm_epsilon=self.norm_epsilon
        )(hidden, hidden if context is None else context)
        if self.use_projection:
            hidden = project(channels, 'project_out')(hidden)

        if self.only_pure_attention or self.explicitly_add_residual:
            hidden = x + hidden
        return hidden
