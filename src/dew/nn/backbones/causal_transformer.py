"""The autoregressive transformer decoder every language model here trains.

Token embedding, rotary positions, pre-norm blocks of grouped-query causal
attention and a gated MLP, a final RMSNorm, and an fp32 head. Attention goes
through the one shared kernel path in dew.nn.attention, so a run picks
reference/xla/cudnn/tpu the same way a diffusion run does, and decoding reuses
the same fixed-size KV cache helpers.

Parameter names mirror the HF decoder layout - embed_tokens,
layers_N.{input_layernorm, self_attn.{q,k,v,o}_proj, post_attention_layernorm,
mlp.{gate,up,down}_proj}, norm, lm_head. A model family is supported only
after its translator and same-weight reference parity test land. Gemma's two
extra norms are the exception: HF calls them post_attention_layernorm and
post_feedforward_layernorm even though they normalize sublayer outputs, so
here they are attention_output_norm and mlp_output_norm and the pre-norms keep
their names. dew.interop.hf_decoders does that rename.

The block holds its token mixer in a slot: any module with the
(x, decode=...) -> x signature of CausalSelfAttention becomes self_attn
without the block changing, which is where a linear-attention mixer goes.
"""

import functools
import math
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from ..attention import (
    causal_attention_mask, open_kv_cache, scaled_dot_product_attention,
)


LAYER_TYPES = ('full_attention', 'sliding_attention')


class RMSNorm(nn.Module):
    """RMSNorm normalized in fp32, with Gemma's (1 + w) scale behind a flag.

    scale_offset also flips the initializer to zeros, so the identity is the
    starting point either way and a Gemma checkpoint's stored weights land
    unchanged.
    """
    epsilon: float = 1e-5
    scale_offset: bool = False
    dtype: Optional[Dtype] = None

    @nn.compact
    def __call__(self, x):
        scale = self.param(
            'scale',
            nn.initializers.zeros if self.scale_offset else nn.initializers.ones,
            (x.shape[-1],), jnp.float32)
        y = x.astype(jnp.float32)
        y = y * jax.lax.rsqrt(jnp.mean(jnp.square(y), axis=-1, keepdims=True) + self.epsilon)
        y = y * (1.0 + scale) if self.scale_offset else y * scale
        return y.astype(self.dtype if self.dtype is not None else x.dtype)


def rotary_freqs(positions, head_dim: int, theta: float):
    """cos/sin of the rotary angles at absolute `positions`: [P, head_dim // 2].

    Computed in fp32 so a token gets the same rotation whether it arrives in a
    prefill or comes back as a single decode step.
    """
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    angles = jnp.asarray(positions, jnp.float32)[:, None] * inv_freq[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def apply_rotary(x, freqs_cos, freqs_sin):
    """Rotate [B, S, H, D] heads, rotate-half convention as in the HF decoders."""
    cos = jnp.concatenate([freqs_cos, freqs_cos], axis=-1)[None, :, None, :]
    sin = jnp.concatenate([freqs_sin, freqs_sin], axis=-1)[None, :, None, :]
    fp32 = x.astype(jnp.float32)
    x1, x2 = jnp.split(fp32, 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return (fp32 * cos + rotated * sin).astype(x.dtype)


class CausalSelfAttention(nn.Module):
    """Causal self-attention with grouped-query heads, rotary positions, qk
    RMSNorm and a fixed-size KV cache.

    decode=True runs the call against the cache: the first call writes the
    whole prompt and each later call appends one token, so prefill and decode
    are one code path. Keys are rotated before they enter the cache, which is
    why the rotary positions come from the cache index rather than from the row
    index of the token.
    """
    emb_features: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    rope_theta: float = 10000.0
    qk_norm: bool = True
    norm_eps: float = 1e-5
    scale_offset: bool = False
    sliding_window: Optional[int] = None
    attention_bias: bool = False  # q/k/v/o biases, as config.attention_bias in HF
    attention_scale: Optional[float] = None  # None: the kernel's own 1/sqrt(head_dim)
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    attention_impl: Optional[str] = None
    force_fp32_for_softmax: bool = True

    def setup(self):
        dense = functools.partial(
            nn.Dense, use_bias=self.attention_bias, dtype=self.dtype, precision=self.precision)
        self.q_proj = dense(self.num_heads * self.head_dim, name='q_proj')
        self.k_proj = dense(self.num_kv_heads * self.head_dim, name='k_proj')
        self.v_proj = dense(self.num_kv_heads * self.head_dim, name='v_proj')
        self.o_proj = dense(self.emb_features, name='o_proj')
        if self.qk_norm:
            self.q_norm = RMSNorm(
                epsilon=self.norm_eps, scale_offset=self.scale_offset,
                dtype=self.dtype, name='q_norm')
            self.k_norm = RMSNorm(
                epsilon=self.norm_eps, scale_offset=self.scale_offset,
                dtype=self.dtype, name='k_norm')

    @nn.compact
    def __call__(self, x, decode: bool = False):
        B, S, _ = x.shape
        query = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim)
        key = self.k_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim)
        value = self.v_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim)
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

        # The cache slot carries position while decoding, so the rotation and
        # the mask both read it instead of the row index of the token.
        append = None
        if decode:
            positions, append = open_kv_cache(self, key, self.max_seq_len)
        else:
            positions = jnp.arange(S)
        freqs_cos, freqs_sin = rotary_freqs(positions, self.head_dim, self.rope_theta)
        query = apply_rotary(query, freqs_cos, freqs_sin)
        key = apply_rotary(key, freqs_cos, freqs_sin)
        if self.attention_scale is not None:
            # Every kernel path scales the logits by 1/sqrt(head_dim) itself, so
            # the query carries the ratio to the scale the checkpoint asks for.
            query = query * jnp.asarray(
                self.attention_scale * math.sqrt(self.head_dim), query.dtype)

        causal, mask = True, None
        if append is not None:
            key, value = append(key, value)
            mask = causal_attention_mask(positions, key.shape[-3], self.sliding_window)
            causal = False

        attention = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            implementation=self.attention_impl, causal=causal,
            sliding_window=None if decode else self.sliding_window, mask=mask)
        return self.o_proj(attention.reshape(B, S, self.num_heads * self.head_dim))


class GatedMLP(nn.Module):
    """down_proj(act(gate_proj(x)) * up_proj(x)): swiglu is silu, geglu is gelu.

    Bias-free, like the gated MLP of every open decoder this loads.
    """
    hidden_features: int
    out_features: int
    activation: str = 'swiglu'
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        if self.activation not in ('swiglu', 'geglu'):
            raise ValueError(
                f"mlp must be 'swiglu' or 'geglu', got {self.activation!r}")
        dense = functools.partial(
            nn.Dense, use_bias=False, dtype=self.dtype, precision=self.precision)
        self.gate_proj = dense(self.hidden_features, name='gate_proj')
        self.up_proj = dense(self.hidden_features, name='up_proj')
        self.down_proj = dense(self.out_features, name='down_proj')

    def __call__(self, x):
        gate = self.gate_proj(x)
        gate = nn.silu(gate) if self.activation == 'swiglu' else nn.gelu(gate)
        return self.down_proj(gate * self.up_proj(x))


class DecoderBlock(nn.Module):
    """Pre-norm decoder block: token mixer, then gated MLP, both residual.

    `mixer` is a factory taking only a name; whatever it builds lands in the
    tree as self_attn and only has to accept (x, decode=...).

    sandwich_norms adds Gemma's second pair of norms, on the output of each
    sublayer rather than on its input; the pre-norms keep their names and their
    places, so a checkpoint without them loads into the same tree minus two
    leaves per layer.
    """
    mixer: Callable[..., nn.Module]
    emb_features: int
    mlp_features: int
    mlp_activation: str = 'swiglu'
    norm_eps: float = 1e-5
    scale_offset: bool = False
    sandwich_norms: bool = False
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        norm = functools.partial(
            RMSNorm, epsilon=self.norm_eps, scale_offset=self.scale_offset, dtype=self.dtype)
        self.input_layernorm = norm(name='input_layernorm')
        self.self_attn = self.mixer(name='self_attn')
        self.post_attention_layernorm = norm(name='post_attention_layernorm')
        if self.sandwich_norms:
            self.attention_output_norm = norm(name='attention_output_norm')
            self.mlp_output_norm = norm(name='mlp_output_norm')
        self.mlp = GatedMLP(
            hidden_features=self.mlp_features,
            out_features=self.emb_features,
            activation=self.mlp_activation,
            dtype=self.dtype,
            precision=self.precision,
            name='mlp')
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, train: bool = False, decode: bool = False):
        mixed = self.self_attn(self.input_layernorm(x), decode=decode)
        if self.sandwich_norms:
            mixed = self.attention_output_norm(mixed)
        x = x + self.dropout(mixed, deterministic=not train)
        hidden = self.mlp(self.post_attention_layernorm(x))
        if self.sandwich_norms:
            hidden = self.mlp_output_norm(hidden)
        return x + self.dropout(hidden, deterministic=not train)


class CausalTransformer(nn.Module):
    """Decoder-only transformer over token ids: [B, S] int32 -> [B, S, vocab] fp32.

    The defaults are a from-scratch training recipe (multi-head attention,
    swiglu, tied embeddings, no softcap); the fields that differ between the
    open decoders are all here, so a Qwen3 or Gemma3 config is a field
    mapping: num_kv_heads/head_dim for grouped-query attention,
    attention_bias, rope_theta with rope_local_theta for Gemma3's two rope
    bases, norm_eps with scale_offset for Gemma's (1 + w) norms and
    sandwich_norms for its second pair of them, attention_scale for its
    query_pre_attn_scalar, embedding_scale and final_logit_softcap for the
    rest of Gemma, and
    layer_types with sliding_window for the mixed full/sliding stacks. How a
    checkpoint's config derives layer_types (Qwen3 makes every layer past
    max_window_layers sliding) belongs to that translation, not here: this
    takes the tuple.
    """
    vocab_size: int
    emb_features: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: Optional[int] = None       # None: as many as the query heads
    head_dim: Optional[int] = None           # None: emb_features // num_heads
    mlp: str = 'swiglu'                      # 'swiglu' | 'geglu'
    mlp_ratio: int = 4
    mlp_features: Optional[int] = None       # None: mlp_ratio * emb_features
    max_seq_len: int = 2048
    rope_theta: float = 10000.0              # full attention layers
    rope_local_theta: Optional[float] = None  # sliding layers, None: rope_theta
    layer_types: Optional[Tuple[str, ...]] = None  # per layer, see LAYER_TYPES
    sliding_window: Optional[int] = None     # keys a sliding layer keeps
    norm_eps: float = 1e-5
    scale_offset: bool = False               # Gemma's (1 + w) RMSNorm scale
    sandwich_norms: bool = False             # Gemma's norms on the sublayer outputs
    qk_norm: bool = True
    attention_bias: bool = False             # q/k/v/o biases (Qwen2-style)
    attention_scale: Optional[float] = None  # None: head_dim ** -0.5
    embedding_scale: bool = False            # Gemma scales embeddings by sqrt(d)
    final_logit_softcap: Optional[float] = None
    tie_embeddings: bool = True
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    attention_impl: Optional[str] = None

    def __post_init__(self):
        if self.layer_types is not None:
            object.__setattr__(self, "layer_types", tuple(self.layer_types))
        super().__post_init__()


    @property
    def kv_heads(self) -> int:
        return self.num_heads if self.num_kv_heads is None else self.num_kv_heads

    @property
    def features_per_head(self) -> int:
        return (self.emb_features // self.num_heads
                if self.head_dim is None else self.head_dim)

    @property
    def hidden_features(self) -> int:
        return (self.mlp_ratio * self.emb_features
                if self.mlp_features is None else self.mlp_features)

    @property
    def per_layer_types(self) -> Tuple[str, ...]:
        if self.layer_types is None:
            return ('full_attention',) * self.num_layers
        return tuple(self.layer_types)

    def setup(self):
        head_dim = self.features_per_head
        if head_dim % 2:
            raise ValueError(
                f"rotary positions rotate pairs, so head_dim must be even, got {head_dim}")
        if self.num_heads % self.kv_heads:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be a multiple of num_kv_heads "
                f"({self.kv_heads})")
        types = self.per_layer_types
        if len(types) != self.num_layers:
            raise ValueError(
                f"layer_types has {len(types)} entries for {self.num_layers} layers")
        unknown = sorted(set(types) - set(LAYER_TYPES))
        if unknown:
            raise ValueError(
                f"unknown layer types {unknown}, expected one of {list(LAYER_TYPES)}")
        if 'sliding_attention' in types and self.sliding_window is None:
            raise ValueError("sliding attention layers need sliding_window set")

        self.embed_tokens = nn.Embed(
            num_embeddings=self.vocab_size, features=self.emb_features,
            dtype=self.dtype, name='embed_tokens')
        self.layers = [
            DecoderBlock(
                mixer=functools.partial(
                    CausalSelfAttention,
                    emb_features=self.emb_features,
                    num_heads=self.num_heads,
                    num_kv_heads=self.kv_heads,
                    head_dim=head_dim,
                    max_seq_len=self.max_seq_len,
                    rope_theta=(self.rope_local_theta
                                if layer_type == 'sliding_attention'
                                and self.rope_local_theta is not None
                                else self.rope_theta),
                    qk_norm=self.qk_norm,
                    norm_eps=self.norm_eps,
                    scale_offset=self.scale_offset,
                    sliding_window=(self.sliding_window
                                    if layer_type == 'sliding_attention' else None),
                    attention_bias=self.attention_bias,
                    attention_scale=self.attention_scale,
                    dtype=self.dtype,
                    precision=self.precision,
                    attention_impl=self.attention_impl,
                    force_fp32_for_softmax=self.force_fp32_for_softmax),
                emb_features=self.emb_features,
                mlp_features=self.hidden_features,
                mlp_activation=self.mlp,
                norm_eps=self.norm_eps,
                scale_offset=self.scale_offset,
                sandwich_norms=self.sandwich_norms,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                precision=self.precision,
                name=f'layers_{index}')
            for index, layer_type in enumerate(types)]
        self.norm = RMSNorm(
            epsilon=self.norm_eps, scale_offset=self.scale_offset,
            dtype=self.dtype, name='norm')
        if not self.tie_embeddings:
            self.lm_head = nn.Dense(
                features=self.vocab_size, use_bias=False, dtype=jnp.float32,
                precision=self.precision, name='lm_head')

    def __call__(self, tokens, train: bool = False, decode: bool = False):
        x = self.embed_tokens(tokens)
        if self.embedding_scale:
            # Gemma casts sqrt(hidden) to the embedding's own dtype, which is
            # the parameter dtype, so the multiply happens in fp32 and only
            # the product is narrowed. Scaling in bf16 instead would round the
            # factor itself: sqrt(1152) becomes 34.0, off by 1.7e-3.
            x = (x.astype(jnp.float32)
                 * math.sqrt(self.emb_features)).astype(x.dtype)
        for layer in self.layers:
            x = layer(x, train=train, decode=decode)
        x = self.norm(x)

        # fp32 head, as in the DiT output projection: the loss is computed in fp32
        if self.tie_embeddings:
            logits = jnp.einsum(
                '...d,vd->...v', x.astype(jnp.float32),
                self.embed_tokens.embedding.astype(jnp.float32),
                precision=self.precision)
        else:
            logits = self.lm_head(x)
        logits = logits.astype(jnp.float32)
        if self.final_logit_softcap is not None:
            cap = jnp.asarray(self.final_logit_softcap, jnp.float32)
            logits = cap * jnp.tanh(logits / cap)
        return logits

    def init_cache(self, batch_size: int):
        """Allocate a zeroed decode cache for `batch_size` sequences.

        cache = model.apply(params, batch_size, method=CausalTransformer.init_cache,
                            mutable=['cache'])[1]['cache']

        The forward pass this runs is a single dummy token whose keys are never
        written: allocation happens on the first decode-mode call, the write on
        the ones after it.
        """
        self(jnp.zeros((batch_size, 1), jnp.int32), decode=True)
