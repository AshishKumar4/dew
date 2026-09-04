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
(x, decode=..., positions=..., segment_ids=...) -> x signature of
CausalSelfAttention becomes self_attn without the block changing, which is
where a linear-attention mixer goes.
"""

import dataclasses
import functools
import math
from typing import Callable, Mapping, Optional, Tuple

import flax.core
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from ..attention import (
    causal_attention_mask, open_kv_cache, scaled_dot_product_attention,
)
from ..moe import SparseMLP
from ..sharding import logical_axes
from dew.registry import models


@dataclasses.dataclass(frozen=True)
class LayerKind:
    """What the layers of one kind in the pattern do differently.

    The pattern already names each layer's kind, so what the kind means
    belongs here rather than in a field name: a windowed kind is what
    "sliding attention" used to say, and `rope_theta` and `head_dim` are the
    model's unless this kind states its own. Rotary positions rotate every
    dimension of a windowed kind, which is where Gemma 4 puts its partial
    rotary (the global layers') and where the sliding layers rotate whole.
    """

    window: Optional[int] = None
    """Keys a layer of this kind attends, its own included; None attends all."""
    rope_theta: Optional[float] = None
    head_dim: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class ResolvedKind:
    """One kind of layer with the model's defaults filled in.

    `LayerKind` is what a config states, so a field it leaves to the model is
    None there. This is what the model resolved it to, so `rope_theta` and
    `head_dim` are numbers; only the window stays optional, because attending
    the whole sequence is what a kind without one does.
    """

    window: Optional[int]
    rope_theta: float
    head_dim: int


@dataclasses.dataclass(frozen=True)
class Mixture:
    """The experts some layers route to, and how the router chooses.

    `experts` is what the rest depends on, which is why they live together:
    a top_k, a cadence or a balancing bias says nothing about a model with
    no experts. `layers` names the sparse layers by index, or `every` makes
    every nth layer sparse counting from the end of the first group, which
    is what Qwen3-MoE's decoder_sparse_step means; neither makes every layer
    sparse, which is Mixtral.

    The routing options are `Router`'s: `score_function` softmax or sigmoid,
    `scaling` on the routed output, `groups` with `groups_per_token` for
    DeepSeek's node limit, and `bias` for its aux-loss-free balancing bias.
    """

    experts: int
    top_k: int = 2
    layers: Optional[Tuple[int, ...]] = None
    every: Optional[int] = None
    score_function: str = 'softmax'
    scaling: float = 1.0
    groups: int = 1
    groups_per_token: int = 1
    bias: bool = False

    def __post_init__(self):
        if self.layers is not None:
            object.__setattr__(self, "layers", tuple(self.layers))
        if self.experts < 1:
            raise ValueError(
                f"a mixture needs experts to route to, got {self.experts}; a "
                "dense model has no mixture at all")
        if self.layers is not None and self.every is not None:
            raise ValueError(
                f"layers ({self.layers}) and every ({self.every}) both choose the "
                "sparse layers, so only one of them can be set")
        if self.every is not None and self.every < 1:
            raise ValueError(f"every must be positive, got {self.every}")


class RMSNorm(nn.Module):
    """RMSNorm normalized in fp32, with Gemma's (1 + w) scale behind a flag.

    scale_offset also flips the initializer to zeros, so the identity is the
    starting point either way and a Gemma checkpoint's stored weights land
    unchanged.

    The families differ in where the scale meets the activation dtype. Gemma
    multiplies in fp32 and casts the product (modeling_gemma3.py:147-150);
    Llama and Qwen3 cast the normalized activations first and multiply by
    the scale in that dtype (modeling_qwen3.py:61-64), which
    scale_after_cast reproduces. The two agree at fp32 and differ under bf16.
    """
    epsilon: float = 1e-5
    scale_offset: bool = False
    scale_after_cast: bool = False
    with_scale: bool = True
    dtype: Optional[Dtype] = None

    @nn.compact
    def __call__(self, x):
        dtype = self.dtype if self.dtype is not None else x.dtype
        y = x.astype(jnp.float32)
        y = y * jax.lax.rsqrt(jnp.mean(jnp.square(y), axis=-1, keepdims=True) + self.epsilon)
        if not self.with_scale:
            # A pure normalization with no learned weight, as Gemma 4 norms
            # its values (modeling_gemma4.py, Gemma4RMSNorm with_scale=False).
            return y.astype(dtype)
        scale = self.param(
            'scale',
            nn.initializers.zeros if self.scale_offset else nn.initializers.ones,
            (x.shape[-1],), jnp.float32)
        weight = (1.0 + scale) if self.scale_offset else scale
        if self.scale_after_cast:
            return y.astype(dtype) * weight.astype(dtype)
        return (y * weight).astype(dtype)
def rotary_freqs(positions, head_dim: int, theta: float, rot_dim: int | None = None):
    """cos/sin of the rotary angles at absolute `positions`: [P, head_dim // 2].

    `positions` may be [P] (one sequence) or [B, P] (a packed batch whose
    documents each restart at 0); the angle axes line up with the trailing
    [B, S] either way. Computed in fp32 so a token gets the same rotation
    whether it arrives in a prefill or comes back as a single decode step.

    rot_dim narrows the rotation to the first rot_dim dimensions, the rest
    keeping frequency zero (cosine one, sine zero, so the rotation is the
    identity there). That is proportional partial rotary
    (modeling_rope_utils.py, _compute_proportional_rope_parameters), whose
    exponents run over the full head_dim with a zero tail.
    """
    pairs = head_dim // 2 if rot_dim is None else rot_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, 2 * pairs, 2, dtype=jnp.float32) / head_dim))
    if rot_dim is not None:
        padding = head_dim // 2 - pairs
        inv_freq = jnp.concatenate([inv_freq, jnp.zeros((padding,), jnp.float32)])
    positions = jnp.asarray(positions, jnp.float32)
    if positions.ndim == 1:
        angles = positions[:, None] * inv_freq[None, :]
    else:
        angles = positions[:, :, None] * inv_freq[None, None, :]
    return jnp.cos(angles), jnp.sin(angles)


def apply_rotary(x, freqs_cos, freqs_sin, scale: Optional[float] = None):
    """Rotate [B, S, H, D] heads, rotate-half convention as in the HF decoders.

    The freqs are [S, D] for one sequence, or [B, S, D] when a packed batch
    restarts positions per document. `scale` multiplies the rotated heads
    inside the fp32 arithmetic, so a query's attention scale narrows once,
    with the product.
    """
    cos = jnp.concatenate([freqs_cos, freqs_cos], axis=-1)
    sin = jnp.concatenate([freqs_sin, freqs_sin], axis=-1)
    if cos.ndim == 3:
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]
    else:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    fp32 = x.astype(jnp.float32)
    x1, x2 = jnp.split(fp32, 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    out = fp32 * cos + rotated * sin
    return (out if scale is None else out * scale).astype(x.dtype)


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
    are one code path. Keys are rotated before they enter the cache, which is
    why the rotary positions come from the cache index rather than from the row
    index of the token.

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
    qk_norm: bool = True
    v_norm: bool = False
    norm_eps: float = 1e-5
    scale_offset: bool = False
    scale_after_cast: bool = False
    kv_shared: bool = False
    kv_store_key: Optional[str] = None
    sliding_window: Optional[int] = None
    attention_bias: bool = False  # q/k/v/o biases, as config.attention_bias in HF
    attention_scale: Optional[float] = None  # None: the kernel's own 1/sqrt(head_dim)
    partial_rotary_factor: Optional[float] = None  # None: every head dim rotates
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    attention_impl: Optional[str] = None
    force_fp32_for_softmax: bool = True

    def setup(self):
        dense = functools.partial(
            nn.Dense, use_bias=self.attention_bias, dtype=self.dtype, precision=self.precision)
        self.q_proj = dense(self.num_heads * self.head_dim, name='q_proj')
        # A sharing layer reads another layer's keys and values, so it owns
        # no projections or key norm of its own, exactly as the reference
        # skips them (modeling_gemma4.py, Gemma4TextAttention.__init__).
        if not self.kv_shared:
            self.k_proj = dense(self.num_kv_heads * self.head_dim, name='k_proj')
            self.v_proj = dense(self.num_kv_heads * self.head_dim, name='v_proj')
        self.o_proj = dense(self.emb_features, name='o_proj')
        if self.qk_norm:
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
        """Head dims the rotary rotates, or None for all of them."""
        factor = self.partial_rotary_factor
        if factor is None:
            return None
        rot_dim = int(self.head_dim * factor)
        if not 0 < factor <= 1 or rot_dim % 2:
            raise ValueError(
                f"partial_rotary_factor must rotate an even positive number of "
                f"head dims, got {factor} of head_dim {self.head_dim}")
        return rot_dim

    @nn.compact
    def __call__(self, x, decode: bool = False,
                 positions=None, segment_ids=None, kv_store=None):
        B, S, _ = x.shape
        query = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim)
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
            key = self.k_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim)
            value = self.v_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim)
            if self.qk_norm:
                key = self.k_norm(key)
            if self.v_norm:
                value = self.values_norm(value)
        if self.qk_norm:
            query = self.q_norm(query)

        # The cache slot carries position while decoding, so the rotation and
        # the mask both read it instead of the row index of the token. A
        # packed batch supplies the position inside its document instead of
        # the row index, which is what restarts RoPE at every boundary.
        append = None
        kv_len = key.shape[-3]
        if decode:
            if not self.causal:
                raise ValueError("full attention has no KV cache to decode against")
            if not self.kv_shared:
                positions, append = open_kv_cache(self, key, self.max_seq_len)
        elif positions is None and not self.kv_shared:
            positions = jnp.arange(S)
        elif not self.kv_shared:
            positions = jnp.asarray(positions)
        freqs_cos, freqs_sin = rotary_freqs(
            positions, self.head_dim, self.rope_theta, rot_dim=self._rot_dim())
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
        if self.kv_shared and decode:
            # No cache of its own: the provider's stashed keys carry the full
            # history, so the mask reads them the way the provider's own
            # decode mask does.
            mask = causal_attention_mask(positions, kv_len, self.sliding_window)
            causal = False
        elif append is not None:
            key, value = append(key, value)
            mask = causal_attention_mask(positions, key.shape[-3], self.sliding_window)
            causal = False
            if kv_store is not None and self.kv_store_key is not None:
                kv_store[self.kv_store_key] = (key, value, positions)
        elif segment_ids is not None:
            # Attention stays inside each packed document: the segment ids
            # make the mask block-diagonal, padding (segment 0) sees nothing,
            # and causality (with the layer's window) travels in the same mask
            # rather than as the kernels' flag.
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

        attention = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            implementation=implementation, causal=causal,
            sliding_window=window, mask=mask)
        return self.o_proj(attention.reshape(B, S, self.num_heads * self.head_dim))

@logical_axes({
    ("gate_proj",): ("embed", "mlp"),
    ("up_proj",): ("embed", "mlp"),
    ("down_proj",): ("mlp", "embed"),
})
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


@logical_axes({
    ("per_layer_input_gate",): ("embed", "mlp"),
    ("per_layer_projection",): ("mlp", "embed"),
})
class DecoderBlock(nn.Module):
    """Pre-norm decoder block: token mixer, then feed-forward, both residual.

    `mixer` and `feedforward` are factories taking only a name. What `mixer`
    builds lands in the tree as self_attn and has to accept (x, decode=...,
    positions=..., segment_ids=...), the last two None outside a packed batch.
    What `feedforward` builds lands there as mlp and takes the normalized
    states alone, which is the one call `GatedMLP` and `moe.SparseMLP` share.

    sandwich_norms adds Gemma's second pair of norms, on the output of each
    sublayer rather than on its input; the pre-norms keep their names and their
    places, so a checkpoint without them loads into the same tree minus two
    leaves per layer.

    kv_store threads one dict down the layer stack so a KV-sharing mixer
    reads its provider's keys and values; a mixer without a kv_store keyword
    fails loudly when a run shares. per_layer_input is the layer's input
    signal for the per-layer residual, None when the model has none.
    """
    mixer: Callable[..., nn.Module]
    feedforward: Callable[..., nn.Module]
    emb_features: int
    norm_eps: float = 1e-5
    scale_offset: bool = False
    scale_after_cast: bool = False
    sandwich_norms: bool = False
    per_layer_input_dim: int = 0
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        norm = functools.partial(
            RMSNorm, epsilon=self.norm_eps, scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast, dtype=self.dtype)
        self.input_layernorm = norm(name='input_layernorm')
        self.self_attn = self.mixer(name='self_attn')
        self.post_attention_layernorm = norm(name='post_attention_layernorm')
        if self.sandwich_norms:
            self.attention_output_norm = norm(name='attention_output_norm')
            self.mlp_output_norm = norm(name='mlp_output_norm')
        self.mlp = self.feedforward(name='mlp')
        if self.per_layer_input_dim:
            dense = functools.partial(
                nn.Dense, use_bias=False, dtype=self.dtype, precision=self.precision)
            self.per_layer_input_gate = dense(self.per_layer_input_dim,
                                              name='per_layer_input_gate')
            self.per_layer_projection = dense(self.emb_features,
                                              name='per_layer_projection')
            self.post_per_layer_input_norm = norm(name='post_per_layer_input_norm')
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, train: bool = False, decode: bool = False,
                 positions=None, segment_ids=None, kv_store=None,
                 per_layer_input=None):
        mixed = self.self_attn(self.input_layernorm(x), decode=decode,
                               positions=positions, segment_ids=segment_ids,
                               **({} if kv_store is None else {"kv_store": kv_store}))
        if self.sandwich_norms:
            mixed = self.attention_output_norm(mixed)
        x = x + self.dropout(mixed, deterministic=not train)
        hidden = self.mlp(self.post_attention_layernorm(x))
        if self.sandwich_norms:
            hidden = self.mlp_output_norm(hidden)
        x = x + self.dropout(hidden, deterministic=not train)
        if self.per_layer_input_dim and per_layer_input is not None:
            # Gemma 3n/4's per-layer residual (modeling_gemma4.py,
            # Gemma4TextDecoderLayer): the layer's own gate over x, activated
            # like its feed-forward, multiplied by the layer's input signal,
            # projected back and normed.
            gated = self.per_layer_input_gate(x)
            gated = nn.silu(gated) if self._gate_activation == 'swiglu' else nn.gelu(gated)
            projected = self.per_layer_projection(gated * per_layer_input)
            x = x + self.post_per_layer_input_norm(projected)
        return x

    @property
    def _gate_activation(self) -> str:
        return getattr(self.mlp, 'activation', 'swiglu')
@models("causal_transformer")
@logical_axes({
    ("embed_tokens",): ("vocab", "embed"),
    ("lm_head",): ("embed", "vocab"),
    ("embed_tokens_per_layer",): ("vocab", None),
    ("per_layer_model_projection",): ("embed", "mlp"),
})
class CausalTransformer(nn.Module):
    """Decoder-only transformer over token ids: [B, S] int32 -> [B, S, vocab] fp32.

    The defaults are a from-scratch training recipe (multi-head attention,
    swiglu, tied embeddings, no softcap); the fields that differ between the
    open decoders are all here, so a Qwen3 or Gemma3 config is a field
    mapping: num_kv_heads/head_dim for grouped-query attention,
    attention_bias, norm_eps with scale_offset for Gemma's (1 + w) norms and
    sandwich_norms for its second pair of them, attention_scale for its
    query_pre_attn_scalar, embedding_scale and final_logit_softcap for the
    rest of Gemma.

    `layer_types` is the pattern, one kind per layer, and `kinds` says what a
    kind does: its window, and its own rope base or head dim where it has
    one. How a checkpoint's config derives the pattern (Qwen3 makes every
    layer past max_window_layers sliding) belongs to that translation, not
    here: this takes the tuple.

    `mixture` turns the feed-forward of some layers into `moe.SparseMLP`,
    routing each token to a few of its experts; a dense layer keeps the
    leaves it always had, and None is a dense model. The LM objective's
    balance_rate is what moves a mixture's balancing bias.

    causal=False turns every layer into full attention with no cache, the
    encoder a masked diffusion language model denoises with; the parameter
    tree is the same either way.

    per_layer_input_dim turns on Gemma 3n/4 style per-layer input embeddings:
    an extra table of per_layer_input_vocab by layers times dim rows, read
    per layer and added to that layer's input through its own gate. None is a
    plain decoder and leaves the tree unchanged.

    num_kv_shared_layers makes the trailing layers of that count reuse the
    keys and values of the last earlier layer of their own kind instead of
    projecting their own (Gemma 3n/4 cross-layer KV sharing). 0 is a plain
    decoder and leaves the tree unchanged, and use_double_wide_mlp, which
    widens the sharing layers' MLP, needs it.

    partial_rotary_factor rotates that fraction of an unwindowed kind's head
    dims and passes the rest through, which is Gemma 4's global layers; a
    windowed kind rotates whole.
    """
    vocab_size: int
    emb_features: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: Optional[int] = None       # None: as many as the query heads
    head_dim: Optional[int] = None           # None: emb_features // num_heads
    mlp: str = 'swiglu'                      # 'swiglu' | 'geglu'
    mlp_features: Optional[int] = None       # None: four times emb_features
    max_seq_len: int = 2048
    rope_theta: float = 10000.0              # the base a kind does not override
    partial_rotary_factor: Optional[float] = None  # None: every dim rotates
    layer_types: Optional[Tuple[str, ...]] = None  # the pattern, one kind per layer
    kinds: Optional[Mapping[str, LayerKind]] = None  # what each named kind does
    norm_eps: float = 1e-5
    scale_offset: bool = False               # Gemma's (1 + w) RMSNorm scale
    scale_after_cast: bool = False           # Llama and Qwen3 scale the cast activations
    sandwich_norms: bool = False             # Gemma's norms on the sublayer outputs
    qk_norm: bool = True
    v_norm: bool = False                     # Gemma 4's scale-free values norm
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
    mixture: Optional[Mixture] = None        # None: every layer is dense
    use_double_wide_mlp: bool = False        # Gemma 4 doubles sharing layers' MLP width
    causal: bool = True                      # False: full attention, no cache
    per_layer_input_dim: Optional[int] = None  # Gemma 3n/4 per-layer inputs
    per_layer_input_vocab: Optional[int] = None  # None: vocab_size
    num_kv_shared_layers: int = 0            # trailing layers reusing a provider's K/V; 0 disables

    def __post_init__(self):
        if self.layer_types is not None:
            object.__setattr__(self, "layer_types", tuple(self.layer_types))
        # A value arrives as a record from a config and as itself from code,
        # and `models.build` already reads one; doing it here too means the
        # plain constructor takes the same records, which is what a test or
        # a notebook writes.
        if isinstance(self.mixture, Mapping):
            object.__setattr__(self, "mixture", Mixture(**self.mixture))
        if self.kinds is not None:
            # Frozen, because a module's fields are static to jit and a plain
            # dict cannot be hashed.
            object.__setattr__(self, "kinds", flax.core.freeze({
                name: kind if isinstance(kind, LayerKind) else LayerKind(**kind)
                for name, kind in self.kinds.items()}))
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
        return (4 * self.emb_features
                if self.mlp_features is None else self.mlp_features)

    @property
    def per_layer_types(self) -> Tuple[str, ...]:
        if self.layer_types is None:
            return ('full_attention',) * self.num_layers
        return tuple(self.layer_types)

    def kind_of(self, layer_type: str) -> "ResolvedKind":
        """What the layers of `layer_type` do, the model's defaults included."""
        kind = (self.kinds or {}).get(layer_type, LayerKind())
        return ResolvedKind(
            window=kind.window,
            rope_theta=self.rope_theta if kind.rope_theta is None else kind.rope_theta,
            head_dim=(self.features_per_head if kind.head_dim is None else kind.head_dim))

    @property
    def sparse_layers(self) -> Tuple[int, ...]:
        """The layers whose feed-forward routes to experts."""
        mixture = self.mixture
        if mixture is None:
            return ()
        if mixture.layers is not None:
            return tuple(mixture.layers)
        if mixture.every is not None:
            return tuple(index for index in range(self.num_layers)
                         if (index + 1) % mixture.every == 0)
        return tuple(range(self.num_layers))

    @property
    def kv_sharing(self) -> dict:
        """Sharing layer index to the provider it reads, both of one layer type.

        The trailing num_kv_shared_layers layers own no K/V and read the last
        non-sharing layer of their own type (modeling_gemma4.py,
        Gemma4TextAttention). Empty unless sharing is on.
        """
        if not self.num_kv_shared_layers:
            return {}
        first = self.num_layers - self.num_kv_shared_layers
        if first <= 0:
            raise ValueError(
                f"num_kv_shared_layers ({self.num_kv_shared_layers}) has to leave "
                f"a provider: it must be between 1 and num_layers - 1 "
                f"({self.num_layers - 1})")
        types = self.per_layer_types
        providers = {}
        for index in range(first, self.num_layers):
            earlier = [j for j in range(first) if types[j] == types[index]]
            if not earlier:
                raise ValueError(
                    f"layer {index} shares K/V but no earlier {types[index]} layer "
                    "exists to provide them")
            providers[index] = earlier[-1]
        return providers

    @property
    def per_layer_vocab(self) -> int:
        return self.vocab_size if self.per_layer_input_vocab is None else self.per_layer_input_vocab

    def setup(self):
        types = self.per_layer_types
        if len(types) != self.num_layers:
            raise ValueError(
                f"layer_types has {len(types)} entries for {self.num_layers} layers")
        unnamed = sorted(set(self.kinds or {}) - set(types))
        if unnamed:
            raise ValueError(
                f"kinds {unnamed} name no layer of this model, whose pattern is "
                f"{sorted(set(types))}")
        kinds = {layer_type: self.kind_of(layer_type) for layer_type in set(types)}
        for layer_type, kind in sorted(kinds.items()):
            if kind.head_dim % 2:
                raise ValueError(
                    "rotary positions rotate pairs, so the head dim of "
                    f"{layer_type!r} must be even, got {kind.head_dim}")
            if kind.window is not None and kind.window < 1:
                raise ValueError(
                    f"the window of {layer_type!r} must be positive, got {kind.window}")
        if self.num_heads % self.kv_heads:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be a multiple of num_kv_heads "
                f"({self.kv_heads})")
        if self.partial_rotary_factor is not None and not 0 < self.partial_rotary_factor <= 1:
            raise ValueError(
                "partial_rotary_factor must be within (0, 1], "
                f"got {self.partial_rotary_factor}")
        sparse = self.sparse_layers
        outside = sorted(index for index in sparse
                         if not 0 <= index < self.num_layers)
        if outside:
            raise ValueError(
                f"the mixture's layers {outside} are outside the "
                f"{self.num_layers} layers of this model")
        sharing = self.kv_sharing
        if self.use_double_wide_mlp and not sharing:
            raise ValueError(
                "use_double_wide_mlp widens the MLP of the layers that share "
                "their keys and values, so it needs num_kv_shared_layers set")
        ple = self.per_layer_input_dim
        if ple is not None and ple < 1:
            raise ValueError(
                f"per_layer_input_dim is the width of a layer's own input, got "
                f"{ple}; None is a model without them")

        self.embed_tokens = nn.Embed(
            num_embeddings=self.vocab_size, features=self.emb_features,
            dtype=self.dtype, name='embed_tokens')
        if ple:
            # The packed table every layer reads its own slice of
            # (modeling_gemma4.py, Gemma4TextModel): one row per token, a
            # hidden_size_per_layer_input slice per layer.
            self.embed_tokens_per_layer = nn.Embed(
                num_embeddings=self.per_layer_vocab, features=self.num_layers * ple,
                dtype=self.dtype, name='embed_tokens_per_layer')
            self.per_layer_model_projection = nn.Dense(
                self.num_layers * ple, use_bias=False,
                dtype=self.dtype, precision=self.precision,
                name='per_layer_model_projection')
            self.per_layer_projection_norm = RMSNorm(
                epsilon=self.norm_eps, scale_offset=self.scale_offset,
                scale_after_cast=self.scale_after_cast, dtype=self.dtype,
                name='per_layer_projection_norm')
        mixture = self.mixture
        routed = None if mixture is None else functools.partial(
            SparseMLP,
            num_experts=mixture.experts,
            top_k=mixture.top_k,
            hidden_features=self.hidden_features,
            out_features=self.emb_features,
            activation=self.mlp,
            score_function=mixture.score_function,
            routed_scaling_factor=mixture.scaling,
            expert_groups=mixture.groups,
            groups_per_token=mixture.groups_per_token,
            expert_bias=mixture.bias,
            dtype=self.dtype,
            precision=self.precision)
        self.layers = [
            DecoderBlock(
                mixer=functools.partial(
                    CausalSelfAttention,
                    emb_features=self.emb_features,
                    num_heads=self.num_heads,
                    num_kv_heads=self.kv_heads,
                    head_dim=kinds[layer_type].head_dim,
                    max_seq_len=self.max_seq_len,
                    causal=self.causal,
                    rope_theta=kinds[layer_type].rope_theta,
                    # A windowed kind rotates every dimension; the partial
                    # rotary belongs to the kinds that attend the whole
                    # sequence, which is where Gemma 4 puts it.
                    partial_rotary_factor=(None if kinds[layer_type].window is not None
                                           else self.partial_rotary_factor),
                    qk_norm=self.qk_norm,
                    v_norm=self.v_norm,
                    norm_eps=self.norm_eps,
                    scale_offset=self.scale_offset,
                    scale_after_cast=self.scale_after_cast,
                    kv_shared=index in sharing,
                    kv_store_key=layer_type,
                    sliding_window=kinds[layer_type].window,
                    attention_bias=self.attention_bias,
                    attention_scale=self.attention_scale,
                    dtype=self.dtype,
                    precision=self.precision,
                    attention_impl=self.attention_impl,
                    force_fp32_for_softmax=self.force_fp32_for_softmax),
                feedforward=(
                    routed
                    if index in sparse and routed is not None else
                    functools.partial(
                        GatedMLP,
                        hidden_features=(2 * self.hidden_features
                                         if self.use_double_wide_mlp and index in sharing
                                         else self.hidden_features),
                        out_features=self.emb_features,
                        activation=self.mlp,
                        precision=self.precision)),
                emb_features=self.emb_features,
                norm_eps=self.norm_eps,
                scale_offset=self.scale_offset,
                scale_after_cast=self.scale_after_cast,
                sandwich_norms=self.sandwich_norms,
                per_layer_input_dim=ple or 0,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                precision=self.precision,
                name=f'layers_{index}')
            for index, layer_type in enumerate(types)]
        self.norm = RMSNorm(
            epsilon=self.norm_eps, scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast, dtype=self.dtype, name='norm')
        if not self.tie_embeddings:
            self.lm_head = nn.Dense(
                features=self.vocab_size, use_bias=False, dtype=jnp.float32,
                precision=self.precision, name='lm_head')

    def __call__(self, tokens, train: bool = False, decode: bool = False,
                 positions=None, segment_ids=None):
        x = self.hidden_states(tokens, train=train, decode=decode,
                               positions=positions, segment_ids=segment_ids)

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

    def hidden_states(self, tokens, train: bool = False, decode: bool = False,
                      positions=None, segment_ids=None):
        """The final normalised states, `[B, S, D]`: everything the forward
        pass does before the head projection.

        A loss that pairs this with `head_weight` scores tokens without ever
        holding the full `[B, S, vocab]` logits tensor. A packed batch passes
        its per-document `positions` and `segment_ids` through to the layers,
        which is where RoPE and the mask read them.
        """
        x = self.embed_tokens(tokens)
        if self.embedding_scale:
            # Gemma casts embed_scale to the embedding weight dtype
            # (modeling_gemma3.py:117). Dew's nn.Embed holds that table in
            # fp32 and returns the compute dtype, so the factor keeps its
            # fp32 value and only the product rounds with the activations.
            # A factor rounded to bf16 would be 34.0 at hidden 1152, where
            # sqrt(1152) is 33.94112549695428.
            scaled = x * jnp.asarray(math.sqrt(self.emb_features),
                                     self.embed_tokens.embedding.dtype)
            x = scaled.astype(x.dtype)
        ple = self.per_layer_inputs(tokens, x) if self.per_layer_input_dim else None
        kv_store = {} if self.num_kv_shared_layers else None
        for index, layer in enumerate(self.layers):
            x = layer(x, train=train, decode=decode,
                      positions=positions, segment_ids=segment_ids,
                      kv_store=kv_store,
                      per_layer_input=None if ple is None else ple[:, :, index, :])
        return self.norm(x)

    def per_layer_inputs(self, tokens, inputs_embeds):
        """Every layer's input signal `[B, S, L, P]` (Gemma 3n/4 PLE).

        The token-identity component is the packed table's row for each
        token, scaled like the main embedding; the context component is the
        input embeddings projected down, scaled and normed. Their sum over
        sqrt(2) is what each layer's gate multiplies in
        (modeling_gemma4.py, get_per_layer_inputs/project_per_layer_inputs).
        """
        ple = self.per_layer_input_dim
        if ple is None:
            raise ValueError(
                "per_layer_inputs is the signal of a model with per-layer input "
                "embeddings, and this one has none: per_layer_input_dim is unset")
        table = self.embed_tokens_per_layer(tokens).reshape(
            *tokens.shape, self.num_layers, ple)
        # The reference scales by sqrt(P) cast to the table's weight dtype
        # (modeling_gemma4.py, Gemma4TextScaledWordEmbedding), not to the
        # activation dtype.
        table = table * jnp.asarray(
            math.sqrt(ple), self.embed_tokens_per_layer.embedding.dtype)
        context = self.per_layer_model_projection(inputs_embeds)
        context = context * jnp.asarray(self.emb_features ** -0.5, context.dtype)
        context = self.per_layer_projection_norm(
            context.reshape(*inputs_embeds.shape[:-1], self.num_layers, ple))
        return (context + table) * jnp.asarray(2.0 ** -0.5, context.dtype)

    def head_weight(self, params):
        """The `[D, vocab]` head matrix in fp32, as the forward multiplies it.

        `params` is the parameter tree the forward runs under, so this is a
        plain read: a tied head is the embedding table transposed, an untied
        one is `lm_head`'s kernel, which is `[D, vocab]` already. The Gemma
        embedding scale multiplies the input embeddings only, so it has no
        place here.
        """
        if self.tie_embeddings:
            return params['embed_tokens']['embedding'].astype(jnp.float32).T
        return params['lm_head']['kernel'].astype(jnp.float32)

    def init_cache(self, batch_size: int):
        """Allocate a zeroed decode cache for `batch_size` sequences.

        cache = model.apply(params, batch_size, method=CausalTransformer.init_cache,
                            mutable=['cache'])[1]['cache']

        The forward pass this runs is a single dummy token whose keys are never
        written: allocation happens on the first decode-mode call, the write on
        the ones after it.
        """
        self(jnp.zeros((batch_size, 1), jnp.int32), decode=True)
