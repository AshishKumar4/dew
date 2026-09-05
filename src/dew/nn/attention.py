"""Attention: the one kernel path, the KV cache, and the blocks the UNets
use, the latter ported from diffusers' attention_flax.py."""

import dataclasses

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional
from flax.typing import Dtype, PrecisionLike
import functools
import math
from .sharding import logical_axes

def repeat_kv_heads(x, num_heads: int):
    """Repeat grouped key/value heads out to the query heads: [B, S, K, D] -> [B, S, N, D].

    Query head n reads key/value head n // (N // K), the grouping
    jax.nn.dot_product_attention uses internally, so the same param tree runs
    on the kernels that group heads themselves and on the ones that need the
    keys materialized.
    """
    kv_heads = x.shape[-2]
    if kv_heads == num_heads:
        return x
    if num_heads % kv_heads:
        raise ValueError(
            f"grouped-query attention needs the query heads ({num_heads}) to be "
            f"a multiple of the key/value heads ({kv_heads}).")
    return jnp.repeat(x, num_heads // kv_heads, axis=-2)


def causal_attention_mask(query_positions, kv_len: int, sliding_window=None):
    """Boolean [1, 1, T, S] mask keeping keys at or before each query's position.

    query_positions are absolute: jnp.arange(T) for a plain forward pass, and
    the cache slots the queries occupy when decoding against a KV cache, where
    a query's row index is no longer its position. sliding_window=w narrows the
    mask to the w most recent keys (the query itself plus the w-1 before it),
    which is what a sliding attention layer means.
    """
    q_pos = jnp.asarray(query_positions)[:, None]
    k_pos = jnp.arange(kv_len)[None, :]
    mask = k_pos <= q_pos
    if sliding_window is not None:
        mask = jnp.logical_and(mask, k_pos > q_pos - sliding_window)
    return mask[None, None]


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
def rotary_freqs(positions, head_dim: int, theta: float, rot_dim: int | None = None,
                 partial_rotary_type: str = 'proportional'):
    """cos/sin of the rotary angles at absolute `positions`: [P, pairs].

    `positions` may be [P] (one sequence) or [B, P] (a packed batch whose
    documents each restart at 0); the angle axes line up with the trailing
    [B, S] either way. Computed in fp32 so a token gets the same rotation
    whether it arrives in a prefill or comes back as a single decode step.

    rot_dim narrows the rotation to the first rot_dim dimensions, and
    `partial_rotary_type` names which of the two published conventions
    that is, because they rotate different angles:

    - 'proportional' (Gemma 4, modeling_rope_utils.py
      `_compute_proportional_rope_parameters`): the exponents run over the
      full head_dim, `theta ** (2i / head_dim)` for the rot_dim // 2 rotated
      pairs, and the rest keep frequency zero (cosine one, sine zero, so the
      rotation is the identity there). The output is head_dim // 2 wide.
    - 'default' (Qwen3.5, modeling_qwen3_5.py:117-124
      `Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters`): the
      rope is a rot_dim-dimensional one, `theta ** (2i / rot_dim)`, and the
      output is rot_dim // 2 wide; `apply_rotary` passes the trailing
      dimensions through untouched, the reference's `q_rot, q_pass` split
      (modeling_qwen3_5.py:581-591).

    With rot_dim None both are the full rotation and the type is moot.
    """
    if partial_rotary_type not in ('proportional', 'default'):
        raise ValueError(
            "partial_rotary_type names the convention of a partial rotary, "
            f"'proportional' or 'default', got {partial_rotary_type!r}")
    pairs = head_dim // 2 if rot_dim is None else rot_dim // 2
    divisor = head_dim if rot_dim is None or partial_rotary_type == 'proportional' else rot_dim
    inv_freq = 1.0 / (theta ** (jnp.arange(0, 2 * pairs, 2, dtype=jnp.float32) / divisor))
    if rot_dim is not None and partial_rotary_type == 'proportional':
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

    The freqs are [S, pairs] for one sequence, or [B, S, pairs] when a packed
    batch restarts positions per document. Freqs narrower than D // 2 rotate
    the first 2 * pairs dimensions and pass the rest through, which is the
    sliced partial rotary of `rotary_freqs(partial_rotary_type='default')`.
    `scale` multiplies the whole head inside the fp32 arithmetic, so a
    query's attention scale narrows once, with the product.
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
    rotated_dims = cos.shape[-1]
    fp32, passed = fp32[..., :rotated_dims], fp32[..., rotated_dims:]
    x1, x2 = jnp.split(fp32, 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    out = fp32 * cos + rotated * sin
    if passed.shape[-1]:
        out = jnp.concatenate([out, passed], axis=-1)
    return (out if scale is None else out * scale).astype(x.dtype)


def open_kv_cache(module: nn.Module, key, max_seq_len):
    """Fixed-size KV cache in the flax MultiHeadDotProductAttention style.

    Declares cached_key/cached_value/cache_index in the 'cache' collection,
    sized from `key` at the full decode length, and returns the absolute
    positions of the tokens this step appends together with the writer that
    appends them. Positions come out before the write because rotary positions
    have to rotate the keys going into the cache, not the ones already in it.

    The writer advances the index by the number of tokens written, so one code
    path covers a whole-prompt prefill and the single-token steps after it. The
    first call only allocates: a freshly initialised model hands back a zeroed
    cache at index 0 rather than one with a dummy token in slot 0.
    """
    if max_seq_len is None:
        raise ValueError(
            "decoding needs max_seq_len: the KV cache is allocated once, at the "
            "full decode length, and never grows.")
    batch, length, heads, head_dim = key.shape
    if length > max_seq_len:
        raise ValueError(
            f"{length} tokens do not fit a KV cache of {max_seq_len}.")
    allocated = module.has_variable('cache', 'cached_key')
    cached_key = module.variable('cache', 'cached_key', jnp.zeros,
                                 (batch, max_seq_len, heads, head_dim), key.dtype)
    cached_value = module.variable('cache', 'cached_value', jnp.zeros,
                                   (batch, max_seq_len, heads, head_dim), key.dtype)
    cache_index = module.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, jnp.int32))
    index = cache_index.value

    def append(key, value):
        if not allocated:
            return key, value
        zero = jnp.array(0, index.dtype)
        cached_key.value = jax.lax.dynamic_update_slice(
            cached_key.value, key.astype(cached_key.value.dtype),
            (zero, index, zero, zero))
        cached_value.value = jax.lax.dynamic_update_slice(
            cached_value.value, value.astype(cached_value.value.dtype),
            (zero, index, zero, zero))
        cache_index.value = index + length
        return cached_key.value, cached_value.value

    return index + jnp.arange(length), append


def _pad_rows(x, rows: int):
    """`x` with `rows` zero rows appended on its sequence axis, [B, S, H, D]."""
    return x if rows == 0 else jnp.pad(x, ((0, 0), (0, rows), (0, 0), (0, 0)))


def cudnn_attention(query, key, value, bias, mask, causal, sliding_window):
    """jax's cudnn flash attention over any sequence length.

    cudnn's kernel has no backward pass for an odd query or key length (jax
    raises NotImplementedError from inside the gradient, so a run found out at
    its first training step), and 77 CLIP text tokens are odd, as is every
    concatenated text-plus-image sequence. One zero row of padding makes the
    length even: a padded query row's output is sliced off, so nothing reads
    it, and a padded key is hidden by the kernel's own padding mask
    (key_value_seq_lengths), so every real query attends to exactly the keys
    it had. The arithmetic on the real rows is the fused kernel's, in fp32
    like the xla path's, which is what tests/test_kernels.py pins.
    """
    q_len, kv_len = query.shape[-3], key.shape[-3]
    q_pad, kv_pad = q_len % 2, kv_len % 2
    if q_pad or kv_pad:
        query = _pad_rows(query, q_pad)
        key, value = _pad_rows(key, kv_pad), _pad_rows(value, kv_pad)
        # Masks and biases broadcast over [B, H, Q, K]; only the last two
        # dimensions grow, and a padded query row or key column sees nothing.
        def pad_tail(x, fill):
            return jnp.pad(x, ((0, 0),) * (x.ndim - 2) + ((0, q_pad), (0, kv_pad)),
                           constant_values=fill)
        mask = None if mask is None else pad_tail(mask, False)
        bias = None if bias is None else pad_tail(bias, 0)
    kv_lengths = None if kv_pad == 0 else jnp.full(key.shape[:1], kv_len, jnp.int32)
    # A left window of l means the l+1 most recent keys on both the xla and
    # the cudnn path, which is the window this function counts.
    out = jax.nn.dot_product_attention(
        query, key, value, bias=bias, mask=mask, is_causal=causal,
        key_value_seq_lengths=kv_lengths,
        local_window_size=None if sliding_window is None else (sliding_window - 1, 0),
        implementation='cudnn')
    return out[:, :q_len] if q_pad else out


CUDNN_DTYPES = (jnp.bfloat16, jnp.float16)
CUDNN_MAX_HEAD_DIM = 128


def cudnn_runs(query) -> bool:
    """Whether cudnn's fused kernel takes this query: a gpu backend, one of its
    two dtypes, and a head dimension it tiles. 'auto' asks this; an explicit
    'cudnn' refuses by name instead."""
    head_dim = query.shape[-1]
    return (jax.default_backend() == 'gpu' and query.dtype in CUDNN_DTYPES
            and head_dim % 8 == 0 and head_dim <= CUDNN_MAX_HEAD_DIM)


def scaled_dot_product_attention(query, key, value, dtype=None, precision=None,
                                 force_fp32_for_softmax=True, implementation=None,
                                 causal=False, sliding_window=None, mask=None, bias=None):
    """The one attention kernel path for every attention module.

    Inputs are [B, S, H, D]. Keys and values may carry fewer heads than the
    query (grouped-query attention); the paths that cannot group heads
    themselves get them repeated out. The param trees of the callers never
    change with the implementation, so checkpoints are interchangeable across
    hardware:

    - None: flax reference attention (einsum + softmax). The only path that
      reads dtype, precision and force_fp32_for_softmax; the portable default.
    - 'auto': 'cudnn' where its kernel runs (a gpu backend, bf16 or fp16
      inputs, a head dimension that is a multiple of 8 and at most 128), 'xla'
      anywhere else. Resolved per trace, so a config logged as 'auto' still
      runs on the next machine.
    - 'xla' / 'cudnn': jax.nn.dot_product_attention, which dispatches to the
      fused cudnn flash kernel on supported GPUs. It takes no dtype, precision
      or softmax argument: the logits accumulate and the softmax runs in fp32
      whatever the inputs are. A dtype that asks for anything other than the
      inputs' own dtype is rejected rather than silently dropped, as are the
      other two. cudnn takes any sequence length (`cudnn_attention` pads an
      odd one), and only bf16 or fp16 inputs.
    - 'tpu': the pallas TPU flash kernel, with the 1/sqrt(d) scale passed
      explicitly (the deleted EfficientAttention passed none, which inflated
      the logits by sqrt(d) and made its checkpoints poisonous).

    causal restricts query i to keys 0..i, top-left aligned exactly as jax's
    is_causal is; sliding_window=w narrows that to the w most recent keys. Both
    are structural, over the row index, so decoding against a KV cache passes
    `mask` instead (built by causal_attention_mask over the cache slots): a
    step's single query sits at the cache index, not at row 0. The fused
    kernels take causality and the window as flags rather than a materialized
    mask, which is where their memory win comes from; the TPU kernel has no
    mask argument, so an explicit mask rides in there as an additive bias.
    `bias` is an additive float array broadcastable to [B, H, Q, K], added to
    the logits on every path; T5's relative position table travels in it.
    """
    if sliding_window is not None and sliding_window < 1:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")

    if implementation is None:
        heads = query.shape[-2]
        key = repeat_kv_heads(key, heads)
        value = repeat_kv_heads(value, heads)
        if causal or sliding_window is not None:
            structural = causal_attention_mask(
                jnp.arange(query.shape[-3]), key.shape[-3], sliding_window)
            mask = structural if mask is None else jnp.logical_and(mask, structural)
        return nn.dot_product_attention(
            query, key, value, bias=bias, mask=mask, dtype=dtype, broadcast_dropout=False,
            dropout_rng=None, precision=precision,
            force_fp32_for_softmax=force_fp32_for_softmax, deterministic=True)

    if implementation == 'auto':
        implementation = 'cudnn' if cudnn_runs(query) else 'xla'

    requested = {str(getattr(p, 'name', p)).upper()
                 for p in (precision if isinstance(precision, (tuple, list)) else (precision,))
                 if p is not None}
    if requested & {'HIGH', 'HIGHEST'}:
        raise ValueError(
            f"attention implementation '{implementation}' cannot honor "
            f"precision={precision}: fused attention accumulates the logits and "
            "runs the softmax in fp32 regardless. Leave precision at DEFAULT, or "
            "use the reference implementation (attention_impl 'reference').")
    if not force_fp32_for_softmax:
        raise ValueError(
            f"attention implementation '{implementation}' cannot honor "
            "force_fp32_for_softmax=False: fused attention runs the softmax in "
            "fp32 regardless. Leave it True, or use the reference implementation "
            "(attention_impl 'reference').")
    if dtype is not None and jnp.dtype(dtype) != query.dtype:
        raise ValueError(
            f"attention implementation '{implementation}' cannot honor "
            f"dtype={dtype}: fused attention computes in the inputs' dtype "
            f"({query.dtype}). Pass dtype=None or leave the inputs in that dtype, or use "
            "the reference implementation (attention_impl 'reference').")

    if implementation == 'cudnn':
        if query.dtype not in CUDNN_DTYPES:
            raise ValueError(
                "cudnn attention needs bf16 or fp16 inputs, the query is "
                f"{query.dtype}. Set dtype bfloat16, or attention_impl 'xla' to "
                "keep this precision.")
        head_dim = query.shape[-1]
        if head_dim % 8 or head_dim > CUDNN_MAX_HEAD_DIM:
            raise ValueError(
                f"cudnn attention needs a head dimension that is a multiple of 8 "
                f"and at most {CUDNN_MAX_HEAD_DIM}, got {head_dim}; use attention_impl "
                "'xla' for this shape.")
        return cudnn_attention(query, key, value, bias, mask, causal, sliding_window)
    if implementation == 'xla':
        # A left window of l means the l+1 most recent keys on both the xla and
        # the cudnn path, which is the window this function counts.
        return jax.nn.dot_product_attention(
            query, key, value, bias=bias, mask=mask, is_causal=causal,
            local_window_size=None if sliding_window is None else (sliding_window - 1, 0),
            implementation='xla')
    if implementation == 'tpu':
        from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
        heads = query.shape[-2]
        key = repeat_kv_heads(key, heads)
        value = repeat_kv_heads(value, heads)
        # pallas wants [B, H, S, D]
        q = jnp.moveaxis(query, -2, -3)
        k = jnp.moveaxis(key, -2, -3)
        v = jnp.moveaxis(value, -2, -3)
        combined = None
        if bias is not None:
            combined = jnp.broadcast_to(bias.astype(q.dtype),
                                        (q.shape[0], q.shape[1], q.shape[2], k.shape[2]))
        if sliding_window is not None:
            band = causal_attention_mask(
                jnp.arange(query.shape[-3]), key.shape[-3], sliding_window)
            mask = band if mask is None else jnp.logical_and(mask, band)
        if mask is not None:
            seated = jnp.broadcast_to(
                jnp.where(mask, 0, jnp.finfo(q.dtype).min).astype(q.dtype),
                (q.shape[0], q.shape[1], q.shape[2], k.shape[2]))
            combined = seated if combined is None else combined + seated
        out = flash_attention(q, k, v, ab=combined, causal=causal,
                              sm_scale=1.0 / math.sqrt(query.shape[-1]))
        return jnp.moveaxis(out, -3, -2)
    raise ValueError(f"Unknown attention implementation: {implementation}")


@logical_axes({
    ("to_q",): ("embed", "heads", "head_dim"),
    ("to_k",): ("embed", "heads", "head_dim"),
    ("to_v",): ("embed", "heads", "head_dim"),
    ("to_out_0",): ("heads", "head_dim", "embed"),
})
class NormalAttention(nn.Module):
    """Multi-head attention over a `[B, S, C]` or `[B, H, W, C]` input.

    causal makes it a decoder attention (query i sees keys 0..i). decode=True
    on a call runs it against a fixed-size KV cache instead, allocated at
    max_seq_len: the first call writes the whole prompt, later calls append one
    token each. Neither flag touches the param tree, so a model trained without
    either reloads into a decoding one unchanged. `freqs_cis` rotates the
    queries and keys of a self-attention call (`rotary_freqs` gives the pair);
    None leaves them unrotated.
    """
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    use_bias: bool = True
    force_fp32_for_softmax: bool = True
    qk_norm: bool = False  # RMSNorm on q/k per head (SD3-style bf16 logit safety)
    attention_impl: Optional[str] = None  # None (reference) | 'auto' | 'xla' | 'cudnn' | 'tpu'
    causal: bool = False
    max_seq_len: Optional[int] = None  # KV cache length, required to decode

    def setup(self):
        dense = functools.partial(
            nn.DenseGeneral,
            features=[self.heads, self.dim_head],
            axis=-1,
            precision=self.precision,
            use_bias=self.use_bias,
            dtype=self.dtype
        )
        self.query = dense(name="to_q")
        self.key = dense(name="to_k")
        self.value = dense(name="to_v")

        if self.qk_norm:
            self.q_norm = nn.RMSNorm(dtype=self.dtype, name="q_norm")
            self.k_norm = nn.RMSNorm(dtype=self.dtype, name="k_norm")

        self.proj_attn = nn.DenseGeneral(
            self.query_dim,
            axis=(-2, -1),
            precision=self.precision,
            use_bias=self.use_bias,
            dtype=self.dtype,
            name="to_out_0",
        )

    @nn.compact
    def __call__(self, x, context=None, decode: bool = False, freqs_cis=None):
        orig_x_shape = x.shape
        if len(x.shape) == 4:
            x = x.reshape((x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        context = x if context is None else context
        if len(context.shape) == 4:
            context = context.reshape(
                (context.shape[0], context.shape[1] * context.shape[2], context.shape[3]))
        query = self.query(x)
        key = self.key(context)
        value = self.value(context)
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        if freqs_cis is not None:
            freqs_cos, freqs_sin = freqs_cis
            query = apply_rotary(query, freqs_cos, freqs_sin)
            key = apply_rotary(key, freqs_cos, freqs_sin)

        causal, mask = self.causal, None
        if decode:
            # Position lives in the cache slot now, not in the row index, so
            # causality travels as a mask rather than the kernels' flag.
            positions, append = open_kv_cache(self, key, self.max_seq_len)
            key, value = append(key, value)
            mask = causal_attention_mask(positions, key.shape[-3])
            causal = False

        hidden_states = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            implementation=self.attention_impl, causal=causal, mask=mask,
        )
        proj = self.proj_attn(hidden_states)
        proj = proj.reshape(orig_x_shape)
        return proj

class FlaxGEGLU(nn.Module):
    """A linear layer into the gated linear unit of Shazeer 2020
    (https://arxiv.org/abs/2002.05202): half the projection gates the other
    half through GELU. The hidden width is four times `dim`."""

    dim: int
    dtype: Optional[Dtype] = jnp.float32
    precision: PrecisionLike = jax.lax.Precision.DEFAULT

    def setup(self):
        inner_dim = self.dim * 4
        self.proj = nn.Dense(inner_dim * 2, dtype=self.dtype, precision=self.precision)

    def __call__(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=-1)
        return hidden_linear * nn.gelu(hidden_gelu)


@logical_axes({("net_0", "proj"): ("embed", "mlp"), ("net_2",): ("mlp", "embed")})
class FlaxFeedForward(nn.Module):
    """GEGLU then a linear layer back to `dim`, diffusers' `FlaxFeedForward`.
    The names `net_0` and `net_2` are the indices the reference's Sequential
    gives the two layers, which is what its checkpoints carry."""

    dim: int
    dtype: Optional[Dtype] = jnp.float32
    precision: PrecisionLike = jax.lax.Precision.DEFAULT

    def setup(self):
        self.net_0 = FlaxGEGLU(self.dim, dtype=self.dtype, precision=self.precision)
        self.net_2 = nn.Dense(self.dim, dtype=self.dtype, precision=self.precision)

    def __call__(self, hidden_states):
        hidden_states = self.net_0(hidden_states)
        hidden_states = self.net_2(hidden_states)
        return hidden_states


class BasicTransformerBlock(nn.Module):
    """Self-attention, cross-attention over `context`, feed-forward, each
    pre-normed with a residual. `use_cross_only` drops the self-attention;
    `only_pure_attention` runs the cross-attention alone with no norm and no
    residual, which is how the UNets' stages attend by default."""
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    use_bias: bool = True
    use_cross_only:bool = False
    only_pure_attention:bool = False
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-4
    attention_impl: Optional[str] = None

    def setup(self):
        attention = functools.partial(
            NormalAttention,
            query_dim=self.query_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            precision=self.precision,
            use_bias=self.use_bias,
            dtype=self.dtype,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            attention_impl=self.attention_impl,
        )
        self.attention1 = attention(name='Attention1')
        self.attention2 = attention(name='Attention2')

        self.ff = FlaxFeedForward(dim=self.query_dim, dtype=self.dtype, precision=self.precision)
        self.norm1 = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)
        self.norm2 = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)
        self.norm3 = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)

    @nn.compact
    def __call__(self, hidden_states, context=None):
        if self.only_pure_attention:
            return self.attention2(hidden_states, context)

        if not self.use_cross_only:
            hidden_states = hidden_states + self.attention1(self.norm1(hidden_states))
        hidden_states = hidden_states + self.attention2(self.norm2(hidden_states), context)
        hidden_states = hidden_states + self.ff(self.norm3(hidden_states))
        return hidden_states


@dataclasses.dataclass(frozen=True)
class Stage:
    """One resolution stage's attention in a UNet, or `None` for a stage that
    has none.

    Every field is a `TransformerBlock` dial, so a stage names what it changes
    and nothing else. `dim_head` is not here: the block's head width is the
    stage's channel count divided by `heads`, which the unet knows and a
    config does not. `dew.registry.from_record` builds one from a record at
    the build boundary, so a stage still arrives as `{"heads": 8}` from a
    command line or a run record, and a misspelled field raises there.

    `dtype` is float32 and not the model's, which is what `with_precision`
    exists to write into every stage. `precision` is the one field whose None
    means "the model's".
    """

    heads: int
    use_linear_attention: bool = True
    use_projection: bool = False
    use_self_and_cross: bool = True
    only_pure_attention: bool = True
    force_fp32_for_softmax: bool = False
    norm_inputs: bool = True
    explicitly_add_residual: bool = True
    norm_epsilon: float = 1e-4
    dtype: Optional[Dtype] = jnp.float32
    precision: PrecisionLike = None


def stage_attention(stage: Stage, channels: int, attention_impl: Optional[str],
                    precision: PrecisionLike, name: str) -> "TransformerBlock":
    """The block a UNet stage's `Stage` describes, at the stage's channel
    count; a stage that names no precision takes the model's."""
    return TransformerBlock(
        heads=stage.heads, dim_head=channels // stage.heads, dtype=stage.dtype,
        attention_impl=attention_impl, use_projection=stage.use_projection,
        use_self_and_cross=stage.use_self_and_cross,
        precision=stage.precision or precision,
        only_pure_attention=stage.only_pure_attention,
        force_fp32_for_softmax=stage.force_fp32_for_softmax,
        norm_inputs=stage.norm_inputs, explicitly_add_residual=stage.explicitly_add_residual,
        use_linear_attention=stage.use_linear_attention, norm_epsilon=stage.norm_epsilon,
        name=name)


class TransformerBlock(nn.Module):
    """A `BasicTransformerBlock` behind an optional projection into and out of
    `heads * dim_head`, dense (`use_linear_attention`) or a 1x1 convolution.
    Without the projection the block runs at the input width."""
    heads: int = 4
    dim_head: int = 32
    use_linear_attention: bool = True
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    use_projection: bool = False
    use_self_and_cross:bool = True
    only_pure_attention:bool = False
    force_fp32_for_softmax: bool = True
    attention_impl: Optional[str] = None
    norm_inputs: bool = True
    explicitly_add_residual: bool = True
    norm_epsilon: float = 1e-4

    @nn.compact
    def __call__(self, x, context=None):
        inner_dim = self.heads * self.dim_head
        C = x.shape[-1]
        if self.norm_inputs:
            x = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)(x)
        if self.use_projection:
            if self.use_linear_attention:
                projected_x = nn.Dense(features=inner_dim,
                                       use_bias=False, precision=self.precision,
                                       dtype=self.dtype, name='project_in')(x)
            else:
                projected_x = nn.Conv(
                    features=inner_dim, kernel_size=(1, 1),
                    strides=(1, 1), padding='VALID', use_bias=False, dtype=self.dtype,
                    precision=self.precision, name='project_in_conv',
                )(x)
        else:
            projected_x = x
            inner_dim = C

        context = projected_x if context is None else context

        projected_x = BasicTransformerBlock(
            query_dim=inner_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            name='Attention',
            precision=self.precision,
            use_bias=False,
            dtype=self.dtype,
            use_cross_only=(not self.use_self_and_cross),
            only_pure_attention=self.only_pure_attention,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            attention_impl=self.attention_impl,
            norm_epsilon=self.norm_epsilon
        )(projected_x, context)

        if self.use_projection:
            if self.use_linear_attention:
                projected_x = nn.Dense(features=C, precision=self.precision,
                                       dtype=self.dtype, use_bias=False,
                                       name='project_out')(projected_x)
            else:
                projected_x = nn.Conv(
                    features=C, kernel_size=(1, 1),
                    strides=(1, 1), padding='VALID', use_bias=False, dtype=self.dtype,
                    precision=self.precision, name='project_out_conv',
                )(projected_x)

        if self.only_pure_attention or self.explicitly_add_residual:
            projected_x = x + projected_x
        return projected_x