"""
Some Code ported from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_flax.py
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Callable, Sequence, Any, Union, Tuple, Optional
from flax.typing import Dtype, PrecisionLike
import einops
import functools
import math
from .blocks import kernel_init

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


def cudnn_supports(query, key) -> Optional[str]:
    """Why the fused cudnn kernel cannot take this call, or None if it can.

    One rule: cudnn's flash kernel has no backward pass for an odd query or
    key length, and jax raises NotImplementedError from inside the gradient.
    The check sees shapes, not the trace, so every call of an odd length is
    refused, forward-only ones included: single-token decode has q_len 1 and
    leaves cudnn for xla, which costs the fused kernel's speed and changes no
    numbers (both run the softmax in fp32). 77 CLIP text tokens are odd,
    which is every cross-attention call in this repo.

    The other cudnn limits (head_dim a multiple of 8 and at most 128 before
    Hopper) fail the forward pass too, so they cannot silently ruin a run and
    are left to jax's own error.
    """
    q_len, kv_len = query.shape[-3], key.shape[-3]
    if q_len % 2 or kv_len % 2:
        return ("has no backward pass for odd sequence lengths, got "
                f"{q_len} and {kv_len}")
    return None


def scaled_dot_product_attention(query, key, value, dtype=None, precision=None,
                                 force_fp32_for_softmax=True, implementation=None,
                                 causal=False, sliding_window=None, mask=None):
    """The one attention kernel path for every attention module.

    Inputs are [B, S, H, D]. Keys and values may carry fewer heads than the
    query (grouped-query attention); the paths that cannot group heads
    themselves get them repeated out. The param trees of the callers never
    change with the implementation, so checkpoints are interchangeable across
    hardware:

    - None: flax reference attention (einsum + softmax). The only path that
      reads dtype, precision and force_fp32_for_softmax; the portable default.
    - 'auto': on a gpu backend 'cudnn' for the shapes cudnn supports and
      'xla' for the rest (`cudnn_supports`), 'xla' anywhere else. Resolved per
      trace, so a config logged as 'auto' still runs on the next machine, and
      per call, since one model can hold both a shape cudnn takes and a shape
      it does not.
    - 'xla' / 'cudnn': jax.nn.dot_product_attention, which dispatches to the
      fused cudnn flash kernel on supported GPUs. It takes no dtype, precision
      or softmax argument: the logits accumulate and the softmax runs in fp32
      whatever the inputs are. dtype is ignored, the other two are rejected
      rather than silently dropped.
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
            query, key, value, mask=mask, dtype=dtype, broadcast_dropout=False,
            dropout_rng=None, precision=precision,
            force_fp32_for_softmax=force_fp32_for_softmax, deterministic=True)

    if implementation == 'auto':
        implementation = ('cudnn' if jax.default_backend() == 'gpu'
                          and cudnn_supports(query, key) is None else 'xla')

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

    if implementation in ('xla', 'cudnn'):
        if implementation == 'cudnn' and query.dtype not in (jnp.bfloat16, jnp.float16):
            raise ValueError(
                "cudnn attention needs bf16 or fp16 inputs, the query is "
                f"{query.dtype}. Set dtype bfloat16, or attention_impl 'xla' to "
                "keep this precision.")
        # A left window of l means the l+1 most recent keys on both the xla and
        # the cudnn path, which is the window this function counts.
        return jax.nn.dot_product_attention(
            query, key, value, mask=mask, is_causal=causal,
            local_window_size=None if sliding_window is None else (sliding_window - 1, 0),
            implementation=implementation)
    if implementation == 'tpu':
        from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
        heads = query.shape[-2]
        key = repeat_kv_heads(key, heads)
        value = repeat_kv_heads(value, heads)
        # pallas wants [B, H, S, D]
        q = jnp.moveaxis(query, -2, -3)
        k = jnp.moveaxis(key, -2, -3)
        v = jnp.moveaxis(value, -2, -3)
        bias = None
        if mask is not None or sliding_window is not None:
            if sliding_window is not None:
                band = causal_attention_mask(
                    jnp.arange(query.shape[-3]), key.shape[-3], sliding_window)
                mask = band if mask is None else jnp.logical_and(mask, band)
            bias = jnp.broadcast_to(
                jnp.where(mask, 0, jnp.finfo(q.dtype).min).astype(q.dtype),
                (q.shape[0], q.shape[1], q.shape[2], k.shape[2]))
        out = flash_attention(q, k, v, ab=bias, causal=causal,
                              sm_scale=1.0 / math.sqrt(query.shape[-1]))
        return jnp.moveaxis(out, -3, -2)
    raise ValueError(f"Unknown attention implementation: {implementation}")


class NormalAttention(nn.Module):
    """
    Simple implementation of the normal attention.

    causal makes it a decoder attention (query i sees keys 0..i). decode=True
    on a call runs it against a fixed-size KV cache instead, allocated at
    max_seq_len: the first call writes the whole prompt, later calls append one
    token each. Neither flag touches the param tree, so a model trained without
    either reloads into a decoding one unchanged.
    """
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    use_bias: bool = True
    # kernel_init: Callable = kernel_init(1.0)
    force_fp32_for_softmax: bool = True
    qk_norm: bool = False  # RMSNorm on q/k per head (SD3-style bf16 logit safety)
    attention_impl: Optional[str] = None  # None (reference) | 'auto' | 'xla' | 'cudnn' | 'tpu'
    causal: bool = False
    max_seq_len: Optional[int] = None  # KV cache length, required to decode
    logical_axes: bool = False

    def setup(self):
        kernel_init = nn.linear.default_kernel_init
        qkv_bias_init = nn.initializers.zeros
        out_bias_init = nn.initializers.zeros
        out_kernel_init = nn.linear.default_kernel_init
        if self.logical_axes:
            kernel_init = nn.with_partitioning(
                kernel_init, ("embed", "heads", "head_dim"))
            qkv_bias_init = nn.with_partitioning(
                qkv_bias_init, ("heads", "head_dim"))
            out_kernel_init = nn.with_partitioning(
                out_kernel_init, ("heads", "head_dim", "embed"))
            out_bias_init = nn.with_partitioning(out_bias_init, ("embed",))
        dense = functools.partial(
            nn.DenseGeneral,
            features=[self.heads, self.dim_head],
            axis=-1,
            precision=self.precision,
            use_bias=self.use_bias,
            kernel_init=kernel_init,
            bias_init=qkv_bias_init,
            dtype=self.dtype,
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
            kernel_init=out_kernel_init,
            bias_init=out_bias_init,
            name="to_out_0",
        )

    @nn.compact
    def __call__(self, x, context=None, decode: bool = False):
        # x has shape [B, H, W, C]
        orig_x_shape = x.shape
        if len(x.shape) == 4:
            B, H, W, C = x.shape
            x = x.reshape((B, H*W, C))
        context = x if context is None else context
        if len(context.shape) == 4:
            context = context.reshape((B, H*W, C))
        query = self.query(x)
        key = self.key(context)
        value = self.value(context)
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

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
    r"""
    Flax implementation of a Linear layer followed by the variant of the gated linear unit activation function from
    https://arxiv.org/abs/2002.05202.

    Parameters:
        dim (:obj:`int`):
            Input hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32
    precision: Any = jax.lax.Precision.DEFAULT

    def setup(self):
        inner_dim = self.dim * 4
        self.proj = nn.Dense(inner_dim * 2, dtype=self.dtype, precision=self.precision)

    def __call__(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=-1)
        return hidden_linear * nn.gelu(hidden_gelu)
    
class FlaxFeedForward(nn.Module):
    r"""
    Flax module that encapsulates two Linear layers separated by a non-linearity. It is the counterpart of PyTorch's
    [`FeedForward`] class, with the following simplifications:
    - The activation function is currently hardcoded to a gated linear unit from:
    https://arxiv.org/abs/2002.05202
    - `dim_out` is equal to `dim`.
    - The number of hidden dimensions is hardcoded to `dim * 4` in [`FlaxGELU`].

    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    dim: int
    dtype: jnp.dtype = jnp.float32
    precision: Any = jax.lax.Precision.DEFAULT

    def setup(self):
        # The second linear layer needs to be called
        # net_2 for now to match the index of the Sequential layer
        self.net_0 = FlaxGEGLU(self.dim, self.dtype, precision=self.precision)
        self.net_2 = nn.Dense(self.dim, dtype=self.dtype, precision=self.precision)

    def __call__(self, hidden_states):
        hidden_states = self.net_0(hidden_states)
        hidden_states = self.net_2(hidden_states)
        return hidden_states

class BasicTransformerBlock(nn.Module):
    # Has self and cross attention
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    use_bias: bool = True
    # kernel_init: Callable = kernel_init(1.0)
    use_cross_only:bool = False
    only_pure_attention:bool = False
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-4
    attention_impl: Optional[str] = None
    
    def setup(self):
        attenBlock = NormalAttention
            
        self.attention1 = attenBlock(
         query_dim=self.query_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            name=f'Attention1',
            precision=self.precision,
            use_bias=self.use_bias,
            dtype=self.dtype,
            # kernel_init=self.kernel_init,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            attention_impl=self.attention_impl
        )
        self.attention2 = attenBlock(
            query_dim=self.query_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            name=f'Attention2',
            precision=self.precision,
            use_bias=self.use_bias,
            dtype=self.dtype,
            # kernel_init=self.kernel_init,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            attention_impl=self.attention_impl
        )
        
        self.ff = FlaxFeedForward(dim=self.query_dim, dtype=self.dtype, precision=self.precision)
        self.norm1 = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)
        self.norm2 = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)
        self.norm3 = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)
        
    @nn.compact
    def __call__(self, hidden_states, context=None):
        if self.only_pure_attention:
            return self.attention2(hidden_states, context)
        
        # self attention
        if not self.use_cross_only:
            hidden_states = hidden_states + self.attention1(self.norm1(hidden_states))
        
        # cross attention
        hidden_states = hidden_states + self.attention2(self.norm2(hidden_states), context)
        # feed forward
        hidden_states = hidden_states + self.ff(self.norm3(hidden_states))
        
        return hidden_states

class TransformerBlock(nn.Module):
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
    # kernel_init: Callable = kernel_init(1.0)
    norm_inputs: bool = True
    explicitly_add_residual: bool = True
    norm_epsilon: float = 1e-4

    @nn.compact
    def __call__(self, x, context=None):
        inner_dim = self.heads * self.dim_head
        C = x.shape[-1]
        if self.norm_inputs:
            x = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)(x)
        if self.use_projection == True:
            if self.use_linear_attention:
                projected_x = nn.Dense(features=inner_dim, 
                                       use_bias=False, precision=self.precision, 
                                    #    kernel_init=self.kernel_init,
                                       dtype=self.dtype, name=f'project_in')(x)
            else:
                projected_x = nn.Conv(
                    features=inner_dim, kernel_size=(1, 1),
                    # kernel_init=self.kernel_init,
                    strides=(1, 1), padding='VALID', use_bias=False, dtype=self.dtype,
                    precision=self.precision, name=f'project_in_conv',
                )(x)
        else:
            projected_x = x
            inner_dim = C
            
        context = projected_x if context is None else context

        projected_x = BasicTransformerBlock(
            query_dim=inner_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            name=f'Attention',
            precision=self.precision,
            use_bias=False,
            dtype=self.dtype,
            use_cross_only=(not self.use_self_and_cross),
            only_pure_attention=self.only_pure_attention,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            attention_impl=self.attention_impl,
            norm_epsilon=self.norm_epsilon
            # kernel_init=self.kernel_init
        )(projected_x, context)
        
        if self.use_projection == True:
            if self.use_linear_attention:
                projected_x = nn.Dense(features=C, precision=self.precision, 
                                       dtype=self.dtype, use_bias=False, 
                                    #    kernel_init=self.kernel_init,
                                       name=f'project_out')(projected_x)
            else:
                projected_x = nn.Conv(
                    features=C, kernel_size=(1, 1),
                    # kernel_init=self.kernel_i nit,
                    strides=(1, 1), padding='VALID', use_bias=False, dtype=self.dtype,
                    precision=self.precision, name=f'project_out_conv',
                )(projected_x)
                
        if self.only_pure_attention or self.explicitly_add_residual:
            projected_x = x + projected_x
            
        out = projected_x
        return out