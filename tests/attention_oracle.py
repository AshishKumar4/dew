"""Attention written out apart from every Dew path: the oracle the attention tests hold each kernel to.

`kept` is the keys each query keeps, written from the structure's definition (causality, windows,
documents, a mask); `float64_weights` and `exact_attention` the softmax over them, sinks and a softcap
included; `attention_terms` each entry's float64 terms for the gate's sizes; `value_and_grads` the
harness that runs a Dew implementation to its output and three input gradients. No Dew attention code
runs here but the implementation a harness names.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.attention import scaled_dot_product_attention


def qkv(shape, dtype=jnp.bfloat16, seed=0, kv_shape=None):
    """Standard-normal queries, keys and values of `shape` (keys and values `kv_shape` where given),
    three streams of one seed."""
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    return (jax.random.normal(keys[0], shape, dtype),
            *(jax.random.normal(key, kv_shape or shape, dtype) for key in keys[1:]))


def packed_ids(*rows, length=256):
    """`[B, S]` segment ids, one row per tuple of document lengths, 1-based,
    with the rest of each row padding (0)."""
    packed = [np.pad(np.repeat(np.arange(1, len(lengths) + 1), lengths),
                     (0, length - sum(lengths))) for lengths in rows]
    return jnp.asarray(np.stack(packed), jnp.int32)


def attention_and_grads(implementation, *, has_sinks=False, live=None, **kwargs):
    """Compile the output and gradients once for repeated rounding draws,
    sinks included, in fp32 or the inputs' wider dtype.

    `live` `[B, S]` keeps the rows the loss reads and the output compares,
    which a packed row's padding is not.
    """
    def loss(q, k, v, s):
        out = scaled_dot_product_attention(q, k, v, implementation=implementation, sinks=s,
                                           **kwargs)
        if live is not None:
            out = jnp.where(live[:, :, None, None], out, 0)
        return jnp.sum(out.astype(jnp.promote_types(out.dtype, jnp.float32)) ** 2), out

    argnums = (0, 1, 2, 3) if has_sinks else (0, 1, 2)
    return jax.jit(jax.value_and_grad(loss, argnums=argnums, has_aux=True))


def value_and_grads(implementation, query, key, value, sinks=None, **kwargs):
    (_, out), grads = attention_and_grads(implementation, has_sinks=sinks is not None, **kwargs)(
        query, key, value, sinks)
    return [np.asarray(x, np.promote_types(x.dtype, np.float32)) for x in (out, *grads)]


def reference(query, key, value, **kwargs):
    """The reference attention of these inputs, computed in fp32.

    HIGHEST keeps a TPU's fp32 matmuls at fp32, where DEFAULT runs them on
    bf16 passes; XLA:CPU computes fp32 either way."""
    return value_and_grads('reference', *(x.astype(jnp.float32) for x in (query, key, value)),
                           precision=jax.lax.Precision.HIGHEST, **kwargs)


def kept(q_len, k_len, *, causal=False, sliding_window=None, window=None, mask=None, segment_ids=None):
    """The keys each query keeps, `[B or 1, 1, q, k]`: k <= q; a causal window's
    w most recent, or without causality the keys within w - 1 of the query; the
    `window` `(left, right)` reaches, -1 bounding neither side; the explicit
    mask; equal segment ids, none for padding."""
    rows, cols = np.arange(q_len)[:, None], np.arange(k_len)[None, :]
    keep = np.ones((1, 1, q_len, k_len), bool)
    if causal:
        keep = keep & (cols <= rows)
    if sliding_window is not None:
        # The keys within w - 1 of the query on either side.
        keep = keep & (cols - rows > -sliding_window) & (cols - rows < sliding_window)
    if window is not None:
        left, right = window
        if left >= 0:
            keep = keep & (cols - rows >= -left)
        if right >= 0:
            keep = keep & (cols - rows <= right)
    if mask is not None:
        keep = keep & np.asarray(mask, bool)
    if segment_ids is not None:
        ids = np.asarray(segment_ids)
        # Segment 0 is padding, which belongs to no document (`document_mask`).
        keep = keep & (ids[:, None, :, None] == ids[:, None, None, :]) & (ids[:, None, :, None] != 0)
    return keep


@jax.custom_vjp
def anchored_softmax(logits):
    """`jax.nn.softmax` over the last axis, whose VJP takes each row's `D = sum p g` about its largest
    weight's `g`: `p ((g - g_a) - sum p (g - g_a))`, the same derivative since the weights sum to one,
    with no term the size of `g` cancelling to a tiny one. The float64 truth's form: at logits near 1e4
    a row's weights span 1 to 1e-22, and the plain form's `g_a - D` cancels below float64's ulp at
    `g_a`'s size, a weight of 1e-22's share lost from the anchor's gradient."""
    return jax.nn.softmax(logits, axis=-1)


def _anchored_forward(logits):
    weights = jax.nn.softmax(logits, axis=-1)
    return weights, weights


def _anchored_backward(weights, cotangent):
    anchor = jnp.take_along_axis(cotangent, jnp.argmax(weights, axis=-1, keepdims=True), axis=-1)
    relative = cotangent - anchor
    return (weights * (relative - jnp.sum(weights * relative, axis=-1, keepdims=True)),)


anchored_softmax.defvjp(_anchored_forward, _anchored_backward)


def float64_weights(q, k, keep, sinks, softcap, *, anchored: bool = False):
    """The attention weights of `[B, S, H, D]` queries over their keys, key
    heads repeated over their group, and the logits before the cap: over
    sqrt(D), capped `c tanh(l / c)`, the keys `keep` `[B or 1, 1, q, k]`
    keeps, a softmax whose denominator a sink joins, its derivative the
    float64 truth's (`anchored_softmax`) where `anchored`. A query that keeps
    no key has no weight."""
    heads, kv_heads = q.shape[2], k.shape[2]
    group = jnp.arange(heads) // (heads // kv_heads)
    raw = jnp.einsum('bqhd,bkhd->bhqk', q, k[:, :, group]) / np.sqrt(q.shape[3])
    logits = raw if softcap is None else softcap * jnp.tanh(raw / softcap)
    keep = np.broadcast_to(keep, logits.shape)
    sees = keep.any(-1, keepdims=True)
    # A row that keeps no key softmaxes zeros instead of only -inf, whose
    # NaN would reach every gradient through the weights zeroed below.
    logits = jnp.where(keep, logits, jnp.where(sees, -jnp.inf, 0.0))
    if sinks is not None:
        sink = jnp.broadcast_to(sinks[None, :, None, None], (*logits.shape[:3], 1))
        logits = jnp.concatenate([logits, sink], -1)
    softmax = anchored_softmax if anchored else functools.partial(jax.nn.softmax, axis=-1)
    return jnp.where(keep & sees, softmax(logits)[..., :keep.shape[-1]], 0.0), raw


def attention_terms(query, key, value, cotangent, keep, sinks=None, softcap=None):
    """Each entry's float64 terms of attention's output and its query, key and
    value gradients under `cotangent` (`reference_error.float64_terms`'s
    sizes), for `[B, S, H, D]` operands with key heads repeated over their
    group and the keys `keep` `[B or 1, 1, q, k]` keeps, a sink joining each
    head's denominator. A logit's gradient P (dP - delta) cancels: on a query
    that keeps one key P is 1 and delta is dP, so it is zero in float64,
    while a flash kernel forms delta as rowsum(out * dout) from its bf16
    output. Each of those products carries the output's rounding, so delta's
    error is of their size, sum |out dout|, however far their sum cancels: a
    query that keeps one key beside a sink, whose output is a rounded share of
    the value, shows it. Its terms are therefore P (|dP| + sum |out dout|),
    and such a row is held to that rounding, not to its own zero."""
    with jax.enable_x64(new_val=True):
        q, k, v, dout = (jnp.asarray(np.asarray(x, np.float64)) for x in (query, key, value, cotangent))
        heads, kv_heads, width = q.shape[2], k.shape[2], q.shape[3]
        group = jnp.arange(heads) // (heads // kv_heads)
        sunk = None if sinks is None else jnp.asarray(np.asarray(sinks, np.float64))
        weights, raw = float64_weights(q, k, keep, sunk, softcap)
        products = jnp.einsum('bqhd,bkhd->bhqk', dout, v[:, :, group])
        out = jnp.einsum('bhqk,bkhd->bqhd', weights, v[:, :, group])
        delta = jnp.einsum('bqhd,bqhd->bhq', jnp.abs(out), jnp.abs(dout))[..., None]
        logit_terms = weights * (jnp.abs(products) + delta) / np.sqrt(width)
        if softcap is not None:
            logit_terms = logit_terms * jnp.abs(1 - jnp.tanh(raw / softcap) ** 2)

        def per_key_head(x):
            return x.reshape(*x.shape[:2], kv_heads, heads // kv_heads, x.shape[-1]).sum(3)

        terms = (jnp.einsum('bhqk,bkhd->bqhd', weights, jnp.abs(v[:, :, group])),
                 jnp.einsum('bhqk,bkhd->bqhd', logit_terms, jnp.abs(k[:, :, group])),
                 per_key_head(jnp.einsum('bhqk,bqhd->bkhd', logit_terms, jnp.abs(q))),
                 per_key_head(jnp.einsum('bhqk,bqhd->bkhd', weights, jnp.abs(dout))))
        return [np.asarray(x) for x in terms]


def exact_attention(q, k, v, s=None, *, softcap=None, keep=None, anchored: bool = False, **structure):
    """Attention written out apart from every Dew path, in the inputs' dtype: the logits of query
    head h against key head h // (Hq // Hkv) over sqrt(D), capped `c tanh(l / c)`, then the keys
    `keep` `[B or 1, 1, q, k]` keeps (`kept`'s by `structure` where None), a softmax whose denominator
    a sink joins, and the values. A query that keeps no key reads nothing: its output is zero. The
    float64 truth takes `anchored`, its softmax's derivative about each row's largest weight
    (`anchored_softmax`); the reference arm, the plain one, as XLA differentiates it."""
    group = jnp.arange(q.shape[2]) // (q.shape[2] // k.shape[2])
    keep = kept(q.shape[1], k.shape[1], **structure) if keep is None else keep
    weights, _ = float64_weights(q, k, keep, s, softcap, anchored=anchored)
    return jnp.einsum('bhqk,bkhd->bqhd', weights, v[:, :, group])


def exact_value_and_grads(query, key, value, sinks=None, live=None, softcap=None, **structure):
    """`value_and_grads`' output and gradients in float64 (`exact_attention`, its anchored
    derivative)."""
    def loss(q, k, v, s):
        out = exact_attention(q, k, v, s, softcap=softcap, anchored=True, **structure)
        if live is not None:
            out = jnp.where(np.asarray(live)[:, :, None, None], out, 0)
        return jnp.sum(out ** 2), out

    argnums = (0, 1, 2) if sinks is None else (0, 1, 2, 3)
    with jax.enable_x64(new_val=True):
        wide = [None if x is None else jnp.asarray(np.asarray(x, np.float64))
                for x in (query, key, value, sinks)]
        (_, out), grads = jax.value_and_grad(loss, argnums=argnums, has_aux=True)(*wide)
        return [np.asarray(x) for x in (out, *grads)]
