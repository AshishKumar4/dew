"""Attention's TPU routes: the pallas splash kernel (`splash_attention`) and its mask descriptor, and the
older pallas flash kernel behind it for the masks splash cannot carry (`tpu_attention`).
"""

import math

import jax
import jax.numpy as jnp

from .attention_kernels import SPLASH_LANES
from .attention_masks import repeat_kv_heads, structural_mask
from .attention_parallel import four_dimensional, per_shard, shard_heads
from .documents import document_ids
from .precision import at_default_precision
from .sharding import SCORES
from .visibility import window_sides

# How much of an explicit boolean mask splash will carry. The array is read
# on the host while the executable is built and its unresolved blocks are
# stored dense inside it, so this bounds host and executable bytes, not
# device memory that grows with the batch. 4 Mi cells is a 16-head 512x512.
SPLASH_DENSE_MASK_CELLS = 1 << 22


def tpu_attention(query, key, value, bias, mask, causal, sliding_window, *,
                  softcap, sinks, segment_ids, interpret: bool):
    """Attend on pallas: splash where its descriptor covers the call, flash
    everywhere else.

    Splash is block-sparse: its mask is a descriptor built with the executable,
    so a causal or windowed sequence costs its live blocks, and packed segment
    ids are a kernel argument. It has no bias argument and needs an explicit
    mask readable on the host. Flash takes both as one additive [B, H, Q, K]
    array over the whole rectangle, so it serves exactly:

    - an additive `bias` (T5's relative position table);
    - a traced `mask` (a cache decode mask, sequence parallelism's striped
      mask), one past `SPLASH_DENSE_MASK_CELLS`, or one that varies by row;
    - a length that is not a multiple of `SPLASH_LANES`.

    Flash has no softcap and no sinks, so such a call raises. `interpret` runs
    splash under pallas's interpreter, which is how the parity tests run its
    arithmetic off a TPU; it is far slower than XLA, so only an explicit 'tpu'
    uses it. The flash kernel has no interpreter and runs only on a TPU.

    A Mosaic kernel cannot be partitioned automatically, so on a mesh both run
    per shard (`per_shard`), on its rows and heads. Splash's descriptor then
    covers one shard's heads, so a mask that differs by head where the mesh
    splits the heads is flash's.
    """
    q_len, kv_len = query.shape[-3], key.shape[-3]
    descriptor = None
    if bias is None and not (q_len % SPLASH_LANES or kv_len % SPLASH_LANES):
        descriptor = splash_mask_descriptor(
            q_len, kv_len, shard_heads(query), causal, sliding_window, mask)
    if descriptor is None and (softcap is not None or sinks is not None):
        raise ValueError(
            "attention implementation 'tpu' takes a softcap and sinks only on "
            "the splash kernel, and this call has a bias, a mask splash cannot "
            "describe or a length that is not a multiple of "
            f"{SPLASH_LANES}. Use attention_impl 'xla'.")

    def local(query, key, value, bias=None, mask=None, sinks=None, segment_ids=None):
        if descriptor is None:
            return pallas_flash_attention(query, key, value, bias, mask, causal,
                                          sliding_window, segment_ids)
        return splash_attention(query, key, value, descriptor, softcap=softcap, sinks=sinks,
                                segment_ids=segment_ids, interpret=interpret)

    def attend(query, key, value, bias, sinks):
        # Splash holds the mask in its descriptor; flash takes it as an array.
        held = None if descriptor is not None or mask is None else four_dimensional(mask)
        return per_shard(local, query, key, value,
                          bias=(None if bias is None else four_dimensional(bias), SCORES),
                          mask=(held, SCORES), sinks=(sinks, ("activation_heads",)),
                          segment_ids=(segment_ids, ("activation_batch", None)))

    if jnp.finfo(query.dtype).bits == 16 and not interpret:
        # Mosaic multiplies 16-bit operands at the default precision only.
        # Their products are exact and accumulate in fp32; splash's fp32 P·V
        # takes one bf16 pass, as at any default-precision run
        # (`at_default_precision`).
        return at_default_precision(attend)(query, key, value, bias, sinks)
    return attend(query, key, value, bias, sinks)


def splash_attention(query, key, value, descriptor, *, softcap, sinks, segment_ids,
                     interpret: bool):
    """Run the splash kernel over [B, S, H, D] arrays under `descriptor`.

    The kernel takes one [H, S, D] example at a time, so the batch rides a
    vmap. It applies no scale, so 1/sqrt(d) goes onto the query in the query's
    dtype, where flax's reference puts it, which makes the two agree exactly in
    fp32. Grouped key heads stay grouped (splash reads q_heads % kv_heads).
    `softcap` is applied to the scaled logits before the mask and `sinks` `[H]`
    seed each head's running maximum and denominator, with their gradient, as
    in `reference_attention`. The kernel
    lets equal segment ids attend, so padding (id 0) reads other padding where
    `document_mask` reads nothing; no document's query reads padding either way.
    The kernel is JAX's, vendored with its backward's delta formed from the
    weights (`dew.nn.kernels.splash`).
    """
    from dew.nn.kernels import splash

    kernel = splash.make_splash_mha(
        descriptor, block_sizes=splash.block_sizes(query.shape[-3], key.shape[-3], width=query.shape[-1],
                                                   itemsize=query.dtype.itemsize,
                                                   softcap=softcap is not None),
        attn_logits_soft_cap=softcap, interpret=interpret)
    scale = jnp.asarray(1.0 / math.sqrt(query.shape[-1]), query.dtype)
    segments = None if segment_ids is None else splash.SegmentIds(*document_ids(segment_ids))

    def example(q, k, v, ids):
        return kernel(q, k, v, segment_ids=ids, sinks=sinks)

    attended = jax.vmap(example)(jnp.moveaxis(query, -2, -3) * scale,
                                 jnp.moveaxis(key, -2, -3), jnp.moveaxis(value, -2, -3),
                                 segments)
    return jnp.moveaxis(attended, -3, -2)


def splash_mask_descriptor(q_len: int, kv_len: int, heads: int, causal: bool,
                           sliding_window: int | None, mask):
    """Build splash's mask for one call, or None when splash cannot describe it.

    CausalMask and LocalMask are comparisons the kernel evaluates per block, so
    emptied blocks leave the grid and nothing Q*K-sized is stored. Dew's window
    holds the causal flag's side already (`window_sides`), so it replaces the
    flag. An explicit boolean mask
    is ANDed in as dense blocks, so only a concrete, small one is taken.
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

    shape = (q_len, kv_len)
    if sliding_window is not None:
        structural = splash_attention_mask.LocalMask(shape, window_sides(causal, sliding_window), 0)
    elif causal:
        structural = splash_attention_mask.CausalMask(shape)
    else:
        structural = splash_attention_mask.FullMask(shape)
    if mask is None:
        return splash_attention_mask.MultiHeadMask([structural] * heads)
    per_head = splash_dense_mask(mask, q_len, kv_len, heads)
    if per_head is None:
        return None
    dense = [splash_attention_mask.NumpyMask(rows) for rows in per_head]
    if isinstance(structural, splash_attention_mask.FullMask):
        return splash_attention_mask.MultiHeadMask(dense)
    return splash_attention_mask.MultiHeadMask(
        [splash_attention_mask.LogicalAnd(structural, head) for head in dense])


def splash_dense_mask(mask, q_len: int, kv_len: int, heads: int):
    """Return an explicit mask as one host [Q, K] boolean array per query head.

    None when splash cannot carry it: a traced mask (the descriptor is built
    with the executable; a decode mask over slots is one), a batch axis wider
    than one (splash has none), or one past `SPLASH_DENSE_MASK_CELLS`, whose
    dense blocks would cost more than the additive-bias path.
    """
    import numpy as np
    from jax.core import Tracer

    if isinstance(mask, Tracer):
        return None
    dense = np.asarray(mask, bool)
    while dense.ndim > 3 and dense.shape[0] == 1:
        dense = dense[0]
    if dense.ndim == 2:
        dense = dense[None]
    if dense.ndim != 3 or dense.shape[-2:] != (q_len, kv_len):
        return None
    if dense.shape[0] not in (1, heads) or dense.size > SPLASH_DENSE_MASK_CELLS:
        return None
    return [dense[head % dense.shape[0]] for head in range(heads)]


def pallas_flash_attention(query, key, value, bias, mask, causal, sliding_window, segment_ids):
    """Run the pallas TPU flash kernel, with every mask but the documents as
    an additive bias.

    A window and an explicit mask become one [B, H, Q, K] array of zeros and the
    dtype's minimum; causality is a flag and packed documents its `segment_ids`,
    with splash's padding rule. That rectangle is what splash avoids, so only the
    calls `tpu_attention` names come here. 1/sqrt(d) goes onto the query and
    `sm_scale` stays 1, because the kernel adds `ab` before it multiplies by
    `sm_scale` (flash_attention.py:402-409), which would scale T5's bias too.
    """
    from jax.experimental.pallas.ops.tpu.flash_attention import SegmentIds, flash_attention

    heads = query.shape[-2]
    key = repeat_kv_heads(key, heads)
    value = repeat_kv_heads(value, heads)
    # pallas wants [B, H, S, D]
    q = jnp.moveaxis(query, -2, -3) * jnp.asarray(1.0 / math.sqrt(query.shape[-1]), query.dtype)
    k = jnp.moveaxis(key, -2, -3)
    v = jnp.moveaxis(value, -2, -3)
    combined = None
    if bias is not None:
        combined = jnp.broadcast_to(bias.astype(q.dtype),
                                    (q.shape[0], q.shape[1], q.shape[2], k.shape[2]))
    # The kernel takes causality as its flag, so only a window becomes a band.
    band = None if sliding_window is None else structural_mask(
        jnp.arange(query.shape[-3]), key.shape[-3], causal, sliding_window)
    if band is not None:
        mask = band if mask is None else jnp.logical_and(mask, band)
    if mask is not None:
        seated = jnp.broadcast_to(
            jnp.where(mask, 0, jnp.finfo(q.dtype).min).astype(q.dtype),
            (q.shape[0], q.shape[1], q.shape[2], k.shape[2]))
        combined = seated if combined is None else combined + seated
    segments = None if segment_ids is None else SegmentIds(*document_ids(segment_ids))
    return jnp.moveaxis(
        flash_attention(q, k, v, ab=combined, segment_ids=segments, causal=causal), -3, -2)
