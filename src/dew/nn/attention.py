"""Attend over queries, keys and values: one kernel path for every attention module.

`AttentionImpl` names the kernels, `resolve_implementation` takes one per trace, and
`scaled_dot_product_attention` and `attention_kernel` run it, with `reference_attention` the portable
path. The masks a call reads live in `dew.nn.attention_masks`, the sequence-parallel wrappers in
`dew.nn.attention_parallel`, the TPU routes in `dew.nn.attention_tpu`, the norms in `dew.nn.norms`,
and the modules built on them in `dew.nn.attention_blocks`.
"""

import contextlib
import contextvars
import dataclasses
import functools
import importlib
from typing import Literal, get_args

import jax
import jax.numpy as jnp
import tokamax
from flax.linen.dtypes import promote_dtype
from jax.ad_checkpoint import checkpoint_name

from .attention_kernels import ATTENTION, DECODE, TPU_BACKEND, AttentionCase, decodes
from .attention_masks import (
    combined_attention_mask,
    key_spans,
    repeat_kv_heads,
    visibility_mask,
    visible_rows,
    with_key_lengths,
    with_key_starts,
)
from .attention_parallel import (
    four_dimensional,
    on_attention_shards,
    pad_rows,
    sequence_parallel_attention,
    shard_heads,
    split_with_queries,
    tensor_shards,
)
from .attention_sinks import sink_softmax
from .attention_tpu import splash_mask_descriptor, tpu_attention
from .decode_geometry import folded_attention, folds
from .documents import (
    CrossDocuments,
    cross_document_mask,
    document_ids,
    document_offsets,
    query_order,
    with_documents,
)
from .inputs import cu_seqlens
from .kernels import decode_attention, generation
from .kernels.generation import bf16_dot_runs, gpu_placed
from .kernels.selection import forced, select, selected
from .kernels.templates import attention as template, attention_route
from .precision import (
    asks_default_precision,
    at_least_fp32,
    capped_logits,
    configured_precision,
    float32_einsum,
    float32_products,
    wide_cpu_arithmetic,
)
from .sharding import SCORES, sequence_shards
from .visibility import window_sides

AttentionImpl = Literal["auto", "reference", "xla", "cudnn", "triton", "flash", "tpu", "template"]
"""Names which kernel an attention call runs.

Every layer that carries the choice spells it the same way: a `ModelConfig`
field, a module's `attention_impl`, and `scaled_dot_product_attention`'s
`implementation`.

'reference' is the portable einsum and softmax, and the only path that reads
dtype, precision and force_fp32_for_softmax. 'xla' and 'cudnn' are
`jax.nn.dot_product_attention`'s own two. 'triton' is tokamax's Pallas-Triton
flash kernel (`triton_attention`). 'flash' is FlashAttention (`flash_attention`),
FlashAttention-3 on Hopper and FlashAttention-2 elsewhere, from the
dew_flash_attn wheel (docs/installation.md). 'tpu' is the
pallas splash kernel, with the older pallas flash kernel behind it for the
calls splash's mask descriptor cannot carry.
'template' is the attention template's route (`attention_route.attend`): the call's visibility as
spans, a softcap as a score hook and sinks as the initial state, on the template's Triton or Mosaic
mainloop on a GPU and its plain-JAX one elsewhere. 'auto' takes it on every GPU, where the call's
declared old route does not take it (`attention_kernels._attention_preferred`, `DECLARED`).
'auto', every module's default, resolves per trace (`resolve_implementation`),
so a configuration logged as 'auto' runs on the next machine too: the attention selector (`ATTENTION`,
`dew.nn.kernels.selection.select`) takes the call's declared old route where a receipt measured it faster,
else the template, where its requirements hold (each kernel's `SUPPORTS`), else the first of flash,
triton, cuDNN, splash, xla and the reference path that takes it; a call of few
positions over key lengths is the decode selector's (`DECODE`). Validity reaches
the kernels as coordinates (causality, a window, documents, key lengths); an
arbitrary mask only the generic paths take. A call off the kernel the target's
row measured fastest is logged at INFO once a process with the reason, and
every choice is reported as the target's route (`dew.nn.kernels.generation.route`).
"""


def attention_implementation(name: str) -> AttentionImpl:
    """`name` as the `AttentionImpl` it names, a recorded or a user's value; refused unless it names
    one."""
    for implementation in get_args(AttentionImpl):
        if name == implementation:
            return implementation
    raise ValueError(f"Unknown attention implementation: {name}; one of {', '.join(get_args(AttentionImpl))}")


def widen_value_heads(query, value):
    """Zero-pad `value`'s head axis out to the query's width.

    `jax.nn.dot_product_attention` refuses a value narrower than the key, which
    is DeepSeek's latent attention (`v_head_dim` 128 against a 192-wide query in
    every released V2/V3 config). Each value column is its own sum, so a zero
    column gives a zero output column the caller crops; transformers 5.16.1 pads
    FlashAttention the same way (integrations/flash_attention.py:63). A wider
    value raises: padding the query and key would move the kernel's 1/sqrt(d)
    off the query's width.
    """
    width, v_width = query.shape[-1], value.shape[-1]
    if v_width > width:
        raise ValueError(
            "fused attention runs one head width for the keys and the values, "
            f"and a value head of {v_width} is wider than the query head of "
            f"{width}: the kernel would scale the logits by the padded width. "
            "Use the reference implementation (attention_impl 'reference').")
    return jnp.pad(value, ((0, 0),) * (value.ndim - 1) + ((0, width - v_width),))


def cudnn_attention(query, key, value, bias, mask, causal, sliding_window,
                    key_value_seq_lengths=None):
    """Run jax's cudnn flash attention over any sequence length.

    cudnn has no backward pass for an odd query or key length (77 CLIP text
    tokens, most text-plus-image sequences), so one zero row pads it even. The
    padded query row is sliced off and the padded key hidden by the kernel's
    key_value_seq_lengths, which never reach past the real keys.
    tests/test_kernels.py pins the equality with the xla path. On a mesh it
    runs per shard (`on_attention_shards`).
    """
    q_len, kv_len = query.shape[-3], key.shape[-3]
    # jax's cudnn call takes a mask or bias only at the full [.., Q, K]
    # (check_layout in jax/_src/cudnn/fused_attention_stablehlo.py), so a
    # key-padding mask broadcast over the queries, [B, 1, 1, K], is widened.
    def spanning(x):
        return None if x is None else jnp.broadcast_to(x, (*x.shape[:-2], q_len, kv_len))
    mask, bias = spanning(mask), spanning(bias)
    q_pad, kv_pad = q_len % 2, kv_len % 2
    if q_pad or kv_pad:
        query = pad_rows(query, q_pad)
        key, value = pad_rows(key, kv_pad), pad_rows(value, kv_pad)
        # Masks and biases broadcast over [B, H, Q, K]; only the last two
        # dimensions grow, and a padded query row or key column sees nothing.
        def pad_tail(x, fill):
            return jnp.pad(x, ((0, 0),) * (x.ndim - 2) + ((0, q_pad), (0, kv_pad)),
                           constant_values=fill)
        mask = None if mask is None else pad_tail(mask, fill=False)
        bias = None if bias is None else pad_tail(bias, 0)
    kv_lengths = key_value_seq_lengths
    if kv_lengths is None and kv_pad:
        kv_lengths = jnp.full(key.shape[:1], kv_len, jnp.int32)

    def local(query, key, value, bias=None, mask=None, key_value_seq_lengths=None):
        # A window of (l, r) keeps the l keys before the query and the r after it
        # on both the xla and the cudnn path, the sides `window_sides` counts.
        return jax.nn.dot_product_attention(
            query, key, value, bias=bias, mask=mask, is_causal=causal,
            key_value_seq_lengths=key_value_seq_lengths,
            local_window_size=None if sliding_window is None else window_sides(causal, sliding_window),
            implementation='cudnn')

    out = on_attention_shards(local, query, key, value,
                    bias=(None if bias is None else four_dimensional(bias), SCORES),
                    mask=(None if mask is None else four_dimensional(mask), SCORES),
                    key_value_seq_lengths=(kv_lengths, ("activation_batch",)))
    return out[:, :q_len] if q_pad else out


def cudnn_runs(query, softcap=None) -> bool:
    """Whether cudnn's fused kernel runs a query of this dtype and width here: the target takes it
    (`CUDNN_SUPPORTS`) and this process places it on its own GPU (`gpu_placed`), which a CUDA
    library call needs."""
    case = attention_case(query, query, softcap=softcap)
    return gpu_placed() and not ATTENTION.kernel('cudnn').failed(case)


def flash_attention(query, key, value, causal: bool, sliding_window=None, sinks=None, segment_ids=None,
                    max_segments: int | None = None):
    """FlashAttention (dew_flash_attn, kernels/flash_attn) over `[B, S, H, D]`
    arrays at jax.nn's default scale: FlashAttention-3 on Hopper and
    FlashAttention-2 elsewhere. A window keeps the keys `window_sides` counts,
    and `sinks` `[H]` puts one value-free logit a head in each row's
    denominator. Packed rows, `segment_ids` `[B, S]` with at most
    `max_segments` documents a row, run as the kernel's varlen sequences: each
    document, and each row's padding after them, one sequence over the
    flattened tokens. Its causal mask is aligned to the last key, where
    jax.nn's is aligned to the first, so a causal call is square, or reads
    another sequence's documents, each document's queries aligned to its last
    keys (`CrossDocuments`), which it orders by id. On a mesh it
    runs per shard (`on_attention_shards`). What it takes is its requirements'
    (`attention_kernels.FLASH_SUPPORTS`), which the selector holds a call to before it runs here."""
    flash = importlib.import_module('dew_flash_attn')
    window = (-1, -1) if sliding_window is None else window_sides(causal, sliding_window)

    def local(query, key, value, sinks=None, segment_ids=None, key_segment_ids=None):
        if segment_ids is None or max_segments is None:
            return flash.flash_mha(query, key, value, is_causal=causal, window_size=window, sinks=sinks)
        rows, length = segment_ids.shape
        order = None
        if key_segment_ids is not None:
            # Another sequence's documents run as packed sequences in id order, padding after them:
            # the keys arrive so (a cache's), and the queries go so and come back in their own.
            order = query_order(segment_ids)
            query = jnp.take_along_axis(query, order[:, :, None, None], axis=1)
            segment_ids = jnp.take_along_axis(segment_ids, order, axis=1)
        flat = (query.reshape(rows * length, *query.shape[2:]),
                *(x.reshape(-1, *x.shape[2:]) for x in (key, value)))
        if key_segment_ids is None:
            # A call's own documents are its runs, whatever their ids.
            changes = segment_ids[:, 1:] != segment_ids[:, :-1]
            begins = jnp.concatenate([jnp.ones((rows, 1), bool), changes], axis=1)
            offsets, own_keys = cu_seqlens(begins, rows * (max_segments + 1)), {}
        else:
            # Another sequence's documents pair with the queries' by id, sequence i each side's
            # document i. FA3 takes a positive bound on the keys' lengths, which an empty key array
            # has none of.
            offsets, key_offsets = (document_offsets(ids, max_segments)
                                    for ids in (segment_ids, key_segment_ids))
            own_keys = {"key_cu_seqlens": key_offsets, "key_max_seqlen": max(1, key_segment_ids.shape[1])}
        out = flash.flash_mha_varlen(*flat, offsets, length, is_causal=causal, window_size=window,
                                     sinks=sinks, **own_keys)
        out = out.reshape(rows, length, *out.shape[1:])
        if order is None:
            return out
        back = jnp.zeros_like(order).at[jnp.arange(rows)[:, None], order].set(jnp.arange(length)[None, :])
        return jnp.take_along_axis(out, back[:, :, None, None], axis=1)

    queries, keys = (None, None) if segment_ids is None else document_ids(segment_ids)
    if queries is not None and keys is not None and (
            query.shape[:-2] != queries.shape or key.shape[:-2] != keys.shape
            or value.shape[:-2] != keys.shape):
        # Packed rows run flattened into one run of tokens, where rows of another count or length can
        # flatten to as many tokens and be read as the next row's.
        raise ValueError(
            f"packed flash attention reads each token by its document's id: queries {query.shape[:-2]} "
            f"beside ids {queries.shape}, keys {key.shape[:-2]} and values {value.shape[:-2]} beside ids "
            f"{keys.shape}")
    return on_attention_shards(local, query, key, value, sinks=(sinks, ("activation_heads",)),
                      segment_ids=(queries, ("activation_batch", None)),
                      key_segment_ids=(keys if isinstance(segment_ids, CrossDocuments) else None,
                                       ("activation_batch", None)))


def triton_attention(query, key, value, causal: bool):
    """tokamax's Pallas-Triton flash kernel over `[B, S, H, D]` arrays, at
    tokamax's heuristic config, with jax.nn's default scale of 1/sqrt(D).

    It takes grouped key heads and any length itself. A Pallas call has no
    partitioning rule, so on a mesh it runs per shard (`on_attention_shards`).

    The queries reach it scaled once, at its scale 1: its forward and dQ
    sweep scale the queries but its dK/dV sweep the keys (tokamax 0.0.15,
    pallas_triton_vjp), so at large logits that sweep's scores round apart
    from the forward's and the weights it rebuilds from the forward's
    residuals go off by their difference's exponential (row 1676: e^76 at
    logits near 1e4, a MeanFlow step's key and value gradients 1e35 times
    xla's error).
    """

    return on_attention_shards(lambda query, key, value: tokamax.dot_product_attention(
        query * query.shape[-1] ** -0.5, key, value, scale=1.0, is_causal=causal,
        implementation='triton'), query, key, value)


def template_attention(query, key, value, causal: bool, sliding_window, *, sinks, softcap, segment_ids,
                       lengths, starts, block_groups=None, image_groups=None, key_valid=None, key_runs=None,
                       places=None, dropout_rate=0.0, dropout_seed=None, dropout_places=None, bias=None):
    """The attention template's route (`attention_route.attend`) over the call's visibility as spans
    (`key_spans`): its causality, window, documents (`segment_ids`, or `CrossDocuments` aligned to their last
    keys), key lengths and starts, block and image groups, and key validity and counted places, with
    probability dropout at `dropout_rate` drawn from `dropout_seed`. A softcap is a score hook, a `bias` one
    term a key `[B or 1, H or 1, 1, K]` (ALiBi's, over the keys' logical places) a key score hook added after
    it (`template.key_biased`), and sinks are the softmax's initial state. On a mesh it runs per shard
    (`on_attention_shards`), as the decode kernel does: the key heads repeat until the tensor axes split
    them, the sinks split with their heads, and the ids, lengths, starts, groups, validity and places with
    their rows. Its requirements (`attention_kernels.TEMPLATE_SUPPORTS`) keep it off a sharded sequence axis.
    """
    if sinks is not None and sinks.shape != (query.shape[2],):
        raise ValueError(f"sinks must have one logit per query head, got {sinks.shape}")

    def local(query, key, value, sinks=None, segment_ids=None, lengths=None, starts=None,
              block_groups=None, image_groups=None, key_valid=None, key_runs=None, places=None,
              global_rows=None, global_heads=None, seed=None, bias=None):
        batch, queries, keys = query.shape[0], query.shape[1], key.shape[1]
        spans = key_spans(queries, keys, causal, sliding_window, segment_ids, lengths, starts,
                          block_groups=block_groups, image_groups=image_groups, key_valid=key_valid,
                          key_runs=key_runs,
                          places=places, rows=batch)
        drop = None
        if seed is not None and dropout_places is not None:
            # A sequence exchange's slice draws at the call's global places (`template.DropPlaces`).
            drop = template.Dropout(seed, dropout_places.rows, dropout_places.heads, dropout_rate,
                                    dropout_places.queries, dropout_places.keys)
        elif seed is not None and global_rows is not None and global_heads is not None:
            drop = template.Dropout(seed, global_rows, global_heads, dropout_rate)
        capped = None if softcap is None else functools.partial(template.capped, softcap)
        hooks = template.Hooks(
            score_mod=capped if bias is None else functools.partial(template.key_biased, capped),
            score_params=() if bias is None else (template.Param(bias[:, :, 0], template.Kind.KEY),),
            initial=None if sinks is None else template.sinks(sinks, batch, queries, value.shape[-1]),
            drop=drop)
        return attention_route.attend(query, key, value, spans, hooks)

    # The GPU arms are Pallas launches, which have no partitioning rule, so each shard runs them on its own
    # rows and heads. A dropping call's draw takes each shard's global rows and heads from the query's own
    # split (`on_attention_shards`'s coordinates), so it does not depend on the mesh or on the rules of the
    # other axes.
    rows = ("activation_batch",)
    return on_attention_shards(local, query, key, value, coordinates=dropout_seed is not None,
                      sinks=(sinks, ("activation_heads",)),
                      segment_ids=(segment_ids, ("activation_batch", None)), lengths=(lengths, rows),
                      starts=(starts, rows), block_groups=(block_groups, ("activation_batch", None)),
                      image_groups=(image_groups, ("activation_batch", None)),
                      key_valid=(key_valid, ("activation_batch", None)),
                      key_runs=(key_runs, ("activation_batch", None, None, None)),
                      places=(places, ("activation_batch", None)), seed=(dropout_seed, (None,)),
                      bias=(None if bias is None else jnp.broadcast_to(
                          bias, (query.shape[0], query.shape[2], 1, key.shape[1])),
                            ("activation_batch", "activation_heads", None, None)))


def weighted_values(equation, weights, value, *, precision=None) -> jax.Array:
    """The probability-value product, with the probabilities in the value's dtype.

    An fp32 softmax leaves fp32 probabilities, and `jnp.einsum` promotes to
    the wider operand, so multiplying them by a bf16 value would run both
    sides at the fp32 rate - forward and backward, since the product's
    cotangent is then fp32 too. Rounding the probabilities first is what
    `jax.nn.dot_product_attention` does on its own xla path
    (`probs.astype(key.dtype)`), so the two paths multiply alike; the softmax
    that produced them still reduced in fp32, which is where the precision
    was needed.

    `equation` is flax's value product, '...hqk,...khd->...qhd'; a CPU under
    Dew's tests sums an fp32 one in float64 and rounds once (`precision.accumulated`).
    """
    if equation != '...hqk,...khd->...qhd':
        raise ValueError(f"weighted_values takes flax's value product, got {equation!r}")
    return float32_einsum(equation, weights.astype(value.dtype), value, precision=precision)


def reference_attention(query, key, value, *, mask=None, bias=None, sinks=None, softcap=None, dtype=None,
                        precision=None, force_fp32_for_softmax=True, dropout_rate=0.0, dropout_seed=None,
                        dropout_places=None, wide_logits=False):
    """Attend over [B, S, H, D] in plain XLA ops, every score policy on one pipeline.

    It is flax's reference attention, op for op: the query scaled by
    1/sqrt(D) in its dtype, the logits, the additive `bias`, the structural
    `mask`, the softmax (in at least fp32 under `force_fp32_for_softmax`), a
    probability draw at `dropout_rate` from `dropout_seed`, each weight its own
    (at its global places where a sequence exchange holds a slice of the call,
    `dropout_places`), and the value product (`weighted_values`). Gemma 2's `softcap` squashes the
    scaled logits into (-softcap, softcap) before the bias and mask
    (modeling_gemma2.py:192-208), and `sinks` `[H]` put one value-free logit
    per head in the softmax's denominator (`sink_softmax`). Keys and values
    may carry fewer heads. Without sinks they repeat to the query's, as flax
    repeats them, which keeps flax's rounding; with sinks the query heads
    group over their key head and no key or value repeats, as the decode
    selector (`attention_kernels.decodes`) needs where it routes a sink
    layer here rather than to the fused decode kernel, which takes sinks on a
    GPU. `wide_logits` computes the logits from the query and keys widened to
    at least fp32 and scales them after the product, as DeepSeek V4 does
    (modeling_deepseek_v4.py:708-736); the value product stays in the inputs'
    dtype.
    """
    query, key, value = promote_dtype(query, key, value, dtype=dtype)
    dtype = query.dtype
    *rows, length, heads, width = query.shape
    kv_heads, keys = key.shape[-2], key.shape[-3]
    if wide_logits:
        scaled_query, scoring = query.astype(at_least_fp32(dtype)), key.astype(at_least_fp32(dtype))
    else:
        scaled_query, scoring = query / jnp.sqrt(width).astype(dtype), key
    if sinks is None:
        scoring, value = repeat_kv_heads(scoring, heads), repeat_kv_heads(value, heads)
        logits = float32_einsum('...qhd,...khd->...hqk', scaled_query, scoring, precision=precision)
    elif kv_heads == 1:
        # One key/value head (DeepSeek V4's): the query heads are a free dim beside the rows, and no
        # size-one head axis rides as a batch dim, which had XLA transpose the fp32 query for the
        # product (a V4 layer's training step on an H100, batch 4 by 2048: 2 GiB more and 5% slower).
        logits = float32_einsum('...tgd,...sd->...gts', scaled_query, scoring[..., 0, :], precision=precision)
    else:
        grouped = scaled_query.reshape(*rows, length, kv_heads, heads // kv_heads, width)
        logits = float32_einsum('...tkgd,...skd->...kgts', grouped, scoring, precision=precision).reshape(
            *rows, heads, length, keys)
    if wide_logits:
        logits = logits * width ** -0.5
    if softcap is not None:
        logits = capped_logits(logits, softcap)
    if bias is not None:
        logits = logits + bias
    if mask is not None:
        # Beside a sink a masked key is -inf, so it weighs nothing whatever the sink's logit: the sink
        # keeps every row's denominator finite. Without one the finite floor keeps a row every key
        # masks a number, as flax's does.
        logits = jnp.where(mask, logits, -jnp.inf if sinks is not None else jnp.finfo(dtype).min)
    if sinks is not None:
        if sinks.shape != (heads,):
            raise ValueError(f"sinks must have one logit per query head, got {sinks.shape}")
        wide = jnp.promote_types(logits.dtype, sinks.dtype)
        weights = sink_softmax(logits, sinks, at_least_fp32(wide) if force_fp32_for_softmax else None)
    elif force_fp32_for_softmax and dtype != jnp.float32:
        weights = jax.nn.softmax(logits.astype(at_least_fp32(dtype)))
    else:
        # A bias wider than the logits widens them; the softmax runs there and its weights round after.
        weights = jax.nn.softmax(logits).astype(dtype)
    if dropout_rate:
        if dropout_seed is None:
            raise ValueError("training attention dropout requires dropout_seed")
        # The keep bits are the draw's, shared with the kernels; the scale 1 / (1 - rate) takes the product's
        # dtype, rounded once from its double (ledger 1533: a bf16 1 / (1 - rate) biases every kept weight).
        # The weights round once, to their dtype.
        wide = at_least_fp32(dtype)
        keep = template.dropout_keep(dropout_seed, dropout_rate, weights.shape, dropout_places)
        scale = jnp.asarray(1.0 / (1.0 - dropout_rate), wide)
        weights = (weights.astype(wide) * jnp.where(keep, scale, 0)).astype(dtype)
    if sinks is None:
        return weighted_values('...hqk,...khd->...qhd', weights, value, precision=precision)
    if kv_heads == 1:
        return float32_einsum('...gts,...sd->...tgd', weights.astype(value.dtype), value[..., 0, :],
                              precision=precision)
    groups = weights.astype(value.dtype).reshape(*rows, kv_heads, heads // kv_heads, length, keys)
    return float32_einsum('...kgts,...skd->...tkgd', groups, value, precision=precision).reshape(
        *rows, length, heads, value.shape[-1])


def scaled_dot_product_attention(query, key, value, dtype=None, precision=None,
                                 force_fp32_for_softmax=True, implementation: AttentionImpl = 'auto',
                                 causal=False, sliding_window=None, mask=None, bias=None,
                                 sinks=None, softcap=None, segment_ids=None, max_segments=None,
                                 key_value_seq_lengths=None, dropout_rate=0.0,
                                 dropout_rng=None, deterministic=True, key_segment_ids=None,
                                 key_value_seq_starts=None, block_groups=None, image_groups=None,
                                 key_valid=None, key_runs=None, places=None):
    """Attend over [B, S, H, D] queries, keys and values: [B, S, H, Dv] out.

    `implementation` names the kernel (`AttentionImpl`, dispatched by
    `attention_kernel`); the parameter tree never depends on it. Keys and values
    may carry fewer heads (grouped-query attention) and the value a narrower
    head width (DeepSeek's latent attention, `widen_value_heads`), so kernel
    selection reads the query alone.

    `causal` restricts query i to keys 0..i, top-left aligned like jax's
    is_causal, and `sliding_window=w` to the w most recent of those, or
    without `causal` to the keys within w - 1 of i (`window_sides`); both read
    the row index, so a decode step against a cache passes `mask` instead.
    `bias` is additive and broadcasts to [B, H, Q, K] (T5's position table).
    `sinks` is one learned value-free logit per query head in the denominator.
    `softcap` is Gemma 2's tanh on the scaled logits. `segment_ids` `[B, S]`
    keeps each packed document to itself, 0 as padding (`document_mask`), and
    `max_segments`, static, bounds the documents a row holds
    (`Coordinates.max_segments`), which flash's packed sequences need.
    `key_segment_ids` `[B, K]` are the keys' own where they are not the
    queries (`CrossDocuments`): each query reads its document's keys, and
    under causality or a window a document's queries are its last keys, in
    order (a cache call's tokens, appended after the keys their row held).
    `key_value_seq_lengths` `[B]` ends each row's keys, which cuDNN skips; a
    mask that only ends rows early costs cuDNN a dense bias per head, so pass
    lengths for right padding. `key_value_seq_starts` `[B]`, below the
    lengths, starts each row's keys there: a windowed decode step's cache,
    which the decode kernel and the template read from the window's first block
    (`decode_attention.attend`) and every other path as the mask they mean.
    `block_groups` and `image_groups` `[B, K]` number the keys' blocks, -1
    outside any: under causality a block's query also sees its block whole,
    an image's its later keys within the window (`template.spans`).
    `key_valid` `[B, K]` marks the real keys, `key_runs` `[B, Q, R, 2]` each
    query's key intervals, whose union it may see, and `places` `[B, S]` the places a
    window counts (`counted_places`). The template reads these as its spans
    (`key_spans`); every other path takes the mask they mean (`visibility_mask`).

    A mesh sequence axis above one runs `sequence_parallel_attention`. Active dropout draws each weight
    from `dropout_rng` on the route the selector takes for it (`reference`, or `template` where named),
    at its global place on a sequence axis above one, and refuses explicit fused kernels. The
    result carries the checkpoint name 'attention_output', so a remat policy
    can save it instead of replaying the kernel (`remat_block` in dit.py).
    """
    shards = sequence_shards()
    if not 0 <= dropout_rate < 1:
        raise ValueError(f"dropout_rate must be within [0, 1), got {dropout_rate}")
    dropping = bool(dropout_rate) and not deterministic
    if dropping and dropout_rng is None:
        raise ValueError("training attention dropout requires dropout_rng")
    if key_segment_ids is not None:
        if segment_ids is None:
            raise ValueError("key_segment_ids are the keys' documents beside the queries' segment_ids")
        segment_ids = CrossDocuments(segment_ids, key_segment_ids)
    if not causal:
        block_groups = image_groups = None
    if sliding_window is None:
        places = None
    spanned = {"block_groups": block_groups, "image_groups": image_groups, "key_valid": key_valid,
               "key_runs": key_runs,
               "places": places}
    if any(x is not None for x in spanned.values()) and shards > 1:
        # Only a single device's template reads groups, key validity and places; across sequence shards the
        # call's spans are the mask they mean, its causality, window, documents, key lengths and starts among
        # them. On one device a route other than the template's takes them so (`_attention_kernel`).
        mask = visibility_mask(query.shape[-3], key.shape[-3], causal, sliding_window, mask, segment_ids,
                               key_value_seq_lengths, key_value_seq_starts, **spanned)
        causal, sliding_window, segment_ids, block_groups, image_groups = False, None, None, None, None
        key_value_seq_lengths = key_value_seq_starts = key_valid = key_runs = places = None
        implementation = kernel_for_materialized_mask(
            implementation, query, key, dtype=dtype, precision=precision,
            force_fp32_for_softmax=force_fp32_for_softmax)
    if (isinstance(segment_ids, CrossDocuments) and (causal or sliding_window is not None)
            and shards > 1):
        # Only a single device's kernels read documents aligned to their last keys; elsewhere they
        # are the mask they mean.
        aligned = cross_document_mask(segment_ids, causal=causal, sliding_window=sliding_window)
        mask = aligned if mask is None else jnp.logical_and(aligned, mask)
        causal, sliding_window, segment_ids = False, None, None
    if key_value_seq_starts is not None and shards > 1:
        # Only a single device's decode kernel reads starts; elsewhere they are the mask they mean.
        mask = with_key_starts(mask, key_value_seq_starts, key.shape[-3])
        key_value_seq_starts = None
    words = template.dropout_seed(dropout_rng) if dropping else None
    if shards > 1 and segment_ids is not None:
        # The exchanges split an explicit mask along the sequence and have no
        # split for the ids, so the documents travel as the mask they stand
        # for, with causality and the window folded in beside them.
        mask = combined_attention_mask(query.shape[-3], key.shape[-3], causal, sliding_window,
                                       with_documents(mask, segment_ids))
        causal, sliding_window, segment_ids = False, None, None
        implementation = kernel_for_materialized_mask(
            implementation, query, key, dtype=dtype, precision=precision,
            force_fp32_for_softmax=force_fp32_for_softmax)
    kernel = functools.partial(
        attention_kernel, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax, implementation=implementation,
        softcap=softcap)
    if shards > 1:
        out = sequence_parallel_attention(
            kernel, query, key, value, shards, causal=causal,
            sliding_window=sliding_window, mask=mask, bias=bias, sinks=sinks,
            key_value_seq_lengths=key_value_seq_lengths,
            dropout_rate=dropout_rate if dropping else 0.0, dropout_seed=words)
    else:
        out = kernel(query, key, value, causal=causal, sliding_window=sliding_window,
                     mask=mask, bias=bias, sinks=sinks, segment_ids=segment_ids,
                     max_segments=max_segments, key_value_seq_lengths=key_value_seq_lengths,
                     key_value_seq_starts=key_value_seq_starts, block_groups=block_groups,
                     image_groups=image_groups, key_valid=key_valid, key_runs=key_runs, places=places,
                     dropout_rate=dropout_rate if dropping else 0.0, dropout_seed=words)
    return checkpoint_name(out, 'attention_output')


def _read_nothing(out: jax.Array, live: jax.Array | None) -> jax.Array:
    """`out` with each row that sees no key (`visible_rows`) exactly zero. Its
    cotangent is zero there too, so no kernel's backward reads those rows."""
    return out if live is None else jnp.where(live, out, jnp.zeros((), out.dtype))


def _fused_attention(query, key, value, bias, mask, causal, sliding_window, implementation, *,
                     softcap, sinks, segment_ids, key_value_seq_lengths, max_segments=None,
                     key_value_seq_starts=None, block_groups=None, image_groups=None, key_valid=None,
                     key_runs=None,
                     places=None, dropout_rate=0.0, dropout_seed=None, dropout_places=None):
    """Run the kernel `implementation` names, at the query's head width: the executor of what the
    attention selector chose (`resolve_implementation`, `_attention_kernel`), which held the call to
    the kernel's requirements (`attention_kernels.ATTENTION`) before it reaches here.

    A narrower value rides in padded and its columns come back out, at no runtime branch, except that the
    template takes its own value width. `key_value_seq_lengths` reaches cudnn, xla and the template as
    lengths and 'tpu' as the mask it means; `key_value_seq_starts` reaches the template and the decode
    kernel alone, and the groups, key validity and places the template alone (`_attention_kernel` folds them
    into the mask elsewhere).
    """
    v_head_dim = value.shape[-1]
    if v_head_dim != query.shape[-1] and implementation != 'template':
        value = widen_value_heads(query, value)

    if implementation == 'cudnn':
        out = cudnn_attention(query, key, value, bias, mask, causal, sliding_window,
                              key_value_seq_lengths)
    elif implementation == 'triton':
        out = triton_attention(query, key, value, causal)
    elif implementation == 'flash':
        out = flash_attention(query, key, value, causal, sliding_window, sinks, segment_ids, max_segments)
    elif implementation == 'decode':
        starts = key_value_seq_starts

        def decoded(query, key, value, *sunk):
            # A Pallas call has no partitioning rule: each shard decodes its own rows and heads.
            def attend(query, key, value, lengths, starts=None, sinks=None):
                return decode_attention.attend(query, key, value, lengths, starts, softcap=softcap,
                                               sinks=sinks)

            rows = ("activation_batch",)
            return on_attention_shards(attend, query, key, value, lengths=(key_value_seq_lengths, rows),
                              starts=(starts, rows), sinks=(sunk[0] if sunk else None, ("activation_heads",)))

        def folded(query, key, value, *sunk):
            visible = with_key_starts(with_key_lengths(None, key_value_seq_lengths, key.shape[-3]), starts,
                                      key.shape[-3])
            if not sunk and softcap is None:
                return folded_attention(query, key, value, visible)
            # The folded form has neither; the reference path computes both over the same keys.
            return reference_attention(query, key, value, mask=visible, sinks=sunk[0] if sunk else None,
                                       softcap=softcap)

        # The kernel has no derivative, so both modes differentiate the attention it stands in for, the
        # sinks among its operands, at the kernel's arithmetic: IEEE products for fp32 operands
        # (`decode_attention`), its 16-bit products summed in fp32 otherwise.
        out = forward_differentiable(decoded, folded, query, key, value, *([] if sinks is None else [sinks]),
                                     precision="highest" if query.dtype == jnp.float32 else None)
    elif implementation == 'xla' and folds(query, key, bias, mask, causal, sliding_window):
        # Key lengths and the prefix mask they mean take one path here, so where no lengths-only
        # kernel takes the call the two forms agree bitwise.
        out = folded_attention(query, key, value, mask if key_value_seq_lengths is None
                               else with_key_lengths(mask, key_value_seq_lengths, key.shape[-3]))
    elif implementation == 'xla':
        # A window of (l, r) keeps the l keys before the query and the r after
        # it on both the xla and the cudnn path, the sides `window_sides` counts.
        # jax.nn's products take the precision their dtype does (`float32_products`) through the
        # matmul config, which it reads for float32 operands.
        with configured_precision(float32_products(query.dtype)):
            out = jax.nn.dot_product_attention(
                query, key, value, bias=bias, mask=mask, is_causal=causal,
                key_value_seq_lengths=key_value_seq_lengths,
                local_window_size=None if sliding_window is None else window_sides(causal, sliding_window),
                implementation='xla')
    elif implementation == 'template':
        out = template_attention(query, key, value, causal, sliding_window, sinks=sinks, softcap=softcap,
                                 segment_ids=segment_ids, lengths=key_value_seq_lengths,
                                 starts=key_value_seq_starts, block_groups=block_groups,
                                 image_groups=image_groups, key_valid=key_valid, key_runs=key_runs,
                                 places=places, dropout_rate=dropout_rate, dropout_seed=dropout_seed,
                                 dropout_places=dropout_places, bias=bias)
    else:  # 'tpu'
        if key_value_seq_lengths is not None:
            mask = with_key_lengths(mask, key_value_seq_lengths, key.shape[-3])
        out = tpu_attention(query, key, value, bias, mask, causal, sliding_window,
                            softcap=softcap, sinks=sinks, segment_ids=segment_ids,
                            interpret=generation.backend() != 'tpu')
    return out if v_head_dim == out.shape[-1] else out[..., :v_head_dim]


FORWARD_MODE = contextvars.ContextVar("forward_mode_attention", default=False)


@contextlib.contextmanager
def forward_mode_attention():
    """Trace the attention calls inside for `jax.jvp`.

    cuDNN's and the TPU's fused kernels are reverse-mode only (`jax.custom_vjp`).
    Inside this context each such call keeps the fused kernel for its value and
    takes its tangent from the reference path's JVP, which materializes the
    `[B, H, Q, K]` probabilities once. Consistency models (sCM, rCM) run their
    JVP inside it and train outside it, on the fused backward.
    """
    token = FORWARD_MODE.set(True)
    try:
        yield
    finally:
        FORWARD_MODE.reset(token)


def attention_kernel(query, key, value, dtype=None, precision=None,
                     force_fp32_for_softmax=True, implementation: AttentionImpl = 'auto',
                     causal=False, sliding_window=None, mask=None, bias=None, sinks=None,
                     softcap=None, segment_ids=None, max_segments=None, key_value_seq_lengths=None,
                     key_value_seq_starts=None, block_groups=None, image_groups=None, key_valid=None,
                     key_runs=None,
                     places=None, dropout_rate=0.0, dropout_seed=None, dropout_places=None):
    """Dispatch one whole-sequence attention call to the kernel it names,
    differentiable in forward mode under `forward_mode_attention`.

    'auto' resolves here, once, against this call's shapes and backend, and
    `_attention_kernel` runs the call on the kernel it resolved to. A query
    row that sees no key comes back zero whichever kernel ran
    (`visible_rows`), inside a sequence exchange too.
    """
    if sliding_window is not None and sliding_window < 1:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")
    if (key_value_seq_lengths is not None and query.shape[-3] == 1 and not causal
            and sliding_window is None and mask is None and bias is None and sinks is None
            and segment_ids is None and query.shape[-2] > key.shape[-2]
            and query.shape[-2] % key.shape[-2] == 0 and dropout_seed is None
            and key_valid is None and key_runs is None and block_groups is None and image_groups is None
            and places is None):
        # The grouped call carries key lengths and starts alone, so a call masking its keys otherwise
        # (key validity, runs, block or image groups, counted places) takes the whole path.
        # The key heads repeated as each shard would hold them (`on_attention_shards`), so the tensor axes
        # split the groups as they split the query heads.
        key, value = split_with_queries(key, value, tensor_shards(query))
        return decode_attention.grouped_query(functools.partial(
            attention_kernel, dtype=dtype, precision=precision, force_fp32_for_softmax=force_fp32_for_softmax,
            implementation=implementation, softcap=softcap, key_value_seq_lengths=key_value_seq_lengths,
            key_value_seq_starts=key_value_seq_starts), query, key, value)
    lengths = (None if key_value_seq_lengths is None
               else jnp.asarray(key_value_seq_lengths, jnp.int32))
    starts = None if key_value_seq_starts is None else jnp.asarray(key_value_seq_starts, jnp.int32)
    masked = mask if lengths is None else with_key_lengths(mask, lengths, key.shape[-3])
    masked = with_key_starts(masked, starts, key.shape[-3])
    resolved = resolve_implementation(
        implementation, query, key, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax, softcap=softcap, sinks=sinks,
        causal=causal, sliding_window=sliding_window, mask=mask, bias=bias,
        segment_ids=segment_ids, max_segments=max_segments, key_value_seq_lengths=lengths,
        key_value_seq_starts=starts, block_groups=block_groups, image_groups=image_groups,
        key_valid=key_valid, key_runs=key_runs, places=places, dropout=dropout_seed is not None)
    call = functools.partial(
        _attention_kernel, dtype=dtype, precision=precision, force_fp32_for_softmax=force_fp32_for_softmax,
        causal=causal, sliding_window=sliding_window, mask=mask, masked=masked, softcap=softcap,
        segment_ids=segment_ids, max_segments=max_segments, lengths=lengths, starts=starts,
        block_groups=block_groups, image_groups=image_groups, key_valid=key_valid, key_runs=key_runs,
        places=places, dropout_rate=dropout_rate, dropout_seed=dropout_seed, dropout_places=dropout_places)
    live = visible_rows(query.shape[-3], key.shape[-3], causal, sliding_window, mask, segment_ids, lengths,
                        starts, block_groups=block_groups, image_groups=image_groups, key_valid=key_valid,
                        key_runs=key_runs,
                        places=places)
    if not FORWARD_MODE.get() or resolved not in ('cudnn', 'triton', 'flash', 'tpu', 'template'):
        return _read_nothing(call(query, key, value, bias=bias, sinks=sinks, implementation=resolved), live)

    def fused(query, key, value, bias, sinks):
        return call(query, key, value, bias=bias, sinks=sinks, implementation=resolved)

    def reference(query, key, value, bias, sinks):
        return call(query, key, value, bias=bias, sinks=sinks, implementation='reference')

    return _read_nothing(forward_differentiable(fused, reference, query, key, value, bias, sinks), live)


def forward_differentiable(value, tangent, *arrays, precision: str | None = None):
    """`value(*arrays)`, whose JVP is `tangent`'s: a kernel that defines only
    its reverse-mode derivative, or none, takes its derivatives from a
    reference computing the same function, reverse mode as the transposed JVP.
    `precision` is the kernel's own arithmetic, which the reference's products
    take too (`jax.default_matmul_precision`), so a kernel of IEEE fp32 products
    is not differentiated by TF32 ones; None leaves the products at the
    caller's. Only the arrays a derivative moves are the reference's tangents: a known zero stays
    out of its derivative, where a product's own derivative (`dew.nn.precision.accumulated`'s on a
    CPU) takes its moving operands alone."""
    from jax.custom_derivatives import SymbolicZero

    @jax.custom_jvp
    def attend(*arrays):
        return value(*arrays)

    def rule(primals, tangents):
        live = [index for index, direction in enumerate(tangents) if not isinstance(direction, SymbolicZero)]
        output = attend(*primals)
        if not live:
            return output, jax.tree.map(jnp.zeros_like, output)

        def moved(*moving):
            arrays = list(primals)
            for index, array in zip(live, moving, strict=True):
                arrays[index] = array
            return tangent(*arrays)

        with contextlib.nullcontext() if precision is None else jax.default_matmul_precision(precision):
            differentiated = jax.jvp(moved, tuple(primals[index] for index in live),
                                     tuple(tangents[index] for index in live))[1]
        return output, differentiated

    attend.defjvp(rule, symbolic_zeros=True)
    return attend(*arrays)


def _attention_kernel(query, key, value, *, bias, sinks, implementation, dtype, precision,
                      force_fp32_for_softmax, causal, sliding_window, mask, masked, softcap,
                      segment_ids, max_segments, lengths, starts, block_groups=None, image_groups=None,
                      key_valid=None, key_runs=None, places=None, dropout_rate=0.0, dropout_seed=None,
                      dropout_places=None):
    """Run one whole-sequence attention call on the kernel `attention_kernel` resolved.

    `masked` is `mask` with the key `lengths` and `starts` in it. Splash takes sinks, a
    softcap and packed segment ids itself, flash takes the ids as its packed sequences given
    `max_segments`, and the template reads the ids (`CrossDocuments` included) as its spans;
    elsewhere segment ids become the document mask, and sinks and a softcap run
    `reference_attention`, since `jax.nn.dot_product_attention` has neither (a fused route this
    host runs as xla included, `_hosted`). The rest goes to `_fused_attention`. Key lengths reach
    cudnn, xla and the template as they are and every other path as the mask they mean, which
    splash cannot describe, so 'auto' never picks 'tpu' for them. Key starts reach the decode
    kernel and the template as they are and every other path as the mask they mean. Block and image
    groups, key validity and counted places reach the template alone; every other path takes the call's
    spans as the mask they mean (`visibility_mask`).
    """
    spanned = (block_groups, image_groups, key_valid, key_runs, places)
    if any(x is not None for x in spanned) and implementation != 'template':
        mask = masked = visibility_mask(query.shape[-3], key.shape[-3], causal, sliding_window, mask,
                                        segment_ids, lengths, starts, block_groups=block_groups,
                                        image_groups=image_groups, key_valid=key_valid, key_runs=key_runs,
                                        places=places)
        causal, sliding_window, segment_ids, lengths, starts = False, None, None, None, None
        block_groups = image_groups = key_valid = key_runs = places = None
    # Another sequence's documents under causality or a window align each document's queries with
    # its last keys (`CrossDocuments`), which splash does not; elsewhere the mask carries them.
    aligned = isinstance(segment_ids, CrossDocuments) and (causal or sliding_window is not None)
    packed = (implementation == 'template' or (implementation == 'tpu' and not aligned)
              or (implementation == 'flash' and max_segments is not None))
    if segment_ids is not None and not packed:
        if aligned:
            inside = cross_document_mask(segment_ids, causal=causal, sliding_window=sliding_window)
            mask, masked = (inside if m is None else jnp.logical_and(m, inside) for m in (mask, masked))
            causal, sliding_window = False, None
        else:
            mask = with_documents(mask, segment_ids)
            masked = with_documents(masked, segment_ids)
        # cuDNN takes the documents as their mask, an additive bias (`kernel_for_materialized_mask`):
        # a cache's aligned ones as it took a cache's mask, a call's own where cuDNN is named.
    if implementation == 'reference' or (implementation == 'xla' and (
            sinks is not None or softcap is not None or dropout_seed is not None)):
        mask = combined_attention_mask(query.shape[-3], key.shape[-3], causal, sliding_window, masked)
        # The accurate form's products are IEEE fp32 (`attention_route.FORMS`), never a GPU's TF32.
        if generation.chosen_kernel("attention_form", "fast") == "accurate":
            precision = jax.lax.Precision.HIGHEST
        return reference_attention(query, key, value, mask=mask, bias=bias, sinks=sinks, softcap=softcap,
                                   dtype=dtype, precision=precision,
                                   force_fp32_for_softmax=force_fp32_for_softmax,
                                   dropout_rate=dropout_rate, dropout_seed=dropout_seed,
                                   dropout_places=dropout_places)

    # The decode kernel and the template read key starts as their own spans; elsewhere they are the mask.
    if implementation not in ('decode', 'template'):
        mask = with_key_starts(mask, starts, key.shape[-3])
        starts = None
    return _fused_attention(query, key, value, bias, mask, causal, sliding_window,
                            implementation, softcap=softcap, sinks=sinks,
                            segment_ids=segment_ids, max_segments=max_segments,
                            key_value_seq_lengths=lengths, key_value_seq_starts=starts,
                            block_groups=block_groups, image_groups=image_groups, key_valid=key_valid,
                            key_runs=key_runs,
                            places=places, dropout_rate=dropout_rate, dropout_seed=dropout_seed,
                            dropout_places=dropout_places)


def reference_only(query, dtype, precision, force_fp32_for_softmax) -> bool:
    """Whether a call asks for arithmetic only the reference path performs.

    The three are what every fused kernel's requirements refuse
    (`attention_kernels`' arithmetic requirement): a matmul precision other
    than DEFAULT (a higher one, a dot algorithm or a preset of one), a
    softmax outside fp32, and a compute dtype other than the inputs'.
    """
    return (not asks_default_precision(precision)
            or not force_fp32_for_softmax
            or (dtype is not None and jnp.dtype(dtype) != query.dtype))


def _xla_kernel_narrows(query) -> bool:
    """Whether jax.nn's xla attention would compute this call below its
    query's precision, so 'auto' and 'xla' take the reference path, which
    computes in the query's dtype (`dew.nn.precision.at_least_fp32`).

    It rounds a float64 query's softmax to float32
    (`_dot_product_attention_core`: "Softmax and it is always carried out in
    fp32"), and names the BF16_BF16_F32 algorithm for a bf16 one, which a GPU
    older than sm80 rejects at run time, past jax's own fallback."""
    return query.dtype == jnp.float64 or (
        query.dtype == jnp.bfloat16 and generation.backend() == 'gpu' and not bf16_dot_runs())


def _xla_kernel_widens_less() -> bool:
    """Whether jax.nn's xla attention would run this call below the arithmetic its process asks of a CPU,
    so 'auto' and 'xla' take the reference path: a CPU under Dew's test setting (`WIDE_CPU_ARITHMETIC`),
    which sums fp32 products in float64 (`precision.accumulated`) where jax's sums them in fp32, and whose
    reference path differentiates the softmax by its own derivative, exactly zero for a query that reads
    one key, where jax's takes rowsum(dO * O) and leaves it a rounding. A process without the setting runs
    jax's kernel on a CPU as on any platform."""
    return wide_cpu_arithmetic() and generation.backend() == 'cpu'


def attention_case(query, key, *, dtype=None, precision=None, force_fp32_for_softmax=True, softcap=None,
                   sinks=None, causal=False, sliding_window=None, mask=None, bias=None, segment_ids=None,
                   max_segments=None, key_value_seq_lengths=None, key_value_seq_starts=None,
                   block_groups=None, image_groups=None, key_valid=None, key_runs=None, places=None,
                   dropout: bool = False, named: str = 'auto') -> AttentionCase:
    """The `AttentionCase` of a call `scaled_dot_product_attention` takes under the `named`
    `AttentionImpl`, which decides whether splash might run it."""
    cross = isinstance(segment_ids, CrossDocuments)
    q_len, kv_len = query.shape[-3], key.shape[-3]
    case = AttentionCase(
        queries=q_len, keys=kv_len, heads=query.shape[-2], kv_heads=key.shape[-2],
        head_dim=query.shape[-1], dtype=jnp.dtype(query.dtype), key_dtype=jnp.dtype(key.dtype),
        causal=bool(causal), window=sliding_window,
        documents=None if segment_ids is None else 'cross' if cross else 'own',
        aligned=cross and (causal or sliding_window is not None), max_segments=max_segments,
        lengths=key_value_seq_lengths is not None, starts=key_value_seq_starts is not None,
        blocks=bool(causal) and (block_groups is not None or image_groups is not None),
        spans_only=((bool(causal) and (block_groups is not None or image_groups is not None))
                    or key_valid is not None or key_runs is not None
                    or (places is not None and sliding_window is not None)),
        mask=mask is not None, bias=bias is not None,
        key_bias=(bias is not None and jnp.ndim(bias) == 4 and jnp.shape(bias)[-2] == 1
                  and jnp.dtype(bias.dtype).itemsize <= 4),
        sinks=sinks is not None, softcap=softcap,
        arithmetic=reference_only(query, dtype, precision, force_fp32_for_softmax),
        narrows=_xla_kernel_narrows(query) or _xla_kernel_widens_less(),
        folds=folds(query, key, bias, mask, causal, sliding_window),
        splash_mask=False, shards=sequence_shards(), dropout=dropout,
        form=generation.chosen_kernel("attention_form", "fast"))
    # Splash's descriptor reads a concrete mask on the host, so it is built only where splash may run,
    # under a name that may choose it on a TPU (`TPU_BACKEND`); its tiling and every other requirement
    # read the case (`TPU_SUPPORTS`).
    if named in ('auto', 'tpu') and TPU_BACKEND.holds(case):
        described = splash_mask_descriptor(q_len, kv_len, shard_heads(query), causal, sliding_window, mask)
        case = dataclasses.replace(case, splash_mask=described is not None)
    return case


def resolve_implementation(implementation: AttentionImpl, query, key, *, dtype=None, precision=None,
                           force_fp32_for_softmax=True, softcap=None, sinks=None, causal=False,
                           sliding_window=None, mask=None, bias=None, segment_ids=None, max_segments=None,
                           key_value_seq_lengths=None, key_value_seq_starts=None, block_groups=None,
                           image_groups=None, key_valid=None, key_runs=None, places=None,
                           dropout: bool = False) -> str:
    """The kernel an `AttentionImpl` names for this call on this host, as the attention selector
    reports it (`select`, over `attention_case`): 'auto' takes the kernel the target's row measured
    fastest for the call where it takes it, else the first that does; a named kernel is forced, and
    raises where it does not take the call (xla handing what it cannot honour to the reference path).
    A call of few positions over key lengths is the decode selector's (`DECODE`), whose decode kernel
    runs as 'decode' (`_fused_attention`); a named xla lowers it there too, as jax.nn's own lowering
    of a decode step, unless `generation.chosen(decode='xla')` names the folded attention. What runs
    is what this host runs (`_hosted`): xla for a CUDA kernel a step traced for an assumed GPU on a
    CPU reports."""
    attention_implementation(implementation)
    case = attention_case(query, key, dtype=dtype, precision=precision,
                          force_fp32_for_softmax=force_fp32_for_softmax, softcap=softcap, sinks=sinks,
                          causal=causal, sliding_window=sliding_window, mask=mask, bias=bias,
                          segment_ids=segment_ids, max_segments=max_segments,
                          key_value_seq_lengths=key_value_seq_lengths,
                          key_value_seq_starts=key_value_seq_starts, block_groups=block_groups,
                          image_groups=image_groups, key_valid=key_valid, key_runs=key_runs, places=places,
                          dropout=dropout, named=implementation)
    if not (decodes(case) and implementation in ('auto', 'cudnn', 'xla')):
        return select(ATTENTION, case, None if implementation == 'auto' else implementation).ran
    named = {'auto': None, 'cudnn': 'cudnn'}.get(implementation)
    if implementation == 'xla':
        named = 'xla' if forced(DECODE.choice) == 'xla' or DECODE.kernel('decode').failed(case) else 'decode'
    return select(DECODE, case, named).ran


def kernel_for_materialized_mask(implementation: AttentionImpl, query, key, *, dtype=None, precision=None,
                                 force_fp32_for_softmax=True) -> AttentionImpl:
    """Send a call that built an explicit mask to the xla kernel.

    cuDNN has no mask argument: jax hands it a bool mask as an additive bias of
    -2**41 in the compute dtype (`combine_bias_and_mask` in
    jax/_src/cudnn/fused_attention_stablehlo.py), and `check_is_flash_attention`
    then refuses an odd length while training. The xla kernel masks by
    exclusion with the same fp32 softmax, at 83.6 ms and 5.80 GiB a step against
    cuDNN's fixed window at 75.8 ms and 4.99 GiB
    (docs/concepts/language_models.md). A call the xla kernel hands off goes to
    the reference path as its requirements say (`attention_kernels.XLA_SUPPORTS`),
    an explicit 'cudnn' among them, which runs here as xla.
    """
    masked = 'xla' if implementation in ('auto', 'cudnn') else implementation
    if masked != 'xla':
        return masked
    case = dataclasses.replace(attention_case(query, key, dtype=dtype, precision=precision,
                                              force_fp32_for_softmax=force_fp32_for_softmax, named='xla'),
                               mask=True, folds=False)
    return attention_implementation(selected(ATTENTION, case, 'xla').name)


def cudnn_takes_mask(implementation: AttentionImpl, query, *, softcap=None, sinks: bool = False) -> bool:
    """Whether cuDNN may take a call's materialized mask as its additive bias:
    the call names cudnn or chooses ('auto'), cuDNN runs it (`cudnn_runs`,
    which also keeps the deterministic-ops lane off it) and it has no sinks,
    which cuDNN cannot honor. A caller adds why its mask suits cuDNN's bias,
    where `kernel_for_materialized_mask` sends the rest to xla: a forward-only
    call, or a mask a window bounds."""
    return implementation in ('auto', 'cudnn') and not sinks and cudnn_runs(query, softcap)
