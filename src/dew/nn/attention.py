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
import math
from collections.abc import Callable
from typing import Literal, NamedTuple, Protocol, get_args, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import tokamax
from flax.linen.dtypes import promote_dtype
from jax._src import core as jax_core
from jax.ad_checkpoint import checkpoint_name
from jax.extend import core
from jax.interpreters import mlir
from jax.typing import DTypeLike

from dew.nn.derivatives import KernelGlue, contracted, declared_tangent, with_tangents_formed_at
from dew.nn.kernels.templates.hooks import Hook, Side
from dew.nn.programs import holes, own_preset
from dew.nn.residuals import Residual

from .attention_kernels import (
    ATTENTION,
    DECODE,
    SPLASH_LANES,
    TPU_BACKEND,
    AttentionCase,
    cudnn_packs,
    decodes,
    flash_target,
    primal_alone,
    reverse_only,
    unlowered,
)
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
    check_rows,
    four_dimensional,
    launched,
    on_attention_shards,
    pad_rows,
    sequence_parallel_attention,
    shard_heads,
    split_with_queries,
    tensor_shards,
    visibility_refused,
)
from .attention_sinks import sink_column, sink_softmax
from .attention_tpu import splash_mask_descriptor, tpu_attention
from .decode_geometry import folded_attention, folds, grouped_query
from .documents import (
    CrossDocuments,
    cross_document_mask,
    document_ids,
    document_offsets,
    own_runs,
    packed_runs,
    query_order,
    with_documents,
)
from .inputs import cu_seqlens
from .kernels import decode_attention, generation
from .kernels.generation import Route, bf16_dot_runs
from .kernels.selection import Unlowerable, forced, select, selected
from .kernels.templates import attention as template, band
from .precision import (
    EXACT_FLOAT32,
    asks_default_precision,
    at_least_fp32,
    at_precision,
    capped_logits,
    configurable_precision,
    configured_precision,
    exact_products,
    float32_einsum,
)
from .sharding import SCORES, sequence_shards
from .visibility import window_sides

AttentionImpl = Literal["auto", "reference", "xla", "cudnn", "triton", "flash", "tpu", "template",
                        "inference", "flash_inference"]
"""Names which kernel an attention call runs.

A run names it once for every attention layer it traces (`generation.chosen(attention=...)`, a run's
`Diagnostics.kernels`, `chosen_attention`); a call names its own (`scaled_dot_product_attention`'s
`implementation`). No model holds one (row 11938).

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
'inference' is the template for a call no derivative is taken of and xla's attention for one
differentiated (`attention_kernels.INFERENCE`): every kernel the attention selector holds is named here, so
a run naming any of them (`dew.kernels.use(attention="inference")`) traces (the API02 review).
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


def chosen_attention() -> AttentionImpl:
    """The attention kernel the run's context names for every attention layer it traces
    (`generation.chosen(attention=...)`), 'auto' where none does: what a layer resolves its calls under
    (`resolve_implementation`), read at its trace."""
    return attention_implementation(forced(generation.Choice.ATTENTION) or "auto")


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class KeyColumns:
    """A value that is its keys' leading `width` columns (`key[..., :width]`), passed as
    `scaled_dot_product_attention`'s `value`: an absorbed latent attention's (DeepSeek's key its latent and
    rotary columns, its value the latent). The template's Mosaic forward reads it from the key's slot, the
    keys loaded once (`attention_mosaic.MosaicForward.in_key`); every other path takes the slice. A static
    pytree node, its width no leaf, so a jitted call's value of it stays a width, never a tracer."""

    width: int


class SeparableBias(NamedTuple):
    """An additive bias of a query's term and a key's, `rows[b, h, q] + columns[b, h, k]`, `rows`
    `[B or 1, H or 1, Q]` and `columns` `[B or 1, H or 1, K]`: FoX's cumulative log forget gate at the query
    less the key's (`forgetting_attention`). The template reads the pair as a query and a key score
    parameter, the logits adding their sum (`template.Biased`), so a logit takes their difference and not
    each term's magnitude; every other route takes it written out (`dense`)."""

    rows: jax.Array
    columns: jax.Array

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """The `[B or 1, H or 1, Q, K]` bias it writes out (`dense`)."""
        rows, columns = jnp.shape(self.rows), jnp.shape(self.columns)
        return max(rows[0], columns[0]), max(rows[1], columns[1]), rows[2], columns[2]

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.result_type(self.rows, self.columns)

    def dense(self) -> jax.Array:
        """The bias written out, `[B or 1, H or 1, Q, K]`."""
        return self.rows[..., :, None] + self.columns[..., None, :]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BandBias:
    """An additive bias held within a band of keys about each query, `values` `[B or 1, H or 1, Q,
    band.band_width(extent)]`: query `q`'s column `c` the bias of key `band.band_origins(offset, Q,
    extent)[q] + c`, every key outside its band biased by nothing. Every key within `extent - 1` before a
    query's place through its place (`offset`, `[]` or `[B]` int32: a query's place less its index, the key
    it is at; 0 from the first key) lies in its band: an input-conditioned relative bias (Inkling 2's,
    `attention_terms.relative_band`) at `Q band_width` entries a head, where its `[B, H, Q, K]` writing out
    holds 96 GiB at 32768 keys. Every route takes it written out (`dense`)."""

    values: jax.Array
    offset: jax.Array
    extent: int = dataclasses.field(metadata={"static": True})

    def __post_init__(self):
        # A tree's rebuild (`jax.tree.map`'s, a transform's) passes leaves of any kind: an array's checked.
        if not hasattr(self.values, "shape"):
            return
        if self.extent < 1 or self.values.shape[-1:] != (band.band_width(self.extent),):
            raise ValueError(f"a band of extent {self.extent} holds {band.band_width(self.extent)} "
                             f"columns a query, got values {jnp.shape(self.values)}")

    @property
    def shape(self) -> tuple[int, ...]:
        """Its values' shape, `[B or 1, H or 1, Q, band_width(extent)]`."""
        return self.values.shape

    @property
    def dtype(self):
        """Its values' dtype."""
        return self.values.dtype

    def dense(self, keys: int) -> jax.Array:
        """The bias written out over `keys`, `[B or 1, H or 1, Q, K]` (`band.band_dense`)."""
        return band.band_dense(self.values, self.offset, self.extent, keys)


def _least_bias(bias) -> jax.Array:
    """The least logit `bias` adds, fp32: a band's least value, or zero past its band."""
    if isinstance(bias, BandBias):
        return jnp.minimum(jnp.min(bias.values.astype(jnp.float32)), 0.0)
    return jnp.min(jnp.asarray(dense_bias(bias), jnp.float32))


@runtime_checkable
class RouteBias(Protocol):
    """A bias that states both forms a route may take: its band (`band`, a `BandBias` the template reads a
    tile at a time) and its writing out over the call's keys (`dense`), the route taking the form it reads
    (`scaled_dot_product_attention`: the band on the template outside forward mode, written out everywhere
    else). The resolution reads its band's `shape` and `dtype` alone, as a `BandBias`'s, so a route that
    writes the bias out traces no band (row 13503)."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> jnp.dtype: ...

    def band(self) -> BandBias: ...

    def dense(self, keys: int) -> jax.Array: ...


def dense_bias(bias, keys: int | None = None):
    """`bias` as an array a route adds to its logits: a `SeparableBias` written out (`SeparableBias.dense`),
    a `BandBias` or a `RouteBias` over `keys` (their `dense`), any other as it is."""
    if isinstance(bias, BandBias | RouteBias):
        if keys is None:
            raise ValueError("a band bias is written out over the call's keys, which its caller states")
        return bias.dense(keys)
    return bias.dense() if isinstance(bias, SeparableBias) else bias


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
            "Use the reference implementation (dew.kernels.use(attention='reference')).")
    return jnp.pad(value, ((0, 0),) * (value.ndim - 1) + ((0, width - v_width),))


def cudnn_attention(query, key, value, bias, mask, causal, sliding_window,
                    key_value_seq_lengths=None, segment_ids=None, max_segments: int | None = None):
    """Run jax's cudnn flash attention over any sequence length.

    cudnn has no backward pass for an odd query or key length (77 CLIP text
    tokens, most text-plus-image sequences), so one zero row pads it even. The
    padded query row is sliced off and the padded key hidden by the kernel's
    key_value_seq_lengths, which never reach past the real keys.
    tests/test_kernels.py pins the equality with the xla path. On a mesh it
    runs per shard (`on_attention_shards`). Packed rows, a call's own
    `segment_ids` `[B, S]` with at most `max_segments` documents a row and
    nothing else hiding keys (`attention_kernels.cudnn_packs`), run as cuDNN's packed sequences
    (`packed_runs`: each document, and each row's padding after them, a
    sequence), as FlashAttention runs them, the padded row in none.
    """
    if segment_ids is not None:
        assert max_segments is not None, "cuDNN packs a call's documents bounded alone (`cudnn_packs`)"
        return _cudnn_packed(query, key, value, causal, sliding_window, segment_ids, max_segments)
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

    local = functools.partial(_jax_nn, causal=causal, sliding_window=sliding_window, implementation='cudnn')
    out = on_attention_shards(local, query, key, value,
                    bias=(None if bias is None else four_dimensional(bias), SCORES),
                    mask=(None if mask is None else four_dimensional(mask), SCORES),
                    key_value_seq_lengths=(kv_lengths, ("activation_batch",)))
    return out[:, :q_len] if q_pad else out


def _cudnn_packed(query, key, value, causal: bool, sliding_window, segment_ids, max_segments: int):
    """`cudnn_attention` of packed rows (`packed_runs`) through jax's cuDNN attention's packed layout (its
    ragged offsets, `fused_attention_stablehlo.dot_product_attention`), at jax.nn's scale and masks: a
    sequence's padding ends it, causality and a window behind the query within it."""
    from jax._src.cudnn import fused_attention_stablehlo as cudnn

    length = query.shape[-3]
    offsets, lengths = packed_runs(segment_ids, max_segments + 1)
    pad = length % 2
    if pad:
        query, key, value = (pad_rows(x, pad) for x in (query, key, value))
    # A window behind the query alone reaches cuDNN (`attention_kernels.CUDNN_SUPPORTS`).
    behind = None if sliding_window is None else window_sides(causal, sliding_window)[0] + 1

    def local(query, key, value, offsets, lengths):
        out = cudnn.dot_product_attention(
            query, key, value, None, None, lengths, lengths, offsets, offsets,
            scale=1.0 / math.sqrt(query.shape[-1]),
            mask_type=cudnn.MaskType.PADDING_CAUSAL if causal else cudnn.MaskType.PADDING,
            sliding_window_length=behind)
        assert isinstance(out, jax.Array), "its FP8 path alone returns the amaxes beside the output"
        return out

    out = on_attention_shards(local, query, key, value, offsets=(offsets, ("activation_batch", None)),
                              lengths=(lengths, ("activation_batch", None)))
    return out[:, :length] if pad else out


def _jax_nn(query, key, value, bias=None, mask=None, key_value_seq_lengths=None, *, causal: bool,
            sliding_window, implementation: Literal["xla", "cudnn"]):
    """jax.nn's attention of the call on its `implementation`, 'xla' or 'cudnn': a window of (l, r) keeps
    the l keys before the query and the r after it, the sides `window_sides` counts."""
    return jax.nn.dot_product_attention(
        query, key, value, bias=bias, mask=mask, is_causal=causal,
        key_value_seq_lengths=key_value_seq_lengths,
        local_window_size=None if sliding_window is None else window_sides(causal, sliding_window),
        implementation=implementation)


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
    # The GPU the call lowers for, an assumed one's where a step compiles ahead (`flash_lowered`).
    flash.targeting(flash_target)
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
            # A call's own documents are its runs, whatever their ids (`own_runs`).
            begins, firsts = own_runs(segment_ids, max_segments + 1)
            offsets, own_keys = cu_seqlens(begins, rows * (max_segments + 1), firsts=firsts), {}
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
                       places=None, query_slots=None, dropout_rate=0.0, dropout_seed=None,
                       dropout_places=None, bias=None, fill: float | None = None, in_key: bool = False):
    """The attention template's route (`attention_route.attend`) over the call's visibility as spans
    (`key_spans`): its causality, window, documents (`segment_ids`, or `CrossDocuments` aligned to their last
    keys), key lengths and starts, block and image groups, and key validity and counted places, with
    probability dropout at `dropout_rate` drawn from `dropout_seed`. A softcap is a score hook, a `bias` a
    score hook added after it (`template.Biased`): one term a key `[B or 1, H or 1, 1, K]` (ALiBi's, over the
    keys' logical places) a key parameter, a dense `[B or 1, H or 1, Q, K]` one a tile parameter, read a
    tile at a time and its gradient the tiles' pulled cotangents, a `SeparableBias` a query and a key
    parameter (FoX's); sinks are the softmax's initial state. A
    `fill` (`mask_fill`) is every hidden key's logit, a score hook after the rest (`template.Filled`), the
    call walking every key. On
    a mesh it runs per shard (`on_attention_shards`), as the decode kernel does: the key heads repeat until
    the tensor axes split them, the sinks split with their heads, and the ids, lengths, starts, groups,
    validity and places with
    their rows. Its requirements (`attention_kernels.TEMPLATE_SUPPORTS`) keep it off a sharded sequence axis.
    """
    if sinks is not None and sinks.shape not in ((query.shape[2],), query.shape[:3]):
        raise ValueError(f"sinks hold one logit a query head, or one a query and head, got {sinks.shape}")
    quantized = quantized_rule(sinks is not None, bool(dropout_rate))
    if quantized == 'reference':
        raise ValueError("the template computes a quantized model's attention under an int8 rule of Qwix's "
                         "dynamic absmax, no sinks and no dropout; the reference path computes the rest")

    def local(query, key, value, sinks=None, segment_ids=None, lengths=None, starts=None,
              block_groups=None, image_groups=None, key_valid=None, key_runs=None, places=None,
              query_slots=None, global_rows=None, global_heads=None, seed=None, bias=None, rows_bias=None,
              columns_bias=None, band_values=None, band_origins=None):
        batch, queries, keys = query.shape[0], query.shape[1], key.shape[1]
        spans = key_spans(queries, keys, causal, sliding_window, segment_ids, lengths, starts,
                          block_groups=block_groups, image_groups=image_groups, key_valid=key_valid,
                          key_runs=key_runs, places=places, query_slots=query_slots, rows=batch)
        drop = None
        if seed is not None and dropout_places is not None:
            # A sequence exchange's slice draws at the call's global places (`template.DropPlaces`).
            drop = template.Dropout(seed, dropout_places.rows, dropout_places.heads, dropout_rate,
                                    dropout_places.queries, dropout_places.keys)
        elif seed is not None and global_rows is not None and global_heads is not None:
            drop = template.Dropout(seed, global_rows, global_heads, dropout_rate)
        capped = None if softcap is None else template.Capped(softcap)
        if rows_bias is not None and columns_bias is not None:
            params = (Side(rows_bias, template.Kind.QUERY),
                      Side(columns_bias, template.Kind.KEY))
        elif band_values is not None and band_origins is not None:
            # A band read a tile at a time at its key tile's column (`band.band_tile`), its origins the
            # side after it (`template.Kind.BAND`).
            params = (Side(band_values, template.Kind.BAND),
                      Side(band_origins[:, None, :], template.Kind.QUERY))
        elif bias is None:
            params = ()
        elif bias.shape[2] == 1:
            params = (Side(bias[:, :, 0], template.Kind.KEY),)
        else:
            params = (Side(bias, template.Kind.TILE),)
        score_mod = (capped if not params else template.Banded(template.Biased(capped))
                     if band_values is not None and band_origins is not None else template.Biased(capped))
        if fill is not None:
            score_mod, params = template.Filled.over(spans, fill, score_mod, params)
            spans = template.spans(queries, keys, batch)
        hooks = template.Hooks(
            score=Hook(score_mod, params),
            initial=None if sinks is None else template.sinks(sinks, batch, queries, value.shape[-1]),
            drop=drop)
        if quantized:
            from .kernels.templates import attention_quantized

            # A quantized model's products, which the template forms as its rule does (row 4858).
            return attention_quantized.attend(query, key, value, spans, hooks)
        from .kernels.templates import attention_route

        return attention_route.attend(query, key, value, spans, hooks, in_key=in_key)
    # The GPU arms are Pallas launches, which have no partitioning rule, so each shard runs them on its own
    # rows and heads. A dropping call's draw takes each shard's global rows and heads from the query's own
    # split (`on_attention_shards`'s coordinates), so it does not depend on the mesh or on the rules of the
    # other axes.
    rows = ("activation_batch",)
    if isinstance(bias, BandBias) and bias.values.shape[2] == 1 and query.shape[1] > 1:
        # One query's band broadcast over the call's queries: its key vector written out, O(K), as every
        # route reads it (`BandBias.dense`), each query its one row's.
        bias = bias.dense(key.shape[1])
    separable = bias if isinstance(bias, SeparableBias) else None
    banded = bias if isinstance(bias, BandBias) else None
    dense = None if isinstance(bias, SeparableBias | BandBias) else bias
    band_origins = None if banded is None else band.band_origins(banded.offset, query.shape[1], banded.extent)
    return on_attention_shards(local, query, key, value, coordinates=dropout_seed is not None,
                      sinks=(sinks, ("activation_heads",) if sinks is None or sinks.ndim == 1
                             else ("activation_batch", "activation_length", "activation_heads")),
                      segment_ids=(segment_ids, ("activation_batch", None)), lengths=(lengths, rows),
                      starts=(starts, rows), block_groups=(block_groups, ("activation_batch", None)),
                      image_groups=(image_groups, ("activation_batch", None)),
                      key_valid=(key_valid, ("activation_batch", None)),
                      key_runs=(key_runs, ("activation_batch", None, None, None)),
                      places=(places, ("activation_batch", None)),
                      query_slots=(query_slots, ("activation_batch", None)), seed=(dropout_seed, (None,)),
                      bias=(None if dense is None else jnp.broadcast_to(
                          dense, (*jnp.shape(dense)[:3], key.shape[1])), _bias_axes(dense)),
                      rows_bias=(None if separable is None else separable.rows,
                                 _bias_axes(None if separable is None else separable.rows)[:3]),
                      columns_bias=(None if separable is None else separable.columns,
                                    _bias_axes(None if separable is None else separable.columns)[:3]),
                      band_values=(None if banded is None else banded.values,
                                   _bias_axes(None if banded is None else banded.values)),
                      band_origins=(band_origins, ("activation_batch" if band_origins is not None
                                                   and band_origins.shape[0] > 1 else None, None)))


def _bias_axes(bias: jax.Array | None) -> tuple[str | None, ...]:
    """A template call's bias's axes on the mesh: its batch rows and heads split with the call's where it
    holds one a row and a head, and shared whole where it holds one for them all (`template.shared_place`:
    T5's `[1, H, Q, K]` table read by every row, never broadcast); none for a call without one. A
    `SeparableBias`'s term, `[B or 1, H or 1, Q or K]`, takes the first three."""
    if bias is None:
        return (None,) * 4
    rows, heads = jnp.shape(bias)[:2]
    return ("activation_batch" if rows > 1 else None, "activation_heads" if heads > 1 else None, None, None)


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


class Products(NamedTuple):
    """`reference_attention`'s two products apart, as a kernel's portable rule states each (Splash's,
    `_splash_portable`): the logits' precision (`scores`), their sums' dtype where it names one (`logits`:
    the query's and key's own operands, fp32 sums), the value product's precision (`values`), and its dtype
    where it names one (`weighted`: the weights unrounded and the values widened to it, the query and key
    left in theirs)."""

    scores: jax.lax.PrecisionLike
    values: jax.lax.PrecisionLike
    logits: DTypeLike | None = None
    weighted: DTypeLike | None = None


def scaled_operands(query: jax.Array, key: jax.Array, *, wide_logits: bool = False,
                    scale_logits: bool = False,
                    query_scale: float | None = None) -> tuple[jax.Array, jax.Array]:
    """The query and key the reference path's logits are the product of (`reference_attention`), in their
    common dtype: widened to fp32 under `wide_logits`, as they are under `scale_logits` (both scaled after,
    `scaled_after`), else the query scaled first in its dtype, by 1/sqrt(D) or `query_scale`."""
    dtype = query.dtype
    if wide_logits:
        return query.astype(at_least_fp32(dtype)), key.astype(at_least_fp32(dtype))
    if scale_logits:
        return query, key
    if query_scale is None:
        return query / jnp.sqrt(query.shape[-1]).astype(dtype), key
    return query * jnp.asarray(query_scale, dtype), key


def unscaled_scores(subscripts: str, query: jax.Array, key: jax.Array, *, precision, dtype,
                    scale_logits: bool = False) -> jax.Array:
    """The reference path's product of `scaled_operands`' query and key (`reference_attention`), `dtype`
    the call's operands' own before `scaled_operands` widened any: at the scores' precision (`Products`),
    its sums in the logits' dtype; under `scale_logits` jax.nn's fp32 sums at the dot algorithm it names for
    `dtype`'s 16-bit operands (`_jax_nn_algorithm`, read off `dtype`, never the widened operands': a bf16
    call's preset kept under wide logits too), never rounded to their dtype."""
    products = precision if isinstance(precision, Products) else Products(precision, precision)
    formed = at_least_fp32(dtype) if scale_logits else products.logits
    algorithm = _jax_nn_algorithm(dtype, products.scores) if scale_logits else None
    if algorithm is None:
        return float32_einsum(subscripts, query, key, precision=products.scores,
                              preferred_element_type=formed)
    return jnp.einsum(subscripts, query, key, precision=algorithm, preferred_element_type=formed)


def scaled_after(logits: jax.Array, width: int, *, wide_logits: bool = False, scale_logits: bool = False,
                 query_scale: float | None = None) -> jax.Array:
    """`unscaled_scores`' logits at the call's scale, where `scaled_operands` left it for after
    (`wide_logits`, `scale_logits`): by 1/sqrt(`width`) or `query_scale`; as they are otherwise."""
    if wide_logits or scale_logits:
        return logits * (width ** -0.5 if query_scale is None else query_scale)
    return logits


def reference_scores(query: jax.Array, key: jax.Array, subscripts: str, *, precision=None,
                     wide_logits: bool = False, scale_logits: bool = False) -> jax.Array:
    """The reference path's logits of `query` and `key` (`reference_attention`), formed as it forms them,
    operand scaling, product and scale after: a computation that trains toward the attention's own weights
    reads its logits here (MSA's teacher, `block_sparse.msa_query_kls`)."""
    dtype = query.dtype
    query, key = scaled_operands(query, key, wide_logits=wide_logits, scale_logits=scale_logits)
    logits = unscaled_scores(subscripts, query, key, precision=precision, dtype=dtype,
                             scale_logits=scale_logits)
    return scaled_after(logits, query.shape[-1], wide_logits=wide_logits, scale_logits=scale_logits)


def reference_attention(query, key, value, *, mask=None, bias=None, sinks=None, softcap=None, dtype=None,
                        precision=None, force_fp32_for_softmax=True, dropout_rate=0.0, dropout_seed=None,
                        dropout_places=None, wide_logits=False, scale_logits=False, mask_fill=None,
                        query_scale=None, logit_floor=None, skipped=None):
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
    repeats them, which keeps flax's rounding, but in jax.nn's form
    (`scale_logits`), whose query heads group over their key head as jax.nn's
    do (`_grouped_as_jax_nn`) where the values share the keys' heads (a call
    whose values carry the query's heads, as Z-Image's joint attention's, has
    no group of jax.nn's to match); with sinks the query heads
    group over their key head and no key or value repeats, as the decode
    selector (`attention_kernels.decodes`) needs where it routes a sink
    layer here rather than to the fused decode kernel, which takes sinks on a
    GPU. `wide_logits` computes the logits from the query and keys widened to
    at least fp32 and scales them after the product, as DeepSeek V4 does
    (modeling_deepseek_v4.py:708-736); the value product stays in the inputs'
    dtype. `scale_logits` forms the logits of the query and keys in their own
    dtype as fp32 (the product's preferred type, at the dot algorithm jax.nn
    names for 16-bit operands, `_jax_nn_algorithm`) and scales them after it, by
    `D ** -0.5` in fp32, the mask and softmax reading them in fp32: jax.nn's xla
    form (`jax.nn.dot_product_attention`) and the fused kernels' own, where
    flax's scales the query before the product and rounds its logits to its
    dtype. `mask_fill`, a float, is every masked logit, `where(mask, logits,
    mask_fill)` before the softmax, as jax.nn's `_apply_masks` fills (its
    -0.7 times fp32's largest) and a written-out program its own (-inf, its
    dtype's least); None fills as flax does, its dtype's least, and -inf beside
    sinks.

    A kernel's portable rule states its own walk (Splash's, `splash._splash_portable`, row 12848):
    `query_scale`, the scale its query takes in place of `D ** -0.5` (1 where the query comes scaled);
    `logit_floor`, its softmax's starting maximum (`m = max(m, floor)`, a walk's start, beneath which a
    logit's weight underflows: a mask fill of 0 over logits of -200 weighs zero, not their mean), the sinks'
    own where it has them; `skipped` `[H or 1, Q // b, K]`, the keys a query block of `b` rows never visits
    (a tile a block-sparse walk skips), -inf after the mask fill and zeroed out of the keys and values each
    block's products read, so a skipped key's non-finite value or tangent never forms `0 * NaN` (row
    2654), a row of no weight left 0; and `precision` its two products' apart (`Products`).
    """
    products = precision if isinstance(precision, Products) else Products(precision, precision)
    values_precision = products.values
    if products.weighted is None:
        query, key, value = promote_dtype(query, key, value, dtype=dtype)
    else:
        (query, key), value = promote_dtype(query, key, dtype=dtype), value.astype(products.weighted)
    dtype = query.dtype
    *rows, length, heads, width = query.shape
    kv_heads, keys = key.shape[-2], key.shape[-3]
    grouped = kv_heads != heads and value.shape[-2] == kv_heads
    if (scale_logits and sinks is None and not dropout_rate and grouped and skipped is None
            and query_scale is None and logit_floor is None and products.weighted is None):
        attend = functools.partial(reference_attention, softcap=softcap, precision=precision,
                                   force_fp32_for_softmax=force_fp32_for_softmax, wide_logits=wide_logits,
                                   scale_logits=True, mask_fill=mask_fill)
        return _grouped_as_jax_nn(attend, query, key, value, mask, bias)
    scaled_query, scoring = scaled_operands(query, key, wide_logits=wide_logits, scale_logits=scale_logits,
                                            query_scale=query_scale)
    logits_of = functools.partial(unscaled_scores, precision=products, dtype=dtype, scale_logits=scale_logits)

    if skipped is not None:
        value, logits = _skipped_logits(logits_of, scaled_query, scoring, value, skipped)
    elif sinks is None:
        scoring, value = repeat_kv_heads(scoring, heads), repeat_kv_heads(value, heads)
        logits = logits_of('...qhd,...khd->...hqk', scaled_query, scoring)
    elif kv_heads == 1:
        # One key/value head (DeepSeek V4's): the query heads are a free dim beside the rows, and no
        # size-one head axis rides as a batch dim, which had XLA transpose the fp32 query for the
        # product (a V4 layer's training step on an H100, batch 4 by 2048: 2 GiB more and 5% slower).
        logits = logits_of('...tgd,...sd->...gts', scaled_query, scoring[..., 0, :])
    else:
        grouped = scaled_query.reshape(*rows, length, kv_heads, heads // kv_heads, width)
        logits = logits_of('...tkgd,...skd->...kgts', grouped, scoring).reshape(*rows, heads, length, keys)
    logits = scaled_after(logits, width, wide_logits=wide_logits, scale_logits=scale_logits,
                          query_scale=query_scale)
    if softcap is not None:
        logits = capped_logits(logits, softcap)
    if bias is not None:
        logits = logits + bias
    weights = _weights(logits, mask=mask, mask_fill=mask_fill, skipped=skipped, sinks=sinks,
                       logit_floor=logit_floor, wide=force_fp32_for_softmax, dtype=dtype)
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
    return _weighted(weights, value, skipped=skipped, sinks=sinks, precision=values_precision)


def _weights(logits: jax.Array, *, mask, mask_fill, skipped, sinks, logit_floor, wide: bool, dtype
             ) -> jax.Array:
    """`reference_attention`'s weights of its `logits` `[..., H, Q, K]`: masked (`mask_fill`, -inf beside
    `sinks`), a walk's `skipped` keys -inf after, then softmaxed beside the sinks, from `logit_floor`, or
    from the row's own maximum, in fp32 at least where `wide` (`force_fp32_for_softmax`)."""
    heads, length = logits.shape[-3], logits.shape[-2]
    if mask is not None:
        # Beside a sink a masked key is -inf, so it weighs nothing whatever the sink's logit: the sink
        # keeps every row's denominator finite. Without one the finite floor keeps a row every key
        # masks a number, as flax's does.
        if mask_fill is None:
            mask_fill = -jnp.inf if sinks is not None else jnp.finfo(dtype).min
        logits = jnp.where(mask, logits, mask_fill)
    if skipped is not None:
        rows_per_block = length // skipped.shape[-2]
        logits = jnp.where(jnp.repeat(skipped, rows_per_block, axis=-2), -jnp.inf, logits)
    walked = logit_floor is not None or skipped is not None
    if sinks is not None and sinks.shape not in ((heads,), (*logits.shape[:-3], logits.shape[-2], heads)):
        raise ValueError(f"sinks hold one logit a query head, or one a query and head, got {sinks.shape}")
    if sinks is not None and not walked:
        widest = jnp.promote_types(logits.dtype, sinks.dtype)
        return sink_softmax(logits, sinks, at_least_fp32(widest) if wide else None)
    if walked:
        # A walk's softmax: from its starting maximum (the floor, the sink beside it), a row of no weight
        # none, never NaN, its sink's -inf and every key's (row 12848's review).
        widest = jnp.promote_types(logits.dtype, sinks.dtype) if sinks is not None else logits.dtype
        weights = _floored_softmax(logits.astype(at_least_fp32(widest)) if wide else logits,
                                   -jnp.inf if logit_floor is None else logit_floor, sinks)
        return weights if wide else weights.astype(dtype)
    if wide and dtype != jnp.float32:
        return jax.nn.softmax(logits.astype(at_least_fp32(dtype)))
    # A bias wider than the logits widens them; the softmax runs there and its weights round after.
    return jax.nn.softmax(logits).astype(dtype)


def _weighted(weights: jax.Array, value: jax.Array, *, skipped, sinks, precision) -> jax.Array:
    """`reference_attention`'s output: its `weights` `[..., H, Q, K]` times `value` (`[..., K, KV, Dv]`, its
    heads repeated where it went through `repeat_kv_heads`), each query block's over the keys it visits where
    a walk `skipped` some, grouped by key head beside `sinks` (`[..., Q, H, Dv]`)."""
    if skipped is not None:
        heads, kv_heads = weights.shape[-3], value.shape[-2]
        if kv_heads == heads:
            return _by_query_block(
                lambda weights_of, visited: weighted_values(
                    '...hqk,...khd->...qhd', weights_of, _visited(visited, value), precision=precision),
                weights, skipped, query_axis=-2, out_axis=-3)
        # Grouped by key head (`_skipped_logits`' keys whole, a mask every head shares).
        *rows, _, length, keys = weights.shape
        grouped = weights.reshape(*rows, kv_heads, heads // kv_heads, length, keys)
        # The weights in the values' dtype, as `weighted_values` takes them, over the grouped product, its
        # key and group axes flattened in the product's own leading shape (rows broadcast).
        out = _by_query_block(
            lambda weights_of, visited: float32_einsum(
                '...kgqs,...skd->...qkgd', weights_of.astype(value.dtype), _visited(visited, value),
                precision=precision),
            grouped, skipped, query_axis=-2, out_axis=-4)
        return out.reshape(*out.shape[:-3], heads, out.shape[-1])
    if sinks is None:
        return weighted_values('...hqk,...khd->...qhd', weights, value, precision=precision)
    *rows, heads, length, keys = weights.shape
    kv_heads = value.shape[-2]
    if kv_heads == 1:
        return float32_einsum('...gts,...sd->...tgd', weights.astype(value.dtype), value[..., 0, :],
                              precision=precision)
    groups = weights.astype(value.dtype).reshape(*rows, kv_heads, heads // kv_heads, length, keys)
    return float32_einsum('...kgts,...skd->...tkgd', groups, value, precision=precision).reshape(
        *rows, length, heads, value.shape[-1])


def _floored_softmax(logits: jax.Array, floor, sinks: jax.Array | None = None) -> jax.Array:
    """The softmax of `logits` `[..., H, Q, K]` over the keys from a starting maximum `floor`
    (`reference_attention`'s `logit_floor`) and the sink beside them where given (`sink_column`): `m = max(m,
    floor, sink)`, each weight `exp(logit - m)` over their sum and the sink's `exp(sink - m)`, a row of no
    weight 0, its maximum -inf shifting by none."""
    most = jnp.maximum(jnp.max(logits, axis=-1, initial=-jnp.inf), jnp.asarray(floor, logits.dtype))
    sink = None if sinks is None else sink_column(sinks.astype(logits.dtype), logits.shape)[..., 0]
    if sink is not None:
        most = jnp.maximum(most, sink)
    shift = jnp.where(jnp.isfinite(most), most, 0.0)
    exponentials = jnp.exp(logits - shift[..., None])
    total = jnp.sum(exponentials, axis=-1)
    if sink is not None:
        total = total + jnp.where(jnp.isfinite(most), jnp.exp(sink - shift), 0.0)
    return jnp.where(total[..., None] > 0, exponentials / jnp.where(total > 0, total, 1.0)[..., None], 0.0)


def _visited(visited: jax.Array, operand: jax.Array) -> jax.Array:
    """`operand`'s keys or values `[..., K, KV, D]` a query block visits (`visited` `[K, H or 1, 1]`), the
    rest zero, so a skipped key's non-finite entry or tangent never forms `0 * NaN` (row 2654)."""
    return jnp.where(visited, operand, jnp.zeros((), operand.dtype))


def _skipped_logits(logits_of: Callable, query: jax.Array, key: jax.Array, value: jax.Array,
                    skipped: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`reference_attention`'s logits `[..., H, Q, K]` where a walk `skipped` keys `[H or 1, Q // b, K]`:
    each query block's products over the keys it visits alone (`_by_query_block`), and the values its
    weights then read. A mask every head shares (`H or 1` one) groups the query heads by key head, the
    keys and values whole (row 12848's review: a grouped-query call's keys repeated a query head each);
    a mask each head holds its own repeats them, a head's keys zeroed by its own."""
    heads, kv_heads = query.shape[-2], key.shape[-2]
    if skipped.shape[-3] != 1 or kv_heads == heads:
        key, value = repeat_kv_heads(key, heads), repeat_kv_heads(value, heads)
        return value, _by_query_block(
            lambda rows_of, visited: logits_of('...qhd,...khd->...hqk', rows_of, _visited(visited, key)),
            query, skipped, query_axis=-3, out_axis=-2)
    *rows, length, _, width = query.shape
    grouped = query.reshape(*rows, length, kv_heads, heads // kv_heads, width)
    logits = _by_query_block(
        lambda rows_of, visited: logits_of('...qkgd,...skd->...kgqs', rows_of, _visited(visited, key)),
        grouped, skipped, query_axis=-4, out_axis=-2)
    # The key and group axes flattened in the scan's own leading shape (its rows broadcast the operands').
    return value, logits.reshape(*logits.shape[:-4], heads, *logits.shape[-2:])


def _by_query_block(product: Callable, operand: jax.Array, skipped: jax.Array, *, query_axis: int,
                    out_axis: int) -> jax.Array:
    """`product(block, visited)` over each query block of `operand` (its queries along `query_axis`, in
    blocks of `Q // skipped.shape[-2]`), `visited` `[K, H or 1, 1]` the keys `skipped` `[H or 1, Q // b, K]`
    leaves that block, in one scan, its blocks' outputs joined along `out_axis` (`reference_attention`'s
    `skipped`: the keys and values zeroed a block at a time, no `[Q, K, D]` formed)."""
    blocks = skipped.shape[-2]
    axis = operand.ndim + query_axis
    split = operand.reshape(*operand.shape[:axis], blocks, operand.shape[axis] // blocks,
                            *operand.shape[axis + 1:])
    visited = jnp.moveaxis(jnp.logical_not(skipped), -2, 0)

    def step(_, pair):
        block, seen = pair
        return None, product(block, jnp.swapaxes(seen, 0, 1)[..., None])

    _, outs = jax.lax.scan(step, None, (jnp.moveaxis(split, axis, 0), visited))
    joined = jnp.moveaxis(outs, 0, out_axis - 1)
    at = joined.ndim + out_axis - 1
    return joined.reshape(*joined.shape[:at], joined.shape[at] * joined.shape[at + 1], *joined.shape[at + 2:])


def _grouped_as_jax_nn(attend: Callable, query, key, value, mask, bias) -> jax.Array:
    """`attend` (the reference's jax.nn form, `scale_logits`) over query heads grouped as jax.nn groups them
    (`_dot_product_attention_xla`): the query `[..., T, K, G, D]`, a mask or bias of every head
    `[..., K, G, T, S]` (one shared by the heads read whole), the call mapped over the group with the keys
    and values whole, so each product is one dot over the grouped batch and no key or value repeats; out
    `[..., T, H, Dv]`."""
    *rows, length, heads, width = query.shape
    kv_heads = key.shape[-2]
    groups = heads // kv_heads

    def grouped(operand):
        # A mask's or bias's axis of the group, its heads split as the query's; None where the heads share it.
        if operand is None or jnp.ndim(operand) < 3 or jnp.shape(operand)[-3] == 1:
            return operand, None
        *leading, _, queries, keys = jnp.shape(operand)
        return jnp.reshape(operand, (*leading, kv_heads, groups, queries, keys)), len(leading) + 1

    (mask, mask_axis), (bias, bias_axis) = grouped(mask), grouped(bias)
    axis = len(rows) + 2
    each = jax.vmap(lambda query, mask, bias: attend(query, key, value, mask=mask, bias=bias),
                    in_axes=(axis, mask_axis, bias_axis), out_axes=axis)
    out = each(query.reshape(*rows, length, kv_heads, groups, width), mask, bias)
    return out.reshape(*rows, length, heads, value.shape[-1])


def _jax_nn_algorithm(dtype, precision) -> jax.lax.DotAlgorithmPreset | None:
    """The dot algorithm jax.nn's xla attention names for its logits' product
    (`_dot_product_attention_core`), which the reference's jax.nn form
    (`scale_logits`) declares as jax.nn does: a 16-bit operand's own product
    with fp32 sums (`dew.nn.programs.own_preset`) where the
    call states no precision above the default and the platform it is placed
    on lowers it (`placed_platform`: bf16's everywhere but a GPU before
    `BF16_GPU`, `bf16_dot_runs`; fp16's on a GPU alone, where a CPU's dot and
    a TPU's refuse it, jax.nn's own attention with them); None otherwise, the
    call's own precision taking the product."""
    if not asks_default_precision(precision):
        return None
    if (dtype == jnp.bfloat16 and bf16_dot_runs()) or (
            dtype == jnp.float16 and generation.placed_platform() == 'gpu'):
        return own_preset(dtype)
    return None

def scaled_dot_product_attention(query, key, value, dtype=None, precision=None,
                                 force_fp32_for_softmax=True, implementation: AttentionImpl = 'auto',
                                 causal=False, sliding_window=None, mask=None, bias=None, wide_logits=False,
                                 sinks=None, softcap=None, segment_ids=None, max_segments=None,
                                 key_value_seq_lengths=None, dropout_rate=0.0,
                                 dropout_rng=None, deterministic=True, key_segment_ids=None,
                                 key_value_seq_starts=None, block_groups=None, image_groups=None,
                                 key_valid=None, key_runs=None, places=None, query_slots=None,
                                 scale_logits=False, mask_fill: float | None = None):
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
    `bias` is additive and broadcasts to [B, H, Q, K] (T5's position table) or is a
    `SeparableBias` (FoX's) or a `BandBias` (a bias within a band of keys about each
    query, Inkling 2's relative one). `sinks` is one learned value-free logit per query head
    in the denominator. `softcap` is Gemma 2's tanh on the scaled logits. `wide_logits`
    forms them wide, which the reference alone does (`reference_only`). `scale_logits`
    applies the scale `D ** -0.5` to the fp32 logits after the product (jax.nn's
    and the fused kernels' form) where by default the query in its dtype takes it
    before (flax's): the reference path's arithmetic alone, which every route takes
    as the default's (`reference_attention`). `mask_fill`, a float (-inf
    included), is the logit of every key a row may not see, the whole call's
    visibility masked so (its causality, window, mask, documents, lengths), as
    jax.nn's and a written-out program's `where(mask, logits, fill)` before the
    softmax: a row that sees no key reads every value alike under a finite fill
    and NaN under -inf, and a finite fill within reach of a row's logits weighs
    its masked keys (`_filled`). None, the default, gives a row that sees no key
    exactly zero and weighs a masked key nothing (`visible_rows`). A sequence
    axis above one takes no fill: its exchanges' partial states hold no masked
    key. `segment_ids` `[B, S]`
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
    query's key intervals, whose union it may see, `places` `[B, S]` the places a
    window counts (`counted_places`), and `query_slots` `[B, S]` a cached call's queries'
    own keys (the slots its cache stores them at, -1 for padding), whose blocks they read
    whole. The template reads these as its spans (`key_spans`); every other path takes
    the mask they mean (`visibility_mask`).

    A mesh sequence axis above one runs `sequence_parallel_attention`. Active dropout draws each weight
    from `dropout_rng` on the route the selector takes for it (`reference`, or `template` where named),
    at its global place on a sequence axis above one, and refuses explicit fused kernels. The
    result carries the checkpoint name `Residual.ATTENTION_OUTPUT`, so a remat policy
    can save it instead of replaying the kernel (`remat_block` in dit.py).
    A `value` given as `KeyColumns(width)` is the keys' leading `width` columns: the template reads it from
    the key's slot, every other path the slice.
    """
    value, in_key = _value_of(key, value)
    shards = sequence_shards()
    mask_fill = _fill_of(mask_fill, shards)
    if not 0 <= dropout_rate < 1:
        raise ValueError(f"dropout_rate must be within [0, 1), got {dropout_rate}")
    dropping = bool(dropout_rate) and not deterministic
    if dropping and dropout_rng is None:
        raise ValueError("training attention dropout requires dropout_rng")
    if key_segment_ids is not None:
        if segment_ids is None:
            raise ValueError("key_segment_ids are the keys' documents beside the queries' segment_ids")
        segment_ids = CrossDocuments(segment_ids, key_segment_ids)
    if not causal and key_runs is None:
        # A call that sees every key sees its groups'; one over runs reads its blocks whole past its window.
        block_groups = image_groups = None
    if sliding_window is None:
        places = None
    visibility = None
    if shards > 1:
        spanned = {"block_groups": block_groups, "image_groups": image_groups, "key_valid": key_valid,
                   "key_runs": key_runs, "places": places, "query_slots": query_slots}
        causal, sliding_window, mask, key_value_seq_lengths, materialized, visibility = _sequence_sharded(
            query, key, shards, causal=causal, sliding_window=sliding_window, mask=mask, spanned=spanned,
            segment_ids=segment_ids, lengths=key_value_seq_lengths, starts=key_value_seq_starts)
        if materialized or (visibility and implementation == 'cudnn'):
            implementation = kernel_for_materialized_mask(
                implementation, query, key, dtype=dtype, precision=precision,
                force_fp32_for_softmax=force_fp32_for_softmax, wide_logits=wide_logits)
    words = template.dropout_seed(dropout_rng) if dropping else None
    kernel = functools.partial(
        attention_kernel, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax, implementation=implementation,
        softcap=softcap, wide_logits=wide_logits, scale_logits=scale_logits, mask_fill=mask_fill,
        in_key=in_key, max_segments=max_segments)
    if shards > 1:
        out = sequence_parallel_attention(
            kernel, query, key, value, shards, causal=causal,
            sliding_window=sliding_window, mask=mask, bias=dense_bias(bias, key.shape[-3]), sinks=sinks,
            key_value_seq_lengths=key_value_seq_lengths,
            dropout_rate=dropout_rate if dropping else 0.0, dropout_seed=words, visibility=visibility)
    else:
        out = kernel(query, key, value, causal=causal, sliding_window=sliding_window,
                     mask=mask, bias=bias, sinks=sinks, segment_ids=segment_ids,
                     key_value_seq_lengths=key_value_seq_lengths,
                     key_value_seq_starts=key_value_seq_starts, block_groups=block_groups,
                     image_groups=image_groups, key_valid=key_valid, key_runs=key_runs, places=places,
                     query_slots=query_slots, dropout_rate=dropout_rate if dropping else 0.0,
                     dropout_seed=words)
    return checkpoint_name(out, Residual.ATTENTION_OUTPUT)


def _value_of(key: jax.Array, value) -> tuple[jax.Array, bool]:
    """A call's value as an array, and whether it was given as its keys' leading columns (`KeyColumns`):
    then their slice, within the keys' width."""
    if not isinstance(value, KeyColumns):
        return value, False
    if not 0 < value.width <= key.shape[-1]:
        raise ValueError(f"a value of the keys' leading {value.width} columns is within the keys' "
                         f"{key.shape[-1]}")
    return key[..., :value.width], True


def _fill_of(mask_fill, shards: int) -> float | None:
    """A call's `mask_fill` as a float, checked: a number or -inf, and none over a sequence axis above one,
    whose exchanges' partial states weigh a masked key nothing where a fill weighs it."""
    if mask_fill is None:
        return None
    fill = float(mask_fill)
    if fill != fill:
        raise ValueError("mask_fill is a masked key's logit, a number or -inf, not NaN")
    if shards > 1:
        raise ValueError("a sequence axis above one takes no mask_fill: its exchanges' partial states "
                         "weigh a masked key nothing, which a fill weighs")
    return fill


def _read_nothing(out: jax.Array, live: jax.Array | None) -> jax.Array:
    """`out` with each row that sees no key (`visible_rows`) exactly zero. Its
    cotangent is zero there too, so no kernel's backward reads those rows."""
    return out if live is None else jnp.where(live, out, jnp.zeros((), out.dtype))


def _filled(ran: Callable[[], jax.Array], filled: Callable[[], jax.Array], fill: float, live, query, key,
            value, *, bias, sinks, softcap, precision, dropping: bool) -> jax.Array:
    """A call's output where every key a row may not see has the logit `fill` (`mask_fill`), exactly the
    portable path's (`reference_attention`'s `where(mask, logits, fill)`): the fused kernel (`ran()`)
    weighs a masked key nothing, which the fill weighs too wherever it lies past fp32's exp from the row's
    logits (`_fill_weighs`: none of the call's logits comes within 105 of it), so its rows that see a key
    are the portable path's, and a row that sees none (`live`, `visible_rows`) reads every value alike
    under a finite fill (`_unread`: each key's logit the fill, beside a head's sink) and NaN under -inf,
    beside sinks 0, whose denominator the sink holds (the value's gradient NaN with it, every key's, as the
    portable path's NaN weights give it). Where the fill may weigh a masked key of a row that sees one (a
    fill above some logit, or within 105 below it), or a dropping call has a row that sees none (its draw
    weighs the uniform weights' keys), the call runs `filled()`, a kernel that weighs the fill itself (the
    template's score hook, `template.Filled`, else the portable path), decided as it runs."""
    if fill == -jnp.inf:
        if sinks is not None or live is None:
            return _read_nothing(ran(), live)
        out = ran()
        unread = _unread(value.astype(out.dtype), query.shape[-2], jnp.nan, None, precision)
        return jnp.where(live, out, unread)
    weighs = _fill_weighs(fill, query, key, bias, softcap)
    if dropping and live is not None:
        weighs = weighs | ~jnp.all(live)

    def fused() -> jax.Array:
        out = ran()
        return out if live is None else jnp.where(
            live, out, _unread(value.astype(out.dtype), query.shape[-2], fill, sinks, precision))

    return jax.lax.cond(weighs, filled, fused)


def _fill_weighs(fill: float, query, key, bias, softcap) -> jax.Array:
    """Whether a masked key's logit `fill` may weigh beside a row's logits: whether it lies above the least
    any logit may be, less 105 (fp32's exp of past -103.97 is 0, so the portable path's softmax weighs the
    fill nothing below that) and a margin for the logits' rounding. The least is the product's own bound,
    each query's and key's norms' product at the call's scale (Cauchy-Schwarz), within a `softcap`, plus the
    bias's least."""
    def norm(x):
        return jnp.max(jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32)), axis=-1)))

    reach = norm(query) * norm(key) * query.shape[-1] ** -0.5
    if softcap is not None:
        reach = jnp.minimum(reach, softcap)
    least = -reach if bias is None else _least_bias(bias) - reach
    return fill > least - 105.0 - jnp.abs(least) * 2.0 ** -6


def _unread(value, heads: int, fill: float, sinks, precision) -> jax.Array:
    """A row's output where it sees no key, `[B, 1, H, Dv]` in the value's dtype: each of the call's keys'
    logit `fill`, so each head weighs its values alike, beside its sink (`sinks` `[H]`) each key's weight
    `e^fill` against the sink's `e^sink`. Each value is weighed before the sum, by the portable path's
    grouped value product (`reference_attention`'s: the weights in the value's dtype, each key head's
    values read by its query heads in place), so values whose sum overflows read their finite mean as that
    path reads them; NaN where `fill` is NaN, whose gradient is NaN at every value."""
    *rows, keys, kv_heads, width = value.shape
    if sinks is not None and sinks.ndim != 1:
        raise ValueError("a row seeing no key under a finite fill reads a head's sink, not a query's: "
                         "fill a query's sinks' masked keys -inf")
    if fill != fill:
        share = jnp.full((heads,), jnp.nan, jnp.float32)
    elif sinks is None:
        share = jnp.full((heads,), 1.0 / keys, jnp.float32)
    else:
        sink = jnp.asarray(sinks, jnp.float32)
        peak = jnp.maximum(sink, fill)
        weight = jnp.exp(fill - peak)
        share = weight / (jnp.exp(sink - peak) + keys * weight)
    groups = share.reshape(kv_heads, heads // kv_heads, 1, 1).astype(value.dtype)
    weights = jnp.broadcast_to(groups, (*rows, kv_heads, heads // kv_heads, 1, keys))
    return float32_einsum('...kgts,...skd->...tkgd', weights, value, precision=precision).reshape(
        *rows, 1, heads, width)


def _fused_attention(query, key, value, bias, mask, causal, sliding_window, implementation, *,
                     softcap, sinks, segment_ids, key_value_seq_lengths, max_segments=None,
                     key_value_seq_starts=None, block_groups=None, image_groups=None, key_valid=None,
                     key_runs=None, places=None, query_slots=None, dropout_rate=0.0, dropout_seed=None,
                     dropout_places=None, mask_fill=None, in_key: bool = False, tpu_descriptor=None):
    """Run the kernel `implementation` names, at the query's head width: the executor of what the
    attention selector chose (`resolve_implementation`, `_attention_kernel`), which held the call to
    the kernel's requirements (`attention_kernels.ATTENTION`) before it reaches here.

    A narrower value rides in padded and its columns come back out, at no runtime branch, except that the
    template takes its own value width. `key_value_seq_lengths` reaches cudnn, xla and the template as
    lengths and 'tpu' as the mask it means; `key_value_seq_starts` reaches the template and the decode
    kernel alone, and the groups, key validity and places the template alone (`_attention_kernel` folds them
    into the mask elsewhere), as `mask_fill`, every hidden key's logit (`template_attention`'s `fill`).
    """
    v_head_dim = value.shape[-1]
    if v_head_dim != query.shape[-1] and implementation != 'template':
        value = widen_value_heads(query, value)

    if implementation == 'cudnn':
        out = cudnn_attention(query, key, value, bias, mask, causal, sliding_window, key_value_seq_lengths,
                              segment_ids, max_segments)
    elif implementation == 'triton':
        out = triton_attention(query, key, value, causal)
    elif implementation == 'flash':
        out = flash_attention(query, key, value, causal, sliding_window, sinks, segment_ids, max_segments)
    elif implementation == 'decode':
        # The keys' lengths and starts are the callbacks' operands, not their closure: a closed-over integer
        # tracer stayed in a nested derivative's call as a constant no lowering takes (a paged decode's
        # forward over reverse, M-DD-13b: "No constant handler for type DynamicJaxprTracer").
        rest = {name: x for name, x in (("starts", key_value_seq_starts), ("sinks", sinks)) if x is not None}

        def decoded(query, key, value, lengths, *given):
            held = dict(zip(rest, given, strict=True))

            # A Pallas call has no partitioning rule: each shard decodes its own rows and heads.
            def attend(query, key, value, lengths, starts=None, sinks=None):
                return decode_attention.attend(query, key, value, lengths, starts, softcap=softcap,
                                               sinks=sinks)

            rows = ("activation_batch",)
            return on_attention_shards(attend, query, key, value, lengths=(lengths, rows),
                                       starts=(held.get("starts"), rows),
                                       sinks=(held.get("sinks"), ("activation_heads",)))

        def decoded_reference(query, key, value, lengths, *given):
            held = dict(zip(rest, given, strict=True))
            visible = with_key_starts(with_key_lengths(None, lengths, key.shape[-3]), held.get("starts"),
                                      key.shape[-3])
            return reference_attention(query, key, value, mask=visible, sinks=held.get("sinks"),
                                       softcap=softcap, scale_logits=True)

        # The kernel has no derivative, so both modes differentiate the attention it stands in for, the
        # sinks among its operands, at the kernel's arithmetic (IEEE products for fp32 operands, 16-bit
        # ones summed in fp32, logits fp32: `scale_logits`) on the reference path, not the folded form,
        # whose products read the keys across heads: an H100's fp32 gradients through it held 1.82 to 2.08
        # times the reference's error, through the reference path 1.00 (M-ET-62, row 5268). A differentiated
        # call's output stays the kernel's: the reference path divides its weights before the values'
        # product, the kernel its output after it, and an A100's fp32 decode value through the reference
        # held 1.53 times the folded attention's error (the composed tier at 475913063).
        kernel_precision = "highest" if query.dtype == jnp.float32 else None
        out = forward_differentiable(decoded, decoded_reference, query, key, value, key_value_seq_lengths,
                                     *rest.values(), precision=kernel_precision,
                                     parameters=tuple(4 + index for index, name in enumerate(rest)
                                                      if name == "sinks"))
    elif implementation == 'xla' and folds(query, key, bias, mask, causal, sliding_window):
        # Key lengths and the prefix mask they mean take one path here, so where no lengths-only
        # kernel takes the call the two forms agree bitwise. Its derivative is the reference path's, as
        # the decode kernel's is: the folded form's products read the keys across heads, and an H100's
        # fp32 gradients through them held twice the reference's error (M-ET-62, row 5268), one forward.
        visible = (mask if key_value_seq_lengths is None
                   else with_key_lengths(mask, key_value_seq_lengths, key.shape[-3]))
        held = () if visible is None else (visible,)

        def folded(query, key, value, *visible):
            return folded_attention(query, key, value, *visible)

        def folded_reference(query, key, value, *visible):
            return reference_attention(query, key, value, mask=visible[0] if visible else None,
                                       scale_logits=True)

        out = forward_differentiable(folded, folded_reference, query, key, value, *held, primal_alone=True)
    elif implementation == 'xla':
        # Its softmax takes the textbook derivative y (g - sum(y g)) (`jax.softmax_custom_jvp`, which jax
        # ships off), exactly zero for a query that reads one key, as torch's and the reference path's.
        # The default differentiates `exp(x - max) / sum`, whose g / e - g e / e^2 is zero only where that
        # key's e is exactly 1, and XLA:CPU contracts the scaled logit and the subtraction of the max
        # into one FMA, which leaves e at 1 +- a rounding of the logit, so a rounding's worth of query
        # and key gradient there. The values are the same bits either way.
        with jax.softmax_custom_jvp(new_val=True):
            out = _jax_nn(query, key, value, bias, mask, key_value_seq_lengths, causal=causal,
                          sliding_window=sliding_window, implementation='xla')
    elif implementation == 'template':
        out = template_attention(query, key, value, causal, sliding_window, sinks=sinks, softcap=softcap,
                                 segment_ids=segment_ids, lengths=key_value_seq_lengths,
                                 starts=key_value_seq_starts, block_groups=block_groups,
                                 image_groups=image_groups, key_valid=key_valid, key_runs=key_runs,
                                 places=places, query_slots=query_slots, dropout_rate=dropout_rate,
                                 dropout_seed=dropout_seed, dropout_places=dropout_places, bias=bias,
                                 fill=mask_fill, in_key=in_key)
    else:  # 'tpu'
        if key_value_seq_lengths is not None:
            mask = with_key_lengths(mask, key_value_seq_lengths, key.shape[-3])
        out = tpu_attention(query, key, value, bias, mask, causal, sliding_window,
                            softcap=softcap, sinks=sinks, segment_ids=segment_ids,
                            interpret=generation.backend() != 'tpu', descriptor=tpu_descriptor)
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
                     key_runs=None, places=None, query_slots=None, dropout_rate=0.0, dropout_seed=None,
                     dropout_places=None, wide_logits: bool = False, scale_logits: bool = False,
                     mask_fill: float | None = None, in_key: bool = False):
    """Dispatch one whole-sequence attention call to the kernel it names,
    differentiable in forward mode under `forward_mode_attention`.

    'auto' resolves here, once, against this call's shapes and backend, and
    `_attention_kernel` runs the call on the kernel it resolved to. A query
    row that sees no key comes back zero whichever kernel ran
    (`visible_rows`), inside a sequence exchange too; under a `mask_fill` it
    reads what the fill gives it (`_filled`).
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
        return grouped_query(functools.partial(
            attention_kernel, dtype=dtype, precision=precision, force_fp32_for_softmax=force_fp32_for_softmax,
            implementation=implementation, softcap=softcap, key_value_seq_lengths=key_value_seq_lengths,
            key_value_seq_starts=key_value_seq_starts, wide_logits=wide_logits, scale_logits=scale_logits,
            mask_fill=mask_fill, in_key=in_key), query, key, value)
    lengths = (None if key_value_seq_lengths is None
               else jnp.asarray(key_value_seq_lengths, jnp.int32))
    starts = None if key_value_seq_starts is None else jnp.asarray(key_value_seq_starts, jnp.int32)
    masked = mask if lengths is None else with_key_lengths(mask, lengths, key.shape[-3])
    masked = with_key_starts(masked, starts, key.shape[-3])
    facts = {"dtype": dtype, "precision": precision, "force_fp32_for_softmax": force_fp32_for_softmax,
             "softcap": softcap, "sinks": sinks, "causal": causal, "sliding_window": sliding_window,
             "mask": mask, "bias": bias, "segment_ids": segment_ids, "max_segments": max_segments,
             "key_value_seq_lengths": lengths, "key_value_seq_starts": starts, "block_groups": block_groups,
             "image_groups": image_groups, "key_valid": key_valid, "key_runs": key_runs, "places": places,
             "query_slots": query_slots, "dropout": dropout_seed is not None, "wide_logits": wide_logits}
    resolved = resolve_implementation(implementation, query, key, **facts).kernel
    if resolved != 'template' or FORWARD_MODE.get():
        # The template reads a separable bias as its two terms and a band a tile at a time
        # (`template_attention`); every other route, 'inference''s two among them, adds it written out, as
        # forward mode's reference and kernel do.
        bias = dense_bias(bias, key.shape[-3])
    elif isinstance(bias, RouteBias):
        # The template's form, formed now it reads it.
        bias = bias.band()
    call = functools.partial(
        _attention_kernel, dtype=dtype, precision=precision, force_fp32_for_softmax=force_fp32_for_softmax,
        causal=causal, sliding_window=sliding_window, softcap=softcap, max_segments=max_segments,
        dropout_rate=dropout_rate, wide_logits=wide_logits, scale_logits=scale_logits, in_key=in_key)
    # Every array operand (masks, segment ids, lengths, spans, dropout) passes into `forward_differentiable`
    # as an argument, as the decode kernel passes its lengths and starts above: a closed-over integer or bool
    # tracer stays in a nested derivative's `custom_jvp` call as a constant (M-DD-13b, and a scanned
    # in-backward step over packed decision rows).
    operands, operands_of = holes(
        {"mask": mask, "masked": masked, "segment_ids": segment_ids, "lengths": lengths, "starts": starts,
         "block_groups": block_groups, "image_groups": image_groups, "key_valid": key_valid,
         "key_runs": key_runs, "places": places, "query_slots": query_slots, "dropout_seed": dropout_seed,
         "dropout_places": dropout_places}, lambda leaf: isinstance(leaf, jax.Array | np.ndarray))
    live = visible_rows(query.shape[-3], key.shape[-3], causal, sliding_window, mask, segment_ids, lengths,
                        starts, block_groups=block_groups, image_groups=image_groups, key_valid=key_valid,
                        key_runs=key_runs, places=places, query_slots=query_slots)

    def ran(kernel: str = resolved, fill: float | None = None) -> jax.Array:
        """The call on `kernel`, the one it resolved to by default, every key it hides at the logit `fill`
        (the reference's and the template's) where given, else a row that sees no key whatever it gives."""
        split = primal_alone(kernel, query.dtype, query.shape[-1], causal)
        if split is not None:
            # A call no derivative is taken of on the split's first kernel, a differentiated one's forward
            # and derivatives its second's (`attention_kernels.primal_alone`).
            def on(implementation):
                def run(query, key, value, bias, sinks, *extra):
                    # An arm this host lowers for no GPU of the target (`attention_kernels.unlowered`)
                    # stages its output's shapes, which a program keeping it refuses, named, as it lowers
                    # (`unlowered_launch`): a rematerialized block stages the primal before its derivative
                    # replaces it (row 13318), the differentiated program compiled ahead whatever it.
                    if (why := unlowered(implementation)) is not None:
                        like = jax.eval_shape(on(differentiated), query, key, value, bias, sinks, *extra)
                        return unlowered_launch(generation.Choice.ATTENTION, implementation, why, like)
                    return call(query, key, value, bias=bias, sinks=sinks, implementation=implementation,
                                **operands_of(extra))
                return run

            primal, differentiated = split
            return forward_differentiable(on(primal), on(differentiated), query, key, value, bias, sinks,
                                          *operands, primal_alone=True, parameters=(3, 4))

        def fused(query, key, value, bias, sinks, *extra):
            return call(query, key, value, bias=bias, sinks=sinks, implementation=kernel, mask_fill=fill,
                        **operands_of(extra))

        if not FORWARD_MODE.get() or not reverse_only(kernel):
            return fused(query, key, value, bias, sinks, *operands)

        def reference(query, key, value, bias, sinks, *extra):
            return call(query, key, value, bias=bias, sinks=sinks, implementation='reference',
                        mask_fill=fill, **operands_of(extra))

        return forward_differentiable(fused, reference, query, key, value, bias, sinks, *operands,
                                      parameters=(3, 4))

    if mask_fill is None:
        return _read_nothing(ran(), live)
    if resolved == 'reference':
        return ran(fill=mask_fill)
    # The template weighs the fill itself where it takes the call (`template.Filled`), else the reference.
    templated = ATTENTION.kernel('template').refused(attention_case(query, key, named='template', **facts))
    filled = functools.partial(ran, 'reference' if templated else 'template', mask_fill)
    return _filled(ran, filled, mask_fill, live, query, key, value, bias=bias, sinks=sinks, softcap=softcap,
                   precision=precision, dropping=dropout_seed is not None)


# A kernel's launch this host lowers for no GPU of the target, staged as its output's shapes alone and
# refused, named, where a program keeps it as it lowers (`selection.Unlowerable`): a split kernel's arm a
# rematerialized block stages before it is differentiated, whose differentiated program never lowers it.
_unlowered_p = core.Primitive("dew_unlowered_kernel")
_unlowered_p.multiple_results = True
_unlowered_p.def_abstract_eval(lambda *_, avals, choice, kernel, why: avals)


def _refused(*_, avals, choice, kernel, why):
    raise Unlowerable(choice, kernel, why)


_unlowered_p.def_impl(_refused)
mlir.register_lowering(_unlowered_p, lambda ctx, *_, avals, choice, kernel, why: _refused(
    choice=choice, kernel=kernel, why=why, avals=avals))


def unlowered_launch[Tree](choice: str, kernel: str, why: str, like: Tree) -> Tree:
    """A launch of `kernel` this host lowers for no GPU of the target (`why`), its output `like`'s tree of
    shapes (`jax.eval_shape` of a kernel computing the same function): staged, a program that keeps it
    refuses it, named, as it lowers, and one whose derivative replaces it (a split kernel's primal under
    `nn.remat`) never lowers it."""
    leaves, tree = jax.tree.flatten(like)
    avals = tuple(jax_core.ShapedArray(leaf.shape, leaf.dtype, sharding=jax_core.get_cur_mesh_sharding())
                  for leaf in leaves)
    outputs = _unlowered_p.bind(avals=avals, choice=str(choice), kernel=kernel, why=why)
    return jax.tree.unflatten(tree, outputs)


def forward_differentiable(value, tangent, *arrays, precision: str | None = None, primal_alone: bool = False,
                           parameters: tuple[int, ...] = ()):
    """`value(*arrays)`, whose JVP is `tangent`'s: a kernel that defines only
    its reverse-mode derivative, or none, takes its derivatives from a
    reference computing the same function, reverse mode as the transposed JVP.
    `precision` is the kernel's own arithmetic, which the reference's products
    take too (`jax.default_matmul_precision`), so a kernel of IEEE fp32 products
    is not differentiated by TF32 ones; None leaves the products at the
    caller's. Only the arrays a derivative moves are the reference's tangents: a known zero stays
    out of its derivative, where a product's own derivative (`dew.nn.precision.accumulated`'s on a
    CPU) takes its moving operands alone. With `primal_alone`, `value` runs a call no derivative is taken
    of, and a differentiated call's output is `tangent`'s too: its forward and its derivatives one
    kernel's (an fp32 inference call's template beside xla's training, `attention_kernels.DECLARED`). The
    arrays at `parameters` are the call's parameters (a bias, the sinks), whose gradients reverse mode
    forms with the others' cotangents (`derivatives.with_tangents_formed_at`)."""
    from jax.custom_derivatives import SymbolicZero

    def attending(*arrays):
        return value(*arrays)

    def rule(primals, tangents):
        live = [index for index, direction in enumerate(tangents) if not isinstance(direction, SymbolicZero)]
        if not live:
            output = tangent(*primals) if primal_alone else attend(*primals)
            return output, jax.tree.map(jnp.zeros_like, output)

        def moved(*moving):
            arrays = list(primals)
            for index, array in zip(live, moving, strict=True):
                arrays[index] = array
            return tangent(*arrays)

        # The live tangents through the rule's dual (`with_tangents_formed_at`, the arrays at `parameters`
        # the call's parameters): reverse mode returns the others' cotangents only with theirs formed.
        held = with_tangents_formed_at(tuple(tangents), parameters)
        with contextlib.nullcontext() if precision is None else jax.default_matmul_precision(precision):
            taken, differentiated = jax.jvp(moved, tuple(primals[index] for index in live),
                                            tuple(held[index] for index in live))
        return (taken if primal_alone else attend(*primals)), differentiated

    attend = contracted("test_a_fused_attention_call_in_forward_mode_holds_every_order")(
        declared_tangent(KernelGlue(), attending, rule, symbolic_zeros=True))
    return attend(*arrays)


def _attention_kernel(query, key, value, *, bias, sinks, implementation, dtype, precision,
                      force_fp32_for_softmax, causal, sliding_window, mask, masked, softcap,
                      segment_ids, max_segments, lengths, starts, block_groups=None, image_groups=None,
                      key_valid=None, key_runs=None, places=None, query_slots=None, dropout_rate=0.0,
                      dropout_seed=None, dropout_places=None, wide_logits: bool = False,
                      scale_logits: bool = False, mask_fill: float | None = None, in_key: bool = False):
    """Run one whole-sequence attention call on the kernel `attention_kernel` resolved.

    `masked` is `mask` with the key `lengths` and `starts` in it. Splash takes sinks, a
    softcap and packed segment ids itself, flash and cudnn take a call's own ids as their packed sequences
    given `max_segments`, and the template reads the ids (`CrossDocuments` included) as its spans;
    elsewhere segment ids become the document mask, and sinks and a softcap run
    `reference_attention`, since `jax.nn.dot_product_attention` has neither. The rest goes to
    `_fused_attention`. Key lengths reach
    cudnn, xla and the template as they are and every other path as the mask they mean, which
    splash cannot describe, so 'auto' never picks 'tpu' for them. Key starts reach the decode
    kernel and the template as they are and every other path as the mask they mean. Block and image
    groups, key validity and counted places reach the template alone; every other path takes the call's
    spans as the mask they mean (`visibility_mask`). A `mask_fill` reaches the reference and the template,
    which weighs it by a score hook (`template.Filled`); the other kernels weigh a masked key nothing
    (`_filled`).
    """
    if implementation != 'template':
        # A separable bias reaches the template as its two terms alone, a band as its band
        # (`template_attention`).
        bias = dense_bias(bias, key.shape[-3])
    spanned = (block_groups, image_groups, key_valid, key_runs, places, query_slots)
    tpu_descriptor = None
    aligned = isinstance(segment_ids, CrossDocuments) and (causal or sliding_window is not None)
    if any(x is not None for x in spanned) and implementation != 'template':
        from jax.core import Tracer

        q_len, kv_len = query.shape[-3], key.shape[-3]
        span_inputs = (lengths, starts, block_groups, image_groups, key_valid, key_runs, places, query_slots)
        if (implementation == 'tpu' and bias is None and not aligned
                and not (q_len % SPLASH_LANES or kv_len % SPLASH_LANES)
                and (segment_ids is None or (block_groups is None and image_groups is None))
                and not any(isinstance(x, Tracer) for x in span_inputs)):
            with jax.ensure_compile_time_eval():
                candidate = key_spans(q_len, kv_len, causal, sliding_window, None, lengths, starts,
                                      block_groups=block_groups, image_groups=image_groups,
                                      key_valid=key_valid, key_runs=key_runs, places=places,
                                      query_slots=query_slots)
            # Resolve the complete Splash descriptor once (including any broadcastable concrete `mask`).
            tpu_descriptor = splash_mask_descriptor(
                q_len, kv_len, shard_heads(query), causal=False, sliding_window=None, mask=mask,
                spans=candidate)
            if tpu_descriptor is not None:
                mask = masked = None
                causal, sliding_window, lengths, starts = False, None, None, None
                block_groups = image_groups = key_valid = key_runs = places = query_slots = None
        if tpu_descriptor is None:
            mask = masked = visibility_mask(q_len, kv_len, causal, sliding_window, mask,
                                            segment_ids, lengths, starts, block_groups=block_groups,
                                            image_groups=image_groups, key_valid=key_valid, key_runs=key_runs,
                                            places=places, query_slots=query_slots)
            causal, sliding_window, segment_ids, lengths, starts = False, None, None, None, None
            block_groups = image_groups = key_valid = key_runs = places = query_slots = None
    # Another sequence's documents under causality or a window align each document's queries with
    # its last keys (`CrossDocuments`), which splash does not; elsewhere the mask carries them.
    packed = (implementation == 'template' or (implementation == 'tpu' and not aligned)
              or (implementation == 'flash' and max_segments is not None)
              or (implementation == 'cudnn' and segment_ids is not None and cudnn_packs(
                  not isinstance(segment_ids, CrossDocuments), max_segments is not None,
                  any(x is not None for x in (mask, bias, lengths, starts)))))
    if segment_ids is not None and not packed:
        if aligned:
            inside = cross_document_mask(segment_ids, causal=causal, sliding_window=sliding_window)
            mask, masked = (inside if m is None else jnp.logical_and(m, inside) for m in (mask, masked))
            causal, sliding_window = False, None
        else:
            mask = with_documents(mask, segment_ids)
            masked = with_documents(masked, segment_ids)
        # cuDNN takes the documents it runs no packed sequences of as their mask, an additive bias
        # (`kernel_for_materialized_mask`): a cache's aligned ones as it took a cache's mask, a call's own
        # unbounded ones where cuDNN is named.
        segment_ids = None
    if implementation == 'reference' or (implementation == 'xla' and (
            sinks is not None or softcap is not None or dropout_seed is not None)):
        mask = combined_attention_mask(query.shape[-3], key.shape[-3], causal, sliding_window, masked)
        return reference_attention(query, key, value, mask=mask, bias=bias, sinks=sinks, softcap=softcap,
                                   dtype=dtype, precision=score_precision(precision),
                                   force_fp32_for_softmax=force_fp32_for_softmax, wide_logits=wide_logits,
                                   scale_logits=scale_logits, mask_fill=mask_fill, dropout_rate=dropout_rate,
                                   dropout_seed=dropout_seed, dropout_places=dropout_places)

    # The decode kernel and the template read key starts as their own spans; elsewhere they are the mask.
    if implementation not in ('decode', 'template'):
        mask = with_key_starts(mask, starts, key.shape[-3])
        starts = None

    def fused(query, key, value, bias, sinks):
        return _fused_attention(query, key, value, bias, mask, causal, sliding_window,
                                implementation, softcap=softcap, sinks=sinks,
                                segment_ids=segment_ids, max_segments=max_segments,
                                key_value_seq_lengths=lengths, key_value_seq_starts=starts,
                                block_groups=block_groups, image_groups=image_groups, key_valid=key_valid,
                                key_runs=key_runs, places=places, query_slots=query_slots,
                                dropout_rate=dropout_rate, dropout_seed=dropout_seed,
                                dropout_places=dropout_places, mask_fill=mask_fill, in_key=in_key,
                                tpu_descriptor=tpu_descriptor)
    # A kernel whose products take no precision of their own (jax.nn's xla attention, Pallas dots: the
    # template's portable mainloop, splash and its flash fallback) multiplies at the call's stated
    # precision: a reverse-only kernel's backward, traced after the call, held at it too (`at_precision`),
    # and JAX's own products by the forward's scope alone, which their tangents and transposes keep, so
    # forward mode differentiates them as written. A stated default is a floor the run's configured
    # precision meets, and a kernel that states its own keeps it.
    if reverse_only(implementation) and implementation != 'tpu' and not asks_default_precision(precision):
        return at_precision(precision, parameters=(3, 4))(fused)(query, key, value, bias, sinks)
    with configured_precision(precision):
        return fused(query, key, value, bias, sinks)


def score_precision(precision: jax.lax.PrecisionLike) -> jax.lax.PrecisionLike:
    """The precision the reference path forms a call's logits at, the call's `precision` stated: the
    accurate form's IEEE fp32 (`attention_route.FORMS`), never a GPU's TF32 nor a narrowing preset, under
    the run's accurate numerics; the stated one otherwise (JAX's configured one where None). A computation
    that trains toward the attention's own weights reads its logits at it (MSA's teacher,
    `block_sparse.msa_query_kls`)."""
    return EXACT_FLOAT32 if generation.numerics().forms == "accurate" else precision


def reference_only(query, dtype, precision, force_fp32_for_softmax, wide_logits: bool = False) -> bool:
    """Whether a call asks for arithmetic only the reference path performs.

    The four are what every fused kernel's requirements refuse
    (`attention_kernels`' arithmetic requirement): a matmul precision, stated
    or the run's configured one, that whole products of the query's dtype
    with fp32 sums do not compute (`exact_products`: a dot algorithm that
    narrows the operands or widens them; any level, a 16-bit operand's
    exact products at 'highest' among them) or that no config names (a
    pair, a dot algorithm: `configurable_precision`), which a fused kernel
    whose products take none of their own runs at (`_attention_kernel`), a
    softmax outside fp32, a compute dtype other than the inputs', and logits
    formed from queries and keys widened to at least fp32 (`wide_logits`).
    """
    return (not exact_products(precision, query.dtype) or not configurable_precision(precision)
            or not force_fp32_for_softmax
            or (dtype is not None and jnp.dtype(dtype) != query.dtype)
            or wide_logits)


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


def quantized_rule(sinks: bool, dropout: bool = False) -> str | None:
    """`AttentionCase.quantized`'s kernel for a call with `sinks` or `dropout`: the template's quantized
    attention where a quantized model's rule owning its products takes it, the reference path where it does
    not; None where no rule owns them or it quantizes no activation."""
    from .kernels.templates import attention_quantized

    rule = attention_quantized.quantizing()
    if rule is None:
        return None
    return 'template' if attention_quantized.takes(rule) and not (sinks or dropout) else 'reference'

def attention_case(query, key, *, dtype=None, precision=None, force_fp32_for_softmax=True, softcap=None,
                   sinks=None, causal=False, sliding_window=None, mask=None, bias=None, segment_ids=None,
                   max_segments=None, key_value_seq_lengths=None, key_value_seq_starts=None,
                   block_groups=None, image_groups=None, key_valid=None, key_runs=None, places=None,
                   query_slots=None, dropout: bool = False, wide_logits: bool = False,
                   named: str = 'auto') -> AttentionCase:
    """The `AttentionCase` of a call `scaled_dot_product_attention` takes under the `named`
    `AttentionImpl`, which decides whether splash might run it."""
    cross = isinstance(segment_ids, CrossDocuments)
    q_len, kv_len = query.shape[-3], key.shape[-3]
    laid = (None if bias is None else bias.shape if isinstance(bias, SeparableBias | RouteBias)
            else jnp.shape(bias))
    # Groups a call reads under causality, or past the window of its runs (`scaled_dot_product_attention`).
    grouped = ((bool(causal) or key_runs is not None)
               and (block_groups is not None or image_groups is not None))
    case = AttentionCase(
        queries=q_len, keys=kv_len, rows=math.prod(query.shape[:-3]), heads=query.shape[-2],
        kv_heads=key.shape[-2],
        head_dim=query.shape[-1], dtype=jnp.dtype(query.dtype), key_dtype=jnp.dtype(key.dtype),
        causal=bool(causal), window=sliding_window,
        documents=None if segment_ids is None else 'cross' if cross else 'own',
        aligned=cross and (causal or sliding_window is not None), max_segments=max_segments,
        lengths=key_value_seq_lengths is not None, starts=key_value_seq_starts is not None,
        blocks=grouped,
        spans_only=(grouped or key_valid is not None or key_runs is not None or query_slots is not None
                    or (places is not None and sliding_window is not None)),
        mask=mask is not None, bias=bias is not None,
        bias_hook=(bias is not None and laid is not None and len(laid) == 4
                   and laid[-2] in (1, query.shape[1]) and laid[0] in (1, query.shape[0])
                   and laid[1] in (1, query.shape[2])
                   and jnp.dtype(bias.dtype).itemsize <= 4),
        dense_bias=(laid is not None and len(laid) == 4 and laid[-2] > 1
                    and not isinstance(bias, SeparableBias | BandBias | RouteBias)),
        separable_bias=isinstance(bias, SeparableBias | BandBias | RouteBias),
        sinks=sinks is not None, query_sinks=sinks is not None and np.ndim(sinks) > 1, softcap=softcap,
        arithmetic=reference_only(query, dtype, precision, force_fp32_for_softmax, wide_logits),
        narrows=_xla_kernel_narrows(query),
        folds=folds(query, key, bias, mask, causal, sliding_window),
        splash_mask=False, shards=sequence_shards(), dropout=dropout, form=generation.numerics().forms,
        quantized=quantized_rule(sinks is not None, dropout),
        # A device's launch on a mesh, `[B, T, N, D]` calls' (`on_attention_shards` splits those alone).
        launch=launched(query, key) if query.ndim == 4 else None)
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
                           image_groups=None, key_valid=None, key_runs=None, places=None, query_slots=None,
                           dropout: bool = False, wide_logits: bool = False) -> Route:
    """The route an `AttentionImpl` takes for this call (`Route.kernel`, `Route.forced` what forced it),
    as the attention selector
    reports it (`select`, over `attention_case`): 'auto' takes the kernel the target's row measured
    fastest for the call where it takes it, else the first that does; a named kernel is forced, and
    raises where it does not take the call (xla handing what it cannot honour to the reference path).
    A call of few positions over key lengths is the decode selector's (`DECODE`), whose decode kernel
    runs as 'decode' (`_fused_attention`); a named xla lowers it there too, as jax.nn's own lowering
    of a decode step, unless `generation.chosen(decode='xla')` names the folded attention. A kernel this
    host cannot lower for an assumed GPU raises (`selection.Unlowerable`)."""
    attention_implementation(implementation)
    case = attention_case(query, key, dtype=dtype, precision=precision,
                          force_fp32_for_softmax=force_fp32_for_softmax, softcap=softcap, sinks=sinks,
                          causal=causal, sliding_window=sliding_window, mask=mask, bias=bias,
                          segment_ids=segment_ids, max_segments=max_segments,
                          key_value_seq_lengths=key_value_seq_lengths,
                          key_value_seq_starts=key_value_seq_starts, block_groups=block_groups,
                          image_groups=image_groups, key_valid=key_valid, key_runs=key_runs, places=places,
                          query_slots=query_slots, dropout=dropout, wide_logits=wide_logits,
                          named=implementation)
    if not (decodes(case) and implementation in ('auto', 'cudnn', 'xla')):
        return select(ATTENTION, case, None if implementation == 'auto' else implementation)
    if implementation == 'xla':
        # An explicit xla's decode step: the decode kernel where it takes the call, else xla's lowering of
        # it, a decode choice `chosen` names overriding both.
        return select(DECODE, case, preference=('decode', 'xla', 'reference'))
    return select(DECODE, case, {'auto': None, 'cudnn': 'cudnn'}[implementation])


def _sequence_sharded(query, key, shards: int, *, causal, sliding_window, mask, segment_ids, lengths, starts,
                      spanned: dict):
    """A call's visibility as the sequence exchange takes it (`sequence_parallel_attention`). Where the
    exchange passes them (`visibility_refused`, the predicate it reads too), the call's per-token ids
    (documents, key starts, `spanned`'s block and image groups, key validity, runs, places and query slots)
    travel as its `visibility`: whole beside the all-to-all's whole sequences, or beside the gather's whole
    keys, each striped query reading them at its slot; no dense `[B, 1, S, S]` mask (row 1495). Elsewhere
    (a cross call the gather takes) each id is the mask it means, its causality, window, documents, key
    lengths and starts among them where `spanned` holds any (`visibility_mask`), and the documents travel
    as the mask they stand for, causality and the window folded in beside them. Returns the causality,
    window, mask and key lengths it takes, whether it materialized a mask its kernel must take
    (`kernel_for_materialized_mask`), and the visibility ids, which a call naming cuDNN takes on that masked
    kernel too (cuDNN takes no mask; its kernel would build one)."""
    queries, keys = query.shape[-3], key.shape[-3]
    ids = {name: x for name, x in {"segment_ids": segment_ids, "key_value_seq_starts": starts,
                                   **spanned}.items() if x is not None}
    check_rows(query.shape[0], {**ids, "key_value_seq_lengths": lengths})
    if ids and visibility_refused(query, key, shards, ids) is None:
        return causal, sliding_window, mask, lengths, False, ids
    if any(x is not None for x in spanned.values()):
        mask = visibility_mask(queries, keys, causal, sliding_window, mask, segment_ids, lengths, starts,
                               **spanned)
        return False, None, mask, None, True, None
    built = False
    if isinstance(segment_ids, CrossDocuments) and (causal or sliding_window is not None):
        aligned = cross_document_mask(segment_ids, causal=causal, sliding_window=sliding_window)
        mask = aligned if mask is None else jnp.logical_and(aligned, mask)
        causal, sliding_window, segment_ids, built = False, None, None, True
    if starts is not None:
        mask, built = with_key_starts(mask, starts, keys), True
    if segment_ids is not None:
        documents = with_documents(mask, segment_ids)
        mask = combined_attention_mask(queries, keys, causal, sliding_window, documents)
        return False, None, mask, lengths, True, None
    return causal, sliding_window, mask, lengths, built, None


def kernel_for_materialized_mask(implementation: AttentionImpl, query, key, *, dtype=None, precision=None,
                                 force_fp32_for_softmax=True, wide_logits: bool = False) -> AttentionImpl:
    """The kernel a call that built an explicit mask runs: a kernel the call names, cuDNN aside, which runs
    the mask as xla does; else the attention selector's (`_masked_kernel`): one a run's context forces
    (`generation.chosen`), else the xla kernel, the reference path where xla hands the call on.

    cuDNN has no mask argument: jax hands it a bool mask as an additive bias of
    -2**41 in the compute dtype (`combine_bias_and_mask` in
    jax/_src/cudnn/fused_attention_stablehlo.py), and `check_is_flash_attention`
    then refuses an odd length while training. The xla kernel masks by
    exclusion with the same fp32 softmax, at 83.6 ms and 5.80 GiB a step against
    cuDNN's fixed window at 75.8 ms and 4.99 GiB
    (docs/concepts/language_models.md).
    """
    if implementation not in ('auto', 'cudnn', 'xla'):
        return implementation
    # cuDNN aside: a call whose run chose it (`chosen_attention`) runs its built mask on xla's kernel, as the
    # xla kernel the call names, never the cuDNN a context forces, which takes no mask.
    return _masked_kernel(implementation, None if implementation == 'auto' else 'xla', query, key,
                          ('xla', 'reference'), dtype=dtype, precision=precision,
                          force_fp32_for_softmax=force_fp32_for_softmax, wide_logits=wide_logits)


def mask_kernel(implementation: AttentionImpl, query, key, *, as_bias: bool, dtype=None, precision=None,
                force_fp32_for_softmax=True, softcap=None, sinks: bool = False,
                wide_logits: bool = False) -> AttentionImpl:
    """The kernel a call that built its mask runs, decided where the call builds it: where the mask suits
    cuDNN's dense bias (`as_bias`: a window's band, a forward-only cache call over several queries) and the
    call chooses ('auto') or names cudnn, cuDNN where it takes the call, then xla (a call cuDNN cannot
    honour: sinks, a dtype or a width it refuses, a GPU this process does not place it on), then the
    reference path; elsewhere `kernel_for_materialized_mask`'s. A kernel a run's context forces
    (`generation.chosen`) is taken as forced."""
    if not (as_bias and implementation in ('auto', 'cudnn')):
        return kernel_for_materialized_mask(implementation, query, key, dtype=dtype, precision=precision,
                                            force_fp32_for_softmax=force_fp32_for_softmax,
                                            wide_logits=wide_logits)
    return _masked_kernel(implementation, None if implementation == 'auto' else implementation, query, key,
                          ('cudnn', 'xla', 'reference'), dtype=dtype, precision=precision,
                          force_fp32_for_softmax=force_fp32_for_softmax, softcap=softcap, sinks=sinks,
                          as_bias=True, wide_logits=wide_logits)


def _masked_kernel(implementation: AttentionImpl, named: str | None, query, key,
                   preference: tuple[str, ...], *, dtype, precision, force_fp32_for_softmax, softcap=None,
                   sinks: bool = False, as_bias: bool = False, wide_logits: bool = False) -> AttentionImpl:
    """The attention selector's kernel for a call of `implementation` with a mask (`selection.selected`,
    reporting nothing: the dispatch that runs it reports its route): `named` where it takes the call, else
    one a run's context forces, else `preference`'s first that takes it, the reference path where xla hands
    it on (`attention_kernels.XLA_SUPPORTS`). A mask the call hands cuDNN as its dense bias (`as_bias`, as
    jax's cuDNN path takes a mask, `combine_bias_and_mask`) is that bias to the selector, its rows' own;
    any other an arbitrary mask (`AttentionCase.mask`), which cuDNN is measured slower at."""
    form = {"bias": True, "dense_bias": True} if as_bias else {"mask": True}
    case = dataclasses.replace(
        attention_case(query, key, dtype=dtype, precision=precision,
                       force_fp32_for_softmax=force_fp32_for_softmax, softcap=softcap,
                       sinks=True if sinks else None, wide_logits=wide_logits, named=implementation),
        folds=False, **form)
    return attention_implementation(selected(ATTENTION, case, named, preference=preference).name)
