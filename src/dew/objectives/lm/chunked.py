"""Cross entropy that never holds the whole logits tensor.

The forward walks the vocabulary in `chunks` tiles, keeping two float32
numbers per token, four with the argmax, rather than a row of `vocab`. It is
exact: logsumexp over a concatenation is `logaddexp` of the parts', the target
logit lives in exactly one chunk, and a strict comparison over chunks taken in vocabulary order keeps
the lowest index among equals, which is `jnp.argmax`'s rule on the whole row.

The backward recomputes each `[token_tile, vocab_tile]` block of logits from
the states, the head and the saved log partition, so neither the logits nor
their cotangent is ever wider than one tile. The full-width head gradient is
an output, not a temporary. The two-dimensional recomputing VJP is Tokamax's
(`linear_softmax_cross_entropy_loss/chunked_xla.py`, `21c5828`); the fp32
projection, the softcap, the tie rule and the differentiable log partition are
this loss's own.

First-order reverse mode only: there is no JVP rule. On one RTX 4080, fp32,
1,024 tokens by 2,816 features by 262,144 columns in four chunks, the default
tile runs the head forward and backward in 250 ms holding 1.07 GiB of
temporaries, against 249 ms and 3.55 GiB for recomputing whole chunks.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import PrecisionLike

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.multimodal import MultimodalTransformer
from dew.nn.precision import head_product, rounds_to_bf16


def vocabulary_chunks(vocab_size: int, chunks: int) -> tuple[tuple[int, int], ...]:
    """Cut `range(vocab_size)` into `chunks` half-open column ranges.

    The last chunk is the short one when the count does not divide the
    vocabulary. Uneven tiles are free here: the full-width chunks are the
    iterations of one loop and the short last one is its own trace
    (`_over_tiles`), so nothing has to be padded to a common width and
    masked back out, and only one chunk's slice of the head exists at a
    time rather than every chunk's, hoisted together ahead of the loop.
    """
    if chunks < 1:
        raise ValueError(f"a vocabulary needs at least one chunk, got {chunks}")
    if chunks > vocab_size:
        raise ValueError(
            f"{chunks} chunks for a vocabulary of {vocab_size} leaves empty tiles")
    width = -(-vocab_size // chunks)
    bounds = tuple((start, min(start + width, vocab_size))
                   for start in range(0, vocab_size, width))
    if len(bounds) != chunks:
        # ceil division can cover the vocabulary early (13 columns in 4 chunks
        # is 4 + 4 + 4 + 1), so the caller's count is the number of tiles.
        raise ValueError(
            f"{vocab_size} columns do not split into {chunks} chunks of {width}")
    return bounds


BF16 = jax.lax.DotAlgorithmPreset.BF16_BF16_F32


def head_precision(dtype, precision: PrecisionLike, bf16: bool) -> jax.lax.PrecisionLike:
    """The precision the head multiplies at: the bf16 algorithm when `bf16`
    asks for it and bf16 compute at the default precision allows it, the
    caller's otherwise."""
    return BF16 if bf16 and rounds_to_bf16(dtype, precision) else precision


def bf16_head(model: nn.Module) -> bool:
    """Whether `model`'s vocabulary head multiplies as bf16
    (`CausalTransformer.bf16_head`), read through the wrappers that hold a
    decoder: a multimodal model's language model, DiffusionGemma's text."""
    if isinstance(model, MultimodalTransformer):
        model = model.language_model
    if isinstance(model, DiffusionGemma):
        model = model.text
    return isinstance(model, CausalTransformer) and model.bf16_head


def _operand_dtype(precision: jax.lax.PrecisionLike):
    """The dtype the head's operands take: bf16 under the bf16 algorithm, fp32
    otherwise."""
    return jnp.bfloat16 if precision is BF16 else jnp.float32


def _capped(logits, softcap):
    if softcap is None:
        return logits
    cap = jnp.asarray(softcap, jnp.float32)
    return cap * jnp.tanh(logits / cap)


def head_logits(hidden, head_weight, *, softcap: float | None,
                precision: PrecisionLike, vocab_major: bool = False,
                bf16: bool = False) -> jax.Array:
    """`hidden @ head_weight` as the model's forward scores it: fp32 states
    against the `[features, vocab]` head (`[vocab, features]` with
    `vocab_major`) in its stored dtype, accumulated in fp32, softcapped when
    the backbone caps; `[..., vocab]`. With `bf16`, bf16 states multiply the
    head as bf16 (`CausalTransformer.bf16_head`,
    `dew.nn.precision.head_product`)."""
    return _capped(head_product('...d,vd->...v' if vocab_major else '...d,dv->...v',
                                hidden, head_weight, precision, bf16=bf16), softcap)


def _tile_logits(states, matrix, precision: jax.lax.PrecisionLike):
    """Uncapped `[tokens, features] @ [columns, features].T` in fp32, over
    operands already in the dtype the head multiplies in."""
    return jnp.einsum('td,vd->tv', states, matrix.astype(states.dtype),
                      precision=precision, preferred_element_type=jnp.float32)


class _ChunkTerms(NamedTuple):
    """One tile's logsumexp and target logit, and its best logit and column
    when a prediction is asked for."""
    lse: jax.Array
    picked: jax.Array
    best: jax.Array | None
    column: jax.Array | None


def _chunk_terms(hidden, head_chunk, targets, start: int, stop: int,
                 softcap: float | None, precision: jax.lax.PrecisionLike,
                 predict: bool) -> _ChunkTerms:
    """One tile's `_ChunkTerms`."""
    logits = _capped(_tile_logits(hidden, head_chunk, precision), softcap)

    inside = (targets >= start) & (targets < stop)
    column = jnp.clip(targets - start, 0, stop - start - 1)
    picked = jnp.take_along_axis(logits, column[:, None], axis=-1)[:, 0]
    lse, picked = jax.nn.logsumexp(logits, axis=-1), jnp.where(inside, picked, 0.0)
    if not predict:
        return _ChunkTerms(lse, picked, None, None)
    # A vocabulary column is int32 whatever the run's default integer width;
    # argmax widens to int64 under x64.
    return _ChunkTerms(lse, picked, jnp.max(logits, axis=-1),
                       jnp.argmax(logits, axis=-1).astype(jnp.int32) + start)


def _over_tiles(carry, count: int, width: int, body: Callable):
    """`body(start, carry, size)` over `count` rows in `width`-row tiles."""
    full, tail = divmod(count, width)
    if full:
        carry = jax.lax.fori_loop(
            0, full, lambda i, inner: body(i * width, inner, width), carry)
    if tail:
        # A slice size is a compile-time constant, so the short last tile is
        # its own trace and cannot join the loop over the full ones.
        carry = body(full * width, carry, tail)
    return carry


def _forward(hidden, table, targets, chunks: int, token_tile: int,
             softcap, precision: jax.lax.PrecisionLike, predict: bool):
    """Losses, top-1 columns (None unless `predict`) and log partitions, a
    token tile at a time."""
    features = table.shape[1]
    flat = hidden.reshape(-1, features)
    labels = targets.reshape(-1)
    bounds = vocabulary_chunks(table.shape[0], chunks)
    width = bounds[0][1]
    operands = _operand_dtype(precision)

    def tokens(start, outputs, size):
        states = jax.lax.dynamic_slice_in_dim(flat, start, size).astype(operands)
        picked_targets = jax.lax.dynamic_slice_in_dim(labels, start, size)

        def columns(first, carry, count):
            terms = _chunk_terms(
                states, jax.lax.dynamic_slice_in_dim(table, first, count),
                picked_targets, first, first + count, softcap, precision, predict)
            total = jnp.logaddexp(carry[0], terms.lse)
            target_logit = carry[1] + terms.picked
            if terms.best is None or terms.column is None:
                return total, target_logit
            better = terms.best > carry[2]
            return (total, target_logit, jnp.where(better, terms.best, carry[2]),
                    jnp.where(better, terms.column, carry[3]))

        initial = (jnp.full((size,), -jnp.inf, jnp.float32), jnp.zeros((size,), jnp.float32))
        if predict:
            initial += (jnp.full((size,), -jnp.inf, jnp.float32), jnp.zeros((size,), jnp.int32))
        total, target_logit, *best = _over_tiles(initial, table.shape[0], width, columns)
        values = (total - target_logit, total, *best[1:])
        return tuple(jax.lax.dynamic_update_slice_in_dim(out, value, start, axis=0)
                     for out, value in zip(outputs, values, strict=True))

    count = labels.shape[0]
    outputs = (jnp.zeros((count,), jnp.float32), jnp.zeros((count,), jnp.float32))
    if predict:
        outputs += (jnp.zeros((count,), jnp.int32),)
    losses, log_z, *predicted = (value.reshape(targets.shape) for value in
                                 _over_tiles(outputs, count, token_tile, tokens))
    return losses, predicted[0] if predict else None, log_z


def _bounded_head_impl(hidden, table, targets, chunks: int, tile: tuple[int, int],
                       softcap, precision: jax.lax.PrecisionLike, predict: bool):
    """Run `_forward` behind a backward that recomputes its logits."""
    return _forward(hidden, table, targets, chunks, tile[0], softcap, precision, predict)


def _bounded_head_fwd(hidden, table, targets, chunks, tile, softcap, precision, predict):
    outputs = _forward(hidden, table, targets, chunks, tile[0], softcap, precision, predict)
    # The residuals are the inputs and one float32 per token. Everything the
    # backward needs beyond them is a recomputed tile.
    return outputs, (hidden, table, targets, outputs[2], softcap)


def _bounded_head_bwd(chunks, tile, precision, predict, residuals, cotangents):
    """Pull the cotangents back through logits recomputed one tile at a time.

    The outer loop walks vocabulary tiles carrying `(d_states, d_table,
    d_cap)`: the full-width state gradient, the head gradient stored tile by
    tile, and the softcap's. The inner loop walks token tiles carrying
    `(d_states, d_matrix, d_cap)`, where `d_matrix` accumulates one
    vocabulary tile's head gradient in fp32 before it is stored.
    """
    del chunks, predict  # The backward tiles by column, and argmax has no gradient.
    hidden, table, targets, log_z, softcap = residuals
    # The operands hold the forward's values, widened to fp32 so the logits'
    # cotangent is not rounded here: the bf16 algorithm rounds it inside the
    # product on a backend that has one, as the full pass's backward does.
    operands = _operand_dtype(precision)
    loss_cotangent, _, partition_cotangent = cotangents
    token_tile, vocab_tile = tile
    features = table.shape[1]
    flat = hidden.reshape(-1, features)
    labels, partitions = targets.reshape(-1), log_z.reshape(-1)
    d_loss = loss_cotangent.reshape(-1)
    d_partition = partition_cotangent.reshape(-1)
    count = labels.shape[0]

    def columns(first, gradients, width):
        d_states, d_table, d_cap = gradients
        matrix = jax.lax.dynamic_slice_in_dim(
            table, first, width, axis=0).astype(operands).astype(jnp.float32)

        def tokens(start, carry, size):
            d_states, d_matrix, d_cap = carry
            states = jax.lax.dynamic_slice_in_dim(
                flat, start, size).astype(operands).astype(jnp.float32)
            token_z = jax.lax.dynamic_slice_in_dim(partitions, start, size)
            token_loss = jax.lax.dynamic_slice_in_dim(d_loss, start, size)
            token_partition = jax.lax.dynamic_slice_in_dim(d_partition, start, size)

            logits, cap_pullback = jax.vjp(
                _capped, _tile_logits(states, matrix, precision), softcap)
            # log Z is the whole row's, so a tile's share of the softmax needs
            # no renormalisation, and a target outside the tile one-hots to
            # zero rather than to a wrapped column.
            probabilities = jnp.exp(logits - token_z[:, None])
            selected = jax.nn.one_hot(
                jax.lax.dynamic_slice_in_dim(labels, start, size) - first, width,
                dtype=jnp.float32)
            d_logits = ((token_loss + token_partition)[:, None] * probabilities
                        - token_loss[:, None] * selected)
            d_raw, cap_tile = cap_pullback(d_logits)
            states_tile = jnp.einsum('tv,vd->td', d_raw, matrix, precision=precision,
                                     preferred_element_type=jnp.float32)
            matrix_tile = jnp.einsum('tv,td->vd', d_raw, states, precision=precision,
                                     preferred_element_type=jnp.float32)
            prior = jax.lax.dynamic_slice_in_dim(d_states, start, size)
            d_states = jax.lax.dynamic_update_slice_in_dim(
                d_states, prior + states_tile, start, axis=0)
            if d_cap is not None:
                d_cap = d_cap + cap_tile
            return d_states, d_matrix + matrix_tile, d_cap

        d_states, d_matrix, d_cap = _over_tiles(
            (d_states, jnp.zeros((width, features), jnp.float32), d_cap),
            count, token_tile, tokens)
        # fp32 until every token tile is in, so a bf16 head does not round
        # once per tile.
        d_table = jax.lax.dynamic_update_slice_in_dim(
            d_table, d_matrix.astype(table.dtype), first, axis=0)
        return d_states, d_table, d_cap

    d_states, d_table, d_cap = _over_tiles(
        (jnp.zeros(flat.shape, jnp.float32),
         jnp.zeros(table.shape, table.dtype),
         None if softcap is None else jnp.zeros_like(softcap)),
        table.shape[0], vocab_tile, columns)
    return (d_states.reshape(hidden.shape).astype(hidden.dtype), d_table, None, d_cap)


# `jax.custom_vjp` is generic in its return type, and a `functools.partial`
# decorator loses that binding, so it is built by hand.
_bounded_head = jax.custom_vjp(_bounded_head_impl, nondiff_argnums=(3, 4, 6, 7))
_bounded_head.defvjp(_bounded_head_fwd, _bounded_head_bwd)


def chunked_cross_entropy(hidden, head_weight, targets, chunks: int, *,
                          softcap: float | None = None,
                          precision: PrecisionLike = None,
                          tile: tuple[int, int] = (1024, 8192),
                          vocab_major: bool = False,
                          predict: bool = True,
                          bf16: bool = False):
    """Per-token cross entropy of `hidden @ head_weight`, its top-1 column
    and its log partition.

    `hidden` is `[..., features]` states in any compute dtype, `head_weight`
    the `[features, vocab]` head in its stored dtype (each tile upcasts its
    own slice; the products are accumulated in fp32), `targets` the `[...]`
    int32 ids. Returns the per-token losses, the argmax prediction (None
    when `predict` is False, which skips the argmax: 0.77 ms of the head's
    8.0 on a TPU v6e) and the row's logsumexp (log Z, which PaLM's z-loss
    squares), all shaped like `targets`, each of the two float32 outputs carrying its own gradient. The
    caller owns the weighting and the mean, and with them the padding id.

    The backward keeps the head it was given. With `vocab_major` the head
    is `[vocab, features]`, the way a tied embedding table is stored, and
    the value kept is the parameter itself; a `[features, vocab]` head is
    transposed here first, and what the backward keeps is that transposed
    array, a vocabulary-sized copy of a tied table (`head_table` on the
    backbone hands out the stored orientation).

    The states are widened to fp32 and multiply the head as stored. With
    `bf16` (`CausalTransformer.bf16_head`), bf16 states at the default
    precision multiply the head as bf16 and accumulate in fp32, forward and
    backward.

    `softcap` is the backbone's `final_logit_softcap`. It is elementwise, so
    capping a tile and capping the row agree. `tile` is a `(tokens, columns)`
    block: the forward walks tokens in the first, the backward walks both and
    holds nothing wider, so it is the only knob on the backward's
    temporaries. The default is measured; a tile wider than the input is one
    tile.
    """
    features = head_weight.shape[1 if vocab_major else 0]
    if hidden.shape[-1] != features:
        raise ValueError(
            f"hidden states are {hidden.shape[-1]} wide and the head is {features}")
    if hidden.shape[:-1] != targets.shape:
        raise ValueError(
            f"{targets.shape} targets for {hidden.shape[:-1]} hidden states")
    if len(tile) != 2 or min(tile) < 1:
        raise ValueError(f"{tile} is not a positive (tokens, columns) tile")

    # A traced cap keeps `final_logit_softcap` differentiable; the head is
    # held vocabulary-major so a column tile is a row slice.
    cap = None if softcap is None else jnp.asarray(softcap, jnp.float32)
    table = head_weight if vocab_major else head_weight.T
    return _bounded_head(hidden, table, targets, chunks, tile, cap,
                         head_precision(hidden.dtype, precision, bf16), predict)
