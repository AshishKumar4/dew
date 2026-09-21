"""Cross entropy that never holds the whole logits tensor.

The forward walks the vocabulary in `chunks` tiles, keeping four float32
numbers per token rather than a row of `vocab`. It is exact: logsumexp over a
concatenation is `logaddexp` of the parts', the target logit lives in exactly
one chunk, and a strict comparison over chunks taken in vocabulary order keeps
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

from typing import Callable

import jax
import jax.numpy as jnp
from flax.typing import PrecisionLike


def vocabulary_chunks(vocab_size: int, chunks: int) -> tuple[tuple[int, int], ...]:
    """`chunks` half-open column ranges covering `range(vocab_size)`.

    The last chunk is the short one when the count does not divide the
    vocabulary. Uneven tiles are free here. The loop over them is a Python
    loop, so each tile is its own matmul and nothing has to be padded to a
    common width and masked back out.
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


def head_logits(hidden, head_weight, *, softcap: float | None,
                precision: PrecisionLike) -> jax.Array:
    """`hidden @ head_weight` as the model's forward scores it: fp32 states
    against the `[features, vocab]` head, accumulated in fp32, softcapped
    when the backbone caps; `[..., vocab]`."""
    logits = jnp.einsum('...d,dv->...v', hidden.astype(jnp.float32), head_weight,
                        precision=precision, preferred_element_type=jnp.float32)
    if softcap is not None:
        cap = jnp.asarray(softcap, jnp.float32)
        logits = cap * jnp.tanh(logits / cap)
    return logits


def _chunk_terms(hidden, head_chunk, targets, start: int, stop: int,
                 softcap: float | None, precision: PrecisionLike):
    """One tile's logsumexp, target logit, best logit and its column."""
    logits = head_logits(hidden, head_chunk, softcap=softcap, precision=precision)

    inside = (targets >= start) & (targets < stop)
    column = jnp.clip(targets - start, 0, stop - start - 1)
    picked = jnp.take_along_axis(logits, column[:, None], axis=-1)[:, 0]
    return (jax.nn.logsumexp(logits, axis=-1),
            jnp.where(inside, picked, 0.0),
            jnp.max(logits, axis=-1),
            # A vocabulary column is int32 whatever the run's default integer
            # width; argmax widens to int64 under x64.
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
             softcap, precision: PrecisionLike):
    """Losses, top-1 columns and log partitions, a token tile at a time."""
    features = table.shape[1]
    flat = hidden.reshape(-1, features)
    labels = targets.reshape(-1)
    bounds = vocabulary_chunks(table.shape[0], chunks)

    def tokens(start, outputs, size):
        states = jax.lax.dynamic_slice_in_dim(flat, start, size).astype(jnp.float32)
        picked_targets = jax.lax.dynamic_slice_in_dim(labels, start, size)
        total = jnp.full((size,), -jnp.inf, jnp.float32)
        target_logit = jnp.zeros((size,), jnp.float32)
        best = jnp.full((size,), -jnp.inf, jnp.float32)
        predicted = jnp.zeros((size,), jnp.int32)
        for first, stop in bounds:
            chunk_lse, picked, chunk_best, chunk_column = _chunk_terms(
                states, table[first:stop].T, picked_targets, first, stop,
                softcap, precision)
            total = jnp.logaddexp(total, chunk_lse)
            target_logit = target_logit + picked
            better = chunk_best > best
            best = jnp.where(better, chunk_best, best)
            predicted = jnp.where(better, chunk_column, predicted)
        values = (total - target_logit, predicted, total)
        return tuple(jax.lax.dynamic_update_slice_in_dim(out, value, start, axis=0)
                     for out, value in zip(outputs, values, strict=True))

    count = labels.shape[0]
    outputs = _over_tiles((jnp.zeros((count,), jnp.float32),
                           jnp.zeros((count,), jnp.int32),
                           jnp.zeros((count,), jnp.float32)),
                          count, token_tile, tokens)
    return tuple(value.reshape(targets.shape) for value in outputs)


def _bounded_head_impl(hidden, table, targets, chunks: int, tile: tuple[int, int],
                       softcap, precision: PrecisionLike):
    """`_forward` behind a backward that recomputes its logits."""
    return _forward(hidden, table, targets, chunks, tile[0], softcap, precision)


def _bounded_head_fwd(hidden, table, targets, chunks, tile, softcap, precision):
    outputs = _forward(hidden, table, targets, chunks, tile[0], softcap, precision)
    # The residuals are the inputs and one float32 per token. Everything the
    # backward needs beyond them is a recomputed tile.
    return outputs, (hidden, table, targets, outputs[2], softcap)


def _bounded_head_bwd(chunks, tile, precision, residuals, cotangents):
    del chunks  # The backward tiles by column, not by the forward's chunks.
    hidden, table, targets, log_z, softcap = residuals
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
            table, first, width, axis=0).astype(jnp.float32)

        def tokens(start, carry, size):
            d_states, d_matrix, d_cap = carry
            states = jax.lax.dynamic_slice_in_dim(flat, start, size).astype(jnp.float32)
            token_z = jax.lax.dynamic_slice_in_dim(partitions, start, size)
            token_loss = jax.lax.dynamic_slice_in_dim(d_loss, start, size)
            token_partition = jax.lax.dynamic_slice_in_dim(d_partition, start, size)

            def project(states, matrix, cap):
                return head_logits(states, matrix.T, softcap=cap, precision=precision)

            logits, pullback = jax.vjp(project, states, matrix, softcap)
            # log Z is the whole row's, so a tile's share of the softmax needs
            # no renormalisation, and a target outside the tile one-hots to
            # zero rather than to a wrapped column.
            probabilities = jnp.exp(logits - token_z[:, None])
            selected = jax.nn.one_hot(
                jax.lax.dynamic_slice_in_dim(labels, start, size) - first, width,
                dtype=jnp.float32)
            d_logits = ((token_loss + token_partition)[:, None] * probabilities
                        - token_loss[:, None] * selected)
            states_tile, matrix_tile, cap_tile = pullback(d_logits)
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
_bounded_head = jax.custom_vjp(_bounded_head_impl, nondiff_argnums=(3, 4, 6))
_bounded_head.defvjp(_bounded_head_fwd, _bounded_head_bwd)


def chunked_cross_entropy(hidden, head_weight, targets, chunks: int, *,
                          softcap: float | None = None,
                          precision: PrecisionLike = None,
                          tile: tuple[int, int] = (1024, 8192)):
    """Per-token cross entropy of `hidden @ head_weight`, its top-1 column
    and its log partition.

    `hidden` is `[..., features]` states in any compute dtype, `head_weight`
    the `[features, vocab]` float32 head, `targets` the `[...]` int32 ids.
    Returns the per-token losses, the argmax prediction and the row's
    logsumexp (log Z, which PaLM's z-loss squares), all shaped like
    `targets`, each of the two float32 outputs carrying its own gradient. The
    caller owns the weighting and the mean, and with them the padding id.

    `softcap` is the backbone's `final_logit_softcap`. It is elementwise, so
    capping a tile and capping the row agree. `tile` is a `(tokens, columns)`
    block: the forward walks tokens in the first, the backward walks both and
    holds nothing wider, so it is the only knob on the backward's
    temporaries. The default is measured; a tile wider than the input is one
    tile.
    """
    features = head_weight.shape[0]
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
    return _bounded_head(hidden, head_weight.T, targets, chunks, tile, cap, precision)
