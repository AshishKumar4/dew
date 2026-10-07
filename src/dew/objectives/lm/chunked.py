"""Cross entropy that never holds the whole logits tensor.

The forward walks the vocabulary in `chunks` tiles, keeping four numbers per
token, six with the argmax, rather than a row of `vocab`: float32, or the
states' own dtype where it is wider (`dew.nn.precision.at_least_fp32`), as
every accumulation here is. It is
exact: logsumexp over a concatenation is `logaddexp` of the parts', the target
logit lives in exactly one chunk, and a strict comparison over chunks taken in vocabulary order keeps
the lowest index among equals, which is `jnp.argmax`'s rule on the whole row.

The backward recomputes each `[token_tile, vocab_tile]` block of logits from
the states, the head and the saved row normalizer, so neither the logits nor
their cotangent is ever wider than one tile. The full-width head gradient is
an output, not a temporary. The two-dimensional recomputing VJP is Tokamax's
(`linear_softmax_cross_entropy_loss/chunked_xla.py`, `21c5828`); the fp32
projection, the softcap, the tie rule and the differentiable log partition are
this loss's own.

The normalizer is each row's maximum and its sum of exponentials scaled to
it, not log Z, whose rounding, |log Z| ulps, would reach every probability and
the vocabulary bias's gradient, their sum. The forward walks the backward's
column tiles, as XLA rounds a product of another width otherwise. At
Qwen3-0.6B's head (2,048 bf16 tokens, 1,024 features, 151,936 fp32 columns,
four chunks) on one RTX 4080 under the trainer's compiler options, forward
and backward took 43.1 ms against 44.4 with the forward in whole chunks at a
(1024, 8192) tile, and 35.6 against 36.3 at (4096, 8192), holding 0.18 and
0.41 GiB of temporaries against 0.18 and 0.42.

First-order reverse mode only: there is no JVP rule. On one RTX 4080, fp32,
1,024 tokens by 2,816 features by 262,144 columns in four chunks, the default
tile runs the head forward and backward in 250 ms holding 1.07 GiB of
temporaries, against 249 ms and 3.55 GiB for recomputing whole chunks.
"""

import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import PrecisionLike
from jax.sharding import PartitionSpec as P

from dew.nn.kernels.generation import device_generation
from dew.nn.precision import at_least_fp32, head_product, rounded_operand, rounded_to, rounds_to_bf16
from dew.nn.protocols import AffineHead, HiddenStates, LogitsFromHidden, OutputTable
from dew.nn.sharding import logical_spec, mesh_axes


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


def _operand_dtype(precision: jax.lax.PrecisionLike, work: jnp.dtype):
    """The dtype the head's operands take: bf16 under the bf16 algorithm, the
    work dtype (`at_least_fp32` of the states') otherwise."""
    return jnp.bfloat16 if precision is BF16 else work


def _logits_cotangent(cotangent, precision: jax.lax.PrecisionLike):
    """The logits' cotangent as both backward products read it.

    Under the bf16 algorithm the logits are bf16 values (`_tile_logits`),
    and their cotangent is rounded to bf16 once, as the gradient of torch
    autocast's bf16 logits is: the state product and the head's own read
    that one copy. Otherwise it stays in its own dtype, fp32 or wider. The
    cast is round-to-nearest-even, the rounding a GPU's BF16_BF16_F32
    applies to an fp32 operand; a CPU runs the algorithm unrounded, so there
    the cast is the rounding."""
    return cotangent.astype(jnp.bfloat16) if precision is BF16 else cotangent


def _capped(logits, softcap, temperature: float = 1.0):
    """Softcapped logits over a sampling temperature, which an engine applies
    to what the model's head returns."""
    if softcap is not None:
        cap = jnp.asarray(softcap, logits.dtype)
        logits = cap * jnp.tanh(logits / cap)
    return logits / temperature


def _biased_logits(logits, bias):
    """The vocabulary bias after the product's rounding, in fp32 or wider."""
    return logits if bias is None else logits + jnp.asarray(bias, logits.dtype)


def head_table(model: nn.Module, variables) -> OutputTable | None:
    """`model`'s head over `variables` as one matrix (`AffineHead.output_table`),
    or None for a model whose head no matrix alone is, or that names none."""
    table = model.apply(variables, method="output_table") if isinstance(model, AffineHead) else None
    if table is not None and not isinstance(table, OutputTable):
        raise TypeError(f"{type(model).__name__}.output_table gives a {type(table).__name__}, "
                        f"not an OutputTable")
    return table


def reads_states(model) -> bool:
    """Whether `model` scores its final states (`HiddenStates`) through a head
    that takes them: a matrix (`AffineHead`) or its exact head
    (`LogitsFromHidden`). One that does not gives its logits only from its
    tokens (`Logits`), which a loss then scores whole (`logits_cross_entropy`)."""
    return isinstance(model, HiddenStates) and isinstance(model, AffineHead | LogitsFromHidden)


def model_logits(model, variables, hidden) -> jax.Array:
    """`model`'s logits of its final states `hidden` over `variables`: the
    matrix `AffineHead.output_table` gives, contracted as the chunked head
    contracts it (`head_logits`), or the model's exact head
    (`LogitsFromHidden`) where no matrix alone is the head."""
    table = head_table(model, variables)
    if table is None:
        return model.apply(variables, hidden, method="logits_from_hidden")
    return head_logits(hidden, table.matrix, softcap=table.softcap, precision=table.precision,
                       vocab_major=table.vocab_major, bias=table.bias)


def head_logits(hidden, head_weight, *, softcap: float | None,
                precision: PrecisionLike, vocab_major: bool = False,
                temperature: float = 1.0, bias=None) -> jax.Array:
    """`hidden @ head_weight` as the model's forward scores it: the states
    against the `[features, vocab]` head (`[vocab, features]` with
    `vocab_major`), accumulated in fp32, softcapped when the backbone caps;
    `[..., vocab]` fp32. The product follows the states' dtype
    (`dew.nn.precision.head_product`)."""
    logits = head_product('...d,vd->...v' if vocab_major else '...d,dv->...v',
                           hidden, head_weight, precision)
    return _capped(_biased_logits(logits, bias), softcap, temperature)


def _without(logits, first, excluded):
    """`logits` of the columns from `first` on, with column `excluded` (a
    traced id) given no mass: -inf, so its probability is 0. With None the
    logits pass untouched and the program holds no comparison."""
    if excluded is None:
        return logits
    columns = first + jax.lax.broadcasted_iota(jnp.int32, logits.shape, logits.ndim - 1)
    return jnp.where(columns == excluded, -jnp.inf, logits)


def _tile_logits(states, matrix, precision: jax.lax.PrecisionLike):
    """Uncapped `[tokens, features] @ [columns, features].T` in at least fp32,
    over operands already in the dtype the head multiplies in. Under the
    bf16 algorithm the logits are rounded to bf16 values, as
    `dew.nn.precision.head_product` rounds the model's."""
    logits = jnp.einsum('td,vd->tv', states, matrix.astype(states.dtype),
                        precision=precision, preferred_element_type=at_least_fp32(states.dtype))
    return rounded_to(logits, jnp.bfloat16) if precision is BF16 else logits


def _rescaled(maximum, total, larger):
    """A sum of exponentials moved from its own maximum to the larger one;
    unchanged where they agree, so two sums whose maximum is -inf add rather
    than turn NaN."""
    return jnp.where(maximum == larger, total, total * jnp.exp(maximum - larger))


def _row_terms(logits, predict: bool) -> tuple[
        jax.Array, tuple[jax.Array, jax.Array] | None, tuple[jax.Array, jax.Array]]:
    """Each row's logsumexp, with `predict` its maximum and the first
    column that holds it, and its maximum and the sum of exponentials scaled
    to it, from one reduction over the row.

    The running maximum, the sum of exponentials scaled to it and the
    maximum's column ride together through `jax.lax.reduce`, so the row is
    read once, rather than once for the maximum and again for the sum. On an
    RTX 4080 (jax 0.11.2), fp32 logits of 4096 x 151936 take 3.9 ms against
    `jax.nn.logsumexp` and `jnp.argmax`'s 7.6, and 8192 x 50304 2.6 against
    5.0, with the same columns and logsumexps 1.9e-6 apart."""
    def combine(first, second):
        larger = jnp.maximum(first[0], second[0])
        total = _rescaled(first[0], first[1], larger) + _rescaled(second[0], second[1], larger)
        if not predict:
            return larger, total
        keep = (first[0] > second[0]) | ((first[0] == second[0]) & (first[2] < second[2]))
        return larger, total, jnp.where(keep, first[2], second[2])

    axis = logits.ndim - 1
    operands = [logits, jnp.ones_like(logits)]
    initial = [jnp.array(-jnp.inf, logits.dtype), jnp.array(0, logits.dtype)]
    if predict:
        # A vocabulary column is int32 whatever the run's default integer width.
        operands.append(jax.lax.broadcasted_iota(jnp.int32, logits.shape, axis))
        initial.append(jnp.array(jnp.iinfo(jnp.int32).max, jnp.int32))
    terms = jax.lax.reduce(tuple(operands), tuple(initial), combine, (axis,))
    return terms[0] + jnp.log(terms[1]), ((terms[0], terms[2]) if predict else None), terms[:2]


class _ChunkTerms(NamedTuple):
    """One tile's logsumexp and target logit, its best logit and column when
    a prediction is asked for, and its maximum and the sum of exponentials
    scaled to it."""
    lse: jax.Array
    picked: jax.Array
    best: jax.Array | None
    column: jax.Array | None
    peak: jax.Array
    mass: jax.Array


def _chunk_terms(hidden, head_chunk, targets, start: int, stop: int,
                 softcap: float | None, precision: jax.lax.PrecisionLike,
                 predict: bool, temperature: float, excluded, bias=None) -> _ChunkTerms:
    """One tile's `_ChunkTerms`."""
    logits = _without(_capped(_biased_logits(_tile_logits(hidden, head_chunk, precision), bias),
                              softcap, temperature), start, excluded)

    inside = (targets >= start) & (targets < stop)
    column = jnp.clip(targets - start, 0, stop - start - 1)
    picked = jnp.where(inside, jnp.take_along_axis(logits, column[:, None], axis=-1)[:, 0], 0.0)
    lse, top, normalizer = _row_terms(logits, predict)
    if top is None:
        return _ChunkTerms(lse, picked, None, None, *normalizer)
    return _ChunkTerms(lse, picked, top[0], top[1] + start, *normalizer)


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


def _forward(hidden, table, targets, excluded, chunks: int, tile: tuple[int, int],
             softcap, precision: jax.lax.PrecisionLike, predict: bool, temperature: float, bias=None):
    """Losses, top-1 columns (None unless `predict`) and log partitions, a
    token tile at a time, and apart from them each row's maximum and the sum
    of its exponentials scaled to it."""
    features = table.shape[1]
    flat = hidden.reshape(-1, features)
    labels = targets.reshape(-1)
    token_tile, width = tile[0], min(tile[1], vocabulary_chunks(table.shape[0], chunks)[0][1])
    work = at_least_fp32(hidden.dtype)
    operands = _operand_dtype(precision, work)
    # Once, not per token tile: every tile multiplies the same rounded table,
    # and a conversion inside the loop reads the fp32 table once per tile.
    table = table.astype(operands)

    def tokens(start, outputs, size):
        states = jax.lax.dynamic_slice_in_dim(flat, start, size).astype(operands)
        picked_targets = jax.lax.dynamic_slice_in_dim(labels, start, size)

        def columns(first, carry, count):
            terms = _chunk_terms(
                states, jax.lax.dynamic_slice_in_dim(table, first, count),
                picked_targets, first, first + count, softcap, precision, predict, temperature, excluded,
                None if bias is None else jax.lax.dynamic_slice_in_dim(bias, first, count))
            peak = jnp.maximum(carry[2], terms.peak)
            merged = (jnp.logaddexp(carry[0], terms.lse), carry[1] + terms.picked, peak,
                      _rescaled(carry[2], carry[3], peak) + _rescaled(terms.peak, terms.mass, peak))
            if terms.best is None or terms.column is None:
                return merged
            better = terms.best > carry[4]
            return (*merged, jnp.where(better, terms.best, carry[4]),
                    jnp.where(better, terms.column, carry[5]))

        initial = (jnp.full((size,), -jnp.inf, work), jnp.zeros((size,), work)) * 2
        if predict:
            initial += (jnp.full((size,), -jnp.inf, work), jnp.zeros((size,), jnp.int32))
        total, target_logit, *rest = _over_tiles(initial, table.shape[0], width, columns)
        values = (total - target_logit, total, *rest[:2], *rest[3:])
        return tuple(jax.lax.dynamic_update_slice_in_dim(out, value, start, axis=0)
                     for out, value in zip(outputs, values, strict=True))

    count = labels.shape[0]
    outputs = (jnp.zeros((count,), work),) * 4
    if predict:
        outputs += (jnp.zeros((count,), jnp.int32),)
    losses, log_z, peak, mass, *predicted = (value.reshape(targets.shape) for value in
                                             _over_tiles(outputs, count, token_tile, tokens))
    return (losses, predicted[0] if predict else None, log_z), (peak, mass)


def _bounded_head_impl(hidden, table, targets, excluded, chunks: int, tile: tuple[int, int],
                       softcap, precision: jax.lax.PrecisionLike, predict: bool, temperature: float, bias):
    """Run `_forward` behind a backward that recomputes its logits."""
    return _forward(hidden, table, targets, excluded, chunks, tile, softcap, precision, predict,
                    temperature, bias)[0]


def _bounded_head_fwd(hidden, table, targets, excluded, chunks, tile, softcap, precision, predict,
                      temperature, bias):
    outputs, normalizer = _forward(hidden, table, targets, excluded, chunks, tile, softcap, precision,
                                   predict, temperature, bias)
    # The residuals are the inputs and two numbers per token. Everything the
    # backward needs beyond them is a recomputed tile.
    return outputs, (hidden, table, targets, excluded, normalizer, softcap, bias)


def _bounded_head_bwd(chunks, tile, precision, predict, temperature, residuals, cotangents):
    """Pull the cotangents back through logits recomputed one tile at a time.

    The outer loop walks vocabulary tiles carrying `(d_states, d_table,
    d_cap)`: the full-width state gradient, the head gradient stored tile by
    tile, and the softcap's. The inner loop walks token tiles carrying
    `(d_states, d_matrix, d_cap)`, where `d_matrix` accumulates one
    vocabulary tile's head gradient in fp32 before it is stored.
    """
    del predict  # Argmax has no gradient.
    hidden, table, targets, excluded, normalizer, softcap, bias = residuals
    work = at_least_fp32(hidden.dtype)
    operands = _operand_dtype(precision, work)
    loss_cotangent, _, partition_cotangent = cotangents
    # The logits are recomputed from operands that hold the forward's values,
    # in the forward's column tiles, so they are the logits its normalizer
    # summed: XLA rounds a product of another width otherwise.
    token_tile, vocab_tile = tile[0], min(tile[1], vocabulary_chunks(table.shape[0], chunks)[0][1])
    features = table.shape[1]
    flat = hidden.reshape(-1, features)
    labels = targets.reshape(-1)
    peaks, masses = (value.reshape(-1) for value in normalizer)
    d_loss = loss_cotangent.reshape(-1)
    d_partition = partition_cotangent.reshape(-1)
    count = labels.shape[0]

    def columns(first, gradients, width):
        d_states, d_table, d_cap, d_bias = gradients
        stored = jax.lax.dynamic_slice_in_dim(table, first, width, axis=0)
        matrix = rounded_operand(stored.astype(work), operands)
        # The state gradient's operand, cast to bf16 once per vocabulary tile
        # rather than once per token tile: the same values, as `matrix` is
        # exact in bf16 under the bf16 algorithm.
        product_matrix = stored.astype(jnp.bfloat16) if precision is BF16 else matrix
        bias_tile = None if bias is None else jax.lax.dynamic_slice_in_dim(bias, first, width)

        def tokens(start, carry, size):
            d_states, d_matrix, d_cap, d_offset = carry
            states = rounded_operand(jax.lax.dynamic_slice_in_dim(
                flat, start, size).astype(work), operands)
            token_peak = jax.lax.dynamic_slice_in_dim(peaks, start, size)
            token_mass = jax.lax.dynamic_slice_in_dim(masses, start, size)
            token_loss = jax.lax.dynamic_slice_in_dim(d_loss, start, size)
            token_partition = jax.lax.dynamic_slice_in_dim(d_partition, start, size)

            logits, pullback = jax.vjp(
                lambda raw, cap: _capped(raw, cap, temperature),
                _biased_logits(_tile_logits(states, matrix, precision), bias_tile), softcap)
            # The maximum and the sum are the whole row's, so a tile's share of
            # the softmax needs no renormalisation, and a target outside the
            # tile one-hots to zero rather than to a wrapped column.
            probabilities = (jnp.exp(_without(logits, first, excluded) - token_peak[:, None])
                             / token_mass[:, None])
            selected = jax.nn.one_hot(
                jax.lax.dynamic_slice_in_dim(labels, start, size) - first, width,
                dtype=work)
            d_logits = ((token_loss + token_partition)[:, None] * probabilities
                        - token_loss[:, None] * selected)
            d_raw, cap_tile = pullback(d_logits)
            # Bias is added after the bf16 product rounds. Its cotangent
            # stays fp32; only the product's two backward operands round.
            if d_offset is not None:
                d_offset = d_offset + jnp.sum(d_raw, axis=0)
            d_raw = _logits_cotangent(d_raw, precision)
            states_tile = jnp.einsum('tv,vd->td', d_raw, product_matrix, precision=precision,
                                     preferred_element_type=work)
            matrix_tile = jnp.einsum(
                'tv,td->vd', d_raw,
                jax.lax.dynamic_slice_in_dim(flat, start, size).astype(operands), precision=precision,
                preferred_element_type=work)
            prior = jax.lax.dynamic_slice_in_dim(d_states, start, size)
            d_states = jax.lax.dynamic_update_slice_in_dim(
                d_states, prior + states_tile, start, axis=0)
            if d_cap is not None:
                d_cap = d_cap + cap_tile
            return d_states, d_matrix + matrix_tile, d_cap, d_offset

        d_states, d_matrix, d_cap, d_offset = _over_tiles(
            (d_states, jnp.zeros((width, features), work), d_cap,
             None if bias is None else jnp.zeros((width,), work)),
            count, token_tile, tokens)
        # fp32 until every token tile is in, so a bf16 head does not round
        # once per tile.
        d_table = jax.lax.dynamic_update_slice_in_dim(
            d_table, d_matrix.astype(table.dtype), first, axis=0)
        if d_bias is not None:
            assert d_offset is not None
            d_bias = jax.lax.dynamic_update_slice_in_dim(d_bias, d_offset, first, axis=0)
        return d_states, d_table, d_cap, d_bias

    d_states, d_table, d_cap, d_bias = _over_tiles(
        (jnp.zeros(flat.shape, work),
         jnp.zeros(table.shape, table.dtype),
         None if softcap is None else jnp.zeros_like(softcap),
         None if bias is None else jnp.zeros(bias.shape, work)),
        table.shape[0], vocab_tile, columns)
    if bias is not None:
        assert d_bias is not None
        d_bias = d_bias.astype(bias.dtype)
    return d_states.reshape(hidden.shape).astype(hidden.dtype), d_table, None, None, d_cap, d_bias


# `jax.custom_vjp` is generic in its return type, and a `functools.partial`
# decorator loses that binding, so it is built by hand.
_bounded_head = jax.custom_vjp(_bounded_head_impl, nondiff_argnums=(4, 5, 7, 8, 9))
_bounded_head.defvjp(_bounded_head_fwd, _bounded_head_bwd)


def _whole_head_terms(hidden, table, targets, excluded, softcap, precision, predict, temperature, bias):
    """The whole `[tokens, vocab]` logits in at least fp32 and what
    `_forward` returns of them: losses, the top-1 column (None unless
    `predict`), log Z."""
    operands = _operand_dtype(precision, at_least_fp32(hidden.dtype))
    flat = hidden.reshape(-1, table.shape[1]).astype(operands)
    labels = targets.reshape(-1)
    raw = _biased_logits(_tile_logits(flat, table.astype(operands), precision), bias)
    logits = _without(_capped(raw, softcap, temperature), 0, excluded)
    log_z, top, _ = _row_terms(logits, predict)
    # A target outside the vocabulary picks no column, as in `_chunk_terms`:
    # the vocabulary-split head shifts each shard's targets by its first row
    # and sums the picked logit over shards, so only the owner may add one.
    inside = (labels >= 0) & (labels < table.shape[0])
    picked = jnp.where(inside, jnp.take_along_axis(
        logits, jnp.clip(labels, 0, table.shape[0] - 1)[:, None], axis=-1)[:, 0], 0.0)
    best = None if top is None else top[1].reshape(targets.shape)
    return raw, ((log_z - picked).reshape(targets.shape), best, log_z.reshape(targets.shape))


def _whole_head_impl(hidden, table, targets, excluded, softcap, chunks, precision, predict,
                     temperature, bias):
    """With nothing to differentiate, the tiled forward: no backward needs
    the whole logits, so an evaluation or a scoring pass stays bounded."""
    return _forward(hidden, table, targets, excluded, chunks, (1024, table.shape[0]), softcap, precision,
                    predict, temperature, bias)[0]


def _whole_head_fwd(hidden, table, targets, excluded, softcap, chunks, precision, predict, temperature, bias):
    del chunks
    raw, outputs = _whole_head_terms(hidden, table, targets, excluded, softcap, precision, predict,
                                     temperature, bias)
    # The fp32 logits before the cap are kept, so the backward multiplies
    # nothing it did not have to: the state and head products alone.
    return outputs, (hidden, table, targets, excluded, raw, outputs[2], softcap, bias)


def _whole_head_bwd(chunks, precision, predict, temperature, residuals, cotangents):
    """The same gradient `_bounded_head_bwd` takes tile by tile, from the kept
    logits: the logits' cotangent as `_logits_cotangent` gives it to both
    products, and the head's gradient accumulated in fp32 once."""
    del chunks, predict
    hidden, table, targets, excluded, raw, log_z, softcap, bias = residuals
    loss_cotangent, _, partition_cotangent = cotangents
    work = at_least_fp32(hidden.dtype)
    operands = _operand_dtype(precision, work)
    features = table.shape[1]
    d_loss, d_partition = loss_cotangent.reshape(-1), partition_cotangent.reshape(-1)
    logits, pullback = jax.vjp(lambda raw, cap: _capped(raw, cap, temperature), raw, softcap)
    probabilities = jnp.exp(_without(logits, 0, excluded) - log_z.reshape(-1)[:, None])
    selected = jax.nn.one_hot(targets.reshape(-1), table.shape[0], dtype=work)
    d_raw, d_cap = pullback((d_loss + d_partition)[:, None] * probabilities
                            - d_loss[:, None] * selected)
    d_bias = None if bias is None else jnp.sum(d_raw, axis=0).astype(bias.dtype)
    d_raw = _logits_cotangent(d_raw, precision)
    d_states = jnp.einsum('tv,vd->td', d_raw, table.astype(operands), precision=precision,
                          preferred_element_type=work)
    d_table = jnp.einsum('tv,td->vd', d_raw, hidden.reshape(-1, features).astype(operands),
                         precision=precision, preferred_element_type=work)
    return (d_states.reshape(hidden.shape).astype(hidden.dtype), d_table.astype(table.dtype),
            None, None, d_cap, d_bias)


_whole_head = jax.custom_vjp(_whole_head_impl, nondiff_argnums=(5, 6, 7, 8))
_whole_head.defvjp(_whole_head_fwd, _whole_head_bwd)


HEAD_TILE_BY_GENERATION: dict[str, tuple[int, int]] = {'sm80': (4096, 8192), 'sm89': (4096, 8192)}
"""The chunked head's `(tokens, columns)` tile where it was measured to beat
the default. On one A100, Qwen3-0.6B's head (vocabulary 151936, 1024 wide,
4096 tokens, bf16) forward and backward took 48.7 ms at 4096 x 8192 and
60.2 ms at 1024 x 8192, at 0.46 and 0.21 GiB of temporaries. On an RTX
4080, a whole training step at those widths (2 layers, bf16): at 4096
tokens 114.6 ms against 123.4 at 1024 x 8192, 119.7 at 2048 x 8192 and
116.6 at 4096 x 4096; at 8192 tokens 219.1 against 228.3 (1024 x 8192),
227.8 (2048 x 16384) and 219.3 (8192 x 4096) with less memory. The whole
logits beat the 4096 x 8192 tile where they fit once the step compiles
without XLA's Triton GEMM fusions (`TRITON_GEMM_OFF_GENERATIONS`): 53.5
against 65.5 ms at 2048 tokens, 93.4 against 110.2 at 4096. With the
fusions, exactly 4096 whole took 286.0 ms."""


def chunked_tile() -> tuple[int, int]:
    """The chunked head's tile on the pool's hardware generation: read off
    global device 0 (`device_generation`), which every process sees alike, so
    a pool's processes agree on the tile and compile one program."""
    return HEAD_TILE_BY_GENERATION.get(device_generation(), (1024, 8192))


def _token_spec(shape: tuple[int, ...]) -> P:
    """How the loss splits `[batch, length, ...]` targets over the mesh in
    context: rows and positions where the rule table puts the hidden's, then
    over every other axis whose shards the tokens still divide into, the
    minor end of the rows first. A token's loss reads no other token, so
    splitting the tokens further only slices a replicated operand."""
    mesh = jax.sharding.get_abstract_mesh()
    names = ("activation_batch", "activation_length", *[None] * len(shape))[:len(shape)]
    spec = logical_spec(names, shape)
    entries = [list(mesh_axes(entry)) for entry in (*spec, *[None] * (len(shape) - len(spec)))]
    for axis in mesh.axis_names:
        if (mesh.shape[axis] == 1 or axis in mesh.manual_axes
                or any(axis in entry for entry in entries)):
            continue
        for size, entry in zip(shape[:2], entries, strict=False):
            if size % (math.prod(mesh.shape[name] for name in entry) * mesh.shape[axis]) == 0:
                entry.append(axis)
                break
    return P(*(entry[0] if len(entry) == 1 else tuple(entry) or None for entry in entries))


def chunked_cross_entropy(hidden, head_weight, targets, chunks: int, *,
                          softcap: float | None = None,
                          precision: PrecisionLike = None,
                          tile: tuple[int, int] | None = (1024, 8192),
                          vocab_major: bool = False,
                          predict: bool = True, temperature: float = 1.0,
                          excluded: int | None = None, bias=None):
    """Per-token cross entropy of `hidden @ head_weight`, its top-1 column
    and its log partition.

    `hidden` is `[..., features]` states in any compute dtype, `head_weight`
    the `[features, vocab]` head in its stored dtype (each tile upcasts its
    own slice; the products are accumulated in fp32), `targets` the `[...]`
    int32 ids. Returns the per-token losses, the argmax prediction (None
    when `predict` is False, which skips the argmax: 0.77 ms of the head's
    8.0 on a TPU v6e) and the row's logsumexp (log Z, which PaLM's z-loss
    squares), all shaped like `targets`, each of the two float outputs carrying its own gradient. The
    caller owns the weighting and the mean, and with them the padding id.

    The backward keeps the head it was given. With `vocab_major` the head
    is `[vocab, features]`, the way a tied embedding table is stored, and
    the value kept is the parameter itself; a `[features, vocab]` head is
    transposed here first, and what the backward keeps is that transposed
    array, a vocabulary-sized copy of a tied table (`head_table` on the
    backbone hands out the stored orientation).

    The product follows the states' dtype, forward and backward: bf16
    states at the default precision multiply the head as bf16 and accumulate
    in fp32, and fp32 states multiply it as stored. The softmax, the
    logsumexp and the loss are fp32 either way, or the states' dtype where
    it is wider (a float64 reference's).

    `softcap` is the backbone's `final_logit_softcap`. It is elementwise, so
    capping a tile and capping the row agree. `tile` is a `(tokens, columns)`
    block: both passes walk tokens in the first and columns in the second
    or a chunk, whichever is narrower, and the backward holds nothing
    wider, so it is the only knob on the backward's temporaries. The
    default is measured; a tile wider than the input is one
    tile. A `tile` of None keeps the whole fp32 logits for the backward
    instead of recomputing them, which is faster wherever they fit: on one
    A100, Qwen3-0.6B's head at 4096 tokens took 33.1 ms with the logits
    held, at 4.6 GiB, against 60.2 ms tiled (`LMObjective.head_tile`).
    `temperature` divides the capped logits (`head_logits`), which
    scores the draws of a sampler at that temperature. `excluded` is a
    column the distribution gives no mass, left out of the partition and the
    prediction in every tile: a masked diffusion model's mask token, which
    MDLM's SUBS parameterization never predicts. A target there scores +inf.
    At None, the default, the compiled program holds no trace of it.

    On a mesh every device scores its own tokens (`_token_spec`): the token
    tiles are dynamic slices in a loop, which GSPMD would only compute on a
    replicated operand, so the split is a `shard_map`. Where the head's
    vocabulary is split over axes that split the tokens too, and the tokens
    of such a group take fewer bytes than the head, the head stays split
    (`_vocabulary_split`): each device scores every token of its group
    against its own columns, and only the tokens, their per-token terms and
    the tokens' gradient cross devices. Otherwise the head is gathered whole
    on each device, and its gradient is the one sum that crosses devices.
    """
    if bias is not None:
        bias = jnp.asarray(bias, at_least_fp32(hidden.dtype))
        if bias.shape != (head_weight.shape[0 if vocab_major else 1],):
            raise ValueError('bias must have one entry per vocabulary column')
    features = head_weight.shape[1 if vocab_major else 0]
    if hidden.shape[-1] != features:
        raise ValueError(
            f"hidden states are {hidden.shape[-1]} wide and the head is {features}")
    if hidden.shape[:-1] != targets.shape:
        raise ValueError(
            f"{targets.shape} targets for {hidden.shape[:-1]} hidden states")
    if tile is not None and (len(tile) != 2 or min(tile) < 1):
        raise ValueError(f"{tile} is not a positive (tokens, columns) tile")

    # A traced cap keeps `final_logit_softcap` differentiable; the head is
    # held vocabulary-major so a column tile is a row slice.
    cap = None if softcap is None else jnp.asarray(softcap, at_least_fp32(hidden.dtype))
    table = head_weight if vocab_major else head_weight.T
    product = BF16 if rounds_to_bf16(hidden.dtype, precision) else precision

    def head(hidden, table, targets, excluded, cap, bias):
        if tile is None:
            return _whole_head(hidden, table, targets, excluded, cap, chunks, product, predict,
                               float(temperature), bias)
        return _bounded_head(hidden, table, targets, excluded, chunks, tile, cap, product,
                             predict, float(temperature), bias)

    def column_logits(states, rows, cap, bias):
        """The capped logit of each state against its own row of the head,
        as the head's tiles score it."""
        effective = BF16 if rounds_to_bf16(states.dtype, precision) else precision
        operands = _operand_dtype(effective, at_least_fp32(states.dtype))
        logits = jax.vmap(lambda state, row: _tile_logits(
            state[None].astype(operands), row[None], effective)[0, 0])(states, rows)
        return _capped(_biased_logits(logits, bias), cap, temperature)

    left_out = None if excluded is None else jnp.asarray(excluded, jnp.int32)
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return head(hidden, table, targets, left_out, cap, bias)
    return _sharded_head(mesh, hidden, table, targets, left_out, cap, bias, head, column_logits,
                         predict=predict)


def head_cross_entropy(model, variables, hidden, targets, chunks: int, *,
                       tile: tuple[int, int] | None = (1024, 8192), predict: bool = True,
                       temperature: float = 1.0, excluded: int | None = None):
    """`chunked_cross_entropy` of `hidden` against `model`'s head over `variables`.

    The head is the matrix `AffineHead.output_table` gives, its bias,
    softcap and precision included. Where no matrix alone is the head (a
    prediction head past the final states, an adapter's factors on it), the
    model's exact logits (`LogitsFromHidden`) are scored whole: that holds
    the fp32 `[..., vocab]` logits for the backward pass, a vocabulary-sized
    row per token, which the tiled head never does.
    """
    table = head_table(model, variables)
    if table is not None:
        return chunked_cross_entropy(
            hidden, table.matrix, targets, chunks, softcap=table.softcap, precision=table.precision,
            tile=tile, vocab_major=table.vocab_major, predict=predict, temperature=temperature,
            excluded=excluded, bias=table.bias)
    return logits_cross_entropy(model.apply(variables, hidden, method="logits_from_hidden"), targets,
                                predict=predict, temperature=temperature, excluded=excluded)


def logits_cross_entropy(logits, targets, *, predict: bool = True, temperature: float = 1.0,
                         excluded: int | None = None):
    """`chunked_cross_entropy`'s per-token losses, top-1 columns and log
    partitions, of whole `[..., vocab]` logits, in fp32 and over `temperature`,
    with column `excluded` given no mass."""
    logits = _without(logits.astype(jnp.float32) / temperature, 0,
                      None if excluded is None else jnp.asarray(excluded, jnp.int32))
    log_z = jax.nn.logsumexp(logits, axis=-1)
    picked = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return log_z - picked, jnp.argmax(logits, axis=-1) if predict else None, log_z


def _sharded_head(mesh, hidden, table, targets, excluded, cap, bias, head, column_logits, *, predict: bool):
    """`head` on a mesh: each device scores its own tokens in a `shard_map`,
    against the head's own columns where they stay split (`_vocabulary_split`)
    or the head gathered whole (`chunked_cross_entropy`)."""
    spec = _token_spec(targets.shape)
    axes = {axis for entry in spec for axis in mesh_axes(entry)}
    if not axes:
        return head(hidden, table, targets, excluded, cap, bias)
    # The head arrives in its own shards on the token axes and is gathered
    # whole inside, so its gradient leaves reduce-scattered onto them rather
    # than summed whole. Unchecked, the transpose sums the cotangents of
    # what reaches every token shard alike, the head over the axes that do
    # not split it and the cap, over the shards; checked, the tile loops'
    # carries, which start from constants, would have to be told they vary.
    # A spec drops its trailing unsplit dimensions; both of the table's count.
    entries = logical_spec(("vocab", "embed"), table.shape)
    kept = [tuple(axis for axis in mesh_axes(entry) if axis in axes)
            for entry in (*entries, *[None] * (2 - len(entries)))]
    held = P(*(entry if entry else None for entry in kept))
    group, widths = kept
    size = math.prod(mesh.shape[axis] for axis in group)
    split = bool(group) and (hidden.size // math.prod(
        mesh.shape[axis] for axis in axes) * size * hidden.dtype.itemsize
        < table.size * table.dtype.itemsize)

    def local(hidden, table, targets, excluded, cap, bias):
        if widths:
            table = jax.lax.all_gather(table, widths, axis=1, tiled=True)
        if split:
            return _vocabulary_split(hidden, table, targets, excluded, cap, bias, group, head, column_logits)
        if group:
            table = jax.lax.all_gather(table, group, axis=0, tiled=True)
            if bias is not None:
                bias = jax.lax.all_gather(bias, group, axis=0, tiled=True)
        return head(hidden, table, targets, excluded, cap, bias)

    # The map holds every axis manual, those that split no token too: in a map
    # that leaves axes automatic, JAX lowers a collective's reducer with its
    # add wrapped in a sharding constraint, and XLA's CPU compiler, which
    # widens a bf16 reduction to fp32 (AllReducePromotion), aborts on a
    # reducer whose root is not the add. Under a bf16 policy the split head's
    # states come back through such a reduction. The states and the head name
    # the axes of one device on their first dimension, where they split
    # nothing, so the transpose sums neither over them. An axis whose shards
    # do not divide the tokens leaves both whole over it, as the tile loops
    # compute them anyway, and the transpose sums the copies' shares.
    alone = tuple(axis for axis in mesh.axis_names if mesh.shape[axis] == 1 and axis not in mesh.manual_axes)

    def naming(entry):
        return (*mesh_axes(entry), *alone) or None

    return jax.shard_map(local, in_specs=(P(naming(spec[0]), *spec[1:], None), P(naming(held[0]), *held[1:]),
                                          spec, P(), P(), None if bias is None else P(naming(held[0]))),
                         out_specs=(spec, spec if predict else None, spec),
                         axis_names=set(mesh.axis_names) - set(mesh.manual_axes), check_vma=False)(
        hidden, table, targets, excluded, cap, bias)


def _vocabulary_split(hidden, table, targets, excluded, cap, bias, group: tuple[str, ...],
                       head, column_logits):
    """`head` inside a `shard_map` over a vocabulary split `group` ways: every
    token of the group against this device's rows of the head, the per-token
    terms combined over the group, and this device's own tokens returned.

    Megatron-LM's parallel cross entropy (Shoeybi et al., 2019). The log
    partition is the logsumexp of the shards' own, the
    target's logit the sum of theirs (one shard holds it), the prediction
    the best of their best columns. Differentiated through the shard_map as
    written, and correct unchecked: after the combine each device keeps its
    own tokens, so a psum's cotangent, summed over the group, is every
    device's share of it, and the gather's transpose scatters the states'
    gradient back to the devices that hold them. The head's gradient never
    leaves its device.
    """
    features = hidden.shape[-1]
    count = targets.size
    states = jax.lax.all_gather(hidden.reshape(count, features), group, axis=0, tiled=True)
    offset = jax.lax.axis_index(group) * table.shape[0]
    labels = jax.lax.all_gather(targets.reshape(count), group, axis=0, tiled=True) - offset
    # The excluded column, like the targets, in this device's own numbering.
    local_excluded = None if excluded is None else excluded - offset
    losses, predicted, log_z = head(states, table, labels, local_excluded, cap, bias)
    # Stopped before the max: pmax has no derivative rule, and a stop after it
    # still differentiates it.
    peak = jax.lax.pmax(jax.lax.stop_gradient(log_z), group)
    whole = peak + jnp.log(jax.lax.psum(jnp.exp(log_z - peak), group))
    target = jax.lax.psum(log_z - losses, group)
    start = jax.lax.axis_index(group) * count

    def own(value):
        return jax.lax.dynamic_slice_in_dim(value, start, count).reshape(targets.shape)

    if predicted is not None:
        best = jax.lax.stop_gradient(column_logits(
            states, jnp.take(table, predicted, axis=0), cap,
            None if bias is None else jnp.take(bias, predicted, axis=0)))
        top = jax.lax.pmax(best, group)
        # The lowest column among the shards that reach the best logit, as an
        # argmax over the whole row picks the first.
        predicted = own(jax.lax.pmin(jnp.where(best == top, predicted + offset,
                                               jnp.iinfo(jnp.int32).max), group))
    return own(whole - target), predicted, own(whole)


SUPPORT_BLOCK = 1 << 15
"""Kept ids one rematerialized step of `support_log_probs` scores, over all rows."""


def support_log_probs(hidden, head_weight, targets, support_ids, support_columns, *,
                      temperature: float = 1.0, softcap: float | None = None,
                      precision: PrecisionLike = None, vocab_major: bool = False, bias=None):
    """Each target's log-probability renormalized over its recorded sampling support.

    Keep-sampling-mask (DeepSeek-V3.2 section 3.1; slime 5bae5bb `loss.py`
    `_build_topp_keep_mask` masks the tempered logits outside the rollout's
    top-p set to -inf before its log-softmax): a target drawn by a top-k or
    top-p sampler scores `l_t - logsumexp_{v in S} l_v` with `l` the capped
    logits over `temperature`, the engine's filtered log-probability.

    `hidden` is `[B, S, features]` target states and `targets` `[B, S]` ids.
    The support is ragged within each row: `support_ids` `[B, C]` holds a
    row's kept ids back to back and `support_columns` `[B, C]` the column of
    the target each belongs to, both -1 on padding, so the arrays shard with
    their rows. Only those columns of the head are scored, `SUPPORT_BLOCK`
    entries at a time and rematerialized in the backward pass. `head_weight`
    is `[features, vocab]`, or `[vocab, features]` with `vocab_major`, as in
    `chunked_cross_entropy`. Returns the log-probs, `-inf` for a target
    outside its support, and whether each target had one; a target with none
    scores 0.0 here.
    """
    table = jnp.asarray(head_weight) if vocab_major else jnp.asarray(head_weight).T
    kept = support_ids.shape[1]
    block = max(1, min(kept, SUPPORT_BLOCK // targets.shape[0]))
    blocks = -(-kept // block)
    pad = blocks * block - kept
    ids = jnp.pad(support_ids, ((0, 0), (0, pad)), constant_values=-1)
    columns = jnp.pad(support_columns, ((0, 0), (0, pad)), constant_values=-1)

    @jax.checkpoint
    def chunk(args):
        chosen, owner = args
        state = jnp.take_along_axis(hidden, jnp.maximum(owner, 0)[..., None], axis=1)
        # The full head's product (`head_product`), over the kept rows only.
        logits = head_product('bcd,bcd->bc', state, table[jnp.maximum(chosen, 0)], precision)
        return _capped(_biased_logits(logits, None if bias is None else bias[jnp.maximum(chosen, 0)]),
                       softcap, temperature)

    pieces = (
        ids.reshape(-1, blocks, block).swapaxes(0, 1),
        columns.reshape(-1, blocks, block).swapaxes(0, 1),
    )
    return _over_supports(jax.lax.map(chunk, pieces).swapaxes(0, 1).reshape(ids.shape), ids, columns, targets)


def logits_support_log_probs(logits, targets, support_ids, support_columns, *, temperature: float = 1.0):
    """`support_log_probs` over a head no matrix gives: `[B, S, vocab]`
    logits, the model's exact head's (`LogitsFromHidden`) or its own
    (`Logits`), whose recorded support entries are read and renormalized."""
    rows = jnp.arange(logits.shape[0])[:, None]
    kept = logits[rows, jnp.maximum(support_columns, 0), jnp.maximum(support_ids, 0)]
    return _over_supports(kept.astype(jnp.float32) / temperature, support_ids, support_columns, targets)


def _over_supports(logits, ids, columns, targets):
    """Each target's `[B, C]` support entries' logits, renormalized within its
    support as `support_log_probs` returns them; -1 columns are padding."""
    width = targets.shape[1]
    labels = jnp.take_along_axis(targets, jnp.maximum(columns, 0), axis=1)

    def row(logits, ids, columns, labels):
        real = columns >= 0
        segment = jnp.where(real, columns, width)
        peak = jax.ops.segment_max(jnp.where(real, jax.lax.stop_gradient(logits), -jnp.inf),
                                   segment, num_segments=width + 1)
        present = jnp.isfinite(peak)
        peak = jnp.where(present, peak, 0.0)
        # Padding exponentiates -inf, so no overflowed entry reaches the backward.
        mass = jax.ops.segment_sum(jnp.exp(jnp.where(real, logits - peak[segment], -jnp.inf)),
                                   segment, num_segments=width + 1)[:width]
        log_z = jnp.log(jnp.where(present[:width], mass, 1.0)) + peak[:width]
        match = real & (ids == labels)
        inside = jax.ops.segment_max(match.astype(jnp.int32), segment, num_segments=width + 1)[:width] > 0
        target = jax.ops.segment_sum(jnp.where(match, logits, 0.0), segment, num_segments=width + 1)[:width]
        filtered = jnp.where(present[:width], jnp.where(inside, target - log_z, -jnp.inf), 0.0)
        return filtered, present[:width]

    return jax.vmap(row)(logits, ids, columns, labels)
