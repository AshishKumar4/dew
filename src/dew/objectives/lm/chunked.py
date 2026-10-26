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
from typing import NamedTuple, overload

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from dew.mesh import logical_spec, mesh_axes
from dew.nn.categorical import label_derivative, label_entry, rescaled, segment_log_partition, target_loss
from dew.nn.kernels.generation import gpu_target
from dew.nn.precision import (
    added_to,
    at_least_fp32,
    bf16_product,
    capped_logits,
    compensated_sum,
    float32_einsum,
    float32_products,
    head_product,
    rounded_operand,
    rounds_to_bf16,
    running_sum,
    running_total,
    scaled_logits,
    wide_cpu_arithmetic,
)
from dew.nn.protocols import (
    AffineHead,
    HiddenStates,
    Lacks,
    Logits,
    LogitsFromHidden,
    OutputTable,
    head_table,
)


def tile_columns(vocab_size: int, chunks: int, columns: int) -> int:
    """The positive column width for the requested vocabulary partition, capped at `columns`.

    `_over_tiles` walks the full-width tiles and any short tail. More chunks
    than columns use one-column tiles, without empty tiles or padding.
    """
    if chunks < 1:
        raise ValueError(f"a vocabulary needs at least one chunk, got {chunks}")
    return min(columns, max(1, -(-vocab_size // chunks)))


BF16 = jax.lax.DotAlgorithmPreset.BF16_BF16_F32


Terms = tuple[jax.Array, jax.Array | None, jax.Array]
"""Each token's loss, top-1 column (None unless asked for) and log Z."""


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
    cast is round-to-nearest-even, as `bf16_dot_general` rounds its operands."""
    return cotangent.astype(jnp.bfloat16) if precision is BF16 else cotangent


def reads_states(model) -> bool:
    """Whether `model` scores its final states (`HiddenStates`) through a head
    that takes them: a matrix (`AffineHead`) or its exact head
    (`LogitsFromHidden`). One that does not gives its logits only from its
    tokens (`Logits`), which a loss then scores whole (`logits_cross_entropy`)."""
    return isinstance(model, HiddenStates) and isinstance(model, AffineHead | LogitsFromHidden)


def refuse_without_logits(model, objective: str) -> None:
    """Refuse a model `objective` reads no logits from: no head over its final
    states (`reads_states`) and no logits from its tokens (`Logits`)."""
    if not reads_states(model) and not isinstance(model, Logits):
        raise Lacks("logits",
                    f"{objective} scores a model's logits, and a {type(model).__name__} gives none: no head "
                    f"over its final states (HiddenStates with AffineHead or LogitsFromHidden) and no Logits")


def final_read(model, variables, *args, **kwargs):
    """`model.apply(variables, *args, **kwargs)` reading what a loss scores:
    the final states of a model that scores them through a head
    (`reads_states`), else the logits it gives from its tokens (`Logits`)."""
    return model.apply(variables, *args, method="hidden_states" if reads_states(model) else "logits",
                       **kwargs)


def model_logits(model, variables, read) -> jax.Array:
    """`model`'s logits of what `final_read` gave over `variables`: of its
    final states, the matrix `AffineHead.output_table` gives
    (`OutputTable.logits`), or the model's exact head (`LogitsFromHidden`)
    where no matrix alone is the head; a model that gives its logits from
    its tokens gave them already."""
    if not reads_states(model):
        return read
    table = head_table(model, variables)
    if table is None:
        return model.apply(variables, read, method="logits_from_hidden")
    return table.logits(read)


def _without(logits, first, excluded, *, value: float = -jnp.inf):
    """`logits` of the columns from `first` on, with column `excluded` (a
    traced id) given no mass: -inf, so its probability is 0, or `value`. With
    None the logits pass untouched and the program holds no comparison."""
    if excluded is None:
        return logits
    columns = first + jax.lax.broadcasted_iota(jnp.int32, logits.shape, logits.ndim - 1)
    return jnp.where(columns == excluded, value, logits)


class _Columns(NamedTuple):
    """A head's per-column terms past its product (`OutputTable.scale` and `bias`), each `[vocab]`
    at the loss's work dtype or None, which a tile slices, a mesh gathers and a backward
    differentiates together."""
    scale: jax.Array | None
    bias: jax.Array | None

    @classmethod
    def of(cls, head: OutputTable, dtype) -> "_Columns":
        vocab = head.matrix.shape[0 if head.vocab_major else 1]
        columns = cls(*(None if terms is None else jnp.asarray(terms, dtype)
                        for terms in (head.scale, head.bias)))
        if any(terms.shape != (vocab,) for terms in jax.tree.leaves(columns)):
            raise ValueError('scale and bias must have one entry per vocabulary column')
        return columns

    def logits(self, product):
        return scaled_logits(product, self.scale, self.bias)

    def sliced(self, first, count) -> "_Columns":
        return jax.tree.map(lambda terms: jax.lax.dynamic_slice_in_dim(terms, first, count), self)


def _token_sum(shape: tuple[int, ...], dtype, count: int, tile: int) -> jax.Array:
    """An empty running sum over `count` tokens' gradients taken `tile` tokens a product (`added_to`,
    `running_total`): where it takes more additions than each product sums tokens, a double-float pair
    and the plain sum beside it for an overflowed term's infinity (`compensated_total`) on every
    backend, else `running_sum`'s. Fewer additions than each product's terms round less than that
    product, which a whole head's one product rounds once; more come to dominate it. On an A100, 171
    tiles of 3 tokens left a tiled head's bias gradient 2.16 times as far from float64 as the whole
    logits' and its head gradient 1.60 times, 0.50 paired, where 8 tiles of 64 left the head 0.63
    times unpaired (SF-a100-numerics-diag), and 43 tiles of a 128-token fp32 fused head left its scale
    and bias 1.5 times. A default tile, 1024 or 4096 tokens, pairs nothing: paired at an H100's four
    1024-token tiles, a bf16 Qwen3-0.6B head at 4096 tokens took 51.0 ms against 27.9 unpaired (node
    receipts SF-time-fix@h100, SF-time2-bf16@h100)."""
    tiles = -(-count // tile)
    return jnp.zeros((3, *shape), dtype) if tiles > tile else running_sum(shape, dtype)


def _tile_logits(states, matrix, precision: jax.lax.PrecisionLike):
    """Uncapped `[tokens, features] @ [columns, features].T` in at least fp32,
    over operands already in the dtype the head multiplies in. Under the
    bf16 algorithm it is the model's own bf16 product (`bf16_product`, as
    `dew.nn.precision.head_product` takes it), so each logit rounds alike."""
    if precision is BF16:
        return bf16_product('td,vd->tv', states, matrix)
    return float32_einsum('td,vd->tv', states, matrix.astype(states.dtype),
                      precision=precision, preferred_element_type=at_least_fp32(states.dtype))


def _row_terms(logits, predict: bool, labels) -> tuple[
        jax.Array, tuple[jax.Array, jax.Array] | None, tuple[jax.Array, jax.Array, jax.Array]]:
    """Each row's logsumexp, with `predict` its maximum and the first
    column that holds it, and its maximum, the sum of exponentials scaled to
    it and the sum over the rest of the row (`labels`' column left out) scaled
    to it, from one pass over the row.

    The running maximum, the two sums scaled to it and the maximum's column
    ride together through `jax.lax.reduce`, so the row is
    read once, rather than once for the maximum and again for the sum. As
    a single reduction over the row, on an RTX 4080 (jax 0.11.2), fp32
    logits of 4096 x 151936 took 3.9 ms against `jax.nn.logsumexp` and
    `jnp.argmax`'s 7.6, and 8192 x 50304 2.6 against 5.0.

    In a process running wide CPU arithmetic (`dew.nn.precision.WIDE_CPU_ARITHMETIC`,
    Dew's test setting), on a CPU the row reduces in levels of `ROW_BLOCK` columns, each level's
    blocks' terms the next level's row, until one block is left
    (`_row_blocks`): XLA's CPU compiler runs a reduction with a combine of
    its own as one loop over the row, whose sum of exponentials rescales its
    running total at each new maximum and rounds at each step, so its error
    grows with the loop's length. Over 512 columns one loop held a whole
    head's state gradient 3.9 times as far from float64 as
    `jax.nn.logsumexp`'s over 52 orders of its columns; two levels of blocks
    of the row's square root's width held a 2053-column head's 1.50 times as
    far as the full pass's over drawn cotangents, where the vocabulary-tiled
    head's 514-column tiles, each in shorter blocks, held 1.18 times
    (tests/test_chunked_cross_entropy.py, row 1592). A GPU reduces a row as a
    tree of its threads' partial sums already, where more levels are more
    passes over the row for nothing, so other backends keep the one
    reduction."""
    def combine(first, second):
        larger = jnp.maximum(first[0], second[0])
        total = rescaled(first[0], first[1], larger) + rescaled(second[0], second[1], larger)
        beside = rescaled(first[0], first[2], larger) + rescaled(second[0], second[2], larger)
        if not predict:
            return larger, total, beside
        keep = (first[0] > second[0]) | ((first[0] == second[0]) & (first[3] < second[3]))
        return larger, total, beside, jnp.where(keep, first[3], second[3])

    axis = logits.ndim - 1
    # A vocabulary column is int32 whatever the run's default integer width.
    columns = jax.lax.broadcasted_iota(jnp.int32, logits.shape, axis)
    # The rest of the row leaves each row's target column out, so the target's entry is its rest summed
    # directly, never its share less one (`_softmax_cotangent`).
    operands = [logits, jnp.ones_like(logits), (columns != labels[..., None]).astype(logits.dtype)]
    initial = [jnp.array(-jnp.inf, logits.dtype), jnp.array(0, logits.dtype), jnp.array(0, logits.dtype)]
    if predict:
        operands.append(columns)
        initial.append(jnp.array(jnp.iinfo(jnp.int32).max, jnp.int32))

    def reduced(*operands):
        return jax.lax.reduce(operands, tuple(initial), combine, (operands[0].ndim - 1,))

    def blocked(*operands):
        row = list(operands)
        while row[0].shape[-1] > ROW_BLOCK:
            firsts = [reduced(*part) for part in _row_blocks(row)]
            row = [jnp.concatenate([first[index] for first in firsts], axis=-1) for index in range(len(row))]
        return reduced(*row)

    terms = (jax.lax.platform_dependent(*operands, cpu=blocked, default=reduced) if wide_cpu_arithmetic()
             else reduced(*operands))
    return terms[0] + jnp.log(terms[1]), ((terms[0], terms[3]) if predict else None), terms[:3]


ROW_BLOCK = 16
"""The columns a level of `_row_terms`' CPU reduction reduces in one loop."""


def _row_blocks(operands: list[jax.Array]) -> list[list[jax.Array]]:
    """`operands`, rows over their last axis, as the parts one level of `_row_terms` reduces, each
    part's blocks on a new last axis: blocks of `ROW_BLOCK` columns, and the columns left past the
    last whole block as one more."""
    width = operands[0].shape[-1]
    block = ROW_BLOCK
    whole = width - width % block
    parts = [[x[..., :whole].reshape(*x.shape[:-1], whole // block, block) for x in operands]]
    if whole < width:
        parts.append([x[..., whole:][..., None, :] for x in operands])
    return parts


class _ChunkTerms(NamedTuple):
    """One tile's target logit, its best logit and column when a prediction
    is asked for, and its maximum, the sum of exponentials scaled to it and
    the sum over the rest of its row (the target's column left out), from
    which its rows' log partitions and cotangents follow (`_forward`)."""
    picked: jax.Array
    best: jax.Array | None
    column: jax.Array | None
    peak: jax.Array
    mass: jax.Array
    rest: jax.Array


def _chunk_terms(hidden, head_chunk, targets, start: int,
                 softcap: float | None, precision: jax.lax.PrecisionLike,
                 predict: bool, temperature: float, excluded, columns: _Columns) -> _ChunkTerms:
    """One tile's `_ChunkTerms`, its columns from `start` on."""
    logits = _without(capped_logits(columns.logits(_tile_logits(hidden, head_chunk, precision)),
                                    softcap, temperature), start, excluded)
    picked = _target_logits(logits, targets - start)
    _, top, normalizer = _row_terms(logits, predict, targets - start)
    if top is None:
        return _ChunkTerms(picked, None, None, *normalizer)
    return _ChunkTerms(picked, top[0], top[1] + start, *normalizer)


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
             softcap, precision: jax.lax.PrecisionLike, predict: bool, temperature: float, columns: _Columns,
             picked: bool = False):
    """Losses, or with `picked` the targets' logits, top-1 columns (None
    unless `predict`) and log partitions, a token tile at a time, and apart
    from them each row's maximum, the sum of its exponentials scaled to it and
    the sum over the rest of the row, scaled likewise. The vocabulary tiles
    carry only those; a row's log partition is formed from them once, after
    its last tile."""
    features = table.shape[1]
    flat = hidden.reshape(-1, features)
    labels = targets.reshape(-1)
    token_tile, width = tile[0], tile_columns(table.shape[0], chunks, tile[1])
    work = at_least_fp32(hidden.dtype)
    operands = _operand_dtype(precision, work)
    # Once, not per token tile: every tile multiplies the same rounded table,
    # and a conversion inside the loop reads the fp32 table once per tile.
    table = table.astype(operands)

    def tokens(start, outputs, size):
        states = jax.lax.dynamic_slice_in_dim(flat, start, size).astype(operands)
        picked_targets = jax.lax.dynamic_slice_in_dim(labels, start, size)

        def vocabulary(first, carry, count):
            terms = _chunk_terms(
                states, jax.lax.dynamic_slice_in_dim(table, first, count),
                picked_targets, first, softcap, precision, predict, temperature, excluded,
                columns.sliced(first, count))
            peak = jnp.maximum(carry[1], terms.peak)
            merged = (carry[0] + terms.picked, peak,
                      rescaled(carry[1], carry[2], peak) + rescaled(terms.peak, terms.mass, peak),
                      rescaled(carry[1], carry[3], peak) + rescaled(terms.peak, terms.rest, peak))
            if terms.best is None or terms.column is None:
                return merged
            better = terms.best > carry[4]
            return (*merged, jnp.where(better, terms.best, carry[4]),
                    jnp.where(better, terms.column, carry[5]))

        initial = (jnp.zeros((size,), work), jnp.full((size,), -jnp.inf, work),
                   jnp.zeros((size,), work), jnp.zeros((size,), work))
        if predict:
            initial += (jnp.full((size,), -jnp.inf, work), jnp.zeros((size,), jnp.int32))
        target_logit, peak, mass, rests, *top = _over_tiles(initial, table.shape[0], width, vocabulary)
        total = peak + jnp.log(mass)
        loss = target_logit if picked else target_loss(peak, mass, target_logit)
        values = (loss, total, peak, mass, rests, *top[1:])
        return tuple(jax.lax.dynamic_update_slice_in_dim(out, value, start, axis=0)
                     for out, value in zip(outputs, values, strict=True))

    count = labels.shape[0]
    outputs = (jnp.zeros((count,), work),) * 5
    if predict:
        outputs += (jnp.zeros((count,), jnp.int32),)
    losses, log_z, peak, mass, rests, *predicted = (value.reshape(targets.shape) for value in
                                                    _over_tiles(outputs, count, token_tile, tokens))
    return (losses, predicted[0] if predict else None, log_z), (peak, mass, rests)


def _shared_partition(outputs, normalizer, group: tuple[str, ...]):
    """A head's outputs and its rows' maximum, sum of exponentials scaled to it and rest of the sum
    (`_row_terms`), those of the whole row where `group` names the axes the vocabulary is split over
    (`_vocabulary_split`): the shards' sums moved to the largest maximum and added, and log Z formed from
    them once. Each shard's softmax then takes its columns' shares, and the target's entry from the rest,
    against the whole row's maximum and sums, as the whole head's are. The first output must be the
    target's logit (`picked`), which another shard's columns do not move."""
    if not group:
        return outputs, normalizer
    first, best, log_z = outputs
    peak, mass, rests = normalizer
    larger = jax.lax.pmax(peak, group)
    mass = jax.lax.psum(rescaled(peak, mass, larger), group)
    rests = jax.lax.psum(rescaled(peak, rests, larger), group)
    return (first, best, (larger + jnp.log(mass)).reshape(log_z.shape)), (larger, mass, rests)


def _shared_cotangents(cotangents, group: tuple[str, ...]):
    """A head's output cotangents with log Z's summed over `group`: the whole row's log Z is every
    shard's output (`_shared_partition`), and each device's cotangent of it is that of its own tokens,
    which the shards' columns all move."""
    if not group:
        return cotangents
    first, best, partition = cotangents
    return first, best, jax.lax.psum(partition, group)


def _capped(raw, cap, columns: _Columns, temperature: float):
    """A tile's logits from its fp32 head product `raw`, through the per-column terms, the softcap and
    the temperature (`capped_logits`), and the pullback `_head_backward` takes their cotangent through:
    the one head calculus the bounded, whole and fused heads share, each a schedule over it. The cap's
    gradient sums the logits' shares of it, rounded once (`compensated_sum`), in the cap's own shape: the
    plain reduction its transpose is rounds as it goes, and over a whole head's logits sat 1.58 times
    as far from float64 as the tiled head's sums of its tiles' on a CPU without YNNPACK's fusions."""
    if cap is None:
        return jax.vjp(lambda raw, cap, terms: capped_logits(terms.logits(raw), cap, temperature), raw, cap,
                       columns)
    logits, pullback = jax.vjp(lambda raw, caps, terms: capped_logits(terms.logits(raw), caps, temperature),
                               raw, jnp.broadcast_to(jnp.asarray(cap, raw.dtype), raw.shape), columns)

    shape = jnp.shape(cap)
    lead = raw.ndim - len(shape)
    # The axes the cap broadcasts along: those it lacks, and those it holds once.
    axes = (*range(lead), *(lead + axis for axis, size in enumerate(shape) if size == 1))

    def pulled(cotangent):
        d_raw, d_caps, d_terms = pullback(cotangent)
        d_cap = compensated_sum(d_caps, axes).reshape(shape)
        return d_raw, d_cap.astype(jnp.asarray(cap).dtype), d_terms

    return logits, pulled


def _head_backward(pullback, cotangent, product_matrix, states, precision: jax.lax.PrecisionLike, work,
                   width: int | None = None):
    """A tile's gradients from its logits' `cotangent` (`_softmax_cotangent`): pulled back through the
    terms and the cap (`_capped`'s `pullback`) to the product's, rounded once as both backward products
    read it (`_logits_cotangent`), then the state gradient against `product_matrix` (over its column
    tiles of `width` where given, `_state_product`) and the head's against `states`, the forward's
    operands; and the cap's and the per-column terms' gradients. Returns `(d_states, d_matrix, d_cap,
    d_terms)`."""
    d_raw, d_cap, d_terms = pullback(cotangent)
    d_raw = _logits_cotangent(d_raw, precision)
    d_states = _state_product(d_raw, product_matrix, precision, work, width)
    d_matrix = float32_einsum('tv,td->vd', d_raw, states, precision=precision, preferred_element_type=work)
    return d_states, d_matrix, d_cap, d_terms


def _bounded_head_impl(hidden, table, targets, excluded, chunks: int, tile: tuple[int, int],
                       softcap, precision: jax.lax.PrecisionLike, predict: bool, temperature: float,
                       columns: _Columns, picked: bool, group: tuple[str, ...]):
    """Run `_forward` behind a backward that recomputes its logits."""
    return _shared_partition(*_forward(hidden, table, targets, excluded, chunks, tile, softcap, precision,
                                       predict, temperature, columns, picked), group)[0]


def _bounded_head_fwd(hidden, table, targets, excluded, chunks, tile, softcap, precision, predict,
                      temperature, columns, picked, group):
    outputs, normalizer = _shared_partition(*_forward(
        hidden, table, targets, excluded, chunks, tile, softcap, precision, predict, temperature, columns,
        picked), group)
    # The residuals are the inputs and two numbers per token. Everything the
    # backward needs beyond them is a recomputed tile.
    return outputs, (hidden, table, targets, excluded, normalizer, softcap, columns)


def _bounded_head_bwd(chunks, tile, precision, predict, temperature, picked, group, residuals, cotangents):
    """Pull the cotangents back through logits recomputed one tile at a time.

    The outer loop walks vocabulary tiles carrying `(d_states, d_table,
    d_cap, d_columns)`: the full-width state gradient, the head gradient
    stored tile by tile, the softcap's and the per-column terms'. The inner
    loop walks token tiles carrying `(d_states, d_matrix, d_cap, d_tile)`,
    where `d_matrix` and `d_tile` accumulate one vocabulary tile's head and
    per-column gradients in fp32 before they are stored, paired over many
    short token tiles (`_token_sum`).
    """
    del predict  # Argmax has no gradient.
    hidden, table, targets, excluded, normalizer, softcap, columns = residuals
    work = at_least_fp32(hidden.dtype)
    operands = _operand_dtype(precision, work)
    loss_cotangent, _, partition_cotangent = _shared_cotangents(cotangents, group)
    # The logits are recomputed from operands that hold the forward's values,
    # in the forward's column tiles, so they are the logits its normalizer
    # summed: XLA rounds a product of another width otherwise.
    token_tile, vocab_tile = tile[0], tile_columns(table.shape[0], chunks, tile[1])
    features = table.shape[1]
    flat = hidden.reshape(-1, features)
    labels = targets.reshape(-1)
    peaks, masses, rests = (value.reshape(-1) for value in normalizer)
    d_loss = loss_cotangent.reshape(-1)
    d_partition = partition_cotangent.reshape(-1)
    count = labels.shape[0]

    def vocabulary(first, gradients, width):
        d_states, d_table, d_cap, d_columns = gradients
        stored = jax.lax.dynamic_slice_in_dim(table, first, width, axis=0)
        matrix = rounded_operand(stored.astype(work), operands)
        # The state gradient's operand, cast to bf16 once per vocabulary tile
        # rather than once per token tile: the same values, as `matrix` is
        # exact in bf16 under the bf16 algorithm.
        product_matrix = stored.astype(jnp.bfloat16) if precision is BF16 else matrix
        columns_tile = columns.sliced(first, width)

        def tokens(start, carry, size):
            d_states, d_matrix, d_cap, d_tile = carry
            states = rounded_operand(jax.lax.dynamic_slice_in_dim(
                flat, start, size).astype(work), operands)
            token_peak = jax.lax.dynamic_slice_in_dim(peaks, start, size)
            token_mass = jax.lax.dynamic_slice_in_dim(masses, start, size)
            token_rest = jax.lax.dynamic_slice_in_dim(rests, start, size)
            token_loss = jax.lax.dynamic_slice_in_dim(d_loss, start, size)
            token_partition = jax.lax.dynamic_slice_in_dim(d_partition, start, size)

            logits, pullback = _capped(_tile_logits(states, matrix, precision), softcap, columns_tile,
                                       temperature)
            # The scale and the bias act after the bf16 product rounds. Their
            # cotangents stay fp32; only the product's two backward operands round.
            states_tile, matrix_tile, cap_tile, terms_tile = _head_backward(
                pullback, _softmax_cotangent(logits, first, excluded, token_peak, token_mass, token_rest,
                                             jax.lax.dynamic_slice_in_dim(labels, start, size), token_loss,
                                             token_partition, picked),
                product_matrix, jax.lax.dynamic_slice_in_dim(flat, start, size).astype(operands), precision,
                work)
            d_tile = jax.tree.map(added_to, d_tile, terms_tile)
            # The state and head gradients run over the vocabulary and token tiles as the whole
            # head's products sum them (`running_sum`; the forms' axis comes first).
            prior = jax.lax.dynamic_slice_in_dim(d_states, start, size, axis=1)
            d_states = jax.lax.dynamic_update_slice_in_dim(
                d_states, added_to(prior, states_tile), start, axis=1)
            if d_cap is not None:
                d_cap = added_to(d_cap, cap_tile)
            return d_states, added_to(d_matrix, matrix_tile), d_cap, d_tile

        d_states, d_matrix, d_cap, d_tile = _over_tiles(
            (d_states, _token_sum((width, features), work, count, token_tile), d_cap,
             jax.tree.map(lambda terms: _token_sum((width,), work, count, token_tile), columns)),
            count, token_tile, tokens)
        # fp32 until every token tile is in, so a bf16 head does not round
        # once per tile.
        d_table = jax.lax.dynamic_update_slice_in_dim(
            d_table, running_total(d_matrix).astype(table.dtype), first, axis=0)
        d_columns = jax.tree.map(
            lambda whole, part: jax.lax.dynamic_update_slice_in_dim(whole, running_total(part), first, 0),
            d_columns, d_tile)
        return d_states, d_table, d_cap, d_columns

    d_states, d_table, d_cap, d_columns = _over_tiles(
        (running_sum(flat.shape, work),
         jnp.zeros(table.shape, table.dtype),
         None if softcap is None else running_sum(jnp.shape(softcap), work),
         jax.tree.map(lambda terms: jnp.zeros(terms.shape, work), columns)),
        table.shape[0], vocab_tile, vocabulary)
    d_columns = jax.tree.map(lambda gradient, terms: gradient.astype(terms.dtype), d_columns, columns)
    d_states = running_total(d_states)
    d_cap = None if d_cap is None else running_total(d_cap).astype(jnp.asarray(softcap).dtype)
    return d_states.reshape(hidden.shape).astype(hidden.dtype), d_table, None, None, d_cap, d_columns


# `jax.custom_vjp` is generic in its return type, and a `functools.partial`
# decorator loses that binding, so it is built by hand.
_bounded_head = jax.custom_vjp(_bounded_head_impl, nondiff_argnums=(4, 5, 7, 8, 9, 11, 12))
_bounded_head.defvjp(_bounded_head_fwd, _bounded_head_bwd)


def _softmax_cotangent(logits, first, excluded, peaks, masses, rests, labels, d_first, d_partition,
                       picked: bool = False):
    """The cotangent of a row tile of logits whose first column is `first`:
    its share of the softmax times the loss's and the partition's
    cotangents, the target's entry taken from the rest of its row; with
    `picked`, where the first output is the target's logit, the share times
    the partition's alone, plus that output's at the target. The share is
    taken against the whole row's maximum and sum of exponentials scaled to
    it, so a tile needs no renormalisation, where dividing by an exponentiated
    log Z would carry log Z's rounding into every share. The cotangent over the
    sum is one number a row, so each column's exponential rounds once more, in
    its product with it, as autodiff of `jax.nn.logsumexp` takes it: a share
    divided out and then scaled rounds twice, which held 61 of a
    vocabulary-split head's 8192 gradient elements more than twice as far from
    float64 as that reference's over 52 orders of its columns, against 5.

    The loss's part of the target's entry is the categorical's label entry
    (`dew.nn.categorical.label_entry`) of the target's share and the rest of its
    row's, `rests` (the row's exponentials scaled to its maximum, the target's
    column left out): its share less one while the share is at most a half,
    else minus the rest, arXiv:2610.31642's eq. 7, which keeps it once the share
    rounds to 1. The partition's part is the target's own share, never 1 less
    the rest, which an unlikely target's share would cancel to 0. Without
    `picked` the entry is `d_partition * p_t + d_first * label_entry`; with it,
    `d_first + d_partition * p_t` taken as `(d_first + d_partition) * p_t -
    d_first * label_entry`, where the loss's cotangents `d_first =
    -d_partition` leave the label entry alone and the partition's alone leaves
    its share. A target outside the tile one-hots to zero rather than to a
    wrapped column."""
    exponentials = jnp.exp(_without(logits, first, excluded) - peaks[:, None])
    # A target at the excluded column scores a constant +inf, which no logit moves.
    selected = _without(jax.nn.one_hot(labels - first, logits.shape[-1], dtype=bool), first,
                        excluded, value=False)
    coefficient = d_partition if picked else d_first + d_partition
    share = exponentials * (coefficient / masses)[:, None]
    entry = label_entry(exponentials / masses[:, None], (rests / masses)[:, None])
    target = (exponentials * ((d_first + d_partition) / masses)[:, None] - d_first[:, None] * entry if picked
              else exponentials * (d_partition / masses)[:, None] + d_first[:, None] * entry)
    return jnp.where(selected, target, share)


def _whole_head_terms(hidden, table, targets, excluded, softcap, precision, predict, temperature,
                      columns: _Columns, picked: bool):
    """The whole `[tokens, vocab]` product in at least fp32, each row's
    maximum, the sum of exponentials scaled to it and the rest of the row's
    sum scaled likewise (`_row_terms`), and what `_forward` returns of the
    logits: losses, the top-1 column (None unless `predict`), log Z."""
    operands = _operand_dtype(precision, at_least_fp32(hidden.dtype))
    flat = hidden.reshape(-1, table.shape[1]).astype(operands)
    labels = targets.reshape(-1)
    raw = _tile_logits(flat, table.astype(operands), precision)
    logits = _without(capped_logits(columns.logits(raw), softcap, temperature), 0, excluded)
    log_z, top, normalizer = _row_terms(logits, predict, labels)
    # The vocabulary-split head shifts each shard's targets by its first row
    # and sums the picked logit over shards, so only the owner may add one.
    picked_logits = _target_logits(logits, labels)
    best = None if top is None else top[1].reshape(targets.shape)
    first = picked_logits if picked else target_loss(normalizer[0], normalizer[1], picked_logits)
    return raw, normalizer, (first.reshape(targets.shape), best, log_z.reshape(targets.shape))


def _whole_head_impl(hidden, table, targets, excluded, softcap, chunks, precision, predict,
                     temperature, columns, picked, group):
    """With nothing to differentiate, the tiled forward: no backward needs
    the whole logits, so an evaluation or a scoring pass stays bounded."""
    return _shared_partition(*_forward(hidden, table, targets, excluded, chunks, (1024, table.shape[0]),
                                       softcap, precision, predict, temperature, columns, picked), group)[0]


def _whole_head_fwd(hidden, table, targets, excluded, softcap, chunks, precision, predict, temperature,
                    columns, picked, group):
    del chunks
    raw, normalizer, outputs = _whole_head_terms(hidden, table, targets, excluded, softcap, precision,
                                                 predict, temperature, columns, picked)
    outputs, normalizer = _shared_partition(outputs, normalizer, group)
    # The fp32 product before the per-column terms and the cap is kept, so
    # the backward multiplies nothing it did not have to: the state and head
    # products alone.
    return outputs, (hidden, table, targets, excluded, raw, normalizer, softcap, columns)


def _whole_head_bwd(chunks, precision, predict, temperature, picked, group, residuals, cotangents):
    """The same gradient `_bounded_head_bwd` takes tile by tile, from the kept
    logits: the logits' cotangent as `_logits_cotangent` gives it to both
    products, the head's gradient accumulated in fp32 once, and an fp32 head's
    state gradient over the tiled head's column tiles (`_state_product`)."""
    del predict
    hidden, table, targets, excluded, raw, (peaks, masses, rests), softcap, columns = residuals
    loss_cotangent, _, partition_cotangent = _shared_cotangents(cotangents, group)
    work = at_least_fp32(hidden.dtype)
    operands = _operand_dtype(precision, work)
    features = table.shape[1]
    logits, pullback = _capped(raw, softcap, columns, temperature)
    width = None if precision is BF16 else tile_columns(table.shape[0], chunks, chunked_tile()[1])
    d_states, d_table, d_cap, d_columns = _head_backward(
        pullback, _softmax_cotangent(logits, 0, excluded, peaks, masses, rests, targets.reshape(-1),
                                     loss_cotangent.reshape(-1), partition_cotangent.reshape(-1), picked),
        table.astype(operands), hidden.reshape(-1, features).astype(operands), precision, work, width)
    return (d_states.reshape(hidden.shape).astype(hidden.dtype), d_table.astype(table.dtype),
            None, None, d_cap, d_columns)


def _state_product(d_raw, table, precision, work, width: int | None) -> jax.Array:
    """The state gradient `d_raw @ table` over the vocabulary: one product, or with `width` the sum of
    its column tiles' products as the tiled head sums them (`running_sum`). On an A100 an fp32 whole
    head's one product over 2053 columns sat 1.47 to 1.50 times as far from float64 as the tiled
    head's, and its former 1024-column blocks, summed as a double-float pair, 2.0 times; its column
    tiles are those the tiled head walks by default (`chunked_tile`). A bf16 head keeps one product,
    as exact as the tiled head's."""
    if width is None or table.shape[0] <= width:
        return float32_einsum('tv,vd->td', d_raw, table, precision=precision, preferred_element_type=work)

    def columns(first, running, size):
        return added_to(running, float32_einsum(
            'tv,vd->td', jax.lax.dynamic_slice_in_dim(d_raw, first, size, axis=1),
            jax.lax.dynamic_slice_in_dim(table, first, size), precision=precision,
            preferred_element_type=work))

    return running_total(_over_tiles(running_sum((d_raw.shape[0], table.shape[1]), work), table.shape[0],
                                     width, columns))


_whole_head = jax.custom_vjp(_whole_head_impl, nondiff_argnums=(5, 6, 7, 8, 10, 11))
_whole_head.defvjp(_whole_head_fwd, _whole_head_bwd)


def chunked_tile() -> tuple[int, int]:
    """The chunked head's `(tokens, columns)` tile on the pool's hardware
    generation: its target row's (`GpuTarget.head_tile`), else 1024 x 8192. The
    generation is `device_generation`'s, the assumed one's or the placed
    platform's where a call sets one, else global device 0's, which every process
    sees alike, so a pool's processes agree on the tile and compile one program.

    A100 and Ada rows name 4096 x 8192, measured to beat the default. On one
    A100, Qwen3-0.6B's head (vocabulary 151936, 1024 wide, 4096 tokens, bf16)
    forward and backward took 48.7 ms at 4096 x 8192 and 60.2 ms at 1024 x
    8192, at 0.46 and 0.21 GiB of temporaries. On an RTX 4080, a whole
    training step at those widths (2 layers, bf16): at 4096 tokens 114.6 ms
    against 123.4 at 1024 x 8192, 119.7 at 2048 x 8192 and 116.6 at 4096 x
    4096; at 8192 tokens 219.1 against 228.3 (1024 x 8192), 227.8 (2048 x
    16384) and 219.3 (8192 x 4096) with less memory. The whole logits beat the
    4096 x 8192 tile where they fit once the step compiles without XLA's
    Triton GEMM fusions (`gpu_target().kernels['xla_triton_gemm']`): 53.5
    against 65.5 ms at 2048 tokens, 93.4 against 110.2 at 4096. With the
    fusions, exactly 4096 whole took 286.0 ms."""
    return gpu_target().head_tile or (1024, 8192)


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


def chunked_cross_entropy(hidden, head: OutputTable, targets, chunks: int, *,
                          tile: tuple[int, int] | None = (1024, 8192),
                          predict: bool = True, temperature: float = 1.0,
                          excluded: int | None = None) -> Terms:
    """Per-token cross entropy of `head.logits(hidden)`, its top-1 column
    and its log partition, the head contracted in tiles.

    `hidden` is `[..., features]` states in any dtype, `head` the table in
    its stored dtype (each tile upcasts its own slice; the products are
    accumulated in fp32), `targets` the `[...]` int32 ids. Returns the
    per-token losses, the argmax prediction (None
    when `predict` is False, which skips the argmax: 0.77 ms of the head's
    8.0 on a TPU v6e) and the row's logsumexp (log Z, which PaLM's z-loss
    squares), all shaped like `targets`, each of the two float outputs carrying its own gradient. The
    caller owns the weighting and the mean, and with them the padding id.

    The backward keeps the head it was given. A `vocab_major` head is
    `[vocab, features]`, the way a tied embedding table is stored, and the
    value kept is the parameter itself; a `[features, vocab]` head is
    transposed here first, and what the backward keeps is that transposed
    array (`head_table` on the backbone hands out the stored orientation).

    The states pass the head's input quantizer first, and the product follows
    the head's compute dtype (the states' where it names none), forward and
    backward: bf16 compute at the default precision multiplies the head as
    bf16 and accumulates in fp32, and fp32 compute multiplies it as stored.
    The softmax, the logsumexp and the loss are fp32 either way, or the
    states' dtype where it is wider (a float64 reference's).

    The softcap is elementwise, so capping a tile and capping the row agree. `tile` is a `(tokens, columns)`
    block: both passes walk tokens in the first and columns in the second
    or a chunk, whichever is narrower, and the backward holds nothing
    wider, so it is the only knob on the backward's temporaries. On a CPU
    under Dew's wide arithmetic (`dew.nn.precision.WIDE_CPU_ARITHMETIC`, its
    test setting) the backward's running sums of the state and head gradients
    keep what each tile's addition drops (`dew.nn.precision.running_sum`),
    three times their bytes, so they round once as the whole head's products do.
    The default is measured; a tile wider than the input is one
    tile. A `tile` of None keeps the whole fp32 logits for the backward
    instead of recomputing them, which is faster wherever they fit: on one
    A100, Qwen3-0.6B's head at 4096 tokens took 33.1 ms with the logits
    held, at 4.6 GiB, against 60.2 ms tiled (`LMObjective.head_tile`).
    `temperature` divides the capped logits (`OutputTable.logits`), which
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
    hidden, vocab_major, precision = head.states(hidden), head.vocab_major, head.precision
    compute = hidden.dtype if head.compute is None else head.compute
    columns = _Columns.of(head, at_least_fp32(hidden.dtype))
    features = head.matrix.shape[1 if vocab_major else 0]
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
    cap = None if head.softcap is None else jnp.asarray(head.softcap, at_least_fp32(hidden.dtype))
    table = head.matrix if vocab_major else head.matrix.T
    product = BF16 if rounds_to_bf16(compute, precision) else float32_products(compute, precision)

    def scored(hidden, table, targets, excluded, cap, columns, *, picked: bool = False,
               group: tuple[str, ...] = ()):
        if tile is None:
            return _whole_head(hidden, table, targets, excluded, cap, chunks, product, predict,
                               float(temperature), columns, picked, group)
        return _bounded_head(hidden, table, targets, excluded, chunks, tile, cap, product,
                             predict, float(temperature), columns, picked, group)

    def column_logits(states, rows, cap, columns):
        """The capped logit of each state against its own row of the head,
        as the head's tiles score it."""
        effective = BF16 if rounds_to_bf16(compute, precision) else float32_products(compute, precision)
        operands = _operand_dtype(effective, at_least_fp32(states.dtype))
        logits = jax.vmap(lambda state, row: _tile_logits(
            state[None].astype(operands), row[None], effective)[0, 0])(states, rows)
        return capped_logits(columns.logits(logits), cap, temperature)

    left_out = None if excluded is None else jnp.asarray(excluded, jnp.int32)
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return scored(hidden, table, targets, left_out, cap, columns)
    return _sharded_head(mesh, hidden, table, targets, left_out, cap, columns, scored, column_logits,
                         predict=predict)


def _fused_terms(hidden, table, labels, weights, z_loss: float, cap, columns: _Columns, chunk: int,
                 precision: jax.lax.PrecisionLike, predict: bool, temperature: float, excluded,
                 gradients: bool):
    """`fused_head_loss`'s per-token terms, a `chunk` of tokens at a time
    against the whole vocabulary, and with `gradients` the loss's gradients
    taken in the same chunk: the states', the head's (fp32 until every chunk
    is in) and the per-column terms'."""
    count, features = hidden.shape
    work = at_least_fp32(hidden.dtype)
    operands = _operand_dtype(precision, work)
    # Once, not per chunk: every chunk multiplies the same rounded table.
    matrix = rounded_operand(table.astype(work), operands)
    product_matrix = table.astype(jnp.bfloat16) if precision is BF16 else matrix

    def tokens(start, carry, size):
        outputs, d_table, d_columns = carry
        states = rounded_operand(jax.lax.dynamic_slice_in_dim(hidden, start, size).astype(work), operands)
        targets = jax.lax.dynamic_slice_in_dim(labels, start, size)
        logits, pullback = _capped(_tile_logits(states, matrix, precision), cap, columns, temperature)
        # One masked row for the partition and the target: a target at the excluded column scores
        # +inf, as the chunked heads score it, and its gradient moves no logit.
        kept = _without(logits, 0, excluded)
        log_z, top, (peaks, masses, rests) = _row_terms(kept, predict, targets)
        values = [target_loss(peaks, masses, _target_logits(kept, targets)), log_z]
        if predict:
            assert top is not None
            values.append(top[1])
        if gradients:
            weight = jax.lax.dynamic_slice_in_dim(weights, start, size).astype(work)
            d_states, d_matrix, _, terms = _head_backward(
                pullback, _softmax_cotangent(logits, 0, excluded, peaks, masses, rests, targets, weight,
                                             2 * z_loss * log_z * weight),
                product_matrix, jax.lax.dynamic_slice_in_dim(hidden, start, size).astype(operands), precision,
                work)
            d_columns = jax.tree.map(added_to, d_columns, terms)
            values.append(d_states)
            # Summed over the token tiles as the bounded head sums them (`_token_sum`).
            d_table = added_to(d_table, d_matrix)
        outputs = tuple(jax.lax.dynamic_update_slice_in_dim(out, value, start, axis=0)
                        for out, value in zip(outputs, values, strict=True))
        return outputs, d_table, d_columns

    outputs = (jnp.zeros((count,), work),) * 2
    if predict:
        outputs += (jnp.zeros((count,), jnp.int32),)
    if gradients:
        outputs += (jnp.zeros((count, features), work),)
    d_table = _token_sum(table.shape, work, count, chunk) if gradients else None
    d_columns = (jax.tree.map(lambda terms: _token_sum(terms.shape, work, count, chunk), columns)
                 if gradients else None)
    outputs, d_table, d_columns = _over_tiles((outputs, d_table, d_columns), count, chunk, tokens)
    summed = None if d_columns is None else jax.tree.map(running_total, d_columns)
    return outputs, None if d_table is None else running_total(d_table), summed


def _fused_total(losses, log_z, weights, z_loss: float):
    """The weighted sum of the losses and the z-loss, summed as `Objective.row_mean` sums."""
    terms = weights.astype(losses.dtype) * (losses + z_loss * jnp.square(log_z) if z_loss else losses)
    return compensated_sum(terms, (0,))


def _fused_impl(hidden, table, labels, weights, columns, z_loss, cap, chunk, precision, predict, temperature,
                excluded):
    (losses, log_z, *predicted), _, _ = _fused_terms(
        hidden, table, labels, weights, z_loss, cap, columns, chunk, precision, predict, temperature,
        excluded, gradients=False)
    return _fused_total(losses, log_z, weights, z_loss), (losses, predicted[0] if predict else None, log_z)


def _fused_fwd(hidden, table, labels, weights, columns, z_loss, cap, chunk, precision, predict, temperature,
               excluded):
    (losses, log_z, *rest), d_table, d_columns = _fused_terms(
        hidden, table, labels, weights, z_loss, cap, columns, chunk, precision, predict, temperature,
        excluded, gradients=True)
    predicted, d_states = (rest[0], rest[1]) if predict else (None, rest[0])
    total = _fused_total(losses, log_z, weights, z_loss)
    # Empty arrays carry the inputs' dtypes, which the gradients return in.
    dtypes = (jnp.zeros((0,), hidden.dtype), jnp.zeros((0,), table.dtype))
    return (total, (losses, predicted, log_z)), (d_states, d_table, d_columns, dtypes)


def _fused_bwd(z_loss, cap, chunk, precision, predict, temperature, excluded, residuals, cotangents):
    """The gradients the forward took, times the total's cotangent; the
    per-token outputs carry none."""
    d_states, d_table, d_columns, (states_dtype, table_dtype) = residuals
    scale = cotangents[0]
    return ((scale * d_states).astype(states_dtype.dtype), (scale * d_table).astype(table_dtype.dtype), None,
            None, jax.tree.map(lambda gradient: scale * gradient, d_columns))


_fused = jax.custom_vjp(_fused_impl, nondiff_argnums=(5, 6, 7, 8, 9, 10, 11))
_fused.defvjp(_fused_fwd, _fused_bwd)


FUSED_LOGIT_BYTES = 2 << 30
"""The fp32 logits one chunk of `fused_head_loss` holds, when no tile asks
for fewer tokens: the most tokens, a power of two, whose logits fit. On an
A100 and an H100, at Qwen3.5-0.8B's head (1024 wide, vocabulary 248320,
12288 tokens, bf16) 2048 tokens a chunk took 123.9 and 49.6 ms against
1024's 129.3 and 56.3 and 4096's 126.3, and at the 8B's (4096 wide,
128256, 4096 tokens) the one chunk this allows took 58.1 ms on the A100
against 2048's 70.8 (tools/benchmark_lm_head.py fused)."""


def fused_tokens(tokens: int, vocab: int, tile: tuple[int, int] | None = None) -> int:
    """`fused_head_loss`'s chunk: `FUSED_LOGIT_BYTES` of fp32 logits, at most
    `tile`'s tokens and every token."""
    fitting = 1 << max(FUSED_LOGIT_BYTES // (4 * vocab), 1).bit_length() - 1
    return min(tokens, fitting, tokens if tile is None else tile[0])


def fused_head_loss(hidden, head: OutputTable, targets, weights, *, z_loss: float = 0.0,
                    tile: tuple[int, int] | None = None, predict: bool = True, temperature: float = 1.0,
                    excluded: int | None = None) -> tuple[jax.Array, Terms]:
    """The weighted sum of `head.logits(hidden)`'s cross entropies at
    `targets` and PaLM's z-loss, sum of w (ce + z_loss log Z^2), with each
    token's loss, top-1 column (None unless `predict`) and log Z beside it.

    Only the sum carries a gradient, and its gradient is taken in the forward
    pass, Liger's fused linear cross entropy: a chunk of tokens
    (`fused_tokens`, `tile`'s tokens at most) scores the whole vocabulary,
    then pulls its softmax back through the head at once, three products a
    chunk and no logits recomputed or held beyond the chunk. The backward
    scales those gradients by the sum's cotangent. The sum is a loss's
    numerator (`Ratio.total`), with no 1/sum(w) in it, so microbatches pool
    by mass; `weights` (`[...]` like `targets`) are known in the forward and
    carry no gradient. The products, the softmax and the rounding are
    `chunked_cross_entropy`'s whole head's, the states through the head's
    input quantizer; the softcap is the head's own, a constant. Under
    `jax.checkpoint` the forward and its gradients run again with the rest
    of the scope, so a step keeps the head outside its rematerialized
    blocks, as `LMObjective`'s does. On a mesh every device sums its own
    tokens against the head gathered whole, the sums added after, and the
    head's gradient leaves reduce-scattered (`_sharded_head`). `excluded` is
    a column the softmax gives no mass, as in `chunked_cross_entropy`.
    """
    if hidden.shape[:-1] != targets.shape or weights.shape != targets.shape:
        raise ValueError(f"{hidden.shape[:-1]} states, {targets.shape} targets and {weights.shape} weights")
    states = head.states(hidden)
    compute = states.dtype if head.compute is None else head.compute
    precision = BF16 if rounds_to_bf16(compute, head.precision) else float32_products(compute, head.precision)
    table = head.matrix if head.vocab_major else head.matrix.T
    columns = _Columns.of(head, at_least_fp32(states.dtype))
    statics = (float(z_loss), head.softcap)

    def summed(hidden, table, targets, excluded, cap, columns):
        del excluded, cap  # Static here, closed over below.
        labels, weights = targets
        chunk = fused_tokens(labels.size, table.shape[0], tile)
        total, (losses, predicted, log_z) = _fused(
            hidden.reshape(-1, hidden.shape[-1]), table, labels.reshape(-1),
            jax.lax.stop_gradient(weights.reshape(-1)), columns, *statics, chunk, precision, predict,
            float(temperature), None if excluded_column is None else int(excluded_column))
        shaped = None if predicted is None else predicted.reshape(labels.shape)
        return total, (losses.reshape(labels.shape), shaped, log_z.reshape(labels.shape))

    excluded_column = excluded
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return summed(states, table, (targets, weights), None, None, columns)
    shares, terms = _sharded_head(mesh, states, table, (targets, weights), None, None, columns, summed, None,
                                  predict=predict)
    return jnp.sum(shares), terms


def head_cross_entropy(model, variables, read, targets, chunks: int, *,
                       tile: tuple[int, int] | None = (1024, 8192), predict: bool = True,
                       temperature: float = 1.0, excluded: int | None = None) -> Terms:
    """The cross entropy of what `final_read` gave, `read`, against `targets`.

    Final states are scored through `model`'s head over `variables`
    (`chunked_cross_entropy`): the matrix `AffineHead.output_table` gives,
    its scale, bias, softcap and precision included. Where no matrix alone is the
    head (a prediction head past the final states, an adapter's factors on
    it), the model's exact logits (`LogitsFromHidden`), and the logits a
    model gives from its tokens, are scored whole: that holds the fp32
    `[..., vocab]` logits for the backward pass, a vocabulary-sized row per
    token, which the tiled head never does.
    """
    table = head_table(model, variables) if reads_states(model) else None
    if table is not None:
        return chunked_cross_entropy(read, table, targets, chunks, tile=tile, predict=predict,
                                     temperature=temperature, excluded=excluded)
    return logits_cross_entropy(model_logits(model, variables, read), targets,
                                predict=predict, temperature=temperature, excluded=excluded)


def _target_logits(logits, targets):
    """Each row of `logits`' entry at its target, 0 for a target outside its
    columns, which picks none: a padding id past the vocabulary, whose weight
    is 0, or another tile's or shard's column."""
    vocab = logits.shape[-1]
    inside = (targets >= 0) & (targets < vocab)
    return jnp.where(inside, jnp.take_along_axis(
        logits, jnp.clip(targets, 0, vocab - 1)[..., None], axis=-1)[..., 0], 0.0)


def logits_cross_entropy(logits, targets, *, predict: bool = True, temperature: float = 1.0,
                         excluded: int | None = None):
    """`chunked_cross_entropy`'s per-token losses, top-1 columns and log
    partitions, of whole `[..., vocab]` logits, in fp32 or wider (a float64
    reference's), as the tiled head sums, and over `temperature`, with column
    `excluded` given no mass. Its derivative is the tiled head's
    (`_whole_losses_tangent`)."""
    logits = _without(logits.astype(at_least_fp32(logits.dtype)) / temperature, 0,
                      None if excluded is None else jnp.asarray(excluded, jnp.int32))
    losses, log_z = _whole_losses(logits, targets)
    return losses, jnp.argmax(logits, axis=-1) if predict else None, log_z


def _whole_partition(logits):
    """Each row's maximum, the sum of its exponentials scaled to it, and those exponentials.

    The maximum is finite, as `jax.nn.logsumexp` takes one: an infinite maximum shifts by 0, so a row
    holding +inf scores +inf, and an all -inf row's partition is -inf, never inf - inf.
    """
    largest = jnp.max(logits, axis=-1)
    peak = jnp.where(jnp.isfinite(largest), largest, 0)
    shifted = jnp.exp(logits - peak[..., None])
    return peak, jnp.sum(shifted, axis=-1), shifted


@jax.custom_jvp
def _whole_losses(logits, targets):
    """Each row's cross entropy and log partition of whole logits, differentiated by `_whole_losses_tangent`.

    Not `_row_terms`: its variadic `jax.lax.reduce` has no derivative, and these losses are differentiated
    (a distilled student's prediction, a whole-logit model).
    """
    peak, mass, _ = _whole_partition(logits)
    return target_loss(peak, mass, _target_logits(logits, targets)), peak + jnp.log(mass)


@_whole_losses.defjvp
def _whole_losses_tangent(primals, tangents):
    """The derivatives of each row's loss, the categorical's own (`dew.nn.categorical.label_derivative`,
    arXiv:2610.31642's eq. 7), and of its log partition, the row's probabilities."""
    logits, targets = primals
    losses, log_z = _whole_losses(logits, targets)
    _, mass, shifted = _whole_partition(logits)
    probability = shifted / mass[..., None]
    d_losses = jnp.sum(label_derivative(probability, targets) * tangents[0], axis=-1)
    d_log_z = jnp.sum(probability * tangents[0], axis=-1)
    return (losses, log_z), (d_losses, d_log_z)


@overload
def _sharded_head(mesh, hidden, table, targets, excluded, cap, columns, head, column_logits: Callable, *,
                  predict: bool) -> Terms: ...


@overload
def _sharded_head(mesh, hidden, table, targets, excluded, cap, columns, head, column_logits: None, *,
                  predict: bool) -> tuple[jax.Array, Terms]: ...


def _sharded_head(mesh, hidden, table, targets, excluded, cap, columns, head, column_logits, *,
                  predict: bool):
    """`head` on a mesh: each device scores its own tokens in a `shard_map`,
    against the head's own columns where they stay split (`_vocabulary_split`)
    or the head gathered whole (`chunked_cross_entropy`). A `head` with no
    `column_logits` returns a sum over its tokens before their outputs
    (`fused_head_loss`), each device's share of it out along the token axes,
    and gathers the head whole. `targets` may be a tuple of arrays alike, split alike."""
    sums = column_logits is None
    spec = _token_spec(jax.tree.leaves(targets)[0].shape)
    axes = {axis for entry in spec for axis in mesh_axes(entry)}
    if not axes:
        return head(hidden, table, targets, excluded, cap, columns)
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
    split = not sums and bool(group) and (hidden.size // math.prod(
        mesh.shape[axis] for axis in axes) * size * hidden.dtype.itemsize
        < table.size * table.dtype.itemsize)

    def local(hidden, table, targets, excluded, cap, columns):
        if widths:
            table = jax.lax.all_gather(table, widths, axis=1, tiled=True)
        if split:
            return _vocabulary_split(hidden, table, targets, excluded, cap, columns, group, head,
                                     column_logits)
        if group:
            table = jax.lax.all_gather(table, group, axis=0, tiled=True)
            columns = jax.tree.map(lambda terms: jax.lax.all_gather(terms, group, axis=0, tiled=True),
                                   columns)
        if sums:
            total, outputs = head(hidden, table, targets, excluded, cap, columns)
            return total.reshape(1), outputs
        return head(hidden, table, targets, excluded, cap, columns)

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

    outputs = (spec, spec if predict else None, spec)
    inputs = (P(naming(spec[0]), *spec[1:], None), P(naming(held[0]), *held[1:]), spec, P(), P(),
              jax.tree.map(lambda terms: P(naming(held[0])), columns))
    return jax.shard_map(local, in_specs=inputs,
                         out_specs=(P(tuple(sorted(axes)) + alone), outputs) if sums else outputs,
                         axis_names=set(mesh.axis_names) - set(mesh.manual_axes), check_vma=False)(
        hidden, table, targets, excluded, cap, columns)


def _vocabulary_split(hidden, table, targets, excluded, cap, columns, group: tuple[str, ...],
                       head, column_logits):
    """`head` inside a `shard_map` over a vocabulary split `group` ways: every
    token of the group against this device's rows of the head, the per-token
    terms combined over the group, and this device's own tokens returned.

    Megatron-LM's parallel cross entropy (Shoeybi et al., 2019). The log
    partition is formed once from the shards' maxima and sums of
    exponentials, combined in the head (`_shared_partition`), so each
    shard's softmax shares round against the whole row's as the whole
    head's do, where a logsumexp of the shards' own log partitions gave
    every column of a shard the rounding of that shard's share of the row.
    The target's logit is the sum of theirs (one shard holds it), the
    prediction the best of their best columns. Differentiated through the
    shard_map as written, and correct unchecked: after the combine each
    device keeps its own tokens, so a psum's cotangent, summed over the
    group, is every device's share of it, as the head sums log Z's
    (`_shared_cotangents`), and the gather's transpose scatters the states'
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
    # The head's first output is each token's target logit (`picked`), not its loss, which is log
    # Z less it: a shard's softmax then takes the shard's share w of the row's mass and the target's
    # one-hot apart, each from its own term, where the loss's gradient would reach it as
    # (1 + (w - 1)), which rounds away the low bits of a small share.
    picked, predicted, whole = head(states, table, labels, local_excluded, cap, columns, picked=True,
                                    group=group)
    target = jax.lax.psum(picked, group)
    start = jax.lax.axis_index(group) * count

    def own(value):
        return jax.lax.dynamic_slice_in_dim(value, start, count).reshape(targets.shape)

    if predicted is not None:
        best = jax.lax.stop_gradient(column_logits(
            states, jnp.take(table, predicted, axis=0), cap,
            jax.tree.map(lambda terms: jnp.take(terms, predicted, axis=0), columns)))
        top = jax.lax.pmax(best, group)
        # The lowest column among the shards that reach the best logit, as an
        # argmax over the whole row picks the first.
        predicted = own(jax.lax.pmin(jnp.where(best == top, predicted + offset,
                                               jnp.iinfo(jnp.int32).max), group))
    return own(whole - target), predicted, own(whole)


SUPPORT_BLOCK = 1 << 15
"""Kept ids one rematerialized step of `support_log_probs` scores, over all rows."""


def support_log_probs(hidden, head: OutputTable, targets, support_ids, support_columns, *,
                      temperature: float = 1.0):
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
    entries at a time and rematerialized in the backward pass, as
    `head.logits` scores them. Returns the log-probs, `-inf` for a target
    outside its support, and whether each target had one; a target with none
    scores 0.0 here.
    """
    hidden = head.states(hidden)
    table = jnp.asarray(head.matrix) if head.vocab_major else jnp.asarray(head.matrix).T
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
        logits = head_product('bcd,bcd->bc', state, table[jnp.maximum(chosen, 0)], head.precision,
                              compute=head.compute)
        scale, bias = (None if terms is None else terms[jnp.maximum(chosen, 0)]
                       for terms in (head.scale, head.bias))
        return capped_logits(scaled_logits(logits, scale, bias), head.softcap, temperature)

    pieces = (
        ids.reshape(-1, blocks, block).swapaxes(0, 1),
        columns.reshape(-1, blocks, block).swapaxes(0, 1),
    )
    return _over_supports(jax.lax.map(chunk, pieces).swapaxes(0, 1).reshape(ids.shape), ids, columns, targets)


def logits_support_log_probs(logits, targets, support_ids, support_columns, *, temperature: float = 1.0):
    """`support_log_probs` over a head no matrix gives: `[B, S, vocab]`
    logits, the model's exact head's (`LogitsFromHidden`) or its own
    (`Logits`), whose recorded support entries are read and renormalized in at
    least float32 (`at_least_fp32`), a float64 head's in float64."""
    rows = jnp.arange(logits.shape[0])[:, None]
    kept = logits[rows, jnp.maximum(support_columns, 0), jnp.maximum(support_ids, 0)]
    return _over_supports(kept.astype(at_least_fp32(kept.dtype)) / temperature, support_ids, support_columns,
                          targets)


def _over_supports(logits, ids, columns, targets):
    """Each target's `[B, C]` support entries' logits, renormalized within its
    support as `support_log_probs` returns them; -1 columns are padding."""
    width = targets.shape[1]
    labels = jnp.take_along_axis(targets, jnp.maximum(columns, 0), axis=1)

    def row(logits, ids, columns, labels):
        real = columns >= 0
        segment = jnp.where(real, columns, width)
        log_z, present = segment_log_partition(logits, segment, width)
        match = real & (ids == labels)
        inside = jax.ops.segment_max(match.astype(jnp.int32), segment, num_segments=width + 1)[:width] > 0
        target = jax.ops.segment_sum(jnp.where(match, logits, 0.0), segment, num_segments=width + 1)[:width]
        filtered = jnp.where(present, jnp.where(inside, target - log_z, -jnp.inf), 0.0)
        return filtered, present

    return jax.vmap(row)(logits, ids, columns, labels)
