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

import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from flax.typing import PrecisionLike
from jax.sharding import PartitionSpec as P

from dew.nn.precision import head_product, rounded_operand, rounded_to, rounds_to_bf16
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


def _operand_dtype(precision: jax.lax.PrecisionLike):
    """The dtype the head's operands take: bf16 under the bf16 algorithm, fp32
    otherwise."""
    return jnp.bfloat16 if precision is BF16 else jnp.float32


def _cotangent_product(subscripts: str, cotangent, operand, precision: jax.lax.PrecisionLike):
    """`cotangent` times an operand that holds the forward's values, in fp32.

    Under the bf16 algorithm the product would round the fp32 cotangent to
    bf16, and that rounding moved a 4-GPU run's gradients past the layout
    parity bound: 1.75 of it on a dense model's final norm, 5758 of it on an
    MoE's expert kernels, against 0.47 and 0.41 with the cotangent kept in
    fp32. So it enters as a bf16 high half and the bf16 rest, two products
    whose sum carries it to about 2^-17 of its size; the operand is exact in
    bf16 already.
    """
    def product(left):
        return jnp.einsum(subscripts, left, operand, precision=precision,
                          preferred_element_type=jnp.float32)

    if precision is not BF16:
        return product(cotangent)
    # A cast pair would be folded away under jit (`rounded_to`), and the rest with it.
    high = rounded_to(cotangent, jnp.bfloat16)
    return product(high) + product(cotangent - high)


def _capped(logits, softcap, temperature: float = 1.0):
    """Softcapped logits over a sampling temperature, which an engine applies
    to what the model's head returns."""
    if softcap is not None:
        cap = jnp.asarray(softcap, jnp.float32)
        logits = cap * jnp.tanh(logits / cap)
    return logits / temperature


def head_logits(hidden, head_weight, *, softcap: float | None,
                precision: PrecisionLike, vocab_major: bool = False,
                temperature: float = 1.0) -> jax.Array:
    """`hidden @ head_weight` as the model's forward scores it: the states
    against the `[features, vocab]` head (`[vocab, features]` with
    `vocab_major`), accumulated in fp32, softcapped when the backbone caps;
    `[..., vocab]` fp32. The product follows the states' dtype
    (`dew.nn.precision.head_product`)."""
    return _capped(head_product('...d,vd->...v' if vocab_major else '...d,dv->...v',
                                hidden, head_weight, precision), softcap, temperature)


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
                 predict: bool, temperature: float) -> _ChunkTerms:
    """One tile's `_ChunkTerms`."""
    logits = _capped(_tile_logits(hidden, head_chunk, precision), softcap, temperature)

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
             softcap, precision: jax.lax.PrecisionLike, predict: bool, temperature: float):
    """Losses, top-1 columns (None unless `predict`) and log partitions, a
    token tile at a time."""
    features = table.shape[1]
    flat = hidden.reshape(-1, features)
    labels = targets.reshape(-1)
    bounds = vocabulary_chunks(table.shape[0], chunks)
    width = bounds[0][1]
    operands = _operand_dtype(precision)
    # Once, not per token tile: every tile multiplies the same rounded table,
    # and a conversion inside the loop reads the fp32 table once per tile.
    table = table.astype(operands)

    def tokens(start, outputs, size):
        states = jax.lax.dynamic_slice_in_dim(flat, start, size).astype(operands)
        picked_targets = jax.lax.dynamic_slice_in_dim(labels, start, size)

        def columns(first, carry, count):
            terms = _chunk_terms(
                states, jax.lax.dynamic_slice_in_dim(table, first, count),
                picked_targets, first, first + count, softcap, precision, predict, temperature)
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
                       softcap, precision: jax.lax.PrecisionLike, predict: bool, temperature: float):
    """Run `_forward` behind a backward that recomputes its logits."""
    return _forward(hidden, table, targets, chunks, tile[0], softcap, precision, predict, temperature)


def _bounded_head_fwd(hidden, table, targets, chunks, tile, softcap, precision, predict, temperature):
    outputs = _forward(hidden, table, targets, chunks, tile[0], softcap, precision, predict, temperature)
    # The residuals are the inputs and one float32 per token. Everything the
    # backward needs beyond them is a recomputed tile.
    return outputs, (hidden, table, targets, outputs[2], softcap)


def _bounded_head_bwd(chunks, tile, precision, predict, temperature, residuals, cotangents):
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
        matrix = rounded_operand(jax.lax.dynamic_slice_in_dim(
            table, first, width, axis=0).astype(jnp.float32), operands)

        def tokens(start, carry, size):
            d_states, d_matrix, d_cap = carry
            states = rounded_operand(jax.lax.dynamic_slice_in_dim(
                flat, start, size).astype(jnp.float32), operands)
            token_z = jax.lax.dynamic_slice_in_dim(partitions, start, size)
            token_loss = jax.lax.dynamic_slice_in_dim(d_loss, start, size)
            token_partition = jax.lax.dynamic_slice_in_dim(d_partition, start, size)

            logits, pullback = jax.vjp(
                lambda raw, cap: _capped(raw, cap, temperature),
                _tile_logits(states, matrix, precision), softcap)
            # log Z is the whole row's, so a tile's share of the softmax needs
            # no renormalisation, and a target outside the tile one-hots to
            # zero rather than to a wrapped column.
            probabilities = jnp.exp(logits - token_z[:, None])
            selected = jax.nn.one_hot(
                jax.lax.dynamic_slice_in_dim(labels, start, size) - first, width,
                dtype=jnp.float32)
            d_logits = ((token_loss + token_partition)[:, None] * probabilities
                        - token_loss[:, None] * selected)
            d_raw, cap_tile = pullback(d_logits)
            states_tile = _cotangent_product('tv,vd->td', d_raw, matrix, precision)
            # The head's own gradient sums over every token in fp32 already
            # (`d_matrix`), and a split here costs as much again as the
            # states': Qwen3-0.6B's step 174.0 against 186.6 ms on an A100.
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
_bounded_head = jax.custom_vjp(_bounded_head_impl, nondiff_argnums=(3, 4, 6, 7, 8))
_bounded_head.defvjp(_bounded_head_fwd, _bounded_head_bwd)


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
                          tile: tuple[int, int] = (1024, 8192),
                          vocab_major: bool = False,
                          predict: bool = True, temperature: float = 1.0):
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

    The product follows the states' dtype, forward and backward: bf16
    states at the default precision multiply the head as bf16 and accumulate
    in fp32, and fp32 states multiply it as stored. The softmax, the
    logsumexp and the loss are fp32 either way.

    `softcap` is the backbone's `final_logit_softcap`. It is elementwise, so
    capping a tile and capping the row agree. `tile` is a `(tokens, columns)`
    block: the forward walks tokens in the first, the backward walks both and
    holds nothing wider, so it is the only knob on the backward's
    temporaries. The default is measured; a tile wider than the input is one
    tile. `temperature` divides the capped logits (`head_logits`), which
    scores the draws of a sampler at that temperature.

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
    def head(hidden, table, targets, cap):
        return _bounded_head(hidden, table, targets, chunks, tile, cap,
                             BF16 if rounds_to_bf16(hidden.dtype, precision) else precision,
                             predict, float(temperature))

    def column_logits(states, rows, cap):
        """The capped logit of each state against its own row of the head,
        as the head's tiles score it."""
        effective = BF16 if rounds_to_bf16(states.dtype, precision) else precision
        operands = _operand_dtype(effective)
        return _capped(jax.vmap(lambda state, row: _tile_logits(
            state[None].astype(operands), row[None], effective)[0, 0])(states, rows), cap, temperature)

    spec = () if jax.sharding.get_abstract_mesh().empty else _token_spec(targets.shape)
    axes = {axis for entry in spec for axis in mesh_axes(entry)}
    if not axes:
        return head(hidden, table, targets, cap)
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
    size = math.prod(jax.sharding.get_abstract_mesh().shape[axis] for axis in group)
    split = bool(group) and (hidden.size // math.prod(
        jax.sharding.get_abstract_mesh().shape[axis] for axis in axes) * size * hidden.dtype.itemsize
        < table.size * table.dtype.itemsize)

    def local(hidden, table, targets, cap):
        if widths:
            table = jax.lax.all_gather(table, widths, axis=1, tiled=True)
        if split:
            return _vocabulary_split(hidden, table, targets, cap, group, head, column_logits)
        if group:
            table = jax.lax.all_gather(table, group, axis=0, tiled=True)
        return head(hidden, table, targets, cap)

    return jax.shard_map(local, in_specs=(P(*spec, None), held, spec, P()),
                         out_specs=(spec, spec if predict else None, spec), axis_names=axes, check_vma=False)(
        hidden, table, targets, cap)


def _vocabulary_split(hidden, table, targets, cap, group: tuple[str, ...], head, column_logits):
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
    losses, predicted, log_z = head(states, table, labels, cap)
    # Stopped before the max: pmax has no derivative rule, and a stop after it
    # still differentiates it.
    peak = jax.lax.pmax(jax.lax.stop_gradient(log_z), group)
    whole = peak + jnp.log(jax.lax.psum(jnp.exp(log_z - peak), group))
    target = jax.lax.psum(log_z - losses, group)
    start = jax.lax.axis_index(group) * count

    def own(value):
        return jax.lax.dynamic_slice_in_dim(value, start, count).reshape(targets.shape)

    if predicted is not None:
        best = jax.lax.stop_gradient(column_logits(states, jnp.take(table, predicted, axis=0), cap))
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
                      precision: PrecisionLike = None):
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
    entries at a time and rematerialized in the backward pass. Returns the log-probs,
    `-inf` for a target outside its support, and whether each target had one;
    a target with none scores 0.0 here.
    """
    table = jnp.asarray(head_weight).T
    width = targets.shape[1]
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
        return _capped(head_product('bcd,bcd->bc', state, table[jnp.maximum(chosen, 0)], precision),
                       softcap, temperature)

    pieces = (ids.reshape(-1, blocks, block).swapaxes(0, 1), columns.reshape(-1, blocks, block).swapaxes(0, 1))
    logits = jax.lax.map(chunk, pieces).swapaxes(0, 1).reshape(ids.shape)
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
