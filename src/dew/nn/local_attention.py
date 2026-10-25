"""Attention over nearby keys in memory linear in the sequence: sliding windows
and chunked local attention, run as one flagged kernel call where a kernel
takes the window, else as banded blocks of the span."""

import functools

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from .attention import (
    SPLASH_LANES,
    attention_kernel,
    chunk_mask,
    combined_attention_mask,
    cudnn_takes_mask,
    kernel_for_materialized_mask,
    resolve_implementation,
    scaled_dot_product_attention,
    split_with_queries,
    tensor_shards,
)
from .documents import with_documents
from .inputs import sequence_blocks, valid_order
from .kernels.selection import forced
from .sharding import (
    HEADS,
    KV_HEADS,
    SEQUENCE_AXIS,
    explicit_spec,
    logical_spec,
    manual_map,
    rows_like,
    rows_spec,
    sequence_shards,
)
from .visibility import same_chunk, visible


def _banded(x, before=None):
    """Each block of `[B, n, W, ...]` behind the block before it: `[B, n, 2W, ...]`.

    `before` `[B, 1, W, ...]` is the block before the first: under a sequence
    axis, the previous shard's last `W` rows. None is zeros, whose rows are
    negative and so outside every query's causal reach.
    """
    if before is None:
        before = jnp.zeros_like(x[:, :1])
    return jnp.concatenate([jnp.concatenate([before, x[:, :-1]], axis=1), x], axis=2)


def local_attention(query, key, value, *, window: int | None = None, chunk: int | None = None,
                    places=None, segment_ids=None, max_segments=None, valid=None,
                    right_padded: bool = False, dtype=None, precision=None, force_fp32_for_softmax=True,
                    implementation='auto', sinks=None, softcap=None):
    """Causal self-attention in which each query reads only nearby keys, in
    memory linear in the sequence.

    `window=w` is sliding attention: a query reads itself and the w - 1 keys
    before it (Mistral, Gemma, MaxText's `sliding_window_size`). `chunk=c` is
    chunked local attention: a query reads the keys at or before it in its
    chunk, `place // c` (Llama 4's `attention_chunk_size`). Exactly one is
    set. Both count tokens: `places` `[S]` or `[B, S]` are each token's place
    counted over its document's real tokens (`dew.nn.inputs.counted_places`,
    None for the row index), `segment_ids` keeps packed documents apart (0 is
    padding), at most `max_segments` a row (`Coordinates.max_segments`), and
    `valid` `[B, S]` drops the keys it marks False, its tokens counted
    without them; `right_padded` says each row's valid slots come first
    (`AttentionMetadata.right_padded`).

    No `[S, S]` array is built above two spans. A window flash, cudnn (no
    sinks) or splash takes as a flag, without packing, runs there, skipping
    blocks outside it.
    Chunks that start at row multiples fold into the batch. Everything else
    runs banded: query blocks of the span against their own block and the one
    before under a `[W, 2W]` mask, so the logits are `[S, 2W]` per head. Banding
    pads to whole spans, so at most two spans take the dense call. A row with
    padding among its tokens runs on its real tokens alone (`compacted`), so
    a window's or a chunk's keys sit within the band, as the cache holds them.
    """
    if (window is None) == (chunk is None):
        raise ValueError("local attention takes exactly one of window and chunk")
    span = window if window is not None else chunk
    assert span is not None
    if span < 1:
        raise ValueError(f"a local span is a positive number of keys, got {span}")
    length = query.shape[1]
    if key.shape[1] != length:
        raise ValueError(
            f"local attention is self-attention: {length} queries against "
            f"{key.shape[1]} keys")
    attend = functools.partial(
        _local_attention, window=window, chunk=chunk, max_segments=max_segments, dtype=dtype,
        precision=precision, force_fp32_for_softmax=force_fp32_for_softmax, implementation=implementation,
        sinks=sinks, softcap=softcap)
    # A window counts tokens by their distance alone, so only a chunk reads the places.
    places = None if chunk is None else places
    if valid is not None and not right_padded:
        return compacted(attend, query, key, value, places, segment_ids, valid)
    return attend(query, key, value, places=places, segment_ids=segment_ids, valid=valid)


def _local_attention(query, key, value, *, window, chunk, places, segment_ids, max_segments, valid,
                     dtype, precision, force_fp32_for_softmax, implementation, sinks, softcap):
    """`local_attention` over rows whose real tokens come first, each token's chunk place given
    (`places`, None for the row index)."""
    length = query.shape[1]
    span = window if window is not None else chunk
    assert span is not None
    whole = functools.partial(
        scaled_dot_product_attention, query, key, value, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax, sinks=sinks, softcap=softcap)
    resolved = resolve_implementation(
        implementation, query, key, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax, softcap=softcap, sinks=sinks,
        causal=True, sliding_window=window, segment_ids=segment_ids, max_segments=max_segments)
    # A mask built here is at most `[S, 2W]` per head, so cudnn takes it as
    # its additive bias where cudnn runs (docs/performance.md: 16384 packed
    # tokens at window 4096 in 0.60 GiB of temporaries on an L4, where xla
    # asked for 10.0 GiB at 8192). Flash takes no mask, and the whole call's
    # packed documents keep cuDNN off it (they reach it only as this mask), so
    # where 'auto' chose either for the window, a masked call still goes to
    # cudnn, unless the caller chose xla (`generation.chosen`).
    banded = resolved == 'flash' or (resolved == 'xla' and forced('attention') is None)
    on_cudnn = resolved == 'cudnn' or (implementation == 'auto' and banded
                                       and cudnn_takes_mask(implementation, query, softcap=softcap,
                                                            sinks=sinks is not None))
    masked = 'cudnn' if on_cudnn else kernel_for_materialized_mask(
        implementation, query, key, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax)
    if window is not None and valid is None:
        # An explicit 'tpu' at a length splash cannot tile would reach the
        # flash kernel, whose default 128-row blocks must divide the length
        # (flash_attention.py:113-128, 1707-1715), so it would raise there.
        on_splash = resolved == 'tpu' and length % SPLASH_LANES == 0
        windowed = resolved == 'flash' or (resolved in ('cudnn', 'tpu') and sinks is None)
        # Flash takes packed documents as its varlen sequences where their cap is declared.
        packs = segment_ids is None or (resolved == 'flash' and max_segments is not None)
        if on_splash or (packs and (windowed or length <= 2 * span)):
            return whole(implementation=implementation, causal=True, sliding_window=window,
                         segment_ids=segment_ids, max_segments=max_segments)
    shards = sequence_shards()
    # Under a sequence axis each shard's first block reads the previous
    # shard's last `span` rows, which only one neighbour holds when a shard's
    # slice is at least that long. A wider span takes the dense call, whose
    # mask the sequence exchange carries.
    halo = length % shards == 0 and length // shards >= span
    if length <= 2 * span or not halo:
        # A mask pairs every query with every key, which an Explicit axis splitting both sides of it
        # would place twice, so the `[B, S]` fields it is built from are each device's whole.
        segment_ids, valid, places = (None if x is None else rows_like(jnp.asarray(x), jnp.asarray(x))
                                      for x in (segment_ids, valid, places))
        chunked = jnp.arange(length) if places is None else places
        mask = (None if chunk is None or (places is None and length <= chunk)
                else chunk_mask(chunked, chunked, chunk))
        if segment_ids is not None:
            mask = with_documents(mask, segment_ids)
        if valid is not None:
            live = jnp.asarray(valid, bool)[:, None, None, :]
            mask = live if mask is None else mask & live
        if mask is None:
            return whole(implementation=implementation, causal=True, sliding_window=window)
        return whole(implementation=masked, mask=combined_attention_mask(
            length, length, causal=True, sliding_window=window, mask=mask))
    kernel = functools.partial(
        attention_kernel, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax, softcap=softcap)
    if shards > 1:
        out = _local_over_sequence(
            kernel, query, key, value, shards, window=window, chunk=chunk, places=places,
            segment_ids=segment_ids, valid=valid, sinks=sinks, implementation=implementation,
            masked=masked)
    else:
        out = _local_blocks(functools.partial(kernel, sinks=sinks), query, key, value,
                            window=window, chunk=chunk, places=places,
                            segment_ids=segment_ids, valid=valid,
                            implementation=implementation, masked=masked)
    return checkpoint_name(out, 'attention_output')


def compacted(attend, query, key, value, places, segment_ids, valid):
    """`attend` over each row's real tokens moved to its front in order, their
    places and documents with them, and its outputs moved back to their slots:
    the one move every attention over rows with padding among their tokens
    shares, this module's local attention and a tier-3 model's attention
    (`dew.interop.torchax_fallback`) alike.

    A window spans `window` tokens and a chunk `chunk` places, and padding
    among a row's tokens spreads them over more slots than the band before a
    query holds: real slots 0, 3, 6 and 9 at places 0 to 3 are one chunk of 4
    the band of slot 9 reads only from slot 4 on, and a window of 2 at slot 9
    reads slot 6 though no slot between holds a token. Compacted, a window's
    or a chunk's real tokens sit within that many consecutive slots. Given
    places move with their tokens; without them each token's place is its
    compacted slot, its rank among the row's tokens. The padding's outputs
    come back unspecified. The moves sort along the sequence, which JAX
    refuses over an Explicit axis that splits it, so each operand is moved
    with its sequence whole and placed back as it came (`_whole_sequence`):
    a sequence axis still runs the band on its own slices, its halo of
    compacted tokens."""
    batch, length = valid.shape
    real = _whole_sequence(jnp.asarray(valid, bool))
    order, count = valid_order(real)
    # Each slot's column in that order: a real token's rank among the real, a padding slot's after them.
    column = jnp.where(real, jnp.cumsum(real, axis=1, dtype=jnp.int32) - 1,
                       count[:, None] + jnp.cumsum(~real, axis=1, dtype=jnp.int32) - 1)

    def moved(x, to):
        x = jnp.asarray(x)
        taken = jnp.take_along_axis(_whole_sequence(x), to.reshape(*to.shape, *(1,) * (x.ndim - 2)), axis=1,
                                    mode='promise_in_bounds')
        placed = explicit_spec(x)
        return taken if placed is None else jax.sharding.reshard(taken, placed)

    out = attend(*(moved(x, order) for x in (query, key, value)),
                 places=None if places is None else moved(jnp.broadcast_to(places, (batch, length)), order),
                 segment_ids=None if segment_ids is None else moved(segment_ids, order),
                 valid=moved(jnp.asarray(valid, bool), order))
    return moved(out, column)


def _whole_sequence(x: jax.Array) -> jax.Array:
    """`x` with its sequence (its second dimension) whole on each device where an Explicit axis
    splits it, its other dimensions placed as they were."""
    placed = explicit_spec(x)
    if placed is None:
        return x
    spec = (placed.spec[0], None, *placed.spec[2:])
    return jax.sharding.reshard(x, jax.sharding.NamedSharding(placed.mesh, jax.sharding.PartitionSpec(*spec)))


def _local_over_sequence(kernel, query, key, value, shards: int, *, window, chunk, places,
                         segment_ids, valid, sinks, implementation, masked):
    """`_local_blocks` over a sequence the mesh's sequence axis splits, in a
    `shard_map`: each shard computes its own rows, and one `ppermute` hands its
    first block the previous shard's last `span` keys, values, places,
    segment ids and validity (zeros on the first shard). That halo is all that
    crosses the interconnect; chunks starting at the shard boundaries need none.
    Heads split over the tensor axis as the exchanges split them."""
    batch, length = query.shape[:2]
    span = window if window is not None else chunk
    assert span is not None
    queries = logical_spec(HEADS, query.shape)
    key, value = split_with_queries(key, value, tensor_shards(query))
    keys = logical_spec(KV_HEADS, key.shape)
    fields = {}
    if places is not None:
        fields['places'] = jnp.broadcast_to(jnp.asarray(places), (batch, length))
    if segment_ids is not None:
        fields['segment_ids'] = segment_ids
    if valid is not None:
        fields['valid'] = jnp.asarray(valid, bool)
    # Each `[B, S]` field split as the queries' rows and sequence are, where an Explicit axis splits them.
    placed = explicit_spec(query)
    if placed is not None:
        split = jax.sharding.NamedSharding(placed.mesh, jax.sharding.PartitionSpec(*placed.spec[:2]))
        fields = {name: jax.sharding.reshard(field, split) for name, field in fields.items()}
    rows = logical_spec(("activation_batch", "activation_length"), (batch, length))
    specs = [rows] * len(fields)
    if sinks is not None:
        fields['sinks'] = sinks
        specs.append(logical_spec(("activation_heads",), sinks.shape))
    names = tuple(fields)
    local_length = length // shards
    # Chunks that start at every shard's first row stay inside their shard.
    aligned = chunk is not None and places is None and local_length % chunk == 0

    def local(query, key, value, *arrays):
        given = dict(zip(names, arrays, strict=True))
        forward = [(shard, shard + 1) for shard in range(shards - 1)]

        def last_rows(x):
            if x is None:
                return None
            return jax.lax.ppermute(x[:, local_length - span:], SEQUENCE_AXIS, forward)

        before = None if aligned else tuple(
            last_rows(x) for x in (key, value, given.get('places'),
                                   given.get('segment_ids'), given.get('valid')))
        return _local_blocks(
            functools.partial(kernel, sinks=given.get('sinks')), query, key, value,
            window=window, chunk=chunk, places=given.get('places'),
            segment_ids=given.get('segment_ids'), valid=given.get('valid'),
            implementation=implementation, masked=masked,
            offset=jax.lax.axis_index(SEQUENCE_AXIS) * local_length, before=before)

    attended = manual_map(local, (queries, keys, keys, *specs), queries)
    return attended(query, key, value, *fields.values())


def _local_blocks(kernel, query, key, value, *, window, chunk, places, segment_ids, valid,
                  implementation, masked, offset=0, before=None):
    """`local_attention`'s blocked computation over the `[B, S, ...]` rows it
    is handed, which start at row `offset` of the sequence.

    `before` holds what the first rows read before them: the previous `span`
    rows' keys, values, places, segment ids and validity, each None where
    the call has none. None is the start of the sequence, where nothing comes
    before, and chunks start at row multiples from there."""
    batch, length = query.shape[0], query.shape[1]
    span = window if window is not None else chunk
    assert span is not None
    blocks = -(-length // span)
    flags_only = segment_ids is None and valid is None

    def folded(x, width: int):
        return x.reshape(batch * blocks, width, *x.shape[3:])

    def row_blocks(x):
        return sequence_blocks(jnp.broadcast_to(jnp.asarray(x), (batch, length)), span, blocks)

    def preceding(index):
        # The halo field at `index`, as the one block before the first.
        return None if before is None else before[index][:, None]

    if chunk is not None and places is None and before is None:
        mask = None
        if not flags_only:
            keep = jnp.ones((batch, blocks, span, span), bool)
            if segment_ids is not None:
                segments = row_blocks(segment_ids)
                keep = keep & ((segments[..., :, None] == segments[..., None, :])
                               & (segments[..., :, None] != 0))
            if valid is not None:
                keep = keep & row_blocks(jnp.asarray(valid, bool))[..., None, :]
            mask = keep.reshape(batch * blocks, 1, span, span)
            implementation = masked
        out = kernel(*(folded(sequence_blocks(x, span, blocks), span) for x in (query, key, value)),
                     implementation=implementation, causal=True, mask=mask)
    else:
        rows = offset + jnp.arange(blocks * span).reshape(1, blocks, span)
        key_rows = jnp.concatenate([rows - span, rows], axis=-1)
        keep = ((key_rows[..., None, :] >= 0)
                & visible(rows[..., :, None], key_rows[..., None, :], causal=True, window=window))
        keep = jnp.broadcast_to(keep, (batch, blocks, span, 2 * span))
        if chunk is not None and places is None:
            keep = keep & same_chunk(rows[..., :, None], key_rows[..., None, :], chunk)
        elif chunk is not None:
            blocked = row_blocks(places)
            keep = keep & same_chunk(blocked[..., :, None], _banded(blocked, preceding(2))[..., None, :],
                                     chunk)
        if segment_ids is not None:
            segments = row_blocks(segment_ids)
            keep = keep & ((segments[..., :, None] == _banded(segments, preceding(3))[..., None, :])
                           & (segments[..., :, None] != 0))
        if valid is not None:
            keep = keep & _banded(row_blocks(jnp.asarray(valid, bool)),
                                  preceding(4))[..., None, :]
        out = kernel(folded(sequence_blocks(query, span, blocks), span),
                     folded(_banded(sequence_blocks(key, span, blocks), preceding(0)), 2 * span),
                     folded(_banded(sequence_blocks(value, span, blocks), preceding(1)), 2 * span),
                     implementation=masked,
                     mask=keep.reshape(batch * blocks, 1, span, 2 * span))
    unfolded = (batch, blocks * span, *out.shape[2:])
    return jax.lax.reshape(out, unfolded, out_sharding=rows_spec(out, len(unfolded)))[:, :length]

