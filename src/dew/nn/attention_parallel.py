"""Attention over a sequence the mesh splits: a kernel run on each shard's rows and heads (`per_shard`),
the heads exchanged all-to-all (`exchanged_heads_attention`) or the keys gathered
(`gathered_keys_attention`), as `sequence_parallel_attention` chooses, causal rows striped so each
shard does equal work (`stripe`).
"""

import math

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from dew.mesh import SEQUENCE_AXIS, LayoutRefused, logical_spec, mesh_axes

from .attention_masks import repeat_kv_heads, structural_mask
from .kernels.templates import attention as template
from .sharding import HEADS, KV_HEADS, SCORES, manual_map


def pad_rows(x, rows: int):
    """`x` with `rows` zero rows appended on its sequence axis, [B, S, H, D]."""
    return x if rows == 0 else jnp.pad(x, ((0, 0), (0, rows), (0, 0), (0, 0)))


def stripe(x, shards: int, axis: int = 1):
    """Reorder `axis` so every shard holds equal causal work.

    The axis is cut into 2 * shards chunks and shard i takes chunks i and
    2 * shards - 1 - i, MaxText's `reorder_sequence` order: for two shards
    [0..7] becomes [0, 1, 6, 7, 2, 3, 4, 5]. Reshape, slice, flip and stack
    lower to collective-permutes of the chunks that move; a gather along a split
    axis lowers to an all-gather of the whole array
    (tests/test_sequence_parallel.py measures both).
    """
    axis %= x.ndim
    length = x.shape[axis]
    if length % (2 * shards):
        raise LayoutRefused(
            f"sequence parallelism over {shards} shards pairs chunks of the "
            f"sequence to balance the causal work, which needs the sequence "
            f"length to be a multiple of {2 * shards}, got {length}")
    chunk = length // (2 * shards)
    chunks = x.reshape((*x.shape[:axis], 2 * shards, chunk, *x.shape[axis + 1:]))
    first = jax.lax.slice_in_dim(chunks, 0, shards, axis=axis)
    second = jnp.flip(jax.lax.slice_in_dim(chunks, shards, 2 * shards, axis=axis), axis=axis)
    return jnp.stack([first, second], axis=axis + 1).reshape(x.shape)


def unstripe(x, shards: int, axis: int = 1):
    """The inverse of `stripe`: `axis` back in sequence order."""
    axis %= x.ndim
    chunk = x.shape[axis] // (2 * shards)
    pairs = x.reshape((*x.shape[:axis], shards, 2, chunk, *x.shape[axis + 1:]))
    first = jax.lax.index_in_dim(pairs, 0, axis=axis + 1, keepdims=False)
    second = jnp.flip(jax.lax.index_in_dim(pairs, 1, axis=axis + 1, keepdims=False), axis=axis)
    return jnp.concatenate([first, second], axis=axis).reshape(x.shape)


def sequence_parallel_attention(kernel, query, key, value, shards: int, *, causal,
                                sliding_window, mask, bias, sinks, key_value_seq_lengths=None,
                                dropout_rate=0.0, dropout_seed=None):
    """Run `kernel` over a sequence the mesh's sequence axis splits.

    `exchanged_heads_attention` (Ulysses) takes a call whose query heads divide
    by tensor times sequence and whose lengths divide by the shard count;
    `gathered_keys_attention` takes any head count and length. Where both run,
    a causal, windowed or masked call takes the all-to-all: its kernel sees
    whole sequences and skips masked blocks, while the gather's striped
    explicit mask is a dense bias no kernel skips (2.0 to 4.0 times the
    all-to-all's time for causal GQA at 4k to 64k tokens on two RTX 3090s,
    docs/guides/multi-node.md), and it never sends more bytes. A call with no
    mask takes the exchange that sends fewer (`all_to_all_moves_less`).
    A dropping call drops what one device would in either (`template.DropPlaces`).
    """
    tensor = tensor_shards(query)
    heads, kv_heads = query.shape[-2], key.shape[-2]
    masked = causal or sliding_window is not None or mask is not None
    exchangeable = (heads % (tensor * shards) == 0
                    and query.shape[-3] % shards == 0 and key.shape[-3] % shards == 0)
    run = (exchanged_heads_attention
           if exchangeable and (masked or all_to_all_moves_less(heads, kv_heads, tensor, shards))
           else gathered_keys_attention)
    return run(kernel, query, key, value, shards, causal=causal,
               sliding_window=sliding_window, mask=mask, bias=bias, sinks=sinks,
               key_value_seq_lengths=key_value_seq_lengths, dropout_rate=dropout_rate,
               dropout_seed=dropout_seed)


def all_to_all_moves_less(heads: int, kv_heads: int, tensor: int, shards: int) -> bool:
    """Whether Ulysses sends fewer bytes per device than gathering the keys,
    for a call with no mask.

    Counted per device, per batch row, in units of S * D * (n - 1) / n for
    a sequence of S rows, head width D and n = `shards`, over one tensor
    shard's heads: H = heads / tensor, K = the key heads it holds.

    The all-to-all sends (n - 1) / n of four local blocks of S / n rows:
    the query and the output at H heads, the key and the value at K', the
    key heads repeated out to lcm(kv_heads, tensor * n) over tensor. That
    is 2 (H + K') / n.

    The gather receives the (n - 1) / n of the whole keys and values it
    does not hold, 2 K, the key heads repeated out to lcm(kv_heads, tensor)
    over tensor. A causal, windowed or masked call also stripes its S / n
    query rows and unstripes its output rows, which adds 2 H / n, so for
    such a call the all-to-all never sends more: K' / n is at most K.

    At H = 32, K = 8, n = 2 with one tensor shard, in those units the
    all-to-all sends 40, the gather 16: grouped-query attention with no mask
    gathers. The gather wins ties, as the path that takes every shape.
    Mask and bias blocks are left out; both paths carry them.
    """
    local_heads = heads // tensor
    repeated = math.lcm(kv_heads, tensor * shards) // tensor
    gathered = math.lcm(kv_heads, tensor) // tensor
    return 2 * (local_heads + repeated) / shards < 2 * gathered


def _entry(spec: P, dimension: int):
    """What `spec` names for `dimension`: a spec leaves off trailing whole ones."""
    return spec[dimension] if dimension < len(spec) else None


def tensor_shards(query) -> int:
    """How many ways the mesh's tensor axes split the query heads."""
    mesh = jax.sharding.get_abstract_mesh()
    return math.prod(mesh.shape[axis]
                     for axis in mesh_axes(_entry(logical_spec(HEADS, query.shape), 2)))


def shard_heads(query) -> int:
    """The query heads one shard holds (`tensor_shards`)."""
    return query.shape[-2] // tensor_shards(query)


def split_with_queries(key, value, split: int):
    """`key` and `value` with their heads repeated to the least common multiple
    of their count and `split`, the ways the mesh splits the query heads, so
    each shard holds the key heads its query heads read."""
    heads = math.lcm(key.shape[-2], split)
    return repeat_kv_heads(key, heads), repeat_kv_heads(value, heads)


def per_shard(local, query, key, value, *, coordinates: bool = False, **extras):
    """`local(query, key, value, **extras)` on each shard's rows and heads
    (`manual_map`). The key heads a shard reads are the ones its key shard holds,
    at their offset in that shard, where the key heads' split begins with the query
    heads' split; a key-head split that does not is set aside, and the key heads are
    held whole on every shard. Where the tensor axes split the query heads more ways
    than the key heads, each shard takes its share of the repeats `split_with_queries`
    makes, so the key heads' gradient is one sum across the shards in the map's
    transpose. Whole off a mesh, or inside a map that already holds every axis manual
    (the sequence exchanges), where the tensor axes split nothing. A Pallas kernel has
    no partitioning rule, and jax's cuDNN attention partitions through
    `custom_partitioning`, whose lowering on several devices names the partitioner by
    a process address (`str(id(...))` in its backend config), which the persistent
    cache key keeps, so every fresh process would compile again; under a fully manual
    map jax lowers neither. `extras` maps each further argument to its array, or None,
    and the logical axes it splits on: a dimension named for the rows or the query heads splits as the
    query's own rows or heads do (`query_split`), whatever the rules give that name alone, so a shard's
    coordinates are its own rows' and heads' where axes contend for them.
    With `coordinates`, `local` also takes `global_rows` and `global_heads`, the shard's global batch rows
    and query heads from the query's own split, so a draw keyed on them is the same whatever the rules give
    the rows and heads alone.
    """
    given = {name: pair for name, pair in extras.items() if pair[0] is not None}
    queries, keys = logical_spec(HEADS, query.shape), logical_spec(KV_HEADS, key.shape)
    query_axes, key_axes = mesh_axes(_entry(queries, 2)), mesh_axes(_entry(keys, 2))
    row_axes = mesh_axes(_entry(queries, 0))
    if key_axes and query_axes[: len(key_axes)] != key_axes:
        keys, key_axes = P(*keys[:2]), ()
    split, kv_heads = tensor_shards(query), key.shape[-2]
    heads = math.lcm(kv_heads, split)
    share, copies = heads // split, heads // kv_heads
    names = tuple(given)

    def shard_start(axes, extent):
        # Where a shard's extent begins along the mesh axes that split it (0 where no axis does).
        return jax.lax.axis_index(axes) * extent if axes else 0

    def shard(query, key, value, *arrays):
        if copies > 1 or key.shape[-2] != share:
            # This shard's share of the repeated key heads (head n is key head n // copies), read from the key
            # heads it holds, which begin at its own key shard's offset.
            read = (jax.lax.axis_index(query_axes) * share + jnp.arange(share)) // copies
            if key_axes:
                read = read - jax.lax.axis_index(key_axes) * key.shape[-2]
            key, value = (jnp.take(array, read, axis=-2) for array in (key, value))
        arguments = dict(zip(names, arrays, strict=True))
        if coordinates:
            rows_at = shard_start(row_axes, query.shape[0])
            arguments["global_rows"] = rows_at + jnp.arange(query.shape[0], dtype=jnp.int32)
            heads_at = shard_start(query_axes, query.shape[2])
            arguments["global_heads"] = heads_at + jnp.arange(query.shape[2], dtype=jnp.int32)
        return local(query, key, value, **arguments)

    # An extra may be a tree of arrays (`CrossDocuments`), each leaf split on its axes.
    specs = (queries, keys, keys,
             *(jax.tree.map(lambda leaf, axes=axes: query_split(axes, leaf, queries, query.shape), x)
               for x, axes in given.values()))
    return manual_map(shard, specs, queries)(query, key, value, *(x for x, _ in given.values()))


def query_split(axes, leaf, queries: P, shape: tuple[int, ...]) -> P:
    """The spec of a per-shard argument `leaf` whose dimensions `axes` names, from the query's own split
    `queries` of its `shape`: a dimension of the rows, the queries or the query heads takes the query's,
    where it is as long (a broadcast dimension of 1 stays whole), any other none. Every manual map over
    attention's rows splits its row-scoped arguments so (`per_shard`, the sequence exchanges, local
    attention over a sequence), so a shard's masks, lengths and ids are its own rows' where axes contend for
    them, whatever the rules give those names alone."""
    of = {"activation_batch": 0, "activation_length": 1, "activation_heads": 2}
    return P(*(_entry(queries, of[axis]) if axis in of and length == shape[of[axis]] else None
               for axis, length in zip(axes, jnp.shape(leaf), strict=False)))


def four_dimensional(x):
    """A mask or a bias `[.., Q, K]` broadcastable to `[B, H, Q, K]`, as four
    dimensions; a broadcast dimension of 1 divides by no mesh axis, so the
    rule table leaves it whole."""
    return x.reshape((1,) * (4 - x.ndim) + x.shape)


def exchanged_heads_attention(kernel, query, key, value, shards: int, *, causal,
                              sliding_window, mask, bias, sinks, key_value_seq_lengths=None,
                              dropout_rate=0.0, dropout_seed=None):
    """DeepSpeed Ulysses: trade a slice of the sequence for a slice of the heads.

    One all-to-all per operand over the sequence axis turns each shard's S/n
    rows of every head into all S rows of H/n heads; the kernel attends whole
    sequences and one more all-to-all puts the output back in rows (Jacobs et
    al. 2023, arXiv:2309.14509). Causality, windows, document masks and splash's
    block skipping work as on one device, causal work is equal without a
    reorder, autodiff transposes each all-to-all, and a shard holds S*K*D/n of
    each key and value against the gather's S*K*D.

    Heads split over tensor first and sequence within it, so H divides by
    tensor times sequence; key and value heads repeat only to the least common
    multiple of their count and that split. A mask or bias head dimension,
    learned sinks and key lengths split with what they belong to. Batch rows
    split over every batch axis that still divides them, in mesh order, and are
    attended alike where it does not. A dropping call draws each shard's
    weights at their global rows and heads, its sequences whole
    (`template.DropPlaces`), so it drops what one device would.
    """
    mesh = jax.sharding.get_abstract_mesh()
    heads = query.shape[2]
    queries = logical_spec(HEADS, query.shape)
    tensor = mesh_axes(_entry(queries, 2))
    split = shards * math.prod(mesh.shape[axis] for axis in tensor)
    if heads % split or query.shape[1] % shards or key.shape[1] % shards:
        raise LayoutRefused(
            f"the all-to-all sequence exchange splits the {heads} query heads "
            f"{split} ways (tensor times sequence) and the query and key lengths "
            f"({query.shape[1]}, {key.shape[1]}) {shards} ways, and one of them does "
            "not divide. `gathered_keys_attention` takes any head count and key "
            "length; `sequence_parallel_attention` picks it for such a call.")
    key, value = split_with_queries(key, value, split)
    keys = logical_spec(KV_HEADS, key.shape)

    # After the all-to-all a shard holds its tensor shard's heads split again
    # over the sequence axis; a mask, a bias or sinks with a head dimension
    # arrive split the same way, their query and key dimensions whole.
    exchanged_heads = (*tensor, SEQUENCE_AXIS)
    extras = {}
    for name, x in (('mask', mask), ('bias', bias)):
        if x is not None:
            x = four_dimensional(x)
            # Its rows split as the query's own (`query_split`), its heads as the exchange leaves them.
            rows = _entry(queries, 0) if x.shape[0] == query.shape[0] else None
            extras[name] = (x, P(rows, exchanged_heads if x.shape[1] == heads else None))
    if sinks is not None:
        extras['sinks'] = (sinks, P(exchanged_heads))
    if key_value_seq_lengths is not None:
        extras['key_value_seq_lengths'] = (key_value_seq_lengths, query_split(
            ("activation_batch",), key_value_seq_lengths, queries, query.shape))
    if dropout_seed is not None:
        # Each shard's global rows and heads, split as its query rows and the exchanged heads.
        extras['dropout_seed'] = (dropout_seed, P(None))
        extras['dropout_rows'] = (jnp.arange(query.shape[0]), P(_entry(queries, 0)))
        extras['dropout_heads'] = (jnp.arange(heads), P(exchanged_heads))

    def local(query, key, value, *arrays):
        query, key, value = (jax.lax.all_to_all(x, SEQUENCE_AXIS, 2, 1, tiled=True)
                             for x in (query, key, value))
        arguments = dict(zip(extras, arrays, strict=True))
        if dropout_seed is not None:
            rows, heads = arguments.pop('dropout_rows'), arguments.pop('dropout_heads')
            arguments.update(dropout_rate=dropout_rate, dropout_places=template.DropPlaces(
                rows, heads, jnp.arange(query.shape[1]), jnp.arange(key.shape[1])))
        out = kernel(query, key, value, causal=causal, sliding_window=sliding_window, **arguments)
        return jax.lax.all_to_all(out, SEQUENCE_AXIS, 1, 2, tiled=True)

    exchanged = manual_map(
        local, (queries, keys, keys, *(spec for _, spec in extras.values())), queries)
    return exchanged(query, key, value, *(x for x, _ in extras.values()))


def gathered_keys_attention(kernel, query, key, value, shards: int, *, causal,
                            sliding_window, mask, bias, sinks, key_value_seq_lengths=None,
                            dropout_rate=0.0, dropout_seed=None):
    """Run `kernel` on each shard's slice of the queries against the whole
    keys and values, which every shard gathers over the sequence axis.

    Any head count and length, at the cost of every shard holding every key and
    value. Batch rows and heads split as `exchanged_heads_attention` splits
    them where the query heads divide by the tensor axis, and heads stay whole
    otherwise. A query length the shards do not divide is padded and the
    padding dropped; such a key length arrives whole. A causal, windowed or
    masked call stripes its queries for equal work and carries their positions
    in the mask, which the kernel reads in place of its causal flag, so the mask
    and the caller's rotary angles stay exact. A dropping call draws each
    shard's weights at their global rows, heads and queries, its keys whole
    (`template.DropPlaces`), so it drops what one device would.
    """
    q_len, kv_len = query.shape[1], key.shape[1]
    positions = jnp.arange(q_len)
    key, value = split_with_queries(key, value, tensor_shards(query))

    reordered = causal or sliding_window is not None or mask is not None
    if reordered:
        def rows_in_order(x: jax.Array | None) -> jax.Array | None:
            # A broadcast query row has the same value in either order.
            if x is not None and x.ndim >= 2 and x.shape[-2] == q_len:
                return stripe(x, shards, axis=-2)
            return x

        query = stripe(query, shards)
        positions = stripe(positions, shards, axis=0)
        mask, bias = rows_in_order(mask), rows_in_order(bias)
        structural = structural_mask(
            stripe(jnp.arange(q_len), shards, axis=0), kv_len, causal, sliding_window)
        if structural is not None:
            mask = structural if mask is None else jnp.logical_and(mask, structural)
    else:
        padding = -q_len % shards
        query = pad_rows(query, padding)
        positions = jnp.pad(positions, (0, padding))
        if bias is not None and bias.ndim >= 2 and bias.shape[-2] == q_len:
            bias = jnp.pad(bias, ((0, 0),) * (bias.ndim - 2) + ((0, padding), (0, 0)))

    # A key length the shards do not divide takes no sequence axis, and
    # arrives whole instead of gathered; so do no keys, whose all-gather
    # does not lower.
    queries, keys = logical_spec(HEADS, query.shape), logical_spec(KV_HEADS, key.shape)
    if not kv_len:
        keys = P(_entry(keys, 0), None, *keys[2:])
    gathered = SEQUENCE_AXIS in mesh_axes(_entry(keys, 1))
    extras = {}
    for name, x in (('mask', mask), ('bias', bias)):
        if x is not None:
            x = four_dimensional(x)
            extras[name] = (x, query_split(SCORES, x, queries, query.shape))
    if sinks is not None:
        extras['sinks'] = (sinks, query_split(("activation_heads",), sinks, queries, query.shape))
    if key_value_seq_lengths is not None:
        extras['key_value_seq_lengths'] = (key_value_seq_lengths, query_split(
            ("activation_batch",), key_value_seq_lengths, queries, query.shape))
    if dropout_seed is not None:
        # Each shard's global rows, heads and (striped) queries, split as its query's own.
        extras['dropout_seed'] = (dropout_seed, P(None))
        extras['dropout_rows'] = (jnp.arange(query.shape[0]), P(_entry(queries, 0)))
        extras['dropout_heads'] = (jnp.arange(query.shape[2]), P(_entry(queries, 2)))
        extras['dropout_queries'] = (positions, P(_entry(queries, 1)))

    def local(query, key, value, *arrays):
        if gathered:
            key, value = (jax.lax.all_gather(x, SEQUENCE_AXIS, axis=1, tiled=True)
                          for x in (key, value))
        arguments = dict(zip(extras, arrays, strict=True))
        if dropout_seed is not None:
            rows, heads, queries = (arguments.pop(name) for name in
                                    ('dropout_rows', 'dropout_heads', 'dropout_queries'))
            arguments.update(dropout_rate=dropout_rate, dropout_places=template.DropPlaces(
                rows, heads, queries, jnp.arange(key.shape[1])))
        return kernel(query, key, value, causal=False, sliding_window=None, **arguments)

    attended = manual_map(
        local, (queries, keys, keys, *(spec for _, spec in extras.values())), queries)
    out = attended(query, key, value, *(x for x, _ in extras.values()))
    return unstripe(out, shards) if reordered else out[:, :q_len]
