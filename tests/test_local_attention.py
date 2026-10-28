"""Bounded local attention against the masks MaxText and Splash define.

`local_attention` runs a sliding window or a chunk without the `[S, S]`
mask. The reference is dense attention under the mask Splash's own
`LocalMask` and `ChunkedCausalMask` produce, which is what MaxText's
`sliding_window_size` and `chunk_attn_window_size` select
(maxtext layers/attention_op.py, the splash mask construction): the band is
on par with it from its float64 run over drawn inputs (check 1,
tests/reference_error.py), as a model's chunked and windowed kinds are with
the same tokens alone, on one device or split. The cuDNN band alone keeps
its own check, until the route it takes changes (its test says why).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as splash
from model_support import model_draw
from reference_error import FEWEST_SEEDS, as_given, assert_as_exact, with_gradients

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import counted_places
from dew.nn.kernels import bf16_dot_runs
from dew.nn.kv_cache import KVCache
from dew.nn.local_attention import local_attention

SEEDS = {'window': 8, 'chunk': 8, 'packed': 8, 'given places': 8, 'gradients': 8, 'shared split': 8,
         'decode': 8, 'holey chunk': 8, 'diagnostic chunk': 8, 'holey window': 16, 'diagnostic window': 16,
         'prefill window': 16, 'padded split': 8}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` (on armada and an A100):
the most any of a site's cases took."""


def on_rows(out, rows):
    """`out` on the `[B, S]` query rows `rows`, zero elsewhere: a row that reads no key is no token's."""
    return jnp.where(jnp.asarray(rows)[:, :, None, None], out, 0)


def masked(keep, rows=None):
    """Dense grouped-head softmax attention under the `[B, S, S]` mask `keep` (the reference
    implementation of `scaled_dot_product_attention`) on the rows `rows`, by default those that read a
    key: check 1's reference, and run in float64 its truth."""
    mask = jnp.asarray(keep)[:, None]
    rows = np.asarray(keep).any(-1) if rows is None else rows
    def dense(q, k, v):
        return on_rows(scaled_dot_product_attention(q, k, v, implementation="reference", mask=mask), rows)
    return dense


def model_of(kind: str, **span):
    """A two-layer decoder whose first layer is `kind`, the local attention kind over `span`."""
    return CausalTransformer(
        vocab_size=64, num_layers=2, emb_features=32, num_heads=4, num_kv_heads=2,
        max_seq_len=24, layer_types=(kind, "full_attention"), kinds={kind: span}, attention_impl="xla")


def chunked_model(chunk: int):
    return model_of("chunked_attention", chunk=chunk)


def float64_run(model, **call):
    """`model` in float64 over widened variables, as a function of a draw: the truth of a site that
    holds two ways of running it to each other."""
    wide = model.clone(dtype=jnp.float64)
    return jax.jit(lambda variables, tokens: wide.apply(variables, tokens, **call))


def drawn(length, **shape):
    """Check 1's draw: a seed's query, key and value (`inputs`)."""
    return lambda seed: inputs(length, **shape, seed=seed)


def inputs(length, heads=4, kv_heads=2, width=16, batch=2, seed=0):
    keys = jax.random.split(jax.random.key(seed), 3)
    query = jax.random.normal(keys[0], (batch, length, heads, width), jnp.float32)
    key = jax.random.normal(keys[1], (batch, length, kv_heads, width), jnp.float32)
    value = jax.random.normal(keys[2], (batch, length, kv_heads, width), jnp.float32)
    return query, key, value


def packed(length, cuts):
    """Segment ids for documents ending at `cuts`, padding after."""
    segments, start = np.zeros(length, np.int32), 0
    for index, stop in enumerate(cuts, start=1):
        segments[start:stop] = index
        start = stop
    return segments


@pytest.mark.parametrize("implementation", ["reference", "xla", "template"])
@pytest.mark.parametrize("length,window", [(37, 8), (64, 16), (5, 8)])
def test_sliding_window_matches_splash_local_mask(implementation, length, window):
    keep = np.asarray(splash.LocalMask((length, length), (window - 1, 0), 0)[:, :])
    def band(q, k, v):
        return local_attention(q, k, v, window=window, implementation=implementation)

    assert_as_exact(jax.jit(band), jax.jit(masked(np.broadcast_to(keep, (2, length, length)))), drawn(length),
                    f"{implementation} window {window} over {length}", seeds=SEEDS['window'])


@pytest.mark.parametrize("implementation", ["reference", "xla", "template"])
@pytest.mark.parametrize("length,chunk", [(37, 8), (64, 16), (6, 8)])
def test_chunks_match_splash_chunked_causal_mask(implementation, length, chunk):
    keep = np.asarray(splash.ChunkedCausalMask((length, length), chunk)[:, :])
    def band(q, k, v):
        return local_attention(q, k, v, chunk=chunk, implementation=implementation)

    assert_as_exact(jax.jit(band), jax.jit(masked(np.broadcast_to(keep, (2, length, length)))), drawn(length),
                    f"{implementation} chunk {chunk} over {length}", seeds=SEEDS['chunk'])


@pytest.mark.parametrize("implementation", ["xla", "template"])
@pytest.mark.parametrize("span", [{"window": 5}, {"chunk": 4}, {"window": 16}, {"chunk": 16}])
def test_packed_documents_restart_their_windows_and_chunks(span, implementation):
    """Chunks count each document's tokens from its first, as it would alone;
    windows count the tokens inside the document; an invalid slot among them
    counts in neither, as the cache counts them. Padding keys (segment 0 and
    invalid) are never read by a real query, one of a document and valid; a
    padding query's output is no token's. The template takes the whole call as
    its spans, a chunk each query's run of its keys (`chunk_runs`)."""
    length = 29
    segments = packed(length, (7, 18, 26))
    valid = np.ones((2, length), bool)
    valid[1, 3] = False
    rows = np.arange(length)
    keep = (
        (rows[None, :] <= rows[:, None]) & (segments[:, None] == segments[None, :]) & (segments[:, None] != 0)
    )
    # Each slot's index among its row's real tokens, and among its document's.
    rank = np.cumsum(valid, axis=1) - 1
    starts = np.concatenate([[True], segments[1:] != segments[:-1]])
    places = rank - np.maximum.accumulate(np.where(starts[None], rank + 1 - valid, 0), axis=1)
    if "window" in span:
        keep = keep[None] & (rank[:, :, None] - rank[:, None, :] < span["window"])
    else:
        keep = keep[None] & (places[:, :, None] // span["chunk"] == places[:, None, :] // span["chunk"])
    keep = keep & valid[:, None, :]
    segment_ids = jnp.broadcast_to(segments, (2, length))
    real = (segments != 0) & valid & keep.any(-1)

    def band(q, k, v):
        places = counted_places(segment_ids, jnp.asarray(valid))
        return on_rows(local_attention(q, k, v, **span, places=places, segment_ids=segment_ids,
                                       valid=jnp.asarray(valid), implementation=implementation), real)

    assert_as_exact(jax.jit(band), jax.jit(masked(keep, real)), drawn(length),
                    f"{implementation} packed {span}", seeds=SEEDS['packed'])


@pytest.mark.parametrize("implementation", ["xla", "template"])
@pytest.mark.parametrize("given", ["origin", "repeated", "wide", "narrow"])
@pytest.mark.parametrize("length", [12, 8], ids=["banded", "dense"])
def test_chunks_count_tokens_from_the_place_a_caller_gives_their_first(given, length, implementation):
    """A chunk is cut at each token's place counted from the place its
    document's first token is given (`counted_places`): places given from 2
    put the first boundary after two tokens, as a continuation's whole pass
    cuts it; places that do not count the tokens (every one 0, as tokens
    sharing a place give) still chunk the tokens by their count. The band (12
    tokens, chunks of 4) and the dense call (8) read the same keys, where the
    band once read the given places within its two blocks and missed a
    chunk's earlier keys. Places given from 2^32 + 1 (64-bit), chunks of 3,
    cut where their own integers say: narrowed to 32 bits first, they would
    move every boundary; int8 places in chunks of 128, past their largest,
    are all one chunk (the template's runs, `chunk_runs`)."""
    chunk, origin = {"wide": (3, 2 ** 32 + 1), "narrow": (128, 0)}.get(given, (4, 2))
    rows = np.arange(length)
    places = rows + origin if given in ("origin", "wide") else rows
    keep = (rows[None, :] <= rows[:, None]) & (places[:, None] // chunk == places[None, :] // chunk)

    def band(q, k, v):
        if given == "narrow":
            return local_attention(q, k, v, chunk=chunk, places=jnp.arange(length, dtype=jnp.int8),
                                   implementation=implementation)
        positions = (jnp.arange(length) + origin if given in ("origin", "wide")
                     else jnp.zeros((length,), jnp.int32))
        places = counted_places(positions=positions)
        return local_attention(q, k, v, chunk=chunk, places=places, implementation=implementation)

    with jax.enable_x64(given == "wide"):
        assert_as_exact(jax.jit(band), jax.jit(masked(np.broadcast_to(keep, (2, length, length)))),
                        drawn(length), f"{implementation} {given} places over {length}",
                        seeds=SEEDS['given places'])


@pytest.mark.parametrize("implementation", ["xla", "template"])
@pytest.mark.parametrize("span", [{"window": 6}, {"chunk": 8}])
def test_gradients_match_the_dense_masked_kernel(span, implementation):
    """The band's output and its gradients under a drawn readout, its cotangent, for the query, key and
    value, on par with the dense masked kernel's (check 1)."""
    length = 40
    rows = np.arange(length)
    keep = rows[None, :] <= rows[:, None]
    if "window" in span:
        keep &= rows[:, None] - rows[None, :] < span["window"]
    else:
        keep &= rows[:, None] // span["chunk"] == rows[None, :] // span["chunk"]

    def draw(seed):
        query, key, value = inputs(length, seed=seed)
        readout = jax.random.normal(jax.random.fold_in(jax.random.key(0xC07), seed), query.shape)
        return query, key, value, readout

    def gradients(fn):
        return jax.jit(with_gradients(fn))

    def band(q, k, v):
        return local_attention(q, k, v, **span, implementation=implementation)

    assert_as_exact(gradients(band),
                    gradients(masked(np.broadcast_to(keep, (2, length, length)))), draw,
                    f"{implementation} {span} gradients", seeds=SEEDS['gradients'])


def test_memory_grows_with_the_window_not_the_square_of_the_sequence():
    """At 4096 tokens the dense windowed call holds `[H, S, S]` fp32 logits,
    256 MiB; the banded one holds `[H, S, 2W]`, a sixteenth of that at a
    window of 128. Both measured from the compiled executables."""
    length, window, heads = 4096, 128, 4
    query, key, value = inputs(length, heads=heads, kv_heads=heads, batch=1)

    def temporaries(fn):
        compiled = jax.jit(fn).lower(query, key, value).compile()
        return compiled.memory_analysis().temp_size_in_bytes

    dense = temporaries(lambda q, k, v: scaled_dot_product_attention(
        q, k, v, causal=True, sliding_window=window, implementation="xla"))
    banded = temporaries(lambda q, k, v: local_attention(
        q, k, v, window=window, implementation="xla"))
    assert dense >= heads * length * length * 4
    assert banded * 4 < dense


@pytest.mark.mesh(devices=2)
def test_chunks_from_a_shared_vector_of_places_split_over_an_explicit_sequence_axis():
    """Places given as one `[S]` vector the rows share, from 2, split over an Explicit sequence axis
    with the tokens: the chunks are cut from the given origin, as on one device, the vector counted as
    a row (`counted_places`)."""
    from jax.sharding import NamedSharding, PartitionSpec

    from dew.training import MeshSpec

    length = 16
    positions = jnp.arange(length) + 2
    rows = np.arange(length)
    keep = (rows[None, :] <= rows[:, None]) & ((rows[:, None] + 2) // 4 == (rows[None, :] + 2) // 4)

    def attend(q, k, v, positions):
        return local_attention(q, k, v, chunk=4, places=counted_places(positions=positions),
                               implementation="xla")

    mesh = MeshSpec(sequence=2, explicit=("sequence",)).build(jax.devices()[:2])
    run = jax.jit(attend)

    @as_given
    def split(q, k, v):
        with jax.set_mesh(mesh):
            placed = [jax.device_put(x, NamedSharding(mesh, PartitionSpec(None, "sequence")))
                      for x in (q, k, v)]
            shared = jax.device_put(positions, NamedSharding(mesh, PartitionSpec("sequence")))
            return np.asarray(run(*placed, shared))

    assert_as_exact(split, jax.jit(lambda q, k, v: attend(q, k, v, positions)), drawn(length, batch=1),
                    "shared places split", seeds=SEEDS['shared split'], truth_fn=masked(keep[None]))


@pytest.mark.mesh(devices=2)
def test_a_row_with_padding_among_its_tokens_bands_its_window_over_a_sequence_axis():
    """Split over a sequence axis, a row with padding among its tokens still runs its window
    banded, on its real tokens compacted: its temporaries a fraction of the dense windowed call's
    `[H, S, S]` logits at 4096 tokens and a window of 128, both measured from the compiled
    executables on the same mesh."""
    from jax.sharding import NamedSharding, PartitionSpec

    from dew.training import MeshSpec

    length, window, heads = 4096, 128, 4
    query, key, value = inputs(length, heads=heads, kv_heads=heads, batch=1)
    valid = jnp.asarray(np.arange(length) % 7 != 3)[None]
    mesh = MeshSpec(sequence=2).build(jax.devices()[:2])
    with jax.set_mesh(mesh):
        placed = [jax.device_put(x, NamedSharding(mesh, PartitionSpec(None, "sequence")))
                  for x in (query, key, value, valid)]

        def temporaries(fn):
            compiled = jax.jit(fn).lower(*placed).compile()
            return compiled.memory_analysis().temp_size_in_bytes

        dense = temporaries(lambda q, k, v, valid: scaled_dot_product_attention(
            q, k, v, causal=True, sliding_window=window, mask=valid[:, None, None, :], implementation="xla"))
        banded = temporaries(lambda q, k, v, valid: local_attention(
            q, k, v, window=window, valid=valid, implementation="xla"))
    assert banded * 4 < dense


@pytest.mark.mesh(devices=8)
def test_row_fields_split_as_the_query_rows_do_where_the_heads_contend_for_the_axis():
    """Eight query heads over a tensor and an expert axis of two each, which the heads take before the rows
    (the rows' rule names the expert axis too), and a sequence axis of two: the window's segment ids and
    validity split as the query's own rows do (`query_split`), not as their names alone would, so each shard
    bands its own rows'. The same output as the whole sequence's."""
    from flax import linen as nn

    from dew.mesh import DEFAULT_RULES
    from dew.training import MeshSpec

    length, window = 64, 8
    segment_ids = jnp.asarray(np.tile(packed(length, (20, 45, 60)), (4, 1)))
    valid = jnp.asarray(np.arange(length) % 7 != 3)[None].repeat(4, axis=0)

    def attend(q, k, v):
        return local_attention(q, k, v, window=window, segment_ids=segment_ids, valid=valid,
                               implementation="xla")

    def draw(seed):
        return inputs(length, heads=8, kv_heads=2, batch=4, seed=seed)

    rules = (("activation_heads", ("tensor", "expert")), ("activation_kv", ("tensor", "expert")),
             ("activation_batch", "expert"), *DEFAULT_RULES)
    mesh, banded = MeshSpec(tensor=2, expert=2, sequence=2).build(), jax.jit(attend)

    @as_given
    def split(query, key, value):
        # The band under the mesh and the contending rules; the whole sequence outside them.
        with jax.set_mesh(mesh), nn.logical_axis_rules(rules):
            return banded(query, key, value)

    assert_as_exact(split, jax.jit(attend), draw, "local attention under contending heads",
                    seeds=FEWEST_SEEDS)


@pytest.mark.parametrize("layout", [KVCache(), KVCache(page_size=4)], ids=["dense", "paged"])
@pytest.mark.parametrize("shared", [False, True], ids=["per-row", "shared"])
@pytest.mark.parametrize("origin", [0, 2], ids=["from-zero", "logical-origin"])
def test_a_chunked_kind_decodes_what_its_whole_sequence_pass_computes(origin, shared, layout):
    """The whole-sequence pass runs `local_attention`; the cached decode runs
    the chunk as a mask over cache slots. Both read the same keys: chunks
    are cut at the tokens' logical places, so a row whose places start at
    `origin` (a continuation, a canvas) cuts them as its whole pass does,
    not at its cache slots, whether each row gives its places or the rows
    share one vector of them, and whether the cache keeps its keys in rows
    or in a pool of pages: the decode is on par with the whole pass over
    drawn models and tokens (check 1)."""
    model = chunked_model(5).with_kv_cache(layout)
    shape = (2, 13)
    positions = jnp.arange(13) + origin if shared else jnp.broadcast_to(jnp.arange(13) + origin, shape)
    made = jax.jit(lambda variables: model.apply(variables, 2, method=CausalTransformer.init_cache,
                                                 mutable=["cache"])[1]["cache"])
    step = jax.jit(lambda variables, cache, tokens, places: model.apply(
        {**variables, "cache": cache}, tokens, positions=places, decode=True, mutable=["cache"]))

    @as_given
    def decoded(variables, tokens):
        logits, mutated = step(variables, made(variables), tokens[:, :4], positions[..., :4])
        steps = [logits[:, -1]]
        for index in range(4, 13):
            logits, mutated = step(variables, mutated["cache"], tokens[:, index:index + 1],
                                   positions[..., index:index + 1])
            steps.append(logits[:, -1])
        return np.stack(steps, 1)

    def whole(model):
        return jax.jit(lambda variables, tokens: model.apply(variables, tokens, positions=positions)[:, 3:])

    assert_as_exact(decoded, whole(model), model_draw(model, shape, 64),
                    f"{layout} {shared} {origin} chunked decode", seeds=SEEDS['decode'],
                    truth_fn=whole(model.clone(dtype=jnp.float64)))


def test_a_chunked_kind_bands_a_row_with_padding_among_its_tokens_by_its_tokens():
    """Chunks span tokens, not slots: a row whose four tokens sit at slots 0,
    3, 6 and 9 among padding, a chunk of 4, attends as the four tokens alone
    do, on par with the row alone from its float64 run (check 1)."""
    model = chunked_model(4)
    slots = jnp.asarray([0, 3, 6, 9])
    valid = jnp.zeros((1, 12), bool).at[:, slots].set(True)

    def holey(variables, real):
        tokens = jnp.zeros((1, 12), jnp.int32).at[:, slots].set(real)
        return model.apply(variables, tokens, attention_mask=valid)[:, slots]

    assert_as_exact(jax.jit(holey), jax.jit(model.apply), model_draw(model, (1, 4), 64), "holey chunked row",
                    seeds=SEEDS['holey chunk'], truth_fn=float64_run(model))


def test_a_chunked_kind_attends_alike_on_every_path_whatever_places_it_is_given():
    """Given places that do not count the tokens (every token's 0, as tokens
    sharing one place give), the whole pass (the band over 13 tokens in
    chunks of 5) and its diagnostic path (an open `qk` collection builds the
    dense mask) chunk the tokens alike, by their count: the whole pass on par
    with the diagnostic path from its float64 run (check 1)."""
    model = chunked_model(5)
    positions = jnp.zeros((2, 13), jnp.int32)

    def diagnostic(model):
        return jax.jit(lambda variables, tokens: model.apply(variables, tokens, positions=positions,
                                                             mutable=["qk"])[0])

    assert_as_exact(jax.jit(lambda variables, tokens: model.apply(variables, tokens, positions=positions)),
                    diagnostic(model), model_draw(model, (2, 13), 64), "chunked whole pass",
                    seeds=SEEDS['diagnostic chunk'], truth_fn=diagnostic(model.clone(dtype=jnp.float64)))


@pytest.mark.parametrize("window", [2, 8], ids=["banded", "dense"])
def test_a_windowed_kind_counts_a_rows_tokens_not_its_slots(window):
    """A window spans tokens, as the cache counts them: a row whose six tokens
    sit among padding attends as the six tokens alone do, on par with the row
    alone from its float64 run (check 1), through the band (a window of 2 over
    16 slots) and the dense call (a window of 8), as does its diagnostic path
    (an open `qk` collection builds the dense mask), and its cached prefill
    computes what its whole pass does. A window counted in slots let the last
    token of slots 0, 3 and 6 see itself alone."""
    model = model_of("sliding_attention", window=window)
    slots = jnp.asarray([0, 3, 6, 7, 11, 15])
    valid = jnp.zeros((1, 16), bool).at[:, slots].set(True)

    def spread(real):
        return jnp.zeros((1, 16), jnp.int32).at[:, slots].set(real)

    def holey(variables, real):
        return model.apply(variables, spread(real), attention_mask=valid)[:, slots]

    def diagnostic(variables, real):
        return model.apply(variables, spread(real), attention_mask=valid, mutable=["qk"])[0][:, slots]

    def prefilled(variables, real):
        cache = model.apply(variables, 1, method=CausalTransformer.init_cache, mutable=["cache"])[1]["cache"]
        return model.apply({**variables, "cache": cache}, spread(real), attention_mask=valid, decode=True,
                           mutable=["cache"])[0][:, slots]

    draw, alone, truth = model_draw(model, (1, 6), 64), jax.jit(model.apply), float64_run(model)
    paths = {"holey window": holey, "diagnostic window": diagnostic, "prefill window": prefilled}
    for label, path in paths.items():
        assert_as_exact(jax.jit(path), alone, draw, f"{window} {label}", seeds=SEEDS[label], truth_fn=truth)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("explicit", [False, True], ids=["auto", "explicit"])
@pytest.mark.parametrize("span", [{"window": 2}, {"chunk": 4}])
def test_a_row_with_padding_among_its_tokens_counts_its_tokens_over_a_sequence_axis(span, explicit):
    """A sequence axis that splits a packed row with padding among its tokens
    in two, Auto or Explicit, its rows placed split: each real query reads
    the keys of its document its tokens' counts say, as near the float64
    truth as the same call on one device, which compacts the row. Compacting
    sorts the sequence axis, which JAX refuses on an Explicit axis that
    splits it, so the split call counts in place, its mask built from fields
    each device holds whole."""
    from jax.sharding import NamedSharding, PartitionSpec

    from dew.training import MeshSpec

    length = 16
    valid = jnp.asarray([[True, False, False, True, True, False, True, True] * 2])
    segments = jnp.asarray(packed(length, (11, length))[None])
    rank = np.cumsum(np.asarray(valid), axis=1) - 1
    # Each token's index among its document's tokens: the second starts at slot 11, rank 6.
    place = np.where(np.arange(length) < 11, rank, rank - 6)
    keep = rank[:, None, :] <= rank[:, :, None]
    keep &= np.asarray(segments)[:, :, None] == np.asarray(segments)[:, None, :]
    if "window" in span:
        keep &= rank[:, :, None] - rank[:, None, :] < span["window"]
    else:
        keep &= place[:, :, None] // span["chunk"] == place[:, None, :] // span["chunk"]
    keep &= np.asarray(valid)[:, None, :]
    real = np.asarray(valid)

    def attend(q, k, v, valid, segments):
        return on_rows(local_attention(q, k, v, **span, places=counted_places(segments, valid),
                                       segment_ids=segments, valid=valid, implementation="xla"), real)

    mesh = MeshSpec(sequence=2, explicit=("sequence",) if explicit else ()).build(jax.devices()[:2])
    run = jax.jit(attend)

    @as_given
    def split(q, k, v):
        with jax.set_mesh(mesh):
            placed = [jax.device_put(x, NamedSharding(mesh, PartitionSpec(None, "sequence")))
                      for x in (q, k, v, valid, segments)]
            return np.asarray(run(*placed))

    assert_as_exact(split, jax.jit(lambda q, k, v: attend(q, k, v, valid, segments)), drawn(length, batch=1),
                    f"{span} {explicit} padded split", seeds=SEEDS['padded split'],
                    truth_fn=masked(keep, real))


@pytest.mark.gpu(reason="the band's GPU route, bf16 on a cuda device of sm80 or later")
@pytest.mark.skipif(not bf16_dot_runs(), reason="the band's GPU route, bf16 on a device of sm80 or later")
def test_packed_windowed_bf16_attention_runs_its_band_on_par(without_deterministic_ops):
    """'auto' runs the band of a packed, windowed bf16 call, forward and gradients, on par with the xla band
    over drawn calls (check 1, float64 the truth), padding rows included, on whatever kernel its route takes:
    the accuracy gate and the measured speed decide the route, never one kernel's equality to another's
    (bible §4). Under ledger row 1321 cuDNN's backward left dQ off zero on a query that reads one key, where
    the xla band is exactly zero, which check 1's zero rows fail: a route through it is red here."""
    batch, length, heads, kv_heads, width, window = 1, 1024, 8, 2, 64, 256
    # Three documents fill 880 of 1024 rows; the rest is padding, segment 0.
    lengths = [300, 200, 380]
    segments = np.concatenate([np.repeat(np.arange(1, 4), lengths),
                               np.zeros(length - sum(lengths), int)])[None]
    segment_ids = jnp.asarray(segments, jnp.int32)

    def draw(seed):
        streams = jax.random.split(jax.random.key(seed), 4)
        return (jax.random.normal(streams[0], (batch, length, heads, width), jnp.bfloat16),
                jax.random.normal(streams[1], (batch, length, kv_heads, width), jnp.bfloat16),
                jax.random.normal(streams[2], (batch, length, kv_heads, width), jnp.bfloat16),
                jax.random.normal(streams[3], (batch, length, heads, width), jnp.bfloat16))

    def band(implementation):
        return jax.jit(with_gradients(lambda q, k, v: local_attention(
            q, k, v, window=window, segment_ids=segment_ids, implementation=implementation)))

    assert_as_exact(band("auto"), band("xla"), draw, "packed windowed band", seeds=FEWEST_SEEDS)
