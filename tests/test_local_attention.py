"""Bounded local attention against the masks MaxText and Splash define.

`local_attention` runs a sliding window or a chunk without the `[S, S]`
mask. The reference is dense float64 attention under the mask Splash's own
`LocalMask` and `ChunkedCausalMask` produce, which is what MaxText's
`sliding_window_size` and `chunk_attn_window_size` select
(maxtext layers/attention_op.py, the splash mask construction).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as splash
from reference_error import assert_as_exact_as_the_reference

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import counted_places
from dew.nn.kv_cache import KVCache
from dew.nn.kernels import bf16_dot_runs
from dew.nn.local_attention import local_attention


def dense_reference(query, key, value, keep):
    """float64 softmax attention with grouped heads under a `[B, S, S]` mask."""
    query, key, value = (np.asarray(x, np.float64) for x in (query, key, value))
    groups = query.shape[2] // key.shape[2]
    key, value = np.repeat(key, groups, axis=2), np.repeat(value, groups, axis=2)
    logits = np.einsum("bqhd,bkhd->bhqk", query, key) / np.sqrt(query.shape[-1])
    logits = np.where(keep[:, None], logits, -np.inf)
    with np.errstate(invalid="ignore"):  # a padding row reads no key; callers skip it
        weights = np.exp(logits - logits.max(-1, keepdims=True))
        weights = weights / weights.sum(-1, keepdims=True)
    return np.einsum("bhqk,bkhd->bqhd", weights, value)


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


@pytest.mark.parametrize("implementation", ["reference", "xla"])
@pytest.mark.parametrize("length,window", [(37, 8), (64, 16), (5, 8)])
def test_sliding_window_matches_splash_local_mask(implementation, length, window):
    query, key, value = inputs(length)
    keep = np.asarray(splash.LocalMask((length, length), (window - 1, 0), 0)[:, :])
    expected = dense_reference(query, key, value, np.broadcast_to(keep, (2, length, length)))
    actual = local_attention(query, key, value, window=window, implementation=implementation)
    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-6, rtol=0)


@pytest.mark.parametrize("implementation", ["reference", "xla"])
@pytest.mark.parametrize("length,chunk", [(37, 8), (64, 16), (6, 8)])
def test_chunks_match_splash_chunked_causal_mask(implementation, length, chunk):
    query, key, value = inputs(length)
    keep = np.asarray(splash.ChunkedCausalMask((length, length), chunk)[:, :])
    expected = dense_reference(query, key, value, np.broadcast_to(keep, (2, length, length)))
    actual = local_attention(query, key, value, chunk=chunk, implementation=implementation)
    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-6, rtol=0)


@pytest.mark.parametrize("span", [{"window": 5}, {"chunk": 4}, {"window": 16}, {"chunk": 16}])
def test_packed_documents_restart_their_windows_and_chunks(span):
    """Chunks count each document's tokens from its first, as it would alone;
    windows count the tokens inside the document; an invalid slot among them
    counts in neither, as the cache counts them. Padding keys (segment 0 and
    invalid) are never read by a real query, one of a document and valid; a
    padding query's output is no token's."""
    length = 29
    query, key, value = inputs(length)
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
    expected = dense_reference(query, key, value, keep)
    segment_ids = jnp.broadcast_to(segments, (2, length))
    actual = local_attention(
        query, key, value, **span, places=counted_places(segment_ids, jnp.asarray(valid)),
        segment_ids=segment_ids, valid=jnp.asarray(valid), implementation="xla")
    real = (segments != 0) & valid & keep.any(-1)
    np.testing.assert_allclose(np.asarray(actual)[real], expected[real], atol=2e-6, rtol=0)


@pytest.mark.parametrize("given", ["origin", "repeated"])
@pytest.mark.parametrize("length", [12, 8], ids=["banded", "dense"])
def test_chunks_count_tokens_from_the_place_a_caller_gives_their_first(given, length):
    """A chunk is cut at each token's place counted from the place its
    document's first token is given (`counted_places`): places given from 2
    put the first boundary after two tokens, as a continuation's whole pass
    cuts it; places that do not count the tokens (every one 0, as tokens
    sharing a place give) still chunk the tokens by their count. The band (12
    tokens, chunks of 4) and the dense call (8) read the same keys, where the
    band once read the given places within its two blocks and missed a
    chunk's earlier keys."""
    query, key, value = inputs(length)
    positions = jnp.arange(length) + 2 if given == "origin" else jnp.zeros((length,), jnp.int32)
    rows = np.arange(length)
    places = rows + 2 if given == "origin" else rows
    keep = (rows[None, :] <= rows[:, None]) & (places[:, None] // 4 == places[None, :] // 4)
    expected = dense_reference(query, key, value, np.broadcast_to(keep, (2, length, length)))
    actual = local_attention(query, key, value, chunk=4, places=counted_places(positions=positions),
                             implementation="xla")
    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-6, rtol=0)


@pytest.mark.parametrize("span", [{"window": 6}, {"chunk": 8}])
def test_gradients_match_the_dense_masked_kernel(span):
    length = 40
    query, key, value = inputs(length)
    rows = np.arange(length)
    keep = rows[None, :] <= rows[:, None]
    if "window" in span:
        keep &= rows[:, None] - rows[None, :] < span["window"]
    else:
        keep &= rows[:, None] // span["chunk"] == rows[None, :] // span["chunk"]
    mask = jnp.asarray(keep)[None, None]
    cotangent = jax.random.normal(jax.random.key(3), query.shape)

    def loss(fn):
        return lambda q, k, v: jnp.sum(fn(q, k, v) * cotangent)

    dense = jax.grad(loss(lambda q, k, v: scaled_dot_product_attention(
        q, k, v, implementation="reference", mask=mask)), argnums=(0, 1, 2))(query, key, value)
    local = jax.grad(loss(lambda q, k, v: local_attention(
        q, k, v, **span, implementation="xla")), argnums=(0, 1, 2))(query, key, value)
    for expected, actual in zip(dense, local, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-5, rtol=0)


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
    query, key, value = inputs(length, batch=1)
    positions = jnp.arange(length) + 2
    rows = np.arange(length)
    keep = (rows[None, :] <= rows[:, None]) & ((rows[:, None] + 2) // 4 == (rows[None, :] + 2) // 4)
    truth = dense_reference(query, key, value, keep[None])

    def attend(q, k, v, positions):
        return local_attention(q, k, v, chunk=4, places=counted_places(positions=positions),
                               implementation="xla")

    alone = attend(query, key, value, positions)
    mesh = MeshSpec(sequence=2, explicit=("sequence",)).build(jax.devices()[:2])
    with jax.set_mesh(mesh):
        placed = [jax.device_put(x, NamedSharding(mesh, PartitionSpec(None, "sequence")))
                  for x in (query, key, value)]
        shared = jax.device_put(positions, NamedSharding(mesh, PartitionSpec("sequence")))
        split = jax.jit(attend)(*placed, shared)
    assert_as_exact_as_the_reference(np.asarray(split), np.asarray(alone), truth, "shared places split")


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
    or in a pool of pages."""
    model = CausalTransformer(
        vocab_size=64, num_layers=2, emb_features=32, num_heads=4, num_kv_heads=2,
        max_seq_len=24, layer_types=("chunked_attention", "full_attention"),
        kinds={"chunked_attention": {"chunk": 5}}, attention_impl="xla").with_kv_cache(layout)
    tokens = jax.random.randint(jax.random.key(1), (2, 13), 0, 64)
    positions = jnp.arange(13) + origin if shared else jnp.broadcast_to(jnp.arange(13) + origin, tokens.shape)
    variables = model.init(jax.random.key(0), tokens)
    whole = model.apply(variables, tokens, positions=positions)
    cache = model.apply(variables, 2, method=CausalTransformer.init_cache, mutable=["cache"])[1]["cache"]
    logits, mutated = model.apply({**variables, "cache": cache}, tokens[:, :4], positions=positions[..., :4],
                                  decode=True, mutable=["cache"])
    steps = [logits[:, -1]]
    for index in range(4, 13):
        logits, mutated = model.apply({**variables, "cache": mutated["cache"]}, tokens[:, index:index + 1],
                                      positions=positions[..., index:index + 1], decode=True,
                                      mutable=["cache"])
        steps.append(logits[:, -1])
    np.testing.assert_allclose(np.stack(steps, 1), whole[:, 3:], atol=1e-5, rtol=0)


def test_a_chunked_kind_bands_a_row_with_padding_among_its_tokens_by_its_tokens():
    """Chunks span tokens, not slots: a row whose four tokens sit at slots 0,
    3, 6 and 9 among padding, a chunk of 4, attends as the four tokens alone
    do, as near the float64 truth of the row alone as the row alone's float32
    run."""
    model = CausalTransformer(
        vocab_size=64, num_layers=2, emb_features=32, num_heads=4, num_kv_heads=2,
        max_seq_len=24, layer_types=("chunked_attention", "full_attention"),
        kinds={"chunked_attention": {"chunk": 4}}, attention_impl="xla")
    real = jax.random.randint(jax.random.key(1), (1, 4), 0, 64)
    slots = jnp.asarray([0, 3, 6, 9])
    tokens = jnp.zeros((1, 12), jnp.int32).at[:, slots].set(real)
    valid = jnp.zeros((1, 12), bool).at[:, slots].set(True)
    variables = model.init(jax.random.key(0), tokens)
    holey = model.apply(variables, tokens, attention_mask=valid)[:, slots]
    alone = model.apply(variables, real)
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables)
        truth = model.clone(dtype=jnp.float64).apply(wide, real)
    assert_as_exact_as_the_reference(np.asarray(holey), np.asarray(alone), np.asarray(truth, np.float64),
                                     "holey row")


def test_a_chunked_kind_attends_alike_on_every_path_whatever_places_it_is_given():
    """Given places that do not count the tokens (every token's 0, as tokens
    sharing one place give), the whole pass (the band over 13 tokens in
    chunks of 5) and its diagnostic path (an open `qk` collection builds the
    dense mask) chunk the tokens alike, by their count, as near the
    diagnostic path's float64 truth as the diagnostic path itself."""
    model = CausalTransformer(
        vocab_size=64, num_layers=2, emb_features=32, num_heads=4, num_kv_heads=2,
        max_seq_len=24, layer_types=("chunked_attention", "full_attention"),
        kinds={"chunked_attention": {"chunk": 5}}, attention_impl="xla")
    tokens = jax.random.randint(jax.random.key(1), (2, 13), 0, 64)
    positions = jnp.zeros(tokens.shape, jnp.int32)
    variables = model.init(jax.random.key(0), tokens)

    def diagnostic(model, variables):
        return model.apply(variables, tokens, positions=positions, mutable=["qk"])[0]

    whole = model.apply(variables, tokens, positions=positions)
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables)
        truth = np.asarray(diagnostic(model.clone(dtype=jnp.float64), wide), np.float64)
    assert_as_exact_as_the_reference(np.asarray(whole), np.asarray(diagnostic(model, variables)), truth,
                                     "whole pass")


@pytest.mark.parametrize("window", [2, 8], ids=["banded", "dense"])
def test_a_windowed_kind_counts_a_rows_tokens_not_its_slots(window):
    """A window spans tokens, as the cache counts them: a row whose six tokens
    sit among padding attends as the six tokens alone do, as near the float64
    truth of the row alone as the row alone's float32 run, through the band
    (a window of 2 over 16 slots) and the dense call (a window of 8), as does
    its diagnostic path (an open `qk` collection builds the dense mask), and
    its cached prefill computes what its whole pass does. A window counted in
    slots let the last token of slots 0, 3 and 6 see itself alone."""
    model = CausalTransformer(
        vocab_size=64, num_layers=2, emb_features=32, num_heads=4, num_kv_heads=2,
        max_seq_len=24, layer_types=("sliding_attention", "full_attention"),
        kinds={"sliding_attention": {"window": window}}, attention_impl="xla")
    real = jax.random.randint(jax.random.key(1), (1, 6), 0, 64)
    slots = jnp.asarray([0, 3, 6, 7, 11, 15])
    tokens = jnp.zeros((1, 16), jnp.int32).at[:, slots].set(real)
    valid = jnp.zeros((1, 16), bool).at[:, slots].set(True)
    variables = model.init(jax.random.key(0), tokens)
    holey = model.apply(variables, tokens, attention_mask=valid)
    alone = model.apply(variables, real)
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables)
        truth = model.clone(dtype=jnp.float64).apply(wide, real)
    assert_as_exact_as_the_reference(np.asarray(holey[:, slots]), np.asarray(alone),
                                     np.asarray(truth, np.float64), "holey row")
    diagnostic = model.apply(variables, tokens, attention_mask=valid, mutable=["qk"])[0]
    assert_as_exact_as_the_reference(np.asarray(diagnostic[:, slots]), np.asarray(alone),
                                     np.asarray(truth, np.float64), "diagnostic path")
    cache = model.apply(variables, 1, method=CausalTransformer.init_cache, mutable=["cache"])[1]["cache"]
    prefilled, _ = model.apply({**variables, "cache": cache}, tokens, attention_mask=valid, decode=True,
                               mutable=["cache"])
    assert_as_exact_as_the_reference(np.asarray(prefilled[:, slots]), np.asarray(alone),
                                     np.asarray(truth, np.float64), "cached prefill")


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
    query, key, value = inputs(length, batch=1)
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
    truth = dense_reference(query, key, value, keep)

    def attend(q, k, v, valid, segments):
        return local_attention(q, k, v, **span, places=counted_places(segments, valid), segment_ids=segments,
                               valid=valid, implementation="xla")

    alone = attend(query, key, value, valid, segments)
    mesh = MeshSpec(sequence=2, explicit=("sequence",) if explicit else ()).build(jax.devices()[:2])
    with jax.set_mesh(mesh):
        placed = [jax.device_put(x, NamedSharding(mesh, PartitionSpec(None, "sequence")))
                  for x in (query, key, value, valid, segments)]
        split = jax.jit(attend)(*placed)
    real = np.asarray(valid)[0]
    assert_as_exact_as_the_reference(np.asarray(split)[:, real], np.asarray(alone)[:, real], truth[:, real],
                                     f"{span} split")


def custom_call_targets(lowered) -> list[str]:
    """The target of every custom call in a lowered program's StableHLO module, in program order."""
    from jax.extend.mlir import ir

    targets: list[str] = []

    def visit(operation):
        if operation.name == "stablehlo.custom_call":
            targets.append(ir.StringAttr(operation.attributes["call_target_name"]).value)
        return ir.WalkResult.ADVANCE

    module = lowered.compiler_ir(dialect="stablehlo")
    with module.context:
        module.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    return targets


@pytest.mark.gpu(reason="needs a cuda device of sm80 or later, cuDNN's bf16 floor")
@pytest.mark.skipif(not bf16_dot_runs(), reason="needs a cuda device of sm80 or later, cuDNN's bf16 floor")
def test_packed_windowed_bf16_attention_runs_its_band_on_cudnn(without_deterministic_ops):
    """'auto' hands the band mask of a packed, windowed bf16 call to cuDNN
    as a bias: finite everywhere, padding rows included, and as close to a
    float64 oracle as the xla band is."""
    batch, length, heads, kv_heads, width, window = 1, 1024, 8, 2, 64, 256
    # Three documents fill 880 of 1024 rows; the rest is padding, segment 0.
    lengths = [300, 200, 380]
    segments = np.concatenate([np.repeat(np.arange(1, 4), lengths),
                               np.zeros(length - sum(lengths), int)])[None]
    segment_ids = jnp.asarray(segments, jnp.int32)
    keys = jax.random.split(jax.random.key(0), 4)
    query = jax.random.normal(keys[0], (batch, length, heads, width), jnp.bfloat16)
    key = jax.random.normal(keys[1], (batch, length, kv_heads, width), jnp.bfloat16)
    value = jax.random.normal(keys[2], (batch, length, kv_heads, width), jnp.bfloat16)
    cotangent = jax.random.normal(keys[3], (batch, length, heads, width), jnp.float32)
    real = jnp.asarray(segments[0] != 0)

    def loss(implementation):
        def run(query, key, value):
            out = local_attention(query, key, value, window=window, segment_ids=segment_ids,
                                  implementation=implementation)
            return jnp.sum(out.astype(jnp.float32) * cotangent * real[None, :, None, None]), out
        return jax.jit(jax.value_and_grad(run, argnums=(0, 1, 2), has_aux=True))

    targets = custom_call_targets(loss("auto").lower(query, key, value))
    assert any(target.startswith("__cudnn$fmha") for target in targets), targets

    def oracle(query, key, value):
        q, k, v = (x.astype(jnp.float64) for x in (query, key, value))
        k, v = (jnp.repeat(x, heads // kv_heads, axis=2) for x in (k, v))
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(width)
        rows = jnp.arange(length)
        keep = ((rows[None, :] <= rows[:, None]) & (rows[:, None] - rows[None, :] < window)
                & (segment_ids[0][:, None] == segment_ids[0][None, :]) & real[:, None])
        logits = jnp.where(keep[None, None], logits, -jnp.inf)
        weights = jnp.nan_to_num(jax.nn.softmax(logits, axis=-1))
        out = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
        return jnp.sum(out * cotangent * real[None, :, None, None]), out

    with jax.enable_x64(new_val=True):
        (_, want), wants = jax.value_and_grad(oracle, argnums=(0, 1, 2), has_aux=True)(
            query, key, value)
    (_, got), grads = loss("auto")(query, key, value)
    (_, xla), xla_grads = loss("xla")(query, key, value)
    assert all(bool(jnp.all(jnp.isfinite(x.astype(jnp.float32)))) for x in (got, *grads))

    def error(have, reference):
        have = np.asarray(have, np.float64)[:, np.asarray(real)] if have.ndim == 4 else have
        reference = np.asarray(reference)[:, np.asarray(real)] if reference.ndim == 4 else reference
        return np.max(np.abs(np.asarray(have, np.float64) - reference)) / np.max(np.abs(reference))

    assert error(got, want) <= 2 * error(xla, want)
    for have, baseline, reference in zip(grads, xla_grads, wants, strict=True):
        assert error(have, reference) <= 2 * error(baseline, reference)
