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

from dew.nn.attention import local_attention, scaled_dot_product_attention
from dew.nn.backbones.causal_transformer import CausalTransformer


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
    """Segment ids and in-document positions for documents ending at `cuts`, padding after."""
    segments, positions, start = np.zeros(length, np.int32), np.zeros(length, np.int32), 0
    for index, stop in enumerate(cuts, start=1):
        segments[start:stop] = index
        positions[start:stop] = np.arange(stop - start)
        start = stop
    return segments, positions


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
    """Chunks count from each document's first token, as `chunked_overlay`
    reads packed positions; windows count rows inside the document. Padding
    keys (segment 0 and invalid) are never read by a real query."""
    length = 29
    query, key, value = inputs(length)
    segments, positions = packed(length, (7, 18, 26))
    valid = np.ones((2, length), bool)
    valid[1, 3] = False
    rows = np.arange(length)
    keep = (rows[None, :] <= rows[:, None]) & (segments[:, None] == segments[None, :]) & (segments[:, None] != 0)
    if "window" in span:
        keep &= rows[:, None] - rows[None, :] < span["window"]
    else:
        keep &= positions[:, None] // span["chunk"] == positions[None, :] // span["chunk"]
    keep = keep[None] & valid[:, None, :]
    expected = dense_reference(query, key, value, keep)
    actual = local_attention(
        query, key, value, **span, segment_ids=jnp.broadcast_to(segments, (2, length)),
        positions=None if "window" in span else jnp.asarray(positions), valid=jnp.asarray(valid),
        implementation="xla")
    real = (segments != 0) & keep.any(-1)
    np.testing.assert_allclose(np.asarray(actual)[real], expected[real], atol=2e-6, rtol=0)


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


@pytest.mark.parametrize("span,length,packing", [
    ({"window": 4096}, 256, True),
    ({"chunk": 8192}, 1024, True),
    ({"chunk": 8192}, 1024, False),
    ({"chunk": 8192}, 2048, False),
])
def test_a_sequence_within_two_spans_costs_no_more_than_the_dense_call(span, length, packing):
    """Banding pads to whole spans, so at or below two spans it would hold
    more than the `[S, S]` mask it replaces: those calls take the dense mask,
    and the compiled temporaries stay at the dense call's."""
    heads = 4
    query, key, value = inputs(length, heads=heads, kv_heads=heads, batch=1)
    segments, positions = packed(length, (length // 3, length - 5))
    segments, positions = jnp.asarray(segments)[None], jnp.asarray(positions)
    rows = jnp.arange(length)
    keep = rows[None, :] <= rows[:, None]
    if packing:
        keep = keep & (segments[0, :, None] == segments[0, None, :]) & (segments[0, :, None] != 0)
    if "window" in span:
        keep = keep & (rows[:, None] - rows[None, :] < span["window"])

    def temporaries(fn):
        compiled = jax.jit(fn).lower(query, key, value).compile()
        return compiled.memory_analysis().temp_size_in_bytes

    dense = temporaries(lambda q, k, v: scaled_dot_product_attention(
        q, k, v, mask=keep[None, None], implementation="xla"))
    local = temporaries(lambda q, k, v: local_attention(
        q, k, v, **span, segment_ids=segments if packing else None,
        positions=positions if packing and "chunk" in span else None, implementation="xla"))
    assert local <= dense * 1.25


def test_a_chunked_kind_decodes_what_its_whole_sequence_pass_computes():
    """The whole-sequence pass runs `local_attention`; the cached decode runs
    the chunk as a mask over cache slots. Both read the same keys."""
    model = CausalTransformer(
        vocab_size=64, num_layers=2, emb_features=32, num_heads=4, num_kv_heads=2,
        max_seq_len=24, layer_types=("chunked_attention", "full_attention"),
        kinds={"chunked_attention": {"chunk": 5}}, attention_impl="xla")
    tokens = jax.random.randint(jax.random.key(1), (2, 13), 0, 64)
    variables = model.init(jax.random.key(0), tokens)
    whole = model.apply(variables, tokens)
    cache = model.apply(variables, 2, method=CausalTransformer.init_cache, mutable=["cache"])[1]["cache"]
    logits, mutated = model.apply({**variables, "cache": cache}, tokens[:, :4],
                                  decode=True, mutable=["cache"])
    steps = [logits[:, -1]]
    for index in range(4, 13):
        logits, mutated = model.apply({**variables, "cache": mutated["cache"]},
                                      tokens[:, index:index + 1], decode=True, mutable=["cache"])
        steps.append(logits[:, -1])
    np.testing.assert_allclose(np.stack(steps, 1), whole[:, 3:], atol=1e-5, rtol=0)


def test_a_kind_reads_a_window_or_a_chunk_not_both():
    model = CausalTransformer(
        vocab_size=16, num_layers=1, emb_features=16, num_heads=2, max_seq_len=8,
        layer_types=("local",), kinds={"local": {"chunk": 4, "window": 4}})
    with pytest.raises(ValueError, match="window and a chunk"):
        model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))
