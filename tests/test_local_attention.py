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
    keys (segment 0 and invalid) are never read by a real query, one of a
    document and valid; a padding query's output is no token's."""
    length = 29
    query, key, value = inputs(length)
    segments, positions = packed(length, (7, 18, 26))
    valid = np.ones((2, length), bool)
    valid[1, 3] = False
    rows = np.arange(length)
    keep = (
        (rows[None, :] <= rows[:, None]) & (segments[:, None] == segments[None, :]) & (segments[:, None] != 0)
    )
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
    real = (segments != 0) & valid & keep.any(-1)
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


@pytest.mark.parametrize("shared", [False, True], ids=["per-row", "shared"])
@pytest.mark.parametrize("origin", [0, 2], ids=["from-zero", "logical-origin"])
def test_a_chunked_kind_decodes_what_its_whole_sequence_pass_computes(origin, shared):
    """The whole-sequence pass runs `local_attention`; the cached decode runs
    the chunk as a mask over cache slots. Both read the same keys: chunks
    are cut at the tokens' logical places, so a row whose places start at
    `origin` (a continuation, a canvas) cuts them as its whole pass does,
    not at its cache slots, whether each row gives its places or the rows
    share one vector of them."""
    model = CausalTransformer(
        vocab_size=64, num_layers=2, emb_features=32, num_heads=4, num_kv_heads=2,
        max_seq_len=24, layer_types=("chunked_attention", "full_attention"),
        kinds={"chunked_attention": {"chunk": 5}}, attention_impl="xla")
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

    assert "__cudnn$fmha" in loss("auto").lower(query, key, value).compile().as_text()

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
