"""Which kernel the 'auto' rule reaches for, and the XLA flags a run passes
to the backend.

'auto' is the attention selector's (`dew.nn.attention_kernels.ATTENTION`): each
kernel's requirements (its `SUPPORTS`) over the call's case, the target's row
first. The requirements read the target, so selections are testable anywhere
within `generation.assumed` by tracing the call without running it, and on a
GPU each requirement is held to the real kernel: a call it declares taken runs,
a call it refuses raises. What actually executes on a CUDA device runs with
`JAX_PLATFORMS=cuda python -m pytest tests/test_kernels.py`. The splash
kernel's own numbers are tests/test_attention_splash.py.
"""

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import attention_and_grads, attention_terms, exact_attention, kept, qkv, value_and_grads
from flax import linen as nn
from reference_error import assert_as_exact, assert_as_exact_as_the_reference, distance, with_gradients

from dew.nn import kernels
from dew.nn.attention import NormalAttention, scaled_dot_product_attention, weighted_values
from dew.nn.kernels import bf16_dot_runs
from dew.nn.kernels.selection import KernelRefused
from dew.nn.kernels.templates import attention as template
from dew.nn.precision import FP32_SUMS, at_least_fp32

pytestmark = pytest.mark.kernel


def test_a_flash_wheel_holds_code_for_its_architectures_major_from_their_minor_up():
    """A CUDA binary for X.Y runs on X.Z from Z = Y up (`flash_built_for`): a wheel built for sm80 and
    sm90 serves an A10, an L4 and an RTX 4080 with sm80's code and an H100 its own, and no Blackwell,
    no part below its minor, and no device that is not a GPU; sm86's code serves an L4 and sm120's
    any later 12.x part."""
    from dew.nn.attention_kernels import flash_built_for

    built = ("sm80", "sm90")
    assert [flash_built_for(device, built) for device in ("sm80", "sm86", "sm89", "sm90")] == [True] * 4
    other = ("sm75", "sm100", "sm120", "cpu", "v6e")
    assert [flash_built_for(device, built) for device in other] == [False] * len(other)
    assert flash_built_for("sm89", ("sm86",)) and flash_built_for("sm121", ("sm120",))
    assert not flash_built_for("sm86", ("sm89",))

def on_gpu(test):
    """Mark `test` for a cuda device of sm80 or later, cuDNN's bf16 floor."""
    reason = "needs a cuda device of sm80 or later, cuDNN's bf16 floor"
    return pytest.mark.gpu(reason=reason)(pytest.mark.skipif(not bf16_dot_runs(), reason=reason)(test))


def flash_here():
    """dew_flash_attn where the installed wheel holds code for this GPU (the fact `flash_refusal` reads);
    a GPU it was not built for skips by name, where 'auto' runs another kernel."""
    flash = pytest.importorskip("dew_flash_attn")
    from dew.nn.attention_kernels import flash_architectures, flash_built_for
    generation = kernels.device_generation()
    architectures = flash_architectures() or ()
    if not flash_built_for(generation, architectures):
        pytest.skip(f"the installed dew_flash_attn holds code for {', '.join(architectures)}, "
                    f"not {generation}")
    return flash


@on_gpu
def test_flash_through_dews_attention_refuses_rows_it_would_misread(without_deterministic_ops):
    """Through Dew's attention (`scaled_dot_product_attention`, 'flash'), keys and values of another
    batch, width, length or head count than the query's are refused as the call traces, before any
    kernel runs, dense and packed. Packing flattens the rows into one run of tokens, where keys of one
    row of 64 beside queries of two rows of 32, or another sequence's keys of one row of 32 beside
    their ids of two rows of 16, flatten to as many tokens and would be read as the next row's: the
    call names the ids. Each well-formed twin, dense with grouped heads and packed, gives its output
    and gradients within tests/reference_error.py's rule of the xla kernel's, both measured from
    float64 per row or document."""
    flash_here()
    rng = np.random.default_rng(31)

    def normal(*shape):
        return jnp.asarray(rng.normal(size=shape), jnp.bfloat16)

    query, cotangent = normal(2, 32, 4, 64), normal(2, 32, 4, 64)
    key, value = normal(2, 32, 2, 64), normal(2, 32, 2, 64)
    own = {"segment_ids": jnp.asarray(documents([[20, 12, 0], [9, 23, 0]])), "max_segments": 2}
    cross = {"segment_ids": own["segment_ids"], "max_segments": 2,
             "key_segment_ids": jnp.asarray(documents([[10, 6, 0], [4, 12, 0]]))}

    def attend(implementation, causal=True, **packed):
        return lambda q, k, v: scaled_dot_product_attention(q, k, v, implementation=implementation,
                                                            causal=causal, **packed)

    malformed = [(normal(1, 32, 2, 64),) * 2, (normal(2, 32, 2, 32),) * 2, (key, normal(2, 16, 2, 64)),
                 (normal(2, 32, 3, 64),) * 2]
    for packed in ({}, own):
        for k, v in malformed:
            with pytest.raises(ValueError):
                jax.eval_shape(attend('flash', **packed), query, k, v)
    # Calls the selector gives flash, so the refusal is the packed path's own, not `KernelRefused`: not
    # causal, since a causal call over its own documents is square.
    for packed, keys in ((own, 64), (cross, 32)):
        with pytest.raises(ValueError) as refused:
            jax.eval_shape(attend('flash', causal=False, **packed), query, normal(1, keys, 2, 64),
                           normal(1, keys, 2, 64))
        assert not isinstance(refused.value, KernelRefused)

    def exact(pieces):
        """`[o, dq, dk, dv]` from float64 and their `window_terms`, each `(row, slice)` causal alone."""
        truth = [np.zeros(x.shape, np.float64) for x in (query, query, key, value)]
        terms = [np.zeros(x.shape, np.float64) for x in (query, query, key, value)]
        with jax.enable_x64(new_val=True):
            for row, piece in pieces:
                parts = [x[row, piece] for x in (query, key, value, cotangent)]
                wide = [jnp.asarray(np.asarray(x, np.float64)) for x in parts]
                for whole, part in zip(truth, output_and_gradients(
                        lambda q, k, v: float64_attention(q, k, v, causal=True), wide[3], *wide[:3]),
                        strict=True):
                    whole[row, piece] = part
                for whole, part in zip(terms, window_terms(*parts, causal=True), strict=True):
                    whole[row, piece] = part
        return truth, terms

    rows = [(0, slice(0, 32)), (1, slice(0, 32))]
    documents_ = [(0, slice(0, 20)), (0, slice(20, 32)), (1, slice(0, 9)), (1, slice(9, 32))]
    for packed, pieces in (({}, rows), (own, documents_)):
        fused = output_and_gradients(jax.jit(attend('flash', **packed)), cotangent, query, key, value)
        reference = output_and_gradients(jax.jit(attend('xla', **packed)), cotangent, query, key, value)
        truth, terms = exact(pieces)
        for name, index in (("o", 0), ("dq", 1), ("dk", 2), ("dv", 3)):
            assert_as_exact_as_the_reference(fused[index], reference[index], truth[index],
                                             f"{name} {'packed' if packed else 'dense'}", terms=terms[index])


SHARDED_SEEDS = 8
"""The per-shard contract's declared draws (check 1, `sharded.assert_runs_per_shard`), sized by
`seeds_for`: the fewest for which one device compared with itself resolves the factor."""


@on_gpu
@pytest.mark.parametrize("implementation", ["cudnn", "flash"])
def test_a_fused_kernel_takes_forward_mode_through_the_reference(implementation, without_deterministic_ops):
    """A fused kernel defines only its reverse-mode derivative, so a JVP
    traced under `forward_mode_attention` takes its tangent from the
    reference (`forward_differentiable`): the tangent agrees with the xla
    kernel's within two bf16 ulps of its scale. Outside the context, jax
    refuses the JVP of the kernel's custom_vjp, which this also holds."""
    from dew.nn.attention import forward_mode_attention
    if implementation == "flash":
        flash_here()
    query, key, value = qkv((2, 256, 8, 64))
    tangents = qkv((2, 256, 8, 64), seed=1)

    def attend(name):
        return lambda q, k, v: scaled_dot_product_attention(q, k, v, implementation=name, causal=True)

    def tangent(name):
        with forward_mode_attention():
            return np.asarray(jax.jit(lambda *a: jax.jvp(attend(name), a[:3], a[3:])[1])(
                query, key, value, *tangents), np.float32)

    with pytest.raises(TypeError):
        jax.jvp(attend(implementation), (query, key, value), tangents)

    got, want = tangent(implementation), tangent("xla")
    assert np.abs(got - want).max() <= 2 ** -6 * np.abs(want).max()


@on_gpu
@pytest.mark.parametrize("q_len, kv_len, causal, head_dim", [
    (1024, 77, False, 64),   # every cross-attention over CLIP's 77 text tokens
    (9, 7, False, 64),       # short enough that one attended pad key would move an eighth of the mass
    (333, 333, True, 64),    # the concatenated text-plus-image sequence, causal
    (3072, 3072, True, 256),  # Qwen3.5's 256-wide heads, which cudnn takes from Hopper on
])
def test_cudnn_trains_odd_lengths_and_agrees_with_xla(q_len, kv_len, causal, head_dim,
                                                      without_deterministic_ops):
    """cudnn's kernel has no backward pass for an odd length; the padded call
    trains, and what it computes for the real rows is what the xla kernel
    computes: within two bf16 ulps of the output scale, forward and backward,
    which is also how far the two kernels sit apart at an even length. A pad
    key left unmasked shifts every output by 1/(kv+1) of the value mass, 12%
    at 7 keys, and a pad query row left in the output changes its shape."""
    from dew.nn.attention_kernels import cudnn_max_head_dim
    if head_dim > cudnn_max_head_dim():
        pytest.skip(f"cudnn takes heads at most {cudnn_max_head_dim()} wide on {kernels.device_generation()}")
    query, _, _ = qkv((2, q_len, 4, head_dim))
    _, key, value = qkv((2, kv_len, 4, head_dim), seed=1)

    fused = value_and_grads('cudnn', query, key, value, causal=causal)
    reference = value_and_grads('xla', query, key, value, causal=causal)

    for got, want in zip(fused, reference, strict=True):
        assert got.shape == want.shape
        assert np.abs(got - want).max() <= 2 ** -6 * np.abs(want).max()


@on_gpu
@pytest.mark.parametrize("q_len, kv_len, heads, kv_heads, head_dim, causal", [
    (256, 256, 12, 12, 64, False),   # SimpleDiT-B's image tokens
    (512, 512, 12, 12, 64, True),    # the small decoder
    (1024, 1024, 16, 8, 128, True),  # Qwen3-0.6B's grouped heads
    (333, 333, 4, 4, 64, True),      # an odd length, which cudnn pads
    (1024, 77, 4, 4, 64, False),     # cross-attention over CLIP's 77 tokens
    (256, 256, 8, 8, 80, True),      # a head width that is no power of two
])
def test_triton_trains_and_agrees_with_xla(q_len, kv_len, heads, kv_heads, head_dim, causal):
    """tokamax's kernel at its heuristic config, the one 'auto' runs, at the
    shapes it is sent: within two bf16 ulps of the output scale of the xla
    kernel, forward and backward, as cudnn is. One of its VJP configs returns
    gradients off by orders of magnitude (64/32/32/64 blocks at 1024 causal
    tokens), so each shape's gradients are checked, not its forward alone."""
    pytest.importorskip("tokamax")
    query, _, _ = qkv((2, q_len, heads, head_dim))
    _, key, value = qkv((2, kv_len, kv_heads, head_dim), seed=1)

    fused = value_and_grads('triton', query, key, value, causal=causal)
    reference = value_and_grads('xla', query, key, value, causal=causal)

    for got, want in zip(fused, reference, strict=True):
        assert got.shape == want.shape
        assert np.abs(got - want).max() <= 2 ** -6 * np.abs(want).max()


LARGE_LOGIT_SEEDS = 16
"""The seeds of the triton route's large-logit site, sized by `seeds_for` (paired) on an A100."""


def _large_logits(spread: bool):
    """Draws of bf16 queries, keys and values `[2, 64, 2, 32]` at logits near 1e4, as a small DiT's
    trained projections give them: one channel of each query drawn in [200, 300], the same channel of the
    keys all 256 where the softmax spreads over every key (`spread`), else drawn in [200, 300] too, which
    saturates most rows on one key; the other channels standard normal."""
    shape = (2, 64, 2, 32)

    def draw(seed):
        streams = jax.random.split(jax.random.PRNGKey(seed), 5)
        query, key, value = (jax.random.normal(stream, shape, jnp.float32) for stream in streams[:3])
        query = query.at[..., 0].set(jax.random.uniform(streams[3], shape[:3], minval=200, maxval=300))
        channel = (jnp.full(shape[:3], 256.0) if spread
                   else jax.random.uniform(streams[4], shape[:3], minval=200, maxval=300))
        key = key.at[..., 0].set(channel)
        return tuple(x.astype(jnp.bfloat16) for x in (query, key, value))

    return draw


def _pulled(implementation):
    """`implementation`'s output and its query, key and value gradients under a squared-output loss."""
    step = attention_and_grads(implementation)

    def run(query, key, value):
        (_, out), grads = step(query, key, value, None)
        return (out, *grads)

    return run


@on_gpu
def test_triton_trains_at_large_logits_on_par_with_xla():
    """tokamax's kernel, as 'triton' runs it, at logits near 1e4 that its softmax spreads over every key
    (`_large_logits`): its output and its query, key and value gradients on par with the xla kernel's from
    float64 (check 1). Its dK/dV sweep scaled the keys where its forward and dQ scale the queries, so its
    scores there rounded apart from the forward's residuals by units, and the weights it rebuilt from them
    were off by their exponential (row 1676: a MeanFlow step's key and value gradients 1e35 to 1e62 times
    xla's error); the route hands it the queries scaled once (`triton_attention`)."""
    pytest.importorskip("tokamax")
    assert_as_exact(_pulled("triton"), _pulled("xla"), _large_logits(spread=True), "triton at large logits",
                    seeds=LARGE_LOGIT_SEEDS)


@on_gpu
def test_triton_at_saturated_large_logits_is_finite_where_xla_is():
    """tokamax's kernel at logits near 1e4 whose rows mostly saturate on one key (`_large_logits`): the
    route is the default form, FlashAttention's math, whose D (rowsum(dO * O), the output rounded to bf16)
    cancels against the saturated key's product where the reference's softmax derivative does not, so it
    holds testing.md's default-form rule there, not check 1: every output and gradient finite wherever
    xla's is (row 1676; measured 3.6 times xla's dQ error and rows of float64 zero it misses, on an A100 and
    an H100). The Mosaic template, which replaces this route, takes D about each query's anchor."""
    pytest.importorskip("tokamax")
    draw, ours, xla = _large_logits(spread=False), jax.jit(_pulled("triton")), jax.jit(_pulled("xla"))
    for seed in range(LARGE_LOGIT_SEEDS):
        inputs = draw(seed)
        for name, mine, theirs in zip(("out", "dq", "dk", "dv"), ours(*inputs), xla(*inputs), strict=True):
            mine, theirs = np.asarray(mine, np.float32), np.asarray(theirs, np.float32)
            lost = np.isfinite(theirs) & ~np.isfinite(mine)
            assert not lost.any(), f"{name} seed {seed}: {int(lost.sum())} values not finite where xla's are"


@on_gpu
@pytest.mark.parametrize("q_len, kv_len, heads, kv_heads, head_dim, causal", [
    (1024, 1024, 16, 8, 128, True),  # Qwen3-0.6B's grouped heads
    (512, 512, 8, 8, 256, True),     # 256-wide heads, which cudnn takes on Hopper alone
    (256, 256, 12, 12, 64, False),   # SimpleDiT-B's image tokens
    (333, 333, 4, 4, 64, True),      # an odd length
    (1024, 77, 4, 4, 64, False),     # cross-attention over CLIP's 77 tokens
    (256, 256, 8, 8, 80, True),      # a head width that is no power of two
])
def test_flash_trains_and_agrees_with_xla(q_len, kv_len, heads, kv_heads, head_dim, causal,
                                          without_deterministic_ops):
    """FlashAttention at the shapes 'auto' sends it, FlashAttention-3 on
    Hopper and FlashAttention-2 elsewhere: within two bf16 ulps of the output
    scale of the xla kernel, forward and backward, as cudnn and triton are.
    FlashAttention-2's backward takes heads over 192 wide only on sm80 and
    sm90 ("requires A100/A800 or H100/H800", measured on an L4)."""
    flash_here()
    if head_dim > 192 and kernels.device_generation() in ('sm86', 'sm87', 'sm89'):
        pytest.skip("FlashAttention-2's backward takes heads over 192 wide only on sm80 and sm90")
    query, _, _ = qkv((2, q_len, heads, head_dim))
    _, key, value = qkv((2, kv_len, kv_heads, head_dim), seed=1)

    fused = value_and_grads('flash', query, key, value, causal=causal)
    reference = value_and_grads('xla', query, key, value, causal=causal)

    for got, want in zip(fused, reference, strict=True):
        assert got.shape == want.shape
        assert np.abs(got - want).max() <= 2 ** -6 * np.abs(want).max()


def float64_attention(query, key, value, causal, window=(-1, -1), sinks=None):
    """`exact_attention` of one row's `[S, H, D]` arrays: a query keeps the keys `window` `(left,
    right)` reaches (-1 bounding neither side) behind causality, and a head's sink logit joins each
    row's denominator with no value."""
    keep = kept(query.shape[0], key.shape[0], causal=causal, window=window)
    return exact_attention(query[None], key[None], value[None], sinks, keep=keep)[0]


def window_terms(query, key, value, cotangent, causal, window=(-1, -1), sinks=None):
    """`attention_terms` of `float64_attention`'s output and gradients for `[S, H, D]` arrays."""
    terms = attention_terms(query[None], key[None], value[None], cotangent[None],
                            kept(query.shape[0], key.shape[0], causal=causal, window=window), sinks)
    return [x[0] for x in terms]


def output_and_gradients(attend, cotangent, *arrays):
    """`attend`'s output and the gradients of its inputs under `cotangent`, as float64 host arrays."""
    out, pull = jax.vjp(attend, *arrays)
    return [np.asarray(x, np.float64) for x in (out, *pull(cotangent))]


@on_gpu
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("sunk", [False, True])
def test_packed_sequences_attend_within_themselves_as_far_from_float64_as_one_at_a_time(
        head_dim, causal, sunk, without_deterministic_ops):
    """Documents packed back to back, one a single token, with an empty sequence between two: each
    document's output and its three input gradients lie within tests/reference_error.py's rule of
    FlashAttention's own on that document alone, both measured from float64, and with a 64-key
    window and sinks so does the sinks' gradient summed over the documents. Not bitwise: on one
    short document the dense forward splits its keys across blocks (`_splits`) and the packed one
    does not, so the two round differently (on an RTX 4080, 1.0003 to 1.012 times apart in RMS)."""
    flash = flash_here()
    if head_dim > 192 and kernels.device_generation() in ('sm86', 'sm87', 'sm89'):
        pytest.skip("FlashAttention-2's backward takes heads over 192 wide only on sm80, sm90 and sm1xx")
    lengths = [700, 1, 0, 333, 64]
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    query, _, _ = qkv((int(starts[-1]), 8, head_dim))
    _, key, value = qkv((int(starts[-1]), 2, head_dim), seed=1)
    cotangent = qkv((int(starts[-1]), 8, head_dim), seed=2)[0]
    logits = jnp.asarray(np.random.default_rng(3).normal(size=(8,)) * 2, jnp.float32)
    window = (63, 0 if causal else 63) if sunk else (-1, -1)

    def options(sinks):
        return {"is_causal": causal, "window_size": window, "sinks": sinks if sunk else None}

    packed = output_and_gradients(
        lambda q, k, v, s: flash.flash_mha_varlen(q, k, v, starts, max(lengths), **options(s)),
        cotangent, query, key, value, logits)
    dense_sinks, true_sinks = [], []
    for begin, end in itertools.pairwise(starts):
        if begin == end:
            continue
        pieces = (query[begin:end], key[begin:end], value[begin:end])
        dense = output_and_gradients(
            lambda q, k, v, s: flash.flash_mha(q[None], k[None], v[None], **options(s))[0],
            cotangent[begin:end], *pieces, logits)
        with jax.enable_x64(new_val=True):
            wide = [jnp.asarray(np.asarray(x, np.float64)) for x in (*pieces, cotangent[begin:end], logits)]
            truth = output_and_gradients(
                lambda q, k, v, s: float64_attention(q, k, v, causal, window, s if sunk else None),
                wide[3], *wide[:3], wide[4])
        dense_sinks.append(dense[4])
        true_sinks.append(truth[4])
        terms = window_terms(*pieces, cotangent[begin:end], causal, window, logits if sunk else None)
        documents = (x[begin:end] for x in packed[:4])
        for name, got, theirs, want, summed in zip(("o", "dq", "dk", "dv"), documents, dense[:4], truth[:4],
                                                   terms, strict=True):
            if distance(theirs, want) == 0:  # a one-token document's attention is exact either way
                np.testing.assert_array_equal(got, theirs)
            else:
                assert_as_exact_as_the_reference(got, theirs, want, f"{name} of the document at {begin}",
                                                 terms=summed)
    if sunk:
        assert_as_exact_as_the_reference(packed[4], np.sum(dense_sinks, axis=0), np.sum(true_sinks, axis=0),
                                         "dsinks")


@on_gpu
@pytest.mark.parametrize("head_dim", [64, 128])
def test_queries_packed_against_keys_of_their_own_lengths_attend_their_own_documents(head_dim,
                                                                                     without_deterministic_ops):
    """Cross-attention between packed documents: sequence i's queries, packed by their own offsets,
    attend sequence i's keys, packed by theirs, as a decision head's option slots read their own
    row's tokens. Each document's output and its query, key and value gradients lie within
    tests/reference_error.py's rule of FlashAttention's own on that document alone, both measured
    from float64, a one-query document and an empty one among them; a document's keys no query
    reads take no gradient."""
    flash = flash_here()
    queries, keys = [30, 1, 0, 45, 7], [700, 333, 50, 120, 64]
    starts, key_starts = (np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
                          for lengths in (queries, keys))
    query = qkv((int(starts[-1]), 8, head_dim))[0]
    _, key, value = qkv((int(key_starts[-1]), 2, head_dim), seed=1)
    cotangent = qkv((int(starts[-1]), 8, head_dim), seed=2)[0]
    packed = output_and_gradients(
        lambda q, k, v: flash.flash_mha_varlen(q, k, v, starts, max(queries), key_cu_seqlens=key_starts,
                                               key_max_seqlen=max(keys)),
        cotangent, query, key, value)
    for (begin, end), (key_begin, key_end) in zip(itertools.pairwise(starts), itertools.pairwise(key_starts),
                                                  strict=True):
        if begin == end:
            np.testing.assert_array_equal(packed[2][key_begin:key_end], 0)
            np.testing.assert_array_equal(packed[3][key_begin:key_end], 0)
            continue
        pieces = (query[begin:end], key[key_begin:key_end], value[key_begin:key_end])
        dense = output_and_gradients(lambda q, k, v: flash.flash_mha(q[None], k[None], v[None])[0],
                                     cotangent[begin:end], *pieces)
        with jax.enable_x64(new_val=True):
            wide = [jnp.asarray(np.asarray(x, np.float64)) for x in (*pieces, cotangent[begin:end])]
            truth = output_and_gradients(lambda q, k, v: float64_attention(q, k, v, causal=False), wide[3],
                                         *wide[:3])
        rows = (slice(begin, end), slice(begin, end), slice(key_begin, key_end), slice(key_begin, key_end))
        for name, got, at, theirs, want in zip(("o", "dq", "dk", "dv"), packed, rows, dense, truth,
                                               strict=True):
            assert_as_exact_as_the_reference(got[at], theirs, want, f"{name} of the document at {begin}")


@on_gpu
@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axis, split", [("data", (0,)), ("tensor", (2,))])
def test_flash_keeps_the_kernel_contract_on_a_mesh(axis, split, without_deterministic_ops):
    """FlashAttention-2 over rows split two ways, or query heads split two ways with the key heads
    they read, runs per shard (`sharded.assert_runs_per_shard`): the output and the three input
    gradients on par with one device's over the site's draws, no operand moved between devices and
    no copy or transpose at the kernel's edge."""
    from jax.sharding import Mesh, PartitionSpec
    from sharded import assert_runs_per_shard

    flash_here()

    def draw(seed):
        query, _, _ = qkv((2, 256, 4, 64), seed=3 * seed)
        _, key, value = qkv((2, 256, 2, 64), seed=3 * seed + 1)
        return query, key, value, qkv((2, 256, 4, 64), seed=3 * seed + 2)[0]

    spec = PartitionSpec(*(axis if dim in split else None for dim in range(3)))
    mesh = Mesh(np.asarray(jax.devices()[:2]), (axis,))
    flash = functools.partial(scaled_dot_product_attention, implementation='flash', causal=True)
    assert_runs_per_shard(with_gradients(flash), draw, [spec] * 4, mesh, f"flash split over {axis}",
                          seeds=SHARDED_SEEDS, truth_fn=with_gradients(
                              jax.vmap(functools.partial(float64_attention, causal=True))))


@on_gpu
@pytest.mark.parametrize("implementation", ["flash", "cudnn"])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_fused_kernels_are_as_exact_as_the_xla_kernel_from_float64(implementation, head_dim,
                                                                   without_deterministic_ops):
    """At Qwen3.5's grouped causal heads, 8 queries over 2 keys, FlashAttention's and cuDNN's output
    and three input gradients lie within tests/reference_error.py's rule of the xla kernel's, both
    measured from float64; on Hopper that holds FlashAttention-3 and cuDNN's 256-wide heads to it."""
    from dew.nn.attention_kernels import cudnn_max_head_dim
    if implementation == 'flash':
        flash_here()
        if head_dim > 192 and kernels.device_generation() in ('sm86', 'sm87', 'sm89'):
            pytest.skip("FlashAttention-2's backward takes heads over 192 wide only on sm80, sm90 and sm1xx")
    elif head_dim > cudnn_max_head_dim():
        pytest.skip(f"cudnn takes heads at most {cudnn_max_head_dim()} wide on {kernels.device_generation()}")
    query, _, _ = qkv((1, 1024, 8, head_dim))
    _, key, value = qkv((1, 1024, 2, head_dim), seed=1)
    cotangent = qkv((1, 1024, 8, head_dim), seed=2)[0]

    def kernel(implementation):
        return jax.jit(lambda q, k, v: scaled_dot_product_attention(
            q, k, v, implementation=implementation, causal=True))

    fused = output_and_gradients(kernel(implementation), cotangent, query, key, value)
    reference = output_and_gradients(kernel('xla'), cotangent, query, key, value)
    with jax.enable_x64(new_val=True):
        wide = [jnp.asarray(np.asarray(x[0], np.float64)) for x in (query, key, value, cotangent)]
        truth = output_and_gradients(functools.partial(float64_attention, causal=True), wide[3],
                                     *wide[:3])
    terms = window_terms(query[0], key[0], value[0], cotangent[0], causal=True)
    for name, got, theirs, want, summed in zip(("o", "dq", "dk", "dv"), fused, reference, truth, terms,
                                               strict=True):
        assert_as_exact_as_the_reference(got[0], theirs[0], want, name, terms=summed)


@on_gpu
@pytest.mark.parametrize("width, value_width, kv_heads, causal, window, sinks", [
    (64, 64, 2, True, None, True),        # gpt-oss-20b's full layers
    (64, 64, 2, True, 128, True),         # and its sliding ones
    (192, 128, 2, True, 128, True),       # MiMo-V2-Flash's sliding layers
    (64, 64, 8, False, 65, False),        # ModernBERT's local layers, both sides
])
def test_flash_with_a_window_and_sinks_is_as_exact_as_the_xla_kernel_from_float64(
        width, value_width, kv_heads, causal, window, sinks, without_deterministic_ops):
    """FlashAttention with a sliding window, attention sinks or a value narrower than the query, as
    'auto' sends such calls on sm80, sm89 and sm90: its output and the gradients of the query, key,
    value and sinks lie within tests/reference_error.py's rule of the xla kernel's, both measured
    from float64."""
    flash_here()
    rng = np.random.default_rng(57)
    query = jnp.asarray(rng.normal(size=(1, 1024, 8, width)), jnp.bfloat16)
    key = jnp.asarray(rng.normal(size=(1, 1024, kv_heads, width)), jnp.bfloat16)
    value = jnp.asarray(rng.normal(size=(1, 1024, kv_heads, value_width)), jnp.bfloat16)
    cotangent = jnp.asarray(rng.normal(size=(1, 1024, 8, value_width)), jnp.bfloat16)
    logits = jnp.asarray(rng.normal(size=(8,)) * 2, jnp.float32)

    def kernel(implementation):
        return jax.jit(lambda q, k, v, s: scaled_dot_product_attention(
            q, k, v, implementation=implementation, causal=causal, sliding_window=window,
            sinks=s if sinks else None))

    fused = output_and_gradients(kernel('flash'), cotangent, query, key, value, logits)
    reference = output_and_gradients(kernel('xla'), cotangent, query, key, value, logits)
    sides = (-1, -1) if window is None else (window - 1, 0 if causal else window - 1)
    with jax.enable_x64(new_val=True):
        wide = [jnp.asarray(np.asarray(x[0], np.float64)) for x in (query, key, value, cotangent)]
        wide_sinks = jnp.asarray(np.asarray(logits, np.float64))
        truth = output_and_gradients(
            lambda q, k, v, s: float64_attention(q, k, v, causal, sides, s if sinks else None),
            wide[3], *wide[:3], wide_sinks)
    terms = window_terms(query[0], key[0], value[0], cotangent[0], causal, sides, logits if sinks else None)
    for name, got, theirs, want, summed in zip(("o", "dq", "dk", "dv"), fused[:4], reference[:4], truth[:4],
                                               terms, strict=True):
        assert_as_exact_as_the_reference(got[0], theirs[0], want, name, terms=summed)
    if sinks:
        assert_as_exact_as_the_reference(fused[4], reference[4], truth[4], "dsinks")


@on_gpu
def test_a_window_over_more_keys_than_queries_keeps_the_keys_each_row_centres(without_deterministic_ops):
    """A bidirectional window over more keys than queries (256 queries over 512 keys, 32 each side)
    keeps the keys `structural_mask` centres on each query's row, Dew's contract, whichever kernel
    'auto' picks on this GPU: its output and gradients within tests/reference_error.py's rule of the
    xla kernel's, both from float64. FlashAttention aligns a window to the last key, the keys 256
    further on, so 'auto' leaves it such a call and an explicit 'flash' refuses it, while it takes
    the same window square."""
    flash_here()
    rng = np.random.default_rng(61)
    query = jnp.asarray(rng.normal(size=(1, 256, 8, 64)), jnp.bfloat16)
    key, value = (jnp.asarray(rng.normal(size=(1, 512, 8, 64)), jnp.bfloat16) for _ in range(2))
    cotangent = jnp.asarray(rng.normal(size=query.shape), jnp.bfloat16)

    def kernel(implementation):
        return jax.jit(lambda q, k, v: scaled_dot_product_attention(
            q, k, v, implementation=implementation, sliding_window=33))

    found = output_and_gradients(kernel('auto'), cotangent, query, key, value)
    reference = output_and_gradients(kernel('xla'), cotangent, query, key, value)
    with jax.enable_x64(new_val=True):
        truth = output_and_gradients(kernel('reference'), jnp.asarray(cotangent, jnp.float64),
                                     *(jnp.asarray(x, jnp.float64) for x in (query, key, value)))
    for name, got, theirs, want in zip(("o", "dq", "dk", "dv"), found, reference, truth, strict=True):
        assert_as_exact_as_the_reference(got[0], theirs[0], want[0], name)
    with pytest.raises(ValueError):
        kernel('flash')(query, key, value)
    kernel('flash')(query, key[:, :256], value[:, :256]).block_until_ready()


def documents(lengths) -> np.ndarray:
    """Packed rows' segment ids, each row's documents of `lengths[:-1]` tokens numbered from 1 and its
    padding of `lengths[-1]` after them."""
    return np.stack([np.concatenate([np.repeat(np.arange(1, len(row)), row[:-1]), np.zeros(row[-1], int)])
                     for row in lengths]).astype(np.int32)


@on_gpu
@pytest.mark.parametrize("window, sinks", [(None, False), (128, True)])
def test_packed_rows_on_flash_are_as_exact_as_the_document_mask_from_float64(window, sinks,
                                                                             without_deterministic_ops):
    """Rows packing documents of uneven lengths, padding after them, through flash as its varlen
    sequences (`segment_ids` with their declared `max_segments`), causal, also with a 128-key window
    and sinks: every real token's output and query gradient, and every key and value gradient, lie
    within tests/reference_error.py's rule of the xla kernel's under the document mask, both measured
    from float64 per document. The padding attends among itself on flash and to nothing under the
    mask, and its rows carry no gradient."""
    flash_here()
    rng = np.random.default_rng(23)
    lengths = [[700, 1, 220, 103], [129, 383, 400, 112]]
    segments = documents(lengths)
    real = segments != 0
    query = jnp.asarray(rng.normal(size=(2, 1024, 8, 64)), jnp.bfloat16)
    key = jnp.asarray(rng.normal(size=(2, 1024, 2, 64)), jnp.bfloat16)
    value = jnp.asarray(rng.normal(size=(2, 1024, 2, 64)), jnp.bfloat16)
    cotangent = jnp.asarray(rng.normal(size=(2, 1024, 8, 64)) * real[:, :, None, None], jnp.bfloat16)
    logits = jnp.asarray(rng.normal(size=(8,)) * 2, jnp.float32)

    def kernel(implementation):
        return jax.jit(lambda q, k, v, s: scaled_dot_product_attention(
            q, k, v, implementation=implementation, causal=True, sliding_window=window,
            sinks=s if sinks else None, segment_ids=jnp.asarray(segments), max_segments=3))

    fused = output_and_gradients(kernel('flash'), cotangent, query, key, value, logits)
    reference = output_and_gradients(kernel('xla'), cotangent, query, key, value, logits)
    sides = (-1, -1) if window is None else (window - 1, 0)
    truth = [np.zeros_like(x) for x in fused]
    terms = [np.zeros_like(x) for x in fused[:4]]
    with jax.enable_x64(new_val=True):
        for row in range(2):
            begin = 0
            for length in lengths[row][:-1]:
                piece = slice(begin, begin + length)
                wide = [jnp.asarray(np.asarray(x[row, piece], np.float64))
                        for x in (query, key, value, cotangent)]
                exact = output_and_gradients(
                    lambda q, k, v, s: float64_attention(q, k, v, causal=True, window=sides,
                                                         sinks=s if sinks else None),
                    wide[3], *wide[:3], jnp.asarray(np.asarray(logits, np.float64)))
                for whole, part in zip(truth[:4], exact[:4], strict=True):
                    whole[row, piece] = part
                for whole, part in zip(terms, window_terms(
                        *(x[row, piece] for x in (query, key, value, cotangent)), causal=True, window=sides,
                        sinks=logits if sinks else None), strict=True):
                    whole[row, piece] = part
                truth[4] += exact[4]
                begin += length
    for name, index in (("o", 0), ("dq", 1)):
        assert_as_exact_as_the_reference(fused[index][real], reference[index][real], truth[index][real], name,
                                         terms=terms[index][real])
    for name, index in (("dk", 2), ("dv", 3)):
        assert_as_exact_as_the_reference(fused[index], reference[index], truth[index], name,
                                         terms=terms[index])
    if sinks:
        assert_as_exact_as_the_reference(fused[4], reference[4], truth[4], "dsinks")


@on_gpu
def test_queries_reading_another_sequences_documents_on_flash_are_as_exact_as_the_document_mask(
        without_deterministic_ops):
    """A decision head's option slots read the window's tokens of their own row: queries packed by
    their documents (`segment_ids`), keys by theirs (`key_segment_ids`), through flash as packed
    sequences of their own lengths. Every real query's output and gradient, and every key and value
    gradient, lie within tests/reference_error.py's rule of the xla kernel's under the document
    mask, both measured from float64 per document; a document no query reads takes no gradient, and
    a padding query, or one whose document holds no key, reads nothing."""
    flash_here()
    rng = np.random.default_rng(29)
    queries = documents([[20, 1, 40, 35], [9, 0, 30, 57]])
    # Row 0's second document has a query and no key.
    keys = documents([[300, 0, 500, 224], [129, 383, 400, 112]])
    real = queries != 0
    query = jnp.asarray(rng.normal(size=(2, 96, 8, 64)), jnp.bfloat16)
    key = jnp.asarray(rng.normal(size=(2, 1024, 8, 64)), jnp.bfloat16)
    value = jnp.asarray(rng.normal(size=(2, 1024, 8, 64)), jnp.bfloat16)
    cotangent = jnp.asarray(rng.normal(size=(2, 96, 8, 64)) * real[:, :, None, None], jnp.bfloat16)

    def kernel(implementation):
        return jax.jit(lambda q, k, v: scaled_dot_product_attention(
            q, k, v, implementation=implementation, segment_ids=jnp.asarray(queries),
            key_segment_ids=jnp.asarray(keys), max_segments=3))

    fused = output_and_gradients(kernel('flash'), cotangent, query, key, value)
    reference = output_and_gradients(kernel('xla'), cotangent, query, key, value)
    np.testing.assert_array_equal(fused[0][~real], 0)
    np.testing.assert_array_equal(fused[0][0][queries[0] == 2], 0)
    truth = [np.zeros_like(x, np.float64) for x in fused]
    with jax.enable_x64(new_val=True):
        for row in range(2):
            for document in range(1, 4):
                asking, read = queries[row] == document, keys[row] == document
                if not asking.any() or not read.any():
                    continue
                pieces = (query[row][asking], key[row][read], value[row][read], cotangent[row][asking])
                wide = [jnp.asarray(np.asarray(x, np.float64)) for x in pieces]
                exact = output_and_gradients(lambda q, k, v: float64_attention(q, k, v, causal=False),
                                             wide[3], *wide[:3])
                for whole, part, at in zip(truth, exact, (asking, asking, read, read), strict=True):
                    whole[row][at] = part
    for name, index in (("o", 0), ("dq", 1)):
        assert_as_exact_as_the_reference(fused[index][real], reference[index][real], truth[index][real], name)
    for name, index in (("dk", 2), ("dv", 3)):
        assert_as_exact_as_the_reference(fused[index], reference[index], truth[index], name)


@on_gpu
def test_queries_over_a_context_of_no_tokens_read_nothing_on_flash(without_deterministic_ops):
    """Queries packed by documents over a context of no tokens at all: every query reads nothing,
    its output and gradient exactly zero on flash as on the reference kernel, and the empty keys
    and values take empty gradients. Called outside jit, so the kernel launches over its empty keys,
    whose length FA3 needs bounded by at least one."""
    flash_here()
    rng = np.random.default_rng(31)
    queries = jnp.asarray(documents([[3, 2, 1]]))
    query = jnp.asarray(rng.normal(size=(1, 6, 8, 64)), jnp.bfloat16)
    key = value = jnp.zeros((1, 0, 8, 64), jnp.bfloat16)
    for implementation in ("flash", "reference"):
        out, pull = jax.vjp(lambda q, k, v, implementation=implementation: scaled_dot_product_attention(
            q, k, v, implementation=implementation, segment_ids=queries,
            key_segment_ids=jnp.zeros((1, 0), jnp.int32), max_segments=2), query, key, value)
        dq, dk, dv = pull(jnp.ones_like(out))
        np.testing.assert_array_equal(np.asarray(out, np.float32), 0)
        np.testing.assert_array_equal(np.asarray(dq, np.float32), 0)
        assert dk.shape == dv.shape == key.shape


@pytest.mark.parametrize("implementation", ["reference", "xla"])
def test_a_query_reading_another_sequences_documents_reads_its_own_documents_keys_alone(implementation):
    """Queries packed by documents read a context packed by its own (`key_segment_ids`): each query
    attends its document's keys as a call over that document alone does, and a query whose document
    the context does not hold, as one of padding, reads nothing, its output and gradient zero."""
    with jax.enable_x64(new_val=True):
        queries, keys = jnp.asarray([[1, 1, 2, 3, 0]]), jnp.asarray([[1, 1, 1, 3, 3, 0]])
        query, key, value = (jax.random.normal(jax.random.key(seed), (1, length, 2, 8), jnp.float64)
                             for seed, length in ((0, 5), (1, 6), (2, 6)))

        def attend(q, k, v, **documents):
            return scaled_dot_product_attention(q, k, v, implementation=implementation, **documents)

        out, pull = jax.vjp(lambda q: attend(q, key, value, segment_ids=queries, key_segment_ids=keys,
                                             max_segments=3), query)
        np.testing.assert_array_equal(out[0, 2:3], 0)
        np.testing.assert_array_equal(out[0, 4:], 0)
        np.testing.assert_array_equal(pull(jnp.ones_like(out))[0][0, 2:3], 0)
        for asking, read in ((slice(0, 2), slice(0, 3)), (slice(3, 4), slice(3, 5))):
            np.testing.assert_allclose(out[:, asking], attend(query[:, asking], key[:, read], value[:, read]),
                                       rtol=1e-12, atol=1e-14)


def test_queries_reading_another_sequences_documents_on_the_tpu_kernels_are_as_exact_as_xla():
    """Packed queries over a context packed by its own documents (`key_segment_ids`) on 'tpu', which
    interprets off a TPU: the output and the three gradients within tests/reference_error.py's rule of
    xla's against float64, a document of the queries' the context lacks among them; xla's sums are
    the TPU's, fp32 (`FP32_SUMS`), and a CPU runs them on its reference path."""
    queries = jnp.asarray([[1] * 40 + [2] * 30 + [3] * 30 + [0] * 28])
    keys = jnp.asarray([[1] * 50 + [3] * 50 + [0] * 28])
    arrays = [jax.random.normal(jax.random.key(seed), (1, 128, 2, 128), jnp.float32) for seed in range(3)]
    cotangent = jax.random.normal(jax.random.key(3), (1, 128, 2, 128), jnp.float32)

    def derivatives(implementation, *arrays):
        def attend(q, k, v):
            return scaled_dot_product_attention(q, k, v, implementation=implementation, segment_ids=queries,
                                                key_segment_ids=keys, max_segments=3)

        out, pull = jax.vjp(attend, *arrays)
        return (out, *pull(cotangent.astype(out.dtype)))

    found = derivatives("tpu", *arrays)
    with jax.default_matmul_precision(FP32_SUMS):
        reference = derivatives("xla", *arrays)
    with jax.enable_x64(new_val=True):
        truth = derivatives("reference", *(jnp.asarray(x, jnp.float64) for x in arrays))
    for name, got, want, exact in zip(("out", "dq", "dk", "dv"), found, reference, truth, strict=True):
        assert_as_exact_as_the_reference(np.asarray(got), np.asarray(want), np.asarray(exact), name)


def key_length_call(implementation, query, key, value, lengths, **kwargs):
    """The output and the three input gradients of one call that ends each
    row's keys at `lengths`, either as lengths or as the mask they mean."""
    if kwargs.pop("as_mask", False):
        kwargs["mask"] = (jnp.arange(key.shape[1]) < lengths[:, None])[:, None, None, :]
    else:
        kwargs["key_value_seq_lengths"] = lengths
    return value_and_grads(implementation, query, key, value, **kwargs)


@pytest.mark.parametrize("implementation", ["reference", "xla"])
@pytest.mark.parametrize("causal", [False, True])
def test_key_lengths_attend_as_the_mask_they_mean(implementation, causal):
    """Row b reads its first lengths[b] keys and no other: the output and
    every gradient are the call that masks the rest, and what a row's keys
    past its length hold reaches nothing, its own key and value gradients
    there included."""
    query, key, value = (x.astype(jnp.float32) for x in qkv((3, 9, 2, 16)))
    lengths = jnp.asarray([9, 5, 1], jnp.int32)
    by_length = key_length_call(implementation, query, key, value, lengths, causal=causal)
    by_mask = key_length_call(implementation, query, key, value, lengths, causal=causal, as_mask=True)
    for got, want in zip(by_length, by_mask, strict=True):
        np.testing.assert_array_equal(got, want)
    past = jnp.arange(9)[None, :, None, None] >= lengths[:, None, None, None]
    moved = key_length_call(implementation, query, jnp.where(past, 3.0, key),
                            jnp.where(past, -2.0, value), lengths, causal=causal)
    np.testing.assert_allclose(moved[0], by_length[0], atol=1e-6, rtol=1e-6)
    for gradient in by_length[2:]:
        assert not np.asarray(gradient)[np.asarray(past)[:, :, 0, 0]].any()


@on_gpu
@pytest.mark.parametrize("as_mask", [False, True], ids=["lengths", "mask"])
@pytest.mark.parametrize("q_len, kv_len, causal", [
    (1024, 1024 + 77, False),  # the image queries of a joint call over padded text
    (77, 77, True),            # its text queries, causal over the text alone
    (78, 78, True),            # an even length, which the kernel takes unpadded
])
def test_cudnn_takes_key_lengths_and_agrees_with_xla(q_len, kv_len, causal, as_mask,
                                                     without_deterministic_ops):
    """cuDNN reads the lengths as its padding mask, odd lengths included:
    within two bf16 ulps of the output scale of the xla kernel, forward and
    backward, as `test_cudnn_trains_odd_lengths_and_agrees_with_xla` bounds
    the unpadded call. The same keys ended by a `[B, 1, 1, K]` mask, the
    shape CLIP's text tower builds, agree the same way: the mask broadcasts
    over the queries on the cudnn path as it does on xla."""
    query, _, _ = qkv((2, q_len, 4, 64))
    _, key, value = qkv((2, kv_len, 4, 64), seed=1)
    lengths = jnp.asarray([kv_len, kv_len - 40], jnp.int32)
    fused = key_length_call('cudnn', query, key, value, lengths, causal=causal, as_mask=as_mask)
    reference = key_length_call('xla', query, key, value, lengths, causal=causal, as_mask=as_mask)
    for got, want in zip(fused, reference, strict=True):
        assert got.shape == want.shape
        assert np.abs(got - want).max() <= 2 ** -6 * np.abs(want).max()


@on_gpu
def test_single_query_cudnn_decode_matches_float64_attention_and_gradients(without_deterministic_ops):
    """One query, GQA and distinct key counts against the highest-precision path.

    The single query is padded for the cuDNN backward even though the
    forward accepts an odd length. The numerical bound is the existing
    two-bf16-ulp bound, not a new tolerance.
    """
    query, _, _ = qkv((3, 1, 4, 64))
    _, key, value = qkv((3, 17, 2, 64), seed=1)
    lengths = jnp.asarray([17, 8, 1], jnp.int32)
    actual = key_length_call("cudnn", query, key, value, lengths)
    with jax.enable_x64():
        def loss(q, k, v):
            attended = scaled_dot_product_attention(
                q, k, v, implementation="reference", key_value_seq_lengths=lengths,
                force_fp32_for_softmax=False, precision=jax.lax.Precision.HIGHEST)
            return jnp.sum(attended ** 2), attended

        (_, expected), gradients = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))(
            query.astype(jnp.float64), key.astype(jnp.float64), value.astype(jnp.float64))
        for got, want in zip(actual, (expected, *gradients), strict=True):
            want = np.asarray(want)
            assert got.shape == want.shape
            assert np.abs(got - want).max() <= 2 ** -6 * np.abs(want).max()


@pytest.mark.skipif(jax.default_backend() != 'cpu', reason="the float64 sum is a CPU's")
def test_cpu_fp32_attention_rounds_closer_to_float64_than_jax_nn():
    """Over 2048 keys of cross-attention-like inputs, Dew's fp32 attention on
    XLA:CPU, its products summed in float64 (`dew.nn.precision.accumulated`),
    is at most three quarters of jax.nn's xla distance from float64 (torch's
    SDPA sits between, docs/performance.md)."""
    rng = np.random.default_rng(0)
    query, key = (rng.normal(size=(1, length, 2, 128)).astype(np.float32) * 0.6 for length in (128, 2048))
    value = rng.normal(size=(1, 2048, 2, 128)).astype(np.float32)
    q, k, v = (np.asarray(x, np.float64) for x in (query, key, value))
    logits = np.einsum('btnh,bsnh->bnts', q, k) / np.sqrt(128)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    truth = np.einsum('bnts,bsnh->btnh', probs / probs.sum(-1, keepdims=True), v)
    dew = scaled_dot_product_attention(query, key, value)
    plain = jax.nn.dot_product_attention(query, key, value, implementation='xla')
    mine, theirs = distance(dew, truth), distance(plain, truth)
    assert mine <= 0.75 * theirs, (mine, theirs)


@pytest.mark.skipif(jax.default_backend() != 'cpu', reason="the float64 sum is a CPU's")
def test_cpu_fp32_attention_summed_in_float64_takes_forward_mode():
    """fp32 reference attention over 300 keys, whose products a CPU sums in float64
    (`dew.nn.precision.accumulated`), as a consistency model's JVP runs it under
    `forward_mode_attention` (sCM, rCM): its value, tangent and three gradients are within
    tests/reference_error.py's rule of jax.nn.dot_product_attention's against float64."""
    from dew.nn.attention import forward_mode_attention

    arrays = [jax.random.normal(jax.random.key(seed), (1, 300, 2, 16), jnp.float32) for seed in range(7)]
    primals, tangents, cotangent = arrays[:3], arrays[3:6], arrays[6]

    def reference(q, k, v):
        return scaled_dot_product_attention(q, k, v, implementation="reference", causal=True)

    def plain(q, k, v):
        return jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="xla")

    def derivatives(attend, *primals):
        with forward_mode_attention():
            out, tangent = jax.jvp(attend, primals, tuple(t.astype(primals[0].dtype) for t in tangents))
        _, pull = jax.vjp(attend, *primals)
        return (out, tangent, *pull(cotangent.astype(out.dtype)))

    found, theirs = derivatives(reference, *primals), derivatives(plain, *primals)
    with jax.enable_x64(new_val=True):
        truth = derivatives(reference, *(jnp.asarray(x, jnp.float64) for x in primals))
    names = ("out", "tangent", "dq", "dk", "dv")
    for name, got, want, exact in zip(names, found, theirs, truth, strict=True):
        assert_as_exact_as_the_reference(np.asarray(got), np.asarray(want), np.asarray(exact), name)


@pytest.mark.skipif(jax.default_backend() != 'cpu', reason="the float64 sum is a CPU's")
def test_a_layer_that_builds_its_mask_sums_its_cpu_fp32_values_in_float64_too():
    """A layer whose rows are padded on the left builds its mask and runs the
    kernel a materialized mask takes (`kernel_for_materialized_mask`). On a
    CPU that is the reference path, whose products sum in float64, the route
    `resolve_implementation` takes for an unmasked call (`_takes_reference`):
    the padded rows' real logits under 'auto' are the 'reference' kernel's,
    bit for bit."""
    from dew.nn.backbones.causal_transformer import CausalTransformer

    def model(implementation):
        return CausalTransformer(vocab_size=64, emb_features=32, num_layers=1, num_heads=2, mlp_features=48,
                                 max_seq_len=320, attention_impl=implementation)

    tokens = jnp.asarray(np.random.default_rng(3).integers(1, 64, (2, 300)), jnp.int32)
    valid = jnp.arange(300)[None] >= jnp.asarray([[0], [7]])
    params = model('auto').init(jax.random.key(0), tokens)
    chosen, reference = (np.asarray(model(name).apply(params, tokens, attention_mask=valid))
                         for name in ('auto', 'reference'))
    np.testing.assert_array_equal(chosen[np.asarray(valid)], reference[np.asarray(valid)])


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
@pytest.mark.parametrize("forced", [True, False])
@pytest.mark.parametrize("dropout_rate", [0.0, 0.25])
def test_the_reference_path_is_flax_attention_op_for_op(dtype, forced, dropout_rate):
    """Without sinks the reference path (`reference_attention`) is flax's
    `dot_product_attention` handed the key heads repeated over their group and
    its probability product (`weighted_values`), bitwise: in each compute dtype,
    the softmax forced to fp32 or not, an fp32 bias wider than half-precision
    logits and a causal mask. Its probability draw is Dew's own (row 1521: the
    keep bits every route draws from one rng, `template.dropout_keep`, so a
    route never changes which weights drop) and its multiplier Dew's (row 1533:
    1 / (1 - rate) rounded once in at least fp32), applied where flax applies
    its own: flax's weights (`dot_product_attention_weights`) under Dew's keep
    mask and multiplier, then the value product. Both sum their fp32 products
    as XLA does (`FP32_SUMS`), the reference path's own float64 sums on a CPU
    being the policy's (tests/test_cpu_accumulation.py)."""
    keys = jax.random.split(jax.random.key(0), 5)
    query = jax.random.normal(keys[0], (2, 6, 4, 8), dtype)
    key, value = (jax.random.normal(drawn, (2, 6, 2, 8), dtype) for drawn in keys[1:3])
    bias = jax.random.normal(keys[3], (2, 4, 6, 6), jnp.float32)
    drop = {"dropout_rate": dropout_rate, "dropout_rng": keys[4], "deterministic": not dropout_rate}
    with jax.default_matmul_precision(FP32_SUMS):
        ours = scaled_dot_product_attention(query, key, value, implementation="reference", causal=True,
                                            bias=bias, force_fp32_for_softmax=forced, **drop)
        weights = nn.dot_product_attention_weights(
            query, jnp.repeat(key, 2, axis=2), bias=bias, mask=jnp.tril(jnp.ones((6, 6), bool))[None, None],
            force_fp32_for_softmax=forced, einsum=jnp.einsum)
        if dropout_rate:
            wide = at_least_fp32(dtype)
            keep = template.dropout_keep(template.dropout_seed(keys[4]), dropout_rate, weights.shape)
            scale = jnp.asarray(1.0 / (1.0 - dropout_rate), wide)
            weights = (weights.astype(wide) * jnp.where(keep, scale, 0)).astype(dtype)
        theirs = weighted_values('...hqk,...khd->...qhd', weights, jnp.repeat(value, 2, axis=2))
    np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))


@pytest.mark.skipif(jax.default_backend() != 'cpu', reason="the routing is a CPU's")
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("implementation", ["auto", "xla"])
def test_a_cpu_attention_over_one_key_moves_no_query_or_key(implementation, dtype):
    """A query that reads one key attends it whatever the scores, so its query's and its key's
    gradients are exactly zero, as the softmax's own derivative gives them: on a CPU 'auto' and 'xla'
    take the reference path in every dtype, where jax.nn's backward, through rowsum(dO * O), left
    each a rounding's worth."""
    keys = jax.random.split(jax.random.key(3), 4)
    query = jax.random.normal(keys[0], (2, 5, 4, 8), dtype)
    key, value = (jax.random.normal(drawn, (2, 7, 4, 8), dtype) for drawn in keys[1:3])
    cotangent = jax.random.normal(keys[3], (2, 5, 4, 8), dtype)
    one = jnp.ones((2,), jnp.int32)

    def attend(query, key, value):
        return scaled_dot_product_attention(query, key, value, implementation=implementation,
                                            key_value_seq_lengths=one)

    _, into_query, into_key, _ = jax.jit(with_gradients(attend))(query, key, value, cotangent)
    assert not np.any(np.asarray(into_query, np.float32))
    assert not np.any(np.asarray(into_key, np.float32))


@pytest.mark.parametrize("implementation", ["reference", "xla"])
def test_normal_attention_ends_each_rows_keys_at_its_length_as_a_prefix_mask_does(implementation):
    """Key lengths keep each row's first keys, as the prefix mask of those lengths
    does, bitwise: a right-padded context's padding reads as nothing either way.
    Bitwise because the lengths become that very mask (`with_key_lengths`) before
    the kernel, so on a GPU, where a call this size folds its heads (8-wide
    heads, which the lengths-only decode kernel does not take), both forms run
    `folded_attention` alike."""
    module = NormalAttention(16, heads=2, dim_head=8, attention_impl=implementation)
    rng = np.random.default_rng(1)
    queries = jnp.asarray(rng.normal(size=(3, 5, 16)), jnp.float32)
    context = jnp.asarray(rng.normal(size=(3, 7, 16)), jnp.float32)
    params = module.init(jax.random.key(0), queries, context)
    lengths = jnp.asarray([7, 4, 1])
    prefix = (jnp.arange(7)[None, :] < lengths[:, None])[:, None, None, :]
    np.testing.assert_array_equal(module.apply(params, queries, context, key_value_seq_lengths=lengths),
                                  module.apply(params, queries, context, mask=prefix))


@pytest.mark.parametrize("implementation", ["reference", "xla"])
def test_normal_attention_reads_only_the_keys_its_mask_keeps(implementation):
    """A key the mask drops leaves the output bitwise unchanged when its value
    moves, which the unmasked call does not. An all-True mask computes the
    unmasked attention, but not always bitwise: XLA:GPU fuses the masked
    softmax differently (2.4e-7 apart on an A100), so it is held to
    tests/reference_error.py's rule against the unmasked call, both measured
    from float64."""
    module = NormalAttention(16, heads=2, dim_head=8, attention_impl=implementation)
    rng = np.random.default_rng(0)
    queries = jnp.asarray(rng.normal(size=(2, 5, 16)), jnp.float32)
    context = jnp.asarray(rng.normal(size=(2, 7, 16)), jnp.float32)
    moved = context.at[:, 3].add(10.0)
    params = module.init(jax.random.key(0), queries, context)
    kept = jnp.ones((2, 1, 5, 7), bool)
    dropped = kept.at[..., 3].set(False)
    np.testing.assert_array_equal(module.apply(params, queries, context, mask=dropped),
                                  module.apply(params, queries, moved, mask=dropped))
    assert not np.array_equal(module.apply(params, queries, context), module.apply(params, queries, moved))
    weights = jax.tree.map(lambda leaf: np.asarray(leaf, np.float64), params["params"])

    def projected(tokens: jax.Array, name: str) -> np.ndarray:
        return np.einsum("bsc,chd->bshd", np.asarray(tokens, np.float64), weights[name]["kernel"]) + \
            weights[name]["bias"]

    q, k, v = projected(queries, "to_q"), projected(context, "to_k"), projected(context, "to_v")
    logits = np.einsum("bshd,bthd->bhst", q, k) / np.sqrt(8)
    probabilities = np.exp(logits - logits.max(-1, keepdims=True))
    attended = np.einsum("bhst,bthd->bshd", probabilities / probabilities.sum(-1, keepdims=True), v)
    truth = np.einsum("bshd,hdc->bsc", attended, weights["to_out_0"]["kernel"]) + weights["to_out_0"]["bias"]
    assert_as_exact_as_the_reference(np.asarray(module.apply(params, queries, context, mask=kept)),
                                     np.asarray(module.apply(params, queries, context)), truth,
                                     f"{implementation} attention under an all-True mask")


def eligibility_cases():
    """Calls covering each GPU kernel's requirements pairwise: dtypes, head widths, causality, a window,
    packed documents with and without a bound, a cache call's aligned documents, a softcap and sinks."""
    return [
        {"dtype": jnp.bfloat16, "head_dim": 64, "causal": True},
        {"dtype": jnp.bfloat16, "head_dim": 128, "causal": False, "window": 16},
        {"dtype": jnp.bfloat16, "head_dim": 256, "causal": True, "window": 16},
        {"dtype": jnp.float16, "head_dim": 96, "causal": True, "documents": "own", "max_segments": 2},
        {"dtype": jnp.bfloat16, "head_dim": 64, "causal": True, "documents": "own"},
        {"dtype": jnp.bfloat16, "head_dim": 128, "causal": True, "documents": "cross", "max_segments": 1,
         "queries": 8},
        {"dtype": jnp.bfloat16, "head_dim": 128, "causal": True, "softcap": 30.0},
        {"dtype": jnp.bfloat16, "head_dim": 64, "causal": True, "sinks": True},
        {"dtype": jnp.float32, "head_dim": 64, "causal": True},
        {"dtype": jnp.float32, "head_dim": 72, "causal": False},
    ]


def eligibility_call(dtype, head_dim, causal, window=None, documents=None, max_segments=None, softcap=None,
                     sinks=False, queries=64):
    """A call of `eligibility_cases` as `scaled_dot_product_attention`'s arguments."""
    from dew.nn.documents import CrossDocuments

    rng = np.random.default_rng(head_dim)
    query = jnp.asarray(rng.normal(size=(2, queries, 4, head_dim)), dtype)
    key, value = (jnp.asarray(rng.normal(size=(2, 64, 2, head_dim)), dtype) for _ in range(2))
    segment_ids = None
    if documents == "own":
        segment_ids = jnp.asarray(np.repeat([[1] * 40 + [2] * 24], 2, axis=0), jnp.int32)
    elif documents == "cross":
        filled = np.arange(64)[None, :] < np.asarray([[30], [64]])
        segment_ids = CrossDocuments(jnp.ones((2, queries), jnp.int32), jnp.asarray(filled, jnp.int32))
    return (query, key, value), {"causal": causal, "sliding_window": window, "segment_ids": segment_ids,
                                 "max_segments": max_segments, "softcap": softcap,
                                 "sinks": jnp.zeros((4,), jnp.float32) if sinks else None}


@pytest.mark.gpu(reason="each requirement is held to the kernel it declares, on this GPU")
@pytest.mark.parametrize("kernel", ["flash", "triton", "cudnn", "xla", "template"])
@pytest.mark.parametrize("call", eligibility_cases(),
                         ids=lambda call: "-".join(f"{name}={value}" for name, value in call.items()))
def test_a_kernels_requirements_are_what_the_kernel_takes_on_this_gpu(kernel, call,
                                                                      without_deterministic_ops):
    """Each GPU kernel's declared requirements (`attention_kernels.ATTENTION`) hold against the kernel
    itself on this GPU: a call whose capability requirements all hold runs forced on that kernel to a
    finite output, and one that fails a capability raises `KernelRefused` naming it; measured
    requirements, which a forced kernel overrides, never stop it."""
    from dew.nn.attention import attention_case
    from dew.nn.attention_kernels import ATTENTION
    from dew.nn.kernels.selection import KernelRefused

    operands, arguments = eligibility_call(**call)
    case = attention_case(operands[0], operands[1], **arguments)
    failed = [requirement for requirement in ATTENTION.kernel(kernel).failed(case, forced=True)
              if requirement.kind == 'capability']

    def run():
        out = jax.jit(lambda *operands: scaled_dot_product_attention(
            *operands, implementation=kernel, **arguments))(*operands)
        return np.asarray(out, np.float32)

    if failed:
        with pytest.raises(KernelRefused) as refused:
            run()
        raised = {refusal.code for refusal in refused.value.refusals}
        assert {requirement.code for requirement in failed} <= raised
    else:
        assert np.all(np.isfinite(run()))


@pytest.mark.parametrize("generation_name, narrow", [("sm80", "flash"), ("sm86", "flash"),
                                                     ("sm90", "flash"), ("sm89", "triton")])
def test_narrow_heads_take_the_kernel_their_target_measured_fastest(generation_name, narrow):
    """A causal bf16 call of 64-wide heads prefers its GPU row's narrow-heads kernel
    ('attention_narrow'): the FA wheel where it was measured fastest, an A10, an A100 and an H100
    (`generation.TARGETS`), tokamax's Triton kernel where the row names none. FlashAttention's own row
    requirement reads the same row, so where the wheel holds code for the GPU the preference is the
    route, an A10's narrow heads included where its attention row names none."""
    from dew.nn.attention import attention_case, resolve_implementation
    from dew.nn.attention_kernels import ATTENTION, kernel_refusals
    from dew.nn.kernels import generation

    query = jnp.zeros((2, 128, 4, 64), jnp.bfloat16)
    with generation.assumed(generation_name), generation.routes() as found:
        resolve_implementation("auto", query, query, causal=True)
        refused = {refusal.code for refusal in kernel_refusals(
            ATTENTION, narrow, attention_case(query, query, causal=True))}
    (route,) = [route for route in found if route.choice == "attention"]
    assert route.preferred == narrow
    # The row qualifies the preferred kernel: only host facts (backend, wheel) may turn it down here.
    assert "row" not in refused


@pytest.mark.parametrize("generation_name, call, kernel", [
    ("sm90", {"causal": True}, "flash"),
    ("sm90", {"causal": True, "softcap": 30.0}, "template"),
    ("sm90", {"causal": True, "key_value_seq_lengths": True}, "template"),
    ("sm90", {"causal": True, "mask": True}, None),
    ("sm80", {"causal": True, "softcap": 30.0}, None),
], ids=["h100-plain", "h100-softcap", "h100-lengths", "h100-mask", "a100-softcap"])
def test_attention_runs_its_declared_cells_and_the_template_elsewhere(generation_name, call, kernel):
    """On a family whose Mosaic template has a routed mainloop (wgmma's), a call runs its declared old route
    where that takes it (`attention_kernels.OLD_ROUTES`: FlashAttention-3 on Hopper), the template on every
    other call it takes (a softcap, key lengths), and the next kernel by legality on a call the template does
    not take (an arbitrary mask); a family without one (mma.sync's) keeps its target row's route, which is
    never the template but where a row names it (bible §4, owner 10-25). `kernel` None: neither the
    template nor the declared route."""
    from dew.nn.attention import resolve_implementation
    from dew.nn.kernels import generation

    query = jnp.zeros((2, 128, 4, 64), jnp.bfloat16)
    keywords = {"causal": call["causal"], "softcap": call.get("softcap")}
    if call.get("key_value_seq_lengths"):
        keywords["key_value_seq_lengths"] = jnp.full((2,), 100, jnp.int32)
    if call.get("mask"):
        keywords["mask"] = jnp.ones((2, 1, 128, 128), bool)
    with generation.assumed(generation_name), generation.routes() as found:
        resolve_implementation("auto", query, query, **keywords)
    (route,) = [route for route in found if route.choice == "attention"]
    if kernel is None:
        assert route.kernel not in ("template", "flash")
    else:
        assert route.kernel == kernel


@pytest.mark.parametrize("dtype, kept", [(jnp.float16, False), (jnp.bfloat16, True)])
def test_an_fp16_call_leaves_the_triton_route_as_a_declared_gap(dtype, kept):
    """A causal call of 16-wide heads where an L4's row prefers tokamax's Triton kernel: an fp16 one is
    refused by its dtype, whose dQ that kernel narrows before the scale the route applies outside it
    (row 1676, `triton_attention`), and the next kernel takes it; a bf16 one keeps it."""
    from dew.nn.attention import attention_case, resolve_implementation
    from dew.nn.attention_kernels import ATTENTION, kernel_refusals
    from dew.nn.kernels import generation

    query = jnp.zeros((2, 128, 4, 16), dtype)
    with generation.assumed("sm89"), generation.routes() as found:
        resolve_implementation("auto", query, query, causal=True)
        refused = {refusal.code for refusal in kernel_refusals(
            ATTENTION, "triton", attention_case(query, query, causal=True))}
    (route,) = [route for route in found if route.choice == "attention"]
    assert ("fp16" not in refused) == kept
    assert (route.kernel == "triton") == kept


@pytest.mark.parametrize("head_dim, kept", [(8, False), (16, True)])
def test_heads_narrower_than_the_triton_kernel_computes_take_the_next_kernel(head_dim, kept):
    """A causal bf16 call of narrow heads prefers tokamax's Triton kernel where its GPU's row names it
    for them, an L4's: 8-wide heads, which that kernel computes wrong (`TRITON_NARROWEST`), are refused
    by their width and the next kernel takes the call; 16-wide heads keep it."""
    from dew.nn.attention import attention_case, resolve_implementation
    from dew.nn.attention_kernels import ATTENTION, kernel_refusals
    from dew.nn.kernels import generation

    query = jnp.zeros((2, 128, 4, head_dim), jnp.bfloat16)
    with generation.assumed("sm89"), generation.routes() as found:
        resolve_implementation("auto", query, query, causal=True)
        refused = {refusal.code for refusal in kernel_refusals(
            ATTENTION, "triton", attention_case(query, query, causal=True))}
    (route,) = [route for route in found if route.choice == "attention"]
    assert route.preferred == "triton"
    assert ("head_dim" not in refused) == kept
    assert (route.kernel == "triton") == kept


@pytest.mark.parametrize("call, refused", [
    ({"precision": jax.lax.Precision.HIGHEST}, ["arithmetic"]),
    ({"key_value_seq_lengths": jnp.array([512])}, ["key_lengths"]),
    ({"bias": jnp.zeros((1, 4, 512, 512), jnp.bfloat16)}, ["bias"]),
], ids=["arithmetic", "key lengths", "bias"])
def test_a_splash_call_tiled_but_refused_otherwise_is_reported_by_its_own_requirement(call, refused):
    """On a TPU, a call splash's blocking tiles but another of its requirements turns down is refused
    by that requirement alone, not for its tiling: the case builder forms the mask-descriptor fact
    under a name that may choose splash on a TPU (`TPU_BACKEND`) whatever else the call asks, and
    `tiles` reads the lengths from the case."""
    from dew.nn.attention import attention_case
    from dew.nn.attention_kernels import ATTENTION, kernel_refusals
    from dew.nn.kernels import generation

    query = jnp.zeros((1, 512, 4, 64), jnp.bfloat16)
    with generation.assumed("v5e"):
        case = attention_case(query, query, causal=True, **call)
        found = [refusal.code for refusal in kernel_refusals(ATTENTION, "tpu", case)]
    assert case.splash_mask and found == refused
