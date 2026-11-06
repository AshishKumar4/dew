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
from attention_oracle import (
    attention_and_grads,
    attention_on_par,
    exact_attention,
    kept,
    qkv,
    value_and_grads,
)
from contract_support import normal_drawn
from flax import linen as nn
from reference_error import as_given, assert_as_exact, widened, with_gradients

from dew.nn import kernels
from dew.nn.attention import scaled_dot_product_attention, weighted_values
from dew.nn.attention_blocks import NormalAttention
from dew.nn.kernels.selection import KernelRefused
from dew.nn.kernels.templates import attention as template
from dew.nn.precision import FP32_SUMS, WIDE_CPU_ARITHMETIC, at_least_fp32
from dew.nn.visibility import window_sides

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


def flash_here():
    """dew_flash_attn where the installed wheel holds code for this GPU (the fact `_flash_built` reads);
    a GPU it was not built for skips by name, where 'auto' runs another kernel."""
    flash = pytest.importorskip("dew_flash_attn")
    from dew.nn.attention_kernels import flash_architectures, flash_built_for
    generation = kernels.device_generation()
    architectures = flash_architectures() or ()
    if not flash_built_for(generation, architectures):
        pytest.skip(f"the installed dew_flash_attn holds code for {', '.join(architectures)}, "
                    f"not {generation}")
    return flash


def flash_through_dews_attention_refuses_rows_it_would_misread(without_deterministic_ops):
    """Through Dew's attention (`scaled_dot_product_attention`, 'flash'), keys and values of another
    batch, width, length or head count than the query's are refused as the call traces, before any
    kernel runs, dense and packed. Packing flattens the rows into one run of tokens, where keys of one
    row of 64 beside queries of two rows of 32, or another sequence's keys of one row of 32 beside
    their ids of two rows of 16, flatten to as many tokens and would be read as the next row's: the
    call names the ids. Each well-formed twin, dense with grouped heads and packed, gives its output
    and gradients on par with the xla kernel's over drawn inputs (check 1), the truth the xla kernel's
    float64 run of the same call."""
    flash_here()
    rng = np.random.default_rng(31)

    def normal(*shape):
        return jnp.asarray(rng.normal(size=shape), jnp.bfloat16)

    query, key = normal(2, 32, 4, 64), normal(2, 32, 2, 64)
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

    def draw(seed):
        streams = np.random.default_rng([31, seed])
        return tuple(jnp.asarray(streams.normal(size=shape), jnp.bfloat16)
                     for shape in ((2, 32, 4, 64), (2, 32, 2, 64), (2, 32, 2, 64), (2, 32, 4, 64)))

    for packed in ({}, own):
        attention_on_par(attend('flash', **packed), attend('xla', **packed), draw,
                         f"flash {'packed' if packed else 'dense'}",
                         keep=kept(32, 32, causal=True, segment_ids=packed.get("segment_ids")))


SHARDED_SEEDS = 8
"""The per-shard contract's declared draws (check 1, `sharded.assert_runs_per_shard`), sized by
`seeds_for`: the fewest for which one device compared with itself resolves the factor."""
KEY_LENGTH_SEEDS = 8
"""Key lengths against their mask's declared draws (check 1), sized by `seeds_for` paired on armada."""
FLAX_SEEDS = 8
"""The reference path against flax's declared draws (check 1), sized by `seeds_for` paired on armada."""


@pytest.mark.parametrize("implementation", ["cudnn", "flash"])
def a_fused_kernel_takes_forward_mode_through_the_reference(implementation, without_deterministic_ops):
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


@pytest.mark.parametrize("q_len, kv_len, causal, head_dim", [
    (1024, 77, False, 64),   # every cross-attention over CLIP's 77 text tokens
    (9, 7, False, 64),       # short enough that one attended pad key would move an eighth of the mass
    (333, 333, True, 64),    # the concatenated text-plus-image sequence, causal
    (3072, 3072, True, 256),  # Qwen3.5's 256-wide heads, which cudnn takes from Hopper on
])
def cudnn_trains_odd_lengths_and_agrees_with_xla(q_len, kv_len, causal, head_dim,
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


@pytest.mark.parametrize("q_len, kv_len, heads, kv_heads, head_dim, causal", [
    (256, 256, 12, 12, 64, False),   # SimpleDiT-B's image tokens
    (512, 512, 12, 12, 64, True),    # the small decoder
    (1024, 1024, 16, 8, 128, True),  # Qwen3-0.6B's grouped heads
    (333, 333, 4, 4, 64, True),      # an odd length, which cudnn pads
    (1024, 77, 4, 4, 64, False),     # cross-attention over CLIP's 77 tokens
    (256, 256, 8, 8, 80, True),      # a head width that is no power of two
])
def triton_trains_and_agrees_with_xla(q_len, kv_len, heads, kv_heads, head_dim, causal):
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


def triton_trains_at_large_logits_on_par_with_xla():
    """tokamax's kernel, as 'triton' runs it, at logits near 1e4 that its softmax spreads over every key
    (`_large_logits`): its output and its query, key and value gradients on par with the xla kernel's from
    float64 (check 1). Its dK/dV sweep scaled the keys where its forward and dQ scale the queries, so its
    scores there rounded apart from the forward's residuals by units, and the weights it rebuilt from them
    were off by their exponential (row 1676: a MeanFlow step's key and value gradients 1e35 to 1e62 times
    xla's error); the route hands it the queries scaled once (`triton_attention`)."""
    pytest.importorskip("tokamax")
    assert_as_exact(_pulled("triton"), _pulled("xla"), _large_logits(spread=True), "triton at large logits",
                    seeds=LARGE_LOGIT_SEEDS)


def triton_at_saturated_large_logits_is_finite_where_xla_is():
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


@pytest.mark.parametrize("q_len, kv_len, heads, kv_heads, head_dim, causal", [
    (1024, 1024, 16, 8, 128, True),  # Qwen3-0.6B's grouped heads
    (512, 512, 8, 8, 256, True),     # 256-wide heads, which cudnn takes on Hopper alone
    (256, 256, 12, 12, 64, False),   # SimpleDiT-B's image tokens
    (333, 333, 4, 4, 64, True),      # an odd length
    (1024, 77, 4, 4, 64, False),     # cross-attention over CLIP's 77 tokens
    (256, 256, 8, 8, 80, True),      # a head width that is no power of two
])
def flash_trains_and_agrees_with_xla(q_len, kv_len, heads, kv_heads, head_dim, causal,
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


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("sunk", [False, True])
def packed_sequences_attend_within_themselves_as_far_from_float64_as_one_at_a_time(
        head_dim, causal, sunk, without_deterministic_ops):
    """Documents packed back to back, one a single token, with an empty sequence between two: the packed
    call's output and its input gradients on par with FlashAttention's own on each document alone over
    drawn inputs (check 1, the truth each document's float64 attention), with a 64-key window and sinks
    the sinks' gradient too, summed over the documents. Not bitwise: on one
    short document the dense forward splits its keys across blocks (`_splits`) and the packed one
    does not, so the two round differently (on an RTX 4080, 1.0003 to 1.012 times apart in RMS)."""
    flash = flash_here()
    if head_dim > 192 and kernels.device_generation() in ('sm86', 'sm87', 'sm89'):
        pytest.skip("FlashAttention-2's backward takes heads over 192 wide only on sm80, sm90 and sm1xx")
    lengths = [700, 1, 0, 333, 64]
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    documents = [slice(begin, end) for begin, end in itertools.pairwise(starts.tolist()) if end > begin]
    window = (63, 0 if causal else 63) if sunk else (-1, -1)
    ids = np.repeat(np.arange(1, len(lengths) + 1), lengths)[None]

    def draw(seed):
        rows = int(starts[-1])
        streams = np.random.default_rng([0x9AC, seed])
        q, k, v, dy = (jnp.asarray(streams.normal(size=(rows, heads, head_dim)), jnp.bfloat16)
                       for heads in (8, 2, 2, 8))
        sinks = (jnp.asarray(streams.normal(size=(8,)) * 2, jnp.float32),) if sunk else ()
        return q, k, v, *sinks, dy

    def options(sinks):
        return {"is_causal": causal, "window_size": window, "sinks": sinks[0] if sinks else None}

    def packed(q, k, v, *s):
        return flash.flash_mha_varlen(q, k, v, starts, max(lengths), **options(s))

    def dense(q, k, v, *s):
        # Each document alone, its rows put back where the packed call holds them.
        return jnp.concatenate([flash.flash_mha(q[None, at], k[None, at], v[None, at], **options(s))[0]
                                for at in documents])

    def by_document(leaves):
        # Each document's rows a leaf of their own, so a document's error answers to its own budget (a
        # one-token document's exactly zero where the reference's is); the sinks' gradient summed whole.
        return (*(leaf[at] for leaf in leaves[:4] for at in documents), *leaves[4:])

    attention_on_par(packed, dense, draw, "packed documents", split=by_document,
                     keep=kept(len(ids[0]), len(ids[0]), causal=causal, window=window, segment_ids=ids))


@pytest.mark.parametrize("head_dim", [64, 128])
def queries_packed_against_keys_of_their_own_lengths_attend_their_own_documents(head_dim,
                                                                                     without_deterministic_ops):
    """Cross-attention between packed documents: sequence i's queries, packed by their own offsets,
    attend sequence i's keys, packed by theirs, as a decision head's option slots read their own
    row's tokens. The packed call's output and its query, key and value gradients are on par with
    FlashAttention's own on each document alone over drawn inputs (check 1, the truth each document's
    float64 attention), a one-query document and an empty one among them; a document's keys no query
    reads take no gradient."""
    flash = flash_here()
    queries, keys = [30, 1, 0, 45, 7], [700, 333, 50, 120, 64]
    starts, key_starts = (np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
                          for lengths in (queries, keys))
    pairs = list(zip(itertools.pairwise(starts.tolist()), itertools.pairwise(key_starts.tolist()),
                     strict=True))

    def draw(seed):
        streams = np.random.default_rng([0xC055, seed])
        rows = (int(starts[-1]), int(key_starts[-1]), int(key_starts[-1]), int(starts[-1]))
        return tuple(jnp.asarray(streams.normal(size=(count, heads, head_dim)), jnp.bfloat16)
                     for count, heads in zip(rows, (8, 2, 2, 8), strict=True))

    def packed(q, k, v):
        return flash.flash_mha_varlen(q, k, v, starts, max(queries), key_cu_seqlens=key_starts,
                                      key_max_seqlen=max(keys))

    def dense(q, k, v):
        # Each document's queries against its own keys, the rows put back where the packed call holds
        # them; an empty document's keys read by no query.
        return jnp.concatenate([
            flash.flash_mha(q[None, begin:end], k[None, key_begin:key_end], v[None, key_begin:key_end])[0]
            for (begin, end), (key_begin, key_end) in pairs if end > begin])

    def by_document(leaves):
        # Each document's query rows, and its key and value rows, a leaf of their own.
        out, dq, dk, dv = leaves
        return (*(x[begin:end] for x in (out, dq) for (begin, end), _ in pairs if end > begin),
                *(x[begin:end] for x in (dk, dv) for _, (begin, end) in pairs))

    query_ids, key_ids = (np.repeat(np.arange(len(lengths)), lengths) for lengths in (queries, keys))
    attention_on_par(packed, dense, draw, "documents against their own keys", split=by_document,
                     keep=kept(len(query_ids), len(key_ids), mask=query_ids[:, None] == key_ids[None, :]))
    # A document no query reads leaves its keys and values no gradient.
    q, k, v, dy = draw(0)
    _, dk, dv = jax.vjp(packed, q, k, v)[1](dy)
    for (begin, end), (key_begin, key_end) in pairs:
        if begin == end:
            np.testing.assert_array_equal(np.asarray(dk[key_begin:key_end]), 0)
            np.testing.assert_array_equal(np.asarray(dv[key_begin:key_end]), 0)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axis, split", [("data", (0,)), ("tensor", (2,))])
def flash_keeps_the_kernel_contract_on_a_mesh(axis, split, without_deterministic_ops):
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


@pytest.mark.parametrize("implementation", ["flash", "cudnn"])
@pytest.mark.parametrize("head_dim", [128, 256])
def fused_kernels_are_as_exact_as_the_xla_kernel_from_float64(implementation, head_dim,
                                                                   without_deterministic_ops):
    """At Qwen3.5's grouped causal heads, 8 queries over 2 keys, FlashAttention's and cuDNN's output
    and three input gradients on par with the xla kernel's over drawn inputs (check 1, the truth its
    float64 run); on Hopper that holds FlashAttention-3 and cuDNN's 256-wide heads to it."""
    from dew.nn.attention_kernels import cudnn_max_head_dim
    if implementation == 'flash':
        flash_here()
        if head_dim > 192 and kernels.device_generation() in ('sm86', 'sm87', 'sm89'):
            pytest.skip("FlashAttention-2's backward takes heads over 192 wide only on sm80, sm90 and sm1xx")
    elif head_dim > cudnn_max_head_dim():
        pytest.skip(f"cudnn takes heads at most {cudnn_max_head_dim()} wide on {kernels.device_generation()}")

    def draw(seed):
        streams = np.random.default_rng([0xF05, seed])
        return tuple(jnp.asarray(streams.normal(size=(1, 1024, heads, head_dim)), jnp.bfloat16)
                     for heads in (8, 2, 2, 8))

    def kernel(implementation):
        return lambda q, k, v: scaled_dot_product_attention(q, k, v, implementation=implementation,
                                                            causal=True)

    attention_on_par(kernel(implementation), kernel('xla'), draw, f"{implementation} at {head_dim}",
                     keep=kept(1024, 1024, causal=True))


@pytest.mark.parametrize("width, value_width, kv_heads, causal, window, sinks", [
    (64, 64, 2, True, None, True),        # gpt-oss-20b's full layers
    (64, 64, 2, True, 128, True),         # and its sliding ones
    (192, 128, 2, True, 128, True),       # MiMo-V2-Flash's sliding layers
    (192, 128, 2, True, 128, False),      # and their narrow values on flash itself
    (64, 64, 8, False, 65, False),        # ModernBERT's local layers, both sides
])
def flash_with_a_window_and_sinks_is_as_exact_as_the_xla_kernel_from_float64(
        width, value_width, kv_heads, causal, window, sinks, without_deterministic_ops):
    """FlashAttention with a sliding window or a value narrower than the query, as 'auto' sends such calls
    on sm80, sm89 and sm90, and 'auto' with attention sinks, which FlashAttention refuses (its backward
    misreads a row a sink dominates, row 2410): the output and the gradients of the query, key, value and
    sinks on par with the xla kernel's over drawn inputs (check 1, the truth its float64 run)."""
    flash_here()

    def draw(seed):
        streams = np.random.default_rng([57, seed])
        q, k, v, dy = (jnp.asarray(streams.normal(size=(1, 1024, heads, size)), jnp.bfloat16)
                       for heads, size in ((8, width), (kv_heads, width), (kv_heads, value_width),
                                           (8, value_width)))
        sunk = (jnp.asarray(streams.normal(size=(8,)) * 2, jnp.float32),) if sinks else ()
        return q, k, v, *sunk, dy

    def kernel(implementation):
        return lambda q, k, v, *s: scaled_dot_product_attention(
            q, k, v, implementation=implementation, causal=causal, sliding_window=window,
            sinks=s[0] if s else None)

    if sinks:
        q, k, v, s, _ = draw(0)
        with pytest.raises(KernelRefused):
            jax.jit(kernel('flash'))(q, k, v, s)
    attention_on_par(kernel('auto' if sinks else 'flash'), kernel('xla'), draw,
                     f"flash at width {width}, window {window}, sinks {sinks}",
                     keep=kept(1024, 1024, causal=causal,
                               window=None if window is None else window_sides(causal=causal,
                                                                               sliding_window=window)))


MAPPED_FLASH_SEEDS = 32
"""The mapped flash call's declared draws (check 1), dense and packed alike."""

MAPPED_DOCUMENTS = (40, 1, 0, 55)
"""The packed mapped call's documents' lengths, one empty and one a single token."""


def _mapped(attend, packed: bool, rowwise: bool = False):
    """`attend` (the flash module, or "truth" for the float64 oracle) over three rows, mapped by `jax.vmap`
    as a model whose rows step their own weights maps its decoder (dew.objectives.lm.stacking), or with
    `rowwise` called a row at a time and stacked: each row's output and its query, key and value gradients
    under its cotangent. Packed, the oracle reads each document alone (`exact_attention` anchored)."""
    starts = np.concatenate([[0], np.cumsum(MAPPED_DOCUMENTS)]).astype(np.int32)
    documents = [slice(begin, end) for begin, end in itertools.pairwise(starts) if end > begin]

    def truth(q, k, v):
        if not packed:
            return exact_attention(q, k, v, keep=kept(q.shape[1], k.shape[1], causal=True), anchored=True)
        return jnp.concatenate([exact_attention(q[None, at], k[None, at], v[None, at], anchored=True,
                                                keep=kept(*(at.stop - at.start,) * 2, causal=True))[0]
                                for at in documents])

    def flash(q, k, v):
        if not packed:
            return attend.flash_mha(q, k, v, is_causal=True)
        return attend.flash_mha_varlen(q, k, v, starts, max(MAPPED_DOCUMENTS), is_causal=True)

    row = with_gradients(truth if attend == "truth" else flash)
    if rowwise:
        return jax.jit(lambda *rows: jax.tree.map(lambda *each: jnp.stack(each),
                                                  *(row(*(x[at] for x in rows)) for at in range(3))))
    return jax.jit(jax.vmap(row))


def _mapped_draw(packed: bool, batch: int = 2, length: int = 128, heads: tuple[int, int] = (4, 2)):
    """A seed's bf16 queries `[3, ..., H, 64]`, keys and values of their own heads (`heads`) and the
    output's cotangent, three mapped rows: of a batch of `batch` rows of `length` tokens, or of the packed
    documents' 96 (`MAPPED_DOCUMENTS`)."""
    rows = (3, sum(MAPPED_DOCUMENTS)) if packed else (3, batch, length)

    def draw(seed):
        streams = jax.random.split(jax.random.PRNGKey(seed), 4)
        return tuple(jax.random.normal(stream, (*rows, count, 64), jnp.float32).astype(jnp.bfloat16)
                     for stream, count in zip(streams, (heads[0], heads[1], heads[1], heads[0]), strict=True))

    return draw


def a_mapped_flash_call_is_each_rows_own(without_deterministic_ops):
    """Under `jax.vmap` the flash kernel's dense and packed calls fold the mapped axis into their own and run
    once, a group of them the kernel's grid holds a call: each mapped row's output and query, key and value
    gradients on par with the same row's own call (check 1, both from the float64 oracle mapped alike,
    rows sized by their float64 terms). The fold is the claim; the kernel's exactness against the xla
    kernel is its other cells'."""
    flash = flash_here()
    # Three mapped calls of 32,768 rows fold one a call, the kernel's 65,535 rows holding no two.
    for packed, draw, label in ((False, _mapped_draw(packed=False), "dense"),
                                (True, _mapped_draw(packed=True), "packed"),
                                (False, _mapped_draw(packed=False, batch=32_768, length=1, heads=(1, 1)),
                                 "dense past the kernel's grid")):
        assert_as_exact(_mapped(flash, packed), _mapped(flash, packed, rowwise=True), draw,
                        f"the mapped flash call, {label}", seeds=MAPPED_FLASH_SEEDS,
                        truth_fn=_mapped("truth", packed), terms=True)


def a_fold_holds_the_kernels_grid_and_its_offsets(without_deterministic_ops):
    """The mapped calls a kernel call folds (`dew_flash_attn._mapped_groups`): as many as keep its rows
    within the 65,535 its grid holds and a packed fold's tokens within its int32 offsets, at least one:
    eight calls of 1,000 rows at once, three of 32,768 one a call, four packed of 2^30 tokens one a call,
    and two of 70,000 rows one a call. A map over no call, dense or packed, launches none: its output and
    its query, key and value gradients are empty along the mapped axis, each of one call's shape."""
    flash = flash_here()
    assert flash._mapped_groups(8, 1_000) == [(0, 8)]
    assert flash._mapped_groups(3, 32_768) == [(0, 1), (1, 2), (2, 3)]
    assert flash._mapped_groups(4, 2, 2 ** 30) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert flash._mapped_groups(2, 70_000) == [(0, 1), (1, 2)]
    for packed in (False, True):
        rows = _mapped_draw(packed)(0)
        empty = _mapped(flash, packed)(*(x[:0] for x in rows))
        assert [x.shape for x in empty] == [(0, *x.shape[1:]) for x in (rows[0], *rows[:3])]


def a_window_over_more_keys_than_queries_keeps_the_keys_each_row_centres(without_deterministic_ops):
    """A bidirectional window over more keys than queries (256 queries over 512 keys, 32 each side)
    keeps the keys `structural_mask` centres on each query's row, Dew's contract, whichever kernel
    'auto' picks on this GPU: its output and gradients on par with the xla kernel's over drawn inputs
    (check 1, the truth the reference kernel's float64 run). FlashAttention aligns a window to the last
    key, the keys 256 further on, so 'auto' leaves it such a call and an explicit 'flash' refuses it,
    while it takes the same window square."""
    flash_here()

    def draw(seed):
        streams = np.random.default_rng([61, seed])
        return tuple(jnp.asarray(streams.normal(size=(1, length, 8, 64)), jnp.bfloat16)
                     for length in (256, 512, 512, 256))

    def kernel(implementation):
        return lambda q, k, v: scaled_dot_product_attention(q, k, v, implementation=implementation,
                                                            sliding_window=33)

    attention_on_par(kernel('auto'), kernel('xla'), draw, "a window over more keys than queries",
                     truth=kernel('reference'))
    query, key, value, _ = draw(0)
    with pytest.raises(ValueError):
        jax.jit(kernel('flash'))(query, key, value)
    kernel('flash')(query, key[:, :256], value[:, :256]).block_until_ready()


def read_by_documents(queries, keys) -> np.ndarray:
    """The keys each query keeps, `[B, 1, q, k]`, where queries packed by their documents' ids
    (`segment_ids`) read keys packed by theirs (`key_segment_ids`): its own document's, none for padding."""
    queries, keys = np.asarray(queries), np.asarray(keys)
    return ((queries[:, :, None] == keys[:, None, :]) & (queries[:, :, None] != 0))[:, None]


def documents(lengths) -> np.ndarray:
    """Packed rows' segment ids, each row's documents of `lengths[:-1]` tokens numbered from 1 and its
    padding of `lengths[-1]` after them."""
    return np.stack([np.concatenate([np.repeat(np.arange(1, len(row)), row[:-1]), np.zeros(row[-1], int)])
                     for row in lengths]).astype(np.int32)


@pytest.mark.parametrize("window, sinks", [(None, False), (128, True)])
def packed_rows_on_flash_are_as_exact_as_the_document_mask_from_float64(window, sinks,
                                                                             without_deterministic_ops):
    """Rows packing documents of uneven lengths, padding after them, through flash as its varlen
    sequences (`segment_ids` with their declared `max_segments`), causal, also with a 128-key window
    and sinks ('auto', which FlashAttention refuses them to): every real token's output and every input
    gradient on par with the xla kernel's under the document mask over drawn inputs (check 1, the truth
    its float64 run). The padding attends among itself on flash and to nothing under the mask, and its
    rows carry no gradient."""
    flash_here()
    lengths = [[700, 1, 220, 103], [129, 383, 400, 112]]
    segments = documents(lengths)
    real = jnp.asarray(segments != 0)[:, :, None, None]

    def draw(seed):
        streams = np.random.default_rng([23, seed])
        q, k, v, dy = (jnp.asarray(streams.normal(size=(2, 1024, heads, 64)), jnp.bfloat16)
                       for heads in (8, 2, 2, 8))
        # The padding's rows take no cotangent: its output is no token's.
        sunk = (jnp.asarray(streams.normal(size=(8,)) * 2, jnp.float32),) if sinks else ()
        return q, k, v, *sunk, jnp.where(real, dy, 0)

    def kernel(implementation):
        # The padding's rows read as zero: flash attends it among itself, the mask to nothing.
        return lambda q, k, v, *s: jnp.where(real, scaled_dot_product_attention(
            q, k, v, implementation=implementation, causal=True, sliding_window=window,
            sinks=s[0] if s else None, segment_ids=jnp.asarray(segments), max_segments=3), 0)

    if sinks:
        q, k, v, s, _ = draw(0)
        with pytest.raises(KernelRefused):
            jax.jit(kernel('flash'))(q, k, v, s)
    attention_on_par(kernel('auto' if sinks else 'flash'), kernel('xla'), draw,
                     f"packed rows on flash, window {window}, sinks {sinks}",
                     keep=kept(1024, 1024, causal=True, segment_ids=segments,
                               window=None if window is None else window_sides(causal=True,
                                                                               sliding_window=window)))


def queries_reading_another_sequences_documents_on_flash_are_as_exact_as_the_document_mask(
        without_deterministic_ops):
    """A decision head's option slots read the window's tokens of their own row: queries packed by
    their documents (`segment_ids`), keys by theirs (`key_segment_ids`), through flash as packed
    sequences of their own lengths. Every query's output and every input gradient on par with the xla
    kernel's under the document mask over drawn inputs (check 1, the truth its float64 run); a document
    no query reads takes no gradient, and a padding query, or one whose document holds no key, reads
    nothing."""
    flash_here()
    queries = documents([[20, 1, 40, 35], [9, 0, 30, 57]])
    # Row 0's second document has a query and no key.
    keys = documents([[300, 0, 500, 224], [129, 383, 400, 112]])
    real = jnp.asarray(queries != 0)[:, :, None, None]

    def draw(seed):
        streams = np.random.default_rng([29, seed])
        q, k, v, dy = (jnp.asarray(streams.normal(size=(2, length, 8, 64)), jnp.bfloat16)
                       for length in (96, 1024, 1024, 96))
        return q, k, v, jnp.where(real, dy, 0)

    def kernel(implementation):
        return lambda q, k, v: scaled_dot_product_attention(
            q, k, v, implementation=implementation, segment_ids=jnp.asarray(queries),
            key_segment_ids=jnp.asarray(keys), max_segments=3)

    attention_on_par(kernel('flash'), kernel('xla'), draw, "queries reading another sequence's documents",
                     keep=read_by_documents(queries, keys))
    query, key, value, _ = draw(0)
    found = np.asarray(jax.jit(kernel('flash'))(query, key, value))
    np.testing.assert_array_equal(found[~np.asarray(real[..., 0, 0])], 0)
    np.testing.assert_array_equal(found[0][queries[0] == 2], 0)


def queries_over_a_context_of_no_tokens_read_nothing_on_flash(without_deterministic_ops):
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


TPU_DOCUMENT_SEEDS = 8
"""The tpu kernels' declared check-1 seeds over another sequence's documents, sized by `seeds_for` paired."""


def test_queries_reading_another_sequences_documents_on_the_tpu_kernels_are_as_exact_as_xla():
    """Packed queries over a context packed by its own documents (`key_segment_ids`) on 'tpu', which
    interprets off a TPU: the output and the three gradients on par with xla's over drawn inputs (check
    1, the truth the reference kernel's float64 run), a document of the queries' the context lacks among
    them; xla's sums are
    the TPU's, fp32 (`FP32_SUMS`), and a CPU runs them on its reference path."""
    queries = jnp.asarray([[1] * 40 + [2] * 30 + [3] * 30 + [0] * 28])
    keys = jnp.asarray([[1] * 50 + [3] * 50 + [0] * 28])
    draw = normal_drawn(0x7B0, *[(1, 128, 2, 128)] * 4)

    def attend(implementation):
        def run(q, k, v):
            # xla's sums are the TPU's, fp32, as its kernels' are.
            with jax.default_matmul_precision(FP32_SUMS if implementation == "xla" else None):
                return scaled_dot_product_attention(q, k, v, implementation=implementation,
                                                    segment_ids=queries, key_segment_ids=keys, max_segments=3)
        return run

    attention_on_par(attend("tpu"), attend("xla"), draw, "the tpu kernels over another sequence's documents",
                     keep=read_by_documents(queries, keys), seeds=TPU_DOCUMENT_SEEDS)


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
    """Row b reads its first lengths[b] keys and no other: the output and every gradient on par with the
    call that masks the rest over drawn operands (check 1), truth the reference path's masked call in
    float64, and what a row's keys past its length hold reaches nothing, its own key and value gradients
    there included."""
    lengths = jnp.asarray([9, 5, 1], jnp.int32)

    def draw(seed):
        return tuple(x.astype(jnp.float32) for x in qkv((3, 9, 2, 16), seed=seed))

    def call(kernel, **options):
        return as_given(lambda query, key, value: tuple(key_length_call(kernel, query, key, value, lengths,
                                                                        causal=causal, **options)))

    assert_as_exact(call(implementation), call(implementation, as_mask=True), draw,
                    f"key lengths on {implementation}, {'causal' if causal else 'full'}",
                    seeds=KEY_LENGTH_SEEDS, truth_fn=call("reference", as_mask=True))
    query, key, value = draw(0)
    by_length = key_length_call(implementation, query, key, value, lengths, causal=causal)
    past = jnp.arange(9)[None, :, None, None] >= lengths[:, None, None, None]
    moved = key_length_call(implementation, query, jnp.where(past, 3.0, key),
                            jnp.where(past, -2.0, value), lengths, causal=causal)
    np.testing.assert_allclose(moved[0], by_length[0], atol=1e-6, rtol=1e-6)
    for gradient in by_length[2:]:
        assert not np.asarray(gradient)[np.asarray(past)[:, :, 0, 0]].any()


@pytest.mark.parametrize("as_mask", [False, True], ids=["lengths", "mask"])
@pytest.mark.parametrize("q_len, kv_len, causal", [
    (1024, 1024 + 77, False),  # the image queries of a joint call over padded text
    (77, 77, True),            # its text queries, causal over the text alone
    (78, 78, True),            # an even length, which the kernel takes unpadded
])
def cudnn_takes_key_lengths_and_agrees_with_xla(q_len, kv_len, causal, as_mask,
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


def single_query_cudnn_decode_matches_float64_attention_and_gradients(without_deterministic_ops):
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


FORWARD_MODE_SEEDS = 8
"""Forward-mode reference attention's declared check-1 seeds, sized by `seeds_for` paired."""


@pytest.mark.skipif(jax.default_backend() != 'cpu', reason="the float64 sum is a CPU's")
def test_cpu_fp32_attention_summed_in_float64_takes_forward_mode():
    """fp32 reference attention over 300 keys, whose products a CPU sums in float64
    (`dew.nn.precision.accumulated`), as a consistency model's JVP runs it under
    `forward_mode_attention` (sCM, rCM): its value, tangent and three gradients on par with
    jax.nn.dot_product_attention's over drawn inputs (check 1, the truth its own float64 run)."""
    from dew.nn.attention import forward_mode_attention

    draw = normal_drawn(0xF0D, *[(1, 300, 2, 16)] * 7)

    def reference(q, k, v):
        return scaled_dot_product_attention(q, k, v, implementation="reference", causal=True)

    def plain(q, k, v):
        return jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="xla")

    def derivatives(attend):
        def run(*arrays):
            primals, tangents, cotangent = arrays[:3], arrays[3:6], arrays[6]
            with forward_mode_attention():
                out, tangent = jax.jvp(attend, primals, tangents)
            _, pull = jax.vjp(attend, *primals)
            return {"out": out, "tangent": tangent, "gradients": pull(cotangent)}
        return run

    # The truth the reference path's own float64 run: raw jax.nn takes its softmax in fp32 whatever its
    # operands (src/dew/nn/attention.py), which Dew's reference path does not.
    assert_as_exact(jax.jit(derivatives(reference)), jax.jit(derivatives(plain)), draw,
                    "reference attention in forward mode", seeds=FORWARD_MODE_SEEDS,
                    truth_fn=derivatives(reference), terms=True)


@pytest.mark.parametrize(("dtype", "key_dtype"), [(jnp.float16, jnp.float16), (jnp.float32, jnp.float64),
                                                   (jnp.float64, jnp.float64)],
                         ids=["float16", "float32-over-float64", "float64"])
def test_a_call_jax_nn_cannot_take_on_a_cpu_runs_the_reference_path(dtype, key_dtype, monkeypatch):
    """A call named onto jax.nn's xla kernel that it cannot take on a CPU, a float16 one (XLA's CPU compiler
    has no F16_F16_F32 dot), a float32 query over float64 keys (jax.nn takes one dtype) or a float64 one
    (jax.nn takes its softmax in float32, a float64 run's attention in float32 precision), hands off
    (`attention_kernels.XLA_SUPPORTS`): it compiles and runs, its output the reference path's bit for
    bit. (A process's own CPU arithmetic: the suite's wide one hands every xla call on, `XLA_SUPPORTS`.)"""
    monkeypatch.delenv(WIDE_CPU_ARITHMETIC, raising=False)
    with jax.enable_x64(new_val=True):
        query, key, value = (jax.random.normal(jax.random.key(seed), (1, 16, 2, 8), kind)
                             for seed, kind in zip(range(3), (dtype, key_dtype, key_dtype), strict=True))
        named, reference = (jax.jit(functools.partial(scaled_dot_product_attention, implementation=name,
                                                      causal=True))(query, key, value)
                            for name in ("xla", "reference"))
    np.testing.assert_array_equal(np.asarray(named), np.asarray(reference))


def test_a_layer_that_builds_its_mask_is_on_par_with_the_reference_path():
    """A layer whose rows are padded on the left builds its mask and runs the kernel a materialized mask
    takes (`kernel_for_materialized_mask`): 'auto' on par with the reference path over the real tokens
    (check 1), the reference model in float64 the truth."""
    from dew.nn.backbones.causal_transformer import CausalTransformer

    valid = jnp.arange(300)[None] >= jnp.asarray([[0], [7]])

    def model(implementation, dtype=jnp.float32):
        return CausalTransformer(vocab_size=64, emb_features=32, num_layers=1, num_heads=2, mlp_features=48,
                                 max_seq_len=320, attention_impl=implementation, dtype=dtype)

    def draw(seed):
        tokens = jnp.asarray(np.random.default_rng(seed).integers(1, 64, (2, 300)), jnp.int32)
        return nn.unbox(model('auto').init(jax.random.key(seed), tokens)), tokens

    def applied(implementation, dtype=jnp.float32):
        def apply(params, tokens):
            out = model(implementation, dtype).apply(params, tokens, attention_mask=valid)
            return jnp.where(valid[..., None], out, 0)
        return apply

    assert_as_exact(jax.jit(applied('auto')), jax.jit(applied('reference')), draw,
                    "a padded layer's built mask", seeds=8, truth_fn=applied('reference', jnp.float64))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
@pytest.mark.parametrize("forced", [True, False])
@pytest.mark.parametrize("dropout_rate", [0.0, 0.25])
def test_the_reference_path_is_flax_attention_op_for_op(dtype, forced, dropout_rate):
    """Without sinks the reference path (`reference_attention`) is flax's `dot_product_attention` handed the
    key heads repeated over their group and its probability product (`weighted_values`), on par with it from
    float64 over drawn operands (check 1): in each compute dtype, the softmax forced to fp32 or not, an fp32
    bias wider than half-precision logits and a causal mask. Its probability draw is Dew's own (row 1521:
    the keep bits every route draws from one rng, `template.dropout_keep`, so a route never changes which
    weights drop) and its multiplier Dew's (row 1533: 1 / (1 - rate) rounded once in at least fp32), applied
    where flax applies its own: flax's weights (`dot_product_attention_weights`) under Dew's keep mask and
    multiplier, then the value product. Both sum their fp32 products as XLA does (`FP32_SUMS`), the
    reference path's own float64 sums on a CPU being the policy's (tests/test_cpu_accumulation.py)."""
    def draw(seed):
        keys = jax.random.split(jax.random.key(seed), 4)
        query = jax.random.normal(keys[0], (2, 6, 4, 8), dtype)
        key, value = (jax.random.normal(drawn, (2, 6, 2, 8), dtype) for drawn in keys[1:3])
        return query, key, value, jax.random.normal(keys[3], (2, 4, 6, 6), jnp.float32), np.int32(seed)

    def dropped(seed):
        return jax.random.fold_in(jax.random.key(5), seed)

    def ours(query, key, value, bias, seed):
        with jax.default_matmul_precision(FP32_SUMS):
            return scaled_dot_product_attention(
                query, key, value, implementation="reference", causal=True, bias=bias,
                force_fp32_for_softmax=forced, dropout_rate=dropout_rate, dropout_rng=dropped(seed),
                deterministic=not dropout_rate)

    def flax(*, softmax_in_fp32, precision=FP32_SUMS):
        def run(query, key, value, bias, seed):
            return flax_attention(query, key, value, bias, seed, softmax_in_fp32=softmax_in_fp32,
                                  precision=precision)
        return run

    def flax_attention(query, key, value, bias, seed, *, softmax_in_fp32, precision):
        with jax.default_matmul_precision(precision):
            weights = nn.dot_product_attention_weights(
                query, jnp.repeat(key, 2, axis=2), bias=bias,
                mask=jnp.tril(jnp.ones((6, 6), bool))[None, None], force_fp32_for_softmax=softmax_in_fp32,
                einsum=jnp.einsum)
            if dropout_rate:
                wide = at_least_fp32(query.dtype)
                keep = template.dropout_keep(template.dropout_seed(dropped(seed)), dropout_rate,
                                             weights.shape)
                scale = jnp.asarray(1.0 / (1.0 - dropout_rate), wide)
                weights = (weights.astype(wide) * jnp.where(keep, scale, 0)).astype(query.dtype)
            return weighted_values('...hqk,...khd->...qhd', weights, jnp.repeat(value, 2, axis=2))

    # Two programs, so on par from float64 over the draws (check 1), not bit for bit (bible §6).
    # The truth's softmax and products in float64: flax's forced softmax would narrow it to fp32, and so
    # would the measured arms' fp32-sums preset its products.
    assert_as_exact(jax.jit(ours), jax.jit(flax(softmax_in_fp32=forced)), draw,
                    f"the reference path against flax, {jnp.dtype(dtype).name}, forced {forced}, "
                    f"rate {dropout_rate}", seeds=FLAX_SEEDS,
                    truth_fn=flax(softmax_in_fp32=False, precision="highest"))


@pytest.mark.skipif(jax.default_backend() != 'cpu', reason="'auto' takes jax.nn's xla kernel on a CPU")
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("implementation", ["auto", "xla"])
def test_an_attention_over_one_key_moves_no_query_or_key(implementation, dtype, monkeypatch):
    """A query that reads one key attends it whatever the scores, so its query's and its key's gradients are
    exactly zero, as the softmax's own derivative gives them and torch's does: through jax.nn's xla kernel
    too, jitted. Its softmax differentiates `exp(x - max) / sum` by default, whose g / e - g e / e^2 is zero
    only where that key's e is exactly 1, and XLA:CPU contracts the scaled logit and the subtraction of the
    max into one FMA, which leaves e at 1 +- a rounding of the logit (2e-7 of gradient in float32, run op by
    op there is none); `_fused_attention` takes jax's textbook derivative (`jax.softmax_custom_jvp`). (A
    process's own CPU arithmetic: the suite's wide one hands every xla call on, `XLA_SUPPORTS`.)"""
    monkeypatch.delenv(WIDE_CPU_ARITHMETIC, raising=False)
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
    params = nn.unbox(module.init(jax.random.key(0), queries, context))
    lengths = jnp.asarray([7, 4, 1])
    prefix = (jnp.arange(7)[None, :] < lengths[:, None])[:, None, None, :]
    np.testing.assert_array_equal(module.apply(params, queries, context, key_value_seq_lengths=lengths),
                                  module.apply(params, queries, context, mask=prefix))


ALL_TRUE_SEEDS = 8
"""An all-True mask's declared check-1 seeds against the unmasked call, sized by `seeds_for` paired."""


@pytest.mark.parametrize("implementation", ["reference", "xla"])
def test_normal_attention_reads_only_the_keys_its_mask_keeps(implementation):
    """A key the mask drops leaves the output bitwise unchanged when its value
    moves, which the unmasked call does not. An all-True mask computes the
    unmasked attention, but not always bitwise: XLA:GPU fuses the masked
    softmax differently (2.4e-7 apart on an A100), so it is on par with the
    unmasked call over drawn tokens (check 1, the truth its float64 run)."""
    module = NormalAttention(16, heads=2, dim_head=8, attention_impl=implementation)
    rng = np.random.default_rng(0)
    queries = jnp.asarray(rng.normal(size=(2, 5, 16)), jnp.float32)
    context = jnp.asarray(rng.normal(size=(2, 7, 16)), jnp.float32)
    moved = context.at[:, 3].add(10.0)
    params = nn.unbox(module.init(jax.random.key(0), queries, context))
    kept = jnp.ones((2, 1, 5, 7), bool)
    dropped = kept.at[..., 3].set(False)
    np.testing.assert_array_equal(module.apply(params, queries, context, mask=dropped),
                                  module.apply(params, queries, moved, mask=dropped))
    assert not np.array_equal(module.apply(params, queries, context), module.apply(params, queries, moved))

    def draw(seed):
        streams = np.random.default_rng([0x5A5, seed])
        return tuple(jnp.asarray(streams.normal(size=(2, length, 16)), jnp.float32) for length in (5, 7))

    assert_as_exact(lambda queries, context: module.apply(params, queries, context, mask=kept),
                    lambda queries, context: module.apply(params, queries, context), draw,
                    f"{implementation} attention under an all-True mask", seeds=ALL_TRUE_SEEDS,
                    truth_fn=lambda queries, context: module.apply(widened(params), queries, context))


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


@pytest.mark.parametrize("kernel", ["flash", "triton", "cudnn", "xla", "template"])
@pytest.mark.parametrize("call", eligibility_cases(),
                         ids=lambda call: "-".join(f"{name}={value}" for name, value in call.items()))
def a_kernels_requirements_are_what_the_kernel_takes_on_this_gpu(kernel, call,
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
        with pytest.raises(KernelRefused):
            run()
    else:
        assert np.all(np.isfinite(run()))
