"""Which kernel the 'auto' rule reaches for, and the XLA flags a run passes
to the backend.

'auto' asks two predicates per trace: `cudnn_runs` for the fused cudnn kernel
and then `tpu_runs` for the pallas splash kernel. Both read the backend, so
both selections are testable anywhere by monkeypatching `jax.default_backend`
and tracing the call without running it. What actually executes on a CUDA
device skips elsewhere; run that half with
`JAX_PLATFORMS=cuda python -m pytest tests/test_kernels.py`. The splash
kernel's own numbers are tests/test_attention_splash.py.
"""

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference, assert_fp32_reduction_bound, distance

from dew.nn import kernels
from dew.nn.attention import (
    NormalAttention,
    folded_attention,
    scaled_dot_product_attention,
    weighted_values,
    with_key_lengths,
)
from dew.nn.kernels import bf16_dot_runs

pytestmark = pytest.mark.kernel


def on_gpu(test):
    """Mark `test` for a cuda device of sm80 or later, cuDNN's bf16 floor."""
    reason = "needs a cuda device of sm80 or later, cuDNN's bf16 floor"
    return pytest.mark.gpu(reason=reason)(pytest.mark.skipif(not bf16_dot_runs(), reason=reason)(test))


def flash_here():
    """dew_flash_attn where the installed wheel holds code for this GPU (the fact `flash_refusal` reads);
    a GPU it was not built for skips by name, where 'auto' runs another kernel."""
    flash = pytest.importorskip("dew_flash_attn")
    from dew.nn.attention import flash_architectures, flash_built_for
    generation = kernels.device_generation()
    architectures = flash_architectures() or ()
    if not flash_built_for(generation, architectures):
        pytest.skip(f"the installed dew_flash_attn holds code for {', '.join(architectures)}, "
                    f"not {generation}")
    return flash


def qkv(shape, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    return tuple(jax.random.normal(key, shape, jnp.bfloat16) for key in keys)


def value_and_grads(implementation, query, key, value, **kwargs):
    def loss(q, k, v):
        out = scaled_dot_product_attention(q, k, v, implementation=implementation, **kwargs)
        return jnp.sum(out.astype(jnp.float32) ** 2), out

    (_, out), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))(
        query, key, value)
    return [np.asarray(x, np.float32) for x in (out, *grads)]


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
@pytest.mark.parametrize("q_len, kv_len, causal", [
    (1024, 77, False),   # every cross-attention over CLIP's 77 text tokens
    (9, 7, False),       # short enough that one attended pad key would move an eighth of the mass
    (333, 333, True),    # the concatenated text-plus-image sequence, causal
])
def test_cudnn_trains_odd_lengths_and_agrees_with_xla(q_len, kv_len, causal,
                                                      without_deterministic_ops):
    """cudnn's kernel has no backward pass for an odd length; the padded call
    trains, and what it computes for the real rows is what the xla kernel
    computes: within two bf16 ulps of the output scale, forward and backward,
    which is also how far the two kernels sit apart at an even length. A pad
    key left unmasked shifts every output by 1/(kv+1) of the value mass, 12%
    at 7 keys, and a pad query row left in the output changes its shape."""
    query, _, _ = qkv((2, q_len, 4, 64))
    _, key, value = qkv((2, kv_len, 4, 64), seed=1)

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


def float64_attention(query, key, value, causal):
    """Attention of `[S, H, D]` arrays as float64 math, key heads repeated over their group."""
    group = query.shape[1] // key.shape[1]
    key, value = jnp.repeat(key, group, axis=1), jnp.repeat(value, group, axis=1)
    logits = jnp.einsum('qhd,khd->hqk', query, key) / np.sqrt(query.shape[-1])
    if causal:
        logits = jnp.where(jnp.tril(jnp.ones(logits.shape[1:], bool)), logits, -jnp.inf)
    return jnp.einsum('hqk,khd->qhd', jax.nn.softmax(logits, axis=-1), value)


def output_and_gradients(attend, cotangent, *arrays):
    """`attend`'s output and the gradients of its inputs under `cotangent`, as float64 host arrays."""
    out, pull = jax.vjp(attend, *arrays)
    return [np.asarray(x, np.float64) for x in (out, *pull(cotangent))]


@on_gpu
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [True, False])
def test_packed_sequences_attend_within_themselves_as_far_from_float64_as_one_at_a_time(
        head_dim, causal, without_deterministic_ops):
    """Documents packed back to back, one a single token, with an empty sequence between two: each
    document's output and its three input gradients lie within tests/reference_error.py's rule of
    FlashAttention-2's own on that document alone, both measured from float64. Not bitwise: on one
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

    packed = output_and_gradients(
        lambda q, k, v: flash.flash_mha_varlen(q, k, v, starts, max(lengths), is_causal=causal),
        cotangent, query, key, value)
    for begin, end in itertools.pairwise(starts):
        if begin == end:
            continue
        pieces = (query[begin:end], key[begin:end], value[begin:end])
        dense = output_and_gradients(
            lambda q, k, v: flash.flash_mha(q[None], k[None], v[None], is_causal=causal)[0],
            cotangent[begin:end], *pieces)
        with jax.enable_x64(new_val=True):
            wide = [jnp.asarray(np.asarray(x, np.float64)) for x in (*pieces, cotangent[begin:end])]
            truth = output_and_gradients(functools.partial(float64_attention, causal=causal), wide[3],
                                         *wide[:3])
        for name, got, theirs, want in zip(("o", "dq", "dk", "dv"), (x[begin:end] for x in packed), dense,
                                           truth, strict=True):
            if distance(theirs, want) == 0:  # a one-token document's attention is exact either way
                np.testing.assert_array_equal(got, theirs)
            else:
                assert_as_exact_as_the_reference(got, theirs, want, f"{name} of the document at {begin}")


@on_gpu
@pytest.mark.parametrize("head_dim", [128, 256])
def test_flash_is_as_exact_as_the_xla_kernel_from_float64(head_dim, without_deterministic_ops):
    """At Qwen3.5's grouped causal heads, 8 queries over 2 keys, FlashAttention's output and its
    three input gradients lie within tests/reference_error.py's rule of the xla kernel's, both
    measured from float64; on Hopper that holds FlashAttention-3 to it."""
    flash_here()
    if head_dim > 192 and kernels.device_generation() in ('sm86', 'sm87', 'sm89'):
        pytest.skip("FlashAttention-2's backward takes heads over 192 wide only on sm80, sm90 and sm1xx")
    query, _, _ = qkv((1, 1024, 8, head_dim))
    _, key, value = qkv((1, 1024, 2, head_dim), seed=1)
    cotangent = qkv((1, 1024, 8, head_dim), seed=2)[0]

    def kernel(implementation):
        return jax.jit(lambda q, k, v: scaled_dot_product_attention(
            q, k, v, implementation=implementation, causal=True))

    fused = output_and_gradients(kernel('flash'), cotangent, query, key, value)
    reference = output_and_gradients(kernel('xla'), cotangent, query, key, value)
    with jax.enable_x64(new_val=True):
        wide = [jnp.asarray(np.asarray(x[0], np.float64)) for x in (query, key, value, cotangent)]
        truth = output_and_gradients(functools.partial(float64_attention, causal=True), wide[3],
                                     *wide[:3])
    for name, got, theirs, want in zip(("o", "dq", "dk", "dv"), fused, reference, truth, strict=True):
        assert_as_exact_as_the_reference(got[0], theirs[0], want, name)


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
        np.testing.assert_allclose(got, want, atol=1e-6, rtol=1e-6)
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


def test_auto_computes_what_xla_computes_under_deterministic_ops(monkeypatch):
    """What the selection did is in the output: on a gpu backend 'auto' would
    fuse this call, and under the flag it returns the xla kernel's numbers
    bit for bit."""
    query, key, value = qkv((1, 8, 2, 64))
    expected = scaled_dot_product_attention(query, key, value, implementation='xla')
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
    routed = scaled_dot_product_attention(query, key, value, implementation='auto')
    assert np.array_equal(np.asarray(routed, np.float32), np.asarray(expected, np.float32))


@pytest.mark.skipif(jax.default_backend() != 'cpu', reason="the blocked sum is XLA:CPU's")
def test_cpu_fp32_attention_rounds_closer_to_float64_in_blocks():
    """Over 2048 keys of cross-attention-like inputs, Dew's fp32 attention on
    XLA:CPU is at most three quarters of jax.nn's xla distance from float64
    (measured 0.55 of it; torch's SDPA sits between, docs/performance.md),
    and the blocked product's gradients are the plain product's."""
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

    weights = jax.nn.softmax(jnp.asarray(logits, jnp.float32))
    blocked = jax.grad(lambda w, v: jnp.sum(weighted_values('...hqk,...khd->...qhd', w, v) ** 2), (0, 1))
    out = weighted_values('...hqk,...khd->...qhd', weights, value)
    cotangent = 2 * out
    want = (jnp.einsum('...qhd,...khd->...hqk', cotangent, value),
            jnp.einsum('...hqk,...qhd->...khd', weights, cotangent))
    for got, expected in zip(blocked(weights, value), want, strict=True):
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("heads", [1, 2])
def test_folded_attention_is_jax_nn_attention(dtype, heads):
    """Every query head against every (key, head) pair, the other heads'
    products thrown away, computes jax.nn's xla attention. In fp32 it only
    reorders three sums, the logits' over the width, the softmax's over the
    keys and the value product's over keys times heads (exact zeros between),
    so it sits within their summed chains of jax.nn's
    (`assert_fp32_reduction_bound`), the logits' rounding carried through the
    softmax at twice the largest logit magnitude. In bf16 it is within
    tests/reference_error.py's rule of jax.nn against float64. Rows read 1,
    17 and all 40 keys."""
    rng = np.random.default_rng(0)
    query = rng.normal(size=(3, 4, heads, 256))
    key, value = rng.normal(size=(2, 3, 40, heads, 256)) * np.array([0.3, 1.0])[:, None, None, None, None]
    lengths = jnp.asarray([1, 17, 40], jnp.int32)
    q, k, v = (jnp.asarray(x, dtype) for x in (query, key, value))
    folded = folded_attention(q, k, v, with_key_lengths(None, lengths, 40))
    plain = jax.nn.dot_product_attention(q, k, v, key_value_seq_lengths=lengths,
                                         implementation='xla')
    assert folded.shape == plain.shape and folded.dtype == plain.dtype

    q, k, v = (np.asarray(x, np.float64) for x in (q, k, v))
    read = np.arange(40)[None, None, None, :] < np.asarray(lengths)[:, None, None, None]
    logits = np.where(read, np.einsum('btnd,bsnd->bnts', q, k) / 16.0, -np.inf)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    truth = np.einsum('bnts,bsnd->btnd', probs, v)
    if dtype == jnp.float32:
        spread = np.where(read, np.einsum('btnd,bsnd->bnts', np.abs(q), np.abs(k)) / 16.0, 0).max(-1)
        carried = 1 + 2 * spread.transpose(0, 2, 1)[..., None]
        magnitudes = np.einsum('bnts,bsnd->btnd', probs, np.abs(v)) * carried
        assert_fp32_reduction_bound(folded, plain, magnitudes, terms=256 + 40 + 40 * heads)
        return
    assert_as_exact_as_the_reference(np.asarray(folded, np.float32), np.asarray(plain, np.float32), truth,
                                     "folded")


@pytest.mark.gpu(reason="the kernel runs on CUDA")
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_the_decode_kernel_reads_each_row_to_its_length(dtype):
    """`dew.nn.kernels.decode_attention` reads each row's keys up to its
    length, block by block, and is within tests/reference_error.py's rule of
    jax.nn's xla attention against float64: rows reading 1 key (an idle
    slot), a block's worth, one past it, and the whole capacity, at
    Qwen3.5-0.8B's widths (4 query positions over 2 key heads of 256)."""
    from dew.nn.kernels import decode_attention

    rng = np.random.default_rng(0)
    capacity = 4 * decode_attention.BLOCK
    lengths = jnp.asarray([1, decode_attention.BLOCK, decode_attention.BLOCK + 1, 77, capacity], jnp.int32)
    rows = lengths.shape[0]
    query = jnp.asarray(rng.normal(size=(rows, 4, 2, 256)), dtype)
    key = jnp.asarray(rng.normal(size=(rows, capacity, 2, 256)) * 0.3, dtype)
    value = jnp.asarray(rng.normal(size=(rows, capacity, 2, 256)), dtype)
    assert decode_attention.fits(query, key)
    out = jax.jit(decode_attention.attend)(query, key, value, lengths)
    plain = jax.nn.dot_product_attention(query, key, value, key_value_seq_lengths=lengths,
                                         implementation='xla')
    assert out.shape == plain.shape and out.dtype == plain.dtype
    q, k, v = (np.asarray(x, np.float64) for x in (query, key, value))
    read = np.arange(capacity)[None, None, None, :] < np.asarray(lengths)[:, None, None, None]
    logits = np.where(read, np.einsum('btnd,bsnd->bnts', q, k) / 16.0, -np.inf)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    truth = np.einsum('bnts,bsnd->btnd', probs / probs.sum(-1, keepdims=True), v)
    assert_as_exact_as_the_reference(np.asarray(out, np.float32), np.asarray(plain, np.float32), truth,
                                     "decode kernel")


@pytest.mark.gpu(reason="the kernel runs on CUDA")
@pytest.mark.parametrize("page_size", [16, 32])
def test_the_paged_decode_kernel_reads_each_row_through_its_table(page_size):
    """`decode_attention.attend_paged` reads a row's pages through its table,
    shuffled and shared between rows, to the row's length, its query heads
    sharing key heads as Qwen3-0.6B's do (16 over 8 of 128), and is within
    tests/reference_error.py's rule of jax.nn's xla attention over the
    gathered rows against float64: rows reading 1 key, a page, one past a
    block, and every page."""
    from dew.nn.kernels import decode_attention

    rng = np.random.default_rng(0)
    per_row, kv_heads, heads, width = 6, 8, 16, 128
    capacity = per_row * page_size
    lengths = jnp.asarray([1, page_size, decode_attention.BLOCK + 1, capacity], jnp.int32)
    rows = lengths.shape[0]
    count = rows * per_row + 3
    table = jnp.asarray(rng.permutation(count)[:rows * per_row].reshape(rows, per_row), jnp.int32)
    table = table.at[1, 0].set(table[0, 0])
    query = jnp.asarray(rng.normal(size=(rows, heads, width)), jnp.bfloat16)
    key_pages = jnp.asarray(rng.normal(size=(kv_heads, count, page_size, width)) * 0.3, jnp.bfloat16)
    value_pages = jnp.asarray(rng.normal(size=(kv_heads, count, page_size, width)), jnp.bfloat16)
    assert decode_attention.fits_paged(query, key_pages)
    out = jax.jit(decode_attention.attend_paged)(query, key_pages, value_pages, table, lengths)
    key, value = (jnp.moveaxis(pages[:, table].reshape(kv_heads, rows, capacity, width), 0, 2)
                  for pages in (key_pages, value_pages))
    plain = jax.nn.dot_product_attention(query[:, None], key, value, key_value_seq_lengths=lengths,
                                         implementation='xla')[:, 0]
    assert out.shape == plain.shape and out.dtype == plain.dtype
    q = np.asarray(query, np.float64).reshape(rows, kv_heads, heads // kv_heads, width)
    k, v = np.asarray(key, np.float64), np.asarray(value, np.float64)
    read = np.arange(capacity)[None, None, None, :] < np.asarray(lengths)[:, None, None, None]
    logits = np.where(read, np.einsum('bngd,bsnd->bngs', q, k) / np.sqrt(width), -np.inf)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    truth = np.einsum('bngs,bsnd->bngd', probs / probs.sum(-1, keepdims=True), v).reshape(rows, heads, width)
    assert_as_exact_as_the_reference(np.asarray(out, np.float32), np.asarray(plain, np.float32), truth,
                                     "paged decode kernel")


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
