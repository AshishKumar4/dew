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

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference, assert_fp32_reduction_bound, distance

from dew.nn import kernels
from dew.nn.attention import (
    SPLASH_LANES,
    SPLASH_MIN_LENGTH,
    VALUE_BLOCK,
    NormalAttention,
    attention_kernel,
    cudnn_runs,
    folded_attention,
    folds,
    fused_attention,
    resolve_implementation,
    scaled_dot_product_attention,
    tpu_runs,
    weighted_values,
)
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.kernels import bf16_dot_runs
from dew.telemetry.devices import apply_xla_flags, deterministic_ops_requested, xla_flag
from dew.training import MeshSpec

on_gpu = pytest.mark.skipif(jax.default_backend() != 'gpu' or not bf16_dot_runs(),
                            reason="needs a cuda device of sm80 or later, cuDNN's bf16 floor")


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
        pytest.importorskip("dew_flash_attn")
    query, key, value = qkv((2, 256, 8, 64))
    tangents = qkv((2, 256, 8, 64), seed=1)

    def attend(name):
        return lambda q, k, v: scaled_dot_product_attention(q, k, v, implementation=name, causal=True)

    def tangent(name):
        with forward_mode_attention():
            return np.asarray(jax.jit(lambda *a: jax.jvp(attend(name), a[:3], a[3:])[1])(
                query, key, value, *tangents), np.float32)

    with pytest.raises(TypeError, match="custom_vjp"):
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


@pytest.mark.parametrize("installed, call, chosen", [
    (True, {}, "triton"),
    (True, {"causal": True}, "triton"),
    (True, {"sliding_window": 64, "causal": True}, "cudnn"),
    (True, {"sliding_window": 64}, "xla"),
    (False, {"sliding_window": 64}, "xla"),
    (True, {"bias": jnp.zeros((1, 1, 128, 128), jnp.bfloat16)}, "cudnn"),
    (True, {"mask": jnp.ones((1, 1, 128, 128), bool)}, "cudnn"),
    (False, {}, "cudnn"),
    (True, {"head_dim": 128}, "cudnn"),
])
def test_auto_sends_a_plain_cudnn_call_to_tokamax_when_it_is_installed(monkeypatch, installed, call,
                                                                        chosen, without_deterministic_ops):
    """Where cudnn's kernel runs, 'auto' takes tokamax's Pallas-Triton kernel
    for heads up to 64 wide and a call with no window, mask or bias, if
    tokamax is installed: the calls it measured faster on
    (`triton_runs`). Anything else stays on cudnn, except a bidirectional
    window, which cudnn cannot take (`window_sides`) and xla does."""
    from importlib import util

    from dew.nn import attention
    monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
    monkeypatch.setattr(attention, 'bf16_dot_runs', lambda: True)
    found = util.find_spec
    # dew_flash_attn, which 'auto' prefers on an A100, is out of this test.
    monkeypatch.setattr(util, 'find_spec', lambda name: object() if name == 'tokamax' and installed
                        else None if name in ('tokamax', 'dew_flash_attn') else found(name))
    query = jnp.zeros((1, 128, 4, call.pop("head_dim", 64)), jnp.bfloat16)
    assert attention.resolve_implementation('auto', query, query, **call) == chosen


@pytest.mark.parametrize("generation, installed, call, chosen", [
    ("sm80", True, {}, "flash"),
    ("sm80", True, {"causal": True}, "flash"),
    ("sm80", True, {"head_dim": 256}, "flash"),
    ("sm80", True, {"causal": True, "keys": 256}, "cudnn"),
    ("sm80", True, {"keys": 256}, "flash"),
    ("sm80", True, {"sliding_window": 64, "causal": True}, "cudnn"),
    ("sm80", True, {"mask": jnp.ones((1, 1, 128, 128), bool)}, "cudnn"),
    ("sm80", True, {"softcap": 30.0}, "xla"),
    ("sm80", True, {"dtype": jnp.float32}, "xla"),
    ("sm80", False, {}, "cudnn"),
    ("sm89", True, {}, "cudnn"),
])
def test_auto_sends_a_plain_call_to_flash_where_it_was_measured(monkeypatch, generation, installed, call,
                                                                chosen, without_deterministic_ops):
    """With dew_flash_attn installed, 'auto' takes FlashAttention-2 where it
    was measured faster than cuDNN, for a bf16 call with no window, mask, bias
    or softcap, heads up to 256 wide and a square causal mask if any."""
    from importlib import util

    from dew.nn import attention
    monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
    monkeypatch.setattr(attention, 'bf16_dot_runs', lambda: True)
    monkeypatch.setattr(kernels.generation, 'device_generation', lambda: generation)
    found = util.find_spec
    monkeypatch.setattr(util, 'find_spec', lambda name: (object() if installed else None)
                        if name == 'dew_flash_attn' else None if name == 'tokamax' else found(name))
    query = jnp.zeros((1, 128, 4, call.pop("head_dim", 128)), call.pop("dtype", jnp.bfloat16))
    key = jnp.zeros((1, call.pop("keys", 128), *query.shape[2:]), query.dtype)
    assert attention.resolve_implementation('auto', query, key, **call) == chosen


@pytest.mark.parametrize("generation", ["sm90", "sm100", "sm75", "cpu"])
def test_flash_refuses_a_device_its_wheel_has_no_code_for(monkeypatch, generation, without_deterministic_ops):
    """The wheel holds sm80 code, which compute capability 8.x runs: an
    explicit 'flash' on any other device is refused by name, before a CUDA
    launch could fail without one."""
    from dew.nn import attention, kernels
    monkeypatch.setattr(kernels.generation, 'device_generation', lambda: generation)
    monkeypatch.setattr(attention, 'device_generation', lambda: generation)
    query = jnp.zeros((1, 128, 4, 64), jnp.bfloat16)
    with pytest.raises(ValueError, match=f"built for sm8x and this device is {generation}"):
        attention.flash_attention(query, query, query, causal=True)


@pytest.mark.parametrize("call, refusal", [
    ({"mask": jnp.ones((1, 1, 128, 128), bool)}, "takes no sinks"),
    ({"dtype": jnp.float32}, "bf16 or fp16"),
    ({"causal": True, "keys": 256}, "as many queries as keys"),
])
def test_flash_refuses_what_it_cannot_take_by_name(call, refusal, without_deterministic_ops):
    """An explicit 'flash' names the mask, fp32 input or causal mask over
    more keys than queries it cannot take, before it imports anything."""
    query = jnp.zeros((1, 128, 4, 64), call.pop("dtype", jnp.bfloat16))
    key = jnp.zeros((1, call.pop("keys", 128), 4, 64), query.dtype)
    arguments = {"bias": None, "mask": None, "causal": False, "sliding_window": None, "softcap": None,
                 "sinks": None, "segment_ids": None, "key_value_seq_lengths": None}
    with pytest.raises(ValueError, match=refusal):
        fused_attention(query, key, key, implementation='flash', **{**arguments, **call})


def test_cudnn_refuses_a_bidirectional_window_by_name(without_deterministic_ops):
    """jax's cuDNN call keeps a window on the left of the query alone, so an
    explicit 'cudnn' names the two-sided window it cannot take instead of
    failing inside jax at trace time."""
    query = jnp.zeros((1, 128, 4, 64), jnp.bfloat16)
    with pytest.raises(ValueError, match="bidirectional"):
        fused_attention(query, query, query, bias=None, mask=None, causal=False, sliding_window=8,
                        implementation='cudnn', softcap=None, sinks=None, segment_ids=None,
                        key_value_seq_lengths=None)


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
    """FlashAttention-2 at the shapes 'auto' sends it: within two bf16 ulps
    of the output scale of the xla kernel, forward and backward, as cudnn and
    triton are. Its backward takes heads over 192 wide only on sm80 and sm90
    ("requires A100/A800 or H100/H800", measured on an L4)."""
    pytest.importorskip("dew_flash_attn")
    if head_dim > 192 and kernels.device_generation() in ('sm86', 'sm87', 'sm89'):
        pytest.skip("FlashAttention-2's backward takes heads over 192 wide only on sm80 and sm90")
    query, _, _ = qkv((2, q_len, heads, head_dim))
    _, key, value = qkv((2, kv_len, kv_heads, head_dim), seed=1)

    fused = value_and_grads('flash', query, key, value, causal=causal)
    reference = value_and_grads('xla', query, key, value, causal=causal)

    for got, want in zip(fused, reference, strict=True):
        assert got.shape == want.shape
        assert np.abs(got - want).max() <= 2 ** -6 * np.abs(want).max()


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


@pytest.mark.parametrize("carried, flags, expected", [
    ("--xla_force_host_platform_device_count=8", "--xla_gpu_autotune_level=4",
     "--xla_force_host_platform_device_count=8 --xla_gpu_autotune_level=4"),
    ("--xla_force_host_platform_device_count=8", None, "--xla_force_host_platform_device_count=8"),
    ("--xla_force_host_platform_device_count=8", "", "--xla_force_host_platform_device_count=8"),
    (None, "--xla_gpu_triton_gemm_any=true", "--xla_gpu_triton_gemm_any=true"),
], ids=["appended", "none", "empty", "unset"])
def test_flags_are_appended_to_what_the_environment_already_carries(monkeypatch, carried, flags, expected):
    """The test suite itself sets a flag, and a run's own flags have to add to
    it, not replace it. No flags leave it alone, and an environment that had
    none takes them as given."""
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    if carried is not None:
        monkeypatch.setenv("XLA_FLAGS", carried)
    apply_xla_flags(flags)
    assert os.environ["XLA_FLAGS"] == expected


@pytest.mark.parametrize("flags, value", [
    ("--xla_gpu_deterministic_ops=true", "true"),
    ("--xla_gpu_deterministic_ops", "true"),
    ("--xla_gpu_deterministic_ops=false", "false"),
    ("--xla_force_host_platform_device_count=8", None),
    ("--xla_gpu_deterministic_ops_level=2", None),
    ("--xla_gpu_deterministic_ops=true --xla_gpu_deterministic_ops=false", "false"),
    ("--xla_gpu_deterministic_ops=false --xla_gpu_autotune_level=4 "
     "--xla_gpu_deterministic_ops=true", "true"),
])
def test_xla_flag_reads_the_value_the_run_asked_for(monkeypatch, flags, value):
    """A bare flag is on, a repeated one keeps its last value, and a longer
    flag name that begins the same way is another flag."""
    monkeypatch.setenv("XLA_FLAGS", flags)
    assert xla_flag("xla_gpu_deterministic_ops") == value


def test_xla_flag_is_none_when_the_variable_is_unset(monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    assert xla_flag("xla_gpu_deterministic_ops") is None


@pytest.mark.parametrize("flags, requested", [
    ("--xla_gpu_deterministic_ops=true", True),
    ("--xla_gpu_deterministic_ops=TRUE", True),
    ("--xla_gpu_deterministic_ops=1", True),
    ("--xla_gpu_deterministic_ops", True),
    ("--xla_gpu_deterministic_ops=false", False),
    ("--xla_gpu_deterministic_ops=0", False),
    ("--xla_force_host_platform_device_count=8", False),
])
def test_deterministic_ops_requested_reads_the_flag(monkeypatch, flags, requested):
    monkeypatch.setenv("XLA_FLAGS", flags)
    assert deterministic_ops_requested() is requested


def test_deterministic_ops_keep_the_auto_rule_off_cudnn(monkeypatch):
    """Everything else the predicate reads holds (a gpu backend, bf16 inputs,
    a 64-wide head, no softcap), so the flag is what decides."""
    query, _, _ = qkv((1, 8, 2, 64))
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(kernels.generation, "device_generation", lambda: "sm89")
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    assert cudnn_runs(query)
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=8 "
                                    "--xla_gpu_deterministic_ops=true")
    assert not cudnn_runs(query)


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


def cudnn_shape(query, key, value):
    """Trace an explicit cudnn call without running it, so what the selection
    decided is readable on a host that has no cudnn kernel to run."""
    return jax.eval_shape(
        lambda q, k, v: scaled_dot_product_attention(q, k, v, implementation='cudnn'),
        query, key, value)


def test_an_explicit_cudnn_call_is_refused_under_deterministic_ops(monkeypatch):
    """XLA cannot execute cudnn attention's backward pass under the flag
    (openxla/xla#46500), so the request is refused by name while the call is
    traced, not at the training step that would have crashed."""
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
    with pytest.raises(ValueError, match="xla_gpu_deterministic_ops"):
        cudnn_shape(*qkv((1, 8, 2, 64)))


@pytest.mark.skipif(jax.default_backend() != 'gpu',
                    reason="jax traces a cudnn call only where cuDNN is installed")
def test_an_explicit_cudnn_call_stands_without_the_flag(without_deterministic_ops):
    """The refusal belongs to the flag, not to the implementation: the same
    call traces to its output shape when the run asked for nothing."""
    assert cudnn_shape(*qkv((1, 8, 2, 64))).shape == (1, 8, 2, 64)


@pytest.fixture
def tpu_backend(monkeypatch):
    """A tpu backend for the selection rules. The pallas kernels lower at
    compile time, not at trace time, so a traced call under this fixture
    names the kernel the rule picked without needing the device that runs
    it, the way `cudnn_shape` reads the cudnn selection off a host with no
    cudnn."""
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")


def kernel_chosen(query, key, value, **kwargs):
    """The names of the kernels one traced dispatch holds."""
    text = jax.make_jaxpr(
        lambda q, k, v: attention_kernel(q, k, v, **kwargs))(query, key, value)
    printed = text.pretty_print(use_color=False)
    return {name for name in ('splash', 'flash_attention') if name in printed}


@pytest.mark.parametrize("structure", [
    {}, {"causal": True}, {"sliding_window": 128},
])
def test_auto_picks_the_splash_kernel_on_a_tpu_backend(tpu_backend, structure):
    """The shapes a decoder trains at meet the kernel's constraints, so a run
    that asked for nothing gets the block-sparse kernel instead of XLA's
    attention, which is the whole point of the rule."""
    query, key, value = qkv((2, 512, 8, 128))
    assert tpu_runs(query, key, **structure)
    assert kernel_chosen(query, key, value, implementation='auto', **structure) == {'splash'}


@pytest.mark.parametrize("shape, reason", [
    ((2, 200, 8, 128), "a length the mask blocking has no block size for"),
    ((2, 1, 8, 128), "a decode step's single query row"),
])
def test_auto_stays_on_xla_where_splash_cannot_tile_the_sequence(tpu_backend, shape, reason):
    """Every length splash takes is a multiple of 128; the lengths that are
    not fall back rather than failing at compile time."""
    query, key, value = qkv(shape)
    assert not tpu_runs(query, key), reason
    assert kernel_chosen(query, key, value, implementation='auto', causal=True) == set()


def test_auto_stays_on_xla_where_the_call_carries_a_bias(tpu_backend):
    """Splash has no bias argument at all, so T5's relative position table
    keeps XLA's attention under 'auto' and the older flash kernel under an
    explicit 'tpu'."""
    query, key, value = qkv((2, 512, 8, 128))
    bias = jnp.zeros((2, 8, 512, 512), jnp.bfloat16)
    assert not tpu_runs(query, key, bias=bias)
    assert kernel_chosen(query, key, value, implementation='auto', bias=bias) == set()
    assert kernel_chosen(query, key, value, implementation='tpu',
                         bias=bias) == {'flash_attention'}


def test_auto_stays_on_xla_where_the_mask_is_a_value_of_the_trace(tpu_backend):
    """A decode mask over cache slots exists only inside the trace, and the
    descriptor is built while the executable is."""
    query, key, value = qkv((2, 512, 8, 128))

    def traced(q, k, v, mask):
        assert not tpu_runs(q, k, mask=mask)
        return attention_kernel(q, k, v, implementation='auto', mask=mask)

    printed = jax.make_jaxpr(traced)(
        query, key, value, jnp.ones((1, 1, 512, 512), bool)).pretty_print(use_color=False)
    assert 'splash' not in printed


@pytest.mark.parametrize("dtype, taken", [
    (jnp.bfloat16, True), (jnp.float32, True), (jnp.float16, False),
])
def test_tpu_runs_reads_the_dtypes_a_tpu_matmul_takes(tpu_backend, dtype, taken):
    """fp16 has no MXU path, so it is the one dtype the rule turns down."""
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    query, key = (jax.random.normal(k, (2, 512, 8, 128), dtype) for k in keys)
    assert tpu_runs(query, key) is taken


def test_tpu_runs_needs_the_tpu_backend(monkeypatch):
    """Everything else the predicate reads holds; only the backend says no,
    because the interpreter that runs splash elsewhere is far slower than the
    XLA attention 'auto' falls back to."""
    query, key, _ = qkv((2, SPLASH_MIN_LENGTH, 8, 128))
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert not tpu_runs(query, key)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    assert tpu_runs(query, key)


@pytest.mark.mesh
def test_auto_stays_on_xla_where_the_mesh_splits_the_sequence(tpu_backend):
    """Outside a shard_map the sequence axis is automatic, which is where
    the gather exchange runs: GSPMD constraints around a whole-sequence
    kernel, and a pallas call with no partitioning rule would have the
    queries gathered back to serve it, undoing the split. The all-to-all
    exchange calls the kernel with the sequence axis manual, and splash
    takes those calls (tests/test_sequence_parallel.py)."""
    query, key, _ = qkv((8, 512, 8, 128))
    assert tpu_runs(query, key)
    with jax.set_mesh(MeshSpec(fsdp=4, sequence=2).build()):
        assert not tpu_runs(query, key)


def test_auto_stays_on_xla_below_the_length_splash_measured_faster(tpu_backend):
    """On a v6e XLA's attention trains faster than splash below
    `SPLASH_MIN_LENGTH` keys, so 'auto' keeps a sequence one lane short of it
    on xla, while an explicit 'tpu' still gets splash."""
    query, key, value = qkv((2, SPLASH_MIN_LENGTH - SPLASH_LANES, 8, 128))
    assert not tpu_runs(query, key, causal=True)
    assert kernel_chosen(query, key, value, implementation='auto', causal=True) == set()
    assert kernel_chosen(query, key, value, implementation='tpu', causal=True) == {'splash'}
    query, key, value = qkv((2, SPLASH_MIN_LENGTH, 8, 128))
    assert kernel_chosen(query, key, value, implementation='auto', causal=True) == {'splash'}


@pytest.mark.parametrize("extra", [
    {"softcap": 30.0},
    {"sinks": jnp.zeros((8,))},
    {"segment_ids": jnp.ones((2, 512), jnp.int32)},
])
def test_softcap_sinks_and_packed_documents_reach_splash(tpu_backend, extra):
    """Gemma's softcap, GPT-OSS's sinks and a packed batch's segment ids are
    arguments of the splash kernel, so 'auto' sends them there rather than
    to the XLA paths that hold the whole [B, H, S, S] logits."""
    query, key, value = qkv((2, 512, 8, 128))
    assert kernel_chosen(query, key, value, implementation='auto', causal=True,
                         **extra) == {'splash'}


def test_a_packed_model_hands_splash_its_segment_ids(tpu_backend):
    """The mixer passes a packed batch's ids down instead of building the
    [B, 1, S, S] document mask, which as a value of the trace would have
    kept the call off splash. A full and a sliding layer, since the sliding
    one runs `local_attention`."""
    model = CausalTransformer(
        vocab_size=64, num_layers=2, emb_features=64, num_heads=4, num_kv_heads=2,
        max_seq_len=512, layer_types=("sliding_attention", "full_attention"),
        kinds={"sliding_attention": {"window": 128}})
    ids = jnp.zeros((2, 512), jnp.int32)
    segment_ids = jnp.asarray(np.repeat(np.arange(1, 5), 128)[None].repeat(2, 0), jnp.int32)
    positions = jnp.asarray(np.tile(np.arange(128), 4)[None].repeat(2, 0))
    params = jax.eval_shape(model.init, jax.random.PRNGKey(0), ids)
    printed = jax.make_jaxpr(lambda p: model.apply(
        p, ids, positions=positions, segment_ids=segment_ids))(params).pretty_print(
            use_color=False)
    # Both layers run splash's segmented forward and nothing builds a mask.
    assert printed.count('pallas_call') == 2
    assert 'splash_mha_fwd_segmented' in printed
    assert 'bool[2,1,512,512]' not in printed and 'bool[2,512,512]' not in printed


def test_an_explicit_tpu_call_splash_cannot_describe_refuses_a_softcap(tpu_backend):
    """A bias sends an explicit 'tpu' to the flash kernel, which has no tanh
    and no sinks, so the call raises rather than dropping the cap."""
    query, key, value = qkv((2, 512, 8, 128))
    bias = jnp.zeros((2, 8, 512, 512), jnp.bfloat16)
    with pytest.raises(ValueError, match="splash"):
        jax.eval_shape(
            lambda q, k, v: attention_kernel(q, k, v, implementation='tpu', softcap=30.0,
                                             bias=bias),
            query, key, value)


@pytest.mark.parametrize('generation,runs', [('sm75', False), ('sm80', True), ('sm89', True),
                                             ('v6e', True), ('cpu', True)])
def test_a_gpu_older_than_sm80_multiplies_bf16_without_the_bf16_algorithm(monkeypatch,
                                                                          generation, runs):
    """sm75 rejects BF16_BF16_F32 at run time, so on it bf16 attention runs
    the reference path for 'auto' and 'xla' alike, and the bf16 operand
    precision keeps the caller's."""
    from dew.nn.attention import resolve_implementation
    from dew.nn.precision import bf16_operand_precision
    monkeypatch.setattr(kernels.generation, 'device_generation', lambda: generation)
    monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu' if generation.startswith('sm') else generation)
    query = jnp.zeros((1, 16, 2, 64), jnp.bfloat16)
    assert (bf16_operand_precision(jnp.bfloat16, None)
            is jax.lax.DotAlgorithmPreset.BF16_BF16_F32) == runs
    for requested in ('auto', 'xla'):
        chosen = resolve_implementation(requested, query, query)
        assert (chosen == 'reference') == (not runs)


def test_the_sm75_route_applies_on_a_gpu_backend_only(monkeypatch):
    """A GPU older than sm80 sends bf16 attention to the reference path, since
    its dot rejects BF16_BF16_F32. The route belongs to the backend that runs
    the call: the same generation read while the call traces for another
    backend keeps xla."""
    from dew.nn.attention import resolve_implementation
    monkeypatch.setattr(kernels.generation, 'device_generation', lambda: 'sm75')
    monkeypatch.setattr(jax, 'default_backend', lambda: 'cpu')
    query = jnp.zeros((1, 16, 2, 64), jnp.bfloat16)
    assert resolve_implementation('xla', query, query) == 'xla'


@pytest.mark.parametrize("dtype, keys, chosen", [
    (jnp.float32, VALUE_BLOCK, "xla"), (jnp.float32, VALUE_BLOCK + 1, "reference"),
    (jnp.float32, 2048, "reference"), (jnp.bfloat16, 2048, "xla")])
def test_cpu_fp32_attention_over_more_than_a_block_of_keys_takes_the_reference_path(monkeypatch, dtype, keys,
                                                                                     chosen):
    """jax.nn's xla attention on XLA:CPU sums an fp32 value product over all
    its keys in one YNNPACK chain; past VALUE_BLOCK keys 'auto' and 'xla'
    take the reference path, whose product sums in blocks. A shorter call
    and a bf16 one compute as before."""
    monkeypatch.setattr(jax, 'default_backend', lambda: 'cpu')
    query, key = jnp.zeros((1, 16, 2, 64), dtype), jnp.zeros((1, keys, 2, 64), dtype)
    for requested in ('auto', 'xla'):
        assert resolve_implementation(requested, query, key) == chosen


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


@pytest.mark.parametrize("call, taken", [
    ({}, True), ({"positions": 8, "heads": 2}, True), ({"positions": 9, "heads": 2}, False),
    ({"kv_heads": 1}, False), ({"causal": True}, False), ({"mask": True}, False),
    ({"backend": "cpu"}, False)])
def test_a_decode_shaped_xla_call_on_a_gpu_folds_its_heads(monkeypatch, call, taken):
    """An xla call of at most FOLDED_PAIRS query positions times heads, over
    keys of its own heads and masked by key lengths at most, reads the cache
    in its layout on a GPU (`folded_attention`); anything else is jax.nn's."""
    monkeypatch.setattr(jax, 'default_backend', lambda: call.get("backend", "gpu"))
    heads = call.get("heads", 2)
    query = jnp.zeros((3, call.get("positions", 4), heads, 256), jnp.bfloat16)
    key = jnp.zeros((3, 40, call.get("kv_heads", heads), 256), jnp.bfloat16)
    mask = jnp.ones((3, 1, query.shape[1], 40), bool) if call.get("mask") else None
    assert folds(query, key, None, mask, call.get("causal", False), None) == taken


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
    folded = folded_attention(q, k, v, lengths)
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


@pytest.mark.skipif(jax.default_backend() != 'gpu', reason="the kernel runs on CUDA")
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


@pytest.mark.skipif(jax.default_backend() != 'gpu', reason="the kernel runs on CUDA")
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
