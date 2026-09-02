"""The GPU kernel paths: pallas device registration, the triton flash kernel,
and the XLA flags a run passes to the backend.

Everything that needs a CUDA device skips elsewhere; the shape rules and the
flag handling are plain functions and run anywhere. Run the GPU half with
`JAX_PLATFORMS=cuda python -m pytest tests/test_kernels.py`.
"""

import os

import jax
import jax.numpy as jnp
import pytest

from dew.nn.attention import pallas_supports, scaled_dot_product_attention
from dew.telemetry.devices import apply_xla_flags, register_pallas_device

on_gpu = pytest.mark.skipif(jax.default_backend() != 'gpu',
                            reason="needs a cuda device")

SHAPE = (2, 256, 8, 64)  # [B, S, H, D], one supported shape of the kernel


def qkv(shape=SHAPE, dtype=jnp.bfloat16, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    return tuple(jax.random.normal(key, shape, jnp.float32).astype(dtype)
                 for key in keys)


def test_registration_is_a_noop_without_a_gpu():
    if jax.default_backend() == 'gpu':
        pytest.skip("this is the CPU answer")
    assert register_pallas_device() is False


@on_gpu
def test_a_triton_kernel_runs_on_this_card_after_registration():
    """The registration exists because pallas refuses a card missing from its
    table, which on this hardware is every consumer card. A kernel that
    compiles is the only proof that the entry was the whole gap."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    assert register_pallas_device() is True

    def double(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2

    x = jnp.arange(256, dtype=jnp.float32)
    out = pl.pallas_call(
        double,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        compiler_params=plgpu.CompilerParams(),
    )(x)
    assert jnp.array_equal(out, x * 2)


@pytest.mark.parametrize("kwargs,shape,dtype,expected", [
    ({}, (2, 77, 8, 64), jnp.bfloat16, "multiple of 128"),
    ({}, (2, 96, 8, 64), jnp.bfloat16, "multiple of 128"),
    ({}, (2, 256, 8, 160), jnp.bfloat16, "shared memory"),
    ({}, SHAPE, jnp.float32, "bf16 or fp16"),
    ({'sliding_window': 64}, SHAPE, jnp.bfloat16, "sliding-window"),
])
def test_the_kernel_says_which_calls_it_cannot_take(kwargs, shape, dtype, expected):
    """Every clause is a limit of the shipped kernel, so a caller finds out at
    trace time instead of getting a lowering error or a silent fallback."""
    query, key, _ = qkv(shape, dtype)
    assert expected in pallas_supports(query, key, **kwargs)


def test_a_materialized_mask_has_nowhere_to_go():
    query, key, _ = qkv()
    mask = jnp.ones((1, 1, SHAPE[1], SHAPE[1]), bool)
    assert "no mask" in pallas_supports(query, key, mask=mask)


def test_cross_attention_lengths_are_refused_for_the_backward_pass():
    """77 text tokens against 256 image tokens is the diffusion cross-attention
    call, and the fused backward pass splits both sequences into the same
    number of blocks."""
    query, _, _ = qkv()
    key, _, _ = qkv((2, 77, 8, 64))
    assert "lengths to match" in pallas_supports(query, key)


def test_supported_shapes_are_accepted():
    query, key, _ = qkv()
    assert pallas_supports(query, key) is None
    short, short_key, _ = qkv((2, 64, 8, 64))
    assert pallas_supports(short, short_key) is None


@on_gpu
@pytest.mark.parametrize("causal", [False, True])
def test_pallas_matches_the_reference_forward(causal):
    query, key, value = qkv()
    kernel = scaled_dot_product_attention(
        query, key, value, implementation='pallas', causal=causal)
    reference = scaled_dot_product_attention(
        query, key, value, implementation=None, causal=causal)

    difference = jnp.max(jnp.abs(kernel.astype(jnp.float32)
                                 - reference.astype(jnp.float32)))
    # Measured 3.9e-3 (causal) and 4.9e-3 (full) at this shape in bf16
    assert difference < 2e-2, f"max difference {difference:.2e}"


@on_gpu
@pytest.mark.parametrize("causal", [False, True])
def test_pallas_matches_the_reference_gradients(causal):
    """mha carries a custom_vjp, so its backward pass is a second kernel that
    the training step is the only caller of."""
    query, key, value = qkv()
    cotangent = jax.random.normal(jax.random.PRNGKey(3), query.shape, jnp.bfloat16)

    def grads(implementation):
        def forward(*args):
            return scaled_dot_product_attention(
                *args, implementation=implementation, causal=causal)
        out, vjp = jax.vjp(forward, query, key, value)
        # The reference path returns fp32, the kernel bf16; the cotangent is
        # the same numbers either way.
        return vjp(cotangent.astype(out.dtype))

    for kernel, reference in zip(grads('pallas'), grads(None)):
        difference = jnp.max(jnp.abs(kernel.astype(jnp.float32)
                                     - reference.astype(jnp.float32)))
        assert difference < 2e-2, f"max difference {difference:.2e}"


@on_gpu
def test_grouped_query_heads_reach_the_kernel():
    """The kernel indexes k and v by the query head, so grouped heads have to
    be repeated out before the call rather than after it."""
    query, _, _ = qkv()
    _, key, value = qkv((2, 256, 2, 64))
    kernel = scaled_dot_product_attention(
        query, key, value, implementation='pallas', causal=True)
    reference = scaled_dot_product_attention(
        query, key, value, implementation=None, causal=True)

    difference = jnp.max(jnp.abs(kernel.astype(jnp.float32)
                                 - reference.astype(jnp.float32)))
    assert difference < 2e-2, f"max difference {difference:.2e}"


@on_gpu
def test_an_unsupported_shape_raises_instead_of_falling_back():
    query, key, value = qkv((2, 96, 8, 64))
    with pytest.raises(ValueError, match="multiple of 128"):
        scaled_dot_product_attention(query, key, value, implementation='pallas')


@on_gpu
def test_auto_trains_cross_attention_over_77_text_tokens():
    """The default kernel choice on a real GPU run of any cross-attending
    model. cudnn takes the forward pass and raises in the gradient, so this
    only fails where it matters: the first training step."""
    query, _, _ = qkv((2, 1024, 4, 64))
    _, context, _ = qkv((2, 77, 4, 64))

    def step(implementation):
        def loss(q, k, v):
            out = scaled_dot_product_attention(q, k, v, implementation=implementation)
            return jnp.sum(out.astype(jnp.float32))
        return jax.grad(loss, argnums=(0, 1, 2))(query, context, context)

    gradients = step('auto')
    assert all(jnp.all(jnp.isfinite(g.astype(jnp.float32))) for g in gradients)
    with pytest.raises(NotImplementedError):
        step('cudnn')


def test_pallas_off_gpu_names_the_backend_it_needs():
    if jax.default_backend() == 'gpu':
        pytest.skip("this is the CPU answer")
    with pytest.raises(ValueError, match="cuda device"):
        scaled_dot_product_attention(*qkv(), implementation='pallas')


def test_flags_are_appended_to_what_the_environment_already_carries(monkeypatch):
    """The test suite itself sets a flag, and a run's own flags have to add to
    it rather than replace it."""
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    apply_xla_flags("--xla_gpu_autotune_level=4")
    assert os.environ["XLA_FLAGS"] == (
        "--xla_force_host_platform_device_count=8 --xla_gpu_autotune_level=4")


@pytest.mark.parametrize("flags", [None, ""])
def test_no_flags_leaves_the_environment_alone(monkeypatch, flags):
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    apply_xla_flags(flags)
    assert os.environ["XLA_FLAGS"] == "--xla_force_host_platform_device_count=8"


def test_flags_reach_an_environment_that_had_none(monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    apply_xla_flags("--xla_gpu_triton_gemm_any=true")
    assert os.environ["XLA_FLAGS"] == "--xla_gpu_triton_gemm_any=true"
