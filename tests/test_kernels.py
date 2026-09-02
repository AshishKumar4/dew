"""The GPU kernel paths: the fused cudnn kernel the 'auto' rule reaches for,
and the XLA flags a run passes to the backend.

Everything that needs a CUDA device skips elsewhere; the flag handling is a
plain function and runs anywhere. Run the GPU half with
`JAX_PLATFORMS=cuda python -m pytest tests/test_kernels.py`.
"""

import os

import jax
import jax.numpy as jnp
import pytest

from dew.nn.attention import scaled_dot_product_attention
from dew.telemetry.devices import apply_xla_flags

on_gpu = pytest.mark.skipif(jax.default_backend() != 'gpu',
                            reason="needs a cuda device")


def qkv(shape):
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    return tuple(jax.random.normal(key, shape, jnp.bfloat16) for key in keys)


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
