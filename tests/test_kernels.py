"""The GPU kernel paths: the fused cudnn kernel the 'auto' rule reaches for,
and the XLA flags a run passes to the backend.

Everything that needs a CUDA device skips elsewhere; the flag handling is a
plain function and runs anywhere. Run the GPU half with
`JAX_PLATFORMS=cuda python -m pytest tests/test_kernels.py`.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh

from dew.nn.attention import scaled_dot_product_attention
from dew.telemetry.devices import apply_xla_flags

on_gpu = pytest.mark.skipif(jax.default_backend() != 'gpu',
                            reason="needs a cuda device")


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
@pytest.mark.parametrize("q_len, kv_len, causal", [
    (1024, 77, False),   # every cross-attention over CLIP's 77 text tokens
    (9, 7, False),       # short enough that one attended pad key would move an eighth of the mass
    (333, 333, True),    # the concatenated text-plus-image sequence, causal
])
def test_cudnn_trains_odd_lengths_and_agrees_with_xla(q_len, kv_len, causal):
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

    for got, want in zip(fused, reference):
        assert got.shape == want.shape
        assert np.all(np.isfinite(got))
        assert np.abs(got - want).max() <= 2 ** -6 * np.abs(want).max()


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
