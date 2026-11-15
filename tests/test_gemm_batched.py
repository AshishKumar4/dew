"""Batched products through the GEMM selector (`gemm.batched_product`, row 5972): XLA's dot where no receipt
routes them, bit for bit, and the template's grouped product of their batches where named."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import runs
from reference_error import assert_as_exact

from dew.nn.kernels import gemm, generation

SEEDS = 8


BATCHED_CONTRACTIONS = [((2,), (-1, -2)), ((2,), (-1, -1)), ((3, 2), (-1, -2)), ((), (-1, -2))]
"""Batched products' leading axes and contractions (`gemm.batched_product`): one batch axis contracting `b`'s
rows or its columns (Newton-Schulz's `B X` and `X Xᵀ`), two batch axes, and none."""


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32, jnp.complex64])
@pytest.mark.parametrize(("batch", "contract"), BATCHED_CONTRACTIONS)
def test_a_batched_product_on_xlas_route_is_xlas_dot_bit_for_bit(batch, contract, dtype):
    """A batched product named onto XLA's route (`generation.chosen(grouped_matmul='xla')`), real or complex,
    over one batch axis, two or none, is bitwise `jax.lax.dot_general` of the same dimensions (row 5972:
    Muon's Newton-Schulz iteration through the GEMM selector, its numbers XLA's wherever XLA serves it)."""
    rng = np.random.default_rng(0)
    last = len(batch) + 1
    a_shape = (*batch, 24, 40) if contract[0] == -1 else (*batch, 40, 24)
    b_shape = (*batch, 40, 32) if contract[1] == -2 else (*batch, 32, 40)

    def draw(shape):
        real = rng.normal(size=shape)
        return jnp.asarray(real + 1j * rng.normal(size=shape) if jnp.issubdtype(dtype, jnp.complexfloating)
                           else real, dtype)

    a, b = draw(a_shape), draw(b_shape)
    contracted = (last + 1 + contract[0], last + 1 + contract[1])
    batch_axes = tuple(range(len(batch)))
    expected = jax.lax.dot_general(a, b, ((contracted[:1], contracted[1:]), (batch_axes, batch_axes)))
    with generation.chosen(grouped_matmul="xla"):
        out = gemm.batched_product(a, b, contracted)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))


@runs("sm90")
@pytest.mark.parametrize(("batch", "contract"), BATCHED_CONTRACTIONS[:3])
def test_a_batched_product_on_the_template_holds_check_1(batch, contract):
    """A bf16 batched product named onto the template (`generation.chosen(grouped_matmul='mosaic')`, its
    batches the grouped product's groups), interpreted on a CPU as an H100: on par with XLA's batched dot
    from float64 truth (check 1)."""
    last = len(batch) + 1
    contracted = (last, last - 1 if contract[1] == -2 else last)
    b_shape = (*batch, 128, 64) if contract[1] == -2 else (*batch, 64, 128)

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(*batch, 64, 128)), jnp.bfloat16),
                jnp.asarray(rng.normal(size=b_shape) / 8, jnp.bfloat16))

    batch_axes = tuple(range(len(batch)))

    def plain(a, b, wide=jnp.float32):
        out = jax.lax.dot_general(a.astype(wide), b.astype(wide), ((contracted[:1], contracted[1:]),
                                                                    (batch_axes, batch_axes)))
        return out if wide == jnp.float64 else out.astype(jnp.bfloat16)

    def templated(a, b):
        with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
            return gemm.batched_product(a, b, contracted, interpret_on_cpu=True)

    assert_as_exact(templated, jax.jit(plain), draw, "a batched product on the template", seeds=SEEDS,
                    truth_fn=functools.partial(plain, wide=jnp.float64))
