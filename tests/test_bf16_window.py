"""A frozen bfloat16 matrix's lossless 13-bit window (`formats.Bf16Window`, row 5387): each weight its bits
again decoded, a group past its escapes refused by its site's name, and a Dense served its window as its
kernel (`quantization.bf16_windows`), its variables' kernels back bitwise (`decoded_bf16_windows`)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew import quantization
from dew.nn.dense import Dense
from dew.nn.kernels.formats import WINDOW_ESCAPES, Bf16Window, WindowRefused


def _bits(values) -> np.ndarray:
    return np.asarray(values, jnp.bfloat16).view(np.uint16)


def _drawn(shape, seed=5387) -> np.ndarray:
    """A normal bfloat16 weight draw (a trained projection's scale) beside the edges its window codes:
    signed zeros, subnormals and a tiny weight among its escapes, a group's floor `2^b` exactly, and a group
    anchored at the largest finite and holding infinities and NaNs (each its bits)."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.02, shape).astype(jnp.bfloat16)
    bits = weights.view(np.uint16)
    bits[0, 0], bits[1, 0], bits[2, 0], bits[3, 0] = 0x8000, 0x0000, 0x0001, 0x8005  # -0, +0, subnormals
    top = int(((bits[:128, 1] >> 7) & 0xFF).max())
    bits[5, 1] = (top - 14) << 7  # exactly 2^b, the window's floor
    bits[128:256, 2] = 0x7F00 | (bits[128:256, 2] & 0x807F)  # near the largest finite
    bits[128, 2], bits[129, 2], bits[130, 2] = 0x7F80, 0xFF80, 0x7FC1  # +inf, -inf, a NaN's payload
    return weights


def test_a_bf16_window_decodes_every_weight_to_its_bits():
    """Every weight of a window, decoded, is the weight it encoded, bit for bit, at 13 bits a weight."""
    weights = _drawn((512, 96))
    window = Bf16Window.of(weights).checked()
    assert window.shape == weights.shape
    np.testing.assert_array_equal(np.asarray(window.decoded()).view(np.uint16), weights.view(np.uint16))
    stored = sum(np.asarray(plane).size * 32 for plane in (window.low, window.offsets, window.meta))
    assert stored / weights.size == 13


def test_a_group_past_its_escapes_is_refused_by_its_sites_name():
    """A group holding more nonzeros below its window than it stores escapes (`WINDOW_ESCAPES`: one weight
    near 2^0 beside five tiny ones, 2^-20 and less) is refused, naming its site, where no window holds it."""
    weights = np.full((128, 4), 0.25, jnp.bfloat16)
    weights[: WINDOW_ESCAPES + 1, 3] = np.asarray([2.0 ** -20] * (WINDOW_ESCAPES + 1), jnp.bfloat16)
    with pytest.raises(WindowRefused) as refused:
        Bf16Window.of(weights, site="layers/q_proj")
    assert (refused.value.site, refused.value.column, refused.value.group) == ("layers/q_proj", 3, 0)
    assert refused.value.count == WINDOW_ESCAPES + 1


def test_a_dense_served_its_window_is_its_kernels_product_and_its_kernel_comes_back():
    """A bfloat16 Dense whose kernel `bf16_windows` stores as its window computes its kernel's product (each
    value dyadic, so every order sums it exactly in fp32, its bias added and rounded once), its kernel
    decoded back bitwise by
    `decoded_bf16_windows`, the collection gone."""
    rng = np.random.default_rng(5387)
    kernel = (rng.integers(-64, 64, (256, 64)) / 16.0).astype(jnp.bfloat16)
    bias = (rng.integers(-8, 8, (64,)) / 4.0).astype(jnp.float32)
    x = jnp.asarray(rng.integers(-4, 4, (8, 256)), jnp.bfloat16)
    layer = Dense(64, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
    variables = {"params": {"kernel": jnp.asarray(kernel), "bias": jnp.asarray(bias, jnp.bfloat16)}}
    windowed = quantization.bf16_windows(variables)
    assert "kernel" not in windowed["params"]
    rounded = np.asarray(bias.astype(jnp.bfloat16), np.float64)
    want = np.asarray(x, np.float64) @ np.asarray(kernel, np.float64) + rounded
    # Its fp32 sum exact, plus its bias, rounded once to the layer's bfloat16.
    np.testing.assert_array_equal(np.asarray(jax.jit(layer.apply)(windowed, x), np.float32),
                                  np.asarray(want.astype(jnp.bfloat16), np.float32))
    back = quantization.decoded_bf16_windows(windowed)
    assert set(back) == {"params"}
    np.testing.assert_array_equal(np.asarray(back["params"]["kernel"]).view(np.uint16),
                                  kernel.view(np.uint16))
