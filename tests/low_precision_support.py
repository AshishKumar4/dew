"""The low-precision products' test operands, one builder each: an NVFP4 matrix of random codes under
power-of-two block scales and its exact values (`nvfp4`), a normal draw whose blocks of the product's depth
span a range of binades (`spread`), as the MX and NVFP4 block-scaled products' tests and matrix cells draw
them, and a float format's values at every edge of its bits (`edge_values`), as a reading or a scaling of
them is tested."""

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn import fake_quant
from dew.nn.kernels import formats


def nvfp4(seed: int, depth: int, width: int, *, scale: float) -> tuple[formats.NvFp4, np.ndarray]:
    """An NVFP4 matrix (`formats.NvFp4.of`) of random E2M1 codes (`fake_quant.E2M1`, each code's value) under
    power-of-two block scales from 2^-3 to 2^3 and the tensor's `scale`, and its values in float64, each
    exact: code times block scale times `scale`."""
    rng = np.random.default_rng(seed)
    magnitudes = fake_quant.E2M1[rng.integers(0, 16, size=(depth, width))].astype(np.float64)
    block = np.exp2(rng.integers(-3, 4, size=(depth // 16, width)))
    matrix = formats.NvFp4.of(jnp.asarray(magnitudes, jnp.float32), jnp.asarray(block, jnp.float8_e4m3fn),
                              scale)
    return matrix, magnitudes * np.repeat(block, 16, axis=0) * scale


def spread(rng: np.random.Generator | int, shape: tuple[int, ...], block: int) -> jnp.ndarray:
    """A normal draw of `shape` in fp32 whose `block`-deep blocks along its last axis span magnitudes from
    2^-6 to 2^6, a block's scale each (`rng` a generator, or a seed)."""
    rng = np.random.default_rng(rng)
    binades = np.exp2(rng.integers(-6, 7, size=(*shape[:-1], shape[-1] // block, 1)))
    values = rng.normal(size=(*shape[:-1], shape[-1] // block, block)) * binades
    return jnp.asarray(values.reshape(shape), jnp.float32)


def edge_values(dtype) -> np.ndarray:
    """A float format's values across every binade, of both signs, from their bits: zeros, subnormals of
    every leading bit's place and the least and largest, the least normal and the largest, normals across
    every exponent, and significands of ties to even at every shift (1.5, 3 and an ulp, a subnormal's 0b11
    and 0b101 leads): every edge of a reading or a scaling of its bits."""
    limits = jnp.finfo(dtype)
    word = np.dtype(f"uint{limits.bits}")
    width = limits.nmant
    infinite = int(np.asarray(np.inf, dtype).view(word)[()])
    one, three = (int(np.asarray(value, dtype).view(word)[()]) for value in (1.5, 3.0))
    bits = np.concatenate([
        np.arange(0, 1 << width, ((1 << width) // 2047) | 1, dtype=word),
        np.arange(1 << width, infinite, ((infinite - (1 << width)) // 2047) | 1, dtype=word),
        # A subnormal led by each bit's place, its lower bits mixed: every leading-bit position.
        np.asarray([(1 << place) | ((1 << place) - 1) // 3 for place in range(width)], word),
        np.asarray([0, 1, 2, 3, (1 << width) - 1, 1 << width, infinite - 1, 3 << (width - 1),
                    *([5 << (width - 3)] if width >= 3 else []), one, three + 1], word)])
    values = bits.view(dtype)
    return np.concatenate([values, -values])


class StageRef:
    """Mutable host-backed Pallas ref stub for bitwise stage-decode tests (`stage_decoded_fp8`/`ternary`)."""

    def __init__(self, arr):
        self.arr = np.asarray(arr)

    def __getitem__(self, key):
        return jnp.asarray(self.arr) if key is Ellipsis else jnp.asarray(self.arr[key])

    def __setitem__(self, key, val):
        rows, sl = key
        self.arr[rows, sl.start:sl.start + sl.size] = np.asarray(val)


def decoded_weight_dot(x, weights, *, dtype=jnp.float32):
    """Plain reference dot against a decoded packed weight bank (`Fp8Weights`, `TernaryWeights`)."""
    w_dec = (weights.decoded(jnp.float64) if dtype == jnp.float64
             else weights.decoded(jnp.bfloat16).astype(dtype))
    return jax.lax.dot(x.astype(dtype), w_dec, preferred_element_type=dtype)
