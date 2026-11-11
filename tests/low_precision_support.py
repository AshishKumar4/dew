"""The low-precision products' test operands, one builder each: an NVFP4 matrix of random codes under
power-of-two block scales and its exact values (`nvfp4`), and a normal draw whose blocks of the product's
depth span a range of binades (`spread`), as the MX and NVFP4 block-scaled products' tests and matrix cells
draw them."""

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
