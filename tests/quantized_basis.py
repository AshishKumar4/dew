"""A quantized KV cache's values dequantized in their own basis, computed apart from the cache.

Plain jnp from the layout's rule (`dew.nn.kv_cache`'s module doc): a float32
scale per vector, its absmax over the format's largest value (one where that
scale would not be a normal float32); int8 keys rotated by the orthonormal
Hadamard basis before their rounding and back after it, values and float8
unrotated. Tests hold a quantized cache's decode to a run over these values,
so a basis conversion dropped in the cache's consumers (a query not put in
the stored keys' basis, keys not read back plain) moves the decode off them,
and hold the codes and scales a cache stores to the rule's (`quantized_codes`,
tests/cache_roundings.py).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np


def hadamard_basis(size: int) -> np.ndarray:
    """Sylvester's Hadamard matrix of order `size`, orthonormal, in float64."""
    basis = np.ones((1, 1))
    while basis.shape[0] < size:
        basis = np.kron(basis, np.array([[1.0, 1.0], [1.0, -1.0]]))
    return basis / math.sqrt(size)


def dequantized_in_own_basis(values, quantized: str, *, keys: bool):
    """`values` [..., D] as a `quantized` cache reads them back in their own basis: quantized
    per vector and dequantized, int8 keys through the Hadamard rotation and back, in `values`'
    dtype. The rotations sum in float64 and round once, where the cache's own sum in float32
    (`dew.nn.kv_cache.rotated`): an int8 consumer agrees with them to check 1, and float8, which
    stores its values unrotated, bit for bit."""
    def rotated(v, basis):
        with jax.enable_x64():
            return jnp.einsum("...d,de->...e", v.astype(jnp.float64), jnp.asarray(basis)).astype(v.dtype)

    rotate = keys and quantized == "int8"
    basis = hadamard_basis(values.shape[-1]) if rotate else None
    stored = rotated(values, basis) if rotate else values
    codes, scale = quantized_codes(stored, quantized)
    width = jnp.promote_types(values.dtype, jnp.float32)
    back = (codes.astype(width) * scale[..., None].astype(width)).astype(values.dtype)
    return rotated(back, basis.T) if rotate else back


def quantized_codes(values, quantized: str) -> tuple[jax.Array, jax.Array]:
    """`values` [..., D] rounded as a `quantized` cache stores them, in the basis they come in: each vector's
    codes and its float32 scale `[...]`, the vector's absmax over the format's largest value, one where that
    would not be a normal float32."""
    # At the values' width, fp32 at least (a float64 truth's float64); the scale is the float32 a cache
    # stores (its quotient rounded once to fp32, as fp32's division rounds it), each code its rounding.
    width = jnp.promote_types(values.dtype, jnp.float32)
    wide = values.astype(width)
    limit = 127.0 if quantized == "int8" else float(jnp.finfo(jnp.float8_e4m3fn).max)
    scaled_amax = (jnp.max(jnp.abs(wide), axis=-1) / limit).astype(jnp.float32)
    scale = jnp.where(scaled_amax >= jnp.finfo(jnp.float32).tiny, scaled_amax, jnp.float32(1))
    scaled = wide / scale[..., None].astype(width)
    codes = (jnp.clip(jnp.rint(scaled), -127, 127).astype(jnp.int8) if quantized == "int8"
             else scaled.astype(jnp.float8_e4m3fn))
    return codes, scale
