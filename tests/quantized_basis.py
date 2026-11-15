"""A quantized KV cache's values dequantized in their own basis, computed apart from the cache's reads.

Plain jnp from the layout's rule (`dew.nn.kv_cache`'s module doc), each vector rounded by the cache's own
rounding (`dew.nn.kv_cache.quantize`, which tests/test_kv_cache.py holds to NumPy's IEEE arithmetic):
int8 keys rotated by the orthonormal Hadamard basis before their rounding and back after it, values and
float8 unrotated. Tests hold a quantized cache's decode to a run over these values, so a basis conversion
dropped in the cache's consumers (a query not put in the stored keys' basis, keys not read back plain)
moves the decode off them.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.kv_cache import KVDtype, quantize


def hadamard_basis(size: int) -> np.ndarray:
    """Sylvester's Hadamard matrix of order `size`, orthonormal, in float64."""
    basis = np.ones((1, 1))
    while basis.shape[0] < size:
        basis = np.kron(basis, np.array([[1.0, 1.0], [1.0, -1.0]]))
    return basis / math.sqrt(size)


def dequantized_in_own_basis(values, quantized: KVDtype, *, keys: bool):
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
    codes, scale = quantize(stored, quantized)
    width = jnp.promote_types(values.dtype, jnp.float32)
    back = (codes.astype(width) * scale[..., None].astype(width)).astype(values.dtype)
    return rotated(back, basis.T) if rotate else back
