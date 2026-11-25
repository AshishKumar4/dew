"""Copy legality (design §6, the view vocabulary's copies): what an asynchronous copy between global and
shared memory takes, the one statement every launch's planner, staging, admission and register model read: a
row of whole 16-byte units (`ROW_BYTES`: a tensor map's box's inner dimension, cp.async's widest piece, a bulk
copy's), at most 256 elements along each dimension of a tensor map's box (`BOX_ELEMENTS`), and shared
addresses aligned to 128 bytes (`SHARED_ALIGNMENT`)."""

import jax

ROW_BYTES = 16
"""The bytes an asynchronous copy's row is a whole number of: a tensor map's box takes an inner dimension of a
multiple of 16 bytes (CUDA's `cuTensorMapEncodeTiled`: `boxDim[0]` times the element's size), and every
global stride but the last one is one ("Async copies require all GMEM strides except the last one to be a
multiple of 16 bytes"); cp.async moves at most 16 bytes a thread, and a row of no whole 16-byte unit stages
unswizzled (`swizzling`)."""

BOX_ELEMENTS = 256
"""The most elements a tensor map's box takes along a dimension (`cuTensorMapEncodeTiled`'s `boxDim`)."""

SHARED_ALIGNMENT = 128
"""The bytes a tensor copy's shared-memory address is aligned to (CUDA's asynchronous copies, Table 26)."""


def whole_rows(elements: int, dtype) -> bool:
    """Whether a row of `elements` of `dtype` is a whole number of `ROW_BYTES` units, by the dtype's bits (a
    packed fp4's half byte): one an asynchronous copy takes as it is, a tensor map's inner box or a stride
    of one, unpadded."""
    return elements * jax.dtypes.itemsize_bits(dtype) % (8 * ROW_BYTES) == 0
