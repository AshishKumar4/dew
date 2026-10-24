"""murmur3's 32-bit finaliser, the one mixing step Dew's integer hashes share, and the one way an
array's contents are fingerprinted."""

import json
from collections.abc import Sequence
from typing import Protocol

import numpy as np


def fmix32(words):
    """murmur3's 32-bit finaliser of a uint32 array, NumPy's or JAX's, in the
    array's own module: every input bit reaches every output bit."""
    word = words.dtype.type
    words = words ^ (words >> word(16))
    words = words * word(0x85EBCA6B)
    words = words ^ (words >> word(13))
    words = words * word(0xC2B2AE35)
    return words ^ (words >> word(16))


class Hasher(Protocol):
    """A running hash, as `hashlib`'s are."""

    def update(self, chunk: bytes, /) -> None: ...


def array_header(dtype: np.dtype | type, shape: Sequence[int]) -> bytes:
    """The header an array of `dtype` and `shape` is fingerprinted with: the dtype by the name NumPy
    reads back (`str(np.dtype)`), which tells apart the float8, float6, float4 and int4 types whose
    `dtype.str` is one '<V1' and keeps byte order, and the shape, each element's bytes to follow."""
    return json.dumps([str(np.dtype(dtype)), [int(size) for size in shape]],
                      separators=(",", ":")).encode()


def array_fingerprint(hasher: Hasher, array: np.ndarray) -> None:
    """Feed `hasher` an array: its header (`array_header`), then its elements' bytes in C order, so
    equal bytes of another dtype or shape hash apart. An object array, whose bytes are pointers, is
    refused."""
    if array.dtype.hasobject:
        raise TypeError(f"an array of {array.dtype} holds pointers, whose bytes fingerprint nothing")
    hasher.update(array_header(array.dtype, array.shape))
    hasher.update(np.ascontiguousarray(array).tobytes())
