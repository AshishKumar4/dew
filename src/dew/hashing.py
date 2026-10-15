"""murmur3's 32-bit finaliser, the one mixing step Dew's integer hashes share."""


def fmix32(words):
    """murmur3's 32-bit finaliser of a uint32 array, NumPy's or JAX's, in the
    array's own module: every input bit reaches every output bit."""
    word = words.dtype.type
    words = words ^ (words >> word(16))
    words = words * word(0x85EBCA6B)
    words = words ^ (words >> word(13))
    words = words * word(0xC2B2AE35)
    return words ^ (words >> word(16))
