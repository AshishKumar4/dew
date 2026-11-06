"""The sketch a recording keeps of a reference's float64 truth (tests/reference_error.py's
`indexed_recording`): its entries summed under random signs a recorded seed draws, exact in float64 but for a
bound of its own (`sketch_rounding`), so a recomputed truth is held to the recorded one without keeping it."""

import math

import jax
import jax.numpy as jnp
import numpy as np

SKETCH = 32
"""The random sign projections a recording keeps of each leaf of its reference's float64 truth
(`truth_sketch`): a move of the truth independent of the signs, a wrong row or index, moves their root
mean square by about its own norm (a single element's by exactly its size); a move built against the
signs, in the span their complement leaves, is not seen, which no fixed sketch can exclude."""
SKETCH_DEVIATIONS = 2.0
"""How many times the oracle rule's bound on a difference's norm the root mean square of its SKETCH sign
projections may reach. A truth recomputed within the oracle rule, its difference independent of the
signs, is refused with probability under 1e-11 (a Rademacher sum's square is no heavier-tailed than a
normal's: Chernoff's bound on a chi-square of SKETCH degrees past 2² times its mean). A single
element's move past twice the bound and the sketches' own rounding (`sketch_rounding`) is refused
always; a wider move past them in norm is refused but
for its chance of lying near the signs' complement, which is no guarantee: two entries swapped whose
sign words agree move no projection, one pair in 2^32 at a seed, and each seed draws its own signs, so
a wrong truth repeated over a site's seeds hides at all of them only by that chance to their power."""
SKETCH_BLOCK = 64
"""The entries `truth_sketch` sums in float64 in one block, each block's sum then summed exactly: its
rounding at most gamma(SKETCH_BLOCK) of the magnitudes', in whatever order a host takes a block."""


def _sign_bits(size: int, seed: int) -> np.ndarray:
    """SKETCH random signs for each of `size` entries at a recorded seed, one bit a sign, the same on every
    host and run, and drawn apart at every seed."""
    return np.random.default_rng([0x5E7C4, size, seed]).integers(0, 1 << SKETCH, size, dtype=np.uint64)


_BYTE_SIGNS = ((np.arange(256)[:, None] >> np.arange(8)) & 1) * 2.0 - 1
"""A (256, 8) float64 table whose [byte, k] entry is +1.0 where the byte's bit k is set and -1.0 where it
is not, least significant bit first."""


def truth_sketch(truth, seed: int) -> np.ndarray:
    """A float64 truth leaf's sketch `[SKETCH]` at a recorded seed: its entries summed under SKETCH random
    signs of that seed's own (`_sign_bits`), each sign's
    product exact, in blocks of SKETCH_BLOCK whose float64 sums are summed exactly (`math.fsum`), so a
    truth recomputed apart is held to the recorded one without the recording keeping it, every host
    rounding the sketch alike within `sketch_rounding` (`indexed_recording`). The products and the
    blocks' sums run on a GPU's float64 units where the process has one (`_device_partials`), never on
    its host while the GPU idles (ledger 1568), and on the host otherwise (`_host_partials`); the signs
    and the exact sum are the host's either way."""
    flat = np.asarray(truth, np.float64).reshape(-1)
    if not flat.size:
        return np.zeros(SKETCH)
    bits = _sign_bits(flat.size, seed)
    partials = (_device_partials if jax.default_backend() == "gpu" else _host_partials)(flat, bits)
    return np.asarray([math.fsum(column) for column in partials.T.tolist()])


def _host_partials(flat: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """`flat`'s blocks' sums `[blocks, SKETCH]` under the signs `bits` holds, on the host, the last block
    padded with zeros: a block's entries summed by each sign byte's value (`np.bincount`) times its eight
    signs (`_BYTE_SIGNS`), one order of its exact products, within gamma(SKETCH_BLOCK) (ledger 2807)."""
    partials = []
    for start in range(0, flat.size, 1 << 12):
        chunk = flat[start:start + (1 << 12)]
        # Sign k of an entry is its word's bit k: its low four bytes, read little-endian whatever the host's
        # order, so the first byte holds bits 0 to 7. An entry's bins: its block's, its byte's, the value's.
        words = bits[start:start + chunk.size].astype("<u8").view(np.uint8).reshape(chunk.size, 8)[:, :4]
        bins = (np.arange(chunk.size)[:, None] // SKETCH_BLOCK * 4 + np.arange(4)) * 256 + words
        summed = np.bincount(bins.reshape(-1), np.repeat(chunk, 4), -(-chunk.size // SKETCH_BLOCK) * 1024)
        partials.append((summed.reshape(-1, 256) @ _BYTE_SIGNS).reshape(-1, SKETCH))
    return np.concatenate(partials)


DEVICE_CHUNK = 1 << 20
"""The entries one call of `_device_block_sums` takes: their `[DEVICE_CHUNK, SKETCH]` float64 products,
256 MiB, a multiple of SKETCH_BLOCK so its blocks are the host's."""


@jax.jit
def _device_block_sums(chunk: jax.Array, words: jax.Array) -> jax.Array:
    """`chunk`'s blocks' sums `[DEVICE_CHUNK // SKETCH_BLOCK, SKETCH]` under the signs `words` hold, each
    sign its word's bit."""
    signs = ((words[:, None] >> jnp.arange(SKETCH, dtype=jnp.uint64)) & 1).astype(jnp.float64) * 2 - 1
    return (chunk[:, None] * signs).reshape(-1, SKETCH_BLOCK, SKETCH).sum(axis=1)


def _device_partials(flat: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """`_host_partials` on the default device in float64: the same products, exact, each block summed
    in whatever order the device takes it, within gamma(SKETCH_BLOCK) of its magnitudes, a chunk at a
    time (`DEVICE_CHUNK`, the last padded with zeros, whose blocks past the entries' are dropped)."""
    blocks = -(-flat.size // SKETCH_BLOCK)
    padded = -flat.size % DEVICE_CHUNK
    flat, bits = np.pad(flat, (0, padded)), np.pad(bits, (0, padded))
    with jax.enable_x64(new_val=True):
        partials = [np.asarray(_device_block_sums(jnp.asarray(flat[start:start + DEVICE_CHUNK]),
                                                  jnp.asarray(bits[start:start + DEVICE_CHUNK])))
                    for start in range(0, flat.size, DEVICE_CHUNK)]
    return np.concatenate(partials)[:blocks]


def sketch_rounding(magnitude: float) -> float:
    """The most `truth_sketch` of a leaf whose magnitudes sum to `magnitude` may round by, on any host:
    gamma(SKETCH_BLOCK) of it within the blocks, and the exact sum's one rounding."""
    eps = float(np.finfo(np.float64).eps)
    return (SKETCH_BLOCK * eps / 2 / (1 - SKETCH_BLOCK * eps / 2) + eps / 2) * magnitude
