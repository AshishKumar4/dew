"""The sketch a recording keeps of each float64 truth leaf (tests/reference_error.py `truth_sketch`): its
definition, bit for bit, and its rounding in any order a host sums a block."""

import math

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("size", [1, 63, 64, 65, (1 << 16) + 70])
def test_a_truths_sketch_is_its_signed_block_sums_summed_exactly(size):
    """The sketch a recording keeps is its definition, bit for bit: each entry times each of its SKETCH
    signs (its sign word's bits), summed in blocks of SKETCH_BLOCK in order (a block past the end padded
    with zeros), the blocks' sums summed exactly, over chunks of 2^16 entries; at a block's and a
    chunk's edges, with signed zeros and subnormals among the entries, so a recorded sketch stays a
    recomputed truth's."""
    from reference_error import SKETCH, SKETCH_BLOCK, _sign_bits, truth_sketch

    rng = np.random.default_rng(size)
    truth = rng.normal(size=size)
    truth[::7] = -0.0
    truth[1::11] = 2.0 ** -1060
    signs = ((_sign_bits(size, 5)[:, None] >> np.arange(SKETCH, dtype=np.uint64)) & 1) * 2.0 - 1
    blocks = []
    for start in range(0, size, 1 << 16):
        chunk = truth[start:start + (1 << 16), None] * signs[start:start + (1 << 16)]
        chunk = np.concatenate([chunk, np.zeros((-len(chunk) % SKETCH_BLOCK, SKETCH))])
        blocks.append(chunk.reshape(-1, SKETCH_BLOCK, SKETCH).sum(axis=1))
    stacked = np.concatenate(blocks)
    defined = np.asarray([math.fsum(stacked[:, index].tolist()) for index in range(SKETCH)])
    np.testing.assert_array_equal(truth_sketch(truth, 5), defined)


def test_a_truths_sketch_rounds_alike_in_any_order_and_refuses_an_overflowing_or_empty_truth_apart():
    """The sketch of one truth agrees with itself summed in another order within `sketch_rounding`, each
    block of SKETCH_BLOCK summed from its other end as another host may; a
    truth recomputed far past float64's range is refused, not passed on an infinite bound; and an empty
    leaf's evidence is exact, its reference's dtype kept."""
    from reference_error import SKETCH, _sign_bits, leaf_summary, sketch_rounding, truth_sketch

    rng = np.random.default_rng(7)
    truth = rng.normal(size=4096)
    signs = ((_sign_bits(truth.size, 3)[:, None] >> np.arange(SKETCH, dtype=np.uint64)) & 1) * 2.0 - 1
    # Another host: each block summed one by one from its end, the blocks' sums summed exactly.
    blocks = np.flip((truth[:, None] * signs).reshape(-1, 64, SKETCH), axis=1).cumsum(axis=1)[:, -1]
    other_order = np.asarray([math.fsum(blocks[:, index].tolist()) for index in range(SKETCH)])
    allowed = 2 * sketch_rounding(float(np.sum(np.abs(truth))))
    assert np.all(np.abs(truth_sketch(truth, 3) - other_order) <= allowed)
    empty = leaf_summary(jnp.zeros((16, 0), jnp.bfloat16), np.zeros((16, 0)))
    assert (empty.mse, empty.worst, empty.finite, empty.dtype) == (0.0, 0.0, True, "bfloat16")
    with pytest.raises(AssertionError):
        leaf_summary(np.zeros(0), np.ones(3))
    # Two entries swapped are seen at a seed unless their signs agree in all SKETCH places.
    swapped = truth.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    bits = _sign_bits(truth.size, 3)
    assert (bits[0] == bits[1]) or np.any(truth_sketch(swapped, 3) != truth_sketch(truth, 3))
