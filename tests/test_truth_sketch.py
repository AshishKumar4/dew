"""The sketch a recording keeps of each float64 truth leaf (tests/reference_error.py `truth_sketch`): its
definition, bit for bit."""

import math

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
