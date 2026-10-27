"""The sketch a recording keeps of each float64 truth leaf (tests/reference_error.py `truth_sketch`): each
version's definition, the dense one's bit for bit, the structured one's near exactly, and the structured
one's moves where the truth moves."""

import math

import numpy as np
import pytest


@pytest.mark.parametrize("size", [1, 63, 64, 65, (1 << 16) + 70])
def test_a_truths_sketch_is_its_signed_block_sums_summed_exactly(size):
    """The dense sketch (version 1, a recording's that names none) is its definition, bit for bit: each
    entry times each of its SKETCH signs (its sign word's bits), summed in blocks of SKETCH_BLOCK in order
    (a block past the end padded with zeros), the blocks' sums summed exactly, over chunks of 2^16 entries;
    at a block's and a chunk's edges, with signed zeros and subnormals among the entries, so a recorded
    sketch stays a recomputed truth's."""
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
    np.testing.assert_array_equal(truth_sketch(truth, 5, version=1), defined)


def _structured_signs(size: int, seed: int) -> np.ndarray:
    """The structured sketch's signs `[size, SKETCH]` at `seed`, from its definition: an entry's own sign
    (its bit of the seed's first stream) times its column's and its row's of the grid of SKETCH_COLUMNS
    columns, or of one row a smaller leaf (drawn after)."""
    from reference_error import SKETCH, SKETCH_COLUMNS

    columns = min(SKETCH_COLUMNS, size)
    rows = -(-size // columns)
    rng = np.random.default_rng([0x5E7C4, 2, size, seed])
    own = np.unpackbits(np.frombuffer(rng.bytes(-(-size // 8)), np.uint8), bitorder="little")[:size] * 2.0 - 1
    across = rng.integers(0, 2, (columns, SKETCH), dtype=np.int8) * 2.0 - 1
    down = rng.integers(0, 2, (rows, SKETCH), dtype=np.int8) * 2.0 - 1
    place = np.arange(size)
    return own[:, None] * across[place % columns] * down[place // columns]


@pytest.mark.parametrize("size", [1, 7, 4095, 4096, 4097, 3 * 4096 + 5, 70 * 4096 + 3])
def test_a_structured_sketch_is_its_signed_sums_near_exactly(size):
    """The structured sketch (version 2) is the exact sum of each entry times its signs, rounded once, but
    for at most 2^-60 of the entries' magnitudes its last level leaves: at a grid row's and a chunk's edges,
    with signed zeros, subnormals and magnitudes over six hundred binades among the entries."""
    from reference_error import SKETCH, truth_sketch

    rng = np.random.default_rng(size)
    truth = rng.normal(size=size) * 2.0 ** rng.integers(-300, 300, size)
    truth[::7] = -0.0
    truth[1::11] = 2.0 ** -1060
    signs = _structured_signs(size, 5)
    exact = np.asarray([math.fsum((truth * signs[:, index]).tolist()) for index in range(SKETCH)])
    allowed = np.abs(exact) * np.finfo(np.float64).eps / 2 + math.ldexp(float(np.sum(np.abs(truth))), -60)
    assert np.all(np.abs(truth_sketch(truth, 5) - exact) <= allowed)


def test_a_structured_sketch_moves_where_its_truth_does():
    """Two entries swapped, in one row of the grid, in one column or anywhere, two rows swapped, and one entry
    moved by a millionth of its size each move the structured sketch at every one of eight seeds, as the
    dense sketch's would."""
    from reference_error import SKETCH_COLUMNS, truth_sketch

    truth = np.random.default_rng(3).normal(size=(8, SKETCH_COLUMNS))
    moves = {"in a row": ((2, 5), (2, 900)), "in a column": ((1, 77), (6, 77)),
             "anywhere": ((0, 3), (7, 4000))}
    for seed in range(8):
        held = truth_sketch(truth, seed)
        for (first, second) in moves.values():
            wrong = truth.copy()
            wrong[first], wrong[second] = truth[second], truth[first]
            assert np.any(truth_sketch(wrong, seed) != held)
        wrong = truth[[1, 0, *range(2, 8)]]
        assert np.any(truth_sketch(wrong, seed) != held)
        wrong = truth.copy()
        wrong[4, 9] *= 1 + 1e-6
        assert np.any(truth_sketch(wrong, seed) != held)
