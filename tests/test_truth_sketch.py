"""The sketch a recording keeps of each float64 truth leaf (tests/reference_error.py `truth_sketch`): its
definition, within the rounding any host may take it at (`sketch_rounding`)."""

import math

import numpy as np
import pytest

DRAWN = {
    "float64": lambda rng, size: rng.standard_normal(size) * 10.0 ** rng.uniform(-30, 30, size),
    "float32": lambda rng, size: rng.standard_normal(size).astype(np.float32),
    "float16": lambda rng, size: rng.standard_normal(size).astype(np.float16),
    "int32": lambda rng, size: rng.integers(-(1 << 31), (1 << 31) - 1, size, dtype=np.int32),
    "bool": lambda rng, size: rng.integers(0, 2, size).astype(bool),
}
"""A truth of each dtype the sketch takes (it reads every one as float64): a float64 truth's magnitudes
across 60 decades, so a block's rounding is its largest entries' and the others' vanish in it."""


@pytest.mark.parametrize("dtype", DRAWN)
@pytest.mark.parametrize("size", [1, 63, 64, 65, 3000, (1 << 16) + 70])
def test_a_truths_sketch_is_its_signed_sums_within_any_hosts_rounding(size, dtype):
    """The sketch a recording keeps is its definition, each entry times each of its SKETCH signs (its sign
    word's bits, +1 where set), every block of SKETCH_BLOCK summed (a block past the end padded with
    zeros) and the blocks' sums summed exactly: within `sketch_rounding` of the signed products' exact
    sum, the most any order of a block's sum may round by, which the gate allows each of two sketches
    (`indexed_recording`). At a block's and a host chunk's edges, with signed zeros and subnormals among
    a floating truth's entries, so a recorded sketch stays a recomputed truth's."""
    from reference_error import SKETCH, _sign_bits, sketch_rounding, truth_sketch

    rng = np.random.default_rng(size)
    truth = DRAWN[dtype](rng, size)
    if np.issubdtype(truth.dtype, np.floating):
        truth[::7] = -0.0
        truth[1::11] = np.finfo(truth.dtype).smallest_subnormal
    values = truth.astype(np.float64)
    signs = ((_sign_bits(size, 5)[:, None] >> np.arange(SKETCH, dtype=np.uint64)) & 1) * 2.0 - 1
    exact = np.asarray([math.fsum(column) for column in (values[:, None] * signs).T.tolist()])
    sketch = truth_sketch(truth, 5)
    assert sketch.shape == (SKETCH,)
    assert np.all(np.abs(sketch - exact) <= sketch_rounding(math.fsum(np.abs(values).tolist())))


def test_a_devices_block_sums_are_the_hosts_where_each_block_sums_exactly():
    """The blocks' sums a GPU takes (`_device_partials`, here on the CPU's device) are the host's
    (`_host_partials`) bit for bit where every block sums exactly in any order (dyadic entries, signed
    zeros among them), across a device chunk's edge and a last block padded with zeros: the same
    products in the same blocks, so on any truth the two part by each block's own rounding alone, within
    gamma(SKETCH_BLOCK) of its magnitudes."""
    from reference_error import DEVICE_CHUNK, _device_partials, _host_partials, _sign_bits

    size = DEVICE_CHUNK + (1 << 16) + 70
    truth = np.random.default_rng(size).integers(-(1 << 20), 1 << 20, size) / 1024.0
    truth[::7] = -0.0
    bits = _sign_bits(size, 5)
    np.testing.assert_array_equal(_device_partials(truth, bits), _host_partials(truth, bits))
