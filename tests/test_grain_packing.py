"""Packed windows against grain's own first-fit packer
(`grain.experimental.FirstFitPackIterDataset`), which `first_fit` plans as."""

import grain
import numpy as np
import pytest
from grain.experimental import FirstFitPackIterDataset

from dew.data.tokens import DocumentChunks, PackedWindows


@pytest.mark.parametrize("window, bins", [(8, 1), (8, 3), (16, 4), (5, 2)])
def test_packed_windows_are_grains_first_fit_windows(window, bins):
    """Documents from one token to well past the window, a few exactly the
    window and one empty, with ids and roles: grain's packer, given the
    chunks `DocumentChunks` cuts in file order and its bins emitted in
    order, writes window for window what `PackedWindows` reads by index,
    the ids, the roles and the segment ids and positions. Grain writes a
    segment-id and position pair per feature; the roles' pair is the ids'
    pair, which is why Dew writes the one."""
    rng = np.random.default_rng(window * 10 + bins)
    lengths = np.concatenate([rng.integers(1, 3 * window, 60), [window, window, 0, 1]])
    documents = [{"text": (np.arange(length) + 1000 * index).astype(np.int32),
                  "text_roles": np.full(length, index % 4, np.int32)} for index, length in enumerate(lengths)]
    source = grain.MapDataset.source(documents)
    packed = PackedWindows(source, lengths, window, bins, "corpus")
    chunks = DocumentChunks(source, lengths, window)
    reference = list(FirstFitPackIterDataset(
        grain.MapDataset.source([chunks[index] for index in range(len(chunks))]).to_iter_dataset(),
        length_struct={"text": window, "text_roles": window}, num_packing_bins=bins, shuffle_bins=False))
    assert len(packed) == len(reference)
    for index, expected in enumerate(reference):
        got = packed[index]
        for key in ("text", "text_roles", "text_segment_ids", "text_positions"):
            np.testing.assert_array_equal(got[key], expected[key], err_msg=f"window {index} {key}")
        for key in ("segment_ids", "positions"):
            np.testing.assert_array_equal(expected[f"text_roles_{key}"], expected[f"text_{key}"])
