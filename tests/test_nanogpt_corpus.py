"""A token directory nanoGPT wrote, read by Dew as nanoGPT reads it
(`tools/nanogpt_reference.py`): its shakespeare_char prepare.py's files and
its train.py `get_batch`'s batches, both run as published."""

from pathlib import Path

import numpy as np
import pytest

from dew.data.sources.text import TokenBytes, TokenWindowSource, token_corpus

NANOGPT = Path(__file__).resolve().parent / "fixtures" / "nanogpt"
BATCHES = np.load(NANOGPT / "batch.npz")


@pytest.mark.parametrize("split", ["train", "val"])
def test_a_nanogpt_directory_reads_the_windows_get_batch_reads(split):
    """The directory has nanoGPT's train.bin, val.bin and no meta.json, so
    the ids are its uint16. A window of `block + 1` ids at any offset is
    `get_batch`'s x and y there: with a stride of one, Dew's record i is
    the window at offset i and there are as many as `get_batch` can draw
    from, one per offset short of the last block."""
    train, val = token_corpus(str(NANOGPT), "nanogpt")
    tokens = train if split == "train" else val
    assert isinstance(tokens, TokenBytes) and tokens.dtype == np.dtype("<u2")
    block = int(BATCHES["block"])
    windows = TokenWindowSource(tokens, block, stride=1)
    assert len(windows) == len(tokens) - block
    for offsets, xs, ys in zip(BATCHES[f"{split}/ix"], BATCHES[f"{split}/x"], BATCHES[f"{split}/y"],
                               strict=True):
        for offset, x, y in zip(offsets, xs, ys, strict=True):
            window = windows[int(offset)]["text"]
            np.testing.assert_array_equal(window[:-1], x)
            np.testing.assert_array_equal(window[1:], y)
