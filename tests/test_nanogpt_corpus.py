"""The token stream nanoGPT wrote, exported as a Dew token store and read as nanoGPT reads it
(`tools/nanogpt_reference.py`): its shakespeare_char prepare.py's files and its train.py
`get_batch`'s batches, both run as published."""

import itertools
from pathlib import Path

import numpy as np
from token_support import write_token_corpus

from dew.data import Loading, presets
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.sources.tokens import TokenFiles

NANOGPT = Path(__file__).resolve().parent / "fixtures" / "nanogpt"
BATCHES = np.load(NANOGPT / "batch.npz")


def test_a_nanogpt_stream_exported_as_a_token_store_reads_the_windows_get_batch_reads(tmp_path):
    """nanoGPT's train.bin holds its ids as uint16, a stream with no document ends, which an export
    commits as a token store. With a stride of one, a training epoch is the window of `block + 1`
    ids at every offset `get_batch` can draw from, one per offset short of the last block, and
    `get_batch`'s x and y at an offset are that window's first and last `block` ids."""
    block = int(BATCHES["block"])
    ids = np.fromfile(NANOGPT / "train.bin", np.uint16).astype(np.int32)
    every = np.lib.stride_tricks.sliding_window_view(ids, block + 1)[:len(ids) - block]
    for offsets, xs, ys in zip(BATCHES["train/ix"], BATCHES["train/x"], BATCHES["train/y"], strict=True):
        for offset, x, y in zip(offsets, xs, ys, strict=True):
            np.testing.assert_array_equal(every[int(offset)][:-1], x)
            np.testing.assert_array_equal(every[int(offset)][1:], y)

    # 16 rows a batch divides the epoch, so its batches are the epoch exactly.
    store = write_token_corpus(tmp_path / "shakespeare_char", ids).train
    windows = TokenFiles(store).shuffle(0).repeat().pipe(presets.token_windows, seq_len=block, stride=1,
                                                          batch=16)
    data = compile(windows, read="memory", loading=Loading(workers=0))
    assert data.records == len(every) and data.records % 16 == 0
    epoch = data.train(DataPartition())
    try:
        read = np.concatenate([batch["text"] for batch in itertools.islice(epoch, data.records // 16)])
    finally:
        epoch.close()
    assert sorted(row.tobytes() for row in read) == sorted(row.tobytes() for row in every)
