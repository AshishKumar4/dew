"""A published packing plan reproduces memory packing and stays indexable."""

import itertools

import numpy as np
import pytest

from dew.data.cache import Cache
from dew.data.compile import compile, publish
from dew.data.dataset import DataPartition, Loading
from dew.data.pipeline import Pipeline
from dew.data.sources.memory import Memory
from dew.data.steps.text import Tokenize
from dew.data.text import chunk

READ = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)


SIZES = (3, 7, 18, 2, 4, 12, 8, 1, 15, 6)
"""Documents of these many characters, three longer than a 9-token window."""


def test_published_first_fit_matches_memory_windows_and_resumes_across_shares(tmp_path):
    source = Memory(tuple({"text": "x" * size} for size in SIZES))
    prefix = Pipeline(source).map(Tokenize())
    policy = prefix.flat_map(chunk(9)).pack(9, bins=3)
    generation = publish(policy, Cache(str(tmp_path / "cache"), spool_bytes=4 * 1024**2,
                                      shard_bytes=256 * 1024), read="indexed", loading=READ)
    warm = compile(Pipeline(source).cache(generation).flat_map(chunk(9)).pack(9, bins=3).shuffle(3).repeat()
                   .batch(4), read="indexed", seed=3, loading=READ)
    cold = compile(policy.shuffle(3).repeat().batch(4), read="memory", seed=3, loading=READ)
    assert warm.plan.lowering == "global"
    assert any(name.endswith("accepted") for name in warm.plan.derived)
    left, right = warm.train(DataPartition()), cold.train(DataPartition())
    try:
        for ours, old in zip(itertools.islice(left, 4), itertools.islice(right, 4), strict=True):
            assert ours.keys() == old.keys()
            for field in ours:
                np.testing.assert_array_equal(ours[field], old[field])
        saved = left.get_state()
        expected = next(left)
    finally:
        left.close()
        right.close()
    shares = []
    for index in range(2):
        reader = warm.train(DataPartition(index, 2))
        reader.set_state(saved)
        shares.append(next(reader))
        reader.close()
    for field in expected:
        joined = np.stack([shares[0][field], shares[1][field]], axis=1).reshape(expected[field].shape)
        np.testing.assert_array_equal(joined, expected[field])


def test_a_pack_places_documents_whole_and_refuses_one_its_window_cannot_hold(tmp_path):
    """Without a chunk step before it, a first-fit pack refuses a document longer than its
    window, where a memory read packs it, where a plan is published, and where a plan
    published with the chunk step is read without it: none cuts a document on its own."""
    source = Memory(tuple({"text": "x" * size} for size in SIZES))
    prefix = Pipeline(source).map(Tokenize())
    with pytest.raises(ValueError):
        compile(prefix.pack(9, bins=3).repeat().batch(4), read="memory", loading=READ)
    def cache(name: str) -> Cache:
        return Cache(str(tmp_path / name), spool_bytes=4 * 1024**2, shard_bytes=256 * 1024)

    with pytest.raises(ValueError):
        publish(prefix.pack(9, bins=3), cache("whole"), read="indexed", loading=READ)
    generation = publish(prefix.flat_map(chunk(9)).pack(9, bins=3), cache("chunked"), read="indexed",
                         loading=READ)
    with pytest.raises(ValueError):
        compile(Pipeline(source).cache(generation).pack(9, bins=3).repeat().batch(4), read="indexed",
                loading=READ)
