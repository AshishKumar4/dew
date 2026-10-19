"""A published packing plan reproduces memory packing and stays indexable."""

import itertools

import jax
import numpy as np
import pytest

from dew.data.cache import Cache
from dew.data.compile import compile, publish
from dew.data.dataset import DataPartition, Loading
from dew.data.pipeline import Pipeline
from dew.data.sources.memory import Memory
from dew.data.text import Chunks, Tokenize

READ = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)


SIZES = (3, 7, 18, 2, 4, 12, 8, 1, 15, 6)
"""Documents of these many characters, three longer than a 9-token window."""


def test_published_first_fit_matches_memory_windows_and_resumes_across_shares(tmp_path):
    source = Memory(tuple({"text": "x" * size} for size in SIZES))
    prefix = Pipeline(source).map(Tokenize())
    policy = prefix.flat_map(Chunks(9)).pack(9, bins=3)
    generation = publish(policy, Cache(str(tmp_path / "cache"), spool_bytes=4 * 1024**2,
                                      shard_bytes=256 * 1024), read="indexed", loading=READ)
    warm = compile(Pipeline(source).cache(generation).pack(9, bins=3).shuffle(3).repeat().batch(4),
                   read="indexed", seed=3, loading=READ)
    cold = compile(policy.shuffle(3).repeat().batch(4), read="memory", seed=3, loading=READ)
    assert warm.plan.lowering == "global"
    assert any(name.endswith("accepted") for name in warm.plan.derived)
    left, right = warm.train(DataPartition()), cold.train(DataPartition())
    try:
        read = list(itertools.islice(left, 4))
        for ours, old in zip(read, itertools.islice(right, 4), strict=True):
            assert jax.tree.structure(ours) == jax.tree.structure(old)
            jax.tree.map(np.testing.assert_array_equal, ours, old)
        # Sixteen windows hold the plan's every one: each batch declares the most documents one
        # holds, which the memory read declares alike, not the window's length.
        assert warm.records <= 16
        assert len({batch["text_coordinates"].max_segments for batch in read}) == 1
        most = max(int(batch["text_coordinates"].segments.max()) for batch in read)
        assert read[0]["text_coordinates"].max_segments == most < 9
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
    # The shares' rows interleave into the whole batch's, leaf by leaf (a window's coordinates too).
    jax.tree.map(lambda first, second, whole: np.testing.assert_array_equal(
        np.stack([first, second], axis=1).reshape(whole.shape), whole), shares[0], shares[1], expected)


def test_a_recipe_given_a_published_packing_reads_the_windows_its_memory_read_packs(tmp_path):
    """A typed recipe given a generation published from its own windows and packing reads that
    plan, shuffled as its default order: the same batches, leaf by leaf, as the recipe packing in
    memory, and a resume at a saved position reads on as the uninterrupted run does."""
    from dew.config.data import TextData, loader

    source = Memory(tuple({"text": "x" * size} for size in SIZES))
    recipe = TextData(source=source, seq_len=8, pack="first-fit", packing_bins=3, eos=False,
                      token_holdout=None)
    train, _, _ = recipe.pipelines()
    published = publish(train, Cache(str(tmp_path / "cache")), read="indexed", loading=READ)
    warm = loader(recipe, batch=2, seed=3, cache=published, read="indexed", loading=READ)
    cold = loader(recipe, batch=2, seed=3, read="memory", loading=READ)
    assert "packing" in warm.plan.manifest.source["sources"][0]
    left, right = warm.train(DataPartition()), cold.train(DataPartition())
    try:
        for ours, packed in zip(itertools.islice(left, 4), itertools.islice(right, 4), strict=True):
            assert jax.tree.structure(ours) == jax.tree.structure(packed)
            jax.tree.map(np.testing.assert_array_equal, ours, packed)
        saved = left.get_state()
        expected = next(left)
    finally:
        left.close()
        right.close()
    resumed = warm.train(DataPartition())
    try:
        resumed.set_state(saved)
        jax.tree.map(np.testing.assert_array_equal, next(resumed), expected)
    finally:
        resumed.close()


def test_a_pack_places_documents_whole_and_refuses_one_its_window_cannot_hold(tmp_path):
    """Without a chunk step before it, a first-fit pack refuses a document longer than its
    window where a memory read packs it and where its plan is published; a plan published
    before packs placed documents whole, which cut them, is read only with the chunk step
    that cuts them alike."""
    from dew.data.analysis import _cache_manifest
    from dew.data.compile import _prepared
    from dew.data.packing import plan_name, plan_windows
    from dew.data.pipeline import Mixture, Pack, Phases

    source = Memory(tuple({"text": "x" * size} for size in SIZES))
    prefix = Pipeline(source).map(Tokenize())
    with pytest.raises(ValueError):
        compile(prefix.pack(9, bins=3).repeat().batch(4), read="memory", loading=READ)

    def cache(name: str) -> Cache:
        return Cache(str(tmp_path / name), spool_bytes=4 * 1024**2, shard_bytes=256 * 1024)

    with pytest.raises(ValueError):
        publish(prefix.pack(9, bins=3), cache("refused"), read="indexed", loading=READ)
    pack = Pack(9, bins=3)

    def cut(published):
        """The plan publications built before packs placed documents whole."""
        lengths = published.source().lengths()
        assert lengths is not None
        return {plan_name(pack, field): value for field, value in plan_windows(lengths, pack).items()}

    filled = cache("cut")
    publish(prefix, filled, read="indexed", loading=READ)
    (prepared,) = _prepared((prefix,), "indexed", provision=False)
    pinned = prepared.source
    assert not isinstance(pinned, (Mixture, Phases))
    generation = filled.publish(_cache_manifest(pinned, prepared.stages, "indexed"), pinned, plan=cut)
    with pytest.raises(ValueError):
        compile(Pipeline(source).cache(generation).pack(9, bins=3).repeat().batch(4), read="indexed",
                loading=READ)
    chunked = compile(Pipeline(source).cache(generation).flat_map(Chunks(9)).pack(9, bins=3).repeat()
                      .batch(4), read="indexed", loading=READ)
    cold = compile(prefix.flat_map(Chunks(9)).pack(9, bins=3).repeat().batch(4), read="memory", loading=READ)
    left, right = chunked.train(DataPartition()), cold.train(DataPartition())
    try:
        for ours, old in zip(itertools.islice(left, 4), itertools.islice(right, 4), strict=True):
            np.testing.assert_array_equal(ours["text"], old["text"])
    finally:
        left.close()
        right.close()
