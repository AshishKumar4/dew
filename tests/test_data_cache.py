"""Artifact coverage, crash boundaries, duplicate records and bounded admission."""

import hashlib
import json
from dataclasses import dataclass
from uuid import uuid4

import grain.python as grain
import pytest

from dew.data.cache import Cache
from dew.data.sources import Memory
from dew.data.storage import Store


@dataclass(frozen=True)
class Manifest:
    canonical: bytes = b'{"revision":"test","source":"memory"}'
    clean: bool = True
    exact: bool = True

    @property
    def key(self):
        return hashlib.sha256(self.canonical).hexdigest()


def _cache(**kwargs):
    return Cache(f"memory://artifact-{uuid4().hex}", **kwargs)


def test_expansion_publication_certifies_empty_and_nonempty_inputs():
    source = Memory([{"id": index} for index in range(5)])
    cache, key = _cache(), Manifest()

    def expand(index, value):
        return tuple({"id": value["id"], "part": part} for part in range(index))

    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), key, source, transform=expand)
    cold = [rows[index] for index in range(len(rows))]
    rows.close()
    generation = cache.publish(key, source)
    assert len(generation) == 10
    assert generation.outputs(0) == ()
    assert [generation.outputs(index) for index in range(5)] == cold
    assert [generation[index] for index in range(len(generation))] == [row for group in cold for row in group]
    assert cache.generation(key).url == generation.url


def test_missing_input_completion_refuses_publication():
    source = Memory([{"id": index} for index in range(3)])
    cache, key = _cache(), Manifest()
    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), key, source)
    assert rows[0] == {"id": 0}
    assert rows[2] == {"id": 2}
    rows.close()
    with pytest.raises(ValueError, match="1 missing"):
        cache.publish(key, source)
    assert cache.generation(key) is None


def test_unsealed_writer_cannot_certify_coverage():
    source = Memory([{"id": 0}, {"id": 1}])
    cache, key = _cache(shard_bytes=1), Manifest()
    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), key, source)
    rows[0]
    with pytest.raises(ValueError, match="missing"):
        cache.publish(key, source)
    rows[1]
    rows.close()
    assert len(cache.publish(key, source)) == 2


def test_conflicting_duplicate_writers_refuse_publication():
    source = Memory([{"id": 0}])
    cache, key = _cache(), Manifest()
    first = cache.mapped(grain.MapDataset.source([{"id": 0}]), key, source)
    second = cache.mapped(grain.MapDataset.source([{"id": 999}]), key, source)
    first[0], second[0]
    first.close()
    second.close()
    with pytest.raises(ValueError, match="conflicting cache duplicate"):
        cache.publish(key, source)


def test_corrupt_shard_never_reaches_current():
    source = Memory([{"id": 0}])
    cache, key = _cache(), Manifest()
    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), key, source)
    rows[0]
    rows.close()
    fs, path = Store(cache.url + "/" + key.key + "/shards").resolve()
    fs.pipe_file(fs.find(path)[0], b"corrupt")
    with pytest.raises(OSError, match="corrupt"):
        cache.publish(key, source)
    assert cache.generation(key) is None


def test_corrupt_committed_index_is_an_error_without_fallback():
    source = Memory([{"id": 0}])
    cache, key = _cache(), Manifest()
    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), key, source)
    rows[0]
    rows.close()
    generation = cache.publish(key, source)
    manifest = json.loads(Store(generation.url + "/manifest.json").read())
    page = next(iter(manifest["pages"].values()))["path"]
    Store(generation.url + "/" + page).write(b"wrong index")
    with pytest.raises(OSError, match="corrupt"):
        generation.outputs(0)


def test_full_spool_stops_admitting_and_training_continues():
    source = Memory([{"bytes": b"x" * 2048}])
    cache, key = _cache(spool_bytes=256, shard_bytes=128), Manifest()
    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), key, source)
    assert rows[0] == source.records("indexed")[0]
    assert not rows.writer.admitting
    rows.close()
    with pytest.raises(ValueError, match="missing"):
        cache.publish(key, source)


def test_cache_identity_and_clean_revision_are_required():
    source = Memory([{"id": 0}])
    rows = grain.MapDataset.source(source.records("indexed"))
    with pytest.raises(ValueError, match="clean Dew revision"):
        _cache().mapped(rows, Manifest(clean=False), source)
    with pytest.raises(ValueError, match="stable identity"):
        _cache().mapped(rows, Manifest(exact=False), source)
    assert Manifest().key != Manifest(b'{"revision":"changed"}').key


def test_stream_completions_use_physical_shards_and_resume_without_rekeying():
    source = Memory([{"id": index} for index in range(8)], shard_rows=2)
    cache, key = _cache(), Manifest()
    manifest = source.manifest()
    reader = cache.streamed(manifest.reader, key, manifest)
    for shard in reversed(manifest.shards):
        iterator = iter(reader(shard))
        assert next(iterator)["id"] == shard.start
        state = iterator.get_state()
        remaining = list(iterator)
        resumed = iter(reader(shard))
        resumed.set_state(state)
        assert list(resumed) == remaining
    generation = cache.publish(key, source)
    for shard in manifest.shards:
        assert generation.outputs((shard.id, 0)) == ({"id": shard.start},)


@dataclass(frozen=True)
class Draw(grain.RandomMapTransform):
    def random_map(self, element, rng):
        return {**element, "draw": int(rng.integers(0, 2**31))}


def test_cold_warm_and_resumed_caches_keep_per_epoch_randomness():
    source = Memory([{"id": index} for index in range(8)])
    cache, key = _cache(), Manifest()
    parent = grain.MapDataset.source(source.records("indexed"))
    cold = cache.mapped(parent, key, source)

    def epochs(rows):
        return rows.repeat(2).seed(31).random_map(Draw()).to_iter_dataset()

    reader = iter(epochs(cold))
    first = [next(reader) for _ in range(5)]
    state = reader.get_state()
    rest = list(reader)
    reader.close()
    cold.close()
    expected = first + rest
    assert [row["id"] for row in expected[:8]] == [row["id"] for row in expected[8:]]
    assert [row["draw"] for row in expected[:8]] != [row["draw"] for row in expected[8:]]
    generation = cache.publish(key, source)
    warm = generation.mapped(parent, key, source)
    assert list(epochs(warm)) == expected
    resumed = iter(epochs(warm))
    resumed.set_state(state)
    assert list(resumed) == rest
    resumed.close()


def test_cache_refuses_stochastic_prefix_by_step_name():
    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline

    source = Memory([{"id": index} for index in range(8)])
    with pytest.raises(ValueError, match=r"cache prefix.*Draw"):
        compile(Pipeline(source).random_map(Draw()).cache(_cache()).batch(2))
