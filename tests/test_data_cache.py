"""Artifact coverage, crash boundaries, duplicate records and bounded admission."""

import contextlib
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from uuid import uuid4

import fsspec
import grain.python as grain
import numpy as np
import pytest
from storage_support import moto_bucket

from dew.data.cache import Abandoned, Cache
from dew.data.pipeline import Rejected
from dew.data.plan import Manifest, StepIdentity
from dew.data.sources import Memory
from dew.data.storage import Store


def _key(source: object = "memory", *, revision: str = "test", clean: bool = True,
         exact: bool = True) -> Manifest:
    """A dependency manifest of fixed fields, keying a cache of `source`'s rows at `revision`: a
    dirty checkout's (`clean=False`), or one whose step has no stable identity (`exact=False`)."""
    steps = () if exact else (StepIdentity("tests.unstable", "<lambda>", exact=False),)
    return Manifest({"store": source}, steps, revision=revision, clean=clean)


def _cache(**kwargs):
    return Cache(f"memory://artifact-{uuid4().hex}", **kwargs)


@dataclass
class CountedStep:
    """A fixed map or changing fetch, with per-row call counts."""

    volatile: bool = False
    reject_first: bool = False
    permanent: bool = True
    calls: dict[int, int] = field(default_factory=dict, repr=False)

    def __call__(self, row):
        row_id = row["id"]
        attempt = self.calls.get(row_id, 0) + 1
        self.calls[row_id] = attempt
        if self.reject_first and row_id == 0 and attempt == 1:
            return Rejected("not_ready", permanent=self.permanent)
        return {**row, "value": attempt if self.volatile else row_id + 10}


def _cached_passes(step, directory, records=8, read="indexed"):
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition, Loading

    source = Memory([{"id": index} for index in range(records)])
    prefix = (source.filter_map(step, identity="counted@1") if step.reject_first
              else source.map(step, identity="counted@1"))
    chain = prefix.cache(str(directory))
    if records > 2:
        chain = chain.take(2)
    chain = chain.repeat(2).batch(1)
    loader = compile(chain, read=read, loading=Loading(workers=0, threads=1, read_buffer=1))
    reader = loader.train(DataPartition())
    try:
        return [(int(batch["id"][0]), int(batch["value"][0])) for batch in reader]
    finally:
        reader.close()


@pytest.mark.parametrize("records", [2, 8])
@pytest.mark.parametrize("read", ["indexed", "streamed"])
def test_volatile_cache_reruns_prefix_on_later_passes(tmp_path, records, read):
    step = CountedStep(volatile=True)
    # A reader may run a later pass's step ahead of delivering an earlier one: each row's two
    # outcomes are its two calls, in whichever order they ran.
    assert sorted(_cached_passes(step, tmp_path, records, read)) == [(0, 1), (0, 2), (1, 1), (1, 2)]
    assert step.calls[0] == step.calls[1] == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("permanent", [True, False])
@pytest.mark.parametrize("read", ["indexed", "streamed"])
def test_volatile_cache_retries_rejected_rows_on_later_passes(tmp_path, permanent, read):
    step = CountedStep(volatile=True, reject_first=True, permanent=permanent)
    assert sorted(_cached_passes(step, tmp_path, read=read)) == [(0, 2), (1, 1), (1, 2), (2, 1)]
    assert step.calls[0] == step.calls[1] == 2
    assert not list(tmp_path.iterdir())


def test_deterministic_cache_serves_prefix_on_later_passes(tmp_path):
    step = CountedStep()
    assert _cached_passes(step, tmp_path) == [(0, 10), (1, 11), (0, 10), (1, 11)]
    assert step.calls[0] == step.calls[1] == 1


def test_expansion_publication_certifies_empty_and_nonempty_inputs():
    from dew.data.sources.codec import encode

    source = Memory([{"id": index} for index in range(5)])
    cache, key = _cache(), _key()

    def expand(index, value):
        return tuple({"id": value["id"], "part": part} for part in range(index))

    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), key, transform=expand)
    cold = [rows[index] for index in range(len(rows))]
    rows.finish()
    rows.close()
    legacy = {}
    for index, outputs in enumerate(cold):
        digest = hashlib.sha256()
        for value in outputs:
            payload = encode(value)
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
        legacy[str(index)] = digest.hexdigest()
    fs, path = Store(cache.url + "/" + key.key + "/segments").resolve()
    for location in fs.find(path):
        if location.endswith("/seal.json"):
            continue
        for completion in json.loads(fs.cat_file(location))["completions"]:
            assert completion["payload_sha256"] == completion["sha256"] == legacy[completion["id"]]
    generation = cache.publish(key, source)
    assert len(generation) == 10
    assert generation.outputs(0) == ()
    assert [generation.outputs(index) for index in range(5)] == cold
    assert [generation[index] for index in range(len(generation))] == [row for group in cold for row in group]
    assert cache.generation(key).url == generation.url


def test_missing_input_completion_refuses_publication():
    """A writer that never read row 1 leaves it uncovered and publication is
    refused; a second writer that reads it completes the generation."""
    source = Memory([{"id": index} for index in range(3)])
    cache, key = _cache(), _key()
    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), key)
    assert rows[0] == {"id": 0}
    assert rows[2] == {"id": 2}
    rows.finish()
    rows.close()
    with pytest.raises(ValueError):
        cache.publish(key, source)
    assert cache.generation(key) is None
    rest = cache.mapped(grain.MapDataset.source(source.records("indexed")), key)
    assert rest[1] == {"id": 1}
    rest.finish()
    rest.close()
    assert len(cache.publish(key, source)) == 3


def test_unsealed_writer_cannot_certify_coverage():
    source = Memory([{"id": 0}, {"id": 1}])
    cache, key = _cache(shard_bytes=1), _key()
    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), key)
    rows[0]
    with pytest.raises(ValueError):
        cache.publish(key, source)
    rows[1]
    rows.finish()
    rows.close()
    assert len(cache.publish(key, source)) == 2


@pytest.mark.parametrize("second_row", [{"id": 0}, {"id": 999}], ids=["agreeing", "conflicting"])
def test_conflicting_duplicate_writers_refuse_publication(second_row):
    """Two writers of one row publish it where they wrote the same row, and
    are refused where they wrote different ones."""
    source = Memory([{"id": 0}])
    cache, key = _cache(), _key()
    first = cache.mapped(grain.MapDataset.source([{"id": 0}]), key)
    second = cache.mapped(grain.MapDataset.source([second_row]), key)
    first[0], second[0]
    first.finish()
    first.close()
    second.finish()
    second.close()
    if second_row == {"id": 0}:
        assert cache.publish(key, source)[0] == {"id": 0}
        return
    with pytest.raises(ValueError):
        cache.publish(key, source)


def test_corrupt_shard_never_reaches_current():
    source = Memory([{"id": 0}])
    cache, key = _cache(), _key()
    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), key)
    rows[0]
    rows.finish()
    rows.close()
    fs, path = Store(cache.url + "/" + key.key + "/shards").resolve()
    shard = fs.find(path)[0]
    intact = fs.cat_file(shard)
    fs.pipe_file(shard, b"corrupt")
    with pytest.raises(OSError):
        cache.publish(key, source)
    assert cache.generation(key) is None
    fs.pipe_file(shard, intact)
    assert cache.publish(key, source)[0] == {"id": 0}


def test_corrupt_committed_index_is_an_error_without_fallback():
    source = Memory([{"id": 0}])
    cache, key = _cache(), _key()
    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), key)
    rows[0]
    rows.finish()
    rows.close()
    generation = cache.publish(key, source)
    manifest = json.loads(Store(generation.url + "/manifest.json").read())
    page = Store(generation.url + "/" + next(iter(manifest["pages"].values()))["path"])
    intact = page.read()
    page.write(b"wrong index")
    with pytest.raises(OSError):
        generation.outputs(0)
    page.write(intact)
    assert list(cache.generation(key).outputs(0)) == [{"id": 0}]


@pytest.mark.parametrize("spool_bytes", [256, 1 << 20])
def test_full_spool_stops_admitting_and_training_continues(spool_bytes):
    """A row larger than the spool is served but not admitted, so its cache
    cannot be published; a spool that holds it publishes."""
    source = Memory([{"bytes": b"x" * 2048}])
    cache, key = _cache(spool_bytes=spool_bytes, shard_bytes=128), _key()
    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), key)
    assert rows[0] == source.records("indexed")[0]
    assert rows.writer.admitting is (spool_bytes > 2048)
    rows.finish()
    rows.close()
    if spool_bytes > 2048:
        assert len(cache.publish(key, source)) == 1
        return
    with pytest.raises(ValueError):
        cache.publish(key, source)


@pytest.mark.parametrize("refused", [{"clean": False}, {"exact": False}], ids=["dirty revision", "unstable"])
def test_cache_identity_and_clean_revision_are_required(refused):
    source = Memory([{"id": 0}])
    rows = grain.MapDataset.source(source.records("indexed"))
    _cache().mapped(rows, _key()).close()
    with pytest.raises(ValueError):
        _cache().mapped(rows, _key(**refused))
    assert _key().key != _key(revision="changed").key


def test_moto_generation_publication_checks_shards_and_concurrent_finalizers():
    import fsspec

    with moto_bucket("cache") as (bucket, options):
        fs = fsspec.filesystem("s3", **options)
        cache, key = Cache(f"s3://{bucket}/cache", storage_options=options), _key()
        source = Memory([{"id": index} for index in range(8)])
        rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), key)
        for index in range(len(rows)):
            rows[index]
        rows.finish()
        rows.close()
        start = threading.Barrier(2)

        def publish():
            from dew.data.storage import PreconditionFailed

            start.wait()
            try:
                return cache.publish(key, source)
            except PreconditionFailed as error:
                return error

        with ThreadPoolExecutor(2) as pool:
            attempts = list(pool.map(lambda _: publish(), range(2)))
        from dew.data.generation import Generation

        assert any(isinstance(attempt, Generation) for attempt in attempts)
        generation = cache.generation(key)
        assert [generation[index] for index in range(len(generation))] == [
            {"id": index} for index in range(8)
        ]
        fs.pipe_file(
            bucket + "/cache/" + key.key + "/" + next(iter(generation.manifest["shards"])), b"broken"
        )
        with pytest.raises(OSError):
            cache.publish(key, source)


def test_stream_completions_use_physical_shards_and_resume_without_rekeying():
    source = Memory([{"id": index} for index in range(8)], shard_rows=2)
    cache, key = _cache(), _key()
    manifest = source.manifest()
    reader = cache.streamed(manifest.reader, key)
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
    cache, key = _cache(), _key()
    parent = grain.MapDataset.source(source.records("indexed"))
    cold = cache.mapped(parent, key)

    def epochs(rows):
        return rows.repeat(2).seed(31).random_map(Draw()).to_iter_dataset()

    reader = iter(epochs(cold))
    first = [next(reader) for _ in range(5)]
    state = reader.get_state()
    rest = list(reader)
    reader.close()
    cold.finish()
    cold.close()
    expected = first + rest
    assert [row["id"] for row in expected[:8]] == [row["id"] for row in expected[8:]]
    assert [row["draw"] for row in expected[:8]] != [row["draw"] for row in expected[8:]]
    generation = cache.publish(key, source)
    warm = grain.MapDataset.source(generation.source().records("indexed"))
    assert list(epochs(warm)) == expected
    resumed = iter(epochs(warm))
    resumed.set_state(state)
    assert list(resumed) == rest
    resumed.close()


def test_cache_refuses_a_stochastic_prefix():
    """A random step after the cache compiles; the same step before it is refused."""
    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline

    source = Memory([{"id": index} for index in range(8)])
    compile(Pipeline(source).cache(_cache()).random_map(Draw()).batch(2))
    with pytest.raises(ValueError):
        compile(Pipeline(source).random_map(Draw()).cache(_cache()).batch(2))


def test_iterate_and_validation_read_a_published_generation_as_its_accepted_rows():
    from dew.batch import VALID_ROWS
    from dew.data.compile import iterate, publish, validation
    from dew.data.dataset import DataPartition
    from dew.data.pipeline import Pipeline

    source = Memory([{"id": index} for index in range(8)])
    even = Pipeline(source).filter(lambda row: row["id"] % 2 == 0, identity="even@1")
    warm = even.cache(publish(even, _cache(), read="memory"))
    assert [int(row["id"]) for row in iterate(warm, read="indexed")] == [0, 2, 4, 6]
    scored = validation(warm.map(lambda row: {"id": row["id"] + 10}), batch=3)(DataPartition())
    try:
        batches = list(scored)
    finally:
        scored.close()
    assert [batch["id"].tolist() for batch in batches] == [[10, 12, 14], [16, 16, 16]]
    assert batches[-1][VALID_ROWS].tolist() == [True, False, False]


def test_a_hash_holdout_publishes_the_rows_its_stream_reads():
    """A hash holdout's shard readers emit only the rows its hash keeps, so its streamed shards
    state no row counts: a cache filled from the training part's stream publishes, and its
    generation holds that part's rows, each once."""
    import numpy as np

    from dew.data.compile import iterate, publish
    from dew.data.pipeline import Pipeline

    stored = Memory([{"id": np.int32(index)} for index in range(20)], shard_rows=7)
    train, _ = Pipeline(stored).holdout(0.5)
    part = sorted(int(row["id"]) for row in iterate(train, read="streamed"))
    tagged = train.map(lambda row: {**row, "tag": row["id"] * 10}, identity="tag@1")
    generation = publish(tagged, _cache(), read="streamed")
    published = [(int(row["id"]), int(row["tag"]))
                 for row in iterate(Pipeline(generation.source()), read="indexed")]
    assert sorted(published) == [(index, index * 10) for index in part] and 0 < len(part) < 20


@pytest.mark.parametrize("changed", ["chunk overlap", "source", "fewer steps"])
def test_a_generation_reads_in_place_of_the_steps_it_ran_and_of_no_others(changed):
    """A published generation reads in place of the source and steps it was published from, as
    the rows those steps make; cached after another source or other steps it would serve rows they
    never make, so the run is refused before it reads."""
    from dew.data.compile import iterate, publish
    from dew.data.pipeline import Pipeline
    from dew.data.text import Chunks, Tokenize

    texts = [{"text": "x" * size} for size in (3, 7, 18, 2, 12)]
    source = Memory(texts)
    chunked = Pipeline(source).map(Tokenize()).flat_map(Chunks(9))
    generation = publish(chunked, _cache(), read="indexed")

    def pieces(pipeline):
        return [row["text"].tolist() for row in iterate(pipeline, read="indexed")]

    assert pieces(chunked.cache(generation)) == pieces(chunked)
    other = {
        "chunk overlap": Pipeline(source).map(Tokenize()).flat_map(Chunks(9, overlap=1)),
        "source": Pipeline(Memory(texts[::-1])).map(Tokenize()).flat_map(Chunks(9)),
        "fewer steps": Pipeline(source).map(Tokenize()),
    }[changed]
    with pytest.raises(ValueError):
        iterate(other.cache(generation), read="indexed")


def test_a_generation_of_one_holdout_part_reads_as_that_part():
    """A generation published from a holdout part holds that part's rows, so it reads in place of
    that part whole, and the other part, whose rows it does not hold, is refused it; one published
    from the store the parts select from serves both, each the rows of the inputs it keeps."""
    from dew.data.compile import iterate, publish
    from dew.data.pipeline import Pipeline

    source = Memory([{"id": index} for index in range(12)])

    def tagged(pipeline):
        return pipeline.map(lambda row: {**row, "tag": row["id"] * 10}, identity="tag@1")

    def ids(pipeline):
        return [(int(row["id"]), int(row["tag"])) for row in iterate(pipeline, read="indexed")]

    train, held = Pipeline(source).holdout(records=5)
    part = publish(tagged(train), _cache(), read="indexed")
    assert ids(tagged(train).cache(part)) == ids(tagged(train))
    assert ids(tagged(train)) == [(index, index * 10) for index in range(5, 12)]
    with pytest.raises(ValueError):
        iterate(tagged(held).cache(part), read="indexed")
    store = publish(tagged(Pipeline(source)), _cache(), read="indexed")
    served = [ids(tagged(branch).cache(store)) for branch in (train, held)]
    assert served == [ids(tagged(train)), ids(tagged(held))]


def test_a_generation_of_a_held_out_part_reads_under_the_holdouts_taken_from_that_part():
    """A generation published from a training part stands in for it under a second holdout taken
    from that part: the second holdout picks from the generation's rows as from the part's."""
    from dew.data.compile import iterate, publish
    from dew.data.pipeline import Pipeline

    source = Memory([{"id": index} for index in range(12)])

    def tagged(pipeline):
        return pipeline.map(lambda row: {**row, "tag": row["id"] * 10}, identity="tag@1")

    def ids(pipeline):
        return [(int(row["id"]), int(row["tag"])) for row in iterate(pipeline, read="indexed")]

    part, _ = Pipeline(source).holdout(records=2)
    generation = publish(tagged(part), _cache(), read="indexed")
    for cached, uncached in zip(tagged(part).cache(generation).holdout(records=1),
                                tagged(part).holdout(records=1), strict=True):
        assert ids(cached) == ids(uncached)


def test_the_prompts_preset_reads_the_generation_its_memory_read_published():
    """The prompts preset states its reward width by reading its rows, so the generation a memory
    read of its rendering published stands in for the same rendering when it reads its rows again."""
    from dew.data import presets
    from dew.data.compile import compile, publish

    # Two of the prompts are longer than 8 bytes, which the preset drops when asked to.
    prompts = Memory([{"prompt": "hi " * (1 + index % 3)} for index in range(6)])

    def rendered(batch, cache=None):
        return prompts.pipe(presets.prompts, "byte", max_prompt_len=8, long="drop", batch=batch, cache=cache)

    generation = publish(rendered(None), _cache(), read="memory")
    data = compile(rendered(2, generation), read="memory")
    assert data.plan.manifest.source["sources"][0]["store"] == "generation"
    assert data.records == compile(rendered(2), read="memory").records == 4


@dataclass(frozen=True)
class Same:
    """A deterministic step that keeps its row and notes each row it runs on."""

    calls: list = field(default_factory=list, compare=False, repr=False)
    identity = "same@1"

    def __call__(self, row):
        self.calls.append(1)
        return row


@dataclass(frozen=True)
class Offset:
    """A deterministic step that notes each row it runs on, so a run shows whether its cache ran it."""

    offset: int
    calls: list = field(default_factory=list, compare=False, repr=False)
    identity = "offset@1"

    def __call__(self, row):
        self.calls.append(int(row["id"]))
        return {**row, "value": int(row["id"]) + self.offset}


@pytest.mark.parametrize(("stored", "read"),
                         [("memory", "memory"), ("memory", "indexed"), ("jsonl", "memory")])
def test_a_cache_reads_the_fill_an_earlier_run_completed_and_fills_another_key_afresh(tmp_path, stored, read):
    """A pass whose cache completes every input publishes it under its key, so a later run of the
    same source and steps reads it without running them, in a pass and in training; a step
    configured otherwise is another key, filled afresh."""
    from dew.data.compile import compile, iterate
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Pipeline
    from dew.data.sources import Files

    rows = [{"id": index} for index in range(6)]
    if stored == "jsonl":
        (tmp_path / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        source = Files(str(tmp_path / "rows.jsonl"))
    else:
        source = Memory(rows)

    def cached(step):
        return Pipeline(source).map(step).cache(str(tmp_path / "cache"))

    def values(step):
        return [(int(row["id"]), int(row["value"])) for row in iterate(cached(step), read=read)]

    first, again, trained, other = Offset(10), Offset(10), Offset(10), Offset(20)
    assert values(first) == values(again) == [(index, index + 10) for index in range(6)]
    assert sorted(first.calls) == list(range(6)) and again.calls == []
    data = compile(cached(trained).batch(3), read=read, loading=Loading(workers=0, threads=1, read_buffer=1))
    reader = data.train(DataPartition())
    try:
        assert sorted(int(value) for _ in range(2) for value in next(reader)["value"]) == list(range(10, 16))
    finally:
        reader.close()
    assert trained.calls == []
    assert values(other) == [(index, index + 20) for index in range(6)]
    assert sorted(other.calls) == list(range(6))


def _ids(batches):
    return [[(int(row), int(value)) for row, value in zip(batch["id"], batch["value"], strict=True)]
            for batch in batches]


def test_a_cache_of_one_record_an_input_reads_each_share_as_the_uncached_order_does(tmp_path):
    """A cache whose steps keep one record for each input, then a shuffle: two shares of an
    indexed read each read what the uncached chain's share reads, the shuffle one order over every
    input. A position saved at two shares restores at one, reading on as the uncached chain does."""
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Pipeline

    loading = Loading(workers=0, threads=1, read_buffer=1)
    source = Memory([{"id": index} for index in range(12)])

    def built(cache):
        stepped = Pipeline(source).map(Offset(10))
        return compile((stepped if cache is None else stepped.cache(cache)).shuffle(0).repeat().batch(2),
                       read="indexed", loading=loading)

    def share(data, index, count, steps, state=None):
        reader = data.train(DataPartition(index=index, count=count))
        try:
            if state is not None:
                reader.set_state(state)
            return _ids(next(reader) for _ in range(steps)), reader.get_state()
        finally:
            reader.close()

    uncached, cached = built(None), built(str(tmp_path / "cache"))
    for index in range(2):
        assert share(cached, index, 2, 4)[0] == share(uncached, index, 2, 4)[0]
    _, saved = share(cached, 0, 2, 3)
    _, uncached_saved = share(uncached, 0, 2, 3)
    assert share(cached, 0, 1, 3, saved)[0] == share(uncached, 0, 1, 3, uncached_saved)[0]


def test_a_cache_skips_the_inputs_a_store_misses_as_the_uncached_read_does_and_publishes_them(tmp_path):
    """A store in memory missing two of its inputs: a cache of one record for each input reads the
    rest as the uncached chain does, and completes its fill with the missing ones, so a later run
    reads the cache without running its step."""
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Pipeline

    loading = Loading(workers=0, threads=1, read_buffer=1)
    records = [None if index in (3, 7) else {"id": index} for index in range(10)]

    def batches(step, cache):
        stepped = Pipeline(Memory(records)).map(step)
        chain = (stepped if cache is None else stepped.cache(cache)).repeat(1).batch(2)
        reader = compile(chain, read="indexed", loading=loading).train(DataPartition())
        try:
            return _ids(reader)
        finally:
            reader.close()

    expected = batches(Offset(10), None)
    filling, reading = Offset(10), Offset(10)
    assert batches(filling, str(tmp_path / "cache")) == expected
    assert batches(reading, str(tmp_path / "cache")) == expected
    assert sorted(filling.calls) == [index for index in range(10) if index not in (3, 7)]
    assert reading.calls == []


def test_a_cache_read_by_two_loading_workers_seals_what_each_filled(tmp_path):
    """A cache read in one pass by two loading workers: each worker's fill is sealed as its pass
    ends, the inputs the store misses among them, so together the sealed fills hold every input,
    which a publication of the cache unions."""
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Pipeline

    # Two inputs the store misses: each worker writes their completions as its pass starts.
    source = Memory([None if index in (5, 11) else {"id": index} for index in range(16)], shard_rows=4)
    cache = str(tmp_path / "cache")
    filling = Offset(10)
    chain = Pipeline(source).map(filling).cache(cache).repeat(1).workers(2).batch(2)
    reader = compile(chain, read="indexed", loading=Loading(workers=0, threads=1, read_buffer=1)).train(
        DataPartition())
    try:
        assert sorted(row for batch in _ids(reader) for row, _ in batch) == [
            index for index in range(16) if index not in (5, 11)]
    finally:
        reader.close()
    fs, path = Store(cache).resolve()
    seals = [location for location in fs.find(path) if location.endswith("/seal.json")]
    # Each seal lists the segments its worker wrote, under the fill's root.
    held = {completion["id"] for seal in seals
            for segment in json.loads(fs.cat_file(seal))["segments"]
            for completion in json.loads(fs.cat_file(
                f"{seal.rsplit('/segments/', 1)[0]}/{segment['path']}"))["completions"]}
    assert len(seals) == 2
    assert held == {str(index) for index in range(16)}


def test_a_cache_of_a_store_that_misses_every_input_publishes_its_empty_fill(tmp_path):
    """A store whose every input is missing: a pass reads no row, and its fill holds every input's
    completion, so a publication runs no step and holds no row."""
    from dew.data.compile import iterate, publish
    from dew.data.dataset import Loading
    from dew.data.pipeline import Pipeline

    loading = Loading(workers=0, threads=1, read_buffer=1)
    source = Memory([None] * 6)
    cache = str(tmp_path / "cache")
    # A pass that reads every input: no training run, which refuses a corpus with no row.
    rows = iter(iterate(Pipeline(source).map(Offset(10)).cache(cache), read="indexed", loading=loading))
    try:
        assert list(rows) == []
    finally:
        rows.close()
    publishing = Offset(10)
    assert len(publish(Pipeline(source).map(publishing), Cache(cache), read="indexed", loading=loading)) == 0
    assert publishing.calls == []


def _rows(reader, batches: int) -> set[tuple[bytes, ...]]:
    """The distinct rows of a reader's next `batches` batches, each its fields' bytes."""
    rows = set()
    for _ in range(batches):
        batch = next(reader)
        names = sorted(batch)
        rows |= {tuple(np.asarray(batch[name])[row].tobytes() for name in names)
                 for row in range(len(np.asarray(batch[names[0]])))}
    return rows


def test_the_prompts_preset_with_a_cache_reads_by_index_the_prompts_it_reads_uncached(tmp_path):
    """The prompts preset dropping prompts too long, cached and read by index in loading workers: the
    cache holds what is left of each input, and the read goes on through every prompt kept, as the
    uncached preset's does, however the drops fall among the workers (row 883)."""
    from dew.data import presets
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition, Loading

    # A third of the twelve prompts are longer than 6 bytes, which the preset drops: eight are kept.
    prompts = Memory([{"prompt": chr(97 + index) + "hi" * (1 + index % 3)} for index in range(12)])

    def loaded(cache=None):
        # Worker processes read the cache on disk.
        chain = prompts.shuffle(0).repeat().pipe(presets.prompts, "byte", max_prompt_len=6, long="drop",
                                                 reward_bytes=8, batch=2, cache=cache)
        return compile(chain, read="indexed", loading=Loading(workers=2))

    cached, uncached = loaded(str(tmp_path / "cache")), loaded()
    assert cached.records == uncached.records
    first, second = cached.train(DataPartition()), uncached.train(DataPartition())
    try:
        kept = _rows(second, 16)
        assert len(kept) == 8
        assert _rows(first, 16) == kept
    finally:
        first.close()
        second.close()


def test_the_images_preset_with_a_fixed_caption_and_a_cache_reads_in_its_uncached_order(tmp_path):
    """The images preset's fixed caption is a configured step, which binds to one that keeps every
    image: cached and read by index, it holds one row for each input, so the chain takes the
    global order it takes uncached and reads the same batches."""
    from dew.data import images as image_steps, presets
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition

    images = [{"image": np.random.default_rng(index).integers(0, 256, (6, 5, 3), np.uint8), "label": index}
              for index in range(80)]

    def loaded(cache=None):
        return compile(Memory(images).shuffle(0).repeat().pipe(
            presets.images, 4, batch=4, caption=image_steps.caption("a photo"), rejects=False, cache=cache),
            read="indexed")

    cached, uncached = loaded(str(tmp_path / "cache")), loaded()
    first, second = cached.train(DataPartition()), uncached.train(DataPartition())
    try:
        for _ in range(20):
            left, right = next(first), next(second)
            assert sorted(left) == sorted(right)
            for name in right:
                np.testing.assert_array_equal(np.asarray(left[name]), np.asarray(right[name]))
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("read", ["memory", "indexed"])
def test_a_repeat_before_a_cache_of_one_record_for_each_input_counts_an_epoch_of_its_inputs(read):
    """A shuffle and a repeat written before a cache that keeps one record for each input, as a
    preset's cache sits after them: an epoch is one pass over the inputs, every record once."""
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Pipeline

    rows = Memory([{"id": index} for index in range(12)])
    data = compile(Pipeline(rows).shuffle(0).repeat().map(Same()).cache(_cache()).batch(4), read=read,
                   loading=Loading(workers=0, threads=1, read_buffer=1))
    assert data.records == 12 and data.steps_per_epoch == 3
    reader = data.train(DataPartition())
    try:
        assert sorted(int(index) for _ in range(3) for index in next(reader)["id"]) == list(range(12))
    finally:
        reader.close()


def test_a_cache_followed_by_a_token_split_seals_the_fill_the_split_reads():
    """A cache, then a token holdout, which reads every document as the run is planned: the fill
    the planning read is sealed, so a publication afterwards runs no step again."""
    from dew.data.compile import compile, publish
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Pipeline

    loading = Loading(workers=0, threads=1, read_buffer=1)
    documents = Memory([{"text": np.arange(3 + index % 4, dtype=np.int32)} for index in range(8)])
    cache = _cache()
    train, _ = Pipeline(documents).map(Same()).cache(cache).holdout(tokens=0.25)
    reader = compile(train.repeat().batch(1), read="indexed", loading=loading).train(DataPartition())
    reader.close()
    publishing = Same()
    publish(Pipeline(documents).map(publishing), cache, read="indexed", loading=loading)
    assert publishing.calls == []


@dataclass(frozen=True)
class Keeps:
    """Rejects a record whose id is a multiple of `every`, permanently: a record steps cannot make."""

    every: int
    identity = "keeps@1"
    rejects = True

    def __call__(self, row):
        return Rejected("unkept") if int(row["id"]) % self.every == 0 else row


@pytest.mark.parametrize("read", ["memory", "indexed"])
def test_a_shuffled_cache_fills_what_a_publication_of_its_steps_reads(tmp_path, read):
    """A store missing two of its inputs, shuffled and then cached: the cache fills each input's
    rows by its place in the store, missing ones too, as a publication of the steps keys them, so
    publishing afterwards runs no step again and holds every kept row."""
    from dew.data.compile import compile, publish
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Pipeline

    loading = Loading(workers=0, threads=1, read_buffer=1)
    source = Memory([None if index in (3, 7) else {"id": index} for index in range(12)])
    cache = str(tmp_path / "cache")
    filling = Offset(10)
    chain = Pipeline(source).shuffle(3).map(filling).cache(cache).repeat(1).batch(2)
    reader = compile(chain, read=read, loading=loading).train(DataPartition())
    try:
        assert sorted(row for batch in _ids(reader) for row, _ in batch) == [
            index for index in range(12) if index not in (3, 7)]
    finally:
        reader.close()
    publishing = Offset(10)
    generation = publish(Pipeline(source).map(publishing), Cache(cache), read=read, loading=loading)
    assert len(generation) == 10
    assert publishing.calls == []


@pytest.mark.parametrize("read", ["memory", "indexed"])
def test_a_published_generation_after_a_shuffle_reads_as_the_steps_it_stands_for(tmp_path, read):
    """A generation published from a step that keeps one row for each input stands in for that
    step after a shuffle, read as the shuffled step's rows; one published from a step that drops
    rows holds no row for each input, so a shuffle before it is refused, naming the cache that
    reads it by input."""
    from dew.data.compile import compile, publish
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Pipeline

    loading = Loading(workers=0, threads=1, read_buffer=1)
    source = Memory([{"id": index} for index in range(12)])

    def batches(chain):
        reader = compile(chain.repeat().batch(3), read=read, loading=loading).train(DataPartition())
        try:
            return _ids(next(reader) for _ in range(4))
        finally:
            reader.close()

    generation = publish(Pipeline(source).map(Offset(10)), Cache(str(tmp_path / "one")), read="indexed",
                         loading=loading)
    expected = batches(Pipeline(source).shuffle(3).map(Offset(10)))
    assert batches(Pipeline(source).shuffle(3).map(Offset(10)).cache(generation)) == expected
    kept = Pipeline(source).map(Offset(10)).filter_map(Keeps(5), identity="keeps@1")
    dropped = publish(kept, Cache(str(tmp_path / "kept")), read="indexed", loading=loading)
    with pytest.raises(ValueError):
        batches(Pipeline(source).shuffle(3).map(Offset(10)).filter_map(Keeps(5), identity="keeps@1")
                .cache(dropped))


@pytest.mark.parametrize("views", ["shuffled", "shuffled in windows", "repeated then shuffled", "taken"])
@pytest.mark.parametrize("read", ["memory", "indexed"])
def test_a_cache_after_views_of_the_inputs_reads_as_the_uncached_chain(tmp_path, read, views):
    """Views of the inputs written before a cache (a shuffle, a window shuffle, a repeat, a take)
    read it as they read the inputs: the batches are the uncached chain's, rejected records
    dropped where they are. A view that reads every input completes the cache, so a later run
    reads its published fill without running its steps; a take reads part of them, and the
    cache, holding every input's, is filled again."""
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Pipeline

    loading = Loading(workers=0, threads=1, read_buffer=1)

    def chain(step, cache):
        viewed = {"shuffled": lambda source: source.shuffle(3),
                  "shuffled in windows": lambda source: source.shuffle(3, window=4),
                  "repeated then shuffled": lambda source: source.repeat(2).shuffle(3),
                  "taken": lambda source: source.shuffle(3).take(7)}[views](Pipeline(Memory(
                      [{"id": index} for index in range(12)])))
        stepped = viewed.map(step).filter_map(Keeps(5), identity="keeps@1")
        return (stepped if cache is None else stepped.cache(cache)).repeat().batch(3)

    def batches(step, cache):
        reader = compile(chain(step, cache), read=read, loading=loading).train(DataPartition())
        try:
            return [[(int(row), int(value)) for row, value in zip(batch["id"], batch["value"], strict=True)]
                    for batch in (next(reader) for _ in range(5))]
        finally:
            reader.close()

    expected = batches(Offset(10), None)
    cache = str(tmp_path / "cache")
    filling, reading = Offset(10), Offset(10)
    assert batches(filling, cache) == expected
    assert batches(reading, cache) == expected
    assert filling.calls
    assert (reading.calls == []) == (views != "taken")


@pytest.mark.parametrize("read", ["memory", "indexed"])
def test_a_cache_of_one_record_an_input_after_views_reads_each_share_as_the_uncached_chain(tmp_path, read):
    """A shuffle and a take of the inputs, then a step that keeps one record for each and a cache:
    each of two shares reads what the uncached chain's share reads, the take one over every input,
    and a position saved at two shares restores at one as the uncached chain's does."""
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Pipeline

    loading = Loading(workers=0, threads=1, read_buffer=1)
    source = Memory([{"id": index} for index in range(12)])

    def built(cache):
        stepped = Pipeline(source).shuffle(0).take(10).map(Offset(10))
        return compile((stepped if cache is None else stepped.cache(cache)).repeat().batch(2), read=read,
                       loading=loading)

    def share(data, index, count, steps, state=None):
        reader = data.train(DataPartition(index=index, count=count))
        try:
            if state is not None:
                reader.set_state(state)
            return _ids(next(reader) for _ in range(steps)), reader.get_state()
        finally:
            reader.close()

    uncached, cached = built(None), built(str(tmp_path / "cache"))
    for index in range(2):
        assert share(cached, index, 2, 4)[0] == share(uncached, index, 2, 4)[0]
    _, saved = share(cached, 0, 2, 3)
    _, uncached_saved = share(uncached, 0, 2, 3)
    assert share(cached, 0, 1, 3, saved)[0] == share(uncached, 0, 1, 3, uncached_saved)[0]


@pytest.mark.parametrize("read", ["memory", "indexed"])
def test_views_before_a_cache_of_a_store_missing_inputs_read_its_present_ones(tmp_path, read):
    """A store missing some inputs, shuffled and taken, then steps that drop records, then a cache:
    the views read the inputs the store holds, as the uncached chain's do, in its order."""
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Pipeline

    loading = Loading(workers=0, threads=1, read_buffer=1)
    records = [None if index in (0, 3, 4) else {"id": index} for index in range(12)]

    def batches(cache):
        stepped = (Pipeline(Memory(records)).shuffle(1).take(6).map(Offset(10))
                   .filter_map(Keeps(5), identity="keeps@1"))
        chain = (stepped if cache is None else stepped.cache(cache)).repeat(1).batch(1)
        reader = compile(chain, read=read, loading=loading).train(DataPartition())
        try:
            return _ids(reader)
        finally:
            reader.close()

    assert batches(str(tmp_path / "cache")) == batches(None)


def test_views_before_a_cache_of_a_grain_store_missing_inputs_read_its_present_ones_in_memory(tmp_path):
    """A Grain dataset missing an input, read in memory, shuffled, then a step and a cache: the
    cache reads the inputs the store holds, as the uncached chain does, in its order."""
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Pipeline
    from dew.data.sources.grain import from_grain

    loading = Loading(workers=0, threads=1, read_buffer=1)
    records = [None if index == 2 else {"id": index} for index in range(6)]
    source = from_grain(grain.MapDataset.source(records), name="missing-two")

    def batches(cache):
        stepped = Pipeline(source).shuffle(1).map(Offset(10))
        chain = (stepped if cache is None else stepped.cache(cache)).repeat(1).batch(1)
        reader = compile(chain, read="memory", loading=loading).train(DataPartition())
        try:
            return _ids(reader)
        finally:
            reader.close()

    expected = batches(None)
    assert sorted(row for batch in expected for row, _ in batch) == [0, 1, 3, 4, 5]
    assert batches(str(tmp_path / "cache")) == expected


@dataclass(frozen=True)
class KeepsBy:
    """A configured step that binds to `step`, which drops records."""

    step: Keeps
    identity = "keeps-by@1"

    def __call__(self, metadata):
        return self.step


@dataclass(frozen=True)
class OffsetBy:
    """A configured step that binds to `step`, keeping every record."""

    step: Offset
    identity = "offset-by@1"

    def __call__(self, metadata):
        return self.step


def test_a_configured_memory_read_fills_the_cache_its_publication_reads(tmp_path):
    """A memory read of a configured step after a shuffle, then a cache: its fill is the one a
    publication of that step over the store reads, so publishing afterwards runs no step."""
    from dew.data.compile import compile, publish
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Configured, Pipeline

    loading = Loading(workers=0, threads=1, read_buffer=1)
    source = Memory([None if index == 4 else {"id": index} for index in range(9)])
    cache = str(tmp_path / "cache")
    filling = Offset(10)
    chain = Pipeline(source).shuffle(2).map(Configured(OffsetBy(filling), needs=())).cache(cache)
    reader = compile(chain.repeat(1).batch(2), read="memory", loading=loading).train(DataPartition())
    try:
        assert sorted(row for batch in _ids(reader) for row, _ in batch) == [
            index for index in range(9) if index != 4]
    finally:
        reader.close()
    assert sorted(filling.calls) == [index for index in range(9) if index != 4]
    publishing = Offset(10)
    publish(Pipeline(source).map(Configured(OffsetBy(publishing), needs=())), Cache(cache), read="memory",
            loading=loading)
    assert publishing.calls == []


@pytest.mark.parametrize("read", ["memory", "indexed"])
def test_a_generation_of_a_configured_step_after_a_shuffle_reads_as_the_steps_it_stands_for(tmp_path, read):
    """A generation published from a configured step that states it keeps every record stands in for
    it after a shuffle, read as the uncached chain's rows; one that binds to a step dropping records
    against its statement is refused."""
    from dew.data.compile import compile, publish
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Configured, Pipeline

    loading = Loading(workers=0, threads=1, read_buffer=1)
    source = Memory([{"id": index} for index in range(12)])

    def stepped(chain):
        return chain.map(Configured(OffsetBy(Offset(10)), needs=(), one_to_one=True))

    def batches(chain):
        reader = compile(chain.repeat().batch(3), read=read, loading=loading).train(DataPartition())
        try:
            return _ids(next(reader) for _ in range(4))
        finally:
            reader.close()

    generation = publish(stepped(Pipeline(source)), Cache(str(tmp_path / "one")), read=read, loading=loading)
    assert batches(stepped(Pipeline(source).shuffle(3)).cache(generation)) == batches(
        stepped(Pipeline(source).shuffle(3)))
    with pytest.raises(ValueError):
        batches(Pipeline(source).map(Configured(KeepsBy(Keeps(5)), needs=(), one_to_one=True)))


@dataclass(frozen=True)
class Bounded:
    """A step that keeps its record and states a bound of it (`_dew_bounds`), which a memory read
    measures over its rows."""

    identity = "bounded@1"

    def __call__(self, row):
        return {**row, "_dew_bounds": {"width": int(row["id"]) + 1}}


@dataclass(frozen=True)
class Widened:
    """A run's configured step: each record carries the widest bound the run measured."""

    identity = "widened@1"

    def __call__(self, metadata):
        return Width(int(metadata["width"]))


@dataclass(frozen=True)
class Width:
    width: int
    identity = "width@1"

    def __call__(self, row):
        return {**row, "width": self.width}


def test_a_memory_read_measures_the_steps_before_a_view_and_its_cache_as_uncached(tmp_path):
    """A memory read runs the steps written before the first view of the inputs over the whole
    source, cache or none, so a run's configured step after the cache binds to the bound they
    state, and the run reads what the uncached chain reads."""
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Configured, Pipeline

    loading = Loading(workers=0, threads=1, read_buffer=1)
    source = Memory([{"id": index} for index in range(12)])

    def batches(cache):
        stepped = Pipeline(source).map(Bounded()).shuffle(3).map(Offset(10))
        cached = stepped if cache is None else stepped.cache(cache)
        chain = cached.map(Configured(Widened(), needs=("width",), scope="run")).repeat().batch(3)
        reader = compile(chain, read="memory", loading=loading).train(DataPartition())
        try:
            return [[(int(row), int(value), int(width))
                     for row, value, width in zip(batch["id"], batch["value"], batch["width"], strict=True)]
                    for batch in (next(reader) for _ in range(4))]
        finally:
            reader.close()

    expected = batches(None)
    assert {width for batch in expected for _, _, width in batch} == {12}
    assert batches(str(tmp_path / "cache")) == expected


@dataclass(frozen=True)
class Stamped:
    """Notes the source position a read gives its record."""

    identity = "stamped@1"

    def __call__(self, index, row):
        return {**row, "at": index}


def test_a_step_reading_its_index_in_a_cache_after_a_memory_read_step_reads_the_source_position(tmp_path):
    """A store missing some inputs, a step a memory read runs over it, a shuffle, then a step that
    reads its record's index and a cache: the cached step reads each record's place in the store,
    as the uncached one does, in each of two passes."""
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Pipeline

    loading = Loading(workers=0, threads=1, read_buffer=1)
    source = Memory([None if index in (0, 4) else {"id": index} for index in range(8)])

    def batches(cache):
        stepped = Pipeline(source).map(Same()).shuffle(0).map_with_index(Stamped(), identity="stamped@1")
        chain = (stepped if cache is None else stepped.cache(cache)).repeat(2).batch(1)
        reader = compile(chain, read="memory", loading=loading).train(DataPartition())
        try:
            return [(int(batch["id"][0]), int(batch["at"][0])) for batch in reader]
        finally:
            reader.close()

    # Two passes, each of every record the store holds.
    expected = batches(None)
    assert sorted(expected) == sorted(2 * [(index, index) for index in range(8) if index not in (0, 4)])
    assert batches(str(tmp_path / "cache")) == expected


def test_a_streamed_view_before_a_cache_is_refused_with_its_reason(tmp_path):
    """A streamed read fills its cache shard by shard, so a shuffle written before the cache is
    refused there, naming the reads that support it."""
    from dew.data.compile import compile
    from dew.data.dataset import Loading
    from dew.data.pipeline import Pipeline

    chain = Pipeline(Memory([{"id": index} for index in range(8)])).shuffle(3, window=4).map(Offset(1))
    with pytest.raises(ValueError):
        compile(chain.cache(str(tmp_path / "cache")).batch(2), read="streamed", loading=Loading(workers=0))


def test_a_streamed_cache_reads_the_generation_published_under_its_key(tmp_path):
    """A streamed fill's shards are filled apart, so `publish` publishes them; a later streamed read
    of the same source and steps then reads the generation, and `publish` returns it as it is."""
    from dew.data.compile import iterate, publish
    from dew.data.pipeline import Pipeline

    source = Memory([{"id": index} for index in range(6)], shard_rows=2)
    cache = Cache(str(tmp_path / "cache"))
    filled, reading, republished = Offset(10), Offset(10), Offset(10)
    generation = publish(Pipeline(source).map(filled), cache, read="streamed")
    rows = sorted((int(row["id"]), int(row["value"]))
                  for row in iterate(Pipeline(source).map(reading).cache(cache), read="streamed"))
    assert rows == [(index, index + 10) for index in range(6)] and reading.calls == []
    assert publish(Pipeline(source).map(republished), cache, read="streamed").url == generation.url
    assert republished.calls == []


def test_a_fill_whose_spool_drops_its_last_shard_reads_on_and_publishes_nothing(tmp_path):
    """A spool too small for the fill's one shard (on_full='stop') stops admitting it: the pass
    reads every row, and the next run, with nothing published, runs the steps again."""
    from dew.data.compile import iterate
    from dew.data.pipeline import Pipeline

    source = Memory([{"id": index} for index in range(3)])
    first, again = Offset(1), Offset(1)
    for step in (first, again):
        cache = Cache(str(tmp_path / "cache"), spool_bytes=256, shard_bytes=256)
        chain = Pipeline(source).map(step).cache(cache)
        assert [int(row["value"]) for row in iterate(chain, read="indexed")] == [1, 2, 3]
    assert sorted(first.calls) == sorted(again.calls) == [0, 1, 2]


def test_a_streamed_generation_serves_its_files_at_another_mount(tmp_path):
    """A streamed fill names its inputs by their files' places, which the same files mounted elsewhere
    keep: a cached read at the second mount finds every input the first filled and runs no step, its
    pieces the chain's own there, and a packing plan publishes onto the generation."""
    import shutil

    from dew.data.compile import iterate, publish
    from dew.data.dataset import Loading
    from dew.data.pipeline import Pipeline
    from dew.data.sources import Files
    from dew.data.text import Chunks, Tokenize

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    (first / "rows.jsonl").write_text("".join(json.dumps({"text": "x" * size}) + "\n" for size in (2, 5, 7)))
    shutil.copytree(first, second)
    tokenized = []

    def counted(row):
        tokenized.append(row["text"])
        return row

    def chunked(mount):
        return (Pipeline(Files(str(mount / "rows.jsonl"))).map(counted, identity="counted@1").map(Tokenize())
                .flat_map(Chunks(4)))

    def pieces(pipeline):
        read = iterate(pipeline, read="streamed", loading=Loading(workers=0, threads=1, read_buffer=1))
        return sorted(row["text"].tolist() for row in read)

    cache = Cache(str(tmp_path / "cache"))
    generation = publish(chunked(first), cache, read="streamed")
    filled = len(tokenized)
    cached = pieces(chunked(second).cache(cache))
    assert len(tokenized) == filled
    assert cached == pieces(chunked(second))
    assert publish(chunked(first).pack(4, bins=2), cache, read="streamed").url != generation.url


@pytest.mark.parametrize("read", ["indexed", "streamed"])
def test_a_warm_cache_read_by_worker_processes_refuses_a_manifest_rewritten_since_it_was_committed(
        tmp_path, read):
    """Worker processes read a published generation's manifest checked against the size and digest
    its cache's CURRENT committed it under, as the process that compiled the read does: a manifest
    rewritten since, the same records in other bytes, is refused, not read."""
    from dew.data.compile import compile, publish
    from dew.data.dataset import DataPartition, Loading

    # Shards of two records, so each worker streams its own.
    chain = Memory([{"id": index} for index in range(8)], shard_rows=2).map(Offset(1))
    cache = Cache(str(tmp_path / "cache"))
    publish(chain, cache, read=read)
    loader = compile(chain.cache(cache).repeat().batch(2), read=read,
                     loading=Loading(workers=2, threads=1, read_buffer=2))
    (manifest,) = (tmp_path / "cache").rglob("generations/*/manifest.json")
    rewritten = json.dumps(json.loads(manifest.read_bytes()), indent=1).encode()
    manifest.unlink()
    manifest.write_bytes(rewritten)
    reader = loader.train(DataPartition())
    try:
        with pytest.raises(OSError):
            next(reader)
    finally:
        reader.close()


@pytest.mark.parametrize("read", ["memory", "indexed"])
def test_a_cached_chain_cut_before_its_repeat_counts_its_cut_as_uncached(tmp_path, read):
    """A chain cut to its first 3 records, repeated, and cached after a step counts the 3 its repeat
    reads an epoch, as the same chain uncached does."""
    from dew.data.compile import compile
    from dew.data.dataset import Loading

    chain = Memory([{"id": index} for index in range(8)]).take(3).repeat().map(Offset(1))
    loading = Loading(workers=0, threads=1, read_buffer=0)
    cached = compile(chain.cache(str(tmp_path / "cache")).batch(1), read=read, loading=loading)
    assert cached.records == compile(chain.batch(1), read=read, loading=loading).records == 3


def test_an_empty_stream_publishes_an_empty_generation():
    """A stream with no records publishes a generation of none, its shards read to their end."""
    from dew.data.compile import publish
    from dew.data.pipeline import Pipeline
    from dew.data.sources import from_iter

    generation = publish(Pipeline(from_iter(lambda: iter(()), name="empty@1", replay=True)).map(Offset(1)),
                         _cache(), read="streamed")
    assert len(generation) == 0


def test_published_generation_exposes_only_declared_order_metadata_and_plan():
    import numpy as np

    source = Memory(
        [{"id": index, "_dew_length": index + 2, "_dew_valid_targets": index + 1} for index in range(4)]
    )
    cache, key = _cache(), _key()
    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), key)
    for index in range(len(rows)):
        rows[index]
    rows.finish()
    rows.close()
    generation = cache.publish(key, source, plan=lambda ready: {"offsets": np.arange(len(ready) + 1)})
    assert generation.source().metadata()["valid_targets"] == 10
    assert generation.source().lengths().tolist() == [2, 3, 4, 5]
    assert generation.plan("offsets").tolist() == [0, 1, 2, 3, 4]
    assert generation.plan_length("offsets") == 5
    assert generation.plan_at("offsets", 3).item() == 3
    # A plan published page by page is named as the same array held whole is (`plan.fingerprint`).
    from dew.data.plan import fingerprint

    assert generation.plan_hash("offsets") == fingerprint(np.arange(5))


def test_transient_failures_never_certify_cache_coverage_but_permanent_rejections_do():
    from dew.data.pipeline import Rejected

    source = Memory([{"id": 0}, {"id": 1}])
    cache, key = _cache(), _key()
    transient = cache.mapped(
        grain.MapDataset.source(source.records("indexed")),
        key,
        transform=lambda index, row: (Rejected("transient", permanent=False),),
    )
    assert transient[0][0].reason == "transient"
    transient.finish()
    transient.close()
    with pytest.raises(ValueError):
        cache.publish(key, source)
    permanent = cache.mapped(
        grain.MapDataset.source(source.records("indexed")),
        key,
        transform=lambda index, row: (Rejected("http_404"),) if index == 0 else (row,),
    )
    permanent[0], permanent[1]
    permanent.finish()
    permanent.close()
    generation = cache.publish(key, source)
    assert len(generation) == 1
    assert generation[0] == {"id": 1}
    assert generation.outputs(0)[0].reason == "http_404"


def test_local_retention_keeps_pinned_generations_and_evicts_closed_ones(tmp_path):
    source = Memory([{"id": 0}])
    cache = Cache(str(tmp_path), keep_local_bytes=250_000, shard_bytes=1)
    first_key, second_key = _key("one"), _key("two")
    first = cache.mapped(grain.MapDataset.source(source.records("indexed")), first_key)
    first[0]
    first.finish()
    first.close()
    generation = cache.publish(first_key, source)
    second = cache.mapped(grain.MapDataset.source(source.records("indexed")), second_key)
    second[0]
    second.finish()
    second.close()
    assert generation[0] == {"id": 0}
    with pytest.raises(ValueError):
        cache.publish(second_key, source)
    generation.close()
    retry = cache.mapped(grain.MapDataset.source(source.records("indexed")), second_key)
    retry[0]
    retry.finish()
    retry.close()
    assert cache.publish(second_key, source)[0] == {"id": 0}
    assert cache.generation(first_key) is None


def test_local_retention_reclaims_the_bytes_of_fills_closed_before_they_sealed(tmp_path):
    """A run that stops mid-fill closes its writer unsealed, and no generation ever reads its
    shards; once no process holds it, retention reclaims its bytes, so fills stopped again and
    again do not fill the cache until a new one is refused. A shard here is one 128 KiB block,
    so the cache holds one."""

    def rows(index: int) -> Memory:
        return Memory([{"id": index * 10 + row} for row in range(2)])

    cache = Cache(str(tmp_path), keep_local_bytes=250_000, shard_bytes=1)
    for index in range(3):
        source = rows(index)
        stopped = cache.mapped(grain.MapDataset.source(source.records("indexed")),
                               _key(index))
        stopped[0]
        stopped.close()
    source, key = Memory([{"id": 99}]), _key("kept")
    filled = cache.mapped(grain.MapDataset.source(source.records("indexed")), key)
    filled[0]
    filled.finish()
    filled.close()
    assert cache.publish(key, source)[0] == {"id": 99}


def test_a_local_cache_fails_a_shard_its_disk_refuses_at_once(tmp_path, monkeypatch):
    """This host's disk refuses a write for reasons a retry keeps, as its retention refusing room
    for a shard does, so a blocking fill ends with that error after one attempt, however many
    retries the cache allows a remote store; it once tried and slept through each of them."""
    import filelock
    import numpy as np

    attempts = []
    acquire = filelock.FileLock.acquire

    def counted(lock, *arguments, **options):
        if str(lock.lock_file).endswith(".retention.lock"):
            attempts.append(lock.lock_file)
        return acquire(lock, *arguments, **options)

    monkeypatch.setattr(filelock.FileLock, "acquire", counted)
    source = Memory([{"id": 0, "blob": np.random.default_rng(0).bytes(200_000)}])
    cache = Cache(str(tmp_path), keep_local_bytes=64 * 1024, on_full="block", shard_bytes=1, retry=3)
    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), _key())
    try:
        assert rows[0]["id"] == 0
        with pytest.raises(OSError):
            rows.finish()
        assert len(attempts) == 1
    finally:
        rows.close()


def test_a_remote_upload_that_fails_ends_the_fill_waiting_for_its_room():
    """A fill that waits for spool room as a shard uploads is woken when the upload fails, at once:
    the transport has retried a failed transfer already, so the cache's retries, which re-read an
    upload that reads back otherwise, try it no more. It ends with the upload's error rather than
    waiting on bytes that will never free. The service fails only once the fill is waiting."""
    import numpy as np
    from fsspec.implementations.memory import MemoryFileSystem

    attempts, waiting = [], threading.Event()

    class Refusing(MemoryFileSystem):
        """Memory objects behind a third-party conditional write that fails once the fill waits."""

        protocol = "refusing"

        @classmethod
        def _strip_protocol(cls, path):
            return super()._strip_protocol(path.replace("refusing://", "memory://", 1))

        def put_conditional(self, path, payload, version):
            attempts.append(path)
            waiting.wait(20)
            raise OSError(f"the service refused {path}")

    class Observed(threading.Condition):
        """The writer's condition, noting when a thread waits on it."""

        def wait(self, timeout=None):
            waiting.set()
            return super().wait(timeout)

    fsspec.register_implementation("refusing", Refusing, clobber=True)
    blobs = np.random.default_rng(0)
    source = Memory([{"id": index, "blob": blobs.bytes(300_000)} for index in range(2)])
    cache = Cache(f"refusing://cache-{uuid4().hex}", spool_bytes=500_000, shard_bytes=1, on_full="block",
                  retry=1)
    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), _key())
    rows.writer.lock = Observed(threading.RLock())
    raised: list[BaseException] = []

    def read() -> None:
        try:
            rows[0], rows[1]
        except Exception as error:  # the test reads what ended the fill
            raised.append(error)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    reader.join(20)
    try:
        assert not reader.is_alive()
        assert [type(error) for error in raised] == [OSError] and len(attempts) == 1
    finally:
        waiting.set()
        rows.close()


def test_training_close_abandons_a_stalled_upload_without_publishing_a_segment():
    from fsspec.implementations.memory import MemoryFileSystem

    from dew.data.dataset import _QUIET_STOP_SECONDS

    entered, release = threading.Event(), threading.Event()
    stored = []

    class Stalled(MemoryFileSystem):
        """Memory objects behind a third-party conditional write whose upload
        hangs until the test releases it."""

        protocol = "stalled"

        @classmethod
        def _strip_protocol(cls, path):
            return super()._strip_protocol(path.replace("stalled://", "memory://", 1))

        def put_conditional(self, path, payload, version):
            entered.set()
            release.wait(15)
            if self.exists(path):
                raise FileExistsError(path)
            self.pipe_file(path, payload)
            stored.append(path)
            return hashlib.sha256(payload).hexdigest()

    fsspec.register_implementation("stalled", Stalled, clobber=True)
    source = Memory([{"id": 0}])
    cache = Cache(f"stalled://cache-{uuid4().hex}", shard_bytes=1, retry=0)
    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), _key())
    try:
        assert rows[0] == {"id": 0}
        assert entered.wait(5)
        started = time.monotonic()
        rows.close()
        assert time.monotonic() - started < _QUIET_STOP_SECONDS + 2
        assert rows.metrics()["abandoned"] > 0
    finally:
        release.set()
        for future in rows.writer.futures:
            with contextlib.suppress(Abandoned):
                future.result(timeout=5)
    assert stored and not any("/segments/" in path for path in stored)


@pytest.mark.network
def test_real_r2_generation_publication_mirror_and_cleanup(tmp_path):
    root = os.environ["DEW_DATA_TEST_URL"]  # a writable remote smoke prefix
    source = Memory([{"id": index, "text": "remote"} for index in range(4)])
    key = _key("r2-smoke artifact-v1")
    cache = Cache(
        root.rstrip("/") + "/generation",
        storage_options={"dewcache": {"cache_storage": str(tmp_path), "max_bytes": 2**20}},
    )
    generation = cache.generation(key)
    if generation is None:
        rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), key)
        for index in range(len(rows)):
            rows[index]
        rows.finish()
        rows.close()
        generation = cache.publish(key, source)
    try:
        assert [generation[index] for index in range(len(generation))] == [
            {"id": index, "text": "remote"} for index in range(4)
        ]
        assert cache.generation(key).url == generation.url
    finally:
        generation.close()
        fs, path = Store(cache.url + "/" + key.key).resolve()
        for object_path in fs.find(path):
            fs.rm(object_path)
        assert not fs.exists(path + "/CURRENT")


def _tagged(pipeline):
    """`pipeline`'s rows with a tag beside their id: other rows than its inputs, one for each."""
    return pipeline.map(lambda row: {**row, "tag": row["id"] * 10}, identity="tag@1")


def _tags(pipeline, read="memory"):
    from dew.data.compile import iterate
    from dew.data.dataset import Loading

    return [int(row["tag"]) for row in iterate(pipeline, read=read, loading=Loading(workers=0))]


@pytest.mark.parametrize("read", ["memory", "streamed"])
def test_a_generation_of_the_store_stands_in_for_both_parts_of_a_hash_holdout(read):
    """A generation published from the store a hash holdout is taken from, its step making other rows
    than its inputs: under the holdout each part reads the rows of the inputs its hash placed there,
    the parts the chain reads uncached, in every read."""
    from dew.data.compile import publish
    from dew.data.pipeline import Pipeline

    source = Memory([{"id": index} for index in range(24)])
    generation = publish(_tagged(Pipeline(source)), _cache(), read="memory")
    parts = Pipeline(source).holdout(0.25)
    uncached = [_tags(_tagged(part)) for part in parts]
    assert uncached[0] and uncached[1]
    assert [_tags(_tagged(part).cache(generation), read) for part in parts] == uncached


def test_a_holdout_by_a_field_its_cached_step_drops_reads_the_parts_it_reads_uncached():
    """A hash holdout by `id` written before a step that keeps only a tag: the holdout reads `id` on
    the inputs, so the generation of that step stands in for both parts as the step does."""
    from dew.data.compile import publish
    from dew.data.pipeline import Pipeline

    def renamed(pipeline):
        return pipeline.map(lambda row: {"tag": row["id"] * 10}, identity="renamed@1")

    source = Memory([{"id": index} for index in range(24)])
    generation = publish(renamed(Pipeline(source)), _cache(), read="memory")
    parts = Pipeline(source).holdout(0.25, key="id")
    uncached = [_tags(renamed(part)) for part in parts]
    assert all(uncached)
    assert [_tags(renamed(part).cache(generation)) for part in parts] == uncached


def test_a_records_holdout_over_missing_rows_and_a_dropping_step_reads_the_inputs_it_holds_out():
    """A store with missing rows and a step that drops odd ids: the generation holds the rows of the
    inputs the step kept, none for a missing one, so the first five inputs held out read their
    rows from it, not the generation's first rows, as the chain does uncached."""
    from dew.data.compile import publish
    from dew.data.pipeline import Pipeline

    def even(pipeline):
        return pipeline.filter_map(
            lambda row: {**row, "tag": row["id"] * 10} if row["id"] % 2 == 0 else Rejected("odd"),
            identity="even@1")

    source = Memory([None if index in (1, 4) else {"id": index} for index in range(16)])
    generation = publish(even(Pipeline(source)), _cache(), read="memory")
    parts = Pipeline(source).holdout(records=5)
    uncached = [_tags(even(part)) for part in parts]
    assert uncached == [[60, 80, 100, 120, 140], [0, 20]]
    assert [_tags(even(part).cache(generation)) for part in parts] == uncached


def test_a_step_that_reads_its_index_before_a_generation_under_a_holdout_is_refused():
    """A step that tags each row with its position sees the holdout's positions uncached and the
    store's in a generation of the whole store, so the generation can't stand in under the holdout;
    a cache of the part, which fills from the part's own positions, reads what the chain reads."""
    from dew.data.compile import publish
    from dew.data.pipeline import Pipeline

    def placed(pipeline):
        return pipeline.map_with_index(lambda index, row: {**row, "tag": index}, identity="placed@1")

    source = Memory([{"id": index} for index in range(12)])
    generation = publish(placed(Pipeline(source)), _cache(), read="memory")
    trained, _ = Pipeline(source).holdout(records=4)
    with pytest.raises(ValueError):
        _tags(placed(trained).cache(generation))
    assert _tags(placed(trained).cache(_cache())) == _tags(placed(trained)) == list(range(8))


def test_a_records_holdout_over_a_stream_filled_generation_serves_where_the_stream_reads_in_index_order(
        tmp_path):
    """A stream numbers its inputs by its shards, a records holdout by index: a store whose stream
    reads its records in index order places the records held out in a generation a stream filled,
    as the chain reads them uncached; files whose stream has an order of its own are refused it."""
    from dew.data import Files
    from dew.data.compile import publish
    from dew.data.pipeline import Pipeline

    memory = Memory([{"id": index} for index in range(12)])
    generation = publish(_tagged(Pipeline(memory)), _cache(), read="streamed")
    parts = Pipeline(memory).holdout(records=5)
    cached = [_tags(_tagged(part).cache(generation)) for part in parts]
    assert cached == [_tags(_tagged(part)) for part in parts]
    (tmp_path / "rows.jsonl").write_text("".join(f'{{"id": {index}}}\n' for index in range(12)))
    files = Files(str(tmp_path / "rows.jsonl"))
    streamed = publish(_tagged(Pipeline(files)), _cache(), read="streamed")
    trained, _ = Pipeline(files).holdout(records=5)
    with pytest.raises(ValueError):
        _tags(_tagged(trained).cache(streamed))


def test_a_hash_part_of_a_generation_reads_in_worker_processes_as_uncached():
    """Worker processes read a hash part of a generation the rows the parent placed, the ones the
    chain reads uncached."""
    import itertools

    from dew.data.compile import compile, publish
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Pipeline

    source = Memory([{"id": index} for index in range(48)])
    generation = publish(_tagged(Pipeline(source)), _cache(), read="memory")
    trained, _ = Pipeline(source).holdout(0.25)

    def epoch(chain):
        data = compile(chain.repeat().batch(4), read="memory", seed=0,
                       loading=Loading(workers=2, threads=1, read_buffer=2))
        reader = data.train(DataPartition())
        try:
            return sorted(int(tag) for batch in itertools.islice(reader, data.steps_per_epoch)
                          for tag in batch["tag"])
        finally:
            reader.close()

    assert epoch(_tagged(trained).cache(generation)) == epoch(_tagged(trained))


def test_stacked_records_parts_that_read_other_records_are_other_plans():
    """The first two records of the store's records past the first, and of those past the second:
    other records, so other plans, as each part names what it selects of the part it selects from."""
    from dew.data.compile import compile, iterate
    from dew.data.dataset import Loading
    from dew.data.pipeline import Pipeline

    store = Pipeline(Memory([{"id": index} for index in range(12)]))
    parts = [store.holdout(records=skipped)[0].holdout(records=2)[1] for skipped in (1, 2)]
    loading = Loading(workers=0)
    plans = [compile(part.batch(1), read="memory", loading=loading).plan.identity() for part in parts]
    assert [[int(row["id"]) for row in iterate(part, read="memory")] for part in parts] == [[1, 2], [2, 3]]
    assert plans[0] != plans[1]


@pytest.mark.parametrize("order", ["records of hash", "hash of records"])
def test_stacked_holdouts_over_a_generation_pick_the_inputs_they_pick_uncached(order):
    """A records holdout of a hash part, and a hash holdout of a records part: each stacked part reads
    the rows of the inputs both selections keep, as uncached, and says how many records it holds out."""
    from dew.data.compile import compile, publish
    from dew.data.dataset import Loading
    from dew.data.pipeline import Pipeline

    source = Memory([{"id": index} for index in range(40)])
    generation = publish(_tagged(Pipeline(source)), _cache(), read="memory")
    store = Pipeline(source)
    first = store.holdout(0.25) if order == "records of hash" else store.holdout(records=6)
    stacked = [part for outer in first for part in
               (outer.holdout(records=3) if order == "records of hash" else outer.holdout(0.5))]
    uncached = [_tags(_tagged(part)) for part in stacked]
    assert all(uncached)
    assert [_tags(_tagged(part).cache(generation)) for part in stacked] == uncached
    def held_out(chain):
        return compile(chain.repeat().batch(1), read="memory", loading=Loading(workers=0)).held_out

    for part in stacked:
        assert held_out(_tagged(part).cache(generation)) == held_out(_tagged(part))


def test_a_generation_a_stream_filled_from_files_mounted_elsewhere_serves_under_a_holdout(tmp_path):
    """The same files mounted at another path read as the store the generation was published from, and
    a stream names its inputs by their files' places, which they keep there: under a holdout the
    generation serves each part the rows the chain reads uncached, at either mount."""
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    from dew.data import Files
    from dew.data.compile import publish
    from dew.data.pipeline import Pipeline

    (tmp_path / "a").mkdir()
    pq.write_table(pa.table({"id": list(range(24))}), tmp_path / "a" / "rows.parquet")
    shutil.copytree(tmp_path / "a", tmp_path / "b")
    generation = publish(_tagged(Pipeline(Files(str(tmp_path / "a"), format="parquet"))), _cache(),
                         read="streamed")
    for mount in ("a", "b"):
        part, _ = Pipeline(Files(str(tmp_path / mount), format="parquet")).holdout(0.25)
        assert _tags(_tagged(part).cache(generation), "streamed") == _tags(_tagged(part), "streamed")


def test_a_generation_written_under_the_earlier_schema_is_refused_as_stale(tmp_path):
    """A cache generation written under the schema before inputs were named by their files' places is
    refused rather than read under names it was not written with, where the same manifest under the
    current schema reads."""
    from dew.data.compile import publish
    from dew.data.generation import Generation
    from dew.data.pipeline import Pipeline

    cache = Cache(str(tmp_path / "cache"))
    generation = publish(Pipeline(Memory([{"id": index} for index in range(8)], shard_rows=4)), cache,
                         read="streamed")
    (manifest,) = (tmp_path / "cache").rglob("generations/*/manifest.json")
    current = json.loads(manifest.read_bytes())
    assert Generation(generation.url, {}, None).manifest == current
    manifest.unlink()
    manifest.write_bytes(json.dumps({**current, "schema": 1}).encode())
    with pytest.raises(ValueError):
        _ = Generation(generation.url, {}, None).manifest


def test_a_generation_a_memory_read_filled_from_unindexed_files_serves_hash_and_records_holdouts(tmp_path):
    """Files with no index, and a hash part, read in memory: the generation a memory read filled
    numbers their inputs by place, so hash and records holdouts over it read the rows of the inputs
    they keep, as the chain does uncached."""
    from dew.data import Files
    from dew.data.compile import publish
    from dew.data.pipeline import Pipeline

    (tmp_path / "rows.jsonl").write_text("".join(f'{{"id": {index}}}\n' for index in range(24)))
    files = Pipeline(Files(str(tmp_path / "rows.jsonl")))
    generation = publish(_tagged(files), _cache(), read="memory")
    for parts in (files.holdout(0.25), files.holdout(records=5)):
        uncached = [_tags(_tagged(part)) for part in parts]
        assert all(uncached)
        assert [_tags(_tagged(part).cache(generation)) for part in parts] == uncached
    trained, _ = files.holdout(0.25)
    part = publish(_tagged(trained), _cache(), read="memory")
    stacked = trained.holdout(records=3)
    uncached = [_tags(_tagged(selected)) for selected in stacked]
    assert all(uncached)
    assert [_tags(_tagged(selected).cache(part)) for selected in stacked] == uncached


def test_a_named_stream_that_holds_other_inputs_than_its_generation_is_refused_under_a_holdout():
    """A stream named for its content, which reads more records than the generation published under
    that name was filled from: under a holdout the generation can't place them, and is refused."""
    from dew.data import from_iter
    from dew.data.compile import publish
    from dew.data.pipeline import Pipeline

    def stream(count):
        return from_iter(lambda: iter([{"id": index} for index in range(count)]), name="rows@1", replay=True)

    generation = publish(_tagged(Pipeline(stream(10))), _cache(), read="streamed")
    trained, _ = Pipeline(stream(12)).holdout(0.25)
    with pytest.raises(ValueError):
        _tags(_tagged(trained).cache(generation), "streamed")


def test_a_worker_reads_a_part_of_a_generation_without_its_store():
    """The reader a worker takes for a part of a generation holds the rows it reads, not the store
    whose inputs placed them: a store of large records pickles large, its part's reader small."""
    import pickle

    from dew.data.analysis import _promoted
    from dew.data.compile import publish
    from dew.data.pipeline import Pipeline

    source = Memory([{"id": index, "blob": bytes(1 << 16)} for index in range(32)])
    generation = publish(_tagged(Pipeline(source)), _cache(), read="memory")
    trained, _ = Pipeline(source).holdout(0.25)
    part = _promoted(_tagged(trained).cache(generation)).source
    assert len(pickle.dumps(part.manifest().reader)) * 100 < len(pickle.dumps(source))


@pytest.mark.parametrize("store", ["grain", "arrow"])
def test_a_records_holdout_over_a_stream_filled_generation_of_a_dataset_read_in_index_order(store):
    """A Grain map dataset and an Arrow dataset stream their records in index order, as they say: a
    records holdout over a generation a stream filled from either reads the rows of the inputs it
    keeps, as the chain does uncached."""
    import datasets
    import grain.python as grain

    from dew.data import from_grain, from_hf
    from dew.data.compile import publish
    from dew.data.pipeline import Pipeline

    rows = [{"id": index} for index in range(12)]
    source = (from_grain(grain.MapDataset.source(rows), name="rows@1") if store == "grain"
              else from_hf(datasets.Dataset.from_list(rows), name="rows@1"))
    generation = publish(_tagged(Pipeline(source)), _cache(), read="streamed")
    parts = Pipeline(source).holdout(records=5)
    uncached = [_tags(_tagged(part)) for part in parts]
    assert all(uncached)
    assert [_tags(_tagged(part).cache(generation)) for part in parts] == uncached


@pytest.mark.parametrize("store", ["hash part", "concatenation"])
def test_a_records_holdout_over_a_stream_filled_generation_of_a_part_or_a_concatenation_serves(store):
    """A hash part of a store in memory, and a concatenation of stores in memory, stream their records
    in index order, as their stores do: a records holdout over a generation a stream filled from
    either reads the rows of the inputs it keeps, as the chain does uncached."""
    from dew.data import concat
    from dew.data.compile import publish
    from dew.data.pipeline import Pipeline

    rows = [{"id": index} for index in range(24)]
    if store == "hash part":
        source, _ = Pipeline(Memory(rows)).holdout(0.25)
    else:
        source = concat([Memory(rows[:10]), Memory(rows[10:])])
    generation = publish(_tagged(source), _cache(), read="streamed")
    parts = source.holdout(records=5)
    uncached = [_tags(_tagged(part)) for part in parts]
    assert all(uncached)
    assert [_tags(_tagged(part).cache(generation)) for part in parts] == uncached
