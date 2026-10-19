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
import pytest

from dew.data.cache import Abandoned, Cache
from dew.data.pipeline import Rejected
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
    cache, key = _cache(), Manifest()

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
    cache, key = _cache(), Manifest()
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
    cache, key = _cache(shard_bytes=1), Manifest()
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
    cache, key = _cache(), Manifest()
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
    cache, key = _cache(), Manifest()
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
    cache, key = _cache(), Manifest()
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
    cache, key = _cache(spool_bytes=spool_bytes, shard_bytes=128), Manifest()
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
    _cache().mapped(rows, Manifest()).close()
    with pytest.raises(ValueError):
        _cache().mapped(rows, Manifest(**refused))
    assert Manifest().key != Manifest(b'{"revision":"changed"}').key


def test_moto_generation_publication_checks_shards_and_concurrent_finalizers():
    import fsspec
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    _, port = server.get_host_and_port()
    options = {"key": "testing", "secret": "testing", "endpoint_url": f"http://127.0.0.1:{port}"}
    fs = fsspec.filesystem("s3", **options)
    bucket = "cache-" + uuid4().hex
    fs.mkdir(bucket)
    cache, key = Cache(f"s3://{bucket}/cache", storage_options=options), Manifest()
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

    try:
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
    finally:
        server.stop()


def test_stream_completions_use_physical_shards_and_resume_without_rekeying():
    source = Memory([{"id": index} for index in range(8)], shard_rows=2)
    cache, key = _cache(), Manifest()
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
    cache, key = _cache(), Manifest()
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
    from dew.data.compile import iterate, publish, validation
    from dew.data.dataset import DataPartition
    from dew.data.pipeline import Pipeline
    from dew.objectives.base import VALID_ROWS

    source = Memory([{"id": index} for index in range(8)])
    even = Pipeline(source).filter(lambda row: row["id"] % 2 == 0, identity="even@1")
    warm = Pipeline(source).cache(publish(even, _cache(), read="memory"))
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
    from the store the parts select from serves both (`rebased`)."""
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


def test_a_loader_reads_a_generation_where_it_holds_the_rows_and_each_other_source_itself():
    """A recipe given the generation of its training branch trains on the generation and scores
    its validation source's own rows; a recipe none of whose rows the generation holds is refused."""
    from dew.config.data import RecordsData, loader
    from dew.data.compile import publish
    from dew.data.dataset import DataPartition, Loading

    read = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)
    recipe = RecordsData(Memory([{"id": index} for index in range(8)]),
                         validation=Memory([{"id": 100 + index} for index in range(4)]))
    train, _, _ = recipe.pipelines()
    generation = publish(train, _cache(), read="indexed", loading=read)
    data = loader(recipe, batch=2, cache=generation, read="indexed", loading=read)
    assert data.plan.manifest.source["sources"][0]["store"] == "generation"
    training, validation = data.train(DataPartition()), data.val(DataPartition())
    try:
        assert {int(index) for _ in range(4) for index in next(training)["id"]} == set(range(8))
        assert sorted(int(index) for batch in validation for index in batch["id"]) == [100, 101, 102, 103]
    finally:
        training.close()
        validation.close()
    with pytest.raises(ValueError):
        loader(RecordsData(Memory([{"id": -1}])), batch=2, cache=generation, read="indexed", loading=read)


def test_published_generation_exposes_only_declared_order_metadata_and_plan():
    import numpy as np

    source = Memory(
        [{"id": index, "_dew_length": index + 2, "_dew_valid_targets": index + 1} for index in range(4)]
    )
    cache, key = _cache(), Manifest()
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
    assert len(generation.plan_hash("offsets")) == 64


def test_transient_failures_never_certify_cache_coverage_but_permanent_rejections_do():
    from dew.data.pipeline import Rejected

    source = Memory([{"id": 0}, {"id": 1}])
    cache, key = _cache(), Manifest()
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
    first_key, second_key = Manifest(b'{"source":"one"}'), Manifest(b'{"source":"two"}')
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
                               Manifest(json.dumps({"source": index}).encode()))
        stopped[0]
        stopped.close()
    source, key = Memory([{"id": 99}]), Manifest(b'{"source":"kept"}')
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
    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), Manifest())
    try:
        assert rows[0]["id"] == 0
        with pytest.raises(OSError):
            rows.finish()
        assert len(attempts) == 1
    finally:
        rows.close()


def test_a_remote_upload_that_fails_ends_the_fill_waiting_for_its_room():
    """A fill that waits for spool room as a shard uploads is woken when the upload fails, after
    the cache's retries, and ends with the upload's error rather than waiting on bytes that will
    never free. The service fails only once the fill is waiting."""
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
    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), Manifest())
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
        assert [type(error) for error in raised] == [OSError] and len(attempts) == 2
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
    rows = cache.mapped(grain.MapDataset.source(source.records("indexed")), Manifest())
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
    root = os.environ.get("DEW_DATA_TEST_URL")
    if not root:
        pytest.skip("DEW_DATA_TEST_URL names a writable remote smoke prefix")
    source = Memory([{"id": index, "text": "remote"} for index in range(4)])
    key = Manifest(b'{"r2-smoke":"artifact-v1"}')
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
