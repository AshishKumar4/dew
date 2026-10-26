"""Automatic caching: a chain's leading deterministic maps read back from what an earlier run made."""

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import grain.python as grain
import numpy as np
import pytest

from dew.data import Loading, Memory, mix
from dew.data.cache import AUTOMATIC, automatic_root
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline, Rejected
from dew.data.sources.grain import from_grain

RECORDS = 8
READS = ["memory", "indexed"]


@dataclass
class Counted:
    """A fixed map that counts its calls, rejecting the records `rejects` names and changing what it
    makes on each call where `volatile`, as a live fetch may; one that costs more to run than to read
    back where `costly` (`dew.data.pipeline.Costly`), as a decode does."""

    volatile: bool = False
    rejects: frozenset[int] = frozenset()
    costly: bool = True
    calls: list[int] = field(default_factory=list, repr=False)

    def __call__(self, row):
        self.calls.append(int(row["id"]))
        if int(row["id"]) in self.rejects:
            return Rejected("odd")
        made = len(self.calls) if self.volatile else int(row["id"]) + 10
        return {**row, "value": np.int32(made)}


@dataclass
class Tokens:
    """A fixed map to a document of token ids under "text", as a learned tokenizer's, costly to run."""

    costly: bool = True

    def __call__(self, row):
        return {"text": np.arange(int(row["id"]), int(row["id"]) + 3 + int(row["id"]) % 4, dtype=np.int32)}


@dataclass
class IndexedVolatile:
    """A map by index that may make another record on each call, as a live fetch may, costly to run."""

    volatile: bool = True
    costly: bool = True
    calls: list[int] = field(default_factory=list, repr=False)

    def __call__(self, index, row):
        self.calls.append(index)
        return {**row, "value": np.int32(len(self.calls))}


def records(offset: int = 0) -> Memory:
    return Memory([{"id": np.int32(offset + index)} for index in range(RECORDS)])


def mapped(step: Counted, source: Memory | None = None) -> Pipeline:
    """`step` over `source` as a chain's leading map, its records shuffled after it: globally, or in a
    window where it rejects some, as a read by index shuffles no order a step can drop from."""
    pipeline = Pipeline(records() if source is None else source)
    if step.rejects:
        return pipeline.filter_map(step, identity="counted@1").shuffle(0, window=4)
    return pipeline.map(step, identity="counted@1").shuffle(0)


def read(chain: Pipeline, how: str, *, cache: bool = True, passes: int = 1) -> list[tuple[int, int]]:
    """`passes` whole passes of `chain`, batched by two, as `(id, value)` pairs in the order read."""
    loader = compile(chain.repeat(passes).batch(2, drop_remainder=False), read=how,
                     loading=Loading(workers=0, threads=1, read_buffer=1, cache=cache))
    stream = loader.train(DataPartition())
    try:
        return [(int(row), int(value)) for batch in stream
                for row, value in zip(batch["id"], batch["value"], strict=True)]
    finally:
        stream.close()


def generations(root: str | None = None) -> list[Path]:
    """The generations published under the automatic root."""
    named = Path(automatic_root(Loading()) if root is None else root)
    return sorted(named.rglob("CURRENT")) if named.exists() else []


@pytest.mark.parametrize("how", READS)
def test_a_run_reads_its_leading_maps_back_from_the_records_an_earlier_run_made(how):
    """With no `cache=`, a run's leading deterministic map over a named store is cached once a pass
    has read it whole: a later run of the same chain reads the same records back, calling the map on
    none, and a run of another map or over other records makes its own."""
    step = Counted()
    first = read(mapped(step), how)
    assert sorted(step.calls) == list(range(RECORDS)) and len(generations()) == 1
    again = Counted()
    assert read(mapped(again), how) == first and again.calls == []
    other = Counted()
    read(mapped(other, records(100)), how)
    assert sorted(other.calls) == list(range(100, 100 + RECORDS)) and len(generations()) == 2


@pytest.mark.parametrize("how", READS)
@pytest.mark.parametrize("rejects", [frozenset(), frozenset({1, 4})], ids=["kept", "rejecting"])
def test_a_chain_reads_the_same_records_and_plan_with_or_without_its_automatic_cache(how, rejects):
    """An automatic cache holds what its maps make, a rejection included, so a run reads the same
    records in the same order whether it fills the cache, reads it back or opts out, and its plan,
    which names no automatic cache, is the same: a position one run saves resumes in either. A read
    by index caches no map that rejects, whose cache it would read back as a stream, slower than it
    reruns the map."""
    def chain():
        return mapped(Counted(rejects=rejects))

    def plan(*, cache: bool):
        loader = compile(chain().repeat(2).batch(2, drop_remainder=False), read=how,
                         loading=Loading(workers=0, threads=1, read_buffer=1, cache=cache))
        assert loader.plan is not None
        return loader.plan.record()

    uncached = read(chain(), how, cache=False, passes=2)
    assert read(chain(), how, passes=2) == uncached
    assert bool(generations()) == (how == "memory" or not rejects)
    assert read(chain(), how, passes=2) == uncached
    assert plan(cache=True) == plan(cache=False)


@dataclass(frozen=True)
class Kept:
    """The records whose id is no multiple of three."""

    def __call__(self, row) -> bool:
        return int(row["id"]) % 3 != 0


LAYOUTS = {
    "windows": lambda: Pipeline(records()).map(Tokens(), identity="tokens@1").windows(4),
    "mixture": lambda: mix([Pipeline(records(offset)).map(Counted(), identity=f"counted-{offset}@1")
                            .filter(Kept(), identity="kept@1") for offset in (0, 100)]),
}


@pytest.mark.parametrize("how", READS)
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_a_laid_out_chain_keeps_its_plan_and_resumes_across_its_automatic_cache(how, layout):
    """The facts a plan derives of a chain's layout (windows cut from its documents, the members of a
    mixture read per share) are the same with or without an automatic cache beneath them, so a
    position saved with it resumes in a run without it."""
    def loader(*, cache: bool):
        return compile(LAYOUTS[layout]().repeat(2).batch(2, drop_remainder=False), read=how,
                       loading=Loading(workers=0, threads=1, read_buffer=1, cache=cache))

    # A first run reads past a pass of every corpus, which fills its caches (a mixture draws on without
    # end); the next reads them back.
    filling = loader(cache=True).train(DataPartition())
    try:
        list(itertools.islice(filling, 4 * RECORDS))
    finally:
        filling.close()
    cached = loader(cache=True)
    stream = cached.train(DataPartition())
    try:
        expected = [next(stream) for _ in range(2)]
        saved = stream.get_state()
        following = [next(stream) for _ in range(2)]
    finally:
        stream.close()
    assert generations() and expected
    uncached = loader(cache=False)
    assert cached.plan is not None and uncached.plan is not None
    assert cached.plan.record() == uncached.plan.record()
    resumed = uncached.train(DataPartition())
    try:
        resumed.set_state(saved)
        for batch, again in zip(following, (next(resumed) for _ in range(2)), strict=True):
            assert batch.keys() == again.keys()
            for name in batch:
                np.testing.assert_array_equal(np.asarray(batch[name]), np.asarray(again[name]), err_msg=name)
    finally:
        resumed.close()


def test_records_changed_between_two_compilations_are_read_as_they_are_now():
    """A store's records may change between two compilations: the second fingerprints them again, so it
    reads what its maps make of them now, not what an earlier run cached of the old ones."""
    rows = [{"id": np.int32(index)} for index in range(RECORDS)]
    first = read(mapped(Counted(), Memory(rows)), "indexed")
    rows[0] = {"id": np.int32(50)}
    second = read(mapped(Counted(), Memory(rows)), "indexed")
    assert (50, 60) in second and (50, 60) not in first


def test_a_root_this_process_may_not_write_caches_nothing_and_reads_on(monkeypatch, tmp_path):
    """An automatic cache stands only where the process may write its root: under a read-only home the
    run reads its records as it would with none, and writes nothing."""
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    monkeypatch.setenv(AUTOMATIC, str(locked / "cache"))
    try:
        step = Counted()
        assert read(mapped(step), "indexed") == read(mapped(Counted()), "indexed", cache=False)
        assert not (locked / "cache").exists()
    finally:
        locked.chmod(0o700)


def test_a_run_or_a_process_opts_out_of_its_automatic_cache(monkeypatch, tmp_path):
    """`Loading(cache=False)` keeps a run from caching, and `DEW_DATA_CACHE=off` every run of the
    process: the map runs on each, and nothing is written."""
    step = Counted()
    read(mapped(step), "indexed", cache=False)
    read(mapped(step), "indexed", cache=False)
    assert len(step.calls) == 2 * RECORDS and not generations()
    monkeypatch.setenv(AUTOMATIC, "off")
    assert automatic_root(Loading()) is None
    read(mapped(step), "indexed")
    assert len(step.calls) == 3 * RECORDS
    monkeypatch.setenv(AUTOMATIC, str(tmp_path / "elsewhere"))
    read(mapped(step), "indexed")
    assert len(generations(str(tmp_path / "elsewhere"))) == 1


def unnamed():
    """A Grain dataset given no name, whose content no cache can key."""
    return from_grain(grain.MapDataset.source([{"id": np.int32(index)} for index in range(RECORDS)]))


@pytest.mark.parametrize("chain", ["cheap step", "unkeyed step", "volatile step", "volatile indexed step",
                                   "unnamed store", "streamed", "image decode"])
def test_a_chain_no_cache_could_serve_or_speed_reads_as_written(chain):
    """A chain whose leading steps no cache could stand for, or none would speed, reads as it is
    written, never refused: a step that costs less than reading its records back, one with no
    identity, one that may make other records on each call, a store that names no content, and a
    streamed read write no automatic cache, and their steps run on every pass."""
    step = Counted(volatile=chain == "volatile step", costly=chain != "cheap step")
    how = "streamed" if chain == "streamed" else "indexed"
    if chain == "image decode":
        # A decode reads an image a record names by path from storage its content does not key.
        from dew.data import presets

        pixels = [{"id": np.int32(index), "image": np.full((4, 4, 3), index, np.uint8)}
                  for index in range(RECORDS)]
        compiled = compile(presets.images(Memory(pixels), 4, batch=2), read="indexed",
                           loading=Loading(workers=0, threads=1, read_buffer=1))
        stream = compiled.train(DataPartition())
        next(stream)
        stream.close()
        assert not generations()
        return
    if chain == "volatile indexed step":
        indexed = IndexedVolatile()
        read(Pipeline(records()).map_with_index(indexed, identity="indexed@1"), how, passes=2)
        assert len(indexed.calls) == 2 * RECORDS and not generations()
        return
    if chain == "unkeyed step":
        pipeline = Pipeline(records()).map(lambda row: step(row)).shuffle(0)
    elif chain == "unnamed store":
        pipeline = unnamed().map(step, identity="counted@1")
    else:
        pipeline = Pipeline(records()).map(step, identity="counted@1")
    read(pipeline, how, passes=2)
    assert len(step.calls) == 2 * RECORDS and not generations()


def test_a_chain_s_own_cache_takes_the_place_of_the_automatic_one(tmp_path):
    """A chain that writes a cache of its own gets no automatic one: its records fill the cache it
    names, and nothing else."""
    step = Counted()
    own = Pipeline(records()).map(step, identity="counted@1").cache(str(tmp_path / "own"))
    read(own.shuffle(0), "indexed")
    assert not generations() and len(generations(str(tmp_path / "own"))) == 1


def test_each_corpus_of_a_mixture_caches_its_own_leading_maps():
    """A mixture's corpora each cache their own leading maps, beneath the mix: a later run of the same
    mixture calls neither map and reads the same records."""
    def mixture(steps: tuple[Counted, Counted]) -> Pipeline:
        members = [Pipeline(records(offset)).map(step, identity=f"counted-{offset}@1")
                   for offset, step in zip((0, 100), steps, strict=True)]
        return mix(members, weights=[1.0, 1.0])

    steps = (Counted(), Counted())
    first = read(mixture(steps), "indexed")
    assert all(step.calls for step in steps) and len(generations()) == 2
    again = (Counted(), Counted())
    assert read(mixture(again), "indexed") == first
    assert not any(step.calls for step in again)
