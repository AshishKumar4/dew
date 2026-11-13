"""Earlier data plans: a plan of 13, which printed what 14 describes by its structure, read by this Dew; and
one of 7, which it reads none of."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from dew.data import Memory
from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loading
from dew.data.pipeline import Pipeline, mix


def test_a_position_a_plan_this_dew_reads_none_of_saved_is_refused():
    """Share 0's position of a two-share streamed read, saved by a version 7 main
    (tests/fixtures/version7_positions.json, written at its commit), is refused by this Dew's read of the
    same chain: it reads plans of the version before its own and its own alone."""
    recorded = json.loads((Path(__file__).with_name("fixtures") / "version7_positions.json").read_text())
    chain = Pipeline(Memory({"id": np.arange(1, 25, dtype=np.int32)}, shard_rows=4)).shuffle(seed=3, window=8)
    loader = compile(chain.batch(2), read="streamed", loading=Loading(workers=0))
    stream = loader.train(DataPartition(index=0, count=2))
    try:
        with pytest.raises(ValueError):
            stream.set_state(recorded["positions"]["plain"]["state"].encode())
    finally:
        stream.close()


@dataclasses.dataclass(frozen=True)
class Scaled:
    """A configured record step: each id times `factor`."""

    factor: int

    def __call__(self, row):
        return {**row, "id": row["id"] * self.factor}


def _taken(count: int, factor: int = 2):
    """A shuffled chain of 40 ids through a configured step, taken to `count` records, in batches of two."""
    rows = Memory({"id": np.arange(1, 41, dtype=np.int32)}, shard_rows=4)
    return Pipeline(rows).shuffle(seed=3, window=8).map(Scaled(factor)).repeat().take(count).batch(2)


def _ids(stream, batches: int) -> list[list[int]]:
    return [np.asarray(next(stream)["id"]).tolist() for _ in range(batches)]


@pytest.mark.parametrize("version", [13, 14])
def test_a_position_resumes_under_the_plan_version_that_saved_it_through_a_grown_take(version):
    """A position saved under data plan 13, which printed its order's stages and its steps' configurations,
    or 14, which describes them by their structure (`STRUCTURED`), resumes compiled at the version that
    saved it: onto the rows read next, through a take grown since, which reads on past the old bound; a
    configured step whose configuration changed is refused, its rows not the saved run's."""
    def loader(count: int, factor: int = 2):
        return compile(_taken(count, factor), read="streamed", loading=Loading(workers=0),
                       plan_version=version)

    stream = loader(20).train(DataPartition())
    _ids(stream, 3)
    saved = stream.get_state()
    after = _ids(stream, 4)
    stream.close()
    for count in (20, 30):
        resumed = loader(count).train(DataPartition())
        resumed.set_state(saved)
        assert _ids(resumed, 4) == after, count
        resumed.close()
    changed = loader(20, factor=3).train(DataPartition())
    with pytest.raises(ValueError):
        changed.set_state(saved)
    changed.close()


@pytest.mark.parametrize("mixed", [False, True], ids=["one corpus", "mixture"])
def test_a_streamed_position_of_13_resumes_in_a_read_compiled_by_this_dew_as_13_read_it(mixed):
    """A streamed read per share saved under plan 13, of one store or a mixture of two, resumes in the same
    chain compiled at this Dew's version as 13 read it (`dew.data.readers._PerShare.resumed_at`): onto the
    rows 13 read next, its plan 13's whole (its spelling, facts and members), so the position it saves in
    turn resumes again onto the rows after."""
    chain = mix([_named("a", 1), _named("b", 2)], seed=4).batch(2) if mixed else _taken(40)
    at_13 = compile(chain, read="streamed", loading=Loading(workers=0), plan_version=13)
    whole = at_13.train(DataPartition())
    ids = _ids(whole, 8)
    whole.close()
    stream = at_13.train(DataPartition())
    _ids(stream, 2)
    saved = stream.get_state()
    stream.close()
    for start in (2, 5):
        resumed = compile(chain, read="streamed", loading=Loading(workers=0)).train(DataPartition())
        try:
            resumed.set_state(saved)
            assert _ids(resumed, 3) == ids[start:start + 3]
            saved = resumed.get_state()
            assert json.loads(saved)["dew_data_plan"]["version"] == 13
        finally:
            resumed.close()

def _saved(chain, **options) -> tuple[bytes, list[list[int]]]:
    """A streamed per-share read of `chain`, saved after two batches, and the two it reads next."""
    stream = compile(chain, read="streamed", loading=Loading(workers=0), **options).train(DataPartition())
    try:
        _ids(stream, 2)
        return stream.get_state(), _ids(stream, 2)
    finally:
        stream.close()


def _edited(state: bytes, edit) -> bytes:
    """`state`, a per-share position, with `edit` applied to its decoded record."""
    decoded = json.loads(state)
    edit(decoded)
    return json.dumps(decoded).encode()


def test_a_position_read_by_another_rule_is_refused_naming_it():
    """A position whose plan read by a rule this Dew reads otherwise (a mix's schedule) is refused, the
    refusal naming the rule (docs/design/data.md, the rules); the position as saved resumes."""
    chain = mix([_named("a", 1), _named("b", 2)], seed=4).batch(2)
    saved, after = _saved(chain)

    def other_mix(decoded):
        decoded["dew_data_plan"]["rules"]["mix"] = "another-schedule"

    for state, refused in ((_edited(saved, other_mix), True), (saved, False)):
        stream = compile(chain, read="streamed", loading=Loading(workers=0)).train(DataPartition())
        try:
            if refused:
                with pytest.raises(ValueError):
                    stream.set_state(state)
            else:
                stream.set_state(state)
                assert _ids(stream, 2) == after
        finally:
            stream.close()


def test_a_position_another_grain_wrote_resumes_where_its_iterators_read_it():
    """A consumed mixture's position whose iterator state another Grain wrote (its `grain` rule) is restored
    through the iterators that own it, and resumes onto the rows read next where they read it; one they
    cannot read (a state of another shape) is refused when it is restored."""
    chain = mix([_named("a", 1), _named("b", 2)], seed=4).batch(2)
    saved, after = _saved(chain)

    def other_grain(decoded):
        decoded["dew_data_plan"]["rules"]["grain"] = "0.0.1"

    def unreadable(decoded):
        other_grain(decoded)
        decoded["iterator"] = {"renamed": decoded["iterator"]}

    stream = compile(chain, read="streamed", loading=Loading(workers=0)).train(DataPartition())
    try:
        stream.set_state(_edited(saved, other_grain))
        assert _ids(stream, 2) == after
    finally:
        stream.close()
    stream = compile(chain, read="streamed", loading=Loading(workers=0)).train(DataPartition())
    try:
        with pytest.raises(ValueError):
            stream.set_state(_edited(saved, unreadable))
    finally:
        stream.close()


@dataclasses.dataclass(frozen=True)
class Recut(Memory):
    """A store of the same content and options as `Memory`'s, cut into its shards in reverse: a store a
    later Dew cuts otherwise, read in another order."""

    def manifest(self):
        cut = super().manifest()
        return dataclasses.replace(cut, shards=cut.shards[::-1])


def test_a_stream_its_store_cuts_into_other_shards_is_refused():
    """A streamed position resumes where its store cuts its content into the shards the read that saved it
    read in turn, and is refused where a store of the same content and options cuts it otherwise (its
    `shards` fact)."""
    ids = {"id": np.arange(1, 25, dtype=np.int32)}

    def chain(kind) -> Pipeline:
        return Pipeline(kind(ids, identity="ids", shard_rows=4)).repeat().batch(2)

    saved, after = _saved(chain(Memory))
    for kind, refused in ((Memory, False), (Recut, True)):
        stream = compile(chain(kind), read="streamed", loading=Loading(workers=0)).train(DataPartition())
        try:
            if refused:
                with pytest.raises(ValueError):
                    stream.set_state(saved)
            else:
                stream.set_state(saved)
                assert _ids(stream, 2) == after
        finally:
            stream.close()


def _named(name: str, tag: int) -> Pipeline:
    """A corpus of 16 ids named `name`, each the `tag`th corpus's, read on without end."""
    return Pipeline(Memory({"id": np.arange(tag * 100, tag * 100 + 16, dtype=np.int32)}, shard_rows=4),
                    name=name).repeat()
