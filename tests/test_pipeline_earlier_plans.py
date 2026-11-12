"""Earlier data plans read by this one: a version 7 plan and position of a streamed read of two shares,
which version 8 reads as 7 did where it reads the order alike, refusing it otherwise; and a plan of 13,
which printed what 14 describes by its structure."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from dew.data import Memory
from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loading
from dew.data.pipeline import Pipeline


@pytest.mark.parametrize(("taken", "bounded", "workers", "pools", "alike"), [
    (True, False, 0, {}, True), (True, False, 2, {}, False), (True, True, 0, {}, False),
    (True, False, 0, {"0": 2}, False), (False, False, 2, {"0": 2}, True)])
def test_a_version_7_streamed_plan_of_two_shares_reads_as_8s_where_8_reads_its_takes_alike(
        taken, bounded, workers, pools, alike):
    """A version 7 plan of a streamed mixture read by two shares, a member's order taking some of it or
    not (a member's `Limit`, within the mixture's order): version 8 reads it as 7 did where no take reads
    it, or no loading workers read it, a member's own pool among them, and it is no bound in examples (its
    manifest's steps hold no `MarkConsumed`); a take read by workers, or a bound, 8 deals otherwise (row
    3350's review)."""
    from dew.data.plan import read_otherwise_at_8

    steps = [{"identity": "Map(step=MarkConsumed())"}] if bounded else []
    member = {"read": "streamed", "stages": ["Limit(count=5)"] if taken else ["Map(step=Shout())"]}
    saved = {"version": 7, "order": {"read": "streamed", "corpora": [member, member]},
             "members": {"0": "a", "1": "b"}, "manifest": {"steps": steps},
             "topology": {"shares": 2, "share": 0, "workers": workers, "pools": pools}}
    assert (read_otherwise_at_8(saved) is None) is alike


@pytest.mark.parametrize("kind", ["plain", "take", "bound"])
def test_a_version_7_position_of_a_two_share_streamed_read_resumes_where_8_reads_it_alike(kind):
    """Share 0's position of a two-share streamed read of 24 records in shards of 4, saved after two
    batches by a version 7 main (tests/fixtures/version7_positions.json, written at its commit): read
    plainly, or taking 18 rows, version 8 reads the order as 7 did, so the position resumes onto the
    rows 7 read next; bounded to 10 examples, which version 8 deals to the shares otherwise, it is refused
    before a row is read (row 3364). Its chain, named none, shuffles by its seed alone under version 9 too,
    so the position reads on under 9 (rows 3840, 4063)."""
    recorded = json.loads((Path(__file__).with_name("fixtures") / "version7_positions.json").read_text())
    saved = recorded["positions"][kind]
    chain = Pipeline(Memory({"id": np.arange(1, 25, dtype=np.int32)}, shard_rows=4)).shuffle(seed=3, window=8)
    if kind == "take":
        chain = chain.repeat().take(18)
    loader = compile(chain.batch(2), read="streamed", loading=Loading(workers=0),
                     examples=10 if kind == "bound" else None)
    stream = loader.train(DataPartition(index=0, count=2))
    try:
        if kind == "bound":
            with pytest.raises(ValueError):
                stream.set_state(saved["state"].encode())
            return
        stream.set_state(saved["state"].encode())
        assert [np.asarray(next(stream)["id"]).tolist() for _ in range(2)] == saved["after"]
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


@pytest.mark.parametrize("read", ["streamed", "indexed"])
@pytest.mark.parametrize("version", [13, 14])
def test_a_position_resumes_under_the_plan_version_that_saved_it_through_a_grown_take(read, version):
    """A position saved under data plan 13, which printed its order's stages and its steps' configurations,
    or 14, which describes them by their structure (`STRUCTURED`), resumes compiled at the version that
    saved it: onto the rows read next, through a take grown since, which reads on past the old bound; a
    configured step whose configuration changed is refused, its rows not the saved run's."""
    def loader(count: int, factor: int = 2):
        return compile(_taken(count, factor), read=read, loading=Loading(workers=0), plan_version=version)

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
