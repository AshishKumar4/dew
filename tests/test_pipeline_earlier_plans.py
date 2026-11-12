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
from dew.data.pipeline import Pipeline


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
