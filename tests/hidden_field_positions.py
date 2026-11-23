"""The chains whose positions main saved under plan 15 (tests/fixtures/positions, `SAVED_UNDER_15` of
test_pipeline_earlier_plans), which described a step without the fields its repr hides: a read and phases
through such a step (`Remapped`), and a global order and phases of global orders batched by such a
function (`Stacked`); under plan 16, before a phases' position's version was read from its reader's,
phases through such a step. A module of
their own, whose bytes their positions hold, so an edit to another leaves them resumable."""

import dataclasses

import numpy as np
from saved_positions_support import READ, source

from dew.data import Memory, Phase, Steps, phases
from dew.data.compile import compile
from dew.data.pipeline import Pipeline


@dataclasses.dataclass(frozen=True)
class Remapped:
    """A configured record step: each id plus its entry in `table`, a table its repr hides, as a large
    array is."""

    table: np.ndarray = dataclasses.field(repr=False)

    def __call__(self, row):
        return {**row, "id": row["id"] + self.table[np.asarray(row["id"]) % len(self.table)]}


@dataclasses.dataclass(frozen=True)
class Stacked:
    """A batch function: the rows stacked, each id shifted by the first entry of `table`, a table its repr
    hides, which plan 15 described the function without."""

    table: np.ndarray = dataclasses.field(repr=False)

    def __call__(self, rows):
        return {name: np.stack([row[name] for row in rows]) + (self.table[0] if name == "id" else 0)
                for name in rows[0]}


def hidden_global():
    """40 ids in one global order read in memory, batched by a function whose table its repr hides."""
    rows = Memory({"id": np.arange(1, 41, dtype=np.int32)}, shard_rows=4)
    chain = Pipeline(rows).shuffle(seed=3).repeat().batch(2, batch_fn=Stacked(np.zeros(4, dtype=np.int32)))
    return compile(chain, read="memory", loading=READ)


def hidden_global_phases():
    """Two global orders read in two phases on the record clock, the first's batched by a function whose
    table its repr hides (`Stacked`), its order holding that function."""
    first = Memory({"id": np.arange(1, 41, dtype=np.int32)}, shard_rows=4)
    second = Memory({"id": np.arange(101, 141, dtype=np.int32)}, shard_rows=4)
    stacked = Stacked(np.zeros(4, dtype=np.int32))
    return compile(phases(Phase(Pipeline(first).shuffle(seed=3).repeat().batch(2, batch_fn=stacked),
                                Steps(6)),
                          Phase(Pipeline(second).shuffle(seed=4).repeat().batch(2, batch_fn=stacked))),
                   read="memory", loading=READ)


def hidden():
    """40 ids shuffled through a step whose table its repr hides (`Remapped`), in batches of two."""
    rows = Memory({"id": np.arange(1, 41, dtype=np.int32)}, shard_rows=4)
    chain = Pipeline(rows).shuffle(seed=3, window=8).map(Remapped(np.arange(8, dtype=np.int32))).repeat()
    return compile(chain.batch(2), read="streamed", loading=READ)


def hidden_phases():
    """Two sources read in two phases, the first's rows through a step whose table its repr hides."""
    return compile(phases(Phase(source(1).map(Remapped(np.arange(8, dtype=np.int32))), Steps(3)),
                          Phase(source(2))),
                   read="indexed", loading=READ)


SAVED = {"hidden": (hidden, 3), "hidden_phases": (hidden_phases, 5), "hidden_global": (hidden_global, 3),
         "hidden_global_phases": (hidden_global_phases, 7)}
"""Each saved position's loader and the steps read before it was saved."""
