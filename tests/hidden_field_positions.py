"""The chains whose positions main saved under plan 15 (tests/fixtures/positions/main-72141706a), which
described a step without the fields its repr hides: a read and phases through such a step (`Remapped`). A
module of their own, whose bytes their positions hold, so an edit to another leaves them resumable."""

import dataclasses

import numpy as np
from saved_positions_support import READ, source

from dew.data import Memory, phases
from dew.data.compile import compile
from dew.data.pipeline import Pipeline


@dataclasses.dataclass(frozen=True)
class Remapped:
    """A configured record step: each id plus its entry in `table`, a table its repr hides, as a large
    array is."""

    table: np.ndarray = dataclasses.field(repr=False)

    def __call__(self, row):
        return {**row, "id": row["id"] + self.table[np.asarray(row["id"]) % len(self.table)]}


def hidden():
    """40 ids shuffled through a step whose table its repr hides (`Remapped`), in batches of two."""
    rows = Memory({"id": np.arange(1, 41, dtype=np.int32)}, shard_rows=4)
    chain = Pipeline(rows).shuffle(seed=3, window=8).map(Remapped(np.arange(8, dtype=np.int32))).repeat()
    return compile(chain.batch(2), read="streamed", loading=READ)


def hidden_phases():
    """Two sources read in two phases, the first's rows through a step whose table its repr hides."""
    return compile(phases((source(1).map(Remapped(np.arange(8, dtype=np.int32))), 3), (source(2), None)),
                   read="indexed", loading=READ)


SAVED = {"hidden": (hidden, 3), "hidden_phases": (hidden_phases, 5)}
"""Each saved position's loader and the steps read before it was saved."""
