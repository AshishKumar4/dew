"""The chains whose positions main saved before phases shared corpora (`tools/positions_saved_by_main.py`,
`tests/fixtures/positions`), each step declaring its identity, so an edit elsewhere in the tests leaves the
positions resumable: phases on their own streams, and a bounded mixture laid out and shuffled."""

import numpy as np
from test_chain_compile import Tokens

from dew.data import Memory, phases
from dew.data.compile import compile
from dew.data.dataset import Loading
from dew.data.pipeline import Pipeline, mix

READ = Loading(workers=0, threads=1, read_buffer=1)

LENGTHS = (3, 11, 5, 17, 2, 9, 4, 13, 6, 8, 3, 21)
"""Documents' lengths, several longer than a window, so a layout cuts them across two, three or more."""


def fits(row) -> bool:
    """Whether a row fits its window: one in five of the lengths does not."""
    return int(row["length"]) % 5 != 0


def source(tag: int):
    """48 rows of the `tag`th source, of uneven lengths, some dropped, batched by 128 tokens: read per
    share."""
    rows = [{"id": np.int32(1000 * tag + index), "length": np.int32(3 + index % 11)} for index in range(48)]
    return (Memory(rows).map(Tokens(), identity="saved-tokens@1")
            .filter(fits, reason="too_long", identity="saved-fits@1").repeat().batch(tokens=128))


def step_phases():
    """Two sources read in two phases on the step clock, the first's to step 3."""
    return compile(phases((source(1), 3), (source(2), None)), read="indexed", loading=READ)


def bounded_mixture(layout: str):
    """Two corpora of documents mixed, laid out by `layout` (`strided` windows shuffled in a window, read
    streamed, or `held` packing shuffled whole, read into memory), read to two of each's examples."""
    pieces = [{"id": np.arange(length, dtype=np.int32) + 1000 * (index + 1)}
              for index, length in enumerate(LENGTHS)]
    corpora = [Pipeline(Memory([{"id": piece["id"] + offset} for piece in pieces], shard_rows=3)).named(name)
               for name, offset in (("left", 0), ("right", 100_000))]
    if layout == "strided":
        laid, read = mix(corpora).windows(4, stride=2, fields=("id",)).shuffle(seed=8, window=5), "streamed"
    else:
        laid, read = mix(corpora).pack(8, fields=("id",), policy="concat").shuffle(seed=4), "memory"
    return compile(laid.read(read).batch(2), read=read, examples=2 * len(LENGTHS))


SAVED = {"step_phases": (step_phases, 2), "bounded_mixture_strided": (lambda: bounded_mixture("strided"), 2),
         "bounded_mixture_held": (lambda: bounded_mixture("held"), 3)}
"""Each saved position's loader and the steps read before it was saved."""
