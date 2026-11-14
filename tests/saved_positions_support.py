"""The chains whose positions a main saved (`tools/positions_saved_by_main.py`, `tests/fixtures/positions`),
each step declaring its identity or defined here, so an edit elsewhere in the tests leaves the positions
resumable: under the plan version before this Dew's, phases on their own streams, a bounded mixture laid out
and shuffled, a shuffled chain through a configured step taken to a bound, streamed and read into memory,
and a seeded mix of two named corpora; before the condition encoders moved to dew.interop (row 5996), phases
and a mix whose steps depend on one (`Conditioned`)."""

import dataclasses
from collections.abc import Mapping

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


@dataclasses.dataclass(frozen=True)
class Scaled:
    """A configured record step: each id times `factor`."""

    factor: int

    def __call__(self, row):
        return {**row, "id": row["id"] * self.factor}


def taken(count: int = 20, factor: int = 2, read: str = "streamed"):
    """40 ids shuffled through a configured step (`Scaled`), taken to `count` records, in batches of two."""
    rows = Memory({"id": np.arange(1, 41, dtype=np.int32)}, shard_rows=4)
    chain = Pipeline(rows).shuffle(seed=3, window=8).map(Scaled(factor)).repeat().take(count).batch(2)
    return compile(chain, read=read, loading=READ)


def named(name: str, tag: int) -> Pipeline:
    """A corpus of 16 ids named `name`, each the `tag`th corpus's, read on without end."""
    return Pipeline(Memory({"id": np.arange(tag * 100, tag * 100 + 16, dtype=np.int32)}, shard_rows=4),
                    name=name).repeat()


def mixture():
    """Two named corpora mixed by a seed, streamed."""
    return compile(mix([named("a", 1), named("b", 2)], seed=4).batch(2), read="streamed", loading=READ)


@dataclasses.dataclass(frozen=True)
class Conditioned:
    """A record step whose rows depend on a condition encoder, as a diffusion run's caption step's do
    (`dew.inputs.InputSpec.dependencies`): its identity names the encoder's class by the path the saving Dew
    imports it from, dew.inputs.encoders before row 5996 and dew.interop.encoders since. Each row as it is."""

    def dependencies(self) -> Mapping[str, object]:
        from dew.inputs import CLIPText
        from dew.registry import import_path

        return {"encoder": {"class": import_path(CLIPText), "fields": {"checkpoint": "saved"}}}

    def __call__(self, row):
        return row


def conditioned_phases():
    """`step_phases` with the first phase's rows through a step that depends on an encoder."""
    return compile(phases((source(1).map(Conditioned()), 3), (source(2), None)), read="indexed", loading=READ)


def conditioned_mixture():
    """`mixture` with the first corpus's rows through a step that depends on an encoder: a member of a
    streamed mix, kept by its digest."""
    return compile(mix([named("a", 1).map(Conditioned()), named("b", 2)], seed=4).batch(2), read="streamed",
                   loading=READ)


SAVED = {"step_phases": (step_phases, 2), "step_phases_later": (step_phases, 5),
         "bounded_mixture_strided": (lambda: bounded_mixture("strided"), 2),
         "bounded_mixture_held": (lambda: bounded_mixture("held"), 3), "taken": (taken, 3),
         "taken_in_memory": (lambda: taken(read="memory"), 3), "mixture": (mixture, 3),
         "conditioned_phases": (conditioned_phases, 5), "conditioned_mixture": (conditioned_mixture, 3)}
"""Each saved position's loader and the steps read before it was saved."""
