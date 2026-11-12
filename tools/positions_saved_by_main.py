"""Positions saved by main before phases shared corpora (row 5497's fixtures): written by this script at the
sha it runs at, into the directory it is given."""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, "tests")

from test_run_budgets import _LENGTHS, _SPLIT, _pieces
from test_step_phases import READ, chain

from dew.data import Memory
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline, mix

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)


def saved(loader, steps: int, name: str) -> None:
    stream = loader.train(DataPartition())
    list(itertools.islice(stream, steps))
    (out / name).write_bytes(stream.get_state())
    stream.close()


saved(compile(chain(3), read="indexed", loading=READ), 2, "step_phases.position")
for layout, steps in (("strided", 2), ("held", 3)):
    laid, read = _SPLIT[layout]
    corpora = [Pipeline(Memory([{"id": document["id"] + offset} for document in _pieces()], shard_rows=3))
               .named(name) for name, offset in (("left", 0), ("right", 100_000))]
    saved(compile(laid(mix(corpora)).read(read).batch(2), read=read, examples=2 * len(_LENGTHS)), steps,
          f"bounded_mixture_{layout}.position")
print("written", sorted(path.name for path in out.iterdir()))
