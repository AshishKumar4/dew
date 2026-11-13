"""Positions a main saves under the plan version before this Dew's (tests/fixtures/positions): run at the sha
that saves them, from the repository's root, with `tests/saved_positions_support.py` beside it, writing each
loader's position after its steps (`SAVED`) into the directory it is given."""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, "tests")

from saved_positions_support import SAVED

from dew.data.dataset import DataPartition

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for name, (loader, steps) in SAVED.items():
    stream = loader().train(DataPartition())
    list(itertools.islice(stream, steps))
    (out / f"{name}.position").write_bytes(stream.get_state())
    stream.close()
print("written", sorted(path.name for path in out.iterdir()))
