"""Which leaves of one request differ between a corpus run's task (from_run) and its export loaded back."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "tests")
from option_witness import answers  # noqa: E402
from test_saved_runs import extracted  # noqa: E402

from dew._cli.main import main  # noqa: E402
from dew.config import RunConfig  # noqa: E402
from dew.inference.loading import pipeline  # noqa: E402
from dew.registry import objectives  # noqa: E402

root = Path(sys.argv[3])
run = extracted(root, sys.argv[1], sys.argv[2])
task = objectives[RunConfig.load(str(run)).objective.name].saved_task
assert main(["export", str(run), str(root / "release")]) == 0
loaded, released = task.from_run(str(run)), pipeline(str(root / "release"))
print("types", type(loaded).__name__, type(released).__name__)
for name in ("precision", "layout", "calibration", "dtype"):
    a, b = getattr(loaded, name, None), getattr(released, name, None)
    print(name, "same" if a == b else f"DIFFERS\n  run:     {a!r}\n  release: {b!r}"[:1500])
for a, b in zip(answers(loaded), answers(released), strict=False):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        same = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        print("array", a.dtype, b.dtype, a.shape, b.shape, "same" if same else
              f"max diff {np.max(np.abs(a.astype(float) - b.astype(float))) if a.shape == b.shape else 'shape'}")
    else:
        print("leaf", "same" if a == b else f"{a!r} != {b!r}"[:300])
