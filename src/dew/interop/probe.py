"""The verification probe's token grid and draw (`dew.interop.verify`), in NumPy alone: a reference's
generator in its release's environment reads it by its path, as it reads the release pins."""

import numpy as np

BATCH, LENGTH = 2, 12
"""The probe's token grid, which the committed fixtures share."""


def probe_ids(vocab_size: int, seed: int = 0, shape: tuple[int, int] = (BATCH, LENGTH)) -> np.ndarray:
    """Return the probe's token ids of draw `seed`, `shape` (BATCH x LENGTH, the fixed ones a reference
    is run on, at seed 0)."""
    return np.random.RandomState(7 + seed).randint(0, vocab_size, shape).astype(np.int32)
