"""The false-red rate of check 1's worst-row rule (tests/reference_error.py `row_factor`,
NARROW_WIDTHS) on equally exact pairs as sites run it: both arms on one truth, so their rows share their
sizes; two stages, 8 declared seeds then REDRAW times as many more; ROWS rows a seed of `width` values,
truths and errors standard normal. Prints a chunk's count of red runs at a factor, from the chunk's own
stream, as one JSON line.

Usage: python tools/row_factor_rates.py <width> <factor> <runs> <chunk>
"""

import json
import sys

import numpy as np

ROWS = 256
DECLARED, ALL = 8, 40
"""A site's declared seeds and its two stages' seeds, 8 and (1 + REDRAW) times 8."""

def reds(width: int, factor: float, runs: int, chunk: int, block: int = 200) -> int:
    """The runs of `runs` whose worst rows fail at `factor` at both stages, as check 1 fails a site."""
    rng = np.random.default_rng([0x5A, width, chunk])
    count = 0
    for start in range(0, runs, block):
        size = min(block, runs - start)
        truth = np.sqrt(np.mean(np.square(rng.standard_normal((size, ALL, ROWS, width))), axis=3))
        mine, theirs = (np.sqrt(np.mean(np.square(rng.standard_normal((size, ALL, ROWS, width))), axis=3))
                        / truth for _ in range(2))
        first = mine[:, :DECLARED].max(axis=(1, 2)) > factor * theirs[:, :DECLARED].max(axis=(1, 2))
        second = mine.max(axis=(1, 2)) > factor * theirs.max(axis=(1, 2))
        count += int((first & second).sum())
    return count


if __name__ == "__main__":
    width, factor, runs, chunk = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    print(json.dumps({"width": width, "factor": factor, "runs": runs, "chunk": chunk,
                      "reds": reds(width, factor, runs, chunk)}))
