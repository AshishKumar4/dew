"""Scratch probe (row 3684): the worst-row rule's false reds and power, the factor by the row's nominal width
against the factor of width one where one element carries most of the reference's worst row's error."""
import json

import numpy as np
import pytest

FACTORS = {1: 90.0, 2: 11.0}
THRESHOLD = 0.9


def factor(width):
    return FACTORS.get(width, 4.0)


def verdicts(rows, width, seeds, shared, fault, runs, chunk, dominance, amplified=True):
    rng = np.random.default_rng([0x3684, rows, width, seeds, int(shared), int(fault * 10), chunk])
    span = 5 * seeds
    reds = 0
    for _ in range(runs):
        truth = np.sqrt(np.mean(np.square(rng.standard_normal((span, rows, width))), axis=2))
        scale = 2.0 ** rng.integers(0, 25, (span, rows))
        arms = []
        for arm in range(2):
            errors = rng.standard_normal((span, rows, width))
            if amplified:
                errors[..., 0] *= scale if shared else 2.0 ** rng.integers(0, 25, (span, rows))
            if arm == 0 and fault:
                # Dew's row off by `fault` times on one row of one seed in four: a row that differs.
                hit = rng.random((span, rows)) < 0.25 / rows
                errors[hit] *= fault
            sq = np.square(errors)
            arms.append((np.sqrt(np.mean(sq, axis=2)) / truth, sq.max(axis=2) / sq.sum(axis=2)))

        def red(stop):
            (mine, _), (theirs, share) = arms[0], arms[1]
            mine, theirs, share = mine[:stop], theirs[:stop], share[:stop]
            at = np.unravel_index(np.argmax(theirs), theirs.shape)
            f = factor(width)
            if dominance and width > 2 and share[at] > THRESHOLD:
                f = factor(1)
            return mine.max() > f * theirs.max()

        if red(seeds) and red(span):
            reds += 1
    return reds


LAYOUTS = [(rows, width, seeds, shared) for rows in (1, 4, 16) for width in (3, 4, 8, 16, 64) for seeds in (8, 64)
           for shared in (False, True, None)]


@pytest.mark.parametrize("fault", [0.0, 10.0])
@pytest.mark.parametrize("rows, width, seeds, shared", LAYOUTS)
def test_probe(rows, width, seeds, shared, fault):
    out = {"rows": rows, "width": width, "seeds": seeds, "shared": shared, "fault": fault, "runs": 4000}
    for dominance in (False, True):
        out[f"reds_{int(dominance)}"] = verdicts(rows, width, seeds, bool(shared), fault, 4000, 0, dominance,
                                                 amplified=shared is not None)
    print("\nDOMINANCE " + json.dumps(out))
