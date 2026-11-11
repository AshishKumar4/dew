"""An upstream's own computation as a check-1 arm (`upstream_arm`): run in its GPU tier reference interpreter
a block of a site's draws at a time, as the gate's reference arm (`reference_error.as_given`), each draw read
back by its operands' bytes."""

import hashlib

import jax
import numpy as np
from reference_error import as_given


def digest(operands) -> str:
    """A draw's key: its operands' bytes, which no two seeds share."""
    leaves = jax.tree.leaves(operands)
    return hashlib.sha256(b"".join(np.asarray(leaf).tobytes() for leaf in leaves)).hexdigest()


def upstream_arm(run_block, draw, block: int, *, most: int, first: int = 0, keyed=lambda *args: args):
    """An arm the gate runs as given whose results are an upstream's: `run_block(draws)` running a list of
    `draw(seed)`s at once (one call of the upstream's, its draws folded into its own batch) and returning
    their results in order, run on a site's seeds a `block` at a time from `first` (a verdict's from 0, its
    declared seeds and their redraw the first block; a sizing's from `parity.SIZING_FROM`); a draw not yet
    run takes the next blocks, up to `most` seeds past `first`. A draw is found by its key's bytes
    (`keyed(*args)`, the arguments the gate passes it, the cotangents left out where they follow)."""
    held: dict[str, object] = {}
    ran = [first]

    def extend():
        drawn = [draw(seed) for seed in range(ran[0], ran[0] + block)]
        ran[0] += block
        for args, result in zip(drawn, run_block(drawn), strict=True):
            held[digest(keyed(*args))] = result

    def reference(*args):
        key = digest(keyed(*args))
        while key not in held and ran[0] < first + most:
            extend()
        return held[key]

    return as_given(reference)
