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


class UpstreamBlocks:
    """An upstream's computation on a site's draws, a `block` of seeds a call: `run_block(draws)` runs a list
    of `draw(seed)`s at once (one call of the upstream's, its draws folded into its own batch) and returns
    their results in order, each draw found by its key's bytes (`keyed(*args)`, `digest`, the cotangents left
    out where they follow). A draw not yet run takes the next blocks from `first`, up to `most` seeds past it
    (none where None); one this arm drew itself (`drawn`, a recording's draws, asked in order) starts its
    block at its own seed. `bounded`, one block's results held at a time: a recording's sizing draws whole
    took every output of every block held. The live cells' arm (`upstream_arm`) and a recording's answers
    (tools/matrix_reference.py) alike."""

    def __init__(self, run_block, draw, block: int, *, most: int | None = None, first: int = 0,
                 keyed=lambda *args: args, bounded: bool = False):
        self.run_block, self.draw, self.block, self.most, self.first = run_block, draw, block, most, first
        self.keyed, self.bounded = keyed, bounded
        self.held: dict[str, object] = {}
        self.seeds: dict[str, int] = {}
        self.next = first

    def drawn(self, seed: int) -> tuple:
        """Seed `seed`'s draw (`draw`'s), its seed kept by its key, so the block that answers it starts
        there."""
        args = self.draw(seed)
        self.seeds[digest(self.keyed(*args))] = seed
        return args

    def _run(self, start: int) -> None:
        drawn = [self.draw(seed) for seed in range(start, start + self.block)]
        self.next = start + self.block
        if self.bounded:
            self.held = {}
        for args, result in zip(drawn, self.run_block(drawn), strict=True):
            self.held[digest(self.keyed(*args))] = result

    def __call__(self, *args):
        key = digest(self.keyed(*args))
        if key not in self.held and key in self.seeds:
            self._run(self.seeds[key])
        while key not in self.held and self.most is not None and self.next < self.first + self.most:
            self._run(self.next)
        return self.held[key]


def upstream_arm(run_block, draw, block: int, *, most: int, first: int = 0, keyed=lambda *args: args):
    """An arm the gate runs as given whose results are an upstream's (`UpstreamBlocks`), run on a site's
    seeds a `block` at a time from `first` (a verdict's from 0, its declared seeds and their redraw the first
    block; a sizing's from `parity.SIZING_FROM`), up to `most` seeds past it."""
    return as_given(UpstreamBlocks(run_block, draw, block, most=most, first=first, keyed=keyed))
