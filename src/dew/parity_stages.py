"""Check 1's two stages (`dew.parity`): how many further seeds a site that does not pass its first draws.
Standard library alone, so a plan that shards the suite (tools/armada/ci.py) reads it by its path without
JAX."""

REDRAW = 4
"""A failing site's pooled ratio is drawn again over REDRAW times its seeds before it fails (check 1's second
stage, `dew.parity._staged`)."""


def staged_units(seeds: int) -> int:
    """The units check 1's two stages draw at a site of `seeds` declared seeds: its seeds and REDRAW times
    as many further ones. A weighed check's (`conditioning=`) are its own declared seeds' so, drawn past
    these."""
    return (1 + REDRAW) * seeds
