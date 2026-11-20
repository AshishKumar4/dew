"""Check 1's two stages (`dew.parity`): how many further seeds a site that does not pass its first draws.
Standard library alone, so a plan that shards the suite (tools/armada/ci.py) reads it by its path without
JAX."""

REDRAW = 4
"""A failing site's pooled ratio is drawn again over REDRAW times its seeds before it fails (check 1's second
stage, `dew.parity._staged`)."""

SIZING_FROM = 1 << 20
"""The first seed a sizing draws (`seeds_for`, paired or of the reference against itself): past every site's
verdict seeds and below every null run's (tools/gate_null_runs.py, from 2^16 runs of at least 80 seeds on), so
the count a site declares is fixed by draws its verdict never sees; a recording's pilots are recorded from it
(tools/block_override_reference.py, without JAX)."""


def staged_units(seeds: int) -> int:
    """The units check 1's two stages draw at a site of `seeds` declared seeds: its seeds and REDRAW times
    as many further ones. A weighed check's (`conditioning=`) are its own declared seeds' so, drawn past
    these."""
    return (1 + REDRAW) * seeds
