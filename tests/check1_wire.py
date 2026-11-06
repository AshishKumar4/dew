"""Check 1's map and judge contract, the one owner of what a sharded site and the plan that shards it share
(tests/map_judge.py, tests/reference_error.py and tools/armada/ci.py, which loads this file by its path): the
second stage's REDRAW, the environment that runs a site's test as a map task or as their judge, and the units
the stages draw. Standard library alone, so the plan reads it without JAX."""

REDRAW = 4
"""A failing site's pooled ratio is drawn again over REDRAW times its seeds before it fails (check 1's second
stage, `reference_error._staged`)."""

DRAWS = "DEW_CHECK1_DRAWS"
"""The environment variable that runs every sharded site as one map task: `start:stop`, the units the
task does (check 1's: its seeds), its records written to RECORDS (`map_judge.record_units`). The site
judges nothing there and passes once they are written."""

RECORDS = "DEW_CHECK1_RECORDS"
"""The directory a map task (DRAWS) writes its records to, one file a site and range."""

POOL = "DEW_CHECK1_POOL"
"""The environment variable that runs every sharded site as the judge of its map tasks (DRAWS): a directory
whose tree holds their records. The site judges them pooled (`map_judge.pooled_units`), the same units by the
same rules as a site that does them itself, and refuses what the pool does not hold."""

LATER = "DEW_CHECK1_LATER"
"""A judge's file to write where its judgement needs the units its pool does not hold yet, from the first
it does not on (`map_judge.ask_later`): the plan marks a check-1 site's second-stage map tasks later, which
armada runs only where the judge's first judgement asks for them, and the judge then judges again over all
its map tasks."""

STAGES = "DEW_CHECK1_STAGES"
"""The file check 1 appends each judgement it completes to, its stage among it (`reference_error._logged`),
which the plan reads to draw a site's later map tasks in its first wave."""


def staged_units(seeds: int) -> int:
    """The units check 1's two stages draw at a site of `seeds` declared seeds: its seeds and REDRAW times
    as many further ones. A weighed check's (`conditioning=`) are its own declared seeds' so, drawn past
    these."""
    return (1 + REDRAW) * seeds
