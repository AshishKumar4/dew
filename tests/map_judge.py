"""A test's work sharded across armada map tasks and judged by one task that pools them (tools/armada/ci.py's
`check1_shards` and `shards` marks): each map task does units `start` to `stop` of a site's work and records
what it found, judging nothing; the judge pools every map task's records and decides, as the site would over
all its units in one process. Check 1 (tests/reference_error.py's `assert_as_exact`) records its draws so, a
witness of run options its outcomes; each reads and writes through `record_units` and `pooled_units` alone.
"""

import hashlib
import json
import os
import pickle
import sys
import time
from collections.abc import Callable
from pathlib import Path

import psutil
from check1_wire import DRAWS, LATER, POOL, RECORDS


class PoolRefused(AssertionError):
    """A judge's pool (POOL) short of what the site asks (no record of it, a gap, too few units, or map
    tasks that disagree on what they ran), never judged; `held`, the units it holds from 0 on."""

    def __init__(self, message: str, held: int = 0):
        super().__init__(message)
        self.held = held


def drawing() -> bool:
    """Whether this process is a map task (DRAWS): it records a site's units and judges nothing, so a site
    that draws alone (`reference_error`'s deferred checks) leaves its judgement to the judge; a site whose
    tasks run checks of their own beside their draws records each task's word (`record_units`,
    `test_architectures.assert_fits_as_drawn`), which the judge pools (`sharded`)."""
    return bool(os.environ.get(DRAWS))


def judging() -> bool:
    """Whether this process is a sharded site's judge (POOL): its draws are its map tasks' records, so it
    draws none of its own and sizes nothing they carry already."""
    return bool(os.environ.get(POOL))


def site_id(label: str) -> str:
    """A site's identity across its map tasks and its judge: the Python it runs on, its test's id (pytest's,
    the same test in each task) and its label, so neither two tests' sites of one label nor one site's runs
    on two Pythons pool each other's records, whatever directory holds them (a run plans a site on each
    Python, their judges one job)."""
    return f"{python()}::{os.environ.get('PYTEST_CURRENT_TEST', '').rsplit(' (', 1)[0]}::{label}"


def python() -> str:
    return ".".join(map(str, sys.version_info[:2]))


def _file(site: str) -> str:
    return hashlib.sha256(site.encode()).hexdigest()[:16]


def record_units(label: str, payload: Callable[[int, int], object], upto: int | None = None,
                 phases: Callable[[], dict] | None = None, offset: int = 0) -> bool:
    """Where DRAWS names a range `start:stop`, `payload(start, stop)` written to RECORDS as the site's record
    of those units (pickled whole, then moved into place), and True; else False. With `upto`, the units
    the site takes at most (a check's own, reference_error's `conditioning`): the range is cut there, and a
    task whose range starts past it records nothing. With `offset`, where its units start in the plan's (a
    weighed check's past its unweighted one's, tools/armada/ci.py `draws_of`): DRAWS is read that far on, a
    task before them recording nothing. Beside
    it, the seconds from the task's process's start to its units (`before`: its imports, collection, the
    site's setup and compiles) and those its units took (`draws`), so a run's map tasks say what each paid
    before any draw (Main, 2026-11-02: the map task's fixed cost measured before it is cut), and the
    seconds of each named phase within its units' (`phases`, read once they are done)."""
    span = os.environ.get(DRAWS)
    if not span:
        return False
    # The site's units sit at `offset` in the plan's (a weighed check's past its unweighted one's).
    start, stop = (max(int(part) - offset, 0) for part in span.split(":"))
    stop = stop if upto is None else min(stop, upto)
    if start >= stop:
        return True
    records = os.environ.get(RECORDS)
    if not records:
        raise ValueError(f"{label}: a map task ({DRAWS}={span}) writes its records to {RECORDS}, which is "
                         f"unset")
    site = site_id(label)
    # The process's start is on the wall clock; the draws' seconds on one that only moves forward.
    began, drawing = time.time(), time.perf_counter()
    found = payload(start, stop)
    seconds = {"site": site, "start": start, "stop": stop, "draws": time.perf_counter() - drawing,
               "before": began - psutil.Process().create_time(), **(phases() if phases else {})}
    path = Path(records) / f"{_file(site)}.{start}-{stop}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    _published(path.with_suffix(".seconds.json"), json.dumps(seconds).encode())
    _published(path, pickle.dumps({"site": site, "start": start, "stop": stop, "payload": found}))
    return True


def sharded(label: str, payload: Callable[[int, int], object], units: int | None = None) -> list | None:
    """A site's units over armada's map tasks, each `payload(start, stop)`: a map task records its units'
    (`record_units`) and gets None, the judge gets the tasks' payloads in order (`pooled_units`, refused
    short of `units`), and a run whole gets `[payload(0, units)]`."""
    if record_units(label, payload):
        return None
    pooled = pooled_units(label, units)
    if pooled is not None:
        return [found for _, _, found in pooled]
    if units is None:
        raise ValueError(f"{label}: a run whole of a sharded site takes its units' count")
    return [payload(0, units)]


def _published(path: Path, data: bytes) -> None:
    """`data` at `path` whole: written beside it, then moved into place, so no reader sees part of it."""
    written = path.with_name(path.name + ".partial")
    written.write_bytes(data)
    written.rename(path)


def pooled_units(label: str, units: int | None = None) -> list[tuple[int, int, object]] | None:
    """Where POOL names a directory, the site's records under it (`record_units`) as (start, stop, payload),
    in order, a range recorded twice read once: refused (PoolRefused) unless they hold every unit from 0
    on, once, without a gap, to `units` where it is given. None where POOL is unset."""
    root = os.environ.get(POOL)
    if not root:
        return None
    site, found = site_id(label), {}
    for path in sorted(Path(root).rglob(f"{_file(site)}.*.pkl")):
        record = pickle.loads(path.read_bytes())
        if record["site"] == site:
            found.setdefault((record["start"], record["stop"]), record["payload"])
    held, pieces = 0, []
    for (start, stop), payload in sorted(found.items()):
        if start != held:
            raise PoolRefused(f"{label}: the map tasks' records under {root} hold units 0 to {held}, and "
                              f"then {start} to {stop}; a pool holds every unit from 0 on, once", held)
        pieces.append((start, stop, payload))
        held = stop
    if not pieces:
        raise PoolRefused(f"{label}: no map task's records under {root}")
    if units is not None and held < units:
        raise PoolRefused(f"{label}: the map tasks' records under {root} hold units 0 to {held} of {units}",
                          held)
    return pieces


class LaterUnits(AssertionError):
    """A judge's judgement needs units its pool does not hold yet, which it asked for (`ask_later`): its row
    reads red until it judges again over them."""


def ask_later(label: str, held: int, stop: int) -> None:
    """Where LATER names a file, raise `LaterUnits` for site `label`, whose judgement needs units `held` to
    `stop` its pool does not hold, the file written first; else nothing (the pool is short, `PoolRefused`)."""
    asked = os.environ.get(LATER)
    if asked:
        Path(asked).parent.mkdir(parents=True, exist_ok=True)
        Path(asked).write_text(json.dumps({"site": site_id(label), "from": held, "to": stop}))
        raise LaterUnits(f"{label}: its judgement needs units {held} to {stop}, which its later map tasks "
                         "draw")
