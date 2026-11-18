"""How tools/armada/ci.py shares a task's wall time among its files' rows: in proportion to their tests' own
times where the task ended, and by which file was running where it ran out of time."""

from collections.abc import Container, Mapping


def apportioned(own_seconds: Mapping[str, float], wall: float, rows: int) -> dict[str, float]:
    """Each file's share of a task's `wall` time, in proportion to its tests' own times (`task`), or an
    even split over the task's `rows` (a file named twice is two) where no test recorded a time."""
    total = sum(own_seconds.values())
    pace = wall / total if total > 0 else 0.0
    return {name: own * pace if pace else wall / rows for name, own in own_seconds.items()}


def deadline_shares(own_seconds: Mapping[str, float], reported: Container[str], wall: float,
                    running: str | None) -> dict[str, float]:
    """The seconds of the rows of a task that ran out of time, of the files that have any, by file: one that
    `reported` tests, its tests' own seconds, and the file `running` at the deadline, the one whose test
    started last (`followed.visited`), its own and all of the task's `wall` time no file's tests account for
    (its imports and its collection, and the test the interrupt stopped, which pytest's report gives no
    time), so its row holds at least what the file needs. A file that reported none and was not running never
    started, and has none: absent. `apportioned` split the wall time, the interrupted test's included, over
    the files that finished, in proportion to their tests' times, or evenly where none finished."""
    unaccounted = max(wall - sum(own_seconds.values()), 0.0)
    return {name: own + (unaccounted if name == running else 0.0) for name, own in own_seconds.items()
            if name == running or name in reported}


def task_shares(tests: list[str], cases: list, wall: float, running: str | None, expired: bool,
                grading) -> dict[str, float]:
    """Each of a task's `tests`' seconds of its `wall` time, by its JUnit `cases` (`grading.of`): in proportion to
    its tests' own times (`apportioned`); where the task ran out of time (`expired`), the row of the file `running`
    at its deadline (its executed log's last visit; a test node's row, which runs in a task of its own) at least
    what it needs, and one that never started none (`deadline_shares`)."""
    own_seconds = {name: sum(float(case.get("time") or 0) for case in cases if grading.of(case, name))
                   for name in tests}
    if not expired:
        return apportioned(own_seconds, wall, len(tests))
    reported = {name for name in tests if any(grading.of(case, name) for case in cases)}
    row = next((name for name in tests if grading.owning(name) == running), running)
    return deadline_shares(own_seconds, reported, wall, row)
