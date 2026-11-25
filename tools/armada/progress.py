"""A pytest plugin that keeps a task's progress where armada keeps a recalled attempt's artifacts:
each test that passed whole (its call and its teardown), a line of PROGRESS under ARMADA_ARTIFACTS as it
ends, so the task's next attempt, recalled while it ran, sets them aside and goes on from there
(tools/armada/ci.py `resumed`, armada's ARMADA_RESUME). Loaded by a CI task's pytest beside executed.py;
writes nothing outside a task."""

import json
import os
import types
from collections.abc import Callable
from pathlib import Path

PROGRESS = "progress.jsonl"
"""The file of the tests a task passed, a JSON line each: its node id, its seconds (its setup's, call's and
teardown's, as JUnit times a test) and, where it compiled a program the shared cache lacked over that time
(tests/conftest.py's property `compiled`), `cold`."""
SET_ASIDE = "DEW_SET_ASIDE"
"""What names the JSON list of the node ids a resumed attempt sets aside (ci.py's `tested`): exactly those,
never a test whose id one of them begins, as pytest's --deselect, which matches by prefix, would (train
671's review: test_host_placement's `..._trains_the_same_model` would have set aside its
`..._over_a_scaled_window`)."""


def pytest_collection_finish(session) -> None:
    """The items whose node ids SET_ASIDE names taken out of the session, said deselected as pytest says its
    own, which executed.py counts: once collection is finished, so after every selection, split.py's cut of a
    group among them, which cuts its file's every test as each group's container does (train 674's review: set
    aside before the cut, a resumed group cut the rest anew, running another group's tests and none of some).
    This module imports no pytest: ci.py, which a planner with none runs, reads PROGRESS and SET_ASIDE from
    it."""
    named = os.environ.get(SET_ASIDE)
    aside = set(json.loads(Path(named).read_text())) if named else set()
    if dropped := [item for item in session.items if item.nodeid in aside]:
        session.items[:] = [item for item in session.items if item.nodeid not in aside]
        session.config.hook.pytest_deselected(items=dropped)


_set_up: dict[str, float] = {}
_called: dict[str, float] = {}
_session: dict[str, str | None] = {}
"""The task's own artifacts (ARMADA_ARTIFACTS), named once its session starts (`pytest_configure`)."""


def pytest_configure(config) -> None:
    """The task's artifacts as its session starts: a test that changes the environment as it runs (one
    that runs CI's tools apart from the task, tests/armada_support.py's `loaded`) moves none of the task's
    progress."""
    _session["artifacts"] = os.environ.get("ARMADA_ARTIFACTS")


def pytest_runtest_logreport(report) -> None:
    """A test's call passed is kept once its teardown passed too; one whose teardown failed is no progress."""
    artifacts = _session.get("artifacts")
    if not artifacts:
        return
    if report.when == "setup" and report.passed:
        _set_up[report.nodeid] = report.duration
    elif report.when == "call" and report.passed:
        _called[report.nodeid] = _set_up.pop(report.nodeid, 0.0) + report.duration
    elif report.when == "teardown" and report.nodeid in _called:
        seconds = _called.pop(report.nodeid) + report.duration
        if report.passed:
            cold = any(name == "compiled" and int(value) > 0 for name, value in report.user_properties)
            Path(artifacts).mkdir(parents=True, exist_ok=True)
            with (Path(artifacts) / PROGRESS).open("a") as progress:
                # The attempt's peak by then, which bounds this test (`task_peak`).
                import host

                peak = task_peak(host)
                progress.write(json.dumps({"node": report.nodeid, "seconds": seconds, "pid": str(os.getpid()),
                                           **({"cold": True} if cold else {}),
                                           **({"peak": peak} if peak is not None else {})}) + "\n")


def task_peak(host: types.ModuleType) -> int | None:
    """This task's resident peak in bytes so far: its cgroup's (cgroup v2's memory.peak, which every process
    the task started counts in, a test's ranks with it), each of its tests' bound above (row 6003: a test's
    four ranks peaked at 7.17 GiB, packed beside another on a medium), read by Dew's one cgroup reader
    through `host` (tools/armada/host.py: src/dew/_cgroups.py, its root DEW_CGROUP_ROOT's where set),
    which a task's pytest imports (tools/armada on its path) and ci.py loads. None where the task's cgroup
    states no peak."""
    reader = host.cgroups()
    return reader.memory_peak(reader.own(host.cgroup_root()))



SEALED = "sealed"
"""A PROGRESS line's key marking a test an attempt set aside as an earlier one passed it (`resumed`)."""
ENDED = "ended"
"""A PROGRESS line's key naming a pytest process that exited, beside its exit code (`ended`)."""


def ended(pid: str | None, code: int) -> None:
    """Keep, among this task's progress, that the pytest of process `pid` exited with `code`: written by ci.py
    as the process exits, before anything else of it is read, so an attempt resumed after sets aside the
    tests a process passed only where it exited 0, or never exited (the fleet recalled the task while it
    ran); a process that failed after its tests passed (an exit's crash) keeps none (`resumed`). Nothing
    outside a task, nor for a process whose id was never written."""
    artifacts = os.environ.get("ARMADA_ARTIFACTS")
    if not artifacts or not pid:
        return
    Path(artifacts).mkdir(parents=True, exist_ok=True)
    with (Path(artifacts) / PROGRESS).open("a") as progress:
        progress.write(json.dumps({ENDED: pid, "code": code}) + "\n")


def resumed(tests: list[str], owning: Callable[[str], str],
            sealed: bool) -> tuple[dict[str, float], set[str], int | None]:
    """The tests of `tests` (each node's file as `owning` reads it) an earlier attempt of this task passed
    before the fleet recalled it (armada keeps a recalled attempt's artifacts and hands them to the next as
    ARMADA_RESUME, #4544), each with its seconds, as this plugin kept them, and those of them that ran
    cold (`cold`), and the most memory those attempts had held by then (`peak`), which bounds them, as
    this attempt's own cgroup did not run them; copied into this attempt's progress, so a second recall
    keeps them too. With `sealed` (a whole file's process), only those of a process that exited 0 (`ended`),
    as one recalled before its end may yet have failed after them (its exit's crash): its file runs whole
    again; else (a split group's) those of a process that did not exit other than 0. Read by ci.py, which
    runs a task's pytest. None outside a resumed task."""
    kept = Path(os.environ.get("ARMADA_RESUME", "/nonexistent")) / PROGRESS
    if not kept.is_file():
        return {}, set(), None
    lines = [json.loads(line) for line in kept.read_text().splitlines()]
    ends = {found[ENDED]: found.get("code") for found in lines if ENDED in found}
    passed, cold, peaks = {}, set(), []
    for found in lines:
        pid = found.get("pid")
        # A line carried from an earlier resumed attempt was set aside there by these rules: kept as it was.
        kept_by = found.get(SEALED) or ends.get(pid) == 0 or (not sealed and ends.get(pid) is None)
        if "node" in found and owning(found["node"]) in tests and kept_by:
            passed[found["node"]] = float(found["seconds"])
            if found.get("cold"):
                cold.add(found["node"])
            if found.get("peak") is not None:
                peaks.append(int(found["peak"]))
    peak = max(peaks, default=None)
    artifacts = os.environ.get("ARMADA_ARTIFACTS")
    if artifacts and passed:
        Path(artifacts).mkdir(parents=True, exist_ok=True)
        lines = ({"node": node, "seconds": seconds, SEALED: True, **({"cold": True} if node in cold else {}),
                  **({"peak": peak} if peak is not None else {})}
                 for node, seconds in passed.items())
        with (Path(artifacts) / PROGRESS).open("a") as carried:
            carried.writelines(json.dumps(line) + "\n" for line in lines)
    if passed:
        print(f"resumed: {len(passed)} tests an earlier attempt passed set aside", flush=True)
    return passed, cold, peak
