"""How a test file's row is graded in its task (tools/armada/ci.py's `tested`): from pytest's JUnit report and
collection, its exit, and the task's deadline, what makes the row red and what turned it red (`graded`,
`red_as`), each case by its node id (`node_of`) and its file (`file_of`). Loaded by ci.py by its path."""

import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping

STALE = "stale"
"""The kind of a row with no verdict of its tests: one its plan kept on a green record its task did not find
and did not run (ci.py's `stale_row`), one its task's deadline ran out before pytest reported any test of, and
one whose pytest never began (`graded`). Not passed (armada reads its exit 1 so), the queue runs it again as a
row of its own before it grades any red (land.sh), never a red of its tests and never a flaky row of its
owner's."""
INTERRUPTED = "the task ran past its deadline"
"""How a row says its task was interrupted: tests after the point it reached did not run."""


def node_of(case: ET.Element) -> str:
    """A JUnit case's pytest node id, as tests/test_durations.json keys it."""
    parts = (case.get("classname") or "").split(".")
    module = next((index for index, part in enumerate(parts) if part.startswith("test_")), len(parts) - 1)
    return "/".join(parts[:module + 1]) + ".py::" + "::".join([*parts[module + 1:], case.get("name") or ""])


def file_of(case: ET.Element) -> str:
    """The test file a JUnit case is from: its class path, or a collection error's own name."""
    parts = (case.get("classname") or case.get("name") or "").split(".")
    module = next((index for index, part in enumerate(parts) if part.startswith("test_")), len(parts) - 1)
    return "/".join(parts[:module + 1]) + ".py"


def owning(name: str) -> str:
    """The test file a row's name is of: the file itself, or a test node's (`file::test`) file."""
    return name.partition("::")[0]


def holds(name: str, node: str) -> bool:
    """Whether test node id `node` is of what a run named `name`: its file, or a test node in it
    (`file::test`, a class's `file::Class`), each of its parametrized cases among them."""
    if "::" not in name:
        return owning(node) == name
    return node == name or node.startswith((f"{name}[", f"{name}::"))


def of(case: ET.Element, name: str) -> bool:
    """Whether JUnit `case` is of what a run named `name` (`holds`): a case with no class path, a file that
    did not collect, is of every name its file holds (2026-11-17: a tier given node ids read every one of
    them as never run, its cases matched against their file's name)."""
    if "::" not in name or not case.get("classname"):
        return file_of(case) == owning(name)
    return holds(name, node_of(case))


def red_as(kind: str, nodes: list[str]) -> dict:
    """What turned a red row red, for grading it again (`land.sh`, `suspect.sh`): `kind` "nodes", its
    `failed` tests alone, by node id; "collect", its file's collection; "deadline", its task's; or
    "whole", its file otherwise (a test that did not collect, a missing import, pytest's own exit); or
    `STALE`, a kept row its task did not run (ci.py's `stale_row`)."""
    return {"kind": kind, "failed": nodes}


def exit_red(code: int, cases: list[ET.Element], own: list[ET.Element], deselected: bool,
             name: str, reached: str | None = None) -> list[str]:
    """Why pytest's exit `code` reddens file `name`'s row, whose own cases are `own` among the process's
    `cases`, and every one of whose tests the selection set aside where `deselected`. pytest exits 1 for
    failed tests and 5 for none collected; any other exit (a crash, a signal, an internal error) leaves even
    a written report untrustworthy, so every row it holds is red, as does an exit 1 no failed case explains
    (a late hook's os._exit): its tests ran in a process that failed. A file whose every test the selection
    set aside has no report however pytest exits, another file's failure among the task's included."""
    failed = any(case.find(kind) is not None for case in cases for kind in ("failure", "error"))
    if code == 1 and cases and not failed:
        return ["the process ended 1 after its tests passed"]
    if code not in (0, 1, 5) or (not own and not deselected and code == 1):
        ended = f"pytest was killed by signal {-code}" if code < 0 else f"pytest exited {code}"
        return [ended + ("" if own else f" with no report of {name}")
                + (f", the last file a test of began in {reached}" if reached and not own else "")]
    return []


def graded(name: str, own: list[ET.Element], cases: list[ET.Element], counted: Mapping[str, int],
           reported: bool, returncode: int, expired: float | None, running: str | None, started: bool = True,
           reached: str | None = None) -> tuple[list[str], list[tuple[ET.Element, str, ET.Element]], dict]:
    """Test file `name`'s grade in its task (ci.py's `tested`): what makes its row red, its failed cases, and
    what turned it red (`red_as`). `own` its cases of the task's JUnit `cases`, `counted` its collection,
    `reported` whether pytest wrote its report, `returncode` pytest's exit; `expired` the task's deadline
    where it ran out of time, `running` the file its time was spent in (its executed log's last visit).
    `started` whether pytest began (executed.py wrote its pid as pytest loaded it): a task whose pytest never
    began failed in its runner (its environment, its launch, an install before pytest), with no verdict of any
    test, so its row is said stale and run again (`STALE`), never a red of its tests, as is one whose task's
    time ran out in another file's test before every test its collection selected reported; one that began
    and ended with no report is a red of its file whole, the file it `reached` last and its exit, a signal's
    named, kept for whoever it routes to (Main, 2026-11-09: a write the runner could not make filed 43 GPU
    tier parts' files as reds)."""
    failed = [(case, kind, node) for case in own for kind in ("failure", "error")
              if (node := case.find(kind)) is not None]
    red = [f"{kind.upper()} {case.get('classname')}::{case.get('name')}: {node.get('message')}"
           for case, kind, node in failed]
    skipped = [f"SKIPPED on a missing import {case.get('classname')}::{case.get('name')}: "
               f"{node.get('message')}" for case in own for node in case.iter("skipped")
               if "could not import" in f"{node.get('message')} {node.text}"]
    deselected = bool(counted) and counted["collected"] == counted["deselected"]
    # The executed log's visits name files: a test node's row is spent where its file's test was running.
    spent = expired is not None and (owning(name) == running or running is None)
    unreached = expired is not None and not spent and not own and not deselected
    # A file whose tests interleave with another's (A[0], B[0], A[1]) has cases however the deadline fell;
    # where it fell in another file's test, its own count of reported tests must reach the tests its
    # collection selected (the selection's set-aside ones among the deselected), or it ran in part
    # (row 12179).
    reported_tests = len({node_of(case) for case in own if case.get("classname")})
    selected = counted["collected"] - counted["deselected"] if counted else None
    # A file whose cases say it is red whole already (a failed case, a skip on a missing import) keeps it.
    partial = (expired is not None and not spent and bool(own) and not failed and not skipped
               and (selected is None or reported_tests < selected))
    unstarted = expired is None and not started and not reported and returncode != 0
    exited = [] if (expired is not None and not spent) or unstarted else \
        exit_red(returncode, cases, own, deselected, name, reached)
    unaccounted = ([] if exited or unreached or unstarted or (reported and (own or deselected))
                   else [f"no evidence {name} ran: " + ("pytest wrote no report" if not reported
                                                        else "pytest reported none of its tests, and its "
                                                             "collection shows them not all deselected")])
    red += skipped + exited + unaccounted
    if spent:
        red.append(f"{INTERRUPTED} of {expired:.0f} s and was interrupted")
    if unreached:
        red.append(f"not graded: its task's {expired:.0f} s were spent in {running}, before pytest reported "
                   "any test of it; it runs again")
    if partial:
        red.append(f"not graded: its task's {expired:.0f} s were spent in {running}, after pytest reported "
                   f"{reported_tests} of its {selected if selected is not None else 'uncounted'} selected "
                   "tests; it runs again")
    if unstarted:
        red.append(f"not graded: pytest never began (its runner exited {returncode} before pytest loaded its "
                   "plugins: the environment, the launch or an install before it); it runs again")
    # A failed case with no class path is a file that did not collect, no test of it.
    nodes = sorted({node_of(case) for case, _, _ in failed if case.get("classname")})
    whole = skipped or exited or unaccounted or not nodes or any(not case.get("classname")
                                                                  for case, _, _ in failed)
    cause = ({} if not red else red_as(STALE, []) if unreached or unstarted or partial
             else red_as("deadline", []) if spent
             else red_as("whole", []) if whole else red_as("nodes", nodes))
    return red, failed, cause


def flow_cells(cases) -> dict[str, dict]:
    """A GPU flow's counts by its cases (each case's JUnit property `cells`, tests/conftest.py from
    tests/gpu_flow.py: passed, failed, skipped and the device), as every row a task runs carries them, an
    empty object where its file runs no flow: the train's native gate reads a flow none of whose cells ran as
    unqualified, and a reused row keeps its counts (ledger 2643)."""
    return {node_of(case): json.loads(item.get("value") or "null") for case in cases
            for item in case.iter("property") if item.get("name") == "cells"}


def ynn_switched(cases) -> list[dict]:
    """The accuracy gate's sites a file's cases moved onto YNNPACK's fusions (each case's JUnit property
    `ynn_switched`, tests/conftest.py), green cases included: each its test node, label and XLA:CPU's
    refusal, as every row a task runs carries them for the train report to count, an empty list where
    nothing moved, so a row without the field is one recorded before it, uncounted."""
    return [{"node": node_of(case), **site} for case in cases
            for item in case.iter("property") if item.get("name") == "ynn_switched"
            for site in json.loads(item.get("value") or "[]")]


def compiled_cold(cases: list) -> list[str]:
    """Of a file's JUnit `cases`, the node ids of those that compiled a program the shared compilation
    cache did not hold (each case's property `compiled`, tests/conftest.py's count from
    tests/remote_cache.py's watch, set only where a shared cache was installed)."""
    return sorted(node_of(case) for case in cases
                  if any(item.get("name") == "compiled" and int(item.get("value") or 0) > 0
                         for item in case.iter("property")))
