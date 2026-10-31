"""Dew's CI's durations record on armada (tools/armada/ci.py): how the plan cuts a test file from it into
groups of the target's weight, and how main's CI is folded into it, on small test files written here."""

import json
import sys
from pathlib import Path

import pytest
from armada_support import commit, interpreter, loaded


@pytest.fixture
def ci(tmp_path, monkeypatch):
    return loaded(tmp_path, monkeypatch)


def test_a_file_whose_recorded_tests_outweigh_its_median_runs_split_by_them(ci, tmp_path):
    """Train 18's census: main's runs timed the file at 6 s with one test; a head recorded the 1,576
    tests it added (here 40 of 30 s). Each record is a test's least time, so their sum past the median
    may mean those runs predate the file's tests, and is a floor on its time: it runs split by its
    records, alone, not packed with the light files by the median it had before."""
    for name in ("test_census", "test_light", "test_other"):
        (tmp_path / "tests" / f"{name}.py").write_text("")
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {**{f"tests/test_census.py::test_cell[{n}]": 30.0 for n in range(40)},
         "tests/test_light.py::test_a": 20.0, "tests/test_other.py::test_b": 20.0}))
    timings = {"files": {"3.12:tests/test_census.py": 6.0, "3.12:tests/test_light.py": 20.0,
                         "3.12:tests/test_other.py": 20.0},
               "rows": {"3.12:tests/test_census.py": 6.0}}
    floor = sorted((entry["tests"], entry["split"], entry["weight"]) for entry in ci.plan(240.0, timings)
                   if entry["python"] == "3.12")
    assert floor == [*((("tests/test_census.py", f"{group}/5", 240.0)) for group in range(1, 6)),
                     ("tests/test_light.py tests/test_other.py", "", 40.0)]


def test_a_split_file_pays_its_setup_once_a_group_and_is_not_cut_finer_for_it(ci):
    """Twenty recorded tests of 10 s that armada measured whole at 400 s paid 200 s besides their
    tests, which every group pays again: at a 150 s target that leaves a group half the target of
    tests, so 3 groups of 270, 270 and 260 s, not slivers each paying it. Measured at 2 groups,
    300 s, each paid 50 s: its setup is not counted again for the groups it ran as, and it is
    measured by its rows at that cut, not by a median that remembers other cuts."""
    tests = [10.0] * 20
    assert ci.split(tests, 400.0, 150.0, cut=1) == [270.0, 270.0, 260.0]
    assert ci.split(tests, 200.0 + 2 * 50.0, 150.0, cut=2) == [150.0, 150.0]
    # A cut whose heaviest group, setup included, holds within half again the target is kept, so
    # its cached programs are: 4 groups of 100 s, and 20 groups of 50 s (an over-cut paying 40 s
    # each), stay as they ran; 2 groups at 130 s of setup each (230 s, past 225) are cut again.
    assert ci.split(tests, 400.0, 150.0, cut=4) == [100.0] * 4
    assert ci.split(tests, 200.0 + 20 * 40.0, 150.0, cut=20) == [50.0] * 20
    assert ci.split(tests, 200.0 + 2 * 130.0, 150.0, cut=2) == [200.0, 200.0, 190.0]
    # The file's median remembers runs at other cuts; its rows are timed at the cut they name. Here
    # four runs at 4 groups (400 s) and the last at 2 (300 s, its rows): measured at 2, it stays at 2.
    rows = {"3.12:tests/test_split.py#1/2": 150.0, "3.12:tests/test_split.py#2/2": 150.0,
            "3.12:tests/test_split.py#3/4": 100.0, "3.14:tests/test_split.py#1/4": 10.0,
            "3.12:tests/test_split.py#collect": 1.0, "3.12:tests/test_whole.py": 30.0}
    timings = {"rows": rows, "files": {"3.12:tests/test_split.py": 400.0}}
    assert ci.measured_cut(timings, "3.12", "tests/test_split.py") == (300.0, 2)
    seconds, count = ci.measured_cut(timings, "3.12", "tests/test_split.py")
    assert ci.split(tests, seconds, 150.0, cut=count) == [150.0, 150.0]
    assert ci.measured_cut(timings, "3.12", "tests/test_whole.py") == (30.0, 1)
    assert ci.measured_cut(timings, "3.12", "tests/test_unseen.py") is None


def test_a_split_files_groups_run_every_test_once(ci, tmp_path):
    """Each task of a split file runs its own group (split.py) of the same
    cut, so the groups together run every test, none twice, a test the
    durations lack included. Each group times the file at what it spent,
    which armada sums over the groups."""
    (tmp_path / "tests/test_split.py").write_text(
        "import pytest\n@pytest.mark.parametrize('n', range(6))\ndef test_n(n): pass\n"
        "def test_unrecorded(): pass\n")
    Path("tests/test_durations.json").write_text(json.dumps(
        {f"tests/test_split.py::test_n[{n}]": seconds for n, seconds in enumerate([5, 1, 1, 1, 5, 1])}))
    interpreter(tmp_path, "3.12")
    ran = []
    for group in ("1/3", "2/3", "3/3"):
        out = tmp_path / f"verdict-{group.replace('/', '-')}.json"
        ci.task("3.12", ["tests/test_split.py"], group, out, 120.0)
        [verdict] = json.loads(out.read_text())["rows"]
        assert verdict["exitCode"] == 0, verdict["output"]
        assert verdict["timings"] == {"3.12:tests/test_split.py": verdict["seconds"]}
        ran.append(sorted(verdict["tests"]))
    assert all(ran)
    every = sorted(node for nodes in ran for node in nodes)
    assert every == sorted([f"tests/test_split.py::test_n[{n}]" for n in range(6)]
                           + ["tests/test_split.py::test_unrecorded"])


def test_the_durations_fold_rewrites_only_a_file_its_record_no_longer_cuts(ci, tmp_path, monkeypatch):
    """`durations` folds main's CI verdict into the record a file at a time, each whole, from its floor rows:
    a file whose cut no longer fits as its tests ran takes what they took, though no test's node id changed
    and its total barely moved (one test grown from 10 s to 180 s of 100; ten of 20 grown from 30 s to 39 s,
    each within half again), so the next plan cuts it to fit; a file whose cut fits keeps its entries byte for
    byte, as does one whose heaviest group is a test past the budget alone, which no cut shortens, though
    its time moves; a file with a red group keeps its own; a file the checkout no longer has loses them; the
    newest Python's times move nothing."""
    (tmp_path / ".armada.json").write_text(json.dumps({"target": 240}))
    record, rows = {}, []

    def file(name: str, recorded: list[float], ran: list[list[float]], red: int | None = None) -> None:
        (tmp_path / f"tests/{name}.py").write_text("")
        nodes = [f"tests/{name}.py::test_{index:03}" for index in range(len(recorded))]
        record.update(zip(nodes, recorded, strict=True))
        start = 0
        for group, times in enumerate(ran, start=1):
            split = f"#{group}/{len(ran)}" if len(ran) > 1 else ""
            tests = dict(zip(nodes[start:start + len(times)], times, strict=True))
            start += len(times)
            rows.append({"name": f"3.12:tests/{name}.py{split}", "exitCode": int(group == red),
                         "seconds": sum(times), "tests": tests})
            rows.append({"name": f"3.14:tests/{name}.py{split}", "exitCode": 0, "seconds": 999.0,
                         "tests": dict.fromkeys(tests, 999.0)})

    file("test_grown", [10.0] * 100, [[180.0] + [10.0] * 19] + [[10.0] * 20] * 4)
    file("test_drift", [30.0] * 20, [[39.0] * 10, [30.0] * 10])
    file("test_steady", [100.0] * 4, [[101.0, 99.0], [104.5, 98.0]])
    file("test_straggler", [381.593, 20.0, 20.0], [[382.0], [20.5, 19.0]])
    file("test_partial", [150.0] * 4, [[400.0, 400.0], [150.0, 150.0]], red=2)
    file("test_empty", [], [[]])  # every test deselected, its collection past the target
    rows[-2]["seconds"] = 300.0
    record["tests/test_gone.py::test_x"] = 5.0
    rows.append({"name": "3.12:tests/test_gone.py", "exitCode": 0, "seconds": 5.0,
                 "tests": {"tests/test_gone.py::test_x": 5.0, "tests/test_gone.py::test_y": 1.0}})
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(record, indent=0) + "\n")
    verdict = tmp_path / "main.json"
    verdict.write_text(json.dumps({"sha": commit(tmp_path, "main"), "part": "all", "rows": rows}))
    monkeypatch.setattr(sys, "argv", ["ci.py", "durations", str(verdict)])
    assert ci.main() == 0
    folded = json.loads((tmp_path / "tests/test_durations.json").read_text())
    observed = {node: seconds for row in rows if row["name"].startswith("3.12:") and not row["exitCode"]
                for node, seconds in row["tests"].items()}
    for name in ("test_grown", "test_drift"):
        assert {node: s for node, s in folded.items() if f"/{name}.py" in node} == {
            node: s for node, s in observed.items() if f"/{name}.py" in node}
    for name in ("test_steady", "test_straggler", "test_partial"):
        assert {node: s for node, s in folded.items() if f"/{name}.py" in node} == {
            node: s for node, s in record.items() if f"/{name}.py" in node}
    assert not [node for node in folded if node.startswith("tests/test_gone.py")]


def test_the_durations_fold_takes_main_s_times_only_for_files_as_main_ran_them(ci, tmp_path, monkeypatch):
    """main's CI ran main's test files: a file the checkout has changed since (a head that renamed its tests,
    composed in a train) keeps its own entries, which name the tests it holds, and a file as main ran it takes
    the times main's rows observed (2026-10-31: train 370's fold replaced a head's 66 renamed corpus test ids
    with main's old ones, which no test collects, and the plan cut the file by count)."""
    renamed, unchanged = "tests/test_renamed.py", "tests/test_unchanged.py"

    def defining(path: str, *names: str) -> None:
        (tmp_path / path).write_text("".join(f"def {name}():\n    pass\n" for name in names))

    (tmp_path / ".armada.json").write_text(json.dumps({"target": 240}))
    defining(renamed, "test_old_a", "test_old_b")
    defining(unchanged, "test_a", "test_b")
    # main's record, which predates test_unchanged's second test.
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {f"{renamed}::test_old_a": 100.0, f"{renamed}::test_old_b": 100.0, f"{unchanged}::test_a": 10.0},
        indent=0) + "\n")
    main = commit(tmp_path, "main")
    rows = [{"name": f"3.12:{renamed}", "exitCode": 0, "seconds": 200.0,
             "tests": {f"{renamed}::test_old_a": 100.0, f"{renamed}::test_old_b": 100.0}},
            {"name": f"3.12:{unchanged}", "exitCode": 0, "seconds": 30.0,
             "tests": {f"{unchanged}::test_a": 10.0, f"{unchanged}::test_b": 20.0}}]
    # A head renames test_renamed's tests, and the queue's record of its composition names them.
    defining(renamed, "test_new_a", "test_new_b")
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {f"{renamed}::test_new_a": 60.0, f"{renamed}::test_new_b": 140.0, f"{unchanged}::test_a": 10.0},
        indent=0) + "\n")
    commit(tmp_path, "a train composed on main")
    verdict = tmp_path / "main.json"
    verdict.write_text(json.dumps({"sha": main, "part": "all", "rows": rows}))
    monkeypatch.setattr(sys, "argv", ["ci.py", "durations", str(verdict)])
    assert ci.main() == 0
    assert json.loads((tmp_path / "tests/test_durations.json").read_text()) == {
        f"{renamed}::test_new_a": 60.0, f"{renamed}::test_new_b": 140.0,
        f"{unchanged}::test_a": 10.0, f"{unchanged}::test_b": 20.0}
