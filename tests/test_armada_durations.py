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
        (tmp_path / "tests" / f"{name}.py").write_text(
            "def test_cell(): pass\ndef test_a(): pass\ndef test_b(): pass\n")
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
    assert ci.cuts.split(ci.session, tests, 400.0, 150.0, cut=1) == [270.0, 270.0, 260.0]
    assert ci.cuts.split(ci.session, tests, 200.0 + 2 * 50.0, 150.0, cut=2) == [150.0, 150.0]
    # A cut whose heaviest group, setup included, holds within half again the target is kept, so
    # its cached programs are: 4 groups of 100 s, and 20 groups of 50 s (an over-cut paying 40 s
    # each), stay as they ran; 2 groups at 130 s of setup each (230 s, past 225) are cut again.
    assert ci.cuts.split(ci.session, tests, 400.0, 150.0, cut=4) == [100.0] * 4
    assert ci.cuts.split(ci.session, tests, 200.0 + 20 * 40.0, 150.0, cut=20) == [50.0] * 20
    assert ci.cuts.split(ci.session, tests, 200.0 + 2 * 130.0, 150.0, cut=2) == [200.0, 200.0, 190.0]
    # The file's median remembers runs at other cuts; its rows are timed at the cut they name. Here
    # four runs at 4 groups (400 s) and the last at 2 (300 s, its rows): measured at 2, it stays at 2.
    rows = {"3.12:tests/test_split.py#1/2": 150.0, "3.12:tests/test_split.py#2/2": 150.0,
            "3.12:tests/test_split.py#3/4": 100.0, "3.14:tests/test_split.py#1/4": 10.0,
            "3.12:tests/test_split.py#collect": 1.0, "3.12:tests/test_whole.py": 30.0}
    timings = {"rows": rows, "files": {"3.12:tests/test_split.py": 400.0}}
    assert ci.measured_cut(timings, "3.12", "tests/test_split.py") == (300.0, 2)
    seconds, count = ci.measured_cut(timings, "3.12", "tests/test_split.py")
    assert ci.cuts.split(ci.session, tests, seconds, 150.0, cut=count) == [150.0, 150.0]
    assert ci.measured_cut(timings, "3.12", "tests/test_whole.py") == (30.0, 1)
    assert ci.measured_cut(timings, "3.12", "tests/test_unseen.py") is None


def test_a_file_no_run_measured_at_a_cut_weighs_its_estimate_and_no_setup_paid_again_by_every_group(
        ci, tmp_path):
    """A file whose rows armada holds at no cut is estimated (its record times how much slower armada ran the
    files measured both ways): the plan spreads that estimate over its groups, their weights its sum, and
    takes none of it for a setup every group pays again (2026-11-12: one file's timing was left, 1.6 times
    its record; the plan charged the 0.6 of each other file's record to every group it cut, 1,123 task-hours
    for a suite of 100, armada sized the run by 6,500 s tasks that took a minute, and train 772's CI ran out
    its deadline on 20 of 688 containers)."""
    for name in ("test_measured", "test_unmeasured"):
        (tmp_path / "tests" / f"{name}.py").write_text("def test_cell(): pass\n")
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {**{f"tests/test_measured.py::test_cell[{n}]": 10.0 for n in range(10)},
         **{f"tests/test_unmeasured.py::test_cell[{n}]": 10.0 for n in range(100)}}))
    # The one file measured took 1.6 times its record, as one whole row.
    timings = {"files": {"3.12:tests/test_measured.py": 160.0},
               "rows": {"3.12:tests/test_measured.py": 160.0}}
    groups = [entry["weight"] for entry in ci.plan(100.0, timings)
              if entry["python"] == "3.12" and entry["tests"] == "tests/test_unmeasured.py"]
    # A hundred tests of 16 s each as estimated, six to a group within the target.
    assert groups == [96.0] * 16 + [64.0]


def test_a_file_armada_holds_only_rows_of_weighs_what_its_rows_took(ci, tmp_path):
    """A file armada holds no file timing of, but whose rows it holds at a cut, weighs those rows' sum, as
    its file timing would say, not its record scaled by the other files' ratio: armada keeps a plan's rows
    from run to run and drops the file timings a run did not itself measure (ledger 5579: after one run that
    reused most rows it held 1 file timing of 1,067, and 688 of 689 files were weighed by a guess)."""
    for name in ("test_measured", "test_rows", "test_neither"):
        (tmp_path / "tests" / f"{name}.py").write_text("def test_cell(): pass\n")
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {**{f"tests/test_measured.py::test_cell[{n}]": 10.0 for n in range(10)},
         **{f"tests/test_rows.py::test_cell[{n}]": 10.0 for n in range(20)},
         **{f"tests/test_neither.py::test_cell[{n}]": 10.0 for n in range(5)}}))
    timings = {"files": {"3.12:tests/test_measured.py": 160.0},
               "rows": {"3.12:tests/test_measured.py": 160.0, "3.12:tests/test_rows.py#1/2": 120.0,
                        "3.12:tests/test_rows.py#2/2": 130.0, "3.12:tests/test_rows.py#collect": 1.0}}
    names = ["tests/test_measured.py", "tests/test_rows.py", "tests/test_neither.py"]
    # The file neither times is its record scaled by how much slower armada ran the two measured (1.6, 1.25).
    assert ci.weights(names, timings, "3.12") == {
        "tests/test_measured.py": 160.0, "tests/test_rows.py": 250.0, "tests/test_neither.py": 80.0}


def test_a_recorded_test_the_file_no_longer_defines_cuts_it_into_no_group(ci, tmp_path):
    """A file whose record still names a function a lane renamed is planned by the tests it defines: the
    stale entries weigh nothing and cut no group, so no task of it runs nothing (2026-11-12:
    tests/test_recurrence.py's 37 tests ran as 133 tasks, 113 of them empty, by a record of 262 with 244 of a
    renamed function's). A class's recorded tests stay while the class does, and one the record lacks still
    spreads over the groups there are."""
    (tmp_path / "tests/test_cut.py").write_text(
        "import pytest\n\n@pytest.mark.parametrize('n', range(4))\ndef test_kept(n): pass\n\n"
        "class TestHeld(Base): pass\n\ntest_made = make()\n\ndef test_new(): pass\n")
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {**{f"tests/test_cut.py::test_renamed[{n}]": 100.0 for n in range(40)},
         **{f"tests/test_cut.py::test_kept[{n}]": 100.0 for n in range(4)},
         "tests/test_cut.py::TestHeld::test_inherited": 100.0, "tests/test_cut.py::test_made": 100.0}))
    planned = [entry for entry in ci.plan(100.0, {}) if entry["python"] == "3.12"
               and entry["tests"] == "tests/test_cut.py"]
    assert [entry["weight"] for entry in planned] == [100.0] * 6
    count = len(planned)
    members = [ci.run_keys.members("tests/test_cut.py", f"{group}/{count}") for group in range(1, count + 1)]
    assert sorted(node.partition("::")[2] for group in members for node in group) == [
        "TestHeld::test_inherited", "test_kept[0]", "test_kept[1]", "test_kept[2]", "test_kept[3]",
        "test_made"]
    assert all(members)


def test_a_whole_verdict_whose_rows_are_not_its_files_collected_tests_is_named(
        ci, tmp_path, monkeypatch, capsys):
    """`ci.py partition VERDICT COLLECTED...`: a whole run's rows on the floor Python against what pytest
    collects: green where each collected test is in one row (a split file's groups and a site's judge
    together, a file CI runs apart left out), and exit 1 naming the file and the tests where one is in no row,
    in two, or in a row and not collected."""
    listed = tmp_path / "collected.txt"
    listed.write_text("tests/test_a.py::test_one\ntests/test_a.py::test_two[x]\ntests/test_b.py::test_site\n"
                      "tests/test_gen_api.py::test_apart\n\n3 tests collected in 1.0s\n")

    def judged(rows: list[dict]) -> tuple[int, list[str]]:
        (tmp_path / "verdict.json").write_text(json.dumps({"rows": rows}))
        monkeypatch.setattr(sys, "argv", ["ci.py", "partition", str(tmp_path / "verdict.json"), str(listed)])
        return ci.main(), capsys.readouterr().out.splitlines()[1:]

    whole = [{"name": "3.12:tests/test_a.py#1/2", "tests": {"tests/test_a.py::test_one": 1.0}},
             {"name": "3.12:tests/test_a.py#2/2", "tests": {"tests/test_a.py::test_two[x]": 1.0}},
             {"name": "3.12:tests/test_b.py#test_site", "tests": {"tests/test_b.py::test_site": 9.0}},
             {"name": "3.14:tests/test_a.py", "tests": {"tests/test_a.py::test_one": 1.0}},
             {"name": "3.12-test_b-0123456789ab-draws-0-8"}]
    assert judged(whole) == (0, [])
    assert judged([whole[0], whole[0], *whole[2:], {"name": "3.12:tests/test_c.py", "tests": {
        "tests/test_c.py::test_gone": 1.0}}]) == (1, [
            "tests/test_a.py: 1 of its 2 tests collected and in no row: test_two[x]",
            "tests/test_a.py: 1 of its 2 tests in two rows: test_one",
            "tests/test_c.py: 1 of its 0 tests in a row and not collected: test_gone"])


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
    each within half again), so the next plan cuts it to fit, and so does a file whose tests together took
    under a third of its record though its cut fits; a file whose cut fits and whose total holds keeps its
    entries byte for byte, as does one whose heaviest group is a test past the budget alone, which no cut
    shortens, though its time moves; a file with a red group keeps its own but for what its rows timed past
    them, which raises them (`raised`); a file the checkout no longer has loses them; the newest Python's
    times move nothing."""
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
    # A file whose cut still fits though its tests now take a sixth of what its record says.
    file("test_shrunk", [100.0] * 6, [[15.0, 15.0], [15.0, 15.0], [15.0, 15.0]])
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
    for name in ("test_grown", "test_drift", "test_shrunk"):
        assert {node: s for node, s in folded.items() if f"/{name}.py" in node} == {
            node: s for node, s in observed.items() if f"/{name}.py" in node}
    for name in ("test_steady", "test_straggler"):
        assert {node: s for node, s in folded.items() if f"/{name}.py" in node} == {
            node: s for node, s in record.items() if f"/{name}.py" in node}
    # Its green group's tests took 400 s of their recorded 150 s: raised; its red group's took no more.
    assert {node: s for node, s in folded.items() if "/test_partial.py" in node} == {
        **{node: s for node, s in record.items() if "/test_partial.py" in node},
        "tests/test_partial.py::test_000": 400.0, "tests/test_partial.py::test_001": 400.0}
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


def test_a_file_a_tasks_deadline_cut_off_takes_the_times_of_the_tests_it_finished(ci, tmp_path, monkeypatch):
    """A file no record names and no green run times weighs nothing the plan reads, so it runs whole and
    its task's deadline cuts it off every run (2026-11-02: base/kernels' test_algebra_laws, 46 of its ~70
    tests in 1,684 s): the fold takes the times its cut-off row observed of the tests it finished, as it
    records any test's, where its record lacks them; the plan then cuts the file by them, a test it never
    reached weighing their mean. A file red otherwise keeps its record."""
    heavy, failed = "tests/test_heavy.py", "tests/test_failed.py"
    (tmp_path / ".armada.json").write_text(json.dumps({"target": 240, "environment": {}}))
    (tmp_path / heavy).write_text("".join(f"def test_{index}():\n    pass\n" for index in range(6)))
    (tmp_path / failed).write_text("def test_a():\n    pass\n")
    (tmp_path / "tests/test_durations.json").write_text("{}\n")
    main = commit(tmp_path, "main")
    rows = [{"name": f"3.12:{heavy}", "exitCode": 1, "kind": "deadline", "seconds": 400.0,
             "tests": {f"{heavy}::test_{index}": 100.0 for index in range(4)}},
            {"name": f"3.12:{failed}", "exitCode": 1, "kind": "nodes", "seconds": 5.0,
             "tests": {f"{failed}::test_a": 5.0}}]
    verdict = tmp_path / "main.json"
    verdict.write_text(json.dumps({"sha": main, "part": "all", "rows": rows}))
    monkeypatch.setattr(sys, "argv", ["ci.py", "durations", str(verdict)])
    assert ci.main() == 0
    # Its last test, which the deadline stopped, is not taken.
    assert json.loads((tmp_path / "tests/test_durations.json").read_text()) == {
        f"{heavy}::test_{index}": 100.0 for index in range(3)}
    planned = sorted((entry["tests"], entry["split"]) for entry in ci.plan(240.0, {})
                     if entry["python"] == "3.12" and entry["tests"] == heavy)
    assert planned == [(heavy, "1/2"), (heavy, "2/2")]


def test_a_recorded_test_that_grew_past_its_deadline_raises_its_record_and_the_next_plan_packs_it_alone(
        ci, tmp_path, monkeypatch):
    """A test recorded at 100 s that now takes 400 s makes the group the old time packed it in outrun its
    task's deadline: the file is red and incomplete, so no green run would ever time it again, and its
    group would die at the deadline every run (Main, 2026-11-17: base/kernels'
    test_attention_mosaic_compile). The fold raises the entry to what the cut-off row timed of it,
    lowering nothing, and the next plan packs it alone."""
    grown = "tests/test_grown.py"
    (tmp_path / ".armada.json").write_text(json.dumps({"target": 240, "environment": {}}))
    (tmp_path / grown).write_text("".join(f"def test_{name}():\n    pass\n" for name in "abcdef"))
    record = {f"{grown}::test_a": 100.0, **{f"{grown}::test_{name}": 20.0 for name in "bcdef"}}
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(record))
    main = commit(tmp_path, "main")
    before = [entry for entry in ci.plan(240.0, {}) if entry["python"] == "3.12" and grown in entry["tests"]]
    # Its group ran test_a for 400 s, then test_b, which the deadline stopped; the other group was green.
    rows = [{"name": f"3.12:{grown}#1/2", "exitCode": 1, "kind": "deadline", "seconds": 420.0,
             "tests": {f"{grown}::test_a": 400.0, f"{grown}::test_b": 5.0}},
            {"name": f"3.12:{grown}#2/2", "exitCode": 0, "seconds": 60.0,
             "tests": {f"{grown}::test_{name}": 10.0 for name in "cde"}}]
    verdict = tmp_path / "main.json"
    verdict.write_text(json.dumps({"sha": main, "part": "all", "rows": rows}))
    monkeypatch.setattr(sys, "argv", ["ci.py", "durations", str(verdict)])
    assert ci.main() == 0
    # Raised where it grew; the stopped test_b and the green group's faster times lower nothing.
    assert json.loads((tmp_path / "tests/test_durations.json").read_text()) == {
        **record, f"{grown}::test_a": 400.0}
    after = [entry for entry in ci.plan(240.0, {}) if entry["python"] == "3.12" and grown in entry["tests"]]
    heaviest = max(after, key=lambda entry: entry["weight"])
    assert len(after) > len(before) and heaviest["weight"] < 400.0 + 20.0


def test_a_cold_run_is_kept_as_its_tests_cold_times_and_every_cut_weighs_them(ci, tmp_path, monkeypatch):
    """A run whose test compiled a program the shared cache lacked (its row's `cold`) times that test cold:
    the fold keeps its newest green cold time beside the warm one in the one record (`session.entry`), a
    red run raising it, and no cold time moves a warm one under it; a test first timed cold takes that as
    its warm
    time too; a test the record drops loses both. Every cut and weight reads a test's cost, its cold time
    where one is recorded, so 14 cells recorded at 1 s warm and 60 s cold are no longer one group of 14 s
    (row 7939: such a group took 1,107 s cold)."""
    (tmp_path / ".armada.json").write_text(json.dumps({"target": 240}))
    (tmp_path / "tests/test_cells.py").write_text("".join(f"def test_{name}(): pass\n" for name in "abc"))
    record = {"tests/test_cells.py::test_a": 1.0, "tests/test_cells.py::test_b": [2.0, 50.0],
              "tests/test_gone.py::test_x": [3.0, 9.0]}
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(record, indent=0) + "\n")
    rows = [{"name": "3.12:tests/test_cells.py", "exitCode": 0, "seconds": 70.0,
             "tests": {"tests/test_cells.py::test_a": 30.0, "tests/test_cells.py::test_b": 40.0,
                       "tests/test_cells.py::test_c": 20.0},
             "cold": ["tests/test_cells.py::test_a", "tests/test_cells.py::test_b",
                      "tests/test_cells.py::test_c"]}]
    verdict = tmp_path / "main.json"
    verdict.write_text(json.dumps({"sha": commit(tmp_path, "main"), "part": "all", "rows": rows}))
    monkeypatch.setattr(sys, "argv", ["ci.py", "durations", str(verdict)])
    assert ci.main() == 0
    folded = json.loads((tmp_path / "tests/test_durations.json").read_text())
    a, b, c = (f"tests/test_cells.py::test_{name}" for name in "abc")
    assert folded == {a: [1.0, 30.0], b: [2.0, 40.0], c: [20.0, 20.0]}
    assert ci.run_keys.durations_record() == {a: 30.0, b: 40.0, c: 20.0}
    # A red run past a cold time raises it; a green warm run moves only the warm time.
    rows = [{"name": "3.12:tests/test_cells.py", "exitCode": 1, "seconds": 60.0,
             "tests": {"tests/test_cells.py::test_a": 60.0}, "cold": ["tests/test_cells.py::test_a"]}]
    verdict.write_text(json.dumps({"sha": commit(tmp_path, "main"), "part": "all", "rows": rows}))
    assert ci.main() == 0
    assert json.loads((tmp_path / "tests/test_durations.json").read_text())[a] == [1.0, 60.0]
    # A cut weighs each cell by its cost: fourteen 1 s warm, 60 s cold cells cut at a 240 s target are no one
    # group.
    cells = [ci.session.cost([1.0, 60.0])] * 14
    assert max(ci.cuts.split(ci.session, cells, sum(cells), 240.0, cut=1)) <= 240.0


def test_a_green_run_lowers_each_test_its_record_holds_well_past_what_it_took(ci, tmp_path, monkeypatch):
    """A test whose record a red, cut-off or contended run once raised comes down where its green run takes
    well under it, test by test (fold.py's `entries`): past `session.KEPT` times what it took and by over an
    eighth of the target, though its file is no whole-file refresh's (one of its groups red) and though it
    compiles cold every run (a cold run bounds its warm time from above); a drift within either bound
    lowers nothing, nor does a red run however fast, a reused row (the times an earlier fold took) or one
    armada ran alone after it failed beside other tasks (row 10776: test_compile_ahead's frontier test held
    274.7 s against the 42 s each green run took). A cold time is the newest green cold run's, and the same
    verdict folded again moves nothing."""
    (tmp_path / ".armada.json").write_text(json.dumps({"target": 240}))
    name = "tests/test_slow.py"
    (tmp_path / name).write_text("".join(f"def test_{x}(): pass\n" for x in "abcdefgh"))
    a, b, c, d, e, f, g, h = (f"{name}::test_{x}" for x in "abcdefgh")
    record = {a: 274.7, b: [274.7, 36.6], c: 50.0, d: 130.0, e: 200.0, f: [10.0, 33.8], g: 200.0, h: 200.0}
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(record, indent=0) + "\n")
    rows = [{"name": f"3.12:{name}#1/4", "exitCode": 0, "seconds": 300.0,
             "tests": {a: 42.0, b: 42.0, c: 30.0, d: 90.0, f: 156.0}, "cold": [b, f]},
            # Another group red: no whole-file refresh, and its fast test lowers nothing.
            {"name": f"3.12:{name}#2/4", "exitCode": 1, "seconds": 20.0, "tests": {e: 20.0}},
            {"name": f"3.12:{name}#3/4", "exitCode": 0, "seconds": 20.0, "tests": {g: 20.0}, "cached": "k"},
            {"name": f"3.12:{name}#4/4", "exitCode": 0, "seconds": 20.0, "tests": {h: 20.0},
             "aloneAfter": "timed out beside other tasks"}]
    verdict = tmp_path / "main.json"
    verdict.write_text(json.dumps({"sha": commit(tmp_path, "main"), "part": "all", "rows": rows}))
    monkeypatch.setattr(sys, "argv", ["ci.py", "durations", str(verdict)])
    lowered = {a: 42.0, b: [42.0, 42.0], c: 50.0, d: 130.0, e: 200.0, f: [10.0, 156.0], g: 200.0, h: 200.0}
    for _ in range(2):
        assert ci.main() == 0
        assert json.loads((tmp_path / "tests/test_durations.json").read_text()) == lowered


def test_one_verdict_folded_again_moves_nothing_and_a_cold_time_follows_only_runs_the_plan_would_make(
        ci, tmp_path, monkeypatch):
    """Each test is lowered before its file is folded whole, so a file whose lowered tests make it stale is
    refreshed at once and the same verdict folded again moves nothing (review: lowered after, the refresh
    fired on the second fold). A cold time follows a green run armada made as the plan packs it, never one
    it ran alone after a failure beside other tasks; a red run's past it raises it whatever order the rows
    come in."""
    (tmp_path / ".armada.json").write_text(json.dumps({"target": 240}))
    pair, cells = "tests/test_pair.py", "tests/test_cells.py"
    (tmp_path / pair).write_text("def test_one(): pass\ndef test_two(): pass\n")
    (tmp_path / cells).write_text("".join(f"def test_{x}(): pass\n" for x in "ijk"))
    one, two = f"{pair}::test_one", f"{pair}::test_two"
    i, j, k = (f"{cells}::test_{x}" for x in "ijk")
    record = {one: 200.0, two: 100.0, i: [10.0, 156.0], j: [10.0, 100.0], k: [10.0, 100.0]}
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(record, indent=0) + "\n")
    rows = [{"name": f"3.12:{pair}#1/2", "exitCode": 0, "seconds": 80.0, "tests": {one: 80.0}},
            {"name": f"3.12:{pair}#2/2", "exitCode": 0, "seconds": 200.0, "tests": {two: 200.0}},
            {"name": f"3.12:{cells}#1/3", "exitCode": 0, "seconds": 42.0, "tests": {i: 42.0}, "cold": [i],
             "aloneAfter": "timed out beside other tasks"},
            {"name": f"3.12:{cells}#2/3", "exitCode": 0, "seconds": 40.0, "tests": {j: 40.0, k: 300.0},
             "cold": [j, k]},
            {"name": f"3.12:{cells}#3/3", "exitCode": 1, "seconds": 300.0, "tests": {j: 300.0, k: 40.0},
             "cold": [j, k]}]
    verdict = tmp_path / "main.json"
    verdict.write_text(json.dumps({"sha": commit(tmp_path, "main"), "part": "all", "rows": rows}))
    monkeypatch.setattr(sys, "argv", ["ci.py", "durations", str(verdict)])
    folded = {one: 80.0, two: 200.0, i: [10.0, 156.0], j: [10.0, 300.0], k: [10.0, 300.0]}
    for _ in range(2):
        assert ci.main() == 0
        assert json.loads((tmp_path / "tests/test_durations.json").read_text()) == folded


def test_a_file_refreshed_whole_takes_no_time_from_a_row_that_is_no_planning_evidence(
        ci, tmp_path, monkeypatch):
    """A row armada ran alone after a failure beside other tasks counts for whether its file is whole and
    which tests it holds, never as their time: lowering its other tests can make the file stale and refresh
    it whole, which keeps the isolated test's record (review: it wrote the isolated run's 40 s)."""
    (tmp_path / ".armada.json").write_text(json.dumps({"target": 120}))
    name = "tests/test_three.py"
    (tmp_path / name).write_text("def test_a(): pass\ndef test_b(): pass\ndef test_c(): pass\n")
    a, b, c = (f"{name}::test_{x}" for x in "abc")
    (tmp_path / "tests/test_durations.json").write_text(json.dumps({a: 160.0, b: 100.0, c: 20.0}) + "\n")
    rows = [{"name": f"3.12:{name}#1/3", "exitCode": 0, "seconds": 40.0, "tests": {a: 40.0}},
            {"name": f"3.12:{name}#2/3", "exitCode": 0, "seconds": 40.0, "tests": {b: 40.0},
             "aloneAfter": "timed out beside other tasks"},
            {"name": f"3.12:{name}#3/3", "exitCode": 0, "seconds": 175.0, "tests": {c: 175.0}}]
    verdict = tmp_path / "main.json"
    verdict.write_text(json.dumps({"sha": commit(tmp_path, "main"), "part": "all", "rows": rows}))
    monkeypatch.setattr(sys, "argv", ["ci.py", "durations", str(verdict)])
    for _ in range(2):
        assert ci.main() == 0
        folded = json.loads((tmp_path / "tests/test_durations.json").read_text())
        assert folded == {a: 40.0, b: 100.0, c: 175.0}
    # A red isolated retry (its last outcome, timed out alone too) still raises what it timed past its record.
    rows[1] = {**rows[1], "exitCode": 1, "kind": "deadline", "tests": {b: 400.0}}
    verdict.write_text(json.dumps({"sha": commit(tmp_path, "main"), "part": "all", "rows": rows}))
    assert ci.main() == 0
    assert json.loads((tmp_path / "tests/test_durations.json").read_text())[b] == 400.0


def test_a_file_armada_has_not_timed_weighs_no_less_than_its_tests_recorded_costs(ci, tmp_path):
    """Scaled by how armada ran the files it timed, a file it has not is never weighed below its own tests'
    recorded costs: here a timed file ran in a tenth of its tests' cold costs, as a warm run does, and the
    other, ten tests of 100 s cold, still weighs their 1,000 s, not 100 s that would pack it whole at a 240 s
    target (row 7939's review)."""
    for name in ("test_timed", "test_untimed"):
        (tmp_path / "tests" / f"{name}.py").write_text("def test_cell(): pass\n")
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {f"tests/{name}.py::test_cell[{n}]": [1.0, 100.0] for name in ("test_timed", "test_untimed")
         for n in range(10)}))
    timings = {"files": {"3.12:tests/test_timed.py": 100.0}}
    assert ci.weights(["tests/test_timed.py", "tests/test_untimed.py"], timings) == {
        "tests/test_timed.py": 1000.0, "tests/test_untimed.py": 1000.0}


def test_a_tasks_peak_bounds_its_tests_and_a_heavy_files_tasks_run_large(ci, tmp_path, monkeypatch):
    """Each row's task peak (`task_peak`) is folded into its tests' entries as their newest, a later lighter
    task lowering it and a heavier raising it, beside their warm and cold times; a file any of whose
    tests' peaks passes `LARGE_FROM` has every task that runs it planned large, so no two share a medium,
    and a file under it is left to armada's own sizing (row 6003: a test's ranks peaked at 7.17 GiB and
    two packed on a medium were killed)."""
    (tmp_path / ".armada.json").write_text(json.dumps({"target": 240, "environment": {}}))
    for name in ("test_ranks", "test_light"):
        (tmp_path / "tests" / f"{name}.py").write_text("def test_a(): pass\ndef test_b(): pass\n")
    record = {"tests/test_ranks.py::test_a": 10.0, "tests/test_ranks.py::test_b": [5.0, 30.0],
              "tests/test_light.py::test_a": 1.0}
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(record))
    ranks, light = 7_700_000_000, 900_000_000
    rows = [{"name": "3.12:tests/test_ranks.py", "exitCode": 0, "seconds": 40.0, "peak": ranks,
             "tests": {"tests/test_ranks.py::test_a": 10.0, "tests/test_ranks.py::test_b": 30.0},
             "cold": ["tests/test_ranks.py::test_b"]},
            {"name": "3.12:tests/test_light.py", "exitCode": 0, "seconds": 1.0, "peak": light,
             "tests": {"tests/test_light.py::test_a": 1.0}}]
    verdict = tmp_path / "main.json"
    verdict.write_text(json.dumps({"sha": commit(tmp_path, "main"), "part": "all", "rows": rows}))
    monkeypatch.setattr(sys, "argv", ["ci.py", "durations", str(verdict)])
    assert ci.main() == 0
    folded = json.loads((tmp_path / "tests/test_durations.json").read_text())
    assert folded == {"tests/test_ranks.py::test_a": [10.0, None, ranks],
                      "tests/test_ranks.py::test_b": [5.0, 30.0, ranks],
                      "tests/test_light.py::test_a": [1.0, None, light]}
    assert ci.run_keys.durations_record()["tests/test_ranks.py::test_b"] == 30.0
    assert ci.session.heavy_files() == {"tests/test_ranks.py"}
    entries = [entry for entry in ci.plan(240.0, {"files": {}}) if entry["python"] == "3.12"]
    # The task that runs the heavy file, light files packed beside it or not, is large.
    assert [entry.get("size") for entry in entries if "tests/test_ranks.py" in entry["tests"]] == ["large"]
    # A newer, lighter task (a group of its own) lowers its tests' peak.
    rows = [{**rows[0], "peak": light}]
    verdict.write_text(json.dumps({"sha": commit(tmp_path, "main"), "part": "all", "rows": rows}))
    assert ci.main() == 0
    lowered = json.loads((tmp_path / "tests/test_durations.json").read_text())
    assert lowered["tests/test_ranks.py::test_a"][2] == light
    assert ci.session.heavy_files() == set()
    assert all("size" not in entry for entry in ci.plan(240.0, {"files": {}}))


def test_a_killed_tasks_peak_raises_its_files_tests_and_heavy_tasks_run_large(ci, tmp_path, monkeypatch):
    """A red row whose pytest left no case (killed before it reported) raises its file's tests' peaks to its
    own, never lowering one; and `sized` puts any task that runs a heavy file's tests on a large
    container, a check-1 site's map task or judge (its `tests` the file) among them, and no task that only
    looks a kept row up."""
    (tmp_path / ".armada.json").write_text(json.dumps({"target": 240, "environment": {}}))
    (tmp_path / "tests" / "test_ranks.py").write_text("def test_a(): pass\ndef test_b(): pass\n")
    record = {"tests/test_ranks.py::test_a": [1.0, None, 9_000_000_000], "tests/test_ranks.py::test_b": 2.0}
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(record))
    killed = 7_700_000_000
    rows = [{"name": "3.12:tests/test_ranks.py#1/2", "exitCode": 1, "seconds": 0.0, "peak": killed,
             "tests": {}}]
    verdict = tmp_path / "main.json"
    verdict.write_text(json.dumps({"sha": commit(tmp_path, "main"), "part": "all", "rows": rows}))
    monkeypatch.setattr(sys, "argv", ["ci.py", "durations", str(verdict)])
    assert ci.main() == 0
    folded = json.loads((tmp_path / "tests/test_durations.json").read_text())
    assert folded["tests/test_ranks.py::test_a"][2] == 9_000_000_000
    assert folded["tests/test_ranks.py::test_b"][2] == killed
    heavy = {"tests/test_ranks.py"}
    assert ci.session.sized({"tests": "tests/test_ranks.py", "split": "test_site"}, heavy)["size"] == "large"
    assert ci.session.sized({"tests": "tests/test_light.py tests/test_ranks.py"}, heavy)["size"] == "large"
    assert "size" not in ci.session.sized({"tests": f"tests/test_ranks.py{ci.session.KEPT_AT}abc"}, heavy)


def test_a_resumed_attempts_rows_carry_the_earlier_attempts_peak(tmp_path, monkeypatch):
    """The progress an earlier attempt kept carries its peak; the next attempt's resumed tests are bounded by
    it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "progress", Path(__file__).parents[1] / "tools/armada/progress.py")
    progress = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(progress)
    (tmp_path / "resume").mkdir()
    (tmp_path / "resume" / progress.PROGRESS).write_text(
        json.dumps({"node": "tests/test_x.py::heavy", "seconds": 3.0, "peak": 7_000}) + "\n"
        + json.dumps({"node": "tests/test_x.py::light", "seconds": 1.0, "peak": 2_000}) + "\n")
    monkeypatch.setenv("ARMADA_RESUME", str(tmp_path / "resume"))
    monkeypatch.setenv("ARMADA_ARTIFACTS", str(tmp_path / "now"))
    passed, cold, peak = progress.resumed(["tests/test_x.py"], lambda node: node.split("::")[0])
    assert sorted(passed) == ["tests/test_x.py::heavy", "tests/test_x.py::light"]
    assert (cold, peak) == (set(), 7_000)
    carried = [json.loads(line) for line in (tmp_path / "now" / progress.PROGRESS).read_text().splitlines()]
    assert {line["peak"] for line in carried} == {7_000}


def test_a_tests_recorded_peak_moves_only_past_its_band_or_across_the_large_bound(ci):
    """A task's exact bytes differ every run: the record keeps a test's peak until a newer one moves by more
    than `PEAK_MOVES` of it or crosses `LARGE_FROM`, so a fold rewrites only the entries whose sizing could
    change (train 1039's review F5)."""
    moved, large = ci.session.moved_peak, ci.session.LARGE_FROM
    assert moved(None, 1_000) == 1_000
    assert moved(1_000_000_000, 1_200_000_000) == 1_000_000_000
    assert moved(1_000_000_000, 1_300_000_000) == 1_300_000_000
    assert moved(large - 10, large + 10) == large + 10
    assert moved(large + 10, large - 10) == large - 10
