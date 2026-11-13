"""Dew's CI on armada (tools/armada/ci.py) reusing a test file's green row: the plan reads the shared test
cache with its read token, keeps the files whose rows ran green on exactly these inputs, spreads their
lookups over tasks of the target's, and a task reuses each kept row it finds fresh and runs the rest; the
inputs a row is keyed by, which a planner's clean checkout and a task's carried one read alike."""

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from armada_support import ROOT, armada_task, cached_checkout, interpreter, loaded, written


@pytest.fixture
def ci(tmp_path, monkeypatch):
    return loaded(tmp_path, monkeypatch)


def planned_checkout(ci, root: Path, monkeypatch, cache_worker,
                     *pinned: str) -> tuple[Path, Callable[[], list[str]]]:
    """Three 3.12 test files, each weighing about a task of its own, run green on CPU class
    GenuineIntel-6-143 against the shared test cache (`cached_checkout`), in a checkout whose .armada.json
    pins its tasks to `pinned` (vendor-family-model, one or several), with the cache's read token: where
    their verdict is written, and the plan's 3.12 entries' files, each entry's sorted."""
    verdict = cached_checkout(ci, root, monkeypatch, cache_worker)
    classes = [{"vendor": vendor, "family": int(family), "model": int(model)}
               for vendor, family, model in (each.split("-") for each in pinned)]
    config = json.loads((ROOT / ".armada.json").read_text())
    config["task"]["cpu"] = classes[0] if len(classes) == 1 else classes
    (root / ".armada.json").write_text(json.dumps(config))
    names = ("a", "b", "c")
    written(root, {f"test_{name}.py": "from pathlib import Path\n"
                                     f"def test_{name}():\n"
                                     f"    with Path('runs-{name}').open('a') as runs: runs.write('x')\n"
                   for name in names})
    (root / "tests/test_durations.json").write_text(json.dumps(
        {f"tests/test_{name}.py::test_{name}": 250.0 for name in names}))
    monkeypatch.setenv("DEW_XLA_CACHE_READ_TOKEN", "token")

    def planned() -> list[str]:
        return sorted(" ".join(sorted(ci.kept_of(entry["tests"].split())[0])) for entry in ci.plan(300.0, {})
                      if entry["python"] == "3.12")

    return verdict, planned


def test_a_plan_counts_why_each_row_it_did_not_keep_was_not_kept(ci, tmp_path, monkeypatch, cache_worker):
    """The plan's lookup counts why each row it did not keep was not, by its first pinned class's lookup: no
    green run on these shared inputs, or an input its last run executed changed, so a train's log states its
    reuse and no one reconstructs it after (Main, 2026-11-13: a count made hours after train 795 read the
    cache as it stood then, not as the plan did)."""
    verdict, _ = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    rows = [(f"tests/test_{name}.py", "") for name in ("a", "b", "c")]

    def looked_up() -> tuple[int, dict[str, int]]:
        missed: dict[str, int] = {}
        return len(ci.reusable("3.12", rows, ci.shared("3.12"), missed)), missed

    assert looked_up() == (0, {"no green run on these shared inputs": 3})
    armada_task(ci, monkeypatch, verdict)
    (tmp_path / "tests/test_c.py").write_text((tmp_path / "tests/test_c.py").read_text() + "\nCHANGED = 1\n")
    assert looked_up() == (2, {"an executed input changed": 1})


def test_a_run_plans_only_the_files_whose_inputs_changed_and_its_verdict_names_every_file(
        ci, tmp_path, monkeypatch, cache_worker):
    """The plan reads the test cache with its read token: the files whose rows ran green on exactly these
    inputs, on the class .armada.json pins the tasks to, share one task, which looks each row up again in
    its container, and only a changed file plans a task of its own, so containers scale with the change;
    the verdict still names every file, the unchanged ones' rows reused (the maintainer, 2026-10-28: a lane
    run with most files unchanged booted a container a task, each exiting after a lookup)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    assert planned() == ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]
    rows = armada_task(ci, monkeypatch, verdict)
    assert len(rows) == 3 and all(row["exitCode"] == 0 for row in rows)
    (tmp_path / "tests/test_c.py").write_text((tmp_path / "tests/test_c.py").read_text() + "\nCHANGED = 1\n")
    assert planned() == ["tests/test_a.py tests/test_b.py", "tests/test_c.py"]
    rows = armada_task(ci, monkeypatch, verdict)
    assert sorted(row["name"] for row in rows) == ["3.12:tests/test_a.py", "3.12:tests/test_b.py",
                                                   "3.12:tests/test_c.py"]
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in ("a", "b", "c")] == [1, 1, 2]


SCRIPT = str(ROOT / "tools/armada/ci.py")


def ran_by_script(ci, root: Path, cache_worker, names: tuple[str, ...], module: str) -> dict[str, str]:
    """Test files `names` that import src/shapes.py (`module` its text), each weighing about a task, run
    green by tools/armada/ci.py as a script, as one 3.12 task against the shared test cache, in a checkout
    whose .armada.json pins its tasks to this host's class: the environment a plan of theirs runs in, which
    names the cache and its tokens."""
    interpreter(root, "3.12")
    held = ci.cpu_class().split("-")
    config = json.loads((ROOT / ".armada.json").read_text())
    config["task"]["cpu"] = {"vendor": held[0], "family": int(held[1]), "model": int(held[2])}
    (root / ".armada.json").write_text(json.dumps(config))
    (root / "src").mkdir()
    (root / "src/shapes.py").write_text(module)
    written(root, {f"test_{name}.py": "import sys\nsys.path.insert(0, 'src')\nfrom shapes import area\n"
                                      f"def test_{name}():\n    assert area(2) == 2\n" for name in names})
    (root / "tests/test_durations.json").write_text(json.dumps(
        {f"tests/test_{name}.py::test_{name}": 250.0 for name in names}))
    (root / ".verdict").mkdir()
    url, _, _ = cache_worker
    env = {**os.environ, "DEW_XLA_CACHE_URL": url, "DEW_XLA_CACHE_TOKEN": "token",
           "DEW_XLA_CACHE_READ_TOKEN": "token"}
    verdict = root / ".verdict/verdict.json"
    tests = "--tests=" + " ".join(f"tests/test_{name}.py" for name in names)
    subprocess.run([sys.executable, SCRIPT, "task", "--python=3.12", tests, f"--out={verdict}", "--cache=on"],
                   cwd=root, env=env, check=True)
    assert [row["exitCode"] for row in json.loads(verdict.read_text())["rows"]] == [0] * len(names)
    return env


def test_a_plan_run_by_another_python_than_its_tasks_keeps_the_files_they_ran_green(
        ci, tmp_path, cache_worker):
    """The plan and a task each run tools/armada/ci.py as a script under whatever Python runs it where it runs
    (a planner's machine, a GitHub runner, a task's image), and key a file's run alike: the files a task ran
    green under one Python, a plan run by another keeps, in one task (2026-10-30: plans under 3.14 and 3.12
    found none of the 3.13 tasks' records fresh, as each Python compiles a module's units to its own digests,
    and every file took a task of its own)."""
    # A module whose units each Python compiles to its own digests: a documented, annotated function.
    env = ran_by_script(ci, tmp_path, cache_worker, ("a", "b"),
                        'def area(width: int, height: int = 1) -> int:\n    """An area."""\n'
                        '    return width * height\n')

    def version(python: str) -> str:
        return subprocess.run([python, "-c", "import sys; print(sys.version_info[:2])"], capture_output=True,
                              text=True, check=True).stdout

    # The task under this Python; the plan under another the image holds: a venv of another tested Python, or
    # the image's own.
    other = next(python for python in (*(str(ROOT / f".venv-{each}/bin/python") for each in ci.PYTHONS),
                                       shutil.which("python3") or "")
                 if Path(python).is_file() and version(python) != version(sys.executable))
    timings = tmp_path / "timings.json"
    planned = subprocess.run([other, SCRIPT, "plan", "--target=300", f"--timings={timings}"], cwd=tmp_path,
                             env=env, capture_output=True, text=True, check=True).stdout
    entries = [entry for entry in json.loads(planned)["include"] if entry["python"] == "3.12"]
    assert [sorted(ci.kept_of(entry["tests"].split())[1]) for entry in entries] == [
        [("tests/test_a.py", ""), ("tests/test_b.py", "")]]


def test_a_plan_reads_each_input_once_however_many_rows_ran_it(ci, tmp_path, cache_worker):
    """The plan keys every row it looks up from one reading of the checkout: a module three files' runs
    executed is read once, as the plan runs in CI, under the Python its keys are hashed by (2026-10-31:
    train 368's plan read each of a row's ~3,000 inputs again for each of 1,682 rows, 9.5 minutes before its
    first task)."""
    names = ("a", "b", "c")
    env = ran_by_script(ci, tmp_path, cache_worker, names,
                        "def area(width, height=1):\n    return width * height\n")
    # The plan as CI's runs it, under the Python its keys are hashed by, each file it opens counted.
    opened = tmp_path / ".verdict/opened.json"
    counted = ("import json, runpy, sys\nopened = []\n"
               "sys.addaudithook(lambda event, args: event == 'open' and opened.append(str(args[0])))\n"
               f"sys.argv = [{SCRIPT!r}, 'plan', '--target=300', '--timings={tmp_path / 'timings.json'}']\n"
               f"try:\n    runpy.run_path({SCRIPT!r}, run_name='__main__')\n"
               f"finally:\n    open({str(opened)!r}, 'w').write(json.dumps(opened))\n")
    planned = subprocess.run(["uv", "run", "--no-project", "--python", ci.HASHED_BY, "python", "-c", counted],
                             cwd=tmp_path, env=env, capture_output=True, text=True)
    entries = [entry for entry in json.loads(planned.stdout)["include"] if entry["python"] == "3.12"]
    assert [sorted(ci.kept_of(entry["tests"].split())[1]) for entry in entries] == [
        [(f"tests/test_{name}.py", "") for name in names]]
    assert json.loads(opened.read_text()).count("src/shapes.py") == 1


def test_a_plan_without_build_output_or_fetched_payloads_finds_what_a_task_recorded_with_them(
        ci, tmp_path, monkeypatch, cache_worker):
    """The plan's checkout, clean, keys rows as the task's checkout does, though the task's holds an editable
    install's metadata under src/ (setuptools' dewml.egg-info) and the fetched fixtures' payloads, which the
    planner never downloads: a build's output is no input, and a fetched payload is keyed by the digest its
    manifest pins (the maintainer, 2026-10-28: a one-file change planned 493 tasks, as the planner's src/
    listing and walk lacked them)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    payload = tmp_path / "tests/fixtures/model.npz"
    payload.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests/fixtures/model.npz.fetched.json").write_text(json.dumps({"sha256": "ab" * 32}))
    built = tmp_path / "src/dewml.egg-info"
    built.mkdir(parents=True)
    (built / "PKG-INFO").write_text("Name: dewml\n")
    payload.write_bytes(b"payload")
    armada_task(ci, monkeypatch, verdict)
    payload.unlink()
    (built / "PKG-INFO").unlink()
    built.rmdir()
    (tmp_path / "tests/test_c.py").write_text((tmp_path / "tests/test_c.py").read_text() + "\nCHANGED = 1\n")
    assert planned() == ["tests/test_a.py tests/test_b.py", "tests/test_c.py"]


def test_a_plan_finds_what_a_task_recorded_beside_a_deleted_packages_bytecode(
        ci, tmp_path, monkeypatch, cache_worker):
    """A task's checkout, carried across commits in its environment's snapshot, holds a package a commit
    deleted, its bytecode alone (src/<package>/__pycache__/); the planner's clean tree has none. Its rows key
    as the plan's: a directory of nothing but build output is no input of its parent's listing (train 636:
    src/dew/cli/ keyed every row apart from its plan's, and 240 kept rows ran in two tasks past their
    deadline)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    stale = tmp_path / "src" / "gone" / "__pycache__"
    stale.mkdir(parents=True)
    (stale / "__init__.cpython-312.pyc").write_bytes(b"bytecode")
    (tmp_path / "tests/test_a.py").write_text(
        "from pathlib import Path\ndef test_a():\n"
        "    assert sorted(path.name for path in Path('src').iterdir())\n"
        "    with Path('runs-a').open('a') as runs: runs.write('x')\n")
    armada_task(ci, monkeypatch, verdict)
    shutil.rmtree(stale.parent)
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"]


def test_a_row_that_read_its_installs_metadata_is_reused_by_a_planner_without_it(
        ci, tmp_path, monkeypatch, cache_worker):
    """A run that read its editable install's metadata (setuptools' <name>.egg-info, which importlib.metadata
    reads for a version, and which a task's environment writes into its checkout) keys no row on it: the
    environment's build output is the locks' (`shared`), and a planner, whose checkout has none, reuses the
    row (train 300: every record held src/dewml.egg-info's files, and its plan reused none of 501)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    metadata = tmp_path / "src/dewml.egg-info/PKG-INFO"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Version: 1\n")
    (tmp_path / "tests/test_a.py").write_text(
        "from pathlib import Path\ndef test_a():\n"
        "    assert Path('src/dewml.egg-info/PKG-INFO').read_text()\n"
        "    with Path('runs-a').open('a') as runs: runs.write('x')\n")
    armada_task(ci, monkeypatch, verdict)
    shutil.rmtree(metadata.parent)
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"]


def test_a_plan_reuses_nothing_for_tasks_pinned_to_another_class(ci, tmp_path, monkeypatch, cache_worker):
    """The plan looks records up under the class .armada.json pins the tasks to, never its own host's: rows
    recorded on another class plan every file a task of its own, as before."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "AuthenticAMD-25-17")
    armada_task(ci, monkeypatch, verdict)
    (tmp_path / "tests/test_c.py").write_text((tmp_path / "tests/test_c.py").read_text() + "\nCHANGED = 1\n")
    assert planned() == ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]


def test_a_plan_pinned_to_several_classes_reuses_a_row_green_on_any_of_them(
        ci, tmp_path, monkeypatch, cache_worker):
    """Tasks pinned to several CPU classes (.armada.json's task.cpu a list) land on any of them, so the plan
    reuses a file whose row ran green on these inputs on any pinned class, and plans the rest: rows split by
    class, the plan reading each pinned class's (the maintainer, 2026-10-29: the suite widened to 25/17 and
    26/17, the fleet's own classes, as one class starved lanes' verdicts)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "AuthenticAMD-25-17",
                                        "GenuineIntel-6-143")
    armada_task(ci, monkeypatch, verdict)
    (tmp_path / "tests/test_c.py").write_text((tmp_path / "tests/test_c.py").read_text() + "\nCHANGED = 1\n")
    assert planned() == ["tests/test_a.py tests/test_b.py", "tests/test_c.py"]


def test_a_row_run_green_on_these_inputs_is_reused_though_another_commits_run_of_its_file_came_after(
        ci, tmp_path, monkeypatch, cache_worker):
    """A file's row run green on exactly a checkout's inputs is reused there whatever runs of the file on
    other inputs came after it, as lanes' checks and trains of other commits run it all day: a lane's one-line
    change over a green verdict reruns only what that line reaches (owner, 2026-10-29: a verdict found 272 of
    its own 504 green files reusable an hour later, the others' records replaced by other commits' runs)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    armada_task(ci, monkeypatch, verdict)
    test_c = tmp_path / "tests/test_c.py"
    first = test_c.read_text()
    # Another commit's run of test_c, on other inputs, after this one's.
    test_c.write_text(first + "\nANOTHER_COMMIT = 1\n")
    armada_task(ci, monkeypatch, verdict)
    test_c.write_text(first)
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"]
    rows = armada_task(ci, monkeypatch, verdict)
    assert len(rows) == 3 and all(row["exitCode"] == 0 for row in rows)
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in "abc"] == [1, 1, 2]


def test_a_plan_keeps_each_green_group_of_a_heavy_file_and_a_task_runs_a_group_whose_green_run_is_gone(
        ci, tmp_path, monkeypatch, cache_worker):
    """A file heavier than the target runs as groups, each a row the plan looks up as it does a whole file's:
    once every group ran green on these inputs, one task holds them with the light files and reuses each
    row, running nothing; a kept group whose green run its task cannot read, it runs (2026-10-30: the plan
    looked up whole files alone, and a docs-only change's 1,034 green groups of heavy files each booted a
    container to read its record)."""
    verdict, _ = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    written(tmp_path, {"test_c.py": "from pathlib import Path\n" + "".join(
        f"def test_{name}():\n    with Path('runs-{name}').open('a') as runs: runs.write('x')\n"
        for name in "cd")})
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {**{f"tests/test_{name}.py::test_{name}": 250.0 for name in "ab"},
         "tests/test_c.py::test_c": 250.0, "tests/test_c.py::test_d": 250.0}))

    def planned() -> list[list[str]]:
        return sorted(sorted(entry["rows"]) for entry in ci.plan(300.0, {}) if entry["python"] == "3.12")

    heavy = ["3.12:tests/test_c.py#1/2", "3.12:tests/test_c.py#2/2"]
    assert planned() == [["3.12:tests/test_a.py"], ["3.12:tests/test_b.py"], heavy[:1], heavy[1:]]
    assert all(row["exitCode"] == 0 for row in armada_task(ci, monkeypatch, verdict))
    assert planned() == [["3.12:tests/test_a.py", "3.12:tests/test_b.py", *heavy]]
    rows = armada_task(ci, monkeypatch, verdict)
    assert sorted(row["name"] for row in rows) == ["3.12:tests/test_a.py", "3.12:tests/test_b.py", *heavy]
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in "abcd"] == [1, 1, 1, 1]
    # A kept group whose green run its task cannot read, its file changed since the plan: the task runs that
    # group alone.
    test_c = tmp_path / "tests/test_c.py"
    test_c.write_text(test_c.read_text() + "\nCHANGED_AFTER_THE_PLAN = 1\n")
    monkeypatch.setattr(sys, "argv", ["ci.py", "task", "--python=3.12", "--tests=tests/test_c.py#2/2@"
                                      "GenuineIntel-6-143:0000", f"--out={verdict}", "--cache=on"])
    assert ci.main() == 0
    ran = json.loads(verdict.read_text())["rows"]
    assert [(row["name"], row["exitCode"]) for row in ran] == [(heavy[1], 0)]
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in "cd"] == [1, 2]


def test_a_lookup_examines_the_entries_of_its_name_alone(ci, tmp_path, monkeypatch):
    """What an import's search of a directory finds is read from the entries of the searched name alone: no
    other entry is examined for being a build's output, which walks into a directory (2026-11-12: a task's
    keys examined every entry of tests/ once a lookup, 47 million of them for 156,000 lookups, and that was
    11 of the 12 minutes of this file's longest test). The answer is the one a directory holding the name's
    entries alone gives."""
    digests = ci.digests
    package = tmp_path / "package"
    for name in ("searched.py", "searched/__init__.py", "searched/inner.py", "searchedmore.py",
                 *(f"sibling{index}/module.py" for index in range(20))):
        (package / name).parent.mkdir(parents=True, exist_ok=True)
        (package / name).write_text("")
    alone = tmp_path / "alone"
    for name in ("searched.py", "searched/__init__.py"):
        (alone / name).parent.mkdir(parents=True, exist_ok=True)
        (alone / name).write_text("")
    examined: list[str] = []
    built = digests.built
    monkeypatch.setattr(digests, "built", lambda path: examined.append(path.name) or built(path))
    found = hashlib.sha256(b"searched\nsearched.py\nsearched/__init__.py").digest()
    assert digests.looked_up(package / "searched") == digests.looked_up(alone / "searched") == found
    assert {"searched.py", "searched"} <= set(examined), sorted(set(examined))
    assert set(examined) <= {"searched.py", "searched", "__init__.py", "inner.py"}, sorted(set(examined))


def test_a_plan_spreads_its_kept_rows_over_tasks_of_the_targets_lookups(ci, tmp_path, monkeypatch,
                                                                        cache_worker):
    """The plan's kept rows fill tasks of at most the target's lookups, a lookup a row in seconds, each of
    which reuses its rows, so no kept task outlasts the rest of the run and none carries more tokens than a
    process starts with (2026-10-31: main de315f9a05's kept rows were one task, whose 149,515-byte --tests
    argument started no task, and its run was not graded)."""
    verdict, _ = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    assert all(row["exitCode"] == 0 for row in armada_task(ci, monkeypatch, verdict))
    names = ["3.12:tests/test_a.py", "3.12:tests/test_b.py", "3.12:tests/test_c.py"]
    entries = [entry for entry in ci.plan(2.0, {}) if entry["python"] == "3.12"]
    assert [(entry["rows"], entry["weight"]) for entry in entries] == [(names[:2], 2.0), (names[2:], 1.0)]
    rows = armada_task(ci, monkeypatch, verdict, target=2.0)
    assert [(row["name"], row["exitCode"]) for row in rows] == [(name, 0) for name in names]
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in "abc"] == [1, 1, 1]


def test_a_kept_tasks_entry_stays_within_what_a_process_starts_with_however_high_the_target(
        ci, tmp_path, monkeypatch, cache_worker):
    """However many lookups the target allows, a kept task's entry, which armada hands its task whole as one
    environment string (ARMADA_ITEM), and its tokens, one argument (--tests), stay within the 131,071 bytes
    Linux starts a process with, and every kept row is reused (2026-10-31: main de315f9a05's kept rows were
    one task, whose 149,515-byte argument started none)."""
    verdict, _ = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    # Files of long names (beside a fetched fixture's manifest name, within a file system's 255 bytes), enough
    # that their kept rows' tokens and names pass what a process starts with together.
    names = [f"tests/test_{index:03d}{'x' * 200}.py" for index in range(300)]
    written(tmp_path, {Path(name).name: f"def test_{index}():\n    pass\n"
                       for index, name in enumerate(names)})
    durations = tmp_path / "tests/test_durations.json"
    durations.write_text(json.dumps({**json.loads(durations.read_text()),
                                     **{f"{name}::test_{index}": 1.0 for index, name in enumerate(names)}}))
    assert all(row["exitCode"] == 0 for row in armada_task(ci, monkeypatch, verdict, target=100_000.0))
    entries = [entry for entry in ci.plan(100_000.0, {}) if entry["python"] == "3.12"]
    every = sorted(f"3.12:{name}" for name in [*names, *(f"tests/test_{letter}.py" for letter in "abc")])
    assert sorted(row for entry in entries for row in entry["rows"]) == every
    assert all(ci.KEPT_AT in token for entry in entries for token in entry["tests"].split())
    assert max(len(f"ARMADA_ITEM={json.dumps(entry)}".encode()) for entry in entries) <= 131_071
    assert max(len(f"--tests={entry['tests']}".encode()) for entry in entries) <= 131_071
    rows = armada_task(ci, monkeypatch, verdict, target=100_000.0)
    assert sorted((row["name"], row["exitCode"]) for row in rows) == [(name, 0) for name in every]
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in "abc"] == [1, 1, 1]


def test_a_kept_task_runs_a_stale_row_only_within_its_deadline_and_says_the_rest_stale_not_run(
        ci, tmp_path, monkeypatch, cache_worker):
    """A kept task whose kept groups' green runs it cannot read runs each only where its recorded seconds fit
    what is left of its deadline: every other is said stale, not run (ci.py's `STALE`), its own outcome the
    queue runs again, never a deadline red, and none runs past the deadline (row 4468: a kept task's 163 of
    165 stale rows ran out its 1680 s one after another, each a deadline red); groups that fit run and
    pass."""
    verdict, _ = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    _, _, seen = cache_worker
    names = "cde"
    written(tmp_path, {"test_c.py": "from pathlib import Path\n" + "".join(
        f"def test_{name}():\n    Path('runs-{name}').write_text('x')\n" for name in names)})
    groups = " ".join(f"tests/test_c.py#{index}/3@GenuineIntel-6-143:0000" for index in (1, 2, 3))

    def kept_task(seconds: float, deadline: int) -> list[tuple[str, int, str | None]]:
        (tmp_path / "tests/test_durations.json").write_text(json.dumps(
            {f"tests/test_c.py::test_{name}": seconds for name in names}))
        monkeypatch.setattr(sys, "argv", ["ci.py", "task", "--python=3.12", f"--tests={groups}",
                                          f"--out={verdict}", "--cache=on", f"--deadline={deadline}"])
        seen.clear()
        assert ci.main() == 0
        return [(row["name"], row["exitCode"], row.get("kind"))
                for row in json.loads(verdict.read_text())["rows"]]

    assert kept_task(250.0, 60) == [(f"3.12:tests/test_c.py#{index}/3", 1, ci.STALE) for index in (1, 2, 3)]
    assert not any((tmp_path / f"runs-{name}").exists() for name in names)
    assert any("tests/test_c.py/1-of-3" in path for path in seen)
    assert kept_task(0.5, 600) == [(f"3.12:tests/test_c.py#{index}/3", 0, None) for index in (1, 2, 3)]
    assert all((tmp_path / f"runs-{name}").exists() for name in names)


def test_a_kept_task_reuses_its_rows_after_another_commit_changes_the_recorded_dependencies(
        ci, tmp_path, monkeypatch, cache_worker):
    """A plan's kept task reuses the green rows the plan found though another commit's run, of other inputs
    (a file importing a helper this commit's never did), replaced their files' last records before the task
    started: the plan hands the task what it found, never a record another run may move (2026-10-30: a
    verdict's kept task found 23 of its 259 files, ran the rest itself and timed out)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    (tmp_path / "tests/later_dependency.py").write_text("VALUE = 1\n")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-p no:cacheprovider")
    armada_task(ci, monkeypatch, verdict)
    common = ci.shared("3.12")
    at = ci.record_key("3.12", "3.12:tests/test_a.py", "GenuineIntel-6-143", common)
    before = ci.recorded(at, "token")
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"]
    entries = [entry for entry in ci.plan(300.0, {}) if entry["python"] == "3.12"]
    assert len(entries) == 1
    test_a = tmp_path / "tests/test_a.py"
    original = test_a.read_text()
    # Another commit runs between this plan and its task, importing a helper this commit never
    # imported. It shares the common hash, so its run replaces the slot this plan read.
    test_a.write_text("import later_dependency\n" + original)
    armada_task(ci, monkeypatch, verdict)
    test_a.write_text(original)
    assert ci.recorded(at, "token")["key"] != before["key"]
    command = json.loads((ROOT / ".armada.json").read_text())["task"]["command"]
    values = {**entries[0], "out": verdict}
    filled = [re.sub(r"\{(\w+)\}", lambda field: str(values[field[1]]), word) for word in command]
    monkeypatch.setattr(sys, "argv", ["ci.py", *filled[2:]])
    assert ci.main() == 0
    rows = json.loads(verdict.read_text())["rows"]
    assert len(rows) == 3 and all(row["exitCode"] == 0 for row in rows)
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in "abc"] == [2, 1, 1]


def test_a_kept_task_reuses_each_files_own_row_where_files_that_ran_in_one_process_share_a_key(
        ci, tmp_path, monkeypatch, cache_worker):
    """Files one task ran in one process share what it executed before them, so their keys are one: a kept
    task still reuses each file's own green row, and names every file once (2026-10-30: a kept task reported
    one file's row for six files, and a verdict was not graded)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {f"tests/test_{name}.py::test_{name}": 50.0 for name in "abc"}))
    armada_task(ci, monkeypatch, verdict)
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"]
    rows = armada_task(ci, monkeypatch, verdict)
    assert sorted(row["name"] for row in rows) == ["3.12:tests/test_a.py", "3.12:tests/test_b.py",
                                                   "3.12:tests/test_c.py"]
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in "abc"] == [1, 1, 1]


def test_a_helper_added_under_tests_reruns_only_the_runs_that_imported_a_module_of_its_name(
        ci, tmp_path, monkeypatch, cache_worker):
    """A helper module added under tests/ reruns no file whose run imported no module of its name, as it
    shadows nothing they import; one named as a module they imported, which tests/ leading sys.path would
    shadow, reruns each of them (owner, 2026-10-29: train 315's one new helper reran all 2490 tasks)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    armada_task(ci, monkeypatch, verdict)
    (tmp_path / "tests/quantized_basis.py").write_text("BASIS = 1\n")
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"]
    (tmp_path / "tests/pathlib.py").write_text("")
    assert planned() == ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]


def test_a_helper_named_as_a_module_only_a_child_process_imported_reruns_its_file(
        ci, tmp_path, monkeypatch, cache_worker):
    """A module only a test's child process imports (a tool run as a script) is that file's input as one its
    own process imports is: a helper of its name added under tests/ reruns the file (row 2502: the recorder
    named the pytest process's imports alone, so such a helper kept the file's green key)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    written(tmp_path, {"test_c.py": "import subprocess, sys\nfrom pathlib import Path\n"
                                    "def test_c():\n"
                                    "    with Path('runs-c').open('a') as runs: runs.write('x')\n"
                                    "    subprocess.run([sys.executable, '-c', 'import wave'],\n"
                                    "                   check=True)\n"})
    armada_task(ci, monkeypatch, verdict)
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"]
    (tmp_path / "tests/wave.py").write_text("")
    assert planned() == ["tests/test_a.py tests/test_b.py", "tests/test_c.py"]


def test_a_task_on_another_pinned_class_reuses_the_rows_its_plan_kept(
        ci, tmp_path, monkeypatch, cache_worker):
    """The files a plan keeps for reuse, their rows green on any class the tasks are pinned to, are reused by
    the task that holds them on whichever pinned class it lands on, as the plan counted them: a task that
    looked up its own class alone ran every kept file itself, in one task, past its deadline (2026-10-29: a
    verdict's kept task of 182 files on 26/17 ran 76 of them into its 1680 s deadline, the rows green on
    25/17)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "AuthenticAMD-25-17",
                                        "GenuineIntel-6-143")
    armada_task(ci, monkeypatch, verdict)
    (tmp_path / "tests/test_c.py").write_text((tmp_path / "tests/test_c.py").read_text() + "\nCHANGED = 1\n")
    assert planned() == ["tests/test_a.py tests/test_b.py", "tests/test_c.py"]
    monkeypatch.setattr(ci, "cpu_class", lambda: "AuthenticAMD-25-17")
    rows = armada_task(ci, monkeypatch, verdict)
    assert len(rows) == 3 and all(row["exitCode"] == 0 for row in rows)
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in "abc"] == [1, 1, 2]


def test_a_row_green_on_a_tree_is_reused_there_after_another_tree_ran_it_executing_another_set(
        ci, tmp_path, monkeypatch, cache_worker):
    """A row run green on a tree is reused when that tree is planned again, though a run of another tree came
    after it and executed another set of inputs (a helper its test called there): the row's list keeps each
    green run's executed set, and the plan keys this tree's inputs with its own set (2026-11-12: keyed with
    the last run's set alone, train 781's CI kept 2 of 4,180 tasks, and a reach of a comment's change planned
    7,161 rows)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    armada_task(ci, monkeypatch, verdict)
    test_a = tmp_path / "tests/test_a.py"
    main = test_a.read_text()
    # Another tree: test_a calls a helper of its own, so its run executes a set this tree's never did.
    helper = tmp_path / "tests/helper_other.py"
    helper.write_text("def value():\n    return 1\n")
    test_a.write_text("from helper_other import value\n"
                      + main.replace("def test_a():\n", "def test_a():\n    assert value() == 1\n"))
    armada_task(ci, monkeypatch, verdict)
    helper.unlink()
    test_a.write_text(main)
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"]
    rows = armada_task(ci, monkeypatch, verdict)
    assert sorted((row["name"], row["exitCode"]) for row in rows) == [
        (f"3.12:tests/test_{name}.py", 0) for name in "abc"]
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in "abc"] == [2, 1, 1]


def test_two_trees_alternating_each_reuse_their_rows_and_a_changed_unit_of_a_set_runs_its_row(
        ci, tmp_path, monkeypatch, cache_worker):
    """Two trees whose runs of a row execute different sets, planned in turn, each reuse their own green rows
    (main's and base's, a train's and a lane's); a unit of a recorded set changed in the tree runs the row
    again: a set is a hit only where every input it names holds what it held."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    test_a = tmp_path / "tests/test_a.py"
    plain = test_a.read_text()
    helper = tmp_path / "tests/helper_other.py"
    calling = "from helper_other import value\n" + plain.replace("def test_a():\n",
                                                                 "def test_a():\n    assert value() == 1\n")

    def tree(*, other: bool, body: str = "def value():\n    return 1\n") -> None:
        if other:
            helper.write_text(body)
            test_a.write_text(calling)
        else:
            helper.unlink(missing_ok=True)
            test_a.write_text(plain)

    for other in (False, True):
        tree(other=other)
        armada_task(ci, monkeypatch, verdict)
    for other in (False, True, False, True):
        tree(other=other)
        assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"], other
    # The other tree's helper changed in the unit test_a ran: its row runs, the rest reused.
    tree(other=True, body="def value():\n    return 2 - 1\n")
    assert planned() == ["tests/test_a.py", "tests/test_b.py tests/test_c.py"]


def flaky_row(root: Path) -> tuple[Path, str]:
    """tests/test_a.py made flaky: red where FAIL is set, its red run executing a unit its green runs never do
    (src/flaky.py's `failing`), so the red's executed set, and key, are its own: the file and its text."""
    (root / "src").mkdir(exist_ok=True)
    (root / "src/flaky.py").write_text("def failing():\n    raise AssertionError('flaky')\n")
    flaky = ("import os, sys\nfrom pathlib import Path\nsys.path.insert(0, 'src')\nimport flaky\n"
             "def test_a():\n"
             "    with Path('runs-a').open('a') as runs: runs.write('x')\n"
             "    if os.environ.get('FAIL'):\n        flaky.failing()\n")
    (root / "tests/test_a.py").write_text(flaky)
    return root / "tests/test_a.py", flaky


def test_a_red_run_of_a_row_revokes_its_earlier_green_runs_by_any_set_token_or_tree(
        ci, tmp_path, monkeypatch, cache_worker):
    """A red run of a row (a flaky one on the very inputs a green run recorded) revokes every green run of it
    recorded before it, on that class and those shared inputs: a plan finds none by any executed set, a task
    handed a plan's kept token from before the red runs the row, and a green run of another tree after the red
    brings back none of this tree's earlier green runs (review, 2026-11-12: a red that executed another set
    revoked nothing by its own key)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    test_a, flaky = flaky_row(tmp_path)
    armada_task(ci, monkeypatch, verdict)
    entries = [entry for entry in ci.plan(300.0, {}) if entry["python"] == "3.12"]
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"] and len(entries) == 1
    # The flaky red, on these very inputs.
    monkeypatch.setenv("FAIL", "1")
    assert [row["exitCode"] for row in armada_task(ci, monkeypatch, verdict, policy="off")] == [1, 0, 0]
    monkeypatch.delenv("FAIL")
    assert planned() == ["tests/test_a.py", "tests/test_b.py tests/test_c.py"]
    # Another tree's green run after the red, then this tree again: it brings back none of this tree's green
    # runs from before the red.
    test_a.write_text(flaky + "\nOTHER_TREE = 1\n")
    armada_task(ci, monkeypatch, verdict)
    test_a.write_text(flaky)
    assert planned() == ["tests/test_a.py", "tests/test_b.py tests/test_c.py"]
    # The kept token the plan made before the red, handed to its task now: the row runs, and its green run,
    # after the red, is reused from then on.
    command = json.loads((ROOT / ".armada.json").read_text())["task"]["command"]
    values = {**entries[0], "out": verdict}
    filled = [re.sub(r"\{(\w+)\}", lambda field: str(values[field[1]]), word) for word in command]
    monkeypatch.setattr(sys, "argv", ["ci.py", *filled[2:]])
    before = len((tmp_path / "runs-a").read_text())
    assert ci.main() == 0
    assert len((tmp_path / "runs-a").read_text()) == before + 1
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"]


def test_a_red_that_read_the_rows_state_before_another_red_still_revokes_the_green_between_them(
        ci, tmp_path, monkeypatch, cache_worker):
    """Two red runs of a row at once, the second having read the row's revocation state before the first wrote
    it, and a green run recorded between their writes: after the second red the green stands no more. A red
    begins a fresh epoch, whatever it read; a count the second red read stale would write the first's number
    again and let the green stand (review, 2026-11-12)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    flaky_row(tmp_path)
    sender = ci.records.SEND

    def red(stale: bool = False) -> None:
        read = []

        def send(method: str, url: str, headers, body, timeout: float):
            # Its first read of the row's state as before the other red's write: the row had none then.
            if stale and method == "GET" and ("-reds/" in url or "-epochs/" in url) and "test_a" in url \
                    and not read:
                read.append(url)
                return 404, {}, io.BytesIO(b"")
            return sender(method, url, headers, body, timeout)

        with monkeypatch.context() as patched:
            patched.setattr(ci.records, "SEND", send)
            patched.setenv("FAIL", "1")
            rows = armada_task(ci, monkeypatch, verdict, policy="off")
        assert [row["exitCode"] for row in rows] == [1, 0, 0]

    armada_task(ci, monkeypatch, verdict)
    red()
    armada_task(ci, monkeypatch, verdict)
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"]
    red(stale=True)
    assert planned() == ["tests/test_a.py", "tests/test_b.py tests/test_c.py"]


def test_a_lookup_whose_epoch_is_unread_or_gone_reuses_nothing(ci, tmp_path, monkeypatch, cache_worker):
    """Where the Worker gives no answer for a row's last red, no green run of the row stands: the plan keeps
    none, as an unread red may revoke any of them (review, 2026-11-12: an unread record read as none)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    armada_task(ci, monkeypatch, verdict)
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"]

    # The Worker unavailable for the rows' epochs alone.
    sender = ci.records.SEND

    def send(method: str, url: str, headers, body, timeout: float):
        if "-epochs/" in url:
            raise ConnectionResetError("the connection to the Worker was reset")
        return sender(method, url, headers, body, timeout)

    with monkeypatch.context() as patched:
        patched.setattr(ci.records, "SEND", send)
        assert planned() == ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]
    # Its epochs gone (the bucket's expiry), every green run recorded in them stands no more: a green written
    # after a red it read nothing of, kept past that red's epoch, is never reused (review, 2026-11-12).
    stored = cache_worker[1]
    for key in [key for key in stored if "-epochs/" in key]:
        del stored[key]
    assert planned() == ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]


def test_a_red_runs_epoch_write_sent_again_after_a_later_one_never_puts_it_back(
        ci, tmp_path, monkeypatch, cache_worker):
    """An epoch's write sent again after a later epoch replaced it (a retry of a write the Worker took but
    whose answer was lost; a kept connection replaying its request) changes nothing, so the green run recorded
    in the epoch between stays revoked by the red that began the later one (review, 2026-11-12)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    flaky_row(tmp_path)
    sent: list[tuple] = []
    sender = ci.records.SEND

    def keeping(method: str, url: str, headers, body, timeout: float):
        if method == "PUT" and "-epochs/" in url and "test_a" in url:
            sent.append((method, url, dict(headers), body, timeout))
        return sender(method, url, headers, body, timeout)

    def red() -> None:
        with monkeypatch.context() as patched:
            patched.setenv("FAIL", "1")
            rows = armada_task(ci, monkeypatch, verdict, policy="off")
        assert [row["exitCode"] for row in rows] == [1, 0, 0]

    armada_task(ci, monkeypatch, verdict)
    with monkeypatch.context() as patched:
        patched.setattr(ci.records, "SEND", keeping)
        red()
    first_red = sent[-1]
    armada_task(ci, monkeypatch, verdict)
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"]
    red()
    # The first red's epoch write, sent again as it was sent.
    status, _, answer = sender(*first_red)
    answer.close()
    assert status in (204, 412)
    assert planned() == ["tests/test_a.py", "tests/test_b.py tests/test_c.py"]
