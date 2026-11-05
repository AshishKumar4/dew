"""What the test cache's keys stand for: each input of a row's key (ci.py's `shared` and `cache_key`, what
`record` keeps, .armada.json's environment key) changed through the real flow on small checkouts, and the
rows that must and must not run again. The harness the tests run is the repository's own tools (`loaded`) or
a copy of them an edit changes (`copied_tools`)."""

import json
import re
from pathlib import Path

import pytest
from armada_support import (
    armada_task,
    cached_checkout,
    cached_runs,
    commit,
    copied_tools,
    fetcher,
    gpu_lane_at,
    green_runs,
    interpreter,
    lane_runs,
    loaded,
    written,
)
from tool_support import load

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ci(tmp_path, monkeypatch):
    return loaded(tmp_path, monkeypatch)


def install_inputs(root: Path, script: str) -> set[str]:
    """The files of the checkout at `root` that `script` (its path there) names in its commands, and those
    the shell scripts among them name in turn: a limited check of what an install runs or reads (the words of
    each line that is no comment, a command substitution's and a quote's taken apart), not a shell
    interpreter, so a path a variable builds (`locks/$name.txt`) is none of them."""
    found, pending = {script}, [script]
    while pending:
        lines = [line for line in (root / pending.pop()).read_text().splitlines()
                 if not line.lstrip().startswith("#")]
        words = re.findall(r"[\w./-]+", "\n".join(lines))
        files = {word for word in words if not word.startswith(("/", "..")) and (root / word).is_file()}
        pending.extend(sorted(name for name in files - found if name.endswith(".sh")))
        found |= files
    return found


def test_the_environments_key_names_every_file_its_install_script_runs_or_reads(tmp_path, monkeypatch):
    """armada rebuilds the environment where the files its key names change (.armada.json), so each file
    tools/armada/install.sh runs or reads is among them (row 3303: tools/fetched_fixtures.py, whose fetch
    placed release files elsewhere, the classifiers of pyproject.toml that set which Pythons get an
    environment, and the .python-version the repository's own is linked to), but ci.py, which install.sh runs
    for those classifiers (`pythons`): its code is the plan's, which no environment is rebuilt for."""
    environment = json.loads((ROOT / ".armada.json").read_text())["environment"]
    # What armada hashes: the files its globs match in the commit, and the recipe's setup and install scripts.
    keyed = {path.relative_to(ROOT).as_posix() for glob in environment["key"] for path in ROOT.glob(glob)}
    keyed |= {environment["setup"], environment["install"]}
    runs = install_inputs(ROOT, environment["install"])
    reader = "tools/armada/ci.py"
    assert reader in runs
    assert sorted(runs - keyed - {reader}) == []
    # The classifiers are what it prints: a version added to them is one more environment, so pyproject.toml,
    # which holds them, is keyed.
    before = copied_tools(tmp_path, monkeypatch).PYTHONS
    pyproject = tmp_path / "pyproject.toml"
    classifier = '"Programming Language :: Python :: 3.14",'
    assert classifier in pyproject.read_text()
    pyproject.write_text(pyproject.read_text().replace(
        classifier, classifier + '\n    "Programming Language :: Python :: 3.15",'))
    after = load("ci", "tools/armada", tmp_path).PYTHONS
    assert after == (*before, "3.15")
    assert "pyproject.toml" in keyed


def test_a_harness_edit_of_comments_and_docstrings_reruns_only_what_observed_the_edited_text(
        tmp_path, monkeypatch, cache_worker):
    """The recorder, session and plugins under tools/armada are in every key by their code, not their bytes,
    and the cache stays on for a change of them that is no code (row 2730: train 350's docstring edits to
    followed.py and child_groups.py turned the cache off and re-keyed every row). What a test observes of the
    harness is its own input all the same: a text it reads is its bytes, a function's `__code__` it takes its
    unit with its docstring. A change to code reruns every row and turns the cache off."""
    ci = copied_tools(tmp_path, monkeypatch)
    observers = {
        "plain": "",
        "reads": "    assert 'def read(' in Path('tools/armada/followed.py').read_text()\n",
        "code": "    import followed\n    assert followed.read.__code__.co_argcount == 3\n"}
    for name, body in observers.items():
        (tmp_path / f"tests/test_{name}.py").write_text(
            f"from pathlib import Path\ndef test_{name}():\n"
            f"    with Path('runs-{name}').open('a') as runs: runs.write('x')\n{body}")
    verdict = cached_checkout(ci, tmp_path, monkeypatch, cache_worker)
    base = commit(tmp_path, "base")
    names = tuple(observers)

    def runs(*more: str) -> list[int]:
        return green_runs(ci, monkeypatch, tmp_path, verdict, ci.policy(base), names)

    assert runs() == [1, 1, 1]
    assert runs() == [1, 1, 1]
    followed = tmp_path / "tools/armada/followed.py"
    # A comment, and a module's docstring in the recorder, the plugin, the session and the process group's.
    followed.write_text(followed.read_text() + "\n# a note on the recorder\n")
    for name in ("session", "child_groups", "lifetime", "executed"):
        path = tmp_path / f"tools/armada/{name}.py"
        path.write_text(path.read_text().replace('"""', '"""A note on it. ', 1))
    commit(tmp_path, "comments and docstrings")
    assert ci.policy(base) == "on"
    assert runs() == [1, 2, 1], "what read the text reruns, nothing else"
    # A docstring of the function whose code one file took.
    signature = "def read(event: str, args: tuple, root: str) -> str | None:\n    \"\"\""
    assert signature in followed.read_text()
    followed.write_text(followed.read_text().replace(signature, signature + "A note on reading. ", 1))
    commit(tmp_path, "a function's docstring")
    assert ci.policy(base) == "on"
    assert runs() == [1, 3, 2], "the text, and the code that took the function's"
    session = tmp_path / "tools/armada/session.py"
    session.write_text(session.read_text().replace("GRACE = 60.0", "GRACE = 61.0"))
    commit(tmp_path, "code")
    assert ci.policy(base) == "off"
    assert runs() == [2, 4, 3]


COUNTED = ("from pathlib import Path\ndef test_counted():\n"
           "    with Path('runs').open('a') as runs: runs.write('x')\n")


@pytest.mark.parametrize(("name", "value"), [
    ("PYTEST_ADDOPTS", "-k no_test_has_this_name"), ("PYTEST_PLUGINS", "no_such_plugin")])
def test_a_variable_that_steers_pytest_never_reaches_a_tasks_run(ci, tmp_path, monkeypatch, cache_worker,
                                                                name, value):
    """pytest's own switches for what it runs (`session.STEERING`) are the supervisor's: a task's pytest takes
    the selection and the plugins its command names, so an inherited `-k` sets no test aside, nor a plugin
    fail its start, and the row a run records is the run its key says (row 2558: a green row with nothing run
    was reused by every run after it)."""
    written(tmp_path, {"test_counted.py": COUNTED})
    run = cached_runs(ci, tmp_path, monkeypatch, cache_worker)
    monkeypatch.setenv(name, value)
    assert run() == 1
    monkeypatch.delenv(name)
    assert run() == 1, "its row is the run of the command alone"


def test_the_xla_and_jax_settings_a_task_inherits_are_part_of_its_rows_key(ci, tmp_path, monkeypatch,
                                                                            cache_worker):
    """What a test computes depends on XLA's flags and JAX's configuration, which its process inherits: a
    green row of one setting is no row of another (row 2558), on a CPU task as on a GPU lane. Where an XLA
    flag only says where XLA writes, or JAX where it keeps its compiled programs, no result differs, and the
    row stays."""
    written(tmp_path, {"test_counted.py": COUNTED})
    run = cached_runs(ci, tmp_path, monkeypatch, cache_worker)
    fast = "--xla_cpu_enable_fast_math="
    monkeypatch.setenv("XLA_FLAGS", f"{fast}false")
    assert [run(), run()] == [1, 1]
    monkeypatch.setenv("XLA_FLAGS", f"{fast}true")
    assert run() == 2
    monkeypatch.setenv("XLA_FLAGS", f"{fast}false")
    assert run() == 2, "the first setting's row is found again"
    monkeypatch.setenv("XLA_FLAGS", f"{fast}false --xla_dump_to={tmp_path / 'dump'}")
    assert run() == 2
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "programs"))
    assert run() == 2
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    assert run() == 3


def test_a_whole_runs_rows_of_a_file_with_cells_are_no_trains(ci, tmp_path, monkeypatch, cache_worker):
    """A whole run reads no record (its cells would never run on a train's row), and the rows it records
    serve a train but for a file that has whole_suite cells: its groups are cut from the tests selection
    leaves (split.py), so a whole run's group and a train's of one name are different tests. Four tests, the
    second a cell, in two groups: the whole run's are (a, b) and (c, d), a train's (a, c) and (d); a helper
    only c calls changes, and the train must run c though the whole run's group of a and b is fresh."""
    (tmp_path / "tests/helper.py").write_text("VALUE = 1\n")
    (tmp_path / "tests/test_mixed.py").write_text(
        "import pytest\nfrom pathlib import Path\n"
        "def note(name):\n    with Path(f'runs-{name}').open('a') as runs: runs.write('x')\n"
        "def test_a():\n    note('a')\n"
        "@pytest.mark.whole_suite\ndef test_b():\n    note('b')\n"
        "def test_c():\n    import helper\n    note('c')\n    assert helper.VALUE\n"
        "def test_d():\n    note('d')\n")
    (tmp_path / "tests/test_durations.json").write_text(
        json.dumps({f"tests/test_mixed.py::test_{name}": 150.0 for name in "abcd"}))
    verdict = cached_checkout(ci, tmp_path, monkeypatch, cache_worker)

    def counts() -> list[int]:
        return [len((tmp_path / f"runs-{name}").read_text()) if (tmp_path / f"runs-{name}").exists() else 0
                for name in "abcd"]

    monkeypatch.setenv("DEW_WHOLE_SUITE", "1")
    assert {entry["cache"] for entry in ci.plan(300.0, {})} == {"off"}
    rows = armada_task(ci, monkeypatch, verdict, "on")
    assert sorted(row["name"] for row in rows) == [
        "3.12:tests/test_mixed.py#1/2", "3.12:tests/test_mixed.py#2/2"]
    assert counts() == [1, 1, 1, 1]
    monkeypatch.delenv("DEW_WHOLE_SUITE")
    assert {entry["cache"] for entry in ci.plan(300.0, {})} == {"on"}
    (tmp_path / "tests/helper.py").write_text("VALUE = 2\n")
    rows = armada_task(ci, monkeypatch, verdict, "on")
    assert [row["exitCode"] for row in rows] == [0, 0]
    assert counts() == [2, 1, 2, 2], "a train's own groups run, each of its tests once, the cell set aside"
    armada_task(ci, monkeypatch, verdict, "on")
    assert counts() == [2, 1, 2, 2], "a train's rows serve the next"


def lane_run_counts(ci, root: Path, names: list[str], processes: int = 1) -> list[int]:
    """One GPU lane of `names` at `root` (`gpu_lane_at`'s machine), every row green: how many times each
    file has run so far."""
    out = root / "verdict.json"
    assert ci.gpu_lane("3.12", names, out, cache="on", processes=processes, deadline=600.0) == 0
    assert [row["exitCode"] for row in json.loads(out.read_text())["rows"]] == [0] * len(names)
    return [len(lane_runs(root, Path(name).stem)) for name in names]


def test_a_gpu_lane_keys_the_xla_and_jax_settings_its_node_gives_it(ci, tmp_path, monkeypatch, cache_worker):
    """A lane on a CUDA node inherits the node's settings, which change what its files compute, so one file
    green under one is no row under another on the same GPU class (row 2558); a flag that only says where
    XLA keeps its picks changes no result, and the row is reused."""
    url, _, _ = cache_worker
    names = gpu_lane_at(tmp_path, monkeypatch, url, 1)
    monkeypatch.setenv("JAX_PLATFORMS", "cuda,cpu")
    interpreter(tmp_path, "3.12")
    fetcher(tmp_path)
    picks = "--xla_gpu_per_fusion_autotune_cache_dir"
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=4")
    assert lane_run_counts(ci, tmp_path, names) == [1]
    assert lane_run_counts(ci, tmp_path, names) == [1]
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=0")
    assert lane_run_counts(ci, tmp_path, names) == [2]
    monkeypatch.setenv("XLA_FLAGS", f"--xla_gpu_autotune_level=0 {picks}=/node/picks")
    assert lane_run_counts(ci, tmp_path, names) == [2]
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    assert lane_run_counts(ci, tmp_path, names) == [3]
