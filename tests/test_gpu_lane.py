"""CI's GPU lane (tools/armada/ci.py `gpu-lane`): each test file in a process of its own, admitted on the
GPU memory its runs measured, and tools/armada/gpu_holds.py, what a lane needs to run a file."""

import functools
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import jax
import pytest
import tier_wire
import truths_ahead
from armada_support import LANE_TEST, cached, fetcher, gpu_driver, gpu_lane_at, interpreter, loaded
from tool_support import load


@pytest.fixture
def ci(tmp_path, monkeypatch):
    return loaded(tmp_path, monkeypatch)


def test_a_gpu_lane_fetches_the_pinned_fixtures_before_any_file_and_stops_where_one_is_due(
        ci, tmp_path, monkeypatch, cache_worker):
    """The tests read pinned fixtures offline (CLOSED_HUB), so the lane fetches them first, while the Hub is
    reachable; a pin whose upload is due stops the lane before any file runs."""
    url, _, _ = cache_worker
    cached(monkeypatch, url)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    gpu_driver(tmp_path, monkeypatch)
    interpreter(tmp_path, "3.12")
    (tmp_path / "tests/test_reads.py").write_text(
        "import time\nfrom pathlib import Path\n"
        "def test_reads(): Path('read').write_text(str(time.monotonic()))\n")
    out = tmp_path / "verdict.json"
    fetched = fetcher(tmp_path)
    assert ci.gpu_lane("3.12", ["tests/test_reads.py"], out, cache="on", processes=1, deadline=600.0) == 0
    assert float(fetched.read_text()) < float((tmp_path / "read").read_text())
    fetched.unlink()
    (tmp_path / "read").unlink()
    fetcher(tmp_path, due=True)
    with pytest.raises(subprocess.CalledProcessError):
        ci.gpu_lane("3.12", ["tests/test_reads.py"], out, cache="on", processes=1, deadline=600.0)
    assert fetched.is_file() and not (tmp_path / "read").exists()


def lane_machine(root: Path, monkeypatch, url: str, name: str, source: str) -> None:
    """A GPU lane's machine at `root` with the test cache at `url` (`gpu_lane_at`), its Python and
    fetcher, and test file `name` holding `source`."""
    gpu_lane_at(root, monkeypatch, url, 0)
    interpreter(root, "3.12")
    fetcher(root)
    (root / name).write_text(source)


def lane_rows(ci, root: Path, tests: list[str], marks: list[Path],
              **network: bool) -> list[tuple[str, int]]:
    """A lane's rows over `tests` at `root`, each name and exit, with the cache on, the files its tests
    mark that they ran (`marks`) cleared first."""
    for mark in marks:
        mark.unlink(missing_ok=True)
    out = root / "verdict.json"
    assert ci.gpu_lane("3.12", tests, out, cache="on", processes=1, deadline=600.0, **network) == 0
    return [(row["name"], row["exitCode"]) for row in json.loads(out.read_text())["rows"]]


def test_a_network_lane_runs_the_network_marked_tests_with_the_hub_open_and_records_them_apart(
        ci, tmp_path, monkeypatch, cache_worker):
    """The network lane runs each file's network-marked tests alone, which every other task sets aside,
    with the Hub open, and records them on a class of their own: the same file's CUDA lane, Hub closed,
    runs its other tests and is no reuse of the network lane's run, and a second network lane on the same
    inputs reuses its own (the maintainer, 2026-10-25: no CI run reached the published checkpoints' parity
    tests)."""
    lane_machine(tmp_path, monkeypatch, cache_worker[0], "tests/test_hub.py",
        "import os, pytest\nfrom pathlib import Path\n"
        "@pytest.mark.network\ndef test_reads_the_hub():\n"
        "    assert not {'HF_HUB_OFFLINE', 'HF_ENDPOINT'} & set(os.environ), 'the Hub is closed'\n"
        "    Path('hub').write_text('ran')\n"
        "def test_offline():\n"
        "    assert os.environ['HF_HUB_OFFLINE'] == '1'\n"
        "    Path('offline').write_text('ran')\n")
    hub, offline = tmp_path / "hub", tmp_path / "offline"
    lane = functools.partial(lane_rows, ci, tmp_path, ["tests/test_hub.py"], [hub, offline])
    assert lane(network=True) == [("3.12:tests/test_hub.py", 0)]
    assert hub.is_file() and not offline.exists()
    assert lane() == [("3.12:tests/test_hub.py", 0)]
    assert offline.is_file() and not hub.exists(), "the CUDA lane's run is its own"
    assert lane(network=True) == [("3.12:tests/test_hub.py", 0)]
    assert not hub.exists(), "the network lane reuses its own green run on the same inputs"


def test_a_gpu_lane_given_the_whole_gpu_hands_its_files_jaxs_default_share(ci, tmp_path, monkeypatch,
                                                                            cache_worker):
    """A job holding a whole GPU, to which a node sets no XLA_PYTHON_CLIENT_MEM_FRACTION (a GPU node's session
    gives one only to a part of a GPU), runs its lane: a file no run measured is given the lane's whole share,
    JAX's own default for a process (`WHOLE_GPU`), rather than the lane failing before any file runs (row
    2570)."""
    url, _, _ = cache_worker
    [name] = gpu_lane_at(tmp_path, monkeypatch, url, 1)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION")
    interpreter(tmp_path, "3.12")
    fetcher(tmp_path)
    out = tmp_path / "verdict.json"
    assert ci.gpu_lane("3.12", [name], out, cache="on", processes=1, deadline=600.0) == 0
    [[share, *_]] = lane_runs(tmp_path, Path(name).stem)
    assert float(share) == float(ci.WHOLE_GPU)


def test_a_lanes_prepare_runs_on_the_cpu_what_its_run_will_hold_and_records_nothing(
        ci, tmp_path, monkeypatch, cache_worker):
    """A lane's prepare (`gpu-lane --prepare`) runs every file of its part, on the CPU under the prepare's
    word (DEW_TIER_PREPARE, under which the suite's conftest has each test prepare what its run reads, one
    that prepares nothing skipped), with the store its environment names: a cut file's group prepares exactly
    the tests its run then holds, as both select and cut the file alike (a prepare that selected some tests
    alone would cut them into other groups, and a run would find a test its part never prepared); every file,
    so a site its run reaches is never unprepared (Main, 2026-11-17); and a prepare records no row, so no lane
    or task reuses it as a test's run. The fixtures are fetched first, as a run's are."""
    url, stored, _ = cache_worker
    gpu_lane_at(tmp_path, monkeypatch, url, 0)
    interpreter(tmp_path, "3.12")
    fetched = fetcher(tmp_path)
    monkeypatch.setenv(tier_wire.TRUTHS, str(tmp_path / "truths"))
    # The suite's conftest's side, which decides what a test prepares: here, a test not marked
    # `prepares` prepares nothing and is skipped.
    (tmp_path / "tests/conftest.py").write_text(
        "import os, pytest\n"
        "def pytest_configure(config):\n"
        "    config.addinivalue_line('markers', 'prepares: prepared on the CPU before its GPU run')\n"
        "def pytest_runtest_setup(item):\n"
        f"    if os.environ.get('{tier_wire.PREPARE}') == '1' and\\\n"
        "            item.get_closest_marker('prepares') is None:\n"
        "        pytest.skip('nothing to prepare')\n")
    (tmp_path / "tests/test_flows.py").write_text(
        "import os, pytest\n"
        "def said(name):\n"
        f"    kind = 'prepared' if os.environ.get('{tier_wire.PREPARE}') == '1' else 'ran'\n"
        "    with open(f'{kind}.log', 'a') as log:\n"
        "        log.write(' '.join([name, os.environ.get('JAX_PLATFORMS', '-'),\n"
        f"                            os.environ.get('{tier_wire.TRUTHS}', '-'),\n"
        f"                            os.environ.get('{tier_wire.UNIT}', '-'),\n"
        f"                            os.environ.get('{tier_wire.PREPARE_BY}') or '-']) + chr(10))\n"
        + "".join(f"{'@pytest.mark.prepares' + chr(10) if index % 2 else ''}"
                  f"def test_{index}():\n    said('test_{index}')\n" for index in range(6)))
    (tmp_path / "tests/test_plain.py").write_text(LANE_TEST)

    def said(kind: str) -> list[list[str]]:
        log = tmp_path / f"{kind}.log"
        lines = [line.split() for line in log.read_text().splitlines()] if log.is_file() else []
        log.unlink(missing_ok=True)
        return lines

    out, marked = tmp_path / "verdict.json", {"test_1", "test_3", "test_5"}
    # A recalled attempt's progress, naming a test it passed: a prepare sets nothing aside by it, and keeps
    # none.
    resume = tmp_path / "resume"
    resume.mkdir()
    (resume / "progress.jsonl").write_text(
        json.dumps({"node": "tests/test_flows.py::test_1", "seconds": 1.0}) + "\n")
    monkeypatch.setenv("ARMADA_RESUME", str(resume))
    # A cap the lane's own environment carries, expired: its prepare clears it.
    monkeypatch.setenv(tier_wire.PREPARE_BY, "1.0")
    prepared = {}
    for group in ("1/2", "2/2"):
        name = f"tests/test_flows.py#{group}"
        assert ci.gpu_prepared("3.12", [name, "tests/test_plain.py"], out, processes=2, deadline=600.0) == 0
        assert [(row["name"], row["exitCode"]) for row in json.loads(out.read_text())["rows"]] == [
            (f"3.12:{name}", 0), ("3.12:tests/test_plain.py", 0)]
        lines = said("prepared")
        # Each told the unit it prepares, as the lane names it, under which its programs' memory is filed;
        # no time cap, as its run reads every program it prepares (only a dry prepare's is capped).
        assert {tuple(line[1:]) for line in lines} == {("cpu", str(tmp_path / "truths"), name, "-")}
        prepared[group] = {line[0] for line in lines}
    assert fetched.is_file() and not lane_runs(tmp_path, "test_plain"), "one that prepares nothing is skipped"
    assert not stored, "a prepare records no row"
    assert not (tmp_path / "artifacts" / "progress.jsonl").exists(), "nor any test as passed (review)"
    monkeypatch.delenv("ARMADA_RESUME")
    assert prepared["1/2"] | prepared["2/2"] == marked and not prepared["1/2"] & prepared["2/2"]
    for group in ("1/2", "2/2"):
        assert ci.gpu_lane("3.12", [f"tests/test_flows.py#{group}"], out, cache="on", processes=1,
                           deadline=600.0) == 0
        assert {line[0] for line in said("ran")} & marked == prepared[group], group
    assert stored, "its run's rows are recorded as any lane's"


VAE_FILE = (
    "import os, pytest\nfrom pathlib import Path\n"
    "pytestmark = pytest.mark.gpu(reason='its kernels')\n"
    "def said(name):\n"
    "    with open('prepared.log', 'a') as log:\n"
    "        log.write(' '.join([name, os.environ['JAX_PLATFORMS'],\n"
    f"            os.environ['{tier_wire.PREPARE}'], Path(os.environ['{tier_wire.TOPOLOGY}']).read_text(),\n"
    f"            os.environ['{tier_wire.UNIT}'],\n"
    "            os.environ.get('DEW_TEST_USERS_ARITHMETIC', '-')]) + chr(10))\n"
    "@pytest.mark.crashes\ndef test_encodes():\n    said('test_encodes')\n"
    "def test_decodes():\n    said('test_decodes')\n"
    "@pytest.mark.users_process\ndef test_as_a_user():\n    said('test_as_a_user')\n")
"""A GPU tier file: a test whose prepare fails before its first check-1 site, one that prepares, and one of
a user's process, each saying the word, platform, GPU, unit and arithmetic it prepared under."""

UNPREPARING = (
    "import json, os, pytest\nfrom pathlib import Path\n"
    "def pytest_configure(config):\n"
    "    for mark in ('gpu', 'crashes', 'users_process'):\n"
    "        config.addinivalue_line('markers', mark)\n"
    "def pytest_runtest_setup(item):\n"
    "    users = os.environ.get('DEW_TEST_USERS_ARITHMETIC') == '1'\n"
    "    if users != (item.get_closest_marker('users_process') is not None):\n"
    "        pytest.skip('runs in the other process')\n"
    "def pytest_runtest_call(item):\n"
    f"    if os.environ.get('{tier_wire.PREPARE}') == '1' and item.get_closest_marker('crashes'):\n"
    f"        store = Path(os.environ['{tier_wire.TRUTHS}']) / 'sites' / item.name\n"
    "        store.mkdir(parents=True)\n"
    f"        (store / '{tier_wire.UNPREPARED}').write_text(json.dumps({{'test': item.nodeid, "
    "'where': 'tests/test_gpu_vae.py:9', 'error': 'lowered for the CPU', 'reasons': []}))\n"
    "        pytest.skip('computes before its first check-1 site')\n")
"""The suite's conftest's side, as tests/conftest.py has it: a GPU test runs in its own process's variant
alone, and a test that fails in its prepare before its first site is recorded unprepared and skipped."""


def dry_checkout(root: Path) -> None:
    """A checkout at `root` holding a dry prepare's recipe (prepares.py's `RECIPE`), its topology's writer
    (which writes the generation's name), the suite's side (`UNPREPARING`) and a GPU tier file
    (`VAE_FILE`)."""
    for name in ("tools/armada/prepares.py", "tests/tier_wire.py", "tools/armada/recycle.py"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("")
    (root / "tests/generation_support.py").write_text(
        "def node_topology(path, gpu):\n    path.write_text(gpu)\n    return path\n")
    (root / "tests/conftest.py").write_text(UNPREPARING)
    (root / "tests/test_gpu_vae.py").write_text(VAE_FILE)


def test_a_dry_prepare_runs_the_tiers_processes_for_one_compile_only_gpu_and_keeps_what_they_left_unprepared(
        ci, tmp_path, monkeypatch, cache_worker):
    """A dry prepare (`dry_prepared`) runs a GPU tier file's processes as the tier's part does (prepares.py's
    `process`): on the CPU under the prepare's word, for a node's topology of one GPU of its row's
    generation, each under its row's name as its unit, a user's process under a user's arithmetic, into a
    store of its own. Each row keeps the cells it left unprepared, is green where main has no run of it to
    hold it to (`BOOTSTRAP`), and is recorded as any row, so its next run on the same inputs reuses it,
    preparing nothing."""
    url, stored, _ = cache_worker
    cached(monkeypatch, url)
    monkeypatch.setattr(ci, "mainline", lambda: "f" * 40)
    interpreter(tmp_path, "3.12")
    dry_checkout(tmp_path)
    out, log = tmp_path / "verdict.json", tmp_path / "prepared.log"
    lane, users = "3.12:tests/test_gpu_vae.py#@prepare-sm90", "3.12:tests/test_gpu_vae.py#@prepare-sm90-users"
    encodes = "tests/test_gpu_vae.py::test_encodes @ tests/test_gpu_vae.py:9"
    expected = {"@prepare-sm90": [(lane, 0, [encodes])],
                "@prepare-sm90-users": [(users, 0, [])]}
    for split, rows in expected.items():
        assert ci.dry_prepared("3.12", ["prepare:tests/test_gpu_vae.py"], split, out, 600.0, "on") == 0
        assert [(row["name"], row["exitCode"], row["unprepared"])
                for row in json.loads(out.read_text())["rows"]] == rows
    assert stored
    assert sorted(log.read_text().splitlines()) == [
        f"test_as_a_user cpu 1 sm90 {users} 1", f"test_decodes cpu 1 sm90 {lane} -"]
    log.unlink()
    for split, rows in expected.items():
        assert ci.dry_prepared("3.12", ["prepare:tests/test_gpu_vae.py"], split, out, 600.0, "on") == 0
        assert [(row["name"], row["exitCode"], row["unprepared"])
                for row in json.loads(out.read_text())["rows"]] == rows
    assert not log.exists(), "each row was reused from its record"


def test_a_flows_deferred_cell_is_read_unprepared_and_a_row_left_no_time_is_stale(ci, tmp_path):
    """A dry prepare reads a flow's cell its prepare deferred to its run (tests/tier_wire.py's `DEFERRED`,
    which the run names unprepared) as unprepared, beside the tests recorded so (`UNPREPARED`), each with its
    first reason (prepares.py's `unprepared_in`); and a task with no time left for its rows says each stale,
    none started (prepares.py's `run`), the plan's next run of it its grade."""
    store = tmp_path / "sites"
    flow, encodes = "tests/test_gpu_vae.py::test_flow", "tests/test_gpu_vae.py::test_encodes"
    for name, test, record, cells in (("a", flow, ci.tier_wire().DEFERRED, ("one", "two")),
                                      ("b", encodes, ci.tier_wire().UNPREPARED, (None,))):
        for cell in cells:
            truths_ahead.reasoned(store / name / record, test, "tests/test_gpu_vae.py:9",
                                  f"{cell or 'a site'}: lowered", cell)
    found = ci.prepares.unprepared_in(tmp_path, (ci.tier_wire().UNPREPARED, ci.tier_wire().DEFERRED),
                                      ci.grading.owning)
    # The flow's two cells deferred at one line are each their own.
    assert sorted(ci.prepares.identity(record) for record in found["tests/test_gpu_vae.py"]) == [
        f"{encodes} @ tests/test_gpu_vae.py:9", f"{flow} @ tests/test_gpu_vae.py:9 [one]",
        f"{flow} @ tests/test_gpu_vae.py:9 [two]"]
    out = tmp_path / "verdict.json"
    assert ci.prepares.run(ci, "3.12", ["prepare:tests/test_gpu_vae.py"], "@prepare-sm90", out, 1.0,
                           "off") == 3
    [row] = json.loads(out.read_text())["rows"]
    assert (row["name"], row["kind"]) == ("3.12:tests/test_gpu_vae.py#@prepare-sm90", ci.grading.STALE)


def test_a_dry_prepares_row_is_red_for_each_cell_it_left_unprepared_that_mains_run_did_not(ci):
    """A dry prepare's row (prepares.py's `graded`) is red where it left a cell unprepared that main's run of
    the row did not (its test and where in it: one deferred cell of a test is no mask for a sibling), its
    new cells its `new_unprepared`, whether this run prepared it (the store's records) or it was reused
    (its record's kept list, read again against the baseline that stands now), red as its group as it ran;
    main with no run of it (`BOOTSTRAP`) reds none; main's not read whole (`UNREAD`) says a row that left
    any unprepared stale, never green, and one that left none green; a row red already stays red, and a
    stale one stays stale."""
    name = "3.12:tests/test_gpu_vae.py#@prepare-sm90"
    held = [{"test": "tests/test_gpu_vae.py::test_flow", "where": where, "error": "lowered"}
            for where in ("cell one", "cell two")]
    cells = [ci.prepares.identity(record) for record in held]

    def graded(before, code=0, reused=False, records=held, kind=None):
        found = {"name": name, "exitCode": code, "output": "", **({"kind": kind} if kind else {}),
                 **({"unprepared": [ci.prepares.identity(record) for record in records]} if reused else {})}
        [row] = ci.prepares.graded([found], {} if reused else {"tests/test_gpu_vae.py": records}, before,
                                   "sm90", lambda named: ci.grading.owning(ci.parsed_row(named)[1]),
                                   ci.grading.STALE, ci.tier_wire().CAPPED)
        return row
    for reused in (False, True):
        red = graded({cells[0]}, reused=reused)
        assert (red["exitCode"], red["kind"], red["failed"]) == (1, "group", [])
        assert (red["unprepared"], red["new_unprepared"]) == (sorted(cells), [cells[1]])
        greens = [graded(before, reused=reused)["exitCode"] for before in (set(cells), ci.prepares.BOOTSTRAP)]
        assert greens == [0, 0]
        unread = graded(ci.prepares.UNREAD, reused=reused)
        assert (unread["exitCode"], unread["kind"]) == (1, ci.grading.STALE)
        assert graded(ci.prepares.UNREAD, reused=reused, records=[])["exitCode"] == 0
    assert graded(set(cells), code=1)["exitCode"] == 1
    assert graded(set(cells), code=1, kind=ci.grading.STALE)["kind"] == ci.grading.STALE


def test_a_cell_its_prepares_cap_left_is_named_and_counted_but_reds_no_row(ci):
    """A cell a dry prepare's cap left unprepared (tests/tier_wire.py's `CAPPED`, its where exactly) is named
    among its row's unprepared and counted apart (`capped`, its output; kept as the row is reused), but no
    red of its row against main's run, nor stale where main's is unread (prepares.py's `graded`): its
    prepare's time, no fault of the test. A cell left otherwise stays red, and one whose node id merely
    holds the cap's words is no capped cell."""
    name = "3.12:tests/test_gpu_vae.py#@prepare-sm90"
    wire = ci.tier_wire()
    capped = {"test": "tests/test_gpu_vae.py::test_flow", "where": wire.CAPPED, "error": "capped",
              "cell": "two"}
    lowered = {"test": "tests/test_gpu_vae.py::test_flow", "where": "cell one", "error": "lowered"}
    named = {"test": f"tests/test_gpu_vae.py::test_flow[x @ {wire.CAPPED}]", "where": "a site", "error": "e"}

    def graded(records, before=frozenset(), found=None):
        found = found or {"name": name, "exitCode": 0, "output": "ran"}
        [row] = ci.prepares.graded(
            [found], {} if "unprepared" in found else {"tests/test_gpu_vae.py": records},
            before if isinstance(before, str) else set(before), "sm90",
            lambda named_: ci.grading.owning(ci.parsed_row(named_)[1]), ci.grading.STALE, wire.CAPPED)
        return row
    row = graded([capped])
    assert (row["exitCode"], row["unprepared"], row["capped"]) == (
        0, [ci.prepares.identity(capped)], [ci.prepares.identity(capped)])
    assert ci.prepares.identity(capped) in row["output"] and row["output"].endswith("ran")
    # Reused from its record (the row as it was rendered): its capped cells kept, still judged by neither,
    # its banner said once.
    reused = graded([], found=row)
    assert (reused["exitCode"], reused["capped"], reused["output"]) == (0, row["capped"], row["output"])
    # Its baseline not written: stale, its cap's fields and banner kept.
    [unwritten] = ci.unbaselined("3.12", [row], {name})
    assert (unwritten["kind"], unwritten["capped"], unwritten["unprepared"]) == (
        ci.grading.STALE, row["capped"], row["unprepared"])
    assert row["output"] in unwritten["output"]
    # Main's runs unread: a row that left only capped cells is no stale.
    assert graded([capped], before=ci.prepares.UNREAD).get("kind") != ci.grading.STALE
    assert graded([capped, lowered], before=ci.prepares.UNREAD)["kind"] == ci.grading.STALE
    red = graded([capped, lowered])
    assert (red["exitCode"], red["new_unprepared"]) == (1, [ci.prepares.identity(lowered)])
    assert graded([named])["new_unprepared"] == [ci.prepares.identity(named)]


@pytest.mark.parametrize("capped", [False, True])
def test_only_a_dry_prepare_runs_under_a_time_cap(ci, tmp_path, monkeypatch, capped):
    """A dry prepare's process (prepares.py's `process`, `capped`) runs within `PREPARE_WALL` whatever its
    task has left, so none sets a train's wall, under a cap (tests/tier_wire.py's `PREPARE_BY`, on the
    monotonic clock) its end's margin short of that, past which it starts nothing more; a GPU tier part's
    prepare (ci.py's `gpu_prepared`), whose run reads every program it prepares, has all its caller gives
    it and no cap."""
    import time

    seen = {}

    def tested(python, files, group, verdict, seconds, *args, **kwargs):
        seen["seconds"], seen["environ"] = seconds, args[3]
        return 0
    monkeypatch.setattr(ci, "tested", tested)
    began = time.monotonic()
    ci.prepares.process(ci, "3.12", "tests/test_gpu_vae.py", "", tmp_path / "v.json", 1500.0, unit="u",
                        threads="1", users=False, variant="@prepare-sm90", capped=capped)
    wire = ci.tier_wire()
    if not capped:
        # Cleared, so a cap the caller's environment carries reaches no part's prepare.
        assert seen["environ"][wire.PREPARE_BY] == "" and not wire.past_cap(seen["environ"])
        assert seen["seconds"] == 1500.0 - ci.session.GRACE - ci.SETTLE - 10.0
        return
    cap = float(seen["environ"][wire.PREPARE_BY])
    assert seen["seconds"] + ci.session.GRACE + ci.SETTLE + 10.0 == ci.prepares.PREPARE_WALL
    assert began + seen["seconds"] - ci.prepares.CAP_MARGIN <= cap <= time.monotonic() + seen["seconds"]


def cgroup_tree(root, levels):
    """A cgroup (v2) hierarchy under `root` this process is a member of, `levels` from the root down: each a
    memory limit (None: `max`), a charge and its inactive page cache, in bytes. Its membership's text."""
    group = root
    for index, (limit, current, inactive) in enumerate(levels):
        group = group / f"level{index}" if index else group
        group.mkdir(parents=True, exist_ok=True)
        (group / "memory.max").write_text("max\n" if limit is None else f"{limit}\n")
        (group / "memory.current").write_text(f"{current}\n")
        (group / "memory.stat").write_text(f"anon {current}\ninactive_file {inactive}\n")
    return "0::/" + "/".join(f"level{index}" for index in range(1, len(levels))) + "\n"


@pytest.mark.parametrize(("levels", "past"), [
    # The task's own cgroup unlimited, its parent's limit binding: its working set past the share.
    ([(None, 0, 0), (10_000, 9_000, 1_000), (None, 8_000, 0)], True),
    # Its charge past the share, but most of it page cache the kernel reclaims first.
    ([(None, 0, 0), (10_000, 9_000, 4_000), (None, 5_000, 0)], False),
    # Under every limit's share.
    ([(None, 0, 0), (10_000, 6_000, 0), (8_000, 5_000, 0)], False),
    # No level states a limit.
    ([(None, 0, 0), (None, 9_000, 0)], False),
])
def test_a_dry_prepare_is_past_its_memory_cap_where_a_limit_binding_its_task_is_mostly_held(
        ci, tmp_path, monkeypatch, levels, past):
    """A dry prepare's task (prepares.py's `process`) caps the working set of each cgroup level binding it,
    its own or an ancestor's under DEW_CGROUP_ROOT, at `HELD_SHARE` of that level's limit
    (tests/tier_wire.py's `PREPARE_HOLDS`), and its pytest is past its cap (`past_cap`) where any level holds
    more (its charge less its inactive page cache): every process under it counts, a flow's own beside the
    prepare's. The limits are read by the task, the pytest reading only what each level holds, so no limit
    keys its rows; a part's prepare runs under no such cap."""
    # A cgroup's name may hold the path separator the old wire split on.
    root = tmp_path / "worker:prepare"
    membership = tmp_path / "cgroup"
    membership.write_text(cgroup_tree(root, levels))
    reader = ci.host.cgroups()
    real_own = reader.own
    monkeypatch.setattr(reader, "own", lambda root: real_own(root, membership))
    monkeypatch.setenv("DEW_CGROUP_ROOT", str(root))
    seen = []

    def tested(python, files, group, verdict, seconds, *args, **kwargs):
        seen.append(args[3])
        return 0
    monkeypatch.setattr(ci, "tested", tested)
    for capped in (True, False):
        ci.prepares.process(ci, "3.12", "tests/test_gpu_vae.py", "", tmp_path / "v.json", 1500.0, unit="u",
                            threads="1", users=False, variant="@prepare-sm90", capped=capped)
    dry, part = (environ[tier_wire.PREPARE_HOLDS] for environ in seen)
    # The pytest reads no limit: with every `memory.max` gone it judges alike.
    for limit in root.rglob("memory.max"):
        limit.unlink()
    assert part == "" and tier_wire.past_cap({tier_wire.PREPARE_HOLDS: dry}) is past


def test_the_plan_prepares_each_gpu_tier_file_dry_for_each_generation(ci, tmp_path):
    """The plan's dry prepares (prepares.py's `entries`): each file holding a `gpu`-marked test (a mark in a
    string is none) for each generation of `PREPARED` (sm90 alone at first), its lane's process and, where it
    holds a test of a user's process, that process's, each row a task of its own whose tests name it as a
    prepare's (`TOKEN`), weighed by its row's last seconds or `PREPARE_SECONDS`."""
    dry_checkout(tmp_path)
    (tmp_path / "tests/test_gpu_vae.py").unlink()
    (tmp_path / "tests/test_gpu_a.py").write_text("import pytest\npytestmark = pytest.mark.gpu\n"
                                                  "def test_one():\n    pass\n")
    (tmp_path / "tests/test_b.py").write_text("import pytest\n@pytest.mark.gpu(reason='x')\n"
                                              "@pytest.mark.users_process\ndef test_one():\n    pass\n")
    (tmp_path / "tests/test_c.py").write_text("TEXT = 'pytest.mark.gpu'\ndef test_one():\n    pass\n")
    timed = "3.12:tests/test_gpu_a.py#@prepare-sm90"
    entries = ci.prepares.entries(ci, "3.12", {"rows": {timed: 41.0}}, 120.0, "on", {"cache": "on"})
    expected = {timed, "3.12:tests/test_b.py#@prepare-sm90", "3.12:tests/test_b.py#@prepare-sm90-users"}
    assert {entry["rows"][0]: (entry["tests"], entry["split"]) for entry in entries} == {
        row: (ci.prepares.TOKEN + row.partition(":")[2].partition("#")[0], row.partition("#")[2])
        for row in expected}
    assert {entry["rows"][0]: entry["weight"] for entry in entries} == {
        row: 41.0 if row == timed else ci.prepares.PREPARE_SECONDS for row in expected}
    assert all(re.fullmatch(r"[A-Za-z0-9._-]{1,80}", entry["name"]) for entry in entries)


def test_a_kept_prepare_rows_token_keeps_its_group_and_a_slow_prepare_is_cut_by_its_work(ci, tmp_path):
    """A kept dry prepare's token (`kept_token`, its group a process variant's, `@prepare-<gen>`) reads back
    its file, its whole group and where its green run is kept (`kept_of`, the locator after the last `@`). The
    plan's dry prepares (`entries`) cut a file by its work, at the plan's target of 120 s each group within
    260 s (`PREPARE_TARGETS` targets, under a dry prepare's cap, `cap_after` of `PREPARE_WALL`): the rows
    `measured_cut` sums (`measured_rows`), never an older cut's, and where the longest of them ran out of its
    time (past `CUT_SHORT` of the cap) twice what they took, a floor of its work (train 1247: seven groups cut
    to their whole deadline were killed at it, and timed so, would be cut alike again); never capped by a
    record's count of its tests. Before any run is timed, one group a `gpu`-marked test of the record."""
    token = ci.kept_token("tests/test_gpu_a.py", "1/2@prepare-sm90-users", "AuthenticAMD-25-17:feed")
    assert ci.kept_of([token]) == ([], {("tests/test_gpu_a.py", "1/2@prepare-sm90-users"):
                                        ("AuthenticAMD-25-17", "feed")})
    name = "tests/test_gpu_flows_x.py"
    (tmp_path / name).write_text(FLOWS_FILE.format(flows=6))
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {f"{name}::test_flow[{index}]": [0.0, None, 0] for index in range(6)}))

    def planned(rows: dict[str, float]) -> list[str]:
        found = ci.prepares.entries(ci, "3.12", {"rows": {f"3.12:{name}#{group}@prepare-sm90": seconds
                                                           for group, seconds in rows.items()}},
                                    120.0, "off", {})
        return [entry["split"] for entry in found if entry["tests"] == f"prepare:{name}"]
    # Timed at two cuts: the two groups' 400 s, two groups of the 260 s cap; the whole file's older 1,583 s,
    # which ran out of its time, doubles nothing.
    assert planned({"": 1583.0, "1/2": 200.0, "2/2": 200.0}) == ["1/2@prepare-sm90", "2/2@prepare-sm90"]
    # Two groups past the cap: 6,332 s of work at least, twenty-five groups of six tests' record.
    assert len(planned({"1/2": 1583.0, "2/2": 1583.0})) == 25
    assert planned({}) == [f"{index}/6@prepare-sm90" for index in range(1, 7)]
    text = ("import pytest\n@pytest.mark.gpu\ndef test_a(): pass\ndef test_cpu(): pass\n"
            "@pytest.mark.gpu(reason='x')\nclass TestB:\n"
            "    def test_b(self): pass\n    def test_c(self): pass\n"
            "@pytest.mark.parametrize('x', [pytest.param(1, marks=pytest.mark.gpu)])\ndef test_d(x): pass\n")
    assert ci.prepares.marked_tests(text, "gpu") == {"test_a", "TestB::test_b", "TestB::test_c", "test_d"}
    # Counted by the record's node ids, each parameter its own; its marked functions where it holds none.
    nodes = ["tests/x.py::test_a[1]", "tests/x.py::test_a[2]", "tests/x.py::test_cpu",
             "tests/x.py::TestB::test_b"]
    assert ci.prepares.gpu_tests("tests/x.py", text, nodes) == 3
    assert ci.prepares.gpu_tests("tests/x.py", text, []) == 4


FLOWS_FILE = (
    "import pytest\n"
    "def said(name):\n"
    "    with open('prepared.log', 'a') as log:\n"
    "        log.write(name + chr(10))\n"
    "def test_cpu():\n    said('test_cpu')\n"
    "@pytest.mark.gpu\n@pytest.mark.parametrize('index', range({flows}))\n"
    "def test_flow(index):\n    said(f'test_flow[{{index}}]')\n")
"""A GPU tier file of `flows` parametrized GPU flows and one CPU test, each saying it ran."""


def test_a_dry_prepares_groups_are_its_tests_by_count_in_its_task_and_its_key(
        ci, tmp_path, monkeypatch, cache_worker):
    """A dry prepare's groups (`1/3@prepare-sm90`) are its file's tests cut by their count (prepares.py's
    `prepared_cut`): in its task, whose split takes its cut from its row's variant alone
    (`session.SPLIT_EVENLY`), and in its key (`members`), so its row is reused for the tests it holds. Not by
    the durations record, which times its GPU tests at nothing on the CPU (train 1247: 18 flows cut 5, 13 and
    16 empty groups), nor by a stored cut of the file, which here packs four flows into one group. A file
    whose collection is not its record's tests (a parameter added, or removed) runs its groups as collected,
    each recorded with no key; a group its cut leaves no test is green at no seconds (`nothing_timed`)."""
    url, stored, _ = cache_worker
    cached(monkeypatch, url)
    monkeypatch.setattr(ci, "mainline", lambda: "f" * 40)
    interpreter(tmp_path, "3.12")
    dry_checkout(tmp_path)
    name = "tests/test_gpu_flows_x.py"
    (tmp_path / name).write_text(FLOWS_FILE.format(flows=6))
    nodes = [f"{name}::test_cpu", *(f"{name}::test_flow[{index}]" for index in range(6))]
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {node: [0.002 if node.endswith("[1]") else 0.0, None, 0] for node in nodes}))
    (tmp_path / "tests/test_cuts.json").write_text(json.dumps({name: [nodes[5], nodes[6]]}))
    out, log = tmp_path / "verdict.json", tmp_path / "prepared.log"

    def ran(group: str) -> tuple[list[str], dict]:
        log.unlink(missing_ok=True)
        assert ci.dry_prepared("3.12", [f"prepare:{name}"], f"{group}@prepare-sm90", out, 600.0, "on") == 0
        [row] = json.loads(out.read_text())["rows"]
        assert (row["name"], row["exitCode"]) == (f"3.12:{name}#{group}@prepare-sm90", 0)
        return sorted(log.read_text().split()) if log.exists() else [], row

    def record(group: str) -> dict:
        return json.loads(stored["/" + ci.record_key("3.12", f"3.12:{name}#{group}@prepare-sm90",
                                                     ci.run_keys.device_class())])

    assert [ran(f"{index}/3")[0] for index in range(1, 4)] == [
        ["test_cpu", "test_flow[0]"], ["test_flow[1]", "test_flow[2]"],
        ["test_flow[3]", "test_flow[4]", "test_flow[5]"]]
    assert [ci.run_keys.members(name, f"{index}/3", evenly=True) for index in range(1, 4)] == [
        nodes[:2], nodes[2:4], nodes[4:]]
    assert record("1/3")["key"] is not None
    # Its row's tests unchanged, it is reused.
    assert ran("1/3")[0] == []
    # A flow collected since, which no record names: its group runs it, its record keeps no key.
    (tmp_path / name).write_text(FLOWS_FILE.format(flows=7))
    assert ran("3/3")[0] == ["test_flow[4]", "test_flow[5]", "test_flow[6]"]
    assert record("3/3")["key"] is None
    # A flow gone since, which the record still names: its groups run as collected, keyed by none.
    (tmp_path / name).write_text(FLOWS_FILE.format(flows=5))
    assert ran("1/3")[0] == ["test_cpu", "test_flow[0]"]
    assert record("1/3")["key"] is None
    # Cut finer than it has tests, a group holds none: green at no seconds.
    found, row = ran("7/7")
    assert (found, row["seconds"], set(row["timings"].values())) == ([], 0.0, {0.0})


def test_a_dry_prepares_baseline_is_mains_whatever_cut_main_made_of_it(ci, monkeypatch):
    """A dry prepare's baseline (prepares.py's `before`) is every cell main's runs of its file's prepare for
    its generation and process left unprepared, in one cut main's commit names whole (listed once), so a
    row this run cut otherwise is held to main's all the same. Main with no run of it at all is
    `BOOTSTRAP`, as is a run of main's own commit (a train's base) that reads none whole, whose baseline it
    is; else a cut read in part, a Worker that fails, a listing or a read that stalls past its time, a run
    main named with no baseline, or a time already past, `UNREAD`."""
    kept = {"3.12:tests/test_gpu_a.py#1/2@prepare-sm90": ["tests/test_gpu_a.py::test_a @ cell"],
            "3.12:tests/test_gpu_a.py#2/2@prepare-sm90": ["tests/test_gpu_a.py::test_b @ cell"],
            "3.12:tests/test_gpu_a.py#1/3@prepare-sm90": ["tests/test_gpu_a.py::test_c @ cell"],
            "3.12:tests/test_gpu_a.py#1/2@prepare-sm80": ["tests/test_gpu_a.py::test_d @ cell"]}
    named: set[str] = set()
    stalls: dict[str, float] = {}
    failing: set[str] = set()
    shapes: dict[str, object] = {}

    def waited(where: str, until: float) -> None:
        """A stall at `where`, cut short at `until` as the Worker's request is (r2's `request`)."""
        stall = stalls.get(where, 0.0)
        if stall:
            time.sleep(max(0.0, min(stall, until - time.monotonic())))
            if time.monotonic() >= until:
                raise ci.r2.Unavailable(f"{where} timed out")

    class Store:
        def __init__(self, url: str, token: str, bucket: str, *, timeout: float, send) -> None:
            self.timeout = timeout

        def keys(self, prefix: str, until: float):
            waited("keys", until)
            if "keys" in failing:
                raise ci.r2.R2Error("the Worker failed")
            listed = kept if "-baseline/" in prefix else named
            for row_name in listed if prefix.endswith("tests/test_gpu_a.py/") else []:
                yield ci.row_place(row_name, "cls").rpartition("/")[2]

        def get(self, key: str, limit: int, until: float) -> bytes | None:
            waited("get", until)
            kind, _, place = key.partition("/")
            if place not in kept:
                return None
            found = shapes.get(kind, {"unprepared": kept[place]})
            return json.dumps(found).encode()

    monkeypatch.setenv("DEW_XLA_CACHE_URL", "http://cache")
    monkeypatch.setenv("DEW_XLA_CACHE_TOKEN", "token")
    monkeypatch.setattr(ci.r2, "Store", Store)
    monkeypatch.setattr(ci, "mainline", lambda: "main")
    monkeypatch.setattr(ci.run_keys, "device_class", lambda: "cls")
    monkeypatch.setattr(ci, "baseline_key", lambda commit, python, place, held: f"baseline/{place}")

    def before(name: str, group: str, seconds: float = 600.0):
        return ci.prepares.before(ci, "3.12", name, group, time.monotonic() + seconds)

    # A run of main's own commit (a train's base) reading no whole baseline of it is its baseline: a cut
    # left part-written (its other group red) holds it to nothing, never stale. One reading a whole
    # baseline is held to it as its heads are.
    monkeypatch.setattr(ci.run_keys, "head_commit", lambda: "main")
    named.add("3.12:tests/test_gpu_a.py#1/3@prepare-sm90-users")
    assert before("tests/test_gpu_a.py", "1/3@prepare-sm90-users") == ci.prepares.BOOTSTRAP
    named.clear()
    assert before("tests/test_gpu_a.py", "@prepare-sm90") == {
        "tests/test_gpu_a.py::test_a @ cell", "tests/test_gpu_a.py::test_b @ cell"}
    # One whose read of a whole baseline fails is no run that read none whole: stale, as a head's.
    failing.add("keys")
    assert before("tests/test_gpu_a.py", "@prepare-sm90") == ci.prepares.UNREAD
    failing.clear()
    shapes["baseline"] = {}
    assert before("tests/test_gpu_a.py", "@prepare-sm90") == ci.prepares.UNREAD
    shapes.clear()
    monkeypatch.setattr(ci.run_keys, "head_commit", lambda: "a head")

    assert before("tests/test_gpu_a.py", "@prepare-sm90") == {
        "tests/test_gpu_a.py::test_a @ cell", "tests/test_gpu_a.py::test_b @ cell"}
    assert before("tests/test_gpu_a.py", "1/3@prepare-sm90-users") == ci.prepares.BOOTSTRAP
    assert before("tests/test_gpu_b.py", "@prepare-sm90") == ci.prepares.BOOTSTRAP
    # A run main named whose baseline is not kept (its write lost) is no first run.
    named.add("3.12:tests/test_gpu_a.py#1/3@prepare-sm90-users")
    assert before("tests/test_gpu_a.py", "1/3@prepare-sm90-users") == ci.prepares.UNREAD
    named.clear()
    assert before("tests/test_gpu_a.py", "@prepare-sm90", seconds=-1) == ci.prepares.UNREAD
    failing.add("keys")
    assert before("tests/test_gpu_a.py", "@prepare-sm90") == ci.prepares.UNREAD
    failing.clear()
    # A stall inside the lookup (a page of its listing, a record's read) past its time is no answer, and the
    # lookup ends at its time, not the stall's.
    for where in ("keys", "get"):
        stalls[where] = 5.0
        began = time.monotonic()
        assert before("tests/test_gpu_a.py", "@prepare-sm90", seconds=0.2) == ci.prepares.UNREAD
        assert time.monotonic() - began < 1.0
        stalls.clear()
    # A malformed baseline (its cells a string, none, a record no object) is no answer either.
    for kind, shape in (("baseline", {"unprepared": "a cell"}), ("baseline", {}),
                        ("baseline", ["no", "record"])):
        shapes[kind] = shape
        assert before("tests/test_gpu_a.py", "@prepare-sm90") == ci.prepares.UNREAD, (kind, shape)
        shapes.clear()
    # A cut one of whose groups is unread is none (1/3 alone of three); 2/2 unread, the 1/2 cut is none too.
    del kept["3.12:tests/test_gpu_a.py#2/2@prepare-sm90"]
    assert before("tests/test_gpu_a.py", "@prepare-sm90") == ci.prepares.UNREAD


def test_a_task_takes_mains_commit_from_its_plan_its_checkout_holding_no_origin_main(ci, monkeypatch,
                                                                                       tmp_path):
    """A task's checkout on armada holds no origin/main, so its own merge-base names no main: main's commit is
    its plan's, given as --main, which its dry prepare's baselines are listed at (prepares.py's `before`), for
    that task alone; an empty --main leaves the checkout's own, and one that is no full commit id names none,
    so no baseline is read at it nor its absence taken for main's first run. With no --main, the commit its
    caller's environment gives (the land queue's regrades) stands."""
    listed: list[str] = []

    class Store:
        def __init__(self, url: str, token: str, bucket: str, *, timeout: float, send) -> None:
            pass

        def keys(self, prefix: str, until: float):
            listed.append(prefix)
            return iter(())

    def dry_prepared(python, tests, split, out, deadline, cache):
        until = time.monotonic() + 5
        seen.append(ci.prepares.before(ci, python, "tests/test_gpu_a.py", "@prepare-sm90", until))
        return 0

    monkeypatch.setenv("DEW_XLA_CACHE_URL", "http://cache")
    monkeypatch.setenv("DEW_XLA_CACHE_TOKEN", "token")
    monkeypatch.delenv(ci.MAINLINE, raising=False)
    monkeypatch.setattr(ci.r2, "Store", Store)
    monkeypatch.setattr(ci.run_keys, "device_class", lambda network=False: "cls")
    monkeypatch.setattr(ci.records, "merge_base", lambda: None)
    monkeypatch.setattr(ci, "dry_prepared", dry_prepared)
    seen: list[object] = []
    def run(*main: str) -> None:
        monkeypatch.setattr(sys, "argv", ["ci.py", "task", "--python=3.12",
                                          "--tests=prepare:tests/test_gpu_a.py",
                                          f"--out={tmp_path / 'rows.json'}", "--cache=on", *main])
        assert ci.main() == 0

    for given in ("a" * 40, "", "main"):
        run(f"--main={given}")
        assert ci.MAINLINE not in os.environ
    # A regrade the land queue runs gives it by its environment alone, kept where no --main is named.
    monkeypatch.setenv(ci.MAINLINE, "b" * 40)
    run()
    assert os.environ[ci.MAINLINE] == "b" * 40
    assert seen == [ci.prepares.BOOTSTRAP, ci.prepares.UNREAD, ci.prepares.UNREAD, ci.prepares.BOOTSTRAP]
    assert [prefix.split("-baseline/")[1].split("/")[0] for prefix in listed if "-baseline/" in prefix] == [
        "a" * 40, "b" * 40]


def test_a_commits_baseline_is_its_green_prepares_and_a_red_run_writes_none(ci, monkeypatch, cache_worker):
    """A dry prepare's green row keeps the cells it left unprepared as the commit's baseline (`baselined`,
    `baseline_key`); a red run of the same inputs, which writes over the green record, writes no baseline,
    so main's stays readable; a row that is no dry prepare has none due."""
    url, stored, _ = cache_worker
    monkeypatch.setenv("DEW_XLA_CACHE_URL", url)
    monkeypatch.setattr(ci.run_keys, "head_commit", lambda: "main")
    place = "3.12:tests/test_gpu_a.py#1/2@prepare-sm90"
    baseline = "/" + ci.baseline_key("main", "3.12", place, "cls")
    assert ci.run_keys.baselined("3.12", place, "cls", "token", {"exitCode": 1, "unprepared": ["a @ x"]})
    assert baseline not in stored
    assert ci.run_keys.baselined("3.12", place, "cls", "token", {"exitCode": 0, "unprepared": ["a @ x"]})
    assert json.loads(stored[baseline]) == {"unprepared": ["a @ x"]}
    assert ci.run_keys.baselined("3.12", place, "cls", "token", {"exitCode": 0})
    assert json.loads(stored[baseline]) == {"unprepared": ["a @ x"]}


@pytest.mark.parametrize("taken", [True, False])
def test_a_green_dry_row_keeps_its_baseline_whatever_its_record_does(ci, monkeypatch, cache_worker, tmp_path,
                                                                      taken):
    """A green dry prepare's baseline is written whatever of its record is kept: a row its record skips (here
    no log of its file's run, so no record nor key, as where the record's write fails or its epoch is
    unread) still leaves the commit its baseline. One the store did not take is returned unnamed, its row
    stale (`unbaselined`), never green with no baseline a later prepare is held to."""
    url, stored, _ = cache_worker
    monkeypatch.setenv("DEW_XLA_CACHE_URL", url)
    monkeypatch.setenv("DEW_XLA_CACHE_TOKEN", "token")
    monkeypatch.setattr(ci.run_keys, "head_commit", lambda: "main")
    monkeypatch.setattr(ci.run_keys, "device_class", lambda network=False: "cls")
    writes = ci.written
    monkeypatch.setattr(ci.run_keys, "written", lambda key, *args, **kwargs: (
        taken or "-baseline/" not in key) and writes(key, *args, **kwargs))
    place = "3.12:tests/test_gpu_a.py#1/2@prepare-sm90"
    log = tmp_path / "lifetime.log"
    log.write_text("")
    green = [{"name": place, "exitCode": 0, "unprepared": ["a @ x"]}]
    unnamed = ci.run_keys.record("3.12", green, log, "common")
    assert (unnamed, "/" + ci.baseline_key("main", "3.12", place, "cls") in stored) == (
        (set(), True) if taken else ({place}, False))


@pytest.mark.parametrize(("exit_code", "taken"), [(0, True), (1, True), (0, False)])
def test_a_reused_row_is_named_the_commits_only_as_its_grade_leaves_it_green(ci, monkeypatch, tmp_path,
                                                                          exit_code, taken):
    """A row reused from its record is named the commit's (`named_at_commit`) only after it is graded, where
    its grade leaves it green and its baseline is written (`baselined`): a reuse graded red publishes
    nothing, and a green one whose baseline the store did not take is named nowhere and stands stale."""
    row = {"name": "3.12:tests/test_gpu_a.py#@prepare-sm90", "exitCode": 0, "unprepared": ["a @ x"]}
    calls = []

    def reused_rows(python, tests, group, common, network, kept, named, reading):
        named[row["name"]] = ("cls", "key")
        return [{**row, "cached": "key"}]

    monkeypatch.setenv("DEW_XLA_CACHE_TOKEN", "token")
    monkeypatch.setattr(ci, "reused_rows", reused_rows)
    monkeypatch.setattr(ci, "row", lambda python, name, group: row["name"])
    monkeypatch.setattr(ci.run_keys, "baselined", lambda *args: taken)
    monkeypatch.setattr(ci.run_keys, "named_at_commit", lambda *args: calls.append(args))
    ci.tested("3.12", ["tests/test_gpu_a.py"], "", tmp_path / "rows.json", 60.0, "on", 5.0, "common", {},
              judged=lambda rows: [{**found, "exitCode": exit_code} for found in rows])
    named = [(row["name"], "cls", "key")] if exit_code == 0 and taken else []
    assert [call[1:4] for call in calls] == named
    [written] = json.loads((tmp_path / "rows.json").read_text())["rows"]
    assert (written["exitCode"], written.get("kind")) == ((0, None) if exit_code == 0 and taken else
                                                          (1, None) if exit_code else (1, ci.grading.STALE))


def test_a_file_that_does_not_collect_is_its_own_red_row_and_the_lanes_other_files_run(
        ci, tmp_path, monkeypatch, cache_worker):
    """A lane reads which of its files hold tests for another process by collecting them: a file that does
    not import is red in its own row, as every lane's is, and the lane's other files run and are reported
    (review: the lane's reading of its files raised on the first that did not collect, and no file ran, no
    row written). A lane given no file collects nothing, where pytest with no path collects the checkout."""
    url, _, _ = cache_worker
    gpu_lane_at(tmp_path, monkeypatch, url, 0)
    interpreter(tmp_path, "3.12")
    fetcher(tmp_path)
    (tmp_path / "tests/test_broken.py").write_text("import a_module_no_environment_holds\n")
    (tmp_path / "tests/test_whole.py").write_text(LANE_TEST)
    out = tmp_path / "verdict.json"
    assert ci.gpu_lane("3.12", ["tests/test_broken.py", "tests/test_whole.py"], out, cache="on", processes=2,
                       deadline=600.0) == 0
    rows = {row["name"]: row["exitCode"] for row in json.loads(out.read_text())["rows"]}
    assert rows == {"3.12:tests/test_broken.py": 1, "3.12:tests/test_whole.py": 0}
    assert ci.collected("3.12", "users_process", []) == {}
    # A reading that was interrupted is no reading: the lane stops, where one that went on would run its files
    # with no second process and leave green rows of tests that ran nowhere (review).
    (tmp_path / "tests/conftest.py").write_text(
        "import pytest\n"
        "def pytest_collection_modifyitems(config, items):\n"
        "    if config.option.markexpr == 'users_process':\n"
        "        pytest.exit('interrupted', returncode=2)\n")
    with pytest.raises(SystemExit):
        ci.gpu_lane("3.12", ["tests/test_whole.py"], out, cache="on", processes=1, deadline=600.0)


def test_a_lanes_prepare_ends_within_its_one_deadline_however_many_files_it_holds(
        ci, tmp_path, monkeypatch, cache_worker):
    """A lane's prepare is given one deadline for all of it: each file's process has what is left of it as
    its turn comes (its own end's allowance kept back), so more files than processes cannot outlast it, and
    a file it leaves too little is a red row of its own, the files before it prepared (a deadline a process
    let a part's files run in waves past its job's prepare bound, and the node's session ended the job
    whole)."""
    url, _, _ = cache_worker
    gpu_lane_at(tmp_path, monkeypatch, url, 0)
    interpreter(tmp_path, "3.12")
    fetcher(tmp_path)
    (tmp_path / "tests/conftest.py").write_text(
        "def pytest_configure(config):\n"
        "    config.addinivalue_line('markers', 'prepares: prepared on the CPU before its GPU run')\n")
    names = [f"tests/test_slow_{index}.py" for index in range(4)]
    for name, seconds in zip(names, (2, 2, 25, 0), strict=True):
        (tmp_path / name).write_text(
            "import pytest, time\nfrom pathlib import Path\n@pytest.mark.prepares\n"
            f"def test_prepares():\n    Path('started-{Path(name).stem}').write_text('')\n"
            f"    time.sleep({seconds})\n")
    # One process at a time, and twenty seconds for all four beside the allowance a process keeps back for
    # its own end: the first two prepare; the third, which needs more than is left, is cut at what is left,
    # where a deadline of its own would have let it end; and the fourth, with none left, is never started.
    out = tmp_path / "verdict.json"
    # A unit left unprepared exits the prepare 1.
    assert ci.gpu_prepared("3.12", names, out, processes=1, deadline=ci.session.GRACE + ci.SETTLE + 30.0) == 1
    rows = {row["name"]: row["exitCode"] for row in json.loads(out.read_text())["rows"]}
    assert rows == {f"3.12:{name}": code for name, code in zip(names, (0, 0, 1, 1), strict=True)}
    assert sorted(path.name for path in tmp_path.glob("started-*")) == [
        "started-test_slow_0", "started-test_slow_1", "started-test_slow_2"]


def test_a_file_with_tests_for_a_users_process_runs_them_in_a_process_of_their_own(
        ci, tmp_path, monkeypatch, cache_worker):
    """XLA reads a process's flags once, so a file's tests that must open under a user's flags (the mark
    `users_process`) run in a second process of the lane, under its word, which the suite's conftest reads
    to run those tests and skip the file's others, as without it it skips them: every GPU test of the file
    runs in exactly one process, a cut file's group as the lane's own process cuts it, and the second
    process's rows and records are its own (`<file>#G/N@users`), so neither stands for the other. A file
    with no such test starts no second process; one that holds some never runs without it, in its prepare
    as in its run."""
    url, _, _ = cache_worker
    gpu_lane_at(tmp_path, monkeypatch, url, 0)
    interpreter(tmp_path, "3.12")
    fetcher(tmp_path)
    # The harness's side, as the suite's own conftest has it.
    (tmp_path / "tests/conftest.py").write_text(
        "import os, pytest\n"
        "def pytest_configure(config):\n"
        "    for mark in ('users_process', 'prepares'):\n"
        "        config.addinivalue_line('markers', mark + ': a lane mark')\n"
        "def pytest_runtest_setup(item):\n"
        "    users = os.environ.get('DEW_TEST_USERS_ARITHMETIC') == '1'\n"
        "    if users != (item.get_closest_marker('users_process') is not None):\n"
        "        pytest.skip('another process of the lane runs it')\n"
        f"    if os.environ.get('{tier_wire.PREPARE}') == '1' and\\\n"
        "            item.get_closest_marker('prepares') is None:\n"
        "        pytest.skip('nothing to prepare')\n")
    (tmp_path / "tests/test_flows.py").write_text(
        "import os, pytest\n"
        "from pathlib import Path\n"
        "def said(name):\n"
        f"    kind = 'prepared' if os.environ.get('{tier_wire.PREPARE}') == '1' else 'ran'\n"
        # A file a run of its own, written whole: nothing a later run reads, so no record's input.
        "    users = os.environ.get('DEW_TEST_USERS_ARITHMETIC', '0')\n"
        "    Path(f'flow-{kind}-{name}-{os.getpid()}').write_text(users)\n"
        + "".join(f"@pytest.mark.prepares\n{'@pytest.mark.users_process' + chr(10) if index % 3 == 0 else ''}"
                  f"def test_{index}():\n    said('test_{index}')\n" for index in range(6)))
    (tmp_path / "tests/test_plain.py").write_text(LANE_TEST)

    def said(kind: str) -> list[tuple[str, str]]:
        """Each test a `kind` of process ran since the last look, and the word it ran under."""
        found = []
        for path in sorted(tmp_path.glob(f"flow-{kind}-*")):
            found.append((path.name.split("-")[2], path.read_text()))
            path.unlink()
        return sorted(found)

    every = sorted((f"test_{index}", "1" if index % 3 == 0 else "0") for index in range(6))
    out, groups = tmp_path / "verdict.json", ["tests/test_flows.py#1/2", "tests/test_flows.py#2/2"]
    assert ci.gpu_prepared("3.12", [*groups, "tests/test_plain.py"], out, processes=2, deadline=600.0) == 0
    assert said("prepared") == every, "each test prepared once, under the flags its process runs it under"
    assert ci.gpu_lane("3.12", [*groups, "tests/test_plain.py"], out, cache="on", processes=2,
                       deadline=600.0) == 0
    assert said("ran") == every, "each test run once, the marked ones in the users' process"
    rows = {row["name"]: row["exitCode"] for row in json.loads(out.read_text())["rows"]}
    assert rows == {"3.12:tests/test_flows.py#1/2": 0, "3.12:tests/test_flows.py#1/2@users": 0,
                    "3.12:tests/test_flows.py#2/2": 0, "3.12:tests/test_flows.py#2/2@users": 0,
                    "3.12:tests/test_plain.py": 0}
    # Its records are its own: the file whole (a group of a file no durations record names keeps no key), a
    # later lane reuses each process's row apart and runs none again.
    whole = ["tests/test_flows.py"]
    assert ci.gpu_lane("3.12", whole, out, cache="on", processes=2, deadline=600.0) == 0
    assert said("ran") == every
    assert ci.gpu_lane("3.12", whole, out, cache="on", processes=2, deadline=600.0) == 0
    reused = {row["name"]: row["exitCode"] for row in json.loads(out.read_text())["rows"]}
    assert said("ran") == []
    assert reused == {"3.12:tests/test_flows.py": 0, "3.12:tests/test_flows.py#@users": 0}


def test_a_gpu_lane_names_its_files_but_with_network(ci, monkeypatch):
    """`gpu-lane` without --tests is the network lane's alone, which runs every file holding a network-marked
    test: a CUDA lane names its files, and one that names none is refused, not run over the network files'
    offline tests (train 137's review)."""
    monkeypatch.setattr(sys, "argv", ["ci.py", "gpu-lane", "--out=rows.json", "--cache=on", "--processes=1"])
    with pytest.raises(SystemExit) as refused:
        ci.main()
    assert refused.value.code == 2


def lane_runs(root: Path, stem: str) -> list[list[str]]:
    """Each run of `stem`'s `LANE_TEST`, oldest first: its share, start, end and growth."""
    return [path.read_text().split() for path in sorted(root.glob(f"ran-{stem}-*"),
                                                        key=lambda path: path.stat().st_mtime)]


def overlapping(runs: list[list[str]]) -> int:
    """The most of `runs` (`lane_runs`) that ran at one moment."""
    spans = [(float(began), float(ended)) for _, began, ended, _ in runs]
    return max(sum(began <= moment < ended for began, ended in spans) for moment, _ in spans)


def test_a_gpu_lane_runs_each_file_in_a_process_of_its_own_within_its_memory_share(
        ci, tmp_path, monkeypatch, cache_worker):
    """The GPU lane runs each file once, in a process of its own. A file no
    run has measured runs alone with the lane's whole share, and so does one
    that started a process it did not follow, its pool growing as it uses it
    so that child has the rest of the share; files with a measured peak run
    side by side, each with its peak and headroom, and one whose child took
    its share's fraction holds a share more beside them (`gpu_needs`). A red
    row stays red: nothing is run again. The rows join one verdict, each
    recorded as a task's are, and a later lane reuses the green ones."""
    url, stored, _ = cache_worker
    names, source = gpu_lane_at(tmp_path, monkeypatch, url, 5), LANE_TEST
    # One that fails on any share, and one that starts a process.
    (tmp_path / "tests/test_red.py").write_text(source + "    assert share == 'all of it'\n")
    (tmp_path / "tests/test_spawner.py").write_text(
        source + "    subprocess.run([sys.executable, '-c', 'pass'], check=True)\n")
    every = [*names, "tests/test_red.py", "tests/test_spawner.py"]
    interpreter(tmp_path, "3.12")
    fetcher(tmp_path)

    def ran(stem: str) -> list[list[str]]:
        return lane_runs(tmp_path, stem)

    out = tmp_path / "verdict.json"
    assert ci.gpu_lane("3.12", every, out, cache="on", processes=4, deadline=600.0) == 0
    rows = json.loads(out.read_text())["rows"]
    assert [(row["name"], row["exitCode"]) for row in rows] == [
        (f"3.12:{name}", int(name == "tests/test_red.py")) for name in every]
    first = [run for name in every for run in ran(Path(name).stem)]
    assert [share for share, *_ in first] == ["0.5"] * len(every), "unmeasured: each alone, the whole share"
    # Each file's torch and OpenMP take its share of this process's CPUs, as four files may run at once.
    threads = str(max(1, len(os.sched_getaffinity(0)) // 4))
    assert {(tmp_path / f"threads-{Path(name).stem}").read_text() for name in every} == {threads}
    # Alone with the whole share, a file grows its pool as it uses it: a child it did not follow, another
    # environment's Python, has what the file leaves of the share.
    assert [grown for *_, grown in first] == ["false"] * len(every)
    assert overlapping(first) == 1
    assert len(ran("test_red")) == 1, "a red row stays red: its file is not run again"
    # Measured peaks of 4 GiB: each file then holds 6 GiB of the 40, and four run at once, each given its
    # 6 GiB as a fraction of the memory the GPU gives a process, which is what its pool is taken of, not of
    # the GPU's whole memory, some of which its driver keeps. Records without
    # a key, as a run on these inputs leaves to be run again, under the last run's and these inputs'.
    # The spawner's child took its fraction, so it holds two shares: 12 GiB beside the others' 24.
    common = ci.lane_inputs("3.12")
    for name, inputs in itertools.product([*names, "tests/test_spawner.py"], (None, common)):
        at = "/" + ci.record_key("3.12", ci.row("3.12", name, ""), ci.run_keys.device_class(), inputs)
        stored[at] = json.dumps({"key": None, "depends": [name], "row": {}, "reads": [], "peak": 4 * 2 ** 30,
                                 "shares": int(name == "tests/test_spawner.py"), "pools": 0}).encode()
    # Their first green runs gone, as the bucket's expiry takes them, so the lane runs them again by these
    # measures rather than reusing those runs of exactly these inputs (a row keeps each green run's set).
    for at in [at for at in stored if "-sets/" in at or "-green/" in at]:
        del stored[at]
    assert ci.gpu_lane("3.12", every, out, cache="on", processes=4, deadline=600.0) == 0
    rows = json.loads(out.read_text())["rows"]
    assert [row["exitCode"] for row in rows] == [int(name == "tests/test_red.py") for name in every]
    measured = [ran(Path(name).stem)[-1] for name in names]
    assert [share for share, *_ in measured] == [str(6 * 2 ** 30 / (81920 * 2 ** 20))] * len(names)
    assert [grown for *_, grown in measured] == ["true"] * len(names), "a measured share is taken whole"
    assert 2 <= overlapping(measured) <= 4
    assert [share for share, *_ in ran("test_spawner")] == ["0.5", measured[0][0]]
    assert overlapping([*measured, ran("test_spawner")[-1]]) >= 2, "a followed spawner packs with the rest"
    gib, budget = 2 ** 30, 40 * 2 ** 30
    assert ci.gpu_needs({"depends": None, "peak": 4 * gib}, budget) == (budget, budget)
    held = {"depends": [], "peak": 4 * gib, "shares": 2, "pools": gib}
    assert ci.gpu_needs(held, budget) == (6 * gib, 19 * gib), "6 GiB, twice more for its children, and 1 GiB"
    assert ci.gpu_lane("3.12", [], out, cache="on", processes=4, deadline=600.0) == 0
    assert json.loads(out.read_text()) == {"rows": []}, "an empty lane runs nothing"
    assert len(ran("test_red")) == 2, "a red row is never reused"
    assert ci.gpu_lane("3.12", names, out, cache="on", processes=4, deadline=600.0) == 0
    assert all(len(ran(Path(name).stem)) == 2 for name in names), "the third lane reused the green rows"
    # Their records, of these inputs as the lane keys them, read no backend: none is in the GPU tier.
    assert not set(names) & set(ci.gpu_tier("3.12"))


def test_a_first_gpu_lane_on_a_class_borrows_another_classs_measured_shares(
        ci, tmp_path, monkeypatch, cache_worker):
    """A lane on a class no run has measured borrows each file's share from another GPU class's last
    measured peak (the largest any class holds), so its files run side by side as a measured lane's do,
    not each alone. A file red under a borrowed share is red, run once, whatever its failure: its memory
    running out as much as an assertion (owner, 2026-11-07: a red under sharing is a red). The lane
    records this class's own measurement, so the next lane on it reads its own, not the borrowed one. A
    file no class measured runs alone, as every unmeasured one does where the Worker does not list its
    records."""
    url, stored, _ = cache_worker
    names, source = gpu_lane_at(tmp_path, monkeypatch, url, 4), LANE_TEST
    # Out of memory on any share but the whole, and an assertion's miss on any: each red, run once. And
    # one out of memory whose borrowed measurement already reached the whole share, run with it.
    (tmp_path / "tests/test_tight.py").write_text(
        source + "    if share != '0.5':\n"
                 "        raise RuntimeError('RESOURCE_EXHAUSTED: Out of memory allocating 9.0GiB')\n")
    (tmp_path / "tests/test_wrong.py").write_text(
        source + "    assert share == '0.5', 'an accuracy miss, not RESOURCE_EXHAUSTED: out of memory'\n")
    (tmp_path / "tests/test_whole.py").write_text(
        source + "    raise RuntimeError('RESOURCE_EXHAUSTED: Out of memory allocating 90.0GiB')\n")
    (tmp_path / "tests/test_new.py").write_text(source)
    every = [*names, "tests/test_tight.py", "tests/test_wrong.py", "tests/test_whole.py", "tests/test_new.py"]
    interpreter(tmp_path, "3.12")
    fetcher(tmp_path)
    # Another class measured 4 GiB for each file but the new one; a third, smaller peaks for some.
    for other, peak in (("NVIDIA-H100-80GB-HBM3-x86", 4), ("NVIDIA-A10-x86", 2)):
        for name in [*names, "tests/test_tight.py", "tests/test_wrong.py", "tests/test_whole.py"]:
            at = "/" + ci.record_key("3.12", ci.row("3.12", name, ""), other)
            whole = 64 if name == "tests/test_whole.py" else peak
            stored[at] = json.dumps({"key": None, "depends": [name], "row": {}, "reads": [],
                                     "peak": whole * 2 ** 30, "shares": 0, "pools": 0}).encode()

    def ran(stem: str) -> list[list[str]]:
        return lane_runs(tmp_path, stem)

    out = tmp_path / "verdict.json"
    assert ci.gpu_lane("3.12", every, out, cache="on", processes=4, deadline=600.0) == 0
    rows = {row["name"]: row for row in json.loads(out.read_text())["rows"]}
    borrowed = str(6 * 2 ** 30 / (81920 * 2 ** 20))
    first = [ran(Path(name).stem)[0] for name in names]
    assert [share for share, *_ in first] == [borrowed] * len(names), "the largest class's 4 GiB and headroom"
    assert overlapping(first) >= 2, "borrowed shares pack as measured ones do"
    assert [share for share, *_ in ran("test_new")] == ["0.5"], "measured on no class: alone, the whole share"
    for stem, said in (("test_tight", "RESOURCE_EXHAUSTED"), ("test_wrong", "an accuracy miss"),
                       ("test_whole", "RESOURCE_EXHAUSTED")):
        found = rows[f"3.12:tests/{stem}.py"]
        assert found["exitCode"] == 1 and said in found["output"], stem
        assert len(ran(stem)) == 1, f"{stem}: red, run once"
    assert ran("test_tight")[0][0] == borrowed
    assert ran("test_whole")[0][0] == "0.5", "its borrowed measurement already reached the whole share"
    # A Worker that does not list its records: nothing is borrowed, and an unmeasured file runs alone.
    (tmp_path / "tests/test_listless.py").write_text(source)
    for other in ("NVIDIA-H100-80GB-HBM3-x86",):
        at = "/" + ci.record_key("3.12", ci.row("3.12", "tests/test_listless.py", ""), other)
        stored[at] = json.dumps({"key": None, "depends": [], "row": {}, "reads": [], "peak": 2 ** 30,
                                 "shares": 0, "pools": 0}).encode()
    listing = ci.r2.Store.keys

    def refused(self, prefix):
        raise ci.r2.Unavailable("listing refused")
        yield

    monkeypatch.setattr(ci.r2.Store, "keys", refused)
    assert ci.gpu_lane("3.12", ["tests/test_listless.py"], out, cache="off", processes=4, deadline=600.0) == 0
    assert [share for share, *_ in ran("test_listless")] == ["0.5"]
    monkeypatch.setattr(ci.r2.Store, "keys", listing)
    # This class's own records now: the next lane reads them, the tight file's red run's, not the borrowed.
    assert ci.gpu_lane("3.12", ["tests/test_tight.py"], out, cache="off", processes=4, deadline=600.0) == 0
    assert [share for share, *_ in ran("test_tight")][-1] != borrowed


def test_a_gpu_lanes_groups_run_each_test_of_the_tips_file_once_their_rows_their_own(
        ci, tmp_path, monkeypatch, cache_worker):
    """A GPU tier's long file runs as CI's groups (`<file>#G/N`), each a task of the lane that cuts the
    tests the file's collection here finds by the seconds the tier's GPUs took them
    (DEW_SPLIT_DURATIONS), a test they never timed weighing their mean: so each test runs exactly once
    over the groups, one added since and one renamed since among them, and each group's row holds its
    own tests' cases (audit, 2026-11-08: the tier named the tests an earlier tier ran, so a new one
    never ran, and a test named apart read "no evidence ran")."""
    url, _, _ = cache_worker
    gpu_lane_at(tmp_path, monkeypatch, url, 0)
    names = ("a", "b", "c", "renamed", "added")
    (tmp_path / "tests/test_long.py").write_text("".join(
        f"def test_{name}():\n    with open('ran-{name}', 'a') as ran:\n        ran.write('1')\n"
        for name in names))
    # The tier's GPU seconds of the tests it ran before: `renamed` was `old`, and `added` is new.
    weights = tmp_path / "gpu-seconds.json"
    weights.write_text(json.dumps({f"tests/test_long.py::test_{name}": 30.0
                                   for name in ("a", "b", "c", "old")}))
    monkeypatch.setenv("DEW_SPLIT_DURATIONS", str(weights))
    interpreter(tmp_path, "3.12")
    fetcher(tmp_path)
    out = tmp_path / "verdict.json"
    groups = ["tests/test_long.py#1/2", "tests/test_long.py#2/2"]
    assert ci.gpu_lane("3.12", groups, out, cache="on", processes=2, deadline=600.0) == 0
    rows = {row["name"]: row for row in json.loads(out.read_text())["rows"]}
    assert sorted(rows) == [f"3.12:{group}" for group in groups]
    assert all(row["exitCode"] == 0 for row in rows.values()), rows
    assert {name: (tmp_path / f"ran-{name}").read_text() for name in names} == dict.fromkeys(names, "1")
    held = [set(row["tests"]) for row in rows.values()]
    assert held[0] and held[1] and not held[0] & held[1]
    assert held[0] | held[1] == {f"tests/test_long.py::test_{name}" for name in names}
    # Cut by other seconds, the groups run again, each test once more: no row an earlier cut left stands
    # for one.
    weights.write_text(json.dumps({"tests/test_long.py::test_a": 300.0}))
    assert ci.gpu_lane("3.12", groups, out, cache="on", processes=2, deadline=600.0) == 0
    assert {name: (tmp_path / f"ran-{name}").read_text() for name in names} == dict.fromkeys(names, "11")


def test_a_gpu_lane_given_test_nodes_grades_each_by_its_own_cases(ci, tmp_path, monkeypatch, cache_worker):
    """A lane given single tests by node id (`file::test`, a parametrized one among them, as a lane's own GPU
    run names them) grades each by its own cases: one that passed is green, one whose case failed red with
    it, none "never ran" (2026-11-17: every node id a tier was given read "no evidence ran", its cases matched
    against their file's name)."""
    url, _, _ = cache_worker
    gpu_lane_at(tmp_path, monkeypatch, url, 0)
    (tmp_path / "tests/test_nodes.py").write_text(
        "import pytest\n"
        "def test_walk():\n    pass\n"
        "@pytest.mark.parametrize('form', ['pipeline', 'streamed'])\n"
        "def test_pipeline(form):\n    assert form == 'pipeline'\n"
        "def test_other():\n    raise SystemExit('not named')\n")
    interpreter(tmp_path, "3.12")
    fetcher(tmp_path)
    out = tmp_path / "verdict.json"
    nodes = ["tests/test_nodes.py::test_walk", "tests/test_nodes.py::test_pipeline"]
    ci.gpu_lane("3.12", nodes, out, cache="off", processes=1, deadline=600.0)
    rows = {row["name"]: row for row in json.loads(out.read_text())["rows"]}
    assert {name: row["exitCode"] for name, row in rows.items()} == {
        "3.12:tests/test_nodes.py::test_walk": 0, "3.12:tests/test_nodes.py::test_pipeline": 1}
    assert "test_pipeline[streamed]" in rows["3.12:tests/test_nodes.py::test_pipeline"]["output"]
    assert all("no evidence" not in row["output"] for row in rows.values())


def test_a_test_nodes_green_run_keeps_a_record_its_next_lane_reuses(ci, tmp_path, monkeypatch, cache_worker):
    """A test node's green run (`file::test`, its parametrized case's id outside the Worker's key alphabet)
    keeps its file's record under a place the Worker takes (`rows.row_place`), so a second lane on the same
    inputs reuses it rather than run it again, and another node of the same file is no reuse of it (review of
    the first node-id grading: a node row found no record of its process, and its place held a colon)."""
    lane_machine(tmp_path, monkeypatch, cache_worker[0], "tests/test_nodes.py",
        "import pytest\nfrom pathlib import Path\n"
        "@pytest.mark.parametrize('form', ['a/b c'])\n"
        "def test_walk(form):\n    Path('walked').write_text(form)\n"
        "def test_other():\n    Path('othered').write_text('ran')\n")
    walked, othered = tmp_path / "walked", tmp_path / "othered"

    def lane(node: str) -> list[tuple[str, int]]:
        return lane_rows(ci, tmp_path, [node], [walked, othered])
    walk, other = "tests/test_nodes.py::test_walk", "tests/test_nodes.py::test_other"
    assert lane(walk) == [(f"3.12:{walk}", 0)]
    assert walked.read_text() == "a/b c"
    assert lane(walk) == [(f"3.12:{walk}", 0)]
    assert not walked.exists(), "the node's green run on the same inputs is reused"
    assert lane(other) == [(f"3.12:{other}", 0)]
    assert othered.is_file() and not walked.exists(), "another node of its file runs its own"


def test_a_test_nodes_record_is_kept_where_the_worker_takes_it_and_read_back_as_its_node(ci):
    """A test node's record place (`rows.row_place`) is a key the Worker takes, however its parametrized
    case is spelled or long, distinct from its file's and its siblings', and its readers (`key_parts`) read
    it back as the node, a long one as its file's node by digest, never a file of its own; a file whose
    name holds an escape's characters reads back as itself."""
    file = "tests/test_nodes.py"
    nodes = [f"{file}::test_walk", f"{file}::test_walk[a/b c]", f"{file}::Walks::test_walk[\u00e9-..]",
             f"{file}::test_walk[{'x' * 400}]"]
    places = [ci.record_key("3.12", ci.row("3.12", node, ""), "sm90") for node in [file, *nodes]]
    assert all(ci.r2.valid(place) for place in places) and len(set(places)) == len(places)
    read = [ci.key_parts(place.removeprefix(f"{ci.CACHE_PREFIX}/3.12/"))[1] for place in places]
    assert read[:4] == [file, *nodes[:3]]
    assert read[4].startswith(f"{file}::") and read[4] != nodes[3]
    odd = "tests/test_a=20b.py"
    assert ci.key_parts(ci.record_key("3.12", ci.row("3.12", odd, ""), "sm90").removeprefix(
        f"{ci.CACHE_PREFIX}/3.12/"))[1] == odd


def test_a_tier_cut_groups_green_row_is_reused_while_its_cut_holds_the_same_tests(
        ci, tmp_path, monkeypatch, cache_worker):
    """A group the tier cuts by its GPUs' own seconds (DEW_SPLIT_DURATIONS) keeps its green row under a key,
    as a CPU group cut by the durations record does, and a later lane on unchanged inputs reuses it: the
    key holds the tests that cut gives the group, so a row stands for its group only while the cut gives it
    the same tests, whatever their seconds, and a cut that moves a test between groups runs both again, each
    test once (a tier's cut files, most of its minutes, ran whole every time: under the tier's cut a group
    read no row and kept none)."""
    url, _, _ = cache_worker
    gpu_lane_at(tmp_path, monkeypatch, url, 0)
    names = ("a", "b", "c", "d")
    # Each run of a test leaves a file of its own, which no later run reads: a file a run appends to is one
    # it read and wrote, and its row would keep no key.
    (tmp_path / "tests/test_long.py").write_text("import os\nfrom pathlib import Path\n" + "".join(
        f"def test_{name}():\n    Path(f'ran-{name}-{{os.getpid()}}').write_text('')\n" for name in names))
    # The tier's seconds, outside the checkout as a node's session keeps them: within it, the record a run's
    # split read would be an input of its row, and other seconds another key.
    weights = tmp_path.with_name(f"{tmp_path.name}-gpu-seconds.json")
    monkeypatch.setenv("DEW_SPLIT_DURATIONS", str(weights))
    interpreter(tmp_path, "3.12")
    fetcher(tmp_path)
    out = tmp_path / "verdict.json"
    groups = ["tests/test_long.py#1/2", "tests/test_long.py#2/2"]

    def lane(**seconds: float) -> tuple[dict[str, str], list[set[str]]]:
        """The lane over both groups, cut by `seconds`: how often each test has run, and each row's tests."""
        weights.write_text(json.dumps({f"tests/test_long.py::test_{name}": seconds[name] for name in names}))
        assert ci.gpu_lane("3.12", groups, out, cache="on", processes=2, deadline=600.0) == 0
        rows = {row["name"]: row for row in json.loads(out.read_text())["rows"]}
        assert sorted(rows) == [f"3.12:{group}" for group in groups]
        assert all(row["exitCode"] == 0 for row in rows.values()), rows
        held = [{node.rpartition("::test_")[2] for node in rows[f"3.12:{group}"]["tests"]}
                for group in groups]
        return {name: "1" * len(list(tmp_path.glob(f"ran-{name}-*"))) for name in names}, held

    assert lane(a=30.0, b=30.0, c=30.0, d=30.0) == (dict.fromkeys(names, "1"), [{"a", "b"}, {"c", "d"}])
    # The same cut again, and one of other seconds that gives each group the same tests: no test runs, and
    # each row is the one its group left.
    assert lane(a=30.0, b=30.0, c=30.0, d=30.0) == (dict.fromkeys(names, "1"), [{"a", "b"}, {"c", "d"}])
    assert lane(a=33.0, b=30.0, c=30.0, d=31.0) == (dict.fromkeys(names, "1"), [{"a", "b"}, {"c", "d"}])
    # A cut that gives the groups other tests: neither row stands for its group, and each test runs once.
    assert lane(a=300.0, b=30.0, c=30.0, d=30.0) == (dict.fromkeys(names, "11"), [{"a"}, {"b", "c", "d"}])
    # The first cut again: its rows, kept under their own keys, are found, and no test runs.
    assert lane(a=30.0, b=30.0, c=30.0, d=30.0) == (dict.fromkeys(names, "11"), [{"a", "b"}, {"c", "d"}])
    # Seconds that name fewer of the file's tests leave the first group its own two, and the test they lack
    # to whichever group the collection gives it: no group's row stands, and every test runs, the one the
    # seconds lack among them (review: the first group's row was reused, and that test ran in no group).
    weights.write_text(json.dumps({f"tests/test_long.py::test_{name}": seconds
                                   for name, seconds in (("a", 30.0), ("b", 30.0), ("c", 60.0))}))
    assert ci.gpu_lane("3.12", groups, out, cache="on", processes=2, deadline=600.0) == 0
    assert {name: len(list(tmp_path.glob(f"ran-{name}-*"))) for name in names} == dict.fromkeys(names, 3)


def test_a_tiers_files_are_named_with_the_generations_their_tests_are_pinned_to(ci, tmp_path, monkeypatch):
    """`gpu-tier` names each GPU-marked file with the GPU generations its tests are pinned to, read from their
    marks as pytest collects them (`generations`, on a test, a parameter of it, or its module), and with none
    where any test of it is pinned to none: a tier's composer gives a kind every file but those pinned to
    other generations, so a file that names no generation runs on every kind (2026-11-11: chosen by the sm90
    words in a file's text, the tier's Hopper side ran none of the model flows, whose file names no
    generation)."""
    interpreter(tmp_path, "3.12")
    (tmp_path / "tests/conftest.py").write_text(
        "def pytest_configure(config):\n"
        "    for mark in ('gpu', 'generations'):\n"
        "        config.addinivalue_line('markers', mark + ': a GPU mark')\n")
    (tmp_path / "tests/test_every.py").write_text(
        "import pytest\npytestmark = pytest.mark.gpu\n"
        "@pytest.mark.generations('sm90')\ndef test_hopper():\n    pass\n"
        "def test_anywhere():\n    pass\n")
    (tmp_path / "tests/test_hopper.py").write_text(
        "import pytest\npytestmark = [pytest.mark.gpu, pytest.mark.generations('sm90')]\n"
        "def test_one():\n    pass\n"
        "@pytest.mark.parametrize('cell', [1, 2])\ndef test_cells(cell):\n    pass\n")
    (tmp_path / "tests/test_either.py").write_text(
        "import pytest\npytestmark = pytest.mark.gpu\n"
        "@pytest.mark.parametrize('cell', [pytest.param(1, marks=pytest.mark.generations('sm80')),\n"
        "    pytest.param(2, marks=pytest.mark.generations('sm90', 'sm100'))])\n"
        "def test_cells(cell):\n    pass\n")
    # A file with a pinned test no GPU runs is none of the tier's.
    (tmp_path / "tests/test_cpu.py").write_text(
        "import pytest\n@pytest.mark.generations('sm90')\ndef test_lowers():\n    pass\n")
    counts, pinned = ci.marked("3.12")
    assert counts == {"tests/test_either.py": 2, "tests/test_every.py": 2, "tests/test_hopper.py": 3}
    assert pinned == {"tests/test_either.py": ["sm100", "sm80", "sm90"], "tests/test_every.py": [],
                      "tests/test_hopper.py": ["sm90"]}


def test_a_tier_cut_group_is_admitted_on_the_most_its_file_held_and_no_need_is_cut_to_the_lane(
        ci, tmp_path, monkeypatch, cache_worker):
    """A group the tier cut by its own seconds (DEW_SPLIT_DURATIONS) holds tests another cut's group of
    that name did not, so its admission reads the most any run of its file on the class held, whole or a
    group of any cut, never the group's own record (2026-11-10: a recurrence file's group 1/3, measured when
    it held forward and decode alone at 0.29 GiB, was given a 1.36 GiB pool when it held port training too,
    which asked 1.52 GiB). And a file whose measured need passes the lane's share is given the whole share and
    said to be, never cut to it without a word (the same day: a 5.9 GiB need cut to a 4 GiB lane)."""
    url, stored, _ = cache_worker
    gpu_lane_at(tmp_path, monkeypatch, url, 0)
    (tmp_path / "tests/test_long.py").write_text(LANE_TEST.replace("def test_runs", "def test_a") + "\n"
                                                  + LANE_TEST.replace("def test_runs", "def test_b"))
    weights = tmp_path / "gpu-seconds.json"
    weights.write_text(json.dumps({"tests/test_long.py::test_a": 30.0, "tests/test_long.py::test_b": 30.0}))
    monkeypatch.setenv("DEW_SPLIT_DURATIONS", str(weights))
    interpreter(tmp_path, "3.12")
    fetcher(tmp_path)
    gib, total = 2 ** 30, 81920 * 2 ** 20
    common = ci.lane_inputs("3.12")

    def measured(row_name: str, peak: int, shares: int = 0, followed: bool = True) -> None:
        for inputs in (None, common):
            at = "/" + ci.record_key("3.12", row_name, ci.run_keys.device_class(), inputs)
            stored[at] = json.dumps({"key": None, "depends": ["tests/test_long.py"] if followed else None,
                                     "row": {}, "reads": [], "peak": peak, "shares": shares,
                                     "pools": 0}).encode()

    out = tmp_path / "verdict.json"
    # Each lane here runs its file (the cache off): a row reused is admitted nothing.
    # Measured on no class: given the whole share, as every such file is, and no need said, as none is known.
    assert ci.gpu_lane("3.12", ["tests/test_long.py#1/2"], out, cache="off", processes=1, deadline=600.0) == 0
    [row] = json.loads(out.read_text())["rows"]
    assert float(lane_runs(tmp_path, "test_long")[-1][0]) == 0.5 and "admitted at" not in row["output"]
    # Its group 1/2 measured small under an earlier cut; the whole file at 8 GiB.
    measured("3.12:tests/test_long.py#1/2", gib // 4)
    measured("3.12:tests/test_long.py", 8 * gib)
    assert ci.gpu_lane("3.12", ["tests/test_long.py#1/2"], out, cache="off", processes=1, deadline=600.0) == 0
    [share, *_] = lane_runs(tmp_path, "test_long")[-1]
    assert float(share) * total == 8 * gib * 1.25 + gib, "the file's whole measure, not the group's"
    # Measured past the lane's 40 GiB: given all 40, its row saying its need was more.
    measured("3.12:tests/test_long.py", 48 * gib)
    assert ci.gpu_lane("3.12", ["tests/test_long.py#1/2"], out, cache="off", processes=1, deadline=600.0) == 0
    [row] = json.loads(out.read_text())["rows"]
    share = lane_runs(tmp_path, "test_long")[-1][0]
    assert float(share) == 0.5
    assert "admitted at 40.0 GiB, below its measured need of 61.0 GiB" in row["output"]
    # A whole run measured before the file gained a test holds less than that test does: the most any run of
    # the file on this class held is its measure, a group's of any cut among them (2026-11-11: a recurrence
    # file's whole run, 3.92 GiB from before its short conv matrix test, gave that test alone a 5.9 GiB lane,
    # which ran out at 5.38 GiB in use, twice).
    measured("3.12:tests/test_long.py", 4 * gib)
    measured("3.12:tests/test_long.py#3/3", 6 * gib)
    assert ci.gpu_lane("3.12", ["tests/test_long.py#1/2"], out, cache="off", processes=1, deadline=600.0) == 0
    [share, *_] = lane_runs(tmp_path, "test_long")[-1]
    assert float(share) * total == 6 * gib * 1.25 + gib, "the largest of the file's measured runs"
    # The most of each run's measure, not one run's: a group's run with a child that took its fraction held
    # more than the whole run of a larger peak, whose share the process is given all the same.
    measured("3.12:tests/test_long.py", 8 * gib)
    measured("3.12:tests/test_long.py#3/3", 4 * gib, shares=1)
    assert ci.gpu_lane("3.12", ["tests/test_long.py#1/2"], out, cache="off", processes=1, deadline=600.0) == 0
    [share, *_] = lane_runs(tmp_path, "test_long")[-1]
    assert float(share) * total == 8 * gib * 1.25 + gib, "the largest peak's share, whichever run held most"
    # A whole run that started a process it did not follow measured nothing: the file takes the lane's whole
    # share, whatever a group's run measured before.
    measured("3.12:tests/test_long.py", 2 * gib, followed=False)
    assert ci.gpu_lane("3.12", ["tests/test_long.py#1/2"], out, cache="off", processes=1, deadline=600.0) == 0
    assert float(lane_runs(tmp_path, "test_long")[-1][0]) == 0.5


def test_a_unit_is_given_no_less_than_dews_admission_of_the_programs_its_prepare_filed(
        ci, tmp_path, monkeypatch, cache_worker):
    """A lane unit whose prepare filed its programs' memory (`tier_wire.needs_file`, its prepare told
    which unit it is by DEW_TIER_UNIT) is given no less than the least pool at which Dew's admission
    (dew.training.memory `least_limit`: `needed`, the GPU's memory, read as its pool's share)
    takes the most demanding of them, beside what its runs measured: a part's float64 truths, which no
    run of it held, are admitted before they run (2026-11-17: two flows' 1.71 GiB terms walk beside
    1.1 GiB of arms ran out of a 2.84 GiB share). One past the lane's share is not run below it: its row
    has no verdict, naming its need and the GPU (Main, 2026-11-17). Dew's own function reads it
    (`least_limit`), here as the lane does, on a lane a tenth of its GPU, where a need read in the wrong
    units is ten times off (rows 7460, 7461: red unschedulable at 34.88 GiB)."""
    from dew.training.memory import least_limit

    url, stored, _ = cache_worker
    gpu_lane_at(tmp_path, monkeypatch, url, 0)
    (tmp_path / "tests/test_flow.py").write_text(LANE_TEST)
    interpreter(tmp_path, "3.12")
    fetcher(tmp_path)
    truths = tmp_path / "truths"
    monkeypatch.setenv(tier_wire.TRUTHS, str(truths))
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    gib, mib, total = 2 ** 30, 2 ** 20, 81920 * 2 ** 20
    budget, whole = 0.1 * total, float(ci.WHOLE_GPU) * total

    def admitted(*programs: tuple[int, int, int]) -> int:
        """Dew's admission of `programs` on this device, in its pool's bytes, as the lane reads it."""
        return least_limit(jax.local_devices()[0], programs)

    def measured() -> None:
        # Its measured 2 GiB, as each run here records its own (with no GPU, none) over it.
        for inputs in (None, ci.lane_inputs("3.12")):
            at = "/" + ci.record_key("3.12", "3.12:tests/test_flow.py", ci.run_keys.device_class(), inputs)
            stored[at] = json.dumps({"key": None, "depends": ["tests/test_flow.py"], "row": {}, "reads": [],
                                     "peak": 2 * gib, "shares": 0, "pools": 0}).encode()

    def filed(*programs: tuple[int, int, int]) -> None:
        name = "tests/test_flow.py"
        listed = [{"resident": r, "beside": b, "temporaries": m} for r, b, m in programs]
        tier_wire.needs_file(truths, name).parent.mkdir(parents=True, exist_ok=True)
        tier_wire.needs_file(truths, name).write_text(json.dumps({"unit": name, "programs": listed}))

    def lane() -> tuple[float, dict]:
        measured()
        out = tmp_path / "verdict.json"
        assert ci.gpu_lane("3.12", ["tests/test_flow.py"], out, cache="off", processes=1, deadline=600.0) == 0
        [row] = json.loads(out.read_text())["rows"]
        return float(lane_runs(tmp_path, "test_flow")[-1][0]) * total, row

    # Its measured 2 GiB alone: 3.5 GiB, its peak and headroom.
    share, row = lane()
    assert (share, row["output"]) == (2 * gib * 1.25 + gib, "")
    # A GH200 part's flows as their prepare filed them (2026-11-17, MAINGPU-8063e3557-PART3of9's
    # needs/117ca29968591c89.json, its two largest of 80 programs), and arms beside a float64 walk: each run,
    # given no less than Dew's admission of them.
    for programs in (((76 * mib, 107 * mib, 1836 * mib), (76 * mib, 107 * mib, 1428 * mib)),
                     ((gib, gib // 2, gib // 4), (gib, gib, 5 * gib))):
        filed(*programs)
        share, row = lane()
        assert admitted(*programs) <= budget
        assert (row["output"], row["exitCode"]) == ("", 0)
        # Its measured peak's share raised to Dew's admission, exactly as the lane gives it.
        record = {"depends": ["tests/test_flow.py"], "peak": 2 * gib, "shares": 0, "pools": 0}
        given = ci.gpu_needs(record, budget, float(admitted(*programs)))[0]
        assert share == float(str(given / total)) * total
    # Its children each take a share's fraction: what it holds is of its raised share, 7 GiB twice over.
    assert ci.gpu_needs({"depends": [], "peak": 2 * gib, "shares": 1, "pools": 0}, 40 * gib, 7 * gib) == (
        7 * gib, 14 * gib)
    # Programs past the lane's share: none runs, none with a verdict, its row naming its need and the GPU,
    # and saying a lane of the whole GPU holds it only where that lane's (JAX's 75%) does.
    for temporaries in (12 * gib, 70 * gib):
        filed((gib, gib, temporaries))
        need = admitted((gib, gib, temporaries))
        runs = len(lane_runs(tmp_path, "test_flow"))
        _, row = lane()
        assert need > budget and len(lane_runs(tmp_path, "test_flow")) == runs and row["exitCode"] == 1
        # No verdict of its tests (`grading.STALE`): Platform's harness evidence, never its file's verdict.
        assert row["kind"] == ci.grading.STALE and f"{need / gib:.2f} GiB" in row["output"]
        held = ci.run_keys.device_class()
        assert held in row["output"] and ("holds it" in row["output"]) == (need <= whole)


def test_gpu_holds_says_what_a_lane_needs_to_run_each_file_as_its_largest_measured_class_held_it(
        ci, tmp_path, monkeypatch, cache_worker, capsys):
    """`gpu_holds.py` says, for each test file (a group as its file), the most any GPU class's last run of
    it held with its children (`gpu_needs`), whole or a group's, in bytes, so a GPU tier sizes each part's job
    by what its lane will admit its files at (2026-11-10: a part given a default 8 GiB had a 4 GiB lane for a
    file measured at 5.9 GiB); a file no class measured is absent."""
    url, stored, _ = cache_worker
    cached(monkeypatch, url)
    gib = 2 ** 30
    for other, peak in (("NVIDIA-H100-80GB-HBM3-x86", 4), ("NVIDIA-A10-x86", 2)):
        at = "/" + ci.record_key("3.12", ci.row("3.12", "tests/test_wide.py", ""), other)
        stored[at] = json.dumps({"key": None, "depends": ["tests/test_wide.py"], "row": {}, "reads": [],
                                 "peak": peak * gib, "shares": 1, "pools": 0}).encode()
    holds = load("gpu_holds", "tools/armada")
    monkeypatch.setattr(sys, "argv", ["gpu_holds.py", "--tests=tests/test_wide.py#2/3 tests/test_new.py"])
    assert holds.main() == 0
    assert json.loads(capsys.readouterr().out) == {"tests/test_wide.py": 2 * (4 * gib * 1.25 + gib)}
    # A file's process under a user's flags is a run of the file too: where it held more than the lane's own,
    # the file's part is admitted for it (its record was no "whole" record's, and the part was sized without
    # it).
    users = "/" + ci.record_key("3.12", ci.row("3.12", "tests/test_wide.py", ci.varied("", "users")),
                                "NVIDIA-H100-80GB-HBM3-x86")
    stored[users] = json.dumps({"key": None, "depends": ["tests/test_wide.py"], "row": {}, "reads": [],
                                "peak": 6 * gib, "shares": 1, "pools": 0}).encode()
    assert holds.main() == 0
    assert json.loads(capsys.readouterr().out) == {"tests/test_wide.py": 2 * (6 * gib * 1.25 + gib)}
    # A group's run is one too, and the file's need the most of each measure: its larger peak, and the child a
    # whole run had beside it.
    group = "/" + ci.record_key("3.12", ci.row("3.12", "tests/test_wide.py", "2/5"),
                                "NVIDIA-H100-80GB-HBM3-x86")
    stored[group] = json.dumps({"key": None, "depends": ["tests/test_wide.py"], "row": {}, "reads": [],
                                "peak": 9 * gib, "shares": 0, "pools": 0}).encode()
    assert holds.main() == 0
    assert json.loads(capsys.readouterr().out) == {"tests/test_wide.py": 2 * (9 * gib * 1.25 + gib)}
    stored[group] = json.dumps({"key": None, "depends": ["tests/test_wide.py"], "row": {}, "reads": [],
                                "peak": 20 * gib, "shares": 0, "pools": 0}).encode()
    assert holds.main() == 0
    assert json.loads(capsys.readouterr().out) == {"tests/test_wide.py": 2 * (20 * gib * 1.25 + gib)}
    # The cache not answering (here, refusing the token) fails the tool, which sizes a job by what it says:
    # read as files no class measured, a part took the default (2026-11-11: a file measured at 5.9 GiB ran
    # out of the 4 GiB lane of a part composed while a read of its record did not come back). A lane on its
    # node, which only packs by it, still runs: each file alone, with the whole share.
    monkeypatch.setenv("DEW_XLA_CACHE_TOKEN", "another")
    with pytest.raises(Exception) as unanswered:
        holds.main()
    assert "R2Error" in {base.__name__ for base in type(unanswered.value).__mro__}
    assert capsys.readouterr().out == ""
    assert ci.borrowed_records(["tests/test_wide.py"], "3.12", "", "another") == {}
