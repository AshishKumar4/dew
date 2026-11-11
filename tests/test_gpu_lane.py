"""CI's GPU lane (tools/armada/ci.py `gpu-lane`): each test file in a process of its own, admitted on the
GPU memory its runs measured, and tools/armada/gpu_holds.py, what a lane needs to run a file."""

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
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


def test_a_network_lane_runs_the_network_marked_tests_with_the_hub_open_and_records_them_apart(
        ci, tmp_path, monkeypatch, cache_worker):
    """The network lane runs each file's network-marked tests alone, which every other task sets aside,
    with the Hub open, and records them on a class of their own: the same file's CUDA lane, Hub closed,
    runs its other tests and is no reuse of the network lane's run, and a second network lane on the same
    inputs reuses its own (the maintainer, 2026-10-25: no CI run reached the published checkpoints' parity
    tests)."""
    url, _, _ = cache_worker
    gpu_lane_at(tmp_path, monkeypatch, url, 0)
    interpreter(tmp_path, "3.12")
    fetcher(tmp_path)
    (tmp_path / "tests/test_hub.py").write_text(
        "import os, pytest\nfrom pathlib import Path\n"
        "@pytest.mark.network\ndef test_reads_the_hub():\n"
        "    assert not {'HF_HUB_OFFLINE', 'HF_ENDPOINT'} & set(os.environ), 'the Hub is closed'\n"
        "    Path('hub').write_text('ran')\n"
        "def test_offline():\n"
        "    assert os.environ['HF_HUB_OFFLINE'] == '1'\n"
        "    Path('offline').write_text('ran')\n")
    out = tmp_path / "verdict.json"
    hub, offline = tmp_path / "hub", tmp_path / "offline"

    def lane(**network: bool) -> list[tuple[str, int]]:
        hub.unlink(missing_ok=True)
        offline.unlink(missing_ok=True)
        assert ci.gpu_lane("3.12", ["tests/test_hub.py"], out, cache="on", processes=1, deadline=600.0,
                           **network) == 0
        return [(row["name"], row["exitCode"]) for row in json.loads(out.read_text())["rows"]]

    assert lane(network=True) == [("3.12:tests/test_hub.py", 0)]
    assert hub.is_file() and not offline.exists()
    assert lane() == [("3.12:tests/test_hub.py", 0)]
    assert offline.is_file() and not hub.exists(), "the CUDA lane's run is its own"
    assert lane(network=True) == [("3.12:tests/test_hub.py", 0)]
    assert not hub.exists(), "the network lane reuses its own green run on the same inputs"


def test_a_gpu_lane_given_the_whole_gpu_hands_its_files_jaxs_default_share(ci, tmp_path, monkeypatch,
                                                                            cache_worker):
    """A job holding a whole GPU, to which a node sets no XLA_PYTHON_CLIENT_MEM_FRACTION (dew-ops' session
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
    """A lane's prepare (`gpu-lane --prepare`) runs each of its files that holds a test marked `prepares`, on
    the CPU under the prepare's word (DEW_TIER_PREPARE, which the suite's conftest reads to skip every other
    test), with the store its environment names: a cut file's group prepares exactly the tests its run then
    holds, as both select and cut the file alike (a prepare that selected the marked tests alone would cut
    them into other groups, and a run would find a test its part never prepared); a file with nothing to
    prepare starts no process; and a prepare records no row, so no lane or task reuses it as a test's run.
    The fixtures are fetched first, as a run's are."""
    url, stored, _ = cache_worker
    gpu_lane_at(tmp_path, monkeypatch, url, 0)
    interpreter(tmp_path, "3.12")
    fetched = fetcher(tmp_path)
    monkeypatch.setenv("DEW_TIER_TRUTHS", str(tmp_path / "truths"))
    # The harness's side, as the suite's own conftest has it: a prepare skips every test but those it
    # prepares.
    (tmp_path / "tests/conftest.py").write_text(
        "import os, pytest\n"
        "def pytest_configure(config):\n"
        "    config.addinivalue_line('markers', 'prepares: prepared on the CPU before its GPU run')\n"
        "def pytest_runtest_setup(item):\n"
        "    if os.environ.get('DEW_TIER_PREPARE') == '1' and item.get_closest_marker('prepares') is None:\n"
        "        pytest.skip('nothing to prepare')\n")
    (tmp_path / "tests/test_flows.py").write_text(
        "import os, pytest\n"
        "def said(name):\n"
        "    kind = 'prepared' if os.environ.get('DEW_TIER_PREPARE') == '1' else 'ran'\n"
        "    with open(f'{kind}.log', 'a') as log:\n"
        "        log.write(' '.join([name, os.environ.get('JAX_PLATFORMS', '-'),\n"
        "                            os.environ.get('DEW_TIER_TRUTHS', '-')]) + chr(10))\n"
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
    prepared = {}
    for group in ("1/2", "2/2"):
        name = f"tests/test_flows.py#{group}"
        assert ci.gpu_prepared("3.12", [name, "tests/test_plain.py"], out, processes=2, deadline=600.0) == 0
        assert [(row["name"], row["exitCode"]) for row in json.loads(out.read_text())["rows"]] == [
            (f"3.12:{name}", 0)]
        lines = said("prepared")
        assert {tuple(line[1:]) for line in lines} == {("cpu", str(tmp_path / "truths"))}
        prepared[group] = {line[0] for line in lines}
    assert fetched.is_file() and not lane_runs(tmp_path, "test_plain"), "nothing to prepare starts no process"
    assert not stored, "a prepare records no row"
    assert not (tmp_path / "artifacts" / "progress.jsonl").exists(), "nor any test as passed (review)"
    monkeypatch.delenv("ARMADA_RESUME")
    assert prepared["1/2"] | prepared["2/2"] == marked and not prepared["1/2"] & prepared["2/2"]
    for group in ("1/2", "2/2"):
        assert ci.gpu_lane("3.12", [f"tests/test_flows.py#{group}"], out, cache="on", processes=1,
                           deadline=600.0) == 0
        assert {line[0] for line in said("ran")} & marked == prepared[group], group
    assert stored, "its run's rows are recorded as any lane's"


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
        "    if os.environ.get('DEW_TIER_PREPARE') == '1' and item.get_closest_marker('prepares') is None:\n"
        "        pytest.skip('nothing to prepare')\n")
    (tmp_path / "tests/test_flows.py").write_text(
        "import os, pytest\n"
        "from pathlib import Path\n"
        "def said(name):\n"
        "    kind = 'prepared' if os.environ.get('DEW_TIER_PREPARE') == '1' else 'ran'\n"
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
        at = "/" + ci.record_key("3.12", ci.row("3.12", name, ""), ci.device_class(), inputs)
        stored[at] = json.dumps({"key": None, "depends": [name], "row": {}, "reads": [], "peak": 4 * 2 ** 30,
                                 "shares": int(name == "tests/test_spawner.py"), "pools": 0}).encode()
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


def test_a_tier_cut_group_is_admitted_on_its_files_whole_measure_and_no_need_is_cut_to_the_lane(
        ci, tmp_path, monkeypatch, cache_worker):
    """A group the tier cut by its own seconds (DEW_SPLIT_DURATIONS) holds tests another cut's group of
    that name did not, so its admission reads the file's whole measured run, which holds any group of it,
    never the group's own record (2026-11-10: a recurrence file's group 1/3, measured when it held forward
    and decode alone at 0.29 GiB, was given a 1.36 GiB pool when it held port training too, which asked
    1.52 GiB). And a file whose measured need passes the lane's share is given the whole share and said to
    be, never cut to it without a word (the same day: a 5.9 GiB need cut to a 4 GiB lane)."""
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

    def measured(row_name: str, peak: int) -> None:
        for inputs in (None, common):
            at = "/" + ci.record_key("3.12", row_name, ci.device_class(), inputs)
            stored[at] = json.dumps({"key": None, "depends": ["tests/test_long.py"], "row": {}, "reads": [],
                                     "peak": peak, "shares": 0, "pools": 0}).encode()

    out = tmp_path / "verdict.json"
    # Measured on no class: given the whole share, as every such file is, and no need said, as none is known.
    assert ci.gpu_lane("3.12", ["tests/test_long.py#1/2"], out, cache="on", processes=1, deadline=600.0) == 0
    [row] = json.loads(out.read_text())["rows"]
    assert float(lane_runs(tmp_path, "test_long")[-1][0]) == 0.5 and "admitted at" not in row["output"]
    # Its group 1/2 measured small under an earlier cut; the whole file at 8 GiB.
    measured("3.12:tests/test_long.py#1/2", gib // 4)
    measured("3.12:tests/test_long.py", 8 * gib)
    assert ci.gpu_lane("3.12", ["tests/test_long.py#1/2"], out, cache="on", processes=1, deadline=600.0) == 0
    [share, *_] = lane_runs(tmp_path, "test_long")[-1]
    assert float(share) * total == 8 * gib * 1.25 + gib, "the file's whole measure, not the group's"
    # Measured past the lane's 40 GiB: given all 40, its row saying its need was more.
    measured("3.12:tests/test_long.py", 48 * gib)
    assert ci.gpu_lane("3.12", ["tests/test_long.py#1/2"], out, cache="on", processes=1, deadline=600.0) == 0
    [row] = json.loads(out.read_text())["rows"]
    share = lane_runs(tmp_path, "test_long")[-1][0]
    assert float(share) == 0.5
    assert "admitted at 40.0 GiB, below its measured need of 61.0 GiB" in row["output"]


def test_gpu_holds_says_what_a_lane_needs_to_run_each_file_as_its_largest_measured_class_held_it(
        ci, tmp_path, monkeypatch, cache_worker, capsys):
    """`gpu_holds.py` says, for each test file (a group as its file), the most any GPU class's last whole run
    of it held with its children (`gpu_needs`), in bytes, so a GPU tier sizes each part's job by what its
    lane will admit its files at (2026-11-10: a part given a default 8 GiB had a 4 GiB lane for a file
    measured at 5.9 GiB); a file no class measured is absent."""
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
