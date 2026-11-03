"""A check-1 site run as armada's map and judge by Dew's CI (tools/armada/ci.py `sites`, `site_entries`): its
seeds sharded across map tasks that record their draws, and one judge that pools them, on small test files
written here: the plan, and each task as armada starts it, in the directories armada names it; armada's own
suite carries its runner's half, the judges' wait on their map tasks and their grading."""

import json
import re
import shutil
import sys
from pathlib import Path

import pytest
from armada_support import cached, interpreter, loaded, written
from tool_support import git

ROOT = Path(__file__).resolve().parents[1]

SITE = """import os
import sys
from pathlib import Path

import pytest


def test_light():
    with Path("runs-light").open("a") as runs:
        runs.write("x")


@pytest.mark.check1_shards(seeds=4, per_task=8)
def test_site():
    # A check-1 site as the gate runs one (tests/reference_error.py): a map task draws its seeds into its
    # records, and the judge pools every map task's, refusing a pool that misses a seed.
    if os.environ.get("DEW_CHECK1_DRAWS"):
        sys.path.insert(0, "src")
        from draws import drawn

        start, stop = (int(part) for part in os.environ["DEW_CHECK1_DRAWS"].split(":"))
        records = Path(os.environ["DEW_CHECK1_RECORDS"])
        records.mkdir(parents=True, exist_ok=True)
        (records / f"site.{start}-{stop}.pkl").write_text(drawn(start, stop))
        with Path("runs-draws").open("a") as runs:
            runs.write("x")
        return
    with Path("runs-judge").open("a") as runs:
        runs.write("x")
    held = sorted(tuple(int(seed) for seed in path.name.split(".")[1].split("-"))
                  for path in Path(os.environ["DEW_CHECK1_POOL"]).rglob("site.*.pkl"))
    assert [start for start, _ in held] == [0, *(stop for _, stop in held[:-1])] and held[-1][1] == 20, held
"""
"""A file of a light test and a site of 4 declared seeds, 8 draws a map task: its maps draw seeds 0 to 20, the
seeds of check 1's two stages, from src/draws.py, which its judge never runs."""

WITNESS = SITE.replace("@pytest.mark.check1_shards(seeds=4, per_task=8)",
                       "@pytest.mark.shards(units=20, per_task=8)")
"""The same file, its site work of 20 units of its own, 8 a map task (`shards`): its maps take units 0 to 20,
given them as a check-1 site's map tasks are given their seeds."""
KINDS = pytest.mark.parametrize("site", [SITE, WITNESS], ids=["check1_shards", "shards"])
"""Each kind of site, the same file."""
MARKED = pytest.mark.parametrize(("site", "per_unit"), [(SITE, 10.0), (WITNESS, 2.0)],
                                 ids=["check1_shards", "shards"])
"""Each kind of site, with the seconds a unit weighs before armada times its map tasks: the site's recorded
40 s over the 4 seeds a whole check-1 run declares, or over a `shards` test's 20 units."""

RED_SITE = """

@pytest.mark.check1_shards(seeds=2, per_task=5)
def test_red_site():
    assert os.environ.get("DEW_CHECK1_DRAWS") != "2:7", "this map task's draws failed"
"""
"""A second site of the file, of 10 draws, its first stage's 2 a map task and its second's 5, whose first
later map task fails."""

DRAWS = "def drawn(start, stop):\n    return ' '.join(str(seed) for seed in range(start, stop))\n"


@pytest.fixture
def ci(tmp_path, monkeypatch):
    return loaded(tmp_path, monkeypatch)


def site_checkout(root: Path, site: str = SITE) -> Path:
    """The site's file, the module its draws run and its durations record (the site 40 s, a whole run's of
    its 4 declared seeds), with the 3.12 environment: where its verdict is written, beside the checkout's
    entries."""
    written(root, {"test_site.py": site})
    (root / "src").mkdir()
    (root / "src/draws.py").write_text(DRAWS)
    (root / "tests/test_durations.json").write_text(json.dumps(
        {"tests/test_site.py::test_light": 1.0, "tests/test_site.py::test_site": 40.0}))
    interpreter(root, "3.12")
    (root / ".verdict").mkdir()
    return root / ".verdict/verdict.json"


def entry_task(ci, monkeypatch, entry: dict, out: Path, artifacts: Path | None = None,
               inputs: Path | None = None) -> int:
    """Plan entry `entry`'s task as armada starts it: .armada.json's command filled from the entry, with the
    directories armada names to every task (ARMADA_ARTIFACTS, ARMADA_INPUTS) here; its exit status."""
    command = json.loads((ROOT / ".armada.json").read_text())["task"]["command"]
    values = {**entry, "out": out}
    filled = [re.sub(r"\{(\w+)\}", lambda field, values=values: str(values[field[1]]), word)
              for word in command]
    for name, path in (("ARMADA_ARTIFACTS", artifacts), ("ARMADA_INPUTS", inputs)):
        if path is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, str(path))
    monkeypatch.setattr(sys, "argv", ["ci.py", *filled[2:]])
    return ci.main()


def planned(ci, target: float = 300.0) -> dict[str, dict]:
    """The plan's 3.12 entries by their task names."""
    return {entry["name"]: entry for entry in ci.plan(target, {}) if entry["python"] == "3.12"}


def runs(root: Path) -> list[int]:
    """How many times the light test, a map task's draws and the judge ran."""
    return [len(path.read_text()) if path.exists() else 0
            for path in (root / "runs-light", root / "runs-draws", root / "runs-judge")]


@MARKED
def test_a_marked_site_plans_a_map_task_each_per_task_draws_and_a_judge_that_waits_on_them(
        ci, tmp_path, monkeypatch, site, per_unit):
    """A test marked check1_shards(seeds=4, per_task=8) plans map tasks of at most 8 of the 20 seeds check
    1's two stages draw (seeds 0 to (1 + REDRAW) times 4), its first stage's 4 apart and its second's
    later, run only where its judge's first judgement asks for them (armada's `later`), and one marked
    shards(units=20, per_task=8) of at most 8 of its 20 units, none later, each its test's own group,
    `<test>.draws-<start>-<stop>`, and one judge waiting on them by their names, whose row is the site's,
    `<file>#<test>`; the file's own row sets the site aside, weighed and cut without its seconds. A map
    task weighs what armada measured it at, else its units at the site's recorded seconds over the units a
    whole run takes (a check-1 site's declared seeds, a `shards` test's units); the judge, its row's
    pooling as armada measured it, else the target; a cold plan adds what a cold task pays to each."""
    site_checkout(tmp_path, site)
    entries = [entry for entry in ci.plan(30.0, {}) if entry["python"] == "3.12"]
    check1 = site is SITE
    spans = [(0, 4), (4, 12), (12, 20)] if check1 else [(0, 8), (8, 16), (16, 20)]
    names = [ci.site_task("3.12", "tests/test_site.py", "test_site", span) for span in spans]
    assert [(entry["name"], entry["tests"], entry["split"], entry.get("rows"), entry.get("after"),
             entry["weight"], entry.get("later", False)) for entry in entries] == [
        ("3.12-1", "tests/test_site.py", "", ["3.12:tests/test_site.py"], None, 1.0, False),
        *((name, "tests/test_site.py", f"test_site.draws-{start}-{stop}",
           [f"3.12:tests/test_site.py#test_site.draws-{start}-{stop}"], None, per_unit * (stop - start),
           check1 and start >= 4)
          for name, (start, stop) in zip(names, spans, strict=True)),
        (ci.site_task("3.12", "tests/test_site.py", "test_site"), "tests/test_site.py", "test_site", None,
         names, 30.0, False)]
    assert all(re.fullmatch(r"[A-Za-z0-9._-]{1,80}", entry["name"]) for entry in entries)
    # Weighed by what armada measured: a map task under its task's name, the judge under its row's.
    measured = {names[0]: 7.0, "3.12:tests/test_site.py#test_site": 3.0}
    weighed = {entry["name"]: entry["weight"] for entry in ci.plan(30.0, {"rows": measured})}
    assert [weighed[name] for name in names] == [7.0, *(per_unit * (stop - start)
                                                          for start, stop in spans[1:])]
    assert weighed[ci.site_task("3.12", "tests/test_site.py", "test_site")] == 3.0
    monkeypatch.setenv("DEW_TEST_CACHE", "off")
    cold = {entry["name"]: entry["weight"] for entry in ci.plan(30.0, {}) if entry["python"] == "3.12"}
    assert [cold[name] for name in names] == [(stop - start) * per_unit + ci.COLD_SECONDS
                                              for start, stop in spans]


def test_a_weighed_sites_units_are_map_tasks_of_their_own_past_its_unweighted_ones(ci, tmp_path):
    """A check-1 site that declares a weighed check, check1_shards(seeds=4, per_task=8, conditioning=2,
    conditioning_per_task=3), plans its unweighted units as without it (0 to 20, in tasks of at most 8) and
    its weighed check's 10 units past them (20 to 30, (1 + REDRAW) times 2) in tasks of at most 3, every one
    later (armada's `later`, drawn only where its judge asks), each unmeasured task of them weighing the
    target, and one judge waiting on all of them."""
    written(tmp_path, {"test_site.py": "import pytest\n"
                                       "@pytest.mark.check1_shards(seeds=4, per_task=8, conditioning=2, "
                                       "conditioning_per_task=3)\n"
                                       "def test_site():\n    pass\n"})
    assert ci.sites("tests/test_site.py") == {"test_site": ci.Mark(units=20, per_task=8, line=2, timed=4,
                                                                   weighed=10, weighed_per_task=3)}
    entries = [entry for entry in ci.plan(30.0, {}) if entry["python"] == "3.12" and entry["split"]]
    spans = [(0, 4), (4, 12), (12, 20), (20, 23), (23, 26), (26, 29), (29, 30)]
    maps = [entry for entry in entries if "after" not in entry]
    assert [(entry["split"], entry.get("later", False)) for entry in maps] == [
        (f"test_site.draws-{start}-{stop}", start >= 4) for start, stop in spans]
    assert all(entry["weight"] == 30.0 for entry in maps[3:])
    (judge,) = [entry for entry in entries if "after" in entry]
    assert judge["after"] == [ci.site_task("3.12", "tests/test_site.py", "test_site", span) for span in spans]


def test_a_mark_the_plan_cannot_read_from_its_tests_decorator_is_refused(ci, tmp_path):
    """The plan reads a site's mark from the source, on the test's own decorator of literal counts, one a
    test: a mark of either kind written any other way, which its file's tasks would set aside with no task of
    its own planned, so that nothing ran its site, is refused, naming its line; a `shards` that is no mark (a
    mesh's) is no site. A class's method marked on its own decorator is a site, `Class.test`, whose tasks
    run its node alone."""
    test = "def test_site():\n    pass\n"
    for marks in ("pytestmark = pytest.mark.check1_shards(seeds=4, per_task=8)\n",
                  "pytestmark = pytest.mark.shards(units=20, per_task=8)\n",
                  "SEEDS = 4\n@pytest.mark.check1_shards(seeds=SEEDS, per_task=8)\n",
                  "UNITS = 20\n@pytest.mark.shards(units=UNITS, per_task=8)\n",
                  "@pytest.mark.check1_shards(seeds=4, per_task=0)\n",
                  "@pytest.mark.shards(units=0, per_task=8)\n",
                  "@pytest.mark.shards(seeds=4, per_task=8)\n",
                  "@pytest.mark.check1_shards(units=20, per_task=8)\n",
                  "@pytest.mark.shards\n",
                  "@pytest.mark.check1_shards(seeds=4, per_task=8)\n"
                  "@pytest.mark.shards(units=20, per_task=8)\n"):
        written(tmp_path, {"test_site.py": f"import pytest\n{marks}{test}"})
        with pytest.raises(SystemExit):
            ci.plan(30.0, {})
    written(tmp_path, {"test_site.py": f"import pytest\nmesh = layout.shards\n"
                                       f"@pytest.mark.shards(units=20, per_task=8)\n{test}"})
    assert ci.sites("tests/test_site.py") == {"test_site": ci.Mark(units=20, per_task=8, line=3, timed=20)}
    written(tmp_path, {"test_site.py": "import pytest\nclass TestSite:\n"
                                       "    @pytest.mark.check1_shards(seeds=4, per_task=8)\n"
                                       "    def test_site(self):\n        pass\n"
                                       "    @pytest.mark.shards(units=30, per_task=8)\n"
                                       "    def test_work(self):\n        pass\n"})
    assert ci.sites("tests/test_site.py") == {
        "TestSite.test_site": ci.Mark(units=20, per_task=8, line=3, timed=4),
        "TestSite.test_work": ci.Mark(units=30, per_task=8, line=6, timed=30)}
    judges = [entry for entry in ci.plan(30.0, {}) if entry["python"] == "3.12" and "after" in entry]
    assert [(entry["split"], len(entry["after"])) for entry in judges] == [("TestSite.test_site", 3),
                                                                          ("TestSite.test_work", 4)]
    assert ci.pytest_command(["pytest"], ["tests/test_site.py"], "TestSite.test_work.draws-24-30", "") == [
        "pytest", "-p", "executed", "tests/test_site.py::TestSite::test_work"]


def test_sites_past_one_judges_jobs_inputs_together_are_planned_and_one_past_it_alone_is_refused(
        ci, tmp_path, monkeypatch):
    """armada cuts the judges of one environment and size into as many judges' jobs as keep each within
    the map tasks' artifacts one job takes (MAX_INPUTS, armada's MAX_INPUT_RUNS; its `judgeChunks`), so
    marks whose map tasks pass it together are planned, every judge waiting on all of its own (Dew's
    base/kernels, 2026-11-02: 1,050 map tasks were refused). A site whose map tasks alone pass it would be
    a judge no job takes, and is refused by the plan before any task runs, naming its line."""
    def marked(mark: str) -> str:
        return f"import pytest\n\n@pytest.mark.{mark}\ndef test_site():\n    pass\n"

    # Two sites a Python, each within one job's inputs and past them together on every Python.
    seeds = ci.MAX_INPUTS // (2 * (1 + ci.REDRAW))
    units = ci.MAX_INPUTS - 1
    written(tmp_path, {"test_a.py": marked(f"check1_shards(seeds={seeds}, per_task=1)"),
                       "test_b.py": marked(f"shards(units={units}, per_task=1)")})
    # Cold, as a policy-off run plans every site on every Python.
    monkeypatch.setenv("DEW_TEST_CACHE", "off")
    judges = [entry for entry in ci.plan(120.0, {}) if "after" in entry]
    assert len(judges) == 2 * len(ci.PYTHONS)
    assert sum(len(entry["after"]) for entry in judges) > ci.MAX_INPUTS
    assert max(len(entry["after"]) for entry in judges) <= ci.MAX_INPUTS
    written(tmp_path, {"test_b.py": marked(f"shards(units={ci.MAX_INPUTS + 1}, per_task=1)")})
    with pytest.raises(SystemExit):
        ci.plan(120.0, {})
    assert ci.sites("tests/test_b.py")["test_site"].line == 3


def test_a_map_task_draws_into_the_artifacts_armada_names_it_and_exits_1_where_its_draws_fail(
        ci, tmp_path, monkeypatch):
    """A map task draws its seeds into the artifacts directory armada names it (ARMADA_ARTIFACTS); one whose
    draws fail writes its red row and exits 1, which armada reads as a map task that did not pass."""
    out = site_checkout(tmp_path, SITE + RED_SITE)
    entries = planned(ci)
    first = entries[ci.site_task("3.12", "tests/test_site.py", "test_site", (0, 4))]
    assert entry_task(ci, monkeypatch, first, out, artifacts=tmp_path / ".verdict/first") == 0
    assert sorted(path.name for path in (tmp_path / ".verdict/first").iterdir()) == ["site.0-4.pkl"]
    red = entries[ci.site_task("3.12", "tests/test_site.py", "test_red_site", (2, 7))]
    assert entry_task(ci, monkeypatch, red, out, artifacts=tmp_path / ".verdict/red") == 1
    [row] = json.loads(out.read_text())["rows"]
    assert (row["name"], row["exitCode"], row["kind"]) == (
        "3.12:tests/test_site.py#test_red_site.draws-2-7", 1, "nodes")


def test_a_judge_judges_its_map_tasks_pooled_draws_from_its_inputs_and_refuses_a_pool_short_of_them(
        ci, tmp_path, monkeypatch):
    """Check 1 itself (tests/reference_error.py), as a task runs it: a product on par, marked to draw its 80
    seeds 40 a map task, plans two map tasks, each drawing its seeds into its artifacts and judging nothing,
    and a judge that pools whatever its inputs hold (ARMADA_INPUTS) and passes, the site's row green; the
    file's own row runs the rest of the file alone. A judge whose inputs miss a map task's seeds is refused
    (the gate's PoolRefused), never judged."""
    from check1_statistics import REDRAW
    from map_judge import DRAWS, POOL, RECORDS

    assert (ci.DRAWS, ci.RECORDS, ci.POOL, ci.REDRAW) == (DRAWS, RECORDS, POOL, REDRAW)
    out = site_checkout(tmp_path, f"""import sys
from pathlib import Path

sys.path.insert(0, {str(ROOT / "tests")!r})
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact


def draw(seed):
    rng = np.random.default_rng([2641, seed])
    return (jnp.asarray(rng.standard_normal((16, 64)), jnp.float32),
            jnp.asarray(rng.standard_normal((64, 8)), jnp.float32))


def test_light():
    with Path("runs-light").open("a") as runs:
        runs.write("x")


@pytest.mark.check1_shards(seeds=16, per_task=40)
def test_site():
    assert_as_exact(jnp.matmul, jnp.matmul, draw, "a product sharded across map tasks", seeds=16)
""")
    entries = planned(ci)
    judge = entries[ci.site_task("3.12", "tests/test_site.py", "test_site")]
    inputs = tmp_path / ".verdict/inputs"
    for index, name in enumerate(judge["after"]):
        assert entry_task(ci, monkeypatch, entries[name], out, artifacts=inputs / "job" / str(index)) == 0
    assert entry_task(ci, monkeypatch, entries["3.12-1"], out) == 0
    [own] = json.loads(out.read_text())["rows"]
    assert entry_task(ci, monkeypatch, judge, out, inputs=inputs) == 0
    [row] = json.loads(out.read_text())["rows"]
    assert [(found["name"], found["exitCode"], sorted(found["tests"])) for found in (own, row)] == [
        ("3.12:tests/test_site.py", 0, ["tests/test_site.py::test_light"]),
        ("3.12:tests/test_site.py#test_site", 0, ["tests/test_site.py::test_site"])]
    shutil.rmtree(inputs / "job/0")
    assert entry_task(ci, monkeypatch, judge, out, inputs=inputs) == 0
    [row] = json.loads(out.read_text())["rows"]
    assert (row["name"], row["exitCode"]) == ("3.12:tests/test_site.py#test_site", 1)
    assert row["first_failure"].partition(":")[0].rpartition(".")[2] == "PoolRefused"


def test_a_judge_asks_for_its_later_draws_only_where_its_first_stage_finds_something(ci, tmp_path,
                                                                                     monkeypatch):
    """Check 1 as armada runs a site with later map tasks: the first stage's map task draws the declared
    seeds, and the judge over them alone passes where the first stage passes, its row green and asking
    nothing, and where it finds something writes its ask (LATER, `ask_later`) and its row is red and asks
    for more (armada's `more`); once the later map tasks have drawn, the judge over them all judges as one
    that drew every seed (Main, 2026-11-02: the second stage drawn every run, read only where the first
    found something)."""
    out = site_checkout(tmp_path, f"""import sys

sys.path.insert(0, {str(ROOT / "tests")!r})
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact


def draw(seed):
    rng = np.random.default_rng([2641, seed])
    return (jnp.asarray(rng.standard_normal((16, 64)), jnp.float32),
            jnp.asarray(rng.standard_normal((64, 8)), jnp.float32))


def halved(x, w):
    return (x.astype(jnp.bfloat16) @ w.astype(jnp.bfloat16)).astype(jnp.float32)


@pytest.mark.check1_shards(seeds=16, per_task=40)
def test_on_par():
    assert_as_exact(jnp.matmul, jnp.matmul, draw, "a product on par", seeds=16)


@pytest.mark.check1_shards(seeds=16, per_task=40)
def test_halved():
    assert_as_exact(halved, jnp.matmul, draw, "a halved product", seeds=16)
""")
    entries = planned(ci)

    def judged(test: str, drawn: list[dict]) -> dict:
        inputs = tmp_path / f".verdict/inputs-{test}-{len(drawn)}"
        for index, entry in enumerate(drawn):
            assert entry_task(ci, monkeypatch, entry, out, artifacts=inputs / "job" / str(index)) == 0
        judge = entries[ci.site_task("3.12", "tests/test_site.py", test)]
        assert entry_task(ci, monkeypatch, judge, out, artifacts=tmp_path / f".verdict/judge-{test}",
                          inputs=inputs) == 0
        [row] = json.loads(out.read_text())["rows"]
        return row

    for test in ("test_on_par", "test_halved"):
        judge = entries[ci.site_task("3.12", "tests/test_site.py", test)]
        maps = [entries[name] for name in judge["after"]]
        assert [entry.get("later", False) for entry in maps] == [False, True, True]
        first = judged(test, maps[:1])
        if test == "test_on_par":
            assert (first["exitCode"], first.get("more")) == (0, None)
            continue
        assert (first["exitCode"], first.get("more")) == (1, True)
        assert first["first_failure"].partition(":")[0].rpartition(".")[2] == "LaterUnits"
        asked = json.loads((tmp_path / ".verdict/judge-test_halved/check1-later.json").read_text())
        assert (asked["from"], asked["to"]) == (16, 80)
        whole = judged(test, maps)
        assert (whole["exitCode"], whole.get("more")) == (1, None)
        assert whole["first_failure"].partition(":")[0].rpartition(".")[2] == "GateFailure"


def run_cached(ci, tmp_path: Path, monkeypatch, cache_worker, site: str, later: bool) -> None:
    """Site `site`'s run as armada runs it with the test cache on, each task recording its rows: its map
    tasks, its later ones too where `later` (its judge's first stage asked for them), then its judge over
    those, and the file's own row."""
    out = site_checkout(tmp_path, site)
    url, _, _ = cache_worker
    monkeypatch.setattr(ci, "cpu_class", lambda: "GenuineIntel-6-143")
    cached(monkeypatch, url)
    monkeypatch.setenv("DEW_XLA_CACHE_READ_TOKEN", "token")
    config = json.loads((ROOT / ".armada.json").read_text())
    config["task"]["cpu"] = {"vendor": "GenuineIntel", "family": 6, "model": 143}
    (tmp_path / ".armada.json").write_text(json.dumps(config))
    entries = list(planned(ci).values())
    [judge] = [entry for entry in entries if "after" in entry]
    inputs = tmp_path / ".verdict/inputs"
    drawn = [entry for name in judge["after"] for entry in entries
             if entry["name"] == name and (later or not entry.get("later", False))]
    for index, entry in enumerate(drawn):
        assert entry_task(ci, monkeypatch, entry, out, artifacts=inputs / "job" / str(index)) == 0
    for entry in entries:
        if not entry["split"] or entry is judge:
            assert entry_task(ci, monkeypatch, entry, out, inputs=inputs) == 0


def replanned_once_its_draws_change(ci, tmp_path: Path) -> list[dict]:
    """The site's file kept whole by the plan after its cached run, its site among it, none of its tasks
    run; then, a module only its draws ran (src/draws.py) changed, its site planned again and the file's
    own row kept: the entries the plan then runs."""
    kept, ran = planned_now(ci)
    assert kept == {("tests/test_site.py", ""), ("tests/test_site.py", "test_site")} and ran == []
    (tmp_path / "src/draws.py").write_text(DRAWS.replace("' '", "','"))
    kept, ran = planned_now(ci)
    assert kept == {("tests/test_site.py", "")}
    return ran


def planned_now(ci) -> tuple[set[tuple[str, str]], list[dict]]:
    """The rows the plan keeps, and the entries it runs, each holding a row no kept task reads."""
    entries = list(planned(ci).values())
    kept = {held for entry in entries for held in ci.kept_of(entry["tests"].split())[1]}
    return kept, [entry for entry in entries if ci.KEPT_AT not in entry["tests"]]


@KINDS
def test_a_kept_judge_plans_no_map_task_and_its_key_covers_what_its_draws_ran(
        ci, tmp_path, monkeypatch, cache_worker, site):
    """A site's judge green on exactly these inputs is kept whole, a check-1 site's and a `shards` test's
    alike: the plan runs none of its map tasks and not its judge, whose row a kept task reads. Its key is the
    site's, its map tasks' inputs (each among its artifacts) with its own, so a module only its draws ran
    (src/draws.py), changed, plans that site's map tasks and judge again and nothing else: the file's own
    row, which never ran it, stays kept."""
    run_cached(ci, tmp_path, monkeypatch, cache_worker, site, later=True)
    assert runs(tmp_path) == [1, 3, 1]
    ran = replanned_once_its_draws_change(ci, tmp_path)
    [again] = [entry for entry in ran if "after" in entry]
    maps = [entry for entry in ran if "after" not in entry]
    assert {ci.SITE.fullmatch(entry["split"])["test"] for entry in ran} == {"test_site"}
    assert maps and set(again["after"]) == {entry["name"] for entry in maps}


def test_a_judge_green_on_its_first_stage_is_kept_though_its_later_map_tasks_never_ran(
        ci, tmp_path, monkeypatch, cache_worker):
    """A check-1 site whose judge's first stage finds nothing, as armada runs it: its later map tasks never
    run, and its judge, green over its first stage's draws alone, keys on those map tasks' inputs with its
    own, so it is kept whole on them: the plan runs none of its map tasks and not its judge. A module its
    draws ran (src/draws.py), changed, plans them again."""
    run_cached(ci, tmp_path, monkeypatch, cache_worker,
               SITE.replace("held[-1][1] == 20", "held[-1][1] == 4"), later=False)
    assert runs(tmp_path) == [1, 1, 1]
    ran = replanned_once_its_draws_change(ci, tmp_path)
    [again] = [entry for entry in ran if "after" in entry]
    assert len(again["after"]) == 3 and len(ran) == 4


def test_a_sites_judge_and_map_tasks_leave_its_files_cut_and_recorded_times_as_they_were(ci, tmp_path):
    """The durations fold sets a site's judge row aside, its pooling no measure of its draws, and keeps the
    site's recorded seconds, which weigh its map tasks before armada times them; a judge armada did not run
    (a row of its task's name) is no file's. The file's cut reads its own rows, neither a map task's seconds,
    which armada keeps under the task's name, nor its judge's, and a Python without them borrows none. The
    verdict is of the checkout's own commit, so the fold reads its file as unchanged since the verdict ran."""
    site_checkout(tmp_path)
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "tests")
    git(tmp_path, "commit", "-qm", "the site")
    record = json.loads((tmp_path / "tests/test_durations.json").read_text())
    judge = ci.site_task("3.12", "tests/test_site.py", "test_site")
    verdict = {"sha": git(tmp_path, "rev-parse", "HEAD").strip(), "rows": [
        {"name": "3.12:tests/test_site.py", "exitCode": 0, "seconds": 2.0,
         "tests": {"tests/test_site.py::test_light": 2.0}},
        {"name": "3.12:tests/test_site.py#test_site", "exitCode": 0, "seconds": 5.0,
         "tests": {"tests/test_site.py::test_site": 5.0}},
        {"name": judge, "exitCode": 1, "output": "its map tasks did not all pass"}]}
    assert ci.folded(record, verdict, 30.0) == record
    verdict["rows"][0]["tests"]["tests/test_site.py::test_new"] = 1.0
    assert ci.folded(record, verdict, 30.0) == {**record, "tests/test_site.py::test_light": 2.0,
                                                "tests/test_site.py::test_new": 1.0}
    timings = {"rows": {"3.12:tests/test_site.py": 2.0, "3.12:tests/test_site.py#test_site": 5.0,
                        ci.site_task("3.12", "tests/test_site.py", "test_site", (0, 8)): 70.0}}
    assert ci.measured_cut(timings, "3.12", "tests/test_site.py") == (2.0, 1)
    assert ci.borrowed(timings, "3.14")["rows"] == {**timings["rows"], "3.14:tests/test_site.py": 2.0,
                                                    "3.14:tests/test_site.py#test_site": 5.0}


def test_a_sites_map_task_past_the_target_is_reported_though_its_test_is_recorded(ci):
    """A check-1 site's map task that runs twice the target is a slow test to report, though main's record
    holds the site's test (its whole run's time): it names its map task and the fix, fewer units a task,
    where the site's judge and a recorded test of a file's own row pass unreported (row 3729: a map task ran
    1,366 s, and nothing said so)."""
    test = "tests/test_site.py::test_site"
    rows = [{"name": "3.12:tests/test_site.py#test_site.draws-0-512", "tests": {test: 240.0}},
            {"name": "3.12:tests/test_site.py#test_site.draws-512-1024", "tests": {test: 60.0}},
            {"name": "3.12:tests/test_site.py#test_site", "tests": {test: 240.0}},
            {"name": "3.12:tests/test_other.py", "tests": {"tests/test_other.py::test_old": 240.0}}]
    base = {test: 300.0, "tests/test_other.py::test_old": 240.0}
    assert ci.slow({"rows": rows}, 120.0, base) == [
        {"test": test, "file": "tests/test_site.py", "seconds": 240.0, "target": 120.0,
         "map_task": "3.12:tests/test_site.py#test_site.draws-0-512",
         "fix": "fewer units a task (per_task, conditioning_per_task)"}]
