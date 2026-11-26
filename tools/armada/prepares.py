"""A GPU tier file's dry prepare on the CPU (ci.py's `dry_prepared`, `entries`): the tier's own prepare
of the file, for one compile-only GPU of a generation, before any GPU is drawn for it. Its rows' process
variants, the files and generations prepared, how a slow file's prepare is cut, and how a row is graded
from the tests it left unprepared. The standard library alone, loaded by ci.py by its path: ci.py hands
`entries`, `before` and `run` itself (`ci`), whose test cache and lane's processes they run on."""

import ast
import functools
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path

PREPARE = "prepare-"
"""A dry prepare's row variant (ci.py's `varied`), its GPU generation after it, `-users` after that for the
process that computes as a user's does (ci.py's `VARIANTS`): `<file>#@prepare-sm90`,
`<file>#1/2@prepare-sm90-users`."""
USERS = "-users"
"""What ends a dry prepare's variant whose process computes as a user's does (`variant`)."""
TOKEN = "prepare:"
"""What begins a dry prepare task's tests (its file, or a kept row's token): no file name, so the land
queue's targeted runs, which read a plan's entries by their files, take no prepare for a run of a file."""
PREPARED = ("sm90",)
"""The GPU generations each GPU tier file is prepared dry for (Main, 2026-11-22: sm90 first; sm80 once sm90's
cuts are timed, its first cuts seeded from them)."""
PREPARE_SECONDS = 600.0
"""A dry prepare's weight where armada timed none of its row: the longest file's, tests/test_gpu_models.py's
for sm90, 580 s on an armada container (2026-11-22; 80 to 322 s on a GPU node's CPU)."""
PREPARE_TARGETS = 4
"""How many of the plan's targets a dry prepare's task may take before its file is prepared in groups
(`groups_of`), as the tier's run cuts it, each group a task of its own."""
PREPARE_WALL = 600.0
"""A dry prepare task's shared budget (`run`), whatever its deadline: its topology, baseline lookups,
processes and their records' publication draw on it (a baseline unwritten by its end leaving its row stale,
row 13014); its restorations, local git calls, may run past it. It sits under a train's wall bound, 626 to 696
s (its 900 s target less its warm-up and plan, its finish and a margin). Its groups are cut to its cap
(`cap_after`), which holds a group however unevenly its count cut (`prepared_cut`) spreads its work (rows
12873, 12429-12439, 12643, 12759: groups cut to 480 s, capped only at their task's 1,680 s, ran 665 to 1,650 s
and set their trains' walls)."""
CUT_SHORT = 0.75
"""The share of a dry prepare's full cap (`cap_after` of `PREPARE_WALL`) past which a timed row ran out of its
time (`planned`): what it took is no more than a floor of its work, which counts twice. A tuning heuristic,
no bound: a process starts late where its task's topology, baseline lookup and earlier rows drew on the shared
budget, its cap that much earlier (a cold attention cut capped at 217 s, 2026-11-25), and a row that
completes may take past it too, counting twice all the same."""
TOPOLOGY_SCRIPT = ("import sys; from pathlib import Path; sys.path.insert(0, 'tests'); "
                   "import generation_support; "
                   "generation_support.node_topology(Path(sys.argv[1]), sys.argv[2])")
"""Writes a node's serialized topology of one compile-only GPU of a generation (`node_topology`)."""
VARIANTS = {"users": ("users_process", {"DEW_TEST_USERS_ARITHMETIC": "1"})}
"""A GPU lane's processes of a file beside its own, by name: the mark of the tests one runs and the
environment it runs under. XLA reads a process's flags once, so tests that must open under a user's flags
(no deterministic ops, JAX's own matmul precision) are a process of their own: under its word the suite's
conftest runs the marked GPU tests and skips the others, as without it it skips the marked ones, so both
processes select and cut a file alike and each test runs in one."""

RECIPE = ("tools/armada/prepares.py", "tests/tier_wire.py", "tests/generation_support.py",
          "tools/armada/recycle.py")
"""What a dry prepare runs beside the tests it records (the topology's writer runs in a process of its own),
which keys its rows besides every row's shared inputs (ci.py's `prepare_shared`)."""


def _is_mark(node: ast.AST, mark: str) -> bool:
    """Whether `node` names pytest's mark `mark` (`pytest.mark.<mark>`, called or not)."""
    node = node.func if isinstance(node, ast.Call) else node
    return (isinstance(node, ast.Attribute) and node.attr == mark and isinstance(node.value, ast.Attribute)
            and node.value.attr == "mark" and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "pytest")


@functools.cache
def marked(text: str, mark: str) -> bool:
    """Whether a test file's source `text` names pytest's mark `mark` anywhere in its code
    (`pytest.mark.<mark>`: a module's pytestmark, a decorator, a parameter's marks), as the plan reads marks
    from the source, importing no test; a mark named only in a string is none."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return any(_is_mark(node, mark) for node in ast.walk(tree))


@functools.cache
def marked_tests(text: str, mark: str) -> frozenset[str]:
    """The test functions of a test file's source `text` that carry pytest's mark `mark`, each as its node id
    names it past its file (`test_a`, `TestB::test_b`): every one where the module's pytestmark names it, else
    each whose own decorators or its class's do; a parameter's mark marks its function."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return frozenset()
    whole = any(isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets)
                and any(_is_mark(each, mark) for each in ast.walk(node.value)) for node in tree.body)

    def tests(body: list[ast.stmt], held: bool, prefix: str) -> set[str]:
        found = set()
        for node in body:
            if isinstance(node, ast.ClassDef):
                found |= tests(node.body, held or any(_is_mark(each, mark) for each in node.decorator_list),
                               f"{prefix}{node.name}::")
            elif (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")
                  and (held or any(_is_mark(each, mark) for decorator in node.decorator_list
                                   for each in ast.walk(decorator)))):
                found.add(prefix + node.name)
        return found
    return frozenset(tests(tree.body, whole, ""))


def gpu_tests(name: str, text: str, nodes: list[str]) -> int:
    """How many tests of test file `name` (its source `text`) are marked `gpu`: its recorded node ids
    (`nodes`, the durations record's) of such a function, each parameter its own, else those functions;
    at least one. What a dry prepare's file is cut into before any run of it is timed (`planned`)."""
    functions = marked_tests(text, "gpu")
    recorded = [node for node in nodes if node.partition("::")[2].partition("[")[0] in functions]
    return max(len(recorded) or len(functions), 1)


def prepared_cut(nodes: list[str], count: int) -> list[list[str]]:
    """A dry prepare's `count` groups of a test file's `nodes` (node-id order): runs of consecutive tests
    whose sizes differ by at most one, each group to one run while there are tests (no group empty a test
    could fill). Cut by count: a selected test prepares wherever its check-1 sites are (tests/conftest.py's,
    `gpu`-marked or not), and the durations record times one at nothing on the CPU, which skips it, so a cut
    by those seconds is as light any way it packs them (train 1247: 18 flows cut 5, 13 and 16 empty
    groups). What split.py's task of the group keeps (`session.SPLIT_EVENLY`) and ci.py's key holds
    (`members`), each of the nodes it has: its collection's, or the record's."""
    runs = min(count, len(nodes))
    groups: list[list[str]] = [[] for _ in range(count)]
    for index in range(runs):
        groups[index] = nodes[index * len(nodes) // runs:(index + 1) * len(nodes) // runs]
    return groups


def tier_files(files: list[str], nodes: Mapping[str, list[str]]) -> list[tuple[str, bool, int]]:
    """Of test `files`, those the GPU tier runs (a test marked `gpu`), each with whether it holds a test of
    a user's process (`users_process`) and how many tests are marked `gpu` (`gpu_tests`, of its recorded node
    ids, `nodes`)."""
    found = []
    for name in files:
        text = Path(name).read_text()
        if marked(text, "gpu"):
            found.append((name, marked(text, "users_process"), gpu_tests(name, text, nodes.get(name, []))))
    return found


def prepared_group(group: str, varied: str) -> bool:
    """Whether a row's `group` (ci.py's `varied`, its process variant after `varied`) is a dry prepare's."""
    return group.partition(varied)[2].startswith(PREPARE)


def variant(generation: str, users: bool) -> str:
    """A dry prepare's row variant for `generation`, a user's process's where `users`."""
    return f"{PREPARE}{generation}{USERS if users else ''}"


def parsed(row_variant: str) -> tuple[str, bool]:
    """`variant`'s generation and whether it is a user's process's."""
    named = row_variant.removeprefix(PREPARE)
    return named.removesuffix(USERS), named.endswith(USERS)


def groups_of(seconds: float | None, target: float, bound: float) -> int:
    """How many groups a file whose prepare took `seconds` (None: never timed) is prepared in: one, or as
    many as hold it within `PREPARE_TARGETS` of the plan's `target` each and a dry prepare's cap (`bound`,
    `cap_after`)."""
    budget = min(PREPARE_TARGETS * target, bound)
    return 1 if seconds is None else max(1, math.ceil(seconds / budget))


def process(ci, python: str, file: str, group: str, verdict: Path, left: float, *, unit: str, threads: str,
            users: bool, variant: str, cache: str = "off", common: str = "", recorded: bool = False,
            environ: Mapping[str, str] | None = None, kept: Mapping[str, tuple[str, str]] | None = None,
            judged: Callable[[list[dict]], list[dict]] | None = None, capped: bool = False,
            written: list[set[str]] | None = None) -> int:
    """One process of a GPU tier part's prepare (ci.py's `gpu_prepared`) or a dry prepare (`run`): test file
    `file`'s `group` run on the CPU alone under the prepare's word and its unit's name (`unit`), a user's
    process's flags where `users` (`VARIANTS`), its row named for `variant`, within what is `left` of its
    caller's deadline. What its own end may take past its deadline is kept back (its grace, its group's
    settling, its reader's last ten seconds); with none left, it is not started, its row saying so. A dry
    prepare's (`capped`) runs under its time cap and its memory's (tests/tier_wire.py's `PREPARE_BY` and
    `PREPARE_HOLDS`), its records published within what is `left` (`tested`'s `publish_by`); a part's never,
    whose run reads every program it prepares."""
    if (mine := left - ci.session.GRACE - ci.SETTLE - 10.0) <= 0:
        why = "not prepared: its lane's prepare had no time left for it"
        verdict.write_text(json.dumps({"rows": [ci.stale_row(python, file, ci.varied(group, variant), why)]}))
        return 0
    wire = ci.tier_wire()
    # Past its cap a dry prepare starts no further test or cell (tests/tier_wire.py's `past_cap`), so what it
    # prepared is kept, its end in time to upload it: a cap of the time it has but its end's.
    # A part's clears any cap it inherits (its environment's, its caller's).
    # Of memory, a share of each limit binding it (`held_caps`), its end's the rest.
    cap = {wire.PREPARE_BY: repr(time.monotonic() + cap_after(ci, left)) if capped else "",
           wire.PREPARE_HOLDS: held_caps(ci) if capped else ""}
    # A dry prepare's tests run in as many processes as keep each under `RECYCLE_AT` (recycle.py).
    return ci.tested(python, [file], group, verdict, mine, cache, ci.session.GRACE, common,
                     {"JAX_PLATFORMS": "cpu", wire.PREPARE: "1", "OMP_NUM_THREADS": threads,
                      **(VARIANTS["users"][1] if users else {}), wire.UNIT: unit, **cap, **(environ or {})},
                     recorded=recorded, variant=variant, kept=kept, judged=judged,
                     recycle_at=RECYCLE_AT if capped else None, written=written,
                     publish_by=time.monotonic() + left if capped else None)


CAP_MARGIN = 240.0
"""What a dry prepare keeps of its time past its cap (`process`) for its process's end: the test or cell
running at its cap, its programs' upload to the shared cache, its report."""


def cap_after(ci, left: float) -> float:
    """The seconds into a dry prepare given `left` past which it starts no further test (`process`'s cap):
    what it has but its end's (its grace, its group's settling, its reader's last ten seconds) and
    `CAP_MARGIN`, or half of that where the margin would take more."""
    mine = left - ci.session.GRACE - ci.SETTLE - 10.0
    return max(mine - CAP_MARGIN, mine / 2)


RECYCLE_AT = 3 << 30
"""The resident bytes past which a dry prepare's pytest ends between two tests, its task running the tests
left in a fresh process (`process`, recycle.py): a process keeps what its tests compiled until it ends (a
recurrence family 1-3.6 GB past every release), so a group of heavy families grew to its container's whole
memory. Under `PREPARE_HOLDS`' 75% of a 10 GiB container, with room for one family's growth past it."""


HELD_SHARE = 0.75
"""The share of each memory limit binding a dry prepare its level may hold in working set before it starts
nothing more (`process`, tests/tier_wire.py's `PREPARE_HOLDS`): the rest is the test, cell or draw running as
it passes it, its process's end and its programs' upload. A starting margin, not a bound: one compile's
allocation within a unit may outgrow it."""


def held_caps(ci) -> str:
    """The working set each cgroup level binding this task may hold in its dry prepare (tests/tier_wire.py's
    `PREPARE_HOLDS`): `HELD_SHARE` of the memory limit of its own cgroup and of each ancestor that states one,
    under DEW_CGROUP_ROOT (host.py's), its caps. Read here, in the task, so its pytest reads no
    limit, which would key every prepare row by its host's (followed.py's `LIMITS`); a JSON list of
    `[path, bytes]` pairs (a cgroup's name may hold any separator), "" where none states one."""
    reader, root = ci.host.cgroups(), ci.host.cgroup_root()
    caps = [[str(level), int(limit * HELD_SHARE)] for level, limit in reader.memory_limits(root)]
    return json.dumps(caps) if caps else ""


def planned(files: list[tuple[str, bool, int]], took: Callable[[str, str], tuple[float, float] | None],
            target: float, varied: str, bound: float) -> list[tuple[str, str]]:
    """The dry prepares' rows (`run`), each a file and its group: each GPU tier file of `files`
    (`tier_files`) for each generation of `PREPARED`, its lane's process and, where it holds a test of a
    user's process, that process's, each a row (`variant`, after `varied`, ci.py's `VARIED`). A file is
    prepared in groups of its tests (`G/N`, `prepared_cut`), each a row: as many as its prepare's work
    needs to hold each within `PREPARE_TARGETS` targets and a dry prepare's cap, `bound` (`cap_after`)
    (`took`, by file and variant: the rows ci.py's `measured_cut` sums, and the longest of them, past
    `CUT_SHORT` of the cap a floor that counts twice, as a capped row stops at it whatever is left), and
    before any run of it is timed one a `gpu`-marked test, so no first prepare of a file sets a train's
    wall (Main, 2026-11-22: tests/test_gpu_flows.py's ten flows, whole, ran past a task's 1,575 s). A
    timed file is cut as its work needs, never by a record's count of its tests, which a parametrization
    since may have outgrown: a group its task finds no prepared test for prepares nothing."""
    found = []
    for name, users, tests in files:
        for generation in PREPARED:
            processes = (False, True) if users else (False,)
            timed = [seconds * (2 if longest >= CUT_SHORT * bound else 1) for each in processes
                     if (measured := took(name, variant(generation, users=each))) is not None
                     for seconds, longest in [measured]]
            count = groups_of(max(timed), target, bound) if timed else tests
            cuts = [f"{index}/{count}" for index in range(1, count + 1)] if count > 1 else [""]
            found += [(name, f"{cut}{varied}{variant(generation, users=each)}")
                      for cut in cuts for each in processes]
    return found


def unprepared_in(truths: Path, records: tuple[str, ...],
                  owning: Callable[[str], str]) -> dict[str, list[dict]]:
    """The cells a prepare into store `truths` left unprepared, by test file (`owning` a test's node id): each
    reason of each of `records` (tests/tier_wire.py's `UNPREPARED`, and its `DEFERRED`, a flow's cell its
    prepare deferred to its run, which the run names unprepared), each its test, where, why and cell label (a
    deferred cell's), so the flow's cells deferred at one line are each their own."""
    found: dict[str, list[dict]] = {}
    for name in records:
        for path in sorted(truths.rglob(name)):
            record = json.loads(path.read_text())
            # A record that lists no reasons gives its own where and why, as its first's.
            for reason in record.get("reasons") or [record]:
                found.setdefault(owning(record["test"]), []).append(
                    {"test": record["test"], "where": reason.get("where", "?"),
                     "error": reason.get("error", "?"),
                     **({"cell": reason["cell"]} if reason.get("cell") else {})})
    return found


def identity(record: Mapping) -> str:
    """What a dry prepare's unprepared cell names (`graded`): its test, where in it, and a deferred cell's
    label, so one deferred cell of a test is no mask for a sibling deferred beside it, at one line or
    another."""
    named = f"{record['test']} @ {record.get('where') or '?'}"
    return f"{named} [{record['cell']}]" if record.get("cell") else named


def skipped_whole(record: Mapping, untried: set[str]) -> bool:
    """Whether a cell `record` is of a test main's run skipped whole (`untried`, node ids): its structured
    test, exactly; a reused row's identity alone only where it reads one way, a skipped node id, " @ ", then
    a remainder holding no " @ " (its where and any label, `identity`), so a parameter id that holds " @ "
    and shares a skipped one's start is never taken for it."""
    if "test" in record:
        return record["test"] in untried
    for test in untried:
        if record["identity"].startswith(f"{test} @ "):
            if " @ " not in record["identity"][len(test) + 3:]:
                return True
    return False


def as_ran(row: dict, stale: str) -> dict:
    """A dry prepare's `row`, red as its whole group as it ran (`group`), however it went red: the land queue
    grades such a row again by its group alone (ops queue/suspect_items.py), whose task runs it as a
    prepare again (ci.py's `task`, by its split's variant); its failed tests run alone would run with no
    prepare, every GPU test skipped. A row said `stale` (not run, or not graded) stays so: the queue runs
    it again."""
    return ({**row, "kind": "group", "failed": []} if row.get("exitCode") and row.get("kind") != stale
            else row)


def graded(rows: list[dict], unprepared: Mapping[str, list[dict]], before: set[str] | str, generation: str,
           file_of: Callable[[str], str], stale: str, capped_at: str) -> list[dict]:
    """A dry prepare's `rows` (`file_of` a row's name its test file), each keeping the cells it left
    unprepared (`identity`) as its `unprepared`: a row this run prepared, its file's records (`unprepared`,
    from `unprepared_in`); a row reused from its record, the list that record keeps. A row is red, its new
    cells its `new_unprepared`, where it left one unprepared that main's run of the row did not (`before`);
    `BOOTSTRAP`, main with no run of it, reds none; `UNREAD`, main's not read whole, says a row that left
    any unprepared `stale`, not graded and run again, never green. A reused row is graded again against the
    baseline that stands now, and every red row is its group's as it ran (`as_ran`). A cell its prepare's
    time cap left (where `capped_at`, tests/tier_wire.py's `CAPPED`) is named and counted (`capped`, kept
    with a reused row) but judged by neither, and a test main's run skipped whole at its cap (its node id at
    the cap, unlabeled: tests/conftest.py's skip) was never tried there, so this run's cells of it are held
    to nothing, as main's runs are where main has none; a cell or site main capped exempts no sibling."""
    judged = []
    for found in rows:
        own = unprepared.get(file_of(found["name"]), [])
        held = ([{"identity": cell} for cell in found["unprepared"]] if "unprepared" in found
                else [{**record, "identity": identity(record)} for record in own])
        cells = sorted(record["identity"] for record in held)
        capped = sorted(found.get("capped", []) if "unprepared" in found
                        else {identity(record) for record in own if record["where"] == capped_at})
        judged_ = [record for record in held if record["identity"] not in set(capped)]
        # Said once: a reused row's output holds its banner (`capped_banner`) already.
        if capped and (found.get("output") or "").startswith(head := capped_banner(capped)):
            found = {**found, "output": found["output"][len(head):].lstrip("\n")}
        if isinstance(before, set):
            # Main's decision_laya prepare capped its five tests, which a run that reached them failed
            # alike at main's commit (2026-11-25): no run of main's says how they go.
            whole = f" @ {capped_at}"
            untried = {cell[:-len(whole)] for cell in before if cell.endswith(whole)}
            new = [record for record in judged_ if record["identity"] not in before
                   and not skipped_whole(record, untried)]
            said = [f"unprepared in its {generation} prepare, which main's did not leave so: "
                    f"{record['identity']}" + (f": {record['error']}" if record.get("error") else "")
                    for record in new]
            red = ({"exitCode": 1, "output": "\n".join([*said, found.get("output") or ""]),
                    "new_unprepared": sorted(record["identity"] for record in new)} if new else {})
        elif before == UNREAD and judged_ and not found.get("exitCode"):
            red = {"exitCode": 1, "kind": stale, "failed": [], "output": (
                f"not graded: it left {len(judged_)} cells unprepared in its {generation} prepare, and "
                "main's runs of it were not read whole; it runs again")}
        else:
            red = {}
        said = ({"capped": capped, "output": "\n".join([capped_banner(capped), red.get(
            "output", found.get("output") or "")])} if capped else {})
        judged.append(as_ran({**found, "unprepared": cells, **red, **said}, stale))
    return judged


def capped_banner(capped: list[str]) -> str:
    """What a row whose prepare's cap left `capped` says first (`graded`)."""
    return "\n".join([
        f"{len(capped)} left unprepared by its prepare's cap, its time's or its memory's, each run "
        "unprepared on its GPU (its group's prepare is too long or holds too much: cut it finer or make its "
        "tests cheaper to prepare):", *capped])


def nothing_timed(row: dict) -> dict:
    """A dry prepare's `row` as armada times it: one whose group held no test (its cut left it none, a file
    of fewer tests than its work's groups, `planned`) green at no seconds, so its launch is no work a later
    plan cuts the file finer for; any other as it ran."""
    if row["exitCode"] != 0 or row.get("tests"):
        return row
    return {**row, "seconds": 0.0, "timings": dict.fromkeys(row.get("timings", {}), 0.0)}


def entries(ci, python: str, timings: dict, target: float, cache: str, stamped: dict) -> list[dict]:
    """The plan's dry prepares (`planned`): a row its record holds green on exactly these inputs (ci.py's
    `reusable`, over its `prepare_shared`) read by a kept task of such rows, as any row is; every other its
    own task, weighed by its last seconds or `PREPARE_SECONDS`, a cold one's more. Its tests begin `TOKEN`,
    so its task runs it as a prepare (`run`)."""
    grouped = ci.grouped_rows(timings.get("rows", {}))

    def took(name: str, row_variant: str) -> tuple[float, float] | None:
        groups = {group.partition(ci.VARIED)[0]: seconds
                  for group, seconds in grouped.get((python, name), {}).items()
                  if group.partition(ci.VARIED)[2] == row_variant}
        found = ci.cuts.measured_rows(groups)
        return None if found is None else (sum(found), max(found))
    parted = ci.run_keys.parted_record()
    files = tier_files(ci.test_files(), {name: list(tests) for name, tests in parted.items()})
    rows_ = planned(files, took, target, ci.VARIED, cap_after(ci, PREPARE_WALL))
    if not rows_:
        return []
    missed: dict[str, int] = {}
    found = ci.reusable(python, rows_, ci.prepare_shared(python), missed) if cache != "off" else {}
    print(f"plan: {python}: dry prepares kept {len(found)} of {len(rows_)}; not kept: {missed}",
          file=sys.stderr)
    measured = timings.get("rows", {})
    entries = [{"name": f"{python}-prepare-kept-{index}", "python": python, "split": "",
                "tests": " ".join(TOKEN + ci.kept_token(name, group, found[(name, group)])
                                  for name, group in held),
                "rows": [ci.row(python, name, group) for name, group in held], "weight": float(len(held)),
                **stamped}
               for index, held in enumerate(ci.kept_tasks(python, found, target), start=1)]
    return entries + [
        {"name": f"{python}-prepare-{index}", "python": python, "tests": TOKEN + name, "split": group,
         "rows": [ci.row(python, name, group)], **stamped,
         "weight": measured.get(ci.row(python, name, group), PREPARE_SECONDS)
         + (ci.COLD_SECONDS if cache != "on" else 0.0)}
        for index, (name, group) in enumerate(rows_, start=1) if (name, group) not in found]


BOOTSTRAP = "bootstrap"
"""A dry prepare's baseline where main has no run of its file's prepare at all (`before`): its first, which no
test it leaves unprepared reds."""
UNREAD = "unread"
"""A dry prepare's baseline where the Worker gave none whole within its time (`before`): no grade of the tests
it leaves unprepared, which says the row stale and runs it again."""
PARTIAL = "partial"
"""Main's baselines of a row read, but none whole (`_before`): a run named with no baseline, or every cut
listed short of a group. `before` gives it as `UNREAD`, but to a run of main's own commit, which is the
baseline (`BOOTSTRAP`)."""
RECORD_BYTES = 1 << 20
"""The most a baseline's record may hold (`before`): a row's record, its unprepared tests among it."""


class Unread(Exception):
    """The Worker gave no answer whole within the baseline's time (`before`)."""


def bounded_get(store, key: str, until: float, failures: tuple[type[BaseException], ...]) -> dict | None:
    """The record at `key` from `store`, read whole within `until` (a `time.monotonic` instant: the Worker's
    `request` bounds its every wait by it), an object, or None where there is none; `Unread` where the Worker
    failed (`failures`), its time ran out, or it gave no object."""
    try:
        body = store.get(key, limit=RECORD_BYTES, until=until)
        found = None if body is None else json.loads(body)
    except (*failures, ValueError) as failure:
        raise Unread(f"{key}: {type(failure).__name__}: {failure}") from failure
    if found is not None and not isinstance(found, dict):
        raise Unread(f"{key} holds no record")
    return found


def cells_of(record: dict | None) -> set[str] | None:
    """The cells a commit's baseline record (`baseline_key`, a green prepare's alone) keeps unprepared, None
    where there is no record; `Unread` where the record is no object of a list of cells (`identity`'s
    strings)."""
    if record is None:
        return None
    kept = record.get("unprepared")
    if not (isinstance(kept, list) and all(isinstance(cell, str) for cell in kept)):
        raise Unread("a baseline's unprepared cells are no list of cells")
    return set(kept)


def listed_cuts(store, prefix: str, row_variant: str, until: float) -> dict[str, list[str]]:
    """The cuts of a file's rows under `row_variant` whose records `prefix` lists (`before`), by their count
    of groups ("whole" for a whole file's row); `Unread` where the listing was read past `until`."""
    parts = list(store.keys(prefix, until=until))
    # A listing read past its time is no answer, empty or not.
    if time.monotonic() >= until:
        raise Unread("main's runs listed past their time")
    cuts: dict[str, list[str]] = {}
    for part in parts:
        cut, own, rest = part.rpartition(f"-as-{row_variant}")
        count = "whole" if cut == "whole" else cut.rpartition("-of-")[2]
        # A key that names no cut (no count of groups) is no run of this file's.
        if own and not rest and (count == "whole" or count.isdecimal()):
            cuts.setdefault(count, []).append(cut)
    return cuts


def before(ci, python: str, name: str, group: str, until: float) -> set[str] | str:
    """The cells main's own runs of file `name`'s prepare for `group`'s generation and process left unprepared
    (`identity`), whatever cut main's plan made of it: each of its rows' baselines main's commit keeps (this
    checkout's `mainline`, ci.py's `baseline_key`, listed under the file's place once; a green run's alone,
    never written over by a red one), whole or a group of any cut: one cut of main's whole, every group of
    it read, so a row this run cut otherwise is held to main's all the same. `BOOTSTRAP` where main has no
    run of it at all (no baseline, nor a run named) or none it kept whole (a cut part-written, a run named
    with no baseline); `UNREAD` where its baselines could not be read: no token or main, a Worker that failed
    or answered late, a baseline that is no list. Every request, each page of the listing among them, is
    bounded by `until` (a `time.monotonic` instant), and a read past it is none."""
    found = _before(ci, python, name, group, until)
    # Main keeping no baseline of it whole holds no run to one, a train's base (main's own commit, whose green
    # run writes it) and its heads alike: graded apart, the base green on BOOTSTRAP and every head stale on
    # the same main, the heads were returned for the base's grade (train 1291: main 2b1b88c39 kept 2 of its
    # 5 attention prepare groups, its other 3 stale in its CI, and three heads were returned on 9 groups). A
    # whole baseline holds both alike (train 1282), and one that could not be read (`UNREAD`) neither.
    return BOOTSTRAP if found == PARTIAL else found


def _before(ci, python: str, name: str, group: str, until: float) -> set[str] | str:
    """`before`'s read of main's baselines, whatever commit this checkout is: `PARTIAL` where they were read
    and are not whole (a run named with no baseline, or no cut every group of which is listed), `UNREAD`
    where they could not be read."""
    token = os.environ.get("DEW_XLA_CACHE_TOKEN") or os.environ.get(ci.READ_TOKEN)
    commit = ci.mainline()
    if not token or commit is None or "DEW_XLA_CACHE_URL" not in os.environ or time.monotonic() >= until:
        return UNREAD
    held, row_variant = ci.run_keys.device_class(), group.partition(ci.VARIED)[2]
    place = ci.row_place(ci.row(python, name, ""), held).rpartition("/")[0] + "/"
    store = ci.r2.Store(os.environ["DEW_XLA_CACHE_URL"], token, "dew-ci-test-cache", timeout=10.0,
                        send=ci.records.SEND)
    failures = (ci.r2.R2Error, OSError)
    try:
        cuts = listed_cuts(store, f"{ci.CACHE_PREFIX}-baseline/{commit}/{python}/{place}", row_variant, until)
        if not cuts:
            # A run main named whose baseline is not kept (its write lost; `named_at_commit` names a run only
            # once its baseline is written, but a run before baselines named its alone) is no first run.
            if listed_cuts(store, f"{ci.CACHE_PREFIX}-main/{commit}/{python}/{place}", row_variant, until):
                print(f"dry prepare: main named a run of {name} whose baseline is not kept; its row is not "
                      "graded", file=sys.stderr)
                return PARTIAL
            return BOOTSTRAP
        # One cut of main's whole, every group of it read with its list: a missing one would read its tests as
        # new to the row.
        whole = False
        for count, held_cuts in sorted(cuts.items(), key=lambda item: item[0] != "whole"):
            if count != "whole" and len(held_cuts) != int(count):
                continue
            whole = True
            found: set[str] = set()
            for cut in held_cuts:
                held_row = ci.row(python, name, ci.varied("" if cut == "whole" else cut.replace("-of-", "/"),
                                                          row_variant))
                kept = cells_of(bounded_get(store, ci.baseline_key(commit, python, held_row, held), until,
                                            failures))
                if kept is None:
                    break
                found |= kept
            else:
                return found
    except (Unread, *failures, ValueError) as failure:
        print(f"dry prepare: main's runs of {name} not read whole ({type(failure).__name__}: {failure}); its "
              "row is not graded", file=sys.stderr)
        return UNREAD
    # Every cut listed short of a group (`PARTIAL`), or one whole whose baseline was not read (`UNREAD`).
    return UNREAD if whole else PARTIAL


def run(ci, python: str, tests: list[str], split: str, out: Path, deadline: float, cache: str) -> int:
    """A task's dry prepares (`entries`): GPU tier files prepared on this CPU as their tier's part prepares
    them on its node's (`process`, the tier's own process of each: under the prepare's word,
    compiling every check-1 site's programs for the node's GPU, a user's process apart), for one
    compile-only GPU of the row's generation serialized as a node's topology (`TOPOLOGY_SCRIPT`), into a
    store of the task's own, before any GPU is drawn for them: a prepare that crashes is a red row here, not
    a part's run spent unprepared on its GPU (row 11834: an sm90 prepare lowered the VAE's Mosaic kernel for
    the CPU, and its cells ran unprepared, the prepare's exit 0). `tests` each begin `TOKEN`: a file, its
    row's group the task's `split`, or a kept row's token, read from its record where it holds green on these
    inputs (ci.py's `prepare_shared`). A row is red where a test fails and where it leaves a test unprepared
    that main's run of it did not (`before`, `graded`), a reused row's kept list read again against that
    baseline; the tests it leaves unprepared are kept in its record, so a train prepares only the rows its
    change reaches; a flow's cell deferred to its run is read as unprepared, as its run names it, and a test
    that declares it compiles at run is neither. A row left no time is said stale. The fixtures are the task's
    own fetch's (ci.py's `task`)."""
    began = time.monotonic()
    # The task's shared budget, PREPARE_WALL: its topology, baseline lookups and processes draw on it, so no
    # dry prepare task's pytest sets a train's wall (its records' publication and restores may run past it).
    deadline = min(deadline, PREPARE_WALL)
    names, kept = ci.kept_of([test.removeprefix(TOKEN) for test in tests])
    units = [(name, split) for name in names] + list(kept)
    wire, rows_ = ci.tier_wire(), []
    common = functools.cache(lambda: ci.prepare_shared(python))
    # What each lookup of main's baseline may take: a twentieth of the task's time, so its rows' prepares keep
    # the rest.
    looked = deadline / 20
    def stale(rest: list[tuple[str, str]], why: str) -> list[dict]:
        """Each of `rest` not prepared, said stale (`ci.stale_row`): the plan's next run of it its grade."""
        return [ci.stale_row(python, name, group, f"not prepared: {why}") for name, group in rest]

    # Each row's checkout writes (`ci.tested`'s), so the next row runs on the commit's checkout (row 13000).
    written: list[set[str]] = []
    with tempfile.TemporaryDirectory(prefix="dry-prepare-", dir=Path.cwd().parent) as scratch:
        for index, (name, group) in enumerate(units):
            left = deadline - (time.monotonic() - began)
            if (room := left - ci.session.GRACE - ci.SETTLE - 10.0) <= 0:
                rows_ += stale(units[index:], "its task had no time left for it")
                break
            row_variant = group.partition(ci.VARIED)[2]
            generation, users = parsed(row_variant)
            truths, topology = Path(scratch, f"truths-{index}"), Path(scratch, f"topology-{generation}")
            truths.mkdir()
            if not topology.exists():
                try:
                    subprocess.run([f".venv-{python}/bin/python", "-c", TOPOLOGY_SCRIPT, str(topology),
                                    generation], check=True, env={**os.environ, "JAX_PLATFORMS": "cpu"},
                                   timeout=room)
                except subprocess.TimeoutExpired:
                    rows_ += stale(units[index:], f"its {generation} topology was not written in time")
                    break
            before_ = before(ci, python, name, group, time.monotonic() + min(looked, room / 4))
            verdict = out.with_name(f"{out.name}.{index}")
            untouched = ci.untracked() if index < len(units) - 1 else set()
            process(
                ci, python, name, ci.cut_of(group), verdict, deadline - (time.monotonic() - began),
                unit=ci.row(python, name, group), threads=str(max(1, len(os.sched_getaffinity(0)))),
                users=users, variant=row_variant, cache=cache, common=common(), recorded=True, capped=True,
                # The node's one GPU, counted as a node's prepare counts its own (tests/lane_environment.py).
                environ={wire.TRUTHS: str(truths), wire.TOPOLOGY: str(topology), "CUDA_VISIBLE_DEVICES": "0"},
                kept={name: kept[(name, group)]} if (name, group) in kept else None, written=written,
                judged=lambda found, truths=truths, before_=before_, generation=generation: graded(
                    found, unprepared_in(truths, (wire.UNPREPARED, wire.DEFERRED), ci.grading.owning),
                    before_, generation, lambda named: ci.grading.owning(ci.parsed_row(named)[1]),
                    ci.grading.STALE, wire.CAPPED))
            rows_ += [nothing_timed(found) for found in json.loads(verdict.read_text())["rows"]]
            if index < len(units) - 1:
                ci.restored(written[-1] if written else set(), untouched, out)
    out.write_text(json.dumps({"rows": rows_}))
    # Its rows its verdict: a red one the suite's red exit, as every task's (.armada.json's task.redExit).
    return ci.told(out, 0)
