"""The inputs, keys and records of a test run (ci.py's): what a row's key is made of (`Reading`,
`cache_key`, `inputs`, the host's class), the record a green run writes and main's name for it (`record`,
`baselined`, `named_at_commit`), and a check-1 site's marks and artifacts (`sites`, `site_inputs`). Loaded
by ci.py by its path, as its neighbours are; it loads the neighbours these read (session.py, records.py and
the rest), which ci.py takes from it: one module of each serves both files, so a test that patches one
patches every caller's."""

import dataclasses
import functools
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import threading
from collections.abc import Collection, Container, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from costs import Mark

HERE = Path(__file__).resolve().parent
PARITY_STAGES = HERE.parents[1] / "src" / "dew" / "parity_stages.py"


@functools.cache
def loaded(path: Path):
    """The module at `path`, loaded once by its path: ci.py and this file run as a script or are loaded by the
    tests, and put nothing on sys.path. ci.py's one loader (`ci._loaded`), of its neighbours here and the
    gate's standard-library contract the plan shares (tests/check1_wire.py), as of this file's."""
    spec = importlib.util.spec_from_file_location(f"armada_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


session, records, followed, digests, grading, costs, cuts, rows = (
    loaded(HERE / f"{name}.py")
    for name in ("session", "records", "followed", "digests", "grading", "costs", "cuts", "rows"))
(store, record_key, green_key, main_key, baseline_key, Unread, UNREAD, epoch, begun, recorded, written) = (
    records.store, records.record_key, records.green_key, records.main_key, records.baseline_key,
    records.Unread, records.UNREAD, records.epoch, records.begun, records.recorded, records.written)
parsed_row, cut_of, units = rows.parsed_row, rows.cut_of, digests.units
prepares = loaded(HERE / "prepares.py")
VARIED = rows.VARIED



COLLECT = "collect"
"""The split of a task that imports test files without running them."""


@dataclasses.dataclass
class Reading:
    """One reading of the checkout's inputs (`cache_key`): each input's part by its path, and each module's
    units by its path, read once, by whichever of the threads looking rows up asks first. A reading lasts
    while the checkout stands still: a plan's lookups, gpu_tier's, a task's before its tests run (2026-10-31:
    train 368's plan read each of a row's ~3,000 inputs again for each of 1,682 rows, 9.5 minutes before its
    first task)."""

    parts: dict[str, bytes] = dataclasses.field(default_factory=dict)
    modules: dict[str, dict[str, bytes] | bytes] = dataclasses.field(default_factory=dict)
    durations: dict[str, dict[str, float]] | None = None
    stored: dict[str, list[str]] | None = None
    cuts: dict[tuple[str, int, bool], list[list[str]]] = dataclasses.field(default_factory=dict)
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    def cut(self, name: str, count: int, *, evenly: bool = False) -> list[list[str]]:
        """File `name`'s tests cut into `count` groups (`members`), a dry prepare's by their count where
        `evenly` (prepares.py's `prepared_cut`), the record parted by file once a reading and each cut once
        (train 662's plan: 3,620 group keys each parsed the record, a minute of two)."""
        # Under the lock: the plan's threads share a reading (train 679: a half-parted record cut empty).
        with self.lock:
            if self.durations is None:
                self.durations, self.stored = parted_record(session.cut_record()), session.stored_cuts()
            if (name, count, evenly) not in self.cuts:
                own = own_tests(name, self.durations.get(name, {}))
                self.cuts[(name, count, evenly)] = (
                    prepares.prepared_cut(list(own), count) if evenly
                    else session.planned(name, list(own), list(own.values()), count, self.stored))
            return self.cuts[(name, count, evenly)]

    def named(self, name: str) -> list[str]:
        """Every test of file `name` the cut's record names (`cut`), whatever group holds it."""
        return [node for group in self.cut(name, 1) for node in group]

    def part(self, path: str) -> bytes:
        """Input `path`'s part of a key: a unit's digest (`unit`), or its contents' (`digests.contents`)."""
        found = self.parts.get(path)
        if found is None:
            with self.lock:
                found = self.parts.get(path)
                if found is None:
                    found = self.parts[path] = self.read(path)
        return found

    def read(self, path: str) -> bytes:
        module, _, executed = path.partition(followed.CODE)
        if not executed or executed == followed.WHOLE or not Path(module).is_file():
            return digests.contents(path)
        if module not in self.modules:
            self.modules[module] = units(module)
        found = self.modules[module]
        return (found.get(executed) or digests.MISSING) if isinstance(found, dict) else found


APART = b"\0each whole file in a pytest of its own"
"""What a whole file's key holds beside its inputs: that its run was its process's alone (`cache_key`)."""


def cache_key(common: str, name: str, split: str, depends: list[str],
              reading: Reading | None = None) -> str:
    """The key of test file `name`'s row in `split`'s group whose run executed or read `depends`
    (executed.py): its `inputs`' contents, its directories' entries, `common`, and for a group of a split
    file, the group, the tests split.py cuts into it from the durations record (`members`) and the names of
    all it holds for the file (`Reading.named`), the record's only part a run's outcome depends on: not
    their seconds, which each train's fold of main's durations moves (the maintainer, 2026-10-28: every
    train re-keyed its split files and reran them cold, row 2000).
    A check-1 site's judge is keyed by its test, its inputs its map tasks' with its own (`site_inputs`): the
    site's key; a map task's, by its seeds besides (`sites`). The inputs as `reading` finds them: the keys of
    one reading read each input once, however many rows name it."""
    digest = hashlib.sha256(common.encode())
    reading = Reading() if reading is None else reading
    # A test node's (`file::test`) inputs are its file's, the node named beside them.
    for path in inputs(grading.owning(name), depends):
        digest.update(path.encode() + reading.part(path))
    if "::" in name:
        digest.update(name.encode())
    if not split:
        # A whole file's run is its own process's (ci.py's `task`): a green run beside others, which may
        # have passed on what they left in it, is none of its (row 3924's review).
        digest.update(APART)
    if session.SITE.fullmatch(split):
        digest.update(split.encode())
    elif split and split != COLLECT:
        # A process variant's (`varied`) by its name beside its cut's tests; a whole file's variant, its name.
        # And every test its record names for the file: one it drops joins a group by the collection alone
        # (`session.kept`), a group whose own tests, and key, stood (review: its row reused, the test unrun).
        cut = cut_of(split)
        evenly = prepares.prepared_group(split, VARIED)
        held = [members(name, cut, reading, evenly=evenly), reading.named(name)] if cut else []
        digest.update(split.encode() + (json.dumps(held).encode() if cut else b""))
    return digest.hexdigest()


def inputs(name: str, depends: list[str]) -> list[str]:
    """What test file `name`'s key covers, given that its run executed or read `depends`: those, its own code
    whole (followed.py's `WHOLE`), and the import lookups that found each module it executed under src/ or
    tools/ (`digests.lookups`): a module or package of its name added beside one it imported (which
    shadows it), or an __init__.py added to a namespace package on the way to one, which executes as the
    import passes, changes its key; a module of another name added beside it does not, unless an import its
    run tried and failed looks for that name (followed.py's `missed`). tools/ is searched so, as a test puts
    it, or a directory under it, on sys.path and imports a tool by its name; src/'s own entries every key
    covers (`shared`). The
    harness's own modules (tools/armada/, which runs every test process and records it, `SHARED`) are no
    import of a test's, so their lookups are none of its inputs (the maintainer, 2026-10-29: train 296's one
    added tool reran 1,794 of 1,795; train 668: one module added to nn/backbones re-keyed 589 rows)."""
    own = f"{name}{followed.CODE}{followed.WHOLE}"
    return sorted({own, *(path for path in depends if not digests.built_input(path)),
                   *(looked for path in depends
                     for looked in digests.lookups(path.partition(followed.CODE)[0]))})


def cpu_class() -> str:
    """This host's CPU microarchitecture, the one class the records, the tests' compilation cache and
    the shared XLA bundles key it by (`digests.host`)."""
    return digests.host().cpu_class()


def device_class(network: bool = False) -> str:
    """The class a task's results belong to: on a CUDA lane (`JAX_PLATFORMS`, as tests/lane_environment.py
    reads it) the models of the GPUs it sees as the driver names them, before the CPU's class, whose float32
    the lane's CPU devices compute (`cpu_class`); else the CPU's class alone. A network lane's (`gpu_lane`) is
    a class of its own, `network` after it: its records hold the network-marked tests alone, run with the Hub
    open, apart from the same file's other tests and their peaks."""
    if "cuda" not in os.environ.get("JAX_PLATFORMS", "").split(","):
        held = cpu_class()
    else:
        held = "-".join([*(re.sub(r"[^A-Za-z0-9.]+", "-", model) for model in sorted(set(gpu_query("name")))),
                         cpu_class()])
    return f"{held}-network" if network else held


def gpu_query(field: str) -> list[str]:
    """`nvidia-smi`'s `field` of each GPU this process sees (`CUDA_VISIBLE_DEVICES`, when set)."""
    given = os.environ.get("CUDA_VISIBLE_DEVICES")
    return subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits",
                           *(["-i", given] if given else [])], capture_output=True, text=True,
                          check=True).stdout.split("\n")[:-1]


@functools.cache
def head_commit() -> str | None:
    """The commit this checkout is at, which a task's records of it name (`main_key`), or None outside git."""
    found = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return (found.stdout.strip() or None) if found.returncode == 0 else None


def baselined(python: str, place: str, held: str, token: str, ran: Mapping,
              until: float | None = None) -> bool:
    """Whether a dry prepare's green row (`ran`) on class `held` has this commit's baseline (`baseline_key`)
    written now, tried once within `until` where one is given, or none is due; a green row with none stands
    stale (`unbaselined`)."""
    if (commit := head_commit()) is None or ran.get("exitCode") != 0 or "unprepared" not in ran:
        return True
    return written(baseline_key(commit, python, place, held), token,
                   json.dumps({"unprepared": ran["unprepared"]}).encode(), tries=1, until=until)


def named_at_commit(python: str, place: str, held: str, key: str, token: str,
                    until: float | None = None) -> None:
    """Row `place`'s green run of `key` on class `held` named this commit's own (`main_key`), tried once; a
    dry prepare's only once its baseline is written (`baselined`), so a name alone is a baseline lost."""
    if (commit := head_commit()) is not None:
        written(main_key(commit, python, place, held), token, json.dumps({"key": key}).encode(), tries=1,
                until=until)


def parted_record(path: Path = session.DURATIONS) -> dict[str, dict[str, float]]:
    """The durations record by test file, each file's tests in node-id order; `path`, the record a task's
    groups are cut by where it is another's (`session.cut_record`: a GPU tier's own seconds)."""
    parted: dict[str, dict[str, float]] = {}
    record = durations_record(path)
    for node in sorted(record):
        parted.setdefault(grading.owning(node), {})[node] = record[node]
    return parted


def own_tests(name: str, tests: dict[str, float]) -> dict[str, float]:
    """Of file `name`'s recorded `tests`, those the file still defines (`session.standing`) but its check-1
    sites', which run apart (`sites`)."""
    return session.own(session.standing(name, tests), sites(name))


def durations_record(path: Path = session.DURATIONS) -> dict[str, float]:
    """The durations record (`session.DURATIONS`, or the one at `path`): each test's node id and its cost
    (`session.cost`)."""
    return session.costs(path)


def members(name: str, split: str, reading: Reading | None = None, *, evenly: bool = False) -> list[str]:
    """The node ids split.py cuts into group `split` ("G/N") of test file `name`'s recorded tests but its
    sites' (`own_tests`), once a reading (`Reading.cut`): as the task's groups are cut, by their record's
    seconds (`session.cut_record`: a GPU tier's, its GPUs'), a dry prepare's by their count where `evenly`
    (`session.SPLIT_EVENLY`), so a row stands for no other cut's group."""
    index, count = (int(part) for part in split.split("/"))
    return (Reading() if reading is None else reading).cut(name, count, evenly=evenly)[index - 1]


SITE_MARK = re.compile(r"\bmark\.(?:" + "|".join(session.SHARDS) + r")\b")
"""A site's mark as a test file's source spells it (`sites`), which a file that holds none never parses."""


INPUTS = "/armada/inputs"
"""Where a judge's task finds its map tasks' artifacts (armada's `{inputs}`), each task's under
`<job>/<index>`, where armada names none (`armada_dir`)."""


def armada_dir(variable: str, default: str) -> Path:
    """The directory armada names to a task by `variable` (ARMADA_ARTIFACTS, ARMADA_INPUTS), which every task
    it starts holds, else where its container keeps it (`default`)."""
    return Path(os.environ.get(variable) or default)


def sites(name: str) -> "dict[str, Mark]":
    """Test file `name`'s tests marked one of `session.SHARDS`, each by its name in the file (a class's
    method's `Class.test_name`) with its mark (`Mark`). Read from the source, as `gangs` reads its models: the
    plan runs under a Python with no pytest (`pinned`) and imports no test, so it reads the mark on the test's
    own decorator, `@pytest.mark.check1_shards(seeds=S, per_task=P)` or `@pytest.mark.shards(units=N,
    per_task=P)`, of literal counts above zero, one a test, or a parameter's of the test's one literal
    parametrize list, `pytest.param(..., id="<id>", marks=pytest.mark.check1_shards(...))`, its site
    `test[id]`. A mark written otherwise (on a class or a module, on a parameter with no literal id or of a
    test parametrized twice, of counts computed, two on one test) would set its site aside from its file's
    tasks (`selected`) and plan none of its own, so nothing would run it: its file is refused, by the lines
    that hold it."""
    path = Path(name)
    text = path.read_text() if path.is_file() else ""
    return marked_sites(name, text) if SITE_MARK.search(text) else {}


@functools.cache
def marked_sites(name: str, text: str) -> "dict[str, Mark]":
    """`sites` of test file `name`, whose source is `text`, read once a text: each site's mark as its source
    declares it (`session.declared_marks`), its units staged as check 1 draws them (`PARITY_STAGES`)."""
    staged = loaded(PARITY_STAGES).staged_units
    return {site: costs.Mark(staged(declared) if kind == "check1_shards" else declared, counts["per_task"],
                       line, declared, staged(counts.get("conditioning", 0)),
                       counts.get("conditioning_per_task", 0))
            for site, (kind, counts, line) in session.declared_marks(name, text).items()
            for declared in [counts[session.SHARDS[kind]]]}


def site_task(python: str, name: str, test: str, span: tuple[int, int] | None = None) -> str:
    """The task name of test file `name`'s site `test` under `python`: its judge's, or with `span` its map
    task's that draws those seeds. armada keeps a task's name where it is at most 80 of [A-Za-z0-9._-],
    which a judge waits on its map tasks by (`after`) and armada keeps a map task's seconds under, so it stays
    one from plan to plan: its file's stem and a digest of the file and test."""
    digest = hashlib.sha256(f"{name}::{test}".encode()).hexdigest()[:12]
    drawn = "" if span is None else f"-draws-{span[0]}-{span[1]}"
    return f"{python}-{Path(name).stem[:40]}-{digest}{drawn}"


def native_inputs(opened: set[str], recorded: set[str]) -> set[str]:
    """The fixture directories, each up to tests/fixtures, of what a task's processes opened natively
    (`watched`: an open no Python process recorded, as a tokenizer's Rust code makes): a native reader's
    probe beside what it opened is no record's, as followed.py's `PROBES` are Python's. The fetched store
    is none: its payloads are content-addressed, and a planner, which never downloads them, keys them as a
    task does (`digests.contents`)."""
    store = digests.FIXTURES / "fetched"
    return {str(parent) for name in opened - recorded if digests.FIXTURES in (path := Path(name)).parents
            and store != path and store not in path.parents
            for parent in path.parents if parent == digests.FIXTURES or digests.FIXTURES in parent.parents}


def wrote_inputs(depends: list[str], changed: Collection[str]) -> list[str]:
    """The inputs among a run's `depends` (followed.py's records, as `inputs` expands them for its key) that a
    file `changed` (`watched.changed`: written, removed or made) names: a file read, probed or run as a
    module, or the entries an import's lookup finds of its name (`followed.LOOKUP`, `digests.looked_up`). A
    directory's listing is weighed by none: a file made in it is new to every checkout a plan reads, so no
    plan finds the key it leaves."""
    written = []
    for item in depends:
        if item.startswith(followed.LOOKUP):
            looked = item[len(followed.LOOKUP):]
            if any(looked_entry(path, looked) for path in changed):
                written.append(looked)
        elif (path := item.removeprefix(followed.PROBED).partition(followed.CODE)[0]) in changed:
            written.append(path)
    return written


def looked_entry(path: str, looked: str) -> bool:
    """Whether `path` is an entry the import lookup of `looked` lists (`digests.looked_up`): its name, a
    module or package of any suffix beside it, or a package's own __init__ of any suffix; no file of another
    name (review: `held/__init__report.json` is none of `held`'s)."""
    parent, _, entry = path.rpartition("/")
    directory, _, name = looked.rpartition("/")
    return ((parent == directory and (entry == name or entry.startswith(f"{name}.")))
            or (parent == looked and entry.partition(".")[0] == "__init__"))


def site_inputs(python: str, row_name: str, depends: list[str] | None) -> list[str] | None:
    """What row `row_name` of a check-1 site's task ran on (`session.SITE`), given what its own run executed
    and read (`depends`, None where that is not known): a map task's own, kept among its artifacts for its
    judge, which reads every one of its map tasks' among its inputs and keys on them with its own
    (`cache_key`), so the site's key covers what its draws ran, which the judge runs none of (an arm run as
    given, a branch of a later seed): its first stage's map tasks', and its later ones' where they ran; None
    where one of those is not known or not found. Any other row's, as it is."""
    _, name, group = parsed_row(row_name)
    site = session.SITE.fullmatch(group)
    if site is None:
        return depends
    test = site["test"]
    if site["start"] is not None:
        span = (int(site["start"]), int(site["stop"]))
        named = f"{site_task(python, name, test, span)}.inputs.json"
        kept = armada_dir("ARMADA_ARTIFACTS", ARTIFACTS) / named
        kept.parent.mkdir(parents=True, exist_ok=True)
        kept.write_text(json.dumps(depends))
        return depends
    # Its map tasks' inputs by the spans they drew, whatever cut the plan made of its units (`cut_draws`).
    prefix, suffix = f"{site_task(python, name, test)}-draws-", ".inputs.json"
    found = {tuple(int(end) for end in path.name[len(prefix):-len(suffix)].split("-")):
             json.loads(path.read_text())
             for path in armada_dir("ARMADA_INPUTS", INPUTS).rglob(f"{prefix}*{suffix}")}
    mark = sites(name).get(test)
    # A judge's later map tasks ran only where its first stage asked for them (`site_entries`), all of them:
    # one that judged without them judged over its first stage's alone. Each stage it judged over is its
    # units whole, so a map task missing from it is not known.
    first = sorted(span for span in found if mark and span[0] < mark.timed)
    later = sorted(span for span in found if mark and span[0] >= mark.timed)
    if (depends is None or mark is None or not drawn_whole(first, 0, mark.timed)
            or (later and not drawn_whole(later, mark.timed, mark.units + mark.weighed))
            or any(found[span] is None for span in (*first, *later))):
        return None
    return sorted({*depends, *(path for span in (*first, *later) for path in found[span])})


def drawn_whole(spans: Sequence[tuple[int, ...]], start: int, stop: int) -> bool:
    """Whether sorted `spans` draw units `start` to `stop` each once, end to end."""
    at = start
    for begin, end in spans:
        if begin != at:
            return False
        at = end
    return at == stop and bool(spans)


def record(python: str, rows: list[dict], log: Path, common: str, network: bool = False,
           opened: set[str] | None = None, unkeyed: Container[str] = (),
           changed: Collection[str] = (), began_in: Mapping[str, str | None] | None = None,
           until: float | None = None) -> set[str]:
    """Keep each row a task ran with the files its tests executed and read, every data file the task's
    processes opened (`opened`, the task's watch: None where it was blind, and no row keeps a key), where
    they read the backend and its peak memory on a GPU and its children's pools: `gpu_tier` reads the
    reads, `gpu_needs` the memory. A green row keeps its key too, which `reused` reads it back by; a red
    row has none, so a later run never reuses an earlier green row of its file; nor has one that read a file
    the task's processes wrote (`changed`, `watched.changed`), whose key would hash the bytes they left, not
    those it read (row 2559). A check-1 site's judge keeps its map tasks' inputs with its own
    (`site_inputs`). A row whose key keeps no inputs keeps what its run read (`read`, reach.py's). A green
    row is written only within the epoch it began in (`began_in`, `epochs_of`): one a red run rotated while
    it ran is an older verdict, and is kept nowhere (row 12180). Every read and write within `until` where
    one is given (a dry prepare's task's shared budget, row 13014), a write past it not made. Returns the
    green rows whose baseline the store did not take (`baselined`), which stand stale (`unbaselined`)."""
    token = os.environ.get("DEW_XLA_CACHE_TOKEN")
    if not token or "DEW_XLA_CACHE_URL" not in os.environ:
        return set()
    unnamed: set[str] = set()
    # Each file's record in the process's log (followed.py's `recorded`): none for a file it never reached.
    executed = followed.recorded(log)
    held = device_class(network)
    if opened is not None:
        recorded = {read for ran in executed.values() for read in ran["depends"] or ()}
        opened = opened | native_inputs(opened, recorded)
    for row_ in rows:
        # A green dry prepare's baseline, whatever of its admission to reuse below (record, key, epoch).
        if not baselined(python, row_["name"], held, token, row_, until):
            unnamed.add(row_["name"])
        name = parsed_row(row_["name"])[1]
        # A test node's row (`file::test`) keeps its file's record, the one its process's log holds.
        file = grading.owning(name)
        ran = executed.get(file, {"depends": None, "reads": [], "peak": None, "shares": 0, "pools": 0})
        why: list[str] = [*(ran.get("why") or ())]  # Each reason the row keeps no key, as its record says.
        site = session.SITE.fullmatch(parsed_row(row_["name"])[2])  # a check-1 judge reads not as its site
        read = (None if ran["depends"] is None or (site is not None and site["start"] is None) else
                sorted({*ran["depends"], *(opened or ())}))
        if ran["depends"] is not None:
            inputs_ = None if opened is None else sorted({*ran["depends"], *opened})
            # Its writes weighed against what it read and ran, then against the import lookups its key covers
            # besides (`inputs`), each as the key reads it (a lookup by its entries' names; row 10800).
            wrote = [] if inputs_ is None else sorted({*wrote_inputs(inputs_, changed), *wrote_inputs(
                [item for item in inputs(file, inputs_) if item.startswith(followed.LOOKUP)], changed)})
            why += (["its task's watch of the checkout was blind"] if inputs_ is None
                    else [f"its task wrote its input {path}" for path in wrote[:6]])
            ran = {**ran, "depends": None if inputs_ is None or wrote else inputs_}
        judged = site_inputs(python, row_["name"], ran["depends"])
        # A check-1 site's map task is never reused (ci.py's `unrun`): its inputs are its judge's, kept among
        # its artifacts (`site_inputs`), and a record of it is read by none (2026-11-25: ~196,000 a day).
        if site is not None and site["start"] is not None:
            continue
        if ran["depends"] is not None and judged is None:
            why.append("its check-1 site's judge ran over map tasks not drawn whole")
        ran = {**ran, "depends": judged}
        if ran["depends"] is not None and file in unkeyed:
            why.append("the durations record names its file's groups not wholly (unkeyed)")
        if row_["exitCode"] == 0 and file not in executed:
            continue
        green = row_["exitCode"] == 0 and ran["depends"] is not None
        # A group of a file the durations record does not wholly name (`unkeyed`, split.py) keeps no key: the
        # plan, which cuts by the record alone, may cut the file otherwise (`session.cut`).
        key = (cache_key(common, name, parsed_row(row_["name"])[2], ran["depends"])
               if green and file not in unkeyed else None)
        # A red run begins the row's next epoch first, so every green run recorded before it stands no
        # more (`epoch`); a green one keeps the epoch it ran in, begun where the row has none, or none where
        # neither was read or written.
        try:
            current, tag = epoch(python, row_["name"], held, common, token, until)
        except Unread:
            current, tag = UNREAD, None
        if row_["exitCode"] != 0:
            if current == UNREAD or begun(python, row_["name"], held, common, token, after=tag,
                                          until=until) == UNREAD:
                continue
            current = None
        elif began_in is not None and (began_in.get(row_["name"], UNREAD) == UNREAD
                                       or began_in[row_["name"]] != current):
            # Not the epoch it began in (a red's rotation while it ran, or one unread): no green of it stands.
            continue
        elif current is None:
            current = begun(python, row_["name"], held, common, token, fresh_only=True, until=until)
        stated = (why or None) if key is None and row_["exitCode"] == 0 else None
        body = json.dumps({"key": key, "why": stated,
                           "depends": ran["depends"], "row": row_, "reads": ran["reads"],
                           **({"read": read} if ran["depends"] is None and read is not None else {}),
                           "peak": ran["peak"], "shares": ran["shares"], "pools": ran["pools"],
                           "epoch": current}).encode()
        # The record under these inputs, which `reused` reads, and the last run's, which the GPU tier and
        # lane read, only once the first is written, so it is never newer.
        if written(record_key(python, row_["name"], held, common), token, body, until=until):
            written(record_key(python, row_["name"], held), token, body, until=until)
            # The run of exactly these inputs, kept by their key (`green_key`): a green one, or a red one,
            # which replaces a green row of the same inputs so no later run reuses it (a flaky red).
            ran_on = key if key is not None else None if ran["depends"] is None or green else cache_key(
                common, name, parsed_row(row_["name"])[2], ran["depends"])
            if ran_on is not None:
                written(green_key(python, row_["name"], held, ran_on), token, body, until=until)
            if key is not None and row_["name"] not in unnamed:
                named_at_commit(python, row_["name"], held, key, token, until)
    return unnamed


ARTIFACTS = "/armada/task/artifacts"
"""Where an armada task's files are kept for `armada map --artifacts`."""
