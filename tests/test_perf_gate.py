"""The performance gate's verdict (tools/perf_gate.py), on samples written here."""

import base64
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from tool_support import load

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gate():
    return load("perf_gate")


@pytest.mark.parametrize("head, code", [
    ({"gated": 10.0, "shown": 14.0}, 0),                   # a regression of an ungated row only shows
    ({"gated": "not run: it broke", "shown": 10.0}, 1),     # a broken row fails the gate
])
def test_the_report_fails_on_a_broken_row_and_not_on_an_ungated_one(gate, tmp_path, monkeypatch, head, code):
    rows = [gate.Row("gated", "ms", "").__dict__, gate.Row("shown", "ms", "", gated=False).__dict__]
    results = {"base": "main", "head": "head", "rows": rows,
               "samples": {"main": [{"gated": 10.0, "shown": 10.0}, {"gated": 10.1, "shown": 10.1}],
                           "head": [head, head]}}
    (tmp_path / "gate.json").write_text(json.dumps(results))
    monkeypatch.setattr(sys, "argv", ["perf_gate.py", "report", str(tmp_path / "gate.json")])
    assert gate.main() == code


def after(command, flag):
    return command[command.index(flag) + 1]


def standing_in(gate, monkeypatch, *, passed: bool, peaks: bool = True) -> list:
    """Stand every tool of the battery in with one writing its output format, a measured peak beside each
    time (`peak_device_bytes`; none where `peaks` is False, as a tool that predates it writes), and return the
    commands run. The reference either passes its claim or measures a head at 0.72 of it and exits 1."""
    peak = 3 * 2 ** 30 if peaks else None
    outputs = {
        "benchmark_step.py": lambda command: [{"ms_per_step": 5.0, "compile_seconds": 30.0,
                                               "peak_device_bytes": peak}],
        "benchmark_forward.py": lambda command: {after(command, "--cases"): {"ms": 5.0,
                                                                             "peak_device_bytes": peak}},
        "benchmark_serving_compile.py": lambda command: {
            "construction_seconds": 8.0, "warm_seconds": 40.0,
            "by_admission_width": {f"admission {rows}": 8.0 for rows in (1, 2, 4, 8)}},
        "benchmark_lm_serving.py": lambda command: {"sweep": [
            {"slots": int(slots), "repeats": [{"output_tokens_per_second": rate} for rate in (5.0, 6.0)],
             "peak_device_bytes": peak}
            for slots in after(command, "--slots").split(",")]},
        "benchmark_attention.py": lambda command: [
            {"implementation": implementation, "forward_backward_ms": 5.0, "peak_device_bytes": peak}
            for implementation in command[command.index("--implementations") + 1:
                                          command.index("--sequence-lengths")]],
        "benchmark_image_pipeline.py": lambda command: {
            "pipeline": {"device_crop_flip_jitter": {"images_per_second": 5.0, "peak_device_bytes": peak}}},
        "qwen35_pair.py": lambda command: {"comparison": {
            "dew_tokens_per_s": 720.0, "dew_over_torch": 1.0 if passed else 0.72, "tolerance": 0.01,
            "performance_claim_passed": passed}},
    }
    commands = []

    def child(command, **kwargs):
        commands.append(command)
        if command[1] == "-c":
            return subprocess.CompletedProcess(command, 0, "NVIDIA A100-SXM4-80GB\n", "")
        # The checkpoint's script, written beside its probe's output.
        if Path(command[1]).is_file() and Path(command[1]).read_text() == gate.CHECKPOINT:
            printed = json.dumps({"save_ms": 5.0, "restore_ms": 6.0, "peak_device_bytes": peak})
            return subprocess.CompletedProcess(command, 0, printed + "\n", "")
        tool = Path(command[1]).name
        flag = "--json-out" if "--json-out" in command else "--out"
        Path(after(command, flag)).write_text(json.dumps(outputs[tool](command)))
        failed = tool == "qwen35_pair.py" and not passed
        return subprocess.CompletedProcess(command, int(failed), "measured a slower head" * failed, "")

    monkeypatch.setattr(gate.subprocess, "run", child)
    return commands


def test_external_reference_subprocess_failure_keeps_its_measurement(gate, tmp_path, monkeypatch):
    """Exit 1 from an external reference still carries measured evidence through the battery, and a
    tree that measures no hardware row starts no process to read the device kind."""
    from types import SimpleNamespace

    commands = standing_in(gate, monkeypatch, passed=False)
    args = SimpleNamespace(only=["reference"], claim=True, model="unused", flowers="unused",
                           torch_python="torch", out=tmp_path / "gate.json")
    name = next(row.name for row in gate.BATTERY if row.head_only)
    sample = gate.measure(ROOT, Path(sys.executable), args, ROOT / "tools", is_head=True)
    assert sample[name] == {"exit_code": 1, "value": 720.0, "ratio": 0.72,
                            "tolerance": 0.01, "passed": False}
    assert Path(sample[f"evidence {name}"]).is_file()
    assert len(commands) == 2
    assert gate.measure(ROOT, Path(sys.executable), args, ROOT / "tools", is_head=False) == {}
    assert len(commands) == 2


def test_a_head_reads_every_row_of_the_battery_from_the_tool_it_names(gate, tmp_path, monkeypatch):
    """Each probe runs the tool its rows name and reads every one of its rows
    from that tool's output, so a head whose tools all ran has a measurement
    for every row of the battery."""
    from types import SimpleNamespace

    standing_in(gate, monkeypatch, passed=True)
    args = SimpleNamespace(only=[], claim=False, model="unused", flowers="unused", torch_python="torch",
                           out=tmp_path / "gate.json")
    sample = gate.measure(ROOT, Path(sys.executable), args, ROOT / "tools", is_head=True)
    for row in gate.BATTERY:
        value = sample[row.name]
        assert value["exit_code"] == 0 if row.head_only else gate.measured(value), (row.name, value)



def test_a_faster_head_whose_measured_peak_grows_regresses_and_an_unmeasured_peak_is_never_zero(
        gate, tmp_path, monkeypatch):
    """Two measures decide (testing.md): a head faster on a step's time whose measured peak grew past its band
    regresses on its peak row, whatever its time did; a base whose tool predates the peak (no
    peak_device_bytes) is not compared on it, never read as zero."""
    [row] = [row for row in gate.BATTERY if row.name == "peak step lm-dense 4x1024"]
    assert row.unit == "GiB" and row.gated
    assert gate.verdict(row, [3.0, 3.0, 3.0], [3.3, 3.3, 3.3])[0] == "regressed"
    [time] = [row for row in gate.BATTERY if row.name == "step lm-dense 4x1024"]
    assert gate.verdict(time, [5.0, 5.1, 5.0], [4.0, 4.1, 4.0])[0] == "faster"
    assert gate.peak({"ms_per_step": 5.0}) is None
    assert gate.verdict(row, [None, None, None], [3.0, 3.0, 3.0])[0] == "not compared"
    # A tree whose tools measure no peak: its peak rows unmeasured, every time row still measured.
    from types import SimpleNamespace

    standing_in(gate, monkeypatch, passed=True, peaks=False)
    args = SimpleNamespace(only=[], claim=False, model="unused", flowers="unused", torch_python="torch",
                           out=tmp_path / "gate.json")
    sample = gate.measure(ROOT, Path(sys.executable), args, ROOT / "tools", is_head=True)
    for battery_row in gate.BATTERY:
        if battery_row.unit == "GiB":
            assert sample[battery_row.name] is None, battery_row.name
        elif not battery_row.head_only:
            assert gate.measured(sample[battery_row.name]), (battery_row.name, sample[battery_row.name])


def standing_in_environments(gate, monkeypatch) -> tuple[list, list]:
    """Every tool of the battery stood in (`standing_in`), each environment's build too. Returns each
    environment the installer built, as (tree, its Python), and each row process, as (tree, command,
    environment), the compile cache it names made."""
    standing_in(gate, monkeypatch, passed=True)
    measured_row = gate.subprocess.run
    built, rows = [], []

    def child(command, **kwargs):
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, "580.82.07, 9.0\n", "")
        if command[0] == "uv":
            (Path(command[-1]) / "bin").mkdir(parents=True)
        elif command[0] == "sh":
            built.append((Path(kwargs["cwd"]), Path(command[command.index("--python") + 1])))
        else:
            env = kwargs.get("env", {})
            if "JAX_COMPILATION_CACHE_DIR" in env:
                Path(env["JAX_COMPILATION_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
            rows.append((Path(kwargs.get("cwd") or "."), command, env))
            return measured_row(command, **kwargs)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(gate.subprocess, "run", child)
    return built, rows


def test_each_tree_runs_in_the_environment_its_own_lock_installs(gate, tmp_path, monkeypatch):
    """A head that changes a locked wheel (a kernel's new version) is measured with it: each tree's
    rows run in the Python of the environment its own lock installs, by its own installer, and two
    trees whose locks agree share one environment, built once."""
    from types import SimpleNamespace

    built, rows = standing_in_environments(gate, monkeypatch)
    locks = [path.name for path in (ROOT / "locks").glob("*.txt")]
    trees = {name: tmp_path / name for name in ("main", "head")}
    for tree, wheel_version in zip(trees.values(), ("0.3.3", "0.3.4"), strict=True):
        (tree / "locks").mkdir(parents=True)
        for lock in locks:
            (tree / "locks" / lock).write_text(f"dew-flash-attn-cu13=={wheel_version}\n")
        (tree / "tools").mkdir()
        shutil.copy(ROOT / "tools" / "install_environment.sh", tree / "tools")
        (tree / ".python-version").write_text("3.12\n")

    def gated(environments):
        built.clear()
        rows.clear()
        args = SimpleNamespace(base=f"main={trees['main']}", head=f"head={trees['head']}", rounds=2,
                               only=["step lm-dense"], claim=False, model="unused", flowers="unused",
                               torch_python="", out=str(tmp_path / "gate.json"),
                               environments=str(environments))
        assert gate.run(args) == 0
        return dict(built)

    pythons = gated(tmp_path / "apart")
    assert pythons.keys() == set(trees.values()) and len(set(pythons.values())) == 2
    assert {tree for tree, _, _ in rows} == set(trees.values())
    assert all(Path(command[0]) == pythons[tree] for tree, command, _ in rows)

    for lock in locks:
        shutil.copy(trees["main"] / "locks" / lock, trees["head"] / "locks" / lock)
    (shared,) = gated(tmp_path / "shared").values()
    assert len(built) == 1 and {Path(command[0]) for _, command, _ in rows} == {shared}


PROJECT = """[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "probe-tree"
version = "0"
"""


def wheel(directory: Path, version: str) -> Path:
    """A wheel of the package `probe_wheel` at `version`: its module, metadata and record, zipped."""
    info = f"probe_wheel-{version}.dist-info"
    files = {"probe_wheel/__init__.py": f"__version__ = {version!r}\n",
             f"{info}/METADATA": f"Metadata-Version: 2.1\nName: probe-wheel\nVersion: {version}\n",
             f"{info}/WHEEL": ("Wheel-Version: 1.0\nGenerator: tests\nRoot-Is-Purelib: true\n"
                               "Tag: py3-none-any\n")}

    def digest(text: str) -> str:
        return base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=").decode()
    record = "".join(f"{name},sha256={digest(text)},{len(text.encode())}\n" for name, text in files.items())
    path = directory / f"probe_wheel-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in {**files, f"{info}/RECORD": f"{record}{info}/RECORD,,\n"}.items():
            archive.writestr(name, text)
    return path


def test_a_tree_imports_the_wheels_its_lock_names_and_dew_from_its_own_source(gate, tmp_path):
    """Built for real by each tree's own installer: a tree's process imports the wheel version its
    lock names and Dew from its own src, also where two trees whose locks agree share one
    environment, which is built once and then reused."""
    trees = {}
    for name, version in (("main", "1.0"), ("head", "2.0")):
        tree = trees[name] = tmp_path / name
        (tree / "locks").mkdir(parents=True)
        (tree / "locks" / "cpu.txt").write_text(f"probe-wheel @ {wheel(tmp_path, version).as_uri()}\n")
        (tree / "tools").mkdir()
        shutil.copy(ROOT / "tools" / "install_environment.sh", tree / "tools")
        (tree / ".python-version").write_text("3.12\n")
        (tree / "src" / "dew").mkdir(parents=True)
        (tree / "src" / "dew" / "__init__.py").write_text("")
        (tree / "pyproject.toml").write_text(PROJECT)

    def imported(tree: Path, python: Path) -> tuple[str, Path]:
        done = gate._run(tree, python, "", ["-c", "import dew, probe_wheel; print(probe_wheel.__version__, "
                                                  "dew.__file__)"], ROOT / "tools", 600)
        assert done.returncode == 0, done.stderr
        version, source = done.stdout.split()
        return version, Path(source)

    cache = tmp_path / "environments"
    pythons = {name: gate.environment(tree, "cpu", cache) for name, tree in trees.items()}
    assert pythons["main"] != pythons["head"]
    for (name, tree), version in zip(trees.items(), ("1.0", "2.0"), strict=True):
        assert imported(tree, pythons[name]) == (version, tree / "src" / "dew" / "__init__.py")

    shutil.copy(trees["main"] / "locks" / "cpu.txt", trees["head"] / "locks" / "cpu.txt")
    finished = (pythons["main"].parent.parent / "complete").stat().st_mtime_ns
    shared = gate.environment(trees["head"], "cpu", cache)
    assert shared == pythons["main"] and (shared.parent.parent / "complete").stat().st_mtime_ns == finished
    assert imported(trees["head"], shared) == ("1.0", trees["head"] / "src" / "dew" / "__init__.py")


def test_a_head_that_grows_its_compile_by_more_than_a_quarter_regresses(gate):
    """A step's compile row fails a head that grows it by more than the materiality `COMPILE_FLOOR` names:
    a fifth more over steady rounds is level, a third more regresses, a third less is faster; its step
    row keeps the 2% floor."""
    compile_row = next(row for row in gate.BATTERY if row.name == "compile lm-dense 4x1024")
    step_row = next(row for row in gate.BATTERY if row.name == "step lm-dense 4x1024")
    base = [60.0, 60.5, 61.0]
    assert gate.verdict(compile_row, base, [72.0, 72.5, 73.0])[0] == "level"
    assert gate.verdict(compile_row, base, [80.0, 80.5, 81.0])[0] == "regressed"
    assert gate.verdict(compile_row, base, [40.0, 40.5, 41.0])[0] == "faster"
    assert gate.verdict(step_row, [10.0, 10.0, 10.01], [10.5, 10.5, 10.51])[0] == "regressed"


CACHED = """
import jax, jax.numpy as jnp
counts = {"/jax/compilation_cache/cache_hits": 0, "/jax/compilation_cache/cache_misses": 0}
def counted(event, **_):
    if event in counts:
        counts[event] += 1
jax.monitoring.register_event_listener(counted)
jax.jit(lambda x: jnp.sin(x) * 2 + 1)(jnp.arange(8.0)).block_until_ready()
print(counts["/jax/compilation_cache/cache_hits"], counts["/jax/compilation_cache/cache_misses"])
"""
"""A child that compiles one small program and prints JAX's own count of cache hits and misses."""


def test_a_tree_compiles_once_into_its_own_cache_and_later_rounds_load_from_it(gate, tmp_path, monkeypatch):
    """Two real children of one tree with its cache: the first compiles and writes the program, though it
    compiles in well under JAX's default one-second threshold, the second loads it; a child of the other
    tree, with its own cache, compiles again; a child run outside a gate run keeps the caller's settings,
    here JAX's cache turned off."""
    for name in ("JAX_COMPILATION_CACHE_DIR", "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
                 "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JAX_ENABLE_COMPILATION_CACHE", "false")

    def child(cache):
        done = gate._run(ROOT, Path(sys.executable), "", ["-c", CACHED], ROOT / "tools", 600, cache)
        assert done.returncode == 0, done.stderr
        return tuple(int(word) for word in done.stdout.split()[-2:])

    caches = tmp_path / "caches"
    hits, compiled = child(caches / "base")
    assert hits == 0 and compiled > 0 and any((caches / "base").iterdir())
    assert child(caches / "base") == (compiled, 0)
    assert child(caches / "head") == (0, compiled)
    assert child(None) == (0, 0)


def locked_tree(tree: Path) -> Path:
    """A tree whose locks name numpy alone, which the install script builds an environment for."""
    (tree / "locks").mkdir(parents=True)
    for lock in (ROOT / "locks").glob("*.txt"):
        (tree / "locks" / lock.name).write_text("numpy\n")
    (tree / "tools").mkdir()
    shutil.copy(ROOT / "tools" / "install_environment.sh", tree / "tools")
    (tree / ".python-version").write_text("3.12\n")
    return tree


def test_one_tree_against_itself_under_a_setting_sets_it_in_the_heads_processes_alone(
        gate, tmp_path, monkeypatch):
    """`--head-env` with one tree as base and head: every process of the head holds the setting, its rows'
    and the device-kind probe of its hardware row alike, every process of the base keeps the caller's,
    and the results and the table name it."""
    from types import SimpleNamespace

    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_to=/caller")
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "caller"))
    _, rows = standing_in_environments(gate, monkeypatch)
    tree = locked_tree(tmp_path / "main")
    out = tmp_path / "gate.json"
    off = "--xla_gpu_enable_command_buffer="
    args = SimpleNamespace(base=f"main={tree}", head=f"buffers-off={tree}", rounds=2,
                           only=["serve", "lm-dense", "reference"], claim=False, model="unused",
                           flowers="unused", torch_python="", out=str(out),
                           environments=str(tmp_path / "environments"), head_env=[f"XLA_FLAGS={off}"])
    assert gate.run(args) == 0
    sides: dict[str, set] = {}
    for _, command, env in rows:
        # A row's process names its side's cache; the head's device-kind probe, for its hardware row, is
        # told by its command, as it keeps the caller's cache.
        probe = command[1] == "-c" and "device_kind" in command[2]
        side = "probe" if probe else Path(env["JAX_COMPILATION_CACHE_DIR"]).name
        sides.setdefault(side, set()).add(env["XLA_FLAGS"])
    assert sides == {"base": {"--xla_dump_to=/caller"}, "head": {off}, "probe": {off}}
    results = json.loads(out.read_text())
    assert results["head settings"] == {"XLA_FLAGS": off}
    table = tmp_path / "gate.md"
    assert gate.report(SimpleNamespace(results=out, table=str(table), claim=False)) == 0
    assert f"buffers-off ran with XLA_FLAGS='{off}'" in table.read_text()


def test_a_gate_keeps_each_trees_compiles_apart_and_judges_compile_rows_on_the_first_round(
        gate, tmp_path, monkeypatch):
    """A run gives each tree a cache of its own beside its results (made where absent), removes them at
    its end, records each tree's compile rows in the first round alone and the step rows in every round,
    and keeps the first round's seconds and the later rounds' apart."""
    from types import SimpleNamespace

    _, rows = standing_in_environments(gate, monkeypatch)
    trees = {name: locked_tree(tmp_path / name) for name in ("main", "head")}
    out = tmp_path / "not" / "yet" / "gate.json"
    args = SimpleNamespace(base=f"main={trees['main']}", head=f"head={trees['head']}", rounds=3,
                           only=["lm-dense 4x1024"], claim=False, model="unused", flowers="unused",
                           torch_python="", out=str(out), environments=str(tmp_path / "environments"))
    assert gate.run(args) == 0
    caches = [(tree.name, env["JAX_COMPILATION_CACHE_DIR"]) for tree, _, env in rows
              if "JAX_COMPILATION_CACHE_DIR" in env]
    by_tree = {tree: {cache for name, cache in caches if name == tree} for tree in trees}
    assert all(len(found) == 1 for found in by_tree.values()) and by_tree["main"] != by_tree["head"]
    assert {Path(cache).name for found in by_tree.values() for cache in found} == {"base", "head"}
    assert not any(Path(cache).exists() for found in by_tree.values() for cache in found)
    results = json.loads(out.read_text())
    for tree in trees:
        compiles = ["compile lm-dense 4x1024" in sample for sample in results["samples"][tree]]
        assert compiles == [True, False, False], (tree, compiles)
    assert set(results["seconds"]) == {"first round", "later rounds"}
    assert gate.report(SimpleNamespace(results=out, table=None, claim=False)) == 0


def test_a_probe_measuring_compiles_alone_runs_in_the_first_round_alone(gate, tmp_path, monkeypatch):
    """A probe whose every row is a first round's compile, the Server's program by program, runs once
    per tree: a later round would recompile it uncached and read nothing. Each of its programs is a
    row of its own, and the whole warm is shown beside them, ungated."""
    from types import SimpleNamespace

    _, rows = standing_in_environments(gate, monkeypatch)
    trees = {name: locked_tree(tmp_path / name) for name in ("main", "head")}
    out = tmp_path / "gate.json"
    args = SimpleNamespace(base=f"main={trees['main']}", head=f"head={trees['head']}", rounds=3,
                           only=["compile serve"], claim=False, model="unused", flowers="unused",
                           torch_python="", out=str(out), environments=str(tmp_path / "environments"))
    assert gate.run(args) == 0
    served = [command for _, command, _ in rows if Path(command[1]).name == "benchmark_serving_compile.py"]
    assert len(served) == len(trees)
    gated = {row.name: row for probe in gate.PROBES for row in probe.rows if "compile serve" in row.name}
    assert sorted(name for name, row in gated.items() if row.gated) == sorted(
        f"compile serve dory-1.0-4b {program}"
        for program in ("construction", "admission 1", "admission 2", "admission 4", "admission 8"))
    assert not gated["compile serve dory-1.0-4b warm 16 slots"].gated
