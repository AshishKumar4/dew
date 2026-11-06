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
STANDING_IN = "NVIDIA A100-SXM4-80GB"
"""The device the stood-in tools report: a row limited to another (an H100's fine-tune) is not measured
on it."""


@pytest.fixture(scope="module")
def gate():
    return load("perf_gate")


@pytest.mark.parametrize("head, code", [
    ({"gated": 10.0, "shown": 14.0}, 0),                   # a regression of an ungated row only shows
    ({"gated": "not run: it broke", "shown": 10.0}, 1),     # a broken row fails the gate
])
def test_the_report_fails_on_a_broken_row_and_not_on_an_ungated_one(gate, tmp_path, monkeypatch, head, code):
    rows = [gate.Row("gated", "ms", "").__dict__, gate.Row("shown", "ms", "", gated=False).__dict__]
    steady = [{"gated": 10.0, "shown": 10.0}, {"gated": 10.0625, "shown": 10.0625}] * 3
    results = {"base": "main", "head": "head", "alone": "main~aa", "rows": rows, "rounds": 6,
               "samples": {"main": steady, "main~aa": steady[::-1], "head": [head] * 6}}
    (tmp_path / "gate.json").write_text(json.dumps(results))
    monkeypatch.setattr(sys, "argv", ["perf_gate.py", "report", str(tmp_path / "gate.json")])
    assert gate.main() == code


def samples(*values) -> list[float]:
    """Six rounds' samples, `values` cycled: literals a binary float holds exactly."""
    return [float(values[index % len(values)]) for index in range(6)]


def test_a_row_is_judged_by_its_rounds_ratios_against_its_own_a_a_and_acted_on_past_its_floor(gate):
    """Each round's head over its base, against each round's A/A over its base, judged at LEVEL
    (`benchmark.judged`): a head whose six ratios all pass 1 but lie within the A/A's interval is level,
    where the spread band a run judged by before called its separated samples regressed; past the A/A and
    past the 2% floor it regresses, or is faster; past the A/A within the floor it differs below its floor,
    and passes; an A/A wider than the floor makes the row insensitive, said beside its state."""
    row = gate.Row("step", "ms", "")
    base = samples(8.0)
    # Ratios 1.03125 and 1.0625 against an A/A whose ratios span 0.9375 to 1.0625.
    state, numbers = gate.verdict(row, base, samples(8.25, 8.5), samples(7.5, 8.5))
    assert (state, numbers["differs"], numbers["insensitive"]) == ("level", False, True)
    assert numbers["interval"] == [1.03125, 1.0625] and numbers["alone"] == [0.9375, 1.0625]
    tight = samples(8.0, 8.0625)
    state, numbers = gate.verdict(row, base, samples(8.5, 8.75), tight)
    assert (state, numbers["ratio"], numbers["interval"], numbers["alone"]) == (
        "regressed", 1.078125, [1.0625, 1.09375], [1.0, 1.0078125])
    assert gate.verdict(row, base, samples(7.5, 7.25), tight)[0] == "faster"
    state, numbers = gate.verdict(row, base, samples(8.125), tight)
    assert (state, numbers["differs"], numbers["insensitive"]) == ("different, below floor", True, False)
    # Higher is better: throughput that falls past the floor regresses.
    assert gate.verdict(gate.Row("rate", "items/s", ""), base, samples(7.5, 7.25), tight)[0] == "regressed"


def test_a_row_any_arm_left_unmeasured_in_a_round_is_partial_and_too_few_rounds_are_unjudged(gate):
    """Rounds every arm measured are never picked out of a run some arm missed: one missing A/A round makes
    the row partial; five complete rounds reach no interval at 0.95 and are unjudged, which fails the
    gate as the gate's own."""
    row = gate.Row("step", "ms", "")
    base, head, alone = samples(8.0), samples(8.5), samples(8.0, 8.0625)
    state, numbers = gate.verdict(row, base, head, [*alone[:5], None])
    assert state == "partial" and numbers["unmeasured rounds"] == {"A/A": 1}
    state, numbers = gate.verdict(row, base[:5], head[:5], alone[:5])
    assert state == "unjudged" and "short of 0.95" in numbers["why"]
    assert "unjudged" in gate.UNQUALIFIED and "partial" in gate.UNQUALIFIED





def after(command, flag):
    return command[command.index(flag) + 1]


def standing_in(gate, monkeypatch, *, passed: bool, peaks: bool = True) -> list:
    """Stand every tool of the battery in with one writing its output format, a measured peak beside each
    time (`peak_device_bytes`; none where `peaks` is False, as a tool that predates it writes), and return the
    commands run. The reference either passes its claim or measures a head at 0.72 of it and exits 1."""
    peak = 3 * 2 ** 30 if peaks else None
    outputs = {
        "benchmark_step.py": lambda command: [{"ms_per_step": 5.0, "compile_seconds": 30.0,
                                               "peak_device_bytes": peak, "mfu": 0.4}],
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
            "dew_tokens_per_s": 720.0, "dew_over_torch": 1.0 if passed else 0.75, "interval": [0.75, 0.75],
            "alone": [0.96875, 1.03125], "differs": not passed, "rounds": 6,
            "performance_claim_passed": passed}},
    }
    commands = []

    def child(command, **kwargs):
        commands.append(command)
        if command[1] == "-c":
            return subprocess.CompletedProcess(command, 0, f"{STANDING_IN}\n", "")
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
    assert sample[name] == {"exit_code": 1, "value": 720.0, "ratio": 0.75, "interval": [0.75, 0.75],
                            "alone": [0.96875, 1.03125], "differs": True, "rounds": 6, "passed": False}
    assert gate.reference_verdict([sample[name]])[0] == "failed"
    assert Path(sample[f"evidence {name}"]).is_file()
    assert len(commands) == 2
    assert gate.measure(ROOT, Path(sys.executable), args, ROOT / "tools", is_head=False) == {}
    assert len(commands) == 2


def test_a_head_reads_every_row_of_the_battery_from_the_tool_it_names(gate, tmp_path, monkeypatch):
    """Each probe runs the tool its rows name and reads every one of its rows
    from that tool's output, so a head whose tools all ran has a measurement
    for every row of the battery its device can take, and says so of the rest."""
    from types import SimpleNamespace

    standing_in(gate, monkeypatch, passed=True)
    args = SimpleNamespace(only=[], claim=False, model="unused", flowers="unused", torch_python="torch",
                           out=tmp_path / "gate.json")
    sample = gate.measure(ROOT, Path(sys.executable), args, ROOT / "tools", is_head=True)
    for row in gate.BATTERY:
        value = sample[row.name]
        if gate.hardware_status(row, STANDING_IN) is not None and not row.head_only:
            assert value == gate.hardware_status(row, STANDING_IN), (row.name, value)
        else:
            assert value["exit_code"] == 0 if row.head_only else gate.measured(value), (row.name, value)



def test_a_faster_head_whose_measured_peak_grows_regresses_and_an_unmeasured_peak_is_never_zero(
        gate, tmp_path, monkeypatch):
    """Two measures decide (testing.md): a head faster on a step's time whose measured peak grew past its
    floor regresses on its peak row, whatever its time did; a base whose tool predates the peak (no
    peak_device_bytes) is not compared on it, never read as zero."""
    [row] = [row for row in gate.BATTERY if row.name == "peak step lm-dense 4x1024"]
    assert row.unit == "GiB" and row.gated
    tight = samples(3.0, 3.0078125)
    assert gate.verdict(row, samples(3.0), samples(3.25, 3.3125), tight)[0] == "regressed"
    [time] = [row for row in gate.BATTERY if row.name == "step lm-dense 4x1024"]
    assert gate.verdict(time, samples(5.0), samples(4.0, 4.125), samples(5.0, 5.015625))[0] == "faster"
    assert gate.peak({"ms_per_step": 5.0}) is None
    assert gate.verdict(row, [None] * 6, samples(3.0), [None] * 6)[0] == "not compared"
    # A tree whose tools measure no peak: its peak rows unmeasured, every time row still measured.
    from types import SimpleNamespace

    standing_in(gate, monkeypatch, passed=True, peaks=False)
    args = SimpleNamespace(only=[], claim=False, model="unused", flowers="unused", torch_python="torch",
                           out=tmp_path / "gate.json")
    sample = gate.measure(ROOT, Path(sys.executable), args, ROOT / "tools", is_head=True)
    for battery_row in gate.BATTERY:
        if gate.hardware_status(battery_row, STANDING_IN) is not None and not battery_row.head_only:
            continue
        if battery_row.unit == "GiB":
            assert sample[battery_row.name] is None, battery_row.name
        elif not battery_row.head_only:
            assert gate.measured(sample[battery_row.name]), (battery_row.name, sample[battery_row.name])


def test_a_step_reports_its_mfu_from_the_same_invocation_and_a_lower_one_regresses(gate):
    """Each step probe's MFU row reads benchmark_step's train/mfu from the invocation that times it, in
    percent, higher the better: a head whose MFU falls past its floor regresses, one whose tool predates the
    MFU is not compared on it, never read as zero."""
    [row] = [row for row in gate.BATTERY if row.name == "mfu step lm-dense 4x1024"]
    assert row.unit == "% mfu" and row.gated
    base, tight = samples(40.0), samples(40.0, 40.125)
    assert gate.verdict(row, base, samples(36.0, 36.25), tight)[0] == "regressed"
    assert gate.verdict(row, base, samples(44.0, 44.25), tight)[0] == "faster"
    assert gate.verdict(row, [None] * 6, samples(40.0), [None] * 6)[0] == "not compared"


def test_a_full_fine_tune_steps_adamw_on_the_device_it_is_sized_for_and_is_unmeasured_off_it(gate):
    """The fine-tune rows time a 4B decoder's full AdamW step on an H100 and a 1.7B one's on an A100:
    each case is one benchmark_step reads, on a declared field of the decoder, stepping the optimizer
    it names, and off its device its four rows are not measured, never compared against another's, nor
    failing a claim, which requires the references alone."""
    import dataclasses

    from dew.nn.backbones.causal_transformer import CausalTransformer

    step = load("benchmark_step")
    declared = {field.name for field in dataclasses.fields(CausalTransformer)}
    h100, a100 = "NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-40GB"
    for name, device, other in (("step finetune-4b 1x2048", h100, a100),
                                ("step finetune-1.7b 1x2048", a100, h100)):
        [case] = step.cases_from_json(json.dumps([gate.STEP_CASES[name]]))
        assert case.optimizer == "adamw" and set(case.config) <= declared
        rows = [row for row in gate.BATTERY if row.name.endswith(name.removeprefix("step "))]
        assert len(rows) == 3
        for row in rows:
            assert not gate.claimed(row)
            assert gate.hardware_status(row, device) is None
            unmeasured = gate.hardware_status(row, other)
            assert unmeasured.startswith("not measured on this hardware")
            assert gate.verdict(row, *[[unmeasured] * 6] * 3)[0] == "not measured on this hardware"


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
    """A compile row fails a head that grows it by more than the materiality `COMPILE_FLOOR` names: a
    fifth more, past its A/A, differs below its floor and passes, a half more regresses, a half less is
    faster; a step row keeps the 2% floor."""
    compile_row = gate.Row("compile lm-dense 4x1024", "s", "", floor=gate.COMPILE_FLOOR)
    step_row = next(row for row in gate.BATTERY if row.name == "step lm-dense 4x1024")
    base, tight = samples(64.0), samples(64.0, 64.5)
    assert gate.verdict(compile_row, base, samples(76.0, 77.0), tight)[0] == "different, below floor"
    assert gate.verdict(compile_row, base, samples(96.0, 97.0), tight)[0] == "regressed"
    assert gate.verdict(compile_row, base, samples(32.0, 33.0), tight)[0] == "faster"
    assert gate.verdict(step_row, base, samples(68.0, 69.0), tight)[0] == "regressed"


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
    and the device-kind probe of its hardware row alike, every process of the base and of its A/A keeps the
    caller's, and the results and the table name it."""
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
        # A row's process names its arm's cache; the head's device-kind probe, for its hardware row, is
        # told by its command, as it keeps the caller's cache.
        probe = command[1] == "-c" and "device_kind" in command[2]
        side = "probe" if probe else Path(env["JAX_COMPILATION_CACHE_DIR"]).name
        sides.setdefault(side, set()).add(env["XLA_FLAGS"])
    assert sides == {"base": {"--xla_dump_to=/caller"}, "alone": {"--xla_dump_to=/caller"}, "head": {off},
                     "probe": {off}}
    results = json.loads(out.read_text())
    assert results["head settings"] == {"XLA_FLAGS": off}
    table = tmp_path / "gate.md"
    assert gate.report(SimpleNamespace(results=out, table=str(table), claim=False)) == 1
    assert f"buffers-off ran with XLA_FLAGS='{off}'" in table.read_text()


def test_a_gate_keeps_each_arms_compiles_apart_and_places_its_arms_at_random_each_round(
        gate, tmp_path, monkeypatch):
    """A run measures three arms, the base, the head and the base again (its A/A), each with a compile cache
    and a peak cache of its own beside its results (made where absent) that it keeps across its rounds and
    removes at its end; each round's arms in an order drawn at random and kept, every arm in every round,
    every row each round; the first round's seconds and the later rounds' apart."""
    from types import SimpleNamespace

    _, rows = standing_in_environments(gate, monkeypatch)
    trees = {name: locked_tree(tmp_path / name) for name in ("main", "head")}
    out = tmp_path / "not" / "yet" / "gate.json"
    args = SimpleNamespace(base=f"main={trees['main']}", head=f"head={trees['head']}", rounds=6,
                           only=["lm-dense 4x1024"], claim=False, model="unused", flowers="unused",
                           torch_python="", out=str(out), environments=str(tmp_path / "environments"))
    assert gate.run(args) == 0
    caches = {(Path(env["JAX_COMPILATION_CACHE_DIR"]).name, env["DEW_PEAK_CACHE"]) for _, _, env in rows
              if "JAX_COMPILATION_CACHE_DIR" in env}
    assert {name for name, _ in caches} == {"base", "head", "alone"} and len(caches) == 3
    assert all(Path(peaks).name == f"{name}-peaks" for name, peaks in caches)
    assert not any(Path(peaks).parent.exists() for _, peaks in caches)
    results = json.loads(out.read_text())
    assert results["alone"] == "main~aa" and set(results["samples"]) == {"main", "head", "main~aa"}
    assert all(sorted(order) == sorted(results["samples"]) for order in results["orders"])
    assert len(results["orders"]) == 6
    for arm, taken in results["samples"].items():
        assert all("step lm-dense 4x1024" in sample for sample in taken) and len(taken) == 6, arm
    assert set(results["seconds"]) == {"first round", "later rounds"}
    assert gate.report(SimpleNamespace(results=out, table=None, claim=False)) == 0


def test_the_servers_compiles_are_measured_cold_in_every_arm_and_round(gate, tmp_path, monkeypatch):
    """The Server's warm compiles, program by program, read and write no compile cache, so each arm's every
    round is a cold sample of each, judged as any row is: each program a gated row of its own, and the whole
    warm shown beside them, ungated."""
    from types import SimpleNamespace

    _, rows = standing_in_environments(gate, monkeypatch)
    trees = {name: locked_tree(tmp_path / name) for name in ("main", "head")}
    out = tmp_path / "gate.json"
    args = SimpleNamespace(base=f"main={trees['main']}", head=f"head={trees['head']}", rounds=3,
                           only=["compile serve"], claim=False, model="unused", flowers="unused",
                           torch_python="", out=str(out), environments=str(tmp_path / "environments"))
    assert gate.run(args) == 0
    served = [command for _, command, _ in rows if Path(command[1]).name == "benchmark_serving_compile.py"]
    assert len(served) == 3 * 3
    gated = {row.name: row for probe in gate.PROBES for row in probe.rows if "compile serve" in row.name}
    assert sorted(name for name, row in gated.items() if row.gated) == sorted(
        f"compile serve dory-1.0-4b {program}"
        for program in ("construction", "admission 1", "admission 2", "admission 4", "admission 8"))
    assert not gated["compile serve dory-1.0-4b warm 16 slots"].gated


def test_compile_plans_each_step_ahead_cold_in_each_arm_and_round_and_refuses_a_substituted_program(
        gate, tmp_path, monkeypatch):
    """`compile` plans and compiles each of a kind's steps ahead of its run for one compile-only GPU of the
    kind's generation and memory, its attention XLA's in every arm and named so in the row (ledger 4616:
    the GPU's attention route lowers on no compile-only GPU), in each tree's environment of the lock the GPU
    run used, three arms a
    round in an order kept, every sample a process of its own with an empty compile cache: a head whose
    planning takes half as long again regresses past COMPILE_FLOOR, and a step whose program substituted a
    kernel is unsupported, failing the gate as the gate's own; each sample's selected plan is kept. A step
    sized for another device is not compiled for this kind."""
    from types import SimpleNamespace

    built, _ = standing_in_environments(gate, monkeypatch)
    building = gate.subprocess.run
    compiled = []

    def child(command, **kwargs):
        if Path(command[1]).name != "compiled.py":
            return building(command, **kwargs)
        case, cache = json.loads(command[2]), Path(kwargs["env"]["JAX_COMPILATION_CACHE_DIR"])
        compiled.append((Path(kwargs["cwd"]).name, command[3:], cache, cache.exists()))
        substituted = ["attention: flash ran as xla at site"] if case["config"] == gate.LM_MOE else []
        seconds = 48.0 if Path(kwargs["cwd"]).name == "head" else 32.0
        printed = json.dumps({"seconds": seconds, "substituted": substituted,
                              "plan": {"split": 1, "rows": 4}})
        return subprocess.CompletedProcess(command, 0, f"compiling\n{printed}\n", "")

    monkeypatch.setattr(gate.subprocess, "run", child)
    trees = {name: locked_tree(tmp_path / name) for name in ("main", "head")}
    out = tmp_path / "compile.json"
    args = SimpleNamespace(base=f"main={trees['main']}", head=f"head={trees['head']}", kind="h100",
                           lock="cuda12", rounds=6, only=["lm-dense", "lm-moe"], out=str(out),
                           environments=str(tmp_path / "environments"))
    assert gate.compile_battery(args) == 0
    results = json.loads(out.read_text())
    assert results["target"] == {"kind": "h100", "generation": "sm90", "capacity_gib": 80, "lock": "cuda12"}
    dense, moe = "compile lm-dense 4x1024 (xla attention)", "compile lm-moe 4x1024 (xla attention)"
    assert [row["name"] for row in results["rows"]] == [dense, moe]
    assert len(compiled) == 2 * 3 * 6
    assert {tuple(argv) for _, argv, _, _ in compiled} == {("sm90", str(80 << 30), "xla")}
    assert len({cache for _, _, cache, _ in compiled}) == len(compiled)
    assert not any(made for *_, made in compiled)
    assert {tree for tree, *_ in compiled} == {"main", "head"}
    assert built and all(python.parent.parent.name.startswith("cuda12-") for _, python in built)
    assert len(results["orders"]) == 6 and all(len(set(order)) == 3 for order in results["orders"])
    assert results["plans"]["head"][0][dense] == {"split": 1, "rows": 4}
    verdicts = tmp_path / "verdicts.json"
    assert gate.report(SimpleNamespace(results=out, table=None, claim=False, verdicts=str(verdicts))) == 1
    states = {name: row["state"] for name, row in json.loads(verdicts.read_text())["rows"].items()}
    assert states == {dense: "regressed", moe: "unsupported"}
    a100 = SimpleNamespace(**{**vars(args), "kind": "a100", "rounds": 1, "only": ["finetune"],
                              "out": str(tmp_path / "a100.json")})
    assert gate.compile_battery(a100) == 0
    assert [row["name"] for row in json.loads(Path(a100.out).read_text())["rows"]] == [
        "compile finetune-1.7b 1x2048 (xla attention)"]
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=4")
    with pytest.raises(SystemExit):
        gate.compile_battery(args)


TINY = {"architecture": "causal_transformer", "dtype": "float32", "batch_size": 2, "seq_len": 8,
        "config": {"vocab_size": 96, "emb_features": 32, "num_layers": 1, "num_heads": 2, "mlp_features": 64,
                   "max_seq_len": 16}}
"""A step small enough to plan and compile ahead in a test."""


def test_a_step_is_planned_ahead_for_an_absent_gpu_and_a_kernel_this_host_cannot_lower_is_named(
        gate, tmp_path):
    """The compile ahead's own process (`COMPILED`), run as `compile` runs it: a decoder's step planned and
    compiled for one compile-only sm90 GPU of a stated memory, here with no GPU attached, with XLA's
    attention (`AHEAD_ATTENTION`), its planner's seconds and selected plan reported and nothing substituted;
    with the attention a GPU routes to, FlashAttention, which lowers on no compile-only GPU (ledger 4616),
    it runs as XLA's, which it names as substituted, so the gate never takes that program for the GPU's."""
    script = gate._script(tmp_path / "compiled.json", gate.COMPILED)

    def compiled(attention: str) -> dict:
        argv = [str(script), json.dumps(TINY), "sm90", str(1 << 30), attention]
        done = gate._run(ROOT, Path(sys.executable), "", argv, ROOT / "tools", 900, tmp_path / attention)
        assert done.returncode == 0, done.stderr[-3000:]
        return json.loads(done.stdout.strip().splitlines()[-1])

    ahead = compiled(gate.AHEAD_ATTENTION)
    assert ahead["seconds"] > 0 and ahead["substituted"] == [] and ahead["plan"]["split"] == 1
    routed = compiled("auto")
    assert any("flash ran as xla" in each for each in routed["substituted"]), routed


def paired(rates: dict[str, list[float]], session: str = "s") -> dict[str, list[dict]]:
    """The pairing's records of three arms (`qwen35_pair.ARMS`), a round each throughput, as each step
    writes one: torch~aa a torch run, as torch."""
    workload = load("qwen35", "tools/reference_runs").workload(17, 96, 32, 2, 8, smoke=True, warmup=3,
                                                               steps=4)
    pair = load("qwen35_pair", "tools/reference_runs")

    def record(arm: str, rate: float) -> dict:
        return {"framework": pair.framework_of(arm), "session": session, "smoke": True, "commit": "c",
                "environment": {"packages": {"x": "1"}, "device": "cpu"},
                "host": {"hostname": "h", "cuda_visible_devices": None},
                "config": {**workload, "parameter_count": 1}, "work": {"flops": 1},
                "memory": {"whole_peak_scope": load("qwen35", "tools/reference_runs").WHOLE_PEAK_SCOPE,
                           "whole_peak_bytes": None},
                "checks": {"finite_losses": True, "no_timing_window_compilation": True},
                "steady": {"window_tokens_per_s": rate, "window_mfu": 0.5}}

    return {arm: [record(arm, rate) for rate in values] for arm, values in rates.items()}


def test_the_qwen_pair_judges_dew_against_torch_beside_torchs_own_a_a(gate):
    """The pairing's verdict: each round's Dew over its torch, against its torch~aa over its torch (judged):
    Dew slower past torch's A/A fails, within it is level, faster past it is faster; torch~aa is a torch run
    whose record says torch, and an A/A record of another framework is refused."""
    pair = load("qwen35_pair", "tools/reference_runs")
    torch, alone = samples(512.0), samples(512.0, 516.0)
    slower = pair.compare(paired({"dew": samples(448.0, 456.0), "torch": torch, "torch~aa": alone}), "s",
                          smoke=True)
    assert (slower["verdict"], slower["passed"], slower["rounds"]) == ("slower", False, 6)
    assert slower["interval"] == [0.875, 0.890625] and slower["alone"] == [1.0, 1.0078125]
    level = pair.compare(paired({"dew": samples(510.0, 514.0), "torch": torch, "torch~aa": alone}), "s",
                         smoke=True)
    assert (level["verdict"], level["passed"]) == ("level", True)
    faster = pair.compare(paired({"dew": samples(576.0), "torch": torch, "torch~aa": alone}), "s", smoke=True)
    assert faster["verdict"] == "faster" and faster["dew_over_torch"] == 1.125
    wrong = paired({"dew": samples(512.0), "torch": torch, "torch~aa": alone})
    wrong["torch~aa"][0]["framework"] = "dew"
    with pytest.raises(ValueError):
        pair.compare(wrong, "s", smoke=True)
