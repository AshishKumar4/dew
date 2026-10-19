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


def standing_in(gate, monkeypatch, *, passed: bool) -> list:
    """Stand every tool of the battery in with one writing its output format,
    and return the commands run. The reference either passes its claim or
    measures a head at 0.72 of it and exits 1."""
    outputs = {
        "benchmark_step.py": lambda command: [{"ms_per_step": 5.0}],
        "benchmark_forward.py": lambda command: {after(command, "--cases"): 5.0},
        "benchmark_lm_serving.py": lambda command: {"sweep": [
            {"slots": int(slots), "repeats": [{"output_tokens_per_second": rate} for rate in (5.0, 6.0)]}
            for slots in after(command, "--slots").split(",")]},
        "benchmark_attention.py": lambda command: [
            {"implementation": implementation, "forward_backward_ms": 5.0}
            for implementation in command[command.index("--implementations") + 1:
                                          command.index("--sequence-lengths")]],
        "benchmark_image_pipeline.py": lambda command: {
            "pipeline": {"device_crop_flip_jitter": {"images_per_second": 5.0}}},
        "qwen35_pair.py": lambda command: {"comparison": {
            "dew_tokens_per_s": 720.0, "dew_over_torch": 1.0 if passed else 0.72, "tolerance": 0.01,
            "performance_claim_passed": passed}},
    }
    commands = []

    def child(command, **kwargs):
        commands.append(command)
        if command[1] == "-c":
            printed = ("NVIDIA A100-SXM4-80GB" if "device_kind" in command[2]
                       else json.dumps({"save_ms": 5.0, "restore_ms": 6.0}))
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



def test_each_tree_runs_in_the_environment_its_own_lock_installs(gate, tmp_path, monkeypatch):
    """A head that changes a locked wheel (a kernel's new version) is measured with it: each tree's
    rows run in the Python of the environment its own lock installs, by its own installer, and two
    trees whose locks agree share one environment, built once."""
    from types import SimpleNamespace

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
            rows.append((Path(kwargs["cwd"]), Path(command[0])))
            return measured_row(command, **kwargs)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(gate.subprocess, "run", child)
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
    assert {tree for tree, _ in rows} == set(trees.values())
    assert all(python == pythons[tree] for tree, python in rows)

    for lock in locks:
        shutil.copy(trees["main"] / "locks" / lock, trees["head"] / "locks" / lock)
    (shared,) = gated(tmp_path / "shared").values()
    assert len(built) == 1 and {python for _, python in rows} == {shared}


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
