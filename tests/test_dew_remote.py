"""The remote CLI preserves arguments, streams output and returns the job's status."""

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_remote_cli_streams_output_and_returns_the_remote_exit_code(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("dew_remote", ROOT / "tools/dew_remote.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / ".config"
    directory.mkdir()
    config = {"endpoint": "https://runner.test", "token": "private"}
    (directory / "dew-remote.json").write_text(json.dumps(config))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["dew-remote", "run", "main", "--python", "3.14", "--", "pytest", "-q"])
    payloads = []

    def open_request(request, **kwargs):
        payloads.append(json.loads(request.data))
        return io.BytesIO(b'{"type":"job","id":"job-1"}\n'
                          b'{"type":"stdout","text":"hello\\n"}\n'
                          b'{"type":"stderr","text":"warning\\n"}\n'
                          b'{"type":"exit","code":7}\n')

    monkeypatch.setattr(module.urllib.request, "urlopen", open_request)
    assert module.main() == 7
    assert payloads == [{"revision": "main", "python": "3.14", "command": ["pytest", "-q"]}]
    output = capsys.readouterr()
    assert output.out == "hello\n"
    assert "warning\n" in output.err
    assert (tmp_path / ".cache/dew/remote/job-1.log").read_text() == "hello\nwarning\n"


def test_the_suite_runs_every_file_once_and_reports_one_outcome(tmp_path, monkeypatch, capsys):
    """The suite packs a revision's test files by their recorded seconds, a
    file heavier than a share split into pytest-split groups of itself; the
    report adds up every command's counts and names each failure, and a
    command whose stream ends without a summary runs again once, then counts
    as unfinished."""
    spec = importlib.util.spec_from_file_location("dew_remote_suite", ROOT / "tools/dew_remote.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkout = tmp_path / "checkout"
    (checkout / "tests").mkdir(parents=True)
    for name in ("test_heavy.py", "test_light.py", "test_new.py", "test_gen_api.py"):
        (checkout / "tests" / name).write_text("")
    (checkout / "tests/test_durations.json").write_text(json.dumps(
        {"tests/test_heavy.py::test_a": 50.0, "tests/test_heavy.py::test_b": 40.0,
         "tests/test_light.py::test_c": 10.0}))
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(["git", "-C", str(checkout), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm",
                    "suite"], check=True)
    (tmp_path / ".config").mkdir()
    config = {"endpoint": "https://runner.test", "token": "t"}
    (tmp_path / ".config/dew-remote.json").write_text(json.dumps(config))
    monkeypatch.setenv("HOME", str(tmp_path))
    outputs = {
        ("tests/test_heavy.py", "1"): ["5 passed, 2 deselected in 3.10s\n"],
        ("tests/test_heavy.py", "2"): ["FAILED tests/test_heavy.py::test_b - AssertionError\n"
                                       "1 failed, 1 deselected in 2.00s\n"],
        ("tests/test_light.py", "tests/test_new.py"): ["Collecting", "Collecting"],
    }
    requested = []

    def open_request(request, **kwargs):
        command = json.loads(request.data)["command"][len(module.SUITE):]
        grouped = "--group" in command
        named = (command[0], command[command.index("--group") + 1]) if grouped else tuple(command)
        requested.append(named)
        text = outputs[named].pop(0)
        code = 0 if "passed" in text and "failed" not in text else 1
        events = [{"type": "job", "id": f"job-{len(requested)}"}, {"type": "stdout", "text": text},
                  {"type": "exit", "code": code}]
        return io.BytesIO("".join(json.dumps(event) + "\n" for event in events).encode())

    monkeypatch.setattr(module.urllib.request, "urlopen", open_request)
    assert module.suite("HEAD", "3.12", 2, checkout=checkout) == 1
    assert sorted(requested) == [("tests/test_heavy.py", "1"), ("tests/test_heavy.py", "2"),
                                 ("tests/test_light.py", "tests/test_new.py"),
                                 ("tests/test_light.py", "tests/test_new.py")]
    report = capsys.readouterr().out.splitlines()
    assert report[1] == "1 failed, 5 passed"
    assert report[2] == "group 2: FAILED tests/test_heavy.py::test_b - AssertionError"
    assert report[3].startswith("group 3: exit 1, no pytest summary")
    assert len(report) == 4
