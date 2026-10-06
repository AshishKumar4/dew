"""The remote CLI preserves arguments, streams output and returns the job's status."""

import importlib.util
import io
import json
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


def test_the_suite_runs_every_group_and_reports_one_outcome(tmp_path, monkeypatch, capsys):
    """Each group runs CI's selection as one pytest-split group; the report
    adds up their counts and names every failure, and a group whose stream
    ends without a summary runs again once, then counts as unfinished."""
    spec = importlib.util.spec_from_file_location("dew_remote_suite", ROOT / "tools/dew_remote.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / ".config").mkdir()
    config = {"endpoint": "https://runner.test", "token": "t"}
    (tmp_path / ".config/dew-remote.json").write_text(json.dumps(config))
    monkeypatch.setenv("HOME", str(tmp_path))
    outputs = {
        "1": ["5 passed, 9 deselected in 3.10s\n"],
        "2": ["FAILED tests/test_a.py::test_b - AssertionError\n1 failed, 4 passed, 1 warning in 2.00s\n"],
        "3": ["Collecting", "Collecting"],
    }
    requested = []

    def open_request(request, **kwargs):
        command = json.loads(request.data)["command"]
        group = command[command.index("--group") + 1]
        requested.append((command[command.index("--splits") + 1], group))
        text = outputs[group].pop(0)
        code = 0 if "passed" in text and "failed" not in text else 1
        events = [{"type": "job", "id": f"job-{group}-{len(requested)}"}, {"type": "stdout", "text": text},
                  {"type": "exit", "code": code}]
        return io.BytesIO("".join(json.dumps(event) + "\n" for event in events).encode())

    monkeypatch.setattr(module.urllib.request, "urlopen", open_request)
    monkeypatch.setattr(sys, "argv", ["dew-remote", "suite", "abc123", "--jobs", "3"])
    assert module.main() == 1
    assert sorted(requested) == [("3", "1"), ("3", "2"), ("3", "3"), ("3", "3")]
    report = capsys.readouterr().out.splitlines()
    assert report[1] == "1 failed, 9 passed"
    assert report[2] == "group 2: FAILED tests/test_a.py::test_b - AssertionError"
    assert report[3].startswith("group 3: exit 1, no pytest summary")
    assert len(report) == 4
