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
