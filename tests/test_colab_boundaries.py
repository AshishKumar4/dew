"""Cooperating child replacement at a real fake-VM process boundary."""

import hashlib
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from tools.colab import boundary
from tools.colab.colab_batch_vm import run_logged
from tools.colab.common import read, save


def wait_file(path, seconds=5):
    deadline = time.monotonic() + seconds
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert path.exists(), "cooperative boundary was not reached"


def test_same_vm_boundary_replacement_preserves_out_and_records_lineage(tmp_path):
    home = tmp_path / "job"
    home.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    script = tmp_path / "driver.py"
    script.write_text(
        "from pathlib import Path\nfrom tools.colab.boundary import checkpoint\n"
        f"Path({str(out / 'e5.json')!r}).write_text('complete')\ncheckpoint('E5')\n"
        f"Path({str(out / 'wrong-next')!r}).touch()\n"
    )
    directory = home / "boundary"
    directory.mkdir()
    replacement = directory / "inputs" / "new.py"
    replacement.parent.mkdir()
    replacement.write_text(
        "from pathlib import Path\n"
        f"assert Path({str(out / 'e5.json')!r}).read_text()=='complete'\n"
        f"Path({str(out / 'e6.json')!r}).write_text('replaced')\n"
    )
    job = {
        "commit": "same-dew-sha",
        "command": f"{sys.executable} {script}",
        "replaceable_scripts": [str(script)],
    }
    boundary.request(directory, "E5", 0, "request")
    updates = []
    result = []
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    thread = threading.Thread(
        target=lambda: result.append(
            boundary.run(job, home, tmp_path, env, 10, run_logged, lambda **fields: updates.append(fields))
        )
    )
    thread.start()
    wait_file(directory / "ready.json")
    assert (out / "e5.json").read_text() == "complete" and not (out / "wrong-next").exists()
    save(
        directory / "decision.json",
        {
            "action": "replace",
            "generation": 0,
            "request_id": "request",
            "commit": job["commit"],
            "preflight_verified": True,
            "snapshot_acknowledged": True,
            "command": job["command"],
            "files": [
                {
                    "staged": str(replacement),
                    "target": str(script),
                    "sha256": hashlib.sha256(replacement.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    thread.join(5)
    assert not thread.is_alive() and result == [0]
    assert (out / "e6.json").read_text() == "replaced" and not (out / "wrong-next").exists()
    assert updates[0]["generation"] == 1 and updates[0]["boundary_lineage"][0]["after"] == "E5"
    assert (
        updates[0]["boundary_lineage"][0]["files"][0]["sha256"]
        == hashlib.sha256(script.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("bad", ["commit", "preflight", "target", "checksum", "escape"])
def test_replacement_refuses_unowned_or_unverified_changes(tmp_path, bad):
    directory = tmp_path / "boundary"
    directory.mkdir()
    source = directory / "inputs" / "new.py"
    source.parent.mkdir()
    source.write_text("pass")
    target = tmp_path / "original.py"
    job = {"commit": "pinned", "replaceable_scripts": [str(target)]}
    decision = {
        "generation": 0,
        "commit": "pinned",
        "preflight_verified": True,
        "files": [
            {
                "staged": str(source),
                "target": str(target),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ],
    }
    if bad == "commit":
        decision["commit"] = "different"
    elif bad == "preflight":
        decision["preflight_verified"] = False
    elif bad == "target":
        decision["files"][0]["target"] = str(tmp_path / "dataset.jsonl")
    elif bad == "checksum":
        decision["files"][0]["sha256"] = "a" * 64
    else:
        decision["files"][0]["staged"] = str(directory / "inputs" / ".." / "outside.py")
    with pytest.raises(ValueError):
        boundary.validate_replacement(job, decision, directory, 0)


def test_boundary_request_is_generation_owned_and_idempotent(tmp_path):
    boundary.request(tmp_path, "E5", 0, "first")
    with pytest.raises(ValueError, match="already"):
        boundary.request(tmp_path, "E6", 0, "duplicate")
    boundary.request(tmp_path, "E6", 1, "next")
    assert read(tmp_path / "request.json")["request_id"] == "next"


def test_child_without_opt_in_never_pauses(monkeypatch):
    monkeypatch.delenv("DEW_COLAB_BOUNDARY_DIR", raising=False)
    assert boundary.checkpoint("any") is None


def test_replacement_does_not_reset_original_job_timeout(tmp_path, monkeypatch):
    directory = tmp_path / "boundary"
    directory.mkdir()
    save(directory / "ready.json", {"request_id": "first", "generation": 0, "name": "E5"})
    save(
        directory / "decision.json",
        {
            "request_id": "first",
            "generation": 0,
            "commit": "same",
            "command": "new",
            "preflight_verified": True,
            "snapshot_acknowledged": True,
            "action": "replace",
            "files": [],
        },
    )
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    limits = []

    def logged(*args, **kwargs):
        limits.append(kwargs["timeout"])
        if len(limits) == 1:
            clock[0] += 7
            return boundary.REPLACED
        return 0

    job = {"commit": "same", "command": "original", "replaceable_scripts": []}
    assert boundary.run(job, tmp_path, tmp_path, {}, 10, logged, lambda **kwargs: None) == 0
    assert limits == [10, 3]


def test_cooperating_child_continues_after_its_decision_deadline(tmp_path):
    directory = tmp_path / "boundary"
    directory.mkdir()
    script = tmp_path / "child.py"
    script.write_text("from pathlib import Path\nfrom tools.colab.boundary import checkpoint\n"
                      "checkpoint('E5')\n" + f"Path({str(tmp_path / 'continued')!r}).touch()\n")
    boundary.request(directory, "E5", 0, "no-controller", decision_seconds=.2)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
           "DEW_COLAB_BOUNDARY_DIR": str(directory), "DEW_COLAB_GENERATION": "0"}
    code = run_logged([sys.executable, str(script)], tmp_path / "run.log", cwd=tmp_path, env=env, timeout=3)
    assert code == 0 and (tmp_path / "continued").exists()
    assert read(directory / "ready.json")["state"] == "boundary decision timed out; continued"
    with pytest.raises(ValueError, match="quiescent"):
        boundary.decide(directory, {"request_id": "no-controller", "generation": 0, "action": "replace"})


def test_expired_request_never_pauses_a_later_boundary(tmp_path, monkeypatch):
    boundary.request(tmp_path, "E5", 0, "expired", expires_at=time.time() - 1)
    monkeypatch.setenv("DEW_COLAB_BOUNDARY_DIR", str(tmp_path))
    monkeypatch.setenv("DEW_COLAB_GENERATION", "0")
    assert boundary.checkpoint("E5") is None
    assert read(tmp_path / "ready.json")["state"] == "expired before boundary; continued"


def test_late_decision_is_refused_before_publishing(tmp_path):
    save(tmp_path / "request.json", {"expires_at": time.time() - 1})
    save(tmp_path / "ready.json", {"request_id": "request", "generation": 0, "state": "quiescent",
                                   "decision_deadline": time.time() - 1})
    with pytest.raises(ValueError, match="expired"):
        boundary.decide(tmp_path, {"request_id": "request", "generation": 0, "action": "replace"})
    assert not (tmp_path / "decision.json").exists()
