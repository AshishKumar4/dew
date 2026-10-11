"""Profile parsing executes on the fake VM; the controller receives small JSON."""

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.colab import colab_sync, profile_analysis
from tools.colab.colab_batch_vm import Batch, run_logged
from tools.colab.common import read, save


@pytest.fixture(autouse=True)
def fake_memory_scope(tmp_path, monkeypatch):
    @contextlib.contextmanager
    def scope(limit):
        group = tmp_path / "memory-group"
        group.mkdir(exist_ok=True)
        (group / "memory.max").write_text(str(limit))
        (group / "memory.events").write_text("oom_kill 0\n")
        yield group

    monkeypatch.setattr(profile_analysis, "memory_scope", scope)
    monkeypatch.setattr(profile_analysis, "memory_bound", lambda: 512 * 1024 * 1024)


def file(path, body=b"fixture"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def fixture_profile(tmp_path):
    tree = tmp_path / "tree"
    capture = tree / "out/experiment/profile/capture-fixture"
    file(capture / "reports/hlo_stats.json", b"{}")
    file(capture / "plugins/profile/date/host.xplane.pb", b"raw")
    file(capture / "manifest.json", b"{}")
    file(tree / "out/experiment/hlo.tgz", b"raw hlo")
    file(tree / "out/experiment/entry.json", b'{"model_flops_per_step":123}')
    profile = {"config": {"trainer": {"profile": {"steps": 5, "directory": str(capture.parent)}}}}
    file(tree / "out/experiment/profiled/custom/run.json", json.dumps(profile).encode())
    return tree, capture


def fixture_tool(tree, failed=False):
    script = (
        "import argparse,json,pathlib,os\np=argparse.ArgumentParser()\n"
        "p.add_argument('--trace');p.add_argument('--hlo');p.add_argument('--out');"
        "p.add_argument('--steps',type=int,default=1);p.add_argument('--entry');p.add_argument('--module')\na=p.parse_args()\n"
        "assert os.environ['JAX_PLATFORMS']=='cpu' and os.environ['CUDA_VISIBLE_DEVICES']==''\n"
        "pathlib.Path(a.out).write_text(json.dumps({'time':{'by_part_ms':{'gemm':4,'layout':2},"
        "'device_idle_ms_per_step':1},'memory':{'peak_gib':20},'steps':a.steps}))\n"
        + ("raise SystemExit(1)\n" if failed else "")
    )
    file(tree / "tools/attribute_step.py", script.encode())
    for args in (
        ("init", "-q"),
        ("add", "."),
        ("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "profile-tool"),
    ):
        subprocess.run(["git", "-C", str(tree), *args], check=True)


def test_discovery_uses_metadata_steps_and_matches_experiment_hlo(tmp_path):
    tree, capture = fixture_profile(tmp_path)
    found = profile_analysis.discover({"outputs": ["out/"]}, tree)
    assert found == [
        {
            "capture": capture,
            "steps": 5,
            "entry": tree / "out/experiment/entry.json",
            "hlo": tree / "out/experiment/hlo.tgz",
        }
    ]


def test_analysis_runs_after_capture_and_retains_only_small_breakdowns(tmp_path):
    tree, _ = fixture_profile(tmp_path)
    fixture_tool(tree)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(sys.executable)
    job = {"outputs": ["out/"]}
    result = profile_analysis.analyze(job, tree, venv, tree, tmp_path, "unused", dict(os.environ), run_logged)
    assert result["complete"] and result["captures"][0]["steps"] == 5
    output = Path(result["captures"][0]["output"])
    breakdown = read(output / "attribution.json")
    assert breakdown["steps"] == 5 and breakdown["memory"]["peak_gib"] == 20
    files = colab_sync.output_files(tree, ["out/"], "measurements", include_raw=False)
    assert any(name.endswith("analysis/capture-0/attribution.json") for name in files)
    assert not any(".xplane.pb" in name or name.endswith("hlo.tgz") or "/reports/" in name for name in files)
    assert any(name.endswith("entry.json") for name in files)


def test_failed_analysis_is_visible_without_hiding_the_training_exit(tmp_path):
    tree, _ = fixture_profile(tmp_path)
    fixture_tool(tree, failed=True)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(sys.executable)
    result = profile_analysis.analyze(
        {"outputs": ["out/"]}, tree, venv, tree, tmp_path, "unused", dict(os.environ), run_logged
    )
    assert result["attempted"] and not result["complete"] and result["captures"][0]["exit"] == 1


def test_unknown_steps_are_labeled_capture_total_not_mfu(tmp_path):
    tree, _ = fixture_profile(tmp_path)
    (tree / "out/experiment/profiled/custom/run.json").unlink()
    fixture_tool(tree)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(sys.executable)
    result = profile_analysis.analyze(
        {"outputs": ["out/"]}, tree, venv, tree, tmp_path, "unused", dict(os.environ), run_logged
    )
    assert result["captures"][0]["normalization"].startswith("capture total")
    assert result["captures"][0]["steps"] is None


def test_non_profile_job_does_not_fetch_analysis_tool(tmp_path):
    tree = tmp_path / "tree"
    file(tree / "out/result.json", b"{}")
    assert profile_analysis.analyze(
        {"outputs": ["out/"]}, tree, tmp_path / "venv", tree, tmp_path, "unused", {}, run_logged
    ) == {"attempted": False, "reason": "no native device profile"}


def test_memory_bound_failure_is_recorded_without_a_parser_fallback(tmp_path, monkeypatch):
    tree, _ = fixture_profile(tmp_path)
    fixture_tool(tree)

    def unavailable(limit):
        raise PermissionError("cgroup unavailable")

    monkeypatch.setattr(profile_analysis, "memory_scope", unavailable)
    result = profile_analysis.analyze(
        {"outputs": ["out/"]}, tree, tmp_path / "venv", tree, tmp_path, "unused", {}, run_logged
    )
    assert not result["complete"]
    assert result["captures"][0]["state"] == "bounded analysis unavailable or failed"
    assert result["captures"][0]["memory_limit_bytes"] == 512 * 1024 * 1024


def test_memory_controller_oom_is_a_named_receipt_state(tmp_path):
    tree, _ = fixture_profile(tmp_path)
    fixture_tool(tree)

    def oom(args, log, **kwargs):
        (kwargs["memory_group"] / "memory.events").write_text("oom_kill 1\n")
        log.write_text("analysis killed by cgroup\n")
        return -9

    result = profile_analysis.analyze(
        {"outputs": ["out/"]}, tree, tmp_path / "venv", tree, tmp_path, "unused", {}, oom
    )
    assert not result["complete"]
    assert result["captures"][0]["state"] == "analysis exceeded its memory bound"


def test_measurement_pack_exists_before_analysis_and_failure_does_not_change_training_exit(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    file(
        repo / "job.py",
        b"from pathlib import Path\nPath('out').mkdir();Path('out/result.json').write_text('{}')\n",
    )
    for args in (
        ("init", "-q"),
        ("add", "."),
        ("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fake-job"),
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    venv = tmp_path / "venvs" / commit
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(sys.executable)
    (venv / ".ready").touch()
    save(tmp_path / "plan.json", {"driver_sha": "driver"})
    worker = Batch(tmp_path)

    def failed(*args):
        home = tmp_path / "job-1"
        assert (home / "outputs.tar.gz").is_file()
        assert read(home / "outputs.tar.gz.manifest.json")
        assert read(home / "status.json")["pre_analysis_manifest"]
        return {"attempted": True, "complete": False, "captures": [], "error": "memory bound"}

    monkeypatch.setattr(profile_analysis, "analyze", failed)
    status = worker.run_job(
        {
            "index": 1,
            "commit": commit,
            "command": "python job.py",
            "minutes": 1,
            "compile_cache": "off",
            "outputs": ["out/"],
            "artifact_policy": "measurements",
        }
    )
    assert status["exit"] == 0 and not status["profile_analysis"]["complete"]
    assert (tmp_path / "job-1/analysis.tar.gz").is_file()
