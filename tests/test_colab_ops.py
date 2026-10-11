"""Colab contracts exercised against files, child processes and a fake service."""

import base64
import contextlib
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import tarfile
import time
import types
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
import requests

from tools.colab import (
    colab_batch as batch,
    colab_batch_vm as vm,
    colab_io as transfer,
    colab_new as allocation,
    colab_sync as sync,
    launch,
    metadata_capture,
    ops_driver,
    ops_supervisor,
)
from tools.colab.common import lock, read, save
from tools.colab.ops_driver import Driver


def write(path, content=b"sample"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def archive(path, members):
    with tarfile.open(path, "w:gz") as packed:
        for name, body in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(body)
            packed.addfile(member, io.BytesIO(body))
    return path


def digests(members):
    return {name: hashlib.sha256(body).hexdigest() for name, body in members.items()}


@pytest.fixture(autouse=True)
def no_colab_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("unexpected real network request")

    monkeypatch.setattr(requests.sessions.Session, "request", refuse)


@pytest.fixture
def fake_vm(tmp_path, monkeypatch):
    remote = tmp_path / "vm"
    remote.mkdir()
    calls = []

    class Response:
        def __init__(self, payload=None, body=b"", status=200):
            self.payload, self.body, self.status_code = payload, body, status
            self.headers = {"Content-Type": "application/octet-stream"}

        def raise_for_status(self):
            if self.status_code != 200:
                raise requests.HTTPError(str(self.status_code))

        def json(self):
            return self.payload

        def iter_content(self, size):
            for offset in range(0, len(self.body), size):
                yield self.body[offset : offset + size]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def get(url, **kwargs):
        calls.append((url, kwargs))
        route = unquote(urlparse(url).path)
        relative = route.removeprefix("/api/contents/").removeprefix("/files/")
        path = remote / relative
        if not path.exists():
            return Response(status=404)
        if route.startswith("/files/"):
            return Response(body=path.read_bytes())
        if path.is_dir():
            return Response(
                {"type": "directory", "content": [{"name": child.name} for child in path.iterdir()]}
            )
        body = path.read_bytes()
        return Response(
            {
                "type": "file",
                "size": len(body),
                "format": "base64",
                "content": base64.b64encode(body).decode(),
            }
        )

    monkeypatch.setattr(requests, "get", get)
    return remote, calls, allocation.Session("owned", "https://fake.invalid/", "private")


def test_measurements_keep_nested_logs_but_never_checkpoint_weights(tmp_path):
    for name in (
        "out/case/train.log",
        "out/case/run/tracking/scalars.jsonl",
        "out/case/run/run.json",
        "out/case/run/3/weights",
        "out/case/run/4.tmp/weights",
        "out/case/run/frozen/weights",
    ):
        write(tmp_path / name)
    files = sync.output_files(tmp_path, ["out/"], "measurements")
    assert set(files) == {
        "out/case/train.log",
        "out/case/run/tracking/scalars.jsonl",
        "out/case/run/run.json",
    }
    assert (tmp_path / "out/case/run/3/weights").exists()


def test_checkpoint_step_requires_both_commit_markers(tmp_path):
    step = tmp_path / "out/run/3"
    write(step / "weights")
    write(step / "commit_success.txt")
    assert sync.output_files(tmp_path, ["out/"]) == {}
    write(step / "_CHECKPOINT_METADATA")
    assert len(sync.output_files(tmp_path, ["out/"])) == 3


def test_snapshot_roundtrip_acknowledges_only_verified_bytes(tmp_path):
    tree = tmp_path / "source"
    write(tree / "out/a.log", b"first")
    cache = tmp_path / "cache"
    packed = tmp_path / "delta.tar.gz"
    current, changed = sync.snapshot(tree, ["out/"], {}, packed)
    assert changed == 1
    sync.apply_snapshot(packed, cache / "outputs", current, manifest=cache / "manifest.json")
    write(tree / "out/a.log", b"second")
    after, changed = sync.snapshot(tree, ["out/"], current, packed)
    sync.apply_snapshot(packed, cache / "outputs", after, current, cache / "manifest.json")
    assert changed == 1 and (cache / "outputs/out/a.log").read_bytes() == b"second"
    restored = tmp_path / "resume.tar.gz"
    sync.pack_retained(cache, restored)
    sync.restore(restored, tmp_path / "fresh", ["out/"])
    assert (tmp_path / "fresh/out/a.log").read_bytes() == b"second"


@pytest.mark.parametrize(
    "members,expected",
    [
        ({"out/a": b"bad"}, {"out/a": b"good"}),
        ({}, {"out/a": b"good"}),
        ({"../escape": b"bad"}, {"../escape": b"bad"}),
    ],
)
def test_bad_snapshot_does_not_promote_or_acknowledge(tmp_path, members, expected):
    output = tmp_path / "outputs"
    write(output / "out/a", b"old")
    manifest = tmp_path / "manifest.json"
    save(manifest, {"out/a": "old"})
    with pytest.raises(ValueError):
        sync.apply_snapshot(
            archive(tmp_path / "delta.tar.gz", members), output, digests(expected), manifest=manifest
        )
    assert (output / "out/a").read_bytes() == b"old" and read(manifest) == {"out/a": "old"}


def test_resume_rejects_interrupted_unacknowledged_promotion(tmp_path):
    write(tmp_path / "outputs/out/log", b"new-unacknowledged")
    save(tmp_path / "manifest.json", digests({"out/log": b"old"}))
    with pytest.raises(ValueError, match="unacknowledged"):
        sync.pack_retained(tmp_path, tmp_path / "resume.tar.gz")


def test_declared_outputs_cannot_leak_an_external_symlink(tmp_path):
    write(tmp_path / "secret", b"private")
    (tmp_path / "out").mkdir()
    (tmp_path / "out/link").symlink_to(tmp_path / "secret")
    with pytest.raises(ValueError, match="escapes"):
        sync.output_files(tmp_path, ["out/"])


def test_restore_rejects_undeclared_member_and_symlink_parent(tmp_path):
    packed = archive(tmp_path / "resume.tar.gz", {"secret": b"bad"})
    with pytest.raises(ValueError, match="outside"):
        sync.restore(packed, tmp_path / "tree", ["out/"])
    (tmp_path / "tree").mkdir()
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "tree/out").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    packed = archive(packed, {"out/log": b"bad"})
    with pytest.raises(ValueError, match="symlink"):
        sync.restore(packed, tmp_path / "tree", ["out/"])


@pytest.mark.parametrize("large", [False, True])
def test_owned_download_is_atomic_without_a_backing_spool(tmp_path, fake_vm, monkeypatch, large):
    remote, calls, session = fake_vm
    body = b"payload" * 30
    write(remote / "content/result.json", body)
    if large:
        monkeypatch.setattr(transfer, "SMALL_LIMIT", 10)
    target = tmp_path / "target"
    transfer.download(session, "/content/result.json", target)
    assert target.read_bytes() == body
    assert any("/files/" in url for url, _ in calls) == large
    assert all("private" not in url for url, _ in calls)
    transfer.reconcile_transfers((tmp_path,))
    assert target.read_bytes() == body and not target.is_symlink()
    assert list(tmp_path.glob(".*.part")) == []


def test_dead_partial_cleanup_preserves_unowned_files(tmp_path):
    partial = write(tmp_path / f".target.colab-{os.getpid()}-0-dead.part")
    live = write(tmp_path / f".target.colab-{os.getpid()}-{transfer.process_start(os.getpid())}-live.part")
    unrelated = write(tmp_path / "unowned")
    forged = tmp_path / ".target.colab-0-0-forged.part"
    forged.symlink_to(unrelated)
    transfer.reconcile_transfers((tmp_path,))
    assert not partial.exists() and live.exists() and forged.is_symlink()
    assert unrelated.read_bytes() == b"sample"


def test_failed_download_keeps_old_target(tmp_path, fake_vm, monkeypatch):
    remote, _, session = fake_vm
    write(remote / "result", b"new")
    target = write(tmp_path / "target", b"old")
    original = requests.get

    def truncated(url, **kwargs):
        response = original(url, **kwargs)
        if kwargs.get("params", {}).get("content") == "1":
            response.json()["content"] = base64.b64encode(b"x").decode()
        return response

    monkeypatch.setattr(requests, "get", truncated)
    with pytest.raises(ValueError, match="changed"):
        transfer.download(session, "/result", target)
    assert target.read_bytes() == b"old" and list(tmp_path.glob(".*.part")) == []


def test_exact_branch_ref_is_not_replaced_by_later_base(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "first",
    )
    first = git("rev-parse", "HEAD")
    git("tag", "proof")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "later",
    )
    assert batch.exact_ref(tmp_path, "proof") == first != git("rev-parse", "HEAD")


@pytest.mark.parametrize("field", ["commit", "command", "outputs", "artifact_policy", "ref"])
def test_staging_reuse_rejects_changed_job_identity(field):
    plan = {
        "driver_sha": "driver",
        "base": "base",
        "hardware": "gpu:A100",
        "jobs": [
            {
                "commit": "a",
                "command": "run",
                "outputs": ["out/"],
                "artifact_policy": "all",
                "ref": "a",
                "compile_cache": "off",
            }
        ],
    }
    changed = json.loads(json.dumps(plan))
    changed["jobs"][0][field] = "changed"
    with pytest.raises(ValueError, match="identity"):
        batch.validate_reuse(plan, changed)


def test_name_lock_excludes_a_second_process(tmp_path):
    path = tmp_path / "session.lock"
    script = "import fcntl,sys; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
    with lock(path, blocking=False):
        result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True)
    assert result.returncode != 0 and b"BlockingIOError" in result.stderr
    assert subprocess.run([sys.executable, "-c", script, str(path)], check=False).returncode == 0


@pytest.fixture
def fake_colab(monkeypatch, tmp_path):
    held = []
    named = {}
    released = []
    cli = types.SimpleNamespace(
        list_assignments=lambda: held, unassign=lambda endpoint: released.append(endpoint)
    )
    store = types.SimpleNamespace(
        get=named.get, remove=lambda key: named.pop(key), add=lambda item: named.update({item.name: item})
    )
    monkeypatch.setitem(
        sys.modules,
        "colab_cli.common",
        types.SimpleNamespace(state=types.SimpleNamespace(client=cli, store=store)),
    )
    member = types.SimpleNamespace(name="GPU", value="GPU")
    shape = types.SimpleNamespace(name="STANDARD", value="STANDARD")
    enums = types.SimpleNamespace(
        Variant={"GPU": member}, Accelerator={"GPU": member}, Shape={"STANDARD": shape}
    )
    monkeypatch.setitem(sys.modules, "colab_cli.client", enums)
    monkeypatch.setitem(
        sys.modules,
        "colab_cli.state",
        types.SimpleNamespace(SessionState=lambda **kwargs: types.SimpleNamespace(**kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "colab_cli.commands.session",
        types.SimpleNamespace(resolve_runtime_options=lambda *args, **kwargs: (member, member, shape)),
    )
    monkeypatch.setattr(allocation, "STATE", tmp_path)
    return allocation.Colab(), held, named, released, cli


def test_allocator_records_uuid_before_post_and_reuses_it_on_timeout(tmp_path, fake_colab, monkeypatch):
    client, _, _, _, service = fake_colab
    journal = tmp_path / "allocation.json"
    seen = []

    def assign(notebook, **kwargs):
        seen.append(str(notebook))
        assert read(journal)["notebook"] == str(notebook)
        if len(seen) == 1:
            raise requests.ReadTimeout()
        return types.SimpleNamespace(
            endpoint="owned",
            runtime_proxy_info=types.SimpleNamespace(
                token="secret", url="https://fake.invalid/", expires_at=lambda: 123
            ),
        )

    service.assign = assign
    monkeypatch.setattr(allocation.time, "sleep", lambda seconds: None)
    assert client.allocate("test", "gpu:A100", high_mem=False, journal=journal) == "owned"
    assert len(seen) == 2 and len(set(seen)) == 1 and "secret" not in journal.read_text()


def test_allocator_refuses_full_account_before_request(tmp_path, fake_colab):
    client, held, _, _, _ = fake_colab
    held.extend(types.SimpleNamespace(endpoint=str(index)) for index in range(4))
    with pytest.raises(RuntimeError, match="no slot"):
        client.allocate("test", "gpu:A100", high_mem=False, journal=tmp_path / "allocation.json")
    assert not (tmp_path / "allocation.json").exists()


def test_release_never_unassigns_a_reused_name(fake_colab):
    client, held, named, released, _ = fake_colab
    held.extend([types.SimpleNamespace(endpoint="owned"), types.SimpleNamespace(endpoint="new")])
    named["test"] = types.SimpleNamespace(endpoint="new")
    assert client.release("test", "owned") == "released"
    assert released == ["owned"] and named["test"].endpoint == "new"


def test_cu_accounting_charges_release_lag_and_setup_once():
    run = {"rate": 6}
    allocation = {"requested_at": 100, "assigned_at": 200}
    driver = {"progress": {"phase": "done"}, "results": [{"stage_seconds": 100}, {"stage_seconds": 200}]}
    result = ops_supervisor.accounting(run, allocation, driver, 800, 999)
    assert result["session_seconds"] == 600 and result["cu_estimate"] == 1
    assert result["cu_upper_bound"] == pytest.approx(700 / 600)
    assert sum(result["per_job_cu"]) == result["cu_estimate"] and not result["cu_provisional"]
    driver["progress"]["phase"] = "failed"
    result = ops_supervisor.accounting(run, allocation, driver, None, 900)
    assert result["cu_provisional"] and "per_job_cu" not in result


def test_emergency_metadata_is_bounded_and_excludes_weights(tmp_path, fake_vm):
    remote, calls, session = fake_vm
    write(remote / "content/out/case/train.log", b"training")
    write(remote / "content/out/case/run/3/weights.json", b"private weights")
    write(remote / "content/out/large.log", b"x" * (4 * 1024 * 1024 + 1))
    plan = {"driver_sha": "driver", "jobs": [{"index": 1, "outputs": ["/content/out/"]}]}
    receipt = metadata_capture.capture(tmp_path / "run", plan, session, "/content/batch")
    assert receipt["bytes"] == len(b"training")
    assert [file["remote"] for file in receipt["files"]] == ["/content/out/case/train.log"]
    before = len(calls)
    assert metadata_capture.capture(tmp_path / "run", plan, session, "/content/batch") == receipt
    assert len(calls) == before
    empty = metadata_capture.capture(tmp_path / "zero", plan, session, "/content/batch", seconds=0)
    assert empty["files"] == [] and len(calls) == before


@pytest.mark.parametrize("exit_code", [0, 1])
def test_profile_preamble_flags_failure_without_blocking_jobs(tmp_path, exit_code):
    tree, venv, output = tmp_path / "tree", tmp_path / "venv", tmp_path / "out"
    script = (
        "import sys,json,pathlib\np=pathlib.Path(sys.argv[2]);p.mkdir(exist_ok=True,parents=True)\n"
        f"(p/'check.json').write_text(json.dumps({{'exit':{exit_code}}}))\nsys.exit({exit_code})\n"
    )
    write(tree / "tools/gpu_profile_check.py", script.encode())
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(sys.executable)
    receipt = vm.profile_preamble(tree, venv, output)
    assert receipt["exit"] == exit_code and receipt["broken"] == bool(exit_code)
    assert json.loads((output / "check.json").read_text())["exit"] == exit_code
    (tree / "tools/gpu_profile_check.py").unlink()
    assert vm.profile_preamble(tree, venv, output) == receipt
    assert not vm.profile_preamble(tree, venv, tmp_path / "absent")["attempted"]


def test_parts_hash_mismatch_never_overwrites_destination(tmp_path):
    write(tmp_path / "part", b"bad")
    destination = write(tmp_path / "target", b"old")
    with pytest.raises(ValueError, match="hash mismatch"):
        vm.join_parts(tmp_path, ["part"], destination, "wrong")
    assert destination.read_bytes() == b"old"


def test_driver_upload_uses_fake_cli_and_preserves_binary_bytes(tmp_path):
    fake = tmp_path / "colab"
    fake.write_text(
        f"#!{sys.executable}\nimport pathlib,shutil,sys\n"
        "assert sys.argv[1:3]==['upload','-s']\nshutil.copyfile(sys.argv[4],sys.argv[5])\n"
    )
    fake.chmod(0o700)
    source = write(tmp_path / "source", bytes(range(256)))
    driver = Driver(tmp_path / "run", None, str(fake))
    driver.upload("fake", source, str(tmp_path / "remote"))
    assert (tmp_path / "remote").read_bytes() == source.read_bytes()


def test_staging_checksum_change_is_rejected_before_allocation(tmp_path):
    plan = {"driver_sha": "driver", "base": "base", "hardware": "gpu:A100", "jobs": []}
    part = write(tmp_path / "bundle.part", b"approved")
    save(tmp_path / "plan.json", {**plan, "staged_files": digests({"bundle.part": b"approved"})})
    part.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        batch.staged(plan, tmp_path)


def test_snapshot_interruption_never_acknowledges_partial_promotion(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    original = {"out/a": b"old", "out/b": b"old"}
    current = {"out/a": b"new", "out/b": b"new"}
    for name, value in original.items():
        write(cache / "outputs" / name, value)
    save(cache / "manifest.json", digests(original))
    packed = archive(tmp_path / "delta.tar.gz", current)
    replace = os.replace

    def interrupted(source, target):
        if Path(target) == cache / "outputs/out/b":
            raise OSError("simulated host loss")
        return replace(source, target)

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(OSError, match="host loss"):
        sync.apply_snapshot(
            packed, cache / "outputs", digests(current), digests(original), cache / "manifest.json"
        )
    assert read(cache / "manifest.json") == digests(original)
    with pytest.raises(ValueError, match="unacknowledged"):
        sync.pack_retained(cache, tmp_path / "resume.tar.gz")


def test_session_budget_uses_provisioning_time_and_no_inferred_estimate():
    run = {"rate": 6, "budget_cu": 1}
    assert ops_supervisor.remaining_budget(run, {"requested_at": 100}, 650) == 50
    assert ops_supervisor.remaining_budget(run, {"requested_at": 100}, 800) == 0
    assert ops_supervisor.remaining_budget({"rate": 6}, {"requested_at": 100}, 800) == float("inf")


def test_finalizer_reports_failed_release_as_provisional_and_retries(tmp_path, fake_colab, monkeypatch):
    client, held, _, released, service = fake_colab
    save(tmp_path / "run.json", {"session": "test", "driver_sha": "driver", "rate": 6, "started_at": 0})
    save(tmp_path / "driver.json", {"endpoint": "owned", "progress": {"phase": "done"}})
    save(tmp_path / "allocation.json", {"endpoint": "owned", "requested_at": 100, "assigned_at": 110})
    held.append(types.SimpleNamespace(endpoint="owned"))

    def unavailable(endpoint):
        raise requests.ConnectionError("service down")

    service.unassign = unavailable
    result = ops_supervisor.finalize(tmp_path, client, "controller died", 1)
    assert result["cu_provisional"] and result["release"].startswith("failed")
    service.unassign = lambda endpoint: released.append(endpoint)
    due = result["release_retry_at"] + 1
    monkeypatch.setattr(time, "time", lambda: due)
    result = ops_supervisor.finalize(tmp_path, client, "retry", 1)
    assert result["release"] == "released" and not result["cu_provisional"] and released == ["owned"]
    assert ops_supervisor.finalize(tmp_path, client) == result and released == ["owned"]


def test_allocation_lock_is_held_across_request(tmp_path, fake_colab):
    client, _, _, _, service = fake_colab
    script = "import fcntl,sys; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"

    def assign(*args, **kwargs):
        attempted = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / "colab-allocation.lock")], capture_output=True
        )
        assert attempted.returncode != 0 and b"BlockingIOError" in attempted.stderr
        return types.SimpleNamespace(
            endpoint="owned",
            runtime_proxy_info=types.SimpleNamespace(
                token="secret", url="https://fake.invalid/", expires_at=lambda: 123
            ),
        )

    service.assign = assign
    assert client.allocate("test", "gpu:A100", high_mem=False, journal=tmp_path / "journal.json") == "owned"


def test_unobserved_job_stays_held_and_unrelated_queue_changes_survive(tmp_path, monkeypatch):
    queue = tmp_path / "queue.md"
    text = (
        "## Pending\n- lane: test first\n  device: gpu\n  ref: one\n  command: run\n"
        "  outputs: out/\n\n- lane: other second\n  device: gpu\n  ref: two\n"
        "  command: other\n  outputs: other/\n\n## Done\n"
    )
    queue.write_text(text)
    monkeypatch.setattr(batch, "QUEUE", queue)
    entry = batch.entries(batch.sections()[1])[0]
    batch.hold_unreported({"jobs": [{"index": 1, "entry": entry}]}, set(), "lost assignment")
    first, second = batch.entries(batch.sections()[1])
    assert "owner approval" in first["fields"]["hold"] and "hold" not in second["fields"]
    batch.finish(entry, {"status": "exit 1", "driver_sha": "driver"})
    assert len(batch.entries(batch.sections()[1])) == 1
    assert batch.entries(batch.sections()[2])[0]["fields"]["status"] == "exit 1"


def test_child_timeout_does_not_leave_profile_grandchild_running(tmp_path):
    marker = tmp_path / "late-output"
    script = (
        "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',"
        f"{('import time,pathlib;time.sleep(1);pathlib.Path(' + repr(str(marker)) + ').touch()')!r}]);"
        "time.sleep(20)"
    )
    result = vm.run_logged(
        [sys.executable, "-c", script], tmp_path / "run.log", timeout=0.2, cwd=tmp_path, env=dict(os.environ)
    )
    assert str(result).startswith("timeout")
    # The read occurs after the grandchild would have produced its marker.
    subprocess.run([sys.executable, "-c", "import time;time.sleep(1.1)"], check=True)
    assert not marker.exists()


def test_incomplete_retained_checkpoint_is_never_released_with_commit_markers(tmp_path):
    members = {
        "out/run/3/weights": b"weights",
        "out/run/3/commit_success.txt": b"ok",
        "out/run/3/_CHECKPOINT_METADATA": b"metadata",
        "out/train.log": b"log",
    }
    for name, value in members.items():
        write(tmp_path / "outputs" / name, value)
    save(tmp_path / "manifest.json", digests(members))
    (tmp_path / "outputs/out/run/3/weights").unlink()
    packed = tmp_path / "resume.tar.gz"
    sync.pack_retained(tmp_path, packed)
    with tarfile.open(packed) as archive:
        assert archive.getnames() == ["out/train.log"]


def test_fake_vm_runs_pinned_jobs_after_one_nonblocking_preamble(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    profile = (
        "from pathlib import Path\nimport sys\n"
        "out=Path(sys.argv[2]);out.mkdir(exist_ok=True,parents=True)\n"
        "(out/'check.json').write_text('{}')\n"
        "with (out/'calls').open('a') as f:f.write('capture\\n')\nsys.exit(1)\n"
    )
    job_script = (
        "from pathlib import Path\n"
        "assert (Path.cwd().parents[1]/'out/_preamble/profile-check/check.json').is_file()\n"
        "Path('out').mkdir();Path('out/result.json').write_text('{\"step\":3}')\n"
    )
    write(repo / "tools/gpu_profile_check.py", profile.encode())
    write(repo / "job.py", job_script.encode())
    for args in (
        ("init", "-q"),
        ("add", "."),
        ("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"),
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    venv = tmp_path / "venvs" / commit
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(sys.executable)
    (venv / ".ready").touch()
    save(tmp_path / "plan.json", {"driver_sha": "driver"})
    worker = vm.Batch(tmp_path)
    for index in (1, 2):
        status = worker.run_job(
            {
                "index": index,
                "commit": commit,
                "command": "python job.py",
                "minutes": 1,
                "compile_cache": "off",
                "outputs": ["out/"],
                "artifact_policy": "measurements",
            }
        )
        assert status["exit"] == 0 and status["commit"] == commit and status["driver_sha"] == "driver"
        home = tmp_path / f"job-{index}"
        manifest = read(home / "outputs.tar.gz.manifest.json")
        sync.apply_snapshot(home / "outputs.tar.gz", home / "collected", manifest)
        assert read(home / "collected/out/result.json") == {"step": 3}
    assert worker.progress["profile_preamble"]["broken"]
    assert (tmp_path / "out/_preamble/profile-check/calls").read_text() == "capture\n"


def test_emergency_capture_deadline_interrupts_a_slow_service(tmp_path, monkeypatch):
    def slow(*args, **kwargs):
        time.sleep(30)
        raise AssertionError("request should have been interrupted")

    monkeypatch.setattr(requests, "get", slow)
    began = time.monotonic()
    receipt = metadata_capture.capture(
        tmp_path,
        {"driver_sha": "driver", "jobs": []},
        allocation.Session("owned", "https://fake.invalid/", "secret"),
        "/content/batch",
        seconds=0.1,
    )
    assert receipt["timed_out"] and receipt["capture_exit"] != 0
    assert time.monotonic() - began < 3


def test_review_transient_progress_and_collection_failures_preserve_running_batch(
    tmp_path, fake_vm, monkeypatch
):
    remote, _, session = fake_vm
    status = {
        "exit": 0,
        "seconds": 1,
        "stage_seconds": 1,
        "artifact_manifest": "outputs.tar.gz.manifest.json",
    }
    save(remote / "content/batch/progress.json", {"phase": "done", "jobs": {"1": status}})
    save(remote / "content/batch/job-1/status.json", status)
    write(remote / "content/batch/job-1/run.log", b"completed")
    write(remote / "content/batch/job-1/dew/out/result.json", b'{"step":3}')
    queue = tmp_path / "queue.md"
    queue.write_text(
        "## Pending\n- lane: test job\n  device: gpu\n  ref: ref\n"
        "  command: run\n  outputs: out/\n\n## Done\n"
    )
    monkeypatch.setattr(batch, "QUEUE", queue)
    monkeypatch.setattr(ops_driver, "STATE", tmp_path / "state")
    monkeypatch.setattr(batch, "STATE", tmp_path / "state")
    failures = {"progress": 0, "collection": 0, "kernel": 0}
    original = requests.get

    def intermittent(url, **kwargs):
        kind = (
            "progress"
            if url.endswith("progress.json")
            else "collection"
            if url.endswith("sync.tar.gz")
            else None
        )
        if kind and failures[kind] < 2:
            failures[kind] += 1
            raise requests.ConnectionError("temporary connection loss")
        return original(url, **kwargs)

    class Kernel:
        def execute_code(self, script, **kwargs):
            if failures["kernel"] < 1:
                failures["kernel"] += 1
                raise RuntimeError("WebSocket channel closed")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                exec(script.replace("/content", str(remote / "content")), {})
            return [{"output_type": "stream", "text": output.getvalue()}]

        def stop(self, **kwargs):
            return None

    client = types.SimpleNamespace(session=lambda endpoint: session, kernel=lambda *args, **kwargs: Kernel())
    pauses = []
    monkeypatch.setattr(requests, "get", intermittent)
    monkeypatch.setattr(time, "sleep", pauses.append)
    job = {
        "index": 1,
        "lane": "test",
        "resume_key": "key",
        "outputs": ["out/"],
        "artifact_policy": "measurements",
        "commit": "sha",
        "entry": batch.entries(batch.sections()[1])[0],
    }
    driver = Driver(tmp_path / "run", client, "unused")
    driver.poll(
        "owned", "/content/batch", {"base": "base", "device": "gpu", "driver_sha": "driver", "jobs": [job]}
    )
    assert failures == {"progress": 2, "collection": 2, "kernel": 1}
    assert driver.state["results"][0]["observed_exit"] == 0
    artifact = Path(driver.state["results"][0]["results"]) / "outputs/out/result.json"
    assert read(artifact) == {"step": 3} and len(pauses) >= 5


def fixture_repository(tmp_path, *, large=False):
    repo, origin = tmp_path / "repo", tmp_path / "origin.git"
    repo.mkdir()
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    if large:
        write(repo / "history.bin", random.Random(1).randbytes(200000))
    else:
        write(repo / "base", b"base")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", str(origin)], check=True)
    subprocess.run(["git", "-C", str(repo), "push", "-q", "origin", "HEAD:main"], check=True)
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return repo, origin, sha


def request(ref="HEAD", command="run", policy="all"):
    fields = {
        "lane": "test",
        "command": command,
        "outputs": ["out/"],
        "ref": ref,
        "gpu_reason": "test",
        "expected_cu": "0.1",
        "decision": "test",
        "memory_estimate": "1GiB",
        "artifact_policy": policy,
    }
    return {"text": "- lane: test job\n", "fields": fields}


def test_review_resume_identity_changes_with_commit_and_command(tmp_path):
    repo, _, sha = fixture_repository(tmp_path)
    first = batch.plan_jobs([request()], "gpu", "gpu:T4", sha, repo)["jobs"][0]["resume_key"]
    other_job = batch.plan_jobs([request(command="other")], "gpu", "gpu:T4", sha, repo)["jobs"][0][
        "resume_key"
    ]
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "new",
        ],
        check=True,
    )
    moved = batch.plan_jobs([request()], "gpu", "gpu:T4", sha, repo)["jobs"][0]["resume_key"]
    assert len({first, other_job, moved}) == 3


def test_review_measurements_never_restore_old_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "STATE", tmp_path / "state")
    cache = tmp_path / "state/colab-resume/key"
    write(cache / "outputs/out/result.json", b"old result")
    save(cache / "manifest.json", digests({"out/result.json": b"old result"}))
    staging = tmp_path / "staging"
    staging.mkdir()
    job = {
        "index": 1,
        "entry": request(policy="measurements"),
        "resume_key": "key",
        "artifact_policy": "measurements",
    }
    batch.stage_inputs({"uploads": {}, "jobs": [job]}, staging)
    assert not job.get("resume_parts") and list(staging.iterdir()) == []


def test_review_reconciler_preserves_first_failure_and_bounds_release_retries(
    tmp_path, fake_colab, monkeypatch
):
    client, held, _, _, service = fake_colab
    clock = [1000.0]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    save(tmp_path / "run.json", {"session": "test", "driver_sha": "driver", "rate": 1, "started_at": 0})
    save(tmp_path / "driver.json", {"endpoint": "owned", "progress": {"phase": "done"}})
    save(tmp_path / "allocation.json", {"endpoint": "owned", "requested_at": 100, "assigned_at": 110})
    held.append(types.SimpleNamespace(endpoint="owned"))
    attempts = []

    def unavailable(endpoint):
        attempts.append(endpoint)
        raise requests.ConnectionError("temporary failure")

    service.unassign = unavailable
    ops_supervisor.finalize(tmp_path, client, "original failure", 17)
    for _ in range(8):
        clock[0] += 3600
        receipt = ops_supervisor.finalize(tmp_path, client, "reconciler", None)
    assert receipt["reason"] == "original failure" and receipt["driver_exit"] == 17
    assert len(attempts) <= 4 and "attention" in receipt["release"]
    assert receipt["cleanup_history"] and receipt["cu_provisional"]


def test_review_published_sources_do_not_upload_a_bundle(tmp_path, monkeypatch):
    repo, _, sha = fixture_repository(tmp_path)
    monkeypatch.setattr(batch, "STATE", tmp_path / "state")
    plan = batch.plan_jobs([request(sha)], "gpu", "gpu:T4", sha, repo)
    netrc_file = write(tmp_path / "netrc", b"machine api.wandb.ai login test password test-key\n")
    batch.stage(plan, tmp_path / "staging", repo, netrc_file=netrc_file)
    assert not list((tmp_path / "staging").glob("bundle*"))
    assert plan["fetch_commits"] == [sha]


def test_review_unpushed_bundle_contains_only_the_new_range(tmp_path, monkeypatch):
    repo, _, sha = fixture_repository(tmp_path, large=True)
    write(repo / "new", b"small delta")
    subprocess.run(["git", "-C", str(repo), "add", "new"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "unpublished",
        ],
        check=True,
    )
    monkeypatch.setattr(batch, "STATE", tmp_path / "state")
    plan = batch.plan_jobs([request()], "gpu", "gpu:T4", sha, repo)
    netrc_file = write(tmp_path / "netrc", b"machine api.wandb.ai login test password test-key\n")
    batch.stage(plan, tmp_path / "staging", repo, netrc_file=netrc_file)
    assert sum(file.stat().st_size for file in (tmp_path / "staging").glob("bundle*")) < 10000
    assert sha in plan["fetch_commits"]


def test_review_completed_download_is_a_regular_file(tmp_path, fake_vm):
    remote, _, session = fake_vm
    write(remote / "result", b"result")
    target = tmp_path / "result"
    transfer.download(session, "/result", target)
    assert target.read_bytes() == b"result" and not target.is_symlink()


def test_review_cuda_selection_respects_the_gpu_sm_floor(tmp_path, monkeypatch):
    tree, venv, binaries = tmp_path / "tree", tmp_path / "venv", tmp_path / "bin"
    tree.mkdir()
    binaries.mkdir()
    write(tree / "pyproject.toml", b"[project.optional-dependencies]\ncuda12=[]\ncuda13=[]\n")
    gpu = binaries / "nvidia-smi"
    gpu.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "print('580.1, 7.0' if 'compute_cap' in str(sys.argv) else '580.1')\n"
    )
    gpu.chmod(0o700)
    uv = binaries / "uv"
    log = tmp_path / "uv.log"
    uv.write_text(
        f"#!{sys.executable}\nimport pathlib,sys\n"
        f"with pathlib.Path({str(log)!r}).open('a') as f:f.write(str(sys.argv)+'\\n')\n"
        "if sys.argv[1]=='venv':\n p=pathlib.Path(sys.argv[-1])/'bin';p.mkdir(parents=True)\n"
        " (p/'python').write_text('#!/bin/sh\\nexit 0\\n');(p/'python').chmod(0o700)\n"
    )
    uv.chmod(0o700)
    monkeypatch.setenv("PATH", str(binaries) + ":" + os.environ["PATH"])
    vm.install(venv, tree, tmp_path / "setup.log")
    assert "cuda12" in log.read_text() and "cuda13" not in log.read_text()


def test_network_outage_is_not_proof_that_the_workload_died(tmp_path, monkeypatch):
    clock = [1000.0]
    polls = []
    monkeypatch.setattr(time, "time", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    def poll():
        polls.append(1)
        clock[0] += 360
        save(tmp_path / "driver.json", {"endpoint": "owned", "heartbeat": clock[0]})
        return 0 if len(polls) > 2 else None

    def unavailable():
        raise requests.ConnectionError("control API unavailable")

    reason = ops_supervisor.watch(
        types.SimpleNamespace(poll=poll),
        tmp_path,
        {"started_at": 1000},
        types.SimpleNamespace(assignments=unavailable),
    )
    assert reason is None and read(tmp_path / "watchdog-attention.json")["notify"] == "Main"


def test_publish_retained_excludes_stale_files_outside_the_acknowledged_manifest(tmp_path):
    cache = tmp_path / "cache"
    write(cache / "outputs/out/old-result.json", b"old")
    write(cache / "outputs/out/current.log", b"current")
    save(cache / "manifest.json", digests({"out/current.log": b"current"}))
    sync.publish_retained(cache, tmp_path / "receipt")
    assert (tmp_path / "receipt/out/current.log").read_bytes() == b"current"
    assert not (tmp_path / "receipt/out/old-result.json").exists()


@pytest.mark.parametrize("unpublished", [False, True])
def test_vm_fetches_exact_shallow_sources_with_optional_delta_bundle(tmp_path, monkeypatch, unpublished):
    repo, origin, sha = fixture_repository(tmp_path)
    if unpublished:
        write(repo / "unpublished", b"new")
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "unpushed",
            ],
            check=True,
        )
    monkeypatch.setattr(batch, "STATE", tmp_path / "state")
    plan = batch.plan_jobs([request()], "gpu", "gpu:T4", sha, repo)
    staging = tmp_path / "staging"
    netrc_file = write(tmp_path / "netrc", b"machine api.wandb.ai login test password test-key\n")
    batch.stage(plan, staging, repo, netrc_file=netrc_file)

    worker = vm.Batch(staging)
    worker.prepare()
    for expected in (sha, plan["jobs"][0]["commit"]):
        subprocess.run(["git", "-C", str(worker.repo), "cat-file", "-e", expected + "^{commit}"], check=True)
    assert (worker.repo / ".git/shallow").is_file()
    assert subprocess.check_output(
        ["git", "-C", str(worker.repo), "remote", "get-url", "origin"], text=True
    ).strip() == str(origin)


@pytest.mark.parametrize("expected,explicit", [(0.1, None), (0.7, None), (0.2, 0.08)])
def test_launcher_always_sets_a_finite_session_budget(tmp_path, monkeypatch, expected, explicit):
    binary = tmp_path / "bin"
    binary.mkdir()
    python = binary / "python"
    python.write_text(f"#!{sys.executable}\nimport json\nprint(json.dumps({{'expected_cu':{expected}}}))\n")
    python.chmod(0o700)
    systemctl = binary / "systemctl"
    systemctl.write_text("#!/bin/sh\nprintf 'loaded\\n'\n")
    systemctl.chmod(0o700)
    invoked = tmp_path / "invoked.json"
    systemd = binary / "systemd-run"
    systemd.write_text(
        f"#!{sys.executable}\nimport sys,json,pathlib\n"
        f"pathlib.Path({str(invoked)!r}).write_text(json.dumps(sys.argv))\n"
    )
    systemd.chmod(0o700)
    monkeypatch.setenv("PATH", str(binary) + ":" + os.environ["PATH"])
    monkeypatch.setattr(sys, "executable", str(python))
    budget = ["--budget-cu", str(explicit)] if explicit is not None else []
    monkeypatch.setattr(
        sys, "argv", ["launch", "attempt", "gpu", "--rate", "1", *budget, "--", "--root", "batch"]
    )
    launch.main()
    command = json.loads(invoked.read_text())
    assert float(command[command.index("--budget-cu") + 1]) == pytest.approx(explicit or expected * 1.5)
    assert command[command.index("--budget-source") + 1] == (
        "explicit" if explicit else "expected_cu_times_1.5"
    )


def test_artifact_exclusions_match_incremental_and_final_without_losing_nested_logs(tmp_path):
    paths = (
        "out/e1/train.log",
        "out/result.jsonl",
        "out/e1/mix/train.jsonl",
        "out/mix-long/train.jsonl",
        "out/mix-long-3072/rows.jsonl",
        "out/autotune/cache.bin",
    )
    for name in paths:
        write(tmp_path / name)
    excludes = ["*/mix/*", "*/mix-long*/*", "*/autotune/*"]
    packed = tmp_path / "delta.tar.gz"
    current, count = sync.snapshot(tmp_path, ["out/"], {}, packed, "measurements", excludes)
    assert count == 2 and set(current) == {"out/e1/train.log", "out/result.jsonl"}
    final = tmp_path / "final.tar.gz"
    sync.pack_outputs(tmp_path, ["out/"], final, "measurements", excludes)
    assert read(Path(str(final) + ".manifest.json")) == current
    assert (tmp_path / "out/e1/mix/train.jsonl").exists()


def test_input_paths_preserve_plus_in_experiment_directories(tmp_path):
    source = write(tmp_path / "step+profile/entry.json", b"metrics")
    found = batch.input_paths({"inputs": f"{source} to /content/reference+metadata/", "command": "run"})
    assert found == [(source, "/content/reference+metadata/entry.json")]


def test_artifact_exclusions_ignore_ancestors_above_archive_root(tmp_path):
    tree = tmp_path / "dew-in-parent" / "tree"
    write(tree / "out/train.log", b"retained")
    assert set(sync.output_files(tree, ["out/"], "measurements", ["*dew*"])) == {"out/train.log"}
    with pytest.raises(ValueError, match="archive-relative"):
        sync.output_files(tree, ["out/"], "measurements", [str(tree / "out/*")])


def test_live_prune_between_inventory_and_open_is_skipped_and_relisted(tmp_path, monkeypatch):
    tree = tmp_path / "source"
    disappearing = write(tree / "out/hlo/module.config.pbtxt", b"ephemeral")
    durable = write(tree / "out/result.json", b"measured")
    real = sync.pack_files

    def prune(files, archive):
        disappearing.unlink()
        return real(files, archive)

    monkeypatch.setattr(sync, "pack_files", prune)
    packed = tmp_path / "delta.tar.gz"
    current, count = sync.snapshot(tree, ["out/"], {}, packed)
    assert count == 1 and set(current) == {"out/result.json"}
    sync.apply_snapshot(packed, tmp_path / "retained", current)
    assert (tmp_path / "retained/out/result.json").read_bytes() == durable.read_bytes()
    monkeypatch.setattr(sync, "pack_files", real)
    write(disappearing, b"recreated")
    second, count = sync.snapshot(tree, ["out/"], current, packed)
    assert count == 1 and "out/hlo/module.config.pbtxt" in second


def test_prune_after_open_uses_the_opened_inode_not_its_vanished_path(tmp_path, monkeypatch):
    file = write(tmp_path / "out/result.json", b"complete")
    original = tarfile.TarFile.gettarinfo

    def unlinked(packed, *args, **kwargs):
        file.unlink()
        return original(packed, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "gettarinfo", unlinked)
    archive = tmp_path / "snapshot.tar.gz"
    found = sync.pack_files({"out/result.json": file}, archive)
    assert found == digests({"out/result.json": b"complete"})
    sync.apply_snapshot(archive, tmp_path / "retained", found)
    assert (tmp_path / "retained/out/result.json").read_bytes() == b"complete"


def test_live_truncation_never_writes_a_partial_tar_member(tmp_path, monkeypatch):
    file = write(tmp_path / "out/log", b"original content")
    original = tarfile.TarFile.gettarinfo

    def shortened(packed, *args, **kwargs):
        info = original(packed, *args, **kwargs)
        file.write_bytes(b"short")
        return info

    monkeypatch.setattr(tarfile.TarFile, "gettarinfo", shortened)
    archive = tmp_path / "snapshot.tar.gz"
    assert sync.pack_files({"out/log": file}, archive) == {}
    with tarfile.open(archive) as packed:
        assert packed.getnames() == []


def test_named_training_roots_keep_tracking_but_exclude_weight_stores(tmp_path):
    root = tmp_path / "out/timed/custom+experiment"
    for name in ("run.json", "tracking/scalars.jsonl", "220/weights", "220/commit_success.txt",
                 "220/_CHECKPOINT_METADATA", "221.tmp/weights", "frozen/weights"):
        write(root / name)
    write(tmp_path / "out/train.log")
    found = sync.output_files(tmp_path, ["out/"], "measurements")
    assert set(found) == {"out/timed/custom+experiment/run.json",
                         "out/timed/custom+experiment/tracking/scalars.jsonl", "out/train.log"}
    complete = sync.output_files(tmp_path, ["out/"], "all")
    assert "out/timed/custom+experiment/220/weights" in complete
    assert not any("221.tmp" in name for name in complete)


def test_named_checkpoint_with_a_missing_member_is_not_restored(tmp_path):
    members = {"out/named/run.json": b"config", "out/named/220/weights": b"weight",
               "out/named/220/commit_success.txt": b"ok", "out/named/220/_CHECKPOINT_METADATA": b"metadata"}
    for name, value in members.items():
        write(tmp_path / "outputs" / name, value)
    save(tmp_path / "manifest.json", digests(members))
    (tmp_path / "outputs/out/named/220/weights").unlink()
    packed = tmp_path / "resume.tar.gz"
    sync.pack_retained(tmp_path, packed)
    with tarfile.open(packed) as archive:
        assert archive.getnames() == ["out/named/run.json"]


def test_remote_snapshot_prune_error_is_retryable_not_a_session_failure(tmp_path):
    class Kernel:
        def execute_code(self, *args, **kwargs):
            return [{"output_type": "error", "ename": "FileNotFoundError", "evalue": "pruned"}]

        def stop(self, **kwargs):
            return None

    client = types.SimpleNamespace(kernel=lambda *args, **kwargs: Kernel())
    driver = Driver(tmp_path / "run", client, "unused")
    with pytest.raises(ops_driver.TransportError, match="re-list"):
        driver.control(allocation.Session("owned", "https://fake.invalid/", "token"), "snapshot", live_read=True)
    assert driver.state["action"] == "poll" and driver.state["results"] == []


def test_emergency_out_measurements_survive_a_slow_batch_log(tmp_path, fake_vm, monkeypatch):
    remote, _, session = fake_vm
    write(remote / "content/out/result.json", b'{"exit":0}')
    write(remote / "content/out/analysis/attribution.json", b'{"device_ms":4}')
    write(remote / "content/batch/setup.log", b"slow log")
    original = requests.get

    def slow(url, **kwargs):
        if url.endswith("setup.log"):
            time.sleep(30)
        return original(url, **kwargs)

    monkeypatch.setattr(requests, "get", slow)
    plan = {"driver_sha": "driver", "jobs": [{"index": 1, "outputs": ["/content/out/"]}]}
    began = time.monotonic()
    receipt = metadata_capture.capture(tmp_path / "run", plan, session, "/content/batch", seconds=1)
    assert time.monotonic() - began < 3
    assert {file["remote"] for file in receipt["files"]} == {
        "/content/out/result.json", "/content/out/analysis/attribution.json"}
    assert receipt["tiers"]["logs"]["timed_out"] and not receipt["tiers"]["measurements"]["timed_out"]


def test_emergency_named_runs_exclude_checkpoints_and_capture_tracking_first(tmp_path, fake_vm):
    remote, _, session = fake_vm
    write(remote / "content/out/experiment/run.json", b"{}")
    write(remote / "content/out/experiment/tracking/scalars.jsonl", b'{"step":52}')
    write(remote / "content/out/experiment/220/weights.json", b"weights")
    plan = {"driver_sha": "driver", "jobs": [{"index": 1, "outputs": ["/content/out/"]}]}
    receipt = metadata_capture.capture(tmp_path / "run", plan, session, "/content/batch")
    assert any("tracking/scalars.jsonl" in file["remote"] for file in receipt["files"])
    assert not any("220/" in file["remote"] for file in receipt["files"])
    assert all(file["tier"] == "measurements" for file in receipt["files"])
