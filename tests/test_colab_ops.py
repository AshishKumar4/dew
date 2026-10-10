"""Colab contracts exercised against files, child processes and a fake service."""

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
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
    metadata_capture,
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
def test_owned_download_is_atomic_and_consumed_spool_is_reconciled(tmp_path, fake_vm, monkeypatch, large):
    remote, calls, session = fake_vm
    body = b"payload" * 30
    write(remote / "content/result.json", body)
    if large:
        monkeypatch.setattr(transfer, "SMALL_LIMIT", 10)
    target, spool = tmp_path / "target", tmp_path / "spool"
    transfer.download(session, "/content/result.json", target, (spool,))
    assert target.read_bytes() == body
    assert any("/files/" in url for url, _ in calls) == large
    assert all("private" not in url for url, _ in calls)
    registry = next(spool.glob("*.json"))
    value = read(registry)
    value["process_start"] = "dead"
    save(registry, value)
    transfer.reconcile_transfers((spool,))
    assert target.read_bytes() == body
    target.unlink()
    transfer.reconcile_transfers((spool,))
    assert list(spool.iterdir()) == []


def test_dead_partial_cleanup_preserves_unowned_files(tmp_path):
    spool = tmp_path / "spool"
    partial = write(spool / "transfer-owned.part")
    unrelated = write(tmp_path / "unowned")
    target = tmp_path / "result"
    promotion = tmp_path / "result.promote-transfer-owned"
    promotion.symlink_to(partial)
    save(
        partial.with_suffix(".json"),
        {
            "file": str(partial),
            "target": str(target),
            "promotion": str(promotion),
            "pid": os.getpid(),
            "process_start": "different",
        },
    )
    save(spool / "transfer-forged.json", {"file": str(unrelated), "target": str(target)})
    transfer.reconcile_transfers((spool,))
    assert not partial.exists() and not promotion.is_symlink() and unrelated.read_bytes() == b"sample"


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
        transfer.download(session, "/result", target, (tmp_path / "spool",))
    assert target.read_bytes() == b"old" and list((tmp_path / "spool").iterdir()) == []


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


def test_finalizer_reports_failed_release_as_provisional_and_retries(tmp_path, fake_colab):
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
