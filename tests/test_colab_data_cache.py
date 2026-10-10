"""VM-only data restore against a fake remote builder's published manifest."""

import hashlib
import json

import pytest
import requests

from tools.colab import colab_batch
from tools.colab.common import save
from tools.colab.data_cache import DataCache


@pytest.fixture
def service(tmp_path, monkeypatch):
    token = tmp_path / "private-token"
    token.write_text("secret")
    payload = b"remote-built rows\n"
    manifest = {
        "schema": 1,
        "name": "decision/mix-v3",
        "dew": "build-sha",
        "files": [
            {
                "path": "nested/rows.jsonl",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        ],
    }
    body = json.dumps(manifest).encode()
    objects = {"decision/mix-v3/manifest.json": body, "decision/mix-v3/nested/rows.jsonl": payload}
    calls = []

    class Response:
        def __init__(self, content=b"", status=200):
            self.content, self.status_code = content, status

        def raise_for_status(self):
            if self.status_code != 200:
                raise requests.HTTPError(str(self.status_code))

        def json(self):
            return json.loads(self.content)

        def iter_content(self, size):
            for offset in range(0, len(self.content), size):
                yield self.content[offset : offset + size]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def get(url, **kwargs):
        assert kwargs["headers"]["Authorization"] == "Bearer secret"
        key = url.removeprefix("https://fake.invalid/")
        calls.append(key)
        return Response(objects[key]) if key in objects else Response(status=404)

    def reject(*args, **kwargs):
        raise AssertionError("restore must never publish data")

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "put", reject)
    return DataCache("https://fake.invalid", token), objects, hashlib.sha256(body).hexdigest(), calls


def test_vm_restores_whole_objects_from_a_pinned_manifest(tmp_path, service):
    cache, _, identity, calls = service
    manifest = cache.restore("decision/mix-v3", identity, tmp_path / "vm")
    assert (tmp_path / "vm/nested/rows.jsonl").read_bytes() == b"remote-built rows\n"
    assert manifest["dew"] == "build-sha" and len(calls) == 2


@pytest.mark.parametrize("which", ["manifest.json", "nested/rows.jsonl"])
def test_corruption_never_overwrites_existing_vm_file(tmp_path, service, which):
    cache, objects, identity, _ = service
    objects["decision/mix-v3/" + which] = b"corrupted"
    destination = tmp_path / "vm/nested/rows.jsonl"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old")
    with pytest.raises(ValueError, match="checksum"):
        cache.restore("decision/mix-v3", identity, tmp_path / "vm")
    assert destination.read_bytes() == b"old" and list(destination.parent.glob(".*")) == []


def test_oversized_stream_is_rejected_before_promotion(tmp_path, service):
    cache, objects, identity, _ = service
    objects["decision/mix-v3/nested/rows.jsonl"] *= 3
    with pytest.raises(ValueError, match="exceeds"):
        cache.restore("decision/mix-v3", identity, tmp_path / "vm")
    assert not (tmp_path / "vm/nested/rows.jsonl").exists()


@pytest.mark.parametrize("path", ["../escape", "/escape", ""])
def test_manifest_cannot_escape_vm_target(tmp_path, service, path):
    cache, objects, _, _ = service
    manifest = json.loads(objects["decision/mix-v3/manifest.json"])
    manifest["files"][0]["path"] = path
    body = json.dumps(manifest).encode()
    objects["decision/mix-v3/manifest.json"] = body
    with pytest.raises(ValueError, match="escapes"):
        cache.restore("decision/mix-v3", hashlib.sha256(body).hexdigest(), tmp_path / "vm")


def test_symlink_parent_is_rejected(tmp_path, service):
    cache, _, identity, _ = service
    (tmp_path / "outside").mkdir()
    (tmp_path / "vm").mkdir()
    (tmp_path / "vm/nested").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        cache.restore("decision/mix-v3", identity, tmp_path / "vm")
    assert list((tmp_path / "outside").iterdir()) == []


def test_staging_sends_only_manifest_identity_and_scoped_credential(tmp_path, service):
    _, _, identity, _ = service
    descriptor = tmp_path / "data.json"
    save(
        descriptor,
        {
            "url": "https://fake.invalid",
            "read_token_file": str(tmp_path / "private-token"),
            "sources": [
                {"prefix": "decision/mix-v3", "manifest_sha256": identity, "target": "/content/data"}
            ],
        },
    )
    job = {
        **colab_batch.data_inputs({"data_inputs": str(descriptor)}),
        "index": 1,
        "entry": {"fields": {"command": "run"}},
        "resume_key": "unused",
        "artifact_policy": "measurements",
    }
    plan = {"jobs": [job], "uploads": {}}
    colab_batch.stage_inputs(plan, tmp_path)
    assert list(plan["uploads"]) == ["/content/.dew-data-read.token"]
    assert plan["data_sources"][0]["manifest_sha256"] == identity
    assert "secret" not in json.dumps(plan)


def test_local_inputs_must_not_shadow_r2_data(tmp_path, service):
    _, _, identity, _ = service
    local = tmp_path / "rows.jsonl"
    local.write_bytes(b"must not stage")
    job = {
        "index": 1,
        "data_sources": [{"target": "/content/data", "manifest_sha256": identity}],
        "data_token_file": str(tmp_path / "private-token"),
        "entry": {"fields": {"command": "run", "inputs": f"{local} to /content/data"}},
        "artifact_policy": "measurements",
    }
    with pytest.raises(ValueError, match="must not also be staged"):
        colab_batch.stage_inputs({"jobs": [job], "uploads": {}}, tmp_path)


def test_write_token_only_descriptor_is_refused_before_staging(tmp_path):
    descriptor = tmp_path / "data.json"
    save(descriptor, {"url": "https://fake.invalid", "token_file": str(tmp_path / "write-token"), "sources": []})
    with pytest.raises(ValueError, match="read_token_file"):
        colab_batch.data_inputs({"data_inputs": str(descriptor)})


def test_repeated_restore_reuses_verified_files_without_refetching_data(tmp_path, service):
    cache, _, identity, calls = service
    cache.restore("decision/mix-v3", identity, tmp_path / "vm")
    cache.restore("decision/mix-v3", identity, tmp_path / "vm")
    assert calls.count("decision/mix-v3/nested/rows.jsonl") == 1
    assert calls.count("decision/mix-v3/manifest.json") == 2


def test_duplicate_destination_aliases_are_rejected_before_data_fetch(tmp_path, service):
    cache, objects, _, calls = service
    manifest = json.loads(objects["decision/mix-v3/manifest.json"])
    manifest["files"].append({**manifest["files"][0], "path": "nested//rows.jsonl"})
    body = json.dumps(manifest).encode()
    objects["decision/mix-v3/manifest.json"] = body
    with pytest.raises(ValueError, match="duplicate"):
        cache.restore("decision/mix-v3", hashlib.sha256(body).hexdigest(), tmp_path / "vm")
    assert calls == ["decision/mix-v3/manifest.json"]


def test_packed_jobs_cannot_restore_different_datasets_over_one_target(tmp_path, service):
    _, _, identity, _ = service
    common = {"index": 1, "entry": {"fields": {"command": "run"}}, "resume_key": "none",
              "data_token_file": str(tmp_path / "private-token"), "artifact_policy": "measurements"}
    first = {**common, "data_sources": [{"target": "/content/data", "manifest_sha256": identity}]}
    second = {**common, "data_sources": [{"target": "/content/data/sub", "manifest_sha256": "a" * 64}]}
    with pytest.raises(ValueError, match="overlapping"):
        colab_batch.stage_inputs({"jobs": [first, second], "uploads": {}}, tmp_path)
