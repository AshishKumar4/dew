"""Data-cache publication and restore through an in-memory HTTP boundary."""
import hashlib
import json
from pathlib import Path

import pytest
import requests

from tools.colab.common import read, save
from tools.colab.data_cache import DataCache


@pytest.fixture
def service(tmp_path, monkeypatch):
    token = tmp_path / "private-token"
    token.write_text("secret")
    objects = {}
    calls = []

    class Response:
        def __init__(self, content=b"", status=200):
            self.content = content
            self.status_code = status
            self.headers = {"Content-Length": str(len(content))}

        def raise_for_status(self):
            if self.status_code not in (200, 204):
                raise requests.HTTPError(str(self.status_code))

        def json(self):
            return json.loads(self.content)

        def iter_content(self, size):
            for offset in range(0, len(self.content), size):
                yield self.content[offset:offset + size]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def request(method, url, **kwargs):
        assert kwargs["headers"]["Authorization"] == "Bearer secret"
        key = url.removeprefix("https://fake.invalid/")
        calls.append((method, key))
        if method == "PUT":
            body = kwargs["data"]
            if not isinstance(body, bytes):
                collected = []
                while block := body.read(32):
                    collected.append(block)
                body = b"".join(collected)
            assert int(kwargs["headers"]["Content-Length"]) == len(body)
            objects[key] = body
            return Response(status=204)
        return Response(objects[key]) if key in objects else Response(status=404)

    monkeypatch.setattr(requests, "request", request)
    monkeypatch.setattr(requests, "put", lambda url, **kwargs: request("PUT", url, **kwargs))
    return DataCache("https://fake.invalid", token), objects, calls


def staged(tmp_path):
    first, second = b"hello", b"world"
    chunks = {"first.part": first, "second.part": second}
    for name, content in chunks.items():
        (tmp_path / name).write_bytes(content)
    plan = {"uploads": {"/content/data/rows.jsonl": {"sha256": hashlib.sha256(first + second).hexdigest(),
                        "parts": list(chunks)}}, "staged_files": {
                        name: hashlib.sha256(value).hexdigest() for name, value in chunks.items()}}
    save(tmp_path / "plan.json", plan)
    return plan


def test_publish_and_restore_preserve_verified_bytes_without_rebuilding(tmp_path, service):
    cache, _, calls = service
    staged(tmp_path)
    receipt = cache.publish(tmp_path, "/content/data/", name="mix", output=tmp_path / "receipt.json", rate=0)
    manifest = cache.restore(receipt["manifest_sha256"], tmp_path / "vm")
    assert (tmp_path / "vm/rows.jsonl").read_bytes() == b"helloworld"
    assert receipt["bytes"] == 10 and receipt["files"] == 1 and manifest["name"] == "mix"
    assert "secret" not in (tmp_path / "receipt.json").read_text()
    assert any(method == "HEAD" for method, _ in calls)


def test_second_publication_deduplicates_parts_by_content(tmp_path, service):
    cache, _, calls = service
    staged(tmp_path)
    cache.publish(tmp_path, "/content/data/", name="mix", output=tmp_path / "receipt.json", rate=0)
    before = len([call for call in calls if call[0] == "PUT" and call[1].startswith("parts/")])
    receipt = cache.publish(tmp_path, "/content/data/", name="mix", output=tmp_path / "receipt.json", rate=0)
    assert len([call for call in calls if call[0] == "PUT" and call[1].startswith("parts/")]) == before
    assert set(receipt["parts"].values()) == {"existing"}


def test_changed_staged_part_is_rejected_before_upload(tmp_path, service):
    cache, objects, _ = service
    staged(tmp_path)
    (tmp_path / "first.part").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        cache.publish(tmp_path, "/content/data/", name="mix", output=tmp_path / "receipt.json", rate=0)
    assert objects == {}


@pytest.mark.parametrize("manifest_bad", [False, True])
def test_corruption_never_overwrites_existing_vm_file(tmp_path, service, manifest_bad):
    cache, objects, _ = service
    staged(tmp_path)
    receipt = cache.publish(tmp_path, "/content/data/", name="mix", output=tmp_path / "receipt.json", rate=0)
    key = receipt["manifest_key"] if manifest_bad else next(key for key in objects if key.startswith("parts/"))
    objects[key] = b"corrupted"
    (tmp_path / "vm").mkdir()
    destination = tmp_path / "vm/rows.jsonl"
    destination.write_bytes(b"old")
    with pytest.raises(ValueError, match="checksum|manifest size"):
        cache.restore(receipt["manifest_sha256"], tmp_path / "vm")
    assert destination.read_bytes() == b"old" and list((tmp_path / "vm").glob(".*")) == []


def test_include_selects_index_without_publishing_other_inputs(tmp_path, service):
    cache, _, _ = service
    plan = staged(tmp_path)
    plan["uploads"]["/content/data/private-token"] = plan["uploads"]["/content/data/rows.jsonl"]
    save(tmp_path / "plan.json", plan)
    receipt = cache.publish(tmp_path, "/content/data/", name="index", output=tmp_path / "receipt.json",
                            rate=0, include=("rows.jsonl",))
    manifest = read(tmp_path / "receipt-manifest.json")
    assert receipt["files"] == 1 and manifest["files"][0]["path"] == "rows.jsonl"


def test_manifest_cannot_escape_vm_target(tmp_path, service):
    cache, objects, _ = service
    manifest = json.dumps({"schema": 1, "files": [{"path": "../escape", "parts": [], "sha256": "a" * 64}]}).encode()
    identity = hashlib.sha256(manifest).hexdigest()
    objects["manifests/" + identity] = manifest
    with pytest.raises(ValueError, match="escapes"):
        cache.restore(identity, tmp_path / "vm")
    assert not (tmp_path / "escape").exists()
