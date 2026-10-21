"""The fetched store of recordings over 1 MiB (tools/fetched_fixtures.py): a recording is committed or
pinned by its size, read back only as its pin's bytes, fetched and uploaded only as checked, and
check.sh refuses an oversized recording in git and an incomplete pin."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from tool_support import load


@pytest.fixture
def store(monkeypatch):
    fetched = load("fetched_fixtures")
    monkeypatch.setattr(fetched, "IN_GIT", 4096)
    return fetched


def _recorded(size: int) -> dict[str, np.ndarray]:
    """A recording's arrays, as record_case writes them, about `size` bytes compressed."""
    rng = np.random.default_rng(0)
    return {"site.reference.y": rng.normal(size=size // 8), "site.truth.y": rng.normal(size=size // 8),
            "site.seeds": np.int32(8), "site.redraw": np.int32(4)}


def _flipped(data: bytes) -> bytes:
    """`data` with one bit of its last byte flipped."""
    return data[:-1] + bytes([data[-1] ^ 1])


def _checkout(root: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "tests/fixtures").mkdir(parents=True)
    return root


def _uploaded(store, root: Path) -> dict:
    """The checkout's manifest with every pin given a revision, as `upload` leaves it."""
    pins = store.manifest(root)
    for pin in pins["fixtures"].values():
        pin["revision"] = "a" * 40
    (root / store.MANIFEST).write_text(json.dumps(pins))
    return pins


def test_a_small_recording_is_committed_and_a_large_one_pinned_and_read_as_its_pin(store, tmp_path):
    root = _checkout(tmp_path)
    small = store.save("tests/fixtures/small.npz", _recorded(256), reference="r", license="MIT", root=root)
    assert small == root / "tests/fixtures/small.npz"
    assert store.local("tests/fixtures/small.npz", root) == small
    large = store.save("tests/fixtures/large.npz", _recorded(1 << 16), reference="r", license="MIT",
                       root=root)
    pin = store.manifest(root)["fixtures"]["tests/fixtures/large.npz"]
    assert large == root / store.LOCAL / f"{pin['sha256']}.npz"
    assert (pin["revision"], pin["bytes"]) == (None, large.stat().st_size)
    assert not (root / "tests/fixtures/large.npz").exists()
    with np.load(store.local("tests/fixtures/large.npz", root)) as read:
        np.testing.assert_array_equal(read["site.truth.y"], _recorded(1 << 16)["site.truth.y"])
    large.write_bytes(_flipped(large.read_bytes()))
    with pytest.raises(AssertionError):
        store.local("tests/fixtures/large.npz", root)
    large.unlink()
    with pytest.raises(FileNotFoundError):
        store.local("tests/fixtures/large.npz", root)
    # Recorded again small, it is committed and its pin dropped.
    store.save("tests/fixtures/large.npz", _recorded(256), reference="r", license="MIT", root=root)
    assert store.manifest(root)["fixtures"] == {}


def test_check_refuses_a_large_recording_in_git_and_a_pin_not_uploaded_or_not_whole(store, tmp_path):
    root = _checkout(tmp_path)
    np.savez(root / "tests/fixtures/weights.npz", w=np.zeros(1 << 12))
    np.savez(root / "tests/fixtures/recorded.npz", **_recorded(1 << 16))
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    [refused] = store.check(root)
    assert refused.startswith("tests/fixtures/recorded.npz")
    subprocess.run(["git", "rm", "-q", "tests/fixtures/recorded.npz"], cwd=root, check=True)
    store.save("tests/fixtures/recorded.npz", _recorded(1 << 16), reference="r", license="MIT", root=root)
    assert len(store.check(root)) == 1
    pins = _uploaded(store, root)
    assert store.check(root) == []
    for broken in ({"license": ""}, {"revision": "main"}, {"path": "recordings/other.npz"}, {"bytes": 100}):
        pin = {**pins["fixtures"]["tests/fixtures/recorded.npz"], **broken}
        broken_pins = {**pins, "fixtures": {"tests/fixtures/recorded.npz": pin}}
        (root / store.MANIFEST).write_text(json.dumps(broken_pins))
        assert len(store.check(root)) == 1


def test_fetch_keeps_a_whole_file_and_takes_a_download_only_as_its_pin(store, tmp_path, monkeypatch):
    import huggingface_hub

    root = _checkout(tmp_path)
    stored = store.save("tests/fixtures/large.npz", _recorded(1 << 16), reference="r", license="MIT",
                        root=root)
    _uploaded(store, root)
    hub = tmp_path / "hub.npz"
    hub.write_bytes(stored.read_bytes())
    asked = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, path, **where: asked.append((repo, path, where)) or str(hub))
    store.fetch(root)
    assert asked == []
    stored.unlink()
    store.fetch(root)
    assert stored.read_bytes() == hub.read_bytes()
    assert asked[0][2] == {"repo_type": "dataset", "revision": "a" * 40}
    stored.unlink()
    hub.write_bytes(_flipped(hub.read_bytes()))
    with pytest.raises(AssertionError):
        store.fetch(root)
    assert not stored.exists()


def test_upload_pins_a_revision_only_once_each_file_reads_back_as_its_pin(store, tmp_path, monkeypatch):
    import huggingface_hub

    root = _checkout(tmp_path)
    stored = store.save("tests/fixtures/large.npz", _recorded(1 << 16), reference="r", license="MIT",
                        root=root)
    commits = []

    class Hub:
        def create_repo(self, repo, **options):
            assert options == {"repo_type": "dataset", "exist_ok": True}

        def create_commit(self, repo, operations, **options):
            commits.append({operation.path_in_repo: operation for operation in operations})
            return type("Commit", (), {"oid": "b" * 40})()

    served = tmp_path / "served.npz"
    monkeypatch.setattr(huggingface_hub, "HfApi", Hub)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, path, **where: str(served))
    served.write_bytes(b"not the file")
    with pytest.raises(AssertionError):
        store.upload(root)
    assert store.manifest(root)["fixtures"]["tests/fixtures/large.npz"]["revision"] is None
    served.write_bytes(stored.read_bytes())
    store.upload(root)
    pin = store.manifest(root)["fixtures"]["tests/fixtures/large.npz"]
    assert pin["revision"] == "b" * 40
    assert set(commits[-1]) == {pin["path"], "README.md"}
    assert store.check(root) == []
