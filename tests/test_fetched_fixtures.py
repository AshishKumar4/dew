"""The fetched store of recordings over 1 MiB (tools/fetched_fixtures.py): a recording is committed or
pinned by its size, read back only as its pin's bytes, fetched and uploaded only as checked, and
check.sh refuses an oversized recording in git and an incomplete pin."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from tool_support import load

LARGE, SMALL = "tests/fixtures/site/large.npz", "tests/fixtures/site/small.npz"


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


def _saved(store, root: Path, name: str = LARGE, size: int = 1 << 16) -> Path:
    return store.save(name, _recorded(size), reference="a reference", license="MIT", root=root)


def _uploaded(store, root: Path) -> None:
    """Every pin given a revision, as `upload` leaves it."""
    for name, pin in store.pins(root).items():
        store._pinned(name, {**pin, "revision": "a" * 40}, root)


def test_a_small_recording_is_committed_and_a_large_one_pinned_and_read_as_its_pin(store, tmp_path):
    root = _checkout(tmp_path)
    small = _saved(store, root, SMALL, 256)
    assert small == root / SMALL
    assert store.local(SMALL, root) == small
    large = _saved(store, root)
    pin = store.pins(root)[LARGE]
    assert large == root / store.LOCAL / f"{pin['sha256']}.npz"
    assert (pin["revision"], pin["bytes"]) == (None, large.stat().st_size)
    assert not (root / LARGE).exists()
    with np.load(store.local(root / LARGE, root)) as read:
        np.testing.assert_array_equal(read["site.truth.y"], _recorded(1 << 16)["site.truth.y"])
    # A file at a pinned fixture's place is not what the test reads: the pin's is, a symlink's too.
    (root / LARGE).write_bytes(small.read_bytes())
    assert store.local(LARGE, root) == large
    (root / LARGE).unlink()
    (tmp_path / "elsewhere.npz").write_bytes(small.read_bytes())
    (root / LARGE).symlink_to(tmp_path / "elsewhere.npz")
    assert store.local(LARGE, root) == large
    (root / LARGE).unlink()
    large.write_bytes(_flipped(large.read_bytes()))
    with pytest.raises(AssertionError):
        store.local(LARGE, root)
    large.unlink()
    with pytest.raises(FileNotFoundError):
        store.local(LARGE, root)
    # Recorded again small, it is committed and its pin dropped.
    _saved(store, root, LARGE, 256)
    assert store.pins(root) == {}
    # A recording outside tests/fixtures/ is read where it is, and none is saved there.
    elsewhere = tmp_path / "elsewhere.npz"
    assert store.local(elsewhere, root) == elsewhere
    with pytest.raises(ValueError):
        _saved(store, root, "tests/../outside.npz")
    # Nor through a symlink at the place that leads out of the fixtures: the file it leads to is kept.
    (root / SMALL).unlink()
    (root / SMALL).symlink_to(elsewhere)
    kept = elsewhere.read_bytes()
    with pytest.raises(ValueError):
        _saved(store, root, SMALL, 256)
    assert elsewhere.read_bytes() == kept


def test_check_refuses_a_large_recording_in_git_and_a_pin_not_uploaded_or_not_whole(store, tmp_path):
    root = _checkout(tmp_path)
    np.savez(root / "tests/fixtures/weights.npz", w=np.zeros(1 << 12))
    np.savez(root / "tests/fixtures/recorded.npz", **_recorded(1 << 16))
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    [refused] = store.check(root)
    assert refused.startswith("tests/fixtures/recorded.npz")
    subprocess.run(["git", "rm", "-q", "--cached", "tests/fixtures/recorded.npz"], cwd=root, check=True)
    _saved(store, root)
    assert len(store.check(root)) == 1
    _uploaded(store, root)
    assert store.check(root) == []
    pin = store.pins(root)[LARGE]
    for broken in ({"license": ""}, {"revision": "main"}, {"path": "recordings/other.npz"}, {"bytes": 100}):
        store._pinned(LARGE, {**pin, **broken}, root)
        assert len(store.check(root)) == 1
    store._pinned(LARGE, pin, root)
    (root / LARGE).write_bytes(b"tracked")
    subprocess.run(["git", "add", LARGE], cwd=root, check=True)
    assert len(store.check(root)) == 1


def test_fetch_keeps_a_whole_file_and_takes_a_download_only_as_its_pin(store, tmp_path, monkeypatch):
    import huggingface_hub

    root = _checkout(tmp_path)
    stored = _saved(store, root)
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


def test_upload_pins_a_revision_only_once_each_file_reads_back_anonymously_as_its_pin(store, tmp_path,
                                                                                    monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    root = _checkout(tmp_path)
    stored = _saved(store, root)
    commits, private = [], [False]
    kept = {"0" * 64: {"fixture": "tests/fixtures/old.npz", "bytes": 5000, "reference": "an older one",
                       "license": "MIT"}}

    class Hub:
        def create_repo(self, repo, **options):
            assert options == {"repo_type": "dataset", "private": False, "exist_ok": True}

        def repo_info(self, repo, **options):
            return type("Info", (), {"private": private[0]})()

        def create_commit(self, repo, operations, **options):
            commits.append({operation.path_in_repo: operation for operation in operations})
            return type("Commit", (), {"oid": "b" * 40})()

    served = tmp_path / "served.npz"
    catalog = tmp_path / "catalog.json"

    def download(repo, path, **where):
        if path == store.CATALOG:
            if not catalog.exists():
                raise EntryNotFoundError("no catalog")
            return str(catalog)
        assert where["token"] is False and where["revision"] == "b" * 40
        return str(served)

    monkeypatch.setattr(huggingface_hub, "HfApi", Hub)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    private[0] = True
    with pytest.raises(SystemExit):
        store.upload(root)
    private[0] = False
    served.write_bytes(b"not the file")
    with pytest.raises(AssertionError):
        store.upload(root)
    assert store.pins(root)[LARGE]["revision"] is None
    served.write_bytes(stored.read_bytes())
    catalog.write_text(json.dumps(kept))
    store.upload(root)
    pin = store.pins(root)[LARGE]
    assert pin["revision"] == "b" * 40
    assert set(commits[-1]) == {pin["path"], store.CATALOG, "README.md"}
    # The catalog keeps every file the store has held, so the card names each one's source.
    assert set(json.loads(commits[-1][store.CATALOG].path_or_fileobj)) == {"0" * 64, pin["sha256"]}
    assert store.check(root) == []


def test_a_release_file_is_pinned_whole_read_by_its_sha_and_leaves_recordings_as_they_were(store, tmp_path):
    """A release's own file (`pin_release`) is kept whole in the store under its sha256 and its own name,
    whatever its size, pinned with its release and license; read back as its pin's bytes, refused once
    a bit flips, and checked to none of the recordings' rules (a small one is not sent back to git).
    Beside it a recording pins, reads and checks exactly as before (ledger 2173)."""
    root = _checkout(tmp_path)
    recording = _saved(store, root)
    before = dict(store.pins(root)[LARGE])
    name, code = "tests/fixtures/hf/site/release/modeling_site.py", b"class Site:\n    pass\n"
    stored = store.pin_release(name, code, reference="org/site@abc/modeling_site.py", license="apache-2.0",
                               root=root)
    pin = store.pins(root)[name]
    held = f"releases/{pin['sha256']}/modeling_site.py"
    assert (pin["kind"], pin["path"], pin["bytes"]) == ("release", held, len(code))
    assert stored == root / store.LOCAL / "releases" / f"{pin['sha256']}.release"
    assert store.local(name, root) == stored
    assert stored.read_bytes() == code
    assert store.pins(root)[LARGE] == before and store.local(LARGE, root) == recording
    _uploaded(store, root)
    assert store.check(root) == []
    stored.write_bytes(_flipped(code))
    with pytest.raises(AssertionError):
        store.local(name, root)


def test_upload_takes_a_release_file_beside_a_recording_each_by_its_own_path(store, tmp_path, monkeypatch):
    """An upload due with a recording and a release file commits each under its own store path, catalogs
    the release file with its kind, path and license and the recording with its four keys alone, and
    the card's recording row stays as it was; each reads back by its own path before its pin takes
    the revision. A release pin naming a file other than its fixture's is refused by check, and a
    release file tracked by git beside its pin too."""
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    root = _checkout(tmp_path)
    recording = _saved(store, root)
    name, code = "tests/fixtures/hf/site/release/modeling_site.py", b"class Site:\n    pass\n"
    released = store.pin_release(name, code, reference="org/site@abc/modeling_site.py", license="apache-2.0",
                                 root=root)
    commits = []

    class Hub:
        def create_repo(self, repo, **options):
            pass

        def repo_info(self, repo, **options):
            return type("Info", (), {"private": False})()

        def create_commit(self, repo, operations, **options):
            commits.append({operation.path_in_repo: operation for operation in operations})
            return type("Commit", (), {"oid": "b" * 40})()

    def download(repo, path, **where):
        if path == store.CATALOG:
            raise EntryNotFoundError("no catalog")
        served = tmp_path / "served" / path
        served.parent.mkdir(parents=True, exist_ok=True)
        served.write_bytes(commits[-1][path].path_or_fileobj)
        return str(served)

    monkeypatch.setattr(huggingface_hub, "HfApi", Hub)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    store.upload(root)
    pins = store.pins(root)
    release, held = pins[name], pins[LARGE]
    assert release["path"] == f"releases/{release['sha256']}/modeling_site.py"
    assert commits[-1][release["path"]].path_or_fileobj == released.read_bytes() == code
    assert commits[-1][held["path"]].path_or_fileobj == recording.read_bytes()
    catalog = json.loads(commits[-1][store.CATALOG].path_or_fileobj)
    assert catalog[release["sha256"]] == {
        "fixture": name, "bytes": len(code), "reference": release["reference"], "license": "apache-2.0",
        "kind": "release", "path": release["path"]}
    assert set(catalog[held["sha256"]]) == {"fixture", "bytes", "reference", "license"}
    card = commits[-1]["README.md"].path_or_fileobj.decode()
    assert f"| `{LARGE}` | `recordings/{held['sha256']}.npz` |" in card
    assert f"| `{name}` | `{release['path']}` |" in card
    assert (release["revision"], held["revision"]) == ("b" * 40, "b" * 40) and store.check(root) == []
    # A pin naming another file than its fixture's is refused, as is the fixture tracked beside its pin.
    store._pinned(name, {**release, "path": f"releases/{release['sha256']}/other.py"}, root)
    assert store.check(root)
    store._pinned(name, release, root)
    (root / name).write_bytes(code)
    subprocess.run(["git", "add", name], cwd=root, check=True)
    assert store.check(root)

