"""The fetched store of recordings over 1 MiB (tools/fetched_fixtures.py): a recording is committed or
pinned by its size, read back only as its pin's bytes, fetched and uploaded only as checked, and
check.sh refuses an oversized recording in git and an incomplete pin. A file whose license lets it be
re-hosted is kept in GitHub's store, every other in the Hub's; the Hub's tests pin under a license GitHub's
store does not take (HUB_ONLY)."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from tool_support import load

LARGE, SMALL = "tests/fixtures/site/large.npz", "tests/fixtures/site/small.npz"
HUB_ONLY = "lfm1.0"
"""A license GitHub's store does not re-host, so a file under it is kept in the Hub's store."""


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


def _hub(commits: list, private: list[bool]):
    """A stand-in for huggingface_hub's `HfApi` over the store's public dataset, private while
    `private[0]`: each commit's operations appended to `commits` by their paths, every commit "b" * 40."""
    class Hub:
        def create_repo(self, repo, **options):
            assert options == {"repo_type": "dataset", "private": False, "exist_ok": True}

        def repo_info(self, repo, **options):
            return type("Info", (), {"private": private[0]})()

        def create_commit(self, repo, operations, **options):
            commits.append({operation.path_in_repo: operation for operation in operations})
            return type("Commit", (), {"oid": "b" * 40})()

    return Hub


def _checkout(root: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "tests/fixtures").mkdir(parents=True)
    return root


def _saved(store, root: Path, name: str = LARGE, size: int = 1 << 16, license: str = HUB_ONLY) -> Path:
    return store.save(name, _recorded(size), reference="a reference", license=license, root=root)


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

    served = tmp_path / "served.npz"
    catalog = tmp_path / "catalog.json"

    def download(repo, path, **where):
        if path == store.CATALOG:
            if not catalog.exists():
                raise EntryNotFoundError("no catalog")
            return str(catalog)
        assert where["token"] is False and where["revision"] == "b" * 40
        return str(served)

    monkeypatch.setattr(huggingface_hub, "HfApi", _hub(commits, private))
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
    stored = store.pin_release(name, code, reference="org/site@abc/modeling_site.py", license=HUB_ONLY,
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
    """An upload due with a recording and release files commits each under its own store path, catalogs
    a release file by its path with its kind and license (two files of the same bytes and two names
    each, and one the store listed before by its sha256 moved to its path) and the recording by its
    sha256 with its four keys alone, and the card's recording row stays as it was; each reads back by
    its own path before its pin takes the revision. A release pin naming a file other than its
    fixture's is refused by check, and a release file tracked by git beside its pin too."""
    import huggingface_hub

    root = _checkout(tmp_path)
    recording = _saved(store, root)
    name, code = "tests/fixtures/hf/site/release/modeling_site.py", b"class Site:\n    pass\n"
    released = store.pin_release(name, code, reference="org/site@abc/modeling_site.py", license=HUB_ONLY,
                                 root=root)
    # The same bytes as another file of another name: two files, each listed.
    twin = "tests/fixtures/hf/site/release/site_twin.py"
    store.pin_release(twin, code, reference="org/site@abc/site_twin.py", license=HUB_ONLY, root=root)
    # A release file the store listed before by its sha256, which the catalog moves to its path.
    listed = {"fixture": "tests/fixtures/hf/old/release/old.py", "bytes": 3, "license": "MIT",
              "reference": "org/old@abc/old.py", "kind": "release", "path": f"releases/{'1' * 64}/old.py"}
    commits = []

    def download(repo, path, **where):
        if path == store.CATALOG:
            catalog = tmp_path / "catalog.json"
            catalog.write_text(json.dumps({"1" * 64: listed}))
            return str(catalog)
        served = tmp_path / "served" / path
        served.parent.mkdir(parents=True, exist_ok=True)
        served.write_bytes(commits[-1][path].path_or_fileobj)
        return str(served)

    monkeypatch.setattr(huggingface_hub, "HfApi", _hub(commits, [False]))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    store.upload(root)
    pins = store.pins(root)
    release, held = pins[name], pins[LARGE]
    assert release["path"] == f"releases/{release['sha256']}/modeling_site.py"
    assert commits[-1][release["path"]].path_or_fileobj == released.read_bytes() == code
    assert commits[-1][held["path"]].path_or_fileobj == recording.read_bytes()
    catalog = json.loads(commits[-1][store.CATALOG].path_or_fileobj)
    assert catalog[release["path"]] == {
        "fixture": name, "bytes": len(code), "reference": release["reference"], "license": HUB_ONLY,
        "kind": "release", "path": release["path"]}
    assert catalog[pins[twin]["path"]]["fixture"] == twin and pins[twin]["sha256"] == release["sha256"]
    assert set(catalog[held["sha256"]]) == {"fixture", "bytes", "reference", "license"}
    assert catalog[listed["path"]] == listed and len(catalog) == 4
    card = commits[-1]["README.md"].path_or_fileobj.decode()
    assert f"| `{LARGE}` | `recordings/{held['sha256']}.npz` |" in card
    for pinned in (name, twin):
        assert f"| `{pinned}` | `{pins[pinned]['path']}` |" in card
    assert (release["revision"], held["revision"]) == ("b" * 40, "b" * 40) and store.check(root) == []
    # A pin naming another file than its fixture's is refused, as is the fixture tracked beside its pin.
    store._pinned(name, {**release, "path": f"releases/{release['sha256']}/other.py"}, root)
    assert store.check(root)
    store._pinned(name, release, root)
    (root / name).write_bytes(code)
    subprocess.run(["git", "add", name], cwd=root, check=True)
    assert store.check(root)



def test_a_release_file_kept_at_its_origin_is_fetched_from_its_release_and_never_uploaded(store, tmp_path,
                                                                                         monkeypatch):
    """A release file its license keeps from being re-hosted is pinned to its release's own repo at the
    commit its reference names (`pin_release(origin=True)`): the pin holds that commit from the start, so
    check passes with no upload, and upload has nothing due for it; `fetch` downloads it from the
    release's model repo at that commit, as the store's are downloaded, and takes it only as its pin's
    bytes. A pin whose repo or path is not its reference's is refused by check."""
    import huggingface_hub

    root = _checkout(tmp_path)
    name, code = "tests/fixtures/hf/site/release/generation_site.py", b"def sample():\n    pass\n"
    commit = "c" * 40
    stored = store.pin_release(name, code, reference=f"org/site@{commit}/generation_site.py",
                               license="other", origin=True, root=root)
    pin = store.pins(root)[name]
    assert (pin["origin"], pin["repo"], pin["revision"], pin["path"]) == (
        "release", "org/site", commit, "generation_site.py")
    assert store.check(root) == [] and store.local(name, root) == stored
    commits = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _hub(commits, [False]))
    store.upload(root)
    assert commits == []
    hub = tmp_path / "generation_site.py"
    hub.write_bytes(code)
    asked = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, path, **where: asked.append((repo, path, where)) or str(hub))
    stored.unlink()
    store.fetch(root)
    assert stored.read_bytes() == code
    assert asked == [("org/site", "generation_site.py", {"repo_type": "model", "revision": commit})]
    stored.unlink()
    hub.write_bytes(_flipped(code))
    with pytest.raises(AssertionError):
        store.fetch(root)
    store._pinned(name, {**pin, "path": "other.py"}, root)
    assert store.check(root)


def _github(store, monkeypatch, releases: list[dict], private: bool = False, exists: bool = True) -> list:
    """GitHub's store stood in for `gh`, its releases `releases` (each its tag and assets), the repository
    private or missing as told: every `gh` call it is asked appended to the list returned, an upload adding
    the asset with GitHub's digest of its bytes and a create adding an empty release."""
    import hashlib

    asked: list[tuple[str, ...]] = []
    served: dict[str, bytes] = {}

    def gh(*arguments: str) -> str:
        asked.append(arguments)
        if arguments[0] == "api" and not exists:
            raise subprocess.CalledProcessError(1, ["gh", *arguments], stderr="Not Found")
        if arguments == ("api", f"repos/{store.GITHUB}"):
            return json.dumps({"private": private})
        if arguments[:3] == ("api", "--paginate", "--slurp"):
            return json.dumps([releases])
        if arguments[:2] == ("release", "create"):
            releases.append({"tag_name": arguments[2], "assets": []})
            return ""
        if arguments[:2] == ("release", "upload"):
            path = Path(arguments[3])
            [release] = [release for release in releases if release["tag_name"] == arguments[2]]
            release["assets"].append({"name": path.name,
                                      "digest": f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"})
            served[f"{arguments[2]}/{path.name}"] = path.read_bytes()
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(store, "_gh", gh)
    monkeypatch.setattr(store, "_downloaded", lambda url: served["/".join(url.split("/")[-2:])])
    return asked, served


def test_a_rehostable_file_is_kept_in_githubs_store_and_any_other_in_the_hubs(store, tmp_path):
    """A recording or release file under a license GitHub's store re-hosts (MIT, Apache-2.0, in any case) is
    pinned to GitHub's store, its asset named by its sha256 (and a release file's own name); one under any
    other license to the Hub's store as before. Check refuses a GitHub pin under a license it may not hold,
    or whose revision is no release of the store."""
    root = _checkout(tmp_path)
    _saved(store, root, license="MIT")
    _saved(store, root, "tests/fixtures/site/other.npz", license=HUB_ONLY)
    name = "tests/fixtures/hf/site/release/modeling_site.py"
    store.pin_release(name, b"class Site:\n    pass\n", reference="org/site@abc/modeling_site.py",
                      license="Apache-2.0", root=root)
    pins = store.pins(root)
    recording, hub, release = pins[LARGE], pins["tests/fixtures/site/other.npz"], pins[name]
    assert (recording["repo"], recording["origin"], recording["path"]) == (
        store.GITHUB, "github", f"{recording['sha256']}.npz")
    assert release["path"] == f"{release['sha256']}-modeling_site.py" and release["origin"] == "github"
    assert (hub["repo"], hub.get("origin"), hub["path"]) == (
        store.STORE, None, f"recordings/{hub['sha256']}.npz")
    for fixture in (LARGE, name):
        store._pinned(fixture, {**pins[fixture], "revision": "store-1"}, root)
    store._pinned("tests/fixtures/site/other.npz", {**hub, "revision": "a" * 40}, root)
    assert store.check(root) == []
    for broken in ({"license": HUB_ONLY}, {"revision": "a" * 40}, {"path": "other.npz"},
                   {"sha256": "not-a-sha"}):
        store._pinned(LARGE, {**recording, "revision": "store-1", **broken}, root)
        assert store.check(root)
    # A release file of a name GitHub would not keep as an asset is refused as it is pinned.
    with pytest.raises(ValueError):
        store.pin_release("tests/fixtures/hf/site/release/model code.py", b"x", license="MIT", root=root,
                          reference="org/site@abc/model code.py")


def test_upload_puts_each_due_file_as_an_asset_reads_it_back_and_never_overwrites_one(store, tmp_path,
                                                                                       monkeypatch):
    """Upload to GitHub's store opens its first release where none is, uploads each due file as its asset
    without overwriting, reads it back anonymously and only then pins the release; an asset of its name
    already held is taken without an upload only where GitHub's digest is the pin's, and refused otherwise;
    a full release opens the next. A private or missing repository is refused before anything is uploaded,
    and none is ever created."""
    root = _checkout(tmp_path)
    stored = _saved(store, root, license="MIT")
    pin = store.pins(root)[LARGE]
    for private, exists in ((True, True), (False, False)):
        asked, _ = _github(store, monkeypatch, [], private=private, exists=exists)
        with pytest.raises(SystemExit):
            store.upload(root)
        assert not [call for call in asked if call[0] == "release"] and store.pins(root)[LARGE] == pin
    releases: list[dict] = []
    asked, served = _github(store, monkeypatch, releases)
    # A read back that is not the pin's bytes pins nothing; the next upload takes the asset held, uploading
    # none.
    monkeypatch.setattr(store, "_downloaded", lambda url: _flipped(stored.read_bytes()))
    with pytest.raises(AssertionError):
        store.upload(root)
    assert store.pins(root)[LARGE]["revision"] is None
    [upload] = [call for call in asked if call[:2] == ("release", "upload")]
    assert "--clobber" not in upload
    asked, served = _github(store, monkeypatch, releases)
    served[f"store-1/{pin['path']}"] = stored.read_bytes()
    store.upload(root)
    assert store.pins(root)[LARGE]["revision"] == "store-1" and store.check(root) == []
    assert not [call for call in asked if call[:2] == ("release", "upload")]
    [asset] = releases[0]["assets"]
    assert asset["name"] == pin["path"] and served[f"store-1/{pin['path']}"] == stored.read_bytes()
    releases[0]["assets"][0]["digest"] = "sha256:" + "0" * 64
    store._pinned(LARGE, pin, root)
    with pytest.raises(SystemExit):
        store.upload(root)
    assert store.pins(root)[LARGE]["revision"] is None
    # A full release opens the one after the highest, whatever releases are missing between.
    full = [{"tag_name": "store-2", "assets": [{"name": f"{index}.npz", "digest": ""}
                                               for index in range(store.ASSETS_PER_RELEASE)]}]
    _github(store, monkeypatch, full)
    store.upload(root)
    assert store.pins(root)[LARGE]["revision"] == "store-3"
    # A pin whose path is not its asset's name writes nothing anywhere.
    store._pinned(LARGE, {**pin, "path": "../outside.npz"}, root)
    asked, _ = _github(store, monkeypatch, [])
    with pytest.raises(SystemExit):
        store.upload(root)
    assert not [call for call in asked if call[0] == "release"] and not (root.parent / "outside.npz").exists()


def test_fetch_reads_a_githubs_store_file_anonymously_and_takes_it_only_as_its_pin(store, tmp_path,
                                                                                    monkeypatch):
    """A file pinned to GitHub's store is fetched from its release's download URL, read anonymously, and kept
    only where its bytes are its pin's."""
    root = _checkout(tmp_path)
    stored = _saved(store, root, license="MIT")
    data, pin = stored.read_bytes(), store.pins(root)[LARGE]
    store._pinned(LARGE, {**pin, "revision": "store-3"}, root)
    asked = []
    monkeypatch.setattr(store, "_downloaded", lambda url: asked.append(url) or served[0])
    served = [data]
    stored.unlink()
    store.fetch(root)
    assert stored.read_bytes() == data
    assert asked == [f"https://github.com/{store.GITHUB}/releases/download/store-3/{pin['sha256']}.npz"]
    stored.unlink()
    served[0] = _flipped(data)
    with pytest.raises(AssertionError):
        store.fetch(root)
    assert not stored.exists()


def test_move_pins_the_hubs_rehostable_files_to_githubs_store_and_leaves_the_rest(store, tmp_path):
    """`moved` re-pins each file of the Hub's store whose license lets GitHub's store re-host it to GitHub's
    store, its upload due, its sha256 and bytes kept; a file under another license, and a release file kept at
    its own origin, stay as they were."""
    root = _checkout(tmp_path)
    _saved(store, root)
    _saved(store, root, "tests/fixtures/site/moving.npz")
    origin = "tests/fixtures/hf/site/release/generation_site.py"
    store.pin_release(origin, b"def sample():\n    pass\n", license="MIT", origin=True, root=root,
                      reference=f"org/site@{'c' * 40}/generation_site.py")
    _uploaded(store, root)
    store._pinned(origin, {**store.pins(root)[origin], "revision": "c" * 40}, root)
    moving = "tests/fixtures/site/moving.npz"
    store._pinned(moving, {**store.pins(root)[moving], "license": "mit"}, root)
    before = store.pins(root)
    assert store.moved(root) == [moving]
    after = store.pins(root)
    assert after[moving] == {**before[moving], "repo": store.GITHUB, "revision": None, "origin": "github",
                             "path": f"{before[moving]['sha256']}.npz"}
    assert after[LARGE] == before[LARGE] and after[origin] == before[origin]
