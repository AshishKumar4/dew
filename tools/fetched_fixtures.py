"""Recorded fixtures larger than IN_GIT bytes, kept in a fetched store rather than in git.

A recording of a reference over seeds (tools/recorded_references.py) at most IN_GIT bytes is committed
where its test reads it, under tests/fixtures/; a larger one is written to the local store,
tests/fixtures/fetched/, under its SHA-256, and pinned beside where it would sit, in
`<fixture>.fetched.json`: the store's repo, the revision its file was uploaded in, its path there
(recordings/<sha256>.npz), its sha256 and bytes, and the reference it records with that reference's
license. `fetch` fills the local store before the tests run (armada's environment build,
tools/armada/install.sh, whose key holds every pin, or by hand), and the tests read it offline
(`local`), checking every file against its pin as they read it: CI's tests never reach the Hub
(tools/armada/ci.py's CLOSED_HUB). `upload`, run by whoever holds the store's token, commits every pin
awaiting its revision, with the store's catalog of every file it has held and the card made from it,
reads each file back anonymously at that commit and checks it before the pin takes the revision.
`check` is check.sh's rule: no recording over IN_GIT in git, and every pin whole and uploaded.
A release's own file a test reads offline (a remote-code release's modeling code, its config) is the
store's second kind (`RELEASE`, `pin_release`): kept whole whatever its size, under
releases/<sha256>/<its name> in the store and releases/<sha256>.release in the checkout's local copy
(no Python suffix, so no linter reads a release's code as the checkout's), its pin naming the release
and its license, read and checked by sha256 as a recording is and held to none of the recordings' rules.

Usage, from the checkout:
  python tools/fetched_fixtures.py fetch
  python tools/fetched_fixtures.py check
  python tools/fetched_fixtures.py upload
"""

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_GIT = 1 << 20
"""The largest recording committed to git, in bytes, as Main set it."""
STORE = "dewml/fixtures"
"""The fetched store: a public dataset repo on the Hugging Face Hub."""
FIXTURES = "tests/fixtures"
LOCAL = "tests/fixtures/fetched"
PINNED = ".fetched.json"
CATALOG = "catalog.json"
FIELDS = ("repo", "revision", "path", "sha256", "bytes", "reference", "license")
RELEASE = "release"
"""The `kind` of a pin that holds a release's own file (`pin_release`); a pin without one is a
recording's."""
REDRAWN = ".redraw.npy"
"""The member every recording holds (record_case's `<site>.redraw`, which `indexed_recording` reads),
which marks an NPZ as one."""


def fixture(path, root: Path = REPO_ROOT) -> str | None:
    """`path` as a fixture's name, relative to the checkout and under tests/fixtures/ (a pin's key), or
    None for a file elsewhere, which no pin holds. The name is the path's own, normalized but not
    followed: a symlink at a pinned fixture's place names that fixture, not its target."""
    named, checkout = Path(os.path.abspath(root / path)), Path(os.path.abspath(root))
    if not named.is_relative_to(checkout / FIXTURES):
        return None
    return named.relative_to(checkout).as_posix()


def pins(root: Path = REPO_ROOT) -> dict[str, dict]:
    """Every pin of the checkout by its fixture's name."""
    return {path.relative_to(root).as_posix().removesuffix(PINNED): json.loads(path.read_text())
            for path in sorted((root / FIXTURES).rglob(f"*{PINNED}"))}


def _pin_file(name: str, root: Path) -> Path:
    return root / f"{name}{PINNED}"


def _pinned(name: str, pin: dict, root: Path) -> None:
    _pin_file(name, root).write_text(json.dumps(pin, indent=1, sort_keys=True) + "\n")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stored(pin: dict, root: Path) -> Path:
    if pin.get("kind") == RELEASE:
        # Under its validated sha256 alone, never the pin's path, so no pin reaches outside the store; and
        # under no Python suffix, so a release's code held here is no Python a checkout's linters read.
        return root / LOCAL / "releases" / f"{_sha(pin)}.release"
    return root / LOCAL / f"{pin['sha256']}.npz"


def _sha(pin: dict) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", str(pin["sha256"])):
        raise ValueError(f"{pin['sha256']!r} is no sha256")
    return pin["sha256"]


def _file_name(path: str) -> str:
    """The last part of `path`, a file's own name: none that is empty, '.' or '..'."""
    name = Path(path).name
    if name in ("", ".", ".."):
        raise ValueError(f"{path!r} names no file")
    return name


def _store_path(pin: dict, name: str) -> str:
    """Where the store keeps the file pinned at the fixture `name`: a recording's under its sha256, a
    release file's under its sha256 and the fixture's own name (`pin_release`)."""
    if pin.get("kind") == RELEASE:
        return f"releases/{pin['sha256']}/{_file_name(name)}"
    return f"recordings/{pin['sha256']}.npz"


def pin_release(path, data: bytes, *, reference: str, license: str, root: Path = REPO_ROOT) -> Path:
    """Keep `data`, a release's own file, in the local store whole, pinned at `path` under
    tests/fixtures/ with no revision until `upload`: by its sha256 and its own name, the release it
    is from (`reference`, its repo, commit and file) and that release's `license`. Returns the stored
    file, which `local(path)` reads."""
    name = fixture(path, root)
    # Written where it resolves to: a symlink at the place, or above it, may not carry a write out.
    if name is None or not (root / name).resolve().is_relative_to((root / FIXTURES).resolve()):
        raise ValueError(f"{path}: a release file is pinned under {FIXTURES}/, and resolves there")
    sha = digest(data)
    pin = {"repo": STORE, "revision": None, "sha256": sha, "bytes": len(data), "reference": reference,
           "license": license, "kind": RELEASE}
    pin["path"] = _store_path(pin, name)
    stored = _stored(pin, root)
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_bytes(data)
    target = root / name
    target.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    _pinned(name, pin, root)
    return stored


def _checked(data: bytes, pin: dict, name: str) -> bytes:
    if len(data) != pin["bytes"] or digest(data) != pin["sha256"]:
        raise AssertionError(f"{name}: {len(data)} bytes of sha256 {digest(data)[:12]}, and its pin "
                             f"holds {pin['bytes']} bytes of {pin['sha256'][:12]}; fetch it again")
    return data


def local(path, root: Path = REPO_ROOT) -> Path:
    """The file a test reads for the recording at `path`: where a pin holds it, its file in the local
    store, checked against the pin on every read, whatever sits at `path`; else the file itself."""
    name = fixture(path, root)
    if name is None or not _pin_file(name, root).is_file():
        return root / path
    pin = json.loads(_pin_file(name, root).read_text())
    stored = _stored(pin, root)
    if not stored.is_file():
        raise FileNotFoundError(f"{name}: pinned ({pin['sha256'][:12]}) and not in {LOCAL}; run "
                                f"`python tools/fetched_fixtures.py fetch`")
    _checked(stored.read_bytes(), pin, name)
    return stored


def save(path, arrays: dict, *, reference: str, license: str, root: Path = REPO_ROOT) -> Path:
    """Write a recording compressed: at `path`, under tests/fixtures/, to commit, where it is at most
    IN_GIT bytes; larger, to the local store under its sha256, pinned with no revision until `upload`,
    and any file at `path` removed. Returns the file written."""
    import numpy as np

    name = fixture(path, root)
    # Written where it resolves to: a symlink at the place, or above it, may not carry a write out.
    if name is None or not (root / name).resolve().is_relative_to((root / FIXTURES).resolve()):
        raise ValueError(f"{path}: a recording is saved under {FIXTURES}/, and resolves there")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    data, target = buffer.getvalue(), root / name
    if len(data) <= IN_GIT:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        _pin_file(name, root).unlink(missing_ok=True)
        return target
    sha = digest(data)
    pin = {"repo": STORE, "revision": None, "sha256": sha, "bytes": len(data), "reference": reference,
           "license": license}
    pin["path"] = _store_path(pin, name)
    stored = _stored(pin, root)
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_bytes(data)
    target.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    _pinned(name, pin, root)
    return stored


def fetch(root: Path = REPO_ROOT) -> None:
    """Every pinned file into the local store, each checked against its pin; one already there and
    whole is kept."""
    for name, pin in pins(root).items():
        stored = _stored(pin, root)
        if stored.is_file() and digest(stored.read_bytes()) == pin["sha256"]:
            continue
        if not pin["revision"]:
            raise SystemExit(f"{name}: pinned with no revision and not in {LOCAL}: its upload is due")
        from huggingface_hub import hf_hub_download

        downloaded = Path(hf_hub_download(pin["repo"], pin["path"], repo_type="dataset",
                                          revision=pin["revision"]))
        data = _checked(downloaded.read_bytes(), pin, name)
        stored.parent.mkdir(parents=True, exist_ok=True)
        partial = stored.with_name(f".{stored.name}.partial")
        partial.write_bytes(data)
        partial.replace(stored)
        print(f"fetched {name} ({pin['bytes']} bytes)", flush=True)


def card(catalog: dict[str, dict]) -> str:
    """The store's card: what it holds, each file's fixture, size, reference and that reference's
    license, from its catalog."""
    rows = "\n".join(f"| `{entry['fixture']}` | `{entry.get('path', f'recordings/{key}.npz')}` | "
                     f"{entry['bytes']:,} | "
                     f"{entry['reference']} | {entry['license']} |"
                     for key, entry in sorted(catalog.items(), key=lambda item: item[1]["fixture"]))
    return ("---\nlicense: other\n---\n\n# Dew's recorded test fixtures\n\n"
            "Reference outputs recorded over the seeds Dew's accuracy tests draw "
            "(https://github.com/AshishKumar4/dew), where the reference cannot run in the test. Each file "
            "is pinned by its SHA-256 beside the fixture it stands for in Dew's tests/fixtures/, and is "
            "licensed as the reference it records.\n\n| fixture | file | bytes | reference | license |\n"
            "|---|---|---|---|---|\n" + rows + "\n")


def upload(root: Path = REPO_ROOT) -> None:
    """Every pin with no revision, uploaded in one commit with the store's catalog, which keeps every
    file the store has held, and its card; each file is read back anonymously at that commit, as a
    fetch reads it, and checked before its pin takes the revision."""
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    due = {name: pin for name, pin in pins(root).items() if not pin["revision"]}
    if not due:
        print("every pin holds its revision")
        return
    api = HfApi()
    api.create_repo(STORE, repo_type="dataset", private=False, exist_ok=True)
    if api.repo_info(STORE, repo_type="dataset").private:
        raise SystemExit(f"{STORE} is private, and a fetch reads it anonymously")
    try:
        catalog = json.loads(Path(hf_hub_download(STORE, CATALOG, repo_type="dataset", token=False,
                                                  force_download=True)).read_text())
    except (EntryNotFoundError, RepositoryNotFoundError):
        catalog = {}
    # A recording is listed by its sha256, which names its file; a release file by its path, since the
    # same bytes may be two files of two names (an entry listed before by its sha256 is moved to it).
    catalog = {entry["path"] if entry.get("kind") == RELEASE else key: entry
               for key, entry in catalog.items()}
    catalog.update({pin["path"] if pin.get("kind") == RELEASE else pin["sha256"]: {
        "fixture": name, "bytes": pin["bytes"], "reference": pin["reference"], "license": pin["license"],
        **({"kind": RELEASE, "path": pin["path"]} if pin.get("kind") == RELEASE else {})}
        for name, pin in due.items()})
    operations = [CommitOperationAdd(pin["path"], _checked(_stored(pin, root).read_bytes(), pin, name))
                  for name, pin in sorted(due.items())]
    listing = json.dumps(catalog, indent=1, sort_keys=True) + "\n"
    operations += [CommitOperationAdd(CATALOG, listing.encode()),
                   CommitOperationAdd("README.md", card(catalog).encode())]
    revision = api.create_commit(STORE, operations, commit_message=f"{len(due)} recorded fixtures",
                                 repo_type="dataset").oid
    with tempfile.TemporaryDirectory() as directory:
        for name, pin in sorted(due.items()):
            back = Path(hf_hub_download(pin["repo"], pin["path"], repo_type="dataset", revision=revision,
                                        cache_dir=directory, force_download=True, token=False))
            _checked(back.read_bytes(), pin, name)
            print(f"{pin['sha256']}  {pin['path']} read back at {revision}", flush=True)
    for name, pin in due.items():
        _pinned(name, {**pin, "revision": revision}, root)


def _recording(path: Path) -> bool:
    with zipfile.ZipFile(path) as archive:
        return any(name.endswith(REDRAWN) for name in archive.namelist())


def check(root: Path = REPO_ROOT) -> list[str]:
    """What check.sh refuses: a recording over IN_GIT that git tracks, and a pin with a field missing, a
    sha256, revision or store path not of its form, IN_GIT bytes or fewer, or a fixture git tracks
    too."""
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
    tracked = set(filter(None, listed.stdout.decode().split("\0")))
    problems = [f"{name}: a recording of {(root / name).stat().st_size:,} bytes in git, over "
                f"{IN_GIT:,}; save it to the fetched store (`save`)"
                for name in sorted(tracked) if name.endswith(".npz") and (root / name).is_file()
                and (root / name).stat().st_size > IN_GIT and _recording(root / name)]
    for name, pin in pins(root).items():
        missing = [field for field in FIELDS if pin.get(field) in (None, "")]
        if missing:
            problems.append(f"{name}: its pin holds no {', '.join(missing)}"
                            + (" (its upload is due)" if "revision" in missing else ""))
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]) or pin.get("kind") not in (None, RELEASE) or (
                pin["path"] != _store_path(pin, name)):
            problems.append(f"{name}: its pin's sha256 {pin['sha256']!r} or path {pin['path']!r} is not "
                            f"of the store's form")
        if not re.fullmatch(r"[0-9a-f]{40}", pin["revision"]):
            problems.append(f"{name}: its pin's revision {pin['revision']!r} is not a commit")
        if pin["bytes"] <= IN_GIT and pin.get("kind") != RELEASE:
            problems.append(f"{name}: {pin['bytes']:,} bytes, at most {IN_GIT:,}, belongs in git")
        if name in tracked:
            problems.append(f"{name}: pinned and tracked by git alike")
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("fetch", "check", "upload"))
    command = parser.parse_args(argv).command
    if command == "check":
        problems = check()
        print("\n".join(problems) if problems else "fetched fixtures: every pin whole, no recording over "
              f"{IN_GIT:,} bytes in git")
        return 1 if problems else 0
    (fetch if command == "fetch" else upload)()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
