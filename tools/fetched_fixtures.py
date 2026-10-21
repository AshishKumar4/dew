"""Recorded fixtures larger than IN_GIT bytes, kept in a fetched store rather than in git.

A recording of a reference over seeds (tools/recorded_references.py) at most IN_GIT bytes is committed
where its test reads it; a larger one is written to the local store, tests/fixtures/fetched/, under its
SHA-256, and pinned in tests/fixtures/fetched.json by the store's repo, the revision its file was
uploaded in, its path there (recordings/<sha256>.npz), its sha256 and bytes, and the reference it
records with that reference's license. `fetch` fills the local store before the tests run (armada's
environment build, tools/armada/install.sh, or by hand), and the tests read it offline (`local`),
checking every file against its pin as they read it: CI's tests never reach the Hub (tools/armada/ci.py's
CLOSED_HUB). `upload`, run by whoever holds the store's token, commits every pin awaiting its revision
with the store's card, reads each file back at that revision and checks it before the pin takes it.
`check` is check.sh's rule: no recording over IN_GIT in git, and every pin whole and uploaded.

Usage, from the checkout:
  python tools/fetched_fixtures.py fetch
  python tools/fetched_fixtures.py check
  python tools/fetched_fixtures.py upload
"""

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_GIT = 1 << 20
"""The largest recording committed to git, in bytes (Main, 2026-10-21)."""
STORE = "dewml/fixtures"
"""The fetched store: a public dataset repo on the Hugging Face Hub."""
MANIFEST = "tests/fixtures/fetched.json"
LOCAL = "tests/fixtures/fetched"
FIELDS = ("repo", "revision", "path", "sha256", "bytes", "reference", "license")
REDRAWN = ".redraw.npy"
"""The member every recording holds (record_case's `<site>.redraw`), which marks an NPZ as one."""


def manifest(root: Path = REPO_ROOT) -> dict:
    path = root / MANIFEST
    return json.loads(path.read_text()) if path.is_file() else {"store": STORE, "fixtures": {}}


def _written(pins: dict, root: Path) -> None:
    (root / MANIFEST).write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stored(pin: dict, root: Path) -> Path:
    return root / LOCAL / f"{pin['sha256']}.npz"


def _checked(data: bytes, pin: dict, relative: str) -> bytes:
    if len(data) != pin["bytes"] or digest(data) != pin["sha256"]:
        raise AssertionError(f"{relative}: {len(data)} bytes of sha256 {digest(data)[:12]}, and its pin "
                             f"holds {pin['bytes']} bytes of {pin['sha256'][:12]}; fetch it again")
    return data


def local(relative: str, root: Path = REPO_ROOT) -> Path:
    """The file a test reads for the recording at `relative`: the file itself where git holds it, else
    its pinned file in the local store, checked against its pin on every read."""
    path = root / relative
    if path.is_file():
        return path
    pin = manifest(root)["fixtures"].get(relative)
    if pin is None:
        raise FileNotFoundError(f"{relative}: neither in the checkout nor pinned in {MANIFEST}")
    stored = _stored(pin, root)
    if not stored.is_file():
        raise FileNotFoundError(f"{relative}: pinned ({pin['sha256'][:12]}) and not in {LOCAL}; run "
                                f"`python tools/fetched_fixtures.py fetch`")
    _checked(stored.read_bytes(), pin, relative)
    return stored


def save(relative: str, arrays: dict, *, reference: str, license: str, root: Path = REPO_ROOT) -> Path:
    """Write a recording compressed: at `relative`, to commit, where it is at most IN_GIT bytes; larger,
    to the local store under its sha256, pinned with no revision until `upload`, and any file at
    `relative` removed, so the pin is what the test reads. Returns the file written."""
    import numpy as np

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    data, target, pins = buffer.getvalue(), root / relative, manifest(root)
    if len(data) <= IN_GIT:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if pins["fixtures"].pop(relative, None) is not None:
            _written(pins, root)
        return target
    sha = digest(data)
    pin = {"repo": STORE, "revision": None, "path": f"recordings/{sha}.npz", "sha256": sha,
           "bytes": len(data), "reference": reference, "license": license}
    stored = _stored(pin, root)
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_bytes(data)
    target.unlink(missing_ok=True)
    pins["fixtures"][relative] = pin
    _written(pins, root)
    return stored


def fetch(root: Path = REPO_ROOT) -> None:
    """Every pinned file into the local store, each checked against its pin; one already there and
    whole is kept."""
    for relative, pin in sorted(manifest(root)["fixtures"].items()):
        stored = _stored(pin, root)
        if stored.is_file() and digest(stored.read_bytes()) == pin["sha256"]:
            continue
        if not pin["revision"]:
            raise SystemExit(f"{relative}: pinned with no revision and not in {LOCAL}: its upload is due")
        from huggingface_hub import hf_hub_download

        downloaded = Path(hf_hub_download(pin["repo"], pin["path"], repo_type="dataset",
                                          revision=pin["revision"]))
        data = _checked(downloaded.read_bytes(), pin, relative)
        stored.parent.mkdir(parents=True, exist_ok=True)
        partial = stored.with_name(f".{stored.name}.partial")
        partial.write_bytes(data)
        partial.replace(stored)
        print(f"fetched {relative} ({pin['bytes']} bytes)", flush=True)


def card(pins: dict) -> str:
    """The store's card: what it holds, each file's fixture, size, reference and that reference's
    license."""
    rows = "\n".join(f"| `{relative}` | `{pin['path']}` | {pin['bytes']:,} | {pin['reference']} | "
                     f"{pin['license']} |" for relative, pin in sorted(pins["fixtures"].items()))
    return ("---\nlicense: other\n---\n\n# Dew's recorded test fixtures\n\n"
            "Reference outputs recorded over the seeds Dew's accuracy tests draw "
            "(https://github.com/AshishKumar4/dew), where the reference cannot run in the test. Each file "
            "is pinned by its SHA-256 in Dew's `tests/fixtures/fetched.json` and is licensed as the "
            "reference it records.\n\n| fixture | file | bytes | reference | license |\n"
            "|---|---|---|---|---|\n" + rows + "\n")


def upload(root: Path = REPO_ROOT) -> None:
    """Every pin with no revision, uploaded in one commit with the store's card; each file is read back
    at that commit and checked before its pin takes the revision."""
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    pins = manifest(root)
    due = {relative: pin for relative, pin in pins["fixtures"].items() if not pin["revision"]}
    if not due:
        print("every pin holds its revision")
        return
    operations = [CommitOperationAdd(pin["path"], _checked(_stored(pin, root).read_bytes(), pin, relative))
                  for relative, pin in sorted(due.items())]
    operations.append(CommitOperationAdd("README.md", card(pins).encode()))
    api = HfApi()
    api.create_repo(STORE, repo_type="dataset", exist_ok=True)
    revision = api.create_commit(STORE, operations, commit_message=f"{len(due)} recorded fixtures",
                                 repo_type="dataset").oid
    with tempfile.TemporaryDirectory() as directory:
        for relative, pin in sorted(due.items()):
            back = Path(hf_hub_download(pin["repo"], pin["path"], repo_type="dataset", revision=revision,
                                        cache_dir=directory, force_download=True))
            _checked(back.read_bytes(), pin, relative)
            print(f"{pin['sha256']}  {pin['path']} read back at {revision}", flush=True)
    for pin in due.values():
        pin["revision"] = revision
    _written(pins, root)


def _recording(path: Path) -> bool:
    with zipfile.ZipFile(path) as archive:
        return any(name.endswith(REDRAWN) for name in archive.namelist())


def check(root: Path = REPO_ROOT) -> list[str]:
    """What check.sh refuses: a recording over IN_GIT that git tracks, and a pin with a field missing, a
    sha256, revision or store path not of its form, IN_GIT bytes or fewer, or a fixture git tracks too."""
    listed = subprocess.run(["git", "ls-files", "-z", "--", "*.npz"], cwd=root, capture_output=True,
                            check=True)
    tracked = set(filter(None, listed.stdout.decode().split("\0")))
    problems = [f"{relative}: a recording of {(root / relative).stat().st_size:,} bytes in git, over "
                f"{IN_GIT:,}; save it to the fetched store (`save`)"
                for relative in sorted(tracked) if (root / relative).is_file()
                and (root / relative).stat().st_size > IN_GIT and _recording(root / relative)]
    for relative, pin in sorted(manifest(root)["fixtures"].items()):
        missing = [name for name in FIELDS if pin.get(name) in (None, "")]
        if missing:
            problems.append(f"{relative}: its pin holds no {', '.join(missing)}"
                            + (" (its upload is due)" if "revision" in missing else ""))
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]) or (
                pin["path"] != f"recordings/{pin['sha256']}.npz"):
            problems.append(f"{relative}: its pin's sha256 {pin['sha256']!r} or path {pin['path']!r} is not "
                            f"of the store's form")
        if not re.fullmatch(r"[0-9a-f]{40}", pin["revision"]):
            problems.append(f"{relative}: its pin's revision {pin['revision']!r} is not a commit")
        if pin["bytes"] <= IN_GIT:
            problems.append(f"{relative}: {pin['bytes']:,} bytes, at most {IN_GIT:,}, belongs in git")
        if relative in tracked:
            problems.append(f"{relative}: pinned and tracked by git alike")
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
