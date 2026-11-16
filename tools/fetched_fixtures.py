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
A release file whose license keeps it from being re-hosted stays in its release's own repo: its pin's
`origin` is the release's (`ORIGINS`), its repo, commit and path the file's there, which `fetch`
reads it from as it reads the store's, and which `upload` leaves alone.

Usage, from the checkout:
  python tools/fetched_fixtures.py fetch
  python tools/fetched_fixtures.py check
  python tools/fetched_fixtures.py upload
  python tools/fetched_fixtures.py move     # the Hub store's re-hostable pins to GitHub's, their upload due
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
from typing import NotRequired, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_GIT = 1 << 20
"""The largest recording committed to git, in bytes, as the maintainer set it."""
STORE = "dewml/fixtures"
"""The fetched store of a file no license lets GitHub's store re-host (`REHOSTED`): a public dataset repo
on the Hugging Face Hub."""
GITHUB = "AshishKumar4/dew-fixtures"
"""The store on GitHub: a public repository whose releases (`GITHUB_RELEASE`) hold each file whose license
lets it be re-hosted (`REHOSTED`), an asset named by its content (`_asset`), never overwritten, read
anonymously over HTTPS (owner, 2026-11-16: GitHub rather than R2)."""
GITHUB_RELEASE = re.compile(r"store-[1-9][0-9]*")
"""A release of GitHub's store, `store-N`; a new one opens as the last holds `ASSETS_PER_RELEASE`."""
ASSETS_PER_RELEASE = 1000
"""The most assets GitHub keeps in one release."""
REHOSTED = frozenset({"apache-2.0", "mit", "mit and apache-2.0", "the owner's own code"})
"""The licenses, as a pin names them in any case, under which GitHub's store re-hosts a file; a file under any
other stays in the Hub's store or at its release's own origin (`ORIGINS`)."""
FIXTURES = "tests/fixtures"
LOCAL = "tests/fixtures/fetched"
PINNED = ".fetched.json"
CATALOG = "catalog.json"


class Pin(TypedDict):
    """A pinned file's record (`PINNED`): the repo and revision it is fetched from and its path there,
    its sha256 and size, the reference it records or is (its repo, commit and file) and that reference's
    license; a release's own file's `kind` (`RELEASE`), and where it is kept at its release (`origin`)."""
    repo: str
    revision: str | None
    path: str
    sha256: str
    bytes: int
    reference: str
    license: str
    kind: NotRequired[str]
    origin: NotRequired[str]


FIELDS = tuple(field for field in Pin.__annotations__ if field in Pin.__required_keys__)
"""The fields every pin holds (`check`), in `Pin`'s order."""
RELEASE = "release"
"""The `kind` of a pin that holds a release's own file (`pin_release`); a pin without one is a
recording's."""
ORIGINS = {"store": "dataset", "release": "model", "github": None}
"""Where a pinned file is fetched from, by its pin's `origin` (the Hub's store where none is named), and that
repo's type on the Hub: the Hub's store (`STORE`), a release's own repo at its commit, for a release file its
license keeps from being re-hosted (`pin_release(origin=)`), or GitHub's store (`GITHUB`), no Hub repo, where
`upload` puts every file whose license lets it (`REHOSTED`)."""
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


def manifest(payload: Path) -> Path:
    """The pin of the fixture whose payload is `payload`: the file beside it its name and `PINNED` name, which
    the checkout carries in the payload's place (and a run's key reads the payload by: digests.contents)."""
    return payload.with_name(payload.name + PINNED)


def _pinned(name: str, pin: dict, root: Path) -> None:
    manifest(root / name).write_text(json.dumps(pin, indent=1, sort_keys=True) + "\n")


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


def _pin(data: bytes, name: str, *, reference: str, license: str, kind: str | None = None,
         origin: bool = False) -> Pin:
    """The pin of `data` kept in the store for the fixture `name`, before its upload gives a revision: its
    sha256 and size, `reference` and `license`, and a release file's `kind`, stored in GitHub's store where
    its license lets it be re-hosted (`_asset`) and it is not kept at its release's `origin`, else in the
    Hub's where `_store_path` puts it."""
    sha = digest(data)
    held = {"sha256": sha, "kind": kind}
    pin: Pin = {"repo": STORE, "revision": None, "sha256": sha, "bytes": len(data), "reference": reference,
                "license": license, "path": _store_path(held, name)}
    if rehosted(license) and not origin:
        pin.update(repo=GITHUB, path=_asset(held, name), origin="github")
    if kind is not None:
        pin["kind"] = kind
    return pin


def rehosted(license: str) -> bool:
    """Whether GitHub's store may re-host a file under `license` (`REHOSTED`)."""
    return license.strip().lower() in REHOSTED


def _asset(pin: dict, name: str) -> str:
    """The asset of GitHub's store holding the file pinned at the fixture `name`: named by its sha256, and a
    release file by its own name after it, so no name ever holds two files' bytes."""
    if pin.get("kind") == RELEASE:
        # GitHub renames an asset with other characters, and a URL reads a space, `#` or `?` as its own.
        if not re.fullmatch(r"[A-Za-z0-9._-]*[A-Za-z0-9_-]", file := _file_name(name)):
            raise ValueError(f"{name}: {file!r} is no name GitHub keeps for an asset ([A-Za-z0-9._-], "
                             "no trailing period)")
        return f"{_sha(pin)}-{file}"
    return f"{_sha(pin)}.npz"


def _url(repo: str, release: str, asset: str) -> str:
    return f"https://github.com/{repo}/releases/download/{release}/{asset}"


def downloaded(url: str) -> bytes:
    """`url` read anonymously over HTTPS, as every machine a test runs on reads it."""
    import urllib.request

    with urllib.request.urlopen(url, timeout=600) as response:
        return response.read()


def _store_path(pin: dict, name: str) -> str:
    """Where the store keeps the file pinned at the fixture `name`: a recording's under its sha256, a
    release file's under its sha256 and the fixture's own name (`pin_release`)."""
    if pin.get("kind") == RELEASE:
        return f"releases/{pin['sha256']}/{_file_name(name)}"
    return f"recordings/{pin['sha256']}.npz"


def pin_release(path, data: bytes, *, reference: str, license: str, origin: bool = False,
                root: Path = REPO_ROOT) -> Path:
    """Keep `data`, a release's own file, in the local store whole, pinned at `path` under
    tests/fixtures/ with no revision until `upload`: by its sha256 and its own name, the release it
    is from (`reference`, its repo, commit and file) and that release's `license`; with `origin`,
    pinned instead to the release's own repo at that commit (`ORIGINS`), which `fetch` reads it from
    and `upload` never re-hosts. Returns the stored file, which `local(path)` reads."""
    name = fixture(path, root)
    # Written where it resolves to: a symlink at the place, or above it, may not carry a write out.
    if name is None or not (root / name).resolve().is_relative_to((root / FIXTURES).resolve()):
        raise ValueError(f"{path}: a release file is pinned under {FIXTURES}/, and resolves there")
    pin = _pin(data, name, reference=reference, license=license, kind=RELEASE, origin=origin)
    if origin:
        repo, revision, file = _released(reference)
        pin.update(repo=repo, revision=revision, path=file, origin="release")
    stored = _stored(pin, root)
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_bytes(data)
    target = root / name
    target.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    _pinned(name, pin, root)
    return stored


def _released(reference: str) -> tuple[str, str, str]:
    """A release file's `reference`, `<repo>@<commit>/<path>`, as its repo, commit and path."""
    found = re.fullmatch(r"([\w.-]+/[\w.-]+)@([0-9a-f]{40})/(.+)", reference)
    if found is None:
        raise ValueError(f"{reference!r} names no file of a repo at a commit, <repo>@<commit>/<path>")
    return found.group(1), found.group(2), found.group(3)


def _checked(data: bytes, pin: dict, name: str) -> bytes:
    if len(data) != pin["bytes"] or digest(data) != pin["sha256"]:
        raise AssertionError(f"{name}: {len(data)} bytes of sha256 {digest(data)[:12]}, and its pin "
                             f"holds {pin['bytes']} bytes of {pin['sha256'][:12]}; fetch it again")
    return data


def local(path, root: Path = REPO_ROOT) -> Path:
    """The file a test reads for the recording at `path`: where a pin holds it, its file in the local
    store, checked against the pin on every read, whatever sits at `path`; else the file itself."""
    name = fixture(path, root)
    if name is None or not manifest(root / name).is_file():
        return root / path
    pin = json.loads(manifest(root / name).read_text())
    stored = _stored(pin, root)
    if not stored.is_file():
        raise FileNotFoundError(f"{name}: pinned ({pin['sha256'][:12]}) and not in {LOCAL}; run "
                                f"`python tools/fetched_fixtures.py fetch`")
    _checked(stored.read_bytes(), pin, name)
    return stored


def provenance(reference: str, revision: str, device: str, command: str, **details):
    """A recording's reference, its revision, the device and the public command that reproduces it,
    as a JSON string array to save beside the recorded arrays (`save`). The command is published with the
    recording: a path of the recording machine's own (from the root or the home, but a directory under
    /tmp the command makes itself) is refused, the argument's name standing in its place."""
    import numpy as np

    if any(part.startswith(("/", "~")) and not part.startswith("/tmp/")
           for word in command.split() for part in word.split("=")[-1].split(":")):
        raise ValueError(f"a recording's command is public, its machine's paths named, not given: {command}")
    return np.asarray(json.dumps({"reference": reference, "revision": revision, "device": device,
                                  "command": command, **details}))


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
        manifest(root / name).unlink(missing_ok=True)
        return target
    pin = _pin(data, name, reference=reference, license=license)
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
        if pin.get("origin") == "github":
            data = _checked(downloaded(_url(pin["repo"], pin["revision"], pin["path"])), pin, name)
        else:
            from huggingface_hub import hf_hub_download

            kind = ORIGINS[pin.get("origin", "store")]
            cached = Path(hf_hub_download(pin["repo"], pin["path"], repo_type=kind, revision=pin["revision"]))
            data = _checked(cached.read_bytes(), pin, name)
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


def github(*arguments: str) -> str:
    """The GitHub CLI's answer to `arguments`, through the credential it holds: the store's one way to GitHub."""
    return subprocess.run(["gh", *arguments], capture_output=True, text=True, check=True).stdout


def upload_github(due: dict[str, dict], root: Path = REPO_ROOT) -> None:
    """Every pin of `due` (GitHub's store's, with no revision) uploaded as its asset (`_asset`) to the open
    release of GitHub's store, `store-N`, the next opened once it holds `ASSETS_PER_RELEASE`; an asset of its
    name already held is taken only where GitHub's digest of it is the pin's sha256, and none is ever
    overwritten (`gh release upload` without --clobber). Each asset is read back anonymously, as a fetch reads
    it, and checked before its pin takes its release. The repository is the owner's to create, public: none is
    created here."""
    try:
        if json.loads(github("api", f"repos/{GITHUB}"))["private"]:
            raise SystemExit(f"{GITHUB} is private, and a fetch reads it anonymously")
        pages = json.loads(github("api", "--paginate", "--slurp", f"repos/{GITHUB}/releases"))
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"{GITHUB} could not be read ({error.stderr.strip()[-200:]}): the owner creates it, "
                         "public, before its first upload") from error
    releases = sorted((release for page in pages for release in page
                       if GITHUB_RELEASE.fullmatch(release["tag_name"])),
                      key=lambda release: int(release["tag_name"].removeprefix("store-")))
    held = {asset["name"]: (release["tag_name"], asset.get("digest") or "")
            for release in releases for asset in release["assets"]}
    counts = {release["tag_name"]: len(release["assets"]) for release in releases}
    with tempfile.TemporaryDirectory() as directory:
        for name, pin in sorted(due.items()):
            # The asset's name is the pin's content's, never a path the pin carries (review).
            asset = _asset(pin, name)
            if pin["path"] != asset:
                raise SystemExit(f"{name}: its pin names {pin['path']!r}, not its asset {asset!r}")
            if asset in held:
                release, said = held[asset]
                if said != f"sha256:{pin['sha256']}":
                    raise SystemExit(f"{name}: {asset} in {GITHUB}'s {release} is not its pin's bytes "
                                     f"({said})")
            else:
                release = next((tag for tag, count in counts.items() if count < ASSETS_PER_RELEASE), None)
                if release is None:
                    # After the highest, whatever releases were removed between (review).
                    highest = max((int(tag.removeprefix("store-")) for tag in counts), default=0)
                    release = f"store-{highest + 1}"
                    github("release", "create", release, "--repo", GITHUB, "--title", release, "--notes",
                        "Dew's recorded test fixtures (https://github.com/AshishKumar4/dew): each asset is "
                        "named by its SHA-256, pinned beside the fixture it stands for in Dew's "
                        "tests/fixtures/, and licensed as the reference it records, as its pin names.")
                    counts[release] = 0
                written = Path(directory) / asset
                written.write_bytes(_checked(_stored(pin, root).read_bytes(), pin, name))
                github("release", "upload", release, str(written), "--repo", GITHUB)
                written.unlink()
                counts[release] += 1
                held[asset] = (release, f"sha256:{pin['sha256']}")
            _checked(downloaded(_url(GITHUB, release, asset)), pin, name)
            print(f"{pin['sha256']}  {asset} read back at {GITHUB}'s {release}", flush=True)
            _pinned(name, {**pin, "revision": release}, root)


def moved(root: Path = REPO_ROOT) -> list[str]:
    """Every pin of the Hub's store whose license lets GitHub's store re-host it (`REHOSTED`), pinned to
    GitHub's store with no revision, its upload due there (`upload`); a release file kept at its origin, and
    every file under another license, stays as it is. Returns the fixtures moved."""
    moving = {name: pin for name, pin in pins(root).items()
              if pin.get("origin", "store") == "store" and pin["repo"] == STORE and rehosted(pin["license"])}
    for name, pin in moving.items():
        moving_to = {"repo": GITHUB, "revision": None, "path": _asset(pin, name), "origin": "github"}
        _pinned(name, {**pin, **moving_to}, root)
    return sorted(moving)


def upload(root: Path = REPO_ROOT) -> None:
    """Every pin with no revision uploaded: GitHub's store's as its assets (`upload_github`), the Hub's in one
    commit with the store's catalog, which keeps every file the store has held, and its card; each file is
    read back anonymously, as a fetch reads it, and checked before its pin takes the revision."""
    due = {name: pin for name, pin in pins(root).items() if not pin["revision"]}
    if not due:
        print("every pin holds its revision")
        return
    if github := {name: pin for name, pin in due.items() if pin.get("origin") == "github"}:
        upload_github(github, root)
    due = {name: pin for name, pin in due.items() if pin.get("origin") != "github"}
    if not due:
        return
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
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
        released, github = pin.get("origin") == "release", pin.get("origin") == "github"
        # A file kept at its release's origin is the release's own file there, its reference's; one of
        # GitHub's store its asset, of a sha256 and a name GitHub keeps (`_asset`).
        try:
            where = ((pin["repo"], pin["revision"], pin["path"]) == _released(pin["reference"]) if released
                     else (pin["repo"], pin["path"]) == (GITHUB, _asset(pin, name)) if github
                     else pin["path"] == _store_path(pin, name) and pin["repo"] == STORE)
        except ValueError:
            where = False
        if (not re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]) or pin.get("kind") not in (None, RELEASE)
                or pin.get("origin", "store") not in ORIGINS or (released and pin.get("kind") != RELEASE)
                or not where):
            problems.append(f"{name}: its pin's sha256 {pin['sha256']!r} or path {pin['path']!r} is not "
                            f"of its origin's form")
        if github and not rehosted(pin["license"]):
            problems.append(f"{name}: its license, {pin['license']!r}, keeps it from GitHub's store")
        form = GITHUB_RELEASE.pattern if github else r"[0-9a-f]{40}"
        if not re.fullmatch(form, pin["revision"]):
            problems.append(f"{name}: its pin's revision {pin['revision']!r} is not "
                            f"{'a release of the store' if github else 'a commit'}")
        if pin["bytes"] <= IN_GIT and pin.get("kind") != RELEASE:
            problems.append(f"{name}: {pin['bytes']:,} bytes, at most {IN_GIT:,}, belongs in git")
        if name in tracked:
            problems.append(f"{name}: pinned and tracked by git alike")
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("fetch", "check", "upload", "move"))
    command = parser.parse_args(argv).command
    if command == "move":
        moving = moved()
        print(f"{len(moving)} pins moved to {GITHUB}, their upload due: `upload` once it exists")
        return 0
    if command == "check":
        problems = check()
        print("\n".join(problems) if problems else "fetched fixtures: every pin whole, no recording over "
              f"{IN_GIT:,} bytes in git")
        return 1 if problems else 0
    (fetch if command == "fetch" else upload)()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
