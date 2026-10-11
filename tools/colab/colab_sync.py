"""Incremental job output snapshots, retained locally for re-release."""

import errno
import fnmatch
import glob
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

COMMIT_MARKERS = ("commit_success.txt", "_CHECKPOINT_METADATA")


def is_run(path):
    """Run identity comes from its metadata, not its user-selected directory name."""
    return path.name == "run" or (path / "run.json").is_file()


def committed_file(file):
    return any(path.name.isdigit() and all((path / marker).is_file() for marker in COMMIT_MARKERS)
               for path in file.parents)


def retained_path(file, policy):
    for path in file.parents:
        parent = path.parent
        numeric = path.name.split(".", 1)[0].isdigit()
        committed = all((path / marker).is_file() for marker in COMMIT_MARKERS)
        checkpoint = numeric and (is_run(parent) or committed)
        frozen = path.name == "frozen" and is_run(parent)
        if policy == "measurements" and (checkpoint or frozen):
            return False
        if checkpoint and (not path.name.isdigit() or not committed):
            return False
    return True


def missing_only(error):
    if not isinstance(error, (FileNotFoundError, NotADirectoryError)):
        raise error


def walk_files(root, policy):
    if not root.is_dir():
        yield root
        return
    for directory, children, names in os.walk(root, followlinks=False, onerror=missing_only):
        current = Path(directory)
        # Prune checkpoint stores before descending, including named runs.
        children[:] = [name for name in children if retained_path(current / name / "_probe", policy)]
        yield from (current / name for name in names)


def output_files(tree, patterns, policy="all", excludes=()):
    """One inventory for incremental and final collection, including nested logs.

    A measurements job declares OUT once, not a fragile list of log globs.
    Only its checkpoint step/frozen subtrees are excluded; scalars, profiles,
    tracking, run.json, and built data remain ordinary output files.
    Exclusion patterns match archive-relative names: an external
    /content/out/x/a.json is stored as content/out/x/a.json.
    """
    if policy not in ("all", "measurements"):
        raise ValueError(f"unknown artifact policy: {policy}")
    if any(Path(pattern).is_absolute() for pattern in excludes):
        raise ValueError("artifact exclusions must be archive-relative")
    tree = Path(tree)
    files = {}
    for pattern in patterns:
        for matched in glob.glob(str(tree / pattern), recursive=True):
            root = Path(matched)
            scope = root.resolve() if root.is_dir() else root.parent.resolve()
            for file in walk_files(root, policy):
                if not file.is_file():
                    continue
                relative = os.path.relpath(file, tree)
                name = str(file).lstrip("/") if relative.startswith("..") else relative
                if any(fnmatch.fnmatch(name, pattern) for pattern in excludes):
                    continue
                if file.is_symlink() and not file.resolve().is_relative_to(scope):
                    raise ValueError(f"output link escapes declared scope: {file}")
                if not retained_path(file, policy):
                    continue
                files[name] = file
    return files


class HashedReader:
    """Hash the bytes actually written into a tar member, not an earlier read."""

    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        value = self.stream.read(size)
        self.digest.update(value)
        return value


def pack_files(files, archive):
    archive = Path(archive)
    temporary = archive.with_name(archive.name + ".tmp")
    manifest = {}
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=1, dereference=True) as packed:
            for name, file in files.items():
                try:
                    with file.open("rb") as stream:
                        # fstat the opened inode: no second pathname lookup can
                        # race HLO/checkpoint pruning after a successful open.
                        info = packed.gettarinfo(arcname=name, fileobj=stream)
                        if info is None or not info.isfile():
                            raise ValueError(f"artifact is not a regular file: {name}")
                        if committed_file(file):
                            reader = HashedReader(stream)
                            packed.addfile(info, reader)
                            manifest[name] = reader.digest.hexdigest()
                            continue
                        with tempfile.SpooledTemporaryFile(
                            max_size=8 * 1024 * 1024, dir=str(archive.parent)
                        ) as copy:
                            remaining = info.size
                            while remaining:
                                block = stream.read(min(1024 * 1024, remaining))
                                if not block:
                                    break
                                copy.write(block)
                                remaining -= len(block)
                            if remaining:
                                # A shrinking live log is retried on the next
                                # inventory, before any tar header is written.
                                continue
                            copy.seek(0)
                            reader = HashedReader(copy)
                            packed.addfile(info, reader)
                            manifest[name] = reader.digest.hexdigest()
                except (FileNotFoundError, NotADirectoryError):
                    continue
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def pack_outputs(tree, patterns, archive, policy="all", excludes=()):
    """Final packaging uses exactly the same output policy as snapshots."""
    files = output_files(tree, patterns, policy, excludes)
    manifest = pack_files(files, archive)
    sidecar = Path(str(archive) + ".manifest.json")
    temporary = sidecar.with_name(sidecar.name + ".tmp")
    temporary.write_text(json.dumps(manifest))
    os.replace(temporary, sidecar)
    return list(manifest)


def snapshot(tree, patterns, known, archive, policy="all", excludes=()):
    tree = Path(tree)
    files = {}
    for name, file in output_files(tree, patterns, policy, excludes).items():
        try:
            before = file.stat()
            with file.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            after = file.stat()
            if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
                files[name] = (file, digest)
        except FileNotFoundError:
            continue
    changed = {name: value for name, value in files.items() if known.get(name) != value[1]}
    actual = pack_files({name: file for name, (file, _) in changed.items()}, archive)
    for name in changed.keys() - actual.keys():
        del files[name]
    for name, digest in actual.items():
        files[name] = (files[name][0], digest)
    return {name: digest for name, (_, digest) in files.items()}, len(actual)


def apply_snapshot(archive, output, current, known=None, manifest=None):
    """Validate every changed byte before promotion; acknowledge only at the end.

    After an interrupted promotion, the old manifest deliberately remains.
    Resume packaging refuses any file whose bytes do not match that manifest.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = {name for name, digest in current.items() if (known or {}).get(name) != digest}
    with tempfile.TemporaryDirectory(prefix=".sync-stage-", dir=output.parent) as staging:
        staging = Path(staging)
        found = set()
        with tarfile.open(archive) as packed:
            for member in packed:
                name = member.name
                if (
                    not member.isfile()
                    or Path(name).is_absolute()
                    or ".." in Path(name).parts
                    or name not in expected
                    or name in found
                ):
                    raise ValueError(f"unsafe or unexpected artifact member: {name}")
                file = staging / name
                file.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                source = packed.extractfile(member)
                if source is None:
                    raise ValueError(f"artifact has no file content: {name}")
                with source, file.open("wb") as destination:
                    while block := source.read(1024 * 1024):
                        destination.write(block)
                        digest.update(block)
                if digest.hexdigest() != current[name]:
                    raise ValueError(f"artifact checksum mismatch: {name}")
                found.add(name)
        if found != expected:
            raise ValueError("artifact archive is incomplete; manifest not acknowledged")
        output.mkdir(parents=True, exist_ok=True)
        for name in sorted(found):
            target = output / name
            if not target.parent.resolve().is_relative_to(output.resolve()):
                raise ValueError(f"artifact parent escapes output directory: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / name, target)
    if manifest is not None:
        manifest = Path(manifest)
        temporary = manifest.with_name(manifest.name + ".tmp")
        temporary.write_text(json.dumps(current))
        os.replace(temporary, manifest)


def retained_files(cache):
    """Only acknowledged bytes may be sent back to a fresh runtime."""
    cache = Path(cache)
    acknowledged = json.loads((cache / "manifest.json").read_text())
    files = {}
    eligible = output_files(cache / "outputs", ["./"])
    missing_steps = set()
    for filename in acknowledged:
        if filename in eligible:
            continue
        for path in (cache / "outputs" / filename).parents:
            if path.name.isdigit() and is_run(path.parent):
                missing_steps.add(path.relative_to(cache / "outputs"))
    for name, expected in acknowledged.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"unsafe retained artifact: {name}")
        file = cache / "outputs" / name
        # Approved retention can remove old checkpoint members. Missing files
        # are not restored, but changed/unacknowledged bytes are an error.
        if name not in eligible:
            continue
        if any(step in Path(name).parents for step in missing_steps):
            continue
        with file.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"unacknowledged retained artifact: {name}; resync required")
        files[name] = file
    return files


def pack_retained(cache, archive):
    return pack_files(retained_files(cache), archive)


def publish_retained(cache, output):
    """Publish only the acknowledged manifest, excluding stale cached outputs."""
    output = Path(output)
    for name, file in retained_files(cache).items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".publish")
        try:
            try:
                os.link(file, temporary)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                # Retained caches may live on the data disk, while integration
                # receipts live on the home filesystem.
                shutil.copyfile(file, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def restore(archive, tree, patterns):
    """Restore only beneath the entry's declared outputs (relative or absolute)."""
    tree = Path(tree)
    absolute = [Path(p.split("*")[0]).parent if "*" in p else Path(p) for p in patterns if p.startswith("/")]
    relative = [
        tree / (Path(p.split("*")[0]).parent if "*" in p else Path(p))
        for p in patterns
        if not p.startswith("/")
    ]
    with tarfile.open(archive) as packed:
        for member in packed:
            if not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise ValueError(f"unsafe resume member: {member.name}")
            outside = Path("/") / member.name
            target = (
                outside if any(outside == p or p in outside.parents for p in absolute) else tree / member.name
            )
            if not any(target == p or p in target.parents for p in [*absolute, *relative]):
                raise ValueError(f"resume member outside declared outputs: {member.name}")
            if target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
                raise ValueError(f"resume destination uses a symlink: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = packed.extractfile(member)
            if source is None:
                raise ValueError(f"resume member has no file content: {member.name}")
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def retained_key(commit: str, identity: dict) -> str:
    return hashlib.sha256(json.dumps([commit, identity], sort_keys=True).encode()).hexdigest()
