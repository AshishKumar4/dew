"""Incremental job output snapshots, retained locally for re-release."""

import glob
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path


def output_files(tree, patterns, policy="all"):
    """One inventory for incremental and final collection, including nested logs.

    A measurements job declares OUT once, not a fragile list of log globs.
    Only its checkpoint step/frozen subtrees are excluded; scalars, profiles,
    tracking, run.json, and built data remain ordinary output files.
    """
    if policy not in ("all", "measurements"):
        raise ValueError(f"unknown artifact policy: {policy}")
    tree = Path(tree)
    files = {}
    for pattern in patterns:
        for matched in glob.glob(str(tree / pattern), recursive=True):
            root = Path(matched)
            scope = root.resolve() if root.is_dir() else root.parent.resolve()
            for file in root.rglob("*") if root.is_dir() else [root]:
                if not file.is_file():
                    continue
                if file.is_symlink() and not file.resolve().is_relative_to(scope):
                    raise ValueError(f"output link escapes declared scope: {file}")
                parts = file.parts
                if policy == "measurements" and any(
                    parts[index] == "run"
                    and (parts[index + 1] == "frozen" or parts[index + 1].split(".", 1)[0].isdigit())
                    for index in range(len(parts) - 1)
                ):
                    continue
                # Orbax's step directory is immutable only after both commit
                # markers exist. Never retain or restore a half-written step.
                checkpoint = None
                for index in range(len(parts) - 1):
                    if parts[index] == "run" and parts[index + 1].split(".", 1)[0].isdigit():
                        checkpoint = Path(*parts[: index + 2])
                        if not parts[index + 1].isdigit():
                            checkpoint = False
                        break
                if checkpoint is False or (
                    checkpoint is not None
                    and not all(
                        (checkpoint / marker).is_file()
                        for marker in ("commit_success.txt", "_CHECKPOINT_METADATA")
                    )
                ):
                    continue
                relative = os.path.relpath(file, tree)
                name = str(file).lstrip("/") if relative.startswith("..") else relative
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
                with file.open("rb") as stream:
                    # Open first, so a concurrent rename never changes the inode
                    # between the size and content reads. Growing logs are a
                    # consistent prefix; committed checkpoints never mutate.
                    info = packed.gettarinfo(str(file), arcname=name)
                    info.size = os.fstat(stream.fileno()).st_size
                    reader = HashedReader(stream)
                    packed.addfile(info, reader)
                    manifest[name] = reader.digest.hexdigest()
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def pack_outputs(tree, patterns, archive, policy="all"):
    """Final packaging uses exactly the same output policy as snapshots."""
    files = output_files(tree, patterns, policy)
    manifest = pack_files(files, archive)
    sidecar = Path(str(archive) + ".manifest.json")
    temporary = sidecar.with_name(sidecar.name + ".tmp")
    temporary.write_text(json.dumps(manifest))
    os.replace(temporary, sidecar)
    return list(files)


def snapshot(tree, patterns, known, archive, policy="all"):
    tree = Path(tree)
    files = {}
    for name, file in output_files(tree, patterns, policy).items():
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
    for name, digest in actual.items():
        files[name] = (files[name][0], digest)
    return {name: digest for name, (_, digest) in files.items()}, len(changed)


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


def pack_retained(cache, archive):
    """Only acknowledged bytes may be sent back to a fresh runtime."""
    cache = Path(cache)
    acknowledged = json.loads((cache / "manifest.json").read_text())
    files = {}
    eligible = output_files(cache / "outputs", ["./"])
    missing_steps = set()
    for filename in acknowledged:
        parts = Path(filename).parts
        for index in range(len(parts) - 1):
            if parts[index] == "run" and parts[index + 1].isdigit() and filename not in eligible:
                missing_steps.add(Path(*parts[: index + 2]))
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
    return pack_files(files, archive)


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


def retained_key(device, ref, lane, outputs):
    return hashlib.sha256(json.dumps([device, ref, lane, outputs]).encode()).hexdigest()
