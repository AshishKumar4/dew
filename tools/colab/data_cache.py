"""Verified VM materialization of remotely built datasets and ordered assemblies."""

import hashlib
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

import requests

SHA = re.compile(r"[0-9a-f]{64}")
# tools/r2_store/worker.js (KEY) owns the object-key grammar; '=' admits Hive partitions.
KEY = re.compile(r"[A-Za-z0-9._=-]+(/[A-Za-z0-9._=-]+)*")


def relative_path(name: str) -> Path:
    path = Path(name)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError("dataset member escapes target")
    if path.as_posix() != name:
        raise ValueError("dataset member path is not canonical")
    return path


def validate_manifest(manifest: dict) -> list[dict]:
    if manifest["schema"] != 1:
        raise ValueError("unsupported dataset manifest")
    normalized = [Path(record["path"]).as_posix() for record in manifest["files"]]
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate dataset destination")
    files = {}
    for record in manifest["files"]:
        name = relative_path(record["path"]).as_posix()
        if not SHA.fullmatch(record["sha256"]) or type(record["bytes"]) is not int or record["bytes"] < 0:
            raise ValueError("invalid dataset file metadata")
        files[name] = record
    assemblies = []
    destinations = set(files)
    for assembly in manifest.get("assemblies", []):
        name = relative_path(assembly["path"]).as_posix()
        shards = assembly["shards"]
        if not shards or any(shard not in files for shard in shards):
            raise ValueError("assembly requires declared physical shards")
        size = sum(files[shard]["bytes"] for shard in shards)
        if (
            not SHA.fullmatch(assembly["sha256"])
            or type(assembly.get("bytes", size)) is not int
            or assembly.get("bytes", size) != size
        ):
            raise ValueError("invalid assembly size or checksum")
        if name in destinations:
            if shards == [name] and assembly["sha256"] == files[name]["sha256"]:
                continue
            raise ValueError("assembly collides with another destination")
        destinations.add(name)
        assemblies.append({**assembly, "bytes": size})
    paths = {Path(name) for name in destinations}
    if any(parent in paths for path in paths for parent in path.parents):
        raise ValueError("dataset file and directory destinations overlap")
    return assemblies


def materialize(record: dict, target: Path, blocks: Iterable[bytes]) -> None:
    """One atomic checksum boundary for downloaded and concatenated files."""
    file = target / relative_path(record["path"])
    if any(parent.is_symlink() for parent in file.parents):
        raise ValueError("dataset destination uses a symlink")
    file.parent.mkdir(parents=True, exist_ok=True)
    if file.is_file() and not file.is_symlink() and file.stat().st_size == record["bytes"]:
        with file.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"]:
                return
    if shutil.disk_usage(file.parent).free < record["bytes"] + 64 * 1024 * 1024:
        raise ValueError("insufficient space for atomic dataset restore")
    descriptor, filename = tempfile.mkstemp(prefix="." + file.name, dir=file.parent)
    temporary = Path(filename)
    digest = hashlib.sha256()
    count = 0
    try:
        with os.fdopen(descriptor, "wb") as destination:
            for block in blocks:
                count += len(block)
                if count > record["bytes"]:
                    raise ValueError("dataset file exceeds manifest size")
                destination.write(block)
                digest.update(block)
            if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError("dataset file checksum mismatch")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, file)
    finally:
        temporary.unlink(missing_ok=True)


def shard_blocks(target: Path, names: list[str]) -> Iterable[bytes]:
    for name in names:
        file = target / relative_path(name)
        if file.is_symlink():
            raise ValueError("assembly shard is a symlink")
        with file.open("rb") as source:
            while block := source.read(1024 * 1024):
                yield block


class DataCache:
    def __init__(self, endpoint: str, token_file: Path):
        url = urlparse(endpoint)
        if (
            url.scheme != "https"
            or not url.netloc
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("data cache requires an HTTPS endpoint without credentials/query/fragment")
        self.endpoint = endpoint.rstrip("/")
        self.headers = {"Authorization": "Bearer " + token_file.read_text().strip()}

    def get(self, key: str, *, stream: bool = False):
        if not KEY.fullmatch(key) or ".." in key.split("/") or len(key) > 512:
            raise ValueError("invalid data-cache key")
        for attempt in range(4):
            try:
                response = requests.get(
                    self.endpoint + "/" + key, headers=self.headers, timeout=(20, 120), stream=stream
                )
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("request retries exhausted")

    def remote_blocks(self, key: str) -> Iterable[bytes]:
        with self.get(key, stream=True) as response:
            yield from response.iter_content(1024 * 1024)

    def restore(self, prefix: str, identity: str, target: Path, *, manifest_key: str | None = None) -> dict:
        if not SHA.fullmatch(identity):
            raise ValueError("invalid manifest identity")
        if target == Path(target.anchor):
            raise ValueError("dataset target cannot be a filesystem root")
        prefix = prefix.strip("/")
        with self.get(manifest_key or prefix + "/manifest.json") as response:
            if hashlib.sha256(response.content).hexdigest() != identity:
                raise ValueError("dataset manifest checksum mismatch")
            manifest = response.json()
        assemblies = validate_manifest(manifest)
        for record in manifest["files"]:
            materialize(record, target, self.remote_blocks(prefix + "/" + record["path"]))
        for assembly in assemblies:
            materialize(assembly, target, shard_blocks(target, assembly["shards"]))
        return manifest
