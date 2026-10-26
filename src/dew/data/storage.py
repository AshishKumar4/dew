"""URL storage, named credentials, bounded transport caching and conditional puts."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import time
import tomllib
import urllib.parse
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import BinaryIO, NamedTuple, Protocol, TypedDict, runtime_checkable

import fsspec
from filelock import FileLock, Timeout
from fsspec.implementations.cached import CachingFileSystem, WholeFileCacheFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from gcsfs import GCSFileSystem
from huggingface_hub import HfFileSystem, parse_hf_uri, try_to_load_from_cache
from huggingface_hub.file_download import REGEX_COMMIT_HASH
from huggingface_hub.hf_file_system import HfFileSystemResolvedRepositoryPath
from s3fs import S3FileSystem

from dew._records import JSON, json_value
from dew.cache import dew_cache_dir
from dew.data.retention import LocalRetention, evictable, read_pin
from dew.files import write_atomically


class PreconditionFailed(FileExistsError):
    """The object changed since the caller read its version."""


@runtime_checkable
class ConditionalWrites(Protocol):
    """An fsspec filesystem that supplies its own conditional writes (`Store`)."""

    def put_conditional(self, path: str, payload: bytes, version: str | None) -> str: ...


class ObjectInfo(TypedDict):
    bytes: int
    sha256: str


class ObjectRef(ObjectInfo):
    path: str


def object_info(payload: bytes | io.BufferedReader) -> ObjectInfo:
    """The size and SHA-256 of `payload`, bytes, or a file read from its start in bounded chunks."""
    if isinstance(payload, bytes):
        return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    payload.seek(0)
    digest = hashlib.file_digest(payload, "sha256")
    size = payload.tell()
    payload.seek(0)
    return {"bytes": size, "sha256": digest.hexdigest()}


def matching(found: ObjectInfo, expected: Mapping[str, object], url: str) -> None:
    """Refuse an object at `url` whose size and digest, `found`, are not what was committed for it."""
    if found["bytes"] != expected["bytes"] or found["sha256"] != expected["sha256"]:
        raise OSError(f"committed object is corrupt: {url}; bytes={found['bytes']}, "
                      f"sha256={found['sha256']}, expected bytes={expected['bytes']}, "
                      f"sha256={expected['sha256']}")


def commit_export(
    root: Store, *, format: str, codec: int | None, rows: int, files: list[Mapping[str, object]],
    metadata: Mapping[str, object] | None = None, certified: Mapping[str, JSON] | None = None,
) -> Mapping[str, object]:
    """Commit the manifest of an export at `root`, its `files` already written there: its schema,
    format, record codec, rows, files and metadata, and what its producer `certified` of its rows (an
    export its count rules and partitioning; a directory of existing ArrayRecord shards nothing);
    then the success marker its digest names. An export (`dew.data.sinks.export`) and existing
    shards provisioned in place (`Files.provision`) both commit through it."""
    manifest = {"schema": 1, "format": format, "codec": codec, "rows": rows, **(certified or {}),
                "files": files, "metadata": dict(metadata or {})}
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    root.child("_dew_manifest.json").put_if_absent(encoded)
    root.child("_SUCCESS").put_if_absent(hashlib.sha256(encoded).hexdigest().encode())
    return manifest


class TransportStats(TypedDict):
    hits: int
    misses: int
    downloaded_bytes: int
    wait_seconds: float


def _byte_size(value: int | str) -> int:
    if isinstance(value, int):
        return value
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?i?B)?\s*", value, re.IGNORECASE)
    if match is None:
        raise ValueError(f"byte budget {value!r} needs an integer or a unit such as MiB or GB")
    unit = (match[2] or "B").upper()
    base = 1024 if "I" in unit else 1000
    exponent = 0 if unit == "B" else "KMGT".index(unit[0]) + 1
    return int(float(match[1]) * base**exponent)


def _secret(value: object, profile: str) -> JSON:
    if isinstance(value, str) and value.startswith("env:"):
        name = value[4:]
        if name not in os.environ:
            raise ValueError(f"storage profile {profile!r} needs environment variable {name}")
        return os.environ[name]
    if isinstance(value, str) and value.startswith("file:"):
        path = Path(value[5:]).expanduser()
        if not path.is_file():
            raise ValueError(f"storage profile {profile!r} needs credential file {path}")
        return path.read_text().strip()
    if isinstance(value, dict):
        return {key: _secret(entry, profile) for key, entry in value.items()}
    return json_value(value, f"storage profile {profile}")


def _profile(name: str, base: str, options: Mapping[str, object]) -> None:
    implementation = fsspec.get_filesystem_class(base)

    def initialize(self, **kwargs):
        resolved = {key: _secret(value, name) for key, value in {**options, **kwargs}.items()}
        implementation.__init__(self, **resolved)

    profile_name = f"{name.title()}FileSystem"
    profile_type = type(
        profile_name,
        (implementation,),
        {"protocol": (name, base), "__init__": initialize, "__module__": __name__},
    )
    globals()[profile_name] = profile_type
    fsspec.register_implementation(name, profile_type, clobber=True)


class _PinnedReader(io.BufferedReader):
    def __init__(self, raw: io.FileIO, pin: BinaryIO):
        super().__init__(raw)
        self.pin = pin

    def close(self) -> None:
        try:
            super().close()
        finally:
            self.pin.close()


class BoundedFileCache(WholeFileCacheFileSystem):
    """A filecache with an LRU byte limit and cross-process reader pins.

    Downloads share the cache directory's lock. Oversized files, and misses
    while every eviction candidate is pinned, stream directly from the store.
    """

    protocol = "dewcache"

    def __init__(self, *, max_bytes: int | str = 8 * 2**30, **kwargs):
        max_bytes = _byte_size(max_bytes)
        if max_bytes <= 0:
            raise ValueError("dewcache max_bytes must be positive")
        if "cache_storage" not in kwargs:
            remote = kwargs.get("fs")
            namespace = kwargs.get("target_protocol")
            if isinstance(remote, AbstractFileSystem):
                namespace = remote.protocol
                if isinstance(remote, S3FileSystem):
                    namespace = (namespace, remote.endpoint_url or remote.client_kwargs.get("endpoint_url"))
            digest = hashlib.sha256(json.dumps(namespace, sort_keys=True).encode()).hexdigest()[:16]
            kwargs["cache_storage"] = os.path.join(dew_cache_dir(), "transport", digest)
        super().__init__(**{"check_files": True, **kwargs})
        if len(self.storage) != 1:
            raise ValueError("dewcache needs one writable cache_storage directory")
        self._dew_max_bytes = max_bytes
        self._dew_hits = 0
        self._dew_misses = 0
        self._dew_downloaded = 0
        self._dew_wait = 0.0

    def __getattribute__(self, name: str):
        if name.startswith("_dew_") or name == "metrics":
            return object.__getattribute__(self, name)
        return super().__getattribute__(name)

    def _open(self, path: str, mode: str = "rb", **kwargs):
        if mode != "rb":
            return self.fs.open(path, mode=mode, **kwargs)
        snapshot = kwargs.pop("source_snapshot", None)
        started = time.monotonic()
        directory = Path(self.storage[-1])
        path = self._strip_protocol(path)
        name = self._mapper(path)
        reservation = directory / (".reserve-" + name)
        with FileLock(str(directory / (".fetch-" + name + ".lock"))):
            with FileLock(str(directory / ".dew-lock")):
                reservation.unlink(missing_ok=True)
                for interrupted in directory.glob(".download-" + name + "-*"):
                    interrupted.unlink()
            size = None
            if snapshot is not None and snapshot.get("sha256") is not None:
                version, size = "sha256:" + snapshot["sha256"], int(snapshot["size"])
            else:
                version = self.fs.ukey(path)
            with FileLock(str(directory / ".dew-lock")):
                self.load_cache()
                cached = self._metadata.check_file(path, cfs=None)
                if cached and (
                    cached[0]["uid"] == version
                    and (not self.expiry or time.time() - cached[0]["time"] <= self.expiry)
                ):
                    self._dew_hits += 1
                    return self._dew_pin(Path(cached[1]), started)
            size = self.fs.size(path) if size is None else size
            if not isinstance(size, int):
                raise ValueError(f"dewcache source {path!r} does not declare its byte size")
            with FileLock(str(directory / ".dew-lock")):
                self.load_cache()
                self._dew_misses += 1
                if not self._dew_discard(path) or not self._dew_room(size):
                    return self.fs.open(path, mode=mode, **kwargs)
                reservation.write_text(str(size))
            local = directory / name
            try:
                self._dew_download(path, local, size, version)
                with FileLock(str(directory / ".dew-lock")):
                    reservation.unlink()
                    return self._dew_pin(local, started)
            finally:
                with FileLock(str(directory / ".dew-lock")):
                    reservation.unlink(missing_ok=True)

    def _dew_pin(self, local: Path, started: float) -> _PinnedReader:
        pin = read_pin(local.with_name(local.name + ".pin"))
        os.utime(local, None)
        self._dew_wait += time.monotonic() - started
        return _PinnedReader(io.FileIO(str(local), "r"), pin)

    def _dew_discard(self, path: str) -> bool:
        previous = self._metadata.cached_files[-1].get(path)
        if previous is None:
            return True
        with evictable(Path(self.storage[-1]) / (previous["fn"] + ".pin")) as free:
            if free:
                self.pop_from_cache(path)
            return free

    def _dew_download(self, path: str, local: Path, size: int, version: str) -> None:
        descriptor, temporary = tempfile.mkstemp(dir=local.parent, prefix=".download-" + local.name + "-")
        os.close(descriptor)
        pending = Path(temporary)
        try:
            if version.startswith("sha256:"):
                with self.fs.open(path, "rb", size=size) as source, pending.open("wb") as target:
                    shutil.copyfileobj(source, target, length=2**20)
                with pending.open("rb") as downloaded:
                    unchanged = hashlib.file_digest(downloaded, "sha256").hexdigest() == version[7:]
            else:
                self.fs.get_file(path, str(pending))
                unchanged = self.fs.ukey(path) == version
            if pending.stat().st_size != size or not unchanged:
                raise OSError(f"dewcache source changed while downloading {path}")
            with FileLock(str(local.parent / ".dew-lock")):
                self.load_cache()
                os.replace(pending, local)
                self._metadata.update_file(
                    path,
                    {"original": path, "fn": local.name, "blocks": True, "time": time.time(), "uid": version},
                )
                self.save_cache()
            self._dew_downloaded += size
        finally:
            pending.unlink(missing_ok=True)

    def _dew_room(self, needed: int) -> bool:
        if needed > self._dew_max_bytes:
            return False
        directory = Path(self.storage[-1])
        reserved = 0
        active = set()
        for reservation in directory.glob(".reserve-*"):
            name = reservation.name.removeprefix(".reserve-")
            try:
                with FileLock(str(directory / (".fetch-" + name + ".lock")), timeout=0):
                    reservation.unlink()
                    for interrupted in directory.glob(".download-" + name + "-*"):
                        interrupted.unlink()
            except Timeout:
                reserved += int(reservation.read_text())
                active.add(name)
        candidates = []
        for path, detail in self._metadata.cached_files[-1].items():
            local = directory / detail["fn"]
            if local.is_file() and local.name not in active:
                stat = local.stat()
                candidates.append((stat.st_mtime_ns, path, local, stat.st_size))
        used = sum(entry[3] for entry in candidates) + reserved
        for _, path, local, size in sorted(candidates):
            if used + needed <= self._dew_max_bytes:
                break
            with evictable(directory / (local.name + ".pin")) as free:
                if not free:
                    continue
                self.pop_from_cache(path)
                used -= size
        self.save_cache()
        return used + needed <= self._dew_max_bytes

    def cat_file(self, path: str, start: int | None = None, end: int | None = None, **kwargs) -> bytes:
        with self._open(path, "rb") as file:
            if start is not None:
                file.seek(start, 2 if start < 0 else 0)
            size = -1 if end is None else max(0, end - file.tell())
            payload = file.read(size)
            if not isinstance(payload, bytes):
                raise TypeError("dewcache binary read returned non-bytes")
            return payload

    def metrics(self) -> TransportStats:
        return {
            "hits": self._dew_hits,
            "misses": self._dew_misses,
            "downloaded_bytes": self._dew_downloaded,
            "wait_seconds": self._dew_wait,
        }


# Registered as this module loads, which reads no configuration, so `dewcache::` URLs open through
# fsspec itself from the moment the data layer is imported.
fsspec.register_implementation("dewcache", BoundedFileCache, clobber=True)

_MEMORY_LOCK = threading.RLock()


def _local_file_key(path: str, size: int) -> tuple[str, int, int, int]:
    """The local metadata that invalidates a persisted content checksum."""
    realpath = os.path.realpath(path)
    metadata = os.stat(realpath)
    if metadata.st_size != size:
        raise OSError(f"local file changed before its content was versioned: {path}")
    return realpath, size, metadata.st_mtime_ns, metadata.st_ino


def content_version(fs: AbstractFileSystem, path: str, size: int) -> str:
    """A local file's full SHA256, persisted in Dew's cache until its metadata changes."""
    directory = Path(dew_cache_dir()) / "content-versions"
    directory.mkdir(parents=True, exist_ok=True)
    lock = hashlib.sha256(os.path.realpath(path).encode()).hexdigest()
    with FileLock(str(directory / (lock + ".lock"))), contextlib.closing(
        sqlite3.connect(directory / "versions.sqlite")
    ) as versions:
        key = _local_file_key(path, size)
        fingerprint = hashlib.sha256(json.dumps(key).encode()).hexdigest()
        versions.execute(
            "CREATE TABLE IF NOT EXISTS versions (file TEXT PRIMARY KEY, sha256 TEXT NOT NULL)"
        )
        cached = versions.execute("SELECT sha256 FROM versions WHERE file = ?", (fingerprint,)).fetchone()
        if cached is not None:
            return str(cached[0])
        with fs.open(path, "rb") as file:
            digest = hashlib.sha256()
            while payload := file.read(2**20):
                digest.update(payload)
        checksum = digest.hexdigest()
        if _local_file_key(path, size) != key:
            raise OSError(f"local file changed while its content was versioned: {path}")
        with versions:
            versions.execute("INSERT INTO versions VALUES (?, ?)", (fingerprint, checksum))
        return checksum


def local(fs: AbstractFileSystem) -> bool:
    """Whether `fs` is this host's own disk, whose paths are where it mounted a file."""
    return isinstance(fs, LocalFileSystem)


def on_gcs(fs: AbstractFileSystem) -> bool:
    """Whether `fs` is Google Cloud Storage, which ArrayRecord's and Arrow's own readers open by URL."""
    return bool({"gs", "gcs"}.intersection(_protocols(fs)))


def in_memory(fs: AbstractFileSystem) -> bool:
    """Whether `fs` is fsspec's in-process memory store: neither this host's disk nor the network."""
    return "memory" in _protocols(fs)


def _protocols(fs: AbstractFileSystem) -> tuple[str, ...]:
    return (fs.protocol,) if isinstance(fs.protocol, str) else tuple(fs.protocol)


@dataclass(frozen=True)
class Versioned:
    data: bytes
    version: str


class Readable(NamedTuple):
    """Where an object's bytes are read (`Store.readable`): the filesystem,
    the object's path on it, and, where a copy is read (a Hub revision's
    downloaded snapshot), the directory the copy's own paths start at; the
    names above it are the copy's location's, not the object's. None where
    the object is read in place."""

    fs: AbstractFileSystem
    path: str
    root: str | None


@dataclass(frozen=True)
class Store:
    """An object URL with compare-and-swap publication and verified uploads.

    `put_if_absent` and `put_if_match` run natively on local files, memory,
    S3 and the S3-compatible endpoints that honour its conditional PUTs, GCS
    and Hugging Face Hub repositories. Any other fsspec filesystem can
    supply them (`ConditionalWrites`) with a method `put_conditional(path,
    payload, version)`, which writes `payload` only while the object is
    absent (`version` None) or still at `version`, and returns the new
    version. It reports a failed precondition with any FileExistsError, so
    it needs nothing from Dew, and `Store` raises that again as
    PreconditionFailed. The filesystem's `ukey(path)` names the same
    versions, and `get_versioned` reads it on both sides of the bytes.
    """

    url: str
    storage_options: Mapping[str, object] | None = field(default=None, repr=False)
    retention: LocalRetention | None = field(default=None, repr=False)

    def resolve(self) -> tuple[AbstractFileSystem, str]:
        _registered()
        options = dict(self.storage_options or {})
        layers = self.url.split("::")
        for cache in ("dewcache", "filecache", "simplecache", "blockcache"):
            if cache not in layers[:-1]:
                options.pop(cache, None)
        return fsspec.core.url_to_fs(self.url, **options)

    def readable(self) -> Readable:
        """Where this object's bytes are read: a Hub repository's file at a
        commit is read from huggingface_hub's own cache where that holds it, as
        huggingface_hub reads its files, so a downloaded revision reads offline
        (HF_HUB_OFFLINE) and without a request online; any other object, and a
        Hub file the cache lacks, where `resolve` places it."""
        fs, path = self.resolve()
        if isinstance(fs, HfFileSystem):
            uri = parse_hf_uri(f"hf://{path}")
            if uri.revision is not None and REGEX_COMMIT_HASH.fullmatch(uri.revision):
                held = try_to_load_from_cache(uri.id, uri.path_in_repo, revision=uri.revision,
                                              repo_type=uri.type)
                if isinstance(held, str):
                    depth = len(PurePosixPath(uri.path_in_repo).parts)
                    return Readable(LocalFileSystem(), held, str(Path(held).parents[depth - 1]))
        return Readable(fs, path, None)

    def cached(self) -> Store:
        if self.url.startswith("dewcache::"):
            return self
        scheme = self.url.split("::", 1)[0] if "::" in self.url else self.url.split("://", 1)[0]
        if "://" not in self.url and "::" not in self.url:
            scheme = "file"
        options = dict(self.storage_options or {})
        transport = options.pop("dewcache", {})
        backend = options.pop(scheme, {})
        if not isinstance(backend, Mapping):
            raise ValueError(f"storage options for {scheme!r} must be a mapping")
        return Store("dewcache::" + self.url, {scheme: {**backend, **options}, "dewcache": transport})

    def child(self, path: str) -> Store:
        path = path.rstrip("/")
        if (
            not path
            or path.startswith("/")
            or "://" in path
            or any(part in ("", ".", "..") for part in path.split("/"))
        ):
            raise ValueError(f"storage child must stay inside its root: {path!r}")
        return Store(self.url.rstrip("/") + "/" + path, self.storage_options, self.retention)

    def describe(self) -> Mapping[str, object]:
        fs, _ = self.resolve()
        while isinstance(fs, CachingFileSystem):
            fs = fs.fs
        # A local path is where this host mounted the data, not how it is reached.
        settings = {"protocol": fs.protocol} if local(fs) else {"url": self.url, "protocol": fs.protocol}
        if isinstance(fs, S3FileSystem):
            settings.update(
                endpoint=fs.client_kwargs.get("endpoint_url") or fs.endpoint_url,
                region=fs.client_kwargs.get("region_name"),
            )
        return settings

    def read(self) -> bytes:
        fs, path = self.resolve()
        return fs.cat_file(path)

    def size(self) -> int:
        fs, path = self.resolve()
        size = fs.size(path)
        if not isinstance(size, int):
            raise ValueError(f"source {self.url} does not declare its byte size")
        return size

    def write(self, payload: bytes) -> None:
        """Write `payload` here whole, replacing what was. A local file is replaced, not
        rewritten in place: a rewrite truncates it first, and a reader that mapped it, a
        store's document index say, faults (SIGBUS) on any page past the cut until the
        write ends. An object store replaces an object whole already."""
        fs, path = self.resolve()
        fs.makedirs(fs._parent(path), exist_ok=True)
        if not local(fs):
            fs.pipe_file(path, payload)
            return
        write_atomically(path, payload)

    def verified(self, payload: bytes) -> ObjectInfo:
        """Write `payload` here where nothing is, and read it back as written."""
        self.put_if_absent(payload)
        written = object_info(payload)
        matching(object_info(self.read()), written, self.url)
        return written

    def checked(self, committed: Mapping[str, object]) -> bytes:
        """This object's bytes, refused unless they are the size and digest `committed` states."""
        payload = self.read()
        matching(object_info(payload), committed, self.url)
        return payload

    @property
    def remote(self) -> bool:
        """Whether this object lives behind a network service, where a failed write may pass and a
        retry is worth making; this host's disk and memory fail for reasons a retry keeps."""
        return self._backend()[2] not in ("file", "memory")

    def put_immutable(self, payload: bytes, retries: int = 0) -> str:
        """Create this object from `payload`, or accept the byte-identical
        object an earlier attempt created, and return the stored version.

        Other bytes already at the URL are refused with a ValueError. An object
        this call created that reads back otherwise is read again, up to
        `retries` times, a longer wait before each, then an OSError: no
        transport checks what a write left. A failed transfer raises at once,
        after the transport's own retries (s3fs's, gcsfs's, huggingface_hub's).
        """
        try:
            self.put_if_absent(payload)
        except PreconditionFailed:
            created = False
        else:
            created = True
        for attempt in range(retries + 1):
            if attempt:
                time.sleep(min(0.25 * 2 ** (attempt - 1), 10))
            found = self.get_versioned()
            if found.data == payload:
                return found.version
            if not created:
                raise ValueError(f"an immutable object already holds other bytes: {self.url}")
        raise OSError(f"uploaded object did not verify: {self.url}")

    def _backend(self) -> tuple[AbstractFileSystem, str, str]:
        fs, path = self.resolve()
        while isinstance(fs, CachingFileSystem):
            fs = fs.fs
        if issubclass(type(fs), ConditionalWrites):
            return fs, path, "conditional"
        if local(fs):
            return fs, path, "file"
        if in_memory(fs):
            return fs, path, "memory"
        for scheme in ("s3", "gs", "gcs", "hf"):
            if scheme in _protocols(fs):
                return fs, path, scheme
        raise ValueError(f"conditional publication is unavailable for {self.url}")

    @contextlib.contextmanager
    def _locked(self, path: str, scheme: str) -> Iterator[None]:
        if scheme == "memory":
            with _MEMORY_LOCK:
                yield
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with FileLock(path + ".cas.lock"):
                yield

    def get_versioned(self) -> Versioned:
        fs, path, scheme = self._backend()
        if scheme in ("file", "memory"):
            with self._locked(path, scheme):
                payload = fs.cat_file(path)
                return Versioned(payload, hashlib.sha256(payload).hexdigest())
        if scheme == "conditional":
            version = str(fs.ukey(path))
            payload = fs.cat_file(path)
            if str(fs.ukey(path)) != version:
                raise OSError(f"{self.url} changed while it was read")
            return Versioned(payload, version)
        if isinstance(fs, S3FileSystem):
            from fsspec.asyn import sync

            loop = fs.loop
            if loop is None:
                raise ValueError("Dew storage needs synchronous S3 filesystem options")
            bucket, key, _ = fs.split_path(path)
            # The filesystem's own request: its configured request settings (an SSE-C key) and its
            # error translation (a missing object raises FileNotFoundError) apply.
            response = fs.call_s3("get_object", Bucket=bucket, Key=key)

            async def read_body() -> bytes:
                async with response["Body"] as body:
                    return await body.read()

            return Versioned(sync(loop, read_body), response["ETag"])
        if scheme in ("gs", "gcs"):
            metadata = fs.info(path)
            version = str(metadata["generation"])
            return Versioned(fs.cat_file(path, generation=version, concurrency=1), version)
        if not isinstance(fs, HfFileSystem):
            raise ValueError(f"conditional publication is unavailable for {self.url}")
        resolved = fs.resolve_path(path)
        if not isinstance(resolved, HfFileSystemResolvedRepositoryPath):
            raise ValueError(f"conditional HF publication needs a repository object URL: {self.url}")
        from huggingface_hub import HfApi

        version = (
            HfApi(token=fs.token)
            .repo_info(resolved.repo_id, repo_type=resolved.repo_type, revision=resolved.revision)
            .sha
        )
        if version is None:
            raise ValueError(f"Hub source {self.url} did not resolve to a commit")
        pinned = f"{resolved.repo_type}s/{resolved.repo_id}@{version}/{resolved.path_in_repo}"
        return Versioned(fs.cat_file(pinned), version)

    def put_if_absent(self, payload: bytes) -> str:
        return self._put(payload, None)

    def put_if_match(self, payload: bytes, version: str) -> str:
        return self._put(payload, version)

    def _put(self, payload: bytes, version: str | None) -> str:
        fs, path, scheme = self._backend()
        if scheme in ("file", "memory"):
            reservation = (
                self.retention.reserve(Path(path), len(payload))
                if self.retention is not None
                else contextlib.nullcontext()
            )
            with reservation, self._locked(path, scheme):
                previous = hashlib.sha256(fs.cat_file(path)).hexdigest() if fs.exists(path) else None
                if previous != version:
                    raise PreconditionFailed(self.url)
                if scheme == "file":
                    write_atomically(path, payload)
                else:
                    fs.pipe_file(path, payload)
                return hashlib.sha256(payload).hexdigest()
        if isinstance(fs, ConditionalWrites):
            try:
                return fs.put_conditional(path, payload, version)
            except FileExistsError as error:
                raise PreconditionFailed(self.url) from error
        if isinstance(fs, S3FileSystem):
            from botocore.exceptions import ClientError

            if fs.loop is None:
                raise ValueError("Dew storage needs synchronous S3 filesystem options")
            bucket, key, _ = fs.split_path(path)
            # Through s3fs's own call, which adds the profile's arguments (its server-side encryption,
            # a KMS key, SSE-C) and its bucket's regional client, as every other write of it does.
            try:
                outcome = fs.call_s3("put_object", Bucket=bucket, Key=key, Body=payload,
                                     **({"IfNoneMatch": "*"} if version is None else {"IfMatch": version}))
            except OSError as error:
                # s3fs translates the refused condition; botocore's answer stays the error's cause.
                cause = error.__cause__
                if isinstance(cause, ClientError) and cause.response.get(
                        "ResponseMetadata", {}).get("HTTPStatusCode") in (409, 412):
                    raise PreconditionFailed(self.url) from error
                raise
            fs.invalidate_cache()
            return outcome["ETag"]
        if isinstance(fs, GCSFileSystem):
            bucket, key, _ = fs.split_path(path)
            # The JSON API's upload root beside the API root the filesystem reads (`base`).
            location = fs.base.removesuffix("storage/v1/")
            endpoint = f"{location}upload/storage/v1/b/{urllib.parse.quote(bucket, safe='')}/o"
            try:
                outcome = fs.call(
                    "POST",
                    endpoint,
                    uploadType="media",
                    name=key,
                    ifGenerationMatch="0" if version is None else version,
                    data=payload,
                    headers={"Content-Type": "application/octet-stream"},
                    json_out=True,
                )
            except FileExistsError as error:
                raise PreconditionFailed(self.url) from error
            fs.invalidate_cache()
            return str(outcome["generation"])
        from huggingface_hub import CommitOperationAdd, HfApi
        from huggingface_hub.errors import HfHubHTTPError

        if not isinstance(fs, HfFileSystem):
            raise ValueError(f"conditional publication is unavailable for {self.url}")
        resolved = fs.resolve_path(path)
        if not isinstance(resolved, HfFileSystemResolvedRepositoryPath):
            raise ValueError(f"conditional HF publication needs a repository object URL: {self.url}")
        api = HfApi(token=fs.token)
        parent = (
            version
            or api.repo_info(resolved.repo_id, repo_type=resolved.repo_type, revision=resolved.revision).sha
        )
        if parent is None:
            raise ValueError(f"HF publication has no parent commit at {self.url}")
        pinned = f"{resolved.repo_type}s/{resolved.repo_id}@{parent}/{resolved.path_in_repo}"
        if version is None and fs.exists(pinned):
            raise PreconditionFailed(self.url)
        try:
            outcome = api.create_commit(
                repo_id=resolved.repo_id,
                repo_type=resolved.repo_type,
                revision=resolved.revision,
                parent_commit=parent,
                commit_message="Publish data artifact",
                operations=[CommitOperationAdd(resolved.path_in_repo, payload)],
            )
        except HfHubHTTPError as error:
            if error.response.status_code in (409, 412):
                raise PreconditionFailed(self.url) from error
            raise
        return outcome.oid


@functools.cache
def _registered() -> None:
    """Register the profiles the storage config names (DEW_STORAGE_CONFIG, else storage.toml in Dew's
    configuration directory, `dew.cache.dew_config_dir`) with fsspec, once, as the first store
    resolves: the configuration is read where it is first used, so a broken one fails that store,
    not the import."""
    from dew.cache import dew_config_dir

    named = os.environ.get("DEW_STORAGE_CONFIG")
    config = Path(named).expanduser() if named else dew_config_dir() / "storage.toml"
    if config.is_file():
        for name, values in tomllib.loads(config.read_text()).items():
            options = dict(values)
            base = options.pop("base")
            _profile(name, base, options)
