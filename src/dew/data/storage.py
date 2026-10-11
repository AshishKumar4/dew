"""URL storage, named credentials, bounded transport caching and conditional puts."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import http.client
import io
import json
import os
import re
import shutil
import tempfile
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Buffer, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, TypedDict

import fsspec
from filelock import FileLock, Timeout
from fsspec.implementations.cached import CachingFileSystem, WholeFileCacheFileSystem
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from gcsfs import GCSFileSystem
from huggingface_hub import HfFileSystem
from huggingface_hub.hf_file_system import HfFileSystemResolvedRepositoryPath
from s3fs import S3FileSystem

from dew.cache import dew_cache_dir
from dew.data.retention import LocalRetention
from dew.records import JSON, json_value


class PreconditionFailed(FileExistsError):
    """The object changed since the caller read its version."""


class ObjectInfo(TypedDict):
    bytes: int
    sha256: str


class ObjectRef(ObjectInfo):
    path: str


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


class DewDataFileSystem(AbstractFileSystem):
    """The scoped dew-data Worker, including range reads and conditional writes.

    A deployment exposes one bucket, named by the URL's first component.
    The token is resolved on first use from DEW_DATA_TOKEN or its configured
    file. Write-token holders can delete objects; lifecycle rules expire abandoned smokes.
    """

    protocol = ("dewdata", "r2")
    cachable = False

    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        token: str | None = None,
        token_file: str | None = None,
        bucket: str = "dew-data",
        **kwargs,
    ):
        super().__init__(**kwargs)
        config = Path(os.environ.get("DEW_DATA_CONFIG", "~/.config/dew/dew-data.json")).expanduser()
        settings = json.loads(config.read_text()) if config.is_file() else {}
        self.endpoint = (
            endpoint_url
            or os.environ.get("DEW_DATA_URL")
            or settings.get("url")
            or "https://dew-data.ashishkmr472.workers.dev"
        ).rstrip("/")
        self.token_file = token_file or settings.get("token_file")
        self.token = token
        self.bucket = bucket

    def _key(self, path: str) -> str:
        stripped = self._strip_protocol(path).lstrip("/")
        bucket, _, key = stripped.partition("/")
        if bucket != self.bucket:
            raise ValueError(f"dew-data profile serves bucket {self.bucket!r}, not {bucket!r}")
        if key and (
            len(key) > 512
            or not re.fullmatch(r"[A-Za-z0-9._=-]+(?:/[A-Za-z0-9._=-]+)*", key)
            or ".." in key.split("/")
        ):
            raise ValueError(f"invalid dew-data object key {key!r}")
        return key

    @contextlib.contextmanager
    def _response(
        self,
        method: str,
        path: str,
        *,
        payload: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        query: str = "",
    ) -> Iterator[http.client.HTTPResponse]:
        token = self.token or os.environ.get("DEW_DATA_TOKEN")
        if not token and self.token_file:
            token = Path(self.token_file).expanduser().read_text().strip()
        if not token:
            raise ValueError("storage profile 'r2' needs DEW_DATA_TOKEN or its configured token_file")
        key = self._key(path)
        request = urllib.request.Request(
            f"{self.endpoint}/{key}{query}",
            data=payload,
            method=method,
            headers={"Authorization": f"Bearer {token}", "User-Agent": "dew-data/1", **(headers or {})},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if headers and "Range" in headers and response.status != 206:
                    raise OSError(f"dew-data did not honour range read for {path}")
                yield response
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise FileNotFoundError(path) from error
            if error.code in (409, 412):
                raise PreconditionFailed(path) from error
            raise OSError(f"dew-data {method} {path}: HTTP {error.code}") from error

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        query: str = "",
    ) -> tuple[bytes, dict[str, str]]:
        with self._response(method, path, payload=payload, headers=headers, query=query) as response:
            return response.read(), dict(response.headers.items())

    def get_file(self, rpath: str, lpath: str, **kwargs) -> None:
        self._get_versioned(rpath, lpath)

    def _get_versioned(self, rpath: str, lpath: str, version: str | None = None) -> None:
        with self._response("GET", rpath) as response, open(lpath, "wb") as target:
            if version is not None and response.headers.get("ETag") != version:
                raise OSError(f"dew-data source changed before downloading {rpath}")
            shutil.copyfileobj(response, target, length=2**20)

    def ukey(self, path: str) -> str:
        return str(self.info(path)["etag"])

    def info(self, path: str, **kwargs) -> dict[str, str | int]:
        key = self._key(path)
        if not key:
            return {"name": self.bucket, "size": 0, "type": "directory"}
        try:
            _, headers = self._request("HEAD", path)
        except FileNotFoundError:
            if next(self._keys(key.rstrip("/") + "/"), None) is None:
                raise
            return {"name": self._strip_protocol(path), "size": 0, "type": "directory"}
        normalized = {key.lower(): value for key, value in headers.items()}
        return {
            "name": self._strip_protocol(path),
            "type": "file",
            "size": int(normalized["content-length"]),
            "etag": normalized["etag"],
        }

    def _keys(self, prefix: str) -> Iterator[str]:
        cursor = ""
        while True:
            options = {"list": prefix, **({"cursor": cursor} if cursor else {})}
            query = "?" + urllib.parse.urlencode(options)
            body, _ = self._request("GET", self.bucket, query=query)
            page = json.loads(body)
            for suffix in page["keys"]:
                yield prefix + suffix
            if page["cursor"] is None:
                break
            cursor = page["cursor"]

    def ls(self, path: str, detail: bool = True, **kwargs):
        prefix = self._key(path.rstrip("/"))
        prefix = prefix.rstrip("/") + "/" if prefix else ""
        entries: dict[str, dict[str, str | int]] = {}
        for key in self._keys(prefix):
            tail = key[len(prefix) :]
            first, slash, _ = tail.partition("/")
            name = f"{self.bucket}/{prefix}{first}"
            entries[name] = (
                {"name": name, "type": "directory", "size": 0}
                if slash
                else self.info(name)
                if detail
                else {"name": name, "type": "file", "size": 0}
            )
        found = sorted(entries.values(), key=lambda entry: str(entry["name"]))
        return found if detail else [entry["name"] for entry in found]

    def cat_file(self, path: str, start: int | None = None, end: int | None = None, **kwargs) -> bytes:
        headers = {}
        if start is not None or end is not None:
            if start is not None and start < 0:
                headers["Range"] = f"bytes={start}"
            else:
                headers["Range"] = f"bytes={start or 0}-{'' if end is None else end - 1}"
            if end is not None and end <= (start or 0):
                return b""
        body, _ = self._request("GET", path, headers=headers)
        return body

    def pipe_file(self, path: str, value: bytes, mode: str = "overwrite", **kwargs) -> None:
        if not 0 < len(value) <= 100_000_000:
            raise ValueError(f"dew-data object {path} must be between 1 byte and 100 MB; write shards")
        self._request("PUT", path, payload=value)

    def rm_file(self, path: str) -> None:
        self._request("DELETE", path)

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        self._key(path.rstrip("/"))

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        autocommit: bool = True,
        cache_options: Mapping[str, object] | None = None,
        **kwargs,
    ):
        if mode == "rb":
            return _WorkerReader(self, path, block_size=block_size or 2**20)
        if mode == "wb":
            return _WorkerWriter(self, path)
        raise ValueError(f"dew-data does not support mode {mode!r}")


class _WorkerReader(AbstractBufferedFile):
    def _fetch_range(self, start: int, end: int) -> bytes:
        return self.fs.cat_file(self.path, start=start, end=end)


class _WorkerWriter(io.BytesIO):
    def __init__(self, filesystem: DewDataFileSystem, path: str):
        super().__init__()
        self.filesystem = filesystem
        self.path = path

    def write(self, payload: Buffer) -> int:
        if self.tell() + memoryview(payload).nbytes > 100_000_000:
            raise ValueError(f"dew-data object {self.path} exceeds 100 MB; write smaller shards")
        return super().write(payload)

    def close(self) -> None:
        if not self.closed:
            try:
                self.filesystem.pipe_file(self.path, self.getvalue())
            finally:
                super().close()


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
                elif isinstance(remote, DewDataFileSystem):
                    namespace = (namespace, remote.endpoint)
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
            if isinstance(self.fs, DewDataFileSystem):
                metadata = self.fs.info(path) if snapshot is None else snapshot
                version, size = str(metadata["etag"]), int(metadata["size"])
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
        pin = local.with_name(local.name + ".pin").open("a+b")
        fcntl.flock(pin.fileno(), fcntl.LOCK_SH)
        os.utime(local, None)
        self._dew_wait += time.monotonic() - started
        return _PinnedReader(io.FileIO(str(local), "r"), pin)

    def _dew_discard(self, path: str) -> bool:
        previous = self._metadata.cached_files[-1].get(path)
        if previous is None:
            return True
        pin_path = Path(self.storage[-1]) / (previous["fn"] + ".pin")
        with pin_path.open("a+b") as pin:
            try:
                fcntl.flock(pin.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            self.pop_from_cache(path)
        return True

    def _dew_download(self, path: str, local: Path, size: int, version: str) -> None:
        descriptor, temporary = tempfile.mkstemp(dir=local.parent, prefix=".download-" + local.name + "-")
        os.close(descriptor)
        pending = Path(temporary)
        try:
            if isinstance(self.fs, DewDataFileSystem):
                self.fs._get_versioned(path, str(pending), version)
                unchanged = True
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
            with (directory / (local.name + ".pin")).open("a+b") as pin:
                try:
                    fcntl.flock(pin.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
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


_MEMORY_LOCK = threading.RLock()


@dataclass(frozen=True)
class Versioned:
    data: bytes
    version: str


@dataclass(frozen=True)
class Store:
    """An object URL with compare-and-swap publication and verified uploads."""

    url: str
    storage_options: Mapping[str, object] | None = field(default=None, repr=False)
    retention: LocalRetention | None = field(default=None, repr=False)

    def resolve(self) -> tuple[AbstractFileSystem, str]:
        options = dict(self.storage_options or {})
        layers = self.url.split("::")
        for cache in ("dewcache", "filecache", "simplecache", "blockcache"):
            if cache not in layers[:-1]:
                options.pop(cache, None)
        return fsspec.core.url_to_fs(self.url, **options)

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
        return Store(self.url.rstrip("/") + "/" + path, self.storage_options, self.retention)

    def describe(self) -> Mapping[str, object]:
        fs, _ = self.resolve()
        while isinstance(fs, CachingFileSystem):
            fs = fs.fs
        settings = {"url": self.url, "protocol": fs.protocol}
        if isinstance(fs, S3FileSystem):
            settings.update(
                endpoint=fs.client_kwargs.get("endpoint_url") or fs.endpoint_url,
                region=fs.client_kwargs.get("region_name"),
            )
        elif isinstance(fs, DewDataFileSystem):
            settings.update(endpoint=fs.endpoint)
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
        fs, path = self.resolve()
        fs.makedirs(fs._parent(path), exist_ok=True)
        fs.pipe_file(path, payload)

    def verified(self, payload: bytes) -> ObjectInfo:
        self.put_if_absent(payload)
        found = self.read()
        digest = hashlib.sha256(payload).hexdigest()
        if len(found) != len(payload) or hashlib.sha256(found).hexdigest() != digest:
            raise OSError(f"uploaded object did not verify: {self.url}")
        return {"bytes": len(payload), "sha256": digest}

    def _backend(self) -> tuple[AbstractFileSystem, str, str]:
        fs, path = self.resolve()
        while isinstance(fs, CachingFileSystem):
            fs = fs.fs
        protocol = fs.protocol
        schemes = (protocol,) if isinstance(protocol, str) else protocol
        for scheme in ("file", "memory", "s3", "gs", "gcs", "hf", "dewdata", "r2"):
            if scheme in schemes:
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
        if isinstance(fs, DewDataFileSystem):
            payload, headers = fs._request("GET", path)
            return Versioned(payload, next(value for key, value in headers.items() if key.lower() == "etag"))
        if isinstance(fs, S3FileSystem):
            from fsspec.asyn import sync

            async def read_s3() -> Versioned:
                bucket, key, _ = fs.split_path(path)
                response = await fs._call_s3("get_object", Bucket=bucket, Key=key)
                async with response["Body"] as body:
                    return Versioned(await body.read(), response["ETag"])

            loop = fs.loop
            if loop is None:
                raise ValueError("Dew storage needs synchronous S3 filesystem options")
            return sync(loop, read_s3)
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
                    with tempfile.NamedTemporaryFile(dir=str(Path(path).parent), delete=False) as pending:
                        pending.write(payload)
                        pending.flush()
                        os.fsync(pending.fileno())
                    os.replace(pending.name, path)
                else:
                    fs.pipe_file(path, payload)
                return hashlib.sha256(payload).hexdigest()
        headers = {"If-None-Match": "*"} if version is None else {"If-Match": version}
        if isinstance(fs, DewDataFileSystem):
            _, response = fs._request("PUT", path, payload=payload, headers=headers)
            return next(value for key, value in response.items() if key.lower() == "etag")
        if isinstance(fs, S3FileSystem):
            from botocore.exceptions import ClientError
            from fsspec.asyn import sync

            bucket, key, _ = fs.split_path(path)

            async def put_s3() -> str:
                client = await fs.set_session()
                try:
                    outcome = await client.put_object(
                        Bucket=bucket,
                        Key=key,
                        Body=payload,
                        **({"IfNoneMatch": "*"} if version is None else {"IfMatch": version}),
                    )
                except ClientError as error:
                    if error.response["ResponseMetadata"]["HTTPStatusCode"] in (409, 412):
                        raise PreconditionFailed(self.url) from error
                    raise
                return outcome["ETag"]

            loop = fs.loop
            if loop is None:
                raise ValueError("Dew storage needs synchronous S3 filesystem options")
            version = sync(loop, put_s3)
            fs.invalidate_cache()
            return version
        if isinstance(fs, GCSFileSystem):
            bucket, key, _ = fs.split_path(path)
            endpoint = f"{fs._location}/upload/storage/v1/b/{urllib.parse.quote(bucket, safe='')}/o"
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


def _register() -> None:
    fsspec.register_implementation("dewcache", BoundedFileCache, clobber=True)
    fsspec.register_implementation("dewdata", DewDataFileSystem, clobber=True)
    fsspec.register_implementation("r2", DewDataFileSystem, clobber=True)
    config = Path(os.environ.get("DEW_STORAGE_CONFIG", "~/.config/dew/storage.toml")).expanduser()
    if config.is_file():
        for name, values in tomllib.loads(config.read_text()).items():
            options = dict(values)
            base = options.pop("base")
            _profile(name, base, options)


_register()
