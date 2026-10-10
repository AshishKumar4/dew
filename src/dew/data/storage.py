"""URL storage, named credentials, bounded transport caching and conditional puts."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import json
import os
import re
import tempfile
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import fsspec
from filelock import FileLock
from fsspec.implementations.cached import WholeFileCacheFileSystem
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem


class PreconditionFailed(FileExistsError):
    """The object changed since the caller read its version."""


def _secret(value: object, profile: str) -> object:
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
        return {key: _secret(item, profile) for key, item in value.items()}
    return value


def _profile(name: str, base: str, options: Mapping[str, object]) -> None:
    implementation = fsspec.get_filesystem_class(base)

    class Profile(implementation):
        protocol = (name, base)

        def __init__(self, **kwargs: object):
            resolved = {key: _secret(value, name) for key, value in {**options, **kwargs}.items()}
            super().__init__(**resolved)

    Profile.__name__ = f"{name.title()}FileSystem"
    Profile.__qualname__ = Profile.__name__
    globals()[Profile.__name__] = Profile
    fsspec.register_implementation(name, Profile, clobber=True)


class DewDataFileSystem(AbstractFileSystem):
    """The scoped dew-data Worker, including range reads and conditional writes.

    A deployment exposes one bucket, named by the URL's first component.
    The token is resolved on first use from DEW_DATA_TOKEN or its configured
    file. A Worker has no delete permission; lifecycle rules own expiry.
    """

    protocol = ("dewdata", "r2")
    cachable = False

    def __init__(self, *, endpoint_url: str | None = None, token: str | None = None,
                 token_file: str | None = None, bucket: str = "dew-data", **kwargs: object):
        super().__init__(**kwargs)
        config = Path(os.environ.get("DEW_DATA_CONFIG", "~/.config/dew/dew-data.json")).expanduser()
        settings = json.loads(config.read_text()) if config.is_file() else {}
        self.endpoint = (endpoint_url or os.environ.get("DEW_DATA_URL") or settings.get("url")
                         or "https://dew-data.ashishkmr472.workers.dev").rstrip("/")
        self.token_file = token_file or settings.get("token_file")
        self.token = token
        self.bucket = bucket

    def _key(self, path: str) -> str:
        stripped = self._strip_protocol(path).lstrip("/")
        bucket, _, key = stripped.partition("/")
        if bucket != self.bucket:
            raise ValueError(f"dew-data profile serves bucket {self.bucket!r}, not {bucket!r}")
        if key and (len(key) > 512 or not re.fullmatch(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*", key)):
            raise ValueError(f"invalid dew-data object key {key!r}")
        return key

    def _request(self, method: str, path: str, *, data: bytes | None = None,
                 headers: Mapping[str, str] | None = None, query: str = "") -> tuple[bytes, dict[str, str]]:
        token = self.token or os.environ.get("DEW_DATA_TOKEN")
        if not token and self.token_file:
            token = Path(self.token_file).expanduser().read_text().strip()
        if not token:
            raise ValueError("storage profile 'r2' needs DEW_DATA_TOKEN or its configured token_file")
        key = self._key(path)
        request = urllib.request.Request(
            f"{self.endpoint}/{key}{query}", data=data, method=method,
            headers={"Authorization": f"Bearer {token}", "User-Agent": "dew-data/1", **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read()
                if headers and "Range" in headers and response.status != 206:
                    raise OSError(f"dew-data did not honour range read for {path}")
                return body, dict(response.headers.items())
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise FileNotFoundError(path) from error
            if error.code in (409, 412):
                raise PreconditionFailed(path) from error
            raise OSError(f"dew-data {method} {path}: HTTP {error.code}") from error

    def info(self, path: str, **kwargs: object) -> dict[str, object]:
        if not self._key(path):
            return {"name": self.bucket, "size": 0, "type": "directory"}
        _, headers = self._request("HEAD", path)
        normalized = {key.lower(): value for key, value in headers.items()}
        return {"name": self._strip_protocol(path), "type": "file",
                "size": int(normalized["content-length"]), "etag": normalized["etag"]}

    def ls(self, path: str, detail: bool = True, **kwargs: object):
        prefix = self._key(path.rstrip("/"))
        cursor = ""
        entries: dict[str, dict[str, object]] = {}
        while True:
            query = "?" + urllib.parse.urlencode({"list": prefix, "cursor": cursor})
            body, _ = self._request("GET", self.bucket, query=query)
            page = json.loads(body)
            for suffix in page["keys"]:
                key = prefix + suffix
                tail = key[len(prefix):].lstrip("/")
                first, slash, _ = tail.partition("/")
                name = f"{self.bucket}/{prefix.rstrip('/') + '/' if prefix else ''}{first}"
                entries[name] = ({"name": name, "type": "directory", "size": 0} if slash
                                 else self.info(f"{self.bucket}/{key}"))
            if page["cursor"] is None:
                break
            cursor = page["cursor"]
        found = sorted(entries.values(), key=lambda entry: str(entry["name"]))
        return found if detail else [entry["name"] for entry in found]

    def cat_file(self, path: str, start: int | None = None, end: int | None = None,
                 **kwargs: object) -> bytes:
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

    def pipe_file(self, path: str, value: bytes, **kwargs: object) -> None:
        if not 0 < len(value) <= 100_000_000:
            raise ValueError(f"dew-data object {path} must be between 1 byte and 100 MB; write shards")
        self._request("PUT", path, data=value)

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        self._key(path.rstrip("/"))

    def _open(self, path: str, mode: str = "rb", block_size: int | None = None,
              **kwargs: object):
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

    def write(self, data: bytes) -> int:
        if self.tell() + len(data) > 100_000_000:
            raise ValueError(f"dew-data object {self.path} exceeds 100 MB; write smaller shards")
        return super().write(data)

    def close(self) -> None:
        if not self.closed:
            try:
                self.filesystem.pipe_file(self.path, self.getvalue())
            finally:
                super().close()


class _PinnedReader(io.BufferedReader):
    def __init__(self, raw: BinaryIO, pin: BinaryIO):
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

    def __init__(self, *, max_bytes: int = 8 * 2**30, **kwargs: object):
        if max_bytes <= 0:
            raise ValueError("dewcache max_bytes must be positive")
        super().__init__(**{"check_files": True, **kwargs})
        if len(self.storage) != 1:
            raise ValueError("dewcache needs one writable cache_storage directory")
        self._dew_max_bytes = max_bytes
        self._dew_hits = 0
        self._dew_misses = 0
        self._dew_downloaded = 0
        self._dew_wait = 0.0

    def __getattribute__(self, name: str):
        if name.startswith("_dew_"):
            return object.__getattribute__(self, name)
        return super().__getattribute__(name)

    def _open(self, path: str, mode: str = "rb", **kwargs: object):
        if mode != "rb":
            return self.fs.open(path, mode=mode, **kwargs)
        started = time.monotonic()
        directory = Path(self.storage[-1])
        with FileLock(str(directory / ".dew-lock")):
            self.load_cache()
            path = self._strip_protocol(path)
            cached = self._check_file(path)
            if not cached:
                self._dew_misses += 1
                size = self.fs.size(path)
                if not self._dew_room(size):
                    return self.fs.open(path, mode=mode, **kwargs)
                self._dew_downloaded += size
            else:
                self._dew_hits += 1
            opened = super()._open(path, mode=mode, **kwargs)
            local = Path(opened.name)
            pin = (directory / (local.name + ".pin")).open("a+b")
            fcntl.flock(pin.fileno(), fcntl.LOCK_SH)
            os.utime(local, None)
            self._dew_wait += time.monotonic() - started
            return _PinnedReader(opened, pin)

    def _dew_room(self, needed: int) -> bool:
        if needed > self._dew_max_bytes:
            return False
        directory = Path(self.storage[-1])
        candidates = []
        for path, detail in self._metadata.cached_files[-1].items():
            local = directory / detail["fn"]
            if local.is_file():
                stat = local.stat()
                candidates.append((stat.st_mtime_ns, path, local, stat.st_size))
        used = sum(entry[3] for entry in candidates)
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


_MEMORY_LOCK = threading.RLock()


@dataclass(frozen=True)
class Versioned:
    data: bytes
    version: str


@dataclass(frozen=True)
class Store:
    """An object URL with compare-and-swap publication and verified uploads."""

    url: str
    storage_options: Mapping[str, object] | None = None

    def resolve(self) -> tuple[AbstractFileSystem, str]:
        return fsspec.core.url_to_fs(self.url, **(self.storage_options or {}))

    def child(self, path: str) -> Store:
        return Store(self.url.rstrip("/") + "/" + path, self.storage_options)

    def read(self) -> bytes:
        fs, path = self.resolve()
        return fs.cat_file(path)

    def write(self, data: bytes) -> None:
        fs, path = self.resolve()
        fs.makedirs(fs._parent(path), exist_ok=True)
        fs.pipe_file(path, data)

    def verified(self, data: bytes) -> dict[str, object]:
        self.put_if_absent(data)
        found = self.read()
        digest = hashlib.sha256(data).hexdigest()
        if len(found) != len(data) or hashlib.sha256(found).hexdigest() != digest:
            raise OSError(f"uploaded object did not verify: {self.url}")
        return {"bytes": len(data), "sha256": digest}

    def _backend(self) -> tuple[AbstractFileSystem, str, str]:
        fs, path = self.resolve()
        while isinstance(fs, WholeFileCacheFileSystem):
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
                data = fs.cat_file(path)
                return Versioned(data, hashlib.sha256(data).hexdigest())
        if isinstance(fs, DewDataFileSystem):
            data, headers = fs._request("GET", path)
            return Versioned(data, next(value for key, value in headers.items() if key.lower() == "etag"))
        info = fs.info(path)
        if scheme == "s3":
            version = info["ETag"]
            return Versioned(fs.cat_file(path, IfMatch=version), version)
        if scheme in ("gs", "gcs"):
            version = str(info["generation"])
            return Versioned(fs.cat_file(path, generation=version), version)
        resolved = fs.resolve_path(path)
        from huggingface_hub import HfApi

        version = HfApi(token=fs.token).repo_info(resolved.repo_id, repo_type=resolved.repo_type,
                                                revision=resolved.revision).sha
        pinned = f"{resolved.repo_type}s/{resolved.repo_id}@{version}/{resolved.path_in_repo}"
        return Versioned(fs.cat_file(pinned), version)

    def put_if_absent(self, data: bytes) -> str:
        return self._put(data, None)

    def put_if_match(self, data: bytes, version: str) -> str:
        return self._put(data, version)

    def _put(self, data: bytes, version: str | None) -> str:
        fs, path, scheme = self._backend()
        if scheme in ("file", "memory"):
            with self._locked(path, scheme):
                previous = hashlib.sha256(fs.cat_file(path)).hexdigest() if fs.exists(path) else None
                if previous != version:
                    raise PreconditionFailed(self.url)
                if scheme == "file":
                    with tempfile.NamedTemporaryFile(dir=str(Path(path).parent), delete=False) as pending:
                        pending.write(data)
                        pending.flush()
                        os.fsync(pending.fileno())
                    os.replace(pending.name, path)
                else:
                    fs.pipe_file(path, data)
                return hashlib.sha256(data).hexdigest()
        headers = {"If-None-Match": "*"} if version is None else {"If-Match": version}
        if isinstance(fs, DewDataFileSystem):
            _, response = fs._request("PUT", path, data=data, headers=headers)
            return next(value for key, value in response.items() if key.lower() == "etag")
        if scheme == "s3":
            from botocore.exceptions import ClientError

            bucket, key, _ = fs.split_path(path)
            try:
                result = fs.call_s3("put_object", Bucket=bucket, Key=key, Body=data,
                                    **({"IfNoneMatch": "*"} if version is None else {"IfMatch": version}))
            except ClientError as error:
                if error.response["ResponseMetadata"]["HTTPStatusCode"] in (409, 412):
                    raise PreconditionFailed(self.url) from error
                raise
            return result["ETag"]
        if scheme in ("gs", "gcs"):
            fs.pipe_file(path, data, if_generation_match=0 if version is None else int(version))
            fs.invalidate_cache(path)
            return str(fs.info(path)["generation"])
        from huggingface_hub import CommitOperationAdd, HfApi
        from huggingface_hub.errors import HfHubHTTPError

        resolved = fs.resolve_path(path)
        api = HfApi(token=fs.token)
        parent = version or api.repo_info(resolved.repo_id, repo_type=resolved.repo_type,
                                          revision=resolved.revision).sha
        if version is None and fs.exists(path):
            raise PreconditionFailed(self.url)
        try:
            result = api.create_commit(repo_id=resolved.repo_id, repo_type=resolved.repo_type,
                                       revision=resolved.revision, parent_commit=parent,
                                       commit_message="Publish data artifact",
                                       operations=[CommitOperationAdd(resolved.path_in_repo, data)])
        except HfHubHTTPError as error:
            if error.response.status_code in (409, 412):
                raise PreconditionFailed(self.url) from error
            raise
        return result.oid


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
