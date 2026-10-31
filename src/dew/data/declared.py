"""Reads of an object as its producer declared it (`Declared`), never as other bytes: held to the version
its store published it at, checked against its digest, or read at a commit its URL pins."""

from __future__ import annotations

import hashlib
import io
import os
from collections.abc import Mapping
from typing import BinaryIO

from fsspec.implementations.cached import CachingFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.spec import AbstractFileSystem
from gcsfs import GCSFileSystem
from huggingface_hub import HfFileSystem, parse_hf_uri
from huggingface_hub.file_download import REGEX_COMMIT_HASH
from huggingface_hub.hf_file_system import HfFileSystemResolvedRepositoryPath
from s3fs import S3FileSystem

from dew._records import integer, text
from dew.data.storage import STORED, Store, _local_file_key, content_version, in_memory, local


class Replaced(OSError):
    """An object read where its producer published it, which holds other bytes than it published:
    replaced or removed since, whose records a read would take under the identity of the ones it
    replaced."""


class Pinned(io.RawIOBase):
    """An object's bytes at the version its producer published it, every read of them one request held
    to it (`ranged`): a file a reader seeks and reads in as it would the object's own."""

    def __init__(self, store: Store, stored: Mapping[str, str], size: int):
        super().__init__()
        self.store, self.stored, self.size, self.position = store, stored, size, 0
        self.name = store.url

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.position = max(0, {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
                            + offset)
        return self.position

    def readinto(self, buffer) -> int:
        count = min(len(buffer), self.size - self.position)
        if count <= 0:
            return 0
        payload = ranged(self.store, self.position, self.position + count, self.stored)
        buffer[:len(payload)] = payload
        self.position += len(payload)
        return len(payload)


PINNED_BUFFER = 8 << 20
"""The bytes a stream of a pinned object asks its store for at once (`Declared.open`)."""


def stored_kind(fs: AbstractFileSystem) -> str | None:
    """The version a remote store `fs` holds an object at and a read of it can be held to (`STORED`),
    none for a store with none such."""
    while isinstance(fs, CachingFileSystem):
        fs = fs.fs
    if isinstance(fs, S3FileSystem | HTTPFileSystem):
        return STORED["s3"]
    if isinstance(fs, GCSFileSystem):
        return STORED["gs"]
    return STORED["hf"] if isinstance(fs, HfFileSystem) else None


def pinned_by_url(fs: AbstractFileSystem, path: str) -> bool:
    """Whether the object at `path` is one version by its URL alone: a Hub repository's file at a full
    commit, which no write moves."""
    if not isinstance(fs, HfFileSystem):
        return False
    revision = parse_hf_uri(f"hf://{path}").revision
    return revision is not None and REGEX_COMMIT_HASH.fullmatch(revision) is not None


def checked_open(fs: AbstractFileSystem, path: str, size: int, digest: str) -> BinaryIO:
    """The local or in-memory file at `path`, open, holding the `size` bytes of `digest`, refused
    (`Replaced`) where it holds others: a local file opened first and its open file's metadata matched to
    the metadata its digest was taken (or kept, `content_version`) at, so the bytes checked are the bytes
    it reads; an in-memory one read whole and checked."""
    if not local(fs):
        payload = fs.cat_file(path)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise Replaced(f"{path} holds other bytes than the {size} its producer declared")
        return io.BytesIO(payload)
    opened = fs.open(os.path.realpath(path), "rb")
    try:
        held = os.fstat(opened.fileno())
        if held.st_size != size:
            raise Replaced(f"{path} holds {held.st_size} bytes, not the {size} its producer declared")
        found = content_version(fs, path, size)
        _, _, modified, inode = _local_file_key(path, size)
        if (held.st_size, held.st_mtime_ns, held.st_ino) != (size, modified, inode) or found != digest:
            raise Replaced(f"{path} holds other bytes than the {size} its producer declared")
    except BaseException:
        opened.close()
        raise
    return opened


class Declared:
    """The object at `store` read as its producer declared it (`declared`: its size, and its digest or
    the version its store published it at, `stored`), never as other bytes:
    - a Hub file at a full commit, which its URL pins, is read as its URL names it;
    - a remote object published at a version is read in ranges held to it (`Pinned`), refused
      (`Replaced`) where its store no longer holds it there;
    - a local or in-memory file is checked as each read opens it (`checked_open`);
    - any other remote object it states a digest of, and one a reader reads by its path on this host, is
      read whole into Dew's mirror and checked there, which a mirror without room for it refuses.
    A declaration with neither a digest nor a version of a store's own is refused: nothing holds a read
    of it to what it declares."""

    def __init__(self, store: Store, declared: Mapping[str, object]):
        self.store = store
        fs, path = store.resolve()
        self.size = integer(declared["bytes"], "declared size")
        digest = declared.get("sha256")
        self.digest = None if digest is None else text(digest, "declared digest")
        kind = stored_kind(fs)
        stored = declared.get("stored")
        # An ETag the declaration states is a version an S3 or HTTP read of it is held to.
        stored = dict(stored) if isinstance(stored, Mapping) else (
            {"etag": declared["ETag"]} if kind == "etag" and "ETag" in declared else {})
        self.stored = {kind: str(stored[kind])} if kind is not None and kind in stored else None
        self.pinned = pinned_by_url(fs, path)
        if not (self.pinned or self.stored or self.digest):
            raise ValueError(f"{store.url} is declared with neither a digest nor its store's version of "
                             "it, so no read of it can be held to the declaration: state its sha256")

    def __eq__(self, other) -> bool:
        return isinstance(other, Declared) and self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self._identity())

    def _identity(self) -> tuple:
        return self.store.url, self.size, self.digest, tuple(sorted((self.stored or {}).items()))

    def mirrored(self) -> io.BufferedReader:
        """The object's copy in Dew's mirror, checked against its digest as it downloads, open: the
        mirror keeps it while it is."""
        mirror, mirrored = self.store.cached().resolve()
        snapshot = None if self.digest is None else {"size": self.size, "sha256": self.digest}
        opened = mirror.open(mirrored, "rb", source_snapshot=snapshot)
        if not isinstance(opened, io.BufferedReader) or not isinstance(opened.name, str):
            opened.close()
            raise ValueError(f"{self.store.url} is read whole into Dew's mirror to check it, which has no "
                             f"room for its {self.size} bytes")
        return opened

    def open(self) -> BinaryIO:
        """A binary file of the declared bytes, a pinned object's read `PINNED_BUFFER` bytes at a time."""
        fs, path = self.store.resolve()
        if self.pinned:
            readable = self.store.readable()
            return readable.fs.open(readable.path, "rb")
        if self.stored is not None:
            return io.BufferedReader(Pinned(self.store, self.stored, self.size), buffer_size=PINNED_BUFFER)
        if self.digest is not None and (local(fs) or in_memory(fs)):
            return checked_open(fs, path, self.size, self.digest)
        return self.mirrored()

    def unbuffered(self) -> Pinned | BinaryIO:
        """A file of the declared bytes for a reader that plans its own reads (Arrow): a pinned object's
        each read one request, any other as `open` gives it."""
        return Pinned(self.store, self.stored, self.size) if self.stored is not None else self.open()

    def read_range(self, start: int, end: int) -> bytes:
        """The declared bytes `start` to `end`: in one request where they are held to a version, else
        from a file opened, checked, for the read alone, which holds nothing open after it."""
        if self.stored is not None:
            return ranged(self.store, start, end, self.stored)
        with self.open() as opened:
            opened.seek(start)
            return opened.read(end - start)

    def leased(self) -> tuple[str, BinaryIO]:
        """A path on this host's disk to the declared bytes, for a reader that opens paths alone
        (ArrayRecord's), and the open file that holds it so until it closes: a local file's own, checked
        (by its open descriptor where the host names one, `/proc/self/fd`, so the path reads the file
        checked), or its copy in Dew's mirror."""
        fs, path = self.store.resolve()
        if self.digest is None:
            # Its whole bytes are checked by their digest; a version alone holds ranged reads only.
            raise ValueError(f"{self.store.url} is read by its path, which needs the sha256 of its bytes "
                             "declared, and its declaration states only its store's version of it")
        if local(fs):
            opened = checked_open(fs, path, self.size, self.digest)
            descriptor = f"/proc/self/fd/{opened.fileno()}"
            return (descriptor if os.path.exists(descriptor) else path), opened
        held = self.mirrored()
        return held.name, held


def ranged(store: Store, start: int, end: int, stored: Mapping[str, str]) -> bytes:
    """Bytes `start` to `end` of the object at `store` as its store holds it at the version `stored` names
    (`STORED`), in one request held to it, which raises `Replaced` where the store no longer holds
    it there: an S3 range read on the object's ETag (`IfMatch`), an HTTP one on its ETag (`If-Match`,
    RFC 9110 13.1.1), a GCS one of its generation, a Hub file's at its commit."""
    fs, path = store.resolve()
    while isinstance(fs, CachingFileSystem):
        fs = fs.fs
    kind = stored_kind(fs)
    version = None if kind is None else stored.get(kind)
    if version is None:
        raise ValueError(f"{store.url} names no version its store holds it at: {dict(stored)}")
    if isinstance(fs, HTTPFileSystem):
        from aiohttp import ClientResponseError

        try:
            # Beside the headers the filesystem sends with every read (its storage options'), which
            # a call's own headers would replace.
            headers = {**dict(fs.kwargs.get("headers") or {}), "If-Match": version}
            return fs.cat_file(path, start=start, end=end, headers=headers)
        except ClientResponseError as error:
            if error.status == 412:
                raise Replaced(f"{store.url} is no longer the object published at ETag "
                               f"{version}") from error
            raise
    if isinstance(fs, S3FileSystem):
        from botocore.exceptions import ClientError
        from fsspec.asyn import sync

        loop = fs.loop
        if loop is None:
            raise ValueError("Dew storage needs synchronous S3 filesystem options")
        bucket, key, _ = fs.split_path(path)
        try:
            response = fs.call_s3("get_object", Bucket=bucket, Key=key,
                                  Range=f"bytes={start}-{end - 1}", IfMatch=version)
        except OSError as error:
            cause = error.__cause__
            if isinstance(cause, ClientError) and cause.response.get(
                    "ResponseMetadata", {}).get("HTTPStatusCode") == 412:
                raise Replaced(f"{store.url} is no longer the object published at ETag "
                               f"{version}") from error
            raise

        async def body() -> bytes:
            async with response["Body"] as read:
                return await read.read()

        return sync(loop, body)
    if isinstance(fs, GCSFileSystem):
        try:
            return fs.cat_file(path, start=start, end=end, generation=version, concurrency=1)
        except FileNotFoundError as error:
            raise Replaced(f"{store.url} holds no generation {version}: replaced or removed") from error
    resolved = fs.resolve_path(path) if isinstance(fs, HfFileSystem) else None
    if not isinstance(resolved, HfFileSystemResolvedRepositoryPath):
        raise ValueError(f"{store.url} is no repository file")
    return fs.cat_file(f"{resolved.repo_type}s/{resolved.repo_id}@{version}/{resolved.path_in_repo}",
                       start=start, end=end)
