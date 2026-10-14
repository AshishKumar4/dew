"""Local artifact retention, with generation and writer pins shared across processes."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from filelock import FileLock


class LocalRetention:
    def __init__(self, directory: Path, max_bytes: int):
        self.directory, self.max_bytes = directory, max_bytes

    @staticmethod
    def pin(directory: Path) -> BinaryIO:
        directory.parent.mkdir(parents=True, exist_ok=True)
        lock = (directory.parent / f".{directory.name}.pin").open("a+b")
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        return lock

    @staticmethod
    @contextlib.contextmanager
    def _exclusive(directory: Path) -> Iterator[bool]:
        directory.parent.mkdir(parents=True, exist_ok=True)
        with (directory.parent / f".{directory.name}.pin").open("a+b") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
            else:
                yield True

    def _keys(self) -> list[Path]:
        return [
            path
            for path in self.directory.iterdir()
            if path.is_dir() and re.fullmatch(r"[a-f0-9]{64}", path.name)
        ]

    def _size(self) -> int:
        return sum(path.stat().st_size for key in self._keys() for path in key.rglob("*") if path.is_file())

    @staticmethod
    def _reclaim(key: Path) -> None:
        referenced = set()
        for manifest in (key / "generations").glob("*/manifest.json"):
            referenced.update(json.loads(manifest.read_bytes())["shards"])
        for writer in (key / "shards").glob("*"):
            if not writer.is_dir() or not re.fullmatch(r"[a-f0-9]{32}", writer.name):
                continue
            if any(f"shards/{writer.name}/" in path for path in referenced):
                continue
            with LocalRetention._exclusive(writer) as available:
                if not available:
                    continue
                segments = key / "segments" / writer.name
                if (segments / "seal.json").is_file():
                    shutil.rmtree(writer)
                    shutil.rmtree(segments)

    @contextlib.contextmanager
    def reserve(self, target: Path, byte_count: int) -> Iterator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.directory / ".retention.lock")):
            replaced = target.stat().st_size if target.is_file() else 0
            needed = max(0, byte_count - replaced)
            candidates = sorted(
                (manifest.stat().st_atime_ns, manifest.parent)
                for key in self._keys()
                for manifest in (key / "generations").glob("*/manifest.json")
                if re.fullmatch(r"[a-f0-9]{32}", manifest.parent.name)
            )
            for _, generation in candidates:
                if self._size() + needed <= self.max_bytes:
                    break
                if target.is_relative_to(generation):
                    continue
                with self._exclusive(generation) as available:
                    if not available:
                        continue
                    key = generation.parent.parent
                    current = key / "CURRENT"
                    if target == current:
                        continue
                    if (
                        current.is_file()
                        and json.loads(current.read_bytes())["generation"] == generation.name
                    ):
                        with FileLock(str(current) + ".cas.lock"):
                            current.unlink()
                    shutil.rmtree(generation)
                    self._reclaim(key)
            if self._size() + needed > self.max_bytes:
                raise OSError(
                    f"local cache {self.directory} cannot admit {needed} bytes within "
                    f"keep_local_bytes={self.max_bytes}; its remaining generations or writers are pinned"
                )
            yield

    @staticmethod
    def touched(directory: Path) -> None:
        manifest = directory / "manifest.json"
        if manifest.is_file():
            os.utime(manifest, None)
