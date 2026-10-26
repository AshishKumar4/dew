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


def read_pin(path: Path) -> BinaryIO:
    """A reader's pin at `path`, shared with every other process's, held until the returned file is
    closed: while any is, what the pin guards is not evicted (`evictable`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pin = path.open("a+b")
    fcntl.flock(pin.fileno(), fcntl.LOCK_SH)
    return pin


@contextlib.contextmanager
def evictable(path: Path) -> Iterator[bool]:
    """Whether no reader holds the pin at `path` (`read_pin`), asked without waiting; when none does,
    the pin is held exclusively for the block, so what it guards is evicted before a reader can pin
    it again. A generation's retention and the transport cache each choose what to evict, by their
    own policy, through it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as pin:
        try:
            fcntl.flock(pin.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            yield True


class LocalRetention:
    def __init__(self, directory: Path, max_bytes: int):
        self.directory, self.max_bytes = directory, max_bytes

    @staticmethod
    def pin(directory: Path) -> BinaryIO:
        return read_pin(_pin_path(directory))

    @staticmethod
    def _exclusive(directory: Path) -> contextlib.AbstractContextManager[bool]:
        return evictable(_pin_path(directory))

    def _keys(self) -> list[Path]:
        return [
            path
            for path in self.directory.iterdir()
            if path.is_dir() and re.fullmatch(r"[a-f0-9]{64}", path.name)
        ]

    def _size(self) -> int:
        return sum(path.stat().st_size for key in self._keys() for path in key.rglob("*") if path.is_file())

    @property
    def _ledger(self) -> Path:
        return self.directory / ".retention.bytes"

    def _held(self) -> int:
        """The bytes the root holds, as the last reservation recorded them (`reserve`), else counted.
        Every write into the root reserves first and an eviction recounts, so the record is never
        less than the root holds; a write that failed or a file removed outside a reservation leaves
        it more, which the next reservation over the limit counts again."""
        try:
            return int(self._ledger.read_text())
        except (FileNotFoundError, ValueError):
            return self._size()

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

    @staticmethod
    def _abandoned(key: Path) -> None:
        """Remove `key`'s writers that stopped before they sealed and no process holds: a run that
        ends mid-fill closes its writer unsealed, and no generation reads an unsealed writer's
        shards (`generation._sealed`), so they hold bytes alone."""
        for writer in (key / "shards").glob("*"):
            if not writer.is_dir() or not re.fullmatch(r"[a-f0-9]{32}", writer.name):
                continue
            segments = key / "segments" / writer.name
            with LocalRetention._exclusive(writer) as available:
                if available and not (segments / "seal.json").is_file():
                    shutil.rmtree(writer)
                    if segments.is_dir():
                        shutil.rmtree(segments)

    @contextlib.contextmanager
    def reserve(self, target: Path, byte_count: int) -> Iterator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.directory / ".retention.lock")):
            replaced = target.stat().st_size if target.is_file() else 0
            needed = max(0, byte_count - replaced)
            # The root is walked only where the recorded bytes would pass the limit: once to count what
            # it holds, then as it evicts. Under the limit a write reads one small record.
            held = self._held()
            if held + needed > self.max_bytes:
                held = self._size()
            if held + needed <= self.max_bytes:
                self._ledger.write_text(str(held + needed))
                yield
                return
            for key in self._keys():
                self._abandoned(key)
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
            held = self._size()
            if held + needed > self.max_bytes:
                raise OSError(
                    f"local cache {self.directory} cannot admit {needed} bytes within "
                    f"keep_local_bytes={self.max_bytes}; its remaining generations or writers are pinned"
                )
            self._ledger.write_text(str(held + needed))
            yield

    @staticmethod
    def touched(directory: Path) -> None:
        manifest = directory / "manifest.json"
        if manifest.is_file():
            os.utime(manifest, None)


def _pin_path(directory: Path) -> Path:
    """Where a local generation's or writer's `directory` is pinned: beside it, hidden."""
    return directory.parent / f".{directory.name}.pin"
