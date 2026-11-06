"""Write-through artifacts with per-input completions and conditional publication."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Literal, TypedDict, overload

import grain.python as grain
import numpy as np

from dew._cache import dew_cache_dir
from dew.data.concurrent import Concurrently
from dew.data.dataset import _QUIET_STOP_SECONDS, Loading
from dew.data.declared import Declared
from dew.data.generation import (
    GENERATION_SCHEMA,
    DependencyManifest,
    Generation,
    Outputs,
    RowId,
    _checked,
    _decode_outputs,
    _hash,
    _json,
    _publish,
    _row_key,
    completion_facts,
    seal_facts,
)
from dew.data.once import LOCK, locked_cached_property
from dew.data.pipeline import Record, Rejected, Volatile
from dew.data.retention import LocalRetention
from dew.data.sinks import write_array_record
from dew.data.sources.base import Shard, ShardOpener, Source
from dew.data.sources.codec import Unencodable, encode
from dew.data.sources.records import ArrayRecords
from dew.data.storage import PreconditionFailed, Store, _byte_size

_log = logging.getLogger(__name__)


AUTOMATIC = "DEW_DATA_CACHE"
"""The environment variable that names the directory a process's runs cache their chains' leading
deterministic steps in (`automatic_root`), or turns that off (`off`)."""


def automatic_root(loading: Loading) -> str | None:
    """Where a run caches what its chains' leading deterministic steps make of their records, with no
    `cache=` (`dew.data.compile._automatic`): `$DEW_DATA_CACHE`, else `data` in Dew's cache directory
    (`dew._cache.dew_cache_dir`), a local root holding at most `Cache.keep_local_bytes`, its oldest
    generations evicted first (`LocalRetention`). None where the run's `loading` opts out
    (`Loading(cache=False)`) or the process does (`DEW_DATA_CACHE=off`)."""
    named = os.environ.get(AUTOMATIC)
    if not loading.cache or (named is not None and named.strip().lower() in ("off", "0", "false", "")):
        return None
    return named or os.path.join(dew_cache_dir(), "data")


def _opened_writer(cache: Cache, key: DependencyManifest, *, expected: int | None = None,
                   source: Source | None = None, shards: int | None = None,
                   halted: set[str] | None = None) -> _Writer | _Computed:
    """A writer of `key`'s fill, of `expected` rows from `source` where an indexed fill knows them, or of
    one of a stream's `shards`, or, where an optional cache's storage cannot hold one or its fill stopped
    on this reader already (`halted`), none: its records are computed (`Cache.read_on`)."""
    if cache.optional and halted is not None and key.key in halted:
        return _Computed()
    try:
        return _Writer(cache, key, expected=expected, source=source, shards=shards, halted=halted)
    except (OSError, sqlite3.Error) as error:
        if not cache.read_on(error, "writes nothing"):
            raise
        return _Computed()


class Abandoned(RuntimeError):
    """An upload or a seal the cache's writer was closed before, which it counts (`abandoned`)."""
type Transform = Callable[[int, Record], Outputs]
type CachedValue = Record | Outputs


class CacheStats(TypedDict):
    admitting: bool
    spool_bytes: int
    uploaded_bytes: int
    upload_seconds: float
    abandoned: int


def _transformed(ordinal: int, value: Record, transform: Transform | None) -> Outputs:
    outputs = (value,) if transform is None else transform(ordinal, value)
    if not isinstance(outputs, tuple):
        raise TypeError("cache input transform must return a tuple of outputs")
    return outputs


def _volatile(transform: Transform) -> bool:
    return isinstance(transform, Volatile) and transform.volatile


@dataclass(frozen=True)
class _Passthrough:
    transform: Transform

    def __call__(self, index: int, value: Record) -> Outputs:
        return _transformed(index, value, self.transform)


@dataclass(frozen=True)
class Cache:
    """Cache a deterministic prefix without changing a filling run's execution plan.

    A fill whose key a generation is published under (CURRENT) reads that
    generation's completions in place of running its steps, so a later run
    reuses an earlier one's complete fill. Otherwise each process keeps its
    own writes; it never discovers another filling writer's rows, and a
    writer that completes every input of the source it is given publishes
    that source's generation. Volatile prefixes run on every read and produce
    no completions. A shard a remote store took but reads back otherwise is
    read again up to `retry` more times (`Store.put_immutable`); a failed
    transfer raises after its transport's own retries, and a write to this
    host's disk or memory fails at once.
    """

    url: str
    spool_bytes: int | str = 256 * 2**20
    mirror_bytes: int | str = 8 * 2**30
    keep_local_bytes: int | str = 8 * 2**30
    on_full: Literal["stop", "block"] = "stop"
    upload_workers: int = 2
    retry: int = 3
    shard_bytes: int | str = 16 * 2**20
    storage_options: Mapping[str, object] | None = field(default=None, repr=False)
    optional: bool = False
    """Whether a run reads on without it where its storage fails (no room, no permission, its
    index's database full or read-only): a cache the compiler placed (`dew.data.compile._automatic`),
    which a run never needs, then looks nothing up, stops admitting and its records are computed,
    as they are with no cache (`read_on`). A cache a chain names stops the run."""
    evicts: bool = True
    """Whether a fill in a local root evicts its oldest generations to stay within `keep_local_bytes`
    (`LocalRetention`), or holds to the room the root has free and stops there: a stream's automatic
    cache, whose corpus may be larger than the root, so a fill that cannot complete takes no generation
    another read keeps (row 4217)."""

    def read_on(self, error: OSError | sqlite3.Error | ValueError | TypeError, doing: str) -> bool:
        """Whether this cache reads on uncached after its storage raised `error` while `doing`: it is
        `optional`, and logs that it does."""
        if not self.optional:
            return False
        _log.warning("cache %s %s: %s; the run reads on uncached", self.url, doing, error)
        return True

    def __post_init__(self) -> None:
        if min(self.spool_limit, _byte_size(self.mirror_bytes), self.local_limit, self.shard_limit) < 1:
            raise ValueError("cache byte budgets must be positive")
        if self.shard_limit > self.spool_limit or self.upload_workers < 1 or self.retry < 0:
            raise ValueError(
                "cache shard_bytes must fit the spool; upload_workers positive and retry nonnegative"
            )
        if self.on_full not in ("stop", "block"):
            raise ValueError("cache on_full is 'stop' or 'block'")

    @property
    def spool_limit(self) -> int:
        return _byte_size(self.spool_bytes)

    @property
    def shard_limit(self) -> int:
        return _byte_size(self.shard_bytes)

    @property
    def local_limit(self) -> int:
        return _byte_size(self.keep_local_bytes)

    def _mirror_options(self) -> Mapping[str, object]:
        options = dict(self.storage_options or {})
        configured = options.get("dewcache", {})
        if not isinstance(configured, Mapping):
            raise ValueError("dewcache storage_options must be a mapping")
        options["dewcache"] = {**configured, "max_bytes": _byte_size(self.mirror_bytes)}
        return options

    def reference(self, manifest: DependencyManifest) -> str:
        """The cache a plan reads, by the key of the rows it serves: where the cache lives is no part
        of the data, so a run resumes against the same rows cached anywhere."""
        return f"filling {manifest.key}"

    def _root(self, manifest: DependencyManifest) -> Store:
        if not manifest.clean:
            raise ValueError("cache writes need a clean Dew revision")
        if not manifest.exact:
            raise ValueError("cache prefix contains a step without a stable identity")
        base = Store(self.url, self.storage_options)
        fs, path, scheme = base._backend()
        if scheme == "file":
            base = Store(
                fs.unstrip_protocol(path), self.storage_options,
                LocalRetention(Path(path), self.local_limit, evicts=self.evicts),
            )
        return base.child(manifest.key)

    def mapped(
        self,
        parent: grain.MapDataset,
        key: DependencyManifest,
        *,
        transform: Transform | None = None,
        source: Source | None = None,
        missing: Sequence[int] = (),
    ) -> _Mapped:
        """The cached rows of `parent`, a source's records by index: a published
        generation's, or filled as they are read. A fill that completes every
        input publishes the generation of `source`, the source they are. The
        inputs `missing` that a read skips (a store's absent records) are written
        as a fill starts, so it can complete without reading them."""
        computed = _Mapped(parent if transform is None else parent.map_with_index(_Passthrough(transform)))
        if transform is not None and _volatile(transform):
            return computed
        try:
            self._root(key)
            published = self.generation(key)
        except (OSError, sqlite3.Error) as error:
            if not self.read_on(error, "looks up nothing"):
                raise
            return computed
        if published is not None:
            # An optional cache's replay computes a record its generation cannot give (`Cache.read_on`).
            replay = (_Published(published, transform, self, parent).at if self.optional
                      else _Published(published, transform))
            return _Mapped(grain.MapDataset.range(len(parent)).map(replay))
        return _CachedMap(parent, self, key, transform, source, tuple(missing))

    def streamed(
        self,
        reader: ShardOpener,
        key: DependencyManifest,
        *,
        transform: Transform | None = None,
        threads: int | None = None,
        source: Source | None = None,
        transparent: bool = False,
    ) -> ShardOpener:
        """`reader`'s shards read through the cache: a published generation's records read back, or each
        shard's fill written as it is read, `source`'s generation published by the last of its shards'
        fills to seal (row 4211). A `transparent` read saves each shard's position as its reader does,
        so a position saved with the cache resumes without it and the reverse (row 4217)."""
        computed = (reader if transform is None
                    else _PassthroughReader(reader, _Passthrough(transform), threads))
        if transform is not None and _volatile(transform):
            return computed
        try:
            self._root(key)
            published = self.generation(key)
        except (OSError, sqlite3.Error) as error:
            if not self.read_on(error, "looks up nothing"):
                raise
            if not transparent:
                return computed
            # A transparent read computes its records in the form it saves with the cache read.
            return _CachedReader(reader, self, key, transform, _Computed(), source, transparent=True)
        # The store's shards, counted once for every shard's writer (`_Writer._published`).
        shards = None if source is None else len(source.manifest().shards)
        return _CachedReader(reader, self, key, transform,
                             None if published is None
                             else _Published(published, transform, self if self.optional else None), source,
                             transparent, shards)

    def generation(self, key: DependencyManifest) -> Generation | None:
        root = self._root(key)
        try:
            current = root.child("CURRENT").get_versioned()
        except FileNotFoundError:
            _log.debug("cache generation has not been published at %s", root.url)
            return None
        record = json.loads(current.data)
        if record["manifest"] != key.key:
            raise ValueError(f"cache CURRENT manifest differs at {root.url}")
        # Pinned, then its manifest read once, checked against what CURRENT committed.
        return Generation.named(root, record, self._mirror_options())

    def publish(
        self,
        key: DependencyManifest,
        source: Source,
        *,
        plan: Callable[[Generation], Mapping[str, np.ndarray]] | None = None,
    ) -> Generation:
        """The generation of `source` under `key`, once its sealed fills cover
        every input: the one published already, unless `plan` adds a packing
        plan, which a new generation carries."""
        with ExitStack() as pins:
            published = _publish(self, key, source, plan, pins)
        assert published is not None
        return published


@dataclass(frozen=True)
class _Published:
    """The completions a generation published under a fill's key holds, read
    in place of running the fill's steps: a fill another run completed."""

    generation: Generation
    transform: Transform | None
    optional: Cache | None = None
    """The optional cache it replays (`Cache.read_on`), whose records are computed from their inputs
    where its generation cannot be read; None replays or refuses."""
    inputs: grain.MapDataset | None = None
    """The inputs by index a replay by index computes a record from where it reads on."""

    def __call__(self, row_id: RowId) -> CachedValue:
        outputs = self.generation.outputs(row_id)
        return outputs if self.transform is not None else outputs[0]

    def at(self, index: int) -> CachedValue:
        """The record at `index` of its inputs, as a replay by index reads it: computed from that
        input where an optional cache's generation cannot give it (`Cache.read_on`)."""
        try:
            return self(index)
        except (OSError, sqlite3.Error, ValueError) as error:
            if (self.inputs is None or self.optional is None
                    or not self.optional.read_on(error, "replays nothing")):
                raise
            outputs = _transformed(index, self.inputs[index], self.transform)
            return outputs if self.transform is not None else outputs[0]

    def process(self, row_id: RowId, ordinal: int, value: Record, transform: Transform | None) -> CachedValue:
        try:
            return self(row_id)
        except (OSError, sqlite3.Error, ValueError) as error:
            if self.optional is None or not self.optional.read_on(error, "replays nothing"):
                raise
            outputs = _transformed(ordinal, value, transform)
            return outputs if transform is not None else outputs[0]

    def finish(self) -> None:
        pass

    def close(self) -> None:
        pass

    def metrics(self) -> CacheStats:
        return _READING


class _Computed:
    """Rows computed and kept nowhere: the inputs of a key published under other IDs."""

    def process(self, row_id: RowId, ordinal: int, value: Record, transform: Transform | None) -> CachedValue:
        outputs = _transformed(ordinal, value, transform)
        return outputs if transform is not None else outputs[0]

    def finish(self) -> None:
        pass

    def close(self) -> None:
        pass

    def metrics(self) -> CacheStats:
        return _READING


_READING: CacheStats = {"admitting": False, "spool_bytes": 0, "uploaded_bytes": 0, "upload_seconds": 0.0,
                        "abandoned": 0}
"""What a read with no fill reports: a published generation's or a volatile prefix's."""


class _Writer:
    def __init__(self, cache: Cache, key: DependencyManifest, *, expected: int | None = None,
                 source: Source | None = None, shards: int | None = None, halted: set[str] | None = None):
        self.cache, self.root = cache, cache._root(key)
        self.key, self.source, self.shards, self.halted = key, source, shards, halted
        self.writer = uuid.uuid4().hex
        _, root_path, scheme = self.root._backend()
        self.pin = LocalRetention.pin(Path(root_path) / "shards" / self.writer) if scheme == "file" else None
        self.directory = tempfile.TemporaryDirectory(prefix="dew-spool-")
        self.pool = ThreadPoolExecutor(cache.upload_workers, thread_name_prefix="dew-cache-upload")
        self.lock = threading.Condition(threading.RLock())
        self.futures: list[Future] = []
        self.segments: list[dict] = []
        self.index = sqlite3.connect(Path(self.directory.name) / "inputs.sqlite", check_same_thread=False)
        self.index.execute(
            "CREATE TABLE inputs (id TEXT PRIMARY KEY, shard INTEGER, offset INTEGER, digest TEXT)"
        )
        self.pending: dict[str, bytes] = {}
        self.pending_bytes = 0
        self.spool_bytes = 0
        self.admitting = True
        self.dropped = False
        """Whether the fill was dropped where its storage or its codec failed an optional cache
        (`_stopped`), so it ends unsealed, unlike a full one's stop, which seals what it admitted."""
        self.closed = False
        self.closing = False
        self.uploaded_bytes = 0
        self.upload_seconds = 0.0
        self.abandoned = 0
        self.expected = expected
        self.seal_future: Future | None = None

    @cached_property
    def _records(self):
        @lru_cache(maxsize=2)
        def opened(shard: int) -> ArrayRecords:
            # Read once its upload completed, by the declaration its segment published, so Dew's mirror
            # holds the copy the publication checks (`dew.data.generation._sealed`), not another beside it.
            declaration = json.loads(_checked(self.root, self.futures[shard].result()))["shard"]
            # The mirror's own options, its budget among them, on the declared store as on the reader.
            store = Store(self.root.child(declaration["path"]).url, self.cache._mirror_options())
            return ArrayRecords((store.url,), store.storage_options, declared=(Declared(store, declaration),))

        return opened

    def process(self, row_id: RowId, ordinal: int, value: Record, transform: Transform | None) -> CachedValue:
        self._completed()
        row_key = _row_key(row_id)
        with self.lock:
            existing = None if self.closing else self.pending.get(row_key)
            completed = (
                None
                if self.closing
                else self.index.execute(
                    "SELECT shard, offset, digest FROM inputs WHERE id=?", (row_key,)
                ).fetchone()
            )
        if existing is not None:
            values = _decode_outputs(existing)
            return values if transform is not None else values[0]
        if completed is not None and self.futures[completed[0]].done():
            uploaded = self.futures[completed[0]]
            if uploaded.exception() is None:
                payload = self._records(completed[0])[completed[1]]
                if not isinstance(payload, bytes) or _hash(payload) != completed[2]:
                    raise ValueError(f"cache writer record is corrupt for row {row_key}")
                values = _decode_outputs(payload)
                return values if transform is not None else values[0]
        outputs = _transformed(ordinal, value, transform)
        if self.closing:
            return outputs if transform is not None else outputs[0]
        try:
            payload = encode(outputs)
        except Unencodable as error:
            # A record the codec cannot write: a required cache refuses it wherever its fill stands; an
            # optional one keeps none of the fill, as where its storage fails, and reads on computed.
            with self.lock:
                if not self.cache.optional or not self.dropped:
                    if not self.cache.read_on(error, "stopped admission"):
                        raise
                    self._stopped()
            return outputs if transform is not None else outputs[0]
        transient = any(isinstance(output, Rejected) and not output.permanent for output in outputs)
        with self.lock:
            if self.closing:
                return outputs if transform is not None else outputs[0]
            existing = self.pending.get(row_key)
            if existing is not None:
                if existing != payload:
                    raise ValueError(f"conflicting cache duplicate for row {row_key}")
                cached = _decode_outputs(existing)
                return cached if transform is not None else cached[0]
            completed = self.index.execute(
                "SELECT shard, offset, digest FROM inputs WHERE id=?", (row_key,)
            ).fetchone()
            if completed is not None:
                if _hash(payload) != completed[2]:
                    raise ValueError(f"conflicting cache duplicate for row {row_key}")
                return outputs if transform is not None else outputs[0]
            if not transient and self.admitting:
                try:
                    self._admit(row_key, payload)
                except (OSError, sqlite3.Error) as error:
                    # Storage that fails mid-fill: an optional cache stops admitting, as a full one
                    # does with on_full='stop', and reads on.
                    if not self.cache.read_on(error, "stopped admission"):
                        raise
                    self._stopped()
        return outputs if transform is not None else outputs[0]

    def _stopped(self) -> None:
        """Stop admitting for good, under the lock: what the writer held unwritten is dropped, and the
        fill is neither sealed nor published (`dropped`), what it wrote left for collection; an optional
        cache's later writers of the key on its reader write nothing (`_halt`)."""
        self._halt()
        self.dropped = True
        self.pending.clear()
        self.pending_bytes = 0

    def _halt(self) -> None:
        """Stop admitting; an optional cache's later shard writers on this reader write nothing, so a fill
        that cannot complete stops once, not at each shard (row 4296)."""
        self.admitting = False
        if self.cache.optional and self.halted is not None:
            self.halted.add(self.key.key)

    def _admit(self, row_key: str, payload: bytes) -> None:
        pages = self.index.execute("PRAGMA page_count").fetchone()[0]
        page_size = self.index.execute("PRAGMA page_size").fetchone()[0]
        if (pages + 2) * page_size > self.cache.local_limit:
            self._halt()
            _log.warning("cache %s stopped admission at its local index budget", self.root.url)
        if len(payload) > self.cache.spool_limit:
            if self.cache.on_full == "block":
                raise ValueError(f"cache row {row_key} exceeds spool_bytes={self.cache.spool_bytes}")
            self._halt()
        while self.admitting and self.spool_bytes + len(payload) > self.cache.spool_limit:
            self._flush()
            if self.cache.on_full == "stop":
                self._halt()
                break
            self.lock.wait_for(lambda: self._room(len(payload)))
        if self.admitting:
            self.pending[row_key] = payload
            self.pending_bytes += len(payload)
            self.spool_bytes += len(payload)
            if self.pending_bytes >= self.cache.shard_limit:
                self._flush()
            if self.expected is not None and self.index.total_changes + len(self.pending) == self.expected:
                self._flush()
                # A full spool can drop the last shard (on_full='stop'), which leaves the fill partial.
                whole = self.index.total_changes == self.expected
                self.seal_future = self.pool.submit(self.complete if whole else self.seal)

    def _flush(self) -> None:
        if not self.pending:
            return
        number = len(self.futures)
        path = Path(self.directory.name) / f"{number}.array_record"
        write_array_record(path, self.pending.values())
        entries = []
        positions = []
        used = self.pending_bytes
        for offset, (row_id, payload) in enumerate(self.pending.items()):
            entries.append({**completion_facts(row_id, payload), "record": offset})
            positions.append((row_id, number, offset, _hash(payload)))
        actual = path.stat().st_size
        if not self._reserve_shard(actual, used, path):
            self.pending.clear()
            self.pending_bytes = 0
            return
        self.index.executemany("INSERT INTO inputs VALUES (?, ?, ?, ?)", positions)
        self.index.commit()
        self.pending.clear()
        self.pending_bytes = 0
        upload = self.pool.submit(self._upload, number, path, entries, actual)
        upload.add_done_callback(self._ended)
        self.futures.append(upload)

    def _reserve_shard(self, actual: int, provisional: int, path: Path) -> bool:
        if actual > self.cache.spool_limit:
            path.unlink()
            self.spool_bytes -= provisional
            if self.cache.on_full == "block":
                raise ValueError(
                    f"closed cache shard requires {actual} bytes, exceeding "
                    f"spool_bytes={self.cache.spool_limit}"
                )
            self._halt()
            _log.warning("cache %s stopped admission: one shard exceeds its spool budget", self.root.url)
            return False
        while self.spool_bytes - provisional + actual > self.cache.spool_limit:
            if self.cache.on_full == "stop":
                path.unlink()
                self.spool_bytes -= provisional
                self._halt()
                _log.warning("cache %s stopped admission at its spool budget", self.root.url)
                return False
            self.lock.wait_for(lambda: self._room(actual - provisional))
        self.spool_bytes += actual - provisional
        return True

    def _room(self, needed: int) -> bool:
        """Whether the spool has room for `needed` more bytes. An upload that ended frees its bytes or
        failed, which raises here; each wakes the waiting writer (`_ended`)."""
        for future in self.futures:
            if future.done():
                future.result()
        return self.spool_bytes + needed <= self.cache.spool_limit

    def _ended(self, _: Future) -> None:
        with self.lock:
            self.lock.notify_all()

    def _upload(self, number: int, path: Path, entries: list[dict], used: int) -> dict:
        started = time.monotonic()
        payload = path.read_bytes()
        shard = f"shards/{self.writer}/{number:06d}.array_record"
        segment = f"segments/{self.writer}/{number:06d}.json"
        # A remote store's write that reads back otherwise is read again, by the store
        # (`Store.put_immutable`); a local disk's failure, or its retention refusing room
        # (`LocalRetention`), is the cache's error at once.
        retries = self.cache.retry if self.root.remote else 0
        if self.closed:
            raise Abandoned("cache upload was cancelled")
        self.root.child(shard).put_immutable(payload, retries=retries)
        encoded = _json(
            {
                "schema": GENERATION_SCHEMA,
                "shard": {"path": shard, "bytes": len(payload), "sha256": _hash(payload)},
                "completions": entries,
            }
        )
        if self.closed:
            raise Abandoned(f"cache upload abandoned before segment publication: {shard}")
        self.root.child(segment).put_immutable(encoded, retries=retries)
        with self.lock:
            self.spool_bytes -= used
            self.uploaded_bytes += used
            self.upload_seconds += time.monotonic() - started
        path.unlink(missing_ok=True)
        return {"path": segment, "bytes": len(encoded), "sha256": _hash(encoded)}

    def metrics(self) -> CacheStats:
        with self.lock:
            return {
                "admitting": self.admitting,
                "spool_bytes": self.spool_bytes,
                "uploaded_bytes": self.uploaded_bytes,
                "upload_seconds": self.upload_seconds,
                "abandoned": self.abandoned,
            }

    def seal(self) -> None:
        if self.dropped:
            # A dropped fill seals nothing; its uploads drain as the writer finishes.
            return
        with self.lock:
            self._flush()
        self.segments = [future.result() for future in self.futures]
        sealed = seal_facts(self.segments, (_checked(self.root, entry) for entry in self.segments))
        payload = _json(sealed)
        if self.closed:
            raise Abandoned(f"cache writer was abandoned before its seal: {self.writer}")
        try:
            self.root.child(f"segments/{self.writer}/seal.json").put_if_absent(payload)
        except PreconditionFailed:
            if self.root.child(f"segments/{self.writer}/seal.json").read() != payload:
                raise

    def complete(self) -> None:
        """Seal a fill holding every input, and publish the generation of its
        source where it was given one, which keeps a generation published
        under its key (`Cache.publish`). Another writer that completed the
        same fill may publish it first, which leaves it published."""
        self.seal()
        if self.source is None or self.dropped:
            return
        try:
            self.cache.publish(self.key, self.source)
        except PreconditionFailed:
            if self.cache.generation(self.key) is None:
                raise

    def _settled(self, work: Callable[[], None]) -> None:
        """Run `work` (a seal, or reading its outcome): an upload that fails
        leaves the fill filling, unless the cache blocks on a full spool."""
        try:
            work()
        except OSError as error:
            if self.cache.on_full == "block":
                raise
            # A root that refused room refuses the next shard too: the fill stops once (row 4296).
            with self.lock:
                self._halt()
            _log.warning("cache %s stops filling after upload failure: %s", self.root.url, error)
        except sqlite3.Error as error:
            if not self.cache.read_on(error, "remains filling"):
                raise

    def _published(self) -> None:
        """Publish the stream's generation once as many writers have sealed as its store has shards: each
        shard's writer seals as it reads the shard whole, so the publication's check of every sealed
        record runs once a stream ends, not at each of its shards' ends (row 4211)."""
        assert self.source is not None and self.shards is not None
        fs, path = self.root.child("segments").resolve()
        if len(fs.glob(f"{path}/*/seal.json")) < self.shards:
            return
        try:
            with ExitStack() as pins:
                _publish(self.cache, self.key, self.source, None, pins, of_several=True)
        except PreconditionFailed:
            if self.cache.generation(self.key) is None:
                raise

    def _completed(self) -> None:
        """The outcome of a fill's completion (`complete`) once it ends,
        observed once, so a reader that never runs out sees it too."""
        with self.lock:
            completion = self.seal_future
            # A completion a closing reader abandoned was cancelled, and counted (`abandoned`).
            if completion is None or not completion.done() or completion.cancelled():
                return
            self.seal_future = None
        self._settled(completion.result)

    def finish(self) -> None:
        with self.lock:
            if self.closing:
                return
            self.closing = True
        try:
            self._settled(self.seal)
            if self.seal_future is not None:
                self._settled(self.seal_future.result)
            if self.shards is not None and not self.dropped:
                # A stream's shard read whole: its source's generation published once every shard's
                # fill has sealed, by whichever seals last (row 4211).
                self._settled(self._published)
        finally:
            self.closed = True
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.index.close()
            self._release()
            self._reported()

    def _reported(self) -> None:
        """Log the fill's report as it ends, in the process that filled it, a loading worker's too (row
        4121): whether it still admitted records, its spool, the bytes it uploaded and at what rate, and
        the uploads it abandoned."""
        held = self.metrics()
        seconds = held["upload_seconds"]
        rate = held["uploaded_bytes"] / seconds / 2**20 if seconds else 0.0
        # A fill that abandoned uploads warns, which a run's default logging shows.
        _log.log(logging.WARNING if held["abandoned"] else logging.INFO,
                 "data/cache %s: admitting=%s spool_bytes=%d uploaded_bytes=%d (%.1f MiB/s) abandoned=%d",
                 self.root.url, held["admitting"], held["spool_bytes"], held["uploaded_bytes"], rate,
                 held["abandoned"])

    def _release(self) -> None:
        """The writer's local spool and its pin let go, once no upload reads them: the one release
        every way a writer ends takes, whichever drains or abandons its uploads first."""
        self.directory.cleanup()
        if self.pin is not None:
            self.pin.close()

    def close(self) -> None:
        with self.lock:
            if self.closing:
                return
            self.closing = True
            pending = len(self.pending)
            self.pending.clear()
            self.spool_bytes -= self.pending_bytes
            self.pending_bytes = 0
        tasks = [*self.futures, *([self.seal_future] if self.seal_future is not None else [])]
        _, unfinished = wait(tasks, timeout=_QUIET_STOP_SECONDS) if tasks else (set(), set())
        with self.lock:
            self.closed = True
            self.abandoned += pending + len(unfinished)
        for future in unfinished:
            future.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.index.close()
        self._reported()
        if unfinished:
            remaining = set(unfinished)

            def released(future: Future) -> None:
                with self.lock:
                    remaining.discard(future)
                    if not remaining:
                        self._release()

            for future in unfinished:
                future.add_done_callback(released)
        else:
            self._release()
        self._completed()


class _Mapped(grain.MapDataset[CachedValue]):
    """An indexed cache read, with no fill for a volatile prefix."""

    def __init__(self, parent: grain.MapDataset):
        super().__init__(parent)
        self.input_parent = parent

    def __len__(self) -> int:
        return len(self.input_parent)

    @overload
    def __getitem__(self, index: slice) -> grain.MapDataset[CachedValue]: ...

    @overload
    def __getitem__(self, index: int) -> CachedValue: ...

    def __getitem__(self, index):
        return self.slice(index) if isinstance(index, slice) else self.record(index)

    def record(self, index: int) -> CachedValue:
        return self.input_parent[index]

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def finish(self) -> None:
        pass

    def metrics(self) -> CacheStats:
        return _READING


class _CachedMap(_Mapped):
    def __init__(
        self,
        parent: grain.MapDataset,
        cache: Cache,
        key: DependencyManifest,
        transform: Transform | None,
        source: Source | None,
        missing: tuple[int, ...] = (),
    ):
        super().__init__(parent)
        # Not `source`, which names MapDataset's own constructor.
        self.cache, self.key, self.transform, self.filled = cache, key, transform, source
        self.missing = missing

    @locked_cached_property
    def writer(self) -> _Writer | _Computed:
        writer = _opened_writer(self.cache, self.key, expected=len(self), source=self.filled)
        if isinstance(writer, _Writer):
            finalizer = weakref.finalize(self, writer.close)
            finalizer.atexit = False
        return writer

    def record(self, index: int) -> CachedValue:
        return self.writer.process(
            index % len(self), index % len(self), self.input_parent[index], self.transform
        )

    def start(self) -> None:
        for position in self.missing:
            self.record(position)

    def close(self) -> None:
        if "writer" in self.__dict__:
            self.writer.close()

    def finish(self) -> None:
        if "writer" in self.__dict__:
            self.writer.finish()

    def metrics(self) -> CacheStats:
        return self.writer.metrics()

    def __getstate__(self) -> dict:
        return {key: value for key, value in self.__dict__.items() if key not in ("writer", LOCK)}


@dataclass(frozen=True)
class _Indexed:
    """A record beside its place in its shard, as a passthrough transform reads it."""

    def __call__(self, index: int, value: Record) -> tuple[int, Record]:
        return index, value


@dataclass(frozen=True)
class _Applied:
    """A transform over a record beside its place (`_Indexed`)."""

    transform: _Passthrough

    def __call__(self, placed: tuple[int, Record]) -> Outputs:
        return self.transform(*placed)


@dataclass(frozen=True)
class _PassthroughReader:
    reader: ShardOpener
    transform: _Passthrough
    threads: int | None = None

    def __call__(self, shard: Shard, options: grain.ReadOptions | None = None) -> grain.IterDataset:
        if self.threads is None:
            return self.reader(shard, options).map_with_index(self.transform)
        # A fetch with a pool of its own runs its records at once, in their order.
        indexed = self.reader(shard, options).map_with_index(_Indexed())
        return Concurrently(indexed, _Applied(self.transform), self.threads)


@dataclass(frozen=True)
class _CachedReader:
    reader: ShardOpener
    cache: Cache
    key: DependencyManifest
    transform: Transform | None
    published: _Published | _Computed | None
    source: Source | None = None
    transparent: bool = False
    shards: int | None = None
    halted: set[str] = field(default_factory=set, compare=False)
    """The keys whose fill stopped on this reader in this process (`_Writer._halt`)."""

    def __call__(self, shard: Shard, options: grain.ReadOptions | None = None) -> grain.IterDataset:
        return _CachedIter(self.reader(shard, options), self.cache, self.key, shard, self.transform,
                           self.published, self.source, self.transparent, self.shards, self.halted)


class _CachedIter(grain.IterDataset):
    def __init__(
        self,
        parent: grain.IterDataset,
        cache: Cache,
        key: DependencyManifest,
        shard: Shard,
        transform: Transform | None,
        published: _Published | _Computed | None,
        source: Source | None,
        transparent: bool,
        shards: int | None,
        halted: set[str],
    ):
        super().__init__(parent)
        self.cache, self.key, self.shard, self.transform = cache, key, shard, transform
        self.published, self.input_parent, self.source = published, parent, source
        self.transparent, self.shards, self.halted = transparent, shards, halted

    def __iter__(self) -> grain.DatasetIterator:
        return _CachedIterator(self.input_parent.__iter__(), self.writer, self.shard, self.transform,
                               self.transparent)

    def writer(self) -> _Writer | _Published | _Computed:
        """Where its shard's records are written or read. A shard keeps its iterator's state either
        way, so a fill resumes onto its published generation. A shard the generation holds under
        another ID (another mount, a stream cut into more shards than it has) runs its steps and
        writes nothing, so the key holds no rows under its IDs."""
        published = self.published
        if published is None:
            return _opened_writer(self.cache, self.key, source=self.source, shards=self.shards,
                                  halted=self.halted)
        if isinstance(published, _Computed) or published.generation.holds((self.shard.id, 0)):
            return published
        return _Computed()


class _CachedIterator(grain.DatasetIterator):
    def __init__(
        self,
        parent: grain.DatasetIterator,
        opens: Callable[[], _Writer | _Published | _Computed],
        shard: Shard,
        transform: Transform | None,
        transparent: bool = False,
    ):
        super().__init__(parent)
        self.parent, self.opens, self.shard, self.transform = parent, opens, shard, transform
        self.transparent = transparent
        self.writer: _Writer | _Published | _Computed | None = None
        self.ordinal: int | None = 0
        """Its records' place in the shard, the key of each; None once a transparent read resumed, whose
        saved position holds no place, so the rest of its shard is computed and kept nowhere."""

    def __next__(self):
        # The writer opens with the first record, so building the iterator reads no storage: Grain
        # builds a worker pool's iterators in this process for their first state, inside the
        # construction of a thread prefetch over the pool, which a refusal raised there leaves half
        # built for the collector to close (grain 0.2.18, ThreadPrefetchDatasetIterator.__init__).
        if self.writer is None:
            self.writer = self.opens() if self.ordinal is not None else _Computed()
        try:
            value = next(self.parent)
        except StopIteration:
            self.writer.finish()
            raise
        ordinal = 0 if self.ordinal is None else self.ordinal
        value = self.writer.process((self.shard.id, ordinal), ordinal, value, self.transform)
        if self.ordinal is not None:
            self.ordinal += 1
        return value

    def get_state(self) -> Mapping[str, object]:
        if self.transparent:
            return self.parent.get_state()
        return {"parent": self.parent.get_state(), "ordinal": self.ordinal}

    def set_state(self, state: Mapping[str, object]) -> None:
        if self.transparent:
            self.parent.set_state(state)
            self.ordinal = None
            return
        ordinal = state["ordinal"]
        parent = state["parent"]
        if not isinstance(ordinal, int) or not isinstance(parent, Mapping):
            raise ValueError("invalid cached shard iterator state")
        self.parent.set_state(parent)
        self.ordinal = ordinal

    def close(self) -> None:
        self.parent.close()
        if self.writer is not None:
            self.writer.close()
