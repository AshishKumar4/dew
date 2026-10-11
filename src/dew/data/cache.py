"""Write-through artifacts with per-input completions and conditional publication."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import sqlite3
import tempfile
import threading
import time
import uuid
import weakref
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, closing
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Literal, Protocol, overload, runtime_checkable

import grain.python as grain
import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter

from dew.data.dataset import Batch
from dew.data.pipeline import Record, Rejected
from dew.data.retention import LocalRetention
from dew.data.sources.base import Shard, ShardManifest, Source
from dew.data.sources.codec import decode, encode
from dew.data.sources.records import ArrayRecords
from dew.data.storage import PreconditionFailed, Store, _byte_size
from dew.records import JSON

_log = logging.getLogger(__name__)
type Outputs = tuple[Record, ...]
type Transform = Callable[[int, Record], Outputs]
type CachedValue = Record | Outputs
type RowId = int | tuple[str, int]


class DependencyManifest(Protocol):
    @property
    def key(self) -> str: ...
    @property
    def canonical(self) -> bytes: ...
    @property
    def clean(self) -> bool: ...
    @property
    def exact(self) -> bool: ...


def _json(value: JSON | Mapping[str, object]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _row_key(value: RowId) -> str:
    return _json(list(value) if isinstance(value, tuple) else value).decode()


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _checked(root: Store, entry: Mapping[str, object]) -> bytes:
    path = entry["path"]
    if not isinstance(path, str):
        raise ValueError("artifact object path is not a string")
    payload = root.child(path).read()
    if len(payload) != entry["bytes"] or _hash(payload) != entry["sha256"]:
        raise OSError(f"committed object is corrupt: {root.child(path).url}")
    return payload


def _outputs_digest(outputs: Sequence[Record]) -> str:
    digest = hashlib.sha256()
    for value in outputs:
        payload = encode(value)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _accepted(outputs: Sequence[Record]) -> Outputs:
    return tuple(value for value in outputs if value is not None and not isinstance(value, Rejected))


@runtime_checkable
class _Volatile(Protocol):
    volatile: bool


def _decode_outputs(encoded: bytes) -> Outputs:
    values = decode(encoded)
    if not isinstance(values, tuple):
        raise ValueError("cache record is not an input completion")
    outputs: list[Record] = []
    for value in values:
        if value is not None and not isinstance(value, (Mapping, bytes, Rejected)):
            raise ValueError("cache output is not a named record, bytes or rejection")
        outputs.append(value)
    return tuple(outputs)


def _fact(value: Batch | bytes | Rejected | None, name: str) -> int | None:
    if not isinstance(value, Mapping) or name not in value:
        return None
    stated = value[name]
    if not isinstance(stated, (int, np.integer)) or stated < 0:
        raise ValueError(f"cache record metadata {name} must be a nonnegative scalar integer")
    return int(stated)


@dataclass(frozen=True)
class Cache:
    """Cache a deterministic prefix without changing a filling run's execution plan.

    CURRENT is read only by an explicit generation open. Each process keeps
    its own writes; it never discovers another filling writer's rows.
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
        return f"filling {self.url.rstrip('/')}/{manifest.key}"

    def _root(self, manifest: DependencyManifest) -> Store:
        if not manifest.clean:
            raise ValueError("cache writes need a clean Dew revision")
        if not manifest.exact:
            raise ValueError("cache prefix contains a step without a stable identity")
        base = Store(self.url, self.storage_options)
        fs, path, scheme = base._backend()
        if scheme == "file":
            base = Store(
                fs.unstrip_protocol(path), self.storage_options, LocalRetention(Path(path), self.local_limit)
            )
        return base.child(manifest.key)

    def mapped(
        self,
        parent: grain.MapDataset,
        key: DependencyManifest,
        source: Source,
        *,
        transform: Transform | None = None,
    ) -> grain.MapDataset:
        self._root(key)
        return _CachedMap(parent, self, key, source, transform)

    def streamed(
        self,
        reader: Callable[[Shard], grain.IterDataset],
        key: DependencyManifest,
        manifest: ShardManifest,
        *,
        transform: Transform | None = None,
    ) -> Callable[[Shard], grain.IterDataset]:
        self._root(key)
        return _CachedReader(reader, self, key, transform)

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
        _checked(
            root,
            {
                "path": f"generations/{record['generation']}/manifest.json",
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            },
        )
        return Generation(root.child(f"generations/{record['generation']}").url, self._mirror_options())

    def publish(
        self,
        key: DependencyManifest,
        source: Source,
        *,
        plan: Callable[[Generation], Mapping[str, np.ndarray]] | None = None,
    ) -> Generation:
        with ExitStack() as pins:
            return self._publish(key, source, plan, pins)

    def _publish(
        self,
        key: DependencyManifest,
        source: Source,
        plan: Callable[[Generation], Mapping[str, np.ndarray]] | None,
        pins: ExitStack,
    ) -> Generation:
        root = self._root(key)
        try:
            current = root.child("CURRENT").get_versioned()
        except FileNotFoundError:
            current = None
        completions, shards = _sealed(root, pins)
        expected = tuple(_source_ids(source, streamed=any(row.startswith("[") for row in completions)))
        expected_keys = {_row_key(row) for row in expected}
        missing, extra = expected_keys - completions.keys(), completions.keys() - expected_keys
        if missing or extra:
            sample = next(iter(missing or extra))
            raise ValueError(
                f"cache coverage is incomplete: {len(missing)} missing, {len(extra)} unexpected "
                f"source IDs; example {sample}"
            )
        generation = uuid.uuid4().hex
        target = root.child(f"generations/{generation}")
        opened = Generation(target.url, self._mirror_options())
        _ = opened.pin
        pages: dict[str, dict] = {}
        dense = []
        for row in expected:
            row_key = _row_key(row)
            entry = completions[row_key]
            page = hashlib.sha256(row_key.encode()).hexdigest()[:2]
            pages.setdefault(page, {})[row_key] = entry
            dense.extend(
                {
                    **entry,
                    "output": index,
                    "length": entry["lengths"][index],
                    "valid_targets": entry["valid_targets"][index],
                }
                for index in range(entry["fanout"])
            )
        indexes = {}
        for page, values in pages.items():
            payload = _json(values)
            path = f"index/inputs-{page}.json"
            target.child(path).verified(payload)
            indexes[page] = {"path": path, "bytes": len(payload), "sha256": _hash(payload)}
        row_pages = []
        for offset in range(0, len(dense), 4096):
            payload = _json(dense[offset : offset + 4096])
            path = f"index/rows-{len(row_pages):06d}.json"
            target.child(path).verified(payload)
            row_pages.append({"path": path, "bytes": len(payload), "sha256": _hash(payload)})
        metadata = {}
        lengths = [entry["length"] for entry in dense]
        if all(length is not None for length in lengths):
            encoded = io.BytesIO()
            np.save(encoded, np.asarray(lengths, dtype=np.int64), allow_pickle=False)
            path = "index/lengths.npy"
            metadata["lengths"] = {"path": path, **target.child(path).verified(encoded.getvalue())}
        targets = [entry["valid_targets"] for entry in dense]
        if all(count is not None for count in targets):
            metadata["valid_targets"] = sum(targets)
            metadata["zero_target_windows"] = targets.count(0)
        source_shards = Counter(row[0] for row in expected if isinstance(row, tuple))
        referenced = {entry["shard"] for entry in completions.values()}
        definition = {
            "schema": 1,
            "manifest": key.key,
            "dependency": json.loads(key.canonical),
            "root": root.url,
            "rows": len(dense),
            "inputs": len(expected),
            "pages": indexes,
            "row_pages": row_pages,
            "shards": {path: shards[path] for path in referenced},
            "generation": generation,
            "metadata": metadata,
            "plan": {},
            "source_shards": dict(source_shards),
        }
        if plan is not None:
            candidate = Generation(target.url, self._mirror_options(), definition)
            for name, array in plan(candidate).items():
                if not name.replace("_", "").isalnum():
                    raise ValueError(f"invalid packing-plan field {name!r}")
                encoded = io.BytesIO()
                np.save(encoded, array, allow_pickle=False)
                path = f"plan/{name}.npy"
                definition["plan"][name] = {"path": path, **target.child(path).verified(encoded.getvalue())}
        payload = _json(definition)
        target.child("manifest.json").verified(payload)
        reference = _json(
            {"generation": generation, "manifest": key.key, "sha256": _hash(payload), "bytes": len(payload)}
        )
        if current is None:
            root.child("CURRENT").put_if_absent(reference)
        else:
            root.child("CURRENT").put_if_match(reference, current.version)
        return opened


def _sealed(root: Store, pins: ExitStack) -> tuple[dict, dict]:
    fs, path = root.child("segments").resolve()
    segment_paths = sorted(fs.find(path)) if fs.exists(path) else []
    seals = [location for location in segment_paths if location.endswith("/seal.json")]
    completions = {}
    shards = {}
    _, directory, scheme = root._backend()
    for location in seals:
        if scheme == "file":
            pins.enter_context(LocalRetention.pin(Path(directory) / "shards" / Path(location).parent.name))
        seal = json.loads(fs.cat_file(location))
        chain = b""
        collected = []
        for segment in seal["segments"]:
            payload = _checked(root, segment)
            chain = hashlib.sha256(chain + payload).digest()
            record = json.loads(payload)
            _checked(root, record["shard"])
            with closing(
                ArrayRecords((root.child(record["shard"]["path"]).url,), root.storage_options)
            ) as rows:
                if sorted(entry["record"] for entry in record["completions"]) != list(range(len(rows))):
                    raise ValueError(
                        f"cache shard has records without one completion: {record['shard']['path']}"
                    )
                for entry in record["completions"]:
                    _completion(entry, rows[entry["record"]])
                    candidate = {**entry, "shard": record["shard"]["path"]}
                    existing = completions.get(entry["id"])
                    if existing is not None and existing["payload_sha256"] != entry["payload_sha256"]:
                        if not record["volatile"]:
                            raise ValueError(f"conflicting cache duplicate for row {entry['id']}")
                        _log.warning(
                            "volatile cache conflict for row %s; keeping the first sealed copy", entry["id"]
                        )
                    completions.setdefault(entry["id"], candidate)
                    collected.append(entry["id"])
            shards[record["shard"]["path"]] = record["shard"]
        if chain.hex() != seal["sha256"] or len(seal["segments"]) != seal["count"]:
            raise ValueError(f"cache writer seal failed verification: {location}")
        if collected and (collected[0] != seal["first"] or collected[-1] != seal["last"]):
            raise ValueError(f"cache writer row range differs: {location}")
    return completions, shards


def _completion(entry: Mapping[str, object], encoded: Batch | bytes | None) -> None:
    if not isinstance(encoded, bytes):
        raise ValueError(f"cache shard record is not bytes for {entry['id']}")
    values = _decode_outputs(encoded)
    accepted = _accepted(values)
    if len(accepted) != entry["fanout"]:
        raise ValueError(f"cache completion fanout differs for {entry['id']}")
    if _outputs_digest(values) != entry["payload_sha256"]:
        raise ValueError(f"cache completion checksum differs for {entry['id']}")
    if _outputs_digest(accepted) != entry["sha256"]:
        raise ValueError(f"cache accepted-output checksum differs for {entry['id']}")
    if entry["outputs"] != [[entry["id"], index] for index in range(len(accepted))]:
        raise ValueError(f"cache completion output IDs differ for {entry['id']}")
    if entry["lengths"] != [_fact(value, "_dew_length") for value in accepted] or entry["valid_targets"] != [
        _fact(value, "_dew_valid_targets") for value in accepted
    ]:
        raise ValueError(f"cache completion metadata differs for {entry['id']}")


def _source_ids(source: Source, *, streamed: bool = False) -> Iterator[RowId]:
    if not streamed and source.supports("indexed"):
        yield from range(len(source.records("indexed")))
        return
    manifest = source.manifest()
    for shard in manifest.shards:
        if shard.rows is not None:
            yield from ((shard.id, index) for index in range(shard.rows))
        else:
            reader = iter(manifest.reader(shard))
            try:
                for index, _ in enumerate(reader):
                    yield shard.id, index
            finally:
                reader.close()


class _Writer:
    def __init__(
        self, cache: Cache, key: DependencyManifest, *, volatile: bool = False, expected: int | None = None
    ):
        self.cache, self.root = cache, cache._root(key)
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
        self.closed = False
        self.volatile = volatile
        self.expected = expected
        self.seal_future: Future | None = None

    @cached_property
    def _records(self):
        @lru_cache(maxsize=2)
        def opened(shard: int) -> ArrayRecords:
            return ArrayRecords(
                (self.root.child(f"shards/{self.writer}/{shard:06d}.array_record").url,),
                self.cache._mirror_options(),
            )

        return opened

    def process(self, row_id: RowId, ordinal: int, value: Record, transform: Transform | None) -> CachedValue:
        row_key = _row_key(row_id)
        with self.lock:
            existing = self.pending.get(row_key)
            completed = self.index.execute(
                "SELECT shard, offset, digest FROM inputs WHERE id=?", (row_key,)
            ).fetchone()
        if existing is not None:
            values = _decode_outputs(existing)
            if not isinstance(values, tuple):
                raise ValueError(f"cache pending row {row_key} is not an input completion")
            return values if transform is not None else values[0]
        if completed is not None and self.futures[completed[0]].done():
            uploaded = self.futures[completed[0]]
            if uploaded.exception() is None:
                payload = self._records(completed[0])[completed[1]]
                if not isinstance(payload, bytes) or _hash(payload) != completed[2]:
                    raise ValueError(f"cache writer record is corrupt for row {row_key}")
                values = _decode_outputs(payload)
                if not isinstance(values, tuple):
                    raise ValueError(f"cache writer record is not a completion for row {row_key}")
                return values if transform is not None else values[0]
        outputs = (value,) if transform is None else transform(ordinal, value)
        if not isinstance(outputs, tuple):
            raise TypeError("cache input transform must return a tuple of outputs")
        payload = encode(outputs)
        transient = any(isinstance(output, Rejected) and not output.permanent for output in outputs)
        with self.lock:
            existing = self.pending.get(row_key)
            if existing is not None:
                if existing != payload:
                    raise ValueError(f"conflicting cache duplicate for row {row_key}")
                return outputs if transform is not None else outputs[0]
            completed = self.index.execute(
                "SELECT shard, offset, digest FROM inputs WHERE id=?", (row_key,)
            ).fetchone()
            if completed is not None:
                if _hash(payload) != completed[2]:
                    if not self.volatile:
                        raise ValueError(f"conflicting cache duplicate for row {row_key}")
                    _log.warning("volatile cache conflict for row %s; keeping the first copy", row_key)
                return outputs if transform is not None else outputs[0]
            if not transient and self.admitting:
                pages = self.index.execute("PRAGMA page_count").fetchone()[0]
                page_size = self.index.execute("PRAGMA page_size").fetchone()[0]
                if (pages + 2) * page_size > self.cache.local_limit:
                    self.admitting = False
                    _log.warning("cache %s stopped admission at its local index budget", self.root.url)
                if len(payload) > self.cache.spool_limit:
                    if self.cache.on_full == "block":
                        raise ValueError(f"cache row {row_key} exceeds spool_bytes={self.cache.spool_bytes}")
                    self.admitting = False
                while self.admitting and self.spool_bytes + len(payload) > self.cache.spool_limit:
                    self._flush()
                    if self.cache.on_full == "stop":
                        self.admitting = False
                        break
                    self.lock.wait(0.1)
                    for future in self.futures:
                        if future.done():
                            future.result()
                if self.admitting:
                    self.pending[row_key] = payload
                    self.pending_bytes += len(payload)
                    self.spool_bytes += len(payload)
                    if self.pending_bytes >= self.cache.shard_limit:
                        self._flush()
                    if (
                        self.expected is not None
                        and self.index.total_changes + len(self.pending) == self.expected
                    ):
                        self._flush()
                        self.seal_future = self.pool.submit(self.seal)
        return outputs if transform is not None else outputs[0]

    def _flush(self) -> None:
        if not self.pending:
            return
        number = len(self.futures)
        path = Path(self.directory.name) / f"{number}.array_record"
        writer = ArrayRecordWriter(str(path), "group_size:1")
        entries = []
        positions = []
        used = self.pending_bytes
        for offset, (row_id, payload) in enumerate(self.pending.items()):
            outputs = _decode_outputs(payload)
            if not isinstance(outputs, tuple):
                raise ValueError(f"cache pending row {row_id} is not an input completion")
            writer.write(payload)
            accepted = _accepted(outputs)
            entries.append(
                {
                    "id": row_id,
                    "fanout": len(accepted),
                    "record": offset,
                    "outputs": [[row_id, index] for index in range(len(accepted))],
                    "sha256": _outputs_digest(accepted),
                    "payload_sha256": _outputs_digest(outputs),
                    "lengths": [_fact(value, "_dew_length") for value in accepted],
                    "valid_targets": [_fact(value, "_dew_valid_targets") for value in accepted],
                    "rejections": [value.reason for value in outputs if isinstance(value, Rejected)],
                }
            )
            positions.append((row_id, number, offset, _hash(payload)))
        writer.close()
        actual = path.stat().st_size
        if not self._reserve_shard(actual, used, path):
            self.pending.clear()
            self.pending_bytes = 0
            return
        self.index.executemany("INSERT INTO inputs VALUES (?, ?, ?, ?)", positions)
        self.index.commit()
        self.pending.clear()
        self.pending_bytes = 0
        self.futures.append(self.pool.submit(self._upload, number, path, entries, actual))

    def _reserve_shard(self, actual: int, provisional: int, path: Path) -> bool:
        if actual > self.cache.spool_limit:
            path.unlink()
            self.spool_bytes -= provisional
            if self.cache.on_full == "block":
                raise ValueError(
                    f"closed cache shard requires {actual} bytes, exceeding "
                    f"spool_bytes={self.cache.spool_limit}"
                )
            self.admitting = False
            _log.warning("cache %s stopped admission: one shard exceeds its spool budget", self.root.url)
            return False
        while self.spool_bytes - provisional + actual > self.cache.spool_limit:
            if self.cache.on_full == "stop":
                path.unlink()
                self.spool_bytes -= provisional
                self.admitting = False
                _log.warning("cache %s stopped admission at its spool budget", self.root.url)
                return False
            self.lock.wait(0.1)
            for future in self.futures:
                if future.done():
                    future.result()
        self.spool_bytes += actual - provisional
        return True

    def _upload(self, number: int, path: Path, entries: list[dict], used: int) -> dict:
        payload = path.read_bytes()
        shard = f"shards/{self.writer}/{number:06d}.array_record"
        segment = f"segments/{self.writer}/{number:06d}.json"
        for attempt in range(self.cache.retry + 1):
            if self.closed:
                raise RuntimeError("cache upload was cancelled")
            try:
                try:
                    verified = self.root.child(shard).verified(payload)
                except PreconditionFailed:
                    existing = self.root.child(shard).read()
                    if existing != payload:
                        raise ValueError(f"cache retry found conflicting shard {shard}") from None
                    verified = {"bytes": len(payload), "sha256": _hash(payload)}
                encoded = _json(
                    {
                        "schema": 1,
                        "shard": {"path": shard, **verified},
                        "completions": entries,
                        "volatile": self.volatile,
                    }
                )
                try:
                    self.root.child(segment).verified(encoded)
                except PreconditionFailed:
                    if self.root.child(segment).read() != encoded:
                        raise ValueError(f"cache retry found conflicting segment {segment}") from None
                with self.lock:
                    self.spool_bytes -= used
                    self.lock.notify_all()
                path.unlink()
                return {"path": segment, "bytes": len(encoded), "sha256": _hash(encoded)}
            except OSError:
                if attempt == self.cache.retry:
                    raise
                time.sleep(min(0.25 * 2**attempt, 10))
        raise RuntimeError("cache upload retry budget exhausted")

    def seal(self) -> None:
        with self.lock:
            self._flush()
        self.segments = [future.result() for future in self.futures]
        chain = b""
        rows = []
        for entry in self.segments:
            payload = _checked(self.root, entry)
            chain = hashlib.sha256(chain + payload).digest()
            rows.extend(completion["id"] for completion in json.loads(payload)["completions"])
        payload = _json(
            {
                "count": len(self.segments),
                "segments": self.segments,
                "first": rows[0] if rows else None,
                "last": rows[-1] if rows else None,
                "sha256": chain.hex(),
            }
        )
        try:
            self.root.child(f"segments/{self.writer}/seal.json").put_if_absent(payload)
        except PreconditionFailed:
            if self.root.child(f"segments/{self.writer}/seal.json").read() != payload:
                raise

    def close(self) -> None:
        if self.closed:
            return
        self.seal()
        if self.seal_future is not None:
            self.seal_future.result()
        self.closed = True
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.index.close()
        self.directory.cleanup()
        if self.pin is not None:
            self.pin.close()


class _CachedMap(grain.MapDataset[CachedValue]):
    def __init__(
        self,
        parent: grain.MapDataset,
        cache: Cache,
        key: DependencyManifest,
        source: Source,
        transform: Transform | None,
    ):
        super().__init__(parent)
        self.cache, self.key, self.input_source, self.transform = cache, key, source, transform
        self.input_parent = parent

    @cached_property
    def writer(self) -> _Writer:
        writer = _Writer(
            self.cache,
            self.key,
            volatile=isinstance(self.transform, _Volatile) and self.transform.volatile,
            expected=len(self),
        )
        weakref.finalize(self, writer.close)
        return writer

    def __len__(self) -> int:
        return len(self.input_parent)

    @overload
    def __getitem__(self, index: slice) -> grain.MapDataset[CachedValue]: ...

    @overload
    def __getitem__(self, index: int) -> CachedValue: ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.slice(index)
        return self.writer.process(
            index % len(self), index % len(self), self.input_parent[index], self.transform
        )

    def close(self) -> None:
        if "writer" in self.__dict__:
            self.writer.close()

    def __getstate__(self) -> dict:
        return {key: value for key, value in self.__dict__.items() if key != "writer"}


@dataclass(frozen=True)
class _CachedReader:
    reader: Callable[[Shard], grain.IterDataset]
    cache: Cache
    key: DependencyManifest
    transform: Transform | None

    def __call__(self, shard: Shard) -> grain.IterDataset:
        return _CachedIter(self.reader(shard), self.cache, self.key, shard, self.transform)


class _CachedIter(grain.IterDataset):
    def __init__(
        self,
        parent: grain.IterDataset,
        cache: Cache,
        key: DependencyManifest,
        shard: Shard,
        transform: Transform | None,
    ):
        super().__init__(parent)
        self.cache, self.key, self.shard, self.transform = cache, key, shard, transform
        self.input_parent = parent

    def __iter__(self) -> grain.DatasetIterator:
        return _CachedIterator(self.input_parent.__iter__(), self.cache, self.key, self.shard, self.transform)


class _CachedIterator(grain.DatasetIterator):
    def __init__(
        self,
        parent: grain.DatasetIterator,
        cache: Cache,
        key: DependencyManifest,
        shard: Shard,
        transform: Transform | None,
    ):
        super().__init__(parent)
        self.parent, self.shard, self.transform = parent, shard, transform
        self.writer = _Writer(cache, key, volatile=isinstance(transform, _Volatile) and transform.volatile)
        self.ordinal = 0

    def __next__(self):
        try:
            value = next(self.parent)
        except StopIteration:
            self.writer.close()
            raise
        value = self.writer.process((self.shard.id, self.ordinal), self.ordinal, value, self.transform)
        self.ordinal += 1
        return value

    def get_state(self) -> dict:
        return {"parent": self.parent.get_state(), "ordinal": self.ordinal}

    def set_state(self, state: Mapping[str, object]) -> None:
        ordinal = state["ordinal"]
        parent = state["parent"]
        if not isinstance(ordinal, int) or not isinstance(parent, Mapping):
            raise ValueError("invalid cached shard iterator state")
        self.parent.set_state(parent)
        self.ordinal = ordinal

    def close(self) -> None:
        self.parent.close()
        self.writer.close()


@dataclass(frozen=True)
class Generation:
    """A named immutable generation. Missing or corrupt objects are errors."""

    url: str
    storage_options: Mapping[str, object] | None = field(default=None, compare=False, hash=False, repr=False)
    _definition: Mapping[str, object] | None = field(default=None, compare=False, hash=False, repr=False)

    def reference(self, manifest: DependencyManifest) -> str:
        return f"generation {self.url}"

    @cached_property
    def pin(self):
        _, path, scheme = Store(self.url, self.storage_options)._backend()
        return LocalRetention.pin(Path(path)) if scheme == "file" else None

    @cached_property
    def manifest(self) -> dict:
        _ = self.pin
        if self._definition is not None:
            return dict(self._definition)
        record = json.loads(Store(self.url, self.storage_options).child("manifest.json").read())
        if record["schema"] != 1:
            raise ValueError(f"unknown cache generation schema at {self.url}")
        _, path, scheme = Store(self.url, self.storage_options)._backend()
        if scheme == "file":
            LocalRetention.touched(Path(path))
        return record

    @cached_property
    def _page(self):
        @lru_cache(maxsize=4)
        def opened(page: str) -> dict:
            return json.loads(_checked(Store(self.url, self.storage_options), self.manifest["pages"][page]))

        return opened

    @cached_property
    def _rows(self):
        @lru_cache(maxsize=2)
        def opened(page: int) -> list[dict]:
            return json.loads(
                _checked(Store(self.url, self.storage_options), self.manifest["row_pages"][page])
            )

        return opened

    @cached_property
    def _shard(self):
        @lru_cache(maxsize=2)
        def opened(path: str) -> ArrayRecords:
            root = Store(self.manifest["root"], self.storage_options)
            mirrored = (
                root
                if root.url.startswith("dewcache::")
                else Store("dewcache::" + root.url, self.storage_options)
            )
            _checked(mirrored, self.manifest["shards"][path])
            return ArrayRecords((root.child(path).url,), self.storage_options)

        return opened

    def __getstate__(self) -> Mapping[str, object]:
        return {"url": self.url, "storage_options": self.storage_options, "_definition": self._definition}

    def source(self) -> Source:
        return _GenerationSource(self)

    def close(self) -> None:
        pin = self.__dict__.pop("pin", None)
        if pin is not None:
            pin.close()
        for cached in ("_shard", "_page", "_rows"):
            self.__dict__.pop(cached, None)

    def plan(self, name: str) -> np.ndarray:
        encoded = _checked(Store(self.url, self.storage_options), self.manifest["plan"][name])
        return np.load(io.BytesIO(encoded), allow_pickle=False)

    def outputs(self, row_id: RowId) -> Outputs:
        row_key = _row_key(row_id)
        page = hashlib.sha256(row_key.encode()).hexdigest()[:2]
        entry = self._page(page)[row_key]
        payload = self._shard(entry["shard"])[entry["record"]]
        if not isinstance(payload, bytes):
            raise ValueError(f"committed cache record is not bytes: {self.url}")
        values = _decode_outputs(payload)
        if not isinstance(values, tuple) or _outputs_digest(values) != entry["payload_sha256"]:
            raise ValueError(f"committed cache completion is corrupt for {row_key}")
        return values

    def __len__(self) -> int:
        return self.manifest["rows"]

    def mapped(
        self,
        parent: grain.MapDataset,
        key: DependencyManifest,
        source: Source,
        *,
        transform: Transform | None = None,
    ) -> grain.MapDataset:
        return grain.MapDataset.source(_GenerationInputs(self, len(parent), None, transform is not None))

    def streamed(
        self,
        reader: Callable[[Shard], grain.IterDataset],
        key: DependencyManifest,
        manifest: ShardManifest,
        *,
        transform: Transform | None = None,
    ) -> Callable[[Shard], grain.IterDataset]:
        return _GenerationReader(self, transform is not None)

    def __getitem__(self, index: int) -> Batch | bytes | None:
        if not 0 <= index < len(self):
            raise IndexError(index)
        entry = self._rows(index // 4096)[index % 4096]
        row_id = json.loads(entry["id"])
        if isinstance(row_id, list):
            row_id = row_id[0], row_id[1]
        value = _accepted(self.outputs(row_id))[entry["output"]]
        if value is None or isinstance(value, (Mapping, bytes)):
            return value
        raise TypeError("generation source records must be mappings or bytes")


@dataclass(frozen=True)
class _GenerationInputs:
    generation: Generation
    count: int
    shard: str | None
    expanded: bool

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        values = self.generation.outputs(index if self.shard is None else (self.shard, index))
        if self.expanded:
            return values
        if len(values) != 1:
            raise ValueError("generation fanout differs from the pipeline's one-to-one cache stage")
        return values[0]


@dataclass(frozen=True)
class _GenerationReader:
    generation: Generation
    expanded: bool

    def __call__(self, shard: Shard) -> grain.IterDataset:
        count = self.generation.manifest["source_shards"][shard.id]
        return grain.MapDataset.source(
            _GenerationInputs(self.generation, count, shard.id, self.expanded)
        ).to_iter_dataset()


@dataclass(frozen=True)
class _GenerationSource:
    generation: Generation

    def describe(self) -> str:
        return f"generation:{self.generation.url}"

    def snapshot(self) -> Mapping[str, object]:
        return {
            "store": "generation",
            "uri": self.generation.url,
            "manifest": _hash(_json(self.generation.manifest)),
        }

    def supports(self, read) -> bool:
        return True

    def records(self, read):
        return self.generation

    def manifest(self) -> ShardManifest:
        count = len(self.generation)
        return ShardManifest(
            tuple(
                Shard(
                    f"{self.generation.url}:{start}",
                    self.generation.url,
                    start,
                    min(start + 4096, count),
                    min(4096, count - start),
                )
                for start in range(0, count, 4096)
            ),
            self._read_shard,
        )

    def _read_shard(self, shard: Shard) -> grain.IterDataset:
        return grain.MapDataset.source(self.generation)[shard.start : shard.stop].to_iter_dataset()

    def metadata(self) -> Mapping[str, object]:
        metadata = self.generation.manifest["metadata"]
        return {
            "rows": len(self.generation),
            "accepted_rows": len(self.generation),
            **(
                {
                    "valid_targets": metadata["valid_targets"],
                    "zero_target_windows": metadata["zero_target_windows"],
                }
                if "valid_targets" in metadata
                else {}
            ),
        }

    def lengths(self) -> np.ndarray | None:
        entry = self.generation.manifest["metadata"].get("lengths")
        if entry is None:
            return None
        encoded = _checked(Store(self.generation.url, self.generation.storage_options), entry)
        return np.load(io.BytesIO(encoded), allow_pickle=False)
