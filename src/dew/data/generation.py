"""Immutable artifact generations and their verified publication."""

from __future__ import annotations

import hashlib
import io
import json
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from dew.data.dataset import Batch, Records
from dew.data.pipeline import Record, Rejected
from dew.data.retention import LocalRetention
from dew.data.sources.base import Read, ShardManifest, Source, _indexed_manifest
from dew.data.sources.codec import decode, fingerprint
from dew.data.sources.index import ArrayIndex
from dew.data.sources.records import ArrayRecords
from dew.data.storage import Store, Versioned
from dew.records import JSON

if TYPE_CHECKING:
    from dew.data.cache import Cache

type Outputs = tuple[Record, ...]
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
    if path.startswith("/") or any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError(f"artifact object path escapes its root: {path!r}")
    payload = root.child(path).read()
    if len(payload) != entry["bytes"] or _hash(payload) != entry["sha256"]:
        raise OSError(
            f"committed object is corrupt: {root.child(path).url}; "
            f"bytes={len(payload)}, sha256={_hash(payload)}, "
            f"expected bytes={entry['bytes']}, sha256={entry['sha256']}"
        )
    return payload


def _accepted(outputs: Sequence[Record]) -> Outputs:
    return tuple(value for value in outputs if value is not None and not isinstance(value, Rejected))


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


def _publish(
    cache: Cache,
    key: DependencyManifest,
    source: Source,
    plan: Callable[[Generation], Mapping[str, np.ndarray]] | None,
    pins: ExitStack,
) -> Generation:
    root = cache._root(key)
    try:
        current = root.child("CURRENT").get_versioned()
    except FileNotFoundError:
        current = None
    completions, shards = _sealed(root, pins)
    if current is not None:
        _retain_committed(root, current, completions, shards, pins)
    expected = tuple(_source_ids(source, streamed=any(row.startswith("[") for row in completions),
                                 completed=completions))
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
    opened = Generation(target.url, cache._mirror_options())
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
        "source_metadata": source.metadata(),
    }
    if plan is not None:
        candidate = Generation(target.url, cache._mirror_options(), definition)
        for name, array in plan(candidate).items():
            if not name.replace("_", "").isalnum():
                raise ValueError(f"invalid packing-plan field {name!r}")
            definition["plan"][name] = _write_plan(target, name, array)
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


def _write_plan(root: Store, name: str, array: np.ndarray) -> Mapping[str, object]:
    if array.dtype.fields is not None or array.dtype.hasobject:
        raise ValueError(f"packing plan {name!r} must be a homogeneous numeric array")
    rows = array.reshape(1) if array.ndim == 0 else array
    row_bytes = max(1, int(np.prod(rows.shape[1:])) * array.dtype.itemsize)
    page_rows = min(4096, max(1, 4 * 2**20 // row_bytes))
    digest = hashlib.sha256(_json({"dtype": array.dtype.str, "shape": list(array.shape)}))
    pages = []
    for offset in range(0, len(rows), page_rows):
        page = np.ascontiguousarray(rows[offset : offset + page_rows])
        digest.update(page.tobytes())
        encoded = io.BytesIO()
        np.save(encoded, page, allow_pickle=False)
        path = f"plan/{name}/{len(pages):06d}.npy"
        pages.append({"path": path, **root.child(path).verified(encoded.getvalue())})
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "rows": len(rows),
        "page_rows": page_rows,
        "pages": pages,
        "sha256": digest.hexdigest(),
    }


def _retain_committed(
    root: Store, current: Versioned, completions: dict, shards: dict, pins: ExitStack
) -> None:
    reference = json.loads(current.data)
    manifest_path = f"generations/{reference['generation']}/manifest.json"
    _checked(root, {"path": manifest_path, "bytes": reference["bytes"], "sha256": reference["sha256"]})
    prior = Generation(root.child(f"generations/{reference['generation']}").url, root.storage_options)
    if prior.pin is not None:
        pins.enter_context(prior.pin)
    for page in prior.manifest["pages"].values():
        retained = json.loads(_checked(Store(prior.url, prior.storage_options), page))
        for row_id, entry in retained.items():
            candidate = completions.get(row_id)
            if candidate is not None and candidate["payload_sha256"] != entry["payload_sha256"]:
                raise ValueError(f"conflicting committed cache row {row_id}")
            completions[row_id] = entry
    shards.update(prior.manifest["shards"])


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
                        raise ValueError(f"conflicting cache duplicate for row {entry['id']}")
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
    if fingerprint(values) != entry["payload_sha256"]:
        raise ValueError(f"cache completion checksum differs for {entry['id']}")
    if fingerprint(accepted) != entry["sha256"]:
        raise ValueError(f"cache accepted-output checksum differs for {entry['id']}")
    if entry["outputs"] != [[entry["id"], index] for index in range(len(accepted))]:
        raise ValueError(f"cache completion output IDs differ for {entry['id']}")
    if entry["lengths"] != [_fact(value, "_dew_length") for value in accepted] or entry["valid_targets"] != [
        _fact(value, "_dew_valid_targets") for value in accepted
    ]:
        raise ValueError(f"cache completion metadata differs for {entry['id']}")


def _source_ids(source: Source, *, streamed: bool = False,
                completed: Iterable[str] = ()) -> Iterator[RowId]:
    """The store's input IDs, refusing subdivided fills before reopening it."""
    if not streamed and source.supports("indexed"):
        yield from range(len(source.records("indexed")))
        return
    manifest = source.manifest()
    if manifest.subdivide is not None:
        store_shards = {shard.id for shard in manifest.shards}
        if any(isinstance(row, list) and row[0] not in store_shards for row in map(json.loads, completed)):
            raise ValueError("cache publication needs the store's own shards; "
                             "row-wise node splits cannot be published")
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
        manifest = Store(self.url, self.storage_options).child("manifest.json")
        try:
            record = json.loads(manifest.read())
        except ValueError as error:
            raise ValueError(f"committed cache manifest is corrupt: {manifest.url}") from error
        if not isinstance(record, dict) or record.get("schema") != 1:
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
            mirrored = root.cached()
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
        metadata = self.manifest["plan"][name]
        pages = [self._plan_page(name, page) for page in range(len(metadata["pages"]))]
        values = np.concatenate(pages) if pages else np.empty(metadata["shape"], dtype=metadata["dtype"])
        return values.reshape(metadata["shape"])

    @cached_property
    def _plan_page(self):
        @lru_cache(maxsize=4)
        def opened(name: str, page: int) -> np.ndarray:
            reference = self.manifest["plan"][name]["pages"][page]
            encoded = _checked(Store(self.url, self.storage_options), reference)
            return np.load(io.BytesIO(encoded), allow_pickle=False)

        return opened

    def plan_at(self, name: str, index: int) -> np.ndarray:
        metadata = self.manifest["plan"][name]
        if not 0 <= index < metadata["rows"]:
            raise IndexError(index)
        page, offset = divmod(index, metadata["page_rows"])
        return np.asarray(self._plan_page(name, page)[offset])

    def plan_length(self, name: str) -> int:
        return self.manifest["plan"][name]["rows"]

    def plan_hash(self, name: str) -> str:
        return self.manifest["plan"][name]["sha256"]

    def outputs(self, row_id: RowId) -> Outputs:
        row_key = _row_key(row_id)
        page = hashlib.sha256(row_key.encode()).hexdigest()[:2]
        entry = self._page(page)[row_key]
        payload = self._shard(entry["shard"])[entry["record"]]
        if not isinstance(payload, bytes):
            raise ValueError(f"committed cache record is not bytes: {self.url}")
        values = _decode_outputs(payload)
        if fingerprint(values) != entry["payload_sha256"]:
            raise ValueError(f"committed cache completion is corrupt for {row_key}")
        return values

    def __len__(self) -> int:
        return self.manifest["rows"]

    def __getitem__(self, index: int) -> Batch | bytes | None:
        if not 0 <= index < len(self):
            raise IndexError(index)
        entry = self._rows(index // 4096)[index % 4096]
        row_id = json.loads(entry["id"])
        if isinstance(row_id, list):
            row_id = row_id[0], row_id[1]
        return _accepted(self.outputs(row_id))[entry["output"]]


@dataclass(frozen=True)
class _GenerationSource:
    generation: Generation

    def describe(self) -> str:
        return f"generation:{self.generation.url}"

    def fingerprint(self) -> Mapping[str, object]:
        return {
            "store": "generation",
            "uri": self.generation.url,
            "manifest": _hash(_json(self.generation.manifest)),
        }

    def supports(self, read: Read) -> bool:
        return True

    def records(self, read: Read) -> Records:
        return self.generation

    def manifest(self) -> ShardManifest:
        return _indexed_manifest(self, self.generation.url, self.generation.url, 4096)

    def metadata(self) -> Mapping[str, object]:
        metadata = self.generation.manifest["metadata"]
        original = self.generation.manifest["source_metadata"]
        stable = {
            name: value
            for name, value in original.items()
            if name not in ("rows", "accepted_rows", "windows", "valid_targets", "zero_target_windows")
        }
        return {
            **stable,
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

    def order_fingerprints(self) -> Mapping[str, str]:
        manifest = self.generation.manifest
        metadata = manifest["metadata"]
        facts = {"accepted_ids": _hash(_json([entry["sha256"] for entry in manifest["row_pages"]]))}
        if "lengths" in metadata:
            facts["lengths"] = metadata["lengths"]["sha256"]
        if "valid_targets" in metadata:
            facts["valid_targets"] = _hash(_json(metadata["valid_targets"]))
        return facts

    def lengths(self) -> np.ndarray | None:
        entry = self.generation.manifest["metadata"].get("lengths")
        if entry is None:
            return None
        return self._length_index.values

    @cached_property
    def _length_index(self) -> ArrayIndex:
        entry = self.generation.manifest["metadata"]["lengths"]
        index = ArrayIndex(
            self.generation.url.rstrip("/") + "/" + entry["path"],
            self.generation.storage_options,
            metadata=entry,
        )
        index.verify(entry)
        return index
