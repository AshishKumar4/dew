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

from dew._records import JSON
from dew.batch import Batch
from dew.data.dataset import Records
from dew.data.pipeline import Record, Rejected
from dew.data.plan import array_digest
from dew.data.retention import LocalRetention
from dew.data.sources.base import Read, Shard, ShardManifest, Source, _indexed_manifest, _uncounted, url_label
from dew.data.sources.codec import decode, fingerprint
from dew.data.sources.index import ArrayIndex
from dew.data.sources.records import ArrayRecords
from dew.data.storage import Store, Versioned, local

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


ROW_PAGE = 4096
"""The dense rows a generation's row index holds in each of its pages (`index/rows-*.json`)."""
GENERATION_SCHEMA = 2
"""How a generation's manifest and a fill's segments name inputs and digests: by the shards' places,
which a store mounted elsewhere keeps, and arrays by their dtypes' names (`dew._hashing.array_header`),
since 2. Another schema's is refused, not read under names it was not written with."""


def input_page(row_key: str) -> str:
    """The page of a generation's input index that holds the input `row_key` names."""
    return hashlib.sha256(row_key.encode()).hexdigest()[:2]


def _row_key(value: RowId) -> str:
    return _json(list(value) if isinstance(value, tuple) else value).decode()


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _checked(root: Store, entry: Mapping[str, object]) -> bytes:
    """The object `entry` names under `root` (`Store.child` keeps it inside), checked against the
    size and digest `entry` states (`Store.checked`)."""
    path = entry["path"]
    if not isinstance(path, str):
        raise ValueError("artifact object path is not a string")
    return root.child(path).checked(entry)


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


def _rule(value: Batch | bytes | Rejected | None) -> list[int | None] | None:
    """The role and pad id a published row's targets were counted by (`CountRule`), or None where
    they are every target's, as a row never counted is."""
    from dew.data.steps.text import CountRule

    rule = CountRule.of(value) if isinstance(value, Mapping) else None
    return None if rule is None else [rule.role, rule.pad_id]


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
    if current is not None and plan is None:
        # The committed generation covers every input already, which a fill can only repeat; a
        # packing plan is what a new generation adds.
        return Generation(root.child(f"generations/{json.loads(current.data)['generation']}").url,
                          cache._mirror_options())
    generation = uuid.uuid4().hex
    target = root.child(f"generations/{generation}")
    opened = Generation(target.url, cache._mirror_options())
    _ = opened.pin
    pages: dict[str, dict] = {}
    dense = []
    for row in expected:
        row_key = _row_key(row)
        entry = completions[row_key]
        pages.setdefault(input_page(row_key), {})[row_key] = entry
        dense.extend(
            {
                **entry,
                "output": index,
                "length": entry["lengths"][index],
                "valid_targets": entry["valid_targets"][index],
                "rule": entry["rules"][index],
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
    for offset in range(0, len(dense), ROW_PAGE):
        payload = _json(dense[offset : offset + ROW_PAGE])
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
    # The rules other than every target's that the rows' targets were counted by, which a plan of
    # their lengths cannot count (`dew.data.analysis._ruled`).
    distinct = {tuple(entry["rule"]) for entry in dense if entry["rule"] is not None}
    metadata["count_rules"] = [list(rule) for rule in sorted(distinct, key=repr)]
    referenced = {entry["shard"] for entry in completions.values()}
    definition = {
        "schema": GENERATION_SCHEMA,
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
    digest = array_digest(array.dtype, array.shape)
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
    prior = Generation.named(root, json.loads(current.data), root.storage_options)
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
        segments = (_segment(root, segment, completions, shards) for segment in seal["segments"])
        if seal_facts(seal["segments"], segments) != seal:
            raise ValueError(f"cache writer seal failed verification: {location}")
    return completions, shards



def _segment(root: Store, segment: Mapping[str, object], completions: dict, shards: dict) -> bytes:
    """A sealed segment's bytes, checked with its shard and every completion it records against the
    shard's records, its completions and shard added to those of the writers read before it."""
    payload = _checked(root, segment)
    record = json.loads(payload)
    if record.get("schema") != GENERATION_SCHEMA:
        raise ValueError(f"the cache segment {segment['path']} was written under schema "
                         f"{record.get('schema')!r}, and this Dew reads schema {GENERATION_SCHEMA}, whose "
                         "inputs and digests are named otherwise; fill the cache again")
    _checked(root, record["shard"])
    with closing(ArrayRecords((root.child(record["shard"]["path"]).url,), root.storage_options)) as rows:
        if sorted(entry["record"] for entry in record["completions"]) != list(range(len(rows))):
            raise ValueError(f"cache shard has records without one completion: {record['shard']['path']}")
        for entry in record["completions"]:
            _completion(entry, rows[entry["record"]])
            candidate = {**entry, "shard": record["shard"]["path"]}
            existing = completions.get(entry["id"])
            if existing is not None and existing["payload_sha256"] != entry["payload_sha256"]:
                raise ValueError(f"conflicting cache duplicate for row {entry['id']}")
            completions.setdefault(entry["id"], candidate)
    shards[record["shard"]["path"]] = record["shard"]
    return payload


def completion_facts(row_id: str, payload: bytes) -> Mapping[str, object]:
    """What a cache commits of one input's completion, `payload` its encoded outputs, besides where
    it is: the outputs it accepted, their ids, digests, lengths, targets and count rules, and the
    reasons it rejected the rest. The writer records them (`cache._Writer._flush`) and a reader
    checks every one against the record it reads (`_completion`)."""
    outputs = _decode_outputs(payload)
    accepted = _accepted(outputs)
    return {
        "id": row_id,
        "fanout": len(accepted),
        "outputs": [[row_id, index] for index in range(len(accepted))],
        "sha256": fingerprint(accepted),
        "payload_sha256": fingerprint(outputs),
        "lengths": [_fact(value, "_dew_length") for value in accepted],
        "valid_targets": [_fact(value, "_dew_valid_targets") for value in accepted],
        "rules": [_rule(value) for value in accepted],
        "rejections": [value.reason for value in outputs if isinstance(value, Rejected)],
    }


def seal_facts(segments: Sequence[Mapping[str, object]], payloads: Iterable[bytes]) -> Mapping[str, object]:
    """What a cache writer's seal commits of its `segments`, `payloads` their bytes in order, each
    folded in as it comes: their count, the hash chain over them and the first and last input they
    complete. The writer seals by it (`cache._Writer.seal`) and a reader checks a seal against it
    (`_sealed`)."""
    chain = b""
    first = last = None
    for payload in payloads:
        chain = hashlib.sha256(chain + payload).digest()
        completed = json.loads(payload)["completions"]
        if completed:
            first = completed[0]["id"] if first is None else first
            last = completed[-1]["id"]
    return {"count": len(segments), "segments": list(segments), "first": first, "last": last,
            "sha256": chain.hex()}


def _completion(entry: Mapping[str, object], encoded: Batch | bytes | None) -> None:
    """Refuse a committed completion whose record is not what its facts say (`completion_facts`)."""
    if not isinstance(encoded, bytes):
        raise ValueError(f"cache shard record is not bytes for {entry['id']}")
    expected = completion_facts(str(entry["id"]), encoded)
    differing = sorted(name for name, value in expected.items() if entry.get(name) != value)
    if differing:
        raise ValueError(f"cache completion {differing} differ for {entry['id']}")


def _source_ids(source: Source, *, streamed: bool = False,
                completed: Iterable[str] = ()) -> Iterator[RowId]:
    """The store's input IDs, refusing subdivided fills before reopening it: a
    fill keyed by place (`completed` holds its numbers) numbers the records
    a memory or indexed read holds."""
    completed = tuple(completed)
    if not streamed and (source.supports("indexed") or completed):
        yield from range(len(source.records("indexed" if source.supports("indexed") else "memory")))
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
            yield from (row_id for row_id, _ in _shard_inputs(manifest, shard))


def _shard_inputs(manifest: ShardManifest, shard: Shard) -> Iterator[tuple[RowId, Batch | bytes | None]]:
    """Each row of `shard` as a stream reads it, with the id a streamed fill gives it (`_source_ids`)."""
    reader = iter(manifest.reader(shard))
    try:
        for index, row in enumerate(reader):
            yield (shard.id, index), row
    finally:
        reader.close()


@dataclass(frozen=True)
class Generation:
    """A named immutable generation. Missing or corrupt objects are errors."""

    url: str
    storage_options: Mapping[str, object] | None = field(default=None, compare=False, hash=False, repr=False)
    _definition: Mapping[str, object] | None = field(default=None, compare=False, hash=False, repr=False)
    committed: Mapping[str, object] | None = field(default=None, compare=False, hash=False, repr=False)
    """The size and digest its cache committed its manifest under (`Cache.generation`), which the one
    read of the manifest is checked against."""

    @classmethod
    def named(cls, root: Store, current: Mapping[str, object],
              storage_options: Mapping[str, object] | None) -> Generation:
        """The generation a cache's CURRENT record `current` names under the cache's `root`, its
        manifest checked, when read, against the size and digest CURRENT committed it under."""
        return cls(root.child(f"generations/{current['generation']}").url, storage_options,
                   committed={"bytes": current["bytes"], "sha256": current["sha256"]})

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
        root = Store(self.url, self.storage_options)
        manifest = root.child("manifest.json")
        payload = (manifest.read() if self.committed is None
                   else _checked(root, {**self.committed, "path": "manifest.json"}))
        try:
            record = json.loads(payload)
        except ValueError as error:
            raise ValueError(f"committed cache manifest is corrupt: {manifest.url}") from error
        if not isinstance(record, dict) or record.get("schema") != GENERATION_SCHEMA:
            schema = record.get("schema") if isinstance(record, dict) else None
            raise ValueError(f"the cache generation at {self.url} was written under schema {schema!r}, and "
                             f"this Dew reads schema {GENERATION_SCHEMA}, whose inputs and digests are named "
                             "otherwise; fill the cache again")
        _, path, scheme = Store(self.url, self.storage_options)._backend()
        if scheme == "file":
            LocalRetention.touched(Path(path))
        return record

    @cached_property
    def _page(self):
        # Every page of the input index: inputs are paged by their key's digest, so a read in any order
        # visits each page all the time, which a few pages held would read and check again each time.
        @lru_cache(maxsize=256)
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
        verified: set[str] = set()

        # A shuffled read visits every shard all the time: each is checked whole once (its size and
        # digest), which stands for every record in it (`outputs`), and its readers are kept open.
        @lru_cache(maxsize=64)
        def opened(path: str) -> ArrayRecords:
            root = Store(self.manifest["root"], self.storage_options)
            if path not in verified:
                # A remote shard is checked as its mirror holds it, which its reader then reads; a
                # local one in place, where a mirror would copy it beside itself.
                _checked(root if local(root.resolve()[0]) else root.cached(), self.manifest["shards"][path])
                verified.add(path)
            return ArrayRecords((root.child(path).url,), self.storage_options)

        return opened

    def __getstate__(self) -> Mapping[str, object]:
        return {"url": self.url, "storage_options": self.storage_options, "_definition": self._definition,
                "committed": self.committed}

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

    def holds(self, row_id: RowId) -> bool:
        """Whether this generation holds the input `row_id` names."""
        row_key = _row_key(row_id)
        page = input_page(row_key)
        return page in self.manifest["pages"] and row_key in self._page(page)

    def outputs(self, row_id: RowId) -> Outputs:
        row_key = _row_key(row_id)
        entry = self._page(input_page(row_key))[row_key]
        # The shard it reads from was checked whole (`_shard`), so its record is the one written.
        payload = self._shard(entry["shard"])[entry["record"]]
        if not isinstance(payload, bytes):
            raise ValueError(f"committed cache record is not bytes: {self.url}")
        return _decode_outputs(payload)

    def __len__(self) -> int:
        return self.manifest["rows"]

    def __getitem__(self, index: int) -> Batch | bytes | None:
        if not 0 <= index < len(self):
            raise IndexError(index)
        entry = self._rows(index // ROW_PAGE)[index % ROW_PAGE]
        row_id = json.loads(entry["id"])
        if isinstance(row_id, list):
            row_id = row_id[0], row_id[1]
        return _accepted(self.outputs(row_id))[entry["output"]]


@dataclass(frozen=True)
class _GenerationSource:
    generation: Generation

    def label(self) -> str:
        return url_label(self.generation.url)

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
        return _indexed_manifest(self, self.generation.url, 4096)

    def metadata(self) -> Mapping[str, object]:
        metadata = self.generation.manifest["metadata"]
        return {
            **_uncounted(self.generation.manifest["source_metadata"]),
            "rows": len(self.generation),
            "accepted_rows": len(self.generation),
            **({"count_rules": metadata["count_rules"]} if "count_rules" in metadata else {}),
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
