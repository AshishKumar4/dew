"""Held-out records: a dataset split into the part a run trains on and the parts it scores."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import itertools
import json
import random
from collections.abc import Callable, Mapping, Sequence
from types import UnionType
from typing import TYPE_CHECKING, Literal, Protocol, TypeAliasType, cast, runtime_checkable

import numpy as np
import tyro

from dew.data.dataset import Batch, Records
from dew.data.pipeline import Dataset, Mixture, Phases, Pipeline, Rejected, Stage
from dew.data.sources.base import Read, Shard, ShardManifest, Source, _consumed
from dew.data.sources.codec import encode
from dew.records import integer

if TYPE_CHECKING:
    from dew.data.sources.files import Format


@runtime_checkable
class _SplitSource(Protocol):
    def with_split(self, name: str) -> Source: ...


def _named_split(source: Source, name: str) -> Source:
    """`source`'s own split `name`, through its `with_split`; a source with no native splits
    refuses it by name."""
    if not isinstance(source, _SplitSource):
        raise ValueError(f"source {source.describe()} has no native split {name!r}")
    return source.with_split(name)


@dataclasses.dataclass(frozen=True)
class _SliceRecords:
    source: Source
    start: int
    stop: int | None = None

    def consumed(self, read: Read) -> _SliceRecords:
        return dataclasses.replace(self, source=_consumed(self.source, read))

    def describe(self) -> str:
        return f"{self.source.describe()}[{self.start}:{self.stop}]"

    def fingerprint(self) -> Mapping[str, object]:
        return {**self.source.fingerprint(), "slice": [self.start, self.stop]}

    def supports(self, read: Read) -> bool:
        return read != "streamed" and self.source.supports(read)

    def records(self, read: Read):
        from dew.data.dataset import SourceSlice

        records = self.source.records(read)
        stop = len(records) if self.stop is None else self.stop
        if not 0 <= self.start <= stop <= len(records):
            raise ValueError(f"{self.describe()} exceeds the source's {len(records)} records")
        return SourceSlice(records, self.start, stop)

    def metadata(self) -> Mapping[str, object]:
        stated = self.source.metadata()
        return {**stated, **({"held_out": self.start} if self.start else {})}

    def lengths(self) -> np.ndarray | None:
        lengths = self.source.lengths()
        return None if lengths is None else lengths[self.start : self.stop]

    def manifest(self) -> ShardManifest:
        raise ValueError("a records= holdout reads by index; hold out fractions of a stream")


def _inside(row: Batch | bytes, low: float, high: float, key: str | None) -> bool:
    """Whether `row`'s identity falls in [`low`, `high`) of the unit interval: the hash of its
    `key` field, or of its whole content, the same wherever and however the row is read."""
    if key is None:
        identity = encode(row)
    elif isinstance(row, Mapping) and key in row:
        identity = encode(row[key])
    else:
        raise ValueError(f"a holdout by {key!r} reads that field, which this record lacks")
    hashed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
    return int(low * 2**64) <= hashed < int(high * 2**64)


@dataclasses.dataclass(frozen=True)
class _HashRows:
    low: float
    high: float
    held: bool
    key: str | None = None

    def __call__(self, row: Batch) -> Batch | Rejected:
        return row if _inside(row, self.low, self.high, self.key) == self.held else Rejected("holdout")


@dataclasses.dataclass(frozen=True)
class _SelectedRow:
    rows: Records

    def __call__(self, index: int) -> Batch | bytes | None:
        return self.rows[index]


@dataclasses.dataclass(frozen=True)
class _HashSource:
    """The rows of `source` whose identity (its `key` field, or its content)
    hashes into [`low`, `high`) of the unit interval (`held`), or the rows
    that do not: a row is in exactly one part, the same in every run, read,
    machine and resume."""

    source: Source
    low: float
    high: float
    held: bool
    key: str | None = None

    def consumed(self, read: Read) -> _HashSource:
        return dataclasses.replace(self, source=_consumed(self.source, read))

    def describe(self) -> str:
        part = f"[{self.low},{self.high}),held={self.held},key={self.key}"
        return f"{self.source.describe()} hash holdout={part}"

    def fingerprint(self) -> Mapping[str, object]:
        return {**self.source.fingerprint(), "holdout": [self.low, self.high], "held": self.held,
                "key": self.key}

    def supports(self, read: Read) -> bool:
        return read != "indexed" and self.source.supports(read)

    def records(self, read: Read):
        import grain.python as grain

        rows = self.source.records(read)
        keys = tuple(index for index in range(len(rows))
                     if _inside(cast("Batch", rows[index]), self.low, self.high, self.key) == self.held)
        return grain.MapDataset.source(keys).map(_SelectedRow(rows))

    def metadata(self) -> Mapping[str, object]:
        return self.source.metadata()

    def lengths(self) -> np.ndarray | None:
        return None

    def manifest(self) -> ShardManifest:
        manifest = self.source.manifest()
        return dataclasses.replace(manifest, reader=functools.partial(self._reader, manifest))

    def _reader(self, manifest: ShardManifest, shard: Shard):
        from dew.data.mixing import Accepted

        rows = manifest.reader(shard).map(_HashRows(self.low, self.high, self.held, self.key))
        return Accepted(rows, self.describe())


def _rebased(dataset: Dataset, store: Callable[[Source], Source]) -> Pipeline:
    """`dataset` with each of its stores read through `store`, its steps unchanged."""
    chain = dataset.chain()
    source = chain.source
    if isinstance(source, (Mixture, Phases)):
        children = tuple(_rebased(child, store) for child in source.pipelines)
        source = dataclasses.replace(source, pipelines=children)
    else:
        source = store(source)
    return dataclasses.replace(chain, source=source)


def holdout(dataset: Dataset, fraction: float | None = None, *, key: str | None = None,
            records: int | None = None, split: str | None = None, **named: float) -> tuple[Pipeline, ...]:
    """`Dataset.holdout`."""
    if sum(form is not None for form in (fraction, records, split)) + bool(named) != 1:
        raise ValueError("holdout takes one of a fraction, records=, split= or named fractions")
    if split is not None:
        held = _rebased(dataset, lambda source: _named_split(source, split))
        return _rebased(dataset, lambda source: source), held
    if records is not None:
        if records < 1:
            raise ValueError(f"holdout records= is at least 1, not {records}")
        return (_rebased(dataset, lambda source: _SliceRecords(source, records)),
                _rebased(dataset, lambda source: _SliceRecords(source, 0, records)))
    shares = (fraction,) if fraction is not None else tuple(named.values())
    if any(not 0 < share < 1 for share in shares) or sum(shares) >= 1:
        raise ValueError(f"held-out fractions are each in (0, 1) and sum under 1, not {shares}")
    bounds = np.concatenate([[0.0], np.cumsum(shares)]).tolist()
    trained = _rebased(dataset, lambda source: _HashSource(source, 0.0, bounds[-1], held=False, key=key))
    held = (functools.partial(_HashSource, low=low, high=high, held=True, key=key)
            for low, high in itertools.pairwise(bounds))
    return (trained, *(_rebased(dataset, part) for part in held))


type SourceValue = Source | str | Sequence[Batch] | Mapping[str, Sequence]
type Validation = HoldOut | Split | Pipeline | Source | str | Sequence[Batch] | Mapping[str, Sequence] | None


def source_argument(annotation: type | UnionType | TypeAliasType):
    def read(words):
        from dew.registry import from_record

        word = words[0]
        if word == "None":
            return None
        if word.startswith(("{", "[")):
            value = json.loads(word)
            return (
                from_record(annotation, value, dtypes=False)
                if isinstance(value, Mapping) and "class" in value
                else value
            )
        return word

    def write(value):
        from dew.registry import to_record

        return [
            value
            if isinstance(value, str)
            else "None"
            if value is None
            else json.dumps(to_record(value, None))
        ]

    return tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="URL|JSON",
        instance_from_str=read,
        is_instance=lambda value: value is None
        or isinstance(value, (str, Source, Mapping, Sequence))
        or dataclasses.is_dataclass(value),
        str_from_instance=write,
    )


@dataclasses.dataclass(frozen=True)
class HoldOut:
    records: int | None = None
    fraction: float | None = None
    by: Literal["index", "hash", "random"] = "index"
    most: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if (self.records is None) == (self.fraction is None):
            raise ValueError("HoldOut needs records or fraction, exactly one")
        if self.records is not None and self.records < 0:
            raise ValueError("HoldOut records must be nonnegative")
        if self.fraction is not None and not 0 <= self.fraction < 1:
            raise ValueError("HoldOut fraction must be in [0, 1)")


@dataclasses.dataclass(frozen=True)
class Split:
    name: str = "validation"


@dataclasses.dataclass(frozen=True)
class _RandomSource:
    source: Source
    split: HoldOut
    held: bool

    def consumed(self, read: Read) -> _RandomSource:
        return dataclasses.replace(self, source=_consumed(self.source, read))

    def describe(self) -> str:
        return f"{self.source.describe()} random holdout={self.split!r},held={self.held}"

    def fingerprint(self) -> Mapping[str, object]:
        return {**self.source.fingerprint(), "holdout": dataclasses.asdict(self.split), "held": self.held}

    def supports(self, read: Read) -> bool:
        return read != "streamed" and self.source.supports(read)

    def _indices(self, count: int) -> tuple[int, ...]:
        size = (
            self.split.records
            if self.split.records is not None
            else int(count * (self.split.fraction or 0.0))
        )
        size = min(size, self.split.most) if self.split.most is not None else size
        order = list(range(count))
        random.Random(self.split.seed).shuffle(order)
        chosen = set(order[:size])
        return tuple(index for index in range(count) if (index in chosen) == self.held)

    def records(self, read: Read):
        import grain.python as grain

        rows = self.source.records(read)
        return grain.MapDataset.source(self._indices(len(rows))).map(_SelectedRow(rows))

    def metadata(self) -> Mapping[str, object]:
        return self.source.metadata()

    def lengths(self) -> np.ndarray | None:
        lengths = self.source.lengths()
        return None if lengths is None else lengths[list(self._indices(len(lengths)))]

    def manifest(self) -> ShardManifest:
        raise ValueError("random-index HoldOut needs indexed records; use by='hash' for streams")


def _source(
    value: SourceValue | Mapping[str, float], *, split: str = "train", format: Format | None = None
) -> Source:
    from dew.data.sources import source

    return source(value, split=split, format=format)


def _split(source: Source, split: Split) -> Source:
    """`source`'s own split `split.name`."""
    return _named_split(source, split.name)


def _branches(
    value: SourceValue | Mapping[str, float],
    validation: Validation,
    *,
    split: str,
    prefix: tuple[Stage, ...],
    format: Format | None = None,
) -> tuple[Pipeline, Pipeline | None]:
    source = _source(value, split=split, format=format)
    held: Source | None = None
    train = Pipeline(source, prefix)
    if isinstance(validation, Pipeline):
        return train, validation
    if isinstance(validation, HoldOut):
        if validation.by == "random":
            train = Pipeline(_RandomSource(source, validation, held=False), prefix)
            held = _RandomSource(source, validation, held=True)
        elif validation.by == "hash":
            if validation.fraction is None:
                raise ValueError("hash HoldOut needs fraction=")
            train = Pipeline(_HashSource(source, 0.0, validation.fraction, held=False), prefix)
            held = _HashSource(source, 0.0, validation.fraction, held=True)
        else:
            count = validation.records
            if count is None:
                facts = source.metadata()
                if "rows" not in facts:
                    raise ValueError(f"fraction HoldOut of {source.describe()} needs authoritative rows")
                if validation.fraction is None:
                    raise ValueError("fraction HoldOut needs fraction=")
                count = int(integer(facts["rows"], "rows") * validation.fraction)
            train = Pipeline(_SliceRecords(source, count), prefix)
            held = _SliceRecords(source, 0, count) if count else None
    elif isinstance(validation, Split):
        held = _split(source, validation)
    elif validation is not None:
        held = _source(validation, format=format)
    evaluated = None if held is None else Pipeline(held, prefix)
    return train, evaluated
