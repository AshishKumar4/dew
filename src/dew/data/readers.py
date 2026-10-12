"""Checkpointed Grain readers, host stages and canonical finite shares."""

from __future__ import annotations

import dataclasses
import json
import weakref
from collections.abc import Callable, Iterator, Mapping
from typing import Literal

import grain.python as pygrain
import numpy as np

from dew.data.dataset import (
    Batch,
    Checkpointable,
    Closeable,
    DataPartition,
    FilledBatch,
    Forwarding,
    GlobalReader,
    LengthGroups,
    Position,
    Reader,
    rows_of,
)
from dew.data.mix import CorpusCounts, state_counts
from dew.data.pipeline import HostStep
from dew.data.plan import Plan
from dew.records import integer, record


class _GroupedIterator(pygrain.DatasetIterator[Batch]):
    def __init__(self, parent: pygrain.DatasetIterator[Batch], groups: LengthGroups, rows: int, seed: int):
        super().__init__(parent)
        self._read, self._groups, self._rows, self._seed = parent, groups, rows, seed
        self._before = parent.get_state()
        self._buffer: list[Batch] = []
        self._emitted, self._window = 0, 0

    def _fill(self) -> None:
        self._before = self._read.get_state()
        held = []
        for _ in range(self._rows * self._groups.window):
            try:
                held.append(next(self._read))
            except StopIteration:
                break
        ranked = sorted(held, key=self._groups.length)
        batches = [ranked[start : start + self._rows] for start in range(0, len(ranked), self._rows)]
        order = np.random.default_rng([self._seed, self._window]).permutation(len(batches))
        self._buffer = [value for index in order for value in batches[index]]
        self._emitted = 0

    def __next__(self) -> Batch:
        if self._emitted == len(self._buffer):
            if self._buffer:
                self._window += 1
            self._fill()
        if not self._buffer:
            raise StopIteration
        value = self._buffer[self._emitted]
        self._emitted += 1
        return value

    def get_state(self) -> Mapping[str, object]:
        return {"parent": self._before, "emitted": self._emitted, "window": self._window}

    def set_state(self, state: Mapping[str, object]) -> None:
        self._read.set_state(record(state["parent"], "group parent"))
        self._window = integer(state["window"], "group window")
        self._fill()
        self._emitted = integer(state["emitted"], "group emitted")
        if not 0 <= self._emitted <= len(self._buffer):
            raise ValueError("saved grouping position exceeds its window")

    def close(self) -> None:
        self._read.close()


class _GroupedRows(pygrain.IterDataset[Batch]):
    def __init__(self, parent: pygrain.IterDataset[Batch], groups: LengthGroups, rows: int, seed: int):
        super().__init__(parent)
        self._records, self._groups, self._rows, self._seed = parent, groups, rows, seed

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        return _GroupedIterator(iter(self._records), self._groups, self._rows, self._seed)


class _Host(Forwarding):
    def __init__(self, source: Iterator[Batch], steps: tuple[HostStep, ...]):
        self._source: Iterator[Batch] | None = source
        self._steps = steps

    def _forwarded(self) -> Iterator[Batch] | None:
        return self._source

    def __iter__(self) -> Iterator[Batch]:
        return self

    def __next__(self) -> Batch:
        if self._source is None:
            raise StopIteration
        value = next(self._source)
        for step in self._steps:
            value = step.map(value) if isinstance(step, pygrain.MapTransform) else step(value)
        return value

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._source = None

    def get_state(self) -> Position:
        if not isinstance(self._source, Checkpointable):
            raise TypeError("map_batch parent does not expose a checkpoint position")
        return self._source.get_state()

    def set_state(self, state: Position) -> None:
        if not isinstance(self._source, Checkpointable):
            raise TypeError("map_batch parent does not expose a checkpoint position")
        self._source.set_state(state)

    @property
    def records(self) -> int:
        if not isinstance(self._source, GlobalReader):
            raise TypeError("host-stage parent has no global record count")
        return self._source.records

    @property
    def order(self) -> str:
        if not isinstance(self._source, GlobalReader):
            raise TypeError("host-stage parent has no global order")
        return self._source.order


class _CanonicalPass:
    def __init__(self, reader: Reader, partition: DataPartition, batch: int):
        import jax

        self._partition = partition
        self._batch = batch
        self._source = reader(DataPartition()) if jax.process_index() == 0 else None

    def __iter__(self) -> Iterator[Batch]:
        return self

    def __next__(self) -> Batch:
        import jax
        from jax.experimental import multihost_utils

        from dew.coordination import agreed, broadcast_from_process_zero
        from dew.data.sources.codec import decode, encode

        root = jax.process_index() == 0
        held = agreed(
            "canonical validation read",
            lambda: next(self._source, None) if self._source is not None else None,
        )
        payload = b"" if held is None else encode(held)
        size = broadcast_from_process_zero(len(payload))
        if not size:
            raise StopIteration
        if jax.process_count() > 1:
            buffer = np.frombuffer(payload, np.uint8).copy() if root else np.zeros(size, np.uint8)
            payload = np.asarray(multihost_utils.broadcast_one_to_all(buffer, is_source=root)).tobytes()
        whole = record(decode(payload), "canonical validation batch")
        return _FillShare(self._partition, self._batch)(whole)

    def close(self) -> None:
        if isinstance(self._source, Closeable):
            self._source.close()
        self._source = None


class _PerShare(Forwarding):
    @property
    def endless(self) -> bool:
        return self._endless

    def __init__(
        self,
        open_reader: Callable[[], pygrain.DatasetIterator[Batch]],
        plan: Plan,
        resume: Literal["exact", "non-exact"],
        *,
        endless: bool = True,
    ):
        self._open, self._plan = open_reader, plan
        self._resume: Literal["exact", "non-exact"] = resume
        self._endless = endless
        self._read: pygrain.DatasetIterator[Batch] | None = None
        self._saved: Mapping[str, object] | None = None

    def _forwarded(self) -> pygrain.DatasetIterator[Batch]:
        if self._read is None:
            self._read = self._open()
            if self._saved is not None:
                self._read.set_state(self._saved)
        return self._read

    def __iter__(self) -> Iterator[Batch]:
        return self

    def __next__(self) -> Batch:
        return next(self._forwarded())

    def get_state(self) -> bytes:
        return json.dumps(
            {"dew_data_plan": self._plan.record(), "iterator": self._forwarded().get_state()}
        ).encode()

    def set_state(self, state: bytes) -> None:
        decoded = record(json.loads(state), "saved per-share position")
        self._plan.check(record(decoded.get("dew_data_plan"), "saved data plan"), resume=self._resume)
        saved = record(decoded.get("iterator"), "saved Grain iterator")
        self.close()
        self._saved = saved

    def identity(self) -> str:
        """What every share of the pool reads, which the pool agrees before it
        reads (`Plan.identity`): the plan without this share's index, which
        only the position it saves keeps, so a share resumes only its own rows."""
        return self._plan.identity()

    def statistics(self) -> Mapping[str, float]:
        if self._read is None:
            return {}
        return {
            key: value
            for counts in state_counts(self._read.get_state()).values()
            for key, value in counts.metrics().items()
        }

    def close(self) -> None:
        if self._read is not None:
            self._read.close()
            self._read = None


class _Statistics:
    def __init__(self, counts: Mapping[str, CorpusCounts]):
        self._counts = counts
        self._readers: list[weakref.ReferenceType[_PerShare]] = []

    def observe(self, stream: _PerShare) -> None:
        self._readers.append(weakref.ref(stream))

    def __call__(self) -> Mapping[str, float]:
        scalars = {key: value for counts in self._counts.values() for key, value in counts.metrics().items()}
        for reference in self._readers:
            stream = reference()
            if stream is not None:
                for name, value in stream.statistics().items():
                    scalars[name] = max(scalars.get(name, 0.0), value)
        return scalars


@dataclasses.dataclass(frozen=True)
class _FillShare:
    partition: DataPartition
    batch: int

    def __call__(self, batch: Batch) -> Batch:
        import jax

        mine = jax.tree.map(lambda value: value[self.partition.index :: self.partition.count], batch)
        empty = rows_of(mine) == 0
        if empty:
            mine = jax.tree.map(lambda value: value[:1], batch)
        return FilledBatch(self.partition.rows(self.batch), real=not empty).map(mine)
