"""First-fit windows by a plan made from document lengths alone.

`plan_windows` makes the plan, which chunks of which documents each window
holds and which windows hold a target. A published generation stores it
(`PackingPlan`, read back by `PackedSource`), and a source that states its
documents' lengths is packed by it when it is opened (`PlannedSource`), so
both read windows by index in one global order, the same at any process
count, without a pass over the tokens.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping

import grain.python as grain
import numpy as np

from dew.data.cache import Generation
from dew.data.dataset import Batch, Records
from dew.data.pipeline import Pack
from dew.data.plan import fingerprint
from dew.data.sources.base import Read, Shard, ShardManifest, Source
from dew.data.tokens import chunk_lengths, chunk_table, first_fit, packed_window
from dew.records import record


def plan_name(pack: Pack, field: str) -> str:
    return "packing_" + fingerprint(dataclasses.asdict(pack))[:20] + "_" + field


def plan_windows(lengths: np.ndarray, pack: Pack) -> Mapping[str, np.ndarray]:
    """The first-fit plan of documents of `lengths` into `pack.seq_len`-token
    windows: each chunk's document, offset and size, the chunks in packing
    order and where each window starts in it, each window's targets, and the
    windows that hold one (`accepted`), with their total and the count left
    out."""
    if pack.policy != "first-fit":
        raise ValueError("indexed packing currently needs policy='first-fit'")
    chunk_sizes = chunk_lengths(lengths, pack.seq_len)
    documents, offsets = chunk_table(lengths, pack.seq_len)
    order, starts = first_fit(chunk_sizes, pack.seq_len, pack.bins)
    targets = np.asarray(
        [
            sum(max(0, int(chunk_sizes[index]) - 1) for index in order[starts[window] : starts[window + 1]])
            for window in range(len(starts) - 1)
        ],
        np.int64,
    )
    kept = np.flatnonzero(targets > 0)
    return {
        "documents": documents,
        "offsets": offsets,
        "sizes": chunk_sizes,
        "order": order,
        "starts": starts,
        "targets": targets,
        "accepted": kept,
        "totals": np.asarray([targets[kept].sum(), np.count_nonzero(targets == 0)], np.int64),
    }


@dataclasses.dataclass(frozen=True)
class PackingPlan:
    pack: Pack

    def __call__(self, generation: Generation) -> Mapping[str, np.ndarray]:
        lengths = generation.source().lengths()
        if lengths is None:
            raise ValueError("published packing needs certified per-document lengths")
        planned = plan_windows(lengths, self.pack)
        return {plan_name(self.pack, field): value for field, value in planned.items()}


@dataclasses.dataclass(frozen=True)
class _Published:
    """A plan as a generation stores it, read a page at a time."""

    generation: Generation
    pack: Pack

    def at(self, field: str, index: int) -> int:
        return int(self.generation.plan_at(plan_name(self.pack, field), index).item())

    def __len__(self) -> int:
        return self.generation.plan_length(plan_name(self.pack, "accepted"))


@dataclasses.dataclass(frozen=True)
class _Held:
    """A plan made when its source is opened, held in memory."""

    arrays: Mapping[str, np.ndarray]

    def at(self, field: str, index: int) -> int:
        return int(self.arrays[field][index])

    def __len__(self) -> int:
        return len(self.arrays["accepted"])


class _PackedRows:
    """The windows a plan makes of `documents`, by index, each with its targets."""

    def __init__(self, documents: Records, plan: _Published | _Held, pack: Pack):
        self._documents, self._plan, self._pack = documents, plan, pack

    def _at(self, field: str, index: int) -> int:
        return self._plan.at(field, index)

    def __len__(self) -> int:
        return len(self._plan)

    def __getitem__(self, index: int) -> Batch:
        if not 0 <= index < len(self):
            raise IndexError(index)
        window = self._at("accepted", index)
        start, stop = self._at("starts", window), self._at("starts", window + 1)
        chunks = []
        for position in range(start, stop):
            chunk = self._at("order", position)
            document = record(self._documents[self._at("documents", chunk)], "packed document")
            offset, size = self._at("offsets", chunk), self._at("sizes", chunk)
            chunks.append(
                {name: np.asarray(document[name])[offset : offset + size] for name in self._pack.fields}
            )
        return {
            **packed_window(chunks, self._pack.seq_len),
            "_dew_valid_targets": self._at("targets", window),
        }


@dataclasses.dataclass(frozen=True)
class PackedSource:
    source: Source
    generation: Generation
    pack: Pack

    def describe(self) -> str:
        return f"{self.source.describe()} published packing={self.pack!r}"

    def snapshot(self) -> Mapping[str, object]:
        return {"source": self.source.snapshot(), "packing": self.order_fingerprints()}

    def order_fingerprints(self) -> Mapping[str, str]:
        return {
            field: self.generation.plan_hash(plan_name(self.pack, field))
            for field in ("documents", "offsets", "sizes", "order", "starts", "targets", "accepted")
        }

    def supports(self, read: Read) -> bool:
        return read in ("memory", "indexed", "streamed")

    def records(self, read: Read) -> _PackedRows:
        return _PackedRows(self.source.records(read), _Published(self.generation, self.pack), self.pack)

    def metadata(self) -> Mapping[str, object]:
        total = self.generation.plan_at(plan_name(self.pack, "totals"), 0)
        return {
            **self.source.metadata(),
            "windows": len(self.records("indexed")),
            "rows": len(self.records("indexed")),
            "valid_targets": int(total.item()),
            "zero_target_windows": 0,
        }

    def lengths(self) -> np.ndarray | None:
        return None

    def manifest(self) -> ShardManifest:
        size = len(self.records("indexed"))
        return ShardManifest(
            tuple(
                Shard(
                    f"{self.generation.url}:{start}",
                    self.generation.url,
                    start,
                    min(start + 1024, size),
                    min(1024, size - start),
                )
                for start in range(0, size, 1024)
            ),
            self._reader,
        )

    def _reader(self, shard: Shard) -> grain.IterDataset[Batch]:
        return grain.MapDataset.source(self.records("indexed"))[shard.start : shard.stop].to_iter_dataset()


@dataclasses.dataclass(frozen=True)
class PlannedSource:
    """A source's documents packed first-fit by the plan of the lengths the
    source states (`plan_windows`), the plan a published generation stores,
    made from those lengths alone when the source is opened."""

    source: Source
    pack: Pack

    @functools.cached_property
    def plan(self) -> Mapping[str, np.ndarray]:
        lengths = self.source.lengths()
        if lengths is None:
            raise ValueError(f"{self.source.describe()} states no document lengths to plan its packing by")
        return plan_windows(lengths, self.pack)

    def describe(self) -> str:
        return f"{self.source.describe()} planned packing={self.pack!r}"

    def snapshot(self) -> Mapping[str, object]:
        return {"source": self.source.snapshot(), "packing": self.order_fingerprints()}

    def order_fingerprints(self) -> Mapping[str, str]:
        return {
            field: fingerprint(self.plan[field])
            for field in ("documents", "offsets", "sizes", "order", "starts", "targets", "accepted")
        }

    def supports(self, read: Read) -> bool:
        return self.source.supports("indexed" if read == "streamed" else read)

    def records(self, read: Read) -> _PackedRows:
        return _PackedRows(self.source.records(read), _Held(self.plan), self.pack)

    def metadata(self) -> Mapping[str, object]:
        windows = len(self.plan["accepted"])
        return {
            **self.source.metadata(),
            "windows": windows,
            "rows": windows,
            "valid_targets": int(self.plan["totals"][0]),
            "zero_target_windows": 0,
        }

    def lengths(self) -> np.ndarray | None:
        return None

    def manifest(self) -> ShardManifest:
        size = len(self.plan["accepted"])
        return ShardManifest(
            tuple(
                Shard(f"{self.describe()}:{start}", self.describe(), start, min(start + 1024, size),
                      min(1024, size - start))
                for start in range(0, size, 1024)
            ),
            self._reader,
        )

    def _reader(self, shard: Shard) -> grain.IterDataset[Batch]:
        return grain.MapDataset.source(self.records("indexed"))[shard.start : shard.stop].to_iter_dataset()
