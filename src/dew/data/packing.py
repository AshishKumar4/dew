"""First-fit windows over a generation's published, paged order arrays."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import grain.python as grain
import numpy as np

from dew.data.cache import Generation
from dew.data.dataset import Batch
from dew.data.pipeline import Pack
from dew.data.plan import fingerprint
from dew.data.sources.base import Read, Shard, ShardManifest, Source
from dew.data.tokens import chunk_lengths, chunk_table, first_fit, packed_window
from dew.records import record


def plan_name(pack: Pack, field: str) -> str:
    return "packing_" + fingerprint(dataclasses.asdict(pack))[:20] + "_" + field


@dataclasses.dataclass(frozen=True)
class PackingPlan:
    pack: Pack

    def __call__(self, generation: Generation) -> Mapping[str, np.ndarray]:
        if self.pack.policy != "first-fit":
            raise ValueError("indexed published packing currently needs policy='first-fit'")
        lengths = generation.source().lengths()
        if lengths is None:
            raise ValueError("published packing needs certified per-document lengths")
        chunk_sizes = chunk_lengths(lengths, self.pack.seq_len)
        documents, offsets = chunk_table(lengths, self.pack.seq_len)
        order, starts = first_fit(chunk_sizes, self.pack.seq_len, self.pack.bins)
        targets = np.asarray(
            [
                sum(
                    max(0, int(chunk_sizes[index]) - 1)
                    for index in order[starts[window] : starts[window + 1]]
                )
                for window in range(len(starts) - 1)
            ],
            np.int64,
        )
        kept = np.flatnonzero(targets > 0)
        arrays = {
            "documents": documents,
            "offsets": offsets,
            "sizes": chunk_sizes,
            "order": order,
            "starts": starts,
            "targets": targets,
            "accepted": kept,
            "totals": np.asarray([targets[kept].sum(), np.count_nonzero(targets == 0)], np.int64),
        }
        return {plan_name(self.pack, field): value for field, value in arrays.items()}


class _PackedRows:
    def __init__(self, source: Source, generation: Generation, pack: Pack, read: Read):
        self._documents, self._generation, self._pack = source.records(read), generation, pack

    def _at(self, field: str, index: int) -> int:
        return int(self._generation.plan_at(plan_name(self._pack, field), index).item())

    def __len__(self) -> int:
        return self._generation.plan_length(plan_name(self._pack, "accepted"))

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
        return _PackedRows(self.source, self.generation, self.pack, read)

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
