"""Grain's sliceable root over physical shards, shared by every streamed source."""

from __future__ import annotations

import grain.experimental as experimental
import grain.python as grain

from .dataset import DataPartition, Loading
from .sources.base import ShardManifest


def streamed(manifest: ShardManifest, *, partition: DataPartition, seed: int | None = None,
             cycle_length: int = 16, loading: Loading, epochs: int | None = 1) -> grain.IterDataset:
    if cycle_length < 1:
        raise ValueError("interleave cycle_length must be positive")
    workers = loading.workers if isinstance(loading.workers, int) else 1
    required = partition.count * max(workers, 1)
    if len(manifest.shards) < required:
        raise ValueError(f"stream has {len(manifest.shards)} physical shards for {required} readers × workers; "
                         "produce more shards or reduce readers/workers")
    root = grain.MapDataset.source(manifest.shards)
    if seed is not None:
        root = root.shuffle(seed=seed)
    if epochs != 1:
        root = root.repeat(epochs)
    root = root[partition.index::partition.count].map(manifest.reader)
    return experimental.InterleaveIterDataset(root, cycle_length=cycle_length,
                                               num_make_iter_threads=loading.threads,
                                               make_iter_buffer_size=loading.read_buffer)
