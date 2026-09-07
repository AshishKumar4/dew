"""The grain experiments dew uses: the elastic iterator behind a training
stream's global position, and first-fit packing of token chunks."""

from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from grain.python import (IterDataset, MapDataset, MultiprocessingOptions, ReadOptions,
                          ShardOptions)

T = TypeVar("T")


class FirstFitPackIterDataset(IterDataset[T], Generic[T]):
    """Whole records packed first-fit into fixed windows, in packing order."""

    def __init__(
        self,
        parent: IterDataset[Any],
        *,
        length_struct: Any,
        num_packing_bins: int,
        seed: int = ...,
        shuffle_bins: bool = ...,
        shuffle_bins_group_by_feature: str | None = ...,
        meta_features: Sequence[str] = ...,
        pack_alignment_struct: Any = ...,
        padding_struct: Any = ...,
        max_sequences_per_bin: int | None = ...,
    ) -> None: ...


class ElasticIterator(IterDataset[T], Generic[T]):
    """Batching and sharding owned by the iterator, so its state is one global
    record offset every shard reports and any shard count can read."""

    def __init__(
        self,
        ds: MapDataset[T],
        global_batch_size: int,
        shard_options: ShardOptions,
        *,
        read_options: ReadOptions = ...,
        multiprocessing_options: MultiprocessingOptions | None = ...,
    ) -> None: ...
