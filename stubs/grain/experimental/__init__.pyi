"""The grain experiments dew uses: first-fit packing of token chunks, and
the bounded thread buffer a streamed split reads through."""

from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from grain.python import IterDataset, MapDataset, MultiprocessingOptions, ReadOptions

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


class ThreadPrefetchIterDataset(IterDataset[T], Generic[T]):
    """`parent` read by one background thread, at most `prefetch_buffer_size`
    elements ahead of the consumer."""

    def __init__(self, parent: IterDataset[T], *, prefetch_buffer_size: int) -> None: ...


class FlatMapIterDataset(IterDataset[T], Generic[T]):
    def __init__(self, parent: IterDataset[Any], transform: Any) -> None: ...


class RepeatIterDataset(IterDataset[T], Generic[T]):
    def __init__(self, parent: IterDataset[T], num_epochs: int | None = ...) -> None: ...


class LimitIterDataset(IterDataset[T], Generic[T]):
    def __init__(self, parent: IterDataset[T], count: int) -> None: ...


class RebatchIterDataset(IterDataset[T], Generic[T]):
    def __init__(self, parent: IterDataset[T], batch_size: int, drop_remainder: bool = ...) -> None: ...


class WindowShuffleMapDataset(MapDataset[T], Generic[T]):
    def __init__(self, parent: MapDataset[T], *, window_size: int, seed: int) -> None: ...


class WindowShuffleIterDataset(IterDataset[T], Generic[T]):
    def __init__(self, parent: IterDataset[T], *, window_size: int, seed: int) -> None: ...


class InterleaveIterDataset(IterDataset[T], Generic[T]):
    def __init__(self, datasets: MapDataset[IterDataset[T]] | Sequence[IterDataset[T] | MapDataset[T]], *,
                 cycle_length: int, num_make_iter_threads: int = ..., make_iter_buffer_size: int = ...,
                 iter_buffer_size: int = ...) -> None: ...


class BestFitPackIterDataset(FirstFitPackIterDataset[T], Generic[T]): ...


class ConcatThenSplitIterDataset(IterDataset[T], Generic[T]):
    def __init__(self, parent: IterDataset[Any], *, length_struct: Any, meta_features: Sequence[str] = ...,
                 split_full_length_features: bool = ...) -> None: ...


class PerformanceConfig:
    multiprocessing_options: MultiprocessingOptions | None
    read_options: ReadOptions | None


def pick_performance_config(ds: IterDataset[Any], *, ram_budget_mb: int | None,
                            max_workers: int | None, max_buffer_size: int | None,
                            samples_to_check: int = ...) -> PerformanceConfig: ...


def batch_and_pad(values: Sequence[T], batch_size: int | None = ...) -> T: ...
