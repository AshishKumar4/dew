"""Used-surface stubs for grain, which ships no `py.typed`.

grain's own source carries inline types, and pyright reads those for what
they cover — but `RandomMap.random_map` is an unannotated abstract method, so
dew's transforms, which return records from it, look like incompatible
overrides. These stubs declare the surface dew uses as it really is: the
real parameter names and defaults (checked against the installed grain),
`random_map` as any record in and any record out (the base is untyped, so
`Any` is the honest annotation, not a silent precise one), and
`MapDataset.__getitem__` with grain's own overloads. Types that name grain
internals dew never touches are written as the public surface dew uses.
Extend as use grows; a use outside this surface fails here, naming it.
"""

import abc
import builtins
from collections.abc import Callable, Iterator, Sequence
from os import PathLike
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, overload

import numpy as np

T = TypeVar("T")
S = TypeVar("S")


class RandomAccessDataSource(Protocol[T]):
    """Random access over records, structurally: dew's sources implement this
    without inheriting it, which is the contract grain declares."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> T: ...


class RandomMapTransform(abc.ABC):
    """One random 1:1 record transformation, run in parallel over records."""

    @abc.abstractmethod
    def random_map(self, element: Any, rng: np.random.Generator) -> Any:
        """Maps a single element: any record in, any record out."""


class Batch:
    """Form batches of `batch_size` records, whole ones only when asked."""

    def __init__(
        self,
        batch_size: int,
        drop_remainder: bool = ...,
        batch_fn: Callable[[Sequence[Any]], Any] | None = ...,
    ) -> None: ...


Transformation: TypeAlias = Batch | RandomMapTransform
"""What a data pipeline step is, of the kinds dew runs: a random map or a batch."""


class MapDataset(Generic[T]):
    """Random-access records with transformations returning new datasets."""

    def __init__(
        self, parents: MapDataset[Any] | Sequence[MapDataset[Any]] = ()
    ) -> None: ...
    @classmethod
    def source(
        cls, source: Sequence[T] | RandomAccessDataSource[T]
    ) -> MapDataset[T]: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, index: builtins.slice) -> MapDataset[T]: ...
    @overload
    def __getitem__(self, index: int) -> T | None: ...
    @property
    def _parent(self) -> MapDataset[T]: ...
    def seed(self, seed: int) -> MapDataset[T]: ...
    def apply(
        self, transformations: Transformation | Sequence[Transformation]
    ) -> MapDataset[Any]: ...
    def slice(self, sl: builtins.slice) -> MapDataset[T]: ...
    def repeat(
        self, num_epochs: int | None = ..., *, reseed_each_epoch: bool = ...
    ) -> MapDataset[T]: ...
    def shuffle(self, seed: int | None = ...) -> MapDataset[T]: ...
    def to_iter_dataset(
        self, read_options: ReadOptions | None = ..., *, allow_nones: bool = ...
    ) -> IterDataset[T]: ...


class IterDataset(Generic[T]):
    """Records read once, in order, with the worker machinery behind them."""

    def __iter__(self) -> Iterator[T]: ...
    def mp_prefetch(
        self,
        options: MultiprocessingOptions | None = ...,
        worker_init_fn: Callable[[int, int], None] | None = ...,
        sequential_slice: bool = ...,
    ) -> IterDataset[T]: ...
    def batch(
        self,
        batch_size: int,
        *,
        drop_remainder: bool = ...,
        batch_fn: Callable[[Sequence[T]], S] | None = ...,
    ) -> IterDataset[S]: ...


class ShardOptions:
    """How a sampler or source splits records over processes."""


class ShardByJaxProcess(ShardOptions):
    """Shard index and count from the JAX process layout."""

    def __init__(self, drop_remainder: bool = ...) -> None: ...


class Sampler:
    """An index stream over records."""


class IndexSampler(Sampler):
    """Shuffled or ordered record indices, reshuffled every epoch."""

    def __init__(
        self,
        num_records: int,
        shard_options: ShardOptions = ...,
        shuffle: bool = ...,
        num_epochs: int | None = ...,
        seed: int | None = ...,
    ) -> None: ...


class ReadOptions:
    """Reader threads and buffers behind an iterable dataset."""

    def __init__(self, num_threads: int = ..., prefetch_buffer_size: int = ...) -> None: ...


class MultiprocessingOptions:
    """Worker processes and buffers ahead of the read."""

    def __init__(
        self,
        num_workers: int = ...,
        per_worker_buffer_size: int = ...,
        enable_profiling: bool = ...,
    ) -> None: ...


class DataLoader:
    """A source, a sampler and operations as an iterable of batches."""

    def __init__(
        self,
        *,
        data_source: RandomAccessDataSource[Any],
        sampler: Sampler,
        operations: Sequence[Transformation] = ...,
        worker_count: int | None = ...,
        worker_buffer_size: int = ...,
        shard_options: ShardOptions | None = ...,
        read_options: ReadOptions | None = ...,
        enable_profiling: bool = ...,
    ) -> None: ...
    def __iter__(self) -> Iterator[Any]: ...


class ArrayRecordDataSource(RandomAccessDataSource[bytes]):
    """Random access over ArrayRecord files."""

    def __init__(
        self,
        paths: str | PathLike[str] | Sequence[str | PathLike[str]],
        reader_options: dict[str, str] | None = ...,
    ) -> None: ...
