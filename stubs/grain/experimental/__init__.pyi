"""The grain experiments dew uses: first-fit packing of token chunks, and
the bounded thread buffer a streamed split reads through."""

from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from grain.python import IterDataset

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
