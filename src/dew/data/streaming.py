"""Images and video clips streamed by URL from Hugging Face datasets of
(url, caption) rows."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import TYPE_CHECKING

from .dataset import Batch, DataPartition, Dataset, DatasetSpec, Loading, Tokenize, tokenized

if TYPE_CHECKING:
    from .online_loader import Fetch


@dataclasses.dataclass(frozen=True)
class OnlineImages(DatasetSpec):
    """Fetches images by url as they are read, as an endless stream.

    `sources` names hub datasets, or `gs://` directories saved with
    `save_to_disk`, whose rows hold a url and a caption. The rows are
    concatenated, shuffled once and sharded by the reader's share. Each
    reader then reads its shard over and over, reshuffling between passes.
    So nothing is held out for validation, and the stream cannot resume
    partway through an epoch. A share yields whichever fetches succeed
    first, so two processes cannot read one share identically, and a
    partition whose shares have more than one reader raises `ValueError`.

    A row is dropped and counted when its url yields no image, or when the
    image is not RGB, is under `min_image_size` on its shorter side, is more
    than 2.4 times as long as it is wide, or is a single flat colour. This
    needs the streaming extra (HF `datasets`).
    """

    sources: tuple[str, ...] = ()
    image_size: int = 256
    min_image_size: int = 128
    loading: Loading = dataclasses.field(
        default=Loading(workers=16, threads=512, worker_buffer=20), kw_only=True)
    """The settings of this spec's own fetch pool, which is not a grain reader.

    `workers` and `threads` size the pool, `worker_buffer` is how many
    batches the fetchers run ahead, and `read_buffer` is not used. The
    default differs from grain's, which is sized for file reads and not for
    a pool waiting on urls.
    """
    timeout: int = 15
    retries: int = 3

    def fetch(self) -> Fetch:
        """Return the settings a fetcher uses to turn one of this spec's urls into a sample."""
        from .online_loader import Fetch

        return Fetch(size=self.image_size, min_size=self.min_image_size,
                     timeout=self.timeout, retries=self.retries)

    def load(self, *, batch: int, tokenize: Tokenize | None = None) -> Dataset:
        if not self.sources:
            raise ValueError(f"{type(self).__name__} needs sources= set to one or more datasets")
        from .online_loader import UrlStream, load_rows

        rows = load_rows(self.sources)

        def stream(partition: DataPartition) -> Iterator[Batch]:
            if partition.readers > 1:
                raise ValueError(
                    f"{type(self).__name__} yields whichever samples its fetches return "
                    f"first, so the {partition.readers} processes that read share "
                    f"{partition.index} of {partition.count} would train on different "
                    f"samples as one; lay the mesh out so every process holds rows of "
                    f"its own (no sequence or stage axis across processes)")
            return UrlStream(
                rows.shard(num_shards=partition.count, index=partition.index),
                batch=partition.rows(batch), fetch=self.fetch(),
                workers=self.loading.workers, threads=self.loading.threads,
                prefetch=self.loading.worker_buffer)

        return Dataset(train=tokenized(stream, tokenize), val=None,
                       records=len(rows), batch=batch)


@dataclasses.dataclass(frozen=True)
class OnlineVideos(OnlineImages):
    """Fetches video clips by url as they are read, as an endless stream.

    Each url's video is decoded the way `VideoDataset` reads a local clip:
    `frames` consecutive frames at 25 fps from a random start, each resized
    to an `image_size` square, without audio. A row is dropped and counted
    when its url yields no video at least `frames` frames long, or when the
    video is under `min_image_size` on its shorter side. This needs the
    streaming extra and the `av` extra.
    """

    frames: int = 16

    def fetch(self) -> Fetch:
        return dataclasses.replace(super().fetch(), frames=self.frames)
