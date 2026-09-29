"""Images and video clips streamed by URL from Hugging Face datasets of
(url, caption) rows."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Iterator

from dew.registry import datasets

from .dataset import Batch, DataPartition, Dataset, DatasetSpec, Loading, Tokenize, tokenized

if TYPE_CHECKING:
    from .online_loader import Fetch


@dataclasses.dataclass(frozen=True)
class OnlineImages(DatasetSpec):
    """Fetches images by url as they are read, an endless stream.

    `sources` name hub datasets or `gs://` directories saved with
    `save_to_disk`, whose rows carry a url and a caption. The rows are
    concatenated, shuffled once and sharded by the reader's share. Each
    reader then walks its shard forever, reshuffling between passes, so
    nothing is held out and the stream cannot resume mid-epoch. What a share
    yields is whichever fetches succeed first, so no two processes can read
    one share alike, and a partition whose shares have several readers is
    refused.

    A row is dropped and counted when its url yields no image, or when the
    image is not RGB, under `min_image_size` on its shorter side, more than
    2.4 times as long as wide, or a single flat colour. Needs the streaming
    extra (HF `datasets`).
    """

    sources: tuple[str, ...] = ()
    image_size: int = 256
    min_image_size: int = 128
    loading: Loading = dataclasses.field(
        default=Loading(workers=16, threads=512, worker_buffer=20), kw_only=True)
    """How the fetch pool runs, which is this spec's own and not a grain
    reader. `workers` and `threads` are the pool's, `worker_buffer` is how
    many batches the fetchers run ahead, and `read_buffer` does not reach
    this path. The grain default is shaped for file reads rather than for a
    pool waiting on urls."""
    timeout: int = 15
    retries: int = 3

    def fetch(self) -> Fetch:
        """How a fetcher turns one of this spec's urls into a sample."""
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


@datasets("combined_online")
@dataclasses.dataclass(frozen=True)
class CombinedOnline(OnlineImages):
    """Reads every url dataset in the dew-datasets-regional bucket.

    The liked sets are listed several times over, which weights them up.
    """

    sources: tuple[str, ...] = (
        "gs://dew-datasets-regional/datasets/laion-aesthetics-12m+mscoco-2017",
        "gs://dew-datasets-regional/datasets/coyo700m-aesthetic-5.4_25M",
        "gs://dew-datasets-regional/datasets/leonardo-liked-1.8m",
        "gs://dew-datasets-regional/datasets/leonardo-liked-1.8m",
        "gs://dew-datasets-regional/datasets/leonardo-liked-1.8m",
        "gs://dew-datasets-regional/datasets/cc12m",
        "gs://dew-datasets-regional/datasets/playground-liked",
        "gs://dew-datasets-regional/datasets/leonardo-liked-1.8m",
        "gs://dew-datasets-regional/datasets/leonardo-liked-1.8m",
        "gs://dew-datasets-regional/datasets/cc3m",
        "gs://dew-datasets-regional/datasets/cc3m",
        "gs://dew-datasets-regional/datasets/laion2B-en-aesthetic-4.2_37M",
    )


@datasets("online_videos")
@dataclasses.dataclass(frozen=True)
class OnlineVideos(OnlineImages):
    """Fetches video clips by url as they are read, an endless stream.

    Each url's video is decoded the way `VideoDataset` reads a local clip:
    `frames` consecutive frames at 25 fps from a random start, each resized
    to an `image_size` square, and no audio. A row is dropped and counted
    when its url yields no video `frames` long, or when the video is under
    `min_image_size` on its shorter side. Needs the streaming extra and the
    `av` extra.
    """

    frames: int = 16

    def fetch(self) -> Fetch:
        return dataclasses.replace(super().fetch(), frames=self.frames)
