"""Images streamed by URL from Hugging Face datasets of (url, caption) rows."""

from __future__ import annotations

import dataclasses

import jax

from dew.registry import datasets

from .dataset import Dataset, DatasetSpec, Loading, Tokenize, local_batch, tokenized


@dataclasses.dataclass(frozen=True)
class OnlineImages(DatasetSpec):
    """Fetches images by url as they are read, an endless stream.

    `sources` name hub datasets or `gs://` directories saved with
    `save_to_disk`, whose rows carry a url and a caption. The rows are
    concatenated, shuffled once and sharded by JAX process. Each process then
    walks its shard forever, reshuffling between passes, so nothing is held
    out and the stream cannot resume mid-epoch.

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

    def load(self, *, batch: int, tokenize: Tokenize | None = None) -> Dataset:
        if not self.sources:
            raise ValueError(f"{type(self).__name__} needs sources= set to one or more datasets")
        from .online_loader import ImageStream, load_rows

        rows = load_rows(self.sources)
        per_process = local_batch(batch)

        def stream():
            return ImageStream(
                rows.shard(num_shards=jax.process_count(), index=jax.process_index()),
                batch=per_process, size=self.image_size, min_size=self.min_image_size,
                workers=self.loading.workers, threads=self.loading.threads,
                timeout=self.timeout, retries=self.retries,
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
