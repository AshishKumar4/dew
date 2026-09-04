"""Images streamed by URL from Hugging Face datasets of (url, caption) rows."""

from __future__ import annotations

import dataclasses

import jax

from dew.registry import datasets

from .dataset import Dataset, DatasetSpec, Loading, local_batch, tokenized


@dataclasses.dataclass(frozen=True)
class OnlineImages(DatasetSpec):
    """Images fetched by URL as they are read, an endless stream.

    `sources` name hub datasets or `gs://` directories saved with
    `save_to_disk`; their rows carry a url and a caption. The rows are
    concatenated, shuffled once, and sharded by JAX process; each process
    then walks its shard forever, reshuffling between passes and dropping
    fetches that fail, so nothing is held out and the stream cannot resume
    mid-epoch. Needs the streaming extra (HF `datasets`).
    """

    sources: tuple[str, ...] = ()
    image_size: int = 256
    min_image_size: int = 128
    """Rows whose image is smaller than this on a side are dropped."""
    loading: Loading = Loading(workers=16, threads=512)
    """The fetch pool has no grain reader, so `read_buffer` does not reach
    this path; `worker_buffer` is the batches prefetched ahead of the run."""
    timeout: int = 15
    retries: int = 3

    def load(self, *, batch: int, tokenize=None) -> Dataset:
        from .online_loader import OnlineStreamingDataLoader, load_rows

        if not self.sources:
            raise ValueError(f"{type(self).__name__} needs sources= set to one or more datasets")
        rows = load_rows(list(self.sources))
        per_process = local_batch(batch)

        def stream():
            return OnlineStreamingDataLoader(
                rows,
                batch_size=per_process,
                num_workers=self.loading.workers,
                num_threads=self.loading.threads,
                image_shape=(self.image_size, self.image_size),
                min_image_shape=(self.min_image_size, self.min_image_size),
                global_process_count=jax.process_count(),
                global_process_index=jax.process_index(),
                prefetch=self.loading.worker_buffer,
                timeout=self.timeout,
                retries=self.retries,
            )

        return Dataset(train=tokenized(stream, tokenize), val=None,
                       records=len(rows), batch=batch)


@datasets("combined_online")
@dataclasses.dataclass(frozen=True)
class CombinedOnline(OnlineImages):
    """The dew-datasets-regional bucket's url datasets, the liked sets several times over."""

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
