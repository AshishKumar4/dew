# The data pipeline

`dew.data` turns a dataset name into a dict the trainer can consume:

```python
{
  "train": callable_returning_an_iterator,
  "train_len": 8189,
  "val": callable_returning_an_iterator,   # when the loader has one
  "val_len": 8189,
  "local_batch_size": 32,
  "global_batch_size": 32,
}
```

`ObjectiveTrainer.fit` calls `data["train"]()`, wraps it in a `DevicePrefetchIterator`, and reads `local_batch_size` and `global_batch_size` for the throughput numbers.

## Source and augmenter

Two abstractions, in `dew.data.sources.base`:

- `DataSource.get_source(path_override)` returns something with `__len__` and `__getitem__`, which is all grain's `IndexSampler` needs. TFDS datasets, ArrayRecord shards on a GCS mount, local video trees and VoxCeleb2 are all sources.
- `DataAugmenter.create_transform(**kwargs)` returns a callable that builds a `pygrain.MapTransform`: decode, resize, augment, and produce the keys the model reads.

`MediaDataset` pairs one of each and records whether it is image or video. `dew.data.registry` holds the names: `datasetMap` for the image loader, `onlineDatasetMap` for the streaming one, `mediaDatasetMap` for the unified media loader.

## Grain loaders

`get_dataset_grain(data_name, batch_size, image_scale, ...)` is the image path, `get_media_dataset_grain(data_name, ..., sequence_length=N)` handles images or video with one source and one augmenter per dataset.

Both build a `pygrain.IndexSampler` with `ShardByJaxProcess`, so each process reads its own shard, and batch to `batch_size // jax.process_count()` with `drop_remainder=True`. Worker processes, read threads and buffer sizes are all arguments; the defaults suit a machine with a fast disk and many cores.

Validation reads the same records in canonical order with a separate unshuffled sampler. For the media loader, `val_count` holds out the first N records as a disjoint slice, and the train loader covers the rest.

## Streaming

`get_dataset_online` builds an `OnlineStreamingDataLoader` over Hugging Face `datasets` URLs, fetching and decoding images or videos in worker threads. It needs the `streaming` extra. `dew.data` imports lazily through a module `__getattr__`, so a training run that only uses grain never pays for that stack, and a host without opencv or PyAV can still `import dew.data`.

## Resuming mid-epoch

Grain iterators report their position through `get_state()`, and the prefetch iterator carries the position of the batch it last handed out. The trainer writes that into the checkpoint and passes it back as `source_state` on the next run, so a resumed job continues where it stopped rather than replaying the epoch from the top. An iterator that cannot report a position simply has no `dataset_state` in the checkpoint.

## Measuring it

`tools/benchmark_data.py` iterates a loader with no model attached and prints samples per second and p50/p95 step latency, with the first steps dropped so worker startup is not counted:

```bash
python tools/benchmark_data.py --dataset oxford_flowers102 --batch-size 32 --steps 100
```

If that number is above what training reaches, the loader is not the bottleneck.
