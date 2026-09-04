# The data pipeline

A dataset is a frozen dataclass in the `datasets` registry, and `load(batch=)` turns it into a `Dataset`, the value a run trains on:

```python
import dew
from dew.data import Dataset, OxfordFlowers, TokenWindows

spec = OxfordFlowers(image_size=128, val_batches=4)     # a DatasetSpec: what the data is and how it is read
assert dew.datasets["oxford_flowers102"] is OxfordFlowers
# data = spec.load(batch=32)                            # a Dataset: train, val, records, batch
```

`Dataset.train()` opens an endless shuffled stream of global batches. `Dataset.val()` opens one pass over the held-out records in a fixed order that ends by itself, and is `None` when nothing is held out. `records` is the count behind the training stream, so `steps_per_epoch` is one pass over them, and `batch` is the global batch. Image and video fields are uint8 in `[0, 255]`; tokenized text is the `{"input_ids", "attention_mask"}` dict an encoder's `tokenize` produces, under `"text"`; a token window is int32 ids under `"text"`. An objective converts pixels to `[-1, 1]` itself through `dew.inputs.unit_range`, so the dataset stays what the reader decoded.

The spec's fields are its knobs, and because a recipe's config holds the spec as a tyro subcommand, they are the recipe's flags too: `data:oxford-flowers --data.image-size 128 --data.worker-count 16`. A new dataset is a dataclass behind `@datasets(name)`; it appears on the command line with nothing else written.

## What every spec shares

`dew.data.dataset` holds the plumbing the image, video and token specs have in common:

- `local_batch(batch)` is the one place the per-process batch is computed, and it refuses a global batch the processes cannot split evenly, since a remainder would train on fewer records a step than the run says.
- `hold_out(source, records, held_out)` slices the head of a source off as the validation split and gives training the rest, so the two are disjoint by construction and FID and CLIP are never measured on records the model trained on.
- `train_stream` builds grain's shuffled, sharded, repeated stream over the training slice, with the transformations applied after `to_iter_dataset` so they run in the workers.
- `validation_pass` reads the held-out slice once in canonical order, sharded by process, and applies the random map before the per-process slice, so a record's augmentation is keyed by its global index and is the same on one host or eight.

Every process holds the same number of validation batches; the trainer confirms that before a pass and scores the minimum, so an uneven split cannot leave one host waiting in a collective.

## Determinism

Decoding, resizing and augmentation draw their randomness from the record's own generator, seeded from the spec's `seed` and the record's index. Each grain read thread runs its own copy of the augmentation pipeline, so a record gets the same pixels and the same caption at any `loading.workers`, any `loading.threads`, and any number of hosts. A failed record is dropped and counted; nothing is ever replaced with zeros.

## The specs

| Spec | Records | Notes |
| --- | --- | --- |
| `OxfordFlowers` | TFDS images, captioned from the class name through a prompt template | needs the `tfds` extra |
| `HFImages` | a Hugging Face hub dataset with a `caption` or `text` column | `name`, `split`; needs the `streaming` extra |
| `Laion12mCoco`, `CC12M`, `LaionaCoco`, `Combined30M` and the other named sets | captioned images from ArrayRecord shards on a GCS mount, one `dew.data.images.ArrayRecordImages` each | `path` is the mount |
| `LocalVideos`, `VoxCeleb2` | video clips with audio, `frames` per record | need the `av` extra |
| `TokenWindows` | `seq_len + 1` ids per record from `train.bin` and `val.bin` | written by `tools/tokenize_text.py` |
| `PackedTokens` | documents packed into rows with segment ids and positions | the language model's packed path |
| `OnlineImages`, `CombinedOnline` | images fetched from URL tables while training | no validation split; needs the `streaming` extra |

`import dew.data` registers all of them and imports none of opencv, albumentations, TFDS, `datasets` or transformers; a spec imports what it needs when it is loaded.

## Resuming mid-epoch

Grain iterators report their position through `get_state()`, and the prefetch iterator carries the position of the batch it last handed out. The trainer writes every process's position into the checkpoint's `position` entry and hands each process its own back on resume, so a resumed job continues where it stopped on every host rather than replaying the epoch from the top. A checkpoint written by two processes refuses to resume on one, with the reason, since a position is where one process's shard stopped and cannot be translated. A stream without `get_state` cannot record a position, and the trainer refuses to checkpoint one.

## Measuring it

`tools/benchmark_data.py` iterates a spec's training stream with no model attached and prints samples per second and p50/p95 step latency, with the first steps dropped so worker startup is not counted:

```bash
python tools/benchmark_data.py --steps 100 data:oxford-flowers --data.image-size 128
```

If that number is above what training reaches, the loader is not the bottleneck.
