# Supplying training data

This guide assumes the [first training run](../getting-started.md). You should know the shape and dtype your loss expects. Dew passes named arrays from a dataset to the objective; the objective decides what those fields mean.

## Start with batches you already have

A `Dataset` contains a callable opening a training iterator, an optional callable opening a validation iterator, the number of training records when known, and a global batch size.

```python
import itertools
import numpy as np
from dew.data import Dataset

x = np.arange(16, dtype=np.float32).reshape(8, 2)
y = x.sum(axis=1, keepdims=True)
batch = {"features": x, "target": y}
data = Dataset(train=lambda: itertools.repeat(batch),
               val=lambda: iter([batch]), records=8, batch=8)
first = next(data.train())
assert first["features"].shape == (8, 2)
np.testing.assert_array_equal(first["target"], y)
```

This example uses the same records for both splits only to demonstrate iterator construction. Use disjoint records for a real validation result. A training iterator should provide enough batches for the requested number of steps. A validation iterator must end after one pass.

The callable matters: `train()` must open a fresh iterator for a new or resumed run. Returning the same partially consumed iterator can change which records the run sees. Iterator lifetime and cancellation are under review; use separate processes for repeated isolated smoke runs until that issue is resolved.

## Match the objective's fields

| Use | Typical fields | Shape and dtype |
|---|---|---|
| Image diffusion | `image` | `(B, H, W, C)`, uint8 in `[0, 255]`; the objective converts to its model range |
| Video diffusion | Configured video field | Batch, frame, height, width, channel dimensions; check the selected source and objective |
| Next-token language modeling | `text` | `(B, S + 1)`, int32 token IDs; inputs and targets overlap by one position |
| Packed language modeling | `text`, `text_segment_ids`, `text_positions` | Aligned token, document, and position arrays |
| SFT | Packed token fields plus `text_roles` | Role IDs aligned with tokens; the objective selects target roles |
| Custom objective | Your field names | Whatever your initialization and loss explicitly support |

For text-conditioned diffusion, the condition encoder tokenizes the caption field and supplies a conditioning value to the model. These are not the same batches as autoregressive token windows. Keep token IDs, attention masks, padding IDs, and the tokenizer's vocabulary consistent with the selected checkpoint.

## Use a dataset specification

Built-in sources such as `OxfordFlowers`, `TokenWindows`, and `ChatMessages` are dataset specifications. Calling `load(batch=...)` prepares their data pipeline and returns a `Dataset`. Source-specific fields describe paths, tokenization, transforms, and splitting.

`Loading` configures Grain worker processes, read threads, and buffers. The defaults can start many workers; for a small local example use `Loading(workers=0)` on specifications that accept it. Increase concurrency only after measuring the input pipeline on your data and hardware.

Image sources can require network access on first use. Token-window sources read files created by the tokenizer preparation tool. Streaming sources may depend on remote servers and may not expose a restorable iterator position. See [recipes](../recipes.md) for entry points and [installation extras](../installation.md#add-optional-dependencies) for dependencies.

## Read a dataset a provider already holds

`dew.data.load("<provider>/<name>", batch=...)` returns the same `Dataset` a specification returns, for data TFDS or Hugging Face already holds. `preprocess(record, rng)` is required to turn a provider record into batch fields; there is no default, because a provider's rows are its own shape.

```python
import dew.data

data = dew.data.load("tfds/dew_images", batch=8, split="train", val_split="test",
                     path="/data/prepared", preprocess=lambda record, rng: {
                         "image": record["image"]})
```

`split` and `val_split` take the provider's own split expressions; the validation pass is read in split order and never shuffled. `records` supplies the record count for a source that cannot report one. `seed` keys the order and the per-record RNG, and `shuffle_buffer` is how many rows a streamed split shuffles through. `Loading` remains performance only.

`tfds/<builder>` reads the ArrayRecords a preparation run wrote under `path`, through TFDS's read-only builder, so the training process needs no TensorFlow. Preparation is never run here. `path` is either a prepared version directory or the `data_dir` above one, in which case `config` and `version` name the directory inside it; either way the prepared metadata is compared with the builder, config and version requested. `decoders` reaches the builder unchanged.

`hf/<name>` reads one Arrow-backed split through `datasets.load_dataset`, which downloads the dataset and writes its Arrow cache when the local cache holds neither. `dataset=` reads a split the caller already built. `config`, `data_files`, `features`, `storage_options` and the rest of that function's arguments are forwarded with its own types.

`streaming=True` reads an `IterableDataset` as it goes. Such a split has no length, so `records` is `None` unless supplied, and `Dataset.steps_per_epoch` is `None`. Processes share the rows through `datasets.distributed.split_dataset_by_node`; a pool with more processes than the split has rows leaves a rank empty and is refused. A streamed split dew loads by name, unshuffled, resumes on the record it stopped at with the same per-record draws. Three kinds report no position instead: one read with `shuffle_buffer` set, because the buffer the shuffle drew from is not in what the library restores; one handed over as `dataset=`, because it arrives through transformations dew did not apply and an `IterableDataset` cannot be asked what it has been through; and one whose source implements no state in `datasets`. Those runs train with `checkpoint_every=None`, which `Trainer.fit` requires of any stream without a position.

## Epochs, batching, and packing

`Dataset.steps_per_epoch` uses integer division of `records` by the global batch size when a record count is available. It returns `None` when the count is unknown. Use an explicit step target for a stream without a finite record count.

Packing combines tokens from several documents into fixed-width rows. Document IDs prevent attention and target scoring from crossing document boundaries; position IDs restart according to the packing convention. All per-token metadata must follow the same slicing and packing as the token IDs.

Padding and packing affect the number of valid targets, even when array shapes match. The current gradient accumulation path averages microbatch gradients, so unequal valid-token counts can differ from one combined token-mean loss. Do not claim those runs have equivalent effective batches without checking the normalization contract.

## Resume from a data position

A restorable iterator implements `get_state()` and `set_state(state)`. Dew checkpoints the consumed iterator position when available. A resume requires the same data ordering, tokenizer, and transforms as the model checkpoint.

A position is one of two kinds, and the kind decides whether the process count is part of the resume contract.

| Kind | Written by | Resumes on |
|---|---|---|
| Global record count | Every record dataset built on `train_stream`: token windows, images, video, prompts, preference pairs | Any process count that divides the global batch |
| Shard offset | `PackedTokens` and `ChatMessages`, whose windows are packed out of one process's own documents, and custom iterators reporting their own state | The process count that wrote it |

A global position is the number of records the whole run has consumed. Every process reports the same number, because the stream owns both the sharding and the batching: step *k* is records `[k * batch, (k + 1) * batch)` of one shuffled order, and process *p* of *n* reads every *n*th record of that step. A checkpoint saved by two processes therefore restores on one or on four, and the steps after the resume are the steps an uninterrupted run would have taken. Restoring is a slice bound, not a replay: the resumed stream reads nothing it has already read.

The position also records the order it counts into: the source's description, its record count, and the shuffle seed. A resume against a different record count or a different seed is refused. Two corpora of the same length and seed are told apart only when the source describes itself; a source without its own `__repr__` is named by type, so give a source you resume across runs a description that names its data.

Which process holds which row of a step still depends on the process count. Record *q* of step *k* is read by process *q % n*, and the mesh lays each process's rows out together, so the same step arrives in a different row order at a different process count. Row-keyed randomness such as diffusion noise or sampled timesteps therefore lands on different records, and two process counts agree on a loss only because it is a mean over the step's rows.

A record's own randomness does not depend on the count: the per-record RNG is keyed by the record's place in the endless shuffled stream. That key changed with the global position, so at a given seed the augmentations, captions, and clip starts of a run differ from those of a run recorded before the change, even where the record order is identical.

A shard offset has no equivalent on another process count, and `Checkpoints.restore` refuses one whose table was written by a different number of processes, naming both counts. Resume such a run on the count that wrote it; there is no conversion to a global position.

[Resuming training](../guides/checkpoints.md) shows the complete save/restore path. A model-only checkpoint cannot recover records consumed by an arbitrary generator.

## Use multiple processes

`Dataset.batch` is global. With multiple JAX processes, each process supplies its share of the global batch; Dew assembles the sharded arrays. Built-in sources handle their process partitioning. A custom source must define it deliberately to avoid training on duplicated records.

Start with a single-device run and inspect the batch. Then validate record identity, disjoint process slices, sharding, and resume behavior on the intended topology. The [distributed guide](distributed.md) explains placement; a local simulated mesh does not validate remote storage or cross-host failure recovery.
