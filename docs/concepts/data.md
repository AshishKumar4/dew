# Supplying training data

This page assumes you have done the [first training run](../getting-started.md) and know the shape and dtype your loss expects. Dew hands named arrays from a dataset to the objective. The objective decides what those fields mean.

## Start with batches you already have

A `Dataset` holds four things: a callable that opens a training iterator, an optional callable that opens a validation iterator, the number of training records if you know it, and the global batch size.

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

This example uses the same records for both splits only to show how the iterators are built. For a real validation result, use records the model does not train on. The training iterator must yield enough batches for the number of steps you ask for. The validation iterator must end after one pass.

`train` is a callable for a reason: each new or resumed run calls `train()` and gets a fresh iterator. If it returned the same, partly consumed iterator, the run could see different records. The iterator belongs to whoever opened it. Close it when you are done if it has a `close` method, and never close the shared dataset or its backing store. I have not finished reviewing iterator lifetime and cancellation, so for now run repeated, isolated smoke runs in separate processes.

## Match the objective's fields

| Use | Typical fields | Shape and dtype |
|---|---|---|
| Image diffusion | `image` | `(B, H, W, C)`, uint8 in `[0, 255]`; the objective converts to its model range |
| Video diffusion | Configured video field | Batch, frame, height, width, channel dimensions; check the selected source and objective |
| Next-token language modeling | `text` | `(B, S + 1)`, int32 token IDs; inputs and targets overlap by one position |
| Packed language modeling | `text`, `text_segment_ids`, `text_positions` | Aligned token, document, and position arrays |
| SFT | Packed token fields plus `text_roles` | Role IDs aligned with tokens; the objective selects target roles |
| Custom objective | Your field names | Whatever your initialization and loss explicitly support |

For text-conditioned diffusion, the condition encoder tokenizes the caption field and passes a conditioning value to the model. These batches are different from the token windows an autoregressive model trains on. Keep the token IDs, attention masks, padding IDs and tokenizer vocabulary consistent with the checkpoint you load.

## Use a dataset specification

Built-in sources such as `OxfordFlowers`, `TokenWindows` and `ChatMessages` are dataset specifications. `load(batch=...)` builds the data pipeline and returns a `Dataset`. Each source has its own fields for paths, tokenization, transforms and splitting.

`Loading` sets the number of Grain worker processes, read threads and buffers. The defaults start many workers (32). For a small local example, pass `Loading(workers=0)` to specifications that accept it. Raise concurrency only after you have measured the input pipeline on your own data and hardware.

Image sources can need network access the first time you use them. Token-window sources read files written by the tokenizer preparation tool. Streaming sources can depend on remote servers and may not expose an iterator position you can restore. See [recipes](../recipes.md) for entry points and [installation extras](../installation.md#add-optional-dependencies) for dependencies.

## Read a dataset a provider already holds

For data that TFDS or Hugging Face already holds, `dew.data.load("<provider>/<name>", batch=...)` returns the same kind of `Dataset` a specification returns. You must pass `preprocess(record, rng)` to turn a provider record into batch fields. There is no default, because each provider's rows have their own shape.

```python
import dew.data
from dew.data import TFDSOptions

data = dew.data.load("tfds/dew_images", batch=8, split="train", val_split="test",
                     options=TFDSOptions(path="/data/prepared"),
                     preprocess=lambda record, rng: {"image": record["image"]})
```

`split` and `val_split` take the provider's own split expressions. Dew reads the validation split in order and never shuffles it. `records` gives the record count for a source that cannot report one. `seed` sets the order and the per-record RNG. `shuffle_buffer` is how many rows a streamed split shuffles through. `Loading` only affects speed.

Provider-specific settings go in `options`: `TFDSOptions` for `tfds/...` and `HFOptions` for `hf/...`. Passing one provider's options to the other raises a `TypeError`.

`tfds/<builder>` reads the ArrayRecord files a preparation run wrote under `TFDSOptions.path`. It goes through TFDS's read-only builder, so the training process does not need TensorFlow. Dew never runs the preparation step itself. `path` is either a prepared version directory or the `data_dir` above one. In the second case, `config` and `version` name the directory inside it. Either way, Dew checks the prepared metadata against the builder, config and version you asked for. `decoders` is passed to the builder unchanged.

`hf/<name>` reads one Arrow-backed split through `datasets.load_dataset`. If the local cache does not have the dataset, that call downloads it and writes its Arrow cache. `dataset=` reads a split you have already built. `HFOptions` carries `config`, `data_files`, `features`, `storage_options` and the other `load_dataset` arguments, with that function's own types.

`streaming=True` reads an `IterableDataset` as it goes. A streamed split has no length, so `records` is `None` unless you pass it, and `Dataset.steps_per_epoch` is `None`. Processes split the rows with `datasets.distributed.split_dataset_by_node`. If there are more processes than rows, one rank would get nothing, and Dew refuses the run.

A streamed split that Dew loads by name without shuffling resumes on the record where it stopped, with the same per-record random draws. Three kinds of streamed split report no position:

- a split read with `shuffle_buffer` set, because the library does not restore the shuffle buffer;
- a split passed as `dataset=`, because it comes through transformations Dew did not apply, and an `IterableDataset` cannot say what was done to it;
- a split whose source implements no state in `datasets`.

Those runs must train with `checkpoint_every=None`. `Trainer.fit` requires it for any stream without a position.

## Epochs, batching, and packing

When the record count is known, `Dataset.steps_per_epoch` is `records` divided by the global batch size, rounded down. A batch-size ramp changes that count. When the record count is unknown, it is `None`. For a stream without a finite record count, give an explicit step target.

Packing puts tokens from several documents into rows of a fixed width. Document IDs stop attention and target scoring from crossing document boundaries. Position IDs restart according to the packing convention. Every per-token array must be sliced and packed the same way as the token IDs.

Padding and packing change the number of valid targets even when the array shapes are the same. Gradient accumulation averages the microbatch gradients. When microbatches hold different numbers of valid tokens, that average can differ from one token-mean loss over the combined batch. Check the normalization before you call two such runs equivalent.

## Resume from a data position

A restorable iterator has `get_state()` and `set_state(state)`. When the iterator has them, Dew saves its position in the checkpoint. A resumed run must use the same data order, tokenizer and transforms as the run that wrote the checkpoint.

There are two kinds of position. The kind decides whether the process count is part of what must match on resume.

| Kind | Written by | Resumes on |
|---|---|---|
| Global record count | Every record dataset built on `train_stream`: token windows, packed documents and conversations (planned over the whole corpus ahead of the shard), weighted mixtures, images, video, prompts, preference pairs | Any process count that divides the global batch |
| Shard offset | Custom iterators reporting their own state | The process count that wrote it |

A global position is the number of records the whole run has consumed. Every process reports the same number, because the stream does both the sharding and the batching. Step *k* is records `[k * batch, (k + 1) * batch)` of one shuffled order, and process *p* of *n* reads every *n*th record of that step. So a checkpoint saved by two processes restores on one or on four, and the steps after the resume are the ones an uninterrupted run would have taken. Restoring sets where the stream starts reading. It does not replay anything, so the resumed stream never reads a record twice.

The position also records the order it counts through: the source's description, its record count and the shuffle seed. Dew refuses to resume with a different record count or seed. Two corpora with the same length and seed can only be told apart if the source describes itself. A source without its own `__repr__` is named by its type, so give any source you resume across runs a description that names its data.

Which process holds which row of a step still depends on the process count. Process *q % n* reads record *q* of step *k*, and the mesh keeps each process's rows together, so the same step arrives in a different row order at a different process count. Randomness keyed by row, such as diffusion noise or sampled timesteps, therefore lands on different records. Two process counts agree on a loss only because the loss is a mean over the step's rows.

A record's own randomness does not depend on the process count. The per-record RNG is keyed by the record's place in the endless shuffled stream. Runs recorded with a Dew version that keyed it differently draw different augmentations, captions and clip starts at the same seed, even where the record order is the same.

A shard offset has no equivalent on another process count. `Checkpoints.restore` refuses one whose table was written by a different number of processes and names both counts. Resume such a run on the process count that wrote it. There is no way to convert it to a global position.

[Resuming training](../guides/checkpoints.md) shows the full save and restore path. A checkpoint that holds only the model cannot recover which records an arbitrary generator has already produced.

## Use multiple processes

`Dataset.batch` is the global batch. With several JAX processes, each process supplies its share and Dew assembles the sharded arrays. Built-in sources split records between processes themselves. A custom source has to do this on purpose, or every process trains on the same records.

Start with a single-device run and look at a batch. Then check record identity, that process slices do not overlap, sharding and resume behavior on the topology you plan to use. The [distributed guide](distributed.md) explains placement. A simulated local mesh does not test remote storage or recovery from a failed host.
