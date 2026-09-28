# Training data

A `Dataset` supplies the batches a run trains and validates on. A batch is a dictionary of NumPy arrays whose first dimension is this process's rows of the global batch (all of them with one process); the trainer joins the processes' rows into global arrays, and the objective reads the fields it needs by name. This page covers the `Dataset` class, the fields each built-in objective expects, the built-in readers, reading data that TFDS or Hugging Face already holds, and resuming the data stream from a checkpoint.

![Dataset to global batch: train(partition) opens an iterator of host batches, and shard_batch assembles each into one jax.Array split over the mesh's batch axes.](../assets/data-pipeline-light.svg)
![](../assets/data-pipeline-dark.svg)

## Dataset

```python
import itertools

import numpy as np
from dew.data import DataPartition, Dataset

x = np.arange(16, dtype=np.float32).reshape(8, 2)
y = x.sum(axis=1, keepdims=True)
batch = {"features": x, "target": y}
data = Dataset(train=lambda partition: itertools.repeat(batch),
               val=lambda partition: iter([batch]), records=8, batch=8)

first = next(data.train(DataPartition()))
print({name: value.shape for name, value in first.items()})
print(data.steps_per_epoch)
```

```text
{'features': (8, 2), 'target': (8, 1)}
1
```

| Field | Meaning |
|---|---|
| `train(partition)` | Returns a new iterator of training batches. It must yield at least as many batches as the run takes steps. |
| `val(partition)` | Returns a new iterator over one pass of the validation records, which then ends. `None` means no validation split. |
| `records` | The number of training records, or `None` when unknown. |
| `batch` | The global batch size. |

The example uses the same records for both splits only to show the two iterators; a validation result needs records the model does not train on.

`train` and `val` are functions rather than iterators so that every new or resumed run opens a fresh iterator. The iterator belongs to the caller that opened it: close it after use if it has a `close` method, and never close the dataset or its backing store.

The argument is a `DataPartition`, the share of every global batch this process reads. `DataPartition()` reads every row, which is correct for a single process. With several processes the trainer asks the mesh for each process's share (`dew.training.distributed.data_partition`), and the built-in readers read only that share.

`Dataset.steps_per_epoch` is `records // batch`, or `None` when `records` is `None`. A stream without a record count needs an explicit `steps` in `fit`.

## Fields per objective

| Objective | Fields | Shape and dtype |
|---|---|---|
| Image diffusion | `image` | `(B, H, W, C)` uint8 in `[0, 255]`; the objective converts to its model's range |
| Video diffusion | The configured video field | `(B, T, H, W, C)`; check the source and the objective |
| Next-token language modeling | `text` | `(B, S + 1)` int32 token ids; inputs and targets overlap by one position |
| Packed language modeling | `text`, `text_segment_ids`, `text_positions` | Token, document and position arrays of the same shape |
| SFT | Packed token fields and `text_roles` | Role ids aligned with the tokens; the objective scores the target roles |
| Custom objective | Your own names | Whatever your `init` and `loss` read |

For text-conditioned diffusion, the condition encoder tokenizes the caption field and passes the encoding to the model. Keep token ids, attention masks, padding ids and the tokenizer vocabulary consistent with the checkpoint you load.

## Dataset specifications

The built-in readers are dataset specifications: frozen dataclasses that describe a dataset, whose `load(batch=...)` builds the reading pipeline and returns a `Dataset`. `TokenWindows` reads fixed windows of `seq_len + 1` token ids from a tokenized directory:

```python
import json
import tempfile
from pathlib import Path

import numpy as np
from dew.data import DataPartition, Loading, TokenWindows

corpus = Path(tempfile.mkdtemp())
np.arange(1000, dtype=np.uint16).tofile(corpus / "train.bin")
np.arange(1000, 1200, dtype=np.uint16).tofile(corpus / "val.bin")
(corpus / "meta.json").write_text(json.dumps({"vocab_size": 1200}))

data = TokenWindows(path=str(corpus), seq_len=8, loading=Loading(workers=0)).load(batch=4)
windows = next(data.train(DataPartition()))
print(windows["text"].shape, windows["text"].dtype, data.records)
print(windows["text"][0])
```

```text
(4, 9) int32 124
[392 393 394 395 396 397 398 399 400]
```

The training stream is shuffled: the first window of the first batch is window 49 of the corpus. `records` is the number of windows, `(1000 - 1) // 8 = 124`.

Each window starts `seq_len` ids after the previous one, so the last id of one window is the first of the next. `tools/tokenize_text.py` writes `train.bin`, `val.bin` and `meta.json` from raw text.

Other specifications include `OxfordFlowers`, `HFImages`, `DocumentChunks`, `ChatMessages` and the video and preference readers; the [API reference](../reference/core-api.md) lists them. Each has its own fields for paths, tokenization, transforms and splits, and two fields every specification shares:

| Field | Meaning |
|---|---|
| `seed` | The record order and the key of each record's random draws (augmentation, captions, clip starts) |
| `loading` | A `Loading` with Grain's throughput settings |

`Loading` changes how fast records are read, never which records a run sees or what is in them:

| `Loading` field | Default | Counts |
|---|---|---|
| `workers` | 32 | Grain worker processes; `0` reads in the training process |
| `threads` | 64 | Record reads one worker keeps in flight |
| `read_buffer` | 128 | Records one worker reads ahead |
| `worker_buffer` | 2 | Batches one worker holds ready |

Use `Loading(workers=0)` for small local runs, and raise the settings only after measuring the input pipeline on your data and hardware.

Image sources can need network access the first time. Token-window sources read files written by `tools/tokenize_text.py`. Streaming sources can depend on remote servers and may have no position to restore. [Recipes](../recipes.md) lists the command-line entry points and [Installation](../installation.md#optional-extras) the extras each source needs.

## TFDS and Hugging Face datasets

`dew.data.load("<provider>/<name>", batch=...)` reads a dataset that TFDS or Hugging Face already holds and returns a `Dataset`. `preprocess(record, rng)` turns one provider record into batch fields and is required, because every provider's rows have their own structure. `dataset=` takes a Hugging Face split that is already in memory:

```python
import datasets
import numpy as np
import dew.data

table = datasets.Dataset.from_dict({"x": [[float(i), float(i)] for i in range(32)],
                                    "label": list(range(32))})
data = dew.data.load("hf/toy", batch=8, dataset=table, loading=dew.data.Loading(workers=0),
                     preprocess=lambda record, rng: {"x": np.asarray(record["x"], np.float32),
                                                     "label": np.int32(record["label"])})
first = next(data.train(dew.data.DataPartition()))
print(first["x"].shape, first["label"][:4], data.records)
```

```text
(8, 2) [26  4 10 30] 32
```

| Argument | Meaning |
|---|---|
| `split`, `val_split` | The provider's own split expressions. The validation split is read in order and never shuffled. |
| `records` | The record count, for a source that cannot report one |
| `seed` | The record order and the per-record random key |
| `shuffle_buffer` | Rows a streamed Hugging Face split shuffles through |
| `options` | `TFDSOptions` for `tfds/...`, `HFOptions` for `hf/...`; the other provider's options raise `TypeError` |
| `loading` | Throughput only, as above |

`tfds/<builder>` reads the ArrayRecord files a TFDS preparation run wrote under `TFDSOptions.path`, through TFDS's read-only builder, so the training process does not import TensorFlow. Dew never prepares the data itself ([Installation](../installation.md#preparing-tfds-data) shows how). `path` is either a prepared version directory or the `data_dir` above one; in the second case `config` and `version` select the directory inside it. Dew checks the prepared metadata against the builder, config and version requested. `decoders` is passed to the builder unchanged.

<!-- not run: needs a prepared TFDS directory -->
```python
import dew.data
from dew.data import TFDSOptions

data = dew.data.load("tfds/dew_images", batch=8, split="train", val_split="test",
                     options=TFDSOptions(path="/data/prepared"),
                     preprocess=lambda record, rng: {"image": record["image"]})
```

`hf/<name>` reads one Arrow-backed split through `datasets.load_dataset`, which downloads the dataset and writes its Arrow cache when the local cache does not have it. `HFOptions` carries `config`, `data_files`, `features`, `storage_options` and the other `load_dataset` arguments with that function's types.

`streaming=True` reads an `IterableDataset` as it goes. A streamed split has no length, so `records` is `None` unless given and `Dataset.steps_per_epoch` is `None`. Processes split the rows with `datasets.distributed.split_dataset_by_node`; when there are more processes than rows, one process would get none, and Dew refuses the run.

A streamed split loaded by name without shuffling resumes at the record where it stopped, with the same per-record random draws. Three kinds of streamed split have no position to resume from, and must train with `checkpoint_every=None` (`Trainer.fit` refuses otherwise):

- a split read with `shuffle_buffer` set, because `datasets` does not restore the shuffle buffer;
- a split passed as `dataset=`, because it may carry transformations Dew did not apply;
- a split whose source implements no state in `datasets`.

## Packing

Packing places tokens from several documents into rows of a fixed width. Segment ids (`text_segment_ids`) stop attention and target scoring from crossing document boundaries, and position ids (`text_positions`) restart at each document. Every per-token array must be sliced and packed the same way as the token ids.

Padding and packing change the number of valid targets even when array shapes are equal. Gradient accumulation adds the loss totals and masses of its microbatches before dividing, so the accumulated gradient is that of one token mean over the whole window; check the normalization before calling two runs with different packing equivalent.

## Resuming the data stream

An iterator with `get_state()` and `set_state(state)` is restorable: Dew saves its position in every checkpoint and restores it on resume. The resumed run must use the same data order, tokenizer and transforms as the run that wrote the checkpoint. There are two kinds of position:

| Kind | Written by | Resumes on |
|---|---|---|
| Global record count | Every reader built on `train_stream`: token windows, packed documents and conversations, weighted mixtures, images, video, prompts, preference pairs | Any process count that divides the global batch |
| Shard offset | Custom iterators that report their own state | Only the process count that wrote it |

A global position is the number of records the whole run has consumed, and every process reports the same number. Step *k* reads records `[k * batch, (k + 1) * batch)` of one shuffled order, and process *p* of *n* reads every *n*-th record of that step. A checkpoint written by two processes therefore restores on one or on four, and the steps after the resume are the ones an uninterrupted run would have taken. Restoring sets where the stream starts reading; no record is read twice.

The position also records what it counts through: the source's description, its record count and the shuffle seed. Dew refuses to resume with a different record count or seed. A source without its own `__repr__` is described by its type name, so two corpora of equal length and seed are only told apart if the source describes its data; give any source you resume across runs a `__repr__` that names its data.

Which process holds which row of a step depends on the process count, so randomness keyed by row, such as diffusion noise or sampled timesteps, falls on different records at a different process count. The resumed run then draws different noise for the same records: it optimizes the same objective, but its losses are not the same numbers an uninterrupted run would have logged. A record's own random draws are keyed by its place in the shuffled stream and do not depend on the process count.

`Checkpoints.restore` refuses a shard offset written by a different number of processes and names both counts; resume such a run on the process count that wrote it. [Checkpoints](../guides/checkpoints.md) shows the whole save and restore path.

## Multiple processes

`Dataset.batch` is the global batch. With several JAX processes each process reads its share and Dew assembles the global arrays with `jax.make_array_from_process_local_data`. The built-in readers split records between processes themselves. A custom `train` function must read only the share its `partition` names, `partition.index` of `partition.count`, or every process trains on the same records.

Before a multi-process run, check on the target topology that process shares do not overlap, that sharding is as expected and that a resume continues the stream. [Distributed training](distributed.md) describes placement.
