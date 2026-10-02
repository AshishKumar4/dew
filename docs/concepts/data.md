# Training data

A `Dataset` supplies the batches a run trains and validates on. A batch is a dictionary of arrays whose first dimension is this process's rows of the global batch (all of them with one process). Most readers yield NumPy arrays; device image augmentation yields resident JAX pixels. The trainer joins the processes' rows into global arrays, and the objective reads the fields it needs by name. This page covers the `Dataset` class, the fields each built-in objective expects, the built-in readers, reading data that TFDS or Hugging Face already holds, and resuming the data stream from a checkpoint.

![Dataset to global batch: train(partition) opens an iterator of host batches, and shard_batch assembles each into one jax.Array split over the mesh's batch axes.](../assets/data-pipeline-light.svg)
![Dataset to global batch: train(partition) opens an iterator of host batches, and shard_batch assembles each into one jax.Array split over the mesh's batch axes.](../assets/data-pipeline-dark.svg)

## Dataset

Records you already hold in memory become a `Dataset` through `Dataset.from_records`:

```python
import numpy as np
from dew.data import DataPartition, Dataset

x = np.arange(32, dtype=np.float32).reshape(16, 2)
y = x.sum(axis=1, keepdims=True)
data = Dataset.from_records({"features": x, "target": y}, batch=8, seed=0,
                            validation={"features": x[:8], "target": y[:8]})

first = next(data.train(DataPartition()))
print({name: value.shape for name, value in first.items()})
print(data.records, data.steps_per_epoch)
```

```text
{'features': (8, 2), 'target': (8, 1)}
16 2
```

`from_records` takes a mapping of columns whose first axis is the record, as here, a list of per-record mappings, or any source with `__len__` and `__getitem__`. Training reshuffles the records from `seed` every epoch, reads each one once per epoch, and saves its position in a checkpoint, so a resumed run reads the records it had not reached. With several processes, each reads its own share of every batch. `validation` is read once in order, in whole batches. The example validates on training records only to show the argument; a validation result needs records the model does not train on.

Every reader returns the same `Dataset` value:

| Field | Meaning |
|---|---|
| `train(partition)` | Returns a new iterator of training batches. It must yield at least as many batches as the run takes steps. |
| `val(partition)` | Returns a new iterator over one pass of the validation records, which then ends. `None` means no validation split. |
| `records` | The number of training records, or `None` when unknown. |
| `batch` | The global batch size. |

`train` and `val` are functions rather than iterators so that every new or resumed run opens a fresh iterator. The iterator belongs to the caller that opened it: close it after use if it has a `close` method, and never close the dataset or its backing store. `Trainer.fit` closes the iterators it opens, whether the run finishes or fails, and a training step's exception kept after the run does not keep its closed prefetch iterator alive.

The argument is a `DataPartition`, the share of every global batch this process reads. `DataPartition()` reads every row, which is correct for a single process. With several processes the trainer asks the mesh for each process's share (`dew.training.distributed.data_partition`), and the built-in readers read only that share.

A `Dataset` can also be built from the two functions directly, for a stream no reader covers:

```python
import itertools

import numpy as np
from dew.data import Dataset

x = np.arange(16, dtype=np.float32).reshape(8, 2)
batch = {"features": x, "target": x.sum(axis=1, keepdims=True)}
data = Dataset(train=lambda partition: itertools.repeat(batch), val=None, records=8, batch=8)
```

Such a function has to do what `from_records` does for you: read only the share its partition names, and give its iterator `get_state` and `set_state` if the run checkpoints (see [Resuming the data stream](#resuming-the-data-stream)). This one does neither, so it suits a single-process run with `checkpoint_every=None`. `Dataset.from_grain` takes a Grain pipeline you built yourself.

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

Each window starts `seq_len` ids after the previous one, so the last id of one window is the first of the next. `dew tokenize` (or `dew.data.write_tokens` in Python) writes `train.bin`, `val.bin` and `meta.json` from raw text; see [Packing](#packing).

Other specifications include `PackedTokens`, `OxfordFlowers`, `HFImages`, `ChatMessages` and the video and preference readers; the [API reference](../reference/core-api.md) lists them. Each has its own fields for paths, tokenization, transforms and splits, and two fields every specification shares:

| Field | Meaning |
|---|---|
| `seed` | The record order and the key of each record's random draws (augmentation, captions, clip starts) |
| `loading` | A `Loading` with Grain's throughput settings |

`Loading` changes how fast records are read, never which records a run sees or what is in them:

| `Loading` field | Default | Counts |
|---|---|---|
| `workers` | 0 | Grain worker processes; `0` reads in the training process |
| `threads` | 64 | Record reads one worker keeps in flight |
| `read_buffer` | 128 | Records one worker reads ahead |
| `worker_buffer` | 2 | Batches one worker holds ready |

The default reads in the training process, as Grain's own default does. Worker processes each import the program again, so they cost seconds and memory before the first batch, and pay off only when decoding or augmentation outruns the reading threads. Raise `workers` only after measuring the input pipeline on your data and hardware. A script that starts workers needs an `if __name__ == "__main__":` guard, because each worker imports the script.

Image sources can need network access the first time. Token-window sources read files written by `dew tokenize`. Streaming sources can depend on remote servers and may have no position to restore. [Recipes](../recipes.md) lists the command-line entry points and [Installation](../installation.md#optional-extras) the extras each source needs. `OnlineImages` and `OnlineVideos` (`data:online-videos` in a recipe) stream Hugging Face tables of urls and captions. `OnlineVideos` decodes each url's video the way `LocalVideos` reads a file: `frames` consecutive frames at 25 fps, resized to `image_size` squares, without audio.

## Image datasets on the Hugging Face Hub

`HFImages` reads a Hub image dataset by index through the image pipeline: decode, resize to `image_size`, augmentation, and captions for text conditioning. Its column fields say where a record keeps its fields. `image_column` holds the image, the caption is the first of `caption_columns` a record has, and `label_column` is the class index a record carries as `label`. CIFAR-10 keeps its image under `img`, a class under `label` and no caption:

<!-- not run: downloads CIFAR-10 on first use -->
```python
from dew.data import DataPartition, HFImages

data = HFImages(name="uoft-cs/cifar10", image_column="img", caption_columns=(),
                image_size=32, augmentation="flip_only", val_split="test",
                val_batches=None).load(batch=64)
batch = next(data.train(DataPartition()))
print({name: (value.shape, value.dtype) for name, value in batch.items()})
print(data.records, data.steps_per_epoch)
```

```text
{'image': ((64, 32, 32, 3), dtype('uint8')), 'label': ((64,), dtype('int32'))}
50000 781
```

`caption_columns=()` reads a dataset without captions, for an unconditional or class-conditional run, and refuses a caption reader. A column the split does not hold is refused when the spec loads, with the columns it does hold. Without `val_split`, `val_batches` batches are held out of the head of the training split; `val_batches=None` with a `val_split` scores the whole named split.

Validation reads each image through the deterministic resize, without the crop, flip and jitter training applies, so a metric scores the images a reference implementation would. `augment_validation=True` applies the training augmentation to validation too, with draws that repeat on every pass. A run record written before this field existed reads it as on, which is what those runs did.

## Device image augmentation

`OxfordFlowers`, `HFImages` and the prepared `ArrayRecordImages` readers share
`ImageDataset`'s transforms. Set `augmentation_backend="device"` to apply
random crop/resize, horizontal flip and colour jitter to each decoded batch
with JAX. The default remains `"host"`, preserving existing runs' OpenCV/NumPy
augmentation. `augmentation="flip_only"` disables colour jitter and `"none"`
keeps the deterministic resize, without random cropping.

<!-- not run: needs a prepared Oxford Flowers version directory -->
```python
from dew.data import Loading, OxfordFlowers

data = OxfordFlowers(
    path="data/oxford_flowers102/2.1.1",
    image_size=128,
    augmentation_backend="device",
    augmentation="flip_jitter",
    crop_scale=(0.6, 1.0),
    augmentation_size=160,
    loading=Loading(workers=0, threads=4),
).load(batch=32)
```

`crop_scale` is the retained area fraction, drawn uniformly and applied as an
integer crop of the staging square; both sides shrink by its square root.
`(1.0, 1.0)` keeps the full image. `augmentation_size=None` stages at
`image_size`; a larger size retains more pixels for crop/resize and increases
transfer and device memory. Decode, the initial area/cubic resize into a dense
square, and caption tokenization still run on the host. The variable-sized
JPEGs cannot be stacked directly, and neither JAX nor this option decodes
them. The subsequent crop's bilinear resize, flip and jitter run on device,
with one final uint8 round/clip. The host path is also needed by CPU consumers
and by `OnlineImages`/video decoding, which this option does not change.

Grain's data seed and global record position produce each example's key.
JAX derives every augmentation draw from that key, not its row in the batch.
Changing the reader threads, batch size or process shares changes no draw;
restoring the data position restores the augmentation without another RNG
state. Switching backend changes the RNG algorithm and is a transform change,
so do not switch it when resuming an existing run.

Augmentation runs on the process's default JAX device, normally its first
local device. With several local devices, that device augments the whole
local batch before the trainer redistributes the rows onto its mesh. This
option does not shard augmentation across the local devices.

Validation reads through the host's deterministic resize unless
`augment_validation=True`, in which case it takes the training crop, flip and
jitter on the same backend, with draws that repeat per record on every pass.

Given identical crop/flip/colour parameters, the JAX op is tested against the
OpenCV bilinear host op in float64. On the RTX 4080 the largest absolute error
was 6.55e-6, with identical rounded uint8 codes across 18 cases. OpenCV's interpolation coefficients are
float32. Four bilinear corner-weight errors and a combined jitter gain below
two give the bound `255 * 8 * eps(float32)`, or at most one code value after
rounding. The default host resize still uses area
interpolation down and cubic up; it is not replaced by bilinear interpolation.

`tools/benchmark_image_pipeline.py` compares OpenCV, PIL, TFDS's NumPy decoder
and torchvision on identical Flowers JPEG bytes, with the same final resize.
It also synchronizes input-pipeline throughput and real prefetched pixel
diffusion updates on the selected device. The decoder microbenchmark excludes
storage reads and startup. The training measurement includes decode, resize,
augmentation, transfer and optimizer updates, excluding compilation/warmup.

Measured on 2026-10-01 with an RTX 4080 (16 GB) and an i9-12900K, JAX
0.11.2.post3, OpenCV 5.0.0, Pillow 11.3.0, TFDS 4.9.10 and torchvision
0.29.0+cpu. A three-image probe first checked the decoder calls; the numbers
below use 512 Flowers JPEGs, RGB in stored orientation, the same 128-square
final OpenCV resize, and the median of three passes. OpenCV used one thread;
the GPU runner capped the host job at four CPU cores.

| Decode + resize | Median images/s | Range over 3 passes | Decoder threads |
|---|---:|---:|---:|
| OpenCV, full JPEG decode | 613 | 486–642 | 1 |
| PIL, full JPEG decode | 554 | 537–557 | 1 |
| TFDS NumPy decoder | 614 | 590–665 | 1 |
| torchvision CPU decoder | 646 | 507–734 | 1 |
| Existing OpenCV reduced JPEG decode | 761 | 745–810 | 1 |

TFDS and torchvision did not beat Dew's existing reduced OpenCV path on this
corpus and size, so the default stays OpenCV. The full-resolution decoders
are run serially, one thread per image; OpenCV and Torch also have their
thread pools limited to one. Pillow was not reduced: `Image.draft` was not
measured. OpenCV is the fastest of the measured paths, not a claim about
every possible reduced Pillow preprocessing path. The full-resolution decoders
produced identical resized bytes on these 512 images. The existing reduced
JPEG decode is a different decode, with a maximum pixel difference of 56
against full decode here; this change does not introduce or alter it. These
results compare TFDS's ArrayRecord NumPy decoder, not a TensorFlow `tf.data`
graph or its scheduling.

The full Flowers source was warmed before either pipeline. Each run used
batch 32, four Grain reader threads, five warmup batches, and three intervals
of 60 batches. CPU time sums the reader threads; it is not wall latency.

| Augmentation | Host images/s | Device images/s | Host CPU ms/batch, before → after |
|---|---:|---:|---:|
| Flip + colour jitter | 1,097 | 1,681 | 65.96 → 47.20 |
| Crop (area 0.6–1.0) + flip + colour jitter | 998 | 2,002 | 83.95 → 47.98 |

The real prefetched `Trainer` update on one RTX 4080 improved from 575 to 874 images/s
(1.52×), for a small pixel-EDM SimpleDiT with patch 16, width 32, one layer,
four heads and float32/HIGHEST computation, 185 updates per backend.
This is an input-bound small model measurement, not a large-model speedup.
Decode and the staging resize still account for host work after augmentation
moves to the device. The CUDA autotuner emitted a delay-kernel timing warning
during compilation; these numbers use synchronized wall-clock intervals after
compilation and warmup, not its kernel timer.

<!-- not run: needs prepared Flowers data and a GPU -->
```bash
python tools/benchmark_image_pipeline.py \
    --flowers data/oxford_flowers102/2.1.1 \
    --images 512 --repeats 3 --batch 32 --steps 60 --threads 4 \
    --out image-pipeline.json
```

The [raw measurements](https://github.com/AshishKumar4/dew/blob/main/tools/measurements/data-rtx4080.json)
include each timing interval, the image-byte SHA-256 and the example outputs.

## Real image and masked-text examples

[`sft_diffusion_gemma_images.py`](https://github.com/AshishKumar4/dew/blob/main/examples/sft_diffusion_gemma_images.py)
trains a small, fresh DiffusionGemma with a Gemma4 vision tower on Oxford
Flowers images and class-name captions. It keeps image placeholders in the
clean prompt and text targets in the response canvas. The script records the
SFT loss, vision-parameter movement and generated captions. It does not
qualify the released 26B DiffusionGemma; that requires a separate 80 GB GPU run.
[`sft_diffusion_gemma.py`](https://github.com/AshishKumar4/dew/blob/main/examples/sft_diffusion_gemma.py)
remains the text-chat LoRA example.

[`train_masked_lm.py`](https://github.com/AshishKumar4/dew/blob/main/examples/train_masked_lm.py)
reads real text prepared by `dew tokenize --tokenizer byte`, trains
the MDLM negative ELBO and unmasks a text sample. The mask is an extra id, 256,
outside the corpus's byte vocabulary. Use WikiText or TinyStories as input.
Both new scripts accept `--smoke` to shrink the model and run while still
reading the supplied real corpus. A few steps establish the workflow, not
caption or language quality.

Both ran for eight updates on the RTX 4080 in float32/HIGHEST. The Flowers
image-SFT run used 1,020 real training images and a 977,362-parameter model.
On the same fixed batch and noise key, SFT loss moved from 11.58235 to
7.64003, the maximum vision-parameter change was 0.0076374, and negating the
conditioning pixels changed the trained loss to 8.16366. The masked-LM run
used 1,984,069 training bytes from a two-million-character WikiText-103
subset and a 147,904-parameter model; fixed-batch NELBO moved from 5.50278 to
4.17817. Both saved checkpoints and generated samples. The samples are
untrained-looking byte text after eight updates, and carry no quality claim.

<!-- not run: needs the real corpora and a GPU -->
```bash
python examples/sft_diffusion_gemma_images.py \
    --flowers data/oxford_flowers102/2.1.1 --smoke --out runs/flowers-caption-smoke
dew tokenize --input data/wikitext103-2m.txt \
    --out data/wikitext-bytes --tokenizer byte
python examples/train_masked_lm.py \
    --tokens data/wikitext-bytes --smoke --out runs/masked-lm-smoke
```

## TFDS and Hugging Face datasets

`dew.data.load("<provider>/<name>", batch=...)` reads a dataset that TFDS or Hugging Face already holds and returns a `Dataset`. `preprocess(record, rng)` turns one provider record into batch fields. Without it, each field of a record reaches the batch as an array: a list column becomes one `[batch, n]` field, 64-bit numbers become 32-bit ones, and strings and bytes stay as they are. Nothing is decoded, renamed or dropped. An integer that does not fit in 32 bits is refused by name. `dataset=` takes a Hugging Face split that is already in memory:

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

A batch stacks each field into one array, so token ids of varying length cannot reach it as they are: tokenizing in `preprocess` and batching the result raises an error that says so. There are two routes to fixed rows.

Offline, `dew tokenize --pack` (or `dew.data.write_tokens(..., pack=True)` in Python) writes a token directory with an eos id after every document, and `PackedTokens` packs it. Its position is a global record count that resumes on any process count. `write_tokens` takes a text file, a directory of `.txt` files, or any iterable of strings, one document each, such as a Hugging Face split's text column.

Online, Grain's packers build the rows as the documents are read, and `Dataset.from_grain` batches them. Each process builds the pipeline over its own share:

<!-- not run: downloads wikitext and the SmolLM2 tokenizer on first use -->
```python
import datasets
import grain
import numpy as np

from dew.data import DataPartition, Dataset, HFTokenizer

tokenizer = HFTokenizer("HuggingFaceTB/SmolLM2-135M")
split = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:2000]")
seq_len = 128


def tokenized(row):
    return {"text": np.asarray(tokenizer.encode(row["text"]), np.int32)}


def documents(partition):
    rows = (grain.MapDataset.source(split).seed(0).shuffle().repeat(None)
            .map(tokenized).filter(lambda row: len(row["text"]) > 0))
    share = rows[partition.index::partition.count].to_iter_dataset()
    return grain.experimental.ConcatThenSplitIterDataset(
        share, length_struct={"text": seq_len + 1})


data = Dataset.from_grain(documents, batch=8)
batch = next(data.train(DataPartition()))
print({name: (value.shape, value.dtype) for name, value in batch.items()})
print([len(set(row)) for row in batch["text_segment_ids"]])
```

```text
{'text': ((8, 129), dtype('int32')), 'text_positions': ((8, 129), dtype('int32')), 'text_segment_ids': ((8, 129), dtype('int32'))}
[1, 1, 2, 2, 3, 2, 1, 2]
```

`ConcatThenSplitIterDataset` concatenates the documents and cuts the stream into rows of `seq_len + 1` ids, splitting a document that crosses a row's end, and writes `text_segment_ids` and `text_positions`, the field names `LMObjective` reads. The last line counts the documents in each row. `grain.experimental.FirstFitPackIterDataset` packs whole documents with padding instead, and refuses a document longer than the row, so cut long documents first. An online pipeline's position is Grain's own iterator state for one share, so it resumes only on the process count that wrote it (see [Resuming the data stream](#resuming-the-data-stream)).

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

`Dataset.batch` is the global batch. With several JAX processes each process reads its share and Dew assembles the global arrays with `jax.make_array_from_process_local_data`. The built-in readers and `Dataset.from_records` split records between processes themselves. A custom `train` function must read only the share its `partition` names, `partition.index` of `partition.count`, or every process trains on the same records.

Before a multi-process run, check on the target topology that process shares do not overlap, that sharding is as expected and that a resume continues the stream. [Distributed training](distributed.md) describes placement.
