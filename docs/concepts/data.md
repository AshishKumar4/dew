# Training data

A `Dataset` supplies training and validation batches. Each batch is a dictionary of arrays. The first dimension holds this process's rows of the global batch, or all rows in a single-process run. Most readers return NumPy arrays. Device image augmentation returns pixels in JAX arrays already on the device. The trainer combines each process's rows into global arrays, and the objective reads fields by name.

Use `Dataset` directly or choose a built-in reader, including readers for prepared TFDS and Hugging Face datasets. The sections below list the fields each objective expects and explain how to resume the data stream from a checkpoint.

![Dataset to global batch: train(partition) opens an iterator of host batches, and shard_batch assembles each into one jax.Array split over the mesh's batch axes.](../assets/data-pipeline-light.svg)
![Dataset to global batch: train(partition) opens an iterator of host batches, and shard_batch assembles each into one jax.Array split over the mesh's batch axes.](../assets/data-pipeline-dark.svg)

## Dataset

For records in memory, use `Dataset.from_records`:

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

`from_records` accepts a mapping of columns with a record dimension first, as above, a list of per-record mappings, or a source with `__len__` and `__getitem__`. Training reshuffles records every epoch using `seed` and reads each record once per epoch. Checkpoints save the position, so a resumed run continues with the unread records. With several processes, each reads its share of every batch.

Validation reads one ordered pass in whole batches. This example uses training records for `validation` to show the argument. To measure validation performance, use records the model does not train on.

Every reader returns the same `Dataset` value:

| Field | Meaning |
|---|---|
| `train(partition)` | Returns a new iterator of training batches. It must yield at least as many batches as the run takes steps. |
| `val(partition)` | Returns a new iterator over one pass of the validation records, which then ends. `None` means no validation split. |
| `records` | The number of training records, or `None` when unknown. |
| `batch` | The global batch size. |

`train` and `val` are functions so that each new or resumed run can open fresh iterators. If you open an iterator, close it after use when it has a `close` method. Do not close the dataset or its backing store. `Trainer.fit` closes its iterators whether the run finishes or fails. Keeping a training-step exception after the run does not keep the closed prefetch iterator alive.

The `DataPartition` argument specifies this process's share of every global batch. Use `DataPartition()` to read all rows in a single process. With several processes, the trainer gets each share from the mesh through `DataPartition.of(mesh)`. Built-in readers read only that share.

For a stream without a built-in reader, pass the two functions directly to `Dataset`:

```python
import itertools

import numpy as np
from dew.data import Dataset

x = np.arange(16, dtype=np.float32).reshape(8, 2)
batch = {"features": x, "target": x.sum(axis=1, keepdims=True)}
data = Dataset(train=lambda partition: itertools.repeat(batch), val=None, records=8, batch=8)
```

Your function must read only the share specified by its partition. For checkpointing, its iterator also needs `get_state` and `set_state`; see [Resuming the data stream](#resuming-the-data-stream). The function above does neither, so use it only for a single-process run with `checkpoint_every=None`. If you have a Grain pipeline, pass it to `Dataset.from_grain`.

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

Built-in readers use dataset specifications, which are frozen dataclasses describing the dataset. Calling `load(batch=...)` builds a reading pipeline and returns a `Dataset`. `TokenWindows` reads fixed windows of `seq_len + 1` token IDs from a tokenized directory:

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

The training stream is shuffled, so this first batch starts with window 49 of the corpus. `records` counts windows: `(1000 - 1) // 8 = 124`.

Each window starts `seq_len` IDs after the previous one. The last ID of one window is therefore the first of the next. To prepare raw text, use `dew tokenize` or `TokenCorpus.write` in Python. They write `train.bin`, `val.bin` and `meta.json`; see [Packing](#packing). For token IDs already in parquet, use `dew.data.load("hf/parquet", options=HFOptions(data_files=...))` or pass a Grain pipeline to `Dataset.from_grain`.

Other specifications include `PackedTokens`, `TFDSImages`, `HFImages`, `ChatMessages` and the video and preference readers. `TFDSImages` reads prepared TFDS images with captions from their class names. The [API reference](../reference/core-api.md) lists all the readers. Each has fields for paths, tokenization, transforms and splits. Every specification also has these two fields:

| Field | Meaning |
|---|---|
| `seed` | The record order and the key of each record's random draws (augmentation, captions, clip starts) |
| `loading` | A `Loading` with Grain's throughput settings |

Use `Loading` to adjust reading throughput. It leaves record order and content unchanged:

| `Loading` field | Default | Counts |
|---|---|---|
| `workers` | 0 | Grain worker processes; `0` reads in the training process |
| `threads` | 64 | Record reads one worker keeps in flight |
| `read_buffer` | 128 | Records one worker reads ahead |
| `worker_buffer` | 2 | Batches one worker holds ready |

The default reads in the training process, matching Grain's default. Each worker process imports the program again, adding seconds and memory before the first batch. Workers help only when decoding or augmentation exceeds the reading threads' capacity. Before raising `workers`, measure the input pipeline on your data and hardware. If your script starts workers, use an `if __name__ == "__main__":` guard so that importing it does not start more workers.

Image sources may need network access on first use. Token-window sources read files written by `dew tokenize`. Streaming sources can depend on remote servers and may not support restoring a position. See [Recipes](../recipes.md) for command-line entry points and [Installation](../installation.md#optional-extras) for each source's extras.

`OnlineImages` and `OnlineVideos` stream Hugging Face tables of URLs and captions. In a recipe, select `OnlineVideos` with `data:online-videos`. It reads each URL like a `LocalVideos` file: `frames` consecutive frames at 25 fps, resized to `image_size` squares, without audio.

## Image datasets on the Hugging Face Hub

`HFImages` reads Hub images by index. The pipeline decodes each image, resizes it to `image_size`, applies augmentation and reads captions for text conditioning. Set `image_column` to the source's image field. The caption comes from the first available field in `caption_columns`. If the dataset has a `label` column, it supplies the class index. CIFAR-10, for example, stores images under `img` and classes under `label`, with no captions:

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

For an unconditional or class-conditional run without captions, set `caption_columns=()`. That setting rejects a caption reader. If you request a column absent from the split, loading raises an error listing the available columns. Without `val_split`, the reader holds out the first `val_batches` batches of the training split. With `val_split` and `val_batches=None`, validation scores the whole named split.

Validation uses deterministic resizing without training's crop, flip or jitter. Metrics therefore score the same images as a reference implementation.

## Device image augmentation

`TFDSImages`, `HFImages` and prepared `ArrayRecordImages` readers use
`ImageDataset`'s transforms. To apply random crop/resize, horizontal flip
and colour jitter to decoded batches with JAX, set
`augmentation_backend="device"`. The default, `"host"`, keeps existing
runs' OpenCV/NumPy augmentation. `augmentation="flip_only"` disables
colour jitter. With `"none"`, images receive deterministic resizing
without random cropping.

Both backends use torchvision's float `ColorJitter(brightness=0.2,
contrast=0.05, saturation=0.2)`. They apply the three factors in random
order and clamp each operation to the pixel range. The result is rounded
to uint8 once, at the end.

<!-- not run: needs a prepared Oxford Flowers version directory -->
```python
from dew.data import Loading, TFDSImages

data = TFDSImages(
    path="data/oxford_flowers102/2.1.1",
    image_size=128,
    augmentation_backend="device",
    augmentation="flip_jitter",
    crop_scale=(0.6, 1.0),
    augmentation_size=160,
    loading=Loading(workers=0, threads=4),
).load(batch=32)
```

`crop_scale` sets the retained area fraction, drawn uniformly. The crop
uses integer coordinates within the staging square, with each side
scaled by the square root of that fraction. `(1.0, 1.0)` keeps the full
image. With `augmentation_size=None`, staging uses `image_size`.
A larger staging size retains more pixels for crop/resize but increases
transfers and device memory.

Decoding, the initial area/cubic resize to a dense square and caption
tokenization run on the host. Variable-sized JPEGs cannot be stacked
directly. JAX does not decode them, and this option adds no JPEG decoder.
The crop's bilinear resize, flip and jitter run on the device, with one
final uint8 round/clip. CPU consumers, `OnlineImages` and video decoding
still need the host path, which this option leaves unchanged.

Grain uses the data seed and global record position for each example's key.
JAX derives augmentation draws from that key. The batch row does not affect
them. Changing reader threads, batch size or process shares leaves the
draws unchanged. Restoring the data position also restores augmentation;
you do not need another RNG state. Switching backend changes the RNG
algorithm and therefore the transform. Do not switch it when resuming a run.

Augmentation runs on the process's default JAX device, normally its first
local device. With several local devices, that device augments the whole
local batch before the trainer redistributes the rows onto its mesh. This
option does not shard augmentation across the local devices.

Validation always uses the host's deterministic resize, regardless of
the training augmentation or backend.

Tests compare the JAX operation with the OpenCV bilinear host operation
in float64, using identical crop/flip/colour parameters. On the RTX 4080,
the largest absolute error was 6.55e-6. Rounded uint8 codes matched across
18 cases. OpenCV's interpolation coefficients are float32. Four bilinear
corner-weight errors and a combined jitter gain below two give the bound
`255 * 8 * eps(float32)`, or at most one code value after rounding.
The default host resize continues to use area interpolation down and
cubic up. It does not use bilinear interpolation.

`tools/benchmark_image_pipeline.py` compares OpenCV, PIL, TFDS's NumPy
decoder and torchvision on identical Flowers JPEG bytes with the same
final resize. It also measures synchronized input throughput and real
prefetched pixel-diffusion updates on the selected device. Decoder timings
exclude storage reads and startup. Training timings include decode,
resize, augmentation, transfer and optimizer updates. They exclude
compilation and warmup.

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

The reduced OpenCV path was fastest among the measured decoders, including
TFDS and torchvision, on this corpus and size. OpenCV remains the default.
Full-resolution decoders ran serially with one thread per image.
OpenCV and Torch thread pools were
also limited to one. I did not measure reduced Pillow decoding with
`Image.draft`, so these results say nothing about that path.

The full-resolution decoders produced identical resized bytes on these
512 images. Dew's existing reduced JPEG decode differs from full decode;
the maximum pixel difference here was 56. Device augmentation leaves
that decode unchanged. The TFDS result measures its ArrayRecord NumPy
decoder. It does not measure a TensorFlow `tf.data` graph or its scheduling.

Both pipelines used the warmed full Flowers source. Each run used batch 32,
four Grain reader threads, five warmup batches and three intervals of
60 batches. CPU time is the sum across reader threads. It measures CPU
work, not elapsed latency.

| Augmentation | Host images/s | Device images/s | Host CPU ms/batch, before → after |
|---|---:|---:|---:|
| Flip + colour jitter | 1,097 | 1,681 | 65.96 → 47.20 |
| Crop (area 0.6–1.0) + flip + colour jitter | 998 | 2,002 | 83.95 → 47.98 |

Real prefetched `Trainer` updates on one RTX 4080 increased from 575 to
874 images/s (1.52×). The small pixel-EDM SimpleDiT used patch 16, width 32,
one layer, four heads and float32/HIGHEST computation. Each backend ran
185 updates. This small model was input-bound; the measurement does not
establish a large-model speedup. Decoding and the staging resize still
require host work with device augmentation.

During compilation, the CUDA autotuner emitted a delay-kernel timing
warning. The reported numbers come from synchronized wall-clock intervals
after compilation and warmup. They do not use the autotuner's kernel timer.

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
the MDLM negative ELBO and unmasks a text sample. The mask uses an extra
ID, 256, outside the corpus's byte vocabulary. Use WikiText or TinyStories
as input. Both scripts accept `--smoke` to reduce the model and run size
while still reading the supplied real corpus. These short runs check the
workflow. They do not demonstrate caption or language quality.

Both ran for eight updates on the RTX 4080 in float32/HIGHEST. The Flowers
image-SFT run used 1,020 real training images and a 977,362-parameter model.
With a fixed batch and noise key, SFT loss fell from 11.58235 to 7.64003.
The maximum vision-parameter change was 0.0076374. Negating the
conditioning pixels changed the trained loss to 8.16366.

The masked-LM run used 1,984,069 training bytes from a
two-million-character WikiText-103 subset and a 147,904-parameter model.
Fixed-batch NELBO fell from 5.50278 to 4.17817. Both runs saved checkpoints
and generated samples. After eight updates, those samples are
untrained-looking byte text; I make no claim about their quality.

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

`dew.data.load("<provider>/<name>", batch=...)` reads a TFDS or Hugging Face dataset and returns a `Dataset`. To turn a provider record into batch fields, pass `preprocess(record, rng)`. Without preprocessing, every field becomes an array in the batch. A list column becomes a `[batch, n]` field, 64-bit numbers become 32-bit numbers, and strings and bytes remain unchanged. The reader does not decode, rename or drop fields. It rejects an integer that does not fit in 32 bits, naming its field.

If a Hugging Face split is already in memory, pass it through `dataset=`:

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

`tfds/<builder>` reads prepared ArrayRecord files at `TFDSOptions.path` through TFDS's read-only builder. The training process therefore does not import TensorFlow. Dew never prepares the data itself; follow [Installation](../installation.md#preparing-tfds-data) to do that first. Set `path` to the prepared version directory or the `data_dir` above it. For `data_dir`, `config` and `version` select the version directory inside it. Dew checks the prepared metadata against the requested builder, config and version. It passes `decoders` to the builder unchanged.

<!-- not run: needs a prepared TFDS directory -->
```python
import dew.data
from dew.data import TFDSOptions

data = dew.data.load("tfds/dew_images", batch=8, split="train", val_split="test",
                     options=TFDSOptions(path="/data/prepared"),
                     preprocess=lambda record, rng: {"image": record["image"]})
```

`hf/<name>` reads one Arrow-backed split through `datasets.load_dataset`. If the local cache lacks the dataset, that function downloads it and writes an Arrow cache. `HFOptions` supplies `config`, `data_files`, `features`, `storage_options` and other `load_dataset` arguments using that function's types.

With `streaming=True`, the reader consumes an `IterableDataset` as records arrive. A streamed split has no length. Unless you supply `records`, it and `Dataset.steps_per_epoch` are `None`. Processes divide rows with `datasets.distributed.split_dataset_by_node`. If there are more processes than rows, Dew rejects the run because at least one process would get no rows.

A streamed split loaded by name without shuffling resumes at the next unread record, with the same per-record random draws. These streamed splits cannot restore a position. They require `checkpoint_every=None`; otherwise, `Trainer.fit` rejects the run:

- A split read with `shuffle_buffer` set, because `datasets` does not restore that buffer.
- A split passed as `dataset=`, because it may include transformations Dew did not apply.
- A split whose source implements no state in `datasets`.

## Packing

Packing combines tokens from several documents into fixed-width rows. Segment IDs (`text_segment_ids`) keep attention and target scoring within document boundaries. Position IDs (`text_positions`) restart at each document. Slice and pack every per-token array the same way as the token IDs.

A batch stacks each field into one array, which requires fixed-width token rows. Tokenizing variable-length text in `preprocess` and batching it directly raises an error. You can make fixed rows offline or online.

Offline, use `dew tokenize --pack` or `TokenCorpus.write(..., pack=True)` in Python. It writes a token directory with an EOS ID after every document. `PackedTokens` packs that directory and tracks a global record count that can resume on any process count. `TokenCorpus.write` accepts a text file, a directory of `.txt` files, or an iterable of strings with one document per string. A Hugging Face split's text column is one such iterable.

Online, Grain packers build rows as they read documents. `Dataset.from_grain` batches those rows. Each process builds a pipeline for its own share:

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

`ConcatThenSplitIterDataset` joins documents and splits the stream into rows of `seq_len + 1` IDs. A document can span rows. It also writes `text_segment_ids` and `text_positions`, the fields `LMObjective` reads. The last output line counts documents in each row.

To pack whole documents with padding, use `grain.experimental.FirstFitPackIterDataset`. It rejects documents longer than a row, so cut those first. An online pipeline stores Grain's iterator state for one process's share. It can resume only with the process count that wrote the checkpoint; see [Resuming the data stream](#resuming-the-data-stream).

Padding and packing change the number of valid targets even when array shapes match. Gradient accumulation adds microbatch loss totals and masses, then divides once. The gradient is therefore for the token mean over the whole window. Before treating differently packed runs as equivalent, check their normalization.

## Resuming the data stream

If an iterator implements `get_state()` and `set_state(state)`, Dew can save its position in every checkpoint and restore it. Keep the same data order, tokenizer and transforms when resuming. There are two kinds of position:

| Kind | Written by | Resumes on |
|---|---|---|
| Global record count | Every reader built on `train_stream`: token windows, packed documents and conversations, weighted mixtures, images, video, prompts, preference pairs | Any process count that divides the global batch |
| Shard offset | Custom iterators that report their own state | Only the process count that wrote it |

A global position counts records consumed by the whole run. Every process reports the same number. Step *k* reads records `[k * batch, (k + 1) * batch)` from one shuffled order. Process *p* of *n* reads every *n*-th record in that step. A checkpoint written by two processes can therefore restore on one or four. The resumed steps read the same records as an uninterrupted run. Restoring starts the stream at that position without reading any record twice.

The saved position includes the source description, record count and shuffle seed. Dew rejects a resume with a different record count or seed. Without a custom `__repr__`, a source's description is its type name. Dew can distinguish two corpora of equal length and seed only if their source descriptions identify the data. For a source you resume across runs, provide a `__repr__` that names its data.

Changing the process count changes which process holds each row. Randomness keyed by row, such as diffusion noise or sampled timesteps, then applies to different records. The resumed run optimizes the same objective but draws different noise for each record, so its logged losses differ from an uninterrupted run. A record's own random draws use its position in the shuffled stream and remain independent of process count.

To resume a shard offset, use the process count that wrote it. `Checkpoints.restore` rejects a different count and reports both values. [Checkpoints](../guides/checkpoints.md) shows the full save and restore workflow.

## Multiple processes

`Dataset.batch` is the global batch size. Each JAX process reads its share, and Dew assembles global arrays with `jax.make_array_from_process_local_data`. Built-in readers and `Dataset.from_records` split the records between processes. In a custom `train` function, read only `partition.index` of `partition.count`, the share specified by `partition`. Otherwise, every process trains on the same records.

Before a multi-process run, check on the target topology that process shares do not overlap, that sharding is as expected and that a resume continues the stream. [Distributed training](distributed.md) describes placement.
