# Training data

A `Dataset` is a store with the steps that read its records. `dew.data.load` reads a named Hub or TFDS dataset into a `Loader`; a custom chain uses `compile` directly. Run-layer data specs describe the same chains and compile themselves with `.load()`. All use Grain for ordering, batching, workers and iterator state.

For a named text corpus, the short form downloads and prepares what its read needs:

<!-- not run: downloads Tiny Shakespeare from the Hub -->
```python
import dew.data

data = dew.data.load("hf/winglian/tiny-shakespeare", batch=8, tokenizer="byte", seq_len=64)
```

`tokenizer=` or `seq_len=` selects packed language-model rows, with the byte tokenizer and sequence length 256 where either is unstated. Text holds out 1% of document identities before tokenization; the remaining documents train, and every held-out document belongs only to validation. Without those options, `load` batches the source records and holds out the first 256. It also accepts a `dew.data.Dataset`. Names can pin a Hub revision (`hf/<repo>@<revision>`) or TFDS version (`tfds/<builder>:<version>`). Other sources are spelled with `Hub()`, `TFDS()` or `Files()`.

## A small in-memory dataset

This example needs NumPy and an installed Dew checkout. It downloads nothing.

```python
import numpy as np

from dew.data import Loading
from dew.config.data import RecordsData, loader
from dew.data.dataset import DataPartition

x = np.arange(32, dtype=np.float32).reshape(16, 2)
y = x.sum(axis=1, keepdims=True)
recipe = RecordsData(source={"features": x, "target": y},
                 validation={"features": x[:8], "target": y[:8]})
data = loader(recipe, batch=8, seed=0, read="memory", loading=Loading(workers=0))
stream = data.train(DataPartition())
try:
    first = next(stream)
    print({name: value.shape for name, value in first.items()})
finally:
    stream.close()
```

A source can hold column arrays or a sequence of record dictionaries. Training reshuffles from `seed` every epoch. Validation reads its own records once in order; a real validation split should be disjoint from training. Its final batch repeats rows to fill the shape and marks those repeats with `valid_rows=False`. Objectives and metrics leave them out.

`Loader.train(partition)` and `Loader.val(partition)` each open a fresh iterator. Close the iterator after use, never the shared dataset. `Trainer.fit` closes the iterators it opens after success or failure.

## Recipes and fields

The specs in `dew.config.data` provide `.load(batch=..., read=...)`; `loader(spec_or_alias, ...)` also accepts their registry aliases. A bare URI names a source, so it belongs in `source=` rather than in place of the spec.

| Recipe and alias | Fields after batching |
|---|---|
| `RecordsData` / `records` | The source's fields, with numeric values in JAX's configured widths |
| `TextData` / `text` | `text`, int32 `[B, seq_len + 1]`; packing adds `text_coordinates` |
| `ChatData` / `chat` | Packed `text`, `text_coordinates` and aligned `text_roles` |
| `PromptData` / `prompts` | Left-padded prompt IDs, prompt lengths and fixed-width reward metadata |
| `PreferenceData` / `preferences` | `input_ids` and `completion_mask`, `[B, 2, seq_len]` |
| `ImageData` / `images` | RGB uint8 `image`, `[B, H, W, 3]`, and optional labels |
| `VideoData` / `video` | Uint8 `video`, `[B, T, H, W, 3]`, and audio features |
| `Decisions` / `decisions` | The objective's encoded question rows |

Specs hold source and transform choices. `seed`, `loading`, `read` and `provision` belong to `.load()`. Overrides of an alias are checked against the spec's fields.

## Sources and read modes

All stored locations are URLs. Plain paths and `file://` select local files; schemes such as `gs://`, `s3://`, `hf://` and named storage profiles select their filesystem. Filesystem options belong to the source or store, not to a second loader implementation.

| Source | Memory | Indexed | Streamed |
|---|---|---|---|
| `Memory` records or columns | Materialize records | Direct random access | A physical shard manifest |
| JSONL or text `Files` | Read the source | Producer offset index, or explicit provisioning | Newline-aligned byte ranges |
| CSV `Files` | Read the source | Parse-aware offset index | Whole files |
| Parquet or Arrow `Files` | Read the selected records | Footer row groups | Physical row groups |
| Produced ArrayRecord tokens | Materialize records | Producer length index | Physical shards |
| `Hub` | Prepared Arrow | Prepared Arrow | Hub Parquet shards |
| `TFDS` | Prepared ArrayRecords | Prepared ArrayRecords | Prepared shard ranges |
| `Torch` | Map-style samples | Map-style samples | Refused |

`read="memory"` explicitly runs deterministic transforms, filtering, expansion and packing over the whole source before training. It keeps a dense record order and the global position contract. Use it for small data.

`read="indexed"` requires an index that already exists. `read="streamed"` reads physical shards without a whole-store pass. The chosen mode applies to every source, including validation and all mixture leaves. Unsupported modes are refused before a reader opens.

A run that states no read (`--data-read`, `load(read=None)`) lets its sources choose: memory for records held in memory, indexed for a store that reads by index, streamed otherwise, and a mixture the least eager of its corpora's choices.

`dew.data.load` provisions what its read needs, as Hugging Face `datasets` downloads on `load_dataset`: it prepares TFDS, a Hub Arrow cache or an offset index, logging the artifacts it prepares. Preparing can download data and take a long time. `provision=False` (the default of `compile`, spec `.load()` and run configurations) instead raises an error naming the missing artifact.

For example, a prepared image source is:

<!-- not run: needs prepared TFDS data -->
```python
from dew.data import Loading, TFDS
from dew.config.data import Augment, ImageData, Split, loader

flowers = ImageData(
    source=TFDS("oxford_flowers102", path="data/oxford_flowers102/2.1.1", split="train"),
    validation=Split("validation"), image_size=128,
    caption_templates=("a photo of a {}",),
    augment=Augment(flip=True, jitter=True),
)
data = loader(flowers, batch=32, read="memory", loading=Loading(workers=0))
```

A named Hub source takes `repo`, `config`, `revision`, `split` and `cache_dir`. A direct Parquet URL uses `Files`. For a large corpus, prepare the source separately or use its streamed Parquet files.

## Validation and branches

`HoldOut(records=N)` reserves the first N source records. `HoldOut(fraction=f, by="hash")` splits by immutable record identity without scanning a stream. `Split("validation")` opens a named split. An explicit source or Pipeline is also valid.

A recipe shares deterministic prefix steps between training and evaluation. Random crop, flip and jitter belong to training; evaluation resizes the full image. Caption encoding is a common host stage. A custom pipeline spells out these branches:

```python
import numpy as np
from dew.data import Loading, Memory
from dew.data.compile import compile
from dew.data.pipeline import Pipeline

records = [{"x": np.float32(index)} for index in range(24)]
train = Pipeline(Memory(records[8:])).shuffle(7).repeat().batch(4)
held = Pipeline(Memory(records[:8])).batch(4)
data = compile(train, validation=held, read="memory", loading=Loading(workers=0), seed=7)
```

Validation is finite and never inherits a training repeat or augmentation stage. `validation_records=N` on spec `.load()` caps accepted evaluation records, before the share cut. A training rejection is dropped before mixtures are weighted; attempted, accepted, truncated and rejected records are counted per corpus.

`map(step)` after `batch` runs in the training process after batching and workers. It forwards position and ownership. Steps on only one branch affect that branch; `compile(..., host=(step,))` applies a common host step to both branches.

For conditional image training, pass `host=(EncodeCaptions(objective.inputs.tokenize),)` to spec `.load()`. Without a caption reader, string fields stay on the host, where metrics read them: the prefetcher places only the fields a device holds.

## Text, chat and packing

`TextData` tokenizes raw text or opens a produced token store. Fixed windows contain `seq_len + 1` IDs with one overlap token. `pack="first-fit"` places documents in padded windows, a document longer than a window cut into window-long pieces first; `"concat"` concatenates and splits them. A packed batch carries `text_coordinates`, a `dew.nn.inputs.Coordinates` pytree built on the host. Its `.segments` array stops attention and target scoring at document boundaries, and its `.positions` array restarts within each document. Its static `.max_segments` is the most documents a window of the run holds: as many as the packing plan or the memory read placed in one, or a streamed pack's `segments`, or else its window length; a flash kernel's work follows it (performance.md). Record-level transforms before batching still read `text_segment_ids` and `text_positions`.

`Chat` renders conversations with the tokenizer's chat template. Its default long-record policy truncates; `long="chunk"` keeps an over-length conversation as chunks, and `long="drop"` rejects it. The recipe emits role IDs for every token. The objective chooses the loss role, for example `LMObjective(..., loss_role=Role.ASSISTANT)`.

`Prompts` rejects over-length prompts by default. `long="truncate"` keeps their tail when that is the desired RL policy. `Preferences` refuses an over-length completion rather than cutting away its supervision.

To produce token records, export a deterministic prefix. This example downloads nothing:

```python
from dew.data import ByteTokenizer, Loading, Memory
from dew.config.data import TextData, loader
from dew.data.pipeline import Pipeline
from dew.data.dataset import DataPartition
from dew.data.sinks import export
import tempfile
from dew.data.steps.text import Tokenize

tokenizer = ByteTokenizer()
documents = Memory([{"text": "one document " * 40}, {"text": "another document " * 40}])
prefix = Pipeline(documents).map(Tokenize("byte", eos=True))
token_store = tempfile.mkdtemp(prefix="dew-tokens-")
export(prefix, token_store, format="array_record", length_field="text",
       metadata={"token_store": True, "tokenizer": "byte",
                 "vocab_size": tokenizer.vocab_size, "dtype": "int32",
                 "eos_id": tokenizer.eos_id})
data = loader(TextData(source=token_store, seq_len=32, pack="first-fit"),
            batch=4, read="memory", loading=Loading(workers=0))
stream = data.train(DataPartition())
try:
    packed = next(stream)
    print(packed["text_coordinates"].segments[0])
    print(packed["text_coordinates"].positions[0])
finally:
    stream.close()
```

Use a fresh output directory: exports commit immutable shards and a manifest, and refuse an existing publication. Export validation to a second URL and pass it as `validation=`. Existing binary token stores remain readable through `TokenFiles`; their tokenizer, dtype and document boundaries come from producer metadata.

`TextData` ends each document with the tokenizer's EOS (`eos=True`) and, where no `validation` is named, holds out the head 1% of its tokens after tokenization and before windows; `token_holdout=None` holds out nothing. A token store keeps its producer's own validation split. Streamed runs need producer-published token splits. `ImageData` holds out its first 256 records where no `validation` is named; `validation=None` keeps every record for training.

## Mixtures, objectives and order

`mix(datasets, weights=..., by="records" | "tokens")` mixes accepted records before sharding and ends at its first exhausted member, as Grain's mix does; a member that repeats never ends. A text recipe can name a mapping of token-store URLs to weights; its shares count valid targets, not padding. `Text(phases=(DataPhase(...), ...))` changes mixtures at step boundaries and continues each corpus where its earlier phase stopped.

## Token-budget batches

`Pipeline.batch(tokens=24576)` takes as many rows as a token budget holds rather than a row count. Every row holds the same number of tokens, which a record step states by declaring `TokenGeometry` (`FixedTokens` for token windows, `PackedRows` for packed decision windows); the compiler does not guess from field names, and a budget that is not whole rows is refused.

Run configurations select this with --trainer.batch-size None, --trainer.batch-tokens 24576 and --trainer.update-tokens 98304. Both budgets are required, positive, and update_tokens must divide into whole micro-batches. Prepared.run records the derived accumulation (four in this example); an explicitly conflicting accumulation is refused. Row-batch ramps cannot be combined with token budgets.

A first-fit or best-fit `Pipeline.pack(seq_len)` places each document whole and refuses one longer than its window: `flat_map(text.chunk(seq_len))` just before it cuts documents into window-long pieces, which a store's plan of document lengths cuts alike. `Pipeline.pack(seq_len, segments=..., along=...)` packs rows of several axes with Grain's packer: `along` names fields packed on axes of their own, each to its own length, and `segments` bounds the rows a window holds. Its options (`bins`, `segments`, `policy`, `along`) are the ones a memory, indexed and streamed read lay out alike; `shuffle` after `pack` orders the windows. Decision rows pack this way, their questions along a second axis. Packed causal LM geometry keeps its shifted final ID; denoising objectives declare their own sequence geometry.

## Decision layouts and explicit order

`Decisions` reads examples or a produced mixture manifest. `DecisionObjective.data()` supplies fit, layout and packing steps. The source owns rows and validation; the objective owns layout and model-specific budgets. See [decision models](../guides/decision-models.md).

Pipeline stages execute where written. Use `map`, `random_map`, `filter`, `filter_map`, `random_filter_map`, `flat_map`, `shuffle`, `repeat`, `slice`, `mix`, `windows`, `pack`, `batch`, `workers` and `prefetch` to choose that order; a `map` after `batch` is a host step on the training batch. A global shuffle requires an indexed pipeline.

## Cache and storage publication

`pipeline.cache(Cache(url, ...))` is a bounded write-through artifact stage. It identifies the source snapshot, transform code, dependencies and tokenizer. A filling run keeps its own plan even if another process publishes a complete generation.

`publish(prefix, Cache(url), read=...)` drains a deterministic prefix, certifies source coverage and publishes one generation transactionally. A future run names `Generation(url)` explicitly to read it. Random steps belong after the cache; a plain function before it needs a stable identity. Cache writes require a clean checkout.

`export(pipeline, url, format="parquet" | "array_record")` writes committed shards. Reads follow fsspec URLs. Transactional publication supports local files, memory, S3 and the S3-compatible endpoints that honour its conditional PUTs, GCS, Hub repositories, and any fsspec filesystem that supplies its own conditional writes through a `put_conditional` method (`dew.data.storage.Store`). It refuses other write protocols rather than putting a partial publication. Producer manifests and indexes are the indexed reader's authority. Transport caches use URL chaining such as `dewcache::s3://bucket/path` and are separate from transformed artifacts.

A storage profile names a filesystem and its options once. Profiles live in `~/.config/dew/storage.toml`, or the file `DEW_STORAGE_CONFIG` names, as one table per profile with the `base` protocol it extends. A value written `env:NAME` or `file:PATH` is read from that environment variable or file when the filesystem opens. Cloudflare R2, for example, is an S3 endpoint:

```toml
[r2]
base = "s3"
endpoint_url = "https://ACCOUNT_ID.r2.cloudflarestorage.com"
key = "env:R2_ACCESS_KEY_ID"
secret = "env:R2_SECRET_ACCESS_KEY"
```

With it, `r2://bucket/path` reads and publishes through s3fs with those options.

## Loading and image augmentation

`Loading` changes concurrency, never data order or per-record draws. `dew.data.load` defaults to `Loading(workers=0)`, which reads in the training process. Spec `.load()` defaults to `workers="auto"` and Grain's memory-budgeted tuning. Threads and buffers can be set explicitly. A script that starts worker processes needs an `if __name__ == "__main__":` guard.

`Augment(backend="device")` runs the crop, flip and colour jitter on a batch with JAX. `backend="host"` uses the host implementation. `augment=None` selects deterministic resize. `staging_size` is the decoded square before cropping, and `crop_scale` is the area fraction kept.

Each record's key comes from Grain's data seed and position. Changing reader threads or batch placement changes no augmentation draw. A backend change changes the transform and cannot be substituted silently on resume. Validation keeps the host's deterministic resize.

The following measurements predate the recipe migration. They compare decoder and augmentation implementations on the stated machine, not startup or remote-storage performance of the new compiler.

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
ran serially, one thread per image, and OpenCV and Torch also had their
thread pools limited to one. I did not measure a reduced Pillow decode
(`Image.draft`), so OpenCV is the fastest of the paths measured here, and the
table says nothing about reduced Pillow preprocessing.

The full-resolution decoders produced identical resized bytes on these 512
images. The existing reduced JPEG decode is a different decode, with a maximum
pixel difference of 56 against full decode here, and device augmentation does
not change it. The TFDS row measures TFDS's ArrayRecord NumPy decoder, not a
TensorFlow `tf.data` graph or its scheduling.

The full Flowers source was warmed before either pipeline ran. Each run used
batch 32, four Grain reader threads, five warmup batches and three intervals
of 60 batches. CPU time is summed over the reader threads, so it measures
work and not wall-clock latency.

| Augmentation | Host images/s | Device images/s | Host CPU ms/batch, before → after |
|---|---:|---:|---:|
| Flip + colour jitter | 1,097 | 1,681 | 65.96 → 47.20 |
| Crop (area 0.6–1.0) + flip + colour jitter | 998 | 2,002 | 83.95 → 47.98 |

In real prefetched `Trainer` updates on one RTX 4080, throughput went from 575
to 874 images/s (1.52×) when augmentation moved to the device. The model was a
small pixel-EDM SimpleDiT with patch 16, width 32, one layer, four heads and
float32/HIGHEST computation, and each backend ran 185 updates. This small
model is input-bound, so the number is not a large-model speedup. Decode and
the staging resize are still host work after augmentation moves to the device.

The CUDA autotuner printed a delay-kernel timing warning during compilation.
These numbers come from synchronized wall-clock intervals after compilation
and warmup, not from the autotuner's kernel timer.

<!-- not run: needs prepared Flowers data and a GPU -->
```bash
python tools/benchmark_image_pipeline.py \
    --flowers data/oxford_flowers102/2.1.1 \
    --images 512 --repeats 3 --batch 32 --steps 60 --threads 4 \
    --out image-pipeline.json
```

The [raw measurements](https://github.com/AshishKumar4/dew/blob/main/tools/measurements/data-rtx4080.json)
include each timing interval, the image-byte SHA-256 and the example outputs.

## Position and multiple processes

The checkpoint stores the compiled plan with the iterator position: source snapshots, step identities, ordering, derived order and packing facts, cache generations and topology constraints. A resume rebuilds that plan and refuses changed data, tokenizer, order or transforms. A change only to the code that ran the steps (Dew's revision, the installed packages, the interpreter, the platform or the bytes of a step's module or of the factory that configured it) is a warning on a global resume and a refusal on a per-share one. A step states a change to what it computes in its configuration or its declared `identity`.

| Lowering | Position | Resume topology |
|---|---|---|
| Indexed, count-preserving or memory-materialized | Global consumed-record count | Any process count dividing the global row batch |
| Streamed, online rejecting, expanding or packing | Grain iterator state per share | The recorded share and worker topology |

Volatile fetches cannot claim exact resume until their bytes are persisted. They use the first-reader rule when readers replicate a share.

`Dataset.batch` is global. `DataPartition(index, count)` names a disjoint share; `DataPartition.of(mesh)` derives it from placement. Canonical validation takes accepted rows before the share cut. A record-level random transform's draw is keyed by the record's place in the run's order: the same in every read mode for one order, and by its place in its share in a read cut per share (one that drops records, or a streamed one). Model noise keyed by batch row can change when process placement changes.

[Checkpoints](../guides/checkpoints.md) and [distributed training](distributed.md) cover the trainer's restore and placement contracts.
