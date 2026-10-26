# Training data

A `Dataset` is one chain: a source, the steps that read its records, and a preset that ends in batches. `Trainer.fit` compiles it into a `Loader`, and a custom loader uses `compile` directly. Both use Grain for ordering, batching, workers and iterator state. [The data design](../design/data.md) is the full account.

For a named text corpus, a source and the `lm` preset download and prepare what the read needs:

<!-- not run: downloads Tiny Shakespeare from the Hub -->
```python
from dew.data import Hub, presets

train, val = Hub("winglian/tiny-shakespeare").holdout(0.01)
data = train.shuffle(seed=0).repeat().pipe(presets.lm, "byte", seq_len=64, batch=8)
held = val.pipe(presets.lm, "byte", seq_len=64, batch=8)
```

`presets.lm` tokenizes each document, ends it with the tokenizer's EOS where it has one, and packs whole documents into rows of `seq_len + 1` tokens. `holdout(0.01)` holds out 1% of document identities before tokenization; the remaining documents train, and every held-out document belongs only to validation. Both parts take the same preset, and only the training part shuffles and repeats. A Hub source pins a revision with `Hub(repo, revision=...)` and a TFDS one a version with `TFDS(name, version=...)`; other stores are read with `Files()`.

## A small in-memory dataset

This example needs NumPy and an installed Dew checkout. It downloads nothing.

```python
import numpy as np

from dew.data import Memory
from dew.data.compile import compile
from dew.data.dataset import DataPartition

x = np.arange(32, dtype=np.float32).reshape(16, 2)
y = x.sum(axis=1, keepdims=True)
train = Memory({"features": x, "target": y}).shuffle(seed=0).repeat().batch(8)
held = Memory({"features": x[:8], "target": y[:8]}).batch(8)
data = compile(train, validation=held)
stream = data.train(DataPartition())
try:
    first = next(stream)
    print({name: value.shape for name, value in first.items()})
finally:
    stream.close()
```

A source can hold column arrays or a sequence of record dictionaries. `shuffle(seed=0)` draws a new order for every epoch of the `repeat`. Validation reads its own records once in order; a real validation split should be disjoint from training. Its final batch repeats rows to fill the shape and marks those repeats with `valid_rows=False`. Objectives and metrics leave them out.

`Loader.train(partition)` and `Loader.val(partition)` each open a fresh iterator. Close the iterator after use, never the shared dataset. `Trainer.fit` closes the iterators it opens after success or failure.

## Presets and fields

A preset is a plain function from a dataset and its settings to a dataset, applied with `pipe`, the same preset for training and validation. Each ends in `batch`, so its result goes straight to `fit`, and `show()` lists the steps it expanded to. Presets live in `dew.data.presets`.

| Preset, or chain | Fields after batching |
|---|---|
| A source and `batch(n)` | The source's fields, with numeric values in JAX's configured widths |
| `lm` | `text`, int32 `[B, seq_len + 1]`; packing adds `text_coordinates` |
| `lm` with a prefix run (`LMObjective(prefix=...)`) | `text_block_groups`, int32 `[B, seq_len + 1]`: the blocks a prefix run reads whole, -1 outside any |
| `sft` | Packed `text`, `text_coordinates` and aligned `text_roles` |
| `canvas` | One row per assistant turn: `text`, int32 `[B, prompt_length + canvases * canvas_size]`, and `canvas_mask` |
| `prompts` | Left-padded prompt IDs, prompt lengths and fixed-width reward metadata |
| `preferences` | `input_ids` and `completion_mask`, `[B, 2, seq_len]` |
| `images` | RGB uint8 `image`, `[B, size, size, 3]`, and the source's other fields, such as `label` |
| `clips` | Uint8 `video`, `[B, frames, size, size, 3]`, and audio features |
| `Decisions` | The objective's encoded question rows |

A preset holds its transform choices. How the chain is read and run belongs to the chain: `read(...)`, `workers(n)`, `threads(n)` and `prefetch(n)`.

## Sources and read modes

All stored locations are URLs. Plain paths and `file://` select local files; schemes such as `gs://`, `s3://`, `hf://` and named storage profiles select their filesystem. Filesystem options belong to the source or store, not to a second loader implementation.

| Source | Memory | Indexed | Streamed |
|---|---|---|---|
| `Memory` records or columns | Materialize records | Direct random access | A physical shard manifest |
| JSONL or text `Files` | Read the source | Producer offset index, or explicit provisioning | Newline-aligned byte ranges |
| CSV `Files` (fields within `csv.field_size_limit`) | Read the source | Parse-aware offset index | Whole files |
| Parquet or Arrow `Files` | Read the selected records | Footer row groups | Physical row groups |
| Produced ArrayRecord tokens | Materialize records | Producer length index | Physical shards |
| `Hub` | Prepared Arrow | Prepared Arrow | Hub Parquet shards |
| `TFDS` | Prepared ArrayRecords | Prepared ArrayRecords | Prepared shard ranges |
| PyTorch datasets (`from_torch`) | Map-style samples | Map-style samples | Refused |

`read="memory"` explicitly runs deterministic transforms, filtering, expansion and packing over the whole source before training. It keeps a dense record order and the global position contract. Use it for small data.

`read="indexed"` requires an index that already exists. `read="streamed"` reads physical shards without a whole-store pass. The chosen mode applies to every source, including validation and all mixture leaves. Unsupported modes are refused before a reader opens.

A chain that states no read (`read(...)`) lets its sources choose: memory for records held in memory, indexed for a store that reads by index, streamed otherwise, and a mixture the least eager of its corpora's choices.

`Trainer.fit` provisions what its read needs, as Hugging Face `datasets` downloads on `load_dataset`: it prepares TFDS, a Hub Arrow cache or an offset index, logging the artifacts it prepares. Preparing can download data and take a long time. `compile`'s default, `provision=False`, instead raises an error naming the missing artifact.

A `Hub` source keeps, with the Hub offline (`HF_HUB_OFFLINE=1`), what an online read pinned: every answer it takes from the Hub (the commit a revision names, the commit's files, its Parquet conversion, each file's size and content hash) is recorded under Dew's cache (`$XDG_CACHE_HOME/dew/hub`, or `DEW_HUB_ANSWERS`), and offline the source reads those answers. A split prepared as Arrow online then reads offline from the cache `datasets` prepared at that commit, and a streamed split's shards keep the identity a compiled cache is keyed by. Shard bytes are not recorded: a streamed read that compiles no cache still needs the Hub. Offline nothing can be provisioned, and a source never read online raises the Hub's offline error. Streamed Hub shards are identified by their size and content hash, so caches compiled before Dew recorded them compile once more.

For example, a prepared image source is:

<!-- not run: needs prepared TFDS data -->
```python
from dew.data import TFDS, images, presets

train, val = TFDS("oxford_flowers102", path="data/oxford_flowers102/2.1.1").holdout(split="validation")
caption = images.caption("a photo of a {label}")
data = train.shuffle(seed=0).repeat().pipe(presets.images, 128, batch=32, augment=True, caption=caption)
held = val.pipe(presets.images, 128, batch=32, caption=caption)
```

A named Hub source takes `repo`, `config`, `revision`, `split` and `cache_dir`. A direct Parquet URL uses `Files`. For a large corpus, prepare the source separately or use its streamed Parquet files.

## Validation and branches

`holdout(records=N)` reserves the first N source records, which it reads by index: a memory or indexed read. `holdout(f)` splits by immutable record identity as each record is read, so it also streams, and its validation pass reads every record to find its own. `holdout(split="validation")` opens the source's named split. A second dataset of your own is also valid validation.

Steps written before `holdout` are shared by both parts. Random crop, flip and jitter belong to training (`augment=True` on its preset); evaluation resizes the full image. A custom chain spells out the branches:

```python
import numpy as np
from dew.data import Memory
from dew.data.compile import compile

records = [{"x": np.float32(index)} for index in range(24)]
train = Memory(records[8:]).shuffle(7).repeat().batch(4)
held = Memory(records[:8]).batch(4)
data = compile(train, validation=held, seed=7)
```

`holdout` keeps the steps written before it on both parts, so write shared deterministic preparation before the split and add `shuffle`, `repeat` and augmentation only to the training part: validation is then finite and unaugmented. `records=N` on a preset caps the accepted records a part reads, before the share cut. A training rejection is dropped before mixtures are weighted; attempted, accepted, truncated and rejected records are counted per corpus.

`map(step)` after `batch` runs in the training process after batching and workers. It forwards position and ownership. Steps on only one branch affect that branch; `compile(..., host=(step,))` applies a common host step to both branches.

For conditional image training, `Trainer.fit` compiles the chain with the objective's own stages (`objective.data()`), and a diffusion objective's encode the captions for its conditions and keep them beside as host text (`EncodeCaptions(keep=CAPTIONS)`), where metrics and rewards read them with their own tokenizer (`CLIPScore` with CLIP's): the prefetcher places only the fields a device holds.

## Text, chat and packing

`presets.lm` tokenizes raw text, or reads a produced token store's ids with `tokenizer=None`. `pack="first-fit"`, the default, places documents in padded windows of `seq_len + 1` IDs, a document longer than a window cut into window-long pieces first; `"concat"` concatenates and splits them, and `pack=None` cuts one stream of tokens into fixed windows with one overlap token. A packed batch carries `text_coordinates`, a `dew.batch.Coordinates` pytree built on the host. Its `.segments` array stops attention and target scoring at document boundaries, and its `.positions` array restarts within each document. Its static `.max_segments` is the most documents a window of the run holds: as many as the packing plan or the memory read placed in one, or a streamed pack's `segments`, or else its window length; a flash kernel's work follows it (performance.md). Record-level transforms before batching still read `text_segment_ids` and `text_positions`.

`presets.sft` renders conversations with the tokenizer's chat template. By default it keeps an over-length conversation as chunks, every turn kept; `long="truncate"` keeps its first window, and `long="drop"` rejects it, counted under its reason. It emits role IDs for every token. The objective chooses the loss role, for example `LMObjective(..., loss_role=Role.ASSISTANT)`.

`presets.canvas` renders the same way for `BlockDiffusionObjective`, which splits every row at its prompt length, so nothing is packed: each assistant turn is one row, the conversation before it padded to `prompt_length` and the turn on whole canvases, EOS closing the last one. A longer prompt or turn keeps its head, as Google's reference cuts both.

`presets.prompts` keeps an over-length prompt's tail by default, its last `max_prompt_len` ids. `long="drop"` rejects it instead, counted as a rejection under its reason (`over_length`). `presets.preferences` refuses an over-length completion rather than cutting away its supervision.

To produce token records, export a deterministic prefix. This example downloads nothing:

```python
import tempfile

from dew.data import Memory, presets, text
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.sources import TokenFiles

documents = Memory([{"text": "one document " * 40}, {"text": "another document " * 40}])
token_store = tempfile.mkdtemp(prefix="dew-tokens-")
text.export_tokens(documents, token_store, "byte")
data = compile(TokenFiles(token_store).pipe(presets.lm, None, seq_len=32, batch=4))
stream = data.train(DataPartition())
try:
    packed = next(stream)
    print(packed["text_coordinates"].segments[0])
    print(packed["text_coordinates"].positions[0])
finally:
    stream.close()
```

Use a fresh output directory: exports commit immutable shards and a manifest, and refuse an existing publication. Export validation to a second URL and read it as a second `TokenFiles` source. A store's tokenizer, dtype and document boundaries come from its producer metadata.

`presets.lm` ends each document with the tokenizer's EOS where it has one (`eos=True` always, `eos=False` never). Nothing is held out unless the chain says so: `holdout(tokens=0.01, fields=("text",))` holds out the first hundredth of a finite order's tokens, and a token store keeps its producer's own validation split, read as a second source. Streamed runs need producer-published token splits. Records held out by slice, `holdout(records=256)`, are read by index; a streamed read holds out by identity, `holdout(f)`, whose validation pass streams the whole source: over the 7,169 records of Oxford Flowers' Hub train split, a 1% pass took 11.2 s against 0.3 s for the 256-record slice read by index.

## Mixtures, objectives and order

`mix(datasets, weights=..., by="records" | "tokens")` mixes accepted records before sharding and ends at its first exhausted member, as Grain's mix does; a member that repeats never ends. With `by="tokens"`, token stores' shares count valid targets, not padding. `phases((first, until_step), ..., (last, None))` reads datasets one after another, each until the training step it names and the last until the run ends; each phase is a whole chain of the same batch. `schedule([(first, tokens), ..., (last, None)])` reads them on the run's committed-token clock instead: each stage ends at the step whose commit brings the run's committed tokens, the count `fit(tokens=)` reads, to the stage's own.

## Token-budget batches

`Pipeline.batch(tokens=24576)` takes as many rows as a token budget holds rather than a row count. Every row holds the same number of tokens, which a record step states by declaring `TokenGeometry` (`FixedTokens` for token windows, `PackedRows` for packed decision windows); the compiler does not guess from field names, and a budget that is not whole rows is refused.

A run configuration takes the micro-batch from the chain's `batch(tokens=24576)` and the update from `--trainer.update-tokens 98304`. Both budgets are positive, and update_tokens must divide into whole micro-batches. Prepared.run records the derived accumulation (four in this example); an explicitly conflicting accumulation is refused. Row-batch ramps cannot be combined with token budgets.

A first-fit or best-fit `Pipeline.pack(seq_len)` places each document whole and refuses one longer than its window: `flat_map(text.Chunks(seq_len))` just before it cuts documents into window-long pieces, which a store's plan of document lengths cuts alike. `Pipeline.pack(seq_len, segments=..., along=...)` packs rows of several axes with Grain's packer: `along` names fields packed on axes of their own, each to its own length, and `segments` bounds the rows a window holds. Its options (`bins`, `segments`, `policy`, `along`) are the ones a memory, indexed and streamed read lay out alike; `shuffle` after `pack` orders the windows. Decision rows pack this way, their questions along a second axis. Packed causal LM geometry keeps its shifted final ID; denoising objectives declare their own sequence geometry.

## Decision layouts and explicit order

`Decisions` reads examples or a produced mixture manifest. `DecisionObjective.data()` supplies fit, layout and packing steps. The source owns rows and validation; the objective owns layout and model-specific budgets. See [decision models](../guides/decision-models.md).

A chain's stages run where written. Use `map`, `random_map`, `filter`, `filter_map`, `random_filter_map`, `flat_map`, `shuffle`, `repeat`, `slice`, `mix`, `windows`, `pack`, `batch`, `workers` and `prefetch` to choose that order; a `map` after `batch` is a host step on the training batch. A global shuffle requires an indexed pipeline.

## Cache and storage publication

`pipeline.cache(Cache(url, ...))` is a bounded write-through artifact stage. It identifies the source snapshot, transform code, dependencies and tokenizer. A filling run keeps its own plan even if another process publishes a complete generation.

`publish(prefix, Cache(url), read=...)` drains a deterministic prefix, certifies source coverage and publishes one generation transactionally. A future run names `Generation(url)` explicitly to read it. Random steps belong after the cache; a plain function before it needs a stable identity. Cache writes require a clean checkout.

`export(pipeline, url, format="parquet" | "array_record")` writes committed shards. Reads follow fsspec URLs. Transactional publication supports local files, memory, S3 and the S3-compatible endpoints that honour its conditional PUTs, GCS, Hub repositories, and any fsspec filesystem that supplies its own conditional writes through a `put_conditional` method (`dew.data.storage.Store`). It refuses other write protocols rather than putting a partial publication. Producer manifests and indexes are the indexed reader's authority. Transport caches use URL chaining such as `dewcache::s3://bucket/path` and are separate from transformed artifacts.

A storage profile names a filesystem and its options once. Profiles live in `storage.toml` in Dew's configuration directory (`$DEW_CONFIG_DIR`, else `$XDG_CONFIG_HOME/dew`, else `~/.config/dew`), or the file `DEW_STORAGE_CONFIG` names, as one table per profile with the `base` protocol it extends. A value written `env:NAME` or `file:PATH` is read from that environment variable or file when the filesystem opens. Cloudflare R2, for example, is an S3 endpoint:

```toml
[r2]
base = "s3"
endpoint_url = "https://ACCOUNT_ID.r2.cloudflarestorage.com"
key = "env:R2_ACCESS_KEY_ID"
secret = "env:R2_SECRET_ACCESS_KEY"
```

With it, `r2://bucket/path` reads and publishes through s3fs with those options.

## Loading and image augmentation

`workers(n)`, `threads(n)` and `prefetch(n)`, or `loading(Loading(...))` for every setting at once, change concurrency, never data order or per-record draws. A chain that names none reads in the training process (`workers=0`); `loading(Loading(workers="auto"))` takes Grain's memory-budgeted tuning. A script that starts worker processes needs an `if __name__ == "__main__":` guard.

`augment=True` on `presets.images` runs the crop, flip and colour jitter on a batch with JAX; `images.DeviceAugment(size, crop_scale=, flip=, jitter=)` sets them itself, and a random host step or a tuple of them, such as `images.RandomCrop(size)` or `images.Albumentations(aug)`, runs on the host instead. Without `augment`, the preset resizes deterministically. `staging_size` is the decoded square before cropping, and `crop_scale` is the area fraction kept.

Each record's key comes from Grain's data seed and position. Changing reader threads or batch placement changes no augmentation draw. A backend change changes the transform and cannot be substituted silently on resume. Validation keeps the host's deterministic resize.

The following measurements predate the chain. They compare decoder and augmentation implementations on the stated machine, not startup or remote-storage performance of the new compiler.

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

The checkpoint stores the compiled plan with the iterator position: source snapshots, step identities, ordering, derived order and packing facts, cache generations and topology constraints. A resume rebuilds that plan and refuses changed data, tokenizer, order or transforms. The plan also records the version of how Dew reads a plan into rows (`dew.data.plan.Plan.version`), and a position saved under another version is refused before any reader opens. A change only to the code that ran the steps (Dew's revision, the installed packages, the interpreter, the platform or the bytes of a step's module or of the factory that configured it) is a warning on a global resume and a refusal on a per-share one. A step states a change to what it computes in its configuration or its declared `identity`.

| Lowering | Position | Resume topology |
|---|---|---|
| Indexed, count-preserving or memory-materialized | Global consumed-record count | Any process count dividing the global row batch |
| Streamed, online rejecting, expanding or packing | Grain iterator state per share | The recorded share and worker topology |

Volatile fetches cannot claim exact resume until their bytes are persisted. They use the first-reader rule when readers replicate a share.

`Dataset.batch` is global. `DataPartition(index, count)` names a disjoint share; `DataPartition.of(mesh)` derives it from placement. Canonical validation takes accepted rows before the share cut. A record-level random transform's draw is keyed by the record's place in the run's order: the same in every read mode for one order, and by its place in its share in a read cut per share (one that drops records, or a streamed one). Model noise keyed by batch row can change when process placement changes.

[Checkpoints](../guides/checkpoints.md) and [distributed training](distributed.md) cover the trainer's restore and placement contracts.
