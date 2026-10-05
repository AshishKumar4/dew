# Installation

Install `dewml` from the GitHub repository and import it as `dew`. You need Python 3.12 or newer. CI tests Python 3.12 and 3.14. The commands below use [uv](https://docs.astral.sh/uv/getting-started/installation/) and a POSIX shell.

## Installing from GitHub

```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install "dewml @ git+https://github.com/AshishKumar4/dew"
```

This installs the repository's current revision. To reproduce a run later, pin a commit with `"dewml @ git+https://github.com/AshishKumar4/dew@<commit>"`. Dew installs JAX 0.11.2 and Flax 0.12.10 or a later 0.12 release from PyPI. For a process pool across GPUs, you need a patched JAX to keep the compilation cache. See [Process pools across GPUs](#process-pools-across-gpus).

The plain install runs JAX on the CPU, which is enough for the [Quickstart](getting-started.md).

## Checking the environment

```python
import jax
import flax
import dew

print("Dew:", dew.__version__)
print("JAX:", jax.__version__)
print("Flax:", flax.__version__)
print("Devices:", jax.devices())
```

`jax.devices()` lists the devices this process can use, which can be fewer than the accelerators in the machine.

## GPUs and TPUs

Add the extra that matches the hardware:

| Hardware | Command |
|---|---|
| NVIDIA GPU, CUDA 13 driver | `uv pip install "dewml[cuda13] @ git+https://github.com/AshishKumar4/dew"` |
| NVIDIA GPU, CUDA 12 driver | `uv pip install "dewml[cuda12] @ git+https://github.com/AshishKumar4/dew"` |
| Google TPU VM | `uv pip install "dewml[tpu] @ git+https://github.com/AshishKumar4/dew"` |

These are JAX's own extras, with the same names. Each installs the accelerator build of JAX 0.11.2, the version Dew requires. Running `-U "jax[...]"` later would replace it with a release Dew has not been tested on. You can combine extras, as in `dewml[cuda13,interop,streaming]`.

The [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) lists the driver each build needs. Set `JAX_PLATFORMS` before importing JAX to select the backend. For example, `JAX_PLATFORMS=cpu python train.py` runs on the CPU even on a GPU machine. The Colab tutorials install `dewml[cuda13]`. See [Cloud TPUs](tpu.md) to provision a TPU.

### Process pools across GPUs

JAX 0.11.2 uses the compiling process's topology in its cache key, including the GPU's NVLink connections. If GPUs in a process pool have different connections, their cache keys differ for the same step. On the next run, some processes load the step from the persistent cache while others compile it. Compilation then waits indefinitely for every process to join.

The fix ([jax-ml/jax#40940](https://github.com/jax-ml/jax/issues/40940)) is not in a JAX release yet. With the released 0.11.2, Dew disables the persistent cache for GPU pools and prints a notice on every process. Training works, but each run recompiles its steps. To keep the cache, install the patched build of 0.11.2 named in `constraints.txt`:

```bash
uv pip install "dewml[cuda13] @ git+https://github.com/AshishKumar4/dew" \
    -c https://raw.githubusercontent.com/AshishKumar4/dew/main/constraints.txt
```

A single-process pool or a TPU or CPU pool needs no patch.

## Optional extras

The plain install includes Transformers, the Hugging Face Hub client and the image libraries. Use these extras for other dependencies:

| Extra | Adds |
|---|---|
| `interop` | safetensors reading and writing |
| `torch` | PyTorch on the host, to read `pytorch_model.bin` files and to check unregistered decoders against Transformers |
| `diffusers` | Loading original-format single-file diffusion checkpoints (`Pretrained.load(..., single_file=)`) |
| `gguf` | Loading GGUF files (`Pretrained.load(..., gguf_file=)`) |
| `wan` | ftfy, which cleans Wan 2.1's prompts as its pipeline does |
| `torchax` | `Pretrained.load(fallback="torchax")`, which runs a Transformers PyTorch model lowered to JAX |
| `guided` | Regex and JSON-schema guided decoding (`dew.sampling.guided`) |
| `streaming` | Hugging Face `datasets` and online sources |
| `tfds` | Reading prepared TFDS ArrayRecords without TensorFlow |
| `av` | Video readers (moviepy, with ffmpeg) |
| `vision` | Hugging Face image processors, run with torchvision on the host |
| `metrics` | SciPy and downloads for image metrics |
| `plots` | Matplotlib charts in local reports |
| `wandb`, `mlflow`, `tensorboard` | Experiment trackers |
| `inference-clients` | The Ollama and OpenAI Python clients, including vLLM-compatible endpoints |
| `hpo` | Optuna, for `dew.config.sweep` |
| `eval-harness` | lm-evaluation-harness tasks through `dew.eval.harness.DewLM` |
| `profile` | The xprof profiler |
| `quantization` | Qwix, for quantized training and serving (`dew.training.quantization`) |
| `test` | The test suite's dependencies and pinned reference libraries |

```bash
uv pip install 'dewml[interop,streaming] @ git+https://github.com/AshishKumar4/dew'
```

Before installing the `vision` extra, install CPU builds of PyTorch and torchvision. This keeps CUDA PyTorch out of JAX's accelerator environment. Without `vision`, native multimodal checkpoints still load, but their processors raise an error when given images.

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install 'dewml[interop,vision] @ git+https://github.com/AshishKumar4/dew'
```

The `profile` extra installs XProf 2.23.1 or later, excluding 2.23.2. That release requires `setuptools<70`, while PyTorch 2.13 and later require `setuptools>=77.0.3`. The conflicting requirements prevent installation with the `torch`, `vision`, `diffusers`, `torchax` or `test` extras. Don't upgrade XProf to 2.23.2 by hand in such an environment.

tokamax is optional. If you install it, `auto` uses its Pallas-Triton attention on sm80 and later for heads up to 64 wide (`dew.nn.attention.triton_runs`). Install it with `uv pip install tokamax -c https://raw.githubusercontent.com/AshishKumar4/dew/main/constraints.txt`. The latest release, tokamax 0.0.14, pins `typeguard==2.13.3`, which tyro excludes. The constraints select the tokamax commit that removed that pin.

## Development install

```bash
git clone https://github.com/AshishKumar4/dew.git
cd dew
uv venv --python 3.14
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -e '.[test,av,tfds,metrics,plots,inference-clients,vision,quantization,profile]' -c constraints.txt
```

CI installs these extras with the patched JAX 0.11.2 named in `constraints.txt`. The multi-process cache tests need that patch. Only the reference tests and image processors use PyTorch. Dew runs model computation in JAX.

## Compilation cache

Dew stores compiled executables in `~/.cache/dew/xla/python3.X`. If `XDG_CACHE_HOME` is set, it uses `$XDG_CACHE_HOME/dew/xla/python3.X` instead. Each Python minor version gets a separate directory. JAX 0.11 on Python 3.14 compresses entries with `compression.zstd`, but its cache key still says `zlib`. Older interpreters cannot read those entries.

If you pass `compilation_cache_dir`, Dew uses that path as given. Do not share an explicit cache directory between Python versions.

## Preparing TFDS data

Training reads prepared TFDS ArrayRecords without importing TensorFlow. To prepare a dataset, you need TensorFlow and sometimes dataset-specific packages. Oxford Flowers, for example, reads its label files with SciPy. TensorFlow 2.21.0 has no Python 3.14 wheels, so prepare the data in a separate Python 3.13 environment:

```bash
uv venv --python 3.13 .venv-tfds-prepare
uv pip install --python .venv-tfds-prepare/bin/python \
    tensorflow-datasets==4.9.10 tensorflow==2.21.0 scipy
export TFDS_DATA_DIR="$HOME/tensorflow_datasets"
.venv-tfds-prepare/bin/python - <<'PY'
import shlex
import tensorflow_datasets as tfds

builder = tfds.builder("oxford_flowers102", try_gcs=False)
builder.download_and_prepare(
    file_format="array_record",
    download_config=tfds.download.DownloadConfig(try_download_gcs=False),
)
print("export DEW_FLOWERS_PATH=" + shlex.quote(str(builder.data_dir)))
PY
```

The script downloads the data if needed and prints an `export` line for the prepared version directory. Run that line in the shell where you will train:

<!-- not run: needs the prepared TFDS directory the script above writes -->
```python
import os
from dew.data import Loading, TFDSImages

data = TFDSImages(
    path=os.environ["DEW_FLOWERS_PATH"],
    image_size=64,
    loading=Loading(workers=0),
).load(batch=4)
```

Pass the same path to a recipe with `--data.path "$DEW_FLOWERS_PATH"`. The diffusion recipe's `data:oxford-flowers102` reads it with flower captions. With `labels=None`, `TFDSImages` reads class names from the `label.labels.txt` file that TFDS writes there. To use another file, set `--data.labels`. If metadata or shards are missing, the reader raises an error asking you to prepare the data. Training never downloads or prepares TFDS data.

## Building the documentation

The site at [dewml.dev](https://dewml.dev) uses pages from `docs/`, notebooks from `tutorials/` with their recorded outputs, and API pages generated from `src/dew` docstrings. To build it, you need Node 22.12 or newer, [pnpm](https://pnpm.io) and uv:

```bash
cd site
pnpm install
pnpm build
pnpm preview
```

`pnpm build` fails on a broken link, a notebook that was not executed top to bottom, a public module without an API page, or a model family the supported-models page does not list. `pnpm dev` serves the site with live reload.
