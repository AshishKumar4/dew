# Installation

Dew's package name is `dewml` and its import name is `dew`; it is installed from the GitHub repository. It needs Python 3.12 or newer; CI tests Python 3.12 and 3.14. The commands below use [uv](https://docs.astral.sh/uv/getting-started/installation/) and a POSIX shell.

## Installing from GitHub

```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install "dewml @ git+https://github.com/AshishKumar4/dew"
```

This installs the current revision of the repository. To reproduce a run later, pin a commit: `"dewml @ git+https://github.com/AshishKumar4/dew@<commit>"`. Dew installs JAX 0.11.2 and Flax 0.12.10 or a later 0.12 release from PyPI. A process pool across GPUs needs a patched JAX to keep its compilation cache; [Process pools across GPUs](#process-pools-across-gpus) covers it.

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

Each extra installs the accelerator build of JAX 0.11.2, the version Dew requires; they are JAX's own extras of the same names. A later `-U "jax[...]"` would replace it with a release Dew isn't tested on. Extras combine, as in `dewml[cuda13,interop,streaming]`.

### Which CUDA build

The driver and the GPU decide the build, newest first:

| Build | NVIDIA driver | GPU (SM) |
|---|---|---|
| `cuda13` | 580 or later | 7.5 or later |
| `cuda12` | 525 or later | 5.2 or later |

`nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader` prints both; with several GPUs the oldest SM counts. These are the minimums in the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) and in Table 3 of NVIDIA's [CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). Neither build runs on an older driver or GPU. The [dewml.dev install script](https://dewml.dev/install.sh) applies the same rule.

`import dew` applies it too: on a machine with an NVIDIA GPU, it stops when JAX would run on the CPU, because no CUDA build is installed or the installed one can't run on this driver, and it prints the install that fixes it. `JAX_PLATFORMS` selects the backend before JAX is imported. `JAX_PLATFORMS=cpu python train.py` runs on the CPU on a GPU machine, and the check leaves it alone. On Colab the tutorials install `dewml[cuda13]`. [Cloud TPUs](tpu.md) covers TPU provisioning.

### Process pools across GPUs

JAX 0.11.2 keys a compiled step by the topology of the process that compiled it, which on a GPU includes its NVLink links. In a process pool across GPUs that are linked differently, the processes key the same step differently. On the next run some load it from the persistent compilation cache while the others compile it, and that compile then waits forever for every process to join. The fix ([jax-ml/jax#40940](https://github.com/jax-ml/jax/issues/40940)) is in no JAX release yet. With the 0.11.2 release, Dew compiles a GPU pool without the persistent cache and prints on every process that it does; the pool trains correctly, but compiles its steps again on every run. To keep the cache, install the patched build of 0.11.2 that `constraints.txt` names:

```bash
uv pip install "dewml[cuda13] @ git+https://github.com/AshishKumar4/dew" \
    -c https://raw.githubusercontent.com/AshishKumar4/dew/main/constraints.txt
```

A pool of one process, and a TPU or CPU pool, needs nothing more.

## Optional extras

The plain install includes Transformers, the Hugging Face Hub client and the image libraries. These extras add more:

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
| `serve` | Starlette and uvicorn, for serving a decision model in Jev's wire format (`examples/serve_decisions.py`) |
| `hpo` | Optuna, for `dew.config.sweep` |
| `eval-harness` | lm-evaluation-harness tasks through `dew.eval.harness.DewLM` |
| `profile` | The xprof profiler |
| `quantization` | Qwix, for quantized training and serving (`dew.training.quantization`) |
| `test` | The test suite's dependencies and pinned reference libraries |

```bash
uv pip install 'dewml[interop,streaming] @ git+https://github.com/AshishKumar4/dew'
```

The `vision` extra expects CPU builds of PyTorch and torchvision, installed first, so that no CUDA PyTorch sits beside JAX's accelerator runtime. Without it the native multimodal checkpoints still load, but their processors raise an error when given images.

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install 'dewml[interop,vision] @ git+https://github.com/AshishKumar4/dew'
```

The `profile` extra installs XProf 2.23.1 or a later release, never 2.23.2. XProf 2.23.2 declares `setuptools<70`, and PyTorch 2.13 and later declare `setuptools>=77.0.3`, so 2.23.2 can't be installed beside the `torch`, `vision`, `diffusers`, `torchax` or `test` extras. Don't upgrade XProf to 2.23.2 by hand in such an environment.

FlashAttention-2 comes as Dew's own wheel, `dew-flash-attn-cu12` or `dew-flash-attn-cu13` (kernels/flash_attn), for the CUDA major of your jax. It is not on PyPI yet: install the wheel the FlashAttention workflow builds with `uv pip install dew_flash_attn_cu13-0.1.0-py3-none-manylinux_2_28_x86_64.whl`. With it installed, `"auto"` runs it on an A100 (sm80) and on Ada GPUs (sm89: the RTX 40 series, L4, L40S) for calls without a window, mask, bias or softcap and heads up to 256 wide, 192 on Ada (`dew.nn.attention.flash_refusal`); a call it turns down on a GPU where it was measured fastest is logged once, with why.

tokamax is not a dependency. With it installed, `"auto"` runs its Pallas-Triton attention on sm80 and later for heads up to 64 wide (`dew.nn.attention.triton_runs`). Install it with `uv pip install 'tokamax>=0.0.15'`. Earlier releases pin `typeguard==2.13.3`, which tyro excludes, so installing one breaks every recipe's command line.

## Development install

```bash
git clone https://github.com/AshishKumar4/dew.git
cd dew
uv venv --python 3.14
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -e '.[test,av,tfds,metrics,plots,inference-clients,serve,vision,quantization,profile]' -c constraints.txt
```

These are the extras CI installs, with the same JAX build CI uses: `constraints.txt` names the patched build of 0.11.2, which the multi-process cache tests need. PyTorch is used only by the reference tests and the image processors; Dew's model computation runs in JAX.

## Compilation cache

Dew stores compiled executables in `~/.cache/dew/xla/python3.X`, or `$XDG_CACHE_HOME/dew/xla/python3.X` when `XDG_CACHE_HOME` is set. Each Python minor version has its own directory because JAX 0.11 on Python 3.14 compresses entries with `compression.zstd` while its cache key still says `zlib`, and an older interpreter cannot read them. A run that passes its own `compilation_cache_dir` uses that path as given, so do not share one explicit directory between Python versions.

## Preparing TFDS data

Training reads prepared TFDS ArrayRecords and does not import TensorFlow. Preparing a dataset does need TensorFlow and sometimes dataset-specific packages (Oxford Flowers reads its label files with SciPy). TFDS 4.9.10 also imports `importlib_resources` while it prepares a dataset but only declares it for Python before 3.9, so the command installs it too.

Where the training environment has them (Python 3.13 or earlier, with `tensorflow` and `importlib_resources` installed), `dew.data.load("tfds/mnist", batch=64)` and `TFDSImages(name="mnist")` prepare the builder themselves, in a separate Python process, into `~/.cache/dew/tfds` (`$XDG_CACHE_HOME/dew/tfds` when that is set), and every later call reads it from there, offline. TensorFlow 2.21.0 has no Python 3.14 wheels, so on 3.14 prepare in a separate Python 3.13 environment:

```bash
uv venv --python 3.13 .venv-tfds-prepare
uv pip install --python .venv-tfds-prepare/bin/python \
    tensorflow-datasets==4.9.10 tensorflow==2.21.0 scipy importlib_resources
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

The script downloads the data if needed and prints an `export` line with the prepared version directory. To have `load("tfds/<builder>")` and `TFDSImages(name=)` find it without a path, prepare it with `data_dir` set to the training environment's `~/.cache/dew/tfds` instead. Otherwise run the printed line in the shell you train from:

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

Recipes take the same path as `--data.path "$DEW_FLOWERS_PATH"`; the diffusion recipe's `data:oxford-flowers102` reads it with flower captions. With `labels=None`, `TFDSImages` reads the class names from the `label.labels.txt` file TFDS writes there; `--data.labels` names another file. If the metadata or shards under a `path` are missing, the reader raises an error that asks you to prepare the data; Dew prepares only into its own directory, never into a `path` you name.

## Building the documentation

The site at [dewml.dev](https://dewml.dev) is built from `docs/`, the notebooks in `tutorials/` with their recorded outputs, and API pages generated from the docstrings in `src/dew`. It needs Node 22.12 or newer, [pnpm](https://pnpm.io) and uv:

```bash
cd site
pnpm install
pnpm build
pnpm preview
```

`pnpm build` fails on a broken link, a notebook that was not executed top to bottom, a public module without an API page, or a model family the supported-models page does not list. `pnpm dev` serves the site with live reload.
