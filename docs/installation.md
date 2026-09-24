# Install Dew

This guide assumes you can create a Python virtual environment and run commands in a terminal. Dew needs Python 3.12 or newer. I recommend Python 3.14 for training, and it is the main version CI runs. That choice is about compatibility and maintenance. A newer Python does not make compiled JAX kernels faster.

The commands below use a POSIX shell and need [uv](https://docs.astral.sh/uv/getting-started/installation/) on your PATH. If the `uv` command is missing, install it first. If your shell is not POSIX, use its own way of activating a virtual environment.

## Install from source

```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install "dew-ml @ git+https://github.com/AshishKumar4/dew"
```

The package is called `dew-ml`, and you import it as `dew`. This command installs whatever revision the repository has today. If you need to reproduce a run later, pin a Git commit in your experiment environment. Dew pins JAX and Flax to builds from GitHub that carry fixes it needs, until releases ship them; `pyproject.toml` names the commits.

If you plan to edit the code, clone it instead:

```bash
git clone https://github.com/AshishKumar4/dew.git
cd dew
uv venv --python 3.14
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -e '.[test,av,tfds,metrics,plots,inference-clients,vision]'
```

This development install adds the reference tests, the data providers, the metrics, the inference clients, and the vision processors. Install the CPU builds of PyTorch and torchvision first, since Dew runs its model computation in JAX. CI installs the same extras on Python 3.12 and 3.14, and type-checks against 3.12, the oldest supported version. The plain install above is smaller.

Local reporting works with the plain install. Add `[plots]` for Matplotlib charts, or `[wandb]`, `[mlflow]`, or `[tensorboard]` for those trackers.

## Check the environment

```python
import jax
import flax
import dew

print("Dew:", dew.__version__)
print("JAX:", jax.__version__)
print("Flax:", flax.__version__)
print("Devices:", jax.devices())
```

A CPU is enough for [the first training example](getting-started.md). `jax.devices()` lists the devices this process can see. That can be fewer than the accelerators physically in the machine.

Dew stores compiled executables in `~/.cache/dew/xla/python3.X`, or under `$XDG_CACHE_HOME/dew/xla/` when that variable is set. Each Python minor version gets its own directory, so two interpreters on one machine never read each other's entries. The reason is a JAX 0.11.1 problem: on Python 3.14 it compresses cache entries with the standard-library `compression.zstd` module, but its cache key still says `zlib`. An older interpreter that reads the same directory then fails to decode them. If a run passes its own `compilation_cache_dir`, Dew uses that exact path. Until JAX puts the codec in its cache key, do not share one explicit cache directory between Python versions.

## Use a GPU or TPU

The plain install brings JAX for the CPU. For an accelerator, add the extra that matches it:

| Hardware | Command |
|---|---|
| NVIDIA GPU, CUDA 13 driver | `uv pip install "dew-ml[cuda13] @ git+https://github.com/AshishKumar4/dew"` |
| NVIDIA GPU, CUDA 12 driver | `uv pip install "dew-ml[cuda12] @ git+https://github.com/AshishKumar4/dew"` |
| Google TPU VM | `uv pip install "dew-ml[tpu] @ git+https://github.com/AshishKumar4/dew"` |

Extras combine, as in `dew-ml[cuda13,interop,streaming]`, and a checkout takes them the same way: `uv pip install -e ".[cuda12]"`. Each one installs the accelerator build of the JAX that Dew pins, a build of 0.11.2 from GitHub that fixes its compilation cache for pools of processes on different GPUs. pip cannot resolve PyPI's JAX extras, such as `jax[cuda13]`, beside that pin in one install, and once PyPI has a JAX newer than the pin, a later `-U "jax[...]"` with pip or uv replaces the pin with it. Use Dew's extras instead.

The [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) lists the driver each build needs. Choose the backend before you import JAX; for example, `JAX_PLATFORMS=cpu python train.py` runs a small smoke test on the CPU even on a GPU machine. On Colab the tutorials install `dew-ml[cuda13]`.

On NVIDIA hardware, check that JAX lists a CUDA device before you run a GPU example. Dew can use cuDNN attention for the GPU shapes and dtypes that support it, but whether it is available depends on your JAX and CUDA install. A TPU needs its own runtime setup. [The TPU guide](tpu.md) describes Dew's provisioning commands. Creating a cloud resource can cost money, so it is not part of this quickstart.

## Add optional dependencies

The plain install already includes Transformers, Hugging Face Hub, and the image-processing libraries. The package metadata does not make these optional. The extras below add more:

| Extra | Use |
|---|---|
| `interop` | Read or write safetensors checkpoints |
| `inference-clients` | Official Ollama/OpenAI Python clients, including vLLM-compatible endpoints |
| `vision` | Run the checkpoint's HF image processor using torchvision on the host |
| `streaming` | Hugging Face datasets and online sources |
| `av` | Video readers (moviepy, which brings ffmpeg) |
| `metrics` | SciPy and download support used by image metrics |
| `tfds` | Read prepared TFDS ArrayRecords without TensorFlow |
| `profile` | The xprof profiler |
| `hpo` | Optuna, the backend of `dew.config.sweep`'s Optuna search |
| `eval-harness` | lm-evaluation-harness task suites through `dew.eval.harness.DewLM` |
| `test` | Development tests and pinned reference-library version |

For example:

```bash
uv pip install 'dew-ml[interop,streaming] @ git+https://github.com/AshishKumar4/dew'
```

For HF vision processors, install matching CPU PyTorch and torchvision wheels first, then add Dew's `vision` extra. The image preprocessing then runs on the host, and no CUDA PyTorch packages sit next to JAX's accelerator runtime. I checked the native Gemma4 processor path with torch 2.14.0+cpu and torchvision 0.29.0+cpu. Without the extra, the native multimodal checkpoints still load, but their processors raise an error when given images.

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install 'dew-ml[interop,vision] @ git+https://github.com/AshishKumar4/dew'
```

Qwix quantization and tokamax kernels need their own packages, and Dew has no extra for either. `uv pip install qwix` adds Qwix.

## Prepare TFDS data separately

Training reads prepared ArrayRecords and does not need TensorFlow. Preparing a dataset can need TensorFlow and packages specific to that dataset. Oxford Flowers, for example, uses SciPy to read its MATLAB label and split files. Keep those in a separate Python 3.13 environment:

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

Preparation downloads the data if it is not there yet. Copy the `export` line it prints last into the shell you train from. The path is the prepared version directory, not the parent TFDS cache directory:

```python
import os
from dew.data import Loading, OxfordFlowers

data = OxfordFlowers(
    path=os.environ["DEW_FLOWERS_PATH"],
    image_size=64,
    loading=Loading(workers=0),
).load(batch=4)
```

Recipes take the same path as `--data.path "$DEW_FLOWERS_PATH"`. TFDS writes `label.labels.txt` into that directory. With `labels=None`, Dew reads the class names from that file; `--data.labels` points at a different class-name file. The reader uses `tfds.builder_from_directory(...).as_data_source(...)`. If the metadata or the selected shards are missing, it raises an error that asks you to prepare the data first. Training never downloads or prepares data, and never falls back to the dataset's generation code. TensorFlow 2.21.0 has no Python 3.14 wheels. That only limits the preparation environment; the training path does not use TensorFlow.

## Build the documentation

The site at [dewml.dev](https://dewml.dev) is built from this repository: the prose in `docs/`, the notebooks in `tutorials/` with their recorded outputs, and an API reference generated from the docstrings in `src/dew`. Building it needs Node 22.12 or newer, [pnpm](https://pnpm.io) and uv:

```bash
cd site
pnpm install
pnpm build
pnpm preview
```

`pnpm build` fails on a broken link, a notebook that was not executed top to bottom, a public module without an API page, or a model family the supported-models page does not name. `pnpm dev` serves the site with live reload. Building it runs no model code and downloads no weights.
