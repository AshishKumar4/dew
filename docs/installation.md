# Install Dew

This guide assumes you can create a Python virtual environment and run commands in a terminal. Dew requires Python 3.12 or newer; Python 3.14 is the recommended training runtime and primary CI version. This is a compatibility and maintenance choice, not a claim that changing Python accelerates compiled JAX kernels.

These commands use a POSIX shell and require [uv](https://docs.astral.sh/uv/getting-started/installation/) on your PATH. Install uv first if the command is unavailable; use the equivalent environment activation for your shell.

## Install from source

```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install "dew-ml @ git+https://github.com/AshishKumar4/dew"
```

The distribution name is `dew-ml`; import it as `dew`. The command installs the current repository revision. Pin a Git commit in your experiment environment when you need to reproduce a run. The declared dependency lower bounds do not establish that every older JAX/Flax combination has been tested.

For a checkout you plan to edit:

```bash
git clone https://github.com/AshishKumar4/dew.git
cd dew
uv venv --python 3.14
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e '.[test,av,tfds,metrics,plots]'
```

This development install includes reference-test, optional data, and metric packages. CPU PyTorch is installed first for reference comparisons; Dew training still uses JAX. CI runs these extras on Python 3.12 and 3.14, with static type checking on the 3.12 compatibility floor. Ordinary users can start with the smaller base installation above.

Local reporting works with the base package. Add `[plots]` for Matplotlib charts or `[wandb]` for the optional Weights & Biases adapter.

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

A CPU device is enough for [the first training example](getting-started.md). `jax.devices()` lists the devices visible to this process, not every accelerator physically installed in the machine.

Compiled executables go to `~/.cache/dew/xla/python3.X`, one directory per Python minor version, so two interpreters on one machine never read each other's entries. JAX 0.11.1 compresses cache entries with the standard-library `compression.zstd` module on Python 3.14 but still labels them `zlib` in its cache key, so an older interpreter reading the same directory fails to decode them. A run that passes an explicit `compilation_cache_dir` keeps that exact path; do not share one explicit directory between Python versions until upstream JAX keys the codec.

## Use a GPU or TPU

Install the JAX build appropriate to your operating system, driver, and accelerator using the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html). Configure the backend before importing JAX. For example, `JAX_PLATFORMS=cpu python train.py` selects CPU for a small smoke run.

For NVIDIA hardware, validate that JAX reports a CUDA device before running a GPU example. Dew can select cuDNN attention for supported GPU shapes and dtypes, but availability depends on the installed JAX and CUDA stack. A TPU requires its own runtime setup. [The TPU guide](tpu.md) describes Dew's provisioning commands; creating a cloud resource can incur charges and is not part of this quickstart.

## Add optional dependencies

The base package currently installs Transformers, Hugging Face Hub, and image-processing dependencies. These are not all optional in the package metadata. Additional extras cover the following uses:

| Extra | Use |
|---|---|
| `interop` | Read or write safetensors checkpoints |
| `inference-clients` | Official Ollama/OpenAI Python clients, including vLLM-compatible endpoints |
| `vision` | Run the checkpoint's HF image processor using torchvision on the host |
| `streaming` | Hugging Face datasets and online sources |
| `av` | Video readers and image resizing |
| `metrics` | SciPy and download support used by image metrics |
| `tfds` | Read prepared TFDS ArrayRecords without TensorFlow |
| `test` | Development tests and pinned reference-library version |

For example:

```bash
uv pip install 'dew-ml[interop,streaming] @ git+https://github.com/AshishKumar4/dew'
```

For HF vision processors, install matching CPU PyTorch and torchvision wheels first,
then add Dew's vision extra. This keeps image preprocessing on the host without
installing CUDA PyTorch packages beside JAX's accelerator runtime. The native
Gemma4 processor path was checked with torch 2.14.0+cpu and torchvision 0.29.0+cpu; without the extra, the native multimodal checkpoints load but their processors raise on images.

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install 'dew-ml[interop,vision] @ git+https://github.com/AshishKumar4/dew'
```

Qwix quantization and tokamax kernels need their own packages; neither is a declared Dew extra. `uv pip install qwix` adds the quantization one.

## Prepare TFDS data separately

The training environment reads prepared ArrayRecords without TensorFlow.
Dataset preparation can require TensorFlow and dataset-specific packages;
Oxford Flowers uses SciPy to read its MATLAB label and split files. Keep
those dependencies in a separate Python 3.13 environment:

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

Preparation downloads the corpus when needed. Copy its final printed export
line into the training shell. The path is the prepared version directory,
not the parent TFDS cache directory:

```python
import os
from dew.data import Loading, OxfordFlowers

data = OxfordFlowers(
    path=os.environ["DEW_FLOWERS_PATH"],
    image_size=64,
    loading=Loading(workers=0),
).load(batch=4)
```

Recipes receive the same location as `--data.path "$DEW_FLOWERS_PATH"`.
TFDS writes `label.labels.txt` there; `labels=None` reads that file, and
`--data.labels` can supply a different class-name file. The reader uses
`tfds.builder_from_directory(...).as_data_source(...)`. Missing metadata or
selected shards produce an error requesting external preparation; training
does not download, prepare, or fall back to dataset generation code.
TensorFlow 2.21.0 has no Python 3.14 wheels; that restricts preparation, not
this TensorFlow-free training path.

## Build the documentation

From a checkout:

```bash
uv pip install mkdocs-material
python -m mkdocs build --strict
python -m mkdocs serve
```

Open the address printed by `mkdocs serve`. Documentation builds do not run accelerator examples or download model weights. Example execution is a separate validation step.
