# Install Dew

This guide assumes you can create a Python virtual environment and run commands in a terminal. The package declares Python 3.11 or newer. Use Python 3.12 for the examples on these pages.

These commands use a POSIX shell and require [uv](https://docs.astral.sh/uv/getting-started/installation/) on your PATH. Install uv first if the command is unavailable; use the equivalent environment activation for your shell.

## Install from source

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install "dew-ml @ git+https://github.com/AshishKumar4/dew"
```

The distribution name is `dew-ml`; import it as `dew`. The command installs the current repository revision. Pin a Git commit in your experiment environment when you need to reproduce a run. The declared dependency lower bounds do not establish that every older JAX/Flax combination has been tested.

For a checkout you plan to edit:

```bash
git clone https://github.com/AshishKumar4/dew.git
cd dew
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[test]'
```

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

## Use a GPU or TPU

Install the JAX build appropriate to your operating system, driver, and accelerator using the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html). Configure the backend before importing JAX. For example, `JAX_PLATFORMS=cpu python train.py` selects CPU for a small smoke run.

For NVIDIA hardware, validate that JAX reports a CUDA device before running a GPU example. Dew can select cuDNN attention for supported GPU shapes and dtypes, but availability depends on the installed JAX and CUDA stack. A TPU requires its own runtime setup. [The TPU guide](tpu.md) describes Dew's provisioning commands; creating a cloud resource can incur charges and is not part of this quickstart.

## Add optional dependencies

The base package currently installs Transformers, Hugging Face Hub, WandB, and image-processing dependencies. These are not all optional in the package metadata. Additional extras cover the following uses:

| Extra | Use |
|---|---|
| `interop` | Read or write safetensors checkpoints |
| `streaming` | Hugging Face datasets and online sources |
| `av` | Video readers and image resizing |
| `metrics` | SciPy and download support used by image metrics |
| `tfds` | TensorFlow Datasets sources |
| `test` | Development tests and pinned reference-library version |

For example:

```bash
uv pip install 'dew-ml[interop,streaming] @ git+https://github.com/AshishKumar4/dew'
```

Qwix quantization and tokamax kernels require their respective external packages; they are not declared as installable Dew extras in the current package metadata. See [capabilities and limitations](reference/support.md) before enabling them.

## Build the documentation

From a checkout:

```bash
uv pip install mkdocs-material
python -m mkdocs build --strict
python -m mkdocs serve
```

Open the address printed by `mkdocs serve`. Documentation builds do not run accelerator examples or download model weights. Example execution is a separate validation step.
