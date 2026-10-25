# Installation

Dew's package name is `dewml` and its import name is `dew`.

<!-- BEGIN least Python (site/scripts/gen_install.py) -->
Dew needs Python 3.12 or newer, and CI tests 3.12 and 3.14.
<!-- END least Python -->

## The install script

```bash
curl -LsSf https://dewml.dev/install.sh | sh
```

The script shows the hardware it found (an NVIDIA GPU with its driver and the CUDA build that driver runs, a TPU VM, Apple silicon or the CPU) and the Python environment that is active. It asks one question, where to install: the active environment, a new `.venv` in the current directory, a conda environment when conda is installed, or another path. Enter takes the first. It installs [uv](https://docs.astral.sh/uv/) with uv's own installer if uv is missing, prints the install command and runs it, then prints the dewml and JAX versions it installed, a command that lists the devices JAX sees, and how to activate the environment.

With `-y`, or `DEW_YES=1`, or no terminal to ask on, it takes every default. Options choose for it: `--cuda 12` or `--cuda 13`, `--tpu`, `--cpu`, `--venv PATH`, `--conda NAME`, `--version X`, and `--pip` to use pip instead of uv. They follow `sh -s --`:

```bash
curl -LsSf https://dewml.dev/install.sh | sh -s -- -y --venv ~/envs/dew --cuda 12
```

## uv or pip

Dew runs on its own build of JAX ([Dew's JAX build](#dews-jax-build)), which Dew's package index serves, so every install names it:

```bash
uv pip install dewml --extra-index-url https://pypi.dewml.dev/simple
```

Each command installs that JAX build and the matching accelerator build of jaxlib for the hardware:

<!-- BEGIN install commands (site/scripts/gen_install.py) -->
| Hardware | uv | pip |
|---|---|---|
| CPU | `uv pip install dewml --extra-index-url https://pypi.dewml.dev/simple` | `pip install dewml --extra-index-url https://pypi.dewml.dev/simple` |
| NVIDIA GPU, driver 580+ (CUDA 13) | `uv pip install "dewml[cuda13]" --extra-index-url https://pypi.dewml.dev/simple` | `pip install "dewml[cuda13]" --extra-index-url https://pypi.dewml.dev/simple` |
| NVIDIA GPU, driver 525+ (CUDA 12) | `uv pip install "dewml[cuda12]" --extra-index-url https://pypi.dewml.dev/simple` | `pip install "dewml[cuda12]" --extra-index-url https://pypi.dewml.dev/simple` |
| Google TPU VM | `uv pip install "dewml[tpu]" --extra-index-url https://pypi.dewml.dev/simple` | `pip install "dewml[tpu]" --extra-index-url https://pypi.dewml.dev/simple` |
<!-- END install commands -->

<!-- BEGIN pip (site/scripts/gen_install.py) -->
Use pip 26.2.2 or newer and uv 0.13.4 or newer: older ones trust the roots of eMudhra's compromised certificate authorities (October 2026). Update pip from its wheel by hash, so the update holds whatever the older pip trusts, and uv through the system's certificates once they are updated:

```bash
python -m pip install "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
UV_SYSTEM_CERTS=true uv self update
```
<!-- END pip -->

Without the index the install stops before it installs anything, at resolving jax, since PyPI has no JAX build Dew accepts ([Dew's JAX build](#dews-jax-build)): uv reports `No solution found when resolving dependencies` and names dewml's jax requirement, and pip reports `No matching distribution found for jax`. Add `--extra-index-url https://pypi.dewml.dev/simple` and run it again.

[Which CUDA build](#which-cuda-build) explains the choice between the two. With uv, make an environment first:

```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install dewml --extra-index-url https://pypi.dewml.dev/simple
```

To install the current revision of the repository instead of the release, name it: `uv pip install "dewml @ git+https://github.com/AshishKumar4/dew" --extra-index-url https://pypi.dewml.dev/simple`, and pin a commit, `"dewml @ git+https://github.com/AshishKumar4/dew@<commit>"`, to reproduce a run later. Dew installs its JAX build and the Flax release pyproject.toml's dependencies allow.

The CPU install is enough for the [Quickstart](getting-started.md).

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

The `cuda13`, `cuda12` and `tpu` extras each install the accelerator build of jaxlib that matches Dew's JAX build; they are JAX's own extras of the same names. A later `-U "jax[...]"` would replace it with a release Dew isn't tested on. Extras combine, as in `dewml[cuda13,interop,streaming]`.

### Which CUDA build

The driver and the GPU decide the build, newest first:

<!-- BEGIN CUDA builds (site/scripts/gen_install.py) -->
| Build | CUDA | NVIDIA driver | GPU (SM) |
|---|---|---|---|
| `cuda13` | 13.0 or later | 580 or later, or an older one under NVIDIA's cuda-compat | 7.5 or later |
| `cuda12` | 12.0 or later | 525 or later, or an older one under NVIDIA's cuda-compat | 5.2 or later |
<!-- END CUDA builds -->

`nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader` prints both; with several GPUs the oldest SM counts. These are the minimums in the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) and in Table 3 of NVIDIA's [CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). The CUDA version is the one the driver library a process loads supports: a driver's own, or a newer one under NVIDIA's [forward-compatibility package](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html) (`cuda-compat`), which runs the `cuda13` build on a 570 driver; Dew reads it from the library (`cuDriverGetVersion`). Neither build runs on an older GPU. The [dewml.dev install script](https://dewml.dev/install.sh) chooses by the driver alone.

A run applies it too: on a machine with an NVIDIA GPU, `prepare_process`, or the first `Trainer.fit` of a program that never called it, raises `dew.devices.CPUOnlyJax` when JAX opened on the CPU, naming the install that fixes it where no CUDA build is installed or the installed one can't run on this driver. `import dew` asks nothing of the hardware, so the check runs once the process's XLA flags are set. `JAX_PLATFORMS` selects the backend before JAX is imported. `JAX_PLATFORMS=cpu python train.py` runs on the CPU on a GPU machine, and the check leaves it alone. On Colab the tutorials install `dewml[cuda13]`. [Cloud TPUs](tpu.md) covers TPU provisioning.

### Dew's JAX build

<!-- BEGIN JAX requirement (site/scripts/gen_install.py) -->
Dew requires `jax>=0.12.0.1.post24,<0.12.1`.
<!-- END JAX requirement -->

It is the JAX release that requirement names, with fixes no JAX release has yet, built from [AshishKumar4/jax](https://github.com/AshishKumar4/jax) and served by Dew's package index, the one every install above names, which serves jax alone. Every other package, jaxlib and its CUDA plugins included, comes from PyPI. The fixes:

- A computation that spans processes keys the persistent compilation cache alike on every process ([jax-ml/jax#40940](https://github.com/jax-ml/jax/issues/40940), sent as [jax-ml/jax#41776](https://github.com/jax-ml/jax/pull/41776)). The 0.11.2 release keys a compiled step by the topology of the process that compiled it, which on a GPU includes its NVLink links, so in a pool across GPUs linked differently some processes load a step from the cache while the others compile it, and that compile waits forever for every process to join.
- A Pallas Triton kernel's cache key leaves out its source locations ([jax-ml/jax#41502](https://github.com/jax-ml/jax/issues/41502), sent as [jax-ml/jax#41777](https://github.com/jax-ml/jax/pull/41777)), so two checkouts share one compiled kernel.
- `linear_call` batches (an all-unmapped call stays unmapped), and differentiates in its residuals, which the experts' fused SwiGLU route needs.
- The device-to-host transfer guard holds on a CPU array's implicit host read (`float`, `item`, `np.asarray`, its buffer) as on an accelerator's copy, which a steady state's guard needs on the CPU suite.
- Pallas Triton divides 16-bit floats and computes `erf_inv` through float32, correctly rounded, which a GEMM template's prologue dividing in bfloat16 needs on CUDA.
- A `custom_vjp` call (and Mosaic GPU's `custom_jvp` call) in a Pallas Triton or Mosaic GPU kernel lowers as its primal function, which a template's score hook through a `jax.custom_vjp` needs.
- Mosaic GPU relays an MMA's accumulator out to a 16-bit MMA's left operand in registers, and an abstract device states its compute capability, which Mosaic GPU lowers for: the templates' Mosaic kernels need both on sm80.
- The persistent compilation cache keys a CPU program by the host's CPU, which XLA:CPU compiles for, and every program by JAX's own build as well as jaxlib's, so neither another CPU nor another build over one jaxlib reads a program it cannot run.
- Mosaic GPU and Pallas Mosaic GPU lower `ceil` and `floor`, which a template's Mosaic kernel that rounds a value needs on an H100.
- Pallas Mosaic GPU lowers `add_any` and `integer_pow` of every integer exponent, which the Mosaic attention backward under a score hook needs: a hook's transposed pullback sums its cotangents, and a temperature's backward raises it to the power -2.
- Splash attention's `BlockSizes.delta_from_weights` forms its backward's D from the weights, in a dq pass of its own, so a query with one key gets exactly zero gradient; it is off unless a kernel turns it on.
- Mosaic GPU's layout inference solves groups of unknowns that no constraint relates separately, so sm80's `mma.sync` GEMM tiles find their layouts.
- A platform-dependent conditional whose lowering keeps one branch lowers that branch inline, so a collective inside it is not a runtime conditional; on the CPU, devices diverged on such a conditional.

Each fix goes from the build once a JAX release has it, and Dew then returns to that release.

## Optional extras

The plain install includes Transformers, the Hugging Face Hub client and the image libraries. These extras add more:

<!-- BEGIN extras (site/scripts/gen_install.py checks them against pyproject.toml) -->
| Extra | Adds |
|---|---|
| `cuda12`, `cuda13` | JAX's CUDA build and Dew's FlashAttention wheel for that CUDA major ([Which CUDA build](#which-cuda-build)) |
| `tpu` | libtpu, for [Cloud TPUs](tpu.md) |
| `interop` | safetensors reading and writing |
| `torch` | PyTorch on the host, to read `pytorch_model.bin` files and to check unregistered decoders against Transformers |
| `diffusers` | Loading original-format single-file diffusion checkpoints (`Pretrained.load(..., single_file=)`) |
| `gguf` | Loading GGUF files (`Pretrained.load(..., gguf_file=)`) |
| `wan` | ftfy, which cleans Wan 2.1's prompts as its pipeline does |
| `torchax` | Tier 3 of `Pretrained.load`, which runs a Transformers PyTorch model lowered to JAX: an unregistered causal LM, or any checkpoint with `fallback="torchax"` |
| `guided` | Regex and JSON-schema guided decoding (`dew.sampling.guided`) |
| `streaming` | Hugging Face `datasets` and online sources |
| `delta` | deltalake, to read Delta tables |
| `curation` | DataTrove's language, quality, fastText and decontamination filters with spaCy's word splitting (`dew.data.text.Language`, `Quality`, `Classifier`, `Decontaminated`) |
| `tfds` | Reading prepared TFDS ArrayRecords without TensorFlow |
| `av` | Video readers (moviepy, with ffmpeg) |
| `vision` | Hugging Face image processors, run with torchvision on the host |
| `augment` | Albumentations pipelines as a seeded random step (`dew.data.images.Albumentations`) |
| `metrics` | requests, which downloads FID's Inception weights |
| `wandb`, `mlflow`, `tensorboard` | Experiment trackers |
| `inference-clients` | The Ollama and OpenAI Python clients, including vLLM-compatible endpoints |
| `serve` | Starlette and uvicorn, for serving a decision model in Jev's wire format (`examples/serve_decisions.py`) |
| `environments` | Agent environments and remote sandboxes for rollouts: OpenEnv and the Modal, E2B and Daytona SDKs |
| `hpo` | Optuna, for `dew.config.sweep` |
| `eval-harness` | lm-evaluation-harness tasks through `dew.eval.harness.DewLM` |
| `profile` | The xprof profiler |
| `quantization` | Qwix, for quantized training and serving (`dew.quantization`) |
| `test` | The test suite's dependencies and pinned reference libraries |
| `ci` | The test suite's whole environment: `test` and every extra a test file imports |
| `kernel-compile` | JAX's CUDA plugin, CUDA's compiler and libtpu, to compile every kernel config on a host with no accelerator |
| `mutation` | cosmic-ray, for mutation testing (`tools/mutation.py`) |
| `notebooks` | nbclient, nbformat and an IPython kernel, to execute notebooks |
| `tutorials` | The tutorial notebooks' environment outside Colab: `streaming`, `interop`, `notebooks` and matplotlib |
| `live` | The live site container's bridge: its WebSocket release, the kernel gateway, an IPython kernel and ipywidgets |
<!-- END extras -->

```bash
uv pip install 'dewml[interop,streaming] @ git+https://github.com/AshishKumar4/dew' --extra-index-url https://pypi.dewml.dev/simple
```

The `vision` extra expects CPU builds of PyTorch and torchvision, installed first, so that no CUDA PyTorch sits beside JAX's accelerator runtime. Without it the native multimodal checkpoints still load, but their processors raise an error when given images.

```bash
uv pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install 'dewml[interop,vision] @ git+https://github.com/AshishKumar4/dew' --extra-index-url https://pypi.dewml.dev/simple
```

Run the first command even if CPU torch is already installed. Installing `vision`
from PyPI alone can add a CUDA-enabled torchvision beside it on Linux. Its
torch requirement accepts the CPU build, but its extension cannot load and importing
it fails with `operator torchvision::nms does not exist`. Resolving both from the
CPU index gives them the same build flavor; `--upgrade` also replaces an existing
torchvision from the wrong index. A package extra cannot select an index.

The `profile` extra installs XProf, never 2.23.2. XProf 2.23.2 declares `setuptools<70`, and PyTorch 2.13 and later declare `setuptools>=77.0.3`, so 2.23.2 can't be installed beside the `torch`, `vision`, `diffusers`, `torchax` or `test` extras. Don't upgrade XProf to 2.23.2 by hand in such an environment.

FlashAttention comes as Dew's own wheel, `dew-flash-attn-cu12` or `dew-flash-attn-cu13` (kernels/flash_attn), for the CUDA major of your jax: FlashAttention-2 for sm8x and sm120, and FlashAttention-3 for Hopper (sm90). The `cuda12` and `cuda13` extras install it on Linux x86_64, the one platform it is built for. With it installed, `"auto"` runs it on the GPUs whose measured kernel it is, for the calls `dew.nn.attention.flash_refusal` admits; a call it turns down on such a GPU is logged once, with why.

tokamax comes with Dew: Dew's Pallas kernels measure their configs through it (`dew.nn.kernels.tuning`). A compile takes the configs measured for its GPU model, else the nearest measured shape's or a heuristic, and measures nothing; `python tools/kernel_picks.py`, or `Trainer(tune_kernels=True)`, measures a model's on a GPU it has to itself. And `"auto"` runs its Pallas-Triton attention on sm80 and later for heads up to 64 wide (`dew.nn.attention.triton_runs`). Earlier releases pin `typeguard==2.13.3`, which tyro excludes, so installing one by hand breaks every recipe's command line.

## Development install

```bash
git clone https://github.com/AshishKumar4/dew.git
cd dew
uv venv --python 3.14
source .venv/bin/activate
sh tools/install_environment.sh cpu   # cuda12 or cuda13 on a GPU host
```

The `ci` extra is the test suite's environment, every package a test file imports. `tools/lock_environment.sh` locks it in `locks/`, and CI, the GPU nodes and the kernel lab's GPU jobs all install the lock for their host, with the same JAX build CI uses: `constraints.txt` names Dew's JAX build, which the multi-process cache tests need. PyTorch is used only by the reference tests and the image processors; Dew's model computation runs in JAX.

## Bumping JAX

CI runs on CPUs, so a JAX bump first runs these on a GPU, on Colab, in one session:

- `tests/test_linear_attention.py::test_the_decode_kernel_steps_as_the_recurrence_does` and `tests/test_decode_attention.py::test_the_decode_kernel_reads_each_row_to_its_length_at_every_config`: the gated delta net's and the decode attention's kernels are on the Pallas Triton backend JAX 0.11 deprecates, and each one's `fits` is its off switch;
- `python tools/gpu_profile_check.py --out out/profile-check`, about a minute: a warmed step captured under `dew.Profiler` with command buffers on and off. It fails unless the capture with them off attributes device time to HLO ops and every capture's device time a step is positive and within its wall time. It records whether the default capture attributes ops (none on 0.11.2) and whether XLA ran its command buffers op by op for the session, so a release that changes either shows. While the cause of the default capture's missing attribution is open, every Colab GPU session runs it first too.

## Compilation cache

A run, a recipe's or a library `Trainer.fit`'s, and a loaded task keep JAX's persistent compilation cache on by default, so a restart reads the programs the last run compiled instead of compiling them again. Dew stores compiled executables in `~/.cache/dew/xla/python3.X`, or `$XDG_CACHE_HOME/dew/xla/python3.X` when `XDG_CACHE_HOME` is set, and the GPU autotuner's picks beside them for each jaxlib. Hosts that share a home directory share it whatever their CPU: XLA compiles a CPU program for the host that compiles it, and Dew's JAX build keys the program by that host's CPU, so no host reads one that would not link there. A directory named in `JAX_COMPILATION_CACHE_DIR` is used as it is. `JAX_ENABLE_COMPILATION_CACHE=false` turns the cache off. JAX writes entries from process 0 alone, so a process pool caches only where every process reads the cache directory as one, which Dew checks as the pool forms: a pool on one host, or on hosts that share a home directory, caches by default, and a pool whose hosts each keep their own compiles every run unless it names a directory on storage they share (`compilation_cache_dir`). Caches apart would find a step on one process and compile it on another, which waits for ever for its peers. JAX keeps every entry (`jax_compilation_cache_max_size` is unbounded by default); set it, or clear the directory, to bound its size. Each Python minor version has its own directory because JAX 0.11 on Python 3.14 compresses entries with `compression.zstd` while its cache key still says `zlib`, and an older interpreter cannot read them. A run that passes its own `compilation_cache_dir` uses that path as given, so do not share one explicit directory between Python versions.

## Preparing TFDS data

Training reads prepared TFDS ArrayRecords and does not import TensorFlow. Preparing a dataset does need TensorFlow and sometimes dataset-specific packages (Oxford Flowers reads its label files with SciPy). TFDS 4.9.10 also imports `importlib_resources` while it prepares a dataset but only declares it for Python before 3.9, so the command installs it too.

Preparation is explicit: call TFDS.provision(), pass provision=True to load, or run dew data provision tfds://mnist. Those choices can download data. Indexed reads otherwise require prepared ArrayRecords and name the missing artifact. TensorFlow 2.21.0 has no Python 3.14 wheels, so prepare in a separate Python 3.13 environment when needed:

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

The script prints the prepared version directory. Run that export line in the training shell and bind it as the TFDS Source's path:

<!-- not run: needs the prepared TFDS directory the script above writes -->
```python
import os
from dew.data import TFDS, images, presets

flowers = TFDS("oxford_flowers102", split="all", path=os.environ["DEW_FLOWERS_PATH"])
train, val = flowers.holdout(records=16)
caption = images.caption("a photo of a {label}")
data = train.shuffle(seed=0).repeat().pipe(presets.images, 64, batch=4, augment=True, caption=caption)
held = val.pipe(presets.images, 64, batch=4, caption=caption)
```

The `images` preset takes its caption's class names from TFDS's class-label metadata. Missing shards are named errors. Non-local, non-GCS TFDS stores currently require an explicit local provision mirror; they are not silently materialized.

## Building the documentation

The site at [dewml.dev](https://dewml.dev) is built from `docs/`, the notebooks in `tutorials/` with their recorded outputs, and API pages generated from the docstrings in `src/dew`. It needs the Node release `engines` names in `site/package.json`, [pnpm](https://pnpm.io) and uv:

```bash
cd site
pnpm install
pnpm build
pnpm preview
```

`pnpm build` fails on a broken link, a notebook that was not executed top to bottom, a public module without an API page, or a model family the supported-models page does not list. `pnpm dev` serves the site with live reload.
