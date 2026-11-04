"""Every module of dew imports as the first thing a fresh interpreter does.

An import cycle only shows when a module is the first of its cycle to load:
`import dew.nn.multimodal` in a fresh process met `dew.nn.backbones`,
whose diffusion transformers imported `dew.diffusion`, whose masked-token
process imported `dew.nn.multimodal` again, half initialized. Imports from
inside the suite never hit it, since `dew` is already loaded.
"""
import ast
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import dew
from tools import gen_exports

SOURCE = Path(dew.__file__).parent.parent


def test_unconditional_sampling_and_weight_io_need_no_optional_backends(tmp_path):
    """Image sampling and safetensors round-trip run with optional imports refused."""
    script = r'''
import importlib.abc
import sys

class Unavailable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'transformers', 'torch', 'tokenizers', 'cv2', 'grain', 'orbax'}:
            raise ImportError(f'optional backend {fullname} is unavailable')

sys.meta_path.insert(0, Unavailable())
import dew
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from dew.diffusion import DirectPredictionTransform, FlowMatchingScheduler, Process
from dew.inputs import Field, InputSpec
from dew.interop import load_params, save_params
from dew.inference import TextToImage
from dew.sampling import Euler

class Zero(nn.Module):
    def __call__(self, x, time, train=False):
        return jnp.zeros_like(x)

weights = {'params': {'weight': np.arange(8, dtype=np.float32)}}
save_params(weights, sys.argv[1])
restored = load_params(sys.argv[1])
np.testing.assert_array_equal(restored['params']['weight'], weights['params']['weight'])
pipe = TextToImage(Zero(), Process(FlowMatchingScheduler(), DirectPredictionTransform()),
                   InputSpec(Field('image', (2, 2, 1))),
                   {}, steps=1, guidance=None, solver=Euler())
result = pipe('', key=0).host()
assert result.images.shape == (1, 2, 2, 1)
np.testing.assert_array_equal(result.images, np.zeros((1, 2, 2, 1), np.float32))
'''
    run = subprocess.run([sys.executable, "-c", script, str(tmp_path / "weights.safetensors")],
                         capture_output=True, text=True,
                         env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SOURCE)}, timeout=60)
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize("named", ["DEW_STORAGE_CONFIG", "DEW_CONFIG_DIR"])
def test_importing_the_data_layer_reads_no_configuration_and_its_workers_run(tmp_path, named):
    """Importing dew.data reads no storage configuration, so a broken one, named by its own variable
    or found in Dew's configuration directory as every configuration of Dew's is, fails the first
    store that resolves, not the import, yet registers Dew's bounded cache, through which fsspec reads
    a file before any store resolves; and a data pass with worker processes runs in a program that
    never parsed absl's flags (`import dew` marks them parsed, as absl.app would)."""
    broken = tmp_path / "storage.toml"
    broken.write_text("this is not toml [")
    script = r"""
import numpy as np
import dew.data
import fsspec
import tempfile
fsspec.filesystem("memory").pipe("/dewcache-probe/value", b"cached bytes")
with fsspec.open("dewcache::memory://dewcache-probe/value", "rb",
                 dewcache={"cache_storage": tempfile.mkdtemp()}) as file:
    cached = file.read().decode()
from dew.data.pipeline import Pipeline
from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loading
from dew.data.sources import Memory
from dew.data.storage import Store
import json
try:
    Store("memory://x").resolve()
except Exception as error:
    refused = type(error).__name__
data = compile(Pipeline(Memory([{"id": np.int32(i)} for i in range(8)])).repeat(1).batch(2),
               read="memory", loading=Loading(workers=2, threads=1, read_buffer=0, worker_buffer=1))
reader = data.train(DataPartition())
print(json.dumps({"refused": refused, "cached": cached,
                  "ids": sorted(int(v) for batch in reader for v in batch["id"])}))
reader.close()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600,
                            env={**os.environ, "PYTHONPATH": str(SOURCE), "JAX_PLATFORMS": "cpu",
                                 named: str(broken if named == "DEW_STORAGE_CONFIG" else tmp_path)})
    assert result.returncode == 0, result.stderr[-2000:]
    reported = json.loads(result.stdout.splitlines()[-1])
    assert reported == {"refused": "TOMLDecodeError", "cached": "cached bytes", "ids": list(range(8))}


def test_a_models_quantization_imports_no_training():
    """The model's quantization, which loading, adapters and serving read, imports nothing of
    training: a checkpoint's input quantizer is the model's, and reading it starts no trainer,
    runtime or checkpointer."""
    script = r'''
import json
import sys
from dew.quantization import InputQuantization, Quantization
training = [name for name in sys.modules if name == "dew.training" or name.startswith("dew.training.")]
print(json.dumps(sorted(training)))
'''
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SOURCE)}, timeout=120)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout.splitlines()[-1]) == []


@pytest.mark.parametrize("module", ["dew.config", "dew.training.budget"])
def test_a_config_or_a_package_imports_no_trainer_until_a_run_trains(module):
    """A run's configuration reads its dataclasses' types and none of the trainer, which a run reads
    where it trains (`RunConfig.train`), and a module of the training layer reads that module, its
    package's names imported as they are used: each imports in a fresh interpreter without
    `dew.training.trainer`."""
    script = f"""
import json
import sys
import {module}
print(json.dumps("dew.training.trainer" in sys.modules))
"""
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SOURCE)}, timeout=120)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout.splitlines()[-1]) is False


IMAGE_GENERATION = ("dew.data.images", "dew.data.audio", "dew.data.video", "dew.diffusion.presets",
                    "dew.nn.vision", "dew.nn.autoencoders", "dew.objectives.diffusion",
                    "dew.inference.images",
                    *(f"dew.nn.backbones.{name}" for name in (
                        "dit", "edm2", "flux", "flux2", "mmdit", "qwen_image", "sd3", "ssm_dit", "unet",
                        "unet3d", "unet_condition", "uvit", "video_dit", "wan", "z_image")))
"""Image generation's and media's modules: shared diffusion math (`dew.diffusion.process`, its schedules)
and text encoders are any domain's."""

FOREIGN = {
    "dew.config": (*IMAGE_GENERATION, "dew.objectives.lm", "dew.decision",
                   "dew.nn.backbones.causal_transformer"),
    "dew.config.lm": IMAGE_GENERATION,
    "dew.config.diffusion": ("dew.nn.backbones.causal_transformer", "dew.nn.mla", "dew.objectives.lm",
                             "dew.data.chat"),
    "dew.decision.config": tuple(name for name in IMAGE_GENERATION if name != "dew.nn.vision"),
}
"""Each run configuration's module and the modules outside its domain it must not import: an LM's no image
diffusion, vision tower, autoencoder or media data; a diffusion model's no decoder language model or chat
data, its text encoders' decoder blocks its own; a decision task's no image generation or autoencoder,
the vision towers its multimodal reader reads its own (row 2042); and the shared modules an objective, a
sampler or a model imports (row 2043): a LoRA reads a stack's layer names, not its plan's mixers; a
sampler's tokens and rounding stream and a saved run's record are the categorical's, the precision's and
the records' own, not an objective's; and a source's causality walk reads the programs an equation runs
as `dew.nn.programs` reads them, not as training's compiled program does (row 3959)."""
FOREIGN |= {"dew.lora": ("dew.nn.backbones.layer_plan", "dew.nn.mixer_base", "dew.objectives"),
            "dew.sampling.text": ("dew.objectives",),
            "dew.sampling.strategies": ("dew.objectives",),
            "dew.inference.tasks": ("dew.objectives",),
            "dew.interop.sequence_causality": ("dew.training",)}


@pytest.mark.parametrize("module", sorted(FOREIGN))
def test_a_module_imports_its_own_domain_alone(module):
    """A run configuration's or a shared module's module imports, in a fresh interpreter, none of the
    modules outside its domain (`FOREIGN`): a change there re-runs none of its tests."""
    script = f"""
import json, sys
import {module}
print(json.dumps(sorted(sys.modules)))
"""
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SOURCE)}, timeout=120)
    assert run.returncode == 0, run.stderr
    loaded = json.loads(run.stdout.splitlines()[-1])
    foreign = FOREIGN[module]
    assert [name for name in loaded
            if name in foreign or name.startswith(tuple(f"{prefix}." for prefix in foreign))] == []


def test_only_the_rl_packages_import_dew_rl_and_it_imports_no_training():
    """Only `dew.rl` and `dew.objectives.rl` import `dew.rl`, and `dew.rl` imports nothing of
    `dew.training` (docs/design/architecture.md). The test reads every module's imports, absolute
    and relative, from its source."""
    package = SOURCE / "dew"

    def imported(path: Path) -> set[str]:
        module = ".".join(path.relative_to(SOURCE).with_suffix("").parts)
        # A relative import resolves from the module's package; an __init__'s module ends in __init__.
        parts = module.split(".")[:-1]
        names = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = ".".join([*parts[: len(parts) - node.level + 1], *([base] if base else [])])
                names.add(base)
                names.update(f"{base}.{alias.name}" for alias in node.names)
        return names

    def within(name: str, prefix: str) -> bool:
        return name == prefix or name.startswith(prefix + ".")

    importers, trained = [], []
    for path in sorted(package.rglob("*.py")):
        module = ".".join(path.relative_to(SOURCE).with_suffix("").parts).removesuffix(".__init__")
        names = imported(path)
        rl = within(module, "dew.rl") or within(module, "dew.objectives.rl")
        if not rl and any(within(name, "dew.rl") for name in names):
            importers.append(module)
        if within(module, "dew.rl") and any(within(name, "dew.training") for name in names):
            trained.append(module)
    assert importers == [] and trained == []


@pytest.mark.parametrize(("package", "read"), [("dew.sampling", "text.Sampling"),
                                               ("dew.nn.backbones", "dit.SimpleDiT"),
                                               ("dew.data", "images.DecodeImage"),
                                               ("dew.data", "presets.sft"),
                                               ("dew.diffusion", "presets.EDM"),
                                               ("dew.diffusion", "discrete.DiscreteProcess")])
def test_a_lazy_packages_modules_read_through_it_as_an_eager_packages_did(package, read):
    """A lazily exporting package (`dew._exports`) reads each of its modules that defines an export as
    its attribute, in a fresh interpreter that imported the package alone, as its eager imports bound
    them: `dew.sampling.text.Sampling` after `import dew.sampling`."""
    module, name = read.split(".")
    script = (f"import importlib, {package}\n"
              f"assert getattr({package}.{module}, {name!r}) is getattr(importlib.import_module("
              f"'{package}.{module}'), {name!r})\n")
    subprocess.run([sys.executable, "-c", script], check=True,
                   env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SOURCE)})


@pytest.mark.parametrize(("package", "modules"), [("dew.data", ("audio", "chat", "images", "preferences",
                                                                "presets", "prompts", "text", "video")),
                                                   ("dew.diffusion", ("discrete", "presets"))])
def test_a_lazy_packages_star_import_binds_its_modules_as_its_eager_one_did(package, modules):
    """`from package import *` binds each module the package exported when it imported them eagerly, each
    the module itself, in a fresh interpreter."""
    script = (f"import importlib\nfrom {package} import *\n"
              f"assert all(globals()[name] is importlib.import_module('{package}.' + name)\n"
              f"           for name in {modules!r})\n")
    subprocess.run([sys.executable, "-c", script], check=True,
                   env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SOURCE)})


@pytest.mark.parametrize("package", [".".join(path.parent.relative_to(SOURCE).parts)
                                     for path in gen_exports.packages(SOURCE / "dew")])
def test_a_lazy_packages_listing_names_what_it_exports(package):
    """Each lazily exporting package, every one that declares its exports (`gen_exports.packages`), lists
    every name it exports to `dir()`, which an editor and an interpreter complete from, though it has
    imported none of them (rows 3206, 3217)."""
    script = (f"import importlib\nmodule = importlib.import_module({package!r})\n"
              "missing = sorted(set(module.__all__) - set(dir(module)))\nassert not missing, missing\n")
    subprocess.run([sys.executable, "-c", script], check=True,
                   env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SOURCE)})


def test_the_release_version_is_read_from_a_module_import_dew_does_not_run():
    """`dew.__version__` comes from `dew._version`, imported when the version is first read, and the package
    metadata takes it from there: a release's bump then changes no module `import dew` executes, whose top
    level every test file's CI key covers (train 396: the 0.1.0 to 0.2.0 line in src/dew/__init__.py re-keyed
    every row of its suite, 0.0% reused, where nothing a test reads had changed)."""
    script = ("import sys\nimport dew\n"
              "assert 'dew._version' not in sys.modules, 'import dew ran the version module'\n"
              "version = dew.__version__\n"
              "assert sys.modules['dew._version'].__version__ == version\n"
              "print(version)\n")
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SOURCE)}, timeout=120)
    assert run.returncode == 0, run.stderr
    metadata = tomllib.loads((SOURCE.parent / "pyproject.toml").read_text())
    assert metadata["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "dew._version.__version__"}
    assert re.fullmatch(r"\d+\.\d+\.\d+", run.stdout.strip())
