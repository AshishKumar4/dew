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
import subprocess
import sys
from pathlib import Path

import pytest

import dew

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
