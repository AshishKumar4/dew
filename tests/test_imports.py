"""Every module of dew imports as the first thing a fresh interpreter does.

An import cycle only shows when a module is the first of its cycle to load:
`import dew.nn.multimodal` in a fresh process met `dew.nn.backbones`,
whose diffusion transformers imported `dew.diffusion`, whose masked-token
process imported `dew.nn.multimodal` again, half initialized. Imports from
inside the suite never hit it, since `dew` is already loaded.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

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
from dew.sampling import Euler, TextToImage

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


def test_importing_the_data_layer_reads_no_configuration_and_its_workers_run(tmp_path):
    """Importing dew.data reads no storage configuration, so a broken one fails the first store
    that resolves, not the import, and a data pass with worker processes runs in a program that
    never parsed absl's flags (`import dew` marks them parsed, as absl.app would)."""
    broken = tmp_path / "storage.toml"
    broken.write_text("this is not toml [")
    script = r"""
import numpy as np
import dew.data
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
print(json.dumps({"refused": refused, "ids": sorted(int(v) for batch in reader for v in batch["id"])}))
reader.close()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600,
                            env={**os.environ, "PYTHONPATH": str(SOURCE), "JAX_PLATFORMS": "cpu",
                                 "DEW_STORAGE_CONFIG": str(broken)})
    assert result.returncode == 0, result.stderr[-2000:]
    reported = json.loads(result.stdout.splitlines()[-1])
    assert reported == {"refused": "TOMLDecodeError", "ids": list(range(8))}
