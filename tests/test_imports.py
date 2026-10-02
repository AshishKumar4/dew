"""Every module of dew imports as the first thing a fresh interpreter does.

An import cycle only shows when a module is the first of its cycle to load:
`import dew.nn.multimodal` in a fresh process met `dew.nn.backbones`,
whose diffusion transformers imported `dew.diffusion`, whose masked-token
process imported `dew.nn.multimodal` again, half initialized. Imports from
inside the suite never hit it, since `dew` is already loaded.
"""
import os
import pkgutil
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
from dew.interop.hub import pull_from_hub
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
                   {}, steps=1, guidance=None, sampler=Euler())
result = pipe('', seed=0).host()
assert result.images.shape == (1, 2, 2, 1)
np.testing.assert_array_equal(result.images, np.zeros((1, 2, 2, 1), np.float32))
'''
    run = subprocess.run([sys.executable, "-c", script, str(tmp_path / "weights.safetensors")],
                         capture_output=True, text=True,
                         env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SOURCE)}, timeout=60)
    assert run.returncode == 0, run.stderr


def test_every_module_imports_first_in_a_fresh_interpreter():
    names = sorted(module.name for module in pkgutil.walk_packages(dew.__path__, "dew."))
    script = """
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

def imported(name):
    run = subprocess.run([sys.executable, "-c", f"import {name}"], capture_output=True, text=True)
    return f"{name}: {run.stderr.strip().splitlines()[-1]}" if run.returncode else None

with ThreadPoolExecutor(os.cpu_count() or 4) as pool:
    print("\\n".join(line for line in pool.map(imported, sys.argv[1:]) if line))
"""
    run = subprocess.run([sys.executable, "-c", script, *names], capture_output=True, text=True,
                         env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SOURCE)}, timeout=1800)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip() == "", run.stdout
