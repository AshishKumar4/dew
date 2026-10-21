"""The same program from two checkouts of Dew shares one compiled executable: a Pallas kernel's Triton
module carries its source locations, which the pinned JAX's cache key strips (constraints.txt)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PROGRAM = """
import json, sys
import jax, jax.numpy as jnp
from dew.cache import enable_compilation_cache
from dew.nn.attention import rms_normalized
from dew.telemetry.instrumentation import compilation_cache_counts, count_compilation_cache
enable_compilation_cache(sys.argv[1])
count_compilation_cache()
x, scale = jnp.ones((4, 128, 1024), jnp.bfloat16), jnp.ones((1024,), jnp.float32)
jax.jit(lambda x, scale: rms_normalized(x, scale, 1e-6, jnp.bfloat16, False, False, True,
                                        implementation="pallas")).lower(x, scale).compile()
print(json.dumps(compilation_cache_counts()))
"""


@pytest.mark.gpu("Pallas lowers its kernels to Triton on a CUDA GPU alone")
def test_two_checkouts_share_one_compiled_kernel(tmp_path):
    """A second checkout (here the same tree under another path) finds the first's executable."""
    other = tmp_path / "checkout" / "src"
    other.mkdir(parents=True)
    (other / "dew").symlink_to(ROOT / "src" / "dew", target_is_directory=True)
    counts = []
    for source in (ROOT / "src", other):
        path = os.pathsep.join(filter(None, (str(source), os.environ.get("PYTHONPATH"))))
        done = subprocess.run([sys.executable, "-c", PROGRAM, str(tmp_path / "cache")], capture_output=True,
                              text=True, timeout=600, env={**os.environ, "PYTHONPATH": path})
        assert done.returncode == 0, done.stderr[-3000:]
        counts.append(json.loads(done.stdout.splitlines()[-1]))
    assert counts[0]["train/cache_misses"] >= 1
    assert counts[1]["train/cache_misses"] == 0 and counts[1]["train/cache_hits"] >= 1
