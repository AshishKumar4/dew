"""A bf16 window's GPU tier (the owner's rule: no unit tests on GPUs, so its GPU case a file of its own,
apart from tests/test_bf16_window.py's, which a GPU tier part's compile-only prepare does not run): a window
decoded for an export in a process whose JAX runs no CPU backend (audit 2026-11-27 P03)."""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu(reason="a process whose JAX runs no CPU backend, its accelerator alone")

WINDOW_WITHOUT_CPU = """
import jax, jax.numpy as jnp, numpy as np
from dew import quantization
kernel = (np.random.default_rng(13987).normal(size=(256, 256)) * 0.02).astype(jnp.bfloat16)
held = quantization.bf16_windows({"params": {"kernel": jnp.asarray(kernel)}}, sites=[""])
decoded = np.asarray(quantization.decoded_packed(held)["params"]["kernel"])
print(jax.default_backend(), bool(np.array_equal(decoded.view(np.uint16), kernel.view(np.uint16))))
"""


@pytest.mark.compiles_at_run(reason="a fresh process, whose JAX runs no CPU backend, compiles where it runs")
def test_a_window_decodes_for_an_export_where_jax_runs_no_cpu():
    """A window decoded for an export (`quantization.decoded_packed`, `Pretrained.save`'s) in a process whose
    JAX runs no CPU backend (`JAX_PLATFORMS=cuda`) decodes on its device, bit for bit, rather than ask for a
    CPU it does not have (audit 2026-11-27 P03)."""
    ran = subprocess.run([sys.executable, "-c", WINDOW_WITHOUT_CPU], capture_output=True, text=True,
                         env={**os.environ, "JAX_PLATFORMS": "cuda"}, timeout=600)
    assert ran.returncode == 0, ran.stderr[-2000:]
    assert ran.stdout.split()[-2:] == ["gpu", "True"], ran.stdout
