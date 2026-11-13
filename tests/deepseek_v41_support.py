"""The DeepSeek-V4.1 tiny the release's tests share (tests/test_deepseek_v41.py, its recorded sites in
tests/test_deepseek_v41_recorded.py): where it lives, its load, and the fp32 products the release
multiplies in."""

from pathlib import Path

import jax
import numpy as np

from dew.interop import Pretrained
from tools.deepseek_v41_numerics import unquantized

ROOT = Path(__file__).parent / "fixtures" / "hf"
TINY = ROOT / "deepseek-v41-tiny"


def fp32_matmuls():
    """The reference multiplies in fp32, where a GPU's default is TF32: a module's autouse fixture."""
    with jax.default_matmul_precision("highest"):
        yield


def source():
    """The tiny loaded at fp32 with the reference attention, its model without quantization, and the
    release's one-seed run (reference.npz): a module's fixture."""
    loaded = Pretrained.load(TINY, dtype="float32", attention_impl="reference")
    return loaded, unquantized(loaded.model), np.load(TINY / "reference.npz")
