"""The CUDA toolkit JAX compiles with on a GPU host is the one Dew's lock installs: JAX's CUDA root
(`jax._src.lib.cuda_path`, which XLA reads its libdevice from and Mosaic GPU its ptxas) is a directory that
holds `bin/ptxas`, never the string 'None' a package of CUDA 13's other tools alone made it (row 4162),
whatever the test environment's wheels lay out under nvidia/."""

import pathlib

import pytest
from jax._src import lib

pytestmark = pytest.mark.gpu(reason="the CUDA root of a GPU host's JAX, which a CPU host need not have")


def test_jaxs_cuda_root_is_a_toolkit_that_holds_ptxas():
    root = lib.cuda_path
    assert root is not None and (pathlib.Path(root) / "bin" / "ptxas").is_file(), root
