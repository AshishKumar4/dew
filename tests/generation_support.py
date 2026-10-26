"""Which assumed GPU generations a test may run its kernels under in this process.

`generation.assumed(name)` routes a call as a `name` GPU would. Lowering under it runs anywhere; running
the routed kernels needs a process that can: a CPU interprets any generation's kernels, and a GPU runs its
own generation's and an older one's (Triton's sm80 kernels on an H100), not a newer one's (Mosaic GPU's
Hopper kernels on an A100, which refuse its pre-Hopper pipelines). The Mosaic GPU interpreter itself
runs in a CPU process only: on a GPU host it reads the device and refuses as the compiler does; and
Mosaic GPU's lowering checks a kernel against the process's own device, so lowering for a newer GPU runs
on a CPU or on that GPU.
"""

import importlib.util

import jax
import pytest

from dew.nn.kernels import generation

INTERPRETS = pytest.mark.skipif(jax.default_backend() != "cpu",
                                reason="interprets the Mosaic GPU kernels, which a CPU process alone does")
SUMS_IN_IEEE = pytest.mark.skipif(
    jax.default_backend() != "cpu",
    reason="compares sums bit for bit as a CPU's IEEE fp32 dots give them, which a GPU's tensor cores, "
           "truncating each add, do not")


def runnable(name: str) -> bool:
    """Whether this process runs the kernels `generation.assumed(name)` routes to: a CPU any (it
    interprets them); a GPU Hopper's warpgroup kernels only if it is that generation (capability 9.0
    alone has the instructions), and any other generation's only if no newer than itself."""
    here = generation.device_generation()
    if not here.startswith("sm") or generation.capability(name) is None:
        return here == "cpu"  # a TPU's launches a CPU alone interprets
    if generation.gpu_target(name).wgmma:
        return name == here
    return generation.capability(name) <= generation.capability(here)


def runs(name: str):
    """A mark skipping a test that runs kernels under `generation.assumed(name)` where this process
    cannot (`runnable`)."""
    return pytest.mark.skipif(not runnable(name),
                              reason=f"runs {name}'s kernels, which this process's device does not")


def lowers(name: str):
    """A mark skipping a test that lowers `name`'s Mosaic GPU kernels where this process cannot run
    them (`runnable`): Mosaic GPU checks a kernel against the process's own device (an A100's 166912
    bytes of shared memory refuse an H100's tile)."""
    return runs(name)


COMPILES_FOR_TPU = pytest.mark.skipif(
    not (jax.default_backend() == "tpu"
         or (jax.default_backend() == "cpu" and importlib.util.find_spec("libtpu") is not None)),
    reason="compiles for a compile-only TPU, which libtpu does in a CPU process with it installed, or a TPU")
"""A mark skipping a test that compiles kernels for a TPU (`kernel_compile.compile_tpu_kernels`) where this
process cannot: a GPU host's process has no TPU toolchain."""


def generations(*names: str) -> list:
    """`names` as parameters, each marked `runs`, for a test parametrized over assumed generations."""
    return [pytest.param(name, marks=runs(name)) for name in names]
