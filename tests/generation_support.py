"""Which assumed GPU generations a test may run its kernels under in this process.

`generation.assumed(name)` routes a call as a `name` GPU would. Lowering under it runs anywhere; running
the routed kernels needs a process that can: a CPU interprets any generation's kernels, and a GPU runs its
own generation's and an older one's (Triton's sm80 kernels on an H100), not a newer one's (Mosaic GPU's
Hopper kernels on an A100, which refuse its pre-Hopper pipelines). The Mosaic GPU interpreter itself
runs in a CPU process only: on a GPU host it reads the device and refuses as the compiler does; and
Mosaic GPU's lowering checks a kernel against the process's own device, so lowering for a newer GPU runs
on a CPU or on that GPU.
"""

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
    """Whether this process runs the kernels `generation.assumed(name)` routes to."""
    here = generation.device_generation()
    if not here.startswith("sm"):
        return here == "cpu"
    return int(name.removeprefix("sm")) <= int(here.removeprefix("sm"))


def runs(name: str):
    """A mark skipping a test that runs kernels under `generation.assumed(name)` where this process
    cannot (`runnable`)."""
    return pytest.mark.skipif(not runnable(name),
                              reason=f"runs {name}'s kernels, which this process's device does not")


def lowers(name: str):
    """A mark skipping a test that lowers `name`'s Mosaic GPU kernels where this process's GPU is older:
    Mosaic GPU checks a kernel's shared memory against the process's own device (an A100's 166912 bytes
    refuse an H100's tile)."""
    return pytest.mark.skipif(not runnable(name),
                              reason=f"lowers {name}'s Mosaic GPU kernels, which it checks against this "
                                     "process's device")


def generations(*names: str) -> list:
    """`names` as parameters, each marked `runs`, for a test parametrized over assumed generations."""
    return [pytest.param(name, marks=runs(name)) for name in names]
