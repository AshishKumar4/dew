"""JAX's CUDA builds, and the newest one a machine's NVIDIA driver and GPUs run.

The standard library alone and no Dew import: tools/environments.py loads this
file by its path to choose a checkout's lock before any environment exists to
import Dew from, and `dew.devices` asks it at `import dew`.
"""

import re

CUDA_BUILDS = (("cuda13", 580, (7, 5), 13000), ("cuda12", 525, (5, 2), 12000))
"""JAX's CUDA builds, newest first, each with the least NVIDIA driver major,
GPU SM and CUDA version (as `cuDriverGetVersion` reports it, 13.0 as 13000)
it runs on: JAX's installation guide, and NVIDIA's CUDA release notes (Table
3, minor-version compatibility). The CUDA version is the user-mode driver's,
which NVIDIA's forward-compatibility package (cuda-compat) raises past what
the kernel driver's major gives. docs/installation.md, "Which CUDA build",
states the rule; site/scripts/gen_install.py generates the standalone install
script's minima from this table."""

PACKAGE_INDEX = "https://pypi.dewml.dev/simple"
"""Dew's package index, the one place Dew's JAX build (pyproject.toml's jax
requirement) is served; every install names it as an extra index beside
PyPI. `dew tpu setup` and tools/smoke_wheel.py read it here, and
site/scripts/gen_install.py writes it into constraints.txt, the standalone
install script and the Colab notebooks' install."""

QUERY = ("nvidia-smi", "--query-gpu=driver_version,compute_cap", "--format=csv,noheader")
"""The question `listed_gpus` reads the answer of."""


def listed_gpus(answer: str | None) -> tuple[str, tuple[int, int]] | None:
    """The driver version and the least SM of the GPUs in nvidia-smi's
    `answer` to `QUERY`, or None where it gave none: no answer, no GPU, or a
    driver too old to report an SM, which prints `[N/A]`."""
    lines = [line for line in (answer or "").splitlines() if line.strip()]
    rows = [[field.strip() for field in line.split(",")] for line in lines]
    if not rows or not all(len(row) == 2 and re.fullmatch(r"\d+\.\d+", row[1]) for row in rows):
        return None
    sms = [(int(major), int(minor)) for major, minor in (row[1].split(".") for row in rows)]
    return rows[0][0], min(sms)


def runs_on(build: str, driver: str, sm: tuple[int, int], cuda: int | None = None) -> bool:
    """Whether JAX's `build` runs on a driver of version `driver` (`580.82.07`)
    and GPUs whose least SM is `sm` (`CUDA_BUILDS`): by the CUDA version the
    user-mode driver reports (`cuda`), where the caller can ask it, else by the
    kernel driver's major."""
    least, least_sm, least_cuda = next((major, s, version) for name, major, s, version in CUDA_BUILDS
                                       if name == build)
    supported = int(driver.split(".")[0]) >= least if cuda is None else cuda >= least_cuda
    return supported and sm >= least_sm


def newest_build(gpus: tuple[str, tuple[int, int]] | None, cuda: int | None = None) -> str | None:
    """The newest of JAX's CUDA builds that runs on `gpus` (`listed_gpus`'s
    answer) and a user-mode driver of CUDA version `cuda` (`runs_on`), or
    None where there are none or none runs."""
    if gpus is None:
        return None
    return next((name for name, _, _, _ in CUDA_BUILDS if runs_on(name, *gpus, cuda)), None)
