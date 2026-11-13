"""JAX's CUDA builds, and the newest one a machine's NVIDIA driver and GPUs run.

The standard library alone and no Dew import: tools/environments.py loads this
file by its path to choose a checkout's lock before any environment exists to
import Dew from, and `dew._devices` asks it at `import dew`.
"""

import os
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

TENSOR_MEMORY_SMS = (10, 11)
"""The SM majors whose GPUs hold tensor memory (PTX ISA, tcgen05: sm_100a and sm_103a, a B200 and B300, and
sm_110a): Dew's Blackwell kernels allocate it."""

LEAST_CUBLAS_BESIDE_TENSOR_MEMORY = (13, 2)
"""The least cuBLAS a GPU holding tensor memory runs Dew under: below it many cuBLAS kernels free their tensor
memory twice, so a kernel beside one, as a Mosaic launch overlaps the kernel before it (programmatic dependent
launch, its tensor memory allocated before its wait), can have its tensor memory taken or its allocation
stranded: a silent wrong result or a hang (JAX's CUDA plugin warns of it at start). cuBLAS's major is its
CUDA build's, so no cuda12 build runs on one."""

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
    kernel driver's major; on a GPU holding tensor memory, only a build whose
    cuBLAS can be 13.2 or newer (`LEAST_CUBLAS_BESIDE_TENSOR_MEMORY`). The least
    SM is the GPUs' oldest, but a machine's GPUs are one model."""
    least, least_sm, least_cuda = next((major, s, version) for name, major, s, version in CUDA_BUILDS
                                       if name == build)
    supported = int(driver.split(".")[0]) >= least if cuda is None else cuda >= least_cuda
    return supported and _builds_for(least_sm, least_cuda, sm)


def _builds_for(least_sm: tuple[int, int], least_cuda: int, sm: tuple[int, int]) -> bool:
    """Whether a CUDA build of least SM `least_sm` and CUDA version `least_cuda` builds for a GPU of `sm`:
    one its SM meets, and for a GPU holding tensor memory (`TENSOR_MEMORY_SMS`) one whose cuBLAS is 13.2
    or newer (`LEAST_CUBLAS_BESIDE_TENSOR_MEMORY`), its CUDA major's."""
    cublas = least_cuda // 1000 >= LEAST_CUBLAS_BESIDE_TENSOR_MEMORY[0] or sm[0] not in TENSOR_MEMORY_SMS
    return sm >= least_sm and cublas


def newest_build(gpus: tuple[str, tuple[int, int]] | None, cuda: int | None = None) -> str | None:
    """The newest of JAX's CUDA builds that runs on `gpus` (`listed_gpus`'s
    answer) and a user-mode driver of CUDA version `cuda` (`runs_on`), or
    None where there are none or none runs."""
    if gpus is None:
        return None
    return next((name for name, _, _, _ in CUDA_BUILDS if runs_on(name, *gpus, cuda)), None)


def generation_builds(generation: str) -> tuple[str, ...]:
    """JAX's CUDA builds a GPU of the CUDA `generation` (`sm90`, `sm100`) runs, newest first: each that builds
    for its SM (`_builds_for`), a GPU holding tensor memory the cuda13 build alone. A step compiled for such a
    GPU is compiled by one of them, as its kernels and spills are that toolchain's."""
    found = re.fullmatch(r"sm_?(\d+)(\d)a?", generation)
    if found is None:
        raise ValueError(f"{generation!r} names no CUDA generation; name one as sm80 or sm100")
    sm = (int(found.group(1)), int(found.group(2)))
    return tuple(name for name, _, least_sm, least_cuda in CUDA_BUILDS
                 if _builds_for(least_sm, least_cuda, sm))


def dew_cache_dir() -> str:
    """Dew's cache directory: `$XDG_CACHE_HOME/dew`, else ~/.cache/dew. Dew's caches (`dew._cache`) and
    tools/environments.py's environments, which cannot import Dew, keep under it."""
    return os.path.expanduser(
        os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join("~", ".cache"), "dew")
    )
