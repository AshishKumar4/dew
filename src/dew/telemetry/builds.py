"""JAX's CUDA builds, and the newest one a machine's NVIDIA driver and GPUs run.

The standard library alone and no Dew import: tools/environments.py loads this
file by its path to choose a checkout's lock before any environment exists to
import Dew from, and `dew.telemetry.devices` asks it at `import dew`.
"""

import re

CUDA_BUILDS = (("cuda13", 580, (7, 5)), ("cuda12", 525, (5, 2)))
"""JAX's CUDA builds, newest first, each with the least NVIDIA driver major
and GPU SM it runs on: JAX's installation guide, and NVIDIA's CUDA release
notes (Table 3, minor-version compatibility). docs/installation.md, "Which
CUDA build", states the rule; site/scripts/gen_install.py generates the
standalone install script's minima from this table."""

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


def runs_on(build: str, driver: str, sm: tuple[int, int]) -> bool:
    """Whether JAX's `build` runs on a driver of version `driver` (`580.82.07`)
    and GPUs whose least SM is `sm` (`CUDA_BUILDS`)."""
    least, least_sm = next((major, s) for name, major, s in CUDA_BUILDS if name == build)
    return int(driver.split(".")[0]) >= least and sm >= least_sm


def newest_build(gpus: tuple[str, tuple[int, int]] | None) -> str | None:
    """The newest of JAX's CUDA builds that runs on `gpus` (`listed_gpus`'s
    answer), or None where there are none or none runs."""
    if gpus is None:
        return None
    return next((name for name, _, _ in CUDA_BUILDS if runs_on(name, *gpus)), None)
