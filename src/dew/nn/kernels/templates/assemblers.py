"""What assembles a launch's kernels (`report`'s verdict keys): a Mosaic GPU kernel's assembler as Mosaic GPU
names it (`mosaic`), a Pallas-Triton kernel's, XLA's, by its ptxas build and the nvJitLink it may load instead
(`triton`), each kind's (`named`). Another assembler's spills and serialized products are another verdict
(row 10782), so a key names each (row 7309)."""

import contextlib
import functools
import importlib
import importlib.metadata
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

_log = logging.getLogger(__name__)


def named(kinds: Iterable[str]) -> list[tuple[str, ...]] | None:
    """What assembles a launch's kernels of `kinds`, each kind's as its assembler is named: a Mosaic GPU
    kernel's (`mosaic`) and a Pallas-Triton kernel's, XLA's (`triton`), which differ where JAX gives XLA a
    CUDA directory Mosaic GPU does not search. A relock or a system CUDA that moves
    either certifies again, where the seal, which holds the CUDA plugins alone, kept a verdict another
    assembler judged (row 7309). None where one is unknown: its verdict is not stored."""
    found = []
    for kind in sorted(set(kinds)):
        if (assembler := (mosaic if kind == "mosaic_gpu" else triton)()) is None:
            return None
        found.append(assembler)
    return found


@functools.cache
def mosaic() -> tuple[str, ...] | None:
    """The assembler Mosaic GPU assembles a kernel's PTX with, as it names it (`_get_assembler`, Dew's JAX
    build from post124): its provider's kind and, where that runs a ptxas, the ptxas's build
    (`ptxas_build`), its path aside, which differs host to host where the build does not. Mosaic GPU finds
    its ptxas in a CUDA directory of its own (`./cuda_sdk_lib`, `CUDA_HOME`, a system CUDA's, the nvcc
    wheel's). Read once a process: neither moves while it runs."""
    # JAX's own module of the extension, which loads it from whichever build provides it: its native
    # attributes are read off the module, which no stub states.
    extension = importlib.import_module("jax._src.lib.mosaic_gpu")._mosaic_gpu_ext
    provider = extension._get_assembler()
    kind = provider.split("(", 1)[0]
    if "ptxas" not in provider:
        return (kind,)
    # The ptxas's path runs to the nvlink the provider names after it, or to the name's end.
    if (ptxas := re.fullmatch(r"[^(]*\(ptxas: (.*?)(?:, nvlink: .*)?\)", provider)) is None:
        _log.warning("certification stores no verdict: no ptxas path read in %s", provider)
        return None
    build = ptxas_build(ptxas.group(1))
    return None if build is None else (kind, build)


@functools.cache
def triton() -> tuple[str, ...] | None:
    """The assembler XLA assembles a Pallas-Triton kernel's PTX with: the ptxas in the CUDA directory JAX
    gives each compile (`jax._src.lib.cuda_path`: `CUDA_ROOT`'s, else the nvcc wheel's), by its build
    (`ptxas_build`), beside the nvJitLink XLA's automatic mode assembles with instead where one loads: the
    wheels' by version, and each library the loader could open by soname (`nvjitlink_libraries`). None
    where JAX names no directory: XLA's own search finds its ptxas, which no key states. Read once a
    process."""
    from jax._src.lib import cuda_path

    if cuda_path is None:
        _log.info("certification stores no Pallas-Triton kernel's verdict: JAX names XLA no CUDA directory")
        return None
    if (build := ptxas_build(os.path.join(cuda_path, "bin", "ptxas"))) is None:
        return None
    named = (((distribution.metadata.get("Name") or "").lower(), distribution)
             for distribution in importlib.metadata.distributions())
    return (build, *sorted(f"{name}=={distribution.version}" for name, distribution in named
                           if re.fullmatch(r"nvidia-nvjitlink(-cu\d+)?", name)), *nvjitlink_libraries())


def nvjitlink_libraries() -> list[str]:
    """Each libnvJitLink the process's dynamic loader could open by soname, as XLA's nvJitLink stub opens
    one: those in `LD_LIBRARY_PATH`'s directories and the linker cache's (`ldconfig -p`), each by its path,
    size and modification, so a system library updated in place where no wheel provides one keys its
    launches afresh (row 12166). A cache that does not list stands for none."""
    paths = [str(path) for directory in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if directory
             for path in sorted(Path(directory).glob("libnvJitLink.so*"))]
    ldconfig = shutil.which("ldconfig") or "/sbin/ldconfig"
    try:
        listed = subprocess.run([ldconfig, "-p"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        listed = ""
    paths += re.findall(r"^\s*libnvJitLink\.so\S*\s.*=>\s*(\S+)$", listed, flags=re.M)
    named = []
    for path in dict.fromkeys(os.path.realpath(path) for path in paths):
        with contextlib.suppress(OSError):
            held = os.stat(path)
            named.append(f"{path}:{held.st_size}:{held.st_mtime_ns}")
    return named


def ptxas_build(ptxas: str) -> str | None:
    """The build `ptxas` states of itself (`--version`'s last line), or None, why logged, where it states
    none: a launch it assembles stores no verdict, its certification run as before."""
    try:
        ran = subprocess.run([ptxas, "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        _log.warning("certification stores no verdict: %s --version did not run: %s", ptxas, error)
        return None
    if ran.returncode or not (lines := ran.stdout.strip().splitlines()):
        _log.warning("certification stores no verdict: %s --version exited %d stating no build: %s", ptxas,
                     ran.returncode, ran.stderr.strip())
        return None
    return lines[-1]
