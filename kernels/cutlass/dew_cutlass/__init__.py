"""CUTLASS's grouped GEMMs (NVIDIA, BSD-3) as XLA FFI targets of a library with no Python ABI: `targets()`
names each target and its handler, which Dew registers with jax's FFI at its first call; `KERNELS` states the
kernels the installed build instantiates, as their compiled types state them, and `ARCHITECTURES` the GPU
generations they are for (Ampere's, run by every compute capability 8.x, and Hopper's; its handlers refuse any
other device)."""

import ctypes
from pathlib import Path
from typing import NamedTuple

_library = ctypes.cdll.LoadLibrary(str(Path(__file__).with_name("libdew_cutlass.so")))
_library.dew_cutlass_kernels.restype = ctypes.c_char_p


class Kernel(NamedTuple):
    """One instantiated kernel: its tile (rows, columns, depth), its ring's stages, its consumer threads (those
    issuing its products), the multiprocessors of its cluster along the rows, and its instruction family
    ("mma.sync": cp.async and Ampere's mma.sync; "wgmma": TMA and Hopper's warpgroup MMA)."""

    tile: tuple[int, int, int]
    stages: int
    consumers: int
    cluster: int
    instructions: str


def _described(entry: str) -> tuple[tuple[str, str], Kernel]:
    """One of the library's entries ("architecture,dtype,rows,columns,depth,stages,consumers,cluster,
    instructions", ffi.cu's `dew_cutlass_kernels`) as its key and kernel."""
    architecture, dtype, *extents, instructions = entry.split(",")
    rows, columns, depth, stages, consumers, cluster = (int(extent) for extent in extents)
    return (architecture, dtype), Kernel((rows, columns, depth), stages, consumers, cluster, instructions)


KERNELS = dict(_described(entry) for entry in _library.dew_cutlass_kernels().decode().split(";"))
"""Each kernel the library instantiates, by its GPU generation ('sm80', 'sm90') and output dtype ('bfloat16',
'float32'): the library's own statement of its compiled types, which the same build's targets run."""

ARCHITECTURES = tuple(dict.fromkeys(architecture for architecture, _ in KERNELS))
"""The GPU generations the library holds grouped kernels for (('sm80', 'sm90'))."""

TARGETS = ("dew_cutlass_grouped_gemm", "dew_cutlass_grouped_gemm_k")
"""The library's FFI targets: the grouped GEMM (its forward, and dx under `transpose_w`) and its weight
gradient, each dispatching on the device's compute capability."""


def targets() -> dict[str, object]:
    """Each target's name and its handler as a PyCapsule, for `jax.ffi.register_ffi_target(name, handler,
    platform="CUDA")`."""
    import jax

    return {name: jax.ffi.pycapsule(getattr(_library, name)) for name in TARGETS}
