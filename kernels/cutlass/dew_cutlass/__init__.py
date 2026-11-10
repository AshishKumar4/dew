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
_library.dew_cutlass_kernel_fields.restype = ctypes.c_char_p
_library.dew_cutlass_architecture.restype = ctypes.c_char_p
_library.dew_cutlass_architecture.argtypes = (ctypes.c_int, ctypes.c_int)


class Kernel(NamedTuple):
    """One instantiated kernel: its tile (rows, columns, depth), its ring's stages, its consumer threads
    (those issuing its products), the multiprocessors of its cluster along the rows, and its instruction
    family ("mma.sync": cp.async and Ampere's mma.sync; "wgmma": TMA and Hopper's warpgroup MMA)."""

    tile: tuple[int, int, int]
    stages: int
    consumers: int
    cluster: int
    instructions: str


_FIELDS = _library.dew_cutlass_kernel_fields().decode().split(",")
"""The fields of each of the library's kernel entries, in its order (ffi.cu's `kNumbers`, its one schema)."""


def _described(entry: str) -> tuple[tuple[str, str], Kernel]:
    """One of the library's entries (ffi.cu's `dew_cutlass_kernels`) as its key and kernel, each field read
    by its name (`_FIELDS`)."""
    field = dict(zip(_FIELDS, entry.split(","), strict=True))
    number = {name: int(value) for name, value in field.items()
              if name not in ("architecture", "dtype", "instructions")}
    kernel = Kernel((number["rows"], number["columns"], number["depth"]), number["stages"],
                    number["consumers"], number["cluster"], field["instructions"])
    return (field["architecture"], field["dtype"]), kernel


KERNELS = dict(_described(entry) for entry in _library.dew_cutlass_kernels().decode().split(";"))
"""Each kernel the library instantiates, by its GPU generation ('sm80', 'sm90') and output dtype ('bfloat16',
'float32'): the library's own statement of its compiled types, which the same build's targets run."""

ARCHITECTURES = tuple(dict.fromkeys(architecture for architecture, _ in KERNELS))
"""The GPU generations the library holds grouped kernels for (('sm80', 'sm90'))."""

def architecture(major: int, minor: int) -> str | None:
    """The generation of the library's kernels (a key of `KERNELS`) a GPU of compute capability
    `major`.`minor` runs, as its targets dispatch on it: Ampere's 'sm80' on every 8.x, Hopper's 'sm90' on
    9.0 alone; None on any other GPU, which its targets refuse."""
    found = _library.dew_cutlass_architecture(major, minor)
    return None if found is None else found.decode()


TARGETS = ("dew_cutlass_grouped_gemm", "dew_cutlass_grouped_gemm_k")
"""The library's FFI targets: the grouped GEMM (its forward, and dx under `transpose_w`) and its weight
gradient, each dispatching on the device's compute capability."""


def targets() -> dict[str, object]:
    """Each target's name and its handler as a PyCapsule, for `jax.ffi.register_ffi_target(name, handler,
    platform="CUDA")`."""
    import jax

    return {name: jax.ffi.pycapsule(getattr(_library, name)) for name in TARGETS}
