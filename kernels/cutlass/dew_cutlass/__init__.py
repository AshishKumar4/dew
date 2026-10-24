"""CUTLASS's grouped GEMMs (NVIDIA, BSD-3) as XLA FFI targets of a library with no Python ABI: `targets()`
names each target and its handler, which Dew registers with jax's FFI at its first call, and
`ARCHITECTURES` says which GPU generations the installed build holds grouped kernels for (Hopper's today;
its handlers refuse any other device)."""

import ctypes
import re
from pathlib import Path

_library = ctypes.cdll.LoadLibrary(str(Path(__file__).with_name("libdew_cutlass.so")))
_library.dew_cutlass_architectures.restype = ctypes.c_char_p

ARCHITECTURES = tuple(f"sm{re.match(r'[0-9]+', entry).group()}"
                      for entry in _library.dew_cutlass_architectures().decode().split(","))
"""The GPU generations the library holds grouped kernels for, as its build compiled them ('sm90')."""

TARGETS = ("dew_cutlass_grouped_gemm", "dew_cutlass_grouped_gemm_k")
"""The library's FFI targets: the grouped GEMM (its forward, and dx under `transpose_w`) and its weight
gradient, each dispatching on the device's compute capability."""


def targets() -> dict[str, object]:
    """Each target's name and its handler as a PyCapsule, for `jax.ffi.register_ffi_target(name, handler,
    platform="CUDA")`."""
    import jax

    return {name: jax.ffi.pycapsule(getattr(_library, name)) for name in TARGETS}
