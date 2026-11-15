"""The multiprocessors an attached GPU's clusters place on, as its CUDA driver reports them
(`cluster_placement`), which an attached device's target reads in place of the table's
(`dew.nn.kernels.generation.runtime_facts`, `GpuTarget.cluster_placed`)."""
import ctypes
from typing import ClassVar

from dew._devices import gpu_attribute, primary_context

COMPUTE_CAPABILITY_MAJOR, MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 75, 97
"""`CUdevice_attribute`s (cuda.h): the compute capability's major, and a block's shared memory once a kernel
opts in."""


class _ClusterDim(ctypes.Structure):
    _fields_: ClassVar = [('x', ctypes.c_uint), ('y', ctypes.c_uint), ('z', ctypes.c_uint)]


class _LaunchAttribute(ctypes.Structure):
    # cuda.h's CUlaunchAttribute: its id, then a 64-byte union aligned to 8, here its cluster dimension.
    _fields_: ClassVar = [('id', ctypes.c_int), ('pad', ctypes.c_char * 4), ('cluster', _ClusterDim),
                ('rest', ctypes.c_char * (64 - ctypes.sizeof(_ClusterDim)))]


class _LaunchConfig(ctypes.Structure):
    _fields_: ClassVar = [('grid', ctypes.c_uint * 3), ('block', ctypes.c_uint * 3),
                          ('shared', ctypes.c_uint),
                          ('stream', ctypes.c_void_p), ('attrs', ctypes.POINTER(_LaunchAttribute)),
                          ('count', ctypes.c_uint)]


CLUSTER_DIMENSION, MAX_DYNAMIC_SHARED_SIZE = 4, 8
"""cuda.h's `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` and `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`."""


def cluster_placement(ordinal: int, programs: int) -> int | None:
    """The multiprocessors of the visible GPU `ordinal` that clusters of `programs` programs place on, each
    program alone on its multiprocessor (a persistent launch's: its block takes the shared memory a block
    may opt into), as the CUDA driver reports it (`cuOccupancyMaxActiveClusters` of an empty kernel, times
    `programs`): a cluster's programs share a GPC, so a GPC's remainder stays unplaced. None where no driver
    library loads or the device takes no clusters (before compute capability 9.0); a driver error raises."""
    major = gpu_attribute(ordinal, COMPUTE_CAPABILITY_MAJOR)
    if major is None or major < 9:
        return None
    shared = gpu_attribute(ordinal, MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    cuda, context, device = primary_context(ordinal)
    # The empty kernel's PTX at Hopper's target and its ISA, which the driver compiles for any later part.
    ptx = (b".version 8.0\n.target sm_90\n.address_size 64\n"
           b".visible .entry dew_placement() { ret; }\n\0")
    module, function, clusters = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_int()
    attribute = _LaunchAttribute(id=CLUSTER_DIMENSION, cluster=_ClusterDim(programs, 1, 1))
    config = _LaunchConfig(grid=(ctypes.c_uint * 3)(programs, 1, 1), block=(ctypes.c_uint * 3)(128, 1, 1),
                           shared=shared, attrs=ctypes.pointer(attribute), count=1)
    try:
        if status := cuda.cuCtxPushCurrent_v2(context):
            raise RuntimeError(f"CUDA driver error {status}")
        try:
            for status in (cuda.cuModuleLoadData(ctypes.byref(module), ptx),
                           cuda.cuModuleGetFunction(ctypes.byref(function), module, b"dew_placement"),
                           cuda.cuFuncSetAttribute(function, MAX_DYNAMIC_SHARED_SIZE, shared),
                           cuda.cuOccupancyMaxActiveClusters(ctypes.byref(clusters), function,
                                                             ctypes.byref(config))):
                if status:
                    raise RuntimeError(f"CUDA driver error {status}")
            return clusters.value * programs
        finally:
            if module.value:
                cuda.cuModuleUnload(module)
            cuda.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))
    finally:
        cuda.cuDevicePrimaryCtxRelease_v2(device)
