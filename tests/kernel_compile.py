"""Compile Pallas kernels for a GPU on this CPU with the compiler the GPU itself runs: the CPU suite's
first filter for declared kernel configs (tests/test_kernel_configs.py). XLA's CUDA backend (this
jaxlib's CUDA plugin: the `kernel-compile` extra on a host with no GPU, a GPU host's own CUDA extra)
compiles a program for a compile-only device of the generation, no GPU attached, through the Triton
jaxlib bundles: so a kernel's shared bytes here are the bytes the GPU asks for, the GPU's own refusal
included. A host without the plugin fails here rather than skip. A GPU's tests are the verdict on
everything else.
"""

import contextlib
import functools
import re
import tempfile
from pathlib import Path

import jax
import jax.experimental.topologies as topologies
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from dew.nn.kernels.generation import assumed, capability, gpu_mesh, gpu_target

THREADS_PER_MULTIPROCESSOR = {(8, 0): 2048, (9, 0): 2048, (10, 0): 2048}
"""Resident threads a multiprocessor holds by compute capability, 1536 on the others the targets name
(CUDA C++ Programming Guide, "Technical Specifications per Compute Capability")."""

REFUSED = re.compile(r"Shared memory size limit exceeded: requested (\d+)")
"""XLA's refusal of a Triton kernel whose shared memory a block of the GPU does not hold."""

KERNEL = re.compile(r"kCustomKernel .*? shared_memory: (\d+) bytes")
"""A custom (Triton) kernel's line in XLA's thunk sequence, with the shared bytes it launches with."""


def target_config(generation: str) -> str:
    """The `gpu_device_info` XLA compiles for, from the CUDA `generation`'s target
    (`generation.TARGETS`): its compute capability, the shared memory a block may opt into, its
    multiprocessors, clock and bandwidth; CUDA's per-capability limits for the rest. The fields past
    the capability and shared memory only steer XLA's own fusions, never a Triton kernel's code."""
    target, version = gpu_target(generation), capability(generation)
    facts = (target.shared_limit, target.multiprocessors, target.clock_hz, target.memory_bandwidth)
    if version is None or any(fact is None for fact in facts):
        raise ValueError(f"no CUDA target states {generation!r}'s compile facts")
    major, minor = version
    shared, multiprocessors, clock, bandwidth = facts
    return f"""gpu_device_info {{
  threads_per_block_limit: 1024
  threads_per_warp: 32
  shared_memory_per_block: 49152
  shared_memory_per_core: {shared + 1024}
  threads_per_core_limit: {THREADS_PER_MULTIPROCESSOR.get((major, minor), 1536)}
  core_count: {multiprocessors}
  fpus_per_core: {64 if (major, minor) == (8, 0) else 128}
  block_dim_limit_x: 2147483647
  block_dim_limit_y: 65535
  block_dim_limit_z: 65535
  memory_bandwidth: {int(bandwidth)}
  clock_rate_ghz: {clock / 1e9}
  shared_memory_per_block_optin: {shared}
  cuda_compute_capability {{
    major: {major}
    minor: {minor}
  }}
  registers_per_core_limit: 65536
  registers_per_block_limit: 65536
}}
platform_name: "CUDA"
dnn_version_info {{
  major: 9
}}
device_description_str: "{target.device_kind}"
"""


@functools.cache
def _device(generation: str) -> Mesh:
    """A one-device mesh of a compile-only device of `generation`."""
    topology = topologies.get_topology_desc("dew-compile", "cuda", target_config=target_config(generation),
                                            topology="1x1x1")
    return Mesh(np.array(topology.devices[:1]), ("data",))


@contextlib.contextmanager
def _uncached():
    """No persistent compilation cache: an executable of a compile-only device is never one to load,
    nor one a GPU's run should find under its key."""
    enabled = jax.config.jax_enable_compilation_cache
    jax.config.update("jax_enable_compilation_cache", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_compilation_cache", enabled)


def failures(found: list[dict]) -> list[str]:
    """Each compiled record's failure, if any: its error, or its shared bytes past its block's limit."""
    return [f"{record['name']}: {record['error']}" if "error" in record else
            f"{record['name']}: {record['shared']} shared bytes, limit {record['limit']}"
            for record in found if "error" in record or not record["shared"] <= record["limit"]]


def compile_kernels(cases, generation: str) -> list[dict]:
    """Each case (`name`, `call()` giving the kernel and abstract operands) on the CUDA `generation`,
    lowered as on that GPU on this host (`assumed`, `gpu_mesh`) and compiled by XLA for a compile-only
    device of it, never run: its name, the generation's block limit, and the shared bytes its Triton
    kernels launch with (the most of any; those XLA refused, where it refused one), or the error.

    A cheap first filter: a kernel that does not compile or takes more shared memory than its GPU's
    block holds fails here as it would there, before a GPU is paid for."""
    mesh = _device(generation)
    replicated = NamedSharding(mesh, PartitionSpec())
    found: list[dict] = []
    for case in cases:
        record: dict = {"name": case.name, "generation": generation,
                        "limit": gpu_target(generation).shared_limit, "shared": 0}
        found.append(record)
        fn, args = case.call()
        placed = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=replicated)
                              if isinstance(leaf, jax.ShapeDtypeStruct) else leaf, args)
        try:
            with assumed(generation), jax.sharding.use_abstract_mesh(gpu_mesh(mesh, generation)):
                lowered = jax.jit(fn).trace(*placed).lower()
        except Exception as error:  # A case that does not lower is its own failure.
            record["error"] = f"it does not lower: {str(error)[-800:]}"
            continue
        with tempfile.TemporaryDirectory(prefix="dew-compile-") as dump, _uncached():
            try:
                lowered.compile(compiler_options={"xla_dump_to": dump})
            except jax.errors.JaxRuntimeError as error:
                refused = REFUSED.search(str(error))
                if refused is None:
                    record["error"] = f"XLA does not compile it: {str(error)[-800:]}"
                else:
                    record["shared"] = int(refused.group(1))
                continue
            sequences = [path.read_text() for path in Path(dump).glob("*thunk_sequence.txt")]
        record["shared"] = max((int(size) for text in sequences for size in KERNEL.findall(text)), default=0)
    return found
