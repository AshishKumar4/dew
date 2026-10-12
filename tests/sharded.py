"""How a computation splits over a mesh: the shared helpers of the distribution tests.

A test that compares a mesh with one device proves that layout only if the
parameters are split. `Layout` replicates every parameter below its
`min_shard` elements, and a tiny model's parameters are all below the
default, so such a test can pass on data parallelism alone. A meshed
parity test checks its placed parameters with `assert_sharded`.

A custom kernel (a `pallas_call`, an FFI call) is opaque to XLA's
partitioner, which replicates it and gathers its inputs to every device
unless the kernel runs per shard. Every such kernel's entry point keeps one
contract, which `assert_runs_per_shard` checks on a mesh of CPU devices in
the suite and of GPUs on a multi-GPU host: split over the mesh it computes
what it computes on one device, within the float64 gate, its compiled step
moves none of its operands between devices, and no copy or transpose sits
at its boundary (`boundary_copies`): a Pallas-Triton call takes and gives
row-major arrays, so a producer or consumer laid out otherwise costs a pass
over the array that the layout at the boundary, not the logical shape, has
to remove. A CPU interprets Pallas kernels, so there only the first two
hold; the third needs the GPU.
"""

import math
import re

import jax
import numpy as np
from jax.sharding import NamedSharding
from reference_error import assert_leaves_as_exact

from dew.nn.sharding import EXPERT_AXIS, FSDP_AXIS, TENSOR_AXIS

PARAMETER_AXES = (FSDP_AXIS, TENSOR_AXIS, EXPERT_AXIS)
"""The mesh axes a `Layout` splits parameters over."""


def split_axes(tree) -> set[str]:
    """Every mesh axis some leaf of `tree` is split over."""
    names: set[str] = set()
    for leaf in jax.tree.leaves(tree):
        for entry in getattr(getattr(leaf, "sharding", None), "spec", None) or ():
            names.update(entry if isinstance(entry, tuple) else () if entry is None else (entry,))
    return names


def assert_sharded(tree, mesh: jax.sharding.Mesh) -> None:
    """Every parameter axis of `mesh` larger than one splits some leaf of `tree`."""
    wanted = {axis for axis in PARAMETER_AXES if mesh.shape.get(axis, 1) > 1}
    missing = wanted - split_axes(tree)
    assert not missing, f"no parameter is split over {sorted(missing)}, so this mesh runs as data parallelism"


COLLECTIVE_OPS = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all", "collective-permute")
COLLECTIVE = re.compile(
    r"^\s*(?:ROOT\s+)?%\S+\s+=\s+(?P<shape>.+?)\s+"
    rf"(?P<op>{'|'.join(COLLECTIVE_OPS)})(?P<start>-start)?\(")
ARRAY = re.compile(r"[a-z]+\d*\[([\d,]*)\]")
TYPED = re.compile(r"([a-z]+\d*)\[([\d,]*)\]")
ITEMSIZE = {"pred": 1, "s8": 1, "u8": 1, "bf16": 2, "f16": 2, "f32": 4, "s32": 4, "u32": 4, "f64": 8,
            "s64": 8}


def _result(match) -> str:
    """A collective's result shape: an async `-start`'s is the last member of its `(operands,
    results)` tuple (SwiftHeron's 8B FSDP step counted its all-gathers' inputs too, 1.5x)."""
    shape = match["shape"]
    if not match["start"]:
        return shape
    depth = 0
    for index in range(len(shape) - 2, 0, -1):
        depth += {")": 1, "]": 1, "}": 1, "(": -1, "[": -1, "{": -1}.get(shape[index], 0)
        if depth == 0 and shape[index] == ",":
            return shape[index + 1:-1]
    return shape


def collectives(text: str) -> list[tuple[str, list[tuple[int, ...]]]]:
    """`(op, result shapes)` of every collective in compiled HLO `text`."""
    found = []
    for line in text.splitlines():
        match = COLLECTIVE.match(line)
        if match:
            shapes = [tuple(int(size) for size in dims.split(",") if size)
                      for dims in ARRAY.findall(_result(match))]
            found.append((match["op"], shapes))
    return found


def collective_bytes(text: str, ops) -> int:
    """Bytes of the results of every collective among `ops` in compiled HLO `text`: a measure no
    combining of collectives into one, or flattening of their operands, changes."""
    total = 0
    for line in text.splitlines():
        match = COLLECTIVE.match(line)
        if match and match["op"] in ops:
            total += sum(ITEMSIZE[dtype] * math.prod(int(size) for size in dims.split(",") if size)
                         for dtype, dims in TYPED.findall(_result(match)))
    return total


MOVES = ("all-gather", "all-to-all", "collective-permute")
"""The collectives that move an operand's entries between devices, which a kernel running per shard
needs none of: a reduction (all-reduce, reduce-scatter) is a gradient's sum, not a move."""


def assert_runs_per_shard(fn, args, specs, mesh, truth, *, moves=()) -> None:
    """The custom-kernel contract (module docstring): `fn(*args)` with `args` placed on `mesh` by
    `specs` (PartitionSpecs) is within the float64 gate (`assert_leaves_as_exact`) of `fn` on one
    device, both measured from `truth`, the float64 tree of `fn`'s outputs; its compiled step holds
    no collective that moves entries between devices (`MOVES`) beyond the ops `moves` names (a
    sequence split's intended exchange), which XLA, unable to split an opaque call, would otherwise
    insert to replicate it; and no copy or transpose at a kernel call's edge (`boundary_copies`)."""
    run = jax.jit(fn)
    whole = run(*args)
    with jax.set_mesh(mesh):
        placed = [jax.device_put(arg, NamedSharding(mesh, spec))
                  for arg, spec in zip(args, specs, strict=True)]
        compiled = run.lower(*placed).compile()
        split = compiled(*placed)
    text = compiled.as_text()
    moved = [(op, shapes) for op, shapes in collectives(text) if op in MOVES and op not in moves]
    assert not moved, f"the step moves operands between devices: {moved}"
    copies = boundary_copies(text)
    assert not copies, f"XLA re-lays out arrays at the kernel's boundary: {copies}"
    assert_leaves_as_exact(jax.tree.map(np.asarray, split), jax.tree.map(np.asarray, whole), truth)


INSTRUCTION = re.compile(
    r"^\s*(?:ROOT\s+)?%(?P<name>\S+)\s+=\s+.+?\s(?P<op>[a-z][a-z0-9-]*)\((?P<operands>.*)\)")
KERNEL_TARGETS = re.compile(r'custom_call_target="(__gpu\$xla\.gpu\.triton|dew_[a-z_]+)"')
"""The custom calls a custom kernel lowers to: a Pallas-Triton call, or one of Dew's FFI handlers."""


def boundary_copies(text: str) -> list[str]:
    """Each copy or transpose in compiled HLO `text` (an instruction of that opcode, or a fusion XLA
    named for one) that produces an operand of a kernel call (`KERNEL_TARGETS`) or reads one of its
    results, directly or through a tuple element: XLA re-laying out an array at the kernel's edge."""
    ops, readers, kernels = {}, {}, set()
    for line in text.splitlines():
        match = INSTRUCTION.match(line)
        if not match:
            continue
        name, op = match["name"], match["op"]
        ops[name] = op
        for operand in re.findall(r"%([\w.-]+)", match["operands"]):
            readers.setdefault(operand, []).append(name)
        if op == "custom-call" and KERNEL_TARGETS.search(line):
            kernels.add((name, tuple(re.findall(r"%([\w.-]+)", match["operands"]))))

    def moves(name):
        return ops.get(name) in ("copy", "transpose") or (
            ops.get(name) == "fusion" and re.search(r"copy|transpose", name) is not None)

    found = []
    for kernel, operands in kernels:
        found += [f"{operand} -> {kernel}" for operand in operands if moves(operand)]
        results = [kernel, *(reader for reader in readers.get(kernel, ())
                             if ops.get(reader) == "get-tuple-element")]
        found += [f"{kernel} -> {reader}" for result in results for reader in readers.get(result, ())
                  if moves(reader)]
    return found
