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
what it computes on one device, within the float64 gate, and its compiled
step moves none of its operands between devices.
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
    rf"(?P<op>{'|'.join(COLLECTIVE_OPS)})(?:-start)?\(")
ARRAY = re.compile(r"[a-z]+\d*\[([\d,]*)\]")
TYPED = re.compile(r"([a-z]+\d*)\[([\d,]*)\]")
ITEMSIZE = {"pred": 1, "s8": 1, "u8": 1, "bf16": 2, "f16": 2, "f32": 4, "s32": 4, "u32": 4, "f64": 8,
            "s64": 8}


def collectives(text: str) -> list[tuple[str, list[tuple[int, ...]]]]:
    """`(op, result shapes)` of every collective in compiled HLO `text`."""
    found = []
    for line in text.splitlines():
        match = COLLECTIVE.match(line)
        if match:
            shapes = [tuple(int(size) for size in dims.split(",") if size)
                      for dims in ARRAY.findall(match["shape"])]
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
                         for dtype, dims in TYPED.findall(match["shape"]))
    return total


MOVES = ("all-gather", "all-to-all", "collective-permute")
"""The collectives that move an operand's entries between devices, which a kernel running per shard
needs none of: a reduction (all-reduce, reduce-scatter) is a gradient's sum, not a move."""


def assert_runs_per_shard(fn, args, specs, mesh, truth, *, moves=()) -> None:
    """The custom-kernel contract (module docstring): `fn(*args)` with `args` placed on `mesh` by
    `specs` (PartitionSpecs) is within the float64 gate (`assert_leaves_as_exact`) of `fn` on one
    device, both measured from `truth`, the float64 tree of `fn`'s outputs; and its compiled step
    holds no collective that moves entries between devices (`MOVES`) beyond the ops `moves` names
    (a sequence split's intended exchange). XLA, which cannot split an opaque call, would otherwise
    replicate it and gather its operands silently."""
    run = jax.jit(fn)
    whole = run(*args)
    with jax.set_mesh(mesh):
        placed = [jax.device_put(arg, NamedSharding(mesh, spec))
                  for arg, spec in zip(args, specs, strict=True)]
        compiled = run.lower(*placed).compile()
        split = compiled(*placed)
    moved = [(op, shapes) for op, shapes in collectives(compiled.as_text())
             if op in MOVES and op not in moves]
    assert not moved, f"the step moves operands between devices: {moved}"
    assert_leaves_as_exact(jax.tree.map(np.asarray, split), jax.tree.map(np.asarray, whole), truth)
