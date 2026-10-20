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
from reference_error import assert_as_exact_as_the_reference

from dew.telemetry.instrumentation import custom_call_kind, hlo_instructions
from dew.training.distributed import parameter_axes


def split_axes(tree) -> set[str]:
    """Every mesh axis some leaf of `tree` is split over."""
    names: set[str] = set()
    for leaf in jax.tree.leaves(tree):
        for entry in getattr(getattr(leaf, "sharding", None), "spec", None) or ():
            names.update(entry if isinstance(entry, tuple) else () if entry is None else (entry,))
    return names


def assert_sharded(tree, mesh: jax.sharding.Mesh) -> None:
    """Every parameter axis of `mesh` larger than one splits some leaf of `tree`."""
    wanted = {axis for axis in parameter_axes(mesh.axis_names) if mesh.shape[axis] > 1}
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


def tpu_mesh(**axes: int) -> jax.sharding.Mesh:
    """A mesh of `axes` over the first of this process's devices, its axes left to the partitioner as
    a run's are (`MeshSpec`), for `tpu_lowering`."""
    from jax.sharding import AxisType

    return jax.make_mesh(tuple(axes.values()), tuple(axes), axis_types=(AxisType.Auto,) * len(axes),
                         devices=jax.devices()[:math.prod(axes.values())])


def tpu_lowering(fn, args, mesh: jax.sharding.Mesh) -> str:
    """`fn` over `args`, ShapeDtypeStructs placed on `mesh`, lowered for a TPU v6e through `jax.export`
    on any host, as the module's text: jax refuses there to lower a Mosaic kernel the partitioner
    would have to split ("Mosaic kernels cannot be automatically partitioned")."""
    from dew.nn.kernels import generation

    with generation.assumed("v6e"), jax.set_mesh(mesh):
        return jax.export.export(jax.jit(fn), platforms=["tpu"])(*args).mlir_module()


def assert_runs_per_shard(fn, args, specs, mesh, truth, *, moves=()) -> None:
    """The custom-kernel contract (module docstring): `fn(*args)` with `args` placed on `mesh` by
    `specs` (PartitionSpecs) is within the float64 gate (`assert_as_exact_as_the_reference`) of `fn` on one
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
    assert_as_exact_as_the_reference(jax.tree.map(np.asarray, split), jax.tree.map(np.asarray, whole), truth, 'assert_runs_per_shard')


VIEWS = frozenset({"bitcast", "get-tuple-element", "tuple", "parameter"})
"""Ops that move no data: a layout pass at a kernel's edge is seen through them."""


def _relays_out(instruction) -> bool:
    """Whether an instruction is a pass over an array that only re-lays it out: a copy or transpose,
    or a fusion of nothing but those and views (one with real work beside a transpose writes the
    array it computes in the layout asked, at no extra pass)."""
    if instruction.op in ("copy", "transpose"):
        return True
    return (instruction.op == "fusion" and bool(instruction.fused & {"copy", "transpose"})
            and instruction.fused <= {"copy", "transpose", "reshape", *VIEWS})


def boundary_copies(text: str) -> list[str]:
    """Each layout pass (`_relays_out`) in compiled HLO `text` that produces an operand of a kernel
    call (a Pallas-Triton call or one of Dew's FFI handlers) or reads one of its results, through any
    views (`VIEWS`): XLA re-laying out an array at the kernel's edge. Read with the profiler's own
    HLO parser (`dew.telemetry.instrumentation.hlo_instructions`)."""
    instructions = hlo_instructions(text)
    readers: dict[str, list[str]] = {}
    for name, instruction in instructions.items():
        for operand in instruction.operands:
            readers.setdefault(operand, []).append(name)

    def producers(name: str):
        instruction = instructions.get(name)
        if instruction is not None and instruction.op in VIEWS - {"parameter"}:
            for operand in instruction.operands:
                yield from producers(operand)
        else:
            yield name

    def consumers(name: str):
        for reader in readers.get(name, ()):
            if instructions[reader].op in VIEWS:
                yield from consumers(reader)
            else:
                yield reader

    found = []
    for name, instruction in instructions.items():
        if instruction.op != "custom-call" or not (
                custom_call_kind(instruction.target) == "triton" or instruction.target.startswith("dew_")):
            continue
        found += [f"{producer} -> {name}" for operand in instruction.operands
                  for producer in producers(operand)
                  if producer in instructions and _relays_out(instructions[producer])]
        found += [f"{name} -> {consumer}" for consumer in consumers(name)
                  if _relays_out(instructions[consumer])]
    return found
