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
what it computes on one device, on par with it over the site's draws (check
1 of the float64 gate, `assert_as_exact`), its compiled step
moves none of its operands between devices, and no copy or transpose sits
at its boundary (`boundary_copies`): a Pallas-Triton call takes and gives
row-major arrays, so a producer or consumer laid out otherwise costs a pass
over the array that the layout at the boundary, not the logical shape, has
to remove. A CPU interprets Pallas kernels, so there only the first two
hold; the third needs the GPU.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
from reference_error import as_given, assert_as_exact, without_ynn

from dew.distributed.placement import parameter_axes
from dew.telemetry.instrumentation import collective, custom_call_kind, hlo_instructions


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


def collectives(text: str) -> list[tuple[str, list[tuple[int, ...]]]]:
    """`(op, result shapes)` of every collective in compiled HLO `text`, read by the profiler's own
    parser (`hlo_instructions`): an async pair once, by its `-done`, which holds what moved."""
    return [(kind, list(instruction.shapes)) for instruction in hlo_instructions(text).values()
            if (kind := collective(instruction.op)) and not instruction.op.endswith("-start")]


def collective_axes(text: str, mesh: jax.sharding.Mesh) -> set[tuple[str, frozenset[str]]]:
    """`(op, axes)` of every collective in compiled HLO `text` over `mesh`: the axes along which
    the devices of one of its groups (`HloInstruction.groups`) differ, each group's positions read
    in the program's device assignment, `mesh`'s devices in order. An async pair once, by its
    `-start`, which holds the groups."""
    found = set()
    for instruction in hlo_instructions(text).values():
        if (kind := collective(instruction.op)) is None or instruction.op.endswith("-done"):
            continue
        groups = instruction.groups or (tuple(range(mesh.size)),)
        coordinates = [np.unravel_index(list(group), mesh.devices.shape) for group in groups]
        found.add((kind, frozenset(name for index, name in enumerate(mesh.axis_names)
                                   if any(len(set(within[index].tolist())) > 1 for within in coordinates))))
    return found


def collective_bytes(text: str, ops) -> int:
    """Bytes of the results of every collective among `ops` in compiled HLO `text`, as XLA sizes
    them (`HloInstruction.buffers`): a measure no combining of collectives into one, or flattening
    of their operands, changes."""
    return sum(sum(instruction.buffers) for instruction in hlo_instructions(text).values()
               if collective(instruction.op) in ops and not instruction.op.endswith("-start"))


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


def placed(tree, mesh: jax.sharding.Mesh, *spec):
    """`tree`'s arrays as abstract values placed on `mesh` by `PartitionSpec(*spec)`, for a step traced
    and lowered for a mesh it does not run on."""
    sharding = NamedSharding(mesh, PartitionSpec(*spec))
    return jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding), tree)


def assert_runs_per_shard(fn, draw, specs, mesh, label: str, *, seeds: int, truth_fn, moves=(),
                          donate=()) -> None:
    """The custom-kernel contract (module docstring): `fn` on `draw(seed)`'s arguments placed on `mesh`
    by `specs` (PartitionSpecs) is on par with `fn` on one device over the site's declared `seeds`
    (check 1, `assert_as_exact`), both measured from `truth_fn`, the float64 tree of `fn`'s outputs on
    the arguments widened; its compiled step, every draw's one program, holds no collective that moves
    entries between devices (`MOVES`) beyond the ops `moves` names (a sequence split's intended
    exchange), which XLA, unable to split an opaque call, would otherwise insert to replicate it; and no
    copy or transpose at a kernel call's edge (`boundary_copies`). `donate` names the arguments `fn`
    takes in place, as its callers donate them (a kernel that aliases its state into its output): a
    call that keeps them would make XLA copy each before the kernel overwrites it. Both arms are Dew's
    programs, compiled here as the gate compiles two Dew arms (`without_ynn`) and run as given: the
    split one is the executable whose collectives the contract reads."""
    run = without_ynn(fn, donate_argnums=donate)

    def put(args):
        return [jax.device_put(arg, NamedSharding(mesh, spec)) for arg, spec in zip(args, specs, strict=True)]

    with jax.set_mesh(mesh):
        compiled = run.lower(*put(draw(0))).compile()
    text = compiled.as_text()
    moved = [(op, shapes) for op, shapes in collectives(text) if op in MOVES and op not in moves]
    assert not moved, f"the step moves operands between devices: {moved}"
    copies = boundary_copies(text)
    assert not copies, f"XLA re-lays out arrays at the kernel's boundary: {copies}"

    def kept(args):
        # A donated argument is consumed; the draw's own array goes on to the other arm and the truth.
        return [jnp.copy(arg) if index in donate else arg for index, arg in enumerate(args)]

    def split(*args):
        with jax.set_mesh(mesh):
            return compiled(*put(kept(args)))

    def one_device(*args):
        return run(*kept(args))

    assert_as_exact(as_given(split), as_given(one_device), draw, label, seeds=seeds, truth_fn=truth_fn)


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
