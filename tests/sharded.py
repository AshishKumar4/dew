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

import contextlib
import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from arm_arithmetic import summed_as_run
from jax.sharding import NamedSharding, PartitionSpec
from reference_error import as_given, assert_as_exact, without_ynn

from dew.distributed.placement import parameter_axes
from dew.telemetry.instrumentation import PALLAS, custom_call_kind, hlo_instructions
from dew.telemetry.trace import collective


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


def collectives(text: str | jax.stages.Compiled) -> list[tuple[str, list[tuple[int, ...]]]]:
    """`(op, result shapes)` of every collective in compiled HLO `text` (or a compiled executable's module,
    `collective_axes`), read by the profiler's own parser (`hlo_instructions`): an async pair once, by its
    `-done`, which holds what moved."""
    return [(kind, list(instruction.shapes)) for instruction in hlo_instructions(text).values()
            if (kind := collective(instruction.op)) and not instruction.op.endswith("-start")]


def collective_axes(text: str | jax.stages.Compiled, mesh: jax.sharding.Mesh
                    ) -> set[tuple[str, frozenset[str]]]:
    """`(op, axes)` of every collective in compiled HLO `text` (or a compiled executable's module, read
    without printing it: XLA prints attributes its parser refuses, an all-reduce-start's
    `output_to_operand_aliasing`) over `mesh`: the axes along which
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


def collective_bytes(text: str | jax.stages.Compiled, ops) -> int:
    """Bytes of the results of every collective among `ops` in compiled HLO `text` (or a compiled
    executable's module, `collective_axes`), as XLA sizes
    them (`HloInstruction.buffers`): a measure no combining of collectives into one, or flattening
    of their operands, changes."""
    return sum(sum(instruction.buffers) for instruction in hlo_instructions(text).values()
               if collective(instruction.op) in ops and not instruction.op.endswith("-start"))


MOVES = ("all-gather", "all-to-all", "collective-permute")
"""The collectives that move an operand's entries between devices, which a kernel running per shard
needs none of: a reduction (all-reduce, reduce-scatter) is a gradient's sum, not a move."""

REDUCTIONS = ("all-reduce", "reduce-scatter")
"""The collectives that sum across devices: each a step takes is one its cell intends, over mesh axes it
names (`split_checked`'s `reductions`), so one on an axis it does not intend fails however the outputs read
(spec alignment, 9001)."""


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


def placed_on(args, specs, mesh: jax.sharding.Mesh) -> list:
    """`args` put on `mesh`, each by its PartitionSpec in `specs`: one for every leaf of its argument, or a
    tree of them for its leaves (a prefix of the argument's tree)."""
    def sharded(spec):
        return jax.tree.map(lambda leaf: NamedSharding(mesh, leaf), spec,
                            is_leaf=lambda leaf: isinstance(leaf, PartitionSpec))

    return [jax.device_put(arg, sharded(spec)) for arg, spec in zip(args, specs, strict=True)]


def split_executable(run, args, specs, mesh: jax.sharding.Mesh):
    """`run` (a jitted function) compiled for `args` placed on `mesh` by `specs`: the split arm's one
    executable, which `assert_runs_per_shard` inspects and runs every draw. Compiled in the scope the
    gate runs each arm in (`summed_as_run`): a compiled executable never retraces, so the arithmetic its
    trace chose is the one every draw runs, and the one-device arm's, traced in that scope, is the same
    (ledger 4393: compiled outside it, a CPU suite's wide sums were frozen into it)."""
    with jax.set_mesh(mesh), summed_as_run():
        return run.lower(*placed_on(args, specs, mesh)).compile()


def _whole_axes(specs, mesh: jax.sharding.Mesh) -> list[frozenset[str]]:
    """Each operand's mesh axes of more than one device that its spec (`specs`' leaves, each a tuple of axis
    names or tuples of them, whose leaves are its names) splits it along none of: the axes its gradient is
    summed over."""
    sized = {name for name, size in mesh.shape.items() if size > 1}
    return [frozenset(sized - set(jax.tree.leaves(tuple(spec))))
            for spec in jax.tree.leaves(specs, is_leaf=lambda leaf: isinstance(leaf, PartitionSpec))]


def split_checked(fn, draw, specs, mesh, *, moves=(), reductions=(), donate=()):
    """`fn` jitted as the gate compiles a Dew arm (`without_ynn`), and its split executable on `draw(0)`'s
    arguments placed by `specs` (`split_executable`), its compiled step holding no collective that moves
    entries between devices (`MOVES`) beyond `moves`, no sum across devices (`REDUCTIONS`) but those it
    intends, and no copy or transpose at a kernel call's edge (`boundary_copies`): `assert_runs_per_shard`'s
    compile-time half. A sum it intends is an operand's gradient's over mesh axes it is whole along (its spec
    naming none of them: `_whole_axes`), as a call of operands to outputs and gradients sums a weight's over
    the rows a mesh splits, or one of `reductions`, each `(op, axes)` over the axes it sums along
    (`collective_axes`): a gathered operand's transpose, which its cell names beside its gather (an fsdp
    weight's reduce-scatter). A sum on axes no operand is whole along and no cell names, which a per-shard
    program should not take, fails, where every reduction passed whole (spec alignment, 9001); a group of
    one device sums nothing across devices."""
    run = without_ynn(fn, donate_argnums=donate)
    compiled = split_executable(run, draw(0), specs, mesh)
    text = compiled.as_text()
    moved = [(op, shapes) for op, shapes in collectives(text) if op in MOVES and op not in moves]
    assert not moved, f"the step moves operands between devices: {moved}"
    found = collective_axes(compiled, mesh)
    whole = _whole_axes(specs, mesh)
    declared = {(op, frozenset(axes)) for op, axes in reductions}

    def intended(op: str, axes: frozenset[str]) -> bool:
        return (op, axes) in declared or any(axes <= each for each in whole)

    summed = sorted((op, sorted(axes)) for op, axes in found
                    if op in REDUCTIONS and axes and not intended(op, axes))
    assert not summed, f"the step sums across devices where no operand's gradient is summed: {summed}"
    copies = boundary_copies(text)
    assert not copies, f"XLA re-lays out arrays at the kernel's boundary: {copies}"
    return run, compiled


def assert_runs_per_shard(fn, draw, specs, mesh, label: str, *, seeds: int, truth_fn, moves=(),
                          reductions=(), donate=(), arm=None, also=None, checked=None):
    """The custom-kernel contract (module docstring): `fn` on `draw(seed)`'s arguments placed on `mesh`
    by `specs` (PartitionSpecs) is on par with `fn` on one device over the site's declared `seeds`
    (check 1, `assert_as_exact`), both measured from `truth_fn`, the float64 tree of `fn`'s outputs on
    the arguments widened; its compiled step, every draw's one program, holds no collective that moves
    entries between devices (`MOVES`) beyond the ops `moves` names (a sequence split's intended
    exchange), which XLA, unable to split an opaque call, would otherwise insert to replicate it; no sum
    across devices but those it intends (`split_checked`: an operand's gradient over the axes it is whole
    along, a weight's over the rows, and the `reductions` a gather's transpose takes); and no copy or
    transpose at a kernel call's edge (`boundary_copies`). `donate` names the arguments `fn`
    takes in place, as its callers donate them (a kernel that aliases its state into its output): a
    call that keeps them would make XLA copy each before the kernel overwrites it. Every arm is Dew's
    program, compiled here as the gate compiles two Dew arms (`without_ynn`) and run as given: the
    split one is the executable whose collectives the contract reads, which it returns. `arm`, when
    given, is the arm the split is compared with in place of `fn` on one device (a plain function,
    compiled as the split is): a call whose rounding depends on the shapes it runs at is compared with
    the shapes its shards run, a call through an exchange with the whole sequence. A caller passing
    `arm` may also assert the split against the reference path (check 1 against float64): `also`,
    `(label, arm)`, is that second arm, compiled alike and compared with the same split runs (each draw's
    split run once, `assert_as_exact`'s further references). `checked`, when given, is `split_checked`'s
    pair already made (a caller reading the split lowering's own records)."""
    run, compiled = checked or split_checked(fn, draw, specs, mesh, moves=moves, reductions=reductions,
                                             donate=donate)
    def kept(args):
        # A donated argument is consumed; the draw's own array goes on to the other arm and the truth.
        return [jnp.copy(arg) if index in donate else arg for index, arg in enumerate(args)]

    def split(*args):
        with jax.set_mesh(mesh):
            return compiled(*placed_on(kept(args), specs, mesh))

    def one_device(*args):
        return run(*kept(args))

    reference = as_given(one_device) if arm is None else as_given(without_ynn(arm))
    further = None if also is None else {also[0]: as_given(without_ynn(also[1]))}
    # Each draw's split run once, against `also`'s arm too (`assert_as_exact`'s further references).
    assert_as_exact(as_given(split), reference, draw, label, seeds=seeds, truth_fn=truth_fn, also=further)
    return compiled


def layer_scope(**named: str) -> contextlib.ExitStack:
    """The scope a run's step traces a layer in, which a per-shard cell's `context` takes (`Shard.context`),
    three pieces in this order: the kernels `named` (`generation.chosen`), which pin the choice under test
    and the routes beside it, as a cell names them; Dew's logical axis rules (`DEFAULT_RULES`), which map
    the layer's declared parameter and activation axes onto the mesh, so its parameters are placed as a run
    places them; and a run's layout (`placing`, every leaf split however small), which a product or a
    dispatch reads to map itself over the mesh's shards. A cell whose layer declares axes and maps a
    product or a dispatch over the mesh takes all three; one that places its operands by its specs alone
    and maps nothing needs none."""
    from flax import linen as nn

    from dew.distributed.placement import Layout
    from dew.mesh import DEFAULT_RULES
    from dew.nn.kernels import generation
    from dew.nn.sharding import placing

    stack = contextlib.ExitStack()
    for context in (generation.chosen(**named), nn.logical_axis_rules(DEFAULT_RULES),
                    placing(Layout(min_shard=1))):
        stack.enter_context(context)
    return stack


@dataclass(frozen=True)
class Shard:
    """A kernel's per-shard cell (testing.md: every Pallas or FFI kernel runs per shard): its layer's path
    with the kernel's choice named (`call`, operands to outputs and gradients), its draw by seed (`draw`),
    where each operand lives on a mesh splitting `split` (`specs(mesh)`, `assert_runs_per_shard`'s), its
    float64 truth (`truth`), its declared check-1 seeds, the moves its layer intends (`moves`: an fsdp
    weight's gather), the scope its layer traces in (`context`: logical axis rules, a layout, the axes its
    parameters declare) and the generation it runs as (`target`)."""

    call: Callable
    draw: Callable
    specs: Callable[[jax.sharding.Mesh], list]
    truth: Callable
    seeds: int
    split: str
    moves: tuple[str, ...] = ()
    context: Callable[[], contextlib.AbstractContextManager] = contextlib.nullcontext
    target: str = "sm90"


def interprets(choice: str, kernel: str, shard: Shard) -> bool:
    """Whether the CPU's devices run `shard`'s kernel as its generation, interpreted: its call on one
    device, traced with the kernel named as its target interpreted (`generation.assumed(...,
    interpreted=True)`), takes the kernel, no requirement refusing it (a library's FFI call, a Triton launch,
    a launch no interpreter derives) and no lowering it cannot make."""
    from dew.nn.kernels import generation, selection

    with (generation.assumed(shard.target, interpreted=True), generation.chosen(**{choice: kernel}),
          shard.context(), generation.routes() as routes):
        try:
            jax.eval_shape(shard.call, *shard.draw(0))
        except (selection.KernelRefused, selection.Unlowerable):
            return False
    return kernel in {found.kernel for found in routes if found.choice == choice}


def assert_kernel_runs_per_shard(choice: str, kernel: str, shard: Shard) -> None:
    """`kernel`, `choice`'s, runs per shard (`assert_runs_per_shard`) on two devices of a run's mesh
    splitting `shard.split` (`MeshSpec`, its other axes one device each), its choice named
    (`generation.chosen`), the CPU's devices running the kernels as `shard.target` interpreted
    (`generation.assumed`; a GPU's or a TPU's as themselves), in its layer's scope: some operand split on
    `shard.split`, and the split program's own lowering (`split_checked`, apart from the one-device arm's
    and the truth's) taking the kernel (`generation.routes`), not a fallback."""
    from dew.distributed.placement import MeshSpec
    from dew.nn.kernels import generation

    # The data axis takes the devices the others leave (`MeshSpec`).
    mesh = MeshSpec(**({} if shard.split == "data" else {shard.split: 2})).build(jax.devices()[:2])
    specs = shard.specs(mesh)
    named = set().union(*(jax.tree.leaves(tuple(spec)) for spec in jax.tree.leaves(
        specs, is_leaf=lambda leaf: isinstance(leaf, PartitionSpec))))
    assert shard.split in named and mesh.shape[shard.split] > 1, (shard.split, sorted(named), mesh.shape)
    from generation_support import here

    target = (generation.assumed(shard.target, interpreted=True) if here() == "cpu"
              else contextlib.nullcontext())
    with target, generation.chosen(**{choice: kernel}), shard.context():
        with generation.routes() as routes:
            checked = split_checked(shard.call, shard.draw, specs, mesh, moves=shard.moves)
        taken = {found.kernel for found in routes if found.choice == choice}
        assert kernel in taken, (choice, kernel, sorted(taken))
        assert_runs_per_shard(shard.call, shard.draw, specs, mesh, f"{choice} {kernel} per shard over "
                              f"{shard.split}", seeds=shard.seeds, truth_fn=shard.truth, moves=shard.moves,
                              checked=checked)


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
    call (a Pallas call of any kind, Triton's, Mosaic GPU's or Mosaic TPU's, as the profiler classifies
    them, `instrumentation.PALLAS`; or one of Dew's FFI handlers) or reads one of its results, through any
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
                custom_call_kind(instruction.target) in PALLAS or instruction.target.startswith("dew_")):
            continue
        found += [f"{producer} -> {name}" for operand in instruction.operands
                  for producer in producers(operand)
                  if producer in instructions and _relays_out(instructions[producer])]
        found += [f"{name} -> {consumer}" for consumer in consumers(name)
                  if _relays_out(instructions[consumer])]
    return found
