"""A launch certified aside replayed from its program's own call of it (`report.capture`): the mesh it is
replayed on and its replay lowered alone, as its program traced it, and compiled (row 8720)."""

from __future__ import annotations

import jax
import numpy as np

from dew.nn.kernels import generation
from dew.nn.kernels.templates import report


def replay_mesh(closed, device) -> jax.sharding.Mesh | None:
    """The mesh a captured launch (`capture`) is replayed on: the one its abstract values are placed on, as
    the trace's concrete mesh holds it (`jax.set_mesh`) or the run it targets does (`targeting`), else as
    compile-only GPUs of `device`'s generation hold it (a trace under an abstract mesh alone,
    `jax.sharding.use_abstract_mesh`), `device` itself for a mesh of one; `device` alone for a launch placed
    on none (a trace in no mesh). None where its values are placed on meshes of two shapes."""
    from jax._src import mesh as mesh_lib

    placed = {tuple(aval.sharding.mesh.shape.items()): aval.sharding.mesh for aval in _avals(closed)
              if getattr(aval, "sharding", None) is not None and not aval.sharding.mesh.empty}
    if not placed:
        return jax.sharding.Mesh([device], ("launch",))
    if len(placed) > 1:
        return None
    ((shape, abstract),) = placed.items()
    held = (mesh_lib.get_concrete_mesh(), report._TARGETED.get())
    found = next((mesh for mesh in held if mesh is not None and not mesh.empty
                  and tuple(mesh.shape.items()) == shape), None)
    if found is not None:
        return found
    devices = ([device] if abstract.size == 1 else
               generation.compile_only_gpus(generation.device_target(device).generation, abstract.size))
    # A shard_map's axes are manual within it alone: the replay's own shard_map makes them so again.
    kinds = tuple(jax.sharding.AxisType.Auto if kind == jax.sharding.AxisType.Manual else kind
                  for kind in abstract.axis_types)
    return jax.sharding.Mesh(np.array(devices).reshape(abstract.axis_sizes), abstract.axis_names,
                             axis_types=kinds)


def _avals(closed) -> list:
    """A captured launch's abstract values: its constants' (a hook's closed-over operand), its arguments'
    and its outputs'."""
    return [*map(jax.typeof, closed.consts), *closed.in_avals, *closed.out_avals]


def _varying(aval) -> tuple[str, ...]:
    """The manual axes `aval` varies over inside a shard_map (its checked type), none outside one."""
    return tuple(sorted(aval.manual_axis_type.varying))


def replayed(closed, mesh: jax.sharding.Mesh, reverse: bool, device, name: str
              ) -> tuple[dict, str | None, bool]:
    """A captured launch (`capture`) lowered alone on `mesh` as its program traced it, inside a shard_map of
    its manual axes where it was traced in one (each operand cast to the axes it varied over, each output
    summed over its), as a call without a gradient runs it or, `reverse`, as reverse mode does (its forward
    and its pullback of a unit cotangent), named `name`; its kernels' verdict read back where stored on
    `device`'s GPU model (`_lowered_verdict_key`), else compiled (`_launch_binaries`). Its kernels, their
    verdict's key, and whether the verdict was read."""
    from jax._src.interpreters import partial_eval as pe
    from jax.extend import core as jax_core
    from jax.sharding import NamedSharding, PartitionSpec

    from dew.nn.kernels import tuning
    from dew.nn.kernels.options import owned_program
    from dew.nn.programs import meshless

    # In the context its capture copied, of no launch tracing, no block's aside and no settle: a replayed
    # derivative's launches name their own and certify in turn, the block's pending none of theirs, and its
    # certifications step past none of the settle's misses, which its own thread reads.
    report._TRACING.set(None)
    report._ASIDE.set(None)
    report._SETTLING.set(None)
    # Its constants (a hook's closed-over operand) its arguments, a closed program of none (`type(closed)`).
    launch = jax_core.jaxpr_as_fun(type(closed)(pe.convert_constvars_jaxpr(closed.jaxpr), ()))
    operands = [*map(jax.typeof, closed.consts), *closed.in_avals]
    manual = frozenset(axis for aval in _avals(closed) if getattr(aval, "sharding", None) is not None
                       for axis in aval.sharding.mesh.manual_axes)
    checked = any(_varying(aval) for aval in _avals(closed))

    def varied(value, aval):
        # A value cast to the axes `aval` varies over, as its program's checked shard_map typed it.
        axes = _varying(aval) if checked else ()
        return jax.lax.pcast(value, axes, to="varying") if axes else value

    def replayed(*values):
        values = [varied(value, aval) for value, aval in zip(values, operands, strict=True)]
        if not reverse:
            outputs = launch(*values)
        else:
            outputs, pullback = jax.vjp(launch, *values)
            ones = [each if isinstance(each, np.ndarray) else varied(each, jax.typeof(output))
                    for each, output in ((report._unit_cotangent(output), output) for output in outputs)]
            outputs = [*outputs, *(cotangent for cotangent in pullback(ones)
                                   if cotangent.dtype != jax.dtypes.float0)]
        # Each output summed over the axes it varies over: a shard_map's outputs stated whole.
        return [jax.lax.psum(each, axes) if (axes := _varying(jax.typeof(each)) if checked else ()) else each
                for each in outputs]

    certified = replayed
    if manual:
        certified = jax.shard_map(replayed, mesh=mesh, in_specs=tuple(PartitionSpec() for _ in operands),
                                  out_specs=PartitionSpec(), axis_names=manual, check_vma=checked)
    certified.__name__ = certified.__qualname__ = name
    specs = [jax.ShapeDtypeStruct(aval.shape, aval.dtype, sharding=NamedSharding(
        mesh, PartitionSpec() if manual or getattr(aval, "sharding", None) is None else aval.sharding.spec))
        for aval in operands]
    # Under the trace's own JAX settings, its default device among them, which its capture carried
    # (`_cache.carried`): a Triton kernel's lowering reads an abstract GPU's architecture off the default
    # device (`pallas.triton.gpu_info`), and the Mosaic GPU lowering cache keys every setting, so the
    # program's lowering on the trace's thread reads the replay's kernels back (row 11803: a replay set its
    # device as the default the trace left unset, and 44 of an every-template decoder's kernels lowered
    # twice).
    with meshless(mesh):
        lowered = owned_program(certified).trace(*specs).lower()
    modules, lowered_key = report._lowered_kernels(lowered)
    target = report.verdict_target(device)
    verdict = None if lowered_key is None else tuning.stored_verdict(target, lowered_key)
    if isinstance(verdict, dict):
        return verdict, lowered_key, True
    return report._launch_binaries(lowered, modules), lowered_key, False
