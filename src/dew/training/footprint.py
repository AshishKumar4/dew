"""A candidate step's device bytes, derived from its shapes at trace cost (rows 2113, 2114).

The fit search admits a step where its temporaries, the new outputs it does not alias and what its
first call builds beside it (`Candidate.held`) take a free block of every device (`step_fits`); the
state it reads is already resident. A footprint states those terms before any compile: the
residuals the loss's pullback keeps between forward and backward at one microbatch, read off the
pullback's traced program (`residual_bytes`), and the trained parameters' gradient, each a device's
share. A step over several devices leaves its residuals' shares to XLA's partitioner, which their
trace does not state, so there the residuals are unknown. XLA's workspace beyond them is no term of
it: what XLA's buffer assignment and scheduling make of a program is no function of its residuals
(tools/benchmark_decoder_remat.py, RTX 4080: 326 MB of residuals against 138 MB of temporaries without
recomputation, 13.5 against 31.9 at 'full'), so the run's one compile confirms the derivation, and the
plan records each term beside what the compile measured.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.extend.core as jcore
import numpy as np
from jax.core import ShapedArray

from dew._records import JSON
from dew.batch import Batch
from dew.data.dataset import rows_of
from dew.distributed.placement import placed_bytes
from dew.mesh import DATA_AXIS, FSDP_AXIS, row_axes
from dew.objectives.base import Objective, Step
from dew.optim import clips_qk
from dew.state import TrainState
from dew.training.narrow import forward_variables
from dew.training.transaction import with_ema


@dataclasses.dataclass(frozen=True)
class Footprint:
    """What a candidate step takes of a device beside the state it reads, derived from its shapes:
    `residuals`, the bytes its loss's pullback keeps from forward to backward at one microbatch;
    `gradients`, the trained parameters' gradient; and `held`, what its first call builds beside the
    resident state (`Candidate.held`), each a device's share. `residuals` is None for a step over
    several devices, whose shares of them the partitioner chooses."""

    residuals: int | None
    gradients: int
    held: int
    resident: Mapping[str, int] = dataclasses.field(default_factory=dict)
    peak: int | None = None
    """The most bytes its loss's gradient program holds at once at one microbatch, in its traced order
    (`peak_bytes`): the residuals still live, the backward's cotangents and the gradient as it is
    built. None where `residuals` is."""
    """The prepared state's logical bytes by field and device, before buffer assignment aliases
    anything. These are not additional temporary bytes: the state's existing allocations already
    count against allocator headroom, and newly built copies and accumulators are in `held`."""

    def record(self) -> dict[str, JSON]:
        """The terms traced, not an admission decision: XLA's fit check remains authoritative."""
        return {"residuals": self.residuals, "gradients": self.gradients, "held": self.held,
                "resident": dict(self.resident), "peak": self.peak}


def device_bytes(tree) -> int:
    """The placed leaves in device memory alone: an offloaded optimizer or EMA is host storage,
    not resident HBM. Each device leaf carries the sharding that `placed_bytes` reads."""
    return placed_bytes([leaf for leaf in jax.tree.leaves(tree)
                         if jax.typeof(leaf).memory_space == jax.memory.Space.Device])


CHEAP = frozenset({
    "add", "and", "bitcast_convert_type", "broadcast_in_dim", "ceil", "clamp", "clz", "eq", "ne", "lt", "le",
    "gt", "ge", "complex", "concatenate", "convert_element_type", "copy", "copy_p", "dynamic_slice",
    "dynamic_update_slice", "floor", "imag", "iota", "is_finite", "max", "min", "mul", "neg", "not", "or",
    "xor", "pad", "population_count", "real", "reduce_precision", "reshape", "squeeze", "expand_dims", "rev",
    "round", "select_n", "shift_left", "shift_right_arithmetic", "shift_right_logical", "slice", "sub",
    "transpose", "integer_pow", "square", "name"})
"""The primitives XLA's CPU fusion takes for cheap, which it duplicates into each consumer rather than
keep a buffer of (`InstructionFusion::IsExpensive`, xla/service/instruction_fusion.cc: every elementwise
op but division, the transcendentals and their kin, and the layout ops), an integer power and a square
lowering to its multiplies, and a checkpoint's name, which computes nothing."""

REAL_CHEAP = frozenset({"abs", "sign", "sin", "cos", "tan", "sinh"})
"""The primitives XLA's CPU fusion takes for cheap over real elements alone (`IsExpensive`)."""

ELEMENTWISE = CHEAP | REAL_CHEAP | frozenset({
    "div", "rem", "exp", "exp2", "log", "log1p", "expm1", "tanh", "logistic", "erf", "erfc", "erf_inv",
    "pow", "sqrt", "rsqrt", "cbrt", "atan2", "asin", "acos", "atan", "cosh", "asinh", "acosh", "atanh"})
"""The elementwise and layout primitives, every one of which XLA's GPU fusion recomputes in each consumer
rather than keep a buffer of: its priority fusion fuses a producer into all its consumers where its cost
model finds the memory traffic saved worth the recompute (xla/backends/gpu/transforms/priority_fusion.cc),
an elementwise one's always on a memory-bound GPU, a transcendental's too (an attention's exponentials and
probabilities, on an H100's compile, recomputed from its scores in the backward)."""


def residual_bytes(loss: Callable, params, arguments: Sequence, *, platform: str) -> int:
    """The device bytes the pullback of `loss(params, *arguments)` in `params` keeps, a device's share,
    as XLA's fusion for `platform` keeps them: each value its traced program (`jax.make_jaxpr` of
    `jax.vjp`'s pullback) closes over counts the values it is computed from without a buffer (`kept`),
    once, but those the program reads, which the state and the batch already hold, and literals; a value
    in pinned host memory counts none. A value the pullback closes over keeps its own traced sharding;
    one whose trace states none, and every other value, count whole. Matching an axis's size to the
    batch size cannot identify its sharding: a weight or a scan's layer axis can have that size too."""
    def pullback(params, *arguments):
        return jax.vjp(lambda trainable: loss(trainable, *arguments), params, has_aux=True)[1]

    closed, shapes = jax.make_jaxpr(pullback, return_shape=True)(params, *arguments)
    program = closed.jaxpr
    placed = {value: shape for value, shape in zip(program.outvars, jax.tree.leaves(shapes), strict=True)
              if not isinstance(value, jcore.Literal)}
    total = 0
    for value in kept(program, platform):
        aval = value.aval
        if not isinstance(aval, ShapedArray):
            raise TypeError(f"a pullback keeps {aval}, which is no array")
        if aval.memory_space != jax.memory.Space.Device:
            continue
        shape = placed.get(value)
        total += (math.prod(aval.shape) * aval.dtype.itemsize if shape is None or shape.sharding is None
                  else placed_bytes(shape))
    return total


def gradient_peak(loss: Callable, params, arguments: Sequence, *, platform: str) -> int:
    """The most device bytes the gradient of `loss(params, *arguments)` in `params` holds at once, a
    device's share, as XLA's fusion for `platform` keeps its values (`peak_bytes`): the program
    `jax.vjp`'s pullback runs on a cotangent of ones, traced whole, forward and backward."""
    def gradient(params, *arguments):
        outputs, pullback, _ = jax.vjp(lambda trainable: loss(trainable, *arguments), params, has_aux=True)
        return pullback(jax.tree.map(jax.numpy.ones_like, outputs))

    return peak_bytes(jax.make_jaxpr(gradient)(params, *arguments).jaxpr, platform)


def peak_bytes(program: jcore.Jaxpr, platform: str) -> int:
    """The most bytes `program` holds at once beside its inputs, its equations run in their traced
    order: each value XLA keeps a buffer of (`kept`'s rule, `recomputed`) from the equation that
    computes it to the last that reads it, the outputs to the end, and a call's, a loop's or a
    checkpoint's own peak beside what is live where it runs. A value in pinned host memory counts none.
    No schedule XLA picks is modelled: a compile confirms."""
    producers = {out: equation for equation in program.eqns for out in equation.outvars}
    inputs = {*program.invars, *program.constvars}
    reads: dict[jcore.Var, frozenset[jcore.Var]] = {}

    def anchors(value) -> frozenset[jcore.Var]:
        if isinstance(value, jcore.Literal) or value in inputs:
            return frozenset()
        if value not in reads:
            equation = producers.get(value)
            if equation is not None and recomputed(equation, platform):
                reads[value] = frozenset().union(*map(anchors, equation.invars))
            else:
                reads[value] = frozenset([value])
        return reads[value]

    last = {anchor: index for index, equation in enumerate(program.eqns)
            for value in equation.invars for anchor in anchors(value)}
    last.update({anchor: len(program.eqns) for value in program.outvars for anchor in anchors(value)})
    live, held, most = 0, set(), 0
    for index, equation in enumerate(program.eqns):
        inner = 0
        if not recomputed(equation, platform):
            inner = max((peak_bytes(body, platform) for body in bodies(equation)), default=0)
            for out in equation.outvars:
                if out in last:
                    held.add(out)
                    live += device_share(out)
        most = max(most, live + inner)
        for done in [value for value in held if last[value] == index]:
            held.discard(done)
            live -= device_share(done)
    return most


def device_share(value: jcore.Var) -> int:
    """The device bytes `value` holds: none for one in pinned host memory or no array (a token)."""
    aval = value.aval
    if not isinstance(aval, ShapedArray) or aval.memory_space != jax.memory.Space.Device:
        return 0
    return math.prod(aval.shape) * aval.dtype.itemsize


def bodies(equation: jcore.JaxprEqn) -> list[jcore.Jaxpr]:
    """The programs `equation` runs within it: a call's, a loop's, a branch's or a checkpoint's."""
    found = []
    for value in equation.params.values():
        for each in value if isinstance(value, tuple | list) else (value,):
            body = each.jaxpr if isinstance(each, jcore.ClosedJaxpr) else each
            if isinstance(body, jcore.Jaxpr):
                found.append(body)
    return found


def kept(program: jcore.Jaxpr, platform: str) -> set[jcore.Var]:
    """The values `program`'s outputs are computed from that XLA keeps a buffer of on `platform`: each
    output, but one a primitive XLA's fusion recomputes in its consumers computes (`recomputed`), whose
    reads are kept in its place; neither the program's inputs nor its literals."""
    producers = {out: equation for equation in program.eqns for out in equation.outvars}
    inputs, found, seen = {*program.invars, *program.constvars}, set(), set()
    pending = [value for value in program.outvars if not isinstance(value, jcore.Literal)]
    while pending:
        value = pending.pop()
        if value in seen or value in inputs:
            continue
        seen.add(value)
        equation = producers.get(value)
        if equation is not None and recomputed(equation, platform):
            pending.extend(read for read in equation.invars if not isinstance(read, jcore.Literal))
        else:
            found.add(value)
    return found


def recomputed(equation: jcore.JaxprEqn, platform: str) -> bool:
    """Whether XLA's fusion on `platform` recomputes what `equation` computes in each consumer rather than
    keep it: on a GPU every elementwise primitive (`ELEMENTWISE`), elsewhere the cheap ones (`CHEAP`, and
    `REAL_CHEAP` over real values); a call, or a custom derivative's primal, of nothing else."""
    name = equation.primitive.name
    if name in ("pjit", "jit", "custom_jvp_call"):
        inner = equation.params["jaxpr" if name != "custom_jvp_call" else "call_jaxpr"].jaxpr
        return all(recomputed(each, platform) for each in inner.eqns)
    if platform == "gpu":
        return name in ELEMENTWISE
    if name in REAL_CHEAP:
        return not any(isinstance(out.aval, ShapedArray) and np.issubdtype(out.aval.dtype, np.complexfloating)
                       for out in equation.outvars)
    return name in CHEAP


def traced_loss[LossT, EffectsT](objective: Objective[LossT, EffectsT], state: TrainState, window: int
                                 ) -> tuple[Callable, tuple]:
    """The loss the step evaluates for `objective` over `state`, with the abstract arguments it is
    traced over (but the batch): the variables, the microstep, the key, the step and the EMA. Shapes
    and dtypes only: a resident frozen leaf sits in another memory space than the moving ones, and
    the loss the realization runs reads a snapshot in one space. The step's key is drawn inside the
    trace, so an abstract state compiles a step as a placed one does."""
    variables = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state.variables)

    def loss(variables, batch, microstep, key, step, ema):
        return objective._loss(variables, batch, Step(
            microstep, jax.random.fold_in(key, step), with_ema(variables, ema),
            qk_stats=clips_qk(state.opt_state), example_count=window * rows_of(batch) or None))

    return loss, (variables, state.microstep, state.key, state.step, state.ema)


def candidate_footprint[LossT, EffectsT](objective: Objective[LossT, EffectsT],
                                          micro: Batch, window: int, prepared: TrainState,
                                          held: int, platform: str) -> Footprint:
    """What a step over microbatches of `micro`'s shapes, its state `prepared` as placed, takes of a
    device of `platform` beside that state (`Footprint`): its loss's residuals and its gradient
    program's peak at one microbatch, as that platform's compiler keeps them, its trained parameters'
    gradient, laid out as they are, and `held`. Over several devices that split the
    microbatch's rows alone (data and fully sharded data parallelism, `row_axes`) the residuals are one
    device's: the loss traced as one device's step, over a mesh of the run's axes each of one device,
    at that device's share of the microbatch (each leaf's shard, `shard_shape`), its parameters whole as
    those layouts gather them. A layout that splits more than the rows (stages, tensor, sequence)
    derives no residuals, which its compile judges."""
    mesh = jax.sharding.get_abstract_mesh()
    split = set() if mesh.empty else {name for name, size in mesh.shape.items() if size > 1}
    # Data and fully sharded data parallelism split the rows alone; an expert axis splits the experts too.
    rows = set(row_axes(rows_of(micro), mesh=mesh)) & {DATA_AXIS, FSDP_AXIS} if split else set()
    if not split <= rows:
        residuals = peak = None
    else:
        loss, (variables, *rest) = traced_loss(objective, prepared, window)
        variables = forward_variables(variables, prepared.compute)

        def trained(params, variables, batch, *rest):
            return loss({**variables, "params": params}, batch, *rest)

        share = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(
            leaf.shape if getattr(leaf, "sharding", None) is None else leaf.sharding.shard_shape(leaf.shape),
            leaf.dtype), micro)
        # The arguments' shapes alone, as one device holds them: no sharding of the run's mesh.
        arguments = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype),
                                 (variables, share, *rest))
        # The run's own first device, which a kernel's choice reads as the run's do.
        placed = jax.sharding.get_mesh()
        device = None if mesh.empty or placed.empty else placed.devices.flat[0]
        with contextlib.nullcontext() if device is None else jax.set_mesh(jax.sharding.Mesh(
                np.asarray([device]).reshape((1,) * len(mesh.axis_names)), mesh.axis_names,
                axis_types=mesh.axis_types)):
            residuals = residual_bytes(trained, arguments[0]["params"], arguments, platform=platform)
            peak = gradient_peak(trained, arguments[0]["params"], arguments, platform=platform)
    resident = {name: device_bytes(value) for name, value in {
        "variables": prepared.variables, "optimizer": prepared.opt_state, "ema": prepared.ema,
        "accumulation": prepared.accumulation, "compute": prepared.compute}.items()}
    return Footprint(residuals, device_bytes(prepared.variables["params"]), held, resident, peak)
