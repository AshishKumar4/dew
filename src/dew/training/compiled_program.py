"""The program XLA compiles of a traced step, as the derivation of its peak reads it (row 2113).

A jaxpr is what JAX traced; XLA compiles what JAX lowers it to. Lowered for one platform, a one-device
`shard_map` and a `platform_dependent` conditional leave no trace of their own (`compiled`), and XLA
fuses a cuBLAS matmul with a slice of its loop's buffer it reads or writes (`Slicing`), which keeps the
workspace the GEMM rewriter gave it. Each rule names the JAX or XLA source it follows; XLA's is its
commit d867969800d4a8fd045505eabc65e1e66f30794f, which jax 0.12.0.1 pins.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.extend.core as jcore
import numpy as np
from jax.core import ShapedArray

from dew.nn.programs import bodies, called, computed, inlined, named, opaque, platform_branch

VIEWS = frozenset({"reshape", "squeeze", "expand_dims"})
"""Primitives a row-major value's layout keeps as a bitcast of their operand (XLA's `ReshapeIsBitcast`
under default layouts)."""


def bitcast(equation: jcore.JaxprEqn) -> bool:
    """Whether `equation`'s result is a bitcast of its one operand under a row-major layout: a shape
    primitive's (`VIEWS`), a stack of one value (a new leading axis of one), or an unstack along an axis of
    one (that axis dropped), which XLA lowers to a reshape (`lax.stack`, `lax.unstack`): a scanned head's
    loop reads its weight gradient's carry by unstacking it and writes it back by stacking, each a view of
    the carry's buffer, which the derivation counted twice over (Qwen3.5-4B's fused tiled head, 2 x 2.37
    GiB, row 4089)."""
    name = named(equation.primitive)
    if name in VIEWS:
        return True
    if name == "stack":
        return len(equation.invars) == 1
    if name == "unstack":
        return len(equation.outvars) == 1
    return False

UNEVALUATED = frozenset({"pure_callback", "io_callback", "debug_callback", "ffi_call", "pallas_call",
                         "axis_index"})
"""Primitives XLA's evaluator computes no offset through: those JAX lowers to a custom call, and the
device's index, a partition or replica id (`GetLeafDependencies`, dynamic_slice_analysis.cc)."""

ALIGNMENT = 256
"""The alignment of every buffer XLA allocates on a GPU (`kXlaAllocatedBufferAlignBytes`,
xla/service/gpu/gpu_constants.h), which a slice fused into a matmul's call must keep (`IsAlignedSlice`,
xla/backends/gpu/transforms/dynamic_slice_fusion_rewriter_v2.cc)."""


def lowering_platform(device) -> str:
    """The platform JAX lowers a program on `device` for, its lowering rules' name: a GPU's `cuda` or
    `rocm`, as its client's platform version states (`PjRtClient::platform_version`, a compile-only
    target's its device kind, CUDA's); any other device's own platform."""
    if device.platform != "gpu":
        return device.platform
    return "rocm" if "rocm" in str(device.client.platform_version).lower() else "cuda"


def deviceless(device) -> bool:
    """Whether `device` is a compile-only target, no device of a running backend: one of a topology JAX
    compiles for without a device (`compile_only_runtime`, `jax.experimental.topologies`)."""
    return device.client.runtime_type == "compile_only_runtime"


def compiled(program: jcore.Jaxpr, lowering: str,
             made: dict[int, tuple[jcore.Jaxpr, jcore.Jaxpr]] | None = None) -> jcore.Jaxpr:
    """`program` as JAX lowers it for the platform `lowering` (`lowering_platform`) and XLA compiles it:
    each `shard_map` over manual axes of one device inline, its body's equations among the rest
    (`_shard_map_lowering_shardy`, jax/_src/shard_map.py, makes no manual computation of it), each
    `platform_dependent` conditional the branch `lowering` keeps, inline, no conditional around it
    (`_cond_lowering`, jax/_src/lax/control_flow/conditionals.py, keeps the branches whose platforms
    hold the lowering's, and lowers a sole branch as the computation itself), each scan of one turn
    inline (`once`), and each call (`CALLS`)
    as its program in its caller, a differentiated checkpoint's operands behind its barrier
    (`prevented`), a nested `jit` among them (a wide sum's, `precision.accumulated`, traced once for its
    types: called, its GEMMs were no fusion's of the slices that read them, row 3193); through every program
    an equation runs but a native Pallas kernel's, which XLA never sees (`opaque`)."""
    # Each program once a call, by its object (`made`): one body traced once is held by every equation that
    # runs it (a layer's checkpoint, a scan's), each inlining it afresh (`inlined`); compiled for each,
    # a Qwen3.5-4B step's gradient compiled 90,259 programs (row 4261).
    made = {} if made is None else made
    if (held := made.get(id(program))) is not None and held[0] is program:
        return held[1]
    renamed: dict[jcore.Var, jcore.Var | jcore.Literal] = {}
    equations: list[jcore.JaxprEqn] = []

    def read(value):
        return renamed.get(value, value) if computed(value) else value

    for equation in program.eqns:
        name, params = named(equation.primitive), dict(equation.params)
        operands = [read(value) for value in equation.invars]
        body, consts, reading = None, (), operands
        if name == "shard_map" and math.prod(params["mesh"].shape[axis]
                                             for axis in params["newly_manual_axes"]) == 1:
            body = params["jaxpr"]
        elif name == "pallas_call" and params["interpret"]:
            run = functools.partial(interpreter(params.pop("interpret")), **params)
            traced = jax.make_jaxpr(run)(*(value.aval for value in equation.invars))
            body, consts = traced.jaxpr, traced.consts
        elif name == "cond" and params.get("branches_platforms") is not None:
            branch = params["branches"][platform_branch(params["branches_platforms"], lowering)]
            body, consts, reading = branch.jaxpr, branch.consts, operands[1:]
        elif name == "scan" and params["length"] == 1 and not params["jaxpr"].jaxpr.effects:
            inner, returned = once(equation, operands, lowering, made)
            equations.extend(inner)
            renamed.update(zip(equation.outvars, returned, strict=True))
            continue
        elif (program_called := called(equation)) is not None:
            body, consts = program_called, tuple(program_called.consts)
            if params.get("differentiated"):
                barrier, reading = prevented(operands, params.get("prevent_cse", False))
                equations.extend(barrier)
        if body is not None:
            inner, returned = inlined(compiled(body, lowering, made), reading, consts)
            equations.extend(inner)
            renamed.update(zip(equation.outvars, returned, strict=True))
            continue
        if not opaque(equation):
            params = {key: within(value, lowering, made) for key, value in params.items()}
        equations.append(equation.replace(invars=operands, params=params))
    lowered = program.replace(eqns=equations, outvars=[read(value) for value in program.outvars])
    made[id(program)] = (program, lowered)
    return lowered


def once(equation: jcore.JaxprEqn, operands: Sequence[jcore.Var | jcore.Literal], lowering: str,
         made: dict[int, tuple[jcore.Jaxpr, jcore.Jaxpr]]
         ) -> tuple[list[jcore.JaxprEqn], list[jcore.Var | jcore.Literal]]:
    """The equations of `equation`, a scan of one turn whose body has no effect, run inline over
    `operands`, and its results: XLA removes a loop of one trip, its body a call it inlines where none of
    its computations has a side effect (`TryRemoveWhileLoop`, xla/service/while_loop_simplifier.cc), so
    what the loop read from memory its body's own equations fuse. Each stacked operand is read as its one
    slice and each stacked result is that slice's stack, each a bitcast (`VIEWS`). Qwen3.5-4B's fused
    tiled head at four microbatches, its rows one tile, was derived holding its zero carry and its cast
    weights beside the loop: 14.4 GiB of temporaries against XLA's 8.1 (row 4089)."""
    params = equation.params
    _, _, stacked = params["ft_in"].update(range(len(operands))).unpack()
    unstacked, sliced = reshaped([operands[at] for at in stacked], lambda value: jax.lax.squeeze(value, (0,)))
    reading = list(operands)
    for at, value in zip(stacked, sliced, strict=True):
        reading[at] = value
    body = params["jaxpr"]
    inner, returned = inlined(compiled(body.jaxpr, lowering, made), reading, body.consts)
    _, turned = params["ft_out"].update(range(len(returned))).unpack()
    restacked, stacks = reshaped([returned[at] for at in turned],
                                 lambda value: jax.lax.reshape(value, (1, *value.shape)))
    for at, value in zip(turned, stacks, strict=True):
        returned[at] = value
    return [*unstacked, *inner, *restacked], returned


def reshaped(values: Sequence[jcore.Var | jcore.Literal], reshape: Callable
             ) -> tuple[list[jcore.JaxprEqn], list[jcore.Var | jcore.Literal]]:
    """The equations that `reshape` each of `values`, and their results."""
    traced = jax.make_jaxpr(lambda *each: [reshape(value) for value in each])(
        *(value.aval for value in values))
    return inlined(traced.jaxpr, values, traced.consts)


def prevented(operands: Sequence[jcore.Var | jcore.Literal], prevent_cse: bool | Sequence[bool]
              ) -> tuple[list[jcore.JaxprEqn], list[jcore.Var | jcore.Literal]]:
    """`operands` as a differentiated checkpoint reads them, and the equations that make them so: those
    `prevent_cse` selects (each, where it is one flag) through one optimization barrier, which XLA keeps
    them in buffers behind, the rest as they are (`_remat_lowering`, jax/_src/ad_checkpoint.py)."""
    chosen = [prevent_cse] * len(operands) if isinstance(prevent_cse, bool) else list(prevent_cse)
    held = [value for value, kept in zip(operands, chosen, strict=True) if kept]
    if not held:
        return [], list(operands)
    traced = jax.make_jaxpr(lambda *values: jax.lax.optimization_barrier(values))(
        *(value.aval for value in held))
    barrier, kept_apart = inlined(traced.jaxpr, held, traced.consts)
    returned = iter(kept_apart)
    return barrier, [next(returned) if kept else value for value, kept in zip(operands, chosen, strict=True)]


def within(value, lowering: str, made: dict[int, tuple[jcore.Jaxpr, jcore.Jaxpr]] | None = None):
    """`value`, an equation's parameter, with each program in it as `compiled` makes it: a plain
    sequence of programs (a conditional's branches) item by item; a named tuple (a gather's dimension
    numbers, a key's implementation) holds none and stays as it is. A program carries its own constants
    (on JAX 0.12 a closed program is a `Jaxpr`), which `compiled` keeps."""
    if isinstance(value, jcore.Jaxpr):
        return compiled(value, lowering, made)
    if type(value) in (tuple, list):
        return type(value)(within(each, lowering, made) for each in value)
    return value


def interpreter(interpret) -> Callable:
    """The function JAX lowers a Pallas call interpreted under `interpret` as (`_pallas_call_lowering`,
    jax/_src/pallas/pallas_call.py): Mosaic's TPU or GPU interpreter for their parameters, and for any
    other the HLO interpreter, the kernel discharged of its state run over its grid in a loop carrying
    its whole outputs (`pallas_call_hlo_interpret`, jax/_src/pallas/hlo_interpreter.py)."""
    from jax._src.pallas import hlo_interpreter
    from jax._src.pallas.mosaic.interpret.interpret_pallas_call import interpret_pallas_call as on_tpu
    from jax._src.pallas.mosaic_gpu.interpret.interpret_pallas_call import interpret_pallas_call as on_gpu
    from jax.experimental.pallas import mosaic_gpu, tpu

    if isinstance(interpret, tpu.InterpretParams):
        return functools.partial(on_tpu, interpret_params=interpret)
    if isinstance(interpret, mosaic_gpu.InterpretGPUParams):
        return functools.partial(on_gpu, interpret_params=interpret)
    return hlo_interpreter.pallas_call_hlo_interpret


def gradients_read(program: jcore.Jaxpr, read: Sequence[frozenset[int]]) -> list[frozenset[int]]:
    """Which gradient leaves each result of `program` reads, its operands reading `read`: a call's
    (`called`, an optax tree helper's `jit` among them) through its program, any other
    equation each of its results all its operands read."""
    reads = dict(zip(program.invars, read, strict=True))
    for equation in program.eqns:
        operands = [reads.get(value, frozenset()) if computed(value) else frozenset()
                    for value in equation.invars]
        program_called = called(equation)
        if program_called is not None:
            reads.update(zip(equation.outvars, gradients_read(program_called, operands), strict=True))
        else:
            reads.update(dict.fromkeys(equation.outvars, frozenset().union(*operands)))
    return [reads.get(value, frozenset()) if computed(value) else frozenset() for value in program.outvars]


def evaluable(equation: jcore.JaxprEqn) -> bool:
    """Whether XLA's evaluator computes `equation` to find an offset: no side effect, no custom call and
    no device index, in it or any program it runs (`UNEVALUATED`)."""
    return (not equation.effects and named(equation.primitive) not in UNEVALUATED
            and all(evaluable(inner) for body in bodies(equation) for inner in body.eqns))


def byte_strides(shape: Sequence[int], itemsize: int) -> list[int]:
    """The byte stride of each axis of a row-major array of `shape` (XLA's default layout,
    `ShapeUtil::ByteStrides`)."""
    return [itemsize * math.prod(shape[axis + 1:]) for axis in range(len(shape))]


def contiguous_aligned(whole: Sequence[int], part: Sequence[int], itemsize: int) -> bool:
    """Whether a `part` of a row-major array of shape `whole` is one contiguous run of its bytes keeping
    their alignment: walked minor to major, every axis more major than the most minor one it cuts is of
    one (`IsContiguousSlice`, xla/service/gpu/ir_emission_utils.cc), and its bytes a multiple of
    `ALIGNMENT` (`IsAlignedSlice`)."""
    if len(whole) != len(part) or not whole:
        return False
    cut = max((axis for axis, (full, size) in enumerate(zip(whole, part, strict=True)) if size < full),
              default=None)
    contiguous = cut is None or all(size == 1 for size in part[:cut])
    return contiguous and math.prod(part) * itemsize % ALIGNMENT == 0


@dataclasses.dataclass(frozen=True)
class Slicing:
    """The slices of a loop's buffers a program the loop runs reads or writes that XLA fuses with the
    cuBLAS matmul reading or writing them (`DynamicSliceFusionRewriterV2`: a matmul whose operand is,
    through bitcasts alone, an aligned contiguous static slice, or a dynamic one `DynamicSliceAnnotator`
    annotated, or whose result is, through single-use bitcasts, the update of such a dynamic slice the
    computation returns; `AnalyzeDynamicSlice`, dynamic_slice_analysis.cc, annotates a contiguous slice
    whose byte offsets over the loop's turns, evaluated from its induction variable, form a linear
    progression, table offsets off by default, xla/debug_options_flags.cc). A scan's turn of its stacked
    inputs and its stacked outputs' updates are such slices (`sliced`, `updated`), and so are those its
    counter (`counter`, its value each turn `turns`) offsets."""

    counter: jcore.Var | None = None
    turns: tuple[int, ...] = ()
    sliced: frozenset = frozenset()
    updated: frozenset = frozenset()

    def within(self, program: jcore.Jaxpr) -> Slicing:
        """This slicing completed over `program`'s equations."""
        readers = _readers(program)
        producers = {out: equation for equation in program.eqns for out in equation.outvars}
        sliced, updated = set(self.sliced), set(self.updated)
        counting: list[jcore.JaxprEqn] = []
        counted = {self.counter} if self.counter is not None else set()
        for equation in program.eqns:
            name = named(equation.primitive)
            reads = [value for value in equation.invars if computed(value)]
            if name == "slice" and _static_slice(equation):
                sliced.update(equation.outvars)
            elif name in ("dynamic_slice", "dynamic_update_slice") and self._annotated(equation, counting):
                if name == "dynamic_slice":
                    sliced.update(equation.outvars)
                elif equation.outvars[0] in program.outvars and computed(equation.invars[1]):
                    updated.update(_bitcast_sources(equation.invars[1], producers, readers))
            elif bitcast(equation) and reads and reads[0] in sliced:
                sliced.update(equation.outvars)
            elif (reads and all(read in counted for read in reads) and name not in ("scan", "while", "cond")
                  and evaluable(equation)):
                counted.update(equation.outvars)
                counting.append(equation)
        # A transpose a matmul reads is folded into its dimensions (`TransposeFolding`,
        # xla/service/transpose_folding.cc, which folds a dot's operand that is itself a transpose).
        folded = {out for equation in program.eqns if named(equation.primitive) == "transpose"
                  and computed(equation.invars[0]) and equation.invars[0] in sliced
                  for out in equation.outvars
                  if all(named(reader.primitive) == "dot_general" for reader in readers.get(out, ()))}
        return dataclasses.replace(self, sliced=frozenset(sliced | folded), updated=frozenset(updated))

    def _annotated(self, equation: jcore.JaxprEqn, counting: Sequence[jcore.JaxprEqn]) -> bool:
        """Whether `DynamicSliceAnnotator` annotates `equation`, a dynamic slice or update: contiguous
        and aligned, its offsets literals or computed from the loop's counter alone, whose byte offsets
        over the loop's turns, clamped as the slice clamps them or not, step evenly."""
        whole = dimensions(equation.invars[0])
        updating = named(equation.primitive) == "dynamic_update_slice"
        part = dimensions(equation.invars[1] if updating else equation.outvars[0])
        itemsize = _itemsize(equation.invars[0])
        if not contiguous_aligned(whole, part, itemsize):
            return False
        offsets = list(equation.invars[2 if updating else 1:])
        produced = {out for each in counting for out in each.outvars}
        if any(computed(offset) and offset is not self.counter and offset not in produced
               for offset in offsets):
            return False
        strides = byte_strides(whole, itemsize)
        turns = self.turns if any(computed(offset) for offset in offsets) else (0,)
        clamped, unclamped = [], []
        for turn in turns:
            values = _evaluated(offsets, counting, self.counter, turn)
            limits = [full - size for full, size in zip(whole, part, strict=True)]
            unclamped.append(sum(value * stride for value, stride in zip(values, strides, strict=True)))
            clamped.append(sum(min(max(value, 0), limit) * stride
                               for value, limit, stride in zip(values, limits, strides, strict=True)))
        largest = max(0, math.prod(whole) * itemsize - math.prod(part) * itemsize)
        return _even(clamped) or (_even(unclamped) and all(
            min(max(offset, 0), largest) == held for offset, held in zip(unclamped, clamped, strict=True)))

    def fused(self, equation: jcore.JaxprEqn) -> bool:
        """Whether XLA fuses `equation`, a cuBLAS matmul, with a slice of its loop's buffers."""
        return (any(computed(operand) and operand in self.sliced for operand in equation.invars)
                or any(out in self.updated for out in equation.outvars))

    @staticmethod
    def entering(equation: jcore.JaxprEqn, body: jcore.Jaxpr) -> Slicing:
        """The slicing `body`, a program `equation` runs, reads of its loop: for a scan, each turn's slice
        of its stacked inputs, the update of each stacked output into the buffer the loop returns, and
        its first carry where its turns count it by one from a literal (the counter a `fori_loop` over
        fixed bounds carries), over its turns; nothing for any other program."""
        if named(equation.primitive) != "scan":
            return Slicing()
        _, carried, stacked = equation.params["ft_in"].update(body.invars).unpack()
        _, started, _ = equation.params["ft_in"].update(equation.invars).unpack()
        returned, outputs = equation.params["ft_out"].update(body.outvars).unpack()
        readers = _readers(body)
        producers = {out: each for each in body.eqns for out in each.outvars}
        sliced = {value for value in stacked if _row_aligned(value)}
        updated = {source for value in outputs if computed(value) and _row_aligned(value)
                   for source in _bitcast_sources(value, producers, readers)}
        counter, turns = None, ()
        counting = carried and returned and carried[0].aval.dtype == np.int32
        if counting and isinstance(started[0], jcore.Literal):
            stepped = any(step.outvars == [returned[0]] and named(step.primitive) == "add"
                          and step.invars[0] is carried[0] and isinstance(step.invars[1], jcore.Literal)
                          and step.invars[1].val == 1 for step in body.eqns)
            if stepped:
                first = int(started[0].val)
                counter, turns = carried[0], tuple(range(first, first + int(equation.params["length"])))
        return Slicing(counter, turns, frozenset(sliced), frozenset(updated))


def _readers(program: jcore.Jaxpr) -> Mapping[jcore.Var, list[jcore.JaxprEqn]]:
    """The equations reading each value of `program`, each once; a value the program returns is read
    by it too, as the (absent) equation `None` stands for."""
    found: dict[jcore.Var, list] = {}
    for equation in program.eqns:
        for value in {value for value in equation.invars if computed(value)}:
            found.setdefault(value, []).append(equation)
    for value in program.outvars:
        if computed(value):
            found.setdefault(value, []).append(_RETURNED)
    return found


class _Returned:
    """The reader that is a program's return, no primitive's."""

    class primitive:
        name = "return"


_RETURNED = _Returned()


def _bitcast_sources(value: jcore.Var, producers: Mapping, readers: Mapping) -> set[jcore.Var]:
    """`value`, read once, and each value it is a bitcast (`VIEWS`) of read once, back to the matmul
    that made it (`ResolveNonTupleSlicedResult`: the matmul's sole user; `ResolveSlicedResult`'s no-op
    path: each no-op's sole user); none where `value` is read more than once."""
    if len(readers.get(value, ())) != 1:
        return set()
    found = {value}
    while (equation := producers.get(value)) is not None and bitcast(equation) \
            and computed(equation.invars[0]) and len(readers.get(equation.invars[0], ())) == 1:
        value = equation.invars[0]
        found.add(value)
    return found


def _static_slice(equation: jcore.JaxprEqn) -> bool:
    """Whether `equation`, a `slice`, takes a contiguous aligned run of its operand at unit strides
    (`ResolveSlicedParameter`'s static slice, which `IsAlignedSlice` admits)."""
    strides = equation.params.get("strides")
    if strides is not None and any(stride != 1 for stride in strides):
        return False
    return contiguous_aligned(dimensions(equation.invars[0]), dimensions(equation.outvars[0]),
                              _itemsize(equation.invars[0]))


def _row_aligned(value: jcore.Var) -> bool:
    """Whether a turn's row `value` of a scan's stacked buffer keeps its alignment: the turn's slice is
    the stacked buffer's whole minor axes, contiguous by construction."""
    return math.prod(dimensions(value)) * _itemsize(value) % ALIGNMENT == 0


def dimensions(value) -> tuple[int, ...]:
    """`value`'s shape, none where it is no array."""
    aval = value.aval
    return tuple(int(size) for size in aval.shape) if isinstance(aval, ShapedArray) else ()


def _itemsize(value) -> int:
    """The bytes of one of `value`'s elements, none where it is no array."""
    aval = value.aval
    return aval.dtype.itemsize if isinstance(aval, ShapedArray) else 0


def _evaluated(offsets: Sequence, counting: Sequence[jcore.JaxprEqn], counter, turn: int) -> list[int]:
    """`offsets`' values at the loop's turn whose counter is `turn`, each computed from it by those of
    `counting`, the equations computing values from the counter and literals alone, it reads, as XLA's
    evaluator computes them at each turn (`EvaluateByteOffsetAtIteration`)."""
    values: dict = {} if counter is None else {counter: np.int32(turn)}

    def read(value):
        return values[value] if computed(value) else value.val

    needed = {offset for offset in offsets if computed(offset)}
    reading = []
    for equation in reversed(counting):
        if needed.intersection(equation.outvars):
            reading.append(equation)
            needed.update(value for value in equation.invars if computed(value))
    # The evaluator computes on the host, outside any mesh a plan for compile-only devices sets.
    with jax.set_mesh(None), jax.ensure_compile_time_eval():
        for equation in reversed(reading):
            operands = list(map(read, equation.invars))
            if named(equation.primitive) == "dot_general" and len(operands) == 3:
                # A product with its bias, as the derivation reads a sum XLA's GEMM computes
                # (`footprint.biased`): the product of the first two, plus the third.
                *operands, bias = operands
                out = equation.primitive.bind(*operands, **equation.params) + bias
            else:
                out = equation.primitive.bind(*operands, **equation.params)
            values.update(zip(equation.outvars, out if equation.primitive.multiple_results else [out],
                              strict=True))
    return [int(read(offset)) for offset in offsets]


def _even(offsets: Sequence[int]) -> bool:
    """Whether `offsets` step by one stride throughout (`get_linear_stride`)."""
    return len({later - earlier for earlier, later in itertools.pairwise(offsets)}) <= 1
