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
from typing import TypeGuard

import jax
import jax.extend.core as jcore
import numpy as np
from jax.core import ShapedArray

VIEWS = frozenset({"reshape", "squeeze", "expand_dims"})
"""Primitives a row-major value's layout keeps as a bitcast of their operand (XLA's `ReshapeIsBitcast`
under default layouts)."""

UNEVALUATED = frozenset({"pure_callback", "io_callback", "debug_callback", "ffi_call", "pallas_call",
                         "axis_index"})
"""Primitives XLA's evaluator computes no offset through: those JAX lowers to a custom call, and the
device's index, a partition or replica id (`GetLeafDependencies`, dynamic_slice_analysis.cc)."""

ALIGNMENT = 256
"""The alignment of every buffer XLA allocates on a GPU (`kXlaAllocatedBufferAlignBytes`,
xla/service/gpu/gpu_constants.h), which a slice fused into a matmul's call must keep (`IsAlignedSlice`,
xla/backends/gpu/transforms/dynamic_slice_fusion_rewriter_v2.cc)."""


def computed(value) -> TypeGuard[jcore.Var]:
    """Whether `value` is a program's variable, not a literal it reads."""
    return isinstance(value, jcore.Var) and not isinstance(value, jcore.Literal)


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


def inlined(body: jcore.Jaxpr, operands: Sequence[jcore.Var | jcore.Literal],
            consts: Sequence = ()) -> tuple[list[jcore.JaxprEqn], list[jcore.Var | jcore.Literal]]:
    """`body`'s equations, each value its own fresh variable, read over `operands` (and its constants
    over `consts`, as literals), and its results: a program run in its caller as one of its own, as one
    body traced once may be run more than once."""
    local: dict[jcore.Var, jcore.Var | jcore.Literal] = dict(zip(body.invars, operands, strict=True))
    local.update({var: jcore.Literal(value, var.aval)
                  for var, value in zip(body.constvars, consts, strict=True)})

    def read(value):
        return local.get(value, value) if computed(value) else value

    equations = []
    for equation in body.eqns:
        fresh = [out if isinstance(out, jcore.DropVar) else jcore.Var(out.aval) for out in equation.outvars]
        equations.append(equation.replace(invars=[read(value) for value in equation.invars], outvars=fresh))
        local.update(zip(equation.outvars, fresh, strict=True))
    return equations, [read(value) for value in body.outvars]


def compiled(program: jcore.Jaxpr, lowering: str) -> jcore.Jaxpr:
    """`program` as JAX lowers it for the platform `lowering` (`lowering_platform`) and XLA compiles it:
    each `shard_map` over manual axes of one device inline, its body's equations among the rest
    (`_shard_map_lowering_shardy`, jax/_src/shard_map.py, makes no manual computation of it), and each
    `platform_dependent` conditional the branch `lowering` keeps, inline, no conditional around it
    (`_cond_lowering`, jax/_src/lax/control_flow/conditionals.py, keeps the branches whose platforms
    hold the lowering's, and lowers a sole branch as the computation itself); through every program an
    equation runs but a native Pallas kernel's, which XLA never sees (`opaque`)."""
    renamed: dict[jcore.Var, jcore.Var | jcore.Literal] = {}
    equations: list[jcore.JaxprEqn] = []

    def read(value):
        return renamed.get(value, value) if computed(value) else value

    for equation in program.eqns:
        name, params = equation.primitive.name, dict(equation.params)
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
            kept = next(index for index, platforms in enumerate(params["branches_platforms"])
                        if platforms is None or lowering in platforms)
            branch = params["branches"][kept]
            body, consts, reading = branch.jaxpr, branch.consts, operands[1:]
        if body is not None:
            inner, returned = inlined(compiled(body, lowering), reading, consts)
            equations.extend(inner)
            renamed.update(zip(equation.outvars, returned, strict=True))
            continue
        if not opaque(equation):
            params = {key: within(value, lowering) for key, value in params.items()}
        equations.append(equation.replace(invars=operands, params=params))
    return program.replace(eqns=equations, outvars=[read(value) for value in program.outvars])


def within(value, lowering: str):
    """`value`, an equation's parameter, with each program in it as `compiled` makes it: a plain
    sequence of programs (a conditional's branches) item by item; a named tuple (a gather's dimension
    numbers, a key's implementation) holds none and stays as it is. A program carries its own constants
    (on JAX 0.12 a closed program is a `Jaxpr`), which `compiled` keeps."""
    if isinstance(value, jcore.Jaxpr):
        return compiled(value, lowering)
    if type(value) in (tuple, list):
        return type(value)(within(each, lowering) for each in value)
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


def opaque(equation: jcore.JaxprEqn) -> bool:
    """Whether XLA sees nothing of `equation`'s body: a native Pallas kernel's, one custom call whose
    results are its outputs alone, its body's values in registers and shared memory, its products
    Triton's (`pallas_call_lowering`, jax/_src/pallas/triton/pallas_call_registration.py). `compiled`
    leaves no interpreted one: it is its interpreter's program (`interpreter`)."""
    return equation.primitive.name == "pallas_call"


def bodies(equation: jcore.JaxprEqn) -> list[jcore.Jaxpr]:
    """The programs `equation` runs within it: a call's, a loop's, a branch's or a checkpoint's. A
    native Pallas kernel runs none XLA sees (`opaque`)."""
    if opaque(equation):
        return []
    found = []
    for value in equation.params.values():
        found.extend(each for each in (value if isinstance(value, tuple | list) else (value,))
                     if isinstance(each, jcore.Jaxpr))
    return found


def evaluable(equation: jcore.JaxprEqn) -> bool:
    """Whether XLA's evaluator computes `equation` to find an offset: no side effect, no custom call and
    no device index, in it or any program it runs (`UNEVALUATED`)."""
    return (not equation.effects and equation.primitive.name not in UNEVALUATED
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
            name = equation.primitive.name
            reads = [value for value in equation.invars if computed(value)]
            if name == "slice" and _static_slice(equation):
                sliced.update(equation.outvars)
            elif name in ("dynamic_slice", "dynamic_update_slice") and self._annotated(equation, counting):
                if name == "dynamic_slice":
                    sliced.update(equation.outvars)
                elif equation.outvars[0] in program.outvars and computed(equation.invars[1]):
                    updated.update(_bitcast_sources(equation.invars[1], producers, readers))
            elif name in VIEWS and reads and reads[0] in sliced:
                sliced.update(equation.outvars)
            elif (reads and all(read in counted for read in reads) and name not in ("scan", "while", "cond")
                  and evaluable(equation)):
                counted.update(equation.outvars)
                counting.append(equation)
        # A transpose a matmul reads is folded into its dimensions (`TransposeFolding`,
        # xla/service/transpose_folding.cc, which folds a dot's operand that is itself a transpose).
        folded = {out for equation in program.eqns if equation.primitive.name == "transpose"
                  and computed(equation.invars[0]) and equation.invars[0] in sliced
                  for out in equation.outvars
                  if all(reader.primitive.name == "dot_general" for reader in readers.get(out, ()))}
        return dataclasses.replace(self, sliced=frozenset(sliced | folded), updated=frozenset(updated))

    def _annotated(self, equation: jcore.JaxprEqn, counting: Sequence[jcore.JaxprEqn]) -> bool:
        """Whether `DynamicSliceAnnotator` annotates `equation`, a dynamic slice or update: contiguous
        and aligned, its offsets literals or computed from the loop's counter alone, whose byte offsets
        over the loop's turns, clamped as the slice clamps them or not, step evenly."""
        whole = dimensions(equation.invars[0])
        updating = equation.primitive.name == "dynamic_update_slice"
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
        if equation.primitive.name != "scan":
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
            stepped = any(step.outvars == [returned[0]] and step.primitive.name == "add"
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
    while (equation := producers.get(value)) is not None and equation.primitive.name in VIEWS \
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
            out = equation.primitive.bind(*map(read, equation.invars), **equation.params)
            values.update(zip(equation.outvars, out if equation.primitive.multiple_results else [out],
                              strict=True))
    return [int(read(offset)) for offset in offsets]


def _even(offsets: Sequence[int]) -> bool:
    """Whether `offsets` step by one stride throughout (`get_linear_stride`)."""
    return len({later - earlier for earlier, later in itertools.pairwise(offsets)}) <= 1
