"""Reading a traced program (a jaxpr) as the platform it is lowered for runs it.

`dew.nn.flops` counts a program's work as an accelerator runs it, and `dew.interop.sequence_causality`
walks a source's program for what each position reads; each meets the platform choices
(`jax.lax.platform_dependent`) a model's numerics make (`dew.nn.precision`) and the programs an equation
runs within it, and this is where they agree on which branch a platform runs and which bodies XLA sees.
"""

from __future__ import annotations

import contextlib
import enum
import functools
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from types import ModuleType
from typing import NamedTuple, TypeGuard

import jax
import jax.extend.core as jcore
import jax.extend.core.primitives as jax_primitives
import jax.numpy as jnp
import numpy as np
from jax._src import (
    config as jax_config,
    core as jax_core,
    dtypes,
    effects as jax_effects,
    mesh as mesh_lib,
    xla_metadata as jax_xla_metadata,
)
from jax._src.interpreters import partial_eval
from jax._src.literals import TypedNdArray
from jax._src.pallas.mpmd import mpmd_map_p
from jax.core import ShapedArray, Tracer
from jax.experimental.pallas import pallas_call_p
from jax.extend.backend import backends

from dew._devices import _compute_capability


def holes[Tree, Leaf](tree: Tree, selected: Callable[[Leaf], bool] | Sequence[bool],
                      is_leaf: Callable[[Leaf], bool] | None = None
                      ) -> tuple[list[Leaf], Callable[[Sequence[Leaf]], Tree]]:
    """The leaves of `tree` (a call's arguments, `(args, kwargs)`) that `selected` keeps, in order, and
    `fill(values)`, `tree` with those leaves replaced by `values` in their order and every other as it is:
    what a trace of a function of some of its arguments (its arrays) binds and what it rebuilds.

    A callable `selected` is a predicate of each leaf. Any other is a mask: one bool per leaf, in JAX's
    flatten order under `is_leaf`, for a choice no leaf alone states (whose paired tangent is live). A
    mask of another length raises `ValueError`."""
    leaves, structure = jax.tree.flatten(tree, is_leaf=is_leaf)
    kept = map(selected, leaves) if callable(selected) else selected
    places = [index for index, keep in zip(range(len(leaves)), kept, strict=True) if keep]

    def fill(values: Sequence[Leaf]) -> Tree:
        filled = list(leaves)
        for index, value in zip(places, values, strict=True):
            filled[index] = value
        return jax.tree.unflatten(structure, filled)

    return [leaves[index] for index in places], fill


def platform_branch(platforms: Sequence[Sequence[str] | None], lowering: str) -> int:
    """The index of the branch of a platform choice that a program lowered for `lowering` runs: of a
    `cond` whose `branches_platforms` name each branch's platforms (None for its default), the first
    whose platforms hold `lowering`, or the default. JAX keeps that branch for each platform it lowers for
    (`_cond_lowering`, jax/_src/lax/control_flow/conditionals.py)."""
    return next(index for index, names in enumerate(platforms) if names is None or lowering in names)


NATIVE_KERNELS = (pallas_call_p, mpmd_map_p)
"""The primitives a Pallas kernel binds: a grid's call (`pallas_call`), and a core map's (`pallas.core_map`,
which a Mosaic GPU kernel, `plgpu.kernel`, runs through), each one custom call where it runs natively
(`pallas_call_lowering`, jax/_src/pallas/triton/pallas_call_registration.py; `_mpmd_map_lowering`,
jax/_src/pallas/mpmd.py)."""



def lowering_platform(device) -> str | None:
    """The platform JAX lowers a program on `device` for, its lowering rules' name: a GPU's `rocm`,
    `oneapi` or `cuda` as its client's platform version names it (`PjRtClient::platform_version`), else
    `cuda` where it states an NVIDIA compute capability (a compile-only target's, whose version is its
    device kind; an abstract one's, a compile-only mesh's, which holds no client), else None; any other
    device's own platform."""
    if device.platform != "gpu":
        return device.platform
    version = _platform_version(device)
    for platform, names in (("rocm", ("rocm",)), ("oneapi", ("oneapi", "sycl")), ("cuda", ("cuda",))):
        if any(name in version for name in names):
            return platform
    capability = _compute_capability(device)
    return "cuda" if capability and re.fullmatch(r"\d+\.\d+", str(capability)) else None


def _platform_version(device) -> str:
    """`device`'s client's platform version, lowercased (`PjRtClient::platform_version`), read at this
    boundary: jax's devices declare no client, a concrete one holds it and an abstract one (a compile-only
    mesh's) none, which gives ''."""
    return str(getattr(getattr(device, "client", None), "platform_version", "")).lower()

def opaque(equation: jcore.JaxprEqn) -> bool:
    """Whether XLA sees nothing of `equation`'s body: a native Pallas kernel's (`NATIVE_KERNELS`), one
    custom call whose results are its outputs alone, its body's values in registers and shared memory,
    its products Triton's or Mosaic GPU's. An interpreted core map is no native kernel: JAX lowers it
    through its interpreter (`_mpmd_map_fallback_lowering`), so its body is read as XLA runs it."""
    if equation.primitive is mpmd_map_p:
        return not equation.params["interpret"]
    return equation.primitive is pallas_call_p


def bodies(equation: jcore.JaxprEqn) -> list[jcore.Jaxpr]:
    """The programs `equation` runs within it (JAX's own finder, `jax.extend.core.jaxprs_in_params`): a
    call's, a loop's, a branch's or a checkpoint's. A native Pallas kernel runs none XLA sees (`opaque`)."""
    if opaque(equation):
        return []
    return list(jcore.jaxprs_in_params(equation.params))



def with_bodies(params: Mapping, rebuilt: Callable[[jcore.Jaxpr], jcore.Jaxpr]) -> dict:
    """`params` with each program within them (`bodies`' finder's: a parameter that is a program, or a tuple
    holding them, a branch's) replaced by `rebuilt(program)`, every other parameter as it is, and a tuple
    itself where `rebuilt` returns each of its programs as it was: what a program's equation retraced at
    other shapes binds, and what a pass that rewrites bodies keeps where it changes none (a program's
    identity is what JAX shares its lowering by)."""
    def replaced(value):
        if isinstance(value, jcore.Jaxpr):
            return rebuilt(value)
        if isinstance(value, tuple) and any(isinstance(each, jcore.Jaxpr) for each in value):
            made = tuple(rebuilt(each) if isinstance(each, jcore.Jaxpr) else each for each in value)
            return value if all(new is old for new, old in zip(made, value, strict=True)) else made
        return value

    return {key: replaced(value) for key, value in params.items()}


def outer_effects(effects: jax_core.Effects) -> jax_core.Effects:
    """`effects` less those that name one of their jaxpr's inputs (`JaxprInputEffect`), which mean an input
    of the jaxpr that holds them, never of the program around it."""
    return {effect for effect in effects if not isinstance(effect, jax_effects.JaxprInputEffect)}


def with_equations(jaxpr: jcore.Jaxpr, eqns: list[jcore.JaxprEqn], **changes) -> jcore.Jaxpr:
    """`jaxpr` holding `eqns` (and `changes`, `Jaxpr.replace`'s), its effects its own and every equation's
    it now holds: a pass that writes an equation in (a region's call, a kernel's launch: an interpreted
    one's ordered IO) states its effect on the program that runs it, whose lowering threads that effect's
    token (row 12217: a call lowered inside a branch that stated none raised `KeyError: OrderedIOEffect`)."""
    return jaxpr.replace(eqns=eqns, effects=jax_core.join_effects(
        jaxpr.effects, *(outer_effects(equation.effects) for equation in eqns)), **changes)


def with_bodies_of(equation: jcore.JaxprEqn, params: dict) -> jcore.JaxprEqn:
    """`equation` binding `params` (`with_bodies`' rewrite of its bodies), its effects its own and every
    effect its bodies now hold (`with_equations`)."""
    return equation.replace(params=params, effects=jax_core.join_effects(
        equation.effects, *(outer_effects(body.effects) for body in jax_core.jaxprs_in_params(params))))


def equations(jaxpr: jcore.Jaxpr) -> Iterator[jcore.JaxprEqn]:
    """Every equation of `jaxpr` and of every program within it, at any depth (`jax.extend.core.subjaxprs`):
    a call's, a loop's, a branch's, a custom derivative's, a kernel's."""
    yield from jaxpr.eqns
    for inner in jcore.subjaxprs(jaxpr):
        yield from equations(inner)


def computed(value) -> TypeGuard[jcore.Var]:
    """Whether `value` is a program's variable, not a literal it reads."""
    return isinstance(value, jcore.Var) and not isinstance(value, jcore.Literal)


class RaggedLayout(enum.Enum):
    """A grouped product's layout as `jax.lax.ragged_dot_general` states it (`ragged_layout`)."""

    ROWS = "rows"
    """Rows grouped and their depth contracted: `[rows, depth]` by `[groups, depth, width]`."""
    ROWS_TRANSPOSED = "rows transposed"
    """The same over matrices held `[groups, width, depth]`."""
    GROUPS = "groups"
    """Each group's rows contracted: `[rows, depth]` by `[rows, width]` into `[groups, depth, width]`, a
    grouped matrix's gradient."""


class DotArithmetic(NamedTuple):
    """What a dot algorithm states of its arithmetic, as JAX lowers it (`dot_arithmetic`): each operand's
    precision in bits, its significand's digits (the implicit one counted) and its largest finite value,
    whether those are TF32's, its accumulation's bits, the primitive products it sums (an X3's three) and
    whether its accumulation may be imprecise."""

    operand_bits: tuple[int, int]
    significands: tuple[int, int]
    largest: tuple[float, float]
    tf32: bool
    accumulation_bits: int
    operations: int
    imprecise: bool

    @property
    def parts(self) -> int:
        """The operand split count stated by this algorithm's primitive products (`_PARTS`); refused for any
        unsupported operation count."""
        parts = _PARTS.get(self.operations)
        if parts is None:
            raise ValueError(f"a dot algorithm of {self.operations} primitive products states no split of "
                             "its operands into parts")
        return parts

    @property
    def digits(self) -> int:
        """The significand digits a product in this algorithm keeps of each operand: the narrower side's
        once per part it splits an operand into (`parts`)."""
        return self.parts * min(self.significands)

# The parts a split dot algorithm takes each operand in, by its primitive products: one product of one
# part; three of two parts' four, the low parts' product dropped (TF32x3, BF16x3); all four of two parts
# (an X4); six of three parts' nine, the three lowest dropped (BF16x6); all nine (BF16x9).
_PARTS = {1: 1, 3: 2, 4: 2, 6: 3, 9: 3}

TRITON_ALGORITHMS = tuple(preset for preset in jax.lax.DotAlgorithmPreset
                          if preset.supported_lhs_types is not None
                          and preset is not jax.lax.DotAlgorithmPreset.F64_F64_F64)
"""The dot algorithms Pallas's Triton lowering runs as named (`_dot_general_lowering`): TF32's one and three
passes and IEEE fp32 as `tt.dot`'s input precisions, each 16-bit one as its operands' own product, and a bf16
split of fp32 operands (X3, X6, X9) as its parts' bf16 products; of those naming their operand types, it
refuses fp64's (the default and the FP8 ones name none)."""


def triton_dot_precision(rows: int, depth: int, columns: int, dtype, *,
                         ieee: bool = False) -> jax.lax.DotAlgorithmPreset:
    """Cheapest Triton dot algorithm (`TRITON_ALGORITHMS`) at least as exact as the dtype's native 16-bit
    product or fp32's TF32 / IEEE (`ieee`) product (`dew.algebra.error.at_least`, `product_precision`,
    rows 5473, 11878)."""
    from dew.algebra import error

    operand = jnp.dtype(dtype)
    if operand == jnp.float32:
        asked = (jax.lax.DotAlgorithmPreset.F32_F32_F32 if ieee
                 else jax.lax.DotAlgorithmPreset.TF32_TF32_F32)
    else:
        asked = own_preset(operand)
        if asked is None:
            raise ValueError(f"Triton dot products take fp32 or 16-bit float operands, not {operand}")
    return error.product_precision(rows, depth, columns, operand,
                                   admits=error.at_least(asked, operand), stated=TRITON_ALGORITHMS)


def triton_dot(a: jax.Array, b: jax.Array, *, ieee: bool = False) -> jax.Array:
    """`a @ b` accumulated in fp32 with its explicit Triton dot algorithm (`triton_dot_precision`)."""
    precision = triton_dot_precision(a.shape[-2], a.shape[-1], b.shape[-1], a.dtype, ieee=ieee)
    return jnp.dot(a, b, preferred_element_type=jnp.float32, precision=precision)


MOSAIC_TPU_ALGORITHMS = {jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3: jax.lax.Precision.HIGH,
                         jax.lax.DotAlgorithmPreset.F32_F32_F32: jax.lax.Precision.HIGHEST}
"""The dot algorithms Pallas's Mosaic TPU lowering runs, each by the precision that names it
(`_dot_general_lowering_rule`: `HIGH` its `contract_precision<bf16x3>`, `HIGHEST` its `<fp32>`); it takes
no algorithm itself, and the default precision names none."""


def mosaic_tpu_fp32_precision(m: int = 64, k: int = 64, n: int = 64) -> jax.lax.Precision:
    """Cheapest Mosaic TPU fp32 dot precision (`MOSAIC_TPU_ALGORITHMS`) at least as exact as IEEE fp32
    (`dew.algebra.error.at_least`, `product_precision`, rows 5473, 11879)."""
    from dew.algebra import error

    wide = jnp.dtype(jnp.float32)
    chosen = error.product_precision(
        m, k, n, wide,
        admits=error.at_least(jax.lax.DotAlgorithmPreset.F32_F32_F32, wide),
        stated=MOSAIC_TPU_ALGORITHMS)
    return MOSAIC_TPU_ALGORITHMS[chosen]


def mosaic_tpu_dot_precision(lhs_dtype, rhs_dtype, m: int, k: int, n: int) -> jax.lax.Precision:
    """Mosaic TPU dot precision for `mosaic_tpu._product` and `splash._dot` (row 11970):
    `mosaic_tpu_fp32_precision(m, k, n)` when both operands are float32, else `lax.Precision.DEFAULT`."""
    if jnp.dtype(lhs_dtype) == jnp.float32 and jnp.dtype(rhs_dtype) == jnp.float32:
        return mosaic_tpu_fp32_precision(m, k, n)
    return jax.lax.Precision.DEFAULT


@functools.cache
def _preset_arithmetic(preset: jax.lax.DotAlgorithmPreset, lhs, rhs) -> DotArithmetic | None:
    """`preset`'s arithmetic over operands of dtypes `lhs` and `rhs`, read from the attribute JAX's own
    conversion lowers it to (`DotAlgorithmPreset._convert_to_hlo_attr`, the one account of a preset's
    precision types, its component counts and its operations); None for the default preset, which names no
    algorithm."""
    from jax.interpreters import mlir

    with mlir.make_ir_context():
        lowered = preset._convert_to_hlo_attr(lhs, rhs)
        return None if lowered is None else algorithm_arithmetic(lowered)


def algorithm_arithmetic(algorithm) -> DotArithmetic:
    """The arithmetic a lowered dot algorithm attribute states (a `stablehlo.DotAlgorithm`, as JAX lowers a
    preset or a `jax.lax.DotAlgorithm`, and as a lowered program's dots carry it): its precision types,
    their component counts and its operations, read where the attribute's context is current."""
    from jax.extend.mlir import ir

    sides = (algorithm.lhs_precision_type, algorithm.rhs_precision_type)
    # Each a float type of the bindings (`ir.FloatType`'s cast holds it to one), whose width is its bits.
    left, right = (ir.FloatType(side).width for side in sides)
    (left_digits, left_largest), (right_digits, right_largest) = map(_format, sides)
    return DotArithmetic(operand_bits=(left, right), significands=(left_digits, right_digits),
                         largest=(left_largest, right_largest),
                         tf32=all(isinstance(side, ir.FloatTF32Type) for side in sides),
                         accumulation_bits=ir.FloatType(algorithm.accumulation_type).width,
                         operations=algorithm.num_primitive_operations,
                         imprecise=algorithm.allow_imprecise_accumulation)


def canonical_precision(precision: jax.lax.PrecisionLike):
    """`precision` as JAX reads it (its own canonicalization: a name, an enum, a pair of them, a dot
    algorithm or a preset), the one place Dew reads it so."""
    from jax._src.lax import lax as lax_internal

    return lax_internal.canonicalize_precision(precision)


def dot_arithmetic(precision: jax.lax.PrecisionLike, lhs_dtype, rhs_dtype) -> DotArithmetic | None:
    """The arithmetic the dot algorithm `precision` names runs a product of operands of those dtypes in, as
    JAX lowers it: a preset's by JAX's own conversion of it (`_preset_arithmetic`), a `jax.lax.DotAlgorithm`'s
    by its fields; None where `precision` names no algorithm (none stated, a `jax.lax.Precision` or a pair of
    them, the default preset), the backend's choice by the operands. The one reading of a preset: the
    algebra's cost prices a product's mode by it (`dew.algebra.cost`) and its rounding bound admits a
    contraction by it (`dew.algebra.terms`), no list of preset names beside JAX's."""
    named = canonical_precision(precision)
    if isinstance(named, jax.lax.DotAlgorithm):
        sides = (named.lhs_precision_type, named.rhs_precision_type)
        return DotArithmetic(operand_bits=(_bits(sides[0]), _bits(sides[1])),
                             significands=(_digits(sides[0]), _digits(sides[1])),
                             largest=(_largest(sides[0]), _largest(sides[1])), tf32=False,
                             accumulation_bits=_bits(named.accumulation_type),
                             operations=named.num_primitive_operations,
                             imprecise=named.allow_imprecise_accumulation)
    if isinstance(named, jax.lax.DotAlgorithmPreset):
        return _preset_arithmetic(named, jnp.dtype(lhs_dtype), jnp.dtype(rhs_dtype))
    return None


def dot_algorithms(dtype) -> tuple[jax.lax.DotAlgorithmPreset, ...]:
    """The dot algorithm presets JAX runs a product of `dtype` operands in as they are stored: those whose
    compute types JAX leaves the operands' own (`get_algorithm_compute_types`, which converts an operand a
    preset does not take) and whose lowering JAX's conversion accepts for them (`dot_arithmetic`: an FP8
    preset names no operand types and refuses any but FP8 ones there). The default preset names no
    algorithm and is none of them."""
    from jax._src.lax import lax as lax_internal

    stored = jnp.dtype(dtype)
    found = []
    for preset in jax.lax.DotAlgorithmPreset:
        left, right, _ = lax_internal.get_algorithm_compute_types(preset, stored, stored, stored)
        if (jnp.dtype(left), jnp.dtype(right)) != (stored, stored):
            continue
        try:
            named = dot_arithmetic(preset, stored, stored) is not None
        except ValueError:  # JAX's own refusal of these operand types for the preset
            continue
        if named:
            found.append(preset)
    return tuple(found)


def own_preset(dtype) -> jax.lax.DotAlgorithmPreset | None:
    """A 16-bit float's own dot algorithm, its products exact and summed once in float32: of the presets
    JAX runs its operands in as they are stored (`dot_algorithms`), the one of one primitive product of its
    own width with float32 sums (BF16_BF16_F32, F16_F16_F32); None for any other dtype. The one statement
    of it, which attention's jax.nn form, a product's precision (`dew.nn.precision`) and the normal form
    read (row 10808)."""
    return _own_preset(jnp.dtype(dtype))


@functools.cache
def _own_preset(dtype: jnp.dtype) -> jax.lax.DotAlgorithmPreset | None:
    """`own_preset` of a dtype."""
    if dtype.itemsize != 2 or not jnp.issubdtype(dtype, jnp.floating):
        return None
    width = _bits(dtype)

    def own(preset: jax.lax.DotAlgorithmPreset) -> bool:
        arithmetic = dot_arithmetic(preset, dtype, dtype)
        return (arithmetic is not None and arithmetic.operand_bits == (width, width)
                and arithmetic.operations == 1 and arithmetic.accumulation_bits == 32
                and not arithmetic.imprecise)

    (found,) = (preset for preset in dot_algorithms(dtype) if own(preset))
    return found


def _bits(dtype) -> int:
    """A dtype's width in bits."""
    return 8 * jnp.dtype(dtype).itemsize


def _digits(dtype) -> int:
    """A float dtype's significand digits, its implicit one counted."""
    return jnp.finfo(dtype).nmant + 1


def _largest(dtype) -> float:
    """A float dtype's largest finite value."""
    return float(jnp.finfo(dtype).max)


def _format(side) -> tuple[int, float]:
    """The significand digits (its implicit one counted) and the largest finite value of a lowered dot
    algorithm's precision type `side`: TF32's ten stored digits and its one over float32's exponents, the
    one such type no dtype lowers to; any other's by the float dtype JAX lowers to it
    (`mlir.dtype_to_ir_type`), no list of types' formats beside the dtypes' own. JAX's float dtypes are read
    from its internal `jax._src.dtypes._float_types`, which a bump of JAX's pin checks."""
    from jax.extend.mlir import ir
    from jax.interpreters import mlir

    if isinstance(side, ir.FloatTF32Type):
        return 11, float(np.ldexp(2 - 2.0 ** -10, jnp.finfo(jnp.float32).maxexp - 1))
    # In the type's own context: a compile's checks read a program's dots outside any of theirs
    # (`report.triton_wrong`, a GPU target's compile).
    with side.context:
        for dtype in map(np.dtype, dtypes._float_types):
            if mlir.dtype_to_ir_type(dtype) == side:
                return _digits(dtype), _largest(dtype)
    raise ValueError(f"no float dtype lowers to the dot algorithm's precision type {side}")


def ragged_layout(lhs_rank: int, rhs_rank: int, numbers) -> RaggedLayout | None:
    """Which grouped product a `ragged_dot_general` of operands of those ranks under `numbers` (its
    `RaggedDotDimensionNumbers`) is, its left operand's rows ragged and no batch axes; None for any other.
    The one reading of those dimensions: the algebra's fusion takes a product's form from it
    (`dew.algebra.fusion`)."""
    (contracted, against), batch = (tuple(map(tuple, pair)) for pair in numbers.dot_dimension_numbers)
    grouped = tuple(numbers.rhs_group_dimensions)
    if batch != ((), ()) or lhs_rank != 2 or tuple(numbers.lhs_ragged_dimensions) != (0,):
        return None
    if grouped == (0,) and rhs_rank == 3 and contracted == (1,) and against in ((1,), (2,)):
        return RaggedLayout.ROWS if against == (1,) else RaggedLayout.ROWS_TRANSPOSED
    if grouped == () and rhs_rank == 2 and contracted == (0,) and against == (0,):
        return RaggedLayout.GROUPS
    return None


def read_by(jaxpr: jcore.Jaxpr, sinks: Iterable) -> set[jcore.Var]:
    """The variables of `jaxpr` that `sinks` read, themselves included: each sink's own, and every operand of
    an equation one of whose results is read. A literal among `sinks` reads nothing. Effects are not
    followed: what a value depends on, not what a program must still run (`live`)."""
    read = {value for value in sinks if computed(value)}
    for equation in reversed(jaxpr.eqns):
        if any(var in read for var in equation.outvars):
            read.update(value for value in equation.invars if computed(value))
    return read


def live(equations: Sequence[jcore.JaxprEqn], outvars: Sequence) -> list[jcore.JaxprEqn]:
    """`equations`, in order, without those nothing reads: the ones `outvars` read (`read_by`'s walk) and
    every equation with an effect, kept whatever reads it."""
    wanted = {value for value in outvars if computed(value)}
    kept: list[jcore.JaxprEqn] = []
    for equation in reversed(equations):
        if equation.effects or any(var in wanted for var in equation.outvars):
            kept.append(equation)
            wanted.update(value for value in equation.invars if computed(value))
    return kept[::-1]


def inlined(body: jcore.Jaxpr, operands: Sequence[jcore.Var | jcore.Literal],
            consts: Sequence = ()) -> tuple[list[jcore.JaxprEqn], list[jcore.Var | jcore.Literal]]:
    """`body`'s equations, each value its own fresh variable, read over `operands` and its constants over
    `consts`, and its results: a program run in its caller as one of its own, as one body traced once may
    be run more than once. A constant is the caller's variable or literal where `consts` gives one (a
    constant the caller holds among its own, `dew.algebra.program`'s span opened), else a literal of the
    value given (a program compiled whole)."""
    local: dict[jcore.Var, jcore.Var | jcore.Literal] = dict(zip(body.invars, operands, strict=True))
    for var, value in zip(body.constvars, consts, strict=True):
        local[var] = value if isinstance(value, jcore.Var | jcore.Literal) else jcore.Literal(value, var.aval)

    def read(value):
        return local.get(value, value) if computed(value) else value

    equations = []
    for equation in body.eqns:
        fresh = [out if isinstance(out, jcore.DropVar) else jcore.Var(out.aval) for out in equation.outvars]
        equations.append(equation.replace(invars=[read(value) for value in equation.invars], outvars=fresh))
        local.update(zip(equation.outvars, fresh, strict=True))
    return equations, [read(value) for value in body.outvars]


_LAX = jax.lax
_EXTENDED = jax_primitives
_XLA_METADATA = jax_xla_metadata

_KINDS: dict[str, tuple[tuple[ModuleType, tuple[str, ...]], ...]] = {
    # A call JAX lowers to its program in the caller (`called`): a `jit`'s, an evaluated jaxpr's, a
    # checkpoint's, a custom derivative's primal and a custom batching rule's.
    "call": ((_EXTENDED, ("jit_p", "call_p", "remat_p", "custom_jvp_call_p", "custom_vjp_call_p",
                          "custom_vmap_p")),),
    # A call that does nothing but its program (`purely_called`): a `jit`'s and an evaluated jaxpr's.
    "pure call": ((_EXTENDED, ("jit_p", "call_p")),),
    "entrywise": (
        (_LAX, ("abs_p", "acos_p", "acosh_p", "add_p", "and_p", "asin_p", "asinh_p", "atan2_p", "atan_p",
                "atanh_p", "bessel_i0e_p", "bessel_i1e_p", "cbrt_p", "ceil_p", "clamp_p", "clz_p",
                "complex_p",
                "conj_p", "convert_element_type_p", "copy_p", "cos_p", "cosh_p", "digamma_p", "div_p", "eq_p",
                "eq_to_p", "erf_inv_p", "erf_p", "erfc_p", "exp2_p", "exp_p", "expm1_p", "floor_p",
                "ge_p",
                "gt_p", "igamma_grad_a_p", "igamma_p", "igammac_p", "imag_p", "integer_pow_p", "is_finite_p",
                "le_p", "le_to_p", "lgamma_p", "log1p_p", "log2_p", "log_p", "logistic_p", "lt_p", "lt_to_p",
                "max_p", "min_p", "mul_p", "mulhi_p", "ne_p", "neg_p", "nextafter_p", "not_p",
                "one_minus_square_p", "or_p", "polygamma_p", "population_count_p", "pow_p", "real_p",
                "reduce_precision_p", "regularized_incomplete_beta_p", "rem_p", "round_p", "rsqrt_p",
                "select_n_p", "shift_left_p", "shift_right_arithmetic_p", "shift_right_logical_p", "sign_p",
                "sin_p", "sinh_p", "sqrt_p", "square_p", "stop_gradient_p", "sub_p", "tan_p", "tanh_p",
                "xor_p", "zeta_p")),
        # The sum a derivative adds its terms with, and the counter-based hash a key's bits are drawn by.
        (_EXTENDED, ("add_jaxvals_p", "threefry2x32_p"))),
    # Mapped over positions but not entrywise over a value's: a key's operations (a key's data extends its
    # shape), a bitcast between widths, a polynomial over its coefficients, a staged value.
    "positionwise": (
        (_LAX, ("bitcast_convert_type_p", "polynomial_p", "stage_p")),
        (_EXTENDED, ("random_bits_p", "random_clone_p", "random_fold_in_p", "random_seed_p", "random_split_p",
                     "random_unwrap_p")),
        # The identity `jax.experimental.xla_metadata.set_xla_metadata` tags a value with, which JAX
        # defines in its private module alone and batches as a vectorized operation (train 1368's review).
        (_XLA_METADATA, ("xla_metadata_value_p",))),
    "rearranges": ((_LAX, ("broadcast_in_dim_p", "concatenate_p", "copy_p", "dynamic_slice_p",
                           "dynamic_update_slice_p", "gather_p", "optimization_barrier_p", "pad_p",
                           "reshape_p", "rev_p", "select_n_p", "slice_p", "sort_p", "split_p", "squeeze_p",
                           "stack_p", "transpose_p", "unstack_p")),),
    "reduction": ((_LAX, ("argmax_p", "argmin_p", "reduce_and_p", "reduce_max_p", "reduce_min_p",
                          "reduce_or_p", "reduce_prod_p", "reduce_sum_p", "reduce_xor_p")),),
    "cumulative": ((_LAX, ("cumlogsumexp_p", "cummax_p", "cummin_p", "cumprod_p", "cumsum_p")),),
    "scatter": ((_LAX, ("scatter_add_p", "scatter_max_p", "scatter_min_p", "scatter_mul_p", "scatter_p",
                        "scatter_sub_p")),),
    # A value made from no operand: a counting sequence, a device's place along a mesh axis, an ordering
    # token.
    "generated": ((_LAX, ("axis_index_p", "create_token_p", "iota_p")),),
}
"""Each kind of operation the analyses of a program read, by the JAX primitives of it, named where JAX's
public modules export them (`jax.lax`, `jax.extend.core.primitives`), or where JAX defines one it exports
nowhere (`set_xla_metadata`'s tag, `jax._src.xla_metadata`), each module imported statically: the one
statement of what each of JAX's operations is, which no attribute of JAX's own `Primitive` states. A name a
release does not define is none of its primitives (`_kind`). A primitive of Dew's own is added where it is
defined (`register_entrywise`, `register_positionwise`, `register_call`); one of another library's or a
user's is no kind, whatever its name."""

_OWN: dict[str, set[jcore.Primitive]] = {"call": set(), "entrywise": set(), "positionwise": set()}
"""Dew's own primitives of a kind (`register_entrywise`, `register_positionwise`, `register_call`), stated
where each is defined."""


@functools.cache
def _kind(kind: str) -> frozenset[jcore.Primitive]:
    """JAX's primitives of `kind` (`_KINDS`) this release exports, read off each module's own namespace."""
    return frozenset(primitive for module, names in _KINDS[kind] for name in names
                     if isinstance(primitive := vars(module).get(name), jcore.Primitive))


def _is(kind: str, primitive: jcore.Primitive) -> bool:
    return primitive in _kind(kind) or primitive in _OWN.get(kind, ())


def register_entrywise(primitive: jcore.Primitive) -> jcore.Primitive:
    """State `primitive`, one of Dew's own, entrywise (`entrywise`) where it is defined; returns it."""
    _OWN["entrywise"].add(primitive)
    return primitive


def register_positionwise(primitive: jcore.Primitive) -> jcore.Primitive:
    """State `primitive`, one of Dew's own, positionwise (`positionwise`: JAX batches it by leaving the
    mapped axis where it found it, `batching.defvectorized`) where it is defined; returns it."""
    _OWN["positionwise"].add(primitive)
    return primitive


def register_call(primitive: jcore.Primitive) -> jcore.Primitive:
    """State `primitive`, one of Dew's own, a call JAX lowers to its program in the caller (`called`), and
    none that does nothing but its program (`purely_called`), where it is defined; returns it."""
    _OWN["call"].add(primitive)
    return primitive


def is_inlined_call(primitive: jcore.Primitive) -> bool:
    """Whether `primitive` is a call JAX lowers to its program in the caller (`called`)."""
    return _is("call", primitive)


def is_transparent_call(primitive: jcore.Primitive) -> bool:
    """Whether `primitive` is a call that does nothing but its program (`purely_called`)."""
    return _is("pure call", primitive)


def called(equation: jcore.JaxprEqn) -> jcore.Jaxpr | None:
    """The program `equation` runs on its operands one for one, returning its results, else None: every
    reader of a call through its program reads it here (`dew.training.transaction.gradients_read`, the
    kernel split's `_inlined`, the float64 terms, the algebra's normal forms). A call is a primitive JAX
    lowers to its program in the caller (`is_inlined_call`): a `jit`'s, which XLA's `CallInliner` inlines
    (`_pjit_lowering`, jax/_src/pjit.py; `GpuCompiler::RunPreSPMDPartitionerPasses`,
    xla/service/gpu/gpu_compiler.cc), an evaluated jaxpr's, a checkpoint's (`_remat_lowering`,
    jax/_src/ad_checkpoint.py), a custom derivative's primal (`_custom_jvp_vjp_call_lowering`,
    jax/_src/custom_derivatives.py), a custom batching rule's (`custom_vmap_p`'s lowering,
    jax/_src/custom_batching.py) and a marked span
    (`dew.nn.derivatives.marked_call_p`). XLA sees no call: a result that is the program's own parameter is
    its operand's buffer, and one that is a cheap value of another is recomputed where it is read, as it
    would be without the call. Its constants are its own (`consts`); a custom rule's leading constants
    (`num_consts`) are among the operands its program reads, one for one. A loop's body runs as many times as
    it steps and a branch's only where it is taken, so neither is a call."""
    body = _program(equation.primitive, equation.params)
    if body is not None and (len(body.invars), len(body.outvars)) == (
            len(equation.invars), len(equation.outvars)):
        return body
    return None


def _program(primitive: jcore.Primitive, params: Mapping) -> jcore.Jaxpr | None:
    """The one program a call's primitive (`is_inlined_call`) holds in `params`, None for any other
    primitive."""
    if not is_inlined_call(primitive):
        return None
    found = list(jcore.jaxprs_in_params(params))
    return found[0] if len(found) == 1 else None


def purely_called(equation: jcore.JaxprEqn) -> jcore.Jaxpr | None:
    """The program `equation` runs (`called`) where its call does nothing else, else None: what a reader that
    runs a call in its caller reads (staging's `flattened`, a template layout's `_call`, torchax's evaluator).
    The pure calls are a `jit`'s and an evaluated jaxpr's (`is_transparent_call`; an optax helper's `jit`
    over a whole tree, `jnp.einsum`'s). A checkpoint's, a custom
    derivative's and a custom batching rule's call do more than their program, a rule of their own, and a
    marked span is a boundary the algebra pass reads (`dew.nn.derivatives.marked`: a protected span's inside
    no rule touches), so each keeps its rule only as a call."""
    return called(equation) if is_transparent_call(equation.primitive) else None


UNROUNDED: dict[jcore.Primitive, Callable] = {}
"""Each primitive whose implementation rounds to a format of its own whatever its operands' dtype, to the
operation it rounds: a function of its operands that computes at their width. JAX's own primitives compute at
the dtype they are given, so none is here; one of Dew's that fixes a format states its operation where it is
defined, as it states its derivative there (`dew.nn.precision.correctly_rounded_divide_p`: the quotient).
What evaluates a traced program wider than it was traced reads this in place of the primitive (a float64
replay, the rounding model's tapped run): bound on wider operands, the primitive itself would still round
to its format."""


def own_derivative(primitive: jcore.Primitive) -> bool:
    """Whether `primitive` is a call whose derivative is a rule of its own (`jax.custom_jvp`'s,
    `jax.custom_vjp`'s): its cotangents and tangents are the rule's, not its program's."""
    return primitive in (jcore.primitives.custom_jvp_call_p, jcore.primitives.custom_vjp_call_p)


def _of(equation: jcore.JaxprEqn, primitive: jcore.Primitive) -> Mapping:
    """`equation`'s parameters, where it binds `primitive`: a reader of one control-flow operation's programs
    is asked of that operation alone, and another's equation is refused by its name."""
    if equation.primitive is not primitive:
        raise TypeError(f"{equation.primitive.name} is not {primitive.name}, whose programs were asked of it")
    return equation.params


def scan_body(equation: jcore.JaxprEqn) -> jcore.Jaxpr:
    """The program a scan's equation (`jax.lax.scan_p`) runs each step."""
    return _of(equation, jax.lax.scan_p)["jaxpr"]


def loop_programs(equation: jcore.JaxprEqn) -> tuple[jcore.Jaxpr, jcore.Jaxpr]:
    """A while loop's equation's (`jax.lax.while_p`) predicate and body."""
    params = _of(equation, jax.lax.while_p)
    return params["cond_jaxpr"], params["body_jaxpr"]


def branches(equation: jcore.JaxprEqn) -> tuple[jcore.Jaxpr, ...]:
    """A conditional's equation's (`jax.lax.cond_p`: `cond`, `switch`, a platform choice) branches, in the
    order its index picks them."""
    return tuple(_of(equation, jax.lax.cond_p)["branches"])


def control_flow(primitive: jcore.Primitive) -> bool:
    """Whether `primitive` is JAX's control flow (a scan, a while loop, a conditional): its programs run a
    step at a time or where they are taken (`scan_body`, `loop_programs`, `branches`), never as a call's."""
    return primitive in (jax.lax.scan_p, jax.lax.while_p, jax.lax.cond_p)


def entrywise(primitive: jcore.Primitive) -> bool:
    """Whether `primitive`'s every output entry reads its operands' entries at its own index (each operand of
    the output's shape, or a scalar) and nothing else (`_KINDS`: JAX's unary and n-ary operations, a cast,
    a selection, a clamp, the sum its derivatives add with; Dew's own `register_entrywise`). The one
    vocabulary the analyses of a program's entries read (`templates.layout`,
    `dew.interop.sequence_causality`, the algebra's fusion). A primitive of another library's or a user's
    states nothing and is none, whatever its name."""
    return _is("entrywise", primitive)


def positionwise(primitive: jcore.Primitive) -> bool:
    """Whether `primitive` maps over its operands' positions, each output position reading its operands' at
    that position and nothing else, whatever it does within one: an entrywise operation (`entrywise`), and
    one JAX batches by leaving the mapped axis where it found it (`_KINDS`), so an added axis anywhere changes
    nothing of it. The operations on PRNG
    keys are such (a key's data extends its shape, and `random_bits` takes the shape it fills past the
    keys'), which are entrywise over a key array's positions and not over a value's."""
    return entrywise(primitive) or _is("positionwise", primitive)


def prng_operation(equation: jcore.JaxprEqn) -> bool:
    """Whether `equation` is a primitive operation on PRNG keys (`jax.dtypes.prng_key`), not a call,
    kernel or control flow: wrapping or unwrapping key data, seeding, splitting, folding in or drawing
    bits, none of which does a product."""
    if opaque(equation) or bodies(equation):
        return False
    return any(isinstance(var.aval, ShapedArray)
               and jnp.issubdtype(var.aval.dtype, jax.dtypes.prng_key)
               for var in (*equation.invars, *equation.outvars))


def rearranges(primitive: jcore.Primitive) -> bool:
    """Whether `primitive` only moves entries, each output entry one operand's entry or a fill, with no
    arithmetic and no rounding (a reshape, a squeeze, a broadcast, a transpose, a reversal, a slice, a
    gather, a concatenation, a padding, a split; `_KINDS`)."""
    return _is("rearranges", primitive)


def reduction(primitive: jcore.Primitive) -> bool:
    """Whether `primitive` reduces the axes its `axes` parameter names (a sum, a product, an extremum or its
    place, a bitwise fold; `_KINDS`)."""
    return _is("reduction", primitive)


def scatter(primitive: jcore.Primitive) -> bool:
    """Whether `primitive` writes updates into its first operand at indices, replacing or combining (a
    scatter, and its add, subtract, multiply, minimum and maximum; `_KINDS`)."""
    return _is("scatter", primitive)


def generated(primitive: jcore.Primitive) -> bool:
    """Whether `primitive` makes its value from no operand and nothing outside the program (an `iota`, a
    device's `axis_index`, an ordering token; `_KINDS`): a host callback reading no operand is none."""
    return _is("generated", primitive)


def cumulative(primitive: jcore.Primitive) -> bool:
    """Whether `primitive` scans along the axis its `axis` parameter names (a cumulative sum, product,
    extremum or log-sum-exp; `_KINDS`)."""
    return _is("cumulative", primitive)


def host_evaluated(primitive, operands: Sequence, params: Mapping) -> list | None:
    """`primitive`'s outputs over concrete `operands`, computed as this host computes a constant: at compile
    time, on its CPU, outside any mesh a program traces under and without the outputs' stated shardings
    (`on_host`); None where an operand is an array this process cannot read whole (one laid across another
    process's devices), or an operand or output is of an extended dtype (a PRNG key, which no host array
    holds), or the primitive runs per device of a mesh it names (a shard_map's body sees one shard, and the
    host holds no mesh to run it over), none a constant of this host. A call is its program, interpreted
    (`called`'s): a constant has no derivative, and a custom derivative's bind takes its rules as functions,
    not the equation's jaxprs. The one evaluator of a program's constants, the compile point's and a compiled
    program's plan alike."""
    if "mesh" in params or any((isinstance(value, jax.Array) and not value.is_fully_addressable)
                               or _extended(value) for value in operands):
        return None
    params = {name: None if name in ("sharding", "out_sharding") else value for name, value in params.items()}
    folded = _host_folded(primitive, operands, params)
    if folded is not None:
        return [folded]
    body = _program(primitive, params)
    with on_host():
        if body is not None:
            made = jcore.jaxpr_as_fun(body)(*operands)
        else:
            made = primitive.bind(*operands, **params)
    made = list(made) if primitive.multiple_results else [made]
    return None if any(map(_extended, made)) else made


def _host_folded(primitive: jcore.Primitive, operands: Sequence[jax.typing.ArrayLike],
                 params: Mapping[str, object]) -> TypedNdArray | None:
    """`scalar_folded` of `operands` as this host's bind takes them, host literals alone (a device array is
    bound): each canonicalized as the bind canonicalizes it (JAX's own, `dtypes.canonicalize_value`: a NumPy
    integer wider than the process's width narrowed, a Python integer at that width), admitted at its type
    (`admitted`), and cast to a stated output dtype first as XLA casts an operand before its operation, the
    output a literal of the type the primitive's abstract rule gives it (its dtype and weak type), under no
    mesh as `host_evaluated` binds a constant (its placement none), which a program's next equation binds as
    the bind's own output. Nothing is traced."""
    # A host literal alone: a device array's value is no constant this host reads without a transfer.
    if primitive not in SCALAR_FOLDED or any(isinstance(value, jax.Array) for value in operands):
        return None
    canonical = [dtypes.canonicalize_value(value) for value in operands]
    # Each at the type JAX's canonical aval states, not NumPy's dispatch on the operand as given.
    types = [jax.typeof(value) for value in canonical]
    if (held := admitted(canonical, [kind.dtype for kind in types])) is None:
        return None
    with meshless():
        out, _ = primitive.abstract_eval(*types, **params)
    if params.get("out_dtype") is not None:
        # A stated output dtype is the output's (`out`), each operand cast to it first; a float one declined
        # before any cast.
        if not _integral(out.dtype):
            return None
        held = [_converted(value, out.dtype) for value in held]
    folded = scalar_folded(primitive, held, params, out.dtype)
    return None if folded is None else TypedNdArray(np.asarray(folded, out.dtype), out)


def admitted(values: Sequence, dtypes: Sequence[jax.typing.DTypeLike]) -> list[np.generic] | None:
    """`values` as the concrete integer and boolean scalars `scalar_folded` computes over, each at its
    caller's canonical dtype in `dtypes` (a bind's operand's, a kernel's atom's), one held at another dtype
    converted as JAX folds a constant's conversion (`_converted`); None where one is not such a scalar (a
    traced value or an array of a device, one with axes, a float, a complex value) or its dtype is neither
    integer nor boolean. The one admission of a host fold's operands, a bind's (`_host_folded`) and a
    kernel's emission's (`dew.nn.kernels.templates.emission`)."""
    if not (all(map(integral_scalar, values)) and all(map(_integral, dtypes))):
        return None
    return [_converted(value, dtype) for value, dtype in zip(values, dtypes, strict=True)]


def integral_scalar(value) -> bool:
    """Whether `value` is a concrete integer or boolean scalar on this host: a Python integer or boolean, or
    a NumPy one of an integer or boolean dtype (JAX's 4-bit integers among them), a scalar or an array of no
    axes. A float is none, nor is a JAX array, whose value no host holds without a transfer."""
    if isinstance(value, jax.Array) or not isinstance(value, (int, np.generic, np.ndarray)):
        return False
    return np.ndim(value) == 0 and _integral(np.result_type(value))


def _converted(value, dtype: jax.typing.DTypeLike) -> np.generic:
    """`value` as `dtype` holds it, a scalar: as it is where it is held at `dtype`, else converted as JAX
    folds a constant's conversion (`convert_element_type`'s constant-folding rule,
    `partial_eval.const_fold_rules`: NumPy's cast, an integer wrapped at its width), no program traced."""
    kind = ShapedArray((), np.dtype(dtype))
    if np.result_type(value) == kind.dtype:
        return np.asarray(value).reshape(-1)[0]
    folded = partial_eval.const_fold_rules[jax.lax.convert_element_type_p](
        [np.asarray(value)], {"new_dtype": kind.dtype, "weak_type": False, "sharding": None}, [kind])
    # JAX's rule folds every host scalar of a numeric dtype, none of an extended one, which none admits.
    if folded is None:
        raise TypeError(f"JAX folds no conversion of {value!r} to {kind.dtype}")
    return np.asarray(folded[0]).reshape(-1)[0]


def scalar_folded(primitive: jcore.Primitive, operands: Sequence[np.generic],
                  params: Mapping[str, object], dtype: jax.typing.DTypeLike) -> np.generic | None:
    """An elementwise `primitive`'s output, of `dtype`, over its operands, concrete integer and boolean
    scalars each at its own dtype (`admitted`: a program's own, a kernel's index held at int64 under its
    body's x64 scope stays int64), as binding it computes it, exactly, in this host's own integers
    (`_SCALAR`), with no program traced, lowered or compiled: a width's wrap, a quotient truncated and a
    remainder of its dividend's sign (by zero, and of the least integer by -1, as XLA defines them), a shift
    past the width at its first operand's width, and the result wrapped to `dtype`, the type its caller's
    backend gives it: XLA's, a stated output dtype's, its operands cast to it first (`_host_folded`); Mosaic
    GPU's Lane lowering, its first operand's, reading no stated dtype (`emission._folded`); a conversion as
    JAX folds a constant's (`_converted`). A quotient by zero is every bit set and a remainder by zero its
    dividend, and a signed least integer's quotient by -1 is itself and its remainder 0, as XLA's elemental
    emitter guards them; a caller lowering otherwise (a Mosaic kernel's native division, which leaves them
    undefined) refuses them first (`emission`'s). None for a primitive outside the arithmetic folded
    (`SCALAR_FOLDED`), a parameter it does not read (`_READ`), and a floating or complex output."""
    if (primitive not in SCALAR_FOLDED or not _integral(dtype)
            or any(value is not None for name, value in params.items() if name not in _READ)):
        return None
    if primitive is jax.lax.convert_element_type_p:
        return _converted(operands[0], dtype)
    values = [int(value) for value in operands]
    return np.dtype(dtype).type(_wrapped(_SCALAR[primitive](operands[0].dtype, *values), dtype))


def _wrapped(value: int, dtype) -> int | bool:
    """`value` as `dtype` holds it: a boolean's truth, an integer's two's complement at its width."""
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return value != 0
    limits = jnp.iinfo(dtype)
    value &= (1 << limits.bits) - 1
    return value - (1 << limits.bits) if value > limits.max else value


def _quotient(dividend: int, divisor: int) -> int:
    """An integer quotient truncated toward zero, unwrapped, by zero -1 (every bit set once `_wrapped`: an
    unsigned width's largest), as XLA defines it."""
    if divisor == 0:
        return -1
    quotient = abs(dividend) // abs(divisor)
    return quotient if (dividend < 0) == (divisor < 0) else -quotient


def _remainder(dividend: int, divisor: int) -> int:
    """An integer remainder of its dividend's sign, by zero the dividend, as XLA defines it."""
    return dividend if divisor == 0 else dividend - divisor * _quotient(dividend, divisor)


def _shifted(primitive, held, value: int, amount: int) -> int:
    """`value`, of `held`'s width, shifted by `amount`, read unsigned: past the width a left or logical
    shift leaves nothing and an arithmetic one its sign, which it reads of the value as signed (an
    unsigned value's too)."""
    bits = jnp.iinfo(held).bits
    amount &= (1 << bits) - 1
    signed = value & ((1 << bits) - 1)
    signed -= (1 << bits) if signed >> (bits - 1) else 0
    if primitive is jax.lax.shift_right_arithmetic_p:
        return signed >> min(amount, bits - 1)
    if amount >= bits:
        return 0
    if primitive is jax.lax.shift_left_p:
        return value << amount
    return (value & ((1 << bits) - 1)) >> amount


_SCALAR: dict[jcore.Primitive, Callable[..., int | bool]] = {
    jax.lax.add_p: lambda held, a, b: a + b,
    jax.lax.sub_p: lambda held, a, b: a - b,
    jax.lax.mul_p: lambda held, a, b: a * b,
    jax.lax.neg_p: lambda held, a: -a,
    jax.lax.div_p: lambda held, a, b: _quotient(a, b),
    jax.lax.rem_p: lambda held, a, b: _remainder(a, b),
    jax.lax.max_p: lambda held, a, b: max(a, b),
    jax.lax.min_p: lambda held, a, b: min(a, b),
    jax.lax.clamp_p: lambda held, low, a, high: min(max(a, low), high),
    jax.lax.lt_p: lambda held, a, b: a < b,
    jax.lax.le_p: lambda held, a, b: a <= b,
    jax.lax.gt_p: lambda held, a, b: a > b,
    jax.lax.ge_p: lambda held, a, b: a >= b,
    jax.lax.eq_p: lambda held, a, b: a == b,
    jax.lax.ne_p: lambda held, a, b: a != b,
    jax.lax.and_p: lambda held, a, b: a & b,
    jax.lax.or_p: lambda held, a, b: a | b,
    jax.lax.xor_p: lambda held, a, b: a ^ b,
    jax.lax.not_p: lambda held, a: not a if held == np.bool_ else ~a,
    jax.lax.shift_left_p: functools.partial(_shifted, jax.lax.shift_left_p),
    jax.lax.shift_right_arithmetic_p: functools.partial(_shifted, jax.lax.shift_right_arithmetic_p),
    jax.lax.shift_right_logical_p: functools.partial(_shifted, jax.lax.shift_right_logical_p),
}
"""Each of `SCALAR_FOLDED`'s primitives over its operands as Python's integers (a boolean's 0 or 1), the
dtype they are held at first: its result before `scalar_folded` wraps it to the output's dtype."""

SCALAR_FOLDED = frozenset({*_SCALAR, jax.lax.convert_element_type_p})
"""The elementwise primitives `scalar_folded` computes over concrete integer and boolean scalars: a
kernel's index arithmetic's and a program's integer constants', a conversion as JAX folds one
(`_converted`)."""

_READ = frozenset({"new_dtype", "out_dtype", "weak_type", "sharding", "out_sharding"})
"""The parameters `scalar_folded` folds with: a conversion's target and weak type, a result's stated dtype
(its caller's `dtype`), and a placement, which states nothing of a scalar's value; a primitive given any
other one set is folded by none."""

def _integral(dtype) -> bool:
    """Whether `dtype` is an integer or boolean one (JAX's 4-bit integers among them)."""
    return jnp.issubdtype(dtype, jnp.integer) or jnp.issubdtype(dtype, jnp.bool_)


def _extended(value) -> bool:
    """Whether `value` is of an extended dtype (a PRNG key's)."""
    return jnp.issubdtype(getattr(value, "dtype", None) or jnp.result_type(value), jax.dtypes.extended)


@contextlib.contextmanager
def meshless(mesh: jax.sharding.Mesh | None = None) -> Iterator[None]:
    """No mesh in context, abstract or concrete, but `mesh` where given, nor its axes bound or checked, under
    a trace too: a program begun within another's trace (a shard_map's body, whose mesh JAX refuses to
    resize, `jax.sharding.use_abstract_mesh`, and whose manual axes a checked shard_map binds and asks its
    values to state their variance over, `check_vma`) takes none of its mesh, `mesh` as a program set on it
    takes it (`jax.set_mesh`: its jits record it as theirs). `jax.set_mesh` does the mesh's swaps but refuses
    to run under a trace."""
    abstract = mesh_lib.abstract_mesh_context_manager.swap_local(
        mesh_lib.empty_abstract_mesh if mesh is None else mesh.abstract_mesh)
    concrete = mesh_lib.device_context.swap_local(mesh_lib.empty_concrete_mesh if mesh is None else mesh)
    axes = jax_core.axis_env_state.swap_local(jax_core.top_axis_env)
    try:
        with jax_config._check_vma(new_val=False):
            yield
    finally:
        jax_core.axis_env_state.set_local(axes)
        mesh_lib.device_context.set_local(concrete)
        mesh_lib.abstract_mesh_context_manager.set_local(abstract)


@contextlib.contextmanager
def on_host() -> Iterator[None]:
    """Compile-time evaluation on this host, no mesh in context (`meshless`): a step traced under its
    devices' mesh (`jax.set_mesh`, a compile-only TPU topology's, a multi-process one's) would place an eager
    primitive's operands on devices that run nothing here."""
    with meshless():
        # This process's own device: its CPU, or, where it runs no CPU backend (`JAX_PLATFORMS=cuda`), its
        # first. Its own: the pool's first device may be another process's, which this one cannot read back.
        host = jax.local_devices(backend="cpu" if "cpu" in backends() else None)[0]
        with jax.ensure_compile_time_eval(), jax.default_device(host):
            yield


def on_tile(jaxpr: jcore.Jaxpr, consts, logical: tuple, tile: tuple, *args) -> list:
    """`jaxpr`, traced at the `logical` shape, evaluated on `args` of the `tile` shape as
    `jax.core.eval_jaxpr` evaluates it: each equation bound by its primitive (`get_bind_params`), a shape
    parameter that is the logical shape the tile's, and a program a parameter holds (a jit's, a custom
    derivative's primal) retraced at the tile's shapes, the equation kept. Its values alone: a caller states
    the derivative (`dew.nn.kernels.norm_swiglu`'s `_Logical`, a row kernel's epilogue on a tile of a row's
    columns; `attention_mosaic_program.BlockPlan.sliced`, an output hook's program on a block of its tile's
    columns)."""
    values = dict(zip(jaxpr.constvars, consts, strict=True)) | dict(zip(jaxpr.invars, args, strict=True))

    def read(atom):
        return atom.val if isinstance(atom, jcore.Literal) else values[atom]

    def shaped(value):
        return tuple(tile) if isinstance(value, tuple) and tuple(value) == logical else value

    def retraced(program: jcore.Jaxpr):
        avals = [jax.ShapeDtypeStruct(tile if aval.shape == logical else aval.shape, aval.dtype)
                 for aval in program.in_avals if isinstance(aval, ShapedArray)]
        return jax.make_jaxpr(
            lambda *operands: on_tile(program, program.consts, logical, tile, *operands))(*avals)

    for equation in jaxpr.eqns:
        given = [read(atom) for atom in equation.invars]
        if equation.primitive is jax.lax.cond_p and not isinstance(given[0], Tracer):
            # A conditional whose index is known (a predicate the activation closes over, a constant in its
            # graph) is the branch it names, run on the tile: Pallas's Triton lowering takes a traced index
            # alone (its rule reads the index's MLIR type; a constant has none).
            branch = equation.params["branches"][int(np.asarray(given[0]))]
            values.update(zip(equation.outvars, on_tile(branch.jaxpr, branch.consts, logical, tile,
                                                        *given[1:]), strict=True))
            continue
        # Each program it runs within it, a call's or each branch's (`programs.with_bodies`), retraced.
        params = with_bodies({key: shaped(value) for key, value in equation.params.items()}, retraced)
        with equation.ctx.manager:
            made = equation.primitive.bind(*given, **equation.primitive.get_bind_params(params))
        outputs = made if equation.primitive.multiple_results else [made]
        values.update(zip(equation.outvars, outputs, strict=True))
    return [read(atom) for atom in jaxpr.outvars]
