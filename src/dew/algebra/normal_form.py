"""The normal form recognition compares programs in (docs/design/algebra.md §2, "Recognition"; row 4783):
`canonical(closed)`, a closed program with its nested programs inlined, its constants of masks and divisors
folded on this host (`dew.nn.programs.host_evaluated`), and each equation in the one form every spelling of
the same arithmetic takes, bit for bit: a square a product, a correctly rounded narrow float's arithmetic its
float32 arithmetic rounded once, `x - 0` and `x / 1` as `x`, axes of one element erased and implicit
broadcasts laid out, an exact widening rounded back its source, equal equations once. Every identity holds for
every value, a zero's sign, infinities and NaNs among them; anything else stays as written.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Hashable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core as jcore

from dew.algebra import terms
from dew.nn.programs import host_evaluated, jax_name, on_host


def canonical(closed: jcore.ClosedJaxpr) -> jcore.ClosedJaxpr:
    """`closed` as recognition compares it: its nested programs inlined (a jit, a closed call, a custom
    derivative's primal), each equation in its normal form (`_normal`: a square a product, a narrow float's
    arithmetic its float32 arithmetic rounded once), each value it makes from no input computed into a
    constant, every constant a
    literal of its value (so a mask the program builds and one a declaration builds are one where they are
    the same booleans), and each equation that repeats another's primitive, parameters and operands, and
    holds no effect, read as that one."""
    flat = jax.make_jaxpr(functools.partial(_evaluated, closed.jaxpr, closed.consts))(
        *(jax.ShapeDtypeStruct(var.aval.shape, var.aval.dtype) for var in closed.jaxpr.invars))
    jaxpr = flat.jaxpr
    local: dict[jcore.Var, jcore.Var | jcore.Literal] = {
        var: jcore.Literal(np.asarray(value), var.aval) for var, value in zip(jaxpr.constvars, flat.consts,
                                                                             strict=True)}

    def read(atom):
        return local.get(atom, atom) if isinstance(atom, jcore.Var) else atom

    seen: dict[Hashable, list] = {}
    equations = []
    for equation in jaxpr.eqns:
        equation = equation.replace(invars=[read(atom) for atom in equation.invars])
        named = _equation_key(equation)
        if named is not None and named in seen:
            local.update(zip(equation.outvars, seen[named], strict=True))
            continue
        if named is not None:
            seen[named] = list(equation.outvars)
        equations.append(equation)
    outvars = [read(atom) for atom in jaxpr.outvars]
    return jcore.ClosedJaxpr(jaxpr.replace(constvars=[], eqns=_live(equations, outvars), outvars=outvars), [])


def _live(equations: list[jcore.JaxprEqn], outvars: list) -> list[jcore.JaxprEqn]:
    """The equations an output reads, in order: a layout the normal form left unread (a value's form before
    its unit axes were erased) is no part of what the program computes. An effect is read by definition."""
    wanted = {atom for atom in outvars if isinstance(atom, jcore.Var)}
    kept = []
    for equation in reversed(equations):
        if equation.effects or wanted & set(equation.outvars):
            kept.append(equation)
            wanted.update(atom for atom in equation.invars if isinstance(atom, jcore.Var))
    return kept[::-1]


BODY_PARAMS = {"jit": "jaxpr", "pjit": "jaxpr", "closed_call": "call_jaxpr", "core_call": "call_jaxpr",
               "custom_jvp_call": "call_jaxpr", "custom_vjp_call": "call_jaxpr",
               "custom_vjp_call_jaxpr": "fun_jaxpr", "remat2": "jaxpr", "checkpoint": "jaxpr"}
"""The nested programs `_evaluated` runs in place, by the parameter holding each one's body: what its call
computes, the primal of a custom derivative's. A program's own derivative rule bars its region (`BARRIERS`);
a declaration's is the call's, kept where the call is written."""


class _Made:
    """A record of values `_evaluated` made, each by its identity, the value held with its entry so no other
    value takes its identity while the record lasts."""

    __slots__ = ("held",)

    def __init__(self):
        self.held: dict[int, tuple] = {}

    def __contains__(self, value) -> bool:
        return id(value) in self.held

    def __getitem__(self, value):
        return self.held[id(value)][1]

    def get(self, value):
        found = self.held.get(id(value))
        return None if found is None else found[1]

    def __setitem__(self, value, entry) -> None:
        self.held[id(value)] = (value, entry)


class _Forms(NamedTuple):
    """What `_evaluated` knows of the values it has made (`_Made`): each exact widening's source (`widened`),
    each layout's source and layout (`laid`), each value laid out from its erased form by unit axes, that form
    (`squeezed`). One for a whole program, its nested programs' among it: a value crosses a nested program's
    boundary as itself."""

    widened: _Made
    laid: _Made
    squeezed: _Made


def _evaluated(jaxpr: jcore.Jaxpr, consts: Sequence, *values, forms: _Forms | None = None) -> list:
    """`jaxpr` run over `values` with every nested program (`BODY_PARAMS`) run in place, each equation in the
    form recognition compares (`_normal`): under a trace its equations each bind once, a value of constants
    alone computed where it is made."""
    env: dict[jcore.Var, object] = dict(zip(jaxpr.constvars, consts, strict=True))
    env.update(zip(jaxpr.invars, values, strict=True))
    forms = forms or _Forms(_Made(), _Made(), _Made())
    widened, laid, squeezed = forms

    def read(atom):
        return atom.val if isinstance(atom, jcore.Literal) else env[atom]

    for equation in jaxpr.eqns:
        operands = [read(atom) for atom in equation.invars]
        body = equation.params.get(BODY_PARAMS.get(equation.primitive.name, ""))
        if body is not None:
            closed = body if isinstance(body, jcore.ClosedJaxpr) else jcore.ClosedJaxpr(body, ())
            given = operands[len(operands) - len(closed.jaxpr.invars):]
            outputs = _evaluated(closed.jaxpr, closed.consts, *given, forms=forms)
        else:
            # A value of constants alone is computed where it is made, so the program and a declaration
            # hold one constant (a mean of zeros, a mask of iotas) however each built it: of the equations
            # such a constant is built from (`FOLDED`), on this host, never one that places a program on a
            # mesh, which binds as traced. An equation over a traced value stays traced; the constants its
            # normal form builds beside it (a broadcast divisor) are computed.
            traced = any(isinstance(value, jax.core.Tracer) for value in operands)
            folded = None if traced or jax_name(equation.primitive) not in FOLDED else host_evaluated(
                equation.primitive, operands, equation.params)
            if folded is not None:
                outputs = folded
            else:
                # A value this host cannot compute (one it cannot read whole, a key) stays an equation.
                with jax.ensure_compile_time_eval() if traced else contextlib.nullcontext():
                    outputs = (_unit_round_trip(equation, operands, squeezed)
                               or _erased(equation, operands, widened, laid, squeezed)
                               or _normal(equation, operands, widened, laid))
        env.update(zip(equation.outvars, outputs, strict=True))
    return [read(atom) for atom in jaxpr.outvars]


FOLDED = frozenset({"iota", "broadcast_in_dim", "reshape", "transpose", "squeeze", "expand_dims",
                    "concatenate", "slice", "convert_element_type", "select_n", "eq", "ne", "lt", "le", "gt",
                    "ge", "and", "or", "not", "add", "sub", "mul", "div", "max", "min", "neg", "abs",
                    "reduce_sum", "reduce_max", "reduce_min", "integer_pow", "square", "sqrt", "rsqrt"})
"""The equations `_evaluated` computes where their operands are constants: those a mask, a divisor or a table
is built from. Any other binds as traced, a collective or a program placed on a mesh among them, whose
eager run needs devices a trace has not."""


VALUE_IDENTITIES = frozenset({"name"})
"""The equations whose value is their operand's (a checkpoint's name), which `_normal` reads through: what
they change is a schedule, never the value compared. A gradient's stop is no identity (`BARRIERS`)."""


NARROW = {jnp.dtype(jnp.bfloat16): (8, 7), jnp.dtype(jnp.float16): (5, 10)}
"""The narrow floats, by the exponent and mantissa bits `reduce_precision` rounds a float32 to them by."""


FULL_RANGE = frozenset({jnp.dtype(jnp.bfloat16)})
"""The narrow floats holding float32's exponent range, whose `reduce_precision` is a round trip through them:
a float16's underflows to zero where the cast keeps a subnormal (StableHLO's reduce_precision)."""


ROUNDED_ONCE = frozenset({"add", "sub", "mul", "div", "max", "min", "neg", "abs", "sqrt"})
"""The elementwise primitives whose narrow-float equation is its float32 one rounded once (`_normal`): IEEE's
correctly rounded arithmetic, which float32 holds at least twice a bfloat16's or float16's digits and two
more for, so a float32 result rounded again rounds as the narrow operation does. No transcendental: a
backend computes a narrow one its own way (XLA:GPU's logistic in the operand's dtype)."""


def _normal(equation: jcore.JaxprEqn, operands: list, widened: _Made,
            laid: _Made) -> list:
    """`equation` bound over `operands` in the form recognition compares (`canonical`): a square a value
    times itself, bit for bit; a narrow float's elementwise equation (`ROUNDED_ONCE`) its float32 one
    rounded once to the narrow dtype; a gradient's stop or a checkpoint's name its operand
    (`VALUE_IDENTITIES`); a float32 rounded to a narrow float's digits in place
    (`reduce_precision`) a round trip through that dtype; a value widened exactly and rounded back to its
    own dtype that value itself (`widened`, each exact widening's result by its source); a value less a
    constant zero that value; a reshape that only adds unit axes the broadcast it is; and an elementwise
    equation over values one layout made from equal shapes, beside scalars, that equation before the
    layout (`laid`, each laid value by its source and layout). Every other equation as it is."""
    name = jax_name(equation.primitive)
    if name in VALUE_IDENTITIES:
        return list(operands)
    if name == "sub" and _zero(operands[1]) and jnp.shape(operands[0]) == equation.outvars[0].aval.shape:
        return [operands[0]]
    if (name in ("div", "mul") and _one(operands[1])
            and jnp.shape(operands[0]) == equation.outvars[0].aval.shape):
        return [operands[0]]
    if name in ("broadcast_in_dim", "reshape"):
        return [_laid(equation, operands[0], laid)]
    if name in ELEMENTWISE and len(equation.outvars) == 1:
        lifted = _lifted(equation, operands, widened, laid)
        if lifted is not None:
            return [lifted]
    if name == "convert_element_type":
        return [_converted(operands[0], equation.params["new_dtype"], widened)]
    if name == "square" or (name == "integer_pow" and equation.params.get("y") == 2):
        return [jax.lax.mul(operands[0], operands[0])]
    if name == "reduce_precision":
        bits = (equation.params["exponent_bits"], equation.params["mantissa_bits"])
        narrow = next((dtype for dtype, held in NARROW.items() if held == bits and dtype in FULL_RANGE), None)
        # A round trip through the narrow float from float32 alone, whose exponent range it holds: a wider
        # float's reduce_precision flushes what the narrow float keeps as a subnormal.
        if narrow is not None and jnp.result_type(operands[0]) == jnp.float32:
            return [_converted(_converted(operands[0], narrow, widened), operands[0].dtype, widened)]
    if name in ROUNDED_ONCE and len(equation.outvars) == 1 and any(map(_uniform, operands)):
        shape = equation.outvars[0].aval.shape
        if all(_uniform(value) or jnp.shape(value) == shape for value in operands):
            # A constant of one value throughout reads, element by element, as that value: the equation over
            # the scalar computes the same elements, so a mean's divisor held as an array is its scalar's.
            scalars = [_scalar(value) if _uniform(value) else value for value in operands]
            if any(jnp.shape(value) == shape for value in scalars):
                return _normal(equation, scalars, widened, laid)
    if name == "dot_general":
        return [equation.primitive.bind(*operands, **_stated(equation.params, operands))]
    out = equation.outvars[0].aval.dtype if len(equation.outvars) == 1 else None
    if name in ROUNDED_ONCE and out in NARROW and all(jnp.result_type(value) == out for value in operands):
        wide = [_converted(value, jnp.float32, widened) for value in operands]
        return [_converted(equation.primitive.bind(*wide, **equation.params), out, widened)]
    bound = equation.primitive.bind(*operands, **equation.params)
    return bound if equation.primitive.multiple_results else [bound]


OWN_PRESETS = {jnp.dtype(jnp.bfloat16): "BF16_BF16_F32", jnp.dtype(jnp.float16): "F16_F16_F32"}
"""Each 16-bit float's own dot algorithm: its products exact and summed in float32."""


def _stated(params: dict, operands: Sequence) -> dict:
    """A product's `params` with its precision as the arithmetic it states: two operands of one 16-bit
    float summed to float32 are exact products summed in float32 under any `jax.lax.Precision` (which
    chooses the passes of wider operands alone), the default algorithm or the dtype's own algorithm
    (`OWN_PRESETS`, jax.nn's attention names it), so each reads as no precision; any other product keeps
    the precision it states."""
    dtypes = {jnp.result_type(value) for value in operands}
    precision = params.get("precision")
    if (len(dtypes) != 1 or (dtype := dtypes.pop()) not in OWN_PRESETS
            or params.get("preferred_element_type") != jnp.float32):
        return params
    enums = precision if isinstance(precision, tuple) else (precision,)
    if all(part is None or isinstance(part, jax.lax.Precision) for part in enums) or (
            isinstance(precision, jax.lax.DotAlgorithmPreset)
            and precision.name in ("DEFAULT", OWN_PRESETS[dtype])):
        return {**params, "precision": None}
    return params


UNIT_ERASED = frozenset({"dot_general", "transpose", "reduce_sum", "reduce_max", "reduce_min", "reshape",
                         "broadcast_in_dim"})
"""The layout-reading equations `_erased` computes over their operands' axes of more than one element."""


def _erased(equation: jcore.JaxprEqn, operands: list, widened: _Made, laid: _Made,
            squeezed: _Made) -> list | None:
    """An equation over values with axes of one element computed over those values with every such axis
    erased (`_squeeze`), then laid back out to its shape: a size-1 axis holds the same elements wherever
    a layout puts it, so a statistic kept `[N, 1]` and one held `[N]` are one computation. The equation's
    axes are renumbered over the erased ones: a product's batch and contracting pairs of one element and a
    reduction over one dropped, a transpose's order kept among the rest. None for an equation that reads
    no layout (`UNIT_ERASED`, `ELEMENTWISE`) or holds no such axis, which `_normal` computes as it is."""
    name = jax_name(equation.primitive)
    if len(equation.outvars) != 1 or not (name in UNIT_ERASED or name in ELEMENTWISE):
        return None
    out = equation.outvars[0].aval
    shapes = [jnp.shape(value) for value in operands]
    if 1 not in out.shape and not any(1 in shape for shape in shapes):
        return None
    kept = [[axis for axis, size in enumerate(shape) if size != 1] for shape in shapes]
    renumbered = [{axis: place for place, axis in enumerate(axes)} for axes in kept]
    inner = [_squeeze(value, squeezed) for value in operands]
    params = dict(equation.params)
    if name == "dot_general":
        params = _stated(params, operands)
        (lc, rc), (lb, rb) = params["dimension_numbers"]
        left, right = renumbered
        contracting = [(a, b) for a, b in zip(lc, rc, strict=True) if a in left]
        batch = [(a, b) for a, b in zip(lb, rb, strict=True) if a in left]
        params["dimension_numbers"] = ((tuple(left[a] for a, _ in contracting),
                                        tuple(right[b] for _, b in contracting)),
                                       (tuple(left[a] for a, _ in batch), tuple(right[b] for _, b in batch)))
    elif name == "transpose":
        order = [axis for axis in params["permutation"] if axis in renumbered[0]]
        params["permutation"] = tuple(renumbered[0][axis] for axis in order)
        if params["permutation"] == tuple(range(len(order))):
            # A transpose moving only axes of one element keeps every element where it was.
            return [_unsqueezed(inner[0], out.shape, squeezed)]
    elif name.startswith("reduce_"):
        params["axes"] = tuple(renumbered[0][axis] for axis in params["axes"] if axis in renumbered[0])
        if not params["axes"]:
            if name != "reduce_sum":
                return [_unsqueezed(inner[0], out.shape, squeezed)]
            # A sum of one element is its value plus the sum's zero, which makes a negative zero positive:
            # no identity, computed as written.
            return None
    elif name == "broadcast_in_dim":
        target = [size for size in out.shape if size != 1]
        at = {axis: place for place, axis in enumerate(a for a, size in enumerate(out.shape) if size != 1)}
        params["shape"] = tuple(target)
        dimensions = tuple(at[params["broadcast_dimensions"][axis]] for axis in kept[0])
        if list(dimensions) != sorted(dimensions):
            # A broadcast that also reorders its operand's axes is another layout: computed as it is.
            return None
        params["broadcast_dimensions"] = dimensions
    elif name == "reshape":
        if params.get("dimensions") is not None:
            return None
        target = tuple(size for size in out.shape if size != 1)
        core = inner[0] if jnp.shape(inner[0]) == target else jax.lax.reshape(inner[0], target)
        return [_unsqueezed(core, out.shape, squeezed)]
    elif any(jnp.shape(value) != () and jnp.shape(value) != out.shape for value in operands):
        if any(jnp.ndim(value) not in (0, len(out.shape)) for value in operands):
            return None
        # An operand of one element along an axis the result spans is broadcast along it, as the
        # equation reads it: laid out to the result's axes of more than one element (`_broadcast`).
        inner = [_broadcast(value, out.shape, squeezed) if jnp.ndim(value) else value for value in operands]
    core_shape = jax.eval_shape(lambda *values: equation.primitive.bind(*values, **params), *inner).shape
    inner_equation = equation.replace(params=params, outvars=[jcore.Var(out.update(shape=core_shape))])
    core = _normal(inner_equation, inner, widened, laid)[0]
    return [_unsqueezed(core, out.shape, squeezed)]


def _broadcast(value, shape: tuple[int, ...], squeezed: _Made):
    """`value`, of the rank of `shape` and one element along each axis it does not share, broadcast to
    `shape`'s axes of more than one element over its erased form (`_squeeze`)."""
    if jnp.shape(value) == tuple(shape):
        return _squeeze(value, squeezed)
    kept = [axis for axis, size in enumerate(jnp.shape(value)) if size != 1]
    targets = [axis for axis, size in enumerate(shape) if size != 1]
    at = {axis: place for place, axis in enumerate(targets)}
    with _beside(value):
        return jax.lax.broadcast_in_dim(_squeeze(value, squeezed), tuple(shape[axis] for axis in targets),
                                        tuple(at[axis] for axis in kept))


def _beside(value) -> contextlib.AbstractContextManager:
    """Where an operation on `value` runs: a constant's on this host (`programs.on_host`), a constant a normal
    form builds beside a traced equation computed where nothing places it on the trace's mesh (a
    compile-only GPU's, whose devices run nothing here); a traced value's in its trace."""
    return contextlib.nullcontext() if isinstance(value, jax.core.Tracer) else on_host()


def _squeeze(value, squeezed: _Made):
    """`value` with its axes of one element erased: the erased value it was laid out from where it was
    (`squeezed`), else its reshape, once a value."""
    shape = jnp.shape(value)
    if 1 not in shape:
        return value
    if value in squeezed:
        return squeezed[value]
    target = tuple(size for size in shape if size != 1)
    made = (np.reshape(np.asarray(value), target) if not isinstance(value, jax.core.Tracer)
            else jax.lax.reshape(value, target))
    squeezed[value] = made
    return made


def _unit_round_trip(equation: jcore.JaxprEqn, operands: list, squeezed: _Made) -> list | None:
    """A reshape back to the shape of the erased value its operand was laid out from by axes of one element
    (`_unsqueezed`), that value: the axes added and taken away move no element; None for any other
    equation."""
    if jax_name(equation.primitive) != "reshape" or equation.params.get("dimensions") is not None:
        return None
    core = squeezed.get(operands[0])
    return [core] if core is not None and jnp.shape(core) == equation.outvars[0].aval.shape else None


def _unsqueezed(core, shape: tuple[int, ...], squeezed: _Made):
    """`core` laid out to `shape`, its axes of one element restored, recorded as `core`'s (`squeezed`)."""
    if jnp.shape(core) == tuple(shape):
        return core
    made = jax.lax.reshape(core, tuple(shape))
    squeezed[made] = core
    return made


ELEMENTWISE = ROUNDED_ONCE | {"convert_element_type", "select_n", "integer_pow", "square", "sign", "floor",
                              "ceil", "round", "pow", "atan2", "rem", "exp", "exp2", "log", "log1p", "expm1",
                              "logistic", "tanh", "erf", "sin", "cos", "rsqrt", "is_finite", "eq", "ne", "lt",
                              "le", "gt", "ge", "and", "or", "not", "stop_gradient"}
"""The equations computed element by element, which a layout of their operands commutes with exactly,
whatever each element's arithmetic (a transcendental's included: the same inputs give the same outputs
wherever a layout puts them)."""


def _uniform(value) -> bool:
    """Whether `value` is a constant (computed, `_evaluated`) of more than one element, all one value bit for
    bit (a zero's sign among its bits)."""
    if isinstance(value, jax.core.Tracer) or jnp.ndim(value) == 0:
        return False
    held = np.ascontiguousarray(value)
    if not held.size:
        return False
    bits = held.view(f"V{held.dtype.itemsize}")
    return bool(np.all(bits == bits.flat[0]))


def _scalar(value) -> np.ndarray:
    """A uniform constant (`_uniform`) as its one value, in its dtype."""
    return np.asarray(value).flat[0].astype(np.asarray(value).dtype)


def _one(value) -> bool:
    """Whether `value` is a constant (computed, `_evaluated`) of ones alone: `x / 1` and `x * 1` are `x` for
    every `x`, its sign, infinities and NaNs among them."""
    return not isinstance(value, jax.core.Tracer) and bool(np.all(np.asarray(value) == 1))


def _zero(value) -> bool:
    """Whether `value` is a constant (computed, `_evaluated`) of positive zeros alone: `x - 0` is `x` for
    every `x`, where `x - (-0)` makes a negative zero positive."""
    if isinstance(value, jax.core.Tracer):
        return False
    held = np.asarray(value)
    return not np.any(held) and not np.any(np.signbit(held))


def _laid(equation: jcore.JaxprEqn, value, laid: _Made):
    """A broadcast's or a reshape's result, a reshape that only adds unit axes written as the broadcast it
    is, recorded in `laid` by its source and layout (`_lifted`)."""
    shape = equation.outvars[0].aval.shape
    if jax_name(equation.primitive) == "reshape":
        ordered = equation.params.get("dimensions") is None
        dimensions = _unit_axes(jnp.shape(value), shape) if ordered else None
        if dimensions is None:
            return equation.primitive.bind(value, **equation.params)
    else:
        dimensions = tuple(equation.params["broadcast_dimensions"])
    made = jax.lax.broadcast_in_dim(value, shape, dimensions)
    if made is not value and isinstance(made, jax.core.Tracer):
        laid[made] = (value, (shape, dimensions))
    return made


def _unit_axes(before: tuple[int, ...], after: tuple[int, ...]) -> tuple[int, ...] | None:
    """Where `before`'s axes lie in `after` where `after` is `before` with unit axes added; else None."""
    dimensions, at = [], 0
    for place, size in enumerate(after):
        if at < len(before) and size == before[at]:
            dimensions.append(place)
            at += 1
        elif size != 1:
            return None
    return tuple(dimensions) if at == len(before) else None


def _lifted(equation: jcore.JaxprEqn, operands: list, widened: _Made, laid: _Made):
    """An elementwise equation whose every operand but scalars one layout made, from sources of one shape:
    the equation over the sources, then the layout; None where its operands are laid otherwise."""
    layouts = {laid[value][1] for value in operands if value in laid}
    sources = [laid[value][0] if value in laid else value for value in operands]
    if len(layouts) != 1 or any(value not in laid and jnp.ndim(value) != 0 for value in operands):
        return None
    if len({jnp.shape(source) for source, value in zip(sources, operands, strict=True)
            if value in laid}) != 1:
        return None
    (shape, dimensions), = layouts
    inner = equation.replace(outvars=[jcore.Var(equation.outvars[0].aval.update(
        shape=jnp.shape(next(source for source, value in zip(sources, operands, strict=True)
                             if value in laid))))])
    made = _normal(inner, sources, widened, laid)[0]
    spread = jax.lax.broadcast_in_dim(made, shape, dimensions)
    laid[spread] = (made, (shape, dimensions))
    return spread


def _converted(value, dtype, widened: _Made):
    """`value` cast to `dtype`: the value it was widened from where it is an exact widening of a value of
    `dtype` (`widened`), else the cast, recorded in `widened` where it widens exactly (a narrow float to
    float32, a float to float64)."""
    dtype = jnp.dtype(dtype)
    source = widened.get(value)
    if source is not None and jnp.result_type(source) == dtype:
        return source
    with _beside(value):
        cast = jax.lax.convert_element_type(value, dtype)
    if cast is not value and jnp.issubdtype(dtype, jnp.floating) and jnp.issubdtype(jnp.result_type(value),
                                                                                    jnp.floating) and (
            jnp.finfo(dtype).nmant > jnp.finfo(jnp.result_type(value)).nmant
            and jnp.finfo(dtype).maxexp >= jnp.finfo(jnp.result_type(value)).maxexp):
        widened[cast] = value
    return cast


def _equation_key(equation: jcore.JaxprEqn) -> Hashable | None:
    """An equation's identity as its value's (`canonical`): its primitive, parameters and operands, each
    literal by its value; None for one holding an effect, whose repetition is not its value's."""
    if equation.effects:
        return None
    operands = tuple(terms.literal_key(atom) if isinstance(atom, jcore.Literal) else atom
                     for atom in equation.invars)
    return (*terms.parameters_key(equation.primitive, equation.params), operands)
