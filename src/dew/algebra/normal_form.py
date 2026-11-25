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
import itertools
from collections.abc import Callable, Hashable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as jax_core
from jax.extend import core as jcore
from jax.extend.core import primitives

from dew.algebra import terms
from dew.algebra.forms import aval
from dew.nn.derivatives import marked_call_p, stated_operation
from dew.nn.programs import SCALAR_FOLDED, called, entrywise, host_evaluated, live, on_host, own_preset


def canonical(closed: jcore.Jaxpr) -> jcore.Jaxpr:
    """`closed` as recognition compares it: its nested programs inlined (a jit, a closed call, a custom
    derivative's primal), each equation in its normal form (`_normal`: a square a product, a narrow float's
    arithmetic its float32 arithmetic rounded once), each value it makes from no input computed into a
    constant, every constant a
    literal of its value (so a mask the program builds and one a declaration builds are one where they are
    the same booleans), and each equation that repeats another's primitive, parameters and operands, and
    holds no effect, read as that one."""
    flat = jax.make_jaxpr(functools.partial(_evaluated, closed.jaxpr, closed.consts))(
        *(jax.ShapeDtypeStruct(aval(var).shape, aval(var).dtype) for var in closed.jaxpr.invars))
    jaxpr = flat.jaxpr
    # A typed PRNG key stays the key it is: NumPy holds none (`terms.typed_key`).
    local: dict[jcore.Var, jcore.Var | jcore.Literal] = {
        var: jcore.Literal(value if terms.typed_key(value) else np.asarray(value), var.aval)
        for var, value in zip(jaxpr.constvars, flat.consts, strict=True)}

    def read(atom):
        return local.get(atom, atom) if isinstance(atom, jcore.Var) else atom

    seen: dict[Hashable, list] = {}
    # Each literal's key once (its value's bytes, `terms.literal_key`), by the value's identity, which the
    # literals above hold for the call: a folded mask a region's every select reads hashed once, where each
    # equation that read it hashed its megabytes again (1,070 hashes, 6.5 s of a start's pass, row 6799).
    literal_keys: dict[int, Hashable] = {}

    def keyed(atom: jcore.Literal) -> Hashable:
        if (found := literal_keys.get(id(atom.val))) is None:
            found = literal_keys[id(atom.val)] = terms.literal_key(atom)
        return found

    equations = []
    for equation in jaxpr.eqns:
        equation = equation.replace(invars=[read(atom) for atom in equation.invars])
        named = _equation_key(equation, keyed)
        if named is not None and named in seen:
            local.update(zip(equation.outvars, seen[named], strict=True))
            continue
        if named is not None:
            seen[named] = list(equation.outvars)
        equations.append(equation)
    outvars = [read(atom) for atom in jaxpr.outvars]
    # A layout the normal form left unread (a value's form before its unit axes were erased) is no part of
    # what the program computes.
    return jcore.Jaxpr(jaxpr.replace(constvars=[], eqns=live(equations, outvars), outvars=outvars), [])


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
    (`squeezed`), and each entrywise equation by its output (`entrywise_of`). One for a whole program, its
    nested programs' among it: a value crosses a nested program's boundary as itself."""

    widened: _Made
    laid: _Made
    squeezed: _Made
    entrywise_of: _Made


def _evaluated(jaxpr: jcore.Jaxpr, consts: Sequence, *values, forms: _Forms | None = None) -> list:
    """`jaxpr` run over `values` with every call's program (`dew.nn.programs.called`) but a marked span's run
    in place, each equation in the form recognition compares (`_normal`): under a trace its equations each
    bind once, a value of constants alone computed where it is made."""
    env: dict[jcore.Var, object] = dict(zip(jaxpr.constvars, consts, strict=True))
    env.update(zip(jaxpr.invars, values, strict=True))
    forms = forms or _Forms(_Made(), _Made(), _Made(), _Made())
    widened, laid, squeezed, entrywise_of = forms

    def read(atom):
        return atom.val if isinstance(atom, jcore.Literal) else env[atom]

    for equation in jaxpr.eqns:
        operands = [read(atom) for atom in equation.invars]
        # A marked span stays as written: a boundary the algebra reads by its kind
        # (`dew.nn.derivatives.marked`, a protected span's inside no rule touches), so no region crosses it;
        # one that states the operation it computes is that operation (`_computed`).
        body = None if equation.primitive is marked_call_p else called(equation)
        made = _computed(equation, operands)
        if made is not None:
            outputs = made
        elif body is not None:
            # What the call computes, the primal of a custom derivative's: its own derivative rule bars its
            # region (`BARRIERS`); a declaration's is the call's, kept where the call is written.
            outputs = _evaluated(body, body.consts, *operands, forms=forms)
        else:
            # A value of constants alone is computed where it is made, so the program and a declaration
            # hold one constant (a mean of zeros, a mask of iotas) however each built it: of the equations
            # such a constant is built from (`FOLDED`), on this host, never one that places a program on a
            # mesh, which binds as traced. An equation over a traced value stays traced; the constants its
            # normal form builds beside it (a broadcast divisor) are computed.
            traced = any(isinstance(value, jax_core.Tracer) for value in operands)
            folded = None if traced or not _admitted(equation.primitive, operands) else _folded(
                equation, operands)
            if folded is not None:
                outputs = folded
            else:
                # A value this host cannot compute (one it cannot read whole, a key) stays an equation.
                with jax.ensure_compile_time_eval() if traced else contextlib.nullcontext():
                    outputs = (_unit_round_trip(equation, operands, squeezed)
                               or _erased(equation, operands, widened, laid, squeezed, entrywise_of)
                               or _normal(equation, operands, widened, laid, entrywise_of))
        env.update(zip(equation.outvars, outputs, strict=True))
    return [read(atom) for atom in jaxpr.outvars]


def stated_primitive(equation: jcore.JaxprEqn) -> jcore.Primitive | None:
    """The primitive a marked span stating the operation it computes (`derivatives.stated_operation`) is read
    as (`_computed`): Dew's one logistic (`precision.logistic_value`) the logistic, `logistic_p`; None for
    any other equation."""
    from dew.nn.precision import logistic_value

    return jax.lax.logistic_p if stated_operation(equation) is logistic_value else None


def _computed(equation: jcore.JaxprEqn, operands: list) -> list | None:
    """A marked span's outputs as the operation it states it computes (`derivatives.stated_operation`): Dew's
    one logistic (`precision.logistic_value`) the logistic, `logistic_p`, of its operand widened to float32
    where narrower and its result converted back (one convert, the logistic, one convert, as the span
    computes and rounds it; at float32 or wider the logistic alone), so a program's own sigmoid and Dew's
    reach one normal form; None for any other equation, a span stating nothing kept as written. The form
    holds the node; turned back into code it is Dew's arithmetic again (`terms.value_of`)."""
    from dew.nn.precision import at_least_fp32

    if stated_primitive(equation) is not jax.lax.logistic_p or len(operands) != 1:
        return None
    (operand,) = operands
    dtype = jnp.result_type(operand)
    # The logistic node a program's own sigmoid binds, its parameters with it (its accuracy unstated, as
    # `jax.lax.logistic` binds it by default): a bare bind carries none, which no program's equation equals.
    return [jax.lax.logistic_p.bind(jnp.asarray(operand).astype(at_least_fp32(dtype)), accuracy=None)
            .astype(dtype)]


FOLDED = frozenset({jax.lax.iota_p, jax.lax.broadcast_in_dim_p, jax.lax.reshape_p, jax.lax.transpose_p,
    jax.lax.squeeze_p, jax.lax.concatenate_p, jax.lax.slice_p, jax.lax.convert_element_type_p,
    jax.lax.select_n_p, jax.lax.clamp_p, jax.lax.eq_p, jax.lax.ne_p, jax.lax.lt_p, jax.lax.le_p,
    jax.lax.gt_p, jax.lax.ge_p, jax.lax.and_p, jax.lax.or_p, jax.lax.not_p, jax.lax.add_p, jax.lax.sub_p,
    jax.lax.mul_p, jax.lax.div_p, jax.lax.max_p, jax.lax.min_p, jax.lax.neg_p, jax.lax.abs_p,
    jax.lax.reduce_sum_p, jax.lax.reduce_max_p, jax.lax.reduce_min_p, jax.lax.integer_pow_p,
    jax.lax.square_p, jax.lax.sqrt_p, jax.lax.rsqrt_p})
"""The equations `_evaluated` computes where their operands are constant arrays: those a mask, a divisor or a
table is built from (a scalar's the shared fold's, `_admitted`). Any other binds as traced, a collective or
a program placed on a mesh among them, whose eager run needs devices a trace has not."""


def _admitted(primitive: jcore.Primitive, operands: list) -> bool:
    """Whether `_evaluated` computes `primitive` over constant `operands` where they are made: over scalars,
    each arithmetic the shared fold computes (`dew.nn.programs.SCALAR_FOLDED`: a shift, a remainder and an XOR
    among them, so `1 << 2` is the literal `4` it computes, where a list of its own kept it an equation: base
    review bb25f261d's F2); over arrays, those a mask, a divisor or a table is built from (`FOLDED`)."""
    if primitive in SCALAR_FOLDED and not any(map(np.ndim, operands)):
        return True
    return primitive in FOLDED


def _folded(equation: jcore.JaxprEqn, operands: list) -> list | None:
    """A constant equation's outputs (`FOLDED`) as this host computes them (`host_evaluated`), a broadcast
    kept a view of its operand and an entrywise equation over broadcast operands computed over their
    compact forms (`terms.compact`) and broadcast back: a mask laid over a batch's heads holds its plane's
    bytes once, where computed whole each of a 2,048-key causal mask's equations made 67 MB, and its fill
    268 MB (row 6799). The values are the same, entry for entry; only their layout in this host's memory
    differs."""
    primitive = equation.primitive
    if len(equation.outvars) != 1 or any(terms.typed_key(value) for value in operands):
        return host_evaluated(primitive, operands, equation.params)
    out = aval(equation.outvars[0])
    shape = out.shape
    if primitive is jax.lax.broadcast_in_dim_p:
        # In the result's dtype, which a weak operand (a Python number) takes where it is broadcast.
        operand = np.asarray(operands[0], out.dtype)
        dimensions = equation.params["broadcast_dimensions"]
        if len(dimensions) > 1:
            order = tuple(sorted(range(len(dimensions)), key=dimensions.__getitem__))
            if order != tuple(range(len(dimensions))):
                operand = np.transpose(operand, order)
        placed = operand[tuple(slice(None) if axis in dimensions else np.newaxis
                               for axis in range(len(shape)))]
        return [np.broadcast_to(placed, shape)]
    if not entrywise(primitive):
        return host_evaluated(primitive, operands, equation.params)
    compacts = [terms.compact(np.asarray(value)) if np.ndim(value) else value for value in operands]
    # Broadcast as the operands broadcast, an axis of none beside one of one element keeping none.
    common = np.broadcast_shapes(*(np.shape(value) for value in compacts if np.ndim(value)))
    if np.prod(common, dtype=np.int64) == np.prod(shape, dtype=np.int64):
        return host_evaluated(primitive, operands, equation.params)
    inner = [np.broadcast_to(value, common) if np.ndim(value) else value for value in compacts]
    made = host_evaluated(primitive, inner, equation.params)
    return None if made is None else [np.broadcast_to(np.asarray(made[0]), shape)]


VALUE_IDENTITIES = (primitives.name_p,)
"""The equations whose value is their operand's (a checkpoint's name), which `_normal` reads through: what
they change is a schedule, never the value compared. A gradient's stop is no identity (`BARRIERS`)."""


NARROW = {jnp.dtype(jnp.bfloat16): (8, 7), jnp.dtype(jnp.float16): (5, 10)}
"""The narrow floats, by the exponent and mantissa bits `reduce_precision` rounds a float32 to them by."""


FULL_RANGE = frozenset({jnp.dtype(jnp.bfloat16)})
"""The narrow floats holding float32's exponent range, whose `reduce_precision` is a round trip through them:
a float16's underflows to zero where the cast keeps a subnormal (StableHLO's reduce_precision)."""


ROUNDED_ONCE = frozenset({jax.lax.add_p, jax.lax.sub_p, jax.lax.mul_p, jax.lax.div_p, jax.lax.max_p,
    jax.lax.min_p, jax.lax.neg_p, jax.lax.abs_p, jax.lax.sqrt_p})
"""The elementwise primitives whose narrow-float equation is its float32 one rounded once (`_normal`): IEEE's
correctly rounded arithmetic, which float32 holds at least twice a bfloat16's or float16's digits and two
more for, so a float32 result rounded again rounds as the narrow operation does. No transcendental: a
backend computes a narrow one its own way (XLA:GPU's logistic in the operand's dtype)."""


def _normal_arith(equation: jcore.JaxprEqn, operands: list, widened: _Made,
                  laid: _Made, entrywise_of: _Made) -> list | None:
    """Arithmetic normalizations for `_normal`: casts, squares, narrow `reduce_precision`, uniform-constant
    operands, products' stated precision, and narrow-float rounded-once ops; None to bind as written."""
    primitive = equation.primitive
    if primitive is jax.lax.convert_element_type_p:
        return [_converted(operands[0], equation.params["new_dtype"], widened)]
    if primitive is jax.lax.square_p or (
            primitive is jax.lax.integer_pow_p and equation.params.get("y") == 2):
        return [jax.lax.mul(operands[0], operands[0])]
    if primitive is jax.lax.reduce_precision_p:
        bits = (equation.params["exponent_bits"], equation.params["mantissa_bits"])
        narrow = next((dtype for dtype, held in NARROW.items() if held == bits and dtype in FULL_RANGE), None)
        # A round trip through the narrow float from float32 alone, whose exponent range it holds: a wider
        # float's reduce_precision flushes what the narrow float keeps as a subnormal.
        if narrow is not None and jnp.result_type(operands[0]) == jnp.float32:
            return [_converted(_converted(operands[0], narrow, widened), operands[0].dtype, widened)]
    if primitive in ROUNDED_ONCE and len(equation.outvars) == 1 and any(map(_uniform, operands)):
        shape = aval(equation.outvars[0]).shape
        if all(_uniform(value) or jnp.shape(value) == shape for value in operands):
            # A constant of one value throughout reads, element by element, as that value: the equation over
            # the scalar computes the same elements, so a mean's divisor held as an array is its scalar's.
            scalars = [_scalar(value) if _uniform(value) else value for value in operands]
            if any(jnp.shape(value) == shape for value in scalars):
                return _normal(equation, scalars, widened, laid, entrywise_of)
    if primitive is jax.lax.dot_general_p:
        return [equation.primitive.bind(*operands, **_stated(equation.params, operands))]
    out = aval(equation.outvars[0]).dtype if len(equation.outvars) == 1 else None
    if (primitive in ROUNDED_ONCE and out in NARROW
            and all(jnp.result_type(value) == out for value in operands)):
        wide = [_converted(value, jnp.float32, widened) for value in operands]
        return [_converted(equation.primitive.bind(*wide, **equation.params), out, widened)]
    return None


def _normal(equation: jcore.JaxprEqn, operands: list, widened: _Made,
            laid: _Made, entrywise_of: _Made) -> list:
    """`equation` bound over `operands` in the form recognition compares (`canonical`): a square a value
    times itself, bit for bit; a narrow float's elementwise equation (`ROUNDED_ONCE`) its float32 one
    rounded once to the narrow dtype; a gradient's stop or a checkpoint's name its operand
    (`VALUE_IDENTITIES`); a float32 rounded to a narrow float's digits in place
    (`reduce_precision`) a round trip through that dtype; a value widened exactly and rounded back to its
    own dtype that value itself (`widened`, each exact widening's result by its source); a value less a
    constant zero that value; a reshape that only adds unit axes the broadcast it is; a transpose pushed into
    broadcasts where each source keeps its own axis order; and an elementwise equation over values one layout
    made from equal shapes, beside scalars, that equation before the layout (`laid`). Every other equation as
    it is."""
    primitive = equation.primitive
    if primitive in VALUE_IDENTITIES:
        return list(operands)
    if (primitive is jax.lax.sub_p and _zero(operands[1])
            and jnp.shape(operands[0]) == aval(equation.outvars[0]).shape):
        return [operands[0]]
    if (primitive in (jax.lax.div_p, jax.lax.mul_p) and _one(operands[1])
            and jnp.shape(operands[0]) == aval(equation.outvars[0]).shape):
        return [operands[0]]
    if primitive in (jax.lax.broadcast_in_dim_p, jax.lax.reshape_p):
        return [_laid(equation, operands[0], laid)]
    if primitive is jax.lax.transpose_p:
        pushed = _pushed_transpose(equation, operands[0], widened, laid, entrywise_of)
        if pushed is not None:
            return [pushed]
    if entrywise(primitive) and len(equation.outvars) == 1:
        lifted = _lifted(equation, operands, widened, laid, entrywise_of)
        if lifted is not None:
            return [lifted]
    if (made := _normal_arith(equation, operands, widened, laid, entrywise_of)) is not None:
        return made
    bound = equation.primitive.bind(*operands, **equation.params)
    if (not equation.primitive.multiple_results and entrywise(primitive) and not equation.effects
            and isinstance(bound, jax_core.Tracer)):
        entrywise_of[bound] = (equation, list(operands))
    return list(bound) if equation.primitive.multiple_results else [bound]


def _stated(params: dict, operands: Sequence) -> dict:
    """A product's `params` with its precision as the arithmetic it states: two operands of one 16-bit
    float summed to float32 are exact products summed in float32 under any `jax.lax.Precision` (which
    chooses the passes of wider operands alone), the default algorithm or the dtype's own algorithm
    (`dew.nn.programs.own_preset`, jax.nn's attention names it), so each reads as no precision; any
    other product keeps the precision it states."""
    dtypes = {jnp.result_type(value) for value in operands}
    precision = params.get("precision")
    if (len(dtypes) != 1 or (own := own_preset(dtypes.pop())) is None
            or params.get("preferred_element_type") != jnp.float32):
        return params
    enums = precision if isinstance(precision, tuple) else (precision,)
    if all(part is None or isinstance(part, jax.lax.Precision) for part in enums) or (
            isinstance(precision, jax.lax.DotAlgorithmPreset)
            and precision in (jax.lax.DotAlgorithmPreset.DEFAULT, own)):
        return {**params, "precision": None}
    return params


UNIT_ERASED = frozenset({jax.lax.dot_general_p, jax.lax.transpose_p, jax.lax.reduce_sum_p,
    jax.lax.reduce_max_p, jax.lax.reduce_min_p, jax.lax.reshape_p, jax.lax.broadcast_in_dim_p})
"""The layout-reading equations `_erased` computes over their operands' axes of more than one element."""


def _erased(equation: jcore.JaxprEqn, operands: list, widened: _Made, laid: _Made,
            squeezed: _Made, entrywise_of: _Made) -> list | None:
    """An equation over values with axes of one element computed over those values with every such axis
    erased (`_squeeze`), then laid back out to its shape: a size-1 axis holds the same elements wherever
    a layout puts it, so a statistic kept `[N, 1]` and one held `[N]` are one computation. The equation's
    axes are renumbered over the erased ones: a product's batch and contracting pairs of one element and a
    reduction over one dropped, a transpose's order kept among the rest. None for an equation that reads
    no layout (`UNIT_ERASED`, an entrywise one) or holds no such axis, which `_normal` computes as it is."""
    primitive = equation.primitive
    if len(equation.outvars) != 1 or not (primitive in UNIT_ERASED or entrywise(primitive)):
        return None
    out = aval(equation.outvars[0])
    shapes = [jnp.shape(value) for value in operands]
    if 1 not in out.shape and not any(1 in shape for shape in shapes):
        return None
    kept = [[axis for axis, size in enumerate(shape) if size != 1] for shape in shapes]
    renumbered = [{axis: place for place, axis in enumerate(axes)} for axes in kept]
    inner = [_squeeze(value, squeezed) for value in operands]
    params = dict(equation.params)
    if primitive is jax.lax.dot_general_p:
        params = _stated(params, operands)
        (lc, rc), (lb, rb) = params["dimension_numbers"]
        left, right = renumbered
        contracting = [(a, b) for a, b in zip(lc, rc, strict=True) if a in left]
        batch = [(a, b) for a, b in zip(lb, rb, strict=True) if a in left]
        params["dimension_numbers"] = ((tuple(left[a] for a, _ in contracting),
                                        tuple(right[b] for _, b in contracting)),
                                       (tuple(left[a] for a, _ in batch), tuple(right[b] for _, b in batch)))
    elif primitive is jax.lax.transpose_p:
        order = [axis for axis in params["permutation"] if axis in renumbered[0]]
        params["permutation"] = tuple(renumbered[0][axis] for axis in order)
        if params["permutation"] == tuple(range(len(order))):
            # A transpose moving only axes of one element keeps every element where it was.
            return [_unsqueezed(inner[0], out.shape, squeezed)]
    elif primitive in (jax.lax.reduce_sum_p, jax.lax.reduce_max_p, jax.lax.reduce_min_p):
        params["axes"] = tuple(renumbered[0][axis] for axis in params["axes"] if axis in renumbered[0])
        if not params["axes"]:
            if primitive is not jax.lax.reduce_sum_p:
                return [_unsqueezed(inner[0], out.shape, squeezed)]
            # A sum of one element is its value plus the sum's zero, which makes a negative zero positive:
            # no identity, computed as written.
            return None
    elif primitive is jax.lax.broadcast_in_dim_p:
        target = [size for size in out.shape if size != 1]
        at = {axis: place for place, axis in enumerate(a for a, size in enumerate(out.shape) if size != 1)}
        params["shape"] = tuple(target)
        dimensions = tuple(at[params["broadcast_dimensions"][axis]] for axis in kept[0])
        if list(dimensions) != sorted(dimensions):
            # A broadcast that also reorders its operand's axes is another layout: computed as it is.
            return None
        params["broadcast_dimensions"] = dimensions
    elif primitive is jax.lax.reshape_p:
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
        inner = [_broadcast(value, out.shape, squeezed, laid) if jnp.ndim(value) else value
                 for value in operands]
    # Its erased value's type, its sharding's spec of that value's rank (an Explicit mesh's: the whole
    # result's spec on it raised).
    core = jax.eval_shape(lambda *values: equation.primitive.bind(*values, **params), *inner)
    inner_equation = equation.replace(params=params, outvars=[jcore.Var(out.update(shape=core.shape,
                                                                                  sharding=core.sharding))])
    core = _normal(inner_equation, inner, widened, laid, entrywise_of)[0]
    return [_unsqueezed(core, out.shape, squeezed)]


def _broadcast(value, shape: tuple[int, ...], squeezed: _Made, laid: _Made):
    """`value`, of the rank of `shape` and one element along each axis it does not share, broadcast to
    `shape`'s axes of more than one element over its erased form (`_squeeze`), recorded in `laid`."""
    if jnp.shape(value) == tuple(shape):
        return _squeeze(value, squeezed)
    kept = [axis for axis, size in enumerate(jnp.shape(value)) if size != 1]
    targets = [axis for axis, size in enumerate(shape) if size != 1]
    at = {axis: place for place, axis in enumerate(targets)}
    source = _squeeze(value, squeezed)
    target = tuple(shape[axis] for axis in targets)
    dimensions = tuple(at[axis] for axis in kept)
    with _beside(value):
        made = jax.lax.broadcast_in_dim(source, target, dimensions)
    if made is not source and isinstance(made, jax_core.Tracer):
        laid[made] = (source, (target, dimensions))
    return made


def _beside(value) -> contextlib.AbstractContextManager:
    """Where an operation on `value` runs: a constant's on this host (`programs.on_host`), a constant a normal
    form builds beside a traced equation computed where nothing places it on the trace's mesh (a
    compile-only GPU's, whose devices run nothing here); a traced value's in its trace."""
    return contextlib.nullcontext() if isinstance(value, jax_core.Tracer) else on_host()


def _squeeze(value, squeezed: _Made):
    """`value` with its axes of one element erased: the erased value it was laid out from where it was
    (`squeezed`), else its reshape, once a value."""
    shape = jnp.shape(value)
    if 1 not in shape:
        return value
    if value in squeezed:
        return squeezed[value]
    target = tuple(size for size in shape if size != 1)
    made = (jax.lax.reshape(value, target) if isinstance(value, jax_core.Tracer) or terms.typed_key(value)
            else np.reshape(np.asarray(value), target))
    squeezed[value] = made
    return made


def _unit_round_trip(equation: jcore.JaxprEqn, operands: list, squeezed: _Made) -> list | None:
    """A reshape back to the shape of the erased value its operand was laid out from by axes of one element
    (`_unsqueezed`), that value: the axes added and taken away move no element; None for any other
    equation."""
    if equation.primitive is not jax.lax.reshape_p or equation.params.get("dimensions") is not None:
        return None
    core = squeezed.get(operands[0])
    return [core] if core is not None and jnp.shape(core) == aval(equation.outvars[0]).shape else None


def _unsqueezed(core, shape: tuple[int, ...], squeezed: _Made):
    """`core` laid out to `shape`, its axes of one element restored, recorded as `core`'s (`squeezed`)."""
    if jnp.shape(core) == tuple(shape):
        return core
    made = jax.lax.reshape(core, tuple(shape))
    squeezed[made] = core
    return made


def _uniform(value) -> bool:
    """Whether `value` is a constant (computed, `_evaluated`) of more than one element, all one value bit for
    bit (a zero's sign among its bits): its compact form one element (`terms.compact`)."""
    if isinstance(value, jax_core.Tracer) or jnp.ndim(value) == 0 or not np.size(value):
        return False
    return terms.compact(np.asarray(value)).size == 1


def _scalar(value) -> np.ndarray:
    """A uniform constant (`_uniform`) as its one value, in its dtype."""
    return np.asarray(value).flat[0].astype(np.asarray(value).dtype)


def _one(value) -> bool:
    """Whether `value` is a real constant (computed, `_evaluated`) of ones alone: `x / 1` and `x * 1` are
    `x` for every real `x`, its sign and infinities among them, and a NaN stays one. A complex one is none:
    a complex product and quotient mix their operands' two parts, so `(-0 - 0j) * 1` is `0 - 0j` and
    `(inf + 2j) * 1` has a NaN part."""
    if isinstance(value, jax_core.Tracer):
        return False
    held = terms.compact(np.asarray(value))
    return not np.iscomplexobj(held) and bool(np.all(held == 1))


def _zero(value) -> bool:
    """Whether `value` is a constant (computed, `_evaluated`) of positive zeros alone: `x - 0` is `x` for
    every `x` (a NaN stays one), where `x - (-0)` makes a negative zero positive; a complex zero in each of
    its two parts, which a complex difference subtracts apart."""
    if isinstance(value, jax_core.Tracer):
        return False
    held = terms.compact(np.asarray(value))
    parts = (np.real(held), np.imag(held)) if np.iscomplexobj(held) else (held,)
    return not np.any(held) and not any(np.any(np.signbit(part)) for part in parts)


def _laid(equation: jcore.JaxprEqn, value, laid: _Made):
    """A broadcast's or a reshape's result, a reshape that only adds unit axes written as the broadcast it
    is, recorded in `laid` by its source and layout (`_lifted`)."""
    shape = aval(equation.outvars[0]).shape
    if equation.primitive is jax.lax.reshape_p:
        ordered = equation.params.get("dimensions") is None
        dimensions = _unit_axes(jnp.shape(value), shape) if ordered else None
        if dimensions is None:
            return equation.primitive.bind(value, **equation.params)
    else:
        dimensions = tuple(equation.params["broadcast_dimensions"])
    out = aval(equation.outvars[0])
    if (isinstance(value, jax_core.Tracer) and jnp.shape(value) == shape
            and dimensions == tuple(range(len(shape))) and jnp.result_type(value) == out.dtype
            and jax.typeof(value).sharding == out.sharding):
        return value
    made = jax.lax.broadcast_in_dim(value, shape, dimensions)
    if made is not value and isinstance(made, jax_core.Tracer):
        laid[made] = (value, (shape, dimensions))
    return made


def _inv_perm(perm: tuple[int, ...]) -> tuple[int, ...]:
    """Inverse axis permutation: `inv[perm[k]] == k`."""
    inv = [0] * len(perm)
    for place, axis in enumerate(perm):
        inv[axis] = place
    return tuple(inv)


def _can_push_transpose(value, inv: tuple[int, ...], laid: _Made, entrywise_of: _Made,
                        memo: dict[int, bool]) -> bool:
    """Whether `transpose(..., perm)` can push bit-exactly through `value` into broadcasts without creating
    any new transpose, memoized per tracer identity so shared DAG operands are visited once."""
    if jnp.ndim(value) == 0:
        return True
    key = id(value)
    if key in memo:
        return memo[key]
    if value in laid:
        _, (_, dimensions) = laid[value]
        moved = tuple(inv[axis] for axis in dimensions)
        ok = all(a < b for a, b in itertools.pairwise(moved))
    elif value in entrywise_of:
        _, operands = entrywise_of[value]
        ok = all(_can_push_transpose(op, inv, laid, entrywise_of, memo) for op in operands)
    else:
        ok = False
    memo[key] = ok
    return ok


def _push_transpose(value, perm: tuple[int, ...], inv: tuple[int, ...],
                    widened: _Made, laid: _Made, entrywise_of: _Made, memo: dict[int, object]):
    """Push `transpose(..., perm)` through `value` (`_can_push_transpose` holds), rebuilding broadcasts and
    entrywise equations in transposed shape once per shared tracer."""
    if jnp.ndim(value) == 0:
        return value
    key = id(value)
    if key in memo:
        return memo[key]
    if value in laid:
        source, (bcast_shape, dimensions) = laid[value]
        new_shape = tuple(bcast_shape[axis] for axis in perm)
        moved = tuple(inv[axis] for axis in dimensions)
        if jnp.shape(source) == new_shape and moved == tuple(range(len(new_shape))):
            made = source
        else:
            made = jax.lax.broadcast_in_dim(source, new_shape, moved)
            if made is not source and isinstance(made, jax_core.Tracer):
                laid[made] = (source, (new_shape, moved))
        memo[key] = made
        return made
    inner_eqn, inner_ops = entrywise_of[value]
    pushed = [_push_transpose(op, perm, inv, widened, laid, entrywise_of, memo) for op in inner_ops]
    core = jax.eval_shape(lambda *vs: inner_eqn.primitive.bind(*vs, **inner_eqn.params), *pushed)
    rebuilt = inner_eqn.replace(outvars=[jcore.Var(aval(inner_eqn.outvars[0]).update(
        shape=core.shape, sharding=core.sharding))])
    made = _normal(rebuilt, pushed, widened, laid, entrywise_of)[0]
    memo[key] = made
    return made


def _pushed_transpose(equation: jcore.JaxprEqn, operand, widened: _Made, laid: _Made,
                      entrywise_of: _Made):
    """A bit-exact transpose elimination: identity permutations return their operand, and permutations over
    broadcasts and entrywise combinations of broadcasts (e.g. `swapaxes(a[:, None] * b[None, :], -1, -2)`)
    push into the broadcasts when each source keeps its own axis order; None otherwise."""
    perm = tuple(equation.params["permutation"])
    out = aval(equation.outvars[0])
    if (perm == tuple(range(len(perm))) and isinstance(operand, jax_core.Tracer)
            and jax.typeof(operand).sharding == out.sharding):
        return operand
    inv = _inv_perm(perm)
    if not _can_push_transpose(operand, inv, laid, entrywise_of, {}):
        return None
    made = _push_transpose(operand, perm, inv, widened, laid, entrywise_of, {})
    return made if jax.typeof(made).sharding == out.sharding else None


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


def _lifted(equation: jcore.JaxprEqn, operands: list, widened: _Made, laid: _Made,
            entrywise_of: _Made):
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
    source = jax.typeof(next(source for source, value in zip(sources, operands, strict=True)
                             if value in laid))
    inner = equation.replace(outvars=[jcore.Var(equation.outvars[0].aval.update(shape=source.shape,
                                                                                 sharding=source.sharding))])
    made = _normal(inner, sources, widened, laid, entrywise_of)[0]
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


def _equation_key(equation: jcore.JaxprEqn,
                  literal_key: Callable[[jcore.Literal], Hashable]) -> Hashable | None:
    """An equation's identity as its value's (`canonical`): its primitive, parameters and operands, each
    literal by its value (`literal_key`, `terms.literal_key` held once a value); None for one holding an
    effect, whose repetition is not its value's."""
    if equation.effects:
        return None
    operands = tuple(literal_key(atom) if isinstance(atom, jcore.Literal) else atom
                     for atom in equation.invars)
    return (*terms.parameters_key(equation.primitive, equation.params), operands)
