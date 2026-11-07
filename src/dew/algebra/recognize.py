"""Recognition at the compile point (docs/design/algebra.md §2 "Fusion", row 4783): a body's regions written
as Dew's kernels where they compute what the kernels' calls compute, as the e-graph proves it.

A family (`FAMILIES`) is one kind of call the compile point writes: Dew's RMSNorm, its stream and gated norms,
and SwiGLU. Its math is declared once, as the call itself on its portable path
(`implementation='xla'`), the arithmetic its kernels are held to, so no second account of it is written here.
Around each equation of the family's anchor (a primitive its declaration always holds: `rsqrt`, `logistic`)
the family names its candidates (`Candidate`): a call over some of the body's values (its operands),
making some of its values (its results), its static arguments read from the region's own constants (an
epsilon), each choice the call takes tried rather than assumed. An anchor's candidates go into one
e-graph (`_equalities`): the equations of the regions they span and every candidate's declaration traced over
the same operands, made canonical (`_canonical`: nested programs inlined, values made from no input computed
into constants, equal equations once) and held in one term region, so a parameter, a cast or a constant the
two share is one term; then saturated under exactly the rule sets the call's numerics admit
(`rules.admitted`: the exact sets alone under the accurate forms, so a product reordered or a scale moved
across one matches only under the fast forms). A candidate whose every result is in its declaration's
e-class, whose region the body reads nothing of but its results (`_between`), and whose call the family's
selector gives a kernel of Dew's (`Candidate.taken`) is the call's, its region written as the call
(`fusion.Written`). Equal, never near: a different constant, cast, axis or order is another program, and stays
as traced.
"""

from __future__ import annotations

import collections
import functools
import hashlib
import itertools
import weakref
from collections.abc import Callable, Hashable, Iterator, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from egglog import eq
from jax.extend import core as jcore

from dew import _cache
from dew.algebra import SATURATION, derive, fusion, program, rules, terms
from dew.nn.kernels import generation
from dew.nn.programs import jax_name

REACH = 8
"""How far from its anchor a family looks for a candidate's operands and results, in equations: past the
deepest declaration's chain on either side of its anchor."""

SPAN = 64
"""The most equations a candidate's region holds (`_between`): a region reaching further reads a value the
candidate does not bind, and is none."""


class Candidate(NamedTuple):
    """A call the compile point may write for a region of a body: its operands and results as the body holds
    them, the call over the operands making the results (at implementation 'auto', so its kernel is the
    selector's), its declaration (the same call on its portable path), and whether the selector gives the
    call a kernel of Dew's."""

    operands: tuple[jcore.Var, ...]
    outputs: tuple[jcore.Var, ...]
    call: Callable
    declared: Callable
    taken: Callable[[], bool]


class Family(NamedTuple):
    """A kind of call (`FAMILIES`): its anchor primitive and its candidates around an equation of it."""

    anchor: str
    candidates: Callable[[fusion.Index, int], Iterator[Candidate]]


def recognized(jaxpr: jcore.Jaxpr, index: fusion.Index, held: frozenset[int] = frozenset()) -> list[
        fusion.Written]:
    """Every region of `jaxpr` (`index`'s) the e-graph proves computes a family's call and whose call takes
    a kernel of Dew's, written as that call (the module's account): the families in order, so a gated
    norm's activation is the norm's rather than a SwiGLU's, an anchor's candidates in the order its family
    names them, the first equal one taken, no two regions sharing an equation nor one `held` (a product's
    epilogue's). Each region's choice is held by its structure and its constants' values (`_HELD_REGIONS`),
    the body's whole recognition by its structure and constants (`_HELD_BODIES`)."""
    key = (_body_key(jaxpr, index), held, derive.context(), _cache.KERNEL_CHOICES.value)
    if key not in _HELD_BODIES:
        _HELD_BODIES[key] = [_positioned(jaxpr, written)
                             for written in _recognizing(jaxpr, index, held)]
        if len(_HELD_BODIES) > fusion.DECISIONS_HELD:
            _HELD_BODIES.popitem(last=False)
    _HELD_BODIES.move_to_end(key)
    return [_placed(jaxpr, written) for written in _HELD_BODIES[key]]


_HELD_BODIES: collections.OrderedDict[Hashable, list] = collections.OrderedDict()
"""Each body's recognition by its structure, its constants and the context it was weighed in
(`recognized`), positioned (`_positioned`): every retrace of a step, and every layer body of one structure,
recognized once, the latest `fusion.DECISIONS_HELD` held."""


def _body_key(jaxpr: jcore.Jaxpr, index: fusion.Index) -> Hashable:
    """A body's structure (`program.structure`, its constants among its inputs) and its constants' values,
    which a call's static arguments read (an epsilon, a mask)."""
    window = jcore.Jaxpr([], [*jaxpr.constvars, *jaxpr.invars], list(jaxpr.outvars), jaxpr.eqns)
    held = tuple(_digest(index.consts[var]) if var in index.consts else None for var in jaxpr.constvars)
    return program.structure(window), held


def _positions(jaxpr: jcore.Jaxpr) -> dict[jcore.Var, int]:
    """Each value of a body by its position: its constants, inputs, then each equation's outputs in order."""
    values = [*jaxpr.constvars, *jaxpr.invars, *(var for equation in jaxpr.eqns for var in equation.outvars)]
    return {var: place for place, var in enumerate(values)}


def _positioned(jaxpr: jcore.Jaxpr, written: fusion.Written) -> tuple:
    """`written` with its values by their positions (`_positions`), for another body of one structure."""
    at = _positions(jaxpr)
    return (written.places, tuple(at[var] for var in written.inputs),
            tuple(at[var] for var in written.outputs), written.at, written.call)


def _placed(jaxpr: jcore.Jaxpr, held: tuple) -> fusion.Written:
    """A positioned recognition (`_positioned`) in `jaxpr`'s own values."""
    values = [*jaxpr.constvars, *jaxpr.invars, *(var for equation in jaxpr.eqns for var in equation.outvars)]
    places, inputs, outputs, at, call = held
    return fusion.Written(places, tuple(values[place] for place in inputs),
                          tuple(values[place] for place in outputs), at, call)


def _recognizing(jaxpr: jcore.Jaxpr, index: fusion.Index, held: frozenset[int]) -> list[fusion.Written]:
    """`recognized`'s search."""
    accurate = generation.numerics().forms == "accurate"
    found: list[fusion.Written] = []
    claimed: set[int] = set(held)
    for family in FAMILIES:
        for place, equation in enumerate(jaxpr.eqns):
            if place in claimed or not _holds(equation, family.anchor):
                continue
            spans = [(candidate, places) for candidate in family.candidates(index, place)
                     if (places := _between(index, candidate.operands, candidate.outputs)) is not None
                     and not claimed & set(places)]
            written = _first_equal(jaxpr, index, spans, accurate)
            if written is not None:
                found.append(written)
                claimed.update(written.places)
    return found


def _first_equal(jaxpr: jcore.Jaxpr, index: fusion.Index, spans: Sequence[tuple[Candidate, tuple[int, ...]]],
                 accurate: bool) -> fusion.Written | None:
    """The first of an anchor's candidates whose call takes a kernel of Dew's and whose region the e-graph
    proves computes it, written as its call; None where none is. Its regions are weighed in the order their
    first candidates come, each once: which of its candidates is equal and taken (`_chosen`), held by the
    region's structure and the values of the constants it reads (`_region_key`), its candidates in one
    e-graph (`_equalities`) built only where one of its calls takes a kernel."""
    regions: dict[tuple[int, ...], list[Candidate]] = {}
    for candidate, places in spans:
        regions.setdefault(places, []).append(candidate)
    for places, members in regions.items():
        key = _region_key(jaxpr, index, places, members[0].outputs, accurate)
        if key not in _HELD_REGIONS:
            # Its probes' routes are no program's: the routes a block collects are the kernels it runs.
            with generation.routes(alone=True):
                _HELD_REGIONS[key] = _chosen(index, places, members, accurate)
            if len(_HELD_REGIONS) > fusion.DECISIONS_HELD:
                _HELD_REGIONS.popitem(last=False)
        _HELD_REGIONS.move_to_end(key)
        chosen = _HELD_REGIONS[key]
        if chosen is not None:
            candidate = members[chosen]
            return fusion.Written(places, candidate.operands, candidate.outputs, places[-1], candidate.call)
    return None


_HELD_REGIONS: collections.OrderedDict[Hashable, int | None] = collections.OrderedDict()
"""Each region's chosen call (`_chosen`) by `_region_key`, the latest `fusion.DECISIONS_HELD` held: every
layer of a stack weighs its norm once."""


def _region_key(jaxpr: jcore.Jaxpr, index: fusion.Index, places: tuple[int, ...],
                outputs: Sequence[jcore.Var], accurate: bool) -> Hashable:
    """What a region's choice reads: its structure (`program.structure`, every value it reads an input), the
    values of the constants among them (`_digest`: a statistic's divisor, an epsilon held as an array), the
    numerics' forms and the context its kernels are chosen in."""
    inputs = fusion._read(jaxpr, places)
    window = jcore.Jaxpr([], list(inputs), list(outputs), [jaxpr.eqns[place] for place in places])
    held = tuple(_digest(index.consts[var]) if var in index.consts else None for var in inputs)
    return program.structure(window), held, accurate, derive.context(), _cache.KERNEL_CHOICES.value


def _digest(value) -> Hashable:
    """A constant by its type and bits."""
    value = np.asarray(value)
    return str(value.dtype), value.shape, hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _chosen(index: fusion.Index, places: tuple[int, ...], members: Sequence[Candidate],
            accurate: bool) -> int | None:
    """The first of one region's candidates whose call takes a kernel and which the e-graph proves equal to
    the region (`_equalities`, of the taken ones alone), its place among `members`; None where none is."""
    taken = [number for number, candidate in enumerate(members) if candidate.taken()]
    # Each alone first, where its results are its declaration's own values; the e-graph only where none is.
    for number in taken:
        if _equalities(index, [(members[number], places)], accurate, saturating=False)[0]:
            return number
    if not taken:
        return None
    equal = _equalities(index, [(members[number], places) for number in taken], accurate)
    return next((number for number, same in zip(taken, equal, strict=True) if same), None)


def _holds(equation: jcore.JaxprEqn, name: str) -> bool:
    """Whether `equation` binds the primitive `name`, or a nested program (`BODY_PARAMS`) whose body does, as
    a jit `jax.nn.silu` traces holds its sigmoid: the region around it reads the call whole, and its
    equality its body inlined (`_canonical`)."""
    if jax_name(equation.primitive) == name:
        return True
    body = equation.params.get(BODY_PARAMS.get(equation.primitive.name, ""))
    if body is None:
        return False
    inner = body.jaxpr if isinstance(body, jcore.ClosedJaxpr) else body
    return any(_holds(nested, name) for nested in inner.eqns)


def _between(index: fusion.Index, operands: Sequence[jcore.Var],
             outputs: Sequence[jcore.Var]) -> tuple[int, ...] | None:
    """The places of the equations computing `outputs` from `operands` and constants, in body order: None
    where they read another value of the body than the operands (an unbound input), hold more than `SPAN`
    equations, hold an equation that changes a derivative (`BARRIERS`), make a value the body reads outside
    them but the outputs, or make an output a reader outside reads before the region's last equation."""
    places: set[int] = set()
    frontier = [var for var in outputs if var not in operands]
    bound = set(operands)
    while frontier:
        var = frontier.pop()
        place = index.producer.get(var)
        if place is None:
            if var not in index.consts:
                return None
            continue
        if place in places:
            continue
        places.add(place)
        if len(places) > SPAN:
            return None
        frontier += [atom for atom in index.jaxpr.eqns[place].invars
                     if isinstance(atom, jcore.Var) and atom not in bound]
    results, last = set(outputs), max(places, default=-1)
    for place in places:
        if jax_name(index.jaxpr.eqns[place].primitive) in BARRIERS:
            return None
        for var in index.jaxpr.eqns[place].outvars:
            outside = index.readers[var] - places
            if var not in results and (var in index.escaping or outside):
                return None
            # A result is written where the region's last equation stood (`fusion._spliced`): a reader
            # outside the region before it would read it before it is made.
            if var in results and any(reader < last for reader in outside):
                return None
    return tuple(sorted(places))


BARRIERS = frozenset({"custom_jvp_call", "custom_vjp_call", "custom_vjp_call_jaxpr"})
"""The equations a region the compile point writes never holds: a program's own derivative rule, which the
call written in its place would not keep. A gradient's stop is no barrier but no identity either
(`VALUE_IDENTITIES`): it stays its own equation, so a region matches a declaration only where the declaration
stops the same value, and a stop on a call's operand stays outside the call."""


def _equalities(index: fusion.Index, spans: Sequence[tuple[Candidate, tuple[int, ...]]],
                accurate: bool, *, saturating: bool = True) -> list[bool]:
    """Whether each candidate's results are its declaration's, in one e-graph of every span's equations and
    every declaration, saturated under the rule sets the numerics admit (the module's account). Where a
    candidate's results are already its declaration's own values in the normal form (`_canonical`: one
    term, which every rule set admits), that candidate is equal and the rest are not weighed: no e-graph
    is built; nor where not `saturating`, which weighs that alone."""
    if not spans:
        return []
    places = sorted(set().union(*(set(span) for _, span in spans)))
    made = {var for place in places for var in index.jaxpr.eqns[place].outvars}
    inputs = list(dict.fromkeys(atom for place in places for atom in index.jaxpr.eqns[place].invars
                                if isinstance(atom, jcore.Var) and atom not in made
                                and atom not in index.consts))
    inputs += [var for candidate, _ in spans for var in candidate.operands if var not in made
               and var not in inputs]
    wanted = list(dict.fromkeys(var for candidate, _ in spans for var in (*candidate.operands,
                                                                         *candidate.outputs)))
    run, consts = _window(index, places, inputs, wanted)

    def joint(*values):
        held = dict(zip(wanted, run(*consts, *values), strict=True))
        results = [held[var] for candidate, _ in spans for var in candidate.outputs]
        for candidate, _ in spans:
            results += _declaration(candidate)(*(held[var] for var in candidate.operands))
        return results

    closed = _canonical(jax.make_jaxpr(joint)(*(jax.ShapeDtypeStruct(var.aval.shape, var.aval.dtype)
                                                 for var in inputs)))
    written = [_atom_key(atom) for atom in closed.jaxpr.outvars]
    shared, program_at = [], 0
    declared_at = sum(len(candidate.outputs) for candidate, _ in spans)
    for candidate, _ in spans:
        width = len(candidate.outputs)
        shared.append(written[program_at:program_at + width] == written[declared_at:declared_at + width])
        program_at += width
        declared_at += width
    if any(shared):
        first = shared.index(True)
        return [number == first for number in range(len(spans))]
    if not saturating:
        return shared
    region = terms.region(closed)
    graph, _, _ = program.graphed(region)
    program.saturate(graph, region, rules.admitted(accurate), SATURATION)
    # Each output's term, the program's results first, then each declaration's in the same order.
    results = region.outputs
    equal, program_at = [], 0
    declared_at = sum(len(candidate.outputs) for candidate, _ in spans)
    for candidate, _ in spans:
        pairs = [(results[program_at + n], results[declared_at + n]) for n in range(len(candidate.outputs))]
        equal.append(all(ours is not None and theirs is not None and graph.check_bool(eq(ours).to(theirs))
                         for ours, theirs in pairs))
        program_at += len(candidate.outputs)
        declared_at += len(candidate.outputs)
    return equal


def _atom_key(atom) -> Hashable:
    """An output of a canonical program as its value: the variable itself, or a literal's value."""
    return terms.literal_key(atom) if isinstance(atom, jcore.Literal) else atom


_DECLARATIONS: collections.OrderedDict[Hashable, Callable] = collections.OrderedDict()
"""Each declaration traced once (`_declaration`), by its call, its choices and its operands' types, the latest
`fusion.DECISIONS_HELD` held: every norm of a stack weighs the same declaration."""


def _declaration(candidate: Candidate) -> Callable:
    """`candidate`'s declaration as its traced program over its operands' types, run over values to its
    results' leaves; traced once for each call, choices and types (`_DECLARATIONS`)."""
    declared = candidate.declared
    function = declared.func
    cells = tuple(cell.cell_contents for cell in function.__closure__ or ())
    keywords = tuple(sorted((function.__kwdefaults__ or {}).items()))
    key = (function.__code__, _typed(function.__defaults__), _typed(keywords), _typed(cells),
           _typed(tuple(sorted(declared.keywords.items()))), _typed(tuple(candidate.operands)))
    if key not in _DECLARATIONS:
        traced = jax.make_jaxpr(lambda *values: jax.tree.leaves(declared(*values)))(
            *(_shape(var) for var in candidate.operands))
        _DECLARATIONS[key] = jcore.jaxpr_as_fun(traced)
        if len(_DECLARATIONS) > fusion.DECISIONS_HELD:
            _DECLARATIONS.popitem(last=False)
    _DECLARATIONS.move_to_end(key)
    return _DECLARATIONS[key]


def _typed(value) -> Hashable:
    """`value` as a declaration's key reads it: a body's value by its type, a sequence item by item."""
    if isinstance(value, jcore.Var):
        return value.aval
    if isinstance(value, (tuple, list)):
        return tuple(_typed(item) for item in value)
    return value


def _window(index: fusion.Index, places: Sequence[int], inputs: Sequence[jcore.Var],
            outputs: Sequence[jcore.Var]) -> tuple[Callable, list]:
    """The equations at `places` as a function of the body's constants they read, then `inputs`, making
    `outputs`, and those constants' values: its constants passed as its leading arguments, as a program
    closing over them computes them."""
    held = [var for var in index.consts if _reads(index, places, var)]
    window = jcore.Jaxpr([], [*held, *inputs], list(outputs), [index.jaxpr.eqns[place] for place in places])
    return jcore.jaxpr_as_fun(jcore.ClosedJaxpr(window, [])), [index.consts[var] for var in held]


def _reads(index: fusion.Index, places: Sequence[int], var: jcore.Var) -> bool:
    """Whether an equation at `places` reads `var`."""
    return bool(index.readers[var] & set(places))


def _canonical(closed: jcore.ClosedJaxpr) -> jcore.ClosedJaxpr:
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
    return jcore.ClosedJaxpr(jaxpr.replace(constvars=[], eqns=equations,
                                           outvars=[read(atom) for atom in jaxpr.outvars]), [])


BODY_PARAMS = {"jit": "jaxpr", "pjit": "jaxpr", "closed_call": "call_jaxpr", "core_call": "call_jaxpr",
               "custom_jvp_call": "call_jaxpr", "custom_vjp_call": "call_jaxpr",
               "custom_vjp_call_jaxpr": "fun_jaxpr", "remat2": "jaxpr", "checkpoint": "jaxpr"}
"""The nested programs `_evaluated` runs in place, by the parameter holding each one's body: what its call
computes, the primal of a custom derivative's. A program's own derivative rule bars its region (`BARRIERS`);
a declaration's is the call's, kept where the call is written."""


def _evaluated(jaxpr: jcore.Jaxpr, consts: Sequence, *values) -> list:
    """`jaxpr` run over `values` with every nested program (`BODY_PARAMS`) run in place, each equation in the
    form recognition compares (`_normal`): under a trace its equations each bind once, a value of constants
    alone computed where it is made."""
    env: dict[jcore.Var, object] = dict(zip(jaxpr.constvars, consts, strict=True))
    env.update(zip(jaxpr.invars, values, strict=True))
    widened: dict[int, object] = {}
    laid: dict[int, tuple] = {}
    squeezed: dict[int, object] = {}

    def read(atom):
        return atom.val if isinstance(atom, jcore.Literal) else env[atom]

    for equation in jaxpr.eqns:
        operands = [read(atom) for atom in equation.invars]
        body = equation.params.get(BODY_PARAMS.get(equation.primitive.name, ""))
        if body is not None:
            closed = body if isinstance(body, jcore.ClosedJaxpr) else jcore.ClosedJaxpr(body, ())
            given = operands[len(operands) - len(closed.jaxpr.invars):]
            outputs = _evaluated(closed.jaxpr, closed.consts, *given)
        else:
            # A value of constants alone is computed where it is made, so the program and a declaration
            # hold one constant (a mean of zeros, a mask of iotas) however each built it.
            with jax.ensure_compile_time_eval():
                outputs = _erased(equation, operands, widened, laid, squeezed) or _normal(
                    equation, operands, widened, laid)
        env.update(zip(equation.outvars, outputs, strict=True))
    return [read(atom) for atom in jaxpr.outvars]


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


def _normal(equation: jcore.JaxprEqn, operands: list, widened: dict[int, object],
            laid: dict[int, tuple]) -> list:
    """`equation` bound over `operands` in the form recognition compares (`_canonical`): a square a value
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
        if narrow is not None:
            return [_converted(_converted(operands[0], narrow, widened), operands[0].dtype, widened)]
    out = equation.outvars[0].aval.dtype if len(equation.outvars) == 1 else None
    if name in ROUNDED_ONCE and out in NARROW and all(jnp.result_type(value) == out for value in operands):
        wide = [_converted(value, jnp.float32, widened) for value in operands]
        return [_converted(equation.primitive.bind(*wide, **equation.params), out, widened)]
    bound = equation.primitive.bind(*operands, **equation.params)
    return bound if equation.primitive.multiple_results else [bound]


UNIT_ERASED = frozenset({"dot_general", "transpose", "reduce_sum", "reduce_max", "reduce_min", "reshape",
                         "broadcast_in_dim"})
"""The layout-reading equations `_erased` computes over their operands' axes of more than one element."""


def _erased(equation: jcore.JaxprEqn, operands: list, widened: dict[int, object], laid: dict[int, tuple],
            squeezed: dict[int, object]) -> list | None:
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
    elif name.startswith("reduce_"):
        params["axes"] = tuple(renumbered[0][axis] for axis in params["axes"] if axis in renumbered[0])
        if not params["axes"]:
            core = inner[0]
            return [_unsqueezed(core, out.shape, squeezed)]
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
        return None
    core_shape = jax.eval_shape(lambda *values: equation.primitive.bind(*values, **params), *inner).shape
    inner_equation = equation.replace(params=params, outvars=[jcore.Var(out.update(shape=core_shape))])
    core = _normal(inner_equation, inner, widened, laid)[0]
    return [_unsqueezed(core, out.shape, squeezed)]


def _squeeze(value, squeezed: dict[int, object]):
    """`value` with its axes of one element erased: the erased value it was laid out from where it was
    (`squeezed`), else its reshape, once a value."""
    shape = jnp.shape(value)
    if 1 not in shape:
        return value
    if id(value) in squeezed:
        return squeezed[id(value)]
    target = tuple(size for size in shape if size != 1)
    made = (np.reshape(np.asarray(value), target) if not isinstance(value, jax.core.Tracer)
            else jax.lax.reshape(value, target))
    squeezed[id(value)] = made
    return made


def _unsqueezed(core, shape: tuple[int, ...], squeezed: dict[int, object]):
    """`core` laid out to `shape`, its axes of one element restored, recorded as `core`'s (`squeezed`)."""
    if jnp.shape(core) == tuple(shape):
        return core
    made = jax.lax.reshape(core, tuple(shape))
    squeezed[id(made)] = core
    return made


ELEMENTWISE = ROUNDED_ONCE | {"convert_element_type", "select_n", "integer_pow", "square", "sign", "floor",
                             "ceil", "round", "pow", "atan2", "rem"}
"""The equations computed element by element, which a layout of their operands commutes with exactly."""


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


def _laid(equation: jcore.JaxprEqn, value, laid: dict[int, tuple]):
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
        laid[id(made)] = (value, (shape, dimensions))
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


def _lifted(equation: jcore.JaxprEqn, operands: list, widened: dict[int, object], laid: dict[int, tuple]):
    """An elementwise equation whose every operand but scalars one layout made, from sources of one shape:
    the equation over the sources, then the layout; None where its operands are laid otherwise."""
    layouts = {laid[id(value)][1] for value in operands if id(value) in laid}
    sources = [laid[id(value)][0] if id(value) in laid else value for value in operands]
    if len(layouts) != 1 or any(id(value) not in laid and jnp.ndim(value) != 0 for value in operands):
        return None
    if len({jnp.shape(source) for source, value in zip(sources, operands, strict=True)
            if id(value) in laid}) != 1:
        return None
    (shape, dimensions), = layouts
    inner = equation.replace(outvars=[jcore.Var(equation.outvars[0].aval.update(
        shape=jnp.shape(next(source for source, value in zip(sources, operands, strict=True)
                             if id(value) in laid))))])
    made = _normal(inner, sources, widened, laid)[0]
    result = jax.lax.broadcast_in_dim(made, shape, dimensions)
    laid[id(result)] = (made, (shape, dimensions))
    return result


def _converted(value, dtype, widened: dict[int, object]):
    """`value` cast to `dtype`: the value it was widened from where it is an exact widening of a value of
    `dtype` (`widened`), else the cast, recorded in `widened` where it widens exactly (a narrow float to
    float32, a float to float64)."""
    dtype = jnp.dtype(dtype)
    source = widened.get(id(value))
    if source is not None and jnp.result_type(source) == dtype:
        return source
    cast = jax.lax.convert_element_type(value, dtype)
    if cast is not value and jnp.issubdtype(dtype, jnp.floating) and jnp.issubdtype(jnp.result_type(value),
                                                                                    jnp.floating) and (
            jnp.finfo(dtype).nmant > jnp.finfo(jnp.result_type(value)).nmant
            and jnp.finfo(dtype).maxexp >= jnp.finfo(jnp.result_type(value)).maxexp):
        widened[id(cast)] = value
    return cast


def _equation_key(equation: jcore.JaxprEqn) -> Hashable | None:
    """An equation's identity as its value's (`_canonical`): its primitive, parameters and operands, each
    literal by its value; None for one holding an effect, whose repetition is not its value's."""
    if equation.effects:
        return None
    operands = tuple(terms.literal_key(atom) if isinstance(atom, jcore.Literal) else atom
                     for atom in equation.invars)
    return (*terms.parameters_key(equation.primitive, equation.params), operands)


def _forward(index: fusion.Index, var: jcore.Var, reach: int = REACH) -> dict[jcore.Var, int]:
    """The values `var` reaches within `reach` equations, with their distance, walked once a body."""
    held = _HELD_WALKS.setdefault(index, {})
    if ("forward", var, reach) not in held:
        held["forward", var, reach] = _walked_forward(index, var, reach)
    return held["forward", var, reach]


def _walked_forward(index: fusion.Index, var: jcore.Var, reach: int) -> dict[jcore.Var, int]:
    """`_forward`'s walk."""
    found, frontier = {var: 0}, collections.deque([var])
    while frontier:
        value = frontier.popleft()
        if found[value] == reach:
            continue
        for place in index.readers[value]:
            for out in index.jaxpr.eqns[place].outvars:
                if out not in found:
                    found[out] = found[value] + 1
                    frontier.append(out)
    return found


_HELD_WALKS: weakref.WeakKeyDictionary[fusion.Index, dict] = weakref.WeakKeyDictionary()
"""Each body's walks from its values (`_backward`, `_forward`), held while its index lives: every candidate
of an anchor walks from the same rows and results again."""


def _backward(index: fusion.Index, var: jcore.Var, reach: int = REACH) -> dict[jcore.Var, int]:
    """The values `var` is made from within `reach` equations, with their distance, walked once a body."""
    held = _HELD_WALKS.setdefault(index, {})
    if (var, reach) not in held:
        held[var, reach] = _walked_back(index, var, reach)
    return held[var, reach]


def _walked_back(index: fusion.Index, var: jcore.Var, reach: int) -> dict[jcore.Var, int]:
    """`_backward`'s walk."""
    found, frontier = {var: 0}, collections.deque([var])
    while frontier:
        value = frontier.popleft()
        place = index.producer.get(value)
        if place is None or found[value] == reach:
            continue
        for atom in index.jaxpr.eqns[place].invars:
            if isinstance(atom, jcore.Var) and atom not in found:
                found[atom] = found[value] + 1
                frontier.append(atom)
    return found


def _extras(index: fusion.Index, row: jcore.Var, output: jcore.Var) -> list[jcore.Var] | None:
    """The values beside `row` the equations from it to `output` read: each a value of the body made from
    neither it nor constants alone (a weight, a gate, a branch), in the order first read; None where `output`
    is not made from it. Made from the row is read as far as a candidate's operands lie from its results
    (twice `REACH`)."""
    descendants = _forward(index, row, 2 * REACH)
    if output not in descendants:
        return None
    extras: dict[jcore.Var, None] = {}
    seen, frontier = set(), [output]
    while frontier:
        var = frontier.pop()
        if var in seen or var == row:
            continue
        seen.add(var)
        if var not in descendants and not _constant(index, var):
            extras[var] = None
            continue
        place = index.producer.get(var)
        if place is not None:
            frontier += [atom for atom in index.jaxpr.eqns[place].invars if isinstance(atom, jcore.Var)]
    return list(extras)


def _constant(index: fusion.Index, var: jcore.Var) -> bool:
    """Whether `var` is made from the body's constants and literals alone (a mask of iotas, a table it
    closed over): read from the body's constant values, found once a body (`_constants`)."""
    return var in _constants(index)


_HELD_CONSTANTS: weakref.WeakKeyDictionary[fusion.Index, frozenset[jcore.Var]] = weakref.WeakKeyDictionary()
"""Each body's constant values (`_constants`), held while its index lives."""


def _constants(index: fusion.Index) -> frozenset[jcore.Var]:
    """The values of the body `index` reads made from its constants and literals alone, in one pass over its
    equations in order."""
    if index not in _HELD_CONSTANTS:
        found: set[jcore.Var] = set(index.consts)
        for equation in index.jaxpr.eqns:
            if all(isinstance(atom, jcore.Literal) or atom in found for atom in equation.invars):
                found.update(equation.outvars)
        _HELD_CONSTANTS[index] = frozenset(found)
    return _HELD_CONSTANTS[index]


def _scalars(index: fusion.Index, places: Sequence[int], primitives: frozenset[str]) -> list[float]:
    """The scalar floating constants the equations at `places` binding one of `primitives` read, each once:
    the static arguments a family's call may take (an epsilon a sum adds)."""
    found: dict[float, None] = {}
    for place in places:
        if jax_name(index.jaxpr.eqns[place].primitive) not in primitives:
            continue
        for atom in index.jaxpr.eqns[place].invars:
            value = atom.val if isinstance(atom, jcore.Literal) else index.consts.get(atom)
            if value is not None and np.ndim(value) == 0 and jnp.issubdtype(np.asarray(value).dtype,
                                                                            jnp.floating):
                found[float(value)] = None
    return list(found)


def _shape(var: jcore.Var) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(var.aval.shape, var.aval.dtype)


def _norms(index: fusion.Index, place: int) -> Iterator[Candidate]:
    """An RMSNorm's candidates around the inverse square root at `place` (Dew's `rms_normalized`, its stream
    and gated norms: `dew.nn.norms`): each row it is made from and each value it makes of the row's shape, the
    row's other operands bound by their shapes (a weight of the row's width, a gate of its shape), the
    region's scalar constants each an epsilon, and the call's choices (the weight's offset, its side of the
    cast, the gate's activation) each tried; the earliest row and the latest result first, so the largest
    region the e-graph proves is the one written. A row is a value its statistic's sum is made from by
    equations of one operand each (a cast, a square: `_summed_rows`). A row made by a sum the body reads on
    is tried as a
    stream's too, the sum among the results. The weight's side of the cast and its association (folded
    into the inverse first, Flax's `x (r w)`, `weight_first`) are each the call's own choices."""
    anchor = index.jaxpr.eqns[place].outvars[0]
    rows = [(0, row) for row in _summed_rows(index, anchor)]
    outputs = sorted(((distance, var) for var, distance in _forward(index, anchor).items()
                      if var is not anchor and var.aval.ndim >= 1), key=lambda pair: -pair[0])
    for (_, row), (_, output) in itertools.product(rows, outputs):
        if row.aval.shape != output.aval.shape or not jnp.issubdtype(output.aval.dtype, jnp.floating):
            continue
        found = _extras(index, row, output)
        if found is None or len(found) > 2:
            continue
        width = row.aval.shape[-1]
        for extras in itertools.product(*(_chain(index, var) for var in found)):
            weights = [var for var in extras if var.aval.shape == (width,)]
            gates = [var for var in extras if var.aval.shape == row.aval.shape]
            if len(weights) > 1 or len(gates) > 1 or len(weights) + len(gates) != len(extras):
                continue
            epsilons = _scalars(index, _between(index, (row, *extras), (output,)) or (), frozenset({"add"}))
            yield from _norm_candidates(index, row, output, weights[0] if weights else None,
                                        gates[0] if gates else None, epsilons)


def _norm_candidates(index: fusion.Index, row, output, weight, gate, epsilons) -> Iterator[Candidate]:
    """`_norms`' candidates over one row and result (the module's account)."""
    from dew.nn.kernels import norm_swiglu
    from dew.nn.norms import gated_rms_normalized, stream_rms_normalized

    dtype = output.aval.dtype
    if gate is not None:
        for epsilon, activation in itertools.product(epsilons, ("silu", "sigmoid")):
            def gated(x, g, *weight_, epsilon=epsilon, activation=activation, implementation='auto'):
                return gated_rms_normalized(x, g, weight_[0] if weight_ else None, epsilon, dtype,
                                            activation, implementation=implementation)

            yield Candidate((row, gate, *([weight] if weight is not None else [])), (output,), gated,
                            functools.partial(gated, implementation='xla'),
                            lambda activation=activation: norm_swiglu.gated_rms_kernel(
                                "auto", _shape(row), _shape(gate), None if weight is None else _shape(weight),
                                dtype, activation) == "pallas")
        return
    added = index.producer.get(row)
    streams = [None]
    if (added is not None and jax_name(index.jaxpr.eqns[added].primitive) == "add"
            and all(isinstance(atom, jcore.Var) and atom.aval.shape == row.aval.shape
                    for atom in index.jaxpr.eqns[added].invars)):
        streams.insert(0, tuple(index.jaxpr.eqns[added].invars))
    # (offset, after_cast, weight_first): the weight folded into the inverse first (Flax's) takes no cast.
    choices = [(False, False, False)] if weight is None else [
        choice for choice in itertools.product((False, True), repeat=3) if not (choice[1] and choice[2])]
    for stream, epsilon, (offset, after_cast, first) in itertools.product(streams, epsilons, choices):
        def normed(x, *rest, stream=stream, epsilon=epsilon, offset=offset, after_cast=after_cast,
                   first=first, implementation='auto'):
            branch, rest = (rest[0], rest[1:]) if stream is not None else (None, rest)
            summed, result = stream_rms_normalized(x, branch, rest[0] if rest else None, epsilon, dtype,
                                                   offset, after_cast, fp32_statistics=True,
                                                   weight_first=first, implementation=implementation)
            return (summed, result) if stream is not None else result

        operands = (*(stream if stream is not None else (row,)), *([weight] if weight is not None else []))
        outputs = (row, output) if stream is not None else (output,)

        def taken(stream=stream) -> bool:
            return norm_swiglu.rms_kernel("auto", _shape(row), None if weight is None else _shape(weight),
                                          dtype, fp32_statistics=True,
                                          branch=None if stream is None else _shape(stream[1])) == "pallas"

        yield Candidate(operands, outputs, normed, functools.partial(normed, implementation='xla'), taken)


def _swiglus(index: fusion.Index, place: int) -> Iterator[Candidate]:
    """A SwiGLU's candidates around the sigmoid at `place` (`dew.nn.activations.gated_product`): each gate it
    is made from and each value it makes of the gate's shape, the up projection the one other operand of
    that shape; or one packed projection `[..., 2, H]` both are read from, its two halves its operands'
    sources (`packed_gated_product`)."""
    from dew.nn.activations import gated_product, packed_gated_product
    from dew.nn.kernels import norm_swiglu

    anchor = index.jaxpr.eqns[place].outvars[0]
    gates = sorted(((distance, var) for var, distance in _backward(index, anchor, 3).items()
                    if var is not anchor and var.aval.ndim >= 1), key=lambda pair: -pair[0])
    outputs = sorted(((distance, var) for var, distance in _forward(index, anchor, 4).items()
                      if var is not anchor and var.aval.ndim >= 1), key=lambda pair: -pair[0])
    for (_, gate), (_, output) in itertools.product(gates, outputs):
        if gate.aval.shape != output.aval.shape:
            continue
        extras = _extras(index, gate, output)
        if extras is None or len(extras) != 1:
            continue
        ups = [var for var in _chain(index, extras[0]) if var.aval.shape == gate.aval.shape]
        if not ups:
            continue
        up, dtype = ups[-1], output.aval.dtype
        packed = _packed(index, gate, up)
        if packed is not None:
            def halves(gate_up, implementation='auto', dtype=dtype):
                return packed_gated_product("swiglu", implementation=implementation)(gate_up).astype(dtype)

            yield Candidate((packed,), (output,), halves, functools.partial(halves, implementation='xla'),
                            lambda packed=packed: norm_swiglu.packed_swiglu_kernel("auto", _shape(packed))
                            == "pallas")

        def product(g, u, implementation='auto', dtype=dtype):
            return gated_product("swiglu", implementation=implementation)(g, u).astype(dtype)

        yield Candidate((gate, up), (output,), product, functools.partial(product, implementation='xla'),
                        lambda gate=gate, up=up: norm_swiglu.swiglu_kernel("auto", _shape(gate), _shape(up))
                        == "pallas")


def _packed(index: fusion.Index, gate: jcore.Var, up: jcore.Var) -> jcore.Var | None:
    """The one value `[..., 2, H]` `gate` and `up` are each read from by a layout alone (a slice, a squeeze,
    an index), where both are; else None."""
    sources = []
    for var in (gate, up):
        made = [value for value, _ in _backward(index, var, 3).items()
                if value.aval.shape == (*gate.aval.shape[:-1], 2, gate.aval.shape[-1])]
        sources.append(made)
    shared = [var for var in sources[0] if var in sources[1]]
    return shared[0] if shared else None


def _chain(index: fusion.Index, var: jcore.Var, reach: int = 3) -> list[jcore.Var]:
    """`var` and the values it is made from by up to `reach` equations of one non-constant operand each (a
    cast, a layout, an activation), nearest first: the values an operand a region reads may be bound as,
    the call's own casts and layouts of it the region's equations (a weight cast and broadcast along the
    rows, a gate's activation)."""
    return _sources(index, var, reach)[::-1]


def _summed_rows(index: fusion.Index, anchor: jcore.Var) -> list[jcore.Var]:
    """The rows an inverse square root's statistic may be read from, farthest first: for each sum of a value
    of at least one axis among the anchor's sources (`_backward`), that value and the values it is made from
    by equations of one non-constant operand each (a cast, a square, a layout: `_sources`), those of the
    summed value's shape."""
    rows: dict[jcore.Var, None] = {}
    for var in _backward(index, anchor):
        made = index.producer.get(var)
        if made is None or jax_name(index.jaxpr.eqns[made].primitive) != "reduce_sum":
            continue
        summed = index.jaxpr.eqns[made].invars[0]
        if isinstance(summed, jcore.Var):
            rows.update(dict.fromkeys(source for source in _sources(index, summed, REACH)
                                      if source.aval.shape == summed.aval.shape))
    return list(rows)


def _sources(index: fusion.Index, var: jcore.Var, reach: int) -> list[jcore.Var]:
    """`var` and the values it is made from by up to `reach` equations of one operand each (a scale by a
    constant, a cast, a layout), farthest first."""
    chain, current = [var], var
    for _ in range(reach):
        place = index.producer.get(current)
        if place is None:
            break
        operands = list(dict.fromkeys(atom for atom in index.jaxpr.eqns[place].invars
                                      if isinstance(atom, jcore.Var) and not _constant(index, atom)))
        if len(operands) != 1:
            break
        current = operands[0]
        chain.append(current)
    return chain[::-1]


FAMILIES = (Family("rsqrt", _norms), Family("logistic", _swiglus))
"""The calls the compile point recognizes, in the order it looks for them: the norms first, so a gated norm's
activation is the norm's rather than a SwiGLU's. Attention is `attention_regions`' until its call takes a
program's mask fill (row 4783)."""
