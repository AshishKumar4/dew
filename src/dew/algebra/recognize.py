"""Recognition at the compile point (docs/design/algebra.md §2 "Fusion", row 4783): a body's regions written
as Dew's kernels where they compute what the kernels' calls compute, as the e-graph proves it.

A family (`ATTENTION`, `NORMS`) is one kind of call the compile point writes: Dew's attention, RMSNorm, its
stream and gated norms, and SwiGLU. Its math is declared once, as the call itself on its portable path
(`implementation='xla'`, attention's `'reference'`), the arithmetic its kernels are held to, so no second
account of it is written here. Around each equation of the family's anchor (a primitive its declaration always
holds: `exp`, `rsqrt`, `logistic`) the family names its candidates (`Candidate`): a call over some of the
body's values (its operands), making some of its values (its results), its static arguments read from the
region's own constants (an epsilon, a softcap, a mask's fill), each choice the call takes tried rather than
assumed. An anchor's candidates go into one e-graph (`_equalities`): the equations of the regions they span
and every candidate's declaration traced over the same operands, made canonical (`normal_form.canonical`) and
held in one term region, so a parameter, a cast or a constant the two share is one term; then saturated under
exactly the rule sets the call's numerics admit (`rules.admitted`: the exact sets alone under the accurate
forms, so a product reordered or a scale moved across one matches only under the fast forms). A candidate
whose every result is in its declaration's e-class, whose region the body reads nothing of but its results
(`_between`), and whose call the family's selector gives a kernel of Dew's (`Candidate.taken`) is the call's,
its region written as the call (`fusion.Written`). Equal, never near: a different constant, cast, axis or
order is another program, and stays as traced. """

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
from dew.algebra import SATURATION, derive, forms, fusion, normal_form, program, rules, terms
from dew.nn.kernels import generation
from dew.nn.programs import bodies, jax_name
from dew.nn.sharding import sequence_shards

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
    """A kind of call (`ATTENTION`, `NORMS`): its anchor primitive and its candidates around an equation of
    it."""

    anchor: str
    candidates: Callable[[fusion.Index, int], Iterator[Candidate]]


def recognized(jaxpr: jcore.Jaxpr, index: fusion.Index, families: Sequence[Family],
               held: frozenset[int] = frozenset()) -> list[fusion.Written]:
    """Every region of `jaxpr` (`index`'s) the e-graph proves computes a family's call and whose call takes
    a kernel of Dew's, written as that call (the module's account): `families` in order (`ATTENTION`,
    `NORMS`), an anchor's candidates in the order its family
    names them, the first equal one taken, no two regions sharing an equation nor one `held` (a product's
    epilogue's). Each region's choice is held by its structure and its constants' values (`_HELD_REGIONS`),
    the body's whole recognition by its structure and constants (`_HELD_BODIES`)."""
    key = (_body_key(jaxpr, index), held, tuple(family.anchor for family in families), derive.context(),
           _cache.KERNEL_CHOICES.value)
    if key not in _HELD_BODIES:
        _HELD_BODIES[key] = [_positioned(jaxpr, written)
                             for written in _recognizing(jaxpr, index, held, families)]
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


def _recognizing(jaxpr: jcore.Jaxpr, index: fusion.Index, held: frozenset[int],
                 families: Sequence[Family]) -> list[fusion.Written]:
    """`recognized`'s search."""
    accurate = generation.numerics().forms == "accurate"
    found: list[fusion.Written] = []
    claimed: set[int] = set(held)
    for family in families:
        for place, equation in enumerate(jaxpr.eqns):
            # An anchor makes the value its region is read around: a launch writing refs in place, its body
            # binding the anchor's primitive, makes none (an in-place rewalk, `attention_mosaic_forward`).
            if place in claimed or not equation.outvars or not _holds(equation, family.anchor):
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
    inputs = fusion.region_inputs(jaxpr, places)
    window = jcore.Jaxpr([], list(inputs), list(outputs), [jaxpr.eqns[place] for place in places])
    held = tuple(_digest(index.consts[var]) if var in index.consts else None for var in inputs)
    return program.structure(window), held, accurate, derive.context(), _cache.KERNEL_CHOICES.value


def _digest(value) -> Hashable:
    """A constant by its type and bits; a traced one (a value an enclosing trace closes over), which holds no
    bits until it runs, by its type alone, as an input is."""
    if isinstance(value, jax.core.Tracer):
        return value.aval
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
    """Whether `equation` binds the primitive `name`, or a nested program whose body does (`bodies`), as a
    jit `jax.nn.silu` traces holds its sigmoid: the region around it reads the call whole, and its equality
    its body inlined (`normal_form.canonical`)."""
    return _binds(equation, frozenset({name}))


def _binds(equation: jcore.JaxprEqn, names: frozenset[str]) -> bool:
    """Whether `equation` binds one of the primitives `names`, at any depth of the programs it nests."""
    return jax_name(equation.primitive) in names or any(
        _binds(nested, names) for body in bodies(equation) for nested in body.eqns)


def _between(index: fusion.Index, operands: Sequence[jcore.Var],
             outputs: Sequence[jcore.Var]) -> tuple[int, ...] | None:
    """The places of the equations computing `outputs` from `operands` and constants, in body order: None
    where they read another value of the body than the operands (an unbound input), hold more than `SPAN`
    equations, hold an effect or an equation that changes a derivative (`BARRIERS`, at any depth of the
    programs they nest: neither does the call written in their place), make a value the body reads outside
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
    made, last = set(outputs), max(places, default=-1)
    for place in places:
        equation = index.jaxpr.eqns[place]
        if equation.effects or _binds(equation, BARRIERS):
            return None
        for var in index.jaxpr.eqns[place].outvars:
            outside = index.readers[var] - places
            if var not in made and (var in index.escaping or outside):
                return None
            # A result is written where the region's last equation stood (`fusion._spliced`): a reader
            # outside the region before it would read it before it is made.
            if var in made and any(reader < last for reader in outside):
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
    candidate's results are already its declaration's own values in the normal form
    (`normal_form.canonical`: one term, which every rule set admits), that candidate is equal and the rest are
    not weighed: no e-graph is built; nor where not `saturating`, which weighs that alone."""
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
        computed = [held[var] for candidate, _ in spans for var in candidate.outputs]
        for candidate, _ in spans:
            computed += _declaration(candidate)(*(held[var] for var in candidate.operands))
        return computed

    closed = normal_form.canonical(jax.make_jaxpr(joint)(*(_shape(var) for var in inputs)))
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
    terms_made = region.outputs
    equal, program_at = [], 0
    declared_at = sum(len(candidate.outputs) for candidate, _ in spans)
    for candidate, _ in spans:
        pairs = [(terms_made[program_at + n], terms_made[declared_at + n])
                 for n in range(len(candidate.outputs))]
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
    results' leaves, each of Dew's choices by the platform its default branch (`forms.inlined`); traced once
    for each call, choices and types (`_DECLARATIONS`)."""
    declared = candidate.declared
    function = declared.func
    cells = tuple(cell.cell_contents for cell in function.__closure__ or ())
    keywords = tuple(sorted((function.__kwdefaults__ or {}).items()))
    key = (function.__code__, _typed(function.__defaults__), _typed(keywords), _typed(cells),
           _typed(tuple(sorted(declared.keywords.items()))), _typed(tuple(candidate.operands)))
    if key not in _DECLARATIONS:
        # Dew's own choices by the platform read as their default branch (`forms.inlined`): its product under
        # a wide arithmetic is the product a program writes; a program's own choices stay as written.
        traced = jax.make_jaxpr(lambda *values: jax.tree.leaves(forms.inlined(declared)(*values)))(
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
        return tuple(_typed(entry) for entry in value)
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


def _extras(index: fusion.Index, row: jcore.Var | Sequence[jcore.Var],
            output: jcore.Var) -> list[jcore.Var] | None:
    """The values beside `row` (one value, or several: attention's query, keys and values) the equations
    from it to `output` read: each a value of the body made from neither it nor constants alone (a weight,
    a gate, a branch, a bias), in the order first read; None where `output` is not made from every one.
    Made from a row is read as far as a candidate's operands lie from its results (twice `REACH`)."""
    rows = (row,) if isinstance(row, jcore.Var) else tuple(row)
    reached = [_forward(index, source, 2 * REACH) for source in rows]
    if not all(output in found for found in reached):
        return None
    descendants = set().union(*reached)
    extras: dict[jcore.Var, None] = {}
    seen, frontier = set(), [output]
    while frontier:
        var = frontier.pop()
        if var in seen or var in rows:
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
            summed, normalized = stream_rms_normalized(x, branch, rest[0] if rest else None, epsilon, dtype,
                                                       offset, after_cast, fp32_statistics=True,
                                                       weight_first=first, implementation=implementation)
            return (summed, normalized) if stream is not None else normalized

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


def _attentions(index: fusion.Index, place: int) -> Iterator[Candidate]:
    """Softmax attention's candidates around the exponential at `place` (Dew's
    `scaled_dot_product_attention` over `[B, S, H, D]`): its query and keys the logits product's operands or
    the values they are made from by up to two equations (a scale, a cast), its values the weights' product's
    other operand or its source, its result that product's output or a value made of it by up to three (a
    layout, a cast); a value the region reads beside them a bias where it is floating and a mask where it is
    boolean, a mask of constants alone folded into the region (`normal_form.canonical`); each scalar constant
    it divides or multiplies by a softcap, or none, each constant the region selects in place of masked logits
    its fill, or the call's own, each band its causal mask keeps a sliding window (`_selections`), the call
    causal or not, its softmax in float32 or in its operands' dtype, and its scale on the query before the
    product (Flax's) or on the float32 logits after it (jax.nn's), each tried. Its declaration is the call's
    portable path, on its reference path (`implementation='reference'`), its products as its kernels compute
    them."""
    anchor = index.jaxpr.eqns[place].outvars[0]
    logits = _nearest_product(index, _backward(index, anchor), upstream=True)
    weighed = _nearest_product(index, _forward(index, anchor), upstream=False)
    if logits is None or weighed is None:
        return
    reached = _forward(index, anchor)
    query, key = index.jaxpr.eqns[logits].invars
    values = next((atom for atom in index.jaxpr.eqns[weighed].invars if atom not in reached), None)
    if not all(isinstance(atom, jcore.Var) for atom in (query, key, values)):
        return
    sources = [_sources(index, query, 2), _sources(index, key, 2), _sources(index, values, 2)]
    made = index.jaxpr.eqns[weighed].outvars[0]
    farthest_first = sorted(((distance, var) for var, distance in _forward(index, made, 3).items()),
                            key=lambda pair: -pair[0])
    for q, k, v in itertools.product(*sources):
        if q.aval.ndim != 4 or k.aval.ndim != 4 or v.aval.ndim != 4:
            continue
        for _, output in farthest_first:
            if output.aval.shape != (*q.aval.shape[:-1], v.aval.shape[-1]):
                continue
            extras = _extras(index, (q, k, v), output)
            if extras is None or len(extras) > 1:
                continue
            logits_shape = (q.aval.shape[0], q.aval.shape[2], q.aval.shape[1], k.aval.shape[1])
            biases = [var for var in _addends(index, logits, anchor) if var.aval.shape == logits_shape]
            for extra in ([extras[0]] if extras else [None, *biases]):
                given = () if extra is None else (extra,)
                region = _between(index, (q, k, v, *given), (output,)) or ()
                fills, windows = _selections(index, (q, k, v, *given), region, output)
                # Under a sequence axis the call takes no fill: its exchanges weigh a masked key nothing.
                fills = fills if sequence_shards() == 1 else []
                yield from _attention_candidates(q, k, v, output, extra,
                                                 [None, *_scalars(index, region, frozenset({"div", "mul"}))],
                                                 [None, *fills], [None, *windows])


def _addends(index: fusion.Index, logits: int, anchor: jcore.Var) -> list[jcore.Var]:
    """The values added to the logits between their product (at `logits`) and the exponential's
    `anchor`, each made without the product: a bias a program computes from its own inputs, the queries
    among them, which `_extras` reads as the region's own arithmetic and not an operand beside it."""
    product = index.jaxpr.eqns[logits].outvars[0]
    downstream = _forward(index, product, 2 * REACH)
    upstream = _backward(index, anchor, 2 * REACH)
    found: dict[jcore.Var, None] = {}
    for var in upstream:
        place = index.producer.get(var)
        if var not in downstream or place is None or jax_name(index.jaxpr.eqns[place].primitive) != "add":
            continue
        for atom in index.jaxpr.eqns[place].invars:
            if isinstance(atom, jcore.Var) and atom not in downstream:
                # The addend as the call takes it, `[B, H, Q, K]`: it or a value it is laid out from (a
                # grouped layout's reshape, a cast).
                found.update(dict.fromkeys(var for var in _sources(index, atom, REACH) if var.aval.ndim == 4))
    return list(found)


def _nearest_product(index: fusion.Index, reached: dict[jcore.Var, int], *, upstream: bool) -> int | None:
    """The place of the `dot_general` nearest the anchor among the values `reached` (made from them where
    `upstream`, reading them otherwise), None where there is none."""
    places = []
    for var, distance in reached.items():
        made = index.producer.get(var)
        near = ([made] if made is not None else []) if upstream else index.readers[var]
        places += [(distance, place) for place in near
                   if jax_name(index.jaxpr.eqns[place].primitive) == "dot_general"]
    return min(places)[1] if places else None


def _attention_candidates(q, k, v, output, extra, softcaps, fills, windows) -> Iterator[Candidate]:
    """`_attentions`' candidates over one query, keys, values and result (the module's account)."""
    from dew.nn.attention import scaled_dot_product_attention

    dtype = output.aval.dtype
    boolean = extra is not None and extra.aval.dtype == jnp.bool_
    choices = itertools.product((False, True), softcaps, (True, False), (True, False), fills, windows)
    for causal, softcap, wide, after, fill, window in choices:
        if window is not None and not causal:
            continue

        def attend(query, key, value, *given, causal=causal, softcap=softcap, wide=wide, after=after,
                   fill=fill, window=window, implementation='auto'):
            extra_ = given[0] if given else None
            return scaled_dot_product_attention(
                query, key, value, implementation=implementation, causal=causal, softcap=softcap,
                sliding_window=window, force_fp32_for_softmax=wide, scale_logits=after, mask_fill=fill,
                mask=extra_ if boolean else None, bias=None if boolean else extra_).astype(dtype)

        operands = (q, k, v, *([extra] if extra is not None else []))
        yield Candidate(operands, (output,), attend, functools.partial(attend, implementation='reference'),
                        functools.partial(_attention_taken, attend, operands))


def _selections(index: fusion.Index, operands: Sequence[jcore.Var], places: Sequence[int],
                output: jcore.Var) -> tuple[list[float], list[int]]:
    """What the region at `places` selects, in its normal form (`normal_form.canonical`: nested programs
    inlined, constants computed): each constant of one value throughout it selects in place of masked
    values (a mask's fill: -inf, a finite large negative), and each window a constant causal band mask keeps
    (a key within `w` positions at or before its query, `w` short of the keys' count: a sliding window),
    each once."""
    if not places:
        return [], []
    read = (atom for place in places for atom in index.jaxpr.eqns[place].invars
            if isinstance(atom, jcore.Var) and atom not in index.consts
            and index.producer.get(atom) not in places)
    inputs = list(dict.fromkeys([*operands, *read]))
    run, consts = _window(index, places, inputs, [output])
    closed = normal_form.canonical(jax.make_jaxpr(lambda *values: run(*consts, *values))(
        *(_shape(var) for var in inputs)))
    fills: dict[float, None] = {}
    windows: dict[int, None] = {}
    for equation in closed.jaxpr.eqns:
        if jax_name(equation.primitive) != "select_n":
            continue
        for atom in equation.invars:
            if not isinstance(atom, jcore.Literal):
                continue
            held = np.asarray(atom.val)
            if held.dtype == np.bool_ and held.ndim >= 2:
                width = _band(held)
                if width is not None:
                    windows[width] = None
            elif held.size and jnp.issubdtype(held.dtype, jnp.floating) and np.all(held == held.flat[0]):
                fills[float(held.flat[0])] = None
    return list(fills), list(windows)


def _band(mask: np.ndarray) -> int | None:
    """The width `w` of a causal band `mask` keeps over its last two axes (query, key), the same for every
    leading index: key `j` kept for query `i` exactly where `0 <= i - j < w`; None where it keeps another
    pattern, or every key at or before its query (plain causality)."""
    queries, keys = mask.shape[-2:]
    distance = np.arange(queries)[:, None] - np.arange(keys)[None, :]
    for width in range(1, keys):
        if np.array_equal(np.broadcast_to((distance >= 0) & (distance < width), mask.shape), mask):
            return width
    return None


KEPT = frozenset({"xla", "reference"})
"""The attention routes that are no kernel of Dew's."""


def _attention_taken(attend: Callable, operands: Sequence[jcore.Var]) -> bool:
    """Whether Dew's selector gives the call a kernel of its own (the template, FlashAttention, cuDNN; none
    of `KEPT`): the call traced over its operands' shapes, its attention route read (`generation.routes`);
    not where a kernel a call names refuses it (`KernelRefused`)."""
    from dew.nn.kernels.selection import KernelRefused

    with generation.routes(alone=True) as found:
        try:
            jax.eval_shape(attend, *(_shape(var) for var in operands))
        except KernelRefused:
            # A kernel a call names that does not take the call: the program's own math runs it.
            return False
    return any(route.choice == "attention" and route.kernel not in KEPT for route in found)


ATTENTION = (Family("exp", _attentions),)
"""Attention, recognized before the products, so a product inside it is none of theirs."""

NORMS = (Family("rsqrt", _norms), Family("logistic", _swiglus))
"""The norms, after the products, so an epilogue's norm is the product's; a gated norm before a SwiGLU, so its
activation is the norm's."""
