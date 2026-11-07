"""Recognition at the compile point (docs/design/algebra.md §2 "Fusion", row 4783): a body's regions written
as Dew's kernels where they compute what the kernels' calls compute, as the e-graph proves it.

A family (`FAMILIES`) is one kind of call the compile point writes: Dew's RMSNorm, its stream and gated norms,
SwiGLU and attention. Its math is declared once, as the call itself on its portable path
(`implementation='xla'`), the arithmetic its kernels are held to, so no second account of it is written here.
Around each equation of the family's anchor (a primitive its declaration always holds: `rsqrt`, `logistic`,
`exp`) the family names its candidates (`Candidate`): a call over some of the body's values (its operands),
making some of its values (its results), its static arguments read from the region's own constants (an
epsilon, a softcap), each choice the call takes tried rather than assumed. An anchor's candidates go into one
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
import itertools
from collections.abc import Callable, Hashable, Iterator, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from egglog import eq
from jax.extend import core as jcore

from dew.algebra import SATURATION, fusion, program, rules, terms
from dew.nn.kernels import generation
from dew.nn.programs import jax_name

REACH = 8
"""How far from its anchor a family looks for a candidate's operands and results, in equations: past the
deepest declaration's chain on either side of its anchor (attention's logits to its value product)."""

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
    epilogue's). Each candidate's decision is fusion's (`fusion._decided`), held by its region's structure
    and context."""
    accurate = generation.numerics().forms == "accurate"
    found: list[fusion.Written] = []
    claimed: set[int] = set(held)
    for family in FAMILIES:
        for place, equation in enumerate(jaxpr.eqns):
            if jax_name(equation.primitive) != family.anchor or place in claimed:
                continue
            spans = [(candidate, places) for candidate in family.candidates(index, place)
                     if (places := _between(index, candidate.operands, candidate.outputs)) is not None
                     and not claimed & set(places)]
            equal = _lazily(lambda spans=spans: _equalities(index, spans, accurate))
            for number, (candidate, places) in enumerate(spans):
                def decide(number=number, candidate=candidate, equal=equal) -> bool:
                    return equal()[number] and candidate.taken()

                if fusion._decided(jaxpr, places, candidate.operands, candidate.outputs, decide):
                    found.append(fusion.Written(places, candidate.operands, candidate.outputs, places[-1],
                                                candidate.call))
                    claimed.update(places)
                    break
    return found


def _lazily(compute: Callable[[], list[bool]]) -> Callable[[], list[bool]]:
    """`compute`'s result, computed at its first call alone: an anchor whose every candidate's decision is
    held (`fusion._decided`) builds no e-graph."""
    return functools.cache(compute)


def _between(index: fusion.Index, operands: Sequence[jcore.Var],
             outputs: Sequence[jcore.Var]) -> tuple[int, ...] | None:
    """The places of the equations computing `outputs` from `operands` and constants, in body order: None
    where they read another value of the body than the operands (an unbound input), hold more than `SPAN`
    equations, or make a value the body reads outside them but the outputs."""
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
    results = set(outputs)
    for place in places:
        for var in index.jaxpr.eqns[place].outvars:
            if var not in results and (var in index.escaping or not index.readers[var] <= places):
                return None
    return tuple(sorted(places))


def _equalities(index: fusion.Index, spans: Sequence[tuple[Candidate, tuple[int, ...]]],
                accurate: bool) -> list[bool]:
    """Whether each candidate's results are its declaration's, in one e-graph of every span's equations and
    every declaration, saturated under the rule sets the numerics admit (the module's account)."""
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
    window = jcore.Jaxpr([var for var in index.consts if _reads(index, places, var)], inputs, wanted,
                         [index.jaxpr.eqns[place] for place in places])
    consts = [index.consts[var] for var in window.constvars]
    run = jcore.jaxpr_as_fun(jcore.ClosedJaxpr(window, consts))

    def joint(*values):
        held = dict(zip(wanted, run(*values), strict=True))
        results = [held[var] for candidate, _ in spans for var in candidate.outputs]
        for candidate, _ in spans:
            results += list(jax.tree.leaves(candidate.declared(*(held[var] for var in candidate.operands))))
        return results

    closed = _canonical(jax.make_jaxpr(joint)(*(jax.ShapeDtypeStruct(var.aval.shape, var.aval.dtype)
                                                 for var in inputs)))
    region = terms.region(closed)
    graph, _, roots = program.graphed(region)
    program.saturate(graph, region, rules.admitted(accurate), SATURATION)
    outputs = sum(len(candidate.outputs) for candidate, _ in spans)
    equal, declared = [], outputs
    taken = 0
    for candidate, _ in spans:
        pairs = range(len(candidate.outputs))
        equal.append(all(roots[taken + n] is not None and roots[declared + n] is not None
                         and graph.check_bool(eq(roots[taken + n]).to(roots[declared + n])) for n in pairs))
        taken += len(candidate.outputs)
        declared += len(candidate.outputs)
    return equal


def _reads(index: fusion.Index, places: Sequence[int], var: jcore.Var) -> bool:
    """Whether an equation at `places` reads `var`."""
    return bool(index.readers[var] & set(places))


def _canonical(closed: jcore.ClosedJaxpr) -> jcore.ClosedJaxpr:
    """`closed` as recognition compares it: its nested programs inlined (a jit, a closed call, a custom
    derivative's primal), each value it makes from no input computed into a constant, every constant a
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


BODY_PARAMS = {"pjit": "jaxpr", "closed_call": "call_jaxpr", "core_call": "call_jaxpr",
               "custom_jvp_call": "call_jaxpr", "custom_vjp_call": "call_jaxpr",
               "custom_vjp_call_jaxpr": "fun_jaxpr", "remat2": "jaxpr", "checkpoint": "jaxpr"}
"""The nested programs `_evaluated` runs in place, by the parameter holding each one's body: what its call
computes, the primal of a custom derivative's (its derivative is the call's own once it is written)."""


def _evaluated(jaxpr: jcore.Jaxpr, consts: Sequence, *values) -> list:
    """`jaxpr` run over `values` with every nested program (`BODY_PARAMS`) run in place: under a trace its
    equations each bind once, a value of constants alone computed where it is made."""
    env: dict[jcore.Var, object] = dict(zip(jaxpr.constvars, consts, strict=True))
    env.update(zip(jaxpr.invars, values, strict=True))

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
            bound = equation.primitive.bind(*operands, **equation.params)
            outputs = bound if equation.primitive.multiple_results else [bound]
        env.update(zip(equation.outvars, outputs, strict=True))
    return [read(atom) for atom in jaxpr.outvars]


def _equation_key(equation: jcore.JaxprEqn) -> Hashable | None:
    """An equation's identity as its value's (`_canonical`): its primitive, parameters and operands, each
    literal by its value; None for one holding an effect, whose repetition is not its value's."""
    if equation.effects:
        return None
    operands = tuple(terms.literal_key(atom) if isinstance(atom, jcore.Literal) else atom
                     for atom in equation.invars)
    return (*terms.parameters_key(equation.primitive, equation.params), operands)


def _forward(index: fusion.Index, var: jcore.Var, reach: int = REACH) -> dict[jcore.Var, int]:
    """The values `var` reaches within `reach` equations, with their distance."""
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


def _backward(index: fusion.Index, var: jcore.Var, reach: int = REACH) -> dict[jcore.Var, int]:
    """The values `var` is made from within `reach` equations, with their distance."""
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
    """The values beside `row` the equations from `row` to `output` read: each a value of the body made
    from neither `row` nor constants alone (a weight, a gate, a branch), in the order first read; None
    where `output` is not made from `row`."""
    descendants = _forward(index, row, SPAN)
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
    closed over)."""
    if var in index.consts:
        return True
    place = index.producer.get(var)
    if place is None:
        return False
    return all(isinstance(atom, jcore.Literal) or _constant(index, atom)
               for atom in index.jaxpr.eqns[place].invars)


def _scalars(index: fusion.Index, places: Sequence[int]) -> list[float]:
    """The scalar floating constants the equations at `places` read, each once: the static arguments a
    family's call may take (an epsilon, a softcap)."""
    found: dict[float, None] = {}
    for place in places:
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
    region the e-graph proves is the one written. A row made by a sum the body reads on is tried as a
    stream's too, the sum among the results."""
    anchor = index.jaxpr.eqns[place].outvars[0]
    rows = sorted(((distance, var) for var, distance in _backward(index, anchor).items()
                   if var is not anchor and var.aval.ndim >= 1), key=lambda pair: -pair[0])
    outputs = sorted(((distance, var) for var, distance in _forward(index, anchor).items()
                      if var is not anchor and var.aval.ndim >= 1), key=lambda pair: -pair[0])
    for (_, row), (_, output) in itertools.product(rows, outputs):
        if row.aval.shape != output.aval.shape or not jnp.issubdtype(output.aval.dtype, jnp.floating):
            continue
        extras = _extras(index, row, output)
        if extras is None:
            continue
        width = row.aval.shape[-1]
        weights = [var for var in extras if var.aval.shape == (width,)]
        gates = [var for var in extras if var.aval.shape == row.aval.shape]
        if len(weights) > 1 or len(gates) > 1 or len(weights) + len(gates) != len(extras):
            continue
        weight = weights[0] if weights else None
        epsilons = _scalars(index, _between(index, (row, *extras), (output,)) or ())
        yield from _norm_candidates(index, row, output, weight, gates[0] if gates else None, epsilons)


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
    choices = [(False, False)] if weight is None else [(False, False), (False, True), (True, False),
                                                       (True, True)]
    for stream, epsilon, (offset, after_cast) in itertools.product(streams, epsilons, choices):
        def normed(x, *rest, stream=stream, epsilon=epsilon, offset=offset, after_cast=after_cast,
                   implementation='auto'):
            branch, rest = (rest[0], rest[1:]) if stream is not None else (None, rest)
            summed, result = stream_rms_normalized(x, branch, rest[0] if rest else None, epsilon, dtype,
                                                   offset, after_cast, fp32_statistics=True,
                                                   implementation=implementation)
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
        if extras is None or len(extras) != 1 or extras[0].aval.shape != gate.aval.shape:
            continue
        up, dtype = extras[0], output.aval.dtype
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


FAMILIES = (Family("rsqrt", _norms), Family("logistic", _swiglus))
"""The calls the compile point recognizes, in the order it looks for them: the norms first, so a gated norm's
activation is the norm's."""
