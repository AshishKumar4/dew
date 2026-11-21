"""The program pass (docs/design/algebra.md §6.7): a function's jaxpr held as terms (`terms.region`), the
admitted rule sets run to their fixpoint or the region's limits, and its roots' cheapest forms under the cost
model extracted together and evaluated in place of the program's own. The bodies of its nested programs,
scans, loops, conditionals, checkpoints and custom derivatives' primal calls are rewritten in place, each a
region of its own, and kept bodies, never unrolled. Each rewrite is recorded in the one kernel report
(`dew.nn.kernels.templates.report.rewrites`)."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import math
import time
import weakref
from collections.abc import Callable, Collection, Hashable, Iterable, Iterator, Mapping, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from egglog import DagCostModel, EGraph, Schedule, String, Vec, get_callable_args, i64, set_
from jax._src import api_util, config as jax_config
from jax._src.interpreters import partial_eval
from jax.core import ShapedArray, eval_jaxpr
from jax.extend import core as jcore

from dew import _cache
from dew._cache import ALGEBRA
from dew.algebra import cost, extract, fusion, rules, terms
from dew.algebra.extract import Linear
from dew.nn.derivatives import Activation, Algebra, KernelGlue, Surrogate, marked_call_p
from dew.nn.kernels import generation
from dew.nn.kernels.templates import report as kernel_report
from dew.nn.programs import called, control_flow, inlined, live, with_bodies


@dataclasses.dataclass(frozen=True)
class Saturation:
    """The limits a region's saturation stops at, from its size (`limits`): nodes at `growth` per equation,
    iterations at `depth` per level of its longest chain of the language's equations, each at least one."""

    growth: int
    depth: int

    def __post_init__(self):
        if self.growth < 1 or self.depth < 1:
            raise ValueError(f"a saturation's limits are each at least one, not growth {self.growth} and "
                             f"depth {self.depth}: none would stop a region before its first iteration")

    def limits(self, region: terms.Region) -> tuple[int, int]:
        """The nodes and iterations `region` may take."""
        return self.sized(region.equations, region.depth)

    def sized(self, equations: int, depth: int) -> tuple[int, int]:
        """The nodes and iterations an input of `equations` equations whose longest chain is `depth` may
        take: a region's (`limits`), and a derivation's whose input is no region, sized by the rows its
        e-graph holds as registered (a scan's token programs' terms, its chunk's root term and leaves)."""
        return self.growth * max(1, equations), self.depth * max(1, depth)


@dataclasses.dataclass(frozen=True)
class AlgebraReport:
    """What one region's rewrite did: the equations it read, the sets admitted and the matches each fired,
    its limits, its iterations and nodes, whether a limit stopped it, the region's price before and after in
    seconds (`cost.model`; None for a region no rule may fire in, `rules.reorderable`, whose terms are never
    built and so never priced, and a scan's own derivations price no form before theirs, `reported`), the
    seconds this program's rewrite of it took, the roots whose cheaper form
    the program took (`spliced`, `_splices`: a picked form that does not land, a value another window's form
    made or a form of another type, is none), and whether an earlier window of its structure derived its
    forms (`reused`, `_derived`: its seconds a lookup's)."""

    equations: int
    admitted: tuple[str, ...]
    fired: Mapping[str, int]
    limits: tuple[int, int]
    iterations: int
    nodes: int
    limited: bool
    before: float | None
    after: float | None
    seconds: float
    spliced: int = 0
    reused: bool = False


@dataclasses.dataclass(frozen=True)
class Rewrite:
    """A region's terms (`terms.region`), each root's cheapest form by its key (`terms.key`; `picked`, empty
    where none is cheaper than the region as traced), and the report of the rewrite that chose them."""

    region: terms.Region
    picked: Mapping[Hashable, terms.Term]
    report: AlgebraReport


@contextlib.contextmanager
def passes(on: bool = True, target: str | None = None, without: Collection[str] = ()) -> Iterator[None]:
    """Within the block, the programs Dew compiles are rewritten (`on`) or not, priced for the GPU `target`
    names (None: the process's own), by every rule set but those `without` names (a receipt's arm, §1.6,
    `rules.nameable`: a name no arm may leave out is refused, as an arm without it would measure nothing): a
    context of JAX's trace, so a program held across the switch traces afresh (`dew._cache.ALGEBRA`). The
    accuracy gate holds it off for a reference arm and the float64 truth."""
    if unknown := sorted(set(without) - rules.nameable()):
        raise ValueError(f"no arm leaves out {unknown}: an arm may leave out {sorted(rules.nameable())}; a "
                         "dev-time set runs at dev time alone, and a derivation's defining set (decoupled, a "
                         "scan's own identities) makes its program, which no arm has without it")
    with ALGEBRA(dataclasses.replace(ALGEBRA.value, on=on, target=target, without=frozenset(without))):
        yield


def rewritten(fn: Callable, *, saturation: Saturation, static_argnums: int | tuple[int, ...] | None = None,
              static_argnames: str | tuple[str, ...] | None = None, kernels: bool = False) -> Callable:
    """`fn` with its program rewritten where the switch is on (`passes`): its static arguments bound as the
    jit around it binds them (`static_argnums`, `static_argnames`), traced over the others, its terms
    saturated under the rule sets the run's numerics admit (`rules.admitted`: the `exact` sets alone under
    the accurate forms, `generation.numerics`, which the trace context keys) and its values evaluated in
    their cheapest forms; with `kernels` (the compile point's, `options.program`), its regions written as
    Dew's kernels where they take them (`fusion.fused_program`), which a program a kernel itself traces (a
    fused prologue's producer, a score mod) never is. Its arguments, outputs and their structure are
    `fn`'s."""
    # The static positions and names as jit resolves them, by JAX's own inference: the spelling omitted is
    # read from the signature, and where both are given each is taken as given (a union of the two made a
    # static of an argument jit traces, passed the other way).
    numbers, names = static_argnums, static_argnames
    signature = api_util.fun_signature(fn)
    if signature is not None:
        numbers, names = api_util.infer_argnums_and_argnames(signature, numbers, names)
    numbers = set(() if numbers is None else (numbers,) if isinstance(numbers, int) else numbers)
    names = set(() if names is None else (names,) if isinstance(names, str) else names)

    @functools.wraps(fn)
    def run(*args, **kwargs):
        # Under `jax.disable_jit` a program runs as Python, its values concrete, so nothing is traced to
        # rewrite.
        if not ALGEBRA.value.on or jax_config.disable_jit.value:
            return fn(*args, **kwargs)
        accurate = generation.numerics().forms == "accurate"
        statics = _statics(args, numbers)
        fixed = {name: value for name, value in kwargs.items() if name in names}
        dynamic, named = _dynamic(args, kwargs, numbers, names)

        def bound(dynamic_args, dynamic_kwargs):
            given = iter(dynamic_args)
            whole = [args[number] if number in statics else next(given) for number in range(len(args))]
            return fn(*whole, **fixed, **dynamic_kwargs)

        closed, shaped = jax.make_jaxpr(bound, return_shape=True)(dynamic, named)
        done = rewrite(closed, jax.tree.leaves((dynamic, named)), accurate=accurate, saturation=saturation,
                       kernels=kernels)
        return jax.tree.unflatten(jax.tree.structure(shaped), done)
    return run


def rewritten_program(closed: jcore.Jaxpr, *, saturation: Saturation) -> jcore.Jaxpr:
    """`closed`, a jit's own trace of a program at the compile point, its bodies and its own math rewritten
    and its regions written as Dew's kernels where the switch is on (`passes`, `rewrite`'s walks,
    `kernels`): `closed` itself where no rule changes it, so the trace jit made lowers as it is, never
    staged again (a Dory-4B training step's replay of its unchanged program, 1.8 seconds of its
    compile). The pass's whole seconds are recorded (`_passed`)."""
    if not ALGEBRA.value.on:
        return closed
    return _passed(closed, generation.numerics().forms == "accurate", saturation, kernels=True)


def _passed(closed: jcore.Jaxpr, accurate: bool, saturation: Saturation, kernels: bool) -> jcore.Jaxpr:
    """One program's pass (`_rewritten`), its whole seconds, its walks of the program among them, recorded
    beside the regions' own (`report.record_pass`): where both entries run it, a jit's trace at the compile
    point (`rewritten_program`) and a function's own (`rewritten`'s `rewrite`)."""
    began = time.perf_counter()
    made = _rewritten(closed, accurate, saturation, kernels)
    kernel_report.record_pass(time.perf_counter() - began)
    return made


def traced_as(traced: jax.stages.Traced, closed: jcore.Jaxpr) -> jax.stages.Traced:
    """jit's own trace `traced` of a program, its program `closed` (its rewritten one, `rewritten_program`, of
    the same arguments and results): what it lowers, compiles and runs is `closed`, never traced again (row
    6977: a replay of the rewritten program was a whole second trace, 1.18 s of a decision step's 7.5)."""
    return jax.stages.Traced(traced._meta_tys_flat, {**traced._params, "jaxpr": closed}, traced._in_tree,
                             traced.out_tree, traced._consts, traced._fun_sourceinfo)


def replaying(closed: jcore.Jaxpr, structure, numbers: Iterable[int],
              names: Iterable[str]) -> Callable:
    """A function of a program's arguments computing `closed` (its rewritten program, `rewritten_program`)
    over its dynamic ones (`_dynamic`, as jit reads them), its outputs in `structure`: what a jit of it
    traces is `closed`'s program."""
    numbers, names = tuple(numbers), frozenset(names)

    def run(*args, **kwargs):
        dynamic, named = _dynamic(args, kwargs, numbers, names)
        return jax.tree.unflatten(structure, eval_jaxpr(closed.jaxpr, closed.consts,
                                                        *jax.tree.leaves((dynamic, named))))
    return run


def _saturating(accurate: bool) -> bool:
    """Whether the pass saturates a program's math: the run's numerics admitting a set past saturation aids
    (`Rules.aid`: the accurate forms' commutativity), where saturation would leave its math as it is."""
    return not all(sets.aid for sets in rules.admitted(accurate))


def _statics(args: tuple, numbers: Iterable[int]) -> set[int]:
    """The positions among `args` the static `numbers` name, one from the end where negative, as jit reads
    them."""
    return {number % len(args) for number in numbers if -len(args) <= number < len(args)}


def _dynamic(args: tuple, kwargs: Mapping, numbers: Iterable[int],
             names: Iterable[str]) -> tuple[list, dict]:
    """A call's dynamic arguments, positional and named, its static ones (`numbers`, `names`) dropped: the
    values jit traces over, in its order."""
    statics, names = _statics(args, numbers), frozenset(names)
    return ([arg for number, arg in enumerate(args) if number not in statics],
            {name: value for name, value in kwargs.items() if name not in names})


def rewrite(closed: jcore.Jaxpr, arguments: list, *, accurate: bool, saturation: Saturation,
            kernels: bool = False) -> list:
    """The outputs of `closed` (a jaxpr and its constants) over `arguments` (flattened), its bodies rewritten
    in place first (`_within`), then its own math (`_windowed`), then, with `kernels`, its regions written as
    Dew's kernels (`fusion.fused_program`)."""
    rewritten = _passed(closed, accurate, saturation, kernels)
    return eval_jaxpr(rewritten, rewritten.consts, *arguments)


WINDOW = 3
"""How deep a window reaches below the head a rule reads (`rules.heads`): a sum of two reductions of
products under a broadcast, the deepest left side the rule sets hold."""

CLUSTER = 64
"""The most equations windows that share one join into, so no term is deeper than a few hundred frames of
egglog's recursion however a program chains its heads: windows past it are held apart, the values one reads
of another read from outside it."""


def _windowed(jaxpr: jcore.Jaxpr, accurate: bool, saturation: Saturation,
              spliced: dict[int, list[jcore.JaxprEqn]] | None = None) -> jcore.Jaxpr:
    """`jaxpr` with the values its windows yield (`_windows`) computed in their cheapest forms, where those
    are cheaper than the windows as traced: each cluster saturated in an e-graph of its own (one e-graph of
    every window's terms grew its extraction with the program, past 500 seconds a gemma3n training step's),
    each cheaper value's form traced and spliced in where its equation stood, the rest of the program kept as
    traced, so a body is never traced again. Two clusters whose union passes `CLUSTER` stay apart though they
    share an equation: a value both yield is made by the first's form (`_splices`' `taken`), each form
    writing the value it replaces under its own name, so readers of either read one value. Its splices are
    kept in `spliced`, by the id of the equation each replaces, where given (`_reclosed` reads them)."""
    made_by = {var: equation for equation in jaxpr.eqns for var in equation.outvars}
    spliced = {} if spliced is None else spliced
    windows = _windows(jaxpr)
    deriving = _deriving(accurate, saturation) if windows else None
    held = _held(jaxpr) if windows else None
    for window, yielded in windows:
        began = time.perf_counter()
        done, reused = _derived(window, accurate, saturation, deriving, _declared(window, held))
        seconds = time.perf_counter() - began
        landed = _splices(window, yielded, done, made_by, spliced) if done.picked else {}
        spliced.update(landed)
        # Recorded once its splice lands: what the program took, not what the extraction picked.
        kernel_report.record_rewrite(dataclasses.replace(done.report, spliced=len(landed), seconds=seconds,
                                                         reused=reused))
    if not spliced:
        return jaxpr
    # Without the equations a spliced form left behind, which nothing reads.
    landed = [kept for equation in jaxpr.eqns for kept in spliced.get(id(equation), [equation])]
    return jaxpr.replace(eqns=live(landed, jaxpr.outvars))


WINDOWS_HELD = 4096
"""The windows' derivations a process keeps (`_derived`), its least recently read dropped past them: the
distinct windows of the steps it compiles (161 of the 2,799 a UNet's planner traced over its compiles, each
33 ms on average to derive again; row 3764)."""

_DERIVED: collections.OrderedDict[Hashable, Rewrite] = collections.OrderedDict()
"""Each window's rewrite by what derived it (`_derived`, held through `_cache.holding`), the least recently
read first."""


def structure(window: jcore.Jaxpr) -> Hashable:
    """What a window's derivation (`_region`) reads of it: each input's type (its aval: shape, dtype, weak
    type and placement, the sharding and memory its forms' values take, as `terms.parameter_key` encodes an
    abstract value); each equation's primitive, its
    parameters as a region holds them (`terms.parameters_key`), its operands (an input's or an earlier
    output's position, or a literal's value and type) and its outputs' types; and which values it yields.
    Two windows of one structure, their values named apart, derive one rewrite: every layer of a stack's,
    every compile's of one step."""
    position = {var: index for index, var in enumerate(window.invars)}

    def operand(atom) -> Hashable:
        if isinstance(atom, jcore.Literal):
            dtype, value = terms.held_bits(atom.val)
            return dtype, value.shape, value.tobytes(), terms.parameter_key(atom.aval)
        return position[atom]

    equations = []
    for equation in window.eqns:
        equations.append((terms.parameters_key(equation.primitive, equation.params),
                          tuple(map(operand, equation.invars)),
                          tuple(terms.parameter_key(var.aval) for var in equation.outvars)))
        for var in equation.outvars:
            position[var] = len(position)
    return (tuple(terms.parameter_key(var.aval) for var in window.invars), tuple(equations),
            tuple(map(operand, window.outvars)))


def _deriving(accurate: bool, saturation: Saturation) -> Hashable:
    """What a rewrite's derivation reads beside its program, which keeps every rewrite the pass holds
    (`_derived`, `_closed`): the numerics' forms, the saturation, the device it is priced for (`cost.target`,
    resolved: a named target, an assumed generation or the process's own), the matmul precision a product
    of none stated is priced at (`product.product_mode`) and the rule sets the registry admits for those
    numerics (`rules.admitted`, by name), so a target assumed, a precision set or a set registered later
    derives afresh."""
    return (accurate, saturation, cost.target(), str(jax.config.jax_default_matmul_precision),
            rules.admitted_names(accurate))


def _held(jaxpr: jcore.Jaxpr) -> terms.Region:
    """What `jaxpr` proves of each value (`terms.Region.held`, its magnitudes, and `terms.Region.apart`, the
    rest), its structure's alone (`terms.Skeleton`), its constants taken as inputs: what its windows read
    of the values they take from outside them."""
    opened = jaxpr.replace(constvars=[], invars=[*jaxpr.constvars, *jaxpr.invars])
    return terms.region(opened, forms=terms.Skeleton)


def _declared(window: jcore.Jaxpr, proven: terms.Region | None) -> dict[int, terms.Proven]:
    """What `window`'s inputs hold (`terms.Proven`, by input index): each one as the program proves it
    (`_held`), its magnitude's bound as its power of two where the program bounds one and the rest
    (`terms.Apart`), so a bound a window cuts its equations below holds within it (a scale's under a
    broadcast, three equations below the sum that reads it), and no more: a value the program proves no
    number (a NaN's tanh made outside the window) bounds nothing the window makes of it, where the window
    took every input for a finite real (spec alignment, 8993)."""
    declared: dict[int, terms.Proven] = {}
    if proven is None:
        return declared
    for index, var in enumerate(window.invars):
        bound = proven.held.get(var)
        magnitude = (None if bound is None or bound >= 1024
                     else 0.0 if bound == terms.NOTHING else math.ldexp(1.0, bound))
        declared[index] = terms.Proven(magnitude, proven.apart.get(var, terms.NOWHERE))
    return declared


def _derived(window: jcore.Jaxpr, accurate: bool, saturation: Saturation, deriving: Hashable,
             ranges: Mapping[int, terms.Proven]) -> tuple[Rewrite, bool]:
    """`window`'s rewrite (`_region`, its inputs' `ranges`) and whether an earlier window derived it: one
    derivation for each structure (`structure`), its inputs' ranges and what else it reads (`deriving`,
    `_deriving`), the process's latest `WINDOWS_HELD` kept. The planner's candidates and a resumed run's
    relowering trace one step's program again and again, each window of it as the last trace's."""
    key = (structure(window), tuple(sorted(ranges.items())), deriving)
    derived = []

    def deriving_once() -> Rewrite:
        derived.append(True)
        return _region(window, accurate=accurate, saturation=saturation, ranges=ranges)
    done = _cache.holding(_DERIVED, key, deriving_once, WINDOWS_HELD)
    return done, not derived


def _splices(window: jcore.Jaxpr, yielded: list[jcore.Var], done: Rewrite,
             made_by: Mapping[jcore.Var, jcore.JaxprEqn], taken: Mapping[int, list]) -> dict[int, list]:
    """The equations each value `window` yields (`yielded`, as the program names them, in program order) is
    made by in its cheapest form (`done`), by the id of the program's equation they stand in for (`made_by`):
    a value another window's form made already (`taken`) kept as that one made it. The forms are traced
    together, each term the roots share made once (`terms.forms_value`), and each equation spliced in at the
    first root that reads it, so a value two roots read is computed once, as extraction priced the roots
    together (spec alignment, 8995: each root traced apart emitted a shared product twice); a root of the
    form an earlier root makes is that root's value (`_placed`)."""
    # Each input as the window holds it (`fusion.typed`): in a manual mesh's body the axes it varies over too,
    # so a form traced over them varies as the values it replaces do.
    leaves = [fusion.typed(var.aval, weak=True) for var in window.invars if isinstance(var.aval, ShapedArray)]
    if len(leaves) != len(window.invars):
        return {}
    chosen = []
    for out, root in zip(yielded, done.region.outputs, strict=True):
        made_at = id(made_by[out])
        if root is None or made_at in taken:
            continue
        form = done.picked.get(terms.key(root))
        if form is None or terms.key(form) == terms.key(root):
            continue
        if made_at not in {at for _, _, at in chosen}:
            chosen.append((out, form, made_at))
    while chosen:
        traced = jax.make_jaxpr(lambda *values, forms=tuple(form for _, form, _ in chosen):
                                terms.forms_value(done.region, list(forms), list(values)))(*leaves)
        kept = [(out, form, made_at, made) for (out, form, made_at), made, aval
                in zip(chosen, traced.jaxpr.outvars, traced.out_avals, strict=True)
                if isinstance(made, jcore.Var) and made not in traced.jaxpr.invars
                and aval.strip_weak_type() == out.aval.strip_weak_type()]
        if traced.consts or len(kept) != len(chosen):
            chosen = [(out, form, made_at) for out, form, made_at, _ in kept] if not traced.consts else []
            continue
        return _placed(window, traced.jaxpr, kept)
    return {}


def _placed(window: jcore.Jaxpr, traced: jcore.Jaxpr, kept: list) -> dict[int, list]:
    """`traced`'s equations (the window's forms made together, `_splices`) each placed at the first of
    `kept`'s roots (in program order) that reads it, renamed to the program's values: its inputs the
    window's, each root's value the one it replaces; a root whose value an earlier root's form made already
    (two roots of one form) that value, held as it is (a conversion to its own type, which compiles to
    nothing)."""
    producer = {var: place for place, equation in enumerate(traced.eqns) for var in equation.outvars}
    named: dict[jcore.Var, jcore.Var] = {}
    aliases: dict[int, list[jcore.JaxprEqn]] = {}
    for out, _, made_at, made in kept:
        if made in named:
            aliases[made_at] = [jcore.new_jaxpr_eqn(
                [named[made]], [out], jax.lax.convert_element_type_p,
                {"new_dtype": out.aval.dtype, "weak_type": out.aval.weak_type, "sharding": None},
                jcore.no_effects)]
        else:
            named[made] = out
    kept = [each for each in kept if each[2] not in aliases]
    first: dict[int, int] = {}
    for spot, (_, _, _, made) in enumerate(kept):
        frontier = [producer[made]] if made in producer else []
        while frontier:
            place = frontier.pop()
            if place in first:
                continue
            first[place] = spot
            frontier += [producer[atom] for atom in traced.eqns[place].invars
                         if not isinstance(atom, jcore.Literal) and atom in producer]
    renamed: dict[jcore.Var, jcore.Var] = {**dict(zip(traced.invars, window.invars, strict=True)), **named}

    def renaming(atom: jcore.Var | jcore.Literal) -> jcore.Var | jcore.Literal:
        return atom if isinstance(atom, jcore.Literal) else renamed.get(atom, atom)

    spliced: dict[int, list[jcore.JaxprEqn]] = {made_at: [] for _, _, made_at, _ in kept}
    for place, equation in enumerate(traced.eqns):
        if place in first:
            spliced[kept[first[place]][2]].append(
                equation.replace(invars=[renaming(atom) for atom in equation.invars],
                                 outvars=[renamed.get(var, var) for var in equation.outvars]))
    return {**spliced, **aliases}


def _windows(jaxpr: jcore.Jaxpr) -> list[tuple[jcore.Jaxpr, list[jcore.Var]]]:
    """`jaxpr`'s windows, each cluster a jaxpr of its own: each head a rule may lower the price at
    (`rules.heads`) with the equations of the language that make its operands, `WINDOW` deep, windows that
    share an equation joined up to `CLUSTER` equations; its inputs each value it reads from outside it, its
    outputs each value it makes that the program reads outside it (the values the rewrite may compute anew),
    in program order; and those values as `jaxpr` names them. None where `jaxpr` holds no head."""
    made = {var: equation for equation in jaxpr.eqns for var in equation.outvars}

    def maker(atom) -> jcore.JaxprEqn | None:
        equation = None if isinstance(atom, jcore.Literal) else made.get(atom)
        return equation if equation is not None and terms.spoken_as(equation) is not None else None

    def made_as(atom) -> str | None:
        """The symbol of the language's equation that makes `atom`, None for a value from outside it."""
        operand = maker(atom)
        return None if operand is None else terms.spoken_as(operand)

    def heads_window(equation: jcore.JaxprEqn) -> bool:
        name = terms.spoken_as(equation)
        return name is not None and rules.heads(name, [made_as(atom) for atom in equation.invars])

    heads = [equation for equation in jaxpr.eqns if heads_window(equation)]
    if not heads:
        return []
    order = {id(equation): index for index, equation in enumerate(jaxpr.eqns)}
    clusters: list[dict[int, jcore.JaxprEqn]] = []
    for head in heads:
        window, frontier = {id(head): head}, [(head, 0)]
        while frontier:
            equation, depth = frontier.pop()
            for atom in equation.invars:
                operand = maker(atom)
                if operand is not None and depth + 1 < WINDOW and id(operand) not in window:
                    window[id(operand)] = operand
                    frontier.append((operand, depth + 1))
        sharing = [cluster for cluster in clusters if cluster.keys() & window.keys()]
        if sharing and len(set(window).union(*(cluster.keys() for cluster in sharing))) <= CLUSTER:
            clusters = [cluster for cluster in clusters if all(cluster is not shared for shared in sharing)]
            window = {key: member for cluster in [*sharing, window] for key, member in cluster.items()}
        clusters.append(window)
    read: dict[jcore.Var, set[int]] = {}
    for equation in jaxpr.eqns:
        for atom in equation.invars:
            if not isinstance(atom, jcore.Literal):
                read.setdefault(atom, set()).add(id(equation))
    returned = {atom for atom in jaxpr.outvars if not isinstance(atom, jcore.Literal)}
    windows = []
    for cluster in clusters:
        outside: dict[jcore.Var, None] = {}
        equations: list[jcore.JaxprEqn] = []
        outputs: list[jcore.Var] = []
        yielded: list[jcore.Var] = []
        fresh: dict[jcore.Var, jcore.Var] = {}
        members = sorted(cluster.values(), key=lambda member: order[id(member)])
        for equation in members:
            for atom in equation.invars:
                if not isinstance(atom, jcore.Literal) and atom not in fresh:
                    outside[atom] = None
            invars = [atom if isinstance(atom, jcore.Literal) else fresh.get(atom, atom)
                      for atom in equation.invars]
            fresh.update({var: jcore.Var(var.aval) for var in equation.outvars})
            outvars = [fresh[var] for var in equation.outvars]
            equations.append(equation.replace(invars=invars, outvars=outvars))
        for equation in members:
            out = equation.outvars[0]
            if out in returned or read.get(out, set()) - cluster.keys():
                outputs.append(fresh[out])
                yielded.append(out)
        windows.append((jcore.Jaxpr([], list(outside), outputs, equations), yielded))
    return windows


def _region(closed: jcore.Jaxpr, *, accurate: bool, saturation: Saturation,
            ranges: terms.Ranges = None) -> Rewrite:
    """`closed`'s roots' forms, its bodies already rewritten, chosen together from its terms saturated under
    the admitted rule sets where they are cheaper than the region as traced (`cost.cheaper`), and the report
    of the rewrite. A region no admitted set may change (`rules.firing`: one no reordering may touch, where
    no exact set past the saturation aids is admitted) builds no e-graph. The saturated
    graph's forms are extracted first; only where one differs from its root is the region as traced
    extracted too, from the graph as it stood (egglog's `push`, `pop`): a region whose forms are its roots'
    (commutativity's alone) takes one extraction."""
    began = time.perf_counter()
    admitted = rules.admitted(accurate)
    # A window's structure decides first, its egglog terms built only where a rule may fire.
    skeleton = terms.region(closed, ranges, forms=terms.Skeleton)
    firing = rules.firing(admitted, skeleton)
    if not firing:
        fired = dict.fromkeys((sets.name for sets in admitted), 0)
        report = AlgebraReport(skeleton.equations, tuple(fired), fired, saturation.limits(skeleton),
                               iterations=0, nodes=0, limited=False, before=None, after=None,
                               seconds=time.perf_counter() - began)
        return Rewrite(skeleton, {}, report)
    region = terms.region(closed, ranges)
    priced = cost.model(region, cost.fused(cost.target()))
    graph, bundle, roots = graphed(region)
    graph.push()
    saturated = saturate(graph, region, firing, saturation)
    nodes = node_count(graph)
    forms, after = extracted(graph, bundle, priced)
    changed = any(terms.key(form) != terms.key(root) for root, form in zip(roots, forms, strict=True))
    graph.pop()
    before = after
    if changed:
        _, before = extract.cheapest(graph, bundle, priced)
    picked = ({terms.key(root): form for root, form in zip(roots, forms, strict=True)}
              if changed and cost.cheaper(after, before) else {})
    # Every admitted set is reported, one that could not fire here (`rules.firing`) at no match.
    fired = {**dict.fromkeys((sets.name for sets in admitted), 0), **saturated.fired}
    report = AlgebraReport(region.equations, tuple(fired), fired, saturated.limits, saturated.iterations,
                           nodes, saturated.limited, before.value, after.value, time.perf_counter() - began)
    return Rewrite(region, picked, report)


def graphed(region: terms.Region) -> tuple[EGraph, terms.Term, list[terms.Term]]:
    """An e-graph of `region`'s terms with the facts its rules read (each term's shape, which arithmetic
    computes in its operands' dtype, which broadcasts lay along one axis, which literals are powers of two or
    one, the primitives a stable rule writes), its roots bundled, the facts saturated."""
    graph = EGraph()
    roots = region.roots()
    bundle = terms.Term.bundle(Vec[terms.Term](*roots))
    stated(graph, region, bundle)
    graph.run(rules.FACTS.saturate())
    return graph, bundle, roots


def stated(graph: EGraph, region: terms.Region, *registered) -> None:
    """The facts `region`'s parameters and literals state, registered in `graph` (`graphed`'s) beside
    `registered` in one call (egglog's every call its own program: 200 facts took 98 ms one at a time, 53 at
    once): each term's shape, which arithmetic computes in its operands' dtype, which broadcasts lay along one
    axis, which literals are powers of two or one, the primitives a stable rule writes, each term's number
    bound (`rules.log2_bound`) and floor away from zero (`rules.log2_floor`, `rules.positive`), the conversion
    back of each value of a narrow float (`rules.narrowed`), the program's reach (`rules.reach`), which a
    reordering's new intermediates are held under (§6.3), and its narrowest float's least normal power
    (`rules.least`), each extent's power of two (`rules.log2_of`) and each reduction's length's
    (`rules.sum_length`)."""
    facts: list = [*registered]
    facts += [set_(rules.shape(term)).to(Vec[i64](*(i64(size) for size in extent)))
              for term, extent in region.shapes]
    arithmetic = (jax.lax.add_p, jax.lax.mul_p, jax.lax.sub_p, jax.lax.div_p)
    for index, (primitive, params) in enumerate(region.params):
        if primitive in arithmetic and params.get("out_dtype") is None:
            facts.append(rules.plain(i64(index)))
        if primitive is jax.lax.broadcast_in_dim_p:
            rank, laid = len(params["shape"]), tuple(params["broadcast_dimensions"])
            missing = [axis for axis in range(rank) if axis not in laid]
            if len(missing) == 1 and laid == tuple(axis for axis in range(rank) if axis != missing[0]):
                facts.append(rules.along(i64(index), i64(missing[0])))
    facts += [set_(rules.stock(String(name))).to(i64(index)) for name, index in region.stock.items()]
    facts += _products(region)
    sizes = {size for _, extent in region.shapes for size in extent}
    facts += [set_(rules.log2_of(i64(size))).to(i64(max(size - 1, 0).bit_length())) for size in sorted(sizes)]
    widened = {np.dtype(narrow): f"convert_{np.dtype(narrow).name}" for narrow in terms.WIDENED}
    for term, _ in region.shapes:
        named = terms.key(term)
        bound = region.magnitudes.get(named)
        if bound is not None:
            facts += rules.made(term, i64(bound))
        held = region.floors.get(named)
        if held is not None and held.floor is not None:
            facts.append(set_(rules.log2_floor(term)).to(i64(held.floor)))
            if held.positive:
                facts.append(rules.positive(term))
        dtype = region.dtypes.get(named)
        if dtype is not None and (back := widened.get(dtype)) is not None:
            facts.append(set_(rules.narrowed(term)).to(i64(region.stock[back])))
        if (args := get_callable_args(term, terms.Term.reduce)) is not None:
            _, params = region.params[terms.integer(args[3])]
            extent = region.shape(args[0])
            length = math.prod(extent[axis] for axis in params["axes"])
            facts.append(set_(rules.sum_length(term)).to(i64(max(length - 1, 0).bit_length())))
    # A bound of numbers alone (the declared ranges', the constants', and what the equations prove of them)
    # holds where the program's values are: a new intermediate under the narrowest float's largest power of
    # two is finite wherever the program is. A bound over an unbounded operand would bound nothing: the
    # program's value may sit far below it while a new intermediate overflows (0.5 x + 0.5 y at 3e38
    # against 0.5 (x + y)), so an unbounded value has none (`terms.Magnitude`).
    floats = [jnp.finfo(dtype).max for dtype in region.dtypes.values() if jnp.issubdtype(dtype, jnp.floating)]
    if floats:
        # The largest power of two under the float's largest, exactly (float64's log2 rounds to 1024); the
        # least normal power of the narrowest float, its smallest normal's.
        facts.append(set_(rules.reach()).to(i64(math.frexp(float(min(floats)))[1] - 1)))
        narrowest = max((dtype for dtype in region.dtypes.values() if jnp.issubdtype(dtype, jnp.floating)),
                        key=lambda dtype: jnp.finfo(dtype).minexp)
        facts.append(set_(rules.least()).to(i64(int(jnp.finfo(narrowest).minexp))))
    if facts:
        graph.register(*facts)


def _products(region: terms.Region) -> list:
    """The facts a region's products state, each of a mode whose rounding is stated
    (`terms.contraction_slack`): which compute in their operands' dtype (`rules.plain`: no
    accumulation wider than their operands they round back from), which of those are matrix products
    (`rules.matrix_product`), which values are matrices (`rules.matrix`), and the broadcasts of a vector to
    each matrix's extent along either axis (`rules.lifted`), their parameters added to the region's table."""
    plain_numbers = (((1,), (0,)), ((), ()))
    facts: list = []
    for term, extent in region.shapes:
        if len(extent) == 2:
            facts.append(rules.matrix(term))
        if (args := get_callable_args(term, terms.Term.dot)) is None:
            continue
        index = terms.integer(args[2])
        _, params = region.params[index]
        left, right = (region.dtypes.get(terms.key(operand)) for operand in args[:2])
        # Its mode's powers (`terms.product_slack`): a product whose rounding nothing states has no
        # bound, so nothing here says what a rule would need to factor it or to make another in its mode.
        slack = terms.product_slack(region, index)
        if slack is None:
            continue
        facts.append(set_(rules.mode_power(i64(index))).to(i64(slack)))
        # Its contracted length's power, the elements each output sums, and its mode's beside it
        # (`rules.contracted_length`): what a product a rule makes in its place, in its mode, adds to its
        # operands' bounds (a split-precision product's one power more, as its own traced bound takes).
        (contracting, _), _ = params["dimension_numbers"]
        length = math.prod(region.shape(args[0])[axis] for axis in contracting)
        facts.append(set_(rules.contracted_length(term)).to(i64(max(length - 1, 0).bit_length() + slack)))
        wanted = params.get("preferred_element_type")
        operands = {left, right}
        # Plain only of one dtype: a product of two makes neither's (`terms.Product.made`), no plain one a
        # rule may move (base review 3232beb7c).
        if len(operands) == 1 and (wanted is None or operands == {jnp.dtype(wanted)}):
            facts.append(rules.plain(i64(index)))
            if jax.tree.map(int, params["dimension_numbers"]) == plain_numbers:
                facts.append(rules.matrix_product(i64(index)))
    matrices = {extent for _, extent in region.shapes if len(extent) == 2}
    for extent in sorted(matrices):
        for axis in (0, 1):
            region.params.append((jax.lax.broadcast_in_dim_p, {
                "shape": extent, "broadcast_dimensions": (axis,), "sharding": None}))
            facts.append(set_(rules.lifted(Vec[i64](*(i64(size) for size in extent)), i64(axis))).to(
                i64(len(region.params) - 1)))
    for index, value in enumerate(region.literals):
        # A typed key's literal (a key a selection or a layout moves) states no number.
        if terms.typed_key(value):
            continue
        literal = terms.Term.literal(i64(index))
        if _power_of_two(value):
            facts.append(rules.exact_scale(literal))
        # A finite positive real alone: an infinite scale keeps no order (a zero times it is NaN, where the
        # maximum taken first is finite), so an extremum takes it out of none (spec alignment, 8992).
        if (np.ndim(value) == 0 and jnp.issubdtype(np.asarray(value).dtype, jnp.inexact) and value > 0
                and np.isfinite(value)):
            facts.append(rules.positive_scale(literal))
        if np.ndim(value) == 0 and jnp.issubdtype(np.asarray(value).dtype, jnp.inexact) and value == 1:
            facts.append(rules.unit(literal))
    return facts


@dataclasses.dataclass(frozen=True)
class Saturated:
    """What a saturation did: the matches each rule set fired, its limits (nodes, iterations), the iterations
    it ran, and whether a limit stopped it."""

    fired: Mapping[str, int]
    limits: tuple[int, int]
    iterations: int
    limited: bool


def saturate(graph: EGraph, region: terms.Region, admitted: tuple[rules.Rules, ...],
             saturation: Saturation, facts: Schedule | None = None) -> Saturated:
    """`graph` run under the `admitted` rule sets to their fixpoint or `region`'s limits (`Saturation`), its
    facts saturated after each set: the engine's (`rules.FACTS`), or `facts` where a caller's rules read
    facts of their own (the derivation's, `dew.algebra.derive`)."""
    facts = rules.FACTS.saturate() if facts is None else facts
    fired = dict.fromkeys((admitted_rules.name for admitted_rules in admitted), 0)
    nodes, steps = saturation.limits(region)
    iterations, limited = 0, False
    while admitted:
        if iterations == steps or node_count(graph) > nodes:
            limited = True
            break
        updated = False
        for admitted_rules in admitted:
            graph.run(facts)
            ran = graph.run(admitted_rules.ruleset)
            fired[admitted_rules.name] += sum(ran.num_matches_per_rule.values())
            updated = updated or ran.updated
        iterations += 1
        if not updated:
            break
    graph.run(facts)
    return Saturated(fired, (nodes, steps), iterations, limited)


def run_within(graph: EGraph, ruleset, limits: tuple[int, int], name: str) -> Saturated:
    """`graph` run under one `ruleset`, reported as `name`, to its fixpoint or `limits` (nodes, iterations;
    `Saturation.sized`), an iteration at a time with the count read between them, as `saturate` reads it:
    its matches, the iterations it ran and whether a limit stopped it."""
    nodes, steps = limits
    iterations, matches = 0, 0
    while iterations < steps and node_count(graph) <= nodes:
        ran = graph.run(ruleset)
        iterations += 1
        matches += sum(ran.num_matches_per_rule.values())
        if not ran.updated:
            return Saturated({name: matches}, limits, iterations, limited=False)
    return Saturated({name: matches}, limits, iterations, limited=True)


def reported(graph: EGraph, sized: int, saturated: Saturated, began: float,
             after: float | None = None) -> AlgebraReport:
    """A derivation's own saturation (`run_within`: a scan's token programs', its chunk's) as the report a
    region's rewrite makes, recorded with the kernels' (`kernel_report.record_rewrite`): its size as
    registered, its limits, its iterations and rows, whether a limit stopped it, and its form's price."""
    return AlgebraReport(sized, tuple(saturated.fired), saturated.fired, saturated.limits,
                         saturated.iterations, node_count(graph), saturated.limited, None, after,
                         time.perf_counter() - began)


def extracted[Price: Linear](graph: EGraph, bundle: terms.Term, model: DagCostModel[Price], *,
                             exact: bool = False, checked: bool = False) -> tuple[list[terms.Term], Price]:
    """The roots' forms `model` prices cheapest together, each shared value once, and their price (the
    model's cost: `cost.Seconds` for the program's), extracted as `extract.cheapest` extracts them."""
    chosen, price = extract.cheapest(graph, bundle, model, exact=exact, checked=checked)
    forms = get_callable_args(chosen, terms.Term.bundle)
    if forms is None:
        raise ValueError(f"extraction chose no bundle of the program's roots: {chosen}")
    return list(forms[0].value), price


def _mapped(equation: jcore.JaxprEqn) -> bool:
    """Whether `equation` is a manual mesh's (`jax.shard_map`'s): its body, each device's program, is a region
    of its own the pass rewrites in the map's context (`_manual`), as a layer mapped per shard (a projection's
    product, `dew.nn.sharding.on_shards`) runs its epilogue's math and products within it (row 6578)."""
    from jax._src import shard_map

    return equation.primitive is shard_map.shard_map_p


@contextlib.contextmanager
def _manual(params: Mapping) -> Iterator[None]:
    """The context a manual mesh's body is traced in (`jax.shard_map`'s staging, jax/_src/shard_map.py): its
    newly manual axes bound, its abstract mesh with them manual, and its variance checked as the map checks
    it. A body rewritten in it (`_within`) traces its new values, a written region's call among them, as the
    map traced the body's own."""
    from jax._src import shard_map

    mesh, manual = params["mesh"], params["newly_manual_axes"]
    with (shard_map._extend_axis_env(mesh, manual),
          jax.sharding.use_abstract_mesh(shard_map._as_manual_mesh(mesh, manual)),
          jax_config._check_vma(params["check_vma"])):
        yield


def _holds_regions(equation: jcore.JaxprEqn) -> bool:
    """Whether the pass rewrites `equation`'s bodies in place, each a region of its own (§1.2): a call's
    program (`Primitive.inlined_call`, `dew.nn.programs.called`'s: a nested program, a checkpoint, a custom
    derivative's or batching rule's primal call, its rules kept) and control flow's
    (`dew.nn.programs.control_flow`: a scan's, a loop's predicate and body, a conditional's branches; nothing
    moves across a body's boundary or a loop's carry). A marked span is read by its kind (`WITHIN`); a manual
    mesh's body in its own context (`_mapped`); any other holder of a program (a native kernel's) is an opaque
    leaf."""
    primitive = equation.primitive
    return (primitive.inlined_call and primitive is not marked_call_p) or control_flow(primitive)


def _within(jaxpr: jcore.Jaxpr, accurate: bool, saturation: Saturation, kernels: bool) -> jcore.Jaxpr:
    """`jaxpr`, its constants kept, with each body (`_holds_regions`) of its equations rewritten, the equation
    itself kept: `jaxpr` itself, and each equation itself, where no body changes. A program's identity is what
    JAX shares its lowering by: every call site of one jitted function holds one body, lowered once (one
    function of the module, its kernels compiled once), so a pass that rebuilt an unchanged body at each call
    site lowered and compiled every layer's apart (a Qwen3.5-4B training step's compile 52 to 487 seconds on
    an H100, nothing rewritten)."""
    changed = False
    equations = []
    for equation in jaxpr.eqns:
        if _mapped(equation):
            # A manual mesh's body, each device's program, rewritten in its own context (`_manual`), within
            # the context the equation was recorded in (its abstract mesh among it: a map staged on a mesh of
            # its own beside the caller's, JAX's replay of an equation, `JaxprEqn.ctx`).
            with equation.ctx.manager, _manual(equation.params):
                params = with_bodies(equation.params,
                                     lambda body: _closed(body, accurate, saturation, kernels))
        elif _holds_regions(equation) or _marked(equation, WITHIN) or (
                _marked(equation, (Algebra,)) and _opening(equation) is None):
            params = with_bodies(equation.params,
                                 lambda body: _closed(body, accurate, saturation, kernels))
        elif _opening(equation) is not None:
            # Its body's own bodies rewritten, its math left to the region it opens into (`_opened`).
            params = with_bodies(equation.params,
                                 lambda body: _within(body, accurate, saturation, kernels))
        else:
            params = equation.params
        if any(params[name] is not value for name, value in equation.params.items()):
            equation = equation.replace(params=params)
            changed = True
        equations.append(equation)
    return jaxpr.replace(eqns=equations) if changed else jaxpr


WITHIN = (Surrogate, KernelGlue, Activation)
"""The marked spans' kinds (`dew.nn.derivatives.marked`) whose body the pass rewrites within its call, as it
rewrites a body (`_holds_regions`): a surrogate's bodies, a kernel derivative's glue, and an activation's
gated or ungated product (the fusion seam its epilogue reads) (docs/design/algebra.md §1.2).
A protected span is an opaque leaf, its body untouched; an Algebra span opens into the region around it
(`_opened`)."""


def _marked(equation: jcore.JaxprEqn, kinds: tuple[type, ...]) -> bool:
    """Whether `equation` is a marked span of one of `kinds`."""
    return equation.primitive is marked_call_p and isinstance(equation.params["kind"], kinds)


def _reclosed(jaxpr: jcore.Jaxpr, spans: Sequence[_Span], spliced: Mapping[int, list]) -> jcore.Jaxpr:
    """`jaxpr`, the windowed program (`_windowed`, `spliced` its splices by the id of the equation each
    replaced), with each opened span (`spans`) no splice crosses called again as it was: none of its body's
    equations replaced, none of their values read by a spliced form but its outputs, its outputs each a value
    its body made and no two of them one value (one value twice would bind one name twice), so an Algebra
    marker stands wherever no rewrite lands across its boundary (spec alignment, 8998: one splice anywhere
    kept every span of the program opened). The program is indexed once and written once."""
    read = {atom for made in spliced.values() for equation in made for atom in equation.invars
            if not isinstance(atom, jcore.Literal)}
    position = {id(equation): place for place, equation in enumerate(jaxpr.eqns)}
    calls: dict[int, jcore.JaxprEqn] = {}
    removed: set[int] = set()
    for span in spans:
        # A splice of any of its equations keeps it open: a replaced equation is in the program no more.
        if span.inlined & spliced.keys():
            continue
        inlined = {made for made in span.inlined if made in position}
        if not inlined or inlined & removed:
            continue
        inside = {var for made in inlined for var in jaxpr.eqns[position[made]].outvars}
        returned = span.returned
        if ((read & inside) - set(returned) or len(set(map(id, returned))) != len(returned)
                or not all(isinstance(atom, jcore.Var) and atom in inside for atom in returned)):
            continue
        calls[min(position[made] for made in inlined)] = span.call.replace(outvars=list(returned))
        removed |= inlined
    if not calls:
        return jaxpr
    equations = []
    for place, equation in enumerate(jaxpr.eqns):
        if place in calls:
            equations.append(calls[place])
        if id(equation) not in removed:
            equations.append(equation)
    return jaxpr.replace(eqns=equations)


def _opening(equation: jcore.JaxprEqn) -> jcore.Jaxpr | None:
    """The body of an Algebra span the pass opens into the region around it (`_opened`), the program its
    call runs on its operands one for one (`called`), None for any other equation: a span of no effects
    over arrays alone, whose renamed variables no effect names (a reference's would). Any other Algebra
    span is rewritten within its call."""
    opens = (_marked(equation, (Algebra,)) and not equation.effects
             and all(isinstance(var.aval, ShapedArray) for var in (*equation.invars, *equation.outvars)))
    body = called(equation) if opens else None
    # A span whose body holds constants (an array it closes over) opens not: inlined, they would be the
    # enclosing program's, which a manual mesh's body and a checkpoint's bind none of (JAX stages both without
    # constants, and their replay and lowering pass none), and a form reading them is no splice's
    # (`_splices`); it is rewritten within its call, as any other marked span (row 6578's review, F1).
    return body if body is not None and not body.consts else None


def _rewritten(closed: jcore.Jaxpr, accurate: bool, saturation: Saturation, kernels: bool) -> jcore.Jaxpr:
    """`closed`'s bodies rewritten in place (`_within`), then its own math (`_windowed`) with each Algebra
    span's body opened into it (`_opened`), so a window reaches across the span's boundary, where the run's
    numerics admit a rewrite of it (`_saturating`), then, with `kernels`, its products fused with their
    consumers where the GEMM template takes them and its attention and norms written as Dew's where a kernel
    of Dew's takes them (`fusion.fused_program`), whatever the numerics admit: accurate forms limit
    reassociation alone (row 4730). The program with its spans kept as traced where no window's form lands
    (`closed` itself where nothing changed). The pass reads the program without its unread work (`_read`),
    so nothing it discards is derived, written as a kernel, certified or built."""
    read = _read(closed)
    made = _within(read, accurate, saturation, kernels)
    if _saturating(accurate):
        spans: list[_Span] = []
        opened = _opened(made, spans)
        spliced: dict[int, list[jcore.JaxprEqn]] = {}
        windowed = _windowed(opened, accurate, saturation, spliced)
        made = made if windowed is opened else _reclosed(windowed, spans, spliced)
    made = fusion.fused_program(made) if kernels else made
    return closed if made is read else made


def _read(jaxpr: jcore.Jaxpr) -> jcore.Jaxpr:
    """`jaxpr`'s equations without the work none of its outputs reads, its binders as its callers bind them:
    JAX's dead-code elimination (`partial_eval.dce_jaxpr`), whose rules enter every region, a call, a marked
    span (`create_call_primitive`'s), a checkpoint, a custom derivative, a loop, and keep of it what is read
    and every effect. A start's model forward is such work: flax makes the parameters inside it, so a block's
    region both makes what the start keeps and computes what it never reads, which a walk of the program's
    own equations alone keeps whole (row 6783: lm-moe's start certified and built two grouped launches of
    its forward, 7.9 s of its compile, which XLA then dropped)."""
    kept, _ = partial_eval.dce_jaxpr(jaxpr, used_outputs=True, instantiate=True)
    return jaxpr.replace(eqns=kept.eqns)


class _Span(NamedTuple):
    """An Algebra span `_opened` opened: its call as the program reads it, the ids of the equations its body
    was inlined as, and the values its outputs became."""

    call: jcore.JaxprEqn
    inlined: frozenset[int]
    returned: list


def _opened(jaxpr: jcore.Jaxpr, spans: list[_Span] | None = None) -> jcore.Jaxpr:
    """`jaxpr` with each Algebra span's call it opens (`_opening`, one of no constants) replaced by its body's
    equations, nested spans opened too, over the call's operands, and each reader of the call's results
    reading what its body computes them as; `jaxpr` itself where it holds none. Each span it opens recorded
    in `spans` (`_Span`), where given, so one no rewrite crosses is closed again (`_reclosed`)."""
    if all(_opening(equation) is None for equation in jaxpr.eqns):
        return jaxpr
    computed: dict[jcore.Var, jcore.Var | jcore.Literal] = {}
    equations: list[jcore.JaxprEqn] = []

    def outer(atom: jcore.Var | jcore.Literal) -> jcore.Var | jcore.Literal:
        return atom if isinstance(atom, jcore.Literal) else computed.get(atom, atom)

    for equation in jaxpr.eqns:
        equation = equation.replace(invars=[outer(atom) for atom in equation.invars])
        opening = _opening(equation)
        if opening is None:
            equations.append(equation)
            continue
        made, returned = inlined(_opened(opening), equation.invars)
        equations += made
        computed.update(zip(equation.outvars, returned, strict=True))
        if spans is not None:
            spans.append(_Span(equation, frozenset(map(id, made)), list(returned)))
    return jaxpr.replace(eqns=equations, outvars=[outer(atom) for atom in jaxpr.outvars])


@dataclasses.dataclass(frozen=True)
class _Body:
    """A body's rewrite as `_closed` holds it: its program (None where nothing changed), what its rewrite
    reported (`fusion.Recorded`), replayed where it is written again, and its regions' rewrites."""

    made: jcore.Jaxpr | None
    record: fusion.Recorded
    rewrites: tuple


_REWRITTEN: weakref.WeakKeyDictionary[jcore.Jaxpr, dict[Hashable, _Body]] = (
    weakref.WeakKeyDictionary())
"""Each body's rewrites (`_closed`) by what a rewrite read (the numerics' forms, the saturation, the
switch's target), held while the body lives: every call site of one body, and every retrace that holds it
(JAX's tracing cache gives a function's one body), takes one rewritten object, which JAX lowers once. A body
nothing changes holds None, not itself, which would keep it alive."""



def _closed(closed: jcore.Jaxpr, accurate: bool, saturation: Saturation, kernels: bool) -> jcore.Jaxpr:
    """A body's program rewritten, over the same inputs, weak types included, and the same outputs, the same
    object for every call site and retrace of one body (`_REWRITTEN`), the body itself where nothing
    changes; a body whose constants another holds (an open jaxpr), or that reads a value of no shape, as it
    is."""
    if len(closed.consts) != len(closed.constvars):
        return closed
    if not all(isinstance(aval, ShapedArray) for aval in closed.in_avals):
        return closed
    from dew._cache import KERNEL_CHOICES
    from dew.algebra import derive

    rewrites = _REWRITTEN.setdefault(closed, {})
    # With kernels, the context the substitutions are decided in too (the derivation's: the mesh, the target
    # and the numerics; the kernel choices a call names; the launches the compiles refused), so a body
    # rewritten under one choice is none of another's (spec alignment, 8994). Not fusion's whole choice
    # context: its trace context holds every context variable set, a call's arrays among them, which this
    # cache, held as long as the body is, would keep alive.
    key = (_deriving(accurate, saturation), kernels,
           (derive.context(), KERNEL_CHOICES.value, kernel_report.refusal_entries()) if kernels else None)
    held = rewrites.get(key)
    reused = held is not None and held.record.still()
    if held is None or not reused:
        with kernel_report.recording() as regions:
            made, record = fusion.recorded(lambda: _rewritten(closed, accurate, saturation, kernels))
        held = rewrites[key] = _Body(None if made is closed else made, record, tuple(regions.rewritten))
    body = held
    # Reported where it is written, as its rewrite reported it, and on a reuse its regions' reports too, so a
    # report around the program holds the body's however often it is written (`fusion._traced`'s, a
    # region's).
    body.record.replay()
    if reused:
        for made_report in body.rewrites:
            kernel_report.record_rewrite(dataclasses.replace(made_report, reused=True))
    return closed if body.made is None else body.made


def _power_of_two(value) -> bool:
    """Whether a literal is a scalar power of two, a multiply by which rounds nothing: a float's of any dtype,
    ml_dtypes' (bfloat16, float8) among them, which numpy's own type tree does not place among its floats."""
    if np.ndim(value) != 0 or not jnp.issubdtype(np.asarray(value).dtype, jnp.floating):
        return False
    mantissa, _ = np.frexp(float(value))
    return mantissa in (0.5, -0.5)


def node_count(graph: EGraph) -> int:
    """The rows of every table of `graph`, its nodes and its facts' (egglog states no node count of its
    own)."""
    return sum(size for _, size in graph.all_function_sizes())


def priced(closed: jcore.Jaxpr, target: generation.GpuTarget) -> cost.Seconds:
    """`closed` as XLA's fusion runs it on `target`: the equations its outputs read (`_read`: none it leaves
    dead) as terms (`terms.region`) in their e-graph (`graphed`), extracted greedily at XLA's schedule
    (`cost.fused`), which on a graph no rule rewrote is the program as written, each shared value priced
    once: its leaves' bytes read, its reductions and products each a kernel, its elementwise and
    special-function work per element (`cost.work`), and one kernel at least, its outputs' fusion, which an
    elementwise program's launch is (a producer formed whole before a GEMM). The one price the algebra
    gives an XLA program it does not rewrite (a fused GEMM's prologue sides, `fused_gemm.priced`); its
    `Seconds`, whose work no rate of `target` prices its schedule tells (`Schedule.unpriced`)."""
    region = terms.region(_read(closed))
    graph, bundle, _ = graphed(region)
    schedule = cost.fused(target)
    found = extract.cheapest(graph, bundle, cost.model(region, schedule))[1]
    launched = cost.Work(kernels=max(0.0, 1.0 - found.work.kernels))  # its outputs' fusion, at least one
    return found + cost.Seconds(launched, schedule)

