"""The generator (docs/design/algebra.md §6.11): each declared stable form (`StableForm`) traced at each of
its cases, saturated with every rule set dev time runs (`rules.admitted(dev=True)`), its candidates measured
against its float64 truth on its cases' draws (check 1, `dew.parity.judge`) and the cheapest its policy
admits chosen (`dew.algebra.error.choose`, §1.5) under each policy, priced for each target; the choices
written beside the declaring module as ordinary Python (`<module>_derived.py`, `printed.source`). The
generated module uses only `dew.algebra._generated` for case dispatch, not the registry, rules or
extractor. Its header names the declarations, rule sets, receipts and egglog version.
`python -m dew.algebra.generate [--check]`; `--check` regenerates in memory and fails
where a written module differs, which tests/test_algebra_generate.py runs on every train.

A declaration's choice is the platform's that measures it: the draws are seeded and fixed, so generation is
deterministic on one platform, and the module records the platform it was measured on."""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import functools
import importlib
import importlib.metadata
import importlib.util
import itertools
import logging
import sys
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray
from jax.extend import core as jcore

from dew import parity
from dew.algebra import _component_sharing, _generated, error, printed, rules
from dew.algebra._generated import case_of
from dew.nn import programs
from dew.nn.kernels import generation
from dew.nn.precision import Numerics

_log = logging.getLogger(__name__)

POLICIES = ("default", "accurate")
"""The precision policies a declaration is generated for (§1.5): under default numerics a candidate is
admitted against the form as written, under the accurate numerics against every other candidate."""

TESTS = "tests/test_algebra_generate.py"
"""The suite every generated module is held by: its forms on par with their declarations (check 1) and the
written module what generation writes (`--check`)."""

DECLARING = ("dew.algebra.acceptance", "dew.algebra.attention_cases", "dew.algebra.format_cases",
             "dew.algebra.recurrence_cases", "dew.sampling.guidance_cases")
"""Modules that declare generated functions, imported before generation reads `DECLARATIONS`."""


@dataclasses.dataclass(frozen=True)
class Cases:
    """Where a declaration is generated and checked: each case's operands, by shape and dtype; the draw of a
    case's operands by seed (`draw(seed, operands)`: the gate's drawn and adversarial families, the
    declaration's own); the seeds its check 1 judges over; the policies it is generated for (`POLICIES`);
    and the targets it is priced for (`dew.algebra.cost.target`'s names)."""

    operands: tuple[tuple[jax.ShapeDtypeStruct, ...], ...]
    draw: Callable[[int, tuple[jax.ShapeDtypeStruct, ...]], tuple]
    seeds: int
    policies: tuple[str, ...] = POLICIES
    targets: tuple[str, ...] = ("sm90",)


@dataclasses.dataclass(frozen=True)
class StableForm:
    """A numeric function of Dew's own: its math as written (the form as written and, at float64, the
    truth) and the cases it is generated at."""

    function: Callable
    cases: Cases


@dataclasses.dataclass(frozen=True)
class ComponentForm:
    """A flat template or surrogate component. The surrounding wrapper, static arguments and initializer
    stay with their owner. A separate fixture is required for each bound static form."""

    function: Callable
    truth: Callable
    cases: Cases

def normal_operands(seed: int, operands: tuple[jax.ShapeDtypeStruct, ...]) -> tuple[np.ndarray, ...]:
    """A case's draw of standard normal operands, each at its shape and dtype: a component's ordinary
    family."""
    rng = np.random.default_rng(seed)
    return tuple(np.asarray(rng.standard_normal(a.shape), dtype=a.dtype) for a in operands)


def float32_cases(*shapes: tuple[tuple[int, ...], ...]) -> Cases:
    """A component's float32 cases, operands at each of `shapes`' cells drawn normal (`normal_operands`) over
    eight seeds under default numerics."""
    return Cases(tuple(tuple(jax.ShapeDtypeStruct(shape, jnp.float32) for shape in cell) for cell in shapes),
                 normal_operands, seeds=8, policies=("default",))


_Tree = tuple[jax.tree_util.PyTreeDef]


@dataclasses.dataclass(frozen=True)
class SurrogateCase:
    """One static argument tuple's flat value and tangent bodies, the latter keyed by inactive slots."""

    name: str
    forward: ComponentForm
    tangents: Mapping[tuple[int, ...], ComponentForm]
    output_tree: _Tree
    input_tree: _Tree | None = None

    def restore(self, flat):
        leaves = flat if isinstance(flat, tuple) else (flat,)
        return jax.tree.unflatten(self.output_tree[0], leaves)

    def tangent_name(self, inactive: tuple[int, ...]) -> str:
        suffix = "_zero_" + "_".join(map(str, inactive)) if inactive else ""
        return self.name + "_tangent" + suffix


@dataclasses.dataclass(frozen=True)
class SurrogateBodies:
    """A written surrogate and its opt-in checked body cases. Production keeps using the written rule."""

    forward: Callable
    tangent: Callable
    nondiff_argnums: tuple[int, ...]
    symbolic_zeros: bool
    cases: tuple[tuple[tuple, SurrogateCase], ...]
    """Each case beside its static arguments, as pairs: a mapping would join `0.0` and `-0.0`, or `True` and
    `1`, before `_exact_key` reads them."""
    _keyed: dict[Hashable, SurrogateCase] = dataclasses.field(init=False, repr=False, compare=False)
    """Each case by its static arguments' exact key (`_exact_key`)."""

    def __post_init__(self):
        # Checked whole before any registration (`declared`): one exact static key a case, hashable, and a
        # name of its own for each body, so no case is registered half or over another.
        keyed: dict[Hashable, SurrogateCase] = {}
        for fixed, case in self.cases:
            key = _exact_key(fixed)
            if key is None:
                raise ValueError(f"surrogate case {case.name}: static arguments {fixed!r} are unhashable")
            if key in keyed:
                raise ValueError(f"surrogate cases {keyed[key].name} and {case.name}: one key, {fixed!r}")
            keyed[key] = case
        names = [name for _, case in self.cases
                 for name in (case.name, *map(case.tangent_name, case.tangents))]
        if len(set(names)) != len(names):
            raise ValueError(f"surrogate cases share a body's name: {sorted(names)}")
        object.__setattr__(self, "_keyed", keyed)

    def declared(self, module: str) -> SurrogateBodies:
        """Register each case's forward and tangent bodies, each under its own name (`declare`)."""
        for _, case in self.cases:
            declare(module, case.name, case.forward)
            for inactive, form in case.tangents.items():
                declare(module, case.tangent_name(inactive), form)
        return self

    def _case(self, fixed: tuple) -> SurrogateCase | None:
        # Static arguments of no exact key (an unhashable one) match no declared case: the written body runs.
        key = _exact_key(fixed)
        return None if key is None else self._keyed.get(key)

    def assembled(self, emitted: ModuleType, *, policy: str = "default", target: str = "sm90"):
        """Opt into the generated module `emitted`'s cases, using the original surrogate rule's argument and
        value dependency: a call whose case its table holds (`_generated.found`, one lookup) runs that form,
        any other the written rule."""
        from jax.custom_derivatives import SymbolicZero

        from dew.nn.derivatives import contracted, surrogate

        fixed_at = sorted(self.nondiff_argnums)

        @functools.wraps(self.forward)
        def forward(*values):
            fixed = tuple(values[index] for index in fixed_at)
            case = self._case(fixed)
            dynamic = tuple(value for index, value in enumerate(values) if index not in fixed_at)
            if (case is not None and (flat := _case_leaves(case, dynamic)) is not None
                    and (form := _generated.found(_generated.table(emitted, case.name), flat, policy,
                                                  target)) is not None):
                return case.restore(form(*flat))
            return self.forward(*values)

        def tangent(*values):
            *fixed, primals, directions, value = values
            case = self._case(tuple(fixed))
            leaves = jax.tree.leaves(directions, is_leaf=lambda leaf: isinstance(leaf, SymbolicZero))
            inactive = tuple(i for i, leaf in enumerate(leaves) if isinstance(leaf, SymbolicZero))
            if case is None or jax.tree.structure(value) != case.output_tree[0]:
                return self.tangent(*values)
            component = case.tangents.get(inactive)
            if component is not None and (flat := _case_leaves(case, primals)) is not None:
                active = tuple(leaf for leaf in leaves if not isinstance(leaf, SymbolicZero))
                operands = (*flat, *active, *jax.tree.leaves(value))
                name = case.tangent_name(inactive)
                if (form := _generated.found(_generated.table(emitted, name), operands, policy,
                                             target)) is not None:
                    return case.restore(form(*operands))
            return self.tangent(*values)

        return contracted("test_generated_rounded_bodies_keep_the_written_surrogate_law")(
            surrogate(forward, tangent, nondiff_argnums=self.nondiff_argnums,
                      symbolic_zeros=self.symbolic_zeros))


_BITS = (float, complex, np.float16, np.float32, np.float64, np.complex64, np.complex128,
         np.dtype(jnp.bfloat16).type)
"""The number types `_exact_key` reads by their bytes: these exactly (by identity, never by a type's hash), a
subclass keeping its own equality."""
_PADDED = (np.longdouble, np.clongdouble)
"""The number types whose bytes hold padding no value sets (x87's 80 bits in 16): read by each part's sign
and its shortest exact digits instead, a NaN by its sign alone."""


def _exact_key(value) -> Hashable | None:
    """`value` as a surrogate's static arguments key it, two values one key only where they are one: a float
    of `_BITS` by its type and bytes (`-0.0` no `0.0`, each NaN its own sign and payload), one of `_PADDED` by
    its parts' signs and digits, a plain tuple by its parts, any other hashable value by its type and itself
    (its own equality: a callable by identity, a subclass by its own). None where it is unhashable, which no
    declared case is."""
    kind = type(value)
    if any(kind is number for number in _BITS):
        return kind, np.asarray(value).tobytes()
    if any(kind is number for number in _PADDED):
        parts = (value.real, value.imag) if kind is np.clongdouble else (value,)
        return kind, tuple((bool(np.signbit(part)), repr(part)) for part in parts)
    if kind is tuple:
        parts = tuple(_exact_key(part) for part in value)
        return None if None in parts else (kind, parts)
    try:
        hash(value)
    except TypeError:
        _log.debug("a surrogate's unhashable static argument %s keys no case: its written body runs", kind)
        return None
    return kind, value


def _case_leaves(case: SurrogateCase, dynamic: tuple) -> tuple | None:
    if case.input_tree is None:
        return dynamic  # The array admission below rejects a nested value in a flat declaration.
    leaves, tree = jax.tree.flatten(dynamic)
    return tuple(leaves) if tree == case.input_tree[0] else None


@dataclasses.dataclass(frozen=True)
class Generated:
    """A generated module: the declarations it holds, its path, the suite that holds it (`TESTS`), the rule
    sets and their receipts it was saturated with, and the egglog version."""

    declarations: tuple[str, ...]
    module: str
    tests: str
    rules: tuple[str, ...]
    egglog: str


DECLARATIONS: dict[str, StableForm | ComponentForm] = {}
"""Generated functions by their declaring module and name (`declare`)."""


def declare[Form: (StableForm, ComponentForm)](module: str, name: str, form: Form) -> Form:
    """`form` registered under its declaring `module` and `name`, generated into `<module>_derived.py` as a
    function of that name; a name already declared another form refused, never overwritten."""
    qualified = f"{module}.{name}"
    if (held := DECLARATIONS.get(qualified)) is not None and held is not form:
        raise ValueError(f"{qualified} is declared twice: each form, and each surrogate case's static "
                         "arguments and tangent activity, needs a name of its own")
    DECLARATIONS[qualified] = form
    return form


def admits(function: Callable, operands: tuple[jax.ShapeDtypeStruct, ...], cases: Cases,
           label: str) -> Callable[[Callable, Callable], bool]:
    """The gate a candidate is admitted by (`error.choose`'s `admits`): whether it is as exact as a
    reference from `function`'s float64 truth over the cases' draws of `operands` (check 1,
    `dew.parity.judge`: a site over or left unresolved admits none), each compiled through Dew's compile
    point with the pass off (`options.owned_program`, `algebra.passes`)."""
    from dew import algebra
    from dew.nn.kernels import options

    truth = options.owned_program(function)

    def admitted(candidate: Callable, reference: Callable) -> bool:
        ours, theirs = options.owned_program(candidate), options.owned_program(reference)

        # Each form computed as it is: the pass off, as the gate holds it for a reference and the truth.
        def draw(seed: int):
            values = cases.draw(seed, operands)
            with algebra.passes(on=False):
                mine, other = ours(*values), theirs(*values)
                with jax.enable_x64(new_val=True):
                    true = truth(*map(_widened, values))
            return mine, other, true

        try:
            parity.judge(draw, cases.seeds, label)
        except parity.GateFailure:
            return False
        return True
    return admitted


def _widened(value) -> jax.Array:
    """A truth's operand: a real float's in float64, a complex one's in complex128, any other as it is."""
    value = jnp.asarray(value)
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        return value.astype(jnp.complex128)
    return value.astype(jnp.float64) if jnp.issubdtype(value.dtype, jnp.floating) else value


@dataclasses.dataclass(frozen=True)
class _Chosen:
    """One checked source body and its declared dispatch key and return convention."""

    name: str
    policy: str
    target: str
    case: int
    operands: tuple[jax.ShapeDtypeStruct, ...]
    body: str | printed.Body = dataclasses.field(hash=False, compare=False)
    single: bool = dataclasses.field(hash=False, compare=False)
    """Whether the declaration returns one array (its form's one output), else a tuple of them."""
    written: jcore.Jaxpr | None = dataclasses.field(default=None, hash=False, compare=False)


def _chosen(qualified: str, form: StableForm | ComponentForm) -> list[_Chosen]:
    """Stable forms are selected by error and price; components preserve the traced arithmetic."""
    from dew import algebra

    name = qualified.rsplit(".", 1)[1]
    if isinstance(form, ComponentForm):
        _check_cases(form.cases)
    found = []
    for target in form.cases.targets:
        for case, operands in enumerate(form.cases.operands):
            for policy in form.cases.policies:
                label = f"{qualified} ({policy}, {target}, {case})"
                written = None
                if isinstance(form, ComponentForm):
                    single, text, written = _component(form, operands, policy, target, label)
                else:
                    gate = admits(form.function, operands, form.cases, label)
                    with algebra.passes(target=target):
                        choice = error.choose(form.function, operands, accurate=policy == "accurate",
                                              saturation=algebra.SATURATION, admits=gate)
                    candidate = choice.chosen
                    single = _single(form.function, operands)
                    closed = jax.make_jaxpr(candidate.form or candidate.function)(*operands)
                    text, _ = printed.source(closed, "form", compensated=candidate.compensated)
                if found and found[0].single != single:
                    raise ValueError(f"{label}: a declaration must keep its array or tuple result convention")
                found.append(_Chosen(name, policy, target, case, operands, text, single, written))
    return found


def _check_cases(cases: Cases) -> None:
    if any(arg.sharding is not None or arg.weak_type for operands in cases.operands for arg in operands):
        raise printed.Unprintable("a flat component case must be unplaced and strongly typed")
    for variants in (tuple(map(case_of, cases.operands)), cases.policies, cases.targets):
        if len(set(variants)) != len(variants):
            raise ValueError("a component has duplicate dispatch keys")


def _component(form: ComponentForm, operands, policy: str, target: str, label: str
               ) -> tuple[bool, printed.Body, jcore.Jaxpr]:
    from dew import algebra

    with (algebra.passes(target=target),
          generation.run_precision(Numerics(forms="accurate" if policy == "accurate" else "fast"))):
        closed, shaped = jax.make_jaxpr(form.function, return_shape=True)(*operands)
    if closed.effects:
        raise printed.Unprintable(f"{label}: a component has effects")
    for atom in (*closed.jaxpr.invars, *closed.jaxpr.outvars):
        if not isinstance(aval := atom.aval, ShapedArray) or not aval.sharding.mesh.empty:
            raise printed.Unprintable(f"{label}: a flat component cannot preserve placement")
    single = _single_result(shaped)
    body = closed.jaxpr
    live = programs.live(body.eqns, body.outvars)
    live_body = closed.replace(jaxpr=programs.with_equations(body, live))
    flattened, _ = printed.formed(live_body, borrow_shapes=True)
    return single, flattened, closed


def _admit_components(module: str, chosen: Sequence[_Chosen], text: str) -> None:
    if not any(found.written is not None for found in chosen):
        return
    emitted = printed.loaded_module(text, module)
    for found in chosen:
        if found.written is None:
            continue
        qualified = f"{module}.{found.name}"
        form = DECLARATIONS[qualified]
        assert isinstance(form, ComponentForm)
        label = f"{qualified} ({found.policy}, {found.target}, {found.case})"
        entry = emitted[found.name]
        assert callable(entry)
        candidate = functools.partial(entry, policy=found.policy, target=found.target)
        reference = jcore.jaxpr_as_fun(found.written)
        single = found.single

        def written(*values, reference=reference, single=single):
            output = reference(*values)
            return output[0] if single else tuple(output)

        def flat(*values, candidate=candidate, single=single):
            value = candidate(*values)
            return (value,) if single else value

        printed.held(flat, found.written)
        if not admits(form.truth, found.operands, form.cases, label)(candidate, written):
            raise ValueError(f"{label}: the emitted entrypoint did not pass Check 1")


def _bodies(chosen: Sequence[_Chosen]):
    active: dict[_Chosen, str] = {}
    by_body: dict[tuple[str, str | printed.Body], str] = {}
    bodies: dict[str, str | printed.Body] = {}
    for found in chosen:
        key = found.name, found.body
        if key not in by_body:
            function = f"_{found.name}_{sum(name == found.name for name, _ in by_body)}"
            by_body[key] = function
            bodies[function] = found.body
        active[found] = by_body[key]
    return bodies, active


def _shared() -> str:
    """The generated module of the component helpers every generated module shares (`_component_sharing`),
    `shared_derived` in the first declaring module's package (`DECLARING`, `dew.algebra.shared_derived`): one
    helper wherever a span recurs, in one declaring module or across several, so a module splits along its
    own seams without writing a helper twice."""
    return f"{DECLARING[0].rsplit('.', 1)[0]}.shared_derived"


@dataclasses.dataclass(frozen=True)
class _Emitted:
    """A declaring module's generated bodies (`_bodies`) once shared: each body it keeps, and each case a
    shared helper's call alone answers, bound in its table (`_component_sharing.bound_call`)."""

    bodies: dict[str, str | printed.Body]
    named: dict[_Chosen, str]
    bound: dict[_Chosen, printed.Call]


def _emitted(by_module: Mapping[str, Sequence[_Chosen]]
             ) -> tuple[dict[str, _Emitted], dict[str, printed.Body]]:
    """Each module's bodies with the spans its components share with any other's factored into helpers
    (`_component_sharing.shared`, over every sharing module at once), and those helpers. A module holding a
    stable form's compensated text (a body of no bindings to share) keeps its bodies as they are."""
    per = {module: _bodies(chosen) for module, chosen in by_module.items()}
    names = [name for bodies, _ in per.values() for name in bodies]
    if len(set(names)) != len(names):
        raise ValueError("two declaring modules emit a body of one name; name their declarations apart")
    sharing = [module for module, (bodies, _) in per.items()
               if not any(isinstance(body, str) for body in bodies.values())]
    components = {name: body for module in sharing for name, body in per[module][0].items()
                  if isinstance(body, printed.Body)}
    declared = sorted({found.name for chosen in by_module.values() for found in chosen})
    shared = _component_sharing.shared(components, reserved=declared) if components else {}
    factored = {name: body for name, body in shared.items() if name not in components}
    made = {}
    for module, (bodies, named) in per.items():
        if module not in sharing:
            made[module] = _Emitted(bodies, named, {})
            continue
        bound = {}
        for found, name in named.items():
            call = _component_sharing.bound_call(shared[name], factored,
                                                 tuple(arg.shape for arg in found.operands))
            if call is not None:
                bound[found] = call
        needed = {name for found, name in named.items() if found not in bound}
        made[module] = _Emitted({name: shared[name] for name in bodies if name in needed}, named, bound)
    return made, factored


def _header(first: str, names: Sequence[str]) -> list[str]:
    sets = rules.admitted(accurate=False, dev=True)
    lines = [f'"""Generated by `python -m dew.algebra.generate` {first}; do not edit.', ""]
    if names:
        lines.append(f"Declarations: {', '.join(names)}.")
    lines += [f"Rule sets: {', '.join(found.name for found in sets)}.",
              f"egglog {importlib.metadata.version('egglog')}; measured on {jax.default_backend()}.",
              "", "Receipts:"]
    lines += [f"- {found.name}: {found.receipt}" for found in sets]
    return [*_flowed(lines), '"""', ""]


def _rendered(sources: Mapping[str, str | printed.Body]) -> list[str]:
    body: list[str] = []
    for name, source in sources.items():
        text = source.render(name) if isinstance(source, printed.Body) else source.replace(
            "def form(", f"def {name}(", 1)
        body += ["", "", *text.splitlines()]
    return body


def _helpers_written(factored: Mapping[str, printed.Body]) -> str:
    """The shared helpers' module: its header and each helper printed."""
    body = _rendered(factored)
    return "\n".join([*_header("for every generated module's shared spans", ()),
                      *printed.imports("\n".join(body)), *body]) + "\n"


def _helper_reads(sources: Iterable[str | printed.Body | printed.Call],
                  factored: Mapping[str, printed.Body]) -> list[str]:
    """The shared helpers `sources` (bodies and bound table calls) call, read off their typed calls, in the
    helpers' module's order. A body of printed text (a compensated form's) shares nothing."""
    found: set[str] = set()

    def visit(value) -> None:
        if isinstance(value, printed.Call):
            if isinstance(value.callee, str) and value.callee in factored:
                found.add(value.callee)
            for part in (*value.operands, *(operand for _, operand in value.keywords)):
                visit(part)
        elif isinstance(value, printed.StaticTuple):
            for part in value.values:
                visit(part)

    for source in sources:
        if isinstance(source, printed.Body):
            for statement in source.statements:
                if isinstance(statement, printed.Binding):
                    visit(statement.value)
            for output in source.outputs:
                visit(output)
        elif isinstance(source, printed.Call):
            visit(source)
    return [name for name in factored if name in found]


def _written(module: str, chosen: Sequence[_Chosen], names: Sequence[str], emitted: _Emitted,
             factored: Mapping[str, printed.Body]) -> tuple[str, str]:
    """The generated module's text: its header, each chosen form printed (`printed.source`), the shared
    helpers it calls imported (`_shared`), and each declaration's function, which picks its form by policy,
    target and its operands' shapes and dtypes; and the same module with those helpers written in place of
    their import, which admission loads before the helpers' module is written (`_admit_components`)."""
    body = _rendered(emitted.bodies)
    declared = sorted({found.name for found in chosen})
    for name in declared:
        own = [found for found in chosen if found.name == name]
        single = own[0].single
        policy, target = own[0].policy, own[0].target
        table = _generated.table_name(name)
        body += ["", "", f"{table}: dict[tuple, Callable[..., tuple]] = {{"]
        for found in own:
            operands = _runs(repr(operand) for operand in case_of(found.operands))
            body += printed.wrapped(f"    ({found.policy!r}, {found.target!r}, (", operands,
                                    ",)):" if len(operands) == 1 else ")):")
            binding = emitted.bound.get(found)
            if binding is None:
                body.append(f"        {emitted.named[found]},")
            elif not binding.keywords:
                body.append(f"        {binding.callee},")
            else:
                args = [str(binding.callee), *(f"{key}={printed.expression(value)}"
                                               for key, value in binding.keywords)]
                body += printed.wrapped("        functools.partial(", args, "),")
        body += ["}", "", "", f"def {name}(*operands, policy: str = {policy!r}, target: str = {target!r}):",
                 f'    """{name}\'s generated form for `policy` and `target` at its operands\' case."""',
                 f"    return _generated.call({table}, operands, policy, target, single={single})"]
    header = _header(f"from {module}", names)
    reads = _helper_reads([*emitted.bodies.values(), *emitted.bound.values()], factored)
    imports = printed.imports("\n".join(body))
    if reads:
        one = f"from {_shared()} import {', '.join(reads)}"
        lines = [one] if len(one) <= printed.WIDTH else [
            f"from {_shared()} import (", *(f"    {name}," for name in reads), ")"]
        # Within Dew's section, in module order: after `dew.algebra`'s own, before `dew.nn`'s.
        at = next((index for index, line in enumerate(imports) if line.startswith("from dew.nn")),
                  len(imports))
        imports[at:at] = lines
    text = "\n".join([*header, *imports, *body]) + "\n"
    inlined = _rendered({name: factored[name] for name in _closure(reads, factored)})
    admitted = "\n".join([*header, *printed.imports("\n".join([*inlined, *body])), *inlined, *body]) + "\n"
    return text, admitted


def _closure(names: Sequence[str], factored: Mapping[str, printed.Body]) -> list[str]:
    """`names` and every helper they call in turn, in the helpers' module order."""
    held, pending = set(), list(names)
    while pending:
        if (name := pending.pop()) in held:
            continue
        held.add(name)
        pending += _helper_reads([factored[name]], factored)
    return [name for name in factored if name in held]


def _runs(keys: Iterable[str]) -> list[str]:
    """A case key's operand entries, each run of one repeated entry starred once (`*(key,) * count`)."""
    return [key if count == 1 else f"*({key},) * {count}"
            for key, count in ((key, len(list(run))) for key, run in itertools.groupby(keys))]


def _single(function: Callable, operands: tuple[jax.ShapeDtypeStruct, ...]) -> bool:
    """Whether `function` returns one array over `operands`, else a tuple of them; another structure is
    refused, a generated form returning its outputs as a tuple."""
    return _single_result(jax.eval_shape(function, *operands))


def _single_result(shaped) -> bool:
    if isinstance(shaped, jax.ShapeDtypeStruct):
        return True
    if isinstance(shaped, tuple) and all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in shaped):
        return False
    raise ValueError(f"a declaration returns an array or a tuple of arrays, not {jax.tree.structure(shaped)}")


def _flowed(lines: list[str]) -> list[str]:
    """`lines` with the header's long ones wrapped at a space within `printed.WIDTH`."""
    flowed = []
    for line in lines:
        while len(line) > printed.WIDTH and " " in line[:printed.WIDTH]:
            cut = line.rindex(" ", 0, printed.WIDTH)
            flowed.append(line[:cut])
            line = "  " + line[cut + 1:]
        flowed.append(line)
    return flowed


def _path(module: str) -> Path:
    """The generated module's path: `<module>_derived.py` beside the declaring module."""
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise ValueError(f"no declaring module {module!r}")
    origin = Path(spec.origin)
    return origin.with_name(f"{origin.stem}_derived.py")


def generate(*, check: bool = False) -> tuple[Generated, ...]:
    """Every declaring module's declarations generated into its module, all together (their components share
    one helper module, `_shared`), or, with `check`, regenerated in memory and compared: raises where a
    written module is not what generation writes."""
    for module in DECLARING:
        importlib.import_module(module)
    if not DECLARATIONS:
        raise ValueError(f"no declarations: none of {DECLARING} declared a stable form here (a registry "
                         f"other than the declaring modules' own reads none)")
    # The declaring modules' own declarations alone: one another module's import made writes nothing.
    by_module: dict[str, list[str]] = {}
    for qualified in sorted(DECLARATIONS):
        if (module := qualified.rsplit(".", 1)[0]) in DECLARING:
            by_module.setdefault(module, []).append(qualified)
    sets = tuple(found.name for found in rules.admitted(accurate=False, dev=True))
    found = {module: [one for qualified in declared for one in _chosen(qualified, DECLARATIONS[qualified])]
             for module, declared in by_module.items()}
    emitted, factored = _emitted(found)
    texts = {module: _written(module, found[module], by_module[module], emitted[module], factored)
             for module in by_module}
    for module, (_, admitted) in texts.items():
        _admit_components(module, found[module], admitted)
    made, stale = [], []
    writes = [(_path(module), text, tuple(by_module[module])) for module, (text, _) in texts.items()]
    package = _path(DECLARING[0]).parent
    writes.append((package / f"{_shared().rsplit('.', 1)[1]}.py", _helpers_written(factored), ()))
    for path, text, declared in writes:
        if check:
            written = path.read_text() if path.exists() else ""
            if written != text:
                stale.append("\n".join([str(path), *itertools.islice(difflib.unified_diff(
                    written.splitlines(), text.splitlines(), "written", "generated", lineterm=""), 40)]))
        else:
            path.write_text(text)
        made.append(Generated(declared, str(path), TESTS, sets, importlib.metadata.version("egglog")))
    if stale:
        raise ValueError("generated modules differ from what generation writes; run `python -m "
                         "dew.algebra.generate`:\n" + "\n".join(stale))
    return tuple(made)


def main(argv: Sequence[str] | None = None) -> int:
    """The generator's command: generate every declaration, or `--check`."""
    parser = argparse.ArgumentParser(prog="python -m dew.algebra.generate", description=__doc__)
    parser.add_argument("--check", action="store_true", help="regenerate in memory and compare")
    arguments = parser.parse_args(argv)
    for found in generate(check=arguments.check):
        sys.stderr.write(f"{'checked' if arguments.check else 'wrote'} {found.module}\n")
    return 0


if __name__ == "__main__":
    # Run as a script this file is `__main__`, a module of its own beside the one the declaring modules
    # register with: the command is that module's.
    from dew.algebra import generate as declared

    sys.exit(declared.main())
