"""The generator (docs/design/algebra.md §6.11): each declared stable form (`StableForm`) traced at each of
its cases, saturated with every rule set dev time runs (`rules.admitted(dev=True)`), its candidates measured
against its float64 truth on its cases' draws (check 1, `dew.parity.judge`) and the cheapest its policy
admits chosen (`dew.algebra.error.choose`, §1.5) under each policy, priced for each target; the choices
written beside the declaring module as ordinary Python (`<module>_derived.py`, `printed.source`). The
generated module uses only `dew.algebra._generated` for case dispatch, not the registry, rules or
extractor. Its header names the declarations, rule sets, receipts and egglog version.
`python -m dew.algebra.generate [names] [--check]`; `--check` regenerates in memory and fails
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
import sys
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend import core as jcore

from dew import parity
from dew.algebra import _component_sharing, error, printed, rules, terms
from dew.algebra._generated import case_of
from dew.nn import programs
from dew.nn.kernels import generation
from dew.nn.precision import Numerics


POLICIES = ("default", "accurate")
"""The precision policies a declaration is generated for (§1.5): under default numerics a candidate is
admitted against the form as written, under the accurate numerics against every other candidate."""

TESTS = "tests/test_algebra_generate.py"
"""The suite every generated module is held by: its forms on par with their declarations (check 1) and the
written module what generation writes (`--check`)."""

DECLARING = ("dew.algebra.acceptance", "dew.algebra.component_cases", "dew.sampling.guidance_cases")
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
    cases: Mapping[tuple, SurrogateCase]

    def declared(self, module: str) -> SurrogateBodies:
        """Register both bodies under the same static-argument and tangent-activity declaration."""
        for case in self.cases.values():
            declare(module, case.name, case.forward)
            for inactive, form in case.tangents.items():
                declare(module, case.tangent_name(inactive), form)
        return self

    @functools.cached_property
    def _keyed(self) -> dict[Hashable, SurrogateCase]:
        """Each case by its static arguments' structural key (`terms.parameter_key`): a float by its bits,
        so `-0.0` and `0.0`, equal as Python values, are two keys."""
        return {terms.parameter_key(fixed): case for fixed, case in self.cases.items()}

    def _case(self, fixed: tuple) -> SurrogateCase | None:
        # A static argument no structural key states (unhashable, or held by identity) matches no declared
        # case, whose key is by value: its written body runs.
        return self._keyed.get(terms.parameter_key(fixed))

    def assembled(self, emitted: Callable[[str], Callable], *, policy: str = "default", target: str = "sm90"):
        """Opt into emitted cases using the original surrogate rule's argument and value dependency."""
        from jax.custom_derivatives import SymbolicZero

        from dew.nn.derivatives import contracted, surrogate

        fixed_at = sorted(self.nondiff_argnums)

        @functools.wraps(self.forward)
        def forward(*values):
            fixed = tuple(values[index] for index in fixed_at)
            case = self._case(fixed)
            dynamic = tuple(value for index, value in enumerate(values) if index not in fixed_at)
            if (case is not None and (flat := _case_leaves(case, dynamic)) is not None
                    and _case_admitted(case.forward.cases, flat, policy, target)):
                return case.restore(emitted(case.name)(*flat, policy=policy, target=target))
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
                if _case_admitted(component.cases, operands, policy, target):
                    run = emitted(case.tangent_name(inactive))
                    return case.restore(run(*operands, policy=policy, target=target))
            return self.tangent(*values)

        return contracted("test_generated_rounded_bodies_keep_the_written_surrogate_law")(
            surrogate(forward, tangent, nondiff_argnums=self.nondiff_argnums,
                      symbolic_zeros=self.symbolic_zeros))


def _case_leaves(case: SurrogateCase, dynamic: tuple) -> tuple | None:
    if case.input_tree is None:
        return dynamic  # The array admission below rejects a nested value in a flat declaration.
    leaves, tree = jax.tree.flatten(dynamic)
    return tuple(leaves) if tree == case.input_tree[0] else None


def _case_admitted(cases: Cases, operands: tuple, policy: str, target: str) -> bool:
    if policy not in cases.policies or target not in cases.targets:
        return False
    for value in operands:
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            return False
        aval = jax.typeof(value)
        if not isinstance(aval, ShapedArray) or aval.weak_type or not aval.sharding.mesh.empty:
            return False
    key = case_of(operands)
    return any(case_of(shapes) == key for shapes in cases.operands)


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
    function of that name."""
    DECLARATIONS[f"{module}.{name}"] = form
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


def _shared_cases(bodies: dict[str, str | printed.Body], active: dict[_Chosen, str], declared: list[str]):
    # Stable forms containing compensated bindings do not participate in component sharing.
    if any(isinstance(body, str) for body in bodies.values()):
        return bodies, {}
    components = {name: body for name, body in bodies.items() if isinstance(body, printed.Body)}
    emitted = _component_sharing.shared(components, reserved=declared)
    shared_names = emitted.keys() - components.keys()
    shared_bodies = {name: emitted[name] for name in shared_names}
    bound = {}
    for found, name in active.items():
        args = _component_sharing.bound_call(emitted[name], shared_bodies,
                                             tuple(arg.shape for arg in found.operands))
        if args is not None:
            bound[found] = args
    needed = {name for found, name in active.items() if found not in bound}
    retained: dict[str, str | printed.Body] = {
        name: body for name, body in emitted.items() if name in shared_names or name in needed}
    return retained, bound


def _written(module: str, chosen: Sequence[_Chosen], names: Sequence[str]) -> str:
    """The generated module's text: its header, each chosen form printed (`printed.source`), and each
    declaration's function, which picks its form by policy, target and its operands' shapes and dtypes."""
    sets = rules.admitted(accurate=False, dev=True)
    lines = [f'"""Generated by `python -m dew.algebra.generate` from {module}; do not edit.', "",
             f"Declarations: {', '.join(names)}.",
             f"Rule sets: {', '.join(found.name for found in sets)}.",
             f"egglog {importlib.metadata.version('egglog')}; measured on {jax.default_backend()}.",
             "", "Receipts:"]
    lines += [f"- {found.name}: {found.receipt}" for found in sets]
    lines = [*_flowed(lines), '"""', ""]
    body: list[str] = []
    bodies, named = _bodies(chosen)
    declared = sorted({found.name for found in chosen})
    emitted, bound = _shared_cases(bodies, named, declared)
    for name, source in emitted.items():
        text = source.render(name) if isinstance(source, printed.Body) else source.replace(
            "def form(", f"def {name}(", 1)
        body += ["", "", *text.splitlines()]
    for name in declared:
        own = [found for found in chosen if found.name == name]
        single = own[0].single
        policy, target = own[0].policy, own[0].target
        table = f"_{name}_forms"
        body += ["", "", f"{table}: dict[tuple, Callable[..., tuple]] = {{"]
        for found in own:
            operands = [repr(operand) for operand in case_of(found.operands)]
            body += printed.wrapped(f"    ({found.policy!r}, {found.target!r}, (", operands,
                                    ",)):" if len(operands) == 1 else ")):")
            binding = bound.get(found)
            if binding is None:
                body.append(f"        {named[found]},")
            elif not binding.keywords:
                body.append(f"        {binding.callee},")
            else:
                args = [str(binding.callee), *(f"{key}={printed.expression(value)}"
                                               for key, value in binding.keywords)]
                body += printed.wrapped("        functools.partial(", args, "),")
        body += ["}", "", "", f"def {name}(*operands, policy: str = {policy!r}, target: str = {target!r}):",
                 f'    """{name}\'s generated form for `policy` and `target` at its operands\' case."""',
                 f"    return _generated.call({table}, operands, policy, target, single={single})"]
    return "\n".join([*lines, *printed.imports("\n".join(body)), *body]) + "\n"


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


def generate(names: Sequence[str] | None = None, *, check: bool = False) -> tuple[Generated, ...]:
    """Each declaring module's declarations (`names`, every one where None) generated into its module, or,
    with `check`, regenerated in memory and compared: raises where a written module is not what generation
    writes."""
    for module in DECLARING:
        importlib.import_module(module)
    if not DECLARATIONS:
        raise ValueError(f"no declarations: none of {DECLARING} declared a stable form here (a registry "
                         f"other than the declaring modules' own reads none)")
    chosen = sorted(names or DECLARATIONS)
    unknown = [name for name in chosen if name not in DECLARATIONS]
    if unknown:
        raise ValueError(f"no declaration {unknown}; declared: {sorted(DECLARATIONS)}")
    by_module: dict[str, list[str]] = {}
    for qualified in chosen:
        by_module.setdefault(qualified.rsplit(".", 1)[0], []).append(qualified)
    made, stale = [], []
    sets = tuple(found.name for found in rules.admitted(accurate=False, dev=True))
    for module, declared in by_module.items():
        if names is not None and set(declared) != {name for name in DECLARATIONS
                                                  if name.rsplit(".", 1)[0] == module}:
            raise ValueError(f"{module}'s module holds every declaration of it: generate them together")
        found = [one for qualified in declared for one in _chosen(qualified, DECLARATIONS[qualified])]
        text, path = _written(module, found, declared), _path(module)
        _admit_components(module, found, text)
        if check:
            written = path.read_text() if path.exists() else ""
            if written != text:
                stale.append("\n".join([str(path), *itertools.islice(difflib.unified_diff(
                    written.splitlines(), text.splitlines(), "written", "generated", lineterm=""), 40)]))
        else:
            path.write_text(text)
        made.append(Generated(tuple(declared), str(path), TESTS, sets, importlib.metadata.version("egglog")))
    if stale:
        raise ValueError("generated modules differ from what generation writes; run `python -m "
                         "dew.algebra.generate`:\n" + "\n".join(stale))
    return tuple(made)


def main(argv: Sequence[str] | None = None) -> int:
    """The generator's command: generate the named declarations (every one where none), or `--check`."""
    parser = argparse.ArgumentParser(prog="python -m dew.algebra.generate", description=__doc__)
    parser.add_argument("names", nargs="*", help="declarations to generate, by module and name")
    parser.add_argument("--check", action="store_true", help="regenerate in memory and compare")
    arguments = parser.parse_args(argv)
    for found in generate(arguments.names or None, check=arguments.check):
        sys.stderr.write(f"{'checked' if arguments.check else 'wrote'} {found.module}\n")
    return 0


if __name__ == "__main__":
    # Run as a script this file is `__main__`, a module of its own beside the one the declaring modules
    # register with: the command is that module's.
    from dew.algebra import generate as declared

    sys.exit(declared.main())
