#!/usr/bin/env python3
"""The one-time conversion of the gate's legacy families to check 1 (`assert_as_exact`), run once over
main lane by lane and then deleted (Main, 2026-10-21).

    python tools/check1_codemod.py survey [paths]    # each site's family and class, counted per lane
    python tools/check1_codemod.py convert [paths]   # rewrite the convertible sites in place

A site is a call of a legacy family (one-draw `assert_as_exact_as_the_reference`, the retiring
`assert_leaves_as_exact`, the over-orders and element rules) or a hand tolerance SLOP020 counts
(tools/lint_tolerances.py). Each is classed by where its arrays come from, followed through the
names its test function binds:
- seeded: every array is computed in the test from inputs one literal seed draws (a NumPy generator,
  a JAX key), and from nothing recorded: check 1 can draw it again at any seed (`convert`);
- recorded: an array is read from a recording (a fixture file, a module-level table of one): the
  reference must be recorded over seeds first (`record_case`), by its lane;
- other: neither (inputs a parametrization or a fixture builds, a constant): its lane converts it.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "tools")]

FAMILIES = {"assert_as_exact_as_the_reference": "one-draw", "assert_leaves_as_exact": "retiring one-draw",
            "assert_as_exact_over_orders": "over-orders", "assert_rows_as_exact_over_orders": "over-orders",
            "assert_elements_as_exact": "elements"}
SEEDERS = {"default_rng", "RandomState", "PRNGKey", "key", "seed", "manual_seed", "Generator"}
LOADERS = {"load", "load_file", "loads", "read_bytes", "read_text", "fetch", "from_pretrained", "indexed_recording",
           "hf_hub_download", "recording"}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


class _Scope:
    """A test function's bindings, name by name, and the module's, to follow an array to its sources."""

    def __init__(self, module: ast.Module, function: ast.AST):
        self.bound: dict[str, list[ast.AST]] = collections.defaultdict(list)
        self.params = set()
        for owner in (module, function):
            for node in ast.walk(owner) if owner is function else owner.body:
                targets = []
                if isinstance(node, ast.Assign):
                    targets, value = node.targets, node.value
                elif isinstance(node, ast.AnnAssign | ast.AugAssign) and node.value is not None:
                    targets, value = [node.target], node.value
                elif isinstance(node, ast.For):
                    targets, value = [node.target], node.iter
                elif isinstance(node, ast.With):
                    for item in node.items:
                        if item.optional_vars is not None:
                            for name in ast.walk(item.optional_vars):
                                if isinstance(name, ast.Name):
                                    self.bound[name.id].append(item.context_expr)
                    continue
                else:
                    continue
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            self.bound[name.id].append(value)
        if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            self.params = {arg.arg for arg in [*function.args.args, *function.args.kwonlyargs]}

    def sources(self, expression: ast.AST) -> set[str]:
        """What `expression` reads, followed through bindings: "seed" for a literal-seeded generator or
        key, "recorded" for a loaded recording, "param" for the test's arguments (a parametrization or
        a fixture), "constant" for literals only."""
        found: set[str] = set()
        seen: set[str] = set()
        work = [expression]
        while work:
            node = work.pop()
            for part in ast.walk(node):
                if isinstance(part, ast.Call):
                    name = _call_name(part)
                    if name in SEEDERS and part.args and isinstance(part.args[0], ast.Constant):
                        found.add("seed")
                    if name in LOADERS:
                        found.add("recorded")
                    if any(isinstance(arg, ast.Constant) and type(arg.value) is int for arg in part.args) or any(
                            keyword.arg in {"seed", "key", "rng"} for keyword in part.keywords):
                        found.add("literal call")
                if isinstance(part, ast.Name) and part.id not in seen:
                    seen.add(part.id)
                    if part.id in self.params:
                        found.add("param")
                    # A module table of recordings (FLAXDIFF = np.load(...)) is followed as any binding.
                    work.extend(self.bound.get(part.id, []))
        return found


def _class(sources: set[str]) -> str:
    if "recorded" in sources:
        return "recorded"
    if "seed" in sources and "param" not in sources:
        return "seeded"
    if "param" in sources:
        return "other: parametrized or a fixture"
    if "literal call" in sources:
        return "other: a helper given a literal"
    return "other: constants"


def sites(path: Path):
    """Each legacy-family call in `path`, with its family, class and line."""
    module = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in ast.walk(module) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)]
    for function in functions:
        scope = None
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and _call_name(node) in FAMILIES and isinstance(
                    node.func, ast.Name | ast.Attribute):
                # A nested function's call belongs to its own scope.
                if any(node in ast.walk(inner) for inner in functions
                       if inner is not function and inner in ast.walk(function)):
                    continue
                scope = scope or _Scope(module, function)
                found = set().union(*(scope.sources(arg) for arg in node.args[:3])) if node.args else set()
                yield {"family": FAMILIES[_call_name(node)], "class": _class(found), "line": node.lineno,
                       "function": function.name}


_INVENTORY: dict = {}


def _lane(relative: str) -> str:
    """A file's lane, by the flip head's inventory, which counts the one-draw calls."""
    if not _INVENTORY:
        shown = subprocess.run(["git", "show", "origin/tests/gate-flip:tools/tolerance_inventory.json"],
                               cwd=REPO, capture_output=True, text=True, check=True).stdout
        _INVENTORY.update(json.loads(shown)["files"])
    entry = _INVENTORY.get(relative)
    return entry["lane"] if entry else "unassigned"


def _float64(expression: ast.AST, scope: _Scope) -> bool:
    """Whether an operand is computed in float64 somewhere along its bindings: a float64 truth."""
    seen: set[str] = set()
    work = [expression]
    while work:
        node = work.pop()
        for part in ast.walk(node):
            if (isinstance(part, ast.Attribute) and part.attr in {"float64", "enable_x64", "double"}) or (
                    isinstance(part, ast.Constant) and part.value in {"float64", "f8"}):
                return True
            if isinstance(part, ast.Name) and part.id not in seen:
                seen.add(part.id)
                if part.id in {"widened", "float64", "truth"} or "f64" in part.id or "truth" in part.id:
                    return True
                work.extend(scope.bound.get(part.id, []))
    return False


CONSTANT_NAMES = {"np", "jnp", "math", "numpy", "torch", "pi", "e", "inf", "nan"}


def _closed_form(node: ast.AST) -> str | None:
    """Whether an operand is a constant: "closed" for one written in closed form (integers, simple
    fractions, π, a few named functions of them), "golden" for a float literal carrying more digits than
    any closed form (a number a run printed); None for a computed value."""
    golden = False
    for part in ast.walk(node):
        if isinstance(part, ast.Name) and part.id not in CONSTANT_NAMES:
            return None
        if isinstance(part, ast.Call) and not (isinstance(part.func, ast.Attribute | ast.Name)
                                               and _call_name(part) in {"sqrt", "log", "exp", "float", "float32",
                                                                        "float64", "asarray", "array", "full",
                                                                        "zeros", "ones", "approx", "log2", "cos",
                                                                        "sin", "tanh"}):
            return None
        if isinstance(part, ast.Constant) and isinstance(part.value, float):
            digits = repr(abs(part.value)).replace(".", "").lstrip("0").split("e")[0]
            if len(digits.rstrip("0")) > 4 and (part.value * 1024) % 1:
                golden = True
    return "golden" if golden else "closed"


def _sides(node: ast.AST) -> list[ast.AST]:
    """The two things a tolerance compares: a call's first two arguments, a relation's error's two
    sides (`abs(a - b) < tol`), or an `approx`'s two."""
    if isinstance(node, ast.Call):
        if _call_name(node) == "approx":
            return node.args[:1]
        return node.args[:2]
    if isinstance(node, ast.Compare):
        for side in [node.left, *node.comparators]:
            for part in ast.walk(side):
                if isinstance(part, ast.BinOp) and isinstance(part.op, ast.Sub):
                    return [part.left, part.right]
                if isinstance(part, ast.Call) and _call_name(part) == "approx":
                    other = node.left if side is not node.left else node.comparators[0]
                    return [other, *part.args[:1]]
        return [node.left, *node.comparators][:2]
    return [node]


def _negated(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    """Whether a tolerance asserts two things differ: under a `not`, as `!=` or `>`/`>=` of a difference
    or a distance, or as a call whose name says it (`assert_raises`, `differ`, `moved`)."""
    parent = parents.get(id(node))
    if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
        return True
    if isinstance(node, ast.Compare):
        if any(isinstance(op, ast.NotEq) for op in node.ops):
            return True
        if all(isinstance(op, ast.Gt | ast.GtE) for op in node.ops):
            return any(isinstance(part, ast.BinOp) and isinstance(part.op, ast.Sub) or (
                isinstance(part, ast.Call) and _call_name(part) in {"abs", "norm", "distance", "max"})
                for part in ast.walk(node.left))
    return False


def _independent(sides: list[ast.AST], scope: "_Scope") -> bool:
    """Whether a side comes from a recording or a reference implementation (a fixture, a torch or
    another library's run): the comparison is then a contract, whatever its test's name."""
    return any("recorded" in scope.sources(side) for side in sides) or any(
        isinstance(part, ast.Name) and part.id in {"torch", "reference", "expected", "oracle"}
        for side in sides for part in ast.walk(side))


def _repeated(sides: list[ast.AST], scope: "_Scope") -> bool:
    """Whether the two sides are one call repeated: the same callee on the same arguments, followed
    through one binding each (a copy, a reload's read, a deterministic rerun)."""
    def origin(side):
        if isinstance(side, ast.Name) and len(scope.bound.get(side.id, [])) == 1:
            return scope.bound[side.id][0]
        return side
    left, right = (origin(side) for side in sides)
    return isinstance(left, ast.Call) and isinstance(right, ast.Call) and ast.dump(left) == ast.dump(right)


def _kind(node: ast.AST, function: ast.AST, scope: "_Scope", parents: dict[int, ast.AST]) -> str:
    """A tolerance without a float64 truth, by what it compares (Main's classes, 2026-10-21, and the
    audited rules: polarity first, then provenance, then kind)."""
    if _negated(node, parents):
        return "(e) two things asserted to differ"
    sides = _sides(node)
    if len(sides) == 1 and isinstance(node, ast.Call):
        return "(c) a constant"
    forms = [_closed_form(side) for side in sides]
    if "golden" in forms:
        return "(d) a golden number"
    if "closed" in forms:
        return "(c) a constant"
    if len(sides) == 2 and ast.dump(sides[0]) == ast.dump(sides[1]):
        return "(d) a value against itself"
    if len(sides) == 2 and _independent(sides, scope):
        return "(a) a contract"
    if len(sides) == 2 and _repeated(sides, scope):
        return "(b) the same computation"
    return "(a) a contract"


def tolerance_sites(relative: str):
    """Each hand tolerance SLOP020 counts in a file (tools/lint_tolerances.py), its form and class: on
    a float64 truth or not, and where its operands come from."""
    import lint_slop

    [module] = list(lint_slop.collect([relative], REPO, "dew"))
    functions = [node for node in ast.walk(module.tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)]
    parents = {id(child): parent for parent in ast.walk(module.tree) for child in ast.iter_child_nodes(parent)}
    for node, what in lint_slop._tolerance_comparisons(module):
        if "one-draw rule" in what:
            continue
        owner = min((function for function in functions if node in ast.walk(function)),
                    key=lambda function: len(list(ast.walk(function))), default=None)
        if owner is None:
            yield {"form": "module level", "class": "other: constants"}
            continue
        scope = _Scope(module.tree, owner)
        operands = (node.args[:2] if isinstance(node, ast.Call) else
                    [node.left, *node.comparators] if isinstance(node, ast.Compare) else [node])
        on_truth = any(_float64(operand, scope) for operand in operands)
        found = set().union(*(scope.sources(operand) for operand in operands)) if operands else set()
        if on_truth:
            yield {"form": "on a float64 truth", "class": _class(found), "line": node.lineno}
        else:
            yield {"form": "no float64 truth", "class": _kind(node, owner, scope, parents), "line": node.lineno}


def survey(paths: list[str]) -> dict:
    listed = paths or subprocess.run(["git", "ls-files", "tests/*.py", "tools/*.py"], cwd=REPO,
                                     capture_output=True, text=True, check=True).stdout.split()
    counts = collections.defaultdict(collections.Counter)
    for relative in listed:
        if relative.endswith("reference_error.py") or relative.endswith("check1_codemod.py"):
            continue
        for site in sites(REPO / relative):
            counts[(site["family"], _lane(relative))][site["class"]] += 1
        for site in tolerance_sites(relative):
            counts[(f"hand tolerance, {site['form']}", _lane(relative))][site["class"]] += 1
    return {f"{family} / {lane}": dict(counter) for (family, lane), counter in sorted(counts.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("survey",))
    parser.add_argument("paths", nargs="*")
    options = parser.parse_args()
    result = survey(options.paths)
    totals = collections.defaultdict(collections.Counter)
    for key, counter in result.items():
        totals[key.split(" / ")[0]].update(counter)
    print(json.dumps({"by family": {family: dict(counter) for family, counter in totals.items()},
                      "by family and lane": result}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
