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
            return any((isinstance(part, ast.BinOp) and isinstance(part.op, ast.Sub)) or (
                isinstance(part, ast.Call) and _call_name(part) in {"abs", "norm", "distance", "max"})
                for part in ast.walk(node.left))
    return False


def _independent(sides: list[ast.AST], scope: _Scope) -> bool:
    """Whether a side comes from a recording or a reference implementation (a fixture, a torch or
    another library's run): the comparison is then a contract, whatever its test's name."""
    return any("recorded" in scope.sources(side) for side in sides) or any(
        isinstance(part, ast.Name) and part.id in {"torch", "reference", "expected", "oracle"}
        for side in sides for part in ast.walk(side))


def _repeated(sides: list[ast.AST], scope: _Scope) -> bool:
    """Whether the two sides are one call repeated: the same callee on the same arguments, followed
    through one binding each (a copy, a reload's read, a deterministic rerun)."""
    def origin(side):
        if isinstance(side, ast.Name) and len(scope.bound.get(side.id, [])) == 1:
            return scope.bound[side.id][0]
        return side
    left, right = (origin(side) for side in sides)
    return isinstance(left, ast.Call) and isinstance(right, ast.Call) and ast.dump(left) == ast.dump(right)


def _kind(node: ast.AST, function: ast.AST, scope: _Scope, parents: dict[int, ast.AST]) -> str:
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


def _exact_form(node: ast.AST, source: str) -> str | None:
    """A tolerance's exact form, or None where none is regular: `assert_allclose(a, b, ...)` as
    `assert_array_equal(a, b)`, `allclose`/`isclose` as `array_equal`, `a == approx(b, ...)` as
    `a == b`, `abs(a - b) < tol` and its `max` as an equality."""
    segment = ast.get_source_segment
    if isinstance(node, ast.Call) and len(node.args) >= 2:
        name = _call_name(node)
        left, right = (segment(source, arg) for arg in node.args[:2])
        if name == "assert_allclose":
            return f"np.testing.assert_array_equal({left}, {right})"
        if name in {"allclose", "isclose"}:
            return f"np.array_equal(np.asarray({left}), np.asarray({right}))"
        if name == "isclose" and isinstance(node.func, ast.Attribute) and _call_name(node.func.value) == "math":
            return f"{left} == {right}"
        return None
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        right = node.comparators[0]
        if isinstance(node.ops[0], ast.Eq) and isinstance(right, ast.Call) and _call_name(right) == "approx" \
                and right.args:
            return f"{segment(source, node.left)} == {segment(source, right.args[0])}"
        if isinstance(node.ops[0], ast.Lt | ast.LtE):
            error = node.left
            outer = _call_name(error) if isinstance(error, ast.Call) else ""
            inner = error.args[0] if outer in {"max", "amax"} and isinstance(error, ast.Call) and error.args else error
            if isinstance(inner, ast.Call) and _call_name(inner) == "abs" and inner.args and isinstance(
                    inner.args[0], ast.BinOp) and isinstance(inner.args[0].op, ast.Sub):
                a, b = (segment(source, side) for side in (inner.args[0].left, inner.args[0].right))
                return f"np.array_equal(np.asarray({a}), np.asarray({b}))"
    return None


def _tolerance_named(node: ast.AST) -> bool:
    """Whether a tolerance's allowance is a name (a parametrized or computed bound): the test then
    means its tolerance, and no exact form is offered."""
    if isinstance(node, ast.Call):
        allowances = [*node.args[2:], *(keyword.value for keyword in node.keywords)]
    elif isinstance(node, ast.Compare):
        right = node.comparators[0]
        allowances = ([*right.args[1:], *(keyword.value for keyword in right.keywords)]
                      if isinstance(right, ast.Call) and _call_name(right) == "approx" else [right])
    else:
        return True
    return any(isinstance(part, ast.Name) and part.id not in CONSTANT_NAMES
               for allowance in allowances for part in ast.walk(allowance))


def exact_pass(relative: str, classes: set[str]) -> list[dict]:
    """Rewrite in place each tolerance of `relative` in `classes` that has a regular exact form, and
    return the rewrites, each with its test function and original text, so one that fails reverts."""
    import lint_slop

    path = REPO / relative
    source = path.read_text(encoding="utf-8")
    [module] = list(lint_slop.collect([relative], REPO, "dew"))
    functions = [node for node in ast.walk(module.tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)]
    parents = {id(child): parent for parent in ast.walk(module.tree) for child in ast.iter_child_nodes(parent)}
    edits = []
    for node, what in lint_slop._tolerance_comparisons(module):
        if "one-draw rule" in what:
            continue
        owner = min((function for function in functions if node in ast.walk(function)),
                    key=lambda function: len(list(ast.walk(function))), default=None)
        if owner is None:
            continue
        scope = _Scope(module.tree, owner)
        operands = (node.args[:2] if isinstance(node, ast.Call) else
                    [node.left, *node.comparators] if isinstance(node, ast.Compare) else [node])
        if any(_float64(operand, scope) for operand in operands):
            continue
        kind = _kind(node, owner, scope, parents)
        replacement = _exact_form(node, source)
        line = source.splitlines()[node.lineno - 1]
        width = len(line) - (len(ast.get_source_segment(source, node).splitlines()[0]) if node.lineno ==
                             node.end_lineno else 0) + len(replacement or "")
        if kind[:3] in classes and replacement and not _tolerance_named(node) and (
                node.lineno != node.end_lineno or width <= 110):
            test = next((function.name for function in functions
                         if function.name.startswith("test") and node in ast.walk(function)), owner.name)
            edits.append({"file": relative, "line": node.lineno, "col": node.col_offset,
                          "end_line": node.end_lineno, "end_col": node.end_col_offset, "test": test,
                          "class": kind, "original": ast.get_source_segment(source, node),
                          "replacement": replacement})
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    text = source
    for edit in sorted(edits, key=lambda edit: (edit["line"], edit["col"]), reverse=True):
        start = offsets[edit["line"] - 1] + len(lines[edit["line"] - 1].encode()[:edit["col"]].decode())
        end = offsets[edit["end_line"] - 1] + len(lines[edit["end_line"] - 1].encode()[:edit["end_col"]].decode())
        text = text[:start] + edit["replacement"] + text[end:]
    if edits:
        path.write_text(text, encoding="utf-8")
    return edits


def revert(manifest: list[dict], refused: set[tuple[str, str]]) -> list[dict]:
    """Undo each rewrite whose (file, test) or (file, line) is in `refused`, from the bottom of each
    file up, so earlier rewrites' positions hold; return the rewrites kept."""
    kept, undone = [], collections.defaultdict(list)
    for edit in manifest:
        if (edit["file"], edit["test"]) in refused or (edit["file"], str(edit["line"])) in refused:
            undone[edit["file"]].append(edit)
        else:
            kept.append(edit)
    for relative, edits in undone.items():
        path = REPO / relative
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        for edit in sorted(edits, key=lambda edit: (edit["line"], edit["col"]), reverse=True):
            start = sum(len(line) for line in lines[:edit["line"] - 1]) + edit["col"]
            assert text[start:start + len(edit["replacement"])] == edit["replacement"], edit
            text = text[:start] + edit["original"] + text[start + len(edit["replacement"]):]
            lines = text.splitlines(keepends=True)
        path.write_text(text, encoding="utf-8")
    return kept


RECIPES = {
    "seeded": "check 1: the test's own literal seed becomes the draw's seed (`assert_as_exact`)",
    "recorded": "record the reference over (1 + REDRAW)·K seeds (`record_case`, its evidence where large) "
                "and judge through `indexed_recording`",
    "other: parametrized or a fixture": "make the fixture or parametrization draw from a seed, then check 1",
    "other: a helper given a literal": "the helper's literal is the seed: pass the draw's seed, then check 1",
    "other: constants": "draw the constants' inputs from a seed, then check 1, or exact where they are exact",
    "(a) a contract": "the simpler arm run in float64 is the truth; both arms through check 1",
    "(b) the same computation": "exact equality",
    "(c) a constant": "exact equality, or a rigorous bound written out",
    "(d) a golden number": "delete",
    "(d) a value against itself": "delete",
    "(e) two things asserted to differ": "`assert_moves`, or exact inequality where the change is "
                                         "structural; delete a check that only re-asserts a detail",
}


def lists(directory: Path) -> dict:
    """Each lane's residue, site by site with its family, class and recipe, written to `directory`."""
    listed = subprocess.run(["git", "ls-files", "tests/*.py", "tools/*.py"], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout.split()
    import lint_slop

    by_lane = collections.defaultdict(list)
    for relative in listed:
        # The gate's own tests test the families being retired, and go with them.
        if relative.endswith("check1_codemod.py") or relative in lint_slop.GATE:
            continue
        for site in sites(REPO / relative):
            by_lane[_lane(relative)].append((relative, site["line"], site["family"], site["class"]))
        for site in tolerance_sites(relative):
            family = "hand tolerance" + (", on a float64 truth" if site["form"] == "on a float64 truth" else "")
            by_lane[_lane(relative)].append((relative, site.get("line", 0), family, site["class"]))
    directory.mkdir(parents=True, exist_ok=True)
    counts = {}
    for lane, rows in sorted(by_lane.items()):
        lines = [f"# {lane}: {len(rows)} sites to judge through check 1 by 2026-11-27", "",
                 "Classes are the codemod's (tools/check1_codemod.py), an estimate per site: confirm the "
                 "class when converting. Each row: site, family, class, recipe.", ""]
        for relative, line, family, kind in sorted(rows):
            lines.append(f"- `{relative}:{line}`: {family}; {kind}; {RECIPES.get(kind, 'convert by hand')}")
        (directory / f"{lane.replace(' ', '-').replace('&', 'and').replace(',', '')}.md").write_text(
            "\n".join(lines) + "\n")
        counts[lane] = len(rows)
    return counts


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
    parser.add_argument("command", choices=("survey", "exact", "revert", "lists"))
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--manifest", type=Path, help="exact: where to write the rewrites made; revert: "
                        "the rewrites to undo from, rewritten to those kept")
    parser.add_argument("--refused", type=Path, help="revert: a file of `file::test` or `file:line` lines")
    parser.add_argument("--directory", type=Path, help="lists: where to write each lane's residue")
    options = parser.parse_args()
    if options.command == "exact":
        edits = [edit for relative in options.paths for edit in exact_pass(relative, {"(b)", "(c)"})]
        options.manifest.write_text(json.dumps(edits, indent=1))
        print(f"{len(edits)} rewrites in {len({edit['file'] for edit in edits})} files")
        return 0
    if options.command == "lists":
        print(json.dumps(lists(options.directory), indent=1))
        return 0
    if options.command == "revert":
        refused = set()
        for line in options.refused.read_text().split():
            relative, _, rest = line.partition("::") if "::" in line else line.partition(":")
            refused.add((relative, rest.split("[")[0]))
        kept = revert(json.loads(options.manifest.read_text()), refused)
        options.manifest.write_text(json.dumps(kept, indent=1))
        print(f"{len(kept)} rewrites kept")
        return 0
    result = survey(options.paths)
    totals = collections.defaultdict(collections.Counter)
    for key, counter in result.items():
        totals[key.split(" / ")[0]].update(counter)
    print(json.dumps({"by family": {family: dict(counter) for family, counter in totals.items()},
                      "by family and lane": result}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
