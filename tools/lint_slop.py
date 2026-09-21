"""Rejects low-evidence Python: the rules a type checker and ruff cannot state.

Ported in spirit from dmmulroy/anti-slop, which rejects TypeScript that claims
less than the author knew: `unknown` in a contract, an open dictionary, a type
assertion with no invariant behind it, `typeof` narrowing instead of parsing at
the boundary, a symbol named for its shape, a mocked module instead of a seam.
Python's version of each is below. No dependency, one output format,
`path:line:col: SLOPxxx message`, exit 1 on any finding.

Scope is per rule, because the rules are not all about the same thing:

- `src/dew` is the published contract, so every rule runs there.
- `tests`, `tools` and `recipes` are scripts and proofs. Their names and
  annotations are local, so the contract rules (SLOP001, SLOP002, SLOP004,
  SLOP005) do not run there. A swallowed exception, a narration comment and an
  unsplittable function are defects anywhere, so those do.
- SLOP008 is about the suite only.

Enforcement is staged. `ENFORCED` names the rules the tree is at zero for, and
those are the ones that fail the gate; `ADOPTING` names the rest, which print
with a per-rule file count and do not set the exit status. A rule moves from
`ADOPTING` to `ENFORCED` in the same commit that takes its count to zero, so
the gate is green from the first commit and gets stricter, never redder.

Analysis boundaries, stated the way anti-slop states its own: this reads one
file's AST, with no imported definitions and no inference across calls.
SLOP004's isinstance half fires only when the annotation it needs is written in
the same scope; a value whose type arrives from another module is not narrowed
by this checker and is not reported. SLOP006 walks the handler body it can see,
so an exception handed to a function that re-raises elsewhere reads as reported.
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Which rules fail the gate. A rule is enforced once the tree is at zero for
# it, and it moves here in the same commit that takes it to zero; until then it
# is counted, printed per rule with the file count, and left out of the exit
# status. Adopting is not suppressing: every finding prints, and a rule never
# moves back. There are no per-finding suppressions in either set.
ENFORCED = frozenset({"SLOP003", "SLOP005", "SLOP006", "SLOP007", "SLOP008"})
# SLOP001, SLOP002 and SLOP004 wait on the typing sweep. SLOP009 measures what
# a split would cost rather than what a change introduced, so it waits on the
# decoder split and is the one rule that may stay here.
ADOPTING = frozenset({"SLOP001", "SLOP002", "SLOP004", "SLOP009"})

# The two sanctioned open-mapping aliases: one variables tree, one batch. Every
# other open dictionary in a contract is SLOP002, and a second declaration of
# either name is an import that was not written.
SANCTIONED_ALIASES = {"Variables": "src/dew/objectives/base.py", "Batch": "src/dew/data/dataset.py"}
MAPPINGS = {"dict", "Dict", "Mapping", "MutableMapping", "defaultdict", "OrderedDict"}
OPEN_VALUES = {("dict", "Any"), ("Dict", "Any"), ("Mapping", "Any"), ("MutableMapping", "Any"),
               ("defaultdict", "Any"), ("OrderedDict", "Any"), ("dict", "object"),
               ("Dict", "object"), ("defaultdict", "object"), ("OrderedDict", "object")}
# The one open mapping that promises what it means: read-only, and every value
# has to be narrowed before it is used. It is the type of a config file just
# parsed out of JSON, and the rule wants those parsed once at a boundary.
BOUNDARY_MAPPING = {("Mapping", "object"), ("MappingProxyType", "object")}

VAGUE = {"tmp", "temp", "obj", "thing", "info", "item", "items", "val", "helper", "helpers",
         "util", "utils", "manager", "handler", "res", "ret", "arr", "lst", "dct", "num",
         "cnt", "idx", "flag", "foo", "bar", "data", "result", "results"}
VAGUE_SUFFIXES = ("_impl", "_v2", "_new", "_old", "_copy")
# Three words the tree earns. `value` is the attention V, `Mean.value` and the
# partner of `key` in 296 more places; `values` is the same plural, the critic
# values of GAE among them; `out` is the output array a numeric function
# returns, which is what numpy calls its own out= parameter. Counted at 405,
# 63 and 56 uses before they were dropped, not allowlisted per call site.
# `attention_impl` is 59 uses of jax.nn.dot_product_attention's own
# `implementation` argument as a module field, so the `_impl` suffix spares it.
DOMAIN_WORDS = {"value", "values", "out", "attention_impl"}
NARRATION = re.compile(
    r"^(now |then |this (function|method|class|line) |here we |we (now|then) |increment|decrement"
    r"|loop over|iterate|call |return the|set the|get the|create (a|the)|initialize|init )",
    re.IGNORECASE)
MARKERS = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")
SUPPRESSIONS = re.compile(r"(?P<hit>typing\.cast\(|(?<![\w.])cast\(|# *type: *ignore"
                          r"|# *pyright: *ignore|# *noqa)")
# A probe that asks a library or the environment what it supports, not a probe
# that asks one of our own values whether it kept its contract.
PROBE_ROOTS = {"jax", "jnp", "np", "numpy", "environ", "os", "sys", "flags", "importlib"}
PROBE_ATTRS = {"sharding", "dtype", "shape", "device", "devices"}
# The seams a test is allowed to replace, because the real one leaves the process.
SEAMS = ("subprocess", "socket", "urllib", "request", "http", "download", "hub", "hf_hub",
         "time", "sleep", "monotonic", "perf_counter", "open", "path", "os.", "environ",
         "fetch", "client", "urlopen", "snapshot")
NARROWERS = {"Mapping", "MutableMapping", "dict", "Dict", "Sequence", "list", "tuple", "str",
             "int", "float", "bool", "bytes"}


@dataclass(frozen=True)
class Finding:
    """One finding, printed as `path:line:col: SLOPxxx message`."""

    path: str
    line: int
    col: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: {self.code} {self.message}"


@dataclass
class Module:
    """A parsed file plus the two things every rule asks about it."""

    path: Path
    relative: str
    source: str
    tree: ast.Module

    @property
    def lines(self) -> list[str]:
        return self.source.splitlines()

    @property
    def is_source(self) -> bool:
        return self.relative.startswith("src/dew/")


def _named(node: ast.expr | None) -> str:
    """The dotted spelling of a name or attribute, or "" for anything else."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _named(node.value)
        return f"{head}.{node.attr}" if head else node.attr
    return ""


def _mapping_kind(node: ast.expr) -> str:
    """"open" for a dictionary type that promises nothing, "boundary" for the
    read-only mapping of unnarrowed values, "" for anything else."""
    if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Tuple):
        return ""
    container = _named(node.value).rsplit(".", 1)[-1]
    if container not in MAPPINGS | {"MappingProxyType"} or len(node.slice.elts) != 2:
        return ""
    pair = (container, _named(node.slice.elts[1]).rsplit(".", 1)[-1])
    return "open" if pair in OPEN_VALUES else "boundary" if pair in BOUNDARY_MAPPING else ""


def _mappings(annotation: ast.expr) -> tuple[set[int], list[ast.expr]]:
    """Every mapping type in one annotation: the node ids whose width a
    mapping already accounts for, and the open dictionaries to report."""
    claimed: set[int] = set()
    reported: list[ast.expr] = []
    for found in ast.walk(annotation):
        if not isinstance(found, ast.expr):
            continue
        kind = _mapping_kind(found)
        if kind and isinstance(found, ast.Subscript):
            claimed |= {id(child) for child in ast.walk(found.slice)}
            if kind == "open":
                reported.append(found)
    return claimed, reported


def _widest(node: ast.expr, skip: set[int]) -> Iterator[ast.expr]:
    """Every `Any` or bare `object` in an annotation, minus claimed subtrees."""
    for child in ast.walk(node):
        if id(child) in skip or not isinstance(child, ast.Name | ast.Attribute):
            continue
        if _named(child).rsplit(".", 1)[-1] in {"Any", "object"}:
            yield child


def _annotations(function: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[tuple[str, ast.expr]]:
    """Every annotation in a signature, paired with the name it belongs to."""
    arguments = function.args
    for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                     arguments.vararg, arguments.kwarg):
        if argument is not None and argument.annotation is not None:
            yield argument.arg, argument.annotation
    if function.returns is not None:
        yield "return", function.returns


def _alias_value(node: ast.stmt) -> tuple[str, ast.expr] | None:
    """The name and target of a type alias, in any of the three spellings."""
    if isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
        return node.name.id, node.value
    if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
            and _named(node.annotation).rsplit(".", 1)[-1] == "TypeAlias" and node.value):
        return node.target.id, node.value
    if (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Subscript | ast.BinOp)
            and any(_named(name).rsplit(".", 1)[-1] in {"Any", "object"}
                    for name in ast.walk(node.value) if isinstance(name, ast.Name | ast.Attribute))):
        return node.targets[0].id, node.value
    return None


def contracts(module: Module) -> Iterator[Finding]:
    """SLOP001 and SLOP002: what a signature and an alias promise."""
    def report(node: ast.expr, code: str, message: str) -> Finding:
        return Finding(module.relative, node.lineno, node.col_offset + 1, code, message)

    for node in ast.walk(module.tree):
        alias = _alias_value(node) if isinstance(node, ast.stmt) else None
        if alias is not None:
            name, target = alias
            sanctioned = SANCTIONED_ALIASES.get(name) == module.relative
            if name in SANCTIONED_ALIASES and not sanctioned:
                yield report(target, "SLOP002",
                             f"re-declares the {name} alias; import it from "
                             f"{SANCTIONED_ALIASES[name].removeprefix('src/').replace('/', '.')}")
            elif not sanctioned:
                claimed, open_dictionaries = _mappings(target)
                for mapping in open_dictionaries:
                    yield report(mapping, "SLOP002",
                                 f"alias {name} is an open dictionary; name the keys it carries")
                for wide in _widest(target, claimed):
                    yield report(wide, "SLOP001",
                                 f"alias {name} resolves to {_named(wide)}; it promises nothing")
            continue
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for name, annotation in _annotations(node):
            claimed, open_dictionaries = _mappings(annotation)
            for mapping in open_dictionaries:
                yield report(mapping, "SLOP002",
                             f"{name} is an open dictionary; name the keys it carries, "
                             f"take one of Variables, Batch, or read it as Mapping[str, object]")
            for wide in _widest(annotation, claimed):
                if _named(wide) == "object" and name in {"cause", "error"}:
                    continue
                yield report(wide, "SLOP001", f"{name} is annotated {_named(wide)}; "
                                              f"declare the type the code relies on")


def suppressions(module: Module) -> Iterator[Finding]:
    """SLOP003: a cast or an ignore comment is evidence nobody produced."""
    for number, text in enumerate(module.lines, start=1):
        code, _, comment = text.partition("#")
        # One sanctioned inline suppression: an import kept for the registry
        # entry it makes, which is a side effect ruff has no way to see. The
        # name may stand on an import line or inside a parenthesized block.
        imported = (code.lstrip().startswith(("import ", "from "))
                    or re.fullmatch(r"\s*[\w.]+ *,? *", code) is not None)
        registration = imported and "F401" in comment and "registers" in comment
        for match in SUPPRESSIONS.finditer(text):
            hit = match.group("hit")
            if registration and hit.lstrip().startswith("#"):
                continue
            yield Finding(module.relative, number, match.start() + 1, "SLOP003",
                          f"{hit.strip()} asserts what the code did not prove")


def _own(scope: ast.AST) -> Iterator[ast.AST]:
    """Every node in one scope, without descending into a nested scope."""
    for node in ast.iter_child_nodes(scope):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef):
            continue
        yield node
        yield from _own(node)


def probes(module: Module) -> Iterator[Finding]:
    """SLOP004: asking a value at runtime what its type already said."""
    modules = {alias.asname or alias.name.split(".")[0]
               for node in ast.walk(module.tree) if isinstance(node, ast.Import)
               for alias in node.names}
    document = ast.get_docstring(module.tree) or ""
    for function in [None, *(node for node in ast.walk(module.tree)
                             if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef
                                           | ast.Lambda))]:
        if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            own = ast.get_docstring(function) or ""
            if function.name.startswith("_") and "boundary" in (own + document).lower():
                continue
        declared = _declared(function)
        for node in _own(function or module.tree):
            if not isinstance(node, ast.Call):
                continue
            name = _named(node.func)
            if name in {"getattr", "hasattr"} and len(node.args) >= 2:
                attribute = node.args[1].value if isinstance(node.args[1], ast.Constant) else None
                root = _named(node.args[0]).split(".")[0]
                if not isinstance(attribute, str) or attribute in PROBE_ATTRS:
                    continue
                if root in PROBE_ROOTS | modules or (name == "getattr" and len(node.args) < 3):
                    continue
                yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP004",
                              f"{name} probes .{attribute} for a contract; declare it")
            elif name == "isinstance" and len(node.args) == 2:
                subject = _named(node.args[0])
                kinds = (node.args[1].elts if isinstance(node.args[1], ast.Tuple)
                         else [node.args[1]])
                if subject not in declared or not kinds:
                    continue
                if all(_named(kind).rsplit(".", 1)[-1] in NARROWERS for kind in kinds):
                    yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP004",
                                  f"{subject} is declared {declared[subject]}; the isinstance "
                                  f"selects a path its own type already decided")


def _declared(function: ast.AST | None) -> dict[str, str]:
    """Names with a written, non-union annotation in this scope."""
    if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
        return {}
    written: dict[str, ast.expr] = dict(_annotations(function))
    for node in _own(function):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            written[node.target.id] = node.annotation
    resolved = {}
    for name, annotation in written.items():
        if isinstance(annotation, ast.BinOp) or name == "return":
            continue
        spelling = _named(annotation) or _named(getattr(annotation, "value", None))
        if spelling and spelling.rsplit(".", 1)[-1] not in {"Any", "object", "Optional", "Union"}:
            resolved[name] = spelling
    return resolved


def names(module: Module) -> Iterator[Finding]:
    """SLOP005: a name that describes its slot instead of its contents.

    A trailing digit is not one of those. In ported numeric code `norm2`,
    `conv2` and `net_2` are the reference module names a checkpoint is keyed
    by, and `mu_x2`, `sigma_y2`, `d2` and `k2` are squares and second-order
    terms of the equations being implemented: 50 of them, against none that
    meant "the second version of". `_v2` still reports.
    """
    comprehended = {id(target) for node in ast.walk(module.tree)
                    if isinstance(node, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp)
                    for generator in node.generators for target in ast.walk(generator.target)}
    # An annotated name in a class body is a field: a config key, a CLI flag
    # and a pytree entry a checkpoint is written with. Renaming one is a
    # migration with a converter, which is not something a linter asks for.
    fields = {id(node.target) for node in ast.walk(module.tree)
              if isinstance(node, ast.ClassDef)
              for statement in node.body if isinstance(statement, ast.AnnAssign)
              for node in [statement]}
    for node in ast.walk(module.tree):
        if isinstance(node, ast.arg):
            found, line, col = node.arg, node.lineno, node.col_offset
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if id(node) in fields or (id(node) in comprehended and len(node.id) <= 3):
                continue
            found, line, col = node.id, node.lineno, node.col_offset
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            found, line, col = node.attr, node.lineno, node.col_offset
        else:
            continue
        if found in DOMAIN_WORDS:
            continue
        if found in VAGUE or found.endswith(VAGUE_SUFFIXES):
            yield Finding(module.relative, line, col + 1, "SLOP005",
                          f"`{found}` names a slot, not a value; say what it holds")


def _reported(handler: ast.ExceptHandler) -> bool:
    """Does the body raise, annotate, or pass the exception on to someone?"""
    for node in ast.walk(handler):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call) and _named(node.func).endswith("add_note"):
            return True
        if handler.name is None:
            continue
        held = {child.id for child in ast.walk(node)
                if isinstance(child, ast.Name) and child.id == handler.name}
        if held and isinstance(node, ast.Return | ast.Assign | ast.Call | ast.AugAssign):
            return True
    return False


def swallowed(module: Module) -> Iterator[Finding]:
    """SLOP006: an except that ends the story instead of telling it.

    `contextlib.suppress(SomethingNarrow)` is a decision written at the site
    and is not reported; a broad suppress is the empty handler with a nicer
    spelling, so it is.
    """
    for node in ast.walk(module.tree):
        if (isinstance(node, ast.Call) and _named(node.func).endswith("suppress")
                and any(_named(kind) in {"Exception", "BaseException"} for kind in node.args)):
            yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP006",
                          "suppress(Exception) hides every failure; name the ones this handles")
        if not isinstance(node, ast.ExceptHandler):
            continue
        kinds = ([_named(kind) for kind in node.type.elts] if isinstance(node.type, ast.Tuple)
                 else [_named(node.type)] if node.type is not None else ["bare except"])
        broad = node.type is None or any(kind in {"Exception", "BaseException"} for kind in kinds)
        empty = len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
        nothing = (len(node.body) == 1 and isinstance(node.body[0], ast.Return)
                   and (node.body[0].value is None
                        or (isinstance(node.body[0].value, ast.Constant)
                            and node.body[0].value.value is None)))
        if empty or nothing:
            yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP006",
                          f"`except {'/'.join(kinds)}` {'passes' if empty else 'returns None'}; "
                          f"the failure leaves no trace")
        elif broad and not _reported(node):
            yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP006",
                          f"`except {'/'.join(kinds)}` neither re-raises, notes, nor reports; "
                          f"narrow it to what this code handles")


def comments(module: Module) -> Iterator[Finding]:
    """SLOP007: a comment that reads the code back instead of saying why."""
    readable = io.StringIO(module.source).readline
    previous = (0, -1)
    for token in tokenize.generate_tokens(readable):
        if token.type != tokenize.COMMENT:
            continue
        text = token.string.lstrip("#").strip()
        line, col = token.start[0], token.start[1] + 1
        # A block of comment lines is one comment. Only its first line opens
        # the thought, so only its first line can open it with narration.
        continuation, previous = previous == (line - 1, col), (line, col)
        if module.is_source and (marker := MARKERS.search(text)) is not None:
            yield Finding(module.relative, line, col, "SLOP007",
                          f"{marker.group(1)} defers the work into a comment")
        if continuation:
            continue
        if NARRATION.match(text):
            yield Finding(module.relative, line, col, "SLOP007",
                          "the comment narrates the next line; say why or delete it")
        elif (word := re.split(r"[^\w]", text)[-1]) and _labels(module, line, word):
            yield Finding(module.relative, line, col, "SLOP007",
                          f"the comment restates `{word}`; say why or delete it")


def _labels(module: Module, line: int, word: str) -> bool:
    """Is `word` the whole name the next statement binds or defines?"""
    lines = module.lines
    for text in lines[line:line + 1]:
        stripped = text.strip()
        if stripped.startswith(("def ", "class ", "async def ")):
            return stripped.split("(")[0].split()[-1] == word
        head = stripped.split("=")[0].strip() if "=" in stripped else ""
        return head.split(":")[0].strip() == word and bool(head)
    return False


def mocks(module: Module) -> Iterator[Finding]:
    """SLOP008: a first-party module replaced instead of a seam taken."""
    for node in ast.walk(module.tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = _named(node.func)
        target = ""
        if name.endswith(("mock.patch", "patch", "patch.object")):
            target = node.args[0].value if isinstance(node.args[0], ast.Constant) else ""
        elif name.endswith("monkeypatch.setattr"):
            target = (node.args[0].value if isinstance(node.args[0], ast.Constant)
                      else _named(node.args[0]))
        if not isinstance(target, str) or not target.startswith(("dew.", "dew")):
            continue
        if not target.startswith("dew.") or any(seam in target.lower() for seam in SEAMS):
            continue
        yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP008",
                      f"patches {target}; the fix is a seam the test can pass a double to")


def size(module: Module) -> Iterator[Finding]:
    """SLOP009: a unit nobody can hold in their head."""
    total = len(module.lines)
    if total > 2500:
        yield Finding(module.relative, total, 1, "SLOP009",
                      f"{total} lines in one module; split it along its own seams")
    for node in ast.walk(module.tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        length = (node.end_lineno or node.lineno) - node.lineno + 1
        if length > 120:
            yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP009",
                          f"{node.name} is {length} lines; name its parts")


CONTRACT_RULES = (contracts, suppressions, probes, names)
UNIVERSAL_RULES = (swallowed, comments, size)


def check(module: Module) -> Iterator[Finding]:
    """Every rule that applies to this file, in code order."""
    rules = (*CONTRACT_RULES, *UNIVERSAL_RULES) if module.is_source else UNIVERSAL_RULES
    if module.relative.startswith("tests/"):
        rules = (*rules, mocks)
    findings = [finding for rule in rules for finding in rule(module)]
    yield from sorted(findings, key=lambda finding: (finding.line, finding.col, finding.code))


def collect(roots: Sequence[str]) -> Iterator[Module]:
    """Every Python file under the named roots, skipping stub-only trees."""
    for root in roots:
        for path in sorted((ROOT / root).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if "/stubs/" in f"/{relative}":
                continue
            yield Module(path, relative, path.read_text(), ast.parse(path.read_text()))


def main() -> int:
    roots = sys.argv[1:] or ["src/dew", "tests", "tools", "recipes"]
    counts: dict[str, int] = {}
    files: dict[str, set[str]] = {}
    for module in collect(roots):
        for finding in check(module):
            print(finding)
            counts[finding.code] = counts.get(finding.code, 0) + 1
            files.setdefault(finding.code, set()).add(finding.path)
    for code, count in sorted(counts.items()):
        where = f"{count} in {len(files[code])} files"
        state = "enforced" if code in ENFORCED else "adopting"
        print(f"{code} ({state}): {where}", file=sys.stderr)
    failures = sum(count for code, count in counts.items() if code in ENFORCED)
    adopting = ", ".join(f"{code} {counts[code]}" for code in sorted(counts)
                         if code in ADOPTING) or "none"
    print(f"enforced: {failures} findings; adopting: {adopting}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
