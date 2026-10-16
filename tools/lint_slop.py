"""Rejects low-evidence Python: the rules a type checker and ruff cannot state.

Ported in spirit from dmmulroy/anti-slop, which rejects TypeScript that claims
less than the author knew: `unknown` in a contract, an open dictionary, a type
assertion with no invariant behind it, `typeof` narrowing instead of parsing at
the boundary, a symbol named for its shape, a mocked module instead of a seam.
Python's version of each is below. No dependency, one output format,
`path:line:col: SLOPxxx message`, exit 1 on any finding.

Scope is per rule, because the rules are not all about the same thing:

- `src/dew` is the published contract, so every rule runs there. A plugin
  package checks itself with `--root` (its checkout) and `--package` (its
  name, so `src/<package>` is the contract and its own modules are the ones
  SLOP008 will not see patched).
- `tests`, `tools`, `recipes`, `examples`, `site`, `docs/assets` and
  `kernels` are scripts, proofs and the packages beside Dew's. Their
  names and annotations are local, so the contract rules (SLOP001, SLOP002,
  SLOP004, SLOP005) do not run there. A suppression, a swallowed exception, a
  narration comment and an unsplittable function are defects anywhere, so
  those do.
- SLOP008 is about the suite only.
- SLOP010 is about the contract's consumers of its models, so it runs in
  `src/<package>`, against an index of the whole package.
- SLOP011 is about the suite. A test proves behaviour, not wording or source.
- SLOP012 runs everywhere but the suite, which SLOP008 covers.
- SLOP013 is about what importing `src/<package>` does to the process.
- SLOP014 is about the mixer call contract in `src/<package>`: a layer's call
  reads every call argument it takes, and a token mixer's `reads` names the
  `MIXING` fields (`src/<package>/nn/inputs.py`) it reads, no more and no
  fewer, since its block refuses a call that sets any other.
- SLOP015 is about the suite. A test imports its own package, so an import
  the package breaks fails the suite instead of skipping it.

Every rule fails the gate, and none has a per-finding suppression.

Analysis boundaries, stated the way anti-slop states its own: this reads one
file's AST, with no imported definitions and no inference across calls, but
for SLOP010, which reads the whole package's imports, re-exports, aliases and
class bases, and Flax's Module from its installed source (never importing
it), to know which classes are models. It sees what Python spells, not what
dynamic dispatch reaches; behaviour across models is the capability matrix's
proof (tests/test_capability_matrix.py).
SLOP011 and SLOP012 read each name through the imports of its own lexical
scope (tools/lint_scopes.py).
SLOP004's isinstance half fires only when the annotation it needs is written in
the same scope; a value whose type arrives from another module is not narrowed
by this checker and is not reported. SLOP006 walks the handler body it can see,
so an exception handed to a function that re-raises elsewhere reads as reported.
"""

from __future__ import annotations

import argparse
import ast
import functools
import io
import re
import sys
import tokenize
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lint_scopes as scopes

ROOT = Path(__file__).resolve().parent.parent

# The two sanctioned open-mapping aliases: one variables tree, one batch, both
# declared in dew.objectives.base, which is the layer both the trainer and the
# data pipeline already depend on. Every other open dictionary in a contract is
# SLOP002, and a second declaration of either name is an import that was not
# written.
SANCTIONED_ALIASES = {"Variables": "src/dew/objectives/base.py",
                      "Batch": "src/dew/objectives/base.py"}
MAPPINGS = {"dict", "Dict", "Mapping", "MutableMapping", "defaultdict", "OrderedDict"}
OPEN_VALUES = {("dict", "Any"), ("Dict", "Any"), ("Mapping", "Any"), ("MutableMapping", "Any"),
               ("defaultdict", "Any"), ("OrderedDict", "Any"), ("dict", "object"),
               ("Dict", "object"), ("defaultdict", "object"), ("OrderedDict", "object")}
# The one open mapping that promises what it means: read-only, and every value
# has to be narrowed before it is used. It is the type of a config file just
# parsed out of JSON, and the rule wants those parsed once at a boundary.
BOUNDARY_MAPPING = {("Mapping", "object"), ("MappingProxyType", "object"),
                    ("Sequence", "object"), ("tuple", "object")}

VAGUE = {"tmp", "temp", "obj", "thing", "info", "item", "items", "val", "helper", "helpers",
         "util", "utils", "manager", "handler", "res", "ret", "arr", "lst", "dct", "num",
         "cnt", "idx", "flag", "foo", "bar", "data", "result", "results"}
VAGUE_SUFFIXES = ("_impl", "_v2", "_new", "_old", "_copy")
# Three words the tree earns. `value` is the attention V, `Ratio.value` and the
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
# What a comment says to make a checker look away.
SUPPRESSIONS = re.compile(r"\b(noqa|type: *ignore|pyright: *ignore)\b")
# `typing.cast`, which typing_extensions re-exports.
CASTS = {"typing.cast", "typing_extensions.cast"}
# The calls that set a warning aside: each ignores one when its action is "ignore".
WARNING_FILTERS = {"warnings.simplefilter", "warnings.filterwarnings", "warnings.catch_warnings",
                   "pytest.mark.filterwarnings"}
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
# The hooks Python documents as set by assignment.
INTERPRETER_HOOKS = {"sys.excepthook", "sys.unraisablehook", "sys.displayhook", "sys.breakpointhook",
                     "threading.excepthook"}
# The calls that hand a test a function's source text.
SOURCE_READERS = {"inspect.getsource", "inspect.getsourcelines", "inspect.getsourcefile"}
# What a layer's call is handed by its block, which it may not take and drop.
CALL_ARGUMENTS = {"decode", "kv_store", "attention_metadata", "call"}
# Process-wide state a module must not change while it is imported.
GLOBAL_WRITES = ("config.update", "environ.setdefault", "environ.update", "putenv", "basicConfig",
                 "register_event_listener", "filterwarnings", "simplefilter")


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
    package: str = "dew"

    @property
    def lines(self) -> list[str]:
        return self.source.splitlines()

    @property
    def is_source(self) -> bool:
        return self.relative.startswith(f"src/{self.package}/")


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
    if not isinstance(node, ast.Subscript):
        return ""
    container = _named(node.value).rsplit(".", 1)[-1]
    if container not in MAPPINGS | {"MappingProxyType", "Sequence", "tuple"}:
        return ""
    # A mapping names its key and its value; a sequence names one element, and
    # `Sequence[object]` is one of the boundary forms below, so both arities
    # are read here rather than only the pair.
    arguments = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    if len(arguments) not in (1, 2):
        return ""
    pair = (container, _named(arguments[-1]).rsplit(".", 1)[-1])
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


def _parses(function: ast.FunctionDef | ast.AsyncFunctionDef, parameter: str) -> bool:
    """Does this function narrow `parameter` and refuse what it cannot read?

    A boundary parser is the one place `object` is the true type of an input:
    the value arrived from JSON or from a library, the function tests what it
    actually is, and anything else raises with the expectation named. Passing
    an `object` along without narrowing it is still SLOP001.
    """
    if function.returns is None or _named(function.returns).rsplit(".", 1)[-1] in {"object", "Any"}:
        return False
    tested = any(isinstance(node, ast.Call) and _named(node.func) in {"isinstance", "issubclass"}
                 and node.args and _named(node.args[0]) == parameter
                 for node in ast.walk(function))
    refuses = any(isinstance(node, ast.Raise)
                  or (isinstance(node, ast.Call) and _named(node.func).endswith("refuse"))
                  for node in ast.walk(function))
    return tested and refuses


def _alias_value(node: ast.stmt) -> tuple[str, ast.expr] | None:
    """The name and target of a type alias, in any of the three spellings."""
    if isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
        return node.name.id, node.value
    if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
            and _named(node.annotation).rsplit(".", 1)[-1] == "TypeAlias" and node.value):
        return node.target.id, node.value
    generic = isinstance(node, ast.Assign) and isinstance(node.value, ast.Subscript) and isinstance(
        node.value.value, ast.Name | ast.Attribute)
    if (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and (isinstance(node.value, ast.BinOp) or generic)
            and any(_named(name).rsplit(".", 1)[-1] in {"Any", "object"}
                    for name in _outside_calls(node.value))):
        return node.targets[0].id, node.value
    return None


def _outside_calls(node: ast.AST) -> Iterator[ast.Name | ast.Attribute]:
    """The names an expression spells outside any call's arguments: in
    `np.asarray(rows, object)[keep]`, `object` is NumPy's object dtype, not
    a type in an alias."""
    if isinstance(node, ast.Call):
        return
    if isinstance(node, ast.Name | ast.Attribute):
        yield node
    for child in ast.iter_child_nodes(node):
        yield from _outside_calls(child)


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
        for name, annotation in scopes.annotations(node):
            claimed, open_dictionaries = _mappings(annotation)
            for mapping in open_dictionaries:
                yield report(mapping, "SLOP002",
                             f"{name} is an open dictionary; name the keys it carries, "
                             f"take one of Variables, Batch, or read it as Mapping[str, object]")
            for wide in _widest(annotation, claimed):
                if _named(wide) == "object" and (name in {"cause", "error"}
                                                 or _parses(node, name)):
                    continue
                yield report(wide, "SLOP001", f"{name} is annotated {_named(wide)}; "
                                              f"declare the type the code relies on")


def suppressions(module: Module) -> Iterator[Finding]:
    """SLOP003: a cast, an ignore comment or an ignored warning is evidence nobody
    produced. Comments are read as Python tokenizes them, so a string that spells
    one is not one; `typing.cast` and the warning filters through each scope's
    imports, so a function of the file's own named `cast` is not one either."""
    for token in tokenize.generate_tokens(io.StringIO(module.source).readline):
        if token.type != tokenize.COMMENT or (match := SUPPRESSIONS.search(token.string)) is None:
            continue
        line, col = token.start
        code = module.lines[line - 1][:col]
        # One sanctioned inline suppression: an import kept for the registry
        # entry it makes, which is a side effect ruff has no way to see. The
        # name may stand on an import line or inside a parenthesized block.
        imported = (code.lstrip().startswith(("import ", "from "))
                    or re.fullmatch(r"\s*[\w.]+ *,? *", code) is not None)
        if imported and "F401" in token.string and "registers" in token.string:
            continue
        yield Finding(module.relative, line, col + match.start() + 1, "SLOP003",
                      f"# {match.group(1)} asserts what the code did not prove")
    for node, table in scopes.scoped(module.tree.body, {}, _parent(module)):
        if not isinstance(node, ast.Call):
            continue
        called = scopes.resolved(node.func, table)
        if called in CASTS:
            yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP003",
                          "cast() asserts what the code did not prove")
        elif called in WARNING_FILTERS and _ignores(called, node):
            yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP003",
                          f"{called} ignores a warning; fix what raises it")


def _ignores(called: str, call: ast.Call) -> bool:
    """Does a warning filter's call set its warnings aside: a warnings call whose
    action, the first argument or `action=`, is "ignore", or a pytest mark any of
    whose filters, each argument, starts with it?"""
    actions = [*(call.args if called == "pytest.mark.filterwarnings" else call.args[:1]),
               *(keyword.value for keyword in call.keywords if keyword.arg == "action")]
    return any(isinstance(action, ast.Constant) and isinstance(action.value, str)
               and action.value.split(":")[0] == "ignore" for action in actions)


def probes(module: Module) -> Iterator[Finding]:
    """SLOP004: asking a value at runtime what its type already said."""
    modules = {alias.asname or alias.name.split(".")[0]
               for node in ast.walk(module.tree) if isinstance(node, ast.Import)
               for alias in node.names}
    document = ast.get_docstring(module.tree) or ""
    unions = _union_aliases(module.tree)
    for function in [None, *(node for node in ast.walk(module.tree)
                             if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef
                                           | ast.Lambda))]:
        if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            own = ast.get_docstring(function) or ""
            if function.name.startswith("_") and "boundary" in (own + document).lower():
                continue
        declared = _declared(function, unions)
        for node in scopes.own(function or module.tree):
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


def _declared(function: ast.AST | None, unions: set[str]) -> dict[str, str]:
    """Names with a written, non-union annotation in this scope.

    `unions` holds the same-file aliases that resolve to a union, because an
    isinstance over a union's members is narrowing, not a redundant check. A
    dotted annotation belongs to another module and is not resolved here, so
    it is not treated as narrow either: that is the boundary this checker
    documents rather than guesses across.
    """
    if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
        return {}
    written: dict[str, ast.expr] = dict(scopes.annotations(function))
    for node in scopes.own(function):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            written[node.target.id] = node.annotation
    resolved = {}
    opaque = {"Any", "object", "Optional", "Union"}
    for name, annotation in written.items():
        if isinstance(annotation, ast.BinOp) or name == "return":
            continue
        spelling = _named(annotation) or _named(getattr(annotation, "value", None))
        if spelling and "." not in spelling and spelling not in unions | opaque:
            resolved[name] = spelling
    return resolved


def _union_aliases(tree: ast.Module) -> set[str]:
    """The module's own aliases whose target is a union of several members."""
    aliases = set()
    for node in tree.body:
        if isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
            name, target = node.name.id, node.value
        elif (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value):
            name, target = node.target.id, node.value
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)):
            name, target = node.targets[0].id, node.value
        else:
            continue
        head = _named(target) or _named(getattr(target, "value", None))
        if isinstance(target, ast.BinOp) or head.rsplit(".", 1)[-1] in {"Union", "Optional"}:
            aliases.add(name)
    return aliases


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


def _patched(node: ast.Call, table: dict[str, str]) -> str:
    """What a patching call replaces, as a dotted path: `mock.patch("a.b")`'s
    string; `patch.object(owner, "b")`'s and `monkeypatch.setattr(owner, "b",
    ...)`'s or `monkeypatch.setattr("a.b", ...)`'s attribute, its owner read
    through the scope's imports (`scopes.resolved`), however it was imported or
    aliased; `monkeypatch.setitem(mapping, ...)`'s and `delitem`'s mapping,
    which a module's table or registry is. "" for any other call."""
    name = _named(node.func)
    first = node.args[0]
    spelled = first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None
    if name.endswith(("mock.patch", ".patch")) or name == "patch":
        return spelled or ""
    if name.endswith(("patch.object", "monkeypatch.setattr", "monkeypatch.delattr")):
        if spelled is not None:
            return spelled
        attribute = node.args[1] if len(node.args) > 1 else None
        attribute_name = (attribute.value if isinstance(attribute, ast.Constant)
                          and isinstance(attribute.value, str) else "*")
        owner = scopes.resolved(first, table)
        return f"{owner}.{attribute_name}" if owner else ""
    if name.endswith(("monkeypatch.setitem", "monkeypatch.delitem")):
        return scopes.resolved(first, table)
    return ""


def _seam(target: str) -> bool:
    """Does `target` leave the process: does it hold one of `SEAMS` as whole parts
    between its dots and underscores, so `dew.cli.tpu.time.sleep` and
    `dew.interop.sources.snapshot` do and `dew.training.narrow.narrowed_paths` does not?"""
    words = [word for word in re.split(r"[._]", target.lower()) if word]
    for seam in SEAMS:
        parts = [word for word in re.split(r"[._]", seam) if word]
        if any(words[start:start + len(parts)] == parts for start in range(len(words))):
            return True
    return False


def mocks(module: Module) -> Iterator[Finding]:
    """SLOP008: a first-party module, or a table or registry it holds, replaced
    instead of a seam taken."""
    for node, table in scopes.scoped(module.tree.body, {}, _parent(module)):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        target = _patched(node, dict(table))
        if not target.startswith(f"{module.package}."):
            continue
        if _seam(target):
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


def _module_name(module: Module) -> str:
    """`src/dew/nn/transformer.py` as `dew.nn.transformer`, a package as its `__init__`'s."""
    parts = module.relative.removeprefix("src/").removesuffix(".py").split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _parent(module: Module) -> list[str]:
    """The package a module's relative imports start from, as its parts."""
    name = _module_name(module)
    return name.split(".") if module.path.name == "__init__.py" else name.split(".")[:-1]


def _aliased(node: ast.stmt) -> tuple[str, ast.expr] | None:
    """A module-level name bound to other names: `Decoders = A | B`, a tuple of them, a type alias."""
    if isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
        return node.name.id, node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
        return node.target.id, node.value
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id, node.value
    return None


def _members(body: Sequence[ast.stmt]) -> set[str]:
    """The private names a class body declares: its methods, its class-level
    fields (type-checking blocks included) and what its methods set on self."""
    declared: set[str] = set()
    for statement in body:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            declared.add(statement.name)
            declared |= {node.attr for node in ast.walk(statement) if isinstance(node, ast.Attribute)
                         and isinstance(node.ctx, ast.Store) and _named(node.value) == "self"}
        elif not isinstance(statement, ast.ClassDef):
            declared |= {node.id for node in [statement, *scopes.own(statement)]
                         if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    return {name for name in declared if name.startswith("_") and not name.startswith("__") and name != "_"}


@dataclass
class ModelIndex:
    """SLOP010's view of a package: which of its classes are Flax models.

    A model class is found, never listed: a class whose bases reach Flax's
    own Module, through the package's imports, re-exports and subclasses. It
    may be asked for by identity only in the module that defines it, and the
    private members model classes declare, Flax's own among them, only by
    the modules that declare them.
    """

    imports: dict[str, dict[str, str]]
    aliases: dict[str, dict[str, ast.expr]]
    bases: dict[str, list[tuple[str, ast.expr]]]
    defined: dict[str, str]
    models: set[str]
    private: dict[str, set[str]]

    @classmethod
    def build(cls, modules: Sequence[Module]) -> ModelIndex:
        index = cls({}, {}, {}, {}, set(), {})
        bodies: dict[str, list[ast.stmt]] = {}
        for module in modules:
            name = _module_name(module)
            index.imports[name] = scopes.imports(module.tree, _parent(module))
            index.aliases[name] = {}
            for node in module.tree.body:
                if isinstance(node, ast.ClassDef):
                    index.bases[f"{name}.{node.name}"] = [(name, base) for base in node.bases]
                    index.defined[f"{name}.{node.name}"] = module.relative
                    bodies[f"{name}.{node.name}"] = node.body
                elif (aliased := _aliased(node)) is not None:
                    index.aliases[name][aliased[0]] = aliased[1]
        roots = {f"flax.{kind}.Module" for kind in ("linen", "nnx")} | {
            f"flax.{kind}.module.Module" for kind in ("linen", "nnx")}
        grown = True
        while grown:
            reached = {name for name, bases in index.bases.items() if name not in index.models and any(
                index._symbol(module, base) & (roots | index.models) for module, base in bases)}
            index.models |= reached
            grown = bool(reached)
        for name in index.models:
            for member in _members(bodies[name]):
                index.private.setdefault(member, set()).add(index.defined[name])
        for member in _flax_members():
            index.private.setdefault(member, set())
        return index

    def _symbol(self, module: str, node: ast.expr, seen: frozenset[str] = frozenset()) -> set[str]:
        """The classes and dotted paths `node`, read in `module`, stands for:
        one name, or every member of a tuple, a union or an alias of them."""
        if isinstance(node, ast.Tuple | ast.List | ast.Set):
            return set().union(*(self._symbol(module, element, seen) for element in node.elts))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return self._symbol(module, node.left, seen) | self._symbol(module, node.right, seen)
        if isinstance(node, ast.Subscript) and _named(node.value).rsplit(".", 1)[-1] in {"Union", "Optional"}:
            return self._symbol(module, node.slice, seen)
        spelled = _named(node)
        if not spelled:
            return set()
        head, _, rest = spelled.partition(".")
        if not rest and head in self.aliases.get(module, {}) and f"{module}.{head}" not in seen:
            return self._symbol(module, self.aliases[module][head], seen | {f"{module}.{head}"})
        if f"{module}.{head}" in self.defined:
            dotted = f"{module}.{spelled}"
        else:
            dotted = f"{self.imports.get(module, {}).get(head, head)}{'.' + rest if rest else ''}"
        return self._canonical(dotted, seen)

    def _canonical(self, dotted: str, seen: frozenset[str]) -> set[str]:
        """Where `dotted` is defined: through a package's re-export or a
        module's alias, to the class itself."""
        if dotted in self.defined or dotted in seen:
            return {dotted}
        parts = dotted.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            owner = ".".join(parts[:cut])
            if owner in self.imports:
                rest: ast.expr = ast.Name(parts[cut])
                for part in parts[cut + 1:]:
                    rest = ast.Attribute(rest, part)
                return self._symbol(owner, rest, seen | {dotted})
        return {dotted}

    def classes(self, module: str, node: ast.expr) -> set[str]:
        """The model classes `node` names in `module`."""
        return self._symbol(module, node) & self.models


def _flax_members() -> set[str]:
    """The private members of Flax's Module and the bases it declares beside
    it, read from Flax's installed source, never imported: `_try_setup`,
    `_state` and every other one a release has."""
    for entry in sys.path:
        source = Path(entry or ".") / "flax" / "linen" / "module.py"
        if source.is_file():
            tree = ast.parse(source.read_text())
            return set().union(*(_members(node.body) for node in tree.body
                                 if isinstance(node, ast.ClassDef) and node.name in {"Module", "ModuleBase"}))
    raise SystemExit("SLOP010 reads Flax's Module from its installed source; install flax")


def _identity(node: ast.AST) -> ast.expr | None:
    """The class an identity test names: `isinstance(x, C)`, `issubclass(x, C)`,
    `type(x) is C`, `x.__class__ == C`, `type(x) in (C, D)`, `case C()`."""
    if (isinstance(node, ast.Call) and _named(node.func) in {"isinstance", "issubclass"}
            and len(node.args) == 2):
        return node.args[1]
    if isinstance(node, ast.MatchClass):
        return node.cls
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(
            node.ops[0], ast.Is | ast.IsNot | ast.Eq | ast.NotEq | ast.In | ast.NotIn):
        left, right = node.left, node.comparators[0]
        for subject, other in ((left, right), (right, left)):
            if (isinstance(subject, ast.Call) and _named(subject.func) == "type") or (
                    isinstance(subject, ast.Attribute) and subject.attr == "__class__"):
                return other
    return None


def _named_class(node: ast.AST) -> list[str]:
    """The class names a `type(x).__name__ == "Name"` test spells as strings."""
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1):
        return []
    for subject, other in ((node.left, node.comparators[0]), (node.comparators[0], node.left)):
        if isinstance(subject, ast.Attribute) and subject.attr in {"__name__", "__qualname__"}:
            constants = other.elts if isinstance(other, ast.Tuple | ast.List | ast.Set) else [other]
            return [constant.value for constant in constants
                    if isinstance(constant, ast.Constant) and isinstance(constant.value, str)]
    return []


def boundaries(module: Module, models: ModelIndex) -> Iterator[Finding]:
    """SLOP010: a consumer asks a model what it can do, never which one it is.

    An identity test on a model class, by isinstance, issubclass, type or a
    class pattern, alias, tuple and union spellings included, and a read of
    a model's private member, are refused everywhere but the module that
    defines the class or declares the member. The way through is a
    capability (`dew.nn.protocols`): what any model may define, and a
    caller checks for and calls."""
    name = _module_name(module)
    simple = {model.rsplit(".", 1)[-1]: model for model in models.models}
    for node in ast.walk(module.tree):
        if not isinstance(node, ast.expr | ast.pattern):
            continue
        named = _identity(node)
        found = models.classes(name, named) if named is not None else set()
        found |= {simple[spelled] for spelled in _named_class(node) if spelled in simple}
        for model in sorted(found):
            if models.defined[model] != module.relative:
                yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP010",
                              f"asks whether a model is {model.rsplit('.', 1)[-1]} "
                              f"({models.defined[model]}); ask for the capability instead")
        member, receiver = None, None
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            member, receiver = node.attr, node.value
        elif (isinstance(node, ast.Call) and _named(node.func) in {"getattr", "hasattr", "setattr"}
                and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)):
            member, receiver = node.args[1].value, node.args[0]
        if (isinstance(member, str) and member in models.private
                and module.relative not in models.private[member]
                and _named(receiver) not in {"self", "cls"}
                and not (isinstance(receiver, ast.Call) and _named(receiver.func) == "super")):
            yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP010",
                          f"reads {member}, private to a model; ask for a public capability instead")


def _script_guard(node: ast.If) -> bool:
    """`if __name__ == "__main__":`, whose body runs only as a script."""
    test = node.test
    return (isinstance(test, ast.Compare) and _named(test.left) == "__name__"
            and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value == "__main__")


def wording(module: Module) -> Iterator[Finding]:
    """SLOP011: a test that checks wording or source text instead of behaviour."""
    for node, table in scopes.scoped(module.tree.body, {}, _parent(module)):
        if isinstance(node, ast.Call):
            called = scopes.resolved(node.func, table)
            if called == "pytest.raises" and any(keyword.arg == "match" for keyword in node.keywords):
                yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP011",
                              "matches an error's wording; assert the refusal, not its text")
            elif called in SOURCE_READERS:
                yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP011",
                              "reads source text; test what the code does")
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and node.value.startswith(f"src/{module.package}/") and node.value.endswith(".py")):
            yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP011",
                          f"names {node.value}; test what the code does, not its source")


def skips(module: Module) -> Iterator[Finding]:
    """SLOP015: a test that skips when its own package fails to import."""
    for node, table in scopes.scoped(module.tree.body, {}, _parent(module)):
        if not (isinstance(node, ast.Call) and scopes.resolved(node.func, table) == "pytest.importorskip"):
            continue
        named = [*node.args[:1], *(keyword.value for keyword in node.keywords if keyword.arg == "modname")]
        if (named and isinstance(named[0], ast.Constant) and isinstance(name := named[0].value, str)
                and (name == module.package or name.startswith(f"{module.package}."))):
            yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP015",
                          f"skips when {name} fails to import; import it, so a broken import fails")


def patches(module: Module) -> Iterator[Finding]:
    """SLOP012: an imported module's attribute replaced or deleted, outside the suite."""
    for node, table in scopes.scoped(module.tree.body, {}, _parent(module)):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store | ast.Del):
            owner, attribute = node.value, node.attr
        elif (isinstance(node, ast.Call) and _named(node.func) in ("setattr", "delattr")
              and len(node.args) > 1):
            owner, named = node.args[:2]
            spelled = isinstance(named, ast.Constant) and isinstance(named.value, str)
            attribute = named.value if spelled else "*"
        else:
            continue
        origin = scopes.resolved(owner, table)
        target = f"{origin}.{attribute}"
        # The library rebinds nothing it imported but the interpreter's own hooks; a
        # script, nothing of our package.
        if origin and target not in INTERPRETER_HOOKS and (
                module.is_source or origin.split(".")[0] == module.package):
            yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP012",
                          f"rebinds {target} for every caller; pass the behaviour in")


def _module_level(body: Sequence[ast.stmt]) -> Iterator[ast.AST]:
    """The statements an import executes: every block but function bodies and the
    script guard's body (its `else` still runs), class bodies included."""
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if isinstance(node, ast.ClassDef):
            yield from _module_level(node.body)
        elif isinstance(node, ast.If):
            if not _script_guard(node):
                yield node.test
                yield from _module_level(node.body)
            yield from _module_level(node.orelse)
        elif isinstance(node, ast.Try | ast.With | ast.For | ast.While):
            for block in ("body", "orelse", "finalbody"):
                yield from _module_level(getattr(node, block, []))
            for handler in getattr(node, "handlers", []):
                yield from _module_level(handler.body)
        else:
            yield node


def imports(module: Module) -> Iterator[Finding]:
    """SLOP013: importing the module changes process-wide state."""
    for statement in _module_level(module.tree.body):
        for node in ast.walk(statement):
            if isinstance(node, ast.Assign):
                written = any(isinstance(target, ast.Subscript) and _named(target.value).endswith("environ")
                              for target in node.targets)
            elif isinstance(node, ast.Call):
                written = _named(node.func).endswith(GLOBAL_WRITES)
            else:
                continue
            if written:
                yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP013",
                              "changes process-wide state at import; do it where the caller asks for it")


def _mixing(module: Module) -> frozenset[str]:
    """The `MIXING` names the package's `nn/inputs.py` assigns, or none where it has none."""
    root = module.path.parents[len(Path(module.relative).parents) - 1]
    return _assigned_mixing(root / "src" / module.package / "nn" / "inputs.py")


@functools.cache
def _assigned_mixing(inputs: Path) -> frozenset[str]:
    if not inputs.is_file():
        return frozenset()
    for node in ast.parse(inputs.read_text()).body:
        if (isinstance(node, ast.Assign) and any(_named(target) == "MIXING" for target in node.targets)
                and isinstance(node.value, ast.Tuple)):
            return frozenset(item.value for item in node.value.elts if isinstance(item, ast.Constant))
    return frozenset()


def _declared_reads(value: ast.expr | None) -> frozenset[str] | str | None:
    """A `reads` declaration's names, "tokenwise" for `TOKENWISE`, or None
    where it is spelled some other way, which this rule cannot read."""
    if _named(value) == "TOKENWISE":
        return "tokenwise"
    if not (isinstance(value, ast.Call) and _named(value.func) == "frozenset" and len(value.args) <= 1):
        return None
    if not value.args:
        return frozenset()
    spelled = value.args[0]
    if isinstance(spelled, ast.Set) and all(isinstance(item, ast.Constant) for item in spelled.elts):
        return frozenset(item.value for item in spelled.elts if isinstance(item, ast.Constant))
    return None


def calls(module: Module) -> Iterator[Finding]:
    """SLOP014: a layer's call that drops what its block hands it, and a mixer
    (a class whose call takes a `call`) whose `reads` is not what it reads."""
    mixing = _mixing(module)
    loaded = {node.attr for node in ast.walk(module.tree)
              if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)}
    for cls in (node for node in ast.walk(module.tree) if isinstance(node, ast.ClassDef)):
        called = [function for function in cls.body
                  if isinstance(function, ast.FunctionDef) and function.name == "__call__"]
        if not called:
            continue
        function = called[-1]
        taken = {argument.arg for argument in (*function.args.args, *function.args.kwonlyargs)}
        declarations = [statement for statement in cls.body
                        if isinstance(statement, ast.Assign | ast.AnnAssign) and statement.value is not None
                        and any(_named(target) == "reads" for target in
                                (statement.targets if isinstance(statement, ast.Assign)
                                 else [statement.target]))]
        declared = _declared_reads(declarations[-1].value) if declarations else frozenset()
        where = declarations[-1] if declarations else cls
        if "call" in taken and declared is None:
            yield Finding(module.relative, where.lineno, where.col_offset + 1, "SLOP014",
                          f"{cls.name} declares reads as neither frozenset({{...}}) nor TOKENWISE")
            continue
        used = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)}
        # A tokenwise mixer is handed the call and has no use for it.
        ignored = {"call"} if declared == "tokenwise" else set()
        for name in sorted((taken & CALL_ARGUMENTS) - used - ignored):
            yield Finding(module.relative, function.lineno, function.col_offset + 1, "SLOP014",
                          f"{cls.name}.__call__ takes {name} and never reads it; read it or refuse it")
        if "call" not in taken or declared is None:
            continue
        read = {node.attr for node in ast.walk(cls) if isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Load)} & mixing
        if declared == "tokenwise":
            extra, unread, unknown = read, set(), set()
        else:
            extra, unread, unknown = read - declared, (declared & mixing) - loaded, declared - mixing
        line, col = where.lineno, where.col_offset + 1
        for name in sorted(extra):
            yield Finding(module.relative, line, col, "SLOP014",
                          f"{cls.name} reads {name} and does not declare it in reads")
        for name in sorted(unread):
            yield Finding(module.relative, line, col, "SLOP014",
                          f"{cls.name} declares it reads {name} and never does; its block would let "
                          f"a call that sets it through")
        if mixing and unknown:
            yield Finding(module.relative, line, col, "SLOP014",
                          f"{cls.name} reads {', '.join(sorted(unknown))}, which no call sets")


CONTRACT_RULES = (contracts, probes, names, imports, calls)
UNIVERSAL_RULES = (suppressions, swallowed, comments, size)


def check(module: Module, models: ModelIndex | None = None) -> Iterator[Finding]:
    """Every rule that applies to this file, in code order. SLOP010 reads
    `models`, the package's index, or this file's own without one."""
    rules = (*CONTRACT_RULES, *UNIVERSAL_RULES) if module.is_source else UNIVERSAL_RULES
    rules = (*rules, mocks, wording, skips) if module.relative.startswith("tests/") else (*rules, patches)
    findings = [finding for rule in rules for finding in rule(module)]
    if module.is_source:
        findings += boundaries(module, models or ModelIndex.build([module]))
    yield from sorted(findings, key=lambda finding: (finding.line, finding.col, finding.code))


def collect(roots: Sequence[str], checkout: Path = ROOT, package: str = "dew") -> Iterator[Module]:
    """Every Python file under the named roots of `checkout`, or the root
    itself when it names a file, skipping stub-only trees and the site's
    installed npm packages (node-gyp ships Python)."""
    for root in roots:
        named = checkout / root
        for path in [named] if named.is_file() else sorted(named.rglob("*.py")):
            relative = path.relative_to(checkout).as_posix()
            if "/stubs/" in f"/{relative}" or "/node_modules/" in f"/{relative}":
                continue
            yield Module(path, relative, path.read_text(), ast.parse(path.read_text()), package)


def default_roots(package: str) -> list[str]:
    """What the gate reads when it names no roots: the package's source and every script, the site's
    and the docs' generators and kernel packages among them."""
    return [f"src/{package}", "tests", "tools", "recipes", "examples", "site", "docs/assets", "kernels"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report low-evidence Python, `path:line:col: SLOPxxx`.")
    parser.add_argument("roots", nargs="*", help="directories or files under the checkout; by default the "
                        "package's source, tests, tools, recipes, examples, site, docs/assets and kernels")
    parser.add_argument("--root", type=Path, default=ROOT, help="the checkout (default: Dew's)")
    parser.add_argument("--package", default="dew", help="the package under src/ (default: dew)")
    parser.add_argument("--print-roots", action="store_true",
                        help="print the roots it reads by default; tools/pre-commit lints their staged files")
    args = parser.parse_args(argv)
    if args.print_roots:
        print(" ".join(default_roots(args.package)))
        return 0
    roots = args.roots or default_roots(args.package)
    counts: dict[str, int] = {}
    files: dict[str, set[str]] = {}
    models = ModelIndex.build(list(collect([f"src/{args.package}"], args.root.resolve(), args.package)))
    for module in collect(roots, args.root.resolve(), args.package):
        for finding in check(module, models):
            print(finding)
            counts[finding.code] = counts.get(finding.code, 0) + 1
            files.setdefault(finding.code, set()).add(finding.path)
    for code, count in sorted(counts.items()):
        print(f"{code}: {count} in {len(files[code])} files", file=sys.stderr)
    failures = sum(counts.values())
    print(f"{failures} findings", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
