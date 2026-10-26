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
- SLOP016 is about `src/<package>`: a choice keyed on which device runs (its
  hardware generation) lives in one table, `nn/kernels/generation.py`'s
  `TARGETS`; any other module reads that table's facts, a library's support
  rule the compute capability, and a launch config the autotuner.
- SLOP018 and SLOP019 are about `src/<package>`, the code a user's process
  runs: it sleeps, polls and retries only where it deals with what leaves
  the process, and it detects no feature by the exception its absence raises.
- SLOP020 is about `tests` and `tools`: every accuracy comparison goes through
  the one assertion of the gate (`tests/reference_error.py`), so a
  float compared within a tolerance anywhere else is a finding. The files that
  still hold such comparisons are the inventory
  (`tools/tolerance_inventory.json`), each at its count and lane, and it only
  shrinks.
- SLOP021 is about every file: a private attribute of JAX's own objects
  (`JAX_PRIVATE`) is read nowhere (Dew reads no other library's private
  attributes), whose public way the finding names.

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
SLOP018 reads one module: a function deals with what leaves the process where
it names such a module (`EXTERNAL`) or calls a function of the module, or a
method reached from `self`, that does; work handed to a future is another's.
It reads a loop over a range whose body reads the number it counts anywhere
but where a failure is handled as a scan by index, not as attempts, so a retry
that reads its attempt number outside its handler (returns or logs it) is not
reported.
SLOP016 follows the device's generation through the names one file binds, in
statement order: both arms of a branch in turn and a loop's body once, so a
name an arm rebinds counts as rebound after it.
SLOP004's isinstance half fires only when the annotation it needs is written in
the same scope; a value whose type arrives from another module is not narrowed
by this checker and is not reported. SLOP006 walks the handler body it can see,
so an exception handed to a function that re-raises elsewhere reads as reported.
"""

from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import io
import json
import re
import subprocess
import sys
import tokenize
from collections.abc import Iterable, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lint_scopes as scopes
import lint_tolerances as tolerance_rules
from forked import forked_map

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
                 "mark_as_parsed", "register_event_listener", "filterwarnings", "simplefilter")


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
    comprehended = {id(target) for node in ast.walk(module.tree) if isinstance(node, scopes.COMPREHENSIONS)
                    for target in scopes.targets(node)}
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


JAX_PRIVATE = {
    "_cache_size": "count the programs a jitted function needs with dew.telemetry.instrumentation.Compiles",
}
"""The private attributes of JAX's objects Dew has read, each with the public way to what it was read
for."""


def jax_private(module: Module) -> Iterator[Finding]:
    """SLOP021: a read of a private attribute of JAX's objects (`JAX_PRIVATE`)."""
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Attribute) and node.attr in JAX_PRIVATE:
            yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP021",
                          f"reads JAX's private {node.attr}; {JAX_PRIVATE[node.attr]}")


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


CAPTURED = {"readouterr", "text", "messages", "records"}
"""What pytest's capsys, capfd and caplog fixtures hand a test of the output it captured."""


def _captured(node: ast.expr, names: Set[str]) -> bool:
    """Is `node` output a test captured: `capsys.readouterr()`'s or capfd's, its `.out` or `.err`,
    caplog's `.text`, `.messages` or `.records`, or a name of `names`, bound to one of them, or its
    `.out` or `.err`?"""
    if isinstance(node, ast.Attribute) and node.attr in ("out", "err"):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Call):
        node = node.func
    return (isinstance(node, ast.Attribute) and node.attr in CAPTURED
            and isinstance(node.value, ast.Name) and node.value.id in ("capsys", "capfd", "caplog"))


def _held(tree: ast.Module) -> dict[int, frozenset[str]]:
    """The names bound to captured output (`_captured`) where each node of a function's own scope is:
    a name the function binds to it, or unpacks it into, until it binds that name again to anything
    else (a parse of it, say); added to, it stays captured. An item or attribute assigned binds no
    name."""
    held: dict[int, frozenset[str]] = {}
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        # A binding takes effect once its statement's value has been read: at the statement's end.
        events: list[tuple[int, int, int, ast.AST]] = []
        for node in scopes.own(function):
            if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
                events.append((node.end_lineno or node.lineno, node.end_col_offset or 0, id(node), node))
            elif isinstance(node, ast.Compare | ast.Call):
                events.append((node.lineno, node.col_offset, id(node), node))
        names: set[str] = set()
        for *_, node in sorted(events, key=lambda event: event[:3]):
            if not isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
                held[id(node)] = frozenset(names)
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            bound = {name.id for target in targets for name in ast.walk(target)
                     if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)}
            if node.value is not None and _captured(node.value, names):
                names |= bound
            elif not isinstance(node, ast.AugAssign):
                names -= bound
    return held


def _spelled_in_captured(node: ast.AST, names: Set[str]) -> bool:
    """Does `node` look for a sentence in captured output: a string the source spells, compared
    with `in` against it, or counted in it?"""
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.In | ast.NotIn):
        return (isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)
                and _captured(node.comparators[0], names))
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "count"
            and _captured(node.func.value, names) and bool(node.args)
            and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str))


def _errors(function: ast.FunctionDef | ast.AsyncFunctionDef,
            raises: Set[int]) -> tuple[frozenset[str], frozenset[str]]:
    """The names a test binds to an error it caught: an `except` clause's, and the target of a
    block whose context is one of the `pytest.raises` calls `raises` holds by id, whose `.value`
    the error is."""
    nodes = list(scopes.own(function))
    caught = {node.name for node in nodes if isinstance(node, ast.ExceptHandler) and node.name}
    raised = {item.optional_vars.id for node in nodes if isinstance(node, ast.With | ast.AsyncWith)
              for item in node.items if isinstance(item.optional_vars, ast.Name)
              and id(item.context_expr) in raises}
    return frozenset(caught), frozenset(raised)


def _worded(node: ast.AST, caught: Set[str], raised: Set[str]) -> bool:
    """Does `node` look for a sentence the source spells in a process's `stdout` or `stderr`, by
    `in`, `startswith`, `endswith` or `count`, or match a caught error's text (`str` or `repr` of
    it) so or by `==`? A process's whole output compared, or parsed first, is the value it reports."""
    def output(expr: ast.expr) -> bool:
        return isinstance(expr, ast.Attribute) and expr.attr in ("stdout", "stderr")

    def error(expr: ast.expr) -> bool:
        if not (isinstance(expr, ast.Call) and _named(expr.func) in ("str", "repr") and len(expr.args) == 1):
            return False
        held = expr.args[0]
        return (isinstance(held, ast.Name) and held.id in caught) or (
            isinstance(held, ast.Attribute) and held.attr == "value" and isinstance(held.value, ast.Name)
            and held.value.id in raised)

    def spelled(expr: ast.expr) -> bool:
        return isinstance(expr, ast.Constant) and isinstance(expr.value, str)

    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left, right = node.left, node.comparators[0]
        if isinstance(node.ops[0], ast.In | ast.NotIn):
            return spelled(left) and (output(right) or error(right))
        return isinstance(node.ops[0], ast.Eq | ast.NotEq) and (
            (spelled(left) and error(right)) or (spelled(right) and error(left)))
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("startswith", "endswith", "count")
            and (output(node.func.value) or error(node.func.value))
            and bool(node.args) and spelled(node.args[0]))


def wording(module: Module) -> Iterator[Finding]:
    """SLOP011: a test that checks wording or source text instead of behaviour: an error's or a
    warning's text matched, a sentence looked for in captured output, in a caught error's text or
    in a process's output, or source text read. A name is read through the imports of its own
    scope, so `pytest.raises` imported under another name is `pytest.raises`."""
    named = list(scopes.scoped(module.tree.body, {}, _parent(module)))
    raises = {id(node) for node, table in named
              if isinstance(node, ast.Call) and scopes.resolved(node.func, table) == "pytest.raises"}
    for function in ast.walk(module.tree):
        if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            caught, raised = _errors(function, raises)
            for node in scopes.own(function):
                if _worded(node, caught, raised):
                    yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP011",
                                  "looks for a sentence in an error's or a process's text; assert its "
                                  "type, its exit status or the value it reports")
    held = _held(module.tree)
    for node, table in named:
        if isinstance(node, ast.expr) and _spelled_in_captured(node, held.get(id(node), frozenset())):
            yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP011",
                          "looks for a sentence in captured output; assert the state it reports")
        if isinstance(node, ast.Call):
            called = scopes.resolved(node.func, table)
            if called in ("pytest.raises", "pytest.warns") and any(keyword.arg == "match"
                                                                   for keyword in node.keywords):
                raised = "an error" if called == "pytest.raises" else "a warning"
                yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP011",
                              f"matches {raised}'s wording; assert that it is raised, not its text")
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
    """The statements an import executes: every block but the script guard's body
    (its `else` still runs). A definition among them runs what `_executed` says:
    its decorators and defaults, and a class's bases, keywords and body."""
    for node in body:
        if isinstance(node, ast.If):
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
    for node in _executed(_module_level(module.tree.body)):
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


# Where the device's hardware generation comes from, and the one module that keys choices on it.
GENERATION = "{package}.nn.kernels.generation"


@dataclass
class _Hardware:
    """What one scope's names hold of the device's hardware, as SLOP016 reads it: the names bound to
    its generation and those bound to its target (`gpu_target()`), and how each node resolves names."""

    owner: str
    tables: Mapping[int, Mapping[str, str]]
    generations: frozenset[str] = frozenset()
    targets: frozenset[str] = frozenset()

    def called(self, node: ast.expr, name: str) -> bool:
        return (isinstance(node, ast.Call)
                and scopes.resolved(node.func, self.tables.get(id(node), {})) == f"{self.owner}.{name}")

    def generation(self, node: ast.expr) -> bool:
        """Is `node` the device's generation: `device_generation()`, a name bound to it, or the
        `.generation` of `gpu_target()` or of a name bound to that?"""
        if isinstance(node, ast.Name):
            return node.id in self.generations
        if isinstance(node, ast.Attribute) and node.attr == "generation":
            return ((isinstance(node.value, ast.Name) and node.value.id in self.targets)
                    or self.called(node.value, "gpu_target"))
        return self.called(node, "device_generation")

    def assigned(self, targets: Sequence[ast.expr], value: ast.expr | None) -> _Hardware:
        """These names after `targets` take `value`: one name taking the generation or the target holds
        it, and every other name they bind no longer holds either."""
        names = [name.id for target in targets for name in ast.walk(target)
                 if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)]
        rest = self.without(names)
        if len(targets) != 1 or not isinstance(targets[0], ast.Name) or value is None:
            return rest
        if self.generation(value):
            return replace(rest, generations=rest.generations | {targets[0].id})
        if self.called(value, "gpu_target"):
            return replace(rest, targets=rest.targets | {targets[0].id})
        return rest

    def without(self, names: Iterable[str]) -> _Hardware:
        """These names but `names`, which hold something else here."""
        held = set(names)
        return replace(self, generations=self.generations - held, targets=self.targets - held)

    def inside(self, scope: ast.AST) -> _Hardware:
        """These names as a nested scope sees them: less its parameters."""
        if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            return self.without(argument.arg for argument in scopes.arguments(scope))
        return self

    def keyed(self, node: ast.AST) -> bool:
        """Does `node` choose by the generation: compare it with one the source spells, match its
        prefix or a spelled case, cut it up, or key a table with it?"""
        if isinstance(node, ast.Compare):
            operands = (node.left, *node.comparators)
            return any(map(self.generation, operands)) and any(map(_spelled, operands))
        if isinstance(node, ast.Subscript):
            return self.generation(node.value) or self.generation(node.slice)
        if isinstance(node, ast.Match):
            spelled = any(_spelled_pattern(case.pattern) for case in node.cases)
            return self.generation(node.subject) and spelled
        if not isinstance(node, ast.Call):
            return False
        if isinstance(node.func, ast.Attribute):
            return ((node.func.attr in ("startswith", "endswith") and self.generation(node.func.value))
                    or (node.func.attr == "get" and bool(node.args) and self.generation(node.args[0])))
        builtin = scopes.resolved(node.func, self.tables.get(id(node), {})) == "builtins.getattr"
        return builtin and len(node.args) > 1 and self.generation(node.args[1])


def _spelled(node: ast.expr) -> bool:
    """Is `node` a string written in the source, or a collection of them?"""
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return bool(node.elts) and all(map(_spelled, node.elts))
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _spelled_pattern(pattern: ast.pattern) -> bool:
    """Is a match case's `pattern` a string the source spells, or several?"""
    if isinstance(pattern, ast.MatchOr):
        return any(map(_spelled_pattern, pattern.patterns))
    return isinstance(pattern, ast.MatchValue) and _spelled(pattern.value)


def _eager(node: ast.AST) -> Iterator[ast.AST]:
    """`node` and every node under it that runs where it stands: not a lambda's body."""
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(node, ast.Lambda) and child is node.body:
            continue
        yield from _eager(child)


def _scope_choices(body: Sequence[ast.stmt], hardware: _Hardware, closure: _Hardware | None,
                   found: list[ast.expr | ast.stmt]) -> _Hardware:
    """Read a scope's `body` in statement order (`_choices`) from `hardware`, its names on entry. A
    function or the module (`closure` None) hands the functions in it its names as they stand once it
    has bound them all, as a closure reads them; a class body hands on its enclosing scope's, as its
    namespace reaches no function in it."""
    if closure is None:
        closure = hardware
        for statement in body:
            closure = _choices(statement, closure, closure, None)
    for statement in body:
        hardware = _choices(statement, hardware, closure, found)
    return hardware


def _choices(node: ast.AST, hardware: _Hardware, closure: _Hardware,
             found: list[ast.expr | ast.stmt] | None) -> _Hardware:
    """Append to `found` each node under `node` that chooses by the device's generation (`keyed`),
    reading `hardware`'s names; return them once `node` has bound its own. A value is read before its
    targets take it, and a comprehension's first iterable before its targets hide a name inside it.
    With `found` None, only the names: nested scopes are not read."""
    def read(child: ast.AST | None, names: _Hardware, inherited: _Hardware = closure) -> _Hardware:
        return names if child is None else _choices(child, names, inherited, found)

    if isinstance(node, ast.Lambda):
        for child in (*node.args.defaults, *node.args.kw_defaults):
            hardware = read(child, hardware)
        if found is not None:
            read(node.body, closure.inside(node), closure.inside(node))
        return hardware
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        for child in (*node.decorator_list, *node.args.defaults, *node.args.kw_defaults):
            hardware = read(child, hardware)
        if found is not None:
            _scope_choices(node.body, closure.inside(node), None, found)
        return hardware.without([node.name])
    if isinstance(node, ast.ClassDef):
        for child in (*node.decorator_list, *node.bases, *node.keywords):
            hardware = read(child, hardware)
        if found is not None:
            _scope_choices(node.body, hardware, closure, found)
        return hardware.without([node.name])
    if isinstance(node, scopes.COMPREHENSIONS):
        first = node.generators[0]
        hardware = read(first.iter, hardware)
        names = [name.id for name in scopes.targets(node)]
        inner, inherited = hardware.without(names), closure.without(names)
        for generator in node.generators:
            for child in (*([] if generator is first else [generator.iter]), *generator.ifs):
                inner = read(child, inner, inherited)
        for child in ((node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,)):
            inner = read(child, inner, inherited)
        if isinstance(node, ast.GeneratorExp):
            # Its body runs later, as it is consumed.
            return hardware
        # An assignment expression in it binds the enclosing scope's name; one in a lambda, the lambda's.
        walrus = {expression.target.id for expression in _eager(node)
                  if isinstance(expression, ast.NamedExpr)}
        outer = hardware.without(walrus)
        return replace(outer, generations=outer.generations | (inner.generations & walrus),
                       targets=outer.targets | (inner.targets & walrus))
    if found is not None and isinstance(node, ast.expr | ast.stmt) and hardware.keyed(node):
        found.append(node)
    if isinstance(node, ast.Assign | ast.AnnAssign | ast.NamedExpr):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        hardware = read(node.value, hardware)
        for target in targets:
            # Read for the keys a target holds; the names it binds are `assigned`'s.
            read(target, hardware)
        return hardware.assigned(targets, node.value)
    for child in ast.iter_child_nodes(node):
        hardware = read(child, hardware)
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
        return hardware.without([node.id])
    if isinstance(node, ast.MatchAs | ast.MatchStar) and node.name is not None:
        return hardware.without([node.name])
    return hardware


def authorities(module: Module) -> Iterator[Finding]:
    """SLOP016: a choice keyed on which device runs, outside the table that owns them: the device's
    generation compared with or prefix-matched against one the source spells, matched against a
    spelled case, cut up, or used as a table's key or attribute name. Comparing it with a recorded
    one, the generation a measurement was taken on, is a measurement's provenance and not reported."""
    owner = GENERATION.format(package=module.package)
    if module.relative == f"src/{owner.replace('.', '/')}.py":
        return
    builtins = {"getattr": "builtins.getattr"}
    tables = {id(node): table for node, table in scopes.scoped(module.tree.body, builtins, _parent(module))}
    found: list[ast.expr | ast.stmt] = []
    _scope_choices(module.tree.body, _Hardware(owner, tables), None, found)
    for node in found:
        yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP016",
                      "chooses by the device's generation; read the target table's fact (`gpu_target()`), "
                      "a library's rule the compute capability (`capability`), a launch config the autotuner")


EXTERNAL = ("subprocess.", "socket.", "ssl.", "select.", "urllib.", "http.", "requests.", "httpx.",
            "aiohttp.", "grpc.", "huggingface_hub.", "fsspec.", "gcsfs.", "google.", "boto3.", "botocore.",
            "filelock.", "multiprocessing.", "concurrent.futures.ProcessPoolExecutor.", "os.kill.",
            "os.killpg.", "os.waitpid.", "os.waitid.", "os.fork.", "os.posix_spawn.", "jax.distributed.",
            "dew.cleanup.signal_group.", "dew.cleanup.exit_code.", "dew.cleanup.groups_ended.",
            "dew.cleanup.end_group.", "dew.cleanup.process_group.",
            "IPython.display.display.", "IPython.display.update_display.", "IPython.display.clear_output.")
"""What leaves the process: a subprocess, the network, a remote service, another process (a
notebook's frontend, which IPython's display functions publish to over the kernel's socket; its
representations, such as `HTML`, are built in the process; a process group `dew.cleanup` signals)."""


WAITS = {"wait", "wait_for", "join", "get", "put", "acquire"}
"""The methods that block on in-process synchronization (an event, a condition, a lock, a queue, a
thread) until a timeout they are given. A future's (`result`, `concurrent.futures.wait`) is not
among them: its work is handed elsewhere, a thread, a process or a service, whose deadline it is."""
FUTURE_WAITS = {"concurrent.futures.wait", "asyncio.wait", "asyncio.wait_for"}


def _waiting(node: ast.AST, table: Mapping[str, str]) -> str:
    """What `node` waits on: "a sleep", or "a timed wait", a blocking call (`WAITS`) given a
    timeout (one of no time asks without waiting); "" else."""
    if not isinstance(node, ast.Call):
        return ""
    called = scopes.resolved(node.func, table)
    if called in ("time.sleep", "asyncio.sleep"):
        return "a sleep"
    method = node.func.attr if isinstance(node.func, ast.Attribute) else ""
    if method not in WAITS or called in FUTURE_WAITS:
        return ""
    # A positional timeout is read where the method is unambiguous (a string's join is not a wait).
    positional = node.args[:1] if method == "wait" else node.args[1:2] if method == "wait_for" else []
    timeouts = [*positional, *(keyword.value for keyword in node.keywords if keyword.arg == "timeout")]
    if any(not (isinstance(value, ast.Constant) and value.value in (None, 0)) for value in timeouts):
        return "a timed wait"
    return ""


def _evaluated(definition: ast.AST) -> list[ast.AST]:
    """The parts of a definition evaluated where it is defined: its decorators and defaults, and a
    class's bases and keywords. An annotation is a type, not an operation (postponed under
    `from __future__ import annotations`, and lazy since Python 3.14), so it is not among them."""
    if isinstance(definition, ast.ClassDef):
        return [*definition.decorator_list, *definition.bases, *definition.keywords]
    if isinstance(definition, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
        signature = definition.args
        defaults: list[ast.AST] = [*signature.defaults,
                                   *(item for item in signature.kw_defaults if item is not None)]
        return defaults if isinstance(definition, ast.Lambda) else [*definition.decorator_list, *defaults]
    return []


def _comprehended(comprehension: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
                  ) -> list[ast.AST]:
    """What a comprehension evaluates in its own scope: every part but its first iterable."""
    first, *rest = comprehension.generators
    elements = [comprehension.key, comprehension.value] if isinstance(comprehension, ast.DictComp) else [
        comprehension.elt]
    return [*first.ifs, *(part for generator in rest for part in (generator.iter, *generator.ifs)), *elements]


def _executed_in(nodes: Iterable[ast.AST], scope: ast.AST) -> Iterator[tuple[ast.AST, ast.AST]]:
    """Every node that runs where `nodes` run, in `scope`, with the scope its names resolve in: a nested
    function's or lambda's evaluated parts (`_evaluated`) but not its body, which runs only where it is
    called, and a class's body, which runs where the class is defined, in the class's own namespace."""
    for node in nodes:
        yield node, scope
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            yield from _executed_in(_evaluated(node), scope)
        elif isinstance(node, ast.ClassDef):
            yield from _executed_in(_evaluated(node), scope)
            yield from _executed_in(node.body, node)
        elif isinstance(node, ast.For) and isinstance(node.iter, ast.GeneratorExp):
            # The loop consumes the generator it iterates as it goes round: the generator's body runs here.
            yield node.iter, scope
            yield from _executed_in([node.iter.generators[0].iter], scope)
            yield from _executed_in(_comprehended(node.iter), node.iter)
            yield from _executed_in([node.target, *node.body, *node.orelse], scope)
        elif isinstance(node, scopes.COMPREHENSIONS):
            # Its first iterable is evaluated where it stands; the rest runs in its own scope, where its
            # targets bind: at once for a list, a set or a dict, and only as it is consumed for a generator.
            yield from _executed_in([node.generators[0].iter], scope)
            if not isinstance(node, ast.GeneratorExp):
                yield from _executed_in(_comprehended(node), node)
        else:
            yield from _executed_in(ast.iter_child_nodes(node), scope)


def _executed(nodes: Iterable[ast.AST]) -> Iterator[ast.AST]:
    """Every node that runs where `nodes` run (`_executed_in`)."""
    for node, _ in _executed_in(nodes, ast.Module(body=[], type_ignores=[])):
        yield node


def _outcomes(statements: Sequence[ast.stmt]) -> frozenset[str]:
    """How running `statements` can end, by Python's control flow: "next", off their end, or by a
    "return", a "raise", or a "break" or "continue" of the loop they sit in. A branch, a `with`, a
    `try` and a `match` end as any of their paths can, a `finally` that completes resumes the exit
    it interrupted, a nested loop keeps its own `break` and `continue`, and a class's body runs where
    it is defined; a function's body runs only where it is called. An exception a call raises is
    not read."""
    ends = frozenset({"next"})
    for statement in statements:
        if "next" not in ends:
            break
        ends = ends - {"next"} | _ending(statement)
    return ends


def _then(ends: frozenset[str], statements: Sequence[ast.stmt]) -> frozenset[str]:
    """`ends` with the path that runs off their end ("next") going on into `statements`."""
    return ends - {"next"} | (_outcomes(statements) if "next" in ends else frozenset())


def _finally(pending: frozenset[str], final: Sequence[ast.stmt]) -> frozenset[str]:
    """How a `try` ends whose paths end as `pending` before its `final` body runs: a `final` body
    that completes resumes each of them, and one that leaves replaces them."""
    ends = _outcomes(final)
    return ends - {"next"} | (pending if "next" in ends else frozenset())


_JUMPS = {ast.Return: "return", ast.Raise: "raise", ast.Break: "break", ast.Continue: "continue"}


def _ending(statement: ast.stmt) -> frozenset[str]:
    """How running one statement can end (`_outcomes`)."""
    if type(statement) in _JUMPS:
        return frozenset({_JUMPS[type(statement)]})
    match statement:
        case ast.If():
            return _outcomes(statement.body) | _outcomes(statement.orelse)
        case ast.With() | ast.AsyncWith() | ast.ClassDef():
            return _outcomes(statement.body)
        case ast.Try() | ast.TryStar():
            pending = _then(_outcomes(statement.body), statement.orelse).union(
                *(_outcomes(handler.body) for handler in statement.handlers))
            return _finally(pending, statement.finalbody)
        case ast.Match():
            done = frozenset[str]().union(*(_outcomes(case.body) for case in statement.cases))
            caught = any(case.guard is None and _irrefutable(case.pattern) for case in statement.cases)
            return done if caught else done | {"next"}
        case ast.For() | ast.AsyncFor() | ast.While():
            body = _outcomes(statement.body)
            forever = (isinstance(statement, ast.While) and isinstance(statement.test, ast.Constant)
                       and bool(statement.test.value))
            ran_out = frozenset[str]() if forever else _outcomes(statement.orelse)
            return body - {"next", "break", "continue"} | ran_out | ({"next"} if "break" in body else set())
    return frozenset({"next"})


def _irrefutable(pattern: ast.pattern) -> bool:
    """Whether `pattern` matches every subject: a wildcard or a capture, also under `as` or as one
    alternative of an `|`."""
    match pattern:
        case ast.MatchAs(pattern=None):
            return True
        case ast.MatchAs(pattern=inner) if inner is not None:
            return _irrefutable(inner)
        case ast.MatchOr(patterns=alternatives):
            return any(map(_irrefutable, alternatives))
    return False


def _attempts(statements: Sequence[ast.stmt], after: Sequence[Sequence[ast.stmt]]
              ) -> Iterator[tuple[ast.Try | ast.TryStar, list[Sequence[ast.stmt]]]]:
    """Each `try` that `statements` run at their loop's own level, with the statements that run after
    it, in turn, on the way to the loop's next round (`after` the ones after `statements`): through
    a `with`, a branch or a `match` case, not into a nested loop or a definition, whose exits are
    their own."""
    for index, statement in enumerate(statements):
        rest = [statements[index + 1:], *after]
        if isinstance(statement, ast.Try | ast.TryStar):
            yield statement, rest
        elif isinstance(statement, ast.With | ast.AsyncWith):
            yield from _attempts(statement.body, rest)
        elif isinstance(statement, ast.If):
            yield from _attempts(statement.body, rest)
            yield from _attempts(statement.orelse, rest)
        elif isinstance(statement, ast.Match):
            for case in statement.cases:
                yield from _attempts(case.body, rest)


def _retrying(loop: ast.For | ast.While) -> bool:
    """Whether `loop` retries one piece of work: a `while`, or a `for` over a `range` whose body
    reads the number it counts only where a failure is handled (attempts, counted to stop; read
    anywhere else, it can choose the work, a scan by index), with a `try` at its own level
    (`_attempts`) that leaves the loop on some path where the work succeeds and goes round again on
    some path where it fails, each path followed through the `try`'s `finally` and every statement
    after it in the loop's body (`_outcomes`)."""
    attempts = list(_attempts(loop.body, []))
    if isinstance(loop, ast.For):
        counted = {name.id for name in ast.walk(loop.target) if isinstance(name, ast.Name)}
        handling = {id(node) for attempt, _ in attempts for handler in attempt.handlers
                    for node in _executed(handler.body)}
        if not (isinstance(loop.iter, ast.Call) and _named(loop.iter.func) == "range") or any(
                isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in counted
                and id(node) not in handling
                for node in _executed(loop.body)):
            return False
    for attempt, rest in attempts:
        def onward(ends: frozenset[str], attempt: ast.Try | ast.TryStar = attempt,
                   rest: list[Sequence[ast.stmt]] = rest) -> frozenset[str]:
            ends = _finally(ends, attempt.finalbody)
            for statements in rest:
                ends = _then(ends, statements)
            return ends

        succeeds = {"break", "return"} & onward(_then(_outcomes(attempt.body), attempt.orelse))
        again = any({"next", "continue"} & onward(_outcomes(handler.body)) for handler in attempt.handlers)
        if succeeds and again:
            return True
    return False


class _Calls:
    """The module's functions a call reaches, read where it is made, as one module's lexical scopes
    give them: a bare name through the scopes that enclose the call (a class body's names reach none
    of its methods) to the definition that binds it there; `self.f` or `cls.f`, the receiver of the
    method the call is in or closes over, to `f` in the method resolution order of that method's
    class among the module's classes; and `self.held.f`, through an attribute of another object, to
    any method `f` of the module's classes. Calls through other modules, callables passed in and
    attributes set at run time are not followed."""

    def __init__(self, tree: ast.Module):
        self.enclosing: dict[int, ast.AST | None] = {}
        self.bindings: dict[int, dict[str, ast.AST | None]] = {}
        self.named: dict[str, list[ast.AST]] = {}
        self._index(tree, None)

    def _index(self, scope: ast.AST, enclosing: ast.AST | None) -> None:
        self.enclosing[id(scope)] = enclosing
        own = list(scopes.own(scope))
        parameters = (scopes.arguments(scope)
                      if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) else [])
        # A name one definition alone binds is that definition; bound otherwise, or twice, no one's.
        definitions = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        self.bindings[id(scope)] = {name: by[0] if len(by) == 1 and isinstance(by[0], definitions) else None
                                    for name, by in scopes.binders(own, parameters).items()}
        for node in own:
            if isinstance(node, scopes.COMPREHENSIONS):
                self._comprehension(node, scope)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and isinstance(scope, ast.ClassDef):
                self.named.setdefault(node.name, []).append(node)
            if isinstance(node, scopes.SCOPES):
                self._index(node, scope)

    def _comprehension(self, node: ast.AST, enclosing: ast.AST) -> None:
        """Index a comprehension, which `_index` meets in the scope that holds it, as a scope of its own: its
        targets bind there, and a comprehension it holds encloses in it."""
        if id(node) in self.bindings:
            return
        assert isinstance(node, scopes.COMPREHENSIONS)
        self.enclosing[id(node)] = enclosing
        self.bindings[id(node)] = {name.id: None for name in scopes.targets(node)}
        for part in _comprehended(node):
            for inner in ast.walk(part):
                if isinstance(inner, scopes.COMPREHENSIONS):
                    self._comprehension(inner, node)

    def _lookup(self, scope: ast.AST | None, name: str) -> ast.AST | None:
        """What `name` stands for, read in `scope`: its own binding, else the enclosing scopes'
        (a class's names reach no scope inside it)."""
        start = scope
        while scope is not None:
            bound = self.bindings.get(id(scope), {})
            if name in bound and (scope is start or not isinstance(scope, ast.ClassDef)):
                return bound[name]
            scope = self.enclosing.get(id(scope))
        return None

    def _order(self, owner: ast.ClassDef, seen: frozenset[int] = frozenset()) -> list[ast.ClassDef]:
        """`owner`'s method resolution order among the module's classes (C3, as Python computes it),
        a base the module does not define ending its line."""
        if id(owner) in seen:
            return [owner]
        bases = [base for base in (self._lookup(self.enclosing.get(id(owner)), node.id)
                                   for node in owner.bases if isinstance(node, ast.Name))
                 if isinstance(base, ast.ClassDef)]
        lines = [self._order(base, seen | {id(owner)}) for base in bases] + [bases]
        order = [owner]
        while any(lines):
            for line in lines:
                head = line[0] if line else None
                if head is not None and not any(head in other[1:] for other in lines):
                    break
            else:
                return order  # No consistent order: Python refuses the class; nothing more is read.
            order.append(head)
            lines = [line[1:] if line and line[0] is head else line for line in lines]
        return order

    def _method(self, scope: ast.AST, receiver: str, name: str) -> ast.AST | None:
        """The method `name` the receiver `receiver`, read in `scope`, reaches: the receiver of the
        method that scope is in or closes over, to the first class in that method's class's resolution
        order that binds `name`. A scope on the way that binds the receiver itself (a comprehension's
        target, a closure's own) makes it another object, whose methods are not followed."""
        current: ast.AST | None = scope
        while current is not None:
            owner = self.enclosing.get(id(current))
            if isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                parameters = [argument.arg for argument in scopes.arguments(current)]
                if parameters[:1] == [receiver] and isinstance(owner, ast.ClassDef):
                    for cls in self._order(owner):
                        if name in self.bindings[id(cls)]:
                            return self.bindings[id(cls)][name]
                    return None
            if receiver in self.bindings.get(id(current), {}):
                return None
            current = owner
        return None

    def reached(self, scope: ast.AST, call: ast.Call) -> list[ast.AST]:
        """The definitions `call`, made in `scope`, reaches."""
        if isinstance(call.func, ast.Name):
            found = self._lookup(scope, call.func.id)
            return [found] if isinstance(found, ast.FunctionDef | ast.AsyncFunctionDef) else []
        if not isinstance(call.func, ast.Attribute):
            return []
        through, root = 0, call.func.value
        while isinstance(root, ast.Attribute):
            through, root = through + 1, root.value
        if not isinstance(root, ast.Name) or root.id not in ("self", "cls"):
            return []
        if through:
            return self.named.get(call.func.attr, [])
        method = self._method(scope, root.id, call.func.attr)
        return [method] if isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef) else []


def _leaves(function: ast.FunctionDef | ast.AsyncFunctionDef, tables: Mapping[int, Mapping[str, str]],
            calls: _Calls, seen: set[int]) -> bool:
    """Whether `function` deals with what leaves the process (`EXTERNAL`): what it runs names such
    a module, or calls a function of the module (`_Calls`) that does."""
    seen.add(id(function))
    for node, scope in _executed_in(function.body, function):
        if isinstance(node, ast.Name | ast.Attribute):
            path = scopes.resolved(node, tables.get(id(node), {}))
            if path and f"{path}.".startswith(EXTERNAL):
                return True
        if isinstance(node, ast.Call):
            for helper in calls.reached(scope, node):
                if (isinstance(helper, ast.FunctionDef | ast.AsyncFunctionDef) and id(helper) not in seen
                        and _leaves(helper, tables, calls, seen)):
                    return True
    return False


def waits(module: Module) -> Iterator[Finding]:
    """SLOP018: a sleep, a poll (a timed wait round a loop) or a retry where nothing leaves the
    process: one inside a function that deals with no subprocess, network, remote service or other
    process (`EXTERNAL`), itself or through the module's functions and methods it calls. In-process
    work is ordered or woken, not waited on; a timed wait made once is a deadline."""
    calls = _Calls(module.tree)
    tables = {id(node): table for node, table in scopes.scoped(module.tree.body, {}, _parent(module))}
    for function in ast.walk(module.tree):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        external = None
        # What runs again and again: a loop's body, a `while`'s test and a generator a `for` consumes,
        # and an eager comprehension's body.
        repeated: list[ast.AST] = []
        for node in _executed(function.body):
            if isinstance(node, ast.For | ast.While):
                repeated += [*node.body, *([node.test] if isinstance(node, ast.While) else []),
                             *(_comprehended(node.iter) if isinstance(node, ast.For)
                               and isinstance(node.iter, ast.GeneratorExp) else [])]
            elif isinstance(node, ast.ListComp | ast.SetComp | ast.DictComp):
                repeated += _comprehended(node)
        looped = {id(inner) for inner in _executed(repeated)}
        for node in _executed(function.body):
            what = _waiting(node, tables.get(id(node), {}))
            if what == "a timed wait":
                # Once, it is a deadline; round a loop, a poll.
                what = "a poll" if id(node) in looped else ""
            if not what and isinstance(node, ast.For | ast.While) and _retrying(node):
                what = "a retry"
            # A wait is a call and a retry a loop: both have a place in the source.
            if not what or not isinstance(node, ast.expr | ast.stmt):
                continue
            if external is None:
                external = _leaves(function, tables, calls, set())
            if not external:
                yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP018",
                              f"{what} where nothing leaves the process; order the work or wake the waiter")


DETECTED = {"ImportError", "ModuleNotFoundError", "AttributeError"}


def detections(module: Module) -> Iterator[Finding]:
    """SLOP019: a feature detected by the exception its absence raises: a handler of ImportError,
    ModuleNotFoundError or AttributeError, by any name it is imported as, that carries on along any
    path instead of raising (`_outcomes`: a raise in a function it only defines does not count)."""
    for node, table in scopes.scoped(module.tree.body, {}, _parent(module)):
        if not isinstance(node, ast.ExceptHandler) or node.type is None:
            continue
        caught = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
        kinds = [scopes.resolved(kind, table) or _named(kind) for kind in caught]
        detected = any(kind.removeprefix("builtins.") in DETECTED for kind in kinds)
        if detected and _outcomes(node.body) != {"raise"}:
            yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP019",
                          f"detects a feature by `except {'/'.join(kinds)}`; require it, or ask for it by "
                          "the version that has it")


GATE = frozenset({"tests/reference_error.py", "tests/op_certificates.py", "tests/certified.py",
                  "tests/reference_graph.py", "tests/reference_trees.py", "tests/residual_orders.py",
                  "tests/selection_margins.py", "tests/test_selection_margins.py",
                  "tests/test_reference_error.py", "tests/test_op_certificates.py",
                  "tests/test_reference_orders.py", "tests/test_reference_graph.py",
                  "tests/test_gate_calibration.py", "tests/test_float64_terms.py",
                  "tests/sampling_gate.py", "tests/test_sampling_gate.py",
                  "tests/learning_gate.py", "tests/test_learning_gate.py"})
"""The gate's modules and the tests of the gate itself, the one place a float is compared within a
bound: every other comparison goes through its assertion; the sampling gate (tests/sampling_gate.py)
compares a statistic's rarity with its level, and the learning gate (tests/learning_gate.py) a claim's t
over its runs with its threshold, each its verdict. The selection margins
(tests/selection_margins.py) are the gate's preconditions for check 1 on a model that selects: their
bounds hold a draw's discrete choices to float64's, and live with it."""
INVENTORY = "tools/tolerance_inventory.json"
RULE = "tools/lint_tolerances.py"
"""SLOP020's reading of a file (`tolerance_rules`), whose change may grow what the inventory holds."""


def _tolerance_comparisons(module: Module) -> Iterator[tuple[ast.AST, str]]:
    """SLOP020's comparisons in a module (`tolerance_rules.comparisons`), its imports read from its own
    package and its calls resolved as SLOP018 resolves them. A function imported by a bare module name
    from a file beside it, as tests import their support (`from test_pipeline_core import delivered`),
    is read as that file's own reading proves it (`_support_functions`), as the same function defined in
    the test would be."""
    parent = tuple(_parent(module))
    tables = {id(node): table for node, table in scopes.scoped(module.tree.body, {}, list(parent))}

    def imported(called: str) -> tolerance_rules.Imported | None:
        source, _, name = called.rpartition(".")
        path = module.path.parent / f"{source}.py"
        if not source or "." in source or not path.is_file():
            return None
        return _support_functions(path, parent, path.stat().st_mtime_ns).get(name)

    return tolerance_rules.comparisons(module.tree, tables, _Calls(module.tree), imported)


@functools.cache
def _support_functions(path: Path, parent: tuple[str, ...], modified: int
                       ) -> dict[str, tolerance_rules.Imported]:
    """What the file at `path` proves of each of its module-level functions
    (`tolerance_rules.support_functions`), read as of its modification time `modified`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tables = {id(node): table for node, table in scopes.scoped(tree.body, {}, list(parent))}
    calls = _Calls(tree)
    # A name the module binds twice, or to anything but its one definition, is no one's.
    bound = calls.bindings[id(tree)]
    return {name: found for name, found in tolerance_rules.support_functions(tree, tables, calls).items()
            if isinstance(bound.get(name), ast.FunctionDef | ast.AsyncFunctionDef)}


def _inventory(checkout: Path) -> dict:
    """The inventory as it is now: read again whenever the file changes, so one process reading two
    states of a checkout sees each."""
    path = checkout / INVENTORY
    if not path.is_file():
        return {}
    stat = path.stat()
    return _read_inventory(path, stat.st_mtime_ns, stat.st_size)


@functools.cache
def _read_inventory(path: Path, mtime: int, size: int) -> dict:
    del mtime, size  # the cache's key: the file's state
    return json.loads(path.read_text())


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=checkout, capture_output=True, text=True,
                          check=True).stdout


@functools.cache
def _inventory_at(checkout: Path, commit: str) -> dict | None:
    shown = subprocess.run(["git", "show", f"{commit}:{INVENTORY}"], cwd=checkout, capture_output=True,
                           text=True, check=False)
    return json.loads(shown.stdout) if shown.returncode == 0 else None


def _loosened(checkout: Path, commit: str, parent: str, mine: dict, theirs: dict,
              authorized: dict[str, int] | None = None,
              grants: dict[str, int] | None = None) -> Iterator[tuple[str, str]]:
    """How `commit`'s inventory `mine` is looser than `parent`'s `theirs`: each file it adds or whose
    count it raises, and how. A file renamed between them carries its entry. Where the
    commit changes SLOP020's reading (`RULE`), a file it leaves unchanged may be added or raised: what
    the rule now reads there was in the file all along, and `grants` takes the count it reached. A file
    may also sit at a count `authorized` holds for it, one a commit between `parent` and `commit`
    reached so (HEAD against its base)."""
    renamed, changed = _changes(checkout, parent, commit)
    reread = RULE in changed
    for name, entry in mine["files"].items():
        kept = theirs["files"].get(name) or theirs["files"].get(renamed.get(name, ""))
        held = (authorized or {}).get(name, (authorized or {}).get(renamed.get(name, ""), -1))
        grown = (reread and name not in changed) or entry["comparisons"] <= held
        if grants is not None and reread and name not in changed:
            grants[name] = max(grants.get(name, 0), entry["comparisons"])
        if kept is None:
            if not grown:
                yield name, f"adds {name}"
            continue
        if entry["comparisons"] > kept["comparisons"] and not grown:
            yield name, f"raises {name}'s count"


@functools.cache
def _changes(checkout: Path, parent: str, commit: str) -> tuple[dict[str, str], frozenset[str]]:
    """What `commit` changed of `parent`'s tests and tools: the files it renamed, by their new names, and
    every path it touched."""
    renamed, changed = {}, set()
    changes = _git(checkout, "diff", "--name-status", "-M", parent, commit, "--", "tests", "tools")
    for change in changes.splitlines():
        status, *paths = change.split("\t")
        changed.update(paths)
        if status.startswith("R") and len(paths) == 2:
            renamed[paths[1]] = paths[0]
    return renamed, frozenset(changed)


def _lineage(checkout: Path, walk: list[tuple[str, list[str]]]) -> dict[str, dict[str, frozenset[str]]]:
    """Each walked commit's inventoried files, each to the entries it descends from (a commit and a name,
    where an entry was first made), composed commit by commit along `walk`, parents before children, so a
    rename stays traced however its file is edited after."""
    lineage: dict[str, dict[str, frozenset[str]]] = {}
    for commit, parents in walk:
        mine = _inventory_at(checkout, commit)
        if mine is None:
            lineage[commit] = {}
            continue
        traced = [(lineage[parent], _changes(checkout, parent, commit)[0])
                  for parent in parents if parent in lineage]
        lineage[commit] = {}
        for name in mine["files"]:
            inherited = [held.get(renamed.get(name, name), frozenset()) for held, renamed in traced]
            lineage[commit][name] = frozenset().union(*inherited) or frozenset({f"{commit}:{name}"})
    return lineage


def _merge_loosens(name: str, per_parent: list[dict]) -> str | None:
    """How a commit loosens the file `name` against all its parents at once (`_loosened` against each,
    `per_parent`), or None: its count past every parent's (each parent's own raise judged at its own
    commit, so one parent's allows it)."""
    if all(name in found for found in per_parent):
        return per_parent[0][name]
    return None


def _duplicated(lineage: dict[str, frozenset[str]]) -> Iterator[str]:
    """A merge keeping one entry under two names (its parents having renamed it apart): each name's
    count is that one entry's allowance, so together they hold it twice."""
    by_origin: dict[str, list[str]] = {}
    for name, origins in sorted(lineage.items()):
        for origin in origins:
            by_origin.setdefault(origin, []).append(name)
    for origin, names in sorted(by_origin.items()):
        if len(names) > 1:
            yield (f"keeps the entry made as {origin.split(':', 1)[1]} as {' and '.join(names)}, its parents "
                   f"having renamed it apart")


def ratchet(checkout: Path, base: str | None = None) -> Iterator[Finding]:
    """SLOP020's inventory never loosens. HEAD's is held to `base`'s merge-base with HEAD (main's, in
    CI), and every commit in HEAD's history whose inventory differs from a parent's, merges included
    (`--full-history`, no simplification), to that parent's (`_loosened`), from the commit that first
    made it on, which has no parent's to keep, and no commit removes it once made. A merge is held to
    its loosest parent's count, file by file, each parent's own raise judged at its own commit, so a
    merge that brings a count another branch raised is no loosening of its own; and to keeping one entry
    only once, however its parents renamed it (`_merge_loosens`, `_lineage`). Only a
    reading of the rule HEAD's own history changed since `base` authorizes a count against it.
    `--ratchet` runs it (CI's lint job, on its whole history); a shallow
    checkout, or one without git, cannot hold the inventory, and it says so rather than passing."""
    try:
        shallow = _git(checkout, "rev-parse", "--is-shallow-repository").strip() == "true"
        # The walk starts where the inventory was first made: the history before it holds none.
        made = _git(checkout, "log", "--full-history", "--diff-filter=A", "--reverse", "--format=%P", "HEAD",
                    "--", INVENTORY).splitlines()
        before = [f"^{parent}" for parent in (made[0].split() if made else [])]
        listed = _git(checkout, "rev-list", "--full-history", "--parents", "HEAD", *before, "--", INVENTORY)
        oldest_first = _git(checkout, "rev-list", "--full-history", "--topo-order", "--reverse", "--parents",
                            "HEAD", *before, "--", INVENTORY)
        merged = _git(checkout, "merge-base", "HEAD", base).strip() if base else None
        # The commits HEAD's history adds to the base's: only their rereadings of the rule authorize.
        fresh = set(_git(checkout, "rev-list", "HEAD", f"^{merged}").split()) if merged else None
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        yield Finding(INVENTORY, 1, 1, "SLOP020", "holds the inventory to its history, which git cannot "
                      f"read here ({error}); run with the repository's history")
        return
    if shallow:
        yield Finding(INVENTORY, 1, 1, "SLOP020", "holds the inventory to its history, and this checkout is "
                      "shallow: fetch it whole (`git fetch --unshallow`)")
        return
    head_id = _git(checkout, "rev-parse", "HEAD").strip()
    head = _inventory_at(checkout, head_id)
    # Each commit against its parents first, so the counts a change of the rule's reading reached
    # (`grants`) hold HEAD against its base too.
    walked, grants = [], {}
    lineage = _lineage(checkout, [(line.split()[0], line.split()[1:])
                                  for line in filter(None, oldest_first.splitlines())])
    for line in filter(None, listed.splitlines()):
        commit, *parents = line.split()
        mine = _inventory_at(checkout, commit)
        # Per parent with an inventory, what the commit loosens against it; a merge loosens a count only
        # past them all.
        loosened = []
        for parent in parents:
            theirs = _inventory_at(checkout, parent)
            if theirs is None:
                continue
            if mine is None:
                walked.append(Finding(INVENTORY, 1, 1, "SLOP020", f"{commit[:9]} removes the inventory "
                                      f"its parent {parent[:9]} kept; an empty inventory stays, as "
                                      f"`files: {{}}`"))
                continue
            granting = grants if fresh is None or commit in fresh else None
            found = _loosened(checkout, commit, parent, mine, theirs, grants=granting)
            loosened.append((parent, dict(found)))
        if not loosened:
            continue
        against = ("its parent " if len(loosened) == 1 else "its parents ") + ", ".join(
            parent[:9] for parent, _ in loosened)
        for name in sorted({name for _, found in loosened for name in found}):
            why = _merge_loosens(name, [found for _, found in loosened])
            if why:
                walked.append(Finding(INVENTORY, 1, 1, "SLOP020", f"{commit[:9]} against {against} {why}"))
        if len(loosened) > 1 and mine is not None:
            walked += [Finding(INVENTORY, 1, 1, "SLOP020", f"{commit[:9]} {why}")
                       for why in _duplicated(lineage.get(commit, {}))]
    if merged is not None:
        kept = _inventory_at(checkout, merged)
        if kept is not None and head is None:
            yield Finding(INVENTORY, 1, 1, "SLOP020",
                          f"HEAD removes the inventory {base}'s {merged[:9]} keeps")
        elif kept is not None:
            for _, why in _loosened(checkout, head_id, merged, head, kept, authorized=grants):
                yield Finding(INVENTORY, 1, 1, "SLOP020", f"HEAD against {base}'s {merged[:9]} {why}")
    yield from walked


def _checkout(module: Module) -> Path:
    return Path(module.path.as_posix().removesuffix(module.relative))


def tolerances(module: Module) -> Iterator[Finding]:
    """SLOP020: a float compared within a tolerance outside the gate (`GATE`, `_tolerance_comparisons`).
    A file the inventory holds may keep its count; above it, every comparison is reported; below it,
    the inventory is told to come down, so the count it keeps is the file's and it only shrinks
    (`ratchet` holds every commit's inventory to its parents')."""
    if not module.relative.startswith(("tests/", "tools/")) or module.relative in GATE:
        return
    found = sorted(_tolerance_comparisons(module),
                   key=lambda pair: (pair[0].lineno, pair[0].col_offset))
    inventory = _inventory(_checkout(module))
    entry = inventory.get("files", {}).get(module.relative)
    gate = "go through the gate's assertion (tests/reference_error.py)"
    if entry is None:
        why = "not in the inventory"
    else:
        if len(found) > entry["comparisons"]:
            why = f"the inventory counts {entry['comparisons']}"
        elif len(found) < entry["comparisons"]:
            yield Finding(module.relative, 1, 1, "SLOP020",
                          f"{INVENTORY} counts {entry['comparisons']} tolerance comparisons here and "
                          f"the file holds {len(found)}: "
                          + ("remove its entry" if not found else f"lower it to {len(found)}"))
            return
        else:
            return
    for node, what in found:
        yield Finding(module.relative, node.lineno, node.col_offset + 1, "SLOP020",
                      f"{what} compares floats within a tolerance ({why}); {gate}")


CONTRACT_RULES = (contracts, probes, names, imports, calls, authorities, waits, detections)
UNIVERSAL_RULES = (suppressions, swallowed, comments, size, jax_private)


def check(module: Module, models: ModelIndex | None = None) -> Iterator[Finding]:
    """Every rule that applies to this file, in code order. SLOP010 reads
    `models`, the package's index, or this file's own without one."""
    rules = (*CONTRACT_RULES, *UNIVERSAL_RULES) if module.is_source else UNIVERSAL_RULES
    rules = (*rules, mocks, wording, skips) if module.relative.startswith("tests/") else (*rules, patches)
    rules = (*rules, tolerances)
    findings = [finding for rule in rules for finding in rule(module)]
    if module.is_source:
        findings += boundaries(module, models or ModelIndex.build([module]))
    yield from sorted(findings, key=lambda finding: (finding.line, finding.col, finding.code))


RECORDED = {"site/src/data/hero.py": "site/src/data/capture.json"}
"""Python files whose bytes a recording pins, each with the recording that names their sha256
(`meta.script_sha256`): the landing hero's script, whose capture the site's build checks it against
(site/src/pages/index.astro), so an edit fails the build until the hero is recorded again on a host as
fast as the recording's (the owner keeps the 60 s recording). The gate reads such a file only
once its bytes are no longer the recorded ones, which an edit makes them."""


def recorded(checkout: Path, relative: str, path: Path) -> bool:
    """Are `path`'s bytes still the ones the recording `RECORDED` names for `relative` pins?"""
    if relative not in RECORDED:
        return False
    capture = json.loads((checkout / RECORDED[relative]).read_text())
    return capture["meta"]["script_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def files(roots: Sequence[str], checkout: Path = ROOT) -> Iterator[tuple[Path, str]]:
    """Every Python file under the named roots of `checkout`, or the root
    itself when it names a file, with its path from the checkout, skipping
    stub-only trees, the site's installed npm packages (node-gyp ships
    Python) and a file whose bytes are still those a recording pins
    (`RECORDED`)."""
    for root in roots:
        named = checkout / root
        for path in [named] if named.is_file() else sorted(named.rglob("*.py")):
            relative = path.relative_to(checkout).as_posix()
            if "/stubs/" in f"/{relative}" or "/node_modules/" in f"/{relative}":
                continue
            if recorded(checkout, relative, path):
                continue
            yield path, relative


def parsed(path: Path, relative: str, package: str = "dew") -> Module:
    return Module(path, relative, path.read_text(), ast.parse(path.read_text()), package)


def collect(roots: Sequence[str], checkout: Path = ROOT, package: str = "dew") -> Iterator[Module]:
    """Every file `files` names, parsed."""
    for path, relative in files(roots, checkout):
        yield parsed(path, relative, package)


INDEX: ModelIndex | None = None
"""The model index `checked` reads: built once by `main`, before any fork, and inherited, never sent."""


def checked(file: tuple[Path, str, str]) -> list[Finding]:
    """One file's findings, against the model index `main` built (`INDEX`)."""
    assert INDEX is not None, "main builds the index before it reads a file"
    return list(check(parsed(*file), INDEX))


def default_roots(package: str) -> list[str]:
    """What the gate reads when it names no roots: the package's source and every script, the site's
    and the docs' generators and kernel packages among them."""
    return [f"src/{package}", "tests", "tools", "recipes", "examples", "site", "docs/assets", "kernels"]


def main(argv: Sequence[str] | None = None, forked: bool = False) -> int:
    """The gate, its files read in forked processes when `forked` (`forked_map`), as the script reads them,
    its findings printed in the files' order, as one process prints them. A caller in a process that may
    hold threads (a test, under JAX) reads them in its own process."""
    parser = argparse.ArgumentParser(description="Report low-evidence Python, `path:line:col: SLOPxxx`.")
    parser.add_argument("roots", nargs="*", help="directories or files under the checkout; by default the "
                        "package's source, tests, tools, recipes, examples, site, docs/assets and kernels")
    parser.add_argument("--root", type=Path, default=ROOT, help="the checkout (default: Dew's)")
    parser.add_argument("--package", default="dew", help="the package under src/ (default: dew)")
    parser.add_argument("--ratchet", action="store_true",
                        help="hold SLOP020's inventory to its git history (`ratchet`), the whole history")
    parser.add_argument("--base", help="with --ratchet, the branch HEAD's inventory is held to (at their "
                        "merge-base): origin/main in CI")
    parser.add_argument("--print-roots", action="store_true",
                        help="print the roots it reads by default; tools/pre-commit lints their staged files")
    args = parser.parse_args(argv)
    if args.ratchet:
        findings = list(ratchet(args.root.resolve(), args.base))
        for finding in findings:
            print(finding)
        print(f"{len(findings)} findings", file=sys.stderr)
        return 1 if findings else 0
    if args.print_roots:
        print(" ".join(default_roots(args.package)))
        return 0
    roots = args.roots or default_roots(args.package)
    counts: dict[str, int] = {}
    coded: dict[str, set[str]] = {}
    global INDEX
    INDEX = ModelIndex.build(list(collect([f"src/{args.package}"], args.root.resolve(), args.package)))
    read = [(path, relative, args.package) for path, relative in files(roots, args.root.resolve())]
    seen = {relative for _, relative, _ in read}
    found = forked_map(checked, read) if forked else [checked(file) for file in read]
    for finding in (finding for each in found for finding in each):
        print(finding)
        counts[finding.code] = counts.get(finding.code, 0) + 1
        coded.setdefault(finding.code, set()).add(finding.path)
    if not args.roots:
        # Every file the inventory names is read on a whole run, so one it names that is gone is stale.
        for relative in sorted(set(_inventory(args.root.resolve()).get("files", {})) - seen):
            print(Finding(INVENTORY, 1, 1, "SLOP020", f"names {relative}, which is gone: remove its entry"))
            counts["SLOP020"] = counts.get("SLOP020", 0) + 1
            coded.setdefault("SLOP020", set()).add(INVENTORY)
    for code, count in sorted(counts.items()):
        print(f"{code}: {count} in {len(coded[code])} files", file=sys.stderr)
    failures = sum(counts.values())
    print(f"{failures} findings", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(forked=True))
