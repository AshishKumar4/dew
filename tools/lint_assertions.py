"""Bible §7's banned assertions, read in a test file for tools/lint_slop.py, each rule a finding's line,
column, code and message (`Found`):

- SLOP008: a first-party module, or a table or registry it holds, replaced instead of a seam taken; or a
  private attribute of an object the test made spied on or replaced (class 3).
- SLOP011: a test of wording or source, not behaviour: an error's message matched, captured output looked
  for a sentence (through the names it is bound to), a rendering's text, a source file read or named
  (class 5).
- SLOP015: a test that skips when its own package fails to import.
- SLOP029: a test whose every assertion is that a process failed, read from its exit status alone, which a
  crash passes as well as the refusal it is after (class 7).
- SLOP033: a choice Dew derives (a config a call named for choosing returns: `derived`, `chosen`, `picked`,
  `offered`, ...) pinned to a literal: its stages, consumers or a block size compared with a number
  (CONTRIBUTING's banned assertion 2). A legality check (a derived tile against a budget it computes) and a
  config the test configures itself pass.

Each reads names through the imports of its own lexical scope (tools/lint_scopes.py)."""

import ast
import itertools
import re
import sys
from collections.abc import Iterator, Sequence, Set
from typing import Protocol, TypeGuard

from tools import lint_scopes as scopes, lint_texts as texts

Found = tuple[int, int, str, str]
"""A finding: its line, column, code and message."""


class Parsed(Protocol):
    """What the rules read of a file (lint_slop's `Module`)."""

    relative: str
    tree: ast.Module
    package: str


SEAMS = ("subprocess", "socket", "urllib", "request", "http", "download", "hub", "hf_hub",
         "time", "sleep", "monotonic", "perf_counter", "open", "path", "os.", "environ",
         "fetch", "client", "urlopen", "snapshot")
"""The seams a test may replace, as the real one leaves the process (SLOP008)."""
SOURCE_READERS = {"inspect.getsource", "inspect.getsourcelines", "inspect.getsourcefile"}
"""The calls that hand a test a function's source text (SLOP011)."""

def _patched(node: ast.Call, table: dict[str, str]) -> str:
    """What a patching call replaces, as a dotted path: `mock.patch("a.b")`'s
    string; `patch.object(owner, "b")`'s and `monkeypatch.setattr(owner, "b",
    ...)`'s or `monkeypatch.setattr("a.b", ...)`'s attribute, its owner read
    through the scope's imports (`scopes.resolved`), however it was imported or
    aliased; `monkeypatch.setitem(mapping, ...)`'s and `delitem`'s mapping,
    which a module's table or registry is. "" for any other call."""
    name = scopes.dotted(node.func)
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
    between its dots and underscores, so `dew._cli.tpu.time.sleep` and
    `dew.interop.sources.snapshot` do and `dew.training.narrow.narrowed_paths` does not?
    A private attribute is its owner's: of a seam's owner it leaves the process as
    the owner does (`dew._coordination.os._exit`), and of a first-party object it
    never does, whatever it is named (`dew.checkpoints.Checkpoints._open`): it is the
    code under test, and the boundary it reaches is observed there instead (an audit
    hook, a process of its own)."""
    owner, _, name = target.rpartition(".")
    if name.startswith("_"):
        # A seam's own private attribute: its owner a standard library module the target reaches by
        # name, as `os` in `dew._coordination.os._exit`, where `Hub` in `...hub.Hub._selected` is not.
        held = owner.rpartition(".")[2]
        return held in sys.stdlib_module_names and _seam(owner)
    words = [word for word in re.split(r"[._]", target.lower()) if word]
    for seam in SEAMS:
        parts = [word for word in re.split(r"[._]", seam) if word]
        if any(words[start:start + len(parts)] == parts for start in range(len(words))):
            return True
    return False


def _spied(node: ast.Call, table: dict[str, str]) -> str:
    """The private attribute (`_name`, not a dunder) a patching call replaces on an object the test holds
    itself, an instance its own code made, as `monkeypatch.setattr(trainer, "_step", counting)` or
    `patch.object(store, "_open", wraps=store._open)` spies on it: "<owner>.<attribute>", or "" where the
    call patches none so, or patches a module's member (`_patched` reads those)."""
    name = scopes.dotted(node.func)
    patching = name.endswith(("patch.object", "monkeypatch.setattr", "monkeypatch.delattr"))
    if not patching or len(node.args) < 2:
        return ""
    owner, attribute = node.args[0], node.args[1]
    if not (isinstance(attribute, ast.Constant) and isinstance(attribute.value, str)) or isinstance(
            owner, ast.Constant):
        return ""
    private = attribute.value.startswith("_") and not (attribute.value.startswith("__")
                                                       and attribute.value.endswith("__"))
    return f"{scopes.dotted(owner) or 'an object'}.{attribute.value}" if private and not scopes.resolved(
        owner, table) else ""


def mocks(module: Parsed, parent: Sequence[str]) -> Iterator[Found]:
    """SLOP008: a first-party module, or a table or registry it holds, replaced
    instead of a seam taken; or a private attribute of an object the test made spied on or replaced
    (`_spied`), whose calls, counts or traces are the code under test's internals (bible §7's class 3:
    observe the public outcome instead: transfers, reuse, release, results)."""
    for node, table in scopes.scoped(module.tree.body, {}, parent):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if spied := _spied(node, dict(table)):
            yield (node.lineno, node.col_offset + 1, "SLOP008",
                   f"spies on or replaces {spied}, a private attribute; assert the public outcome")
            continue
        target = _patched(node, dict(table))
        if not target.startswith(f"{module.package}."):
            continue
        if _seam(target):
            continue
        yield (node.lineno, node.col_offset + 1, "SLOP008",
               f"patches {target}; the fix is a seam the test can pass a double to")


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


def _returned(tree: ast.Module) -> dict[str, frozenset[int] | None]:
    """The functions `tree` defines, at any depth, that return captured output (`_captured`), by name: None
    where one returns it whole, else the positions of a returned tuple that hold it (as a helper that runs a
    command and returns `(code, capsys.readouterr().out)` does), so a test's sentence looked for in what it
    returns is one looked for in captured output (bible §7's class 5)."""
    found: dict[str, frozenset[int] | None] = {}
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        names: set[str] = set()
        for node in scopes.own(function):
            if isinstance(node, ast.Assign) and _captured(node.value, names):
                names |= {name.id for target in node.targets for name in ast.walk(target)
                          if isinstance(name, ast.Name)}
        for node in scopes.own(function):
            if not isinstance(node, ast.Return) or node.value is None:
                continue
            if _captured(node.value, names):
                found[function.name] = None
            elif isinstance(node.value, ast.Tuple) and (positions := frozenset(
                    index for index, element in enumerate(node.value.elts) if _captured(element, names))
            ) and (held := found.get(function.name, frozenset())) is not None:
                found[function.name] = held | positions
    return found


def _bound_captured(node: ast.Assign | ast.AnnAssign | ast.AugAssign, names: Set[str],
                    returned: dict[str, frozenset[int] | None]) -> set[str]:
    """The names an assignment binds to captured output: every name it binds where its value is captured
    output, or a call of a function returning it whole (`_returned`); a tuple target's names at the
    positions such a function's tuple holds it."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    bound = {name.id for target in targets for name in ast.walk(target)
             if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)}
    if node.value is None:
        return set()
    if _captured(node.value, names):
        return bound
    called = scopes.dotted(node.value.func) if isinstance(node.value, ast.Call) else ""
    if called not in returned:
        return set()
    positions = returned[called]
    if positions is None:
        return bound
    return {element.id for target in targets if isinstance(target, ast.Tuple)
            for index in positions
            if index < len(target.elts) and isinstance(element := target.elts[index], ast.Name)}


def _held(tree: ast.Module) -> dict[int, frozenset[str]]:
    """The names bound to captured output (`_captured`) where each node of a function's own scope is:
    a name the function binds to it, or unpacks it into, or to what a helper returning it returns
    (`_returned`), until it binds that name again to anything else (a parse of it, say); added to, it
    stays captured. An item or attribute assigned binds no name."""
    held: dict[int, frozenset[str]] = {}
    returned = _returned(tree)
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
            if captured := _bound_captured(node, names, returned):
                names |= captured
                names -= bound - captured
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


RENDERINGS = frozenset({"plain", "table", "render", "renderable", "getvalue", "readouterr", "out", "err",
                        "stdout", "stderr", "text"})
"""The attributes and methods a rendering's text is read through (a rich Text's `plain`, a table's
`table()`, captured output): the values SLOP011's layout rule reads (`_rendered`)."""


def _rendered(expr: ast.expr) -> bool:
    """Is `expr` a rendering's text: read through RENDERINGS, its lines (`.splitlines()`) or a str()
    of something, as a parsed value (`json.loads`) is not?"""
    while isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr in (
            "splitlines", "strip", "rstrip", "lstrip"):
        expr = expr.func.value
    if isinstance(expr, ast.Call):
        if scopes.dotted(expr.func) == "str":
            return True
        expr = expr.func
    return isinstance(expr, ast.Attribute) and expr.attr in RENDERINGS


def _laid_out(expr: ast.expr) -> bool:
    """Is `expr` a rendering's layout spelled out: a string, or a list or tuple of strings, holding a
    run of two or more spaces between its words, a frozen line of a table or panel whose spacing is
    the snapshot (row 1878)? Its content claims are words, numbers and column ends, asserted as such."""
    texts = [expr] if isinstance(expr, ast.Constant) else (
        list(expr.elts) if isinstance(expr, ast.List | ast.Tuple) else [])
    return any(isinstance(text, ast.Constant) and isinstance(text.value, str)
               and re.search(r"\S {2,}\S", text.value) is not None for text in texts)


def _worded(node: ast.AST, caught: Set[str], raised: Set[str]) -> TypeGuard[ast.Compare | ast.Call]:
    """Does `node` look for a sentence the source spells in a process's `stdout` or `stderr`, by
    `in`, `startswith`, `endswith` or `count`, or match a caught error's text (`str` or `repr` of
    it) so or by `==`? A process's whole output compared, or parsed first, is the value it reports."""
    def output(expr: ast.expr) -> bool:
        return isinstance(expr, ast.Attribute) and expr.attr in ("stdout", "stderr")

    def error(expr: ast.expr) -> bool:
        if not (isinstance(expr, ast.Call) and scopes.dotted(expr.func) in ("str", "repr")
                and len(expr.args) == 1):
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


def wording(module: Parsed, parent: Sequence[str]) -> Iterator[Found]:
    """SLOP011: a test that checks wording or source text instead of behaviour: an error's or a
    warning's text matched, a sentence looked for in captured output, in a caught error's text or
    in a process's output, source text read, or an assertion on production source or a program's IR
    text (`lint_texts.texts`). A name is read through the imports of its own scope, so
    `pytest.raises` imported under another name is `pytest.raises`."""
    named = list(scopes.scoped(module.tree.body, {}, parent))
    raises = {id(node) for node, table in named
              if isinstance(node, ast.Call) and scopes.resolved(node.func, table) == "pytest.raises"}
    for function in ast.walk(module.tree):
        if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            caught, raised = _errors(function, raises)
            for node in scopes.own(function):
                if _worded(node, caught, raised):
                    yield (node.lineno, node.col_offset + 1, "SLOP011",
                           "looks for a sentence in an error's or a process's text; assert its "
                           "type, its exit status or the value it reports")
                elif (isinstance(node, ast.Compare) and len(node.ops) == 1
                      and isinstance(node.ops[0], ast.Eq | ast.NotEq)
                      and ((_laid_out(node.left) and _rendered(node.comparators[0]))
                           or (_laid_out(node.comparators[0]) and _rendered(node.left)))):
                    yield (node.lineno, node.col_offset + 1, "SLOP011",
                           "freezes a rendering's spacing; assert its words, numbers and column "
                           "ends")
    tables = {id(node): table for node, table in named}
    for function in ast.walk(module.tree):
        if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            for node in texts.texts(function, tables, module.package):
                yield (node.lineno, node.col_offset + 1, "SLOP011",
                       "asserts on production source or a program's IR text; test what the code does, "
                       "reading a program's structure as objects")
    held = _held(module.tree)
    for node, table in named:
        if isinstance(node, ast.expr) and _spelled_in_captured(node, held.get(id(node), frozenset())):
            yield (node.lineno, node.col_offset + 1, "SLOP011",
                   "looks for a sentence in captured output; assert the state it reports")
        if isinstance(node, ast.Call):
            called = scopes.resolved(node.func, table)
            if called in ("pytest.raises", "pytest.warns") and any(keyword.arg == "match"
                                                                   for keyword in node.keywords):
                raised = "an error" if called == "pytest.raises" else "a warning"
                yield (node.lineno, node.col_offset + 1, "SLOP011",
                       f"matches {raised}'s wording; assert that it is raised, not its text")
            elif called in SOURCE_READERS:
                yield (node.lineno, node.col_offset + 1, "SLOP011",
                       "reads source text; test what the code does")
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and node.value.startswith(f"src/{module.package}/") and node.value.endswith(".py")):
            yield (node.lineno, node.col_offset + 1, "SLOP011",
                   f"names {node.value}; test what the code does, not its source")


EXITS = {"subprocess.call", "os.system"}
"""The calls whose value is a process's exit status alone (SLOP029)."""


def _fails_only(test: ast.expr, table: dict[str, str]) -> bool:
    """Is assertion `test` that a process failed, read from its exit status and nothing else: its
    `returncode` (or a call of `EXITS`) compared unequal to 0 or equal to 1, or tested for truth, or such
    checks joined by `and`? Any failure passes it, a crash as well as the refusal it is after: 1 is what
    Python exits with on an error nothing caught."""
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return all(_fails_only(value, table) for value in test.values)

    def status(expr: ast.expr) -> bool:
        return (isinstance(expr, ast.Attribute) and expr.attr == "returncode") or (
            isinstance(expr, ast.Call) and scopes.resolved(expr.func, table) in EXITS)

    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        left, right = test.left, test.comparators[0]
        constant = right if status(left) else left if status(right) else None
        # A code of the process's own (a signal's 128 + N, a usage error's 2) is its report; "anything but 0"
        # and an uncaught error's 1 are any failure.
        return isinstance(constant, ast.Constant) and (
            (isinstance(test.ops[0], ast.NotEq) and constant.value == 0)
            or (isinstance(test.ops[0], ast.Eq) and constant.value == 1))
    return status(test)


def statuses(module: Parsed, parent: Sequence[str]) -> Iterator[Found]:
    """SLOP029: a test whose every assertion is that a process failed, read from its exit status alone
    (`_fails_only`), which a crash passes as well as the refusal it is after (bible §7's class 7): assert
    the structured state the refusal reports or leaves (its verdict, its records, the files it never
    wrote) as well."""
    named = {id(node): table for node, table in scopes.scoped(module.tree.body, {}, parent)}
    for function in ast.walk(module.tree):
        defined = isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef)
        if not (defined and function.name.startswith("test")):
            continue
        asserts = [node for node in scopes.own(function) if isinstance(node, ast.Assert)]
        if asserts and all(_fails_only(node.test, dict(named.get(id(node), {}))) for node in asserts):
            yield (function.lineno, function.col_offset + 1, "SLOP029",
                   "asserts only that a process failed, from its exit status, which a crash passes; "
                   "assert the state the refusal reports or leaves")


def skips(module: Parsed, parent: Sequence[str]) -> Iterator[Found]:
    """SLOP015: a test that skips when its own package fails to import."""
    for node, table in scopes.scoped(module.tree.body, {}, parent):
        if not (isinstance(node, ast.Call) and scopes.resolved(node.func, table) == "pytest.importorskip"):
            continue
        named = [*node.args[:1], *(keyword.value for keyword in node.keywords if keyword.arg == "modname")]
        if (named and isinstance(named[0], ast.Constant) and isinstance(name := named[0].value, str)
                and (name == module.package or name.startswith(f"{module.package}."))):
            yield (node.lineno, node.col_offset + 1, "SLOP015",
                   f"skips when {name} fails to import; import it, so a broken import fails")


DERIVERS = re.compile(r"(?:^|_)(?:choose|chosen|choice|pick|picked|derive|derived|select|selected|offered|"
                      r"best|plan|planned|config_for|tuned|decide|decided)(?:_|s?$)")
"""The names of the calls a test reads a derived choice from (SLOP033)."""
DERIVED_FIELDS = ("stages", "consumers")
"""A derived config's fields SLOP033 reads, beside every `block_*`."""


def _bindings(function: ast.AST) -> list[tuple[tuple[int, int], str, bool]]:
    """Each name `function`'s own scope binds (an assignment, annotated or not, a walrus, a loop's or a
    comprehension's target): its target's position, the name, and whether a call `DERIVERS` names gave it."""
    found = []
    for node in scopes.own(function):
        if isinstance(node, ast.Assign):
            value, targets = node.value, node.targets
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            value, targets = node.value, [node.target]
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            value, targets = node.iter, [node.target]
        else:
            continue
        derived = bool(DERIVERS.search(scopes.called(value)))
        found += [((name.lineno, name.col_offset), name.id, derived) for target in targets
                  for name in ast.walk(target) if isinstance(name, ast.Name)]
    return sorted(found)


def _chosen(side: ast.expr, at: tuple[int, int], bindings: list[tuple[tuple[int, int], str, bool]]) -> str:
    """The field `side` reads of a derived choice at `at` (`_bindings`' names), or ""."""
    if not isinstance(side, ast.Attribute) or not (side.attr in DERIVED_FIELDS
                                                   or side.attr.startswith("block_")):
        return ""
    base = side.value
    if isinstance(base, ast.Call):
        return side.attr if DERIVERS.search(scopes.called(base)) else ""
    if not isinstance(base, ast.Name):
        return ""
    last = [derived for where, name, derived in bindings if name == base.id and where < at]
    return side.attr if last and last[-1] else ""


def pinned_choices(module: Parsed, parent: Sequence[str]) -> Iterator[Found]:
    """SLOP033: an assertion comparing a derived config's stages, consumers or block size (`DERIVED_FIELDS`,
    `block_*`) with a number literal beside it in the comparison: the config a call `DERIVERS` names, or a
    name the test's function last bound to one before the assertion (a later binding to anything else, the
    test's own config, frees it)."""
    del parent

    def number(node: ast.expr) -> bool:
        return isinstance(node, ast.Constant) and type(node.value) in (int, float)

    for function in (node for node in ast.walk(module.tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
        bindings = _bindings(function)
        for check in (node for node in scopes.own(function) if isinstance(node, ast.Assert)):
            at = (check.lineno, check.col_offset)
            fields = [field for compare in ast.walk(check.test) if isinstance(compare, ast.Compare)
                      for left, right in itertools.pairwise([compare.left, *compare.comparators])
                      for side, other in ((left, right), (right, left))
                      if number(other) and (field := _chosen(side, at, bindings))]
            if fields:
                yield (check.lineno, check.col_offset + 1, "SLOP033",
                       f"pins a derived choice's {fields[0]} to a literal; assert the choice legal and its "
                       "result correct, or configure the value the test needs explicitly")


RULES = (mocks, wording, skips, statuses, pinned_choices)
"""The rules, in lint_slop's order."""
