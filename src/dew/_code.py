"""The digest of the code a record of what Dew computed is a function of (`code_digest`): its producer's
modules and every module of Dew's their code can import, read from their source, none imported, so a record
is computed again where that code changed and kept where Dew changed elsewhere (row 10743: every record
keyed by the whole package, any edit certified every launch again).

A module's imports are its code's every import statement, a function's own and a class body's among them,
but not those of an `if TYPE_CHECKING:` block, which never runs; each parent package's `__init__`; each
`import_module` of a name its code states (`f"{__name__}.program"`, a relative name with its package); and
each name read from a lazily exporting package (`_EXPORTS`, `_NAMED`, `_SUBMODULES`: `dew._exports`),
resolved to the module that defines it, or the package's whole table where its code holds the package as a
value (`getattr(package, name)`, an argument). A name bound twice is read as either. An import by a name its
code computes is a dynamic site, named by the expression it imports: each of Dew's is reviewed (`REVIEWED`)
or resolved by the literal table it selects from (`SELECTED`), and a closure holding any other, or a lazy
table no literal states, is keyed by the whole package instead (`dew._cache.package_digest`), logged, so a
site added unreviewed costs warm records, never a stale one.

Every read is of one snapshot of the tree a process takes once (`_snapshot`: each file's digest), so a
closure and its digest state one revision; a process whose files changed after keeps no record. A closure
is walked once a tree and kept in Dew's cache, so a start reads its files' bytes, not their syntax: a walk
parses an op's 153 files in about a second."""

import ast
import contextlib
import functools
import hashlib
import json
import logging
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from dew._builds import dew_cache_dir
from dew._cache import VERSION_FILE, package_digest
from dew._files import write_atomically
from dew._records import JSON

_log = logging.getLogger(__name__)

SOURCE = Path(__file__).resolve().parent.parent
"""The directory holding Dew's packages: a checkout's src, an installation's site-packages."""

PACKAGES = ("dew", "_dew_torch")
"""Dew's packages, each a module of which a closure follows; every other import is a library's, which its
release states."""

REVIEWED = {
    "dew._exports:lookup.__getattr__:module":
        "a lazily exporting package's `_EXPORTS` value, each name a closure reads resolved to its module",
    "dew._exports:lookup.__getattr__:modules[name]":
        "a lazily exporting package's own module of an `_EXPORTS` value, resolved as the name is",
    "dew.nn.kernels:__getattr__:f'{__name__}.{module}'":
        "the package's `_NAMED` value, each name a closure reads resolved to its module",
    "dew.algebra:__getattr__:f'{__name__}.{name}'":
        "the package's `_SUBMODULES` member, each name a closure reads resolved to its module",
    "dew.algebra:__getattr__:f'{__name__}.{_NAMED[name]}'":
        "the package's `_NAMED` value, each name a closure reads resolved to its module",
    "dew._devices:cublas_version:f'jax_{build}_plugin._versions'": "JAX's CUDA plugin's version reader",
    "dew.algebra.terms:_sort:declared.ident.module or 'egglog'":
        "the module of egglog's sort a term's argument declares, a library's",
    "dew.nn.kernels.tuning:_module:module":
        "the module a kernel's stored picks name, an op's class or a hook: a launch's records key on the "
        "code of each (`report.launch_code`)",
    "dew.quantization:_qwix:module": "Qwix's modules, a library's",
    "dew.registry:imported:module":
        "a class a record names, a model's: it reaches a kernel's launch only as its op's fields and hooks, "
        "whose code a launch's records key on (`report.launch_code`)",
    "dew.registry:_holds:module": "the base class of a registry's kind, as `dew.registry:imported:module`",
    "dew.registry:import_trusted:package": "a package the run trusts, as `dew.registry:imported:module`",
    "dew.registry:import_trusted:module": "a trusted package's module a record names, as its package",
    "dew.registry:module_from_file:<loader>": "a user's file a run names, the user's code",
    **dict.fromkeys((
        "dew._cache:unintercepted:sys.modules", "dew._devices:_seen:sys.modules",
        "dew._devices:backend_open:sys.modules", "dew._devices:flags_hold:sys.modules",
        "dew._sources:file_of:sys.modules", "dew._sources:function_reads:sys.modules",
        "dew._sources:generated_fields:sys.modules", "dew._sources:own_source:sys.modules",
        "dew._sources:recorded:sys.modules", "dew._sources:source_file:sys.modules",
        "dew._stated:_coded:sys.modules", "dew._stated:_converter:sys.modules",
        "dew._stated:_named:sys.modules", "dew.data.pipeline:_main_importable:sys.modules",
        "dew.data.sources.memory:_table_records:sys.modules"),
        "a module this process already imported, a library's, the user's or the one defining a value it "
        "holds, read by its variables or the name that value states: it imports none of Dew's"),
}
"""Each dynamic import site of Dew's (`module:function:expression`) whose code a closure need not follow,
and why: a changed or added site is another site, unreviewed until it is."""

SELECTED = {"dew.quantization:_complex_in_float.get_intercept_map:module": "UNQUANTIZED"}
"""Each dynamic import site that imports the modules of the dotted names a literal table of its module holds
(`module.function` keys), which a closure follows as it follows a stated import."""

_IMPORTERS = frozenset({"import_module", "__import__"})
_LOADERS = frozenset({"spec_from_file_location", "exec_module", "run_path", "run_module"})


@functools.cache
def code_digest(*roots: str) -> str | None:
    """The digest of the modules `roots` name and every module of Dew's their code can import (`closure`),
    each by its path and contents (`_snapshot`), the release's version (`dew._version`) not read: what a
    record of what those modules computed keys on. The whole package's (`package_digest`) where the
    closure holds a dynamic import site not `REVIEWED`, which is logged: its code could import any module.
    None where Dew's source is not the one this process first read (a checkout edited while it ran), whose
    code no digest of the files states: such a process keeps and reads no record, logged. Read once a
    process."""
    try:
        modules, sites = _kept_closure(roots)
        current = _listed() == _snapshot()
    except _Changed:
        current = False
    if not current:
        _log.warning("Dew's source changed while this process ran: it keeps no record of %s",
                     ", ".join(roots))
        return None
    unreviewed = sorted(set(sites) - REVIEWED.keys())
    if unreviewed:
        _log.warning("the code of %s imports by computed names at %s, none reviewed (dew._code.REVIEWED): "
                     "its records are keyed by all of Dew's code", ", ".join(roots), ", ".join(unreviewed))
        return package_digest()
    files = _snapshot()
    digest = hashlib.sha256()
    for module in sorted(modules):
        relative, contents = files[module]
        if relative != f"dew/{VERSION_FILE}":
            digest.update(f"{relative}\0{contents}\0".encode())
    return digest.hexdigest()


def defining_modules(values: Sequence[object]) -> frozenset[str]:
    """The modules of Dew's that define `values`' code: a class's own module, any other value's class's;
    none for a library's value or the user's own, whose release or source states it."""
    found = set()
    for value in values:
        module = (value if isinstance(value, type) else type(value)).__module__
        if isinstance(module, str) and _path(module) is not None:
            found.add(module)
    return frozenset(found)


class _Changed(Exception):
    """A source file is not the one this process's snapshot of Dew's source holds (`_snapshot`)."""


def _listed() -> dict[str, tuple[str, str]]:
    """Each module of Dew's packages as its source is now: its file's path under `SOURCE` and the digest of
    its contents, by its name."""
    files = {}
    for package in PACKAGES:
        for path in sorted((SOURCE / package).rglob("*.py")):
            relative = path.relative_to(SOURCE)
            name = ".".join(relative.with_suffix("").parts).removesuffix(".__init__")
            files[name] = (relative.as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
    return files


@functools.cache
def _snapshot() -> dict[str, tuple[str, str]]:
    """Dew's source as this process first read it (`_listed`): which modules exist, where, and what each
    holds. A closure is walked from it (`_path`, `_source`) and a digest computed from it, one tree
    throughout, however the files change after."""
    return _listed()


def _kept_closure(roots: tuple[str, ...]) -> tuple[frozenset[str], tuple[str, ...]]:
    """`closure(*roots)`, kept in Dew's cache under the digest of the source it was walked from
    (`_snapshot`), the Python that parsed it and the roots, so a later start of the same tree reads it,
    walking nothing; one that does not name its roots, or names a module the tree has not, is walked
    again."""
    tree = hashlib.sha256(json.dumps(sorted(_snapshot().values())).encode()).hexdigest()
    named = hashlib.sha256(json.dumps(roots).encode()).hexdigest()[:16]
    python = "{}.{}".format(*sys.version_info[:2])
    kept = Path(dew_cache_dir()) / "code" / f"{tree}-{python}-{named}.json"
    with contextlib.suppress(FileNotFoundError, ValueError):
        found = json.loads(kept.read_text())
        if _sound(found, roots):
            return frozenset(found["modules"]), tuple(found["sites"])
    modules, sites = closure(*roots)
    kept.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(kept, json.dumps({"roots": list(roots), "modules": sorted(modules),
                                       "sites": list(sites)}))
    return modules, sites


def _sound(found: JSON, roots: tuple[str, ...]) -> bool:
    """Whether `found` is a closure of `roots` as `_kept_closure` keeps one."""
    if not isinstance(found, dict) or found.get("roots") != list(roots):
        return False
    modules, sites = found.get("modules"), found.get("sites")
    return (isinstance(modules, list) and isinstance(sites, list)
            and all(isinstance(module, str) and _path(module) is not None for module in modules)
            and set(roots) <= set(modules) and all(isinstance(site, str) for site in sites))


@functools.cache
def closure(*roots: str) -> tuple[frozenset[str], tuple[str, ...]]:
    """The modules of Dew's `roots` name and their code can import, and the dynamic import sites among
    them (`module:function:expression`). A root naming no module of Dew's is refused."""
    absent = [root for root in roots if _path(root) is None]
    if absent:
        raise LookupError(f"no module of Dew's is named {', '.join(absent)}")
    seen: set[str] = set()
    sites: set[str] = set()
    pending = list(roots)
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        imported, found = _imports(module)
        sites.update(found)
        pending.extend(imported - seen)
    return frozenset(seen), tuple(sorted(sites))


def _path(module: str) -> Path | None:
    """The source file of Dew's `module` in this process's snapshot of the tree (`_snapshot`), None where
    it holds none (a library's, an absent one)."""
    found = _snapshot().get(module)
    return None if found is None else SOURCE / found[0]


def _ours(module: str) -> set[str]:
    """`module` and each package above it, those of Dew's alone: what importing it runs."""
    parts = module.split(".")
    return {name for name in (".".join(parts[:end]) for end in range(1, len(parts) + 1))
            if _path(name) is not None}


@functools.cache
def _source(module: str) -> bytes:
    """`module`'s source as this process's snapshot of the tree holds it (`_snapshot`); `_Changed` where
    its file holds another now."""
    path = _path(module)
    if path is None:
        raise LookupError(f"no module of Dew's is named {module}")
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != _snapshot()[module][1]:
        raise _Changed(f"{path} changed after this process first read Dew's source")
    return contents


@functools.cache
def _tree(module: str) -> ast.Module:
    tree = ast.parse(_source(module), str(_path(module)))
    tree.body = list(_running(tree, tree.body))
    return tree


def _running(tree: ast.Module, body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """A module's top-level statements that run: an `if TYPE_CHECKING:` (`_guarded`) replaced by its `else`,
    itself so read."""
    for node in body:
        if isinstance(node, ast.If) and _guarded(tree, node):
            yield from _running(tree, node.orelse)
        else:
            yield node


def _guarded(tree: ast.Module, node: ast.If) -> bool:
    """Whether `node`, a statement of a module's top level, is an `if TYPE_CHECKING:`, whose body never
    runs: its test `TYPE_CHECKING` as the module imports it from `typing`, or `typing.TYPE_CHECKING` with
    `typing` imported, each bound by that import alone. One nested deeper, or under any other binding, is
    kept whole, both branches read."""
    bindings = [target for each in ast.walk(tree) for target in _bound(each)]
    if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
        imported = any(isinstance(each, ast.ImportFrom) and each.module == "typing"
                       and any(alias.name == "TYPE_CHECKING" and alias.asname is None for alias in each.names)
                       for each in tree.body)
        return imported and bindings.count("TYPE_CHECKING") == 1
    if ast.unparse(node.test) == "typing.TYPE_CHECKING":
        imported = any(isinstance(each, ast.Import) and any(alias.name == "typing" and alias.asname is None
                                                            for alias in each.names) for each in tree.body)
        return imported and bindings.count("typing") == 1
    return False


def _binds(statement: ast.stmt) -> list[str]:
    """The names a module's top-level `statement` binds."""
    if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        return [name.id for target in targets for name in ast.walk(target) if isinstance(name, ast.Name)]
    return _bound(statement)


def _bound(node: ast.AST) -> list[str]:
    """The names `node` binds: an import's, an assignment's or a loop's targets, a function's arguments."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [alias.asname or alias.name.partition(".")[0] for alias in node.names]
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        return [node.id]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name is not None:
        return [node.name]
    if isinstance(node, ast.MatchMapping) and node.rest is not None:
        return [node.rest]
    if isinstance(node, ast.ExceptHandler) and node.name is not None:
        return [node.name]
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return list(node.names)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    return []


def _literal(node: ast.expr) -> dict[str, str] | list[str] | str | None:
    """The strings, the collection of them or the mapping of names to them a literal states (a dict's
    `**dict.fromkeys(names, value)` among them, a `frozenset({...})`); None for any other expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call) and ast.unparse(node.func) in ("frozenset", "set", "tuple", "list") and len(
            node.args) == 1 and not node.keywords:
        return _literal(node.args[0])
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [_literal(each) for each in node.elts]
        strings = [item for item in items if isinstance(item, str)]
        return strings if len(strings) == len(items) else None
    if not isinstance(node, ast.Dict):
        return None
    held: dict[str, str] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        if key is not None:
            name, target = _literal(key), _literal(value)
            if not isinstance(name, str) or not isinstance(target, str):
                return None
            held[name] = target
        elif (isinstance(value, ast.Call) and ast.unparse(value.func) == "dict.fromkeys"
              and len(value.args) == 2):
            names, target = _literal(value.args[0]), _literal(value.args[1])
            if not isinstance(names, list) or not isinstance(target, str):
                return None
            held.update(dict.fromkeys(names, target))
        elif isinstance(inner := _literal(value), dict):
            held.update(inner)
        else:
            return None
    return held


def _assigned(module: str, name: str) -> ast.expr | None:
    """The value `module` assigns `name` at its top level, None where it assigns none."""
    for node in _tree(module).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        named = any(isinstance(target, ast.Name) and target.id == name for target in targets)
        if named and value is not None:
            return value
    return None


@functools.cache
def _table(package: str) -> dict[str, str] | None:
    """The names lazily exporting `package` resolves (`dew._exports`, its own `__getattr__`), each to the
    module that defines it: its `_EXPORTS` (a name to a module), `_NAMED` (a name to a submodule) and
    `_SUBMODULES` (its submodules by name). Empty for a module that exports lazily none; None where a
    table is no literal, which no reading resolves."""
    path = _path(package)
    if path is None or path.name != "__init__.py":
        return {}
    table: dict[str, str] = {}
    tree = _tree(package)
    # A table the package's dispatcher reads by any other name, or a dispatcher of its own with no table
    # named here, resolves nothing this walk reads: unreadable.
    lookups = [node.args[1] for node in ast.walk(tree) if isinstance(node, ast.Call) and len(node.args) == 2
               and ast.unparse(node.func).rpartition(".")[2].lstrip("_") == "lookup"]
    top = {name for node in tree.body for name in _binds(node)}
    if any(ast.unparse(table) != "_EXPORTS" for table in lookups) or (
            "__getattr__" in top and not lookups and not {"_NAMED", "_SUBMODULES"} & top):
        return None
    for name, read in (("_EXPORTS", lambda target: target), ("_NAMED", lambda target: f"{package}.{target}")):
        value = _assigned(package, name)
        found = None if value is None else _literal(value)
        if value is not None and not isinstance(found, dict):
            return None
        table.update({key: read(target) for key, target in (found or {}).items()})
    value = _assigned(package, "_SUBMODULES")
    members = None if value is None else _literal(value)
    if value is not None and not isinstance(members, list):
        return None
    table.update({member: f"{package}.{member}" for member in members or ()})
    return table


class _Imports(ast.NodeVisitor):
    """What a module's code imports: the modules of Dew's (`modules`) and its dynamic sites (`sites`)."""

    def __init__(self, module: str):
        self.module = module
        path = _path(module)
        is_package = path is not None and path.name == "__init__.py"
        self.package = module if is_package else module.rpartition(".")[0]
        self.modules: set[str] = set()
        self.sites: set[str] = set()
        self.aliases: dict[str, set[str]] = {}
        self.importers: dict[str, set[str]] = {"__import__": {"__import__"}}
        self.within: list[str] = []
        self.read_whole: set[int] = set()
        self.calling: set[int] = set()

    def read(self, module: str) -> None:
        self.modules |= _ours(module)

    def site(self, expression: ast.expr | None) -> None:
        stated = "<loader>" if expression is None else ast.unparse(expression)
        self.sites.add(f"{self.module}:{'.'.join(self.within) or '<module>'}:{stated}")

    def bind(self, name: str, module: str) -> None:
        self.aliases.setdefault(name, set()).add(module)

    def attribute(self, module: str, name: str) -> str | None:
        """The module reading `name` of `module` imports, its submodule or what its lazy table names; None
        where it imports none. A table no literal states is a site."""
        if _path(f"{module}.{name}") is not None:
            return f"{module}.{name}"
        table = _table(module)
        if table is None:
            self.sites.add(f"{module}:<table>:{name}")
            return None
        return table.get(name)

    def whole(self, module: str) -> None:
        """`module` held as a value: every name its lazy table resolves is one its code may read."""
        table = _table(module)
        if table is None:
            self.sites.add(f"{module}:<table>:*")
        for each in (table or {}).values():
            self.read(each)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.within.append(node.name)
        self.generic_visit(node)
        self.within.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.read(alias.name)
            self.bind(alias.asname or alias.name.partition(".")[0],
                      alias.name if alias.asname else alias.name.partition(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = node.module or ""
        if node.level:
            parts = self.package.split(".")
            base = ".".join([*parts[:len(parts) - node.level + 1], *([node.module] if node.module else [])])
        self.read(base)
        for alias in node.names:
            if alias.name == "*":
                self.whole(base)
                continue
            if base in ("importlib", "builtins") and alias.name in _IMPORTERS:
                self.importers.setdefault(alias.asname or alias.name, set()).add(alias.name)
            if base == "sys" and alias.name == "modules":
                self.site(ast.Name("sys.modules"))
            found = self.attribute(base, alias.name)
            if found is not None:
                self.read(found)
                if found == f"{base}.{alias.name}":
                    self.bind(alias.asname or alias.name, found)

    def resolved(self, node: ast.expr) -> set[str]:
        """The modules an expression of names and attributes holds (`dew.algebra.scan`), each module its
        reading imports read; marks its names as read here."""
        self.read_whole.add(id(node))
        if isinstance(node, ast.Name):
            return set(self.aliases.get(node.id, ()))
        if isinstance(node, ast.Call) and self.importing(node):
            self.calling.add(id(node))
            return self.imported(node)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
                and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)):
            node = ast.Attribute(node.args[0], node.args[1].value, ast.Load())
        if not isinstance(node, ast.Attribute):
            return set()
        held = set()
        for module in self.resolved(node.value):
            if module == "sys" and node.attr == "modules":
                self.site(node)
            found = self.attribute(module, node.attr)
            if found is not None:
                self.read(found)
                if found == f"{module}.{node.attr}":
                    held.add(found)
        return held

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _IMPORTERS and id(node) not in self.calling:
            self.site(node)
        if id(node) not in self.read_whole:
            for module in self.resolved(node):
                self.whole(module)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.importers and id(node) not in self.calling:
            self.site(node)
        if id(node) not in self.read_whole and isinstance(node.ctx, ast.Load):
            for module in self.aliases.get(node.id, ()):
                self.whole(module)

    def visit_Assign(self, node: ast.Assign) -> None:
        # A module bound to a name (`package = import_module("dew.data")`): the name reads as the module,
        # and, held as a value, every name of its table may be read.
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(
                node.value, (ast.Name, ast.Attribute, ast.Call)):
            for module in self.resolved(node.value):
                self.bind(node.targets[0].id, module)
                self.whole(module)
        self.generic_visit(node)

    def importing(self, node: ast.Call) -> bool:
        """Whether `node` calls `import_module` or `__import__`, by any name the module binds them to."""
        function = node.func
        return ((isinstance(function, ast.Attribute) and function.attr in _IMPORTERS)
                or (isinstance(function, ast.Name) and function.id in self.importers))

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        called = (function.attr if isinstance(function, ast.Attribute)
                  else function.id if isinstance(function, ast.Name) else "")
        if self.importing(node):
            self.calling.add(id(function))
            if id(node) not in self.calling:
                self.calling.add(id(node))
                # Its module held as a value: every name of its table may be read.
                for module in self.imported(node):
                    self.whole(module)
        elif called in _LOADERS:
            self.site(None)
        elif called == "getattr" and len(node.args) >= 2:
            name = node.args[1]
            stated = name.value if isinstance(name, ast.Constant) and isinstance(name.value, str) else None
            escaped = id(node) not in self.read_whole
            receivers = self.resolved(node.args[0])
            if receivers & {"sys", "importlib", "builtins"} and (stated is None or stated in {
                    "modules", *_IMPORTERS}):
                self.site(node)
            for module in receivers:
                found = None if stated is None else self.attribute(module, stated)
                if stated is None:
                    self.whole(module)
                elif found is not None:
                    self.read(found)
                    if escaped and found == f"{module}.{stated}":
                        self.whole(found)
        self.generic_visit(node)

    def imported(self, node: ast.Call) -> set[str]:
        """An `import_module` or `__import__` call: the modules it may return, the one it names read; none,
        a site, where its name is computed."""
        self.calling.add(id(node.func))
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        name = node.args[0] if node.args else keywords.get("name")
        package = node.args[1] if len(node.args) > 1 else keywords.get("package")
        found = None if name is None else self.stated(name)
        if found is not None and found.startswith("."):
            base = None if package is None else self.stated(package)
            found = None if base is None else self.relative(found, base)
        within = ".".join(self.within) or "<module>"
        site = f"{self.module}:{within}:{'' if name is None else ast.unparse(name)}"
        called = ast.unparse(node.func).rpartition(".")[2]
        kinds = self.importers.get(called, {called}) if isinstance(node.func, ast.Name) else {called}
        returned: set[str] = set()
        if found is not None and "import_module" in kinds:
            returned.add(found)
        if "__import__" in kinds:
            builtin = self.builtin(node, found, keywords)
            found = None if builtin is None else found
            returned |= set() if builtin is None else {builtin}
        if found is not None:
            self.read(found)
            return returned
        selected = self.selected(SELECTED[site]) if site in SELECTED else None
        if selected is None:
            self.site(name)
        for dotted in selected or ():
            self.read(dotted.rpartition(".")[0])
        return set()

    def builtin(self, node: ast.Call, found: str | None, keywords: dict[str | None, ast.expr]) -> str | None:
        """The module `__import__(name, globals, locals, fromlist, level)` returns, `found` its name: with a
        fromlist the module, each of its names read from it as `from` reads one (`*` its whole table),
        else its top-level package; None, a site, for a level other than 0 or a fromlist no literal
        states."""
        level = node.args[4] if len(node.args) > 4 else keywords.get("level")
        fromlist = node.args[3] if len(node.args) > 3 else keywords.get("fromlist")
        absolute = level is None or (isinstance(level, ast.Constant) and level.value == 0)
        if found is None or not absolute:
            return None
        names = [] if fromlist is None else _literal(fromlist)
        if not isinstance(names, list):
            return None
        for name in names:
            if name == "*":
                self.whole(found)
            elif (attribute := self.attribute(found, name)) is not None:
                self.read(attribute)
        return found if names else found.partition(".")[0]

    def selected(self, table: str) -> list[str] | None:
        """The dotted names (`module.function`) the literal table `table` of this module holds, None where
        it holds none as a literal."""
        value = _assigned(self.module, table)
        found = None if value is None else _literal(value)
        names = list(found) if isinstance(found, (dict, list)) else None
        if names is None or not all("." in name for name in names):
            return None
        return names

    def stated(self, node: ast.expr) -> str | None:
        """The string `node` states without running: a constant, `__name__`, `__package__`, or an f-string
        of those and constants."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in ("__name__", "__package__"):
            return self.module if node.id == "__name__" else self.package
        if isinstance(node, ast.JoinedStr):
            parts = [self.stated(each.value) if isinstance(each, ast.FormattedValue) else self.stated(each)
                     for each in node.values]
            return None if None in parts else "".join(part for part in parts if part is not None)
        return None

    def relative(self, name: str, package: str) -> str:
        level = len(name) - len(name.lstrip("."))
        parts = package.split(".")
        return ".".join([*parts[:len(parts) - level + 1], *([name[level:]] if name[level:] else [])])


@functools.cache
def _imports(module: str) -> tuple[frozenset[str], tuple[str, ...]]:
    """The modules of Dew's `module`'s code imports, and its dynamic import sites."""
    found = _Imports(module)
    # Every binding first, so a name its code reads before the statement that binds it (a function's, read
    # at its call) resolves as after it.
    for node in ast.walk(_tree(module)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.visit(node)
    found.visit(_tree(module))
    return frozenset(found.modules - {module}), tuple(sorted(found.sites))

