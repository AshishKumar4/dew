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

A closure is walked once a tree (its whole package's digest) and kept in Dew's cache, so a start reads its
files' bytes, not their syntax: a walk parses a kernel's 225 files in about 1.5 s."""

import ast
import contextlib
import functools
import hashlib
import json
import logging
import sys
import types
from collections.abc import Iterable
from pathlib import Path

from dew._builds import dew_cache_dir
from dew._cache import VERSION_FILE, package_digest
from dew._files import write_atomically

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
}
"""Each dynamic import site of Dew's (`module:function:expression`) whose code a closure need not follow,
and why: a changed or added site is another site, unreviewed until it is."""

SELECTED = {"dew.quantization:_complex_in_float.get_intercept_map:module": "UNQUANTIZED"}
"""Each dynamic import site that imports the modules of the dotted names a literal table of its module holds
(`module.function` keys), which a closure follows as it follows a stated import."""

_IMPORTERS = frozenset({"import_module", "__import__"})
_LOADERS = frozenset({"spec_from_file_location", "exec_module", "run_path", "run_module"})


@functools.cache
def code_digest(*roots: str) -> str:
    """The digest of the modules `roots` name and every module of Dew's their code can import (`closure`),
    each by its path and contents, the release's version (`dew._version`) not read: what a record of what
    those modules computed keys on. The whole package's (`package_digest`) where the closure holds a
    dynamic import site not `REVIEWED`, which is logged: its code could import any module. Read once a
    process, as the whole package's is."""
    modules, sites = _kept_closure(roots)
    unreviewed = sorted(set(sites) - REVIEWED.keys())
    if unreviewed:
        _log.warning("the code of %s imports by computed names at %s, none reviewed (dew._code.REVIEWED): "
                     "its records are keyed by all of Dew's code", ", ".join(roots), ", ".join(unreviewed))
        return package_digest()
    digest = hashlib.sha256()
    for module in sorted(modules):
        path = _path(module)
        if path is None:
            raise LookupError(f"{module}, which the code of {', '.join(roots)} imports, has no source file")
        relative = path.relative_to(SOURCE)
        if relative != Path("dew", VERSION_FILE):
            digest.update(relative.as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def defining_modules(values: Iterable[object]) -> frozenset[str]:
    """The modules of Dew's that define `values`' code, by what each is: a function's or a class's module,
    a partial's or a bound method's function's, any other object's class's; none for a library's value or
    the user's own code, whose release or source states it."""
    found = set()
    for value in values:
        while isinstance(value, functools.partial):
            value = value.func
        if isinstance(value, types.MethodType):
            value = value.__func__
        module = (value if isinstance(value, (types.FunctionType, type)) else type(value)).__module__
        if isinstance(module, str) and _path(module) is not None:
            found.add(module)
    return frozenset(found)


def _kept_closure(roots: tuple[str, ...]) -> tuple[frozenset[str], tuple[str, ...]]:
    """`closure(*roots)`, kept in Dew's cache under the whole package's digest, the Python that parsed it
    and the roots, so a later start of the same tree reads it, walking nothing. A walk is kept only where
    the package's files were those the digest names throughout it; a kept closure that does not name its
    roots, or names a module with no file, is walked again."""
    whole = package_digest()
    named = hashlib.sha256(json.dumps(roots).encode()).hexdigest()[:16]
    python = "{}.{}".format(*sys.version_info[:2])
    kept = Path(dew_cache_dir()) / "code" / f"{whole}-{python}-{named}.json"
    with contextlib.suppress(FileNotFoundError, ValueError):
        found = json.loads(kept.read_text())
        if _sound(found, roots):
            return frozenset(found["modules"]), tuple(found["sites"])
    unchanged = package_digest.__wrapped__() == whole
    modules, sites = closure(*roots)
    if unchanged and package_digest.__wrapped__() == whole:
        kept.parent.mkdir(parents=True, exist_ok=True)
        write_atomically(kept, json.dumps({"roots": list(roots), "modules": sorted(modules),
                                           "sites": list(sites)}))
    return modules, sites


def _sound(found: object, roots: tuple[str, ...]) -> bool:
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


@functools.cache
def _path(module: str) -> Path | None:
    """The source file of Dew's `module`, None where none is (a library's, an absent one)."""
    if module.partition(".")[0] not in PACKAGES:
        return None
    base = SOURCE.joinpath(*module.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def _ours(module: str) -> set[str]:
    """`module` and each package above it, those of Dew's alone: what importing it runs."""
    parts = module.split(".")
    return {name for name in (".".join(parts[:end]) for end in range(1, len(parts) + 1))
            if _path(name) is not None}


class _Running(ast.NodeTransformer):
    """A module's tree without its `if TYPE_CHECKING:` blocks, which never run, their `else` kept: the guard
    `typing.TYPE_CHECKING`, or the name `TYPE_CHECKING` where the module imports it from `typing`."""

    def __init__(self, guards: set[str]):
        self.guards = guards

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt] | None:
        test = node.test
        guarded = isinstance(test, ast.Name) and test.id in self.guards
        if guarded or ast.unparse(test) == "typing.TYPE_CHECKING":
            kept: list[ast.stmt] = []
            for statement in node.orelse:
                visited = self.visit(statement)
                kept.extend(visited if isinstance(visited, list) else [] if visited is None else [visited])
            return kept or None
        return self.generic_visit(node)


@functools.cache
def _tree(module: str) -> ast.Module:
    path = _path(module)
    if path is None:
        raise LookupError(f"no module of Dew's is named {module}")
    tree = ast.parse(path.read_bytes(), str(path))
    guards = {alias.asname or alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
              and node.module == "typing" for alias in node.names if alias.name == "TYPE_CHECKING"}
    _Running(guards).generic_visit(tree)
    return tree


def _literal(node: ast.expr) -> object:
    """The value a literal states: Python's literals, and a mapping's `**dict.fromkeys(names, value)`."""
    if isinstance(node, ast.Dict) and None in node.keys:
        held: dict = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if key is not None:
                held[ast.literal_eval(key)] = _literal(value)
            elif (isinstance(value, ast.Call) and ast.unparse(value.func) == "dict.fromkeys"
                  and len(value.args) == 2):
                held.update(dict.fromkeys(ast.literal_eval(value.args[0]), ast.literal_eval(value.args[1])))
            else:
                held.update(_literal(value))
        return held
    return ast.literal_eval(node)


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
    try:
        if (exports := _assigned(package, "_EXPORTS")) is not None:
            table.update(_literal(exports))
        if (named := _assigned(package, "_NAMED")) is not None:
            table.update({name: f"{package}.{module}" for name, module in _literal(named).items()})
        if (submodules := _assigned(package, "_SUBMODULES")) is not None:
            called = isinstance(submodules, ast.Call) and submodules.args
            members = submodules.args[0] if isinstance(submodules, ast.Call) and called else submodules
            table.update({name: f"{package}.{name}" for name in _literal(members)})
    except (ValueError, TypeError, AttributeError):
        return None
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
        self.importers = {"__import__"}
        self.within: list[str] = []
        self.read_whole: set[int] = set()

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
            if base == "importlib" and alias.name in _IMPORTERS:
                self.importers.add(alias.asname or alias.name)
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
        if not isinstance(node, ast.Attribute):
            return set()
        held = set()
        for module in self.resolved(node.value):
            found = self.attribute(module, node.attr)
            if found is not None:
                self.read(found)
                if found == f"{module}.{node.attr}":
                    held.add(found)
        return held

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if id(node) not in self.read_whole:
            for module in self.resolved(node):
                self.whole(module)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if id(node) not in self.read_whole and isinstance(node.ctx, ast.Load):
            for module in self.aliases.get(node.id, ()):
                self.whole(module)

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        called = (function.attr if isinstance(function, ast.Attribute)
                  else function.id if isinstance(function, ast.Name) else "")
        if called in _IMPORTERS or (isinstance(function, ast.Name) and called in self.importers):
            self.imported(node)
        elif called in _LOADERS:
            self.site(None)
        elif called == "getattr" and len(node.args) >= 2:
            self.read_whole.add(id(node.args[0]))
            name = node.args[1]
            for module in self.resolved(node.args[0]):
                if isinstance(name, ast.Constant) and isinstance(name.value, str):
                    if (found := self.attribute(module, name.value)) is not None:
                        self.read(found)
                else:
                    self.whole(module)
        self.generic_visit(node)

    def imported(self, node: ast.Call) -> None:
        """An `import_module` or `__import__` call: the module it names, or a site."""
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        name = node.args[0] if node.args else keywords.get("name")
        package = node.args[1] if len(node.args) > 1 else keywords.get("package")
        found = None if name is None else self.stated(name)
        if found is not None and found.startswith("."):
            base = None if package is None else self.stated(package)
            found = None if base is None else self.relative(found, base)
        within = ".".join(self.within) or "<module>"
        site = f"{self.module}:{within}:{'' if name is None else ast.unparse(name)}"
        if found is not None:
            self.read(found)
        elif site in SELECTED:
            for dotted in _literal(_assigned(self.module, SELECTED[site]) or ast.Dict([], [])):
                self.read(dotted.rpartition(".")[0])
        else:
            self.site(name)

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

