"""The digest of the code a record of what Dew computed is a function of (`code_digest`): its producer's
modules and every module of Dew's their code can import, read from their source, none imported, so a record
is computed again where that code changed and kept where Dew changed elsewhere (row 10743: every record
keyed by the whole package, any edit certified, verified and decided every one of them again).

A module's imports are its code's every import statement, a function's own among them, but not those of an
`if TYPE_CHECKING:` block, which never runs; a parent package's `__init__` with each; and each name read
from a lazily exporting package (`_EXPORTS`, `_NAMED`, `_SUBMODULES`: `dew._exports`), resolved to the
module that defines it, the whole table where a name is read by a value (`getattr(package, name)`). An
import by a computed name no table resolves (`import_module(name)`) is a dynamic site: each one Dew holds is
reviewed (`REVIEWED`), and a closure holding any other is keyed by the whole package instead
(`dew._cache.package_digest`), so a site added unreviewed costs warm records, never a stale one.

A closure is walked once a tree (its whole package's digest) and kept in Dew's cache, so a start reads its
files' bytes, not their syntax: a walk parses a kernel's 225 files in about 1.5 s."""

import ast
import contextlib
import functools
import hashlib
import json
import logging
import sys
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
    "dew._exports:lookup.__getattr__":
        "a lazily exporting package's `_EXPORTS`, each name a closure reads resolved to its module",
    "dew.nn.kernels:__getattr__": "the package's `_NAMED`, each name a closure reads resolved to its module",
    "dew.algebra:__getattr__":
        "the package's `_SUBMODULES` and `_NAMED`, each name a closure reads resolved to its module",
    "dew._devices:cublas_version": "JAX's CUDA plugin's version reader, a library's code",
    "dew.algebra.terms:_sort": "egglog's sort a term's argument declares, by its class's module",
    "dew.nn.kernels.tuning:_module":
        "the module a kernel's stored picks name (an op's class, a hook): it rebuilds a pick, and a record "
        "of that op's launch is keyed by the kernels' code (`report.kernels_code`)",
    "dew.quantization:_qwix": "Qwix's modules, a library's code",
    "dew.quantization:_complex_in_float.get_intercept_map":
        "the functions `UNQUANTIZED` names, each run where a model's code calls it, which imports it",
    "dew.registry:imported": "a class or function a record names, a model's: a record reading it states it",
    "dew.registry:_holds": "the base class a registry's kind names, a class a record names derives from",
    "dew.registry:import_trusted": "a package the run trusts and the modules of its a record names",
    "dew.registry:module_from_file": "a user's file a run names, the user's code",
    "dew.data.sources.codec:_dataclass_decoded": "a dataclass of Dew's an artifact names, rebuilt as data",
    "dew.interop.parts:_known_class": "a library's class a checkpoint's parts name (transformers')",
}
"""Each dynamic import site of Dew's (`module:function`) whose code a closure need not follow, and why."""

_IMPORTERS = frozenset({"import_module", "__import__"})
_LOADERS = frozenset({"spec_from_file_location", "exec_module", "run_path", "run_module"})


def code_digest(*roots: str) -> str:
    """The digest of the modules `roots` name and every module of Dew's their code can import (`closure`),
    each by its path and contents, the release's version (`dew._version`) not read: what a record of what
    those modules computed keys on. The whole package's (`package_digest`) where the closure holds a
    dynamic import site not `REVIEWED`, which is logged: its code could import any module."""
    modules, sites = _kept_closure(roots)
    unreviewed = sorted(set(sites) - REVIEWED.keys())
    if unreviewed:
        _log.warning("the code of %s imports by computed names at %s, none reviewed (dew._code.REVIEWED): "
                     "its records are keyed by all of Dew's code", ", ".join(roots), ", ".join(unreviewed))
        return package_digest()
    digest = hashlib.sha256()
    for path in sorted(path for path in map(_path, modules) if path is not None):
        relative = path.relative_to(SOURCE)
        if relative != Path("dew", VERSION_FILE):
            digest.update(relative.as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


@functools.cache
def _kept_closure(roots: tuple[str, ...]) -> tuple[frozenset[str], tuple[str, ...]]:
    """`closure(*roots)`, kept in Dew's cache under the whole package's digest and the Python that parsed
    it, so a later start of the same tree reads it, walking nothing; walked where none is kept."""
    python = "{}.{}".format(*sys.version_info[:2])
    kept = Path(dew_cache_dir()) / "code" / f"{package_digest()}-{python}.json"
    held: dict[str, object] = {}
    with contextlib.suppress(FileNotFoundError, ValueError):
        if isinstance(read := json.loads(kept.read_text()), dict):
            held = read
    found = held.get(json.dumps(roots))
    if isinstance(found, list) and len(found) == 2:
        return frozenset(found[0]), tuple(found[1])
    modules, sites = closure(*roots)
    held[json.dumps(roots)] = [sorted(modules), list(sites)]
    kept.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(kept, json.dumps(held, sort_keys=True))
    return modules, sites


@functools.cache
def closure(*roots: str) -> tuple[frozenset[str], tuple[str, ...]]:
    """The modules of Dew's `roots` name and their code can import, and the dynamic import sites among
    them (`module:function`). A root naming no module of Dew's is refused."""
    absent = [root for root in roots if _path(root) is None]
    if absent:
        raise LookupError(f"no module of Dew's is named {', '.join(absent)}")
    seen: set[str] = set()
    sites: list[str] = []
    pending = list(roots)
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        imported, found = _imports(module)
        sites.extend(found)
        pending.extend(imported - seen)
    return frozenset(seen), tuple(sorted(set(sites)))


def every_module(package: str) -> tuple[str, ...]:
    """Each module of Dew's `package` holds, itself and every module beneath it, by name."""
    path = _path(package)
    if path is None or path.name != "__init__.py":
        raise LookupError(f"no package of Dew's is named {package}")
    return tuple(sorted(".".join(found.relative_to(SOURCE).with_suffix("").parts).removesuffix(".__init__")
                        for found in path.parent.rglob("*.py")))


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
    """A module's tree without its `if TYPE_CHECKING:` blocks, which never run (their `else` kept)."""

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt] | None:
        if ast.unparse(node.test) in ("TYPE_CHECKING", "typing.TYPE_CHECKING"):
            return [self.visit(each) for each in node.orelse] or None
        return self.generic_visit(node)


@functools.cache
def _tree(module: str) -> ast.Module:
    path = _path(module)
    if path is None:
        raise LookupError(f"no module of Dew's is named {module}")
    tree = ast.parse(path.read_bytes(), str(path))
    _Running().generic_visit(tree)
    return tree


@functools.cache
def _table(package: str) -> dict[str, str]:
    """The names lazily exporting `package` resolves (`dew._exports`, its own `__getattr__`), each to the
    module that defines it: its `_EXPORTS` (a name to a module), `_NAMED` (a name to a submodule) and
    `_SUBMODULES` (its submodules by name). Empty for any other module."""
    path = _path(package)
    if path is None or path.name != "__init__.py":
        return {}
    table: dict[str, str] = {}
    for node in _tree(package).body:
        if not (isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.expr)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = {target.id for target in targets if isinstance(target, ast.Name)}
        value = ast.literal_eval(node.value) if names & {"_EXPORTS", "_NAMED"} else None
        if "_EXPORTS" in names and isinstance(value, dict):
            table.update(value)
        elif "_NAMED" in names and isinstance(value, dict):
            table.update({name: f"{package}.{module}" for name, module in value.items()})
        elif "_SUBMODULES" in names and isinstance(node.value, ast.Call) and node.value.args:
            table.update({name: f"{package}.{name}" for name in ast.literal_eval(node.value.args[0])})
    return table


def _attribute(module: str, name: str) -> str | None:
    """The module reading `name` of `module` imports: its submodule of that name, or the module its lazy
    table names; None where it imports none."""
    if _path(f"{module}.{name}") is not None:
        return f"{module}.{name}"
    return _table(module).get(name)


def _constant(node: ast.expr, module: str) -> str | None:
    """The module name `node` states without running: a string, or `f"{__name__}.<name>"`."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values:
        first, *rest = node.values
        if (isinstance(first, ast.FormattedValue) and isinstance(first.value, ast.Name)
                and first.value.id == "__name__" and all(isinstance(each, ast.Constant) for each in rest)):
            return module + "".join(str(each.value) for each in rest if isinstance(each, ast.Constant))
    return None


class _Imports(ast.NodeVisitor):
    """What a module's code imports: the modules of Dew's (`modules`) and its dynamic sites (`sites`)."""

    def __init__(self, module: str):
        self.module = module
        path = _path(module)
        is_package = path is not None and path.name == "__init__.py"
        self.package = module if is_package else module.rpartition(".")[0]
        self.modules: set[str] = set()
        self.sites: list[str] = []
        self.aliases: dict[str, str] = {}
        self.within: list[str] = []

    def read(self, module: str) -> None:
        self.modules |= _ours(module)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.within.append(node.name)
        self.generic_visit(node)
        self.within.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.read(alias.name)
            bound = alias.name if alias.asname else alias.name.partition(".")[0]
            self.aliases[alias.asname or bound] = bound

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = node.module or ""
        if node.level:
            parts = self.package.split(".")[:len(self.package.split(".")) - node.level + 1]
            base = ".".join([*parts, *([node.module] if node.module else [])])
        self.read(base)
        for alias in node.names:
            if alias.name == "*":
                for module in _table(base).values():
                    self.read(module)
                continue
            found = _attribute(base, alias.name)
            if found is not None:
                self.read(found)
                if found == f"{base}.{alias.name}":
                    self.aliases[alias.asname or alias.name] = found

    def visit_Attribute(self, node: ast.Attribute) -> None:
        chain = [node.attr]
        cursor: ast.expr = node.value
        while isinstance(cursor, ast.Attribute):
            chain.append(cursor.attr)
            cursor = cursor.value
        if isinstance(cursor, ast.Name) and cursor.id in self.aliases:
            current: str | None = self.aliases[cursor.id]
            for name in reversed(chain):
                found = _attribute(current, name) if current is not None else None
                if found is None:
                    break
                self.read(found)
                current = found if found == f"{current}.{name}" else None
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
        if name in _IMPORTERS and node.args:
            found = _constant(node.args[0], self.module)
            if found is None:
                self.sites.append(f"{self.module}:{'.'.join(self.within) or '<module>'}")
            else:
                self.read(found)
        elif name in _LOADERS:
            self.sites.append(f"{self.module}:{'.'.join(self.within) or '<module>'}")
        elif (name == "getattr" and len(node.args) >= 2 and isinstance(node.args[0], ast.Name)
              and node.args[0].id in self.aliases):
            package = self.aliases[node.args[0].id]
            attribute = _constant(node.args[1], "") if isinstance(node.args[1], ast.Constant) else None
            if attribute is not None:
                found = _attribute(package, attribute)
                if found is not None:
                    self.read(found)
            else:
                for module in _table(package).values():
                    self.read(module)
        self.generic_visit(node)


@functools.cache
def _imports(module: str) -> tuple[frozenset[str], tuple[str, ...]]:
    """The modules of Dew's `module`'s code imports, and its dynamic import sites."""
    found = _Imports(module)
    found.visit(_tree(module))
    return frozenset(found.modules - {module}), tuple(found.sites)
