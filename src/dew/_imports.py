"""The modules of a source tree a module's code can import (`Walk`), read from its source, none imported:
every import statement, a function's own and a class body's among them, but not those of a top-level `if
TYPE_CHECKING:`, which never runs; each parent package's `__init__`; each `import_module` of a name its code
states; each name read from a lazily exporting package (`_EXPORTS`, `_NAMED`, `_SUBMODULES`:
`dew._exports`), resolved to the module that defines it through each package that exports it in turn, or
the package's whole table where its code holds the package as a value; and each dynamic import site, an
import by a name its code computes, named `module:function:expression`. The one reader of a lazy package's
`_EXPORTS` (`exported`) and of the imports a module's run executes: `dew._code` keys records on it,
tools/lint_layers.py holds Dew's layers by it, tools/gen_exports.py writes a lazy package's blocks from it.
It reads the standard library alone, so a tool loads it by its path without importing Dew."""

import ast
from collections.abc import Callable, Iterator, Mapping

_IMPORTERS = frozenset({"import_module", "__import__"})
_LOADERS = frozenset({"spec_from_file_location", "exec_module", "run_path", "run_module"})


class Walk:
    """A source tree's import closures: `files`, each of its modules' file by name (a package's its
    `__init__.py`), `read`, a module's source, and `selected`, each dynamic site a literal table of its module
    resolves (`module:function:expression` to the table's name, whose dotted names it imports)."""

    def __init__(self, files: Mapping[str, str], read: Callable[[str], bytes],
                 selected: Mapping[str, str] | None = None):
        self.files, self.read, self.selected = files, read, dict(selected or {})
        self._parsed: dict[str, ast.Module] = {}
        self._trees: dict[str, ast.Module] = {}
        self._tables: dict[str, dict[str, str] | None] = {}
        self._imports: dict[str, tuple[frozenset[str], tuple[str, ...]]] = {}

    def holds(self, module: str) -> bool:
        return module in self.files

    def is_package(self, module: str) -> bool:
        return self.files.get(module, "").endswith("__init__.py")

    def ours(self, module: str) -> set[str]:
        """`module` and each package above it, those the tree holds alone: what importing it runs."""
        parts = module.split(".")
        prefixes = (".".join(parts[:end]) for end in range(1, len(parts) + 1))
        return {name for name in prefixes if name in self.files}

    def parsed(self, module: str) -> ast.Module:
        """`module`'s whole tree, its TYPE_CHECKING blocks among it, parsed once a walk: what a type checker
        reads of it (tools/lint_layers.py's named imports), and what `tree` prunes."""
        parsed = self._parsed.get(module)
        if parsed is None:
            parsed = self._parsed[module] = ast.parse(self.read(module), self.files[module])
        return parsed

    def tree(self, module: str) -> ast.Module:
        """`module`'s tree as its run executes it: `parsed`, its top-level TYPE_CHECKING blocks pruned
        (`running`)."""
        tree = self._trees.get(module)
        if tree is None:
            whole = self.parsed(module)
            pruned = list(running(whole, whole.body))
            tree = self._trees[module] = ast.Module(body=pruned, type_ignores=whole.type_ignores)
        return tree

    def assigned(self, module: str, name: str) -> ast.expr | None:
        """The value `module` assigns `name` at its top level, None where it assigns none."""
        return assigned(self.tree(module), name)

    def table(self, package: str) -> dict[str, str] | None:
        """The names lazily exporting `package` resolves (`dew._exports`, its own `__getattr__`), each to the
        module that defines it: its `_EXPORTS` (a name to a module), `_NAMED` (a name to a submodule) and
        `_SUBMODULES` (its submodules by name). Empty for a module that exports lazily none; None where a
        table is no literal, which no reading resolves."""
        if package not in self._tables:
            self._tables[package] = table(self.tree(package), package) if self.is_package(package) else {}
        return self._tables[package]

    def imports(self, module: str) -> tuple[frozenset[str], tuple[str, ...]]:
        """The modules of the tree `module`'s code imports, and its dynamic import sites, read once a walk."""
        if module in self._imports:
            return self._imports[module]
        found = _Imports(self, module)
        # Every binding first, so a name its code reads before the statement that binds it (a function's,
        # read at its call) resolves as after it.
        tree = self.tree(module)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                found.visit(node)
        found.visit(tree)
        self._imports[module] = frozenset(found.modules - {module}), tuple(sorted(found.sites))
        return self._imports[module]

    def closure(self, *roots: str) -> tuple[frozenset[str], tuple[str, ...]]:
        """The modules `roots` name and their code can import, and the dynamic import sites among them. A
        root the tree holds no module of is refused. The syntax trees it parsed are let go once it is done,
        the imports and tables it read of them kept: a process's walk (`dew._code`'s) held every module's
        tree past it (1.1M of a cold recurrent step's 4.7M live objects, its full collection 3.9 s and
        its exit 18 s, row 8420)."""
        absent = [root for root in roots if root not in self.files]
        if absent:
            raise LookupError(f"no module of the tree is named {', '.join(absent)}")
        seen: set[str] = set()
        sites: set[str] = set()
        pending = list(roots)
        while pending:
            module = pending.pop()
            if module in seen:
                continue
            seen.add(module)
            imported, found = self.imports(module)
            sites.update(found)
            pending.extend(imported - seen)
        self._parsed.clear()
        self._trees.clear()
        return frozenset(seen), tuple(sorted(sites))


def resolved(name: str, package: str) -> str:
    """The module a relative `name` (`..sibling`, `.`) names from `package`, the package its module's
    relative imports start from; an absolute `name` itself. The one resolver of a relative import: a `from`
    statement's (`imported_from`), an `import_module`'s, tools/lint_scopes.py's."""
    level = len(name) - len(name.lstrip("."))
    if not level:
        return name
    parts = package.split(".") if package else []
    return ".".join([*parts[:len(parts) - level + 1], *([name[level:]] if name[level:] else [])])


def imported_from(node: ast.ImportFrom, package: str) -> str:
    """The module a `from` statement reads, a relative one resolved from `package` (`resolved`)."""
    return resolved("." * node.level + (node.module or ""), package)


def assigned(tree: ast.Module, name: str) -> ast.expr | None:
    """The value `tree` assigns `name` at its top level, None where it assigns none."""
    for node in tree.body:
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


def exported(tree: ast.Module) -> dict[str, str]:
    """A lazy package's `_EXPORTS` read from its `__init__.py`'s tree, the literal dict of each name to the
    absolute module that defines it (`dew._exports.lookup`), or to the package's own module of that name;
    empty where it assigns none; refused (ValueError) where no literal states it. The one reader of the
    declaration: tools/gen_exports.py writes a package's blocks from it, and `Walk` follows the imports a
    lookup of each name runs."""
    value = assigned(tree, "_EXPORTS")
    found = {} if value is None else literal(value)
    if not isinstance(found, dict):
        raise ValueError("_EXPORTS is no literal mapping of names to modules")
    return found


def table(tree: ast.Module, package: str) -> dict[str, str] | None:
    """`Walk.table` of `package`'s `__init__` `tree`."""
    # A table the package's dispatcher reads by any other name, or a dispatcher of its own with no table
    # named here, resolves nothing this walk reads: unreadable.
    lookups = [node.args[1] for node in ast.walk(tree) if isinstance(node, ast.Call) and len(node.args) == 2
               and ast.unparse(node.func).rpartition(".")[2].lstrip("_") == "lookup"]
    top = {name for node in tree.body for name in _binds(node)}
    if any(ast.unparse(each) != "_EXPORTS" for each in lookups) or (
            "__getattr__" in top and not lookups and not {"_NAMED", "_SUBMODULES"} & top):
        return None
    value = assigned(tree, "_EXPORTS")
    names = {} if value is None else literal(value)
    value = assigned(tree, "_NAMED")
    named = {} if value is None else literal(value)
    if not isinstance(names, dict) or not isinstance(named, dict):
        return None
    value = assigned(tree, "_SUBMODULES")
    members = None if value is None else literal(value)
    if value is not None and not isinstance(members, list):
        return None
    return {**names, **{key: f"{package}.{target}" for key, target in named.items()},
            **{member: f"{package}.{member}" for member in members or ()}}


def running(tree: ast.Module, body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """A module's top-level statements that run: an `if TYPE_CHECKING:` (`_guarded`) replaced by its `else`,
    itself so read."""
    for node in body:
        if isinstance(node, ast.If) and _guarded(tree, node):
            yield from running(tree, node.orelse)
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


def literal(node: ast.expr) -> dict[str, str] | list[str] | str | None:
    """The strings, the collection of them or the mapping of names to them a literal states (a dict's
    `**dict.fromkeys(names, value)` among them, a `frozenset({...})`); None for any other expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call) and ast.unparse(node.func) in ("frozenset", "set", "tuple", "list") and len(
            node.args) == 1 and not node.keywords:
        return literal(node.args[0])
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        members = [literal(each) for each in node.elts]
        strings = [member for member in members if isinstance(member, str)]
        return strings if len(strings) == len(members) else None
    if not isinstance(node, ast.Dict):
        return None
    held: dict[str, str] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        if key is not None:
            name, target = literal(key), literal(value)
            if not isinstance(name, str) or not isinstance(target, str):
                return None
            held[name] = target
        elif (isinstance(value, ast.Call) and ast.unparse(value.func) == "dict.fromkeys"
              and len(value.args) == 2):
            names, target = literal(value.args[0]), literal(value.args[1])
            if not isinstance(names, list) or not isinstance(target, str):
                return None
            held.update(dict.fromkeys(names, target))
        elif isinstance(inner := literal(value), dict):
            held.update(inner)
        else:
            return None
    return held


class _Imports(ast.NodeVisitor):
    """What a module's code imports: the modules of Dew's (`modules`) and its dynamic sites (`sites`)."""

    def __init__(self, walk: "Walk", module: str):
        self.walk, self.module = walk, module
        self.package = module if walk.is_package(module) else module.rpartition(".")[0]
        self.modules: set[str] = set()
        self.sites: set[str] = set()
        self.aliases: dict[str, set[str]] = {}
        self.importers: dict[str, set[str]] = {"__import__": {"__import__"}}
        self.within: list[str] = []
        self.read_whole: set[int] = set()
        self.calling: set[int] = set()

    def read(self, module: str) -> None:
        self.modules |= self.walk.ours(module)

    def site(self, expression: ast.expr | None) -> None:
        stated = "<loader>" if expression is None else ast.unparse(expression)
        self.sites.add(f"{self.module}:{'.'.join(self.within) or '<module>'}:{stated}")

    def bind(self, name: str, module: str) -> None:
        self.aliases.setdefault(name, set()).add(module)

    def attribute(self, module: str, name: str) -> str | None:
        """The module reading `name` of `module` imports, its submodule or what its lazy table names, each
        module a lazy package's export of it is read through read (`dew._exports.lookup`'s `getattr` reads a
        package's export of the name in turn); None where it imports none. A table no literal states is a
        site."""
        if self.walk.holds(f"{module}.{name}"):
            return f"{module}.{name}"
        seen: set[str] = set()
        found = None
        while module not in seen:
            seen.add(module)
            table = self.walk.table(module)
            if table is None:
                self.sites.add(f"{module}:<table>:{name}")
                return found
            target = table.get(name)
            if target is None:
                return found
            if found is not None:
                self.read(found)
            found = target
            if target == f"{module}.{name}" or not self.walk.is_package(target):
                return found
            module = target
        return found

    def whole(self, module: str) -> None:
        """`module` held as a value: every name its lazy table resolves is one its code may read."""
        table = self.walk.table(module)
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
        base = imported_from(node, self.package)
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
            found = None if base is None else resolved(found, base)
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
        selected = self.selected(self.walk.selected[site]) if site in self.walk.selected else None
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
        names = [] if fromlist is None else literal(fromlist)
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
        value = self.walk.assigned(self.module, table)
        found = None if value is None else literal(value)
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



