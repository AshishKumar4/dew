"""How CI's test cache keys Python source (tools/armada/ci.py's keys): a module by its code (`code_of`), and
a code module by its units (`unit`, `units`, `digested`), each the digest of what running it does, never of
what it says of itself or where it sits. Loaded by ci.py by its path, beside followed.py, whose unit names
(`followed.TOP`) and code flags it reads."""

import ast
import builtins
import collections
import dataclasses
import hashlib
import importlib.util
import json
import types
from collections.abc import Callable
from pathlib import Path


def _followed():
    """followed.py, beside this file, loaded by its path as ci.py loads its neighbours."""
    path = Path(__file__).resolve().with_name("followed.py")
    spec = importlib.util.spec_from_file_location("armada_followed", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


followed = _followed()


def code_of(name: str) -> bytes:
    """The hash of Python file `name`'s code: its syntax tree, with no comment, no docstring and no line
    number, so a change of what the harness says of itself re-keys no row (row 2730: a docstring edit to a
    SHARED module ran the suite cold); its bytes' where it does not parse, or b"missing"."""
    path = Path(name)
    if not path.is_file():
        return b"missing"
    data = path.read_bytes()
    try:
        tree = ast.parse(data, name)
    except SyntaxError:
        return hashlib.sha256(data).digest()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree).encode()).digest()


def unit(path: str, name: str) -> bytes:
    """The digest of unit `name` of code module `path`, which a run executed (followed.py's `code_unit`):
    a function outside any function, by its qualified name (a method's by its class's), its text whole, the
    functions and classes nested in it among it; an inert definition of the module's top level, by the name
    it binds (`followed.DEFINED`, `defining`): an import, a constant, a class's body or a function's
    signature, defaults and decorators, each with what a function value shows without running it (its
    docstring, whether it is a generator), which a run keys where it reaches its name (`reach`); or the top
    level (`followed.TOP`): the rest of it, which runs as the module is imported, and the skeleton of what
    every statement binds and reads then (`parted`), which every importer keys. Two functions of one name (a
    property's getter and setter, a definition under each branch) are one unit. Positions are none of it, so
    moving code reruns nothing; a run that reads a module's text (a traceback, inspect) reads its file
    whole, which keys it whole, and one that takes a function's code (`__code__`) records its unit; a test
    asserting a source position it never read is out of contract (the owner's test rules ban pinning source
    text or positions). A module that does not compile is its text in every unit, and a unit it no longer
    holds is b"missing"."""
    found = units(path)
    return found.get(name, b"missing") if isinstance(found, dict) else found


@dataclasses.dataclass(frozen=True)
class Analysis:
    """What a code module's text holds for the keys (`analysed`): each unit's digest, the names each unit
    reads (`loaded`), the names its top level defines inertly (`defining`) and those it binds at all
    (`stored`), what each name an import binds is (`imported`: its module and the name taken from it, None for
    the module itself), and the modules its star imports take names from, by their dotted names."""

    digests: dict[str, bytes]
    reads: dict[str, frozenset[str]]
    defined: frozenset[str]
    bound: frozenset[str]
    imports: dict[str, tuple[str, str | None]]
    stars: tuple[str, ...]


ANALYSED: dict[bytes, Analysis | None] = {}
"""Each module text's analysis (`analysed`), by the text's digest: a plan keys a module's units for many
files, and reads and parses its text once a key (`cache_key`'s `read`)."""
ROOTS = ("src", "tests", "tools")
"""The checkout's directories on a test's import path, whose modules a dotted name finds (`module_path`)."""


def analysed(path: str) -> Analysis | None:
    """Module `path`'s analysis (`Analysis`), None where it does not compile: its text then keys its every
    unit."""
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).digest()
    if digest not in ANALYSED:
        ANALYSED[digest] = analysis(data, path)
    return ANALYSED[digest]


def units(path: str) -> dict[str, bytes] | bytes:
    """Module `path`'s units' digests (`digested`): its text's digest where it does not compile."""
    found = analysed(path)
    return hashlib.sha256(Path(path).read_bytes()).digest() if found is None else found.digests


def analysis(source: bytes, path: str) -> Analysis | None:
    """Module text `source`'s analysis, None where it does not compile (`compiled`): a body left out would
    hide what fails it. Each outermost function's text is its own unit, read whole; the top level, its text
    with each such function's body left out but its docstring, is cut by statement (`parted`): each inert
    definition a unit of its own (`followed.DEFINED`), its text and what compiling the bodies under it makes
    of the code around it (`shapes`: whether a function is a generator or a coroutine, and the cells a class
    body makes for its methods), and the rest, with the skeleton of what the module's import reads, the
    module's own top level (`followed.TOP`)."""
    code = compiled(source, path)
    if isinstance(code, str):
        return None
    tree = ast.parse(source)
    found: dict[str, list[str]] = collections.defaultdict(list)
    reads: dict[str, set[str]] = collections.defaultdict(set)

    class Hollow(ast.NodeTransformer):
        """The module's top level: each outermost function's text kept under its name, and what it reads, its
        body left out but its docstring, which the function value shows without running."""

        def __init__(self) -> None:
            self.scope: list[str] = []

        def visit_ClassDef(self, node):
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()
            return node

        def visit_FunctionDef(self, node):
            name = ".".join([*self.scope, node.name])
            found[name].append(ast.dump(node))
            reads[name] |= loaded(node)
            documented = ast.get_docstring(node, clean=False) is not None
            node.body = [*node.body[:1 if documented else 0], ast.Pass()]
            return node

        visit_AsyncFunctionDef = visit_FunctionDef

    hollow = Hollow().visit(tree)
    effects, defined, skeleton = parted(hollow.body)
    shaped: dict[str, list] = collections.defaultdict(list)
    for qualified, shape in shapes(code):
        root = qualified.split(".")[0]
        shaped[root if root in defined else followed.TOP].append([qualified, shape])
    named = {f"{followed.TOP}{followed.DEFINED}{name}": statements for name, statements in defined.items()}
    top = "\n".join(ast.dump(each) for each in effects) + json.dumps([skeleton, shaped[followed.TOP]])
    digests = {**{name: hashlib.sha256("\n".join(dumps).encode()).digest() for name, dumps in found.items()},
               followed.TOP: hashlib.sha256(top.encode()).digest(),
               **{unit: hashlib.sha256(("\n".join(ast.dump(each) for each in statements) + json.dumps(
                   shaped[unit.rpartition(followed.DEFINED)[2]])).encode()).digest()
                  for unit, statements in named.items()}}
    reads[followed.TOP] = set().union(*(loaded(each) for each in effects))
    for unit, statements in named.items():
        reads[unit] = set().union(*(loaded(each) for each in statements))
    name = module_name(path)
    package = name if Path(path).name == "__init__.py" else name.rpartition(".")[0]
    imports = {name: taken for statements in defined.values() for statement in statements
               if isinstance(statement, (ast.Import, ast.ImportFrom)) for name in defining(statement) or ()
               if (taken := imported(statement, name, package)) is not None}
    stars = tuple(resolved_from(each, package) for each in effects
                  if isinstance(each, ast.ImportFrom) and any(alias.name == "*" for alias in each.names))
    return Analysis(digests, {unit: frozenset(names) for unit, names in reads.items()}, frozenset(defined),
                    frozenset(stored(tree.body)), imports, stars)


INERT_DECORATORS = frozenset({
    "dataclass", "dataclasses.dataclass", "struct.dataclass", "flax.struct.dataclass", "property",
    "staticmethod", "classmethod", "functools.cache", "functools.lru_cache", "functools.cached_property",
    "cached_property", "abc.abstractmethod", "abstractmethod", "typing.overload", "overload", "typing.final",
    "final", "typing.override", "override", "contextlib.contextmanager", "contextmanager", "jax.jit", "jit",
    "functools.partial", "partial", "nn.compact", "compact", "nn.nowrap", "nowrap",
    "jax.tree_util.register_dataclass", "tree_util.register_dataclass", "jax.tree_util.register_static",
    "tree_util.register_static", "jax.tree_util.register_pytree_node_class",
    "tree_util.register_pytree_node_class"})
"""The decorators whose effect is the object they decorate alone (`inert`): a dataclass's fields, a property,
a cache, a flax module's compact method, a pytree registration of the class's own instances, a jitted
function. A decorator of any other name may put the object where every importer reads it (a registry, by its
name), so its statement is the top level's, which every importer keys."""


def dotted(node: ast.AST) -> str | None:
    """`node` as the dotted name it is (`jax.tree_util.register_dataclass`), a call of one by its callee's,
    None for any other expression."""
    if isinstance(node, ast.Call):
        return dotted(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and (owner := dotted(node.value)) is not None:
        return f"{owner}.{node.attr}"
    return None


def inert(node: ast.AST | None) -> bool:
    """Whether evaluating expression `node` as its module is imported reads names and builds values alone, so
    it runs nothing a test sees beside what it defines: a literal, a name or an attribute of one, a subscript
    of such (a typing generic), a union, a container or a formatted string of such, a number's sign; a call of
    an inert decorator (`INERT_DECORATORS`) with such arguments. A name it reads that is gone fails the
    import, which the skeleton holds for every importer (`parted`)."""
    if node is None or isinstance(node, (ast.Constant, ast.Name)):
        return True
    if isinstance(node, ast.Attribute):
        return inert(node.value)
    if isinstance(node, ast.Subscript):
        return inert(node.value) and inert(node.slice)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(inert(each) for each in node.elts)
    if isinstance(node, ast.Dict):
        return all(inert(each) for each in (*node.keys, *node.values))
    if isinstance(node, ast.JoinedStr):
        return all(inert(each) for each in node.values)
    if isinstance(node, ast.FormattedValue):
        return inert(node.value)
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.UAdd, ast.USub)) and isinstance(node.operand, ast.Constant)
    if isinstance(node, ast.BinOp):
        return isinstance(node.op, ast.BitOr) and inert(node.left) and inert(node.right)
    if isinstance(node, ast.Call):
        return (dotted(node.func) in INERT_DECORATORS and all(inert(each) for each in node.args)
                and all(inert(each.value) for each in node.keywords))
    return False


def decorating(node: ast.expr) -> bool:
    """Whether decorator `node` is inert (`INERT_DECORATORS`): a decorator is called on what it decorates, so
    a name is one only where it is one of them, as a call that makes the decorator is where it calls one."""
    return dotted(node) in INERT_DECORATORS and inert(node) if isinstance(node, ast.Call) else (
        dotted(node) in INERT_DECORATORS)


def headed(statement: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    """What a function statement evaluates as it runs: its decorators, defaults, annotations and return's."""
    arguments = statement.args
    named = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
             *(each for each in (arguments.vararg, arguments.kwarg) if each is not None)]
    return [*statement.decorator_list, *arguments.defaults, *(each for each in arguments.kw_defaults if each),
            *(each.annotation for each in named if each.annotation),
            *([statement.returns] if statement.returns else [])]


def defining(statement: ast.stmt) -> list[str] | None:
    """The names a top-level `statement` binds where it is an inert definition (`inert`), None where it is
    not: an import of named names; a function, or a class, whose decorators, defaults, annotations, bases and
    body evaluate inertly; an assignment of an inert value to names."""
    if isinstance(statement, ast.Import):
        return [(alias.asname or alias.name).split(".")[0] for alias in statement.names]
    if isinstance(statement, ast.ImportFrom):
        return None if any(alias.name == "*" for alias in statement.names) else [
            alias.asname or alias.name for alias in statement.names]
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        head = [each for each in headed(statement) if each not in statement.decorator_list]
        return [statement.name] if all(decorating(each) for each in statement.decorator_list) and all(
            inert(each) for each in head) else None
    if isinstance(statement, ast.ClassDef):
        if not all(decorating(each) for each in statement.decorator_list):
            return None
        head = [*statement.bases, *(each.value for each in statement.keywords)]
        body = all(isinstance(each, ast.Pass)
                   or (isinstance(each, ast.Expr) and isinstance(each.value, ast.Constant))
                   or (not isinstance(each, (ast.Import, ast.ImportFrom)) and defining(each) is not None)
                   for each in statement.body)
        return [statement.name] if body and all(inert(each) for each in head) else None
    if isinstance(statement, ast.Assign) and inert(statement.value) and all(
            isinstance(target, ast.Name) for target in statement.targets):
        return [target.id for target in statement.targets if isinstance(target, ast.Name)]
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name) and inert(
            statement.annotation) and inert(statement.value):
        return [statement.target.id]
    return None


SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.ListComp, ast.SetComp,
          ast.DictComp, ast.GeneratorExp)
"""The nodes that open a scope of their own, whose names `stored` leaves to them."""


def stored(body: list[ast.AST]) -> set[str]:
    """The names statements `body` bind in their own scope, which reads of them there do not leave: assigned,
    looped over, imported, defined, caught, bound by `with` or a walrus, deleted; not those a nested scope
    binds (`SCOPES`), nor those declared `global` or `nonlocal`."""
    found, declared, pending = set(), set(), list(body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            found.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            found.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            declared.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            found.add(node.name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
            pending.extend([*node.decorator_list, *(node.bases if isinstance(node, ast.ClassDef) else [])])
            continue
        if isinstance(node, SCOPES):
            continue
        pending.extend(ast.iter_child_nodes(node))
    return found - declared


def parameters(arguments: ast.arguments) -> set[str]:
    """The names a function's `arguments` bind."""
    rest = [each for each in (arguments.vararg, arguments.kwarg) if each is not None]
    return {each.arg for each in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, *rest)}


def loaded(node: ast.AST, bound: frozenset[str] = frozenset()) -> set[str]:
    """The names `node` reads that no scope of its own binds (`stored`, `parameters`), each as the dotted
    chain it reads where it reads attributes of one (`dotted`: `nn.Module`, `attention.CausalSelfAttention`):
    what it reads of its module and the builtins as it runs. A function's head and a class's bases read in the
    scope around them, a method none of its class's names; a call of `globals()` may read any
    (`followed.EVERY`). `bound` holds the names the scopes around `node` bind."""
    found: set[str] = set()
    chain = dotted(node) if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) else None
    if chain is not None:
        if chain.split(".")[0] not in bound:
            found.add(chain)
        return found
    if isinstance(node, ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id not in bound:
            found.add(followed.EVERY if node.id == "globals" else node.id)
        return found
    if isinstance(node, ast.Global):
        return set(node.names)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        head = headed(node) if not isinstance(node, ast.Lambda) else [*node.args.defaults, *(
            each for each in node.args.kw_defaults if each)]
        body = node.body if isinstance(node.body, list) else [node.body]
        inner = bound | parameters(node.args) | stored(body)
        return set().union(*(loaded(each, bound) for each in head),
                           *(loaded(each, frozenset(inner)) for each in body))
    if isinstance(node, ast.ClassDef):
        head = [*node.decorator_list, *node.bases, *(each.value for each in node.keywords)]
        own = frozenset(bound | stored(node.body))
        return set().union(*(loaded(each, bound) for each in head), *(
            loaded(each, bound if isinstance(each, (ast.FunctionDef, ast.AsyncFunctionDef)) else own)
            for each in node.body))
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        inner = frozenset(bound | stored([generator.target for generator in node.generators]))
        return set().union(*(loaded(each, inner) for each in ast.iter_child_nodes(node)))
    return set().union(set(), *(loaded(each, bound) for each in ast.iter_child_nodes(node)))


def imported(statement: ast.Import | ast.ImportFrom, name: str,
             package: str) -> tuple[str, str | None] | None:
    """What name `name`, which import `statement` of a module of `package` binds, is: the module it names and
    the name it takes from it (None for the module itself), by their dotted names, a relative one resolved."""
    if isinstance(statement, ast.Import):
        aliases = statement.names
        return next(((alias.name if alias.asname else alias.name.split(".")[0], None) for alias in aliases
                     if (alias.asname or alias.name).split(".")[0] == name), None)
    return next(((resolved_from(statement, package), alias.name) for alias in statement.names
                 if (alias.asname or alias.name) == name), None)


def resolved_from(statement: ast.ImportFrom, package: str) -> str:
    """The dotted name of the module from-import `statement` of a module of `package` names, a relative one
    resolved against the package."""
    if not statement.level:
        return statement.module or ""
    parts = package.split(".") if package else []
    base = parts[:len(parts) - statement.level + 1] if statement.level > 1 else parts
    return ".".join([*base, *([statement.module] if statement.module else [])])


def importing(statement: ast.stmt) -> list[ast.AST]:
    """What top-level `statement` evaluates as its module is imported: a function's head (`headed`), a
    class's head and body but its methods' bodies, an import nothing of the module, any other the whole."""
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return headed(statement)
    if isinstance(statement, ast.ClassDef):
        return [*statement.decorator_list, *statement.bases, *(each.value for each in statement.keywords), *(
            part for each in statement.body for part in (headed(each) if isinstance(
                each, (ast.FunctionDef, ast.AsyncFunctionDef)) else importing(each)))]
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return []
    return [statement]


def parted(body: list[ast.stmt]) -> tuple[list[ast.stmt], dict[str, list[ast.stmt]], list]:
    """A module's top level `body` (its functions' bodies left out) cut by statement: those that are not
    inert definitions (`defining`); each inert definition's statements by the name they bind (two of one name
    are one); and the skeleton its import walks, each statement's in order: the builtins it shadows, and the
    names it reads as it runs with whether the module bound them before, so a definition taken away, or moved
    past a statement that reads it, which fails the import, keys every importer though none reaches it."""
    effects: list[ast.stmt] = []
    defined: dict[str, list[ast.stmt]] = collections.defaultdict(list)
    skeleton, bound = [], set()
    for statement in body:
        names = defining(statement)
        if names is None:
            effects.append(statement)
        roots = sorted({read.split(".")[0] for each in importing(statement) for read in loaded(each)}
                       - {followed.EVERY})
        # A statement that shadows no builtin and reads nothing as it runs, as a definition added or taken
        # away is, leaves the skeleton as it was.
        if (shadowed := sorted(set(names or ()) & BUILTINS)) or roots:
            skeleton.append([shadowed, [[root, root in bound] for root in roots]])
        for name in names or ():
            defined[name].append(statement)
        bound.update(names or ())
    return effects, dict(defined), skeleton


BUILTINS = frozenset(dir(builtins))
"""The names a module reads from the builtins where it binds none of its own: one it binds, shadowing one, is
in the skeleton (`parted`), as no unit that reads it names it."""


def module_path(dotted_name: str) -> str | None:
    """The checkout module a dotted module name finds on a test's import path (`ROOTS`), a module or a
    package's __init__.py, None where none of the checkout's does (a third party's, the locks' to key)."""
    parts = dotted_name.split(".")
    for root in ROOTS:
        for candidate in (Path(root, *parts).with_suffix(".py"), Path(root, *parts, "__init__.py")):
            if all(parts) and candidate.is_file():
                return str(candidate)
    return None


def module_name(path: str) -> str:
    """The dotted name module `path` imports as from its root (`ROOTS`): a package's __init__.py its
    package's."""
    parts = Path(path).with_suffix("").parts
    parts = parts[1:] if parts[0] in ROOTS else parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def reach(path: str, chain: str, found: set[str], visited: set[tuple[str, str]],
          analysed_of: Callable[[str], Analysis | None] = lambda path: analysed(path)) -> None:
    """What reading dotted name `chain` in module `path` reaches, added to `found`: the definition of its
    first name there (`followed.DEFINED`), or its absence, and what that definition reads in turn; through an
    import, the module it binds and what the rest of the chain names there; a package's submodule of the name
    and the rest of the chain in it; through a star import, the name in each module it takes names from; every
    definition where `chain` is `followed.EVERY`. `visited` holds each module's chains read already, and
    `analysed_of` gives a module's analysis (ci.py's reading, which reads each module once)."""
    if (path, chain) in visited or not Path(path).is_file() or (analysis := analysed_of(path)) is None:
        return
    visited.add((path, chain))
    root, _, rest = chain.partition(".")
    if root == followed.EVERY:
        for name in analysis.defined:
            reach(path, name, found, visited, analysed_of)
        return
    if root in analysis.defined:
        found.add(f"{path}{followed.CODE}{followed.TOP}{followed.DEFINED}{root}")
        if root in analysis.imports:
            module, name = analysis.imports[root]
            taken = module if name is None else f"{module}.{name}"
            if (submodule := module_path(taken)) is not None:
                if rest:
                    reach(submodule, rest, found, visited, analysed_of)
            elif name is not None and (origin := module_path(module)) is not None:
                reach(origin, f"{name}.{rest}" if rest else name, found, visited, analysed_of)
        for read in analysis.reads.get(f"{followed.TOP}{followed.DEFINED}{root}", ()):
            reach(path, read, found, visited, analysed_of)
        return
    if root in analysis.bound or root in BUILTINS:
        # Bound by the top level's own statements, which every importer keys (`followed.TOP`), or a builtin.
        return
    # Unbound here: a definition of the name arriving re-keys the reader; a package's submodule of the name,
    # and a star import's modules, may bind it now.
    found.add(f"{path}{followed.CODE}{followed.TOP}{followed.DEFINED}{root}")
    submodule = module_path(f"{module_name(path)}.{root}") if Path(path).name == "__init__.py" else None
    if submodule is not None and rest:
        reach(submodule, rest, found, visited, analysed_of)
    for star in analysis.stars:
        if (origin := module_path(star)) is not None:
            reach(origin, chain, found, visited, analysed_of)


def compiled(source: bytes, path: str) -> types.CodeType | str:
    """Module text `source` compiled, or why it does not compile, as Python says."""
    try:
        return compile(source, path, "exec", dont_inherit=True)
    except SyntaxError as error:
        return str(error)


def shapes(module: types.CodeType) -> list[list]:
    """What compiling each function body outside any function makes of the code around it, shown without
    running it: each such function's flags (a generator's, a coroutine's), and each class body's cells for
    its methods (super(), __class__), by qualified name."""
    found, pending = [], [module]
    while pending:
        code = pending.pop()
        for const in code.co_consts:
            if isinstance(const, types.CodeType) and ".<locals>" not in const.co_qualname:
                outer = bool(const.co_flags & followed.OPTIMIZED)
                found.append([const.co_qualname, const.co_flags if outer else list(const.co_cellvars)])
                pending.append(const)
    return sorted(found, key=json.dumps)
