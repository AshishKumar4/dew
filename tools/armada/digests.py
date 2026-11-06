"""How CI's test cache keys Python source (tools/armada/ci.py's keys): a module by its code (`code_of`), and
a code module by its units (`unit`, `units`, `digested`), each the digest of what running it does, never of
what it says of itself or where it sits. Loaded by ci.py by its path, beside followed.py, whose unit names
(`followed.TOP`) and code flags it reads."""

import ast
import collections
import hashlib
import importlib.util
import json
import types
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
    functions and classes nested in it among it; or the module's top level (`followed.TOP`), its text with
    every such function's body left out but what a function value shows without running it (its docstring,
    whether it is a generator): its imports, its constants, its classes' bodies, and its functions'
    signatures, defaults and decorators, which run as it is imported. Two functions of one name (a
    property's getter and setter, a definition under each branch) are one unit. Positions are none of it, so
    moving code reruns nothing; a run that reads a module's text (a traceback, inspect) reads its file
    whole, which keys it whole, and one that takes a function's code (`__code__`) records its unit; a test
    asserting a source position it never read is out of contract (the owner's test rules ban pinning source
    text or positions). A module that does not compile is its text in every unit, and a unit it no longer
    holds is b"missing"."""
    found = units(path)
    return found.get(name, b"missing") if isinstance(found, dict) else found


UNITS: dict[bytes, dict[str, bytes] | bytes] = {}
"""Each module text's units' digests (`digested`), by the text's digest: a plan keys a module's units for
many files, and reads and hashes its text once a key (`cache_key`'s `read`)."""


def units(path: str) -> dict[str, bytes] | bytes:
    """Module `path`'s units' digests (`digested`): its text's digest where it does not compile."""
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).digest()
    if digest not in UNITS:
        UNITS[digest] = digested(data, path)
    found = UNITS[digest]
    return digest if found is None else found


def digested(source: bytes, path: str) -> dict[str, bytes] | None:
    """The digest of each unit of module text `source` (`unit`), or None where it does not compile
    (`compiled`): a body left out would hide what fails it. The top level holds, beside its text with each
    outermost function's body left out but its docstring, what compiling a body makes of the code around it
    (`shapes`): whether a function is a generator or a coroutine, and the cells a class body makes for its
    methods (super())."""
    code = compiled(source, path)
    if isinstance(code, str):
        return None
    tree = ast.parse(source)
    found: dict[str, list[str]] = collections.defaultdict(list)

    class Hollow(ast.NodeTransformer):
        """The module's top level: each outermost function's text kept under its name, its body left out but
        its docstring, which the function value shows without running."""

        def __init__(self) -> None:
            self.scope: list[str] = []

        def visit_ClassDef(self, node):
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()
            return node

        def visit_FunctionDef(self, node):
            found[".".join([*self.scope, node.name])].append(ast.dump(node))
            documented = ast.get_docstring(node, clean=False) is not None
            node.body = [*node.body[:1 if documented else 0], ast.Pass()]
            return node

        visit_AsyncFunctionDef = visit_FunctionDef

    top = ast.dump(Hollow().visit(tree)) + json.dumps(shapes(code))
    digests = {name: hashlib.sha256("\n".join(dumps).encode()).digest() for name, dumps in found.items()}
    return {**digests, followed.TOP: hashlib.sha256(top.encode()).digest()}


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
