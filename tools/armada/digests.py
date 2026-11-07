"""How CI's test cache reads each input a row is keyed by (tools/armada/ci.py's keys, `contents`): a Python
module by its code (`code_of`), a code module by its units (`unit`, `units`, `digested`), each the digest of
what running it does, never of what it says of itself or where it sits; a probe by its answer (`probed`), an
import's lookup by what it finds (`looked_up`), and a data file by its bytes. Loaded by ci.py by its path,
beside followed.py, whose unit names (`followed.TOP`) and code flags it reads."""

import ast
import collections
import functools
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


def code_of(name: str, data: bytes | None = None) -> bytes:
    """The hash of Python file `name`'s code, or of `data` as its text where given (a commit's, which
    `policy` compares): its syntax tree, with no comment, no docstring and no line number, so a change of
    what the harness says of itself re-keys no row (row 2730: a docstring edit to a SHARED module ran the
    suite cold); its bytes' where it does not parse, or b"missing"."""
    path = Path(name)
    if data is None and not path.is_file():
        return b"missing"
    data = path.read_bytes() if data is None else data
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


HARNESS = "tools/armada/"
"""The harness's directory: the recorder, plugins and cache client every test process runs, this file's."""


FIXTURES = Path("tests/fixtures")
"""The tests' fixtures: each its openers', listers' and probers' input alone (`shared`), and a native reader's
directories up to it (`native_inputs`)."""


FETCHED = ".fetched.json"
"""A fetched fixture's manifest's suffix, beside the payload it pins (tools/fetched_fixtures.py)."""


def built(path: Path) -> bool:
    """Whether `path` is a build's output, never an input: compiled bytecode, or an editable install's
    metadata (setuptools' <name>.egg-info), which a task's environment writes into its checkout and a
    planner's lacks; or a directory holding nothing else, as a package a commit deleted leaves its bytecode
    in a checkout the environment's snapshot carries across commits, which a planner's clean tree lacks (train
    636: src/dew/cli/, its __pycache__ alone, listed in src/dew's entries, keyed every task's rows apart from
    their plan's, and two tasks ran 120 kept rows each past their deadline)."""
    if "__pycache__" in path.parts or any(part.endswith(".egg-info") for part in path.parts):
        return True
    return path.is_dir() and not path.is_symlink() and all(
        entry.name.startswith(".") or built(entry) for entry in path.iterdir())


def contents(name: str) -> bytes:
    """The hash of file `name`'s bytes, of directory `name`'s entries' names but its hidden ones and a
    build's (`built`), or b"missing". A fetched fixture's payload is the digest its manifest pins, whether
    it is downloaded here or not, so a planner that never fetches it keys it as a task does. An execution
    limit a run read (followed.py's `LIMITS`: "limit:cpu", "limit:memory") is the one binding this process
    (host.py's `limit`): a row green on 4 vCPUs is not reused on 8, nor on a planner, which reads its own (row
    1998, 2026-10-28). A path a run probed (followed.py's `PROBED`: "probe:" and the path) is what a
    probe answers of it (`probed`), never its bytes, which no probe reads. An import's search of a checkout
    directory for a name (followed.py's `LOOKUP`) is what that search finds there (`looked_up`)."""
    if name.startswith("limit:"):
        return hashlib.sha256(host().limit(name).encode()).digest()
    if name.startswith(followed.PROBED):
        return probed(Path(name.removeprefix(followed.PROBED)))
    if name.startswith(followed.LOOKUP):
        return looked_up(Path(name.removeprefix(followed.LOOKUP)))
    if followed.CODE in name:
        path, _, executed = name.partition(followed.CODE)
        if not Path(path).is_file():
            return b"missing"
        return code_of(path) if executed == followed.WHOLE else unit(path, executed)
    path = Path(name)
    manifest = path.parent / (path.name + FETCHED)
    if path.name and manifest.is_file():
        return bytes.fromhex(json.loads(manifest.read_text())["sha256"])
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).digest()
    if path.is_dir():
        entries = sorted(entry.name for entry in path.iterdir()
                         if not entry.name.startswith(".") and not built(entry))
        return hashlib.sha256("\n".join(entries).encode()).digest()
    return b"missing"


def probed(path: Path) -> bytes:
    """What an existence or size probe (followed.py's `PROBES`) answers of `path`: missing, a directory, or a
    file, whether it executes and, but for a Python module (its code its runners' input, `code_unit`), its
    size; never its bytes or times (train 668: a probed helper's comment edit re-keyed its 458 probers)."""
    if path.is_dir():
        return b"directory"
    if not path.is_file():
        return b"missing"
    stat = path.stat()
    size = "" if path.suffix == ".py" else f" {stat.st_size}"
    return f"file{size} {int(bool(stat.st_mode & 0o111))}".encode()


def looked_up(looked: Path) -> bytes:
    """What an import of name `looked.name` finds in directory `looked.parent`: the entries there of that
    name, a module of any suffix (source, bytecode, an extension) or a package, and a package's __init__ of
    any suffix, which decides whether it is a regular package or a namespace's portion; none of any other
    name (train 668: a module added under nn/backbones re-keyed the 589 rows that imported its siblings)."""
    directory = looked.parent
    if not directory.is_dir():
        return b"missing"
    found = sorted(entry.name for entry in directory.iterdir() if not built(entry) and (
        entry.name == looked.name or entry.name.startswith(f"{looked.name}.")))
    if looked.is_dir() and not looked.is_symlink():
        found += sorted(f"{looked.name}/{entry.name}" for entry in looked.iterdir()
                        if entry.name.partition(".")[0] == "__init__" and not built(entry))
    return hashlib.sha256("\n".join(found).encode()).digest()


@functools.cache
def lookups(path: str) -> tuple[str, ...]:
    """The import lookups that find module `path` under src/ or tools/ (followed.py's `LOOKUP`), its own name
    in its directory and each package's name on the way down to it, none for any other path. A function of
    the path's text alone, kept: a plan asks it of each of a row's ~3,000 inputs for every row."""
    if not path.startswith(("src/", "tools/")) or not path.endswith(".py") or path.startswith(HARNESS):
        return ()
    module = Path(path)
    found = module.parent if module.name == "__init__.py" else module.with_suffix("")
    # src/'s own entries every key covers (`shared`); tools' own lookup, a namespace package's at the root, a
    # row keys (train 672's review: a tools/__init__.py added re-keyed none of its 69 importers' rows).
    return tuple(f"{followed.LOOKUP}{step}" for step in (found, *found.parents)
                 if step.parts and step.parts[0] in ("src", "tools") and step != Path("src"))


@functools.cache
def built_input(path: str) -> bool:
    """Whether input `path` (a file, a unit of one, or a probe of one) is a build's output (`built`). A
    build's output (an editable install's metadata, bytecode) is the environment's, which the locks key
    (`shared`): a planner's checkout has none (train 300's plan reused none of 501 files). Kept, as
    `packages` is."""
    named = path.removeprefix(followed.PROBED).removeprefix(followed.LOOKUP)
    return built(Path(named.partition(followed.CODE)[0]))


@functools.cache
def host() -> types.ModuleType:
    """host.py beside this file, loaded by its path, as this file puts nothing on sys.path."""
    spec = importlib.util.spec_from_file_location("dew_host", Path(__file__).with_name("host.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
