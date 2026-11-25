"""The digest of the code a record of what Dew computed is a function of (`code_digest`): its producer's
modules and every module of Dew's their code can import (`dew._imports.Walk`, the one reader of the imports a
module's run executes), so a record is computed again where that code changed and kept where Dew changed
elsewhere (row 10743: every record keyed by the whole package, any edit certified every launch again). An
import by a name its code computes is a dynamic site, named by the expression it imports: each of Dew's is
reviewed (`REVIEWED`) or resolved by the literal table it selects from (`SELECTED`), and a closure holding
any other, or a lazy table no literal states, is keyed by the whole package instead (`whole_digest`),
logged, so a site added unreviewed costs warm records, never a stale one.

Every read is of one snapshot of the tree a process takes (`_snapshot`: each module's file as it is first
looked up, its bytes as first read), so a closure and its digest state one revision, the one its imports
read, and a digest reads, and lists, its closure's files alone: a test of a kernel's records reads
the files its kernel's code is, not every file of Dew's (2026-11-23: a snapshot that read every file whole
made each such test's CI key every file of Dew's, so any edit anywhere ran 61% of the suite again). A
closure is walked once a tree and kept in Dew's cache, so a start reads its files' bytes, not their syntax:
a walk parses an op's 153 files in about a second.

A process's digests are of its files as it first reads them: a source file of Dew's edited in place inside a
running process is not supported (a module imported before the edit runs code its digest does not state).
Edit a checkout no process runs, or start the process again; a module made a package or a package a module
under a running process is untied, its records keyed by that process alone (`_tied`)."""

import contextlib
import dataclasses
import functools
import hashlib
import json
import logging
import os
import sys
import threading
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

from dew._cache import VERSION_FILE, keep, kept_beside, read_kept
from dew._imports import Walk
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
    "dew.clusters:tpu_chips:<loader>":
        "JAX's hardware_utils, read by its file to count TPU chips: JAX's code",
    "dew.data.sources.codec:_dataclass_decoded:module":
        "a dataclass a stored artifact names, rebuilt as data from its fields, as "
        "`dew.registry:imported:module`",
    "dew.interop.parts:_known_class:module":
        "a library's class a checkpoint's parts name (transformers'), its release stating its code",
    **dict.fromkeys((
        "dew.registry:_importable:sys.modules", "dew.registry:declared_type:sys.modules",
        "dew.registry:imported:sys.modules", "dew.registry:module_from_file:sys.modules",
        "dew.registry:parameter_type:sys.modules"),
        "a module already imported, by its name or a record-named class's: whether it is, or its namespace, "
        "which a class's annotations are evaluated among, as `dew.registry:imported:module`"),
    **dict.fromkeys((
        "dew._cache:unintercepted:sys.modules", "dew._devices:_seen:sys.modules",
        "dew._devices:backend_open:sys.modules", "dew._devices:flags_hold:sys.modules",
        "dew._devices:opens_tpu:sys.modules",
        "dew._code:_tied:sys.modules",
        "dew._sources:file_of:sys.modules", "dew._sources:function_reads:sys.modules",
        "dew._sources:generated_fields:sys.modules", "dew._sources:own_source:sys.modules",
        "dew._sources:recorded:sys.modules", "dew._sources:source_file:sys.modules",
        "dew._stated:_coded:sys.modules", "dew._stated:_converter:sys.modules",
        "dew._stated:_named:sys.modules", "dew._stated:_stateless:sys.modules",
        "dew._stated:_walked:sys.modules", "dew._stated:_wrapping:sys.modules",
        "dew.data.pipeline:_main_importable:sys.modules", "dew.nn.dense:providing:sys.modules",
        "dew.quantization:_quantized_array:sys.modules",
        "dew.data.sources.memory:_table_records:sys.modules"),
        "a module this process already imported, a library's, the user's or the one defining a value it "
        "holds, read by its variables or the name that value states: it imports none of Dew's"),
}
"""Each dynamic import site of Dew's (`module:function:expression`) whose code a closure need not follow,
and why: a changed or added site is another site, unreviewed until it is."""

SELECTED = {"dew.quantization:_complex_in_float.get_intercept_map:module": "UNQUANTIZED"}
"""Each dynamic import site that imports the modules of the dotted names a literal table of its module holds
(`module.function` keys), which a closure follows as it follows a stated import."""

@functools.cache
def code_digest(*roots: str) -> str:
    """The digest of the modules `roots` name and every module of Dew's their code can import (`closure`),
    each by its path and contents as this process's snapshot of the tree holds them (`_snapshot`), the
    release's version (`dew._version`) not read: what a record of what those modules computed keys on. The
    whole package's (`whole_digest`) where the closure holds a dynamic import site not `REVIEWED`, which is
    logged (its code could import any module), or where a file the walk reads is gone since the tree was
    listed (`_Changed`): keyed by the snapshot, as the process's imports are by the tree they read."""
    try:
        modules, sites = _kept_closure(roots)
        unreviewed = sorted(set(sites) - REVIEWED.keys())
        if unreviewed:
            _log.warning("the code of %s imports by computed names at %s, none reviewed "
                         "(dew._code.REVIEWED): its records are keyed by all of Dew's code",
                         ", ".join(roots), ", ".join(unreviewed))
            return whole_digest()
        return _digest(modules)
    except _Changed:
        return whole_digest()
    except _Untied as untied:
        return _untied_digest(str(untied))


@functools.cache
def whole_digest() -> str:
    """The digest of every module of Dew's as this process's snapshot holds them (`_snapshot`), the release's
    version not read: what a record of code no closure bounds keys on (a launch with a hook,
    `provenance_code`; a rung, a verified mapping, a saved source's export), which reads every file. A
    file gone since the tree was listed is none of it. Every file of the packages is read, a module's whose
    name is no identifier among them (importlib loads `dew/extra-name.py`)."""
    digest = hashlib.sha256()
    try:
        for relative in _listed():
            with contextlib.suppress(_Changed):
                digest.update(f"{relative}\0{_contents_at(relative)}\0".encode())
    except _Untied as untied:
        return _untied_digest(str(untied))
    return digest.hexdigest()


@functools.cache
def _untied_digest(why: str) -> str:
    """A digest no other process computes: this process's code where a file it reads cannot be tied to the
    revision it imported (`_Untied`), so no record it keys is another process's, nor kept for one."""
    _log.warning("%s: records this process computes are keyed by it alone", why)
    return hashlib.sha256(b"untied\0" + os.urandom(16)).hexdigest()


def _digest(modules: Collection[str]) -> str:
    """The digest of `modules` by their paths and contents as this process's snapshot holds them
    (`_contents`): the files of `modules` alone are read."""
    files = _snapshot()
    digest = hashlib.sha256()
    for module in sorted(modules):
        digest.update(f"{files[module]}\0{_contents(module)}\0".encode())
    return digest.hexdigest()


def _contents(module: str) -> str:
    """The digest of `module`'s contents as this process's snapshot holds them (`_source`)."""
    return _contents_at(_file(module))


def _file(module: str) -> str:
    """Dew's `module`'s file under `SOURCE` in this process's snapshot of the tree (`_snapshot`); LookupError
    where it holds none."""
    found = _snapshot().get(module)
    if found is None:
        raise LookupError(f"no module of Dew's is named {module}")
    return found


@functools.cache
def _contents_at(relative: str) -> str:
    """The digest of the file at `relative` under `SOURCE` as this process first read it (`_read`)."""
    return hashlib.sha256(_read(relative)).hexdigest()


def provenance(values: Sequence[object]) -> frozenset[str] | None:
    """The modules of Dew's defining `values` and every value they hold (a dataclass's fields', a
    collection's members): a class's own module, any other value's class's, none for a library's or the
    user's own, whose release or source states it. None, the whole package, where any is a callable no class
    of Dew's states (`stated_class`): a function (a hook, a prepared one), a bound method, a library's
    wrapper (a `functools.partial`) or an instance of a class made in a function. A kernel's identity traces
    a hook at one tile alone (`attention.hook_program`), and no value names the code a wrapper calls (row
    10743). A callable instance of a class of Dew's at its module's top level (a Flax module, a template's
    hook such as `Capped`) is its class's code, as any other instance is, its fields walked as theirs are."""
    parts = held(values)
    if any(callable(part) and not isinstance(part, type) and not stated_class(type(part)) for part in parts):
        return None
    modules = {(part if isinstance(part, type) else type(part)).__module__ for part in parts}
    return frozenset(module for module in modules if _path(module) is not None)


def stated_class(kind: type) -> bool:
    """Whether `kind`'s code is a module of Dew's, which defines it at its top level, so an instance's code is
    that module's and what it imports (`code_digest`)."""
    return _path(kind.__module__) is not None and "<locals>" not in kind.__qualname__


def op_provenance(op, arguments: Mapping[str, object]) -> frozenset[str] | None:
    """A kernel launch's provenance (`provenance`): its op's class, its fields' values and the hooks among its
    `arguments` (`op.HOOKS`), which a certification's verdict and an attention choice key on."""
    fields = ([getattr(op, field.name) for field in dataclasses.fields(type(op))]
              if dataclasses.is_dataclass(type(op)) else [])
    return provenance([type(op), *fields, *(value for name, value in arguments.items() if name in op.HOOKS)])


def provenance_code(root: str, provenance: frozenset[str] | None) -> list[str]:
    """The digests (`code_digest`) of the code a record of `root`'s, of values whose `provenance` it names,
    keys on: `root`'s and each module's of `provenance`, or the whole package's (`whole_digest`) where it
    is None."""
    if provenance is None:
        return [whole_digest()]
    return [code_digest(module) for module in (root, *sorted(provenance))]


def held(values: Sequence[object]) -> list[object]:
    """Every value `values` hold (`_held`), each value's own first."""
    return [part for value in values for part in _held(value)]


def _held(value) -> Sequence[object]:
    """`value` and every value it holds: a dataclass's fields', a plain class of Dew's attributes, a
    sequence's or a set's members, a mapping's values, each once however often it is held, so a graph that
    holds itself (a cyclic hook graph's) ends."""
    found: list[object] = []
    seen: set[int] = set()
    pending = [value]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        found.append(current)
        if dataclasses.is_dataclass(current) and not isinstance(current, type):
            pending.extend(getattr(current, field.name) for field in dataclasses.fields(current))
        elif isinstance(current, (tuple, list, frozenset, set)):
            pending.extend(current)
        elif isinstance(current, Mapping):
            pending.extend(current.values())
        elif not isinstance(current, type) and stated_class(type(current)):
            # A plain class of Dew's holds its values as attributes (an objective's criterion, its model).
            pending.extend(vars(current).values())
    return found


class _Changed(Exception):
    """A source file the tree's listing named is gone (`_snapshot`)."""


class _Untied(Exception):
    """A source file of a module this process imported whose bytes are not those it imported (`_tied`)."""


def _listed() -> list[str]:
    """Each Python file of Dew's packages, by its path under `SOURCE`, a module's whose name is no identifier
    among them, but the release's version (`VERSION_FILE`) and a file a link places outside `SOURCE`: a
    listing of the whole tree, which `whole_digest` alone reads (a directory listed is a read of it)."""
    root = SOURCE.resolve()
    files = []
    for package in PACKAGES:
        for path in sorted((SOURCE / package).rglob("*.py")):
            relative = path.relative_to(SOURCE)
            if relative != Path("dew", VERSION_FILE) and path.resolve().is_relative_to(root):
                files.append(relative.as_posix())
    return files


class _Tree(Mapping[str, str]):
    """Dew's modules by name, each looked up as it is asked for, its file's path under `SOURCE`: no directory
    is listed, so a closure's walk reads, and its run is keyed by, the files of its closure alone. Each name's
    answer is the snapshot's (`looked`), the same at every later ask."""

    def __init__(self) -> None:
        self.looked: dict[str, str | None] = {}

    def __getitem__(self, module: str) -> str:
        if module not in self.looked:
            self.looked[module] = _located(module)
        found = self.looked[module]
        if found is None:
            raise KeyError(module)
        return found

    def __iter__(self):
        named = (".".join(Path(relative).with_suffix("").parts).removesuffix(".__init__")
                 for relative in _listed())
        return iter([name for name in named if name in self])

    def __len__(self) -> int:
        return sum(1 for _ in self)


def _located(module: str) -> str | None:
    """Dew's `module`'s file under `SOURCE`, by its name, or None: a package's `__init__.py` where a module
    file of its name is beside it, as the import system finds it; the release's version (`VERSION_FILE`),
    which no code of Dew's computes from, none, so a release that moves it alone keeps every record."""
    parts = module.split(".")
    # Any name importlib takes (`dew.extra-name` among them), none that would leave the package's directory.
    if parts[0] not in PACKAGES or any(not part or part in (".", "..") or "/" in part or "\\" in part
                                       for part in parts):
        return None
    for relative in (Path(*parts, "__init__.py"), Path(*parts).with_suffix(".py")):
        if relative != Path("dew", VERSION_FILE) and (SOURCE / relative).is_file():
            return relative.as_posix()
    return None


_trees: list[_Tree] = []
_walks: list[Walk] = []
_making = threading.Lock()


def _snapshot() -> _Tree:
    """Dew's source as this process first finds it: each module's file as first looked up (`_Tree`), and its
    bytes as first read (`_source`), one tree throughout, however the files change after, made once however
    many threads ask at once."""
    with _making:
        if not _trees:
            _trees.append(_Tree())
        return _trees[0]


def _kept_closure(roots: tuple[str, ...]) -> tuple[frozenset[str], tuple[str, ...]]:
    """`closure(*roots)`, kept beside the compilation cache the verdicts it keys live in (`kept_beside`)
    under the tree's digest (`whole_digest`), the Python that parsed it and the roots, so a later start of the
    same tree reads it, walking nothing; one that does not name its roots, or names a module the tree has
    not, is walked again. Where nothing is kept (a bucket's cache, a read-only home: `dew._cache.keep`), the
    walk's closure serves the process."""
    directory = kept_beside("dew-code")
    if directory is None:
        return closure(*roots)
    named = hashlib.sha256(json.dumps([roots, _policy()]).encode()).hexdigest()[:16]
    kept = directory / "{}.{}-{}".format(*sys.version_info[:2], named)
    found = read_kept(kept)
    if isinstance(found, dict) and _sound(found, roots):
        held, at = found["modules"], found["sites"]
        assert isinstance(held, list) and isinstance(at, list)
        return frozenset(map(str, held)), tuple(map(str, at))
    modules, sites = closure(*roots)
    record: dict[str, JSON] = {"roots": [*roots], "modules": [*sorted(modules)], "sites": [*sites],
                               "digests": {module: _contents(module) for module in sorted(modules)},
                               "looked": {name: _snapshot().get(name)
                                          for name in sorted(_walk().footprint(modules))}}
    keep(kept, record, f"the closure of {', '.join(roots)}")
    return modules, sites





@functools.cache
def _policy() -> str:
    """The digest of how a closure is walked: the walker's code (`dew._imports`) and the sites it follows by
    their tables (`SELECTED`), which a kept closure's name holds, so a change to either walks every closure
    again, though no closure imports the walker."""
    walker = _read("dew/_imports.py")
    return hashlib.sha256(walker + json.dumps(SELECTED, sort_keys=True).encode()).hexdigest()


def _sound(found: dict[str, JSON], roots: tuple[str, ...]) -> bool:
    """Whether `found` is a closure of `roots` as `_kept_closure` keeps one, the tree as its walk found it:
    each module of it the same file, its contents the same, and each name the walk looked up (`_Tree`) found
    where it was, or still absent. A closure's own code states what it imports, by names whose lookups the
    walk made: where all of those find what they found, so does the walk, whatever changed beside it (a file
    added that a guarded import would now find, a module become a package, is a lookup that differs)."""
    modules, sites = found.get("modules"), found.get("sites")
    digests, looked = found.get("digests"), found.get("looked")
    if found.get("roots") != list(roots) or not isinstance(modules, list) or not isinstance(sites, list) \
            or not isinstance(digests, dict) or not isinstance(looked, dict):
        return False
    tree = _snapshot()
    if not (all(isinstance(site, str) for site in sites) and set(roots) <= set(modules) == set(digests)
            and all(tree.get(name) == where for name, where in looked.items())
            and all(isinstance(module, str) and tree.get(module) is not None for module in modules)):
        return False
    try:
        return all(isinstance(module, str) and _contents(module) == digests[module] for module in modules)
    except _Changed:
        return False


def closure(*roots: str) -> tuple[frozenset[str], tuple[str, ...]]:
    """The modules of Dew's `roots` name and their code can import, and the dynamic import sites among
    them (`dew._imports.Walk.closure` over this process's snapshot of the tree). A root naming no module of
    Dew's is refused."""
    return _walk().closure(*roots)


def _walk() -> Walk:
    """The walk of this process's snapshot of Dew's source (`_snapshot`), each module read checked against it
    (`_source`), made once however many threads ask at once: a closure and its footprint read one walk."""
    tree = _snapshot()
    with _making:
        if not _walks:
            _walks.append(Walk(tree, _source, SELECTED))
        return _walks[0]


def _path(module: str) -> Path | None:
    """The source file of Dew's `module` in this process's snapshot of the tree (`_snapshot`), None where
    it holds none (a library's, an absent one)."""
    found = _snapshot().get(module)
    return None if found is None else SOURCE / found


def _source(module: str) -> bytes:
    """`module`'s source as this process's snapshot of the tree holds it (`_snapshot`, `_read`)."""
    return _read(_file(module))


_reads: dict[str, bytes] = {}
_reading = threading.Lock()


def _read(relative: str) -> bytes:
    """The bytes of the file at `relative` under `SOURCE` as this process first reads them, one read however
    many threads ask at once, so a closure's walk parses and its digest hashes the same bytes; `_Changed`
    where the file is gone since the tree was listed, `_Untied` where its module was imported from another
    file (`_tied`)."""
    with _reading:
        if relative not in _reads:
            path = SOURCE / relative
            try:
                source = path.read_bytes()
            except FileNotFoundError as missing:
                raise _Changed(f"{path} was removed after this process first listed Dew's source") \
                    from missing
            _tied(relative)
            _reads[relative] = source
        return _reads[relative]


def _tied(relative: str) -> None:
    """Raise `_Untied` where a module this process imported from another file is now located at `relative`
    (a module made a package, or a package a module, after its import): its imports were read from a file
    its digest does not hash. Only the file the module's name finds now, as an import would (`_located`): a
    module file a package of its name shadows, which `whole_digest` reads too, is no module's. An edit of a
    file in place inside a running process is not checked (see the module's doc)."""
    name = ".".join(Path(relative).with_suffix("").parts).removesuffix(".__init__")
    if _located(name) != relative:
        return
    loaded = sys.modules.get(name)
    spec = None if loaded is None else loaded.__spec__
    if spec is not None and spec.origin is not None and \
            Path(spec.origin).resolve() != (SOURCE / relative).resolve():
        raise _Untied(f"dew's {name} was imported from {spec.origin}, and is now at {relative}")
