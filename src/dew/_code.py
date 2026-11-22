"""The digest of the code a record of what Dew computed is a function of (`code_digest`): its producer's
modules and every module of Dew's their code can import (`dew._imports.Walk`, the one reader of the imports a
module's run executes), so a record is computed again where that code changed and kept where Dew changed
elsewhere (row 10743: every record keyed by the whole package, any edit certified every launch again). An
import by a name its code computes is a dynamic site, named by the expression it imports: each of Dew's is
reviewed (`REVIEWED`) or resolved by the literal table it selects from (`SELECTED`), and a closure holding
any other, or a lazy table no literal states, is keyed by the whole package instead (`whole_digest`),
logged, so a site added unreviewed costs warm records, never a stale one.

Every read is of one snapshot of the tree a process takes once (`_snapshot`: each file's digest), so a
closure and its digest state one revision, the one its imports read. A closure
is walked once a tree and kept in Dew's cache, so a start reads its files' bytes, not their syntax: a walk
parses an op's 153 files in about a second."""

import contextlib
import dataclasses
import functools
import hashlib
import json
import logging
import sys
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

from dew._cache import VERSION_FILE, default_compilation_cache_dir
from dew._files import write_atomically
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
        "dew._sources:file_of:sys.modules", "dew._sources:function_reads:sys.modules",
        "dew._sources:generated_fields:sys.modules", "dew._sources:own_source:sys.modules",
        "dew._sources:recorded:sys.modules", "dew._sources:source_file:sys.modules",
        "dew._stated:_coded:sys.modules", "dew._stated:_converter:sys.modules",
        "dew._stated:_named:sys.modules", "dew._stated:_stateless:sys.modules",
        "dew._stated:_wrapping:sys.modules",
        "dew.data.pipeline:_main_importable:sys.modules",
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
    logged (its code could import any module), or where a file the walk reads is no longer the snapshot's
    (`_Changed`): keyed by the snapshot, as the process's imports are by the tree they read."""
    try:
        modules, sites = _kept_closure(roots)
    except _Changed:
        return whole_digest()
    unreviewed = sorted(set(sites) - REVIEWED.keys())
    if unreviewed:
        _log.warning("the code of %s imports by computed names at %s, none reviewed (dew._code.REVIEWED): "
                     "its records are keyed by all of Dew's code", ", ".join(roots), ", ".join(unreviewed))
        return whole_digest()
    return _digest(modules)


@functools.cache
def whole_digest() -> str:
    """The digest of every module of Dew's as this process's snapshot holds them (`_snapshot`), the release's
    version not read: what a record of code no closure bounds keys on (a launch with a hook,
    `provenance_code`; a rung, a verified mapping, a saved source's export)."""
    return _digest(_snapshot())


def _digest(modules: Collection[str]) -> str:
    """The digest of `modules` by their paths and contents as this process's snapshot holds them."""
    files = _snapshot()
    digest = hashlib.sha256()
    for module in sorted(modules):
        relative, contents = files[module]
        digest.update(f"{relative}\0{contents}\0".encode())
    return digest.hexdigest()


def provenance(values: Sequence[object]) -> frozenset[str] | None:
    """The modules of Dew's defining `values` and every value they hold (a dataclass's fields', a
    collection's members): a class's own module, any other value's class's, none for a library's or the
    user's own, whose release or source states it. None, the whole package, where any is a function (a
    hook, a prepared one): a kernel's identity traces a hook at one tile alone
    (`attention.hook_program`), and no value names the code a wrapper calls (row 10743)."""
    held = [part for value in values for part in _held(value)]
    if any(callable(part) and not isinstance(part, type) for part in held):
        return None
    modules = {(part if isinstance(part, type) else type(part)).__module__ for part in held}
    return frozenset(module for module in modules if _path(module) is not None)


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


def _held(value) -> Sequence[object]:
    """`value` and every value it holds: a dataclass's fields', a sequence's or a set's members, a mapping's
    values, each once however often it is held, so a graph that holds itself (a cyclic hook graph's) ends."""
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
    return found


class _Changed(Exception):
    """A source file is not the one this process's snapshot of Dew's source holds (`_snapshot`)."""


def _listed() -> dict[str, tuple[str, str]]:
    """Each module of Dew's packages as its source is now: its file's path under `SOURCE` and the digest of
    its contents, by its name."""
    files = {}
    for package in PACKAGES:
        for path in sorted((SOURCE / package).rglob("*.py")):
            relative = path.relative_to(SOURCE)
            # The release's version, which no code of Dew's computes from, is no part of its source here: a
            # release that moves it alone keeps every record (`VERSION_FILE`).
            if relative == Path("dew", VERSION_FILE):
                continue
            name = ".".join(relative.with_suffix("").parts).removesuffix(".__init__")
            # A package's `__init__` is what its name imports where a module file of that name is beside it.
            shadowed = name in files and files[name][0].endswith("/__init__.py")
            if (contents := _hashed(path)) is not None and not shadowed:
                files[name] = (relative.as_posix(), contents)
    return files


def _hashed(path: Path) -> str | None:
    """The digest of `path`'s contents; None for a file removed since it was listed, which is absent."""
    with contextlib.suppress(FileNotFoundError):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return None


@functools.cache
def _snapshot() -> dict[str, tuple[str, str]]:
    """Dew's source as this process first read it (`_listed`): which modules exist, where, and what each
    holds. A closure is walked from it (`_path`, `_source`) and a digest computed from it, one tree
    throughout, however the files change after."""
    return _listed()


def _kept_closure(roots: tuple[str, ...]) -> tuple[frozenset[str], tuple[str, ...]]:
    """`closure(*roots)`, kept beside the compilation cache the verdicts it keys live in
    (`dew._cache.default_compilation_cache_dir`) under the tree's digest (`whole_digest`), the Python that
    parsed it and the roots, so a later start of the same tree reads it, walking nothing; one that does not
    name its roots, or names a module the tree has not, is walked again. A directory that cannot be read or
    written (a read-only home) keeps none, a miss logged, and a bucket's (gs://) is not written to: the walk's
    closure serves the process."""
    named = hashlib.sha256(json.dumps(roots).encode()).hexdigest()[:16]
    python = "{}.{}".format(*sys.version_info[:2])
    cache = default_compilation_cache_dir()
    if "://" in cache:
        # A bucket's compilation cache (gs://), whose client fails in ways of its own: a closure is kept on a
        # local file system alone, and walked again each process elsewhere.
        return closure(*roots)
    kept = Path(cache) / "dew-code" / f"{whole_digest()}-{python}-{named}"
    with contextlib.suppress(OSError, ValueError):
        found = json.loads(kept.read_text())
        if isinstance(found, dict) and _sound(found, roots):
            return frozenset(found["modules"]), tuple(found["sites"])
    modules, sites = closure(*roots)
    try:
        kept.parent.mkdir(parents=True, exist_ok=True)
        write_atomically(kept, json.dumps({"roots": list(roots), "modules": sorted(modules),
                                           "sites": list(sites)}))
    except OSError as refused:
        _log.warning("the closure of %s is not kept (%s): a later start walks it again", ", ".join(roots),
                     refused)
    return modules, sites


def _sound(found: dict[str, JSON], roots: tuple[str, ...]) -> bool:
    """Whether `found` is a closure of `roots` as `_kept_closure` keeps one."""
    modules, sites = found.get("modules"), found.get("sites")
    if found.get("roots") != list(roots) or not isinstance(modules, list) or not isinstance(sites, list):
        return False
    return (all(isinstance(module, str) and _path(module) is not None for module in modules)
            and set(roots) <= set(modules) and all(isinstance(site, str) for site in sites))


def closure(*roots: str) -> tuple[frozenset[str], tuple[str, ...]]:
    """The modules of Dew's `roots` name and their code can import, and the dynamic import sites among
    them (`dew._imports.Walk.closure` over this process's snapshot of the tree). A root naming no module of
    Dew's is refused."""
    return _walk().closure(*roots)


@functools.cache
def _walk() -> Walk:
    """The walk of this process's snapshot of Dew's source (`_snapshot`), each module read checked against it
    (`_source`)."""
    return Walk({module: relative for module, (relative, _) in _snapshot().items()}, _source, SELECTED)


def _path(module: str) -> Path | None:
    """The source file of Dew's `module` in this process's snapshot of the tree (`_snapshot`), None where
    it holds none (a library's, an absent one)."""
    found = _snapshot().get(module)
    return None if found is None else SOURCE / found[0]


@functools.cache
def _source(module: str) -> bytes:
    """`module`'s source as this process's snapshot of the tree holds it (`_snapshot`); `_Changed` where
    its file holds another now."""
    path = _path(module)
    if path is None:
        raise LookupError(f"no module of Dew's is named {module}")
    try:
        contents = path.read_bytes()
    except FileNotFoundError as missing:
        raise _Changed(f"{path} was removed after this process first read Dew's source") from missing
    if hashlib.sha256(contents).hexdigest() != _snapshot()[module][1]:
        raise _Changed(f"{path} changed after this process first read Dew's source")
    return contents
