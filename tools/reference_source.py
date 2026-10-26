"""Published reference code, fetched at a pinned URL and run as written.

The fixture generators take a reference's own definitions rather than a
transcription of them: `selected` picks the named classes, functions and
assignments of a source file (or the members of one class in it) in file
order, and refuses a name the file no longer defines, so a moved or renamed
upstream definition fails the generator instead of leaving a stale one in
scope. `defined_in` (`definitions` for a URL) runs them in the generator's
scope; a generator that retargets a dtype, drops a decorator or builds a
class around them does so between `selected` and `ran`. `simple_statements`
takes single lines out of a published function. `verified` keeps a
reference file on disk, fetched once and checked against its pinned SHA-256
on every use; `github` keeps one of a GitHub repository's, and
`published_files` a pinned set of them. `loaded` runs one file as a module,
and `package` imports a directory of them as a package whose own `__init__`
never runs (`hub_modules` a Hugging Face repo's remote code).
`stub_torch_utils` stands in for the two `torch_utils` names NVIDIA's EDM and
EDM2 networks import. `saved_fixture` writes a generator's arrays with its
provenance as the suite reads them, `git_head` names the commit of the
checkout a record came from, and `recipe_images` draws the images a recipe
names.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import subprocess
import sys
import types
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

TIMEOUT = 120
"""Seconds a fetch may take."""
CACHE = Path.home() / ".cache" / "dew" / "upstream"
"""Where `verified` keeps fetched reference files, each under its own source's path."""


def stub_torch_utils() -> None:
    """NVIDIA's `torch_utils` as EDM and EDM2's networks import it:
    `persistence.persistent_class` is only pickling metadata, and
    `misc.const_like` a tensor of a value on another tensor's dtype and device."""
    import torch

    persistence = types.ModuleType("torch_utils.persistence")
    persistence.persistent_class = lambda cls: cls
    misc = types.ModuleType("torch_utils.misc")
    misc.const_like = lambda ref, value: torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
    package = types.ModuleType("torch_utils")
    package.persistence, package.misc = persistence, misc
    sys.modules.update({"torch_utils": package, "torch_utils.persistence": persistence,
                        "torch_utils.misc": misc})


def saved_fixture(path: Path, arrays: dict, meta: dict) -> None:
    """`arrays` compressed to `path`, with `meta` as the JSON bytes of an
    array named "meta", the provenance a test reads back."""
    import json

    import numpy as np

    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def git_head(path: str | Path) -> str:
    """The commit the git checkout at `path` is at; a record without one cannot be traced, so a
    checkout git cannot read fails the run."""
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def dew_revision() -> str:
    """The commit of the Dew this process imports (`dew._checkout.revision`); an installed Dew has
    none, so a record of it could not be traced and the run fails at once."""
    from dew._checkout import revision

    found = revision()
    if found is None:
        raise SystemExit("the Dew this process imports is installed, not checked out, so a record cannot "
                         "name its commit; run from a checkout")
    return found


def recipe_images(recipe: dict):
    """The uint8 images a fixture's recipe names (`{"seed", "shape"}`): NumPy's legacy generator at its
    seed, which a test reproduces from the recipe the fixture records."""
    import numpy as np

    return np.random.RandomState(recipe["seed"]).randint(0, 256, tuple(recipe["shape"]), dtype=np.uint8)


def fetched(url: str) -> str:
    """The text of the file at `url`."""
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return response.read().decode()


def verified(url: str, local: Path, digest: str) -> Path:
    """`local`, the file at `url` fetched there if it is not yet, refused
    unless its SHA-256 is `digest`."""
    if not local.is_file():
        local.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            local.write_bytes(response.read())
    found = hashlib.sha256(local.read_bytes()).hexdigest()
    if found != digest:
        raise RuntimeError(f"{local} has SHA-256 {found}, not the pinned {digest}")
    return local


def github(repo: str, commit: str, path: str, digest: str) -> Path:
    """`path` of the GitHub repository `repo` at `commit`, kept under `CACHE` as `verified` keeps it."""
    return verified(f"https://raw.githubusercontent.com/{repo}/{commit}/{path}", CACHE / repo / commit / path,
                    digest)


def published_files(repo: str, commit: str, files: Mapping[str, str]) -> Path:
    """Where `github` keeps `repo` at `commit`, holding each of `files` (a path in it, to its pinned
    SHA-256)."""
    for path, digest in files.items():
        github(repo, commit, path, digest)
    return CACHE / repo / commit


def loaded(name: str, path: Path | str) -> types.ModuleType:
    """The Python file at `path` run as the module `name`, which `sys.modules` holds before it runs,
    so its dataclasses and pickles find their module, bound on its package where that is imported,
    as an import binds a submodule."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} is not a Python source file; is its checkout there?")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    parent, _, leaf = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], leaf, module)
    spec.loader.exec_module(module)
    return module


def package(name: str, directory: Path) -> types.ModuleType:
    """`directory` as the package `name`, in place of any `name` and modules of it imported before:
    `name.module` reads `directory`'s module.py and a subdirectory is a namespace package, while no
    `__init__.py` runs, so nothing a published package's own would import is needed. Each call
    imports its modules afresh, so two revisions of one package load side by side as the module
    objects each call's imports returned."""
    for imported in [key for key in sys.modules if key == name or key.startswith(f"{name}.")]:
        del sys.modules[imported]
    module = types.ModuleType(name)
    module.__path__ = [str(directory)]
    sys.modules[name] = module
    return module


def hub_modules(name: str, repo: str, revision: str, files: Sequence[str]) -> tuple[types.ModuleType, ...]:
    """The remote code `files` at the top of the Hugging Face repo `repo` at `revision`, fetched
    through the hub's own cache and each imported as a module of the package `name` (`package`), so
    their relative imports find each other."""
    from huggingface_hub import hf_hub_download

    paths = [Path(hf_hub_download(repo, file, revision=revision)) for file in files]
    package(name, paths[0].parent)
    return tuple(importlib.import_module(f"{name}.{path.stem}") for path in paths)


def _defined(node: ast.stmt) -> set[str]:
    if isinstance(node, ast.FunctionDef | ast.ClassDef):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {target.id for target in node.targets if isinstance(target, ast.Name)}
    if isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name):
        return {node.target.id}
    return set()


def selected(text: str, names: Iterable[str], *, within: str | None = None,
             filename: str = "<reference>") -> list[ast.stmt]:
    """The top-level statements of `text` defining `names` (or the members of
    its class `within`), in file order. Every name must be defined there."""
    body = ast.parse(text).body
    if within is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == within).body
    wanted = set(names)
    chosen = [node for node in body if _defined(node) & wanted]
    missing = wanted - set().union(*(_defined(node) for node in chosen))
    assert not missing, f"{filename} defines none of {sorted(missing)}"
    return chosen


def simple_statements(text: str, node: ast.AST, starts: Iterable[str]) -> list[str]:
    """The source of the first statement under `node` (`text`'s), in walk
    order, holding no block of its own, that begins with each of `starts`:
    lines a generator runs out of a function too long to run whole. Every
    one must be there."""
    starts = list(starts)
    found: dict[str, str] = {}
    blocks = ast.For | ast.If | ast.With | ast.While | ast.Try
    for child in ast.walk(node):
        if isinstance(child, ast.stmt) and not isinstance(child, blocks):
            source = ast.get_source_segment(text, child) or ""
            for start in starts:
                if source.startswith(start):
                    found.setdefault(start, source)
    missing = [start for start in starts if start not in found]
    assert not missing, f"the source no longer holds {missing}"
    return [found[start] for start in starts]


def ran(statements: Iterable[ast.stmt], scope: dict, filename: str = "<reference>") -> dict:
    """Run `statements` in `scope`, in order, and return `scope`."""
    module = ast.fix_missing_locations(ast.Module(body=list(statements), type_ignores=[]))
    exec(compile(module, filename, "exec"), scope)
    return scope


def defined_in(text: str, names: Iterable[str], scope: dict, *, within: str | None = None,
               filename: str = "<reference>") -> dict:
    """Run `selected(text, names, within=within)` in `scope`, and return it."""
    return ran(selected(text, names, within=within, filename=filename), scope, filename)


def definitions(url: str, names: Iterable[str], scope: dict, *, within: str | None = None) -> dict:
    """`defined_in` over the file at `url`."""
    return defined_in(fetched(url), names, scope, within=within, filename=url)
