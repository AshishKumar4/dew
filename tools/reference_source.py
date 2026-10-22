"""Published reference code, fetched at a pinned URL and run as written.

The fixture generators take a reference's own definitions rather than a
transcription of them: `selected` picks the named classes, functions and
assignments of a source file (or the members of one class in it) in file
order, and refuses a name the file no longer defines, so a moved or renamed
upstream definition fails the generator instead of leaving a stale one in
scope. `defined_in` (`definitions` for a URL) runs them in the generator's
scope; a generator that retargets a dtype, drops a decorator or builds a
class around them does so between `selected` and `ran`. `simple_statements`
takes single lines out of a published function. `verified` keeps a reference
file on disk, fetched once and checked against its pinned SHA-256 on every
use; `imported` runs such a file as a module of its own, and `packaged` and
`module_at` let an upstream's files import each other, as `published` does
for a package of files it fetches.
`stub_torch_utils` stands in for the two `torch_utils` names NVIDIA's EDM and
EDM2 networks import. `saved_fixture` writes a generator's arrays with its
provenance as the suite reads them, and `recipe_images` draws the images a
recipe names.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import sys
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

TIMEOUT = 120
"""Seconds a fetch may take."""
CACHE = Path.home() / ".cache" / "dew" / "upstream"
"""Where `verified` keeps fetched reference files, each under its own source's path."""


def stub_torch_utils() -> None:
    """NVIDIA's `torch_utils` as EDM and EDM2's networks import it:
    `persistence.persistent_class` is only pickling metadata, and
    `misc.const_like` a tensor of a value on another tensor's dtype and device."""
    import sys
    import types

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


def module_at(name: str, path: Path) -> ModuleType:
    """The file at `path` run as the module `name`, registered under it before it runs, so the modules it
    imports, and those that import it, find it."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def imported(url: str, local: Path, digest: str, name: str) -> ModuleType:
    """The file at `url`, kept at `local` and checked against `digest` (`verified`), run as the module
    `name` (`module_at`)."""
    return module_at(name, verified(url, local, digest))


def packaged(names: Iterable[str]) -> None:
    """Empty packages named `names`, registered, for an upstream's modules run one by one (`module_at`) to
    import each other under their own package names."""
    for name in names:
        package = ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package


HOSTS = {"github": "https://raw.githubusercontent.com/{repo}/{revision}/{path}",
         "hub": "https://huggingface.co/{repo}/resolve/{revision}/{path}"}
"""Where a published file is read: a GitHub repository's raw file, or a Hugging Face repository's."""


def published(package: str, repo: str, revision: str, files: Iterable[str], *, host: str = "github",
              within: str = "") -> list[ModuleType]:
    """`files`, the names of files in the directory `within` of `repo` at `revision` on `host` (`HOSTS`),
    fetched once into the cache under the repository and revision, and imported in their order as the
    modules of the package `package`, which that cache directory puts on the import path."""
    cache = CACHE / repo / revision
    directory = cache / package
    directory.mkdir(parents=True, exist_ok=True)
    names = list(files)
    for name in names:
        if not (directory / name).is_file():
            path = f"{within}/{name}" if within else name
            url = HOSTS[host].format(repo=repo, revision=revision, path=path)
            (directory / name).write_text(fetched(url))
    (directory / "__init__.py").touch()
    sys.path.insert(0, str(cache))
    return [importlib.import_module(f"{package}.{name.removesuffix('.py')}") for name in names]


def left_padded(ids):
    """`ids` as the recorders run them left-padded: the second row's first three tokens padded away. The
    attention mask, the padded ids (0 where masked) and each token's position among its row's kept ones."""
    import numpy as np

    mask = np.ones_like(ids, bool)
    mask[1, :3] = False
    return mask, np.where(mask, ids, 0), np.maximum(np.cumsum(mask, axis=-1) - 1, 0)


def _defined(node: ast.stmt) -> set[str]:
    if isinstance(node, ast.FunctionDef | ast.ClassDef):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {target.id for target in node.targets if isinstance(target, ast.Name)}
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
