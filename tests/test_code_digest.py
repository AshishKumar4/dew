"""The digest of a producer's code (`dew._code.code_digest`), which a kernel launch's stored verdict and
choice and a program's kept communicators key on (row 10743): an edit outside a store's code keeps its
records, one inside it computes them again, and a closure holding an import by a computed name no review
covers is keyed by the whole package."""

import shutil
from pathlib import Path

from process_support import REPO_ROOT, fresh

DIGESTS = """
import json, logging
logged = []


class Kept(logging.Handler):
    def emit(self, record):
        logged.append(record.getMessage())


logging.getLogger("dew._code").addHandler(Kept())
from dew._code import code_digest, whole_digest
print(json.dumps({name: code_digest(*roots) for name, roots in (
    ("certification", ("dew.nn.kernels.templates.report",)), ("communicators", ("dew.communicators",)),
    ("reader", ("dew._reader",)), ("computed", ("dew._computed",)))} | {"whole": whole_digest(),
                                                                         "logged": logged}))
"""

READER = '''from typing import TYPE_CHECKING

from dew.nn.kernels import gpu_target

if TYPE_CHECKING:
    from dew import _unread
else:
    if TYPE_CHECKING:
        from dew import _unread
    else:
        from dew import _spare


def scan():
    from dew.algebra import Scan

    return Scan, gpu_target, _spare
'''
"""A module whose imports are a lazy table's names (one a `dict.fromkeys` states, one read in a function)
and a nested `if TYPE_CHECKING:`'s live `else`."""

COMPUTED = '''from importlib import import_module as load


def loaded(name):
    return load(name)
'''
"""A module importing by a computed name, through an alias of `import_module`."""


def test_a_stores_digest_follows_its_code_and_an_unreviewed_import_keys_the_whole_package(tmp_path):
    """An edit of Dew's code outside the certification's and the communicators' (a trainer's), of a module
    only a `TYPE_CHECKING` block imports and of the release's version keeps every digest; an edit of a
    module imported by a lazy table's name, a function's own import or a nested guard's `else` changes the
    digest of the code importing it, and one of a GPU family's template (`mosaic_sm90`) the
    certification's. Neither store's code holds an unreviewed dynamic import: neither is the whole
    package's, and nothing of theirs is logged. A module importing by a computed name, through an alias of
    `import_module`, is keyed by the whole package, which any edit changes, its site logged by the
    expression it imports."""
    tree = tmp_path / "src"
    for name in ("dew", "_dew_torch"):
        shutil.copytree(REPO_ROOT / "src" / name, tree / name, ignore=shutil.ignore_patterns("__pycache__"))
    package = tree / "dew"
    (package / "_reader.py").write_text(READER)
    (package / "_computed.py").write_text(COMPUTED)
    for name in ("_unread.py", "_spare.py"):
        (package / name).write_text("")

    def digests(*edits: Path) -> dict:
        for path in edits:
            path.write_text(path.read_text() + "\n# edited\n")
        return fresh(DIGESTS, source=tree, platforms="cpu", cache_dir=tmp_path / "xla",
                     variables={"XDG_CACHE_HOME": str(tmp_path)})

    before = digests()
    assert before["computed"] == before["whole"]
    assert before["whole"] not in (before["certification"], before["communicators"], before["reader"])
    assert len(before["logged"]) == 1 and "dew._computed:loaded:name" in before["logged"][0], before
    kept = digests(package / "training" / "rungs.py", package / "_unread.py", package / "_version.py")
    assert {name: kept[name] == before[name] for name in ("certification", "communicators", "reader",
                                                          "computed")} == {
        "certification": True, "communicators": True, "reader": True, "computed": False}
    for parts in (("_spare.py",), ("algebra", "scan.py"), ("nn", "kernels", "generation.py")):
        now = digests(package.joinpath(*parts))
        assert now["reader"] != kept["reader"], parts
        kept = now
    now = digests(package / "nn" / "kernels" / "templates" / "mosaic_sm90.py")
    assert now["certification"] != kept["certification"]


IMPORTED = """
import json, sys
import {module}
from dew._code import closure
held = closure("{module}")[0]
print(json.dumps(sorted(name for name in sys.modules
                        if name.partition(".")[0] in ("dew", "_dew_torch") and name not in held)))
"""


def test_every_module_a_stores_code_imports_is_in_its_closure():
    """A store's closure holds every module of Dew's that importing its code imports, each in a fresh
    process: the certification's, an attention op's and the communicators'."""
    for module in ("dew.nn.kernels.templates.report", "dew.nn.kernels.templates.attention_mosaic",
                   "dew.communicators"):
        assert fresh(IMPORTED.format(module=module), platforms="cpu") == [], module


TREES_LEFT = """
import ast, gc, json
from dew._code import closure
closure("dew.nn.kernels.templates.report")
gc.collect()
print(json.dumps(sum(isinstance(each, ast.AST) for each in gc.get_objects())))
"""


def test_a_closure_holds_none_of_the_source_trees_its_walk_parsed():
    """A store's closure, its walk of Dew's source done, holds no syntax tree it parsed alive, in a fresh
    process: the walk keeps the imports it read of each module, not its tree, where a process held every
    module's (1.1M nodes of a cold recurrent step's 4.7M live objects, its full collection 3.9 s and its
    exit 18 s, row 8420)."""
    assert fresh(TREES_LEFT, platforms="cpu") == 0


KERNELS = """
import json
from pathlib import Path
import dew.nn.kernels
from dew._code import REVIEWED, closure
root = Path(dew.nn.kernels.__file__).parent
parts = [("dew", "nn", "kernels", *path.relative_to(root).with_suffix("").parts)
         for path in root.rglob("*.py")]
modules = sorted(".".join(each).removesuffix(".__init__") for each in parts)
print(json.dumps({module: sorted(set(closure(module)[1]) - REVIEWED.keys()) for module in modules}))
"""


def test_every_kernel_modules_code_is_keyed_by_its_closure_no_import_site_unreviewed():
    """Every module of `dew.nn.kernels`, each a launch's op's possible home (`op_provenance`), holds in its
    closure no dynamic import site `REVIEWED` does not: a site added unreviewed keys a family's every launch
    by the whole package (train 1159's review F6: the recurrence family's closures reached the codec's and the
    checkpoint parts' sites, unreviewed, through short_conv's linear layer)."""
    found = fresh(KERNELS, platforms="cpu")
    assert len(found) > 100 and {module: sites for module, sites in found.items() if sites} == {}


KEPT = """
import json, logging, os
logged = []


class Kept(logging.Handler):
    def emit(self, record):
        logged.append(record.getMessage())


logging.getLogger("dew._code").addHandler(Kept())
from dew._code import code_digest
digest = code_digest("dew.communicators")
kept = os.environ["JAX_COMPILATION_CACHE_DIR"] + "/dew-code"
print(json.dumps({"digest": digest, "kept": sorted(os.listdir(kept)) if os.path.isdir(kept) else [],
                  "logged": logged}))
"""


def test_a_closure_is_kept_beside_the_compilation_cache_and_an_unwritable_one_keeps_none(tmp_path):
    """A walked closure is kept beside the compilation cache the verdicts it keys live in (the directory
    JAX_COMPILATION_CACHE_DIR names), and a later start's digest is the same; a cache where it cannot be
    written (a file where its directory would be, refused whatever the process's user) keeps none, logs
    the miss and gives the same digest (row 11743: a read-only home raised before a prewarmed verdict was
    read)."""
    writable, refusing = tmp_path / "xla", tmp_path / "refusing"
    refusing.mkdir()
    (refusing / "dew-code").write_text("a file, no directory")
    first = fresh(KEPT, platforms="cpu", cache_dir=writable)
    again = fresh(KEPT, platforms="cpu", cache_dir=writable)
    refused = fresh(KEPT, platforms="cpu", cache_dir=refusing)
    assert len(first["kept"]) == 1 and first["logged"] == [] and again == first
    assert refused["digest"] == first["digest"]
    assert len(refused["logged"]) == 1 and "not kept" in refused["logged"][0], refused
