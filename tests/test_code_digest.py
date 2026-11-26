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


def _edited_run(script: str, tree: Path, tmp_path: Path, edits: tuple[Path, ...]) -> dict:
    """`script`'s printed JSON, run fresh over the copied package `tree` once each of `edits` has a line
    appended."""
    for path in edits:
        path.write_text(path.read_text() + "\n# edited\n")
    return fresh(script, source=tree, platforms="cpu", cache_dir=tmp_path / "xla",
                 variables={"XDG_CACHE_HOME": str(tmp_path)})


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
        return _edited_run(DIGESTS, tree, tmp_path, edits)

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


READ = """
import json, sys
from pathlib import Path
from dew import _code
source = str(_code.SOURCE)
opened, listed = set(), set()


def heard(event, arguments):
    named = arguments[0] if arguments and isinstance(arguments[0], str) else None
    if named is None or not named.startswith(source):
        return
    if event == "open" and named.endswith(".py"):
        opened.add(str(Path(named).relative_to(source)))
    elif event in ("os.scandir", "os.listdir"):
        listed.add(named)


sys.addaudithook(heard)
_code.code_digest("dew.communicators")
held = _code.closure("dew.communicators")[0]
print(json.dumps({"opened": sorted(opened), "listed": sorted(listed),
                  "closure": sorted(_code._snapshot()[name] for name in held)}))
"""


def test_a_digest_reads_its_closures_files_alone(tmp_path):
    """A code digest reads the files of its closure alone and lists no directory of Dew's: a test of a
    record's code then reads, and so is keyed by, the code its record keys on (2026-11-23: a snapshot reading
    every file whole made each such test's CI key every file, and any edit ran 61% of the suite again)."""
    found = fresh(READ, platforms="cpu", cache_dir=tmp_path / "xla")
    assert found["opened"] and set(found["opened"]) <= set(found["closure"])
    assert found["listed"] == []


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
# A tree's own nodes, not the operator and context singletons every parse shares.
print(json.dumps(sum(isinstance(each, (ast.mod, ast.stmt, ast.expr)) for each in gc.get_objects())))
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


logging.getLogger("dew._cache").addHandler(Kept())
logging.getLogger("dew._code").addHandler(Kept())
from dew._code import code_digest
digest = code_digest("dew.communicators")
kept = os.environ["JAX_COMPILATION_CACHE_DIR"] + "/dew-code"
print(json.dumps({"digest": digest, "kept": sorted(os.listdir(kept)) if os.path.isdir(kept) else [],
                  "logged": logged}))
"""


KEPT_AGAIN = """
import json, sys
from dew._code import code_digest
print(json.dumps(code_digest("dew.communicators")))
"""


def test_a_kept_closure_is_walked_again_where_a_lookup_it_made_finds_another_file(tmp_path):
    """A kept closure is held to its walk's lookups as well as its modules' contents: a module added that a
    guarded import of the closure would now find (an absent name the walk looked up) walks it again, its
    module in the digest, while an edit beside the closure keeps it."""
    tree = tmp_path / "src"
    for name in ("dew", "_dew_torch"):
        shutil.copytree(REPO_ROOT / "src" / name, tree / name, ignore=shutil.ignore_patterns("__pycache__"))
    (tree / "dew" / "_guarded.py").write_text("try:\n    from dew import _maybe\nexcept ImportError:\n"
                                              "    _maybe = None\n")
    communicators = tree / "dew" / "communicators.py"
    communicators.write_text(communicators.read_text() + "\nfrom dew import _guarded\n")

    def digest() -> str:
        return fresh(KEPT_AGAIN, source=tree, platforms="cpu", cache_dir=tmp_path / "xla")

    first = digest()
    (tree / "dew" / "training" / "rungs.py").write_text((tree / "dew" / "training" / "rungs.py").read_text()
                                                        + "\n# edited\n")
    assert digest() == first
    (tree / "dew" / "_maybe.py").write_text("VALUE = 1\n")
    assert digest() != first


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


WALKED = """
import json, sys
from dew import _code
walks = []
walked = _code.closure
_code.closure = lambda *roots: walks.append(roots) or walked(*roots)
if len(sys.argv) > 1:
    _code.SELECTED = {**_code.SELECTED, sys.argv[1]: "UNQUANTIZED"}
_code.code_digest("dew.communicators")
print(json.dumps(len(walks)))
"""


def test_a_kept_closure_is_walked_again_where_how_closures_are_walked_changed(tmp_path):
    """A closure is kept under how it was walked as well as its roots: a site the walker follows by its table
    added (`SELECTED`) walks it again, though its code imports no walker, while a start under the same
    walker reads it walking nothing."""
    cache = tmp_path / "xla"
    assert fresh(WALKED, platforms="cpu", cache_dir=cache) == 1
    assert fresh(WALKED, platforms="cpu", cache_dir=cache) == 0
    assert fresh(WALKED, "dew.added:site:module", platforms="cpu", cache_dir=cache) == 1


MADE_A_PACKAGE = """
import json, sys
import dew.communicators
from dew import _code
if len(sys.argv) > 1:
    module = _code.SOURCE / "dew" / "communicators.py"
    package = module.with_suffix("")
    package.mkdir()
    module.rename(package / "__init__.py")
print(json.dumps(_code.code_digest("dew.communicators")))
"""


def test_a_digest_of_a_module_made_a_package_after_its_import_is_its_processs_alone(tmp_path):
    """A module made a package after a process imported it keys that process's records by a digest no other
    process computes: its imports were read from a file the digest no longer hashes. Unchanged, a later
    start's digest is the same."""
    tree = tmp_path / "src"
    for name in ("dew", "_dew_torch"):
        shutil.copytree(REPO_ROOT / "src" / name, tree / name, ignore=shutil.ignore_patterns("__pycache__"))

    def digest(*arguments: str) -> str:
        return fresh(MADE_A_PACKAGE, *arguments, source=tree, platforms="cpu", cache_dir=tmp_path / "xla")

    kept = digest()
    assert digest() == kept
    assert digest("made") != kept


WHOLE = """
import importlib, json
from dew._code import code_digest, whole_digest
importlib.import_module("dew.extra-name")
import dew.shadowed
print(json.dumps([whole_digest(), code_digest("dew.extra-name")]))
"""


def test_the_whole_packages_digest_reads_a_file_whose_name_is_no_identifier(tmp_path):
    """The whole package's digest reads every Python file of Dew's, one whose name no import statement can
    state among them, which importlib loads by its name (`dew/extra-name.py`), whose own digest names it
    too; and a module file a package of its name shadows, imported, keys a digest every start computes
    alike (row 3944's review: the shadowed file read as a module imported from another file)."""
    tree = tmp_path / "src"
    for name in ("dew", "_dew_torch"):
        shutil.copytree(REPO_ROOT / "src" / name, tree / name, ignore=shutil.ignore_patterns("__pycache__"))
    (tree / "dew" / "extra-name.py").write_text("VALUE = 1\n")
    (tree / "dew" / "shadowed").mkdir()
    (tree / "dew" / "shadowed" / "__init__.py").write_text("VALUE = 1\n")
    (tree / "dew" / "shadowed.py").write_text("VALUE = 2\n")
    first = fresh(WHOLE, source=tree, platforms="cpu")
    assert fresh(WHOLE, source=tree, platforms="cpu") == first
    (tree / "dew" / "extra-name.py").write_text("VALUE = 2\n")
    again = fresh(WHOLE, source=tree, platforms="cpu")
    assert again[0] != first[0] and again[1] != first[1]


FOOTPRINT = """
import json, os, sys
from pathlib import Path
from dew import _code
from dew._cache import read_kept
if len(sys.argv) > 1:
    _code.whole_digest()
_code.code_digest("dew.communicators")
[kept] = (Path(os.environ["JAX_COMPILATION_CACHE_DIR"]) / "dew-code").iterdir()
print(json.dumps(sorted(read_kept(kept)["looked"])))
"""


def test_a_kept_closure_holds_the_lookups_of_its_own_walk_alone(tmp_path):
    """A kept closure is held to the names its own walk looked up, whatever else the process looked up
    before it (the whole package's digest looks up every module): a closure kept after one holds the same
    lookups as one kept first."""
    alone = fresh(FOOTPRINT, platforms="cpu", cache_dir=tmp_path / "alone")
    after = fresh(FOOTPRINT, "whole", platforms="cpu", cache_dir=tmp_path / "after")
    assert alone == after


def test_a_callable_instance_of_a_class_of_dews_is_its_class_code_and_any_other_callable_is_all_of_it():
    """A value's provenance (`dew._code.provenance`) holds a callable instance of a class of Dew's at its
    module's top level by that class's module, as any other instance (a Flax module, a template's
    `Capped` hook); a function, a `functools.partial` over one of Dew's, an instance of a class made in a
    function, and a class of Dew's holding a function, are code no class states: all of Dew's (None)."""
    import functools

    from dew._code import provenance
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.kernels.templates import attention as template

    def local():
        class Made:
            def __call__(self):
                return None

        return Made()

    model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=1, num_heads=1, mlp_features=8,
                              max_seq_len=8)
    assert provenance([template.Capped(30.0)]) == {"dew.nn.kernels.templates.attention"}
    assert "dew.nn.backbones.causal_transformer" in (provenance([model]) or ())
    assert [provenance([held]) for held in (
        local(), provenance, functools.partial(provenance, ()),
        template.Biased(lambda logits, q, k: logits))] == [None] * 4


RUNG_CODE = """
import dataclasses
import json

import jax
import optax

from dew._code import op_provenance, provenance, provenance_code, whole_digest
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.kernels.templates.attention import Capped
from dew.nn.kernels.templates.attention_triton import TritonForward
from dew.objectives.lm import LMObjective
from dew.objectives.residuals import Residuals
from dew.objectives.supervised import Supervised
from dew.optim import Adam, OptimConfig, Optax
from dew.training import Trainer
from dew.training.rungs import _code


class Fitted(Residuals):
    def fresh_variables(self, key, held):
        return {"params": {"p": 0.0}}

    def residuals(self, variables, batch, step):
        return batch["x"] - variables["params"]["p"]


model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=1, num_heads=1, mlp_features=8,
                          max_seq_len=8)
lm = LMObjective(model, 4)
configured = Trainer(lm, OptimConfig(optimizer=Adam(), lr=1e-3), key=jax.random.key(0))._settings()
handed = Trainer(lm, optax.sgd(1e-2), key=jax.random.key(0))._settings()
criterion = Supervised(model, lambda outputs, batch: outputs.sum(), inputs="text")
def factory(learning_rate) -> optax.GradientTransformation:
    return optax.sgd(learning_rate)


elsewhere = OptimConfig(optimizer=Optax(path="__main__:factory"), lr=1e-3)
pathed = dataclasses.replace(configured, optimizer_source=elsewhere)
whole = [whole_digest()]
launch = op_provenance(TritonForward(score_mod=Capped(30.0)), {})
print(json.dumps({"lm": _code(configured, lm.program_key(), None),
                  "opaque": [_code(handed, lm.program_key(), None) == whole,
                             _code(pathed, lm.program_key(), None) == whole,
                             _code(dataclasses.replace(configured, objective=Fitted()), (), None) == whole,
                             _code(dataclasses.replace(configured, objective=criterion), (), None) == whole],
                  "launch": provenance_code("dew.nn.kernels.templates.report", launch) == whole,
                  "capped": provenance_code("dew.nn.kernels.templates.report", provenance([Capped(30.0)]))}))
"""


def test_a_rung_records_code_is_its_step_its_objective_and_its_models_and_none_else(tmp_path):
    """A rung record keys on the code that builds its step (`dew.training.rungs._code`), not every file of
    Dew's, as the compiler hands it the run's settings: a file no step's code imports leaves an LM's whose
    optimizer is a config alike, and its model's module moves it. A step whose code no value names keys on
    all of Dew's: an optimizer handed in as a transformation or named by an import path outside Optax, a
    user's objective (a model-free `Residuals`'s subclass), and one of Dew's holding a user's criterion.
    Every file's digest made each rung test's CI key every file of Dew's (22% of the suite's rows). A hook
    of a class of Dew's (`Capped`), keyed by every file before, keys on its class's module: an edit there
    moves its code, an unrelated file's leaves it; a launch holding it (a Triton forward's score hook,
    `op_provenance`) holds it prepared, its runner a function, and keys on all of Dew's as before."""
    tree = tmp_path / "src"
    for name in ("dew", "_dew_torch"):
        shutil.copytree(REPO_ROOT / "src" / name, tree / name, ignore=shutil.ignore_patterns("__pycache__"))
    package = tree / "dew"

    def codes(*edits: Path) -> dict:
        return _edited_run(RUNG_CODE, tree, tmp_path, edits)

    (package / "_spare_rung.py").write_text("")
    before = codes()
    assert before["opaque"] == [True, True, True, True] and before["launch"]
    assert codes(package / "_spare_rung.py") == before
    hooked = codes(package / "nn" / "kernels" / "templates" / "attention.py")
    assert hooked["capped"] != before["capped"]
    model = codes(package / "nn" / "backbones" / "causal_transformer.py")
    assert model["lm"] != hooked["lm"]
