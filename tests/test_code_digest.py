"""The digest of a producer's code (`dew._code.code_digest`), which a kernel's stored verdicts and choices
and a program's kept communicators key on (row 10743): an edit outside a store's code keeps its records, one
inside it computes them again, and a closure holding an import by a computed name no review covers is keyed
by the whole package."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src"

DIGESTS = """
import json, logging
logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
from dew._cache import package_digest
from dew._code import code_digest, every_module
print(json.dumps({"kernels": code_digest(*every_module("dew.nn.kernels")),
                  "communicators": code_digest("dew.communicators"),
                  "synthetic": code_digest("dew._synthetic"), "whole": package_digest()}))
"""

SYNTHETIC = '''import importlib


def load(name):
    return importlib.import_module(name)
'''


def digests(tree: Path) -> tuple[dict[str, str], str]:
    """`DIGESTS` run over the Dew tree at `tree` in a process of its own, and what it logged."""
    done = subprocess.run([sys.executable, "-c", DIGESTS], check=True, capture_output=True, text=True, env={
        **os.environ, "PYTHONPATH": str(tree), "JAX_PLATFORMS": "cpu", "XDG_CACHE_HOME": str(tree / "cache")})
    return json.loads(done.stdout.strip().splitlines()[-1]), done.stderr


def edited(path: Path) -> None:
    path.write_text(path.read_text() + "\n# edited\n")


def test_a_stores_digest_follows_its_own_code_and_an_unreviewed_import_keys_it_by_the_whole_package(tmp_path):
    """An edit of Dew's code outside the kernels' and the communicators' (a trainer's) and of the release's
    version keeps both digests; an edit of a GPU family's template (`mosaic_sm90`) changes the kernels'.
    Neither store's code holds an unreviewed dynamic import, so neither is the whole package's, and nothing
    is logged; a module that imports by a computed name is keyed by the whole package, which any edit
    changes, and the site is named."""
    for name in ("dew", "_dew_torch"):
        shutil.copytree(SOURCE / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "dew" / "_synthetic.py").write_text(SYNTHETIC)
    before, logged = digests(tmp_path)
    assert before["synthetic"] == before["whole"]
    assert "dew._synthetic:load" in logged
    assert "dew.nn.kernels" not in logged and "dew.communicators" not in logged
    assert before["whole"] not in (before["kernels"], before["communicators"])

    edited(tmp_path / "dew" / "training" / "rungs.py")
    edited(tmp_path / "dew" / "_version.py")
    after, _ = digests(tmp_path)
    assert (after["kernels"], after["communicators"]) == (before["kernels"], before["communicators"])
    assert after["synthetic"] != before["synthetic"]

    edited(tmp_path / "dew" / "nn" / "kernels" / "templates" / "mosaic_sm90.py")
    assert digests(tmp_path)[0]["kernels"] != after["kernels"]


IMPORTED = """
import json, sys
import {module}
from dew._code import closure, every_module
held = closure(*{roots})[0]
print(json.dumps(sorted(name for name in sys.modules
                        if name.partition(".")[0] in ("dew", "_dew_torch") and name not in held)))
"""


def test_every_module_a_stores_code_imports_is_in_its_closure():
    """A store's closure holds every module of Dew's that importing its code imports: the kernels' (their
    certification, `report`) and the communicators', each in a fresh process."""
    for module, roots in (("dew.nn.kernels.templates.report", 'every_module("dew.nn.kernels")'),
                          ("dew.communicators", '("dew.communicators",)')):
        done = subprocess.run([sys.executable, "-c", IMPORTED.format(module=module, roots=roots)], check=True,
                              capture_output=True, text=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
        assert json.loads(done.stdout.strip().splitlines()[-1]) == [], module
