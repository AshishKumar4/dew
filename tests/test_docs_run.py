"""Execute user-documentation Python blocks from an empty namespace.

Each page gets a separate process and temporary working directory. Blocks
run in page order; setup must appear on the page. Download-dependent blocks
labelled runs elsewhere are syntax-checked only. Notebook checks validate
imports and signatures without claiming notebook execution.
"""

import ast
import importlib
import inspect
import json
import os
import re
import sys
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "README.md",
         *(x for x in sorted((ROOT / "docs").rglob("*.md"))
           if x.relative_to(ROOT).parts[1] not in ("design", "research"))]
BLOCK = re.compile(r"```python\n(.*?)```", re.S)
NOTEBOOKS = sorted((ROOT / "tutorials").glob("*.ipynb"))
PENDING = (
    # Written against the pre-registry API, down to the trainer, the model
    # builder and the sampler classes, and parked for a rewrite, not a port.
    # 01 is not here: it builds diffusion from scratch and names no dew symbol,
    # so it is checked like any other notebook.
    "02-train-a-diffusion-model.ipynb",
    "03-text-to-image-with-guidance.ipynb",
    "04-samplers-and-schedules.ipynb",
    "05-train-a-language-model.ipynb",
    "06-jepa-representation-learning.ipynb",
    "07-scaling-on-many-devices.ipynb",
    "08-load-a-pretrained-decoder.ipynb",
)
ELSEWHERE = re.compile(r"^\s*# runs elsewhere: \S")

def blocks(path: Path) -> list[tuple[int, str]]:
    text = path.read_text()
    return [(text.count("\n", 0, match.start()) + 2, match.group(1))
            for match in BLOCK.finditer(text)]


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_the_documented_code_runs(path, tmp_path):
    executable = []
    for line, source in blocks(path):
        location = f"{path.relative_to(ROOT)}:{line}"
        compile(source, location, "exec")
        if not ELSEWHERE.match(source):
            executable.append((location, source))
    if not executable:
        pytest.skip("no offline executable Python blocks")
    runner = (
        "import json, sys\n"
        "namespace = {'__name__': '__main__'}\n"
        "for location, source in json.loads(sys.argv[1]):\n"
        "    exec(compile(source, location, 'exec'), namespace)\n"
    )
    environment = dict(os.environ)
    for name in ("XLA_FLAGS", "JAX_DEFAULT_MATMUL_PRECISION", "JAX_NUM_CPU_DEVICES"):
        environment.pop(name, None)
    environment.update(PYTHONPATH=str(ROOT / "src"), JAX_PLATFORMS="cpu",
                       JAX_NUM_CPU_DEVICES="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run(
        [sys.executable, "-c", runner, json.dumps(executable)],
        cwd=tmp_path, capture_output=True, text=True, timeout=180,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_api_page_is_the_code():
    """docs/api.md is generated; a module whose exports changed regenerates it."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "api_page.py"), "--check"],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "JAX_PLATFORMS": "cpu"})
    assert result.returncode == 0, result.stderr


def notebook_source(path: Path) -> str:
    """The notebook's code cells as one module, with the lines a kernel eats.

    Magics and shell lines are not python, and a cell that ends in an
    expression is fine: nothing here runs.
    """
    cells = json.loads(path.read_text())["cells"]
    lines = []
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        for line in cell["source"]:
            stripped = line.lstrip()
            lines.append("" if stripped[:1] in ("%", "!", "?") else line.rstrip("\n"))
        lines.append("")
    return "\n".join(lines)


def dew_imports(tree) -> list[tuple[str, str, str, int]]:
    """`(module, name, bound_as, line)` for every dew name a cell imports."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "dew":
            for alias in node.names:
                found.append((node.module, alias.name, alias.asname or alias.name, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "dew":
                    found.append((alias.name, "", alias.asname or alias.name, node.lineno))
    return found


def resolved(module: str, name: str):
    """What the import binds, or None when the library does not have it."""
    try:
        held = importlib.import_module(module)
    except ImportError:
        return None
    if not name:
        return held
    if hasattr(held, name):
        return getattr(held, name)
    try:
        return importlib.import_module(f"{module}.{name}")
    except ImportError:
        return None


def keyword_errors(tree, bound: dict) -> list[str]:
    """Calls of an imported dew symbol whose keywords it does not take."""
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.keywords:
            continue
        target = node.func.id if isinstance(node.func, ast.Name) else None
        held = bound.get(target)
        if held is None or not (inspect.isclass(held) or inspect.isfunction(held)):
            continue
        try:
            signature = inspect.signature(held)
        except (TypeError, ValueError):
            continue
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            continue
        unknown = sorted({keyword.arg for keyword in node.keywords
                          if keyword.arg and keyword.arg not in signature.parameters})
        if unknown:
            problems.append(f"line {node.lineno}: {target}() takes no {unknown}")
    return problems


def notebook_cases() -> list:
    """One case per tutorial; the parked ones as a strict xfail."""
    parked = [pytest.mark.xfail(strict=True, reason="parked for a rewrite")]
    return [pytest.param(path, marks=parked if path.name in PENDING else [], id=path.name)
            for path in NOTEBOOKS]


def test_the_parked_list_names_tutorials_that_exist():
    """A renamed or deleted notebook cannot leave an entry behind that would
    quietly excuse a file nobody has."""
    assert set(PENDING) <= {path.name for path in NOTEBOOKS}


@pytest.mark.parametrize("path", notebook_cases())
def test_a_tutorial_names_symbols_the_library_has(path):
    """A notebook is not executed here, so its imports and its call keywords
    are checked instead: those are what rot when the API moves."""
    tree = ast.parse(notebook_source(path), filename=str(path))
    bound, missing = {}, []
    for module, name, alias, line in dew_imports(tree):
        held = resolved(module, name)
        if held is None:
            missing.append(f"line {line}: {module}.{name}" if name else f"line {line}: {module}")
        else:
            bound[alias] = held

    problems = missing + keyword_errors(tree, bound)
    assert not problems, f"{path.name} names what dew does not have:\n" + "\n".join(problems)
