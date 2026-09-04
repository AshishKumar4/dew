"""The code in the documentation runs.

Every ```python block in README.md and docs/**/*.md is executed in file order,
in one namespace per file, on CPU. The namespace starts with the small real
objects a snippet refers to by name without building them (`data`, `steps`,
`prompt`, `key`, a tiny `model`, `objective`, `trainer`, `state`, `process`,
`inputs`, `text`, `fields`), so a block that trains does so for a few steps on
random records, and a block that names a symbol the library no longer has, or
calls it with arguments it no longer takes, fails here rather than in a
reader's terminal.

A block whose first line is `# runs elsewhere: <reason>` is compiled but not
run; the reason is for the reader (a download, a run directory on disk, a
process pool). A syntax error or a misspelled import in it still fails.

The tutorials are notebooks, so they are checked rather than executed: every
`dew` name a code cell imports has to exist, and every call of one of those
names has to match its signature. That is what a markdown block gets for free
by being run, and without it a notebook keeps naming a symbol the library
deleted until a reader finds out.

The seven tutorials on the pre-registry API are named in `PENDING` and their
check is a strict xfail, so the marker erases itself: the first notebook
ported to the built API fails as an unexpected pass until its name comes off
the list, and a new notebook is checked from the day it is added.
"""

import ast
import importlib
import inspect
import itertools
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
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

RES, TOKENS, BATCH = 16, 8, 8


def blocks(path: Path) -> list[tuple[int, str]]:
    text = path.read_text()
    return [(text.count("\n", 0, match.start()) + 2, match.group(1))
            for match in BLOCK.finditer(text)]


def tiny_world(tmp_path):
    """The objects a snippet assumes exist, at a size that trains in seconds."""
    import jax
    import optax

    import dew
    from dew import Checkpoints, Condition, Field, InputSpec, MeshSpec, Trainer, models, presets
    from dew.data import ByteTokenizer, Dataset
    from dew.inputs import CharTable
    from dew.objectives.diffusion import DiffusionObjective

    def batches():
        rng = np.random.RandomState(0)
        encoder = CharTable.from_pretrained(tokens=TOKENS)
        while True:
            yield {"image": rng.randint(0, 256, (BATCH, RES, RES, 3), np.uint8),
                   "text": encoder.tokenize(["a flower"] * BATCH),
                   "label": rng.randint(0, 5, (BATCH,), np.int32)}

    data = Dataset(train=batches, val=lambda: itertools.islice(batches(), 1),
                   records=4 * BATCH, batch=BATCH)
    text = CharTable.from_pretrained(tokens=TOKENS)
    inputs = InputSpec(Field("image", (RES, RES, 3)), {"textcontext": Condition(text)})
    fields = dict(patch_size=4, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1)
    model = models.SimpleDiT(**fields)
    process = presets.EDM()()
    objective = DiffusionObjective(model, process, inputs, steps=2)
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0), mesh=MeshSpec(fsdp=1),
                      checkpoints=Checkpoints(str(tmp_path / "runs" / "flowers")))
    state = trainer.fit(data, steps=2, log_every=1)
    return dict(
        __name__="__docs__", dew=dew, jax=jax, optax=optax, np=np,
        data=data, steps=2, key=jax.random.key(0), text=text, inputs=inputs, fields=fields,
        model=model, process=process, objective=objective, trainer=trainer, state=state,
        prompt=np.asarray([list(b"ROMEO:")], np.int32), tokenizer=ByteTokenizer(),
        optimizer=optax.adam(1e-3), meta={"vocab_size": 256},
    )


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_the_documented_code_runs(path, tmp_path, monkeypatch):
    found = blocks(path)
    if not found:
        pytest.skip("no python blocks")
    monkeypatch.chdir(tmp_path)
    namespace = tiny_world(tmp_path)
    # Concept pages name jax bare where a snippet builds an array.
    if path != ROOT / "README.md":
        namespace.update(dict(jnp=__import__("jax.numpy", fromlist=["x"]), jax=__import__("jax")))
    for line, source in found:
        where = f"{path.relative_to(ROOT)}:{line}"
        code = compile(source, where, "exec")
        if ELSEWHERE.match(source):
            continue
        try:
            exec(code, namespace)
        except Exception as error:  # noqa: BLE001 - the report names the block
            raise AssertionError(f"{where} failed: {type(error).__name__}: {error}") from error


def test_the_api_page_is_the_code():
    """docs/api.md is generated; a module whose exports changed regenerates it."""
    import subprocess
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
