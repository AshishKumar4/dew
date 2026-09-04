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
"""

import itertools
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
BLOCK = re.compile(r"```python\n(.*?)```", re.S)
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
    from dew.data import Dataset
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
        prompt=np.asarray([list(b"ROMEO:")], np.int32), optimizer=optax.adam(1e-3),
    )


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_the_documented_code_runs(path, tmp_path, monkeypatch):
    found = blocks(path)
    if not found:
        pytest.skip("no python blocks")
    monkeypatch.chdir(tmp_path)
    namespace = tiny_world(tmp_path)
    for line, source in found:
        where = f"{path.relative_to(ROOT)}:{line}"
        code = compile(source, where, "exec")
        if ELSEWHERE.match(source):
            continue
        try:
            exec(code, namespace)
        except Exception as error:  # noqa: BLE001 - the report names the block
            raise AssertionError(f"{where} failed: {type(error).__name__}: {error}") from error
