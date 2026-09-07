"""One front door: a source on disk or the Hub as the inference task for its kind.

A run directory holds `run.json` and checkpoints; a diffusion run becomes a
`TextToImage`, a language-model run a `TextGeneration`. A source checkpoint
(a Hub repository or a directory in its layout) loads through
`dew.interop.load_pretrained` and becomes a `TextGeneration`, or a
`BlockGeneration` for a DiffusionGemma. Weights are placed once: on a mesh
under a layout, the way the trainer places a train state. The default mesh
uses the current pool's devices. A just-trained state needs no reload; its objective's
`pipeline(state)` binds it in place.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from etils import epath

from dew.checkpoints import RUN_FILE
from dew.inference.tasks import BlockGeneration, TextGeneration
from dew.nn.inputs import ModelInputs
from dew.sampling.pipelines import TextToImage, cast_floating, restore_variables
from dew.sampling.text import Sampling

if TYPE_CHECKING:
    from dew.training.distributed import Layout, MeshSpec


def pipeline(source: str, *, mesh: MeshSpec | None = None, layout: Layout | None = None,
             dtype: str | None = None, ema: bool = True, step: int | None = None,
             revision: str | None = None) -> TextToImage | TextGeneration | BlockGeneration:
    """The inference task for `source`, its weights placed once.

    `source` is a run directory, or a source checkpoint directory or Hub
    repository. `mesh` places the weights on that mesh under `layout` (the
    trainer's default when None). Without `mesh`, data parallelism uses the
    current pool's devices. `dtype` casts a run's floating weights, or is the source loader's
    compute dtype. `ema` reads a run's averaged weights when it kept them;
    `step` selects a run's checkpoint; `revision` pins a Hub source.
    """
    root = epath.Path(source)
    if (root / RUN_FILE).is_file():
        if revision is not None:
            raise ValueError("revision pins a Hub source; a run directory has checkpoints, selected by step")
        return _from_run(root, mesh=mesh, layout=layout, dtype=dtype, ema=ema, step=step)
    if step is not None:
        raise ValueError("step selects a run's checkpoint; a source checkpoint has one set of weights")
    return _from_source(source, mesh=mesh, layout=layout, dtype=dtype, revision=revision)


def _from_run(root: epath.Path, *, mesh, layout, dtype, ema, step):
    record = json.loads((root / RUN_FILE).read_text())
    directory = str(root)
    if "objective" not in record:
        return TextToImage.from_run(directory, ema=ema, step=step, mesh=mesh, layout=layout, dtype=dtype)
    kind = record["objective"]
    if kind != "lm":
        raise TypeError(f"a {kind} run has no standalone inference task; train it and use Objective.pipeline")
    from dew.config import ModelConfig
    from dew.objectives.lm.objective import model_variables

    model = ModelConfig.from_dict(record["model"]).build()
    variables = restore_variables(directory, ema=ema, step=step, mesh=mesh, layout=layout, dtype=dtype)
    tokenizer = _run_tokenizer(record.get("tokenizer", "byte"))
    budget = record.get("sample_tokens")
    return TextGeneration(model, model_variables(variables), RunProcessor(tokenizer),
                          sampling=Sampling(eos_id=tokenizer.eos_id),
                          max_new_tokens=budget if isinstance(budget, int) and budget > 0 else None)


def _from_source(source: str, *, mesh, layout, dtype, revision):
    from dew.interop import load_pretrained
    from dew.nn.diffusion_gemma import DiffusionGemma

    options = {} if dtype is None else {"dtype": dtype}
    loaded = load_pretrained(source, revision=revision, **options)
    task = loaded.block_generation() if isinstance(loaded.model, DiffusionGemma) else loaded.text_generation()
    return task.bind(place(loaded.variables, mesh, layout))


def place(variables, mesh: MeshSpec | None, layout: Layout | None):
    """`variables` on the mesh `mesh` describes, sharded the way the trainer
    shards a train state's parameters under `layout`."""
    from dew.training.distributed import Layout as DefaultLayout, MeshSpec as DefaultMesh, build_mesh

    device_mesh = build_mesh(DefaultMesh() if mesh is None else mesh)
    chosen_layout = DefaultLayout() if layout is None else layout
    shardings = chosen_layout.shardings(device_mesh, variables)
    chosen_layout.check(variables, shardings, device_mesh)
    return jax.device_put(variables, shardings)


def _run_tokenizer(name: str):
    from dew.data.text import ByteTokenizer, HFTokenizer

    return ByteTokenizer() if name == "byte" else HFTokenizer(name)


class RunTokenizer(Protocol):
    """What a run's tokenizer offers: `dew.data.ByteTokenizer` and `HFTokenizer` do."""

    @property
    def eos_id(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids) -> str: ...


@dataclass(frozen=True)
class RunProcessor:
    """A run's tokenizer as a task's host processor: left-padded prompt rows in, one string per row out."""

    tokenizer: RunTokenizer

    def __call__(self, text: str | Sequence[str], *, images: object | None = None,
                 audio: object | None = None) -> ModelInputs:
        if images is not None or audio is not None:
            raise ValueError("a text run takes no images or audio")
        rows = [text] if isinstance(text, str) else list(text)
        ids = [self.tokenizer.encode(row) for row in rows]
        if any(not row for row in ids):
            raise ValueError("every prompt must tokenize to at least one id")
        width = max(len(row) for row in ids)
        tokens = np.zeros((len(ids), width), np.int32)
        mask = np.zeros((len(ids), width), bool)
        for index, row in enumerate(ids):
            tokens[index, width - len(row):] = row
            mask[index, width - len(row):] = True
        return ModelInputs(jnp.asarray(tokens), {"attention_mask": jnp.asarray(mask)})

    def decode(self, tokens) -> list[str]:
        return [self.tokenizer.decode(row) for row in np.asarray(tokens)]
