"""Load the native inference task for a saved run or published checkpoint.

A run directory holds `run.json` and checkpoints; a diffusion run becomes a
`TextToImage`, a language-model run a `TextGeneration`. A source checkpoint
(a Hub repository or a directory in its layout) loads through
`dew.interop.load_pretrained` and becomes a `TextGeneration`, or a
`BlockGeneration` for a DiffusionGemma. Diffusion sources produce an image
task. Weights are placed once on a mesh
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
from dew.nn.inputs import ModelInputs, pad_token_rows
from dew.sampling.pipelines import TextToImage, restore_variables
from dew.objectives.base import Variables
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


def _from_run(root: epath.Path, *, mesh: MeshSpec | None, layout: Layout | None,
              dtype: str | None, ema: bool, step: int | None) -> TextToImage | TextGeneration | BlockGeneration:
    from dew.config import ModelConfig
    from dew.data import tokenizer_for
    from dew.registry import objectives
    import dew.objectives.lm  # noqa: F401 registers the saved objective kinds
    import dew.objectives.rl  # noqa: F401 registers the saved objective kinds
    from dew.objectives.lm.objective import model_variables

    record = json.loads((root / RUN_FILE).read_text())
    if not isinstance(record, dict) or not isinstance(record.get("objective"), str):
        raise ValueError("run.json must name its objective kind")
    kind = record["objective"]
    directory = str(root)
    if kind == "diffusion":
        return TextToImage.from_run(directory, ema=ema, step=step, mesh=mesh, layout=layout, dtype=dtype)
    if kind not in ("lm", "dpo", "grpo", "ppo", "block_diffusion"):
        raise TypeError(f"{kind!r} has no saved generation task; supported kinds are diffusion, lm, dpo, grpo, ppo and block_diffusion")
    model_config = ModelConfig.from_dict(record["model"])
    tokenizer = tokenizer_for(record["tokenizer"])
    if kind == "block_diffusion":
        from dew.diffusion.block import BlockProcess
        from dew.interop import diffusion_gemma
        model = diffusion_gemma.build(model_config.config, dtype=model_config.dtype,
                                      attention_impl=model_config.attention_impl,
                                      max_seq_len=model_config.config["max_seq_len"])
        model = model.clone(text=model.text.clone(layer_scalar="trainable"))
        variables = restore_variables(directory, ema=ema, step=step, mesh=mesh, layout=layout, dtype=dtype)
        return BlockGeneration(model, variables, BlockProcess(model.canvas_length, model.vocab_size),
                               RunProcessor(tokenizer), pad_token_id=int(record.get("pad_token_id", 0)))
    objective_type = objectives[kind]
    variables = restore_variables(directory, ema=ema and not objective_type._ema_is_reference,
                                  step=step, mesh=mesh, layout=layout, dtype=dtype)
    if kind == "ppo":
        from dew.objectives.rl.ppo import _part
        variables = _part(variables, "policy")
    model = model_config.build()
    if record.get("quantization") is not None:
        from dew.training.quantization import Quantization, apply_quantization
        model = apply_quantization(model, Quantization(**record["quantization"]))
    budget = record.get("sample_tokens")
    if budget is not None and (type(budget) is not int or budget < 0):
        raise ValueError("sample_tokens must be a nonnegative integer")
    controls = record.get("sampling")
    if controls is None and budget:
        raise ValueError("run.json lacks the sampling policy for its text previews")
    if controls is not None and not isinstance(controls, dict):
        raise ValueError("the run's sampling policy must be a Sampling record")
    sampling = Sampling() if controls is None else Sampling(**controls)
    return TextGeneration(model, model_variables(variables), RunProcessor(tokenizer), sampling=sampling,
                          max_new_tokens=budget if budget else None)

def _from_source(source: str, *, mesh: MeshSpec | None, layout: Layout | None,
                 dtype: str | None, revision: str | None) -> TextToImage | TextGeneration | BlockGeneration:
    from dew.interop import load_pretrained
    from dew.nn.diffusion_gemma import DiffusionGemma

    loaded = (load_pretrained(source, revision=revision) if dtype is None else
              load_pretrained(source, revision=revision, dtype=dtype))
    if loaded.process is not None:
        task = loaded.text_to_image()
    elif isinstance(loaded.model, DiffusionGemma):
        task = loaded.block_generation()
    else:
        task = loaded.text_generation()
    return task.bind(place(loaded.variables, mesh, layout))


def place(variables: Variables, mesh: MeshSpec | None, layout: Layout | None) -> Variables:
    """`variables` on the mesh `mesh` describes, sharded the way the trainer
    shards a train state's parameters under `layout`."""
    from dew.training.distributed import Layout as DefaultLayout, MeshSpec as DefaultMesh, build_mesh

    device_mesh = build_mesh(DefaultMesh() if mesh is None else mesh)
    chosen_layout = DefaultLayout() if layout is None else layout
    shardings = chosen_layout.shardings(device_mesh, variables)
    chosen_layout.check(variables, shardings, device_mesh)
    return jax.device_put(variables, shardings)




class RunTokenizer(Protocol):
    """What a run's tokenizer offers: `dew.data.ByteTokenizer` and `HFTokenizer` do."""


    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: jax.typing.ArrayLike | Sequence[int]) -> str: ...


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
        tokens, fields = pad_token_rows(ids)
        return ModelInputs(jnp.asarray(tokens), jax.tree.map(jnp.asarray, fields))

    def decode(self, tokens: jax.typing.ArrayLike) -> list[str]:
        return [self.tokenizer.decode(row) for row in np.asarray(tokens)]
