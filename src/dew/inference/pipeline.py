"""Load the native inference task for a saved run or published checkpoint.

A run directory holds run.json and checkpoints; sources load through
dew.interop.load_pretrained. Causal text uses TextGeneration, native MDLM
uses MaskedGeneration, DiffusionGemma uses BlockGeneration, and image
diffusion uses TextToImage. Weights are placed once on a mesh
under a layout, the way the trainer places a train state. The default mesh
uses the current pool's devices. A just-trained state needs no reload; its objective's
`pipeline(state)` binds it in place.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from etils import epath

from dew.checkpoints import RUN_FILE
from dew.inference.tasks import BlockGeneration, MaskedGeneration, TextGeneration
from dew.nn.inputs import ModelInputs, pad_token_rows
from dew.sampling.pipelines import TextToImage, restore_variables
from dew.objectives.base import Variables
from dew.sampling.text import Sampling
from dew.registry import dtype_name, resolve_dtype

if TYPE_CHECKING:
    from dew.training.distributed import Layout, MeshSpec


def pipeline(source: str, *, mesh: MeshSpec | None = None, layout: Layout | None = None,
             dtype: str | None = None, param_dtype: str | None = None,
             ema: bool = True, step: int | None = None,
             revision: str | None = None) -> TextToImage | TextGeneration | BlockGeneration | MaskedGeneration:
    """The inference task for `source`, its weights placed once.

    `source` is a run directory, or a source checkpoint directory or Hub
    repository. `mesh` places the weights on that mesh under `layout` (the
    trainer's default when None). Without `mesh`, data parallelism uses the
    current pool's devices. dtype selects computation. param_dtype selects
    parameter storage: None preserves a run's stored dtypes and uses FP32
    masters for a source. ema reads a run's averaged weights; step selects
    its checkpoint and revision pins a Hub source.
    """
    resolve_dtype(dtype)
    resolve_dtype(param_dtype)
    root = epath.Path(source)
    if (root / RUN_FILE).is_file():
        if revision is not None:
            raise ValueError("revision pins a Hub source; a run directory has checkpoints, selected by step")
        return _from_run(root, mesh=mesh, layout=layout, dtype=dtype, param_dtype=param_dtype,
                         ema=ema, step=step)
    if step is not None:
        raise ValueError("step selects a run's checkpoint; a source checkpoint has one set of weights")
    return _from_source(source, mesh=mesh, layout=layout, dtype=dtype, param_dtype=param_dtype,
                        revision=revision)


def _from_run(root: epath.Path, *, mesh: MeshSpec | None, layout: Layout | None,
              dtype: str | None, param_dtype: str | None, ema: bool,
              step: int | None) -> TextToImage | TextGeneration | BlockGeneration | MaskedGeneration:
    from dew.config import ModelConfig
    from dew.data import tokenizer_for
    from dew.registry import objectives
    import dew.objectives.lm  # noqa: F401 registers the saved objective kinds
    import dew.objectives.rl  # noqa: F401 registers the saved objective kinds
    from dew.objectives.base import thaw

    record = json.loads((root / RUN_FILE).read_text())
    if not isinstance(record, dict) or not isinstance(record.get("objective"), str):
        raise ValueError("run.json must name its objective kind")
    kind = record["objective"]
    directory = str(root)
    if kind == "diffusion":
        return TextToImage.from_run(directory, ema=ema, step=step, mesh=mesh, layout=layout,
                                    dtype=dtype, param_dtype=param_dtype)
    if kind not in ("lm", "dpo", "grpo", "ppo", "block_diffusion", "masked_diffusion"):
        raise TypeError(f"{kind!r} has no saved generation task; supported kinds are diffusion, lm, dpo, grpo, ppo, block_diffusion and masked_diffusion")
    model_config = ModelConfig.from_dict(record["model"])
    compute = dtype_name(resolve_dtype(dtype))
    if compute is not None:
        model_config = replace(model_config, dtype=compute)
    tokenizer = tokenizer_for(record["tokenizer"])
    if kind == "block_diffusion":
        from dew.diffusion.block import BlockProcess
        from dew.interop import diffusion_gemma
        model = diffusion_gemma.build(model_config.config, dtype=model_config.dtype,
                                      attention_impl=model_config.attention_impl,
                                      max_seq_len=model_config.config["max_seq_len"])
        model = model.clone(text=model.text.clone(layer_scalar="trainable"))
        variables = restore_variables(directory, ema=ema, step=step, mesh=mesh, layout=layout,
                                      param_dtype=param_dtype)
        return BlockGeneration(model, variables, BlockProcess(model.canvas_length, model.vocab_size),
                               RunProcessor(tokenizer), pad_token_id=int(record.get("pad_token_id", 0)))
    budget = record.get("sample_tokens")
    if budget is not None and (type(budget) is not int or budget < 0):
        raise ValueError("sample_tokens must be a nonnegative integer")
    if kind == "masked_diffusion":
        from dew.diffusion.discrete import MDLM
        model = model_config.build()
        mask_id = getattr(model, "mask_token_id", None)
        if getattr(model, "causal", True) or type(mask_id) is not int:
            raise ValueError("a saved masked run requires causal=False and a mask_token_id")
        variables = restore_variables(directory, ema=ema, step=step, mesh=mesh, layout=layout,
                                      param_dtype=param_dtype)
        return MaskedGeneration(model, variables, MDLM(mask_id=mask_id)(), RunProcessor(tokenizer),
                                pad_token_id=int(record.get("pad_token_id", 0)),
                                max_new_tokens=budget or None)
    objective_type = objectives[kind]
    variables = restore_variables(directory, ema=ema and not objective_type._ema_is_reference,
                                  step=step, mesh=mesh, layout=layout, param_dtype=param_dtype)
    if kind == "ppo":
        from dew.objectives.rl.ppo import _part
        variables = _part(variables, "policy")
    model = model_config.build()
    if record.get("quantization") is not None:
        from dew.training.quantization import Quantization, apply_quantization
        model = apply_quantization(model, Quantization(**record["quantization"]))
    controls = record.get("sampling")
    if controls is None and budget:
        raise ValueError("run.json lacks the sampling policy for its text previews")
    if controls is not None and not isinstance(controls, dict):
        raise ValueError("the run's sampling policy must be a Sampling record")
    sampling = Sampling() if controls is None else Sampling(**controls)
    return TextGeneration(model, thaw(variables), RunProcessor(tokenizer), sampling=sampling,
                          max_new_tokens=budget if budget else None)

def _from_source(source: str, *, mesh: MeshSpec | None, layout: Layout | None,
                 dtype: str | None, param_dtype: str | None,
                 revision: str | None) -> TextToImage | TextGeneration | BlockGeneration | MaskedGeneration:
    from dew.interop import load_pretrained
    from dew.nn.diffusion_gemma import DiffusionGemma

    storage = "float32" if param_dtype is None else param_dtype
    loaded = (load_pretrained(source, revision=revision, param_dtype=storage) if dtype is None else
              load_pretrained(source, revision=revision, dtype=dtype, param_dtype=storage))
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
