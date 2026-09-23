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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from etils import epath

from dew.checkpoints import RUN_FILE
from dew.inference.tasks import BlockGeneration, MaskedGeneration, TextGeneration
from dew.nn.inputs import Media, ModelInputs, pad_token_rows
from dew.objectives.base import Variables
from dew.registry import resolve_dtype
from dew.sampling.pipelines import TextToImage
from dew.telemetry.instrumentation import default_compilation_cache_dir, enable_compilation_cache

if TYPE_CHECKING:
    from dew.training.distributed import Layout, MeshSpec


def pipeline(source: str, *, mesh: MeshSpec | None = None, layout: Layout | None = None,
             dtype: str | None = None, param_dtype: str | None = None,
             ema: bool = True, step: int | None = None,
             revision: str | None = None) -> TextToImage | TextGeneration | BlockGeneration | MaskedGeneration:
    """Load the inference task for `source`, its weights placed once.

    `source` is a run directory, or a source checkpoint directory or Hub
    repository. `mesh` places the weights on that mesh under `layout` (the
    trainer's default when None). Without `mesh`, data parallelism uses the
    current pool's devices. dtype selects computation. param_dtype selects
    parameter storage: None preserves a run's stored dtypes and uses FP32
    masters for a source, and 'auto' stores the stored dtypes either way (a
    source's config dtype or first floating tensor, as transformers'
    dtype='auto' reads it). ema reads a run's averaged weights; step selects
    its checkpoint and revision pins a Hub source.

    Loading a task also points XLA at the on-disk executable cache, so a
    restarted process reuses what it already compiled.
    """
    _persist_compilations()
    resolve_dtype(dtype)
    if param_dtype != "auto":
        resolve_dtype(param_dtype)
    root = epath.Path(source)
    if (root / RUN_FILE).is_file():
        if revision is not None:
            raise ValueError("revision pins a Hub source; a run directory has checkpoints, selected by step")
        return _from_run(root, mesh=mesh, layout=layout, dtype=dtype,
                         param_dtype=None if param_dtype == "auto" else param_dtype, ema=ema, step=step)
    if step is not None:
        raise ValueError("step selects a run's checkpoint; a source checkpoint has one set of weights")
    return _from_source(source, mesh=mesh, layout=layout, dtype=dtype, param_dtype=param_dtype,
                        revision=revision)


def _persist_compilations() -> None:
    """Point XLA at the on-disk executable cache, unless a directory is set.

    A served request compiles for several seconds the first time its shapes
    are seen, and a serving process restarts. Training turns the same cache
    on in `prepare_process`; inference has no such entry point, and
    `pipeline` is the one place every task is built, so it goes here.
    Reading the setting is what makes it idempotent and what leaves a
    trainer's own directory, or a caller's, alone.
    """
    if jax.config.jax_compilation_cache_dir:
        return
    enable_compilation_cache(default_compilation_cache_dir())


SAVED_TASKS: Mapping[str, type[TextToImage] | type[TextGeneration] | type[BlockGeneration]
                     | type[MaskedGeneration]] = {
    "diffusion": TextToImage, "lm": TextGeneration, "dpo": TextGeneration,
    "grpo": TextGeneration, "ppo": TextGeneration, "block_diffusion": BlockGeneration,
    "masked_diffusion": MaskedGeneration}
"""Which task each saved objective kind generates through.

One entry per kind a run publishes weights for; each task's own `from_run`
holds the construction, so this is the whole of what the front door knows
about a run beyond the name its `run.json` records.
"""


def _from_run(root: epath.Path, *, mesh: MeshSpec | None, layout: Layout | None,
              dtype: str | None, param_dtype: str | None, ema: bool,
              step: int | None) -> TextToImage | TextGeneration | BlockGeneration | MaskedGeneration:
    record = json.loads((root / RUN_FILE).read_text())
    if not isinstance(record, dict) or not isinstance(record.get("objective"), str):
        raise ValueError("run.json must name its objective kind")
    kind = record["objective"]
    task = SAVED_TASKS.get(kind)
    if task is None:
        supported = ", ".join(list(SAVED_TASKS)[:-1]) + f" and {list(SAVED_TASKS)[-1]}"
        raise TypeError(f"{kind!r} has no saved generation task; supported kinds are {supported}")
    return task.from_run(str(root), ema=ema, step=step, mesh=mesh, layout=layout,
                         dtype=dtype, param_dtype=param_dtype)


def _from_source(source: str, *, mesh: MeshSpec | None, layout: Layout | None,
                 dtype: str | None, param_dtype: str | None,
                 revision: str | None) -> TextToImage | TextGeneration | BlockGeneration | MaskedGeneration:
    from dew.interop import load_pretrained
    from dew.nn.diffusion_gemma import DiffusionGemma
    from dew.training.distributed import MeshSpec as DefaultMesh

    storage = "float32" if param_dtype is None else param_dtype
    placement = DefaultMesh() if mesh is None else mesh
    loaded = (load_pretrained(source, revision=revision, param_dtype=storage, mesh=placement, layout=layout)
              if dtype is None else
              load_pretrained(source, revision=revision, dtype=dtype, param_dtype=storage, mesh=placement,
                              layout=layout))
    if loaded.process is not None:
        return loaded.text_to_image()
    if isinstance(loaded.model, DiffusionGemma):
        return loaded.block_generation()
    return loaded.text_generation()


def place(variables: Variables, mesh: MeshSpec | None, layout: Layout | None) -> Variables:
    """Place `variables` on the mesh `mesh` describes, one leaf at a time.

    The sharding is the one the trainer gives a train state's parameters under
    `layout`. Each leaf lands on its sharding before the next is read
    (`dew.training.host.place_leaf`); a `SourceLeaf` is read from the
    mapped checkpoint one device shard at a time, so no host copy of the
    whole tree is made.
    """
    from dew.training.distributed import Layout as DefaultLayout, MeshSpec as DefaultMesh, build_mesh
    from dew.training.host import place_leaf

    device_mesh = build_mesh(DefaultMesh() if mesh is None else mesh)
    chosen_layout = DefaultLayout() if layout is None else layout
    shardings = chosen_layout.shardings(device_mesh, variables)
    chosen_layout.check(variables, shardings, device_mesh)
    return jax.tree.map(place_leaf, variables, shardings)


class RunTokenizer(Protocol):
    """Declares what a run's tokenizer offers, as `ByteTokenizer` and `HFTokenizer` do."""

    @property
    def bos_id(self) -> int | None: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: jax.typing.ArrayLike | Sequence[int]) -> str: ...


@dataclass(frozen=True)
class RunProcessor:
    """Adapts a run's tokenizer to a task's host processor.

    Left-padded prompt rows go in and one string per row comes out.
    """

    tokenizer: RunTokenizer

    def __call__(self, text: str | Sequence[str], *, images: Media | None = None,
                 audio: Media | None = None) -> ModelInputs:
        if images is not None or audio is not None:
            raise ValueError("a text run takes no images or audio")
        rows = [text] if isinstance(text, str) else list(text)
        ids = [self.tokenizer.encode(row) for row in rows]
        tokens, fields = pad_token_rows(ids)
        return ModelInputs(jnp.asarray(tokens), jax.tree.map(jnp.asarray, fields))

    def decode(self, tokens: jax.typing.ArrayLike) -> list[str]:
        return [self.tokenizer.decode(row) for row in np.asarray(tokens)]

    @property
    def bos_id(self) -> int | None:
        """The id the run's tokenizer starts a sequence with, or None."""
        return self.tokenizer.bos_id
