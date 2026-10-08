"""Load the native inference task for a saved run or published checkpoint.

A run directory holds run.json and checkpoints; sources load through
`dew.interop.Pretrained.load`. Causal text uses TextGeneration, native MDLM
uses MaskedGeneration, DiffusionGemma uses BlockGeneration, and image
diffusion uses TextToImage. Weights are placed once on a mesh
under a layout, the way the trainer places a train state. The default mesh
uses the current pool's devices. A just-trained state needs no reload; its objective's
`pipeline(state)` binds it in place.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
from etils import epath
from jax.typing import DTypeLike

from dew.cache import persist_compilations
from dew.checkpoints import RUN_FILE
from dew.data.text import Tokenizer
from dew.inference.tasks import BlockGeneration, MaskedGeneration, TextGeneration
from dew.nn.inputs import Media, ModelInputs, pad_token_rows
from dew.objectives.base import SavedTask, Variables
from dew.registry import dtype_name
from dew.sampling.pipelines import TextToImage

if TYPE_CHECKING:
    from dew.training.distributed import Layout, MeshSpec


def pipeline(
    source: str,
    *,
    mesh: MeshSpec | None = None,
    layout: Layout | None = None,
    dtype: DTypeLike | None = None,
    param_dtype: DTypeLike | Literal["auto"] | None = None,
    ema: bool | None = None,
    step: int | str | None = None,
    revision: str | None = None,
    trust: Sequence[str] = (),
) -> TextToImage | TextGeneration | BlockGeneration | MaskedGeneration | SavedTask:
    """Load the inference task for `source`, with its weights placed once.

    `source` is a run directory, a source checkpoint directory or a Hub
    repository holding either. A run loads as the task its recorded objective declares
    (`Objective.saved_task`), including a plugin objective's own task.
    `mesh` places the weights on that mesh under `layout`, or under the
    trainer's default layout when `layout` is None. Without `mesh`, data
    parallelism uses the current pool's devices.

    `dtype` sets the computation dtype, given as a dtype (`jnp.bfloat16`) or
    its name. `param_dtype` sets parameter storage. None keeps a run's
    stored dtypes and uses FP32 masters for a source, and `'auto'` keeps the
    stored dtypes either way; for a source, that is its config dtype or its
    first floating tensor, as Transformers' `dtype='auto'` reads it. `ema`
    selects a run's averaged weights: None reads them when the run kept them
    and its live weights otherwise, and True always reads them. `step`
    selects a run's checkpoint and `revision` pins a Hub source; passing
    `revision` for a run directory or `step` for a source raises
    `ValueError`. `trust` names the packages outside Dew a run's record may
    import, as `trust_remote_code` does in transformers.

    Loading a task also points XLA at the on-disk executable cache, unless a
    cache directory is already set, so a restarted process reuses what it
    already compiled.
    """
    persist_compilations()
    dtype = dtype_name(dtype)
    param_dtype = "auto" if param_dtype == "auto" else dtype_name(param_dtype)
    root = epath.Path(source)
    if not root.is_dir():
        # A run published whole (`HfApi().upload_folder`) is pulled at the commit
        # its metadata resolved, and loads as the run.
        from dew.interop import sources
        from dew.interop.hub import pull_from_hub
        metadata = sources.snapshot(source, revision, weights=False)
        if (metadata / RUN_FILE).is_file():
            root, revision = epath.Path(pull_from_hub(source, metadata.name)), None
    if root.is_dir() and (
            (root / RUN_FILE).is_file() or any(path.name.isdecimal() for path in root.iterdir())):
        if revision is not None:
            raise ValueError("revision pins a Hub source; a run directory has checkpoints, selected by step")
        return _from_run(root, mesh=mesh, layout=layout, dtype=dtype,
                         param_dtype=None if param_dtype == "auto" else param_dtype, ema=ema, step=step,
                         trust=trust)
    if step is not None:
        raise ValueError("step selects a run's checkpoint; a source checkpoint has one set of weights")
    return _from_source(source, mesh=mesh, layout=layout, dtype=dtype, param_dtype=param_dtype,
                        revision=revision)


def _from_run(root: epath.Path, *, mesh: MeshSpec | None, layout: Layout | None,
              dtype: str | None, param_dtype: str | None, ema: bool | None,
              step: int | str | None, trust: Sequence[str]) -> SavedTask:
    """The task the run's recorded objective declares (`Objective.saved_task`)."""
    from dew.inference.tasks import run_record
    from dew.records import text
    from dew.registry import objectives

    kind = text(run_record(str(root), step, trust)['objective'], 'objective')
    task = objectives[kind].saved_task
    if task is None:
        raise TypeError(f"a run of the {kind!r} objective ({objectives[kind].__name__}) loads as no task: "
                        "its class declares no `saved_task`")
    return task.from_run(str(root), ema=ema, step=step, mesh=mesh, layout=layout,
                         dtype=dtype, param_dtype=param_dtype)


def _from_source(source: str, *, mesh: MeshSpec | None, layout: Layout | None,
                 dtype: str | None, param_dtype: str | None,
                 revision: str | None) -> TextToImage | TextGeneration | BlockGeneration | MaskedGeneration:
    from dew.inference.projections import _inference_projections
    from dew.interop import (
        Pretrained,
        PretrainedBlockDecoder,
        PretrainedDecoder,
        PretrainedMaskedDecoder,
        PretrainedPipeline,
    )
    from dew.training.distributed import MeshSpec as DefaultMesh

    storage = "float32" if param_dtype is None else param_dtype
    placement = DefaultMesh() if mesh is None else mesh
    def prepared(model, variables):
        with jax.set_mesh(placement.build()):
            return _inference_projections(model, variables)
    loaded = (Pretrained._load(source, revision=revision, param_dtype=storage, mesh=placement, layout=layout,
                              prepare=prepared)
              if dtype is None else
              Pretrained._load(source, revision=revision, dtype=dtype, param_dtype=storage, mesh=placement,
                               layout=layout, prepare=prepared))
    match loaded:
        case PretrainedPipeline():
            return loaded.text_to_image()
        case PretrainedBlockDecoder():
            return loaded.block_generation()
        case PretrainedDecoder() | PretrainedMaskedDecoder():
            return loaded.text_generation()
    raise TypeError(f"{source} loaded as {type(loaded).__name__}, which has no generation task")


def place(variables: Variables, mesh: MeshSpec | None, layout: Layout | None) -> Variables:
    """Place `variables` on the mesh `mesh` describes, one leaf at a time.

    The sharding is the one the trainer gives a train state's parameters under
    `layout`; `dew.training.host.stream` places the leaves, updating the
    dict nodes of `variables` in place. A task holds its variables frozen, and
    a frozen node can't be updated, so frozen nodes are first rebuilt as
    dicts over the same leaves.
    """
    from dew.training.distributed import Layout as DefaultLayout, MeshSpec as DefaultMesh
    from dew.training.host import stream

    variables = _updatable(variables)
    device_mesh = (DefaultMesh() if mesh is None else mesh).build()
    chosen_layout = DefaultLayout() if layout is None else layout
    shardings = chosen_layout.shardings(device_mesh, variables)
    chosen_layout.check(variables, shardings, device_mesh)
    return stream(variables, shardings)


def _updatable(tree: Variables) -> Variables:
    """`tree` with each node that is not a plain dict (a FrozenDict) rebuilt
    as one over the same children; its dict nodes stay the same objects, so
    `stream` still updates them in place."""
    rebuilt: dict[str, object] = tree if type(tree) is dict else dict(tree)
    for name, child in rebuilt.items():
        if isinstance(child, Mapping):
            rebuilt[name] = _updatable(child)
    return rebuilt


@dataclass(frozen=True)
class RunProcessor:
    """Adapts a run's tokenizer to the `Processor` interface a task uses.

    Calling it encodes each text prompt and left-pads the rows into
    `ModelInputs`; a text run takes no images or audio. `decode` returns one
    string per token row.
    """

    tokenizer: Tokenizer

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
