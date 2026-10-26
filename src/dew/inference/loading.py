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

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
from etils import epath
from jax.typing import DTypeLike

from dew.batch import Variables
from dew.cache import persist_compilations
from dew.checkpoints import PYTHON_RUN, RUN_FILE
from dew.data.text import Tokenizer
from dew.inference.images import TextToImage
from dew.inference.tasks import BlockGeneration, MaskedGeneration, TextGeneration
from dew.nn.inputs import Media, ModelInputs, pad_token_rows
from dew.objectives.base import SavedTask
from dew.registry import dtype_name

if TYPE_CHECKING:
    from jax.sharding import Mesh

    from dew.distributed.placement import Layout, MeshSpec


def pipeline(
    source: str,
    *,
    mesh: MeshSpec | Mesh | None = None,
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
    its name. `param_dtype` sets parameter storage. None and `'auto'` keep
    each tensor's stored dtype, a run's or a source's (a quantized source's
    packed weights decode to the dtype it declares); `jnp.float32` gives a
    source FP32 masters. `ema`
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
        # A Hub source is read at the commit its metadata resolved; one holding
        # a run (`HfApi().upload_folder`) is pulled whole and loads as the run.
        from huggingface_hub import snapshot_download

        from dew.interop import sources
        metadata = sources.snapshot(source, revision, weights=False)
        if (metadata / RUN_FILE).is_file():
            root, revision = epath.Path(snapshot_download(repo_id=source, revision=metadata.name)), None
        else:
            revision = metadata.name
    if root.is_dir() and not (root / RUN_FILE).is_file() and any(path.name.isdecimal()
                                                                for path in root.iterdir()):
        raise ValueError(f"{source} holds checkpoint steps but no {RUN_FILE}, the record at the root of "
                         f"every run; a run saved without one gains it at its next save, or write "
                         f"{json.dumps(PYTHON_RUN)} there for a run defined in Python")
    if root.is_dir() and (root / RUN_FILE).is_file():
        if revision is not None:
            raise ValueError("revision pins a Hub source; a run directory has checkpoints, selected by step")
        return _from_run(root, mesh=mesh, layout=layout, dtype=dtype,
                         param_dtype=None if param_dtype == "auto" else param_dtype, ema=ema, step=step,
                         trust=trust)
    if step is not None:
        raise ValueError("step selects a run's checkpoint; a source checkpoint has one set of weights")
    return _from_source(source, mesh=mesh, layout=layout, dtype=dtype, param_dtype=param_dtype,
                        revision=revision)


def _from_run(root: epath.Path, *, mesh: MeshSpec | Mesh | None, layout: Layout | None,
              dtype: str | None, param_dtype: str | None, ema: bool | None,
              step: int | str | None, trust: Sequence[str]) -> SavedTask:
    """The task the run's recorded objective declares (`Objective.saved_task`), loaded at the step
    pinned here; the packages its record may import stay trusted for the process once pinned
    (`dew.registry.import_trusted`), so the task's own load reads it too."""
    from dew.inference.tasks import SavedRun

    run = SavedRun.pinned(str(root), step, trust)
    task = run.objective.saved_task
    if task is None:
        raise TypeError(f"a run of {run.objective.__name__} loads as no task: its class declares no "
                        "`saved_task`")
    return task.from_run(run.directory, ema=ema, step=run.step, mesh=mesh, layout=layout,
                         dtype=dtype, param_dtype=param_dtype)


def _from_source(source: str, *, mesh: MeshSpec | Mesh | None, layout: Layout | None,
                 dtype: str | None, param_dtype: str | None,
                 revision: str | None) -> SavedTask:
    from dew.distributed.placement import MeshSpec as DefaultMesh
    from dew.inference.projections import packed_owned
    from dew.interop import Pretrained, PretrainedPipeline

    storage = "auto" if param_dtype is None else param_dtype
    placement = DefaultMesh() if mesh is None else mesh
    loaded = (Pretrained.load(source, revision=revision, param_dtype=storage, mesh=placement, layout=layout)
              if dtype is None else
              Pretrained.load(source, revision=revision, dtype=dtype, param_dtype=storage, mesh=placement,
                              layout=layout))
    if not isinstance(loaded, PretrainedPipeline):
        # The task owns the weights it loaded: their projections pack once, where they were placed.
        # (Public bundle.pipeline borrows weights; packing here may relinquish the unpacked device
        # buffers without invalidating a caller's tree.)
        loaded = replace(loaded, variables=packed_owned(loaded.model, loaded.variables))
    return loaded.pipeline()


def place(variables: Variables, mesh: MeshSpec | Mesh | None, layout: Layout | None) -> Variables:
    """Place `variables` on `mesh`, a native mesh or the one a `MeshSpec`
    describes, one leaf at a time.

    The sharding is the one the trainer gives a train state's parameters under
    `layout`; `dew.distributed.host.stream` places the leaves, updating the
    dict nodes of `variables` in place and rebuilding a frozen one, which a
    task holds, as a dict over the same leaves.
    """
    from dew.distributed.host import stream
    from dew.distributed.placement import Layout as DefaultLayout, MeshSpec as DefaultMesh, device_mesh

    placed_on = device_mesh(DefaultMesh() if mesh is None else mesh)
    chosen_layout = DefaultLayout() if layout is None else layout
    shardings = chosen_layout.shardings(placed_on, variables)
    chosen_layout.check(variables, shardings, placed_on)
    return stream(variables, shardings)


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

    def encode(self, text: str, *, specials: bool) -> list[int]:
        """`text`'s ids from the run's tokenizer, with the special tokens it adds around a sequence
        where `specials`, and none otherwise."""
        return self.tokenizer.encode(text, add_special_tokens=specials)

    def spelling(self, ids: Sequence[int]) -> str:
        """The text the run's tokenizer decodes `ids` to, its special tokens spelled out."""
        return self.tokenizer.decode(list(ids))

    @property
    def bos_id(self) -> int | None:
        """The id the run's tokenizer starts a sequence with, or None."""
        return self.tokenizer.bos_id
