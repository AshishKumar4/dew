"""A saved Dew run as the source another run starts from.

A source names a saved run by its directory, and `pinned_run` resolves it, for every loader, to the
run's checkpoint at the step its selection pins. `dew export` writes that checkpoint as a release
(`export_run`): a task that publishes its own released layout writes it (a decision run's,
`ReleasedTask`), any other run its model in the source layout its family reads
(`Pretrained.from_saved`). A language model or decision run that starts from a saved run reads that
release (`released_run`): every loader takes a source by its directory, so the run's load options
(its compute and storage dtypes, its mesh, its source options, an adapter) apply to it as to any
release, and its first weights are the export's. Dew writes no release of a diffusion run
(`samples_images`), and a diffusion run starts from the parts and policy of the run's own sampling
task instead (`pipeline_of_run`).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

import jax
from etils import epath
from jax.sharding import Mesh

from dew._coordination import agreed_same, from_every_process, reading_home, seen_by_every_process
from dew.cache import dew_cache_dir, package_digest
from dew.checkpoints import RUN_FILE
from dew.files import staged

if TYPE_CHECKING:
    from flax import linen as nn

    from dew.batch import Variables
    from dew.diffusion.process import Process
    from dew.diffusion.schedules.source import SourceGrid
    from dew.distributed.placement import Layout, MeshSpec
    from dew.inference.tasks import SavedRun
    from dew.inputs import InputSpec
    from dew.lora import LoRA
    from dew.nn.attention import AttentionImpl
    from dew.nn.autoencoders import AutoEncoder
    from dew.sampling.guidance import Guidance
    from dew.sampling.solvers.common import Solver


def export_run(run: SavedRun, destination: str | Path, *, ema: bool | None = None,
               mesh: MeshSpec | Mesh | None = None, layout: Layout | None = None) -> None:
    """Write `run`'s pinned checkpoint (`SavedRun.pinned`) to `destination` as its release, its EMA
    weights where `ema`: a task's own released layout (`ReleasedTask`), or the model in its family's
    source layout (`Pretrained.from_saved`). Its weights restore where `mesh` and `layout` place them,
    so no device holds more of them than a load does, and the writer reads them home a tensor at a time
    as it writes them, one shard of the release at a time (`save_sharded`). On a pool every process
    runs this same writer, reading each tensor that spans the pool's processes home together
    (`reading_home`), so each host holds a shard of the release, not the model (ledger 2874)."""
    from dew.interop.pretrained import Pretrained
    from dew.objectives.base import ReleasedTask

    task = run.objective.saved_task
    if task is not None and issubclass(task, ReleasedTask):
        released = task.from_saved(run, ema=ema, mesh=mesh, layout=layout)
        with reading_home("saved run export"):
            released.save_pretrained(str(destination))
    else:
        bundle = Pretrained.from_saved(run, ema=ema, mesh=mesh, layout=layout)
        with reading_home("saved run export"):
            bundle.save(destination)


def is_saved_run(name_or_dir: str | Path) -> bool:
    """Whether `name_or_dir` is a saved Dew run's directory, on this host or remote storage
    (`etils.epath`): one holding its run record (`RUN_FILE`)."""
    path = epath.Path(name_or_dir)
    return path.is_dir() and (path / RUN_FILE).is_file()


def _fingerprint(run: SavedRun, ema: bool | None) -> str:
    """What an export of `run`'s pinned step with `ema` is a function of: the run's location, its step,
    the EMA choice, the code that writes it (`package_digest`, ledger 2775), the contents of the run's
    own records beside its checkpoints (its run record and a task's assets, a decision task's
    calibration among them), and the step's metadata files, by size and write time, which a new save of
    the step rewrites."""
    root = epath.Path(run.directory)
    digest = hashlib.sha256(json.dumps([str(root), run.step, ema, package_digest()]).encode())
    for path in sorted((path for path in root.iterdir() if path.is_file()), key=str):
        digest.update(f"{path.name}\0".encode() + path.read_bytes())
    for path in sorted((root / str(run.step)).iterdir(), key=str):
        if path.is_file():
            stat = path.stat()
            digest.update(f"{path.name}\0{stat.length}\0{stat.mtime}\0".encode())
    return digest.hexdigest()[:24]


def pinned_run(name_or_dir: str | Path, *, step: int | str | None = None, ema: bool | None = None,
               trust: Sequence[str] = ()) -> SavedRun | None:
    """The saved Dew run `name_or_dir` names (`is_saved_run`) at its checkpoint `step` (`SavedRun.pinned`:
    a step, `latest`, `best`, `best:<metric>`, or the one its task loads by default), on which every
    process of a pool agrees; None for a source that is no saved run, beside which a step or an EMA
    choice is refused: it selects nothing. Every loader that takes a saved run resolves it here."""
    from dew.inference.tasks import SavedRun

    if not is_saved_run(name_or_dir):
        if step is not None or ema is not None:
            raise ValueError(f"{name_or_dir} is no saved Dew run (no {RUN_FILE}), so a step or an EMA choice "
                             "selects nothing of it; name a run's directory, or drop them")
        return None
    return SavedRun.pinned(str(name_or_dir), agreed_same("saved run step",
                                                       lambda: SavedRun.pinned(str(name_or_dir), step,
                                                                               trust).step), trust)


def samples_images(run: SavedRun) -> bool:
    """Whether `run` is a diffusion run, its task a `TextToImage`: Dew writes no release of one, and a run
    starts from its task's parts (`pipeline_of_run`)."""
    from dew.inference.images import TextToImage

    task = run.objective.saved_task
    return task is not None and issubclass(task, TextToImage)


@dataclass(frozen=True)
class RunPipeline:
    """A saved diffusion run as the source a diffusion run starts from (`pipeline_of_run`): its sampling
    task's denoiser `model`, `variables`, `process`, `inputs` and `autoencoder`, and the `solver`,
    `steps`, `guidance` and `grid` it samples by, which a `DiffusionObjective` trains as it trains a
    published pipeline (`PipelineSource`). Its variables are held as given, so an adapter's split of
    them stays split for training, where a task reads a split tree whole."""

    model: nn.Module
    variables: Variables
    process: Process
    inputs: InputSpec
    autoencoder: AutoEncoder | None
    solver: Solver
    steps: int
    guidance: Guidance | None
    grid: SourceGrid | None

    def adapt(self, lora: LoRA, *, key: int | jax.Array) -> RunPipeline:
        """This source with `lora` attached to its denoiser and its factors drawn (`LoRA.apply`), as
        `Pretrained.adapt` adapts a pipeline: the factors under `params` and the denoiser's weights under
        `frozen`, so an objective over it trains the factors alone, and since B starts at zero it computes
        what this source does."""
        adapter = lora.apply(self.model, self.variables, key=key)
        return replace(self, model=adapter.model, variables=adapter.variables)


def pipeline_of_run(run: SavedRun, checkpoint: str, *, dtype: str, param_dtype: str | None,
                    revision: str | None, attention_impl: AttentionImpl, size: tuple[int, ...] | None,
                    variables: Variables | None, mesh: MeshSpec | Mesh | None, layout: Layout | None,
                    text: bool, ema: bool | None) -> RunPipeline:
    """The diffusion run `run` (`checkpoint`) as a diffusion run starts from it: its own task's parts and
    policy (`TextToImage.from_saved`), computing in `dtype`, over its tree as its checkpoint holds it
    (`restored_tree`: an adapter's split of it kept, so its base stays frozen), stored in `param_dtype` (as
    saved where None) where `mesh` and `layout` place it, or over supplied `variables`, the starting run's
    own tree, held as given; its denoiser attending through `attention_impl` where its class takes a
    kernel.

    A run of another task, a revision of a directory, and `text=False`, which the restore of a run's
    tree cannot skip, are refused, as is a kernel other than `auto` for a denoiser that takes none. So
    are a run that tuned its autoencoder (`end_to_end`), whose tuned autoencoder a run started from it
    records nothing of (ledger 2791), and a `size` other than the run's own, whose conditioning, grid
    and process are bound to its size (ledger 2792)."""
    from dew._records import record
    from dew.diffusion.schedules.source import SourceGrid
    from dew.inference.images import ImageSettings, TextToImage, restored_tree
    from dew.inference.tasks import saved_settings
    from dew.objectives.diffusion.objective import without_loss_heads
    from dew.registry import from_record

    if not samples_images(run):
        raise ValueError(f"{checkpoint} is a {run.objective.__name__} run, which Pretrained.load reads; a "
                         "diffusion run starts from a published pipeline or a saved diffusion run")
    if revision is not None:
        raise ValueError(f"{checkpoint} is a saved run's directory, which has no revision {revision!r}")
    if not text:
        raise ValueError(f"{checkpoint} is a saved run, whose tree restores its text encoder with its "
                         "denoiser; text=False skips none of it")
    settings = saved_settings(run.record, ImageSettings)
    if settings.end_to_end is not None:
        raise ValueError(f"{checkpoint} tuned its autoencoder end to end, which a run started from it "
                         "records nothing of; resume it under its name (--trainer.resume), or serve it "
                         "(TextToImage.from_run)")
    # The tree as the checkpoint holds it, not as the task reads it whole, its loss heads the run's own.
    tree = (without_loss_heads(restored_tree(run, ema=ema, mesh=mesh, layout=layout, param_dtype=param_dtype))
            if variables is None else variables)
    task = TextToImage.from_saved(run, dtype=dtype, variables=tree)
    sample = task.inputs.sample
    if size is not None and tuple(size) != sample.shape[:-1]:
        raise ValueError(f"{checkpoint} trained on {sample.key} of shape {sample.shape[:-1]}, to which its "
                         f"conditioning, grid and process are bound, and this run's data is {tuple(size)}; "
                         "train on data of the run's size")
    model = task.model
    if "attention_impl" in {field.name for field in fields(model)}:
        model = model.clone(attention_impl=attention_impl)
    elif attention_impl != "auto":
        raise ValueError(f"{type(model).__name__} takes no attention_impl: the attention its record states "
                         "is its run's")
    # The grid as its record states it, the one `from_saved` built the task's from.
    grid = None if settings.grid is None else from_record(SourceGrid, record(settings.grid, "grid"))
    return RunPipeline(model, tree, task.process, task.inputs, task.autoencoder, task.solver, task.steps,
                       task.guidance, grid)


def released_run(name_or_dir: str | Path, *, step: int | str | None = None, ema: bool | None = None,
                 trust: Sequence[str] = (), mesh: MeshSpec | Mesh | None = None,
                 layout: Layout | None = None) -> str | Path:
    """`name_or_dir` as a language model's or a decision task's loader reads it: itself, or where it is a
    saved Dew run (`pinned_run`), the release its checkpoint at `step` with `ema` exports to
    (`export_run`), written once under Dew's cache by what the export reads (`_fingerprint`) and read
    again by every later load of it. A diffusion run, which has no release (`samples_images`), is
    refused before anything is restored.

    Every process of a pool agrees on whether any of them lacks the release, and then all of them export
    it, each writing its own cache, as a restore is collective; the export restores where the load
    would place it (`mesh`, `layout`), so it holds no more of the weights on a device than the load
    does. A load without weights (a resume's, which its restore fills) reads the release's records and
    headers, so a host whose cache lacks the release (a resumed job on another node, a cleared cache, a
    run whose records changed) writes it as the first load did (ledger 2774)."""
    run = pinned_run(name_or_dir, step=step, ema=ema, trust=trust)
    if run is None:
        return name_or_dir
    if samples_images(run):
        raise ValueError(f"{name_or_dir} is a diffusion run, which Dew writes no release of: a diffusion run "
                         "starts from it (load_diffusion_source), and TextToImage.from_run serves it")
    release = Path(dew_cache_dir()) / "saved_runs" / _fingerprint(run, ema)
    lacking = any(from_every_process(not release.is_dir()))
    if not lacking:
        return release
    # Every process restores and writes, the pool reading its tensors home together. Where the pool
    # shares the cache and every process asks for the one release, the first publishes it and the others
    # write theirs to a directory of their own, discarded; elsewhere each publishes its own, and one that
    # holds it already keeps it (`staged`). A key reads each host's own checkpoint files
    # (`_fingerprint`), which copies of one checkpoint on several hosts may stamp apart.
    shared = (jax.process_count() > 1 and len(set(from_every_process(release.name))) == 1
              and all(seen_by_every_process(str(release.parent))))
    if shared and jax.process_index() != 0:
        with tempfile.TemporaryDirectory() as discarded:
            export_run(run, discarded, ema=ema, mesh=mesh, layout=layout)
    else:
        with staged(release) as destination:
            export_run(run, destination, ema=ema, mesh=mesh, layout=layout)
    from_every_process(None)  # every process reads the release once it is published
    return release
