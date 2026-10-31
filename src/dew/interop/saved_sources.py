"""A saved Dew run as the source another run starts from.

A source names a saved run by its directory, and `pinned_run` resolves it, for every loader, to the
run's checkpoint at the step its selection pins. `dew export` writes that checkpoint as a release
(`export_run`): a task that publishes its own released layout writes it (a decision run's,
`ReleasedTask`), any other run its model in the source layout its family reads
(`Pretrained.from_saved`). A language model or decision run that starts from a saved run reads that
release (`released_run`): every loader takes a source by its directory, so the run's load options
(its compute and storage dtypes, its mesh, its source options, an adapter) apply to it as to any
release, and its first weights are the export's. Dew writes no release of a diffusion run
(`samples_images`), and a diffusion run starts from the run's own sampling task instead
(`load_diffusion_source`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import Field, replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import jax
from etils import epath
from jax.sharding import Mesh

from dew._coordination import agreed_same, collective_host, from_every_process, seen_by_every_process
from dew.cache import dew_cache_dir, package_digest
from dew.checkpoints import RUN_FILE
from dew.files import staged

if TYPE_CHECKING:
    from dew.batch import Variables
    from dew.distributed.placement import Layout, MeshSpec
    from dew.inference.tasks import SavedRun


@runtime_checkable
class _Weighted(Protocol):
    """A released task as an export writes it: a dataclass of its weights (`variables`)."""

    __dataclass_fields__: ClassVar[dict[str, Field[object]]]
    variables: Variables

    def save_pretrained(self, directory: str) -> None: ...


def export_run(run: SavedRun, destination: str | Path, *, ema: bool | None = None,
               mesh: MeshSpec | Mesh | None = None, layout: Layout | None = None,
               shared: bool = False) -> None:
    """Write `run`'s pinned checkpoint (`SavedRun.pinned`) to `destination` as its release, its EMA
    weights where `ema`: a task's own released layout (`ReleasedTask`), or the model in its family's
    source layout (`Pretrained.from_saved`). Its weights restore where `mesh` and `layout` place them,
    so no device holds more of them than a load does.

    On one process the writer reads them from the devices as it writes, a shard at a time
    (`save_sharded`). A pool reads them home together (`collective_host`): every process, each writing
    a release of its own, or where the pool `shared` the destination, the first alone, which writes it
    once while the others take part in the restore and the gathers (ledger 2776)."""
    from dew.interop.pretrained import Pretrained
    from dew.objectives.base import ReleasedTask

    task = run.objective.saved_task
    if task is not None and issubclass(task, ReleasedTask):
        released = task.from_saved(run, ema=ema, mesh=mesh, layout=layout)
        if not isinstance(released, _Weighted):
            raise TypeError(f"{task.__name__} releases no dataclass of its weights (variables), which an "
                            "export reads home before it writes them")
        hosted = _written(released.variables, shared=shared)
        if hosted is not None:
            replace(released, variables=hosted).save_pretrained(str(destination))
    else:
        bundle = Pretrained.from_saved(run, ema=ema, mesh=mesh, layout=layout)
        hosted = _written(bundle.variables, shared=shared)
        if hosted is not None:
            replace(bundle, variables=hosted).save(destination)


def _written(variables: Variables, *, shared: bool) -> Variables | None:
    """`variables` as this process's writer of an export reads them (`export_run`), None on a process that
    writes none: as restored on one process, and across a pool read home by every process, or by the
    first alone where the pool `shared` the destination."""
    if jax.process_count() == 1:
        return variables
    return collective_host(variables, phase="saved run export", held_by="first" if shared else "every")


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
    starts from its own task (`load_diffusion_source`)."""
    from dew.inference.images import TextToImage

    task = run.objective.saved_task
    return task is not None and issubclass(task, TextToImage)


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
    # Every process restores, a collective. Where the pool shares the cache, the first writes the release
    # alone; elsewhere each writes its own, and one that holds it already keeps it (`staged`).
    shared = jax.process_count() > 1 and all(seen_by_every_process(str(release.parent)))
    if shared and jax.process_index() != 0:
        export_run(run, release, ema=ema, mesh=mesh, layout=layout, shared=True)
    else:
        with staged(release) as destination:
            export_run(run, destination, ema=ema, mesh=mesh, layout=layout, shared=shared)
    from_every_process(None)  # every process reads the release once it is published
    return release
