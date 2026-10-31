"""A saved Dew run as the source another run starts from.

`dew export` writes a run's checkpoint at a step as a release (`export_run`): a task that publishes its
own released layout writes it (a decision run's, `ReleasedTask`), any other run its model in the
source layout its family reads (`Pretrained.from_saved`). A run that starts from a saved run reads
that release (`released_run`): every loader takes a source by its directory, so the run's load
options (its compute and storage dtypes, its mesh, its source options, an adapter) apply to it as to
any release, and its first weights are the export's.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import Field, replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from etils import epath
from jax.sharding import Mesh

from dew._coordination import agreed_same, collective_host, from_every_process
from dew.cache import dew_cache_dir
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
               mesh: MeshSpec | Mesh | None = None, layout: Layout | None = None) -> None:
    """Write `run`'s pinned checkpoint (`SavedRun.pinned`) to `destination` as its release, its EMA
    weights where `ema`: a task's own released layout (`ReleasedTask`), or the model in its family's
    source layout (`Pretrained.from_saved`). Its weights restore where `mesh` and `layout` place them,
    so no device holds more of them than a load does, and every process of a pool, each writing a
    release of its own, reads them home together (`collective_host`) for the writer, which writes
    whole host tensors."""
    from dew.interop.pretrained import Pretrained
    from dew.objectives.base import ReleasedTask

    task = run.objective.saved_task
    if task is not None and issubclass(task, ReleasedTask):
        released = task.from_saved(run, ema=ema, mesh=mesh, layout=layout)
        if not isinstance(released, _Weighted):
            raise TypeError(f"{task.__name__} releases no dataclass of its weights (variables), which an "
                            "export reads home before it writes them")
        hosted = collective_host(released.variables, phase="saved run export")
        replace(released, variables=hosted).save_pretrained(str(destination))
    else:
        bundle = Pretrained.from_saved(run, ema=ema, mesh=mesh, layout=layout)
        hosted = collective_host(bundle.variables, phase="saved run export")
        replace(bundle, variables=hosted).save(destination)


def is_saved_run(name_or_dir: str | Path) -> bool:
    """Whether `name_or_dir` is a saved Dew run's directory, on this host or remote storage
    (`etils.epath`): one holding its run record (`RUN_FILE`)."""
    path = epath.Path(name_or_dir)
    return path.is_dir() and (path / RUN_FILE).is_file()


def _fingerprint(run: SavedRun, ema: bool | None) -> str:
    """What an export of `run`'s pinned step with `ema` is a function of: the run's location, its step,
    the EMA choice, the contents of the run's own records beside its checkpoints (its run record and a
    task's assets, a decision task's calibration among them), and the step's metadata files, by size and
    write time, which a new save of the step rewrites."""
    root = epath.Path(run.directory)
    digest = hashlib.sha256(json.dumps([str(root), run.step, ema]).encode())
    for path in sorted((path for path in root.iterdir() if path.is_file()), key=str):
        digest.update(f"{path.name}\0".encode() + path.read_bytes())
    for path in sorted((root / str(run.step)).iterdir(), key=str):
        if path.is_file():
            stat = path.stat()
            digest.update(f"{path.name}\0{stat.length}\0{stat.mtime}\0".encode())
    return digest.hexdigest()[:24]


def released_run(name_or_dir: str | Path, *, step: int | str | None = None, ema: bool | None = None,
                 trust: Sequence[str] = (), weights: bool = True, mesh: MeshSpec | Mesh | None = None,
                 layout: Layout | None = None) -> str | Path:
    """`name_or_dir` as a loader reads it: itself, or where it is a saved Dew run (`is_saved_run`), the
    release its checkpoint at `step` (`SavedRun.pinned`: a step, `latest`, `best`, `best:<metric>`, or
    the one its task loads by default) with `ema` exports to (`export_run`), written once under Dew's
    cache by what the export reads (`_fingerprint`) and read again by every later load of it.

    Every process of a pool agrees on the step and on whether any of them lacks the release, and then
    all of them export it, each writing its own cache, as a restore is collective; the export restores
    where the load would place it (`mesh`, `layout`), so it holds no more of the weights on a device
    than the load does. A load `weights=False` (a resume's, which its restore fills) exports none: it
    reads the release's records alone where they are written, else the run's own at that step. A step
    or EMA choice beside a source that is no saved run is refused: it selects nothing."""
    from dew.inference.tasks import SavedRun

    if not is_saved_run(name_or_dir):
        if step is not None or ema is not None:
            raise ValueError(f"{name_or_dir} is no saved Dew run (no {RUN_FILE}), so a step or an EMA choice "
                             "selects nothing of it; name a run's directory, or drop them")
        return name_or_dir
    run = SavedRun.pinned(str(name_or_dir), agreed_same("saved run step",
                                                      lambda: SavedRun.pinned(str(name_or_dir), step,
                                                                              trust).step), trust)
    release = Path(dew_cache_dir()) / "saved_runs" / _fingerprint(run, ema)
    lacking = any(from_every_process(not release.is_dir()))
    if not lacking:
        return release
    if not weights:
        raise ValueError(f"{name_or_dir} at step {run.step} has no release written under Dew's cache, and a "
                         "load without weights (a resume's) restores none to write one; load it with its "
                         "weights once, or start the run from its export (dew export)")
    # Every process restores, a collective; one that holds the release already, or that another process
    # of a shared cache published first, keeps it (`staged`).
    with staged(release) as destination:
        export_run(run, destination, ema=ema, mesh=mesh, layout=layout)
    return release
