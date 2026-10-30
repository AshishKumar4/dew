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
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from dew.cache import dew_cache_dir
from dew.checkpoints import RUN_FILE, Checkpoints

if TYPE_CHECKING:
    from dew.inference.tasks import SavedRun


def export_run(run: SavedRun, destination: str | Path, *, ema: bool | None = None) -> None:
    """Write `run`'s pinned checkpoint (`SavedRun.pinned`) to `destination` as its release, its EMA
    weights where `ema`: a task's own released layout, or the model in its family's source layout."""
    from dew.interop.pretrained import Pretrained
    from dew.objectives.base import ReleasedTask

    task = run.objective.saved_task
    if task is not None and issubclass(task, ReleasedTask):
        task.from_saved(run, ema=ema).save_pretrained(str(destination))
    else:
        Pretrained.from_saved(run, ema=ema).save(destination)


def is_saved_run(name_or_dir: str | Path) -> bool:
    """Whether `name_or_dir` is a saved Dew run's directory: one holding its run record (`RUN_FILE`)."""
    return Path(name_or_dir).is_dir() and (Path(name_or_dir) / RUN_FILE).is_file()


def released_run(name_or_dir: str | Path, *, step: int | str | None = None, ema: bool | None = None,
                 trust: Sequence[str] = ()) -> str | Path:
    """`name_or_dir` as a loader reads it: itself, or where it is a saved Dew run (`is_saved_run`), the
    release its checkpoint at `step` (`SavedRun.pinned`: a step, `best`, `best:<metric>`, or the one its
    task loads by default) with `ema` exports to (`export_run`), written once under Dew's cache by the
    run, its step's checkpoint and the EMA choice, and read again by every later load of them. A step or
    EMA choice beside a source that is no saved run is refused: it selects nothing."""
    from dew.inference.tasks import SavedRun

    if not is_saved_run(name_or_dir):
        if step is not None or ema is not None:
            raise ValueError(f"{name_or_dir} is no saved Dew run (no {RUN_FILE}), so a step or an EMA choice "
                             "selects nothing of it; name a run's directory, or drop them")
        return name_or_dir
    directory = os.path.realpath(name_or_dir)
    run = SavedRun.pinned(directory, step, trust)
    with Checkpoints(directory) as checkpoints:
        written = os.stat(checkpoints.path(run.step)).st_mtime_ns
    key = hashlib.sha256(f"{directory}\0{run.step}\0{ema}\0{written}".encode()).hexdigest()[:24]
    release = Path(dew_cache_dir()) / "saved_runs" / key
    if not release.is_dir():
        release.parent.mkdir(parents=True, exist_ok=True)
        staged = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=release.parent))
        export_run(run, staged, ema=ema)
        try:
            staged.rename(release)
        except OSError:
            if not release.is_dir():
                raise
            # Another process wrote the same release first.
            shutil.rmtree(staged)
    return release
