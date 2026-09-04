"""Publishing a run's checkpoint, as a step a recipe takes after `fit`.

Training persists; publishing is reporting. The trainer never uploads or
deletes anything, so a registry outage cannot take a run down and a checkpoint
on disk is never the copy that gets removed.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from dew.interop.manifest import FILE as MANIFEST_FILE
from dew.training.tracker import WandbTracker

REGISTRY = "wandb-registry-model"


def publish(directory: str, name: str, *, tracker: WandbTracker,
            aliases: Sequence[str] = (), registry: str = REGISTRY):
    """Log the checkpoint step directory at `directory` as a model artifact of
    the tracker's run, with the run's manifest beside it, and link it into the
    W&B model registry under `name`.

    `directory` is one step directory (`Checkpoints.path(step)`), and the
    manifest is read from its parent, the run directory, so what is published
    is exactly what `Pipeline.from_run` needs. The artifact carries 'latest'
    and `aliases`; the registry link carries `aliases`.
    """
    import wandb

    artifact = wandb.Artifact(name=name, type="model")
    artifact.add_dir(directory)
    manifest = os.path.join(os.path.dirname(directory.rstrip(os.sep)), MANIFEST_FILE)
    if os.path.exists(manifest):
        artifact.add_file(manifest)
    logged = tracker.run.log_artifact(artifact, aliases=["latest", *aliases])
    tracker.run.link_artifact(
        artifact=logged, target_path=f"{registry}/{name}", aliases=list(aliases))
    return logged
