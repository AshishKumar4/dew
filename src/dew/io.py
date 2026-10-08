"""Publishing a run's checkpoint, as a step a recipe takes after `fit`.

The trainer never uploads or deletes anything: a registry outage leaves a
run running, and a checkpoint on disk is never the copy that gets removed.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
from etils import epath

from dew.checkpoints import RUN_FILE, frozen_entries, is_uri
from dew.training.tracker import WandbTracker

REGISTRY = "wandb-registry-model"


def publish(directory: str, name: str, *, tracker: WandbTracker,
            aliases: Sequence[str] = ()):
    """Log the checkpoint step directory at `directory` as a model artifact of
    the tracker's run, laid out as its run directory is, and link it into the
    W&B model registry under `name`.

    `directory` is one step directory (`Checkpoints.path(step)`). The artifact
    holds it under its own name, the run's `run.json` and the
    `dew.checkpoints.FROZEN_STORE` entries the step records, each at its place
    in the run directory, so a download is a run directory `from_run` and
    `Checkpoints.restore` read. The artifact carries 'latest' and `aliases`;
    the registry link carries `aliases`.

    Only process zero publishes, and it returns None everywhere else: every
    process holds the same checkpoint, so a second upload would duplicate
    the first.

    A directory on a filesystem is uploaded. A URI is referenced instead: a
    pod writes its checkpoints to the bucket, and the bytes are already
    somewhere the registry can point at.
    """
    if jax.process_index() != 0:
        return None

    import wandb

    artifact = wandb.Artifact(name=name, type="model")
    path = epath.Path(directory)
    run = path.parent
    spec = run / RUN_FILE
    entries = frozen_entries(directory)
    if is_uri(directory):
        artifact.add_reference(str(path), name=path.name)
        if spec.exists():
            artifact.add_reference(str(spec), name=RUN_FILE)
        for entry in entries:
            artifact.add_reference(str(run / entry), name=entry)
    else:
        artifact.add_dir(directory, name=path.name)
        if spec.exists():
            artifact.add_file(str(spec), name=RUN_FILE)
        for entry in entries:
            artifact.add_dir(str(run / entry), name=entry)
    logged = tracker.run.log_artifact(artifact, aliases=["latest", *aliases])
    tracker.run.link_artifact(
        artifact=logged, target_path=f"{REGISTRY}/{name}", aliases=list(aliases))
    return logged
