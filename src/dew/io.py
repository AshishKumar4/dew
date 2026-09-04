"""Publishing a run's checkpoint, as a step a recipe takes after `fit`.

Training persists; publishing is reporting. The trainer never uploads or
deletes anything, so a registry outage cannot take a run down and a checkpoint
on disk is never the copy that gets removed.
"""

from __future__ import annotations

from collections.abc import Sequence

from etils import epath
import jax

from dew.checkpoints import is_uri
from dew.config import RUN_FILE
from dew.training.tracker import WandbTracker

REGISTRY = "wandb-registry-model"


def publish(directory: str, name: str, *, tracker: WandbTracker,
            aliases: Sequence[str] = (), registry: str = REGISTRY):
    """Log the checkpoint step directory at `directory` as a model artifact of
    the tracker's run, with the run's `run.json` beside it, and link it into the
    W&B model registry under `name`.

    `directory` is one step directory (`Checkpoints.path(step)`), and the run
    spec is read from its parent, the run directory, so what is published is
    exactly what `Pipeline.from_run` needs. The artifact carries 'latest'
    and `aliases`; the registry link carries `aliases`.

    Only process zero publishes, and it returns None everywhere else: every
    process holds the same checkpoint, so a second upload is a duplicate of
    the first rather than another part of the run.

    A directory on a filesystem is uploaded. A URI is referenced instead: a
    pod writes its checkpoints to the bucket, and the bytes are already
    somewhere the registry can point at.
    """
    if jax.process_index() != 0:
        return None

    import wandb

    artifact = wandb.Artifact(name=name, type="model")
    path = epath.Path(directory)
    spec = path.parent / RUN_FILE
    if is_uri(directory):
        artifact.add_reference(str(path))
        if spec.exists():
            artifact.add_reference(str(spec))
    else:
        artifact.add_dir(directory)
        if spec.exists():
            artifact.add_file(str(spec))
    logged = tracker.run.log_artifact(artifact, aliases=["latest", *aliases])
    tracker.run.link_artifact(
        artifact=logged, target_path=f"{registry}/{name}", aliases=list(aliases))
    return logged
