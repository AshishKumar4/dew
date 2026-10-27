"""Publishing a run's checkpoint, as a step a recipe takes after `fit`, and fetching one back.

The trainer never uploads or deletes anything: a registry outage leaves a
run running, and a checkpoint on disk is never the copy that gets removed.
`promote` publishes a run only where it ranks among its project's or
sweep's best, and `downloaded` fetches a published run back from a W&B run
or the registry, which `dew.pipeline("wandb:...")` loads in one call.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Literal

import jax
from etils import epath

from dew.checkpoints import frozen_entries, is_uri
from dew.training.tracker import WandbTracker

REGISTRY = "wandb-registry-model"


def publish(directory: str, name: str, *, tracker: WandbTracker,
            aliases: Sequence[str] = ()):
    """Log the checkpoint step directory at `directory` as a model artifact of
    the tracker's run, laid out as its run directory is, and link it into the
    W&B model registry under `name`.

    `directory` is one step directory (`Checkpoints.path(step)`). The artifact
    holds it under its own name, every file at the run's root (its `run.json`,
    and what the run wrote beside it, a decision run's calibrated task) and the
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
    files = sorted((entry for entry in run.iterdir() if entry.is_file()), key=lambda entry: entry.name)
    entries = frozen_entries(directory)
    if is_uri(directory):
        artifact.add_reference(str(path), name=path.name)
        for file in files:
            artifact.add_reference(str(file), name=file.name)
        for entry in entries:
            artifact.add_reference(str(run / entry), name=entry)
    else:
        artifact.add_dir(directory, name=path.name)
        for file in files:
            artifact.add_file(str(file), name=file.name)
        for entry in entries:
            artifact.add_dir(str(run / entry), name=entry)
    logged = tracker.run.log_artifact(artifact, aliases=["latest", *aliases])
    tracker.run.link_artifact(
        artifact=logged, target_path=f"{REGISTRY}/{name}", aliases=list(aliases))
    return logged


def ranked(run, metric: str, *, mode: Literal["min", "max"],
           filters: Mapping[str, object] | None = None) -> list:
    """The other finished and running W&B runs `run` is compared with, best first by the `metric` their
    summaries hold, those that hold none left out: the runs of its sweep where it is one's, else those of
    its project that W&B's run `filters` select (a MongoDB-style query, `{"config.dataset": "flowers"}`,
    as `wandb.Api().runs` reads it). `mode` is "min" where lower is better."""
    import wandb

    api = wandb.Api()
    sweep = getattr(run, "sweep_id", None)
    if sweep is not None and filters is not None:
        raise ValueError("a sweep's runs are its own set; filters select among a project's runs")
    path = f"{run.entity}/{run.project}"
    others = api.sweep(f"{path}/{sweep}").runs if sweep is not None else api.runs(path, filters=filters)
    scored = [other for other in others if other.id != run.id and metric in other.summary]
    return sorted(scored, key=lambda other: other.summary[metric], reverse=mode == "max")


@dataclasses.dataclass(frozen=True)
class Promotion:
    """What a configured run is ranked by before it publishes (`promote`): its summary's `metric`, lower
    better under "min", among the best `top` of its sweep's or project's runs."""

    metric: str
    mode: Literal["min", "max"] = "min"
    top: int = 5

    def __post_init__(self) -> None:
        if self.top < 1:
            raise ValueError(f"a promotion ranks a run among the best top, at least one; got {self.top}")


def promote(directory: str, name: str, *, tracker: WandbTracker, metric: str,
            mode: Literal["min", "max"] = "min", top: int = 5,
            filters: Mapping[str, object] | None = None):
    """Publish the checkpoint step at `directory` as `name` (`publish`) where the tracker's run ranks among
    the best `top` of the runs it is compared with (`ranked`) by its summary's `metric`, aliased "best" in
    the registry where it ranks first, as FlaxDiff's trainer promoted its sweeps' runs. A run tied with
    another ranks below it, so a promotion needs a strictly better score. Returns the logged artifact, or
    None where the run ranks below the best `top`, which publishes nothing and leaves the checkpoint where
    it is; only process zero compares and publishes, and it returns None everywhere else."""
    Promotion(metric, mode, top)
    if jax.process_index() != 0:
        return None
    run = tracker.run
    if metric not in run.summary:
        raise ValueError(f"the run's summary holds no {metric!r} to rank it by; log it before promoting")
    value = run.summary[metric]
    better = (lambda score: score <= value) if mode == "min" else (lambda score: score >= value)
    ahead = sum(better(other.summary[metric]) for other in ranked(run, metric, mode=mode, filters=filters))
    if ahead >= top:
        return None
    return publish(directory, name, tracker=tracker, aliases=("best",) if ahead == 0 else ())


def downloaded(reference: str, *, root: str | None = None) -> str:
    """The run directory a published run downloads to (`publish`'s layout), from W&B: `reference` is an
    artifact's path, holding its version or alias after a colon (`wandb-registry-model/name:best`,
    `entity/project/name:v3`), or a run's, `entity/project/run_id`, whose newest logged model artifact it
    fetches. `root` is where it lands, W&B's own artifacts directory where None."""
    import wandb

    api = wandb.Api()
    if ":" in reference:
        artifact = api.artifact(reference, type="model")
    else:
        models = [logged for logged in api.run(reference).logged_artifacts() if logged.type == "model"]
        if not models:
            raise ValueError(f"the W&B run {reference} logged no model artifact; publish one "
                             "(dew.io.publish)")
        artifact = models[-1]
    return str(artifact.download(root=root))
