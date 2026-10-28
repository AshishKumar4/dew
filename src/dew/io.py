"""Publishing a run's checkpoint, as a step a recipe takes after `fit`, and fetching one back.

The trainer never uploads or deletes anything: a registry outage leaves a
run running, and a checkpoint on disk is never the copy that gets removed.
`promote` publishes a run only where it ranks among its project's or
sweep's best, its score and how it is ranked recorded on the version it
publishes, and `downloaded` fetches a published run back from a W&B run or
the registry, which `dew.pipeline("wandb:...")` loads in one call. A
collection's best version (`BEST`) is derived from those records whenever it
is read (`best_version`): no promoter writes an alias, so no promoter's
write can follow another's.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Literal

import jax
import numpy as np
from etils import epath

from dew.checkpoints import frozen_entries, is_uri
from dew.training.tracker import WandbTracker

REGISTRY = "wandb-registry-model"


def publish(directory: str, name: str, *, tracker: WandbTracker,
            aliases: Sequence[str] = (), metadata: Mapping[str, object] | None = None):
    """Log the checkpoint step directory at `directory` as a model artifact of
    the tracker's run, laid out as its run directory is, and link it into the
    W&B model registry under `name`.

    `directory` is one step directory (`Checkpoints.path(step)`). The artifact
    holds it under its own name, every file at the run's root (its `run.json`,
    and what the run wrote beside it, a decision run's calibrated task) and the
    `dew.checkpoints.FROZEN_STORE` entries the step records, each at its place
    in the run directory, so a download is a run directory `from_run` and
    `Checkpoints.restore` read. The artifact carries 'latest' and `aliases`;
    the registry link carries `aliases`, and the artifact `metadata`, which a registry's versions are
    ranked by (`promote`).

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

    artifact = wandb.Artifact(name=name, type="model", metadata=dict(metadata or {}))
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


BEST = "best"
"""What a reference names after its colon for a collection's best version, which Dew derives from its
versions' records when it is read (`best_version`) rather than any alias a registry holds."""

RANKING = "dew_ranking"
"""The metadata key under which a version `promote` publishes records how its score ranks: the metric
its score is, beside it under its own name, and the mode, "min" where lower is better."""


def scored(value: float | int | str | None) -> float | None:
    """`value` as a score a ranking reads, None where it is none: a NaN, an infinity, or no number (a
    summary's number, a NumPy scalar among them)."""
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def score_of(run, metric: str) -> float:
    """A ranked run's finite score (`ranked` keeps no other)."""
    value = scored(run.summary.get(metric))
    assert value is not None, "a ranked run holds a finite score"
    return value


def ranked(run, metric: str, *, mode: Literal["min", "max"],
           filters: Mapping[str, object] | None = None) -> list:
    """The other finished and running W&B runs `run` is compared with, best first by the `metric` their
    summaries hold, those that hold no finite score left out (`scored`): the runs of its sweep where it is
    one's, else those of its project that W&B's run `filters` select (a MongoDB-style query,
    `{"config.dataset": "flowers"}`, as `wandb.Api().runs` reads it). `mode` is "min" where lower is
    better."""
    import wandb

    api = wandb.Api()
    sweep = run.sweep_id
    if sweep is not None and filters is not None:
        raise ValueError("a sweep's runs are its own set; filters select among a project's runs")
    path = f"{run.entity}/{run.project}"
    others = (api.sweep(f"{path}/{sweep}").runs if sweep is not None
              else api.runs(path, filters=None if filters is None else dict(filters)))
    held = [other for other in others if other.id != run.id and scored(other.summary.get(metric)) is not None]
    return sorted(held, key=lambda other: score_of(other, metric), reverse=mode == "max")


@dataclasses.dataclass(frozen=True)
class Promotion:
    """What a configured run is ranked by before it publishes (`promote`): its summary's `metric`, lower
    better under "min", among the best `top` of its sweep's or project's runs, published into the registry
    collection `collection`, which every run of the sweep shares, so its best version is the sweep's
    winner (`dew.pipeline("wandb:wandb-registry-model/<collection>:best")`)."""

    metric: str
    collection: str
    mode: Literal["min", "max"] = "min"
    top: int = 5

    def __post_init__(self) -> None:
        if self.top < 1:
            raise ValueError(f"a promotion ranks a run among the best top, at least one; got {self.top}")


def best_version(collection: str):
    """The best version of the W&B artifact collection `collection` (a registry's,
    `wandb-registry-model/<name>`, or a project's, `entity/project/<name>`), read afresh: of the versions
    `promote` published, each recording its score and how it ranks (`RANKING`), the best, the earliest of
    a tie. Derived when read, it is whatever the collection holds then, however its promoters
    interleaved. Raises ValueError for a collection none of whose versions records a ranking, or whose
    versions rank by different metrics or modes."""
    import wandb

    ranked_versions = [version for version in wandb.Api().artifacts("model", collection)
                       if RANKING in version.metadata]
    rules = {(version.metadata[RANKING]["metric"], version.metadata[RANKING]["mode"])
             for version in ranked_versions}
    if not rules:
        raise ValueError(f"no version of {collection} records how it ranks (dew.io.promote); name a "
                         "version or an alias of the registry's")
    if len(rules) > 1:
        raise ValueError(f"the versions of {collection} rank by {sorted(rules)}; a collection ranks by one")
    ((metric, mode),) = rules

    def score(version) -> float:
        value = scored(version.metadata.get(metric))
        assert value is not None, "a version publishes a finite score"
        return value if mode == "min" else -value

    return min(ranked_versions, key=lambda version: (score(version), int(version.version.removeprefix("v"))))


def promote(directory: str, name: str, *, tracker: WandbTracker, metric: str,
            mode: Literal["min", "max"] = "min", top: int = 5,
            filters: Mapping[str, object] | None = None):
    """Publish the checkpoint step at `directory` into the registry collection `name` (`publish`) where
    the tracker's run ranks among the best `top` of the runs it is compared with (`ranked`) by its
    summary's `metric`, as FlaxDiff's trainer promoted its sweeps' runs, its score and how it ranks
    recorded as its metadata (`RANKING`), from which the collection's best version is derived when it is
    read (`best_version`). Publishing adds a version and writes nothing else, so promoters of a collection
    need no order among themselves. A run tied with another ranks below it, so a promotion needs a
    strictly better score. Returns the logged artifact, or None where the run ranks below the best `top`,
    which publishes nothing and leaves the checkpoint where it is; only process zero compares and
    publishes, and it returns None everywhere else. Raises ValueError for a run whose summary holds no
    finite score: a diverged run is never published."""
    Promotion(metric, name, mode, top)
    if jax.process_index() != 0:
        return None
    run = tracker.run
    value = scored(run.summary.get(metric))
    if value is None:
        raise ValueError(f"the run's summary holds no finite {metric!r} to rank it by, "
                         f"{run.summary.get(metric)!r}; log one before promoting")
    better = (lambda score: score <= value) if mode == "min" else (lambda score: score >= value)
    ahead = sum(better(score_of(other, metric)) for other in ranked(run, metric, mode=mode, filters=filters))
    if ahead >= top:
        return None
    return publish(directory, name, tracker=tracker,
                   metadata={metric: value, RANKING: {"metric": metric, "mode": mode}})


def downloaded(reference: str, *, root: str | None = None) -> str:
    """The run directory a published run downloads to (`publish`'s layout), from W&B: `reference` is an
    artifact's path, holding after a colon its version, an alias, or `BEST` for the collection's best
    version, derived from its versions' records (`best_version`; `wandb-registry-model/name:best`,
    `entity/project/name:v3`), or a run's, `entity/project/run_id`, whose newest logged model artifact it
    fetches. `root` is where it lands, W&B's own artifacts directory where None."""
    import wandb

    collection, colon, version = reference.rpartition(":")
    if colon and version == BEST:
        return str(best_version(collection).download(root=root))
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
