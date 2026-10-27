"""Publishing a run's checkpoint, as a step a recipe takes after `fit`, and fetching one back.

The trainer never uploads or deletes anything: a registry outage leaves a
run running, and a checkpoint on disk is never the copy that gets removed.
`promote` publishes a run only where it ranks among its project's or
sweep's best, and `downloaded` fetches a published run back from a W&B run
or the registry, which `dew.pipeline("wandb:...")` loads in one call.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
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
"""The alias a registry collection's best version holds (`promote`)."""


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


LEASE_POLL_SECONDS = 1.0
"""How often a promoter waiting on another's lease reads it again."""


@dataclasses.dataclass(frozen=True)
class Lease:
    """The one promoter at a time of a registry collection (`promote`), held at `url`, an object at any
    URL `dew.data.storage.Store` writes conditionally (a shared directory's file, a bucket's object),
    which every promoter of the collection names: for at most `hold` seconds, after which another may
    take it over, as from a promoter that died holding it, and waited on for at most `wait` seconds."""

    url: str
    hold: float = 300.0
    wait: float = 900.0


@contextlib.contextmanager
def leased(lease: Lease) -> Iterator[Callable[[], None]]:
    """Hold `lease`, taken by a compare-and-swap of its object (`Store.put_if_absent`, `put_if_match`)
    where it is absent, released or past its hold, else read again every `LEASE_POLL_SECONDS` until it is,
    up to its `wait`, then TimeoutError: the library's one wait on a promotion's boundary. Yields a check
    a holder calls before each write it makes under the lease, which raises once less than half its hold
    remains, so no write starts that another holder could follow. Released on the way out, an error
    included; a holder that dies keeps it until its hold passes."""
    from dew.data.storage import PreconditionFailed, Store

    store, holder = Store(lease.url), uuid.uuid4().hex
    deadline = time.monotonic() + lease.wait
    while True:
        try:
            found = store.get_versioned()
        except FileNotFoundError:
            found = None
        held = None if found is None else json.loads(found.data)
        now = time.time()
        if held is None or held["holder"] is None or held["until"] <= now:
            until = now + lease.hold
            record = json.dumps({"holder": holder, "until": until}).encode()
            try:
                version = (store.put_if_absent(record) if found is None
                           else store.put_if_match(record, found.version))
                break
            except PreconditionFailed:
                continue  # another promoter took it between the read and the write
        if time.monotonic() >= deadline:
            raise TimeoutError(f"the promotion lease {lease.url} stayed held for {lease.wait:.0f} s, by "
                               f"{held['holder']} until {held['until']:.0f}")
        time.sleep(LEASE_POLL_SECONDS)

    def live() -> None:
        if time.time() > until - lease.hold / 2:
            raise TimeoutError(f"the promotion lease {lease.url} has less than half its {lease.hold:.0f} s "
                               "hold left; promote again")

    try:
        yield live
    finally:
        try:
            store.put_if_match(json.dumps({"holder": None, "until": 0.0}).encode(), version)
        except PreconditionFailed as taken:
            raise RuntimeError(f"the promotion lease {lease.url} was taken over before its holder "
                               "released it: a holder outlived its hold") from taken


@dataclasses.dataclass(frozen=True)
class Promotion:
    """What a configured run is ranked by before it publishes (`promote`): its summary's `metric`, lower
    better under "min", among the best `top` of its sweep's or project's runs, published into the registry
    collection `collection`, which every run of the sweep shares, so its `best` alias follows the sweep's
    winner (`dew.pipeline("wandb:wandb-registry-model/<collection>:best")`). `lease` is the URL of the
    collection's lease (`Lease`), which every promoter of it names: by default beside the runs'
    checkpoints, which a sweep's trials share (`lease_at`)."""

    metric: str
    collection: str
    mode: Literal["min", "max"] = "min"
    top: int = 5
    lease: str | None = None

    def __post_init__(self) -> None:
        if self.top < 1:
            raise ValueError(f"a promotion ranks a run among the best top, at least one; got {self.top}")

    def lease_at(self, checkpoints: str) -> Lease:
        """The collection's lease: `lease`, else `<checkpoints>/<collection>.promotion-lease`."""
        return Lease(self.lease or f"{checkpoints.rstrip('/')}/{self.collection}.promotion-lease")


def crown(name: str, metric: str, *, mode: Literal["min", "max"]):
    """Put the registry collection `name`'s `BEST` alias on its best version by the `metric` its
    versions' metadata records (`publish`), the earliest of a tie, a version of no finite score never,
    from one fresh read of the collection, and return that version, or None for a collection of no
    scored version. One write, which the promoter holding the collection's lease makes (`promote`): no
    other promoter publishes or crowns between its read and its write."""
    import wandb

    def score(version) -> float:
        value = scored(version.metadata.get(metric))
        assert value is not None, "a version without a score is never ranked"
        return value if mode == "min" else -value

    versions = [version for version in wandb.Api().artifacts("model", f"{REGISTRY}/{name}")
                if scored(version.metadata.get(metric)) is not None]
    if not versions:
        return None
    best = min(versions, key=lambda version: (score(version), int(version.version.removeprefix("v"))))
    if BEST not in best.aliases:
        best.aliases.append(BEST)
        best.save()
    return best


def promote(directory: str, name: str, *, tracker: WandbTracker, metric: str, lease: Lease,
            mode: Literal["min", "max"] = "min", top: int = 5,
            filters: Mapping[str, object] | None = None):
    """Publish the checkpoint step at `directory` into the registry collection `name` (`publish`), its
    score recorded as its metadata, where the tracker's run ranks among the best `top` of the runs it is
    compared with (`ranked`) by its summary's `metric`, as FlaxDiff's trainer promoted its sweeps' runs, and
    put the collection's `BEST` alias on its best version (`crown`). Publishing and crowning happen under
    the collection's `lease`, one promoter at a time, so each crown reads every version published before
    it and no promoter's write follows another's on a stale read. A run tied with another ranks below
    it, so a promotion needs a strictly better score. Returns the logged artifact, or None where the run
    ranks below the best `top`, which publishes nothing and leaves the checkpoint where it is; only process
    zero compares and publishes, and it returns None everywhere else. Raises ValueError for a run whose
    summary holds no finite score: a diverged run never takes the alias."""
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
    with leased(lease) as live:
        live()
        logged = publish(directory, name, tracker=tracker, metadata={metric: value})
        live()
        crown(name, metric, mode=mode)
    return logged


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
