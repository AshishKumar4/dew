"""Live inspection of a run's parameters, gradients and updates (row 2317).

A run names what it watches (`Inspection`): parameter paths by glob, the trees whose leaves at those paths
it summarizes, and every how many steps. The summaries are reduced inside the jitted step, over the global
arrays (each process's share reduced in the one SPMD program, replicas counted once), so only scalars
leave the devices; they ride the step's commit telemetry (`dew.training.transaction.Committed`) and are
read one log late off the step loop (`dew.training.interval.LogWriter`). A run with no inspection compiles
its step unchanged; one with an inspection compiles a second, sampled step, which the loop takes on the
steps it samples."""

from __future__ import annotations

import dataclasses
import fnmatch
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, SupportsFloat

import jax
import jax.numpy as jnp
import optax

from dew.nn.inspection import Watched, statistics
from dew.optim import parameter_name

if TYPE_CHECKING:
    from dew.batch import Batch
    from dew.training.trainer import Trainer

Target = Literal["params", "grads", "updates", "deltas"]
"""What a summary reads at a sampled commit: the parameters the update applies to, the pooled gradient the
optimizer is handed (unscaled, before its own transforms such as clipping), the optimizer's update, and the
parameters' actual change (`deltas`), which a constraint's projection or a post-step effect makes differ
from the update."""


@dataclasses.dataclass(frozen=True)
class Inspection:
    """What a run watches and how often (row 2317).

    `paths` are globs over a parameter's path under `params` (`dew.optim.parameter_name`:
    `layers_0/attn/q_proj/kernel`, `*norm*`), as a `ParamGroup`'s patterns are; each leaf a glob
    selects is summarized on its own, for each target: its mean, standard deviation, root mean square,
    largest magnitude, non-finite share and zero fraction. Every sampled step also reports each
    target's global L2 norm, over every parameter, and the update-to-weight ratio,
    ||update|| / ||params||. `every` is the steps
    between samples; the run's logs carry the latest sample, so a cadence of the logs' or a multiple of
    it reports each sample once."""

    paths: tuple[str, ...] = ()
    targets: tuple[Target, ...] = ("params", "grads", "updates", "deltas")
    every: int = 100
    modules: tuple[str, ...] = ()
    """Module kinds whose outputs a sampled step summarizes, by class-name glob (`*Attention`, `MLP`):
    each call's output, its first array where it returns several, one summary a layer in a scanned
    stack (`dew.nn.inspection.watching`), logged as `inspect/activations/<module path>/<statistic>`."""
    histogram: bool = False
    """Whether each summary also holds the share of its magnitudes in each decade
    (`dew.nn.inspection.HISTOGRAM_EDGES`), logged one scalar a bin."""

    @property
    def watched(self) -> Watched | None:
        """The activations a sampled step watches, where `modules` names any."""
        return Watched(self.modules, self.histogram) if self.modules else None

    def __post_init__(self):
        if self.every < 1:
            raise ValueError(f"an inspection samples every positive number of steps, not {self.every}")
        unknown = {str(target) for target in self.targets} - {"params", "grads", "updates", "deltas"}
        if unknown:
            raise ValueError(f"an inspection's targets are params, grads, updates and deltas, not "
                             f"{sorted(unknown)}")

    def samples(self, step: int) -> bool:
        """Whether the step that ends at `step` is sampled."""
        return step % self.every == 0


def _leaf_statistics(leaf: jax.Array, *, histogram: bool) -> dict[str, jax.Array]:
    """A leaf's summary over its global array, as a watched activation's is
    (`dew.nn.inspection.statistics`), and the fraction of it that is exactly zero; a histogram's bins
    one scalar each, `histogram/<bin>`."""
    found = {**statistics(leaf, histogram=histogram), "zero_fraction": (leaf == 0).mean()}
    bins = found.pop("histogram", None)
    return found if bins is None else {**found, **{f"histogram/{index}": bins[index]
                                                   for index in range(bins.shape[0])}}


def summaries(inspection: Inspection, params, grads, updates, stepped) -> dict[str, jax.Array]:
    """The scalars a sampled commit reports, by name: `<target>/<path>/<statistic>` for each leaf a path
    glob selects, `<target>/l2` for each target, and `update_to_weight`. `stepped` is the parameters after
    the commit, of which `deltas` are the change."""
    trees: dict[str, object] = {"params": params, "grads": grads, "updates": updates,
                                "deltas": jax.tree.map(lambda new, old: new.astype(jnp.float32)
                                                       - old.astype(jnp.float32), stepped, params)}
    found: dict[str, jax.Array] = {}
    for target in inspection.targets:
        tree = trees[target]
        found[f"{target}/l2"] = optax.global_norm(jax.tree.map(lambda leaf: leaf.astype(jnp.float32), tree))
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
            name = parameter_name(path)
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in inspection.paths):
                found.update({f"{target}/{name}/{statistic}": value
                              for statistic, value in _leaf_statistics(
                                  leaf, histogram=inspection.histogram).items()})
    weights = optax.global_norm(jax.tree.map(lambda leaf: leaf.astype(jnp.float32), params))
    found["update_to_weight"] = optax.global_norm(
        jax.tree.map(lambda leaf: leaf.astype(jnp.float32), updates)) / weights
    return found


def logged(inspected: Mapping[str, SupportsFloat]) -> dict[str, float]:
    """A sample's scalars as a run logs them, under `inspect/`."""
    return {f"inspect/{name}": float(value) for name, value in inspected.items()}


def inspected(trainer: Trainer, batch: Batch, *, steps: int | None = None) -> dict[str, float]:
    """The scalars `trainer`'s inspection logs, of one sampled step over `batch` from the state the
    trainer places, its checkpoints' latest where it has them, else its start: the view a live run
    logs, for a loaded run on one batch offline. An `OptimConfig` is built over `steps`, the run's
    length, as `fit(steps=...)` builds it. The stepped state is discarded."""
    if trainer.inspection is None:
        raise ValueError("the trainer names no inspection (Trainer(inspection=...)) to report")
    trainer.scheduled(None if steps is None else steps // trainer.accumulation)
    state, _, _ = trainer.place()
    *_, committed = trainer.compile(state, batch)(state, batch, sampled=True)
    assert committed is not None, "a trainer that inspects runs its own transaction, which commits"
    return logged(jax.device_get(committed.inspected))
