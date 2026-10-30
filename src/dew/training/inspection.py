"""Live inspection of a run's parameters, gradients and updates (row 2317).

A run names what it watches (`Inspection`): parameter paths by glob, the trees whose leaves at those paths
it summarizes, and every how many optimizer updates. The summaries are reduced inside the jitted step,
over the global arrays (each process's share reduced in the one SPMD program, replicas counted once), so
only scalars leave the devices; they ride the step's commit telemetry (`dew.training.transaction.Committed`)
and are read one log late off the step loop (`dew.training.interval.LogWriter`). A run with no inspection
compiles its step unchanged; one with an inspection compiles a second, sampled step, which the loop takes
on the steps it samples."""

from __future__ import annotations

import dataclasses
import fnmatch
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

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
    ||update|| / ||params||. `every` is the optimizer updates between samples, each the step that commits
    its update, however many steps an update pools (`Trainer.accumulation`); the run's logs carry the
    latest sample, so a cadence of the logs' or a multiple of it reports each sample once."""

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
            raise ValueError(f"an inspection samples every positive number of updates, not {self.every}")
        unknown = {str(target) for target in self.targets} - {"params", "grads", "updates", "deltas"}
        if unknown:
            raise ValueError(f"an inspection's targets are params, grads, updates and deltas, not "
                             f"{sorted(unknown)}")

    def samples(self, step: int, accumulation: int) -> bool:
        """Whether the step that ends at `step` is sampled: the one that commits every `every`-th update,
        `accumulation` steps an update, since a step that commits nothing reports nothing."""
        return step % accumulation == 0 and (step // accumulation) % self.every == 0


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
                              for statistic, value in statistics(
                                  leaf, histogram=inspection.histogram).items()})
    weights = optax.global_norm(jax.tree.map(lambda leaf: leaf.astype(jnp.float32), params))
    found["update_to_weight"] = optax.global_norm(
        jax.tree.map(lambda leaf: leaf.astype(jnp.float32), updates)) / weights
    return found


@struct.dataclass
class Summaries:
    """A sampled commit's scalars (`summaries`, `dew.training.transaction.watched_activations`) as one
    float32 vector, `values`, beside their names, `names`, which its step fixes as it traces: a sample
    crosses the step loop and reaches the host as one array, named off it by the log's reader
    (`logged`), not one scalar each (row 2723)."""

    values: jax.Array
    names: tuple[str, ...] = struct.field(pytree_node=False)


def stacked(scalars: Mapping[str, jax.Array]) -> Summaries | None:
    """`scalars` as one vector (`Summaries`) in their order, None where there are none, a step that samples
    nothing."""
    if not scalars:
        return None
    names = tuple(scalars)
    return Summaries(jnp.stack([jnp.asarray(scalars[name], jnp.float32) for name in names]), names)


def logged(inspected: Summaries) -> dict[str, float]:
    """A sample's scalars as a run logs them, under `inspect/`, each by its name."""
    return {f"inspect/{name}": float(value)
            for name, value in zip(inspected.names, np.asarray(inspected.values), strict=True)}


def inspected[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], batch: Batch, *,
                                steps: int | None = None) -> dict[str, float]:
    """The scalars `trainer`'s inspection logs, of one sampled update over `batch` from the state the
    trainer places, its checkpoints' latest where it has them, else its start: the view a live run
    logs, for a loaded run on one batch offline. A trainer that accumulates steps to the end of its window,
    from where its state stands in it (a fresh window or a restored part of one), `batch` each of its
    microbatches, and reports the step that commits. An `OptimConfig` is built over
    `steps`, the run's length, as `fit(steps=...)` builds it. The stepped state is discarded."""
    if trainer.inspection is None:
        raise ValueError("the trainer names no inspection (Trainer(inspection=...)) to report")
    # An objective that names its input field alone reads its shape off the batch, as `fit` does.
    trainer._declare_input(lambda: batch)
    trainer.scheduled(None if steps is None else steps // trainer.accumulation)
    state, _, _ = trainer.place()
    step, window = trainer.compile(state, batch), int(state.window_size)
    # A window is split x accumulation microbatches (`TrainState.window_size`), and each call steps a
    # split's parts of it, a rejected part's place too: the window's rest, rounded up, in calls.
    split = window // trainer.accumulation
    committed = None
    for _ in range(-(-(window - int(state.microstep) % window) // split)):
        state, *_, committed = step(state, batch, sampled=True)
    if committed is None or committed.inspected is None or not int(committed.count):
        raise ValueError("the sampled window did not commit over this batch (a part of it rejected, its "
                         "loss not finite); inspect a batch whose window commits")
    return logged(jax.device_get(committed.inspected))
