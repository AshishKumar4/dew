"""What the trainer is optimizing.

The trainer owns training mechanics: the mesh, the compiled step, EMA
bookkeeping, checkpoints, logging. An `Objective` owns what is being learned:
the parameter tree it initialises, the loss it computes from a batch, and what
its evaluation produces. Swapping the objective swaps the research question
without touching any of the mechanics.

An objective receives schedule and randomness through Step, and returns
additive loss statistics with Aux reports. These values are JAX PyTrees.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeAlias
from typing_extensions import TypeVar

from flax import struct
import jax
import jax.numpy as jnp
import optax

from dew.artifacts import Artifacts

if TYPE_CHECKING:
    from dew.inputs import InputSpec
    from dew.inference.tasks import BlockGeneration, TextGeneration
    from dew.sampling.pipelines import TextToImage

    Task: TypeAlias = TextGeneration | BlockGeneration | TextToImage

Variables: TypeAlias = Mapping[str, Any]
"""A flax variables dict: the `params` collection plus any other collection
the modules keep (`moe`, `batch_stats`, an objective's frozen encoders)."""

Batch: TypeAlias = Mapping[str, Any]
Path: TypeAlias = tuple[str, ...]
PathFilter: TypeAlias = Callable[[Path], bool]
"""Selects leaves of a variables tree by the tuple of dict keys above them.
One filter type serves the EMA selection, `optax.multi_transform` labels and
frozen subtrees."""

@struct.dataclass
class Mean:
    """A scalar sum with nonnegative, parameter-independent support mass.

    Zero mass declares a zero numerator and no contribution.
    """
    total: jax.Array
    mass: jax.Array


def mean_loss(stats: Mean) -> tuple[jax.Array, jax.Array]:
    """Reduce a shared-denominator estimator, including empty support."""
    mass = jax.lax.stop_gradient(stats.mass)
    active = mass > 0
    dtype = jnp.result_type(stats.total.dtype, mass.dtype, jnp.float32)
    value = stats.total.astype(dtype) / jnp.where(active, mass.astype(dtype), 1)
    return jnp.where(active, value, 0), active


Loss = TypeVar("Loss", default=Mean | jax.Array | float)
Effects = TypeVar("Effects", default=None)


@struct.dataclass
class Step:
    """Accepted-microbatch schedule index, attempted-work key, and EMA view."""
    step: jax.Array
    key: jax.Array
    ema: Variables | None
    """The variables tree with the averaged leaves in place of the live ones,
    or None when the objective keeps no EMA."""


@struct.dataclass
class Aux(Generic[Effects]):
    """Reports, sequential mutable replacements, and deferred effects."""
    metrics: dict[str, jax.Array]
    variables: Variables | None = None
    """Complete nonparameter replacements from one accepted microbatch,
    such as BatchNorm statistics. The optimizer owns the params collection;
    deferred router-bias updates belong in effects instead."""
    qk_stats: Variables | None = None
    """The `qk` collection the attention layers sowed, for the optimizer's
    QK-Clip: nested by module path, each attention layer holding
    `max_logits` as a one-tuple of an fp32 `[rows, heads]` array of per-head
    logit maxima, an MLA layer additionally holding `qk_nope` as an int
    scalar naming its nope width. Rows are the batch's rows, microbatches
    concatenated under a pipeline. None when the loss never opened the
    collection, in which case the clip steps aside."""
    effects: Effects | None = None
    """Additive observations applied once on a supported optimizer commit."""


def everything(path: Path) -> bool:
    return True


def under(*prefix: str) -> PathFilter:
    """Leaves below `prefix`, as in `under("params", "context_encoder")`."""
    return lambda path: path[:len(prefix)] == prefix


def select(tree: Variables, keep: PathFilter) -> Variables:
    """The subtree of `tree` whose leaves `keep` accepts, with the same nesting.

    A branch that keeps no leaf is dropped, so the result is what the EMA
    stores and what `merge` puts back.
    """
    def prune(node, path):
        if isinstance(node, Mapping):
            kept = {name: prune(child, path + (name,)) for name, child in node.items()}
            return {name: child for name, child in kept.items() if child is not None} or None
        return node if keep(path) else None

    selected = prune(tree, ())
    if not selected:
        raise ValueError("the filter selected no leaf of the variables tree")
    return selected


def merge(tree: Variables, overlay: Variables) -> Variables:
    """`tree` with every leaf `overlay` holds replaced by the overlay's."""
    merged = dict(tree)
    for name, child in overlay.items():
        held = tree.get(name)
        merged[name] = (merge(held, child)
                        if isinstance(held, Mapping) and isinstance(child, Mapping)
                        else child)
    return merged


@dataclass(frozen=True)
class EMASpec:
    """Which leaves of the variables the EMA copy tracks, and how fast.

    decay is a step-indexed schedule, since momentum ramps matter for some
    objectives (I-JEPA anneals 0.996 to 1.0). The step it reads is the count
    of completed optimizer updates.
    """
    decay: optax.Schedule
    select: PathFilter = everything


class Objective(ABC, Generic[Loss, Effects]):
    """What is being learned: parameters, loss, what evaluation produces."""

    inputs: InputSpec
    """Per-example shapes and dtypes the parameter tree is initialised from."""
    ema: EMASpec | None = None
    artifact: type | None = None
    """The artifact type `evaluate` returns, or None when it returns nothing."""

    @abstractmethod
    def init(self, key: jax.Array) -> Variables:
        """The whole variables tree, every collection, from one key. Pure, so
        the trainer traces it once for shapes and once for values."""

    @abstractmethod
    def loss(self, params: Variables, batch: Batch, step: Step) -> tuple[Loss, Aux[Effects]]:
        """Additive loss statistics and the reports from one realized batch.

        Mean declares a shared normalization mass. A plain scalar is one
        unit-mass term. Composite statistics are objective-owned Flax PyTrees;
        their leaves add across records before reduce_loss is evaluated.
        """

    def reduce_loss(self, stats: Loss) -> tuple[jax.Array, jax.Array]:
        """The objective value and whether its statistical support is active."""
        if isinstance(stats, Mean):
            return mean_loss(stats)
        if isinstance(stats, (jax.Array, float, int)):
            value = jnp.asarray(stats)
            value = value.astype(jnp.promote_types(value.dtype, jnp.float32))
            if value.ndim != 0:
                raise ValueError("a unit-mass loss must be scalar")
            return value, jnp.asarray(True)
        raise TypeError("custom loss statistics require Objective.reduce_loss")

    def apply_effects(self, variables: Variables, effects: Effects) -> Variables:
        """Nonparameter replacements from accepted-window observations."""
        raise TypeError("deferred effects require Objective.apply_effects")

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> Artifacts | None:
        """Scoring artifacts for every row of the coordinated global batch.

        Every rank participates in numerical work outside the optimizer jit.
        No display sampling or decoding belongs here. Each scoring batch has
        a distinct key; `step.ema` holds the averaged weights.
        """
        return None

    def pipeline(self, state, *, ema: bool = True) -> Task:
        """The trained model as its inference task over `state`'s weights.

        `state` is the trainer's `TrainState`; the task binds `state.averaged`
        when the objective keeps an EMA and `ema` asks for it, else
        `state.params`, and runs on the mesh the state is placed on. No
        reload, no copy. Objectives that generate nothing raise.
        """
        raise TypeError(f"{type(self).__name__} has no inference task")

    def preview(self, params: Variables, batch: Batch, step: Step, *,
                scored: Artifacts | None = None) -> Artifacts | None:
        """One display per event, reusing first-batch scoring when available.

        Called on every rank with a separate preview key. Before an internal
        collective, coordinate local setup and generation failures with
        agree_process_phase so every rank reaches the same boundary. Complete
        all gathers before root-only decoding. The trainer coordinates the
        hook's final outcome before any subsequent collective.
        """
        return scored if scored is not None else self.evaluate(params, batch, step)

def published(state, ema: bool) -> Variables:
    """The weights a train state publishes: the EMA copy merged over the live
    variables when the objective keeps one and `ema` asks for it."""
    return state.averaged if ema and state.ema is not None else state.params


def scalar_loss(objective: Objective[Loss, Effects], variables: Variables,
                batch: Batch, step: Step) -> tuple[jax.Array, Aux[Effects]]:
    """Evaluate and reduce canonical statistics for direct JAX differentiation."""
    stats, aux = objective.loss(variables, batch, step)
    value, _ = objective.reduce_loss(stats)
    return value, aux


S = TypeVar("S")


class Metric(Protocol[S]):
    """Host-local statistics, merged immediately and finalized once.

    The first contribution initializes a pass. State belongs to that pass
    alone; merge may update its owned buffers in place. Metrics must never
    perform process collectives or retain state between passes.
    """

    @property
    def name(self) -> str: ...

    @property
    def reads(self) -> type:
        """The scoring artifact type this metric reads."""
        ...

    def __call__(self, artifact: Any, batch: Batch, /) -> S:
        """One complete batch's sufficient statistics."""
        ...

    def merge(self, accumulated: S, contribution: S, /) -> S:
        """Combine a contribution with the pass-owned accumulator."""
        ...

    def finalize(self, accumulated: S, /) -> float:
        """The completed pass's scalar."""
        ...
