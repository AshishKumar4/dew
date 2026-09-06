"""What the trainer is optimizing.

The trainer owns training mechanics: the mesh, the compiled step, EMA
bookkeeping, checkpoints, logging. An `Objective` owns what is being learned:
the parameter tree it initialises, the loss it computes from a batch, and what
its evaluation produces. Swapping the objective swaps the research question
without touching any of the mechanics.

Everything an objective sees in one call arrives as a `Step`, and everything
it reports back rides in an `Aux`. Both are pytrees, so they cross `jit`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeVar

from flax import struct
import jax
import optax

from dew.artifacts import Artifacts

if TYPE_CHECKING:
    from dew.inputs import InputSpec

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
class Step:
    """What an objective sees in one call."""
    step: jax.Array
    key: jax.Array
    ema: Variables | None
    """The variables tree with the averaged leaves in place of the live ones,
    or None when the objective keeps no EMA."""


@struct.dataclass
class Aux:
    """What a loss reports beside its scalar."""
    metrics: dict[str, jax.Array]
    variables: Variables | None = None
    """Non-parameter collections to write back into the state as a whole: the
    MoE balancing bias, batch statistics, sown values. The `params` collection
    is the optimizer's and cannot be written this way."""
    qk_stats: Variables | None = None
    """The `qk` collection the attention layers sowed, for the optimizer's
    QK-Clip: nested by module path, each attention layer holding
    `max_logits` as a one-tuple of an fp32 `[rows, heads]` array of per-head
    logit maxima, an MLA layer additionally holding `qk_nope` as an int
    scalar naming its nope width. Rows are the batch's rows, microbatches
    concatenated under a pipeline. None when the loss never opened the
    collection, in which case the clip steps aside."""


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


class Objective(ABC):
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
    def loss(self, params: Variables, batch: Batch, step: Step) -> tuple[jax.Array, Aux]:
        """Scalar loss over the batch and what to report beside it.

        Differentiated with respect to `params["params"]`; every other
        collection is read as state and rewritten only through `Aux.variables`.
        """

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> Artifacts | None:
        """Scoring artifacts for every row of the coordinated global batch.

        Every rank participates in numerical work outside the optimizer jit.
        No display sampling or decoding belongs here. Each scoring batch has
        a distinct key; `step.ema` holds the averaged weights.
        """
        return None

    def preview(self, params: Variables, batch: Batch, step: Step, *,
                scored: Artifacts | None = None) -> Artifacts | None:
        """One display per event, reusing first-batch scoring when available.

        Called on every rank with a separate preview key. Complete all device
        work and gathers before rank-zero decoding or other host side effects.
        The trainer coordinates hook errors before any subsequent collective.
        """
        return scored if scored is not None else self.evaluate(params, batch, step)


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
