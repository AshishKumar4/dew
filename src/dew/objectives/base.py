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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

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
    """Non-parameter collections to write back into the state, whole: the MoE
    balancing bias, batch statistics, sown values. The `params` collection is
    the optimizer's and cannot be written this way."""


def everything(path: Path) -> bool:
    return True


def under(*prefix: str) -> PathFilter:
    """Leaves below `prefix`, as in `under("params", "context_encoder")`."""
    return lambda path: path[:len(prefix)] == prefix


def select(tree: Variables, keep: PathFilter) -> Variables:
    """The subtree of `tree` whose leaves `keep` accepts, with the same nesting.

    A branch that keeps no leaf is dropped rather than left empty, so the
    result is what the EMA stores and what `merge` puts back.
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

    decay is a step-indexed schedule rather than a constant because momentum
    ramps are load-bearing for some objectives (I-JEPA anneals 0.996 to 1.0).
    The step it reads is the count of completed optimizer updates.
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
        """The whole variables tree, every collection, from one key. Pure: the
        trainer traces it once for shapes and once for values."""

    @abstractmethod
    def loss(self, params: Variables, batch: Batch, step: Step) -> tuple[jax.Array, Aux]:
        """Scalar loss over the batch and what to report beside it.

        Differentiated with respect to `params["params"]`; every other
        collection is read as state and rewritten only through `Aux.variables`.
        """

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> Artifacts | None:
        """What a validation batch produces: one artifact, or a tuple of them.

        Called once per validation batch with the arrays already on the mesh
        and outside any jit, so an objective jits its device work here and
        decodes to host strings after it. `step.ema` holds the averaged weights.
        """
        return None


class Metric(Protocol):
    """A per-batch measurement of one artifact type, and its reduction over a pass."""

    name: str
    reads: type
    """The artifact type this metric scores; the trainer hands it that one."""

    def __call__(self, artifact: Any, batch: Batch) -> Any:
        """One batch's measurement, whatever `reduce` needs of it."""

    def reduce(self, values: Sequence[Any]) -> float:
        """The pass's value from every batch's measurement."""
