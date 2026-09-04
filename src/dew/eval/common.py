"""The metric shape the trainer consumes: one artifact type, a per-batch
measurement, the mean over the pass."""

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from jax.typing import ArrayLike

from dew.artifacts import ImageGrid, VideoGrid


def frames(artifact) -> Any:
    """The pixels an image metric scores, `[N, H, W, C]` or `[N, T, H, W, C]`
    in [-1, 1], from either grid."""
    return artifact.videos if isinstance(artifact, VideoGrid) else artifact.images


def paired(artifact, batch, field: str):
    """The sampled pixels and the records they were sampled for, row for row.

    An objective samples a fixed few rows of a batch rather than all of them,
    so a metric that measures a sample against its record takes the leading
    rows of the batch, and says so when the batch is the shorter of the two.
    """
    from dew.inputs import unit_range

    samples = frames(artifact)
    targets = unit_range(batch[field])
    if targets.shape[0] < samples.shape[0]:
        raise ValueError(
            f"the artifact holds {samples.shape[0]} rows and batch[{field!r}] only "
            f"{targets.shape[0]}; a metric that pairs them needs at least as many "
            "records as samples")
    return samples, targets[:samples.shape[0]]


@dataclass(frozen=True, eq=False)
class ImageMetric:
    """A measurement of an `ImageGrid` against the batch it was sampled for,
    averaged over the validation pass."""

    name: str
    measure: Callable[[Any, Any], ArrayLike]
    """A scalar per batch, which a metric computes on device and `__call__`
    brings to a float."""
    reads: type = ImageGrid

    def __call__(self, artifact, batch) -> float:
        return float(np.asarray(self.measure(artifact, batch)))

    def reduce(self, values: Sequence[float]) -> float:
        return float(np.mean(values))
