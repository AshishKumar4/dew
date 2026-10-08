"""Typed values that an objective's evaluation produces.

An objective returns these from its scoring and preview hooks. A metric
reads the scoring artifact of the type it expects, and a tracker renders
each preview according to its type. The array fields are pytree leaves, so
they can pass through jit, while the optional captions and decoded text stay
on the host as static metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
from flax import struct
from jax.typing import ArrayLike
from numpy.typing import NDArray

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


@struct.dataclass
class ImageGrid:
    """Images in [-1, 1], `[N, H, W, C]`, with the text each was conditioned on
    where there was any."""
    images: jax.Array
    captions: tuple[str, ...] = struct.field(pytree_node=False, default=())


@struct.dataclass
class VideoGrid:
    """Clips in [-1, 1], `[N, T, H, W, C]`, with the text each was conditioned on
    where there was any."""
    videos: jax.Array
    captions: tuple[str, ...] = struct.field(pytree_node=False, default=())


@struct.dataclass
class TextSamples:
    """Generated token rows, with optional decoded preview text and prompt."""
    tokens: jax.Array | np.ndarray
    prompt: str = struct.field(pytree_node=False, default="")
    texts: tuple[str, ...] = struct.field(pytree_node=False, default=())


@struct.dataclass
class Representations:
    """Encoder outputs `[N, D]` and the labels of the records they came from,
    for a probe to score."""
    features: jax.Array
    labels: jax.Array


@struct.dataclass
class TokenScores:
    """Teacher-forced per-token losses `[N, L]` and the weight of each target.

    A weight is 1 where the target counts and 0 where it is padding or a
    document's first token. A perplexity is exp of the weighted mean loss
    over a whole pass, so a batch with no counted target adds nothing to it.
    """
    losses: jax.Array
    weights: jax.Array
    correct: jax.Array
    """Per-token top-1 correctness, from the same logits that produced the losses."""


@struct.dataclass
class Decisions:
    """A decision model's probabilities `[N, Q, K]` over the options of each
    row's questions, the real options `[N, Q, K]`, each question's right option
    `[N, Q]`, which questions `[N, Q]` ask a score, whose options are ordered
    levels, and which `[N, Q]` have a known answer to score."""
    probabilities: jax.Array
    options: jax.Array
    labels: jax.Array
    ordinal: jax.Array
    scored: jax.Array


type Artifact = DataclassInstance
"""What an objective's scoring and preview hooks return: a dataclass whose
per-row fields lead with the batch's rows, which a validation pass cuts to
the real ones. Dew's own are the classes above. A package's objective may
score into a dataclass of its own, which its metrics read by type
(`Metric.reads`); a metric picks exactly one scoring artifact, and previews
never satisfy metrics. A tracker shows a preview of Dew's types, so a
package shows its own as one of them, a spike raster as an `ImageGrid`."""

type Artifacts = Artifact | tuple[Artifact, ...]
"""One artifact, or several."""


def uint8_pixels(images: ArrayLike) -> NDArray[np.uint8]:
    """Convert [-1, 1] pixels, as `ImageGrid` and `VideoGrid` hold them, to uint8 in [0, 255].

    Each value maps to its nearest level, computed as `(x + 1) * 127.5` in
    float32 and rounded half to even (`np.rint`). The result is clipped,
    because a sample can leave the range. Metrics score these bytes and
    trackers preview them, so both see the same image.
    """
    levels = np.rint((np.asarray(images, np.float32) + 1.0) * 127.5)
    return np.clip(levels, 0, 255).astype(np.uint8)


__all__ = [
    "Artifact",
    "Decisions",
    "ImageGrid",
    "Representations",
    "TextSamples",
    "TokenScores",
    "VideoGrid",
    "uint8_pixels",
]
