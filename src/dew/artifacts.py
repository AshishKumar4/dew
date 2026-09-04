"""What an objective's evaluation produces, as typed values.

An objective's `evaluate` returns one of these; a `Tracker` renders it by
dispatching on the type; a metric is a function of it and the batch. The
types carry arrays and nothing else, so they cross `jit` and say nothing
about where they will be drawn.
"""

from __future__ import annotations

from flax import struct
import jax


@struct.dataclass
class ImageGrid:
    """Images in [-1, 1], `[N, H, W, C]`, with the text each was conditioned on
    where there was any."""
    images: jax.Array
    captions: tuple[str, ...] = struct.field(pytree_node=False, default=())


@struct.dataclass
class VideoGrid:
    """Clips in [-1, 1], `[N, T, H, W, C]`."""
    videos: jax.Array
    captions: tuple[str, ...] = struct.field(pytree_node=False, default=())


@struct.dataclass
class TextSamples:
    """Decoded continuations of a prompt, and the token ids they came from."""
    tokens: jax.Array
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
    """Teacher-forced per-token losses `[N, L]` and the weight of each target,
    1 where it counts and 0 where it is padding or a document's first token.
    A perplexity is exp of the weighted mean over a whole pass, so a batch
    with no counted target weighs nothing."""
    losses: jax.Array
    weights: jax.Array


# `evaluate` returns one artifact or a tuple of them; a tracker renders each
# by type and a metric picks the type it reads. An LM returns TokenScores
# every pass and TextSamples when samples are configured.
Artifact = ImageGrid | VideoGrid | TextSamples | Representations | TokenScores
Artifacts = Artifact | tuple[Artifact, ...]
