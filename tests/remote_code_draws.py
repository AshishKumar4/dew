"""The draws a remote-code release is recorded over seeds at (tools/remote_code_reference.py --record) and
its test draws again (`indexed_recording`'s `inputs_of`): NumPy alone, as the release's own environment,
which serves its side of the recording, holds no JAX. Each release's sites are `SITES`; each site's inputs
a seed draws are `draw`'s."""

import numpy as np

ROWS, LENGTH = 2, 16
"""Each draw's rows of token ids and their length."""

DOCUMENTS = (3, 5, 8)
"""A packed row's documents (`packed`): two shorter than LFM3's six cached convolution inputs, so a
convolution reads across no document's start."""

CHUNKS = (5, 3, 1, 1, 1, 1, 1, 1, 1, 1)
"""The calls a cached decode (`decode`) feeds its rows' tokens in: a prefill, a second chunk after it, then
one token a call."""

PADDED = 4
"""The left padding of a padded draw's second row (`biased`), shorter than a convolution's history."""

SITES = {"lfm3-tiny": ("logits", "packed", "decode", "gradients"), "lfm3-biased-tiny": ("biased",),
         "mimo-v2-tiny": ("logits",), "dream-tiny": ("logits",), "llada-tiny": ("logits",)}
"""Each release fixture's sites: its logits over plain rows; LFM3's over a packed row, through the decode
cache in chunks, and its parameter gradients under a drawn cotangent; and, with its convolution's bias
(`conv_bias`), over left-padded rows."""


def draw(site: str, seed: int, vocabulary: int) -> dict[str, np.ndarray]:
    """`site`'s inputs at `seed`: `ROWS` rows of `LENGTH` token ids (one row of them packed for
    `packed`), with a cotangent on the logits of float32 normals for `gradients` and the second row's
    first `PADDED` slots padding for `biased` (`attention_mask` 0 there)."""
    rng = np.random.default_rng([sum(map(ord, site)), seed])
    rows = 1 if site == "packed" else ROWS
    inputs = {"ids": rng.integers(0, vocabulary, (rows, LENGTH)).astype(np.int32)}
    if site == "gradients":
        inputs["cotangent"] = rng.normal(size=(rows, LENGTH, vocabulary)).astype(np.float32)
    if site == "biased":
        mask = np.ones((rows, LENGTH), np.int32)
        mask[1, :PADDED] = 0
        inputs["attention_mask"] = mask
    return inputs


def positions(documents: tuple[int, ...] = DOCUMENTS) -> np.ndarray:
    """A packed row's positions `[1, sum(documents)]`, each document's from 0."""
    return np.concatenate([np.arange(length) for length in documents])[None].astype(np.int32)
