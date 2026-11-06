"""Decision rows as a run reads them, for the tests of decision objectives and their packing, and drawn
weights for the tests of decision releases against their own code."""

from __future__ import annotations

import numpy as np


def decision_rows(examples, batch: int | None, validation=None, *, objective=None, tokens: int | None = None):
    """`examples` as a run's decision batches of `batch` rows (`dew.decision.decisions`), or of `tokens`
    tokens, read in memory, with `validation`'s examples as its held-out split, compiled with
    `objective`'s stages (`dew.data.compile.compile`)."""
    from dew.data import Memory
    from dew.data.compile import compile
    from dew.decision import decisions

    def chain(rows):
        parsed = Memory([{"example": example} for example in rows]).pipe(decisions)
        return parsed.batch(tokens=tokens) if tokens is not None else parsed.batch(batch)

    held = None if validation is None else chain(validation)
    return compile(chain(examples), validation=held, read="memory", objective=objective)


def drawn_weights(variables, rng: np.random.Generator) -> dict[tuple[str, ...], np.ndarray]:
    """Each leaf of `variables` drawn in float32 by its flattened path: kernels and embeddings at their
    initializer's scale, norms' scales about one and biases off zero."""
    from flax.traverse_util import flatten_dict

    drawn = {}
    for path, value in flatten_dict(variables).items():
        shape = np.shape(value)
        if path[-1] == "scale":
            leaf = rng.normal(1.0, 0.3, shape)
        elif path[-1] == "bias":
            leaf = rng.normal(0.0, 0.3, shape)
        else:
            leaf = rng.normal(0.0, 1.0, shape) / np.sqrt(np.prod(shape[:-1]))
        drawn[path] = leaf.astype(np.float32)
    return drawn
