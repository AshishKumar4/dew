"""Token batches as a language-model objective reads them."""

import jax.numpy as jnp
import numpy as np

from dew.batch import TEXT_KEY


def token_batch_of(vocab: int, seq: int, batch: int = 4, seed: int = 0) -> dict:
    """A `[batch, seq + 1]` batch of ids under `vocab`, drawn from `seed`, as the token pipeline packs
    them."""
    ids = np.random.RandomState(seed).randint(0, vocab, (batch, seq + 1))
    return {TEXT_KEY: jnp.asarray(ids, jnp.int32)}
