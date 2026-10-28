"""A trainable bigram tool policy over a compact tool vocabulary, and its starting transitions."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.nn.protocols import OutputTable

# Compact tool vocabulary: a call, a tool result, a final answer, and EOS.
PAD, START, CALL_THREE, EOS, NINE, ANSWER_NINE, WRONG, CALL_FOUR, SIXTEEN, ANSWER_SIXTEEN = range(10)
VOCAB = 12
PROMPT, RESPONSE, TURNS, GROUPS = 8, 3, 3, 4


def transition_logits():
    logits = np.full((VOCAB, VOCAB), -1000., np.float32)
    logits[:, EOS] = 1000.
    logits[START, :] = -1000.
    logits[START, CALL_THREE] = .2
    logits[START, CALL_FOUR] = 0.
    for observation, answer in ((NINE, ANSWER_NINE), (SIXTEEN, ANSWER_SIXTEEN)):
        logits[observation, :] = -1000.
        logits[observation, answer] = .25
        logits[observation, WRONG] = 0.
    return logits


class ToolPolicy(nn.Module):
    """A trainable bigram policy, by default over a finite tool-call vocabulary.

    The shared sampler draws calls and final answers from logits; the test
    environment executes square(3) or square(4). No generated code executes.
    `logits` builds the starting `[vocab_size, vocab_size]` table.
    """

    vocab_size: int = VOCAB
    max_seq_len: int = PROMPT + RESPONSE
    logits: Callable[[], np.ndarray] = transition_logits

    def setup(self):
        self.table = self.param("table", lambda key: jnp.asarray(self.logits()))

    def hidden_states(self, tokens, train=False, **packing):
        # A bigram reads each token alone, so a packed row's positions and documents change nothing.
        return jax.nn.one_hot(tokens, self.vocab_size)

    def output_table(self):
        return OutputTable(self.table, vocab_major=False)

    @nn.compact
    def init_cache(self, batch_size):
        self.variable("cache", "seen", lambda: jnp.zeros(batch_size, jnp.int32))

    def __call__(
        self, tokens, train=False, decode=False, attention_mask=None, positions=None, segment_ids=None
    ):
        if decode:
            seen = self.get_variable("cache", "seen")
            valid = jnp.ones_like(tokens, bool) if attention_mask is None else attention_mask
            self.put_variable("cache", "seen", seen + valid.sum(-1))
        return self.hidden_states(tokens) @ self.table
