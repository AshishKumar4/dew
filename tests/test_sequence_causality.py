"""The static causality walk tier 3 admits a source without its shrunken probe by
(`dew.interop.sequence_causality`, ledger 1595): over small programs whose causality is known, traced at
two coprime lengths, it admits each one whose every output position reads no later input through the
operations it composes, and refuses each one that reads a later position, that it cannot show reads
none, or whose program differs between the lengths."""

import jax
import jax.numpy as jnp
import pytest

from dew.interop.sequence_causality import CAUSAL_ATTENTION, TOKENWISE, causality_refusal

LENGTHS, D, VOCAB = (11, 13), 8, 50


def embedded(tokens):
    """The tokens' rows of a table the tokens do not reach, `[1, S, D]`."""
    return jnp.ones((VOCAB, D), jnp.float32)[tokens]


def conv(states, padding):
    """A depthwise convolution of three taps over the sequence, padded as given."""
    taps = jnp.ones((3, 1, D), jnp.float32)
    return jax.lax.conv_general_dilated(states, taps, (1,), (padding,),
                                        dimension_numbers=("NWC", "WIO", "NWC"), feature_group_count=D)


def attention(q, k, v, causal: bool):
    length = q.shape[1]
    scores = jnp.einsum("bqd,bkd->bqk", q, k)
    if causal:
        scores = jnp.where(jnp.tril(jnp.ones((length, length), bool)), scores, -jnp.inf)
    return jnp.einsum("bqk,bkd->bqd", jax.nn.softmax(scores, axis=-1), v)


def named(function, name):
    function.__name__ = name
    return jax.jit(function)


def scanned(states, initial, reverse: bool):
    """A recurrence over the positions from `initial`, each step's state its output."""
    def step(carry, x):
        carry = carry * 0.5 + x
        return carry, carry
    return jax.lax.scan(step, initial, states[0], reverse=reverse)[1][None]


def shifted_left(states):
    """Each position takes the next one's state (slice and concatenate, no contraction)."""
    return jnp.concatenate([states[:, 1:], jnp.zeros_like(states[:, :1])], axis=1)


def shifted_right(states):
    """Each position takes the previous one's state."""
    return jnp.concatenate([jnp.zeros_like(states[:, :1]), states[:, :-1]], axis=1)


ADMITTED = {
    "projection": lambda tokens: embedded(tokens) @ jnp.ones((D, 4)),
    "causal conv": lambda tokens: conv(embedded(tokens), (2, 0)),
    "forward cumsum": lambda tokens: jnp.cumsum(embedded(tokens), axis=1),
    "forward scan": lambda tokens: scanned(embedded(tokens), jnp.zeros((D,)), reverse=False),
    "previous token": lambda tokens: embedded(tokens)[:, jnp.maximum(jnp.arange(tokens.shape[1]) - 1, 0)],
    "a right shift": lambda tokens: shifted_right(embedded(tokens)),
    "heads split and merged": lambda tokens: embedded(tokens).reshape(1, tokens.shape[1], 2, 4).transpose(
        0, 2, 1, 3).transpose(0, 2, 1, 3).reshape(1, tokens.shape[1], D),
    "Dew's attention": lambda tokens: named(lambda q, k, v: attention(q, k, v, causal=True),
                                            CAUSAL_ATTENTION)(*(embedded(tokens),) * 3),
    "a token-wise call": lambda tokens: named(lambda x: jnp.sort(x, axis=-1), TOKENWISE)(embedded(tokens)[0]),
    "layers scanned": lambda tokens: jax.lax.scan(lambda h, w: (jnp.cumsum(h @ w, axis=1), None),
                                                  embedded(tokens), jnp.ones((3, D, D)))[0],
}

REFUSED = {
    "mean over the sequence": lambda tokens: embedded(tokens) - embedded(tokens).mean(axis=1, keepdims=True),
    "backward cumsum": lambda tokens: jax.lax.cumsum(embedded(tokens), axis=1, reverse=True),
    "anticausal conv": lambda tokens: conv(embedded(tokens), (0, 2)),
    "centered conv": lambda tokens: conv(embedded(tokens), (1, 1)),
    "a left shift by slice and concatenate": lambda tokens: shifted_left(embedded(tokens)),
    "a reversed scan": lambda tokens: scanned(embedded(tokens), jnp.zeros((D,)), reverse=True),
    "a scan seeded by the future": lambda tokens: scanned(embedded(tokens), embedded(tokens)[0, -1],
                                                          reverse=False),
    "next token": lambda tokens: embedded(tokens)[:, jnp.minimum(jnp.arange(tokens.shape[1]) + 1,
                                                                tokens.shape[1] - 1)],
    "plain attention": lambda tokens: attention(*(embedded(tokens),) * 3, causal=False),
    "masked attention outside Dew's": lambda tokens: attention(*(embedded(tokens),) * 3, causal=True),
    "a reversed sequence": lambda tokens: embedded(tokens)[:, ::-1],
    "a branch on the sequence's length": lambda tokens: (
        jnp.cumsum(embedded(tokens), axis=1) if tokens.shape[1] == 13 else embedded(tokens) * 2),
    "layers scanned with a mean": lambda tokens: jax.lax.scan(
        lambda h, w: (h @ w - (h @ w).mean(axis=1, keepdims=True), None), embedded(tokens),
        jnp.ones((3, D, D)))[0],
}


def refusal(function):
    programs = {length: (jax.make_jaxpr(function)(jnp.zeros((1, length), jnp.int32)), 0)
                for length in LENGTHS}
    return causality_refusal(programs)


@pytest.mark.parametrize("name", sorted(ADMITTED))
def test_a_program_whose_positions_read_no_later_one_is_admitted(name):
    assert refusal(ADMITTED[name]) is None


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_a_program_reading_a_later_position_or_unproven_is_refused(name):
    assert refusal(REFUSED[name]) is not None
