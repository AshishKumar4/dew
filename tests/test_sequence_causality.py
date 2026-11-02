"""The static causality walk tier 3 admits a source without its shrunken probe by
(`dew.interop.sequence_causality`, ledger 1595): over small programs whose causality is known, it
admits each one whose every position reads only earlier ones by a structure it holds causal, and refuses
each one that reads a later position or that it cannot show reads none, naming the operation."""

import jax
import jax.numpy as jnp
import pytest

from dew.interop.sequence_causality import CAUSAL_ATTENTION, TOKENWISE, causality_refusal

S, D, VOCAB = 13, 8, 50


def embedded(tokens):
    """The tokens' rows of a table the tokens do not reach, `[1, S, D]`."""
    return jnp.ones((VOCAB, D), jnp.float32)[tokens]


def conv(states, padding):
    """A depthwise convolution of three taps over the sequence, padded as given."""
    taps = jnp.ones((3, 1, D), jnp.float32)
    return jax.lax.conv_general_dilated(states, taps, (1,), (padding,),
                                        dimension_numbers=("NWC", "WIO", "NWC"), feature_group_count=D)


def attention(q, k, v, causal: bool):
    scores = jnp.einsum("bqd,bkd->bqk", q, k)
    if causal:
        scores = jnp.where(jnp.tril(jnp.ones((S, S), bool)), scores, -jnp.inf)
    return jnp.einsum("bqk,bkd->bqd", jax.nn.softmax(scores, axis=-1), v)


def named(function, name):
    function.__name__ = name
    return jax.jit(function)


def scanned(states, reverse: bool):
    def step(carry, x):
        carry = carry * 0.5 + x
        return carry, carry
    return jax.lax.scan(step, jnp.zeros((D,)), states[0], reverse=reverse)[1][None]


ADMITTED = {
    "projection": lambda tokens: embedded(tokens) @ jnp.ones((D, 4)),
    "causal conv": lambda tokens: conv(embedded(tokens), (2, 0)),
    "forward cumsum": lambda tokens: jnp.cumsum(embedded(tokens), axis=1),
    "forward scan": lambda tokens: scanned(embedded(tokens), reverse=False),
    "previous token": lambda tokens: embedded(tokens)[:, jnp.maximum(jnp.arange(S) - 1, 0)],
    "Dew's attention": lambda tokens: named(lambda q, k, v: attention(q, k, v, causal=True),
                                            CAUSAL_ATTENTION)(*(embedded(tokens),) * 3),
    "a token-wise call": lambda tokens: named(lambda x: jnp.sort(x, axis=-1), TOKENWISE)(embedded(tokens)),
    "layers scanned": lambda tokens: jax.lax.scan(lambda h, w: (h @ w, None), embedded(tokens),
                                                  jnp.ones((3, D, D)))[0],
}

REFUSED = {
    "mean over the sequence": lambda tokens: embedded(tokens) - embedded(tokens).mean(axis=1, keepdims=True),
    "backward cumsum": lambda tokens: jax.lax.cumsum(embedded(tokens), axis=1, reverse=True),
    "anticausal conv": lambda tokens: conv(embedded(tokens), (0, 2)),
    "centered conv": lambda tokens: conv(embedded(tokens), (1, 1)),
    "backward scan": lambda tokens: scanned(embedded(tokens), reverse=True),
    "next token": lambda tokens: embedded(tokens)[:, jnp.minimum(jnp.arange(S) + 1, S - 1)],
    "plain attention": lambda tokens: attention(*(embedded(tokens),) * 3, causal=False),
    "masked attention outside Dew's": lambda tokens: attention(*(embedded(tokens),) * 3, causal=True),
    "a reversed sequence": lambda tokens: embedded(tokens)[:, ::-1],
}


def refusal(function):
    return causality_refusal(jax.make_jaxpr(function)(jnp.zeros((1, S), jnp.int32)), 0, S)


@pytest.mark.parametrize("name", sorted(ADMITTED))
def test_a_structurally_causal_program_is_admitted(name):
    assert refusal(ADMITTED[name]) is None


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_a_program_reading_later_positions_or_unproven_is_refused_naming_its_operation(name):
    assert refusal(REFUSED[name]) is not None
