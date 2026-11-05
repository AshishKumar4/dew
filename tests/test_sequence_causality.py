"""The static causality walk tier 3 proves a source causal by at each length it runs, without its shrunken
probe (`dew.interop.sequence_causality`, ledger 1595): over small programs whose causality is known, traced
at 11 and 13 tokens, it admits each one whose every output position reads no later input through the
operations it composes, and refuses each one that reads a later position or that it cannot show reads
none; a program that computes otherwise past a length is proven below it and refused above. Dew's own
attention and token-wise programs are read by their rules where Dew's code bound them (`attested`); a
program merely named like one is walked inside."""

import jax
import jax.numpy as jnp
import pytest

from dew.interop.sequence_causality import CAUSAL_ATTENTION, TOKENWISE, attested, causality_refusal

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
    """`function` jitted under the name of a call Dew's code binds, which identifies nothing."""
    function.__name__ = name
    return jax.jit(function)


def scanned(states, initial, reverse: bool):
    """A recurrence over the positions from `initial`, each step's state its output."""
    def step(carry, x):
        carry = carry * 0.5 + x
        return carry, carry
    return jax.lax.scan(step, initial, states[0], reverse=reverse)[1][None]


def spread_primitive():
    """A primitive of its own named `exp`, an entrywise operation's name, that gives each entry the sum of
    its operand: no operation one entry at a time, whatever its name."""
    from jax.extend.core import Primitive

    spread = Primitive("exp")
    spread.def_abstract_eval(lambda value: value)
    return spread


SPREAD = spread_primitive()


def shifted_left(states):
    """Each position takes the next one's state (slice and concatenate, no contraction)."""
    return jnp.concatenate([states[:, 1:], jnp.zeros_like(states[:, :1])], axis=1)


def shifted_right(states):
    """Each position takes the previous one's state (a pad before the first, the last sliced off)."""
    return jnp.pad(states, ((0, 0), (1, 0), (0, 0)))[:, :-1]


ADMITTED = {
    "projection": lambda tokens: embedded(tokens) @ jnp.ones((D, 4)),
    "causal conv": lambda tokens: conv(embedded(tokens), (2, 0)),
    "forward cumsum": lambda tokens: jnp.cumsum(embedded(tokens), axis=1),
    "forward scan": lambda tokens: scanned(embedded(tokens), jnp.zeros((D,)), reverse=False),
    "previous token": lambda tokens: embedded(tokens)[:, jnp.maximum(jnp.arange(tokens.shape[1]) - 1, 0)],
    "a right shift": lambda tokens: shifted_right(embedded(tokens)),
    "heads split and merged": lambda tokens: embedded(tokens).reshape(1, tokens.shape[1], 2, 4).transpose(
        0, 2, 1, 3).transpose(0, 2, 1, 3).reshape(1, tokens.shape[1], D),
    "Dew's attention": lambda tokens: attested(lambda q, k, v: attention(q, k, v, causal=True),
                                               CAUSAL_ATTENTION)(*(embedded(tokens),) * 3),
    "a token-wise call": lambda tokens: attested(lambda x: jnp.sort(x, axis=-1), TOKENWISE)(
        embedded(tokens)[0])[None],
    "layers scanned": lambda tokens: jax.lax.scan(lambda h, w: (jnp.cumsum(h @ w, axis=1), None),
                                                  embedded(tokens), jnp.ones((3, D, D)))[0],
    "a token's own random bits": lambda tokens: jax.vmap(jax.vmap(lambda token: jax.random.bits(
        jax.random.fold_in(jax.random.key(0), token), ())))(tokens)[..., None],
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
    "a left shift named as Dew's attention": lambda tokens: named(lambda q, k, v: shifted_left(v),
                                                                  CAUSAL_ATTENTION)(*(embedded(tokens),) * 3),
    "a left shift named token-wise": lambda tokens: named(lambda x: shifted_left(x[None])[0], TOKENWISE)(
        embedded(tokens)[0])[None],
    "a reversed sequence": lambda tokens: embedded(tokens)[:, ::-1],
    "a primitive taking an entrywise name": lambda tokens: SPREAD.bind(embedded(tokens)),
    "an operation with no rule": lambda tokens: jax.lax.linalg.tridiagonal_solve(
        jnp.full((tokens.shape[1],), 0.5), jnp.full((tokens.shape[1],), 2.0),
        jnp.full((tokens.shape[1],), 0.5), embedded(tokens)[0, :, :1])[None],
    "positions moved to the features' axis": lambda tokens: jnp.broadcast_to(
        embedded(tokens)[:, :, :1], (1, tokens.shape[1], tokens.shape[1])).transpose(0, 2, 1),
    "a reshape that transposes": lambda tokens: jax.lax.reshape(
        jnp.broadcast_to(embedded(tokens)[:, :, :1], (1, tokens.shape[1], tokens.shape[1])),
        (1, tokens.shape[1], tokens.shape[1]), dimensions=(0, 2, 1)),
    "a partial update over a left shift": lambda tokens: jax.lax.dynamic_update_slice(
        shifted_left(embedded(tokens))[:, :, :2], jnp.zeros((1, tokens.shape[1], 1)), (0, 0, 0)),
    "a loop of no passes from a left shift": lambda tokens: jax.lax.while_loop(
        lambda carry: carry[1] < 0, lambda carry: (jnp.zeros_like(carry[0]), carry[1] + 1),
        (shifted_left(embedded(tokens)), 0))[0],
    # Each position's state beside the next one's as two batch groups, which the convolution moves into
    # its two features: the second reads a later position.
    "a convolution grouping the positions by batch": lambda tokens: jax.lax.conv_general_dilated(
        jnp.concatenate([embedded(tokens), shifted_left(embedded(tokens))], axis=1)[0, :, :1][:, :, None],
        jnp.ones((1, 1, 2)), (1,), ((0, 0),), dimension_numbers=("NWC", "WIO", "NWC"),
        batch_group_count=2).reshape(1, tokens.shape[1], 2),
    "layers scanned with a mean": lambda tokens: jax.lax.scan(
        lambda h, w: (h @ w - (h @ w).mean(axis=1, keepdims=True), None), embedded(tokens),
        jnp.ones((3, D, D)))[0],
}


def refusal(function, lengths=LENGTHS):
    """The walk's refusal of `function` at the first of `lengths` it refuses, None where it proves each."""
    for length in lengths:
        program = jax.make_jaxpr(function)(jnp.zeros((1, length), jnp.int32))
        found = causality_refusal(program, 0, length, (0,))
        if found is not None:
            return found
    return None


@pytest.mark.parametrize("name", sorted(ADMITTED))
def test_a_program_whose_positions_read_no_later_one_is_admitted(name):
    assert refusal(ADMITTED[name]) is None


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_a_program_reading_a_later_position_or_unproven_is_refused(name):
    assert refusal(REFUSED[name]) is not None


def test_a_program_that_reads_later_positions_past_a_length_is_proven_below_it_and_refused_above():
    """A forward that shifts each position's state left past 16 tokens: its program at 13 tokens is
    causal and at 17 it is not, so each length is proven on its own program, never one for all
    (`torchax_fallback.TorchGraph.certify`)."""
    def thresholded(tokens):
        states = embedded(tokens)
        return shifted_left(states) if tokens.shape[1] > 16 else states

    assert refusal(thresholded, (13,)) is None
    assert refusal(thresholded, (17,)) is not None


def test_a_length_past_a_machine_word_is_walked_whole():
    """Positions past 64 are walked as at any length (each position's set an integer of any width): a
    projection of 70 tokens is causal, and its left shift reads a later one."""
    assert refusal(ADMITTED["projection"], (70,)) is None
    assert refusal(REFUSED["a left shift by slice and concatenate"], (70,)) is not None


def test_a_sequence_axis_kept_through_reshapes_is_walked_at_any_batch():
    """512 rows of 4096 tokens, more entries a call than the walk sizes a reshaped axis anew (`CONCRETE`):
    the heads split and merged keep the sequence axis in place, whose reads the walk carries as they are,
    so the program is causal, and its left shift reads a later position (ledger 3677)."""
    def heads(tokens):
        split = embedded(tokens).reshape(*tokens.shape, 2, 4).transpose(0, 2, 1, 3)
        return split.transpose(0, 2, 1, 3).reshape(*tokens.shape, D)

    tokens = jax.ShapeDtypeStruct((512, 4096), jnp.int32)
    for function, causal in ((heads, True), (lambda tokens: shifted_left(heads(tokens)), False)):
        program = jax.make_jaxpr(function)(tokens)
        assert (causality_refusal(program, 0, 4096, (0,)) is None) == causal


def test_a_convolution_padding_an_empty_sequence_reads_nothing():
    """A convolution over a sequence with no positions whose padding still gives outputs: each output
    reads no position, so the program is causal, where the walk's convolution rule read an empty
    operand's rows."""
    def padded(tokens):
        empty = embedded(tokens)[:, :0, :1]
        out = jax.lax.conv_general_dilated(empty, jnp.ones((2, 1, 1)), (1,), ((1, 1),),
                                           dimension_numbers=("NWC", "WIO", "NWC"))
        return embedded(tokens) + jnp.sum(out)

    program = jax.make_jaxpr(padded)(jax.ShapeDtypeStruct((1, 6), jnp.int32))
    assert causality_refusal(program, 0, 6, (0,)) is None
