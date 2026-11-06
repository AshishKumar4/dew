"""Metamorphic properties of the chunked recurrence (row 3711): transformations of a call that leave its
output and final memory unchanged, judged by check 1 (tests/reference_error.py) of the transformed call
against the untransformed one, the float64 token recurrence the truth: never bitwise, so a property holds of
whatever form the route runs. Hypothesis draws each transformation (the suite's derandomized profile,
docs/design/testing.md): the chunk size the portable math walks, and a sequence split into two calls, the
second entering the first's final memory. Every hook combination the chunked op takes runs the properties on
its portable math, one row and packed documents; the template's split runs interpreted, as an H100 runs it,
on the forms it takes, at fixed splits where Hypothesis's draws do not fit a group's time."""

import dataclasses
import functools

import hypothesis
import jax
import jax.numpy as jnp
import pytest
from hypothesis import strategies as st
from reference_error import assert_as_exact
from test_recurrence_chunk import FORMED, HOOKS, ROWS, chunk_on, draw, formed, token_by_token

from dew.nn.kernels.recurrence_chunk import Chunk

pytestmark = pytest.mark.kernel

SEEDS = 8

PORTABLE = hypothesis.settings(max_examples=2, deadline=None)
"""Each portable property's draws: a draw's three compiles (each arm and the truth at its shape), about 17 s a
case at four draws on armada's CPU (props at f3cf82c98), every form one row and packed."""


FORMS = HOOKS + FORMED
"""Every hook combination the chunked op takes (test_recurrence_chunk's)."""

TEMPLATE_SPLIT = 30
"""Where the template's call is split: inside a chunk and inside a document of either row, the second call
starting off the chunks' grid."""

LENGTHS = st.integers(49, 100)
"""A draw's tokens: past its packed documents' last start (48)."""


def ids(forms) -> list[str]:
    """Each form's test id: its place among `FORMS`."""
    return [f"form{FORMS.index(hooks)}" for hooks in forms]


def _drawn(hooks, packed: bool, length: int, narrow: bool = False):
    """A seed's draw (test_recurrence_chunk's) of `length` tokens; its queries, keys, values and write gate in
    bf16 where `narrow`, a 16-bit call as the template's training takes it."""
    def drawn(seed):
        (operands,) = draw(hooks, packed, seed, length=length)
        if not narrow:
            return (operands,)
        return (dataclasses.replace(operands, **{
            name: getattr(operands, name).astype(jnp.bfloat16) for name in ("query", "key", "value", "beta")
            if getattr(operands, name) is not None}),)

    return drawn


def _timed(operands: Chunk, length: int, span: slice) -> Chunk:
    """`operands`' tokens in `span`: every field laid `[rows, length, ...]`, the rest as they are."""
    return dataclasses.replace(operands, **{
        field.name: value[:, span] for field in dataclasses.fields(operands)
        if (value := getattr(operands, field.name)) is not None and getattr(value, "ndim", 0) >= 2
        and value.shape[:2] == (ROWS, length)})


@pytest.mark.parametrize("packed", [False, True], ids=["one sequence", "packed"])
@pytest.mark.parametrize("hooks", FORMS, ids=ids(FORMS))
@PORTABLE
@hypothesis.given(length=LENGTHS, chunk_size=st.sampled_from([4, 8, 16, 32, 64, 128]))
def test_the_portable_math_is_the_same_recurrence_at_any_chunk_size(hooks, packed, length, chunk_size):
    """The portable math's output and final memory walking chunks of `chunk_size` tokens, on par with the same
    call at the default chunk."""
    walked = functools.partial(chunk_on, "xla", hooks, chunk_size=chunk_size)
    assert_as_exact(jax.jit(walked), jax.jit(functools.partial(chunk_on, "xla", hooks)),
                    _drawn(hooks, packed, length), f"{hooks}, {length} tokens in chunks of {chunk_size}",
                    seeds=SEEDS, truth_fn=functools.partial(token_by_token, hooks))


def split(kernel: str, hooks, at: int, length: int):
    """The call as two: its first `at` tokens, then the rest entering the first's final memory, outputs
    joined."""
    def run(operands: Chunk):
        first, entered = chunk_on(kernel, hooks, _timed(operands, length, slice(None, at)))
        rest = dataclasses.replace(_timed(operands, length, slice(at, None)), state=entered)
        second, final = chunk_on(kernel, hooks, rest)
        return jnp.concatenate([first, second], axis=1), final

    return run


@pytest.mark.parametrize("packed", [False, True], ids=["one sequence", "packed"])
@pytest.mark.parametrize("hooks", FORMS, ids=ids(FORMS))
@PORTABLE
@hypothesis.given(length=LENGTHS, cut=st.floats(0.0, 1.0))
def test_the_portable_math_carried_across_a_split_is_the_whole_recurrence(hooks, packed, length, cut):
    """The call split at a drawn token into two, the second entering the first's final memory: its output
    and final memory on par with the call whole."""
    at = 1 + int(cut * (length - 2))
    whole = functools.partial(chunk_on, "xla", hooks)
    assert_as_exact(jax.jit(split("xla", hooks, at, length)), jax.jit(whole), _drawn(hooks, packed, length),
                    f"{hooks}, {length} tokens split at {at}", seeds=SEEDS,
                    truth_fn=functools.partial(token_by_token, hooks))


def the_template_carried_across_a_split_is_the_whole_recurrence(hooks):
    """The template's 16-bit call over packed documents split at `TEMPLATE_SPLIT` into two, the second
    entering the first's final memory, interpreted: its output and final memory on par with the call whole.
    Its numbers are the run's fast forms', at the default product precision they round at (`formed`; the
    suite's lanes configure 'highest'): an l2 norm hands the rule fp32 queries and keys, whose products the
    template rounds there."""
    length, at = 70, TEMPLATE_SPLIT
    whole = functools.partial(chunk_on, "mosaic_gpu", hooks)
    assert_as_exact(jax.jit(formed("fast", split("mosaic_gpu", hooks, at, length))),
                    jax.jit(formed("fast", whole)), _drawn(hooks, packed=True, length=length, narrow=True),
                    f"the template, {hooks}, packed, split at {at}", seeds=SEEDS,
                    truth_fn=functools.partial(token_by_token, hooks))


# The template's split interpreted, a form a check-1 site of its own over map tasks of two draws: at two
# drawn splits a form, Kimi delta attention's took 371 s, the gated delta net's 145 (props2 at e9406e7d8).
@pytest.mark.check1_shards(seeds=8, per_task=2)
def test_the_gated_delta_nets_split_on_the_template_is_its_whole_call_interpreted():
    """`the_template_carried_across_a_split_is_the_whole_recurrence`: the gated delta net's gates and l2
    norm."""
    the_template_carried_across_a_split_is_the_whole_recurrence(FORMED[0])


@pytest.mark.check1_shards(seeds=8, per_task=2)
def test_the_delta_rules_split_on_the_template_is_its_whole_call_interpreted():
    """`the_template_carried_across_a_split_is_the_whole_recurrence`: the delta rule with a decay a head."""
    the_template_carried_across_a_split_is_the_whole_recurrence(HOOKS[4])


@pytest.mark.check1_shards(seeds=8, per_task=2)
def test_kimi_delta_attentions_split_on_the_template_is_its_whole_call_interpreted():
    """`the_template_carried_across_a_split_is_the_whole_recurrence`: the delta rule with a decay a key
    dimension."""
    the_template_carried_across_a_split_is_the_whole_recurrence(HOOKS[8])


@pytest.mark.check1_shards(seeds=8, per_task=2)
def test_the_rated_scans_split_on_the_template_is_its_whole_call_interpreted():
    """`the_template_carried_across_a_split_is_the_whole_recurrence`: Mamba2's rated scan."""
    the_template_carried_across_a_split_is_the_whole_recurrence(FORMED[4])
